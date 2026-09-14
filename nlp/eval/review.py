"""A4a (issue #74): G1 review sampling from persisted themes, and the scorecard.

Section 8 of ``docs/PHASE_0_SPEC.md`` reads gate G1 -- theme-assignment
human agreement, >= 75% over >= 80 sampled assignments -- off review sheets
that K3 reviewers complete.  This module produces those sheets and reads
them back.  It covers G1 only; G2 waits on A2/A3 persisting the summaries a
reader actually sees, and nothing here calls a model.

**The population is what was persisted.**  A review row is one story's
placement inside one stored theme set -- a theme, "Other coverage", or an
exclusion -- read from the Phase 0 database in one snapshot
(:meth:`phase0.repository.Phase0Reader.theme_population`) exactly as the
``themes`` stage left it over the *current* authoritative story
generation.  M5 is never re-run at sampling time.  A partition whose theme
set is missing, stale, or inconsistent with its stories is recorded as
skipped with its reason, never quietly left out and never reviewed as
healthy output.

**Origin is a fact about persistence, not a claim anyone makes.**  A row
that came out of SQLite is not thereby real ingested evidence:
``Phase0Admin.insert_raw_items`` writes rows indistinguishable from fetched
ones, and ``raw_items`` carries no link to the run that fetched it.  So
every sample carries an :class:`OriginStatus` derived from *how it was
read*: ``SYNTHETIC`` for the committed fixture, ``UNVERIFIED`` for any
database.  ``VERIFIED_LIVE`` exists in the vocabulary and has no producer:
persistence would have to link each raw item structurally to the fetch
run that wrote it, and a reviewed code change would then classify it.  An
operator's attestation is recorded on the manifest as audit metadata and
changes nothing.

**Authority.**  A gate is scored only from artifacts that can be held to
something: the sample manifest (whose captured snapshot is
integrity-bound and re-verified on read), the completed reviewer sheets
(held to that snapshot column by column), the adjudication sheet, the
in-code protocol registry, and the locked release requirements.  Nothing
serialized is believed about ratification, eligibility, reviewer count,
adjudication, population compatibility, or thresholds.

**Four facts, four fields.**  A scorecard never carries a ``meets_gate``
boolean.  ``threshold_met`` is arithmetic; ``review_complete`` is whether
the review finished; ``gate_eligible`` is whether these numbers may settle
anything; ``gate_result`` is derived from the three in that order.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import math
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from nlp.eval.trust import (
    DatasetKind,
    LabelingStatus,
    MetricsPurpose,
    TrustContract,
)
from phase0.redaction import contains_credential, redact_text
from phase0.rss import STAGE_FETCH as RSS_FETCH_STAGE
from phase0.rss import STAGE_INGEST as RSS_INGEST_STAGE
from phase0.stories import STAGE as STORIES_STAGE
from phase0.themes import STAGE as THEMES_STAGE
from phase0.yahoo import STAGE as YAHOO_FETCH_STAGE

#: Bumped when the manifest's shape changes, so a committed sample can
#: always be read by the code that claims to understand it.
MANIFEST_SCHEMA = "a4a-review-sample/3"
SHEET_KIND = "theme_assignment"
GATE = "G1"

#: Section 8's release requirements for G1.  Fixed by the spec, fixed
#: here, and never read from an artifact or a flag: a development override
#: exists, and it forces a development evaluation that cannot be eligible.
RELEASE_G1_THRESHOLD = 0.75
RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS = 80
#: Issue #74's round size.  Rounds are combined toward the 80 by explicit
#: prior-round linkage, never by adding up whatever was scored together.
DEFAULT_ROUND_SIZE = 40

SAMPLE_METHOD = "seeded_uniform_without_replacement"

ASSIGNMENT_THEME = "theme"
ASSIGNMENT_OTHER = "other_coverage"
ASSIGNMENT_EXCLUDED = "excluded"
ASSIGNMENT_TYPES = (ASSIGNMENT_THEME, ASSIGNMENT_OTHER, ASSIGNMENT_EXCLUDED)

#: ``run_log`` stages that mean evidence was fetched from a provider.
INGESTION_STAGES = frozenset({YAHOO_FETCH_STAGE, RSS_FETCH_STAGE, RSS_INGEST_STAGE})
DOWNSTREAM_STAGES = (STORIES_STAGE, THEMES_STAGE)
HEALTHY_STORY_STAGE = "m3.semantic"
DEGRADED_STORY_STAGE = "m2.exact"

SOURCE_PERSISTED = "persisted"
SOURCE_FIXTURE = "fixture"

#: Whether a theme set is known to have been built over the story
#: generation it now sits on.  Nothing persisted records the generation a
#: set was built from -- ``reconcile_themes`` checks the signature inside
#: its transaction and drops it -- so a database set is ``unverified``: the
#: *current* signature is reported as current and proves nothing about the
#: build.  A fixture set is clustered in-process, so build and current are
#: one act.  ``verified`` has no producer until persistence keeps the
#: build-time signature.
GENERATION_BINDING_VERIFIED = "verified"
GENERATION_BINDING_UNVERIFIED = "unverified"
GENERATION_BINDING_IN_PROCESS = "in_process"

LIST_SEPARATOR = " | "


class ReviewSamplingError(ValueError):
    """A sample, manifest, or completed sheet is not usable."""


class GateResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


class OriginStatus(str, Enum):
    """Where a sample's rows came from, as far as persistence can prove."""

    #: Each row is structurally linked to the provider fetch that wrote
    #: it.  Nothing produces this today: ``raw_items`` has no such link.
    VERIFIED_LIVE = "verified_live"
    #: Authored for development; the source declares it so.
    SYNTHETIC = "synthetic"
    #: Persisted rows whose ingestion origin cannot be established.
    UNVERIFIED = "unverified"


class AdjudicationState(str, Enum):
    """What happened between reviewers, stated rather than inferred."""

    #: One reviewer: there was nobody to disagree with.
    NOT_APPLICABLE = "not_applicable"
    #: Two reviewers, no disagreement on any marked row.
    UNANIMOUS = "unanimous"
    #: Every disagreement carries a final verdict from an adjudicator.
    RESOLVED = "resolved"
    #: At least one disagreement has no final verdict.
    OPEN = "open"


# -- The labeling protocol registry -----------------------------------------


@dataclass(frozen=True)
class Protocol:
    """A K3 review protocol: its vocabulary and what it counts as adjudicated.

    ``adjudicated_states`` is empty for the provisional protocol on
    purpose.  Whether a unanimous two-reviewer round counts as adjudicated,
    or only one with disagreements resolved by a third party, is K3's
    call; until K3 makes it nothing is adjudicated in the gate's sense.
    """

    id: str
    positive_verdict: str
    negative_verdict: str
    adjudicated_states: frozenset[AdjudicationState]

    @property
    def vocabulary(self) -> frozenset[str]:
        return frozenset({self.positive_verdict, self.negative_verdict})


UNRATIFIED_PROTOCOL = "unratified"
PROVISIONAL_PROTOCOL = Protocol(
    id=UNRATIFIED_PROTOCOL,
    positive_verdict="correct",
    negative_verdict="incorrect",
    adjudicated_states=frozenset(),
)
#: Ratification lives here and only here.  K3 adds its protocol in a
#: reviewed change; an artifact naming an id that is not in this mapping
#: is scored as unratified whatever else it says about itself.
RATIFIED_PROTOCOLS: Mapping[str, Protocol] = {}


def resolve_protocol(protocol_id: Any) -> tuple[Protocol, bool]:
    """The protocol to score under, and whether it is ratified -- from code."""

    identifier = str(protocol_id or "").strip()
    if identifier in RATIFIED_PROTOCOLS:
        return RATIFIED_PROTOCOLS[identifier], True
    return PROVISIONAL_PROTOCOL, False


def require_known_protocol(protocol_id: str) -> Protocol:
    """At sampling time only a registered id or ``unratified`` is accepted."""

    identifier = str(protocol_id or "").strip()
    if identifier == UNRATIFIED_PROTOCOL:
        return PROVISIONAL_PROTOCOL
    if identifier in RATIFIED_PROTOCOLS:
        return RATIFIED_PROTOCOLS[identifier]
    raise ReviewSamplingError(
        f"unknown labeling protocol {identifier!r}; use "
        f"{UNRATIFIED_PROTOCOL!r} or one of {sorted(RATIFIED_PROTOCOLS)}"
    )


# -- Rows ------------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentRow:
    """One story's placement in one persisted theme set, as a reviewer sees it.

    The eight fields after ``row_id`` are the durable identity.  The rest,
    up to the reviewer fields, is the context a human used to judge the
    placement -- and every one of those columns is captured in the
    manifest snapshot and held to it when the sheet comes back.
    ``sibling_story_titles`` is the other members of the same theme; "does
    this story belong here" cannot be answered from the story alone.
    """

    row_id: str
    theme_set_id: str
    pipeline_version: str
    ticker: str
    trading_day: str
    story_id: str
    assignment_type: str
    theme_key: str
    placement_reason: str
    story_title: str
    story_description: str
    story_canonical_url: str
    story_outlets: str
    story_stage: str
    theme_label: str
    theme_label_source: str
    theme_story_count: str
    sibling_story_titles: str
    reviewer_id: str = ""
    reviewed_at: str = ""
    reviewer_verdict: str = ""
    reviewer_notes: str = ""

    def identity(self) -> dict[str, str]:
        return row_identity(**{key: getattr(self, key) for key in IDENTITY_FIELDS})

    def snapshot(self) -> dict[str, str]:
        """Everything a reviewer saw, reviewer columns excluded."""

        return {key: getattr(self, key) for key in SNAPSHOT_FIELDS}


#: Two columns bind a sheet to the exact manifest and snapshot it was cut
#: from.  They are written on every row of a blank sheet, are not part of
#: the row's own snapshot (they describe the artifact, not the placement),
#: and a completed sheet must carry the scored manifest's values on every
#: row -- so a review completed against snapshot A cannot be handed in
#: against a re-authored snapshot B, however alike their rows look.
BINDING_FIELDS = ("manifest_id", "snapshot_sha256")
REVIEWER_FIELDS = ("reviewer_id", "reviewed_at", "reviewer_verdict", "reviewer_notes")
SNAPSHOT_FIELDS = tuple(f.name for f in dataclasses.fields(AssignmentRow))
SNAPSHOT_FIELDS = tuple(f for f in SNAPSHOT_FIELDS if f not in REVIEWER_FIELDS)
ASSIGNMENT_FIELDNAMES = SNAPSHOT_FIELDS + BINDING_FIELDS + REVIEWER_FIELDS
IDENTITY_FIELDS = (
    "theme_set_id",
    "pipeline_version",
    "ticker",
    "trading_day",
    "story_id",
    "assignment_type",
    "theme_key",
    "placement_reason",
)
CONTEXT_FIELDS = tuple(
    f for f in SNAPSHOT_FIELDS if f not in IDENTITY_FIELDS and f != "row_id"
)
ADJUDICATION_FIELDNAMES = (
    "row_id",
    "final_verdict",
    "adjudicator_id",
    "adjudicated_at",
    "adjudication_notes",
)


def row_identity(**fields: str) -> dict[str, str]:
    """The durable identity of one review row, in a fixed key order."""

    missing = sorted(set(IDENTITY_FIELDS) - set(fields))
    if missing:
        raise ReviewSamplingError(f"row identity is missing {missing}")
    return {"gate": GATE, **{key: str(fields[key]) for key in IDENTITY_FIELDS}}


def row_id_for(identity: Mapping[str, str]) -> str:
    return f"g1-{sha256_of(identity)}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_of(value: Any) -> str:
    """Full SHA-256 over the canonical JSON encoding of ``value``."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value: Any) -> str:
    """One string of review context, credentials removed."""

    return redact_text(str(value or ""))


def _join(values: Iterable[str]) -> str:
    return LIST_SEPARATOR.join(v for v in values if v)


#: Provenance fields that are identifiers: they are compared, matched, and
#: digested, so rewriting one would corrupt the identity it names.  A
#: credential-looking value in one is refused rather than redacted, and the
#: refusal names the field, never the value.
PROVENANCE_IDENTIFIER_FIELDS = (
    "pipeline_version",
    "ticker",
    "method",
    "config_fingerprint",
    "algorithm_version",
    "model_name",
    "model_revision",
    "theme_key",
    "label_source",
)


def _credential_bearing_fields(values: Mapping[str, Any]) -> list[str]:
    """Which identifier fields carry credential-like text, by name only."""

    return sorted(
        field
        for field, value in values.items()
        if value is not None and contains_credential(str(value))
    )


def _require_clean_identifier(value: Any, field: str) -> str:
    text = str(value or "")
    if contains_credential(text):
        raise ReviewSamplingError(
            f"{field} carries credential-like text and cannot be used as an "
            "identifier; the value is not shown"
        )
    if text.strip() == REJECTED_IDENTIFIER:
        raise ReviewSamplingError(
            f"{field} {REJECTED_IDENTIFIER!r} is reserved: it is the sentinel A4a "
            "records for a refused identifier and never names a real one"
        )
    return text


# -- Population --------------------------------------------------------------


@dataclass(frozen=True)
class SkippedPartition:
    """A selected partition that contributed no rows, and why."""

    ticker: str
    trading_day: str
    pipeline_version: str
    reason: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


SKIP_M2_ONLY = "m2_only_degraded"
SKIP_THEMES_NOT_GENERATED = "themes_not_generated"
SKIP_MIXED_STAGES = "mixed_story_stages"
SKIP_NO_STORY_OUTPUT = "no_story_output"
SKIP_STALE_SET_NO_STORIES = "theme_set_without_live_stories"
SKIP_STALE_SET_DEGRADED = "theme_set_over_degraded_stories"
SKIP_SET_INCONSISTENT = "theme_set_inconsistent_with_stories"
SKIP_PROVENANCE_CREDENTIAL = "provenance_identifier_credential"
SKIP_RESERVED_IDENTIFIER = "reserved_provenance_identifier"
#: Stands in for an identifier that was refused at discovery.  It names no
#: partition and matches nothing: the unsafe value is not retained at all.
#: The literal is reserved on both boundaries -- a persisted version equal
#: to it is skipped and a supplied one is refused -- so no real identity can
#: ever equal the sentinel, and no filter can select one.
REJECTED_IDENTIFIER = "<rejected>"


@dataclass(frozen=True)
class ThemeSetProvenance:
    """What produced one partition's theme set, and over which stories."""

    theme_set_id: str
    ticker: str
    trading_day: str
    pipeline_version: str
    method: str
    config_fingerprint: str
    algorithm_version: str
    model_name: str | None
    model_revision: str | None
    embedding_dimension: int | None
    updated_at: str | None
    #: The story generation the partition holds *now*.  Reported as current
    #: because that is all it is: nothing persisted says which generation
    #: the set was built over.
    story_generation_signature_current: str
    #: The generation the set was built over.  ``None`` for a database set,
    #: because persistence does not keep it; the fixture's own digest for a
    #: set clustered in-process.
    theme_build_story_generation_signature: str | None
    generation_binding: str
    theme_count: int
    other_coverage_count: int
    excluded_count: int

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def selection_digest(
    *,
    trading_days: Sequence[str],
    tickers: Sequence[str],
    pipeline_versions: Sequence[str],
    theme_set_ids: Sequence[str],
    source_mode: Any,
    theme_sets: Sequence[Mapping[str, Any]],
    skipped_partitions: Sequence[Mapping[str, Any]],
) -> str:
    """The one digest of what was selected, computed the same way everywhere.

    Every field that defines the selection is in it -- days, tickers,
    versions, theme-set ids, source mode, the theme sets' provenance and
    the partitions that were skipped -- so an edit to any of them without
    recomputing the digest is a mismatch, and the digest is never accepted
    as an opaque string.
    """

    return sha256_of(
        {
            "trading_days": list(trading_days),
            "tickers": list(tickers),
            "pipeline_versions": list(pipeline_versions),
            "theme_set_ids": list(theme_set_ids),
            "source_mode": source_mode,
            "theme_sets": list(theme_sets),
            "skipped_partitions": list(skipped_partitions),
        }
    )


@dataclass(frozen=True)
class Population:
    """Every reviewable placement in the selected partitions, exactly once."""

    rows: tuple[AssignmentRow, ...]
    theme_sets: tuple[ThemeSetProvenance, ...]
    skipped: tuple[SkippedPartition, ...]
    trading_days: tuple[str, ...]
    tickers: tuple[str, ...]
    pipeline_versions: tuple[str, ...]
    source: Mapping[str, Any]
    #: Ledger facts about the population's evidence; ``None`` for fixtures.
    indicators: Mapping[str, Any] | None = None

    @property
    def digest(self) -> str:
        """Identifies the population: every row's full context and provenance.

        Two rounds may be combined only when they share this.  A story
        re-placed, re-titled, or re-clustered between generations changes
        it, so such rounds are incompatible rather than "more unique rows".
        """

        return sha256_of(
            {
                "rows": [row.snapshot() for row in self.rows],
                "theme_sets": [s.as_dict() for s in self.theme_sets],
            }
        )

    @property
    def selection_digest(self) -> str:
        return selection_digest(
            trading_days=self.trading_days,
            tickers=self.tickers,
            pipeline_versions=self.pipeline_versions,
            theme_set_ids=[s.theme_set_id for s in self.theme_sets],
            source_mode=self.source.get("mode"),
            theme_sets=[s.as_dict() for s in self.theme_sets],
            skipped_partitions=[s.as_dict() for s in self.skipped],
        )

    @property
    def by_assignment_type(self) -> dict[str, int]:
        counts = {kind: 0 for kind in ASSIGNMENT_TYPES}
        for row in self.rows:
            counts[row.assignment_type] += 1
        return counts


def classify_generation_binding(population: Any) -> tuple[str, str | None]:
    """Whether a persisted theme set is known to be built over its stories.

    Returns ``(binding, build_signature)``.  Nothing persisted records the
    story generation a set was built over -- ``reconcile_themes`` verifies
    the signature inside its transaction and does not keep it -- so this
    is ``(unverified, None)`` for every database set.  When persistence
    keeps the build-time signature, this is where it is read and compared
    with ``population.stories.signature``, in a reviewed change.
    """

    return GENERATION_BINDING_UNVERIFIED, None


def _partition_rows(population: Any) -> tuple[list[AssignmentRow], ThemeSetProvenance]:
    """Project one :class:`ThemePopulation` onto review rows.

    The caller has already established the theme set is a valid view of
    the current story generation; this turns it into rows and redacts
    every piece of human-facing context on the way.
    """

    theme_set = population.theme_set
    ticker = population.ticker
    day = population.trading_day
    version = population.pipeline_version
    set_id = str(theme_set.theme_set_id)
    stories = {story.story_id: story for story in population.stories.stories}

    def describe(story_id: int) -> tuple[str, str, str]:
        story = stories[story_id]
        members = sorted(
            story.members,
            key=lambda m: (m.raw_item_id != story.canonical_item_id, m.position),
        )
        description = next((m.description for m in members if m.description), "")
        outlets: list[str] = []
        for value in [story.outlet] + [m.outlet for m in members]:
            text = _clean(value)
            if text and text not in outlets:
                outlets.append(text)
        return _clean(description), _join(outlets), _clean(story.canonical_url)

    def base(story_id: int, kind: str, key: str, reason: str) -> dict[str, str]:
        story = stories[story_id]
        description, outlets, url = describe(story_id)
        identity = row_identity(
            theme_set_id=set_id,
            pipeline_version=version,
            ticker=ticker,
            trading_day=day,
            story_id=str(story_id),
            assignment_type=kind,
            theme_key=key,
            placement_reason=reason,
        )
        return {
            "row_id": row_id_for(identity),
            **{k: v for k, v in identity.items() if k != "gate"},
            "story_title": _clean(story.canonical_title),
            "story_description": description,
            "story_canonical_url": url,
            "story_outlets": outlets,
            "story_stage": _clean(story.stage),
        }

    rows: list[AssignmentRow] = []
    for theme in population.themes:
        key = _clean(theme.theme_key)
        for story_id in theme.story_ids:
            siblings = [
                _clean(stories[other].canonical_title)
                for other in theme.story_ids
                if other != story_id
            ]
            rows.append(
                AssignmentRow(
                    **base(story_id, ASSIGNMENT_THEME, key, ""),
                    theme_label=_clean(theme.label),
                    theme_label_source=_clean(theme.label_source),
                    theme_story_count=str(
                        theme.story_count
                        if theme.story_count is not None
                        else len(theme.story_ids)
                    ),
                    sibling_story_titles=_join(siblings),
                )
            )
    for entry in population.other_coverage:
        rows.append(
            AssignmentRow(
                **base(entry.story_id, ASSIGNMENT_OTHER, "", _clean(entry.reason)),
                theme_label="",
                theme_label_source="",
                theme_story_count="",
                sibling_story_titles="",
            )
        )
    for entry in population.excluded:
        rows.append(
            AssignmentRow(
                **base(entry.story_id, ASSIGNMENT_EXCLUDED, "", _clean(entry.reason)),
                theme_label="",
                theme_label_source="",
                theme_story_count="",
                sibling_story_titles="",
            )
        )
    binding, build_signature = classify_generation_binding(population)
    provenance = ThemeSetProvenance(
        theme_set_id=set_id,
        ticker=ticker,
        trading_day=day,
        pipeline_version=version,
        method=theme_set.method,
        config_fingerprint=theme_set.config_fingerprint,
        algorithm_version=theme_set.algorithm_version,
        model_name=theme_set.model_name,
        model_revision=theme_set.model_revision,
        embedding_dimension=theme_set.embedding_dimension,
        updated_at=theme_set.updated_at,
        story_generation_signature_current=population.stories.signature,
        theme_build_story_generation_signature=build_signature,
        generation_binding=binding,
        theme_count=len(population.themes),
        other_coverage_count=len(population.other_coverage),
        excluded_count=len(population.excluded),
    )
    return rows, provenance


def _classify_partition(population: Any) -> tuple[str, str] | None:
    """Why a partition is not reviewable, or ``None`` when it is.

    The theme set is judged against the *current* authoritative story
    generation: the stories it places must be exactly the live stories,
    and every one of them must be healthy ``m3.semantic`` output.  A set
    left behind by an earlier generation, or built while M3 was down, is
    named as such rather than reviewed as if it were M5 output.
    """

    live = population.stories.stories
    stages = sorted(population.stories.stages)
    theme_set = population.theme_set
    identifiers: dict[str, Any] = {
        "pipeline_version": population.pipeline_version,
        "ticker": population.ticker,
    }
    if theme_set is not None:
        identifiers.update(
            {
                "method": theme_set.method,
                "config_fingerprint": theme_set.config_fingerprint,
                "algorithm_version": theme_set.algorithm_version,
                "model_name": theme_set.model_name,
                "model_revision": theme_set.model_revision,
            }
        )
        for theme in population.themes:
            identifiers[f"themes[{theme.theme_id}].theme_key"] = theme.theme_key
            identifiers[f"themes[{theme.theme_id}].label_source"] = theme.label_source
    tainted = _credential_bearing_fields(identifiers)
    if tainted:
        return (
            SKIP_PROVENANCE_CREDENTIAL,
            f"provenance identifier field(s) {tainted} carry credential-like text; "
            "identifiers are not rewritten, so the partition is not reviewable "
            "(values withheld)",
        )
    if theme_set is None:
        if not live:
            return SKIP_NO_STORY_OUTPUT, "no live stories and no theme set"
        if stages == [DEGRADED_STORY_STAGE]:
            return (
                SKIP_M2_ONLY,
                f"{len(live)} stories carry stage={DEGRADED_STORY_STAGE}; M3 did "
                "not complete and, by decision H, no theme set exists",
            )
        if stages == [HEALTHY_STORY_STAGE]:
            return (
                SKIP_THEMES_NOT_GENERATED,
                f"{len(live)} healthy stories but no theme set was persisted",
            )
        return SKIP_MIXED_STAGES, f"{len(live)} stories across stages {stages}"
    if not live:
        return (
            SKIP_STALE_SET_NO_STORIES,
            f"theme set {theme_set.theme_set_id} persists but the partition holds "
            "no live stories; the set is stale",
        )
    if stages != [HEALTHY_STORY_STAGE]:
        return (
            SKIP_STALE_SET_DEGRADED,
            f"theme set {theme_set.theme_set_id} persists over stories with stages "
            f"{stages}; the authoritative generation is not healthy M3 output",
        )
    placed: list[int] = [
        story_id for theme in population.themes for story_id in theme.story_ids
    ]
    placed += [entry.story_id for entry in population.other_coverage]
    placed += [entry.story_id for entry in population.excluded]
    live_ids = sorted(story.story_id for story in live)
    if sorted(placed) != live_ids:
        return (
            SKIP_SET_INCONSISTENT,
            f"theme set {theme_set.theme_set_id} places {sorted(placed)} but the "
            f"live generation is {live_ids}",
        )
    recorded = (theme_set.source_metadata or {}).get("story_count")
    if isinstance(recorded, int) and recorded != len(live):
        return (
            SKIP_SET_INCONSISTENT,
            f"theme set {theme_set.theme_set_id} recorded {recorded} input stories "
            f"but the live generation holds {len(live)}",
        )
    return None


def _indicators(read: Any, days: Sequence[str], populations: Sequence[Any]) -> dict:
    """Ledger facts a reader can check the sample against.

    Descriptive only.  A database seeded through ``insert_raw_items``
    satisfies every count here, which is exactly why none of them feeds
    the origin classification.
    """

    runs: dict[str, dict[str, dict[str, int]]] = {}
    for day in days:
        per_stage: dict[str, dict[str, int]] = {}
        for row in read.run_log_rows(trading_day=day):
            stage = str(row["stage"])
            if stage not in INGESTION_STAGES and stage not in DOWNSTREAM_STAGES:
                continue
            status = str(row["status"])
            per_stage.setdefault(stage, {})
            per_stage[stage][status] = per_stage[stage].get(status, 0) + 1
        runs[day] = per_stage
    members = [m for population in populations for m in population.member_provenance]
    by_scheme: dict[str, int] = {}
    for member in members:
        scheme = str(member.source or "").split(":", 1)[0] or "unknown"
        by_scheme[scheme] = by_scheme.get(scheme, 0) + 1
    return {
        "runs_by_day": runs,
        "member_raw_items": {
            "count": len(members),
            "with_fetched_at": sum(1 for m in members if m.fetched_at),
            "with_provider_payload": sum(1 for m in members if m.has_payload),
            "with_feed_snapshot": sum(1 for m in members if m.has_feed_snapshot),
            "ingest_status_valid": sum(
                1 for m in members if m.ingest_status == "valid"
            ),
            "with_external_id": sum(1 for m in members if m.external_id),
            "by_scheme": by_scheme,
        },
        "note": (
            "descriptive only; rows written by Phase0Admin.insert_raw_items "
            "satisfy every count here, so none of them establishes origin"
        ),
    }


def load_persisted_population(
    database: str | Path,
    *,
    trading_days: Sequence[str],
    tickers: Sequence[str] | None = None,
    pipeline_version: str | None = None,
) -> Population:
    """Every placement persisted for the selected partitions, exactly once.

    Partitions are enumerated by
    :meth:`phase0.repository.Phase0Reader.partition_generations` -- live
    stories *or* a theme set -- narrowed by ``tickers`` when given, and
    each is read as one snapshot.  When the selection spans more than one
    ``pipeline_version`` the caller has to name one.  The file must
    already exist: the repository would otherwise create an empty one.
    """

    from phase0.repository import Phase0Repository

    path = Path(database)
    if not path.exists():
        raise ReviewSamplingError(f"no Phase 0 database at {_clean(path)}")
    if pipeline_version is not None:
        pipeline_version = _require_clean_identifier(
            pipeline_version, "pipeline_version"
        )
    read = Phase0Repository(path).read
    days = _normalized_days(trading_days)
    wanted = None if not tickers else {t.strip().upper() for t in tickers}
    for ticker in wanted or ():
        _require_clean_identifier(ticker, "ticker")

    selected: list[tuple[str, str, str]] = []
    versions: set[str] = set()
    skipped: list[SkippedPartition] = []
    for day in days:
        found = [
            g
            for g in read.partition_generations(day)
            if pipeline_version is None or g.pipeline_version == pipeline_version
        ]
        present = {g.ticker for g in found}
        if wanted is not None:
            for ticker in sorted(wanted - present):
                selected.append((ticker, day, ""))
        for generation in found:
            if wanted is not None and generation.ticker not in wanted:
                continue
            # A discovered version is checked before it is kept anywhere:
            # not in the version set the mixed-version check formats, not in
            # a skip record, not in the population.  Only the field is named.
            if contains_credential(generation.pipeline_version):
                skipped.append(
                    SkippedPartition(
                        generation.ticker,
                        day,
                        REJECTED_IDENTIFIER,
                        SKIP_PROVENANCE_CREDENTIAL,
                        "discovered pipeline_version carries credential-like text; "
                        "the partition is not reviewable (value withheld)",
                    )
                )
                continue
            if generation.pipeline_version.strip() == REJECTED_IDENTIFIER:
                skipped.append(
                    SkippedPartition(
                        generation.ticker,
                        day,
                        REJECTED_IDENTIFIER,
                        SKIP_RESERVED_IDENTIFIER,
                        "discovered pipeline_version equals the reserved sentinel "
                        f"{REJECTED_IDENTIFIER!r}, which never names a real "
                        "identity; the partition is not reviewable",
                    )
                )
                continue
            versions.add(generation.pipeline_version)
            selected.append((generation.ticker, day, generation.pipeline_version))
    if pipeline_version is None and len(versions) > 1:
        raise ReviewSamplingError(
            f"selected partitions span pipeline versions {sorted(versions)}; "
            "name one with pipeline_version"
        )
    if pipeline_version is not None:
        versions = {pipeline_version}

    rows: list[AssignmentRow] = []
    sets: list[ThemeSetProvenance] = []
    populations: list[Any] = []
    for ticker, day, version in selected:
        if not version:
            skipped.append(
                SkippedPartition(
                    ticker,
                    day,
                    pipeline_version or "",
                    SKIP_NO_STORY_OUTPUT,
                    "no stories and no theme set persisted for this partition",
                )
            )
            continue
        population = read.theme_population(ticker, day, version)
        verdict = _classify_partition(population)
        if verdict is not None:
            reason, detail = verdict
            skipped.append(
                SkippedPartition(ticker, day, version, reason, _clean(detail))
            )
            continue
        set_rows, provenance = _partition_rows(population)
        rows.extend(set_rows)
        sets.append(provenance)
        populations.append(population)

    if not rows and not skipped:
        raise ReviewSamplingError(
            f"no persisted stories or theme sets for {days} in {path}; nothing to "
            "sample (and no fixture is substituted)"
        )
    rows.sort(key=lambda row: tuple(row.identity().values()))
    return Population(
        rows=tuple(rows),
        theme_sets=tuple(sets),
        skipped=tuple(skipped),
        trading_days=tuple(days),
        tickers=tuple(
            sorted({t for t, _, _ in selected} | {s.ticker for s in skipped})
        ),
        pipeline_versions=tuple(sorted(versions)),
        source={"mode": SOURCE_PERSISTED, "database": _clean(path)},
        indicators=_indicators(read, days, populations),
    )


def _normalized_days(values: Sequence[str]) -> list[str]:
    days: list[str] = []
    for value in values:
        try:
            day = datetime.strptime(str(value).strip(), "%Y-%m-%d").date().isoformat()
        except ValueError as exc:
            raise ReviewSamplingError(
                f"trading day {value!r} must be YYYY-MM-DD"
            ) from exc
        if day not in days:
            days.append(day)
    if not days:
        raise ReviewSamplingError("at least one trading day is required")
    return sorted(days)


# -- Fixture population (explicit development mode only) --------------------


def load_fixture_population(
    fixture_path: str | Path | None = None,
    vectors_path: str | Path | None = None,
) -> Population:
    """The committed M5 fixture, clustered offline, as a review population.

    Explicit development mode.  Nothing here was persisted, so identity is
    keyed on the fixture's digest rather than a ``theme_sets`` row, and the
    fixture's own manifest is checked to declare itself synthetic.
    """

    from nlp.embeddings import compose_embedding_text
    from nlp.themes.config import ThemeConfig
    from nlp.themes.dataset import DEFAULT_FIXTURE_PATH, load_ticker_days, tickers_of
    from nlp.themes.service import cluster_themes
    from nlp.themes.vectors import (
        DEFAULT_VECTOR_PATH,
        FixtureEncoder,
        load_story_vectors,
    )

    fixture = Path(fixture_path or DEFAULT_FIXTURE_PATH)
    vectors = Path(vectors_path or DEFAULT_VECTOR_PATH)
    day_set = load_ticker_days(fixture)
    if day_set.trust_contract.dataset_kind is not DatasetKind.SYNTHETIC_DEVELOPMENT:
        raise ReviewSamplingError(
            "the fixture's own manifest does not declare synthetic_development"
        )
    encoder = FixtureEncoder(load_story_vectors(vectors))
    encoder.bind(
        {
            story.story_key: compose_embedding_text(story.title, story.description)
            for day in day_set.days
            for story in day.stories
        }
    )
    config = ThemeConfig(supported_tickers=tickers_of(day_set))
    fixture_digest = _sha256_file(fixture)
    version = f"fixture:{fixture_digest[:12]}"

    rows: list[AssignmentRow] = []
    sets: list[ThemeSetProvenance] = []
    for day in day_set.days:
        theme_set = cluster_themes(
            day.stories,
            ticker=day.ticker,
            trading_day=day.trading_day,
            config=config,
            encoder=encoder,
        )
        ticker = day.ticker
        trading_day = day.trading_day.isoformat()
        set_id = f"fixture:{ticker}:{trading_day}:{fixture_digest[:12]}"
        titles = {s.story_key: _clean(s.title) for s in day.stories}
        tainted = _credential_bearing_fields(
            {
                "ticker": ticker,
                "method": theme_set.method.value,
                "config_fingerprint": theme_set.config_fingerprint,
                "algorithm_version": theme_set.algorithm_version,
                "model_name": theme_set.model_name,
                "model_revision": theme_set.model_revision,
                **{
                    f"theme_key[{i}]": t.theme_key
                    for i, t in enumerate(theme_set.themes)
                },
            }
        )
        if tainted:
            raise ReviewSamplingError(
                f"fixture {ticker} {trading_day}: provenance identifier field(s) "
                f"{tainted} carry credential-like text (values withheld)"
            )

        def make(story_key: str, kind: str, key: str, reason: str, **context) -> None:
            identity = row_identity(
                theme_set_id=set_id,
                pipeline_version=version,
                ticker=ticker,
                trading_day=trading_day,
                story_id=story_key,
                assignment_type=kind,
                theme_key=key,
                placement_reason=reason,
            )
            rows.append(
                AssignmentRow(
                    row_id=row_id_for(identity),
                    **{k: v for k, v in identity.items() if k != "gate"},
                    **context,
                )
            )

        for theme in theme_set.themes:
            for entry in theme.evidence:
                make(
                    entry.story_key,
                    ASSIGNMENT_THEME,
                    theme.theme_key,
                    "",
                    story_title=_clean(entry.title),
                    story_description=_clean(entry.description),
                    story_canonical_url=_join(
                        _clean(link[1]) for link in entry.source_links
                    ),
                    story_outlets=_join(_clean(o) for o in entry.outlets),
                    story_stage="fixture",
                    theme_label=_clean(theme.label),
                    theme_label_source=_clean(theme.label_source),
                    theme_story_count=str(theme.story_count),
                    sibling_story_titles=_join(
                        titles[k]
                        for k in theme.member_story_keys
                        if k != entry.story_key
                    ),
                )
        for entry in theme_set.other_coverage:
            evidence = entry.evidence
            make(
                evidence.story_key,
                ASSIGNMENT_OTHER,
                "",
                entry.reason.value,
                story_title=_clean(evidence.title),
                story_description=_clean(evidence.description),
                story_canonical_url=_join(
                    _clean(link[1]) for link in evidence.source_links
                ),
                story_outlets=_join(_clean(o) for o in evidence.outlets),
                story_stage="fixture",
                theme_label="",
                theme_label_source="",
                theme_story_count="",
                sibling_story_titles="",
            )
        for entry in theme_set.excluded:
            make(
                entry.story_key,
                ASSIGNMENT_EXCLUDED,
                "",
                entry.reason.value,
                story_title=titles.get(entry.story_key, ""),
                story_description="",
                story_canonical_url="",
                story_outlets="",
                story_stage="fixture",
                theme_label="",
                theme_label_source="",
                theme_story_count="",
                sibling_story_titles="",
            )
        sets.append(
            ThemeSetProvenance(
                theme_set_id=set_id,
                ticker=ticker,
                trading_day=trading_day,
                pipeline_version=version,
                method=theme_set.method.value,
                config_fingerprint=theme_set.config_fingerprint,
                algorithm_version=theme_set.algorithm_version,
                model_name=theme_set.model_name,
                model_revision=theme_set.model_revision,
                embedding_dimension=theme_set.embedding_dimension,
                updated_at=None,
                story_generation_signature_current=f"fixture:{fixture_digest}",
                theme_build_story_generation_signature=f"fixture:{fixture_digest}",
                generation_binding=GENERATION_BINDING_IN_PROCESS,
                theme_count=len(theme_set.themes),
                other_coverage_count=len(theme_set.other_coverage),
                excluded_count=len(theme_set.excluded),
            )
        )
    rows.sort(key=lambda row: tuple(row.identity().values()))
    return Population(
        rows=tuple(rows),
        theme_sets=tuple(sets),
        skipped=(),
        trading_days=tuple(sorted({s.trading_day for s in sets})),
        tickers=tuple(sorted({s.ticker for s in sets})),
        pipeline_versions=(version,),
        source={
            "mode": SOURCE_FIXTURE,
            "fixture": str(fixture),
            "fixture_sha256": fixture_digest,
            "vectors": str(vectors),
            "vectors_sha256": _sha256_file(vectors),
        },
        indicators=None,
    )


# -- Origin ------------------------------------------------------------------


UNVERIFIED_DETAIL = (
    "persisted rows whose ingestion origin cannot be established: raw_items "
    "carries no link to a fetch run, and Phase0Admin.insert_raw_items writes "
    "rows identical to fetched ones"
)
SYNTHETIC_DETAIL = "authored for development by the fixture's own declaration"


def classify_origin(source: Mapping[str, Any]) -> tuple[OriginStatus, str]:
    """Where the rows came from, from how they were read and nothing else.

    A database is ``UNVERIFIED``.  Not because its rows are suspected, but
    because nothing in persistence can distinguish a fetched row from a
    written one, and a classification that cannot be checked is a claim.
    ``VERIFIED_LIVE`` is returned by nothing; the day persistence records
    the fetch run behind each raw item, this function is where that
    evidence would be read, in a reviewed change.
    """

    mode = source.get("mode")
    if mode == SOURCE_FIXTURE:
        return OriginStatus.SYNTHETIC, SYNTHETIC_DETAIL
    if mode == SOURCE_PERSISTED:
        return OriginStatus.UNVERIFIED, UNVERIFIED_DETAIL
    raise ReviewSamplingError(f"unknown sample source mode {mode!r}")


@dataclass(frozen=True)
class OperatorAttestation:
    """What an operator said about the rows.  Audit metadata; changes nothing."""

    attested_by: str
    statement: str
    attested_at: str

    def as_dict(self) -> dict[str, str]:
        return {
            "attested_by": _clean(self.attested_by),
            "statement": _clean(self.statement),
            "attested_at": self.attested_at,
            "effect": "none: an attestation is recorded, never used to classify origin",
        }


# -- Sampling ----------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentSample:
    population: Population
    rows: tuple[AssignmentRow, ...]
    seed: str
    round_id: str
    requested_size: int
    excluded_row_ids: tuple[str, ...]
    prior_rounds: tuple[Mapping[str, Any], ...]

    @property
    def actual_size(self) -> int:
        return len(self.rows)

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_size - self.actual_size)


def sample_assignments(
    population: Population,
    *,
    seed: str,
    size: int = DEFAULT_ROUND_SIZE,
    round_id: str = "round-1",
    prior_manifests: Sequence[Mapping[str, Any]] = (),
) -> AssignmentSample:
    """Draw ``size`` placements uniformly, without replacement, from ``seed``.

    A prior round's manifest removes its rows from the candidates and is
    recorded by digest, so the rounds can later prove their linkage.  A
    prior drawn from a different population is refused.  When fewer
    candidates remain than requested, all of them are taken and the
    shortfall is visible; nothing is padded.
    """

    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ReviewSamplingError("sample size must be a positive integer")
    seed = str(seed or "").strip()
    if not seed:
        raise ReviewSamplingError("a seed is required so the draw is reproducible")
    excluded: set[str] = set()
    priors: list[dict[str, Any]] = []
    for prior in prior_manifests:
        if prior["population"]["digest"] != population.digest:
            raise ReviewSamplingError(
                f"prior round {prior['sample']['round_id']!r} was drawn from a "
                "different population; rounds cannot be combined"
            )
        excluded.update(prior["sample"]["row_ids"])
        priors.append(
            {
                "round_id": prior["sample"]["round_id"],
                "manifest_sha256": prior["_sha256"],
                "snapshot_sha256": prior["snapshot"]["sha256"],
                "row_count": len(prior["sample"]["row_ids"]),
            }
        )
    candidates = [row for row in population.rows if row.row_id not in excluded]
    if not candidates:
        raise ReviewSamplingError(
            "every placement in the population was already sampled"
        )
    actual = min(size, len(candidates))
    rng = random.Random(seed)
    chosen = rng.sample(candidates, actual)
    chosen.sort(key=lambda row: tuple(row.identity().values()))
    return AssignmentSample(
        population=population,
        rows=tuple(chosen),
        seed=seed,
        round_id=str(round_id),
        requested_size=size,
        excluded_row_ids=tuple(sorted(excluded)),
        prior_rounds=tuple(priors),
    )


# -- Manifest and files ------------------------------------------------------


def code_identity(root: Path | None = None) -> dict[str, Any]:
    """The commit the sample was produced from, and whether the tree was clean."""

    root = root or Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def snapshot_digest(
    rows: Sequence[Mapping[str, Any]], theme_sets: Sequence[Mapping]
) -> str:
    """Full SHA-256 over the captured review context and its provenance."""

    return sha256_of({"rows": list(rows), "theme_sets": list(theme_sets)})


def build_manifest(
    sample: AssignmentSample,
    *,
    protocol_id: str = UNRATIFIED_PROTOCOL,
    csv_name: str,
    generated_at: datetime | None = None,
    code: Mapping[str, Any] | None = None,
    operator_attestation: OperatorAttestation | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce the draw and to hold the review to it.

    ``snapshot`` is the review boundary: every non-reviewer column of every
    sampled row plus the theme-set provenance behind it, digested with
    SHA-256.  Sheets are checked against it, and the live database is not
    consulted again -- a review is of what the reviewer saw.
    """

    population = sample.population
    protocol = require_known_protocol(protocol_id)
    origin, detail = classify_origin(population.source)
    when = generated_at or datetime.now(timezone.utc)
    snapshot_rows = [row.snapshot() for row in sample.rows]
    theme_sets = [s.as_dict() for s in population.theme_sets]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "sheet_kind": SHEET_KIND,
        "gate": GATE,
        "generated_at": when.isoformat(),
        "code": dict(code if code is not None else code_identity()),
        "source": dict(population.source),
        "origin": {"status": origin.value, "detail": detail},
        "operator_attestation": (
            None if operator_attestation is None else operator_attestation.as_dict()
        ),
        "selection": {
            "trading_days": list(population.trading_days),
            "tickers": list(population.tickers),
            "pipeline_versions": list(population.pipeline_versions),
            "theme_set_ids": [s.theme_set_id for s in population.theme_sets],
            "skipped_partitions": [s.as_dict() for s in population.skipped],
            "digest": population.selection_digest,
        },
        "population": {
            "size": len(population.rows),
            "digest": population.digest,
            "by_assignment_type": population.by_assignment_type,
            "indicators": population.indicators,
        },
        "sample": {
            "method": SAMPLE_METHOD,
            "seed": sample.seed,
            "round_id": sample.round_id,
            "requested_size": sample.requested_size,
            "actual_size": sample.actual_size,
            "shortfall": sample.shortfall,
            "excluded_prior_row_ids": len(sample.excluded_row_ids),
            "prior_rounds": [dict(p) for p in sample.prior_rounds],
            "row_ids": [row.row_id for row in sample.rows],
        },
        "snapshot": {
            "rows": snapshot_rows,
            "theme_sets": theme_sets,
            "sha256": snapshot_digest(snapshot_rows, theme_sets),
        },
        "gate_requirements": {
            "threshold": RELEASE_G1_THRESHOLD,
            "required_unique_assignments": RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS,
            "round_size": sample.requested_size,
            "note": "informational; scoring uses the constants in code",
        },
        "labeling_protocol": {
            "id": protocol.id,
            "vocabulary_at_sampling": {
                "positive": protocol.positive_verdict,
                "negative": protocol.negative_verdict,
            },
            "note": (
                "ratification is decided by nlp.eval.review.RATIFIED_PROTOCOLS at "
                "scoring time, never by this file"
            ),
        },
    }
    # The manifest's identity is everything above; the sheet block below
    # describes the CSV cut from it and carries the CSV's own digest, so it
    # cannot be inside the identity the CSV binds to.
    binding = {
        "manifest_id": manifest_identity(manifest),
        "snapshot_sha256": manifest["snapshot"]["sha256"],
    }
    manifest["binding"] = binding
    manifest["sheet"] = {
        "csv": csv_name,
        "blank_sha256": _sha256_text(render_csv(sample.rows, binding)),
        "columns": list(ASSIGNMENT_FIELDNAMES),
        "identity_columns": list(IDENTITY_FIELDS),
        "context_columns": list(CONTEXT_FIELDS),
        "binding_columns": list(BINDING_FIELDS),
        "reviewer_columns": list(REVIEWER_FIELDS),
    }
    return manifest


#: Keys outside the manifest's identity: the sheet block (it carries the
#: CSV's digest, and the CSV carries the identity) and the binding block
#: itself; read-time annotations are excluded by their leading underscore.
_NON_IDENTITY_KEYS = frozenset({"sheet", "binding"})


def manifest_identity(manifest: Mapping[str, Any]) -> str:
    """SHA-256 over the manifest's content, minus the sheet and binding blocks."""

    return sha256_of(
        {
            key: value
            for key, value in manifest.items()
            if key not in _NON_IDENTITY_KEYS and not str(key).startswith("_")
        }
    )


def render_csv(rows: Sequence[AssignmentRow], binding: Mapping[str, str]) -> str:
    """The blank sheet: every row carries the manifest and snapshot it binds to."""

    if set(binding) != set(BINDING_FIELDS):
        raise ReviewSamplingError(f"sheet binding must supply {BINDING_FIELDS}")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=ASSIGNMENT_FIELDNAMES, lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({**dataclasses.asdict(row), **binding})
    return buffer.getvalue()


def write_sample(
    sample: AssignmentSample,
    csv_path: str | Path,
    *,
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write the blank sheet and its manifest beside it (``<name>.manifest.json``)."""

    csv_location = Path(csv_path)
    csv_location.parent.mkdir(parents=True, exist_ok=True)
    csv_location.write_text(
        render_csv(sample.rows, manifest["binding"]), encoding="utf-8"
    )
    manifest_location = manifest_path_for(csv_location)
    manifest_location.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_location, manifest_location


def manifest_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + ".manifest.json")


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest and re-verify its integrity before anything trusts it.

    Every row id is recomputed from its identity, the snapshot digest is
    recomputed from the snapshot, and the listings of the sample (row ids,
    snapshot rows) must agree.  A manifest whose captured context was
    altered without its digest being recomputed is refused; one whose
    digest *was* recomputed is a different artifact, and the scorecard
    names the digest it scored so the substitution is visible.
    """

    location = Path(path)
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewSamplingError(f"{location}: cannot read manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ReviewSamplingError(
            f"{location}: not a {MANIFEST_SCHEMA} manifest "
            f"(schema={payload.get('schema') if isinstance(payload, dict) else None!r})"
        )
    for key in (
        "gate",
        "source",
        "origin",
        "selection",
        "population",
        "sample",
        "snapshot",
        "labeling_protocol",
        "binding",
        "sheet",
    ):
        if key not in payload:
            raise ReviewSamplingError(f"{location}: manifest is missing {key!r}")
    if payload["gate"] != GATE:
        raise ReviewSamplingError(
            f"{location}: manifest is for gate {payload['gate']!r}"
        )
    snapshot = payload["snapshot"]
    rows = snapshot.get("rows", [])
    ids = payload["sample"].get("row_ids", [])
    if [r.get("row_id") for r in rows] != ids or len(set(ids)) != len(ids):
        raise ReviewSamplingError(
            f"{location}: sample row ids and snapshot rows disagree"
        )
    for row in rows:
        if set(row) != set(SNAPSHOT_FIELDS):
            raise ReviewSamplingError(
                f"{location}: snapshot row {row.get('row_id')} does not carry the "
                "captured review columns"
            )
        identity = row_identity(**{k: row[k] for k in IDENTITY_FIELDS})
        if row_id_for(identity) != row["row_id"]:
            raise ReviewSamplingError(
                f"{location}: row {row['row_id']} does not match its own identity"
            )
    expected = snapshot_digest(rows, snapshot.get("theme_sets", []))
    if snapshot.get("sha256") != expected:
        raise ReviewSamplingError(
            f"{location}: snapshot digest {snapshot.get('sha256')!r} does not match "
            f"the captured rows ({expected}); the review context was altered"
        )
    if not isinstance(payload["population"].get("digest"), str):
        raise ReviewSamplingError(f"{location}: population.digest is missing")
    _verify_selection(payload, location)
    for prior in payload["sample"].get("prior_rounds", []):
        for key in ("round_id", "manifest_sha256", "snapshot_sha256"):
            if not isinstance(prior.get(key), str):
                raise ReviewSamplingError(
                    f"{location}: prior round record lacks {key!r}"
                )
    binding = payload["binding"]
    if binding.get("snapshot_sha256") != snapshot["sha256"] or binding.get(
        "manifest_id"
    ) != manifest_identity(payload):
        raise ReviewSamplingError(
            f"{location}: binding does not match the manifest's own identity; the "
            "manifest was altered after its sheet was cut"
        )
    payload["_sha256"] = _sha256_file(location)
    payload["_path"] = str(location)
    return payload


def _verify_selection(payload: Mapping[str, Any], location: Path) -> None:
    """Recompute ``selection.digest`` and hold the snapshot to the selection."""

    selection = payload["selection"]
    snapshot = payload["snapshot"]
    try:
        expected = selection_digest(
            trading_days=selection["trading_days"],
            tickers=selection["tickers"],
            pipeline_versions=selection["pipeline_versions"],
            theme_set_ids=selection["theme_set_ids"],
            source_mode=payload["source"].get("mode"),
            theme_sets=snapshot["theme_sets"],
            skipped_partitions=selection["skipped_partitions"],
        )
    except (KeyError, TypeError) as exc:
        raise ReviewSamplingError(f"{location}: selection is malformed: {exc}") from exc
    if selection.get("digest") != expected:
        raise ReviewSamplingError(
            f"{location}: selection digest {selection.get('digest')!r} does not "
            f"match the selection fields ({expected}); the selection was altered"
        )
    days = set(selection["trading_days"])
    tickers = set(selection["tickers"])
    versions = set(selection["pipeline_versions"])
    set_ids = set(selection["theme_set_ids"])
    provenance_ids = [str(s.get("theme_set_id")) for s in snapshot["theme_sets"]]
    if sorted(provenance_ids) != sorted(set_ids) or len(provenance_ids) != len(set_ids):
        raise ReviewSamplingError(
            f"{location}: selection.theme_set_ids and the snapshot's theme sets differ"
        )
    for provenance in snapshot["theme_sets"]:
        if (
            provenance.get("ticker") not in tickers
            or provenance.get("trading_day") not in days
            or provenance.get("pipeline_version") not in versions
        ):
            raise ReviewSamplingError(
                f"{location}: theme set {provenance.get('theme_set_id')} lies outside "
                "the selection"
            )
    for row in snapshot["rows"]:
        if (
            row["ticker"] not in tickers
            or row["trading_day"] not in days
            or row["pipeline_version"] not in versions
            or row["theme_set_id"] not in set_ids
        ):
            raise ReviewSamplingError(
                f"{location}: snapshot row {row['row_id']} lies outside the selection"
            )
    for skipped in selection["skipped_partitions"]:
        if (
            skipped.get("trading_day") not in days
            or skipped.get("ticker") not in tickers
        ):
            raise ReviewSamplingError(
                f"{location}: skipped partition {skipped.get('ticker')} "
                f"{skipped.get('trading_day')} lies outside the selection"
            )


# -- Completed sheets --------------------------------------------------------


@dataclass(frozen=True)
class SheetRow:
    row_id: str
    verdict: str
    reviewer_id: str
    reviewed_at: str
    notes: str


@dataclass(frozen=True)
class CompletedSheet:
    path: str
    sha256: str
    reviewer_id: str
    rows: Mapping[str, SheetRow]


def _read_csv(
    location: Path, required: Sequence[str], what: str
) -> list[dict[str, str]]:
    try:
        with location.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            raw_rows = list(reader)
    except OSError as exc:
        raise ReviewSamplingError(f"{location}: cannot read {what}: {exc}") from exc
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ReviewSamplingError(f"{location}: {what} lacks columns {missing}")
    return raw_rows


def read_completed_sheet(
    path: str | Path, manifest: Mapping[str, Any], protocol: Protocol
) -> CompletedSheet:
    """Read one reviewer's sheet and hold it to the manifest snapshot.

    The row-id set must equal the manifest's exactly; every non-reviewer
    column must equal the snapshot's value for that row -- identity and
    the context the judgment was made on alike; verdicts must be in the
    protocol's vocabulary or blank; every verdict must name its reviewer;
    and one sheet is one reviewer's work.
    """

    location = Path(path)
    allowed = protocol.vocabulary | {""}
    expected = {row["row_id"]: row for row in manifest["snapshot"]["rows"]}
    binding = manifest["binding"]
    raw_rows = _read_csv(location, ASSIGNMENT_FIELDNAMES, "sheet")

    rows: dict[str, SheetRow] = {}
    reviewers: set[str] = set()
    for raw in raw_rows:
        row_id = (raw.get("row_id") or "").strip()
        for column in BINDING_FIELDS:
            if (raw.get(column) or "").strip() != binding[column]:
                raise ReviewSamplingError(
                    f"{location}: row {row_id!r} column {column!r} does not name the "
                    "manifest being scored; this sheet was cut from a different "
                    "snapshot and cannot be rebound to this one"
                )
        if row_id in rows:
            raise ReviewSamplingError(f"{location}: duplicate row_id {row_id!r}")
        if row_id not in expected:
            raise ReviewSamplingError(
                f"{location}: row {row_id!r} is not in the manifest; rows cannot be "
                "added or replaced after sampling"
            )
        for column in SNAPSHOT_FIELDS:
            if (raw.get(column) or "") != expected[row_id][column]:
                raise ReviewSamplingError(
                    f"{location}: row {row_id} column {column!r} differs from the "
                    "captured snapshot; the review context was altered"
                )
        verdict = (raw.get("reviewer_verdict") or "").strip()
        if verdict not in allowed:
            raise ReviewSamplingError(
                f"{location}: row {row_id} verdict {verdict!r} is not "
                f"{protocol.positive_verdict!r}, {protocol.negative_verdict!r}, "
                "or blank"
            )
        reviewer = (raw.get("reviewer_id") or "").strip()
        if verdict and not reviewer:
            raise ReviewSamplingError(
                f"{location}: row {row_id} carries a verdict but no reviewer_id"
            )
        if reviewer:
            reviewers.add(reviewer)
        rows[row_id] = SheetRow(
            row_id=row_id,
            verdict=verdict,
            reviewer_id=reviewer,
            reviewed_at=(raw.get("reviewed_at") or "").strip(),
            notes=(raw.get("reviewer_notes") or "").strip(),
        )
    absent = sorted(set(expected) - set(rows))
    if absent:
        raise ReviewSamplingError(
            f"{location}: sheet is missing manifest rows {absent[:5]}"
            f"{' ...' if len(absent) > 5 else ''}"
        )
    if len(reviewers) > 1:
        raise ReviewSamplingError(
            f"{location}: sheet names {sorted(reviewers)} as reviewers; one sheet "
            "is one reviewer's work"
        )
    return CompletedSheet(
        path=str(location),
        sha256=_sha256_file(location),
        reviewer_id=next(iter(reviewers), ""),
        rows=rows,
    )


@dataclass(frozen=True)
class AdjudicationRow:
    row_id: str
    final_verdict: str
    adjudicator_id: str
    adjudicated_at: str
    notes: str


@dataclass(frozen=True)
class AdjudicationSheet:
    path: str
    sha256: str
    rows: Mapping[str, AdjudicationRow]


def read_adjudication_sheet(
    path: str | Path, manifest: Mapping[str, Any], protocol: Protocol
) -> AdjudicationSheet:
    location = Path(path)
    allowed = protocol.vocabulary | {""}
    known = {row["row_id"] for row in manifest["snapshot"]["rows"]}
    raw_rows = _read_csv(location, ADJUDICATION_FIELDNAMES, "adjudication sheet")
    rows: dict[str, AdjudicationRow] = {}
    for raw in raw_rows:
        row_id = (raw.get("row_id") or "").strip()
        if row_id in rows:
            raise ReviewSamplingError(f"{location}: duplicate row_id {row_id!r}")
        if row_id not in known:
            raise ReviewSamplingError(
                f"{location}: row {row_id!r} is not in the manifest"
            )
        verdict = (raw.get("final_verdict") or "").strip()
        if verdict not in allowed:
            raise ReviewSamplingError(
                f"{location}: row {row_id} final_verdict {verdict!r} is not in the "
                "protocol vocabulary"
            )
        adjudicator = (raw.get("adjudicator_id") or "").strip()
        if verdict and not adjudicator:
            raise ReviewSamplingError(
                f"{location}: row {row_id} carries a final verdict but no "
                "adjudicator_id"
            )
        rows[row_id] = AdjudicationRow(
            row_id=row_id,
            final_verdict=verdict,
            adjudicator_id=adjudicator,
            adjudicated_at=(raw.get("adjudicated_at") or "").strip(),
            notes=(raw.get("adjudication_notes") or "").strip(),
        )
    return AdjudicationSheet(
        path=str(location), sha256=_sha256_file(location), rows=rows
    )


# -- Round scoring -------------------------------------------------------------


UNRESOLVED_BLANK = "blank"
UNRESOLVED_DISAGREEMENT = "disagreement_unadjudicated"


@dataclass(frozen=True)
class RowOutcome:
    row_id: str
    verdicts: tuple[str, ...]
    final_verdict: str
    resolved: bool | None
    unresolved_reason: str
    adjudicated: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RoundResult:
    """One round's sheets resolved against its validated manifest.

    Produced only by :func:`score_round`, from the artifacts it names.  It
    is a report of that computation and is never read back from disk as
    an input: :func:`score_gate` re-derives origin, protocol, and
    requirements from ``manifest`` (the validated object) and the code,
    and takes reviewer and adjudication facts from the sheets that were
    actually parsed.
    """

    manifest: Mapping[str, Any]
    protocol: Protocol
    ratified: bool
    reviewer_ids: tuple[str, ...]
    adjudicator_ids: tuple[str, ...]
    adjudication_state: AdjudicationState
    sheets: tuple[Mapping[str, str], ...]
    outcomes: tuple[RowOutcome, ...]
    agreement_rate: float | None
    comparable_count: int

    @property
    def round_id(self) -> str:
        return str(self.manifest["sample"]["round_id"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["_sha256"])

    @property
    def reviewer_count(self) -> int:
        return len(self.reviewer_ids)

    @property
    def unresolved(self) -> tuple[RowOutcome, ...]:
        return tuple(o for o in self.outcomes if o.resolved is None)

    @property
    def requested_size(self) -> int:
        return int(self.manifest["sample"]["requested_size"])

    @property
    def actual_size(self) -> int:
        return int(self.manifest["sample"]["actual_size"])

    @property
    def shortfall(self) -> int:
        return max(0, self.requested_size - len(self.outcomes))

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "manifest_path": self.manifest.get("_path"),
            "manifest_sha256": self.manifest_sha256,
            "snapshot_sha256": self.manifest["snapshot"]["sha256"],
            "population_digest": self.manifest["population"]["digest"],
            "protocol_id": self.protocol.id,
            "protocol_ratified": self.ratified,
            "reviewer_ids": list(self.reviewer_ids),
            "adjudicator_ids": list(self.adjudicator_ids),
            "adjudication_state": self.adjudication_state.value,
            "sheets": [dict(s) for s in self.sheets],
            "outcomes": [o.as_dict() for o in self.outcomes],
            "agreement_rate": self.agreement_rate,
            "comparable_count": self.comparable_count,
            "requested_size": self.requested_size,
            "actual_size": self.actual_size,
            "shortfall": self.shortfall,
            "note": "report only; never an input to gate scoring",
        }


def score_round(
    manifest: Mapping[str, Any],
    sheets: Sequence[str | Path],
    *,
    adjudicated: str | Path | None = None,
) -> RoundResult:
    """Resolve one round: one or two reviewer sheets, optionally adjudicated.

    Blank -> unresolved.  One reviewer -> that verdict, provisionally.
    Two agreeing -> that verdict.  Two disagreeing -> the adjudicated
    final verdict, or unresolved.  Agreement is measured only over rows
    both reviewers marked.  The adjudication *state* is derived from what
    happened to disagreements, never from whether a file was supplied.
    """

    if "_sha256" not in manifest:
        raise ReviewSamplingError("manifest must come from read_manifest")
    if not sheets:
        raise ReviewSamplingError("at least one completed sheet is required")
    if len(sheets) > 2:
        raise ReviewSamplingError("at most two reviewer sheets are scored per round")
    protocol, ratified = resolve_protocol(manifest["labeling_protocol"].get("id"))
    completed = [read_completed_sheet(path, manifest, protocol) for path in sheets]
    reviewer_ids = tuple(sheet.reviewer_id for sheet in completed)
    if len(completed) == 2 and reviewer_ids[0] and reviewer_ids[0] == reviewer_ids[1]:
        raise ReviewSamplingError(
            f"both sheets are signed by {reviewer_ids[0]!r}; two sheets from one "
            "reviewer are not two reviewers"
        )
    adjudication = (
        None
        if adjudicated is None
        else read_adjudication_sheet(adjudicated, manifest, protocol)
    )
    if adjudication is not None and len(completed) < 2:
        raise ReviewSamplingError(
            "an adjudication sheet needs two reviewer sheets to adjudicate between"
        )

    outcomes: list[RowOutcome] = []
    disagreements = 0
    open_disagreements = 0
    positive = protocol.positive_verdict
    for row_id in manifest["sample"]["row_ids"]:
        verdicts = tuple(sheet.rows[row_id].verdict for sheet in completed)
        if any(v == "" for v in verdicts):
            outcomes.append(
                RowOutcome(row_id, verdicts, "", None, UNRESOLVED_BLANK, False)
            )
            continue
        if len(verdicts) == 1 or verdicts[0] == verdicts[1]:
            outcomes.append(
                RowOutcome(
                    row_id, verdicts, verdicts[0], verdicts[0] == positive, "", False
                )
            )
            continue
        disagreements += 1
        final = ""
        if adjudication is not None and row_id in adjudication.rows:
            final = adjudication.rows[row_id].final_verdict
        if not final:
            open_disagreements += 1
            outcomes.append(
                RowOutcome(row_id, verdicts, "", None, UNRESOLVED_DISAGREEMENT, False)
            )
            continue
        outcomes.append(
            RowOutcome(row_id, verdicts, final, final == positive, "", True)
        )

    if len(completed) < 2:
        state = AdjudicationState.NOT_APPLICABLE
    elif open_disagreements:
        state = AdjudicationState.OPEN
    elif disagreements:
        state = AdjudicationState.RESOLVED
    else:
        state = AdjudicationState.UNANIMOUS

    agreement_rate: float | None = None
    comparable = 0
    if len(completed) == 2:
        pairs = [
            (completed[0].rows[r].verdict, completed[1].rows[r].verdict)
            for r in manifest["sample"]["row_ids"]
        ]
        marked = [(a, b) for a, b in pairs if a and b]
        comparable = len(marked)
        if marked:
            agreement_rate = sum(1 for a, b in marked if a == b) / comparable

    sheet_records: list[dict[str, str]] = [
        {"path": s.path, "sha256": s.sha256, "reviewer_id": s.reviewer_id}
        for s in completed
    ]
    if adjudication is not None:
        sheet_records.append(
            {
                "path": adjudication.path,
                "sha256": adjudication.sha256,
                "role": "adjudication",
            }
        )
    return RoundResult(
        manifest=manifest,
        protocol=protocol,
        ratified=ratified,
        reviewer_ids=tuple(r for r in reviewer_ids if r),
        adjudicator_ids=tuple(
            sorted(
                {
                    row.adjudicator_id
                    for row in adjudication.rows.values()
                    if row.adjudicator_id
                }
            )
            if adjudication is not None
            else ()
        ),
        adjudication_state=state,
        sheets=tuple(sheet_records),
        outcomes=tuple(outcomes),
        agreement_rate=agreement_rate,
        comparable_count=comparable,
    )


# -- The scorecard -------------------------------------------------------------


@dataclass(frozen=True)
class DevelopmentOverrides:
    """Looser requirements for a development evaluation.

    Using them makes the evaluation a development one: the scorecard is
    ``NOT_ELIGIBLE`` whatever else is true, because the release
    requirements are the spec's and not the caller's.
    """

    threshold: float | None = None
    required_unique_assignments: int | None = None

    def __post_init__(self) -> None:
        if self.threshold is not None:
            value = self.threshold
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ReviewSamplingError("threshold must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReviewSamplingError("threshold must be finite and within [0, 1]")
        if self.required_unique_assignments is not None:
            count = self.required_unique_assignments
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ReviewSamplingError(
                    "required_unique_assignments must be a positive integer"
                )

    @property
    def active(self) -> bool:
        return (
            self.threshold is not None or self.required_unique_assignments is not None
        )


@dataclass(frozen=True)
class Scorecard:
    """Gate G1 read off one or more rounds, with the four facts kept apart."""

    gate: str
    evaluation_mode: str
    threshold: float
    rate: float | None
    threshold_met: bool | None
    review_complete: bool
    gate_eligible: bool
    gate_result: GateResult
    origin_status: OriginStatus
    origin_detail: str
    trust_contract: TrustContract | None
    protocol_id: str
    protocol_ratified: bool
    reviewer_count: int
    adjudication_state: AdjudicationState
    unique_assignments: int
    required_unique_assignments: int
    resolved_count: int
    positive_count: int
    unresolved_count: int
    unresolved_row_ids: tuple[str, ...]
    agreement_rate: float | None
    reviewer_ids: tuple[str, ...]
    adjudicator_ids: tuple[str, ...]
    eligibility_blockers: tuple[str, ...]
    incompleteness: tuple[str, ...]
    rounds: tuple[Mapping[str, Any], ...]

    @property
    def banner(self) -> str:
        if self.trust_contract is not None:
            return self.trust_contract.banner()
        return (
            "WARNING: Origin unverifiable. Persisted rows whose ingestion origin "
            "cannot be established.\nMetrics are development-only and not gate "
            "eligible; no dataset kind is claimed.\n"
            f"  origin_status          {self.origin_status.value}\n"
            f"  reviewer_count         {self.reviewer_count}\n"
            f"  adjudication_state     {self.adjudication_state.value}\n"
            f"  protocol               {self.protocol_id} "
            f"({'ratified' if self.protocol_ratified else 'unratified'})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "evaluation_mode": self.evaluation_mode,
            "threshold": self.threshold,
            "rate": self.rate,
            "threshold_met": self.threshold_met,
            "review_complete": self.review_complete,
            "gate_eligible": self.gate_eligible,
            "gate_result": self.gate_result.value,
            "origin": {
                "status": self.origin_status.value,
                "detail": self.origin_detail,
            },
            "trust_contract": (
                None if self.trust_contract is None else self.trust_contract.as_dict()
            ),
            "protocol": {"id": self.protocol_id, "ratified": self.protocol_ratified},
            "reviewer_count": self.reviewer_count,
            "adjudication_state": self.adjudication_state.value,
            "unique_assignments": self.unique_assignments,
            "required_unique_assignments": self.required_unique_assignments,
            "resolved_count": self.resolved_count,
            "positive_count": self.positive_count,
            "unresolved_count": self.unresolved_count,
            "unresolved_row_ids": list(self.unresolved_row_ids),
            "agreement_rate": self.agreement_rate,
            "reviewer_ids": list(self.reviewer_ids),
            "adjudicator_ids": list(self.adjudicator_ids),
            "eligibility_blockers": list(self.eligibility_blockers),
            "incompleteness": list(self.incompleteness),
            "rounds": [dict(r) for r in self.rounds],
            "note": (
                "A4 reports one gate. The Phase 0 GO / NO-GO decision combines "
                "G1-G7 and Q1-Q3 and is recorded by K4, not computed here."
            ),
        }


def derive_gate_result(
    *, gate_eligible: bool, review_complete: bool, threshold_met: bool | None
) -> GateResult:
    """The locked precedence: eligibility, then completeness, then arithmetic."""

    if not gate_eligible:
        return GateResult.NOT_ELIGIBLE
    if not review_complete:
        return GateResult.INCOMPLETE
    if threshold_met:
        return GateResult.PASS
    return GateResult.FAIL


def _reverify_round(result: RoundResult) -> RoundResult:
    """Re-derive a round from the artifacts it names; refuse a report that differs.

    A :class:`RoundResult` is a report.  Before it counts toward a gate the
    manifest is re-read and re-verified from disk and must be the one the
    report was built on, and the sheets are re-parsed and re-scored; the
    recomputed report must equal the given one field for field.  A report
    whose reviewers, adjudication, outcomes, protocol, or provenance were
    changed after the fact -- in memory or otherwise -- does not survive
    this, and neither does a sheet edited after it was scored.
    """

    path = result.manifest.get("_path")
    if not path:
        raise ReviewSamplingError("a round's manifest must have been read from disk")
    disk = read_manifest(path)
    given = {k: v for k, v in result.manifest.items() if not k.startswith("_")}
    found = {k: v for k, v in disk.items() if not k.startswith("_")}
    if disk["_sha256"] != result.manifest.get("_sha256") or canonical_json(
        given
    ) != canonical_json(found):
        raise ReviewSamplingError(
            f"round {result.round_id!r}: the manifest on disk differs from the one "
            "this report was built on"
        )
    sheets = [s["path"] for s in result.sheets if s.get("role") != "adjudication"]
    adjudication = next(
        (s["path"] for s in result.sheets if s.get("role") == "adjudication"), None
    )
    fresh = score_round(disk, sheets, adjudicated=adjudication)
    if fresh.as_dict() != result.as_dict():
        raise ReviewSamplingError(
            f"round {result.round_id!r}: the report does not match its source "
            "artifacts (manifest, sheets, adjudication); nothing serialized or "
            "hand-built is scored"
        )
    return fresh


def _check_round_compatibility(rounds: Sequence[RoundResult]) -> None:
    """Rounds combine only when they are provably one review of one population.

    Same gate, same population digest (full context), same selection
    provenance, same source mode, same protocol id -- and an explicit
    prior-round chain: each round after the first must have been drawn
    with every earlier round excluded, recorded by manifest and snapshot
    digest at draw time.  Anything else is refused, because two
    unlinked rounds cannot show they are not the same forty twice.
    """

    first = rounds[0].manifest
    for result in rounds[1:]:
        manifest = result.manifest
        for label, path in (
            ("gate", ("gate",)),
            ("population digest", ("population", "digest")),
            ("selection provenance", ("selection", "digest")),
            ("source mode", ("source", "mode")),
            ("labeling protocol", ("labeling_protocol", "id")),
        ):
            a, b = first, manifest
            for key in path:
                a, b = a[key], b[key]
            if a != b:
                raise ReviewSamplingError(
                    f"round {result.round_id!r} does not share the {label} of round "
                    f"{rounds[0].round_id!r} ({b!r} vs {a!r}); rounds cannot be "
                    "combined"
                )
    ordered = sorted(rounds, key=lambda r: len(r.manifest["sample"]["prior_rounds"]))
    seen_rows: set[str] = set()
    for index, result in enumerate(ordered):
        priors = {
            prior["manifest_sha256"]: prior["snapshot_sha256"]
            for prior in result.manifest["sample"]["prior_rounds"]
        }
        for earlier in ordered[:index]:
            recorded = priors.get(earlier.manifest_sha256)
            if recorded != earlier.manifest["snapshot"]["sha256"]:
                raise ReviewSamplingError(
                    f"round {result.round_id!r} was not drawn excluding round "
                    f"{earlier.round_id!r} (manifest {earlier.manifest_sha256[:12]}); "
                    "every earlier round must have been excluded when a later one "
                    "was drawn (--exclude-manifest), so rounds cannot be combined"
                )
        overlap = seen_rows & set(result.manifest["sample"]["row_ids"])
        if overlap:
            raise ReviewSamplingError(
                f"round {result.round_id!r} repeats {len(overlap)} rows of an earlier "
                "round; the same assignment cannot count twice"
            )
        seen_rows.update(result.manifest["sample"]["row_ids"])


def score_gate(
    rounds: Sequence[RoundResult],
    *,
    development: DevelopmentOverrides | None = None,
) -> Scorecard:
    """Combine rounds into the G1 scorecard.

    Origin comes from the manifests' source, ratification from the code
    registry, requirements from the constants (or from ``development``,
    which makes the evaluation ineligible), reviewer and adjudication
    facts from the sheets each round parsed.  Nothing serialized is
    believed about any of those.
    """

    if not rounds:
        raise ReviewSamplingError("at least one scored round is required")
    if any(not isinstance(r, RoundResult) for r in rounds):
        raise ReviewSamplingError("rounds must come from score_round")
    rounds = [_reverify_round(r) for r in rounds]
    _check_round_compatibility(rounds)

    development = development or DevelopmentOverrides()
    mode = "development" if development.active else "release"
    threshold = (
        RELEASE_G1_THRESHOLD if development.threshold is None else development.threshold
    )
    required = (
        RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS
        if development.required_unique_assignments is None
        else development.required_unique_assignments
    )

    origin, origin_detail = classify_origin(rounds[0].manifest["source"])
    protocol, ratified = resolve_protocol(
        rounds[0].manifest["labeling_protocol"].get("id")
    )

    outcomes = [o for r in rounds for o in r.outcomes]
    resolved = [o for o in outcomes if o.resolved is not None]
    unresolved = sorted(o.row_id for o in outcomes if o.resolved is None)
    positives = sum(1 for o in resolved if o.resolved)
    rate = (positives / len(resolved)) if resolved else None
    threshold_met = None if rate is None else rate >= threshold

    incompleteness: list[str] = []
    if unresolved:
        incompleteness.append(f"{len(unresolved)} sampled rows are unresolved")
    if len(outcomes) < required:
        incompleteness.append(
            f"{len(outcomes)} unique sampled assignments; section 8 requires "
            f">= {required}"
        )
    for result in rounds:
        if result.shortfall:
            incompleteness.append(
                f"round {result.round_id} drew {result.actual_size} of "
                f"{result.requested_size} requested (population exhausted)"
            )
    review_complete = not incompleteness

    reviewer_ids = tuple(sorted({r for result in rounds for r in result.reviewer_ids}))
    adjudicator_ids = tuple(
        sorted({a for result in rounds for a in result.adjudicator_ids})
    )
    reviewer_count = min(result.reviewer_count for result in rounds)
    states = [result.adjudication_state for result in rounds]
    if AdjudicationState.OPEN in states:
        adjudication_state = AdjudicationState.OPEN
    elif AdjudicationState.NOT_APPLICABLE in states:
        adjudication_state = AdjudicationState.NOT_APPLICABLE
    elif AdjudicationState.RESOLVED in states:
        adjudication_state = AdjudicationState.RESOLVED
    else:
        adjudication_state = AdjudicationState.UNANIMOUS
    adjudicated = ratified and adjudication_state in protocol.adjudicated_states

    blockers: list[str] = []
    if origin is not OriginStatus.VERIFIED_LIVE:
        blockers.append(f"origin is {origin.value}: {origin_detail}")
    bindings = sorted(
        {
            str(provenance.get("generation_binding"))
            for result in rounds
            for provenance in result.manifest["snapshot"]["theme_sets"]
        }
    )
    if bindings != [GENERATION_BINDING_VERIFIED]:
        blockers.append(
            f"theme-set build provenance is {bindings}: nothing persisted records "
            "the story generation a theme set was built over, so the current "
            "generation's signature proves nothing about the build"
        )
    if not ratified:
        blockers.append(
            f"labeling protocol {rounds[0].manifest['labeling_protocol'].get('id')!r} "
            "is not in the ratified registry (nlp.eval.review.RATIFIED_PROTOCOLS)"
        )
    if reviewer_count < 2:
        blockers.append("fewer than two reviewers; section 8 requires two, adjudicated")
    elif not adjudicated:
        blockers.append(
            f"adjudication state {adjudication_state.value!r} is not one the "
            "protocol counts as adjudicated"
        )
    if development.active:
        blockers.append(
            f"development overrides in effect (threshold={threshold}, "
            f"required={required}); release requirements are fixed by the spec"
        )
    gate_eligible = not blockers

    if reviewer_count < 2:
        labeling = LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED
        contract_reviewers = 1
    elif adjudicated:
        labeling = LabelingStatus.MULTI_REVIEWER_ADJUDICATED
        contract_reviewers = reviewer_count
    else:
        labeling = LabelingStatus.MULTI_REVIEWER_UNADJUDICATED
        contract_reviewers = reviewer_count

    contract: TrustContract | None = None
    if origin is OriginStatus.SYNTHETIC:
        contract = TrustContract(
            dataset_kind=DatasetKind.SYNTHETIC_DEVELOPMENT,
            real_ingested_evidence=False,
            labeling_status=labeling,
            reviewer_count=contract_reviewers,
            adjudicated=adjudicated,
            gate_eligible=False,
            metrics_purpose=MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY,
        )
    elif origin is OriginStatus.VERIFIED_LIVE:
        contract = TrustContract(
            dataset_kind=DatasetKind.SAMPLED_PRODUCTION,
            real_ingested_evidence=True,
            labeling_status=labeling,
            reviewer_count=contract_reviewers,
            adjudicated=adjudicated,
            gate_eligible=gate_eligible,
            metrics_purpose=(
                MetricsPurpose.GATE_ACCEPTANCE
                if gate_eligible
                else MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY
            ),
        )
    # UNVERIFIED: no dataset kind is truthful, so no contract is constructed.

    return Scorecard(
        gate=GATE,
        evaluation_mode=mode,
        threshold=threshold,
        rate=rate,
        threshold_met=threshold_met,
        review_complete=review_complete,
        gate_eligible=gate_eligible,
        gate_result=derive_gate_result(
            gate_eligible=gate_eligible,
            review_complete=review_complete,
            threshold_met=threshold_met,
        ),
        origin_status=origin,
        origin_detail=origin_detail,
        trust_contract=contract,
        protocol_id=protocol.id,
        protocol_ratified=ratified,
        reviewer_count=reviewer_count,
        adjudication_state=adjudication_state,
        unique_assignments=len(outcomes),
        required_unique_assignments=required,
        resolved_count=len(resolved),
        positive_count=positives,
        unresolved_count=len(unresolved),
        unresolved_row_ids=tuple(unresolved),
        agreement_rate=_pooled_agreement(rounds),
        reviewer_ids=reviewer_ids,
        adjudicator_ids=adjudicator_ids,
        eligibility_blockers=tuple(blockers),
        incompleteness=tuple(incompleteness),
        rounds=tuple(
            {
                "round_id": r.round_id,
                "manifest_path": r.manifest.get("_path"),
                "manifest_sha256": r.manifest_sha256,
                "snapshot_sha256": r.manifest["snapshot"]["sha256"],
                "population_digest": r.manifest["population"]["digest"],
                "reviewer_ids": list(r.reviewer_ids),
                "adjudication_state": r.adjudication_state.value,
                "actual_size": r.actual_size,
                "requested_size": r.requested_size,
                "agreement_rate": r.agreement_rate,
                "sheets": [dict(s) for s in r.sheets],
            }
            for r in rounds
        ),
    )


def _pooled_agreement(rounds: Sequence[RoundResult]) -> float | None:
    comparable = sum(r.comparable_count for r in rounds)
    if not comparable:
        return None
    agreed = sum(
        (r.agreement_rate or 0.0) * r.comparable_count
        for r in rounds
        if r.agreement_rate is not None
    )
    return agreed / comparable


def render_scorecard(scorecard: Scorecard) -> str:
    """The scorecard as an operator reads it: banner first, then the four facts."""

    lines = [scorecard.banner, ""]
    lines.append(f"gate               {scorecard.gate} ({scorecard.evaluation_mode})")
    lines.append(f"gate_result        {scorecard.gate_result.value}")
    met = scorecard.threshold_met
    lines.append(f"threshold_met      {'n/a' if met is None else str(met).lower()}")
    lines.append(f"review_complete    {str(scorecard.review_complete).lower()}")
    lines.append(f"gate_eligible      {str(scorecard.gate_eligible).lower()}")
    rate = "n/a" if scorecard.rate is None else f"{scorecard.rate:.4f}"
    lines.append(f"rate               {rate}  (threshold {scorecard.threshold})")
    lines.append(
        f"assignments        {scorecard.unique_assignments} unique "
        f"(required >= {scorecard.required_unique_assignments}); "
        f"{scorecard.resolved_count} resolved, {scorecard.unresolved_count} unresolved"
    )
    if scorecard.agreement_rate is not None:
        lines.append(f"agreement_rate     {scorecard.agreement_rate:.4f}")
    lines.append(f"reviewers          {', '.join(scorecard.reviewer_ids) or '-'}")
    if scorecard.adjudicator_ids:
        lines.append(f"adjudicators       {', '.join(scorecard.adjudicator_ids)}")
    lines.append(f"adjudication       {scorecard.adjudication_state.value}")
    for r in scorecard.rounds:
        lines.append(
            f"round {r['round_id']}: manifest {r['manifest_sha256'][:12]} "
            f"snapshot {r['snapshot_sha256'][:12]}"
        )
    if scorecard.eligibility_blockers:
        lines.append("not gate eligible because:")
        lines.extend(f"  - {b}" for b in scorecard.eligibility_blockers)
    if scorecard.incompleteness:
        lines.append("review incomplete because:")
        lines.extend(f"  - {i}" for i in scorecard.incompleteness)
    lines.append("")
    lines.append(
        "A4 reports one gate; the Phase 0 GO / NO-GO decision is K4's, not this tool's."
    )
    return "\n".join(lines)


__all__ = [
    "ADJUDICATION_FIELDNAMES",
    "ASSIGNMENT_FIELDNAMES",
    "BINDING_FIELDS",
    "CONTEXT_FIELDS",
    "GENERATION_BINDING_IN_PROCESS",
    "GENERATION_BINDING_UNVERIFIED",
    "GENERATION_BINDING_VERIFIED",
    "DEFAULT_ROUND_SIZE",
    "IDENTITY_FIELDS",
    "MANIFEST_SCHEMA",
    "PROVISIONAL_PROTOCOL",
    "RATIFIED_PROTOCOLS",
    "RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS",
    "REJECTED_IDENTIFIER",
    "RELEASE_G1_THRESHOLD",
    "SNAPSHOT_FIELDS",
    "UNRATIFIED_PROTOCOL",
    "AdjudicationState",
    "AssignmentRow",
    "AssignmentSample",
    "DevelopmentOverrides",
    "GateResult",
    "OperatorAttestation",
    "OriginStatus",
    "Population",
    "Protocol",
    "ReviewSamplingError",
    "RoundResult",
    "Scorecard",
    "SkippedPartition",
    "build_manifest",
    "classify_generation_binding",
    "classify_origin",
    "code_identity",
    "derive_gate_result",
    "load_fixture_population",
    "load_persisted_population",
    "manifest_identity",
    "manifest_path_for",
    "read_manifest",
    "render_scorecard",
    "resolve_protocol",
    "row_id_for",
    "row_identity",
    "sample_assignments",
    "score_gate",
    "score_round",
    "selection_digest",
    "sha256_of",
    "snapshot_digest",
    "write_sample",
]

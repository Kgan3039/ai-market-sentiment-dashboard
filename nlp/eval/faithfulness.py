"""A4b (issue #74): G2 sentence-faithfulness review sampling, and its scorecard.

Section 8 of ``docs/PHASE_0_SPEC.md`` reads gate G2 -- summary-sentence
faithfulness, "supported by cited source", >= 95%, "every sentence from 2
sampled days" -- off human review sheets.  This module produces those sheets
and reads them back.  It generates nothing and calls no model: what is
reviewed is what was persisted.

**The population is what a reader may be shown at sampling time.**  For each
candidate day, each of the five tickers is read as one
:meth:`~phase0.repository.Phase0Reader.theme_population` snapshot, and each
theme's summary is looked up with
:func:`phase0.summary_lifecycle.current_summary_artifact` under the
production policy -- the same reader and the same policy the narrative API
serves with.  Only an artifact current *for that snapshot* is reviewed; a
partition whose population moves while it is read is skipped as
``population_changed_during_sampling``, never assembled from two moments.
The claim is "artifacts eligible to be served at sampling time", not "every
artifact served during the day": schema 16 does not keep the frozen evidence
of superseded artifacts, and no HTTP request is logged.

**Two days, drawn, never picked.**  The operator names candidate days; a day
is eligible only if it holds at least one reviewable current summary; the
two reviewed days are a seeded draw from the eligible ones.  Candidate,
eligible, excluded (with reason), seed and selected days are all in the
manifest, and reading the manifest re-runs the draw.

**The review unit is one sentence with its complete ordered citation set.**
Every sentence of every reviewed artifact on the two days is one row, once.
What "supported" means when citations disagree, when support is partial, or
when a claim cannot be checked is K3's protocol, not this module's: the
vocabulary here is the provisional, unratified ``supported`` /
``unsupported``, and nothing scored under it can be gate eligible.

**The manifest proves itself offline.**  Each artifact's frozen evidence is
captured whole, so reading a manifest recomputes the artifact's
``input_fingerprint`` (A2's digest of what the model saw) and its
``content_digest`` (A3's digest of what was accepted) without a database.
Evidence that does not reproduce the fingerprint, a sentence or citation
that does not reproduce the digest, or a row that is not the exact
projection of its artifact is refused.

**Four facts, kept apart**, exactly as for G1: ``threshold_met`` is
arithmetic, ``review_complete`` is whether the census was finished,
``gate_eligible`` is whether the numbers may settle anything, and
``gate_result`` is derived from the three.  A rate of 0.95 or more on an
ineligible round is a measurement, never a PASS.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import io
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ai.guarded_summary import (
    EvidenceStory,
    GenerationPolicy,
    GuardedSummaryError,
    ThemeReference,
    citation_id_for,
    compute_input_fingerprint,
    compute_policy_fingerprint,
)
from ai.summarization import ProviderConfigurationError
from nlp.eval.review import (
    BINDING_FIELDS,
    GENERATION_BINDING_VERIFIED,
    REJECTED_IDENTIFIER,
    REVIEWER_FIELDS,
    SKIP_NO_STORY_OUTPUT,
    SKIP_PROVENANCE_CREDENTIAL,
    SKIP_RESERVED_IDENTIFIER,
    SOURCE_PERSISTED,
    UNRATIFIED_PROTOCOL,
    AdjudicationState,
    GateResult,
    OperatorAttestation,
    OriginStatus,
    Protocol,
    ReviewSamplingError,
    RoundResult,
    SheetSpec,
    _clean,
    _normalized_days,
    _require_clean_identifier,
    _sha256_file,
    _sha256_text,
    canonical_json,
    classify_generation_binding,
    classify_origin,
    code_identity,
    derive_gate_result,
    score_round,
    sha256_of,
)
from phase0.redaction import contains_credential
from phase0.repository import (
    DATABASE_READ_ERRORS,
    Phase0Reader,
    summary_artifact_digest,
)
from phase0.summaries import (
    SummaryInputError,
    assess_population,
    build_generation_input,
)
from phase0.summary_lifecycle import current_summary_artifact
from phase0.tickers import TICKER_UNIVERSE

#: Bumped when the manifest's shape changes.  Distinct from A4a's, so a G1
#: manifest is never read as a G2 one, nor the reverse.
MANIFEST_SCHEMA = "a4b-g2-review-sample/1"
SHEET_KIND = "sentence_faithfulness"
GATE = "G2"

#: Section 8's release requirements for G2, fixed here and never read from
#: an artifact or a flag.  A development override exists and forces a
#: development evaluation that cannot be eligible.
RELEASE_G2_THRESHOLD = 0.95
RELEASE_G2_REQUIRED_DAYS = 2

DRAW_METHOD = "seeded_uniform_without_replacement"
CENSUS_METHOD = "census_of_selected_days"
#: A window this long is almost certainly a mistake, and every candidate
#: day costs one read per ticker.
MAX_CANDIDATE_DAYS = 366

CLAIM = (
    "artifacts eligible to be served at sampling time under the production "
    "policy; not every artifact served during the day, and not proof that any "
    "artifact was served over HTTP"
)

# -- Partition, theme, and day outcomes ----------------------------------------

PARTITION_ENUMERATED = "enumerated"
PARTITION_POPULATION_REFUSED = "population_refused"
PARTITION_POPULATION_CHANGED = "population_changed_during_sampling"

THEME_CURRENT = "current_summary"
THEME_NO_CURRENT = "no_current_summary"
THEME_INPUT_REFUSED = "input_refused"
THEME_WITHHELD = "withheld"

#: Why a current artifact is withheld from review.  The text under review,
#: and the evidence it is judged against, are never redacted: a redacted
#: sentence is not the sentence that was served, and redacted evidence would
#: no longer reproduce the input fingerprint.  So a credential-like value in
#: any of them withholds the artifact, naming the field and never the value.
WITHHELD_SENTENCE = "sentence_credential"
WITHHELD_LABEL = "label_credential"
WITHHELD_EVIDENCE = "evidence_credential"
WITHHELD_IDENTIFIER = "identifier_credential"

DAY_NO_PARTITIONS = "no_partitions"
DAY_NO_CURRENT_SUMMARY = "no_current_summary"
DAY_NO_REVIEWABLE_SUMMARY = "no_reviewable_current_summary"
DAY_POPULATION_CHANGED = PARTITION_POPULATION_CHANGED


# -- The G2 protocol registry --------------------------------------------------

#: G2's provisional protocol.  Binary, sentence-level, and unratified: K3
#: owns what "supported" means for multiple citations, partial support,
#: unverifiable claims and contradictions, and whether a reviewer may open
#: the publisher's page.  Nothing is adjudicated in the gate's sense until
#: K3 says what counts.
PROVISIONAL_G2_PROTOCOL = Protocol(
    id=UNRATIFIED_PROTOCOL,
    positive_verdict="supported",
    negative_verdict="unsupported",
    adjudicated_states=frozenset(),
)
#: Ratification for G2 lives here and only here, separate from G1's
#: registry: a protocol ratified for theme assignment says nothing about
#: sentence faithfulness.
RATIFIED_G2_PROTOCOLS: Mapping[str, Protocol] = {}


def resolve_g2_protocol(protocol_id: Any) -> tuple[Protocol, bool]:
    """The G2 protocol to score under, and whether it is ratified -- from code."""

    identifier = str(protocol_id or "").strip()
    if identifier in RATIFIED_G2_PROTOCOLS:
        return RATIFIED_G2_PROTOCOLS[identifier], True
    return PROVISIONAL_G2_PROTOCOL, False


def require_known_g2_protocol(protocol_id: str) -> Protocol:
    """At sampling time only ``unratified`` or a ratified G2 id is accepted."""

    identifier = require_clean_operator_value(protocol_id, "protocol")
    if identifier == UNRATIFIED_PROTOCOL:
        return PROVISIONAL_G2_PROTOCOL
    if identifier in RATIFIED_G2_PROTOCOLS:
        return RATIFIED_G2_PROTOCOLS[identifier]
    raise ReviewSamplingError(
        f"unknown G2 labeling protocol {identifier!r}; use "
        f"{UNRATIFIED_PROTOCOL!r} or one of {sorted(RATIFIED_G2_PROTOCOLS)}"
    )


def require_clean_operator_value(value: Any, field: str) -> str:
    """An operator-supplied value that the manifest records verbatim.

    A seed or round id cannot be redacted after the fact -- the seed *is*
    the draw -- so a credential-like one is refused before it is used,
    naming the field and never the value.
    """

    return _require_clean_identifier(str(value or "").strip(), field).strip()


# -- Rows ----------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceRow:
    """One generated sentence with its complete ordered citation set.

    The eleven fields after ``row_id`` are the durable identity: the
    artifact (by id, fingerprints and content digest) and the sentence
    within it (ordinal and ordered cited story ids).  The context columns
    are what the reviewer judges: the sentence, the summary label, and the
    frozen evidence of every cited story, in citation order.
    """

    row_id: str
    pipeline_version: str
    ticker: str
    trading_day: str
    theme_key: str
    theme_id: str
    artifact_id: str
    input_fingerprint: str
    policy_fingerprint: str
    content_digest: str
    sentence_ordinal: str
    citation_story_ids: str
    summary_label: str
    sentence_count: str
    sentence_text: str
    cited_evidence: str
    reviewer_id: str = ""
    reviewed_at: str = ""
    reviewer_verdict: str = ""
    reviewer_notes: str = ""

    def snapshot(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in SNAPSHOT_FIELDS}


IDENTITY_FIELDS = (
    "pipeline_version",
    "ticker",
    "trading_day",
    "theme_key",
    "theme_id",
    "artifact_id",
    "input_fingerprint",
    "policy_fingerprint",
    "content_digest",
    "sentence_ordinal",
    "citation_story_ids",
)
SNAPSHOT_FIELDS = tuple(
    f.name for f in dataclasses.fields(SentenceRow) if f.name not in REVIEWER_FIELDS
)
CONTEXT_FIELDS = tuple(
    f for f in SNAPSHOT_FIELDS if f not in IDENTITY_FIELDS and f != "row_id"
)
SENTENCE_FIELDNAMES = SNAPSHOT_FIELDS + BINDING_FIELDS + REVIEWER_FIELDS

#: The G2 sheet: its own columns, its own protocol registry, A4a's reader.
G2_SHEET = SheetSpec(
    fieldnames=SENTENCE_FIELDNAMES,
    snapshot_fields=SNAPSHOT_FIELDS,
    resolve_protocol=resolve_g2_protocol,
)

LIST_SEPARATOR = " | "


EVALUATION_RELEASE = "release"
EVALUATION_DEVELOPMENT = "development"


def row_id_for(identity: Mapping[str, str], *, development_override: bool) -> str:
    """``g2-`` + SHA-256 over the gate, the draw's evaluation shape, and the identity.

    Never a G1 id.  A row drawn under a development override has a different
    id from the same sentence drawn for release, so a development review
    cannot be relabelled as a release one without every row id -- and
    every reviewer's sheet -- changing with it.
    """

    missing = sorted(set(IDENTITY_FIELDS) - set(identity))
    if missing:
        raise ReviewSamplingError(f"row identity is missing {missing}")
    payload = {
        "gate": GATE,
        "evaluation": (
            EVALUATION_DEVELOPMENT if development_override else EVALUATION_RELEASE
        ),
        **{key: str(identity[key]) for key in IDENTITY_FIELDS},
    }
    return f"g2-{sha256_of(payload)}"


def render_cited_evidence(
    citations: Sequence[Mapping[str, Any]], evidence: Mapping[str, Mapping[str, Any]]
) -> str:
    """Every cited story's frozen evidence, in citation order, as a reviewer reads it.

    Title and description are exactly what the model was shown; the URL is
    a reference only (redacted, and not needed to judge support).
    """

    blocks = []
    for citation in citations:
        story = evidence[citation_id_for(int(citation["story_id"]))]
        urls = story.get("urls") or []
        blocks.append(
            "\n".join(
                (
                    f"[{citation['position']}] {story['citation_id']}"
                    f"{LIST_SEPARATOR}{story['outlet'] or '(no outlet)'}"
                    f"{LIST_SEPARATOR}"
                    f"{story['published_at'] or '(no publication time)'}",
                    f"Title: {story['title']}",
                    f"Description: {story['description'] or '(none)'}",
                    f"URL: {urls[0] if urls else '(none)'}",
                )
            )
        )
    return "\n\n".join(blocks)


def rows_for_artifact(
    artifact: Mapping[str, Any], *, development_override: bool
) -> list[SentenceRow]:
    """The artifact's sentences as review rows: a pure projection of the snapshot."""

    evidence = {story["citation_id"]: story for story in artifact["evidence"]}
    sentences = artifact["sentences"]
    rows = []
    for sentence in sentences:
        citations = sorted(sentence["citations"], key=lambda c: int(c["position"]))
        identity = {
            "pipeline_version": str(artifact["pipeline_version"]),
            "ticker": str(artifact["ticker"]),
            "trading_day": str(artifact["trading_day"]),
            "theme_key": str(artifact["theme_key"]),
            "theme_id": str(artifact["theme_id"]),
            "artifact_id": str(artifact["artifact_id"]),
            "input_fingerprint": str(artifact["input_fingerprint"]),
            "policy_fingerprint": str(artifact["policy_fingerprint"]),
            "content_digest": str(artifact["content_digest"]),
            "sentence_ordinal": str(sentence["ordinal"]),
            "citation_story_ids": LIST_SEPARATOR.join(
                citation_id_for(int(c["story_id"])) for c in citations
            ),
        }
        rows.append(
            SentenceRow(
                row_id=row_id_for(identity, development_override=development_override),
                **identity,
                summary_label=str(artifact["label"]),
                sentence_count=str(len(sentences)),
                sentence_text=str(sentence["text"]),
                cited_evidence=render_cited_evidence(citations, evidence),
            )
        )
    return rows


# -- Candidate days ------------------------------------------------------------


def expand_candidate_days(
    days: Sequence[str] = (), window: tuple[str, str] | None = None
) -> list[str]:
    """Named days plus every calendar day of an inclusive window, sorted, unique."""

    named = list(days)
    if window is not None:
        start, end = _normalized_days([window[0]])[0], _normalized_days([window[1]])[0]
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        if last < first:
            raise ReviewSamplingError(f"window end {end} is before its start {start}")
        span = (last - first).days + 1
        if span > MAX_CANDIDATE_DAYS:
            raise ReviewSamplingError(
                f"a {span}-day window exceeds {MAX_CANDIDATE_DAYS} candidate days"
            )
        named.extend((first + timedelta(days=n)).isoformat() for n in range(span))
    if not named:
        raise ReviewSamplingError("name candidate days or a candidate window")
    expanded = _normalized_days(named)
    if len(expanded) > MAX_CANDIDATE_DAYS:
        raise ReviewSamplingError(
            f"{len(expanded)} candidate days exceed {MAX_CANDIDATE_DAYS}"
        )
    return expanded


def draw_days(eligible: Sequence[str], *, seed: str, size: int) -> list[str]:
    """Draw ``size`` days uniformly without replacement, purely from ``seed``."""

    pool = sorted(eligible)
    if len(pool) < size:
        raise ReviewSamplingError(
            f"{len(pool)} eligible day(s) {pool}; {size} must be drawn -- widen the "
            "candidate window"
        )
    return sorted(random.Random(seed).sample(pool, size))


# -- Population ----------------------------------------------------------------


@dataclass(frozen=True)
class SentencePopulation:
    """Every candidate day's partitions and reviewable artifacts, read once."""

    candidate_input: Mapping[str, Any]
    candidate_days: tuple[str, ...]
    pipeline_version: str
    policy: Mapping[str, Any]
    partitions: tuple[Mapping[str, Any], ...]
    #: Reviewable artifacts on every candidate day, in canonical order.
    artifacts: tuple[Mapping[str, Any], ...]
    source: Mapping[str, Any]

    def day_accounting(self) -> list[dict[str, Any]]:
        return day_accounting(self.candidate_days, self.partitions)

    @property
    def eligible_days(self) -> list[str]:
        return [d["trading_day"] for d in self.day_accounting() if d["eligible"]]


def _policy_record(policy: GenerationPolicy) -> dict[str, Any]:
    return {
        "fingerprint": policy.fingerprint,
        "model": policy.model,
        "max_attempts": policy.max_attempts,
        "temperature": policy.temperature,
        "max_output_tokens": policy.max_output_tokens,
        "rules": [list(rule) for rule in policy.rules],
    }


def _population_state(population: Any) -> str:
    """Everything the partition's layout depends on, digested for comparison."""

    theme_set = population.theme_set
    return sha256_of(
        {
            "theme_set": (
                None
                if theme_set is None
                else [
                    theme_set.theme_set_id,
                    theme_set.updated_at,
                    theme_set.config_fingerprint,
                ]
            ),
            "signature": population.stories.signature,
            "stories": sorted(s.story_id for s in population.stories.stories),
            "themes": [
                [t.theme_id, t.theme_key, t.salience_rank, list(t.story_ids)]
                for t in population.themes
            ],
            "other": [[o.story_id, o.position] for o in population.other_coverage],
            "excluded": [e.story_id for e in population.excluded],
        }
    )


def _withheld_field(current: Any) -> tuple[str, str] | None:
    """The first credential-bearing field of a current artifact, by name only."""

    artifact = current.artifact
    for name in (
        "ticker",
        "pipeline_version",
        "theme_key",
        "input_fingerprint",
        "policy_fingerprint",
        "content_digest",
        "citation_convention",
        "prompt_version",
        "model",
        "guarantee",
    ):
        if contains_credential(str(getattr(artifact, name))):
            return WITHHELD_IDENTIFIER, name
    if contains_credential(artifact.label):
        return WITHHELD_LABEL, "label"
    for sentence in artifact.sentences:
        if contains_credential(sentence.text):
            return WITHHELD_SENTENCE, f"sentence {sentence.ordinal}"
    for story in current.generation_input.evidence:
        for name in ("title", "description", "outlet", "published_at"):
            if contains_credential(getattr(story, name)):
                return WITHHELD_EVIDENCE, f"{story.citation_id} {name}"
    return None


def _artifact_record(current: Any, salience_rank: int) -> dict[str, Any]:
    artifact = current.artifact
    reference = current.generation_input.theme
    return {
        "artifact_id": artifact.artifact_id,
        "ticker": artifact.ticker,
        "trading_day": artifact.trading_day,
        "pipeline_version": artifact.pipeline_version,
        "theme_id": artifact.theme_id,
        "theme_key": artifact.theme_key,
        "theme_label": _clean(reference.label),
        "salience_rank": salience_rank,
        "input_fingerprint": artifact.input_fingerprint,
        "policy_fingerprint": artifact.policy_fingerprint,
        "citation_convention": artifact.citation_convention,
        "prompt_version": artifact.prompt_version,
        "model": artifact.model,
        "label": artifact.label,
        "guarantee": artifact.guarantee,
        "content_digest": artifact.content_digest,
        "sentences": [
            {
                "ordinal": sentence.ordinal,
                "text": sentence.text,
                "citations": [
                    {"position": c.position, "story_id": c.story_id}
                    for c in sentence.citations
                ],
            }
            for sentence in artifact.sentences
        ],
        "evidence": [
            {
                "citation_id": story.citation_id,
                "persisted_story_id": story.persisted_story_id,
                "title": story.title,
                "description": story.description,
                "outlet": story.outlet,
                "published_at": story.published_at,
                "raw_item_ids": list(story.raw_item_ids),
                "urls": [_clean(url) for url in story.urls],
            }
            for story in current.generation_input.evidence
        ],
    }


def _partition_record(
    ticker: str,
    day: str,
    version: str,
    outcome: str,
    *,
    reason: str = "",
    detail: str = "",
    generation_binding: str | None = None,
    themes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    reviewed = [t["artifact_id"] for t in themes if t["outcome"] == THEME_CURRENT]
    return {
        "ticker": ticker,
        "trading_day": day,
        "pipeline_version": version,
        "outcome": outcome,
        "reason": reason,
        "detail": _clean(detail),
        "generation_binding": generation_binding,
        "theme_count": len(themes),
        "degraded_theme_count": sum(
            1 for t in themes if t["outcome"] in (THEME_NO_CURRENT, THEME_INPUT_REFUSED)
        ),
        "withheld_artifact_count": sum(
            1 for t in themes if t["outcome"] == THEME_WITHHELD
        ),
        "reviewed_artifact_ids": reviewed,
        "themes": [dict(t) for t in themes],
    }


def _read_partition(
    reader: Phase0Reader,
    ticker: str,
    day: str,
    version: str,
    policy: GenerationPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One partition, from one snapshot, with a race guard; nothing written."""

    first = reader.theme_population(ticker, day, version)
    refused = assess_population(first)
    if refused is not None:
        code, detail = refused
        return (
            _partition_record(
                ticker,
                day,
                version,
                PARTITION_POPULATION_REFUSED,
                reason=code,
                detail=detail,
            ),
            [],
        )
    binding, _ = classify_generation_binding(first)
    themes: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    changed = False
    for membership in sorted(first.themes, key=lambda t: (t.salience_rank, t.theme_id)):
        entry: dict[str, Any] = {
            "theme_id": membership.theme_id,
            "theme_key": _clean(membership.theme_key or ""),
            "salience_rank": membership.salience_rank,
            "outcome": "",
            "reason": "",
            "field": "",
            "artifact_id": None,
        }
        try:
            layout = build_generation_input(first, membership.theme_id)
        except SummaryInputError as exc:
            entry.update(outcome=THEME_INPUT_REFUSED, reason=exc.code)
            themes.append(entry)
            continue
        current = current_summary_artifact(
            reader, ticker, day, version, membership.theme_id, policy
        )
        if current is None:
            entry.update(outcome=THEME_NO_CURRENT)
        elif current.generation_input.input_fingerprint != layout.input_fingerprint:
            changed = True
            break
        else:
            withheld = _withheld_field(current)
            if withheld is not None:
                entry.update(
                    outcome=THEME_WITHHELD, reason=withheld[0], field=withheld[1]
                )
            else:
                entry.update(
                    outcome=THEME_CURRENT, artifact_id=current.artifact.artifact_id
                )
                artifacts.append(_artifact_record(current, membership.salience_rank))
        themes.append(entry)
    if not changed:
        last = reader.theme_population(ticker, day, version)
        changed = _population_state(last) != _population_state(first)
    if changed:
        return (
            _partition_record(
                ticker,
                day,
                version,
                PARTITION_POPULATION_CHANGED,
                reason=PARTITION_POPULATION_CHANGED,
                detail="the partition changed between the reads that enumerate "
                "its themes and the reads that reconstruct their artifacts; "
                "nothing from it is reviewed",
                generation_binding=binding,
            ),
            [],
        )
    return (
        _partition_record(
            ticker,
            day,
            version,
            PARTITION_ENUMERATED,
            generation_binding=binding,
            themes=themes,
        ),
        artifacts,
    )


@contextlib.contextmanager
def _database_read(path: Path) -> Iterator[None]:
    """Report a database that cannot be read as an input error, not a trace.

    A missing ``-wal``/``-shm`` in a directory the reader may not write, a
    file that is not SQLite, a locked or corrupt database: each is an
    operator-facing condition of *this* database, stated with its SQLite
    class and message.
    """

    try:
        yield
    except DATABASE_READ_ERRORS as exc:
        raise ReviewSamplingError(
            f"cannot read the Phase 0 database {_clean(path.name)}: "
            f"{type(exc).__name__}: {_clean(str(exc))}"
        ) from exc


def _default_policy() -> GenerationPolicy:
    # The scheduler's and the API's own resolver: one definition of the
    # production policy.  It constructs a lazy client and calls nothing.
    from phase0.summary_runner import production_generation_policy

    return production_generation_policy()


def load_sentence_population(
    database: str | Path,
    *,
    candidate_days: Sequence[str] = (),
    window: tuple[str, str] | None = None,
    pipeline_version: str | None = None,
    policy_resolver: Callable[[], GenerationPolicy] | None = None,
) -> SentencePopulation:
    """Read every candidate day's partitions and reviewable current artifacts.

    Read-only: a :class:`~phase0.repository.Phase0Reader` and nothing that
    writes, generates, or reaches a provider.  Every ticker of the Phase 0
    universe is accounted for on every candidate day.
    """

    path = Path(database)
    if not path.is_file():
        raise ReviewSamplingError(f"no Phase 0 database at {_clean(path.name)}")
    days = expand_candidate_days(candidate_days, window)
    if pipeline_version is not None:
        pipeline_version = _require_clean_identifier(
            pipeline_version, "pipeline_version"
        )
    try:
        policy = (policy_resolver or _default_policy)()
    except (ProviderConfigurationError, GuardedSummaryError) as exc:
        # Expected operator misconfiguration (a malformed GEMINI_* value);
        # the messages name the setting, never its value.
        raise ReviewSamplingError(
            f"the production summary policy cannot be resolved: {exc}"
        ) from exc
    if not isinstance(policy, GenerationPolicy):
        raise ReviewSamplingError("the policy resolver must return a GenerationPolicy")
    if contains_credential(policy.model):
        raise ReviewSamplingError(
            "the production policy's model carries credential-like text; the value "
            "is not shown"
        )
    with _database_read(path):
        reader = Phase0Reader(path)

        discovered: dict[str, dict[str, Any]] = {day: {} for day in days}
        versions: set[str] = set()
        rejected: list[tuple[str, str, str]] = []
        for day in days:
            for generation in reader.partition_generations(day):
                if generation.ticker not in TICKER_UNIVERSE:
                    continue
                if pipeline_version is not None:
                    if generation.pipeline_version == pipeline_version:
                        discovered[day][generation.ticker] = generation
                    continue
                if contains_credential(generation.pipeline_version):
                    rejected.append(
                        (generation.ticker, day, SKIP_PROVENANCE_CREDENTIAL)
                    )
                    continue
                if generation.pipeline_version.strip() == REJECTED_IDENTIFIER:
                    rejected.append((generation.ticker, day, SKIP_RESERVED_IDENTIFIER))
                    continue
                versions.add(generation.pipeline_version)
                discovered[day].setdefault(generation.ticker, generation)
        if pipeline_version is None:
            if len(versions) > 1:
                raise ReviewSamplingError(
                    "candidate days span pipeline versions "
                    f"{sorted(versions)}; name one with pipeline_version"
                )
            if not versions:
                raise ReviewSamplingError(
                    f"no persisted partitions on any candidate day {days}; nothing to "
                    "sample"
                )
            pipeline_version = next(iter(versions))

        rejected_at = {(t, d): reason for t, d, reason in rejected}
        partitions: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        for day in days:
            for ticker in TICKER_UNIVERSE:
                if (ticker, day) in rejected_at and ticker not in discovered[day]:
                    partitions.append(
                        _partition_record(
                            ticker,
                            day,
                            REJECTED_IDENTIFIER,
                            rejected_at[(ticker, day)],
                            reason=rejected_at[(ticker, day)],
                            detail="discovered pipeline_version is not a usable "
                            "identifier (value withheld)",
                        )
                    )
                    continue
                if ticker not in discovered[day]:
                    partitions.append(
                        _partition_record(
                            ticker,
                            day,
                            pipeline_version,
                            SKIP_NO_STORY_OUTPUT,
                            reason=SKIP_NO_STORY_OUTPUT,
                            detail="no stories and no theme set persisted for this "
                            "partition",
                        )
                    )
                    continue
                record, found = _read_partition(
                    reader, ticker, day, pipeline_version, policy
                )
                partitions.append(record)
                artifacts.extend(found)
    order = {ticker: n for n, ticker in enumerate(TICKER_UNIVERSE)}
    artifacts.sort(key=lambda a: _artifact_order(a, order))
    return SentencePopulation(
        candidate_input={
            "days": _normalized_days(candidate_days) if candidate_days else [],
            "window": None
            if window is None
            else {"start": window[0], "end": window[1]},
        },
        candidate_days=tuple(days),
        pipeline_version=pipeline_version,
        policy=_policy_record(policy),
        partitions=tuple(partitions),
        artifacts=tuple(artifacts),
        source={"mode": SOURCE_PERSISTED, "database": _clean(path.name)},
    )


def _artifact_order(artifact: Mapping[str, Any], order: Mapping[str, int]) -> tuple:
    return (
        artifact["trading_day"],
        order.get(artifact["ticker"], len(order)),
        artifact["salience_rank"],
        artifact["theme_id"],
    )


def day_accounting(
    candidate_days: Sequence[str], partitions: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per candidate day: reviewable artifacts, eligibility, and why not."""

    accounting = []
    for day in candidate_days:
        mine = [p for p in partitions if p["trading_day"] == day]
        reviewable = sum(len(p["reviewed_artifact_ids"]) for p in mine)
        if reviewable:
            reason = ""
        elif any(p["outcome"] == PARTITION_POPULATION_CHANGED for p in mine):
            reason = DAY_POPULATION_CHANGED
        elif any(p["withheld_artifact_count"] for p in mine):
            reason = DAY_NO_REVIEWABLE_SUMMARY
        elif all(p["outcome"] != PARTITION_ENUMERATED for p in mine) and not any(
            p["outcome"] == PARTITION_POPULATION_REFUSED for p in mine
        ):
            reason = DAY_NO_PARTITIONS
        else:
            reason = DAY_NO_CURRENT_SUMMARY
        accounting.append(
            {
                "trading_day": day,
                "reviewable_artifacts": reviewable,
                "eligible": bool(reviewable),
                "reason": reason,
            }
        )
    return accounting


# -- Sampling ------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceSample:
    population: SentencePopulation
    seed: str
    draw_size: int
    #: Whether the caller asked for a development draw -- recorded as a fact
    #: of its own, never inferred from the size: a development draw of two
    #: days is still a development draw.
    development_override: bool
    round_id: str
    selected_days: tuple[str, ...]
    artifacts: tuple[Mapping[str, Any], ...]
    rows: tuple[SentenceRow, ...]


def sample_sentences(
    population: SentencePopulation,
    *,
    seed: str,
    draw_size: int | None = None,
    round_id: str = "g2",
) -> SentenceSample:
    """Draw the review days from the eligible ones and take every sentence on them.

    ``draw_size`` is a development override: supplying it at all -- even as
    :data:`RELEASE_G2_REQUIRED_DAYS` -- records a development draw, and the
    scorecard is not eligible.  The seed and round id are checked for
    credential-like text before the draw and before anything is recorded.
    """

    development_override = draw_size is not None
    size = RELEASE_G2_REQUIRED_DAYS if draw_size is None else draw_size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ReviewSamplingError("the draw size must be a positive integer")
    seed = require_clean_operator_value(seed, "seed")
    if not seed:
        raise ReviewSamplingError("a seed is required so the draw is reproducible")
    round_id = require_clean_operator_value(round_id, "round_id")
    if not round_id:
        raise ReviewSamplingError("a round id is required")
    selected = draw_days(population.eligible_days, seed=seed, size=size)
    artifacts = tuple(
        a for a in population.artifacts if a["trading_day"] in set(selected)
    )
    rows = tuple(
        row
        for artifact in artifacts
        for row in rows_for_artifact(
            artifact, development_override=development_override
        )
    )
    return SentenceSample(
        population=population,
        seed=seed,
        draw_size=size,
        development_override=development_override,
        round_id=round_id,
        selected_days=tuple(selected),
        artifacts=artifacts,
        rows=rows,
    )


# -- Manifest ------------------------------------------------------------------


def _selection_block(
    population: SentencePopulation,
    *,
    seed: str,
    draw_size: int,
    development_override: bool,
    selected: Sequence[str],
) -> dict[str, Any]:
    accounting = population.day_accounting()
    block = {
        "candidate_input": dict(population.candidate_input),
        "candidate_days": list(population.candidate_days),
        "pipeline_version": population.pipeline_version,
        "tickers": list(TICKER_UNIVERSE),
        "source_mode": population.source.get("mode"),
        "policy_fingerprint": population.policy["fingerprint"],
        "day_accounting": accounting,
        "eligible_days": [d["trading_day"] for d in accounting if d["eligible"]],
        "excluded_days": [
            {"trading_day": d["trading_day"], "reason": d["reason"]}
            for d in accounting
            if not d["eligible"]
        ],
        "draw": {
            "method": DRAW_METHOD,
            "seed": seed,
            "size": draw_size,
            "required_days": RELEASE_G2_REQUIRED_DAYS,
            "development_override": development_override,
        },
        "selected_days": list(selected),
    }
    block["digest"] = selection_digest(block)
    return block


def selected_day_counts(
    partitions: Sequence[Mapping[str, Any]],
    selected_days: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """The selected days' totals, derived from the partition records below them.

    Written into the manifest for a reader's convenience and recomputed on
    every read: the detailed theme outcomes are the source of truth, and a
    total that disagrees with them is refused, never believed.
    """

    selected = set(selected_days)
    on_selected = [p for p in partitions if p["trading_day"] in selected]
    return {
        "artifact_count": len(artifacts),
        "sentence_count": len(rows),
        "degraded_theme_count": sum(p["degraded_theme_count"] for p in on_selected),
        "withheld_artifact_count": sum(
            p["withheld_artifact_count"] for p in on_selected
        ),
        "population_changed_partition_count": sum(
            1 for p in on_selected if p["outcome"] == PARTITION_POPULATION_CHANGED
        ),
    }


def selection_digest(selection: Mapping[str, Any]) -> str:
    """SHA-256 over every selection field except the digest itself."""

    return sha256_of({k: v for k, v in selection.items() if k != "digest"})


def population_digest(
    partitions: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]]
) -> str:
    return sha256_of({"partitions": list(partitions), "artifacts": list(artifacts)})


def snapshot_digest(
    rows: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]]
) -> str:
    return sha256_of({"rows": list(rows), "artifacts": list(artifacts)})


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


def render_csv(rows: Sequence[SentenceRow], binding: Mapping[str, str]) -> str:
    if set(binding) != set(BINDING_FIELDS):
        raise ReviewSamplingError(f"sheet binding must supply {BINDING_FIELDS}")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SENTENCE_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({**dataclasses.asdict(row), **binding})
    return buffer.getvalue()


def build_manifest(
    sample: SentenceSample,
    *,
    protocol_id: str = UNRATIFIED_PROTOCOL,
    csv_name: str,
    generated_at: datetime | None = None,
    code: Mapping[str, Any] | None = None,
    operator_attestation: OperatorAttestation | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce the draw and to hold the review to it."""

    population = sample.population
    protocol = require_known_g2_protocol(protocol_id)
    csv_name = require_clean_operator_value(csv_name, "output file name")
    origin, detail = classify_origin(population.source)
    when = generated_at or datetime.now(timezone.utc)
    partitions = [dict(p) for p in population.partitions]
    artifacts = [dict(a) for a in sample.artifacts]
    snapshot_rows = [row.snapshot() for row in sample.rows]
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
        "claim": CLAIM,
        "policy": dict(population.policy),
        "selection": _selection_block(
            population,
            seed=sample.seed,
            draw_size=sample.draw_size,
            development_override=sample.development_override,
            selected=sample.selected_days,
        ),
        "population": {
            "partitions": partitions,
            "digest": population_digest(partitions, artifacts),
            "selected_days": selected_day_counts(
                partitions, sample.selected_days, artifacts, snapshot_rows
            ),
        },
        "sample": {
            "method": CENSUS_METHOD,
            "round_id": sample.round_id,
            "requested_size": len(sample.rows),
            "actual_size": len(sample.rows),
            "row_ids": [row.row_id for row in sample.rows],
            "prior_rounds": [],
        },
        "snapshot": {
            "rows": snapshot_rows,
            "artifacts": artifacts,
            "sha256": snapshot_digest(snapshot_rows, artifacts),
        },
        "gate_requirements": {
            "threshold": RELEASE_G2_THRESHOLD,
            "required_days": RELEASE_G2_REQUIRED_DAYS,
            "note": "informational; scoring uses the constants in code",
        },
        "labeling_protocol": {
            "id": protocol.id,
            "vocabulary_at_sampling": {
                "positive": protocol.positive_verdict,
                "negative": protocol.negative_verdict,
            },
            "note": (
                "ratification is decided by nlp.eval.faithfulness."
                "RATIFIED_G2_PROTOCOLS at scoring time, never by this file"
            ),
        },
    }
    binding = {
        "manifest_id": manifest_identity(manifest),
        "snapshot_sha256": manifest["snapshot"]["sha256"],
    }
    manifest["binding"] = binding
    manifest["sheet"] = {
        "csv": csv_name,
        "blank_sha256": _sha256_text(render_csv(sample.rows, binding)),
        "columns": list(SENTENCE_FIELDNAMES),
        "identity_columns": list(IDENTITY_FIELDS),
        "context_columns": list(CONTEXT_FIELDS),
        "binding_columns": list(BINDING_FIELDS),
        "reviewer_columns": list(REVIEWER_FIELDS),
    }
    return manifest


def manifest_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + ".manifest.json")


def protected_database_paths(database: str | Path) -> list[Path]:
    """The database and every SQLite companion file an output must never be.

    Companions are named after the file SQLite actually opens, so they are
    derived from the supplied path *and* from its resolved target: with
    ``alias.db -> real.db``, both ``alias.db-journal`` and
    ``real.db-journal`` are protected.
    """

    supplied = Path(database)
    identities: list[Path] = []
    for base in (supplied, supplied.resolve()):
        if base not in identities:
            identities.append(base)
    return [
        base.with_name(base.name + suffix)
        for base in identities
        for suffix in ("", "-wal", "-shm", "-journal")
    ]


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def check_output_paths(
    outputs: Sequence[str | Path], *, inputs: Sequence[str | Path] = ()
) -> None:
    """Refuse, before anything is opened for writing, an output that could destroy.

    Every output must be new -- nothing, not even a dangling symlink, may
    already sit at its path -- distinct from every other output, and none
    of the inputs, compared after resolving ``..``, relative paths and
    symlinks (and by inode where both exist).  There is no overwrite mode.
    """

    candidates = [Path(o) for o in outputs]
    for index, output in enumerate(candidates):
        for source in (Path(i) for i in inputs):
            if _same_file(output, source):
                raise ReviewSamplingError(
                    f"output {_clean(output.name)} is an input "
                    f"({_clean(source.name)}); refusing to write over it"
                )
        for other in candidates[index + 1 :]:
            if _same_file(output, other):
                raise ReviewSamplingError(
                    f"outputs {_clean(output.name)} and {_clean(other.name)} are "
                    "the same file"
                )
        if os.path.lexists(output):
            raise ReviewSamplingError(
                f"output {_clean(output.name)} already exists; review artifacts are "
                "never overwritten -- choose a new path"
            )


def create_new_file(path: str | Path, text: str) -> None:
    """Create ``path`` exclusively (``O_EXCL``) holding ``text``, wholly or not at all.

    A file that already exists is refused and left exactly as it was.  Once
    this call has created the file, any failure before it is complete --
    in the write, the flush, the sync or the close -- removes it, so a full
    disk never leaves a truncated review artifact behind.  Only the file
    this call created is ever removed.
    """

    location = Path(path)
    try:
        handle = location.open("x", encoding="utf-8", newline="")
    except FileExistsError as exc:
        raise ReviewSamplingError(
            f"output {_clean(location.name)} already exists; review artifacts are "
            "never overwritten"
        ) from exc
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as exc:
        location.unlink(missing_ok=True)
        if isinstance(exc, OSError):
            raise ReviewSamplingError(
                f"cannot write output {_clean(location.name)}: "
                f"{_clean(exc.strerror or type(exc).__name__)}; nothing was left "
                "behind"
            ) from exc
        raise


def write_sample(
    sample: SentenceSample, csv_path: str | Path, *, manifest: Mapping[str, Any]
) -> tuple[Path, Path]:
    """Create the sheet and its manifest as a pair; neither may exist already.

    Both are created exclusively.  If either cannot be completed, every
    file this call created is removed, so a failure leaves no part of a
    pair; nothing that existed before is touched.
    """

    location = Path(csv_path)
    manifest_location = manifest_path_for(location)
    check_output_paths([location, manifest_location])
    location.parent.mkdir(parents=True, exist_ok=True)
    create_new_file(location, render_csv(sample.rows, manifest["binding"]))
    try:
        create_new_file(
            manifest_location, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except BaseException:
        location.unlink(missing_ok=True)
        raise
    return location, manifest_location


# -- Reading a manifest back ---------------------------------------------------


def _refuse(location: Path, message: str) -> ReviewSamplingError:
    return ReviewSamplingError(f"{location}: {message}")


def _verify_policy(payload: Mapping[str, Any], location: Path) -> None:
    policy = payload["policy"]
    try:
        expected = compute_policy_fingerprint(
            model=policy["model"],
            max_attempts=policy["max_attempts"],
            rules=tuple(tuple(rule) for rule in policy["rules"]),
            temperature=policy["temperature"],
            max_output_tokens=policy["max_output_tokens"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(location, f"policy is malformed: {exc}") from exc
    if policy.get("fingerprint") != expected:
        raise _refuse(location, "policy fingerprint does not match the recorded policy")


def _verify_artifact(
    artifact: Mapping[str, Any], policy: Mapping[str, Any], location: Path
) -> None:
    """Recompute the artifact's two digests from the snapshot alone."""

    where = f"artifact {artifact.get('artifact_id')}"
    try:
        sentences = artifact["sentences"]
        evidence = [
            EvidenceStory(
                citation_id=story["citation_id"],
                persisted_story_id=story["persisted_story_id"],
                title=story["title"],
                description=story["description"],
                outlet=story["outlet"],
                published_at=story["published_at"],
                raw_item_ids=tuple(story["raw_item_ids"]),
                urls=tuple(story["urls"]),
            )
            for story in artifact["evidence"]
        ]
        reference = ThemeReference(
            theme_id=artifact["theme_id"],
            theme_key=artifact["theme_key"],
            label=artifact["theme_label"],
            pipeline_version=artifact["pipeline_version"],
        )
        fingerprint = compute_input_fingerprint(
            artifact["ticker"], artifact["trading_day"], reference, evidence
        )
        digest = summary_artifact_digest(
            ticker=artifact["ticker"],
            trading_day=artifact["trading_day"],
            pipeline_version=artifact["pipeline_version"],
            theme_id=artifact["theme_id"],
            theme_key=artifact["theme_key"],
            input_fingerprint=artifact["input_fingerprint"],
            policy_fingerprint=artifact["policy_fingerprint"],
            citation_convention=artifact["citation_convention"],
            prompt_version=artifact["prompt_version"],
            model=artifact["model"],
            label=artifact["label"],
            guarantee=artifact["guarantee"],
            sentences=[
                (
                    s["ordinal"],
                    s["text"],
                    [(c["position"], c["story_id"]) for c in s["citations"]],
                )
                for s in sentences
            ],
        )
    except (KeyError, TypeError, ValueError, GuardedSummaryError) as exc:
        raise _refuse(location, f"{where} is malformed: {exc}") from exc
    if fingerprint != artifact["input_fingerprint"]:
        raise _refuse(
            location,
            f"{where}: its frozen evidence does not reproduce its input fingerprint; "
            "the evidence was altered",
        )
    if digest != artifact["content_digest"]:
        raise _refuse(
            location,
            f"{where}: its sentences and citations do not reproduce its content "
            "digest; the summary was altered",
        )
    if (
        artifact["policy_fingerprint"] != policy["fingerprint"]
        or artifact["model"] != policy["model"]
    ):
        raise _refuse(location, f"{where} was not current under the recorded policy")
    ids = {story.citation_id for story in evidence}
    if [s["ordinal"] for s in sentences] != list(range(1, len(sentences) + 1)):
        raise _refuse(location, f"{where}: sentence ordinals are not 1..N")
    for sentence in sentences:
        positions = [c["position"] for c in sentence["citations"]]
        stories = [c["story_id"] for c in sentence["citations"]]
        if not positions or positions != list(range(len(positions))):
            raise _refuse(location, f"{where}: citation positions are not 0..M-1")
        if len(stories) != len(set(stories)) or any(
            citation_id_for(int(s)) not in ids for s in stories
        ):
            raise _refuse(
                location, f"{where}: a citation is duplicated or not in its evidence"
            )


def _verify_selection(payload: Mapping[str, Any], location: Path) -> None:
    selection = payload["selection"]
    partitions = payload["population"]["partitions"]
    try:
        if selection_digest(selection) != selection["digest"]:
            raise _refuse(location, "selection digest does not match; it was altered")
        candidate = selection["candidate_input"]
        window = candidate.get("window")
        expected_days = expand_candidate_days(
            candidate.get("days") or (),
            None if window is None else (window["start"], window["end"]),
        )
        if selection["candidate_days"] != expected_days:
            raise _refuse(location, "candidate days do not follow from their input")
        if selection["tickers"] != list(TICKER_UNIVERSE):
            raise _refuse(location, "the ticker scope is not the Phase 0 universe")
        if selection["source_mode"] != payload["source"].get("mode"):
            raise _refuse(location, "selection and source disagree on the source mode")
        if selection["policy_fingerprint"] != payload["policy"]["fingerprint"]:
            raise _refuse(location, "selection and policy disagree")
        accounting = day_accounting(expected_days, partitions)
        if selection["day_accounting"] != accounting:
            raise _refuse(
                location, "day accounting does not follow from the partitions"
            )
        eligible = [d["trading_day"] for d in accounting if d["eligible"]]
        if selection["eligible_days"] != eligible:
            raise _refuse(location, "eligible days do not follow from the accounting")
        excluded = [
            {"trading_day": d["trading_day"], "reason": d["reason"]}
            for d in accounting
            if not d["eligible"]
        ]
        if selection["excluded_days"] != excluded:
            raise _refuse(location, "excluded days do not follow from the accounting")
        draw = selection["draw"]
        if draw["method"] != DRAW_METHOD or draw["required_days"] != (
            RELEASE_G2_REQUIRED_DAYS
        ):
            raise _refuse(location, "the draw is not this module's draw")
        override = draw["development_override"]
        if not isinstance(override, bool):
            raise _refuse(location, "draw.development_override must be true or false")
        if not override and draw["size"] != RELEASE_G2_REQUIRED_DAYS:
            raise _refuse(
                location,
                "a draw of other than two days must record its development override",
            )
        if selection["selected_days"] != draw_days(
            eligible, seed=draw["seed"], size=draw["size"]
        ):
            raise _refuse(
                location, "selected days are not the seeded draw from the eligible days"
            )
    except (KeyError, TypeError) as exc:
        raise _refuse(location, f"selection is malformed: {exc}") from exc
    expected_partitions = [
        (day, ticker) for day in expected_days for ticker in TICKER_UNIVERSE
    ]
    found = [(p["trading_day"], p["ticker"]) for p in partitions]
    if found != expected_partitions:
        raise _refuse(
            location, "the partitions are not every candidate day x every ticker, once"
        )
    for partition in partitions:
        if partition["pipeline_version"] not in (
            selection["pipeline_version"],
            REJECTED_IDENTIFIER,
        ):
            raise _refuse(location, "a partition lies outside the pipeline version")


PARTITION_OUTCOMES = frozenset(
    {
        PARTITION_ENUMERATED,
        PARTITION_POPULATION_REFUSED,
        PARTITION_POPULATION_CHANGED,
        SKIP_NO_STORY_OUTPUT,
        SKIP_PROVENANCE_CREDENTIAL,
        SKIP_RESERVED_IDENTIFIER,
    }
)
WITHHELD_REASONS = frozenset(
    {WITHHELD_SENTENCE, WITHHELD_LABEL, WITHHELD_EVIDENCE, WITHHELD_IDENTIFIER}
)
_THEME_KEYS = frozenset(
    {
        "theme_id",
        "theme_key",
        "salience_rank",
        "outcome",
        "reason",
        "field",
        "artifact_id",
    }
)


def _verify_theme_outcome(theme: Any, where: str, location: Path) -> None:
    """One theme outcome is internally consistent; it is the source of truth."""

    if not isinstance(theme, dict) or set(theme) != _THEME_KEYS:
        raise _refuse(location, f"{where}: a theme outcome is malformed")
    outcome, reason, field = theme["outcome"], theme["reason"], theme["field"]
    artifact_id = theme["artifact_id"]
    current = outcome == THEME_CURRENT
    if current != (isinstance(artifact_id, int) and not isinstance(artifact_id, bool)):
        raise _refuse(
            location, f"{where}: only a current summary names an artifact, and it must"
        )
    if not current and artifact_id is not None:
        raise _refuse(location, f"{where}: a non-current theme names an artifact")
    valid = (
        (outcome in (THEME_CURRENT, THEME_NO_CURRENT) and reason == "" and field == "")
        or (outcome == THEME_INPUT_REFUSED and bool(reason) and field == "")
        or (outcome == THEME_WITHHELD and reason in WITHHELD_REASONS and bool(field))
    )
    if not valid:
        raise _refuse(location, f"{where}: theme outcome {outcome!r} is inconsistent")


def _verify_partitions(payload: Mapping[str, Any], location: Path) -> None:
    """Every partition total is re-derived from its theme outcomes.

    ``theme_count``, ``degraded_theme_count``, ``withheld_artifact_count``
    and ``reviewed_artifact_ids`` are what the day accounting, the census,
    and the scorecard's completeness are built from.  They are recomputed
    here from the per-theme records, bottom-up, and a stored value that
    differs is refused; so are selected-day totals that differ from the
    partitions.
    """

    partitions = payload["population"].get("partitions")
    if not isinstance(partitions, list):
        raise _refuse(location, "population.partitions is missing")
    for partition in partitions:
        if not isinstance(partition, dict):
            raise _refuse(location, "a partition record is malformed")
        where = f"partition {partition.get('ticker')} {partition.get('trading_day')}"
        try:
            outcome = partition["outcome"]
            themes = partition["themes"]
            if outcome not in PARTITION_OUTCOMES or not isinstance(themes, list):
                raise _refuse(location, f"{where}: unknown outcome or themes")
            if outcome != PARTITION_ENUMERATED and themes:
                raise _refuse(
                    location, f"{where}: only an enumerated partition has themes"
                )
            for theme in themes:
                _verify_theme_outcome(theme, where, location)
            expected = _partition_record(
                partition["ticker"],
                partition["trading_day"],
                partition["pipeline_version"],
                outcome,
                reason=partition["reason"],
                generation_binding=partition["generation_binding"],
                themes=themes,
            )
            expected["detail"] = partition["detail"]
        except (KeyError, TypeError) as exc:
            raise _refuse(location, f"{where} is malformed: {exc}") from exc
        if partition != expected:
            raise _refuse(
                location,
                f"{where}: its totals do not follow from its theme outcomes; the "
                "accounting was altered",
            )


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a G2 manifest and re-verify every binding before anything trusts it."""

    location = Path(path)
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refuse(location, f"cannot read manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        found = payload.get("schema") if isinstance(payload, dict) else None
        raise _refuse(location, f"not a {MANIFEST_SCHEMA} manifest (schema={found!r})")
    if payload.get("gate") != GATE or payload.get("sheet_kind") != SHEET_KIND:
        raise _refuse(
            location,
            f"manifest is for gate {payload.get('gate')!r} / sheet kind "
            f"{payload.get('sheet_kind')!r}, not {GATE} / {SHEET_KIND}",
        )
    for key in (
        "source",
        "origin",
        "policy",
        "selection",
        "population",
        "sample",
        "snapshot",
        "labeling_protocol",
        "binding",
        "sheet",
    ):
        if key not in payload:
            raise _refuse(location, f"manifest is missing {key!r}")
    classify_origin(payload["source"])
    _verify_policy(payload, location)
    _verify_partitions(payload, location)
    _verify_selection(payload, location)

    snapshot = payload["snapshot"]
    artifacts = snapshot.get("artifacts", [])
    selected = set(payload["selection"]["selected_days"])
    partitions = payload["population"]["partitions"]
    reviewed = sorted(
        artifact_id
        for p in partitions
        if p["trading_day"] in selected
        for artifact_id in p["reviewed_artifact_ids"]
    )
    if sorted(a.get("artifact_id") for a in artifacts) != reviewed:
        raise _refuse(
            location,
            "the snapshot's artifacts are not exactly the reviewed artifacts of the "
            "selected days",
        )
    order = {ticker: n for n, ticker in enumerate(TICKER_UNIVERSE)}
    if artifacts != sorted(artifacts, key=lambda a: _artifact_order(a, order)):
        raise _refuse(location, "the snapshot's artifacts are not in canonical order")
    for artifact in artifacts:
        partition = next(
            (
                p
                for p in partitions
                if p["trading_day"] == artifact.get("trading_day")
                and p["ticker"] == artifact.get("ticker")
            ),
            None,
        )
        if (
            partition is None
            or artifact.get("artifact_id") not in partition["reviewed_artifact_ids"]
            or artifact.get("pipeline_version") != partition["pipeline_version"]
            or not any(
                t["artifact_id"] == artifact.get("artifact_id")
                and t["theme_id"] == artifact.get("theme_id")
                and t["salience_rank"] == artifact.get("salience_rank")
                for t in partition["themes"]
            )
        ):
            raise _refuse(
                location,
                f"artifact {artifact.get('artifact_id')} does not belong to its "
                "partition",
            )
        _verify_artifact(artifact, payload["policy"], location)
    if payload["population"].get("digest") != population_digest(partitions, artifacts):
        raise _refuse(location, "population digest does not match; it was altered")

    rows = snapshot.get("rows", [])
    override = payload["selection"]["draw"]["development_override"]
    expected_rows = [
        row.snapshot()
        for a in artifacts
        for row in rows_for_artifact(a, development_override=override)
    ]
    if rows != expected_rows:
        raise _refuse(
            location,
            "the snapshot's rows are not the exact sentence census of its artifacts; "
            "a row was added, removed, reordered, or altered",
        )
    ids = [row["row_id"] for row in rows]
    sample = payload["sample"]
    if sample.get("row_ids") != ids or len(set(ids)) != len(ids):
        raise _refuse(location, "sample row ids and snapshot rows disagree")
    if sample.get("method") != CENSUS_METHOD or sample.get("prior_rounds") != []:
        raise _refuse(location, "a G2 round is one census, never a linked draw")
    if sample.get("requested_size") != len(ids) or sample.get("actual_size") != len(
        ids
    ):
        raise _refuse(location, "sample sizes disagree with the census")
    counts = selected_day_counts(
        partitions, payload["selection"]["selected_days"], artifacts, rows
    )
    if payload["population"].get("selected_days") != counts:
        raise _refuse(
            location,
            "selected-day totals do not follow from the partitions and the census; "
            "the accounting was altered",
        )
    if snapshot.get("sha256") != snapshot_digest(rows, artifacts):
        raise _refuse(location, "snapshot digest does not match; it was altered")
    binding = payload["binding"]
    if binding.get("snapshot_sha256") != snapshot["sha256"] or binding.get(
        "manifest_id"
    ) != manifest_identity(payload):
        raise _refuse(
            location,
            "binding does not match the manifest's own identity; the manifest was "
            "altered after its sheet was cut",
        )
    payload["_sha256"] = _sha256_file(location)
    payload["_path"] = str(location)
    return payload


# -- Scoring -------------------------------------------------------------------


def score_sentence_round(
    manifest: Mapping[str, Any],
    sheets: Sequence[str | Path],
    *,
    adjudicated: str | Path | None = None,
) -> RoundResult:
    """Resolve one G2 round with A4a's reviewer and adjudication rules."""

    if "_sha256" not in manifest or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ReviewSamplingError(
            "a G2 round must be scored from a manifest read by "
            "nlp.eval.faithfulness.read_manifest"
        )
    return score_round(manifest, sheets, adjudicated=adjudicated, spec=G2_SHEET)


@dataclass(frozen=True)
class G2DevelopmentOverrides:
    """A looser threshold for development review; forces NOT_ELIGIBLE."""

    threshold: float | None = None

    def __post_init__(self) -> None:
        value = self.threshold
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReviewSamplingError("threshold must be a number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ReviewSamplingError("threshold must be finite and within [0, 1]")

    @property
    def active(self) -> bool:
        return self.threshold is not None


def faithfulness_rate(
    positive: int, resolved: int, threshold: float
) -> tuple[float | None, bool | None]:
    """The raw rate and whether it meets ``threshold``, compared exactly.

    ``None`` for both when nothing resolved.  The comparison is on
    fractions, so 19/20 meets 0.95 however floating point rounds it.
    """

    if resolved <= 0:
        return None, None
    exact = Fraction(positive, resolved)
    return float(exact), exact >= Fraction(str(threshold))


@dataclass(frozen=True)
class G2Scorecard:
    """Gate G2 read off one census round, with the four facts kept apart."""

    evaluation_mode: str
    threshold: float
    rate: float | None
    threshold_met: bool | None
    review_complete: bool
    gate_eligible: bool
    gate_result: GateResult
    origin_status: OriginStatus
    origin_detail: str
    protocol_id: str
    protocol_ratified: bool
    reviewer_count: int
    adjudication_state: AdjudicationState
    selected_days: tuple[str, ...]
    required_days: int
    sentence_count: int
    resolved_count: int
    positive_count: int
    unresolved_count: int
    unresolved_row_ids: tuple[str, ...]
    agreement_rate: float | None
    reviewer_ids: tuple[str, ...]
    adjudicator_ids: tuple[str, ...]
    eligibility_blockers: tuple[str, ...]
    incompleteness: tuple[str, ...]
    round: Mapping[str, Any]
    gate: str = GATE

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
            "protocol": {"id": self.protocol_id, "ratified": self.protocol_ratified},
            "reviewer_count": self.reviewer_count,
            "adjudication_state": self.adjudication_state.value,
            "selected_days": list(self.selected_days),
            "required_days": self.required_days,
            "sentence_count": self.sentence_count,
            "resolved_count": self.resolved_count,
            "positive_count": self.positive_count,
            "unresolved_count": self.unresolved_count,
            "unresolved_row_ids": list(self.unresolved_row_ids),
            "agreement_rate": self.agreement_rate,
            "reviewer_ids": list(self.reviewer_ids),
            "adjudicator_ids": list(self.adjudicator_ids),
            "eligibility_blockers": list(self.eligibility_blockers),
            "incompleteness": list(self.incompleteness),
            "round": dict(self.round),
            "claim": CLAIM,
            "note": (
                "rate is a measurement; only gate_result is a gate verdict. The "
                "Phase 0 GO / NO-GO decision combines G1-G7 and Q1-Q3 and is K4's."
            ),
        }


def _reverify_round(result: RoundResult) -> RoundResult:
    """Re-derive a G2 round from its manifest and sheets; refuse any other report."""

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
    fresh = score_sentence_round(disk, sheets, adjudicated=adjudication)
    if fresh.as_dict() != result.as_dict():
        raise ReviewSamplingError(
            f"round {result.round_id!r}: the report does not match its source "
            "artifacts (manifest, sheets, adjudication)"
        )
    return fresh


def score_g2(
    result: RoundResult, *, development: G2DevelopmentOverrides | None = None
) -> G2Scorecard:
    """The G2 scorecard from exactly one census round.

    Origin from the manifest's source mode, ratification from
    :data:`RATIFIED_G2_PROTOCOLS`, the threshold and day count from the
    constants, reviewer and adjudication facts from the parsed sheets.
    """

    if not isinstance(result, RoundResult):
        raise ReviewSamplingError("the round must come from score_sentence_round")
    result = _reverify_round(result)
    manifest = result.manifest
    development = development or G2DevelopmentOverrides()
    threshold = (
        RELEASE_G2_THRESHOLD if development.threshold is None else development.threshold
    )
    selected = tuple(manifest["selection"]["selected_days"])
    # Verified on read: false only for a two-day draw made without an override.
    development_draw = manifest["selection"]["draw"]["development_override"]
    mode = "development" if development.active or development_draw else "release"

    origin, origin_detail = classify_origin(manifest["source"])
    protocol, ratified = resolve_g2_protocol(manifest["labeling_protocol"].get("id"))
    outcomes = list(result.outcomes)
    resolved = [o for o in outcomes if o.resolved is not None]
    unresolved = sorted(o.row_id for o in outcomes if o.resolved is None)
    positives = sum(1 for o in resolved if o.resolved)
    rate, threshold_met = faithfulness_rate(positives, len(resolved), threshold)

    incompleteness: list[str] = []
    if unresolved:
        incompleteness.append(f"{len(unresolved)} sentences are unresolved")
    counts = manifest["population"]["selected_days"]
    if counts["population_changed_partition_count"]:
        incompleteness.append(
            f"{counts['population_changed_partition_count']} partition(s) on the "
            "selected days changed while sampling and were not reviewed; the census "
            "is incomplete"
        )
    if counts["withheld_artifact_count"]:
        incompleteness.append(
            f"{counts['withheld_artifact_count']} current artifact(s) on the selected "
            "days were withheld for credential-like text; the census is incomplete"
        )
    if not outcomes:
        incompleteness.append("the census holds no sentences")
    review_complete = not incompleteness

    adjudicated = ratified and result.adjudication_state in protocol.adjudicated_states
    blockers: list[str] = []
    if origin is not OriginStatus.VERIFIED_LIVE:
        blockers.append(f"origin is {origin.value}: {origin_detail}")
    bindings = sorted(
        {
            str(p.get("generation_binding"))
            for p in manifest["population"]["partitions"]
            if p["trading_day"] in set(selected) and p["reviewed_artifact_ids"]
        }
    )
    if bindings != [GENERATION_BINDING_VERIFIED]:
        blockers.append(
            f"theme-set build provenance is {bindings}: nothing persisted records the "
            "story generation a theme set was built over"
        )
    if not ratified:
        blockers.append(
            f"labeling protocol {manifest['labeling_protocol'].get('id')!r} is not in "
            "the ratified G2 registry (nlp.eval.faithfulness.RATIFIED_G2_PROTOCOLS)"
        )
    if result.reviewer_count < 2:
        blockers.append("fewer than two reviewers; section 8 requires two, adjudicated")
    elif not adjudicated:
        blockers.append(
            f"adjudication state {result.adjudication_state.value!r} is not one the "
            "protocol counts as adjudicated"
        )
    if development_draw:
        blockers.append(
            f"a development draw of {len(selected)} day(s) was requested; section 8 "
            f"requires exactly {RELEASE_G2_REQUIRED_DAYS}, drawn without an override"
        )
    if development.active:
        blockers.append(
            f"development threshold {threshold} in effect; release requirements are "
            "fixed by the spec"
        )
    gate_eligible = not blockers

    return G2Scorecard(
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
        protocol_id=protocol.id,
        protocol_ratified=ratified,
        reviewer_count=result.reviewer_count,
        adjudication_state=result.adjudication_state,
        selected_days=selected,
        required_days=RELEASE_G2_REQUIRED_DAYS,
        sentence_count=len(outcomes),
        resolved_count=len(resolved),
        positive_count=positives,
        unresolved_count=len(unresolved),
        unresolved_row_ids=tuple(unresolved),
        agreement_rate=result.agreement_rate,
        reviewer_ids=result.reviewer_ids,
        adjudicator_ids=result.adjudicator_ids,
        eligibility_blockers=tuple(blockers),
        incompleteness=tuple(incompleteness),
        round={
            "round_id": result.round_id,
            "manifest_path": manifest.get("_path"),
            "manifest_sha256": result.manifest_sha256,
            "snapshot_sha256": manifest["snapshot"]["sha256"],
            "population_digest": manifest["population"]["digest"],
            "selection_digest": manifest["selection"]["digest"],
            "sheets": [dict(s) for s in result.sheets],
        },
    )


def render_scorecard(scorecard: G2Scorecard) -> str:
    """The measured rate and the gate verdict, on separate lines, never merged."""

    met = scorecard.threshold_met
    rate = "n/a" if scorecard.rate is None else f"{scorecard.rate:.4f}"
    lines = [
        f"gate               {scorecard.gate} ({scorecard.evaluation_mode})",
        f"gate_result        {scorecard.gate_result.value}",
        f"gate_eligible      {str(scorecard.gate_eligible).lower()}",
        f"review_complete    {str(scorecard.review_complete).lower()}",
        f"measured rate      {rate}  (threshold {scorecard.threshold}; "
        "a measurement, not a verdict)",
        f"threshold_met      {'n/a' if met is None else str(met).lower()}",
        f"origin             {scorecard.origin_status.value}",
        f"protocol           {scorecard.protocol_id} "
        f"({'ratified' if scorecard.protocol_ratified else 'unratified'})",
        f"days               {', '.join(scorecard.selected_days)} "
        f"(required {scorecard.required_days})",
        f"sentences          {scorecard.sentence_count}; "
        f"{scorecard.resolved_count} resolved, {scorecard.unresolved_count} unresolved",
    ]
    if scorecard.agreement_rate is not None:
        lines.append(f"agreement_rate     {scorecard.agreement_rate:.4f}")
    lines.append(f"reviewers          {', '.join(scorecard.reviewer_ids) or '-'}")
    lines.append(f"adjudication       {scorecard.adjudication_state.value}")
    lines.append(
        f"manifest           {scorecard.round['manifest_sha256'][:12]} "
        f"snapshot {scorecard.round['snapshot_sha256'][:12]}"
    )
    if scorecard.eligibility_blockers:
        lines.append("not gate eligible because:")
        lines.extend(f"  - {b}" for b in scorecard.eligibility_blockers)
    if scorecard.incompleteness:
        lines.append("review incomplete because:")
        lines.extend(f"  - {i}" for i in scorecard.incompleteness)
    lines.append("")
    lines.append(f"Scope: {CLAIM}.")
    lines.append(
        "A4 reports one gate; the Phase 0 GO / NO-GO decision is K4's, not this tool's."
    )
    return "\n".join(lines)


__all__ = [
    "CLAIM",
    "CONTEXT_FIELDS",
    "G2_SHEET",
    "GATE",
    "IDENTITY_FIELDS",
    "MANIFEST_SCHEMA",
    "PROVISIONAL_G2_PROTOCOL",
    "RATIFIED_G2_PROTOCOLS",
    "RELEASE_G2_REQUIRED_DAYS",
    "RELEASE_G2_THRESHOLD",
    "SENTENCE_FIELDNAMES",
    "SHEET_KIND",
    "SNAPSHOT_FIELDS",
    "G2DevelopmentOverrides",
    "G2Scorecard",
    "SentencePopulation",
    "SentenceRow",
    "SentenceSample",
    "build_manifest",
    "check_output_paths",
    "create_new_file",
    "day_accounting",
    "draw_days",
    "expand_candidate_days",
    "faithfulness_rate",
    "load_sentence_population",
    "manifest_path_for",
    "protected_database_paths",
    "read_manifest",
    "render_cited_evidence",
    "render_scorecard",
    "require_clean_operator_value",
    "require_known_g2_protocol",
    "resolve_g2_protocol",
    "row_id_for",
    "rows_for_artifact",
    "sample_sentences",
    "score_g2",
    "score_sentence_round",
    "selected_day_counts",
    "write_sample",
]

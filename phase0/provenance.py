"""A4c: what makes persisted production provenance *verified*, in one place.

Migration 017 links each row a reviewed summary depends on to the logged
stage run that wrote it: a raw item to the ingestion run that inserted it,
a story to the story run that last wrote its content, a theme set to the
theme run that last wrote the partition's theme output and to the
story-generation signature that run verified before writing.  A summary
artifact was already linked, through its one accepted
``summary_generations`` row, to the summaries run that produced it.

This module decides whether those links hold.  It reads no database: every
function here takes the plain facts a repository read returned (or that a
review manifest recorded) and says, deterministically, whether each hop
verifies and why not.  The same functions run when a manifest is written
and again, offline, when it is scored, so a verdict is never stored and
trusted -- it is always re-derived from the facts beside it.

**The claim, exactly.**  A verified hop means: the row names a ``run_log``
row of the expected stage, for the expected partition, that recorded a
repository mutation (a ``last_mutation_id``); for ingestion, one that was
not a replay; and the derived content still matches what the binding was
made over.  It is **relational** provenance through repository-controlled
logged writes and existing deterministic content digests.

**Who can set a binding.**  Only the repository's logged mutation paths:
migration 017's triggers refuse any other write that sets one, however
valid the run id it names (see ``phase0.repository._trusted_provenance_write``).
Ordinary SQL can clear a binding, never create or restore one.

**Not claimed.**  That a network fetch happened; that SQLite was not edited
by someone who drops the triggers or registers an impostor authorization
function (such a writer can forge ``run_log`` rows and recompute every
digest); cryptographic authenticity; host, build, or CI identity.  Nothing
here is a signature or an attestation.

**Content identity is not production origin.**  A matching
``input_fingerprint`` proves an artifact was generated from input equal to
what the database now holds.  It says nothing about who wrote that input;
these checks do.

**Pipeline health is not provenance.**  A run's final ``status`` is
recorded but never required: a run can admit evidence and later settle
``degraded`` or ``failed`` because a different operation failed, and the
admitted rows are still exactly what that run wrote.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

#: The one version of the story-generation signature format this code
#: understands (``Phase0Repository._story_generation_signature``).  It is
#: persisted beside every theme-set binding.  A binding under any other
#: version -- older or newer -- is *unverified*, never reinterpreted: a
#: signature is only comparable with one the same function produced.
#: Bump it, in a reviewed change, whenever that function's encoding moves;
#: ``tests/test_review_provenance_binding.py`` holds a golden vector for it.
#: The column is an exact SQLite INTEGER (migration 017), and readers pass
#: it through uncoerced.
STORY_SIGNATURE_VERSION = 1
KNOWN_STORY_SIGNATURE_VERSIONS = frozenset({STORY_SIGNATURE_VERSION})

#: Stage names, spelled here rather than imported: this module stays free of
#: the fetchers and the clustering stack.  ``tests/test_review_provenance_
#: binding.py`` holds them equal to the owning modules' constants.
STORIES_STAGE = "stories"
THEMES_STAGE = "themes"
SUMMARIES_STAGE = "summaries"
#: The logged stages that insert raw items through ``ingest_raw_items``:
#: ``phase0.yahoo.STAGE`` and ``phase0.rss.STAGE_INGEST``.
INGESTION_STAGES = frozenset({"fetch_yahoo", "ingest_rss"})

VERIFIED = "verified"
UNVERIFIED = "unverified"


def run_facts(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The ``run_log`` facts a hop is checked against, as plain values."""

    if row is None or row.get("run_id") is None:
        return None
    return {
        "run_id": str(row["run_id"]),
        "stage": str(row["stage"]),
        "ticker": None if row.get("ticker") is None else str(row["ticker"]),
        "trading_day": str(row["trading_day"]),
        "pipeline_version": str(row["pipeline_version"]),
        "replay": bool(row["replay"]),
        "status": str(row["status"]),
        "recorded_mutation": bool(row["recorded_mutation"]),
    }


def _check_run(
    run: Any,
    *,
    run_id: Any,
    stage: str,
    ticker: str | None,
    trading_day: str,
    pipeline_version: str | None,
    forbid_replay: bool,
    partition_run: bool,
) -> str | None:
    """Why ``run`` does not authorize this hop, or ``None`` when it does.

    ``partition_run`` is true for the derived stages, whose runs always name
    their ticker: a ticker-less run's partition checks would pass for every
    ticker, so one is refused rather than read as "any".  Ingestion runs may
    legitimately be ticker-less (an RSS feed slice), and then the item's own
    ticker is what the insert path already held to the run.
    """

    if not isinstance(run_id, str) or not run_id:
        return "no run is recorded"
    if not isinstance(run, Mapping):
        return f"run {run_id!r} has no {stage} run_log row"
    if run.get("run_id") != run_id or run.get("stage") != stage:
        return f"run {run_id!r} is not a {stage} run"
    if run.get("recorded_mutation") is not True:
        return f"run {run_id!r} recorded no repository mutation"
    if forbid_replay and run.get("replay") is not False:
        return f"run {run_id!r} was a replay"
    if run.get("trading_day") != trading_day:
        return f"run {run_id!r} covers {run.get('trading_day')}, not {trading_day}"
    if partition_run and run.get("ticker") is None:
        return f"run {run_id!r} names no ticker"
    if ticker is not None and run.get("ticker") not in (None, ticker):
        return f"run {run_id!r} covers {run.get('ticker')}, not {ticker}"
    if pipeline_version is not None and run.get("pipeline_version") != (
        pipeline_version
    ):
        return (
            f"run {run_id!r} is pipeline version {run.get('pipeline_version')}, "
            f"not {pipeline_version}"
        )
    return None


def verify_raw_item(facts: Mapping[str, Any]) -> str | None:
    """Was this raw item inserted by a logged, non-replay ingestion run?"""

    try:
        item = facts["raw_item_id"]
        stage = facts["ingest_stage"]
        run_id = facts["ingest_run_id"]
        if run_id is None and stage is None:
            return f"raw item {item} has no ingestion provenance"
        if stage not in INGESTION_STAGES:
            return f"raw item {item} was not inserted by an ingestion stage"
        problem = _check_run(
            facts["run"],
            run_id=run_id,
            stage=stage,
            ticker=facts["ticker"],
            trading_day=facts["effective_day"],
            pipeline_version=None,
            forbid_replay=True,
            partition_run=False,
        )
    except (KeyError, TypeError) as exc:
        return f"raw item provenance is malformed: {exc}"
    return None if problem is None else f"raw item {item}: {problem}"


def verify_story(facts: Mapping[str, Any]) -> str | None:
    """Was this story's current content written by a logged story run?"""

    try:
        story = facts["story_id"]
        problem = _check_run(
            facts["run"],
            run_id=facts["build_run_id"],
            stage=STORIES_STAGE,
            ticker=facts["ticker"],
            trading_day=facts["trading_day"],
            pipeline_version=facts["pipeline_version"],
            forbid_replay=False,
            partition_run=True,
        )
    except (KeyError, TypeError) as exc:
        return f"story provenance is malformed: {exc}"
    return None if problem is None else f"story {story}: {problem}"


def theme_fingerprint_matches(
    ticker: str, trading_day: str, fingerprint: Any, member_keys: Sequence[Any]
) -> bool:
    """Does the stored ``themes.fingerprint`` recompute from its membership?

    The existing M5 content digest (:func:`nlp.themes.theme_fingerprint_for`)
    over the member stories' ``cluster_fingerprint`` values -- not a second
    theme hash.
    """

    from nlp.themes.service import ThemeInputError, theme_fingerprint_for

    try:
        return fingerprint == theme_fingerprint_for(
            ticker, date.fromisoformat(trading_day), list(member_keys)
        )
    except (ThemeInputError, TypeError, ValueError):
        return False


def verify_theme_build(facts: Mapping[str, Any] | None) -> str | None:
    """Is this theme set bound to the story generation it now sits on?

    All of: a complete binding under a recognized signature version; that
    signature equal to the partition's *current* story-generation signature
    (read in the same snapshot); a themes-stage run for this partition that
    recorded a mutation; and every theme's stored fingerprint recomputing
    from its persisted membership.
    """

    if facts is None:
        return "there is no theme set"
    try:
        signature = facts["build_story_signature"]
        version = facts["build_story_signature_version"]
        if signature is None or version is None or facts["build_run_id"] is None:
            return "the theme set carries no build binding"
        # Exactly an int: ``True`` and ``1.0`` compare equal to 1 and are
        # still not a version this code wrote.
        if type(version) is not int or version not in KNOWN_STORY_SIGNATURE_VERSIONS:
            return f"story-signature version {version!r} is not recognized"
        if signature != facts["current_story_signature"]:
            return (
                "the theme set was built over a story generation that is no longer "
                "the partition's"
            )
        problem = _check_run(
            facts["run"],
            run_id=facts["build_run_id"],
            stage=THEMES_STAGE,
            ticker=facts["ticker"],
            trading_day=facts["trading_day"],
            pipeline_version=facts["pipeline_version"],
            forbid_replay=False,
            partition_run=True,
        )
        if problem is not None:
            return f"theme build: {problem}"
        for theme in facts["themes"]:
            if not theme_fingerprint_matches(
                facts["ticker"],
                facts["trading_day"],
                theme["fingerprint"],
                theme["member_keys"],
            ):
                return (
                    f"theme {theme['theme_id']}: its stored fingerprint does not "
                    "recompute from its membership"
                )
    except (KeyError, TypeError) as exc:
        return f"theme build provenance is malformed: {exc}"
    return None


def verify_summary(
    facts: Mapping[str, Any] | None, artifact: Mapping[str, Any]
) -> str | None:
    """Was this artifact produced by its accepted generation in a summaries run?

    The producing generation is the one ``outcome = 'accepted'`` row naming
    the artifact; ``discarded_duplicate`` rows also name it but produced
    nothing, and a cache hit writes no generation at all.
    """

    where = f"artifact {artifact.get('artifact_id')}"
    if facts is None:
        return f"{where} has no accepted generation"
    try:
        generation = facts["generation"]
        if generation is None:
            return f"{where} has no accepted generation"
        if generation["outcome"] != "accepted":
            return f"{where}: its recorded producer is not an accepted generation"
        for key in (
            "artifact_id",
            "ticker",
            "trading_day",
            "pipeline_version",
            "theme_id",
            "input_fingerprint",
            "policy_fingerprint",
        ):
            if generation[key] != artifact.get(key):
                return f"{where}: its accepted generation disagrees on {key}"
        problem = _check_run(
            facts["run"],
            run_id=generation["run_id"],
            stage=SUMMARIES_STAGE,
            ticker=generation["ticker"],
            trading_day=generation["trading_day"],
            pipeline_version=generation["pipeline_version"],
            forbid_replay=False,
            partition_run=True,
        )
    except (KeyError, TypeError) as exc:
        return f"{where}: summary provenance is malformed: {exc}"
    return None if problem is None else f"{where}: {problem}"


__all__ = [
    "INGESTION_STAGES",
    "KNOWN_STORY_SIGNATURE_VERSIONS",
    "STORIES_STAGE",
    "STORY_SIGNATURE_VERSION",
    "SUMMARIES_STAGE",
    "THEMES_STAGE",
    "UNVERIFIED",
    "VERIFIED",
    "run_facts",
    "theme_fingerprint_matches",
    "verify_raw_item",
    "verify_story",
    "verify_summary",
    "verify_theme_build",
]

"""Cache-aware Gemini summarization of M5 theme sets into Phase 0's database.

Author: Abhi
Responsibility: The A1 pipeline stage. Turns one ticker-day's real
`nlp.themes.models.ThemeSet` into persisted `themes` rows, calling
`ai.summarization.summarize()` only for themes whose content actually
changed since the last run - the mechanism issue #73 (A3) asks for.

Nothing before this wired M5's clustering output into the real database at
all; this module is that wiring, with caching and per-call token/cost/
latency logging (via `Phase0Repository.record_summarization_usage`, #73)
built in from the start rather than bolted on afterward.

Cache key: `compute_content_hash` hashes exactly what the summarizer saw
(member story id/title/description/outlet/published_at) - deliberately
distinct from M5's own `Theme.fingerprint`, which only hashes the *set* of
member keys. A corrected story title changes what should be resummarized
without changing `fingerprint`, so `fingerprint` alone is not a safe cache
key for this purpose; `reconcile_themes` still uses `fingerprint` as the
theme's row identity, exactly as it already does for every other caller.

Stored `themes.summary` holds the sentences shape `ai.summarization.
ThemeSummary` produces (`[{text, citation_ids}, ...]`), with citation_ids
left as the original story keys - matching the citation contract
`docs/phase0_api_contract.md` documents for the read API. Resolving those
same citations to raw-item ids for the database's own `theme_citations`
table (a different, internal-only concern) happens separately here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ai.summarization import GenerationAttempt, GeminiClient, ThemeInput, summarize
from nlp.embeddings import serialize_vector
from nlp.themes.models import ThemeSet
from nlp.themes.summarization import theme_to_summarizer_input

from .models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    ReconciliationReport,
    ThemeRecord,
    ThemeSetRecord,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .repository import Phase0Repository


def compute_content_hash(theme_input: ThemeInput) -> str:
    """Hash exactly what the summarizer saw for one theme.

    Deliberately distinct from M5's `Theme.fingerprint` - see module
    docstring. `theme_to_summarizer_input` preserves `Theme.evidence`'s
    deterministic order, so this is stable across runs of unchanged
    content and changes whenever a member's title, description, outlet,
    or timestamp does, even if membership itself did not.
    """

    payload = {
        "ticker": theme_input.ticker,
        "trading_day": theme_input.trading_day,
        "member_stories": [
            {
                "id": story.id,
                "title": story.title,
                "description": story.description,
                "outlet": story.outlet,
                "published_at": story.published_at,
            }
            for story in theme_input.member_stories
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SummarizationRunReport:
    """What one `summarize_theme_set` call actually did."""

    cache_hits: int
    cache_misses: int
    attempts: tuple[GenerationAttempt, ...]
    reconciliation: ReconciliationReport


def _cache_by_fingerprint(stored_theme_set: Optional[dict]) -> dict[str, dict]:
    if stored_theme_set is None:
        return {}
    return {
        str(row["fingerprint"]): row
        for row in stored_theme_set["themes"]
        if row.get("fingerprint")
    }


def _sentences_from_json(raw: Optional[str]) -> list[dict]:
    if not raw:
        return []
    return json.loads(raw)


def _cited_story_keys(sentences: list[dict]) -> set[str]:
    return {
        citation_id
        for sentence in sentences
        for citation_id in sentence.get("citation_ids", [])
    }


def _cache_hit_sentences(stored_row: dict, content_hash: str) -> Optional[tuple[str, list[dict]]]:
    """The stored (label, sentences) if this row is a valid cache hit, else None."""

    if stored_row.get("content_hash") != content_hash or not stored_row.get("summary"):
        return None
    return str(stored_row["label"]), _sentences_from_json(stored_row.get("summary"))


def _summarize_and_log(
    theme_input: ThemeInput,
    theme_fingerprint: str,
    *,
    client: Optional[GeminiClient],
    attempts: list[GenerationAttempt],
    calls_for_log: list[dict[str, Any]],
) -> tuple[str, list[dict]]:
    """Call the real summarizer for one theme, recording every attempt."""

    recorded_before = len(attempts)
    theme_summary = summarize(theme_input, client=client, on_attempt=attempts.append)
    for attempt in attempts[recorded_before:]:
        calls_for_log.append(
            {
                "theme_fingerprint": theme_fingerprint,
                "attempt": attempt.attempt,
                "success": attempt.success,
                "latency_ms": attempt.latency_ms,
                # Named input_size/output_size, not *_tokens: see
                # Phase0Repository.record_summarization_usage's docstring -
                # a key containing "token" gets its value silently redacted
                # before it reaches run_log.
                "input_size": attempt.usage.input_tokens if attempt.usage else None,
                "output_size": attempt.usage.output_tokens if attempt.usage else None,
                "error": attempt.error,
            }
        )
    sentences = [sentence.model_dump() for sentence in theme_summary.sentences]
    return theme_summary.label, sentences


def _citation_item_ids(
    sentences: list[dict],
    *,
    story_id_by_fingerprint: dict[str, int],
    raw_items_by_story: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """Every raw item id a sentence's story-level citation resolves to."""

    cited_story_ids = {
        story_id_by_fingerprint[key] for key in _cited_story_keys(sentences)
    }
    return tuple(
        sorted(
            {
                item_id
                for story_id in cited_story_ids
                for item_id in raw_items_by_story.get(story_id, ())
            }
        )
    )


def summarize_theme_set(
    theme_set: ThemeSet,
    *,
    repository: "Phase0Repository",
    run_id: str,
    pipeline_version: str,
    client: Optional[GeminiClient] = None,
) -> SummarizationRunReport:
    """Summarize (or reuse) every theme in `theme_set` and persist the result.

    Cache hit: a stored theme with the same `fingerprint` and the same
    `compute_content_hash` result already has a summary -> that summary is
    reused verbatim, with zero calls to `ai.summarization.summarize`. Cache
    miss: `summarize()` is called, and its per-attempt latency/token usage
    captured via `on_attempt`. Both paths, plus other_coverage/excluded,
    land together through `repository.reconcile_themes`, and the run's LLM
    accounting through `repository.record_summarization_usage` - both
    inside one `stage_run`, the same multi-call-one-run pattern
    `phase0/yahoo.py` already uses.
    """

    trading_day = theme_set.trading_day.isoformat()

    all_fingerprints = sorted(
        {entry.story_key for theme in theme_set.themes for entry in theme.evidence}
        | {entry.story_key for entry in theme_set.other_coverage}
        | {entry.story_key for entry in theme_set.excluded}
    )
    story_id_by_fingerprint = repository.story_ids_for_fingerprints(
        ticker=theme_set.ticker,
        trading_day=trading_day,
        pipeline_version=pipeline_version,
        fingerprints=all_fingerprints,
    )
    raw_items_by_story = repository.raw_item_ids_for_stories(
        story_ids=list(story_id_by_fingerprint.values())
    )

    stored = repository.theme_set(
        ticker=theme_set.ticker,
        trading_day=trading_day,
        pipeline_version=pipeline_version,
    )
    stored_by_fingerprint = _cache_by_fingerprint(stored)

    attempts: list[GenerationAttempt] = []
    calls_for_log: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0
    theme_records: list[ThemeRecord] = []

    for theme in theme_set.themes:
        theme_input = theme_to_summarizer_input(theme)
        content_hash = compute_content_hash(theme_input)
        stored_row = stored_by_fingerprint.get(theme.fingerprint)

        cache_hit = _cache_hit_sentences(stored_row, content_hash) if stored_row else None
        if cache_hit is not None:
            cache_hits += 1
            label, sentences = cache_hit
        else:
            cache_misses += 1
            label, sentences = _summarize_and_log(
                theme_input,
                theme.fingerprint,
                client=client,
                attempts=attempts,
                calls_for_log=calls_for_log,
            )

        member_fingerprints = [entry.story_key for entry in theme.evidence]
        story_ids = tuple(story_id_by_fingerprint[key] for key in member_fingerprints)
        citation_item_ids = _citation_item_ids(
            sentences,
            story_id_by_fingerprint=story_id_by_fingerprint,
            raw_items_by_story=raw_items_by_story,
        )

        theme_records.append(
            ThemeRecord(
                fingerprint=theme.fingerprint,
                theme_key=theme.theme_key,
                label=label,
                story_ids=story_ids,
                citation_item_ids=citation_item_ids,
                label_source=theme.label_source,
                summary=json.dumps(sentences, separators=(",", ":")),
                status="ready",
                salience=theme.salience,
                salience_rank=theme.salience_rank,
                cohesion=theme.cohesion,
                min_pairwise_cohesion=theme.min_pairwise_cohesion,
                story_count=theme.story_count,
                outlet_count=theme.outlet_count,
                latest_published_at=theme.salience_features.latest_published_at,
                salience_story_component=theme.salience_features.story_component,
                salience_outlet_component=theme.salience_features.outlet_component,
                salience_recency_component=theme.salience_features.recency_component,
                centroid=serialize_vector(theme.centroid) if theme.centroid else None,
                matched_previous_key=theme.matched_previous_key,
                method=theme.method.value,
                content_hash=content_hash,
            )
        )

    other_coverage_records = tuple(
        OtherCoverageRecord(
            story_id=story_id_by_fingerprint[entry.story_key], reason=entry.reason.value
        )
        for entry in theme_set.other_coverage
    )
    excluded_records = tuple(
        ExcludedStoryRecord(
            story_id=story_id_by_fingerprint[entry.story_key], reason=entry.reason.value
        )
        for entry in theme_set.excluded
    )

    theme_set_record = ThemeSetRecord(
        method=theme_set.method.value,
        method_reason=theme_set.method_reason,
        quality=dataclasses.asdict(theme_set.quality),
        source_metadata=(
            dataclasses.asdict(theme_set.source_metadata)
            if theme_set.source_metadata is not None
            else None
        ),
        config_fingerprint=theme_set.config_fingerprint,
        algorithm_version=theme_set.algorithm_version,
        model_name=theme_set.model_name,
        model_revision=theme_set.model_revision,
        embedding_dimension=theme_set.embedding_dimension,
    )

    with repository.stage_run(
        run_id=run_id,
        stage="summarize",
        ticker=theme_set.ticker,
        trading_day=trading_day,
        pipeline_version=pipeline_version,
    ) as run:
        reconciliation = repository.reconcile_themes(
            run=run,
            ticker=theme_set.ticker,
            trading_day=trading_day,
            pipeline_version=pipeline_version,
            theme_set=theme_set_record,
            themes=theme_records,
            other_coverage=other_coverage_records,
            excluded=excluded_records,
            terminal=False,
        )
        repository.record_summarization_usage(
            run=run,
            calls=calls_for_log,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            terminal=True,
        )

    return SummarizationRunReport(
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        attempts=tuple(attempts),
        reconciliation=reconciliation,
    )

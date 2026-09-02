"""Content-hash caching for ai.summarization.summarize() (issue #73 / A3).

Deliberately narrow and repository-agnostic: this module knows nothing
about Phase 0's database, story ids, ThemeRecord, or stage/run lifecycle -
those belong to whoever wires M5's ThemeSet into the database (I5's
downstream-stage work; `phase0/README.md`'s ownership table has no
summarization/themes-wiring entry, and `docs/PHASE0_DATA_PIPELINE.md`
states M5 "is implemented in `nlp/`, not registered" yet).

This module answers exactly one question - does this theme's content AND
the summarizer's own policy match what was cached last time, or does
`summarize()` need to run again - and reports what happened, so a caller
persists and logs the outcome under its own stage's contract. Per
`phase0/README.md`'s worked `reconcile_themes` example, the data-owning
call in that contract should be the *terminal* one, called last; per
`phase0/yahoo.py`'s pattern, the summarizer call itself belongs *inside*
whatever `stage_run` a caller opens, so a failure is recorded rather than
silently escaping an unaccounted run. Neither of those integration
concerns is implemented here - see the module docstring above.

Cache key: `compute_content_hash` hashes both the member story content a
caller fetched (title/description/outlet/published_at - what
`ai.summarization.summarize` actually reads) and
`ai.summarization.policy_fingerprint` (prompt, model, generation config,
output schema). Both must match a previous run's stored value for a
summary to be reused; either changing invalidates it. This is what keeps a
prompt or model change from silently reusing prose written under an old
policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional

from .summarization import (
    GenerationAttempt,
    GeminiClient,
    ThemeInput,
    policy_fingerprint,
    summarize,
)


@dataclass(frozen=True)
class StoredSummary:
    """What a caller already has on file for one theme, if anything.

    Deliberately repository-agnostic: a caller adapts whatever it stores
    (a database row, a fixture, anything) into this shape before calling
    `summarize_with_cache`.
    """

    content_hash: str
    label: str
    sentences: tuple[dict, ...]


@dataclass(frozen=True)
class CachedTheme:
    """One theme's cache-checked, possibly-freshly-summarized result."""

    content_hash: str
    label: str
    sentences: tuple[dict, ...]
    cache_hit: bool
    attempts: tuple[GenerationAttempt, ...]


def compute_content_hash(theme_input: ThemeInput, *, model: str) -> str:
    """Hash exactly what would decide whether `summarize()` must run again.

    Mixes two independent things that must *both* still match for a stored
    summary to remain valid: the member story content (`theme_input`) and
    the summarizer's own policy identity (`ai.summarization.
    policy_fingerprint`). A story's text being exactly what it was last
    time is not enough on its own if the prompt or model changed in
    between - reusing prose written under a different policy would be
    silently wrong, not merely stale.
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
        "policy_fingerprint": policy_fingerprint(model),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarize_with_cache(
    theme_input: ThemeInput,
    *,
    stored: Optional[StoredSummary],
    client: Optional[GeminiClient] = None,
) -> CachedTheme:
    """Reuse `stored` if its content_hash still matches; otherwise summarize.

    Zero calls to `ai.summarization.summarize` on a cache hit - the
    mechanism issue #73 asks for. Cache eligibility is judged purely on
    `content_hash` equality, which already accounts for both story content
    and summarizer policy (see `compute_content_hash`), so a policy change
    invalidates every cached theme automatically with no separate
    invalidation logic to keep in sync.
    """

    resolved_client = client or GeminiClient()
    content_hash = compute_content_hash(theme_input, model=resolved_client.model)

    if stored is not None and stored.content_hash == content_hash:
        return CachedTheme(
            content_hash=content_hash,
            label=stored.label,
            sentences=stored.sentences,
            cache_hit=True,
            attempts=(),
        )

    attempts: list[GenerationAttempt] = []
    theme_summary = summarize(theme_input, client=resolved_client, on_attempt=attempts.append)
    sentences = tuple(sentence.model_dump() for sentence in theme_summary.sentences)
    return CachedTheme(
        content_hash=content_hash,
        label=theme_summary.label,
        sentences=sentences,
        cache_hit=False,
        attempts=tuple(attempts),
    )

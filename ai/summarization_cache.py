"""Content-hash caching for ai.guarded_summary (issue #73 / A3).

Deliberately narrow and repository-agnostic: this module knows nothing
about Phase 0's database, story ids, ThemeRecord, or stage/run lifecycle -
those belong to whoever wires a persisted theme's evidence into generation
end to end (`phase0/summaries.py` already builds the input this module
consumes; nothing yet calls it in production).

This module answers exactly one question - does a previously generated
summary still match both this theme's evidence AND the summarizer's own
policy, or does `ai.guarded_summary.generate_guarded_summary` need to run
again - and reports what happened, so a caller persists and logs the
outcome under its own stage's contract. Per `phase0/README.md`'s worked
`reconcile_themes` example, the data-owning call in that contract should be
the *terminal* one, called last; per `phase0/yahoo.py`'s pattern, the
generation call itself belongs *inside* whatever `stage_run` a caller
opens, so a failure is recorded rather than silently escaping an
unaccounted run. Neither of those integration concerns is implemented
here.

Cache key: both `SummaryGenerationInput.input_fingerprint` (the evidence a
caller already built, via `phase0/summaries.py::build_generation_input` or
equivalent) and `ai.guarded_summary.compute_policy_fingerprint` (prompt
version, model, generation config, output schema, copy rules) must match a
previous run's stored values for a summary to be reused. Either changing
invalidates it - this is what keeps a prompt edit, a model swap, or a
copy-rule change from silently reusing prose generated under a different
policy. Both fingerprints are computed by `ai.guarded_summary` itself, not
reinvented here, so this module can never drift from what that module
actually used to decide acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .guarded_summary import (
    Rules,
    SummaryGenerationInput,
    SummaryGenerationResult,
    compute_policy_fingerprint,
    generate_guarded_summary,
    load_copy_rules,
)
from .summarization import DEFAULT_MAX_OUTPUT_TOKENS, Sentence, ThemeSummary


@dataclass(frozen=True)
class StoredSummary:
    """What a caller already has on file for one theme, if anything.

    Deliberately repository-agnostic: a caller adapts whatever it stores
    (a database row, a fixture, anything) into this shape before calling
    `summarize_with_cache`. `sentences` is the plain JSON-shaped form
    (`{"text": ..., "citation_ids": [...]}` per entry), matching how
    `ai.summarization.ThemeSummary` round-trips through JSON.
    """

    input_fingerprint: str
    policy_fingerprint: str
    label: str
    sentences: tuple[dict, ...]

    def as_theme_summary(self) -> ThemeSummary:
        return ThemeSummary(
            label=self.label,
            sentences=[Sentence(**entry) for entry in self.sentences],
        )


@dataclass(frozen=True)
class CachedSummary:
    """One theme's cache-checked, possibly-freshly-generated result.

    `summary`/`accepted` are populated the same way on a cache hit and on
    an accepted cache miss, so a caller does not need to branch on
    `cache_hit` to get the summary it actually needs. `result` is `None`
    on a cache hit (nothing was generated) and the full
    `SummaryGenerationResult` - including every attempt's usage/latency -
    on a cache miss, for a caller to log.
    """

    input_fingerprint: str
    policy_fingerprint: str
    cache_hit: bool
    accepted: bool
    summary: Optional[ThemeSummary]
    result: Optional[SummaryGenerationResult]


def summarize_with_cache(
    generation_input: SummaryGenerationInput,
    *,
    stored: Optional[StoredSummary],
    client: Any,
    max_attempts: int = 2,
    rules: Optional[Rules] = None,
) -> CachedSummary:
    """Reuse `stored` if both fingerprints still match; otherwise generate.

    Zero calls to `ai.guarded_summary.generate_guarded_summary` (and so
    zero provider calls) on a cache hit - the mechanism issue #73 asks
    for. The policy fingerprint is computed here with the exact same
    inputs `generate_guarded_summary` uses internally (model identity,
    `max_attempts`, the active copy rules, `max_output_tokens`), so a
    cache decision never disagrees with what a fresh generation would
    itself report as its policy fingerprint - see
    `test_precomputed_policy_fingerprint_matches_a_fresh_generation`.
    """

    active_rules = load_copy_rules() if rules is None else rules
    policy_fingerprint = compute_policy_fingerprint(
        model=str(getattr(client, "model", "unspecified")),
        max_attempts=max_attempts,
        rules=active_rules,
        max_output_tokens=int(getattr(client, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
    )

    if (
        stored is not None
        and stored.input_fingerprint == generation_input.input_fingerprint
        and stored.policy_fingerprint == policy_fingerprint
    ):
        return CachedSummary(
            input_fingerprint=generation_input.input_fingerprint,
            policy_fingerprint=policy_fingerprint,
            cache_hit=True,
            accepted=True,
            summary=stored.as_theme_summary(),
            result=None,
        )

    result = generate_guarded_summary(
        generation_input, client=client, max_attempts=max_attempts, rules=active_rules
    )
    return CachedSummary(
        input_fingerprint=result.input_fingerprint,
        policy_fingerprint=result.policy_fingerprint,
        cache_hit=False,
        accepted=result.accepted,
        summary=result.summary,
        result=result,
    )


def usage_log_entries(result: SummaryGenerationResult) -> tuple[dict[str, Any], ...]:
    """Flatten one generation's attempts into plain dicts for logging.

    A caller persisting telemetry (e.g. via a `Phase0Repository` logged
    mutation) adapts these rather than reaching into `AttemptRecord`
    itself, so this module stays the one place that knows the attempt
    shape well enough to flatten it.
    """

    entries = []
    for attempt in result.attempts:
        usage = attempt.usage
        entries.append(
            {
                "attempt": attempt.attempt,
                "outcome": attempt.outcome,
                "validation_codes": list(attempt.validation_codes),
                "latency_ms": attempt.latency_ms,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "candidate_tokens": usage.candidate_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "error": attempt.error,
            }
        )
    return tuple(entries)

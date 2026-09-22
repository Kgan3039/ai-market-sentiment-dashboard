"""Tests for ai.summarization_cache (issue #73 / A3).

unittest.TestCase style, matching tests/test_ai_summarization.py.
"""

from __future__ import annotations

import re
import unittest

from ai.guarded_summary import EvidenceStory, SummaryGenerationInput, ThemeReference, citation_id_for
from ai.summarization import GenerationUsage
from ai.summarization_cache import CachedSummary, StoredSummary, summarize_with_cache, usage_log_entries

ID_LINE_RE = re.compile(r"- id: (\S+)")


def make_generation_input(num_stories: int = 2, *, ticker: str = "NVDA") -> SummaryGenerationInput:
    theme = ThemeReference(theme_id=1, theme_key="theme-1", label="Test theme", pipeline_version="v1")
    evidence = tuple(
        EvidenceStory(
            citation_id=citation_id_for(index + 1),
            persisted_story_id=index + 1,
            title=f"Story {index + 1} headline",
            description=f"Story {index + 1} description",
            outlet="Reuters",
            published_at="2026-09-01T12:00:00+00:00",
        )
        for index in range(num_stories)
    )
    return SummaryGenerationInput.compose(
        ticker=ticker, trading_day="2026-09-01", theme=theme, evidence=evidence
    )


class FakeGeneratingClient:
    """Deterministic stand-in for a guarded-generation client.

    Extracts the citation ids actually serialized into the prompt (via
    ai.guarded_summary.build_prompt) and answers with a schema-valid
    candidate that cites real ids, so it works generically without
    hardcoding per-input responses.
    """

    model = "fake-model"
    max_output_tokens = 512

    def __init__(self) -> None:
        self.calls = 0
        self.last_usage = None

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        self.last_usage = GenerationUsage(prompt_tokens=50, candidate_tokens=10, total_tokens=60)
        ids = ID_LINE_RE.findall(user_prompt)
        assert ids
        sentences = [
            {"text": f"Coverage sentence {index + 1}.", "citation_ids": [citation_id]}
            for index, citation_id in enumerate(ids[:2])
        ]
        if len(sentences) < 2:
            sentences.append({"text": "Additional coverage.", "citation_ids": [ids[0]]})
        return response_schema.model_validate({"label": "Coverage of recent developments", "sentences": sentences})


class NeverCallClient:
    """Fails the test immediately if summarize_with_cache calls it at all."""

    model = "fake-model"
    max_output_tokens = 512

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        raise AssertionError("generate_guarded_summary must not be called on a cache hit")


class SummarizeWithCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generation_input = make_generation_input()

    def test_no_stored_value_is_always_a_cache_miss(self) -> None:
        result = summarize_with_cache(self.generation_input, stored=None, client=FakeGeneratingClient())
        self.assertIsInstance(result, CachedSummary)
        self.assertFalse(result.cache_hit)
        self.assertTrue(result.accepted)
        self.assertIsNotNone(result.result)
        self.assertEqual(result.result.provider_calls, 1)

    def test_matching_fingerprints_are_a_cache_hit_with_zero_calls(self) -> None:
        client = FakeGeneratingClient()
        first = summarize_with_cache(self.generation_input, stored=None, client=client)
        self.assertEqual(client.calls, 1)

        stored = StoredSummary(
            input_fingerprint=first.input_fingerprint,
            policy_fingerprint=first.policy_fingerprint,
            label=first.summary.label,
            sentences=tuple(sentence.model_dump() for sentence in first.summary.sentences),
        )
        second = summarize_with_cache(self.generation_input, stored=stored, client=NeverCallClient())

        self.assertTrue(second.cache_hit)
        self.assertIsNone(second.result)
        self.assertTrue(second.accepted)
        self.assertEqual(second.summary.label, first.summary.label)

    def test_mismatched_input_fingerprint_is_a_cache_miss(self) -> None:
        stored = StoredSummary(
            input_fingerprint="stale-input-fingerprint",
            policy_fingerprint="anything",
            label="Old label",
            sentences=({"text": "Old.", "citation_ids": ["story:1"]},),
        )
        result = summarize_with_cache(self.generation_input, stored=stored, client=FakeGeneratingClient())
        self.assertFalse(result.cache_hit)

    def test_a_policy_change_invalidates_a_cache_hit_with_identical_evidence(self) -> None:
        """Same evidence, but the stored policy_fingerprint was computed
        under a different max_attempts - reuse must not happen (issue
        #73's central ask: a summarization-policy change must not
        silently reuse old prose)."""

        client = FakeGeneratingClient()
        first = summarize_with_cache(self.generation_input, stored=None, client=client, max_attempts=1)

        stored = StoredSummary(
            input_fingerprint=first.input_fingerprint,
            policy_fingerprint=first.policy_fingerprint,
            label=first.summary.label,
            sentences=tuple(sentence.model_dump() for sentence in first.summary.sentences),
        )

        # Same evidence (same input_fingerprint), different max_attempts ->
        # different policy_fingerprint -> must not reuse.
        result = summarize_with_cache(self.generation_input, stored=stored, client=FakeGeneratingClient(), max_attempts=2)
        self.assertFalse(result.cache_hit)

    def test_precomputed_policy_fingerprint_matches_a_fresh_generation(self) -> None:
        """The cache-check's own policy_fingerprint computation must never
        drift from what generate_guarded_summary itself reports, or a
        cache decision and a fresh generation could disagree about
        identity."""

        result = summarize_with_cache(self.generation_input, stored=None, client=FakeGeneratingClient())
        # A second cache check against the just-generated result's own
        # reported fingerprints must recognize it as a hit.
        stored = StoredSummary(
            input_fingerprint=result.input_fingerprint,
            policy_fingerprint=result.policy_fingerprint,
            label=result.summary.label,
            sentences=tuple(sentence.model_dump() for sentence in result.summary.sentences),
        )
        second = summarize_with_cache(self.generation_input, stored=stored, client=NeverCallClient())
        self.assertTrue(second.cache_hit)


class UsageLogEntriesTests(unittest.TestCase):
    def test_flattens_one_entry_per_attempt_with_expected_fields(self) -> None:
        result = summarize_with_cache(
            make_generation_input(), stored=None, client=FakeGeneratingClient()
        ).result
        entries = usage_log_entries(result)
        self.assertEqual(len(entries), result.provider_calls)
        entry = entries[0]
        self.assertEqual(set(entry), {
            "attempt", "outcome", "validation_codes", "latency_ms",
            "prompt_tokens", "candidate_tokens", "total_tokens", "error",
        })
        self.assertEqual(entry["prompt_tokens"], 50)
        self.assertEqual(entry["candidate_tokens"], 10)
        self.assertEqual(entry["total_tokens"], 60)


if __name__ == "__main__":
    unittest.main()

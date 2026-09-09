"""Tests for ai.summarization_cache (issue #73 / A3).

unittest.TestCase style, matching tests/test_ai_summarization.py - this
module is squarely part of ai/, not phase0/, since the rework that split it
out of phase0/summarization.py (see review discussion on the original PR).
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from ai.summarization import GenerationUsage, MemberStory, ThemeInput
from ai.summarization_cache import CachedTheme, StoredSummary, compute_content_hash, summarize_with_cache

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "ai" / "fixtures" / "theme_fixtures.json"
ID_LINE_RE = re.compile(r"- id: (\S+)")


def load_first_fixture_theme() -> ThemeInput:
    with FIXTURES_PATH.open(encoding="utf-8") as handle:
        raw_themes = json.load(handle)
    raw = raw_themes[0]
    stories = [MemberStory(**story) for story in raw["member_stories"]]
    return ThemeInput(ticker=raw["ticker"], trading_day=raw.get("trading_day"), member_stories=stories)


class FakeGeminiClient:
    """Same prompt-id-extraction trick as tests/test_ai_summarization.py's
    fake, plus the `.model`/`.usage` attributes summarize_with_cache reads."""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0
        self.usage = GenerationUsage(input_tokens=50, output_tokens=10)

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        story_ids = ID_LINE_RE.findall(user_prompt)
        assert story_ids
        sentences = [
            {"text": f"Coverage sentence {index + 1}.", "citation_ids": [story_id]}
            for index, story_id in enumerate(story_ids[:2])
        ]
        if len(sentences) < 2:
            sentences.append({"text": "Additional coverage.", "citation_ids": [story_ids[0]]})
        return response_schema.model_validate({"label": "Coverage of recent developments", "sentences": sentences})


class NeverCallGeminiClient:
    """Fails the test immediately if summarize_with_cache calls it at all."""

    model = "fake-model"

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        raise AssertionError("summarize() must not be called on a cache hit")


class ComputeContentHashTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = load_first_fixture_theme()

    def test_stable_for_identical_input(self) -> None:
        first = compute_content_hash(self.theme, model="gemini-2.5-flash")
        second = compute_content_hash(self.theme, model="gemini-2.5-flash")
        self.assertEqual(first, second)

    def test_changes_when_story_content_changes(self) -> None:
        before = compute_content_hash(self.theme, model="gemini-2.5-flash")
        changed_stories = list(self.theme.member_stories)
        changed_stories[0] = MemberStory(
            id=changed_stories[0].id,
            title=changed_stories[0].title + " (updated)",
            description=changed_stories[0].description,
            outlet=changed_stories[0].outlet,
            published_at=changed_stories[0].published_at,
        )
        changed_theme = ThemeInput(
            ticker=self.theme.ticker, trading_day=self.theme.trading_day, member_stories=changed_stories
        )
        after = compute_content_hash(changed_theme, model="gemini-2.5-flash")
        self.assertNotEqual(before, after)

    def test_changes_when_model_changes_even_with_identical_stories(self) -> None:
        """Proves the policy-fingerprint mixing actually works: same
        content, different model -> different hash."""

        before = compute_content_hash(self.theme, model="gemini-2.5-flash")
        after = compute_content_hash(self.theme, model="gemini-2.5-pro")
        self.assertNotEqual(before, after)


class SummarizeWithCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = load_first_fixture_theme()

    def test_no_stored_value_is_always_a_cache_miss(self) -> None:
        result = summarize_with_cache(self.theme, stored=None, client=FakeGeminiClient())
        self.assertIsInstance(result, CachedTheme)
        self.assertFalse(result.cache_hit)
        self.assertEqual(len(result.attempts), 1)

    def test_matching_content_hash_is_a_cache_hit_with_zero_calls(self) -> None:
        client = FakeGeminiClient()
        first = summarize_with_cache(self.theme, stored=None, client=client)
        self.assertEqual(client.calls, 1)

        stored = StoredSummary(content_hash=first.content_hash, label=first.label, sentences=first.sentences)
        second = summarize_with_cache(self.theme, stored=stored, client=NeverCallGeminiClient())

        self.assertTrue(second.cache_hit)
        self.assertEqual(second.attempts, ())
        self.assertEqual(second.label, first.label)
        self.assertEqual(second.sentences, first.sentences)

    def test_mismatched_content_hash_is_a_cache_miss(self) -> None:
        stored = StoredSummary(content_hash="stale-hash", label="Old label", sentences=({"text": "Old.", "citation_ids": ["x"]},))
        result = summarize_with_cache(self.theme, stored=stored, client=FakeGeminiClient())
        self.assertFalse(result.cache_hit)
        self.assertEqual(len(result.attempts), 1)

    def test_a_model_change_invalidates_a_cache_hit(self) -> None:
        """Same story content, but the stored hash was computed under a
        different model - reuse must not happen (issue #73's central ask:
        a summarization-policy change must not silently reuse old prose)."""

        stale_content_hash = compute_content_hash(self.theme, model="gemini-2.5-pro")
        stored = StoredSummary(content_hash=stale_content_hash, label="Old label", sentences=({"text": "Old.", "citation_ids": ["x"]},))

        client = FakeGeminiClient()
        client.model = "gemini-2.5-flash"
        result = summarize_with_cache(self.theme, stored=stored, client=client)

        self.assertFalse(result.cache_hit)
        self.assertEqual(client.calls, 1)

    def test_cache_miss_attempt_carries_usage(self) -> None:
        result = summarize_with_cache(self.theme, stored=None, client=FakeGeminiClient())
        self.assertEqual(result.attempts[0].usage, GenerationUsage(input_tokens=50, output_tokens=10))


if __name__ == "__main__":
    unittest.main()

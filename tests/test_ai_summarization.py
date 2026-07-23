"""Tests for the Phase 0 cited theme summarization module (issue #65 / A1)."""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path

from ai.summarization import (
    DEFAULT_TEMPERATURE,
    MAX_LABEL_WORDS,
    MAX_SENTENCES,
    MIN_SENTENCES,
    SYSTEM_PROMPT,
    MemberStory,
    SummarizationError,
    ThemeInput,
    ThemeSummary,
    build_generation_config_kwargs,
    resolve_citations,
    summarize,
)

FIXTURES_PATH = Path(__file__).resolve().parents[1] / "ai" / "fixtures" / "theme_fixtures.json"
ID_LINE_RE = re.compile(r"- id: (\S+)")


def load_fixture_themes() -> list[ThemeInput]:
    with FIXTURES_PATH.open(encoding="utf-8") as handle:
        raw_themes = json.load(handle)

    themes = []
    for raw in raw_themes:
        stories = [MemberStory(**story) for story in raw["member_stories"]]
        themes.append(
            ThemeInput(ticker=raw["ticker"], trading_day=raw.get("trading_day"), member_stories=stories)
        )
    return themes


class FakeGeminiClient:
    """Deterministic stand-in for GeminiClient.generate.

    Extracts the story ids that were actually serialized into the prompt (via
    build_user_prompt) and builds a schema-valid ThemeSummary that cites real
    ids, so the same fake works generically across all 20 fixtures without
    hardcoding per-fixture responses.
    """

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        story_ids = ID_LINE_RE.findall(user_prompt)
        assert story_ids, "fake client expected at least one story id in the prompt"

        sentence_count = max(MIN_SENTENCES, min(MAX_SENTENCES, len(story_ids)))
        sentences = []
        for index in range(sentence_count):
            cited_id = story_ids[index % len(story_ids)]
            sentences.append({"text": f"Coverage sentence {index + 1} about the theme.", "citation_ids": [cited_id]})

        return response_schema.model_validate(
            {"label": "Coverage of recent developments", "sentences": sentences}
        )


class AlwaysMalformedGeminiClient:
    """Simulates a provider that never returns a schema-valid response."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        raise ValueError("provider returned non-JSON text")


class FlakyGeminiClient:
    """Fails on the first call, then succeeds like FakeGeminiClient."""

    def __init__(self) -> None:
        self.calls = 0
        self._fake = FakeGeminiClient()

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        if self.calls == 1:
            raise ValueError("transient provider error")
        return self._fake.generate(system_prompt, user_prompt, response_schema)


class InventedCitationGeminiClient:
    """Always cites a story id that was never in the prompt - simulates the
    model hallucinating a citation instead of using a real member story."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        return response_schema.model_validate(
            {
                "label": "Coverage of recent developments",
                "sentences": [
                    {"text": "First sentence about the theme.", "citation_ids": ["not-a-real-story-id"]},
                    {"text": "Second sentence about the theme.", "citation_ids": ["not-a-real-story-id"]},
                ],
            }
        )


class BlankSentenceGeminiClient:
    """Returns a sentence with whitespace-only text, citing a real story id."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        story_ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": "Coverage of recent developments",
                "sentences": [
                    {"text": "   ", "citation_ids": [story_ids[0]]},
                    {"text": "Second sentence about the theme.", "citation_ids": [story_ids[0]]},
                ],
            }
        )


class SummarizeFixtureTests(unittest.TestCase):
    """Covers the issue #65 DoD: valid JSON across all 20 fixture themes."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.themes = load_fixture_themes()

    def test_fixture_file_has_twenty_themes(self) -> None:
        self.assertEqual(len(self.themes), 20)

    def test_all_five_tickers_are_represented(self) -> None:
        tickers = {theme.ticker for theme in self.themes}
        self.assertEqual(tickers, {"TSLA", "NVDA", "AMD", "AAPL", "META"})

    def test_summarize_produces_valid_theme_summary_for_every_fixture(self) -> None:
        for theme in self.themes:
            with self.subTest(theme=theme.ticker, stories=len(theme.member_stories)):
                result = summarize(theme, client=FakeGeminiClient())

                self.assertIsInstance(result, ThemeSummary)
                self.assertLessEqual(len(result.label.split()), MAX_LABEL_WORDS)
                self.assertGreaterEqual(len(result.sentences), MIN_SENTENCES)
                self.assertLessEqual(len(result.sentences), MAX_SENTENCES)

                known_ids = {story.id for story in theme.member_stories}
                for sentence in result.sentences:
                    self.assertGreaterEqual(len(sentence.citation_ids), 1)
                    for citation_id in sentence.citation_ids:
                        self.assertIn(citation_id, known_ids)

    def test_summarize_output_round_trips_through_json(self) -> None:
        theme = self.themes[0]
        result = summarize(theme, client=FakeGeminiClient())
        payload = json.loads(result.model_dump_json())
        self.assertIn("label", payload)
        self.assertIn("sentences", payload)
        for sentence in payload["sentences"]:
            self.assertIn("text", sentence)
            self.assertIn("citation_ids", sentence)


class SummarizeRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = load_fixture_themes()[0]

    def test_retries_once_on_malformed_response_then_succeeds(self) -> None:
        client = FlakyGeminiClient()
        result = summarize(self.theme, client=client)
        self.assertIsInstance(result, ThemeSummary)
        self.assertEqual(client.calls, 2)

    def test_raises_summarization_error_after_exhausting_retries(self) -> None:
        client = AlwaysMalformedGeminiClient()
        with self.assertRaises(SummarizationError):
            summarize(self.theme, client=client)
        self.assertEqual(client.calls, 2)  # initial attempt + one retry

    def test_raises_on_theme_with_no_member_stories(self) -> None:
        empty_theme = ThemeInput(ticker="TSLA", member_stories=[])
        with self.assertRaises(SummarizationError):
            summarize(empty_theme, client=FakeGeminiClient())

    def test_raises_summarization_error_on_invented_citation_id(self) -> None:
        client = InventedCitationGeminiClient()
        with self.assertRaises(SummarizationError):
            summarize(self.theme, client=client)
        self.assertEqual(client.calls, 2)  # initial attempt + one retry

    def test_raises_summarization_error_on_blank_sentence_text(self) -> None:
        client = BlankSentenceGeminiClient()
        with self.assertRaises(SummarizationError):
            summarize(self.theme, client=client)
        self.assertEqual(client.calls, 2)  # initial attempt + one retry


class ResolveCitationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = load_fixture_themes()[0]

    def test_no_unresolved_citations_for_valid_summary(self) -> None:
        result = summarize(self.theme, client=FakeGeminiClient())
        self.assertEqual(resolve_citations(self.theme, result), set())

    def test_flags_citation_id_not_in_member_stories(self) -> None:
        result = summarize(self.theme, client=FakeGeminiClient())
        result.sentences[0].citation_ids.append("not-a-real-story-id")
        self.assertEqual(resolve_citations(self.theme, result), {"not-a-real-story-id"})


class GuardrailPromptAndConfigTests(unittest.TestCase):
    def test_system_prompt_forbids_advice_prediction_causal_and_uncited_claims(self) -> None:
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("advice", lowered)
        self.assertIn("predict", lowered)
        self.assertIn("causal", lowered)
        self.assertIn("citation_ids", lowered)
        self.assertIn("json", lowered)

    def test_generation_config_uses_low_temperature_and_json_schema(self) -> None:
        kwargs = build_generation_config_kwargs(SYSTEM_PROMPT, ThemeSummary)
        self.assertEqual(kwargs["temperature"], DEFAULT_TEMPERATURE)
        self.assertLess(DEFAULT_TEMPERATURE, 0.3)
        self.assertEqual(kwargs["response_mime_type"], "application/json")
        self.assertIs(kwargs["response_schema"], ThemeSummary)
        self.assertEqual(kwargs["system_instruction"], SYSTEM_PROMPT)


@unittest.skipUnless(os.environ.get("GEMINI_API_KEY"), "set GEMINI_API_KEY to run a live smoke test against Gemini")
class LiveGeminiSmokeTest(unittest.TestCase):
    """Optional manual sanity check against the real API. Skipped in CI by default."""

    def test_summarize_against_real_gemini_for_one_fixture(self) -> None:
        theme = load_fixture_themes()[0]
        result = summarize(theme)
        self.assertIsInstance(result, ThemeSummary)


if __name__ == "__main__":
    unittest.main()

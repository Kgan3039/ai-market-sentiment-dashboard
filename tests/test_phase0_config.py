"""Unit checks for the Phase 0 configuration reference matcher."""

import unittest

from tools.validate_phase0_config import ALIASES_PATH, load_yaml, match_outcome, matched_tickers, validate


class Phase0ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.aliases = load_yaml(ALIASES_PATH)

    def test_configuration_is_valid(self) -> None:
        validate()

    def test_positive_match_for_each_phase_zero_ticker(self) -> None:
        cases = {
            "Tesla Inc reports quarterly revenue.": {"TSLA"},
            "NVIDIA announced a new Blackwell GPU.": {"NVDA"},
            "AMD Ryzen processor demand grew.": {"AMD"},
            "Apple Inc released a new iPhone.": {"AAPL"},
            "Meta Platforms expands Reality Labs.": {"META"},
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(matched_tickers(text, self.aliases), expected)

    def test_documented_false_positive_exclusions(self) -> None:
        cases = (
            "Apple pie recipes are popular in autumn.",
            "A meta-analysis reviewed the evidence.",
            "The clinic studies age-related macular degeneration.",
            "A Tesla coil powered the classroom demonstration.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(matched_tickers(text, self.aliases), set())

    def test_generic_ambiguous_terms_need_context(self) -> None:
        cases = (
            "Apple harvests are ahead of schedule.",
            "The meta framework is documented online.",
            "The AMD department published a report.",
            "Tesla is discussed in a history lecture.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(matched_tickers(text, self.aliases), set())

    def test_matcher_uses_phrase_boundaries_not_substrings(self) -> None:
        cases = (
            "Pineapple exports increased this quarter.",
            "The metadata schema was updated.",
            "The preamdifier circuit was replaced.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(matched_tickers(text, self.aliases), set())

    def test_elon_musk_without_tesla_is_not_tsla(self) -> None:
        self.assertEqual(matched_tickers("Elon Musk discussed his other companies.", self.aliases), set())

    def test_multi_ticker_article_is_ambiguous(self) -> None:
        outcome, tickers = match_outcome("Apple Inc and NVIDIA announced a partnership.", self.aliases)
        self.assertEqual(outcome, "ambiguous")
        self.assertEqual(tickers, {"AAPL", "NVDA"})


if __name__ == "__main__":
    unittest.main()

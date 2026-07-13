"""Tests for the Phase 0 approved-copy and banned-language configuration."""

import unittest

from tools.validate_phase0_copy_rules import (
    GENERATED_CONTENT_SCOPES,
    RAW_SOURCE_CONTENT_SCOPES,
    REQUIRED_COPY,
    detected_categories,
    load_copy_deck,
    load_rules,
    validate,
)


class Phase0CopyRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.copy_deck = load_copy_deck()
        cls.rules = load_rules()

    def detected_generated_categories(self, text: str) -> set[str]:
        return detected_categories(text, self.rules, content_scope="generated_summary")

    def test_files_are_valid_and_non_empty(self) -> None:
        validate()
        self.assertTrue(self.copy_deck)
        self.assertTrue(self.rules)

    def test_required_approved_phrases_exist(self) -> None:
        for phrase in REQUIRED_COPY:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.copy_deck)

    def test_direct_recommendations_predictions_and_causal_claims_are_detected(self) -> None:
        cases = {
            "The stock fell because of the earnings report.": "causal_price",
            "This move was driven by weak demand.": "causal_price",
            "The shares will rise next week.": "prediction",
            "The stock is expected to fall.": "prediction",
            "NVIDIA is a buy.": "advisory",
            "Tesla looks like a buy.": "advisory",
            "NVIDIA looks like a strong buy.": "advisory",
            "We recommend buying NVIDIA.": "advisory",
            "We recommend selling Tesla.": "advisory",
            "You should sell Tesla.": "advisory",
            "Buy the stock.": "advisory",
            "Sell the stock.": "advisory",
            "This is a strong buy opportunity.": "advisory",
            "The price target implies the stock will rise.": "advisory",
            "Investors should buy AMD.": "advisory",
            "Investors should hold the stock.": "advisory",
            "Investors should hold NVIDIA.": "advisory",
            "Shares will be higher next week.": "prediction",
            "Shares will move higher.": "prediction",
            "The stock is going to rally.": "prediction",
            "This stock is guaranteed to outperform.": "unsupported_certainty",
            "The model confidence is high.": "model_confidence",
            "There is a 75% probability of moving higher.": "model_confidence",
            "This certainly explains the rally.": "unsupported_certainty",
        }
        for text, expected_category in cases.items():
            with self.subTest(text=text):
                self.assertIn(expected_category, self.detected_generated_categories(text))

    def test_attributed_analyst_ratings_and_price_target_changes_are_allowed(self) -> None:
        examples = (
            "Morgan Stanley upgraded NVIDIA to Buy.",
            "The analyst lowered Tesla from Buy to Hold.",
            "Goldman Sachs downgraded Tesla to Hold.",
            "The firm raised its price target from $180 to $210.",
            "Several analysts published bullish price targets.",
            "Several firms published Buy ratings.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(self.detected_generated_categories(text), set())

    def test_approved_descriptive_examples_are_not_detected(self) -> None:
        examples = (
            "Themes dominating current coverage",
            "Key narratives around today’s move",
            "Coverage today is dominated by: AI infrastructure investment",
            "The most-covered storyline is: product releases",
            "Data as of 10:30",
            "Summary unavailable — source stories are still available",
            "AI-generated from cited sources. Informational only — not investment advice.",
        )
        for text in examples:
            with self.subTest(text=text):
                self.assertEqual(self.detected_generated_categories(text), set())

    def test_approved_copy_avoids_causal_certainty(self) -> None:
        approved_strings = (
            "Coverage today is dominated by: {theme}",
            "The most-covered storyline is: {theme}",
            "Data may be delayed. Last successful update: HH:MM.",
        )
        for text in approved_strings:
            with self.subTest(text=text):
                self.assertEqual(self.detected_generated_categories(text), set())

    def test_raw_publisher_text_is_outside_the_rule_scope(self) -> None:
        source_text = "The stock fell because of the earnings report."
        self.assertIn("causal_price", self.detected_generated_categories(source_text))
        for content_scope in RAW_SOURCE_CONTENT_SCOPES:
            with self.subTest(content_scope=content_scope):
                self.assertEqual(
                    detected_categories(source_text, self.rules, content_scope=content_scope), set()
                )

    def test_generated_scopes_are_the_only_scopes_where_rules_apply(self) -> None:
        for content_scope in GENERATED_CONTENT_SCOPES:
            with self.subTest(content_scope=content_scope):
                self.assertIn(
                    "advisory",
                    detected_categories("NVIDIA is a buy.", self.rules, content_scope=content_scope),
                )


if __name__ == "__main__":
    unittest.main()

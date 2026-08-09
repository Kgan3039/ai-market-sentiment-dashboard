"""Tests for the Phase 0 review-sampling and scorecard tooling (issue #74 / A4)."""

from __future__ import annotations

import csv
import re
import tempfile
import unittest
from pathlib import Path

from nlp.eval.review import (
    AssignmentSample,
    ReviewSamplingError,
    SentenceSample,
    load_theme_sets,
    read_csv_rows,
    sample_assignments,
    sample_summary_sentences,
    score_assignments,
    score_summaries,
    write_assignment_csv,
    write_sentence_csv,
)
from nlp.eval.trust import DatasetKind, LabelingStatus, MetricsPurpose, TrustContract, TrustContractError

ID_LINE_RE = re.compile(r"- id: (\S+)")


class FakeGeminiClient:
    """Deterministic stand-in for ai.summarization.GeminiClient.generate.

    Same trick as tests/test_ai_summarization.py's fake: extracts the real
    story ids serialized into the prompt and cites only those, so it works
    generically across every theme without hardcoding per-theme responses.
    """

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        story_ids = ID_LINE_RE.findall(user_prompt)
        assert story_ids, "fake client expected at least one story id in the prompt"
        sentences = [
            {"text": f"Coverage sentence {index + 1} about the theme.", "citation_ids": [story_id]}
            for index, story_id in enumerate(story_ids[:2])
        ]
        if len(sentences) < 2:
            sentences.append({"text": "Additional coverage sentence.", "citation_ids": [story_ids[0]]})
        return response_schema.model_validate({"label": "Coverage of recent developments", "sentences": sentences})


def _fill_verdicts(path: Path, verdicts: dict[str, str], *, verdict_field: str = "reviewer_verdict") -> None:
    rows = read_csv_rows(path)
    fieldnames = list(rows[0].keys())
    for row in rows:
        if row["row_id"] in verdicts:
            row[verdict_field] = verdicts[row["row_id"]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_adjudicated(path: Path, verdicts: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "final_verdict"])
        writer.writeheader()
        for row_id, verdict in verdicts.items():
            writer.writerow({"row_id": row_id, "final_verdict": verdict})


class LoadThemeSetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.day_set, cls.theme_sets = load_theme_sets()

    def test_loads_all_three_committed_ticker_days(self) -> None:
        self.assertEqual(len(self.theme_sets), 3)
        tickers = {ticker for ticker, _ in self.theme_sets}
        self.assertEqual(tickers, {"AAPL", "NVDA", "TSLA"})

    def test_every_theme_set_accounts_for_every_story(self) -> None:
        for theme_set in self.theme_sets.values():
            self.assertTrue(theme_set.complete)

    def test_thirty_stories_total_across_all_days(self) -> None:
        total = sum(len(theme_set.accounted_story_keys) for theme_set in self.theme_sets.values())
        self.assertEqual(total, 30)


class SampleAssignmentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.theme_sets = load_theme_sets()

    def test_population_size_matches_total_stories(self) -> None:
        sample = sample_assignments(self.theme_sets, sample_size=1000)
        self.assertEqual(sample.population_size, 30)

    def test_takes_full_population_when_requested_exceeds_it(self) -> None:
        sample = sample_assignments(self.theme_sets, sample_size=40)
        self.assertEqual(sample.requested_sample_size, 40)
        self.assertEqual(sample.population_size, 30)
        self.assertEqual(sample.actual_sample_size, 30)
        self.assertEqual(len(sample.rows), 30)

    def test_deterministic_for_the_same_seed(self) -> None:
        first = sample_assignments(self.theme_sets, sample_size=10, seed="fixed-seed")
        second = sample_assignments(self.theme_sets, sample_size=10, seed="fixed-seed")
        self.assertEqual([row.row_id for row in first.rows], [row.row_id for row in second.rows])

    def test_different_seeds_can_produce_different_samples(self) -> None:
        first = sample_assignments(self.theme_sets, sample_size=10, seed="seed-a")
        second = sample_assignments(self.theme_sets, sample_size=10, seed="seed-b")
        self.assertNotEqual(
            [row.row_id for row in first.rows], [row.row_id for row in second.rows]
        )

    def test_full_sample_partitions_into_theme_other_coverage_and_excluded(self) -> None:
        sample = sample_assignments(self.theme_sets, sample_size=1000)
        assignment_types = {row.assignment_type for row in sample.rows}
        self.assertLessEqual(assignment_types, {"theme", "other_coverage", "excluded"})
        row_ids = {row.row_id for row in sample.rows}
        self.assertEqual(len(row_ids), len(sample.rows))  # no duplicate row_ids

    def test_theme_rows_carry_theme_identity_other_rows_do_not(self) -> None:
        sample = sample_assignments(self.theme_sets, sample_size=1000)
        for row in sample.rows:
            if row.assignment_type == "theme":
                self.assertNotEqual(row.theme_key, "")
                self.assertEqual(row.reason, "")
            else:
                self.assertEqual(row.theme_key, "")
                self.assertNotEqual(row.reason, "")


class AssignmentCsvTests(unittest.TestCase):
    def test_write_and_read_round_trip(self) -> None:
        _, theme_sets = load_theme_sets()
        sample = sample_assignments(theme_sets, sample_size=5, seed="csv-test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assignments.csv"
            write_assignment_csv(sample, path)
            rows = read_csv_rows(path)
            self.assertEqual(len(rows), 5)
            self.assertEqual(
                set(rows[0].keys()),
                {
                    "row_id", "ticker", "trading_day", "story_key", "story_title",
                    "story_outlets", "assignment_type", "theme_key", "theme_label",
                    "reason", "reviewer_verdict", "reviewer_notes",
                },
            )
            self.assertEqual(rows[0]["reviewer_verdict"], "")


class SampleSummarySentencesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.theme_sets = load_theme_sets()

    def test_samples_requested_number_of_days(self) -> None:
        sample = sample_summary_sentences(self.theme_sets, client=FakeGeminiClient(), days=2, seed="days-seed")
        self.assertEqual(sample.requested_days, 2)
        self.assertEqual(sample.actual_days, 2)
        self.assertEqual(sample.day_population_size, 3)
        self.assertEqual(len(sample.sampled_days), 2)

    def test_caps_days_at_available_population(self) -> None:
        sample = sample_summary_sentences(self.theme_sets, client=FakeGeminiClient(), days=10, seed="days-seed")
        self.assertEqual(sample.actual_days, 3)

    def test_deterministic_for_the_same_seed(self) -> None:
        first = sample_summary_sentences(self.theme_sets, client=FakeGeminiClient(), days=2, seed="fixed")
        second = sample_summary_sentences(self.theme_sets, client=FakeGeminiClient(), days=2, seed="fixed")
        self.assertEqual(first.sampled_days, second.sampled_days)
        self.assertEqual([row.row_id for row in first.rows], [row.row_id for row in second.rows])

    def test_every_citation_resolves_to_a_real_story_in_its_own_theme(self) -> None:
        sample = sample_summary_sentences(self.theme_sets, client=FakeGeminiClient(), days=3, seed="all-days")
        self.assertGreater(len(sample.rows), 0)
        for row in sample.rows:
            self.assertNotIn("", row.cited_story_titles.split(";"))
            for title in row.cited_story_titles.split(";"):
                self.assertNotEqual(title, "")

    def test_requires_an_explicit_client(self) -> None:
        with self.assertRaises(TypeError):
            sample_summary_sentences(self.theme_sets, days=1)  # missing required client kwarg


class SentenceCsvTests(unittest.TestCase):
    def test_write_and_read_round_trip(self) -> None:
        _, theme_sets = load_theme_sets()
        # days=3 (all committed days) guarantees at least one themed day
        # (NVDA/TSLA) is included; a day below M5's 4-story clustering floor
        # (AAPL) legitimately produces zero themes and thus zero sentences.
        sample = sample_summary_sentences(theme_sets, client=FakeGeminiClient(), days=3, seed="csv-test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summaries.csv"
            write_sentence_csv(sample, path)
            rows = read_csv_rows(path)
            self.assertEqual(len(rows), len(sample.rows))
            self.assertGreater(len(rows), 0)
            self.assertEqual(
                set(rows[0].keys()),
                {
                    "row_id", "ticker", "trading_day", "theme_key", "theme_label",
                    "sentence_index", "sentence_text", "citation_ids",
                    "cited_story_titles", "cited_outlets", "reviewer_verdict", "reviewer_notes",
                },
            )


class ScoreAssignmentsTests(unittest.TestCase):
    def setUp(self) -> None:
        _, theme_sets = load_theme_sets()
        self.sample = sample_assignments(theme_sets, sample_size=10, seed="score-test")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sheet_path = Path(self.tmp.name) / "sheet.csv"
        write_assignment_csv(self.sample, self.sheet_path)
        self.row_ids = [row.row_id for row in self.sample.rows]

    def test_single_reviewer_rate_matches_marked_verdicts(self) -> None:
        verdicts = {row_id: "correct" for row_id in self.row_ids[:8]}
        verdicts.update({row_id: "incorrect" for row_id in self.row_ids[8:]})
        _fill_verdicts(self.sheet_path, verdicts)

        scorecard = score_assignments([self.sheet_path])
        self.assertEqual(scorecard.reviewer_count, 1)
        self.assertIsNone(scorecard.agreement_rate)
        self.assertEqual(scorecard.resolved_count, 10)
        self.assertEqual(scorecard.unresolved_count, 0)
        self.assertAlmostEqual(scorecard.rate, 0.8)
        self.assertEqual(scorecard.trust_contract.labeling_status, LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED)
        self.assertFalse(scorecard.trust_contract.gate_eligible)

    def test_meets_gate_true_only_above_threshold_with_no_unresolved(self) -> None:
        _fill_verdicts(self.sheet_path, {row_id: "correct" for row_id in self.row_ids})
        scorecard = score_assignments([self.sheet_path], threshold=0.75)
        self.assertEqual(scorecard.rate, 1.0)
        self.assertTrue(scorecard.meets_gate)

        below = score_assignments([self.sheet_path], threshold=1.01)
        self.assertFalse(below.meets_gate)

    def test_blank_verdicts_are_unresolved_and_excluded_from_rate(self) -> None:
        verdicts = {row_id: "correct" for row_id in self.row_ids[:5]}
        _fill_verdicts(self.sheet_path, verdicts)  # remaining 5 rows stay blank
        scorecard = score_assignments([self.sheet_path])
        self.assertEqual(scorecard.resolved_count, 5)
        self.assertEqual(scorecard.unresolved_count, 5)
        self.assertEqual(scorecard.rate, 1.0)
        self.assertFalse(scorecard.meets_gate)  # unresolved rows block the gate

    def test_two_agreeing_reviewers_yield_full_agreement_rate(self) -> None:
        sheet_2 = Path(self.tmp.name) / "sheet2.csv"
        write_assignment_csv(self.sample, sheet_2)
        verdicts = {row_id: "correct" for row_id in self.row_ids}
        _fill_verdicts(self.sheet_path, verdicts)
        _fill_verdicts(sheet_2, verdicts)

        scorecard = score_assignments([self.sheet_path, sheet_2])
        self.assertEqual(scorecard.reviewer_count, 2)
        self.assertEqual(scorecard.agreement_rate, 1.0)
        self.assertEqual(scorecard.rate, 1.0)
        self.assertEqual(scorecard.unresolved_count, 0)
        self.assertEqual(
            scorecard.trust_contract.labeling_status, LabelingStatus.MULTI_REVIEWER_UNADJUDICATED
        )
        self.assertFalse(scorecard.trust_contract.adjudicated)

    def test_disagreement_without_adjudication_is_unresolved(self) -> None:
        sheet_2 = Path(self.tmp.name) / "sheet2.csv"
        write_assignment_csv(self.sample, sheet_2)
        verdicts_1 = {row_id: "correct" for row_id in self.row_ids}
        verdicts_2 = dict(verdicts_1)
        verdicts_2[self.row_ids[0]] = "incorrect"  # one disagreement
        _fill_verdicts(self.sheet_path, verdicts_1)
        _fill_verdicts(sheet_2, verdicts_2)

        scorecard = score_assignments([self.sheet_path, sheet_2])
        self.assertEqual(scorecard.unresolved_count, 1)
        self.assertIn(self.row_ids[0], scorecard.unresolved_row_ids)
        self.assertEqual(scorecard.resolved_count, 9)
        self.assertAlmostEqual(scorecard.agreement_rate, 9 / 10)

    def test_adjudication_resolves_a_disagreement(self) -> None:
        sheet_2 = Path(self.tmp.name) / "sheet2.csv"
        write_assignment_csv(self.sample, sheet_2)
        verdicts_1 = {row_id: "correct" for row_id in self.row_ids}
        verdicts_2 = dict(verdicts_1)
        verdicts_2[self.row_ids[0]] = "incorrect"
        _fill_verdicts(self.sheet_path, verdicts_1)
        _fill_verdicts(sheet_2, verdicts_2)

        adjudicated_path = Path(self.tmp.name) / "adjudicated.csv"
        _write_adjudicated(adjudicated_path, {self.row_ids[0]: "correct"})

        scorecard = score_assignments([self.sheet_path, sheet_2], adjudicated=adjudicated_path)
        self.assertEqual(scorecard.unresolved_count, 0)
        self.assertEqual(scorecard.resolved_count, 10)
        self.assertEqual(scorecard.rate, 1.0)
        self.assertEqual(
            scorecard.trust_contract.labeling_status, LabelingStatus.MULTI_REVIEWER_ADJUDICATED
        )
        self.assertTrue(scorecard.trust_contract.adjudicated)

    def test_mismatched_row_ids_between_sheets_raises(self) -> None:
        # A different sample_size guarantees a different row_id set
        # regardless of RNG outcome, so this failure mode is deterministic.
        _, theme_sets = load_theme_sets()
        sheet_2 = Path(self.tmp.name) / "sheet2.csv"
        other_sample = sample_assignments(theme_sets, sample_size=9, seed="different-seed")
        write_assignment_csv(other_sample, sheet_2)
        with self.assertRaises(ReviewSamplingError):
            score_assignments([self.sheet_path, sheet_2])

    def test_invalid_verdict_value_raises(self) -> None:
        _fill_verdicts(self.sheet_path, {self.row_ids[0]: "maybe"})
        with self.assertRaises(ReviewSamplingError):
            score_assignments([self.sheet_path])

    def test_never_produces_a_gate_eligible_scorecard(self) -> None:
        _fill_verdicts(self.sheet_path, {row_id: "correct" for row_id in self.row_ids})
        scorecard = score_assignments([self.sheet_path], threshold=0.0)
        self.assertTrue(scorecard.meets_gate)  # the metric itself clears its threshold
        self.assertFalse(scorecard.trust_contract.gate_eligible)  # but is still not gate-eligible


class ScoreSummariesTests(unittest.TestCase):
    def test_faithfulness_rate_and_threshold(self) -> None:
        _, theme_sets = load_theme_sets()
        # days=3 guarantees a themed day is included regardless of seed; a
        # day below M5's clustering floor (AAPL) has zero themes to sample.
        sample = sample_summary_sentences(theme_sets, client=FakeGeminiClient(), days=3, seed="summary-score")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sentences.csv"
            write_sentence_csv(sample, path)
            row_ids = [row.row_id for row in sample.rows]
            self.assertGreater(len(row_ids), 1)
            verdicts = {row_id: "supported" for row_id in row_ids}
            if len(row_ids) > 1:
                verdicts[row_ids[-1]] = "not_supported"
            _fill_verdicts(path, verdicts)

            scorecard = score_summaries([path], threshold=0.95)
            expected_rate = (len(row_ids) - (1 if len(row_ids) > 1 else 0)) / len(row_ids)
            self.assertAlmostEqual(scorecard.rate, expected_rate)
            self.assertEqual(scorecard.meets_gate, expected_rate >= 0.95)


class TrustContractSafetyNetTests(unittest.TestCase):
    """Proves the gate-eligibility refusal is structural, not a convention this
    module merely follows -- nlp.eval.trust itself refuses the combination."""

    def test_synthetic_dataset_kind_cannot_claim_gate_eligible(self) -> None:
        with self.assertRaises(TrustContractError):
            TrustContract(
                dataset_kind=DatasetKind.SYNTHETIC_DEVELOPMENT,
                real_ingested_evidence=False,
                labeling_status=LabelingStatus.MULTI_REVIEWER_ADJUDICATED,
                reviewer_count=2,
                adjudicated=True,
                gate_eligible=True,
                metrics_purpose=MetricsPurpose.GATE_ACCEPTANCE,
            )


if __name__ == "__main__":
    unittest.main()

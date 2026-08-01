"""Tests for the M4 labeled dedup set and its evaluator (issue #67).

Two things are under test and they are different things: that the *loader*
refuses a dataset it cannot vouch for, and that the *scorer* is arithmetic
nobody can nudge. The committed set is also checked as data — its
composition, its provenance marking, and the M2 baseline it produces.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from nlp.eval import (
    SUPPORTED_SCHEMA_VERSION,
    EvalDatasetError,
    PairPrediction,
    default_pair_set,
    evaluate,
    evaluate_m2,
    load_pair_set,
    sweep_thresholds,
)
from nlp.eval.dataset import DEFAULT_META_PATH
from nlp.eval.dedup import config_for, m2_predictor, to_raw_items
from nlp.eval.metrics import Confusion, Metrics
from nlp.eval.report import render_sweep, render_text, to_payload
from tools import eval_dedup

REPO_ROOT = Path(__file__).resolve().parents[1]

_MINIMAL_META = {
    "schema_version": SUPPORTED_SCHEMA_VERSION,
    "dataset_id": "test-set",
    "pairs_file": "pairs.jsonl",
    "tickers": ["NVDA", "TSLA"],
    "labels": ["duplicate", "distinct", "ambiguous"],
    "expected_stages": ["m2", "m3", "none"],
    "confidences": ["high", "medium", "low"],
    "categories": ["exact_duplicate", "semantic_rewrite", "hard_negative", "ambiguous"],
    "provenance": {"kind": "synthetic"},
    "labeling": {"status": "test"},
}


def _pair(pair_id: str, **overrides: object) -> dict:
    payload: dict = {
        "pair_id": pair_id,
        "ticker": "NVDA",
        "label": "duplicate",
        "category": "exact_duplicate",
        "expected_stage": "m2",
        "rationale": "identical text",
        "confidence": "high",
        "item_a": {
            "item_id": f"{pair_id}-a",
            "title": "Nvidia reports record revenue",
            "description": "The chipmaker beat estimates.",
            "url": "https://example.test/a",
            "source": "Reuters",
            "published_at": "2026-03-02T13:00:00+00:00",
        },
        "item_b": {
            "item_id": f"{pair_id}-b",
            "title": "Nvidia reports record revenue",
            "description": "The chipmaker beat estimates.",
            "url": "https://example.test/b",
            "source": "CNBC",
            "published_at": "2026-03-02T13:30:00+00:00",
        },
    }
    payload.update(overrides)
    return payload


def write_set(tmp_path: Path, pairs, meta_overrides: dict | None = None) -> Path:
    """Write a manifest plus pairs file and return the manifest path."""

    meta = dict(_MINIMAL_META)
    meta.update(meta_overrides or {})
    meta_path = tmp_path / "set.meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / str(meta["pairs_file"])).write_text(
        "\n".join(json.dumps(pair) for pair in pairs) + "\n", encoding="utf-8"
    )
    return meta_path


# --------------------------------------------------------------------------
# Dataset schema validation
# --------------------------------------------------------------------------


def test_a_well_formed_set_loads(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001"), _pair("P002")]))

    assert [pair.pair_id for pair in pair_set] == ["P001", "P002"]
    assert pair_set.dataset_id == "test-set"
    assert len(pair_set) == 2


def test_an_unknown_schema_version_is_refused(tmp_path):
    meta_path = write_set(
        tmp_path, [_pair("P001")], {"schema_version": "phase0.dedup_eval.v99"}
    )

    with pytest.raises(EvalDatasetError, match="unsupported schema_version"):
        load_pair_set(meta_path)


def test_a_manifest_missing_a_required_key_is_refused(tmp_path):
    meta = dict(_MINIMAL_META)
    del meta["categories"]
    meta_path = tmp_path / "set.meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text(json.dumps(_pair("P001")), encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="missing"):
        load_pair_set(meta_path)


def test_a_repeated_vocabulary_value_is_refused(tmp_path):
    meta_path = write_set(
        tmp_path, [_pair("P001")], {"tickers": ["NVDA", "NVDA", "TSLA"]}
    )

    with pytest.raises(EvalDatasetError, match="repeats a value"):
        load_pair_set(meta_path)


def test_duplicate_pair_ids_are_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001"), _pair("P001")])

    with pytest.raises(EvalDatasetError, match="duplicate pair_id: P001"):
        load_pair_set(meta_path)


def test_an_item_id_reused_across_pairs_is_refused(tmp_path):
    second = _pair("P002")
    second["item_a"] = dict(second["item_a"], item_id="P001-a")
    meta_path = write_set(tmp_path, [_pair("P001"), second])

    with pytest.raises(EvalDatasetError, match="duplicate item_id: P001-a"):
        load_pair_set(meta_path)


def test_both_sides_sharing_one_item_id_is_refused(tmp_path):
    pair = _pair("P001")
    pair["item_b"] = dict(pair["item_b"], item_id="P001-a")
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="both sides share item_id"):
        load_pair_set(meta_path)


def test_an_invalid_label_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", label="maybe")])

    with pytest.raises(EvalDatasetError, match="label='maybe' is not one of"):
        load_pair_set(meta_path)


def test_an_invalid_category_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", category="typo")])

    with pytest.raises(EvalDatasetError, match="category='typo'"):
        load_pair_set(meta_path)


def test_an_unsupported_ticker_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", ticker="AMZN")])

    with pytest.raises(EvalDatasetError, match="ticker='AMZN'"):
        load_pair_set(meta_path)


@pytest.mark.parametrize(
    "label,stage",
    [("duplicate", "none"), ("distinct", "m2"), ("ambiguous", "m3")],
)
def test_a_label_contradicting_its_expected_stage_is_refused(tmp_path, label, stage):
    meta_path = write_set(tmp_path, [_pair("P001", label=label, expected_stage=stage)])

    with pytest.raises(EvalDatasetError, match="incompatible with expected_stage"):
        load_pair_set(meta_path)


def test_a_missing_pair_field_is_refused(tmp_path):
    pair = _pair("P001")
    del pair["rationale"]
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match=r"missing field\(s\) \['rationale'\]"):
        load_pair_set(meta_path)


def test_an_unknown_pair_field_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", notes="extra")])

    with pytest.raises(EvalDatasetError, match=r"unknown field\(s\) \['notes'\]"):
        load_pair_set(meta_path)


def test_an_unknown_item_field_is_refused(tmp_path):
    pair = _pair("P001")
    pair["item_a"] = dict(pair["item_a"], outlet="Reuters")
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match=r"unknown field\(s\) \['outlet'\]"):
        load_pair_set(meta_path)


def test_a_blank_rationale_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", rationale="   ")])

    with pytest.raises(EvalDatasetError, match="rationale must be a non-blank string"):
        load_pair_set(meta_path)


def test_an_empty_optional_field_is_refused_rather_than_read_as_absent(tmp_path):
    pair = _pair("P001")
    pair["item_a"] = dict(pair["item_a"], description="")
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="omit the key"):
        load_pair_set(meta_path)


def test_a_naive_timestamp_is_refused_at_load_time(tmp_path):
    pair = _pair("P001")
    pair["item_a"] = dict(pair["item_a"], published_at="2026-03-02T13:00:00")
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="must carry a timezone offset"):
        load_pair_set(meta_path)


def test_an_unparseable_timestamp_is_refused(tmp_path):
    pair = _pair("P001")
    pair["item_a"] = dict(pair["item_a"], published_at="last tuesday")
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="not an ISO-8601 timestamp"):
        load_pair_set(meta_path)


def test_a_malformed_json_row_is_refused_with_its_line_number(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001")])
    (tmp_path / "pairs.jsonl").write_text(
        json.dumps(_pair("P001")) + "\n{not json}\n", encoding="utf-8"
    )

    with pytest.raises(EvalDatasetError, match="line 2: not valid JSON"):
        load_pair_set(meta_path)


def test_a_row_that_is_not_an_object_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001")])
    (tmp_path / "pairs.jsonl").write_text('["P001"]\n', encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="must be a JSON object"):
        load_pair_set(meta_path)


def test_an_empty_pairs_file_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001")])
    (tmp_path / "pairs.jsonl").write_text("\n\n", encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="holds no pairs"):
        load_pair_set(meta_path)


def test_a_missing_manifest_is_reported_clearly(tmp_path):
    with pytest.raises(EvalDatasetError, match="manifest not found"):
        load_pair_set(tmp_path / "absent.meta.json")


def test_a_missing_pairs_file_is_reported_clearly(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001")])
    (tmp_path / "pairs.jsonl").unlink()

    with pytest.raises(EvalDatasetError, match="pairs file not found"):
        load_pair_set(meta_path)


# --------------------------------------------------------------------------
# Deterministic ordering
# --------------------------------------------------------------------------


def test_rows_out_of_pair_id_order_are_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P002"), _pair("P001")])

    with pytest.raises(EvalDatasetError, match="must be sorted by pair_id"):
        load_pair_set(meta_path)


def test_the_committed_set_is_sorted_and_stable():
    pair_set = default_pair_set()
    identifiers = [pair.pair_id for pair in pair_set]

    assert identifiers == sorted(identifiers)
    assert identifiers == [pair.pair_id for pair in load_pair_set(DEFAULT_META_PATH)]


def test_blank_lines_are_tolerated_but_do_not_shift_line_numbers(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001")])
    (tmp_path / "pairs.jsonl").write_text(
        "\n" + json.dumps(_pair("P001")) + "\n\n", encoding="utf-8"
    )

    assert len(load_pair_set(meta_path)) == 1


# --------------------------------------------------------------------------
# Evaluator arithmetic
# --------------------------------------------------------------------------


def constant(merged: bool, **kwargs):
    return lambda pair: PairPrediction(merged=merged, **kwargs)


def test_a_perfect_predictor_scores_one(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001"),
                _pair(
                    "P002",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                ),
            ],
        )
    )

    report = evaluate(
        pair_set,
        lambda pair: PairPrediction(merged=pair.is_positive),
        name="oracle",
    )

    assert report.overall.precision == 1.0
    assert report.overall.recall == 1.0
    assert report.overall.f1 == 1.0
    assert report.overall.accuracy == 1.0


def test_confusion_cells_hold_the_pair_ids(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001"),
                _pair("P002"),
                _pair(
                    "P003",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                ),
                _pair(
                    "P004",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                ),
            ],
        )
    )

    # Merge P001 and P003: one right, one wrong.
    report = evaluate(
        pair_set,
        lambda pair: PairPrediction(merged=pair.pair_id in {"P001", "P003"}),
        name="partial",
    )
    confusion = report.overall.confusion

    assert confusion.true_positives == ("P001",)
    assert confusion.false_negatives == ("P002",)
    assert confusion.false_positives == ("P003",)
    assert confusion.true_negatives == ("P004",)
    assert report.overall.precision == 0.5
    assert report.overall.recall == 0.5
    assert report.overall.f1 == 0.5
    assert report.overall.accuracy == 0.5


def test_ambiguous_pairs_are_excluded_and_reported_separately(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001"),
                _pair(
                    "P002",
                    label="ambiguous",
                    category="ambiguous",
                    expected_stage="none",
                ),
            ],
        )
    )

    report = evaluate(pair_set, constant(True), name="merge-everything")

    assert report.overall.confusion.total == 1
    assert report.overall.precision == 1.0
    assert report.ambiguous_count == 1
    assert report.ambiguous_merged == ("P002",)


def test_candidate_recall_bounds_merge_recall(tmp_path):
    pair_set = load_pair_set(
        write_set(tmp_path, [_pair("P001"), _pair("P002"), _pair("P003")])
    )

    # Considered all three, merged only one.
    report = evaluate(
        pair_set,
        lambda pair: PairPrediction(merged=pair.pair_id == "P001", candidate=True),
        name="picky",
    )

    assert report.candidate_recall == 1.0
    assert report.merge_recall == pytest.approx(1 / 3)


def test_breakdowns_partition_the_scored_pairs(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001", ticker="NVDA"),
                _pair(
                    "P002",
                    ticker="TSLA",
                    category="semantic_rewrite",
                    expected_stage="m3",
                ),
                _pair(
                    "P003",
                    ticker="TSLA",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                ),
            ],
        )
    )

    report = evaluate(pair_set, constant(True), name="all")

    assert {entry.key for entry in report.by_ticker} == {"NVDA", "TSLA"}
    assert sum(entry.metrics.confusion.total for entry in report.by_ticker) == 3
    assert sum(entry.metrics.confusion.total for entry in report.by_category) == 3
    assert sum(entry.metrics.confusion.total for entry in report.by_expected_stage) == 3


def test_a_predictor_returning_the_wrong_type_is_rejected(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))

    def not_a_prediction(pair):
        return True

    with pytest.raises(TypeError, match="expected PairPrediction"):
        evaluate(pair_set, not_a_prediction, name="bad")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Zero-denominator cases
# --------------------------------------------------------------------------


def test_precision_is_undefined_rather_than_zero_when_nothing_is_merged(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))

    report = evaluate(pair_set, constant(False), name="merge-nothing")

    assert report.overall.precision is None
    assert report.overall.recall == 0.0
    assert report.overall.f1 is None


def test_recall_is_undefined_when_the_slice_holds_no_positives(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair(
                    "P001",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                )
            ],
        )
    )

    report = evaluate(pair_set, constant(False), name="merge-nothing")

    assert report.overall.recall is None
    assert report.overall.precision is None
    assert report.overall.accuracy == 1.0


def test_an_empty_confusion_yields_no_numbers_at_all():
    metrics = Metrics.from_confusion(Confusion())

    assert (metrics.precision, metrics.recall, metrics.f1, metrics.accuracy) == (
        None,
        None,
        None,
        None,
    )


def test_f1_is_undefined_when_precision_and_recall_are_both_zero():
    metrics = Metrics.from_confusion(
        Confusion(false_positives=("A",), false_negatives=("B",))
    )

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 is None


def test_an_undefined_metric_never_clears_a_gate():
    undefined = Metrics.from_confusion(Confusion(true_negatives=("A",)))
    passing = Metrics.from_confusion(Confusion(true_positives=("A",)))

    assert not undefined.meets(precision_floor=0.0, recall_floor=0.0)
    assert passing.meets(precision_floor=1.0, recall_floor=1.0)
    assert not passing.meets(precision_floor=1.0, recall_floor=1.01)


# --------------------------------------------------------------------------
# Threshold sweeps
# --------------------------------------------------------------------------


def scored_set(tmp_path: Path):
    """Three positives and one negative, each with a fixed similarity."""

    return load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001"),
                _pair("P002"),
                _pair("P003"),
                _pair(
                    "P004",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                ),
            ],
        )
    )


SIMILARITY = {"P001": 0.95, "P002": 0.85, "P003": 0.60, "P004": 0.88}


def threshold_predictor(threshold: float):
    def predict(pair):
        score = SIMILARITY[pair.pair_id]
        return PairPrediction(
            merged=score >= threshold, score=score, candidate=True, stage="fake"
        )

    return predict


def test_a_sweep_trades_recall_for_precision(tmp_path):
    points = sweep_thresholds(
        scored_set(tmp_path), threshold_predictor, [0.9, 0.5, 0.87], name="fake"
    )

    assert [point.threshold for point in points] == [0.5, 0.87, 0.9]
    # 0.5 merges everything: all three positives and the negative.
    assert points[0].recall == 1.0
    assert points[0].precision == 0.75
    # 0.87 drops the 0.85 positive but still merges the 0.88 negative.
    assert points[1].recall == pytest.approx(1 / 3)
    assert points[1].precision == 0.5
    # 0.9 keeps only the 0.95 positive: perfect precision, worse recall.
    assert points[2].precision == 1.0
    assert points[2].recall == pytest.approx(1 / 3)


def test_a_sweep_is_returned_in_ascending_order_however_it_was_requested(tmp_path):
    pair_set = scored_set(tmp_path)
    forward = sweep_thresholds(pair_set, threshold_predictor, [0.5, 0.9], name="fake")
    backward = sweep_thresholds(pair_set, threshold_predictor, [0.9, 0.5], name="fake")

    assert [point.threshold for point in forward] == [
        point.threshold for point in backward
    ]
    assert forward[0].precision == backward[0].precision


def test_a_sweep_rejects_repeated_thresholds(tmp_path):
    with pytest.raises(ValueError, match="must be distinct"):
        sweep_thresholds(
            scored_set(tmp_path), threshold_predictor, [0.8, 0.8], name="fake"
        )


def test_a_sweep_needs_at_least_one_threshold(tmp_path):
    with pytest.raises(ValueError, match="at least one threshold"):
        sweep_thresholds(scored_set(tmp_path), threshold_predictor, [], name="fake")


def test_sweep_points_carry_the_threshold_they_were_scored_at(tmp_path):
    points = sweep_thresholds(
        scored_set(tmp_path), threshold_predictor, [0.9], name="f"
    )

    assert points[0].report.threshold == 0.9
    assert points[0].report.predictor == "f@0.9"
    assert points[0].report.scores["P001"] == 0.95


def test_the_sweep_table_renders_every_point(tmp_path):
    rendered = render_sweep(
        sweep_thresholds(
            scored_set(tmp_path), threshold_predictor, [0.5, 0.9], name="f"
        )
    )

    assert "threshold" in rendered
    assert rendered.count("\n") == 3


# --------------------------------------------------------------------------
# Failure-detail output
# --------------------------------------------------------------------------


def test_the_text_report_names_every_failure_with_its_rationale(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001", rationale="a duplicate the stage missed"),
                _pair(
                    "P002",
                    label="distinct",
                    category="hard_negative",
                    expected_stage="none",
                    rationale="two different events",
                ),
            ],
        )
    )
    report = evaluate(
        pair_set,
        lambda pair: PairPrediction(
            merged=not pair.is_positive, detail=f"decision for {pair.pair_id}"
        ),
        name="inverted",
    )

    rendered = render_text(
        report, rationales={pair.pair_id: pair.rationale for pair in pair_set}
    )

    assert "false positives (merged, must not have been): 1" in rendered
    assert "false negatives (not merged, should have been): 1" in rendered
    assert "P001  decision for P001" in rendered
    assert "a duplicate the stage missed" in rendered
    assert "two different events" in rendered


def test_undefined_metrics_render_as_not_available(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))

    rendered = render_text(evaluate(pair_set, constant(False), name="none"))

    assert "precision        n/a" in rendered
    assert "recall           0.0000" in rendered


def test_the_json_payload_is_sortable_and_complete(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))
    payload = to_payload(evaluate(pair_set, constant(True), name="all"))

    assert payload["overall"]["counts"]["true_positive"] == 1
    assert payload["overall"]["pair_ids"]["true_positive"] == ["P001"]
    assert payload["dataset_id"] == "test-set"
    # Serializable without a custom encoder: it is committed to the repo.
    json.dumps(payload, sort_keys=True)


# --------------------------------------------------------------------------
# The M2 baseline over the committed set
# --------------------------------------------------------------------------


def test_the_committed_set_is_marked_synthetic_and_names_its_blockers():
    metadata = default_pair_set().metadata
    provenance = metadata["provenance"]

    assert provenance["kind"] == "synthetic"
    assert set(provenance["blocked_by"]) == {"#60", "#61", "#62"}
    assert metadata["labeling"]["adjudicated"] is False


def test_the_committed_set_has_the_composition_the_issue_asks_for():
    pair_set = default_pair_set()
    composition = pair_set.composition()

    assert 140 <= len(pair_set) <= 160
    assert set(composition["ticker"]) == {"AAPL", "AMD", "META", "NVDA", "TSLA"}
    assert min(composition["ticker"].values()) >= 20
    assert composition["label"]["duplicate"] >= 60
    assert composition["label"]["distinct"] >= 60
    assert composition["label"]["ambiguous"] >= 1
    # Positives on both sides of the M2/M3 boundary, and hard negatives.
    assert composition["expected_stage"]["m2"] >= 30
    assert composition["expected_stage"]["m3"] >= 25
    for category in (
        "exact_duplicate",
        "syndicated_copy",
        "trivial_title_variant",
        "semantic_rewrite",
        "same_template_different_event",
        "repeated_quarterly",
        "role_change",
        "guidance_direction",
        "approval_decision",
        "beat_miss",
        "profit_loss",
        "number_change",
        "date_change",
        "quarter_change",
        "currency_change",
        "unit_change",
        "range_change",
        "sign_change",
        "different_company",
        "ambiguous",
    ):
        assert composition["category"].get(category, 0) >= 1, category


def test_every_positive_and_negative_ticker_is_represented():
    pair_set = default_pair_set()
    positives = {pair.ticker for pair in pair_set if pair.label == "duplicate"}
    negatives = {pair.ticker for pair in pair_set if pair.label == "distinct"}

    assert positives == {"AAPL", "AMD", "META", "NVDA", "TSLA"}
    assert negatives == {"AAPL", "AMD", "META", "NVDA", "TSLA"}


def test_m2_makes_no_false_merge_on_the_committed_set():
    report = evaluate_m2()

    assert report.overall.confusion.false_positives == ()
    assert report.overall.precision == 1.0
    assert report.ambiguous_merged == ()


def test_m2_catches_every_positive_it_is_responsible_for():
    report = evaluate_m2()
    by_stage = {entry.key: entry.metrics for entry in report.by_expected_stage}

    assert by_stage["m2"].recall == 1.0
    assert by_stage["m2"].confusion.fn == 0


def test_m2_leaves_the_semantic_rewrites_for_m3():
    """The measured case for issue #70 existing at all."""

    report = evaluate_m2()
    by_stage = {entry.key: entry.metrics for entry in report.by_expected_stage}

    assert by_stage["m3"].confusion.tp == 0
    assert by_stage["m3"].recall == 0.0
    # AC-3 needs recall >= 0.75; exact matching alone cannot reach it.
    assert report.overall.recall is not None
    assert report.overall.recall < 0.75
    assert not report.overall.meets(precision_floor=0.85, recall_floor=0.75)


def test_m2_scoring_is_repeatable_within_a_process():
    first = to_payload(evaluate_m2())
    second = to_payload(evaluate_m2())

    assert first == second


def test_m2_scoring_is_repeatable_across_processes():
    """Nothing in the pipeline may depend on PYTHONHASHSEED."""

    script = (
        "import json;"
        "from nlp.eval import evaluate_m2;"
        "from nlp.eval.report import to_payload;"
        "print(json.dumps(to_payload(evaluate_m2()), sort_keys=True))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(completed.stdout)

    assert len(outputs) == 1


def test_the_committed_baseline_file_matches_a_fresh_run():
    committed = json.loads(
        (
            REPO_ROOT / "nlp" / "eval" / "data" / "results" / "m2_baseline.json"
        ).read_text(encoding="utf-8")
    )

    assert committed == to_payload(evaluate_m2())


def test_pair_projection_preserves_the_raw_item_contract():
    pair = default_pair_set().by_id("P001")
    left, right = to_raw_items(pair)

    assert left.item_id == pair.item_a.item_id
    assert left.ticker == pair.ticker
    assert right.published_at == pair.item_b.published_at


def test_the_evaluation_config_uses_the_manifest_ticker_universe():
    pair_set = default_pair_set()

    assert config_for(pair_set).ticker_universe == frozenset(
        pair_set.metadata["tickers"]
    )


def test_the_m2_predictor_explains_why_a_pair_did_not_merge():
    pair_set = default_pair_set()
    predict = m2_predictor(config_for(pair_set))

    guidance = predict(pair_set.by_id("P103"))
    rewrite = predict(pair_set.by_id("P049"))

    assert not guidance.merged
    assert "description" in guidance.detail
    assert rewrite.detail == "not merged: no signal"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_the_cli_reports_and_exits_zero_without_a_gate(capsys):
    assert eval_dedup.main(["--stage", "m2"]) == 0
    assert "precision" in capsys.readouterr().out


def test_the_cli_fails_the_ac3_recall_gate_for_m2(capsys):
    exit_code = eval_dedup.main(
        ["--stage", "m2", "--precision-floor", "0.85", "--recall-floor", "0.75"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GATE FAILED: recall" in captured.err
    assert "GATE FAILED: precision" not in captured.err


def test_the_cli_writes_the_json_report(tmp_path, capsys):
    target = tmp_path / "nested" / "report.json"

    assert eval_dedup.main(["--stage", "m2", "--json", "--write", str(target)]) == 0
    capsys.readouterr()

    assert json.loads(target.read_text(encoding="utf-8"))["predictor"] == "m2"


def test_the_cli_prints_the_dataset_composition(capsys):
    assert eval_dedup.main(["--composition"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["pair_count"] == len(default_pair_set())


def test_the_cli_reports_a_dataset_error_without_a_traceback(tmp_path, capsys):
    assert eval_dedup.main(["--dataset", str(tmp_path / "absent.json")]) == 2
    assert "dataset error" in capsys.readouterr().err


def test_the_cli_refuses_to_sweep_a_stage_with_no_threshold(capsys):
    assert eval_dedup.main(["--stage", "m2", "--sweep"]) == 2
    assert "no tunable threshold" in capsys.readouterr().err

"""Tests for the M4 labeled dedup sets and their evaluators (issue #67).

Four things are under test and they are different things: that the
*loaders* refuse a dataset they cannot vouch for, that the *scorers* are
arithmetic nobody can nudge, that every artefact carries the provenance a
reader of a metric needs, and that the corrected pairs say what they claim.

The committed sets are also checked as data - their composition, their
trust contract, and the baselines they produce.
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
    TrustContractError,
    default_cluster_cases,
    default_pair_set,
    evaluate_isolated_pairs,
    evaluate_m2_clusters,
    evaluate_m2_isolated_pairs,
    load_cluster_cases,
    load_pair_set,
    sweep_thresholds,
    validate_thresholds,
)
from nlp.eval.clusters import (
    PartitionAccountingError,
    canonical_partition,
    check_cross_fixture_claims,
    evaluate_clusters,
    m2_cluster_predictor,
    permutations_of,
    validate_predicted_partition,
)
from nlp.eval.dataset import DEFAULT_META_PATH
from nlp.eval.dedup import config_for, m2_isolated_pair_predictor, to_raw_items
from nlp.eval.metrics import (
    ISOLATED_PAIR_LIMITATION,
    Confusion,
    Metrics,
)
from nlp.eval.report import (
    CLUSTER_PAYLOAD_VERSION,
    ISOLATED_PAIR_PAYLOAD_VERSION,
    SWEEP_PAYLOAD_VERSION,
    cluster_payload,
    render_clusters,
    render_sweep,
    render_text,
    sweep_payload,
    to_payload,
)
from nlp.eval.validation import (
    GateValueError,
    validate_optional_unit_interval,
    validate_unit_interval,
)
from tools import eval_dedup

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "nlp" / "eval" / "data" / "results"

_TRUST = {
    "dataset_kind": "synthetic_development",
    "real_ingested_evidence": False,
    "labeling_status": "single_author_unadjudicated",
    "reviewer_count": 1,
    "adjudicated": False,
    "gate_eligible": False,
    "metrics_purpose": "development_regression_only",
}
_PROVENANCE = {
    "kind": "synthetic",
    "collection_method": "authored",
    "statement": "authored for tests",
    "urls_are_synthetic": True,
    "uses_real_outlet_names": True,
    "why_synthetic": "no ingestion on main",
    "blocked_by": {"#61": "open"},
}
_LABELING = {
    "status": "single_author_unadjudicated",
    "reviewer_count": 1,
    "reviewers": [],
    "adjudicated": False,
    "gate_eligible": False,
    "protocol": "one author",
}
_MINIMAL_META = {
    "schema_version": SUPPORTED_SCHEMA_VERSION,
    "dataset_id": "test-set",
    "pairs_file": "pairs.jsonl",
    "tickers": ["NVDA", "TSLA"],
    "labels": ["duplicate", "distinct", "ambiguous"],
    "expected_stages": ["m2", "m3", "none"],
    "confidences": ["high", "medium", "low"],
    "categories": ["exact_duplicate", "semantic_rewrite", "hard_negative"],
    "trust_contract": dict(_TRUST),
    "provenance": dict(_PROVENANCE),
    "labeling": dict(_LABELING),
}


def _item(prefix: str, side: str, **overrides) -> dict:
    payload = {
        "item_id": f"{prefix}-{side}",
        "title": "Nvidia reports record revenue",
        "description": "The chipmaker beat estimates.",
        "url": f"https://example.test/{prefix}{side}",
        "canonical_url": None,
        "source": "Reuters" if side == "a" else "CNBC",
        "published_at": "2026-03-02T13:00:00+00:00"
        if side == "a"
        else "2026-03-02T13:30:00+00:00",
    }
    payload.update(overrides)
    return payload


def _pair(pair_id: str, **overrides: object) -> dict:
    payload: dict = {
        "pair_id": pair_id,
        "ticker": "NVDA",
        "label": "duplicate",
        "category": "exact_duplicate",
        "expected_stage": "m2",
        "rationale": "identical text",
        "confidence": "high",
        "item_a": _item(pair_id, "a"),
        "item_b": _item(pair_id, "b"),
    }
    payload.update(overrides)
    return payload


def write_set(tmp_path: Path, pairs, meta_overrides: dict | None = None) -> Path:
    """Write a manifest plus pairs file and return the manifest path."""

    meta = {key: value for key, value in _MINIMAL_META.items()}
    meta.update(meta_overrides or {})
    meta_path = tmp_path / "set.meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / str(meta["pairs_file"])).write_text(
        "\n".join(json.dumps(pair) for pair in pairs) + "\n", encoding="utf-8"
    )
    return meta_path


def negative(pair_id: str, **overrides) -> dict:
    return _pair(
        pair_id,
        label="distinct",
        category="hard_negative",
        expected_stage="none",
        **overrides,
    )


# --------------------------------------------------------------------------
# The corrected pairs say what they claim
# --------------------------------------------------------------------------


def test_the_two_undecidable_pairs_are_labelled_ambiguous():
    """A conversion and a summarizing estimate are not hard negatives."""

    pair_set = default_pair_set()

    for pair_id in ("P133", "P138"):
        pair = pair_set.by_id(pair_id)
        assert pair.label == "ambiguous", pair_id
        assert pair.confidence == "low", pair_id
        assert not pair.is_scored, pair_id
    assert "conversion" in pair_set.by_id("P133").rationale
    assert "range" in pair_set.by_id("P138").rationale


def test_the_article_level_cases_are_decided_by_the_contract_not_left_open():
    """P149-P153 were class balancing; the contract settles all five."""

    pair_set = default_pair_set()

    for pair_id in ("P149", "P150", "P151", "P152", "P153"):
        pair = pair_set.by_id(pair_id)
        assert pair.label == "distinct", pair_id
        assert pair.category == "same_event_different_article", pair_id
        assert pair.is_scored, pair_id


def test_the_ambiguous_label_is_now_reserved_for_undecidable_records():
    pair_set = default_pair_set()

    assert {pair.pair_id for pair in pair_set.ambiguous} == {"P133", "P138"}


@pytest.mark.parametrize(
    "pair_id,needle",
    [
        ("P031", "byte-identical"),
        ("P100", "committee"),
        ("P006", "same feed identifier"),
    ],
)
def test_a_corrected_rationale_describes_the_actual_difference(pair_id, needle):
    assert needle in default_pair_set().by_id(pair_id).rationale


def test_p031_is_no_longer_filed_as_a_typography_case():
    """Both titles are byte-identical; the old rationale claimed otherwise."""

    pair = default_pair_set().by_id("P031")

    assert pair.item_a.title == pair.item_b.title
    assert pair.category == "exact_duplicate"


@pytest.mark.parametrize(
    "pair_id,side,expected_year",
    [
        ("P002", "item_a", "2026-04"),
        ("P055", "item_a", "2026-04"),
        ("P089", "item_a", "2025-11"),
        ("P089", "item_b", "2026-02"),
        ("P090", "item_a", "2026-01"),
        ("P090", "item_b", "2026-04"),
        ("P091", "item_a", "2025-05"),
        ("P150", "item_a", "2026-04"),
    ],
)
def test_timestamps_now_match_the_reporting_cadence_their_links_carry(
    pair_id, side, expected_year
):
    item = getattr(default_pair_set().by_id(pair_id), side)

    assert item.published_at is not None
    assert item.published_at.startswith(expected_year)


@pytest.mark.parametrize("pair_id", ["P054", "P068", "P084", "P144", "P145", "P146"])
def test_a_pair_that_asserted_an_identity_now_carries_the_evidence(pair_id):
    """Both sides must say something that supports the rationale."""

    pair = default_pair_set().by_id(pair_id)

    assert pair.item_a.description, pair_id
    assert pair.item_b.description, pair_id


def test_p144_no_longer_files_an_automaker_story_under_a_phone_maker():
    pair = default_pair_set().by_id("P144")

    assert pair.item_a.ticker == pair.item_b.ticker == "TSLA"
    assert "Wolfsberg" in (pair.item_b.description or "")


def test_p131_pins_a_single_market_so_the_currencies_really_contradict():
    pair = default_pair_set().by_id("P131")

    assert "Mexican market" in (pair.item_a.description or "")
    assert "Mexican market" in (pair.item_b.description or "")
    assert pair.label == "distinct"


# --------------------------------------------------------------------------
# Trust contract: provenance is unavoidable
# --------------------------------------------------------------------------


def test_a_manifest_without_a_trust_contract_is_refused(tmp_path):
    meta = {
        key: value for key, value in _MINIMAL_META.items() if key != "trust_contract"
    }
    meta_path = tmp_path / "set.meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text(json.dumps(_pair("P001")), encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="trust_contract"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", sorted(_TRUST))
def test_a_trust_contract_missing_any_field_is_refused(tmp_path, field):
    trust = {key: value for key, value in _TRUST.items() if key != field}
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="missing"):
        load_pair_set(meta_path)


def test_synthetic_data_may_not_claim_real_ingestion(tmp_path):
    trust = dict(_TRUST, real_ingested_evidence=True)
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(
        TrustContractError, match="requires real_ingested_evidence=false"
    ):
        load_pair_set(meta_path)


def test_synthetic_data_may_not_claim_gate_eligibility(tmp_path):
    trust = dict(_TRUST, gate_eligible=True)
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="cannot claim gate_eligible"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("value", [0, -1, 1.5, "one", True, None])
def test_reviewer_count_must_be_a_positive_integer(tmp_path, value):
    trust = dict(_TRUST, reviewer_count=value)
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="positive integer"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", sorted(_PROVENANCE))
def test_provenance_missing_any_required_field_is_refused(tmp_path, field):
    provenance = {key: value for key, value in _PROVENANCE.items() if key != field}
    meta_path = write_set(tmp_path, [_pair("P001")], {"provenance": provenance})

    with pytest.raises(TrustContractError, match="provenance is missing"):
        load_pair_set(meta_path)


def test_provenance_must_be_a_mapping_not_a_sentence(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001")], {"provenance": "synthetic"})

    with pytest.raises(TrustContractError, match="provenance must be a mapping"):
        load_pair_set(meta_path)


def test_a_synthetic_set_may_not_claim_its_urls_are_real(tmp_path):
    provenance = dict(_PROVENANCE, urls_are_synthetic=False)
    meta_path = write_set(tmp_path, [_pair("P001")], {"provenance": provenance})

    with pytest.raises(TrustContractError, match="urls_are_synthetic=true"):
        load_pair_set(meta_path)


def test_a_synthetic_set_may_not_claim_it_was_collected(tmp_path):
    provenance = dict(_PROVENANCE, collection_method="sampled")
    meta_path = write_set(tmp_path, [_pair("P001")], {"provenance": provenance})

    with pytest.raises(TrustContractError, match="provenance.collection_method"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", sorted(_LABELING))
def test_labeling_missing_any_required_field_is_refused(tmp_path, field):
    labeling = {key: value for key, value in _LABELING.items() if key != field}
    meta_path = write_set(tmp_path, [_pair("P001")], {"labeling": labeling})

    with pytest.raises(TrustContractError, match="labeling is missing"):
        load_pair_set(meta_path)


def test_labeling_may_not_contradict_the_trust_contract(tmp_path):
    labeling = dict(_LABELING, reviewer_count=3)
    meta_path = write_set(tmp_path, [_pair("P001")], {"labeling": labeling})

    with pytest.raises(TrustContractError, match="contradicts trust_contract"):
        load_pair_set(meta_path)


def test_labeling_may_not_name_more_reviewers_than_it_declares(tmp_path):
    labeling = dict(_LABELING, reviewers=["a", "b"])
    meta_path = write_set(tmp_path, [_pair("P001")], {"labeling": labeling})

    with pytest.raises(TrustContractError, match="names 2 reviewers"):
        load_pair_set(meta_path)


def test_every_loaded_record_is_stamped_synthetic():
    """Dataset-enforced marking: a record cannot travel without provenance."""

    pair_set = default_pair_set()

    assert all(pair.synthetic for pair in pair_set)
    assert all(pair.item_a.synthetic and pair.item_b.synthetic for pair in pair_set)
    assert all(
        item.synthetic for case in default_cluster_cases() for item in case.items
    )


# --------------------------------------------------------------------------
# Provenance reaches every output
# --------------------------------------------------------------------------


def test_the_text_report_prints_the_warning_before_and_after_the_numbers():
    rendered = render_text(evaluate_m2_isolated_pairs())
    lines = rendered.splitlines()

    assert lines[0].startswith("WARNING:")
    assert rendered.rstrip().endswith("development_regression_only")
    assert rendered.count("WARNING:") == 2
    assert rendered.index("WARNING:") < rendered.index("isolated_pair_metrics")


@pytest.mark.parametrize("field", sorted(_TRUST))
def test_the_text_report_states_every_trust_field(field):
    assert field in render_text(evaluate_m2_isolated_pairs())


@pytest.mark.parametrize("field", sorted(_TRUST))
def test_the_json_report_states_every_trust_field(field):
    payload = to_payload(evaluate_m2_isolated_pairs())

    assert field in payload["trust_contract"]


def test_the_cluster_report_carries_the_same_contract_in_both_formats():
    report = evaluate_m2_clusters()

    assert render_clusters(report).splitlines()[0].startswith("WARNING:")
    assert cluster_payload(report)["trust_contract"]["gate_eligible"] is False


@pytest.mark.parametrize(
    "path",
    [
        "m2_baseline.json",
        "m2_clusters_exact_stage.json",
        "m2_clusters_ground_truth.json",
    ],
)
def test_every_committed_result_file_carries_the_trust_contract(path):
    payload = json.loads((RESULTS / path).read_text(encoding="utf-8"))
    trust = payload["trust_contract"]

    assert trust["dataset_kind"] == "synthetic_development"
    assert trust["real_ingested_evidence"] is False
    assert trust["gate_eligible"] is False
    assert trust["metrics_purpose"] == "development_regression_only"
    assert trust["warning"].startswith("WARNING:")


def test_the_cli_prints_the_warning_in_text_mode(capsys):
    assert eval_dedup.main(["--stage", "m2"]) == 0

    assert capsys.readouterr().out.startswith("WARNING:")


def test_the_cli_puts_the_contract_in_json_mode(capsys):
    assert eval_dedup.main(["--stage", "m2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["trust_contract"]["gate_eligible"] is False


def test_the_cli_composition_output_carries_the_contract(capsys):
    assert eval_dedup.main(["--composition"]) == 0

    assert json.loads(capsys.readouterr().out)["trust_contract"]["reviewer_count"] == 1


def test_the_committed_banner_is_derived_not_supplied():
    trust = default_pair_set().trust

    assert trust.warning == trust.summary.text
    assert trust.summary.level == "WARNING"


# --------------------------------------------------------------------------
# The isolated-pair limitation is stated, not implied away
# --------------------------------------------------------------------------


def test_the_limitation_names_what_a_pairwise_metric_cannot_see():
    for needle in (
        "two-item invocation",
        "cluster-wide compatibility",
        "transitivity",
        "quarantine",
        "capacity",
        "cannot on their own validate production clustering",
    ):
        assert needle in ISOLATED_PAIR_LIMITATION, needle


def test_the_report_is_scoped_and_never_calls_itself_overall():
    report = evaluate_m2_isolated_pairs()
    payload = to_payload(report)

    assert report.scope == "isolated_pairs"
    assert "isolated_pair_metrics" in payload
    assert "overall" not in payload
    assert "production_cluster_metrics" not in payload
    assert not hasattr(report, "overall")


def test_the_limitation_travels_with_every_rendering():
    report = evaluate_m2_isolated_pairs()
    # The text renderer wraps it to the report width, so compare on the
    # collapsed whitespace rather than on the literal string.
    rendered = " ".join(render_text(report).split())

    assert " ".join(report.limitation.split()) in rendered
    assert to_payload(report)["limitation"] == ISOLATED_PAIR_LIMITATION


def test_no_module_claims_pairwise_evaluation_reproduces_clustering():
    """The old docstring asserted this; it was not true."""

    for name in ("dedup.py", "metrics.py", "clusters.py", "__init__.py"):
        source = (REPO_ROOT / "nlp" / "eval" / name).read_text(encoding="utf-8")
        assert "faithful because" not in source, name


# --------------------------------------------------------------------------
# Dataset integrity
# --------------------------------------------------------------------------


def test_duplicate_pair_ids_are_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001"), _pair("P001")])

    with pytest.raises(EvalDatasetError, match="duplicate pair_id: P001"):
        load_pair_set(meta_path)


def test_a_pair_repeating_another_pairs_content_is_refused(tmp_path):
    twin = _pair("P002")
    twin["item_a"] = _item("P001", "a", item_id="P002-a")
    twin["item_b"] = _item("P001", "b", item_id="P002-b")
    meta_path = write_set(tmp_path, [_pair("P001"), twin])

    with pytest.raises(EvalDatasetError, match=r"P002 repeats the content of P001"):
        load_pair_set(meta_path)


def test_a_pair_repeating_another_pairs_content_reversed_is_refused(tmp_path):
    """Swapping the two sides does not make it a different pair."""

    mirrored = _pair("P002")
    mirrored["item_a"] = _item("P001", "b", item_id="P002-a")
    mirrored["item_b"] = _item("P001", "a", item_id="P002-b")
    meta_path = write_set(tmp_path, [_pair("P001"), mirrored])

    with pytest.raises(EvalDatasetError, match="sides swapped"):
        load_pair_set(meta_path)


def test_a_negative_pair_of_two_identical_records_is_refused(tmp_path):
    same = negative("P001")
    same["item_b"] = _item("P001", "a", item_id="P001-b")
    meta_path = write_set(tmp_path, [same])

    with pytest.raises(EvalDatasetError, match="cannot be different events"):
        load_pair_set(meta_path)


def test_a_duplicate_pair_of_two_identical_records_is_allowed(tmp_path):
    """A feed re-poll emits the same row twice; that is provider_repeat."""

    repost = _pair("P001")
    repost["item_b"] = _item("P001", "a", item_id="P001-b")
    meta_path = write_set(tmp_path, [repost])

    assert len(load_pair_set(meta_path)) == 1


def test_an_item_missing_canonical_url_is_refused(tmp_path):
    pair = _pair("P001")
    del pair["item_a"]["canonical_url"]
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="canonical_url must be"):
        load_pair_set(meta_path)


def test_every_committed_item_carries_canonical_url_explicitly():
    raw = (REPO_ROOT / "nlp" / "eval" / "data" / "dedup_pairs.jsonl").read_text("utf-8")

    for line in raw.splitlines():
        payload = json.loads(line)
        for side in ("item_a", "item_b"):
            assert "canonical_url" in payload[side], payload["pair_id"]


@pytest.mark.parametrize(
    "url",
    ["ftp://example.test/a", "not-a-url", "https://nohost/a", "https://a.test/x y"],
)
def test_an_unusable_url_is_refused(tmp_path, url):
    pair = _pair("P001")
    pair["item_a"]["url"] = url
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="url"):
        load_pair_set(meta_path)


@pytest.mark.parametrize(
    "stamp", ["1889-01-01T00:00:00+00:00", "2101-01-01T00:00:00+00:00"]
)
def test_a_timestamp_outside_the_dedup_cores_range_is_refused(tmp_path, stamp):
    pair = _pair("P001")
    pair["item_a"]["published_at"] = stamp
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="outside the range the dedup core"):
        load_pair_set(meta_path)


def test_a_naive_timestamp_is_refused_at_load_time(tmp_path):
    pair = _pair("P001")
    pair["item_a"]["published_at"] = "2026-03-02T13:00:00"
    meta_path = write_set(tmp_path, [pair])

    with pytest.raises(EvalDatasetError, match="must carry a timezone offset"):
        load_pair_set(meta_path)


def test_the_committed_timestamps_are_all_inside_the_cores_range():
    """The pair set and the dedup core must agree on what a date is."""

    from nlp.dedup.normalization import (
        MAX_PLAUSIBLE_PUBLISHED_AT,
        MIN_PLAUSIBLE_PUBLISHED_AT,
    )
    from datetime import datetime

    for pair in default_pair_set():
        for item in (pair.item_a, pair.item_b):
            if item.published_at is None:
                continue
            stamp = datetime.fromisoformat(item.published_at)
            assert MIN_PLAUSIBLE_PUBLISHED_AT <= stamp < MAX_PLAUSIBLE_PUBLISHED_AT


@pytest.mark.parametrize(
    "label,stage",
    [("duplicate", "none"), ("distinct", "m2"), ("ambiguous", "m3")],
)
def test_a_label_contradicting_its_expected_stage_is_refused(tmp_path, label, stage):
    meta_path = write_set(tmp_path, [_pair("P001", label=label, expected_stage=stage)])

    with pytest.raises(EvalDatasetError, match="incompatible with expected_stage"):
        load_pair_set(meta_path)


def test_an_invalid_label_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", label="maybe")])

    with pytest.raises(EvalDatasetError, match="label='maybe' is not one of"):
        load_pair_set(meta_path)


def test_an_unknown_pair_field_is_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P001", notes="extra")])

    with pytest.raises(EvalDatasetError, match=r"unknown field\(s\) \['notes'\]"):
        load_pair_set(meta_path)


def test_rows_out_of_pair_id_order_are_refused(tmp_path):
    meta_path = write_set(tmp_path, [_pair("P002"), _pair("P001")])

    with pytest.raises(EvalDatasetError, match="must be sorted by pair_id"):
        load_pair_set(meta_path)


def test_an_unknown_schema_version_is_refused(tmp_path):
    meta_path = write_set(
        tmp_path, [_pair("P001")], {"schema_version": "phase0.dedup_eval.v99"}
    )

    with pytest.raises(EvalDatasetError, match="unsupported schema_version"):
        load_pair_set(meta_path)


def test_the_committed_set_is_sorted_and_stable():
    pair_set = default_pair_set()
    identifiers = [pair.pair_id for pair in pair_set]

    assert identifiers == sorted(identifiers)
    assert identifiers == [pair.pair_id for pair in load_pair_set(DEFAULT_META_PATH)]


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def test_valid_thresholds_come_back_sorted():
    assert validate_thresholds([0.9, 0.1, 0.5]) == (0.1, 0.5, 0.9)


# --------------------------------------------------------------------------
# Isolated-pair evaluator arithmetic
# --------------------------------------------------------------------------


def constant(merged: bool, **kwargs):
    return lambda pair: PairPrediction(merged=merged, **kwargs)


def test_confusion_cells_hold_the_pair_ids(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [_pair("P001"), _pair("P002"), negative("P003"), negative("P004")],
        )
    )

    report = evaluate_isolated_pairs(
        pair_set,
        lambda pair: PairPrediction(merged=pair.pair_id in {"P001", "P003"}),
        name="partial",
    )
    confusion = report.isolated_pair_metrics.confusion

    assert confusion.true_positives == ("P001",)
    assert confusion.false_negatives == ("P002",)
    assert confusion.false_positives == ("P003",)
    assert confusion.true_negatives == ("P004",)
    assert report.isolated_pair_metrics.precision == 0.5


def test_ambiguous_pairs_are_excluded_and_reported_separately(tmp_path):
    pair_set = load_pair_set(
        write_set(
            tmp_path,
            [
                _pair("P001"),
                _pair("P002", label="ambiguous", expected_stage="none"),
            ],
        )
    )

    report = evaluate_isolated_pairs(pair_set, constant(True), name="merge-everything")

    assert report.isolated_pair_metrics.confusion.total == 1
    assert report.ambiguous_count == 1
    assert report.ambiguous_merged == ("P002",)


def test_precision_is_undefined_rather_than_zero_when_nothing_is_merged(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))

    report = evaluate_isolated_pairs(pair_set, constant(False), name="merge-nothing")

    assert report.isolated_pair_metrics.precision is None
    assert report.isolated_pair_metrics.recall == 0.0
    assert report.isolated_pair_metrics.f1 is None


def test_an_empty_confusion_yields_no_numbers_at_all():
    metrics = Metrics.from_confusion(Confusion())

    assert (metrics.precision, metrics.recall, metrics.f1, metrics.accuracy) == (
        None,
        None,
        None,
        None,
    )


def test_an_undefined_metric_never_clears_a_gate():
    undefined = Metrics.from_confusion(Confusion(true_negatives=("A",)))
    passing = Metrics.from_confusion(Confusion(true_positives=("A",)))

    assert not undefined.meets(precision_floor=0.0, recall_floor=0.0)
    assert passing.meets(precision_floor=1.0, recall_floor=1.0)


def test_candidate_recall_bounds_merge_recall(tmp_path):
    pair_set = load_pair_set(
        write_set(tmp_path, [_pair("P001"), _pair("P002"), _pair("P003")])
    )

    report = evaluate_isolated_pairs(
        pair_set,
        lambda pair: PairPrediction(merged=pair.pair_id == "P001", candidate=True),
        name="picky",
    )

    assert report.candidate_recall == 1.0
    assert report.merge_recall == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# Structured evaluation errors
# --------------------------------------------------------------------------


def exploding(pair_ids: set[str], error: Exception):
    def predict(pair):
        if pair.pair_id in pair_ids:
            raise error
        return PairPrediction(merged=pair.is_positive)

    return predict


def test_a_raising_pair_is_reported_rather_than_ending_the_run(tmp_path):
    pair_set = load_pair_set(
        write_set(tmp_path, [_pair("P001"), _pair("P002"), negative("P003")])
    )

    report = evaluate_isolated_pairs(
        pair_set, exploding({"P002"}, RuntimeError("stage exploded")), name="flaky"
    )

    assert report.failed_case_count == 1
    assert report.failed_case_ids == ("P002",)
    assert report.failures[0].error_type == "RuntimeError"
    assert report.failures[0].message == "stage exploded"
    assert not report.complete


def test_a_failed_pair_leaves_every_denominator(tmp_path):
    pair_set = load_pair_set(
        write_set(tmp_path, [_pair("P001"), _pair("P002"), negative("P003")])
    )

    report = evaluate_isolated_pairs(
        pair_set, exploding({"P002"}, ValueError("bad input")), name="flaky"
    )
    confusion = report.isolated_pair_metrics.confusion

    assert confusion.total == 2
    assert "P002" not in confusion.false_negatives
    assert report.evaluated_case_count == 2
    assert report.isolated_pair_metrics.recall == 1.0


def test_a_capacity_refusal_is_captured_with_its_type(tmp_path):
    from nlp.dedup import DedupCapacityError

    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))
    report = evaluate_isolated_pairs(
        pair_set,
        exploding({"P001"}, DedupCapacityError("NVDA", 900, 250)),
        name="over-capacity",
    )

    assert report.failures[0].error_type == "DedupCapacityError"
    assert "900" in report.failures[0].message
    assert report.isolated_pair_metrics.confusion.total == 0


def test_an_incomplete_report_says_so_in_both_formats(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001"), _pair("P002")]))
    report = evaluate_isolated_pairs(
        pair_set, exploding({"P001"}, RuntimeError("boom")), name="flaky"
    )

    payload = to_payload(report)["completeness"]
    rendered = render_text(report)

    assert payload["complete"] is False
    assert payload["failed_case_ids"] == ["P001"]
    assert payload["failures"][0]["error_type"] == "RuntimeError"
    assert "complete              false" in rendered
    assert "excluded from every denominator" in rendered


def test_a_complete_run_reports_no_failures():
    report = evaluate_m2_isolated_pairs()

    assert report.complete
    assert report.failed_case_count == 0
    assert report.evaluated_case_count == len(default_pair_set())


def test_a_predictor_returning_the_wrong_type_still_raises(tmp_path):
    """A type error is the caller's bug, not a stage failure to tabulate."""

    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))

    def not_a_prediction(pair):
        return True

    with pytest.raises(TypeError, match="expected PairPrediction"):
        evaluate_isolated_pairs(pair_set, not_a_prediction, name="bad")


# --------------------------------------------------------------------------
# Multi-item cluster evaluation
# --------------------------------------------------------------------------


def test_the_cluster_fixture_covers_every_required_batch_behaviour():
    categories = {case.category for case in default_cluster_cases()}

    assert categories >= {
        "sparse_bridge_contradictory",
        "sparse_bridge_compatible",
        "provider_conflict_group",
        "url_reuse_group",
        "repeated_quarterly_group",
        "semantic_transitivity",
        "mixed_stage_group",
        "permutation_equivalence",
    }


def test_a_partition_that_misses_an_item_is_refused(tmp_path):
    meta = json.loads(
        (REPO_ROOT / "nlp" / "eval" / "data" / "cluster_cases.meta.json").read_text(
            "utf-8"
        )
    )
    case = json.loads(
        (REPO_ROOT / "nlp" / "eval" / "data" / "cluster_cases.jsonl")
        .read_text("utf-8")
        .splitlines()[0]
    )
    case["expected_partition"] = [[case["items"][0]["item_id"]]]
    case["cross_fixture_claims"] = []
    meta["cases_file"] = "cases.jsonl"
    (tmp_path / "cases.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(json.dumps(case), encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="does not account for"):
        load_cluster_cases(tmp_path / "cases.meta.json")


def test_a_sparse_record_does_bridge_compatible_endpoints():
    report = evaluate_m2_clusters()
    outcome = next(o for o in report.outcomes if o.case_id == "C002")

    assert outcome.predicted == canonical_partition(
        [frozenset({"C002-1", "C002-2", "C002-3"})]
    )


def test_m2_is_permutation_stable_on_every_case():
    assert evaluate_m2_clusters().permutation_failures == ()


def test_the_cluster_evaluator_runs_the_whole_group_in_one_call():
    """Not pair-by-pair: the batch is what makes the behaviour observable."""

    sizes: list[int] = []
    config = eval_dedup.config_for_cluster_set(default_cluster_cases())
    predictor = m2_cluster_predictor(config)

    def spy(case):
        sizes.append(len(case.items))
        return predictor(case)

    evaluate_clusters(default_cluster_cases(), spy, name="spy")

    assert min(sizes) >= 3


def test_an_unknown_cluster_target_is_refused():
    with pytest.raises(ValueError, match="unknown target"):
        evaluate_clusters(
            default_cluster_cases(), lambda case: (), name="x", target="?"
        )


# --------------------------------------------------------------------------
# The committed sets and baselines
# --------------------------------------------------------------------------


def test_the_committed_set_has_the_composition_the_issue_asks_for():
    pair_set = default_pair_set()
    composition = pair_set.composition()

    assert 140 <= len(pair_set) <= 160
    assert set(composition["ticker"]) == {"AAPL", "AMD", "META", "NVDA", "TSLA"}
    assert min(composition["ticker"].values()) >= 20
    assert composition["label"]["duplicate"] >= 60
    assert composition["label"]["distinct"] >= 60
    assert composition["label"]["ambiguous"] >= 1
    assert composition["expected_stage"]["m2"] >= 30
    assert composition["expected_stage"]["m3"] >= 25
    for category in pair_set.metadata["categories"]:
        assert composition["category"].get(category, 0) >= 1, category


def test_every_positive_and_negative_ticker_is_represented():
    pair_set = default_pair_set()
    positives = {pair.ticker for pair in pair_set if pair.label == "duplicate"}
    negatives = {pair.ticker for pair in pair_set if pair.label == "distinct"}

    assert positives == {"AAPL", "AMD", "META", "NVDA", "TSLA"}
    assert negatives == {"AAPL", "AMD", "META", "NVDA", "TSLA"}


def test_m2_makes_no_false_merge_on_the_committed_set():
    report = evaluate_m2_isolated_pairs()

    assert report.isolated_pair_metrics.confusion.false_positives == ()
    assert report.isolated_pair_metrics.precision == 1.0
    assert report.ambiguous_merged == ()


def test_m2_catches_every_positive_it_is_responsible_for():
    report = evaluate_m2_isolated_pairs()
    by_stage = {entry.key: entry.metrics for entry in report.by_expected_stage}

    assert by_stage["m2"].recall == 1.0
    assert by_stage["m2"].confusion.fn == 0


def test_m2_leaves_the_semantic_rewrites_for_m3():
    """The measured case for issue #70 existing at all."""

    report = evaluate_m2_isolated_pairs()
    by_stage = {entry.key: entry.metrics for entry in report.by_expected_stage}

    assert by_stage["m3"].confusion.tp == 0
    assert report.isolated_pair_metrics.recall is not None
    assert report.isolated_pair_metrics.recall < 0.75
    assert not report.isolated_pair_metrics.meets(
        precision_floor=0.85, recall_floor=0.75
    )


def test_m2_scoring_is_repeatable_across_processes():
    """Nothing in the pipeline may depend on PYTHONHASHSEED."""

    script = (
        "import json;"
        "from nlp.eval import evaluate_m2_isolated_pairs;"
        "from nlp.eval.report import to_payload;"
        "print(json.dumps(to_payload(evaluate_m2_isolated_pairs()), sort_keys=True))"
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
    committed = json.loads((RESULTS / "m2_baseline.json").read_text(encoding="utf-8"))

    assert committed == to_payload(evaluate_m2_isolated_pairs())


@pytest.mark.parametrize(
    "path,target",
    [
        ("m2_clusters_exact_stage.json", "exact_stage_partition"),
        ("m2_clusters_ground_truth.json", "expected_partition"),
    ],
)
def test_the_committed_cluster_results_match_a_fresh_run(path, target):
    committed = json.loads((RESULTS / path).read_text(encoding="utf-8"))

    assert committed == cluster_payload(evaluate_m2_clusters(target=target))


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
    predict = m2_isolated_pair_predictor(config_for(pair_set))

    guidance = predict(pair_set.by_id("P103"))
    rewrite = predict(pair_set.by_id("P049"))

    assert not guidance.merged
    assert "description" in guidance.detail
    assert rewrite.detail == "not merged: no signal"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_the_cli_fails_the_ac3_recall_gate_for_m2(capsys):
    exit_code = eval_dedup.main(
        ["--stage", "m2", "--precision-floor", "0.85", "--recall-floor", "0.75"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "GATE FAILED: recall" in captured.err
    assert "not acceptance" in captured.err


def test_the_cli_scores_clusters(capsys):
    assert eval_dedup.main(["--stage", "m2", "--scope", "clusters"]) == 0
    out = capsys.readouterr().out

    assert "multi_item_cluster_metrics" in out
    assert "exact partition match  9/9" in out


def test_the_cli_writes_both_report_kinds(tmp_path, capsys):
    pairs = tmp_path / "pairs.json"
    clusters = tmp_path / "clusters.json"

    assert eval_dedup.main(["--stage", "m2", "--json", "--write", str(pairs)]) == 0
    assert (
        eval_dedup.main(
            ["--stage", "m2", "--scope", "clusters", "--json", "--write", str(clusters)]
        )
        == 0
    )
    capsys.readouterr()

    assert json.loads(pairs.read_text("utf-8"))["scope"] == "isolated_pairs"
    assert json.loads(clusters.read_text("utf-8"))["scope"] == "multi_item_clusters"


def test_the_cli_reports_a_dataset_error_without_a_traceback(tmp_path, capsys):
    assert eval_dedup.main(["--dataset", str(tmp_path / "absent.json")]) == 2
    assert "dataset error" in capsys.readouterr().err


def test_the_cli_refuses_to_sweep_a_stage_with_no_threshold(capsys):
    assert eval_dedup.main(["--stage", "m2", "--sweep"]) == 2
    assert "no tunable threshold" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Corrected cluster cases C001, C003, C009
# --------------------------------------------------------------------------


def case(case_id: str):
    return default_cluster_cases().by_id(case_id)


def test_c001_does_not_encode_traversal_order_as_human_truth():
    """The sparse record's placement is not decided by the records."""

    c001 = case("C001")

    assert c001.indeterminate_item_ids == frozenset({"C001-2"})
    assert c001.expected_partition == canonical_partition(
        [frozenset({"C001-1"}), frozenset({"C001-3"})]
    )
    # The implementation's answer is kept, separately, where it belongs.
    assert c001.exact_stage_partition == canonical_partition(
        [frozenset({"C001-1", "C001-2"}), frozenset({"C001-3"})]
    )
    assert c001.expected_partition != c001.exact_stage_partition


def test_c001_scores_no_pair_involving_the_indeterminate_record():
    report = evaluate_m2_clusters(target="expected_partition")
    outcome = next(o for o in report.outcomes if o.case_id == "C001")

    scored = {
        item
        for pair in outcome.over_merged_pairs + outcome.under_merged_pairs
        for item in pair
    }
    assert "C001-2" not in scored
    assert outcome.exact_match
    assert outcome.indeterminate_item_ids == ("C001-2",)


def test_c001_still_asserts_the_decidable_part():
    """Ground truth still says the two contradicting records stay apart."""

    c001 = case("C001")

    assert not any({"C001-1", "C001-3"} <= group for group in c001.expected_partition)


def test_c003_separates_quarantine_policy_from_human_truth():
    c003 = case("C003")

    # As articles: the two identical wiper stories are one story.
    assert c003.expected_partition == canonical_partition(
        [frozenset({"C003-1", "C003-3"}), frozenset({"C003-2"})]
    )
    # As implementation: quarantine emits three singletons.
    assert all(len(group) == 1 for group in c003.exact_stage_partition)


def test_c003_shows_quarantine_as_an_under_merge_against_ground_truth():
    """The honest cost of the policy, visible rather than defined away."""

    ground = evaluate_m2_clusters(target="expected_partition")
    stage = evaluate_m2_clusters(target="exact_stage_partition")

    assert "C003" in ground.under_merge_case_ids
    assert "C003" not in stage.under_merge_case_ids
    assert "C003" not in ground.over_merge_case_ids


def test_c009_takes_the_same_decision_as_p152():
    c009 = case("C009")
    p152 = default_pair_set().by_id("P152")

    assert c009.status == "decidable"
    assert p152.label == "distinct"
    assert not any({"C009-1", "C009-3"} <= group for group in c009.expected_partition)
    claim = next(c for c in c009.cross_fixture_claims if c.pair_id == "P152")
    assert claim.relationship == "different_story"


def test_no_cluster_case_is_left_ambiguous_without_reason():
    for entry in default_cluster_cases():
        assert entry.status == "decidable", entry.case_id


# --------------------------------------------------------------------------
# Cross-fixture consistency
# --------------------------------------------------------------------------


def test_every_committed_claim_agrees_with_its_pair():
    pair_set = default_pair_set()
    cases = default_cluster_cases()

    claimed = 0
    for entry in cases:
        for claim in entry.cross_fixture_claims:
            pair = pair_set.by_id(claim.pair_id)
            expected = "same_story" if pair.label == "duplicate" else "different_story"
            assert claim.relationship == expected, (entry.case_id, claim.pair_id)
            claimed += 1
    assert claimed >= 8


def _mutate_case(entry, **changes):
    from dataclasses import replace

    return replace(entry, **changes)


def test_a_claim_contradicting_its_pair_is_refused():
    from nlp.eval.clusters import CrossFixtureClaim

    pair_set = default_pair_set()
    entry = _mutate_case(
        case("C009"),
        cross_fixture_claims=(
            CrossFixtureClaim(
                pair_id="P152",
                item_ids=("C009-1", "C009-2"),
                relationship="same_story",
            ),
        ),
    )

    with pytest.raises(EvalDatasetError, match="may not contradict a pair fixture"):
        check_cross_fixture_claims(entry, pair_set)


def test_a_documented_divergence_is_allowed():
    from nlp.eval.clusters import CrossFixtureClaim

    pair_set = default_pair_set()
    entry = _mutate_case(
        case("C009"),
        cross_fixture_claims=(
            CrossFixtureClaim(
                pair_id="P152",
                item_ids=("C009-1", "C009-2"),
                relationship="same_story",
                divergence_reason="these two are the wire copy, not the interview",
            ),
        ),
    )

    check_cross_fixture_claims(entry, pair_set)


def test_a_claim_may_not_borrow_authority_from_an_ambiguous_pair():
    from nlp.eval.clusters import CrossFixtureClaim

    pair_set = default_pair_set()
    entry = _mutate_case(
        case("C009"),
        cross_fixture_claims=(
            CrossFixtureClaim(
                pair_id="P133",
                item_ids=("C009-1", "C009-2"),
                relationship="same_story",
            ),
        ),
    )

    with pytest.raises(EvalDatasetError, match="the records do not decide"):
        check_cross_fixture_claims(entry, pair_set)


def test_a_claim_disagreeing_with_its_own_partition_is_refused():
    from nlp.eval.clusters import CrossFixtureClaim

    entry = _mutate_case(
        case("C009"),
        cross_fixture_claims=(
            CrossFixtureClaim(
                pair_id="P152",
                item_ids=("C009-1", "C009-3"),
                relationship="same_story",
            ),
        ),
    )

    with pytest.raises(EvalDatasetError, match="expected_partition places"):
        check_cross_fixture_claims(entry, default_pair_set())


def test_a_claim_about_an_indeterminate_item_is_refused():
    from nlp.eval.clusters import CrossFixtureClaim

    entry = _mutate_case(
        case("C001"),
        cross_fixture_claims=(
            CrossFixtureClaim(
                pair_id="P117",
                item_ids=("C001-1", "C001-2"),
                relationship="different_story",
            ),
        ),
    )

    with pytest.raises(EvalDatasetError, match="indeterminate item"):
        check_cross_fixture_claims(entry, default_pair_set())


# --------------------------------------------------------------------------
# Trust invariants
# --------------------------------------------------------------------------


def test_the_committed_sets_declare_the_full_trust_contract():
    from nlp.eval.trust import DatasetKind, LabelingStatus, MetricsPurpose

    for trust in (default_pair_set().trust, default_cluster_cases().trust):
        assert trust.dataset_kind is DatasetKind.SYNTHETIC_DEVELOPMENT
        assert trust.real_ingested_evidence is False
        assert trust.labeling_status is LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED
        assert trust.reviewer_count == 1
        assert trust.adjudicated is False
        assert trust.gate_eligible is False
        assert trust.metrics_purpose is MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_kind", "made_up"),
        ("labeling_status", "someone_looked"),
        ("metrics_purpose", "shipping"),
    ],
)
def test_an_unknown_enum_value_is_refused(tmp_path, field, value):
    trust = dict(_TRUST, **{field: value})
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="is not one of"):
        load_pair_set(meta_path)


@pytest.mark.parametrize(
    "overrides,labeling_overrides,message",
    [
        # synthetic + final_acceptance
        (
            {"metrics_purpose": "final_acceptance"},
            {},
            "must declare a metrics_purpose in",
        ),
        # synthetic + gate_acceptance
        (
            {"metrics_purpose": "gate_acceptance"},
            {},
            "must declare a metrics_purpose in",
        ),
        # synthetic + gate_eligible
        (
            {"gate_eligible": True},
            {"gate_eligible": True},
            "cannot claim gate_eligible",
        ),
        # synthetic + real evidence
        (
            {"real_ingested_evidence": True},
            {},
            "requires real_ingested_evidence=false",
        ),
        # single author + adjudicated
        (
            {"adjudicated": True},
            {"adjudicated": True},
            "cannot be adjudicated",
        ),
        # single author + reviewer_count 3
        (
            {"reviewer_count": 3},
            {"reviewer_count": 3},
            "requires reviewer_count=1",
        ),
        # single author + gate_eligible
        (
            {
                "gate_eligible": True,
                "dataset_kind": "sampled_production",
                "real_ingested_evidence": True,
                "metrics_purpose": "gate_acceptance",
            },
            {"gate_eligible": True},
            "cannot be gate_eligible",
        ),
        # adjudicated status but adjudicated false
        (
            {
                "dataset_kind": "sampled_production",
                "real_ingested_evidence": True,
                "labeling_status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
            },
            {"status": "multi_reviewer_adjudicated", "reviewer_count": 2},
            "but adjudicated=false",
        ),
        # multi-reviewer status with one reviewer
        (
            {
                "dataset_kind": "sampled_production",
                "real_ingested_evidence": True,
                "labeling_status": "multi_reviewer_unadjudicated",
            },
            {"status": "multi_reviewer_unadjudicated"},
            "requires reviewer_count >= 2",
        ),
        # gate eligible without real evidence
        (
            {
                "dataset_kind": "sampled_production",
                "labeling_status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
                "adjudicated": True,
                "gate_eligible": True,
                "metrics_purpose": "gate_acceptance",
            },
            {
                "status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
                "adjudicated": True,
                "gate_eligible": True,
            },
            "requires real_ingested_evidence=true",
        ),
        # gate eligible without adjudication
        (
            {
                "dataset_kind": "sampled_production",
                "labeling_status": "multi_reviewer_unadjudicated",
                "reviewer_count": 2,
                "real_ingested_evidence": True,
                "gate_eligible": True,
                "metrics_purpose": "gate_acceptance",
            },
            {
                "status": "multi_reviewer_unadjudicated",
                "reviewer_count": 2,
                "gate_eligible": True,
            },
            "requires adjudicated=true",
        ),
        # gate purpose without gate_eligible
        (
            {
                "dataset_kind": "sampled_production",
                "labeling_status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
                "adjudicated": True,
                "real_ingested_evidence": True,
                "metrics_purpose": "final_acceptance",
            },
            {
                "status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
                "adjudicated": True,
            },
            "but gate_eligible is false",
        ),
        # gate eligible with development purpose
        (
            {
                "dataset_kind": "sampled_production",
                "labeling_status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
                "adjudicated": True,
                "real_ingested_evidence": True,
                "gate_eligible": True,
            },
            {
                "status": "multi_reviewer_adjudicated",
                "reviewer_count": 2,
                "adjudicated": True,
                "gate_eligible": True,
            },
            "requires a metrics_purpose in",
        ),
    ],
)
def test_a_contradictory_trust_combination_is_refused(
    tmp_path, overrides, labeling_overrides, message
):
    trust = dict(_TRUST, **overrides)
    labeling = dict(_LABELING, **labeling_overrides)
    provenance = dict(_PROVENANCE)
    if trust["dataset_kind"] != "synthetic_development":
        provenance = {
            "kind": "sampled",
            "collection_method": "sampled",
            "statement": "sampled from stored raw_items",
            "urls_are_synthetic": False,
            "uses_real_outlet_names": True,
            "ingestion_source": "I2 Yahoo fetcher, raw_items",
            "sample_selection": "stratified, seeded",
        }
    meta_path = write_set(
        tmp_path,
        [_pair("P001")],
        {"trust_contract": trust, "labeling": labeling, "provenance": provenance},
    )

    with pytest.raises(TrustContractError, match=message):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", ["adjudicated", "gate_eligible"])
@pytest.mark.parametrize("value", ["false", 0, None])
def test_adjudicated_and_gate_eligible_must_be_booleans(tmp_path, field, value):
    trust = dict(_TRUST, **{field: value})
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="must be a boolean"):
        load_pair_set(meta_path)


# --------------------------------------------------------------------------
# Cluster-member accounting
# --------------------------------------------------------------------------


def clusterer(builder):
    return lambda entry: builder(entry)


def singletons(entry):
    return [[item_id] for item_id in sorted(entry.item_ids)]


@pytest.mark.parametrize(
    "builder,message,attribute,expected",
    [
        (
            lambda entry: singletons(entry) + [["INVENTED-1"]],
            "not in the case",
            "unexpected_item_ids",
            ("INVENTED-1",),
        ),
        (
            lambda entry: singletons(entry)[1:],
            "missing",
            "missing_item_ids",
            None,
        ),
        (
            lambda entry: singletons(entry) + [[sorted(entry.item_ids)[0]]],
            "in more than one group",
            "duplicated_item_ids",
            None,
        ),
        (
            lambda entry: singletons(entry) + [[]],
            "empty group",
            "missing_item_ids",
            (),
        ),
    ],
)
def test_a_partition_that_does_not_account_for_the_case_fails_it(
    builder, message, attribute, expected
):
    report = evaluate_clusters(
        default_cluster_cases(), clusterer(builder), name="broken"
    )

    assert not report.complete
    assert report.failed_case_count == len(default_cluster_cases())
    assert report.scored_case_count == 0
    assert report.outcomes == ()
    assert message in report.failures[0].message
    assert report.failures[0].error_type == "PartitionAccountingError"
    if expected is not None:
        assert getattr(report, attribute) == expected or expected == ()
    assert report.accounting_failure_ids == tuple(
        sorted(entry.case_id for entry in default_cluster_cases())
    )


def test_an_invented_id_never_earns_perfect_credit():
    """The failure mode this whole check exists for."""

    def almost_right(entry):
        base = [sorted(group) for group in entry.exact_stage_partition]
        return base + [["GHOST"]]

    report = evaluate_clusters(
        default_cluster_cases(), almost_right, name="ghost-writer"
    )

    assert report.exact_partition_matches == 0
    assert report.pairwise.precision is None
    assert report.pairwise.recall is None
    assert report.unexpected_item_ids == ("GHOST",)
    assert not report.complete


def test_the_accounting_violation_names_every_id_class():
    def messy(entry):
        members = sorted(entry.item_ids)
        return [[members[0]], [members[0]], ["GHOST"]]

    report = evaluate_clusters(default_cluster_cases(), messy, name="messy")
    violation = report.accounting_violations[0]

    assert violation.duplicated_item_ids == (sorted(case("C001").item_ids)[0],)
    assert violation.unexpected_item_ids == ("GHOST",)
    assert violation.missing_item_ids


def test_the_partition_validator_rejects_each_defect_directly():
    universe = frozenset({"a", "b"})

    with pytest.raises(PartitionAccountingError, match="not in the case"):
        validate_predicted_partition([["a"], ["b"], ["c"]], universe, where="t")
    with pytest.raises(PartitionAccountingError, match="missing"):
        validate_predicted_partition([["a"]], universe, where="t")
    with pytest.raises(PartitionAccountingError, match="more than one group"):
        validate_predicted_partition([["a", "b"], ["a"]], universe, where="t")
    with pytest.raises(PartitionAccountingError, match="empty group"):
        validate_predicted_partition([["a", "b"], []], universe, where="t")
    with pytest.raises(PartitionAccountingError, match="non-blank strings"):
        validate_predicted_partition([["a", "b"], [" "]], universe, where="t")
    with pytest.raises(PartitionAccountingError, match="sequence of groups"):
        validate_predicted_partition("ab", universe, where="t")


def test_a_valid_partition_passes_the_validator():
    universe = frozenset({"a", "b", "c"})

    assert validate_predicted_partition(
        [["b", "a"], ["c"]], universe, where="t"
    ) == canonical_partition([frozenset({"a", "b"}), frozenset({"c"})])


# --------------------------------------------------------------------------
# Cluster scoring against the corrected fixture
# --------------------------------------------------------------------------


def test_m2_reproduces_the_exact_stage_partition_on_every_case():
    report = evaluate_m2_clusters(target="exact_stage_partition")

    assert report.exact_partition_matches == report.scored_case_count == 9
    assert report.exact_partition_rate == 1.0
    assert report.pairwise.precision == 1.0
    assert report.pairwise.recall == 1.0
    assert report.over_merge_case_ids == ()
    assert report.under_merge_case_ids == ()
    assert report.complete
    assert report.accounting_violations == ()


def test_m2_under_merges_only_where_the_design_says_it_should():
    report = evaluate_m2_clusters(target="expected_partition")

    assert set(report.under_merge_case_ids) == {"C003", "C006", "C007"}
    assert report.over_merge_case_ids == ()
    assert report.pairwise.precision == 1.0
    assert report.pairwise.recall is not None and report.pairwise.recall < 1.0


def test_over_merging_is_detected_and_named():
    report = evaluate_clusters(
        default_cluster_cases(),
        lambda entry: [sorted(entry.item_ids)],
        name="merge-everything",
    )

    assert set(report.over_merge_case_ids) >= {"C001", "C004", "C005"}
    assert report.exact_partition_matches < report.scored_case_count
    assert report.pairwise.precision is not None and report.pairwise.precision < 1.0


def test_under_merging_is_detected_and_named():
    report = evaluate_clusters(
        default_cluster_cases(), singletons, name="merge-nothing"
    )

    assert set(report.under_merge_case_ids) >= {"C002", "C004", "C007", "C008"}
    assert report.over_merge_case_ids == ()
    assert report.pairwise.recall == 0.0


def test_a_transitive_bridge_is_caught_as_an_over_merge():
    report = evaluate_clusters(
        default_cluster_cases(),
        lambda entry: [sorted(entry.item_ids)],
        name="transitive",
    )
    outcome = next(o for o in report.outcomes if o.case_id == "C001")

    assert ("C001-1", "C001-3") in outcome.over_merged_pairs
    assert not outcome.exact_match


def test_a_raising_case_is_reported_rather_than_ending_the_cluster_run():
    def explode(entry):
        if entry.case_id == "C003":
            raise RuntimeError("stage exploded")
        return singletons(entry)

    report = evaluate_clusters(default_cluster_cases(), explode, name="flaky")

    assert report.failed_case_ids == ("C003",)
    assert not report.complete
    assert all(outcome.case_id != "C003" for outcome in report.outcomes)
    assert report.scored_case_count == 8


# --------------------------------------------------------------------------
# Permutation coverage
# --------------------------------------------------------------------------


def test_small_cases_are_checked_under_every_ordering():
    import math

    for entry in default_cluster_cases():
        orderings = permutations_of(entry)
        assert len(orderings) == math.factorial(len(entry.items)), entry.case_id
        assert len({tuple(i.item_id for i in o) for o in orderings}) == len(orderings)


def test_a_large_case_uses_a_documented_representative_set():
    from dataclasses import replace
    from nlp.eval.clusters import MAX_EXHAUSTIVE_PERMUTATIONS, REPRESENTATIVE_SHUFFLES

    big = case("C008")
    big = replace(big, items=big.items + big.items[:3])
    orderings = permutations_of(big)

    assert len(big.items) == 8
    assert len(orderings) < MAX_EXHAUSTIVE_PERMUTATIONS
    # original, reverse, 7 rotations, then the seeded shuffles
    assert len(orderings) >= 2 + (len(big.items) - 1)
    assert orderings[0] == big.items
    assert orderings[1] == tuple(reversed(big.items))
    assert permutations_of(big) == orderings  # deterministic, seeded on case_id
    assert REPRESENTATIVE_SHUFFLES >= 4


def test_every_case_reports_its_permutation_count():
    report = evaluate_m2_clusters()

    for outcome in report.outcomes:
        assert outcome.permutation_count >= 6
        assert outcome.unstable_permutation_count == 0
    assert report.permutation_failures == ()


def test_permutation_instability_is_detected_and_counted():
    seen: dict[str, int] = {}

    def order_sensitive(entry):
        seen[entry.case_id] = seen.get(entry.case_id, 0) + 1
        if seen[entry.case_id] == 1:
            return [sorted(entry.item_ids)]
        return singletons(entry)

    report = evaluate_clusters(
        default_cluster_cases(), order_sensitive, name="order-sensitive"
    )

    assert report.permutation_failures
    unstable = next(o for o in report.outcomes if not o.permutation_stable)
    assert unstable.unstable_permutation_count == unstable.permutation_count - 1


# --------------------------------------------------------------------------
# Shared gate-value validator
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,message",
    [
        (float("nan"), "real number"),
        (float("inf"), "finite"),
        (float("-inf"), "finite"),
        (-0.1, r"\[0.0, 1.0\]"),
        (1.1, r"\[0.0, 1.0\]"),
        ("0.8", "must be a number"),
        (True, "not a boolean"),
        (False, "not a boolean"),
        (None, "must be a number"),
    ],
)
def test_the_shared_validator_rejects_every_unusable_gate_value(value, message):
    with pytest.raises(GateValueError, match=message):
        validate_unit_interval(value, "field")


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0, 1])
def test_the_shared_validator_accepts_a_real_probability(value):
    assert validate_unit_interval(value, "field") == float(value)


def test_the_optional_validator_passes_none_through():
    assert validate_optional_unit_interval(None, "field") is None
    with pytest.raises(GateValueError):
        validate_optional_unit_interval(float("nan"), "field")


@pytest.mark.parametrize(
    "thresholds,message",
    [
        ([], "at least one threshold"),
        ([0.8, 0.8], "must be distinct"),
        ([float("nan")], "real number"),
        ([float("inf")], "finite"),
        ([float("-inf")], "finite"),
        ([-0.1], r"\[0.0, 1.0\]"),
        ([1.5], r"\[0.0, 1.0\]"),
        (["0.8"], "must be a number"),
        ([True], "not a boolean"),
    ],
)
def test_an_unusable_sweep_threshold_is_refused(thresholds, message):
    with pytest.raises(GateValueError, match=message):
        validate_thresholds(thresholds)


def test_a_report_threshold_goes_through_the_same_validator(tmp_path):
    pair_set = load_pair_set(write_set(tmp_path, [_pair("P001")]))

    with pytest.raises(GateValueError, match="real number"):
        evaluate_isolated_pairs(
            pair_set, constant(False), name="x", threshold=float("nan")
        )


@pytest.mark.parametrize(
    "flag",
    [
        "--precision-floor=nan",
        "--recall-floor=nan",
        "--precision-floor=inf",
        "--recall-floor=-inf",
        "--precision-floor=-0.1",
        "--recall-floor=1.1",
    ],
)
def test_the_cli_exits_non_zero_on_an_unusable_gate_value(flag):
    completed = subprocess.run(
        [sys.executable, "-m", "tools.eval_dedup", "--stage", "m2", flag],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid gate value" in completed.stderr


def test_a_valid_cli_floor_still_runs():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.eval_dedup",
            "--stage",
            "m2",
            "--precision-floor=0.85",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0


# --------------------------------------------------------------------------
# Every payload carries the trust block
# --------------------------------------------------------------------------


def all_payloads():
    """Every public JSON-producing function, with a real argument."""

    pair_report = evaluate_m2_isolated_pairs()
    cluster_report = evaluate_m2_clusters()
    points = sweep_thresholds(
        default_pair_set(),
        lambda threshold: (lambda pair: PairPrediction(merged=False, score=threshold)),
        [0.5, 0.9],
        name="fake",
    )
    return {
        "to_payload": to_payload(pair_report),
        "cluster_payload": cluster_payload(cluster_report),
        "sweep_payload": sweep_payload(points),
    }


@pytest.mark.parametrize("name", ["to_payload", "cluster_payload", "sweep_payload"])
def test_every_public_payload_carries_the_full_contract(name):
    payload = all_payloads()[name]

    for field in sorted(_TRUST):
        assert field in payload["trust_contract"], (name, field)
    assert payload["trust_contract"]["warning"].startswith("WARNING:")
    assert payload["dataset_id"]
    assert payload["schema_version"]
    assert payload["scope"]
    assert payload["limitation"]


@pytest.mark.parametrize("name", ["to_payload", "cluster_payload", "sweep_payload"])
def test_every_public_payload_reports_completeness(name):
    payload = all_payloads()[name]
    block = payload.get("completeness", payload)

    assert block["complete"] is True
    assert block["evaluated_case_count"] > 0
    assert block["failed_case_count"] == 0
    assert block["failed_case_ids"] == []


def test_the_raw_sweep_rows_are_inside_the_contract_document():
    """The bare list a caller could paste without provenance is gone."""

    payload = all_payloads()["sweep_payload"]

    assert isinstance(payload, dict)
    assert isinstance(payload["points"], list)
    assert len(payload["points"]) == 2
    for point in payload["points"]:
        assert point["scope"] == "isolated_pairs"
        assert point["complete"] is True
        assert "evaluated_case_count" in point


def test_a_sweep_report_carries_the_contract_in_rendered_form():
    points = sweep_thresholds(
        default_pair_set(),
        lambda threshold: (lambda pair: PairPrediction(merged=False, score=threshold)),
        [0.5, 0.9],
        name="fake",
    )

    assert render_sweep(points).startswith("WARNING:")


def test_the_payload_versions_are_stated():
    payloads = all_payloads()

    assert payloads["to_payload"]["schema_version"] == ISOLATED_PAIR_PAYLOAD_VERSION
    assert payloads["cluster_payload"]["schema_version"] == CLUSTER_PAYLOAD_VERSION
    assert payloads["sweep_payload"]["schema_version"] == SWEEP_PAYLOAD_VERSION


# --------------------------------------------------------------------------
# The dataset-kind invariant matrix
# --------------------------------------------------------------------------

from nlp.eval.trust import (  # noqa: E402
    DATASET_KIND_RULES,
    DatasetKind,
    LabelingStatus,
    MetricsPurpose,
    TrustContract,
    derive_trust_summary,
    rules_for,
)

_PRODUCTION_PROVENANCE = {
    "kind": "sampled",
    "collection_method": "sampled",
    "statement": "sampled from stored raw_items",
    "urls_are_synthetic": False,
    "uses_real_outlet_names": True,
    "ingestion_source": "I2 Yahoo fetcher, raw_items table, 2026-03-02..06",
    "sample_selection": "stratified by similarity band, seeded",
}


def production_trust(**overrides) -> dict:
    base = {
        "dataset_kind": "sampled_production",
        "real_ingested_evidence": True,
        "labeling_status": "multi_reviewer_adjudicated",
        "reviewer_count": 2,
        "adjudicated": True,
        "gate_eligible": False,
        "metrics_purpose": "development_regression_only",
    }
    base.update(overrides)
    return base


def production_labeling(trust: dict, **overrides) -> dict:
    base = {
        "status": trust["labeling_status"],
        "reviewer_count": trust["reviewer_count"],
        "reviewers": [],
        "adjudicated": trust["adjudicated"],
        "gate_eligible": trust["gate_eligible"],
        "protocol": "co-labelled and adjudicated under K3",
    }
    base.update(overrides)
    return base


def write_production_set(tmp_path, trust: dict, provenance: dict | None = None):
    return write_set(
        tmp_path,
        [_pair("P001")],
        {
            "trust_contract": trust,
            "labeling": production_labeling(trust),
            "provenance": provenance or dict(_PRODUCTION_PROVENANCE),
        },
    )


def test_the_matrix_covers_every_dataset_kind():
    """A new kind cannot silently inherit permissive behaviour."""

    assert set(DATASET_KIND_RULES) == set(DatasetKind)
    for kind in DatasetKind:
        assert rules_for(kind).kind is kind


def test_sampled_production_without_real_evidence_is_refused(tmp_path):
    """The defect this correction exists for."""

    meta_path = write_production_set(
        tmp_path, production_trust(real_ingested_evidence=False)
    )

    with pytest.raises(
        TrustContractError, match="requires real_ingested_evidence=true"
    ):
        load_pair_set(meta_path)


def test_synthetic_claiming_real_evidence_is_still_refused(tmp_path):
    trust = dict(_TRUST, real_ingested_evidence=True)
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(
        TrustContractError, match="requires real_ingested_evidence=false"
    ):
        load_pair_set(meta_path)


@pytest.mark.parametrize(
    "provenance_overrides,message",
    [
        ({"kind": "synthetic"}, "requires a provenance.kind in"),
        ({"collection_method": "authored"}, "requires a provenance.collection_method"),
        ({"urls_are_synthetic": True}, "urls_are_synthetic=false"),
    ],
)
def test_sampled_production_may_not_claim_synthetic_provenance(
    tmp_path, provenance_overrides, message
):
    provenance = dict(_PRODUCTION_PROVENANCE, **provenance_overrides)
    meta_path = write_production_set(tmp_path, production_trust(), provenance)

    with pytest.raises(TrustContractError, match=message):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", ["ingestion_source", "sample_selection"])
def test_sampled_production_must_identify_its_source(tmp_path, field):
    provenance = {
        key: value for key, value in _PRODUCTION_PROVENANCE.items() if key != field
    }
    meta_path = write_production_set(tmp_path, production_trust(), provenance)

    with pytest.raises(TrustContractError, match="provenance is missing"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", ["ingestion_source", "sample_selection"])
def test_a_blank_source_identification_is_refused(tmp_path, field):
    provenance = dict(_PRODUCTION_PROVENANCE, **{field: "   "})
    meta_path = write_production_set(tmp_path, production_trust(), provenance)

    with pytest.raises(TrustContractError, match=f"provenance.{field}"):
        load_pair_set(meta_path)


def test_a_synthetic_set_need_not_name_an_ingestion_source():
    """The extra fields are per kind, not universal."""

    assert (
        "ingestion_source"
        not in rules_for(DatasetKind.SYNTHETIC_DEVELOPMENT).extra_provenance_fields
    )
    assert (
        "why_synthetic"
        in rules_for(DatasetKind.SYNTHETIC_DEVELOPMENT).extra_provenance_fields
    )


def test_valid_unadjudicated_production_evidence_loads(tmp_path):
    trust = production_trust(
        labeling_status="multi_reviewer_unadjudicated", adjudicated=False
    )
    pair_set = load_pair_set(write_production_set(tmp_path, trust))

    assert pair_set.trust.dataset_kind is DatasetKind.SAMPLED_PRODUCTION
    assert pair_set.trust.real_ingested_evidence is True
    assert pair_set.trust.gate_eligible is False


def test_valid_adjudicated_non_gate_production_evaluation_loads(tmp_path):
    pair_set = load_pair_set(write_production_set(tmp_path, production_trust()))

    assert pair_set.trust.adjudicated is True
    assert pair_set.trust.gate_eligible is False
    assert pair_set.trust.metrics_purpose is MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY


def test_a_valid_gate_eligible_production_contract_loads(tmp_path):
    trust = production_trust(gate_eligible=True, metrics_purpose="final_acceptance")
    pair_set = load_pair_set(write_production_set(tmp_path, trust))

    assert pair_set.trust.gate_eligible is True
    assert pair_set.trust.metrics_purpose is MetricsPurpose.FINAL_ACCEPTANCE
    assert pair_set.trust.reviewer_count == 2


def test_gate_eligible_production_without_adjudication_is_refused(tmp_path):
    trust = production_trust(
        labeling_status="multi_reviewer_unadjudicated",
        adjudicated=False,
        gate_eligible=True,
        metrics_purpose="gate_acceptance",
    )
    meta_path = write_production_set(tmp_path, trust)

    with pytest.raises(TrustContractError, match="requires adjudicated=true"):
        load_pair_set(meta_path)


def test_gate_eligible_production_with_one_reviewer_is_refused(tmp_path):
    trust = production_trust(
        labeling_status="single_author_unadjudicated",
        reviewer_count=1,
        adjudicated=False,
        gate_eligible=True,
        metrics_purpose="gate_acceptance",
    )
    meta_path = write_production_set(tmp_path, trust)

    with pytest.raises(TrustContractError, match="cannot be gate_eligible"):
        load_pair_set(meta_path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("dataset_kind", "warehouse_dump", "is not one of"),
        ("metrics_purpose", "looks_fine", "is not one of"),
        ("labeling_status", "somebody_checked", "is not one of"),
    ],
)
def test_an_unknown_enum_value_is_refused_for_production_too(
    tmp_path, field, value, message
):
    meta_path = write_production_set(tmp_path, production_trust(**{field: value}))

    with pytest.raises(TrustContractError, match=message):
        load_pair_set(meta_path)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"reviewer_count": 1}, "requires reviewer_count >= 2"),
        (
            {"labeling_status": "multi_reviewer_unadjudicated"},
            "requires an adjudicated labeling_status",
        ),
        (
            {"adjudicated": False},
            "but adjudicated=false",
        ),
    ],
)
def test_inconsistent_reviewer_and_adjudication_fields_are_refused(
    tmp_path, overrides, message
):
    trust = production_trust(**overrides)
    meta_path = write_set(
        tmp_path,
        [_pair("P001")],
        {
            "trust_contract": trust,
            "labeling": production_labeling(trust),
            "provenance": dict(_PRODUCTION_PROVENANCE),
        },
    )

    with pytest.raises(TrustContractError, match=message):
        load_pair_set(meta_path)


# --------------------------------------------------------------------------
# The derived banner
# --------------------------------------------------------------------------


def contract_of(**overrides) -> TrustContract:
    base = {
        "dataset_kind": DatasetKind.SAMPLED_PRODUCTION,
        "real_ingested_evidence": True,
        "labeling_status": LabelingStatus.MULTI_REVIEWER_ADJUDICATED,
        "reviewer_count": 2,
        "adjudicated": True,
        "gate_eligible": False,
        "metrics_purpose": MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY,
    }
    base.update(overrides)
    return TrustContract(**base)


def test_the_current_synthetic_state_produces_the_expected_warning():
    summary = default_pair_set().trust.summary

    assert summary.level == "WARNING"
    assert summary.text == (
        "WARNING: Synthetic, single-author, unadjudicated development dataset.\n"
        "Metrics are not valid for K3/G4 or final AC-3 acceptance."
    )


def test_unadjudicated_production_evidence_produces_the_expected_warning():
    summary = derive_trust_summary(
        contract_of(
            labeling_status=LabelingStatus.MULTI_REVIEWER_UNADJUDICATED,
            adjudicated=False,
        )
    )

    assert summary.text == (
        "WARNING: Production-sampled evidence has not completed independent "
        "adjudication.\n"
        "Metrics are development-only and not gate eligible."
    )


def test_adjudicated_non_gate_production_produces_the_expected_notice():
    summary = derive_trust_summary(contract_of())

    assert summary.text == (
        "NOTICE: Production-sampled, independently adjudicated evaluation "
        "dataset.\n"
        "Metrics are not configured as a release gate."
    )


def test_a_gate_eligible_dataset_produces_the_expected_notice():
    summary = derive_trust_summary(
        contract_of(gate_eligible=True, metrics_purpose=MetricsPurpose.FINAL_ACCEPTANCE)
    )

    assert summary.text == (
        "NOTICE: Production-sampled, independently adjudicated gate-eligible "
        "dataset."
    )
    assert summary.level == "NOTICE"


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "labeling_status": LabelingStatus.MULTI_REVIEWER_UNADJUDICATED,
            "adjudicated": False,
        },
        {},
        {"gate_eligible": True, "metrics_purpose": MetricsPurpose.FINAL_ACCEPTANCE},
    ],
)
def test_no_production_state_ever_receives_synthetic_wording(overrides):
    summary = derive_trust_summary(contract_of(**overrides))

    assert "ynthetic" not in summary.text
    assert "Production-sampled" in summary.headline


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "labeling_status": LabelingStatus.MULTI_REVIEWER_UNADJUDICATED,
            "reviewer_count": 3,
        },
        {
            "labeling_status": LabelingStatus.MULTI_REVIEWER_ADJUDICATED,
            "reviewer_count": 4,
            "adjudicated": True,
        },
    ],
)
def test_no_synthetic_state_ever_receives_a_production_notice(overrides):
    settings = {
        "dataset_kind": DatasetKind.SYNTHETIC_DEVELOPMENT,
        "real_ingested_evidence": False,
        "labeling_status": LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED,
        "reviewer_count": 1,
        "adjudicated": False,
    }
    settings.update(overrides)
    summary = derive_trust_summary(contract_of(**settings))

    assert summary.level == "WARNING"
    assert summary.headline.startswith("Synthetic,")
    assert "Production-sampled" not in summary.text
    assert "adjudicated" in summary.headline or "unadjudicated" in summary.headline


def test_the_labeling_phrase_is_derived_from_the_reviewer_fields():
    single = derive_trust_summary(
        contract_of(
            dataset_kind=DatasetKind.SYNTHETIC_DEVELOPMENT,
            real_ingested_evidence=False,
            labeling_status=LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED,
            reviewer_count=1,
            adjudicated=False,
        )
    )
    several = derive_trust_summary(
        contract_of(
            dataset_kind=DatasetKind.SYNTHETIC_DEVELOPMENT,
            real_ingested_evidence=False,
            labeling_status=LabelingStatus.MULTI_REVIEWER_ADJUDICATED,
            reviewer_count=3,
            adjudicated=True,
        )
    )

    assert "single-author, unadjudicated" in single.headline
    assert "3-reviewer, independently adjudicated" in several.headline


def test_a_manifest_may_not_supply_its_own_banner(tmp_path):
    trust = dict(_TRUST, warning="NOTICE: everything is fine, ship it")
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="may not supply a warning"):
        load_pair_set(meta_path)


def test_an_unknown_trust_field_is_refused(tmp_path):
    trust = dict(_TRUST, banner_override="anything")
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="unknown field"):
        load_pair_set(meta_path)


def test_the_committed_manifests_supply_no_banner():
    for name in ("dedup_pairs.meta.json", "cluster_cases.meta.json"):
        payload = json.loads(
            (REPO_ROOT / "nlp" / "eval" / "data" / name).read_text("utf-8")
        )
        assert "warning" not in payload["trust_contract"], name


@pytest.mark.parametrize("name", ["to_payload", "cluster_payload", "sweep_payload"])
def test_every_payload_carries_both_the_block_and_the_derived_summary(name):
    payload = all_payloads()[name]

    assert set(payload["trust_summary"]) == {"level", "headline", "detail", "text"}
    assert payload["trust_summary"]["text"] == payload["trust_contract"]["warning"]
    assert payload["trust_summary"]["level"] == "WARNING"
    assert payload["trust_contract"]["dataset_kind"] == "synthetic_development"


def test_both_renderers_use_the_derived_summary():
    pair_report = evaluate_m2_isolated_pairs()
    cluster_report = evaluate_m2_clusters()

    for rendered, report in (
        (render_text(pair_report), pair_report),
        (render_clusters(cluster_report), cluster_report),
    ):
        assert rendered.splitlines()[0] == report.trust.summary.text.splitlines()[0]


def test_no_renderer_hardcodes_a_dataset_judgement():
    """The wording lives in one derivation, not in the renderers."""

    for name in ("report.py", "metrics.py", "clusters.py", "dataset.py"):
        source = (REPO_ROOT / "nlp" / "eval" / name).read_text("utf-8")
        assert "Synthetic, single-author" not in source, name


# --------------------------------------------------------------------------
# The construction boundary
# --------------------------------------------------------------------------

_SYNTHETIC_FIELDS = {
    "dataset_kind": "synthetic_development",
    "real_ingested_evidence": False,
    "labeling_status": "single_author_unadjudicated",
    "reviewer_count": 1,
    "adjudicated": False,
    "gate_eligible": False,
    "metrics_purpose": "development_regression_only",
}
_PRODUCTION_FIELDS = {
    "dataset_kind": "sampled_production",
    "real_ingested_evidence": True,
    "labeling_status": "multi_reviewer_adjudicated",
    "reviewer_count": 2,
    "adjudicated": True,
    "gate_eligible": False,
    "metrics_purpose": "development_regression_only",
}


def build(base: dict, **overrides) -> TrustContract:
    return TrustContract(**{**base, **overrides})


@pytest.mark.parametrize(
    "base,overrides,message",
    [
        (
            _PRODUCTION_FIELDS,
            {"real_ingested_evidence": False},
            "requires real_ingested_evidence=true",
        ),
        (
            _SYNTHETIC_FIELDS,
            {"gate_eligible": True},
            "cannot claim gate_eligible",
        ),
        (
            _SYNTHETIC_FIELDS,
            {"metrics_purpose": "final_acceptance"},
            "must declare a metrics_purpose in",
        ),
        (
            _SYNTHETIC_FIELDS,
            {"adjudicated": True},
            "cannot be adjudicated",
        ),
        (
            _SYNTHETIC_FIELDS,
            {"reviewer_count": 4},
            "requires reviewer_count=1",
        ),
        (
            _PRODUCTION_FIELDS,
            {"labeling_status": "multi_reviewer_unadjudicated"},
            "requires an adjudicated labeling_status",
        ),
        (
            _PRODUCTION_FIELDS,
            {"reviewer_count": 1},
            "requires reviewer_count >= 2",
        ),
        (
            _PRODUCTION_FIELDS,
            {
                "real_ingested_evidence": False,
                "gate_eligible": True,
                "metrics_purpose": "gate_acceptance",
            },
            "requires real_ingested_evidence=true",
        ),
        (
            _PRODUCTION_FIELDS,
            {
                "labeling_status": "multi_reviewer_unadjudicated",
                "adjudicated": False,
                "gate_eligible": True,
                "metrics_purpose": "gate_acceptance",
            },
            "requires adjudicated=true",
        ),
        (
            _PRODUCTION_FIELDS,
            {
                "labeling_status": "single_author_unadjudicated",
                "reviewer_count": 1,
                "adjudicated": False,
                "gate_eligible": True,
                "metrics_purpose": "gate_acceptance",
            },
            "cannot be gate_eligible",
        ),
        (
            _PRODUCTION_FIELDS,
            {"gate_eligible": True},
            "requires a metrics_purpose in",
        ),
        (
            _PRODUCTION_FIELDS,
            {"metrics_purpose": "final_acceptance"},
            "but gate_eligible is false",
        ),
        (_SYNTHETIC_FIELDS, {"dataset_kind": "warehouse_dump"}, "is not one of"),
        (_SYNTHETIC_FIELDS, {"labeling_status": "somebody_looked"}, "is not one of"),
        (_SYNTHETIC_FIELDS, {"metrics_purpose": "shipping"}, "is not one of"),
        (_SYNTHETIC_FIELDS, {"dataset_kind": 7}, "must be a non-blank string"),
        (_SYNTHETIC_FIELDS, {"real_ingested_evidence": 0}, "must be a boolean"),
        (_SYNTHETIC_FIELDS, {"adjudicated": "false"}, "must be a boolean"),
        (_SYNTHETIC_FIELDS, {"gate_eligible": None}, "must be a boolean"),
        (_SYNTHETIC_FIELDS, {"reviewer_count": True}, "must be a positive integer"),
        (_SYNTHETIC_FIELDS, {"reviewer_count": 0}, "must be a positive integer"),
        (_SYNTHETIC_FIELDS, {"reviewer_count": "1"}, "must be a positive integer"),
    ],
)
def test_direct_construction_rejects_an_invalid_contract(base, overrides, message):
    with pytest.raises(TrustContractError, match=message):
        build(base, **overrides)


@pytest.mark.parametrize(
    "fields",
    [
        _SYNTHETIC_FIELDS,
        dict(
            _PRODUCTION_FIELDS,
            labeling_status="multi_reviewer_unadjudicated",
            adjudicated=False,
        ),
        _PRODUCTION_FIELDS,
        dict(
            _PRODUCTION_FIELDS,
            gate_eligible=True,
            metrics_purpose="final_acceptance",
        ),
    ],
    ids=["synthetic", "production-unadjudicated", "production-adjudicated", "gate"],
)
def test_direct_construction_accepts_every_valid_state(fields):
    contract = TrustContract(**fields)

    assert contract.dataset_kind is DatasetKind(fields["dataset_kind"])
    assert contract.labeling_status is LabelingStatus(fields["labeling_status"])
    assert contract.metrics_purpose is MetricsPurpose(fields["metrics_purpose"])
    # The summary derives from the object, and the block round-trips.
    assert contract.summary.text == contract.warning
    assert contract.as_dict()["warning"] == contract.summary.text
    assert contract.summary.level in {"WARNING", "NOTICE"}


@pytest.mark.parametrize(
    "fields",
    [
        _SYNTHETIC_FIELDS,
        dict(
            _PRODUCTION_FIELDS,
            labeling_status="multi_reviewer_unadjudicated",
            adjudicated=False,
        ),
        _PRODUCTION_FIELDS,
        dict(
            _PRODUCTION_FIELDS,
            gate_eligible=True,
            metrics_purpose="final_acceptance",
        ),
    ],
    ids=["synthetic", "production-unadjudicated", "production-adjudicated", "gate"],
)
def test_a_directly_built_contract_is_accepted_by_the_public_report_apis(fields):
    """A valid contract of any state travels through the payload builders."""

    from dataclasses import replace

    contract = TrustContract(**fields)
    report = replace(evaluate_m2_isolated_pairs(), trust=contract)
    payload = to_payload(report)

    assert payload["trust_contract"] == contract.as_dict()
    assert payload["trust_summary"] == contract.summary.as_dict()
    assert render_text(report).splitlines()[0] == contract.summary.text.splitlines()[0]


def test_the_parser_and_the_constructor_produce_the_same_object(tmp_path):
    """Not two paths with two standards: parsing is a constructor call."""

    parsed = load_pair_set(write_set(tmp_path, [_pair("P001")])).trust
    direct = TrustContract(**_SYNTHETIC_FIELDS)

    assert parsed == direct
    assert parsed.as_dict() == direct.as_dict()
    assert parsed.summary == direct.summary
    assert parsed.banner() == direct.banner()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"real_ingested_evidence": True}, "requires real_ingested_evidence=false"),
        ({"gate_eligible": True}, "cannot claim gate_eligible"),
        ({"adjudicated": True}, "cannot be adjudicated"),
        ({"metrics_purpose": "gate_acceptance"}, "must declare a metrics_purpose in"),
        ({"reviewer_count": 5}, "requires reviewer_count=1"),
    ],
)
def test_the_parser_reports_the_same_objection_as_the_constructor(
    tmp_path, overrides, message
):
    trust = dict(_TRUST, **overrides)
    labeling = dict(_LABELING)
    for field in ("adjudicated", "gate_eligible", "reviewer_count"):
        if field in overrides:
            labeling[field] = overrides[field]
    meta_path = write_set(
        tmp_path, [_pair("P001")], {"trust_contract": trust, "labeling": labeling}
    )

    with pytest.raises(TrustContractError, match=message):
        load_pair_set(meta_path)
    with pytest.raises(TrustContractError, match=message):
        build(_SYNTHETIC_FIELDS, **overrides)


def test_the_parser_adds_the_file_location_to_the_constructor_message(tmp_path):
    meta_path = write_set(
        tmp_path,
        [_pair("P001")],
        {"trust_contract": dict(_TRUST, gate_eligible=True)},
    )

    with pytest.raises(TrustContractError) as excinfo:
        load_pair_set(meta_path)

    assert str(meta_path) in str(excinfo.value)
    assert "cannot claim gate_eligible" in str(excinfo.value)


# --------------------------------------------------------------------------
# No unchecked escape hatch
# --------------------------------------------------------------------------


def test_dataclasses_replace_cannot_derive_an_invalid_contract():
    from dataclasses import replace

    valid = TrustContract(**_SYNTHETIC_FIELDS)

    with pytest.raises(TrustContractError, match="cannot claim gate_eligible"):
        replace(valid, gate_eligible=True)
    with pytest.raises(TrustContractError, match="cannot be adjudicated"):
        replace(valid, adjudicated=True)
    with pytest.raises(TrustContractError, match="is not one of"):
        replace(valid, dataset_kind="warehouse_dump")


def test_dataclasses_replace_still_works_for_a_valid_change():
    from dataclasses import replace

    valid = TrustContract(**_PRODUCTION_FIELDS)
    gated = replace(
        valid, gate_eligible=True, metrics_purpose=MetricsPurpose.FINAL_ACCEPTANCE
    )

    assert gated.gate_eligible is True
    assert gated.summary.level == "NOTICE"
    assert "gate-eligible" in gated.summary.headline


def test_the_contract_stays_immutable_after_construction():
    import dataclasses

    contract = TrustContract(**_SYNTHETIC_FIELDS)

    assert dataclasses.fields(contract)
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.gate_eligible = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.reviewer_count = 9  # type: ignore[misc]


def test_there_is_one_invariant_definition_and_the_constructor_calls_it():
    """No second copy of the matrix to drift away from the first."""

    source = (REPO_ROOT / "nlp" / "eval" / "trust.py").read_text("utf-8")

    from nlp.eval.trust import check_trust_invariants

    assert callable(check_trust_invariants)
    assert source.count("def check_trust_invariants(") == 1
    assert source.count("check_trust_invariants(self)") == 1
    assert source.count("DATASET_KIND_RULES: dict") == 1
    # parse_trust_contract must not re-implement the field checks.
    parse = source[source.index("def parse_trust_contract(") :]
    parse = parse[: parse.index("def validate_provenance(")]
    assert "check_trust_invariants" not in parse
    assert "TrustContract(" in parse


def test_no_public_helper_constructs_a_contract_without_validation():
    """Every construction site in the package goes through __init__."""

    import re

    for name in ("trust.py", "dataset.py", "clusters.py", "metrics.py", "report.py"):
        source = (REPO_ROOT / "nlp" / "eval" / name).read_text("utf-8")
        assert "object.__new__(TrustContract" not in source, name
        assert not re.search(r"TrustContract\.__new__", source), name
        # object.__setattr__ on a contract is only allowed inside its own
        # __post_init__, where the coercions happen before validation.
        if name != "trust.py":
            assert "object.__setattr__" not in source, name

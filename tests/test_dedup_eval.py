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
    canonical_partition,
    evaluate_clusters,
    m2_cluster_predictor,
)
from nlp.eval.dataset import DEFAULT_META_PATH
from nlp.eval.dedup import config_for, m2_isolated_pair_predictor, to_raw_items
from nlp.eval.metrics import (
    ISOLATED_PAIR_LIMITATION,
    Confusion,
    Metrics,
)
from nlp.eval.report import (
    cluster_payload,
    render_clusters,
    render_sweep,
    render_text,
    to_payload,
)
from nlp.eval.trust import WARNING_BANNER
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


def test_the_committed_sets_declare_the_full_trust_contract():
    for trust in (default_pair_set().trust, default_cluster_cases().trust):
        assert trust.dataset_kind == "synthetic_development"
        assert trust.real_ingested_evidence is False
        assert trust.labeling_status == "single_author_unadjudicated"
        assert trust.reviewer_count == 1
        assert trust.adjudicated is False
        assert trust.gate_eligible is False
        assert trust.metrics_purpose == "development_regression_only"


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

    with pytest.raises(TrustContractError, match="cannot claim real_ingested_evidence"):
        load_pair_set(meta_path)


def test_synthetic_data_may_not_claim_gate_eligibility(tmp_path):
    trust = dict(_TRUST, gate_eligible=True)
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="cannot claim gate_eligible"):
        load_pair_set(meta_path)


def test_one_reviewer_may_not_claim_adjudication(tmp_path):
    trust = dict(_TRUST, adjudicated=True)
    labeling = dict(_LABELING, adjudicated=True)
    meta_path = write_set(
        tmp_path,
        [_pair("P001")],
        {"trust_contract": trust, "labeling": labeling},
    )

    with pytest.raises(TrustContractError, match="at least two reviewers"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("value", [0, -1, 1.5, "one", True, None])
def test_reviewer_count_must_be_a_positive_integer(tmp_path, value):
    trust = dict(_TRUST, reviewer_count=value)
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="positive integer"):
        load_pair_set(meta_path)


@pytest.mark.parametrize("field", ["adjudicated", "gate_eligible"])
@pytest.mark.parametrize("value", ["false", 0, None])
def test_adjudicated_and_gate_eligible_must_be_booleans(tmp_path, field, value):
    trust = dict(_TRUST, **{field: value})
    meta_path = write_set(tmp_path, [_pair("P001")], {"trust_contract": trust})

    with pytest.raises(TrustContractError, match="must be a boolean"):
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

    with pytest.raises(TrustContractError, match="cannot claim its URLs are real"):
        load_pair_set(meta_path)


def test_a_synthetic_set_may_not_claim_it_was_collected(tmp_path):
    provenance = dict(_PROVENANCE, collection_method="sampled")
    meta_path = write_set(tmp_path, [_pair("P001")], {"provenance": provenance})

    with pytest.raises(TrustContractError, match="nothing here was collected"):
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


def test_a_sweep_report_carries_the_contract_too():
    pair_set = default_pair_set()
    points = sweep_thresholds(
        pair_set,
        lambda threshold: (lambda pair: PairPrediction(merged=False, score=threshold)),
        [0.5, 0.9],
        name="fake",
    )

    assert render_sweep(points).startswith("WARNING:")


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


def test_the_default_banner_matches_the_committed_warning():
    assert default_pair_set().trust.warning.startswith(WARNING_BANNER.split("\n")[0])


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


@pytest.mark.parametrize(
    "thresholds,message",
    [
        ([], "at least one threshold"),
        ([0.8, 0.8], "must be distinct"),
        ([float("nan")], "must be finite"),
        ([float("inf")], "must be finite"),
        ([-0.1], r"\[0.0, 1.0\]"),
        ([1.5], r"\[0.0, 1.0\]"),
        (["0.8"], "must be numbers"),
        ([True], "must be numbers"),
    ],
)
def test_an_unusable_sweep_threshold_is_refused(thresholds, message):
    with pytest.raises(ValueError, match=message):
        validate_thresholds(thresholds)


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

    assert categories == {
        "sparse_bridge_contradictory",
        "sparse_bridge_compatible",
        "provider_conflict_group",
        "url_reuse_group",
        "repeated_quarterly_group",
        "semantic_transitivity",
        "mixed_stage_group",
        "permutation_equivalence",
    }


def test_every_cluster_case_declares_two_valid_partitions():
    for case in default_cluster_cases():
        for partition in (case.expected_partition, case.exact_stage_partition):
            covered = {item for group in partition for item in group}
            assert covered == set(case.item_ids), case.case_id
            assert sum(len(group) for group in partition) == len(case.items)


def test_a_partition_that_places_an_item_twice_is_refused(tmp_path):
    from nlp.eval.clusters import _as_partition

    with pytest.raises(EvalDatasetError, match="more than one group"):
        _as_partition([["a", "b"], ["b"]], where="test")


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
    meta["cases_file"] = "cases.jsonl"
    (tmp_path / "cases.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(json.dumps(case), encoding="utf-8")

    with pytest.raises(EvalDatasetError, match="but the case holds"):
        load_cluster_cases(tmp_path / "cases.meta.json")


def test_m2_reproduces_the_exact_stage_partition_on_every_decidable_case():
    report = evaluate_m2_clusters(target="exact_stage_partition")

    assert report.exact_partition_matches == report.scored_case_count
    assert report.exact_partition_rate == 1.0
    assert report.pairwise.precision == 1.0
    assert report.pairwise.recall == 1.0
    assert report.over_merge_case_ids == ()
    assert report.under_merge_case_ids == ()
    assert report.complete


def test_the_sparse_record_cannot_bridge_contradictory_endpoints():
    """The batch behaviour a pairwise metric is structurally blind to."""

    report = evaluate_m2_clusters()
    outcome = next(o for o in report.outcomes if o.case_id == "C001")

    assert outcome.exact_match
    assert not any({"C001-1", "C001-3"} <= set(group) for group in outcome.predicted)


def test_a_sparse_record_does_bridge_compatible_endpoints():
    report = evaluate_m2_clusters()
    outcome = next(o for o in report.outcomes if o.case_id == "C002")

    assert outcome.predicted == canonical_partition(
        [frozenset({"C002-1", "C002-2", "C002-3"})]
    )


def test_a_provider_conflict_quarantines_the_whole_group():
    report = evaluate_m2_clusters()
    outcome = next(o for o in report.outcomes if o.case_id == "C003")

    assert all(len(group) == 1 for group in outcome.predicted)


def test_m2_under_merges_the_semantic_cases_against_ground_truth():
    """Correct behaviour for M2, and it must be visible as an under-merge."""

    report = evaluate_m2_clusters(target="expected_partition")

    assert set(report.under_merge_case_ids) == {"C006", "C007"}
    assert report.over_merge_case_ids == ()
    assert report.pairwise.precision == 1.0
    assert report.pairwise.recall is not None
    assert report.pairwise.recall < 1.0


def test_over_merging_is_detected_and_named():
    """A clusterer that puts everything together must be caught."""

    case_set = default_cluster_cases()
    report = evaluate_clusters(
        case_set,
        lambda case: canonical_partition([frozenset(case.item_ids)]),
        name="merge-everything",
    )

    assert set(report.over_merge_case_ids) >= {"C001", "C003", "C004", "C005"}
    assert report.exact_partition_matches < report.scored_case_count
    assert report.pairwise.precision is not None
    assert report.pairwise.precision < 1.0


def test_under_merging_is_detected_and_named():
    case_set = default_cluster_cases()
    report = evaluate_clusters(
        case_set,
        lambda case: canonical_partition(
            [frozenset({item_id}) for item_id in case.item_ids]
        ),
        name="merge-nothing",
    )

    assert set(report.under_merge_case_ids) >= {"C002", "C004", "C007", "C008"}
    assert report.over_merge_case_ids == ()
    assert report.pairwise.recall == 0.0


def test_a_transitive_bridge_is_caught_as_an_over_merge():
    """Chaining A-B and B-C into one group where ground truth splits them."""

    case_set = default_cluster_cases()
    report = evaluate_clusters(
        case_set,
        lambda case: canonical_partition([frozenset(case.item_ids)]),
        name="transitive",
    )
    outcome = next(o for o in report.outcomes if o.case_id == "C001")

    assert ("C001-1", "C001-3") in outcome.over_merged_pairs
    assert not outcome.exact_match


def test_missing_and_duplicated_items_are_reported():
    case_set = default_cluster_cases()

    def drops_one(case):
        keep = sorted(case.item_ids)[1:]
        return canonical_partition([frozenset({item_id}) for item_id in keep])

    def duplicates_one(case):
        first = sorted(case.item_ids)[0]
        return canonical_partition(
            [frozenset({item_id}) for item_id in case.item_ids] + [frozenset({first})]
        )

    dropped = evaluate_clusters(case_set, drops_one, name="lossy")
    doubled = evaluate_clusters(case_set, duplicates_one, name="doubling")

    assert dropped.accounting_failures
    assert all(o.missing_items for o in dropped.outcomes)
    assert doubled.accounting_failures
    assert all(o.duplicated_items for o in doubled.outcomes)


def test_permutation_instability_is_detected():
    """Every case is re-run shuffled; an order-sensitive stage is named."""

    seen: dict[str, int] = {}

    def order_sensitive(case):
        seen[case.case_id] = seen.get(case.case_id, 0) + 1
        if seen[case.case_id] == 1:
            return canonical_partition([frozenset(case.item_ids)])
        return canonical_partition([frozenset({item_id}) for item_id in case.item_ids])

    report = evaluate_clusters(
        default_cluster_cases(), order_sensitive, name="order-sensitive"
    )

    assert report.permutation_failures


def test_m2_is_permutation_stable_on_every_case():
    assert evaluate_m2_clusters().permutation_failures == ()


def test_a_raising_case_is_reported_rather_than_ending_the_cluster_run():
    def explode(case):
        if case.case_id == "C003":
            raise RuntimeError("stage exploded")
        return canonical_partition([frozenset({item_id}) for item_id in case.item_ids])

    report = evaluate_clusters(default_cluster_cases(), explode, name="flaky")

    assert report.failed_case_ids == ("C003",)
    assert not report.complete
    assert all(outcome.case_id != "C003" for outcome in report.outcomes)
    assert report.scored_case_count == 7


def test_ambiguous_cluster_cases_are_excluded_from_the_headline_numbers():
    report = evaluate_m2_clusters()

    assert report.ambiguous_case_count == 1
    assert report.scored_case_count == len(default_cluster_cases()) - 1


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
    assert "not AC-3 acceptance" in captured.err


def test_the_cli_scores_clusters(capsys):
    assert eval_dedup.main(["--stage", "m2", "--scope", "clusters"]) == 0
    out = capsys.readouterr().out

    assert "multi_item_cluster_metrics" in out
    assert "exact partition match  8/8" in out


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

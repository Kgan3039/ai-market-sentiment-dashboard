"""``eval_dedup.py``: score a deduplication stage against the labeled sets.

Issue #67 names this script and requires its results to be committed.  It is
deterministic end to end — same dataset, same code, same bytes — so the
committed JSON under ``nlp/eval/data/results/`` is a diffable record rather
than a snapshot of one machine.

There are two measurements and ``--scope`` chooses between them:

    python -m tools.eval_dedup --stage m2                    # isolated pairs
    python -m tools.eval_dedup --stage m2 --scope clusters   # whole batches
    python -m tools.eval_dedup --stage m2 --json

Every output, in both formats, carries the loaded dataset's trust contract
and the summary derived from it, above and below the numbers.  The banner is
a function of the validated metadata, never text a manifest supplied, so it
cannot describe a dataset as something the fields say it is not.

Exit status is 0 when the run completed, 1 when a supplied floor was not
cleared or the run was incomplete, and 2 for a usage or dataset error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

from nlp.eval.clusters import DEFAULT_META_PATH as DEFAULT_CLUSTER_META_PATH
from nlp.eval.clusters import (
    ClusterCaseSet,
    ClusterPredictor,
    evaluate_clusters,
    load_cluster_cases,
    m2_cluster_predictor,
)
from nlp.eval.dataset import (
    DEFAULT_META_PATH,
    EvalDatasetError,
    PairSet,
    load_pair_set,
)
from nlp.eval.dedup import config_for, m2_isolated_pair_predictor
from nlp.eval.semantic import (
    CachingEncoder,
    pipeline_cluster_predictor,
    pipeline_isolated_pair_predictor,
    semantic_config_for,
)
from nlp.eval.metrics import (
    EvaluationReport,
    PairPredictor,
    evaluate_isolated_pairs,
    sweep_thresholds,
)
from nlp.eval.validation import validate_thresholds
from nlp.eval.report import (
    cluster_payload,
    render_clusters,
    render_sweep,
    render_text,
    sweep_payload,
    to_payload,
)
from nlp.eval.trust import TrustContractError
from nlp.eval.validation import GateValueError, validate_optional_unit_interval


def config_for_cluster_set(case_set: ClusterCaseSet) -> Any:
    """Build the dedup configuration the cluster case set declares."""

    from nlp.dedup import DedupConfig

    return DedupConfig(supported_tickers=tuple(case_set.metadata["tickers"]))


_ENCODER: Any = None


def _shared_encoder() -> Any:
    """Build the real embedding service once, on first use.

    Lazily, because scoring M2 must never load a model, and cached, because
    a sweep re-scores the same stories at every threshold and would
    otherwise measure the encoder rather than the predicate.
    """

    global _ENCODER
    if _ENCODER is None:
        from nlp.embeddings import EmbeddingService

        _ENCODER = CachingEncoder(EmbeddingService())
    return _ENCODER


#: Stages this build can score one pair at a time.
StageFactory = Callable[[PairSet], PairPredictor]


def _pipeline_pairs(pair_set: PairSet, threshold: float | None = None) -> PairPredictor:
    return pipeline_isolated_pair_predictor(
        config_for(pair_set),
        semantic_config_for(pair_set, threshold),
        _shared_encoder(),
    )


STAGES: dict[str, StageFactory] = {
    "m2": lambda pair_set: m2_isolated_pair_predictor(config_for(pair_set)),
    "m2+m3": _pipeline_pairs,
}

#: Stages this build can score on whole batches.
ClusterStageFactory = Callable[[ClusterCaseSet], ClusterPredictor]


def _pipeline_clusters(case_set: ClusterCaseSet) -> ClusterPredictor:
    return pipeline_cluster_predictor(
        config_for_cluster_set(case_set),
        semantic_config_for(case_set),
        _shared_encoder(),
    )


CLUSTER_STAGES: dict[str, ClusterStageFactory] = {
    "m2": lambda case_set: m2_cluster_predictor(config_for_cluster_set(case_set)),
    "m2+m3": _pipeline_clusters,
}

#: Stages whose merge predicate has a tunable threshold, mapped to the
#: factory a sweep needs.  M2 has none by construction: no similarity value
#: participates in any of its accept decisions.  M3's cosine floor is one,
#: and it is the only tunable value in the pipeline.
SWEEPABLE: dict[str, Callable[[PairSet, float], PairPredictor]] = {
    "m2+m3": _pipeline_pairs,
}

DEFAULT_SWEEP = (0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_dedup",
        description=(
            "Measure a deduplication stage against the labeled sets. Every "
            "report states the loaded dataset's trust contract and a "
            "summary derived from it, above and below the numbers; read "
            "that before quoting any figure."
        ),
    )
    parser.add_argument(
        "--stage",
        default="m2",
        choices=sorted(STAGES),
        help="which stage to score (default: m2)",
    )
    parser.add_argument(
        "--scope",
        default="pairs",
        choices=("pairs", "clusters"),
        help=(
            "pairs = isolated_pair_metrics, two records per invocation; "
            "clusters = multi_item_cluster_metrics, a whole group per "
            "invocation (default: pairs)"
        ),
    )
    parser.add_argument(
        "--target",
        default=None,
        choices=("exact_stage_partition", "expected_partition"),
        help=(
            "cluster scope only: which committed expectation to score "
            "against (default: exact_stage_partition for m2, "
            "expected_partition for m2+m3, since closing the gap to ground "
            "truth is what the semantic stage is for)"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="path to the manifest (defaults to the committed set for --scope)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of the text report"
    )
    parser.add_argument(
        "--write", type=Path, help="also write the JSON report to this path"
    )
    parser.add_argument(
        "--sweep",
        nargs="*",
        type=float,
        metavar="THRESHOLD",
        help="sweep the stage's merge threshold; no values means the default grid",
    )
    parser.add_argument(
        "--threshold", type=float, help="run the stage at one specific threshold"
    )
    parser.add_argument(
        "--precision-floor",
        type=float,
        help="fail the run when precision is below this value (AC-3 uses 0.85)",
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        help="fail the run when recall is below this value (AC-3 uses 0.75)",
    )
    parser.add_argument(
        "--composition",
        action="store_true",
        help="print the dataset composition instead of scoring",
    )
    return parser


def _predictor_for(
    stage: str, pair_set: PairSet, threshold: float | None
) -> PairPredictor:
    if threshold is None:
        return STAGES[stage](pair_set)
    if stage not in SWEEPABLE:
        raise SystemExit(
            f"stage {stage!r} has no tunable threshold; "
            "its accept decisions use no similarity value"
        )
    return SWEEPABLE[stage](pair_set, threshold)


def _validated_floors(args: argparse.Namespace) -> tuple[float | None, float | None]:
    """Validate the gate floors before any comparison is made against them.

    ``argparse``'s ``type=float`` accepts ``nan``, ``inf`` and ``-inf``.
    NaN then loses every ``<`` and ``>=`` silently, so a gate checked
    against it never fails for the reason the operator thinks. The floors
    go through the shared validator first, and a bad one ends the run.
    """

    return (
        validate_optional_unit_interval(args.precision_floor, "--precision-floor"),
        validate_optional_unit_interval(args.recall_floor, "--recall-floor"),
    )


def _gate(report: EvaluationReport, args: argparse.Namespace) -> int:
    failures: list[str] = []
    if args.precision_floor is not None:
        precision = report.isolated_pair_metrics.precision
        if precision is None or precision < args.precision_floor:
            failures.append(
                f"precision {precision if precision is not None else 'n/a'} "
                f"< floor {args.precision_floor}"
            )
    if args.recall_floor is not None:
        recall = report.isolated_pair_metrics.recall
        if recall is None or recall < args.recall_floor:
            failures.append(
                f"recall {recall if recall is not None else 'n/a'} "
                f"< floor {args.recall_floor}"
            )
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    if failures:
        if not report.trust.gate_eligible:
            print(
                f"NOTE: {report.trust.summary.headline} A floor checked "
                "against this dataset is a development regression guard, "
                "not acceptance.",
                file=sys.stderr,
            )
    return 1 if failures else 0


#: A refusal whose reason names the similarity floor rather than a guard.
_THRESHOLD_REFUSAL = "below_threshold"

#: Decimal places every similarity score is serialized at.
#:
#: Cosine values come out of a float32 model and the last bits are not
#: reproducible across BLAS builds, thread counts or hardware.  Rounding to
#: six places makes a committed artifact diffable and re-derivable on the
#: same model build, and it is far finer than any decision this stage
#: makes: the tightest margin in the sweep is 0.0247, four orders of
#: magnitude above the rounding step, so no decision boundary is obscured.
#: Across *different* model executions the artifacts are equal to within
#: this precision - not byte-identical, and nothing here claims they are.
SCORE_PRECISION = 6


def _round_scores(payload: dict[str, Any]) -> dict[str, Any]:
    """Round every serialized similarity to :data:`SCORE_PRECISION`."""

    scores = payload.get("scores")
    if isinstance(scores, dict):
        payload["scores"] = {
            key: round(value, SCORE_PRECISION) if isinstance(value, float) else value
            for key, value in scores.items()
        }
    payload["score_precision"] = SCORE_PRECISION
    return payload


def _confusion_detail(report: EvaluationReport) -> dict[str, Any]:
    """Split a report's confusion into the fields a reader needs.

    In particular it separates the two reasons a true positive can be
    missed, which the aggregate recall hides: the pair scored under the
    similarity floor, or a guard refused it outright.  A guard-driven false
    negative is a guard defect and a threshold-driven one is the cost of
    the margin, and conflating them makes the second look like the first.
    """

    confusion = report.isolated_pair_metrics.confusion
    guard_driven: list[str] = []
    threshold_driven: list[str] = []
    other: list[str] = []
    for pair_id in confusion.false_negatives:
        detail = report.details.get(pair_id, "")
        if _THRESHOLD_REFUSAL in detail:
            threshold_driven.append(pair_id)
        elif "not merged:" in detail:
            guard_driven.append(pair_id)
        else:
            other.append(pair_id)
    return {
        "counts": {
            "true_positive": confusion.tp,
            "false_positive": confusion.fp,
            "true_negative": confusion.tn,
            "false_negative": confusion.fn,
        },
        "true_positive_ids": list(confusion.true_positives),
        "false_positive_ids": list(confusion.false_positives),
        "true_negative_ids": list(confusion.true_negatives),
        "false_negative_ids": list(confusion.false_negatives),
        "guard_rejected_positive_ids": sorted(guard_driven),
        "threshold_rejected_positive_ids": sorted(threshold_driven),
        "unclassified_negative_ids": sorted(other),
        "worst_false_positive_score": (
            max(
                (
                    report.scores[pair_id]
                    for pair_id in confusion.false_positives
                    if pair_id in report.scores
                ),
                default=None,
            )
        ),
        "quarantine_skipped_pair_ids": sorted(
            pair_id
            for pair_id, detail in report.details.items()
            if "provider quarantine" in detail
        ),
        "evaluated_case_count": report.evaluated_case_count,
        "failed_case_count": report.failed_case_count,
        "failed_case_ids": list(report.failed_case_ids),
        "complete": report.complete,
    }


def _selection_block(stage: str, points: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Describe which sweep point the build ships, and why.

    Computed from the sweep rows rather than typed in, so the rationale a
    reader sees cannot drift from the numbers beside it.  ``None`` for a
    stage with no configured threshold.
    """

    if stage not in SWEEPABLE:
        return None
    from nlp.semdedup.config import DEFAULT_SIMILARITY_THRESHOLD

    chosen = next(
        (
            point
            for point in points
            if abs(point["threshold"] - DEFAULT_SIMILARITY_THRESHOLD) < 1e-9
        ),
        None,
    )
    if chosen is None:
        return None
    dirty = [point for point in points if point["counts"]["false_positive"]]
    clean = [point for point in points if not point["counts"]["false_positive"]]
    lowest_clean = min(point["threshold"] for point in clean) if clean else None
    worst_fp_score = max(
        (
            score
            for point in points
            for score in (point.get("worst_false_positive_score"),)
            if score is not None
        ),
        default=None,
    )
    return {
        "selected_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "status": "provisional",
        "basis": "precision_first",
        "precision_at_selected": chosen["precision"],
        "recall_at_selected": chosen["recall"],
        "f1_at_selected": chosen["f1"],
        "false_positives_at_selected": chosen["false_positives"],
        "guard_rejected_positive_ids_at_selected": chosen.get(
            "guard_rejected_positive_ids", []
        ),
        "threshold_rejected_positive_ids_at_selected": chosen.get(
            "threshold_rejected_positive_ids", []
        ),
        "highest_threshold_still_producing_a_false_merge": (
            max(point["threshold"] for point in dirty) if dirty else None
        ),
        "highest_surviving_false_positive_score": worst_fp_score,
        "margin_over_worst_false_positive": (
            round(DEFAULT_SIMILARITY_THRESHOLD - worst_fp_score, 6)
            if worst_fp_score is not None
            else None
        ),
        "lowest_threshold_with_zero_false_merges": lowest_clean,
        "selected_is_lowest_clean_threshold": (
            lowest_clean is not None
            and abs(lowest_clean - DEFAULT_SIMILARITY_THRESHOLD) < 1e-9
        ),
        "max_f1_threshold": max(points, key=lambda point: point["f1"] or 0.0)[
            "threshold"
        ],
        "rationale": (
            "Precision-first. The lowest tested threshold with zero false "
            "merges is reported as lowest_threshold_with_zero_false_merges; "
            "the selected value sits above it by a deliberate margin over the "
            "highest-scoring surviving false positive, rather than on the "
            "knife edge. The F1 maximum is not used: it buys F1 with false "
            "merges. Note that not every false negative is threshold-driven - "
            "guard_rejected_positive_ids_at_selected lists the ones a guard "
            "refused."
        ),
        "caveat": (
            "Provisional and development-only. Selected on a synthetic, "
            "single-author, unadjudicated dataset that is not gate eligible; "
            "these numbers cannot clear K3/G4 or final AC-3."
        ),
    }


def _dump(payload: object, path: Path | None) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def _incomplete(report: Any) -> int:
    if report.complete:
        return 0
    print(
        f"INCOMPLETE: {report.failed_case_count} case(s) raised and were "
        f"excluded from every denominator: {', '.join(report.failed_case_ids)}",
        file=sys.stderr,
    )
    return 1


def _run_clusters(args: argparse.Namespace) -> int:
    try:
        case_set = load_cluster_cases(args.dataset or DEFAULT_CLUSTER_META_PATH)
    except (EvalDatasetError, TrustContractError) as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2
    if args.composition:
        print(
            json.dumps(
                {
                    "trust_contract": case_set.trust.as_dict(),
                    "dataset_id": case_set.dataset_id,
                    "case_count": len(case_set),
                    "composition": case_set.composition(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    target = args.target or (
        "expected_partition" if args.stage == "m2+m3" else "exact_stage_partition"
    )
    report = evaluate_clusters(
        case_set,
        CLUSTER_STAGES[args.stage](case_set),
        name=args.stage,
        target=target,
    )
    text = _dump(cluster_payload(report), args.write)
    if args.json:
        print(text, end="")
    else:
        print(
            render_clusters(
                report,
                rationales={case.case_id: case.rationale for case in case_set},
            ),
            end="",
        )
    return _incomplete(report)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _validated_floors(args)
        if args.threshold is not None:
            validate_optional_unit_interval(args.threshold, "--threshold")
        if args.sweep:
            validate_thresholds(args.sweep)
    except GateValueError as exc:
        print(f"invalid gate value: {exc}", file=sys.stderr)
        return 2
    if args.scope == "clusters":
        return _run_clusters(args)

    try:
        pair_set = load_pair_set(args.dataset or DEFAULT_META_PATH)
    except (EvalDatasetError, TrustContractError) as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2

    if args.composition:
        print(
            json.dumps(
                {
                    "trust_contract": pair_set.trust.as_dict(),
                    "dataset_id": pair_set.dataset_id,
                    "pair_count": len(pair_set),
                    "scored_pair_count": len(pair_set.scored),
                    "composition": pair_set.composition(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.sweep is not None:
        if args.stage not in SWEEPABLE:
            print(
                f"stage {args.stage!r} has no tunable threshold to sweep",
                file=sys.stderr,
            )
            return 2
        thresholds = tuple(args.sweep) or DEFAULT_SWEEP
        points = sweep_thresholds(
            pair_set,
            lambda threshold: SWEEPABLE[args.stage](pair_set, threshold),
            thresholds,
            name=args.stage,
        )
        payload = sweep_payload(points)
        payload["stage"] = args.stage
        payload["score_precision"] = SCORE_PRECISION
        for point, entry in zip(points, payload["points"]):
            detail = _confusion_detail(point.report)
            if detail["worst_false_positive_score"] is not None:
                detail["worst_false_positive_score"] = round(
                    detail["worst_false_positive_score"], SCORE_PRECISION
                )
            entry.update(detail)
        selection = _selection_block(args.stage, payload["points"])
        if selection is not None:
            payload["selection"] = selection
        text = _dump(payload, args.write)
        print(text if args.json else render_sweep(points), end="")
        return 0

    # Record the threshold the stage actually ran at, not the one the user
    # happened to type: a committed report whose threshold reads "null"
    # cannot be checked against the sweep it is supposed to come from.
    effective = args.threshold
    if effective is None and args.stage in SWEEPABLE:
        effective = semantic_config_for(pair_set).similarity_threshold
    report = evaluate_isolated_pairs(
        pair_set,
        _predictor_for(args.stage, pair_set, args.threshold),
        name=args.stage,
        threshold=effective,
    )
    text = _dump(_round_scores(to_payload(report)), args.write)
    if args.json:
        print(text, end="")
    else:
        print(
            render_text(
                report,
                rationales={pair.pair_id: pair.rationale for pair in pair_set},
            ),
            end="",
        )
    return _gate(report, args) or _incomplete(report)


if __name__ == "__main__":
    raise SystemExit(main())

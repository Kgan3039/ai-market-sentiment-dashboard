"""Deterministic rendering of an evaluation report.

Every renderer is a pure function of the report: no clock, no host name, no
run identifier, no dictionary iteration order.  That is what lets the JSON
form be committed to the repository and diffed — a re-run that changes a
byte means a stage changed, not that the file was regenerated.

Every renderer also emits the dataset's trust contract **before** any
number, in both formats.  Somebody will read ``m2_baseline.json``, or a CLI
transcript pasted into a ticket, without the README anywhere nearby; the
provenance has to be in the artefact itself or it is not stated at all.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .clusters import ClusterEvaluationReport
from .metrics import CategoryBreakdown, EvaluationReport, Metrics, ThresholdPoint


#: Payload shapes, versioned so a consumer can tell which contract it has.
ISOLATED_PAIR_PAYLOAD_VERSION = "phase0.eval_report.isolated_pairs.v2"
CLUSTER_PAYLOAD_VERSION = "phase0.eval_report.clusters.v2"
SWEEP_PAYLOAD_VERSION = "phase0.eval_report.sweep.v2"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _metrics_payload(metrics: Metrics) -> dict[str, Any]:
    confusion = metrics.confusion
    return {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "accuracy": metrics.accuracy,
        "counts": {
            "true_positive": confusion.tp,
            "false_positive": confusion.fp,
            "true_negative": confusion.tn,
            "false_negative": confusion.fn,
        },
        "pair_ids": {
            "true_positive": list(confusion.true_positives),
            "false_positive": list(confusion.false_positives),
            "true_negative": list(confusion.true_negatives),
            "false_negative": list(confusion.false_negatives),
        },
    }


def _breakdown_payload(
    breakdown: Sequence[CategoryBreakdown],
) -> dict[str, dict[str, Any]]:
    return {entry.key: _metrics_payload(entry.metrics) for entry in breakdown}


def _completeness(report: Any) -> dict[str, Any]:
    return {
        "evaluated_case_count": report.evaluated_case_count,
        "failed_case_count": report.failed_case_count,
        "failed_case_ids": list(report.failed_case_ids),
        "complete": report.complete,
        "failures": [
            {
                "case_id": failure.case_id,
                "error_type": failure.error_type,
                "message": failure.message,
            }
            for failure in report.failures
        ],
    }


def to_payload(report: EvaluationReport) -> dict[str, Any]:
    """Return the JSON-serializable form of an isolated-pair report."""

    return {
        "trust_contract": report.trust.as_dict(),
        "trust_summary": report.trust.summary.as_dict(),
        "dataset_id": report.dataset_id,
        "schema_version": ISOLATED_PAIR_PAYLOAD_VERSION,
        "predictor": report.predictor,
        "scope": report.scope,
        "limitation": report.limitation,
        "threshold": report.threshold,
        "isolated_pair_metrics": _metrics_payload(report.isolated_pair_metrics),
        "candidate_recall": report.candidate_recall,
        "merge_recall": report.merge_recall,
        "by_expected_stage": _breakdown_payload(report.by_expected_stage),
        "by_category": _breakdown_payload(report.by_category),
        "by_ticker": _breakdown_payload(report.by_ticker),
        "ambiguous": {
            "count": report.ambiguous_count,
            "merged": list(report.ambiguous_merged),
        },
        "completeness": _completeness(report),
        "details": dict(sorted(report.details.items())),
        "scores": dict(sorted(report.scores.items())),
    }


def cluster_payload(report: ClusterEvaluationReport) -> dict[str, Any]:
    """Return the JSON-serializable form of a multi-item cluster report."""

    return {
        "trust_contract": report.trust.as_dict(),
        "trust_summary": report.trust.summary.as_dict(),
        "dataset_id": report.dataset_id,
        "schema_version": CLUSTER_PAYLOAD_VERSION,
        "predictor": report.predictor,
        "scope": report.scope,
        "limitation": report.limitation,
        "target": report.target,
        "multi_item_cluster_metrics": {
            "exact_partition_matches": report.exact_partition_matches,
            "scored_case_count": report.scored_case_count,
            "exact_partition_rate": report.exact_partition_rate,
            "pairwise_co_clustering": _metrics_payload(report.pairwise),
            "over_merge_case_ids": list(report.over_merge_case_ids),
            "under_merge_case_ids": list(report.under_merge_case_ids),
            "permutation_failures": list(report.permutation_failures),
            "ambiguous_case_count": report.ambiguous_case_count,
        },
        "accounting": {
            "failed_case_ids": list(report.accounting_failure_ids),
            "missing_item_ids": list(report.missing_item_ids),
            "duplicated_item_ids": list(report.duplicated_item_ids),
            "unexpected_item_ids": list(report.unexpected_item_ids),
            "violations": [
                {
                    "case_id": entry.case_id,
                    "missing_item_ids": list(entry.missing_item_ids),
                    "duplicated_item_ids": list(entry.duplicated_item_ids),
                    "unexpected_item_ids": list(entry.unexpected_item_ids),
                    "empty_cluster_count": entry.empty_cluster_count,
                    "message": entry.message,
                }
                for entry in report.accounting_violations
            ],
        },
        "cases": [
            {
                "case_id": outcome.case_id,
                "category": outcome.category,
                "status": outcome.status,
                "resolvable_by": outcome.resolvable_by,
                "expected_partition": [sorted(group) for group in outcome.expected],
                "predicted_partition": [sorted(group) for group in outcome.predicted],
                "exact_match": outcome.exact_match,
                "over_merged_pairs": [list(pair) for pair in outcome.over_merged_pairs],
                "under_merged_pairs": [
                    list(pair) for pair in outcome.under_merged_pairs
                ],
                "indeterminate_item_ids": list(outcome.indeterminate_item_ids),
                "permutation_count": outcome.permutation_count,
                "permutation_stable": outcome.permutation_stable,
                "unstable_permutation_count": outcome.unstable_permutation_count,
            }
            for outcome in report.outcomes
        ],
        "completeness": _completeness(report),
    }


def sweep_payload(points: Sequence[ThresholdPoint]) -> dict[str, Any]:
    """Return the JSON-serializable form of a threshold sweep.

    A full document, not a bare list of rows.  The rows are the part
    somebody quotes, so the trust contract, the dataset identity, the scope
    and the completeness of each point have to be in the same object; a
    caller that wrapped the list itself could forget, and one did.
    """

    if not points:
        raise ValueError("a sweep report needs at least one point")
    first = points[0].report
    return {
        "trust_contract": first.trust.as_dict(),
        "trust_summary": first.trust.summary.as_dict(),
        "dataset_id": first.dataset_id,
        "schema_version": SWEEP_PAYLOAD_VERSION,
        "scope": first.scope,
        "limitation": first.limitation,
        "complete": all(point.report.complete for point in points),
        "evaluated_case_count": first.evaluated_case_count,
        "failed_case_count": sum(point.report.failed_case_count for point in points),
        "failed_case_ids": sorted(
            {case_id for point in points for case_id in point.report.failed_case_ids}
        ),
        "points": [
            {
                "threshold": point.threshold,
                "scope": point.report.scope,
                "precision": point.precision,
                "recall": point.recall,
                "f1": point.f1,
                "candidate_recall": point.report.candidate_recall,
                "counts": {
                    "true_positive": point.report.isolated_pair_metrics.confusion.tp,
                    "false_positive": point.report.isolated_pair_metrics.confusion.fp,
                    "false_negative": point.report.isolated_pair_metrics.confusion.fn,
                },
                "false_positives": list(
                    point.report.isolated_pair_metrics.confusion.false_positives
                ),
                "complete": point.report.complete,
                "evaluated_case_count": point.report.evaluated_case_count,
                "failed_case_count": point.report.failed_case_count,
                "failed_case_ids": list(point.report.failed_case_ids),
            }
            for point in points
        ],
    }


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    ]


def _wrap(text: str, width: int = 74, indent: str = "  ") -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(indent + current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(indent + current)
    return lines


def render_banner(report: Any) -> str:
    """The provenance block printed above every metric, in every renderer.

    The heading is :meth:`~nlp.eval.trust.TrustContract.banner`, derived
    from the validated fields. No renderer composes wording of its own, so
    a production dataset can never be described as synthetic here.
    """

    return report.trust.banner() + "\n"


def _completeness_lines(report: Any) -> list[str]:
    lines = [
        "",
        "completeness",
        f"  evaluated_case_count  {report.evaluated_case_count}",
        f"  failed_case_count     {report.failed_case_count}",
        f"  complete              {str(report.complete).lower()}",
    ]
    if report.failures:
        lines.append(f"  failed_case_ids       {', '.join(report.failed_case_ids)}")
        for failure in report.failures:
            lines.append(
                f"    {failure.case_id}  {failure.error_type}: {failure.message}"
            )
        lines.append("  NOTE: failed cases are excluded from every denominator above.")
    return lines


def render_text(
    report: EvaluationReport,
    *,
    rationales: Mapping[str, str] | None = None,
) -> str:
    """Render a human-readable isolated-pair report.

    ``rationales`` maps pair id to its labelling rationale; when supplied,
    every confusion entry is printed with the reason a human gave for the
    label, which is what makes a false positive reviewable without opening
    the dataset.
    """

    lines: list[str] = [render_banner(report).rstrip(), ""]
    lines += [
        f"dataset:   {report.dataset_id}",
        f"predictor: {report.predictor}",
        f"scope:     {report.scope}",
    ]
    if report.threshold is not None:
        lines.append(f"threshold: {report.threshold:g}")
    lines += ["", "limitation"] + _wrap(report.limitation)
    metrics = report.isolated_pair_metrics
    confusion = metrics.confusion
    lines += [
        "",
        "isolated_pair_metrics (ambiguous pairs excluded)",
        f"  precision        {_number(metrics.precision)}",
        f"  recall           {_number(metrics.recall)}",
        f"  f1               {_number(metrics.f1)}",
        f"  accuracy         {_number(metrics.accuracy)}",
        f"  candidate recall {_number(report.candidate_recall)}",
        f"  tp/fp/tn/fn      {confusion.tp}/{confusion.fp}/"
        f"{confusion.tn}/{confusion.fn}",
        "",
        "by expected stage",
    ]
    lines += [
        "  " + line
        for line in _table(
            [["stage", "P", "R", "tp", "fp", "fn"]]
            + [
                [
                    entry.key,
                    _number(entry.metrics.precision),
                    _number(entry.metrics.recall),
                    str(entry.metrics.confusion.tp),
                    str(entry.metrics.confusion.fp),
                    str(entry.metrics.confusion.fn),
                ]
                for entry in report.by_expected_stage
            ]
        )
    ]
    lines += ["", "by category"]
    lines += [
        "  " + line
        for line in _table(
            [["category", "P", "R", "tp", "fp", "tn", "fn"]]
            + [
                [
                    entry.key,
                    _number(entry.metrics.precision),
                    _number(entry.metrics.recall),
                    str(entry.metrics.confusion.tp),
                    str(entry.metrics.confusion.fp),
                    str(entry.metrics.confusion.tn),
                    str(entry.metrics.confusion.fn),
                ]
                for entry in report.by_category
            ]
        )
    ]
    for heading, pair_ids in (
        ("false positives (merged, must not have been)", confusion.false_positives),
        ("false negatives (not merged, should have been)", confusion.false_negatives),
    ):
        lines += ["", f"{heading}: {len(pair_ids)}"]
        for pair_id in pair_ids:
            detail = report.details.get(pair_id, "")
            lines.append(f"  {pair_id}  {detail}".rstrip())
            if rationales and pair_id in rationales:
                lines.append(f"      label rationale: {rationales[pair_id]}")
    lines += [
        "",
        f"ambiguous pairs: {report.ambiguous_count} "
        f"(excluded); merged by this stage: "
        f"{', '.join(report.ambiguous_merged) or 'none'}",
    ]
    lines += _completeness_lines(report)
    lines += ["", render_banner(report).rstrip()]
    return "\n".join(lines) + "\n"


def render_clusters(
    report: ClusterEvaluationReport,
    *,
    rationales: Mapping[str, str] | None = None,
) -> str:
    """Render a human-readable multi-item cluster report."""

    lines: list[str] = [render_banner(report).rstrip(), ""]
    lines += [
        f"dataset:   {report.dataset_id}",
        f"predictor: {report.predictor}",
        f"scope:     {report.scope}",
        f"target:    {report.target}",
        "",
        "limitation",
    ]
    lines += _wrap(report.limitation)
    lines += [
        "",
        "multi_item_cluster_metrics (ambiguous cases excluded)",
        f"  exact partition match  {report.exact_partition_matches}"
        f"/{report.scored_case_count}"
        f"  ({_number(report.exact_partition_rate)})",
        f"  co-clustering P/R/F1   {_number(report.pairwise.precision)}"
        f" / {_number(report.pairwise.recall)}"
        f" / {_number(report.pairwise.f1)}",
        f"  over-merged cases      {', '.join(report.over_merge_case_ids) or 'none'}",
        f"  under-merged cases     {', '.join(report.under_merge_case_ids) or 'none'}",
        f"  permutation failures   "
        f"{', '.join(report.permutation_failures) or 'none'}",
        f"  accounting failures    "
        f"{', '.join(report.accounting_failure_ids) or 'none'}",
        f"  missing item ids       {', '.join(report.missing_item_ids) or 'none'}",
        f"  duplicated item ids    "
        f"{', '.join(report.duplicated_item_ids) or 'none'}",
        f"  unexpected item ids    "
        f"{', '.join(report.unexpected_item_ids) or 'none'}",
        f"  ambiguous cases        {report.ambiguous_case_count} (excluded)",
        "",
        "cases",
    ]
    for outcome in report.outcomes:
        mark = "ok " if outcome.exact_match else "DIFF"
        lines.append(
            f"  [{mark}] {outcome.case_id}  {outcome.category}"
            f"  ({outcome.status}, resolvable_by {outcome.resolvable_by})"
        )
        lines.append(
            "         expected  "
            + " | ".join("+".join(sorted(group)) for group in outcome.expected)
        )
        lines.append(
            "         predicted "
            + " | ".join("+".join(sorted(group)) for group in outcome.predicted)
        )
        if outcome.over_merged_pairs:
            lines.append(
                "         over-merged  "
                + ", ".join("+".join(pair) for pair in outcome.over_merged_pairs)
            )
        if outcome.under_merged_pairs:
            lines.append(
                "         under-merged "
                + ", ".join("+".join(pair) for pair in outcome.under_merged_pairs)
            )
        if outcome.indeterminate_item_ids:
            lines.append(
                "         indeterminate (unscored) "
                + ", ".join(outcome.indeterminate_item_ids)
            )
        lines.append(
            f"         permutations {outcome.permutation_count} checked, "
            f"{outcome.unstable_permutation_count} disagreed"
        )
        if not outcome.permutation_stable:
            lines.append("         PERMUTATION UNSTABLE")
        if rationales and outcome.case_id in rationales:
            lines += _wrap(rationales[outcome.case_id], indent="         ")
    lines += _completeness_lines(report)
    lines += ["", render_banner(report).rstrip()]
    return "\n".join(lines) + "\n"


def render_sweep(points: Sequence[ThresholdPoint]) -> str:
    """Render a threshold sweep as a fixed-width table."""

    rows = [["threshold", "P", "R", "F1", "cand.R", "tp", "fp", "fn"]]
    rows += [
        [
            f"{point.threshold:g}",
            _number(point.precision),
            _number(point.recall),
            _number(point.f1),
            _number(point.report.candidate_recall),
            str(point.report.isolated_pair_metrics.confusion.tp),
            str(point.report.isolated_pair_metrics.confusion.fp),
            str(point.report.isolated_pair_metrics.confusion.fn),
        ]
        for point in points
    ]
    header = points[0].report.trust.banner() if points else ""
    body = "\n".join(_table(rows))
    scope = "scope: isolated_pairs"
    return f"{header}\n\n{scope}\n{body}\n\n{header}\n" if header else body + "\n"

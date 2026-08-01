"""Deterministic rendering of an evaluation report.

Both renderers are pure functions of the report: no clock, no host name, no
run identifier, no dictionary iteration order.  That is what lets the JSON
form be committed to the repository and diffed — a re-run that changes a
byte means a stage changed, not that the file was regenerated.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .metrics import CategoryBreakdown, EvaluationReport, Metrics, ThresholdPoint


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


def to_payload(report: EvaluationReport) -> dict[str, Any]:
    """Return the JSON-serializable form of a report."""

    return {
        "dataset_id": report.dataset_id,
        "predictor": report.predictor,
        "threshold": report.threshold,
        "overall": _metrics_payload(report.overall),
        "candidate_recall": report.candidate_recall,
        "merge_recall": report.merge_recall,
        "by_expected_stage": _breakdown_payload(report.by_expected_stage),
        "by_category": _breakdown_payload(report.by_category),
        "by_ticker": _breakdown_payload(report.by_ticker),
        "ambiguous": {
            "count": report.ambiguous_count,
            "merged": list(report.ambiguous_merged),
        },
        "details": dict(sorted(report.details.items())),
        "scores": dict(sorted(report.scores.items())),
    }


def sweep_payload(points: Sequence[ThresholdPoint]) -> list[dict[str, Any]]:
    """Return the JSON-serializable form of a threshold sweep."""

    return [
        {
            "threshold": point.threshold,
            "precision": point.precision,
            "recall": point.recall,
            "f1": point.f1,
            "candidate_recall": point.report.candidate_recall,
            "counts": {
                "true_positive": point.report.overall.confusion.tp,
                "false_positive": point.report.overall.confusion.fp,
                "false_negative": point.report.overall.confusion.fn,
            },
            "false_positives": list(point.report.overall.confusion.false_positives),
        }
        for point in points
    ]


def _table(rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    ]


def render_text(
    report: EvaluationReport,
    *,
    rationales: Mapping[str, str] | None = None,
) -> str:
    """Render a human-readable report.

    ``rationales`` maps pair id to its labelling rationale; when supplied,
    every confusion entry is printed with the reason a human gave for the
    label, which is what makes a false positive reviewable without opening
    the dataset.
    """

    lines: list[str] = [
        f"dataset:   {report.dataset_id}",
        f"predictor: {report.predictor}",
    ]
    if report.threshold is not None:
        lines.append(f"threshold: {report.threshold:g}")
    overall = report.overall
    confusion = overall.confusion
    lines += [
        "",
        "overall (ambiguous pairs excluded)",
        f"  precision        {_number(overall.precision)}",
        f"  recall           {_number(overall.recall)}",
        f"  f1               {_number(overall.f1)}",
        f"  accuracy         {_number(overall.accuracy)}",
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
            str(point.report.overall.confusion.tp),
            str(point.report.overall.confusion.fp),
            str(point.report.overall.confusion.fn),
        ]
        for point in points
    ]
    return "\n".join(_table(rows)) + "\n"

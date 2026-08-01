"""``eval_dedup.py``: score a deduplication stage against the labeled set.

Issue #67 names this script and requires its results to be committed.  It is
deterministic end to end — same dataset, same code, same bytes — so the
committed JSON under ``nlp/eval/data/results/`` is a diffable record rather
than a snapshot of one machine.

    python -m tools.eval_dedup --stage m2
    python -m tools.eval_dedup --stage m2 --json
    python -m tools.eval_dedup --stage m2 --write nlp/eval/data/results/m2_baseline.json

Exit status is 0 when the run completed, 1 when ``--precision-floor`` or
``--recall-floor`` was supplied and the stage did not clear it, and 2 for a
usage or dataset error.  Passing no floor asks a question; passing a floor
asserts a gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable, Sequence

from nlp.eval.dataset import DEFAULT_META_PATH, EvalDatasetError, PairSet, load_pair_set
from nlp.eval.dedup import config_for, m2_predictor
from nlp.eval.metrics import EvaluationReport, PairPredictor, evaluate, sweep_thresholds
from nlp.eval.report import render_sweep, render_text, sweep_payload, to_payload

#: Stages this build can score.  M3 registers itself here when it lands; M4
#: deliberately does not reference an unmerged module.
StageFactory = Callable[[PairSet], PairPredictor]

STAGES: dict[str, StageFactory] = {
    "m2": lambda pair_set: m2_predictor(config_for(pair_set)),
}

#: Stages whose merge predicate has a tunable threshold, mapped to the
#: factory a sweep needs.  M2 has none by construction: no similarity value
#: participates in any of its accept decisions.
SWEEPABLE: dict[str, Callable[[PairSet, float], PairPredictor]] = {}

DEFAULT_SWEEP = (0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_dedup",
        description="Measure a deduplication stage against the labeled pair set.",
    )
    parser.add_argument(
        "--stage",
        default="m2",
        choices=sorted(STAGES),
        help="which stage to score (default: m2)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_META_PATH,
        help="path to the pair-set manifest",
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


def _gate(report: EvaluationReport, args: argparse.Namespace) -> int:
    failures: list[str] = []
    if args.precision_floor is not None:
        precision = report.overall.precision
        if precision is None or precision < args.precision_floor:
            failures.append(
                f"precision {precision if precision is not None else 'n/a'} "
                f"< floor {args.precision_floor}"
            )
    if args.recall_floor is not None:
        recall = report.overall.recall
        if recall is None or recall < args.recall_floor:
            failures.append(
                f"recall {recall if recall is not None else 'n/a'} "
                f"< floor {args.recall_floor}"
            )
    for failure in failures:
        print(f"GATE FAILED: {failure}", file=sys.stderr)
    return 1 if failures else 0


def _dump(payload: object, path: Path | None) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        pair_set = load_pair_set(args.dataset)
    except EvalDatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 2

    if args.composition:
        print(
            json.dumps(
                {
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
        payload = {
            "dataset_id": pair_set.dataset_id,
            "stage": args.stage,
            "sweep": sweep_payload(points),
        }
        text = _dump(payload, args.write)
        print(text if args.json else render_sweep(points), end="")
        return 0

    report = evaluate(
        pair_set,
        _predictor_for(args.stage, pair_set, args.threshold),
        name=args.stage,
        threshold=args.threshold,
    )
    payload = to_payload(report)
    text = _dump(payload, args.write)
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
    return _gate(report, args)


if __name__ == "__main__":
    raise SystemExit(main())

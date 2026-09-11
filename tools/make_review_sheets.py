"""``make_review_sheets.py``: sample and score Phase 0 human-review sheets.

Issue #74 (A4) needs two review-sheet kinds for Section 8's go/no-go gates:

    (a) story -> theme assignments, sampled for gate G1 (>=75%)
    (b) summary sentences with resolved citations, sampled for gate G2 (>=95%)

**Assignment sampling is offline by default**, same posture as
``tools.eval_themes``: clustering replays the committed
``nlp/themes/data/story_vectors.json`` fixture rather than loading a model.

**Summary sampling calls the real Gemini API** through
``ai.summarization.GeminiClient`` -- there is no offline substitute for what
a live model says, so this fails fast if ``GEMINI_API_KEY`` is unset rather
than fabricating output.

**No real soak-window data exists yet** (#57/I1 is still open), so
everything sampled here comes from M5's own eval fixture: 3 ticker-days, 30
stories. Every scorecard this tool prints therefore carries
``dataset_kind=synthetic_development`` / ``gate_eligible=false`` -- see
``nlp/eval/review.py``'s module docstring for why, and why that is enforced
structurally rather than by convention.

    python -m tools.make_review_sheets sample-assignments --out review/assignments.csv
    python -m tools.make_review_sheets sample-summaries --out review/summaries.csv
    python -m tools.make_review_sheets score-assignments review/assignments_r1.csv review/assignments_r2.csv --json
    python -m tools.make_review_sheets score-summaries review/summaries_r1.csv --json

Exit status: 0 on success (sampling), or when a scored gate is met; 1 when a
scored gate is not met; 2 for a usage/fixture/sheet error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from nlp.eval.review import (
    DEFAULT_ASSIGNMENT_THRESHOLD,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SUMMARY_DAYS,
    DEFAULT_SUMMARY_THRESHOLD,
    ReviewSamplingError,
    Scorecard,
    load_theme_sets,
    sample_assignments,
    sample_summary_sentences,
    score_assignments,
    score_summaries,
    write_assignment_csv,
    write_sentence_csv,
)
from ai.summarization import SummarizationError
from nlp.themes.dataset import DEFAULT_FIXTURE_PATH
from nlp.themes.errors import ThemeError
from nlp.themes.vectors import DEFAULT_VECTOR_PATH


def _add_fixture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR_PATH)


def _cmd_sample_assignments(args: argparse.Namespace) -> int:
    _, theme_sets = load_theme_sets(args.fixture, args.vectors)
    sample = sample_assignments(theme_sets, sample_size=args.sample_size, seed=args.seed)
    path = write_assignment_csv(sample, args.out)
    print(
        f"wrote {sample.actual_sample_size}/{sample.requested_sample_size} "
        f"assignment rows (population {sample.population_size}) to {path}"
    )
    if sample.actual_sample_size < sample.requested_sample_size:
        print(
            "NOTE: population is smaller than the requested sample size; "
            "the full population was taken instead. This is expected until "
            "real soak-window data (#57) replaces the M5 fixture.",
            file=sys.stderr,
        )
    return 0


def _cmd_sample_summaries(args: argparse.Namespace) -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set. sample-summaries calls the real "
            "Gemini API and has no offline substitute -- set the key to "
            "run it for real.",
            file=sys.stderr,
        )
        return 2

    from ai.summarization import GeminiClient

    _, theme_sets = load_theme_sets(args.fixture, args.vectors)
    client = GeminiClient()
    sample = sample_summary_sentences(theme_sets, client=client, days=args.days, seed=args.seed)
    path = write_sentence_csv(sample, args.out)
    print(
        f"wrote {len(sample.rows)} sentence rows from "
        f"{sample.actual_days}/{sample.requested_days} sampled day(s) "
        f"(of {sample.day_population_size} available) to {path}"
    )
    return 0


def _print_scorecard(scorecard: Scorecard, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(scorecard.as_dict(), indent=2, sort_keys=True))
        return
    print(scorecard.trust_contract.warning)
    print(f"sample_size      {scorecard.sample_size}")
    print(f"reviewer_count   {scorecard.reviewer_count}")
    if scorecard.agreement_rate is not None:
        print(f"agreement_rate   {scorecard.agreement_rate:.4f}")
    print(f"resolved         {scorecard.resolved_count}")
    print(f"unresolved       {scorecard.unresolved_count}")
    if scorecard.unresolved_row_ids:
        print(f"  unresolved ids: {', '.join(scorecard.unresolved_row_ids)}")
    print(f"rate             {'n/a' if scorecard.rate is None else f'{scorecard.rate:.4f}'}")
    print(f"gate_threshold   {scorecard.gate_threshold}")
    print(f"meets_gate       {'yes' if scorecard.meets_gate else 'NO'}")


def _cmd_score(args: argparse.Namespace, *, scorer) -> int:
    scorecard = scorer(args.sheets, adjudicated=args.adjudicated, threshold=args.threshold)
    _print_scorecard(scorecard, as_json=args.json)
    return 0 if scorecard.meets_gate else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_review_sheets",
        description="Sample and score Phase 0 human-review sheets (issue #74 / A4).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_assign = subparsers.add_parser(
        "sample-assignments", help="sample story->theme assignments (gate G1)"
    )
    _add_fixture_args(p_assign)
    p_assign.add_argument("--out", type=Path, required=True)
    p_assign.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p_assign.add_argument("--seed", default="phase0-a4-assignments")

    p_summ = subparsers.add_parser(
        "sample-summaries", help="sample cited summary sentences (gate G2)"
    )
    _add_fixture_args(p_summ)
    p_summ.add_argument("--out", type=Path, required=True)
    p_summ.add_argument("--days", type=int, default=DEFAULT_SUMMARY_DAYS)
    p_summ.add_argument("--seed", default="phase0-a4-summaries")

    p_score_a = subparsers.add_parser(
        "score-assignments", help="compute gate G1 from completed sheet(s)"
    )
    p_score_a.add_argument("sheets", nargs="+", type=Path)
    p_score_a.add_argument("--adjudicated", type=Path)
    p_score_a.add_argument("--threshold", type=float, default=DEFAULT_ASSIGNMENT_THRESHOLD)
    p_score_a.add_argument("--json", action="store_true")

    p_score_s = subparsers.add_parser(
        "score-summaries", help="compute gate G2 from completed sheet(s)"
    )
    p_score_s.add_argument("sheets", nargs="+", type=Path)
    p_score_s.add_argument("--adjudicated", type=Path)
    p_score_s.add_argument("--threshold", type=float, default=DEFAULT_SUMMARY_THRESHOLD)
    p_score_s.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.command == "sample-assignments":
            return _cmd_sample_assignments(args)
        if args.command == "sample-summaries":
            return _cmd_sample_summaries(args)
        if args.command == "score-assignments":
            return _cmd_score(args, scorer=score_assignments)
        if args.command == "score-summaries":
            return _cmd_score(args, scorer=score_summaries)
    except (ThemeError, ReviewSamplingError, SummarizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 2  # pragma: no cover - argparse enforces a valid command


if __name__ == "__main__":
    raise SystemExit(main())

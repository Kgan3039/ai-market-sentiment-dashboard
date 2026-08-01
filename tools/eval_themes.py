"""``eval_themes.py``: measure M5 theme clustering on the committed days.

Issue #72's DoD asks for AC-4 demonstrated on three ticker-days of varying
volume.  This runs that demonstration and writes a diffable record.

    python -m tools.eval_themes
    python -m tools.eval_themes --json \
        --write nlp/themes/data/results/theme_quality.json

Exit status is 0 when every day satisfies AC-4's shape and loses no story,
1 when one does not, and 2 for a usage or fixture error.

The numbers are produced with the real Phase 0 encoder, so this script
loads a model; the unit tests do not.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from nlp.themes.config import ThemeConfig
from nlp.themes.dataset import DEFAULT_FIXTURE_PATH, load_ticker_days, tickers_of
from nlp.themes.errors import ThemeError
from nlp.themes.quality import TickerDayReport, evaluate_ticker_day


def _payload(report: TickerDayReport) -> dict[str, Any]:
    data = asdict(report)
    data["trading_day"] = report.trading_day.isoformat()
    data["themes"] = [list(theme) for theme in report.themes]
    data["other_coverage"] = list(report.other_coverage)
    # Runtime is a property of the machine, not of the algorithm; keeping it
    # out of the committed file means a re-run on a slower box does not read
    # as a behaviour change.
    data.pop("elapsed_seconds", None)
    return data


def _render(reports: Sequence[TickerDayReport]) -> str:
    lines: list[str] = []
    for report in reports:
        lines += [
            f"{report.ticker} {report.trading_day} ({report.volume} volume)",
            f"  stories           {report.story_count}",
            f"  method            {report.method}",
            f"  reason            {report.method_reason}",
            f"  themes            {report.theme_count}"
            f" (singletons {report.singleton_theme_count})",
            f"  other coverage    {report.other_coverage_count}",
            f"  excluded          {report.excluded_count}",
            f"  theme coverage    {report.theme_coverage:.4f}",
            "  mean cohesion     "
            + (
                "n/a" if report.mean_cohesion is None else f"{report.mean_cohesion:.4f}"
            ),
            "  max inter-theme   "
            + (
                "n/a"
                if report.max_inter_theme_similarity is None
                else f"{report.max_inter_theme_similarity:.4f}"
            ),
            f"  AC-4 shape        {'yes' if report.meets_ac4_shape else 'NO'}",
            f"  no story lost     {'yes' if report.no_story_lost else 'NO'}",
            f"  permutation safe  {'yes' if report.permutation_stable else 'NO'}",
            f"  rerun keeps ids   {'yes' if report.rerun_keeps_identity else 'NO'}",
            "  perturbation      "
            f"membership {report.perturbation.membership_retained:.2f}, "
            f"identity {report.perturbation.identity_retained:.2f}, "
            f"{report.perturbation.theme_count_before} -> "
            f"{report.perturbation.theme_count_after} themes",
            f"  runtime           {report.elapsed_seconds:.3f}s",
        ]
        for rank, label, stories, outlets, salience in report.themes:
            lines.append(
                f"    {rank}. {label}  "
                f"[{stories} stories, {outlets} outlets, salience {salience:.4f}]"
            )
        if report.other_coverage:
            lines.append(f"    other: {', '.join(report.other_coverage)}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_themes",
        description="Measure theme clustering on the committed ticker-days.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(argv)

    try:
        day_set = load_ticker_days(args.fixture)
    except ThemeError as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 2

    from nlp.embeddings import EmbeddingService

    encoder = EmbeddingService()
    config = ThemeConfig(supported_tickers=tickers_of(day_set))
    reports = [
        evaluate_ticker_day(
            day.stories,
            ticker=day.ticker,
            trading_day=day.trading_day,
            volume=day.volume,
            config=config,
            encoder=encoder,
        )
        for day in day_set.days
    ]

    payload = {
        "dataset_id": day_set.dataset_id,
        "model_name": encoder.model_name,
        "model_revision": encoder.model_revision,
        "ticker_days": [_payload(report) for report in reports],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    print(text if args.json else _render(reports), end="")

    failures = [
        report
        for report in reports
        if not (report.meets_ac4_shape and report.no_story_lost)
    ]
    for report in failures:
        print(
            f"AC-4 FAILED: {report.ticker} {report.trading_day}",
            file=sys.stderr,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""``eval_themes.py``: measure M5 theme clustering on the committed days.

Issue #72's DoD asks for AC-4 demonstrated on three ticker-days of varying
volume.  This runs that demonstration and writes a diffable record.

    python -m tools.eval_themes
    python -m tools.eval_themes --json \
        --write nlp/themes/data/results/theme_quality.json

Exit status is 0 when every day satisfies AC-4's shape and loses no story,
1 when one does not, and 2 for a usage or fixture error.

**Offline by default.**  The vectors come from the committed
``nlp/themes/data/story_vectors.json``, produced once from the real Phase 0
encoder and rounded to a documented precision.  That makes this script -
and the test that regenerates its artifact - run with no model load and no
network call, and makes the committed JSON byte-identical on any machine.

``--real-model`` recomputes the vectors from the encoder instead; add
``--write-vectors PATH`` to refresh the committed fixture.  That is the
only path that loads a model, and it is not on the default test route.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from nlp.themes.compatibility import (
    policy_fingerprint as compatibility_policy_fingerprint,
)
from nlp.themes.config import (
    ALGORITHM_VERSION,
    SCORE_PRECISION,
    SEMANTIC_INPUT_COMPOSITION,
    ThemeConfig,
)
from nlp.themes.dataset import (
    DEFAULT_FIXTURE_PATH,
    SUPPORTED_SCHEMA_VERSION,
    load_ticker_days,
    tickers_of,
)
from nlp.themes.errors import ThemeError
from nlp.themes.trust import derive_stage_trust_summary
from nlp.themes.vectors import (
    DEFAULT_VECTOR_PATH,
    FixtureEncoder,
    load_story_vectors,
    write_story_vectors,
)
from nlp.themes.quality import TickerDayReport, evaluate_ticker_day


def _plain(value: Any) -> Any:
    """Render tuples as lists and round every float, on one code path.

    One path, one precision: a value rounded in three places eventually
    gets rounded differently in one of them, and the committed artifact
    stops being byte-stable without anything having changed.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, SCORE_PRECISION)
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _payload(report: TickerDayReport) -> dict[str, Any]:
    data = {key: _plain(value) for key, value in asdict(report).items()}
    data["trading_day"] = report.trading_day.isoformat()
    # Runtime is a property of the machine, not of the algorithm; keeping it
    # out of the committed file means a re-run on a slower box does not read
    # as a behaviour change.
    data.pop("elapsed_seconds", None)
    for name, stability in (
        ("permutation", report.permutation),
        ("perturbation", report.perturbation),
    ):
        data[name]["interpretation"] = stability.interpretation
    # The accounting the "no story lost" boolean stands on, spelled out: a
    # bool cannot distinguish a lost story from an invented one.
    data["story_accounting"] = {
        "input_story_count": report.story_count,
        "in_themes": sum(detail.member_count for detail in report.theme_details),
        "other_coverage": report.other_coverage_count,
        "excluded": report.excluded_count,
        "accounted": (
            sum(detail.member_count for detail in report.theme_details)
            + report.other_coverage_count
            + report.excluded_count
        ),
        "missing_story_keys": list(report.missing_story_keys),
        "unexpected_story_keys": list(report.unexpected_story_keys),
        "duplicate_membership_keys": list(report.duplicate_membership_keys),
        "complete": report.no_story_lost
        and not report.duplicate_membership_keys
        and not report.unexpected_story_keys,
    }
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
            "  min pairwise      "
            + (
                "n/a"
                if report.min_pairwise_cohesion is None
                else f"{report.min_pairwise_cohesion:.4f}"
            ),
            f"  AC-4 detail       {report.ac4_shape_detail}",
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
            f"                    {report.perturbation.interpretation}",
            f"  runtime           {report.elapsed_seconds:.3f}s",
        ]
        for detail in report.theme_details:
            flag = " NEAR COHESION FLOOR" if detail.near_cohesion_floor else ""
            lines.append(
                f"    {detail.rank}. {detail.label}  "
                f"[{detail.member_count} stories, {detail.outlet_count} outlets, "
                f"salience {detail.salience:.4f}, cohesion {detail.cohesion:.4f} "
                f"(min pair {detail.min_pairwise_cohesion:.4f}, "
                f"margin {detail.cohesion_margin:+.4f}){flag}]"
            )
            lines.append(f"        members: {', '.join(detail.member_story_keys)}")
        for reason, keys in report.other_coverage_by_reason.items():
            lines.append(f"    other ({reason}): {', '.join(keys)}")
        for reason, keys in report.excluded_by_reason.items():
            lines.append(f"    excluded ({reason}): {', '.join(keys)}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_themes",
        description="Measure theme clustering on the committed ticker-days.",
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR_PATH)
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="encode with the real model instead of the committed vectors",
    )
    parser.add_argument(
        "--write-vectors",
        type=Path,
        help="refresh the committed vector fixture (implies --real-model)",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", type=Path)
    args = parser.parse_args(argv)

    try:
        day_set = load_ticker_days(args.fixture)
    except ThemeError as exc:
        print(f"fixture error: {exc}", file=sys.stderr)
        return 2

    if args.real_model or args.write_vectors:
        from nlp.embeddings import EmbeddingService

        encoder: Any = EmbeddingService()
        if args.write_vectors:
            write_story_vectors(day_set, encoder, args.write_vectors)
    else:
        try:
            fixture_encoder = FixtureEncoder(load_story_vectors(args.vectors))
        except ThemeError as exc:
            print(f"vector fixture error: {exc}", file=sys.stderr)
            return 2
        from nlp.embeddings import compose_embedding_text

        fixture_encoder.bind(
            {
                story.story_key: compose_embedding_text(story.title, story.description)
                for day in day_set.days
                for story in day.stories
            }
        )
        encoder = fixture_encoder
    config = ThemeConfig(supported_tickers=tickers_of(day_set))
    # One ticker-day that refuses to cluster must not erase the days that
    # did.  Each failure is named and counted; the run reports incomplete
    # rather than reporting fewer days as if that were the whole fixture.
    reports: list[TickerDayReport] = []
    failed: list[dict[str, str]] = []
    for day in day_set.days:
        case_id = f"{day.ticker} {day.trading_day.isoformat()}"
        try:
            reports.append(
                evaluate_ticker_day(
                    day.stories,
                    ticker=day.ticker,
                    trading_day=day.trading_day,
                    volume=day.volume,
                    config=config,
                    encoder=encoder,
                )
            )
        except ThemeError as exc:
            failed.append(
                {"case_id": case_id, "error": type(exc).__name__, "detail": str(exc)}
            )

    evaluated = [_payload(report) for report in reports]
    payload = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "dataset_id": day_set.dataset_id,
        "dataset_version": day_set.metadata.get("schema_version"),
        "issue": day_set.metadata.get("issue"),
        "acceptance_criteria": day_set.metadata.get("acceptance_criteria"),
        "trust_contract": day_set.trust_contract.as_dict(),
        "trust_summary": day_set.trust_summary.as_dict(),
        "stage_specific_trust_summary": derive_stage_trust_summary(
            day_set.trust_contract
        ).as_dict(),
        "vector_source": getattr(encoder, "source", "real_model"),
        "known_limitations": list(day_set.known_limitations),
        "model_name": encoder.model_name,
        "model_revision": encoder.model_revision,
        "embedding_dimension": getattr(encoder, "dimension", None),
        "semantic_input_composition": SEMANTIC_INPUT_COMPOSITION,
        "theme_config_fingerprint": config.fingerprint(
            model_name=encoder.model_name,
            model_revision=encoder.model_revision,
            embedding_dimension=getattr(encoder, "dimension", None),
        ),
        "theme_policy_components": {
            key: _plain(value)
            for key, value in config.fingerprint_components(
                model_name=encoder.model_name,
                model_revision=encoder.model_revision,
                embedding_dimension=getattr(encoder, "dimension", None),
            ).items()
        },
        "compatibility_policy_fingerprint": compatibility_policy_fingerprint(),
        "algorithm_version": ALGORITHM_VERSION,
        "score_precision": SCORE_PRECISION,
        "case_count": len(day_set.days),
        "evaluated_case_count": len(evaluated),
        "failed_case_count": len(failed),
        "failed_cases": failed,
        "complete": not failed
        and all(row["story_accounting"]["complete"] for row in evaluated),
        "ticker_days": evaluated,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
    if args.json:
        print(text, end="")
    else:
        banner = (
            day_set.trust_summary.text
            + "\n"
            + derive_stage_trust_summary(day_set.trust_contract).text
        )
        print(banner)
        print()
        print(_render(reports), end="")
        print(banner)

    for case in failed:
        print(
            f"EVALUATION FAILED: {case['case_id']}: {case['detail']}", file=sys.stderr
        )
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
    return 1 if (failures or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())

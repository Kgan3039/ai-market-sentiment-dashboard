#!/usr/bin/env python3
"""Benchmark all Phase 0 narrative endpoints against SQLite or live HTTP.

Local mode can build a deterministic, provider-free representative database.
Remote mode sends the same request mix to a deployed API. The JSON report
records request counts, per-route latency percentiles, dataset size when local,
and enough runtime information to reproduce the measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "backend", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import ai.summarization as summarization  # noqa: E402
from app.phase0 import routes as phase0_routes  # noqa: E402
from app.phase0.sqlite_repository import SqliteNarrativeRepository  # noqa: E402
from main import create_app  # noqa: E402
from nlp.dedup.selection import cluster_fingerprint_for  # noqa: E402
from phase0.models import (  # noqa: E402
    OtherCoverageRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import (  # noqa: E402
    Phase0Reader,
    Phase0Repository,
    sqlite_runtime_version,
)
from phase0.summary_lifecycle import ensure_summary  # noqa: E402
from phase0.summary_runner import PRODUCTION_MAX_ATTEMPTS  # noqa: E402
from phase0.tickers import TICKER_UNIVERSE  # noqa: E402

DEFAULT_VERSION = "phase0-benchmark-v1"
DEFAULT_END_DATE = "2026-10-30"
DEFAULT_DAYS = 30
DEFAULT_THEMES = 4
DEFAULT_STORIES_PER_THEME = 6
DEFAULT_OTHER_STORIES = 6
DEFAULT_REQUESTS = 200
DEFAULT_WARMUP = 10
DEFAULT_MAX_P95_MS = 300.0
OUTLETS = ("Reuters", "Bloomberg", "CNBC", "MarketWatch", "TechWire")
ID_LINE_RE = re.compile(r"- id: (\S+)")


class BenchmarkSummaryClient:
    """Deterministic fake used only while building benchmark artifacts."""

    model = summarization.DEFAULT_MODEL
    max_output_tokens = summarization.DEFAULT_MAX_OUTPUT_TOKENS

    def generate(self, system_prompt, user_prompt, response_schema):
        del system_prompt
        citation_ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": "Representative coverage theme",
                "sentences": [
                    {
                        "text": "Coverage focused on this representative theme.",
                        "citation_ids": citation_ids,
                    },
                    {
                        "text": "Multiple reports contributed supporting evidence.",
                        "citation_ids": citation_ids[:1],
                    },
                ],
            }
        )


@dataclass
class BenchmarkClock:
    now: datetime

    def __call__(self) -> datetime:
        return self.now


def _trading_days(end_date: str, count: int) -> list[str]:
    cursor = date.fromisoformat(end_date)
    days: list[str] = []
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor -= timedelta(days=1)
    return list(reversed(days))


def _story_record(
    *,
    ticker: str,
    raw_item_id: int,
    title: str,
    outlet: str,
    published_at: str,
) -> StoryRecord:
    fingerprint = cluster_fingerprint_for(ticker, [str(raw_item_id)])
    url = f"https://benchmark.example/{ticker.lower()}/{raw_item_id}"
    return StoryRecord(
        cluster_fingerprint=fingerprint,
        canonical_title=title,
        members=(
            StoryMemberRecord(
                raw_item_id=raw_item_id,
                position=0,
                outlet=outlet,
                url=url,
                canonical_url=url,
            ),
        ),
        canonical_item_id=raw_item_id,
        outlet=outlet,
        outlet_count=1,
        published_at=published_at,
        canonical_url=url,
        content_hash=f"benchmark-{fingerprint[:16]}",
        stage="m3.semantic",
        member_story_keys=(fingerprint,),
        algorithm_version="benchmark-m3",
        config_fingerprint="benchmark-config",
        model_name="benchmark-model",
        model_revision="v1",
        embedding_dimension=4,
    )


def _partition_payloads(
    *,
    ticker: str,
    trading_day: str,
    themes: int,
    stories_per_theme: int,
    other_stories: int,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    assignments: list[tuple[str, str]] = []
    for theme_index in range(themes):
        label = f"Theme {theme_index + 1}"
        for story_index in range(stories_per_theme):
            assignments.append((label, f"story-{story_index + 1}"))
    assignments.extend(
        ("other", f"story-{index + 1}") for index in range(other_stories)
    )

    payloads = []
    for index, (label, story_name) in enumerate(assignments):
        outlet = OUTLETS[index % len(OUTLETS)]
        hour, minute = divmod(index, 60)
        published_at = f"{trading_day}T{13 + hour:02d}:{minute:02d}:00+00:00"
        title = f"{ticker} {trading_day} {label} {story_name}"
        payloads.append(
            {
                "source": f"yahoo:{outlet}",
                "ticker": ticker,
                "title": title,
                "description": f"Representative description for {title}.",
                "url": (
                    f"https://benchmark.example/{ticker.lower()}/"
                    f"{trading_day}/{index}"
                ),
                "canonical_url": (
                    f"https://benchmark.example/{ticker.lower()}/{trading_day}/{index}"
                ),
                "published_at": published_at,
                "fetched_at": f"{trading_day}T20:00:00+00:00",
                "raw_json": {"benchmark": True, "index": index},
            }
        )
    return payloads, assignments


def _seed_partition(
    repository: Phase0Repository,
    *,
    ticker: str,
    trading_day: str,
    pipeline_version: str,
    themes: int,
    stories_per_theme: int,
    other_stories: int,
) -> list[int]:
    payloads, assignments = _partition_payloads(
        ticker=ticker,
        trading_day=trading_day,
        themes=themes,
        stories_per_theme=stories_per_theme,
        other_stories=other_stories,
    )
    inserted = repository.admin.insert_raw_items(payloads)
    story_records = []
    for payload, result in zip(payloads, inserted):
        story_records.append(
            _story_record(
                ticker=ticker,
                raw_item_id=result.item_id,
                title=payload["title"],
                outlet=payload["source"].split(":", 1)[1],
                published_at=payload["published_at"],
            )
        )

    partition_key = f"{ticker.lower()}-{trading_day}"
    with repository.stage_run(
        run_id=f"benchmark-stories-{partition_key}",
        stage="stories",
        ticker=ticker,
        trading_day=trading_day,
        pipeline_version=pipeline_version,
    ) as run:
        repository.reconcile_stories(
            run=run,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=pipeline_version,
            stories=story_records,
        )

    rows = repository.stories_for_day(trading_day, ticker)
    story_by_title = {row["canonical_title"]: row for row in rows}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for payload, (label, _) in zip(payloads, assignments):
        grouped.setdefault(label, []).append(story_by_title[payload["title"]])

    theme_records = []
    for theme_index in range(themes):
        label = f"Theme {theme_index + 1}"
        members = grouped[label]
        theme_records.append(
            ThemeRecord(
                fingerprint=f"benchmark-{partition_key}-theme-{theme_index + 1}",
                theme_key=f"benchmark:{partition_key}:theme:{theme_index + 1}",
                label=label,
                label_source="canonical_story_title",
                story_ids=tuple(row["id"] for row in members),
                citation_item_ids=tuple(row["canonical_item_id"] for row in members),
                status="ready",
                salience_rank=theme_index + 1,
                story_count=len(members),
            )
        )
    other_rows = grouped.get("other", [])
    with repository.stage_run(
        run_id=f"benchmark-themes-{partition_key}",
        stage="themes",
        ticker=ticker,
        trading_day=trading_day,
        pipeline_version=pipeline_version,
    ) as run:
        repository.reconcile_themes(
            run=run,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=pipeline_version,
            theme_set=ThemeSetRecord(
                method="hdbscan",
                method_reason="representative benchmark",
                source_metadata={"story_count": len(rows)},
                config_fingerprint="benchmark-config",
                algorithm_version="benchmark-m5",
                model_name="benchmark-model",
                model_revision="v1",
                embedding_dimension=4,
            ),
            themes=theme_records,
            other_coverage=[
                OtherCoverageRecord(
                    story_id=row["id"],
                    reason="clustering_noise",
                    position=index,
                )
                for index, row in enumerate(other_rows)
            ],
            excluded=[],
            terminal=True,
        )

    population = repository.read.theme_population(ticker, trading_day, pipeline_version)
    return [theme.theme_id for theme in population.themes]


def build_representative_database(
    database_path: Path,
    *,
    pipeline_version: str = DEFAULT_VERSION,
    end_date: str = DEFAULT_END_DATE,
    days: int = DEFAULT_DAYS,
    themes: int = DEFAULT_THEMES,
    stories_per_theme: int = DEFAULT_STORIES_PER_THEME,
    other_stories: int = DEFAULT_OTHER_STORIES,
) -> dict[str, Any]:
    """Create one deterministic synthetic database through real write paths."""

    if database_path.exists():
        raise FileExistsError(f"refusing to replace existing database: {database_path}")
    for value, name in (
        (days, "days"),
        (themes, "themes"),
        (stories_per_theme, "stories_per_theme"),
        (other_stories, "other_stories"),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    trading_days = _trading_days(end_date, days)
    clock = BenchmarkClock(datetime.fromisoformat(f"{trading_days[0]}T20:00:00+00:00"))
    repository = Phase0Repository(database_path, clock=clock)
    repository.migrate()
    latest_theme_ids: dict[str, list[int]] = {}

    for trading_day in trading_days:
        clock.now = datetime.fromisoformat(f"{trading_day}T20:00:00+00:00")
        for ticker in TICKER_UNIVERSE:
            latest_theme_ids[ticker] = _seed_partition(
                repository,
                ticker=ticker,
                trading_day=trading_day,
                pipeline_version=pipeline_version,
                themes=themes,
                stories_per_theme=stories_per_theme,
                other_stories=other_stories,
            )

    latest_day = trading_days[-1]
    client = BenchmarkSummaryClient()
    for ticker, theme_ids in latest_theme_ids.items():
        for theme_id in theme_ids:
            with repository.stage_run(
                run_id=f"benchmark-summary-{ticker.lower()}-{theme_id}",
                stage="summaries",
                ticker=ticker,
                trading_day=latest_day,
                pipeline_version=pipeline_version,
            ) as run:
                ensure_summary(
                    repository,
                    run=run,
                    ticker=ticker,
                    trading_day=latest_day,
                    pipeline_version=pipeline_version,
                    theme_id=theme_id,
                    client=client,
                    max_attempts=PRODUCTION_MAX_ATTEMPTS,
                )

    return {
        "synthetic": True,
        "tickers": len(TICKER_UNIVERSE),
        "trading_days": days,
        "themes_per_ticker_day": themes,
        "stories_per_theme": stories_per_theme,
        "other_stories_per_ticker_day": other_stories,
        "latest_day": latest_day,
        "pipeline_version": pipeline_version,
    }


def database_metrics(database_path: Path) -> dict[str, Any]:
    tables = (
        "raw_items",
        "stories",
        "themes",
        "theme_sets",
        "run_log",
        "summary_artifacts",
    )
    reader = Phase0Reader(database_path)
    counts = {table: reader.count(table) for table in tables}
    return {
        "database_bytes": database_path.stat().st_size,
        "sqlite": sqlite_runtime_version(),
        "rows": counts,
    }


def _percentile_95(samples: list[float]) -> float:
    return quantiles(samples, n=100, method="inclusive")[94]


def _measure(
    get: Callable[[str], Any],
    paths: Iterable[str],
    *,
    requests: int,
    warmup: int,
) -> dict[str, Any]:
    path_list = list(paths)
    for index in range(warmup):
        response = get(path_list[index % len(path_list)])
        if response.status_code != 200:
            raise RuntimeError(
                f"warmup returned {response.status_code}: {response.text[:200]}"
            )

    latencies = []
    response_bytes = []
    for index in range(requests):
        path = path_list[index % len(path_list)]
        started = perf_counter()
        response = get(path)
        elapsed_ms = (perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(
                f"{path} returned {response.status_code}: {response.text[:200]}"
            )
        latencies.append(elapsed_ms)
        response_bytes.append(len(response.content))

    return {
        "requests": requests,
        "warmup_requests": warmup,
        "p50_ms": round(median(latencies), 3),
        "p95_ms": round(_percentile_95(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "mean_response_bytes": round(sum(response_bytes) / len(response_bytes), 1),
    }


def _endpoint_groups() -> dict[str, list[str]]:
    return {
        "tickers": ["/api/v1/tickers"],
        "themes": [f"/api/v1/tickers/{ticker}/themes" for ticker in TICKER_UNIVERSE],
        "status": ["/api/v1/meta/status"],
    }


def _runtime_report(mode: str, *, sqlite_version: str | None = None) -> dict[str, Any]:
    return {
        "mode": mode,
        "python": platform.python_version(),
        "sqlite": sqlite_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def _run_measurements(
    get: Callable[[str], Any],
    *,
    requests: int,
    warmup: int,
    max_p95_ms: float,
) -> tuple[dict[str, Any], bool]:
    endpoints = {
        name: _measure(get, paths, requests=requests, warmup=warmup)
        for name, paths in _endpoint_groups().items()
    }
    for result in endpoints.values():
        result["target_p95_ms"] = max_p95_ms
        result["passes"] = result["p95_ms"] < max_p95_ms
    return endpoints, all(result["passes"] for result in endpoints.values())


def benchmark_local(
    database_path: Path,
    *,
    pipeline_version: str,
    requests: int,
    warmup: int,
    max_p95_ms: float,
) -> dict[str, Any]:
    repository = SqliteNarrativeRepository(
        database_path=database_path,
        pipeline_version=pipeline_version,
    )
    app = create_app(serving_profile="phase0")
    app.dependency_overrides[phase0_routes.repository_dependency] = lambda: repository
    with TestClient(app) as client:
        endpoints, passed = _run_measurements(
            client.get,
            requests=requests,
            warmup=warmup,
            max_p95_ms=max_p95_ms,
        )
    dataset = database_metrics(database_path)
    return {
        "passed": passed,
        "environment": _runtime_report(
            "in-process FastAPI TestClient + SQLite",
            sqlite_version=dataset["sqlite"],
        ),
        "dataset": dataset,
        "endpoints": endpoints,
        "total_measured_requests": requests * len(endpoints),
    }


def benchmark_remote(
    base_url: str,
    *,
    requests: int,
    warmup: int,
    max_p95_ms: float,
    basic_auth_user: str | None = None,
    basic_auth_password: str | None = None,
) -> dict[str, Any]:
    auth = None
    if basic_auth_user:
        auth = (basic_auth_user, basic_auth_password or "")
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=15, auth=auth) as client:
        endpoints, passed = _run_measurements(
            client.get,
            requests=requests,
            warmup=warmup,
            max_p95_ms=max_p95_ms,
        )
    return {
        "passed": passed,
        "environment": _runtime_report("HTTP"),
        "base_url": base_url,
        "dataset": {"note": "Record host SQLite metrics alongside this report."},
        "endpoints": endpoints,
        "total_measured_requests": requests * len(endpoints),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Phase 0 narrative endpoint p95 latency"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--database", type=Path)
    target.add_argument("--base-url")
    parser.add_argument("--pipeline-version", default=DEFAULT_VERSION)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--max-p95-ms", type=float, default=DEFAULT_MAX_P95_MS)
    parser.add_argument("--seed-representative", action="store_true")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--themes", type=int, default=DEFAULT_THEMES)
    parser.add_argument(
        "--stories-per-theme", type=int, default=DEFAULT_STORIES_PER_THEME
    )
    parser.add_argument("--other-stories", type=int, default=DEFAULT_OTHER_STORIES)
    parser.add_argument("--basic-auth-user")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.requests < 2:
        parser.error("--requests must be at least 2")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.max_p95_ms <= 0:
        parser.error("--max-p95-ms must be positive")
    if args.seed_representative and args.database is None:
        parser.error("--seed-representative requires --database")
    if args.basic_auth_user and args.base_url is None:
        parser.error("--basic-auth-user requires --base-url")

    seed = None
    if args.database is not None:
        if args.seed_representative:
            seed = build_representative_database(
                args.database,
                pipeline_version=args.pipeline_version,
                end_date=args.end_date,
                days=args.days,
                themes=args.themes,
                stories_per_theme=args.stories_per_theme,
                other_stories=args.other_stories,
            )
        if not args.database.is_file():
            parser.error(f"database does not exist: {args.database}")
        report = benchmark_local(
            args.database,
            pipeline_version=args.pipeline_version,
            requests=args.requests,
            warmup=args.warmup,
            max_p95_ms=args.max_p95_ms,
        )
    else:
        password = os.environ.get("PHASE0_BENCHMARK_PASSWORD")
        if args.basic_auth_user and not password:
            parser.error(
                "--basic-auth-user requires PHASE0_BENCHMARK_PASSWORD "
                "in the environment"
            )
        report = benchmark_remote(
            args.base_url,
            requests=args.requests,
            warmup=args.warmup,
            max_p95_ms=args.max_p95_ms,
            basic_auth_user=args.basic_auth_user,
            basic_auth_password=password,
        )

    report["command"] = shlex.join([Path(sys.executable).name, *sys.argv])
    if seed is not None:
        report["seed"] = seed
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

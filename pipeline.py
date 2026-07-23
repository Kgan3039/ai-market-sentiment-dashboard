#!/usr/bin/env python3
"""Phase 0 scheduled ingestion pipeline.

Examples:
    python pipeline.py
    python pipeline.py --database /var/lib/ticker-narratives/phase0.sqlite3
    python pipeline.py --replay --date 2026-07-23
    python pipeline.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from phase0.repository import DEFAULT_DATABASE_PATH, Phase0Repository
from phase0.rss import RSSFetcher
from phase0.yahoo import YahooFinanceFetcher


ROOT = Path(__file__).resolve().parent
DEFAULT_FEEDS = ROOT / "config" / "feeds.yaml"
DEFAULT_ALIASES = ROOT / "config" / "aliases.yaml"
PIPELINE_VERSION = os.getenv("PHASE0_PIPELINE_VERSION", "phase0-v1")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
)
LOGGER = logging.getLogger("phase0.pipeline")


@dataclass
class Stage:
    name: str
    action: Callable[[], tuple[dict[str, int], list[dict[str, Any]]]]


def _log_event(event: str, **details: Any) -> None:
    LOGGER.info(json.dumps({"event": event, **details}, sort_keys=True, default=str))


def _stage_status(counts: dict[str, Any], errors: list[dict[str, Any] | str]) -> str:
    if not errors:
        return "success"
    success_counters = ("tickers_succeeded", "feeds_succeeded")
    if any(counter in counts and counts[counter] == 0 for counter in success_counters):
        return "failed"
    return "degraded"


def _run_stage(
    repository: Phase0Repository,
    *,
    run_id: str,
    trading_day: str,
    stage: Stage,
    pipeline_version: str,
) -> bool:
    started = datetime.now(timezone.utc)
    counts: dict[str, Any] = {}
    errors: list[dict[str, Any] | str] = []
    status = "success"
    try:
        counts, errors = stage.action()
        status = _stage_status(counts, errors)
    except Exception as exc:
        errors = [{"type": "stage_error", "error": str(exc)}]
        status = "failed"
    completed = datetime.now(timezone.utc)
    repository.log_stage(
        run_id=run_id,
        stage=stage.name,
        counts=counts,
        duration_ms=round((completed - started).total_seconds() * 1000),
        errors=errors,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        trading_day=trading_day,
        pipeline_version=pipeline_version,
        status=status,
    )
    _log_event(
        "stage_complete",
        run_id=run_id,
        stage=stage.name,
        status=status,
        counts=counts,
        errors=errors,
    )
    return status != "failed"


def run_live(
    repository: Phase0Repository,
    *,
    feeds_path: Path,
    aliases_path: Path,
    pipeline_version: str = PIPELINE_VERSION,
    trading_day: str | None = None,
) -> int:
    repository.migrate()
    day = trading_day or date.today().isoformat()
    run_id = str(uuid.uuid4())
    stages = [
        Stage("fetch_yahoo", YahooFinanceFetcher(repository).fetch),
        Stage(
            "fetch_rss",
            RSSFetcher(
                repository,
                feeds_path=feeds_path,
                aliases_path=aliases_path,
            ).fetch,
        ),
    ]
    _log_event("run_started", run_id=run_id, trading_day=day, mode="live")
    successes = [
        _run_stage(
            repository,
            run_id=run_id,
            trading_day=day,
            stage=stage,
            pipeline_version=pipeline_version,
        )
        for stage in stages
    ]
    # M2/M3/M5/A1/A2 register their stages here as they land. Keeping the
    # orchestration sequential makes those additions explicit and replayable.
    _log_event("run_complete", run_id=run_id, success=all(successes))
    return 0 if all(successes) else 1


def run_replay(
    repository: Phase0Repository,
    *,
    trading_day: str,
    pipeline_version: str = PIPELINE_VERSION,
) -> int:
    """Prepare a day for deterministic derived-stage replay.

    Raw input is never deleted. Until downstream dedup/cluster/summarize stages
    land, replay validates the immutable input and clears only derived rows.
    """
    repository.migrate()
    datetime.strptime(trading_day, "%Y-%m-%d")
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc)
    repository.clear_derived_for_day(trading_day)
    raw_count = len(repository.raw_items_for_day(trading_day))
    completed = datetime.now(timezone.utc)
    errors: list[dict[str, str]] = []
    status = "success"
    if raw_count == 0:
        errors.append({"type": "no_raw_input", "date": trading_day})
        status = "degraded"
    repository.log_stage(
        run_id=run_id,
        stage="replay_prepare",
        counts={"raw_items": raw_count, "derived_tables_cleared": 2},
        duration_ms=round((completed - started).total_seconds() * 1000),
        errors=errors,
        started_at=started.isoformat(),
        completed_at=completed.isoformat(),
        trading_day=trading_day,
        pipeline_version=pipeline_version,
        status=status,
    )
    _log_event(
        "replay_prepared",
        run_id=run_id,
        trading_day=trading_day,
        raw_items=raw_count,
        status=status,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 0 data pipeline")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--date", help="Trading day in YYYY-MM-DD format")
    parser.add_argument("--replay", action="store_true", help="Replay stored raw input")
    parser.add_argument(
        "--status", action="store_true", help="Print latest stage status"
    )
    parser.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Phase0Repository(args.database)
    if args.status:
        repository.migrate()
        print(json.dumps(repository.pipeline_status(), indent=2))
        return 0
    if args.replay:
        if not args.date:
            raise SystemExit("--replay requires --date YYYY-MM-DD")
        return run_replay(
            repository,
            trading_day=args.date,
            pipeline_version=args.pipeline_version,
        )
    return run_live(
        repository,
        feeds_path=args.feeds,
        aliases_path=args.aliases,
        pipeline_version=args.pipeline_version,
        trading_day=args.date,
    )


if __name__ == "__main__":
    sys.exit(main())

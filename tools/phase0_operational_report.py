#!/usr/bin/env python3
"""Read-only Phase 0 reliability evidence for an explicit UTC window.

The database has partition run rows, not an invocation ledger. This report
shows what those rows prove and keeps the G5–G7 decisions unmeasured when
their denominators or historical visibility cannot be established.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phase0.repository import Phase0Reader  # noqa: E402
from phase0.tickers import TICKER_UNIVERSE  # noqa: E402


EASTERN = ZoneInfo("America/New_York")
INVOCATION_ID = re.compile(
    r"^(phase0-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-" r"[0-9a-f]{4}-[0-9a-f]{12})(?=:)"
)
ACCEPTABLE = {"success"}
OUTCOME_SEVERITY = {
    "success": 0,
    "empty": 1,
    "partial": 2,
    "degraded": 3,
    "failed": 4,
    "unknown": 5,
}


def instant(value: str) -> datetime:
    """Require an unambiguous, timezone-aware ISO timestamp."""

    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return moment.astimezone(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def scheduled_slots(start: datetime, end: datetime) -> list[datetime]:
    """Evaluate the checked-in cron cadence in Eastern wall time, including DST."""

    candidate = start.replace(minute=(start.minute // 30) * 30, second=0, microsecond=0)
    slots = []
    while candidate < end:
        eastern = candidate.astimezone(EASTERN)
        weekday = eastern.weekday() < 5
        market = weekday and 9 <= eastern.hour <= 16
        if candidate >= start and (
            eastern.minute == 0 or (market and eastern.minute == 30)
        ):
            slots.append(candidate)
        candidate += timedelta(minutes=30)
    return slots


def _invocation_id(run_id: str) -> str:
    match = INVOCATION_ID.match(run_id)
    return match.group(1) if match else run_id


def _source_result(row: dict) -> tuple[str, str] | None:
    counts = json.loads(row["counts"])
    if row["stage"] == "fetch_yahoo" and row["ticker"]:
        source = f'yahoo:{row["ticker"]}'
    elif row["stage"] == "checkpoint_rss":
        parts = row["run_id"].rsplit(":", 3)
        if len(parts) != 4 or parts[-2] != "checkpoint":
            return None
        source = f"rss:{parts[-3]}"
    else:
        return None
    status = counts.get("source_state_status")
    if status not in {"success", "partial", "empty", "failed", "unknown"}:
        status = (
            row["status"]
            if row["status"] in {"success", "degraded", "failed"}
            else "unknown"
        )
    return source, status


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (95 * len(ordered) + 99) // 100 - 1  # nearest rank, one based
    return round(ordered[index], 3)


def build_report(
    reader: Phase0Reader, *, start: datetime, end: datetime, pipeline_version: str
) -> dict:
    if start >= end:
        raise ValueError("start must be earlier than end")
    if not pipeline_version.strip():
        raise ValueError("pipeline_version is required")

    slots = scheduled_slots(start, end)
    rows = reader.operational_run_rows(
        start=start, end=end, pipeline_version=pipeline_version
    )
    raw = reader.operational_raw_items(start=start, end=end)
    known_sources = set(reader.operational_sources())
    known_sources.update(f"yahoo:{ticker}" for ticker in TICKER_UNIVERSE)
    for row in rows:
        result = _source_result(row)
        if result:
            known_sources.add(result[0])
    sources = sorted(known_sources)

    invocations: dict[str, list[dict]] = defaultdict(list)
    replay_rows = 0
    for row in rows:
        # The RSS offline reclassifier uses its own stage name but does not
        # set run_log.replay, so both signals are required here.
        if row["replay"] or row["stage"] == "reclassify_rss":
            replay_rows += 1
            continue
        invocations[_invocation_id(row["run_id"])].append(row)

    checkpoint_times: dict[str, dict[str, datetime | None]] = {
        source: {"last_recorded_at": None, "last_successful_at": None}
        for source in sources
    }
    for group in invocations.values():
        for row in group:
            counts = json.loads(row["counts"])
            if "source_state_status" not in counts:
                continue
            result = _source_result(row)
            if result is None:
                continue
            source, outcome = result
            completed = instant(row["completed_at"])
            current = checkpoint_times[source]
            if (
                current["last_recorded_at"] is None
                or completed > current["last_recorded_at"]
            ):
                current["last_recorded_at"] = completed
            if outcome in {"success", "partial", "empty"} and (
                current["last_successful_at"] is None
                or completed > current["last_successful_at"]
            ):
                current["last_successful_at"] = completed

    starts = [
        min(instant(row["started_at"]) for row in group)
        for group in invocations.values()
    ]
    by_slot: dict[int, list[dict]] = defaultdict(list)
    unattributed = 0
    for (identity, group), first in zip(invocations.items(), starts):
        index = bisect_right(slots, first) - 1
        if index < 0:
            unattributed += 1
            continue
        by_slot[index].append({"id": identity, "rows": group})

    source_totals = {source: Counter() for source in sources}
    opportunities = []
    clean_candidates = 0
    missing = 0
    for index, slot in enumerate(slots):
        groups = by_slot.get(index, [])
        source_results: dict[str, str] = {}
        stage_failures = []
        for group in groups:
            for row in group["rows"]:
                result = _source_result(row)
                if result:
                    source, outcome = result
                    previous = source_results.get(source)
                    if (
                        previous is None
                        or OUTCOME_SEVERITY[outcome] > OUTCOME_SEVERITY[previous]
                    ):
                        source_results[source] = outcome
                if row["status"] in {"failed", "degraded"}:
                    stage_failures.append(row["stage"])
        for source in sources:
            source_totals[source][source_results.get(source, "missing")] += 1
        if not groups:
            missing += 1
            status = "no_durable_run"
        elif (
            sources
            and all(source_results.get(source) in ACCEPTABLE for source in sources)
            and not stage_failures
        ):
            clean_candidates += 1
            status = "clean_candidate"
        else:
            status = "incomplete_or_degraded"
        opportunities.append(
            {
                "scheduled_at": iso(slot),
                "market_hours": slot.astimezone(EASTERN).weekday() < 5
                and 9 <= slot.astimezone(EASTERN).hour <= 16,
                "status": status,
                "invocations_with_durable_rows": len(groups),
                "sources": {
                    source: source_results.get(source, "missing") for source in sources
                },
                "failed_or_degraded_stages": sorted(set(stage_failures)),
            }
        )

    lag_minutes = []
    missing_published = 0
    missing_provenance = 0
    invalid_items = 0
    future_published = 0
    by_source = Counter()
    by_feed = Counter()
    for item in raw:
        by_source[item["source"]] += 1
        for feed in (item["feed_sources"] or "").split(","):
            if feed:
                by_feed[feed] += 1
        missing_provenance += not bool(item["ingest_run_id"])
        invalid_items += item["ingest_status"] != "valid"
        if item["published_at"] is None:
            missing_published += 1
        else:
            lag = (
                instant(item["fetched_at"]) - instant(item["published_at"])
            ).total_seconds() / 60
            future_published += lag < 0
            lag_minutes.append(lag)

    market_slots = [entry for entry in opportunities if entry["market_hours"]]
    gap_minutes = 0
    streak = 0
    previous_day = None
    for entry in market_slots:
        day = instant(entry["scheduled_at"]).astimezone(EASTERN).date()
        if day != previous_day:
            streak = 0
            previous_day = day
        if entry["status"] == "no_durable_run":
            streak += 30
            gap_minutes = max(gap_minutes, streak)
        else:
            streak = 0

    return {
        "window": {
            "start": iso(start),
            "end": iso(end),
            "pipeline_version": pipeline_version,
        },
        "schedule_timezone": "America/New_York",
        "expected_sources": sources,
        "source_registry_note": (
            "RSS expectations are inferred from persisted source_state and run "
            "rows; a never-recorded feed is unknowable from SQLite."
        ),
        "scheduled_opportunities": len(slots),
        "opportunities_with_no_durable_run": missing,
        "clean_candidates": clean_candidates,
        "max_market_gap_without_durable_run_minutes": gap_minutes,
        "unattributed_invocations": unattributed,
        "excluded_replay_rows": replay_rows,
        "source_outcomes": {
            source: dict(sorted(counts.items()))
            for source, counts in source_totals.items()
        },
        "source_checkpoints_from_window_runs": {
            source: {
                key: iso(moment) if moment is not None else None
                for key, moment in times.items()
            }
            for source, times in checkpoint_times.items()
        },
        "raw_evidence": {
            "items": len(raw),
            "by_publisher_source": dict(sorted(by_source.items())),
            "by_feed_source": dict(sorted(by_feed.items())),
            "invalid_or_ambiguous": invalid_items,
            "missing_published_at": missing_published,
            "missing_ingest_run_id": missing_provenance,
            "future_published_at": future_published,
            "publish_to_fetch_p95_minutes": _p95(lag_minutes),
            "publish_to_fetch_samples": len(lag_minutes),
        },
        "gates": {
            "G5": {
                "status": "unmeasured",
                "candidate_successes": clean_candidates,
                "scheduled_opportunities": len(slots),
                "reason": (
                    "Partition rows cannot prove invocation completion or "
                    "distinguish a pre-write crash from a skipped invocation. "
                    "Correlate structured process logs before scoring."
                ),
            },
            "G6": {
                "status": "unmeasured",
                "reason": (
                    "publish-to-fetch is available above, but historical "
                    "publish-to-visible timestamps are not stored."
                ),
            },
            "G7": {
                "status": "unmeasured",
                "reason": (
                    "The ledger does not retain a historical count of themes "
                    "shipped with and without a current accepted summary."
                ),
            },
        },
        "opportunities": opportunities,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--start", required=True, help="Inclusive ISO timestamp with UTC offset"
    )
    parser.add_argument(
        "--end", required=True, help="Exclusive ISO timestamp with UTC offset"
    )
    parser.add_argument("--pipeline-version", default="phase0-v1")
    args = parser.parse_args(argv)
    try:
        report = build_report(
            Phase0Reader(args.database),
            start=instant(args.start),
            end=instant(args.end),
            pipeline_version=args.pipeline_version,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

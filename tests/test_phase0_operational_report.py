"""Operational evidence is deterministic, bounded, and honest about unknowns."""

from datetime import datetime, timezone
from uuid import UUID

from phase0.repository import Phase0Repository
from tools.phase0_operational_report import build_report, instant, scheduled_slots


DAY = "2026-08-18"
START = instant("2026-08-18T13:00:00Z")
END = instant("2026-08-18T15:00:00Z")
TICKERS = ("AAPL", "AMD", "META", "NVDA", "TSLA")


def log(
    repo,
    invocation,
    stage,
    suffix,
    at,
    *,
    ticker=None,
    status="success",
    source_status=None,
    replay=False,
):
    counts = {} if source_status is None else {"source_state_status": source_status}
    repo.admin.log_stage(
        run_id=f"{invocation}:{suffix}",
        stage=stage,
        counts=counts,
        duration_ms=10,
        errors=[{"type": "fixture_failure"}] if status != "success" else [],
        started_at=at,
        completed_at=at,
        trading_day=DAY,
        pipeline_version="phase0-v1",
        status=status,
        ticker=ticker,
        replay=replay,
    )


def test_schedule_handles_the_repeated_fall_dst_hour():
    start = instant("2026-11-01T04:00:00Z")
    end = instant("2026-11-01T08:00:00Z")
    assert scheduled_slots(start, end) == [
        datetime(2026, 11, 1, hour, tzinfo=timezone.utc) for hour in (4, 5, 6, 7)
    ]


def test_report_separates_quiet_coverage_failures_missing_slots_and_replay(tmp_path):
    repo = Phase0Repository(tmp_path / "phase0.sqlite3")
    repo.migrate()
    first = f"phase0-{UUID(int=1)}"
    second = f"phase0-{UUID(int=2)}"
    replay = f"phase0-{UUID(int=3)}"
    recovery = f"phase0-{UUID(int=4)}"
    clean = f"phase0-{UUID(int=5)}"

    for ticker in TICKERS:
        log(
            repo,
            first,
            "fetch_yahoo",
            f"yahoo:{ticker}:{DAY}",
            "2026-08-18T13:00:02+00:00",
            ticker=ticker,
            source_status="empty" if ticker == "NVDA" else "success",
        )
        log(
            repo,
            second,
            "fetch_yahoo",
            f"yahoo:{ticker}:{DAY}",
            "2026-08-18T13:30:02+00:00",
            ticker=ticker,
            status="degraded" if ticker == "NVDA" else "success",
            source_status="failed" if ticker == "NVDA" else "success",
        )
    for invocation, at, status in (
        (first, "2026-08-18T13:00:03+00:00", "success"),
        (second, "2026-08-18T13:30:03+00:00", "failed"),
    ):
        log(
            repo,
            invocation,
            "checkpoint_rss",
            f"rss:alpha:checkpoint:{DAY}",
            at,
            status="degraded" if status == "failed" else "success",
            source_status=status,
        )
    # A manual retry inside the same scheduled slot must not erase the
    # earlier failure from that slot's diagnostic account.
    log(
        repo,
        recovery,
        "fetch_yahoo",
        f"yahoo:NVDA:{DAY}",
        "2026-08-18T13:35:00+00:00",
        ticker="NVDA",
        source_status="success",
    )
    log(
        repo,
        recovery,
        "checkpoint_rss",
        f"rss:alpha:checkpoint:{DAY}",
        "2026-08-18T13:35:01+00:00",
        source_status="success",
    )
    log(
        repo,
        replay,
        "reclassify_rss",
        f"rss:alpha:scan:{DAY}",
        "2026-08-18T14:00:02+00:00",
    )
    log(
        repo,
        replay,
        "fetch_yahoo",
        f"yahoo:NVDA:{DAY}",
        "2026-08-18T14:00:03+00:00",
        ticker="NVDA",
        source_status="success",
        replay=True,
    )
    for ticker in TICKERS:
        log(
            repo,
            clean,
            "fetch_yahoo",
            f"yahoo:{ticker}:{DAY}",
            "2026-08-18T14:30:02+00:00",
            ticker=ticker,
            source_status="success",
        )
    log(
        repo,
        clean,
        "checkpoint_rss",
        f"rss:alpha:checkpoint:{DAY}",
        "2026-08-18T14:30:03+00:00",
        source_status="success",
    )

    before = repo.read.run_log_rows()
    report = build_report(repo.read, start=START, end=END, pipeline_version="phase0-v1")
    assert report == build_report(
        repo.read, start=START, end=END, pipeline_version="phase0-v1"
    )
    assert repo.read.run_log_rows() == before
    assert report["scheduled_opportunities"] == 4
    assert report["clean_candidates"] == 2
    assert report["opportunities_with_no_durable_run"] == 1
    assert report["excluded_replay_rows"] == 2
    assert report["opportunities"][1]["invocations_with_durable_rows"] == 2
    assert [item["status"] for item in report["opportunities"]] == [
        "clean_candidate",
        "incomplete_or_degraded",
        "no_durable_run",
        "clean_candidate",
    ]
    assert report["source_outcomes"]["yahoo:NVDA"] == {
        "empty": 1,
        "failed": 1,
        "missing": 1,
        "success": 1,
    }
    assert report["source_outcomes"]["rss:alpha"] == {
        "failed": 1,
        "missing": 1,
        "success": 2,
    }
    assert (
        report["source_checkpoints_from_window_runs"]["yahoo:NVDA"][
            "last_successful_at"
        ]
        == "2026-08-18T14:30:02+00:00"
    )
    assert all(gate["status"] == "unmeasured" for gate in report["gates"].values())
    assert report["gates"]["G5"]["denominator"] == 4
    assert report["gates"]["G5"]["unknowns"] == 4
    assert all(gate["numerator"] is None for gate in report["gates"].values())


def test_raw_freshness_is_a_fetch_proxy_and_missing_dates_remain_visible(tmp_path):
    repo = Phase0Repository(tmp_path / "phase0.sqlite3")
    repo.migrate()
    base = {
        "source": "yahoo:Example",
        "ticker": "NVDA",
        "title": "A valid headline",
        "description": "Fixture",
        "url": "https://example.com/one",
        "canonical_url": "https://example.com/one",
        "fetched_at": "2026-08-18T13:10:00+00:00",
        "raw_json": {"fixture": True},
    }
    repo.admin.insert_raw_item({**base, "published_at": "2026-08-18T13:00:00+00:00"})
    repo.admin.insert_raw_item(
        {
            **base,
            "url": "https://example.com/two",
            "canonical_url": "https://example.com/two",
            "published_at": None,
        }
    )
    report = build_report(repo.read, start=START, end=END, pipeline_version="phase0-v1")
    evidence = report["raw_evidence"]
    assert evidence["items"] == 2
    assert evidence["missing_published_at"] == 1
    assert evidence["missing_ingest_run_id"] == 2
    assert evidence["publish_to_fetch_p95_minutes"] == 10
    assert report["gates"]["G6"]["status"] == "unmeasured"


def test_ingest_only_yahoo_partition_does_not_claim_a_completed_check(tmp_path):
    repo = Phase0Repository(tmp_path / "phase0.sqlite3")
    repo.migrate()
    invocation = f"phase0-{UUID(int=6)}"
    for ticker in TICKERS:
        log(
            repo,
            invocation,
            "fetch_yahoo",
            f"yahoo:{ticker}:{DAY}",
            "2026-08-18T13:00:02+00:00",
            ticker=ticker,
            source_status=None if ticker == "NVDA" else "success",
        )
    log(
        repo,
        invocation,
        "checkpoint_rss",
        f"rss:alpha:checkpoint:{DAY}",
        "2026-08-18T13:00:03+00:00",
        source_status="success",
    )

    report = build_report(repo.read, start=START, end=END, pipeline_version="phase0-v1")

    assert report["opportunities"][0]["sources"]["yahoo:NVDA"] == "missing"
    assert report["opportunities"][0]["status"] == "incomplete_or_degraded"
    assert report["source_checkpoints_from_window_runs"]["yahoo:NVDA"] == {
        "last_recorded_at": None,
        "last_successful_at": None,
    }

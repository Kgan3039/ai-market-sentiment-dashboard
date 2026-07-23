import pytest

from phase0.repository import Phase0Repository
from pipeline import Stage, _run_stage, main, run_replay


def test_replay_keeps_raw_items_and_clears_derived_rows(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.insert_raw_item(
        {
            "source": "rss:test",
            "ticker": "NVDA",
            "title": "NVIDIA headline",
            "url": "https://example.com/nvidia",
            "canonical_url": "https://example.com/nvidia",
            "published_at": "2026-07-23T12:00:00+00:00",
        }
    )
    with repository.connect() as connection:
        connection.execute(
            """
            INSERT INTO stories
                (ticker, trading_day, canonical_title, outlet_count, member_ids)
            VALUES ('NVDA', '2026-07-23', 'NVIDIA headline', 1, '[1]')
            """
        )

    assert run_replay(repository, trading_day="2026-07-23") == 0
    assert repository.count("raw_items") == 1
    assert repository.count("stories") == 0
    assert repository.latest_stage_status()[0]["stage"] == "replay_prepare"


def test_replay_only_clears_selected_date(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    with repository.connect() as connection:
        for trading_day in ("2026-07-22", "2026-07-23"):
            connection.execute(
                """
                INSERT INTO stories
                    (ticker, trading_day, canonical_title, outlet_count, member_ids)
                VALUES ('NVDA', ?, 'Headline', 1, '[]')
                """,
                (trading_day,),
            )

    run_replay(repository, trading_day="2026-07-23")

    with repository.connect() as connection:
        days = [
            row[0]
            for row in connection.execute(
                "SELECT trading_day FROM stories ORDER BY trading_day"
            )
        ]
    assert days == ["2026-07-22"]


def test_cli_rejects_replay_without_date(tmp_path):
    with pytest.raises(SystemExit, match="--replay requires --date"):
        main(["--database", str(tmp_path / "db.sqlite3"), "--replay"])


def test_cli_rejects_invalid_replay_date(tmp_path):
    with pytest.raises(ValueError):
        main(
            [
                "--database",
                str(tmp_path / "db.sqlite3"),
                "--replay",
                "--date",
                "07/23/2026",
            ]
        )


def test_stage_fails_when_every_provider_target_fails(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    stage = Stage(
        "fetch_yahoo",
        lambda: (
            {"tickers_succeeded": 0},
            [{"ticker": "NVDA", "error": "offline"}],
        ),
    )

    succeeded = _run_stage(
        repository,
        run_id="run-1",
        trading_day="2026-07-23",
        stage=stage,
        pipeline_version="test",
    )

    assert succeeded is False
    assert repository.latest_stage_status()[0]["status"] == "failed"


def test_stage_is_degraded_when_other_targets_still_succeed(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    stage = Stage(
        "fetch_yahoo",
        lambda: (
            {"tickers_succeeded": 4},
            [{"ticker": "NVDA", "error": "offline"}],
        ),
    )

    succeeded = _run_stage(
        repository,
        run_id="run-1",
        trading_day="2026-07-23",
        stage=stage,
        pipeline_version="test",
    )

    assert succeeded is True
    assert repository.latest_stage_status()[0]["status"] == "degraded"

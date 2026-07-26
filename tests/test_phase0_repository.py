from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest

from phase0.repository import Phase0Repository


def sample_item(url="https://example.com/story?utm_source=test"):
    return {
        "source": "yahoo:Example",
        "ticker": "NVDA",
        "title": "NVIDIA announces a product",
        "description": "A description",
        "url": url,
        "canonical_url": "https://example.com/story",
        "published_at": "2026-07-23T12:00:00+00:00",
        "fetched_at": "2026-07-23T12:01:00+00:00",
        "raw_json": {"title": "NVIDIA announces a product"},
    }


def test_migration_enables_wal_and_creates_expected_tables(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    with repository.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert {
        "raw_items",
        "stories",
        "themes",
        "run_log",
        "eval_labels",
        "source_state",
        "pipeline_stage_keys",
        "raw_item_tickers",
        "raw_item_candidates",
        "story_members",
        "theme_stories",
        "theme_citations",
    } <= tables
    assert journal_mode == "wal"
    with repository.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_migrations_can_be_applied_repeatedly_without_schema_changes(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    with repository.connect() as connection:
        before = list(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            )
        )

    repository.migrate()

    with repository.connect() as connection:
        after = list(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            )
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert before == after


def test_raw_item_insert_is_idempotent(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    first = repository.insert_raw_item(sample_item())
    second = repository.insert_raw_item(sample_item("https://example.com/story"))

    assert first.inserted is True
    assert second.inserted is False
    assert first.item_id == second.item_id
    assert repository.count("raw_items") == 1


def test_concurrent_duplicate_inserts_remain_idempotent(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda _: repository.insert_raw_item(sample_item()), range(40))
        )

    assert sum(result.inserted for result in results) == 1
    assert len({result.item_id for result in results}) == 1
    assert repository.count("raw_items") == 1
    with repository.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_stage_status_decodes_structured_fields(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.log_stage(
        run_id="run-1",
        stage="fetch_yahoo",
        counts={"inserted": 2},
        duration_ms=5,
        errors=[{"ticker": "TSLA", "error": "offline"}],
        started_at="2026-07-23T12:00:00+00:00",
        completed_at="2026-07-23T12:00:01+00:00",
        trading_day="2026-07-23",
        pipeline_version="test",
    )

    status = repository.latest_stage_status()

    assert status[0]["counts"] == {"inserted": 2}
    assert status[0]["errors"][0]["ticker"] == "TSLA"
    assert status[0]["status"] == "degraded"
    assert repository.pipeline_status()["data_as_of"] == status[0]["completed_at"]


def test_source_state_persists_conditional_request_metadata(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.set_source_state(
        "rss:test",
        etag='"abc"',
        last_modified="Thu, 23 Jul 2026 12:00:00 GMT",
        checked_at="2026-07-23T12:01:00+00:00",
        successful=True,
        metadata={"item_count": 10},
    )

    state = repository.source_state("rss:test")

    assert state["etag"] == '"abc"'
    assert state["metadata"] == {"item_count": 10}
    assert state["last_success_at"] == "2026-07-23T12:01:00+00:00"


def test_derived_stage_idempotency_key_allows_retry_but_not_repeat_success(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    key = {
        "stage": "cluster",
        "ticker": "NVDA",
        "trading_day": "2026-07-23",
        "pipeline_version": "v1",
    }

    assert repository.claim_stage_key(**key, run_id="run-1") is True
    repository.complete_stage_key(**key, run_id="run-1", status="failed")
    assert repository.claim_stage_key(**key, run_id="run-2") is True
    repository.complete_stage_key(**key, run_id="run-2", status="success")
    assert repository.claim_stage_key(**key, run_id="run-3") is False


def test_exactly_one_concurrent_stage_claim_wins(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    def claim(index):
        return repository.claim_stage_key(
            stage="cluster",
            ticker="NVDA",
            trading_day="2026-07-23",
            pipeline_version="v1",
            run_id=f"run-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        claims = list(executor.map(claim, range(8)))

    assert claims.count(True) == 1
    assert claims.count(False) == 7


def test_replay_cleanup_removes_stage_keys_but_keeps_raw_evidence(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.insert_raw_item(sample_item())
    assert repository.claim_stage_key(
        stage="cluster",
        ticker="NVDA",
        trading_day="2026-07-23",
        pipeline_version="v1",
        run_id="run-1",
    )
    repository.complete_stage_key(
        stage="cluster",
        ticker="NVDA",
        trading_day="2026-07-23",
        pipeline_version="v1",
        run_id="run-1",
    )

    repository.clear_derived_for_day("2026-07-23")

    assert repository.count("raw_items") == 1
    assert repository.count("pipeline_stage_keys") == 0
    assert repository.claim_stage_key(
        stage="cluster",
        ticker="NVDA",
        trading_day="2026-07-23",
        pipeline_version="v1",
        run_id="run-2",
    )


def test_failed_migration_rolls_back_schema_and_version(tmp_path):
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_broken.sql").write_text(
        """
        CREATE TABLE partial_table (id INTEGER PRIMARY KEY);
        INSERT INTO missing_table (id) VALUES (1);
        """,
        encoding="utf-8",
    )
    repository = Phase0Repository(
        tmp_path / "phase0.sqlite3", migrations_path=migrations
    )

    with pytest.raises(sqlite3.OperationalError):
        repository.migrate()

    with repository.connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'partial_table'"
            ).fetchone()[0]
            == 0
        )


def test_story_theme_and_label_references_are_foreign_key_enforced(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    first = repository.insert_raw_item(sample_item()).item_id
    second_item = sample_item("https://example.com/second")
    second_item["canonical_url"] = "https://example.com/second"
    second = repository.insert_raw_item(second_item).item_id
    story_id = repository.insert_story(
        ticker="NVDA",
        trading_day="2026-07-23",
        canonical_title="NVIDIA story",
        member_ids=[first, second],
        outlet_count=2,
    )
    theme_id = repository.insert_theme(
        ticker="NVDA",
        trading_day="2026-07-23",
        label="Product coverage",
        story_ids=[story_id],
        citation_ids=[first],
        salience_rank=1,
        status="ready",
        content_hash="hash-1",
        pipeline_version="v1",
    )
    label_id = repository.insert_eval_label(
        label_type="dedup",
        item_a_id=first,
        item_b_id=second,
        reviewer="reviewer",
        label="duplicate",
    )

    assert story_id > 0
    assert theme_id > 0
    assert label_id > 0
    assert repository.count("story_members") == 2
    assert repository.count("theme_citations") == 1
    with repository.connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        declared = connection.execute(
            """
            SELECT COUNT(*) FROM pragma_foreign_key_list('story_members')
            """
        ).fetchone()[0]
        assert declared > 0


def test_invalid_relationships_roll_back_parent_rows(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    with pytest.raises(sqlite3.IntegrityError):
        repository.insert_story(
            ticker="NVDA",
            trading_day="2026-07-23",
            canonical_title="Missing member",
            member_ids=[999],
        )
    assert repository.count("stories") == 0

    first = repository.insert_raw_item(sample_item()).item_id
    second_item = sample_item("https://example.com/second")
    second_item["canonical_url"] = "https://example.com/second"
    second = repository.insert_raw_item(second_item).item_id
    story_id = repository.insert_story(
        ticker="NVDA",
        trading_day="2026-07-23",
        canonical_title="NVIDIA story",
        member_ids=[first],
    )
    with pytest.raises(ValueError, match="citations"):
        repository.insert_theme(
            ticker="NVDA",
            trading_day="2026-07-23",
            label="Invalid citation",
            story_ids=[story_id],
            citation_ids=[second],
            salience_rank=1,
            status="ready",
            content_hash="hash",
            pipeline_version="v1",
        )
    assert repository.count("themes") == 0


def test_raw_evidence_and_associations_survive_reconnect(tmp_path):
    database = tmp_path / "phase0.sqlite3"
    repository = Phase0Repository(database)
    repository.migrate()
    evidence = sample_item()
    evidence.update(
        {
            "ticker": None,
            "tickers": ["NVDA", "AMD"],
            "candidate_tickers": [
                {"ticker": "NVDA", "reason": "headline"},
                {"ticker": "AMD", "reason": "description"},
            ],
        }
    )
    item_id = repository.insert_raw_item(evidence).item_id

    reopened = Phase0Repository(database)
    assert reopened.count("raw_items") == 1
    assert reopened.raw_item_tickers(item_id) == ["AMD", "NVDA"]
    assert reopened.count("raw_item_candidates") == 2
    with reopened.connect() as connection:
        stored = connection.execute(
            "SELECT raw_json FROM raw_items WHERE id = ?", (item_id,)
        ).fetchone()[0]
    assert json.loads(stored)["title"] == "NVIDIA announces a product"


def test_invalid_raw_json_timestamp_and_status_are_rejected(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    invalid_json = sample_item()
    invalid_json["raw_json"] = "{not-json"
    with pytest.raises(ValueError, match="valid JSON"):
        repository.insert_raw_item(invalid_json)

    invalid_time = sample_item()
    invalid_time["published_at"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        repository.insert_raw_item(invalid_time)

    with pytest.raises(ValueError, match="run status"):
        repository.log_stage(
            run_id="run",
            stage="fetch",
            counts={},
            duration_ms=1,
            errors=[],
            started_at="2026-07-23T12:00:00Z",
            completed_at="2026-07-23T12:00:01Z",
            trading_day="2026-07-23",
            pipeline_version="v1",
            status="unknown",
        )

    with pytest.raises(ValueError, match="negative"):
        repository.log_stage(
            run_id="run",
            stage="fetch",
            counts={},
            duration_ms=-1,
            errors=[],
            started_at="2026-07-23T12:00:00Z",
            completed_at="2026-07-23T12:00:01Z",
            trading_day="2026-07-23",
            pipeline_version="v1",
        )


def test_database_constraints_reject_invalid_status_json_and_timestamps(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO run_log (
                    run_id, stage, counts, duration_ms, errors, started_at,
                    completed_at, status, trading_day, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "run",
                    "fetch",
                    "{bad-json",
                    1,
                    "[]",
                    "not-a-time",
                    "2026-07-23T12:00:01+00:00",
                    "unknown",
                    "2026-07-23",
                    "v1",
                ),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO raw_items (
                    source, title, url, canonical_url, fetched_at,
                    ingest_status, validation_errors, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "rss:test",
                    None,
                    None,
                    "urn:test",
                    "not-a-time",
                    "valid",
                    "[]",
                    "{}",
                ),
            )


def test_invalid_raw_evidence_is_preserved_without_required_display_fields(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    result = repository.insert_raw_item(
        {
            "source": "rss:test",
            "canonical_url": "urn:rss:test:entry-1",
            "external_id": "entry-1",
            "ingest_status": "invalid",
            "validation_errors": ["missing title", "missing link"],
            "raw_json": {"guid": "entry-1"},
        }
    )

    with repository.connect() as connection:
        row = connection.execute(
            "SELECT * FROM raw_items WHERE id = ?", (result.item_id,)
        ).fetchone()
    assert row["title"] is None
    assert row["url"] is None
    assert json.loads(row["validation_errors"]) == ["missing title", "missing link"]


def test_secret_bearing_error_fields_are_redacted(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.log_stage(
        run_id="run",
        stage="fetch",
        counts={},
        duration_ms=1,
        errors=[
            {
                "api_key": "top-secret",
                "message": "token=abc123 request failed",
            }
        ],
        started_at="2026-07-23T12:00:00Z",
        completed_at="2026-07-23T12:00:01Z",
        trading_day="2026-07-23",
        pipeline_version="v1",
    )

    error = repository.latest_stage_status()[0]["errors"][0]
    assert error["api_key"] == "[REDACTED]"
    assert "abc123" not in error["message"]

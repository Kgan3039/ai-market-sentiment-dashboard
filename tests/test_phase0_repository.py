from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3

import pytest

from phase0.repository import MIGRATIONS_PATH, Phase0Repository
from phase0.schema import load_migrations


LATEST_SCHEMA_VERSION = max(
    migration.version for migration in load_migrations(MIGRATIONS_PATH)
)


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

    with repository.admin.connect_writable() as connection:
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
    with repository.admin.connect_writable() as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == LATEST_SCHEMA_VERSION
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_migrations_can_be_applied_repeatedly_without_schema_changes(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    with repository.admin.connect_writable() as connection:
        before = list(
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            )
        )

    repository.migrate()

    with repository.admin.connect_writable() as connection:
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

    first = repository.admin.insert_raw_item(sample_item())
    second = repository.admin.insert_raw_item(sample_item("https://example.com/story"))

    assert first.inserted is True
    assert second.inserted is False
    assert first.item_id == second.item_id
    assert repository.count("raw_items") == 1


def test_concurrent_duplicate_inserts_remain_idempotent(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: repository.admin.insert_raw_item(sample_item()), range(40)
            )
        )

    assert sum(result.inserted for result in results) == 1
    assert len({result.item_id for result in results}) == 1
    assert repository.count("raw_items") == 1
    with repository.admin.connect_writable() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_stage_status_decodes_structured_fields(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.admin.log_stage(
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
    repository.admin.set_source_state(
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
    repository.admin.complete_stage_key(**key, run_id="run-1", status="failed")
    assert repository.claim_stage_key(**key, run_id="run-2") is True
    repository.admin.complete_stage_key(**key, run_id="run-2", status="success")
    assert repository.claim_stage_key(**key, run_id="run-3") is False


def test_exactly_one_concurrent_stage_claim_wins(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    def claim(index):
        return (
            index,
            repository.claim_stage_key(
                stage="cluster",
                ticker="NVDA",
                trading_day="2026-07-23",
                pipeline_version="v1",
                run_id=f"run-{index}",
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(claim, range(8)))

    claims = [claimed for _, claimed in attempts]
    assert claims.count(True) == 1
    assert claims.count(False) == 7
    winning_index = next(index for index, claimed in attempts if claimed)
    with repository.admin.connect_writable() as connection:
        owner = connection.execute("SELECT run_id FROM pipeline_stage_keys").fetchone()[
            0
        ]
    assert owner == f"run-{winning_index}"


def test_running_stage_claim_uses_lease_and_only_expires_at_boundary(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    key = {
        "stage": "cluster",
        "ticker": "NVDA",
        "trading_day": "2026-07-23",
        "pipeline_version": "v1",
    }

    assert repository.claim_stage_key(
        **key,
        run_id="run-1",
        lease_seconds=60,
        claimed_at="2026-07-23T12:00:00Z",
    )
    assert not repository.claim_stage_key(
        **key,
        run_id="run-2",
        lease_seconds=60,
        claimed_at="2026-07-23T12:00:59Z",
    )
    with repository.admin.connect_writable() as connection:
        active = dict(
            connection.execute(
                """
                SELECT run_id, lease_expires_at
                FROM pipeline_stage_keys
                """
            ).fetchone()
        )
    assert active == {
        "run_id": "run-1",
        "lease_expires_at": "2026-07-23T12:01:00+00:00",
    }

    assert repository.claim_stage_key(
        **key,
        run_id="run-2",
        lease_seconds=120,
        claimed_at="2026-07-23T12:01:00Z",
    )
    with repository.admin.connect_writable() as connection:
        reclaimed = dict(
            connection.execute(
                """
                SELECT run_id, lease_expires_at
                FROM pipeline_stage_keys
                """
            ).fetchone()
        )
    assert reclaimed == {
        "run_id": "run-2",
        "lease_expires_at": "2026-07-23T12:03:00+00:00",
    }


def test_replay_cleanup_removes_stage_keys_but_keeps_raw_evidence(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.admin.insert_raw_item(sample_item())
    assert repository.claim_stage_key(
        stage="cluster",
        ticker="NVDA",
        trading_day="2026-07-23",
        pipeline_version="v1",
        run_id="run-1",
    )
    repository.admin.complete_stage_key(
        stage="cluster",
        ticker="NVDA",
        trading_day="2026-07-23",
        pipeline_version="v1",
        run_id="run-1",
    )

    repository.admin.clear_derived_for_day("2026-07-23")

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

    with repository.admin.connect_writable() as connection:
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
    first = repository.admin.insert_raw_item(sample_item()).item_id
    second_item = sample_item("https://example.com/second")
    second_item["canonical_url"] = "https://example.com/second"
    second = repository.admin.insert_raw_item(second_item).item_id
    story_id = repository.admin.insert_story(
        ticker="NVDA",
        trading_day="2026-07-23",
        canonical_title="NVIDIA story",
        member_ids=[first, second],
        outlet_count=2,
        pipeline_version="v1",
    )
    theme_id = repository.admin.insert_theme(
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
    label_id = repository.admin.insert_eval_label(
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
    with repository.admin.connect_writable() as connection:
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
        repository.admin.insert_story(
            ticker="NVDA",
            trading_day="2026-07-23",
            canonical_title="Missing member",
            member_ids=[999],
            pipeline_version="v1",
        )
    assert repository.count("stories") == 0

    first = repository.admin.insert_raw_item(sample_item()).item_id
    second_item = sample_item("https://example.com/second")
    second_item["canonical_url"] = "https://example.com/second"
    second = repository.admin.insert_raw_item(second_item).item_id
    story_id = repository.admin.insert_story(
        ticker="NVDA",
        trading_day="2026-07-23",
        canonical_title="NVIDIA story",
        member_ids=[first],
        pipeline_version="v1",
    )
    with pytest.raises(ValueError, match="citations"):
        repository.admin.insert_theme(
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


def test_database_rejects_citation_outside_theme_member_stories(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    member = repository.admin.insert_raw_item(sample_item()).item_id
    other_item = sample_item("https://example.com/other")
    other_item["canonical_url"] = "https://example.com/other"
    non_member = repository.admin.insert_raw_item(other_item).item_id
    story_id = repository.admin.insert_story(
        ticker="NVDA",
        trading_day="2026-07-23",
        canonical_title="Member story",
        member_ids=[member],
        pipeline_version="v1",
    )
    theme_id = repository.admin.insert_theme(
        ticker="NVDA",
        trading_day="2026-07-23",
        label="Coverage",
        story_ids=[story_id],
        citation_ids=[member],
        salience_rank=1,
        status="ready",
        content_hash="hash",
        pipeline_version="v1",
    )

    with repository.admin.connect_writable() as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="member story",
        ):
            connection.execute(
                """
                INSERT INTO theme_citations (theme_id, raw_item_id)
                VALUES (?, ?)
                """,
                (theme_id, non_member),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="story member",
        ):
            connection.execute(
                """
                DELETE FROM story_members
                WHERE story_id = ? AND raw_item_id = ?
                """,
                (story_id, member),
            )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="theme story",
        ):
            connection.execute(
                """
                DELETE FROM theme_stories
                WHERE theme_id = ? AND story_id = ?
                """,
                (theme_id, story_id),
            )


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
    item_id = repository.admin.insert_raw_item(evidence).item_id

    reopened = Phase0Repository(database)
    assert reopened.count("raw_items") == 1
    assert reopened.raw_item_tickers(item_id) == ["AMD", "NVDA"]
    assert reopened.count("raw_item_candidates") == 2
    with reopened.admin.connect_writable() as connection:
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
        repository.admin.insert_raw_item(invalid_json)

    invalid_time = sample_item()
    invalid_time["published_at"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="ISO-8601"):
        repository.admin.insert_raw_item(invalid_time)

    with pytest.raises(ValueError, match="run status"):
        repository.admin.log_stage(
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
        repository.admin.log_stage(
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

    with repository.admin.connect_writable() as connection:
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
    result = repository.admin.insert_raw_item(
        {
            "source": "rss:test",
            "canonical_url": "urn:rss:test:entry-1",
            "external_id": "entry-1",
            "ingest_status": "invalid",
            "validation_errors": ["missing title", "missing link"],
            "raw_json": {"guid": "entry-1"},
        }
    )

    with repository.admin.connect_writable() as connection:
        row = connection.execute(
            "SELECT * FROM raw_items WHERE id = ?", (result.item_id,)
        ).fetchone()
    assert row["title"] is None
    assert row["url"] is None
    assert json.loads(row["validation_errors"]) == ["missing title", "missing link"]


def test_secret_bearing_error_fields_are_redacted(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.admin.log_stage(
        run_id="run",
        stage="fetch",
        counts={},
        duration_ms=1,
        errors=[
            {
                "api_key": "top-secret",
                "message": "token=abc123 request failed",
                "authorization_error": (
                    "Authorization: Bearer bearer-secret request failed"
                ),
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
    assert "bearer-secret" not in error["authorization_error"]


def test_nested_source_metadata_is_redacted(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    repository.admin.set_source_state(
        "rss:test",
        etag=None,
        last_modified=None,
        checked_at="2026-07-23T12:00:00Z",
        successful=False,
        metadata={
            "nested": {
                "authorization": "Bearer source-secret",
                "message": "Authorization: Bearer inline-secret failed",
            }
        },
    )

    metadata = repository.source_state("rss:test")["metadata"]

    assert metadata["nested"]["authorization"] == "[REDACTED]"
    assert "inline-secret" not in metadata["nested"]["message"]


def test_actual_legacy_v2_database_upgrades_to_latest_without_data_loss(tmp_path):
    database = tmp_path / "phase0.sqlite3"
    legacy_migrations = Path(__file__).parent / "fixtures" / "legacy_v2_migrations"
    legacy = Phase0Repository(database, migrations_path=legacy_migrations)
    legacy.migrate()
    with legacy.admin.connect_writable() as connection:
        # Databases published at v2 predate the migration ledger, so the
        # fixture drops it to reproduce one faithfully.
        connection.execute("DROP TABLE IF EXISTS schema_migrations")
    with legacy.admin.connect_writable() as connection:
        connection.executemany(
            """
            INSERT INTO raw_items (
                id, source, ticker, title, description, url,
                canonical_url, published_at, fetched_at, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "rss:test",
                    "NVDA",
                    "NVIDIA story",
                    "Description",
                    "https://example.com/one",
                    "https://example.com/one",
                    "2026-07-23T12:00:00+00:00",
                    "2026-07-23T12:01:00+00:00",
                    '{"guid":"one"}',
                ),
                (
                    2,
                    "rss:test",
                    "NVDA",
                    "Second story",
                    "Description",
                    "https://example.com/two",
                    "https://example.com/two",
                    "2026-07-23T12:02:00+00:00",
                    "2026-07-23T12:03:00+00:00",
                    '{"guid":"two"}',
                ),
            ],
        )
        connection.execute(
            """
            INSERT INTO stories (
                id, ticker, trading_day, canonical_title,
                outlet_count, member_ids
            ) VALUES (1, 'NVDA', '2026-07-23', 'NVIDIA story', 1, '[1]')
            """
        )
        connection.execute(
            """
            INSERT INTO themes (
                id, ticker, trading_day, label, citations,
                salience_rank, status, content_hash, pipeline_version
            ) VALUES (
                1, 'NVDA', '2026-07-23', 'Coverage', '[1]',
                1, 'ready', 'hash', 'v1'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO eval_labels (
                label_type, item_a_id, item_b_id, reviewer,
                label, created_at
            ) VALUES (
                'dedup', 1, 2, 'reviewer', 'different',
                '2026-07-23T13:00:00+00:00'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO run_log (
                run_id, stage, counts, duration_ms, errors,
                started_at, completed_at, status, trading_day,
                pipeline_version
            ) VALUES (
                'run', 'fetch', '{"inserted":2}', 5, '[]',
                '2026-07-23T12:00:00+00:00',
                '2026-07-23T12:00:01+00:00',
                'success', '2026-07-23', 'v1'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO source_state (
                source, last_checked_at, last_success_at, metadata
            ) VALUES (
                'rss:test', '2026-07-23T12:00:00+00:00',
                '2026-07-23T12:00:00+00:00', '{"status":"success"}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO pipeline_stage_keys (
                stage, ticker, trading_day, pipeline_version,
                status, run_id, updated_at
            ) VALUES (
                'cluster', 'NVDA', '2026-07-23', 'v1',
                'running', 'abandoned',
                '2026-07-23T12:00:00+00:00'
            )
            """
        )
    with legacy.admin.connect_writable() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.count("raw_items") == 2
    assert upgraded.count("story_members") == 1
    assert upgraded.count("theme_stories") == 1
    assert upgraded.count("theme_citations") == 1
    assert upgraded.count("eval_labels") == 1
    assert upgraded.count("run_log") == 1
    assert upgraded.count("source_state") == 1
    with upgraded.admin.connect_writable() as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == LATEST_SCHEMA_VERSION
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        lease = connection.execute(
            """
            SELECT lease_expires_at FROM pipeline_stage_keys
            WHERE run_id = 'abandoned'
            """
        ).fetchone()[0]
    assert lease == "2026-07-23T12:00:00+00:00"

from concurrent.futures import ThreadPoolExecutor

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

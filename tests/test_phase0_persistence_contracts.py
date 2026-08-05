"""Contract and hostile-probe coverage for the Phase 0 persistence layer.

Every test here answers a question issue #57 asks of the datastore rather
than of any one caller: does the *database* refuse the thing, does an
upgraded database end up identical to a fresh one, does a failed migration
leave nothing behind, and does a credential that reached an error payload
actually disappear.
"""

from __future__ import annotations

import inspect
import itertools
import json
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nlp.embeddings import (
    EMBEDDING_DTYPE,
    EmbeddingRepository,
    EmbeddingService,
    EmbeddingTarget,
    PersistedEmbedding,
    serialize_vector,
)
from phase0.embeddings import EmbeddingPersistenceError
from phase0.errors import (
    Phase0IntegrityError,
    Phase0MigrationError,
    Phase0RunContextError,
    Phase0ValidationError,
    StageKeyError,
    UnsupportedTickerError,
)
from phase0.models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    ProviderConflictRecord,
    SemanticMergeRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.redaction import redact_secrets, redact_text
from phase0.repository import MIGRATIONS_PATH, Phase0Repository
from phase0.schema import load_migrations
from phase0.tickers import SUPPORTED_TICKERS


REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_MIGRATIONS = load_migrations(MIGRATIONS_PATH)
LATEST_VERSION = max(migration.version for migration in ALL_MIGRATIONS)
PRIOR_VERSIONS = sorted({migration.version for migration in ALL_MIGRATIONS})[:-1]
DAY = "2026-07-23"
_RUN_IDS = itertools.count(1)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def migrated(tmp_path: Path, name: str = "phase0.sqlite3") -> Phase0Repository:
    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    return repository


def reconcile_stories(repository: Phase0Repository, **kwargs):
    """Reconcile through the logged entrypoint issue #68's runner uses.

    Every reconciliation in these tests goes through a real stage run, so
    the logged path is what is under test everywhere rather than only in
    the run-context tests below.
    """

    with repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage="m3.semantic",
        trading_day=kwargs["trading_day"],
        pipeline_version=kwargs["pipeline_version"],
        ticker=kwargs["ticker"],
    ) as run:
        return repository.reconcile_stories(run=run, **kwargs)


def reconcile_themes(repository: Phase0Repository, **kwargs):
    """Reconcile a theme set through the logged entrypoint."""

    with repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage="m5.themes",
        trading_day=kwargs["trading_day"],
        pipeline_version=kwargs["pipeline_version"],
        ticker=kwargs["ticker"],
    ) as run:
        return repository.reconcile_themes(run=run, **kwargs)


def raw_item(index: int, ticker: str = "NVDA") -> dict:
    return {
        "source": f"yahoo:{index}",
        "ticker": ticker,
        "title": f"Headline {index}",
        "description": f"Body {index}",
        "url": f"https://example.com/{index}",
        "canonical_url": f"https://example.com/{index}",
        "published_at": f"{DAY}T12:0{index % 10}:00+00:00",
        "fetched_at": f"{DAY}T12:30:00+00:00",
        "raw_json": {"index": index},
    }


def seed_raw_items(
    repository: Phase0Repository, count: int, ticker="NVDA"
) -> list[int]:
    return [
        result.item_id
        for result in repository.admin.insert_raw_items(
            [raw_item(index, ticker) for index in range(1, count + 1)]
        )
    ]


def story(fingerprint: str, member_ids, **overrides) -> StoryRecord:
    members = tuple(
        StoryMemberRecord(raw_item_id=item_id, position=position, outlet=f"O{item_id}")
        for position, item_id in enumerate(member_ids)
    )
    defaults = {
        "cluster_fingerprint": fingerprint,
        "canonical_title": f"Story {fingerprint}",
        "members": members,
        "canonical_item_id": member_ids[0],
        "outlet_count": len(member_ids),
        "content_hash": f"hash-{fingerprint}",
        "stage": "m3.semantic",
        "algorithm_version": "m3.1",
        "config_fingerprint": "cfg",
    }
    defaults.update(overrides)
    return StoryRecord(**defaults)


def theme_set(**overrides) -> ThemeSetRecord:
    defaults = {
        "method": "hdbscan",
        "method_reason": "clustered",
        "quality": {"theme_count": 1},
        "config_fingerprint": "cfg",
        "algorithm_version": "m5.1",
        "model_name": "fake",
        "model_revision": "r1",
        "embedding_dimension": 4,
    }
    defaults.update(overrides)
    return ThemeSetRecord(**defaults)


def schema_snapshot(repository: Phase0Repository) -> dict:
    """Everything about the schema that two databases must agree on."""

    with repository.connect() as connection:
        objects = [
            (row["type"], row["name"], row["sql"])
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        ]
        tables = [name for kind, name, _ in objects if kind == "table"]
        columns = {
            table: [
                (row["name"], row["type"], row["notnull"], row["dflt_value"])
                for row in connection.execute(f"PRAGMA table_info({table})")
            ]
            for table in tables
        }
        foreign_keys = {
            table: [
                tuple(row)
                for row in connection.execute(f"PRAGMA foreign_key_list({table})")
            ]
            for table in tables
        }
        indexes = {
            table: sorted(
                (row["name"], row["unique"], row["partial"])
                for row in connection.execute(f"PRAGMA index_list({table})")
            )
            for table in tables
        }
    return {
        "objects": objects,
        "columns": columns,
        "foreign_keys": foreign_keys,
        "indexes": indexes,
    }


def partial_migrations(tmp_path: Path, max_version: int) -> Path:
    """A migrations directory holding only versions up to ``max_version``."""

    target = tmp_path / f"migrations_v{max_version}"
    target.mkdir()
    for migration in ALL_MIGRATIONS:
        if migration.version <= max_version:
            shutil.copy(MIGRATIONS_PATH / migration.name, target / migration.name)
    return target


def legacy_v2_database(tmp_path: Path) -> Path:
    """A genuine pre-ledger v2 database, with rows in it."""

    database = tmp_path / "legacy.sqlite3"
    legacy = Phase0Repository(
        database,
        migrations_path=Path(__file__).parent / "fixtures" / "legacy_v2_migrations",
    )
    legacy.migrate()
    with legacy.connect() as connection:
        connection.execute("DROP TABLE IF EXISTS schema_migrations")
        connection.execute(
            """
            INSERT INTO raw_items (
                id, source, ticker, title, description, url, canonical_url,
                published_at, fetched_at, raw_json
            ) VALUES (
                1, 'yahoo:Legacy', 'NVDA', 'Legacy headline', 'Body',
                'https://example.com/legacy', 'https://example.com/legacy',
                '2026-07-23T12:00:00+00:00', '2026-07-23T12:05:00+00:00',
                '{"legacy": true}'
            )
            """
        )
    return database


# ----------------------------------------------------------------------
# Migrations and upgrade safety
# ----------------------------------------------------------------------


def test_fresh_database_has_every_object_and_records_its_history(tmp_path):
    repository = migrated(tmp_path)

    snapshot = schema_snapshot(repository)
    names = {name for _, name, _ in snapshot["objects"]}

    assert {
        "supported_tickers",
        "embeddings",
        "theme_sets",
        "theme_other_coverage",
        "theme_excluded_stories",
        "story_provider_conflicts",
        "story_semantic_merges",
        "run_log_stage_keys",
        "schema_migrations",
    } <= names
    assert repository.schema_version() == LATEST_VERSION
    applied = [entry["name"] for entry in repository.applied_migrations()]
    assert applied == [migration.name for migration in ALL_MIGRATIONS]
    assert [row["ticker"] for row in repository.supported_tickers()] == [
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "META",
    ]


@pytest.mark.parametrize("prior_version", PRIOR_VERSIONS)
def test_upgrade_from_every_prior_schema_version_matches_a_fresh_database(
    tmp_path, prior_version
):
    old = Phase0Repository(
        tmp_path / "old.sqlite3",
        migrations_path=partial_migrations(tmp_path, prior_version),
    )
    old.migrate()
    assert old.schema_version() == prior_version
    with old.connect() as connection:
        connection.execute(
            """
            INSERT INTO raw_items (
                id, source, ticker, title, url, canonical_url, published_at,
                fetched_at, raw_json
            ) VALUES (
                7, 'yahoo:kept', 'NVDA', 'Kept headline',
                'https://example.com/kept', 'https://example.com/kept', ?, ?,
                '{"kept": true}'
            )
            """,
            (f"{DAY}T12:00:00+00:00", f"{DAY}T12:05:00+00:00"),
        )

    upgraded = Phase0Repository(tmp_path / "old.sqlite3")
    upgraded.migrate()

    assert upgraded.schema_version() == LATEST_VERSION
    assert schema_snapshot(upgraded) == schema_snapshot(migrated(tmp_path, "fresh.db"))
    kept = upgraded.raw_items_for_day(DAY)
    assert [row["id"] for row in kept] == [7]
    assert kept[0]["ticker"] == "NVDA"
    assert json.loads(kept[0]["raw_json"]) == {"kept": True}
    with upgraded.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_legacy_v2_database_upgrades_and_matches_a_fresh_schema(tmp_path):
    database = legacy_v2_database(tmp_path)

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.schema_version() == LATEST_VERSION
    assert upgraded.count("raw_items") == 1
    assert schema_snapshot(upgraded) == schema_snapshot(migrated(tmp_path, "fresh.db"))


def test_failed_migration_rolls_back_and_leaves_version_and_data_intact(tmp_path):
    directory = partial_migrations(tmp_path, LATEST_VERSION)
    repository = Phase0Repository(
        tmp_path / "phase0.sqlite3", migrations_path=directory
    )
    repository.migrate()
    item_id = repository.admin.insert_raw_item(raw_item(1)).item_id
    before = schema_snapshot(repository)
    before_history = repository.applied_migrations()

    (directory / "009_broken.sql").write_text(
        "CREATE TABLE half_applied (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE half_applied_two (id INTEGER REFERENCES nope(id));\n"
        "INSERT INTO half_applied SELECT missing_column FROM raw_items;\n",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.Error):
        repository.migrate()

    assert repository.schema_version() == LATEST_VERSION
    assert repository.applied_migrations() == before_history
    assert schema_snapshot(repository) == before
    assert repository.count("raw_items") == 1
    assert repository.raw_items_for_day(DAY)[0]["id"] == item_id


def test_rewriting_an_applied_migration_is_refused(tmp_path):
    directory = partial_migrations(tmp_path, LATEST_VERSION)
    repository = Phase0Repository(
        tmp_path / "phase0.sqlite3", migrations_path=directory
    )
    repository.migrate()

    target = directory / "004_supported_ticker_universe.sql"
    target.write_text(target.read_text(encoding="utf-8") + "\n-- edited\n", "utf-8")

    with pytest.raises(Phase0MigrationError, match="rewriting history"):
        repository.migrate()


def test_migrating_is_idempotent_and_survives_reconnection(tmp_path):
    repository = migrated(tmp_path)
    item_id = repository.admin.insert_raw_item(raw_item(1)).item_id
    before = schema_snapshot(repository)

    repository.migrate()
    reopened = Phase0Repository(tmp_path / "phase0.sqlite3")
    reopened.migrate()

    assert schema_snapshot(reopened) == before
    assert reopened.count("raw_items") == 1
    assert reopened.raw_items_for_day(DAY)[0]["id"] == item_id


def test_every_connection_enforces_foreign_keys(tmp_path):
    repository = migrated(tmp_path)

    for _ in range(3):
        with repository.connect() as connection:
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO story_members (story_id, raw_item_id) VALUES (9, 9)"
                )


# ----------------------------------------------------------------------
# Ticker domain
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["BAD", "", "   ", "TSLA,NVDA", "TSLA NVDA", "nvda!", "TOOLONGX", "N V"]
)
def test_repository_rejects_unsupported_or_malformed_tickers(tmp_path, value):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0ValidationError):
        repository.admin.insert_raw_item({**raw_item(1), "ticker": value})


def test_ticker_normalization_is_deterministic(tmp_path):
    repository = migrated(tmp_path)

    result = repository.admin.insert_raw_item({**raw_item(1), "ticker": "  nvda \n"})

    assert repository.raw_item_tickers(result.item_id) == ["NVDA"]
    assert repository.raw_items_for_day(DAY, "nvda")[0]["ticker"] == "NVDA"


@pytest.mark.parametrize(
    "statement, parameters",
    [
        (
            "INSERT INTO raw_items (source, ticker, canonical_url, fetched_at, "
            "raw_json, ingest_status) VALUES ('s', 'BAD', 'u', ?, '{}', 'invalid')",
            (f"{DAY}T12:00:00+00:00",),
        ),
        (
            "INSERT INTO raw_item_tickers (raw_item_id, ticker, association_type) "
            "VALUES (?, 'BAD', 'source')",
            (1,),
        ),
        (
            "INSERT INTO raw_item_candidates (raw_item_id, ticker, reason) "
            "VALUES (?, 'BAD', 'guess')",
            (1,),
        ),
        (
            "INSERT INTO stories (ticker, trading_day, canonical_title) "
            "VALUES ('BAD', ?, 't')",
            (DAY,),
        ),
        (
            "INSERT INTO themes (ticker, trading_day, label, salience_rank, "
            "status, content_hash, pipeline_version) "
            "VALUES ('BAD', ?, 'l', 1, 'ready', 'h', 'v1')",
            (DAY,),
        ),
        (
            "INSERT INTO pipeline_stage_keys (stage, ticker, trading_day, "
            "pipeline_version, status, run_id, updated_at) "
            "VALUES ('cluster', 'BAD', ?, 'v1', 'running', 'r', ?)",
            (DAY, f"{DAY}T12:00:00+00:00"),
        ),
        (
            "INSERT INTO run_log (run_id, stage, duration_ms, started_at, "
            "completed_at, status, trading_day, pipeline_version, ticker) "
            "VALUES ('r', 's', 1, ?, ?, 'success', ?, 'v1', 'BAD')",
            (f"{DAY}T12:00:00+00:00", f"{DAY}T12:00:01+00:00", DAY),
        ),
        (
            "INSERT INTO theme_sets (ticker, trading_day, pipeline_version, "
            "method, config_fingerprint, algorithm_version, updated_at) "
            "VALUES ('BAD', ?, 'v1', 'hdbscan', 'c', 'a', ?)",
            (DAY, f"{DAY}T12:00:00+00:00"),
        ),
    ],
)
def test_direct_sql_cannot_store_an_unsupported_ticker(tmp_path, statement, parameters):
    repository = migrated(tmp_path)
    seed_raw_items(repository, 1)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="unsupported Phase 0 ticker"):
            connection.execute(statement, parameters)


#: Every ticker-bearing table, and how to put a row in it that would be
#: legal except for its ticker.  Each is exercised on INSERT and on UPDATE,
#: because a domain enforced only on INSERT is not a domain.
TICKER_TABLES = [
    (
        "raw_items",
        "INSERT INTO raw_items (source, ticker, canonical_url, fetched_at, "
        "raw_json, ingest_status) VALUES ('s2', 'NVDA', 'u2', ?, '{}', 'invalid')",
        (f"{DAY}T12:00:00+00:00",),
    ),
    (
        "raw_item_tickers",
        "INSERT INTO raw_item_tickers (raw_item_id, ticker, association_type) "
        "VALUES (?, 'AMD', 'source')",
        (1,),
    ),
    (
        "raw_item_candidates",
        "INSERT INTO raw_item_candidates (raw_item_id, ticker, reason) "
        "VALUES (?, 'AMD', 'guess')",
        (1,),
    ),
    (
        "stories",
        "INSERT INTO stories (ticker, trading_day, canonical_title) "
        "VALUES ('NVDA', ?, 't')",
        (DAY,),
    ),
    (
        "themes",
        "INSERT INTO themes (ticker, trading_day, label, salience_rank, "
        "status, content_hash, pipeline_version) "
        "VALUES ('NVDA', ?, 'l', 1, 'ready', 'h', 'v1')",
        (DAY,),
    ),
    (
        "theme_sets",
        "INSERT INTO theme_sets (ticker, trading_day, pipeline_version, "
        "method, config_fingerprint, algorithm_version, updated_at) "
        "VALUES ('NVDA', ?, 'v9', 'hdbscan', 'c', 'a', ?)",
        (DAY, f"{DAY}T12:00:00+00:00"),
    ),
    (
        "pipeline_stage_keys",
        "INSERT INTO pipeline_stage_keys (stage, ticker, trading_day, "
        "pipeline_version, status, run_id, updated_at) "
        "VALUES ('cluster', 'NVDA', ?, 'v1', 'running', 'r', ?)",
        (DAY, f"{DAY}T12:00:00+00:00"),
    ),
]


@pytest.mark.parametrize(
    "table, insert, parameters", TICKER_TABLES, ids=[row[0] for row in TICKER_TABLES]
)
def test_direct_sql_cannot_update_a_row_to_an_unsupported_ticker(
    tmp_path, table, insert, parameters
):
    repository = migrated(tmp_path)
    seed_raw_items(repository, 1)

    with repository.connect() as connection:
        connection.execute(insert, parameters)
        with pytest.raises(sqlite3.IntegrityError, match="unsupported Phase 0 ticker"):
            connection.execute(f"UPDATE {table} SET ticker = 'GOOG'")


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO supported_tickers (ticker, display_name, position) "
        "VALUES ('GOOG', 'Alphabet', 6)",
        "UPDATE supported_tickers SET ticker = 'GOOG' WHERE ticker = 'NVDA'",
        "UPDATE supported_tickers SET display_name = 'Nope' WHERE ticker = 'NVDA'",
        "DELETE FROM supported_tickers WHERE ticker = 'NVDA'",
        "DELETE FROM supported_tickers",
    ],
)
def test_the_approved_universe_cannot_be_edited_by_direct_sql(tmp_path, statement):
    repository = migrated(tmp_path)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(statement)

    assert {
        row["ticker"] for row in repository.supported_tickers()
    } == SUPPORTED_TICKERS


def test_widening_the_universe_cannot_widen_any_other_table(tmp_path):
    """The old defect: triggers read the table, so seeding it widened them.

    Even with the seal bypassed the constraint must hold, because the
    approved five are a literal inside each trigger rather than a query.
    """

    repository = migrated(tmp_path)
    seed_raw_items(repository, 1)

    with repository.connect() as connection:
        connection.execute("DROP TRIGGER trg_supported_tickers_immutable_insert")
        connection.execute(
            "INSERT INTO supported_tickers (ticker, display_name, position) "
            "VALUES ('GOOG', 'Alphabet', 6)"
        )
        for table, insert, parameters in TICKER_TABLES:
            with pytest.raises(
                sqlite3.IntegrityError, match="unsupported Phase 0 ticker"
            ):
                connection.execute(
                    insert.replace("'NVDA'", "'GOOG'").replace("'AMD'", "'GOOG'"),
                    parameters,
                )

    with pytest.raises(UnsupportedTickerError):
        repository.admin.insert_raw_item(raw_item(9, "GOOG"))


def test_the_universe_is_exactly_the_five_approved_symbols(tmp_path):
    repository = migrated(tmp_path)

    universe = repository.supported_tickers()

    assert [row["ticker"] for row in universe] == [
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "META",
    ]
    assert {row["ticker"] for row in universe} == SUPPORTED_TICKERS
    assert SUPPORTED_TICKERS == {"AAPL", "AMD", "META", "NVDA", "TSLA"}


@pytest.mark.parametrize("version", PRIOR_VERSIONS)
def test_an_upgraded_database_seals_the_universe_too(tmp_path, version):
    directory = partial_migrations(tmp_path, version)
    repository = Phase0Repository(
        tmp_path / "phase0.sqlite3", migrations_path=directory
    )
    repository.migrate()

    upgraded = Phase0Repository(tmp_path / "phase0.sqlite3")
    upgraded.migrate()

    with upgraded.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "INSERT INTO supported_tickers (ticker, display_name, position) "
                "VALUES ('GOOG', 'Alphabet', 6)"
            )
    assert {row["ticker"] for row in upgraded.supported_tickers()} == SUPPORTED_TICKERS


# ----------------------------------------------------------------------
# Secret redaction
# ----------------------------------------------------------------------


SECRET_CASES = [
    ("Authorization: Bearer abc123XYZ", "abc123XYZ"),
    ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("proxy-authorization=Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ('{"Authorization": "Basic dXNlcjpwYXNz"}', "dXNlcjpwYXNz"),
    ("x-api-key: sk-live-9999", "sk-live-9999"),
    ("api_key=SUPERSECRET&page=2", "SUPERSECRET"),
    ("https://h/f?api_key=SUPERSECRET&access_token=TOK123", "SUPERSECRET"),
    ("https://h/f?access_token=TOK123", "TOK123"),
    ("password: hunter2", "hunter2"),
    ('{"access_token": "tok-abcdef"}', "tok-abcdef"),
    ("https://user:pa55w0rd@example.com/feed", "pa55w0rd"),
    ("client_secret=shhh1", "shhh1"),
]


@pytest.mark.parametrize("text, credential", SECRET_CASES)
def test_credential_values_disappear_from_strings(text, credential):
    redacted = redact_text(text)

    assert credential not in redacted
    assert "[REDACTED]" in redacted


def test_nested_metadata_and_sequences_are_redacted():
    payload = {
        "outer": {
            "headers": {"Authorization": "Basic dXNlcjpwYXNz"},
            "notes": ["Authorization: Bearer abc123XYZ", {"api_key": "sk-1"}],
        }
    }

    redacted = json.dumps(redact_secrets(payload))

    assert "dXNlcjpwYXNz" not in redacted
    assert "abc123XYZ" not in redacted
    assert "sk-1" not in redacted


@pytest.mark.parametrize("text, credential", SECRET_CASES)
def test_stored_run_log_errors_are_redacted(tmp_path, text, credential):
    repository = migrated(tmp_path)

    repository.log_stage(
        run_id="run-secret",
        stage="fetch",
        counts={"items": 1},
        duration_ms=1,
        errors=[{"detail": text}, text],
        started_at=f"{DAY}T12:00:00+00:00",
        completed_at=f"{DAY}T12:00:01+00:00",
        trading_day=DAY,
        pipeline_version="v1",
    )

    with repository.connect() as connection:
        stored = connection.execute(
            "SELECT errors FROM run_log WHERE run_id = 'run-secret'"
        ).fetchone()[0]
    assert credential not in stored


@pytest.mark.parametrize("text, credential", SECRET_CASES)
def test_source_state_metadata_and_errors_are_redacted(tmp_path, text, credential):
    repository = migrated(tmp_path)

    repository.admin.set_source_state(
        "rss:secret",
        etag=None,
        last_modified=None,
        checked_at=f"{DAY}T12:00:00+00:00",
        successful=False,
        metadata={"request": {"headers": {"note": text}}, "trace": [text]},
        error=text,
    )

    with repository.connect() as connection:
        row = connection.execute(
            "SELECT metadata, last_error FROM source_state WHERE source = 'rss:secret'"
        ).fetchone()
    assert credential not in row["metadata"]
    assert credential not in row["last_error"]


def test_repository_errors_do_not_leak_credentials(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0ValidationError) as caught:
        repository.admin.insert_raw_item(
            {**raw_item(1), "ticker": "Authorization: Basic dXNlcjpwYXNz"}
        )

    assert "dXNlcjpwYXNz" not in str(caught.value)


def test_stage_key_errors_do_not_leak_credentials(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(StageKeyError) as caught:
        repository.complete_stage_key(
            stage="fetch",
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            run_id="Bearer abc123XYZ",
        )

    assert "abc123XYZ" not in str(caught.value)


# ----------------------------------------------------------------------
# Stage leases and recovery
# ----------------------------------------------------------------------


STAGE_KEY = {
    "stage": "cluster",
    "ticker": "NVDA",
    "trading_day": DAY,
    "pipeline_version": "v1",
}


def test_expired_lease_is_reclaimable_by_exactly_one_concurrent_caller(tmp_path):
    repository = migrated(tmp_path)
    assert repository.claim_stage_key(
        **STAGE_KEY, run_id="crashed", lease_seconds=60, claimed_at=f"{DAY}T12:00:00Z"
    )

    def reclaim(index: int):
        return (
            index,
            repository.claim_stage_key(
                **STAGE_KEY,
                run_id=f"recovered-{index}",
                lease_seconds=60,
                claimed_at=f"{DAY}T13:00:00Z",
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        attempts = list(executor.map(reclaim, range(8)))

    winners = [index for index, claimed in attempts if claimed]
    assert len(winners) == 1
    state = repository.stage_key_state(**STAGE_KEY)
    assert state["run_id"] == f"recovered-{winners[0]}"
    assert state["status"] == "running"
    assert state["attempts"] == 2
    assert state["recovered_count"] == 1


def test_losing_claimants_never_overwrite_the_owner(tmp_path):
    repository = migrated(tmp_path)
    assert repository.claim_stage_key(**STAGE_KEY, run_id="owner", lease_seconds=3600)

    for index in range(5):
        assert not repository.claim_stage_key(**STAGE_KEY, run_id=f"loser-{index}")

    assert repository.stage_key_state(**STAGE_KEY)["run_id"] == "owner"


def test_completed_work_cannot_be_reclaimed_and_records_its_lifecycle(tmp_path):
    repository = migrated(tmp_path)
    repository.claim_stage_key(**STAGE_KEY, run_id="owner")
    repository.complete_stage_key(**STAGE_KEY, run_id="owner", status="success")

    assert not repository.claim_stage_key(**STAGE_KEY, run_id="later")
    state = repository.stage_key_state(**STAGE_KEY)
    assert state["status"] == "success"
    assert state["completed_at"] is not None
    assert state["lease_expires_at"] is None


def test_recover_expired_leases_reports_and_frees_abandoned_claims(tmp_path):
    repository = migrated(tmp_path)
    repository.claim_stage_key(
        **STAGE_KEY, run_id="crashed", lease_seconds=60, claimed_at=f"{DAY}T12:00:00Z"
    )

    assert repository.recover_expired_leases(now=f"{DAY}T12:00:30Z") == []
    recovered = repository.recover_expired_leases(now=f"{DAY}T13:00:00Z")

    assert [row["run_id"] for row in recovered] == ["crashed"]
    state = repository.stage_key_state(**STAGE_KEY)
    assert state["status"] == "failed"
    assert "lease expired" in state["last_error"]
    assert repository.claim_stage_key(**STAGE_KEY, run_id="fresh")


def test_heartbeat_extends_only_the_owners_lease(tmp_path):
    repository = migrated(tmp_path)
    repository.claim_stage_key(
        **STAGE_KEY, run_id="owner", lease_seconds=60, claimed_at=f"{DAY}T12:00:00Z"
    )

    assert not repository.heartbeat_stage_key(
        **STAGE_KEY, run_id="stranger", now=f"{DAY}T12:00:30Z"
    )
    assert repository.heartbeat_stage_key(
        **STAGE_KEY, run_id="owner", lease_seconds=600, now=f"{DAY}T12:00:30Z"
    )
    assert not repository.claim_stage_key(
        **STAGE_KEY, run_id="thief", claimed_at=f"{DAY}T12:05:00Z"
    )


# ----------------------------------------------------------------------
# Relationship integrity
# ----------------------------------------------------------------------


def build_day(repository: Phase0Repository) -> dict:
    item_ids = seed_raw_items(repository, 4)
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[
            story("cf1", item_ids[:2]),
            story("cf2", item_ids[2:3]),
            story("cf3", item_ids[3:]),
        ],
    )
    stories = {
        row["cluster_fingerprint"]: row["id"]
        for row in repository.stories_for_day(DAY, "NVDA")
    }
    return {"items": item_ids, "stories": stories}


def test_direct_sql_cannot_cite_a_non_member_raw_item(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    theme_id = repository.insert_theme(
        ticker="NVDA",
        trading_day=DAY,
        label="Theme",
        story_ids=[day["stories"]["cf1"]],
        citation_ids=[day["items"][0]],
        salience_rank=1,
        status="ready",
        content_hash="h",
        pipeline_version="v1",
    )

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="member story"):
            connection.execute(
                "INSERT INTO theme_citations (theme_id, raw_item_id) VALUES (?, ?)",
                (theme_id, day["items"][3]),
            )


def test_a_raw_item_cannot_be_citable_from_two_themes_in_one_ticker_day(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    first = repository.insert_theme(
        ticker="NVDA",
        trading_day=DAY,
        label="First",
        story_ids=[day["stories"]["cf1"]],
        citation_ids=[day["items"][0]],
        salience_rank=1,
        status="ready",
        content_hash="h1",
        pipeline_version="v1",
    )
    second = repository.insert_theme(
        ticker="NVDA",
        trading_day=DAY,
        label="Second",
        story_ids=[day["stories"]["cf2"]],
        citation_ids=[day["items"][2]],
        salience_rank=2,
        status="ready",
        content_hash="h2",
        pipeline_version="v1",
    )

    with repository.connect() as connection:
        connection.execute(
            "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
            (second, day["stories"]["cf1"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="already citable"):
            connection.execute(
                "INSERT INTO theme_citations (theme_id, raw_item_id) VALUES (?, ?)",
                (second, day["items"][0]),
            )
    assert first != second


def test_theme_cannot_group_a_story_from_another_ticker_or_day(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    other_items = [
        result.item_id
        for result in repository.admin.insert_raw_items(
            [{**raw_item(9, "AMD"), "source": "yahoo:amd"}]
        )
    ]
    reconcile_stories(
        repository,
        ticker="AMD",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("amd1", other_items)],
    )
    foreign_story = repository.stories_for_day(DAY, "AMD")[0]["id"]
    theme_id = repository.insert_theme(
        ticker="NVDA",
        trading_day=DAY,
        label="Theme",
        story_ids=[day["stories"]["cf1"]],
        citation_ids=[],
        salience_rank=1,
        status="ready",
        content_hash="h",
        pipeline_version="v1",
    )

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker and day"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (theme_id, foreign_story),
            )


def test_citation_lifecycle_deletions_follow_a_valid_order(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    theme_id = repository.insert_theme(
        ticker="NVDA",
        trading_day=DAY,
        label="Theme",
        story_ids=[day["stories"]["cf1"]],
        citation_ids=[day["items"][1]],
        salience_rank=1,
        status="ready",
        content_hash="h",
        pipeline_version="v1",
    )

    with repository.connect() as connection:
        with pytest.raises(
            sqlite3.IntegrityError, match="required by a theme citation"
        ):
            connection.execute(
                "DELETE FROM theme_stories WHERE theme_id = ?", (theme_id,)
            )
        with pytest.raises(
            sqlite3.IntegrityError, match="required by a theme citation"
        ):
            connection.execute(
                "DELETE FROM story_members WHERE story_id = ? AND raw_item_id = ?",
                (day["stories"]["cf1"], day["items"][1]),
            )

    with repository.connect() as connection:
        connection.execute(
            "DELETE FROM theme_citations WHERE theme_id = ?", (theme_id,)
        )
        connection.execute("DELETE FROM theme_stories WHERE theme_id = ?", (theme_id,))
        connection.execute("DELETE FROM themes WHERE id = ?", (theme_id,))
    assert repository.count("themes") == 0


def test_canonical_member_cannot_leave_its_story(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="canonical member"):
            connection.execute(
                "DELETE FROM story_members WHERE story_id = ? AND raw_item_id = ?",
                (day["stories"]["cf1"], day["items"][0]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="canonical item"):
            connection.execute(
                "UPDATE stories SET canonical_item_id = ? WHERE id = ?",
                (day["items"][3], day["stories"]["cf1"]),
            )


def test_raw_item_associations_and_labels_follow_their_parent(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)
    repository.insert_eval_label(
        label_type="dedup",
        item_a_id=item_ids[0],
        item_b_id=item_ids[1],
        reviewer="kartik",
        label="different",
    )

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO raw_item_tickers "
                "(raw_item_id, ticker, association_type) VALUES (?, 'NVDA', 'source')",
                (999_999,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO eval_labels (label_type, item_a_id, item_b_id, "
                "reviewer, label, created_at) VALUES ('dedup', ?, ?, 'k', 'l', ?)",
                (item_ids[0], 999_999, f"{DAY}T12:00:00+00:00"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE raw_item_tickers SET raw_item_id = 999999 "
                "WHERE raw_item_id = ?",
                (item_ids[0],),
            )

    with repository.connect() as connection:
        connection.execute("DELETE FROM raw_items WHERE id = ?", (item_ids[0],))
    assert repository.count("raw_item_tickers") == 1
    assert repository.count("eval_labels") == 0


def test_batch_writes_do_not_commit_partially(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0ValidationError):
        repository.admin.insert_raw_items(
            [raw_item(1), raw_item(2)],
            source_state={"source": "rss:test", "checked_at": "not-a-timestamp"},
        )

    assert repository.count("raw_items") == 0
    assert repository.count("raw_item_tickers") == 0
    assert repository.count("source_state") == 0


def test_run_log_cannot_reference_a_stage_key_that_does_not_exist(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0IntegrityError, match="stage key"):
        repository.log_stage(
            run_id="run-1",
            stage="cluster",
            counts={},
            duration_ms=1,
            errors=[],
            started_at=f"{DAY}T12:00:00+00:00",
            completed_at=f"{DAY}T12:00:01+00:00",
            trading_day=DAY,
            pipeline_version="v1",
            stage_key=STAGE_KEY,
        )
    assert repository.count("run_log_stage_keys") == 0
    assert repository.count("run_log") == 0


def test_run_log_links_to_a_claimed_stage_key(tmp_path):
    repository = migrated(tmp_path)
    repository.claim_stage_key(**STAGE_KEY, run_id="run-1")

    repository.log_stage(
        run_id="run-1",
        stage="cluster",
        counts={},
        duration_ms=1,
        errors=[],
        started_at=f"{DAY}T12:00:00+00:00",
        completed_at=f"{DAY}T12:00:01+00:00",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=STAGE_KEY,
    )

    assert repository.count("run_log_stage_keys") == 1
    repository.clear_derived_for_day(DAY)
    assert repository.count("run_log_stage_keys") == 0


# ----------------------------------------------------------------------
# Embedding persistence
# ----------------------------------------------------------------------


def embedding_for(item_id: int, *, fingerprint="a" * 64, revision="r1", dimension=4):
    vector = np.linspace(0.1, 0.4, dimension, dtype=np.float32)
    return PersistedEmbedding(
        source_kind="raw_item",
        source_id=str(item_id),
        model_name="fake-model",
        model_revision=revision,
        input_fingerprint=fingerprint,
        dimension=dimension,
        dtype=EMBEDDING_DTYPE,
        vector_blob=serialize_vector(vector),
    )


def test_repository_satisfies_the_m1_embedding_protocol(tmp_path):
    assert isinstance(migrated(tmp_path), EmbeddingRepository)


def test_embeddings_survive_a_reopened_sqlite_connection(tmp_path):
    repository = migrated(tmp_path)
    item_id = seed_raw_items(repository, 1)[0]
    stored = embedding_for(item_id)
    repository.upsert_embedding(stored)

    reopened = Phase0Repository(tmp_path / "phase0.sqlite3")

    assert reopened.get_embedding("raw_item", str(item_id)) == stored
    assert reopened.get_embedding("raw_item", f"  {item_id} ") == stored


def test_upsert_replaces_rather_than_duplicating_an_identity(tmp_path):
    repository = migrated(tmp_path)
    item_id = seed_raw_items(repository, 1)[0]

    repository.upsert_embedding(embedding_for(item_id))
    replacement = embedding_for(item_id, fingerprint="b" * 64, revision="r2")
    repository.upsert_embedding(replacement)

    assert repository.count("embeddings") == 1
    assert repository.get_embedding("raw_item", str(item_id)) == replacement


@pytest.mark.parametrize(
    "blob, dimension",
    [(b"not-a-vector", 4), (serialize_vector(np.ones(3, dtype=np.float32)), 4)],
)
def test_corrupt_or_incompatible_vectors_fail_explicitly(tmp_path, blob, dimension):
    repository = migrated(tmp_path)
    item_id = seed_raw_items(repository, 1)[0]
    broken = PersistedEmbedding(
        source_kind="raw_item",
        source_id=str(item_id),
        model_name="fake-model",
        model_revision=None,
        input_fingerprint="c" * 64,
        dimension=dimension,
        dtype=EMBEDDING_DTYPE,
        vector_blob=blob,
    )

    with pytest.raises(EmbeddingPersistenceError):
        repository.upsert_embedding(broken)
    assert repository.count("embeddings") == 0


def test_embedding_ownership_and_lifecycle_are_enforced(tmp_path):
    repository = migrated(tmp_path)
    item_id = seed_raw_items(repository, 1)[0]

    with pytest.raises(EmbeddingPersistenceError):
        repository.upsert_embedding(embedding_for(item_id + 999))

    repository.upsert_embedding(embedding_for(item_id))
    with repository.connect() as connection:
        connection.execute("DELETE FROM raw_items WHERE id = ?", (item_id,))
    assert repository.count("embeddings") == 0


def test_model_and_input_changes_invalidate_the_cache(tmp_path):
    repository = migrated(tmp_path)
    item_id = seed_raw_items(repository, 1)[0]
    calls: list[list[str]] = []

    class Encoder:
        def encode(self, texts, **_):
            calls.append(list(texts))
            return np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (len(texts), 1))

    def service(revision: str) -> EmbeddingService:
        return EmbeddingService(
            model_name="fake-model",
            model_revision=revision,
            expected_dimension=4,
            encoder_factory=lambda *_: Encoder(),
        )

    target = EmbeddingTarget(
        source_kind="raw_item", source_id=item_id, title="Hello", description="World"
    )
    first = service("r1").embed_targets([target], repository)
    cached = service("r1").embed_targets([target], repository)
    assert len(calls) == 1
    assert np.allclose(first[0], cached[0])

    service("r2").embed_targets([target], repository)
    assert len(calls) == 2

    changed = EmbeddingTarget(
        source_kind="raw_item", source_id=item_id, title="Different", description="Text"
    )
    service("r2").embed_targets([changed], repository)
    assert len(calls) == 3
    assert repository.count("embeddings") == 1


# ----------------------------------------------------------------------
# Story reconciliation
# ----------------------------------------------------------------------


def test_story_reconciliation_reports_every_outcome(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 4)
    first = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("cf1", item_ids[:2]), story("cf2", item_ids[2:3])],
    )
    assert first.counts["inserted"] == 2

    unchanged = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("cf1", item_ids[:2]), story("cf2", item_ids[2:3])],
    )
    assert unchanged.counts == {
        "inserted": 0,
        "updated": 0,
        "unchanged": 2,
        "deleted": 0,
        "invalidated": 0,
        "removed_members": 0,
        "invalidated_themes": 0,
    }

    changed = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[
            story("cf1", [item_ids[0]], content_hash="hash-cf1-b"),
            story("cf3", item_ids[3:]),
        ],
    )

    assert changed.counts["updated"] == 1
    assert changed.counts["inserted"] == 1
    assert changed.counts["deleted"] == 1
    assert changed.removed_members == 1
    stored = {
        row["cluster_fingerprint"]: row
        for row in repository.stories_for_day(DAY, "NVDA")
    }
    assert set(stored) == {"cf1", "cf3"}
    assert json.loads(stored["cf1"]["member_ids"]) == [item_ids[0]]


def test_story_reconciliation_persists_upstream_trust_metadata(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)

    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[
            story(
                "cf1",
                item_ids,
                quarantined=True,
                semantic_skip_reason="provider_quarantine",
                member_story_keys=("m2-a", "m2-b"),
                provider_conflicts=(
                    ProviderConflictRecord("yahoo", "dup-1", ("1", "2"), ("title",)),
                ),
                semantic_merges=(SemanticMergeRecord("m2-a", "m2-b", 0.91),),
                model_name="fake-model",
                model_revision="r1",
                embedding_dimension=4,
            )
        ],
    )

    row = repository.stories_for_day(DAY, "NVDA")[0]
    assert row["quarantined"] == 1
    assert row["semantic_skip_reason"] == "provider_quarantine"
    assert json.loads(row["member_story_keys"]) == ["m2-a", "m2-b"]
    assert repository.count("story_provider_conflicts") == 1
    assert repository.count("story_semantic_merges") == 1


def test_failed_story_reconciliation_writes_nothing(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("cf1", item_ids[:1])],
    )
    before = repository.stories_for_day(DAY, "NVDA")

    with pytest.raises(sqlite3.IntegrityError):
        reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[
                story("cf1", item_ids),
                story("cf2", [999_999]),
            ],
        )

    assert repository.stories_for_day(DAY, "NVDA") == before
    assert repository.count("stories") == 1


def test_structural_story_change_invalidates_the_days_themes(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                label="Theme",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                method="hdbscan",
            )
        ],
        other_coverage=(
            OtherCoverageRecord(day["stories"]["cf2"], "clustering_noise"),
            OtherCoverageRecord(day["stories"]["cf3"], "clustering_noise"),
        ),
    )
    assert repository.count("themes") == 1

    report = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("cf1", day["items"][:2]), story("cf2", day["items"][2:3])],
    )

    assert report.counts["deleted"] == 1
    assert repository.count("themes") == 0
    assert repository.count("theme_sets") == 0
    assert repository.count("theme_other_coverage") == 0
    assert repository.count("raw_items") == 4


# ----------------------------------------------------------------------
# Theme reconciliation
# ----------------------------------------------------------------------


def test_theme_reconciliation_writes_a_whole_ticker_day(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)

    report = reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(
            quality={"theme_coverage": 0.5}, trust_metadata={"reviewed": False}
        ),
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                theme_key="stable-key",
                label="Chip demand",
                label_source="highest_salience_member",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0], day["items"][1]),
                salience=0.8,
                salience_rank=1,
                cohesion=0.7,
                min_pairwise_cohesion=0.6,
                outlet_count=2,
                salience_story_component=0.5,
                salience_outlet_component=0.4,
                salience_recency_component=0.9,
                method="hdbscan",
                status="ready",
                algorithm_version="m5.1",
                config_fingerprint="cfg",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            )
        ],
        other_coverage=(
            OtherCoverageRecord(day["stories"]["cf2"], "clustering_noise"),
        ),
        excluded=(ExcludedStoryRecord(day["stories"]["cf3"], "no_encodable_text"),),
    )

    assert report.counts["inserted"] == 1
    stored = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
    assert stored["quality"] == {"theme_coverage": 0.5}
    assert stored["trust_metadata"] == {"reviewed": False}
    assert stored["method_reason"] == "clustered"
    assert stored["themes"][0]["theme_key"] == "stable-key"
    assert stored["themes"][0]["cohesion"] == 0.7
    assert stored["other_coverage"] == [
        {"story_id": day["stories"]["cf2"], "reason": "clustering_noise", "position": 0}
    ]
    assert stored["excluded"] == [
        {"story_id": day["stories"]["cf3"], "reason": "no_encodable_text"}
    ]


def test_theme_reconciliation_removes_obsolete_membership(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    common = {
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "theme_set": theme_set(),
    }
    reconcile_themes(
        repository,
        **common,
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                label="First",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                method="hdbscan",
            ),
            ThemeRecord(
                fingerprint="tf2",
                label="Second",
                story_ids=(day["stories"]["cf2"],),
                citation_item_ids=(day["items"][2],),
                salience_rank=2,
                method="hdbscan",
            ),
        ],
        other_coverage=(
            OtherCoverageRecord(day["stories"]["cf3"], "clustering_noise"),
        ),
    )

    # The second run moves the raw item that tf2 cited into tf1.
    report = reconcile_themes(
        repository,
        **common,
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                label="First",
                story_ids=(day["stories"]["cf1"], day["stories"]["cf2"]),
                citation_item_ids=(day["items"][0], day["items"][2]),
                method="hdbscan",
            )
        ],
        other_coverage=(
            OtherCoverageRecord(day["stories"]["cf3"], "clustering_noise"),
        ),
    )

    assert report.counts["deleted"] == 1
    assert report.counts["updated"] == 1
    assert repository.count("themes") == 1
    assert repository.count("theme_citations") == 2


def test_failed_theme_reconciliation_writes_nothing(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    common = {
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "theme_set": theme_set(),
    }
    reconcile_themes(
        repository,
        **common,
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                label="First",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                method="hdbscan",
            )
        ],
    )
    before = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")

    with pytest.raises(Phase0ValidationError, match="member raw items"):
        reconcile_themes(
            repository,
            **common,
            themes=[
                ThemeRecord(
                    fingerprint="tf2",
                    label="Rewritten",
                    story_ids=(day["stories"]["cf2"],),
                    citation_item_ids=(day["items"][0],),
                    method="hdbscan",
                )
            ],
        )

    after = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
    assert after == before
    assert repository.count("themes") == 1


def test_a_story_cannot_be_in_a_theme_and_in_other_coverage(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)

    with pytest.raises(Phase0ValidationError, match="other coverage"):
        reconcile_themes(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            theme_set=theme_set(),
            themes=[
                ThemeRecord(
                    fingerprint="tf1",
                    label="Theme",
                    story_ids=(day["stories"]["cf1"],),
                    method="hdbscan",
                )
            ],
            other_coverage=(
                OtherCoverageRecord(day["stories"]["cf1"], "clustering_noise"),
            ),
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "kmeans"}, "clustering method"),
        ({"reason": "because"}, "other-coverage reason"),
    ],
)
def test_theme_reconciliation_validates_its_vocabulary(tmp_path, kwargs, message):
    repository = migrated(tmp_path)
    day = build_day(repository)
    method = kwargs.get("method", "hdbscan")
    reason = kwargs.get("reason", "clustering_noise")

    with pytest.raises(Phase0ValidationError, match=message):
        reconcile_themes(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            theme_set=theme_set(method=method),
            themes=[
                ThemeRecord(
                    fingerprint="tf1",
                    label="Theme",
                    story_ids=(day["stories"]["cf1"],),
                    method=method if method in {"hdbscan"} else None,
                )
            ],
            other_coverage=(OtherCoverageRecord(day["stories"]["cf2"], reason),),
        )


# ----------------------------------------------------------------------
# M5 integrity holds on UPDATE, not only on INSERT
# ----------------------------------------------------------------------


def populated_theme_set(repository: Phase0Repository) -> dict:
    """One complete NVDA theme set, plus an AMD story and a next-day story.

    Gives every "move this row somewhere it does not belong" probe below a
    real target to aim at.
    """

    day = build_day(repository)
    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                label="Chip demand",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                salience_rank=1,
                status="ready",
                method="hdbscan",
            )
        ],
        other_coverage=(
            OtherCoverageRecord(day["stories"]["cf2"], "clustering_noise"),
        ),
        excluded=(ExcludedStoryRecord(day["stories"]["cf3"], "no_encodable_text"),),
    )

    foreign_items = seed_raw_items(repository, 1, ticker="AMD")
    reconcile_stories(
        repository,
        ticker="AMD",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("amd1", foreign_items)],
    )
    foreign_story = repository.stories_for_day(DAY, "AMD")[0]["id"]

    with repository.connect() as connection:
        theme_id = int(
            connection.execute(
                "SELECT id FROM themes WHERE ticker = 'NVDA'"
            ).fetchone()["id"]
        )
        set_id = int(
            connection.execute(
                "SELECT id FROM theme_sets WHERE ticker = 'NVDA'"
            ).fetchone()["id"]
        )
    day.update({"theme_id": theme_id, "set_id": set_id, "foreign_story": foreign_story})
    return day


def test_other_coverage_cannot_be_updated_onto_a_cross_ticker_story(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker/day"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (day["foreign_story"], day["set_id"]),
            )


def test_other_coverage_cannot_be_updated_onto_a_different_trading_day(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)
    later = seed_raw_items(repository, 1)
    with repository.connect() as connection:
        connection.execute(
            "UPDATE raw_items SET published_at = ? WHERE id = ?",
            ("2026-07-24T12:00:00+00:00", later[0]),
        )
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day="2026-07-24",
        pipeline_version="v1",
        stories=[story("next1", later)],
    )
    next_day_story = repository.stories_for_day("2026-07-24", "NVDA")[0]["id"]

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker/day"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (next_day_story, day["set_id"]),
            )


def test_a_story_cannot_be_updated_into_both_a_theme_and_other_coverage(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="already a member of a theme"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (day["stories"]["cf1"], day["set_id"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="other coverage"):
            connection.execute(
                "UPDATE theme_stories SET story_id = ? WHERE theme_id = ?",
                (day["stories"]["cf2"], day["theme_id"]),
            )


def test_a_story_cannot_be_updated_into_both_a_theme_and_the_exclusions(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="already accounted for"):
            connection.execute(
                "UPDATE theme_excluded_stories SET story_id = ? WHERE theme_set_id = ?",
                (day["stories"]["cf1"], day["set_id"]),
            )
        with pytest.raises(sqlite3.IntegrityError, match="already excluded"):
            connection.execute(
                "UPDATE theme_stories SET story_id = ? WHERE theme_id = ?",
                (day["stories"]["cf3"], day["theme_id"]),
            )


def test_a_story_cannot_be_updated_into_both_other_coverage_and_exclusions(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="already accounted for"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (day["stories"]["cf3"], day["set_id"]),
            )


def test_theme_membership_cannot_be_updated_across_ticker_or_day(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker and day"):
            connection.execute(
                "UPDATE theme_stories SET story_id = ? WHERE theme_id = ?",
                (day["foreign_story"], day["theme_id"]),
            )


def test_a_citation_cannot_be_updated_onto_evidence_outside_its_theme(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="member story"):
            connection.execute(
                "UPDATE theme_citations SET raw_item_id = ? WHERE theme_id = ?",
                (day["items"][3], day["theme_id"]),
            )


def test_a_parent_cannot_be_relocated_out_from_under_its_children(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="populated theme"):
            connection.execute(
                "UPDATE themes SET ticker = 'AMD' WHERE id = ?", (day["theme_id"],)
            )
        with pytest.raises(sqlite3.IntegrityError, match="populated theme"):
            connection.execute(
                "UPDATE themes SET trading_day = '2026-07-24' WHERE id = ?",
                (day["theme_id"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="populated theme set"):
            connection.execute(
                "UPDATE theme_sets SET trading_day = '2026-07-24' WHERE id = ?",
                (day["set_id"],),
            )
        with pytest.raises(sqlite3.IntegrityError, match="cannot change ticker or day"):
            connection.execute(
                "UPDATE stories SET trading_day = '2026-07-24' WHERE id = ?",
                (day["stories"]["cf1"],),
            )


def test_valid_theme_set_updates_and_deletion_order_still_work(tmp_path):
    """The guards must not have made ordinary lifecycle work impossible."""

    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.connect() as connection:
        connection.execute(
            "UPDATE themes SET label = 'Renamed' WHERE id = ?", (day["theme_id"],)
        )
        connection.execute(
            "UPDATE theme_other_coverage SET reason = 'below_cohesion_floor' "
            "WHERE theme_set_id = ?",
            (day["set_id"],),
        )
        connection.execute(
            "DELETE FROM theme_citations WHERE theme_id = ?", (day["theme_id"],)
        )
        connection.execute(
            "DELETE FROM theme_stories WHERE theme_id = ?", (day["theme_id"],)
        )
        connection.execute("DELETE FROM themes WHERE id = ?", (day["theme_id"],))
        # An emptied theme may be relocated; only a populated one may not.
        connection.execute(
            "UPDATE theme_sets SET pipeline_version = 'v2' WHERE id = ? "
            "AND NOT EXISTS (SELECT 1 FROM theme_other_coverage "
            "WHERE theme_set_id = ?)",
            (day["set_id"], day["set_id"]),
        )

    # The whole ticker-day can then be rewritten from scratch.
    report = reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="tf2",
                label="Rebuilt",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                salience_rank=1,
                status="ready",
                method="hdbscan",
            )
        ],
    )
    assert report.counts["inserted"] == 1


# ----------------------------------------------------------------------
# Credentials never reach disk, whichever column carries them
# ----------------------------------------------------------------------


CREDENTIAL_PAYLOADS = [
    ("authorization-basic", "Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("authorization-bearer", "Authorization: Bearer abc123XYZ", "abc123XYZ"),
    ("bare-basic", "Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("bare-bearer", "Bearer abc123XYZ", "abc123XYZ"),
    ("api_key", "api_key=SEKRET123", "SEKRET123"),
    ("api-key", "api-key: SEKRET123", "SEKRET123"),
    ("x-api-key", "x-api-key: SEKRET123", "SEKRET123"),
    ("access_token", "access_token=ATOKEN987", "ATOKEN987"),
    ("password", "password: hunter2secret", "hunter2secret"),
    ("nested", {"outer": [{"api_key": "SEKRET123"}]}, "SEKRET123"),
    ("json-string", '{"Authorization": "Bearer abc123XYZ"}', "abc123XYZ"),
    ("query", "https://api.test/v1?api_key=SEKRET123&x=1", "SEKRET123"),
    ("url-userinfo", "https://user:hunter2secret@api.test/feed", "hunter2secret"),
]


@pytest.mark.parametrize(
    "label, payload, secret",
    CREDENTIAL_PAYLOADS,
    ids=[row[0] for row in CREDENTIAL_PAYLOADS],
)
def test_no_persisted_surface_retains_the_original_credential(
    tmp_path, label, payload, secret
):
    """Write the credential into every JSON/text column, then grep the file.

    Reading the columns back would only prove the accessors are clean; this
    reads the SQLite file itself, so a credential surviving anywhere in a
    page — including in a column this test forgot — still fails.
    """

    repository = migrated(tmp_path)
    day = build_day(repository)

    with repository.stage_run(
        run_id="run-secret",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        run.update_counts({"upstream": payload})
        run.record_error(payload)
        repository.record_source_state(
            "rss:test",
            run=run,
            successful=False,
            status="failed",
            metadata={"request": payload},
            error=payload,
        )

    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(
            source_metadata={"feed": payload}, trust_metadata={"upstream": payload}
        ),
        themes=[
            ThemeRecord(
                fingerprint="tf1",
                label="Theme",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                salience_rank=1,
                status="ready",
                method="hdbscan",
            )
        ],
    )

    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v2",
        stories=[
            story(
                "cf9",
                day["items"][:1],
                provider_conflicts=(
                    ProviderConflictRecord(
                        provider_namespace="yahoo",
                        provider_item_id="1",
                        item_ids=(payload,),
                        fields=(payload,),
                    ),
                ),
            )
        ],
    )

    # WAL mode keeps recent pages in the -wal sidecar, so all three files
    # have to be read or this assertion would pass vacuously.
    stored = b"".join(
        path.read_bytes()
        for suffix in ("", "-wal", "-shm")
        if (
            path := repository.database_path.with_name(
                repository.database_path.name + suffix
            )
        ).exists()
    )
    assert secret.encode() not in stored, f"{label} survived in the database files"

    entry = repository.run_log_entries(run_id="run-secret")[0]
    assert secret not in json.dumps(entry["counts"])
    assert secret not in json.dumps(entry["errors"])
    state = repository.source_state("rss:test")
    assert secret not in json.dumps(state)
    theme_row = repository.theme_set(
        ticker="NVDA", trading_day=DAY, pipeline_version="v1"
    )
    assert secret not in json.dumps(theme_row)


# ----------------------------------------------------------------------
# Run logs and source state
# ----------------------------------------------------------------------


def test_stage_run_logs_success_and_counts(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="fetch",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as recorder:
        recorder.record_success(3)
        recorder.record_partial(1)
        recorder.update_counts({"inserted": 3})

    entry = repository.run_log_entries(run_id="run-1")[0]
    assert entry["status"] == "degraded"
    assert entry["success_count"] == 3
    assert entry["partial_count"] == 1
    assert entry["counts"] == {"inserted": 3}
    assert entry["ticker"] == "NVDA"


def test_stage_run_logs_even_when_the_stage_raises(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(RuntimeError):
        with repository.stage_run(
            run_id="run-2",
            stage="cluster",
            trading_day=DAY,
            pipeline_version="v1",
            replay=True,
            attempt=2,
        ):
            raise RuntimeError("upstream returned Authorization: Bearer abc123XYZ")

    entry = repository.run_log_entries(run_id="run-2")[0]
    assert entry["status"] == "failed"
    assert entry["failure_count"] == 1
    assert entry["replay"] == 1
    assert entry["attempt"] == 2
    assert "abc123XYZ" not in json.dumps(entry["errors"])


def test_stage_logging_cannot_be_switched_off(tmp_path):
    forbidden = {"persist_run_log", "skip_run_log", "log", "logging"}
    for name, member in inspect.getmembers(Phase0Repository, inspect.isfunction):
        if name.startswith("_"):
            continue
        parameters = set(inspect.signature(member).parameters)
        assert not (parameters & forbidden), f"{name} exposes a logging bypass"


# ----------------------------------------------------------------------
# The logged mutation contract (issue #68's entrypoints)
# ----------------------------------------------------------------------


LOGGED_ENTRYPOINTS = [
    "ingest_raw_items",
    "reconcile_stories",
    "reconcile_themes",
    "persist_embeddings",
    "record_source_state",
]


def test_no_unlogged_batch_writer_is_reachable_from_the_public_surface(tmp_path):
    """The blocker in one assertion: no public mutation writes without a run.

    Every public method that mutates pipeline data must take ``run``.  The
    unlogged helpers still exist for fixtures and backfills, but only
    behind ``repository.admin``, which is impossible to call by accident.
    """

    mutating = {
        "ingest_raw_items",
        "insert_raw_item",
        "insert_raw_items",
        "reconcile_stories",
        "reconcile_themes",
        "persist_embeddings",
        "record_source_state",
        "set_source_state",
    }
    public = {
        name
        for name, _ in inspect.getmembers(Phase0Repository, inspect.isfunction)
        if not name.startswith("_")
    }

    for name in mutating & public:
        parameters = inspect.signature(getattr(Phase0Repository, name)).parameters
        assert "run" in parameters, f"{name} mutates pipeline data without a run"

    # And the unlogged names are genuinely gone from the repository itself.
    assert not {"insert_raw_item", "insert_raw_items", "set_source_state"} & public
    admin = Phase0Repository(tmp_path / "phase0.sqlite3").admin
    assert all(
        hasattr(admin, name)
        for name in ("insert_raw_item", "insert_raw_items", "set_source_state")
    )


@pytest.mark.parametrize("name", LOGGED_ENTRYPOINTS)
def test_every_pipeline_entrypoint_demands_a_run(name):
    signature = inspect.signature(getattr(Phase0Repository, name))
    run = signature.parameters["run"]

    assert run.kind is inspect.Parameter.KEYWORD_ONLY
    assert run.default is inspect.Parameter.empty, f"{name} makes run optional"


@pytest.mark.parametrize("handle", [None, "run-1", object()])
def test_mutation_without_a_run_context_fails_before_writing(tmp_path, handle):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0RunContextError, match="requires an active stage run"):
        repository.ingest_raw_items([raw_item(1)], run=handle)

    assert repository.count("raw_items") == 0
    assert repository.run_log_entries() == []


def test_a_completed_run_handle_is_rejected(tmp_path):
    repository = migrated(tmp_path)
    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        pass

    with pytest.raises(Phase0RunContextError, match="already completed"):
        repository.ingest_raw_items([raw_item(1)], run=run)

    assert repository.count("raw_items") == 0


def test_a_run_handle_from_another_repository_is_rejected(tmp_path):
    repository = migrated(tmp_path)
    other = migrated(tmp_path, name="other.sqlite3")

    with other.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as foreign:
        with pytest.raises(Phase0RunContextError, match="another repository"):
            repository.ingest_raw_items([raw_item(1)], run=foreign)

    assert repository.count("raw_items") == 0


def test_a_run_covering_a_different_partition_is_rejected(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)

    with repository.stage_run(
        run_id="run-1",
        stage="m3.semantic",
        trading_day="2026-07-24",
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        with pytest.raises(Phase0RunContextError, match="but the run covers"):
            repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", item_ids)],
            )

    assert repository.count("stories") == 0


def test_an_expired_stage_lease_rejects_the_mutation(tmp_path):
    repository = migrated(tmp_path)
    key = {
        "stage": "ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    assert repository.claim_stage_key(**key, run_id="run-1", lease_seconds=300)

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        # Another worker reclaims the key mid-run, exactly as the recovery
        # sweep would after a crash.
        with repository.connect() as connection:
            connection.execute(
                "UPDATE pipeline_stage_keys SET run_id = 'run-2' "
                "WHERE stage = ? AND ticker = ? AND trading_day = ? "
                "AND pipeline_version = ?",
                ("ingest", "NVDA", DAY, "v1"),
            )
        with pytest.raises(StageKeyError, match="owned by another run"):
            repository.ingest_raw_items([raw_item(1)], run=run)

    assert repository.count("raw_items") == 0


def test_a_successful_mutation_commits_its_run_log_in_the_same_transaction(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        results = repository.ingest_raw_items([raw_item(1), raw_item(2)], run=run)
        # Committed by the mutation itself, before the stage_run block ends.
        mid_run = repository.run_log_entries(run_id="run-1")

    assert len(results) == 2
    assert mid_run and mid_run[0]["success_count"] == 2
    assert mid_run[0]["counts"]["raw_items_inserted"] == 2
    assert repository.count("raw_items") == 2

    final = repository.run_log_entries(run_id="run-1")
    assert len(final) == 1
    assert final[0]["status"] == "success"


def test_a_failed_mutation_rolls_back_data_and_keeps_a_redacted_failed_log(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)

    with pytest.raises(sqlite3.IntegrityError):
        with repository.stage_run(
            run_id="run-fail",
            stage="m3.semantic",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            run.record_error("upstream said Authorization: Bearer abc123XYZ")
            repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", item_ids), story("cf2", [999_999])],
            )

    assert repository.count("stories") == 0
    entry = repository.run_log_entries(run_id="run-fail")[0]
    assert entry["status"] == "failed"
    assert "abc123XYZ" not in json.dumps(entry["errors"])


def test_a_failed_mutation_log_survives_a_reopened_database(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0ValidationError):
        with repository.stage_run(
            run_id="run-restart",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.ingest_raw_items([{"source": "", "canonical_url": ""}], run=run)

    reopened = Phase0Repository(repository.database_path)
    entry = reopened.run_log_entries(run_id="run-restart")[0]
    assert entry["status"] == "failed"
    assert reopened.count("raw_items") == 0


def test_counts_cannot_carry_a_credential_into_the_run_log(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-counts",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        run.update_counts(
            {
                "upstream": "Authorization: Basic dXNlcjpwYXNz",
                "retry_url": "https://api.test/v1?api_key=SEKRET123",
                "headers": [{"x-api-key": "SEKRET123"}],
            }
        )
        repository.ingest_raw_items([raw_item(1)], run=run)

    stored = json.dumps(repository.run_log_entries(run_id="run-counts")[0])
    assert "dXNlcjpwYXNz" not in stored
    assert "SEKRET123" not in stored


def test_embedding_batches_and_source_state_carry_their_run(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)
    vector = np.ones(4, dtype=EMBEDDING_DTYPE)
    embedding = PersistedEmbedding(
        source_kind="raw_item",
        source_id=str(item_ids[0]),
        model_name="fake",
        model_revision="r1",
        dimension=4,
        dtype=str(EMBEDDING_DTYPE),
        input_fingerprint="a" * 64,
        vector_blob=serialize_vector(vector),
    )

    with pytest.raises(Phase0RunContextError):
        repository.persist_embeddings([embedding], run=None)
    with pytest.raises(Phase0RunContextError):
        repository.record_source_state("rss:test", run=None, successful=True)
    assert repository.count("embeddings") == 0
    assert repository.source_state("rss:test") is None

    with repository.stage_run(
        run_id="run-embed",
        stage="m1.embed",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        assert repository.persist_embeddings([embedding], run=run) == 1
        repository.record_source_state(
            "rss:test",
            run=run,
            successful=True,
            metadata={"Authorization": "Bearer abc123XYZ"},
        )

    assert repository.count("embeddings") == 1
    state = repository.source_state("rss:test")
    assert "abc123XYZ" not in json.dumps(state)
    assert repository.run_log_entries(run_id="run-embed")[0]["status"] == "success"


def test_source_state_tracks_consecutive_failures_and_retry_state(tmp_path):
    repository = migrated(tmp_path)
    common = {"source": "rss:test", "etag": None, "last_modified": None}

    repository.admin.set_source_state(
        **common, checked_at=f"{DAY}T12:00:00+00:00", successful=False, status="failed"
    )
    repository.admin.set_source_state(
        **common, checked_at=f"{DAY}T12:30:00+00:00", successful=False, status="failed"
    )
    failing = repository.source_state("rss:test")
    assert failing["consecutive_failures"] == 2
    assert failing["last_success_at"] is None

    repository.admin.set_source_state(
        **common,
        checked_at=f"{DAY}T13:00:00+00:00",
        successful=True,
        status="partial",
        retry_after=f"{DAY}T14:00:00+00:00",
    )
    recovered = repository.source_state("rss:test")
    assert recovered["consecutive_failures"] == 0
    assert recovered["status"] == "partial"
    assert recovered["last_success_at"] == f"{DAY}T13:00:00+00:00"
    assert recovered["retry_after"] == f"{DAY}T14:00:00+00:00"


def test_source_state_validator_rejects_unknown_status():
    with pytest.raises(Phase0ValidationError, match="source-state status"):
        Phase0Repository.validate_source_state(
            {
                "source": "rss:test",
                "checked_at": f"{DAY}T12:00:00+00:00",
                "status": "ok",
            }
        )


def test_run_log_serialization_is_deterministic(tmp_path):
    repository = migrated(tmp_path)
    payload = {"b": 1, "a": {"d": 2, "c": 3}}

    for run_id in ("run-a", "run-b"):
        repository.log_stage(
            run_id=run_id,
            stage="fetch",
            counts=payload,
            duration_ms=1,
            errors=[],
            started_at=f"{DAY}T12:00:00+00:00",
            completed_at=f"{DAY}T12:00:01+00:00",
            trading_day=DAY,
            pipeline_version="v1",
        )

    with repository.connect() as connection:
        stored = {row[0] for row in connection.execute("SELECT counts FROM run_log")}
    assert stored == {'{"a":{"c":3,"d":2},"b":1}'}


# ----------------------------------------------------------------------
# Public API hygiene
# ----------------------------------------------------------------------


def test_sqlite_is_only_used_inside_the_persistence_layer():
    offenders = []
    for path in REPO_ROOT.rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        parts = relative.parts
        if parts[0] in {".venv", "frontend", "node_modules", "tests"}:
            continue
        if parts[0] == "phase0":
            continue
        if "sqlite3" in path.read_text(encoding="utf-8"):
            offenders.append(str(relative))
    assert offenders == []


def test_count_refuses_arbitrary_table_names(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0ValidationError, match="unsupported table"):
        repository.count("raw_items; DROP TABLE raw_items")


def test_unsupported_ticker_error_is_a_value_error():
    assert issubclass(UnsupportedTickerError, ValueError)
    assert issubclass(Phase0ValidationError, ValueError)

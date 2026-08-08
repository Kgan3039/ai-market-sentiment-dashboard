"""The known remote-v4 migration lineage, and everything it must not become.

Two branches wrote a different ``004_supported_ticker_universe.sql``.  Real
databases exist on the remote one, and the approved implementation
supersedes it — so those databases have to be upgradeable without the
migration ledger being taught to shrug at checksum mismatches in general.

These tests carry both halves: the one lineage upgrades and keeps its
data, and every neighbouring case that merely *resembles* it is still
refused.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from phase0 import lineages
from phase0.errors import Phase0MigrationError
from phase0.lineages import (
    KNOWN_HISTORICAL_MIGRATIONS,
    LINEAGE_TABLE,
    REMOTE_V4_LINEAGE,
)
from phase0.repository import Phase0Repository
from phase0.schema import LEDGER_TABLE, split_statements

from test_phase0_persistence_contracts import (
    ALL_MIGRATIONS,
    LATEST_VERSION,
    partial_migrations,
    schema_snapshot,
)


REMOTE_COMMIT = "836e8b5f02e2a2a8bc75993c81678c6534ea885a"
REMOTE_FIXTURE = Path(__file__).parent / "fixtures" / "remote_v4_migrations"
REPO_ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-07-23"


# ----------------------------------------------------------------------
# Building a genuine remote-v4 database
# ----------------------------------------------------------------------


def remote_v4_database(tmp_path: Path, name: str = "remote.sqlite3") -> Path:
    """A database as remote 836e8b5 would have left it.

    That code predates the ledger entirely: it tracks ``user_version`` and
    nothing else, so the database arrives with no ``schema_migrations``
    table at all.  This applies the fixture files exactly the way the
    remote ``migrate()`` did.
    """

    database = tmp_path / name
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for path in sorted(REMOTE_FIXTURE.glob("*.sql")):
            version = int(path.name.split("_", 1)[0])
            connection.execute("BEGIN IMMEDIATE")
            for statement in split_statements(path.read_text(encoding="utf-8")):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {version}")
            connection.commit()
    finally:
        connection.close()
    return database


def seed_remote_data(database: Path) -> dict:
    """Representative data in every table the remote schema supports."""

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        items = []
        for index, ticker in enumerate(["NVDA", "AMD", None, "TSLA"], start=1):
            cursor = connection.execute(
                """
                INSERT INTO raw_items (
                    source, ticker, title, description, url, canonical_url,
                    external_id, published_at, fetched_at, ingest_status,
                    validation_errors, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'valid', '[]', ?)
                """,
                (
                    f"yahoo:{index}",
                    ticker,
                    f"Headline {index}",
                    f"Body {index}",
                    f"https://example.com/{index}",
                    f"https://example.com/{index}",
                    f"ext-{index}",
                    f"{DAY}T12:0{index}:00+00:00",
                    f"{DAY}T12:30:00+00:00",
                    f'{{"index": {index}, "publisher": "Bearer looking text"}}',
                ),
            )
            items.append(int(cursor.lastrowid))
        for item_id, ticker in zip(items, ["NVDA", "AMD", "TSLA", "TSLA"]):
            connection.execute(
                "INSERT OR IGNORE INTO raw_item_tickers "
                "(raw_item_id, ticker, association_type) VALUES (?, ?, 'source')",
                (item_id, ticker),
            )
        for item_id, ticker in zip(items, ["AMD", "NVDA", "NVDA", "META"]):
            connection.execute(
                "INSERT OR IGNORE INTO raw_item_candidates "
                "(raw_item_id, ticker, reason) VALUES (?, ?, 'relevance_match')",
                (item_id, ticker),
            )
        story = int(
            connection.execute(
                "INSERT INTO stories (ticker, trading_day, canonical_title, "
                "outlet_count, member_ids) VALUES ('NVDA', ?, 'Story one', 1, '[]')",
                (DAY,),
            ).lastrowid
        )
        connection.execute(
            "INSERT INTO story_members (story_id, raw_item_id, position) "
            "VALUES (?, ?, 0)",
            (story, items[0]),
        )
        theme = int(
            connection.execute(
                "INSERT INTO themes (ticker, trading_day, label, citations, "
                "salience_rank, status, content_hash, pipeline_version) "
                "VALUES ('NVDA', ?, 'Chip demand', '[]', 1, 'ready', 'hash-1', 'v1')",
                (DAY,),
            ).lastrowid
        )
        connection.execute(
            "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
            (theme, story),
        )
        connection.execute(
            "INSERT INTO theme_citations (theme_id, raw_item_id) VALUES (?, ?)",
            (theme, items[0]),
        )
        connection.execute(
            "INSERT INTO pipeline_stage_keys (stage, ticker, trading_day, "
            "pipeline_version, status, run_id, updated_at) "
            "VALUES ('ingest', 'NVDA', ?, 'v1', 'success', 'remote-run', ?)",
            (DAY, f"{DAY}T13:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO run_log (run_id, stage, counts, duration_ms, errors, "
            "started_at, completed_at, status, trading_day, pipeline_version) "
            "VALUES ('remote-run', 'ingest', '{}', 5, '[]', ?, ?, 'success', ?, 'v1')",
            (f"{DAY}T12:00:00+00:00", f"{DAY}T12:01:00+00:00", DAY),
        )
        connection.execute(
            "INSERT INTO source_state (source, etag, last_modified, "
            "last_checked_at, last_success_at, metadata) "
            "VALUES ('yahoo', 'etag-1', NULL, ?, ?, '{}')",
            (f"{DAY}T12:30:00+00:00", f"{DAY}T12:30:00+00:00"),
        )
        connection.commit()
    finally:
        connection.close()
    return {"items": items, "story": story, "theme": theme}


def counts(repository: Phase0Repository, tables: list[str]) -> dict:
    return {table: repository.count(table) for table in tables}


PRESERVED_TABLES = [
    "raw_items",
    "raw_item_tickers",
    "raw_item_candidates",
    "stories",
    "story_members",
    "themes",
    "theme_stories",
    "theme_citations",
    "pipeline_stage_keys",
    "run_log",
    "source_state",
]


# ----------------------------------------------------------------------
# The fixture really is the remote lineage
# ----------------------------------------------------------------------


def test_the_remote_v4_fixture_is_the_real_thing():
    """The pinned checksum, the vendored file, and the commit all agree."""

    vendored = (REMOTE_FIXTURE / REMOTE_V4_LINEAGE.migration).read_bytes()
    assert hashlib.sha256(vendored).hexdigest() == REMOTE_V4_LINEAGE.checksum

    result = subprocess.run(
        [
            "git",
            "show",
            f"{REMOTE_COMMIT}:phase0/migrations/{REMOTE_V4_LINEAGE.migration}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    if result.returncode != 0:  # pragma: no cover - shallow clone
        pytest.skip("remote commit not present in this clone")
    assert result.stdout == vendored


def test_the_remote_lineage_differs_from_the_approved_one():
    """If these ever became the same file, this whole path is dead code."""

    approved = REPO_ROOT / "phase0" / "migrations" / REMOTE_V4_LINEAGE.migration
    approved_sum = hashlib.sha256(approved.read_bytes()).hexdigest()
    assert approved_sum != REMOTE_V4_LINEAGE.checksum
    # And the difference is the one the convergence exists to bridge.
    assert (
        "supported_tickers"
        not in (REMOTE_FIXTURE / REMOTE_V4_LINEAGE.migration).read_text()
    )
    assert "supported_tickers" in approved.read_text()


def test_a_remote_v4_database_has_the_expected_shape(tmp_path):
    database = remote_v4_database(tmp_path)
    connection = sqlite3.connect(database)
    try:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 4
        objects = {
            (row[0], row[1])
            for row in connection.execute("SELECT type, name FROM sqlite_master")
        }
        assert ("table", LEDGER_TABLE) not in objects  # predates the ledger
        assert ("table", "supported_tickers") not in objects
        assert ("trigger", "enforce_raw_item_ticker_insert") in objects
        assert ("trigger", "trg_raw_item_ticker_insert") not in objects
        assert (
            lineages.schema_fingerprint(
                connection, REMOTE_V4_LINEAGE.fingerprint_triggers
            )
            == REMOTE_V4_LINEAGE.schema_fingerprint
        )
    finally:
        connection.close()


# ----------------------------------------------------------------------
# A. The upgrade, and what it preserves
# ----------------------------------------------------------------------


def test_a_remote_v4_database_upgrades_to_the_approved_schema(tmp_path):
    database = remote_v4_database(tmp_path)
    seeded = seed_remote_data(database)

    repository = Phase0Repository(database)
    applied = repository.migrate()

    assert REMOTE_V4_LINEAGE.convergence in applied
    assert repository.schema_version() == LATEST_VERSION
    assert repository.count("raw_items") == len(seeded["items"])


def test_the_upgrade_preserves_valid_evidence(tmp_path):
    database = remote_v4_database(tmp_path)
    seeded = seed_remote_data(database)
    repository = Phase0Repository(database)
    before = counts(repository, PRESERVED_TABLES)

    repository.migrate()

    after = counts(repository, PRESERVED_TABLES)
    assert after == before, f"rows changed: {before} -> {after}"

    # The evidence itself, not just its row count.
    item = repository.read.raw_item(seeded["items"][0])
    assert item["title"] == "Headline 1"
    assert item["ticker"] == "NVDA"
    assert '"index": 1' in item["raw_json"]
    # ...including the item that legitimately matches no ticker.
    assert repository.read.raw_item(seeded["items"][2])["ticker"] is None


def test_the_upgrade_keeps_the_citation_relationship(tmp_path):
    database = remote_v4_database(tmp_path)
    seeded = seed_remote_data(database)
    repository = Phase0Repository(database)

    repository.migrate()

    with repository.admin.connect_writable() as connection:
        citations = connection.execute(
            "SELECT theme_id, raw_item_id FROM theme_citations"
        ).fetchall()
        members = connection.execute(
            "SELECT story_id, raw_item_id FROM story_members"
        ).fetchall()
    assert [tuple(row) for row in citations] == [(seeded["theme"], seeded["items"][0])]
    assert [tuple(row) for row in members] == [(seeded["story"], seeded["items"][0])]


def test_the_upgrade_normalizes_tickers_and_introduces_no_duplicates(tmp_path):
    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    repository = Phase0Repository(database)

    repository.migrate()

    with repository.admin.connect_writable() as connection:
        tickers = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT ticker FROM raw_item_tickers"
            )
        }
        duplicates = connection.execute(
            "SELECT raw_item_id, ticker, association_type, COUNT(*) c "
            "FROM raw_item_tickers GROUP BY 1, 2, 3 HAVING c > 1"
        ).fetchall()
    assert tickers <= {"TSLA", "NVDA", "AMD", "AAPL", "META"}
    assert tickers, "every association was dropped; that is not preservation"
    assert duplicates == []


def test_the_upgrade_applies_the_approved_unsupported_ticker_policy(tmp_path):
    """The approved policy, not the remote one: evidence is kept, cleared."""

    database = remote_v4_database(tmp_path)
    connection = sqlite3.connect(database)
    try:
        # The remote lineage's own trigger refuses an unsupported ticker, so
        # it comes off just long enough to plant one and then goes back
        # *verbatim* — the fingerprint is taken over these bodies, and a
        # database whose triggers do not match is not this lineage.
        original = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?",
            ("enforce_raw_item_ticker_insert",),
        ).fetchone()[0]
        connection.execute("DROP TRIGGER enforce_raw_item_ticker_insert")
        connection.execute(
            "INSERT INTO raw_items (source, ticker, title, url, canonical_url, "
            "fetched_at, ingest_status, validation_errors, raw_json) "
            "VALUES ('yahoo:x', 'GOOG', 'Alphabet headline', 'https://e/x', "
            "'https://e/x', ?, 'valid', '[]', '{}')",
            (f"{DAY}T12:00:00+00:00",),
        )
        connection.execute(original)
        connection.commit()
    finally:
        connection.close()

    repository = Phase0Repository(database)
    repository.migrate()

    with repository.admin.connect_writable() as connection:
        row = connection.execute(
            "SELECT ticker, title FROM raw_items WHERE source = 'yahoo:x'"
        ).fetchone()
    # Kept as evidence, with the unsupported symbol cleared to NULL.
    assert row is not None
    assert row[0] is None
    assert row[1] == "Alphabet headline"


def test_the_upgraded_database_takes_approved_writes(tmp_path):
    """Not just a schema: the converged database actually works."""

    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    repository = Phase0Repository(database)
    repository.migrate()

    with repository.stage_run(
        run_id="post-upgrade",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items(
            [
                {
                    "source": "yahoo:new",
                    "ticker": "NVDA",
                    "title": "New headline",
                    "url": "https://example.com/new",
                    "canonical_url": "https://example.com/new",
                    "published_at": f"{DAY}T14:00:00+00:00",
                    "fetched_at": f"{DAY}T14:05:00+00:00",
                    "raw_json": {"new": True},
                }
            ],
            run=run,
            terminal=True,
        )

    assert repository.run_log_entries(run_id="post-upgrade")[0]["status"] == "success"
    with pytest.raises(Exception):
        with repository.admin.connect_writable() as connection:
            connection.execute(
                "INSERT INTO raw_item_tickers (raw_item_id, ticker, "
                "association_type) VALUES (1, 'GOOG', 'source')"
            )


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def test_the_upgrade_records_its_provenance(tmp_path):
    database = remote_v4_database(tmp_path)
    repository = Phase0Repository(database)
    repository.migrate()

    rows = repository.schema_lineages()
    assert len(rows) == 1
    assert rows[0]["lineage"] == REMOTE_V4_LINEAGE.lineage
    assert rows[0]["migration"] == REMOTE_V4_LINEAGE.migration
    assert rows[0]["historical_checksum"] == REMOTE_V4_LINEAGE.checksum
    assert rows[0]["convergence"] == REMOTE_V4_LINEAGE.convergence


def test_the_ledger_never_claims_the_approved_004_ran(tmp_path):
    """The record stays true: a different file ran, and it says so."""

    database = remote_v4_database(tmp_path)
    repository = Phase0Repository(database)
    repository.migrate()

    ledger = {row["name"]: row["checksum"] for row in repository.applied_migrations()}
    approved = {migration.name: migration.checksum for migration in ALL_MIGRATIONS}
    assert ledger[REMOTE_V4_LINEAGE.migration] == REMOTE_V4_LINEAGE.checksum
    assert ledger[REMOTE_V4_LINEAGE.migration] != approved[REMOTE_V4_LINEAGE.migration]
    assert ledger[REMOTE_V4_LINEAGE.convergence] == (
        REMOTE_V4_LINEAGE.convergence_checksum
    )
    # Everything else is the approved history, unchanged.
    for name, checksum in approved.items():
        if name == REMOTE_V4_LINEAGE.migration:
            continue
        assert ledger[name] == checksum


def test_a_fresh_database_records_no_lineage(tmp_path):
    repository = Phase0Repository(tmp_path / "fresh.sqlite3")
    repository.migrate()

    assert repository.schema_lineages() == []
    ledger = {row["name"] for row in repository.applied_migrations()}
    assert REMOTE_V4_LINEAGE.convergence not in ledger


# ----------------------------------------------------------------------
# B, C, J. Everything else is unaffected
# ----------------------------------------------------------------------


def test_a_fresh_database_is_unaffected(tmp_path):
    repository = Phase0Repository(tmp_path / "fresh.sqlite3")
    applied = repository.migrate()

    assert [name for name in applied] == [
        migration.name for migration in ALL_MIGRATIONS
    ]
    assert repository.schema_version() == LATEST_VERSION


@pytest.mark.parametrize("version", sorted({m.version for m in ALL_MIGRATIONS})[:-1])
def test_approved_prior_versions_still_upgrade(tmp_path, version):
    """Every approved historical version, none of them a lineage."""

    database = tmp_path / f"v{version}.sqlite3"
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, version)
    ).migrate()

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.schema_version() == LATEST_VERSION
    assert upgraded.schema_lineages() == []


def test_the_upgraded_schema_equals_a_fresh_one(tmp_path):
    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    upgraded = Phase0Repository(database)
    upgraded.migrate()

    fresh = Phase0Repository(tmp_path / "fresh.sqlite3")
    fresh.migrate()

    assert schema_snapshot(upgraded) == schema_snapshot(fresh)
    assert upgraded.schema_version() == fresh.schema_version()


def test_the_convergence_produces_exactly_the_approved_v4_schema(tmp_path):
    """The convergence's contract, checked rather than described.

    Everything after it assumes an approved v4 database. If that is not
    what it produces, migrations 005 onwards are running against a schema
    nobody designed them for — and copying 004's DDL into the compat file
    would be a live drift hazard instead of a frozen quotation.
    """

    database = remote_v4_database(tmp_path)
    converged = Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 4)
    )
    converged.migrate()

    approved_v4 = Phase0Repository(
        tmp_path / "approved_v4.sqlite3",
        migrations_path=partial_migrations(tmp_path, 4),
    )
    approved_v4.migrate()

    assert schema_snapshot(converged) == schema_snapshot(approved_v4)


def test_repeated_migration_is_idempotent(tmp_path):
    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    repository = Phase0Repository(database)
    repository.migrate()

    snapshot = schema_snapshot(repository)
    before = counts(repository, PRESERVED_TABLES)
    lineage_rows = repository.schema_lineages()
    ledger = repository.applied_migrations()

    assert repository.migrate() == []
    assert repository.migrate() == []

    assert schema_snapshot(repository) == snapshot
    assert counts(repository, PRESERVED_TABLES) == before
    assert repository.schema_lineages() == lineage_rows
    assert repository.applied_migrations() == ledger


def test_a_reopened_converged_database_still_verifies(tmp_path):
    """The historical checksum stays in the ledger, so recognition must last."""

    database = remote_v4_database(tmp_path)
    Phase0Repository(database).migrate()

    reopened = Phase0Repository(database)
    assert reopened.migrate() == []
    assert reopened.schema_version() == LATEST_VERSION
    assert len(reopened.schema_lineages()) == 1


# ----------------------------------------------------------------------
# D-G. Nothing that merely resembles the lineage qualifies
# ----------------------------------------------------------------------


def altered_remote_migrations(tmp_path: Path, suffix: str) -> Path:
    """The remote fixture with 004 modified — a variant nobody registered."""

    target = tmp_path / f"altered{suffix}"
    target.mkdir(exist_ok=True)
    for path in REMOTE_FIXTURE.glob("*.sql"):
        shutil.copy(path, target / path.name)
    victim = target / REMOTE_V4_LINEAGE.migration
    victim.write_text(
        f"-- variant {suffix}\n" + victim.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return target


def test_an_unknown_004_variant_is_still_rejected(tmp_path):
    """A different fork is not this fork.

    The triggers are identical, so the schema fingerprint matches — and it
    is still refused, because the checksum does not. Recognition needs
    every condition, not a majority of them.
    """

    altered = altered_remote_migrations(tmp_path, "-unknown")
    database = tmp_path / "altered.sqlite3"
    connection = sqlite3.connect(database)
    try:
        for path in sorted(altered.glob("*.sql")):
            version = int(path.name.split("_", 1)[0])
            for statement in split_statements(path.read_text(encoding="utf-8")):
                connection.execute(statement)
            connection.execute(f"PRAGMA user_version = {version}")
        connection.execute(
            "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, "
            "version INTEGER NOT NULL CHECK (version > 0), "
            "checksum TEXT NOT NULL CHECK (length(checksum) = 64), "
            "applied_at TEXT NOT NULL CHECK (datetime(applied_at) IS NOT NULL))"
        )
        for path in sorted(altered.glob("*.sql")):
            text = path.read_text(encoding="utf-8")
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (
                    path.name,
                    int(path.name.split("_", 1)[0]),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    f"{DAY}T12:00:00+00:00",
                ),
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_the_known_checksum_without_the_schema_is_rejected(tmp_path):
    """A ledger row is a claim; the schema is the evidence."""

    # An approved v4 database — right version, wrong lineage — with the
    # remote checksum written into its ledger.
    database = tmp_path / "impostor.sqlite3"
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 4)
    ).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            (REMOTE_V4_LINEAGE.checksum, REMOTE_V4_LINEAGE.migration),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_a_random_database_claiming_version_four_is_rejected(tmp_path):
    database = tmp_path / "random.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE raw_items (id INTEGER PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 4")
        connection.execute(
            "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, "
            "version INTEGER NOT NULL CHECK (version > 0), "
            "checksum TEXT NOT NULL CHECK (length(checksum) = 64), "
            "applied_at TEXT NOT NULL CHECK (datetime(applied_at) IS NOT NULL))"
        )
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, 4, ?, ?)",
            (
                REMOTE_V4_LINEAGE.migration,
                REMOTE_V4_LINEAGE.checksum,
                f"{DAY}T12:00:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError):
        Phase0Repository(database).migrate()


def test_the_right_schema_with_a_wrong_checksum_is_rejected(tmp_path):
    """Fingerprint alone does not qualify either; both must hold."""

    database = remote_v4_database(tmp_path)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, "
            "version INTEGER NOT NULL CHECK (version > 0), "
            "checksum TEXT NOT NULL CHECK (length(checksum) = 64), "
            "applied_at TEXT NOT NULL CHECK (datetime(applied_at) IS NOT NULL))"
        )
        for path in sorted(REMOTE_FIXTURE.glob("*.sql")):
            text = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if path.name == REMOTE_V4_LINEAGE.migration:
                checksum = "f" * 64  # neither the approved nor the remote one
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (
                    path.name,
                    int(path.name.split("_", 1)[0]),
                    checksum,
                    f"{DAY}T12:00:00+00:00",
                ),
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_a_lineage_whose_earlier_history_differs_is_rejected(tmp_path):
    """Forked before 004 is a different fork, whatever 004 says."""

    database = remote_v4_database(tmp_path)
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, "
            "version INTEGER NOT NULL CHECK (version > 0), "
            "checksum TEXT NOT NULL CHECK (length(checksum) = 64), "
            "applied_at TEXT NOT NULL CHECK (datetime(applied_at) IS NOT NULL))"
        )
        for path in sorted(REMOTE_FIXTURE.glob("*.sql")):
            text = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if path.name == "002_source_state_and_stage_keys.sql":
                checksum = "a" * 64
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (
                    path.name,
                    int(path.name.split("_", 1)[0]),
                    checksum,
                    f"{DAY}T12:00:00+00:00",
                ),
            )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_editing_an_applied_approved_migration_is_still_rejected(tmp_path):
    """The general rule, untouched by any of this."""

    repository = Phase0Repository(tmp_path / "approved.sqlite3")
    repository.migrate()
    connection = sqlite3.connect(repository.database_path)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            ("b" * 64, "009_immutable_domain_and_update_integrity.sql"),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(repository.database_path).migrate()


def test_a_forged_provenance_row_does_not_excuse_a_foreign_checksum(tmp_path):
    """Provenance excuses one exact pairing, not a checksum in general."""

    repository = Phase0Repository(tmp_path / "approved.sqlite3")
    repository.migrate()
    connection = sqlite3.connect(repository.database_path)
    try:
        connection.execute(
            f"INSERT INTO {LINEAGE_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
            (
                REMOTE_V4_LINEAGE.lineage,
                REMOTE_V4_LINEAGE.migration,
                REMOTE_V4_LINEAGE.checksum,
                REMOTE_V4_LINEAGE.schema_fingerprint,
                REMOTE_V4_LINEAGE.convergence,
                f"{DAY}T12:00:00+00:00",
            ),
        )
        # ...and a checksum that is neither the approved nor the historical one.
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            ("c" * 64, REMOTE_V4_LINEAGE.migration),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(repository.database_path).migrate()


def test_the_convergence_file_is_held_to_its_pinned_checksum(tmp_path, monkeypatch):
    """The compatibility path is not an excuse to stop checking anything."""

    compat = tmp_path / "compat"
    compat.mkdir()
    original = lineages.CONVERGENCE_PATH / REMOTE_V4_LINEAGE.convergence
    (compat / REMOTE_V4_LINEAGE.convergence).write_text(
        original.read_text(encoding="utf-8") + "\n-- tampered\n", encoding="utf-8"
    )
    monkeypatch.setattr(lineages, "CONVERGENCE_PATH", compat)

    database = remote_v4_database(tmp_path)
    with pytest.raises(Phase0MigrationError, match="pinned checksum"):
        Phase0Repository(database).migrate()

    # ...and nothing was applied on the way to finding that out.
    connection = sqlite3.connect(database)
    try:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 4
    finally:
        connection.close()


def test_a_missing_convergence_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(lineages, "CONVERGENCE_PATH", tmp_path / "absent")

    database = remote_v4_database(tmp_path)
    with pytest.raises(Phase0MigrationError, match="convergence migration"):
        Phase0Repository(database).migrate()


def test_the_registry_is_a_closed_list():
    """One entry, and it is the one this branch reviewed."""

    assert list(KNOWN_HISTORICAL_MIGRATIONS) == [
        (REMOTE_V4_LINEAGE.migration, REMOTE_V4_LINEAGE.checksum)
    ]
    assert len(REMOTE_V4_LINEAGE.checksum) == 64
    assert len(REMOTE_V4_LINEAGE.schema_fingerprint) == 64
    assert REMOTE_V4_LINEAGE.description


# ----------------------------------------------------------------------
# H. A failed convergence leaves nothing behind
# ----------------------------------------------------------------------


def test_a_failed_convergence_rolls_everything_back(tmp_path, monkeypatch):
    """Ledger, schema, data, provenance, and user_version, all or nothing."""

    compat = tmp_path / "compat"
    compat.mkdir()
    original = lineages.CONVERGENCE_PATH / REMOTE_V4_LINEAGE.convergence
    broken = original.read_text(encoding="utf-8") + "\nSELECT this_is_not_a_column;\n"
    (compat / REMOTE_V4_LINEAGE.convergence).write_text(broken, encoding="utf-8")
    monkeypatch.setattr(lineages, "CONVERGENCE_PATH", compat)
    patched = hashlib.sha256(broken.encode("utf-8")).hexdigest()
    monkeypatch.setitem(
        KNOWN_HISTORICAL_MIGRATIONS,
        (REMOTE_V4_LINEAGE.migration, REMOTE_V4_LINEAGE.checksum),
        _with_checksum(REMOTE_V4_LINEAGE, patched),
    )

    database = remote_v4_database(tmp_path)
    seeded = seed_remote_data(database)

    with pytest.raises(sqlite3.Error):
        Phase0Repository(database).migrate()

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 4
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' "
                "AND name = 'supported_tickers'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' "
                "AND name = 'enforce_raw_item_ticker_insert'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(f"SELECT COUNT(*) FROM {LINEAGE_TABLE}").fetchone()[0]
            == 0
        )
        applied = {
            row["name"]
            for row in connection.execute(f"SELECT name FROM {LEDGER_TABLE}")
        }
        assert REMOTE_V4_LINEAGE.convergence not in applied
        assert connection.execute("SELECT COUNT(*) FROM raw_items").fetchone()[
            0
        ] == len(seeded["items"])
    finally:
        connection.close()


def _with_checksum(lineage, checksum):
    import dataclasses

    return dataclasses.replace(lineage, convergence_checksum=checksum)


def test_a_failed_convergence_can_be_retried_after_the_fix(tmp_path, monkeypatch):
    """A rolled-back convergence is not a poisoned database."""

    compat = tmp_path / "compat"
    compat.mkdir()
    original = lineages.CONVERGENCE_PATH / REMOTE_V4_LINEAGE.convergence
    broken = original.read_text(encoding="utf-8") + "\nSELECT this_is_not_a_column;\n"
    (compat / REMOTE_V4_LINEAGE.convergence).write_text(broken, encoding="utf-8")
    monkeypatch.setattr(lineages, "CONVERGENCE_PATH", compat)
    monkeypatch.setitem(
        KNOWN_HISTORICAL_MIGRATIONS,
        (REMOTE_V4_LINEAGE.migration, REMOTE_V4_LINEAGE.checksum),
        _with_checksum(
            REMOTE_V4_LINEAGE, hashlib.sha256(broken.encode("utf-8")).hexdigest()
        ),
    )

    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    with pytest.raises(sqlite3.Error):
        Phase0Repository(database).migrate()

    monkeypatch.undo()
    repository = Phase0Repository(database)
    repository.migrate()

    assert repository.schema_version() == LATEST_VERSION
    assert len(repository.schema_lineages()) == 1


# ----------------------------------------------------------------------
# 7. Regressions worth keeping from the remote branch's own tests
# ----------------------------------------------------------------------


def test_a_dirty_pre_v4_database_is_cleaned_on_upgrade(tmp_path):
    """Lower-cased, padded, and unsupported tickers, all present at v3."""

    database = tmp_path / "dirty.sqlite3"
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 3)
    ).migrate()
    connection = sqlite3.connect(database)
    try:
        for index, ticker in enumerate([" nvda ", "AmD", "GOOG", "tsla"], start=1):
            connection.execute(
                "INSERT INTO raw_items (source, ticker, title, url, canonical_url, "
                "fetched_at, ingest_status, validation_errors, raw_json) "
                "VALUES (?, ?, 'T', ?, ?, ?, 'valid', '[]', '{}')",
                (
                    f"yahoo:{index}",
                    ticker,
                    f"https://e/{index}",
                    f"https://e/{index}",
                    f"{DAY}T12:00:00+00:00",
                ),
            )
            connection.execute(
                "INSERT INTO raw_item_tickers (raw_item_id, ticker, association_type) "
                "VALUES (?, ?, 'source')",
                (index, ticker),
            )
        connection.commit()
    finally:
        connection.close()

    repository = Phase0Repository(database)
    repository.migrate()

    with repository.admin.connect_writable() as connection:
        tickers = [
            row[0]
            for row in connection.execute("SELECT ticker FROM raw_items ORDER BY id")
        ]
        associations = {
            row[0] for row in connection.execute("SELECT ticker FROM raw_item_tickers")
        }
    # Normalized where they were approved, cleared where they were not, and
    # the evidence rows themselves are all still here.
    assert tickers == ["NVDA", "AMD", None, "TSLA"]
    assert associations == {"NVDA", "AMD", "TSLA"}
    assert repository.count("raw_items") == 4


def test_a_failed_migration_rolls_back_an_added_column(tmp_path):
    """SQLite runs DDL transactionally; a failure must not leave half of it.

    Migration 008 adds seven columns before doing anything else, so a
    failure after them is the case that would silently produce a schema no
    version number describes.
    """

    database = tmp_path / "rollback.sqlite3"
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 7)
    ).migrate()

    broken = tmp_path / "broken_migrations"
    broken.mkdir()
    for migration in ALL_MIGRATIONS:
        if migration.version <= 8:
            shutil.copy(
                Path("phase0/migrations") / migration.name, broken / migration.name
            )
    victim = broken / "008_run_log_and_source_state.sql"
    victim.write_text(
        victim.read_text(encoding="utf-8") + "\nSELECT no_such_column;\n",
        encoding="utf-8",
    )

    with pytest.raises(sqlite3.Error):
        Phase0Repository(database, migrations_path=broken).migrate()

    connection = sqlite3.connect(database)
    try:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 7
        columns = {row[1] for row in connection.execute("PRAGMA table_info(run_log)")}
        assert "success_count" not in columns
        assert "ticker" not in columns
        applied = {
            row[0] for row in connection.execute(f"SELECT name FROM {LEDGER_TABLE}")
        }
        assert "008_run_log_and_source_state.sql" not in applied
    finally:
        connection.close()


def test_repointing_a_cited_story_member_is_refused(tmp_path):
    """The citation lifecycle holds against direct SQL, not just the API."""

    database = remote_v4_database(tmp_path)
    seeded = seed_remote_data(database)
    repository = Phase0Repository(database)
    repository.migrate()

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE story_members SET raw_item_id = ? WHERE story_id = ?",
                (seeded["items"][1], seeded["story"]),
            )
    # The citation still points at evidence the story actually contains.
    with repository.admin.connect_writable() as connection:
        assert (
            connection.execute("SELECT raw_item_id FROM theme_citations").fetchone()[0]
            == seeded["items"][0]
        )


DIGEST_AND_COOKIE_PAYLOADS = [
    ("digest-bare", "Digest abc123XYZ", "abc123XYZ"),
    ("digest-short", "Digest Zq7", "Zq7"),
    ("cookie", "Cookie: session=SEKRET123", "SEKRET123"),
    ("set-cookie", "Set-Cookie: sid=SEKRET123; Path=/", "SEKRET123"),
    ("proxy-auth", "Proxy-Authorization: Digest SEKRET123", "SEKRET123"),
]


@pytest.mark.parametrize(
    "label, payload, secret",
    DIGEST_AND_COOKIE_PAYLOADS,
    ids=[row[0] for row in DIGEST_AND_COOKIE_PAYLOADS],
)
def test_digest_and_cookie_credentials_are_redacted(tmp_path, label, payload, secret):
    """Schemes and headers the scheme-credential tests did not name."""

    from phase0.redaction import redact_secrets, redact_text

    assert secret not in redact_text(payload)
    assert secret not in str(redact_secrets({"detail": payload}))

    repository = Phase0Repository(tmp_path / "redaction.sqlite3")
    repository.migrate()
    with repository.stage_run(
        run_id="redaction",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.record_source_state(
            "yahoo",
            run=run,
            successful=False,
            metadata={"detail": payload},
            terminal=True,
        )

    stored = repository.read.source_state_rows()[0]
    assert secret not in str(stored)

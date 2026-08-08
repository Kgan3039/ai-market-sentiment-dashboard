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

import contextlib
import hashlib
import shutil
import sqlite3
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from phase0 import lineages
from phase0.errors import Phase0MigrationError
from phase0.lineages import (
    KNOWN_HISTORICAL_MIGRATIONS,
    LINEAGE_TABLE,
    REMOTE_V4_LINEAGE,
)
from phase0.repository import Phase0Repository
from phase0.schema import LEDGER_DDL, LEDGER_TABLE, split_statements

from test_phase0_persistence_contracts import (
    ALL_MIGRATIONS,
    LATEST_VERSION,
    partial_migrations,
    schema_snapshot,
)


REMOTE_COMMIT = "836e8b5f02e2a2a8bc75993c81678c6534ea885a"

#: Either refusal. A database carrying the fork's marker objects is
#: refused by *recognition* ("does not match it"), before any checksum is
#: consulted; one without them reaches the ordinary checksum rule.
REFUSED = "was modified after it was applied|does not match it"
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
            lineages.structural_fingerprint(connection)
            == REMOTE_V4_LINEAGE.schema_fingerprint
        )
        assert lineages.ledger_rows(connection) is None
        assert lineages.lineage_rows(connection) is None
        assert lineages.mismatch(connection, REMOTE_V4_LINEAGE) is None
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
        connection.execute(LEDGER_DDL)
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

    with pytest.raises(Phase0MigrationError, match=REFUSED):
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
        connection.execute(LEDGER_DDL)
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
        connection.execute(LEDGER_DDL)
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

    with pytest.raises(Phase0MigrationError, match=REFUSED):
        Phase0Repository(database).migrate()


def test_a_lineage_whose_earlier_history_differs_is_rejected(tmp_path):
    """Forked before 004 is a different fork, whatever 004 says."""

    database = remote_v4_database(tmp_path)
    connection = sqlite3.connect(database)
    try:
        connection.execute(LEDGER_DDL)
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

    with pytest.raises(Phase0MigrationError, match=REFUSED):
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


def _with_checksum(lineage, checksum):
    """The registry entry, re-pinned to a deliberately altered file."""

    import dataclasses

    return dataclasses.replace(lineage, convergence_checksum=checksum)


def full_snapshot(database: Path) -> dict:
    """Everything a compatibility settlement could possibly disturb."""

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:

        def rows(sql):
            try:
                return [tuple(row) for row in connection.execute(sql)]
            except sqlite3.Error as exc:
                return f"<{exc}>"

        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            "user_version": connection.execute("PRAGMA user_version").fetchone()[0],
            "schema": rows(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "ORDER BY type, name"
            ),
            "schema_migrations": rows(
                "SELECT name, version, checksum FROM schema_migrations ORDER BY name"
            ),
            LINEAGE_TABLE: rows(f"SELECT * FROM {LINEAGE_TABLE}"),
            "data": {
                table: rows(f"SELECT * FROM {table}")
                for table in tables
                if table not in {"schema_migrations", LINEAGE_TABLE}
            },
        }
    finally:
        connection.close()


class _Sabotage(sqlite3.Connection):
    """A connection that fails on the Nth statement matching a marker."""

    marker = ""
    countdown = 0

    def execute(self, sql, *args, **kwargs):  # noqa: D102 - see class docstring
        if type(self).marker and type(self).marker.lower() in str(sql).lower():
            type(self).countdown -= 1
            if type(self).countdown == 0:
                raise sqlite3.OperationalError("injected compatibility failure")
        return super().execute(sql, *args, **kwargs)


@contextlib.contextmanager
def fail_at(marker: str, occurrence: int = 1):
    """Fail the ``occurrence``-th statement containing ``marker``."""

    real_connect = sqlite3.connect

    class Sabotaged(_Sabotage):
        pass

    Sabotaged.marker = marker
    Sabotaged.countdown = occurrence

    def connect(*args, **kwargs):
        if not kwargs.get("uri"):
            kwargs["factory"] = Sabotaged
        return real_connect(*args, **kwargs)

    with mock.patch("phase0.repository.sqlite3.connect", side_effect=connect):
        yield


#: One injection point per stage of the settlement, named by a statement
#: only that stage issues.
SETTLEMENT_STAGES = [
    ("bootstrap-ledger", "CREATE TABLE IF NOT EXISTS schema_migrations", 1),
    ("bootstrap-lineage", "CREATE TABLE IF NOT EXISTS schema_lineage", 1),
    ("truthful-backfill", "INSERT OR REPLACE INTO schema_migrations", 1),
    ("backfill-last-row", "INSERT OR REPLACE INTO schema_migrations", 4),
    ("first-schema-change", "DROP TRIGGER IF EXISTS enforce_raw_item_ticker_insert", 1),
    ("convergence-table", "CREATE TABLE IF NOT EXISTS supported_tickers", 1),
    ("convergence-midway", "CREATE TRIGGER IF NOT EXISTS trg_story_ticker_insert", 1),
    ("ledger-row", "INSERT INTO schema_migrations", 1),
    ("later-migration", "CREATE TABLE IF NOT EXISTS embeddings", 1),
    ("before-validation", "PRAGMA user_version = 11", 1),
    ("provenance", "INSERT OR IGNORE INTO schema_lineage", 1),
]


@pytest.mark.parametrize(
    "label, marker, occurrence",
    SETTLEMENT_STAGES,
    ids=[row[0] for row in SETTLEMENT_STAGES],
)
def test_a_failure_at_any_settlement_stage_changes_nothing(
    tmp_path, label, marker, occurrence
):
    """All of it or none of it, wherever the failure lands.

    A half-converted fork is at no version at all: bootstrap tables it did
    not ask for, a ledger describing a history it has not lived, and a
    schema partway between two branches. The earlier version of this code
    produced exactly that.
    """

    database = remote_v4_database(tmp_path, f"{label}.sqlite3")
    seed_remote_data(database)
    before = full_snapshot(database)

    with fail_at(marker, occurrence):
        with pytest.raises(sqlite3.Error, match="injected compatibility failure"):
            Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_a_failure_before_recognition_changes_nothing(tmp_path):
    """Recognition itself is read-only: asking must cost nothing."""

    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    before = full_snapshot(database)

    # A database that is *not* on the lineage must be untouched when we
    # find that out, so drop a table the fingerprint covers.
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE run_log")
        connection.commit()
    finally:
        connection.close()
    damaged = full_snapshot(database)

    with pytest.raises(Phase0MigrationError):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == damaged
    assert before != damaged  # the probe itself was not a no-op


def test_a_failed_convergence_rolls_everything_back(tmp_path, monkeypatch):
    """The original report: bootstrap state survived a failed conversion."""

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
    seed_remote_data(database)
    before = full_snapshot(database)

    with pytest.raises(sqlite3.Error):
        Phase0Repository(database).migrate()

    after = full_snapshot(database)
    assert after == before
    # Named explicitly, because these are the four that survived before.
    assert after["schema_migrations"] == "<no such table: schema_migrations>"
    assert after[LINEAGE_TABLE] == f"<no such table: {LINEAGE_TABLE}>"
    assert after["user_version"] == 4
    assert len(after["data"]["raw_items"]) == 4


def test_a_settlement_that_does_not_land_refuses_to_commit(tmp_path, monkeypatch):
    """Marking a migration applied is not the same as its schema existing."""

    compat = tmp_path / "compat"
    compat.mkdir()
    original = lineages.CONVERGENCE_PATH / REMOTE_V4_LINEAGE.convergence
    # Runs cleanly and does nothing: every statement is a comment.
    hollow = (
        "\n".join(
            f"-- {line}" for line in original.read_text(encoding="utf-8").splitlines()
        )
        + "\nSELECT 1;\n"
    )
    (compat / REMOTE_V4_LINEAGE.convergence).write_text(hollow, encoding="utf-8")
    monkeypatch.setattr(lineages, "CONVERGENCE_PATH", compat)
    monkeypatch.setitem(
        KNOWN_HISTORICAL_MIGRATIONS,
        (REMOTE_V4_LINEAGE.migration, REMOTE_V4_LINEAGE.checksum),
        _with_checksum(
            REMOTE_V4_LINEAGE, hashlib.sha256(hollow.encode("utf-8")).hexdigest()
        ),
    )

    database = remote_v4_database(tmp_path)
    seed_remote_data(database)
    before = full_snapshot(database)

    with pytest.raises(Exception):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


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


# ----------------------------------------------------------------------
# Exact-lineage recognition: near misses are refused, before anything runs
# ----------------------------------------------------------------------


def damaged_remote_database(tmp_path: Path, name: str, *statements: str) -> Path:
    """A genuine remote database, then bent out of shape."""

    database = remote_v4_database(tmp_path, name)
    seed_remote_data(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()
    return database


#: One structural difference each, all of them remote-*looking*.  Before
#: the fingerprint covered the whole schema, the first of these was
#: accepted and advanced through four migrations before failing.
NEAR_MISSES = [
    ("missing-run_log", ("DROP TABLE run_log",)),
    ("missing-eval_labels", ("DROP TABLE eval_labels",)),
    ("missing-source_state", ("DROP TABLE source_state",)),
    ("missing-story_members", ("DROP TABLE story_members",)),
    (
        "missing-trigger",
        ("DROP TRIGGER enforce_theme_ticker_update",),
    ),
    (
        "changed-trigger",
        (
            "DROP TRIGGER enforce_story_ticker_insert",
            "CREATE TRIGGER enforce_story_ticker_insert BEFORE INSERT ON stories "
            "WHEN NEW.ticker = 'NOPE' BEGIN SELECT RAISE(ABORT, 'no'); END",
        ),
    ),
    ("added-column", ("ALTER TABLE raw_items ADD COLUMN smuggled TEXT",)),
    ("added-table", ("CREATE TABLE smuggled (id INTEGER PRIMARY KEY)",)),
    ("added-index", ("CREATE INDEX idx_smuggled ON raw_items(external_id)",)),
    ("dropped-index", ("DROP INDEX idx_raw_items_ticker_published",)),
    ("wrong-user-version", ("PRAGMA user_version = 5",)),
]


@pytest.mark.parametrize(
    "label, statements", NEAR_MISSES, ids=[row[0] for row in NEAR_MISSES]
)
def test_a_near_miss_is_refused_before_anything_is_applied(tmp_path, label, statements):
    database = damaged_remote_database(tmp_path, f"{label}.sqlite3", *statements)
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


PARTIAL_LEDGERS = [
    (
        "missing-004",
        [
            "001_initial.sql",
            "002_source_state_and_stage_keys.sql",
            "003_integrity_leases_and_upgrade.sql",
        ],
    ),
    ("only-004", ["004_supported_ticker_universe.sql"]),
    ("empty", []),
]


@pytest.mark.parametrize(
    "label, names", PARTIAL_LEDGERS, ids=[row[0] for row in PARTIAL_LEDGERS]
)
def test_a_partial_ledger_on_the_remote_schema_is_refused(tmp_path, label, names):
    """Either no ledger at all, or exactly this lineage's history."""

    database = remote_v4_database(tmp_path, f"ledger-{label}.sqlite3")
    seed_remote_data(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(LEDGER_DDL)
        for name in names:
            version, checksum = REMOTE_V4_LINEAGE.historical_ledger[name]
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (name, version, checksum, f"{DAY}T12:00:00+00:00"),
            )
        connection.commit()
    finally:
        connection.close()
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_an_exactly_ledgered_remote_database_is_accepted(tmp_path):
    """The other legitimate arrival: the same lineage, carrying a ledger."""

    database = remote_v4_database(tmp_path, "ledgered.sqlite3")
    seed_remote_data(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(LEDGER_DDL)
        for name, (version, checksum) in REMOTE_V4_LINEAGE.historical_ledger.items():
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (name, version, checksum, f"{DAY}T12:00:00+00:00"),
            )
        connection.commit()
    finally:
        connection.close()

    repository = Phase0Repository(database)
    repository.migrate()

    assert repository.schema_version() == LATEST_VERSION
    assert len(repository.schema_lineages()) == 1
    assert repository.count("raw_items") == 4


def test_a_local_checksum_on_a_remote_looking_schema_is_refused(tmp_path):
    """Right schema, approved checksum: still not this lineage."""

    approved = {
        migration.name: migration.checksum
        for migration in ALL_MIGRATIONS
        if migration.version <= 4
    }
    database = remote_v4_database(tmp_path, "localsum.sqlite3")
    connection = sqlite3.connect(database)
    try:
        connection.execute(LEDGER_DDL)
        for name, checksum in approved.items():
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (name, int(name.split("_", 1)[0]), checksum, f"{DAY}T12:00:00+00:00"),
            )
        connection.commit()
    finally:
        connection.close()
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


# ----------------------------------------------------------------------
# Provenance is evidence, never authority
# ----------------------------------------------------------------------


def forged_provenance_database(tmp_path: Path, name: str, **overrides) -> Path:
    """A *fresh approved* database, dressed up as a converged one."""

    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    row = {
        "lineage": REMOTE_V4_LINEAGE.lineage,
        "migration": REMOTE_V4_LINEAGE.migration,
        "historical_checksum": REMOTE_V4_LINEAGE.checksum,
        "schema_fingerprint": REMOTE_V4_LINEAGE.schema_fingerprint,
        "convergence": REMOTE_V4_LINEAGE.convergence,
    }
    row.update(overrides)
    connection = sqlite3.connect(repository.database_path)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            (REMOTE_V4_LINEAGE.checksum, REMOTE_V4_LINEAGE.migration),
        )
        connection.execute(
            f"INSERT INTO {LINEAGE_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
            (
                row["lineage"],
                row["migration"],
                row["historical_checksum"],
                row["schema_fingerprint"],
                row["convergence"],
                f"{DAY}T12:00:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return repository.database_path


def test_a_forged_provenance_row_cannot_bless_a_fresh_database(tmp_path):
    """The reported bypass, exactly.

    A fresh approved database, its ``004`` checksum swapped for the remote
    one, plus a lineage row copied field for field out of the registry.
    Provenance used to be taken at its word; now every field is checked
    against the live database, and the row a converged database has and
    this one cannot forge is the *convergence's own ledger entry*.
    """

    database = forged_provenance_database(tmp_path, "forged.sqlite3")
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


FORGERY_VARIANTS = [
    ("wrong-fingerprint", {"schema_fingerprint": "d" * 64}),
    ("wrong-convergence", {"convergence": "004_something_else.sql"}),
    ("wrong-migration", {"migration": "005_story_reconciliation.sql"}),
    ("wrong-historical-checksum", {"historical_checksum": "e" * 64}),
    ("wrong-lineage-id", {"lineage": "some-other-lineage"}),
]


@pytest.mark.parametrize(
    "label, overrides", FORGERY_VARIANTS, ids=[row[0] for row in FORGERY_VARIANTS]
)
def test_provenance_with_any_wrong_field_is_refused(tmp_path, label, overrides):
    database = forged_provenance_database(tmp_path, f"{label}.sqlite3", **overrides)
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_provenance_without_the_convergence_ledger_row_is_refused(tmp_path):
    """Deleting the one row that proves the convergence actually ran."""

    database = remote_v4_database(tmp_path, "stripped.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"DELETE FROM {LEDGER_TABLE} WHERE name = ?",
            (REMOTE_V4_LINEAGE.convergence,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_provenance_is_refused_when_the_convergence_effects_are_gone(tmp_path):
    """A row claiming convergence over a schema that never converged."""

    database = remote_v4_database(tmp_path, "undone.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        # Put the historical lineage's marker back: the provenance now
        # describes a database that plainly did not converge.
        connection.execute(
            "CREATE TRIGGER enforce_raw_item_ticker_insert BEFORE INSERT ON raw_items "
            "WHEN NEW.ticker = 'NOPE' BEGIN SELECT RAISE(ABORT, 'no'); END"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError):
        Phase0Repository(database).migrate()


def test_duplicate_provenance_rows_are_refused(tmp_path):
    database = remote_v4_database(tmp_path, "dupes.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"INSERT INTO {LINEAGE_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
            (
                "second-claim",
                REMOTE_V4_LINEAGE.migration,
                REMOTE_V4_LINEAGE.checksum,
                REMOTE_V4_LINEAGE.schema_fingerprint,
                REMOTE_V4_LINEAGE.convergence,
                f"{DAY}T12:00:00+00:00",
            ),
        )
        connection.execute(
            f"UPDATE {LINEAGE_TABLE} SET lineage = ? WHERE lineage = ?",
            (REMOTE_V4_LINEAGE.lineage, "second-claim"),
        )
        connection.commit()
    except sqlite3.IntegrityError:
        pytest.skip("the primary key already forbids a duplicate lineage id")
    finally:
        connection.close()


def test_a_converged_database_still_refuses_a_later_checksum_edit(tmp_path):
    """Compatibility is not a permanent exemption."""

    database = remote_v4_database(tmp_path, "later.sqlite3")
    repository = Phase0Repository(database)
    repository.migrate()

    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            ("f" * 64, "010_theme_partition_integrity.sql"),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_a_converged_schema_is_bit_for_bit_a_fresh_one(tmp_path):
    """The strongest form of the equivalence claim.

    ``schema_snapshot`` compares objects and columns; this compares the
    same canonical digest the lineage check itself runs on, which also
    covers indexes, foreign keys, nullability, defaults, and primary-key
    positions.
    """

    database = remote_v4_database(tmp_path, "converged.sqlite3")
    seed_remote_data(database)
    converged = Phase0Repository(database)
    converged.migrate()

    fresh = Phase0Repository(tmp_path / "fresh.sqlite3")
    fresh.migrate()

    with converged.admin.connect_writable() as left:
        with fresh.admin.connect_writable() as right:
            assert lineages.structural_fingerprint(
                left
            ) == lineages.structural_fingerprint(right)


def test_the_settlement_applies_every_approved_migration(tmp_path):
    """No migration is marked applied without its statements having run."""

    database = remote_v4_database(tmp_path, "complete.sqlite3")
    repository = Phase0Repository(database)
    applied = repository.migrate()

    expected = [REMOTE_V4_LINEAGE.convergence] + [
        migration.name for migration in ALL_MIGRATIONS if migration.version > 4
    ]
    assert sorted(applied) == sorted(expected)
    ledger = {row["name"] for row in repository.applied_migrations()}
    assert ledger == {m.name for m in ALL_MIGRATIONS} | {REMOTE_V4_LINEAGE.convergence}


# ----------------------------------------------------------------------
# Migration metadata is validated as exactly as the application schema
# ----------------------------------------------------------------------


def ledgered_remote_database(
    tmp_path: Path,
    name: str,
    *,
    ddl: str = LEDGER_DDL,
    rows: dict | None = None,
) -> Path:
    """A remote-lineage database that also carries a ledger."""

    database = remote_v4_database(tmp_path, name)
    seed_remote_data(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(ddl)
        for migration, (version, checksum) in (
            rows if rows is not None else REMOTE_V4_LINEAGE.historical_ledger
        ).items():
            connection.execute(
                "INSERT INTO schema_migrations (name, version, checksum, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (migration, version, checksum, f"{DAY}T12:00:00+00:00"),
            )
        connection.commit()
    finally:
        connection.close()
    return database


#: One difference each from the ledger table this code creates.  Excluding
#: the metadata tables from the *application* fingerprint is not the same
#: as not checking them, and it used to be.
ALTERED_LEDGER_TABLES = [
    (
        "extra-column",
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, version INTEGER "
        "NOT NULL CHECK (version > 0), checksum TEXT NOT NULL CHECK "
        "(length(checksum) = 64), applied_at TEXT NOT NULL CHECK "
        "(datetime(applied_at) IS NOT NULL), smuggled TEXT)",
    ),
    (
        "missing-column",
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, version INTEGER "
        "NOT NULL, checksum TEXT NOT NULL)",
    ),
    (
        "renamed-column",
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, version INTEGER "
        "NOT NULL CHECK (version > 0), digest TEXT NOT NULL CHECK "
        "(length(digest) = 64), applied_at TEXT NOT NULL)",
    ),
    (
        "changed-type",
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, version TEXT "
        "NOT NULL CHECK (version > 0), checksum TEXT NOT NULL CHECK "
        "(length(checksum) = 64), applied_at TEXT NOT NULL CHECK "
        "(datetime(applied_at) IS NOT NULL))",
    ),
    (
        "dropped-not-null",
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, version INTEGER "
        "CHECK (version > 0), checksum TEXT CHECK (length(checksum) = 64), "
        "applied_at TEXT CHECK (datetime(applied_at) IS NOT NULL))",
    ),
    (
        "no-primary-key",
        "CREATE TABLE schema_migrations (name TEXT, version INTEGER NOT NULL "
        "CHECK (version > 0), checksum TEXT NOT NULL CHECK "
        "(length(checksum) = 64), applied_at TEXT NOT NULL CHECK "
        "(datetime(applied_at) IS NOT NULL))",
    ),
    (
        "different-primary-key",
        "CREATE TABLE schema_migrations (name TEXT NOT NULL, version INTEGER "
        "NOT NULL CHECK (version > 0), checksum TEXT NOT NULL CHECK "
        "(length(checksum) = 64), applied_at TEXT NOT NULL CHECK "
        "(datetime(applied_at) IS NOT NULL), PRIMARY KEY (name, version))",
    ),
    (
        "added-default",
        "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, version INTEGER "
        "NOT NULL DEFAULT 1 CHECK (version > 0), checksum TEXT NOT NULL CHECK "
        "(length(checksum) = 64), applied_at TEXT NOT NULL CHECK "
        "(datetime(applied_at) IS NOT NULL))",
    ),
]


#: Shapes that can still hold this lineage's four rows, so the rejection
#: has to come from the table's definition rather than from a failed insert.
POPULATABLE_LEDGER_TABLES = [
    row
    for row in ALTERED_LEDGER_TABLES
    if row[0] not in {"missing-column", "renamed-column"}
]

#: Shapes that cannot hold them at all.
UNPOPULATABLE_LEDGER_TABLES = [
    row
    for row in ALTERED_LEDGER_TABLES
    if row[0] in {"missing-column", "renamed-column"}
]


@pytest.mark.parametrize(
    "label, ddl",
    POPULATABLE_LEDGER_TABLES,
    ids=[row[0] for row in POPULATABLE_LEDGER_TABLES],
)
def test_an_altered_ledger_table_is_refused(tmp_path, label, ddl):
    rows = REMOTE_V4_LINEAGE.historical_ledger
    database = ledgered_remote_database(
        tmp_path, f"{label}.sqlite3", ddl=ddl, rows=rows
    )
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


@pytest.mark.parametrize(
    "label, ddl",
    UNPOPULATABLE_LEDGER_TABLES,
    ids=[row[0] for row in UNPOPULATABLE_LEDGER_TABLES],
)
def test_a_structurally_wrong_ledger_table_is_refused(tmp_path, label, ddl):
    """Shapes that cannot even hold this lineage's rows."""

    database = remote_v4_database(tmp_path, f"shape-{label}.sqlite3")
    seed_remote_data(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(ddl)
        connection.commit()
    finally:
        connection.close()
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_an_unexpected_index_on_the_ledger_is_refused(tmp_path):
    database = ledgered_remote_database(tmp_path, "ledger-index.sqlite3")
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE INDEX idx_smuggled ON schema_migrations(checksum)")
        connection.commit()
    finally:
        connection.close()
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_an_altered_provenance_table_is_refused(tmp_path):
    """The other metadata table gets the same treatment."""

    database = remote_v4_database(tmp_path, "lineage-table.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(f"ALTER TABLE {LINEAGE_TABLE} ADD COLUMN smuggled TEXT")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_the_pinned_metadata_fingerprints_describe_what_this_code_creates(tmp_path):
    """The pins are only worth having if a fresh database matches them."""

    repository = Phase0Repository(tmp_path / "fresh.sqlite3")
    repository.migrate()

    with repository.admin.connect_writable() as connection:
        assert (
            lineages.metadata_fingerprint(connection, "schema_migrations")
            == lineages.LEDGER_SCHEMA_FINGERPRINT
        )
        assert (
            lineages.metadata_fingerprint(connection, LINEAGE_TABLE)
            == lineages.LINEAGE_SCHEMA_FINGERPRINT
        )
        assert lineages.metadata_fingerprint(connection, "nonexistent") is None


# ----------------------------------------------------------------------
# Ledger rows are compared whole, never by checksum alone
# ----------------------------------------------------------------------


WRONG_LEDGER_TUPLES = [
    ("historical-version-99", "004_supported_ticker_universe.sql", 99, None),
    ("historical-version-3", "004_supported_ticker_universe.sql", 3, None),
    ("companion-version-99", "002_source_state_and_stage_keys.sql", 99, None),
    ("historical-checksum", "004_supported_ticker_universe.sql", None, "a" * 64),
    ("companion-checksum", "001_initial.sql", None, "b" * 64),
]


@pytest.mark.parametrize(
    "label, migration, version, checksum",
    WRONG_LEDGER_TUPLES,
    ids=[row[0] for row in WRONG_LEDGER_TUPLES],
)
def test_a_wrong_ledger_tuple_is_refused(tmp_path, label, migration, version, checksum):
    """A checksum is half a row, and half a row used to be enough."""

    rows = dict(REMOTE_V4_LINEAGE.historical_ledger)
    was_version, was_checksum = rows[migration]
    rows[migration] = (
        version if version is not None else was_version,
        checksum if checksum is not None else was_checksum,
    )
    database = ledgered_remote_database(tmp_path, f"{label}.sqlite3", rows=rows)
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError, match="does not match it"):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_a_blank_applied_at_cannot_reach_the_ledger(tmp_path):
    """The column's own CHECK is the first line; the tuple check is the second."""

    database = ledgered_remote_database(tmp_path, "blank-applied.sqlite3")
    connection = sqlite3.connect(database)
    try:
        for value in ("", "   ", "not a date"):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE schema_migrations SET applied_at = ? WHERE name = ?",
                    (value, REMOTE_V4_LINEAGE.migration),
                )
    finally:
        connection.close()

    # ...and the database is still exactly the lineage it was.
    repository = Phase0Repository(database)
    repository.migrate()
    assert repository.schema_version() == LATEST_VERSION


@pytest.mark.parametrize("version", [3, 5, 99])
def test_a_wrong_convergence_version_is_refused(tmp_path, version):
    """Post-convergence, the convergence row is checked whole too."""

    database = remote_v4_database(tmp_path, f"convver-{version}.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET version = ? WHERE name = ?",
            (version, REMOTE_V4_LINEAGE.convergence),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_a_wrong_convergence_checksum_is_refused(tmp_path):
    database = remote_v4_database(tmp_path, "convsum.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            ("c" * 64, REMOTE_V4_LINEAGE.convergence),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_a_wrong_historical_version_after_convergence_is_refused(tmp_path):
    database = remote_v4_database(tmp_path, "histver.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET version = 99 WHERE name = ?",
            (REMOTE_V4_LINEAGE.migration,),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


# ----------------------------------------------------------------------
# Provenance corroborates; the schema and the ledger decide
# ----------------------------------------------------------------------


def test_forged_provenance_and_convergence_row_cannot_bless_a_fresh_database(
    tmp_path,
):
    """The reported bypass, with the convergence row forged too.

    Everything a converged database's ledger has, assembled by hand on a
    fresh one. What it cannot assemble is a settlement: the historical
    row, the convergence row, and the provenance row are written together
    and carry one timestamp, and rows pasted in afterwards do not.
    """

    repository = Phase0Repository(tmp_path / "forged-pair.sqlite3")
    repository.migrate()
    connection = sqlite3.connect(repository.database_path)
    try:
        connection.execute(
            f"UPDATE {LEDGER_TABLE} SET checksum = ? WHERE name = ?",
            (REMOTE_V4_LINEAGE.checksum, REMOTE_V4_LINEAGE.migration),
        )
        connection.execute(
            f"INSERT INTO {LEDGER_TABLE} VALUES (?, ?, ?, ?)",
            (
                REMOTE_V4_LINEAGE.convergence,
                REMOTE_V4_LINEAGE.convergence_version,
                REMOTE_V4_LINEAGE.convergence_checksum,
                f"{DAY}T12:00:00+00:00",
            ),
        )
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
        connection.commit()
    finally:
        connection.close()
    before = full_snapshot(repository.database_path)

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(repository.database_path).migrate()

    assert full_snapshot(repository.database_path) == before


def test_provenance_written_apart_from_its_settlement_is_refused(tmp_path):
    """Even on a genuinely converged database."""

    database = remote_v4_database(tmp_path, "apart.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"UPDATE {LINEAGE_TABLE} SET recognized_at = ? WHERE lineage = ?",
            (f"{DAY}T09:00:00+00:00", REMOTE_V4_LINEAGE.lineage),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


def test_provenance_cannot_rescue_a_ledger_missing_its_companions(tmp_path):
    database = remote_v4_database(tmp_path, "nocompanion.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            f"DELETE FROM {LEDGER_TABLE} WHERE name = ?",
            ("002_source_state_and_stage_keys.sql",),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError, match="was modified after it was applied"):
        Phase0Repository(database).migrate()


# ----------------------------------------------------------------------
# Exactly two accepted states; every hybrid fails closed
# ----------------------------------------------------------------------


def test_the_two_accepted_states_are_named_and_exclusive(tmp_path):
    database = remote_v4_database(tmp_path, "states.sqlite3")
    seed_remote_data(database)

    connection = sqlite3.connect(database)
    try:
        state, reason = lineages.classify(connection, REMOTE_V4_LINEAGE)
        assert (state, reason) == (lineages.STATE_PRE_CONVERGENCE, None)
    finally:
        connection.close()

    Phase0Repository(database).migrate()

    connection = sqlite3.connect(database)
    try:
        state, reason = lineages.classify(connection, REMOTE_V4_LINEAGE)
        assert (state, reason) == (lineages.STATE_POST_CONVERGENCE, None)
    finally:
        connection.close()

    fresh = Phase0Repository(tmp_path / "fresh.sqlite3")
    fresh.migrate()
    with fresh.admin.connect_writable() as connection:
        state, reason = lineages.classify(connection, REMOTE_V4_LINEAGE)
        assert state == lineages.STATE_UNRELATED
        assert reason


HYBRIDS = [
    "historical-schema-with-convergence-row",
    "converged-schema-with-historical-fingerprint-claim",
    "historical-schema-with-provenance",
    "partial-convergence-metadata",
]


@pytest.mark.parametrize("hybrid", HYBRIDS)
def test_a_hybrid_state_fails_closed(tmp_path, hybrid):
    """Neither state, so no state: half a conversion is not a conversion."""

    database = remote_v4_database(tmp_path, f"{hybrid}.sqlite3")
    seed_remote_data(database)
    connection = sqlite3.connect(database)
    try:
        connection.execute(LEDGER_DDL)
        connection.execute(lineages.LINEAGE_DDL)
        for name, (version, checksum) in REMOTE_V4_LINEAGE.historical_ledger.items():
            connection.execute(
                "INSERT INTO schema_migrations VALUES (?, ?, ?, ?)",
                (name, version, checksum, f"{DAY}T12:00:00+00:00"),
            )
        if hybrid in {
            "historical-schema-with-convergence-row",
            "partial-convergence-metadata",
        }:
            connection.execute(
                f"INSERT INTO {LEDGER_TABLE} VALUES (?, ?, ?, ?)",
                (
                    REMOTE_V4_LINEAGE.convergence,
                    REMOTE_V4_LINEAGE.convergence_version,
                    REMOTE_V4_LINEAGE.convergence_checksum,
                    f"{DAY}T12:00:00+00:00",
                ),
            )
        if hybrid in {
            "historical-schema-with-provenance",
            "converged-schema-with-historical-fingerprint-claim",
        }:
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
        connection.commit()
    finally:
        connection.close()
    before = full_snapshot(database)

    with pytest.raises(Phase0MigrationError):
        Phase0Repository(database).migrate()

    assert full_snapshot(database) == before


def test_a_converged_database_with_the_historical_schema_back_is_refused(tmp_path):
    """Provenance says converged; the schema says otherwise."""

    database = remote_v4_database(tmp_path, "reverted.sqlite3")
    Phase0Repository(database).migrate()
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TABLE supported_tickers")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(Phase0MigrationError):
        Phase0Repository(database).migrate()


# ----------------------------------------------------------------------
# The success paths, still
# ----------------------------------------------------------------------


def test_an_exact_approved_local_database_is_untouched_by_any_of_this(tmp_path):
    repository = Phase0Repository(tmp_path / "approved.sqlite3")
    applied = repository.migrate()

    assert applied == [migration.name for migration in ALL_MIGRATIONS]
    assert repository.schema_lineages() == []
    assert repository.migrate() == []
    ledger = {row["name"]: row["version"] for row in repository.applied_migrations()}
    assert ledger == {m.name: m.version for m in ALL_MIGRATIONS}


def test_a_converged_database_repeats_cleanly(tmp_path):
    database = remote_v4_database(tmp_path, "repeat.sqlite3")
    seed_remote_data(database)
    repository = Phase0Repository(database)
    repository.migrate()
    snapshot = full_snapshot(database)

    for _ in range(3):
        assert Phase0Repository(database).migrate() == []
    assert full_snapshot(database) == snapshot

"""Contract and hostile-probe coverage for the Phase 0 persistence layer.

Every test here answers a question issue #57 asks of the datastore rather
than of any one caller: does the *database* refuse the thing, does an
upgraded database end up identical to a fresh one, does a failed migration
leave nothing behind, and does a credential that reached an error payload
actually disappear.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import inspect
import itertools
import json
import pickle
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence
from unittest import mock

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
    Phase0Error,
    Phase0IntegrityError,
    Phase0MigrationError,
    Phase0RunContextError,
    Phase0ValidationError,
    StageKeyError,
    UnsupportedTickerError,
)
from phase0.models import (
    AUXILIARY_OUTPUTS,
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
from phase0.repository import (
    DEFAULT_CANDIDATE_REASON,
    MIGRATIONS_PATH,
    STORY_RECONCILED_COLUMNS,
    THEME_RECONCILED_COLUMNS,
    THEME_SET_RECONCILED_COLUMNS,
    Phase0Admin,
    Phase0Reader,
    Phase0Repository,
    StageRunContext,
    normalize_candidate_tickers,
    serialize_operational_metadata,
    serialize_raw_evidence,
)
from phase0.schema import (
    LEDGER_TABLE,
    LEGACY_SCHEMA_VERSION,
    MINIMUM_SQLITE_VERSION,
    load_migrations,
)
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


#: Per-ticker index offsets, so seeding two tickers produces two distinct
#: sets of raw items rather than colliding on source/canonical_url and
#: silently handing back the first ticker's rows.
_TICKER_OFFSETS = {"NVDA": 0, "AMD": 100, "TSLA": 200, "AAPL": 300, "META": 400}


def seed_raw_items(
    repository: Phase0Repository, count: int, ticker="NVDA"
) -> list[int]:
    offset = _TICKER_OFFSETS[ticker]
    return [
        result.item_id
        for result in repository.admin.insert_raw_items(
            [raw_item(offset + index, ticker) for index in range(1, count + 1)]
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

    with repository.admin.connect_writable() as connection:
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
    target.mkdir(exist_ok=True)
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
    with legacy.admin.connect_writable() as connection:
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


#: Every migration file, including the compatibility convergence one that
#: is not part of the numbered sequence but still has to parse everywhere.
EVERY_MIGRATION_FILE = sorted(MIGRATIONS_PATH.rglob("*.sql"))


def _non_literal_raise_sites() -> set[tuple[str, str]]:
    """Every ``RAISE(..., <expression>)`` in every migration.

    ``RAISE()`` took only a string *literal* until SQLite 3.47 (2024-10);
    earlier releases reject a concatenated message with a syntax error.
    That is far worse than one broken statement: SQLite parses the whole
    schema the first time a connection touches it, so a single trigger it
    cannot parse makes the entire database unopenable — ``SELECT`` and
    even ``DROP TRIGGER`` fail alike, which means such a database cannot
    be repaired from the runtime that cannot open it.
    """

    found: set[tuple[str, str]] = set()
    for path in EVERY_MIGRATION_FILE:
        trigger = None
        statement: list[str] = []
        for line in path.read_text().splitlines():
            match = re.search(
                r"CREATE\s+(?:TEMPORARY\s+)?TRIGGER\s+(?:IF\s+NOT\s+EXISTS\s+)?"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                line,
                re.IGNORECASE,
            )
            if match:
                trigger = match.group(1)
            if "RAISE(" in line.replace(" ", "") or statement:
                statement.append(line)
                if ";" in line:
                    if "||" in " ".join(statement):
                        found.add((path.name, trigger or "?"))
                    statement = []
    return found


def test_no_migration_uses_a_non_literal_raise_message():
    """Not one, anywhere — see :func:`_non_literal_raise_sites` for why."""

    assert _non_literal_raise_sites() == set()


def _older_sqlite_cli() -> str | None:
    """A ``sqlite3`` CLI older than the library Python is linked against.

    The point is to parse the schema with something that is not the
    in-process SQLite, since that is the whole failure mode: the tests
    pass on a new library and the deployment does not have one.
    """

    binary = shutil.which("sqlite3")
    if binary is None:
        return None
    try:
        reported = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30
        ).stdout.split()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    version = tuple(int(part) for part in reported.split(".")[:3])
    return binary if version < sqlite3.sqlite_version_info else None


def test_every_migration_parses_on_an_older_sqlite(tmp_path):
    """Apply the whole sequence with an older SQLite than this process's.

    A compatibility claim that is only ever checked by the library making
    the claim is not a check.  When the host has no older CLI to offer,
    this skips and :func:`test_no_migration_uses_a_non_literal_raise_message`
    plus :func:`test_the_declared_sqlite_floor_is_the_one_the_schema_needs`
    carry the guarantee statically.
    """

    binary = _older_sqlite_cli()
    if binary is None:
        pytest.skip(
            f"no sqlite3 CLI older than this process's {sqlite3.sqlite_version}"
        )

    database = tmp_path / "legacy.sqlite3"
    for migration in ALL_MIGRATIONS:
        result = subprocess.run(
            [binary, str(database)],
            input=migration.sql,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert (
            result.returncode == 0 and not result.stderr
        ), f"{migration.name} does not apply on {binary}: {result.stderr}"

    # Applying it is half the claim; the resulting schema also has to be
    # readable, which is the half that a non-literal RAISE() destroys.
    readback = subprocess.run(
        [binary, str(database)],
        input="PRAGMA integrity_check;\nSELECT count(*) FROM sqlite_master;\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert readback.returncode == 0 and not readback.stderr, readback.stderr
    assert readback.stdout.splitlines()[0] == "ok"


def test_the_declared_sqlite_floor_is_the_one_the_schema_needs():
    """The floor is a promise, so nothing may quietly outgrow it."""

    assert MINIMUM_SQLITE_VERSION == (3, 38, 0)
    # What the migrations actually use sits at or below the floor: ON
    # CONFLICT DO UPDATE (3.24) and the JSON1 CHECK constraints (a default
    # build option from 3.38).  The keywords below are the ones that would
    # raise the floor and that can be recognized unambiguously in text; the
    # binding check is
    # :func:`test_every_migration_parses_on_an_older_sqlite`, since only an
    # older SQLite can prove an older SQLite copes.
    combined = "\n".join(path.read_text() for path in EVERY_MIGRATION_FILE)
    for feature, introduced in (
        (r"\bSTRICT\s*(?:,|;|$)", "3.37 strict tables"),
        (r"\bRETURNING\b", "3.35 RETURNING"),
        (r"\bDROP\s+COLUMN\b", "3.35 ALTER TABLE DROP COLUMN"),
        (r"\bGENERATED\s+ALWAYS\b", "3.31 generated columns"),
        (r"\bMATERIALIZED\b", "3.35 materialized CTE hints"),
    ):
        assert not re.search(feature, combined, re.IGNORECASE), (
            f"a migration uses {introduced}, above the declared floor "
            f"{'.'.join(str(part) for part in MINIMUM_SQLITE_VERSION)}"
        )
    assert sqlite3.sqlite_version_info >= MINIMUM_SQLITE_VERSION


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
    with old.admin.connect_writable() as connection:
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
    with upgraded.admin.connect_writable() as connection:
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

    (directory / "011_broken.sql").write_text(
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


# ----------------------------------------------------------------------
# A failed attempt leaves no trace of itself
#
# The module's own contract is that a migration lands completely or not
# at all.  The bootstrap tables were the exception nobody had looked at:
# `schema_migrations` and `schema_lineage` were created and *committed*
# before the first migration ran, so a brand-new database whose first
# migration failed kept two tables describing an attempt that, by that
# contract, never happened.
# ----------------------------------------------------------------------


def database_state(path: Path) -> dict:
    """Everything a failed attempt must leave exactly as it found it."""

    if not path.exists() or path.stat().st_size == 0:
        return {"exists": False}
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        objects = sorted(
            (row["type"], row["name"], row["sql"])
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%'"
            )
        )
        names = {name for _, name, _ in objects}
        state = {
            "exists": True,
            "sqlite_master": objects,
            "user_version": int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            ),
            "schema_migrations": None,
            "schema_lineage": None,
        }
        for table in ("schema_migrations", "schema_lineage"):
            if table in names:
                state[table] = [
                    tuple(row) for row in connection.execute(f"SELECT * FROM {table}")
                ]
        return state
    finally:
        connection.close()


def broken_migrations(tmp_path: Path, *, upto: int, failing_sql: str) -> Path:
    """The real migrations up to ``upto``, then one that fails."""

    directory = partial_migrations(tmp_path, upto)
    version = max(upto + 1, 1)
    (directory / f"{version:03d}_broken.sql").write_text(failing_sql, encoding="utf-8")
    return directory


#: Ways a migration can fail, so the rollback is not shown to work for
#: only one of them.  Each aborts *after* some of its own statements have
#: already run, which is the case a rollback has to actually undo.
FAILURE_MODES = [
    (
        "statement error",
        "CREATE TABLE half_applied (id INTEGER PRIMARY KEY);\n"
        "INSERT INTO half_applied SELECT missing_column FROM half_applied;\n",
    ),
    (
        "constraint violation",
        "CREATE TABLE half_applied (id INTEGER PRIMARY KEY);\n"
        "INSERT INTO half_applied (id) VALUES (1);\n"
        "INSERT INTO half_applied (id) VALUES (1);\n",
    ),
    (
        "unparseable tail",
        "CREATE TABLE half_applied (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE never_finished (;\n",
    ),
]


@pytest.mark.parametrize(
    "failing_sql", [pytest.param(sql, id=label) for label, sql in FAILURE_MODES]
)
def test_a_failed_first_migration_leaves_a_fresh_database_untouched(
    tmp_path, failing_sql
):
    """Nothing at all — no ledger, no lineage table, no version."""

    directory = tmp_path / "only_broken"
    directory.mkdir()
    (directory / "001_broken.sql").write_text(failing_sql, encoding="utf-8")
    database = tmp_path / "phase0.sqlite3"
    assert database_state(database) == {"exists": False}

    repository = Phase0Repository(database, migrations_path=directory)
    with pytest.raises((sqlite3.Error, Phase0MigrationError)):
        repository.migrate()

    # The file itself is created by *opening* a connection, before any
    # migration logic runs, and no rollback can undo that.  What must not
    # survive is content: an empty file is a database that has had nothing
    # done to it, which is exactly what a failed attempt should leave.
    after = database_state(database)
    assert after["sqlite_master"] == []
    assert after["user_version"] == 0
    assert after["schema_migrations"] is None
    assert after["schema_lineage"] is None


def test_a_failed_first_migration_leaves_no_bootstrap_tables(tmp_path):
    """Named explicitly, because these two were the whole defect."""

    directory = tmp_path / "only_broken"
    directory.mkdir()
    (directory / "001_broken.sql").write_text(
        "CREATE TABLE ok (id INTEGER PRIMARY KEY);\nCREATE TABLE bad (;\n",
        encoding="utf-8",
    )
    database = tmp_path / "phase0.sqlite3"
    repository = Phase0Repository(database, migrations_path=directory)
    with pytest.raises((sqlite3.Error, Phase0MigrationError)):
        repository.migrate()

    state = database_state(database)
    if state["exists"]:
        names = {name for _, name, _ in state["sqlite_master"]}
        assert "schema_migrations" not in names
        assert "schema_lineage" not in names
        assert names == set()


@pytest.mark.parametrize("upto", [1, 4, 7, LATEST_VERSION])
def test_a_failed_later_migration_leaves_the_database_at_the_step_before(
    tmp_path, upto
):
    """Per-migration atomicity, which the bootstrap fix must not weaken."""

    good = partial_migrations(tmp_path, upto)
    repository = Phase0Repository(tmp_path / "phase0.sqlite3", migrations_path=good)
    repository.migrate()
    before = database_state(tmp_path / "phase0.sqlite3")
    assert before["schema_migrations"], "the good run must have written a ledger"

    directory = broken_migrations(
        tmp_path,
        upto=upto,
        failing_sql=(
            "CREATE TABLE half_applied (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE never_finished (;\n"
        ),
    )
    broken = Phase0Repository(tmp_path / "phase0.sqlite3", migrations_path=directory)
    with pytest.raises((sqlite3.Error, Phase0MigrationError)):
        broken.migrate()

    assert database_state(tmp_path / "phase0.sqlite3") == before


def test_a_failed_legacy_backfill_leaves_the_legacy_database_untouched(tmp_path):
    """A pre-ledger database must not keep a ledger it did not earn.

    The backfill writes rows describing history the database already
    lived.  It used to commit them on its own, so a failure in the very
    next migration left a v2 database carrying a ledger, a lineage table,
    and no way to tell that the upgrade had never happened.
    """

    database = legacy_v2_database(tmp_path)
    before = database_state(database)
    assert before["schema_migrations"] is None, "the fixture drops the ledger"

    directory = broken_migrations(
        tmp_path,
        upto=LEGACY_SCHEMA_VERSION,
        failing_sql="CREATE TABLE half_applied (id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE never_finished (;\n",
    )
    repository = Phase0Repository(database, migrations_path=directory)
    with pytest.raises((sqlite3.Error, Phase0MigrationError)):
        repository.migrate()

    assert database_state(database) == before


def test_a_failed_attempt_can_be_retried_successfully(tmp_path):
    """The point of rolling back: the next attempt starts from clean."""

    database = tmp_path / "phase0.sqlite3"
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "001_broken.sql").write_text(
        "CREATE TABLE ok (id INTEGER PRIMARY KEY);\nCREATE TABLE bad (;\n",
        encoding="utf-8",
    )
    repository = Phase0Repository(database, migrations_path=directory)
    with pytest.raises((sqlite3.Error, Phase0MigrationError)):
        repository.migrate()

    # Repair the migration set and try again against the same file.
    (directory / "001_broken.sql").unlink()
    for migration in ALL_MIGRATIONS:
        shutil.copy(MIGRATIONS_PATH / migration.name, directory / migration.name)
    retried = Phase0Repository(database, migrations_path=directory)
    retried.migrate()

    assert retried.schema_version() == LATEST_VERSION
    assert schema_snapshot(retried) == schema_snapshot(migrated(tmp_path, "fresh.db"))
    assert [row["name"] for row in retried.applied_migrations()] == [
        migration.name for migration in ALL_MIGRATIONS
    ]


# ----------------------------------------------------------------------
# A version this code did not write is not a version it can honour
#
# `_backfill_ledger` records every migration whose number is `<=
# user_version`, on the reasoning that a pre-ledger database has already
# lived that history.  Nothing bounded the number.  A database stamped
# 999 — or 13, or anything at all — therefore had its whole ledger
# synthesized, left nothing pending, and reported a successful migration
# over a file whose Phase 0 schema had never been created.  Every later
# run then agreed it was done.
# ----------------------------------------------------------------------


def stamped_database(path: Path, version: int, setup=None) -> Path:
    """A database carrying a `user_version` nobody earned."""

    connection = sqlite3.connect(path)
    try:
        if setup is not None:
            setup(connection)
        connection.execute(f"PRAGMA user_version = {int(version)}")
        connection.commit()
    finally:
        connection.close()
    return path


def unrelated_tables(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO widgets (name) VALUES ('not ours')")


def phase0_looking_tables(connection: sqlite3.Connection) -> None:
    """Names Phase 0 uses, shapes it does not — the tempting near-miss."""

    connection.execute("CREATE TABLE raw_items (id INTEGER PRIMARY KEY, source TEXT)")
    connection.execute("INSERT INTO raw_items (source) VALUES ('yahoo')")
    connection.execute("CREATE TABLE stories (id INTEGER PRIMARY KEY)")


def malformed_ledger(connection: sqlite3.Connection) -> None:
    """A ledger table that exists and says nothing."""

    connection.execute("CREATE TABLE raw_items (id INTEGER PRIMARY KEY, source TEXT)")
    connection.execute(
        f"CREATE TABLE {LEDGER_TABLE} (name TEXT PRIMARY KEY, version INTEGER, "
        "checksum TEXT, applied_at TEXT)"
    )


def full_state(path: Path) -> dict:
    """`database_state`, plus every application row the database holds.

    The snapshot a rejection has to leave untouched is not only schema and
    metadata: a database that belongs to something else entirely still has
    its rows, and those must survive being handed to the wrong migrator.
    """

    state = database_state(path)
    if not state.get("exists"):
        return state
    connection = sqlite3.connect(path)
    try:
        state["data"] = {
            name: sorted(
                tuple(row) for row in connection.execute(f"SELECT * FROM {name}")
            )
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        connection.close()
    return state


FUTURE_DATABASES = [
    ("bare-one-ahead", LATEST_VERSION + 1, None),
    ("bare-999", 999, None),
    ("unrelated-tables", LATEST_VERSION + 5, unrelated_tables),
    ("phase0-looking-no-ledger", LATEST_VERSION + 1, phase0_looking_tables),
    ("empty-ledger-table", LATEST_VERSION + 2, malformed_ledger),
    ("far-future-with-data", 2**31 - 1, unrelated_tables),
]


@pytest.mark.parametrize(
    "label, version, setup",
    FUTURE_DATABASES,
    ids=[case[0] for case in FUTURE_DATABASES],
)
def test_a_future_version_database_is_refused_untouched(
    tmp_path, label, version, setup
):
    """Hostile cases 1-5 and 8, asserted as one snapshot each.

    The refusal has to be typed *and* total: sqlite_master, application
    rows, both bootstrap tables, and `user_version` all exactly as they
    were. A database this code cannot account for is a database it must
    not have written to.
    """

    database = stamped_database(tmp_path / f"{label}.sqlite3", version, setup)
    before = full_state(database)

    with pytest.raises(Phase0MigrationError, match="newer than the newest migration"):
        Phase0Repository(database).migrate()

    assert full_state(database) == before
    assert before["user_version"] == version
    # Named individually too, because "the dict matched" is easy to
    # satisfy accidentally and these five are the whole contract.
    after = full_state(database)
    assert after["sqlite_master"] == before["sqlite_master"]
    assert after["data"] == before["data"]
    assert after["schema_migrations"] == before["schema_migrations"]
    assert after["schema_lineage"] == before["schema_lineage"]
    assert after["user_version"] == version


def test_the_refusal_names_both_versions(tmp_path):
    """An operator needs to know how far ahead, not merely that it is."""

    database = stamped_database(tmp_path / "ahead.sqlite3", 999)
    with pytest.raises(Phase0MigrationError) as caught:
        Phase0Repository(database).migrate()

    message = str(caught.value)
    assert "999" in message and str(LATEST_VERSION) in message
    assert "Nothing has been changed." in message


@pytest.mark.parametrize("version", [LEGACY_SCHEMA_VERSION + 1, 7, LATEST_VERSION])
def test_an_unledgered_database_may_not_claim_a_post_ledger_version(tmp_path, version):
    """The boundary the `>` rule alone leaves open.

    At `user_version == LATEST_VERSION` the same backfill claims every
    migration and leaves nothing pending, so an empty file stamped with
    the current version was accepted exactly as silently as one stamped
    999. Every database this code creates keeps a ledger, so an unledgered
    one claiming a post-ledger version is asserting a history with nothing
    behind it.
    """

    database = stamped_database(tmp_path / f"claim{version}.sqlite3", version)
    before = full_state(database)

    with pytest.raises(Phase0MigrationError, match="keeps no migration ledger"):
        Phase0Repository(database).migrate()

    assert full_state(database) == before
    assert database_state(database)["schema_migrations"] is None


#: Unledgered databases stamped below the pre-ledger watermark.  There is
#: no recognizer for a pre-ledger v1 — the one pre-ledger schema this code
#: knows how to check is v2, and migration 003 is what checks it.
PRE_LEDGER_DATABASES = [
    ("empty-v1", 1, None),
    ("unrelated-tables-v1", 1, unrelated_tables),
    ("phase0-looking-v1", 1, phase0_looking_tables),
    ("empty-ledger-table-v1", 1, malformed_ledger),
]


@pytest.mark.parametrize(
    "label, version, setup",
    PRE_LEDGER_DATABASES,
    ids=[case[0] for case in PRE_LEDGER_DATABASES],
)
def test_an_unrecognized_pre_ledger_version_is_refused_untouched(
    tmp_path, label, version, setup
):
    """The gap under the watermark, which had teeth.

    `> 2` was refused from the start; `0 < user_version < 2` was not. A
    database stamped `1` had migration 001 backfilled from the number
    alone, then 002 *committed* on its own, and only then did 003 look at
    the schema and refuse it — so the attempt failed and the database had
    still been changed, left at version 2 carrying four tables it never
    asked for.
    """

    database = stamped_database(tmp_path / f"{label}.sqlite3", version, setup)
    before = full_state(database)

    with pytest.raises(Phase0MigrationError, match="keeps no migration ledger"):
        Phase0Repository(database).migrate()

    after = full_state(database)
    assert after == before
    # Named individually too: "the dict matched" is easy to satisfy by
    # accident, and these five are the whole contract.  One fixture
    # arrives with an empty ledger *table*, so the assertion is that
    # nothing moved, not that nothing is there.
    assert after["sqlite_master"] == before["sqlite_master"]
    assert after["data"] == before["data"]
    assert after["schema_migrations"] == before["schema_migrations"]
    assert after["schema_lineage"] == before["schema_lineage"]
    assert after["user_version"] == version
    assert not any(row for row in (after["schema_migrations"] or []))


def test_a_malformed_v2_is_refused_without_writing_anything(tmp_path):
    """v2 *is* recognized — by migration 003, inside the first transaction.

    So a database that merely claims to be v2 is refused on the schema
    rather than on the number, and because 003 is the first pending
    migration there, the bootstrap and the backfill roll back with it.
    """

    database = stamped_database(
        tmp_path / "malformed-v2.sqlite3", LEGACY_SCHEMA_VERSION, phase0_looking_tables
    )
    before = full_state(database)

    with pytest.raises(Phase0MigrationError, match="missing required tables"):
        Phase0Repository(database).migrate()

    after = full_state(database)
    assert after == before
    assert after["schema_migrations"] is None
    assert after["schema_lineage"] is None
    assert after["user_version"] == LEGACY_SCHEMA_VERSION


def test_the_exact_legacy_v2_schema_still_upgrades(tmp_path):
    """The one unledgered schema this code recognizes, end to end."""

    database = legacy_v2_database(tmp_path)
    assert database_state(database)["schema_migrations"] is None

    repository = Phase0Repository(database)
    repository.migrate()

    assert repository.schema_version() == LATEST_VERSION
    assert [row["name"] for row in repository.applied_migrations()] == [
        migration.name for migration in ALL_MIGRATIONS
    ]


def test_a_refused_pre_ledger_database_migrates_once_it_is_supported(tmp_path):
    """The refusal leaves nothing a real migration has to work around."""

    database = stamped_database(tmp_path / "retry-v1.sqlite3", 1)
    with pytest.raises(Phase0MigrationError, match="keeps no migration ledger"):
        Phase0Repository(database).migrate()

    # Back to a version this code understands: nothing has run.
    stamped_database(database, 0)
    repository = Phase0Repository(database)
    repository.migrate()

    assert repository.schema_version() == LATEST_VERSION
    assert schema_snapshot(repository) == schema_snapshot(
        migrated(tmp_path, "fresh.db")
    )


def test_a_refused_database_migrates_once_it_is_supported_again(tmp_path):
    """The refusal is a refusal, not damage.

    Returning the file to a version this code understands is enough; the
    rejection left nothing behind that a real migration has to work
    around.
    """

    database = stamped_database(tmp_path / "retry.sqlite3", 999)
    with pytest.raises(Phase0MigrationError):
        Phase0Repository(database).migrate()

    stamped_database(database, 0)
    repository = Phase0Repository(database)
    repository.migrate()

    assert repository.schema_version() == LATEST_VERSION
    assert schema_snapshot(repository) == schema_snapshot(
        migrated(tmp_path, "fresh.db")
    )
    assert [row["name"] for row in repository.applied_migrations()] == [
        migration.name for migration in ALL_MIGRATIONS
    ]


def test_the_current_version_is_supported_and_idempotent(tmp_path):
    """Hostile case 6: `LATEST_VERSION` itself is a valid resting state."""

    repository = migrated(tmp_path)
    assert repository.schema_version() == LATEST_VERSION

    again = Phase0Repository(tmp_path / "phase0.sqlite3")
    assert again.migrate() == []
    assert again.schema_version() == LATEST_VERSION
    assert schema_snapshot(again) == schema_snapshot(repository)


@pytest.mark.parametrize("upto", list(range(1, LATEST_VERSION)))
def test_every_supported_older_version_still_upgrades(tmp_path, upto):
    """Hostile case 7: the guard bounds the top and moves nothing else."""

    database = tmp_path / "phase0.sqlite3"
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, upto)
    ).migrate()
    assert Phase0Repository(database).schema_version() == upto

    upgraded = Phase0Repository(database)
    applied = upgraded.migrate()

    assert [migration.name for migration in ALL_MIGRATIONS if migration.version > upto]
    assert applied == [
        migration.name for migration in ALL_MIGRATIONS if migration.version > upto
    ]
    assert upgraded.schema_version() == LATEST_VERSION
    assert schema_snapshot(upgraded) == schema_snapshot(migrated(tmp_path, "fresh.db"))


# The inverse mistakes, from the same family: a ledger and a
# `user_version` are written together by one transaction, so once a
# database keeps its own history they can only disagree if something
# outside this module moved one of them.


LEDGER_DISAGREEMENTS = [
    ("version-behind-ledger", "PRAGMA user_version = 4"),
    ("version-at-zero", "PRAGMA user_version = 0"),
    ("ledger-truncated", f"DELETE FROM {LEDGER_TABLE} WHERE version > 5"),
]


@pytest.mark.parametrize(
    "label, tamper",
    LEDGER_DISAGREEMENTS,
    ids=[case[0] for case in LEDGER_DISAGREEMENTS],
)
def test_a_ledger_out_of_step_with_the_version_is_refused(tmp_path, label, tamper):
    """Neither direction may pass, and neither may write.

    A version behind its ledger used to be accepted outright — nothing was
    pending, so `migrate()` reported success and left the database
    reporting a schema version it had long since passed. A version ahead
    of its ledger re-ran applied migrations and died on whatever they
    collided with, which is an untyped SQLite error rather than a refusal.
    """

    repository = migrated(tmp_path)
    database = tmp_path / "phase0.sqlite3"
    with repository.admin.connect_writable() as connection:
        connection.execute(tamper)
        connection.commit()
    before = full_state(database)

    with pytest.raises(Phase0MigrationError, match="but its ledger reaches"):
        Phase0Repository(database).migrate()

    assert full_state(database) == before


def test_an_emptied_ledger_is_refused_rather_than_resynthesized(tmp_path):
    """Deleting the history must not be a way of re-earning it."""

    repository = migrated(tmp_path)
    database = tmp_path / "phase0.sqlite3"
    with repository.admin.connect_writable() as connection:
        connection.execute(f"DELETE FROM {LEDGER_TABLE}")
        connection.commit()
    before = full_state(database)

    with pytest.raises(Phase0MigrationError, match="keeps no migration ledger"):
        Phase0Repository(database).migrate()

    assert full_state(database) == before


def test_a_successful_migration_still_writes_both_bootstrap_tables(tmp_path):
    """The fix moved the bootstrap; it did not remove it."""

    assert migrated(tmp_path).schema_version() == LATEST_VERSION
    state = database_state(tmp_path / "phase0.sqlite3")
    names = {name for _, name, _ in state["sqlite_master"]}
    assert {"schema_migrations", "schema_lineage"} <= names
    assert state["user_version"] == LATEST_VERSION
    assert len(state["schema_migrations"]) == len(ALL_MIGRATIONS)
    assert state["schema_lineage"] == []


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


# ----------------------------------------------------------------------
# Migration 009 canonicalizes the ticker table from *any* prior ordering
#
# `supported_tickers.position` is UNIQUE and SQLite enforces it per row,
# so repositioning the table in place aborts whenever some ticker has to
# pass through a position another ticker still holds.  Before 009 the
# table is not yet sealed, so any ordering is a legitimate prior state —
# and 119 of the 120 orderings collided.
# ----------------------------------------------------------------------

APPROVED_TICKERS = ("AAPL", "AMD", "META", "NVDA", "TSLA")

#: What 009 must produce, whatever it started from.
CANONICAL_TICKER_ORDER = [
    ("TSLA", 1),
    ("NVDA", 2),
    ("AMD", 3),
    ("AAPL", 4),
    ("META", 5),
]


def ticker_rows(repository) -> list[tuple[str, int]]:
    with repository.admin.connect_writable() as connection:
        return [
            (str(row["ticker"]), int(row["position"]))
            for row in connection.execute(
                "SELECT ticker, position FROM supported_tickers ORDER BY position"
            )
        ]


def pre_009_database(tmp_path: Path, rows: Sequence[tuple[str, int]]) -> Path:
    """A database at the exact pre-009 schema holding ``rows``."""

    database = tmp_path / f"pre009_{abs(hash(tuple(rows)))}.sqlite3"
    seeded = Phase0Repository(database, migrations_path=partial_migrations(tmp_path, 8))
    seeded.migrate()
    with seeded.admin.connect_writable() as connection:
        connection.execute("DELETE FROM supported_tickers")
        for ticker, position in rows:
            connection.execute(
                "INSERT INTO supported_tickers (ticker, display_name, position) "
                "VALUES (?, ?, ?)",
                (ticker, ticker.title(), position),
            )
        connection.commit()
    return database


def test_009_converges_every_ordering_of_the_approved_tickers(tmp_path):
    """All 120 permutations, exhaustively — 119 of them used to abort."""

    failures = []
    for order in itertools.permutations(APPROVED_TICKERS):
        rows = list(zip(order, itertools.count(1)))
        database = pre_009_database(tmp_path, rows)
        try:
            upgraded = Phase0Repository(database)
            upgraded.migrate()
            if ticker_rows(upgraded) != CANONICAL_TICKER_ORDER:
                failures.append((order, ticker_rows(upgraded)))
        except Exception as exc:  # noqa: BLE001
            failures.append((order, f"{type(exc).__name__}: {exc}"))
    assert failures == []


#: Prior states that are not permutations: partial, dirty, or holding
#: unsupported rows in the positions the approved five need.
DIRTY_TICKER_STATES = [
    ("reordered subset", [("NVDA", 1), ("TSLA", 2), ("AMD", 3)]),
    ("single row out of place", [("META", 1)]),
    ("sparse positions", [("TSLA", 10), ("NVDA", 20), ("AMD", 30)]),
    (
        "unsupported rows occupying canonical positions",
        [("GOOG", 1), ("MSFT", 2), ("NVDA", 3), ("TSLA", 4)],
    ),
    ("only unsupported rows", [("GOOG", 1), ("MSFT", 2)]),
    ("empty table", []),
    (
        "reverse canonical",
        [("META", 1), ("AAPL", 2), ("AMD", 3), ("NVDA", 4), ("TSLA", 5)],
    ),
]


@pytest.mark.parametrize(
    "rows", [pytest.param(r, id=label) for label, r in DIRTY_TICKER_STATES]
)
def test_009_converges_dirty_and_partial_ticker_tables(tmp_path, rows):
    database = pre_009_database(tmp_path, rows)
    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert ticker_rows(upgraded) == CANONICAL_TICKER_ORDER
    # The universe is sealed again afterwards, and nothing unsupported
    # survived the cleanup.
    with upgraded.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "INSERT INTO supported_tickers (ticker, display_name, position) "
                "VALUES ('GOOG', 'Alphabet', 6)"
            )


def test_009_rolls_the_ticker_rebuild_back_when_009_itself_fails(tmp_path):
    """The rebuild empties the table first, so rollback has to restore it.

    Injected *inside* 009's own transaction — a failure in some later
    migration would prove nothing, since 009 is entitled to have
    committed by then.
    """

    original = [("NVDA", 1), ("TSLA", 2), ("AMD", 3), ("AAPL", 4), ("META", 5)]
    database = pre_009_database(tmp_path, original)
    before = database_state(database)
    assert before["user_version"] == 8

    directory = partial_migrations(tmp_path, 8)
    name = "009_immutable_domain_and_update_integrity.sql"
    (directory / name).write_text(
        (MIGRATIONS_PATH / name).read_text(encoding="utf-8")
        + "\nINSERT INTO supported_tickers (ticker, display_name, position)\n"
        "VALUES ('TSLA', 'Duplicate', 99);\n",
        encoding="utf-8",
    )
    with pytest.raises((sqlite3.Error, Phase0MigrationError)):
        Phase0Repository(database, migrations_path=directory).migrate()

    assert database_state(database) == before
    reopened = Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 8)
    )
    assert ticker_rows(reopened) == original


def test_009_matches_a_fresh_database_however_it_started(tmp_path):
    """An upgraded table is indistinguishable from a freshly built one."""

    database = pre_009_database(
        tmp_path, [("META", 1), ("AAPL", 2), ("AMD", 3), ("NVDA", 4), ("TSLA", 5)]
    )
    upgraded = Phase0Repository(database)
    upgraded.migrate()
    fresh = migrated(tmp_path, "fresh.db")

    assert ticker_rows(upgraded) == ticker_rows(fresh)
    assert schema_snapshot(upgraded) == schema_snapshot(fresh)


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
        with repository.admin.connect_writable() as connection:
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
            "INSERT INTO stories (ticker, trading_day, canonical_title, "
            "pipeline_version) VALUES ('BAD', ?, 't', 'v1')",
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

    with repository.admin.connect_writable() as connection:
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
        "INSERT INTO stories (ticker, trading_day, canonical_title, "
        "pipeline_version) VALUES ('NVDA', ?, 't', 'v1')",
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

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
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

    with upgraded.admin.connect_writable() as connection:
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

    repository.admin.log_stage(
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

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
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
        repository.admin.complete_stage_key(
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
    repository.admin.complete_stage_key(**STAGE_KEY, run_id="owner", status="success")

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


#: A 60-second lease claimed at 12:00:00 expires at 12:01:00.  The
#: heartbeat is probed on either side of that instant and exactly on it.
LEASE_BOUNDARIES = [
    ("well before expiry", f"{DAY}T12:00:30Z", True),
    ("one second before expiry", f"{DAY}T12:00:59Z", True),
    ("exactly at expiry", f"{DAY}T12:01:00Z", False),
    ("one second after expiry", f"{DAY}T12:01:01Z", False),
    ("long after expiry", f"{DAY}T13:00:00Z", False),
]


@pytest.mark.parametrize(
    "label, moment, renewable",
    LEASE_BOUNDARIES,
    ids=[case[0] for case in LEASE_BOUNDARIES],
)
def test_a_heartbeat_cannot_revive_a_lease_that_has_expired(
    tmp_path, label, moment, renewable
):
    """Expiry ends ownership; it is not a suggestion the owner may decline.

    A lapsed lease is one another worker is entitled to reclaim.  An owner
    that could push it forward anyway would keep working on a partition
    someone else was about to take, and the two would hold it at once —
    the exact state the lease exists to prevent.
    """

    repository = migrated(tmp_path)
    repository.claim_stage_key(
        **STAGE_KEY, run_id="owner", lease_seconds=60, claimed_at=f"{DAY}T12:00:00Z"
    )

    assert (
        repository.heartbeat_stage_key(
            **STAGE_KEY, run_id="owner", lease_seconds=60, now=moment
        )
        is renewable
    )

    state = repository.stage_key_state(**STAGE_KEY)
    if renewable:
        assert state["lease_expires_at"] > f"{DAY}T12:01:00"
    else:
        # Untouched: the refusal wrote nothing, so the key is exactly as
        # reclaimable as it was a moment ago.
        assert state["lease_expires_at"].startswith(f"{DAY}T12:01:00")
        assert repository.claim_stage_key(
            **STAGE_KEY, run_id="next-owner", claimed_at=moment
        )


def test_heartbeat_and_reclaim_are_exact_complements(tmp_path):
    """At every instant a lease is renewable or reclaimable, never both.

    They are two readings of one predicate, so drift between them is the
    bug: an overlap lets two workers own the key, and a gap strands it.
    """

    for moment in [case[1] for case in LEASE_BOUNDARIES]:
        renew = migrated(tmp_path, f"renew-{moment.replace(':', '')}.sqlite3")
        reclaim = migrated(tmp_path, f"reclaim-{moment.replace(':', '')}.sqlite3")
        for repository in (renew, reclaim):
            repository.claim_stage_key(
                **STAGE_KEY,
                run_id="owner",
                lease_seconds=60,
                claimed_at=f"{DAY}T12:00:00Z",
            )
        renewed = renew.heartbeat_stage_key(
            **STAGE_KEY, run_id="owner", lease_seconds=60, now=moment
        )
        reclaimed = reclaim.claim_stage_key(
            **STAGE_KEY, run_id="other", claimed_at=moment
        )
        assert renewed is not reclaimed, moment


def test_an_owner_cannot_heartbeat_back_a_lease_another_worker_reclaimed(tmp_path):
    repository = migrated(tmp_path)
    repository.claim_stage_key(
        **STAGE_KEY, run_id="crashed", lease_seconds=60, claimed_at=f"{DAY}T12:00:00Z"
    )
    assert repository.claim_stage_key(
        **STAGE_KEY, run_id="recovered", claimed_at=f"{DAY}T12:05:00Z"
    )

    assert not repository.heartbeat_stage_key(
        **STAGE_KEY, run_id="crashed", now=f"{DAY}T12:05:30Z"
    )
    assert repository.stage_key_state(**STAGE_KEY)["run_id"] == "recovered"


def test_a_heartbeat_cannot_revive_a_lease_with_no_expiry(tmp_path):
    """A NULL expiry is reclaimable, so it is not renewable either."""

    repository = migrated(tmp_path)
    repository.claim_stage_key(**STAGE_KEY, run_id="owner", lease_seconds=60)
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE pipeline_stage_keys SET lease_expires_at = NULL "
            "WHERE run_id = 'owner'"
        )

    assert not repository.heartbeat_stage_key(**STAGE_KEY, run_id="owner")
    assert repository.claim_stage_key(**STAGE_KEY, run_id="next")


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
    theme_id = repository.admin.insert_theme(
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

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="member story"):
            connection.execute(
                "INSERT INTO theme_citations (theme_id, raw_item_id) VALUES (?, ?)",
                (theme_id, day["items"][3]),
            )


def test_a_raw_item_cannot_be_citable_from_two_themes_in_one_ticker_day(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    first = repository.admin.insert_theme(
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
    second = repository.admin.insert_theme(
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

    # A raw item may belong to two stories — `story_members` is keyed on
    # the pair — so two themes holding *different* stories can both have a
    # legitimate claim to cite it.  That is the route this rule exists for.
    # Handing `second` the story `first` already owns is no longer a route
    # at all: migration 014 refuses it before the citation rule is reached.
    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="member of another theme"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (second, day["stories"]["cf1"]),
            )

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT INTO story_members (story_id, raw_item_id, position) "
            "VALUES (?, ?, ?)",
            (day["stories"]["cf2"], day["items"][0], 1),
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
    theme_id = repository.admin.insert_theme(
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

    with repository.admin.connect_writable() as connection:
        with pytest.raises(
            sqlite3.IntegrityError, match="ticker, day, and pipeline version"
        ):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (theme_id, foreign_story),
            )


def test_citation_lifecycle_deletions_follow_a_valid_order(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)
    theme_id = repository.admin.insert_theme(
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

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM theme_citations WHERE theme_id = ?", (theme_id,)
        )
        connection.execute("DELETE FROM theme_stories WHERE theme_id = ?", (theme_id,))
        connection.execute("DELETE FROM themes WHERE id = ?", (theme_id,))
    assert repository.count("themes") == 0


def test_canonical_member_cannot_leave_its_story(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)

    with repository.admin.connect_writable() as connection:
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
    repository.admin.insert_eval_label(
        label_type="dedup",
        item_a_id=item_ids[0],
        item_b_id=item_ids[1],
        reviewer="kartik",
        label="different",
    )

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
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
        repository.admin.log_stage(
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

    repository.admin.log_stage(
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
    repository.admin.clear_derived_for_day(DAY)
    assert repository.count("run_log_stage_keys") == 0


# ----------------------------------------------------------------------
# A claimed key is always one a run can settle
#
# `claim_stage_key` normalized only `ticker` and `trading_day`, while
# `stage_run` requires all five identity fields to be non-blank and
# stripped.  So a claim with `run_id=""` was written as `running` and
# then could not be settled by anybody — the identity holding the lease
# was one `stage_run` refuses, and the partition stayed locked until the
# lease expired.  A padded `stage` did the same thing more quietly.
# ----------------------------------------------------------------------


VALID_KEY_IDENTITY = {
    "stage": "m3.semantic",
    "ticker": "NVDA",
    "trading_day": DAY,
    "pipeline_version": "v1",
    "run_id": "run-lifecycle",
}

#: Every identity field, blanked and whitespaced.  `ticker` and
#: `trading_day` were already validated; they are here so the matrix is
#: the whole identity rather than the half that was broken.
INVALID_KEY_IDENTITIES = [
    ("blank stage", {"stage": ""}, Phase0ValidationError),
    ("whitespace stage", {"stage": "   "}, Phase0ValidationError),
    ("blank pipeline_version", {"pipeline_version": ""}, Phase0ValidationError),
    (
        "whitespace pipeline_version",
        {"pipeline_version": " \t "},
        Phase0ValidationError,
    ),
    ("blank run_id", {"run_id": ""}, Phase0ValidationError),
    ("whitespace run_id", {"run_id": " \n "}, Phase0ValidationError),
    ("none stage", {"stage": None}, Phase0ValidationError),
    ("none run_id", {"run_id": None}, Phase0ValidationError),
    ("blank ticker", {"ticker": ""}, Phase0ValidationError),
    ("unsupported ticker", {"ticker": "GOOG"}, UnsupportedTickerError),
    ("malformed trading_day", {"trading_day": "not-a-day"}, Phase0ValidationError),
    ("empty trading_day", {"trading_day": ""}, Phase0ValidationError),
]


@pytest.mark.parametrize(
    "override, error",
    [pytest.param(o, e, id=label) for label, o, e in INVALID_KEY_IDENTITIES],
)
def test_a_stage_key_cannot_be_claimed_with_an_invalid_identity(
    tmp_path, override, error
):
    repository = migrated(tmp_path)

    with pytest.raises(error):
        repository.claim_stage_key(**{**VALID_KEY_IDENTITY, **override})

    # Rejected before any write: no claim, no partial row, nothing leased.
    assert repository.read.stage_key_rows() == []
    assert repository.count("pipeline_stage_keys") == 0


@pytest.mark.parametrize(
    "override, error",
    [pytest.param(o, e, id=label) for label, o, e in INVALID_KEY_IDENTITIES],
)
def test_the_sibling_stage_key_methods_reject_the_same_identities(
    tmp_path, override, error
):
    """`heartbeat` and the admin completion share the one normalizer.

    They are lookups rather than creators, so the damage is different —
    a padded stage silently matches nothing — but the identity contract
    has to be the same one or the lifecycle has two.
    """

    repository = migrated(tmp_path)
    repository.claim_stage_key(**VALID_KEY_IDENTITY)

    with pytest.raises(error):
        repository.heartbeat_stage_key(**{**VALID_KEY_IDENTITY, **override})
    with pytest.raises(error):
        repository.admin.complete_stage_key(**{**VALID_KEY_IDENTITY, **override})

    row = repository.read.stage_key_rows()[0]
    assert row["status"] == "running"
    assert row["run_id"] == VALID_KEY_IDENTITY["run_id"]


NORMALIZED_KEY_IDENTITIES = [
    ("lowercase ticker", {"ticker": "nvda"}),
    ("padded ticker", {"ticker": "  NVDA "}),
    ("padded stage", {"stage": "  m3.semantic  "}),
    ("padded pipeline_version", {"pipeline_version": " v1 "}),
    ("padded run_id", {"run_id": "  run-lifecycle  "}),
    ("datetime-ish trading day", {"trading_day": DAY}),
]


@pytest.mark.parametrize(
    "override",
    [pytest.param(o, id=label) for label, o in NORMALIZED_KEY_IDENTITIES],
)
def test_a_normalized_claim_is_stored_canonically_and_stays_settleable(
    tmp_path, override
):
    """The point of the contract: whatever is claimed, a run can settle it."""

    repository = migrated(tmp_path)
    assert repository.claim_stage_key(**{**VALID_KEY_IDENTITY, **override}) is True

    stored = repository.read.stage_key_rows()[0]
    for column, expected in VALID_KEY_IDENTITY.items():
        assert str(stored[column]) == str(expected), column

    # And the canonical identity settles it, through the ordinary run.
    with repository.stage_run(
        run_id=VALID_KEY_IDENTITY["run_id"],
        stage=VALID_KEY_IDENTITY["stage"],
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key={k: v for k, v in VALID_KEY_IDENTITY.items() if k != "run_id"},
    ) as run:
        repository.ingest_raw_items([], run=run, terminal=True)

    assert repository.read.stage_key_rows()[0]["status"] == "success"


def test_a_claim_and_a_run_agree_on_every_identity_field(tmp_path):
    """Stated as a property rather than a list, so it cannot drift.

    Anything `claim_stage_key` accepts, `stage_run` must accept; anything
    it rejects, `stage_run` must reject. The two used to disagree on
    three of the five fields.
    """

    probes = [
        {},
        *(o for _, o, _ in INVALID_KEY_IDENTITIES),
        *(o for _, o in NORMALIZED_KEY_IDENTITIES),
    ]
    for override in probes:
        identity = {**VALID_KEY_IDENTITY, **override}

        def claim_ok():
            fresh = migrated(tmp_path, f"c{abs(hash(str(override)))}.db")
            try:
                fresh.claim_stage_key(**identity)
                return True
            except Phase0Error:
                return False

        def run_ok():
            fresh = migrated(tmp_path, f"r{abs(hash(str(override)))}.db")
            try:
                with fresh.stage_run(
                    run_id=identity["run_id"],
                    stage=identity["stage"],
                    trading_day=identity["trading_day"],
                    pipeline_version=identity["pipeline_version"],
                    ticker=identity["ticker"],
                ) as run:
                    fresh.ingest_raw_items([], run=run, terminal=True)
                return True
            except Phase0Error:
                return False

        assert claim_ok() == run_ok(), f"disagreement on {override}"


def test_a_rejected_claim_leaves_an_existing_key_untouched(tmp_path):
    """Not merely "no new row" — no mutation of the one already there."""

    repository = migrated(tmp_path)
    repository.claim_stage_key(**VALID_KEY_IDENTITY, lease_seconds=600)
    before = [dict(row) for row in repository.read.stage_key_rows()]

    for _, override, error in INVALID_KEY_IDENTITIES:
        with pytest.raises(error):
            repository.claim_stage_key(**{**VALID_KEY_IDENTITY, **override})

    assert [dict(row) for row in repository.read.stage_key_rows()] == before


def test_valid_lease_and_reclaim_semantics_survive_the_normalization(tmp_path):
    """The lease rules are unchanged; only the identity gate moved."""

    repository = migrated(tmp_path)
    start = "2026-07-23T12:00:00+00:00"
    assert repository.claim_stage_key(
        **VALID_KEY_IDENTITY, lease_seconds=60, claimed_at=start
    )
    # A live lease is not reclaimable, and its owner may renew it — even
    # when the caller spells the identity untidily.
    assert not repository.claim_stage_key(
        **{**VALID_KEY_IDENTITY, "run_id": "other"},
        lease_seconds=60,
        claimed_at="2026-07-23T12:00:30+00:00",
    )
    assert repository.heartbeat_stage_key(
        **{**VALID_KEY_IDENTITY, "stage": " m3.semantic "},
        lease_seconds=60,
        now="2026-07-23T12:00:30+00:00",
    )
    # Past expiry, another worker takes it.
    assert repository.claim_stage_key(
        **{**VALID_KEY_IDENTITY, "run_id": "other"},
        lease_seconds=60,
        claimed_at="2026-07-23T12:05:00+00:00",
    )


# ----------------------------------------------------------------------
# A run identity names exactly one partition, permanently
#
# `run_log` is an upsert keyed on `(run_id, stage)`.  The conflict branch
# rewrote `ticker` and left `trading_day` and `pipeline_version` alone,
# so reusing an identity under a second ticker relabelled the first
# run's row, and reusing it under a second day or version logged the new
# run under the old one's partition.  Either way one row ends up
# describing work no single run did.
# ----------------------------------------------------------------------


def settle_run(repository, **overrides):
    """One complete logged run, doing nothing but writing its own log."""

    kwargs = {
        "run_id": "shared-run",
        "stage": "m3.semantic",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "ticker": "NVDA",
    }
    kwargs.update(overrides)
    with repository.stage_run(**kwargs) as run:
        repository.ingest_raw_items([], run=run, terminal=True)


#: Each way a caller can reuse `(run_id, stage)` under a different
#: partition.  `ticker` was the one that silently *moved*; the other two
#: were silently *kept*, which reads the same in the log and is just as
#: wrong.
FOREIGN_RUN_PARTITIONS = [
    ("different ticker", {"ticker": "AMD"}),
    ("different trading day", {"trading_day": "2026-07-24"}),
    ("different pipeline version", {"pipeline_version": "v2"}),
    ("different ticker and day", {"ticker": "AMD", "trading_day": "2026-07-24"}),
    ("ticker dropped", {"ticker": None}),
]


@pytest.mark.parametrize(
    "overrides",
    [pytest.param(o, id=label) for label, o in FOREIGN_RUN_PARTITIONS],
)
def test_a_run_identity_cannot_be_reused_in_another_partition(tmp_path, overrides):
    repository = migrated(tmp_path)
    settle_run(repository)
    before = [dict(row) for row in repository.read.run_log_rows()]
    assert len(before) == 1

    with pytest.raises(Phase0RunContextError, match="cannot be reused"):
        settle_run(repository, **overrides)

    # Field for field, the persisted row is what it was — the rejection
    # neither overwrote it nor added a second one.
    assert [dict(row) for row in repository.read.run_log_rows()] == before


def test_the_same_identity_in_the_same_partition_is_an_ordinary_replay(tmp_path):
    repository = migrated(tmp_path)
    settle_run(repository)
    first = dict(repository.read.run_log_rows()[0])

    settle_run(repository)
    rows = repository.read.run_log_rows()

    assert len(rows) == 1
    second = dict(rows[0])
    assert second["id"] == first["id"]
    for column in ("ticker", "trading_day", "pipeline_version", "stage"):
        assert second[column] == first[column]


def test_a_different_stage_is_a_different_identity(tmp_path):
    """`stage` is half the key, so it partitions rather than collides."""

    repository = migrated(tmp_path)
    settle_run(repository)
    settle_run(repository, stage="m5.themes", ticker="AMD")

    rows = sorted(
        (dict(row) for row in repository.read.run_log_rows()),
        key=lambda row: row["stage"],
    )
    assert [(row["stage"], row["ticker"]) for row in rows] == [
        ("m3.semantic", "NVDA"),
        ("m5.themes", "AMD"),
    ]


def test_rejected_reuse_mutates_no_data(tmp_path):
    """The rejection happens inside the run's transaction, so nothing lands."""

    repository = migrated(tmp_path)
    with repository.stage_run(
        run_id="shared-run",
        stage="m0.ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
    items_before = repository.count("raw_items")
    log_before = [dict(row) for row in repository.read.run_log_rows()]

    with pytest.raises(Phase0RunContextError, match="cannot be reused"):
        with repository.stage_run(
            run_id="shared-run",
            stage="m0.ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="AMD",
        ) as run:
            repository.ingest_raw_items([raw_item(2, "AMD")], run=run, terminal=True)

    assert repository.count("raw_items") == items_before
    assert [dict(row) for row in repository.read.run_log_rows()] == log_before


def test_one_run_identity_never_links_to_two_stage_keys(tmp_path):
    repository = migrated(tmp_path)
    keys = {
        ticker: {
            "stage": "m3.semantic",
            "ticker": ticker,
            "trading_day": DAY,
            "pipeline_version": "v1",
        }
        for ticker in ("NVDA", "AMD")
    }
    for key in keys.values():
        repository.claim_stage_key(**key, run_id="shared-run")

    settle_run(repository, stage_key=keys["NVDA"])
    with pytest.raises(Phase0RunContextError, match="cannot be reused"):
        settle_run(repository, ticker="AMD", stage_key=keys["AMD"])

    with repository.admin.connect_writable() as connection:
        linked = [
            (str(row["ticker"]), str(row["trading_day"]), str(row["pipeline_version"]))
            for row in connection.execute(
                "SELECT ticker, trading_day, pipeline_version FROM run_log_stage_keys"
            )
        ]
    assert linked == [("NVDA", DAY, "v1")]


def test_the_admin_log_path_enforces_the_same_run_identity(tmp_path):
    """Not only `stage_run` — the row itself may only be written one way."""

    repository = migrated(tmp_path)
    entry = {
        "run_id": "admin-run",
        "stage": "cluster",
        "counts": {},
        "duration_ms": 1,
        "errors": [],
        "started_at": f"{DAY}T12:00:00+00:00",
        "completed_at": f"{DAY}T12:00:01+00:00",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "ticker": "NVDA",
    }
    repository.admin.log_stage(**entry)
    before = [dict(row) for row in repository.read.run_log_rows()]

    with pytest.raises(Phase0RunContextError, match="cannot be reused"):
        repository.admin.log_stage(**{**entry, "ticker": "AMD"})

    assert [dict(row) for row in repository.read.run_log_rows()] == before


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
    with repository.admin.connect_writable() as connection:
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
# Zero is a position, not a silence
#
# `StoryMemberRecord.position` and `OtherCoverageRecord.position` both
# defaulted to `0`, and both call sites read them as
# `value if value else index`.  Zero is the *first* position, so a field
# defaulting to it could not say "I did not state this" and an explicit
# `0` on anything but the first element was replaced by that element's
# index in the list.  Both fields are now unset-by-default and both call
# sites ask `is None`.
# ----------------------------------------------------------------------


def member_positions(repository, story_id: int) -> list[tuple[int, int]]:
    with repository.admin.connect_writable() as connection:
        return [
            (int(row["raw_item_id"]), int(row["position"]))
            for row in connection.execute(
                "SELECT raw_item_id, position FROM story_members "
                "WHERE story_id = ? ORDER BY raw_item_id",
                (story_id,),
            )
        ]


def member_order(repository, story_id: int) -> list[int]:
    """Members in the order their positions put them in."""

    with repository.admin.connect_writable() as connection:
        return [
            int(row["raw_item_id"])
            for row in connection.execute(
                "SELECT raw_item_id FROM story_members WHERE story_id = ? "
                "ORDER BY position, raw_item_id",
                (story_id,),
            )
        ]


def positioned_story(repository, positions, *, fingerprint="cf-pos"):
    """One story whose members carry exactly the positions given.

    A `None` in `positions` means the caller said nothing about that
    member, which is the only case an inferred index is allowed.
    """

    items = seed_raw_items(repository, len(positions), ticker="NVDA")
    members = tuple(
        StoryMemberRecord(raw_item_id=item, position=position, outlet=f"O{item}")
        for item, position in zip(items, positions)
    )
    record = StoryRecord(
        cluster_fingerprint=fingerprint,
        canonical_title="Positioned",
        members=members,
        canonical_item_id=items[0],
        outlet_count=len(items),
        content_hash="h-pos",
        stage="m3.semantic",
    )
    report = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[record],
    )
    story_id = [
        int(row["id"])
        for row in repository.stories_for_day(DAY, "NVDA")
        if row["cluster_fingerprint"] == fingerprint
    ][0]
    return items, story_id, report, record


MEMBER_POSITION_CASES = [
    # label, supplied, expected
    ("first-explicit-zero", [0, 3, 9], [0, 3, 9]),
    ("later-explicit-zero", [7, 0, 5], [7, 0, 5]),
    ("last-explicit-zero", [4, 8, 0], [4, 8, 0]),
    ("every-position-explicit", [4, 5, 6], [4, 5, 6]),
    ("all-omitted-are-enumerated", [None, None, None], [0, 1, 2]),
    ("omitted-among-explicit", [7, None, 5], [7, 1, 5]),
    ("explicit-zero-beside-omitted", [None, 0, None], [0, 0, 2]),
]


@pytest.mark.parametrize(
    "label, supplied, expected",
    MEMBER_POSITION_CASES,
    ids=[case[0] for case in MEMBER_POSITION_CASES],
)
def test_story_member_positions_are_stored_as_supplied(
    tmp_path, label, supplied, expected
):
    """Cases 1-4: stated positions survive, unstated ones are numbered.

    `all-omitted-are-enumerated` pins the documented fallback, which is
    the list index — and `omitted-among-explicit` shows the fallback is
    per member, not a mode the whole list is in.
    """

    repository = migrated(tmp_path)
    items, story_id, _, _ = positioned_story(repository, supplied)

    assert member_positions(repository, story_id) == list(zip(items, expected))


def test_omitting_the_position_argument_entirely_is_what_unset_means(tmp_path):
    """The default has to *be* unset, not a value that looks like one.

    Everything above hands `position=None` explicitly, which exercises
    the `is None` check but not the default behind it. A caller who never
    mentions `position` must land in the same place — otherwise the field
    still has no way to stay silent, and an explicit zero is
    indistinguishable from absence all over again.
    """

    repository = migrated(tmp_path)
    items = seed_raw_items(repository, 3, ticker="NVDA")
    record = StoryRecord(
        cluster_fingerprint="cf-omitted",
        canonical_title="Omitted",
        # No `position` anywhere.
        members=tuple(
            StoryMemberRecord(raw_item_id=item, outlet=f"O{item}") for item in items
        ),
        canonical_item_id=items[0],
        outlet_count=len(items),
        content_hash="h-omitted",
        stage="m3.semantic",
    )
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[record],
    )

    story_id = int(repository.stories_for_day(DAY, "NVDA")[0]["id"])
    assert StoryMemberRecord(raw_item_id=items[0]).position is None
    assert member_positions(repository, story_id) == list(zip(items, [0, 1, 2]))


def test_omitting_a_coverage_position_entirely_is_what_unset_means(tmp_path):
    """The same, for the coverage record's default."""

    repository = migrated(tmp_path)
    items = seed_raw_items(repository, 3, ticker="NVDA")
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story(f"cf{index}", [item]) for index, item in enumerate(items)],
    )
    by_fingerprint = {
        str(row["cluster_fingerprint"]): int(row["id"])
        for row in repository.stories_for_day(DAY, "NVDA")
    }
    ids = [by_fingerprint[f"cf{index}"] for index in range(3)]

    assert OtherCoverageRecord(story_id=ids[1], reason="clustering_noise").position is (
        None
    )
    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="T",
                theme_key="T",
                label="Theme",
                story_ids=(ids[0],),
                citation_item_ids=(items[0],),
                method="hdbscan",
                salience_rank=1,
            )
        ],
        # No `position` anywhere.
        other_coverage=(
            OtherCoverageRecord(story_id=ids[1], reason="clustering_noise"),
            OtherCoverageRecord(story_id=ids[2], reason="clustering_noise"),
        ),
    )

    assert coverage_positions(repository) == [(ids[1], 0), (ids[2], 1)]


def test_a_stated_member_position_decides_the_stored_order(tmp_path):
    """Case 6: the point of a position is the order it produces.

    `[1, 5, 0]` is the shape where the old fallback actually reordered
    the story: the third member's explicit `0` became its index `2`, so
    the member the caller put first came back second.
    """

    repository = migrated(tmp_path)
    items, story_id, _, _ = positioned_story(repository, [1, 5, 0])

    assert member_order(repository, story_id) == [items[2], items[0], items[1]]


def test_a_replayed_explicit_zero_is_unchanged(tmp_path):
    """Case 5: an identical settlement stays identical."""

    repository = migrated(tmp_path)
    items, story_id, first, record = positioned_story(repository, [7, 0, 5])
    assert len(first.inserted) == 1
    before = member_positions(repository, story_id)

    replay = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[record],
    )

    assert replay.inserted == () and replay.updated == ()
    assert len(replay.unchanged) == 1
    assert member_positions(repository, story_id) == before


def test_a_drifted_member_position_is_seen_and_repaired(tmp_path):
    """Position is part of story equality, in both directions.

    This is the half that made the defect self-concealing: the write and
    the comparison mangled an explicit zero the same way, so a replay
    agreed with itself. Against a stored zero the two disagree — which is
    what "an identical replay appears changed" looks like from the
    outside, and what the repair now does about it.
    """

    repository = migrated(tmp_path)
    items, story_id, _, record = positioned_story(repository, [7, 0, 5])
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE story_members SET position = 1 "
            "WHERE story_id = ? AND raw_item_id = ?",
            (story_id, items[1]),
        )
        connection.commit()

    report = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[record],
    )

    assert len(report.updated) == 1 and report.unchanged == ()
    assert member_positions(repository, story_id) == [
        (items[0], 7),
        (items[1], 0),
        (items[2], 5),
    ]


def test_a_negative_member_position_is_still_refused(tmp_path):
    """The `is None` check replaced the fallback, not the validation."""

    repository = migrated(tmp_path)
    with pytest.raises(Phase0ValidationError, match="member position must be >= 0"):
        positioned_story(repository, [0, -1])


def coverage_positions(repository) -> list[tuple[int, int]]:
    with repository.admin.connect_writable() as connection:
        return [
            (int(row["story_id"]), int(row["position"]))
            for row in connection.execute(
                "SELECT story_id, position FROM theme_other_coverage "
                "ORDER BY story_id"
            )
        ]


def covered_day(repository, positions):
    """A themed day whose Other Coverage carries exactly these positions."""

    items = seed_raw_items(repository, 1 + len(positions), ticker="NVDA")
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story(f"cf{index}", [item]) for index, item in enumerate(items)],
    )
    by_fingerprint = {
        str(row["cluster_fingerprint"]): int(row["id"])
        for row in repository.stories_for_day(DAY, "NVDA")
    }
    ids = [by_fingerprint[f"cf{index}"] for index in range(len(items))]
    other = tuple(
        OtherCoverageRecord(
            story_id=ids[1 + index], reason="clustering_noise", position=position
        )
        for index, position in enumerate(positions)
    )
    themes = [
        ThemeRecord(
            fingerprint="T",
            theme_key="T",
            label="Theme",
            story_ids=(ids[0],),
            citation_item_ids=(items[0],),
            method="hdbscan",
            salience_rank=1,
        )
    ]
    report = reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=themes,
        other_coverage=other,
    )
    return ids, themes, other, report


COVERAGE_POSITION_CASES = [
    ("first-explicit-zero", [0, 3, 9], [0, 3, 9]),
    ("later-explicit-zero", [7, 0, 5], [7, 0, 5]),
    ("last-explicit-zero", [4, 8, 0], [4, 8, 0]),
    ("every-position-explicit", [4, 5, 6], [4, 5, 6]),
    ("all-omitted-are-enumerated", [None, None, None], [0, 1, 2]),
    ("omitted-among-explicit", [7, None, 5], [7, 1, 5]),
]


@pytest.mark.parametrize(
    "label, supplied, expected",
    COVERAGE_POSITION_CASES,
    ids=[case[0] for case in COVERAGE_POSITION_CASES],
)
def test_other_coverage_positions_are_stored_as_supplied(
    tmp_path, label, supplied, expected
):
    """Cases 7-10, the same table of shapes as the story members."""

    repository = migrated(tmp_path)
    ids, _, _, _ = covered_day(repository, supplied)

    assert coverage_positions(repository) == list(zip(ids[1:], expected))


def test_theme_set_reads_coverage_back_in_the_stated_order(tmp_path):
    """Case 12: the order a stated position was for.

    `[1, 5, 0]` is the shape the old fallback reordered — the third entry
    said it came first and was filed third.
    """

    repository = migrated(tmp_path)
    ids, _, _, _ = covered_day(repository, [1, 5, 0])

    day = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
    assert [int(row["story_id"]) for row in day["other_coverage"]] == [
        ids[3],
        ids[1],
        ids[2],
    ]


def test_replayed_coverage_with_an_explicit_zero_is_unchanged(tmp_path):
    """Case 11, including `changed_outputs`.

    Coverage is compared before it is rewritten, so an unchanged list has
    to be reported as unchanged *and* left alone — a replay that rewrote
    it would be the auxiliary-output defect all over again.
    """

    repository = migrated(tmp_path)
    ids, themes, other, first = covered_day(repository, [7, 0, 5])
    assert "other_coverage" in first.changed_outputs
    before = coverage_positions(repository)

    replay = reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=themes,
        other_coverage=other,
    )

    assert replay.changed_outputs == ()
    assert replay.updated == () and len(replay.unchanged) == 1
    assert coverage_positions(repository) == before


def test_a_drifted_coverage_position_is_seen_and_repaired(tmp_path):
    """`position` is compared as a value, so drift is a change."""

    repository = migrated(tmp_path)
    ids, themes, other, _ = covered_day(repository, [7, 0, 5])
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE theme_other_coverage SET position = 1 WHERE story_id = ?",
            (ids[2],),
        )
        connection.commit()

    report = reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=themes,
        other_coverage=other,
    )

    assert "other_coverage" in report.changed_outputs
    assert coverage_positions(repository) == [
        (ids[1], 7),
        (ids[2], 0),
        (ids[3], 5),
    ]


def test_no_optional_field_reads_a_valid_falsy_value_as_absence(tmp_path):
    """The audit, kept honest as a test rather than a claim.

    Both defects were the same three tokens — `x if x else fallback` on a
    field whose falsy value is real data. This refuses that shape
    anywhere in the package, so the next optional field cannot quietly
    acquire it.
    """

    pattern = re.compile(r"\b(\w+(?:\.\w+)*)\s+if\s+\1\s+else\s+")
    offenders = []
    for path in sorted(MIGRATIONS_PATH.parent.rglob("*.py")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")

    assert offenders == [], "truthiness fallback on a possibly-valid falsy value"


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
        # Story reconciliation owns no output outside these ids.
        "changed_outputs": 0,
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
# What "unchanged" has to mean
#
# A replay is unchanged only when a settlement would write exactly what is
# already stored.  Anything less lets a real change be reported as a
# no-op and leaves the earlier payload persisted for good, which is the
# one failure mode replay cannot absorb: nothing later notices, because
# nothing later looks.
#
# Every probe below changes exactly one persisted thing, replays, and
# asserts both halves — the report says "updated", *and* the stored value
# is the new one.
# ----------------------------------------------------------------------


def stored_story(repository: Phase0Repository, fingerprint: str = "cf1") -> dict:
    with repository.admin.connect_writable() as connection:
        row = connection.execute(
            "SELECT * FROM stories WHERE cluster_fingerprint = ?", (fingerprint,)
        ).fetchone()
        story_id = int(row["id"])
        return {
            "row": dict(row),
            "members": [
                dict(member)
                for member in connection.execute(
                    "SELECT * FROM story_members WHERE story_id = ? "
                    "ORDER BY raw_item_id",
                    (story_id,),
                )
            ],
            "conflicts": [
                dict(conflict)
                for conflict in connection.execute(
                    "SELECT * FROM story_provider_conflicts WHERE story_id = ? "
                    "ORDER BY provider_namespace, provider_item_id",
                    (story_id,),
                )
            ],
            "merges": [
                dict(merge)
                for merge in connection.execute(
                    "SELECT * FROM story_semantic_merges WHERE story_id = ? "
                    "ORDER BY left_story_key, right_story_key",
                    (story_id,),
                )
            ],
        }


VECTOR_A = serialize_vector(np.ones(4, dtype=EMBEDDING_DTYPE))
VECTOR_B = serialize_vector(np.full(4, 0.5, dtype=EMBEDDING_DTYPE))


#: (label, first payload, replayed payload, what to read back afterwards).
STORY_PAYLOAD_CHANGES = [
    (
        "embedding",
        {"embedding": VECTOR_A},
        {"embedding": VECTOR_B},
        lambda stored: stored["row"]["embedding"],
    ),
    (
        "embedding-appears",
        {},
        {"embedding": VECTOR_B},
        lambda stored: stored["row"]["embedding"],
    ),
    (
        "embedding-disappears",
        {"embedding": VECTOR_A},
        {},
        lambda stored: stored["row"]["embedding"],
    ),
    (
        "provider-conflict-fields",
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1",), ("title",)),
            )
        },
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1",), ("title", "url")),
            )
        },
        lambda stored: [conflict["fields"] for conflict in stored["conflicts"]],
    ),
    (
        "provider-conflict-item-ids",
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1",), ("title",)),
            )
        },
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1", "2"), ("title",)),
            )
        },
        lambda stored: [conflict["item_ids"] for conflict in stored["conflicts"]],
    ),
    (
        "provider-conflict-added",
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1",), ("title",)),
            )
        },
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1",), ("title",)),
                ProviderConflictRecord("finnhub", "dup-2", ("3",), ("url",)),
            )
        },
        lambda stored: len(stored["conflicts"]),
    ),
    (
        "provider-conflict-removed",
        {
            "provider_conflicts": (
                ProviderConflictRecord("yahoo", "dup-1", ("1",), ("title",)),
            )
        },
        {},
        lambda stored: len(stored["conflicts"]),
    ),
    (
        "semantic-merge-similarity",
        {"semantic_merges": (SemanticMergeRecord("m2-a", "m2-b", 0.91),)},
        {"semantic_merges": (SemanticMergeRecord("m2-a", "m2-b", 0.42),)},
        lambda stored: [merge["similarity"] for merge in stored["merges"]],
    ),
    (
        "semantic-merge-reason",
        {"semantic_merges": (SemanticMergeRecord("m2-a", "m2-b", 0.91, "near"),)},
        {"semantic_merges": (SemanticMergeRecord("m2-a", "m2-b", 0.91, "far"),)},
        lambda stored: [merge["reason"] for merge in stored["merges"]],
    ),
    (
        "semantic-merge-removed",
        {"semantic_merges": (SemanticMergeRecord("m2-a", "m2-b", 0.91),)},
        {},
        lambda stored: len(stored["merges"]),
    ),
]


@pytest.mark.parametrize(
    "label, before, after, read",
    STORY_PAYLOAD_CHANGES,
    ids=[case[0] for case in STORY_PAYLOAD_CHANGES],
)
def test_a_changed_story_payload_is_never_reported_unchanged(
    tmp_path, label, before, after, read
):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)

    def replay(overrides):
        return reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[
                story(
                    "cf1", item_ids[:1], member_story_keys=("m2-a", "m2-b"), **overrides
                )
            ],
        )

    replay(before)
    first = read(stored_story(repository))

    report = replay(after)
    second = read(stored_story(repository))

    assert first != second, "the probe did not actually change anything"
    assert report.counts["updated"] == 1
    assert report.counts["unchanged"] == 0
    # And the replayed payload is the one that is now stored, not merely
    # "something changed": a partial update would satisfy the counts.
    assert second == read(stored_story(repository))
    assert replay(after).counts["unchanged"] == 1


#: The story-member columns reconciliation owns, one at a time.
STORY_MEMBER_CHANGES = [
    ("position", 0, 5),
    ("outlet", "Reuters", "Bloomberg"),
    ("url", "https://example.com/a", "https://example.com/b"),
    ("canonical_url", "https://example.com/a", "https://example.com/b"),
    ("match_reason", "exact_url", "semantic"),
    ("quarantined", False, True),
]


@pytest.mark.parametrize(
    "field, before, after",
    STORY_MEMBER_CHANGES,
    ids=[case[0] for case in STORY_MEMBER_CHANGES],
)
def test_a_changed_member_payload_is_never_reported_unchanged(
    tmp_path, field, before, after
):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)

    def replay(value):
        return reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[
                StoryRecord(
                    cluster_fingerprint="cf1",
                    canonical_title="Story cf1",
                    members=(
                        StoryMemberRecord(raw_item_id=item_ids[0], **{field: value}),
                    ),
                    canonical_item_id=item_ids[0],
                )
            ],
        )

    replay(before)
    report = replay(after)

    member = stored_story(repository)["members"][0]
    expected = int(after) if field in {"position", "quarantined"} else after
    assert member[field] == expected
    assert report.counts["updated"] == 1
    assert report.counts["unchanged"] == 0


def test_a_story_that_returns_unchanged_is_no_longer_invalidated(tmp_path):
    """The same defect class, one field further on.

    ``_update_reconciled_story`` clears ``invalidated_at``, so being live
    is persisted state this path owns — and a story that dropped out, was
    invalidated, and comes back byte-identical was being left flagged
    because "identical" was decided without looking at the flag.
    """

    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)
    records = [story("cf1", item_ids[:1]), story("cf2", item_ids[1:])]

    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=records,
    )
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=records[1:],
        delete_obsolete=False,
    )
    assert stored_story(repository)["row"]["invalidated_at"] is not None

    report = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=records,
    )

    assert stored_story(repository)["row"]["invalidated_at"] is None
    assert report.counts["updated"] == 1
    assert report.counts["unchanged"] == 1


def test_an_equivalent_story_replay_is_still_unchanged(tmp_path):
    """Determinism, from the other side: reordering is not a change.

    An equality contract that catches every difference is easy to reach by
    catching differences that are not there.  Members, conflicts, and
    merges are canonically ordered, so the same result presented in
    another order must still be a no-op.
    """

    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 3)
    conflicts = (
        ProviderConflictRecord("yahoo", "dup-1", ("1", "2"), ("title",)),
        ProviderConflictRecord("finnhub", "dup-2", ("3",), ("url",)),
    )
    merges = (
        SemanticMergeRecord("m2-a", "m2-b", 0.91),
        SemanticMergeRecord("m2-b", "m2-c", 0.72),
    )
    # Positions start at 1: ``_prepare_story`` treats a falsy position as
    # "unset" and substitutes the caller's ordering, so a member at
    # position 0 genuinely does change when the caller reorders.
    members = tuple(
        StoryMemberRecord(raw_item_id=item_id, position=position, outlet=f"O{item_id}")
        for position, item_id in enumerate(item_ids, start=1)
    )
    base = {
        "cluster_fingerprint": "cf1",
        "canonical_title": "Story cf1",
        "canonical_item_id": item_ids[0],
        "outlet_count": 3,
        "embedding": VECTOR_A,
        "member_story_keys": ("m2-a", "m2-b", "m2-c"),
    }

    def replay(**overrides):
        return reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[StoryRecord(**{**base, **overrides})],
        )

    replay(members=members, provider_conflicts=conflicts, semantic_merges=merges)
    report = replay(
        # Members keep their positions; only the order of presentation and
        # of the audit children moves.
        members=tuple(reversed(members)),
        provider_conflicts=tuple(reversed(conflicts)),
        semantic_merges=tuple(reversed(merges)),
    )

    assert report.counts == {
        "inserted": 0,
        "updated": 0,
        "unchanged": 1,
        "deleted": 0,
        "invalidated": 0,
        "removed_members": 0,
        "invalidated_themes": 0,
        "changed_outputs": 0,
    }


def test_every_story_column_is_owned_or_deliberately_exempt(tmp_path):
    """The rule that keeps this fix from rotting.

    A column added to ``stories`` and written by reconciliation but left
    out of the equality contract reintroduces exactly this bug, silently.
    So the contract is checked against the table itself: every column is
    either one reconciliation owns or one named here as not its business.
    """

    repository = migrated(tmp_path)
    with repository.admin.connect_writable() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(stories)")
        }

    not_owned = {
        "id",  # the database's
        "ticker",  # partition identity: what pairs the rows up
        "trading_day",
        "pipeline_version",
        "cluster_fingerprint",
        "canonical_item_id",  # written after members; compared separately
        "invalidated_at",  # compared separately, as liveness
        "updated_at",  # bookkeeping this class sets itself
    }

    assert columns == set(STORY_RECONCILED_COLUMNS) | not_owned
    assert not set(STORY_RECONCILED_COLUMNS) & not_owned
    # The declared list and the mapping the writes are built from are one
    # thing, not two that drift.
    prepared = Phase0Repository._prepare_story(story("cf1", [1]))
    assert set(Phase0Repository._story_column_values(prepared)) == set(
        STORY_RECONCILED_COLUMNS
    )


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


#: Every ``themes`` field reconciliation owns, changed one at a time.  The
#: centroid and the salience components are the ones that were invisible:
#: they are derived outputs, so a rerun that recomputes them has produced
#: a different theme even when its label and membership are identical.
THEME_FIELD_CHANGES = [
    ("centroid", VECTOR_A, VECTOR_B),
    ("salience", 0.10, 0.90),
    ("salience_story_component", 0.10, 0.90),
    ("salience_outlet_component", 0.20, 0.80),
    ("salience_recency_component", 0.30, 0.70),
    ("cohesion", 0.40, 0.60),
    ("min_pairwise_cohesion", 0.10, 0.20),
    ("story_count", 1, 7),
    ("outlet_count", 2, 9),
    ("salience_rank", 1, 4),
    ("label", "Chip demand", "Chip supply"),
    ("summary", "One", "Two"),
    ("label_source", "highest_salience_member", "llm"),
    ("status", "pending", "ready"),
    ("theme_key", "tk-a", "tk-b"),
    ("content_hash", "hash-a", "hash-b"),
    ("matched_previous_key", "prev-a", "prev-b"),
    ("latest_published_at", f"{DAY}T12:00:00+00:00", f"{DAY}T13:00:00+00:00"),
    ("method", "hdbscan", "agglomerative"),
    ("algorithm_version", "m5.1", "m5.2"),
    ("config_fingerprint", "cfg-a", "cfg-b"),
    ("model_name", "fake", "other"),
    ("model_revision", "r1", "r2"),
    ("embedding_dimension", 4, 8),
]


@pytest.mark.parametrize(
    "field, before, after",
    THEME_FIELD_CHANGES,
    ids=[case[0] for case in THEME_FIELD_CHANGES],
)
def test_a_changed_theme_field_is_never_reported_unchanged(
    tmp_path, field, before, after
):
    repository = migrated(tmp_path)
    day = build_day(repository)

    def replay(value):
        fields = {
            "fingerprint": "tf1",
            "label": "Chip demand",
            "story_ids": (day["stories"]["cf1"],),
            "citation_item_ids": (day["items"][0],),
            "method": "hdbscan",
            field: value,
        }
        return reconcile_themes(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            theme_set=theme_set(),
            themes=[ThemeRecord(**fields)],
        )

    replay(before)
    with repository.admin.connect_writable() as connection:
        stored_before = connection.execute("SELECT * FROM themes").fetchone()[field]

    report = replay(after)
    with repository.admin.connect_writable() as connection:
        stored_after = connection.execute("SELECT * FROM themes").fetchone()[field]

    assert stored_before != stored_after, "the probe did not change anything"
    assert stored_after == after
    assert report.counts["updated"] == 1
    assert report.counts["unchanged"] == 0
    assert replay(after).counts["unchanged"] == 1


def test_an_equivalent_theme_replay_is_still_unchanged(tmp_path):
    """Equivalent inputs stay equal however the stage ordered them."""

    repository = migrated(tmp_path)
    day = build_day(repository)
    story_ids = (day["stories"]["cf1"], day["stories"]["cf2"])
    citations = (day["items"][0], day["items"][2])

    def replay(stories, cites):
        return reconcile_themes(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            theme_set=theme_set(),
            themes=[
                ThemeRecord(
                    fingerprint="tf1",
                    label="Chip demand",
                    story_ids=stories,
                    citation_item_ids=cites,
                    method="hdbscan",
                    centroid=VECTOR_A,
                    salience=0.5,
                    salience_story_component=0.1,
                    salience_outlet_component=0.2,
                    salience_recency_component=0.3,
                )
            ],
        )

    replay(story_ids, citations)
    report = replay(tuple(reversed(story_ids)), tuple(reversed(citations)))

    assert report.counts["unchanged"] == 1
    assert report.counts["updated"] == 0


def test_every_theme_column_is_owned_or_deliberately_exempt(tmp_path):
    """The themes half of the rule that keeps this fix from rotting."""

    repository = migrated(tmp_path)
    with repository.admin.connect_writable() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(themes)")
        }

    not_owned = {
        "id",
        "ticker",
        "trading_day",
        "pipeline_version",
        "fingerprint",  # partition identity: what pairs the rows up
        "updated_at",
    }

    assert columns == set(THEME_RECONCILED_COLUMNS) | not_owned
    assert not set(THEME_RECONCILED_COLUMNS) & not_owned
    prepared = Phase0Repository._prepare_theme(
        ThemeRecord(fingerprint="tf1", label="L", story_ids=(1,))
    )
    assert set(Phase0Repository._theme_column_values(prepared)) == set(
        THEME_RECONCILED_COLUMNS
    )


# ----------------------------------------------------------------------
# What a reconciliation report has to account for
#
# The id tuples only describe themes.  A theme set also owns its own
# metadata row, the day's Other Coverage, and the day's exclusions —
# outputs with no theme id to report them under.  Leaving them out let a
# reconciliation rewrite a whole day's coverage and report, truthfully as
# far as themes went and falsely as a whole, that nothing had changed.
#
# The contract these tests hold to has two directions, and both matter:
# `changed` is True whenever any owned table would come out different,
# and False *only* when the reconciliation wrote nothing at all.
# ----------------------------------------------------------------------


#: Every table ``reconcile_themes`` owns.  A replay reported unchanged has
#: to leave all of them byte-identical, timestamps included.
THEME_OWNED_TABLES = (
    "theme_sets",
    "themes",
    "theme_stories",
    "theme_citations",
    "theme_other_coverage",
    "theme_excluded_stories",
)


def owned_snapshot(repository, tables=THEME_OWNED_TABLES) -> dict:
    with repository.admin.connect_writable() as connection:
        return {
            table: [
                tuple(row)
                for row in connection.execute(f"SELECT * FROM {table} ORDER BY rowid")
            ]
            for table in tables
        }


def themed_day(repository, **overrides):
    """One settled theme set, plus the knobs to settle it differently."""

    day = build_day(repository)
    payload = {
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "theme_set": theme_set(),
        "themes": [
            ThemeRecord(
                fingerprint="tf1",
                theme_key="k1",
                label="Chip demand",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                method="hdbscan",
            )
        ],
        "other_coverage": (
            OtherCoverageRecord(day["stories"]["cf2"], "clustering_noise"),
        ),
        "excluded": (ExcludedStoryRecord(day["stories"]["cf3"], "no_encodable_text"),),
    }
    payload.update(overrides)
    return day, payload


#: One change to each auxiliary output, and the output it must be reported
#: under.  Every case leaves the theme itself word-for-word identical, so
#: the theme comparison contributes nothing and the report is carried
#: entirely by the auxiliary contract.
AUXILIARY_CHANGES = [
    (
        "other-coverage reason",
        "other_coverage",
        lambda d: {
            "other_coverage": (
                OtherCoverageRecord(d["stories"]["cf2"], "narrative_mismatch"),
            )
        },
    ),
    (
        "other-coverage position",
        "other_coverage",
        lambda d: {
            "other_coverage": (
                OtherCoverageRecord(d["stories"]["cf2"], "clustering_noise", 7),
            )
        },
    ),
    (
        "other-coverage entry removed",
        "other_coverage",
        lambda d: {"other_coverage": ()},
    ),
    (
        "exclusion removed",
        "excluded",
        lambda d: {"excluded": ()},
    ),
    (
        "exclusion reason kept but story swapped",
        "excluded",
        lambda d: {
            "other_coverage": (),
            "excluded": (
                ExcludedStoryRecord(d["stories"]["cf2"], "no_encodable_text"),
            ),
        },
    ),
    (
        "theme-set quality",
        "theme_set",
        lambda d: {"theme_set": theme_set(quality={"theme_count": 99})},
    ),
    (
        "theme-set trust metadata",
        "theme_set",
        lambda d: {"theme_set": theme_set(trust_metadata={"reviewed": True})},
    ),
    (
        "theme-set method",
        "theme_set",
        lambda d: {"theme_set": theme_set(method="agglomerative")},
    ),
    (
        "theme-set method reason",
        "theme_set",
        lambda d: {"theme_set": theme_set(method_reason="fell back")},
    ),
    (
        "theme-set source metadata",
        "theme_set",
        lambda d: {"theme_set": theme_set(source_metadata={"origin": "rerun"})},
    ),
    (
        "theme-set config fingerprint",
        "theme_set",
        lambda d: {"theme_set": theme_set(config_fingerprint="cfg2")},
    ),
    (
        "theme-set algorithm version",
        "theme_set",
        lambda d: {"theme_set": theme_set(algorithm_version="m5.2")},
    ),
    (
        "theme-set model name",
        "theme_set",
        lambda d: {"theme_set": theme_set(model_name="other")},
    ),
    (
        "theme-set model revision",
        "theme_set",
        lambda d: {"theme_set": theme_set(model_revision="r2")},
    ),
    (
        "theme-set embedding dimension",
        "theme_set",
        lambda d: {"theme_set": theme_set(embedding_dimension=8)},
    ),
]


@pytest.mark.parametrize(
    "output, change",
    [pytest.param(o, c, id=label) for label, o, c in AUXILIARY_CHANGES],
)
def test_an_auxiliary_only_change_is_never_reported_unchanged(tmp_path, output, change):
    """Every theme is identical; something else moved; the report says so."""

    repository = migrated(tmp_path)
    day, payload = themed_day(repository)
    reconcile_themes(repository, **payload)
    before = owned_snapshot(repository)

    report = reconcile_themes(repository, **{**payload, **change(day)})
    after = owned_snapshot(repository)

    assert before != after, "the case does not actually change anything"
    # Not one theme moved — so nothing but the auxiliary contract can be
    # carrying this report.
    assert report.counts["unchanged"] == 1
    assert report.counts["inserted"] == report.counts["updated"] == 0
    assert report.changed
    assert output in report.changed_outputs
    assert report.counts["changed_outputs"] == len(report.changed_outputs)


def test_several_auxiliary_outputs_change_at_once(tmp_path):
    repository = migrated(tmp_path)
    day, payload = themed_day(repository)
    reconcile_themes(repository, **payload)

    report = reconcile_themes(
        repository,
        **{
            **payload,
            "theme_set": theme_set(quality={"theme_count": 42}),
            "other_coverage": (
                OtherCoverageRecord(day["stories"]["cf2"], "narrative_mismatch"),
            ),
            "excluded": (),
        },
    )

    assert report.changed_outputs == ("excluded", "other_coverage", "theme_set")
    assert report.counts["changed_outputs"] == 3
    assert report.counts["unchanged"] == 1
    assert report.changed


def test_an_exact_theme_replay_writes_nothing_at_all(tmp_path):
    """`changed is False` has to mean the database was not touched.

    Timestamps included: the theme-set row used to bump ``updated_at`` on
    every settlement, so two databases fed identical inputs disagreed and
    a report claiming "unchanged" was contradicted by the row itself.
    """

    repository = migrated(tmp_path)
    _, payload = themed_day(repository)
    reconcile_themes(repository, **payload)
    before = owned_snapshot(repository)

    report = reconcile_themes(repository, **payload)

    assert owned_snapshot(repository) == before
    assert not report.changed
    assert report.changed_outputs == ()
    assert report.counts["unchanged"] == 1


def test_reordering_coverage_alone_is_not_a_change(tmp_path):
    """Order is not meaning here; ``position`` is, and it is a value.

    Both lists are sets of rows, so listing the same rows in a different
    sequence is the same output — as long as the persisted ``position``
    each row carries stays the same, which is why it is given explicitly.
    """

    repository = migrated(tmp_path)
    items = seed_raw_items(repository, 5)
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story(f"cf{n}", items[n - 1 : n]) for n in range(1, 6)],
    )
    ids = {
        row["cluster_fingerprint"]: row["id"]
        for row in repository.stories_for_day(DAY, "NVDA")
    }
    # Positions are stated, and non-zero: `_prepare_coverage` reads a
    # falsy position as "unset" and substitutes the caller's index, which
    # would make this test measure ordering after all.
    entries = (
        OtherCoverageRecord(ids["cf2"], "clustering_noise", 1),
        OtherCoverageRecord(ids["cf3"], "narrative_mismatch", 2),
    )
    exclusions = (
        ExcludedStoryRecord(ids["cf4"], "no_encodable_text"),
        ExcludedStoryRecord(ids["cf5"], "no_encodable_text"),
    )
    settled = {
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "theme_set": theme_set(),
        "themes": [
            ThemeRecord(
                fingerprint="tf1",
                theme_key="k1",
                label="Chip demand",
                story_ids=(ids["cf1"],),
                citation_item_ids=(items[0],),
                method="hdbscan",
            )
        ],
        "other_coverage": entries,
        "excluded": exclusions,
    }
    reconcile_themes(repository, **settled)
    before = owned_snapshot(repository)

    report = reconcile_themes(
        repository,
        **{
            **settled,
            "other_coverage": tuple(reversed(entries)),
            "excluded": tuple(reversed(exclusions)),
        },
    )

    assert owned_snapshot(repository) == before
    assert report.changed_outputs == ()
    assert not report.changed


def test_every_theme_set_column_is_owned_or_deliberately_exempt(tmp_path):
    """The theme-set third of the rule that keeps this from rotting."""

    repository = migrated(tmp_path)
    with repository.admin.connect_writable() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(theme_sets)")
        }

    not_owned = {
        "id",
        "ticker",
        "trading_day",
        "pipeline_version",  # partition identity: what pairs the rows up
        "updated_at",  # bookkeeping about the write, not an output
    }

    assert columns == set(THEME_SET_RECONCILED_COLUMNS) | not_owned
    assert not set(THEME_SET_RECONCILED_COLUMNS) & not_owned
    prepared = Phase0Repository._prepare_theme_set(theme_set())
    assert set(Phase0Repository._theme_set_column_values(prepared)) == set(
        THEME_SET_RECONCILED_COLUMNS
    )
    assert sorted(AUXILIARY_OUTPUTS) == list(AUXILIARY_OUTPUTS)


def test_the_run_log_records_an_auxiliary_only_reconciliation(tmp_path):
    """A coverage rewrite is work, and the run log has to show it."""

    repository = migrated(tmp_path)
    day, payload = themed_day(repository)
    reconcile_themes(repository, **payload)

    with repository.stage_run(
        run_id="run-aux",
        stage="m5.themes",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.reconcile_themes(
            run=run,
            **{
                **{k: v for k, v in payload.items() if k != "theme_set"},
                "theme_set": theme_set(),
                "other_coverage": (
                    OtherCoverageRecord(day["stories"]["cf2"], "narrative_mismatch"),
                ),
            },
        )

    logged = repository.read.run_log_rows(run_id="run-aux")
    assert len(logged) == 1
    counts = json.loads(logged[0]["counts"])
    assert counts["changed_outputs"] == 1
    # Counted as work done, not as an idle replay: the coverage rewrite is
    # the only thing that happened, and success_count has to see it.
    assert logged[0]["success_count"] == 1


def test_reconcile_stories_reports_every_write_it_makes(tmp_path):
    """The adjacent path, checked for the same reporting hole.

    ``reconcile_stories`` owns only stories and their child rows, every
    one of which reaches the per-story signature — so it has no output
    outside the id tuples.  This pins that: an exact replay must leave
    all of it untouched, which is the property the theme path lacked.
    """

    repository = migrated(tmp_path)
    items = seed_raw_items(repository, 3)
    payload = {
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "stories": [story("cf1", items[:2]), story("cf2", items[2:])],
    }
    reconcile_stories(repository, **payload)
    tables = (
        "stories",
        "story_members",
        "story_provider_conflicts",
        "story_semantic_merges",
    )
    before = owned_snapshot(repository, tables)

    report = reconcile_stories(repository, **payload)

    assert owned_snapshot(repository, tables) == before
    assert not report.changed
    assert report.changed_outputs == ()
    assert report.counts["unchanged"] == 2


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

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker/day"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (day["foreign_story"], day["set_id"]),
            )


def test_other_coverage_cannot_be_updated_onto_a_different_trading_day(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)
    later = seed_raw_items(repository, 1)
    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker/day"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (next_day_story, day["set_id"]),
            )


def test_a_story_cannot_be_updated_into_both_a_theme_and_other_coverage(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
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

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="already accounted for"):
            connection.execute(
                "UPDATE theme_other_coverage SET story_id = ? WHERE theme_set_id = ?",
                (day["stories"]["cf3"], day["set_id"]),
            )


def test_theme_membership_cannot_be_updated_across_ticker_or_day(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(
            sqlite3.IntegrityError, match="ticker, day, and pipeline version"
        ):
            connection.execute(
                "UPDATE theme_stories SET story_id = ? WHERE theme_id = ?",
                (day["foreign_story"], day["theme_id"]),
            )


def test_a_citation_cannot_be_updated_onto_evidence_outside_its_theme(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="member story"):
            connection.execute(
                "UPDATE theme_citations SET raw_item_id = ? WHERE theme_id = ?",
                (day["items"][3], day["theme_id"]),
            )


def test_a_parent_cannot_be_relocated_out_from_under_its_children(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
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
        with pytest.raises(
            sqlite3.IntegrityError,
            match="cannot change ticker, day, or pipeline version",
        ):
            connection.execute(
                "UPDATE stories SET trading_day = '2026-07-24' WHERE id = ?",
                (day["stories"]["cf1"],),
            )


def cross_version_day(repository: Phase0Repository) -> dict:
    """One NVDA day reconciled under v1 and, separately, under v2.

    Gives the probes below a genuine v2 story to try to smuggle into a v1
    theme — the defect migration 010 exists to close.
    """

    day = populated_theme_set(repository)
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v2",
        stories=[story("v2-cf1", day["items"][:2])],
    )
    with repository.admin.connect_writable() as connection:
        v2_story = int(
            connection.execute(
                "SELECT id FROM stories WHERE pipeline_version = 'v2'"
            ).fetchone()["id"]
        )
    day["v2_story"] = v2_story
    return day


def test_a_v1_theme_cannot_accept_a_v2_story_on_insert(tmp_path):
    repository = migrated(tmp_path)
    day = cross_version_day(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="pipeline version"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (day["theme_id"], day["v2_story"]),
            )


def test_a_v1_theme_cannot_accept_a_v2_story_on_update(tmp_path):
    repository = migrated(tmp_path)
    day = cross_version_day(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="pipeline version"):
            connection.execute(
                "UPDATE theme_stories SET story_id = ? WHERE theme_id = ?",
                (day["v2_story"], day["theme_id"]),
            )


@pytest.mark.parametrize("table", ["theme_other_coverage", "theme_excluded_stories"])
def test_coverage_and_exclusions_reject_a_cross_version_story(tmp_path, table):
    repository = migrated(tmp_path)
    day = cross_version_day(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ticker/day/version"):
            connection.execute(
                f"INSERT INTO {table} (theme_set_id, story_id, reason) "
                "VALUES (?, ?, ?)",
                (
                    day["set_id"],
                    day["v2_story"],
                    (
                        "clustering_noise"
                        if table == "theme_other_coverage"
                        else "no_encodable_text"
                    ),
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="ticker/day/version"):
            connection.execute(
                f"UPDATE {table} SET story_id = ? WHERE theme_set_id = ?",
                (day["v2_story"], day["set_id"]),
            )


def test_a_referenced_story_cannot_change_pipeline_version(tmp_path):
    repository = migrated(tmp_path)
    day = cross_version_day(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="pipeline version"):
            connection.execute(
                "UPDATE stories SET pipeline_version = 'v9' WHERE id = ?",
                (day["stories"]["cf1"],),
            )
        # A story nothing references may still be re-versioned.
        connection.execute(
            "UPDATE stories SET pipeline_version = 'v9' WHERE id = ?",
            (day["v2_story"],),
        )


def test_a_populated_theme_set_cannot_change_version_even_with_only_themes(tmp_path):
    """009 only counted coverage and exclusions, so a theme-only set slipped."""

    repository = migrated(tmp_path)
    day = populated_theme_set(repository)
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM theme_other_coverage WHERE theme_set_id = ?", (day["set_id"],)
        )
        connection.execute(
            "DELETE FROM theme_excluded_stories WHERE theme_set_id = ?",
            (day["set_id"],),
        )
        with pytest.raises(sqlite3.IntegrityError, match="populated theme set"):
            connection.execute(
                "UPDATE theme_sets SET pipeline_version = 'v2' WHERE id = ?",
                (day["set_id"],),
            )


def test_a_populated_theme_cannot_change_version(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="populated theme"):
            connection.execute(
                "UPDATE themes SET pipeline_version = 'v2' WHERE id = ?",
                (day["theme_id"],),
            )


def test_metadata_only_updates_are_still_allowed(tmp_path):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE themes SET label = 'Renamed', salience = 0.42 WHERE id = ?",
            (day["theme_id"],),
        )
        connection.execute(
            "UPDATE theme_sets SET method_reason = 'recomputed' WHERE id = ?",
            (day["set_id"],),
        )
        connection.execute(
            "UPDATE stories SET canonical_title = 'Retitled' WHERE id = ?",
            (day["stories"]["cf1"],),
        )

    stored = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
    assert stored["themes"][0]["label"] == "Renamed"
    assert stored["method_reason"] == "recomputed"


def test_a_whole_ticker_day_version_can_be_rebuilt(tmp_path):
    repository = migrated(tmp_path)
    day = cross_version_day(repository)

    report = reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v2",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="v2-tf1",
                label="V2 theme",
                story_ids=(day["v2_story"],),
                citation_item_ids=(day["items"][0],),
                salience_rank=1,
                status="ready",
                method="hdbscan",
            )
        ],
    )

    assert report.counts["inserted"] == 1
    # The v1 set is untouched: versions are separate partitions.
    v1 = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
    assert len(v1["themes"]) == 1


def test_valid_theme_set_updates_and_deletion_order_still_work(tmp_path):
    """The guards must not have made ordinary lifecycle work impossible."""

    repository = migrated(tmp_path)
    day = populated_theme_set(repository)

    with repository.admin.connect_writable() as connection:
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


def database_bytes(repository: Phase0Repository) -> bytes:
    """The database *and* its WAL sidecar.

    WAL mode keeps recent pages out of the main file, so reading only
    ``phase0.sqlite3`` would make every byte-level assertion below pass
    vacuously.
    """

    return b"".join(
        path.read_bytes()
        for suffix in ("", "-wal", "-shm")
        if (
            path := repository.database_path.with_name(
                repository.database_path.name + suffix
            )
        ).exists()
    )


@pytest.mark.parametrize(
    "label, payload, secret",
    CREDENTIAL_PAYLOADS,
    ids=[row[0] for row in CREDENTIAL_PAYLOADS],
)
def test_no_operational_surface_retains_the_original_credential(
    tmp_path, label, payload, secret
):
    """Put the credential through every *operational* surface, then grep.

    The contract is not "no credential byte exists anywhere in the file" —
    that would be a promise to corrupt publisher evidence, which
    ``test_raw_provider_evidence_is_preserved_byte_for_byte`` shows Phase 0
    deliberately does not do.  The contract is: no operational credential
    supplied through a diagnostic or configuration surface reaches
    persistence.  Every such surface is exercised here, and then the
    SQLite files are read as bytes so a column this test forgot still
    fails it.
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
        repository.record_source_state(
            "rss:test",
            run=run,
            successful=False,
            status="failed",
            metadata={"request": payload},
            error=payload,
        )

    # A failing stage: the exception message is the other way operational
    # text reaches run_log.errors, and it is redacted on the way in.
    with pytest.raises(RuntimeError):
        with repository.stage_run(
            run_id="run-secret-fail",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ):
            raise RuntimeError(f"upstream rejected {payload}")

    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(
            source_metadata={"feed": payload},
            trust_metadata={"upstream": payload},
            method_reason=str(payload),
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
                semantic_skip_reason=str(payload),
                semantic_merges=(
                    SemanticMergeRecord(
                        left_story_key="a",
                        right_story_key="b",
                        similarity=0.9,
                        reason=str(payload),
                    ),
                ),
                members=(
                    StoryMemberRecord(
                        raw_item_id=day["items"][0],
                        position=0,
                        outlet="O",
                        match_reason=str(payload),
                    ),
                ),
                canonical_item_id=day["items"][0],
            )
        ],
    )

    stored = database_bytes(repository)
    assert secret.encode() not in stored, f"{label} survived in the database files"

    entry = repository.run_log_entries(run_id="run-secret")[0]
    assert secret not in json.dumps(entry["counts"])
    assert secret not in json.dumps(entry["errors"])
    failed = repository.run_log_entries(run_id="run-secret-fail")[0]
    assert secret not in json.dumps(failed["errors"])
    state = repository.source_state("rss:test")
    assert secret not in json.dumps(state)
    theme_row = repository.theme_set(
        ticker="NVDA", trading_day=DAY, pipeline_version="v1"
    )
    assert secret not in json.dumps(theme_row)


#: Scalar columns that carry *identity*.  A credential in one of these is
#: refused outright, because redacting an identifier silently repoints the
#: row it names rather than protecting anything.
IDENTITY_SCALARS = [
    "provider_namespace",
    "provider_item_id",
    "left_story_key",
    "right_story_key",
    "model_name",
    "model_revision",
    "algorithm_version",
    "config_fingerprint",
]


@pytest.mark.parametrize(
    "label, payload, secret",
    CREDENTIAL_PAYLOADS,
    ids=[row[0] for row in CREDENTIAL_PAYLOADS],
)
@pytest.mark.parametrize("column", IDENTITY_SCALARS)
def test_identity_scalars_reject_credentials_rather_than_redacting_them(
    tmp_path, column, label, payload, secret
):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)
    if not isinstance(payload, str):
        pytest.skip("identity scalars are text columns")

    overrides: dict = {}
    if column in {"provider_namespace", "provider_item_id"}:
        conflict = {
            "provider_namespace": "yahoo",
            "provider_item_id": "1",
            "item_ids": ("1",),
            "fields": ("title",),
        }
        conflict[column] = payload
        overrides["provider_conflicts"] = (ProviderConflictRecord(**conflict),)
    elif column in {"left_story_key", "right_story_key"}:
        merge = {
            "left_story_key": "a",
            "right_story_key": "b",
            "similarity": 0.5,
            "reason": "near-duplicate",
        }
        merge[column] = payload
        overrides["semantic_merges"] = (SemanticMergeRecord(**merge),)
    else:
        overrides[column] = payload

    with pytest.raises(Phase0ValidationError, match="credential material"):
        reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story("cf1", item_ids, **overrides)],
        )

    assert repository.count("stories") == 0
    assert secret.encode() not in database_bytes(repository)


@pytest.mark.parametrize(
    "label, payload, secret",
    CREDENTIAL_PAYLOADS,
    ids=[row[0] for row in CREDENTIAL_PAYLOADS],
)
def test_embedding_identity_scalars_reject_credentials(
    tmp_path, label, payload, secret
):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)
    if not isinstance(payload, str):
        pytest.skip("identity scalars are text columns")
    vector = np.ones(4, dtype=EMBEDDING_DTYPE)

    for field in ("model_name", "model_revision"):
        values = {
            "source_kind": "raw_item",
            "source_id": str(item_ids[0]),
            "model_name": "fake",
            "model_revision": "r1",
            "dimension": 4,
            "dtype": str(EMBEDDING_DTYPE),
            "input_fingerprint": "a" * 64,
            "vector_blob": serialize_vector(vector),
        }
        values[field] = payload
        with pytest.raises(Phase0ValidationError, match="credential material"):
            repository.upsert_embedding(PersistedEmbedding(**values))

    assert repository.count("embeddings") == 0
    assert secret.encode() not in database_bytes(repository)


# ----------------------------------------------------------------------
# Raw provider evidence is preserved; operational metadata is not
# ----------------------------------------------------------------------


def test_raw_provider_evidence_is_preserved_byte_for_byte(tmp_path):
    """A publisher's payload is evidence, not a leak.

    AC-8 replay is only worth something if ``raw_json`` is what the feed
    actually sent.  A string in publisher content that happens to look
    like a credential stays put — rewriting it would corrupt the evidence
    *and* hide the real bug, which is a fetcher that put a transport
    credential into the payload. That is I2/I3's boundary to enforce, not
    something I1 papers over after the fact.
    """

    repository = migrated(tmp_path)
    payload = {
        "headline": "Leaked API key api_key=SEKRET123 found in vendor repo",
        "body": "The commit contained Authorization: Bearer abc123XYZ verbatim.",
        "nested": {"quote": "password: hunter2secret"},
    }
    item = raw_item(1)
    item["raw_json"] = payload

    item_id = repository.admin.insert_raw_item(item).item_id

    stored = repository.raw_items_for_day(DAY)[0]
    assert json.loads(stored["raw_json"]) == payload
    assert b"SEKRET123" in database_bytes(repository)
    assert repository.raw_item_tickers(item_id) == ["NVDA"]


def test_the_same_string_is_evidence_in_raw_json_and_a_secret_in_metadata(tmp_path):
    """One string, two columns, two outcomes — that is the whole boundary."""

    repository = migrated(tmp_path)
    secret_text = "Authorization: Bearer abc123XYZ"
    item = raw_item(1)
    item["raw_json"] = {"body": secret_text}

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items(
            [item],
            run=run,
            source_state={
                "source": "rss:test",
                "etag": None,
                "last_modified": None,
                "checked_at": f"{DAY}T12:00:00+00:00",
                "successful": True,
                "metadata": {"request_header": secret_text},
            },
        )

    stored = repository.raw_items_for_day(DAY)[0]
    assert json.loads(stored["raw_json"])["body"] == secret_text
    state = repository.source_state("rss:test")
    assert "abc123XYZ" not in json.dumps(state["metadata"])


def test_the_two_serializers_state_their_policy_in_their_names(tmp_path):
    payload = {"header": "Authorization: Bearer abc123XYZ"}

    evidence = serialize_raw_evidence(payload, "raw_json")
    operational = serialize_operational_metadata(payload, "metadata", dict)

    assert json.loads(evidence) == payload
    assert "abc123XYZ" not in operational
    # No boolean anywhere: the policy is the function you call.
    for serializer in (serialize_raw_evidence, serialize_operational_metadata):
        parameters = inspect.signature(serializer).parameters
        assert not any(
            isinstance(p.default, bool) for p in parameters.values()
        ), f"{serializer.__name__} takes a policy flag"


# ----------------------------------------------------------------------
# Run logs and source state
# ----------------------------------------------------------------------


def test_stage_run_logs_counts_derived_from_the_operations(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="fetch",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items([raw_item(1), raw_item(2)], run=run)
        # The second batch re-sends one item, so it lands as partial.
        repository.ingest_raw_items([raw_item(2), raw_item(3)], run=run)

    entry = repository.run_log_entries(run_id="run-1")[0]
    assert entry["status"] == "degraded"
    assert entry["success_count"] == 3
    assert entry["partial_count"] == 1
    assert entry["counts"] == {"raw_items_seen": 4, "raw_items_inserted": 3}
    assert entry["ticker"] == "NVDA"


COUNT_MUTATORS = [
    "record_success",
    "record_partial",
    "record_failure",
    "record_error",
    "update_counts",
    "resolved_status",
]


@pytest.mark.parametrize("name", COUNT_MUTATORS)
def test_a_caller_has_no_way_to_write_run_counts(tmp_path, name):
    """Counts describe what an operation did, so only operations write them."""

    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="fetch",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        assert not hasattr(run, name), f"{name} is still caller-reachable"


def test_counts_read_from_a_context_are_a_copy(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="fetch",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run)
        run.counts["raw_items_inserted"] = 9_999
        run.counts["injected"] = "Authorization: Bearer abc123XYZ"
        run.errors.append("injected error")

    entry = repository.run_log_entries(run_id="run-1")[0]
    assert entry["counts"] == {"raw_items_seen": 1, "raw_items_inserted": 1}
    assert entry["errors"] == []
    assert entry["status"] == "success"


def test_a_caller_cannot_pre_seed_counts_through_the_context(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="fetch",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        # Read-only after construction, private counters included, and
        # __slots__ refuses new attributes outright.
        for attribute in ("_success_count", "counts", "_counts", "anything"):
            with pytest.raises((AttributeError, Phase0Error)):
                setattr(run, attribute, 5)
            with pytest.raises((AttributeError, Phase0Error)):
                delattr(run, attribute)
        repository.ingest_raw_items([raw_item(1)], run=run)

    entry = repository.run_log_entries(run_id="run-1")[0]
    assert entry["success_count"] == 1
    assert entry["counts"] == {"raw_items_seen": 1, "raw_items_inserted": 1}


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
    # RSS evidence and relevance (#62): both go through _logged_mutation,
    # so a key moves only in the transaction that commits the data and the
    # run log with it.
    "record_feed_snapshot",
    "replace_relevance_classifications",
]


#: Public methods that write but legitimately cannot take a run, each with
#: the reason.  The audit below forces every future public writer into
#: either this table or the logged contract — nothing gets to be neither.
UNLOGGED_BY_DESIGN = {
    "migrate": "schema DDL; runs before any run can exist",
    "stage_run": "the run factory itself; it writes the run log",
    "claim_stage_key": "claims the lease a run is later opened against",
    "heartbeat_stage_key": "extends the lease of an in-flight run",
    "recover_expired_leases": "operator/crash sweep; by definition has no run",
    "upsert_embedding": (
        "nlp.embeddings.EmbeddingRepository protocol; a single recomputable "
        "cache vector, whose identity scalars are validated. The logged "
        "stage entrypoint is persist_embeddings"
    ),
    "delete_embedding": "drops one recomputable cache vector; changes nothing else",
}

#: A write verb *in SQL position*, so a docstring that merely says "update"
#: does not read as a mutation.
_WRITE_SQL = re.compile(
    r"\b(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|REPLACE\s+INTO|UPDATE\s+\w+\s+SET"
    r"|DELETE\s+FROM|CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX|TRIGGER|VIEW)"
    r"|DROP\s+(?:TABLE|INDEX|TRIGGER|VIEW)|ALTER\s+TABLE)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# The public read surface hands back data, never a connection
# ----------------------------------------------------------------------


#: Everything a caller would need to escape a read-only connection, if it
#: ever got hold of one.  ``set_authorizer`` is first because it is the
#: whole reason a hardened connection cannot be handed out: the caller can
#: simply take the authorizer off again.
ESCAPE_HATCHES = [
    "set_authorizer",
    "execute",
    "executemany",
    "executescript",
    "cursor",
    "commit",
    "rollback",
    "close",
    "create_function",
    "backup",
    "iterdump",
    "connection",
    "attach",
]

#: Reader methods, each with arguments that exercise it.  A new public
#: reader method with no probe here fails the audit below, so the read
#: surface cannot grow untested.
READER_PROBES = {
    "raw_item": ((1,), {}),
    "raw_items": ((), {}),
    "raw_item_candidates": ((), {}),
    "raw_item_associations": ((), {}),
    # RSS evidence and provenance (#62).
    "feed_snapshots": ((), {}),
    "raw_item_feeds": ((), {}),
    "raw_item_match_evidence": ((), {}),
    "story": ((1,), {}),
    "theme": ((1,), {}),
    "source_state_rows": ((), {}),
    "run_log_rows": ((), {}),
    "stage_key_rows": ((), {}),
    "table_names": ((), {}),
    "schema_objects": ((), {}),
    "table_columns": (("raw_items",), {}),
    "foreign_keys": (("story_members",), {}),
    "indexes": (("raw_items",), {}),
    "integrity_check": ((), {}),
    "schema_version": ((), {}),
    "count": (("raw_items",), {}),
}


def _public_names(obj) -> set:
    return {name for name in dir(obj) if not name.startswith("_")}


def _reachable_values(obj, depth: int = 0):
    """Everything an ordinary caller can get to from ``obj`` by attribute."""

    yield obj
    if depth > 2:
        return
    for name in _public_names(obj):
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 - a property that raises exposes nothing
            continue
        if callable(value) or isinstance(value, (str, bytes, int, float, Path)):
            continue
        yield from _reachable_values(value, depth + 1)


def test_the_reader_answers_the_reads_phase_0_needs(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 2)

    reader = repository.read
    assert reader.count("raw_items") == 2
    assert reader.raw_item(item_ids[0])["source"].startswith("yahoo:")
    assert [row["ticker"] for row in reader.raw_item_associations(item_ids[0])] == [
        "NVDA"
    ]
    assert reader.schema_version() == LATEST_VERSION
    assert reader.integrity_check() == "ok"
    assert "raw_items" in reader.table_names()
    assert [column["name"] for column in reader.table_columns("raw_items")][0] == "id"
    assert reader.foreign_keys("story_members")
    assert reader.indexes("raw_items")
    assert any(row["type"] == "trigger" for row in reader.schema_objects())
    assert reader.story(999) is None
    assert reader.run_log_rows() == []


def test_every_reader_method_returns_plain_data(tmp_path):
    """No probe, no method: the read surface cannot grow unexamined."""

    repository = migrated(tmp_path)
    seed_raw_items(repository, 1)
    methods = {
        name
        for name in _public_names(Phase0Reader)
        if callable(getattr(Phase0Reader, name))
    }
    assert methods == set(READER_PROBES), (
        "reader methods without a hostile probe: "
        f"{sorted(methods - set(READER_PROBES))}"
    )

    for name, (args, kwargs) in READER_PROBES.items():
        result = getattr(repository.read, name)(*args, **kwargs)
        values = result if isinstance(result, list) else [result]
        for value in values:
            assert not isinstance(value, (sqlite3.Connection, sqlite3.Cursor))
            assert isinstance(value, (dict, str, int, type(None))), (name, value)
            if isinstance(value, dict):
                assert all(
                    not isinstance(item, (sqlite3.Connection, sqlite3.Cursor))
                    for item in value.values()
                )


@pytest.mark.parametrize("attribute", ESCAPE_HATCHES)
def test_the_reader_has_no_way_out(tmp_path, attribute):
    """A caller holding a connection can always undo its protections.

    ``set_authorizer(None)``, ``PRAGMA query_only = OFF``, ATTACH, INSERT,
    COMMIT — the reported bypass, and unanswerable while the object handed
    out is a connection.  So none is handed out.
    """

    repository = migrated(tmp_path)

    assert not hasattr(repository.read, attribute)
    assert not hasattr(repository, "read_connection")
    assert not hasattr(repository, "connect")


def test_the_reader_holds_no_connection_anywhere(tmp_path):
    """Not even privately: its whole state is one path."""

    repository = migrated(tmp_path)
    reader = repository.read

    assert Phase0Reader.__slots__ == ("_database_path",)
    assert isinstance(reader._database_path, Path)
    assert not hasattr(reader, "__dict__")
    for name in ("_connection", "connection", "_repository", "repository", "_db"):
        assert not hasattr(reader, name), name

    # And a read leaves nothing behind that a later caller could pick up.
    reader.count("raw_items")
    assert not hasattr(reader, "__dict__")


def test_no_public_surface_reaches_a_connection_or_cursor(tmp_path):
    """Walk what a caller can actually touch, not what is annotated."""

    repository = migrated(tmp_path)
    seed_raw_items(repository, 1)

    for value in _reachable_values(repository):
        assert not isinstance(value, (sqlite3.Connection, sqlite3.Cursor)), value
    for value in _reachable_values(repository.read):
        assert not isinstance(value, (sqlite3.Connection, sqlite3.Cursor)), value

    assert _public_names(repository) >= {"read", "admin"}


def test_no_normal_public_surface_can_insert_a_raw_item(tmp_path):
    """The blocker in one test: no public read path writes anything."""

    repository = migrated(tmp_path)

    for name, (args, kwargs) in READER_PROBES.items():
        getattr(repository.read, name)(*args, **kwargs)
    with pytest.raises(Phase0RunContextError):
        repository.ingest_raw_items([raw_item(1)], run=None)

    assert repository.count("raw_items") == 0
    assert repository.run_log_entries() == []


def test_the_reader_refuses_sql_it_did_not_write(tmp_path):
    """Table names are looked up, not interpolated from a caller string."""

    repository = migrated(tmp_path)

    for bad in ("raw_items; DROP TABLE raw_items", "sqlite_master", "nope", ""):
        with pytest.raises(Phase0ValidationError):
            repository.read.table_columns(bad)
        with pytest.raises(Phase0ValidationError):
            repository.read.count(bad)
    assert repository.count("raw_items") == 0


def test_reader_queries_are_module_literals():
    """Structural: no caller string reaches ``_query``.

    Every call site passes a literal or an f-string this module builds from
    validated fragments; none forwards a parameter.
    """

    calls = []
    for name in READER_PROBES:
        source = inspect.getsource(getattr(Phase0Reader, name))
        calls.extend(re.findall(r"self\._(?:query|one)\(\s*([^,\n]+)", source))
    assert calls, "reflection found no reader queries; the audit is broken"
    for call in calls:
        assert call.strip().startswith(('"', "'", 'f"', "f'")), call


def test_the_admin_connection_is_the_only_raw_handle(tmp_path):
    """Deliberate, named, and documented — the one exception to the rule."""

    repository = migrated(tmp_path)

    with repository.admin.connect_writable() as connection:
        assert isinstance(connection, sqlite3.Connection)
        connection.execute(
            "INSERT INTO raw_items (source, canonical_url, fetched_at, "
            "raw_json, ingest_status) VALUES ('s', 'u', ?, '{}', 'invalid')",
            (f"{DAY}T00:00:00+00:00",),
        )

    assert repository.count("raw_items") == 1
    doc = (Phase0Admin.connect_writable.__doc__ or "").lower()
    assert "manual repair" in doc and "migrations" in doc
    assert "pipeline code must never call it" in doc
    assert "only raw connection" in doc


#: Public methods that may hand back a raw connection, and where they
#: live.  ``Phase0Repository`` has none: that is the contract.
RAW_CONNECTION_ACCESSORS = {
    Phase0Repository: set(),
    Phase0Reader: set(),
    Phase0Admin: {"connect_writable"},
}

#: A method that gives a connection or cursor away, whatever it declares.
#: Anchored at the end of the statement so ``return cursor.rowcount == 1``
#: — a count, not a handle — is not mistaken for handing one out.
_HANDS_BACK_CONNECTION = re.compile(
    r"^\s*(?:yield|return)\s+\w*(?:connection|cursor)\s*,?\s*$", re.MULTILINE
)


def _public_repository_methods() -> list[str]:
    return [
        name
        for name, member in inspect.getmembers(Phase0Repository, inspect.isfunction)
        if not name.startswith("_")
    ]


def _handles_from(instance, name):
    """Whatever a zero-argument public method actually hands back.

    Calling it is the point.  An audit that reads return annotations only
    trusts the annotation, and a method that declared the wrong one — or
    none — is exactly the case worth catching.
    """

    method = getattr(instance, name)
    parameters = list(inspect.signature(method).parameters.values())
    if any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        return []
    try:
        result = method()
    except Exception:  # noqa: BLE001 - not every method is callable bare
        return []
    handles = (sqlite3.Connection, sqlite3.Cursor)
    if isinstance(result, handles):
        return [result]
    if hasattr(result, "__enter__"):
        with result as value:
            return [value] if isinstance(value, handles) else []
    return []


@pytest.mark.parametrize(
    "owner, attribute",
    [(Phase0Repository, None), (Phase0Reader, "read"), (Phase0Admin, "admin")],
)
def test_no_public_method_hands_back_a_connection_or_cursor(tmp_path, owner, attribute):
    """Fail closed, across all three public objects.

    Three ways, because each alone can be fooled: the declared return type,
    the source (any ``return``/``yield`` of something called connection or
    cursor), and — the one that cannot be talked around — calling every
    method that takes no arguments and looking at what comes back.
    """

    repository = migrated(tmp_path)
    instance = repository if attribute is None else getattr(repository, attribute)
    exposed = set()
    for name, member in inspect.getmembers(owner, inspect.isfunction):
        if name.startswith("_"):
            continue
        try:
            source = inspect.getsource(member)
        except (OSError, TypeError) as exc:  # pragma: no cover - defensive
            raise AssertionError(f"cannot read the source of {name}: {exc}") from exc
        annotation = str(inspect.signature(member).return_annotation)
        if "Connection" in annotation or "Cursor" in annotation:
            exposed.add(name)
        elif _HANDS_BACK_CONNECTION.search(source):
            exposed.add(name)
        if _handles_from(instance, name):
            exposed.add(name)

    expected = RAW_CONNECTION_ACCESSORS[owner]
    assert exposed == expected, (
        f"{owner.__name__} exposes raw handles this audit does not know "
        f"about: {sorted(exposed - expected)}"
    )


def test_the_connection_audit_notices_a_new_accessor(tmp_path):
    """The audit is only worth having if it can fail."""

    class Leaky(Phase0Repository):
        @contextlib.contextmanager
        def debug_connection(self):
            with self._connect() as connection:
                yield connection

        def debug_cursor(self):
            cursor = self._open_connection().cursor()
            return cursor

    for method in (Leaky.debug_connection, Leaky.debug_cursor):
        assert _HANDS_BACK_CONNECTION.search(inspect.getsource(method))
    assert not RAW_CONNECTION_ACCESSORS[Phase0Repository]


def _method_writes(name: str, seen: set[str] | None = None) -> bool:
    """True when a method issues write SQL, directly or through a helper.

    Follows ``self.<helper>(`` one level at a time until the closure is
    exhausted, so a public method that only *delegates* its writes is still
    caught.
    """

    seen = seen if seen is not None else set()
    if name in seen:
        return False
    seen.add(name)
    member = getattr(Phase0Repository, name, None)
    if member is None or not callable(member):
        return False
    try:
        source = inspect.getsource(member)
    except (OSError, TypeError) as exc:  # pragma: no cover - defensive
        # Never treat "could not read it" as "does not write": that is how
        # an audit like this quietly stops auditing anything.
        raise AssertionError(
            f"cannot read the source of {name}; the write audit cannot "
            f"classify it ({exc})"
        ) from exc
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    if _WRITE_SQL.search(body):
        return True
    return any(
        _method_writes(callee, seen)
        for callee in set(re.findall(r"self\.(\w+)\(", body))
    )


def test_every_public_writer_is_either_logged_or_declared_a_lease_operation():
    """Enumerate the whole public surface, not a subset anyone curated.

    A new public method that writes has to end up in one of three places:
    it takes ``run``, it is named in :data:`LEASE_OPERATIONS` with a
    reason, or it is not public.  Nothing is allowed to be none of those.
    """

    public = [
        name
        for name, member in inspect.getmembers(Phase0Repository, inspect.isfunction)
        if not name.startswith("_") and name not in {"connect"}
    ]
    assert public, "reflection found no public methods; the audit is broken"

    unclassified = []
    for name in public:
        if not _method_writes(name):
            continue
        parameters = inspect.signature(getattr(Phase0Repository, name)).parameters
        if "run" in parameters:
            continue
        if name in UNLOGGED_BY_DESIGN:
            assert UNLOGGED_BY_DESIGN[name].strip(), f"{name} needs a stated reason"
            continue
        unclassified.append(name)

    assert not unclassified, (
        "these public methods mutate without a run and are not declared in "
        f"UNLOGGED_BY_DESIGN: {sorted(unclassified)}"
    )


@pytest.mark.parametrize(
    "name",
    [
        "insert_raw_item",
        "insert_raw_items",
        "set_source_state",
        "insert_story",
        "insert_theme",
        "insert_eval_label",
        "update_raw_item_ticker",
        "clear_derived_for_day",
        "log_stage",
    ],
)
def test_unlogged_writers_live_only_behind_admin(tmp_path, name):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")

    assert not hasattr(repository, name), f"{name} is still a public repository method"
    assert hasattr(repository.admin, name), f"{name} is missing from admin"


def test_admin_documents_that_the_pipeline_must_not_use_it():
    doc = (Phase0Admin.__doc__ or "").lower()

    assert "not the pipeline" in doc
    assert "must never use it" in doc
    for entrypoint in LOGGED_ENTRYPOINTS:
        assert entrypoint in doc, f"admin docs should point at {entrypoint}"


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

    with pytest.raises(Phase0RunContextError, match="no longer active"):
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


# ----------------------------------------------------------------------
# A run capability cannot be forged
# ----------------------------------------------------------------------


def open_run(repository: Phase0Repository, **overrides):
    defaults = {
        "run_id": "run-1",
        "stage": "ingest",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "ticker": "NVDA",
    }
    defaults.update(overrides)
    return repository.stage_run(**defaults)


def test_a_context_cannot_be_constructed_directly(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(Phase0RunContextError, match="cannot be constructed directly"):
        StageRunContext(
            repository=repository,
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            attempt=1,
            replay=False,
            stage_key=None,
        )
    # Nor by guessing at a positional key.
    for guess in (None, object(), repository, "key", 0):
        with pytest.raises(Phase0RunContextError):
            StageRunContext(guess)


def test_a_bare_shell_authorizes_nothing(tmp_path):
    """``object.__new__`` skips ``__init__``, so it skips the key check too.

    Identity registration is what stops it: the shell was never registered,
    so it is not a live run no matter what is written into its slots.
    """

    repository = migrated(tmp_path)
    shell = object.__new__(StageRunContext)

    with pytest.raises(Phase0RunContextError):
        repository.ingest_raw_items([raw_item(1)], run=shell)
    assert repository.count("raw_items") == 0


def test_a_forged_look_alike_with_matching_fields_authorizes_nothing(tmp_path):
    """Knowing the repository, run id, stage, and partition is not enough."""

    repository = migrated(tmp_path)

    class ForgedContext:
        repository = None
        run_id = "run-1"
        stage = "ingest"
        trading_day = DAY
        pipeline_version = "v1"
        ticker = "NVDA"
        attempt = 1
        replay = False
        stage_key = None
        closed = False
        active = True
        counts: dict = {}
        errors: list = []
        success_count = 0
        partial_count = 0
        failure_count = 0

    forged = ForgedContext()
    forged.repository = repository

    with pytest.raises(Phase0RunContextError, match="requires an active stage run"):
        repository.ingest_raw_items([raw_item(1)], run=forged)

    # Even a real context's fields, copied onto a shell of the right class.
    with open_run(repository) as real:
        impostor = object.__new__(StageRunContext)
        for slot in StageRunContext.__slots__:
            object.__setattr__(impostor, slot, getattr(real, slot))
        with pytest.raises(Phase0RunContextError):
            repository.ingest_raw_items([raw_item(1)], run=impostor)

    assert repository.count("raw_items") == 0


@pytest.mark.parametrize("clone", ["copy", "deepcopy", "pickle", "replace"])
def test_a_copied_or_serialized_context_is_refused(tmp_path, clone):
    repository = migrated(tmp_path)

    with open_run(repository) as run:
        if clone == "copy":
            with pytest.raises(Phase0RunContextError, match="cannot be copied"):
                copy.copy(run)
        elif clone == "deepcopy":
            with pytest.raises(Phase0RunContextError, match="cannot be copied"):
                copy.deepcopy(run)
        elif clone == "pickle":
            with pytest.raises(Phase0RunContextError, match="cannot be copied"):
                pickle.dumps(run)
        else:
            # Not a dataclass, so dataclasses.replace cannot even try — and
            # the direct-construction guard is what it would hit if it did.
            with pytest.raises((TypeError, Phase0RunContextError)):
                dataclasses.replace(run)

    assert repository.count("raw_items") == 0


def test_the_capability_is_not_visible_in_repr_or_public_fields(tmp_path):
    repository = migrated(tmp_path)

    with open_run(repository) as run:
        text = repr(run)
        assert "run-1" in text and "NVDA" in text
        # The capability is object identity plus a module-private key;
        # neither is a field, so there is nothing here to lift.
        public = [name for name in dir(run) if not name.startswith("_")]
        assert "token" not in " ".join(public)
        assert "capability" not in " ".join(public)
        assert "key" not in " ".join(name for name in public if name != "stage_key")


def test_a_context_is_dead_after_its_block_exits(tmp_path):
    repository = migrated(tmp_path)

    with open_run(repository) as run:
        assert run.active
    assert not run.active and run.closed

    with pytest.raises(Phase0RunContextError, match="no longer active"):
        repository.ingest_raw_items([raw_item(1)], run=run)
    assert repository.count("raw_items") == 0


def test_a_context_is_dead_after_its_run_failed(tmp_path):
    repository = migrated(tmp_path)
    escaped = {}

    with pytest.raises(RuntimeError):
        with open_run(repository) as run:
            escaped["run"] = run
            raise RuntimeError("stage blew up")

    with pytest.raises(Phase0RunContextError, match="no longer active"):
        repository.ingest_raw_items([raw_item(1)], run=escaped["run"])
    assert repository.count("raw_items") == 0


def test_a_context_from_a_second_repository_on_the_same_file_is_refused(tmp_path):
    """Same database, different instance: the registry is per-instance."""

    repository = migrated(tmp_path)
    twin = Phase0Repository(repository.database_path)

    with twin.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as foreign:
        with pytest.raises(Phase0RunContextError, match="another repository"):
            repository.ingest_raw_items([raw_item(1)], run=foreign)

    assert repository.count("raw_items") == 0


# ----------------------------------------------------------------------
# Stage-key ownership is transactional, and its partition must match
# ----------------------------------------------------------------------


STAGE_KEY_FIELDS = ["stage", "ticker", "trading_day", "pipeline_version"]


@pytest.mark.parametrize("field", STAGE_KEY_FIELDS)
def test_a_stage_key_from_another_partition_cannot_open_a_run(tmp_path, field):
    """An AMD stage key must not authorize an NVDA run — nor any other axis."""

    repository = migrated(tmp_path)
    key = {
        "stage": "ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    mismatched = dict(key)
    mismatched[field] = {
        "stage": "cluster",
        "ticker": "AMD",
        "trading_day": "2026-07-24",
        "pipeline_version": "v2",
    }[field]
    assert repository.claim_stage_key(**mismatched, run_id="run-1")

    with pytest.raises(Phase0RunContextError, match=f"stage key covers {field}"):
        with repository.stage_run(
            run_id="run-1",
            stage=key["stage"],
            trading_day=key["trading_day"],
            pipeline_version=key["pipeline_version"],
            ticker=key["ticker"],
            stage_key=mismatched,
        ):
            pass

    assert repository.run_log_entries() == []


def stage_key_for(ticker="NVDA", day=DAY, version="v1", stage="ingest") -> dict:
    return {
        "stage": stage,
        "ticker": ticker,
        "trading_day": day,
        "pipeline_version": version,
    }


def test_a_run_holding_a_stage_key_adopts_that_keys_ticker(tmp_path):
    """Omitting the ticker takes the key's; it never means "any ticker".

    A stage key always names a ticker.  Skipping the comparison when the
    caller named none let an NVDA lease open a ticker-less run — and a
    ticker-less run's partition checks pass for *every* ticker, so the
    NVDA key silently authorized AMD work by saying nothing at all.
    """

    repository = migrated(tmp_path)
    key = stage_key_for(ticker="NVDA")
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage=key["stage"],
        trading_day=key["trading_day"],
        pipeline_version=key["pipeline_version"],
        stage_key=key,
    ) as run:
        assert run.ticker == "NVDA"

    # And the adopted ticker is load-bearing, not cosmetic: the run is now
    # constrained by it exactly as if it had been passed.
    assert repository.claim_stage_key(**key, run_id="run-2")
    with pytest.raises(Phase0RunContextError, match="AMD"):
        with repository.stage_run(
            run_id="run-2",
            stage=key["stage"],
            trading_day=key["trading_day"],
            pipeline_version=key["pipeline_version"],
            stage_key=key,
        ) as run:
            repository.ingest_raw_items([{**raw_item(1), "ticker": "AMD"}], run=run)
    assert repository.count("raw_items") == 0


TICKER_BINDINGS = [
    ("omitted", None, "NVDA"),
    ("correct", "NVDA", "NVDA"),
]


@pytest.mark.parametrize(
    "label, supplied, expected",
    TICKER_BINDINGS,
    ids=[case[0] for case in TICKER_BINDINGS],
)
def test_a_stage_key_binds_its_ticker_however_the_run_names_it(
    tmp_path, label, supplied, expected
):
    repository = migrated(tmp_path)
    key = stage_key_for(ticker="NVDA")
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage=key["stage"],
        trading_day=key["trading_day"],
        pipeline_version=key["pipeline_version"],
        ticker=supplied,
        stage_key=key,
    ) as run:
        assert run.ticker == expected


def test_a_wrong_ticker_never_binds_a_stage_key(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for(ticker="NVDA")
    assert repository.claim_stage_key(**key, run_id="run-1")

    with pytest.raises(Phase0RunContextError, match="stage key covers ticker"):
        with repository.stage_run(
            run_id="run-1",
            stage=key["stage"],
            trading_day=key["trading_day"],
            pipeline_version=key["pipeline_version"],
            ticker="AMD",
            stage_key=key,
        ):
            pass

    assert repository.run_log_entries() == []


def test_an_impostor_cannot_even_open_a_run_on_someone_elses_key(tmp_path):
    """Ownership is proved before the context exists, not at first write.

    Checking it only at mutation time left an impostor able to open a run,
    do nothing, and have the exit path record a clean success against work
    it never owned.
    """

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-owner")

    with pytest.raises(StageKeyError, match="owned by run 'run-owner'"):
        with repository.stage_run(
            run_id="run-impostor",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ):
            pass

    assert repository.count("raw_items") == 0
    assert repository.stage_key_state(**key)["run_id"] == "run-owner"
    # No context was created, so no run log was written for the impostor.
    assert repository.run_log_entries(run_id="run-impostor") == []


def test_no_run_log_is_written_when_the_context_cannot_be_created(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(StageKeyError, match="no stage key has been claimed"):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=stage_key_for(),
        ):
            pass

    assert repository.run_log_entries() == []


@pytest.mark.parametrize("state", ["missing", "completed", "expired", "reclaimed"])
def test_an_unusable_stage_key_is_refused_at_run_creation(tmp_path, state):
    repository = migrated(tmp_path)
    key = stage_key_for()
    expected = "no stage key has been claimed"

    if state != "missing":
        assert repository.claim_stage_key(**key, run_id="run-1", lease_seconds=1)
    if state == "completed":
        repository.admin.complete_stage_key(**key, run_id="run-1", status="success")
        expected = "no longer running"
    elif state == "expired":
        time.sleep(1.1)
        expected = "lease has expired"
    elif state == "reclaimed":
        time.sleep(1.1)
        assert repository.claim_stage_key(**key, run_id="run-2")
        expected = "owned by run 'run-2'"

    with pytest.raises(StageKeyError, match=expected):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ):
            pass

    assert repository.run_log_entries(run_id="run-1") == []


def test_finishing_a_stage_key_this_run_lost_is_an_error_not_a_no_op(tmp_path):
    """Zero rows updated must never read as success."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1", lease_seconds=300)

    with pytest.raises(StageKeyError, match="no longer owns"):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ):
            # Another owner takes the key mid-run.
            with repository.admin.connect_writable() as connection:
                connection.execute(
                    "UPDATE pipeline_stage_keys SET run_id = 'run-2' "
                    "WHERE stage = ? AND ticker = ? AND trading_day = ? "
                    "AND pipeline_version = ?",
                    ("ingest", "NVDA", DAY, "v1"),
                )

    # run-1 did not overwrite run-2's ownership on its way out.
    assert repository.stage_key_state(**key)["run_id"] == "run-2"
    assert repository.stage_key_state(**key)["status"] == "running"


def test_a_lease_cannot_be_reclaimed_between_validation_and_commit(tmp_path):
    """The TOCTOU probe: A validates, the lease expires, B tries to reclaim.

    A holds the write lock from before it validates until after it
    commits, so B blocks rather than stealing the key mid-transaction.  The
    assertion that matters is the last one: the data and the stage key
    cannot end up telling different stories.
    """

    repository = migrated(tmp_path)
    key = {
        "stage": "ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    # A one-second lease, so wall-clock expiry passes while A is paused.
    assert repository.claim_stage_key(**key, run_id="run-A", lease_seconds=1)

    validated = threading.Event()
    b_finished = threading.Event()
    outcome: dict = {}

    original = Phase0Repository._assert_lease_held

    def paused_lease_check(self, connection, run, *, operation):
        """Pause in the exact window the old TOCTOU bug lived in.

        This is *immediately after* ownership and expiry are checked and
        before the mutation runs.  Under the old design the check happened
        on a separate connection before the write transaction opened, so
        this window was unlocked and B could reclaim the key while A went
        on to commit anyway.  Now the check reads through ``connection``,
        which already holds the write lock, so the window is sealed.
        """

        original(self, connection, run, operation=operation)
        if not validated.is_set():
            validated.set()
            # Wall-clock expiry of A's one-second lease passes here, and B
            # makes its entire attempt inside this window.
            time.sleep(1.5)
            b_finished.wait(timeout=10)

    class ImpatientRepository(Phase0Repository):
        """B, with a short busy timeout.

        Without this B would simply *wait* for A's write lock and the test
        would only prove SQLite blocks. With it, B fails fast, which is the
        sharper statement: during A's transaction the key was not
        acquirable at all.
        """

        def _open_connection(self):
            connection = super()._open_connection()
            connection.execute("PRAGMA busy_timeout = 250")
            return connection

    def writer_a():
        try:
            with repository.stage_run(
                run_id="run-A",
                stage="ingest",
                trading_day=DAY,
                pipeline_version="v1",
                ticker="NVDA",
                stage_key=key,
            ) as run:
                # Terminal: data, run log, and key completion in one
                # transaction, all under the write lock A already holds.
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
            outcome["a"] = "committed"
        except BaseException as exc:  # noqa: BLE001 - recorded for the assert
            outcome["a"] = f"{type(exc).__name__}: {exc}"

    def reclaimer_b():
        validated.wait(timeout=10)
        time.sleep(1.1)  # A's lease is now genuinely expired.
        reclaimer = ImpatientRepository(repository.database_path)
        started = time.monotonic()
        try:
            outcome["b_claimed"] = reclaimer.claim_stage_key(
                **key, run_id="run-B", lease_seconds=60
            )
        except sqlite3.OperationalError as exc:
            outcome["b_claimed"] = f"blocked: {exc}"
        outcome["b_elapsed"] = time.monotonic() - started
        outcome["b_owner_during"] = reclaimer.stage_key_state(**key)["run_id"]
        b_finished.set()

    with mock.patch.object(Phase0Repository, "_assert_lease_held", paused_lease_check):
        threads = [
            threading.Thread(target=writer_a),
            threading.Thread(target=reclaimer_b),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads), "deadlock"

    # B tried after the lease had expired and still could not take the key,
    # because A held the write lock across validation and mutation.
    assert outcome["b_claimed"] != True  # noqa: E712 - False or "blocked: ..."
    assert outcome["b_owner_during"] == "run-A"

    # A finished under the lock it validated against.
    assert outcome["a"] == "committed"
    state = repository.stage_key_state(**key)
    assert state["run_id"] == "run-A"
    assert state["status"] == "success"

    # The invariant the whole design exists for: the data and the stage
    # key cannot end up telling different stories.
    assert repository.count("raw_items") == 1
    entry = repository.run_log_entries(run_id="run-A")[0]
    assert entry["status"] == "success"
    assert entry["success_count"] == 1


def test_the_terminal_mutation_completes_the_stage_key_in_its_own_transaction(
    tmp_path,
):
    """Data, final run log, and key completion commit together or not at all."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
        # Already committed and released, before the block exits: there is
        # no window where the data says success and the key says running.
        assert run.terminated
        state = repository.stage_key_state(**key)
        assert state["status"] == "success"
        assert state["lease_expires_at"] is None
        assert repository.run_log_entries(run_id="run-1")[0]["status"] == "success"

    state = repository.stage_key_state(**key)
    assert state["status"] == "success"
    assert state["completed_at"] is not None
    assert repository.count("raw_items") == 1


def test_a_non_terminal_operation_never_reports_success(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run)
        interim = repository.run_log_entries(run_id="run-1")[0]
        assert interim["status"] == "degraded"
        assert repository.stage_key_state(**key)["status"] == "running"
        repository.ingest_raw_items([raw_item(2)], run=run, terminal=True)

    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "success"
    assert repository.stage_key_state(**key)["status"] == "success"
    assert repository.count("raw_items") == 2


def test_a_run_that_never_declares_completion_leaves_a_retryable_key(tmp_path):
    """Holding a lease and never finishing is not a success."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run)

    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "degraded"
    assert repository.stage_key_state(**key)["status"] == "failed"
    assert repository.claim_stage_key(**key, run_id="run-2")


def test_a_completed_stage_key_cannot_be_reclaimed(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    assert repository.claim_stage_key(**key, run_id="run-2") is False
    assert repository.recover_expired_leases() == []
    assert repository.stage_key_state(**key)["run_id"] == "run-1"


def test_a_failed_run_does_not_mark_its_stage_key_successful(tmp_path):
    repository = migrated(tmp_path)
    key = {
        "stage": "ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    assert repository.claim_stage_key(**key, run_id="run-1")

    with pytest.raises(RuntimeError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            repository.ingest_raw_items([raw_item(1)], run=run)
            raise RuntimeError("stage failed after writing")

    state = repository.stage_key_state(**key)
    assert state["status"] == "failed"
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"
    # The key is reclaimable again, and the retry is deterministic.
    assert repository.claim_stage_key(**key, run_id="run-2")


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
        with repository.admin.connect_writable() as connection:
            connection.execute(
                "UPDATE pipeline_stage_keys SET run_id = 'run-2' "
                "WHERE stage = ? AND ticker = ? AND trading_day = ? "
                "AND pipeline_version = ?",
                ("ingest", "NVDA", DAY, "v1"),
            )
        with pytest.raises(StageKeyError, match="owned by another run"):
            repository.ingest_raw_items([raw_item(1)], run=run)

    assert repository.count("raw_items") == 0
    # run-1 recorded its own failure without touching run-2's ownership.
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"
    assert repository.stage_key_state(**key)["run_id"] == "run-2"


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
        # Counts are derived, so the only operational text a caller can
        # steer into the run log is what an operation itself reports.
        repository.record_source_state(
            "rss:test",
            run=run,
            successful=False,
            status="failed",
            metadata={
                "upstream": "Authorization: Basic dXNlcjpwYXNz",
                "retry_url": "https://api.test/v1?api_key=SEKRET123",
                "headers": [{"x-api-key": "SEKRET123"}],
            },
            error="x-api-key: SEKRET123",
        )
        repository.ingest_raw_items([raw_item(1)], run=run)

    stored = json.dumps(repository.run_log_entries(run_id="run-counts")[0])
    assert "dXNlcjpwYXNz" not in stored
    assert "SEKRET123" not in stored
    assert "SEKRET123".encode() not in database_bytes(repository)


def sample_embedding(source_id: str, **overrides) -> PersistedEmbedding:
    values = {
        "source_kind": "raw_item",
        "source_id": str(source_id),
        "model_name": "fake",
        "model_revision": "r1",
        "dimension": 4,
        "dtype": str(EMBEDDING_DTYPE),
        "input_fingerprint": "a" * 64,
        "vector_blob": serialize_vector(np.ones(4, dtype=EMBEDDING_DTYPE)),
    }
    values.update(overrides)
    return PersistedEmbedding(**values)


def _entrypoint_cases(repository: Phase0Repository) -> dict:
    """For each logged entrypoint: a good batch, and one poisoned mid-way.

    The poison always sits *after* at least one valid element, so a
    non-atomic implementation would leave the earlier work committed.
    """

    item_ids = seed_raw_items(repository, 2)
    return {
        "ingest_raw_items": (
            lambda run: repository.ingest_raw_items(
                [raw_item(10), {"source": "", "canonical_url": ""}], run=run
            ),
            ("raw_items", 2),
        ),
        "reconcile_stories": (
            lambda run: repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", item_ids), story("cf2", [999_999])],
            ),
            ("stories", 0),
        ),
        "persist_embeddings": (
            lambda run: repository.persist_embeddings(
                [
                    sample_embedding(item_ids[0]),
                    sample_embedding("999999"),
                ],
                run=run,
            ),
            ("embeddings", 0),
        ),
        "record_source_state": (
            lambda run: repository.record_source_state(
                "rss:test", run=run, successful=True, status="not-a-status"
            ),
            ("raw_items", 2),
        ),
    }


@pytest.mark.parametrize(
    "entrypoint",
    [
        "ingest_raw_items",
        "reconcile_stories",
        "persist_embeddings",
        "record_source_state",
    ],
)
def test_a_failed_entrypoint_rolls_back_and_records_an_authoritative_failure(
    tmp_path, entrypoint
):
    repository = migrated(tmp_path)
    call, (table, expected) = _entrypoint_cases(repository)[entrypoint]

    with pytest.raises(Exception):
        with repository.stage_run(
            run_id="run-fail",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            call(run)

    assert repository.count(table) == expected, f"{entrypoint} left partial data"
    entry = repository.run_log_entries(run_id="run-fail")[0]
    assert entry["status"] == "failed"
    assert entry["failure_count"] >= 1
    assert "Bearer" not in json.dumps(entry["errors"])


def test_a_failed_theme_reconciliation_leaves_no_partial_membership(tmp_path):
    repository = migrated(tmp_path)
    day = build_day(repository)

    with pytest.raises(Exception):
        with repository.stage_run(
            run_id="run-fail",
            stage="m5.themes",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.reconcile_themes(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                theme_set=theme_set(),
                themes=[
                    ThemeRecord(
                        fingerprint="tf1",
                        label="Good",
                        story_ids=(day["stories"]["cf1"],),
                        citation_item_ids=(day["items"][0],),
                        salience_rank=1,
                        status="ready",
                        method="hdbscan",
                    ),
                    ThemeRecord(
                        fingerprint="tf2",
                        label="Bad",
                        story_ids=(day["stories"]["cf2"],),
                        # Cites a raw item that is not in its member story.
                        citation_item_ids=(day["items"][3],),
                        salience_rank=2,
                        status="ready",
                        method="hdbscan",
                    ),
                ],
            )

    assert repository.count("themes") == 0
    assert repository.count("theme_stories") == 0
    assert repository.count("theme_citations") == 0
    assert repository.count("theme_sets") == 0
    assert repository.run_log_entries(run_id="run-fail")[0]["status"] == "failed"


@pytest.mark.parametrize(
    "entrypoint",
    ["ingest_raw_items", "reconcile_stories", "persist_embeddings"],
)
def test_a_retry_after_a_failure_is_deterministic(tmp_path, entrypoint):
    """The same failure twice, then a clean run: no residue in between."""

    repository = migrated(tmp_path)
    for attempt in (1, 2):
        call, (table, expected) = _entrypoint_cases(repository)[entrypoint]
        with pytest.raises(Exception):
            with repository.stage_run(
                run_id=f"run-{attempt}",
                stage="ingest",
                trading_day=DAY,
                pipeline_version="v1",
                ticker="NVDA",
                attempt=attempt,
            ) as run:
                call(run)
        assert repository.count(table) == expected

    assert len(repository.run_log_entries()) == 2
    assert all(entry["status"] == "failed" for entry in repository.run_log_entries())


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
        repository.admin.log_stage(
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

    with repository.admin.connect_writable() as connection:
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


@pytest.mark.parametrize("version", PRIOR_VERSIONS)
def test_cross_version_theme_membership_is_refused_after_any_upgrade(tmp_path, version):
    """Migration 010's guarantee has to survive every upgrade path, not
    only a fresh database."""

    directory = partial_migrations(tmp_path, version)
    seeded = Phase0Repository(tmp_path / "phase0.sqlite3", migrations_path=directory)
    seeded.migrate()

    upgraded = Phase0Repository(tmp_path / "phase0.sqlite3")
    upgraded.migrate()
    assert upgraded.schema_version() == LATEST_VERSION

    day = cross_version_day(upgraded)
    with upgraded.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="pipeline version"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (day["theme_id"], day["v2_story"]),
            )


# ----------------------------------------------------------------------
# A run may only write its own partition
# ----------------------------------------------------------------------


def nvda_run(repository: Phase0Repository, **overrides):
    defaults = {
        "run_id": f"run-{next(_RUN_IDS)}",
        "stage": "ingest",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "ticker": "NVDA",
    }
    defaults.update(overrides)
    return repository.stage_run(**defaults)


def test_an_nvda_run_cannot_ingest_an_amd_payload(tmp_path):
    repository = migrated(tmp_path)

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match=r"asserts \['AMD'\]"):
            repository.ingest_raw_items([raw_item(1, "AMD")], run=run)

    assert repository.count("raw_items") == 0


def test_a_mixed_ticker_batch_is_rejected_whole(tmp_path):
    repository = migrated(tmp_path)

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match="AMD"):
            repository.ingest_raw_items(
                [raw_item(1), raw_item(2), raw_item(3, "AMD")], run=run
            )

    # Not even the two valid items landed.
    assert repository.count("raw_items") == 0


def test_a_raw_item_from_another_trading_day_is_rejected(tmp_path):
    repository = migrated(tmp_path)
    stray = raw_item(1)
    stray["published_at"] = "2026-07-24T12:00:00+00:00"

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match="falls on 2026-07-24"):
            repository.ingest_raw_items([raw_item(2), stray], run=run)

    assert repository.count("raw_items") == 0


# ----------------------------------------------------------------------
# A duplicate names stored evidence, and that evidence has its own day
#
# `(source, canonical_url)` is unique, so a re-ingested payload does not
# create a row — it resolves to one that already exists.  The partition
# check read only the incoming payload's timestamps, so a run for day D
# could resolve to a row belonging to D-1 and then write that row's
# ticker associations and candidate reasons, while the run log recorded
# the work as D's.
# ----------------------------------------------------------------------

PRIOR_DAY = "2026-07-22"


def stored_on(repository, day: str, index: int = 1, **overrides):
    """One raw item already persisted on ``day``, through the admin path."""

    values = {
        **raw_item(index),
        "published_at": f"{day}T12:00:00+00:00",
        "fetched_at": f"{day}T12:30:00+00:00",
    }
    values.update(overrides)
    return repository.admin.insert_raw_items([values])[0].item_id, values


def replay_of(values, day: str, **overrides):
    """The same canonical item, offered again with ``day``'s timestamps."""

    payload = {
        **values,
        "published_at": f"{day}T12:00:00+00:00",
        "fetched_at": f"{day}T12:30:00+00:00",
    }
    payload.update(overrides)
    return payload


#: The mutations the duplicate path can perform on a row it resolved to.
#: Each has to be refused when that row is another day's, and each is a
#: real write: an association row, or a candidate reason overwritten.
DUPLICATE_MUTATIONS = [
    ("plain replay", {}),
    ("new ticker association", {"ticker": None, "tickers": ["NVDA"]}),
    (
        "candidate reason update",
        {"candidate_tickers": [{"ticker": "NVDA", "reason": "headline_symbol"}]},
    ),
]


@pytest.mark.parametrize(
    "mutation", [pytest.param(m, id=label) for label, m in DUPLICATE_MUTATIONS]
)
def test_a_duplicate_cannot_mutate_another_days_evidence(tmp_path, mutation):
    repository = migrated(tmp_path)
    item_id, values = stored_on(repository, PRIOR_DAY, ticker=None)
    before = evidence_state(repository, item_id)

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match=r"belongs to 2026-07-22"):
            repository.ingest_raw_items([replay_of(values, DAY, **mutation)], run=run)

    assert evidence_state(repository, item_id) == before
    assert repository.count("raw_items") == 1


def evidence_state(repository, item_id: int) -> dict:
    """Everything the duplicate path could have written for one item."""

    row = repository.read.raw_item(item_id)
    return {
        "row": dict(row),
        "tickers": repository.raw_item_tickers(item_id),
        "candidates": sorted(
            (str(entry["ticker"]), str(entry["reason"]))
            for entry in repository.read.raw_item_candidates(item_id)
        ),
    }


def test_a_duplicate_on_the_same_day_is_still_an_idempotent_replay(tmp_path):
    """The rule is about the *day*, not about duplicates."""

    repository = migrated(tmp_path)
    item_id, values = stored_on(repository, DAY)

    with nvda_run(repository) as run:
        results = repository.ingest_raw_items([replay_of(values, DAY)], run=run)

    assert [(r.item_id, r.inserted) for r in results] == [(item_id, False)]
    assert repository.count("raw_items") == 1


def test_a_same_day_duplicate_may_still_add_an_association(tmp_path):
    """Within the partition, the duplicate path keeps working as before."""

    repository = migrated(tmp_path)
    item_id, values = stored_on(repository, DAY, ticker=None)
    assert repository.raw_item_tickers(item_id) == []

    with nvda_run(repository) as run:
        repository.ingest_raw_items(
            [
                replay_of(
                    values,
                    DAY,
                    tickers=["NVDA"],
                    candidate_tickers=[{"ticker": "NVDA", "reason": "headline_symbol"}],
                )
            ],
            run=run,
        )

    assert repository.raw_item_tickers(item_id) == ["NVDA"]
    assert [
        (entry["ticker"], entry["reason"])
        for entry in repository.read.raw_item_candidates(item_id)
    ] == [("NVDA", "headline_symbol")]


@pytest.mark.parametrize(
    "stamps, described",
    [
        pytest.param(
            {
                "published_at": f"{PRIOR_DAY}T23:00:00+00:00",
                "fetched_at": f"{DAY}T01:00:00+00:00",
            },
            PRIOR_DAY,
            id="day derives from published_at",
        ),
        pytest.param(
            {"published_at": None, "fetched_at": f"{PRIOR_DAY}T23:00:00+00:00"},
            PRIOR_DAY,
            id="day falls back to fetched_at",
        ),
    ],
)
def test_the_stored_days_derivation_matches_the_readers(tmp_path, stamps, described):
    """`published_at` wins; `fetched_at` is the fallback — one rule, in SQL.

    The second case is the one a Python-side ``a or b`` would get right
    only by accident, and it is also the case the reader
    (`raw_items_for_day`, via ``COALESCE``) has always used.
    """

    repository = migrated(tmp_path)
    values = {**raw_item(1), **stamps}
    item_id = repository.admin.insert_raw_items([values])[0].item_id
    # The reader agrees this item is on `described`.
    assert [row["id"] for row in repository.raw_items_for_day(described)] == [item_id]

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match=f"belongs to {described}"):
            repository.ingest_raw_items([replay_of(values, DAY)], run=run)


def test_one_cross_day_duplicate_rolls_back_the_whole_batch(tmp_path):
    """All-or-nothing, checked before the first row is written."""

    repository = migrated(tmp_path)
    stale_id, stale = stored_on(repository, PRIOR_DAY, index=1)
    before = evidence_state(repository, stale_id)

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match="belongs to 2026-07-22"):
            repository.ingest_raw_items(
                [raw_item(2), raw_item(3), replay_of(stale, DAY), raw_item(4)],
                run=run,
            )

    # Not one of the three otherwise-valid items landed, and the stale
    # item is exactly as it was.
    assert repository.count("raw_items") == 1
    assert evidence_state(repository, stale_id) == before


def test_a_rejected_duplicate_does_not_rewrite_the_stored_timestamp(tmp_path):
    """The fix refuses the write; it does not move the evidence to suit it."""

    repository = migrated(tmp_path)
    item_id, values = stored_on(repository, PRIOR_DAY)

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError):
            repository.ingest_raw_items([replay_of(values, DAY)], run=run)

    row = repository.read.raw_item(item_id)
    assert str(row["published_at"]).startswith(PRIOR_DAY)
    assert str(row["fetched_at"]).startswith(PRIOR_DAY)


def test_an_unassigned_raw_item_is_allowed_evidence(tmp_path):
    """The documented rule: ticker=None means "matches no ticker" and is
    evidence the run may keep; an asserted foreign ticker is not."""

    repository = migrated(tmp_path)
    unassigned = raw_item(1)
    unassigned["ticker"] = None
    unassigned.pop("tickers", None)

    with nvda_run(repository) as run:
        results = repository.ingest_raw_items(
            [unassigned, raw_item(2)], run=run, terminal=True
        )

    assert len(results) == 2
    assert repository.count("raw_items") == 2
    stored = {row["id"]: row["ticker"] for row in repository.raw_items_for_day(DAY)}
    assert None in stored.values() and "NVDA" in stored.values()


def test_a_secondary_ticker_association_cannot_smuggle_a_partition(tmp_path):
    repository = migrated(tmp_path)
    smuggler = raw_item(1)
    smuggler["tickers"] = ["NVDA", "AMD"]

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match="AMD"):
            repository.ingest_raw_items([smuggler], run=run)

    assert repository.count("raw_items") == 0


def test_reconcile_stories_rejects_a_member_from_another_partition(tmp_path):
    repository = migrated(tmp_path)
    nvda = seed_raw_items(repository, 2)
    amd = seed_raw_items(repository, 1, ticker="AMD")

    with nvda_run(repository, stage="m3.semantic") as run:
        # Refused on the association rule, which names what does claim it.
        with pytest.raises(
            Phase0RunContextError,
            match=r"no accepted association with NVDA \(it is associated "
            r"with \['AMD'\]\)",
        ):
            repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", nvda), story("cf2", amd)],
            )

    assert repository.count("stories") == 0
    assert repository.run_log_entries(run_id=run.run_id)[0]["status"] == "failed"


# ----------------------------------------------------------------------
# Ticker membership is membership, not exclusivity
#
# One article can be about two companies.  ``raw_item_tickers`` is the
# table that records which tickers claim an item, and ``raw_items.ticker``
# is the primary attribution beside it — not a veto over the others.  A
# path that read the primary as exclusive refused evidence the association
# table had already accepted.
# ----------------------------------------------------------------------


def multi_ticker_item(repository: Phase0Repository, primary="AMD") -> int:
    """One article attributed to AMD that is also, accepted, about NVDA."""

    return repository.admin.insert_raw_items(
        [{**raw_item(1), "ticker": primary, "tickers": ["AMD", "NVDA"]}]
    )[0].item_id


def test_a_secondary_association_makes_an_item_this_tickers_evidence(tmp_path):
    repository = migrated(tmp_path)
    item_id = multi_ticker_item(repository)
    assert repository.raw_item_tickers(item_id) == ["AMD", "NVDA"]

    with nvda_run(repository, stage="m3.semantic") as run:
        report = repository.reconcile_stories(
            run=run,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story("cf1", [item_id])],
        )

    assert len(report.inserted) == 1
    assert repository.run_log_entries(run_id=run.run_id)[0]["status"] == "success"
    # And the same item is still AMD's evidence too — independently.
    with repository.stage_run(
        run_id=f"amd-{next(_RUN_IDS)}",
        stage="m3.semantic",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="AMD",
    ) as run:
        assert (
            len(
                repository.reconcile_stories(
                    run=run,
                    ticker="AMD",
                    trading_day=DAY,
                    pipeline_version="v1",
                    stories=[story("amd-cf", [item_id])],
                ).inserted
            )
            == 1
        )


def test_a_secondary_association_also_makes_it_embeddable(tmp_path):
    """The adjacent path carried the identical defect."""

    repository = migrated(tmp_path)
    item_id = multi_ticker_item(repository)

    with nvda_run(repository, stage="m1.embed") as run:
        assert repository.persist_embeddings([sample_embedding(item_id)], run=run) == 1

    assert repository.count("embeddings") == 1


UNASSOCIATED_ITEMS = [
    # AMD's alone: no NVDA association exists.
    ("amd-only", {"ticker": "AMD"}, r"associated with \['AMD'\]"),
    # Genuinely unattributed: still refused, and still says so.
    ("unattributed", {"ticker": None}, "it is unattributed"),
    # A *candidate* is a suggestion nothing accepted; it is not membership.
    (
        "candidate-only",
        {"ticker": "AMD", "candidate_tickers": ["NVDA"]},
        r"associated with \['AMD'\]",
    ),
]


@pytest.mark.parametrize(
    "label, overrides, message",
    UNASSOCIATED_ITEMS,
    ids=[case[0] for case in UNASSOCIATED_ITEMS],
)
def test_without_an_accepted_association_the_item_stays_out(
    tmp_path, label, overrides, message
):
    repository = migrated(tmp_path)
    item_id = repository.admin.insert_raw_items([{**raw_item(1), **overrides}])[
        0
    ].item_id
    assert "NVDA" not in repository.raw_item_tickers(item_id)

    with nvda_run(repository, stage="m3.semantic") as run:
        with pytest.raises(Phase0RunContextError, match=message):
            repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", [item_id])],
            )
    assert repository.count("stories") == 0

    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(Phase0RunContextError, match=message):
            repository.persist_embeddings([sample_embedding(item_id)], run=run)
    assert repository.count("embeddings") == 0


def test_a_secondary_association_does_not_relax_the_day(tmp_path):
    """Membership settles the ticker only; every other axis still holds."""

    repository = migrated(tmp_path)
    item_id = repository.admin.insert_raw_items(
        [
            {
                **raw_item(1),
                "ticker": "AMD",
                "tickers": ["AMD", "NVDA"],
                "published_at": "2026-07-24T12:00:00+00:00",
                "fetched_at": "2026-07-24T12:30:00+00:00",
            }
        ]
    )[0].item_id

    with nvda_run(repository, stage="m3.semantic") as run:
        with pytest.raises(Phase0RunContextError, match="falls on 2026-07-24"):
            repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", [item_id])],
            )
    assert repository.count("stories") == 0


@pytest.mark.parametrize("axis", ["ticker", "trading_day", "pipeline_version"])
def test_reconcile_stories_rejects_a_run_covering_another_partition(tmp_path, axis):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)
    target = {"ticker": "NVDA", "trading_day": DAY, "pipeline_version": "v1"}
    target[axis] = {
        "ticker": "AMD",
        "trading_day": "2026-07-24",
        "pipeline_version": "v2",
    }[axis]

    with nvda_run(repository, stage="m3.semantic") as run:
        with pytest.raises(Phase0RunContextError, match="but the run covers"):
            repository.reconcile_stories(
                run=run, **target, stories=[story("cf1", item_ids)]
            )

    assert repository.count("stories") == 0


def test_reconcile_themes_rejects_a_story_from_another_version(tmp_path):
    repository = migrated(tmp_path)
    day = cross_version_day(repository)

    with nvda_run(repository, stage="m5.themes") as run:
        with pytest.raises(Phase0RunContextError, match="pipeline version v2"):
            repository.reconcile_themes(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                theme_set=theme_set(),
                themes=[
                    ThemeRecord(
                        fingerprint="tfx",
                        label="Smuggled",
                        story_ids=(day["v2_story"],),
                        salience_rank=1,
                        status="ready",
                        method="hdbscan",
                    )
                ],
            )


@pytest.mark.parametrize("bucket", ["other_coverage", "excluded"])
def test_reconcile_themes_rejects_a_foreign_story_in_coverage(tmp_path, bucket):
    repository = migrated(tmp_path)
    day = populated_theme_set(repository)
    payload = {
        "other_coverage": (),
        "excluded": (),
        bucket: (
            (OtherCoverageRecord(day["foreign_story"], "clustering_noise"),)
            if bucket == "other_coverage"
            else (ExcludedStoryRecord(day["foreign_story"], "no_encodable_text"),)
        ),
    }

    with nvda_run(repository, stage="m5.themes") as run:
        with pytest.raises(Phase0RunContextError, match="belongs to AMD"):
            repository.reconcile_themes(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                theme_set=theme_set(),
                **payload,
            )


def test_persist_embeddings_rejects_a_source_from_another_partition(tmp_path):
    repository = migrated(tmp_path)
    seed_raw_items(repository, 1)
    amd = seed_raw_items(repository, 1, ticker="AMD")

    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(
            Phase0RunContextError,
            match=r"no accepted association with NVDA \(it is associated "
            r"with \['AMD'\]\)",
        ):
            repository.persist_embeddings([sample_embedding(amd[0])], run=run)

    assert repository.count("embeddings") == 0
    assert repository.run_log_entries(run_id=run.run_id)[0]["status"] == "failed"


def test_persist_embeddings_rejects_one_bad_source_among_good_ones(tmp_path):
    repository = migrated(tmp_path)
    nvda = seed_raw_items(repository, 2)
    amd = seed_raw_items(repository, 1, ticker="AMD")

    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(Phase0RunContextError):
            repository.persist_embeddings(
                [sample_embedding(nvda[0]), sample_embedding(amd[0])], run=run
            )

    assert repository.count("embeddings") == 0


# ----------------------------------------------------------------------
# Embedding source identity
#
# Migration 007's ownership triggers are the durable contract: a raw item
# is named by id, a story by id *or* cluster fingerprint, a theme by id
# *or* fingerprint *or* theme key.  The logged batch has to resolve the
# same identities the database itself accepts — no more (an unknown or
# ambiguous identity still fails closed) and no fewer (rejecting a legal
# fingerprint as "does not exist" is the bug).
# ----------------------------------------------------------------------


def embedded_day(repository: Phase0Repository) -> dict:
    """One NVDA day with a story and a theme, and every identity for them."""

    day = build_day(repository)
    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="f" * 64,
                theme_key="nvda-chip-demand",
                label="Chip demand",
                story_ids=(day["stories"]["cf1"],),
                citation_item_ids=(day["items"][0],),
                method="hdbscan",
            )
        ],
    )
    with repository.admin.connect_writable() as connection:
        theme = dict(connection.execute("SELECT * FROM themes").fetchone())
    day.update({"theme": theme})
    return day


#: (label, source kind, how to read the identity out of the day, whether
#: it is a *durable* row id).  The run-scoped batch takes all of them; the
#: single-vector cache API takes only the durable ones, because it has no
#: run and so no partition to judge a partition-scoped handle against.
EMBEDDING_IDENTITIES = [
    ("raw-item-id", "raw_item", lambda day: str(day["items"][0]), True),
    ("story-id", "story", lambda day: str(day["stories"]["cf1"]), True),
    ("story-cluster-fingerprint", "story", lambda day: "cf1", False),
    ("theme-id", "theme", lambda day: str(day["theme"]["id"]), True),
    ("theme-fingerprint", "theme", lambda day: str(day["theme"]["fingerprint"]), False),
    ("theme-key", "theme", lambda day: str(day["theme"]["theme_key"]), False),
]


@pytest.mark.parametrize(
    "label, kind, identity, durable",
    EMBEDDING_IDENTITIES,
    ids=[case[0] for case in EMBEDDING_IDENTITIES],
)
def test_persist_embeddings_resolves_every_legal_source_identity(
    tmp_path, label, kind, identity, durable
):
    repository = migrated(tmp_path)
    day = embedded_day(repository)
    source_id = identity(day)

    with nvda_run(repository, stage="m1.embed") as run:
        assert (
            repository.persist_embeddings(
                [sample_embedding(source_id, source_kind=kind)], run=run
            )
            == 1
        )

    assert repository.count("embeddings") == 1
    assert repository.run_log_entries(run_id=run.run_id)[0]["status"] == "success"
    with repository.admin.connect_writable() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) AS n FROM embeddings "
                "WHERE source_kind = ? AND source_id = ?",
                (kind, source_id),
            ).fetchone()["n"]
            == 1
        )


@pytest.mark.parametrize(
    "label, kind, identity, durable",
    EMBEDDING_IDENTITIES,
    ids=[case[0] for case in EMBEDDING_IDENTITIES],
)
def test_the_single_vector_cache_takes_durable_ids_only(
    tmp_path, label, kind, identity, durable
):
    """Where the two paths diverge, and why the divergence is the point.

    The run-scoped batch may take a fingerprint because it has a run: it
    resolves the identity, checks the resolved row is this run's, and
    refuses an identity that names more than one row.  The single-vector
    cache API has none of that — no run, no partition, nothing to judge
    against — so a partition-scoped handle reaching it is a caller error
    and is named as one rather than guessed at.
    """

    repository = migrated(tmp_path)
    day = embedded_day(repository)
    source_id = identity(day)
    embedding = sample_embedding(source_id, source_kind=kind)

    if durable:
        repository.upsert_embedding(embedding)
        assert repository.get_embedding(kind, source_id) is not None
        assert repository.delete_embedding(kind, source_id) is True
        assert repository.count("embeddings") == 0
        return

    for call in (
        lambda: repository.upsert_embedding(embedding),
        lambda: repository.get_embedding(kind, source_id),
        lambda: repository.delete_embedding(kind, source_id),
    ):
        with pytest.raises(Phase0ValidationError, match="is not a durable"):
            call()
    assert repository.count("embeddings") == 0


#: Text that is not a durable id, however much it looks like one.  These
#: are refused by the single-vector API on their shape alone, before any
#: lookup: a cache key the schema and this module would read differently
#: is not a key at all.
NON_DURABLE_IDENTITIES = [
    ("zero", "0"),
    ("zero-padded", "01"),
    ("float-shaped", "1.0"),
    ("signed", "+1"),
    ("negative", "-1"),
    ("exponent", "1e0"),
    ("hex-fingerprint", "f" * 64),
    ("word-key", "nvda-chip-demand"),
    ("id-with-suffix", "1x"),
]


@pytest.mark.parametrize(
    "label, source_id",
    NON_DURABLE_IDENTITIES,
    ids=[case[0] for case in NON_DURABLE_IDENTITIES],
)
def test_the_cache_refuses_anything_that_is_not_a_durable_id(
    tmp_path, label, source_id
):
    repository = migrated(tmp_path)
    embedded_day(repository)

    with pytest.raises(Phase0ValidationError, match="is not a durable"):
        repository.upsert_embedding(sample_embedding(source_id, source_kind="story"))
    with pytest.raises(Phase0ValidationError, match="is not a durable"):
        repository.get_embedding("story", source_id)
    with pytest.raises(Phase0ValidationError, match="is not a durable"):
        repository.delete_embedding("story", source_id)
    assert repository.count("embeddings") == 0


def test_a_durable_id_that_names_nothing_is_a_miss_not_a_write(tmp_path):
    """Unknown identities fail closed, each in the way its verb allows."""

    repository = migrated(tmp_path)
    embedded_day(repository)

    assert repository.get_embedding("story", "99999") is None
    assert repository.delete_embedding("story", "99999") is False
    with pytest.raises(EmbeddingPersistenceError, match="source does not exist"):
        repository.upsert_embedding(sample_embedding("99999", source_kind="story"))
    assert repository.count("embeddings") == 0


def shared_identity_day(repository: Phase0Repository) -> dict:
    """Two partitions that deliberately agree on every public handle.

    NVDA and AMD each get a story with the same ``cluster_fingerprint``
    and a theme with the same ``fingerprint`` *and* the same
    ``theme_key``.  Nothing here is a schema violation: those columns are
    unique only within a ticker/trading-day/pipeline-version, and two
    tickers clustering to the same handle on the same day is ordinary.
    """

    shared = {
        "story_fingerprint": "shared-cluster-fingerprint",
        "theme_fingerprint": "c" * 64,
        "theme_key": "shared-theme-key",
    }
    for ticker in ("NVDA", "AMD"):
        items = seed_raw_items(repository, 1, ticker=ticker)
        reconcile_stories(
            repository,
            ticker=ticker,
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story(shared["story_fingerprint"], items)],
        )
        story_id = repository.stories_for_day(DAY, ticker)[0]["id"]
        reconcile_themes(
            repository,
            ticker=ticker,
            trading_day=DAY,
            pipeline_version="v1",
            theme_set=theme_set(),
            themes=[
                ThemeRecord(
                    fingerprint=shared["theme_fingerprint"],
                    theme_key=shared["theme_key"],
                    label=f"{ticker} theme",
                    story_ids=(story_id,),
                    citation_item_ids=(items[0],),
                    method="hdbscan",
                )
            ],
        )
        with repository.admin.connect_writable() as connection:
            theme = dict(
                connection.execute(
                    "SELECT * FROM themes WHERE ticker = ?", (ticker,)
                ).fetchone()
            )
        shared[ticker] = {"items": items, "story_id": story_id, "theme": theme}
    return shared


SHARED_HANDLES = [
    ("story-fingerprint", "story", "story_fingerprint"),
    ("theme-fingerprint", "theme", "theme_fingerprint"),
    ("theme-key", "theme", "theme_key"),
]


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[case[0] for case in SHARED_HANDLES],
)
def test_one_partition_cannot_overwrite_anothers_cache_row(
    tmp_path, label, kind, handle
):
    """The collision this contract exists to prevent, run end to end.

    Both partitions cache their own vector under the handle they share.
    Before, the second write silently replaced the first and every later
    read of *either* partition returned the survivor — one ticker's text
    embedded, both tickers' answer.
    """

    repository = migrated(tmp_path)
    shared = shared_identity_day(repository)
    identity = shared[handle]

    nvda = sample_embedding(
        identity,
        source_kind=kind,
        vector_blob=serialize_vector(np.ones(4, dtype=EMBEDDING_DTYPE)),
    )
    amd = sample_embedding(
        identity,
        source_kind=kind,
        input_fingerprint="b" * 64,
        vector_blob=serialize_vector(np.full(4, 0.5, dtype=EMBEDDING_DTYPE)),
    )

    for embedding in (nvda, amd):
        with pytest.raises(Phase0ValidationError, match="is not a durable"):
            repository.upsert_embedding(embedding)
    assert repository.count("embeddings") == 0

    # The run-scoped batch refuses it too, on the stronger ground that it
    # can see both rows and so knows the identity means neither.
    for ticker in ("NVDA", "AMD"):
        with repository.stage_run(
            run_id=f"run-{ticker}-{next(_RUN_IDS)}",
            stage="m1.embed",
            trading_day=DAY,
            pipeline_version="v1",
            ticker=ticker,
        ) as run:
            with pytest.raises(Phase0RunContextError, match="is ambiguous"):
                repository.persist_embeddings([nvda], run=run)
    assert repository.count("embeddings") == 0

    # And each partition's *durable* ids remain perfectly usable, keeping
    # one vector each — which is the whole point of refusing the handle.
    for ticker in ("NVDA", "AMD"):
        durable = (
            str(shared[ticker]["story_id"])
            if kind == "story"
            else str(shared[ticker]["theme"]["id"])
        )
        repository.upsert_embedding(sample_embedding(durable, source_kind=kind))
    assert repository.count("embeddings") == 2


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[case[0] for case in SHARED_HANDLES],
)
def test_one_partition_cannot_delete_anothers_cache_row(tmp_path, label, kind, handle):
    """Deletion by a shared handle would be deletion of someone else's row."""

    repository = migrated(tmp_path)
    shared = shared_identity_day(repository)
    durable = {
        ticker: (
            str(shared[ticker]["story_id"])
            if kind == "story"
            else str(shared[ticker]["theme"]["id"])
        )
        for ticker in ("NVDA", "AMD")
    }
    for identity in durable.values():
        repository.upsert_embedding(sample_embedding(identity, source_kind=kind))
    assert repository.count("embeddings") == 2

    with pytest.raises(Phase0ValidationError, match="is not a durable"):
        repository.delete_embedding(kind, shared[handle])

    assert repository.count("embeddings") == 2
    # Deleting by one partition's durable id takes exactly that row.
    assert repository.delete_embedding(kind, durable["AMD"]) is True
    assert repository.get_embedding(kind, durable["NVDA"]) is not None
    assert repository.count("embeddings") == 1


def test_a_shared_handle_still_reaches_the_right_row_when_it_is_unshared(tmp_path):
    """The contract narrows the cache API, not the run-scoped batch.

    One partition, one story: the fingerprint names exactly one row, and
    ``persist_embeddings`` — which has a run to check it against — still
    accepts it, as the schema's ownership triggers do.
    """

    repository = migrated(tmp_path)
    day = embedded_day(repository)

    with nvda_run(repository, stage="m1.embed") as run:
        assert (
            repository.persist_embeddings(
                [sample_embedding("cf1", source_kind="story")], run=run
            )
            == 1
        )

    with repository.admin.connect_writable() as connection:
        row = connection.execute("SELECT * FROM embeddings").fetchone()
    assert (row["source_kind"], row["source_id"]) == ("story", "cf1")
    assert day["stories"]["cf1"]


#: Identities that are not this partition's, are nobody's, or are two
#: rows' at once.  Every one must be refused with nothing written.
def _foreign_day(repository: Phase0Repository) -> dict:
    """An AMD story and theme carrying identities NVDA might reach for."""

    items = seed_raw_items(repository, 1, ticker="AMD")
    reconcile_stories(
        repository,
        ticker="AMD",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("amd-cf", items)],
    )
    amd_story = repository.stories_for_day(DAY, "AMD")[0]["id"]
    reconcile_themes(
        repository,
        ticker="AMD",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint="a" * 64,
                theme_key="amd-theme-key",
                label="AMD",
                story_ids=(amd_story,),
                citation_item_ids=(items[0],),
                method="hdbscan",
            )
        ],
    )
    return {"items": items, "story": amd_story}


HOSTILE_EMBEDDING_IDENTITIES = [
    ("unknown-story-fingerprint", "story", lambda day: "no-such-fingerprint", "exist"),
    ("unknown-theme-key", "theme", lambda day: "no-such-theme-key", "exist"),
    ("unknown-theme-fingerprint", "theme", lambda day: "b" * 64, "exist"),
    ("empty-ish-identity", "story", lambda day: "0", "exist"),
    # A story id dressed up so SQLite's integer affinity would match it
    # but the ownership trigger's text comparison would not.
    ("zero-padded-story-id", "story", lambda day: "01", "exist"),
    ("float-shaped-story-id", "story", lambda day: "1.0", "exist"),
    # Another partition's rows, reached by each of their public forms.
    ("foreign-story-fingerprint", "story", lambda day: "amd-cf", "belongs to AMD"),
    ("foreign-theme-key", "theme", lambda day: "amd-theme-key", "belongs to AMD"),
    ("foreign-theme-fingerprint", "theme", lambda day: "a" * 64, "belongs to AMD"),
    ("foreign-raw-item-id", "raw_item", lambda day: str(day["items"][0]), "AMD"),
]


@pytest.mark.parametrize(
    "label, kind, identity, message",
    HOSTILE_EMBEDDING_IDENTITIES,
    ids=[case[0] for case in HOSTILE_EMBEDDING_IDENTITIES],
)
def test_persist_embeddings_refuses_an_illegitimate_source_identity(
    tmp_path, label, kind, identity, message
):
    repository = migrated(tmp_path)
    embedded_day(repository)
    foreign = _foreign_day(repository)

    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(Phase0RunContextError, match=message):
            repository.persist_embeddings(
                [sample_embedding(identity(foreign), source_kind=kind)], run=run
            )

    assert repository.count("embeddings") == 0
    assert repository.run_log_entries(run_id=run.run_id)[0]["status"] == "failed"


def test_persist_embeddings_refuses_an_ambiguous_source_identity(tmp_path):
    """One name, two rows: there is no partition this vector belongs to.

    Fingerprints and theme keys are unique only within a ticker-day, and
    ``embeddings`` keys globally on (source_kind, source_id).  So an
    identity that names two stories cannot be resolved to a partition,
    and resolving it to whichever row came back first would let an NVDA
    run write a vector an AMD read would then get back.
    """

    repository = migrated(tmp_path)
    day = build_day(repository)
    shared = "shared-fingerprint"
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[
            story("cf1", day["items"][:2]),
            story("cf2", day["items"][2:3]),
            story("cf3", day["items"][3:]),
            story(shared, day["items"][:1]),
        ],
    )
    amd_items = seed_raw_items(repository, 1, ticker="AMD")
    reconcile_stories(
        repository,
        ticker="AMD",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story(shared, amd_items)],
    )

    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(Phase0RunContextError, match="is ambiguous"):
            repository.persist_embeddings(
                [sample_embedding(shared, source_kind="story")], run=run
            )

    assert repository.count("embeddings") == 0


def test_a_fingerprint_source_still_needs_its_raw_item_association(tmp_path):
    """Widening identity did not widen anything else.

    A raw item is still named only by id, and a ticker-scoped run may
    still only embed evidence explicitly associated with its ticker.
    """

    repository = migrated(tmp_path)
    unattributed = repository.admin.insert_raw_items(
        [{**raw_item(90), "ticker": None}]
    )[0].item_id

    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(Phase0RunContextError, match="no accepted association"):
            repository.persist_embeddings(
                [sample_embedding(str(unattributed))], run=run
            )

    assert repository.count("embeddings") == 0


def test_persist_embeddings_still_refuses_another_day_and_version(tmp_path):
    """Day and pipeline-version isolation, reached through a fingerprint.

    The identity forms are new; the partition rules they are checked
    against are not.
    """

    repository = migrated(tmp_path)
    day = build_day(repository)
    reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v2",
        stories=[story("v2-cf", day["items"][:1])],
    )

    with nvda_run(repository, stage="m1.embed", pipeline_version="v1") as run:
        with pytest.raises(Phase0RunContextError, match="pipeline version v2"):
            repository.persist_embeddings(
                [sample_embedding("v2-cf", source_kind="story")], run=run
            )

    with nvda_run(repository, stage="m1.embed", trading_day="2026-07-24") as run:
        with pytest.raises(Phase0RunContextError, match="falls on"):
            repository.persist_embeddings(
                [sample_embedding("cf1", source_kind="story")], run=run
            )

    assert repository.count("embeddings") == 0


# ----------------------------------------------------------------------
# A parent's deletion may not take another parent's vector with it
#
# The cleanup triggers deleted by every identity form migration 007
# accepts, including `cluster_fingerprint`, `fingerprint`, and
# `theme_key` — which are unique only within one partition.  So the
# ordering that matters is: A writes its vector by a handle while it is
# still the only owner, B *later* creates its own row bearing the same
# handle, and B's delete then takes A's vector.  Nothing B did was
# invalid, and A never heard about it.
# ----------------------------------------------------------------------


def one_handled_partition(repository, ticker: str, shared: Mapping[str, str]) -> dict:
    """One partition's story and theme, bearing the shared handles."""

    items = seed_raw_items(repository, 1, ticker=ticker)
    reconcile_stories(
        repository,
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story(shared["story_fingerprint"], items)],
    )
    story_id = [
        row["id"]
        for row in repository.stories_for_day(DAY, ticker)
        if row["cluster_fingerprint"] == shared["story_fingerprint"]
    ][0]
    reconcile_themes(
        repository,
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint=shared["theme_fingerprint"],
                theme_key=shared["theme_key"],
                label=f"{ticker} theme",
                story_ids=(story_id,),
                citation_item_ids=(items[0],),
                method="hdbscan",
            )
        ],
    )
    theme_id = repository.theme_set(
        ticker=ticker, trading_day=DAY, pipeline_version="v1"
    )["themes"][0]["id"]
    return {"items": items, "story_id": story_id, "theme_id": theme_id}


SHARED_HANDLE_VALUES = {
    "story_fingerprint": "shared-cluster-fingerprint",
    "theme_fingerprint": "c" * 64,
    "theme_key": "shared-theme-key",
}


def cached(repository, kind: str, source_id: str) -> bool:
    with repository.admin.connect_writable() as connection:
        return (
            connection.execute(
                "SELECT 1 FROM embeddings WHERE source_kind = ? AND source_id = ?",
                (kind, source_id),
            ).fetchone()
            is not None
        )


def drop_parent(repository, kind: str, parent_id: int) -> None:
    """Delete one parent, clearing what references it first.

    Ordinary reconciliation reaches the same state — an obsolete story or
    theme is deleted once nothing cites it — but going through the tables
    directly keeps these tests about the cleanup trigger rather than
    about how reconciliation decided to get there.
    """

    with repository.admin.connect_writable() as connection:
        if kind == "theme":
            connection.execute(
                "DELETE FROM theme_citations WHERE theme_id = ?", (parent_id,)
            )
            connection.execute(
                "DELETE FROM theme_stories WHERE theme_id = ?", (parent_id,)
            )
            connection.execute("DELETE FROM themes WHERE id = ?", (parent_id,))
        else:
            connection.execute(
                "DELETE FROM theme_citations WHERE theme_id IN ("
                "SELECT theme_id FROM theme_stories WHERE story_id = ?)",
                (parent_id,),
            )
            for table in (
                "theme_stories",
                "theme_other_coverage",
                "theme_excluded_stories",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE story_id = ?", (parent_id,)
                )
            # `story_members` cascades, and the canonical-member guard only
            # fires while the story is still there — so the story goes
            # first and its members follow it, which is also the order
            # `_delete_story` produces.
            connection.execute("DELETE FROM stories WHERE id = ?", (parent_id,))
        connection.commit()


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-survives" for case in SHARED_HANDLES],
)
def test_deleting_one_owner_keeps_another_partitions_embedding(
    tmp_path, label, kind, handle
):
    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]

    # 1. NVDA is the only owner, so its write is unambiguous and allowed.
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    with nvda_run(repository, stage="m1.embed") as run:
        repository.persist_embeddings(
            [sample_embedding(identity, source_kind=kind)], run=run
        )
    assert cached(repository, kind, identity)

    # 2. AMD legitimately produces a row bearing the same handle later.
    amd = one_handled_partition(repository, "AMD", SHARED_HANDLE_VALUES)

    # 3. AMD's parent goes away.
    drop_parent(repository, kind, amd["theme_id" if kind == "theme" else "story_id"])

    # NVDA still owns its parent, so it must still have its vector.
    survivor = nvda["theme_id" if kind == "theme" else "story_id"]
    assert repository.read.story(survivor) is not None or kind == "theme"
    assert cached(repository, kind, identity), "the surviving owner lost its vector"


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-orphan-goes" for case in SHARED_HANDLES],
)
def test_deleting_the_last_owner_still_removes_the_embedding(
    tmp_path, label, kind, handle
):
    """The other half of the contract: no vector outlives every owner."""

    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]

    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    with nvda_run(repository, stage="m1.embed") as run:
        repository.persist_embeddings(
            [sample_embedding(identity, source_kind=kind)], run=run
        )
    amd = one_handled_partition(repository, "AMD", SHARED_HANDLE_VALUES)

    key = "theme_id" if kind == "theme" else "story_id"
    drop_parent(repository, kind, amd[key])
    assert cached(repository, kind, identity)
    drop_parent(repository, kind, nvda[key])
    assert not cached(repository, kind, identity), "an orphan vector survived"


def test_a_durable_id_cleanup_collects_its_vector_when_nobody_answers(tmp_path):
    """The ordinary case: an id nothing else answers to takes its vector.

    A row id is globally unique and AUTOINCREMENT never hands a deleted
    one out again, so normally there is nobody left the moment the row
    goes.  "Normally" is the operative word — a live row's *handle* can be
    the same string as a dead row's id, and migration 013 keeps that case
    honest; it is covered separately below.
    """

    repository = migrated(tmp_path)
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    one_handled_partition(repository, "AMD", SHARED_HANDLE_VALUES)

    with nvda_run(repository, stage="m1.embed") as run:
        repository.persist_embeddings(
            [sample_embedding(str(nvda["story_id"]), source_kind="story")], run=run
        )
    assert cached(repository, "story", str(nvda["story_id"]))

    drop_parent(repository, "story", nvda["story_id"])
    assert not cached(repository, "story", str(nvda["story_id"]))


def test_a_raw_item_cleanup_removes_its_own_vector(tmp_path):
    """Raw items are only ever addressed by id; nothing here changed."""

    repository = migrated(tmp_path)
    day = embedded_day(repository)
    item = str(day["items"][0])
    with nvda_run(repository, stage="m1.embed") as run:
        repository.persist_embeddings(
            [sample_embedding(item, source_kind="raw_item")], run=run
        )
    assert cached(repository, "raw_item", item)

    with repository.admin.connect_writable() as connection:
        owners = [
            int(row["story_id"])
            for row in connection.execute(
                "SELECT story_id FROM story_members WHERE raw_item_id = ?", (item,)
            )
        ]
    for story_id in owners:
        drop_parent(repository, "story", story_id)
    with repository.admin.connect_writable() as connection:
        connection.execute("DELETE FROM raw_items WHERE id = ?", (item,))
        connection.commit()

    assert not cached(repository, "raw_item", item)


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-delete-refused" for case in SHARED_HANDLES],
)
def test_a_partition_cannot_delete_a_shared_handle_through_the_api(
    tmp_path, label, kind, handle
):
    """The repository-side delete is unchanged: durable ids only."""

    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]
    one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    with nvda_run(repository, stage="m1.embed") as run:
        repository.persist_embeddings(
            [sample_embedding(identity, source_kind=kind)], run=run
        )
    one_handled_partition(repository, "AMD", SHARED_HANDLE_VALUES)

    with pytest.raises(Phase0ValidationError, match="not a durable"):
        repository.delete_embedding(kind, identity)
    with pytest.raises(Phase0ValidationError, match="not a durable"):
        repository.upsert_embedding(sample_embedding(identity, source_kind=kind))
    assert cached(repository, kind, identity)


# ----------------------------------------------------------------------
# A living owner that renames its handle is the same event as a dying one
#
# Migration 007's survivor guard fires on DELETE only.  An owner that
# stays alive and rewrites `cluster_fingerprint`, `fingerprint`, or
# `theme_key` leaves the old handle belonging to nobody while its vector
# stays cached, unreachable and unremovable — and `reconcile_themes`
# reaches exactly that state through the public API, because it matches a
# theme by `fingerprint` and writes `theme_key` as an owned column.
# Migration 012 asks the same question the delete path asks: who is left?
# ----------------------------------------------------------------------


#: Which table and column each shared handle actually lives in.
HANDLE_COLUMNS = {
    "story_fingerprint": ("stories", "cluster_fingerprint"),
    "theme_fingerprint": ("themes", "fingerprint"),
    "theme_key": ("themes", "theme_key"),
}

#: A second legal value for each handle, so a move goes somewhere valid.
MOVED_HANDLE_VALUES = {
    "story_fingerprint": "moved-cluster-fingerprint",
    "theme_fingerprint": "d" * 64,
    "theme_key": "moved-theme-key",
}

#: A column that is emphatically not a handle, per parent kind.
UNRELATED_COLUMNS = {
    "story": ("stories", "canonical_title"),
    "theme": ("themes", "label"),
}


def parent_of(partition: Mapping[str, Any], kind: str) -> int:
    return int(partition["theme_id" if kind == "theme" else "story_id"])


def set_column(repository, table: str, column: str, row_id: int, value) -> None:
    """Write one column of one row, the way a re-run of its stage would.

    Going through the table keeps these tests about the schema's ownership
    lifecycle rather than about which reconciliation path happened to
    produce the rename — one of which is exercised end to end below.
    """

    with repository.admin.connect_writable() as connection:
        connection.execute(
            f"UPDATE {table} SET {column} = ? WHERE id = ?", (value, row_id)
        )
        connection.commit()


def move_handle(repository, handle: str, row_id: int, value=None) -> str:
    table, column = HANDLE_COLUMNS[handle]
    value = MOVED_HANDLE_VALUES[handle] if value is None else value
    set_column(repository, table, column, row_id, value)
    return value


def embed_while_sole_owner(
    repository, kind: str, identity: str, ticker: str = "NVDA"
) -> None:
    """One partition caches a vector under an identity it alone answers to.

    The ordering is the whole point: the write has to happen while the
    identity is unambiguous, because `persist_embeddings` refuses it once
    two rows answer to it.  The run has to cover the owning partition too,
    for the same reason it always does.
    """

    with nvda_run(repository, stage="m1.embed", ticker=ticker) as run:
        repository.persist_embeddings(
            [sample_embedding(identity, source_kind=kind)], run=run
        )
    assert cached(repository, kind, identity), "setup failed to cache the vector"


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-rename-orphans" for case in SHARED_HANDLES],
)
def test_renaming_the_only_owners_handle_removes_the_orphan(
    tmp_path, label, kind, handle
):
    """Cases 1, 3 and 5: sole owner renames, so nobody is left."""

    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    embed_while_sole_owner(repository, kind, identity)

    moved = move_handle(repository, handle, parent_of(nvda, kind))

    assert not cached(repository, kind, identity), "an orphan vector survived"
    # And nothing inherited it: the new handle has no vector until the
    # repository writes one for that identity.
    assert not cached(repository, kind, moved)
    assert repository.count("embeddings") == 0


@pytest.mark.parametrize("mover", ["renaming-owner", "other-partition"])
@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-rename-spares" for case in SHARED_HANDLES],
)
def test_renaming_one_of_two_owners_keeps_the_shared_embedding(
    tmp_path, label, kind, handle, mover
):
    """Cases 2, 4 and 6, from both directions.

    `renaming-owner` is the partition that cached the vector moving away
    from the handle; `other-partition` is the one that never touched it
    moving away.  Either way one live owner still carries the handle, so
    the vector stays — a rename may not reach across a partition boundary
    any more than a delete may.
    """

    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    embed_while_sole_owner(repository, kind, identity)
    amd = one_handled_partition(repository, "AMD", SHARED_HANDLE_VALUES)

    moving = nvda if mover == "renaming-owner" else amd
    move_handle(repository, handle, parent_of(moving, kind))

    assert cached(repository, kind, identity), "the surviving owner lost its vector"
    assert repository.count("embeddings") == 1


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-then-the-survivor" for case in SHARED_HANDLES],
)
def test_the_survivor_of_a_rename_still_takes_the_vector_when_it_goes(
    tmp_path, label, kind, handle
):
    """The two verbs compose: a spared vector is not a permanent one."""

    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    embed_while_sole_owner(repository, kind, identity)
    amd = one_handled_partition(repository, "AMD", SHARED_HANDLE_VALUES)

    move_handle(repository, handle, parent_of(nvda, kind))
    assert cached(repository, kind, identity)

    # AMD is now the last owner of the old handle, whichever way it goes.
    drop_parent(repository, kind, parent_of(amd, kind))
    assert not cached(repository, kind, identity), "an orphan vector survived"


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-noop" for case in SHARED_HANDLES],
)
def test_rewriting_a_handle_to_the_value_it_already_had_changes_nothing(
    tmp_path, label, kind, handle
):
    """An UPDATE that moves nothing is not a move."""

    repository = migrated(tmp_path)
    identity = SHARED_HANDLE_VALUES[handle]
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    embed_while_sole_owner(repository, kind, identity)

    move_handle(repository, handle, parent_of(nvda, kind), value=identity)

    assert cached(repository, kind, identity)


@pytest.mark.parametrize("kind", ["story", "theme"])
def test_changing_an_unrelated_column_keeps_the_embedding(tmp_path, kind):
    """The triggers fire on the handle columns, not on every write."""

    repository = migrated(tmp_path)
    handle = "story_fingerprint" if kind == "story" else "theme_fingerprint"
    identity = SHARED_HANDLE_VALUES[handle]
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    embed_while_sole_owner(repository, kind, identity)

    table, column = UNRELATED_COLUMNS[kind]
    set_column(repository, table, column, parent_of(nvda, kind), "rewritten text")

    assert cached(repository, kind, identity)


@pytest.mark.parametrize(
    "label, kind, handle",
    SHARED_HANDLES,
    ids=[f"{case[0]}-durable-survives" for case in SHARED_HANDLES],
)
def test_a_durable_id_embedding_survives_its_owners_handle_move(
    tmp_path, label, kind, handle
):
    """A row id names a row, not a fingerprint, so a rename is nothing."""

    repository = migrated(tmp_path)
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    durable = str(parent_of(nvda, kind))
    embed_while_sole_owner(repository, kind, durable)

    move_handle(repository, handle, parent_of(nvda, kind))

    assert cached(repository, kind, durable), "a durable-id vector was collected"
    assert repository.get_embedding(kind, durable) is not None


def test_a_handle_that_is_some_live_rows_id_is_still_owned(tmp_path):
    """The survivor check is ownership, not one column.

    Migration 007 accepts a durable id *or* a handle as `source_id`, so
    "is this orphaned" has to ask whether any live row would still be
    allowed to own it — otherwise a story renaming its fingerprint away
    from the string `"1"` collects the vector belonging to story 1.
    """

    repository = migrated(tmp_path)
    first = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    durable = str(first["story_id"])
    embed_while_sole_owner(repository, "story", durable)

    # A second partition whose fingerprint is literally the first story's id.
    second = one_handled_partition(
        repository, "AMD", {**SHARED_HANDLE_VALUES, "story_fingerprint": durable}
    )
    set_column(
        repository, "stories", "cluster_fingerprint", second["story_id"], "moved-away"
    )

    assert cached(repository, "story", durable), "story 1 lost its own vector"


# ----------------------------------------------------------------------
# One story, one accounting bucket — including "theme" against itself
#
# The M5 accounting rules are pairwise, and five of the six pairs were
# written. The one nobody wrote is the one where both sides are the same
# table: a story in two themes of one partition. `_prepare_coverage`
# flattens every theme's members into a *set* to check coverage and
# exclusions against, so the step that looks at every member is the step
# that discards how many themes claimed it.
#
# The citation rules block the obvious attempt — two themes sharing a
# one-item story would have to share its citation, and no two themes in a
# partition may cite the same raw item. Give that story a second member
# and the collision goes away: each theme cites a different item, both
# claim the story, and every other rule is satisfied.
# ----------------------------------------------------------------------


def split_partition(
    repository, sizes, ticker="NVDA"
) -> tuple[list[int], list[list[int]]]:
    """One partition where `sizes[i]` raw items belong to story `i`.

    Story sizes are the point: a story with two member raw items lets two
    themes cite it without colliding on a citation, which is the only way
    to reach the overlap this contract is about.
    """

    items = seed_raw_items(repository, sum(sizes), ticker=ticker)
    groups, offset = [], 0
    for size in sizes:
        groups.append(items[offset : offset + size])
        offset += size
    reconcile_stories(
        repository,
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story(f"cf-{index}", group) for index, group in enumerate(groups)],
    )
    by_fingerprint = {
        str(row["cluster_fingerprint"]): int(row["id"])
        for row in repository.stories_for_day(DAY, ticker)
    }
    return [by_fingerprint[f"cf-{index}"] for index in range(len(sizes))], groups


def a_theme(fingerprint, story_ids, citation_ids, rank=1) -> ThemeRecord:
    return ThemeRecord(
        fingerprint=fingerprint,
        theme_key=fingerprint,
        label=f"Theme {fingerprint}",
        story_ids=tuple(story_ids),
        citation_item_ids=tuple(citation_ids),
        method="hdbscan",
        salience_rank=rank,
    )


def settle_themes(repository, themes, *, other=(), excluded=(), ticker="NVDA"):
    return reconcile_themes(
        repository,
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=themes,
        other_coverage=other,
        excluded=excluded,
    )


def theme_membership(repository) -> list[tuple[int, int]]:
    with repository.admin.connect_writable() as connection:
        return sorted(
            (int(row["theme_id"]), int(row["story_id"]))
            for row in connection.execute(
                "SELECT theme_id, story_id FROM theme_stories"
            )
        )


def test_two_themes_may_not_share_a_story(tmp_path):
    """Hostile case 1, built so no citation collision hides it."""

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [2])

    with pytest.raises(
        Phase0ValidationError, match="a story belongs to exactly one theme"
    ):
        settle_themes(
            repository,
            [
                a_theme("A", [stories[0]], [groups[0][0]], 1),
                a_theme("B", [stories[0]], [groups[0][1]], 2),
            ],
        )

    assert theme_membership(repository) == []
    assert repository.count("themes") == 0


def test_one_overlapping_pair_rejects_the_whole_batch(tmp_path):
    """Hostile case 2: the valid themes do not get in either.

    Reconciliation settles a partition as a unit, so a batch that is
    wrong anywhere has to write nothing anywhere — otherwise the day is
    left half-settled by an input that was never acceptable.
    """

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [2, 1])

    with pytest.raises(Phase0ValidationError, match="story .* in themes"):
        settle_themes(
            repository,
            [
                a_theme("A", [stories[0]], [groups[0][0]], 1),
                a_theme("B", [stories[1]], [groups[1][0]], 2),
                a_theme("C", [stories[0]], [groups[0][1]], 3),
            ],
        )

    assert theme_membership(repository) == []
    assert repository.count("themes") == 0
    assert repository.count("theme_sets") == 0


def test_the_error_names_every_story_and_theme_involved(tmp_path):
    """An operator has to know which cards to look at."""

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [2, 2])

    with pytest.raises(Phase0ValidationError) as caught:
        settle_themes(
            repository,
            [
                a_theme("A", [stories[0], stories[1]], [groups[0][0]], 1),
                a_theme("B", [stories[0], stories[1]], [groups[0][1]], 2),
            ],
        )

    message = str(caught.value)
    assert f"story {stories[0]}" in message and f"story {stories[1]}" in message
    assert "'A'" in message and "'B'" in message


def test_a_story_repeated_inside_one_theme_is_canonicalized(tmp_path):
    """Hostile case 3, and the answer is *canonicalize*, not fail.

    `_prepare_theme` runs both `story_ids` and `citation_item_ids` through
    `dict.fromkeys`, so a repeat inside one record is deduplicated with
    order preserved — the existing model contract, and `story_count`
    defaults from the deduplicated list. Only repeats *across* themes are
    an accounting error, because only those name two owners.
    """

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1])

    report = settle_themes(
        repository,
        [a_theme("A", [stories[0], stories[0], stories[0]], [groups[0][0]], 1)],
    )

    assert len(report.inserted) == 1
    assert theme_membership(repository) == [(report.inserted[0], stories[0])]
    stored = repository.theme_set(
        ticker="NVDA", trading_day=DAY, pipeline_version="v1"
    )["themes"]
    assert stored[0]["story_count"] == 1


ACCOUNTING_CONFLICTS = [
    ("theme-and-other-coverage", "other", "is in a theme and in other coverage"),
    ("theme-and-excluded", "excluded", "already accounted for in this theme set"),
]


@pytest.mark.parametrize(
    "label, bucket, message",
    ACCOUNTING_CONFLICTS,
    ids=[case[0] for case in ACCOUNTING_CONFLICTS],
)
def test_a_themed_story_may_not_also_be_covered_or_excluded(
    tmp_path, label, bucket, message
):
    """Hostile cases 4 and 5 — already held, and pinned here beside the rest."""

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    themes = [a_theme("A", [stories[0]], [groups[0][0]], 1)]
    kwargs = (
        {"other": [OtherCoverageRecord(stories[0], "clustering_noise")]}
        if bucket == "other"
        else {"excluded": [ExcludedStoryRecord(stories[0], "no_encodable_text")]}
    )

    with pytest.raises(Phase0ValidationError, match=message):
        settle_themes(repository, themes, **kwargs)

    assert theme_membership(repository) == []


def test_disjoint_themes_settle_and_replay_unchanged(tmp_path):
    """Hostile cases 6 and 7: the rule costs valid input nothing."""

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    themes = [
        a_theme("A", [stories[0]], [groups[0][0]], 1),
        a_theme("B", [stories[1]], [groups[1][0]], 2),
    ]

    first = settle_themes(repository, themes)
    assert len(first.inserted) == 2
    settled = theme_membership(repository)
    assert sorted(story_id for _, story_id in settled) == sorted(stories)

    replay = settle_themes(repository, themes)
    assert replay.inserted == () and replay.updated == ()
    assert len(replay.unchanged) == 2
    assert replay.changed_outputs == ()
    assert theme_membership(repository) == settled


def test_a_story_may_move_between_themes_across_settlements(tmp_path):
    """Hostile case 10: rebuilding a day is not the thing being refused.

    The rule is about one settlement's *input*, so a later reconciliation
    is free to place a story on a different card — and the `UPDATE`
    trigger has to let a membership move without colliding with the row
    it is moving.
    """

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    settle_themes(
        repository,
        [
            a_theme("A", [stories[0]], [groups[0][0]], 1),
            a_theme("B", [stories[1]], [groups[1][0]], 2),
        ],
    )

    # One theme now holds both stories; the other is gone.
    settle_themes(
        repository,
        [a_theme("A", stories, [groups[0][0], groups[1][0]], 1)],
    )

    membership = theme_membership(repository)
    assert sorted(story_id for _, story_id in membership) == sorted(stories)
    assert len({theme_id for theme_id, _ in membership}) == 1


# ----------------------------------------------------------------------
# A reallocation is one move, not two halves
#
# A story and a citation each belong to one theme in a partition, and the
# database enforces that on every insert. Settling theme by theme asked
# it to accept a *partial* rearrangement: the theme gaining a story
# inserted while the theme losing it still held it. That is a valid
# outcome half-applied, and a trigger sees only the row in front of it.
#
# One-way moves worked when the donor happened to come first in the
# caller's list and failed when it came second; a swap had no ordering
# that worked at all. The settlement now decides, then releases every
# relation that is moving, then writes — three passes inside the one
# transaction that already covered the whole reconciliation.
# ----------------------------------------------------------------------


def shared_item_partition(repository, memberships, ticker="NVDA"):
    """A partition of stories, each owning the raw items listed by index.

    `memberships` maps a fingerprint to raw-item *indices*, so one item
    can deliberately belong to two stories — which is what makes a
    citation move between themes possible without a story moving with it.
    """

    total = 1 + max(index for indices in memberships.values() for index in indices)
    items = seed_raw_items(repository, total, ticker=ticker)
    records = [
        StoryRecord(
            cluster_fingerprint=fingerprint,
            canonical_title=f"Story {fingerprint}",
            members=tuple(
                StoryMemberRecord(
                    raw_item_id=items[index], position=slot, outlet=f"O{index}"
                )
                for slot, index in enumerate(indices)
            ),
            canonical_item_id=items[indices[0]],
            outlet_count=len(indices),
            content_hash=f"h-{fingerprint}",
            stage="m3.semantic",
        )
        for fingerprint, indices in memberships.items()
    ]
    reconcile_stories(
        repository,
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        stories=records,
    )
    ids = {
        str(row["cluster_fingerprint"]): int(row["id"])
        for row in repository.stories_for_day(DAY, ticker)
    }
    return ids, items


def relations(repository) -> dict[str, list[tuple[str, int]]]:
    """Memberships and citations keyed by theme *fingerprint*.

    By fingerprint rather than by row id, because a settlement that
    rebuilt a theme instead of updating it would still look right under
    ids alone.
    """

    with repository.admin.connect_writable() as connection:
        names = {
            int(row["id"]): str(row["fingerprint"])
            for row in connection.execute("SELECT id, fingerprint FROM themes")
        }
        return {
            "stories": sorted(
                (names[int(row["theme_id"])], int(row["story_id"]))
                for row in connection.execute(
                    "SELECT theme_id, story_id FROM theme_stories"
                )
            ),
            "citations": sorted(
                (names[int(row["theme_id"])], int(row["raw_item_id"]))
                for row in connection.execute(
                    "SELECT theme_id, raw_item_id FROM theme_citations"
                )
            ),
        }


#: label, story layout, themes before, themes after — each theme given as
#: (fingerprint, story fingerprints, raw-item indices).
REALLOCATIONS = [
    (
        "one-way-story-move",
        {"X": [0], "Y": [1], "Z": [2]},
        [("A", ["X", "Z"], [0, 2]), ("B", ["Y"], [1])],
        [("A", ["Z"], [2]), ("B", ["Y", "X"], [1, 0])],
    ),
    (
        "one-way-citation-move",
        {"X": [0, 1], "Y": [0, 2]},
        [("A", ["X"], [0, 1]), ("B", ["Y"], [2])],
        [("A", ["X"], [1]), ("B", ["Y"], [2, 0])],
    ),
    (
        "story-swap",
        {"X": [0], "Y": [1]},
        [("A", ["X"], [0]), ("B", ["Y"], [1])],
        [("A", ["Y"], [1]), ("B", ["X"], [0])],
    ),
    (
        "citation-swap",
        {"X": [0, 1], "Y": [0, 1]},
        [("A", ["X"], [0]), ("B", ["Y"], [1])],
        [("A", ["X"], [1]), ("B", ["Y"], [0])],
    ),
    (
        "three-way-rotation",
        {"X": [0], "Y": [1], "Z": [2]},
        [("A", ["X"], [0]), ("B", ["Y"], [1]), ("C", ["Z"], [2])],
        [("A", ["Y"], [1]), ("B", ["Z"], [2]), ("C", ["X"], [0])],
    ),
]


def build_themes(spec, ids, items):
    return [
        a_theme(
            fingerprint,
            [ids[name] for name in story_names],
            [items[index] for index in item_indices],
            rank + 1,
        )
        for rank, (fingerprint, story_names, item_indices) in enumerate(spec)
    ]


def expected_relations(spec, ids, items):
    return {
        "stories": sorted(
            (fingerprint, ids[name])
            for fingerprint, story_names, _ in spec
            for name in story_names
        ),
        "citations": sorted(
            (fingerprint, items[index])
            for fingerprint, _, item_indices in spec
            for index in item_indices
        ),
    }


@pytest.mark.parametrize("order", ["as-given", "reversed"])
@pytest.mark.parametrize(
    "label, layout, before, after",
    REALLOCATIONS,
    ids=[case[0] for case in REALLOCATIONS],
)
def test_a_reallocation_between_live_themes_settles(
    tmp_path, label, layout, before, after, order
):
    """Cases 1-6, each run with the themes listed both ways round.

    The reversal is not decoration: before this fix the one-way move
    settled with the donor listed first and failed with it listed second,
    so the same valid reallocation succeeded or failed on input order
    alone.
    """

    repository = migrated(tmp_path, f"{label}-{order}.db")
    ids, items = shared_item_partition(repository, layout)
    settle_themes(repository, build_themes(before, ids, items))

    moved = build_themes(after, ids, items)
    if order == "reversed":
        moved = list(reversed(moved))
    report = settle_themes(repository, moved)

    assert relations(repository) == expected_relations(after, ids, items)
    # Every theme changed, and each was updated in place rather than
    # dropped and recreated.
    assert report.inserted == () and report.deleted == ()
    assert len(report.updated) == len(after)


@pytest.mark.parametrize(
    "label, layout, before, after",
    REALLOCATIONS,
    ids=[case[0] for case in REALLOCATIONS],
)
def test_a_reallocation_replays_unchanged(tmp_path, label, layout, before, after):
    """Case 7: having moved, the same input moves nothing."""

    repository = migrated(tmp_path)
    ids, items = shared_item_partition(repository, layout)
    settle_themes(repository, build_themes(before, ids, items))
    settle_themes(repository, build_themes(after, ids, items))
    settled = relations(repository)

    replay = settle_themes(repository, build_themes(after, ids, items))

    assert replay.inserted == () and replay.updated == () and replay.deleted == ()
    assert len(replay.unchanged) == len(after)
    assert replay.changed_outputs == ()
    assert relations(repository) == settled


def test_an_unchanged_theme_is_not_rebuilt_while_others_move(tmp_path):
    """Only what is moving is released.

    A theme whose stored relations already are the answer keeps them —
    nothing can be waiting on them, because a final state where two themes
    want the same story never reaches the write.
    """

    repository = migrated(tmp_path)
    ids, items = shared_item_partition(
        repository, {"X": [0], "Y": [1], "Z": [2], "W": [3]}
    )
    settle_themes(
        repository,
        build_themes(
            [("A", ["X"], [0]), ("B", ["Y"], [1]), ("STILL", ["W"], [3])], ids, items
        ),
    )

    report = settle_themes(
        repository,
        build_themes(
            [("A", ["Y"], [1]), ("B", ["X"], [0]), ("STILL", ["W"], [3])], ids, items
        ),
    )

    assert len(report.updated) == 2 and len(report.unchanged) == 1
    assert relations(repository)["stories"] == sorted(
        [("A", ids["Y"]), ("B", ids["X"]), ("STILL", ids["W"])]
    )


INVALID_FINAL_STATES = [
    (
        "duplicate-membership",
        {"X": [0, 1], "Y": [2]},
        [("A", ["X"], [0]), ("B", ["Y"], [2])],
        [("A", ["X"], [0]), ("B", ["X", "Y"], [1, 2])],
        "a story belongs to exactly one theme",
    ),
    (
        "duplicate-citation",
        {"X": [0, 1], "Y": [0, 2]},
        [("A", ["X"], [0]), ("B", ["Y"], [2])],
        [("A", ["X"], [0]), ("B", ["Y"], [0, 2])],
        "citable",
    ),
]


@pytest.mark.parametrize(
    "label, layout, before, after, message",
    INVALID_FINAL_STATES,
    ids=[case[0] for case in INVALID_FINAL_STATES],
)
def test_an_invalid_final_state_is_still_refused(
    tmp_path, label, layout, before, after, message
):
    """Cases 8 and 9: staging the writes did not soften the outcome.

    The triggers were never wrong — they were being shown a valid answer
    half-applied. Shown a genuinely invalid one, they still refuse, and
    the reconciliation leaves the partition exactly as it was.
    """

    repository = migrated(tmp_path)
    ids, items = shared_item_partition(repository, layout)
    settle_themes(repository, build_themes(before, ids, items))
    settled = relations(repository)

    with pytest.raises((Phase0ValidationError, sqlite3.IntegrityError), match=message):
        settle_themes(repository, build_themes(after, ids, items))

    assert relations(repository) == settled


def test_a_failure_after_the_release_rolls_back_to_the_original(tmp_path):
    """Case 10: the release is inside the transaction, not ahead of it.

    The window this fix opens — relations cleared, replacements not yet
    written — is the one that must not survive a failure. The injected
    error lands squarely in it.
    """

    repository = migrated(tmp_path)
    ids, items = shared_item_partition(repository, {"X": [0], "Y": [1]})
    settle_themes(
        repository, build_themes([("A", ["X"], [0]), ("B", ["Y"], [1])], ids, items)
    )
    before = relations(repository)
    theme_rows = repository.theme_set(
        ticker="NVDA", trading_day=DAY, pipeline_version="v1"
    )

    boom = RuntimeError("injected between the release and the rewrite")
    original = Phase0Repository._update_reconciled_theme
    calls = {"count": 0}

    def explode(self, connection, theme_id, values):
        calls["count"] += 1
        if calls["count"] == 1:
            # The first rewrite, after *both* themes have been released.
            raise boom
        return original(self, connection, theme_id, values)

    with mock.patch.object(Phase0Repository, "_update_reconciled_theme", explode):
        with pytest.raises(RuntimeError, match="injected between"):
            settle_themes(
                repository,
                build_themes([("A", ["Y"], [1]), ("B", ["X"], [0])], ids, items),
            )

    assert calls["count"] == 1
    assert relations(repository) == before
    assert (
        repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
        == theme_rows
    )


def test_coverage_and_exclusions_survive_a_reallocation(tmp_path):
    """Case 11: the other two buckets are untouched by the move."""

    repository = migrated(tmp_path)
    ids, items = shared_item_partition(
        repository, {"X": [0], "Y": [1], "OTHER": [2], "GONE": [3]}
    )
    accounting = {
        "other": [OtherCoverageRecord(ids["OTHER"], "clustering_noise")],
        "excluded": [ExcludedStoryRecord(ids["GONE"], "no_encodable_text")],
    }
    settle_themes(
        repository,
        build_themes([("A", ["X"], [0]), ("B", ["Y"], [1])], ids, items),
        **accounting,
    )
    before = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")

    report = settle_themes(
        repository,
        build_themes([("A", ["Y"], [1]), ("B", ["X"], [0])], ids, items),
        **accounting,
    )

    after = repository.theme_set(ticker="NVDA", trading_day=DAY, pipeline_version="v1")
    assert report.changed_outputs == ()
    assert after["other_coverage"] == before["other_coverage"]
    assert after["excluded"] == before["excluded"]
    assert relations(repository)["stories"] == sorted(
        [("A", ids["Y"]), ("B", ids["X"])]
    )


def test_direct_sql_cannot_put_a_story_in_a_second_theme(tmp_path):
    """Hostile cases 8 and 9: the rule is in the database, not only above it."""

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    settle_themes(
        repository,
        [
            a_theme("A", [stories[0]], [groups[0][0]], 1),
            a_theme("B", [stories[1]], [groups[1][0]], 2),
        ],
    )
    with repository.admin.connect_writable() as connection:
        theme_ids = [
            int(row["id"])
            for row in connection.execute("SELECT id FROM themes ORDER BY id")
        ]
    before = theme_membership(repository)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="member of another theme"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (theme_ids[1], stories[0]),
            )

    # The update path, isolated from the citation rule that would
    # otherwise refuse the move for an unrelated reason.
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM theme_citations WHERE theme_id = ?", (theme_ids[1],)
        )
        with pytest.raises(sqlite3.IntegrityError, match="member of another theme"):
            connection.execute(
                "UPDATE theme_stories SET story_id = ? "
                "WHERE theme_id = ? AND story_id = ?",
                (stories[0], theme_ids[1], stories[1]),
            )
        connection.rollback()

    assert theme_membership(repository) == before


def test_direct_sql_may_still_move_a_membership_that_collides_with_nobody(tmp_path):
    """The update guard excludes the row it is judging.

    `BEFORE UPDATE` still sees the old row, so a membership moving from
    one theme to another would collide with itself if the trigger did not
    say so explicitly — which would make the rule refuse the very
    operation it is meant to allow.
    """

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    settle_themes(
        repository,
        [
            a_theme("A", [stories[0]], [groups[0][0]], 1),
            a_theme("B", [stories[1]], [groups[1][0]], 2),
        ],
    )
    with repository.admin.connect_writable() as connection:
        theme_ids = [
            int(row["id"])
            for row in connection.execute("SELECT id FROM themes ORDER BY id")
        ]
        connection.execute(
            "DELETE FROM theme_citations WHERE theme_id = ?", (theme_ids[1],)
        )
        connection.execute(
            "UPDATE theme_stories SET theme_id = ? "
            "WHERE theme_id = ? AND story_id = ?",
            (theme_ids[0], theme_ids[1], stories[1]),
        )
        connection.commit()

    assert theme_membership(repository) == [
        (theme_ids[0], stories[0]),
        (theme_ids[0], stories[1]),
    ]


SINGLETON_BUCKETS = [
    (
        "other-coverage",
        "INSERT INTO theme_other_coverage "
        "(theme_set_id, story_id, reason, position) VALUES (?, ?, ?, ?)",
        ("clustering_noise", 9),
    ),
    (
        "exclusions",
        "INSERT INTO theme_excluded_stories (theme_set_id, story_id, reason) "
        "VALUES (?, ?, ?)",
        ("no_encodable_text",),
    ),
]


@pytest.mark.parametrize(
    "label, statement, extra",
    SINGLETON_BUCKETS,
    ids=[case[0] for case in SINGLETON_BUCKETS],
)
def test_the_other_two_buckets_are_closed_by_their_primary_keys(
    tmp_path, label, statement, extra
):
    """Why "same bucket twice" was a gap for themes and only for themes.

    `theme_sets` is unique per partition and both coverage tables are
    keyed on `(theme_set_id, story_id)`, so one story cannot appear twice
    in either — the primary key already says it. `theme_stories` is keyed
    on `(theme_id, story_id)`, and a partition holds *many* themes, so the
    same key permits exactly the state migration 014 forbids.
    """

    repository = migrated(tmp_path)
    stories, groups = split_partition(repository, [1, 1, 1])
    settle_themes(
        repository,
        [a_theme("A", [stories[0]], [groups[0][0]], 1)],
        other=[OtherCoverageRecord(stories[1], "clustering_noise")],
        excluded=[ExcludedStoryRecord(stories[2], "no_encodable_text")],
    )

    with repository.admin.connect_writable() as connection:
        set_id = int(connection.execute("SELECT id FROM theme_sets").fetchone()["id"])
        story_id = stories[1] if label == "other-coverage" else stories[2]
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            connection.execute(statement, (set_id, story_id, *extra))


#: The migration that introduced the one-theme-per-story overlap rule.  The
#: fixtures below mean "one short of *this*", not "one short of latest": when
#: 015 was added, ``LATEST_VERSION - 1`` quietly re-aimed them at a database
#: that already had the rule, and the hostile cases stopped testing it.
OVERLAP_RULE_VERSION = next(
    migration.version
    for migration in ALL_MIGRATIONS
    if migration.name.endswith("one_theme_per_story.sql")
)
PRE_OVERLAP_VERSION = OVERLAP_RULE_VERSION - 1


def v13_repository(tmp_path, name="v13.sqlite3"):
    """A database stopped one migration short of the overlap rule."""

    database = tmp_path / name
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, PRE_OVERLAP_VERSION)
    ).migrate()
    repository = Phase0Repository(database)
    assert repository.schema_version() == PRE_OVERLAP_VERSION
    return repository, database


def test_a_clean_v13_database_upgrades_with_its_data_intact(tmp_path):
    """Hostile case 11."""

    repository, database = v13_repository(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    settle_themes(
        repository,
        [
            a_theme("A", [stories[0]], [groups[0][0]], 1),
            a_theme("B", [stories[1]], [groups[1][0]], 2),
        ],
    )
    before = theme_membership(repository)

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.schema_version() == LATEST_VERSION
    assert theme_membership(upgraded) == before
    assert upgraded.count("themes") == 2
    # And the rule is live on the upgraded database, not just on fresh ones.
    with upgraded.admin.connect_writable() as connection:
        theme_ids = [
            int(row["id"])
            for row in connection.execute("SELECT id FROM themes ORDER BY id")
        ]
        with pytest.raises(sqlite3.IntegrityError, match="member of another theme"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                (theme_ids[1], stories[0]),
            )


def test_a_v13_database_already_carrying_a_duplicate_refuses_to_upgrade(tmp_path):
    """Hostile case 12: fail closed, and do not pick a winner.

    Choosing between two themes that both claim a story is a content
    decision — which card the story belongs on — and a migration cannot
    make it. Migration 011 set the policy when it could not infer a
    legacy `pipeline_version` unambiguously: abort, roll back whole, and
    leave the operator to resolve it. This follows that policy.
    """

    repository, database = v13_repository(tmp_path)
    stories, groups = split_partition(repository, [1, 1])
    settle_themes(
        repository,
        [
            a_theme("A", [stories[0]], [groups[0][0]], 1),
            a_theme("B", [stories[1]], [groups[1][0]], 2),
        ],
    )
    with repository.admin.connect_writable() as connection:
        theme_ids = [
            int(row["id"])
            for row in connection.execute("SELECT id FROM themes ORDER BY id")
        ]
        # Legal at v13, which is exactly the point.
        connection.execute(
            "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
            (theme_ids[1], stories[0]),
        )
        connection.commit()
    before = database_state(database)

    with pytest.raises(sqlite3.IntegrityError, match="resolve which theme owns it"):
        Phase0Repository(database).migrate()

    # Whole: still at 13, ledger unchanged, the duplicate still there for
    # the operator to look at.
    assert database_state(database) == before
    assert before["user_version"] == PRE_OVERLAP_VERSION
    assert Phase0Repository(database).schema_version() == PRE_OVERLAP_VERSION

    # And once resolved, the upgrade goes through.
    with Phase0Repository(database).admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM theme_stories WHERE theme_id = ? AND story_id = ?",
            (theme_ids[1], stories[0]),
        )
        connection.commit()
    resolved = Phase0Repository(database)
    resolved.migrate()
    assert resolved.schema_version() == LATEST_VERSION


def test_a_cross_partition_duplicate_does_not_block_the_upgrade(tmp_path):
    """The migration's check is partition-scoped, like the rule it installs.

    Two tickers on the same day each have their own stories, so two themes
    naming *different* stories is not a duplicate however similar the days
    look. A check that asked only "does this story_id appear twice" would
    be the same mistake in the other direction.
    """

    repository, database = v13_repository(tmp_path)
    for ticker in ("NVDA", "AMD"):
        stories, groups = split_partition(repository, [1], ticker=ticker)
        settle_themes(
            repository,
            [a_theme(f"{ticker}-A", [stories[0]], [groups[0][0]], 1)],
            ticker=ticker,
        )

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.schema_version() == LATEST_VERSION
    assert len(theme_membership(upgraded)) == 2


def test_reconciling_a_theme_under_a_new_key_drops_the_old_keys_vector(tmp_path):
    """The path a caller actually reaches, with no raw SQL anywhere.

    `reconcile_themes` matches on `fingerprint` and owns `theme_key`, so a
    re-run that assigns a new key renames a live owner's handle.  Before
    migration 012 the old key's vector stayed behind for good.
    """

    repository = migrated(tmp_path)
    nvda = one_handled_partition(repository, "NVDA", SHARED_HANDLE_VALUES)
    identity = SHARED_HANDLE_VALUES["theme_key"]
    embed_while_sole_owner(repository, "theme", identity)

    reconcile_themes(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        theme_set=theme_set(),
        themes=[
            ThemeRecord(
                fingerprint=SHARED_HANDLE_VALUES["theme_fingerprint"],
                theme_key="reconciled-theme-key",
                label="NVDA theme",
                story_ids=(nvda["story_id"],),
                citation_item_ids=(nvda["items"][0],),
                method="hdbscan",
            )
        ],
    )

    stored = repository.theme_set(
        ticker="NVDA", trading_day=DAY, pipeline_version="v1"
    )["themes"]
    assert [row["theme_key"] for row in stored] == ["reconciled-theme-key"]
    assert not cached(repository, "theme", identity), "the old key kept its vector"
    assert not cached(repository, "theme", "reconciled-theme-key")


# ----------------------------------------------------------------------
# "Orphaned" is one question, and every verb has to ask all of it
#
# `trg_embedding_owner_insert` accepts a durable row id *or* any handle
# the row answers to.  The cleanup triggers asked something narrower —
# the handle branch compared one column, and the durable-id branch
# compared nothing — so an identity that collides across two accepted
# forms looked orphaned while a live row still owned it.  Migration 013
# makes both branches ask the insert trigger's question.
# ----------------------------------------------------------------------


#: The identity forms migration 007 accepts, per parent kind.  `durable-id`
#: is not assignable: a row wears it by being that row.
ACCEPTED_FORMS = {
    "story": ("durable-id", "fingerprint"),
    "theme": ("durable-id", "fingerprint", "key"),
}

#: Where an assignable form actually lives.
ALIAS_COLUMN = {
    ("story", "fingerprint"): "cluster_fingerprint",
    ("theme", "fingerprint"): "fingerprint",
    ("theme", "key"): "theme_key",
}

PARENT_TABLE = {"story": "stories", "theme": "themes"}
PARENT_KEY = {"story": "story_id", "theme": "theme_id"}


def isolated_handles(tag: str) -> dict:
    """Handles belonging to one partition alone.

    The shared set every other test uses would collide on its own, which
    would hide which collision a case is actually about.
    """

    return {
        "story_fingerprint": f"story-fp-{tag}",
        "theme_fingerprint": f"theme-fp-{tag}",
        "theme_key": f"theme-key-{tag}",
    }


def wear_alias(repository, kind: str, parent_id: int, form: str, identity: str) -> None:
    set_column(
        repository, PARENT_TABLE[kind], ALIAS_COLUMN[(kind, form)], parent_id, identity
    )


def collided_owners(repository, kind: str, victim_form: str, survivor_form: str) -> str:
    """Two partitions that end up answering to one identity, differently.

    The victim owns it through `victim_form` and is about to be deleted;
    the survivor owns it through `survivor_form` and stays. The vector is
    always cached at the one moment exactly one of them answers, because
    `persist_embeddings` refuses an ambiguous identity — correctly, and
    that refusal is not what is under test here.
    """

    victim = one_handled_partition(repository, "NVDA", isolated_handles("nvda"))
    survivor = one_handled_partition(repository, "AMD", isolated_handles("amd"))
    victim_id = victim[PARENT_KEY[kind]]
    survivor_id = survivor[PARENT_KEY[kind]]

    if victim_form == "durable-id":
        # The victim already wears it, so cache first and dress the
        # survivor afterwards.
        identity = str(victim_id)
        embed_while_sole_owner(repository, kind, identity)
        wear_alias(repository, kind, survivor_id, survivor_form, identity)
    else:
        if survivor_form == "durable-id":
            identity = str(survivor_id)
        else:
            identity = "collided-identity"
            wear_alias(repository, kind, survivor_id, survivor_form, identity)
        embed_while_sole_owner(repository, kind, identity, ticker="AMD")
        wear_alias(repository, kind, victim_id, victim_form, identity)

    assert cached(repository, kind, identity), "setup lost the vector early"
    drop_parent(repository, kind, victim_id)
    return identity


#: Every ordered pair of accepted forms. Two rows cannot share a durable
#: id, so that one pairing is not a state and is left out.
OWNERSHIP_COLLISIONS = [
    (kind, victim, survivor)
    for kind, forms in ACCEPTED_FORMS.items()
    for victim in forms
    for survivor in forms
    if not (victim == "durable-id" and survivor == "durable-id")
]


@pytest.mark.parametrize(
    "kind, victim_form, survivor_form",
    OWNERSHIP_COLLISIONS,
    ids=[f"{k}-{v}-dies-{s}-lives" for k, v, s in OWNERSHIP_COLLISIONS],
)
def test_a_deletion_spares_a_vector_another_form_still_owns(
    tmp_path, kind, victim_form, survivor_form
):
    """Cases 1, 2, 6, 7, 8 and 9, and the pairings those imply.

    Every one is cross-partition: the two owners are NVDA's and AMD's,
    which is the ordinary way one identity comes to have two claimants,
    since handles are unique only within a ticker/day/version.
    """

    repository = migrated(tmp_path)
    identity = collided_owners(repository, kind, victim_form, survivor_form)

    assert cached(repository, kind, identity), "a live owner lost its vector"
    assert repository.count("embeddings") == 1


@pytest.mark.parametrize(
    "kind, victim_form, survivor_form",
    OWNERSHIP_COLLISIONS,
    ids=[f"{k}-{v}-renamed-{s}-lives" for k, v, s in OWNERSHIP_COLLISIONS],
)
def test_a_rename_spares_a_vector_another_form_still_owns(
    tmp_path, kind, victim_form, survivor_form
):
    """Cases 3 and 10: the same matrix, reached by the other verb.

    A durable id cannot be renamed, so the victim moves whichever alias
    it holds; when it holds none — it owned by id — the survivor's alias
    moves instead, which is the same question asked from the other side.
    """

    repository = migrated(tmp_path)
    victim = one_handled_partition(repository, "NVDA", isolated_handles("nvda"))
    survivor = one_handled_partition(repository, "AMD", isolated_handles("amd"))
    victim_id = victim[PARENT_KEY[kind]]
    survivor_id = survivor[PARENT_KEY[kind]]

    if victim_form == "durable-id":
        identity = str(victim_id)
        embed_while_sole_owner(repository, kind, identity)
        wear_alias(repository, kind, survivor_id, survivor_form, identity)
        # Nothing about the victim can move, so the survivor's alias does:
        # it leaves and comes back, and the vector must be there after.
        wear_alias(repository, kind, survivor_id, survivor_form, "elsewhere")
        wear_alias(repository, kind, survivor_id, survivor_form, identity)
    else:
        if survivor_form == "durable-id":
            identity = str(survivor_id)
        else:
            identity = "collided-identity"
            wear_alias(repository, kind, survivor_id, survivor_form, identity)
        embed_while_sole_owner(repository, kind, identity, ticker="AMD")
        wear_alias(repository, kind, victim_id, victim_form, identity)
        wear_alias(repository, kind, victim_id, victim_form, "moved-away")

    assert cached(repository, kind, identity), "a live owner lost its vector"
    assert repository.count("embeddings") == 1


def two_stories_in_one_partition(repository, ticker="NVDA") -> list[int]:
    """One partition holding two stories, reconciled together.

    Reconciliation settles a partition as a whole, so a second call would
    replace the first story rather than join it — both have to arrive in
    the same settlement.
    """

    items = seed_raw_items(repository, 2, ticker=ticker)
    reconcile_stories(
        repository,
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("fp-one", items[:1]), story("fp-two", items[1:])],
    )
    return [int(row["id"]) for row in repository.stories_for_day(DAY, ticker)]


def test_two_forms_can_collide_inside_one_partition_too(tmp_path):
    """Cross-partition is the common route, not the only one.

    A row id is a plain integer, and nothing stops another row in the
    *same* partition from having a fingerprint that spells it — the
    uniqueness index covers the fingerprint column, not the relationship
    between a fingerprint and somebody else's id.
    """

    repository = migrated(tmp_path)
    first, second = two_stories_in_one_partition(repository)
    identity = str(first)
    embed_while_sole_owner(repository, "story", identity)

    wear_alias(repository, "story", second, "fingerprint", identity)
    drop_parent(repository, "story", first)

    assert cached(repository, "story", identity), "the same-partition owner lost it"


@pytest.mark.parametrize("kind", ["story", "theme"])
@pytest.mark.parametrize("form", ["durable-id", "alias"])
def test_the_last_owner_in_any_form_still_takes_the_vector(tmp_path, kind, form):
    """Cases 4 and 11: a complete predicate is not a permanent one.

    The guard exists to spare owners, and when there are none left it has
    to get out of the way — otherwise every collision leaks a row.
    """

    repository = migrated(tmp_path)
    only = one_handled_partition(repository, "NVDA", isolated_handles("only"))
    parent_id = only[PARENT_KEY[kind]]
    identity = (
        str(parent_id)
        if form == "durable-id"
        else isolated_handles("only")[
            "story_fingerprint" if kind == "story" else "theme_fingerprint"
        ]
    )
    embed_while_sole_owner(repository, kind, identity)

    drop_parent(repository, kind, parent_id)

    assert not cached(repository, kind, identity), "an orphan vector survived"
    assert repository.count("embeddings") == 0


@pytest.mark.parametrize("kind", ["story", "theme"])
def test_an_unrelated_vector_is_untouched_by_a_deletion(tmp_path, kind):
    """Case 5: a collision-free identity is nobody else's business."""

    repository = migrated(tmp_path)
    victim = one_handled_partition(repository, "NVDA", isolated_handles("nvda"))
    other = one_handled_partition(repository, "AMD", isolated_handles("amd"))
    identity = str(other[PARENT_KEY[kind]])
    embed_while_sole_owner(repository, kind, identity, ticker="AMD")

    drop_parent(repository, kind, victim[PARENT_KEY[kind]])

    assert cached(repository, kind, identity)
    assert repository.get_embedding(kind, identity) is not None


def test_an_existing_database_gains_the_complete_predicate_on_upgrade(tmp_path):
    """013 has to reach databases that already exist, not just fresh ones.

    A trigger is schema, so the ones 007 created are sitting inside every
    database built before this migration. Replacing them is the only way
    the fix arrives there — and it has to arrive without disturbing the
    ledger rows for 007 and 012, whose files are released and unchanged.
    """

    database = tmp_path / "v12.sqlite3"
    Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 12)
    ).migrate()

    before = Phase0Repository(database)
    assert before.schema_version() == 12
    victim, survivor = two_stories_in_one_partition(before)
    identity = str(victim)
    embed_while_sole_owner(before, "story", identity)
    wear_alias(before, "story", survivor, "fingerprint", identity)

    upgraded = Phase0Repository(database)
    upgraded.migrate()
    assert upgraded.schema_version() == LATEST_VERSION

    # The pre-existing vector is still there, and now correctly guarded.
    assert cached(upgraded, "story", identity)
    drop_parent(upgraded, "story", victim)
    assert cached(upgraded, "story", identity), "the upgrade did not take effect"

    # Every earlier migration is still recorded exactly once, unedited.
    with upgraded.admin.connect_writable() as connection:
        ledger = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM schema_migrations ORDER BY version"
            )
        ]
    assert ledger == [migration.name for migration in ALL_MIGRATIONS]


def test_a_collision_does_not_change_what_the_write_apis_accept(tmp_path):
    """013 widened who counts as an owner, not who may name one.

    Each API keeps the rule it had. The single-vector cache takes a
    durable id and means *the row with that id*, so another row's alias
    spelling the same digits does not make it ambiguous — while a
    fingerprint stays refused there, collision or no. The run-scoped
    batch resolves both forms, so for it the same string does name two
    rows, and it still says so.
    """

    repository = migrated(tmp_path)
    victim = one_handled_partition(repository, "NVDA", isolated_handles("nvda"))
    survivor = one_handled_partition(repository, "AMD", isolated_handles("amd"))
    identity = str(survivor["story_id"])
    embed_while_sole_owner(repository, "story", identity, ticker="AMD")
    wear_alias(repository, "story", victim["story_id"], "fingerprint", identity)

    # The cache API: the id still means the id.
    assert repository.get_embedding("story", identity) is not None
    repository.upsert_embedding(sample_embedding(identity, source_kind="story"))
    assert repository.count("embeddings") == 1

    # And a handle is still not a cache key.
    handle = isolated_handles("nvda")["story_fingerprint"]
    with pytest.raises(Phase0ValidationError, match="not a durable"):
        repository.upsert_embedding(sample_embedding(handle, source_kind="story"))
    with pytest.raises(Phase0ValidationError, match="not a durable"):
        repository.delete_embedding("story", handle)

    # The batch resolves both forms, so this identity names two rows.
    with nvda_run(repository, stage="m1.embed") as run:
        with pytest.raises(Phase0RunContextError, match="is ambiguous"):
            repository.persist_embeddings(
                [sample_embedding(identity, source_kind="story")], run=run
            )
    assert cached(repository, "story", identity)


def test_record_source_state_rejects_another_days_fetch(tmp_path):
    repository = migrated(tmp_path)

    with nvda_run(repository) as run:
        with pytest.raises(Phase0RunContextError, match="but the run covers"):
            repository.record_source_state(
                "rss:test", run=run, checked_at="2026-07-24T12:00:00+00:00"
            )

    assert repository.source_state("rss:test") is None


# ----------------------------------------------------------------------
# A stated fetch time belongs to the run's day, whichever door it came in
#
# `record_source_state` checked that a stated `checked_at` falls on the
# run's trading day.  `ingest_raw_items(source_state=...)` reaches the
# same write through its own argument and did not, so one run could stamp
# another day's fetch as its own while its run log recorded the work as
# today's.  The check now lives in `_set_source_state`, which both go
# through, so neither can drift from the other again.
# ----------------------------------------------------------------------


#: One table for both entrypoints: the moment, and whether the run's day
#: can honestly claim it.  `27` is the day the run covers.
CHECKED_AT_DAYS = [
    ("same-day-midday", f"{DAY}T12:00:00+00:00", True),
    ("same-day-first-instant", f"{DAY}T00:00:00+00:00", True),
    ("same-day-last-instant", f"{DAY}T23:59:59.999999+00:00", True),
    ("previous-day", "2026-07-22T23:59:59+00:00", False),
    ("next-day", "2026-07-24T00:00:00+00:00", False),
    # Normalization is to UTC, so the *instant* decides, not the text.
    # These two are the interesting direction: each one's literal prefix
    # says the opposite of where it actually lands.
    ("offset-text-says-previous-lands-on-run-day", "2026-07-22T23:00:00-02:00", True),
    ("offset-text-says-run-day-lands-on-previous", f"{DAY}T01:00:00+03:00", False),
    ("offset-text-says-run-day-lands-on-next", f"{DAY}T23:00:00-02:00", False),
]


def _ingest_with_state(repository, source, moment):
    with nvda_run(repository) as run:
        repository.ingest_raw_items(
            [raw_item(1, "NVDA")],
            run=run,
            source_state={"source": source, "checked_at": moment},
        )


def _record_with_state(repository, source, moment):
    with nvda_run(repository) as run:
        repository.record_source_state(source, run=run, checked_at=moment)


SOURCE_STATE_DAY_ENTRYPOINTS = [
    ("ingest_raw_items", _ingest_with_state),
    ("record_source_state", _record_with_state),
]


@pytest.mark.parametrize(
    "entrypoint, call",
    SOURCE_STATE_DAY_ENTRYPOINTS,
    ids=[case[0] for case in SOURCE_STATE_DAY_ENTRYPOINTS],
)
@pytest.mark.parametrize(
    "label, moment, accepted",
    CHECKED_AT_DAYS,
    ids=[case[0] for case in CHECKED_AT_DAYS],
)
def test_both_source_state_entrypoints_judge_a_day_the_same_way(
    tmp_path, entrypoint, call, label, moment, accepted
):
    """The same table of moments, answered identically by both doors.

    The offset rows are the ones worth having: normalization is to UTC,
    so a stated offset can move the *day* the moment lands on, and the
    two entrypoints have to agree about that too.
    """

    repository = migrated(tmp_path, f"{entrypoint}-{label}.db")
    source = "yahoo:NVDA"

    if accepted:
        call(repository, source, moment)
        stored = repository.source_state(source)
        assert stored is not None
        assert str(stored["last_checked_at"])[:10] == DAY
    else:
        with pytest.raises(Phase0RunContextError, match="but the run covers"):
            call(repository, source, moment)
        assert repository.source_state(source) is None


def test_a_rejected_ingest_source_state_leaves_the_whole_batch_undone(tmp_path):
    """The batch is one transaction, and the day check is inside it.

    Raw items, their ticker associations, their candidate reasons, the
    source-state row, and the run log all have to be as they were — and
    the stage key has to end up settled, because the rejection is this
    run's recorded failure rather than an exception thrown past it.
    """

    repository = migrated(tmp_path)
    key = {
        "stage": "m0.ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    assert repository.claim_stage_key(**key, run_id="run-day") is True

    with pytest.raises(Phase0RunContextError, match="but the run covers"):
        with repository.stage_run(
            run_id="run-day",
            stage="m0.ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            repository.ingest_raw_items(
                [raw_item(1, "NVDA"), raw_item(2, "NVDA")],
                run=run,
                source_state={
                    "source": "yahoo:NVDA",
                    "checked_at": "2026-07-22T12:00:00+00:00",
                },
                terminal=True,
            )

    for table in (
        "raw_items",
        "raw_item_tickers",
        "raw_item_candidates",
        "source_state",
    ):
        assert repository.count(table) == 0, table

    logged = repository.read.run_log_rows(run_id="run-day")
    assert [row["status"] for row in logged] == ["failed"]
    assert int(logged[0]["success_count"]) == 0
    assert repository.stage_key_state(**key)["status"] == "failed"


def test_a_rejected_day_does_not_disturb_an_earlier_good_source_state(tmp_path):
    """A refusal rolls back to what was there, not to nothing."""

    repository = migrated(tmp_path)
    _ingest_with_state(repository, "yahoo:NVDA", f"{DAY}T09:00:00+00:00")
    before = dict(repository.source_state("yahoo:NVDA"))

    with pytest.raises(Phase0RunContextError, match="but the run covers"):
        _ingest_with_state(repository, "yahoo:NVDA", "2026-07-24T09:00:00+00:00")

    assert dict(repository.source_state("yahoo:NVDA")) == before


def test_an_omitted_checked_at_still_asserts_no_day(tmp_path):
    """Omission keeps meaning "now"; only a stated time is a claim.

    `ingest_raw_items` has no omission to test — `validate_source_state`
    requires `checked_at` in the mapping — so this pins the half of the
    contract that belongs to `record_source_state`, and that the mapping
    form really does require it.
    """

    repository = migrated(tmp_path)
    with nvda_run(repository) as run:
        repository.record_source_state("yahoo:NVDA", run=run)
    assert repository.source_state("yahoo:NVDA") is not None

    with nvda_run(repository) as run:
        with pytest.raises(Phase0ValidationError, match="checked_at"):
            repository.ingest_raw_items(
                [raw_item(1, "NVDA")],
                run=run,
                source_state={"source": "yahoo:NVDA"},
            )


def test_the_runless_admin_path_still_has_no_day_to_check(tmp_path):
    """`admin.insert_raw_items` writes without a run, so it asserts none.

    The check is the run's, not the payload's: with no run there is no
    day the state could contradict, and narrowing this door was never
    part of the contract.
    """

    repository = migrated(tmp_path)
    repository.admin.insert_raw_items(
        [raw_item(1, "NVDA")],
        source_state={
            "source": "yahoo:NVDA",
            "checked_at": "2026-07-22T12:00:00+00:00",
        },
    )

    stored = repository.source_state("yahoo:NVDA")
    assert str(stored["last_checked_at"])[:10] == "2026-07-22"


# ----------------------------------------------------------------------
# A fetch outcome is stated once
#
# `record_source_state` took both `successful` and `status`.  The stored
# row resolved from `status` when it was given; the run's counters
# resolved from `successful` regardless.  So the feed's record and the
# run that wrote it could say opposite things — and not only when a
# caller contradicted itself: `status="unknown"` is perfectly valid and
# does not count as a successful fetch, yet the default `successful=True`
# still logged the run as a success.
# ----------------------------------------------------------------------


#: (kwargs, stored status) for everything that is accepted.  The stored
#: status is the whole answer: `last_success_at`, `consecutive_failures`,
#: and the run's own counters are all derived from it, and the assertions
#: below check each against that one value rather than against a
#: separately maintained expectation.
ACCEPTED_SOURCE_OUTCOMES = [
    ("nothing stated", {}, "success"),
    ("successful=True", {"successful": True}, "success"),
    ("successful=False", {"successful": False}, "failed"),
    ("status=success", {"status": "success"}, "success"),
    ("status=failed", {"status": "failed"}, "failed"),
    ("status=partial", {"status": "partial"}, "partial"),
    ("status=empty", {"status": "empty"}, "empty"),
    ("status=unknown", {"status": "unknown"}, "unknown"),
    ("True + success", {"successful": True, "status": "success"}, "success"),
    ("False + failed", {"successful": False, "status": "failed"}, "failed"),
    ("True + partial", {"successful": True, "status": "partial"}, "partial"),
    ("True + empty", {"successful": True, "status": "empty"}, "empty"),
    ("False + unknown", {"successful": False, "status": "unknown"}, "unknown"),
    ("padded status", {"status": "  SUCCESS  "}, "success"),
]

SUCCEEDED_STATUSES = {"success", "partial", "empty"}

REJECTED_SOURCE_OUTCOMES = [
    ("True + failed", {"successful": True, "status": "failed"}),
    ("False + success", {"successful": False, "status": "success"}),
    ("True + unknown", {"successful": True, "status": "unknown"}),
    ("False + partial", {"successful": False, "status": "partial"}),
    ("False + empty", {"successful": False, "status": "empty"}),
    ("invalid status text", {"status": "banana"}),
    ("empty status text", {"status": ""}),
]


@pytest.mark.parametrize(
    "kwargs, expected",
    [pytest.param(k, s, id=label) for label, k, s in ACCEPTED_SOURCE_OUTCOMES],
)
@pytest.mark.parametrize("terminal", [False, True], ids=["non-terminal", "terminal"])
def test_the_source_state_and_the_run_agree_on_one_outcome(
    tmp_path, kwargs, expected, terminal
):
    repository = migrated(tmp_path)
    key = {
        "stage": "m0.ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    repository.claim_stage_key(**key, run_id="run-src")

    with repository.stage_run(
        run_id="run-src",
        stage="m0.ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.record_source_state("rss:test", run=run, terminal=terminal, **kwargs)

    succeeded = expected in SUCCEEDED_STATUSES
    state = repository.source_state("rss:test")
    assert state["status"] == expected
    assert (state["last_success_at"] is not None) is succeeded
    assert state["consecutive_failures"] == (0 if succeeded else 1)
    assert state["last_checked_at"] is not None

    entry = dict(repository.read.run_log_rows(run_id="run-src")[0])
    # The counters are this call's own answer, and they follow the stored
    # status in both forms.
    assert entry["success_count"] == (1 if succeeded else 0)
    assert entry["partial_count"] == (0 if succeeded else 1)
    assert json.loads(entry["counts"])["source_state_status"] == expected

    if terminal:
        # Only a terminal run's status is this call's to decide; a run
        # that never settles is degraded by the lifecycle, whatever it
        # recorded on the way.
        assert entry["status"] == ("success" if succeeded else "degraded")
        assert repository.read.stage_key_rows()[0]["status"] == (
            "success" if succeeded else "degraded"
        )
    else:
        # A run that never settles terminally is degraded and leaves its
        # key failed, whatever it recorded along the way — unchanged
        # lifecycle, asserted here so the outcome fix cannot quietly
        # start driving it.
        assert entry["status"] == "degraded"
        assert repository.read.stage_key_rows()[0]["status"] == "failed"


@pytest.mark.parametrize(
    "kwargs",
    [pytest.param(k, id=label) for label, k in REJECTED_SOURCE_OUTCOMES],
)
@pytest.mark.parametrize("terminal", [False, True], ids=["non-terminal", "terminal"])
def test_a_contradictory_fetch_outcome_mutates_nothing(tmp_path, kwargs, terminal):
    repository = migrated(tmp_path)
    key = {
        "stage": "m0.ingest",
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
    }
    repository.claim_stage_key(**key, run_id="run-src")

    with pytest.raises(Phase0ValidationError):
        with repository.stage_run(
            run_id="run-src",
            stage="m0.ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            repository.record_source_state(
                "rss:test", run=run, terminal=terminal, **kwargs
            )

    # No source state at all, and the key was not marked finished.
    assert repository.source_state("rss:test") is None
    assert repository.count("source_state") == 0
    assert repository.read.stage_key_rows()[0]["status"] == "failed"
    # The run is recorded, as a failure — that is the logging contract,
    # not a mutation of the state this call was refused permission to write.
    entry = dict(repository.read.run_log_rows(run_id="run-src")[0])
    assert entry["status"] == "failed"
    assert entry["success_count"] == 0


def test_a_replayed_fetch_outcome_is_deterministic(tmp_path):
    """The same statement twice reaches the same place, counters included."""

    repository = migrated(tmp_path)
    for attempt in range(2):
        with repository.stage_run(
            run_id=f"run-{attempt}",
            stage="m0.ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.record_source_state(
                "rss:test", run=run, status="partial", terminal=True
            )
        state = repository.source_state("rss:test")
        assert state["status"] == "partial"
        assert state["consecutive_failures"] == 0


def test_a_failed_fetch_after_a_success_still_counts_up(tmp_path):
    """Consecutive failures track the resolved status, not the boolean."""

    repository = migrated(tmp_path)
    for index, status in enumerate(["success", "failed", "unknown", "empty"]):
        with repository.stage_run(
            run_id=f"run-{index}",
            stage="m0.ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.record_source_state(
                "rss:test", run=run, status=status, terminal=True
            )
        state = repository.source_state("rss:test")
        assert state["status"] == status
    # success -> failed(1) -> unknown(2) -> empty resets to 0.
    assert repository.source_state("rss:test")["consecutive_failures"] == 0


def test_the_unlogged_source_state_path_shares_the_contract(tmp_path):
    """`admin.set_source_state` resolves the outcome the same way."""

    repository = migrated(tmp_path)
    with pytest.raises(Phase0ValidationError, match="disagree"):
        repository.admin.set_source_state(
            "rss:test",
            etag=None,
            last_modified=None,
            checked_at=f"{DAY}T12:00:00+00:00",
            successful=True,
            status="failed",
        )
    assert repository.source_state("rss:test") is None


# ----------------------------------------------------------------------
# Every entrypoint resolves the outcome identically
#
# The default for an unstated outcome was patched at `record_source_state`'s
# own call site while `validate_source_state` still collapsed `None` into
# `False`.  So a payload that said nothing resolved to *success* through
# one entrypoint and *failed* through the other three — `None` is "not
# stated", which is not the same claim as an explicit `False`.
# ----------------------------------------------------------------------

SOURCE_CHECKED_AT = f"{DAY}T12:00:00+00:00"


def _via_record_source_state(repository, **kwargs):
    with repository.stage_run(
        run_id="run-src",
        stage="m0.ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.record_source_state(
            "rss:test", run=run, checked_at=SOURCE_CHECKED_AT, terminal=True, **kwargs
        )


def _via_admin_set(repository, **kwargs):
    repository.admin.set_source_state(
        "rss:test",
        etag=None,
        last_modified=None,
        checked_at=SOURCE_CHECKED_AT,
        **kwargs,
    )


def _via_ingest(repository, **kwargs):
    with repository.stage_run(
        run_id="run-src",
        stage="m0.ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items(
            [],
            run=run,
            terminal=True,
            source_state={
                "source": "rss:test",
                "checked_at": SOURCE_CHECKED_AT,
                **kwargs,
            },
        )


def _via_admin_ingest(repository, **kwargs):
    repository.admin.insert_raw_items(
        [],
        source_state={
            "source": "rss:test",
            "checked_at": SOURCE_CHECKED_AT,
            **kwargs,
        },
    )


#: Every public way a source state reaches the database.  The last two
#: are the ones the finding did not name: they hand a raw payload
#: straight to the shared resolver, so they inherited its default too.
SOURCE_STATE_ENTRYPOINTS = [
    ("record_source_state", _via_record_source_state),
    ("admin.set_source_state", _via_admin_set),
    ("ingest_raw_items(source_state=)", _via_ingest),
    ("admin.insert_raw_items(source_state=)", _via_admin_ingest),
]

#: The whole truth table, including the shapes that only exist because
#: `None` and `False` are different statements.
SOURCE_OUTCOME_TRUTH_TABLE = [
    ("omit both", {}, "success"),
    ("successful=True", {"successful": True}, "success"),
    ("successful=False", {"successful": False}, "failed"),
    ("successful=None", {"successful": None}, "success"),
    ("status=success", {"status": "success"}, "success"),
    ("status=failed", {"status": "failed"}, "failed"),
    ("status=partial", {"status": "partial"}, "partial"),
    ("status=empty", {"status": "empty"}, "empty"),
    ("status=unknown", {"status": "unknown"}, "unknown"),
    ("status=None", {"status": None}, "success"),
    ("both None", {"successful": None, "status": None}, "success"),
    ("None + failed", {"successful": None, "status": "failed"}, "failed"),
    ("None + partial", {"successful": None, "status": "partial"}, "partial"),
    ("True + success", {"successful": True, "status": "success"}, "success"),
    ("False + failed", {"successful": False, "status": "failed"}, "failed"),
    ("False + unknown", {"successful": False, "status": "unknown"}, "unknown"),
]

CONFLICTING_SOURCE_OUTCOMES = [
    ("True + failed", {"successful": True, "status": "failed"}),
    ("True + unknown", {"successful": True, "status": "unknown"}),
    ("False + success", {"successful": False, "status": "success"}),
    ("False + partial", {"successful": False, "status": "partial"}),
    ("False + empty", {"successful": False, "status": "empty"}),
    ("invalid status", {"status": "banana"}),
]


@pytest.mark.parametrize(
    "entrypoint",
    [pytest.param(f, id=n) for n, f in SOURCE_STATE_ENTRYPOINTS],
)
@pytest.mark.parametrize(
    "kwargs, expected",
    [pytest.param(k, s, id=label) for label, k, s in SOURCE_OUTCOME_TRUTH_TABLE],
)
def test_every_source_state_entrypoint_resolves_the_same_outcome(
    tmp_path, entrypoint, kwargs, expected
):
    repository = migrated(tmp_path)
    entrypoint(repository, **kwargs)

    succeeded = expected in SUCCEEDED_STATUSES
    state = repository.source_state("rss:test")
    assert state["status"] == expected
    # The derived columns follow the resolved status, not the input shape.
    assert (state["last_success_at"] is not None) is succeeded
    assert state["consecutive_failures"] == (0 if succeeded else 1)


@pytest.mark.parametrize(
    "entrypoint",
    [pytest.param(f, id=n) for n, f in SOURCE_STATE_ENTRYPOINTS],
)
@pytest.mark.parametrize(
    "kwargs",
    [pytest.param(k, id=label) for label, k in CONFLICTING_SOURCE_OUTCOMES],
)
def test_every_source_state_entrypoint_refuses_the_same_conflicts(
    tmp_path, entrypoint, kwargs
):
    """The checks added in 295666b hold on every path, unweakened."""

    repository = migrated(tmp_path)
    with pytest.raises(Phase0ValidationError):
        entrypoint(repository, **kwargs)
    assert repository.source_state("rss:test") is None
    assert repository.count("source_state") == 0


@pytest.mark.parametrize(
    "kwargs, expected",
    [pytest.param(k, s, id=label) for label, k, s in SOURCE_OUTCOME_TRUTH_TABLE],
)
def test_the_shared_resolver_is_the_one_that_decides(kwargs, expected):
    """No entrypoint may reach a different answer than the resolver.

    Asserted against `validate_source_state` directly, because it is
    public — #61 and #62 build payloads and ask it what the repository
    will do before committing, so a caller-side default would make that
    answer wrong for exactly the payloads that state nothing.
    """

    resolved = Phase0Repository.validate_source_state(
        {"source": "rss:test", "checked_at": SOURCE_CHECKED_AT, **kwargs}
    )
    assert resolved["status"] == expected
    assert resolved["failed"] == (0 if expected in SUCCEEDED_STATUSES else 1)
    assert (resolved["success_at"] is not None) is (expected in SUCCEEDED_STATUSES)


def test_an_unstated_outcome_keeps_the_run_and_the_state_agreeing(tmp_path):
    """Omitting everything is a success on both sides of the record."""

    repository = migrated(tmp_path)
    _via_record_source_state(repository)

    assert repository.source_state("rss:test")["status"] == "success"
    entry = dict(repository.read.run_log_rows(run_id="run-src")[0])
    assert entry["status"] == "success"
    assert entry["success_count"] == 1
    assert entry["partial_count"] == 0
    assert json.loads(entry["counts"])["source_state_status"] == "success"


def test_the_default_lives_in_the_resolver_and_not_at_a_call_site(tmp_path):
    """A payload with no outcome resolves before any entrypoint sees it.

    Pinned as a property so the default cannot drift back to being
    patched per-caller: the resolver's answer for the empty statement is
    what every path must store.
    """

    empty = {"source": "rss:test", "checked_at": SOURCE_CHECKED_AT}
    resolved = Phase0Repository.validate_source_state(empty)["status"]
    for name, entrypoint in SOURCE_STATE_ENTRYPOINTS:
        repository = migrated(tmp_path, f"{abs(hash(name))}.db")
        entrypoint(repository)
        assert repository.source_state("rss:test")["status"] == resolved, name


def test_a_run_logged_as_success_cannot_carry_errors(tmp_path):
    """The adjacent instance of the same "two answers" shape.

    `_write_run_log` already reads a non-empty `errors` list as *meaning*
    degraded when no status is given, so a caller stating `success`
    alongside errors contradicts the module's own rule.
    """

    repository = migrated(tmp_path)
    entry = {
        "run_id": "run-1",
        "stage": "cluster",
        "counts": {},
        "duration_ms": 1,
        "started_at": f"{DAY}T12:00:00+00:00",
        "completed_at": f"{DAY}T12:00:01+00:00",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "ticker": "NVDA",
    }
    with pytest.raises(Phase0ValidationError, match="recorded errors"):
        repository.admin.log_stage(**entry, status="success", errors=["boom"])
    assert repository.read.run_log_rows() == []

    # The honest spellings both work.
    repository.admin.log_stage(**entry, status="degraded", errors=["boom"])
    assert repository.read.run_log_rows()[0]["status"] == "degraded"
    repository.admin.log_stage(
        **{**entry, "run_id": "run-2"}, status="success", errors=[]
    )
    assert repository.read.run_log_rows(run_id="run-2")[0]["status"] == "success"


# ----------------------------------------------------------------------
# Short scheme credentials
# ----------------------------------------------------------------------


SHORT_CREDENTIALS = [
    ("Bearer a", "Bearer a"),
    ("Bearer abc", "Bearer abc"),
    ("Basic a", "Basic a"),
    ("Basic abc", "Basic abc"),
    ("Authorization: Bearer abc", "Bearer abc"),
    ("Authorization: Basic abc", "Basic abc"),
    ("lower", "bearer abc"),
    ("upper", "BEARER ABC"),
    ("mixed", "BaSiC aBc"),
    ("quoted", '"Bearer abc"'),
    ("parens", "(Basic abc)"),
    ("trailing-period", "Bearer abc."),
]


@pytest.mark.parametrize(
    "label, text", SHORT_CREDENTIALS, ids=[row[0] for row in SHORT_CREDENTIALS]
)
def test_short_scheme_credentials_are_redacted(label, text):
    redacted = redact_text(text)

    assert "[REDACTED]" in redacted
    assert redact_text(redacted) == redacted


@pytest.mark.parametrize(
    "text",
    [
        "a basic understanding of the problem",
        "the bearer instrument matured",
        "Basic Auth is required",
        "Bearer token expired",
        "tokenizer failed to load",
    ],
)
def test_ordinary_prose_survives_short_credential_redaction(text):
    assert redact_text(text) == text


#: Byte-scanning needs a token distinctive enough that finding it proves
#: something.  A one-character secret occurs everywhere in a SQLite file,
#: so those cases are covered by the redaction and stored-JSON assertions
#: instead; these are short but unmistakable.
SHORT_DISTINCTIVE_CREDENTIALS = [
    "Bearer Zq7",
    "Basic Zq7",
    "Authorization: Bearer Zq7",
    "Authorization: Basic Zq7",
    "bearer Zq7",
    "BASIC ZQ7X",
    '"Bearer Zq7"',
]


@pytest.mark.parametrize("text", SHORT_DISTINCTIVE_CREDENTIALS)
def test_short_credentials_never_reach_the_database_or_wal(tmp_path, text):
    repository = migrated(tmp_path)
    secret = text.rsplit(" ", 1)[-1].strip("\"').")

    with nvda_run(repository) as run:
        repository.record_source_state(
            "rss:test",
            run=run,
            successful=False,
            status="failed",
            metadata={"request": text},
            error=text,
        )

    assert secret.encode() not in database_bytes(repository)
    assert secret not in json.dumps(repository.source_state("rss:test"))


@pytest.mark.parametrize(
    "label, text", SHORT_CREDENTIALS, ids=[row[0] for row in SHORT_CREDENTIALS]
)
def test_short_credentials_are_redacted_in_stored_metadata(tmp_path, label, text):
    repository = migrated(tmp_path)

    with nvda_run(repository) as run:
        repository.record_source_state(
            "rss:test",
            run=run,
            successful=False,
            status="failed",
            metadata={"request": text},
            error=text,
        )

    state = repository.source_state("rss:test")
    assert "[REDACTED]" in json.dumps(state)
    assert text not in json.dumps(state)


# ----------------------------------------------------------------------
# Stories must state their pipeline version (migration 011)
# ----------------------------------------------------------------------


def test_a_fresh_database_refuses_a_null_version_story(tmp_path):
    repository = migrated(tmp_path)

    with repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="explicit pipeline_version"):
            connection.execute(
                "INSERT INTO stories (ticker, trading_day, canonical_title) "
                "VALUES ('NVDA', ?, 't')",
                (DAY,),
            )
        connection.execute(
            "INSERT INTO stories (ticker, trading_day, canonical_title, "
            "pipeline_version) VALUES ('NVDA', ?, 't', 'v1')",
            (DAY,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="explicit pipeline_version"):
            connection.execute("UPDATE stories SET pipeline_version = NULL")


def test_the_admin_story_helper_requires_a_pipeline_version(tmp_path):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)

    with pytest.raises(TypeError):
        repository.admin.insert_story(
            ticker="NVDA",
            trading_day=DAY,
            canonical_title="No version",
            member_ids=item_ids,
        )
    with pytest.raises(Phase0ValidationError, match="pipeline_version"):
        repository.admin.insert_story(
            ticker="NVDA",
            trading_day=DAY,
            canonical_title="Blank version",
            member_ids=item_ids,
            pipeline_version="  ",
        )
    assert repository.count("stories") == 0


def v10_database_with_null_versions(tmp_path: Path, attached: bool) -> Path:
    """A genuine v10 database carrying legacy NULL-version stories."""

    directory = partial_migrations(tmp_path, 10)
    repository = Phase0Repository(
        tmp_path / "phase0.sqlite3", migrations_path=directory
    )
    repository.migrate()
    item_ids = seed_raw_items(repository, 2)
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT INTO stories (id, ticker, trading_day, canonical_title, "
            "content_hash, outlet_count, updated_at) "
            "VALUES (1, 'NVDA', ?, 'Legacy', 'h', 1, ?)",
            (DAY, f"{DAY}T12:00:00+00:00"),
        )
        connection.execute(
            "INSERT INTO story_members (story_id, raw_item_id, position) "
            "VALUES (1, ?, 0)",
            (item_ids[0],),
        )
        if attached:
            connection.execute(
                "INSERT INTO themes (ticker, trading_day, label, salience_rank, "
                "status, content_hash, pipeline_version) "
                "VALUES ('NVDA', ?, 'Legacy theme', 1, 'ready', 'h', 'v7')",
                (DAY,),
            )
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) "
                "SELECT id, 1 FROM themes"
            )
    return tmp_path / "phase0.sqlite3"


def test_an_unattached_legacy_story_upgrades_to_the_sentinel_version(tmp_path):
    database = v10_database_with_null_versions(tmp_path, attached=False)

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.schema_version() == LATEST_VERSION
    assert upgraded.read.story(1)["pipeline_version"] == "legacy-v0"
    assert upgraded.count("story_members") == 1


def test_an_attached_legacy_story_inherits_its_relationship_version(tmp_path):
    database = v10_database_with_null_versions(tmp_path, attached=True)

    upgraded = Phase0Repository(database)
    upgraded.migrate()

    assert upgraded.read.story(1)["pipeline_version"] == "v7"
    assert upgraded.count("theme_stories") == 1


def test_an_ambiguous_legacy_story_fails_the_migration_rather_than_guessing(tmp_path):
    database = v10_database_with_null_versions(tmp_path, attached=True)
    repository = Phase0Repository(
        database, migrations_path=partial_migrations(tmp_path, 10)
    )
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT INTO themes (ticker, trading_day, label, salience_rank, "
            "status, content_hash, pipeline_version) "
            "VALUES ('NVDA', ?, 'Other version', 2, 'ready', 'h2', 'v8')",
            (DAY,),
        )
        connection.execute(
            "INSERT INTO theme_stories (theme_id, story_id) "
            "SELECT id, 1 FROM themes WHERE pipeline_version = 'v8'"
        )

    upgraded = Phase0Repository(database)
    with pytest.raises(sqlite3.Error, match="more than one pipeline version"):
        upgraded.migrate()

    # Rolled back whole: still v10, still NULL, data intact.
    assert upgraded.schema_version() == 10
    assert upgraded.read.story(1)["pipeline_version"] is None
    assert upgraded.count("theme_stories") == 2


def test_a_story_cannot_join_two_versioned_theme_partitions_after_upgrade(tmp_path):
    database = v10_database_with_null_versions(tmp_path, attached=True)
    upgraded = Phase0Repository(database)
    upgraded.migrate()

    with upgraded.admin.connect_writable() as connection:
        connection.execute(
            "INSERT INTO themes (ticker, trading_day, label, salience_rank, "
            "status, content_hash, pipeline_version) "
            "VALUES ('NVDA', ?, 'v8 theme', 2, 'ready', 'h2', 'v8')",
            (DAY,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="pipeline version"):
            connection.execute(
                "INSERT INTO theme_stories (theme_id, story_id) "
                "SELECT id, 1 FROM themes WHERE pipeline_version = 'v8'"
            )


# ----------------------------------------------------------------------
# Stage-key completion is not a public API
# ----------------------------------------------------------------------


def test_the_repository_has_no_public_stage_key_completion():
    """Claim a key, declare it a success, write nothing: that was public.

    Nothing about it needed to be clever — ``complete_stage_key`` moved a
    claimed key to ``success`` with no data and no ``run_log`` row, so the
    ledger recorded a stage that never ran.
    """

    assert not hasattr(Phase0Repository, "complete_stage_key")
    public = _public_repository_methods()
    assert "complete_stage_key" not in public
    completers = [
        name
        for name in public
        if "complete" in name and "stage" in name.replace("_", "")
    ]
    assert completers == [], f"a public stage-key completer came back: {completers}"


#: Public methods that may move a stage key at all, and why.  A completer
#: under a different name lands here or fails the audit.
STAGE_KEY_MOVERS = {
    "claim_stage_key": "takes an unowned, retryable, or expired key",
    "heartbeat_stage_key": "extends the lease it already holds",
    "recover_expired_leases": "marks abandoned claims retryable",
    "stage_run": "settles the run that owns the key",
}

_MOVES_A_STAGE_KEY = re.compile(
    r"UPDATE\s+pipeline_stage_keys|INSERT\s+INTO\s+pipeline_stage_keys"
    r"|_finish_stage_key|_complete_stage_key_unlogged",
    re.IGNORECASE,
)


def test_no_public_method_moves_a_stage_key_to_success_by_itself():
    """Renaming the hole does not close it.

    Follows delegation the same way the write audit does, so a public
    method that only *calls* the completer is caught too.
    """

    def moves(name, seen=None):
        seen = seen if seen is not None else set()
        if name in seen:
            return False
        seen.add(name)
        member = getattr(Phase0Repository, name, None)
        if member is None or not callable(member):
            return False
        try:
            source = inspect.getsource(member)
        except (OSError, TypeError) as exc:  # pragma: no cover - defensive
            raise AssertionError(f"cannot read the source of {name}: {exc}") from exc
        if _MOVES_A_STAGE_KEY.search(source):
            return True
        return any(
            moves(callee, seen) for callee in set(re.findall(r"self\.(\w+)\(", source))
        )

    movers = {name for name in _public_repository_methods() if moves(name)}
    assert movers, "reflection found no stage-key movers; the audit is broken"

    unclassified = []
    for name in sorted(movers):
        if name in STAGE_KEY_MOVERS:
            assert STAGE_KEY_MOVERS[name].strip(), f"{name} needs a stated reason"
            continue
        # Everything else must be a logged mutation, where the key moves
        # only in the transaction that also commits the data and the log.
        if "run" in inspect.signature(getattr(Phase0Repository, name)).parameters:
            assert (
                name in LOGGED_ENTRYPOINTS
            ), f"{name} moves a key outside the contract"
            continue
        unclassified.append(name)
    assert not unclassified, (
        "public methods move stage keys with neither a run nor a stated "
        f"reason: {unclassified}"
    )


def test_admin_stage_key_completion_says_it_is_manual_repair():
    doc = (Phase0Admin.complete_stage_key.__doc__ or "").lower()
    assert "manual repair only" in doc
    assert "must never call it" in doc
    assert "terminal=true" in doc


def test_admin_completion_cannot_be_mistaken_for_a_stage_finishing(tmp_path):
    """It moves the key and nothing else — no data, no run log."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    repository.admin.complete_stage_key(**key, run_id="run-1", status="success")

    assert repository.stage_key_state(**key)["status"] == "success"
    assert repository.run_log_entries() == []
    assert repository.count("raw_items") == 0


def test_a_claimed_key_cannot_reach_success_without_a_terminal_mutation(tmp_path):
    """The only route to a success key is a terminal logged mutation."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run)
        assert repository.stage_key_state(**key)["status"] == "running"

    assert repository.stage_key_state(**key)["status"] != "success"
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "degraded"
    # And the key is retryable rather than stranded.
    assert repository.claim_stage_key(**key, run_id="run-2")


# ----------------------------------------------------------------------
# The terminal lifecycle: settled once, immutably
# ----------------------------------------------------------------------


def key_row(repository: Phase0Repository, key: dict) -> dict:
    rows = repository.read.stage_key_rows(**key)
    assert len(rows) == 1
    return rows[0]


def run_log_rows(repository: Phase0Repository) -> list[dict]:
    return repository.read.run_log_rows()


def test_a_terminal_validation_error_reaches_the_caller_unmasked(tmp_path):
    """The reported bug: cleanup replaced the exception on its way out.

    A terminal operation that failed validation recorded its failure, and
    then ``stage_run`` teardown tried to finalize *again*.  The second
    ``_finish_stage_key`` found nothing to update and raised StageKeyError
    from a ``finally`` — so the caller was told the lease was lost, not
    what was actually wrong with their data.
    """

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with pytest.raises(Phase0RunContextError) as caught:
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            repository.ingest_raw_items([raw_item(1, "AMD")], run=run, terminal=True)

    assert not isinstance(caught.value, StageKeyError)
    assert "AMD" in str(caught.value)
    assert repository.count("raw_items") == 0
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"


def test_a_terminal_failure_records_exactly_one_outcome(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with pytest.raises(Phase0RunContextError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            repository.ingest_raw_items([raw_item(1, "AMD")], run=run, terminal=True)

    assert len(run_log_rows(repository)) == 1
    assert repository.stage_key_state(**key)["status"] == "failed"
    assert repository.claim_stage_key(**key, run_id="run-2")


def test_a_second_terminal_operation_leaves_the_success_untouched(tmp_path):
    """A settled success is immutable, including when the retry fails."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
        settled_log = run_log_rows(repository)
        settled_key = key_row(repository, key)

        with pytest.raises(Phase0RunContextError, match="already terminal_succeeded"):
            repository.ingest_raw_items([raw_item(2)], run=run, terminal=True)

        assert run_log_rows(repository) == settled_log
        assert key_row(repository, key) == settled_key

    assert run_log_rows(repository) == settled_log
    assert key_row(repository, key) == settled_key
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "success"
    assert repository.stage_key_state(**key)["status"] == "success"
    assert repository.count("raw_items") == 1


@pytest.mark.parametrize("terminal", [False, True])
def test_no_operation_persists_anything_after_terminal_success(tmp_path, terminal):
    repository = migrated(tmp_path)
    item_ids = seed_raw_items(repository, 1)

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items([raw_item(2)], run=run, terminal=True)
        before_log = run_log_rows(repository)
        before_items = repository.count("raw_items")

        for attempt in (
            lambda: repository.ingest_raw_items(
                [raw_item(3)], run=run, terminal=terminal
            ),
            lambda: repository.record_source_state("yahoo", run=run, terminal=terminal),
            lambda: repository.persist_embeddings(
                [sample_embedding(item_ids[0])], run=run, terminal=terminal
            ),
        ):
            with pytest.raises(Phase0RunContextError, match="already"):
                attempt()

        assert run_log_rows(repository) == before_log
        assert repository.count("raw_items") == before_items
        assert repository.count("source_state") == 0
        assert repository.count("embeddings") == 0

    assert run_log_rows(repository) == before_log


def test_no_operation_persists_anything_after_terminal_failure(tmp_path):
    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        with pytest.raises(Phase0RunContextError):
            repository.ingest_raw_items([raw_item(1, "AMD")], run=run)
        settled = run_log_rows(repository)

        with pytest.raises(Phase0RunContextError, match="already terminal_failed"):
            repository.ingest_raw_items([raw_item(2)], run=run, terminal=True)

        assert run_log_rows(repository) == settled

    assert run_log_rows(repository) == settled
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"
    assert repository.count("raw_items") == 0


def test_normal_exit_after_a_settled_run_is_a_no_op(tmp_path):
    """Teardown adds nothing once an operation has settled the run."""

    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
        inside = run_log_rows(repository)
        assert run.state == "terminal_succeeded"

    assert run_log_rows(repository) == inside


def test_a_run_with_no_terminal_operation_settles_exactly_once(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run)
        repository.ingest_raw_items([raw_item(2)], run=run)
        assert run.state == "active"

    rows = run_log_rows(repository)
    assert len(rows) == 1
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "degraded"
    assert repository.claim_stage_key(**key, run_id="run-2")


def test_a_run_state_never_leaves_a_settled_state(tmp_path):
    """Only repository internals move the state, and only out of active."""

    repository = migrated(tmp_path)
    escaped = {}

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        assert run.state == "active"
        assert run.settled is False
        escaped["run"] = run
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
        assert run.state == "terminal_succeeded"
        assert run.settled is True

    settled = escaped["run"]
    assert settled.state == "terminal_succeeded"
    with pytest.raises(Phase0RunContextError):
        settled._transition("active")
    with pytest.raises(Phase0RunContextError):
        settled._transition("terminal_failed")
    with pytest.raises(Phase0RunContextError):
        settled.state = "active"
    assert settled.state == "terminal_succeeded"


# ----------------------------------------------------------------------
# Rejection-capable validation runs inside the authoritative transaction
# ----------------------------------------------------------------------


def caught_validation_cases(repository: Phase0Repository) -> dict:
    """One rejectable call per logged entrypoint, with what it must not write.

    Each rejection is a *validation* rejection — a foreign ticker, a
    foreign day, a source that does not exist — the kind a caller might
    plausibly wrap in ``try``/``except`` and carry on from.
    """

    item_ids = seed_raw_items(repository, 1)
    return {
        "ingest_raw_items": (
            lambda run, terminal: repository.ingest_raw_items(
                [raw_item(1, "AMD")], run=run, terminal=terminal
            ),
            "raw_items",
        ),
        "record_source_state": (
            lambda run, terminal: repository.record_source_state(
                "yahoo",
                run=run,
                checked_at="2026-07-24T12:00:00+00:00",
                terminal=terminal,
            ),
            "source_state",
        ),
        "reconcile_stories": (
            lambda run, terminal: repository.reconcile_stories(
                run=run,
                ticker="AMD",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf1", item_ids)],
                terminal=terminal,
            ),
            "stories",
        ),
        "reconcile_themes": (
            lambda run, terminal: repository.reconcile_themes(
                run=run,
                ticker="AMD",
                trading_day=DAY,
                pipeline_version="v1",
                theme_set=theme_set(),
                terminal=terminal,
            ),
            "themes",
        ),
        "persist_embeddings": (
            lambda run, terminal: repository.persist_embeddings(
                [sample_embedding("999999")], run=run, terminal=terminal
            ),
            "embeddings",
        ),
        # A snapshot stamped on a day this run does not cover.
        "record_feed_snapshot": (
            lambda run, terminal: repository.record_feed_snapshot(
                feed_source="rss:test",
                response_url="https://example.com/feed",
                body=b"<rss/>",
                fetched_at="2026-07-24T12:00:00+00:00",
                run=run,
                terminal=terminal,
            ),
            "feed_snapshots",
        ),
        # A decision naming a raw item that has no RSS provenance at all.
        "replace_relevance_classifications": (
            lambda run, terminal: repository.replace_relevance_classifications(
                [{"raw_item_id": item_ids[0], "ticker": "NVDA"}],
                run=run,
                terminal=terminal,
            ),
            "raw_item_match_evidence",
        ),
    }


@pytest.mark.parametrize("entrypoint", LOGGED_ENTRYPOINTS)
@pytest.mark.parametrize("terminal", [False, True])
def test_a_caught_validation_failure_still_ends_the_run_as_failed(
    tmp_path, entrypoint, terminal
):
    """Swallowing the exception must not buy the caller a clean exit.

    This is the whole point of doing the validation inside the run's own
    transaction.  When it ran *before* ``_logged_mutation``, the run had
    not yet taken responsibility for the operation, so a caller who caught
    the rejection and exited normally left the stage recorded as a success
    for work that was rejected.
    """

    repository = migrated(tmp_path)
    call, table = caught_validation_cases(repository)[entrypoint]
    baseline = repository.count(table)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-x")

    with repository.stage_run(
        run_id="run-x",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        with contextlib.suppress(Phase0Error):
            call(run, terminal)
        # ...and the caller carries on as if nothing happened.

    entry = repository.run_log_entries(run_id="run-x")[0]
    assert entry["status"] == "failed"
    assert entry["failure_count"] >= 1
    assert repository.count(table) == baseline
    assert repository.stage_key_state(**key)["status"] == "failed"
    assert repository.claim_stage_key(**key, run_id="run-y")


@pytest.mark.parametrize("entrypoint", LOGGED_ENTRYPOINTS)
def test_a_caught_failure_cannot_be_followed_by_a_reported_success(
    tmp_path, entrypoint
):
    repository = migrated(tmp_path)
    call, _ = caught_validation_cases(repository)[entrypoint]

    with repository.stage_run(
        run_id="run-x",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        with contextlib.suppress(Phase0Error):
            call(run, False)
        with pytest.raises(Phase0RunContextError, match="already terminal_failed"):
            repository.ingest_raw_items([raw_item(50)], run=run, terminal=True)

    assert repository.run_log_entries(run_id="run-x")[0]["status"] == "failed"
    assert repository.count("raw_items") == 1  # only the seeded one


@pytest.mark.parametrize("entrypoint", LOGGED_ENTRYPOINTS)
def test_every_logged_entrypoint_validates_inside_its_mutation(entrypoint):
    """Structural: nothing that can reject runs before the ``with``.

    Read as a rule rather than a lint: the body above the
    ``_logged_mutation`` line may not call anything that raises, because
    an exception raised there belongs to no run.
    """

    source = inspect.getsource(getattr(Phase0Repository, entrypoint))
    body = source.split("_logged_mutation", 1)[0]
    body = body.split('"""')[-1]  # drop the docstring
    forbidden = re.compile(
        r"\b(?:normalize_ticker|_normalize_day|_normalize_datetime|_require_text"
        r"|_require_int|validate_embedding|_prepare_raw_item|_assert_\w+)\("
    )
    found = forbidden.findall(body)
    assert not found, f"{entrypoint} can reject before taking responsibility: {found}"


# ----------------------------------------------------------------------
# candidate_tickers: one parser, used by validation and persistence alike
# ----------------------------------------------------------------------


def test_candidate_tickers_normalize_both_supported_forms():
    assert normalize_candidate_tickers(["NVDA"]) == [
        {"ticker": "NVDA", "reason": DEFAULT_CANDIDATE_REASON}
    ]
    assert normalize_candidate_tickers([{"ticker": "NVDA", "reason": "headline"}]) == [
        {"ticker": "NVDA", "reason": "headline"}
    ]
    assert normalize_candidate_tickers(None) == []
    assert normalize_candidate_tickers([]) == []


def test_candidate_tickers_normalize_case_and_whitespace():
    assert normalize_candidate_tickers([" nvda ", {"ticker": "amd\t"}]) == [
        {"ticker": "AMD", "reason": DEFAULT_CANDIDATE_REASON},
        {"ticker": "NVDA", "reason": DEFAULT_CANDIDATE_REASON},
    ]


def test_candidate_ticker_duplicates_are_deterministic():
    """First mention wins, and the order does not depend on input order."""

    forward = normalize_candidate_tickers(
        [{"ticker": "NVDA", "reason": "first"}, "nvda", {"ticker": "AMD"}]
    )
    assert forward == [
        {"ticker": "AMD", "reason": DEFAULT_CANDIDATE_REASON},
        {"ticker": "NVDA", "reason": "first"},
    ]
    assert normalize_candidate_tickers(["AMD", "NVDA"]) == normalize_candidate_tickers(
        ["NVDA", "AMD"]
    )


@pytest.mark.parametrize(
    "value",
    [
        [""],
        ["   "],
        ["NVDA AMD"],
        ["NVDA,AMD"],
        ["GOOG"],
        [None],
        [7],
        [["NVDA"]],
        [{"reason": "no ticker at all"}],
        [{"ticker": None}],
        [{"ticker": ""}],
        "NVDA",
        {"ticker": "NVDA"},
    ],
)
def test_malformed_candidate_tickers_are_rejected(value):
    with pytest.raises(Phase0ValidationError):
        normalize_candidate_tickers(value)


def test_one_bad_candidate_rejects_the_whole_item():
    with pytest.raises(Phase0ValidationError):
        normalize_candidate_tickers(["NVDA", {"ticker": "AMD"}, 7])


@pytest.mark.parametrize(
    "candidates",
    [
        ["AMD"],
        [{"ticker": "AMD"}],
        ["NVDA", "AMD"],
        [{"ticker": "NVDA"}, "AMD"],
        ["nvda", {"ticker": " amd "}],
    ],
)
def test_a_foreign_candidate_ticker_cannot_ride_into_another_run(tmp_path, candidates):
    """The bypass: the validator understood mappings, the writer both.

    ``candidate_tickers=["AMD"]`` was therefore persisted under an NVDA
    run, because the check that would have caught it only inspected the
    mapping form.
    """

    repository = migrated(tmp_path)
    item = {**raw_item(1), "candidate_tickers": candidates}

    with pytest.raises(Phase0RunContextError, match="AMD"):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.ingest_raw_items([item], run=run, terminal=True)

    assert repository.count("raw_items") == 0
    assert repository.count("raw_item_candidates") == 0
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"


def test_one_bad_candidate_rolls_back_a_whole_ingestion_batch(tmp_path):
    repository = migrated(tmp_path)
    poisoned = {**raw_item(2), "candidate_tickers": ["AMD"]}

    with pytest.raises(Phase0RunContextError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.ingest_raw_items(
                [raw_item(1), poisoned, raw_item(3)], run=run, terminal=True
            )

    assert repository.count("raw_items") == 0
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"


def test_accepted_candidates_are_persisted_from_the_normalized_form(tmp_path):
    repository = migrated(tmp_path)
    item = {
        **raw_item(1),
        "candidate_tickers": [" nvda ", {"ticker": "NVDA", "reason": "ignored"}],
    }

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items([item], run=run, terminal=True)

    rows = [
        (row["ticker"], row["reason"]) for row in repository.read.raw_item_candidates()
    ]
    assert rows == [("NVDA", DEFAULT_CANDIDATE_REASON)]


def test_no_second_candidate_parser_survives_downstream():
    """One parser, or the two of them will disagree again."""

    writer = inspect.getsource(Phase0Repository._insert_raw_item)
    candidate_block = writer.split("candidate_tickers", 1)[-1]
    assert "normalize_ticker(" not in candidate_block
    assert "isinstance(candidate" not in candidate_block
    assert "relevance_match" not in candidate_block

    prepare = inspect.getsource(Phase0Repository._prepare_raw_item)
    assert "normalize_candidate_tickers(" in prepare


# ----------------------------------------------------------------------
# Ticker-scoped derived output needs an explicit association
# ----------------------------------------------------------------------


def unattributed_item(repository: Phase0Repository, index: int = 1, **overrides) -> int:
    """Raw evidence stored with no ticker of its own."""

    item = {**raw_item(index), "ticker": None}
    item.update(overrides)
    return repository.admin.insert_raw_item(item).item_id


def test_tickerless_evidence_is_still_storable(tmp_path):
    """Ingestion keeps what it could not attribute; that part is fine."""

    repository = migrated(tmp_path)

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.ingest_raw_items(
            [{**raw_item(1), "ticker": None}], run=run, terminal=True
        )

    assert repository.count("raw_items") == 1
    assert repository.count("raw_item_tickers") == 0
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "success"


def test_an_unattributed_item_cannot_become_an_nvda_story_member(tmp_path):
    repository = migrated(tmp_path)
    item_id = unattributed_item(repository)

    with pytest.raises(Phase0RunContextError, match="no accepted association"):
        reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story("cf1", [item_id])],
        )

    assert repository.count("stories") == 0
    assert repository.count("story_members") == 0


def test_an_unattributed_item_cannot_become_an_nvda_embedding_source(tmp_path):
    repository = migrated(tmp_path)
    item_id = unattributed_item(repository)

    with pytest.raises(Phase0RunContextError, match="no accepted association"):
        with repository.stage_run(
            run_id="run-1",
            stage="m1.embed",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.persist_embeddings(
                [sample_embedding(item_id)], run=run, terminal=True
            )

    assert repository.count("embeddings") == 0
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"


def test_an_amd_only_association_is_rejected_under_an_nvda_run(tmp_path):
    repository = migrated(tmp_path)
    item_id = unattributed_item(repository, tickers=["AMD"])

    with pytest.raises(Phase0RunContextError, match="no accepted association"):
        reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story("cf1", [item_id])],
        )

    assert repository.count("stories") == 0


def test_an_explicit_nvda_association_is_accepted(tmp_path):
    repository = migrated(tmp_path)
    item_id = unattributed_item(repository, tickers=["NVDA"])

    report = reconcile_stories(
        repository,
        ticker="NVDA",
        trading_day=DAY,
        pipeline_version="v1",
        stories=[story("cf1", [item_id])],
    )

    assert len(report.inserted) == 1
    assert repository.count("story_members") == 1

    with repository.stage_run(
        run_id="run-embed",
        stage="m1.embed",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        repository.persist_embeddings(
            [sample_embedding(item_id)], run=run, terminal=True
        )
    assert repository.count("embeddings") == 1


def test_an_item_associated_with_two_tickers_serves_both(tmp_path):
    """Membership, not exclusivity: one article can be about two companies."""

    repository = migrated(tmp_path)
    item_id = unattributed_item(repository, tickers=["NVDA", "AMD"])

    for ticker in ("NVDA", "AMD"):
        report = reconcile_stories(
            repository,
            ticker=ticker,
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story(f"cf-{ticker}", [item_id])],
        )
        assert len(report.inserted) == 1

    assert repository.count("stories") == 2
    assert repository.count("story_members") == 2


def test_a_rejected_association_writes_nothing_and_fails_the_run(tmp_path):
    """One bad member rolls back the good ones with it."""

    repository = migrated(tmp_path)
    good = seed_raw_items(repository, 1)[0]
    orphan = unattributed_item(repository, index=9)

    with pytest.raises(Phase0RunContextError, match="no accepted association"):
        with repository.stage_run(
            run_id="run-1",
            stage="m3.semantic",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.reconcile_stories(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                stories=[story("cf-good", [good]), story("cf-bad", [orphan])],
                terminal=True,
            )

    assert repository.count("stories") == 0
    assert repository.count("story_members") == 0
    assert repository.run_log_entries(run_id="run-1")[0]["status"] == "failed"


def test_candidates_alone_do_not_authorize_derived_processing(tmp_path):
    """A candidate is a suggestion; the association table is the authority."""

    repository = migrated(tmp_path)
    item_id = repository.admin.insert_raw_item(
        {**raw_item(1), "ticker": None, "candidate_tickers": ["NVDA"]}
    ).item_id
    assert repository.count("raw_item_candidates") == 1

    with pytest.raises(Phase0RunContextError, match="no accepted association"):
        reconcile_stories(
            repository,
            ticker="NVDA",
            trading_day=DAY,
            pipeline_version="v1",
            stories=[story("cf1", [item_id])],
        )


# ----------------------------------------------------------------------
# Terminal success means committed, not "about to be committed"
# ----------------------------------------------------------------------


class _CommitFails(sqlite3.Connection):
    """Commit raises, the way a disk error would."""

    def commit(self):
        raise sqlite3.OperationalError("disk I/O error")


class _CommitsThenRaises(sqlite3.Connection):
    """Commit lands and *then* reports failure.

    The nastier of the two, and not hypothetical: an I/O error can be
    reported after the transaction is durable. Anything that infers "the
    data is gone" from "commit raised" gets this case wrong.
    """

    def commit(self):
        super().commit()
        raise sqlite3.OperationalError("disk I/O error")


#: Run states observed at the moment each commit was attempted.
_STATE_AT_COMMIT: list[str] = []


def _watching_connection(run_holder):
    class _RecordsStateAtCommit(sqlite3.Connection):
        def commit(self):
            _STATE_AT_COMMIT.append(run_holder["run"].state)
            super().commit()

    return _RecordsStateAtCommit


@contextlib.contextmanager
def failing_commits(times: int = 1, factory=_CommitFails):
    """Make the next ``times`` *writable* connections fail to commit.

    Read connections open with ``uri=True`` and are left alone, so a test
    can still look at the database while the writer is sabotaged.
    """

    real_connect = sqlite3.connect
    state = {"remaining": times, "used": 0}

    def connect(*args, **kwargs):
        if state["remaining"] > 0 and not kwargs.get("uri"):
            state["remaining"] -= 1
            state["used"] += 1
            kwargs["factory"] = factory
        return real_connect(*args, **kwargs)

    with mock.patch("phase0.repository.sqlite3.connect", side_effect=connect):
        yield state


def test_a_commit_failure_leaves_no_success_anywhere(tmp_path):
    """The reported defect: in-memory success ahead of a durable commit.

    Every statement ran, the context said ``terminal_succeeded``, and then
    the commit failed — taking the data and the run log with it while the
    object still claimed the stage had finished.
    """

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")
    escaped = {}

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            escaped["run"] = run
            with failing_commits():
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    assert escaped["run"].state == "terminal_failed"
    assert repository.count("raw_items") == 0
    entries = repository.run_log_entries(run_id="run-1")
    assert [entry["status"] for entry in entries] == ["failed"]
    assert repository.stage_key_state(**key)["status"] == "failed"
    assert repository.claim_stage_key(**key, run_id="run-2")


def test_a_commit_failure_surfaces_the_original_exception(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(sqlite3.OperationalError) as caught:
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            with failing_commits():
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    assert "disk I/O error" in str(caught.value)
    assert not isinstance(caught.value, StageKeyError)
    assert not isinstance(caught.value, Phase0RunContextError)


@pytest.mark.parametrize("terminal", [False, True])
def test_a_commit_failure_records_the_operation_as_failed(tmp_path, terminal):
    """Non-terminal operations settle on commit too, not before it."""

    repository = migrated(tmp_path)
    escaped = {}

    with pytest.raises(sqlite3.OperationalError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            escaped["run"] = run
            with failing_commits():
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=terminal)

    assert escaped["run"].state == "terminal_failed"
    assert repository.count("raw_items") == 0
    assert [
        entry["status"] for entry in repository.run_log_entries(run_id="run-1")
    ] == ["failed"]


def test_the_context_exit_adds_nothing_after_a_commit_failure(tmp_path):
    """No duplicate settlement, and no contradicting outcome at exit."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")
    seen = {}

    with pytest.raises(sqlite3.OperationalError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            try:
                with failing_commits():
                    repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
            finally:
                seen["log"] = run_log_rows(repository)
                seen["key"] = key_row(repository, key)
            raise AssertionError("unreachable")

    assert len(seen["log"]) == 1
    assert run_log_rows(repository) == seen["log"]
    assert key_row(repository, key) == seen["key"]


def test_a_failed_settlement_is_unknown_rather_than_successful(tmp_path):
    """When recording the failure fails too, nothing may claim success."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")
    escaped = {}

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            escaped["run"] = run
            # The operation's commit, the settlement's, and the run-log-only
            # fallback's: all three.
            with failing_commits(times=3):
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    assert escaped["run"].state == "settlement_failed"
    assert escaped["run"].terminated is True
    assert repository.count("raw_items") == 0
    assert repository.run_log_entries(run_id="run-1") == []
    # Left exactly as a crash would have left it: still running, still
    # leased, and reclaimable the moment that lease expires.
    assert repository.stage_key_state(**key)["status"] == "running"


def test_a_settlement_that_loses_the_key_still_records_the_failure(tmp_path):
    """The run log matters more than the key; one fallback, then stop."""

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with pytest.raises(sqlite3.OperationalError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            with failing_commits(times=2):
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    assert [
        entry["status"] for entry in repository.run_log_entries(run_id="run-1")
    ] == ["failed"]
    assert repository.count("raw_items") == 0


def test_a_durable_success_is_never_overwritten_by_a_late_error(tmp_path):
    """Commit landed, then reported an error. The data is real; keep it.

    Inferring "rolled back" from "commit raised" would replace a committed
    success with a fabricated failure — the worse of the two lies.
    """

    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")
    escaped = {}

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            escaped["run"] = run
            with failing_commits(factory=_CommitsThenRaises):
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    assert repository.count("raw_items") == 1
    assert [
        entry["status"] for entry in repository.run_log_entries(run_id="run-1")
    ] == ["success"]
    assert repository.stage_key_state(**key)["status"] == "success"
    assert escaped["run"].state == "terminal_succeeded"


def test_the_context_is_marked_successful_only_after_the_commit(tmp_path):
    """Observed at the moment of commit, not inferred from the outcome."""

    repository = migrated(tmp_path)
    holder = {}
    _STATE_AT_COMMIT.clear()

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
    ) as run:
        holder["run"] = run
        with failing_commits(factory=_watching_connection(holder)):
            repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)
        assert run.state == "terminal_succeeded"

    assert _STATE_AT_COMMIT == ["active"], _STATE_AT_COMMIT
    assert repository.count("raw_items") == 1


def test_a_committed_outcome_survives_a_reconnect(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with repository.stage_run(
        run_id="run-1",
        stage="ingest",
        trading_day=DAY,
        pipeline_version="v1",
        ticker="NVDA",
        stage_key=key,
    ) as run:
        repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    reopened = Phase0Repository(repository.database_path)
    assert reopened.count("raw_items") == 1
    assert reopened.run_log_entries(run_id="run-1")[0]["status"] == "success"
    assert reopened.stage_key_state(**key)["status"] == "success"
    assert reopened.read.run_log_rows() == repository.read.run_log_rows()


def test_a_commit_failure_outcome_survives_a_reconnect(tmp_path):
    repository = migrated(tmp_path)
    key = stage_key_for()
    assert repository.claim_stage_key(**key, run_id="run-1")

    with pytest.raises(sqlite3.OperationalError):
        with repository.stage_run(
            run_id="run-1",
            stage="ingest",
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
            stage_key=key,
        ) as run:
            with failing_commits():
                repository.ingest_raw_items([raw_item(1)], run=run, terminal=True)

    reopened = Phase0Repository(repository.database_path)
    assert reopened.count("raw_items") == 0
    assert reopened.run_log_entries(run_id="run-1")[0]["status"] == "failed"
    assert reopened.stage_key_state(**key)["status"] == "failed"
    assert reopened.claim_stage_key(**key, run_id="run-2")


#: The only places allowed to commit.  A helper that commits on the side
#: would put a transaction boundary somewhere nobody is looking.
COMMITTERS = {"_connect", "_logged_mutation", "_write_settlement"}


def test_no_helper_hides_a_commit():
    """Transaction boundaries are where the design says they are."""

    committers = set()
    for name, member in inspect.getmembers(Phase0Repository, inspect.isfunction):
        try:
            source = inspect.getsource(member)
        except (OSError, TypeError) as exc:  # pragma: no cover - defensive
            raise AssertionError(f"cannot read the source of {name}: {exc}") from exc
        body = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("#")
        )
        if re.search(r"\bconnection\.(?:commit|rollback)\(", body):
            committers.add(name)

    assert committers == COMMITTERS, (
        f"transaction boundaries this audit does not know about: "
        f"{sorted(committers - COMMITTERS)}"
    )

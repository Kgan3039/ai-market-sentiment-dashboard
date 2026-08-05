"""Additive, atomic schema migrations for the Phase 0 database.

Two rules shape this module.

**Migrations are additive and are never rewritten.**  An applied migration
is recorded by name and checksum in ``schema_migrations``; editing a file
that a database has already applied is refused rather than silently
producing two different schemas from the same version number.  A schema
change is a new file, always.

**A migration either lands completely or not at all.**  Each file runs
inside one ``BEGIN IMMEDIATE`` transaction that also writes the ledger row
and advances ``user_version``.  A failure rolls the whole thing back, so a
failed upgrade leaves both the schema and ``user_version`` untouched.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .errors import Phase0MigrationError


LEDGER_TABLE = "schema_migrations"

_LEDGER_DDL = f"""
CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (
    name TEXT PRIMARY KEY,
    version INTEGER NOT NULL CHECK (version > 0),
    checksum TEXT NOT NULL CHECK (length(checksum) = 64),
    applied_at TEXT NOT NULL CHECK (datetime(applied_at) IS NOT NULL)
)
"""

#: Schema version of the originally published Phase 0 database, before the
#: integrity/lease work.  Databases at this version upgrade through
#: migration 003 rather than by re-running rewritten 001/002 files.
LEGACY_SCHEMA_VERSION = 2

LEGACY_UPGRADE_MIGRATION = "003_integrity_leases_and_upgrade.sql"


@dataclass(frozen=True)
class Migration:
    """One migration file, addressed by name rather than by number."""

    name: str
    version: int
    sql: str
    checksum: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_statements(sql: str) -> Iterator[str]:
    """Yield complete SQL statements, keeping trigger bodies intact."""

    pending = ""
    for line in sql.splitlines():
        pending += f"{line}\n"
        if sqlite3.complete_statement(pending):
            statement = pending.strip()
            if statement:
                yield statement
            pending = ""
    if pending.strip():
        raise Phase0MigrationError("migration contains an incomplete statement")


def load_migrations(migrations_path: Path) -> list[Migration]:
    """Load every migration file in deterministic (version, name) order."""

    migrations: list[Migration] = []
    seen_names: set[str] = set()
    for path in sorted(Path(migrations_path).glob("*.sql")):
        prefix = path.name.split("_", 1)[0]
        try:
            version = int(prefix)
        except ValueError as exc:
            raise Phase0MigrationError(
                f"migration {path.name} does not start with a version number"
            ) from exc
        if version <= 0:
            raise Phase0MigrationError(
                f"migration {path.name} must use a positive version number"
            )
        if path.name in seen_names:
            raise Phase0MigrationError(f"duplicate migration name: {path.name}")
        seen_names.add(path.name)
        text = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                name=path.name,
                version=version,
                sql=text,
                checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    migrations.sort(key=lambda migration: (migration.version, migration.name))
    return migrations


def _applied_migrations(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["name"]): str(row["checksum"])
        for row in connection.execute(f"SELECT name, checksum FROM {LEDGER_TABLE}")
    }


def _backfill_ledger(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    user_version: int,
) -> None:
    """Record migrations a pre-ledger database already carries.

    Databases created before the ledger existed only report ``user_version``.
    Everything at or below that version is, by definition, already applied.
    """

    applied = _applied_migrations(connection)
    pending = [
        migration
        for migration in migrations
        if migration.version <= user_version and migration.name not in applied
    ]
    if not pending:
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        for migration in pending:
            connection.execute(
                f"INSERT INTO {LEDGER_TABLE} (name, version, checksum, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (migration.name, migration.version, migration.checksum, _utc_now()),
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _verify_history(
    connection: sqlite3.Connection, migrations: Sequence[Migration]
) -> None:
    applied = _applied_migrations(connection)
    known = {migration.name: migration.checksum for migration in migrations}
    for name, checksum in sorted(applied.items()):
        expected = known.get(name)
        if expected is None:
            raise Phase0MigrationError(
                f"database has applied unknown migration {name}; "
                "this database is newer than this code"
            )
        if expected != checksum:
            raise Phase0MigrationError(
                f"migration {name} was modified after it was applied; "
                "add a new additive migration instead of rewriting history"
            )


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    *,
    legacy_upgrade: Callable[[sqlite3.Connection], None] | None = None,
) -> list[str]:
    """Apply every unapplied migration atomically; return their names."""

    connection.execute(_LEDGER_DDL)
    connection.commit()
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    _backfill_ledger(connection, migrations, user_version)
    _verify_history(connection, migrations)

    applied = set(_applied_migrations(connection))
    newly_applied: list[str] = []
    for migration in migrations:
        if migration.name in applied:
            continue
        statements = list(split_statements(migration.sql))
        connection.execute("BEGIN IMMEDIATE")
        try:
            if (
                migration.name == LEGACY_UPGRADE_MIGRATION
                and legacy_upgrade is not None
                and user_version == LEGACY_SCHEMA_VERSION
            ):
                legacy_upgrade(connection)
            for statement in statements:
                connection.execute(statement)
            connection.execute(
                f"INSERT INTO {LEDGER_TABLE} (name, version, checksum, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (migration.name, migration.version, migration.checksum, _utc_now()),
            )
            target = max(user_version, migration.version)
            connection.execute(f"PRAGMA user_version = {int(target)}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        user_version = max(user_version, migration.version)
        newly_applied.append(migration.name)
    return newly_applied


__all__ = [
    "LEDGER_TABLE",
    "LEGACY_SCHEMA_VERSION",
    "LEGACY_UPGRADE_MIGRATION",
    "Migration",
    "apply_migrations",
    "load_migrations",
    "split_statements",
]

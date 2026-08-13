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

from . import lineages
from .errors import Phase0MigrationError
from .lineages import HistoricalLineage


LEDGER_TABLE = "schema_migrations"

LEDGER_DDL = f"""
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

#: The oldest SQLite these migrations are written against.
#:
#: Chosen from what the schema actually uses, not from what happens to be
#: installed: ``ON CONFLICT ... DO UPDATE`` needs 3.24, and the ``json_valid``
#: / ``json_type`` CHECK constraints need JSON1, which became a default build
#: option in 3.38.  Nothing here needs anything newer, and that is a
#: constraint worth keeping — several current distributions ship Python
#: against a SQLite in the 3.4x range.
#:
#: The one trap this floor exists to catch is ``RAISE()``: before 3.47 its
#: message had to be a string *literal*, and because SQLite parses the whole
#: schema the first time a connection touches it, a single trigger it cannot
#: parse makes the entire database unopenable rather than merely breaking the
#: statement that would have fired.  ``tests/test_phase0_persistence_contracts``
#: enforces both halves: no migration may use a non-literal message, and every
#: migration must apply on the oldest SQLite the test host can offer.
MINIMUM_SQLITE_VERSION = (3, 38, 0)


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


def latest_version(migrations: Sequence[Migration]) -> int:
    """The newest schema version this code knows how to produce."""

    return max((migration.version for migration in migrations), default=0)


def _has_ledger_rows(connection: sqlite3.Connection) -> bool:
    """Whether this database keeps its own history — read-only.

    Asked before the ledger table is created, so it cannot go through
    :func:`_applied_migrations`: the question is precisely whether that
    table is there yet.
    """

    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (LEDGER_TABLE,),
    ).fetchone()
    if exists is None:
        return False
    return (
        connection.execute(f"SELECT 1 FROM {LEDGER_TABLE} LIMIT 1").fetchone()
        is not None
    )


def _assert_version_is_known(
    connection: sqlite3.Connection, migrations: Sequence[Migration]
) -> None:
    """Refuse a database newer than anything this code can produce.

    Read-only, and deliberately the first thing that happens — before
    lineage recognition, before the ledger exists, before any decision
    about what to write.  A database this code does not understand has to
    be exactly as it was when we say so.

    ``user_version`` is not a claim about *which* files ran, only about how
    far some build of this schema got, so a version above the newest
    bundled migration was written by code we do not have.  Proceeding meant
    ``_backfill_ledger`` recording every migration as applied — each one is
    numerically ``<= user_version``, which was the only test it had —
    leaving nothing pending and reporting success over a database whose
    Phase 0 schema had never been created.  Reading a high number as "all
    of this ran" is exactly backwards: the further ahead a database is, the
    less of it we can vouch for.
    """

    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    latest = latest_version(migrations)
    if user_version > latest:
        raise Phase0MigrationError(
            f"database reports schema version {user_version}, which is newer "
            f"than the newest migration this code has ({latest}); it was "
            "written by a later release and cannot be migrated backwards or "
            "reinterpreted. Nothing has been changed."
        )


def _assert_ledger_agrees_with_version(
    connection: sqlite3.Connection, user_version: int
) -> None:
    """Refuse a ledger and a ``user_version`` that describe different databases.

    The two are written together, in one transaction, by ``_apply_one``:
    a ledger row and the version bump that goes with it.  So once a
    database keeps its own history the two can only disagree if something
    outside this module moved one of them, and either direction is a
    silent wrong answer rather than a loud one.

    A version *behind* the ledger was accepted outright — nothing is
    pending, so ``migrate()`` reported success and left the database
    reporting a schema version it had long since passed.  A version
    *ahead* of the ledger sent already-applied migrations through again
    and died on whatever they collided with, which fails closed only by
    luck and only after the attempt is under way.

    The convergence path asks this same question of the historical
    lineages and always has; this is the ordinary path catching up to it.
    """

    if not _has_ledger_rows(connection):
        return
    reached = int(
        connection.execute(f"SELECT max(version) FROM {LEDGER_TABLE}").fetchone()[0]
    )
    if reached != user_version:
        raise Phase0MigrationError(
            f"database reports schema version {user_version} but its ledger "
            f"reaches {reached}; the two are written together, so one of them "
            "has been changed outside this module and neither can be trusted "
            "to say what ran. Nothing has been changed."
        )


def _assert_history_is_evidenced(
    connection: sqlite3.Connection, user_version: int
) -> None:
    """Refuse an unledgered database claiming a post-ledger version.

    The "newer than this code" rule alone stops one short of the hole,
    because at *equality* the same backfill claims every migration and
    leaves nothing pending: an empty file stamped with the current version
    was accepted just as silently as one stamped 999.

    Every database this code has ever created carries a ledger, written in
    the same transaction as its first migration.  So an unledgered
    database is either the pre-ledger v2 schema or a registered historical
    lineage — which has already been recognized and returned by the time
    this runs — and anything else claiming a later version is asserting a
    history with nothing behind it.

    At or below the watermark the backfill is not a guess: what it infers
    from the version, migration 003 then checks against the tables a real
    v2 database has, and a database that merely claims to be v2 is refused
    there by name.
    """

    if user_version > LEGACY_SCHEMA_VERSION and not _has_ledger_rows(connection):
        raise Phase0MigrationError(
            f"database reports schema version {user_version} but keeps no "
            f"migration ledger; only the pre-ledger v{LEGACY_SCHEMA_VERSION} "
            "schema and registered historical lineages may claim a version "
            "without one, so nothing here is evidence that any migration "
            "ran. Nothing has been changed."
        )


def _backfill_ledger(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    user_version: int,
    lineage: HistoricalLineage | None,
) -> None:
    """Record migrations a pre-ledger database already carries.

    Databases created before the ledger existed only report ``user_version``.
    Everything at or below that version is, by definition, already applied.

    This runs *only* against an empty ledger.  Once a database keeps its own
    history, ``user_version`` stops being evidence of what has run: a later
    migration may share a version number with one already applied (numbers
    collide across stacked branches), and inferring "applied" from the
    number would skip it forever without ever executing it.

    When the database is on a recognized historical ``lineage``, that
    lineage's file is backfilled with *its* checksum, not the local file's.
    Writing the local checksum would be the worst thing here: it is a
    silent claim that the approved migration ran, on a database where a
    different one did.

    The caller owns the transaction.  This used to open and commit its
    own, which meant a backfilled ledger outlived the attempt that
    produced it: on a database whose first migration then failed, the
    rollback had nothing left to undo.
    """

    applied = _applied_migrations(connection)
    if applied:
        return
    pending = [
        migration for migration in migrations if migration.version <= user_version
    ]
    for migration in pending:
        checksum = migration.checksum
        if lineage is not None and migration.name == lineage.migration:
            checksum = lineage.checksum
        connection.execute(
            f"INSERT INTO {LEDGER_TABLE} (name, version, checksum, applied_at) "
            "VALUES (?, ?, ?, ?)",
            (migration.name, migration.version, checksum, _utc_now()),
        )


def _verify_history(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    lineage: HistoricalLineage | None,
) -> None:
    """Every applied migration must be one this code knows, unchanged.

    The single exception is a checksum that a registered historical
    lineage pins *and* that this database has proved it is on — either by
    still carrying that lineage's schema, or by having converged, in which
    case its whole ledger and its whole schema must be exactly what a
    settlement produces.  Provenance corroborates that; it never supplies
    it.  Any other mismatch is refused exactly as before: an unknown
    variant is not a lineage, it is a modified file.
    """

    applied = _applied_migrations(connection)
    known = {migration.name: migration.checksum for migration in migrations}
    known.update(lineages.convergence_migrations())
    excused = dict(lineages.recorded(connection, migrations))
    if lineage is not None:
        excused[lineage.migration] = lineage
    for name, checksum in sorted(applied.items()):
        expected = known.get(name)
        if expected is None:
            raise Phase0MigrationError(
                f"database has applied unknown migration {name}; "
                "this database is newer than this code"
            )
        if expected == checksum:
            continue
        historical = excused.get(name)
        if historical is not None and historical.checksum == checksum:
            continue
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
    """Apply every unapplied migration atomically; return their names.

    Two paths, and the fork is decided *before anything is written*.

    A database on a registered historical lineage takes the compatibility
    settlement: one transaction covering everything, because a partial
    conversion of a forked database is not a state anyone can reason about.
    Every other database takes the ordinary path, unchanged — one
    transaction per migration, exactly as before.
    """

    # Read-only, and first: a version this code cannot have produced is
    # unsupportable whatever the database turns out to look like, and the
    # answer must not depend on how far recognition gets.
    _assert_version_is_known(connection, migrations)

    # Read-only.  Recognition must not create the ledger, must not
    # backfill, and must not advance anything: a database that turns out
    # not to be a known lineage has to be untouched when we find out.
    lineage, reason = lineages.recognize(connection)
    if lineage is not None and reason is not None:
        # It carries a fork's marker objects and is not that fork. The
        # ordinary path assumes a pre-ledger database ran the approved
        # migrations up to its user_version, which is precisely the
        # assumption a fork breaks, so nothing here may proceed on a guess.
        raise Phase0MigrationError(
            f"database looks like the {lineage.lineage} lineage but does not "
            f"match it: {reason}. Nothing has been changed."
        )
    if lineage is not None:
        return _converge(connection, migrations, lineage)

    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    # Still read-only, and still before the transaction opens: the
    # historical lineages are the only unledgered databases with a
    # provenance this code can stand behind, and they have returned above.
    _assert_history_is_evidenced(connection, user_version)

    newly_applied: list[str] = []

    # The bootstrap tables, and any truthful backfill of them, belong to
    # the *first* attempt rather than preceding it.  They used to be
    # created and committed up front, so a brand-new database whose first
    # migration then failed was left carrying `schema_migrations` and
    # `schema_lineage` — metadata describing an attempt that, by the
    # contract this module opens with, never happened.  Folding them into
    # the first migration's transaction restores "a failed attempt changes
    # nothing" without weakening the per-migration atomicity that follows:
    # every later migration still commits on its own, so a failure at step
    # *n* still leaves a database honestly at step *n-1*.
    pending: list[Migration] = []
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(LEDGER_DDL)
        connection.execute(lineages.LINEAGE_DDL)
        _backfill_ledger(connection, migrations, user_version, None)
        _verify_history(connection, migrations, None)
        # After the checksum rule, not before it: a tampered ledger is
        # usually *also* out of step with the version, and "this migration
        # was edited" says more about what happened than "these two
        # numbers differ".  Inside the transaction for the same reason
        # `_verify_history` is — the raise rolls it back, so a refusal
        # still leaves the database exactly as it was.
        _assert_ledger_agrees_with_version(connection, user_version)

        applied = set(_applied_migrations(connection))
        pending = [
            migration for migration in migrations if migration.name not in applied
        ]
        if pending:
            _apply_one(connection, pending[0], user_version, legacy_upgrade)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    if pending:
        user_version = max(user_version, pending[0].version)
        newly_applied.append(pending[0].name)

    for migration in pending[1:]:
        connection.execute("BEGIN IMMEDIATE")
        try:
            _apply_one(connection, migration, user_version, legacy_upgrade)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        user_version = max(user_version, migration.version)
        newly_applied.append(migration.name)
    return newly_applied


def _apply_one(
    connection: sqlite3.Connection,
    migration: Migration,
    user_version: int,
    legacy_upgrade: Callable[[sqlite3.Connection], None] | None,
) -> None:
    """Run one migration, its ledger row, and its ``user_version`` bump.

    The caller owns the transaction, so all three land together or none
    of them does — including a migration whose text will not parse, which
    is why the file is only split here and not up front: parsing every
    file eagerly would reject an *already applied* migration that a later
    edit left unparseable, before the checksum check could report the far
    more useful "rewriting history".
    """

    if (
        migration.name == LEGACY_UPGRADE_MIGRATION
        and legacy_upgrade is not None
        and user_version == LEGACY_SCHEMA_VERSION
    ):
        legacy_upgrade(connection)
    for statement in split_statements(migration.sql):
        connection.execute(statement)
    connection.execute(
        f"INSERT INTO {LEDGER_TABLE} (name, version, checksum, applied_at) "
        "VALUES (?, ?, ?, ?)",
        (migration.name, migration.version, migration.checksum, _utc_now()),
    )
    connection.execute(
        f"PRAGMA user_version = {int(max(user_version, migration.version))}"
    )


def _converge(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    lineage: HistoricalLineage,
) -> list[str]:
    """Convert a recognized historical database, all of it or none of it.

    The ordinary path commits one migration at a time, which is right: each
    approved migration is a step every database takes, and a failure at
    step *n* leaves a database that is honestly at step *n-1*.  None of
    that holds here.  A half-converted fork is at no version at all — it
    has bootstrap tables it did not ask for, a backfilled ledger describing
    a history it has not lived, and a schema partway between two branches.
    The earlier version of this code produced exactly that.

    So the whole thing is one transaction: bootstrap, truthful backfill,
    convergence, every remaining approved migration, final validation, and
    provenance.  On any failure the rollback restores the original
    database — tables, data, ledger, provenance, and ``user_version``.

    ``legacy_upgrade`` is not offered here: it converts the v2 schema, and
    a recognized lineage is at its own pinned version by definition.
    """

    # Read and verify the convergence file *before* opening the
    # transaction, so a missing or tampered file costs the database nothing.
    convergence = Migration(
        name=lineage.convergence,
        version=lineage.user_version,
        sql=lineages.load_convergence(lineage),
        checksum=lineage.convergence_checksum,
    )
    plan = [convergence] + [
        migration
        for migration in migrations
        if migration.version > lineage.user_version
    ]
    statements = {
        migration.name: list(split_statements(migration.sql)) for migration in plan
    }

    connection.execute("BEGIN IMMEDIATE")
    try:
        # Recognition happened outside the write lock; another writer could
        # have moved underneath it.  Cheap to repeat, and the whole
        # settlement rests on it.
        reason = lineages.mismatch(connection, lineage)
        if reason is not None:
            raise Phase0MigrationError(
                f"database stopped matching the {lineage.lineage} lineage "
                f"before conversion could start: {reason}"
            )

        connection.execute(LEDGER_DDL)
        connection.execute(lineages.LINEAGE_DDL)

        # The history this database actually lived, told truthfully: the
        # shared approved files at their own checksums, and the historical
        # migration at *its* checksum, never the approved one.
        #
        # One timestamp for the whole settlement.  The historical row, the
        # convergence row, and the provenance row are written together and
        # say so, which is a thing a ledger assembled from parts does not.
        settled_at = _utc_now()
        for migration in migrations:
            if migration.version > lineage.user_version:
                continue
            checksum = (
                lineage.checksum
                if migration.name == lineage.migration
                else migration.checksum
            )
            connection.execute(
                f"INSERT OR REPLACE INTO {LEDGER_TABLE} "
                "(name, version, checksum, applied_at) VALUES (?, ?, ?, ?)",
                (migration.name, migration.version, checksum, settled_at),
            )

        newly_applied: list[str] = []
        for migration in plan:
            for statement in statements[migration.name]:
                connection.execute(statement)
            applied_at = (
                settled_at if migration.name == lineage.convergence else _utc_now()
            )
            connection.execute(
                f"INSERT INTO {LEDGER_TABLE} (name, version, checksum, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (migration.name, migration.version, migration.checksum, applied_at),
            )
            newly_applied.append(migration.name)

        target = max(migration.version for migration in migrations)
        connection.execute(f"PRAGMA user_version = {int(target)}")

        # Provenance first, because the settlement is then held to the very
        # rule that will later be asked to accept its result.
        lineages.record(connection, lineage, settled_at)
        _assert_converged(connection, migrations, lineage, target)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return newly_applied


def _assert_converged(
    connection: sqlite3.Connection,
    migrations: Sequence[Migration],
    lineage: HistoricalLineage,
    target: int,
) -> None:
    """Refuse to commit a settlement that did not actually land.

    Marking migrations applied is not the same as their schema existing.
    This checks the invariants that distinguish the two, still inside the
    transaction, so a settlement that only *claims* to have converged
    rolls back with everything else.
    """

    applied = _applied_migrations(connection)
    missing = [
        migration.name for migration in migrations if migration.name not in applied
    ]
    if missing:
        raise Phase0MigrationError(
            f"compatibility settlement did not apply {missing}; refusing to commit"
        )
    if applied.get(lineage.migration) != lineage.checksum:
        raise Phase0MigrationError(
            "compatibility settlement lost the historical checksum; refusing to commit"
        )
    if applied.get(lineage.convergence) != lineage.convergence_checksum:
        raise Phase0MigrationError(
            "compatibility settlement did not record its convergence; "
            "refusing to commit"
        )
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    triggers = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    absent = set(lineage.converged_tables) - tables
    if absent:
        raise Phase0MigrationError(
            f"compatibility settlement left {sorted(absent)} uncreated; "
            "refusing to commit"
        )
    surviving = set(lineage.fingerprint_triggers) & triggers
    if surviving:
        raise Phase0MigrationError(
            f"compatibility settlement left the historical triggers "
            f"{sorted(surviving)} in place; refusing to commit"
        )
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != target:
        raise Phase0MigrationError(
            f"compatibility settlement ended at user_version {version}, "
            f"not {target}; refusing to commit"
        )
    # The closing condition, and the one that ties this path to the next
    # ``migrate()``: a settlement may only commit a database that this code
    # would itself recognize as converged — exact ledger, exact schema,
    # exact provenance.  Anything less would commit a state that the very
    # next run has to refuse.
    problem = lineages.post_convergence_problem(connection, lineage, migrations)
    if problem is not None:
        raise Phase0MigrationError(
            f"compatibility settlement produced a database it would not "
            f"itself accept ({problem}); refusing to commit"
        )


__all__ = [
    "LEDGER_DDL",
    "LEDGER_TABLE",
    "LEGACY_SCHEMA_VERSION",
    "LEGACY_UPGRADE_MIGRATION",
    "LINEAGE_TABLE",
    "MINIMUM_SQLITE_VERSION",
    "Migration",
    "apply_migrations",
    "load_migrations",
    "split_statements",
]

LINEAGE_TABLE = lineages.LINEAGE_TABLE

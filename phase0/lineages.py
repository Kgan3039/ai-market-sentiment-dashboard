"""Known historical migration lineages, and how a database proves it is one.

Migration checksums are immutable: a file that a database has already
applied may never be edited, and :func:`phase0.schema._verify_history`
refuses the database rather than guessing.  That rule is what makes the
ledger worth having, and nothing here weakens it.

What it cannot express on its own is a *fork*.  Two branches wrote a
different ``004_supported_ticker_universe.sql``, both were applied to real
databases, and one of the two implementations supersedes the other.  The
superseded database is not corrupt and its checksum is not wrong — it is
evidence of a different, known transition.

So the exception is a closed registry rather than a policy.  Each entry
names one exact file, one exact checksum, the schema that file must have
produced, and the additive convergence migration that brings such a
database onto the approved lineage.  A database qualifies only by
matching *all* of it; "claims to be version 4" is not evidence of
anything.  Everything else keeps failing exactly as before.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .errors import Phase0MigrationError


#: Provenance, kept beside the ledger rather than inside it.  The ledger
#: records what ran; this records which historical lineage a database
#: arrived on.  Both are created for *every* database, so a converged one
#: is schema-identical to a fresh one — it just has a row here.
LINEAGE_TABLE = "schema_lineage"

LINEAGE_DDL = f"""
CREATE TABLE IF NOT EXISTS {LINEAGE_TABLE} (
    lineage TEXT PRIMARY KEY,
    migration TEXT NOT NULL,
    historical_checksum TEXT NOT NULL CHECK (length(historical_checksum) = 64),
    schema_fingerprint TEXT NOT NULL CHECK (length(schema_fingerprint) = 64),
    convergence TEXT NOT NULL,
    recognized_at TEXT NOT NULL CHECK (datetime(recognized_at) IS NOT NULL)
)
"""

#: Convergence migrations live apart from the ordinary ones.  They are not
#: part of any database's forward history — only a recognized lineage ever
#: runs one — so they must never be picked up by the directory scan.
CONVERGENCE_PATH = Path(__file__).with_name("migrations") / "compat"


@dataclass(frozen=True)
class HistoricalLineage:
    """One known historical variant, and the proof required to accept it.

    Every field is part of the proof.  ``checksum`` alone would let any
    database that once ran that file through; the schema conditions are
    what establish that this database is *specifically* the lineage the
    convergence migration was written against.
    """

    #: Stable identifier, recorded in provenance and never reused.
    lineage: str
    #: Exact ledger name of the historical migration.
    migration: str
    #: Exact SHA-256 of the historical file's bytes.
    checksum: str
    #: ``user_version`` a database on this lineage reports before converging.
    user_version: int
    #: SHA-256 over the *whole* schema; see :func:`structural_fingerprint`.
    #: Every table, column, type, nullability, default, primary key, foreign
    #: key, index, and trigger body is in it.
    schema_fingerprint: str
    #: The historical lineage's own triggers.  Named here so the converged
    #: check can insist they are gone.
    fingerprint_triggers: tuple[str, ...]
    #: Objects no approved database ever has.  Finding one means this
    #: database is *claiming* the lineage, and it then has to match it
    #: exactly or be refused — never quietly handled as something else.
    signature_objects: tuple[str, ...]
    #: The exact ledger a database on this lineage may carry.  The genuine
    #: one predates the ledger and has none at all; anything else must be
    #: precisely this, no more and no less.
    historical_ledger: Mapping[str, str]
    #: Tables the convergence creates, checked when validating provenance
    #: and after the settlement.
    converged_tables: tuple[str, ...]
    #: The additive migration that brings this lineage onto the approved one.
    convergence: str
    convergence_checksum: str
    #: What this lineage was, in one line, for the error message.
    description: str = field(default="")


#: The approved 001–003 are byte-identical across both lineages; requiring
#: them keeps a database that diverged *earlier* from qualifying here.
_APPROVED_001_003 = {
    "001_initial.sql": (
        "a6c2f3c9b1380ce6e030090e376ffc24fb7d8e9ef4b6870529382ae3323226dc"
    ),
    "002_source_state_and_stage_keys.sql": (
        "cf809b4ac63c326cb3b103b58f344a51283dd27655d81bf9a29d32191df80850"
    ),
    "003_integrity_leases_and_upgrade.sql": (
        "136463c74511a46ebacc5537db4b92b7129a35c70bf30d9b7383b4978f14beec"
    ),
}

#: The 12 ticker triggers the remote ``004`` created.  Their names are the
#: giveaway — the approved file calls them ``trg_*`` — and their bodies are
#: what the fingerprint is taken over.
_REMOTE_V4_TRIGGERS = (
    "enforce_raw_item_association_insert",
    "enforce_raw_item_association_update",
    "enforce_raw_item_candidate_insert",
    "enforce_raw_item_candidate_update",
    "enforce_raw_item_ticker_insert",
    "enforce_raw_item_ticker_update",
    "enforce_stage_key_ticker_insert",
    "enforce_stage_key_ticker_update",
    "enforce_story_ticker_insert",
    "enforce_story_ticker_update",
    "enforce_theme_ticker_insert",
    "enforce_theme_ticker_update",
)

REMOTE_V4_LINEAGE = HistoricalLineage(
    lineage="remote-v4-supported-ticker-universe",
    migration="004_supported_ticker_universe.sql",
    checksum="fd4d208833984199a0a4307b82a8693767349c89e039ec1ec4d93eada78b9eab",
    user_version=4,
    schema_fingerprint=(
        "ea38ea10cf7f0657f45f68fec86ffc18bd86a3f087d61924e67b235d8db7df5d"
    ),
    fingerprint_triggers=_REMOTE_V4_TRIGGERS,
    # The approved 004 calls its triggers ``trg_*``; nothing on the approved
    # lineage has ever created one of these.
    signature_objects=_REMOTE_V4_TRIGGERS,
    # The genuine database predates the ledger, so it has none.  A database
    # that has been given one must carry exactly this and nothing else: the
    # shared, byte-identical 001–003, and the historical 004.
    historical_ledger={
        **_APPROVED_001_003,
        "004_supported_ticker_universe.sql": (
            "fd4d208833984199a0a4307b82a8693767349c89e039ec1ec4d93eada78b9eab"
        ),
    },
    # The approved 004 creates this table; the remote one never did, which
    # is why the approved 008 and 009 cannot run on such a database.
    converged_tables=("supported_tickers",),
    convergence="004_remote_v4_convergence.sql",
    convergence_checksum=(
        "eb0609c33ea7de41c46eda8684fa7b4012cea198f96f25dc1f68e75646e0532e"
    ),
    description=(
        "the remote 004 that enforced the ticker universe with literal "
        "IN-lists and enforce_* triggers, and never created supported_tickers"
    ),
)


#: Keyed exactly as a reviewer would write it: (filename, checksum).
KNOWN_HISTORICAL_MIGRATIONS: dict[tuple[str, str], HistoricalLineage] = {
    (REMOTE_V4_LINEAGE.migration, REMOTE_V4_LINEAGE.checksum): REMOTE_V4_LINEAGE,
}


def convergence_migrations() -> dict[str, str]:
    """Convergence file names and their pinned checksums.

    ``_verify_history`` needs these: a converged database has a ledger row
    for one, and an unrecognized ledger row is refused.
    """

    return {
        lineage.convergence: lineage.convergence_checksum
        for lineage in KNOWN_HISTORICAL_MIGRATIONS.values()
    }


#: Excluded from the structural fingerprint and checked by their own rules.
#: They describe migration *state* rather than schema, and the genuine
#: historical database has neither of them.
METADATA_TABLES = frozenset({LINEAGE_TABLE, "schema_migrations"})


def _objects(connection: sqlite3.Connection, kind: str) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (kind,)
        )
    }


def _table_names(connection: sqlite3.Connection) -> list[str]:
    return sorted(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
        if str(row[0]) not in METADATA_TABLES
    )


def structural_fingerprint(connection: sqlite3.Connection) -> str:
    """A digest of the *whole* schema, not a sample of it.

    Every object in ``sqlite_master`` with its stored SQL, and for every
    table its full ``table_info`` (name, declared type, nullability,
    default, primary-key position), ``foreign_key_list``, and every index
    with its columns.  Deterministic, and exhaustive on purpose: a
    fingerprint over a chosen handful of triggers accepted a database with
    ``run_log`` missing entirely, which then advanced through four
    migrations before failing.

    Migration-state tables are excluded and checked separately — the
    genuine historical database has neither, and a database that has been
    given one is a different question from what its schema looks like.
    """

    parts: list[str] = []
    for row in connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ):
        kind, name, table, sql = (str(value or "") for value in row)
        if name in METADATA_TABLES or table in METADATA_TABLES:
            continue
        parts.append(f"object\t{kind}\t{name}\t{table}\t{sql.strip()}")
    for table in _table_names(connection):
        for column in connection.execute(f"PRAGMA table_info({table})"):
            parts.append(
                f"column\t{table}\t" + "\t".join(str(value) for value in column)
            )
        foreign_keys = sorted(
            tuple(str(value) for value in row)
            for row in connection.execute(f"PRAGMA foreign_key_list({table})")
        )
        for row in foreign_keys:
            parts.append(f"fk\t{table}\t" + "\t".join(row))
        indexes = sorted(
            tuple(str(value) for value in row)
            for row in connection.execute(f"PRAGMA index_list({table})")
        )
        for row in indexes:
            parts.append(f"index\t{table}\t" + "\t".join(row))
            for column in connection.execute(f"PRAGMA index_info({row[1]})"):
                parts.append(
                    f"indexcol\t{table}\t{row[1]}\t"
                    + "\t".join(str(value) for value in column)
                )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def ledger_rows(connection: sqlite3.Connection) -> dict[str, str] | None:
    """The ledger as it stands, or ``None`` when there is no ledger table.

    ``None`` and ``{}`` are different facts: the genuine historical
    database predates the ledger entirely, while an empty ledger table is
    something else's doing.
    """

    if "schema_migrations" not in _objects(connection, "table"):
        return None
    return {
        str(row[0]): str(row[1])
        for row in connection.execute("SELECT name, checksum FROM schema_migrations")
    }


def lineage_rows(connection: sqlite3.Connection) -> list[dict[str, str]] | None:
    if LINEAGE_TABLE not in _objects(connection, "table"):
        return None
    return [
        {
            "lineage": str(row[0]),
            "migration": str(row[1]),
            "historical_checksum": str(row[2]),
            "schema_fingerprint": str(row[3]),
            "convergence": str(row[4]),
        }
        for row in connection.execute(
            f"SELECT lineage, migration, historical_checksum, schema_fingerprint, "
            f"convergence FROM {LINEAGE_TABLE}"
        )
    ]


def mismatch(connection: sqlite3.Connection, lineage: HistoricalLineage) -> str | None:
    """Why this database is *not* ``lineage``, or ``None`` when it is.

    Read-only, and total: every condition is checked, and the first one
    that fails is named so a rejection can say what was wrong rather than
    just refusing.
    """

    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != lineage.user_version:
        return f"user_version is {version}, not {lineage.user_version}"

    fingerprint = structural_fingerprint(connection)
    if fingerprint != lineage.schema_fingerprint:
        return (
            "the schema does not match this lineage exactly "
            f"(fingerprint {fingerprint[:12]}…, expected "
            f"{lineage.schema_fingerprint[:12]}…)"
        )

    applied = ledger_rows(connection)
    if applied is not None and applied != dict(lineage.historical_ledger):
        return "the migration ledger does not match this lineage's history"

    provenance = lineage_rows(connection)
    if provenance:
        return "this database already carries lineage provenance"
    return None


def claimed(connection: sqlite3.Connection) -> HistoricalLineage | None:
    """The lineage this database *claims*, by carrying its marker objects.

    Claiming is not qualifying.  It only means the ordinary path must not
    have this database: a pre-ledger database is assumed to have run the
    approved migrations up to its ``user_version``, and that assumption is
    exactly what a fork invalidates.
    """

    names = _objects(connection, "trigger") | _objects(connection, "table")
    for lineage in KNOWN_HISTORICAL_MIGRATIONS.values():
        if names & set(lineage.signature_objects):
            return lineage
    return None


def recognize(
    connection: sqlite3.Connection,
) -> tuple[HistoricalLineage | None, str | None]:
    """What this database is, and — when it falls short — why.

    Read-only.  Nothing is created, backfilled, or advanced by asking.

    Three answers.  ``(None, None)``: no historical lineage is claimed, so
    the ordinary path applies.  ``(lineage, None)``: it is exactly that
    lineage.  ``(lineage, reason)``: it claims that lineage and is not it,
    which is a refusal — a database carrying half of a fork's schema is not
    something to guess about, and letting it fall through to the ordinary
    path is how one missing ``run_log`` advanced four migrations before
    anything noticed.
    """

    lineage = claimed(connection)
    if lineage is None:
        return None, None
    return lineage, mismatch(connection, lineage)


def converged_mismatch(
    connection: sqlite3.Connection, lineage: HistoricalLineage
) -> str | None:
    """Why this database's provenance is not trustworthy, or ``None``.

    Provenance is evidence, never authority.  A row saying "I came through
    the remote-v4 path" is checked against the live database every time:
    the ledger must record the historical checksum *and* the convergence
    that only this path applies, the row's own fields must be the
    registry's, and the convergence's effects must actually be present.

    Without that, replacing a fresh database's ``004`` checksum and
    inserting a matching row is enough to be waved through — which is
    exactly what it was.
    """

    rows = [
        row
        for row in (lineage_rows(connection) or [])
        if row["lineage"] == lineage.lineage
    ]
    if not rows:
        return "no provenance row for this lineage"
    if len(rows) > 1:
        return "duplicate provenance rows for this lineage"
    row = rows[0]
    expected = {
        "migration": lineage.migration,
        "historical_checksum": lineage.checksum,
        "schema_fingerprint": lineage.schema_fingerprint,
        "convergence": lineage.convergence,
    }
    for field_name, want in expected.items():
        if row[field_name] != want:
            return f"provenance {field_name} is not this lineage's"

    applied = ledger_rows(connection)
    if applied is None:
        return "provenance without a migration ledger"
    if applied.get(lineage.migration) != lineage.checksum:
        return "the ledger does not record this lineage's historical checksum"
    if applied.get(lineage.convergence) != lineage.convergence_checksum:
        # The one row a converged database has and a tampered fresh one
        # does not: the convergence migration actually ran.
        return "the ledger does not record the convergence that ran"

    tables = _objects(connection, "table")
    triggers = _objects(connection, "trigger")
    if set(lineage.converged_tables) - tables:
        return "the convergence's effects are not present in this schema"
    if set(lineage.fingerprint_triggers) & triggers:
        return "this schema still carries the historical lineage's triggers"
    return None


def recorded(connection: sqlite3.Connection) -> dict[str, HistoricalLineage]:
    """Lineages whose provenance this database can actually stand behind.

    After convergence the schema no longer looks like the historical one —
    that was the point — so this is what keeps the ledger's historical
    checksum explicable on every later ``migrate()``.  Each row is
    re-verified against the live database; an unverifiable one is simply
    not returned, and the ordinary checksum rule then refuses the database.
    """

    if lineage_rows(connection) is None:
        return {}
    found: dict[str, HistoricalLineage] = {}
    for lineage in KNOWN_HISTORICAL_MIGRATIONS.values():
        if converged_mismatch(connection, lineage) is None:
            found[lineage.migration] = lineage
    return found


def record(
    connection: sqlite3.Connection, lineage: HistoricalLineage, recognized_at: str
) -> None:
    """Write provenance, inside the convergence migration's transaction.

    Deliberately additive: the ledger keeps saying the historical checksum
    is what ran, because it is.  Nothing here rewrites that row.
    """

    connection.execute(
        f"INSERT OR IGNORE INTO {LINEAGE_TABLE} "
        "(lineage, migration, historical_checksum, schema_fingerprint, "
        " convergence, recognized_at) VALUES (?, ?, ?, ?, ?, ?)",
        (
            lineage.lineage,
            lineage.migration,
            lineage.checksum,
            lineage.schema_fingerprint,
            lineage.convergence,
            recognized_at,
        ),
    )


def load_convergence(lineage: HistoricalLineage) -> str:
    """Read the convergence file and hold it to its pinned checksum.

    The compatibility path is not an excuse to stop checking anything: the
    file that performs it is verified the same way every other migration
    is, and a modified one is refused.
    """

    path = CONVERGENCE_PATH / lineage.convergence
    if not path.exists():
        raise Phase0MigrationError(
            f"database is on the {lineage.lineage} lineage but its "
            f"convergence migration {lineage.convergence} is missing"
        )
    text = path.read_text(encoding="utf-8")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if checksum != lineage.convergence_checksum:
        raise Phase0MigrationError(
            f"convergence migration {lineage.convergence} does not match its "
            "pinned checksum; it may not be modified in place"
        )
    return text


__all__ = [
    "CONVERGENCE_PATH",
    "HistoricalLineage",
    "KNOWN_HISTORICAL_MIGRATIONS",
    "LINEAGE_DDL",
    "LINEAGE_TABLE",
    "REMOTE_V4_LINEAGE",
    "METADATA_TABLES",
    "claimed",
    "converged_mismatch",
    "convergence_migrations",
    "ledger_rows",
    "lineage_rows",
    "load_convergence",
    "mismatch",
    "recognize",
    "record",
    "recorded",
    "structural_fingerprint",
]

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
    #: SHA-256 over the defining schema objects; see :func:`schema_fingerprint`.
    schema_fingerprint: str
    #: Trigger names the fingerprint is computed over, in sorted order.
    fingerprint_triggers: tuple[str, ...]
    #: Objects the lineage must have, and must not have.
    required_tables: tuple[str, ...]
    forbidden_tables: tuple[str, ...]
    #: Objects the *approved* lineage would have at this version, proving
    #: this database did not come through the approved 004.
    forbidden_triggers: tuple[str, ...]
    #: Other ledger rows that must match the approved checksums exactly, so
    #: a database that forked earlier than this file does not qualify.
    companion_checksums: Mapping[str, str]
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
        "8c4b16cb453668d3383263eafedadf4ff1c011857e39bdf289a3ee7044587b31"
    ),
    fingerprint_triggers=_REMOTE_V4_TRIGGERS,
    # The 001–003 shape, which this lineage shares with the approved one.
    required_tables=(
        "pipeline_stage_keys",
        "raw_item_candidates",
        "raw_item_tickers",
        "raw_items",
        "stories",
        "story_members",
        "theme_citations",
        "themes",
    ),
    # The approved 004 creates this table; the remote one never did.  Its
    # absence at user_version 4 is the single clearest tell.
    forbidden_tables=("supported_tickers",),
    forbidden_triggers=("trg_raw_item_ticker_insert", "trg_supported_ticker_delete"),
    companion_checksums=_APPROVED_001_003,
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


def schema_fingerprint(connection: sqlite3.Connection, names: tuple[str, ...]) -> str:
    """Digest the stored SQL of the named triggers, in sorted order.

    Names alone would be trivial to fake; this is taken over what SQLite
    actually stored, so a database with a trigger of the right name and a
    different body does not match.
    """

    rows = {
        str(row[0]): str(row[1] or "").strip()
        for row in connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    if set(names) - set(rows):
        return ""
    blob = "\n".join(f"{name}\n{rows[name]}" for name in sorted(names))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _objects(connection: sqlite3.Connection, kind: str) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ?", (kind,)
        )
    }


def matches(
    connection: sqlite3.Connection,
    lineage: HistoricalLineage,
    applied: Mapping[str, str],
) -> bool:
    """True when this database really is ``lineage``, on every count.

    The ledger is checked when there is one.  The genuine remote database
    predates the ledger entirely and reports nothing but ``user_version``,
    which is exactly why the schema conditions carry the proof.
    """

    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != lineage.user_version:
        return False
    tables = _objects(connection, "table")
    if set(lineage.required_tables) - tables:
        return False
    if set(lineage.forbidden_tables) & tables:
        return False
    if set(lineage.forbidden_triggers) & _objects(connection, "trigger"):
        return False
    if schema_fingerprint(connection, lineage.fingerprint_triggers) != (
        lineage.schema_fingerprint
    ):
        return False
    if applied:
        # A ledger that disagrees about the shared history means this
        # database forked somewhere this registry says nothing about.
        for name, checksum in lineage.companion_checksums.items():
            if applied.get(name, checksum) != checksum:
                return False
        recorded = applied.get(lineage.migration)
        if recorded is not None and recorded != lineage.checksum:
            return False
    return True


def recognize(
    connection: sqlite3.Connection, applied: Mapping[str, str]
) -> HistoricalLineage | None:
    """The known lineage this database is on, if it is on one."""

    for lineage in KNOWN_HISTORICAL_MIGRATIONS.values():
        if matches(connection, lineage, applied):
            return lineage
    return None


def recorded(connection: sqlite3.Connection) -> dict[str, HistoricalLineage]:
    """Lineages this database has already been recognized as, by migration.

    After convergence the schema no longer looks like the historical one —
    that was the point — so provenance is what keeps the ledger's
    historical checksum explicable on every later ``migrate()``.
    """

    names = _objects(connection, "table")
    if LINEAGE_TABLE not in names:
        return {}
    found: dict[str, HistoricalLineage] = {}
    for row in connection.execute(
        f"SELECT lineage, migration, historical_checksum FROM {LINEAGE_TABLE}"
    ):
        lineage = KNOWN_HISTORICAL_MIGRATIONS.get((str(row[1]), str(row[2])))
        if lineage is not None and lineage.lineage == str(row[0]):
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
    "convergence_migrations",
    "load_convergence",
    "matches",
    "recognize",
    "record",
    "recorded",
    "schema_fingerprint",
]

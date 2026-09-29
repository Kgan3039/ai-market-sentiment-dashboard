"""B3: keep the live database's WAL coordination files provisioned.

A read-only API cannot read a WAL database whose ``-wal`` and ``-shm`` files
do not exist, and SQLite deletes both when the *last* connection to the
database closes -- which, with a cron writer that opens, commits and closes,
is the end of every scheduled run.  This process is the fix the B1 contract
names: it runs as the **writer** identity (the one that may create files in
the database directory), holds one connection open, and so keeps ``-wal``
and ``-shm`` in place across the pipeline's own open/write/close cycles.

**It is not a writer.**  The one connection it holds is SQLite ``mode=ro``
with ``query_only`` and :mod:`phase0.repository`'s write-denying authorizer,
and startup proves the connection refuses a write before reporting ready.
It never creates a missing database, never migrates, never changes the
journal mode, and never checkpoints: the pipeline stays the only thing that
mutates the database.  Nothing here imports a provider, a fetcher or the
network.

**It holds no snapshot.**  The connection runs in autocommit
(``isolation_level=None``), every statement is fully consumed, and startup
confirms no transaction is open, so the pipeline's automatic checkpoints are
never blocked by this process.

**It watches its own identity.**  The ``(st_dev, st_ino)`` of the path is
captured before and after opening.  If the path later disappears or names a
different file -- a restore, a replacement -- the keeper logs why and exits
nonzero so its supervisor restarts it against the file that is there now,
rather than holding an obsolete inode forever.  The same happens if either
coordination file disappears underneath it.

Run it as ``python -m phase0.wal_keeper --database /abs/path.sqlite3``;
``deploy/phase0-wal-keeper.service`` is the supported supervisor.

Exit codes: ``0`` clean shutdown on SIGTERM/SIGINT, ``2`` refused to start
(the database is not in a state the API may be served from), ``3`` lost the
database identity or its coordination files while running.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import stat
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
from urllib.parse import quote

from .redaction import redact_secrets
from .repository import MIGRATIONS_PATH, _read_only_authorizer
from .schema import latest_version, load_migrations

LOGGER = logging.getLogger("phase0.wal_keeper")

EXIT_STOPPED = 0
EXIT_REFUSED = 2
EXIT_LOST = 3

DEFAULT_HEARTBEAT_SECONDS = 30.0
#: The heartbeat is how the keeper notices a replaced or removed database or
#: coordination file, so it must be short against the pipeline's shortest
#: cron interval (30 minutes): at most 5 minutes, so a restore is picked up
#: well before the next scheduled run.
MAX_HEARTBEAT_SECONDS = 300.0
SIDECAR_SUFFIXES = ("-wal", "-shm")

#: The 16-byte header every SQLite 3 database file starts with.
SQLITE_MAGIC = b"SQLite format 3\x00"


class KeeperRefused(RuntimeError):
    """The database is not in a state the keeper may hold open."""


class KeeperLost(RuntimeError):
    """The database the keeper opened is no longer the one at its path."""


def check_heartbeat(value: float) -> float:
    """A finite interval in ``(0, MAX_HEARTBEAT_SECONDS]``, or :class:`KeeperRefused`.

    ``nan`` compares false with everything, so a plain ``<= 0`` test lets it
    through (and ``Event.wait(nan)`` returns at once: a busy loop), and
    ``inf`` overflows ``Event.wait``.  Checked before anything is opened.
    """

    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= MAX_HEARTBEAT_SECONDS:
        raise KeeperRefused(
            f"--heartbeat-seconds must be a finite number in "
            f"(0, {MAX_HEARTBEAT_SECONDS:g}], got {value!r}"
        )
    return seconds


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int

    @classmethod
    def of(cls, path: Path) -> "FileIdentity":
        info = os.stat(path, follow_symlinks=False)
        return cls(device=info.st_dev, inode=info.st_ino)


def expected_schema_version() -> int:
    """The schema version this checkout's migrations produce (the API's too)."""

    return latest_version(load_migrations(MIGRATIONS_PATH))


def sidecar_paths(database: Path) -> tuple[Path, Path]:
    return tuple(  # type: ignore[return-value]
        database.with_name(database.name + suffix) for suffix in SIDECAR_SUFFIXES
    )


def read_only_uri(database: Path) -> str:
    """``mode=ro`` and nothing else: never ``immutable``, never ``nolock``."""

    return f"file:{quote(str(database))}?mode=ro"


def header_journal_is_wal(database: Path) -> bool:
    """Read the file header directly: WAL is read/write format version 2.

    Checked before SQLite opens the file, so a database that is not WAL is
    refused without a connection ever being made to it.
    """

    with database.open("rb") as handle:
        header = handle.read(20)
    if len(header) < 20 or header[:16] != SQLITE_MAGIC:
        raise KeeperRefused(f"{database} is not an SQLite 3 database")
    return header[18] == 2 and header[19] == 2


def check_database_path(raw: Optional[str]) -> Path:
    """The configured path, or :class:`KeeperRefused`.  Creates nothing."""

    if raw is None or not str(raw).strip():
        raise KeeperRefused(
            "no database path configured (--database or PHASE0_DATABASE_PATH)"
        )
    path = Path(str(raw))
    if not path.is_absolute():
        raise KeeperRefused(f"database path must be absolute, got {str(path)!r}")
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise KeeperRefused(
            f"no database at {path}; the keeper never creates one"
        ) from None
    if stat.S_ISLNK(info.st_mode):
        raise KeeperRefused(f"{path} is a symlink; configure the real database path")
    if not stat.S_ISREG(info.st_mode):
        raise KeeperRefused(f"{path} is not a regular file")
    return path


def _query(connection: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    """Run one statement and consume it completely, so nothing stays pinned."""

    cursor = connection.execute(sql)
    try:
        return cursor.fetchall()
    finally:
        cursor.close()


def _refuses_writes(connection: sqlite3.Connection) -> bool:
    """Whether a write through ``connection`` is refused, proved harmlessly.

    ``DELETE ... WHERE 0`` matches no row, so it cannot change anything even
    on a writable connection; on this one SQLite must refuse to run it at
    all.  The authorizer is not installed yet when this runs, so the refusal
    proved here is ``mode=ro``/``query_only`` itself.
    """

    try:
        _query(connection, "DELETE FROM schema_migrations WHERE 0")
    except sqlite3.OperationalError as exc:
        return "readonly" in str(exc)
    return False


@dataclass(frozen=True)
class ReadOnlyProbe:
    journal_mode: str
    write_refused: bool


def read_only_probe(database: Path) -> ReadOnlyProbe:
    """What an API-style connection sees and refuses, changing nothing.

    For ``tools/phase0_api_preflight.py``, which may not use SQLite directly
    (only :mod:`phase0` does).  The connection is opened exactly as the
    keeper's -- ``mode=ro``, ``query_only`` -- with no authorizer, so a
    refused write proves those two.  Raises ``sqlite3.Error`` if the
    database cannot be read at all; the caller must only call this once the
    coordination files exist, or a writable directory would get them.
    """

    connection = sqlite3.connect(
        read_only_uri(database), uri=True, timeout=10, isolation_level=None
    )
    try:
        _query(connection, "PRAGMA busy_timeout = 10000")
        _query(connection, "PRAGMA query_only = ON")
        journal_mode = str(_query(connection, "PRAGMA journal_mode")[0][0]).lower()
        return ReadOnlyProbe(
            journal_mode=journal_mode, write_refused=_refuses_writes(connection)
        )
    finally:
        connection.close()


class WalKeeper:
    """Hold the database open, read-only, for as long as the API serves it."""

    def __init__(
        self,
        database: Path,
        *,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.database = database
        self.heartbeat_seconds = check_heartbeat(heartbeat_seconds)
        self._log = log or log_event
        self._stop = threading.Event()
        self._connection: Optional[sqlite3.Connection] = None
        self.identity: Optional[FileIdentity] = None
        self.stop_reason: Optional[str] = None

    # -- Startup -----------------------------------------------------------

    def open(self) -> dict[str, Any]:
        """Validate, open and prove the connection; return the ready report."""

        expected = expected_schema_version()
        if not header_journal_is_wal(self.database):
            raise KeeperRefused(
                f"{self.database} is not in WAL mode; the pipeline's migrate() "
                "sets it, the keeper never changes it"
            )
        before = FileIdentity.of(self.database)
        connection = sqlite3.connect(
            read_only_uri(self.database), uri=True, timeout=10, isolation_level=None
        )
        try:
            _query(connection, "PRAGMA busy_timeout = 10000")
            _query(connection, "PRAGMA query_only = ON")
            if _query(connection, "PRAGMA query_only")[0][0] != 1:
                raise KeeperRefused("SQLite did not enable query_only")
            # The first read maps the WAL index: with the writer's directory
            # permissions, it is what provisions -wal and -shm.  It runs
            # before the write probe so a failed *read* can never pass for a
            # refused write.
            journal_mode = str(_query(connection, "PRAGMA journal_mode")[0][0]).lower()
            schema_version = int(_query(connection, "PRAGMA user_version")[0][0])
            if not _refuses_writes(connection):
                raise KeeperRefused("the keeper connection accepted a write")
            connection.set_authorizer(_read_only_authorizer)
            if journal_mode != "wal":
                raise KeeperRefused(f"journal_mode is {journal_mode!r}, expected 'wal'")
            if schema_version != expected:
                raise KeeperRefused(
                    f"schema version {schema_version}, expected {expected}; "
                    "the keeper never migrates"
                )
            if connection.in_transaction:
                raise KeeperRefused("the keeper connection is holding a transaction")
            after = FileIdentity.of(self.database)
            if after != before:
                raise KeeperRefused(
                    f"{self.database} was replaced while it was being opened"
                )
            missing = [str(p) for p in sidecar_paths(self.database) if not p.exists()]
            if missing:
                raise KeeperRefused(
                    f"WAL coordination files were not provisioned: {missing}"
                )
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self.identity = before
        return {
            "database": str(self.database),
            "device": before.device,
            "inode": before.inode,
            "journal_mode": journal_mode,
            "schema_version": schema_version,
            "query_only": True,
            "write_refused": True,
            "heartbeat_seconds": self.heartbeat_seconds,
        }

    # -- Running -----------------------------------------------------------

    def check(self) -> None:
        """Raise :class:`KeeperLost` if the path no longer names what we hold."""

        try:
            current = FileIdentity.of(self.database)
        except FileNotFoundError:
            raise KeeperLost(f"{self.database} disappeared") from None
        if current != self.identity:
            raise KeeperLost(
                f"{self.database} now names a different file "
                f"(inode {self.identity.inode} -> {current.inode})"
            )
        for sidecar in sidecar_paths(self.database):
            if not sidecar.exists():
                raise KeeperLost(
                    f"{sidecar} disappeared while the keeper held the database"
                )

    def request_stop(self, reason: str) -> None:
        self.stop_reason = self.stop_reason or reason
        self._stop.set()

    def run(self) -> int:
        """Open, then heartbeat until told to stop.  Returns the exit code."""

        try:
            ready = self.open()
        except (KeeperRefused, sqlite3.Error, OSError) as exc:
            self._log(
                "keeper_refused", database=str(self.database), reason=_reason(exc)
            )
            return EXIT_REFUSED
        self._log("keeper_ready", **ready)
        code = EXIT_STOPPED
        try:
            while not self._stop.wait(self.heartbeat_seconds):
                self.check()
                self._log(
                    "keeper_heartbeat",
                    database=str(self.database),
                    inode=self.identity.inode,
                )
        except KeeperLost as exc:
            self.stop_reason = _reason(exc)
            code = EXIT_LOST
        finally:
            self.close()
        self._log(
            "keeper_stopped",
            database=str(self.database),
            reason=self.stop_reason or "stopped",
            exit_code=code,
        )
        return code

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def _reason(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def log_event(event: str, **details: Any) -> None:
    """One redacted JSON line, the same shape ``pipeline.py`` logs."""

    payload = redact_secrets({"event": event, **details})
    LOGGER.info(json.dumps(payload, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hold the Phase 0 database's WAL files provisioned (read-only)"
    )
    parser.add_argument(
        "--database",
        default=os.getenv("PHASE0_DATABASE_PATH"),
        help="Absolute database path (default: $PHASE0_DATABASE_PATH; required)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help=(
            "Seconds between identity checks, finite, in "
            f"(0, {MAX_HEARTBEAT_SECONDS:g}] (default {DEFAULT_HEARTBEAT_SECONDS:g})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s", stream=sys.stderr
    )
    args = build_parser().parse_args(argv)
    try:
        heartbeat = check_heartbeat(args.heartbeat_seconds)
        database = check_database_path(args.database)
    except KeeperRefused as exc:
        log_event("keeper_refused", database=args.database, reason=_reason(exc))
        return EXIT_REFUSED
    keeper = WalKeeper(database, heartbeat_seconds=heartbeat)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(
            signum,
            lambda number, _frame: keeper.request_stop(signal.Signals(number).name),
        )
    return keeper.run()


if __name__ == "__main__":
    sys.exit(main())

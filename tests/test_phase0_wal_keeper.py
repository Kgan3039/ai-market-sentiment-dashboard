"""B3: the WAL keeper, across real process boundaries.

SQLite's WAL coordination files live and die with *process* lifetimes --
the last connection to close deletes them -- so every scenario here runs
the keeper, the scheduled writer, and the API reader as separate Python
processes against a temporary database built by the real ``migrate()``.

One OS user runs all of them, so the two deployment identities are
simulated with modes: ``api_view`` makes the directory ``0555`` and the
database and coordination files ``0444`` (what the API user sees on the
host), ``writer_view`` restores the writer's access around each pipeline
write.  That proves SQLite's behaviour under those modes; it does not prove
Linux ownership between two real users, which is the host smoke check in
``docs/phase0_deployment_handoff.md``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import phase0.wal_keeper as wal_keeper  # noqa: E402
from phase0.repository import Phase0Reader, Phase0Repository  # noqa: E402

VERSION = "phase0-v1"
DAY = "2026-09-25"

needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file and directory permissions",
)


# -- Processes ---------------------------------------------------------------


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "backend")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("PHASE0_DATABASE_PATH", None)
    return env


class Keeper:
    """``python -m phase0.wal_keeper`` as its own process, events parsed."""

    def __init__(
        self, database: Path, *, heartbeat: float = 0.1, args=None, extra_env=None
    ):
        argv = args if args is not None else ["--database", str(database)]
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "phase0.wal_keeper",
                *argv,
                # "=" form, so a value like "-inf" is not read as a flag.
                f"--heartbeat-seconds={heartbeat}",
            ],
            cwd=ROOT,
            env={**child_env(), **(extra_env or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.events: "queue.Queue[dict]" = queue.Queue()
        self.lines: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.process.stderr:
            self.lines.append(line)
            if line.startswith("{"):
                self.events.put(json.loads(line))

    def wait_event(self, name: str, timeout: float = 30) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.1)
            except queue.Empty:
                continue
            if event["event"] == name:
                return event
        raise AssertionError(f"no {name} event; stderr: {''.join(self.lines)}")

    def exit_code(self, timeout: float = 30) -> int:
        return self.process.wait(timeout=timeout)

    def terminate(self) -> int:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
        return self.exit_code()

    def kill(self) -> int:
        self.process.kill()
        return self.exit_code()


API_ACTOR = r"""
import json, sys
from pathlib import Path
from phase0.repository import Phase0Reader
db = Path(sys.argv[1])
for line in sys.stdin:
    cmd = line.strip()
    try:
        if cmd == "read":
            reader = Phase0Reader(db)
            out = {"ok": True, "schema": reader.schema_version(),
                   "runs": reader.count("run_log")}
        elif cmd == "status":
            from app.phase0.repository import NarrativeUnavailableError
            from app.phase0.sqlite_repository import SqliteNarrativeRepository
            repository = SqliteNarrativeRepository(
                database_path=db, pipeline_version=sys.argv[2])
            try:
                out = {"ok": True,
                       "data_as_of": repository.get_status().data_as_of.isoformat()}
            except NarrativeUnavailableError as exc:
                out = {"ok": False, "error": "unavailable",
                       "cause": str(exc.__cause__)}
        else:
            out = {"ok": False, "error": "unknown command"}
    except Exception as exc:
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out), flush=True)
"""


class Api:
    """A long-lived API-style reader process; every read opens per query."""

    def __init__(self, database: Path):
        self.process = subprocess.Popen(
            [sys.executable, "-c", API_ACTOR, str(database), VERSION],
            cwd=ROOT,
            env=child_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def ask(self, command: str) -> dict:
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=30)


WRITER = r"""
import sys, uuid
from pathlib import Path
from phase0.repository import Phase0Repository
repository = Phase0Repository(Path(sys.argv[1]))
with repository.stage_run(run_id=f"b3-{uuid.uuid4().hex}", stage="fetch_yahoo",
                          trading_day=sys.argv[2], pipeline_version=sys.argv[3],
                          ticker="TSLA"):
    pass
"""

CRASHING_WRITER = r"""
import os, signal, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("BEGIN IMMEDIATE")
connection.execute("CREATE TABLE crash_probe (x)")
connection.execute("INSERT INTO crash_probe VALUES (1)")
os.kill(os.getpid(), signal.SIGKILL)
"""


def write_run(database: Path) -> None:
    """One logged run, committed by a short-lived process, as cron does."""

    subprocess.run(
        [sys.executable, "-c", WRITER, str(database), DAY, VERSION],
        cwd=ROOT,
        env=child_env(),
        check=True,
        timeout=60,
    )


# -- Files -------------------------------------------------------------------


def make_database(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    database = directory / "phase0.sqlite3"
    Phase0Repository(database).migrate()
    return database


def sidecars(database: Path) -> list[Path]:
    return list(wal_keeper.sidecar_paths(database))


def entries(database: Path) -> set[str]:
    return set(os.listdir(database.parent))


def digest(database: Path) -> str:
    return hashlib.sha256(database.read_bytes()).hexdigest()


def api_view(database: Path) -> None:
    """What the API identity is given: read files, traverse, never write."""

    for path in [database, *sidecars(database)]:
        if path.exists():
            os.chmod(path, 0o444)
    os.chmod(database.parent, 0o555)


def writer_view(database: Path) -> None:
    os.chmod(database.parent, 0o755)
    for path in [database, *sidecars(database)]:
        if path.exists():
            os.chmod(path, 0o644)


@contextlib.contextmanager
def as_writer(database: Path):
    writer_view(database)
    try:
        yield
    finally:
        api_view(database)


@pytest.fixture
def cleanup():
    """Keepers, API processes and modes are always put back."""

    keepers: list[Keeper] = []
    apis: list[Api] = []
    databases: list[Path] = []
    yield keepers, apis, databases
    for api in apis:
        with contextlib.suppress(Exception):
            api.close()
    for keeper in keepers:
        with contextlib.suppress(Exception):
            keeper.kill()
    for database in databases:
        with contextlib.suppress(Exception):
            writer_view(database)


@pytest.fixture
def db(tmp_path, cleanup) -> Path:
    database = make_database(tmp_path / "live")
    cleanup[2].append(database)
    return database


def start_keeper(cleanup, database: Path, **kwargs) -> Keeper:
    keeper = Keeper(database, **kwargs)
    cleanup[0].append(keeper)
    return keeper


def start_api(cleanup, database: Path) -> Api:
    api = Api(database)
    cleanup[1].append(api)
    return api


# -- H: refusing to start ------------------------------------------------------


def _refused(keeper: Keeper) -> dict:
    event = keeper.wait_event("keeper_refused")
    assert keeper.exit_code() == wal_keeper.EXIT_REFUSED
    return event


def test_keeper_never_creates_a_missing_database(tmp_path, cleanup):
    database = tmp_path / "absent" / "phase0.sqlite3"
    database.parent.mkdir()
    event = _refused(start_keeper(cleanup, database))
    assert "never creates" in event["reason"]
    assert os.listdir(database.parent) == []


def test_keeper_requires_an_explicit_absolute_path(tmp_path, cleanup):
    event = _refused(start_keeper(cleanup, tmp_path, args=[]))
    assert "no database path configured" in event["reason"]
    event = _refused(
        start_keeper(cleanup, tmp_path, args=["--database", "data/phase0.sqlite3"])
    )
    assert "must be absolute" in event["reason"]


def test_keeper_refuses_a_symlinked_path(db, tmp_path, cleanup):
    link = tmp_path / "link.sqlite3"
    link.symlink_to(db)
    event = _refused(start_keeper(cleanup, link))
    assert "symlink" in event["reason"]


def test_keeper_refuses_a_non_wal_database_and_leaves_it_so(db, cleanup):
    with sqlite3.connect(db) as connection:
        connection.execute("PRAGMA journal_mode = DELETE")
    connection.close()
    before = entries(db)
    event = _refused(start_keeper(cleanup, db))
    assert "not in WAL mode" in event["reason"]
    assert not wal_keeper.header_journal_is_wal(db)  # never switched
    assert entries(db) == before  # no connection was made


def test_keeper_refuses_the_wrong_schema_and_never_migrates(db, cleanup):
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA user_version = 15")
    migrations = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()
    connection.close()
    event = _refused(start_keeper(cleanup, db))
    assert "schema version 15, expected 16" in event["reason"]
    connection = sqlite3.connect(db)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 15
    assert (
        connection.execute("SELECT count(*) FROM schema_migrations").fetchone()
        == migrations
    )
    connection.close()


def test_keeper_refuses_a_file_that_is_not_sqlite(tmp_path, cleanup):
    database = tmp_path / "phase0.sqlite3"
    database.write_bytes(b"not a database" * 10)
    event = _refused(start_keeper(cleanup, database))
    assert "not an SQLite 3 database" in event["reason"]


def test_expected_schema_is_the_migration_contract():
    from phase0.repository import MIGRATIONS_PATH
    from phase0.schema import latest_version, load_migrations

    assert wal_keeper.expected_schema_version() == 16
    assert wal_keeper.expected_schema_version() == latest_version(
        load_migrations(MIGRATIONS_PATH)
    )


# -- A: provisioning, and the keeper is not a writer ---------------------------


def test_keeper_provisions_the_wal_files_and_stays_up(db, cleanup):
    assert not any(path.exists() for path in sidecars(db))
    before = digest(db)
    keeper = start_keeper(cleanup, db)
    ready = keeper.wait_event("keeper_ready")
    assert ready["journal_mode"] == "wal"
    assert ready["schema_version"] == 16
    assert ready["query_only"] is True and ready["write_refused"] is True
    assert ready["inode"] == os.stat(db).st_ino
    assert all(path.exists() for path in sidecars(db))
    keeper.wait_event("keeper_heartbeat")
    assert keeper.process.poll() is None
    assert digest(db) == before
    assert sidecars(db)[0].stat().st_size == 0  # nothing was logged
    assert keeper.terminate() == wal_keeper.EXIT_STOPPED


def test_keeper_connection_refuses_every_write(db):
    migrations = Phase0Reader(db).count("schema_migrations")
    before = digest(db)
    keeper = wal_keeper.WalKeeper(db)
    keeper.open()
    connection = keeper._connection
    try:
        for statement in (
            "INSERT INTO schema_migrations (name, version, checksum, applied_at) "
            "VALUES ('x', 99, 'x', 'x')",
            "DELETE FROM schema_migrations",
            "CREATE TABLE intruder (x)",
            "PRAGMA query_only = OFF",
            "PRAGMA journal_mode = DELETE",
            "PRAGMA wal_checkpoint(TRUNCATE)",
            f"ATTACH DATABASE '{db}' AS alias",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(statement).fetchall()
        assert not connection.in_transaction
    finally:
        keeper.close()
    assert Phase0Reader(db).count("schema_migrations") == migrations
    assert digest(db) == before


def test_keeper_holds_no_snapshot_so_the_writer_can_checkpoint(db, cleanup):
    """A full TRUNCATE checkpoint needs every reader off the WAL.

    The writer succeeding at one while the keeper is up is the proof that
    the keeper pins no read transaction -- and that the pipeline's own
    checkpoints still bound the WAL while it runs.
    """

    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    for _ in range(20):
        write_run(db)
    connection = sqlite3.connect(db)
    busy, log_frames, checkpointed = connection.execute(
        "PRAGMA wal_checkpoint(TRUNCATE)"
    ).fetchone()
    connection.close()
    assert busy == 0 and log_frames == checkpointed
    assert sidecars(db)[0].stat().st_size == 0
    assert keeper.process.poll() is None


# -- B/C/D/E/F/G: the API across keeper and writer lifetimes -----------------


@needs_permissions
def test_api_before_keeper_fails_closed_and_creates_nothing(db, cleanup):
    write_run(db)
    api_view(db)
    before = entries(db)
    api = start_api(cleanup, db)
    read = api.ask("read")
    assert not read["ok"] and "readonly" in read["error"]
    status = api.ask("status")
    assert status == {**status, "ok": False, "error": "unavailable"}
    assert entries(db) == before


@needs_permissions
def test_api_reads_while_the_keeper_holds_and_sees_new_commits(db, cleanup):
    write_run(db)
    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    api_view(db)
    before = entries(db)
    api = start_api(cleanup, db)

    first = api.ask("read")
    assert first == {"ok": True, "schema": 16, "runs": 1}
    first_status = api.ask("status")
    assert first_status["ok"]

    time.sleep(0.01)
    with as_writer(db):
        write_run(db)  # a separate, short-lived writer process
    assert all(path.exists() for path in sidecars(db))  # its close kept them

    assert api.ask("read")["runs"] == 2  # same API process, no restart
    second_status = api.ask("status")
    assert second_status["data_as_of"] > first_status["data_as_of"]
    assert entries(db) == before


@needs_permissions
def test_keeper_restart_opens_a_failure_window_the_api_recovers_from(db, cleanup):
    write_run(db)
    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    api_view(db)
    api = start_api(cleanup, db)
    assert api.ask("read")["ok"]

    stopped = keeper.terminate()
    assert stopped == wal_keeper.EXIT_STOPPED
    assert "SIGTERM" in keeper.wait_event("keeper_stopped")["reason"]
    # A read-only keeper never deletes the files; the next writer to close
    # as the last connection does, and the window opens.
    with as_writer(db):
        write_run(db)
    assert not any(path.exists() for path in sidecars(db))
    failed = api.ask("read")
    assert not failed["ok"] and "readonly" in failed["error"]
    assert api.ask("status")["error"] == "unavailable"

    writer_view(db)
    restarted = start_keeper(cleanup, db)
    restarted.wait_event("keeper_ready")
    api_view(db)
    assert api.ask("read") == {"ok": True, "schema": 16, "runs": 2}


@needs_permissions
def test_a_restarted_api_process_reads_while_the_keeper_stays(db, cleanup):
    write_run(db)
    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    api_view(db)
    first = start_api(cleanup, db)
    assert first.ask("read")["ok"]
    first.close()
    second = start_api(cleanup, db)
    assert second.ask("read") == {"ok": True, "schema": 16, "runs": 1}
    assert second.ask("status")["ok"]
    assert keeper.process.poll() is None


@needs_permissions
def test_writer_and_keeper_crashes_lose_nothing_committed(db, cleanup):
    write_run(db)
    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    crashed = subprocess.run(
        [sys.executable, "-c", CRASHING_WRITER, str(db)],
        cwd=ROOT,
        env=child_env(),
        timeout=60,
    )
    assert crashed.returncode == -signal.SIGKILL
    api_view(db)
    api = start_api(cleanup, db)
    assert api.ask("read") == {"ok": True, "schema": 16, "runs": 1}

    assert keeper.kill() == -signal.SIGKILL
    writer_view(db)
    restarted = start_keeper(cleanup, db)
    restarted.wait_event("keeper_ready")
    api_view(db)
    assert api.ask("read") == {"ok": True, "schema": 16, "runs": 1}
    assert api.ask("status")["ok"]

    writer_view(db)
    connection = sqlite3.connect(db)
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'crash_probe'"
    ).fetchall()
    connection.close()


# -- I: the database identity ------------------------------------------------


def test_keeper_exits_when_the_database_is_replaced(db, tmp_path, cleanup):
    keeper = start_keeper(cleanup, db)
    ready = keeper.wait_event("keeper_ready")
    replacement = tmp_path / "live" / "restored.sqlite3"
    shutil.copyfile(db, replacement)
    os.replace(replacement, db)  # a restore: same path, new inode
    stopped = keeper.wait_event("keeper_stopped")
    assert keeper.exit_code() == wal_keeper.EXIT_LOST
    assert "different file" in stopped["reason"]
    assert str(ready["inode"]) in stopped["reason"]

    restarted = start_keeper(cleanup, db)
    assert restarted.wait_event("keeper_ready")["inode"] == os.stat(db).st_ino


def test_keeper_exits_when_the_database_disappears(db, cleanup):
    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    db.rename(db.with_name("moved.sqlite3"))
    assert "disappeared" in keeper.wait_event("keeper_stopped")["reason"]
    assert keeper.exit_code() == wal_keeper.EXIT_LOST


def test_keeper_exits_when_a_coordination_file_disappears(db, cleanup):
    keeper = start_keeper(cleanup, db)
    keeper.wait_event("keeper_ready")
    sidecars(db)[1].unlink()
    assert "-shm disappeared" in keeper.wait_event("keeper_stopped")["reason"]
    assert keeper.exit_code() == wal_keeper.EXIT_LOST


# -- K: no provider, no network, no credential --------------------------------


def test_keeper_imports_no_provider_or_network_client():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, phase0.wal_keeper; print('\\n'.join(sorted(sys.modules)))",
        ],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(probe.stdout.split())
    forbidden = {
        "google.genai",
        "requests",
        "httpx",
        "urllib3",
        "yfinance",
        "feedparser",
        "ai.summarization",
        "phase0.summary_runner",
        "phase0.yahoo",
        "phase0.rss",
        "pipeline",
    }
    assert not loaded & forbidden


def test_keeper_opens_with_the_network_refused(db, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the keeper reached for the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in list(os.environ):
        if name.startswith("GEMINI"):
            monkeypatch.delenv(name)
    keeper = wal_keeper.WalKeeper(db)
    try:
        assert keeper.open()["write_refused"] is True
    finally:
        keeper.close()


def test_keeper_needs_no_credential_and_logs_none(db, cleanup):
    secret = "AIza-B3-KEEPER-SECRET"
    keeper = start_keeper(cleanup, db, extra_env={"GEMINI_API_KEY": secret})
    keeper.wait_event("keeper_ready")
    assert keeper.terminate() == wal_keeper.EXIT_STOPPED
    assert secret not in "".join(keeper.lines)


# -- Heartbeat configuration ---------------------------------------------------


@pytest.mark.parametrize(
    "value", ["0", "-1", "-0.5", "nan", "NaN", "inf", "-inf", "300.001", "1e308"]
)
def test_invalid_heartbeat_is_refused_before_the_database_is_touched(
    db, cleanup, value
):
    before_entries, before_digest = entries(db), digest(db)
    keeper = start_keeper(cleanup, db, heartbeat=value)
    event = _refused(keeper)
    assert "--heartbeat-seconds must be a finite number in (0, 300]" in event["reason"]
    assert not any('"keeper_ready"' in line for line in keeper.lines)
    assert "Traceback" not in "".join(keeper.lines)
    assert entries(db) == before_entries  # no -wal, no -shm: SQLite never opened
    assert digest(db) == before_digest


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1])
def test_keeper_object_refuses_an_invalid_heartbeat(db, value):
    with pytest.raises(wal_keeper.KeeperRefused):
        wal_keeper.WalKeeper(db, heartbeat_seconds=value)
    assert not any(path.exists() for path in sidecars(db))


def test_heartbeat_range_and_default():
    assert wal_keeper.check_heartbeat(0.1) == 0.1
    assert wal_keeper.check_heartbeat(wal_keeper.MAX_HEARTBEAT_SECONDS) == 300.0
    default = wal_keeper.build_parser().parse_args([]).heartbeat_seconds
    assert default == wal_keeper.DEFAULT_HEARTBEAT_SECONDS == 30.0
    assert wal_keeper.check_heartbeat(default) == 30.0


def test_valid_small_heartbeat_runs_normally(db, cleanup):
    keeper = start_keeper(cleanup, db, heartbeat=0.05)
    assert keeper.wait_event("keeper_ready")["heartbeat_seconds"] == 0.05
    keeper.wait_event("keeper_heartbeat")
    assert keeper.terminate() == wal_keeper.EXIT_STOPPED

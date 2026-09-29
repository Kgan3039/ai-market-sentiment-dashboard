"""B3: the API preflight, the hard gate before ``PHASE0_NARRATIVE_SOURCE=sqlite``.

The live WAL state is provisioned the supported way -- a real
``phase0.wal_keeper`` process -- and the API identity's view is simulated
with modes (see ``tests/test_phase0_wal_keeper.py``).  One OS user runs the
suite, so the ownership check is exercised by handing ``run_preflight`` a
different effective uid; every ``os.access`` check still runs for real.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import phase0_api_preflight as preflight  # noqa: E402
from test_phase0_wal_keeper import (  # noqa: E402
    VERSION,
    Keeper,
    api_view,
    child_env,
    digest,
    entries,
    make_database,
    needs_permissions,
    sidecars,
    write_run,
    writer_view,
)
from phase0.repository import DEFAULT_DATABASE_PATH  # noqa: E402

#: Not this process's uid: the API identity must not own the database.
API_UID = os.geteuid() + 1
SECRET = "AIza-B3-PREFLIGHT-SECRET"

pytestmark = needs_permissions


@pytest.fixture
def live(tmp_path):
    """A migrated database with one completed run and the keeper holding it."""

    database = make_database(tmp_path / "live")
    write_run(database)
    keeper = Keeper(database)
    keeper.wait_event("keeper_ready")
    api_view(database)
    yield database
    keeper.kill()
    writer_view(database)


@pytest.fixture
def bare(tmp_path):
    """A migrated database nothing holds open: no coordination files."""

    database = make_database(tmp_path / "bare")
    yield database
    writer_view(database)


def env(database, **extra):
    values = {"PHASE0_PIPELINE_VERSION": VERSION, **extra}
    if database is not None:
        values["PHASE0_DATABASE_PATH"] = str(database)
    return values


def run(database, *, euid=API_UID, **extra):
    return preflight.run_preflight(env(database, **extra), euid=euid)


def failed(report) -> set[str]:
    return set(report.as_dict()["failed_checks"])


# -- PASS --------------------------------------------------------------------


def test_ready_when_the_keeper_holds_and_the_api_cannot_write(live, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the preflight reached for the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    before_entries, before_digest = entries(live), digest(live)

    report = run(live)
    payload = report.as_dict()

    assert payload["infrastructure_ready"] is True, payload["failed_checks"]
    assert payload["failed_checks"] == []
    checks = {c["check"] for c in payload["infrastructure"]}
    assert {
        "database_path_configured",
        "database_not_symlink",
        "directory_not_writable",
        "journal_mode_wal",
        "wal_present",
        "shm_present",
        "schema_version",
        "reader_can_query",
        "write_refused",
        "no_files_created",
    } <= checks
    assert payload["availability"]["completed_run"] is True
    assert payload["availability"]["status_route"] == "ok"
    assert set(payload["availability"]["tickers"]) == {
        "TSLA",
        "NVDA",
        "AMD",
        "AAPL",
        "META",
    }
    assert "data_as_of" in payload["freshness"]
    assert entries(live) == before_entries
    assert digest(live) == before_digest


# -- FAIL: configuration and path ----------------------------------------------


def test_unset_path_fails_and_the_checkout_default_is_never_used(tmp_path):
    existed = DEFAULT_DATABASE_PATH.exists()
    report = run(None)
    assert not report.ready
    assert "database_path_configured" in failed(report)
    assert DEFAULT_DATABASE_PATH.exists() == existed


def test_relative_path_fails():
    report = run("data/phase0.sqlite3")
    assert "database_path_absolute" in failed(report)


def test_missing_database_fails_and_is_never_created(tmp_path):
    database = tmp_path / "phase0.sqlite3"
    report = run(database)
    assert "database_exists" in failed(report)
    assert os.listdir(tmp_path) == []


def test_symlinked_path_fails(live, tmp_path):
    link = tmp_path / "link.sqlite3"
    link.symlink_to(live)
    assert "database_not_symlink" in failed(run(link))


def test_unknown_narrative_source_fails(live):
    report = run(live, PHASE0_NARRATIVE_SOURCE="sqlite-please")
    assert "narrative_source_valid" in failed(report)


# -- FAIL: schema and WAL ------------------------------------------------------


def test_wrong_schema_fails_and_nothing_migrates(tmp_path):
    database = make_database(tmp_path / "old")
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 15")
    connection.close()
    # The keeper refuses this database, so hold it with a plain writer-side
    # connection to provision the files and reach the schema check.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]);"
            "c.execute('SELECT 1 FROM sqlite_master').fetchall();"
            "print('up', flush=True); sys.stdin.read()",
            str(database),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "up"
        api_view(database)
        report = run(database)
        assert failed(report) == {"schema_version"}
    finally:
        holder.kill()
        holder.wait()
        writer_view(database)
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 15
    connection.close()


def test_non_wal_database_fails_and_is_not_switched(bare):
    connection = sqlite3.connect(bare)
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.close()
    api_view(bare)
    before = entries(bare)
    report = run(bare)
    assert {"journal_mode_wal", "wal_present", "shm_present"} <= failed(report)
    assert entries(bare) == before
    assert bare.read_bytes()[18:20] == b"\x01\x01"


@pytest.mark.parametrize("which", ["wal", "shm"])
def test_a_missing_coordination_file_fails(tmp_path, which):
    database = make_database(tmp_path / "live")
    keeper = Keeper(database)
    keeper.wait_event("keeper_ready")
    keeper.terminate()  # a read-only keeper leaves the files behind
    wal, shm = sidecars(database)
    (wal if which == "wal" else shm).unlink()
    api_view(database)
    try:
        report = run(database)
        assert failed(report) == {f"{which}_present"}
        assert not (wal if which == "wal" else shm).exists()
    finally:
        writer_view(database)


def test_no_keeper_fails_without_opening_sqlite(bare):
    """Even with a writable directory, the preflight provisions nothing."""

    before = entries(bare)
    report = run(bare)
    assert {"wal_present", "shm_present", "directory_not_writable"} <= failed(report)
    assert entries(bare) == before  # SQLite never opened: no -wal, no -shm


def test_unreadable_coordination_file_fails(live):
    shm = sidecars(live)[1]
    os.chmod(shm, 0o000)
    try:
        assert "shm_readable" in failed(run(live))
    finally:
        os.chmod(shm, 0o444)


# -- FAIL: the permission boundary ---------------------------------------------


def test_writable_directory_fails(live):
    os.chmod(live.parent, 0o755)
    before = entries(live)
    report = run(live)
    assert failed(report) == {"directory_not_writable"}
    assert entries(live) == before


def test_writable_database_or_coordination_file_fails(live):
    os.chmod(live, 0o644)
    os.chmod(sidecars(live)[0], 0o644)
    assert {"database_not_writable", "wal_not_writable"} <= failed(run(live))


def test_api_identity_that_owns_the_database_fails(live):
    assert {
        "directory_not_owned_by_api",
        "wal_not_owned_by_api",
        "shm_not_owned_by_api",
    } <= failed(run(live, euid=os.geteuid()))


def test_root_is_refused_rather_than_trusted(live):
    report = run(live, euid=0)
    assert failed(report) == {"not_root"}


def test_reader_that_cannot_query_fails(tmp_path):
    directory = tmp_path / "corrupt"
    directory.mkdir()
    database = directory / "phase0.sqlite3"
    header = bytearray(b"SQLite format 3\x00" + b"\x10\x00\x02\x02" + b"\x00" * 80)
    database.write_bytes(bytes(header) + b"\xff" * 8192)
    for sidecar in sidecars(database):
        sidecar.write_bytes(b"")
    api_view(database)
    try:
        report = run(database)
        assert not report.ready
        assert "reader_can_query" in failed(report)
    finally:
        writer_view(database)


# -- REPORTED, NOT FAILED --------------------------------------------------------


def test_no_completed_run_is_availability_not_infrastructure(tmp_path):
    database = make_database(tmp_path / "empty")
    keeper = Keeper(database)
    keeper.wait_event("keeper_ready")
    api_view(database)
    try:
        payload = run(database).as_dict()
    finally:
        keeper.kill()
        writer_view(database)
    assert payload["infrastructure_ready"] is True
    assert payload["availability"]["completed_run"] is False
    assert "503" in payload["availability"]["detail"]


def test_stale_data_is_freshness_not_infrastructure(live):
    # A weekday at noon New York time, long after the run completed.
    report = preflight.run_preflight(
        env(live),
        euid=API_UID,
        now_provider=lambda: datetime(2031, 3, 5, 17, 0, tzinfo=timezone.utc),
    )
    assert report.ready
    assert report.freshness["is_stale"] is True


def test_no_current_summary_is_reported_not_failed(live):
    payload = run(live).as_dict()
    assert payload["infrastructure_ready"] is True
    assert all(
        ticker.get("current_summaries", 0) == 0
        for ticker in payload["availability"]["tickers"].values()
    )


def test_policy_and_model_settings_are_reported_not_failed(live, monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-b3-test-model")
    report = run(live, GEMINI_MODEL="gemini-b3-test-model")
    assert report.ready
    policy = report.freshness["summary_policy"]
    assert policy["resolved"] is True
    assert policy["model"] == "gemini-b3-test-model"
    assert report.configuration["gemini_model_env"] == "gemini-b3-test-model"
    shown = report.as_dict()
    assert shown["freshness"]["summary_policy"]["output_cap"] == policy["output_cap"]
    assert "[REDACTED]" not in json.dumps(shown)


# -- The command line ------------------------------------------------------------


def test_cli_reports_json_and_exits_nonzero_without_leaking_the_key(live):
    completed = subprocess.run(
        [sys.executable, "tools/phase0_api_preflight.py"],
        cwd=ROOT,
        env={**child_env(), **env(live), "GEMINI_API_KEY": SECRET},
        capture_output=True,
        text=True,
        timeout=120,
    )
    # One OS user runs the suite, so the API identity owns the database here
    # and the real CLI must refuse.
    assert completed.returncode == preflight.EXIT_NOT_READY
    payload = json.loads(completed.stdout)
    assert payload["failed_checks"] == [
        "directory_not_owned_by_api",
        "wal_not_owned_by_api",
        "shm_not_owned_by_api",
    ]
    assert payload["configuration"]["api_env_has_gemini_access"] is True
    assert SECRET not in completed.stdout + completed.stderr


def test_main_exits_zero_when_ready(live, monkeypatch, capsys):
    monkeypatch.setattr(preflight.os, "geteuid", lambda: API_UID)
    for name, value in env(live).items():
        monkeypatch.setenv(name, value)
    assert preflight.main([]) == preflight.EXIT_READY
    assert json.loads(capsys.readouterr().out)["infrastructure_ready"] is True


def test_invalid_arguments_exit_two():
    completed = subprocess.run(
        [sys.executable, "tools/phase0_api_preflight.py", "--no-such-flag"],
        cwd=ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 2


def test_preflight_never_invokes_the_migrating_pipeline_reports():
    source = (ROOT / "tools" / "phase0_api_preflight.py").read_text()
    code = source.split('"""', 2)[2]  # ignore the docstring that names them
    for forbidden in ("--status", "--database-info", ".migrate(", "import pipeline"):
        assert forbidden not in code


# -- J: the default source ------------------------------------------------------


def test_fixture_remains_the_default_narrative_source(monkeypatch):
    import importlib

    monkeypatch.delenv("PHASE0_NARRATIVE_SOURCE", raising=False)
    config = importlib.import_module("app.config")
    assert config.Settings(_env_file=None).PHASE0_NARRATIVE_SOURCE == "fixture"


# -- Pipeline version configuration ----------------------------------------------


def _no_availability(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("availability was queried without a valid version")

    monkeypatch.setattr(preflight, "check_availability_and_freshness", refuse)
    monkeypatch.setattr(preflight, "_policy_report", refuse)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_env_pipeline_version_is_a_named_failure(live, monkeypatch, blank):
    _no_availability(monkeypatch)
    before_entries, before_digest = entries(live), digest(live)
    report = preflight.run_preflight(
        {**env(live), "PHASE0_PIPELINE_VERSION": blank}, euid=API_UID
    )
    payload = report.as_dict()
    assert payload["infrastructure_ready"] is False
    assert payload["failed_checks"] == ["pipeline_version_valid"]
    assert payload["configuration"]["pipeline_version"] is None
    assert payload["availability"] == {} and payload["freshness"] == {}
    assert entries(live) == before_entries and digest(live) == before_digest


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_explicit_pipeline_version_is_not_replaced(live, monkeypatch, blank):
    """An explicit blank never falls back to the env value or the default."""

    _no_availability(monkeypatch)
    report = preflight.run_preflight(env(live), pipeline_version=blank, euid=API_UID)
    assert failed(report) == {"pipeline_version_valid"}
    detail = next(
        c["detail"]
        for c in report.infrastructure
        if c["check"] == "pipeline_version_valid"
    )
    assert "--pipeline-version is set but blank" in detail


def test_unset_pipeline_version_uses_the_api_default(live):
    values = env(live)
    del values["PHASE0_PIPELINE_VERSION"]
    report = preflight.run_preflight(values, euid=API_UID)
    assert report.ready
    assert report.configuration["pipeline_version"] == "phase0-v1"
    assert report.availability["completed_run"] is True


def test_pipeline_version_is_normalized_like_the_read_path(live):
    report = preflight.run_preflight(
        {**env(live), "PHASE0_PIPELINE_VERSION": f"  {VERSION}  "}, euid=API_UID
    )
    assert report.ready
    assert report.configuration["pipeline_version"] == VERSION
    assert report.availability["completed_run"] is True


@pytest.mark.parametrize(
    "argv, environment",
    [
        ([], {"PHASE0_PIPELINE_VERSION": ""}),
        ([], {"PHASE0_PIPELINE_VERSION": "   "}),
        (["--pipeline-version", ""], {}),
        (["--pipeline-version", "   "], {}),
    ],
)
def test_cli_blank_pipeline_version_emits_json_without_a_traceback(
    live, argv, environment
):
    before_entries, before_digest = entries(live), digest(live)
    completed = subprocess.run(
        [sys.executable, "tools/phase0_api_preflight.py", *argv],
        cwd=ROOT,
        env={**child_env(), **env(live), **environment},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == preflight.EXIT_NOT_READY
    assert "Traceback" not in completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert "pipeline_version_valid" in payload["failed_checks"]
    assert payload["availability"] == {} and payload["freshness"] == {}
    assert entries(live) == before_entries and digest(live) == before_digest

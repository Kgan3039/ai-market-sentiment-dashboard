#!/usr/bin/env python3
"""B3: the hard gate before an API is switched to ``PHASE0_NARRATIVE_SOURCE=sqlite``.

Run it **as the API service user**, with the environment the API will run
with::

    sudo -u <api-user> env \\
        PHASE0_DATABASE_PATH=<the absolute database path> \\
        PYTHONPATH=/opt/ticker-narratives:/opt/ticker-narratives/backend \\
        /opt/ticker-narratives/.venv/bin/python tools/phase0_api_preflight.py

It prints one JSON document and exits:

* ``0`` -- **infrastructure ready**: the API identity can read the live WAL
  database through the B1 read path and cannot create, change or remove
  anything in the database directory;
* ``1`` -- **not ready**: at least one infrastructure check failed (each is
  named in ``infrastructure``);
* ``2`` -- the command line itself was invalid.

Three things are reported separately and only the first decides the exit
code, because they fail for different reasons and are fixed by different
people:

* ``infrastructure`` -- path, file, schema, WAL, coordination files,
  permissions, readability, write refusal;
* ``availability`` -- whether the pipeline has completed a run for this
  pipeline version, and what the API would serve per ticker;
* ``freshness`` -- ``data_as_of`` and ``is_stale``, exactly as the API
  computes them, plus the summary policy that decides currentness.

**It is read-only.**  It never calls ``pipeline.py --status`` or
``--database-info`` (both migrate), never opens SQLite until every
filesystem precondition has passed -- so it cannot be the thing that
creates ``-wal``/``-shm`` -- and every connection it makes is ``mode=ro``
with ``query_only``.  The write-refusal probe is ``DELETE ... WHERE 0``,
which matches no row even where writes are allowed.  It needs no
``GEMINI_API_KEY`` and makes no network request; it reports whether the
API environment carries a Gemini key, never the key.

It refuses to judge permissions as root: root bypasses them, so a pass
would say nothing about the API user.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "backend", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from phase0.redaction import redact_secrets  # noqa: E402
from phase0.errors import Phase0Error  # noqa: E402
from phase0.repository import DATABASE_READ_ERRORS, Phase0Reader  # noqa: E402
from phase0.tickers import TICKER_UNIVERSE  # noqa: E402
from phase0.wal_keeper import (  # noqa: E402
    SQLITE_MAGIC,
    expected_schema_version,
    read_only_probe,
    sidecar_paths,
)

EXIT_READY = 0
EXIT_NOT_READY = 1

DEFAULT_PIPELINE_VERSION = "phase0-v1"
NARRATIVE_SOURCES = ("fixture", "sqlite")


@dataclass
class Report:
    infrastructure: list[dict[str, Any]] = field(default_factory=list)
    availability: dict[str, Any] = field(default_factory=dict)
    freshness: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.infrastructure.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ready(self) -> bool:
        return bool(self.infrastructure) and all(c["ok"] for c in self.infrastructure)

    def as_dict(self) -> dict[str, Any]:
        return redact_secrets(
            {
                "infrastructure_ready": self.ready,
                "failed_checks": [
                    c["check"] for c in self.infrastructure if not c["ok"]
                ],
                "infrastructure": self.infrastructure,
                "availability": self.availability,
                "freshness": self.freshness,
                "configuration": self.configuration,
            }
        )


def _can(path: Path, mode: int) -> bool:
    effective = os.access in os.supports_effective_ids
    return os.access(path, mode, effective_ids=effective)


def _entries(directory: Path) -> set[str]:
    return set(os.listdir(directory))


def _journal_is_wal(database: Path) -> tuple[bool, str]:
    with database.open("rb") as handle:
        header = handle.read(20)
    if len(header) < 20 or header[:16] != SQLITE_MAGIC:
        return False, "not an SQLite 3 database"
    if header[18] == 2 and header[19] == 2:
        return True, "header: WAL"
    return (
        False,
        "header: not WAL (the pipeline's migrate() sets WAL; preflight never does)",
    )


def check_infrastructure(
    report: Report,
    raw_path: Optional[str],
    *,
    euid: int,
) -> Optional[Path]:
    """The filesystem and SQLite checks.  Returns the path if all passed."""

    if not report.check(
        "not_root",
        euid != 0,
        "run as the API service user; root bypasses the permission checks"
        if euid == 0
        else f"euid {euid}",
    ):
        return None
    configured = raw_path is not None and bool(str(raw_path).strip())
    if not report.check(
        "database_path_configured",
        configured,
        "PHASE0_DATABASE_PATH (or --database) must be set explicitly for SQLite mode"
        if not configured
        else "",
    ):
        return None
    database = Path(str(raw_path))
    if not report.check(
        "database_path_absolute", database.is_absolute(), str(database)
    ):
        return None
    directory = database.parent
    if not report.check(
        "database_directory_exists", directory.is_dir(), str(directory)
    ):
        return None
    try:
        info = os.lstat(database)
    except FileNotFoundError:
        report.check("database_exists", False, f"no database at {database}")
        return None
    except OSError as exc:
        report.check("database_exists", False, f"cannot stat {database}: {exc}")
        return None
    report.check("database_exists", True, str(database))
    if not report.check(
        "database_not_symlink",
        not stat.S_ISLNK(info.st_mode),
        "configure the real database path, not a symlink",
    ):
        return None
    if not report.check("database_regular_file", stat.S_ISREG(info.st_mode), ""):
        return None

    # -- The permission boundary (before anything opens SQLite) ------------
    directory_info = os.stat(directory)
    ok = True
    ok &= report.check(
        "directory_not_writable",
        not _can(directory, os.W_OK),
        "the API identity must not be able to create or remove files in "
        f"{directory}",
    )
    ok &= report.check(
        "directory_not_owned_by_api",
        directory_info.st_uid != euid and info.st_uid != euid,
        "the API identity must not own the database or its directory "
        "(an owner can chmod its way to write access)",
    )
    ok &= report.check("directory_traversable", _can(directory, os.X_OK), "")
    ok &= report.check("database_readable", _can(database, os.R_OK), "")
    ok &= report.check(
        "database_not_writable", not _can(database, os.W_OK), "API must not write it"
    )
    try:
        wal, detail = _journal_is_wal(database)
    except OSError as exc:
        wal, detail = False, f"cannot read header: {exc}"
    ok &= report.check("journal_mode_wal", wal, detail)
    for sidecar in sidecar_paths(database):
        name = "wal" if sidecar.name.endswith("-wal") else "shm"
        try:
            sidecar_info = os.lstat(sidecar)
        except FileNotFoundError:
            report.check(
                f"{name}_present",
                False,
                f"{sidecar} missing: is the WAL keeper running as the writer identity?",
            )
            ok = False
            continue
        report.check(f"{name}_present", True, str(sidecar))
        ok &= report.check(
            f"{name}_regular_file",
            stat.S_ISREG(sidecar_info.st_mode),
            "must be a regular file, not a symlink",
        )
        ok &= report.check(f"{name}_readable", _can(sidecar, os.R_OK), "")
        ok &= report.check(
            f"{name}_not_writable", not _can(sidecar, os.W_OK), "API must not write it"
        )
        ok &= report.check(
            f"{name}_not_owned_by_api",
            sidecar_info.st_uid != euid,
            "the writer provisions it; one the API created is the wrong topology",
        )
    if not ok:
        # Nothing opens SQLite until every filesystem precondition holds, so
        # a failing preflight can never be what provisions the sidecars.
        return None

    # -- SQLite, read-only --------------------------------------------------
    before = _entries(directory)
    try:
        expected = expected_schema_version()
        found = Phase0Reader(database).schema_version()
        report.check(
            "schema_version", found == expected, f"found {found}, expected {expected}"
        )
        report.check(
            "reader_can_query",
            Phase0Reader(database).count("run_log") >= 0,
            "Phase0Reader.count('run_log')",
        )
        # mode=ro and query_only, no authorizer; DELETE ... WHERE 0 could
        # not change a row even if it were allowed to run.
        probe = read_only_probe(database)
        report.check(
            "sqlite_journal_mode_wal",
            probe.journal_mode == "wal",
            f"PRAGMA journal_mode={probe.journal_mode}",
        )
        report.check(
            "write_refused",
            probe.write_refused,
            "write refused (mode=ro, query_only)"
            if probe.write_refused
            else "an API-style connection did not refuse a write statement",
        )
    except (*DATABASE_READ_ERRORS, Phase0Error, OSError, ValueError) as exc:
        report.check("reader_can_query", False, f"{type(exc).__name__}: {exc}")
    created = sorted(_entries(directory) - before)
    report.check(
        "no_files_created",
        not created,
        f"preflight reads created {created}" if created else "",
    )
    return database if report.ready else None


def _policy_report() -> dict[str, Any]:
    from phase0.summary_runner import production_generation_policy

    try:
        policy = production_generation_policy()
    except Exception as exc:  # reported, never fatal: reads serve degraded
        return {
            "resolved": False,
            "reason": type(exc).__name__,
            "effect": "every theme would be served degraded",
        }
    return {
        "resolved": True,
        "model": policy.model,
        "output_cap": policy.max_output_tokens,
        "fingerprint": policy.fingerprint,
    }


def check_availability_and_freshness(
    report: Report,
    database: Path,
    pipeline_version: str,
    *,
    now_provider: Callable[[], datetime],
) -> None:
    """What the API would serve.  Reported; never part of the exit code."""

    from app.phase0.repository import NarrativeUnavailableError
    from app.phase0.sqlite_repository import SqliteNarrativeRepository

    reader = Phase0Reader(database)
    completed = None
    for statuses in (("success",), ("degraded",)):
        completed = reader.latest_run_completion(pipeline_version, statuses)
        if completed is not None:
            break
    report.availability["completed_run"] = completed is not None
    if completed is None:
        report.availability["detail"] = (
            f"no completed run for pipeline version {pipeline_version!r}: "
            "the narrative routes will answer 503 until the pipeline completes one"
        )
        return

    repository = SqliteNarrativeRepository(
        database_path=database,
        pipeline_version=pipeline_version,
        now_provider=now_provider,
    )
    try:
        status = repository.get_status()
    except NarrativeUnavailableError:
        report.availability["status_route"] = "unavailable"
        return
    report.availability["status_route"] = "ok"
    report.freshness["data_as_of"] = status.data_as_of.isoformat()
    report.freshness["is_stale"] = status.is_stale
    tickers: dict[str, Any] = {}
    for ticker in TICKER_UNIVERSE:
        try:
            payload = repository.get_themes(ticker, None)
        except NarrativeUnavailableError:
            tickers[ticker] = {"themes_route": "unavailable"}
            continue
        tickers[ticker] = {
            "themes_route": "ok",
            "date": str(payload.date),
            "themes": len(payload.themes),
            "current_summaries": sum(not theme.degraded for theme in payload.themes),
            "other_coverage_stories": payload.other_coverage.story_count,
        }
    report.availability["tickers"] = tickers


def resolve_pipeline_version(
    explicit: Optional[str], environ: Mapping[str, str]
) -> tuple[Optional[str], str]:
    """The pipeline version the API would read with, or ``None`` if invalid.

    Precedence is by presence, not truthiness: an explicit ``--pipeline-version``
    wins even when blank, and a set-but-blank ``PHASE0_PIPELINE_VERSION`` is
    not replaced by the default -- the API's settings would use it as given.
    Only an unset variable means the default, as in ``backend/app/config.py``.
    The value is normalized exactly as the read path normalizes it
    (``SqliteNarrativeRepository`` and ``phase0.repository``'s required-text
    rule: strip, then require non-empty).  Returns ``(version, source)``.
    """

    if explicit is not None:
        raw, origin = explicit, "--pipeline-version"
    elif "PHASE0_PIPELINE_VERSION" in environ:
        raw, origin = environ["PHASE0_PIPELINE_VERSION"], "PHASE0_PIPELINE_VERSION"
    else:
        return DEFAULT_PIPELINE_VERSION, "default"
    version = str(raw).strip()
    return (version or None), origin


def run_preflight(
    environ: Mapping[str, str],
    *,
    database: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    euid: Optional[int] = None,
    now_provider: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Report:
    report = Report()
    raw_path = database if database is not None else environ.get("PHASE0_DATABASE_PATH")
    version, version_origin = resolve_pipeline_version(pipeline_version, environ)
    source = environ.get("PHASE0_NARRATIVE_SOURCE", "fixture")
    report.configuration = {
        "narrative_source_env": source,
        "pipeline_version": version,
        "database": raw_path,
        "gemini_model_env": environ.get("GEMINI_MODEL"),
        # Key names avoid the redactor's credential words ("token", "key"),
        # which would otherwise blank these non-secret values.
        "gemini_output_cap_env": environ.get("GEMINI_MAX_OUTPUT_TOKENS"),
        # Presence only, never the value.  The API needs no Gemini
        # credential and should not be given one; reported, not failed.
        "api_env_has_gemini_access": bool(environ.get("GEMINI_API_KEY")),
    }
    report.check(
        "narrative_source_valid",
        source in NARRATIVE_SOURCES,
        f"PHASE0_NARRATIVE_SOURCE={source!r}; expected one of {NARRATIVE_SOURCES}",
    )
    # Configuration: which persisted population the API would serve.  A
    # blank version fails here; the infrastructure checks still run and are
    # reported, but availability is never queried without a valid version.
    report.check(
        "pipeline_version_valid",
        version is not None,
        f"pipeline version from {version_origin}"
        if version is not None
        else f"{version_origin} is set but blank; set the scheduler's version",
    )
    checked = check_infrastructure(
        report, raw_path, euid=os.geteuid() if euid is None else euid
    )
    if checked is None or version is None:
        return report
    report.freshness["summary_policy"] = _policy_report()
    check_availability_and_freshness(
        report, checked, version, now_provider=now_provider
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only readiness gate for serving Phase 0 narratives from SQLite"
        )
    )
    parser.add_argument(
        "--database", help="Database path (default: $PHASE0_DATABASE_PATH; required)"
    )
    parser.add_argument(
        "--pipeline-version",
        help=(
            "Pipeline version (default: $PHASE0_PIPELINE_VERSION or "
            f"{DEFAULT_PIPELINE_VERSION})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        os.environ, database=args.database, pipeline_version=args.pipeline_version
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str))
    return EXIT_READY if report.ready else EXIT_NOT_READY


if __name__ == "__main__":
    sys.exit(main())

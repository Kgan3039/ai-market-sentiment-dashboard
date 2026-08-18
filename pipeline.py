#!/usr/bin/env python3
"""Phase 0 pipeline orchestration (issue #68).

This module calls components.  It does not persist anything.

The distinction is the whole design.  I1 made every durable write happen
inside a run that names exactly one partition -- one stage, one ticker,
one trading day, one pipeline version -- and I2 and I3 settle their own
partitions against that contract.  An orchestrator that also wrote run
rows would be inventing a second, weaker audit beside the authoritative
one, so this module writes none: no ``log_stage``, no ``set_source_state``,
no ``insert_raw_items``, no connection, no ``run_log`` row of its own.
What it produces is a process-level summary, and it says so in its own
vocabulary rather than borrowing the repository's.

Examples::

    python pipeline.py
    python pipeline.py --database /var/lib/ticker-narratives/phase0.db
    python pipeline.py --replay
    python pipeline.py --status
    python pipeline.py --database-info
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from zoneinfo import ZoneInfo

from phase0.repository import (
    DEFAULT_DATABASE_PATH,
    Phase0Repository,
    redact_secrets,
)
from phase0.rss import RSSFetcher
from phase0.yahoo import YahooFinanceFetcher


ROOT = Path(__file__).resolve().parent
DEFAULT_FEEDS = ROOT / "config" / "feeds.yaml"
DEFAULT_ALIASES = ROOT / "config" / "aliases.yaml"
PIPELINE_VERSION = os.getenv("PHASE0_PIPELINE_VERSION", "phase0-v1")

# Used for the invocation's own date label and for nothing else.  See
# ``invocation_day``.
MARKET_TIMEZONE = ZoneInfo("America/New_York")

# One code per outcome, because one bit cannot carry three answers.  A
# degraded invocation persisted real evidence and must not be read as a
# clean run; a failed one must not be read as a partial success.
EXIT_CODES = {"success": 0, "degraded": 1, "failed": 2, "skipped": 0}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
)
LOGGER = logging.getLogger("phase0.pipeline")


# -- Invocation identity -------------------------------------------------


def new_invocation_id() -> str:
    """A correlation id for one execution of this file.

    **This is not a repository run id and never becomes one.**  It is
    handed to a component as a *base*, and the component derives its own
    partition identities from it -- ``partition_run_id`` in both
    :mod:`phase0.yahoo` and :mod:`phase0.rss` appends the partition to the
    base before anything is recorded.  Nothing ever opens a run under the
    bare base, which is what keeps ``run_log``'s ``UNIQUE(run_id, stage)``
    meaning "one partition" rather than "one process".
    """

    return f"phase0-{uuid.uuid4()}"


def invocation_day(now: datetime | None = None) -> str:
    """The America/New_York date this invocation started.

    A label for logs, CLI output, and operators -- deliberately *not* a
    partition.  Evidence gets its day from its own timestamps: I2 and I3
    both dropped their ``trading_day`` arguments because the repository
    derives each item's day and refuses a batch that disagrees with its
    run, so a day announced by the scheduler could only ever be ignored or
    fatal.  A fetch that starts at 23:55 and returns yesterday's article
    stores it under yesterday, whatever this function says.

    Host local time is not consulted.  A UTC instant late in the evening
    belongs to the previous Eastern date, and the offset moves with EST/EDT
    rather than being assumed.
    """

    moment = datetime.now(timezone.utc) if now is None else now
    if moment.tzinfo is None:
        raise ValueError("invocation_day requires an aware datetime")
    return moment.astimezone(MARKET_TIMEZONE).date().isoformat()


# -- Component results ---------------------------------------------------


@dataclass(frozen=True)
class ComponentResult:
    """What one component reported, kept whole.

    The counters and errors are the component's own summary, copied rather
    than reduced.  Collapsing them into a status here and discarding the
    rest is exactly how a partial Yahoo day comes to look like a clean one.
    """

    name: str
    status: str
    counts: dict[str, Any]
    errors: list[Any]
    duration_ms: int
    mandatory: bool
    run_id_base: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.name,
            "status": self.status,
            "counts": dict(self.counts),
            "errors": list(self.errors),
            "duration_ms": self.duration_ms,
            "mandatory": self.mandatory,
            "run_id_base": self.run_id_base,
        }


@dataclass(frozen=True)
class InvocationResult:
    """The process-level aggregate.  Held in memory, logged, never stored.

    There is no pipeline-level audit table in the final schema, and faking
    one into ``run_log`` would mean writing a row whose ``run_id`` names no
    partition -- the one thing I1's identity rule exists to prevent.  Until
    a product requirement justifies real invocation-level schema, this
    object plus the structured log *is* the invocation record, and the
    per-partition ``run_log`` rows the components wrote remain the durable
    truth about execution.
    """

    invocation_id: str
    mode: str
    status: str
    started_at: str
    completed_at: str
    duration_ms: int
    invocation_day: str
    pipeline_version: str
    components: tuple[ComponentResult, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    def as_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "mode": self.mode,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "invocation_day": self.invocation_day,
            "pipeline_version": self.pipeline_version,
            "components": [result.as_dict() for result in self.components],
            **self.detail,
        }


@dataclass(frozen=True)
class Stage:
    """One orchestrated component and how to read its answer.

    ``action`` receives the base run id for this invocation and returns the
    component's own ``(counts, errors)``.  Opening runs, writing evidence,
    and settling partitions all happen inside the component; this record
    only says which one to call.

    ``settled`` and ``unsettled`` name the counters that say whether
    anything durable came of the call.  They differ per component because
    the components count different things -- tickers and feeds are not
    interchangeable -- and naming them here is what lets one status rule
    serve both without pretending their counters are the same.
    """

    name: str
    action: Callable[[str], tuple[dict[str, Any], list[Any]]]
    settled: tuple[str, ...] = ()
    unsettled: tuple[str, ...] = ()
    mandatory: bool = True


# M1--M5 register here as they land.  The tuple is empty on purpose: the
# CLI reads it to report what can and cannot be replayed, so an aspirational
# entry would become a false claim rather than a to-do.
DOWNSTREAM_STAGES: tuple[Stage, ...] = ()


def component_status(
    counts: dict[str, Any],
    errors: Sequence[Any],
    *,
    settled: Sequence[str] = (),
    unsettled: Sequence[str] = (),
) -> str:
    """Status for one component, from what it actually settled.

    ``degraded`` means evidence was persisted *and* something went wrong --
    the case that must not round to either neighbour.  ``failed`` is
    reserved for a component that settled nothing, so a run that stored
    four tickers out of five is never reported as a failure, and one that
    stored none is never reported as a partial success.

    A component reporting no targets and no errors is a success: there was
    nothing to do, and inventing a failure from an empty feed list would
    make an idle schedule look broken.
    """

    settled_total = sum(int(counts.get(key, 0)) for key in settled)
    unsettled_total = sum(int(counts.get(key, 0)) for key in unsettled)
    if not errors and not unsettled_total:
        return "success"
    if settled_total:
        return "degraded"
    return "failed"


def invocation_status(components: Sequence[ComponentResult]) -> str:
    """Status for the invocation, from its components.

    Mandatory components decide failure: the invocation is ``failed`` only
    when every one of them settled nothing, because that is the case where
    no part of the result can be trusted.  Anything between that and total
    success is ``degraded`` -- one source down while another persisted a
    full day is a real, usable, incomplete result, and calling it either
    "success" or "failed" would misreport it.
    """

    if not components:
        return "success"
    if all(result.status == "success" for result in components):
        return "success"
    mandatory = [result for result in components if result.mandatory] or list(
        components
    )
    if all(result.status == "failed" for result in mandatory):
        return "failed"
    return "degraded"


# -- Structured logging --------------------------------------------------


def _log_event(event: str, **details: Any) -> None:
    """Emit one redacted structured line.

    Redaction is I1's ``redact_secrets``, reused rather than reimplemented,
    and applied to the whole payload -- component errors arrive already
    redacted, but an error this module built from an exception has not been
    through it yet, and a second pass over clean data costs nothing.
    """

    payload = redact_secrets({"event": event, **details})
    LOGGER.info(json.dumps(payload, sort_keys=True, default=str))


# -- Single-instance execution -------------------------------------------


@contextlib.contextmanager
def single_instance(lock_path: Path) -> Iterator[bool]:
    """Hold an exclusive lock for the duration of one invocation.

    Every schedule in ``deploy/phase0-pipeline.cron`` fires on a fixed
    interval, and cron will happily start a second copy while the first is
    still running -- two live fetchers then race for the same feeds and the
    same provider slots.  The lock is advisory, non-blocking, and
    process-scoped: a second invocation is told the answer immediately and
    exits, rather than queueing behind a run that may itself be wedged.

    Yields ``True`` when this process holds the lock and ``False`` when
    another one already does.  The lock releases with the file descriptor,
    so a crashed or killed invocation does not leave it held.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


# -- Stage execution -----------------------------------------------------


def execute_stage(stage: Stage, *, invocation_id: str) -> ComponentResult:
    """Run one component and bring back its answer, whatever happened.

    An exception never leaves this function.  Failure isolation is the
    point: evidence another component already committed is durable, and a
    crash in one source must not stop the next from running or unwind what
    is already stored.  There is no cross-source transaction to unwind.
    """

    base_run_id = f"{invocation_id}:{stage.name}"
    started = time.monotonic()
    try:
        counts, errors = stage.action(base_run_id)
        status = component_status(
            counts, errors, settled=stage.settled, unsettled=stage.unsettled
        )
    except Exception as exc:  # noqa: BLE001 - isolation is the contract here
        counts = {}
        errors = [
            {"type": "component_error", "component": stage.name, "error": str(exc)}
        ]
        status = "failed"
    duration_ms = round((time.monotonic() - started) * 1000)
    result = ComponentResult(
        name=stage.name,
        status=status,
        counts=dict(counts),
        errors=list(redact_secrets(errors)),
        duration_ms=duration_ms,
        mandatory=stage.mandatory,
        run_id_base=base_run_id,
    )
    _log_event(
        "component_completed",
        invocation_id=invocation_id,
        **result.as_dict(),
    )
    return result


def _refuse_network(*args: Any, **kwargs: Any) -> Any:
    """The HTTP callable a replay fetcher is built with.

    Replay reads persisted evidence and nothing else.  Making that
    structural rather than documentary means a future edit that reaches for
    the network during replay fails loudly here instead of quietly
    refetching.
    """

    raise RuntimeError("replay must not fetch; it reads persisted evidence only")


# -- Component stages ----------------------------------------------------
#
# Each builder returns a stage whose action *constructs* its component and
# then calls it.  Construction is deliberately inside the action rather
# than before the stage list, because both constructors do real work that
# can fail: ``YahooFinanceFetcher`` validates its arguments, and
# ``RSSFetcher`` reads and validates ``feeds.yaml`` and ``aliases.yaml``.
# Building them eagerly put that work outside ``execute_stage``, where a
# YAML typo did not fail one component -- it took the whole invocation
# down before Yahoo had run at all, and left the process with a traceback
# instead of an exit code.


def yahoo_stage(repository: Phase0Repository, *, pipeline_version: str) -> Stage:
    """The Yahoo component, built when it runs rather than before."""

    def action(base_run_id: str) -> tuple[dict[str, Any], list[Any]]:
        fetcher = YahooFinanceFetcher(repository, pipeline_version=pipeline_version)
        return fetcher.fetch(run_id=base_run_id)

    return Stage(
        "yahoo",
        action,
        settled=("tickers_succeeded", "tickers_partial", "tickers_empty"),
        unsettled=("tickers_failed", "tickers_rejected"),
    )


def rss_stage(
    repository: Phase0Repository,
    *,
    feeds_path: Path,
    aliases_path: Path,
    pipeline_version: str,
) -> Stage:
    """The RSS component, built when it runs rather than before.

    A missing or malformed ``feeds.yaml``/``aliases.yaml`` is now this
    component's own failure: RSS is recorded ``failed``, Yahoo still runs,
    and the invocation reports ``degraded`` with an exit code rather than
    an uncaught traceback.
    """

    def action(base_run_id: str) -> tuple[dict[str, Any], list[Any]]:
        fetcher = RSSFetcher(
            repository,
            feeds_path=feeds_path,
            aliases_path=aliases_path,
            pipeline_version=pipeline_version,
        )
        return fetcher.fetch(run_id=base_run_id)

    return Stage(
        "rss",
        action,
        settled=("feeds_succeeded", "feeds_partial", "feeds_not_modified"),
        unsettled=("feeds_failed",),
    )


def rss_replay_stage(
    repository: Phase0Repository,
    *,
    feeds_path: Path,
    aliases_path: Path,
    pipeline_version: str,
) -> Stage:
    """The replay component, built with an HTTP callable that refuses.

    Lazily like the others, and for the same reason: replay reads a config
    file too, and a broken one should be a failed component rather than a
    traceback.
    """

    def action(base_run_id: str) -> tuple[dict[str, Any], list[Any]]:
        fetcher = RSSFetcher(
            repository,
            feeds_path=feeds_path,
            aliases_path=aliases_path,
            pipeline_version=pipeline_version,
            get=_refuse_network,
        )
        return fetcher.reclassify_persisted(run_id=base_run_id)

    return Stage("rss_relevance_replay", action, settled=("updated",))


def _finish(
    *,
    invocation_id: str,
    mode: str,
    components: Sequence[ComponentResult],
    started_at: datetime,
    pipeline_version: str,
    day: str,
    detail: dict[str, Any] | None = None,
    status: str | None = None,
) -> InvocationResult:
    completed_at = datetime.now(timezone.utc)
    result = InvocationResult(
        invocation_id=invocation_id,
        mode=mode,
        status=invocation_status(components) if status is None else status,
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        duration_ms=round((completed_at - started_at).total_seconds() * 1000),
        invocation_day=day,
        pipeline_version=pipeline_version,
        components=tuple(components),
        detail=dict(detail or {}),
    )
    _log_event("invocation_completed", **result.as_dict())
    return result


# -- Live orchestration --------------------------------------------------


def run_live(
    repository: Phase0Repository,
    *,
    feeds_path: Path,
    aliases_path: Path,
    pipeline_version: str = PIPELINE_VERSION,
    invocation_id: str | None = None,
    now: datetime | None = None,
) -> InvocationResult:
    """Fetch every source, then report what each of them settled.

    Ordering is Yahoo, then RSS, then whatever is registered in
    ``DOWNSTREAM_STAGES``.  Each component runs to completion independently:
    it opens its own runs, settles its own partitions, and its evidence is
    durable the moment it commits, so a later component failing cannot cost
    an earlier one its day.
    """

    repository.migrate()
    started_at = datetime.now(timezone.utc)
    correlation = new_invocation_id() if invocation_id is None else invocation_id
    day = invocation_day(now)
    stages = [
        yahoo_stage(repository, pipeline_version=pipeline_version),
        rss_stage(
            repository,
            feeds_path=feeds_path,
            aliases_path=aliases_path,
            pipeline_version=pipeline_version,
        ),
        *DOWNSTREAM_STAGES,
    ]
    _log_event(
        "invocation_started",
        invocation_id=correlation,
        mode="live",
        invocation_day=day,
        pipeline_version=pipeline_version,
        schema_version=repository.schema_version(),
        stages=[stage.name for stage in stages],
    )
    components = [execute_stage(stage, invocation_id=correlation) for stage in stages]
    return _finish(
        invocation_id=correlation,
        mode="live",
        components=components,
        started_at=started_at,
        pipeline_version=pipeline_version,
        day=day,
    )


# -- Replay --------------------------------------------------------------


def replay_capabilities() -> dict[str, Any]:
    """What ``--replay`` can and cannot rebuild today.

    Reported rather than assumed, because "replay the pipeline" is a claim
    this file cannot currently honour: RSS relevance is the only derived
    state any registered component owns.  Dedup, clustering, summarization
    and the rest are not registered, so their replay is unimplemented, not
    merely untested.
    """

    return {
        "supported": ["rss_relevance"],
        "unsupported": [
            "yahoo_refetch",
            "dedup",
            "clustering",
            "summarization",
        ],
        "downstream_stages_registered": len(DOWNSTREAM_STAGES),
        "scope": "all persisted RSS evidence",
        "scoped_replay_available": False,
    }


def run_replay(
    repository: Phase0Repository,
    *,
    feeds_path: Path,
    aliases_path: Path,
    pipeline_version: str = PIPELINE_VERSION,
    invocation_id: str | None = None,
    now: datetime | None = None,
) -> InvocationResult:
    """Rebuild derived state from stored evidence, touching no network.

    What this does today is I3's ``reclassify_persisted``: every persisted
    RSS item is reclassified, and each ``(ticker, day)`` partition's derived
    state is *replaced* inside that partition's own terminal run.  Raw
    evidence -- snapshots, provenance, ``raw_json``, the parser's own
    verdict -- is read and never written, so replay is idempotent and a
    partition that fails keeps the derived state it already had rather than
    being cleared first and rebuilt after.

    Nothing is deleted, no stage key is reset, and no partition outside the
    ones being replaced is touched.  There is deliberately no "clear the
    day's derived tables" step: that was the old orchestrator's idea of
    replay, and it destroyed state it could not rebuild.
    """

    repository.migrate()
    started_at = datetime.now(timezone.utc)
    correlation = new_invocation_id() if invocation_id is None else invocation_id
    day = invocation_day(now)
    capabilities = replay_capabilities()
    stages = [
        rss_replay_stage(
            repository,
            feeds_path=feeds_path,
            aliases_path=aliases_path,
            pipeline_version=pipeline_version,
        )
    ]
    _log_event(
        "invocation_started",
        invocation_id=correlation,
        mode="replay",
        invocation_day=day,
        pipeline_version=pipeline_version,
        schema_version=repository.schema_version(),
        stages=[stage.name for stage in stages],
        replay=capabilities,
    )
    components = [execute_stage(stage, invocation_id=correlation) for stage in stages]
    return _finish(
        invocation_id=correlation,
        mode="replay",
        components=components,
        started_at=started_at,
        pipeline_version=pipeline_version,
        day=day,
        detail={"replay": capabilities},
    )


# -- Read-only CLI reports -----------------------------------------------


def status_report(repository: Phase0Repository) -> dict[str, Any]:
    """The latest durable stage status, straight from ``run_log``.

    Read through I1's own reader.  The rows are per partition, so a stage
    appears once per partition it settled -- that granularity is the point
    and is not summed away here.
    """

    repository.migrate()
    return redact_secrets(
        {
            **repository.pipeline_status(),
            "schema_version": repository.schema_version(),
            "replay": replay_capabilities(),
        }
    )


def database_report(repository: Phase0Repository) -> dict[str, Any]:
    """Schema version, applied migrations, and stored row counts."""

    repository.migrate()
    return redact_secrets(
        {
            "database": str(repository.database_path),
            "schema_version": repository.schema_version(),
            "applied_migrations": [
                migration["name"] for migration in repository.applied_migrations()
            ],
            "counts": {
                table: repository.count(table)
                for table in ("raw_items", "feed_snapshots", "run_log", "source_state")
            },
        }
    )


# -- CLI -----------------------------------------------------------------


def _default_database_path() -> Path:
    return Path(os.getenv("PHASE0_DATABASE_PATH", str(DEFAULT_DATABASE_PATH)))


def resolve_lock_file(args: argparse.Namespace) -> Path:
    """The one lock this invocation contends on.

    ``<database>.lock`` is the *local development* default: two checkouts
    on one laptop hold different databases and should not block each other.

    It is the wrong default for a deployment, where cron, systemd, and an
    operator's shell may each name the database differently while targeting
    one pipeline. Every production entrypoint therefore passes an explicit
    ``--lock-file``, and ``deploy/phase0-pipeline.cron`` documents the same
    path for all of them. Acquisition lives here and nowhere else -- a
    shell-level ``flock`` wrapped around this would be a *second*,
    different lock, so a cron run and a manual run would each hold one and
    both would proceed.
    """

    return args.lock_file or Path(f"{args.database}.lock")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase 0 data pipeline")
    parser.add_argument("--database", type=Path, default=_default_database_path())
    parser.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--pipeline-version", default=PIPELINE_VERSION)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=None,
        help="Single-instance lock (default: <database>.lock)",
    )
    parser.add_argument(
        "--date",
        help="Not accepted: components derive each partition's day from evidence",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--replay",
        action="store_true",
        help="Rebuild RSS relevance from persisted evidence; no network",
    )
    mode.add_argument(
        "--status", action="store_true", help="Print the latest durable stage status"
    )
    mode.add_argument(
        "--database-info",
        action="store_true",
        help="Print schema version, applied migrations, and row counts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.date:
        # Honouring this is not possible and pretending to would be worse.
        # A live day is decided by each item's own timestamps, and replay
        # has no scope parameter to filter on -- see replay_capabilities().
        raise SystemExit(
            "--date is not supported: live partitions are derived from evidence "
            "timestamps, and replay covers all persisted RSS evidence"
        )
    repository = Phase0Repository(args.database)
    if args.status:
        print(json.dumps(status_report(repository), indent=2, sort_keys=True))
        return 0
    if args.database_info:
        print(json.dumps(database_report(repository), indent=2, sort_keys=True))
        return 0

    lock_path = resolve_lock_file(args)
    with single_instance(lock_path) as acquired:
        if not acquired:
            _log_event(
                "invocation_skipped",
                reason="another invocation holds the lock",
                lock_file=str(lock_path),
                mode="replay" if args.replay else "live",
            )
            return EXIT_CODES["skipped"]
        runner = run_replay if args.replay else run_live
        result = runner(
            repository,
            feeds_path=args.feeds,
            aliases_path=args.aliases,
            pipeline_version=args.pipeline_version,
        )
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())

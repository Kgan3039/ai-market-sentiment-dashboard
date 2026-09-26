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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence
from zoneinfo import ZoneInfo

from phase0 import rss as rss_module
from phase0 import yahoo as yahoo_module
from phase0.coordinator import PartitionCoordinator
from phase0.repository import (
    DEFAULT_DATABASE_PATH,
    STAGE_DEGRADED,
    Phase0Repository,
    StageEpisode,
    StageOutcome,
    redact_secrets,
)
from phase0 import summary_runner
from phase0.rss import RSSFetcher
from phase0.stories import STAGE as STORIES_STAGE
from phase0.themes import STAGE as THEMES_STAGE
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


#: How a downstream stage is registered: a builder, not a built stage.
#:
#: A ``Stage`` binds a repository, and a module-level tuple has no
#: repository to bind at import time, so the registry holds constructors
#: and ``run_live`` calls them beside ``yahoo_stage`` and ``rss_stage``.
#: Every builder takes the same keyword arguments so the loop in
#: ``run_live`` stays one line whatever lands here next.
DownstreamStageBuilder = Callable[..., "Stage | None"]

#: The name of the component that produces stories and themes.  A unit of
#: work, not an algorithm: the durable ``stories`` and ``themes`` run rows
#: it leaves are the repository's, written by the reconcilers under their
#: own stage names (decision A4).  This is only what the invocation calls
#: the component that drove them.
INTELLIGENCE_STAGE = "intelligence"

#: The run-identity component a *retry* attempt is opened under, as
#: distinct from an attempt triggered by evidence this invocation ingested.
#:
#: The distinction has to be durable, because the retry window is
#: anchored on it: a failure that arrived with new evidence opens a
#: window, and a failure that is merely a retry of the same input must
#: not reopen one.  The ledger already records which run produced each
#: outcome, so the trigger travels in the run identity rather than in a
#: new column.  ``execute_stage`` derives ``<invocation>:intelligence`` for
#: the component; retried days derive ``<invocation>:intelligence-retry``
#: from it, and every partition run under that base ends in
#: ``:intelligence-retry:<ticker>:<day>``.
#:
#: Recognition is structural, not textual -- see :func:`is_retry_run`.
#: The caller chooses the invocation id and may put anything in it,
#: including this very string; the three trailing components are the
#: pipeline's, so those are the only ones read.
RETRY_RUN_SUFFIX = "-retry"
RETRY_COMPONENT = f"{INTELLIGENCE_STAGE}{RETRY_RUN_SUFFIX}"

#: The two durable stages the intelligence component drives.
INTELLIGENCE_STAGES = (STORIES_STAGE, THEMES_STAGE)

#: The ingestion stages whose runs mean "this partition's evidence, or
#: which ticker it belongs to, may have changed".  Named by inclusion so
#: a feed checkpoint or a snapshot -- runs that touch feed state and never
#: an item -- do not make a day look touched, and so a stage added later
#: has to be added here on purpose.
EVIDENCE_STAGES = (
    yahoo_module.STAGE,
    rss_module.STAGE_INGEST,
    rss_module.STAGE_CLASSIFY,
    rss_module.STAGE_RECLASSIFY,
)

#: The run-log counters that mean an evidence-stage run *durably changed*
#: what a partition's story stage will read: a raw item inserted under
#: the partition, or an item whose association with the partition's
#: ticker -- or whose eligibility -- is different after the run from
#: before.  A run of an evidence stage that recorded neither saw only
#: what was already stored; a provider serving the same article again,
#: or a classifier re-deciding the same association, is not news.
DURABLE_CHANGE_COUNTERS = ("raw_items_inserted", "relevance_changed")

#: How long an unresolved intelligence failure keeps being retried, and
#: how far back an evidence-writing run is looked at for partitions the
#: pipeline never got to.  One horizon for both: they are the two ways
#: the same unattended schedule falls behind, and a day the operator
#: would want caught up for one reason is a day they would want caught
#: up for the other.
#: without new evidence arriving for its partition, measured from the
#: attempt that *opened* the episode -- the first failure after the last
#: success, or after the last evidence-triggered attempt -- and never
#: from a retry.  A retry that fails again does not extend its own
#: window; after this long it stops, and only new evidence, arriving
#: through the touched-day path, opens another.
#:
#: Three days covers the ordinary shape of a transient outage -- a model
#: cache missing on Friday's last run is retried on Monday's first -- and
#: bounds two costs: the ``run_log`` scan that finds such episodes, and
#: how many times a partition with a *permanent* defect is re-attempted
#: before it is left alone.  At the scheduled cadence that is roughly 150
#: attempts, each of which reads the partition and writes a failed run
#: row; unchanged healthy partitions on the same day settle without
#: rewriting stories or themes but do write their own run rows.
RETRY_HORIZON = timedelta(days=3)


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


# -- Intelligence: stories and themes over what ingestion persisted --------
#
# The component calls ``PartitionCoordinator`` and nothing else.  The
# coordinator owns the ordering that makes story and theme output correct
# -- capture previous theme identities, then stories, then themes with the
# captured identities -- and owns the isolation between partitions.  What
# is decided here is only *which days* to hand it.


Partition = tuple[str, str]
"""``(ticker, trading_day)`` -- the grain every scheduling fact has."""


@dataclass(frozen=True)
class IntelligenceSelection:
    """Which partitions this invocation reconciles, why, and under what name.

    Every fact here is per partition, because that is the grain the
    ledger keeps them at; the day is only the unit the coordinator
    executes.  ``touched`` are the partitions this invocation's own
    evidence-stage runs durably changed, read back from ``run_log``
    rather than remembered.  ``retried`` are partitions whose unresolved
    failure episode is still inside its retry window.  ``recovered`` are
    partitions, not touched, with recent evidence whose intelligence work
    this pipeline version never began, or began and never carried as far
    as themes.  ``unresolved`` are the partitions on the selected days whose
    newest ``stories`` or ``themes`` outcome is :func:`unresolved` --
    an episode, whether its window is open or has closed.  ``days`` is
    every day named by the first three, sorted.

    **Identity -- touched, then episode, then everything else.**  The
    identity a partition runs under decides whether a failure there can
    anchor a new retry window, so it follows the strongest fact about
    *that partition*, never about its neighbours on the day.  Evidence
    this invocation changed is the strongest: a new reason to process,
    and a failure on it legitimately opens a fresh window.  An existing
    episode comes next: a partition that has one and was not touched
    runs under the retry identity, whether it was selected as a retry,
    is rerun because its day was, or is an expired episode that came
    along -- so no attempt on it can renew a deadline it already owns.
    Every other partition -- never attempted, interrupted before
    themes, or a settled neighbour -- runs under the evidence-triggered
    identity, so if its attempt fails, that failure is an anchor and the
    partition is retried on its own account.
    """

    touched: frozenset[Partition]
    retried: frozenset[Partition]
    recovered: frozenset[Partition]
    unresolved: frozenset[Partition]
    days: tuple[str, ...]

    @property
    def touched_days(self) -> tuple[str, ...]:
        return _days(self.touched)

    @property
    def retried_days(self) -> tuple[str, ...]:
        return _days(self.retried)

    @property
    def recovered_days(self) -> tuple[str, ...]:
        return _days(self.recovered)

    def runs_as_retry(self, ticker: str, day: str) -> bool:
        """Does this partition run under the retry identity?

        It has an episode and this invocation did not change its
        evidence.  Nothing about any other partition enters into it.
        """

        partition = (ticker, day)
        return partition in self.unresolved and partition not in self.touched


def _days(partitions: frozenset[Partition]) -> tuple[str, ...]:
    return tuple(sorted({day for _, day in partitions}))


def unresolved(outcome: StageOutcome) -> bool:
    """Does this outcome leave its partition's intelligence work undone?

    A ``failed`` run does.  A ``degraded`` run does when it carries a
    ``stage_degraded`` marker -- semantic dedup was unavailable, or themes
    were refused because the stories were M2-only -- because the stored
    generation is honest and intermediate and a later run with the model
    back would replace it.  A ``degraded`` run with **no** marker does
    not: that is an identical replay whose unchanged rows counted as
    partial work, and there is nothing to redo.  The marker is the whole
    distinction, and ``record_degradation`` puts it there so a scheduler
    never has to infer intent from the word ``degraded``.
    """

    if outcome.status == "failed":
        return True
    if outcome.status == "degraded":
        return any(kind == STAGE_DEGRADED for kind, _ in outcome.markers)
    return False


def is_retry_run(run_id: str) -> bool:
    """Was this partition run opened as a retry?

    An intelligence partition run id is ``<base>:<component>:<ticker>:<day>``
    where ``<base>`` is whatever the caller named the invocation and the
    last three components were appended by this file and by
    ``partition_run_id``.  The ticker and the day contain no colon, so
    splitting three times from the right isolates exactly the component
    this pipeline chose -- whatever the caller put in front of it.  A
    caller-supplied invocation id that happens to contain
    ``:intelligence-retry:`` therefore still runs, and is still read back,
    as an ordinary evidence-triggered attempt.
    """

    parts = run_id.rsplit(":", 3)
    if len(parts) != 4:
        return False
    _, component, ticker, day = parts
    if component != RETRY_COMPONENT or not ticker:
        return False
    try:
        date.fromisoformat(day)
    except ValueError:
        return False
    return True


def episode_anchor(episode: StageEpisode) -> str | None:
    """When this episode's retry window opened, or ``None`` if never.

    The anchor is the newest unresolved attempt that was *not itself a
    retry* -- the first failure after the last success, or the failure
    that followed new evidence.  Retries are recognised by the identity
    they ran under and never move it, so a permanent failure retried
    every half hour cannot keep its own window open.  An episode made of
    retries alone has no anchor; it cannot exist through this scheduler,
    since a retry needs a live window to be scheduled at all, and it is
    reported as expired rather than guessed at.
    """

    anchoring = [
        attempt.completed_at
        for attempt in episode.attempts
        if not is_retry_run(attempt.run_id)
    ]
    return max(anchoring) if anchoring else None


def retryable(episode: StageEpisode, *, since: str) -> bool:
    """Is this episode still inside its retry window?"""

    if not unresolved(episode.newest):
        return False
    anchor = episode_anchor(episode)
    return anchor is not None and anchor >= since


def needs_recovery(latest_stories: StageOutcome | None) -> bool:
    """Given no ``themes`` run, does the newest ``stories`` outcome leave a
    partition for recovery rather than for the retry window?

    ``None`` -- no ``stories`` run under this version -- is a partition
    whose intelligence work never began: recover.  A resolved outcome --
    ``success``, or a marker-free ``degraded`` replay -- is one the
    coordinator would have carried on to themes from, so the missing
    themes run means the process was interrupted between the two:
    recover.  An :func:`unresolved` outcome -- ``failed``, or ``degraded``
    with a ``stage_degraded`` marker -- already has an episode, anchored
    on the attempt that recorded it; recovery must not touch it, or the
    anchor would move.  Whether that episode is still inside its window
    is :func:`retryable`'s question, and if it is not, the partition
    stays where the retry contract leaves it.
    """

    return latest_stories is None or not unresolved(latest_stories)


def select_intelligence_days(
    repository: Phase0Repository,
    *,
    invocation_id: str,
    pipeline_version: str,
    now: datetime,
    horizon: timedelta = RETRY_HORIZON,
) -> IntelligenceSelection:
    """Decide which partitions the intelligence component reconciles.

    **Touched partitions** are read from the ledger under this
    invocation's own prefix: runs of :data:`EVIDENCE_STAGES` that name a
    ticker and recorded one of :data:`DURABLE_CHANGE_COUNTERS` above
    zero.  Every component derives its partition run ids from the base
    the orchestrator gave it, so the prefix is the authoritative record
    of what was opened, and the counters -- written by the mutations
    themselves -- are the authoritative record of whether anything
    changed.  A late-arriving article for last Tuesday inserts under last
    Tuesday's partition and so selects it; a provider serving an article
    already stored, or a classifier re-deciding an association that
    already stood, opens runs that record no change and selects nothing.
    Touched is therefore never a way for a repeat sighting to reopen an
    expired failure.

    **Retried partitions** hold an :func:`unresolved` newest ``stories``
    or ``themes`` outcome whose episode's :func:`episode_anchor` is
    within :data:`RETRY_HORIZON` of ``now``.  Without this, an unattended
    pipeline would leave a transient failure failed for as long as no
    fresh evidence happened to arrive for that partition; with the
    anchor, a permanent failure stops being retried once its window
    closes, whatever the retries themselves recorded.

    **Recovered partitions** close the gap the other two leave.  An
    invocation that persists evidence and then dies before this component
    runs leaves partitions with evidence and no intelligence rows; one
    that dies inside it can leave stories settled and themes never
    opened.  Neither has a failure episode, because nothing failed.  So
    every day an evidence-writing run completed within the horizon is
    checked, ticker by ticker, against :func:`needs_recovery`: the
    partition holds authoritative evidence, this pipeline version has
    **no ``themes`` run** for it, and its newest ``stories`` outcome --
    if there is one -- is resolved.  That last clause is what keeps
    recovery honest.  A partition whose stories *failed* also has no
    ``themes`` row, because the coordinator does not open themes over a
    story failure; but that partition has an episode, with an anchor and
    a deadline, and running it here under the evidence-triggered identity
    would give it a new anchor every time -- the deadline would never
    arrive.  Such a partition is the retry window's, inside the window
    and outside it.  The ``themes`` row is the witness that a partition
    was carried through: the coordinator writes one for every partition
    it finishes -- healthy, empty, M2-only, or failed at capture -- and a
    healthy completed day has one for every partition and is never
    selected again.

    ``now`` and the ledger's ``completed_at`` values come from the same
    clock -- the repository's -- so every comparison here means what it
    says.  Evidence timestamps are not consulted.

    The coordinator works a whole day at a time, so a selected partition
    brings its day's other partitions with it.  Those settle without
    rewriting stories or themes, but each writes its own run row -- under
    its own identity, see :class:`IntelligenceSelection`.
    """

    reader = repository.read
    touched = frozenset(
        reader.changed_partitions(
            f"{invocation_id}:",
            stages=EVIDENCE_STAGES,
            counters=DURABLE_CHANGE_COUNTERS,
        )
    )
    since = (now.astimezone(timezone.utc) - horizon).isoformat()
    episodes = reader.stage_outcome_episodes(
        INTELLIGENCE_STAGES, pipeline_version=pipeline_version, completed_since=since
    )
    retried = frozenset(
        (episode.ticker, episode.trading_day)
        for episode in episodes
        if retryable(episode, since=since)
    )
    recovered: set[Partition] = set()
    for day in reader.recent_run_days(EVIDENCE_STAGES, completed_since=since):
        with_evidence = set(reader.evidence_partition_tickers(day))
        themes_attempted = set(
            reader.attempted_partitions(
                THEMES_STAGE, day, pipeline_version=pipeline_version
            )
        )
        latest_stories = reader.latest_partition_outcomes(
            STORIES_STAGE, day, pipeline_version=pipeline_version
        )
        recovered.update(
            (ticker, day)
            for ticker in with_evidence - themes_attempted
            if needs_recovery(latest_stories.get(ticker))
        )
    # Evidence this invocation just changed is touched, whatever else is
    # true of it; recovery is for evidence nobody has processed *before*.
    recovered -= touched
    days = _days(touched | retried | frozenset(recovered))
    unresolved_partitions: set[Partition] = set()
    for day in days:
        for stage in INTELLIGENCE_STAGES:
            newest = reader.latest_partition_outcomes(
                stage, day, pipeline_version=pipeline_version
            )
            unresolved_partitions.update(
                (ticker, day)
                for ticker, outcome in newest.items()
                if unresolved(outcome)
            )
    return IntelligenceSelection(
        touched=touched,
        retried=retried,
        recovered=frozenset(recovered),
        unresolved=frozenset(unresolved_partitions),
        days=days,
    )


def intelligence_stage(
    repository: Phase0Repository,
    *,
    pipeline_version: str,
    invocation_id: str,
    encoder: Any | None = None,
) -> Stage:
    """The stories-and-themes component, built when it runs.

    ``encoder`` is the one M1 service both reconcilers share; ``None``
    means the default local model, resolved lazily by the coordinator on
    the first partition that needs a vector.  Tests hand in a fake so no
    model is ever loaded, the same way they hand ``yahoo_stage`` a fake
    provider.

    Not mandatory, and the reason is arithmetic rather than importance.
    ``invocation_status`` reports ``failed`` only when every mandatory
    component settled nothing, and a run with no partitions to reconcile
    is an honest ``success`` -- which, were this component mandatory,
    would stop an invocation whose Yahoo *and* RSS fetches both failed
    from being called ``failed``.  Marking it non-mandatory cannot make a
    failure here look green: ``success`` requires every component to
    succeed, so a failed or degraded intelligence run is a ``degraded``
    invocation at best.
    """

    def action(base_run_id: str) -> tuple[dict[str, Any], list[Any]]:
        selection = select_intelligence_days(
            repository,
            invocation_id=invocation_id,
            pipeline_version=pipeline_version,
            now=repository.now(),
        )
        coordinator = PartitionCoordinator(
            repository, pipeline_version=pipeline_version, encoder=encoder
        )
        counts: dict[str, Any] = {
            "days_touched": len(selection.touched_days),
            "days_retried": len(selection.retried_days),
            "days_recovered": len(selection.recovered_days),
            "days_selected": len(selection.days),
            "partitions_touched": len(selection.touched),
            "partitions_retried": len(selection.retried),
            "partitions_recovered": len(selection.recovered),
            "partitions": 0,
        }
        errors: list[Any] = []
        retry_base = f"{base_run_id}{RETRY_RUN_SUFFIX}"
        for day in selection.days:
            # Each partition runs under its own identity: a partition with
            # an episode this invocation did not touch runs as a retry, so
            # its outcome can never anchor a new window; every other one
            # runs evidence-triggered, so a failure there *is* an anchor.
            def identity(ticker: str, day: str = day) -> str:
                return (
                    retry_base if selection.runs_as_retry(ticker, day) else base_run_id
                )

            day_counts, day_errors = coordinator.run(
                day, run_id=base_run_id, identity=identity
            )
            for key, value in day_counts.items():
                if isinstance(value, int):
                    counts[key] = counts.get(key, 0) + value
            errors.extend(day_errors)
        return counts, errors

    return Stage(
        INTELLIGENCE_STAGE,
        action,
        # A partition counts as settled when its stories were written --
        # healthy or explicitly degraded -- because that generation is
        # durable whatever the themes then did.  A theme failure on top of
        # settled stories is a degraded component, not a failed one, and
        # the error that says so travels in ``errors``.
        settled=("stories_succeeded", "stories_degraded"),
        unsettled=("stories_failed", "stories_not_attempted"),
        mandatory=False,
    )


# -- Summaries: guarded summaries over the persisted themes (A3b) ---------
#
# Feature-gated and off by default.  The component calls
# ``phase0.summary_runner`` and nothing else; the A2/A3 lifecycle it drives
# owns caching, currentness and accounting.  It runs after intelligence and
# independently of it: it sweeps whatever healthy theme populations are
# persisted, so an intelligence failure this invocation leaves it the
# populations the last good run stored.

#: The name of the component that makes theme summaries current.  Its
#: durable run rows are the repository's, under ``summary_runner.STAGE``.
SUMMARIES_STAGE = "summaries"


def summaries_stage(
    repository: Phase0Repository,
    *,
    pipeline_version: str,
    invocation_id: str,
    **_: Any,
) -> Stage | None:
    """The summaries component, or ``None`` when the feature is off.

    ``None`` is the flag's whole effect: nothing is built, no client is
    constructed, no summary table is read.  A flag value that is neither
    on nor off is not guessed at -- the component is built to report
    ``summaries_misconfigured`` and does nothing else, because this
    builder runs before ``execute_stage`` and must not raise.

    Not mandatory, and never in :data:`INTELLIGENCE_STAGES`: a summary
    outcome can make an invocation ``degraded``, but it cannot fail one,
    cannot unwind an earlier component, and cannot open or extend a story
    or theme retry episode.
    """

    try:
        enabled = summary_runner.summaries_enabled()
    except summary_runner.SummaryConfigError as exc:
        message = str(exc)

        def refuse(base_run_id: str) -> tuple[dict[str, Any], list[Any]]:
            return summary_runner.empty_counts(), [
                {"type": "summaries_misconfigured", "error": message}
            ]

        return Stage(
            SUMMARIES_STAGE,
            refuse,
            settled=("partitions_settled",),
            unsettled=("partitions_failed",),
            mandatory=False,
        )
    if not enabled:
        return None

    def action(base_run_id: str) -> tuple[dict[str, Any], list[Any]]:
        return summary_runner.run_scheduled_summaries(
            repository,
            pipeline_version=pipeline_version,
            base_run_id=base_run_id,
            horizon=RETRY_HORIZON,
        )

    return Stage(
        SUMMARIES_STAGE,
        action,
        # A partition is settled when its summary work reached a resolved
        # state -- current, refused by the health gate, or processed --
        # whatever each theme's outcome was.  An unavailable theme on a
        # settled partition is a degraded component, reported in errors.
        settled=("partitions_settled",),
        unsettled=("partitions_failed",),
        mandatory=False,
    )


#: Downstream components, in the order they run after ingestion.  Builders,
#: bound inside ``run_live``; see :data:`DownstreamStageBuilder`.  A builder
#: may return ``None`` for a component switched off by configuration.
DOWNSTREAM_STAGES: tuple[DownstreamStageBuilder, ...] = (
    intelligence_stage,
    summaries_stage,
)


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
    encoder: Any | None = None,
) -> InvocationResult:
    """Fetch every source, then reconcile what they persisted, then report.

    Ordering is Yahoo, then RSS, then every builder in
    ``DOWNSTREAM_STAGES`` -- the intelligence component, which turns the
    evidence the first two committed into persisted stories and themes,
    then, when ``PHASE0_SUMMARIES_ENABLED`` is on, the summaries
    component.  Each component runs to completion independently: it opens its
    own runs, settles its own partitions, and its output is durable the
    moment it commits, so a later component failing cannot cost an earlier
    one its day.

    ``encoder`` is passed through to the downstream builders; ``None``
    means the default local embedding model.

    **There is one clock, and it is the repository's.**  The day label,
    and every comparison a downstream component makes against run
    timestamps, are read from ``repository.now()`` -- the same source
    that stamps every run's ``started_at`` and ``completed_at``.  There
    is deliberately no ``now`` parameter here: a caller-supplied instant
    would govern the comparison but not the timestamps it compares
    against, which is two clocks, and two clocks disagree.  A test that
    needs to move time hands the repository a clock.  Production leaves
    it unset and gets UTC wall time.
    """

    repository.migrate()
    started_at = datetime.now(timezone.utc)
    correlation = new_invocation_id() if invocation_id is None else invocation_id
    day = invocation_day(repository.now())
    stages = [
        yahoo_stage(repository, pipeline_version=pipeline_version),
        rss_stage(
            repository,
            feeds_path=feeds_path,
            aliases_path=aliases_path,
            pipeline_version=pipeline_version,
        ),
    ]
    for build in DOWNSTREAM_STAGES:
        stage = build(
            repository,
            pipeline_version=pipeline_version,
            invocation_id=correlation,
            encoder=encoder,
        )
        if stage is not None:
            stages.append(stage)
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
    this file cannot fully honour.  RSS relevance is the only derived
    state ``run_replay`` rebuilds.  Stories and themes are now produced by
    the *live* path -- the intelligence component reconciles them after
    every ingestion -- but ``--replay`` does not drive that component, and
    there is no scoped "rebuild this partition" entry point.  Summaries are
    generated by the live, feature-gated summaries component only: replay
    has no network and never builds it.
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
        "live_only": [INTELLIGENCE_STAGE, SUMMARIES_STAGE],
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
    day = invocation_day(repository.now())
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

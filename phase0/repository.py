"""SQLite persistence shared by Phase 0 pipeline stages.

This module is the only place in the project that talks to SQLite.  Every
stage — ingestion, dedup, clustering, summarization, and the read API —
goes through :class:`Phase0Repository`, so the schema contract, the ticker
universe, secret redaction, and transaction boundaries are stated once.

Three properties are worth stating up front, because callers depend on
them:

* **Every public write is one transaction.**  A batch either lands whole or
  not at all; no method commits part of a batch on its way through.
* **Fingerprints are not identifiers.**  M2's ``cluster_fingerprint``, M3's
  ``story_fingerprint``, and M5's ``fingerprint``/``theme_key`` are stored
  as change-detection handles beside the durable row ids, never instead of
  them.
* **Stage logging is not optional.**  There is no flag that turns
  ``run_log`` writes off; :meth:`Phase0Repository.stage_run` records a row
  even when the stage body raises.
* **Pipeline mutations carry their run.**  The operations issue #68 drives
  — raw-item ingestion, story and theme reconciliation, embedding batches,
  ingestion-coupled source state — take a ``run`` handle from
  :meth:`Phase0Repository.stage_run` and write their ``run_log`` row in the
  *same* transaction as the data.  Calling one without a usable run raises
  before anything is written.  The unlogged row helpers still exist for
  fixtures and backfills, but only behind :attr:`Phase0Repository.admin`,
  where nothing can mistake them for the pipeline API.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from nlp.embeddings import PersistedEmbedding

from .embeddings import (
    EmbeddingPersistenceError,
    embedding_from_row,
    normalize_source_id,
    normalize_source_kind,
    validate_embedding,
)
from .errors import (
    Phase0Error,
    Phase0IntegrityError,
    Phase0MigrationError,
    Phase0RunContextError,
    Phase0ValidationError,
    StageKeyError,
    UnsupportedTickerError,
)
from .models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    ProviderConflictRecord,
    ReconciliationReport,
    SemanticMergeRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from .redaction import SECRET_KEY_PATTERN, redact_secrets, redact_text
from .scalars import (
    require_safe_identifier_scalar,
    sanitize_diagnostic_scalar,
    validate_safe_identifier_scalar,
)
from .schema import apply_migrations, load_migrations, split_statements
from .tickers import SUPPORTED_TICKERS, TICKER_UNIVERSE, normalize_ticker


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "phase0.sqlite3"
MIGRATIONS_PATH = Path(__file__).with_name("migrations")
RUN_STATUSES = {"success", "degraded", "failed"}
STAGE_KEY_STATUSES = {"success", "degraded", "failed"}
RUNNING_STATUS = "running"
THEME_STATUSES = {"pending", "ready", "degraded", "failed"}
INGEST_STATUSES = {"valid", "invalid", "ambiguous"}
SOURCE_STATE_STATUSES = {"success", "partial", "empty", "failed", "unknown"}
STORY_STAGES = {"m2.exact", "m3.semantic"}
CLUSTERING_METHODS = {
    "hdbscan",
    "agglomerative",
    "small_n_fallback",
    "no_separable_structure",
}
OTHER_COVERAGE_REASONS = {
    "below_clustering_floor",
    "clustering_noise",
    "below_theme_size_floor",
    "below_cohesion_floor",
    "narrative_mismatch",
    "surplus_to_theme_cap",
    "theme_incompatible",
    "provider_quarantine",
    "semantic_skip",
    "degenerate_embedding_geometry",
    "insufficient_theme_structure",
}
EXCLUSION_REASONS = {"no_encodable_text"}

#: The only reason a raw item may be pulled into a ticker's derived output.
#: ``raw_item_tickers`` is the authoritative association table; a row in
#: ``raw_item_candidates`` is a *suggestion* that nothing has accepted yet,
#: which is exactly why it does not appear here.
RAW_ITEM_ASSOCIATION_TABLE = "raw_item_tickers"

#: The default reason recorded for a candidate given as a bare symbol.
DEFAULT_CANDIDATE_REASON = "relevance_match"

# ----------------------------------------------------------------------
# What a public read connection is allowed to do.
#
# ``mode=ro`` already refuses to write the file it opened.  It does not
# refuse to open a *second* file: a caller could ATTACH the very same
# database under another schema name and write through the alias, which is
# a hole straight through the logged-mutation contract.  The authorizer
# below closes it by allow-listing, so an action nobody thought about is
# denied rather than permitted.
# ----------------------------------------------------------------------

#: Actions a reader legitimately performs.  Everything else — INSERT,
#: UPDATE, DELETE, every CREATE/DROP/ALTER, REINDEX, ATTACH, DETACH — is
#: absent on purpose.
_READ_ONLY_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
        # Read transactions and savepoints move no data on their own, and a
        # writing statement inside one is still denied on its own account.
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
    }
)

#: Schema-inspection pragmas, which take the name of the object to inspect.
_READ_ONLY_PRAGMAS_WITH_ARGUMENT = frozenset(
    {
        "foreign_key_list",
        "index_info",
        "index_list",
        "index_xinfo",
        "table_info",
        "table_list",
        "table_xinfo",
    }
)

#: Pragmas allowed only in their *query* form.  The value form of several
#: of these writes (``user_version = 5``, ``journal_mode = DELETE``), and
#: ``query_only = OFF`` is the first move of the attack this guards
#: against, so a supplied argument is refused even for a name on the list.
_READ_ONLY_QUERY_PRAGMAS = frozenset(
    {
        "application_id",
        "auto_vacuum",
        "busy_timeout",
        "cache_size",
        "collation_list",
        "compile_options",
        "data_version",
        "database_list",
        "encoding",
        "foreign_key_check",
        "foreign_keys",
        "freelist_count",
        "function_list",
        "integrity_check",
        "journal_mode",
        "locking_mode",
        "max_page_count",
        "module_list",
        "page_count",
        "page_size",
        "pragma_list",
        "query_only",
        "quick_check",
        "schema_version",
        "secure_delete",
        "synchronous",
        "temp_store",
        "user_version",
        "wal_autocheckpoint",
    }
)


def _read_only_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    database: str | None,
    trigger_or_view: str | None,
) -> int:
    """Allow reads; deny everything else, in every schema.

    Denying by allow-list matters more than the individual entries: the
    bypass this replaces worked precisely because ``ATTACH`` was not on
    anybody's list of writes.  An action that is not named here is refused,
    whichever schema — ``main``, ``temp``, or an attached alias — it names.
    """

    if action in _READ_ONLY_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        name = (arg1 or "").strip().lower()
        if name in _READ_ONLY_PRAGMAS_WITH_ARGUMENT:
            return sqlite3.SQLITE_OK
        if name in _READ_ONLY_QUERY_PRAGMAS and arg2 is None:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


#: Tables :meth:`Phase0Repository.count` will count.  Kept explicit so the
#: helper can never be turned into an arbitrary-SQL escape hatch.
COUNTABLE_TABLES = frozenset(
    {
        "embeddings",
        "eval_labels",
        "pipeline_stage_keys",
        "raw_item_candidates",
        "raw_item_tickers",
        "raw_items",
        "run_log",
        "run_log_stage_keys",
        "schema_migrations",
        "source_state",
        "stories",
        "story_members",
        "story_provider_conflicts",
        "story_semantic_merges",
        "supported_tickers",
        "theme_citations",
        "theme_excluded_stories",
        "theme_other_coverage",
        "theme_sets",
        "theme_stories",
        "themes",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_datetime(
    value: str | datetime | None, field: str, *, optional: bool = False
) -> str | None:
    if value in (None, ""):
        if optional:
            return None
        raise Phase0ValidationError(f"{field} is required")
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise Phase0ValidationError(
                f"{field} must be an ISO-8601 timestamp"
            ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_day(value: str | date, field: str = "trading_day") -> str:
    if isinstance(value, datetime):
        raise Phase0ValidationError(f"{field} must be a date, not a datetime")
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError as exc:
        raise Phase0ValidationError(f"{field} must use YYYY-MM-DD format") from exc
    return parsed.isoformat()


def _require_text(value: Any, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise Phase0ValidationError(f"{field} is required")
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_int(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase0ValidationError(f"{field} must be a number")
    if isinstance(value, float) and not float(value).is_integer():
        raise Phase0ValidationError(f"{field} must be a whole number")
    result = int(value)
    if minimum is not None and result < minimum:
        raise Phase0ValidationError(f"{field} must be >= {minimum}")
    return result


def _optional_float(
    value: Any, field: str, *, low: float | None = None, high: float | None = None
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase0ValidationError(f"{field} must be a number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise Phase0ValidationError(f"{field} must be finite")
    if low is not None and result < low:
        raise Phase0ValidationError(f"{field} must be >= {low}")
    if high is not None and result > high:
        raise Phase0ValidationError(f"{field} must be <= {high}")
    return result


def _parse_json(value: Any, field: str, expected_type: type) -> Any:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise Phase0ValidationError(f"{field} must contain valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, expected_type):
        raise Phase0ValidationError(f"{field} must be a {expected_type.__name__}")
    return parsed


def _dump_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def serialize_raw_evidence(value: Any, field: str, expected_type: type = dict) -> str:
    """Serialize immutable provider evidence — **never redacted**.

    ``raw_items.raw_json`` is what the publisher actually sent.  Phase 0's
    replay guarantee (AC-8) is only worth something if that payload is
    preserved exactly, so this serializer normalizes and validates the
    JSON and changes nothing else.  A string that *looks* like a
    credential inside publisher content is evidence, not a leak, and a
    downstream reader comparing against the upstream feed must find it
    unchanged.

    Transport credentials are the fetcher's problem: I2/I3 must not put an
    ``Authorization`` header into the evidence payload in the first place.
    Silently rewriting it here would hide that bug *and* corrupt the
    evidence.  See :func:`serialize_operational_metadata` for the other
    side of the boundary.
    """

    return _dump_json(_parse_json(value, field, expected_type))


def serialize_operational_metadata(value: Any, field: str, expected_type: type) -> str:
    """Serialize Phase 0's own diagnostics — **always redacted**.

    Run-log counts and errors, source-state metadata, theme-set source and
    trust metadata, reconciliation diagnostics: everything the pipeline
    says *about* a fetch rather than everything the publisher said.  These
    are the surfaces a credential actually reaches, and nothing keys off
    them, so redaction is free here.  ``redact_secrets`` builds new
    containers, so the caller's own structure is never mutated.
    """

    return _dump_json(redact_secrets(_parse_json(value, field, expected_type)))


#: Kept as the operational spelling: every existing call site is
#: operational metadata, and the evidence path names itself explicitly.
_serialize_json = serialize_operational_metadata


def _optional_blob(value: Any, field: str) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise Phase0ValidationError(f"{field} must be bytes")
    return bytes(value)


# Retained under their historical private names: PR #82 and #83 already
# import the public ones, and the older name is still referenced in review
# threads.
_redact_secrets = redact_secrets
_migration_statements = split_statements


@dataclass(frozen=True)
class InsertResult:
    item_id: int
    inserted: bool


#: Module-private construction key.  It is never exported, never stored on
#: an instance, and never reachable from the public API, so a caller cannot
#: build a :class:`StageRunContext` even by copying every visible field.
_CONTEXT_KEY = object()


class StageRunContext:
    """Proof that a mutation belongs to a claimed, still-live run.

    A handle is only worth something if it cannot be manufactured.  Two
    independent things make this one unforgeable:

    * construction requires ``_CONTEXT_KEY``, which lives in this module
      and is not exported, so ``StageRunContext(...)`` from outside fails;
    * authorization is by **object identity** against a registry the owning
      repository keeps.  A copy, a pickle, an ``object.__new__`` shell, or
      a hand-built look-alike with every field matching is a different
      object, so none of them authorize anything.

    Identity is also what makes expiry total: leaving the ``stage_run``
    block unregisters the object, and no later reconstruction can
    re-register it.

    The counts are derived, not reported.  There is deliberately no public
    way to add to them — see :meth:`Phase0Repository.stage_run`.
    """

    __slots__ = (
        "_repository",
        "_run_id",
        "_stage",
        "_trading_day",
        "_pipeline_version",
        "_ticker",
        "_attempt",
        "_replay",
        "_stage_key",
        "_counts",
        "_errors",
        "_success_count",
        "_partial_count",
        "_failure_count",
        "_terminated",
        "_started_at",
    )

    def __init__(self, _key: Any = None, **kwargs: Any) -> None:
        if _key is not _CONTEXT_KEY:
            raise Phase0RunContextError(
                "StageRunContext cannot be constructed directly; obtain one "
                "from Phase0Repository.stage_run(...)"
            )
        setter = object.__setattr__
        setter(self, "_repository", kwargs["repository"])
        setter(self, "_run_id", kwargs["run_id"])
        setter(self, "_stage", kwargs["stage"])
        setter(self, "_trading_day", kwargs["trading_day"])
        setter(self, "_pipeline_version", kwargs["pipeline_version"])
        setter(self, "_ticker", kwargs["ticker"])
        setter(self, "_attempt", kwargs["attempt"])
        setter(self, "_replay", kwargs["replay"])
        setter(self, "_stage_key", kwargs["stage_key"])
        setter(self, "_counts", {})
        setter(self, "_errors", [])
        setter(self, "_success_count", 0)
        setter(self, "_partial_count", 0)
        setter(self, "_failure_count", 0)
        setter(self, "_terminated", False)
        setter(self, "_started_at", datetime.now(timezone.utc))

    def __setattr__(self, name: str, value: Any) -> None:
        """Read-only after construction, including the private counters.

        Counts describe what an operation did; a caller who could pre-seed
        or overwrite them could make the run log say anything.  The
        repository accumulates through :meth:`_record_outcome` and friends,
        which go around this on purpose.
        """

        raise Phase0RunContextError(
            "a StageRunContext is read-only; its counts are derived by the "
            "operations that run under it"
        )

    def __delattr__(self, name: str) -> None:
        raise Phase0RunContextError("a StageRunContext is read-only")

    # -- Read-only view of the partition this run covers ----------------

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def trading_day(self) -> str:
        return self._trading_day

    @property
    def pipeline_version(self) -> str:
        return self._pipeline_version

    @property
    def ticker(self) -> str | None:
        return self._ticker

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def replay(self) -> bool:
        return self._replay

    @property
    def stage_key(self) -> dict[str, Any] | None:
        return dict(self._stage_key) if self._stage_key is not None else None

    @property
    def counts(self) -> dict[str, Any]:
        """A copy: mutating the result cannot reach what gets persisted."""

        return dict(self._counts)

    @property
    def errors(self) -> list[Any]:
        return list(self._errors)

    @property
    def success_count(self) -> int:
        return self._success_count

    @property
    def partial_count(self) -> int:
        return self._partial_count

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def terminated(self) -> bool:
        """True once a terminal operation completed this run's stage key."""

        return self._terminated

    @property
    def started_at(self) -> datetime:
        return self._started_at

    @property
    def active(self) -> bool:
        return self._repository._is_active_run(self)

    @property
    def closed(self) -> bool:
        return not self.active

    # -- Repository-private accumulation --------------------------------
    #
    # Named with a leading underscore and absent from the public surface on
    # purpose: counts describe what an operation *did*, so only the
    # operations may write them.

    def _record_outcome(
        self, *, success: int = 0, partial: int = 0, failure: int = 0
    ) -> None:
        setter = object.__setattr__
        setter(self, "_success_count", self._success_count + int(success))
        setter(self, "_partial_count", self._partial_count + int(partial))
        setter(self, "_failure_count", self._failure_count + int(failure))

    def _mark_terminated(self) -> None:
        object.__setattr__(self, "_terminated", True)

    def _record_error(self, error: Mapping[str, Any] | str) -> None:
        self._errors.append(error)

    def _merge_counts(self, counts: Mapping[str, Any]) -> None:
        for key, value in counts.items():
            self._counts[str(key)] = (
                self._counts.get(str(key), 0) + value
                if isinstance(value, int)
                and isinstance(self._counts.get(str(key)), int)
                else value
            )

    def _resolved_status(self) -> str:
        if self._failure_count:
            return "failed"
        if self._errors or self._partial_count:
            return "degraded"
        return "success"

    # -- Nothing about this object may be copied or serialized ----------

    def __repr__(self) -> str:
        return (
            f"<StageRunContext run_id={self._run_id!r} stage={self._stage!r} "
            f"ticker={self._ticker!r} trading_day={self._trading_day!r} "
            f"pipeline_version={self._pipeline_version!r} active={self.active}>"
        )

    def _refuse(self, *args: Any, **kwargs: Any) -> Any:
        raise Phase0RunContextError(
            "a StageRunContext cannot be copied or serialized; a duplicate "
            "would outlive the run it authorizes"
        )

    __copy__ = _refuse
    __deepcopy__ = _refuse
    __reduce__ = _refuse
    __reduce_ex__ = _refuse
    __getstate__ = _refuse


#: Older stage bodies referred to the counter sheet by this name.
StageRunRecorder = StageRunContext


class Phase0Admin:
    """Migration, test-fixture, and manual-repair writes. **Not the pipeline.**

    Everything here mutates without a run and therefore without a
    ``run_log`` row.  That is legitimate for a migration backfilling a
    column, a test fixture seeding a day, or an operator repairing one bad
    row by hand.

    **Normal ingestion and orchestration must never use it.**  Issue #68's
    runner, and issues #82 and #83, use the logged entrypoints on
    :class:`Phase0Repository` — ``ingest_raw_items``, ``reconcile_stories``,
    ``reconcile_themes``, ``persist_embeddings``, ``record_source_state`` —
    each of which requires the :class:`StageRunContext` from ``stage_run``
    and writes its run-log row in the same transaction as the data.

    Keeping these behind ``repository.admin`` is the whole point: an
    unlogged write cannot happen without the call site saying so, and a
    reviewer greps ``.admin.`` to find every one of them.
    """

    def __init__(self, repository: "Phase0Repository") -> None:
        self._repository = repository

    # -- Connections ----------------------------------------------------

    @contextmanager
    def connect_writable(
        self, *, immediate: bool = False
    ) -> Iterator[sqlite3.Connection]:
        """A raw **writable** connection, for manual repair and migrations.

        Nothing validates what you do with this: no ticker check, no
        partition check, no run log.  It exists because an operator
        occasionally has to fix one row by hand, and because migrations
        need it.  Pipeline code must never call it — use the logged
        entrypoints on :class:`Phase0Repository`, which is why the public
        connection accessor (``read_connection``) cannot write.
        """

        with self._repository._connect(immediate=immediate) as connection:
            yield connection

    # -- Raw evidence ---------------------------------------------------

    def insert_raw_item(self, item: Mapping[str, Any]) -> InsertResult:
        return self._repository._insert_raw_items_unlogged([item])[0]

    def insert_raw_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        source_state: Mapping[str, Any] | None = None,
    ) -> list[InsertResult]:
        return self._repository._insert_raw_items_unlogged(
            items, source_state=source_state
        )

    def update_raw_item_ticker(self, item_id: int, ticker: str | None) -> None:
        return self._repository._update_raw_item_ticker_unlogged(item_id, ticker)

    # -- Source state ---------------------------------------------------

    def set_source_state(self, source: str, **kwargs: Any) -> None:
        return self._repository._set_source_state_unlogged(source, **kwargs)

    # -- Derived output -------------------------------------------------

    def reconcile_stories(self, **kwargs: Any) -> ReconciliationReport:
        return self._repository._reconcile_stories_unlogged(**kwargs)

    def reconcile_themes(self, **kwargs: Any) -> ReconciliationReport:
        return self._repository._reconcile_themes_unlogged(**kwargs)

    def insert_story(self, **kwargs: Any) -> int:
        return self._repository._insert_story_unlogged(**kwargs)

    def insert_theme(self, **kwargs: Any) -> int:
        return self._repository._insert_theme_unlogged(**kwargs)

    def insert_eval_label(self, **kwargs: Any) -> int:
        return self._repository._insert_eval_label_unlogged(**kwargs)

    def clear_derived_for_day(self, trading_day: str | date) -> None:
        return self._repository._clear_derived_for_day_unlogged(trading_day)

    # -- Run log ---------------------------------------------------------

    def log_stage(self, **kwargs: Any) -> int:
        """Write a run-log row directly, with no mutation attached.

        For repairing or backfilling history only.  A stage records itself
        through ``stage_run``.
        """

        return self._repository._log_stage_unlogged(**kwargs)


class Phase0Repository:
    """Repository contract shared by the scheduled writer and API readers."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
        *,
        migrations_path: str | Path = MIGRATIONS_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        self.migrations_path = Path(migrations_path)
        self.admin = Phase0Admin(self)
        # Identity registry of live run contexts.  A context authorizes a
        # write only while it is in here, and only the object itself does —
        # never a copy carrying the same fields.
        self._active_runs: dict[int, StageRunContext] = {}
        self._run_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connections and migrations
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        enforced = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        if not enforced:
            connection.close()
            raise Phase0IntegrityError(
                "SQLite refused to enable foreign-key enforcement"
            )
        return connection

    @contextmanager
    def _write_scope(
        self, connection: sqlite3.Connection | None
    ) -> Iterator[sqlite3.Connection]:
        """Join the caller's transaction, or open one when there is none.

        Lets a reconciler run either on its own (administrative use) or
        inside the transaction that also writes the run log, without the
        body needing two versions.
        """

        if connection is not None:
            yield connection
        else:
            with self._connect(immediate=True) as owned:
                yield owned

    @contextmanager
    def _connect(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """A **writable** connection, private on purpose.

        Handing one of these out publicly was a hole big enough to drive
        the whole logged-mutation contract through: a caller could insert
        raw items directly and leave no run log at all.  Writable access is
        now reachable only from inside this class or, explicitly, from
        :meth:`Phase0Admin.connect_writable`.

        ``immediate=True`` takes the write lock up front, which is what a
        read-modify-write sequence needs: without it two writers can both
        read, then one fails to upgrade its transaction.
        """

        connection = self._open_connection()
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        """A read-only connection for reporting, debugging, and the API.

        Read-only is enforced by SQLite, not by this docstring, and in two
        independent ways:

        * the handle is opened with the ``mode=ro`` URI flag, so writes to
          the database it opened fail with "attempt to write a readonly
          database";
        * a **statement authorizer** allows reads and refuses everything
          else, in every schema.

        The second one is not belt-and-braces.  ``mode=ro`` protects the
        file it opened and says nothing about any other: a caller could
        turn ``query_only`` off, ``ATTACH`` this very database under a
        second name, and write through the alias — committing data with no
        run log at all.  ``ATTACH`` and ``DETACH`` are denied outright, as
        are all DML, all DDL, and any pragma that could re-enable writing,
        so ``main``, ``temp``, and any alias are equally out of reach.

        The authorizer is installed *here only*.  Internal transactions use
        the private writable connection, and deliberate manual repair uses
        :meth:`Phase0Admin.connect_writable`; neither is restricted.
        """

        if not self.database_path.exists():
            raise Phase0ValidationError(
                f"no Phase 0 database at {self.database_path}; call migrate() first"
            )
        connection = sqlite3.connect(
            f"file:{self.database_path}?mode=ro", uri=True, timeout=10
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.execute("PRAGMA query_only = ON")
            # Installed last: the two pragmas above are the settings the
            # authorizer then refuses to let anyone change.
            connection.set_authorizer(_read_only_authorizer)
            yield connection
        finally:
            connection.close()

    def migrate(self) -> list[str]:
        """Apply every unapplied migration atomically; return their names."""

        connection = self._open_connection()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            migrations = load_migrations(self.migrations_path)
            if not migrations:
                raise Phase0MigrationError("no migrations were found")
            return apply_migrations(
                connection, migrations, legacy_upgrade=self._upgrade_legacy_v2
            )
        finally:
            connection.close()

    def schema_version(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def applied_migrations(self) -> list[dict[str, Any]]:
        """The migration ledger, oldest first."""

        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT name, version, checksum, applied_at "
                    "FROM schema_migrations ORDER BY version, name"
                )
            ]

    def _upgrade_legacy_v2(self, connection: sqlite3.Connection) -> None:
        """Convert the originally published v2 schema before migration 003."""
        raw_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(raw_items)")
        }
        if "external_id" in raw_columns:
            return

        legacy_tables = {
            "raw_items",
            "stories",
            "themes",
            "run_log",
            "eval_labels",
            "source_state",
            "pipeline_stage_keys",
        }
        existing_tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing = legacy_tables - existing_tables
        if missing:
            raise Phase0MigrationError(
                "legacy v2 database is missing required tables: "
                + ", ".join(sorted(missing))
            )

        for index in (
            "idx_raw_items_ticker_published",
            "idx_stories_ticker_day",
            "idx_themes_ticker_day",
            "idx_run_log_stage_started",
        ):
            connection.execute(f"DROP INDEX IF EXISTS {index}")
        for table in legacy_tables:
            connection.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy_v2")

        for migration_name in (
            "001_initial.sql",
            "002_source_state_and_stage_keys.sql",
        ):
            sql = (self.migrations_path / migration_name).read_text(encoding="utf-8")
            for statement in split_statements(sql):
                connection.execute(statement)

        connection.execute(
            """
            INSERT INTO raw_items (
                id, source, ticker, title, description, url, canonical_url,
                external_id, published_at, fetched_at, ingest_status,
                validation_errors, raw_json
            )
            SELECT
                id,
                source,
                ticker,
                NULLIF(trim(title), ''),
                description,
                NULLIF(trim(url), ''),
                canonical_url,
                NULL,
                CASE
                    WHEN published_at IS NULL THEN NULL
                    WHEN datetime(published_at) IS NOT NULL THEN published_at
                    ELSE NULL
                END,
                CASE
                    WHEN datetime(fetched_at) IS NOT NULL THEN fetched_at
                    ELSE '1970-01-01T00:00:00+00:00'
                END,
                CASE
                    WHEN length(trim(COALESCE(title, ''))) > 0
                     AND length(trim(COALESCE(url, ''))) > 0
                    THEN 'valid'
                    ELSE 'invalid'
                END,
                CASE
                    WHEN datetime(fetched_at) IS NULL
                      OR (
                        published_at IS NOT NULL
                        AND datetime(published_at) IS NULL
                      )
                    THEN json_array(
                        'legacy timestamps normalized during migration'
                    )
                    ELSE '[]'
                END,
                CASE
                    WHEN json_valid(raw_json)
                     AND json_type(raw_json) = 'object'
                    THEN raw_json
                    ELSE json_object('legacy_value', raw_json)
                END
            FROM raw_items_legacy_v2
            """
        )
        connection.execute(
            """
            INSERT INTO raw_item_tickers (
                raw_item_id, ticker, association_type
            )
            SELECT id, trim(upper(ticker)), 'source'
            FROM raw_items
            WHERE ticker IS NOT NULL AND length(trim(ticker)) > 0
            """
        )
        connection.execute(
            """
            INSERT INTO stories (
                id, ticker, trading_day, canonical_title, embedding,
                outlet_count, member_ids
            )
            SELECT
                id,
                ticker,
                CASE
                    WHEN date(trading_day) = trading_day
                    THEN trading_day
                    ELSE '1970-01-01'
                END,
                canonical_title,
                embedding,
                CASE WHEN outlet_count > 0 THEN outlet_count ELSE 1 END,
                CASE
                    WHEN json_valid(member_ids)
                     AND json_type(member_ids) = 'array'
                    THEN member_ids
                    ELSE '[]'
                END
            FROM stories_legacy_v2
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO story_members (
                story_id, raw_item_id, position
            )
            SELECT
                stories_legacy_v2.id,
                raw_items.id,
                CAST(member.key AS INTEGER)
            FROM stories_legacy_v2
            JOIN json_each(
                CASE
                    WHEN json_valid(stories_legacy_v2.member_ids)
                     AND json_type(stories_legacy_v2.member_ids) = 'array'
                    THEN stories_legacy_v2.member_ids
                    ELSE '[]'
                END
            ) AS member
            JOIN raw_items ON raw_items.id = CAST(member.value AS INTEGER)
            WHERE member.type = 'integer'
            """
        )
        connection.execute(
            """
            INSERT INTO themes (
                id, ticker, trading_day, label, summary, citations,
                salience_rank, status, centroid, content_hash,
                pipeline_version
            )
            SELECT
                id,
                ticker,
                CASE
                    WHEN date(trading_day) = trading_day
                    THEN trading_day
                    ELSE '1970-01-01'
                END,
                label,
                summary,
                CASE
                    WHEN json_valid(citations)
                     AND json_type(citations) = 'array'
                    THEN citations
                    ELSE '[]'
                END,
                CASE WHEN salience_rank > 0 THEN salience_rank ELSE 1 END,
                CASE
                    WHEN status IN (
                        'pending', 'ready', 'degraded', 'failed'
                    )
                    THEN status
                    ELSE 'failed'
                END,
                centroid,
                content_hash,
                pipeline_version
            FROM themes_legacy_v2
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO theme_stories (theme_id, story_id)
            SELECT themes.id, stories.id
            FROM themes
            JOIN stories
              ON stories.ticker = themes.ticker
             AND stories.trading_day = themes.trading_day
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO theme_citations (
                theme_id, raw_item_id
            )
            SELECT
                themes_legacy_v2.id,
                raw_items.id
            FROM themes_legacy_v2
            JOIN json_each(
                CASE
                    WHEN json_valid(themes_legacy_v2.citations)
                     AND json_type(themes_legacy_v2.citations) = 'array'
                    THEN themes_legacy_v2.citations
                    ELSE '[]'
                END
            ) AS citation
            JOIN raw_items
              ON raw_items.id = CAST(citation.value AS INTEGER)
            WHERE citation.type = 'integer'
              AND EXISTS (
                SELECT 1
                FROM theme_stories
                JOIN story_members
                  ON story_members.story_id = theme_stories.story_id
                WHERE theme_stories.theme_id = themes_legacy_v2.id
                  AND story_members.raw_item_id = raw_items.id
              )
            """
        )
        connection.execute(
            """
            INSERT INTO run_log (
                id, run_id, stage, counts, duration_ms, errors,
                started_at, completed_at, status, trading_day,
                pipeline_version
            )
            SELECT
                id,
                run_id,
                stage,
                CASE WHEN json_valid(counts) THEN counts ELSE '{}' END,
                CASE WHEN duration_ms >= 0 THEN duration_ms ELSE 0 END,
                CASE WHEN json_valid(errors) THEN errors ELSE '[]' END,
                CASE
                    WHEN datetime(started_at) IS NOT NULL THEN started_at
                    ELSE '1970-01-01T00:00:00+00:00'
                END,
                CASE
                    WHEN datetime(completed_at) IS NOT NULL THEN completed_at
                    ELSE '1970-01-01T00:00:00+00:00'
                END,
                CASE
                    WHEN status IN ('success', 'degraded', 'failed')
                    THEN status
                    ELSE 'failed'
                END,
                CASE
                    WHEN date(trading_day) = trading_day
                    THEN trading_day
                    ELSE '1970-01-01'
                END,
                pipeline_version
            FROM run_log_legacy_v2
            """
        )
        connection.execute(
            """
            INSERT INTO eval_labels (
                id, label_type, item_a_id, item_b_id, reviewer,
                label, notes, created_at
            )
            SELECT
                eval_labels_legacy_v2.id,
                label_type,
                item_a_id,
                item_b_id,
                COALESCE(reviewer, 'legacy'),
                COALESCE(label, 'unlabeled'),
                notes,
                CASE
                    WHEN datetime(created_at) IS NOT NULL THEN created_at
                    ELSE '1970-01-01T00:00:00+00:00'
                END
            FROM eval_labels_legacy_v2
            JOIN raw_items AS item_a ON item_a.id = item_a_id
            JOIN raw_items AS item_b ON item_b.id = item_b_id
            WHERE item_a_id <> item_b_id
            """
        )
        connection.execute(
            """
            INSERT INTO source_state (
                source, etag, last_modified, last_checked_at,
                last_success_at, metadata
            )
            SELECT
                source,
                etag,
                last_modified,
                CASE
                    WHEN datetime(last_checked_at) IS NOT NULL
                    THEN last_checked_at
                    ELSE '1970-01-01T00:00:00+00:00'
                END,
                CASE
                    WHEN datetime(last_success_at) IS NOT NULL
                    THEN last_success_at
                    ELSE NULL
                END,
                CASE WHEN json_valid(metadata) THEN metadata ELSE '{}' END
            FROM source_state_legacy_v2
            """
        )
        connection.execute(
            """
            INSERT INTO pipeline_stage_keys (
                stage, ticker, trading_day, pipeline_version,
                status, run_id, updated_at
            )
            SELECT
                stage,
                ticker,
                CASE
                    WHEN date(trading_day) = trading_day
                    THEN trading_day
                    ELSE '1970-01-01'
                END,
                pipeline_version,
                CASE
                    WHEN status IN (
                        'running', 'success', 'degraded', 'failed'
                    )
                    THEN status
                    ELSE 'failed'
                END,
                run_id,
                CASE
                    WHEN datetime(updated_at) IS NOT NULL THEN updated_at
                    ELSE '1970-01-01T00:00:00+00:00'
                END
            FROM pipeline_stage_keys_legacy_v2
            """
        )

        for table in sorted(legacy_tables, reverse=True):
            connection.execute(f"DROP TABLE {table}_legacy_v2")

    # ------------------------------------------------------------------
    # Ticker universe
    # ------------------------------------------------------------------

    def supported_tickers(self) -> list[dict[str, Any]]:
        """The authoritative Phase 0 universe, in spec order."""

        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT ticker, display_name, position "
                    "FROM supported_tickers ORDER BY position"
                )
            ]

    # ------------------------------------------------------------------
    # Raw items
    # ------------------------------------------------------------------

    def _prepare_raw_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        source = str(item.get("source") or "").strip()
        canonical_url = str(item.get("canonical_url") or "").strip()
        if not source or not canonical_url:
            raise Phase0ValidationError("raw item requires source and canonical_url")
        ingest_status = str(item.get("ingest_status") or "valid")
        if ingest_status not in INGEST_STATUSES:
            raise Phase0ValidationError("invalid raw-item ingest_status")
        title = str(item.get("title") or "").strip() or None
        url = str(item.get("url") or "").strip() or None
        if ingest_status == "valid" and (title is None or url is None):
            raise Phase0ValidationError("valid raw items require title and url")
        raw_payload = item.get("raw_json", item)
        # Publisher evidence, stored exactly as supplied; see
        # serialize_raw_evidence for why this one is not redacted.
        raw_json = serialize_raw_evidence(raw_payload, "raw_json")
        validation_errors = serialize_operational_metadata(
            list(item.get("validation_errors") or []), "validation_errors", list
        )
        return {
            "source": source,
            "ticker": normalize_ticker(item.get("ticker"), optional=True),
            "title": title,
            "description": str(item.get("description") or "").strip() or None,
            "url": url,
            "canonical_url": canonical_url,
            "external_id": str(item.get("external_id") or "").strip() or None,
            "published_at": _normalize_datetime(
                item.get("published_at"), "published_at", optional=True
            ),
            "fetched_at": _normalize_datetime(
                item.get("fetched_at") or utc_now(), "fetched_at"
            ),
            "ingest_status": ingest_status,
            "validation_errors": validation_errors,
            "raw_json": raw_json,
            "tickers": sorted(
                {
                    normalize_ticker(ticker)
                    for ticker in item.get("tickers", [])
                    if str(ticker).strip()
                }
            ),
            "candidate_tickers": item.get("candidate_tickers") or [],
        }

    @staticmethod
    def _insert_raw_item(
        connection: sqlite3.Connection, values: Mapping[str, Any]
    ) -> InsertResult:
        cursor = connection.execute(
            """
            INSERT INTO raw_items (
                source, ticker, title, description, url, canonical_url,
                external_id, published_at, fetched_at, ingest_status,
                validation_errors, raw_json
            ) VALUES (
                :source, :ticker, :title, :description, :url, :canonical_url,
                :external_id, :published_at, :fetched_at, :ingest_status,
                :validation_errors, :raw_json
            )
            ON CONFLICT(source, canonical_url) DO NOTHING
            """,
            values,
        )
        if cursor.rowcount:
            item_id = int(cursor.lastrowid)
            inserted = True
        else:
            row = connection.execute(
                "SELECT id FROM raw_items WHERE source = ? AND canonical_url = ?",
                (values["source"], values["canonical_url"]),
            ).fetchone()
            item_id = int(row["id"])
            inserted = False

        tickers = set(values.get("tickers") or [])
        if values.get("ticker"):
            tickers.add(values["ticker"])
        for ticker in sorted(tickers):
            connection.execute(
                """
                INSERT OR IGNORE INTO raw_item_tickers
                    (raw_item_id, ticker, association_type)
                VALUES (?, ?, 'source')
                """,
                (item_id, ticker),
            )
        for candidate in values.get("candidate_tickers") or []:
            if isinstance(candidate, Mapping):
                ticker = normalize_ticker(
                    candidate.get("ticker"), field="candidate ticker"
                )
                reason = str(candidate.get("reason") or "relevance_match")
            else:
                ticker = normalize_ticker(candidate, field="candidate ticker")
                reason = "relevance_match"
            connection.execute(
                """
                INSERT INTO raw_item_candidates (raw_item_id, ticker, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(raw_item_id, ticker)
                DO UPDATE SET reason = excluded.reason
                """,
                (item_id, ticker, reason),
            )
        return InsertResult(item_id, inserted)

    def _insert_raw_items_unlogged(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        source_state: Mapping[str, Any] | None = None,
    ) -> list[InsertResult]:
        """Persist a batch and optional source state in one transaction.

        Private on purpose: it writes without a run, so it is reachable only
        through :attr:`Phase0Repository.admin`.  The pipeline entrypoint is
        :meth:`ingest_raw_items`.
        """

        prepared = [self._prepare_raw_item(item) for item in items]
        with self._connect() as connection:
            results = [self._insert_raw_item(connection, values) for values in prepared]
            if source_state is not None:
                self._set_source_state(connection, source_state)
            return results

    def raw_items_for_day(
        self, trading_day: str | date, ticker: str | None = None
    ) -> list[dict[str, Any]]:
        day = _normalize_day(trading_day)
        query = """
            SELECT DISTINCT raw_items.*
            FROM raw_items
            LEFT JOIN raw_item_tickers
                ON raw_item_tickers.raw_item_id = raw_items.id
            WHERE substr(COALESCE(published_at, fetched_at), 1, 10) = ?
        """
        parameters: list[Any] = [day]
        if ticker:
            normalized_ticker = normalize_ticker(ticker)
            query += " AND (raw_items.ticker = ? OR raw_item_tickers.ticker = ?)"
            parameters.extend([normalized_ticker, normalized_ticker])
        query += " ORDER BY COALESCE(published_at, fetched_at), raw_items.id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def _update_raw_item_ticker_unlogged(
        self, item_id: int, ticker: str | None
    ) -> None:
        normalized = normalize_ticker(ticker, optional=True)
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE raw_items SET ticker = ? WHERE id = ?", (normalized, item_id)
            )
            if cursor.rowcount != 1:
                raise Phase0ValidationError("unknown raw item")
            if normalized:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO raw_item_tickers
                        (raw_item_id, ticker, association_type)
                    VALUES (?, ?, 'relevance')
                    """,
                    (item_id, normalized),
                )

    def raw_item_tickers(self, item_id: int) -> list[str]:
        with self._connect() as connection:
            return [
                row["ticker"]
                for row in connection.execute(
                    """
                    SELECT DISTINCT ticker FROM raw_item_tickers
                    WHERE raw_item_id = ? ORDER BY ticker
                    """,
                    (item_id,),
                )
            ]

    # ------------------------------------------------------------------
    # Source state
    # ------------------------------------------------------------------

    def source_state(self, source: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_state WHERE source = ?", (source,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    @staticmethod
    def validate_source_state(state: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize one source-state update.

        Public because ingestion (#61, #62) builds these payloads and needs
        the same answer the repository will give before it commits.
        """

        if not isinstance(state, Mapping):
            raise Phase0ValidationError("source state must be a mapping")
        source = _require_text(state.get("source"), "source state source")
        checked_at = _normalize_datetime(state.get("checked_at"), "checked_at")
        successful = bool(state.get("successful"))
        status = state.get("status")
        if status is None:
            resolved_status = "success" if successful else "failed"
        else:
            resolved_status = str(status).strip().lower()
            if resolved_status not in SOURCE_STATE_STATUSES:
                raise Phase0ValidationError(
                    f"invalid source-state status: {resolved_status}"
                )
        succeeded = resolved_status in {"success", "partial", "empty"}
        error = state.get("error")
        return {
            "source": source,
            "etag": _optional_text(state.get("etag")),
            "last_modified": _optional_text(state.get("last_modified")),
            "checked_at": checked_at,
            "success_at": checked_at if succeeded else None,
            "status": resolved_status,
            "failed": 0 if succeeded else 1,
            "last_error": None if error is None else redact_text(str(error)),
            "retry_after": _normalize_datetime(
                state.get("retry_after"), "retry_after", optional=True
            ),
            "metadata": _serialize_json(
                redact_secrets(dict(state.get("metadata") or {})),
                "source metadata",
                dict,
            ),
        }

    # Backwards-compatible alias for the pre-review private helper.
    _prepare_source_state = validate_source_state

    @classmethod
    def _set_source_state(
        cls, connection: sqlite3.Connection, state: Mapping[str, Any]
    ) -> None:
        values = cls.validate_source_state(state)
        connection.execute(
            """
            INSERT INTO source_state (
                source, etag, last_modified, last_checked_at,
                last_success_at, metadata, status, consecutive_failures,
                last_error, retry_after
            ) VALUES (
                :source, :etag, :last_modified, :checked_at,
                :success_at, :metadata, :status, :failed,
                :last_error, :retry_after
            )
            ON CONFLICT(source) DO UPDATE SET
                etag = COALESCE(excluded.etag, source_state.etag),
                last_modified = COALESCE(
                    excluded.last_modified, source_state.last_modified
                ),
                last_checked_at = excluded.last_checked_at,
                last_success_at = CASE
                    WHEN excluded.last_success_at IS NOT NULL
                    THEN excluded.last_success_at
                    ELSE source_state.last_success_at
                END,
                metadata = excluded.metadata,
                status = excluded.status,
                consecutive_failures = CASE
                    WHEN excluded.consecutive_failures = 0 THEN 0
                    ELSE source_state.consecutive_failures + 1
                END,
                last_error = excluded.last_error,
                retry_after = excluded.retry_after
            """,
            values,
        )

    def _set_source_state_unlogged(
        self,
        source: str,
        *,
        etag: str | None,
        last_modified: str | None,
        checked_at: str,
        successful: bool,
        metadata: Mapping[str, Any] | None = None,
        status: str | None = None,
        error: Any = None,
        retry_after: str | None = None,
    ) -> None:
        """Write source state with no run attached; see :attr:`admin`."""

        with self._connect() as connection:
            self._set_source_state(
                connection,
                {
                    "source": source,
                    "etag": etag,
                    "last_modified": last_modified,
                    "checked_at": checked_at,
                    "successful": successful,
                    "metadata": metadata or {},
                    "status": status,
                    "error": error,
                    "retry_after": retry_after,
                },
            )

    # ------------------------------------------------------------------
    # Embeddings (nlp.embeddings.EmbeddingRepository)
    # ------------------------------------------------------------------

    def get_embedding(
        self, source_kind: str, source_id: str
    ) -> PersistedEmbedding | None:
        """Return the current embedding for a source, or ``None``."""

        kind = normalize_source_kind(source_kind)
        identity = normalize_source_id(source_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM embeddings WHERE source_kind = ? AND source_id = ?",
                (kind, identity),
            ).fetchone()
        if row is None:
            return None
        return embedding_from_row(row)

    def upsert_embedding(self, embedding: PersistedEmbedding) -> None:
        """Atomically insert or replace one source's embedding.

        This is M1's cache protocol (``nlp.embeddings.EmbeddingRepository``),
        not a pipeline stage: it writes a single derived vector that can be
        recomputed from the raw item at any time.  The batch that *is* a
        pipeline stage is :meth:`persist_embeddings`, which carries a run.
        """

        values = validate_embedding(embedding)
        with self._connect() as connection:
            self._write_embedding(connection, values)

    @staticmethod
    def _write_embedding(
        connection: sqlite3.Connection, values: Mapping[str, Any]
    ) -> None:
        try:
            connection.execute(
                """
                INSERT INTO embeddings (
                    source_kind, source_id, model_name, model_revision,
                    dimension, dtype, input_fingerprint, vector_blob,
                    created_at, updated_at
                ) VALUES (
                    :source_kind, :source_id, :model_name, :model_revision,
                    :dimension, :dtype, :input_fingerprint, :vector_blob,
                    :now, :now
                )
                ON CONFLICT(source_kind, source_id) DO UPDATE SET
                    model_name = excluded.model_name,
                    model_revision = excluded.model_revision,
                    dimension = excluded.dimension,
                    dtype = excluded.dtype,
                    input_fingerprint = excluded.input_fingerprint,
                    vector_blob = excluded.vector_blob,
                    updated_at = excluded.updated_at
                """,
                {**values, "now": utc_now()},
            )
        except sqlite3.IntegrityError as exc:
            raise EmbeddingPersistenceError(str(exc)) from exc

    def delete_embedding(self, source_kind: str, source_id: str) -> bool:
        """Drop one source's embedding; ``True`` when a row was removed."""

        kind = normalize_source_kind(source_kind)
        identity = normalize_source_id(source_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM embeddings WHERE source_kind = ? AND source_id = ?",
                (kind, identity),
            )
            return bool(cursor.rowcount)

    # ------------------------------------------------------------------
    # Stories
    # ------------------------------------------------------------------

    def _insert_story_unlogged(
        self,
        *,
        ticker: str,
        trading_day: str | date,
        canonical_title: str,
        member_ids: Sequence[int],
        pipeline_version: str,
        embedding: bytes | None = None,
        outlet_count: int = 1,
    ) -> int:
        """Insert one story directly.  ``pipeline_version`` is required.

        It used to default to NULL, which migration 011 removed as a
        category: a story with no stated version could join a v1 theme and
        a v2 theme at once.  Fixtures must now say which partition they are
        seeding.
        """

        if not canonical_title.strip() or not member_ids:
            raise Phase0ValidationError(
                "stories require a title and at least one member"
            )
        day = _normalize_day(trading_day)
        normalized_ticker = normalize_ticker(ticker)
        version = _require_text(pipeline_version, "pipeline_version")
        members = [int(item_id) for item_id in dict.fromkeys(member_ids)]
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO stories (
                    ticker, trading_day, canonical_title, embedding,
                    outlet_count, member_ids, updated_at, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_ticker,
                    day,
                    canonical_title.strip(),
                    _optional_blob(embedding, "embedding"),
                    _require_int(outlet_count, "outlet_count", minimum=1),
                    _serialize_json(members, "member_ids", list),
                    utc_now(),
                    version,
                ),
            )
            story_id = int(cursor.lastrowid)
            for position, item_id in enumerate(members):
                connection.execute(
                    """
                    INSERT INTO story_members (story_id, raw_item_id, position)
                    VALUES (?, ?, ?)
                    """,
                    (story_id, item_id, position),
                )
            return story_id

    @staticmethod
    def _prepare_story(record: StoryRecord) -> dict[str, Any]:
        if not isinstance(record, StoryRecord):
            raise Phase0ValidationError("stories must be StoryRecord instances")
        fingerprint = _require_text(record.cluster_fingerprint, "cluster_fingerprint")
        title = _require_text(record.canonical_title, "canonical_title")
        if not record.members:
            raise Phase0ValidationError("a story requires at least one member")
        members: list[dict[str, Any]] = []
        seen: set[int] = set()
        for position, member in enumerate(record.members):
            if not isinstance(member, StoryMemberRecord):
                raise Phase0ValidationError(
                    "story members must be StoryMemberRecord instances"
                )
            raw_item_id = _require_int(member.raw_item_id, "raw_item_id", minimum=1)
            if raw_item_id in seen:
                raise Phase0ValidationError(
                    f"story {fingerprint} lists raw item {raw_item_id} twice"
                )
            seen.add(raw_item_id)
            members.append(
                {
                    "raw_item_id": raw_item_id,
                    "position": _require_int(
                        member.position if member.position else position,
                        "member position",
                        minimum=0,
                    ),
                    "outlet": _optional_text(member.outlet),
                    "url": _optional_text(member.url),
                    "canonical_url": _optional_text(member.canonical_url),
                    "match_reason": sanitize_diagnostic_scalar(
                        _optional_text(member.match_reason), "match_reason"
                    ),
                    "quarantined": 1 if member.quarantined else 0,
                }
            )
        canonical_item_id = record.canonical_item_id
        if canonical_item_id is not None:
            canonical_item_id = _require_int(
                canonical_item_id, "canonical_item_id", minimum=1
            )
            if canonical_item_id not in seen:
                raise Phase0ValidationError(
                    "canonical_item_id must be one of the story members"
                )
        stage = _require_text(record.stage, "stage")
        if stage not in STORY_STAGES:
            raise Phase0ValidationError(f"unsupported story stage: {stage}")
        conflicts = []
        for conflict in record.provider_conflicts:
            if not isinstance(conflict, ProviderConflictRecord):
                raise Phase0ValidationError(
                    "provider conflicts must be ProviderConflictRecord instances"
                )
            conflicts.append(
                {
                    # Identity — policy B: refused, never rewritten, because
                    # these are what a conflict is looked up by.
                    "provider_namespace": require_safe_identifier_scalar(
                        _require_text(
                            conflict.provider_namespace, "provider_namespace"
                        ),
                        "provider_namespace",
                    ),
                    "provider_item_id": require_safe_identifier_scalar(
                        _require_text(conflict.provider_item_id, "provider_item_id"),
                        "provider_item_id",
                    ),
                    "item_ids": serialize_operational_metadata(
                        [
                            require_safe_identifier_scalar(
                                value, "provider conflict item id"
                            )
                            for value in conflict.item_ids
                        ],
                        "provider conflict item_ids",
                        list,
                    ),
                    "fields": serialize_operational_metadata(
                        [
                            require_safe_identifier_scalar(
                                value, "provider conflict field"
                            )
                            for value in conflict.fields
                        ],
                        "provider conflict fields",
                        list,
                    ),
                }
            )
        merges = []
        for merge in record.semantic_merges:
            if not isinstance(merge, SemanticMergeRecord):
                raise Phase0ValidationError(
                    "semantic merges must be SemanticMergeRecord instances"
                )
            merges.append(
                {
                    # Story keys are identity (policy B); the reason is
                    # free-form diagnostics (policy A) and is redacted.
                    "left_story_key": require_safe_identifier_scalar(
                        _require_text(merge.left_story_key, "left_story_key"),
                        "left_story_key",
                    ),
                    "right_story_key": require_safe_identifier_scalar(
                        _require_text(merge.right_story_key, "right_story_key"),
                        "right_story_key",
                    ),
                    "similarity": _optional_float(
                        merge.similarity, "similarity", low=-1.0, high=1.0
                    ),
                    "reason": sanitize_diagnostic_scalar(
                        _require_text(merge.reason, "merge reason"), "merge reason"
                    ),
                }
            )
        return {
            "cluster_fingerprint": fingerprint,
            "canonical_title": title,
            "members": members,
            "member_ids": sorted(seen),
            "canonical_item_id": canonical_item_id,
            "outlet_count": _require_int(
                record.outlet_count, "outlet_count", minimum=1
            ),
            "published_at": _normalize_datetime(
                record.published_at, "published_at", optional=True
            ),
            "canonical_url": _optional_text(record.canonical_url),
            "source": _optional_text(record.source),
            "outlet": _optional_text(record.outlet),
            "content_hash": _optional_text(record.content_hash) or fingerprint,
            # Algorithm/config/model identity keys replayability (policy B).
            "algorithm_version": validate_safe_identifier_scalar(
                _optional_text(record.algorithm_version), "algorithm_version"
            ),
            "config_fingerprint": validate_safe_identifier_scalar(
                _optional_text(record.config_fingerprint), "config_fingerprint"
            ),
            "stage": stage,
            "model_name": validate_safe_identifier_scalar(
                _optional_text(record.model_name), "model_name"
            ),
            "model_revision": validate_safe_identifier_scalar(
                _optional_text(record.model_revision), "model_revision"
            ),
            "embedding_dimension": (
                None
                if record.embedding_dimension is None
                else _require_int(
                    record.embedding_dimension, "embedding_dimension", minimum=1
                )
            ),
            "quarantined": 1 if record.quarantined else 0,
            # Free-form explanation (policy A).
            "semantic_skip_reason": sanitize_diagnostic_scalar(
                _optional_text(record.semantic_skip_reason), "semantic_skip_reason"
            ),
            "member_story_keys": serialize_operational_metadata(
                [
                    require_safe_identifier_scalar(key, "member story key")
                    for key in record.member_story_keys
                ],
                "member_story_keys",
                list,
            ),
            "embedding": _optional_blob(record.embedding, "embedding"),
            "provider_conflicts": conflicts,
            "semantic_merges": merges,
        }

    @staticmethod
    def _story_signature(values: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            values["canonical_title"],
            values["outlet_count"],
            values["published_at"],
            values["canonical_url"],
            values["source"],
            values["outlet"],
            values["content_hash"],
            values["algorithm_version"],
            values["config_fingerprint"],
            values["stage"],
            values["model_name"],
            values["model_revision"],
            values["embedding_dimension"],
            values["quarantined"],
            values["semantic_skip_reason"],
            values["member_story_keys"],
            values["canonical_item_id"],
            tuple(values["member_ids"]),
        )

    def reconcile_stories(
        self,
        *,
        run: Any,
        ticker: str,
        trading_day: str | date,
        pipeline_version: str,
        stories: Sequence[StoryRecord],
        delete_obsolete: bool = True,
        terminal: bool = False,
    ) -> ReconciliationReport:
        """Replace one ticker/trading-day's canonical stories atomically.

        This is the boundary issue #68's runner uses: a whole M2/M3 result
        goes in, and what actually changed comes back.  It is deliberately
        not a loop of independent upserts — a partial rewrite of a day is
        exactly the state AC-8's replay guarantee cannot tolerate.

        ``run`` is the :class:`StageRunContext` from :meth:`stage_run`.  The
        stories and the ``run_log`` row commit in one transaction, and the
        counts are derived here rather than supplied by the caller.  Every
        member raw item must already belong to the run's ticker-day, so a
        story cannot quietly drag foreign evidence into this partition.
        """

        normalized_ticker = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        with self._logged_mutation(
            run,
            operation="reconcile_stories",
            ticker=normalized_ticker,
            trading_day=day,
            pipeline_version=version,
            terminal=terminal,
        ) as (connection, context):
            self._assert_members_in_partition(
                connection,
                stories,
                ticker=normalized_ticker,
                trading_day=day,
            )
            report = self._reconcile_stories_unlogged(
                ticker=normalized_ticker,
                trading_day=day,
                pipeline_version=version,
                stories=stories,
                delete_obsolete=delete_obsolete,
                connection=connection,
            )
            context._record_outcome(
                success=len(report.inserted) + len(report.updated),
                partial=len(report.unchanged),
            )
            context._merge_counts(report.counts)
            return report

    @staticmethod
    def _assert_members_in_partition(
        connection: sqlite3.Connection,
        stories: Sequence[StoryRecord],
        *,
        ticker: str,
        trading_day: str,
    ) -> None:
        """Every member raw item must sit in this ticker-day already."""

        for story in stories:
            for member in getattr(story, "members", ()):  # validated later
                raw_item_id = getattr(member, "raw_item_id", None)
                if raw_item_id is None:
                    continue
                row = connection.execute(
                    "SELECT ticker, "
                    "substr(COALESCE(published_at, fetched_at), 1, 10) AS day "
                    "FROM raw_items WHERE id = ?",
                    (raw_item_id,),
                ).fetchone()
                if row is None:
                    continue  # the foreign key reports this one precisely.
                if row["ticker"] is not None and str(row["ticker"]) != ticker:
                    raise Phase0RunContextError(
                        f"raw item {raw_item_id} belongs to {row['ticker']} but "
                        f"this reconciliation covers {ticker}"
                    )
                if str(row["day"]) != trading_day:
                    raise Phase0RunContextError(
                        f"raw item {raw_item_id} falls on {row['day']} but this "
                        f"reconciliation covers {trading_day}"
                    )

    def _reconcile_stories_unlogged(
        self,
        *,
        ticker: str,
        trading_day: str | date,
        pipeline_version: str,
        stories: Sequence[StoryRecord],
        delete_obsolete: bool = True,
        connection: sqlite3.Connection | None = None,
    ) -> ReconciliationReport:
        """Reconcile stories with no run attached; see :attr:`admin`.

        Themes are derived from stories, so a structural change (a story
        inserted, removed, or re-membered) invalidates the day's theme set
        inside the same transaction rather than leaving themes citing a
        membership that no longer exists.
        """

        normalized_ticker = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        prepared = [self._prepare_story(record) for record in stories]
        fingerprints = [values["cluster_fingerprint"] for values in prepared]
        duplicates = {key for key in fingerprints if fingerprints.count(key) > 1}
        if duplicates:
            raise Phase0ValidationError(
                f"duplicate cluster fingerprints: {sorted(duplicates)}"
            )
        incoming = dict(zip(fingerprints, prepared))

        with self._write_scope(connection) as connection:
            existing = {
                str(row["cluster_fingerprint"]): row
                for row in connection.execute(
                    """
                    SELECT * FROM stories
                    WHERE ticker = ? AND trading_day = ? AND pipeline_version = ?
                      AND cluster_fingerprint IS NOT NULL
                    """,
                    (normalized_ticker, day, version),
                )
            }
            existing_members = {
                fingerprint: self._member_ids(connection, int(row["id"]))
                for fingerprint, row in existing.items()
            }

            inserted: list[int] = []
            updated: list[int] = []
            unchanged: list[int] = []
            structural: set[int] = set()
            removed_members = 0

            for fingerprint, values in incoming.items():
                row = existing.get(fingerprint)
                if row is None:
                    story_id = self._insert_reconciled_story(
                        connection, normalized_ticker, day, version, values
                    )
                    inserted.append(story_id)
                    structural.add(story_id)
                    continue
                story_id = int(row["id"])
                stored = self._stored_story_signature(
                    row, existing_members[fingerprint]
                )
                if stored == self._story_signature(values):
                    unchanged.append(story_id)
                    continue
                member_change = set(existing_members[fingerprint]) != set(
                    values["member_ids"]
                )
                if member_change:
                    structural.add(story_id)
                removed_members += self._update_reconciled_story(
                    connection, story_id, values
                )
                updated.append(story_id)

            obsolete = [
                (fingerprint, int(row["id"]))
                for fingerprint, row in existing.items()
                if fingerprint not in incoming
            ]
            deleted: list[int] = []
            invalidated: list[int] = []
            invalidated_themes: tuple[int, ...] = ()
            if obsolete or structural:
                invalidated_themes = self._invalidate_theme_set(
                    connection, normalized_ticker, day, version
                )
            if obsolete and delete_obsolete:
                for _, story_id in obsolete:
                    if self._story_is_referenced(connection, story_id):
                        connection.execute(
                            "UPDATE stories SET invalidated_at = ?, updated_at = ? "
                            "WHERE id = ?",
                            (utc_now(), utc_now(), story_id),
                        )
                        invalidated.append(story_id)
                    else:
                        self._delete_story(connection, story_id)
                        deleted.append(story_id)
            elif obsolete:
                for _, story_id in obsolete:
                    connection.execute(
                        "UPDATE stories SET invalidated_at = ?, updated_at = ? "
                        "WHERE id = ?",
                        (utc_now(), utc_now(), story_id),
                    )
                    invalidated.append(story_id)

            return ReconciliationReport(
                inserted=tuple(inserted),
                updated=tuple(updated),
                unchanged=tuple(unchanged),
                deleted=tuple(deleted),
                invalidated=tuple(invalidated),
                removed_members=removed_members,
                invalidated_theme_ids=invalidated_themes,
            )

    @staticmethod
    def _member_ids(connection: sqlite3.Connection, story_id: int) -> list[int]:
        return [
            int(row["raw_item_id"])
            for row in connection.execute(
                "SELECT raw_item_id FROM story_members WHERE story_id = ? "
                "ORDER BY raw_item_id",
                (story_id,),
            )
        ]

    @classmethod
    def _stored_story_signature(
        cls, row: sqlite3.Row, member_ids: Sequence[int]
    ) -> tuple[Any, ...]:
        return (
            row["canonical_title"],
            int(row["outlet_count"]),
            row["published_at"],
            row["canonical_url"],
            row["source"],
            row["outlet"],
            row["content_hash"],
            row["algorithm_version"],
            row["config_fingerprint"],
            row["stage"],
            row["model_name"],
            row["model_revision"],
            row["embedding_dimension"],
            int(row["quarantined"]),
            row["semantic_skip_reason"],
            row["member_story_keys"],
            None if row["canonical_item_id"] is None else int(row["canonical_item_id"]),
            tuple(sorted(member_ids)),
        )

    def _insert_reconciled_story(
        self,
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        version: str,
        values: Mapping[str, Any],
    ) -> int:
        now = utc_now()
        cursor = connection.execute(
            """
            INSERT INTO stories (
                ticker, trading_day, canonical_title, embedding, outlet_count,
                member_ids, cluster_fingerprint, pipeline_version, stage,
                canonical_url, source, outlet, published_at, content_hash,
                algorithm_version, config_fingerprint, model_name,
                model_revision, embedding_dimension, quarantined,
                semantic_skip_reason, member_story_keys, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?
            )
            """,
            (
                ticker,
                day,
                values["canonical_title"],
                values["embedding"],
                values["outlet_count"],
                json.dumps(values["member_ids"], separators=(",", ":")),
                values["cluster_fingerprint"],
                version,
                values["stage"],
                values["canonical_url"],
                values["source"],
                values["outlet"],
                values["published_at"],
                values["content_hash"],
                values["algorithm_version"],
                values["config_fingerprint"],
                values["model_name"],
                values["model_revision"],
                values["embedding_dimension"],
                values["quarantined"],
                values["semantic_skip_reason"],
                values["member_story_keys"],
                now,
            ),
        )
        story_id = int(cursor.lastrowid)
        self._write_story_children(connection, story_id, values)
        return story_id

    def _update_reconciled_story(
        self,
        connection: sqlite3.Connection,
        story_id: int,
        values: Mapping[str, Any],
    ) -> int:
        # The canonical member is released first so membership can be
        # replaced without tripping the canonical-member trigger.
        connection.execute(
            "UPDATE stories SET canonical_item_id = NULL WHERE id = ?", (story_id,)
        )
        keep = set(values["member_ids"])
        current = set(self._member_ids(connection, story_id))
        removed = sorted(current - keep)
        for raw_item_id in removed:
            connection.execute(
                "DELETE FROM story_members WHERE story_id = ? AND raw_item_id = ?",
                (story_id, raw_item_id),
            )
        connection.execute(
            """
            UPDATE stories SET
                canonical_title = ?, embedding = ?, outlet_count = ?,
                member_ids = ?, stage = ?, canonical_url = ?, source = ?,
                outlet = ?, published_at = ?, content_hash = ?,
                algorithm_version = ?, config_fingerprint = ?, model_name = ?,
                model_revision = ?, embedding_dimension = ?, quarantined = ?,
                semantic_skip_reason = ?, member_story_keys = ?,
                invalidated_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (
                values["canonical_title"],
                values["embedding"],
                values["outlet_count"],
                json.dumps(values["member_ids"], separators=(",", ":")),
                values["stage"],
                values["canonical_url"],
                values["source"],
                values["outlet"],
                values["published_at"],
                values["content_hash"],
                values["algorithm_version"],
                values["config_fingerprint"],
                values["model_name"],
                values["model_revision"],
                values["embedding_dimension"],
                values["quarantined"],
                values["semantic_skip_reason"],
                values["member_story_keys"],
                utc_now(),
                story_id,
            ),
        )
        connection.execute(
            "DELETE FROM story_provider_conflicts WHERE story_id = ?", (story_id,)
        )
        connection.execute(
            "DELETE FROM story_semantic_merges WHERE story_id = ?", (story_id,)
        )
        self._write_story_children(connection, story_id, values)
        return len(removed)

    def _write_story_children(
        self,
        connection: sqlite3.Connection,
        story_id: int,
        values: Mapping[str, Any],
    ) -> None:
        for member in values["members"]:
            connection.execute(
                """
                INSERT INTO story_members (
                    story_id, raw_item_id, position, outlet, url,
                    canonical_url, match_reason, quarantined
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(story_id, raw_item_id) DO UPDATE SET
                    position = excluded.position,
                    outlet = excluded.outlet,
                    url = excluded.url,
                    canonical_url = excluded.canonical_url,
                    match_reason = excluded.match_reason,
                    quarantined = excluded.quarantined
                """,
                (
                    story_id,
                    member["raw_item_id"],
                    member["position"],
                    member["outlet"],
                    member["url"],
                    member["canonical_url"],
                    member["match_reason"],
                    member["quarantined"],
                ),
            )
        for conflict in values["provider_conflicts"]:
            connection.execute(
                """
                INSERT INTO story_provider_conflicts (
                    story_id, provider_namespace, provider_item_id,
                    item_ids, fields
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    story_id,
                    conflict["provider_namespace"],
                    conflict["provider_item_id"],
                    conflict["item_ids"],
                    conflict["fields"],
                ),
            )
        for merge in values["semantic_merges"]:
            connection.execute(
                """
                INSERT INTO story_semantic_merges (
                    story_id, left_story_key, right_story_key, similarity,
                    reason
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    story_id,
                    merge["left_story_key"],
                    merge["right_story_key"],
                    merge["similarity"],
                    merge["reason"],
                ),
            )
        if values["canonical_item_id"] is not None:
            connection.execute(
                "UPDATE stories SET canonical_item_id = ? WHERE id = ?",
                (values["canonical_item_id"], story_id),
            )

    @staticmethod
    def _story_is_referenced(connection: sqlite3.Connection, story_id: int) -> bool:
        row = connection.execute(
            "SELECT 1 FROM theme_stories WHERE story_id = ? LIMIT 1", (story_id,)
        ).fetchone()
        return row is not None

    @staticmethod
    def _delete_story(connection: sqlite3.Connection, story_id: int) -> None:
        connection.execute(
            "UPDATE stories SET canonical_item_id = NULL WHERE id = ?", (story_id,)
        )
        connection.execute("DELETE FROM story_members WHERE story_id = ?", (story_id,))
        connection.execute("DELETE FROM stories WHERE id = ?", (story_id,))

    @staticmethod
    def _delete_themes(
        connection: sqlite3.Connection, theme_ids: Sequence[int]
    ) -> None:
        """Delete themes in the order the integrity triggers require."""

        for theme_id in theme_ids:
            connection.execute(
                "DELETE FROM theme_citations WHERE theme_id = ?", (theme_id,)
            )
        for theme_id in theme_ids:
            connection.execute(
                "DELETE FROM theme_stories WHERE theme_id = ?", (theme_id,)
            )
        for theme_id in theme_ids:
            connection.execute("DELETE FROM themes WHERE id = ?", (theme_id,))

    @classmethod
    def _invalidate_theme_set(
        cls,
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        version: str,
    ) -> tuple[int, ...]:
        theme_ids = [
            int(row["id"])
            for row in connection.execute(
                "SELECT id FROM themes WHERE ticker = ? AND trading_day = ? "
                "AND pipeline_version = ? ORDER BY id",
                (ticker, day, version),
            )
        ]
        cls._delete_themes(connection, theme_ids)
        connection.execute(
            "DELETE FROM theme_sets WHERE ticker = ? AND trading_day = ? "
            "AND pipeline_version = ?",
            (ticker, day, version),
        )
        return tuple(theme_ids)

    def stories_for_day(
        self,
        trading_day: str | date,
        ticker: str | None = None,
        *,
        pipeline_version: str | None = None,
        include_invalidated: bool = False,
    ) -> list[dict[str, Any]]:
        day = _normalize_day(trading_day)
        query = "SELECT * FROM stories WHERE trading_day = ?"
        parameters: list[Any] = [day]
        if ticker is not None:
            query += " AND ticker = ?"
            parameters.append(normalize_ticker(ticker))
        if pipeline_version is not None:
            query += " AND pipeline_version = ?"
            parameters.append(_require_text(pipeline_version, "pipeline_version"))
        if not include_invalidated:
            query += " AND invalidated_at IS NULL"
        query += " ORDER BY id"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    # ------------------------------------------------------------------
    # Themes
    # ------------------------------------------------------------------

    def _insert_theme_unlogged(
        self,
        *,
        ticker: str,
        trading_day: str | date,
        label: str,
        story_ids: Sequence[int],
        citation_ids: Sequence[int],
        salience_rank: int,
        status: str,
        content_hash: str,
        pipeline_version: str,
        summary: str | None = None,
        centroid: bytes | None = None,
    ) -> int:
        if status not in THEME_STATUSES:
            raise Phase0ValidationError("invalid theme status")
        day = _normalize_day(trading_day)
        normalized_ticker = normalize_ticker(ticker)
        stories = [int(value) for value in dict.fromkeys(story_ids)]
        citations = [int(value) for value in dict.fromkeys(citation_ids)]
        if not label.strip() or not stories:
            raise Phase0ValidationError("themes require a label and at least one story")
        with self._connect() as connection:
            self._assert_story_membership(
                connection, normalized_ticker, day, stories, citations
            )
            cursor = connection.execute(
                """
                INSERT INTO themes (
                    ticker, trading_day, label, summary, citations,
                    salience_rank, status, centroid, content_hash,
                    pipeline_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_ticker,
                    day,
                    label.strip(),
                    summary,
                    _serialize_json(citations, "citations", list),
                    _require_int(salience_rank, "salience_rank", minimum=1),
                    status,
                    _optional_blob(centroid, "centroid"),
                    content_hash,
                    pipeline_version,
                    utc_now(),
                ),
            )
            theme_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
                [(theme_id, story_id) for story_id in stories],
            )
            connection.executemany(
                "INSERT INTO theme_citations (theme_id, raw_item_id) VALUES (?, ?)",
                [(theme_id, item_id) for item_id in citations],
            )
            return theme_id

    @staticmethod
    def _assert_story_membership(
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        story_ids: Sequence[int],
        citation_ids: Sequence[int],
    ) -> None:
        if not story_ids:
            raise Phase0ValidationError("a theme requires at least one story")
        placeholders = ",".join("?" for _ in story_ids)
        story_rows = connection.execute(
            f"SELECT id, ticker, trading_day, invalidated_at FROM stories "
            f"WHERE id IN ({placeholders})",
            list(story_ids),
        ).fetchall()
        if len(story_rows) != len(story_ids) or any(
            row["ticker"] != ticker or row["trading_day"] != day for row in story_rows
        ):
            raise Phase0ValidationError(
                "theme stories must exist for the same ticker/day"
            )
        if any(row["invalidated_at"] is not None for row in story_rows):
            raise Phase0ValidationError("theme stories must not be invalidated")
        member_rows = connection.execute(
            f"SELECT DISTINCT raw_item_id FROM story_members "
            f"WHERE story_id IN ({placeholders})",
            list(story_ids),
        ).fetchall()
        member_ids = {int(row["raw_item_id"]) for row in member_rows}
        if not set(citation_ids).issubset(member_ids):
            raise Phase0ValidationError(
                "theme citations must reference member raw items"
            )

    @staticmethod
    def _prepare_theme(record: ThemeRecord) -> dict[str, Any]:
        if not isinstance(record, ThemeRecord):
            raise Phase0ValidationError("themes must be ThemeRecord instances")
        fingerprint = _require_text(record.fingerprint, "theme fingerprint")
        label = _require_text(record.label, "theme label")
        status = str(record.status or "pending")
        if status not in THEME_STATUSES:
            raise Phase0ValidationError("invalid theme status")
        method = _optional_text(record.method)
        if method is not None and method not in CLUSTERING_METHODS:
            raise Phase0ValidationError(f"unsupported clustering method: {method}")
        story_ids = [
            _require_int(value, "theme story id", minimum=1)
            for value in dict.fromkeys(record.story_ids)
        ]
        if not story_ids:
            raise Phase0ValidationError("a theme requires at least one story")
        citation_ids = [
            _require_int(value, "citation raw item id", minimum=1)
            for value in dict.fromkeys(record.citation_item_ids)
        ]
        return {
            "fingerprint": fingerprint,
            "theme_key": require_safe_identifier_scalar(
                _optional_text(record.theme_key) or fingerprint, "theme_key"
            ),
            "label": label,
            "label_source": validate_safe_identifier_scalar(
                _optional_text(record.label_source), "label_source"
            ),
            "summary": _optional_text(record.summary),
            "status": status,
            "story_ids": story_ids,
            "citation_ids": citation_ids,
            "salience": _optional_float(record.salience, "salience"),
            "salience_rank": _require_int(
                record.salience_rank, "salience_rank", minimum=1
            ),
            "cohesion": _optional_float(
                record.cohesion, "cohesion", low=-1.0, high=1.0
            ),
            "min_pairwise_cohesion": _optional_float(
                record.min_pairwise_cohesion,
                "min_pairwise_cohesion",
                low=-1.0,
                high=1.0,
            ),
            "story_count": (
                len(story_ids)
                if record.story_count is None
                else _require_int(record.story_count, "story_count", minimum=0)
            ),
            "outlet_count": (
                None
                if record.outlet_count is None
                else _require_int(record.outlet_count, "outlet_count", minimum=0)
            ),
            "latest_published_at": _normalize_datetime(
                record.latest_published_at, "latest_published_at", optional=True
            ),
            "salience_story_component": _optional_float(
                record.salience_story_component,
                "salience_story_component",
                low=0.0,
                high=1.0,
            ),
            "salience_outlet_component": _optional_float(
                record.salience_outlet_component,
                "salience_outlet_component",
                low=0.0,
                high=1.0,
            ),
            "salience_recency_component": _optional_float(
                record.salience_recency_component,
                "salience_recency_component",
                low=0.0,
                high=1.0,
            ),
            "centroid": _optional_blob(record.centroid, "centroid"),
            "matched_previous_key": validate_safe_identifier_scalar(
                _optional_text(record.matched_previous_key), "matched_previous_key"
            ),
            "method": method,
            "content_hash": _optional_text(record.content_hash) or fingerprint,
            "algorithm_version": validate_safe_identifier_scalar(
                _optional_text(record.algorithm_version), "algorithm_version"
            ),
            "config_fingerprint": validate_safe_identifier_scalar(
                _optional_text(record.config_fingerprint), "config_fingerprint"
            ),
            "model_name": validate_safe_identifier_scalar(
                _optional_text(record.model_name), "model_name"
            ),
            "model_revision": validate_safe_identifier_scalar(
                _optional_text(record.model_revision), "model_revision"
            ),
            "embedding_dimension": (
                None
                if record.embedding_dimension is None
                else _require_int(
                    record.embedding_dimension, "embedding_dimension", minimum=1
                )
            ),
        }

    @staticmethod
    def _prepare_theme_set(record: ThemeSetRecord) -> dict[str, Any]:
        if not isinstance(record, ThemeSetRecord):
            raise Phase0ValidationError("theme_set must be a ThemeSetRecord")
        method = _require_text(record.method, "theme set method")
        if method not in CLUSTERING_METHODS:
            raise Phase0ValidationError(f"unsupported clustering method: {method}")
        return {
            "method": method,
            "method_reason": sanitize_diagnostic_scalar(
                str(record.method_reason or ""), "method_reason"
            ),
            "quality": serialize_operational_metadata(
                dict(record.quality or {}), "quality", dict
            ),
            "source_metadata": (
                None
                if record.source_metadata is None
                else serialize_operational_metadata(
                    dict(record.source_metadata), "source_metadata", dict
                )
            ),
            "trust_metadata": serialize_operational_metadata(
                dict(record.trust_metadata or {}), "trust_metadata", dict
            ),
            "config_fingerprint": validate_safe_identifier_scalar(
                str(record.config_fingerprint or ""), "config_fingerprint"
            ),
            "algorithm_version": validate_safe_identifier_scalar(
                str(record.algorithm_version or ""), "algorithm_version"
            ),
            "model_name": validate_safe_identifier_scalar(
                _optional_text(record.model_name), "model_name"
            ),
            "model_revision": validate_safe_identifier_scalar(
                _optional_text(record.model_revision), "model_revision"
            ),
            "embedding_dimension": (
                None
                if record.embedding_dimension is None
                else _require_int(
                    record.embedding_dimension, "embedding_dimension", minimum=1
                )
            ),
        }

    def reconcile_themes(
        self,
        *,
        run: Any,
        ticker: str,
        trading_day: str | date,
        pipeline_version: str,
        theme_set: ThemeSetRecord,
        themes: Sequence[ThemeRecord] = (),
        other_coverage: Sequence[OtherCoverageRecord] = (),
        excluded: Sequence[ExcludedStoryRecord] = (),
        terminal: bool = False,
    ) -> ReconciliationReport:
        """Replace one ticker/trading-day's theme set atomically.

        Membership, evidence, "Other coverage" reasons, ranking, cohesion,
        and the run's algorithm/config/model fingerprints all land together
        or not at all — together with the ``run_log`` row for ``run``.
        Citation membership is enforced by the database as well as here, so
        a direct-SQL writer cannot produce a theme that cites a story it
        does not contain.

        Every story named anywhere in the payload — theme membership,
        Other Coverage, exclusions — must already sit in this ticker, day,
        and pipeline version, and a mismatch rejects the whole batch before
        anything is written.
        """

        normalized_ticker = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        with self._logged_mutation(
            run,
            operation="reconcile_themes",
            ticker=normalized_ticker,
            trading_day=day,
            pipeline_version=version,
            terminal=terminal,
        ) as (connection, context):
            self._assert_stories_in_partition(
                connection,
                themes=themes,
                other_coverage=other_coverage,
                excluded=excluded,
                ticker=normalized_ticker,
                trading_day=day,
                pipeline_version=version,
            )
            report = self._reconcile_themes_unlogged(
                ticker=normalized_ticker,
                trading_day=day,
                pipeline_version=version,
                theme_set=theme_set,
                themes=themes,
                other_coverage=other_coverage,
                excluded=excluded,
                connection=connection,
            )
            context._record_outcome(
                success=len(report.inserted) + len(report.updated),
                partial=len(report.unchanged),
            )
            context._merge_counts(report.counts)
            return report

    @staticmethod
    def _assert_stories_in_partition(
        connection: sqlite3.Connection,
        *,
        themes: Sequence[ThemeRecord],
        other_coverage: Sequence[OtherCoverageRecord],
        excluded: Sequence[ExcludedStoryRecord],
        ticker: str,
        trading_day: str,
        pipeline_version: str,
    ) -> None:
        story_ids: set[int] = set()
        for theme in themes:
            story_ids.update(int(value) for value in getattr(theme, "story_ids", ()))
        for row in list(other_coverage) + list(excluded):
            story_id = getattr(row, "story_id", None)
            if story_id is not None:
                story_ids.add(int(story_id))
        for story_id in sorted(story_ids):
            found = connection.execute(
                "SELECT ticker, trading_day, pipeline_version FROM stories "
                "WHERE id = ?",
                (story_id,),
            ).fetchone()
            if found is None:
                continue  # the foreign key reports this one precisely.
            actual = (
                str(found["ticker"]),
                str(found["trading_day"]),
                found["pipeline_version"],
            )
            if actual[0] != ticker or actual[1] != trading_day:
                raise Phase0RunContextError(
                    f"story {story_id} belongs to {actual[0]}/{actual[1]} but "
                    f"this theme set covers {ticker}/{trading_day}"
                )
            if actual[2] is not None and str(actual[2]) != pipeline_version:
                raise Phase0RunContextError(
                    f"story {story_id} belongs to pipeline version {actual[2]} "
                    f"but this theme set covers {pipeline_version}"
                )

    def _reconcile_themes_unlogged(
        self,
        *,
        ticker: str,
        trading_day: str | date,
        pipeline_version: str,
        theme_set: ThemeSetRecord,
        themes: Sequence[ThemeRecord] = (),
        other_coverage: Sequence[OtherCoverageRecord] = (),
        excluded: Sequence[ExcludedStoryRecord] = (),
        connection: sqlite3.Connection | None = None,
    ) -> ReconciliationReport:
        """Reconcile a theme set with no run attached; see :attr:`admin`."""

        normalized_ticker = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        set_values = self._prepare_theme_set(theme_set)
        prepared = [self._prepare_theme(record) for record in themes]
        fingerprints = [values["fingerprint"] for values in prepared]
        duplicates = {key for key in fingerprints if fingerprints.count(key) > 1}
        if duplicates:
            raise Phase0ValidationError(
                f"duplicate theme fingerprints: {sorted(duplicates)}"
            )
        keys = [values["theme_key"] for values in prepared]
        duplicate_keys = {key for key in keys if keys.count(key) > 1}
        if duplicate_keys:
            raise Phase0ValidationError(
                f"duplicate theme keys: {sorted(duplicate_keys)}"
            )
        coverage = self._prepare_coverage(other_coverage, excluded, prepared)
        incoming = dict(zip(fingerprints, prepared))

        with self._write_scope(connection) as connection:
            existing = {
                str(row["fingerprint"]): row
                for row in connection.execute(
                    """
                    SELECT * FROM themes
                    WHERE ticker = ? AND trading_day = ? AND pipeline_version = ?
                      AND fingerprint IS NOT NULL
                    """,
                    (normalized_ticker, day, version),
                )
            }
            obsolete = [
                int(row["id"])
                for fingerprint, row in existing.items()
                if fingerprint not in incoming
            ]
            # Obsolete themes go first: their citations would otherwise
            # block a raw item from moving to the theme that now owns it.
            self._delete_themes(connection, obsolete)

            theme_set_id = self._upsert_theme_set(
                connection, normalized_ticker, day, version, set_values
            )
            connection.execute(
                "DELETE FROM theme_other_coverage WHERE theme_set_id = ?",
                (theme_set_id,),
            )
            connection.execute(
                "DELETE FROM theme_excluded_stories WHERE theme_set_id = ?",
                (theme_set_id,),
            )

            inserted: list[int] = []
            updated: list[int] = []
            unchanged: list[int] = []
            for fingerprint, values in incoming.items():
                self._assert_story_membership(
                    connection,
                    normalized_ticker,
                    day,
                    values["story_ids"],
                    values["citation_ids"],
                )
                row = existing.get(fingerprint)
                if row is None:
                    inserted.append(
                        self._insert_reconciled_theme(
                            connection, normalized_ticker, day, version, values
                        )
                    )
                    continue
                theme_id = int(row["id"])
                if self._stored_theme_signature(connection, row) == (
                    self._theme_signature(values)
                ):
                    unchanged.append(theme_id)
                    continue
                self._update_reconciled_theme(connection, theme_id, values)
                updated.append(theme_id)

            for entry in coverage["other"]:
                connection.execute(
                    """
                    INSERT INTO theme_other_coverage (
                        theme_set_id, story_id, reason, position
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        theme_set_id,
                        entry["story_id"],
                        entry["reason"],
                        entry["position"],
                    ),
                )
            for entry in coverage["excluded"]:
                connection.execute(
                    """
                    INSERT INTO theme_excluded_stories (
                        theme_set_id, story_id, reason
                    ) VALUES (?, ?, ?)
                    """,
                    (theme_set_id, entry["story_id"], entry["reason"]),
                )

            return ReconciliationReport(
                inserted=tuple(inserted),
                updated=tuple(updated),
                unchanged=tuple(unchanged),
                deleted=tuple(obsolete),
            )

    @staticmethod
    def _prepare_coverage(
        other_coverage: Sequence[OtherCoverageRecord],
        excluded: Sequence[ExcludedStoryRecord],
        themes: Sequence[Mapping[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        member_story_ids = {
            story_id for values in themes for story_id in values["story_ids"]
        }
        other: list[dict[str, Any]] = []
        seen: set[int] = set()
        for position, entry in enumerate(other_coverage):
            if not isinstance(entry, OtherCoverageRecord):
                raise Phase0ValidationError(
                    "other coverage must be OtherCoverageRecord instances"
                )
            story_id = _require_int(entry.story_id, "story_id", minimum=1)
            reason = _require_text(entry.reason, "other-coverage reason")
            if reason not in OTHER_COVERAGE_REASONS:
                raise Phase0ValidationError(
                    f"unsupported other-coverage reason: {reason}"
                )
            if story_id in member_story_ids:
                raise Phase0ValidationError(
                    f"story {story_id} is in a theme and in other coverage"
                )
            if story_id in seen:
                raise Phase0ValidationError(
                    f"story {story_id} is listed twice under other coverage"
                )
            seen.add(story_id)
            other.append(
                {
                    "story_id": story_id,
                    "reason": reason,
                    "position": _require_int(
                        entry.position if entry.position else position,
                        "position",
                        minimum=0,
                    ),
                }
            )
        excluded_entries: list[dict[str, Any]] = []
        for entry in excluded:
            if not isinstance(entry, ExcludedStoryRecord):
                raise Phase0ValidationError(
                    "exclusions must be ExcludedStoryRecord instances"
                )
            story_id = _require_int(entry.story_id, "story_id", minimum=1)
            reason = _require_text(entry.reason, "exclusion reason")
            if reason not in EXCLUSION_REASONS:
                raise Phase0ValidationError(f"unsupported exclusion reason: {reason}")
            if story_id in member_story_ids or story_id in seen:
                raise Phase0ValidationError(
                    f"story {story_id} is already accounted for in this theme set"
                )
            seen.add(story_id)
            excluded_entries.append({"story_id": story_id, "reason": reason})
        return {"other": other, "excluded": excluded_entries}

    @staticmethod
    def _upsert_theme_set(
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        version: str,
        values: Mapping[str, Any],
    ) -> int:
        connection.execute(
            """
            INSERT INTO theme_sets (
                ticker, trading_day, pipeline_version, method, method_reason,
                quality, source_metadata, trust_metadata, config_fingerprint,
                algorithm_version, model_name, model_revision,
                embedding_dimension, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, trading_day, pipeline_version) DO UPDATE SET
                method = excluded.method,
                method_reason = excluded.method_reason,
                quality = excluded.quality,
                source_metadata = excluded.source_metadata,
                trust_metadata = excluded.trust_metadata,
                config_fingerprint = excluded.config_fingerprint,
                algorithm_version = excluded.algorithm_version,
                model_name = excluded.model_name,
                model_revision = excluded.model_revision,
                embedding_dimension = excluded.embedding_dimension,
                updated_at = excluded.updated_at
            """,
            (
                ticker,
                day,
                version,
                values["method"],
                values["method_reason"],
                values["quality"],
                values["source_metadata"],
                values["trust_metadata"],
                values["config_fingerprint"],
                values["algorithm_version"],
                values["model_name"],
                values["model_revision"],
                values["embedding_dimension"],
                utc_now(),
            ),
        )
        row = connection.execute(
            "SELECT id FROM theme_sets WHERE ticker = ? AND trading_day = ? "
            "AND pipeline_version = ?",
            (ticker, day, version),
        ).fetchone()
        return int(row["id"])

    @staticmethod
    def _theme_signature(values: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            values["theme_key"],
            values["label"],
            values["label_source"],
            values["summary"],
            values["status"],
            values["salience"],
            values["salience_rank"],
            values["cohesion"],
            values["min_pairwise_cohesion"],
            values["story_count"],
            values["outlet_count"],
            values["latest_published_at"],
            values["matched_previous_key"],
            values["method"],
            values["content_hash"],
            values["algorithm_version"],
            values["config_fingerprint"],
            values["model_name"],
            values["model_revision"],
            values["embedding_dimension"],
            tuple(sorted(values["story_ids"])),
            tuple(sorted(values["citation_ids"])),
        )

    @staticmethod
    def _stored_theme_signature(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> tuple[Any, ...]:
        theme_id = int(row["id"])
        story_ids = tuple(
            sorted(
                int(item["story_id"])
                for item in connection.execute(
                    "SELECT story_id FROM theme_stories WHERE theme_id = ?",
                    (theme_id,),
                )
            )
        )
        citation_ids = tuple(
            sorted(
                int(item["raw_item_id"])
                for item in connection.execute(
                    "SELECT raw_item_id FROM theme_citations WHERE theme_id = ?",
                    (theme_id,),
                )
            )
        )
        return (
            row["theme_key"],
            row["label"],
            row["label_source"],
            row["summary"],
            row["status"],
            row["salience"],
            int(row["salience_rank"]),
            row["cohesion"],
            row["min_pairwise_cohesion"],
            row["story_count"],
            row["outlet_count"],
            row["latest_published_at"],
            row["matched_previous_key"],
            row["method"],
            row["content_hash"],
            row["algorithm_version"],
            row["config_fingerprint"],
            row["model_name"],
            row["model_revision"],
            row["embedding_dimension"],
            story_ids,
            citation_ids,
        )

    def _insert_reconciled_theme(
        self,
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        version: str,
        values: Mapping[str, Any],
    ) -> int:
        cursor = connection.execute(
            """
            INSERT INTO themes (
                ticker, trading_day, label, summary, citations, salience_rank,
                status, centroid, content_hash, pipeline_version, theme_key,
                fingerprint, label_source, method, salience, cohesion,
                min_pairwise_cohesion, story_count, outlet_count,
                latest_published_at, salience_story_component,
                salience_outlet_component, salience_recency_component,
                matched_previous_key, algorithm_version, config_fingerprint,
                model_name, model_revision, embedding_dimension, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                ticker,
                day,
                values["label"],
                values["summary"],
                _serialize_json(values["citation_ids"], "citations", list),
                values["salience_rank"],
                values["status"],
                values["centroid"],
                values["content_hash"],
                version,
                values["theme_key"],
                values["fingerprint"],
                values["label_source"],
                values["method"],
                values["salience"],
                values["cohesion"],
                values["min_pairwise_cohesion"],
                values["story_count"],
                values["outlet_count"],
                values["latest_published_at"],
                values["salience_story_component"],
                values["salience_outlet_component"],
                values["salience_recency_component"],
                values["matched_previous_key"],
                values["algorithm_version"],
                values["config_fingerprint"],
                values["model_name"],
                values["model_revision"],
                values["embedding_dimension"],
                utc_now(),
            ),
        )
        theme_id = int(cursor.lastrowid)
        self._write_theme_membership(connection, theme_id, values)
        return theme_id

    def _update_reconciled_theme(
        self,
        connection: sqlite3.Connection,
        theme_id: int,
        values: Mapping[str, Any],
    ) -> None:
        connection.execute(
            "DELETE FROM theme_citations WHERE theme_id = ?", (theme_id,)
        )
        connection.execute("DELETE FROM theme_stories WHERE theme_id = ?", (theme_id,))
        connection.execute(
            """
            UPDATE themes SET
                label = ?, summary = ?, citations = ?, salience_rank = ?,
                status = ?, centroid = ?, content_hash = ?, theme_key = ?,
                label_source = ?, method = ?, salience = ?, cohesion = ?,
                min_pairwise_cohesion = ?, story_count = ?, outlet_count = ?,
                latest_published_at = ?, salience_story_component = ?,
                salience_outlet_component = ?, salience_recency_component = ?,
                matched_previous_key = ?, algorithm_version = ?,
                config_fingerprint = ?, model_name = ?, model_revision = ?,
                embedding_dimension = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                values["label"],
                values["summary"],
                _serialize_json(values["citation_ids"], "citations", list),
                values["salience_rank"],
                values["status"],
                values["centroid"],
                values["content_hash"],
                values["theme_key"],
                values["label_source"],
                values["method"],
                values["salience"],
                values["cohesion"],
                values["min_pairwise_cohesion"],
                values["story_count"],
                values["outlet_count"],
                values["latest_published_at"],
                values["salience_story_component"],
                values["salience_outlet_component"],
                values["salience_recency_component"],
                values["matched_previous_key"],
                values["algorithm_version"],
                values["config_fingerprint"],
                values["model_name"],
                values["model_revision"],
                values["embedding_dimension"],
                utc_now(),
                theme_id,
            ),
        )
        self._write_theme_membership(connection, theme_id, values)

    @staticmethod
    def _write_theme_membership(
        connection: sqlite3.Connection,
        theme_id: int,
        values: Mapping[str, Any],
    ) -> None:
        connection.executemany(
            "INSERT INTO theme_stories (theme_id, story_id) VALUES (?, ?)",
            [(theme_id, story_id) for story_id in values["story_ids"]],
        )
        connection.executemany(
            "INSERT INTO theme_citations (theme_id, raw_item_id) VALUES (?, ?)",
            [(theme_id, item_id) for item_id in values["citation_ids"]],
        )

    def theme_set(
        self, *, ticker: str, trading_day: str | date, pipeline_version: str
    ) -> dict[str, Any] | None:
        """The stored theme set for one ticker-day, with its accounting."""

        normalized_ticker = normalize_ticker(ticker)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM theme_sets WHERE ticker = ? AND trading_day = ? "
                "AND pipeline_version = ?",
                (normalized_ticker, day, version),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["quality"] = json.loads(result["quality"])
            result["trust_metadata"] = json.loads(result["trust_metadata"])
            result["source_metadata"] = (
                None
                if result["source_metadata"] is None
                else json.loads(result["source_metadata"])
            )
            result["themes"] = [
                dict(theme)
                for theme in connection.execute(
                    "SELECT * FROM themes WHERE ticker = ? AND trading_day = ? "
                    "AND pipeline_version = ? ORDER BY salience_rank, id",
                    (normalized_ticker, day, version),
                )
            ]
            result["other_coverage"] = [
                dict(entry)
                for entry in connection.execute(
                    "SELECT story_id, reason, position FROM theme_other_coverage "
                    "WHERE theme_set_id = ? ORDER BY position, story_id",
                    (int(row["id"]),),
                )
            ]
            result["excluded"] = [
                dict(entry)
                for entry in connection.execute(
                    "SELECT story_id, reason FROM theme_excluded_stories "
                    "WHERE theme_set_id = ? ORDER BY story_id",
                    (int(row["id"]),),
                )
            ]
            return result

    # ------------------------------------------------------------------
    # Evaluation labels
    # ------------------------------------------------------------------

    def _insert_eval_label_unlogged(
        self,
        *,
        label_type: str,
        item_a_id: int,
        item_b_id: int,
        reviewer: str,
        label: str,
        notes: str | None = None,
        created_at: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO eval_labels (
                    label_type, item_a_id, item_b_id, reviewer,
                    label, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _require_text(label_type, "label_type"),
                    _require_int(item_a_id, "item_a_id", minimum=1),
                    _require_int(item_b_id, "item_b_id", minimum=1),
                    _require_text(reviewer, "reviewer"),
                    _require_text(label, "label"),
                    notes,
                    _normalize_datetime(created_at or utc_now(), "created_at"),
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    # Stage keys, leases, and recovery
    # ------------------------------------------------------------------

    def claim_stage_key(
        self,
        *,
        stage: str,
        ticker: str,
        trading_day: str,
        pipeline_version: str,
        run_id: str,
        lease_seconds: int = 300,
        claimed_at: str | datetime | None = None,
    ) -> bool:
        """Atomically claim an unowned, retryable, or expired stage key."""
        if _require_int(lease_seconds, "lease_seconds") <= 0:
            raise Phase0ValidationError("lease_seconds must be positive")
        day = _normalize_day(trading_day)
        normalized_ticker = normalize_ticker(ticker)
        normalized_claimed_at = _normalize_datetime(
            claimed_at or utc_now(),
            "claimed_at",
        )
        claimed_datetime = datetime.fromisoformat(normalized_claimed_at)
        lease_expires_at = (
            claimed_datetime + timedelta(seconds=int(lease_seconds))
        ).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pipeline_stage_keys (
                    stage, ticker, trading_day, pipeline_version,
                    status, run_id, updated_at, lease_expires_at,
                    attempts, claimed_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, 1, ?)
                ON CONFLICT(stage, ticker, trading_day, pipeline_version)
                DO UPDATE SET
                    status = 'running',
                    run_id = excluded.run_id,
                    updated_at = excluded.updated_at,
                    lease_expires_at = excluded.lease_expires_at,
                    claimed_at = excluded.claimed_at,
                    completed_at = NULL,
                    attempts = pipeline_stage_keys.attempts + 1,
                    recovered_count = pipeline_stage_keys.recovered_count + CASE
                        WHEN pipeline_stage_keys.status = 'running' THEN 1
                        ELSE 0
                    END
                WHERE pipeline_stage_keys.status IN ('failed', 'degraded')
                   OR (
                        pipeline_stage_keys.status = 'running'
                        AND (
                            pipeline_stage_keys.lease_expires_at IS NULL
                            OR pipeline_stage_keys.lease_expires_at
                                <= excluded.updated_at
                        )
                   )
                """,
                (
                    stage,
                    normalized_ticker,
                    day,
                    pipeline_version,
                    run_id,
                    normalized_claimed_at,
                    lease_expires_at,
                    normalized_claimed_at,
                ),
            )
            return cursor.rowcount == 1

    def heartbeat_stage_key(
        self,
        *,
        stage: str,
        ticker: str,
        trading_day: str,
        pipeline_version: str,
        run_id: str,
        lease_seconds: int = 300,
        now: str | datetime | None = None,
    ) -> bool:
        """Extend the caller's own lease; ``False`` when it no longer owns it."""

        if _require_int(lease_seconds, "lease_seconds") <= 0:
            raise Phase0ValidationError("lease_seconds must be positive")
        day = _normalize_day(trading_day)
        normalized_ticker = normalize_ticker(ticker)
        moment = _normalize_datetime(now or utc_now(), "now")
        expires = (
            datetime.fromisoformat(moment) + timedelta(seconds=int(lease_seconds))
        ).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_stage_keys
                SET updated_at = ?, lease_expires_at = ?
                WHERE stage = ? AND ticker = ? AND trading_day = ?
                  AND pipeline_version = ? AND run_id = ? AND status = 'running'
                """,
                (
                    moment,
                    expires,
                    stage,
                    normalized_ticker,
                    day,
                    pipeline_version,
                    run_id,
                ),
            )
            return cursor.rowcount == 1

    def complete_stage_key(
        self,
        *,
        stage: str,
        ticker: str,
        trading_day: str,
        pipeline_version: str,
        run_id: str,
        status: str = "success",
        error: Any = None,
    ) -> None:
        if status not in STAGE_KEY_STATUSES:
            raise Phase0ValidationError("invalid stage-key status")
        day = _normalize_day(trading_day)
        normalized_ticker = normalize_ticker(ticker)
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_stage_keys
                SET status = ?, updated_at = ?, lease_expires_at = NULL,
                    completed_at = ?, last_error = ?
                WHERE stage = ? AND ticker = ? AND trading_day = ?
                    AND pipeline_version = ? AND run_id = ? AND status = 'running'
                """,
                (
                    status,
                    now,
                    now,
                    None if error is None else redact_text(str(error)),
                    stage,
                    normalized_ticker,
                    day,
                    pipeline_version,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise StageKeyError("stage key was not claimed by this run")

    def recover_expired_leases(
        self,
        *,
        now: str | datetime | None = None,
        stage: str | None = None,
        trading_day: str | date | None = None,
    ) -> list[dict[str, Any]]:
        """Mark abandoned running claims as retryable, and report them.

        A crashed owner leaves a ``running`` key behind.  Recovery is
        deterministic rather than manual: once the lease has expired the key
        becomes claimable again, and this method makes that explicit for an
        operator or the runner's start-up sweep.
        """

        moment = _normalize_datetime(now or utc_now(), "now")
        clauses = [
            "status = ?",
            "lease_expires_at IS NOT NULL",
            "lease_expires_at <= ?",
        ]
        parameters: list[Any] = [RUNNING_STATUS, moment]
        if stage is not None:
            clauses.append("stage = ?")
            parameters.append(_require_text(stage, "stage"))
        if trading_day is not None:
            clauses.append("trading_day = ?")
            parameters.append(_normalize_day(trading_day))
        where = " AND ".join(clauses)
        with self._connect(immediate=True) as connection:
            expired = [
                dict(row)
                for row in connection.execute(
                    f"SELECT * FROM pipeline_stage_keys WHERE {where} "
                    "ORDER BY stage, ticker, trading_day, pipeline_version",
                    parameters,
                )
            ]
            if expired:
                connection.execute(
                    f"UPDATE pipeline_stage_keys SET status = 'failed', "
                    f"updated_at = ?, lease_expires_at = NULL, "
                    f"recovered_count = recovered_count + 1, "
                    f"last_error = 'lease expired; claim abandoned' "
                    f"WHERE {where}",
                    [moment, *parameters],
                )
            return expired

    def stage_key_state(
        self,
        *,
        stage: str,
        ticker: str,
        trading_day: str | date,
        pipeline_version: str,
    ) -> dict[str, Any] | None:
        """Lifecycle and retry state for one stage key."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pipeline_stage_keys
                WHERE stage = ? AND ticker = ? AND trading_day = ?
                  AND pipeline_version = ?
                """,
                (
                    _require_text(stage, "stage"),
                    normalize_ticker(ticker),
                    _normalize_day(trading_day),
                    _require_text(pipeline_version, "pipeline_version"),
                ),
            ).fetchone()
        return None if row is None else dict(row)

    def stage_keys_for_day(self, trading_day: str | date) -> list[dict[str, Any]]:
        day = _normalize_day(trading_day)
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM pipeline_stage_keys WHERE trading_day = ? "
                    "ORDER BY stage, ticker, pipeline_version",
                    (day,),
                )
            ]

    def _clear_derived_for_day_unlogged(self, trading_day: str | date) -> None:
        """Delete derived rows and their idempotency keys, retaining raw input."""
        day = _normalize_day(trading_day)
        with self._connect(immediate=True) as connection:
            theme_ids = [
                int(row["id"])
                for row in connection.execute(
                    "SELECT id FROM themes WHERE trading_day = ? ORDER BY id", (day,)
                )
            ]
            self._delete_themes(connection, theme_ids)
            connection.execute("DELETE FROM theme_sets WHERE trading_day = ?", (day,))
            connection.execute(
                "UPDATE stories SET canonical_item_id = NULL WHERE trading_day = ?",
                (day,),
            )
            connection.execute(
                "DELETE FROM story_members WHERE story_id IN "
                "(SELECT id FROM stories WHERE trading_day = ?)",
                (day,),
            )
            connection.execute("DELETE FROM stories WHERE trading_day = ?", (day,))
            connection.execute(
                "DELETE FROM pipeline_stage_keys WHERE trading_day = ?", (day,)
            )

    # ------------------------------------------------------------------
    # Run log
    # ------------------------------------------------------------------

    def _log_stage_unlogged(
        self,
        *,
        run_id: str,
        stage: str,
        counts: Mapping[str, Any],
        duration_ms: int,
        errors: Sequence[Mapping[str, Any] | str],
        started_at: str,
        completed_at: str,
        trading_day: str,
        pipeline_version: str,
        status: str | None = None,
        ticker: str | None = None,
        success_count: int = 0,
        partial_count: int = 0,
        failure_count: int = 0,
        attempt: int = 1,
        replay: bool = False,
        stage_key: Mapping[str, Any] | None = None,
    ) -> int:
        """Record one stage of one run.  There is no way to skip this."""

        with self._connect() as connection:
            return self._write_run_log(
                connection,
                run_id=run_id,
                stage=stage,
                counts=counts,
                duration_ms=duration_ms,
                errors=errors,
                started_at=started_at,
                completed_at=completed_at,
                trading_day=trading_day,
                pipeline_version=pipeline_version,
                status=status,
                ticker=ticker,
                success_count=success_count,
                partial_count=partial_count,
                failure_count=failure_count,
                attempt=attempt,
                replay=replay,
                stage_key=stage_key,
            )

    def _write_run_log(
        self,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        stage: str,
        counts: Mapping[str, Any],
        duration_ms: int,
        errors: Sequence[Mapping[str, Any] | str],
        started_at: str,
        completed_at: str,
        trading_day: str,
        pipeline_version: str,
        status: str | None = None,
        ticker: str | None = None,
        success_count: int = 0,
        partial_count: int = 0,
        failure_count: int = 0,
        attempt: int = 1,
        replay: bool = False,
        stage_key: Mapping[str, Any] | None = None,
    ) -> int:
        """Write the run-log row on an existing transaction.

        Logged pipeline mutations call this *inside* their own write
        transaction so the row and the data it describes commit together.
        """

        resolved_status = status or ("degraded" if errors else "success")
        if resolved_status not in RUN_STATUSES:
            raise Phase0ValidationError("invalid run status")
        if _require_int(duration_ms, "duration_ms") < 0:
            raise Phase0ValidationError("duration_ms cannot be negative")
        normalized_started = _normalize_datetime(started_at, "started_at")
        normalized_completed = _normalize_datetime(completed_at, "completed_at")
        if normalized_completed < normalized_started:
            raise Phase0ValidationError("completed_at cannot precede started_at")
        normalized_ticker = normalize_ticker(ticker, optional=True)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        connection.execute(
            """
            INSERT INTO run_log (
                run_id, stage, counts, duration_ms, errors, started_at,
                completed_at, status, trading_day, pipeline_version,
                ticker, success_count, partial_count, failure_count,
                attempt, replay
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, stage) DO UPDATE SET
                counts = excluded.counts,
                duration_ms = excluded.duration_ms,
                errors = excluded.errors,
                completed_at = excluded.completed_at,
                status = excluded.status,
                ticker = excluded.ticker,
                success_count = excluded.success_count,
                partial_count = excluded.partial_count,
                failure_count = excluded.failure_count,
                attempt = excluded.attempt,
                replay = excluded.replay
            """,
            (
                _require_text(run_id, "run_id"),
                _require_text(stage, "stage"),
                _serialize_json(dict(counts), "counts", dict),
                int(duration_ms),
                _serialize_json(list(errors), "errors", list),
                normalized_started,
                normalized_completed,
                resolved_status,
                day,
                version,
                normalized_ticker,
                _require_int(success_count, "success_count", minimum=0),
                _require_int(partial_count, "partial_count", minimum=0),
                _require_int(failure_count, "failure_count", minimum=0),
                _require_int(attempt, "attempt", minimum=1),
                1 if replay else 0,
            ),
        )
        row = connection.execute(
            "SELECT id FROM run_log WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        ).fetchone()
        run_log_id = int(row["id"])
        if stage_key is not None:
            self._link_stage_key(connection, run_log_id, stage_key)
        return run_log_id

    @staticmethod
    def _link_stage_key(
        connection: sqlite3.Connection,
        run_log_id: int,
        stage_key: Mapping[str, Any],
    ) -> None:
        try:
            connection.execute(
                """
                INSERT OR IGNORE INTO run_log_stage_keys (
                    run_log_id, stage, ticker, trading_day, pipeline_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_log_id,
                    _require_text(stage_key.get("stage"), "stage_key stage"),
                    normalize_ticker(stage_key.get("ticker")),
                    _normalize_day(stage_key.get("trading_day")),
                    _require_text(
                        stage_key.get("pipeline_version"),
                        "stage_key pipeline_version",
                    ),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise Phase0IntegrityError(
                "run log references a stage key that does not exist"
            ) from exc

    @contextmanager
    def stage_run(
        self,
        *,
        run_id: str,
        stage: str,
        trading_day: str | date,
        pipeline_version: str,
        ticker: str | None = None,
        attempt: int = 1,
        replay: bool = False,
        stage_key: Mapping[str, Any] | None = None,
    ) -> Iterator[StageRunContext]:
        """Run one stage and record it, whatever happens inside.

        Mandatory by construction: the ``run_log`` row is written in a
        ``finally`` block, so an exception in the stage body produces a
        ``failed`` row and then propagates.  There is deliberately no
        ``persist_run_log=False``-style flag anywhere in this class.

        When a ``stage_key`` is supplied, ownership is proved *before* the
        context exists: one authoritative transaction loads the key and
        checks its partition, its owner, its status, and its lease.  A
        missing, foreign, completed, reclaimed, expired, or mismatched key
        means no context is created, nothing is registered, and no run log
        is written — an impostor cannot open a run and record an empty
        success.

        **Terminal operations.**  A stage that holds a key declares itself
        done by passing ``terminal=True`` to its last logged mutation; that
        one transaction commits the data, the final run log, and the key's
        completion together.  This block does not separately mark success
        afterwards, because that gap is exactly where another owner could
        reclaim.  A stage that holds a key and never declares a terminal
        operation ends ``degraded`` with the key left retryable — it never
        said the work was finished, so nothing here will say it for it.
        """

        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        normalized_ticker = normalize_ticker(ticker, optional=True)
        normalized_stage = _require_text(stage, "stage")
        normalized_run_id = _require_text(run_id, "run_id")
        key = self._normalize_stage_key(stage_key) if stage_key is not None else None
        if key is not None:
            # The stage key and the run must describe the same work, or the
            # lease being held is a lease on something else entirely.
            self._assert_stage_key_matches(
                key,
                operation="stage_run",
                stage=normalized_stage,
                ticker=normalized_ticker,
                trading_day=day,
                pipeline_version=version,
            )
            # Ownership, status, and lease, proved before anything exists.
            with self._connect(immediate=True) as connection:
                self._assert_stage_key_owned(
                    connection, key, normalized_run_id, operation="stage_run"
                )

        context = StageRunContext(
            _CONTEXT_KEY,
            repository=self,
            run_id=normalized_run_id,
            stage=normalized_stage,
            trading_day=day,
            pipeline_version=version,
            ticker=normalized_ticker,
            attempt=_require_int(attempt, "attempt", minimum=1),
            replay=bool(replay),
            stage_key=key,
        )
        self._register_run(context)
        started = datetime.now(timezone.utc)
        object.__setattr__(context, "_started_at", started)
        failure: BaseException | None = None
        try:
            yield context
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            failure = exc
            context._record_outcome(failure=1)
            context._record_error({"error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            self._unregister_run(context)
            if failure is not None:
                status = "failed"
            elif key is not None and not context.terminated:
                # The stage held a lease and never declared completion.
                # Retryable, not successful: saying otherwise here would be
                # the separate success-marking this design removed.
                status = "degraded"
            else:
                status = context._resolved_status()

            if context.terminated and failure is None:
                # The terminal mutation already committed the final run log
                # and released the key, atomically.  Nothing to add.
                pass
            else:
                with self._connect(immediate=True) as connection:
                    self._write_final_run_log(connection, context, status)
                    if key is not None:
                        self._finish_stage_key(
                            connection,
                            context,
                            "failed" if status != "success" else "success",
                        )

    def _write_final_run_log(
        self,
        connection: sqlite3.Connection,
        context: StageRunContext,
        status: str,
    ) -> None:
        completed = datetime.now(timezone.utc)
        self._write_run_log(
            connection,
            run_id=context.run_id,
            stage=context.stage,
            counts=context.counts,
            duration_ms=int((completed - context.started_at).total_seconds() * 1000),
            errors=context.errors,
            started_at=context.started_at.isoformat(),
            completed_at=completed.isoformat(),
            trading_day=context.trading_day,
            pipeline_version=context.pipeline_version,
            status=status,
            ticker=context.ticker,
            success_count=context.success_count,
            partial_count=context.partial_count,
            failure_count=context.failure_count,
            attempt=context.attempt,
            replay=context.replay,
            stage_key=context._stage_key,
        )

    @staticmethod
    def _assert_stage_key_owned(
        connection: sqlite3.Connection,
        key: Mapping[str, Any],
        run_id: str,
        *,
        operation: str,
    ) -> None:
        """Load the key on this transaction and prove this run owns it."""

        row = connection.execute(
            """
            SELECT run_id, status, lease_expires_at
            FROM pipeline_stage_keys
            WHERE stage = ? AND ticker = ? AND trading_day = ?
              AND pipeline_version = ?
            """,
            (
                key["stage"],
                key["ticker"],
                key["trading_day"],
                key["pipeline_version"],
            ),
        ).fetchone()
        if row is None:
            raise StageKeyError(
                f"{operation}: no stage key has been claimed for this partition"
            )
        if str(row["run_id"]) != run_id:
            raise StageKeyError(
                f"{operation}: the stage key is owned by run "
                f"{row['run_id']!r}, not {run_id!r}"
            )
        if str(row["status"]) != "running":
            raise StageKeyError(
                f"{operation}: the stage key is no longer running "
                f"(status={row['status']!r})"
            )
        expires = row["lease_expires_at"]
        if expires is not None and str(expires) <= utc_now():
            raise StageKeyError(f"{operation}: the stage lease has expired")

    # -- The run registry: identity is the capability -------------------

    def _register_run(self, context: StageRunContext) -> None:
        with self._run_lock:
            # Held strongly, so no other object can reuse this id() while
            # the run is live.
            self._active_runs[id(context)] = context

    def _unregister_run(self, context: StageRunContext) -> None:
        with self._run_lock:
            self._active_runs.pop(id(context), None)

    def _is_active_run(self, context: StageRunContext) -> bool:
        with self._run_lock:
            return self._active_runs.get(id(context)) is context

    def _finish_stage_key(
        self,
        connection: sqlite3.Connection,
        context: StageRunContext,
        status: str,
    ) -> None:
        """Release the run's stage key on the run's own transaction.

        Updating zero rows is an error, never a silent success: it means
        the key was reclaimed, completed, or never owned by this run, and
        writing a success anywhere on that basis would let two owners
        disagree about the same work.
        """

        key = context._stage_key
        if key is None:
            return
        terminal = "success" if status == "success" else status
        if terminal not in STAGE_KEY_STATUSES:
            terminal = "failed"
        error = None
        if context.errors:
            error = redact_text(str(context.errors[-1]))
        now = utc_now()
        cursor = connection.execute(
            """
            UPDATE pipeline_stage_keys
            SET status = ?, updated_at = ?, lease_expires_at = NULL,
                completed_at = ?, last_error = ?
            WHERE stage = ? AND ticker = ? AND trading_day = ?
              AND pipeline_version = ? AND run_id = ? AND status = 'running'
            """,
            (
                terminal,
                now,
                now,
                error,
                key["stage"],
                key["ticker"],
                key["trading_day"],
                key["pipeline_version"],
                context.run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise StageKeyError(
                "cannot finish a stage key this run no longer owns; it was "
                "reclaimed, already completed, or never claimed by "
                f"{context.run_id!r}"
            )

    # ------------------------------------------------------------------
    # The logged pipeline contract (issue #68)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_stage_key(stage_key: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(stage_key, Mapping):
            raise Phase0ValidationError("stage_key must be a mapping")
        return {
            "stage": _require_text(stage_key.get("stage"), "stage_key stage"),
            "ticker": normalize_ticker(stage_key.get("ticker")),
            "trading_day": _normalize_day(stage_key.get("trading_day")),
            "pipeline_version": _require_text(
                stage_key.get("pipeline_version"), "stage_key pipeline_version"
            ),
        }

    @staticmethod
    def _assert_stage_key_matches(
        key: Mapping[str, Any],
        *,
        operation: str,
        stage: str,
        ticker: str | None,
        trading_day: str,
        pipeline_version: str,
    ) -> None:
        """A lease on one partition never authorizes work on another.

        Checked field by field so the error names the one that is wrong —
        an AMD stage key silently authorizing an NVDA run is exactly the
        class of bug this exists to make impossible.
        """

        expected = {
            "stage": stage,
            "ticker": ticker,
            "trading_day": trading_day,
            "pipeline_version": pipeline_version,
        }
        for field, want in expected.items():
            if want is None:
                continue
            got = key.get(field)
            if got != want:
                raise Phase0RunContextError(
                    f"{operation}: the stage key covers {field}={got!r} but the "
                    f"run covers {field}={want!r}"
                )

    def _authorize_run(
        self,
        connection: sqlite3.Connection,
        run: Any,
        *,
        operation: str,
        ticker: str | None = None,
        trading_day: str | None = None,
        pipeline_version: str | None = None,
    ) -> StageRunContext:
        """Authorize a mutation **on the transaction that will perform it**.

        Everything here reads through ``connection``, which already holds
        the write lock.  That is the whole point: checking the lease on a
        second connection first would leave a window in which it expires
        and is reclaimed while this transaction still commits.  Holding the
        write lock across validation *and* mutation means a concurrent
        reclaimer cannot get in between them — it blocks until this
        transaction ends, and then sees the finished state.
        """

        if not isinstance(run, StageRunContext):
            raise Phase0RunContextError(
                f"{operation} requires an active stage run; open one with "
                "Phase0Repository.stage_run(...) and pass run=<context>"
            )
        # Identity, not field equality: a copy, a pickle, an
        # ``object.__new__`` shell, or a look-alike with every attribute
        # matching is a different object and authorizes nothing.
        if not self._is_active_run(run):
            if getattr(run, "_repository", None) is not self:
                raise Phase0RunContextError(
                    f"{operation} was given a stage run from another repository"
                )
            raise Phase0RunContextError(
                f"{operation} was given a stage run that is no longer active; "
                "a run authorizes writes only inside its stage_run block"
            )
        if trading_day is not None and run.trading_day != trading_day:
            raise Phase0RunContextError(
                f"{operation} targets {trading_day} but the run covers "
                f"{run.trading_day}"
            )
        if pipeline_version is not None and run.pipeline_version != pipeline_version:
            raise Phase0RunContextError(
                f"{operation} targets pipeline version {pipeline_version} but the "
                f"run covers {run.pipeline_version}"
            )
        if ticker is not None and run.ticker is not None and run.ticker != ticker:
            raise Phase0RunContextError(
                f"{operation} targets {ticker} but the run covers {run.ticker}"
            )
        self._assert_lease_held(connection, run, operation=operation)
        return run

    def _assert_lease_held(
        self,
        connection: sqlite3.Connection,
        run: StageRunContext,
        *,
        operation: str,
    ) -> None:
        """Ownership, expiry, and partition, read inside the write lock."""

        key = run._stage_key
        if key is None:
            return
        self._assert_stage_key_matches(
            key,
            operation=operation,
            stage=run.stage,
            ticker=run.ticker,
            trading_day=run.trading_day,
            pipeline_version=run.pipeline_version,
        )
        row = connection.execute(
            """
            SELECT run_id, status, lease_expires_at
            FROM pipeline_stage_keys
            WHERE stage = ? AND ticker = ? AND trading_day = ?
              AND pipeline_version = ?
            """,
            (
                key["stage"],
                key["ticker"],
                key["trading_day"],
                key["pipeline_version"],
            ),
        ).fetchone()
        if row is None:
            raise StageKeyError(f"{operation}: the run's stage key no longer exists")
        if str(row["run_id"]) != run.run_id:
            raise StageKeyError(f"{operation}: the stage key is owned by another run")
        if str(row["status"]) != "running":
            raise StageKeyError(
                f"{operation}: the stage key is no longer running "
                f"(status={row['status']!r})"
            )
        expires = row["lease_expires_at"]
        if expires is not None and str(expires) <= utc_now():
            raise StageKeyError(f"{operation}: the stage lease has expired")

    @contextmanager
    def _logged_mutation(
        self,
        run: Any,
        *,
        operation: str,
        ticker: str | None = None,
        trading_day: str | None = None,
        pipeline_version: str | None = None,
        terminal: bool = False,
    ) -> Iterator[tuple[sqlite3.Connection, StageRunContext]]:
        """One transaction holding authorization, mutation, log, and release.

        The order matters and is the fix for both the TOCTOU hole and the
        completion gap that followed it:

        1. take the write lock (``BEGIN IMMEDIATE``);
        2. authorize the run and its lease *on that connection*;
        3. mutate;
        4. write the derived counts and the run-log row;
        5. when ``terminal``, transition the stage key to its final status
           and release the lease — still inside this transaction;
        6. commit, releasing the lock.

        A concurrent reclaimer blocks at step 1 and cannot get in anywhere
        between steps 2 and 6.  Because step 5 is inside, there is no
        committed state in which the data says success and the stage key is
        still sitting there reclaimable as ``running``.

        A non-terminal operation never records ``success``: its interim
        run-log row is written as ``degraded`` until some operation
        declares the stage finished.

        On failure the data rolls back, and the failed run log *and* the
        key's retryable transition are written together in a second
        authoritative transaction.
        """

        try:
            with self._connect(immediate=True) as connection:
                context = self._authorize_run(
                    connection,
                    run,
                    operation=operation,
                    ticker=ticker,
                    trading_day=trading_day,
                    pipeline_version=pipeline_version,
                )
                yield connection, context
                if terminal:
                    status = context._resolved_status()
                    self._write_final_run_log(connection, context, status)
                    if context._stage_key is not None:
                        self._finish_stage_key(connection, context, status)
                    context._mark_terminated()
                else:
                    # Never "success" before the stage says it is finished.
                    self._write_final_run_log(connection, context, "degraded")
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            # Only a run that was genuinely authorized gets a failed record;
            # a rejected forgery has no run to write against.
            if isinstance(run, StageRunContext) and self._is_active_run(run):
                run._record_outcome(failure=1)
                run._record_error(
                    {"operation": operation, "error": f"{type(exc).__name__}: {exc}"}
                )
                try:
                    with self._connect(immediate=True) as connection:
                        self._write_final_run_log(connection, run, "failed")
                        if run._stage_key is not None:
                            self._finish_stage_key(connection, run, "failed")
                            run._mark_terminated()
                except StageKeyError:
                    # The key was already lost — which is usually *why* this
                    # failed. Recording that is best-effort; the original
                    # exception is the one the caller needs to see, so it
                    # must not be masked by the bookkeeping.
                    with self._connect(immediate=True) as connection:
                        self._write_final_run_log(connection, run, "failed")
                    run._mark_terminated()
            raise

    def ingest_raw_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        run: Any,
        source_state: Mapping[str, Any] | None = None,
        terminal: bool = False,
    ) -> list[InsertResult]:
        """Persist an ingestion batch and its run log in one transaction.

        **Partition rule for raw evidence (#82, #83 must follow this).**
        A raw item may carry ``ticker=None``, which the spec defines as
        "matches no ticker"; that is evidence the run is allowed to store,
        because a fetcher legitimately keeps items it could not attribute.
        Any ticker it *does* assert — in ``ticker``, in ``tickers``, or in
        ``candidate_tickers`` — must equal the run's ticker.  A batch
        mixing tickers is rejected whole, before anything is written: it
        means the caller sliced its work wrong, and a partial write would
        leave a ticker-day nobody can replay.

        The item's trading day, derived as the spec does from
        ``published_at`` falling back to ``fetched_at``, must equal the
        run's trading day.
        """

        prepared = [self._prepare_raw_item(item) for item in items]
        if self._is_active_run(run) if isinstance(run, StageRunContext) else False:
            self._assert_raw_item_partition(prepared, run)
        with self._logged_mutation(
            run, operation="ingest_raw_items", terminal=terminal
        ) as (connection, context):
            results = [self._insert_raw_item(connection, values) for values in prepared]
            if source_state is not None:
                self._set_source_state(connection, source_state)
            inserted = sum(1 for result in results if result.inserted)
            context._record_outcome(success=inserted, partial=len(results) - inserted)
            context._merge_counts(
                {"raw_items_seen": len(results), "raw_items_inserted": inserted}
            )
            return results

    @staticmethod
    def _assert_raw_item_partition(
        prepared: Sequence[Mapping[str, Any]], run: Any
    ) -> None:
        """Reject a mixed or foreign-partition ingestion batch before writing."""

        if not isinstance(run, StageRunContext):
            return  # _logged_mutation rejects the handle itself.
        for position, values in enumerate(prepared):
            asserted = {values["ticker"]} | set(values["tickers"])
            asserted |= {
                normalize_ticker(candidate.get("ticker"))
                for candidate in values["candidate_tickers"]
                if isinstance(candidate, Mapping) and candidate.get("ticker")
            }
            asserted.discard(None)
            if run.ticker is not None and asserted - {run.ticker}:
                raise Phase0RunContextError(
                    f"ingest_raw_items item {position} asserts "
                    f"{sorted(asserted - {run.ticker})} but the run covers "
                    f"{run.ticker}"
                )
            if run.ticker is None and len(asserted) > 1:
                raise Phase0RunContextError(
                    f"ingest_raw_items item {position} mixes tickers "
                    f"{sorted(asserted)} in a run with no ticker"
                )
            stamp = values["published_at"] or values["fetched_at"]
            day = str(stamp)[:10]
            if day != run.trading_day:
                raise Phase0RunContextError(
                    f"ingest_raw_items item {position} falls on {day} but the "
                    f"run covers {run.trading_day}"
                )

    def persist_embeddings(
        self,
        embeddings: Sequence[PersistedEmbedding],
        *,
        run: Any,
        terminal: bool = False,
    ) -> int:
        """Persist an M1 embedding batch and its run log in one transaction.

        Every source must already live in the run's partition.  An
        embedding is derived from a raw item, story, or theme, so a vector
        whose source belongs to another ticker-day is either a bug or a
        cross-partition write dressed up as a cache fill.
        """

        prepared = [validate_embedding(embedding) for embedding in embeddings]
        with self._logged_mutation(
            run, operation="persist_embeddings", terminal=terminal
        ) as (connection, context):
            self._assert_embedding_partition(connection, prepared, context)
            for values in prepared:
                self._write_embedding(connection, values)
            context._record_outcome(success=len(prepared))
            context._merge_counts({"embeddings_written": len(prepared)})
            return len(prepared)

    @staticmethod
    def _assert_embedding_partition(
        connection: sqlite3.Connection,
        prepared: Sequence[Mapping[str, Any]],
        run: StageRunContext,
    ) -> None:
        queries = {
            "raw_item": (
                "SELECT ticker, substr(COALESCE(published_at, fetched_at), 1, 10) "
                "AS day, NULL AS pipeline_version FROM raw_items WHERE id = ?"
            ),
            "story": (
                "SELECT ticker, trading_day AS day, pipeline_version "
                "FROM stories WHERE id = ?"
            ),
            "theme": (
                "SELECT ticker, trading_day AS day, pipeline_version "
                "FROM themes WHERE id = ?"
            ),
        }
        for position, values in enumerate(prepared):
            row = connection.execute(
                queries[values["source_kind"]], (values["source_id"],)
            ).fetchone()
            if row is None:
                raise Phase0RunContextError(
                    f"persist_embeddings source {position} "
                    f"({values['source_kind']} {values['source_id']}) does not exist"
                )
            if row["ticker"] is not None and run.ticker is not None:
                if str(row["ticker"]) != run.ticker:
                    raise Phase0RunContextError(
                        f"persist_embeddings source {position} belongs to "
                        f"{row['ticker']} but the run covers {run.ticker}"
                    )
            if str(row["day"]) != run.trading_day:
                raise Phase0RunContextError(
                    f"persist_embeddings source {position} falls on {row['day']} "
                    f"but the run covers {run.trading_day}"
                )
            version = row["pipeline_version"]
            if version is not None and str(version) != run.pipeline_version:
                raise Phase0RunContextError(
                    f"persist_embeddings source {position} belongs to pipeline "
                    f"version {version} but the run covers {run.pipeline_version}"
                )

    def record_source_state(
        self,
        source: str,
        *,
        run: Any,
        etag: str | None = None,
        last_modified: str | None = None,
        checked_at: str | None = None,
        successful: bool = True,
        metadata: Mapping[str, Any] | None = None,
        status: str | None = None,
        error: Any = None,
        retry_after: str | None = None,
        terminal: bool = False,
    ) -> None:
        """Record ingestion-coupled source state and its run log together.

        Source state is keyed by feed, not by ticker-day, so there is no
        ticker to cross-check.  What is checked is the coupling the name
        promises: when the caller states a ``checked_at``, it must fall on
        the run's trading day, so a run cannot stamp another day's fetch as
        its own.  Omitting it means "now", which asserts no day.
        """

        moment = _normalize_datetime(checked_at or utc_now(), "checked_at")
        if (
            checked_at is not None
            and isinstance(run, StageRunContext)
            and self._is_active_run(run)
            and moment[:10] != run.trading_day
        ):
            raise Phase0RunContextError(
                f"record_source_state checked {moment[:10]} but the run covers "
                f"{run.trading_day}"
            )
        state = {
            "source": source,
            "etag": etag,
            "last_modified": last_modified,
            "checked_at": moment,
            "successful": successful,
            "metadata": metadata or {},
            "status": status,
            "error": error,
            "retry_after": retry_after,
        }
        with self._logged_mutation(
            run, operation="record_source_state", terminal=terminal
        ) as (connection, context):
            self._set_source_state(connection, state)
            context._record_outcome(
                success=1 if successful else 0,
                partial=0 if successful else 1,
            )
            context._merge_counts({"source_states_recorded": 1})

    def run_log_entries(
        self,
        *,
        trading_day: str | date | None = None,
        stage: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if trading_day is not None:
            clauses.append("trading_day = ?")
            parameters.append(_normalize_day(trading_day))
        if stage is not None:
            clauses.append("stage = ?")
            parameters.append(_require_text(stage, "stage"))
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(_require_text(run_id, "run_id"))
        query = "SELECT * FROM run_log"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode_run_log_row(row) for row in rows]

    @staticmethod
    def _decode_run_log_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["counts"] = json.loads(result["counts"])
        result["errors"] = json.loads(result["errors"])
        return result

    def latest_stage_status(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT log.*
                FROM run_log AS log
                JOIN (
                    SELECT stage, MAX(id) AS latest_id
                    FROM run_log GROUP BY stage
                ) AS latest ON latest.latest_id = log.id
                ORDER BY log.stage
                """
            ).fetchall()
        return [self._decode_run_log_row(row) for row in rows]

    def pipeline_status(self) -> dict[str, Any]:
        stages = self.latest_stage_status()
        successful_times = [
            row["completed_at"] for row in stages if row["status"] == "success"
        ]
        any_times = [row["completed_at"] for row in stages]
        return {
            "data_as_of": max(successful_times or any_times, default=None),
            "stages": stages,
        }

    def count(self, table: str) -> int:
        if table not in COUNTABLE_TABLES:
            raise Phase0ValidationError("unsupported table")
        with self._connect() as connection:
            return int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )


__all__ = [
    "COUNTABLE_TABLES",
    "DEFAULT_DATABASE_PATH",
    "EmbeddingPersistenceError",
    "ExcludedStoryRecord",
    "InsertResult",
    "MIGRATIONS_PATH",
    "OtherCoverageRecord",
    "Phase0Admin",
    "Phase0Error",
    "Phase0IntegrityError",
    "Phase0MigrationError",
    "Phase0Repository",
    "Phase0RunContextError",
    "Phase0ValidationError",
    "ProviderConflictRecord",
    "ReconciliationReport",
    "RUN_STATUSES",
    "SECRET_KEY_PATTERN",
    "SOURCE_STATE_STATUSES",
    "SUPPORTED_TICKERS",
    "STAGE_KEY_STATUSES",
    "SemanticMergeRecord",
    "StageKeyError",
    "StageRunContext",
    "StageRunRecorder",
    "StoryMemberRecord",
    "StoryRecord",
    "THEME_STATUSES",
    "TICKER_UNIVERSE",
    "ThemeRecord",
    "ThemeSetRecord",
    "UnsupportedTickerError",
    "normalize_ticker",
    "redact_secrets",
    "redact_text",
    "require_safe_identifier_scalar",
    "sanitize_diagnostic_scalar",
    "serialize_operational_metadata",
    "serialize_raw_evidence",
    "utc_now",
    "validate_safe_identifier_scalar",
]

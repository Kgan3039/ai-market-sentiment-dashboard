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

import contextlib
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
    normalize_source_kind,
    require_durable_source_id,
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
from .schema import (
    LINEAGE_TABLE,
    apply_migrations,
    load_migrations,
    split_statements,
)
from .tickers import SUPPORTED_TICKERS, TICKER_UNIVERSE, normalize_ticker


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "phase0.sqlite3"
MIGRATIONS_PATH = Path(__file__).with_name("migrations")
RUN_STATUSES = {"success", "degraded", "failed"}
STAGE_KEY_STATUSES = {"success", "degraded", "failed"}
RUNNING_STATUS = "running"
THEME_STATUSES = {"pending", "ready", "degraded", "failed"}
INGEST_STATUSES = {"valid", "invalid", "ambiguous"}
SOURCE_STATE_STATUSES = {"success", "partial", "empty", "failed", "unknown"}

#: The statuses the schema itself treats as a successful fetch: these are
#: the ones that stamp ``last_success_at`` and leave
#: ``consecutive_failures`` at zero.  The run-log outcome is derived from
#: the same set, so a source state and the run that recorded it cannot
#: disagree about whether the fetch worked.
SUCCEEDED_SOURCE_STATE_STATUSES = frozenset({"success", "partial", "empty"})
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
# What reconciliation owns.
#
# Story and theme reconciliation decide "unchanged" by comparing what is
# stored against what a settlement would write.  That comparison is only
# as honest as its column list, so the list lives here, next to the
# statements built from it, and the contract tests assert that every
# column of ``stories``/``themes`` is either on it or deliberately
# exempt.  A column added to the table and written by reconciliation but
# missed here would make a real change look like a replay.
# ----------------------------------------------------------------------

#: ``stories`` columns one reconciliation owns; see
#: :meth:`Phase0Repository._story_column_values` for what is excluded and
#: why.  Sorted, because the order is a comparison key, not a schema.
STORY_RECONCILED_COLUMNS: tuple[str, ...] = (
    "algorithm_version",
    "canonical_title",
    "canonical_url",
    "config_fingerprint",
    "content_hash",
    "embedding",
    "embedding_dimension",
    "member_ids",
    "member_story_keys",
    "model_name",
    "model_revision",
    "outlet",
    "outlet_count",
    "published_at",
    "quarantined",
    "semantic_skip_reason",
    "source",
    "stage",
)

#: ``themes`` columns one reconciliation owns; see
#: :meth:`Phase0Repository._theme_column_values`.
THEME_RECONCILED_COLUMNS: tuple[str, ...] = (
    "algorithm_version",
    "centroid",
    "citations",
    "cohesion",
    "config_fingerprint",
    "content_hash",
    "embedding_dimension",
    "label",
    "label_source",
    "latest_published_at",
    "matched_previous_key",
    "method",
    "min_pairwise_cohesion",
    "model_name",
    "model_revision",
    "outlet_count",
    "salience",
    "salience_outlet_component",
    "salience_rank",
    "salience_recency_component",
    "salience_story_component",
    "status",
    "story_count",
    "summary",
    "theme_key",
)

#: ``theme_sets`` columns one reconciliation owns; see
#: :meth:`Phase0Repository._theme_set_column_values`.  The theme set is not
#: a theme, so none of these reach the per-theme comparison — they need
#: their own, or a reconciliation that rewrites the day's quality and trust
#: metadata reports that nothing happened.
THEME_SET_RECONCILED_COLUMNS: tuple[str, ...] = (
    "algorithm_version",
    "config_fingerprint",
    "embedding_dimension",
    "method",
    "method_reason",
    "model_name",
    "model_revision",
    "quality",
    "source_metadata",
    "trust_metadata",
)

#: Columns whose stored value is compared as an ``int``.  SQLite returns
#: the right type for a declared INTEGER column, but a comparison that
#: silently depends on that is a comparison waiting to go wrong.
_INTEGER_RECONCILED_COLUMNS = frozenset(
    {
        "embedding_dimension",
        "outlet_count",
        "quarantined",
        "salience_rank",
        "story_count",
    }
)

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
        "schema_lineage",
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


def normalize_candidate_tickers(value: Any) -> list[dict[str, str]]:
    """Normalize ``candidate_tickers`` once, for validation *and* storage.

    Two accepted forms, and nothing else:

    * a bare symbol — ``"NVDA"``, ``" nvda "`` — which records the reason
      ``relevance_match``;
    * a mapping with ``ticker`` and an optional ``reason`` —
      ``{"ticker": "NVDA", "reason": "headline_match"}``.

    Anything else (``None`` in the list, a number, a nested list, a mapping
    with no ``ticker``, a blank string, ``"NVDA AMD"``) raises, and one bad
    candidate rejects the whole item rather than being dropped quietly.

    **Duplicates:** the first mention of a symbol wins, including its
    reason, and the result is sorted by symbol.  Deterministic order
    matters because these rows are compared across replays.

    This function existing *once* is the fix, not the parsing it does.  A
    second parser downstream is how ``candidate_tickers=["AMD"]`` came to
    be stored under an NVDA run: the validator understood only the mapping
    form, and the writer understood both.
    """

    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        raise Phase0ValidationError(
            "candidate_tickers must be a sequence of symbols or mappings"
        )
    if not isinstance(value, Sequence):
        raise Phase0ValidationError(
            "candidate_tickers must be a sequence of symbols or mappings"
        )

    seen: dict[str, str] = {}
    for position, candidate in enumerate(value):
        if isinstance(candidate, Mapping):
            if "ticker" not in candidate:
                raise Phase0ValidationError(
                    f"candidate ticker {position} is a mapping with no 'ticker'"
                )
            ticker = normalize_ticker(
                candidate.get("ticker"), field=f"candidate ticker {position}"
            )
            raw_reason = candidate.get("reason")
            reason = str(raw_reason).strip() if raw_reason is not None else ""
            reason = reason or DEFAULT_CANDIDATE_REASON
        elif isinstance(candidate, str):
            ticker = normalize_ticker(candidate, field=f"candidate ticker {position}")
            reason = DEFAULT_CANDIDATE_REASON
        else:
            raise Phase0ValidationError(
                f"candidate ticker {position} must be a symbol or a mapping, "
                f"not {type(candidate).__name__}"
            )
        # normalize_ticker raises rather than returning None here.
        seen.setdefault(str(ticker), reason)
    return [{"ticker": ticker, "reason": seen[ticker]} for ticker in sorted(seen)]


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


# ----------------------------------------------------------------------
# The lifecycle of one run, as states rather than a pair of booleans.
#
# Two booleans could disagree, and did: a terminal operation that failed
# set "terminated", teardown then read it as "already finalized, but the
# stage also failed" and finalized a second time — whose StageKeyError,
# raised from a ``finally``, replaced the real exception on its way out.
# A state is one fact, and the transitions out of a terminal state are
# refused rather than merged.
#
# Every transition out of ACTIVE happens *after* the transaction that
# earns it has committed.  In-memory success ahead of a durable commit is
# a lie waiting for a disk error: the rollback takes the data and the run
# log with it, while the object still says the stage succeeded.
# ----------------------------------------------------------------------

#: Open for business: operations may run, nothing final has been written.
RUN_STATE_ACTIVE = "active"
#: A terminal operation's data, run log, and key release are **committed**.
#: Immutable — nothing may overwrite that outcome afterwards.
RUN_STATE_TERMINAL_SUCCEEDED = "terminal_succeeded"
#: An operation failed and the failure settlement committed, once.
RUN_STATE_TERMINAL_FAILED = "terminal_failed"
#: An operation failed and recording that failure *also* failed to commit.
#: The persisted outcome is unknown, not successful: the stage key is left
#: as it was, so the lease expires and ordinary recovery reclaims it.
RUN_STATE_SETTLEMENT_FAILED = "settlement_failed"
#: The block ended with no operation ever declaring itself terminal, so
#: teardown committed the single degraded/retryable outcome.
RUN_STATE_CLOSED_WITHOUT_TERMINAL = "closed_without_terminal"

RUN_STATES = frozenset(
    {
        RUN_STATE_ACTIVE,
        RUN_STATE_TERMINAL_SUCCEEDED,
        RUN_STATE_TERMINAL_FAILED,
        RUN_STATE_SETTLEMENT_FAILED,
        RUN_STATE_CLOSED_WITHOUT_TERMINAL,
    }
)

#: States in which this run's outcome is decided and teardown adds nothing.
#: ``settlement_failed`` is here on purpose: whatever is on disk, trying
#: again from a ``finally`` would be the duplicate settlement this design
#: exists to prevent.
_SETTLED_RUN_STATES = frozenset(
    {
        RUN_STATE_TERMINAL_SUCCEEDED,
        RUN_STATE_TERMINAL_FAILED,
        RUN_STATE_SETTLEMENT_FAILED,
        RUN_STATE_CLOSED_WITHOUT_TERMINAL,
    }
)

#: States an operation reached by settling (or failing to settle) itself,
#: as opposed to the block simply ending.
_TERMINAL_RUN_STATES = frozenset(
    {
        RUN_STATE_TERMINAL_SUCCEEDED,
        RUN_STATE_TERMINAL_FAILED,
        RUN_STATE_SETTLEMENT_FAILED,
    }
)


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
        "_state",
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
        setter(self, "_state", RUN_STATE_ACTIVE)
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
    def state(self) -> str:
        """Where this run is in its lifecycle; see ``RUN_STATE_*``.

        Read-only, like everything else here.  Only repository internals
        move it, through :meth:`_transition`, and only out of
        ``RUN_STATE_ACTIVE``.
        """

        return self._state

    @property
    def terminated(self) -> bool:
        """True once an operation settled this run, whichever way it went."""

        return self._state in _TERMINAL_RUN_STATES

    @property
    def settled(self) -> bool:
        """True once this run's outcome is written and may not change."""

        return self._state in _SETTLED_RUN_STATES

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

    def _transition(self, state: str) -> None:
        """Move the run to a settled state, once.

        Leaving a settled state is refused rather than ignored: every way
        of reaching one has already written an authoritative outcome, and
        overwriting it is precisely the bug — a second terminal call
        rewriting a committed success to ``failed``.  Callers check
        :attr:`settled` first, so reaching here twice is a repository bug,
        not a caller error.
        """

        if state not in RUN_STATES:
            raise Phase0RunContextError(f"unknown run state {state!r}")
        if self._state in _SETTLED_RUN_STATES:
            raise Phase0RunContextError(
                f"this run is already {self._state}; its outcome is written "
                "and cannot be changed"
            )
        object.__setattr__(self, "_state", state)

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

        The only raw connection anything outside this module can obtain,
        and it is spelled ``repository.admin.connect_writable()`` so that
        the call site says what it is.  Nothing validates what you do with
        it: no ticker check, no partition check, no run log.

        It exists because an operator occasionally has to fix one row by
        hand, and because migrations need it.

        Pipeline code must never call it: use the logged entrypoints on
        :class:`Phase0Repository` for writes and :class:`Phase0Reader`
        (``repository.read``) for reads, neither of which hands back a
        connection at all.
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

    # -- Stage keys ------------------------------------------------------

    def complete_stage_key(self, **kwargs: Any) -> None:
        """Force a stage key to a terminal status. **Manual repair only.**

        This writes no data and no ``run_log`` row, so a ``success`` written
        here is an assertion by an operator, not evidence that a stage ran.
        Use it to release a key an operator has finished repairing by hand.

        A pipeline stage must never call it — it declares completion by
        passing ``terminal=True`` to its last logged mutation, which commits
        the data, the final run log, and the key's transition in one
        transaction.  That is why this lives behind ``repository.admin`` and
        not on :class:`Phase0Repository`.
        """

        return self._repository._complete_stage_key_unlogged(**kwargs)

    # -- Run log ---------------------------------------------------------

    def log_stage(self, **kwargs: Any) -> int:
        """Write a run-log row directly, with no mutation attached.

        For repairing or backfilling history only.  A stage records itself
        through ``stage_run``.
        """

        return self._repository._log_stage_unlogged(**kwargs)


class Phase0Reader:
    """The public read surface: named queries, and no connection.

    Handing out a live :class:`sqlite3.Connection` cannot be made safe by
    hardening the connection.  Whatever the handle arrives with —
    ``mode=ro``, ``query_only``, an authorizer — the caller holding it can
    take back off again::

        connection.set_authorizer(None)
        connection.execute("PRAGMA query_only = OFF")
        connection.execute("ATTACH DATABASE '...' AS alias")
        connection.execute("INSERT INTO alias.raw_items ...")
        connection.commit()

    So this object never yields one.  Each method opens a private
    read-only connection, runs *one query this module wrote*, converts the
    rows to plain dictionaries, and closes the connection before
    returning.  There is no ``execute``, no ``cursor``, no
    ``executescript``, no ``commit``/``rollback``, no ``set_authorizer``,
    and no attribute holding a connection: ``__slots__`` is a single
    :class:`~pathlib.Path`.

    Callers who need a read Phase 0 does not expose should add a method
    here rather than reaching for raw SQL — that keeps the read surface
    reviewable, which is the point.  Deliberate raw access still exists
    for operators, spelled out at :meth:`Phase0Admin.connect_writable`.
    """

    __slots__ = ("_database_path",)

    def __init__(self, database_path: str | Path) -> None:
        object.__setattr__(self, "_database_path", Path(database_path))

    def __repr__(self) -> str:
        return f"<Phase0Reader {self._database_path}>"

    # -- The one place a connection exists, and it never leaves ---------

    def _query(self, sql: str, parameters: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run one module-authored query and return plain rows.

        ``sql`` is always a literal from the method calling this; it is
        never a caller's string.  The read-only open mode and the
        authorizer stay as defense in depth for the moment this connection
        is alive, but the real guarantee is that it does not outlive this
        call.
        """

        if not self._database_path.exists():
            raise Phase0ValidationError(
                f"no Phase 0 database at {self._database_path}; call migrate() first"
            )
        connection = sqlite3.connect(
            f"file:{self._database_path}?mode=ro", uri=True, timeout=10
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 10000")
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(_read_only_authorizer)
            return [dict(row) for row in connection.execute(sql, tuple(parameters))]
        finally:
            connection.close()

    def _one(self, sql: str, parameters: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self._query(sql, parameters)
        return rows[0] if rows else None

    def _table(self, table: str) -> str:
        """A real table name, checked against the schema, never a fragment.

        ``PRAGMA table_info(?)`` is not a thing SQLite will bind, so the
        name has to be interpolated; it is looked up in ``sqlite_master``
        first so what gets interpolated is a name the database already
        has.
        """

        name = _require_text(table, "table")
        if name not in set(self.table_names()):
            raise Phase0ValidationError(f"unknown Phase 0 table {name!r}")
        return name

    # -- Raw evidence ---------------------------------------------------

    def raw_item(self, item_id: int) -> dict[str, Any] | None:
        return self._one(
            "SELECT * FROM raw_items WHERE id = ?",
            (_require_int(item_id, "item_id", minimum=1),),
        )

    def raw_item_candidates(self, item_id: int | None = None) -> list[dict[str, Any]]:
        if item_id is None:
            return self._query(
                "SELECT * FROM raw_item_candidates ORDER BY raw_item_id, ticker"
            )
        return self._query(
            "SELECT * FROM raw_item_candidates WHERE raw_item_id = ? ORDER BY ticker",
            (_require_int(item_id, "item_id", minimum=1),),
        )

    def raw_item_associations(self, item_id: int | None = None) -> list[dict[str, Any]]:
        if item_id is None:
            return self._query(
                "SELECT * FROM raw_item_tickers ORDER BY raw_item_id, ticker"
            )
        return self._query(
            "SELECT * FROM raw_item_tickers WHERE raw_item_id = ? ORDER BY ticker",
            (_require_int(item_id, "item_id", minimum=1),),
        )

    # -- Derived output --------------------------------------------------

    def story(self, story_id: int) -> dict[str, Any] | None:
        return self._one(
            "SELECT * FROM stories WHERE id = ?",
            (_require_int(story_id, "story_id", minimum=1),),
        )

    def theme(self, theme_id: int) -> dict[str, Any] | None:
        return self._one(
            "SELECT * FROM themes WHERE id = ?",
            (_require_int(theme_id, "theme_id", minimum=1),),
        )

    def source_state_rows(self) -> list[dict[str, Any]]:
        return self._query("SELECT * FROM source_state ORDER BY source")

    # -- The operational ledger ------------------------------------------

    def run_log_rows(
        self,
        *,
        run_id: str | None = None,
        stage: str | None = None,
        trading_day: str | date | None = None,
    ) -> list[dict[str, Any]]:
        """Whole ``run_log`` rows, in insertion order.

        :meth:`Phase0Repository.run_log_entries` decodes counts and errors
        for a reader; this is the raw row, for tests and operators
        comparing two states of the ledger exactly.
        """

        clauses: list[str] = []
        parameters: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            parameters.append(_require_text(run_id, "run_id"))
        if stage is not None:
            clauses.append("stage = ?")
            parameters.append(_require_text(stage, "stage"))
        if trading_day is not None:
            clauses.append("trading_day = ?")
            parameters.append(_normalize_day(trading_day))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._query(f"SELECT * FROM run_log{where} ORDER BY id", parameters)

    def stage_key_rows(
        self,
        *,
        stage: str | None = None,
        ticker: str | None = None,
        trading_day: str | date | None = None,
        pipeline_version: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if stage is not None:
            clauses.append("stage = ?")
            parameters.append(_require_text(stage, "stage"))
        if ticker is not None:
            clauses.append("ticker = ?")
            parameters.append(normalize_ticker(ticker))
        if trading_day is not None:
            clauses.append("trading_day = ?")
            parameters.append(_normalize_day(trading_day))
        if pipeline_version is not None:
            clauses.append("pipeline_version = ?")
            parameters.append(_require_text(pipeline_version, "pipeline_version"))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._query(
            f"SELECT * FROM pipeline_stage_keys{where} "
            "ORDER BY trading_day, ticker, stage",
            parameters,
        )

    # -- Schema inspection, for #82/#83/#68 rebases -----------------------

    def table_names(self) -> list[str]:
        return [
            row["name"]
            for row in self._query(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]

    def schema_objects(self) -> list[dict[str, Any]]:
        return self._query(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )

    def table_columns(self, table: str) -> list[dict[str, Any]]:
        return self._query(f"PRAGMA table_info({self._table(table)})")

    def foreign_keys(self, table: str) -> list[dict[str, Any]]:
        return self._query(f"PRAGMA foreign_key_list({self._table(table)})")

    def indexes(self, table: str) -> list[dict[str, Any]]:
        return self._query(f"PRAGMA index_list({self._table(table)})")

    def integrity_check(self) -> str:
        rows = self._query("PRAGMA integrity_check")
        return str(next(iter(rows[0].values()))) if rows else "unknown"

    def schema_version(self) -> int:
        rows = self._query("PRAGMA user_version")
        return int(next(iter(rows[0].values()))) if rows else 0

    def count(self, table: str) -> int:
        if table not in COUNTABLE_TABLES:
            raise Phase0ValidationError(f"unknown Phase 0 table {table!r}")
        rows = self._query(f"SELECT COUNT(*) AS total FROM {table}")
        return int(rows[0]["total"])


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
        #: The public read surface.  Holds a path, never a connection.
        self.read = Phase0Reader(self.database_path)
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

    def schema_lineages(self) -> list[dict[str, Any]]:
        """Historical lineages this database was recognized as, if any.

        Empty for a database built on the approved migrations, which is
        every fresh one.  A row here says this database arrived from a
        known historical fork and names it — the ledger still records the
        *historical* checksum for the migration that actually ran, so the
        two together tell the whole story without either one lying.
        """

        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    f"SELECT * FROM {LINEAGE_TABLE} ORDER BY lineage"
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
            "candidate_tickers": normalize_candidate_tickers(
                item.get("candidate_tickers")
            ),
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
        # Already normalized by :func:`normalize_candidate_tickers`, in
        # ``_prepare_raw_item``.  Parsing them a second time here is what
        # let a bare ``"AMD"`` past a validator that only understood the
        # mapping form, so this loop deliberately understands one shape.
        for candidate in values.get("candidate_tickers") or []:
            connection.execute(
                """
                INSERT INTO raw_item_candidates (raw_item_id, ticker, reason)
                VALUES (?, ?, ?)
                ON CONFLICT(raw_item_id, ticker)
                DO UPDATE SET reason = excluded.reason
                """,
                (item_id, candidate["ticker"], candidate["reason"]),
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
        # One resolved outcome, and only one.  ``status`` is the richer
        # statement — the boolean cannot express ``partial``, ``empty``,
        # or ``unknown`` at all — so when it is given it decides.
        # ``successful`` is then a claim about the same thing, and a
        # caller that states both and disagrees with itself is refused
        # rather than silently having one half honoured.
        status = state.get("status")
        successful = state.get("successful")
        if status is None:
            # ``None`` is "not stated", which is not an explicit ``False``.
            # Collapsing the two with ``bool(successful)`` made a payload
            # that said nothing resolve to *failed* here while
            # ``record_source_state`` — which patched the default at its
            # own call site — resolved the same payload to *success*.  The
            # default belongs in the one resolver, so every entrypoint
            # inherits it: saying nothing means success.
            if successful is None:
                resolved_status = "success"
            else:
                resolved_status = "success" if successful else "failed"
        else:
            resolved_status = str(status).strip().lower()
            if resolved_status not in SOURCE_STATE_STATUSES:
                raise Phase0ValidationError(
                    f"invalid source-state status: {resolved_status}"
                )
            if successful is not None and bool(successful) != (
                resolved_status in SUCCEEDED_SOURCE_STATE_STATUSES
            ):
                raise Phase0ValidationError(
                    f"source state for {source!r} says successful="
                    f"{bool(successful)} and status={resolved_status!r}, which "
                    f"disagree; state the outcome once"
                )
        succeeded = resolved_status in SUCCEEDED_SOURCE_STATE_STATUSES
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
    ) -> dict[str, Any]:
        """Write one source state; return the values actually persisted.

        Returning them is what lets the caller's run outcome be derived
        from the *resolved* status rather than from its own second copy
        of the same question.
        """

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
        return values

    def _set_source_state_unlogged(
        self,
        source: str,
        *,
        etag: str | None,
        last_modified: str | None,
        checked_at: str,
        successful: bool | None = None,
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
        """Return the current embedding for a source, or ``None``.

        Takes a durable row id; see :func:`~phase0.embeddings
        .require_durable_source_id` for why a fingerprint is refused
        rather than looked up.  A *missing* id is an ordinary cache miss
        and returns ``None``; a partition-scoped one is a caller error and
        is raised, because silently missing forever would be worse.
        """

        kind = normalize_source_kind(source_kind)
        identity = require_durable_source_id(kind, source_id)
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

        Because it carries no run, it has no partition to judge an
        identity against, so it takes only the identity that needs none: a
        durable row id.  See :func:`~phase0.embeddings
        .require_durable_source_id`.  The protocol is unchanged — its
        ``source_id`` is text either way — and this narrows which text is
        a cache key, not the shape of the call.
        """

        values = validate_embedding(embedding)
        values["source_id"] = require_durable_source_id(
            values["source_kind"], values["source_id"]
        )
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
        """Drop one source's embedding; ``True`` when a row was removed.

        Durable ids only, and here the reason is sharper than for a read:
        a fingerprint two partitions share names one row, so honouring it
        would let either partition delete the other's cache entry by
        asking for its own.
        """

        kind = normalize_source_kind(source_kind)
        identity = require_durable_source_id(kind, source_id)
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
    def _story_column_values(values: Mapping[str, Any]) -> dict[str, Any]:
        """The exact ``stories`` columns one reconciliation owns and writes.

        This mapping is the single definition of "what a settlement puts
        in the row": :meth:`_insert_reconciled_story` and
        :meth:`_update_reconciled_story` both build their statement from
        it, and :meth:`_story_signature` compares it.  A column therefore
        cannot be persisted by a settlement and left out of the comparison
        that decides whether a settlement is needed — which is precisely
        how a stale embedding survived an otherwise-identical replay.

        Absent on purpose: the partition identity (``ticker``,
        ``trading_day``, ``pipeline_version``, ``cluster_fingerprint``),
        which is what pairs an incoming story with a stored one rather
        than something to compare; ``updated_at`` and ``invalidated_at``,
        which are bookkeeping this class sets itself; and
        ``canonical_item_id``, which the canonical-member trigger requires
        to be written *after* membership exists.  The last two are still
        part of the equality contract — see :meth:`_story_signature`.
        """

        return {
            "canonical_title": values["canonical_title"],
            "embedding": values["embedding"],
            "outlet_count": values["outlet_count"],
            "member_ids": json.dumps(values["member_ids"], separators=(",", ":")),
            "stage": values["stage"],
            "canonical_url": values["canonical_url"],
            "source": values["source"],
            "outlet": values["outlet"],
            "published_at": values["published_at"],
            "content_hash": values["content_hash"],
            "algorithm_version": values["algorithm_version"],
            "config_fingerprint": values["config_fingerprint"],
            "model_name": values["model_name"],
            "model_revision": values["model_revision"],
            "embedding_dimension": values["embedding_dimension"],
            "quarantined": values["quarantined"],
            "semantic_skip_reason": values["semantic_skip_reason"],
            "member_story_keys": values["member_story_keys"],
        }

    @classmethod
    def _story_signature(cls, values: Mapping[str, Any]) -> tuple[Any, ...]:
        """Everything about a story that one reconciliation persists.

        Equality here means one thing exactly: *a settlement would write
        what is already stored*.  So it covers every owned column, the
        canonical member, whether the row is live, and the full payload of
        every child relation this path owns — not merely the child
        identities.  A replay that changes only an embedding, only a
        provider conflict's fields, only a merge's similarity, or only a
        member's outlet is a change, and has to take the update path.

        Ordering is canonical on both sides (members by raw item,
        conflicts by provider identity, merges by story-key pair), so two
        semantically identical results compare equal however the caller
        happened to order them.
        """

        columns = cls._story_column_values(values)
        return (
            tuple(columns[column] for column in STORY_RECONCILED_COLUMNS),
            values["canonical_item_id"],
            # A reconciled story is live by definition: the update path
            # clears ``invalidated_at``.  A story that was invalidated and
            # comes back unchanged is therefore *not* unchanged.
            True,
            tuple(
                (
                    member["raw_item_id"],
                    member["position"],
                    member["outlet"],
                    member["url"],
                    member["canonical_url"],
                    member["match_reason"],
                    member["quarantined"],
                )
                for member in sorted(
                    values["members"], key=lambda member: member["raw_item_id"]
                )
            ),
            tuple(
                (
                    conflict["provider_namespace"],
                    conflict["provider_item_id"],
                    conflict["item_ids"],
                    conflict["fields"],
                )
                for conflict in sorted(
                    values["provider_conflicts"],
                    key=lambda conflict: (
                        conflict["provider_namespace"],
                        conflict["provider_item_id"],
                    ),
                )
            ),
            tuple(
                (
                    merge["left_story_key"],
                    merge["right_story_key"],
                    merge["similarity"],
                    merge["reason"],
                )
                for merge in sorted(
                    values["semantic_merges"],
                    key=lambda merge: (
                        merge["left_story_key"],
                        merge["right_story_key"],
                    ),
                )
            ),
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
        member raw item must already belong to the run's ticker-day *and*
        be explicitly associated with its ticker, so a story cannot quietly
        drag foreign — or unattributed — evidence into this partition.

        Argument normalization happens inside the logged mutation together
        with those checks: an unsupported ticker or a malformed day is a
        recorded failure of this run, not an exception raised before the
        run took responsibility for the operation.
        """

        with self._logged_mutation(
            run, operation="reconcile_stories", terminal=terminal
        ) as (connection, context):
            normalized_ticker = normalize_ticker(ticker)
            day = _normalize_day(trading_day)
            version = _require_text(pipeline_version, "pipeline_version")
            self._assert_run_partition(
                context,
                operation="reconcile_stories",
                ticker=normalized_ticker,
                trading_day=day,
                pipeline_version=version,
            )
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
    def _assert_raw_item_association(
        connection: sqlite3.Connection,
        raw_item_id: int,
        ticker: str,
        *,
        operation: str,
    ) -> None:
        """The one rule for pulling raw evidence into ticker-scoped output.

        A raw item may take part in ``ticker``'s derived processing only
        when it is *explicitly* associated with it, which means either:

        * ``raw_items.ticker`` is that symbol; or
        * an accepted association exists in ``raw_item_tickers`` — the
          authoritative relationship table.  A ``raw_item_candidates`` row
          is a suggestion nothing has accepted, and does not count.

        Ingestion may still store an item with ``ticker=None`` and no
        associations: unattributable evidence is real, and the spec says to
        keep it.  What it may not do is drift, later, into being an NVDA
        story member or an NVDA embedding source just because an NVDA run
        happened to be the one holding the transaction.

        **Multiple associations.**  An item associated with both NVDA and
        AMD is legitimate — one article can be about two companies — and
        the rule is membership, not exclusivity: each run requires its own
        ticker to be among the associations and ignores the others.  Each
        ticker's derived output is therefore built independently, and an
        AMD-only item stays out of NVDA's.
        """

        associated = connection.execute(
            f"""
            SELECT 1 FROM {RAW_ITEM_ASSOCIATION_TABLE}
            WHERE raw_item_id = ? AND ticker = ?
            """,
            (raw_item_id, ticker),
        ).fetchone()
        if associated is not None:
            return
        # Name the associations it *does* have.  This is the message a
        # caller sees for a genuinely foreign item, and "belongs to AMD"
        # was the old wording for a case that turned out to include
        # legitimately multi-ticker evidence; saying which tickers claim
        # it is both truer and more useful.
        held = [
            str(row["ticker"])
            for row in connection.execute(
                f"SELECT ticker FROM {RAW_ITEM_ASSOCIATION_TABLE} "
                "WHERE raw_item_id = ? ORDER BY ticker",
                (raw_item_id,),
            )
        ]
        belongs = f"it is associated with {held}" if held else "it is unattributed"
        raise Phase0RunContextError(
            f"{operation}: raw item {raw_item_id} has no accepted association "
            f"with {ticker} ({belongs}); associate it explicitly before using "
            f"it in {ticker}'s derived output"
        )

    @classmethod
    def _assert_members_in_partition(
        cls,
        connection: sqlite3.Connection,
        stories: Sequence[StoryRecord],
        *,
        ticker: str,
        trading_day: str,
    ) -> None:
        """Every member raw item must sit in this ticker-day already.

        Ticker membership is decided by
        :meth:`_assert_raw_item_association` and nowhere else.  There used
        to be a second, stricter test here — ``raw_items.ticker`` had to
        be this ticker or nothing — which read the primary ticker as
        *exclusive* and so refused an AMD-primary article that also
        carried an accepted NVDA association.  One article about two
        companies is ordinary, ``raw_item_tickers`` is the table that
        records it, and ``raw_items_for_day`` already read it that way;
        this path was the one disagreeing.

        Nothing is loosened by removing it: an item with no NVDA
        association is still refused, and unattributed evidence is still
        refused, both by the rule immediately below.
        """

        for story in stories:
            for member in getattr(story, "members", ()):  # validated later
                raw_item_id = getattr(member, "raw_item_id", None)
                if raw_item_id is None:
                    continue
                row = connection.execute(
                    "SELECT substr(COALESCE(published_at, fetched_at), 1, 10) "
                    "AS day FROM raw_items WHERE id = ?",
                    (raw_item_id,),
                ).fetchone()
                if row is None:
                    continue  # the foreign key reports this one precisely.
                if str(row["day"]) != trading_day:
                    raise Phase0RunContextError(
                        f"raw item {raw_item_id} falls on {row['day']} but this "
                        f"reconciliation covers {trading_day}"
                    )
                cls._assert_raw_item_association(
                    connection, int(raw_item_id), ticker, operation="reconcile_stories"
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
                stored = self._stored_story_signature(connection, row)
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

    @staticmethod
    def _stored_columns(row: sqlite3.Row, columns: Sequence[str]) -> tuple[Any, ...]:
        """One stored row's owned columns, typed to compare with prepared."""

        stored: list[Any] = []
        for column in columns:
            value = row[column]
            if value is None:
                stored.append(None)
            elif column in _INTEGER_RECONCILED_COLUMNS:
                stored.append(int(value))
            elif isinstance(value, memoryview):
                stored.append(bytes(value))
            else:
                stored.append(value)
        return tuple(stored)

    @classmethod
    def _stored_story_signature(
        cls, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> tuple[Any, ...]:
        """The stored counterpart of :meth:`_story_signature`, read whole.

        The child relations are read back from the tables rather than from
        the ``member_ids`` summary column, because the summary is exactly
        the thing that stays true while the payload beneath it rots.
        """

        story_id = int(row["id"])
        members = tuple(
            (
                int(member["raw_item_id"]),
                int(member["position"]),
                member["outlet"],
                member["url"],
                member["canonical_url"],
                member["match_reason"],
                int(member["quarantined"]),
            )
            for member in connection.execute(
                "SELECT * FROM story_members WHERE story_id = ? ORDER BY raw_item_id",
                (story_id,),
            )
        )
        conflicts = tuple(
            (
                conflict["provider_namespace"],
                conflict["provider_item_id"],
                conflict["item_ids"],
                conflict["fields"],
            )
            for conflict in connection.execute(
                "SELECT * FROM story_provider_conflicts WHERE story_id = ? "
                "ORDER BY provider_namespace, provider_item_id",
                (story_id,),
            )
        )
        merges = tuple(
            (
                merge["left_story_key"],
                merge["right_story_key"],
                float(merge["similarity"]),
                merge["reason"],
            )
            for merge in connection.execute(
                "SELECT * FROM story_semantic_merges WHERE story_id = ? "
                "ORDER BY left_story_key, right_story_key",
                (story_id,),
            )
        )
        return (
            cls._stored_columns(row, STORY_RECONCILED_COLUMNS),
            None if row["canonical_item_id"] is None else int(row["canonical_item_id"]),
            row["invalidated_at"] is None,
            members,
            conflicts,
            merges,
        )

    def _insert_reconciled_story(
        self,
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        version: str,
        values: Mapping[str, Any],
    ) -> int:
        columns = {
            "ticker": ticker,
            "trading_day": day,
            "pipeline_version": version,
            "cluster_fingerprint": values["cluster_fingerprint"],
            **self._story_column_values(values),
            "updated_at": utc_now(),
        }
        names = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        cursor = connection.execute(
            f"INSERT INTO stories ({names}) VALUES ({placeholders})",
            tuple(columns.values()),
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
        columns = self._story_column_values(values)
        assignments = ", ".join(f"{column} = ?" for column in columns)
        connection.execute(
            f"UPDATE stories SET {assignments}, invalidated_at = NULL, "
            "updated_at = ? WHERE id = ?",
            (*columns.values(), utc_now(), story_id),
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

        As with :meth:`reconcile_stories`, normalization and validation run
        inside the run's own transaction so that a rejection is recorded as
        this run's failure.
        """

        with self._logged_mutation(
            run, operation="reconcile_themes", terminal=terminal
        ) as (connection, context):
            normalized_ticker = normalize_ticker(ticker)
            day = _normalize_day(trading_day)
            version = _require_text(pipeline_version, "pipeline_version")
            self._assert_run_partition(
                context,
                operation="reconcile_themes",
                ticker=normalized_ticker,
                trading_day=day,
                pipeline_version=version,
            )
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
                # Rewriting the day's coverage, exclusions, or theme-set
                # metadata is work this run did, whether or not any theme
                # moved with it; leaving it out logged a real write as an
                # idle replay.
                success=(
                    len(report.inserted)
                    + len(report.updated)
                    + len(report.changed_outputs)
                ),
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
        self._assert_themes_are_disjoint(prepared)
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

            theme_set_id, set_changed = self._upsert_theme_set(
                connection, normalized_ticker, day, version, set_values
            )
            changed_outputs: set[str] = {"theme_set"} if set_changed else set()

            # Coverage and exclusions are compared before they are rewritten,
            # for two reasons: an unchanged list must not be reported as a
            # change, and a changed one must not go unreported just because
            # it holds no themes.  The delete stays ahead of the theme loop
            # whenever it happens at all, so a story moving out of coverage
            # and into a theme does not trip the "already accounted for"
            # trigger on its way.
            stored_coverage = self._stored_coverage(connection, theme_set_id)
            incoming_coverage = self._incoming_coverage(coverage)
            rewrite_other = stored_coverage["other"] != incoming_coverage["other"]
            rewrite_excluded = (
                stored_coverage["excluded"] != incoming_coverage["excluded"]
            )
            if rewrite_other:
                changed_outputs.add("other_coverage")
                connection.execute(
                    "DELETE FROM theme_other_coverage WHERE theme_set_id = ?",
                    (theme_set_id,),
                )
            if rewrite_excluded:
                changed_outputs.add("excluded")
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

            if rewrite_other:
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
            if rewrite_excluded:
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
                changed_outputs=tuple(sorted(changed_outputs)),
            )

    @staticmethod
    def _assert_themes_are_disjoint(themes: Sequence[Mapping[str, Any]]) -> None:
        """No story may be claimed by two themes in one reconciliation.

        The accounting rule is that each canonical story in a partition is
        placed exactly once — in one theme, in Other Coverage, or in
        exclusions.  Coverage and exclusions were checked against the
        themes; the themes were never checked against each other, and
        :meth:`_prepare_coverage` flattens them into a *set*, so the very
        step that looks at every member is the step that discards how many
        themes claimed it.

        Nothing downstream caught it either.  A theme's citations must
        belong to its member stories and no two themes in a partition may
        cite the same raw item, which blocks the obvious case — two themes
        sharing a single-item story would have to share its citation — but
        a story with two members lets each theme cite a different one and
        every remaining rule is satisfied.

        Raised before any write, so a batch that is invalid anywhere
        writes nothing anywhere.
        """

        owners: dict[int, list[str]] = {}
        for values in themes:
            for story_id in values["story_ids"]:
                owners.setdefault(story_id, []).append(values["fingerprint"])
        shared = {
            story_id: sorted(fingerprints)
            for story_id, fingerprints in owners.items()
            if len(fingerprints) > 1
        }
        if shared:
            detail = "; ".join(
                f"story {story_id} in themes {fingerprints}"
                for story_id, fingerprints in sorted(shared.items())
            )
            raise Phase0ValidationError(
                f"a story belongs to exactly one theme, but {detail}"
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
    def _theme_set_column_values(values: Mapping[str, Any]) -> dict[str, Any]:
        """The exact ``theme_sets`` columns one reconciliation owns.

        Same rule as :meth:`_story_column_values` and
        :meth:`_theme_column_values`: this mapping drives the insert, the
        update, *and* the equality test, so a column the stage writes
        cannot become invisible to the next settlement's comparison.

        Absent on purpose: the partition identity (``ticker``,
        ``trading_day``, ``pipeline_version``), the surrogate ``id``, and
        ``updated_at`` — which is bookkeeping about the write rather than
        an output, and is only touched when one of these actually moves.
        """

        return {column: values[column] for column in THEME_SET_RECONCILED_COLUMNS}

    @classmethod
    def _upsert_theme_set(
        cls,
        connection: sqlite3.Connection,
        ticker: str,
        day: str,
        version: str,
        values: Mapping[str, Any],
    ) -> tuple[int, bool]:
        """Settle the theme-set row; report whether that changed anything.

        Returns the row id and whether this call wrote.  An identical
        replay writes nothing at all — not even ``updated_at`` — because
        "unchanged" has to mean the stored row already is what a
        settlement would produce, and a bumped timestamp would make that
        claim false the moment anyone compared two databases.
        """

        columns = cls._theme_set_column_values(values)
        stored = connection.execute(
            "SELECT * FROM theme_sets WHERE ticker = ? AND trading_day = ? "
            "AND pipeline_version = ?",
            (ticker, day, version),
        ).fetchone()
        if stored is not None:
            theme_set_id = int(stored["id"])
            if all(stored[column] == value for column, value in columns.items()):
                return theme_set_id, False
            assignments = ", ".join(f"{column} = ?" for column in columns)
            connection.execute(
                f"UPDATE theme_sets SET {assignments}, updated_at = ? WHERE id = ?",
                (*columns.values(), utc_now(), theme_set_id),
            )
            return theme_set_id, True

        names = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        cursor = connection.execute(
            f"INSERT INTO theme_sets (ticker, trading_day, pipeline_version, "
            f"{names}, updated_at) "
            f"VALUES (?, ?, ?, {placeholders}, ?)",
            (ticker, day, version, *columns.values(), utc_now()),
        )
        return int(cursor.lastrowid), True

    @staticmethod
    def _stored_coverage(
        connection: sqlite3.Connection, theme_set_id: int
    ) -> dict[str, list[tuple[Any, ...]]]:
        """The day's stored Other Coverage and exclusions, canonicalized.

        Both tables are *sets* of rows — neither has an inherent order, so
        the comparison is keyed by story and the rows are sorted by it.
        ``position`` is compared as a value because it is a persisted
        column that ranking reads back; the order the caller happened to
        list two entries in is not.
        """

        other = [
            (int(row["story_id"]), str(row["reason"]), int(row["position"]))
            for row in connection.execute(
                "SELECT story_id, reason, position FROM theme_other_coverage "
                "WHERE theme_set_id = ? ORDER BY story_id",
                (theme_set_id,),
            )
        ]
        excluded = [
            (int(row["story_id"]), str(row["reason"]))
            for row in connection.execute(
                "SELECT story_id, reason FROM theme_excluded_stories "
                "WHERE theme_set_id = ? ORDER BY story_id",
                (theme_set_id,),
            )
        ]
        return {"other": other, "excluded": excluded}

    @staticmethod
    def _incoming_coverage(
        coverage: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> dict[str, list[tuple[Any, ...]]]:
        """The same shape as :meth:`_stored_coverage`, from the payload."""

        return {
            "other": sorted(
                (
                    int(entry["story_id"]),
                    str(entry["reason"]),
                    int(entry["position"]),
                )
                for entry in coverage["other"]
            ),
            "excluded": sorted(
                (int(entry["story_id"]), str(entry["reason"]))
                for entry in coverage["excluded"]
            ),
        }

    @staticmethod
    def _theme_column_values(values: Mapping[str, Any]) -> dict[str, Any]:
        """The exact ``themes`` columns one reconciliation owns and writes.

        The theme counterpart of :meth:`_story_column_values`, with the
        same rule: this mapping drives the insert, the update, *and* the
        equality test, so a derived value cannot be written by a
        settlement and then be invisible to the next one.  The centroid
        and every salience component are derived outputs of the theme
        stage — a run that recomputes them has produced a different
        theme even when its label and membership are word-for-word the
        same.

        Absent on purpose: the partition identity (``ticker``,
        ``trading_day``, ``pipeline_version``, ``fingerprint``) and
        ``updated_at``.
        """

        return {
            "label": values["label"],
            "summary": values["summary"],
            # Sorted, so the denormalized copy of a citation *set* is the
            # same bytes whatever order the stage emitted it in.  Without
            # that this column would make equality order-sensitive, and
            # two identical theme sets would take the update path forever.
            "citations": _serialize_json(
                sorted(values["citation_ids"]), "citations", list
            ),
            "salience_rank": values["salience_rank"],
            "status": values["status"],
            "centroid": values["centroid"],
            "content_hash": values["content_hash"],
            "theme_key": values["theme_key"],
            "label_source": values["label_source"],
            "method": values["method"],
            "salience": values["salience"],
            "cohesion": values["cohesion"],
            "min_pairwise_cohesion": values["min_pairwise_cohesion"],
            "story_count": values["story_count"],
            "outlet_count": values["outlet_count"],
            "latest_published_at": values["latest_published_at"],
            "salience_story_component": values["salience_story_component"],
            "salience_outlet_component": values["salience_outlet_component"],
            "salience_recency_component": values["salience_recency_component"],
            "matched_previous_key": values["matched_previous_key"],
            "algorithm_version": values["algorithm_version"],
            "config_fingerprint": values["config_fingerprint"],
            "model_name": values["model_name"],
            "model_revision": values["model_revision"],
            "embedding_dimension": values["embedding_dimension"],
        }

    @classmethod
    def _theme_signature(cls, values: Mapping[str, Any]) -> tuple[Any, ...]:
        """Everything about a theme that one reconciliation persists.

        Every owned column plus both membership relations, canonically
        ordered so equivalent inputs compare equal regardless of the order
        the stage produced them in.
        """

        columns = cls._theme_column_values(values)
        return (
            tuple(columns[column] for column in THEME_RECONCILED_COLUMNS),
            tuple(sorted(values["story_ids"])),
            tuple(sorted(values["citation_ids"])),
        )

    @classmethod
    def _stored_theme_signature(
        cls, connection: sqlite3.Connection, row: sqlite3.Row
    ) -> tuple[Any, ...]:
        """The stored counterpart of :meth:`_theme_signature`."""

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
            cls._stored_columns(row, THEME_RECONCILED_COLUMNS),
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
        columns = {
            "ticker": ticker,
            "trading_day": day,
            "pipeline_version": version,
            "fingerprint": values["fingerprint"],
            **self._theme_column_values(values),
            "updated_at": utc_now(),
        }
        names = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        cursor = connection.execute(
            f"INSERT INTO themes ({names}) VALUES ({placeholders})",
            tuple(columns.values()),
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
        columns = self._theme_column_values(values)
        assignments = ", ".join(f"{column} = ?" for column in columns)
        connection.execute(
            f"UPDATE themes SET {assignments}, updated_at = ? WHERE id = ?",
            (*columns.values(), utc_now(), theme_id),
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
        """Atomically claim an unowned, retryable, or expired stage key.

        Every identity field is normalized and validated first, so a key
        this method creates is always one :meth:`stage_run` will accept.
        """

        if _require_int(lease_seconds, "lease_seconds") <= 0:
            raise Phase0ValidationError("lease_seconds must be positive")
        identity = self._stage_key_identity(
            stage=stage,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=pipeline_version,
            run_id=run_id,
        )
        stage = identity["stage"]
        day = identity["trading_day"]
        normalized_ticker = identity["ticker"]
        pipeline_version = identity["pipeline_version"]
        run_id = identity["run_id"]
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
        """Extend the caller's own lease; ``False`` when it no longer owns it.

        Only a lease that is still *live* can be extended.  The condition
        below is the exact complement of the one
        :meth:`claim_stage_key` reclaims under — that treats a key as
        available when ``lease_expires_at IS NULL OR lease_expires_at <=
        now`` — so at any instant a lease is renewable or reclaimable and
        never both.  Without that, an owner whose lease had already
        lapsed could push it forward and keep working while another
        worker was entitled to take it, and the two would then hold the
        same partition at once.  Expiry is the moment ownership ends, not
        a suggestion the previous owner may decline.
        """

        if _require_int(lease_seconds, "lease_seconds") <= 0:
            raise Phase0ValidationError("lease_seconds must be positive")
        identity = self._stage_key_identity(
            stage=stage,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=pipeline_version,
            run_id=run_id,
        )
        stage = identity["stage"]
        day = identity["trading_day"]
        normalized_ticker = identity["ticker"]
        pipeline_version = identity["pipeline_version"]
        run_id = identity["run_id"]
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
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?
                """,
                (
                    moment,
                    expires,
                    stage,
                    normalized_ticker,
                    day,
                    pipeline_version,
                    run_id,
                    moment,
                ),
            )
            return cursor.rowcount == 1

    def _complete_stage_key_unlogged(
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
        """Force a stage key to a terminal status with no run behind it.

        Private, and reachable only through
        :meth:`Phase0Admin.complete_stage_key`.  As a *public* method this
        was a hole with nothing subtle about it: claim a key, call this
        with ``status="success"``, and the ledger says the stage finished
        while no data moved and no ``run_log`` row exists.  A pipeline
        stage now reaches ``success`` exactly one way — a terminal logged
        mutation, which commits the data, the run log, and this transition
        together.
        """

        if status not in STAGE_KEY_STATUSES:
            raise Phase0ValidationError("invalid stage-key status")
        identity = self._stage_key_identity(
            stage=stage,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=pipeline_version,
            run_id=run_id,
        )
        stage = identity["stage"]
        day = identity["trading_day"]
        normalized_ticker = identity["ticker"]
        pipeline_version = identity["pipeline_version"]
        run_id = identity["run_id"]
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
        # `status` and `errors` describe the same outcome, and the line
        # above says how: an unstated status *means* degraded when there
        # are errors.  A caller that states `success` alongside errors is
        # therefore contradicting the module's own rule, and `degraded`
        # is the word this vocabulary has for "worked, with problems".
        # `StageRunContext._resolved_status` cannot produce the pairing,
        # so only the admin path could, and silently.
        if resolved_status == "success" and errors:
            raise Phase0ValidationError(
                f"run {run_id!r} stage {stage!r} is logged as 'success' with "
                f"{len(list(errors))} error(s); a run that recorded errors is "
                f"'degraded' or 'failed'"
            )
        if _require_int(duration_ms, "duration_ms") < 0:
            raise Phase0ValidationError("duration_ms cannot be negative")
        normalized_started = _normalize_datetime(started_at, "started_at")
        normalized_completed = _normalize_datetime(completed_at, "completed_at")
        if normalized_completed < normalized_started:
            raise Phase0ValidationError("completed_at cannot precede started_at")
        normalized_ticker = normalize_ticker(ticker, optional=True)
        day = _normalize_day(trading_day)
        version = _require_text(pipeline_version, "pipeline_version")
        self._assert_run_identity_partition(
            connection,
            run_id=_require_text(run_id, "run_id"),
            stage=_require_text(stage, "stage"),
            ticker=normalized_ticker,
            trading_day=day,
            pipeline_version=version,
        )
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

    #: The run-log columns that say *which partition a run belongs to*.
    #: They are settled by the first write of a ``(run_id, stage)`` and
    #: are never rewritten; everything else on the row is outcome.
    RUN_IDENTITY_COLUMNS: tuple[str, ...] = (
        "ticker",
        "trading_day",
        "pipeline_version",
    )

    @classmethod
    def _assert_run_identity_partition(
        cls,
        connection: sqlite3.Connection,
        *,
        run_id: str,
        stage: str,
        ticker: str | None,
        trading_day: str,
        pipeline_version: str,
    ) -> None:
        """``(run_id, stage)`` names exactly one partition, permanently.

        The row is an upsert keyed on ``(run_id, stage)``, and the
        conflict branch used to rewrite ``ticker`` while leaving
        ``trading_day`` and ``pipeline_version`` alone.  Either half is
        wrong in its own direction: reusing the identity under a second
        ticker silently relabelled the first run's row, and reusing it
        under a second day or pipeline version logged the new run under
        the old one's partition.  In both cases one run-log row ends up
        describing work no single run did.

        So the partition is settled by whoever writes first and is
        immutable thereafter.  Retrying or replaying the same identity in
        the *same* partition is untouched — that is the documented
        lifecycle, and it is the only thing this identity is for.
        """

        stored = connection.execute(
            "SELECT ticker, trading_day, pipeline_version FROM run_log "
            "WHERE run_id = ? AND stage = ?",
            (run_id, stage),
        ).fetchone()
        if stored is None:
            return
        incoming = (ticker, trading_day, pipeline_version)
        existing = (
            None if stored["ticker"] is None else str(stored["ticker"]),
            str(stored["trading_day"]),
            str(stored["pipeline_version"]),
        )
        if existing == incoming:
            return
        differences = ", ".join(
            f"{column} {was!r} -> {now!r}"
            for column, was, now in zip(cls.RUN_IDENTITY_COLUMNS, existing, incoming)
            if was != now
        )
        raise Phase0RunContextError(
            f"run {run_id!r} stage {stage!r} is already recorded in a "
            f"different partition ({differences}); a run identity names one "
            f"partition and cannot be reused for another"
        )

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
            # A key always names a ticker, so a run holding one is never
            # ticker-less: omitting it *adopts* the key's ticker rather
            # than meaning "any".  Read the other way round, an omitted
            # ticker used to skip the ticker comparison entirely, so an
            # NVDA lease opened a run with no ticker — and a ticker-less
            # run's partition checks pass for every ticker, which is an
            # NVDA key authorizing AMD work by saying nothing at all.
            if normalized_ticker is None:
                normalized_ticker = key["ticker"]
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
            if context.settled:
                # A terminal operation already committed the whole outcome:
                # data, final run log, and stage-key transition, in one
                # transaction.  Finalizing again here is what used to
                # rewrite a committed success and — when the second
                # ``_finish_stage_key`` found nothing to update — throw a
                # StageKeyError out of this ``finally`` in place of the
                # exception the caller was actually raising.
                pass
            else:
                if failure is not None:
                    status = "failed"
                elif key is not None:
                    # The stage held a lease and never declared completion.
                    # Retryable, not successful: saying otherwise here would
                    # be the separate success-marking this design removed.
                    status = "degraded"
                else:
                    status = context._resolved_status()
                self._settle_run(
                    context,
                    status,
                    key_status="success" if status == "success" else "failed",
                    settled_state=RUN_STATE_CLOSED_WITHOUT_TERMINAL,
                    # An exception from the stage body is the one the caller
                    # needs; bookkeeping must not replace it on the way out.
                    suppress_errors=failure is not None,
                )

    def _settle_run(
        self,
        context: StageRunContext,
        status: str,
        *,
        key_status: str,
        settled_state: str,
        suppress_errors: bool,
    ) -> str:
        """Commit a run's single final outcome, then record which one.

        The run-log row and the key release belong in one transaction, but
        the run log matters more: when the key is gone — reclaimed,
        already completed — that is usually *why* this run is ending, and
        losing the record of it would hide the failure entirely.  So a lost
        key falls back to writing the run log alone.

        The context moves only *after* a commit returns.  Nothing here
        marks a run settled on the strength of statements that have not
        reached the disk yet.

        Returns the state the context ended in.  ``suppress_errors`` is for
        teardown after an exception, where raising from a ``finally`` would
        replace the real exception.
        """

        try:
            observed = self._write_settlement(
                context, status, key_status, include_key=True
            )
        except Exception:
            if not suppress_errors:
                raise
            try:
                observed = self._write_settlement(
                    context, status, key_status, include_key=False
                )
            except Exception:  # noqa: BLE001 - the caller's exception wins
                # Nothing durable was written and nothing may claim
                # otherwise.  The stage key is left exactly as it was, so
                # the lease expires and ordinary recovery reclaims it —
                # which is also what a crash here would have looked like.
                context._transition(RUN_STATE_SETTLEMENT_FAILED)
                return RUN_STATE_SETTLEMENT_FAILED
        if observed == "already_succeeded":
            # The operation's own commit *did* land, and only reporting it
            # failed.  Overwriting a durable success with a failure would
            # be the worse of the two lies.
            context._transition(RUN_STATE_TERMINAL_SUCCEEDED)
            return RUN_STATE_TERMINAL_SUCCEEDED
        context._transition(settled_state)
        return settled_state

    def _write_settlement(
        self,
        context: StageRunContext,
        status: str,
        key_status: str,
        *,
        include_key: bool,
    ) -> str:
        """One transaction: the final run log, and optionally the key.

        Conditional as well as atomic.  A commit that raises has not
        necessarily failed to land — an I/O error can be reported after
        the transaction is durable — so this looks at what is actually on
        disk first, and refuses to overwrite a committed success with a
        failure it inferred from an exception.
        """

        connection = self._open_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if status != "success" and self._already_succeeded(connection, context):
                connection.rollback()
                return "already_succeeded"
            self._write_final_run_log(connection, context, status)
            if include_key and context._stage_key is not None:
                self._finish_stage_key(connection, context, key_status)
            connection.commit()
            return "settled"
        except BaseException:
            with contextlib.suppress(Exception):
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _already_succeeded(
        connection: sqlite3.Connection, context: StageRunContext
    ) -> bool:
        """True when this run's success is already on disk.

        Read on the settlement's own transaction, so it sees the committed
        state rather than whatever the failed transaction thought.
        """

        row = connection.execute(
            "SELECT status FROM run_log WHERE run_id = ? AND stage = ?",
            (context.run_id, context.stage),
        ).fetchone()
        if row is None or str(row["status"]) != "success":
            return False
        key = context._stage_key
        if key is None:
            return True
        key_row = connection.execute(
            """
            SELECT run_id, status FROM pipeline_stage_keys
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
        return (
            key_row is not None
            and str(key_row["run_id"]) == context.run_id
            and str(key_row["status"]) == "success"
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
    def _stage_key_identity(
        *,
        stage: Any,
        ticker: Any,
        trading_day: Any,
        pipeline_version: Any,
        run_id: Any,
    ) -> dict[str, str]:
        """The five immutable fields of a stage-key lifecycle, normalized.

        One definition, shared by every entrypoint that creates, renews,
        or settles a key, using exactly the helpers ``stage_run`` uses.
        It had been three definitions: the key methods normalized only
        ``ticker`` and ``trading_day`` and passed ``stage``,
        ``pipeline_version``, and ``run_id`` through untouched, while
        ``stage_run`` required all five to be non-blank and stripped.

        The gap was not cosmetic.  A claim with ``run_id=""`` was written
        as ``running`` and then could not be settled by anybody: the
        identity that owned the lease was one ``stage_run`` refuses, so
        the partition stayed locked until the lease expired.  A padded
        ``stage`` did the same thing more quietly — the key was stored
        under a name no normalized lookup would ever match.
        """

        return {
            "stage": _require_text(stage, "stage"),
            "ticker": normalize_ticker(ticker),
            "trading_day": _normalize_day(trading_day),
            "pipeline_version": _require_text(pipeline_version, "pipeline_version"),
            "run_id": _require_text(run_id, "run_id"),
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

        Every field is compared, ``ticker`` included.  It reaches here
        already resolved: :meth:`stage_run` adopts the key's ticker when
        the caller omitted one, so ``None`` never means "unconstrained".
        """

        expected = {
            "stage": stage,
            "ticker": ticker,
            "trading_day": trading_day,
            "pipeline_version": pipeline_version,
        }
        for field, want in expected.items():
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
        self._assert_lease_held(connection, run, operation=operation)
        return run

    @staticmethod
    def _assert_run_partition(
        run: StageRunContext,
        *,
        operation: str,
        ticker: str | None = None,
        trading_day: str | None = None,
        pipeline_version: str | None = None,
    ) -> None:
        """The operation's partition must be the one the run covers.

        Called from inside :meth:`_logged_mutation`, on the authoritative
        transaction, so a rejection here rolls the operation back and is
        recorded as this run's failure — rather than being raised before
        the run took responsibility, where a caller could swallow it and
        let the stage go on to report success.
        """

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
        terminal: bool = False,
    ) -> Iterator[tuple[sqlite3.Connection, StageRunContext]]:
        """One transaction holding authorization, mutation, log, and release.

        The transaction is managed by hand rather than by ``_connect``, and
        that is the whole point of this shape.  A context manager commits
        on the way out, *after* the body has run — so marking the run
        successful in the body meant marking it successful before the
        commit, and an injected commit failure left the object saying
        "terminal_succeeded" over a database that had rolled the data,
        the run log, and the key release all back.

        The order:

        1. open a private writable connection;
        2. ``BEGIN IMMEDIATE`` — take the write lock;
        3. authorize the run, its owner, and its lease *on that connection*;
        4. validate the payload (the caller's body does this);
        5. mutate;
        6. write the derived counts and the run-log row;
        7. when ``terminal``, transition the stage key and release the
           lease — still inside this transaction;
        8. ``commit()``;
        9. **only now**, mark the context ``TERMINAL_SUCCEEDED``.

        A concurrent reclaimer blocks at step 2 and cannot get in anywhere
        between steps 3 and 8.  Because step 7 is inside, there is no
        committed state in which the data says success and the stage key is
        still sitting there reclaimable as ``running``.

        A non-terminal operation never records ``success``: its interim
        run-log row is written as ``degraded`` until some operation
        declares the stage finished.  It, too, changes nothing in memory
        until its own commit returns.

        If anything raises — including ``commit()`` — the data rolls back,
        the failure settlement runs in its own transaction, the context
        becomes ``TERMINAL_FAILED`` only once *that* commits, and the
        original exception propagates untouched.

        A run whose outcome is already settled is refused *before* any
        transaction opens.  That ordering matters too: the previous
        version let a second call reach the failure path, which rewrote a
        committed ``success`` run log to ``failed`` while the stage key it
        described stayed ``success``.
        """

        # ``getattr`` because a hand-made ``object.__new__`` shell has no
        # state at all; it is not terminal, and ``_authorize_run`` below is
        # what refuses it.
        if (
            isinstance(run, StageRunContext)
            and getattr(run, "_state", RUN_STATE_ACTIVE) in _TERMINAL_RUN_STATES
        ):
            # Before opening a connection, before ``_authorize_run``, and
            # before the failure path below: a run that has already settled
            # must not reach code that writes an outcome, because that code
            # would overwrite the one it has.
            raise Phase0RunContextError(
                f"{operation}: this run is already {run.state} and its outcome "
                "is written; open a new stage_run to do more work"
            )
        connection: sqlite3.Connection | None = None
        context: StageRunContext | None = None
        committed = False
        try:
            connection = self._open_connection()
            connection.execute("BEGIN IMMEDIATE")
            context = self._authorize_run(connection, run, operation=operation)
            yield connection, context
            if terminal:
                status = context._resolved_status()
                self._write_final_run_log(connection, context, status)
                if context._stage_key is not None:
                    self._finish_stage_key(connection, context, status)
            else:
                # Never "success" before the stage says it is finished.
                self._write_final_run_log(connection, context, "degraded")
            connection.commit()
            committed = True
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if connection is not None:
                # Best effort, and deliberately silent: a rollback that
                # cannot run must not become the exception the caller sees.
                with contextlib.suppress(Exception):
                    connection.rollback()
            # Only a run that was genuinely authorized gets a failed record;
            # a rejected forgery has no run to write against, and a run that
            # is already settled must not have its outcome rewritten.
            if (
                isinstance(run, StageRunContext)
                and self._is_active_run(run)
                and not run.settled
            ):
                run._record_outcome(failure=1)
                run._record_error(
                    {"operation": operation, "error": f"{type(exc).__name__}: {exc}"}
                )
                # Settles the run in its own transaction and only then moves
                # the context — and never at the cost of the exception the
                # caller is waiting for: the failure below *is* the news.
                self._settle_run(
                    run,
                    "failed",
                    key_status="failed",
                    settled_state=RUN_STATE_TERMINAL_FAILED,
                    suppress_errors=True,
                )
            raise
        finally:
            if connection is not None:
                connection.close()
        if terminal and committed and context is not None:
            # Durable first, then said out loud.
            context._transition(RUN_STATE_TERMINAL_SUCCEEDED)

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
        run's trading day — and so must the day of any *stored* item the
        payload duplicates.  A duplicate ``(source, canonical_url)`` does
        not create a row, it names one, and this path then writes that
        row's ticker associations and candidate reasons.  Validating only
        the incoming payload let a run for day D mutate evidence that
        belongs to D-1 while the run log recorded the work as D's.

        Preparation and partition checks both run *inside* the logged
        mutation.  Anything that can reject the batch has to, or a caller
        that catches the rejection outside leaves the stage free to go on
        and report success for work that never happened.
        """

        with self._logged_mutation(
            run, operation="ingest_raw_items", terminal=terminal
        ) as (connection, context):
            prepared = [self._prepare_raw_item(item) for item in items]
            self._assert_raw_item_partition(connection, prepared, context)
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
        connection: sqlite3.Connection,
        prepared: Sequence[Mapping[str, Any]],
        run: StageRunContext,
    ) -> None:
        """Reject a mixed or foreign-partition ingestion batch before writing.

        Reads the *normalized* candidates, so a bare ``"AMD"`` string is
        checked exactly like ``{"ticker": "AMD"}``.  When this walked the
        raw values it understood only the mapping form, and the string form
        went straight through into an NVDA run.

        Every item in the batch is checked before any of them is written,
        so one bad duplicate rolls the whole batch back rather than
        leaving the items ahead of it persisted.
        """

        for position, values in enumerate(prepared):
            asserted = {values["ticker"]} | set(values["tickers"])
            asserted |= {
                candidate["ticker"] for candidate in values["candidate_tickers"]
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

            # A payload that duplicates `(source, canonical_url)` does not
            # create a row — it *names* one, and that row has its own
            # effective day, which need not be the day the payload claims.
            # Checking only the payload let a run for D reach back and add
            # associations and candidate reasons to evidence belonging to
            # D-1, while the run log recorded the work as D's.
            #
            # The stored day is derived here exactly as `raw_items_for_day`
            # derives it, in SQL, so "which day is this evidence on" has one
            # definition and not two that can drift apart.
            stored = connection.execute(
                "SELECT id, substr(COALESCE(published_at, fetched_at), 1, 10) "
                "AS day FROM raw_items WHERE source = ? AND canonical_url = ?",
                (values["source"], values["canonical_url"]),
            ).fetchone()
            if stored is not None and str(stored["day"]) != run.trading_day:
                raise Phase0RunContextError(
                    f"ingest_raw_items item {position} duplicates stored raw "
                    f"item {int(stored['id'])}, which belongs to "
                    f"{stored['day']}, but the run covers {run.trading_day}; "
                    f"a run may not mutate another day's evidence"
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

        A raw-item source additionally needs an *explicit* association with
        the run's ticker; see :meth:`_assert_raw_item_association`.
        Validation and partition checks both run inside the transaction the
        run owns, so a rejection is recorded as this run's failure.
        """

        with self._logged_mutation(
            run, operation="persist_embeddings", terminal=terminal
        ) as (connection, context):
            prepared = [validate_embedding(embedding) for embedding in embeddings]
            self._assert_embedding_partition(connection, prepared, context)
            for values in prepared:
                self._write_embedding(connection, values)
            context._record_outcome(success=len(prepared))
            context._merge_counts({"embeddings_written": len(prepared)})
            return len(prepared)

    @classmethod
    def _resolve_embedding_source(
        cls,
        connection: sqlite3.Connection,
        source_kind: str,
        source_id: str,
    ) -> list[sqlite3.Row]:
        """Every row an embedding source identity names, per the schema.

        ``embeddings.source_id`` is text, and migration 007's ownership
        triggers say exactly which text is allowed to name a row: a raw
        item by id, a story by id *or* cluster fingerprint, a theme by id
        *or* fingerprint *or* theme key.  That is the durable contract, so
        it is the contract resolved here — a partition check that only
        understood integers rejected identities the database itself
        accepts, and rejected them as "does not exist".

        The comparisons mirror the triggers' ``CAST(id AS TEXT) = …``
        exactly rather than relying on SQLite's integer affinity, so
        ``'007'`` does not resolve to story 7 here and then be refused by
        the trigger a moment later.

        Every match is returned, not just the first: fingerprints and
        theme keys are unique only *within* a ticker-day partition, so an
        identity naming more than one row is genuinely ambiguous and the
        caller fails closed on it.
        """

        queries = {
            "raw_item": (
                "SELECT id, ticker, "
                "substr(COALESCE(published_at, fetched_at), 1, 10) AS day, "
                "NULL AS pipeline_version FROM raw_items "
                "WHERE CAST(id AS TEXT) = ?",
                1,
            ),
            "story": (
                "SELECT id, ticker, trading_day AS day, pipeline_version "
                "FROM stories "
                "WHERE CAST(id AS TEXT) = ? OR cluster_fingerprint = ?",
                2,
            ),
            "theme": (
                "SELECT id, ticker, trading_day AS day, pipeline_version "
                "FROM themes "
                "WHERE CAST(id AS TEXT) = ? OR fingerprint = ? OR theme_key = ?",
                3,
            ),
        }
        sql, arity = queries[source_kind]
        return list(connection.execute(sql, (source_id,) * arity))

    @classmethod
    def _assert_embedding_partition(
        cls,
        connection: sqlite3.Connection,
        prepared: Sequence[Mapping[str, Any]],
        run: StageRunContext,
    ) -> None:
        for position, values in enumerate(prepared):
            matches = cls._resolve_embedding_source(
                connection, values["source_kind"], values["source_id"]
            )
            if not matches:
                raise Phase0RunContextError(
                    f"persist_embeddings source {position} "
                    f"({values['source_kind']} {values['source_id']}) does not exist"
                )
            if len(matches) > 1:
                # One identity, several rows it could mean.  The embedding
                # is keyed globally by (source_kind, source_id), so there
                # is no partition this vector could honestly belong to.
                raise Phase0RunContextError(
                    f"persist_embeddings source {position} "
                    f"({values['source_kind']} {values['source_id']}) is ambiguous: "
                    f"it names {values['source_kind']}s "
                    f"{sorted(int(match['id']) for match in matches)}"
                )
            row = matches[0]
            # A story or a theme belongs to exactly one partition, so its
            # ticker column *is* exclusive.  A raw item's is not: it is
            # the primary attribution beside an association table that may
            # legitimately name several tickers, so membership for one is
            # settled below by ``_assert_raw_item_association`` rather than
            # by refusing anything whose primary ticker reads differently.
            exclusive = values["source_kind"] != "raw_item"
            if exclusive and row["ticker"] is not None and run.ticker is not None:
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
            if values["source_kind"] == "raw_item" and run.ticker is not None:
                # A ticker-scoped run embeds *this ticker's* evidence.  An
                # unattributed item is storable, but it is not NVDA's until
                # something says so; borrowing an NVDA run to embed it is
                # how it would silently become NVDA's.
                cls._assert_raw_item_association(
                    connection,
                    int(row["id"]),
                    run.ticker,
                    operation="persist_embeddings",
                )

    def record_source_state(
        self,
        source: str,
        *,
        run: Any,
        etag: str | None = None,
        last_modified: str | None = None,
        checked_at: str | None = None,
        successful: bool | None = None,
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

        **The outcome is stated once.** ``status`` is the richer of the
        two — ``partial``, ``empty``, and ``unknown`` have no boolean
        spelling — so when it is given it decides, and ``successful``
        becomes a claim about the same thing that must agree.  Stating
        both and disagreeing is refused.  Whichever way it resolves, the
        stored status, the ``last_success_at`` stamp, the failure
        counter, and this run's own counters all come from that one
        answer, so the feed's record and the run that wrote it can no
        longer say opposite things.

        Omitting both still means success, as it always has.

        Both the timestamp normalization and that check happen inside the
        logged mutation, so rejecting either one is this run's recorded
        failure rather than an exception a caller can catch and walk away
        from.
        """

        with self._logged_mutation(
            run, operation="record_source_state", terminal=terminal
        ) as (connection, context):
            moment = _normalize_datetime(checked_at or utc_now(), "checked_at")
            if checked_at is not None and moment[:10] != context.trading_day:
                raise Phase0RunContextError(
                    f"record_source_state checked {moment[:10]} but the run covers "
                    f"{context.trading_day}"
                )
            state = {
                "source": source,
                "etag": etag,
                "last_modified": last_modified,
                "checked_at": moment,
                # Passed through as given, `None` included: the default
                # for an unstated outcome lives in
                # `validate_source_state`, so every entrypoint inherits
                # the same one instead of each patching it locally.
                "successful": successful,
                "metadata": metadata or {},
                "status": status,
                "error": error,
                "retry_after": retry_after,
            }
            persisted = self._set_source_state(connection, state)
            # Derived from what was actually stored, not from a second
            # copy of the question: `failed` is 1 exactly when the
            # resolved status is one the schema does not count as a
            # successful fetch.
            succeeded = not persisted["failed"]
            context._record_outcome(
                success=1 if succeeded else 0,
                partial=0 if succeeded else 1,
            )
            context._merge_counts(
                {
                    "source_states_recorded": 1,
                    "source_state_status": persisted["status"],
                }
            )

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
    "DEFAULT_CANDIDATE_REASON",
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
    "Phase0Reader",
    "Phase0Repository",
    "Phase0RunContextError",
    "Phase0ValidationError",
    "ProviderConflictRecord",
    "RAW_ITEM_ASSOCIATION_TABLE",
    "ReconciliationReport",
    "RUN_STATES",
    "RUN_STATE_ACTIVE",
    "RUN_STATE_CLOSED_WITHOUT_TERMINAL",
    "RUN_STATE_TERMINAL_FAILED",
    "RUN_STATE_SETTLEMENT_FAILED",
    "RUN_STATE_TERMINAL_SUCCEEDED",
    "RUN_STATUSES",
    "SECRET_KEY_PATTERN",
    "SOURCE_STATE_STATUSES",
    "STORY_RECONCILED_COLUMNS",
    "SUPPORTED_TICKERS",
    "STAGE_KEY_STATUSES",
    "SemanticMergeRecord",
    "StageKeyError",
    "StageRunContext",
    "StageRunRecorder",
    "StoryMemberRecord",
    "StoryRecord",
    "THEME_RECONCILED_COLUMNS",
    "THEME_SET_RECONCILED_COLUMNS",
    "THEME_STATUSES",
    "TICKER_UNIVERSE",
    "ThemeRecord",
    "ThemeSetRecord",
    "UnsupportedTickerError",
    "normalize_candidate_tickers",
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

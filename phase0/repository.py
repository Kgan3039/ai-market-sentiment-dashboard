"""SQLite persistence shared by Phase 0 pipeline stages."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "phase0.sqlite3"
MIGRATIONS_PATH = Path(__file__).with_name("migrations")
RUN_STATUSES = {"success", "degraded", "failed"}
STAGE_KEY_STATUSES = {"success", "degraded", "failed"}
THEME_STATUSES = {"pending", "ready", "degraded", "failed"}
SECRET_KEY_PATTERN = re.compile(
    r"(authorization|cookie|password|secret|token|api[_-]?key)", re.IGNORECASE
)
SECRET_VALUE_PATTERN = re.compile(
    r"(?i)\b(authorization|password|secret|token|api[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;&]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_datetime(
    value: str | datetime | None, field: str, *, optional: bool = False
) -> str | None:
    if value in (None, ""):
        if optional:
            return None
        raise ValueError(f"{field} is required")
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_day(value: str | date, field: str = "trading_day") -> str:
    if isinstance(value, datetime):
        raise ValueError(f"{field} must be a date, not a datetime")
    try:
        parsed = value if isinstance(value, date) else date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must use YYYY-MM-DD format") from exc
    return parsed.isoformat()


def _serialize_json(value: Any, field: str, expected_type: type) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must contain valid JSON") from exc
    else:
        parsed = value
    if not isinstance(parsed, expected_type):
        raise ValueError(f"{field} must be a {expected_type.__name__}")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if SECRET_KEY_PATTERN.search(str(key))
                else _redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE_PATTERN.sub(r"\1\2[REDACTED]", value)
    return value


def _migration_statements(sql: str) -> Iterator[str]:
    pending = ""
    for line in sql.splitlines():
        pending += f"{line}\n"
        if sqlite3.complete_statement(pending):
            statement = pending.strip()
            if statement:
                yield statement
            pending = ""
    if pending.strip():
        raise ValueError("migration contains an incomplete SQL statement")


@dataclass(frozen=True)
class InsertResult:
    item_id: int
    inserted: bool


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

    def _open_connection(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = self._open_connection()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        """Apply each migration atomically and advance its version on success."""
        connection = self._open_connection()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            for migration in sorted(self.migrations_path.glob("*.sql")):
                version = int(migration.name.split("_", 1)[0])
                if version <= current:
                    continue
                statements = list(
                    _migration_statements(migration.read_text(encoding="utf-8"))
                )
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    for statement in statements:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {version}")
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
                current = version
        finally:
            connection.close()

    def _prepare_raw_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        source = str(item.get("source") or "").strip()
        canonical_url = str(item.get("canonical_url") or "").strip()
        if not source or not canonical_url:
            raise ValueError("raw item requires source and canonical_url")
        ingest_status = str(item.get("ingest_status") or "valid")
        if ingest_status not in {"valid", "invalid", "ambiguous"}:
            raise ValueError("invalid raw-item ingest_status")
        title = str(item.get("title") or "").strip() or None
        url = str(item.get("url") or "").strip() or None
        if ingest_status == "valid" and (title is None or url is None):
            raise ValueError("valid raw items require title and url")
        raw_payload = item.get("raw_json", item)
        raw_json = _serialize_json(raw_payload, "raw_json", dict)
        validation_errors = _serialize_json(
            list(item.get("validation_errors") or []), "validation_errors", list
        )
        return {
            "source": source,
            "ticker": str(item["ticker"]).upper() if item.get("ticker") else None,
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
                    str(ticker).upper()
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
        for ticker in tickers:
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
                ticker = str(candidate.get("ticker") or "").upper()
                reason = str(candidate.get("reason") or "relevance_match")
            else:
                ticker = str(candidate).upper()
                reason = "relevance_match"
            if ticker:
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

    def insert_raw_item(self, item: Mapping[str, Any]) -> InsertResult:
        return self.insert_raw_items([item])[0]

    def insert_raw_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        source_state: Mapping[str, Any] | None = None,
    ) -> list[InsertResult]:
        """Persist a batch and optional source state in one transaction."""
        prepared = [self._prepare_raw_item(item) for item in items]
        with self.connect() as connection:
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
            query += " AND (raw_items.ticker = ? OR raw_item_tickers.ticker = ?)"
            parameters.extend([ticker.upper(), ticker.upper()])
        query += " ORDER BY COALESCE(published_at, fetched_at), raw_items.id"
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def update_raw_item_ticker(self, item_id: int, ticker: str | None) -> None:
        normalized = ticker.upper() if ticker else None
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE raw_items SET ticker = ? WHERE id = ?", (normalized, item_id)
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown raw item")
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
        with self.connect() as connection:
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

    def source_state(self, source: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_state WHERE source = ?", (source,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    @staticmethod
    def _prepare_source_state(state: Mapping[str, Any]) -> dict[str, Any]:
        checked_at = _normalize_datetime(state.get("checked_at"), "checked_at")
        successful = bool(state.get("successful"))
        return {
            "source": str(state.get("source") or "").strip(),
            "etag": state.get("etag"),
            "last_modified": state.get("last_modified"),
            "checked_at": checked_at,
            "success_at": checked_at if successful else None,
            "metadata": _serialize_json(
                dict(state.get("metadata") or {}), "source metadata", dict
            ),
        }

    @classmethod
    def _set_source_state(
        cls, connection: sqlite3.Connection, state: Mapping[str, Any]
    ) -> None:
        values = cls._prepare_source_state(state)
        if not values["source"]:
            raise ValueError("source state requires source")
        connection.execute(
            """
            INSERT INTO source_state (
                source, etag, last_modified, last_checked_at,
                last_success_at, metadata
            ) VALUES (
                :source, :etag, :last_modified, :checked_at,
                :success_at, :metadata
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
                metadata = excluded.metadata
            """,
            values,
        )

    def set_source_state(
        self,
        source: str,
        *,
        etag: str | None,
        last_modified: str | None,
        checked_at: str,
        successful: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self.connect() as connection:
            self._set_source_state(
                connection,
                {
                    "source": source,
                    "etag": etag,
                    "last_modified": last_modified,
                    "checked_at": checked_at,
                    "successful": successful,
                    "metadata": metadata or {},
                },
            )

    def insert_story(
        self,
        *,
        ticker: str,
        trading_day: str | date,
        canonical_title: str,
        member_ids: Sequence[int],
        embedding: bytes | None = None,
        outlet_count: int = 1,
    ) -> int:
        if not canonical_title.strip() or not member_ids:
            raise ValueError("stories require a title and at least one member")
        day = _normalize_day(trading_day)
        members = [int(item_id) for item_id in dict.fromkeys(member_ids)]
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO stories (
                    ticker, trading_day, canonical_title, embedding,
                    outlet_count, member_ids
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker.upper(),
                    day,
                    canonical_title.strip(),
                    embedding,
                    int(outlet_count),
                    _serialize_json(members, "member_ids", list),
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

    def insert_theme(
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
            raise ValueError("invalid theme status")
        day = _normalize_day(trading_day)
        stories = [int(value) for value in dict.fromkeys(story_ids)]
        citations = [int(value) for value in dict.fromkeys(citation_ids)]
        if not label.strip() or not stories:
            raise ValueError("themes require a label and at least one story")
        with self.connect() as connection:
            story_rows = connection.execute(
                f"""
                SELECT id, ticker, trading_day FROM stories
                WHERE id IN ({",".join("?" for _ in stories)})
                """,
                stories,
            ).fetchall()
            if len(story_rows) != len(stories) or any(
                row["ticker"] != ticker.upper() or row["trading_day"] != day
                for row in story_rows
            ):
                raise ValueError("theme stories must exist for the same ticker/day")
            member_rows = connection.execute(
                f"""
                SELECT DISTINCT raw_item_id FROM story_members
                WHERE story_id IN ({",".join("?" for _ in stories)})
                """,
                stories,
            ).fetchall()
            member_ids = {int(row["raw_item_id"]) for row in member_rows}
            if not set(citations).issubset(member_ids):
                raise ValueError("theme citations must reference member raw items")
            cursor = connection.execute(
                """
                INSERT INTO themes (
                    ticker, trading_day, label, summary, citations,
                    salience_rank, status, centroid, content_hash, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker.upper(),
                    day,
                    label.strip(),
                    summary,
                    _serialize_json(citations, "citations", list),
                    int(salience_rank),
                    status,
                    centroid,
                    content_hash,
                    pipeline_version,
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

    def insert_eval_label(
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
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO eval_labels (
                    label_type, item_a_id, item_b_id, reviewer,
                    label, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    label_type,
                    int(item_a_id),
                    int(item_b_id),
                    reviewer,
                    label,
                    notes,
                    _normalize_datetime(created_at or utc_now(), "created_at"),
                ),
            )
            return int(cursor.lastrowid)

    def claim_stage_key(
        self,
        *,
        stage: str,
        ticker: str,
        trading_day: str,
        pipeline_version: str,
        run_id: str,
    ) -> bool:
        """Atomically claim an unowned or retryable stage key."""
        day = _normalize_day(trading_day)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pipeline_stage_keys (
                    stage, ticker, trading_day, pipeline_version,
                    status, run_id, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?)
                ON CONFLICT(stage, ticker, trading_day, pipeline_version)
                DO UPDATE SET
                    status = 'running',
                    run_id = excluded.run_id,
                    updated_at = excluded.updated_at
                WHERE pipeline_stage_keys.status IN ('failed', 'degraded')
                """,
                (
                    stage,
                    ticker.upper(),
                    day,
                    pipeline_version,
                    run_id,
                    utc_now(),
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
    ) -> None:
        if status not in STAGE_KEY_STATUSES:
            raise ValueError("invalid stage-key status")
        day = _normalize_day(trading_day)
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_stage_keys
                SET status = ?, updated_at = ?
                WHERE stage = ? AND ticker = ? AND trading_day = ?
                    AND pipeline_version = ? AND run_id = ? AND status = 'running'
                """,
                (
                    status,
                    utc_now(),
                    stage,
                    ticker.upper(),
                    day,
                    pipeline_version,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("stage key was not claimed by this run")

    def clear_derived_for_day(self, trading_day: str | date) -> None:
        """Delete derived rows and their idempotency keys, retaining raw input."""
        day = _normalize_day(trading_day)
        with self.connect() as connection:
            connection.execute("DELETE FROM themes WHERE trading_day = ?", (day,))
            connection.execute("DELETE FROM stories WHERE trading_day = ?", (day,))
            connection.execute(
                "DELETE FROM pipeline_stage_keys WHERE trading_day = ?", (day,)
            )

    def log_stage(
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
    ) -> None:
        resolved_status = status or ("degraded" if errors else "success")
        if resolved_status not in RUN_STATUSES:
            raise ValueError("invalid run status")
        if int(duration_ms) < 0:
            raise ValueError("duration_ms cannot be negative")
        normalized_started = _normalize_datetime(started_at, "started_at")
        normalized_completed = _normalize_datetime(completed_at, "completed_at")
        if normalized_completed < normalized_started:
            raise ValueError("completed_at cannot precede started_at")
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO run_log (
                    run_id, stage, counts, duration_ms, errors, started_at,
                    completed_at, status, trading_day, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, stage) DO UPDATE SET
                    counts = excluded.counts,
                    duration_ms = excluded.duration_ms,
                    errors = excluded.errors,
                    completed_at = excluded.completed_at,
                    status = excluded.status
                """,
                (
                    run_id,
                    stage,
                    _serialize_json(dict(counts), "counts", dict),
                    int(duration_ms),
                    _serialize_json(_redact_secrets(list(errors)), "errors", list),
                    normalized_started,
                    normalized_completed,
                    resolved_status,
                    _normalize_day(trading_day),
                    pipeline_version,
                ),
            )

    def latest_stage_status(self) -> list[dict[str, Any]]:
        with self.connect() as connection:
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
        results = []
        for row in rows:
            result = dict(row)
            result["counts"] = json.loads(result["counts"])
            result["errors"] = json.loads(result["errors"])
            results.append(result)
        return results

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
        allowed = {
            "raw_items",
            "raw_item_tickers",
            "raw_item_candidates",
            "stories",
            "story_members",
            "themes",
            "theme_stories",
            "theme_citations",
            "run_log",
            "eval_labels",
            "source_state",
            "pipeline_stage_keys",
        }
        if table not in allowed:
            raise ValueError("unsupported table")
        with self.connect() as connection:
            return int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )

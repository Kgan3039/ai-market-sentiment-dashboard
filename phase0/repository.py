"""SQLite persistence shared by Phase 0 pipeline stages."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parents[1] / "data" / "phase0.sqlite3"
MIGRATIONS_PATH = Path(__file__).with_name("migrations")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class InsertResult:
    item_id: int
    inserted: bool


class Phase0Repository:
    """Small repository layer with one connection per operation.

    WAL mode and a busy timeout make the database safe for the scheduled writer
    and FastAPI readers without introducing a service or connection pool.
    """

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        """Apply ordered, idempotent SQL migrations."""
        with self.connect() as connection:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            for migration in sorted(MIGRATIONS_PATH.glob("*.sql")):
                version = int(migration.name.split("_", 1)[0])
                if version > current:
                    connection.executescript(migration.read_text(encoding="utf-8"))
                    current = version

    def insert_raw_item(self, item: Mapping[str, Any]) -> InsertResult:
        required = ("source", "title", "url", "canonical_url")
        missing = [
            field for field in required if not str(item.get(field) or "").strip()
        ]
        if missing:
            raise ValueError(f"raw item missing required fields: {', '.join(missing)}")

        raw_payload = item.get("raw_json", item)
        raw_json = (
            raw_payload
            if isinstance(raw_payload, str)
            else json.dumps(raw_payload, sort_keys=True, default=str)
        )
        values = {
            "source": str(item["source"]).strip(),
            "ticker": str(item["ticker"]).upper() if item.get("ticker") else None,
            "title": str(item["title"]).strip(),
            "description": str(item.get("description") or "").strip(),
            "url": str(item["url"]).strip(),
            "canonical_url": str(item["canonical_url"]).strip(),
            "published_at": item.get("published_at"),
            "fetched_at": item.get("fetched_at") or utc_now(),
            "raw_json": raw_json,
        }
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO raw_items (
                    source, ticker, title, description, url, canonical_url,
                    published_at, fetched_at, raw_json
                ) VALUES (
                    :source, :ticker, :title, :description, :url, :canonical_url,
                    :published_at, :fetched_at, :raw_json
                )
                ON CONFLICT(source, canonical_url) DO NOTHING
                """,
                values,
            )
            if cursor.rowcount:
                return InsertResult(int(cursor.lastrowid), True)
            row = connection.execute(
                "SELECT id FROM raw_items WHERE source = ? AND canonical_url = ?",
                (values["source"], values["canonical_url"]),
            ).fetchone()
            return InsertResult(int(row["id"]), False)

    def raw_items_for_day(
        self, trading_day: str | date, ticker: str | None = None
    ) -> list[dict[str, Any]]:
        day = trading_day.isoformat() if isinstance(trading_day, date) else trading_day
        query = """
            SELECT * FROM raw_items
            WHERE substr(COALESCE(published_at, fetched_at), 1, 10) = ?
        """
        parameters: list[Any] = [day]
        if ticker:
            query += " AND ticker = ?"
            parameters.append(ticker.upper())
        query += " ORDER BY COALESCE(published_at, fetched_at), id"
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def update_raw_item_ticker(self, item_id: int, ticker: str | None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE raw_items SET ticker = ? WHERE id = ?",
                (ticker.upper() if ticker else None, item_id),
            )

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
            connection.execute(
                """
                INSERT INTO source_state (
                    source, etag, last_modified, last_checked_at,
                    last_success_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
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
                (
                    source,
                    etag,
                    last_modified,
                    checked_at,
                    checked_at if successful else None,
                    json.dumps(dict(metadata or {}), sort_keys=True),
                ),
            )

    def claim_stage_key(
        self,
        *,
        stage: str,
        ticker: str,
        trading_day: str,
        pipeline_version: str,
        run_id: str,
    ) -> bool:
        """Claim a derived-stage key, returning false after prior success.

        Failed/degraded keys may be retried. Call ``complete_stage_key`` only
        after the stage has durably written its output.
        """
        with self.connect() as connection:
            existing = connection.execute(
                """
                SELECT status FROM pipeline_stage_keys
                WHERE stage = ? AND ticker = ? AND trading_day = ?
                    AND pipeline_version = ?
                """,
                (stage, ticker.upper(), trading_day, pipeline_version),
            ).fetchone()
            if existing and existing["status"] == "success":
                return False
            connection.execute(
                """
                INSERT INTO pipeline_stage_keys (
                    stage, ticker, trading_day, pipeline_version,
                    status, run_id, updated_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?)
                ON CONFLICT(stage, ticker, trading_day, pipeline_version)
                DO UPDATE SET status = 'running', run_id = excluded.run_id,
                    updated_at = excluded.updated_at
                """,
                (
                    stage,
                    ticker.upper(),
                    trading_day,
                    pipeline_version,
                    run_id,
                    utc_now(),
                ),
            )
        return True

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
        if status not in {"success", "degraded", "failed"}:
            raise ValueError("invalid stage-key status")
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_stage_keys
                SET status = ?, updated_at = ?
                WHERE stage = ? AND ticker = ? AND trading_day = ?
                    AND pipeline_version = ? AND run_id = ?
                """,
                (
                    status,
                    utc_now(),
                    stage,
                    ticker.upper(),
                    trading_day,
                    pipeline_version,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("stage key was not claimed by this run")

    def clear_derived_for_day(self, trading_day: str | date) -> None:
        """Delete reproducible derived data only; immutable raw input is retained."""
        day = trading_day.isoformat() if isinstance(trading_day, date) else trading_day
        with self.connect() as connection:
            connection.execute("DELETE FROM themes WHERE trading_day = ?", (day,))
            connection.execute("DELETE FROM stories WHERE trading_day = ?", (day,))

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
                    json.dumps(dict(counts), sort_keys=True),
                    max(0, int(duration_ms)),
                    json.dumps(list(errors), sort_keys=True),
                    started_at,
                    completed_at,
                    resolved_status,
                    trading_day,
                    pipeline_version,
                ),
            )

    def latest_stage_status(self) -> list[dict[str, Any]]:
        """Return the newest log row per stage for the API status endpoint."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT log.*
                FROM run_log AS log
                JOIN (
                    SELECT stage, MAX(id) AS latest_id
                    FROM run_log
                    GROUP BY stage
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
        if table not in {
            "raw_items",
            "stories",
            "themes",
            "run_log",
            "eval_labels",
            "source_state",
            "pipeline_stage_keys",
        }:
            raise ValueError("unsupported table")
        with self.connect() as connection:
            return int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )

"""Yahoo Finance headline ingestion, settled through the Phase 0 run lifecycle.

Issue #61 fetches five tickers' headlines.  Issue #57 then made every
pipeline mutation carry a run: ``insert_raw_items``, ``set_source_state``,
and ``log_stage`` are gone from the public repository, and the logged
entrypoints — ``ingest_raw_items`` and ``record_source_state`` — each
require the unforgeable :class:`~phase0.repository.StageRunContext` handed
out by ``stage_run``.  This module is that port.

**What a run may speak for.**  A run covers exactly one
ticker/trading-day/pipeline-version partition.  The repository enforces
both halves: a batch asserting a foreign ticker is rejected, and so is an
item whose effective day — ``published_at`` falling back to ``fetched_at``
— is not the run's day.  A Yahoo response does not respect that shape.
One request for one symbol routinely returns articles published across
several days, so one response fans out into several partitions, each with
its own run identity and its own ``run_log`` row.  ``run_log`` is unique
on ``(run_id, stage)``, so the identity has to carry the partition:
:func:`partition_run_id` spells it ``<base>:<ticker>:<day>``.

**Where the source-state checkpoint goes.**  Source state is keyed by
feed, not by ticker-day, and it answers "when did we last check this
feed".  That is the *fetch* day, not the day an article was published, so
it is recorded on the partition whose day matches ``checked_at`` — the
repository refuses any other pairing.  It is written last, as the terminal
mutation of that run, which is what makes the feed's checkpoint and the
run that wrote it settle together.

**Why a failed persist leaves the checkpoint alone.**  If evidence cannot
be stored, this module does not stamp the feed as checked.  The old code
wrote a ``failed`` source state from outside any transaction; here the
failure is already durable — it is the partition's ``failed`` run-log row
— and writing a checkpoint on top of it would claim a check that never
completed.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import os
from queue import Empty, Queue
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
import uuid

from .errors import Phase0ValidationError
from .redaction import redact_secrets
from .repository import InsertResult, Phase0Repository, utc_now
from .tickers import TICKER_UNIVERSE, normalize_ticker
from .urls import canonicalize_url


#: The approved universe in the order the spec lists it.  Deliberately not
#: ``sorted(SUPPORTED_TICKERS)``: that is alphabetical order invented here,
#: and the shared ticker module already states the order once.
TICKERS = tuple(TICKER_UNIVERSE)

#: Every run this module opens is this stage; the partition is in the id.
STAGE = "fetch_yahoo"

LOGGER = logging.getLogger(__name__)


class YahooFetchError(RuntimeError):
    """A provider request that still failed after its configured attempts."""

    def __init__(self, ticker: str, attempts: int, original: Exception) -> None:
        super().__init__(str(original))
        self.ticker = ticker
        self.attempts = attempts
        self.original = original


def partition_run_id(base_run_id: str, ticker: str, trading_day: str) -> str:
    """The run identity for one ticker/day partition.

    ``run_log`` is ``UNIQUE(run_id, stage)``, and I1 refuses outright to
    record one identity against a second partition — a run identity names
    one partition.  Sharing a base id across the days a single response
    spans would therefore not merely muddle the audit; the second day
    would fail to persist at all.
    """

    return f"{base_run_id}:{ticker}:{trading_day}"


def effective_day(item: Mapping[str, Any]) -> str:
    """The trading day a raw item belongs to.

    Mirrors the repository's own derivation (``published_at`` falling back
    to ``fetched_at``) because the grouping done here and the partition
    check done there have to agree exactly; if they drift, a batch this
    module considers same-day is rejected on arrival.
    """

    stamp = item.get("published_at") or item.get("fetched_at")
    return str(stamp)[:10]


def _iso_timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            absolute = abs(numeric)
            if absolute >= 1_000_000_000_000_000:
                numeric /= 1_000_000
            elif absolute >= 1_000_000_000_000:
                numeric /= 1_000
            try:
                parsed = datetime.fromtimestamp(numeric, tz=timezone.utc)
            except (ValueError, OSError) as exc:
                raise ValueError("published timestamp is out of range") from exc
        else:
            text = str(value).strip()
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(text)
                except (TypeError, ValueError) as exc:
                    raise ValueError("published timestamp is not valid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _nested_url(value: Any) -> str | None:
    if isinstance(value, Mapping):
        value = value.get("url")
    return str(value).strip() if value else None


def normalize_yahoo_item(
    ticker: str,
    item: Mapping[str, Any],
    *,
    fetched_at: str | None = None,
) -> dict[str, Any]:
    """Validate and normalize one legacy or current yfinance news record.

    ``fetched_at`` is threaded through rather than taken per item so that
    every item from one response shares one fetch instant.  Reading the
    clock per item let a response straddling midnight put two undated
    articles on two different days — two partitions, from one request.
    """

    symbol = normalize_ticker(ticker)
    if not isinstance(item, Mapping):
        raise TypeError("Yahoo news item must be an object")

    content_value = item.get("content")
    content = content_value if isinstance(content_value, Mapping) else {}
    title = str(item.get("title") or content.get("title") or "").strip()
    canonical_url = _nested_url(content.get("canonicalUrl"))
    click_url = _nested_url(content.get("clickThroughUrl"))
    legacy_url = _nested_url(item.get("link"))
    url = canonical_url or click_url or legacy_url
    if not title:
        raise ValueError("Yahoo news item is missing title")
    if not url:
        raise ValueError("Yahoo news item is missing URL")
    normalized_url = canonicalize_url(url)
    parsed_url = urlsplit(normalized_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Yahoo news item URL must be absolute HTTP(S)")

    provider_value = content.get("provider")
    provider = provider_value if isinstance(provider_value, Mapping) else {}
    source = (
        str(item.get("publisher") or provider.get("displayName") or "Yahoo Finance")
        .strip()
        .replace("\x00", "")
    )
    published = (
        item.get("providerPublishTime")
        or content.get("pubDate")
        or content.get("displayTime")
    )
    return {
        "source": f"yahoo:{source}",
        "ticker": symbol,
        "tickers": [symbol],
        "title": title,
        "description": str(content.get("summary") or item.get("summary") or "").strip(),
        "url": normalized_url,
        "canonical_url": normalized_url,
        "published_at": _iso_timestamp(published),
        "fetched_at": fetched_at or utc_now(),
        "raw_json": dict(item),
    }


def _invalid_evidence(
    ticker: str,
    payload: Any,
    error: Exception,
    *,
    fetched_at: str | None = None,
) -> dict[str, Any]:
    """Keep an unusable provider record as evidence rather than dropping it.

    It carries no ``published_at`` — the timestamp is exactly what could
    not be trusted — so it settles on the fetch day.
    """

    raw_payload = (
        dict(payload) if isinstance(payload, Mapping) else {"payload": payload}
    )
    encoded = json.dumps(
        raw_payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    external_id = str(raw_payload.get("id") or raw_payload.get("uuid") or digest)
    return {
        "source": f"yahoo:{ticker}",
        "ticker": ticker,
        "tickers": [ticker],
        "canonical_url": f"urn:yahoo:{ticker.lower()}:{digest}",
        "external_id": external_id,
        "ingest_status": "invalid",
        "validation_errors": [redact_secrets(str(error))],
        "fetched_at": fetched_at or utc_now(),
        "raw_json": raw_payload,
    }


class YahooFinanceFetcher:
    """Fetch approved-ticker headlines and settle them partition by partition."""

    def __init__(
        self,
        repository: Phase0Repository,
        *,
        ticker_factory: Callable[[str], Any] | None = None,
        request_timeout_seconds: float = 30,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        throttle_seconds: float = 0,
        sleep: Callable[[float], None] = time.sleep,
        pipeline_version: str = "phase0-v1",
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_seconds < 0 or throttle_seconds < 0:
            raise ValueError("retry and throttle delays cannot be negative")
        if not pipeline_version.strip():
            raise ValueError("pipeline_version is required")
        self.repository = repository
        self._ticker_factory = ticker_factory
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.throttle_seconds = float(throttle_seconds)
        self._sleep = sleep
        self.pipeline_version = pipeline_version.strip()

    # -- Provider access -------------------------------------------------

    def _news(self, ticker: str) -> list[Any]:
        if self._ticker_factory is None:
            import yfinance as yf

            cache_path = os.getenv(
                "PHASE0_YFINANCE_CACHE",
                str(self.repository.database_path.parent / ".yfinance-cache"),
            )
            yf.set_tz_cache_location(cache_path)
            factory = yf.Ticker
        else:
            factory = self._ticker_factory
        return list(factory(ticker).news or [])

    def _news_with_timeout(self, ticker: str) -> list[Any]:
        outcomes: Queue[tuple[bool, Any]] = Queue(maxsize=1)

        def load() -> None:
            try:
                outcomes.put((True, self._news(ticker)))
            except Exception as exc:
                outcomes.put((False, exc))

        threading.Thread(
            target=load,
            name=f"yahoo-{ticker}",
            daemon=True,
        ).start()
        try:
            succeeded, result = outcomes.get(timeout=self.request_timeout_seconds)
        except Empty as exc:
            raise TimeoutError(
                f"Yahoo request exceeded {self.request_timeout_seconds:g}s"
            ) from exc
        if not succeeded:
            raise result
        return result

    def _news_with_retry(self, ticker: str) -> tuple[list[Any], int]:
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._news_with_timeout(ticker), attempt
            except Exception as exc:
                if attempt == attempts:
                    raise YahooFetchError(ticker, attempt, exc) from exc
                self._sleep(self.retry_backoff_seconds * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")

    # -- Normalization ---------------------------------------------------

    def _normalize_batch(
        self, ticker: str, items: Sequence[Any], fetched_at: str
    ) -> tuple[list[dict[str, Any]], list[bool], list[dict[str, Any]]]:
        """Normalize a response, keeping unusable records as invalid evidence."""

        normalized: list[dict[str, Any]] = []
        valid_flags: list[bool] = []
        item_errors: list[dict[str, Any]] = []
        for index, payload in enumerate(items):
            try:
                normalized.append(
                    normalize_yahoo_item(ticker, payload, fetched_at=fetched_at)
                )
                valid_flags.append(True)
            except Exception as exc:
                item_errors.append(
                    {
                        "ticker": ticker,
                        "item_index": index,
                        "error": redact_secrets(str(exc)),
                    }
                )
                normalized.append(
                    _invalid_evidence(ticker, payload, exc, fetched_at=fetched_at)
                )
                valid_flags.append(False)
        return normalized, valid_flags, item_errors

    @staticmethod
    def _provider_status(items: Sequence[Any], valid_count: int, invalid: int) -> str:
        """Map a handled provider outcome onto the source-state vocabulary.

        The four names are the repository's own
        (``SOURCE_STATE_STATUSES``), so the checkpoint and the run that
        writes it describe the same event in one language.
        """

        if not items:
            return "empty"
        if invalid and valid_count:
            return "partial"
        if invalid:
            return "failed"
        return "success"

    # -- Settlement ------------------------------------------------------

    def _settle_partition(
        self,
        *,
        ticker: str,
        trading_day: str,
        base_run_id: str,
        items: Sequence[Mapping[str, Any]],
        source_state: Mapping[str, Any] | None,
    ) -> list[InsertResult]:
        """Persist one ticker/day partition inside one run.

        Evidence goes in first; when this partition also carries the feed's
        checkpoint, that checkpoint is the terminal mutation, so the
        partition's final ``run_log`` row and the source-state write commit
        together.  Otherwise the ingest is itself terminal.
        """

        with self.repository.stage_run(
            run_id=partition_run_id(base_run_id, ticker, trading_day),
            stage=STAGE,
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version=self.pipeline_version,
        ) as run:
            results: list[InsertResult] = []
            if items or source_state is None:
                results = self.repository.ingest_raw_items(
                    list(items),
                    run=run,
                    terminal=source_state is None,
                )
            if source_state is not None:
                self.repository.record_source_state(
                    source_state["source"],
                    run=run,
                    checked_at=source_state["checked_at"],
                    status=source_state["status"],
                    metadata=source_state["metadata"],
                    error=source_state.get("error"),
                    terminal=True,
                )
            return results

    def _checkpoint(
        self,
        *,
        ticker: str,
        checked_at: str,
        status: str,
        metadata: Mapping[str, Any],
        error: Any = None,
    ) -> dict[str, Any]:
        return {
            "source": f"yahoo:{ticker}",
            "checked_at": checked_at,
            "status": status,
            "metadata": dict(metadata),
            "error": error,
        }

    def _settle_failed_fetch(
        self,
        *,
        ticker: str,
        base_run_id: str,
        checked_at: str,
        attempts: int,
        error_type: str,
        message: str,
    ) -> None:
        """Record an exhausted provider request through the run lifecycle.

        No evidence exists to ingest, so the checkpoint is the run's only
        mutation and therefore its terminal one.  This is the path that
        used to call ``log_stage`` from outside any transaction.
        """

        self._settle_partition(
            ticker=ticker,
            trading_day=checked_at[:10],
            base_run_id=base_run_id,
            items=(),
            source_state=self._checkpoint(
                ticker=ticker,
                checked_at=checked_at,
                status="failed",
                metadata={
                    "status": "failed",
                    "attempts": attempts,
                    "error_type": error_type,
                    "request_timeout_seconds": self.request_timeout_seconds,
                },
                error=message,
            ),
        )

    # -- One ticker ------------------------------------------------------

    def _fetch_ticker(
        self,
        ticker: str,
        *,
        base_run_id: str,
        counts: dict[str, int],
        errors: list[dict[str, Any]],
    ) -> None:
        checked_at = utc_now()
        fetch_day = checked_at[:10]
        try:
            items, attempts = self._news_with_retry(ticker)
        except YahooFetchError as exc:
            counts["retries"] += exc.attempts - 1
            counts["tickers_failed"] += 1
            if isinstance(exc.original, TimeoutError):
                counts["timeouts"] += 1
            message = redact_secrets(str(exc.original))
            errors.append({"ticker": ticker, "error": message})
            LOGGER.error(
                "Yahoo Finance headline fetch failed for ticker=%s error_type=%s",
                ticker,
                type(exc.original).__name__,
            )
            try:
                self._settle_failed_fetch(
                    ticker=ticker,
                    base_run_id=base_run_id,
                    checked_at=checked_at,
                    attempts=exc.attempts,
                    error_type=type(exc.original).__name__,
                    message=message,
                )
                counts["partitions"] += 1
            except Exception as settle_error:
                errors.append(
                    {
                        "ticker": ticker,
                        "error": redact_secrets(
                            f"failure settlement failed: {settle_error}"
                        ),
                    }
                )
            return

        counts["retries"] += attempts - 1
        counts["fetched"] += len(items)
        normalized, valid_flags, item_errors = self._normalize_batch(
            ticker, items, checked_at
        )
        counts["invalid"] += len(item_errors)
        errors.extend(item_errors)
        valid_count = valid_flags.count(True)
        status = self._provider_status(items, valid_count, len(item_errors))
        if status == "empty":
            errors.append({"ticker": ticker, "error": "empty provider response"})
            LOGGER.warning("Yahoo Finance returned no headlines for ticker=%s", ticker)

        by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
        flag_by_day: dict[str, list[bool]] = defaultdict(list)
        for item, is_valid in zip(normalized, valid_flags):
            day = effective_day(item)
            by_day[day].append(item)
            flag_by_day[day].append(is_valid)

        checkpoint = self._checkpoint(
            ticker=ticker,
            checked_at=checked_at,
            status=status,
            metadata={
                "status": status,
                "provider_count": len(items),
                "valid_count": valid_count,
                "invalid_count": len(item_errors),
                "attempts": attempts,
                "request_timeout_seconds": self.request_timeout_seconds,
                "errors": item_errors,
                "trading_days": sorted(by_day),
            },
        )

        # Published-day partitions first, the fetch-day partition last: the
        # checkpoint rides on that one, and it must not claim the feed was
        # checked before the evidence it describes is durable.
        settled: list[InsertResult] = []
        settled_flags: list[bool] = []
        try:
            for day in sorted(day for day in by_day if day != fetch_day):
                settled.extend(
                    self._settle_partition(
                        ticker=ticker,
                        trading_day=day,
                        base_run_id=base_run_id,
                        items=by_day[day],
                        source_state=None,
                    )
                )
                settled_flags.extend(flag_by_day[day])
                counts["partitions"] += 1
            settled.extend(
                self._settle_partition(
                    ticker=ticker,
                    trading_day=fetch_day,
                    base_run_id=base_run_id,
                    items=by_day.get(fetch_day, ()),
                    source_state=checkpoint,
                )
            )
            settled_flags.extend(flag_by_day.get(fetch_day, []))
            counts["partitions"] += 1
        except Exception as exc:
            counts["tickers_failed"] += 1
            errors.append(
                {
                    "ticker": ticker,
                    "error": redact_secrets(f"persistence failed: {exc}"),
                }
            )
            LOGGER.error(
                "Yahoo Finance headline persistence failed for ticker=%s "
                "error_type=%s",
                ticker,
                type(exc).__name__,
            )
            return

        counts[
            {
                "success": "tickers_succeeded",
                "partial": "tickers_partial",
                "empty": "tickers_empty",
                "failed": "tickers_failed",
            }[status]
        ] += 1
        for result, is_valid in zip(settled, settled_flags):
            if is_valid:
                key = "inserted" if result.inserted else "duplicates"
            else:
                key = (
                    "invalid_evidence_inserted"
                    if result.inserted
                    else "invalid_evidence_duplicates"
                )
            counts[key] += 1

    # -- Entry point -----------------------------------------------------

    def fetch(
        self,
        tickers: Iterable[str] = TICKERS,
        *,
        run_id: str | None = None,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Fetch each approved ticker and settle every partition it produces.

        There is no ``trading_day`` argument any more.  A run's day is a
        partition identity that the evidence decides, not a label a caller
        may put on it: the repository derives each item's day and refuses a
        batch that disagrees with its run, so an override could only ever
        be ignored or fatal.

        Returns the aggregate counters and the redacted error list.  Those
        are a summary for the caller — the durable audit is one ``run_log``
        row per ticker/day partition, written inside the same transaction
        as the data it describes.
        """

        base_run_id = f"yahoo-{uuid.uuid4()}" if run_id is None else str(run_id).strip()
        if not base_run_id:
            raise ValueError("run_id is required")
        counts = {
            "fetched": 0,
            "inserted": 0,
            "duplicates": 0,
            "invalid": 0,
            "invalid_evidence_inserted": 0,
            "invalid_evidence_duplicates": 0,
            "retries": 0,
            "timeouts": 0,
            "partitions": 0,
            "tickers_succeeded": 0,
            "tickers_partial": 0,
            "tickers_empty": 0,
            "tickers_failed": 0,
            "tickers_rejected": 0,
        }
        errors: list[dict[str, Any]] = []
        processed = 0
        for ticker in tickers:
            try:
                symbol = normalize_ticker(ticker)
            except Phase0ValidationError:
                counts["tickers_rejected"] += 1
                errors.append(
                    {
                        "ticker": str(ticker).strip().upper(),
                        "error": "unsupported Yahoo ticker",
                    }
                )
                continue
            if processed and self.throttle_seconds:
                self._sleep(self.throttle_seconds)
            processed += 1
            self._fetch_ticker(
                symbol, base_run_id=base_run_id, counts=counts, errors=errors
            )
        return counts, redact_secrets(errors)


__all__ = [
    "STAGE",
    "TICKERS",
    "YahooFetchError",
    "YahooFinanceFetcher",
    "effective_day",
    "normalize_yahoo_item",
    "partition_run_id",
]

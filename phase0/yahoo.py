"""Yahoo Finance headline ingestion with ticker-local failure isolation."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Any, Callable, Iterable

from .repository import Phase0Repository, utc_now
from .urls import canonicalize_url


TICKERS = ("TSLA", "NVDA", "AMD", "AAPL", "META")
LOGGER = logging.getLogger(__name__)


def _iso_timestamp(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return str(value)


def normalize_yahoo_item(ticker: str, item: dict[str, Any]) -> dict[str, Any] | None:
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    title = item.get("title") or content.get("title")
    canonical = content.get("canonicalUrl")
    canonical_url_value = (
        canonical.get("url") if isinstance(canonical, dict) else canonical
    )
    click = content.get("clickThroughUrl")
    click_url = click.get("url") if isinstance(click, dict) else click
    url = item.get("link") or canonical_url_value or click_url
    if not title or not url:
        return None
    provider = (
        content.get("provider") if isinstance(content.get("provider"), dict) else {}
    )
    source = item.get("publisher") or provider.get("displayName") or "Yahoo Finance"
    published = (
        item.get("providerPublishTime")
        or content.get("pubDate")
        or content.get("displayTime")
    )
    return {
        "source": f"yahoo:{source}",
        "ticker": ticker.upper(),
        "title": title,
        "description": content.get("summary") or item.get("summary") or "",
        "url": url,
        "canonical_url": canonicalize_url(url),
        "published_at": _iso_timestamp(published),
        "fetched_at": utc_now(),
        "raw_json": item,
    }


class YahooFinanceFetcher:
    def __init__(
        self,
        repository: Phase0Repository,
        *,
        ticker_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.repository = repository
        self._ticker_factory = ticker_factory

    def _news(self, ticker: str) -> Iterable[dict[str, Any]]:
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
        return factory(ticker).news or []

    def fetch(
        self, tickers: Iterable[str] = TICKERS
    ) -> tuple[dict[str, int], list[dict[str, str]]]:
        counts = {
            "fetched": 0,
            "inserted": 0,
            "duplicates": 0,
            "invalid": 0,
            "tickers_succeeded": 0,
            "tickers_empty": 0,
        }
        errors: list[dict[str, str]] = []
        for ticker in tickers:
            symbol = ticker.upper()
            try:
                items = list(self._news(symbol))
                counts["tickers_succeeded"] += 1
                counts["fetched"] += len(items)
                if not items:
                    counts["tickers_empty"] += 1
                    errors.append(
                        {"ticker": symbol, "error": "empty provider response"}
                    )
                    LOGGER.warning(
                        "Yahoo Finance returned no headlines for ticker=%s", symbol
                    )
                for payload in items:
                    normalized = normalize_yahoo_item(symbol, payload)
                    if normalized is None:
                        counts["invalid"] += 1
                        continue
                    result = self.repository.insert_raw_item(normalized)
                    counts["inserted" if result.inserted else "duplicates"] += 1
            except Exception as exc:
                errors.append({"ticker": symbol, "error": str(exc)})
                LOGGER.exception(
                    "Yahoo Finance headline fetch failed for ticker=%s", symbol
                )
        return counts, errors

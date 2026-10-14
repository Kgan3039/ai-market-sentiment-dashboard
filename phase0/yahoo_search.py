"""Yahoo Search news: the retrieval path that replaced ``Ticker.news`` (YF-1B).

**Why not ``yf.Ticker(t).news``.**  In yfinance 0.2.65 through 1.7.0 it
POSTs to ``finance.yahoo.com/xhr/ncp``, which Yahoo retired by October
2026.  The endpoint answers 404 with a JSON body, ``get_news`` never looks
at the status, and the result is an empty list that reads exactly like a
quiet news day.  YF-1A traced AWS-0's zero-article runs to that; the Search
endpoint served the same five tickers normally (YF-1A.1).

**Why not ``yf.Search`` itself.**  ``Search`` fetches through
``YfData.cache_get``, an ``lru_cache`` on yfinance's process-wide
singleton.  A short-lived pipeline never notices, but a long-running
process — the dashboard backend — would get its first answer for a ticker
back for as long as the entry survives.  Clearing that cache would also
evict the history, quote, and fundamentals responses it holds for other
callers.  So this module asks the same endpoint, with the parameters
``Search`` sends, through ``YfData.get``: the uncached sibling, with the
same session, cookie/crumb handling, and ``YFRateLimitError`` on 429.

**Why the status is checked here.**  Search answers a genuinely empty
query with HTTP 200 and ``"news": []``.  Anything else — a non-200, a body
that is not JSON, an object with no ``news`` list — is the provider failing,
and it raises :class:`YahooSearchError` instead of becoming ``[]``.  That is
what keeps an endpoint retirement from settling as five empty tickers again.
"""

from __future__ import annotations

from typing import Any, Mapping

SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"

#: What the retired ``get_news`` asked for by default.
DEFAULT_NEWS_COUNT = 10

#: yfinance's own default for every session call.
DEFAULT_TIMEOUT_SECONDS = 30.0


class YahooSearchError(RuntimeError):
    """The Search endpoint answered with something other than a news list."""


def search_params(ticker: str, count: int = DEFAULT_NEWS_COUNT) -> dict[str, Any]:
    """The query ``yf.Search`` sends, minus the sections nothing here reads."""

    return {
        "q": ticker,
        "quotesCount": 1,
        "enableFuzzyQuery": False,
        "newsCount": int(count),
        "quotesQueryId": "tss_match_phrase_query",
        "newsQueryId": "news_cie_vespa",
        "listsCount": 0,
        "enableCb": False,
        "enableNavLinks": False,
        "enableResearchReports": False,
        "enableCulturalAssets": False,
        "recommendedCount": 0,
    }


def fetch_search_news(
    ticker: str,
    *,
    count: int = DEFAULT_NEWS_COUNT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    data: Any = None,
) -> list[Any]:
    """One uncached Search request; the raw ``news`` records it returned.

    ``data`` is the yfinance session object to ask, defaulting to the
    process-wide ``YfData``.  ``YFRateLimitError`` and transport errors
    propagate unchanged: the caller's retry policy decides what they cost.
    """

    if data is None:
        from yfinance.data import YfData

        data = YfData()
    response = data.get(
        SEARCH_URL, params=search_params(ticker, count), timeout=timeout
    )
    status = getattr(response, "status_code", None)
    if status != 200:
        raise YahooSearchError(f"Yahoo Search returned HTTP {status}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise YahooSearchError("Yahoo Search returned a body that is not JSON") from exc
    if not isinstance(payload, Mapping):
        raise YahooSearchError("Yahoo Search returned JSON that is not an object")
    news = payload.get("news")
    if not isinstance(news, list):
        raise YahooSearchError("Yahoo Search response carries no news list")
    return list(news)


class SearchTicker:
    """The ``ticker_factory`` default: ``Ticker(t).news``'s shape, over Search.

    Every read of :attr:`news` is a fresh request.  Nothing is memoized,
    because freshness is the property the backend needs and the fetcher's
    single-flight gate already stops duplicate in-flight work.
    """

    def __init__(
        self,
        ticker: str,
        *,
        count: int = DEFAULT_NEWS_COUNT,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.ticker = ticker
        self.count = count
        self.timeout = timeout

    @property
    def news(self) -> list[Any]:
        return fetch_search_news(self.ticker, count=self.count, timeout=self.timeout)


def attribution_error(ticker: str, item: Any) -> str | None:
    """Why *item* may not be filed under *ticker*, or ``None`` if it may.

    ``relatedTickers`` verifies; it never assigns.  Search is a keyword
    query, so the requested symbol appearing in the article's own ticker
    list is the evidence that the result is about it.  The other symbols in
    that list confer nothing — they are broad, unordered, and include
    instruments outside the universe.

    A record that does not carry the field at all (the retired endpoint's
    shapes) has nothing to verify against and passes, exactly as it did
    before; the request's ticker remains its only association.
    """

    if not isinstance(item, Mapping):
        return None
    related = item.get("relatedTickers")
    if related is None:
        return None
    if not isinstance(related, list):
        return "Yahoo news item relatedTickers is not a list"
    symbols = {str(symbol).strip().upper() for symbol in related}
    if ticker.strip().upper() not in symbols:
        return f"Yahoo news item relatedTickers does not include {ticker}"
    return None


__all__ = [
    "DEFAULT_NEWS_COUNT",
    "DEFAULT_TIMEOUT_SECONDS",
    "SEARCH_URL",
    "SearchTicker",
    "YahooSearchError",
    "attribution_error",
    "fetch_search_news",
    "search_params",
]

"""Yahoo Search as the headline source (YF-1B).

Everything here runs against a fake ``yfinance.data.YfData``: the
production default factory is exercised end to end, but no request leaves
the process.
"""

from datetime import datetime, timezone
import inspect
import threading

import pytest
from yfinance.exceptions import YFRateLimitError

from phase0.repository import Phase0Repository
from phase0 import yahoo_search
from phase0.yahoo import YahooFinanceFetcher, YahooProviderGate, normalize_yahoo_item
from phase0.yahoo_search import (
    SEARCH_URL,
    SearchTicker,
    YahooSearchError,
    attribution_error,
    fetch_search_news,
    search_params,
)


def now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def search_item(ticker, *, uuid=None, related=None, **updates):
    """A record in the shape YF-1A.1 observed from Search."""

    item = {
        "uuid": uuid or f"{ticker.lower()}-0000-uuid",
        "title": f"{ticker} search headline",
        "publisher": "Motley Fool",
        "link": (
            f"https://finance.yahoo.com/markets/stocks/articles/{ticker.lower()}-1.html"
        ),
        "providerPublishTime": now_epoch(),
        "type": "STORY",
        "relatedTickers": [ticker] if related is None else related,
    }
    item.update(updates)
    return item


class FakeResponse:
    def __init__(self, status_code=200, payload=None, *, body_error=None):
        self.status_code = status_code
        self._payload = payload
        self._body_error = body_error

    def json(self):
        if self._body_error is not None:
            raise self._body_error
        return self._payload


class FakeData:
    """Stands in for yfinance's ``YfData``.  It has no ``cache_get`` on purpose."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def get(self, url, params=None, timeout=30):
        self.calls.append(
            {"url": url, "params": dict(params or {}), "timeout": timeout}
        )
        answer = self.answer(params["q"]) if callable(self.answer) else self.answer
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def yfdata(monkeypatch, tmp_path):
    """Install a fake ``YfData`` and return a setter for its answer."""

    monkeypatch.setenv("PHASE0_YFINANCE_CACHE", str(tmp_path / "yf-cache"))
    holder = {}

    def install(answer):
        data = FakeData(answer)
        holder["data"] = data
        monkeypatch.setattr("yfinance.data.YfData", lambda: data)
        return data

    return install


def ok(news):
    return FakeResponse(200, {"count": len(news), "quotes": [], "news": news})


def repository_at(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    return repository


def search_fetcher(repository, **options):
    """A fetcher on the production default factory -- no ``ticker_factory``."""

    options.setdefault("max_retries", 0)
    options.setdefault("provider_gate", YahooProviderGate())
    return YahooFinanceFetcher(repository, **options)


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_the_request_asks_search_for_the_ticker_news():
    data = FakeData(ok([search_item("TSLA")]))

    news = fetch_search_news("TSLA", count=7, timeout=12.5, data=data)

    assert [item["uuid"] for item in news] == ["tsla-0000-uuid"]
    (call,) = data.calls
    assert call["url"] == SEARCH_URL
    assert call["params"]["q"] == "TSLA"
    assert call["params"]["newsCount"] == 7
    assert call["timeout"] == 12.5
    assert call["params"] == search_params("TSLA", 7)


def test_a_well_formed_empty_answer_is_an_empty_list():
    """HTTP 200 with ``news: []`` is how Search reports a quiet query."""

    assert fetch_search_news("TSLA", data=FakeData(ok([]))) == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(404, {"message": "Not Found"}), "HTTP 404"),
        (FakeResponse(500, {}), "HTTP 500"),
        (FakeResponse(200, body_error=ValueError("no json")), "not JSON"),
        (FakeResponse(200, ["not", "an", "object"]), "not an object"),
        (FakeResponse(200, {"quotes": []}), "no news list"),
        (FakeResponse(200, {"news": {"not": "a list"}}), "no news list"),
    ],
    ids=["404", "500", "non-json", "non-object", "missing-news", "news-not-list"],
)
def test_a_broken_answer_raises_instead_of_reading_as_empty(response, message):
    """The defect YF-1A found: a retired endpoint must not look like a quiet day."""

    with pytest.raises(YahooSearchError, match=message):
        fetch_search_news("TSLA", data=FakeData(response))


def test_rate_limiting_propagates_for_the_retry_policy_to_handle():
    with pytest.raises(YFRateLimitError):
        fetch_search_news("TSLA", data=FakeData(YFRateLimitError()))


def test_every_read_is_a_fresh_request(yfdata):
    """Nothing is memoized: a long-lived process sees each new answer."""

    answers = iter(
        [
            ok([search_item("TSLA", uuid="first")]),
            ok([search_item("TSLA", uuid="second")]),
        ]
    )
    data = yfdata(lambda _: next(answers))
    ticker = SearchTicker("TSLA")

    assert [item["uuid"] for item in ticker.news] == ["first"]
    assert [item["uuid"] for item in ticker.news] == ["second"]
    assert len(data.calls) == 2


def test_the_request_goes_through_yfinance_uncached_get():
    """``YfData.get`` is the seam this module depends on; hold its shape."""

    from yfinance.data import YfData

    parameters = inspect.signature(YfData.get).parameters
    assert {"url", "params", "timeout"} <= set(parameters)


# ---------------------------------------------------------------------------
# relatedTickers verifies, never assigns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"relatedTickers": ["TSLA"]}, None),
        ({"relatedTickers": ["SPCX", "tsla ", "TSLA"]}, None),
        ({}, None),
        ({"relatedTickers": None}, None),
        ({"relatedTickers": ["NVDA", "AMD"]}, "does not include TSLA"),
        ({"relatedTickers": []}, "does not include TSLA"),
        ({"relatedTickers": "TSLA"}, "not a list"),
    ],
    ids=[
        "listed",
        "listed-among-noise",
        "absent",
        "null",
        "other-tickers",
        "empty",
        "string",
    ],
)
def test_attribution_is_checked_only_against_the_requested_ticker(item, expected):
    error = attribution_error("TSLA", item)

    if expected is None:
        assert error is None
    else:
        assert expected in error


def test_a_search_record_normalizes_like_any_other_yahoo_record():
    published = now_epoch()
    item = search_item(
        "NVDA",
        uuid="5bfa9b58-9569-36e3-9d2d-d15047137138",
        related=["AAPL", "NVDA", "SPCX"],
        providerPublishTime=published,
    )

    normalized = normalize_yahoo_item(
        "NVDA", item, fetched_at="2026-10-15T00:00:00+00:00"
    )

    assert normalized["external_id"] == "5bfa9b58-9569-36e3-9d2d-d15047137138"
    assert normalized["source"] == "yahoo:Motley Fool"
    assert normalized["ticker"] == "NVDA"
    assert normalized["tickers"] == ["NVDA"]
    assert normalized["title"] == "NVDA search headline"
    assert normalized["url"] == normalized["canonical_url"]
    assert normalized["canonical_url"].startswith("https://finance.yahoo.com/")
    assert (
        normalized["published_at"]
        == datetime.fromtimestamp(published, timezone.utc).isoformat()
    )
    assert normalized["raw_json"]["relatedTickers"] == ["AAPL", "NVDA", "SPCX"]


def test_an_unattributed_search_record_is_rejected_by_normalization():
    with pytest.raises(ValueError, match="does not include NVDA"):
        normalize_yahoo_item("NVDA", search_item("NVDA", related=["AMD"]))


# ---------------------------------------------------------------------------
# Through the fetcher's default factory
# ---------------------------------------------------------------------------


def test_the_default_fetcher_settles_search_news_with_its_uuid(tmp_path, yfdata):
    data = yfdata(lambda symbol: ok([search_item(symbol)]))
    repository = repository_at(tmp_path)

    counts, errors = search_fetcher(repository).fetch(["TSLA", "NVDA"])

    assert errors == []
    assert counts["tickers_succeeded"] == 2
    assert counts["inserted"] == 2
    assert [call["params"]["q"] for call in data.calls] == ["TSLA", "NVDA"]
    (row,) = repository.raw_items_for_day(today(), "TSLA")
    assert row["external_id"] == "tsla-0000-uuid"
    assert row["ingest_status"] == "valid"


def test_the_fetcher_timeout_is_handed_to_the_search_request(tmp_path, yfdata):
    data = yfdata(lambda symbol: ok([search_item(symbol)]))

    search_fetcher(repository_at(tmp_path), request_timeout_seconds=7).fetch(["TSLA"])

    assert data.calls[0]["timeout"] == 7.0


def test_other_related_tickers_gain_no_association(tmp_path, yfdata):
    yfdata(lambda symbol: ok([search_item(symbol, related=["NVDA", "AMD", "AAPL"])]))
    repository = repository_at(tmp_path)

    search_fetcher(repository).fetch(["NVDA"])

    assert len(repository.raw_items_for_day(today(), "NVDA")) == 1
    assert repository.raw_items_for_day(today(), "AMD") == []
    assert repository.raw_items_for_day(today(), "AAPL") == []


def test_an_unattributed_record_is_kept_as_invalid_evidence(tmp_path, yfdata):
    yfdata(
        lambda symbol: ok(
            [search_item(symbol), search_item(symbol, uuid="stray", related=["AMD"])]
        )
    )
    repository = repository_at(tmp_path)

    counts, errors = search_fetcher(repository).fetch(["NVDA"])

    assert (counts["inserted"], counts["invalid"]) == (1, 1)
    assert counts["tickers_partial"] == 1
    assert errors == [
        {
            "ticker": "NVDA",
            "item_index": 1,
            "error": "Yahoo news item relatedTickers does not include NVDA",
        }
    ]
    statuses = sorted(
        row["ingest_status"] for row in repository.raw_items_for_day(today(), "NVDA")
    )
    assert statuses == ["invalid", "valid"]


def test_conflicting_identifiers_are_recorded_on_the_checkpoint(tmp_path, yfdata):
    yfdata(lambda symbol: ok([search_item(symbol, id="prov-9001", uuid="uuid-1")]))
    repository = repository_at(tmp_path)

    counts, errors = search_fetcher(repository).fetch(["NVDA"])

    assert errors == []
    assert counts["id_conflicts"] == 1
    assert counts["tickers_succeeded"] == 1
    (row,) = repository.raw_items_for_day(today(), "NVDA")
    assert row["ingest_status"] == "valid"
    assert row["external_id"] is None
    (conflict,) = repository.source_state("yahoo:NVDA")["metadata"]["id_conflicts"]
    assert conflict["item_index"] == 0
    assert "disagree" in conflict["conflict"]


# ---------------------------------------------------------------------------
# Failure, rate limiting, retries, timeouts, and empty answers
# ---------------------------------------------------------------------------


def test_a_retired_endpoint_fails_the_ticker_instead_of_settling_empty(
    tmp_path, yfdata
):
    yfdata(FakeResponse(404, {"message": "Not Found"}))
    repository = repository_at(tmp_path)

    counts, errors = search_fetcher(repository).fetch(["TSLA"])

    assert counts["tickers_failed"] == 1
    assert counts["tickers_empty"] == 0
    assert errors == [{"ticker": "TSLA", "error": "Yahoo Search returned HTTP 404"}]
    state = repository.source_state("yahoo:TSLA")
    assert state["status"] == "failed"
    assert state["metadata"]["error_type"] == "YahooSearchError"


def test_rate_limiting_is_retried_then_reported(tmp_path, yfdata):
    data = yfdata(YFRateLimitError())
    sleeps = []
    repository = repository_at(tmp_path)

    counts, _ = search_fetcher(
        repository, max_retries=2, retry_backoff_seconds=1, sleep=sleeps.append
    ).fetch(["TSLA"])

    assert len(data.calls) == 3
    assert sleeps == [1, 2]
    assert (counts["retries"], counts["tickers_failed"]) == (2, 1)
    assert repository.source_state("yahoo:TSLA")["metadata"]["error_type"] == (
        "YFRateLimitError"
    )


def test_a_rate_limit_that_clears_on_retry_succeeds(tmp_path, yfdata):
    answers = iter([YFRateLimitError(), ok([search_item("TSLA")])])
    yfdata(lambda _: next(answers))

    counts, errors = search_fetcher(
        repository_at(tmp_path), max_retries=1, sleep=lambda _: None
    ).fetch(["TSLA"])

    assert errors == []
    assert (counts["retries"], counts["tickers_succeeded"]) == (1, 1)


def test_a_hanging_search_request_times_out(tmp_path, yfdata):
    release = threading.Event()

    def hang(_):
        release.wait(10)
        return ok([])

    yfdata(hang)
    try:
        counts, errors = search_fetcher(
            repository_at(tmp_path), request_timeout_seconds=0.05
        ).fetch(["TSLA"])
    finally:
        release.set()

    assert (counts["timeouts"], counts["tickers_failed"]) == (1, 1)
    assert "exceeded" in errors[0]["error"]


def test_one_quiet_ticker_is_not_called_an_outage(tmp_path, yfdata):
    yfdata(lambda symbol: ok([] if symbol == "META" else [search_item(symbol)]))

    counts, errors = search_fetcher(repository_at(tmp_path)).fetch(
        ["TSLA", "NVDA", "AMD", "AAPL", "META"]
    )

    assert counts["tickers_empty"] == 1
    assert errors == [{"ticker": "META", "error": "empty provider response"}]


def test_every_ticker_empty_is_named_as_a_possible_outage(tmp_path, yfdata):
    yfdata(ok([]))

    counts, errors = search_fetcher(repository_at(tmp_path)).fetch(
        ["TSLA", "NVDA", "AMD", "AAPL", "META"]
    )

    assert counts["tickers_empty"] == 5
    diagnostics = [error for error in errors if "diagnostic" in error]
    assert diagnostics == [
        {
            "diagnostic": "all_tickers_empty",
            "tickers": 5,
            "error": (
                "every requested Yahoo ticker returned an empty provider "
                "response; possible provider outage rather than a quiet "
                "news period"
            ),
        }
    ]


def test_a_single_requested_ticker_cannot_diagnose_an_outage(tmp_path, yfdata):
    yfdata(ok([]))

    _, errors = search_fetcher(repository_at(tmp_path)).fetch(["TSLA"])

    assert errors == [{"ticker": "TSLA", "error": "empty provider response"}]


def test_a_failed_ticker_is_not_counted_toward_all_empty(tmp_path, yfdata):
    yfdata(lambda symbol: FakeResponse(500, {}) if symbol == "TSLA" else ok([]))

    _, errors = search_fetcher(repository_at(tmp_path)).fetch(["TSLA", "NVDA"])

    assert not [error for error in errors if "diagnostic" in error]


def test_the_module_reads_yfdata_at_call_time(monkeypatch):
    """The default session is looked up per request, so callers can swap it."""

    data = FakeData(ok([]))
    monkeypatch.setattr("yfinance.data.YfData", lambda: data)

    yahoo_search.fetch_search_news("AMD")

    assert len(data.calls) == 1

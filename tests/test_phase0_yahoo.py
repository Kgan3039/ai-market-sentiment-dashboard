from phase0.repository import Phase0Repository
from phase0.yahoo import TICKERS, YahooFinanceFetcher, normalize_yahoo_item


def test_normalizes_current_nested_yahoo_shape():
    item = normalize_yahoo_item(
        "NVDA",
        {
            "content": {
                "title": "NVIDIA headline",
                "summary": "Company update",
                "canonicalUrl": {"url": "https://example.com/story?utm_source=yahoo"},
                "provider": {"displayName": "Example News"},
                "pubDate": "2026-07-23T12:00:00Z",
            }
        },
    )

    assert item["ticker"] == "NVDA"
    assert item["source"] == "yahoo:Example News"
    assert item["canonical_url"] == "https://example.com/story"


def test_ticker_failure_does_not_abort_other_tickers(tmp_path, caplog):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    class FakeTicker:
        def __init__(self, ticker):
            if ticker == "TSLA":
                raise RuntimeError("provider unavailable")
            self.news = [
                {
                    "title": f"{ticker} headline",
                    "link": f"https://example.com/{ticker}?utm_source=yahoo",
                    "providerPublishTime": 1784808000,
                }
            ]

    counts, errors = YahooFinanceFetcher(repository, ticker_factory=FakeTicker).fetch(
        ["TSLA", "NVDA"]
    )

    assert counts["tickers_succeeded"] == 1
    assert counts["inserted"] == 1
    assert errors[0]["ticker"] == "TSLA"
    assert "ticker=TSLA" in caplog.text


def test_three_consecutive_runs_are_idempotent_for_all_five_tickers(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    class FakeTicker:
        def __init__(self, ticker):
            self.news = [
                {
                    "title": f"{ticker} headline",
                    "link": f"https://example.com/{ticker}?utm_source=yahoo",
                    "providerPublishTime": 1784808000,
                }
            ]

    fetcher = YahooFinanceFetcher(repository, ticker_factory=FakeTicker)
    runs = [fetcher.fetch() for _ in range(3)]

    assert runs[0][0]["inserted"] == 5
    assert runs[1][0]["duplicates"] == 5
    assert runs[2][0]["duplicates"] == 5
    assert all(not errors for _, errors in runs)
    with repository.connect() as connection:
        stored = {
            row[0]
            for row in connection.execute("SELECT DISTINCT ticker FROM raw_items")
        }
    assert stored == set(TICKERS)


def test_empty_provider_response_is_degraded(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()

    class EmptyTicker:
        news = []

        def __init__(self, ticker):
            pass

    counts, errors = YahooFinanceFetcher(repository, ticker_factory=EmptyTicker).fetch(
        ["NVDA"]
    )

    assert counts["tickers_empty"] == 1
    assert errors == [{"ticker": "NVDA", "error": "empty provider response"}]

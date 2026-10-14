"""Dashboard headlines from Yahoo Search (YF-1B), offline against a fake session."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from app.services.data_service import DataService  # noqa: E402


class FakeSearchResponse:
    status_code = 200

    def __init__(self, news):
        self._news = news

    def json(self):
        return {"count": len(self._news), "quotes": [], "news": self._news}


class FakeYfData:
    """yfinance's session object, answering Search from a list of responses.

    It has no ``cache_get``: the headline path must not need one.
    """

    def __init__(self, answers):
        self._answers = iter(answers)
        self.calls = 0

    def get(self, url, params=None, timeout=30):
        self.calls += 1
        return FakeSearchResponse(next(self._answers))


def search_item(uuid, title, related):
    return {
        "uuid": uuid,
        "title": title,
        "publisher": "Motley Fool",
        "link": f"https://finance.yahoo.com/markets/stocks/articles/{uuid}.html",
        "providerPublishTime": 1791998863,
        "type": "STORY",
        "relatedTickers": related,
    }


@pytest.fixture(autouse=True)
def clean_service_cache():
    original_ttl = DataService._CACHE_TTL_SECONDS
    DataService._PROVIDER_CACHE.clear()
    DataService._PROVIDER_STATUS.clear()
    yield
    DataService._CACHE_TTL_SECONDS = original_ttl
    DataService._PROVIDER_CACHE.clear()
    DataService._PROVIDER_STATUS.clear()


def test_headlines_are_fresh_once_the_service_cache_expires(monkeypatch):
    """A long-running backend must not keep serving yfinance's first answer."""

    fake = FakeYfData(
        [
            [search_item("first-uuid", "First NVDA story", ["NVDA"])],
            [search_item("second-uuid", "Second NVDA story", ["NVDA"])],
        ]
    )
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    first = DataService.get_headlines("NVDA")
    DataService._CACHE_TTL_SECONDS = -1
    second = DataService.get_headlines("NVDA")

    assert [item.id for item in first] == ["first-uuid"]
    assert [item.id for item in second] == ["second-uuid"]
    assert second[0].headline == "Second NVDA story"
    assert fake.calls == 2


def test_the_service_cache_still_answers_within_its_ttl(monkeypatch):
    """Freshness comes from the service's own TTL, not from refetching per call."""

    fake = FakeYfData([[search_item("only-uuid", "NVDA story", ["NVDA"])]])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    DataService.get_headlines("NVDA")
    DataService.get_headlines("NVDA")

    assert fake.calls == 1


def test_headlines_skip_results_not_attributed_to_the_ticker(monkeypatch):
    fake = FakeYfData(
        [
            [
                search_item("amd-only", "An AMD story", ["AMD"]),
                search_item("nvda-uuid", "An NVDA story", ["AMD", "NVDA"]),
            ]
        ]
    )
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    headlines = DataService.get_headlines("NVDA")

    assert [item.id for item in headlines] == ["nvda-uuid"]
    assert headlines[0].url.startswith("https://finance.yahoo.com/")
    assert headlines[0].source == "Motley Fool"
    assert DataService.get_headlines_status("NVDA")["status"] == "live"

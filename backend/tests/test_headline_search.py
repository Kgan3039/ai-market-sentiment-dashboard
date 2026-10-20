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


class FailingSearchResponse:
    status_code = 404

    def json(self):
        return {"message": "Not Found"}


def identity_item(**fields):
    item = {
        "title": "NVDA story",
        "publisher": "Motley Fool",
        "link": "https://finance.yahoo.com/m/abc/story.html",
        "providerPublishTime": 1791998863,
        "relatedTickers": ["NVDA"],
    }
    item.update(fields)
    return item


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"id": "same-id", "uuid": "same-id"}, "same-id"),
        ({"id": "only-id"}, "only-id"),
        ({"uuid": "only-uuid"}, "only-uuid"),
        ({"id": "  padded-id  "}, "padded-id"),
        ({"uuid": "\tpadded-uuid\n"}, "padded-uuid"),
        ({"id": " same ", "uuid": "same"}, "same"),
        ({"id": "   ", "uuid": "the-uuid"}, "the-uuid"),
        ({"id": "", "uuid": "the-uuid"}, "the-uuid"),
        ({"id": None, "uuid": "the-uuid"}, "the-uuid"),
    ],
    ids=[
        "matching",
        "id-only",
        "uuid-only",
        "id-whitespace",
        "uuid-whitespace",
        "match-after-strip",
        "blank-id",
        "empty-id",
        "null-id",
    ],
)
def test_headline_id_is_the_ingestion_provider_id(fields, expected):
    from phase0.yahoo import provider_item_id

    item = identity_item(**fields)
    headline = DataService._normalize_yfinance_news_item("NVDA", item, 3)

    assert headline.id == expected
    assert headline.id == provider_item_id(item)


@pytest.mark.parametrize(
    "fields",
    [
        {"id": "article-a", "uuid": "article-b"},
        {},
        {"id": "   ", "uuid": "  "},
        {"content": {"id": "content-only-id"}},
        {"id": "", "content": {"id": "content-only-id"}},
    ],
    ids=["conflict", "missing", "all-blank", "content-id-only", "blank-id-content-id"],
)
def test_headline_without_a_provider_id_gets_only_a_ui_key(fields):
    """Conflicts, blanks and ``content.id`` never become provider identity."""

    from phase0.yahoo import provider_item_id

    item = identity_item(**fields)
    headline = DataService._normalize_yfinance_news_item("NVDA", item, 3)

    assert provider_item_id(item) is None
    assert headline.id == "NVDA-3"


def test_conflicting_ids_reach_the_dashboard_as_a_ui_key(monkeypatch):
    fake = FakeYfData(
        [
            [
                identity_item(id="article-a", uuid="article-b", title="Conflict"),
                identity_item(uuid="good-uuid", title="Good"),
            ]
        ]
    )
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    headlines = DataService.get_headlines("NVDA")

    assert [item.id for item in headlines] == ["NVDA-0", "good-uuid"]


@pytest.mark.parametrize(
    "batch",
    [
        [search_item("valid-uuid", "Valid NVDA story", ["NVDA"]), None],
        [None, search_item("valid-uuid", "Valid NVDA story", ["NVDA"])],
        [
            "not an article",
            search_item("valid-uuid", "Valid NVDA story", ["NVDA"]),
            42,
            ["NVDA"],
        ],
    ],
    ids=["valid-then-none", "none-then-valid", "mixed-types"],
)
def test_malformed_articles_do_not_discard_valid_headlines(monkeypatch, batch):
    fake = FakeYfData([batch])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    headlines = DataService.get_headlines("NVDA")

    assert [item.id for item in headlines] == ["valid-uuid"]
    status = DataService.get_headlines_status("NVDA")
    assert status["status"] == "live"
    assert status["count"] == 1


def test_articles_with_a_non_text_title_are_skipped_alone(monkeypatch):
    fake = FakeYfData(
        [
            [
                search_item("bad-title", {"text": "nested"}, ["NVDA"]),
                search_item("valid-uuid", "Valid NVDA story", ["NVDA"]),
            ]
        ]
    )
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    assert [item.id for item in DataService.get_headlines("NVDA")] == ["valid-uuid"]


def test_a_completely_malformed_batch_is_empty_not_a_provider_failure(monkeypatch):
    fake = FakeYfData([[None, "text", 7, [], {"title": ["not", "text"]}]])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)
    monkeypatch.setattr(DataService, "_pipeline_headlines", lambda *_args: [])

    assert DataService.get_headlines("NVDA") == []
    status = DataService.get_headlines_status("NVDA")
    assert status["status"] == "unavailable"
    assert status["message"] == "Headline provider returned no articles."


def test_a_failing_provider_is_still_reported_as_a_failure(monkeypatch):
    class FailingYfData:
        def get(self, url, params=None, timeout=30):
            return FailingSearchResponse()

    monkeypatch.setattr("yfinance.data.YfData", FailingYfData)
    monkeypatch.setattr(DataService, "_pipeline_headlines", lambda *_args: [])

    assert DataService.get_headlines("NVDA") == []
    status = DataService.get_headlines_status("NVDA")
    assert status["status"] == "unavailable"
    assert status["message"] == "Headline provider failed: YahooSearchError."

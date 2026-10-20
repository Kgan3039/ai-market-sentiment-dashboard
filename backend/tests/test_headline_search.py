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


def valid_item(uuid):
    return search_item(uuid, f"Story {uuid}", ["NVDA"])


def bad_title_item(uuid="bad-title"):
    return search_item(uuid, {"text": "nested"}, ["NVDA"])


def headline_ids(monkeypatch, batch, limit):
    fake = FakeYfData([batch])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)
    monkeypatch.setattr(DataService, "_pipeline_headlines", lambda *_args: [])
    return [item.id for item in DataService.get_headlines("NVDA", limit=limit)]


@pytest.mark.parametrize(
    "batch",
    [[bad_title_item(), valid_item("v1")], [valid_item("v1"), bad_title_item()]],
    ids=["invalid-first", "valid-first"],
)
def test_a_skipped_record_does_not_spend_the_limit(monkeypatch, batch):
    assert headline_ids(monkeypatch, batch, limit=1) == ["v1"]


def test_several_skipped_records_before_valid_ones(monkeypatch):
    batch = [
        bad_title_item("bad-1"),
        None,
        {**valid_item("bad-link"), "link": {"url": "x"}},
        bad_title_item("bad-2"),
        valid_item("v1"),
        valid_item("v2"),
        valid_item("v3"),
    ]

    assert headline_ids(monkeypatch, batch, limit=2) == ["v1", "v2"]


def test_a_limit_above_the_valid_count_returns_every_valid_record(monkeypatch):
    batch = [valid_item("v1"), bad_title_item(), valid_item("v2")]

    assert headline_ids(monkeypatch, batch, limit=10) == ["v1", "v2"]


def test_valid_records_keep_provider_order_under_a_limit(monkeypatch):
    batch = [valid_item(f"v{n}") for n in range(1, 6)]

    assert headline_ids(monkeypatch, batch, limit=3) == ["v1", "v2", "v3"]


@pytest.mark.parametrize("limit", [0, -1])
def test_the_fetch_loop_returns_nothing_for_a_limit_below_one(monkeypatch, limit):
    """Nothing requested, nothing returned -- never a slice from the end."""

    fake = FakeYfData([[valid_item("v1"), valid_item("v2")]])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    assert DataService._fetch_yfinance_headlines("NVDA", limit) == []


def committed_nvda_fallback_count(limit):
    """Demo headlines the committed dataset really holds for NVDA, unmocked."""

    return len(DataService._pipeline_headlines("NVDA", limit))


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_returns_nothing_before_any_lookup(monkeypatch, limit):
    """Not the cache, not Search, not the committed dataset fallback."""

    assert committed_nvda_fallback_count(6) > 0
    calls = []

    def spy(name):
        def record(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was consulted for limit={limit}")

        return record

    monkeypatch.setattr("yfinance.data.YfData", spy("YfData"))
    monkeypatch.setattr(DataService, "_cache_lookup", spy("_cache_lookup"))
    monkeypatch.setattr(
        DataService, "_fetch_yfinance_headlines", spy("_fetch_yfinance_headlines")
    )
    monkeypatch.setattr(DataService, "_pipeline_headlines", spy("_pipeline_headlines"))

    assert DataService.get_headlines("NVDA", limit=limit) == []
    assert calls == []
    assert DataService.get_headlines_status("NVDA", limit=limit) == {
        "available": False,
        "status": "unavailable",
        "source": "Yahoo Finance via yfinance",
        "message": f"No headlines were requested (limit {limit}).",
        "count": 0,
    }


@pytest.mark.parametrize("limit", [0, -1])
def test_a_limit_below_one_ignores_the_real_dataset_fallback(monkeypatch, limit):
    """Empty Search answer, real committed NVDA dataset: still nothing."""

    assert committed_nvda_fallback_count(6) > 0
    fake = FakeYfData([[]])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    assert DataService.get_headlines("NVDA", limit=limit) == []
    assert fake.calls == 0


@pytest.mark.parametrize("limit", [1, 2, 6])
def test_positive_limits_still_use_the_real_dataset_fallback(monkeypatch, limit):
    expected = committed_nvda_fallback_count(limit)
    assert expected == limit
    fake = FakeYfData([[]])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    headlines = DataService.get_headlines("NVDA", limit=limit)

    assert len(headlines) == expected
    assert fake.calls == 1
    status = DataService.get_headlines_status("NVDA", limit=limit)
    assert status["status"] == "cached"
    assert status["source"] == "Committed demo dataset"
    assert status["count"] == expected


@pytest.mark.parametrize("limit", [1, 2])
def test_positive_limits_still_cap_live_headlines(monkeypatch, limit):
    batch = [valid_item("v1"), valid_item("v2"), valid_item("v3")]

    assert headline_ids(monkeypatch, batch, limit=limit) == ["v1", "v2"][:limit]


def test_ui_keys_stay_unique_when_records_are_skipped(monkeypatch):
    batch = [bad_title_item(), identity_item(title="A"), identity_item(title="B")]

    assert headline_ids(monkeypatch, batch, limit=2) == ["NVDA-1", "NVDA-2"]


MALFORMED_FIELDS = {
    "link-object": {"link": {"url": "https://finance.yahoo.com/x"}},
    "link-number": {"link": 12345},
    "publisher-object": {"publisher": {"name": "Motley Fool"}},
    "publisher-number": {"publisher": 7},
    "summary-list": {"summary": ["not", "text"]},
    "summary-number": {"summary": 3.5},
}


@pytest.mark.parametrize("malformed", MALFORMED_FIELDS.values(), ids=MALFORMED_FIELDS)
@pytest.mark.parametrize(
    "malformed_first", [True, False], ids=["bad-first", "bad-last"]
)
def test_a_malformed_field_skips_only_its_article(
    monkeypatch, malformed, malformed_first
):
    bad = {**valid_item("bad"), **malformed}
    good = valid_item("good")
    batch = [bad, good] if malformed_first else [good, bad]

    assert headline_ids(monkeypatch, batch, limit=6) == ["good"]
    status = DataService.get_headlines_status("NVDA")
    assert status["status"] == "live"
    assert status["count"] == 1


def test_malformed_fields_are_rejected_not_coerced(monkeypatch):
    """A number in a text field is not turned into text."""

    batch = [{**valid_item("bad"), "publisher": 7, "summary": 3.5}]

    assert headline_ids(monkeypatch, batch, limit=6) == []


def test_a_batch_of_only_malformed_fields_is_an_empty_result(monkeypatch):
    batch = [
        {**valid_item(name), **fields} for name, fields in MALFORMED_FIELDS.items()
    ]

    assert headline_ids(monkeypatch, batch, limit=6) == []
    status = DataService.get_headlines_status("NVDA")
    assert status["status"] == "unavailable"
    assert status["message"] == "Headline provider returned no articles."


@pytest.mark.parametrize("published", [10**30, float("nan"), float("inf")])
def test_an_unrepresentable_publish_time_reads_as_missing(monkeypatch, published):
    batch = [{**valid_item("v1"), "providerPublishTime": published}]
    fake = FakeYfData([batch])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)

    headlines = DataService.get_headlines("NVDA")

    assert [item.id for item in headlines] == ["v1"]
    assert headlines[0].published_at is None
    assert headlines[0].time is None


def test_a_programming_error_in_normalization_is_not_swallowed(monkeypatch):
    """Only schema validation is per-article; other errors stay provider failures."""

    def broken(*_args):
        raise TypeError("bug in normalization")

    fake = FakeYfData([[valid_item("v1")]])
    monkeypatch.setattr("yfinance.data.YfData", lambda: fake)
    monkeypatch.setattr(DataService, "_normalize_yfinance_news_item", broken)
    monkeypatch.setattr(DataService, "_pipeline_headlines", lambda *_args: [])

    assert DataService.get_headlines("NVDA") == []
    status = DataService.get_headlines_status("NVDA")
    assert status["message"] == "Headline provider failed: TypeError."

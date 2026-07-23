from pathlib import Path

from phase0.relevance import load_alias_config, match_ticker
from phase0.repository import Phase0Repository
from phase0.rss import RSSFetcher, parse_feed
from phase0.urls import canonicalize_url


ALIASES = {
    "tickers": [
        {
            "ticker": "AAPL",
            "cashtag": "$AAPL",
            "official_company_name": "Apple Inc.",
            "strong_aliases": ["iPhone"],
            "context_required_aliases": ["Apple"],
            "context_terms": ["iPhone", "Tim Cook"],
            "exclusion_terms": ["apple pie"],
        },
        {
            "ticker": "META",
            "cashtag": "$META",
            "official_company_name": "Meta Platforms, Inc.",
            "strong_aliases": ["Instagram"],
            "context_required_aliases": ["Meta"],
            "context_terms": ["Facebook"],
            "exclusion_terms": ["metadata"],
        },
    ]
}


def test_canonicalize_url_removes_tracking_and_fragment():
    value = canonicalize_url(
        "HTTPS://Example.COM:443/news/?b=2&utm_source=x&a=1#section"
    )
    assert value == "https://example.com/news?a=1&b=2"


def test_relevance_requires_context_and_honors_exclusion():
    assert match_ticker("New iPhone arrives", "", ALIASES).ticker == "AAPL"
    assert match_ticker("Apple harvest guide", "apple pie", ALIASES).ticker is None
    assert match_ticker("Metadata tools", "", ALIASES).ticker is None


def test_relevance_flags_multiple_tickers():
    result = match_ticker("iPhone adds Instagram feature", "", ALIASES)
    assert result.ticker is None
    assert result.ambiguous is True
    assert result.matches == ("AAPL", "META")


def test_production_aliases_match_symbols_and_documented_examples():
    aliases = load_alias_config(Path("config/aliases.yaml"))
    cases = {
        "TSLA shares rose.": "TSLA",
        "NVIDIA announced Blackwell.": "NVDA",
        "NVDA shares were active.": "NVDA",
        "AMD Ryzen processor demand grew.": "AMD",
        "AAPL released quarterly results.": "AAPL",
        "Meta Platforms expands Reality Labs.": "META",
    }
    for text, expected in cases.items():
        assert match_ticker(text, "", aliases).ticker == expected


def test_production_aliases_reject_documented_false_positives():
    aliases = load_alias_config(Path("config/aliases.yaml"))
    cases = [
        "Apple pie recipes are popular.",
        "A meta-analysis reviewed the evidence.",
        "The clinic studies age-related macular degeneration.",
        "A Tesla coil powered the classroom.",
        "Elon Musk discussed SpaceX.",
        "Pineapple exports increased.",
        "The metadata schema changed.",
    ]
    assert all(match_ticker(text, "", aliases).ticker is None for text in cases)


def test_parse_rss_and_atom_links():
    rss = b"""
    <rss><channel><item><title>NVIDIA news</title>
    <description>Chip update</description>
    <link>https://example.com/a</link>
    <pubDate>Thu, 23 Jul 2026 12:00:00 GMT</pubDate>
    </item></channel></rss>
    """
    atom = b"""
    <feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Apple news</title>
    <summary>Device update</summary><link href="https://example.com/b"/>
    <updated>2026-07-23T12:00:00Z</updated></entry></feed>
    """
    assert parse_feed(rss)[0]["url"] == "https://example.com/a"
    assert parse_feed(atom)[0]["url"] == "https://example.com/b"


def test_rss_ambiguous_item_is_stored_unassigned_and_logged(tmp_path, monkeypatch):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    feeds.write_text(
        "feeds:\n  - id: test\n    url: https://example.com/feed\n    enabled: true\n",
        encoding="utf-8",
    )
    aliases.write_text(
        """
tickers:
  - ticker: AAPL
    strong_aliases: [iPhone]
  - ticker: META
    strong_aliases: [Instagram]
""",
        encoding="utf-8",
    )

    class Response:
        content = (
            b"<rss><channel><item><title>iPhone and Instagram</title>"
            b"<link>https://example.com/story</link></item></channel></rss>"
        )
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
    )
    counts, errors = fetcher.fetch()

    # The fixture has no publication date, so query directly for this assertion.
    with repository.connect() as connection:
        stored = dict(connection.execute("SELECT * FROM raw_items").fetchone())
    assert counts["ambiguous"] == 1
    assert errors[0]["type"] == "ambiguous_ticker"
    assert stored["ticker"] is None


def test_rss_uses_persisted_conditional_headers_and_handles_304(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    feeds.write_text(
        "feeds:\n  - id: test\n    url: https://example.com/feed\n    enabled: true\n",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")
    seen_headers = []

    class FirstResponse:
        status_code = 200
        headers = {
            "ETag": '"feed-v1"',
            "Last-Modified": "Thu, 23 Jul 2026 12:00:00 GMT",
        }
        content = b"<rss><channel></channel></rss>"

        def raise_for_status(self):
            return None

    class NotModifiedResponse:
        status_code = 304
        headers = {}
        content = b""

        def raise_for_status(self):
            return None

    responses = iter([FirstResponse(), NotModifiedResponse()])

    def fake_get(*args, **kwargs):
        seen_headers.append(kwargs["headers"])
        return next(responses)

    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=fake_get,
    )
    first_counts, first_errors = fetcher.fetch()
    second_counts, second_errors = fetcher.fetch()

    assert not first_errors
    assert not second_errors
    assert first_counts["feeds_succeeded"] == 1
    assert second_counts["feeds_not_modified"] == 1
    assert seen_headers[1]["If-None-Match"] == '"feed-v1"'
    assert seen_headers[1]["If-Modified-Since"].endswith("GMT")


def test_malformed_feed_does_not_prevent_later_feed_success(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    feeds.write_text(
        """
feeds:
  - id: broken
    url: https://example.com/broken
  - id: working
    url: https://example.com/working
""",
        encoding="utf-8",
    )
    aliases.write_text(
        """
tickers:
  - ticker: NVDA
    strong_aliases: [NVIDIA]
""",
        encoding="utf-8",
    )

    class Response:
        status_code = 200
        headers = {}

        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        if url.endswith("broken"):
            return Response(b"<not-valid")
        return Response(
            b"<rss><channel><item><title>NVIDIA update</title>"
            b"<link>https://example.com/nvidia</link></item></channel></rss>"
        )

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=fake_get,
    ).fetch()

    assert counts["feeds_succeeded"] == 1
    assert counts["inserted"] == 1
    assert errors[0]["feed"] == "broken"
    assert repository.count("raw_items") == 1


def test_duplicate_rss_item_can_be_assigned_after_initial_unmatched_insert(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    feeds.write_text(
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class Response:
        status_code = 200
        headers = {}
        content = (
            b"<rss><channel><item><title>NVIDIA update</title>"
            b"<link>https://example.com/nvidia</link></item></channel></rss>"
        )

        def raise_for_status(self):
            return None

    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
    )
    fetcher.fetch()
    aliases.write_text(
        "tickers:\n  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n",
        encoding="utf-8",
    )
    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
    )
    counts, errors = fetcher.fetch()

    with repository.connect() as connection:
        ticker = connection.execute("SELECT ticker FROM raw_items").fetchone()[0]
    assert counts["duplicates"] == 1
    assert not errors
    assert ticker == "NVDA"


def test_fifty_item_relevance_spot_check_exceeds_ninety_percent():
    aliases = load_alias_config(Path("config/aliases.yaml"))
    templates = {
        "TSLA": "Tesla Inc reports item {index} about electric vehicle revenue.",
        "NVDA": "NVIDIA reports item {index} about Blackwell GPU demand.",
        "AMD": "AMD reports item {index} about Ryzen processor demand.",
        "AAPL": "Apple Inc reports item {index} about iPhone demand.",
        "META": "Meta Platforms reports item {index} about Instagram.",
    }
    sample = [
        (ticker, template.format(index=index))
        for ticker, template in templates.items()
        for index in range(10)
    ]
    correct = sum(
        match_ticker(text, "", aliases).ticker == expected for expected, text in sample
    )
    assert correct / len(sample) >= 0.90

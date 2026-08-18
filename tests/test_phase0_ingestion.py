"""RSS ingestion and relevance (#62) against the final I1/I2 contract.

Every persistence assertion here goes through the final public surface --
``Phase0Reader`` for evidence, ``run_log_entries``/``source_state`` for the
run lifecycle.  Nothing reaches for ``repository.connect()`` or the
pre-#57 write APIs, because those are gone; where a test needs privileged
setup it says so by going through ``repository.admin``.
"""

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import yaml

from phase0.errors import Phase0RunContextError
from phase0.relevance import load_alias_config, match_ticker
from phase0.repository import Phase0Reader, Phase0Repository
from phase0.rss import (
    MAX_FEED_BYTES,
    RSSFetcher,
    STAGE_CHECKPOINT,
    STAGE_CLASSIFY,
    STAGE_FETCH,
    STAGE_INGEST,
    STAGE_RECLASSIFY,
    parse_feed,
)
from phase0.urls import canonicalize_url


#: Write APIs #57 removed.  The stale I3 branch used every one of them from
#: production code; ``test_rss_uses_no_removed_persistence_api`` is what
#: keeps them from coming back.
REMOVED_REPOSITORY_APIS = (
    "connect",
    "insert_raw_items",
    "insert_raw_items_with_classifier",
    "insert_feed_snapshot",
    "set_source_state",
    "log_stage",
    "rss_raw_items_for_reclassification",
    "replace_rss_classifications",
)


def migrated(tmp_path, name="phase0.sqlite3"):
    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    return repository


def reader(repository):
    return Phase0Reader(repository.database_path)


def runs(repository, stage=None):
    """Run-log rows as ``(stage, ticker, trading_day, status)`` tuples."""

    rows = (
        repository.run_log_entries(stage=stage)
        if stage
        else repository.run_log_entries()
    )
    return [(r["stage"], r["ticker"], r["trading_day"], r["status"]) for r in rows]


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


def _write_feeds(path, yaml_text, encoding="utf-8"):
    """Fill the production-required metadata for concise test feed fixtures."""
    config = yaml.safe_load(yaml_text)
    for feed in config["feeds"]:
        feed.setdefault("name", f"Test feed {feed.get('id', 'unknown')}")
        feed.setdefault("enabled", True)
        feed.setdefault("format", "rss2")
        feed.setdefault("intended_role", "test coverage")
        feed.setdefault(
            "expected_fields",
            {
                "title": "title",
                "url": "link",
                "description": "description",
                "published_at": "pubDate",
            },
        )
        polling = feed.setdefault("polling", {})
        polling.setdefault("interval_minutes", 30)
        polling.setdefault("conditional_get", True)
        polling.setdefault("timeout_seconds", 20)
        feed.setdefault("notes", ["Synthetic test fixture."])
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding=encoding)


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


def test_relevance_reports_exact_context_and_exclusion_evidence():
    matched = match_ticker("Apple announces changes", "Tim Cook spoke.", ALIASES)
    excluded = match_ticker("Apple harvest guide", "Try this apple pie.", ALIASES)

    assert matched.ticker == "AAPL"
    matched_rules = {
        (evidence["rule"], evidence["term"], evidence["field"])
        for decision in matched.evidence
        for evidence in decision["evidence"]
    }
    assert ("context_alias", "Apple", "title") in matched_rules
    assert ("context_term", "Tim Cook", "description") in matched_rules
    assert excluded.ticker is None
    assert excluded.evidence == (
        {
            "ticker": "AAPL",
            "decision": "excluded",
            "evidence": [
                {
                    "rule": "exclusion",
                    "term": "apple pie",
                    "field": "description",
                }
            ],
        },
    )


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


def test_parser_retains_guid_and_malformed_entries():
    rss = b"""
    <rss><channel>
      <item>
        <guid isPermaLink="false">entry-123</guid>
        <description>Evidence without display fields</description>
      </item>
      <item>
        <guid>https://publisher.example/from-guid</guid>
        <title>GUID permalink</title>
      </item>
    </channel></rss>
    """

    items = parse_feed(rss, feed_url="https://feeds.example/rss.xml")

    assert len(items) == 2
    assert items[0]["external_id"] == "entry-123"
    assert items[0]["validation_errors"] == ["missing title", "missing link"]
    assert items[1]["url"] == "https://publisher.example/from-guid"


def test_parser_resolves_relative_links_against_redirected_feed_url():
    atom = b"""
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>tag:example,2026:1</id>
        <title>Relative article</title>
        <link href="../articles/one"/>
        <updated>2026-07-23T12:00:00Z</updated>
      </entry>
    </feed>
    """

    item = parse_feed(
        atom,
        feed_url="https://cdn.example/redirected/feed.xml",
        expected_format="atom",
    )[0]

    assert item["url"] == "https://cdn.example/articles/one"
    assert item["published_at"] == "2026-07-23T12:00:00+00:00"


@pytest.mark.parametrize(
    "body,error",
    [
        (
            b'<!DOCTYPE rss [<!ENTITY x "unsafe">]><rss><channel/></rss>',
            "document types are not allowed",
        ),
        (b"<html><body>not a feed</body></html>", "unsupported feed root"),
    ],
)
def test_parser_rejects_unsafe_or_non_feed_documents(body, error):
    with pytest.raises(ValueError, match=error):
        parse_feed(body)


def test_fetcher_uses_final_response_url_for_relative_article_links(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class Response:
        status_code = 200
        headers = {}
        url = "https://cdn.example/redirected/feed.xml"
        content = (
            b"<rss><channel><item><title>Relative story</title>"
            b"<link>../articles/one?utm_source=feed&amp;ref=rss</link>"
            b"</item></channel></rss>"
        )

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()

    assert counts["inserted"] == 1
    assert not errors
    row = reader(repository).raw_items()[0]
    assert row["url"] == "https://cdn.example/articles/one?utm_source=feed&ref=rss"
    assert row["canonical_url"] == "https://cdn.example/articles/one"
    assert json.loads(row["raw_json"])["feed"]["response_url"] == Response.url


def test_rss_ambiguous_item_is_stored_unassigned_and_logged(tmp_path, monkeypatch):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
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
    stored = reader(repository).raw_items()[0]
    assert counts["ambiguous"] == 1
    assert errors[0]["type"] == "ambiguous_ticker"
    assert stored["ticker"] is None
    assert stored["ingest_status"] == "ambiguous"
    reopened = Phase0Repository(repository.database_path)
    assert reopened.count("raw_item_candidates") == 2
    read = reader(reopened)
    candidates = {row["ticker"] for row in read.raw_item_candidates()}
    candidate_reasons = [
        json.loads(row["reason"]) for row in read.raw_item_candidates()
    ]
    evidence = read.raw_item_match_evidence()
    snapshot_body = read.feed_snapshots()[0]["body"]
    raw_payload = json.loads(stored["raw_json"])
    assert candidates == {"AAPL", "META"}
    assert {evidence["ticker"] for evidence in candidate_reasons} == {"AAPL", "META"}
    assert all(evidence["evidence"] for evidence in candidate_reasons)
    assert {row["ticker"] for row in evidence} == {"AAPL", "META"}
    assert all(row["decision"] == "matched" for row in evidence)
    assert {
        item["term"] for row in evidence for item in json.loads(row["evidence"])
    } == {"iPhone", "Instagram"}
    assert snapshot_body == Response.content
    assert raw_payload["feed_snapshot_id"] > 0
    assert raw_payload["feed"]["id"] == "test"
    # There is no aggregate run to carry this: an ambiguous item belongs to
    # both tickers, so each candidate's evidence is written by that
    # candidate's own partition run and neither claims the other's.
    classify = [row for row in runs(reopened) if row[0] == STAGE_CLASSIFY]
    assert {row[1] for row in classify} == {"AAPL", "META"}
    assert all(row[3] == "success" for row in classify)
    assert [row[0] for row in runs(reopened)] == [
        STAGE_FETCH,
        STAGE_INGEST,
        STAGE_CLASSIFY,
        STAGE_CLASSIFY,
        STAGE_CHECKPOINT,
    ]


def test_rss_uses_persisted_conditional_headers_and_handles_304(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
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


def test_initial_304_is_not_treated_as_success_without_a_snapshot(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class Response:
        status_code = 304
        headers = {}
        content = b""

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()

    assert counts["feeds_succeeded"] == 0
    assert counts["feeds_partial"] == 1
    assert errors[0]["type"] == "not_modified_without_baseline"
    state = repository.source_state("rss:test")
    # Final I1 counts `partial` as a successful *check*, so `last_success_at`
    # is stamped; what must not happen is the feed looking like it has a
    # stored body.  The metadata is what the baseline test reads, and it
    # still says there is none.
    assert state["status"] == "partial"
    assert state["metadata"]["status"] == "not_modified_without_baseline"
    assert repository.count("feed_snapshots") == 0


def test_malformed_feed_does_not_prevent_later_feed_success(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
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
    assert repository.count("feed_snapshots") == 2


def test_duplicate_rss_item_can_be_assigned_after_initial_unmatched_insert(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
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

    ticker = reader(repository).raw_items()[0]["ticker"]
    assert counts["duplicates"] == 1
    assert not errors
    assert ticker == "NVDA"


def test_representative_labeled_relevance_set_exceeds_ninety_percent():
    aliases = load_alias_config(Path("config/aliases.yaml"))
    cases = json.loads(
        Path("tests/fixtures/rss_relevance_cases.json").read_text(encoding="utf-8")
    )
    correct = 0
    categories = set()
    for case in cases:
        result = match_ticker(case["title"], case["description"], aliases)
        correct += set(result.matches) == set(case["expected_matches"])
        categories.add(case["category"])

    assert len(cases) >= 50
    assert {
        "positive",
        "exclusion",
        "hard_negative",
        "ambiguous",
        "hard_positive",
        "hard_exclusion",
    } <= categories
    assert correct / len(cases) >= 0.90


@pytest.mark.parametrize(
    "yaml_text,error",
    [
        (
            """
feeds:
  - id: duplicate
    url: https://example.com/one
  - id: duplicate
    url: https://example.com/two
""",
            "duplicate feed id",
        ),
        (
            "feeds:\n  - id: insecure\n    url: http://example.com/feed\n",
            "absolute HTTPS",
        ),
        (
            """
feeds:
  - id: wrong-type
    url: https://example.com/feed
    enabled: "sometimes"
""",
            "enabled must be a boolean",
        ),
        (
            """
feeds:
  - id: bad-timeout
    url: https://example.com/feed
    polling:
      timeout_seconds: fast
""",
            "timeout_seconds must be a positive number",
        ),
    ],
)
def test_feed_config_rejects_duplicate_ids_invalid_urls_and_types(
    tmp_path, yaml_text, error
):
    feeds = tmp_path / "feeds.yaml"
    _write_feeds(feeds, yaml_text, encoding="utf-8")

    from phase0.rss import load_feed_config

    with pytest.raises(ValueError, match=error):
        load_feed_config(feeds)


def test_feed_config_requires_all_operational_and_provenance_fields(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    feeds.write_text(
        "feeds:\n  - id: incomplete\n    url: https://example.com/feed\n",
        encoding="utf-8",
    )

    from phase0.rss import load_feed_config

    with pytest.raises(ValueError, match="missing required fields"):
        load_feed_config(feeds)


def test_malformed_entries_are_preserved_and_feed_is_partial(tmp_path):
    database = tmp_path / "phase0.sqlite3"
    repository = Phase0Repository(database)
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class Response:
        status_code = 200
        headers = {"ETag": '"partial-v1"'}
        content = b"""
        <rss><channel>
          <item>
            <guid isPermaLink="false">missing-fields</guid>
            <description>Raw evidence survives</description>
          </item>
          <item>
            <guid isPermaLink="false">bad-date</guid>
            <title>Valid display fields</title>
            <link>https://publisher.example/story</link>
            <pubDate>not-a-date</pubDate>
          </item>
        </channel></rss>
        """

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()

    assert {error["type"] for error in errors} == {"invalid_entry"}
    assert counts["fetched"] == 2
    assert counts["inserted"] == 2
    assert counts["invalid"] == 2
    assert counts["feeds_partial"] == 1
    reopened = Phase0Repository(database)
    assert reopened.count("raw_items") == 2
    state = reopened.source_state("rss:test")
    assert state["metadata"]["status"] == "partial"
    # The conditional-request marker is withheld: a partial feed must not
    # let the next fetch skip a response whose entries did not all land.
    assert state["etag"] is None
    rows = reader(reopened).raw_items()
    assert {row["external_id"] for row in rows} == {
        "missing-fields",
        "bad-date",
    }
    assert all(row["ingest_status"] == "invalid" for row in rows)


def test_classifier_failure_keeps_raw_feed_evidence_and_marks_state_failed(
    tmp_path, monkeypatch
):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class Response:
        status_code = 200
        headers = {"ETag": '"v1"'}
        content = (
            b"<rss><channel><item><title>Story</title>"
            b"<link>https://example.com/story</link></item></channel></rss>"
        )

        def raise_for_status(self):
            return None

    def fail_classifier(*args, **kwargs):
        raise RuntimeError("classifier failed")

    monkeypatch.setattr("phase0.rss.match_ticker", fail_classifier)
    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()

    assert counts["feeds_succeeded"] == 0
    assert counts["feeds_failed"] == 1
    assert errors[0]["type"] == "processing_error"
    assert repository.count("raw_items") == 1
    assert repository.count("raw_item_feeds") == 1
    assert repository.count("feed_snapshots") == 1
    state = repository.source_state("rss:test")
    assert state["metadata"]["status"] == "failed"
    assert state["last_success_at"] is None


def test_transient_feed_request_retries_with_timeout_and_backoff(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        """
feeds:
  - id: test
    url: https://example.com/feed
    polling:
      timeout_seconds: 7
""",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")
    calls = []
    delays = []

    class Response:
        status_code = 200
        headers = {}
        content = b"<rss><channel></channel></rss>"

        def raise_for_status(self):
            return None

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise requests.Timeout("temporary")
        return Response()

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=fake_get,
        max_retries=2,
        retry_backoff_seconds=0.25,
        sleep=delays.append,
    ).fetch()

    assert not errors
    assert len(calls) == 3
    assert all(call["timeout"] == 7 for call in calls)
    assert all(call["stream"] is True for call in calls)
    assert delays == [0.25, 0.5]
    assert counts["retries"] == 2
    assert counts["feeds_succeeded"] == 1


def test_throttle_is_applied_between_enabled_feeds(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        """
feeds:
  - id: first
    url: https://example.com/first
  - id: disabled
    url: https://example.com/disabled
    enabled: false
  - id: second
    url: https://example.com/second
""",
        encoding="utf-8",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")
    delays = []

    class Response:
        status_code = 200
        headers = {}
        content = b"<rss><channel></channel></rss>"

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
        throttle_seconds=0.75,
        sleep=delays.append,
    ).fetch()

    assert not errors
    assert counts["feeds_succeeded"] == 2
    assert delays == [0.75]


def test_same_story_from_two_feeds_has_one_identity_and_both_provenance_records(
    tmp_path,
):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        """
feeds:
  - id: first
    url: https://feeds.example/first
  - id: second
    url: https://feeds.example/second
""",
        encoding="utf-8",
    )
    aliases.write_text(
        "tickers:\n  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n",
        encoding="utf-8",
    )

    class Response:
        status_code = 200
        headers = {}
        content = (
            b"<rss><channel><item><guid>shared-1</guid>"
            b"<title>NVIDIA update</title>"
            b"<link>https://publisher.example/shared</link>"
            b"</item></channel></rss>"
        )

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()

    assert not errors
    assert counts["inserted"] == 1
    assert counts["duplicates"] == 1
    read = reader(repository)
    sources = {row["feed_source"] for row in read.raw_item_feeds()}
    raw_source = read.raw_items()[0]["source"]
    assert repository.count("raw_items") == 1
    assert repository.count("raw_item_feeds") == 2
    assert sources == {"rss:first", "rss:second"}
    assert raw_source == "rss:publisher.example"


def test_exact_feed_response_bytes_are_preserved_losslessly(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: exact\n    url: https://feeds.example/exact\n",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")
    original_body = (
        b"<?xml version='1.0' encoding='UTF-8'?>\n"
        b"<rss><!--preserve this comment--><channel>\n"
        b"  <item data-order='first'><guid>exact-1</guid>"
        b"<title><![CDATA[Exact & untouched]]></title>"
        b"<link>https://publisher.example/exact</link></item>\n"
        b"</channel></rss>"
    )

    class Response:
        status_code = 200
        headers = {
            "Content-Type": "application/rss+xml; charset=UTF-8",
            "Content-Encoding": "identity",
        }
        content = original_body
        url = "https://cdn.example/final.xml"

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()

    assert counts["inserted"] == 1
    assert not errors
    read = reader(repository)
    snapshot = read.feed_snapshots()[0]
    provenance = read.raw_item_feeds()[0]
    assert snapshot["body"] == original_body
    assert snapshot["response_url"] == Response.url
    assert snapshot["content_type"].startswith("application/rss+xml")
    assert provenance["snapshot_id"] == snapshot["id"]
    assert provenance["external_id"] == "exact-1"


def test_content_length_limit_rejects_response_before_body_read(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: large\n    url: https://feeds.example/large\n",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class OversizedResponse:
        status_code = 200
        headers = {"Content-Length": str(MAX_FEED_BYTES + 1)}

        @property
        def content(self):
            raise AssertionError("oversized response body must not be read")

        def iter_content(self, chunk_size):
            raise AssertionError("oversized response stream must not be read")

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: OversizedResponse(),
        max_retries=0,
    ).fetch()

    assert counts["feeds_failed"] == 1
    assert "size limit" in errors[0]["error"]
    assert repository.count("feed_snapshots") == 0


def test_streaming_limit_stops_download_when_chunks_cross_bound(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: chunked\n    url: https://feeds.example/chunked\n",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")
    yielded = []

    class ChunkedResponse:
        status_code = 200
        headers = {}

        @property
        def content(self):
            raise AssertionError("streaming response must not use .content")

        def iter_content(self, chunk_size):
            for chunk in (b"x" * MAX_FEED_BYTES, b"y", b"unreachable"):
                yielded.append(len(chunk))
                yield chunk

        def raise_for_status(self):
            return None

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: ChunkedResponse(),
        max_retries=0,
    ).fetch()

    assert counts["feeds_failed"] == 1
    assert "size limit" in errors[0]["error"]
    assert yielded == [MAX_FEED_BYTES, 1]
    assert repository.count("feed_snapshots") == 0


@pytest.mark.parametrize(
    "provider_message,credential",
    [
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("Authorization: Bearer bearer-secret", "bearer-secret"),
        ("X-API-Key: api-secret", "api-secret"),
        ("Authorization: Custom opaque full value", "opaque full value"),
    ],
)
def test_rss_provider_secrets_are_redacted_everywhere(
    tmp_path, caplog, provider_message, credential
):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: secret\n    url: https://feeds.example/secret\n",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    def fail_request(*args, **kwargs):
        raise requests.ConnectionError(provider_message)

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=fail_request,
        max_retries=0,
    ).fetch()

    assert counts["feeds_failed"] == 1
    for value in (errors, caplog.text, repository.source_state("rss:secret")):
        assert credential not in str(value)
    run = repository.latest_stage_status()[0]
    assert credential not in str(run)
    # The checkpoint run *succeeded at recording a failure*, which I1 spells
    # ``degraded``; the feed's own state is what says ``failed``.  A run
    # reading ``success`` over a failed checkpoint would be the contradiction.
    assert run["stage"] == STAGE_CHECKPOINT
    assert run["status"] == "degraded"
    assert repository.source_state("rss:secret")["status"] == "failed"


def test_fetch_has_no_public_audit_bypass_and_uses_pipeline_context(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
    )
    aliases.write_text("tickers: []\n", encoding="utf-8")

    class Response:
        status_code = 200
        headers = {}
        content = b"<rss><channel></channel></rss>"

        def raise_for_status(self):
            return None

    with pytest.raises(TypeError, match="persist_run_log"):
        RSSFetcher(
            repository,
            feeds_path=feeds,
            aliases_path=aliases,
            get=lambda *args, **kwargs: Response(),
            persist_run_log=False,
        )

    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
        pipeline_version="pipeline-test",
    ).fetch(run_id="pipeline-run")

    rows = repository.run_log_entries()
    # Every run the fetch opened carries the caller's identity, the day it
    # was told to cover, and the configured pipeline version -- and each one
    # names exactly one partition, which is what a run identity means.
    assert {row["pipeline_version"] for row in rows} == {"pipeline-test"}
    # The day is the evidence's, never the caller's -- ``fetch`` takes no
    # ``trading_day`` at all, for the reason I2 states.
    assert "trading_day" not in inspect.signature(RSSFetcher.fetch).parameters
    assert all(row["run_id"].startswith("pipeline-run:") for row in rows)
    assert len({(row["run_id"], row["stage"]) for row in rows}) == len(rows)
    assert [row["stage"] for row in rows] == [STAGE_FETCH, STAGE_CHECKPOINT]


def test_offline_reclassification_replaces_only_derived_rss_state(tmp_path):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
    )
    aliases.write_text(
        "tickers:\n  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n",
        encoding="utf-8",
    )
    original_body = (
        b"<rss><channel><item><guid>story-1</guid>"
        b"<title>NVIDIA product update</title>"
        b"<link>https://publisher.example/story</link>"
        b"</item></channel></rss>"
    )

    class Response:
        status_code = 200
        headers = {}
        content = original_body

        def raise_for_status(self):
            return None

    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    ).fetch()
    read = reader(repository)
    before = read.raw_items()[0]
    snapshot_before = bytes(read.feed_snapshots()[0]["body"])
    provenance_before = read.raw_item_feeds()

    aliases.write_text(
        "tickers:\n  - ticker: AMD\n    strong_aliases: [NVIDIA]\n",
        encoding="utf-8",
    )
    network_calls = []

    def forbidden_network(*args, **kwargs):
        network_calls.append((args, kwargs))
        raise AssertionError("offline reclassification must not use the network")

    reclassifier = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=forbidden_network,
        pipeline_version="aliases-amd",
    )
    # The day is *not* a parameter -- it comes from each row's own evidence,
    # so a replay cannot re-file a row under a day it never belonged to.
    first, errors = reclassifier.reclassify_persisted(run_id="reclassify-one")
    assert not errors
    # Two decisions for one item: AMD takes it, and NVDA gives it up.
    # Withdrawing is explicit work, because only NVDA's own run is allowed
    # to clear NVDA's rows -- an omission would leave them behind forever.
    assert first == {
        "scanned": 1,
        "updated": 2,
        "assigned": 1,
        "unmatched": 0,
        "ambiguous": 0,
        "invalid": 0,
    }

    settled, errors = reclassifier.reclassify_persisted(run_id="reclassify-two")
    assert not errors
    # Nothing left to withdraw, so the replay converges: from here it is a
    # fixed point, which is what idempotent means for a replacement.
    assert settled == {
        "scanned": 1,
        "updated": 1,
        "assigned": 1,
        "unmatched": 0,
        "ambiguous": 0,
        "invalid": 0,
    }
    again, errors = reclassifier.reclassify_persisted(run_id="reclassify-three")
    assert (again, errors) == (settled, [])

    read = reader(repository)
    assigned = read.raw_items()[0]
    relevance_tickers = [
        row["ticker"]
        for row in read.raw_item_associations()
        if row["association_type"] == "relevance"
    ]
    assert assigned["id"] == before["id"]
    assert assigned["ticker"] == "AMD"
    assert assigned["ingest_status"] == "valid"
    # Raw evidence is what a replay reads, never what it writes.
    assert assigned["raw_json"] == before["raw_json"]
    assert bytes(read.feed_snapshots()[0]["body"]) == snapshot_before
    assert read.raw_item_feeds() == provenance_before
    assert relevance_tickers == ["AMD"]

    aliases.write_text(
        """
tickers:
  - ticker: AAPL
    strong_aliases: [NVIDIA]
  - ticker: META
    strong_aliases: [NVIDIA]
""",
        encoding="utf-8",
    )
    ambiguous_reclassifier = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=forbidden_network,
        pipeline_version="aliases-ambiguous",
    )
    counts, errors = ambiguous_reclassifier.reclassify_persisted(
        run_id="reclassify-ambiguous",
    )

    assert counts["ambiguous"] == 1
    assert errors == [
        {
            "type": "ambiguous_ticker",
            "raw_item_id": before["id"],
            "matches": ["AAPL", "META"],
        }
    ]
    read = reader(repository)
    row = read.raw_items()[0]
    candidates = [item["ticker"] for item in read.raw_item_candidates()]
    evidence = [item["ticker"] for item in read.raw_item_match_evidence()]
    snapshot_after = bytes(read.feed_snapshots()[0]["body"])
    assert row["ticker"] is None
    assert row["ingest_status"] == "ambiguous"
    assert row["raw_json"] == before["raw_json"]
    assert candidates == ["AAPL", "META"]
    assert evidence == ["AAPL", "META"]
    # AMD held the item before this replay and holds nothing after it: the
    # withdrawal ran in AMD's own partition.
    assert "AMD" not in candidates + evidence
    assert read.raw_item_associations() == []
    # One run per (ticker, day) partition, every replay -- never one run
    # writing across the partitions a feed happens to mention.
    replays = repository.run_log_entries(stage=STAGE_RECLASSIFY)
    assert {row["ticker"] for row in replays} == {"AMD", "NVDA", "AAPL", "META"}
    assert all(row["status"] in {"success", "degraded"} for row in replays)
    assert repository.count("raw_items") == 1
    assert repository.count("raw_item_feeds") == 1
    assert repository.count("feed_snapshots") == 1
    assert snapshot_before == snapshot_after == original_body
    assert not network_calls


def test_failed_offline_reclassification_is_logged_and_keeps_prior_state(
    tmp_path, monkeypatch
):
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n  - id: test\n    url: https://example.com/feed\n",
    )
    aliases.write_text(
        "tickers:\n  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n",
        encoding="utf-8",
    )

    class Response:
        status_code = 200
        headers = {}
        content = (
            b"<rss><channel><item><title>NVIDIA update</title>"
            b"<link>https://example.com/story</link></item></channel></rss>"
        )

        def raise_for_status(self):
            return None

    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *args, **kwargs: Response(),
        max_retries=0,
    )
    fetcher.fetch()
    monkeypatch.setattr(
        "phase0.rss.match_ticker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Authorization: Basic cmVwbGF5OnNlY3JldA==")
        ),
    )

    with pytest.raises(RuntimeError, match="Authorization"):
        fetcher.reclassify_persisted(
            run_id="reclassify-failed",
        )

    item = reader(repository).raw_items()[0]
    replays = repository.run_log_entries(stage=STAGE_RECLASSIFY)
    # The last good derived state is exactly where it was: a replay that
    # cannot classify replaces nothing, rather than clearing first and
    # failing to rebuild.
    assert item["ticker"] == "NVDA"
    assert item["ingest_status"] == "valid"
    assert reader(repository).raw_item_associations()[0]["ticker"] == "NVDA"
    # The failure is recorded, against the day whose evidence broke it and
    # under no ticker -- deciding the ticker is what failed.
    assert len(replays) == 1
    assert replays[0]["status"] == "failed"
    assert replays[0]["ticker"] is None
    assert "cmVwbGF5OnNlY3JldA==" not in str(replays[0]["errors"])
    assert repository.count("raw_items") == 1
    assert repository.count("raw_item_feeds") == 1
    assert repository.count("raw_item_match_evidence") == 1


# ---------------------------------------------------------------------------
# The final I1/I2 run contract (#83)
# ---------------------------------------------------------------------------


def _one_feed(tmp_path, aliases_text="tickers: []\n", feed_id="test"):
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        f"feeds:\n  - id: {feed_id}\n    url: https://example.com/{feed_id}\n",
    )
    aliases.write_text(aliases_text, encoding="utf-8")
    return feeds, aliases


def _responder(body, *, status_code=200, headers=None):
    class Response:
        def __init__(self):
            self.status_code = status_code
            self.headers = dict(headers or {})
            self.content = body

        def raise_for_status(self):
            return None

    return lambda *args, **kwargs: Response()


def test_rss_production_code_uses_no_removed_persistence_api():
    """The stale I3 branch called all of these; none of them exists now.

    Reading the source is the point.  A test that merely exercised the
    happy path would keep passing the day someone reached for
    ``repository.connect()`` on a branch this one never covers.
    """

    source = Path("phase0/rss.py").read_text(encoding="utf-8")
    for name in REMOVED_REPOSITORY_APIS:
        assert f".{name}(" not in source, f"phase0/rss.py still calls {name}()"
    assert "persist_run_log" not in source
    for name in REMOVED_REPOSITORY_APIS:
        assert not hasattr(Phase0Repository, name), f"{name} is back on the repository"


def test_every_rss_run_names_exactly_one_partition(tmp_path):
    """A run identity means one ticker, one day, one stage, one version.

    RSS is where this is easy to break: one response can carry several
    tickers and several days at once, and the stale implementation logged
    the whole fetch under a single identity.
    """

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(
        tmp_path,
        "tickers:\n"
        "  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n"
        "  - ticker: AMD\n    strong_aliases: [Radeon]\n",
    )
    body = (
        b"<rss><channel>"
        b"<item><guid>a</guid><title>NVIDIA launch</title>"
        b"<link>https://p.example/a</link>"
        b"<pubDate>Mon, 13 Jul 2026 10:00:00 GMT</pubDate></item>"
        b"<item><guid>b</guid><title>Radeon refresh</title>"
        b"<link>https://p.example/b</link>"
        b"<pubDate>Tue, 14 Jul 2026 10:00:00 GMT</pubDate></item>"
        b"</channel></rss>"
    )
    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(body),
        max_retries=0,
    ).fetch(run_id="multi")

    rows = repository.run_log_entries()
    identities = [(row["run_id"], row["stage"]) for row in rows]
    assert len(identities) == len(set(identities)), "an identity covered two partitions"
    partitions = {
        (row["run_id"], row["stage"]): (row["ticker"], row["trading_day"])
        for row in rows
    }
    assert len(partitions) == len(rows)

    # Two days of evidence, two ingest runs; two tickers, two classify runs.
    ingest = [row for row in rows if row["stage"] == STAGE_INGEST]
    classify = [row for row in rows if row["stage"] == STAGE_CLASSIFY]
    assert {row["trading_day"] for row in ingest} == {"2026-07-13", "2026-07-14"}
    assert {row["ticker"] for row in ingest} == {None}
    assert {(row["ticker"], row["trading_day"]) for row in classify} == {
        ("NVDA", "2026-07-13"),
        ("AMD", "2026-07-14"),
    }
    # The snapshot and the checkpoint belong to the feed, not to a ticker,
    # and both sit on the day the fetch really happened.
    fetch_day = repository.run_log_entries(stage=STAGE_FETCH)[0]["trading_day"]
    for stage in (STAGE_FETCH, STAGE_CHECKPOINT):
        feed_runs = [row for row in rows if row["stage"] == stage]
        assert [(row["ticker"], row["trading_day"]) for row in feed_runs] == [
            (None, fetch_day)
        ]


def test_no_rss_run_writes_outside_its_own_partition(tmp_path):
    """The repository refuses it, so the fetcher cannot do it by accident."""

    repository = migrated(tmp_path)
    with repository.stage_run(
        run_id="snap",
        stage=STAGE_FETCH,
        trading_day="2026-07-13",
        pipeline_version="phase0-v1",
    ) as run:
        snapshot_id = repository.record_feed_snapshot(
            feed_source="rss:test",
            response_url="https://example.com/feed",
            body=b"<rss/>",
            fetched_at="2026-07-13T10:00:00+00:00",
            run=run,
            terminal=True,
        )
    with repository.stage_run(
        run_id="ingest",
        stage=STAGE_INGEST,
        trading_day="2026-07-13",
        pipeline_version="phase0-v1",
    ) as run:
        item_id = repository.ingest_raw_items(
            [
                {
                    "source": "rss:p.example",
                    "canonical_url": "https://p.example/a",
                    "title": "NVIDIA launch",
                    "url": "https://p.example/a",
                    "fetched_at": "2026-07-13T10:00:00+00:00",
                    "raw_json": {},
                    "feed_provenance": [
                        {
                            "feed_source": "rss:test",
                            "external_id": "a",
                            "snapshot_id": snapshot_id,
                            "entry_digest": "a" * 64,
                        }
                    ],
                }
            ],
            run=run,
            terminal=True,
        )[0].item_id

    # Another ticker's decision.
    with pytest.raises(Phase0RunContextError, match="but the run covers"):
        with repository.stage_run(
            run_id="wrong-ticker",
            stage=STAGE_CLASSIFY,
            trading_day="2026-07-13",
            pipeline_version="phase0-v1",
            ticker="NVDA",
        ) as run:
            repository.replace_relevance_classifications(
                [{"raw_item_id": item_id, "ticker": "AMD"}], run=run, terminal=True
            )

    # Another day's evidence.
    with pytest.raises(Phase0RunContextError, match="another day's evidence"):
        with repository.stage_run(
            run_id="wrong-day",
            stage=STAGE_CLASSIFY,
            trading_day="2026-07-14",
            pipeline_version="phase0-v1",
            ticker="NVDA",
        ) as run:
            repository.replace_relevance_classifications(
                [{"raw_item_id": item_id, "ticker": "NVDA"}], run=run, terminal=True
            )

    # A snapshot stamped on a day the run does not cover.
    with pytest.raises(Phase0RunContextError, match="but the run covers"):
        with repository.stage_run(
            run_id="wrong-snapshot-day",
            stage=STAGE_FETCH,
            trading_day="2026-07-14",
            pipeline_version="phase0-v1",
        ) as run:
            repository.record_feed_snapshot(
                feed_source="rss:test",
                response_url="https://example.com/feed",
                body=b"<rss>x</rss>",
                fetched_at="2026-07-13T10:00:00+00:00",
                run=run,
                terminal=True,
            )

    assert reader(repository).raw_items()[0]["ticker"] is None
    assert repository.count("raw_item_match_evidence") == 0


def test_a_multi_ticker_article_keeps_one_row_and_both_associations(tmp_path):
    """Multi-ticker associations stay explicit, each written by its own run."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(
        tmp_path,
        "tickers:\n"
        "  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n"
        "  - ticker: AMD\n    strong_aliases: [Radeon]\n",
    )
    body = (
        b"<rss><channel><item><guid>both</guid>"
        b"<title>NVIDIA and Radeon both launch</title>"
        b"<link>https://p.example/both</link>"
        b"</item></channel></rss>"
    )
    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(body),
        max_retries=0,
    ).fetch()

    assert counts["ambiguous"] == 1
    assert errors[0]["type"] == "ambiguous_ticker"
    read = reader(repository)
    item = read.raw_items()[0]
    # One article, one row: ambiguity is recorded, not duplicated.
    assert repository.count("raw_items") == 1
    assert item["ticker"] is None
    assert item["ingest_status"] == "ambiguous"
    assert {row["ticker"] for row in read.raw_item_candidates()} == {"NVDA", "AMD"}
    assert {row["ticker"] for row in read.raw_item_match_evidence()} == {"NVDA", "AMD"}
    # Each ticker's evidence came from that ticker's own run.
    classify = repository.run_log_entries(stage=STAGE_CLASSIFY)
    assert {row["ticker"] for row in classify} == {"NVDA", "AMD"}


def test_an_exclusion_does_not_clear_another_tickers_assignment(tmp_path):
    """Two partitions touch one row; neither may undo the other's work."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(
        tmp_path,
        "tickers:\n"
        "  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n"
        "  - ticker: AMD\n    strong_aliases: [Radeon]\n"
        "    exclusion_terms: [NVIDIA]\n",
    )
    body = (
        b"<rss><channel><item><guid>x</guid><title>NVIDIA launch</title>"
        b"<link>https://p.example/x</link></item></channel></rss>"
    )
    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(body),
        max_retries=0,
    ).fetch()

    read = reader(repository)
    evidence = {
        row["ticker"]: row["decision"] for row in read.raw_item_match_evidence()
    }
    assert evidence == {"NVDA": "matched", "AMD": "excluded"}
    # AMD's run ran over the same row and left the assignment alone.
    assert read.raw_items()[0]["ticker"] == "NVDA"
    assert [row["ticker"] for row in read.raw_item_associations()] == ["NVDA"]


def test_a_feed_checkpoint_does_not_advance_when_evidence_fails_to_persist(
    tmp_path, monkeypatch
):
    """The marker is a promise that what came before it is stored."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(tmp_path)
    body = (
        b"<rss><channel><item><guid>x</guid><title>Story</title>"
        b"<link>https://p.example/x</link></item></channel></rss>"
    )

    def refuse(*args, **kwargs):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(Phase0Repository, "ingest_raw_items", refuse)
    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(body, headers={"ETag": '"v9"'}),
        max_retries=0,
    ).fetch()

    assert counts["feeds_failed"] == 1
    state = repository.source_state("rss:test")
    assert state["status"] == "failed"
    # No conditional marker: the next fetch must not be told it can skip a
    # response whose entries were never stored.
    assert state["etag"] is None
    assert state["last_modified"] is None
    assert repository.count("raw_items") == 0
    # The bytes, however, are already durable -- that is the whole point of
    # committing them in their own run first.
    assert repository.count("feed_snapshots") == 1


def test_reclassification_needs_no_network_and_no_working_fetcher(tmp_path):
    """Replay reads persisted evidence only; there is nothing to call out to."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(tmp_path)
    body = (
        b"<rss><channel><item><guid>x</guid><title>NVIDIA update</title>"
        b"<link>https://p.example/x</link></item></channel></rss>"
    )
    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(body),
        max_retries=0,
    ).fetch()

    aliases.write_text(
        "tickers:\n  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n",
        encoding="utf-8",
    )

    def exploding_session(*args, **kwargs):
        raise AssertionError("replay reached for the network")

    counts, errors = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=exploding_session,
    ).reclassify_persisted(run_id="offline")

    assert not errors
    assert counts["scanned"] == 1
    assert counts["assigned"] == 1
    assert reader(repository).raw_items()[0]["ticker"] == "NVDA"


def test_a_partition_that_fails_mid_replay_leaves_the_others_committed(
    tmp_path, monkeypatch
):
    """Replacement is atomic per partition, and only per partition.

    This is the boundary the docstring claims, asserted rather than
    described: a partition that fails rolls back whole, and the partitions
    that already committed stay committed.
    """

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(
        tmp_path,
        "tickers:\n"
        "  - ticker: AAPL\n    strong_aliases: [iPhone]\n"
        "  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n",
    )
    body = (
        b"<rss><channel>"
        b"<item><guid>a</guid><title>iPhone news</title>"
        b"<link>https://p.example/a</link></item>"
        b"<item><guid>n</guid><title>NVIDIA news</title>"
        b"<link>https://p.example/n</link></item>"
        b"</channel></rss>"
    )
    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(body),
        max_retries=0,
    ).fetch()
    assert {row["ticker"] for row in reader(repository).raw_item_associations()} == {
        "AAPL",
        "NVDA",
    }

    real = Phase0Repository.replace_relevance_classifications

    def fail_for_nvda(self, decisions, *, run, terminal=False):
        if run.ticker == "NVDA":
            raise RuntimeError("replay exploded for NVDA")
        return real(self, decisions, run=run, terminal=terminal)

    monkeypatch.setattr(
        Phase0Repository, "replace_relevance_classifications", fail_for_nvda
    )
    offline = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=lambda *a, **k: None,
    )
    with pytest.raises(RuntimeError, match="replay exploded"):
        offline.reclassify_persisted(run_id="partial-replay")

    read = reader(repository)
    # AAPL's partition committed; NVDA's kept the state it already had.
    assert {row["ticker"] for row in read.raw_item_associations()} == {"AAPL", "NVDA"}
    assert repository.count("raw_items") == 2
    assert repository.count("feed_snapshots") == 1
    assert repository.count("raw_item_feeds") == 2
    statuses = {
        (row["ticker"], row["status"])
        for row in repository.run_log_entries(stage=STAGE_RECLASSIFY)
    }
    assert ("NVDA", "failed") in statuses
    assert ("AAPL", "success") in statuses


# ---------------------------------------------------------------------------
# Repeated observations, authoritative evidence, and XML Base (#83 review)
# ---------------------------------------------------------------------------


UNDATED = (
    b"<rss><channel><item><guid>g1</guid><title>NVIDIA update</title>"
    b"<link>https://p.example/a</link></item></channel></rss>"
)
NVDA_ALIASES = "tickers:\n  - ticker: NVDA\n    strong_aliases: [NVIDIA]\n"


def _on_day(day):
    """Freeze the fetcher's clock, which is what dates an undated entry."""

    return patch("phase0.rss.utc_now", return_value=f"{day}T10:00:00+00:00")


def test_an_undated_entry_repeated_the_next_day_does_not_wedge_the_feed(tmp_path):
    """The article has no date, so its day comes from when it was fetched.

    Seeing it again tomorrow therefore looked like evidence that had moved
    to tomorrow, and I1 rightly refuses to let one day's run mutate another
    day's row.  The refusal was correct; re-ingesting was the mistake.  A
    feed that repeats an undated entry -- which is most feeds -- failed
    every run from the second day on, for good.
    """

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(tmp_path, NVDA_ALIASES)
    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(UNDATED, headers={"ETag": '"v1"'}),
        max_retries=0,
    )

    with _on_day("2026-07-13"):
        first, errors = fetcher.fetch(run_id="day-d")
    assert not errors
    assert first["inserted"] == 1

    for day in ("2026-07-14", "2026-07-15", "2026-07-16"):
        with _on_day(day):
            counts, errors = fetcher.fetch(run_id=f"day-{day}")
        assert not errors, f"{day} failed: {errors}"
        assert counts["feeds_failed"] == 0
        assert counts["duplicates"] == 1
        assert counts["inserted"] == 0
        # The checkpoint keeps moving; it is not stuck on a failure.
        assert repository.source_state("rss:test")["status"] == "success"

    # One row, still filed under the day it was first seen.
    assert repository.count("raw_items") == 1
    item = reader(repository).raw_items()[0]
    assert item["fetched_at"][:10] == "2026-07-13"
    assert item["ticker"] == "NVDA"


def test_a_repeated_undated_entry_from_another_feed_adds_provenance(tmp_path):
    """A second feed's later sighting is provenance, not a second article."""

    repository = migrated(tmp_path)
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n"
        "  - id: alpha\n    url: https://a.example/feed\n"
        "  - id: beta\n    url: https://b.example/feed\n",
    )
    aliases.write_text(NVDA_ALIASES, encoding="utf-8")
    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(UNDATED),
        max_retries=0,
    )

    with _on_day("2026-07-13"):
        fetcher.fetch(run_id="day-d")
    read = reader(repository)
    provenance_before = {
        (row["feed_source"], row["snapshot_id"]) for row in read.raw_item_feeds()
    }
    assert provenance_before == {("rss:alpha", 1), ("rss:beta", 2)}

    with _on_day("2026-07-14"):
        counts, errors = fetcher.fetch(run_id="day-d1")

    assert not errors
    assert counts["duplicates"] == 2
    assert repository.count("raw_items") == 1
    read = reader(repository)
    # Both feeds still recorded, each still pointing at the snapshot whose
    # bytes produced the stored text.  Provenance is as immutable as the
    # evidence it describes.
    assert {
        (row["feed_source"], row["snapshot_id"]) for row in read.raw_item_feeds()
    } == provenance_before
    assert read.raw_items()[0]["fetched_at"][:10] == "2026-07-13"
    for source in ("rss:alpha", "rss:beta"):
        assert repository.source_state(source)["status"] == "success"


def test_a_later_observation_is_recorded_in_the_items_own_day(tmp_path):
    """The run that writes it covers the day the item has always been on."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(tmp_path, NVDA_ALIASES)
    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(UNDATED),
        max_retries=0,
    )
    with _on_day("2026-07-13"):
        fetcher.fetch(run_id="day-d")
    with _on_day("2026-07-14"):
        fetcher.fetch(run_id="day-d1")

    observe = repository.run_log_entries(stage="observe_rss")
    assert [(row["ticker"], row["trading_day"]) for row in observe] == [
        (None, "2026-07-13")
    ]
    assert all(row["status"] == "success" for row in observe)
    # The snapshot and checkpoint of the *second* fetch still sit on the
    # day that fetch really happened.
    assert [
        row["trading_day"] for row in repository.run_log_entries(stage=STAGE_FETCH)
    ] == ["2026-07-13", "2026-07-14"]


def test_a_syndicated_variant_never_classifies_text_that_is_not_stored(tmp_path):
    """Feed B's words must not decide feed A's row.

    ``ingest_raw_items`` returns the stored row for a duplicate and does not
    overwrite its text, so classifying the *parsed* entry meant writing a
    ticker justified by words the row does not contain -- and the next
    offline replay, reading the row, silently reversed it.
    """

    repository = migrated(tmp_path)
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n"
        "  - id: alpha\n    url: https://a.example/feed\n"
        "  - id: beta\n    url: https://b.example/feed\n",
    )
    aliases.write_text(NVDA_ALIASES, encoding="utf-8")

    plain = (
        b"<rss><channel><item><guid>s1</guid><title>Quarterly results</title>"
        b"<link>https://p.example/story</link></item></channel></rss>"
    )
    variant = (
        b"<rss><channel><item><guid>s1</guid>"
        b"<title>NVIDIA quarterly results</title>"
        b"<link>https://p.example/story</link></item></channel></rss>"
    )

    def get(url, **kwargs):
        body = plain if "a.example" in url else variant

        class Response:
            status_code = 200
            headers: dict = {}
            content = body

            def raise_for_status(self):
                return None

        return Response()

    RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=get,
        max_retries=0,
    ).fetch(run_id="live")

    read = reader(repository)
    item = read.raw_items()[0]
    assert repository.count("raw_items") == 1
    # Alpha got there first, so the stored words are Alpha's.
    assert item["title"] == "Quarterly results"
    # And the decision is about those words, not Beta's.
    assert item["ticker"] is None
    assert read.raw_item_associations() == []
    assert read.raw_item_match_evidence() == []
    # Beta's sighting is still recorded as provenance.
    assert {row["feed_source"] for row in read.raw_item_feeds()} == {
        "rss:alpha",
        "rss:beta",
    }


@pytest.mark.parametrize(
    "first_matches", [True, False], ids=["stored-matches", "variant-matches"]
)
def test_live_classification_and_offline_replay_agree_on_a_syndicated_story(
    tmp_path, first_matches
):
    """The drift this pair of paths must never show, in both orderings.

    Only one of the two drifts, and which one depends on nothing the code
    can see: whichever feed is polled first owns the stored text.  When the
    *variant* was the one mentioning the ticker, live classification wrote
    a decision the row could not justify and the next replay took it
    straight back off.  Both paths now read the same persisted row through
    the same routine, so agreeing is structural rather than luck.
    """

    repository = migrated(tmp_path)
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    _write_feeds(
        feeds,
        "feeds:\n"
        "  - id: alpha\n    url: https://a.example/feed\n"
        "  - id: beta\n    url: https://b.example/feed\n",
    )
    aliases.write_text(NVDA_ALIASES, encoding="utf-8")

    def story(title):
        return (
            b"<rss><channel><item><guid>s1</guid><title>"
            + title
            + b"</title><link>https://p.example/story</link>"
            b"</item></channel></rss>"
        )

    matching = story(b"NVIDIA results")
    plain = story(b"Quarterly results")
    # Alpha is polled first, so Alpha's words are the ones the row keeps.
    alpha, beta = (matching, plain) if first_matches else (plain, matching)

    def get(url, **kwargs):
        body = alpha if "a.example" in url else beta

        class Response:
            status_code = 200
            headers: dict = {}
            content = body

            def raise_for_status(self):
                return None

        return Response()

    RSSFetcher(
        repository, feeds_path=feeds, aliases_path=aliases, get=get, max_retries=0
    ).fetch(run_id="live")

    def derived():
        read = reader(repository)
        item = read.raw_items()[0]
        return (
            item["title"],
            item["ticker"],
            [row["ticker"] for row in read.raw_item_associations()],
            [
                (row["ticker"], row["decision"])
                for row in read.raw_item_match_evidence()
            ],
        )

    live = derived()

    def no_network(*args, **kwargs):
        raise AssertionError("replay reached for the network")

    RSSFetcher(
        repository, feeds_path=feeds, aliases_path=aliases, get=no_network
    ).reclassify_persisted(run_id="replay")

    assert derived() == live
    # And the answer follows the stored words, whichever feed supplied them.
    assert live[0] == ("NVIDIA results" if first_matches else "Quarterly results")
    assert live[1] == ("NVDA" if first_matches else None)


def test_repeated_fetches_of_the_same_feed_are_stable(tmp_path):
    """Replaying a fetch changes nothing, so a scheduler cannot drift."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(tmp_path, NVDA_ALIASES)
    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(UNDATED),
        max_retries=0,
    )
    with _on_day("2026-07-13"):
        fetcher.fetch(run_id="first")

    def snapshot_of_everything():
        read = reader(repository)
        return (
            read.raw_items(),
            read.raw_item_feeds(),
            read.raw_item_associations(),
            read.raw_item_match_evidence(),
            read.raw_item_candidates(),
            [bytes(row["body"]) for row in read.feed_snapshots()],
        )

    before = snapshot_of_everything()
    for day, run_id in (("2026-07-14", "second"), ("2026-07-15", "third")):
        with _on_day(day):
            counts, errors = fetcher.fetch(run_id=run_id)
        # Stable because it kept working, not because it failed the same
        # way twice -- the wedged feed was "stable" in that sense too.
        assert not errors
        assert counts["feeds_succeeded"] == 1
    assert snapshot_of_everything() == before


def test_a_classifier_failure_after_a_repeat_still_keeps_the_evidence(
    tmp_path, monkeypatch
):
    """The observation path must not weaken the raw-evidence guarantee."""

    repository = migrated(tmp_path)
    feeds, aliases = _one_feed(tmp_path, NVDA_ALIASES)
    fetcher = RSSFetcher(
        repository,
        feeds_path=feeds,
        aliases_path=aliases,
        get=_responder(UNDATED),
        max_retries=0,
    )
    with _on_day("2026-07-13"):
        fetcher.fetch(run_id="first")

    monkeypatch.setattr(
        "phase0.rss.match_ticker",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with _on_day("2026-07-14"):
        counts, errors = fetcher.fetch(run_id="second")

    assert counts["feeds_failed"] == 1
    assert errors[0]["type"] == "processing_error"
    # The classifier is what broke, not the repeat: a cross-day refusal
    # here would mean this test was passing for the wrong reason.
    assert "boom" in errors[0]["error"]
    assert repository.count("raw_items") == 1
    assert repository.count("raw_item_feeds") == 1
    assert repository.count("feed_snapshots") == 1
    # The prior decision is untouched, and the checkpoint did not advance.
    assert reader(repository).raw_items()[0]["ticker"] == "NVDA"
    assert repository.source_state("rss:test")["status"] == "failed"


@pytest.mark.parametrize(
    "xml,expected",
    [
        pytest.param(
            b'<rss xml:base="https://cdn.example/a/"><channel><item>'
            b"<title>T</title><link>one</link></item></channel></rss>",
            "https://cdn.example/a/one",
            id="absolute-root-base",
        ),
        pytest.param(
            b'<rss xml:base="articles/"><channel><item>'
            b"<title>T</title><link>one</link></item></channel></rss>",
            "https://example.com/news/articles/one",
            id="relative-root-base",
        ),
        pytest.param(
            b'<rss xml:base="articles/"><channel xml:base="2026/">'
            b'<item xml:base="jul/"><title>T</title><link>one</link>'
            b"</item></channel></rss>",
            "https://example.com/news/articles/2026/jul/one",
            id="nested-relative-bases",
        ),
        pytest.param(
            b'<rss xml:base="articles/"><channel><item><title>T</title>'
            b'<link xml:base="deep/">one</link></item></channel></rss>',
            "https://example.com/news/articles/deep/one",
            id="base-on-the-link-itself",
        ),
        pytest.param(
            b"<rss><channel><item><title>T</title><link>one</link>"
            b"</item></channel></rss>",
            "https://example.com/news/one",
            id="no-base-at-all",
        ),
        pytest.param(
            b'<rss xml:base="articles/"><channel><item><title>T</title>'
            b"<link>https://other.example/x</link></item></channel></rss>",
            "https://other.example/x",
            id="absolute-href-ignores-base",
        ),
    ],
)
def test_xml_base_resolves_against_the_feed_url(xml, expected):
    """A base is relative until something makes it absolute.

    A relative root ``xml:base`` used to be treated as if it were already
    absolute, which produced a relative link that then failed validation as
    "not absolute HTTP(S)" -- the feed's own URL is what it was always
    meant to be resolved against.
    """

    entry = parse_feed(xml, feed_url="https://example.com/news/feed.xml")[0]

    assert entry["url"] == expected
    assert entry["validation_errors"] == []


def test_an_unusable_xml_base_still_fails_safely():
    """Nothing here may raise: a bad base is evidence, not a crash."""

    entry = parse_feed(
        b'<rss xml:base="articles/"><channel><item><title>T</title>'
        b"<link>one</link></item></channel></rss>",
        feed_url="not-a-url",
    )[0]

    # No absolute base to resolve against, so the link stays unusable --
    # recorded as invalid evidence rather than thrown away or crashed on.
    assert entry["validation_errors"] == ["link must resolve to absolute HTTP(S)"]

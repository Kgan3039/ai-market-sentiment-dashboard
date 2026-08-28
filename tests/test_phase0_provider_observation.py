"""The I5 provider-observation tool, exercised without a network.

The live observations this tool produces are *evidence*, and evidence is
not a unit test: it changes every time the providers publish something.
What these tests hold is the machinery that turns a payload into a
finding — extraction, stability comparison, collision reporting, source
capture, redaction, and determinism — so that a committed artifact means
what it says.

The one thing asserted about the tool's relationship to the database is
the strongest available: the fetcher it builds has no repository at all,
so a future edit that tries to persist from here fails loudly rather than
quietly writing a row.
"""

from datetime import datetime, timedelta
import itertools
import json

import pytest

from phase0.rss import RSSFetcher
from tools import observe_phase0_providers as observe


FEEDS = "config/feeds.yaml"
ALIASES = "config/aliases.yaml"
WHEN = "2026-08-22T12:00:00+00:00"


def current_item(
    article_id,
    *,
    title="Apple ships a thing",
    url=None,
    publisher="Motley Fool",
    source_id="motleyfool.com",
):
    """The payload shape yfinance returns today: ``id`` plus ``content``."""

    return {
        "id": article_id,
        "content": {
            "id": article_id,
            "contentType": "STORY",
            "title": title,
            "summary": "A standfirst.",
            "pubDate": "2026-08-22T10:00:00Z",
            "provider": {
                "displayName": publisher,
                "url": "http://www.fool.com/",
                "sourceId": source_id,
            },
            "canonicalUrl": {"url": url or f"https://www.fool.com/{article_id}"},
        },
    }


def legacy_item(uuid_value, *, title="Legacy headline", url=None):
    """The older shape ``phase0/yahoo.py`` still reads: ``uuid`` and ``link``."""

    return {
        "uuid": uuid_value,
        "title": title,
        "link": url or f"https://example.com/{uuid_value}",
        "publisher": "Example News",
        "providerPublishTime": 1_787_000_000,
    }


def observed(ticker, items, *, attempt=1, when=WHEN):
    return observe.observe_yahoo_response(
        ticker, items, attempt=attempt, observed_at=when
    )


def later(seconds, *, start=WHEN):
    """A timestamp a stated distance from the window's start."""

    return (datetime.fromisoformat(start) + timedelta(seconds=seconds)).isoformat()


def advancing_clock(step_seconds=3600, start=WHEN):
    """A deterministic clock that moves on every read.

    ``collect`` reads the clock once per attempt and once per provider, so
    a frozen clock would put every observation of an article at the same
    instant -- which is exactly the condition under which no stability
    claim can be made, and not what most of these tests are about.
    """

    counter = itertools.count()

    def clock():
        return later(step_seconds * next(counter), start=start)

    return clock


def rss_xml(*entries, channel_title="Feed"):
    body = "".join(entries)
    return (
        '<?xml version="1.0"?>'
        f'<rss version="2.0"><channel><title>{channel_title}</title>'
        f"{body}</channel></rss>"
    ).encode("utf-8")


def entry(title, link, *, guid=None, published="Fri, 22 Aug 2026 10:00:00 GMT"):
    guid_xml = f"<guid>{guid}</guid>" if guid else ""
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<description>d</description><pubDate>{published}</pubDate>"
        f"{guid_xml}</item>"
    )


@pytest.fixture
def fetcher():
    return observe.observation_fetcher(FEEDS, ALIASES)


@pytest.fixture
def feed():
    return {
        "id": "marketwatch-top-stories",
        "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "format": "rss2",
    }


# -- Candidate identifier extraction -------------------------------------


def test_the_current_payload_shape_carries_id_and_content_id():
    ids = observe.candidate_ids(current_item("199a-7bbe"))

    assert ids == {"id": "199a-7bbe", "uuid": None, "content.id": "199a-7bbe"}


def test_the_legacy_payload_shape_carries_uuid_alone():
    ids = observe.candidate_ids(legacy_item("legacy-uuid-1"))

    assert ids == {"id": None, "uuid": "legacy-uuid-1", "content.id": None}


def test_a_payload_with_no_identifier_reports_absence_rather_than_inventing_one():
    payload = current_item("x")
    del payload["id"]
    del payload["content"]["id"]

    assert observe.candidate_ids(payload) == {
        "id": None,
        "uuid": None,
        "content.id": None,
    }


def test_an_empty_identifier_is_absent_not_present():
    """``""`` is not an id.  Counting it as present would inflate coverage."""

    payload = current_item("x")
    payload["id"] = ""

    assert observe.candidate_ids(payload)["id"] is None


def test_a_non_mapping_payload_is_observed_rather_than_crashing_the_run():
    observation = observe.observe_yahoo_item(
        "AAPL", ["not", "an", "object"], attempt=1, observed_at=WHEN
    )

    assert observation.valid is False
    assert observation.candidate_ids == {"id": None, "uuid": None, "content.id": None}


def test_a_field_missing_from_every_item_is_reported_absent():
    findings = observe.summarize_candidate(
        observed("AAPL", [current_item("a")]), "uuid"
    )

    assert findings.semantics == "absent"
    assert findings.present_count == 0
    assert findings.presence_fraction == 0.0


def test_partial_coverage_is_reported_as_a_fraction_of_valid_items():
    payload = current_item("b")
    del payload["id"]
    observations = observed("AAPL", [current_item("a"), payload])

    findings = observe.summarize_candidate(observations, "id")

    assert (findings.present_count, findings.valid_item_count) == (1, 2)
    assert findings.presence_fraction == 0.5


# -- Stability across repeated observation -------------------------------


def test_one_article_seen_twice_with_one_id_is_article_scoped():
    observations = observed("AAPL", [current_item("a")], attempt=1) + observed(
        "AAPL", [current_item("a")], attempt=2, when="2026-08-22T12:05:00+00:00"
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.articles_repeated == 1
    assert findings.unstable_articles == ()
    assert findings.semantics == "article_scoped"


def test_one_article_that_changes_id_between_attempts_is_unstable():
    first = observed("AAPL", [current_item("a", url="https://www.fool.com/story")])
    second = observed(
        "AAPL",
        [current_item("b", url="https://www.fool.com/story")],
        attempt=2,
        when="2026-08-22T12:05:00+00:00",
    )

    findings = observe.summarize_candidate(first + second, "id")

    assert findings.semantics == "unstable"
    assert [row["ids"] for row in findings.unstable_articles] == [["a", "b"]]


def test_a_single_attempt_cannot_confirm_stability_and_says_so():
    """The dangerous verdict is the one that sounds fine and tested nothing."""

    findings = observe.summarize_candidate(observed("AAPL", [current_item("a")]), "id")

    assert findings.articles_repeated == 0
    assert findings.semantics == "article_scoped_unconfirmed"


def test_an_id_that_tracks_the_response_position_is_named_position_scoped():
    first = observed(
        "AAPL",
        [
            current_item("slot-0", url="https://www.fool.com/one"),
            current_item("slot-1", url="https://www.fool.com/two"),
        ],
    )
    second = observed(
        "AAPL",
        [
            current_item("slot-0", url="https://www.fool.com/two"),
            current_item("slot-1", url="https://www.fool.com/one"),
        ],
        attempt=2,
        when="2026-08-22T12:05:00+00:00",
    )

    findings = observe.summarize_candidate(first + second, "id")

    assert findings.semantics == "response_position_scoped"
    assert len(findings.position_varying_articles) == 2


# -- Cross-ticker semantics ----------------------------------------------


def test_one_article_under_two_tickers_with_one_id_is_still_article_scoped():
    shared = current_item("shared", url="https://www.fool.com/shared")
    observations = observed("AAPL", [shared]) + observed("NVDA", [shared])

    findings = observe.summarize_candidate(observations, "id")

    assert findings.cross_ticker_articles == 1
    assert findings.cross_ticker_divergent_articles == ()
    assert findings.semantics == "article_scoped"


def test_one_article_carrying_a_different_id_per_ticker_is_ticker_scoped():
    url = "https://www.fool.com/shared"
    observations = observed("AAPL", [current_item("aapl-1", url=url)]) + observed(
        "NVDA", [current_item("nvda-1", url=url)]
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.semantics == "ticker_scoped"
    assert findings.cross_ticker_divergent_articles


# -- Collisions ----------------------------------------------------------


def test_two_different_articles_sharing_an_id_are_reported_as_colliding():
    """Two publishers, two titles, one id: nothing can explain that away."""

    observations = observed(
        "AAPL",
        [
            current_item("dup", title="One", url="https://www.fool.com/one"),
            current_item(
                "dup",
                title="Two",
                url="https://www.barrons.com/two",
                publisher="Barrons.com",
                source_id="Barrons.com",
            ),
        ],
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.semantics == "colliding"
    assert findings.colliding_ids[0]["id"] == "dup"
    assert findings.colliding_ids[0]["distinct_titles"] is True
    assert findings.colliding_ids[0]["publishers"] == [
        "yahoo:Barrons.com",
        "yahoo:Motley Fool",
    ]


def test_one_publisher_reusing_an_id_across_articles_is_named_publisher_scoped():
    observations = observed(
        "AAPL",
        [
            current_item("fool", title="One", url="https://www.fool.com/one"),
            current_item("fool", title="Two", url="https://www.fool.com/two"),
        ],
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.colliding_ids[0]["publishers"] == ["yahoo:Motley Fool"]
    assert findings.semantics == "publisher_scoped"


#: The semantics that mean "this identifier is shared by two articles".
#: Which of the two is reached is a statement about *how* it is shared; both
#: are collisions and both are disqualifying.
COLLISION_SEMANTICS = {"colliding", "publisher_scoped"}


def test_one_article_seen_twice_at_one_url_with_one_title_does_not_collide():
    """The ordinary case: an article observed again is not two articles."""

    observations = observed(
        "AAPL", [current_item("same", title="One", url="https://www.fool.com/one")]
    ) + observed(
        "AAPL",
        [current_item("same", title="One", url="https://www.fool.com/one")],
        attempt=2,
        when=later(3 * 3600),
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.colliding_ids == ()
    assert findings.semantics == "article_scoped"


def test_one_article_whose_headline_was_rewritten_does_not_collide():
    """One canonical URL is one article, whatever the desk did to the headline."""

    observations = observed(
        "AAPL", [current_item("same", title="One", url="https://www.fool.com/one")]
    ) + observed(
        "AAPL",
        [current_item("same", title="One, revised", url="https://www.fool.com/one")],
        attempt=2,
        when=later(3 * 3600),
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.colliding_ids == ()
    assert findings.articles_observed == 1
    assert findings.semantics == "article_scoped"


def test_one_id_on_two_urls_collides_even_when_the_headlines_match():
    """The bug this replaces: an identical headline excused the collision.

    Two canonical URLs are two articles here -- that is the study's article
    identity -- and a recurring or templated headline is exactly what a
    real collision looks like, not a reason to look away from one.
    """

    observations = observed(
        "AAPL",
        [
            current_item(
                "same", title="Market wrap", url="https://www.fool.com/wrap-monday"
            ),
            current_item(
                "same", title="Market wrap", url="https://www.fool.com/wrap-tuesday"
            ),
        ],
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.colliding_ids[0]["id"] == "same"
    assert findings.colliding_ids[0]["distinct_titles"] is False
    assert findings.colliding_ids[0]["article_urls"] == [
        "https://www.fool.com/wrap-monday",
        "https://www.fool.com/wrap-tuesday",
    ]
    assert findings.semantics in COLLISION_SEMANTICS


def test_one_id_on_two_urls_collides_when_the_headlines_differ_too():
    observations = observed(
        "AAPL",
        [
            current_item("same", title="One", url="https://www.fool.com/one"),
            current_item(
                "same",
                title="Two",
                url="https://www.barrons.com/two",
                publisher="Barrons.com",
                source_id="Barrons.com",
            ),
        ],
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.colliding_ids[0]["distinct_titles"] is True
    assert findings.semantics in COLLISION_SEMANTICS


def test_two_ids_sharing_a_headline_across_urls_is_not_an_id_collision():
    """Collision is a claim about identifiers, not about headlines."""

    observations = observed(
        "AAPL",
        [
            current_item("first", title="Market wrap", url="https://www.fool.com/one"),
            current_item("second", title="Market wrap", url="https://www.fool.com/two"),
        ],
    )

    findings = observe.summarize_candidate(observations, "id")

    assert findings.colliding_ids == ()
    assert findings.distinct_ids == 2
    assert findings.articles_observed == 2
    assert findings.semantics not in COLLISION_SEMANTICS


def test_a_collision_disqualifies_the_field_however_long_it_was_watched():
    """No amount of stable observation makes a shared identifier a key."""

    url = "https://www.fool.com/wrap-{}"
    observations = []
    for attempt, when in enumerate([WHEN, later(20 * 3600)], start=1):
        observations += observed(
            "AAPL",
            [
                current_item("same", title="Market wrap", url=url.format("monday")),
                current_item("same", title="Market wrap", url=url.format("tuesday")),
            ],
            attempt=attempt,
            when=when,
        )
    findings = {
        field: observe.summarize_candidate(observations, field)
        for field in observe.CANDIDATE_FIELDS
    }

    verdict = observe._external_id_verdict(findings, span_seconds=20 * 3600)

    assert findings["id"].stability_span_met is True
    assert verdict["verdict"] == "UNSAFE"
    assert verdict["verdict"] != "SAFE TO IMPLEMENT"
    assert verdict["semantics"] in COLLISION_SEMANTICS


# -- Yahoo source strings ------------------------------------------------


def test_the_stored_yahoo_source_is_captured_exactly_as_ingestion_would_write_it():
    sources = observe.summarize_yahoo_sources(observed("AAPL", [current_item("a")]))

    assert [row["stored_source"] for row in sources] == ["yahoo:Motley Fool"]
    assert sources[0]["publisher_field"] == "content.provider.displayName"
    assert sources[0]["provider_source_ids"] == ["motleyfool.com"]
    assert sources[0]["article_hosts"] == ["www.fool.com"]


def test_the_legacy_publisher_field_wins_and_is_named_as_the_one_that_did():
    sources = observe.summarize_yahoo_sources(observed("AAPL", [legacy_item("u1")]))

    assert sources[0]["stored_source"] == "yahoo:Example News"
    assert sources[0]["publisher_field"] == "publisher"


def test_a_payload_with_no_publisher_records_the_fallback_string():
    payload = current_item("a")
    del payload["content"]["provider"]

    sources = observe.summarize_yahoo_sources(observed("AAPL", [payload]))

    assert sources[0]["stored_source"] == "yahoo:Yahoo Finance"
    assert sources[0]["publisher_field"] == "fallback:Yahoo Finance"


def test_an_unusable_record_is_reported_under_the_ticker_scoped_invalid_source():
    payload = current_item("a")
    del payload["content"]["canonicalUrl"]

    observations = observed("AAPL", [payload])
    invalid = observe.summarize_invalid_yahoo(observations)

    assert observations[0].valid is False
    assert invalid[0]["stored_source"] == "yahoo:AAPL"
    assert invalid[0]["candidate_ids_present"] == ["content.id", "id"]


# -- RSS source strings --------------------------------------------------


def test_the_stored_rss_source_is_the_resolved_article_host_not_the_feed_host(
    fetcher, feed
):
    body = rss_xml(
        entry("Apple rallies", "https://www.marketwatch.com/story/apple-1", guid="mw-1")
    )

    observations = observe.observe_rss_response(
        fetcher, feed, body, response_url=feed["url"], attempt=1, observed_at=WHEN
    )

    assert observations[0].stored_source == "rss:www.marketwatch.com"
    assert observations[0].resolved_host == "www.marketwatch.com"
    assert observations[0].external_id == "mw-1"
    assert "dowjones" in feed["url"]


def test_one_feed_can_produce_several_stored_sources(fetcher, feed):
    body = rss_xml(
        entry("A", "https://www.marketwatch.com/story/a", guid="a"),
        entry("B", "https://www.barrons.com/story/b", guid="b"),
    )

    observations = observe.observe_rss_response(
        fetcher, feed, body, response_url=feed["url"], attempt=1, observed_at=WHEN
    )

    assert sorted(row.stored_source for row in observations) == [
        "rss:www.barrons.com",
        "rss:www.marketwatch.com",
    ]


def test_an_unusable_entry_falls_back_to_the_feed_scoped_source(fetcher, feed):
    body = rss_xml(entry("No link", "", guid="broken"))

    observations = observe.observe_rss_response(
        fetcher, feed, body, response_url=feed["url"], attempt=1, observed_at=WHEN
    )

    assert observations[0].stored_source == "rss:marketwatch-top-stories"
    assert observations[0].resolved_host is None
    assert observations[0].ingest_status == "invalid"
    assert observe.summarize_rss_sources(observations)[0]["is_feed_scoped_fallback"]


def test_the_observation_fetcher_has_no_repository_to_write_to(fetcher):
    """Not a mock: there is nothing there, so a persist attempt raises."""

    assert isinstance(fetcher, RSSFetcher)
    assert fetcher.repository is None
    with pytest.raises(AssertionError):
        fetcher._get("https://example.com")


# -- Redaction and minimality --------------------------------------------


def test_a_credential_in_a_link_is_redacted_before_it_can_reach_the_artifact(
    fetcher, feed
):
    body = rss_xml(
        entry(
            "Leaky",
            "https://www.marketwatch.com/story/a?api_key=supersecret",
            guid="leak",
        )
    )

    observations = observe.observe_rss_response(
        fetcher, feed, body, response_url=feed["url"], attempt=1, observed_at=WHEN
    )

    assert "supersecret" not in observations[0].canonical_url
    assert "supersecret" not in (observations[0].entry_link or "")
    assert "[REDACTED]" in observations[0].canonical_url


def test_a_credential_in_a_yahoo_url_is_redacted_too():
    payload = current_item("a", url="https://www.fool.com/x?access_token=hunter2")

    observation = observed("AAPL", [payload])[0]

    assert "hunter2" not in (observation.canonical_url or "")


def test_the_observation_keeps_only_the_fields_the_conclusions_need():
    """Thumbnails, resolutions, and premium flags establish nothing here."""

    payload = current_item("a")
    payload["content"]["thumbnail"] = {"originalUrl": "https://media/x.png"}
    payload["content"]["finance"] = {"premiumFinance": {"isPremiumNews": True}}

    blob = json.dumps(observed("AAPL", [payload])[0].__dict__, default=str)

    assert "thumbnail" not in blob
    assert "premiumFinance" not in blob
    assert "media/x.png" not in blob


# -- Equivalence findings ------------------------------------------------


def build(yahoo_observations, rss_observations):
    return observe.equivalence_findings(
        observe.summarize_yahoo_sources(yahoo_observations),
        observe.summarize_rss_sources(rss_observations),
        rss_observations,
    )


def test_a_shared_publisher_host_confirms_an_equivalence(fetcher, feed):
    yahoo = observed(
        "AAPL",
        [
            current_item(
                "a",
                url="https://www.marketwatch.com/story/a",
                publisher="MarketWatch",
                source_id="marketwatch.com",
            )
        ],
    )
    rss = observe.observe_rss_response(
        fetcher,
        feed,
        rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a")),
        response_url=feed["url"],
        attempt=1,
        observed_at=WHEN,
    )

    findings, unknown = build(yahoo, rss)

    assert findings[0]["verdict"] == "CONFIRMED"
    assert findings[0]["yahoo_source"] == "yahoo:MarketWatch"
    assert findings[0]["rss_source"] == "rss:www.marketwatch.com"
    assert unknown == 0


def test_a_matching_name_with_no_article_evidence_is_only_likely(fetcher, feed):
    """Yahoo hosts the article itself, so nothing connects the two but the name."""

    yahoo = observed(
        "AAPL",
        [
            current_item(
                "a",
                url="https://finance.yahoo.com/news/a.html",
                publisher="MarketWatch",
                source_id="",
            )
        ],
    )
    rss = observe.observe_rss_response(
        fetcher,
        feed,
        rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a")),
        response_url=feed["url"],
        attempt=1,
        observed_at=WHEN,
    )

    findings, _ = build(yahoo, rss)

    assert findings[0]["verdict"] == "LIKELY_BUT_NOT_PROVEN"


def test_a_matching_name_the_provider_contradicts_is_not_equivalent(fetcher, feed):
    yahoo = observed(
        "AAPL",
        [
            current_item(
                "a",
                url="https://finance.yahoo.com/news/a.html",
                publisher="MarketWatch",
                source_id="marketwatch.co.uk",
            )
        ],
    )
    rss = observe.observe_rss_response(
        fetcher,
        feed,
        rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a")),
        response_url=feed["url"],
        attempt=1,
        observed_at=WHEN,
    )

    findings, _ = build(yahoo, rss)

    assert findings[0]["verdict"] == "NOT_EQUIVALENT"


def test_unrelated_sources_are_counted_unknown_rather_than_listed(fetcher, feed):
    yahoo = observed("AAPL", [current_item("a")])
    rss = observe.observe_rss_response(
        fetcher,
        feed,
        rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a")),
        response_url=feed["url"],
        attempt=1,
        observed_at=WHEN,
    )

    findings, unknown = build(yahoo, rss)

    assert findings == []
    assert unknown == 1


def test_the_comparable_token_is_candidate_generation_and_not_a_mapping():
    assert observe.comparable_token("Motley Fool") == "motleyfool"
    assert observe.comparable_token("24/7 Wall St.") == "247wallst"
    assert observe._host_tokens("www.marketwatch.com") == {
        "marketwatchcom",
        "marketwatch",
    }


# -- Collection, artifact, determinism -----------------------------------


class FakeResponse:
    def __init__(self, body, *, url, status_code=200):
        self.content = body
        self.url = url
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(body)), "Content-Type": "text/xml"}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.content


def clock_from(moments):
    values = iter(moments)
    last = {"value": moments[-1]}

    def clock():
        try:
            last["value"] = next(values)
        except StopIteration:
            pass
        return last["value"]

    return clock


def test_collect_separates_attempts_in_time_rather_than_firing_them_together(
    fetcher, feed
):
    slept = []
    body = rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a"))

    records, yahoo, rss = observe.collect(
        tickers=["AAPL"],
        feeds=[feed],
        fetcher=fetcher,
        attempts=3,
        interval_seconds=300,
        news_for=lambda ticker: [current_item("a")],
        feed_get=lambda url, timeout: FakeResponse(body, url=url),
        clock=lambda: WHEN,
        sleep=slept.append,
    )

    assert [record.attempt for record in records] == [1, 2, 3]
    assert slept == [300, 300]
    assert len(yahoo) == 3
    assert len(rss) == 3


def test_one_provider_failing_does_not_lose_the_other_provider_s_evidence(
    fetcher, feed
):
    body = rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a"))

    def refuse(ticker):
        raise RuntimeError("provider said no: api_key=secret")

    records, yahoo, rss = observe.collect(
        tickers=["AAPL"],
        feeds=[feed],
        fetcher=fetcher,
        attempts=1,
        interval_seconds=0,
        news_for=refuse,
        feed_get=lambda url, timeout: FakeResponse(body, url=url),
        clock=lambda: WHEN,
        sleep=lambda seconds: None,
    )

    assert yahoo == []
    assert len(rss) == 1
    assert "secret" not in records[0].yahoo["AAPL"]["error"]
    assert "[REDACTED]" in records[0].yahoo["AAPL"]["error"]


def artifact_from(fetcher, feed, *, attempts=2, items=None, step_seconds=3600):
    body = rss_xml(entry("A", "https://www.marketwatch.com/story/a", guid="a"))
    payloads = [current_item("a")] if items is None else items
    records, yahoo, rss = observe.collect(
        tickers=["AAPL"],
        feeds=[feed],
        fetcher=fetcher,
        attempts=attempts,
        interval_seconds=0,
        news_for=lambda ticker: payloads,
        feed_get=lambda url, timeout: FakeResponse(body, url=url),
        clock=advancing_clock(step_seconds),
        sleep=lambda seconds: None,
    )
    return observe.build_artifact(
        records=records,
        yahoo_observations=yahoo,
        rss_observations=rss,
        tickers=["AAPL"],
        feeds=[feed],
        feeds_path=FEEDS,
        interval_seconds=0,
        generated_at=WHEN,
        commit="0" * 40,
        dirty=False,
        yfinance_version="0.0.0",
        python_version="3.11.0",
    )


def test_the_artifact_is_byte_identical_when_the_observations_are(fetcher, feed):
    """A committed artifact has to diff as evidence, not as churn."""

    first = artifact_from(fetcher, feed)
    second = artifact_from(observe.observation_fetcher(FEEDS, ALIASES), feed)

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_artifact_records_what_produced_it(fetcher, feed):
    artifact = artifact_from(fetcher, feed)

    assert artifact["schema"] == observe.ARTIFACT_SCHEMA
    assert artifact["code"] == {"commit": "0" * 40, "dirty": False}
    assert artifact["environment"]["yfinance_version"] == "0.0.0"
    assert artifact["window"]["attempts"] == 2
    assert artifact["tickers"] == ["AAPL"]
    assert artifact["feeds"]["enabled_feed_ids"] == ["marketwatch-top-stories"]
    assert len(artifact["feeds"]["config_sha256"]) == 64


def test_the_artifact_keeps_the_records_its_conclusions_were_computed_from(
    fetcher, feed
):
    """A verdict nobody can recompute is a verdict nobody can correct."""

    artifact = artifact_from(fetcher, feed)
    records = artifact["yahoo"]["observations"]

    assert len(records) == artifact["yahoo"]["item_observation_count"]
    assert {row["observed_at"] for row in records} == {later(3600), later(14_400)}
    assert all(row["canonical_url"] for row in records)

    recomputed = observe.summarize_candidate(
        [
            observe.YahooItemObservation(
                attempt=row["attempt"],
                observed_at=row["observed_at"],
                ticker=row["ticker"],
                position=row["position"],
                candidate_ids=row["candidate_ids"],
                valid=row["valid"],
                validation_error=None,
                stored_source=row["stored_source"],
                raw_publisher=None,
                publisher_field=None,
                provider_display_name=None,
                provider_source_id=None,
                provider_url=None,
                canonical_url=row["canonical_url"],
                title=row["title"],
            )
            for row in records
        ],
        "id",
    )

    assert (
        dict(recomputed.repeat_span_summary)
        == artifact["yahoo"]["provider_id_candidates"]["id"]["repeat_span_summary"]
    )
    assert recomputed.semantics == (
        artifact["yahoo"]["provider_id_candidates"]["id"]["semantics"]
    )


def test_a_stable_fully_present_identifier_is_safe_to_implement(fetcher, feed):
    verdict = artifact_from(fetcher, feed)["yahoo"]["external_id_verdict"]

    assert verdict["verdict"] == "SAFE TO IMPLEMENT"
    assert verdict["field"] in observe.CANDIDATE_FIELDS
    assert verdict["semantics"] == "article_scoped"


def test_an_identifier_missing_from_some_items_is_only_partially_safe(fetcher, feed):
    """Stable where present is not the same claim as present."""

    carried = current_item("a", url="https://www.fool.com/a")
    missing = current_item("b", url="https://www.fool.com/b")
    del missing["id"]
    del missing["content"]["id"]

    artifact = artifact_from(fetcher, feed, items=[carried, missing])
    verdict = artifact["yahoo"]["external_id_verdict"]

    assert verdict["verdict"] == "PARTIALLY SAFE"
    assert verdict["semantics"] == "article_scoped"
    assert verdict["presence_fraction"] == 0.5


def test_an_identifier_no_item_carries_leaves_the_decision_unknown(fetcher, feed):
    payload = current_item("a")
    del payload["id"]
    del payload["content"]["id"]

    artifact = artifact_from(fetcher, feed, items=[payload])

    assert artifact["yahoo"]["external_id_verdict"]["verdict"] == "UNKNOWN"
    assert artifact["yahoo"]["external_id_verdict"]["semantics"] == "absent"


def test_the_verdict_names_the_field_it_rests_on():
    """ "Safe" with no field named is not a decision anyone can act on."""

    findings = {
        field: observe.summarize_candidate(
            observed("AAPL", [legacy_item("u1")], attempt=1)
            + observed("AAPL", [legacy_item("u1")], attempt=2, when=later(3 * 3600)),
            field,
        )
        for field in observe.CANDIDATE_FIELDS
    }

    verdict = observe._external_id_verdict(findings)

    assert verdict["field"] == "uuid"
    assert verdict["verdict"] == "SAFE TO IMPLEMENT"


def test_an_unstable_identifier_is_reported_unsafe():
    url = "https://www.fool.com/story"
    observations = observed("AAPL", [current_item("a", url=url)]) + observed(
        "AAPL", [current_item("b", url=url)], attempt=2
    )
    findings = {
        field: observe.summarize_candidate(observations, field)
        for field in observe.CANDIDATE_FIELDS
    }

    verdict = observe._external_id_verdict(findings)

    assert verdict["verdict"] == "UNSAFE"
    assert verdict["semantics"] == "unstable"


def test_the_markdown_is_rendered_from_the_artifact_and_nothing_else(fetcher, feed):
    artifact = artifact_from(fetcher, feed)

    markdown = observe.render_markdown(artifact)

    assert "# I5 provider observation" in markdown
    assert "yahoo:Motley Fool" in markdown
    assert "rss:www.marketwatch.com" in markdown
    assert artifact["yahoo"]["external_id_verdict"]["verdict"] in markdown
    assert artifact["code"]["commit"] in markdown
    for limitation in artifact["limitations"]:
        assert limitation in markdown


def test_no_rendered_line_ends_in_whitespace(fetcher, feed):
    """The artifact is committed, so the repo's whitespace checks apply to it.

    Markdown's hard break is two trailing spaces, which renders as a line
    break and reads as an error to ``git diff --check``. Every structure
    the renderer needs — nested bullets, blank lines, tables — is reachable
    without them, so the check is over the whole document rather than the
    lines that happened to break it.
    """

    markdown = observe.render_markdown(artifact_from(fetcher, feed))

    assert [line for line in markdown.splitlines() if line != line.rstrip()] == []
    # The example entries are where the hard breaks were, so a document
    # rendered without that section would pass the check above vacuously.
    assert "Example entries" in markdown
    assert "  - stored canonical URL:" in markdown


def test_every_semantics_verdict_the_summary_can_reach_has_a_stated_meaning():
    """A table cell nobody can read is not a finding."""

    assert set(observe.SEMANTICS) >= {
        "absent",
        "article_scoped",
        "article_scoped_unconfirmed",
        "ticker_scoped",
        "response_position_scoped",
        "publisher_scoped",
        "colliding",
        "unstable",
    }


# -- Stability window ----------------------------------------------------


def records_at(*moments):
    return [
        observe.AttemptRecord(attempt=index + 1, started_at=moment, yahoo={}, rss={})
        for index, moment in enumerate(moments)
    ]


def test_the_span_is_measured_from_the_attempts_not_the_configured_interval():
    """A slow or failed round moves the real span; the claim rests on that."""

    span = observe.observation_span_seconds(
        records_at(
            "2026-08-22T12:00:00+00:00",
            "2026-08-22T12:05:00+00:00",
            "2026-08-22T15:30:00+00:00",
        )
    )

    assert span == 12600.0


def test_a_single_attempt_spans_nothing():
    assert (
        observe.observation_span_seconds(records_at("2026-08-22T12:00:00+00:00")) == 0.0
    )


def repeated_at(article, *moments, ids=None, title="Headline"):
    """One article, observed at each stated moment, carrying one id."""

    observations = []
    identifiers = ids or [article] * len(moments)
    for attempt, (moment, identifier) in enumerate(zip(moments, identifiers), start=1):
        observations += observed(
            "AAPL",
            [
                current_item(
                    identifier,
                    title=title,
                    url=f"https://www.fool.com/{article}",
                )
            ],
            attempt=attempt,
            when=moment,
        )
    return observations


def verdict_for(observations, *, span_seconds=0.0):
    findings = {
        field: observe.summarize_candidate(observations, field)
        for field in observe.CANDIDATE_FIELDS
    }
    return observe._external_id_verdict(findings, span_seconds=span_seconds)


def test_a_short_window_marks_the_verdict_provisional(fetcher, feed):
    """Attempts minutes apart test a claim about minutes, however many there are."""

    artifact = artifact_from(fetcher, feed, step_seconds=60)
    stability = artifact["yahoo"]["external_id_verdict"]["stability_window"]

    assert stability["longest_repeat_span_seconds"] == 180.0
    assert stability["meets_decision_g"] is False
    assert artifact["yahoo"]["external_id_verdict"]["verdict"] == "UNKNOWN"
    assert "provisional" in observe.render_markdown(artifact)


def test_a_window_of_hours_earns_the_verdict_outright():
    verdict = verdict_for(repeated_at("a", WHEN, later(3 * 3600)), span_seconds=10_800)

    assert verdict["verdict"] == "SAFE TO IMPLEMENT"
    assert verdict["stability_window"]["meets_decision_g"] is True
    assert verdict["stability_window"]["longest_repeat_span_seconds"] == 10_800.0


def test_a_long_run_of_repeats_minutes_apart_does_not_meet_the_bar():
    """The defect this replaces: the run's own length stood in for evidence.

    Twenty hours of observation in which no article was ever seen twice
    more than ten minutes apart tests a ten-minute claim. It is reported,
    and it earns nothing.
    """

    observations = repeated_at("a", WHEN, later(600)) + repeated_at(
        "b", later(20 * 3600), later(20 * 3600 + 600)
    )

    verdict = verdict_for(observations, span_seconds=20 * 3600)
    stability = verdict["stability_window"]

    assert stability["observation_window_span_seconds"] == 72_000
    assert stability["longest_repeat_span_seconds"] == 600.0
    assert stability["repeated_article_count"] == 2
    assert stability["articles_meeting_required_span"] == 0
    assert stability["meets_decision_g"] is False
    assert verdict["verdict"] == "UNKNOWN"


def test_a_short_run_with_one_article_repeated_over_the_bar_meets_it():
    """Three hours of run, one article watched for two and a quarter of them."""

    verdict = verdict_for(
        repeated_at("a", WHEN, later(2 * 3600 + 900)), span_seconds=3 * 3600
    )
    stability = verdict["stability_window"]

    assert stability["longest_repeat_span_seconds"] == 8_100.0
    assert stability["articles_meeting_required_span"] == 1
    assert stability["meets_decision_g"] is True
    assert verdict["verdict"] == "SAFE TO IMPLEMENT"


def test_several_repeated_articles_all_under_the_bar_do_not_add_up_to_it():
    """Four articles watched for an hour each is not one article watched four."""

    observations = []
    for index, article in enumerate("abcd"):
        start = index * 4 * 3600
        observations += repeated_at(
            article, later(start), later(start + 3600), title=f"Headline {article}"
        )

    verdict = verdict_for(observations, span_seconds=16 * 3600)
    stability = verdict["stability_window"]

    assert stability["repeated_article_count"] == 4
    assert stability["articles_meeting_required_span"] == 0
    assert stability["longest_repeat_span_seconds"] == 3_600.0
    assert stability["median_repeat_span_seconds"] == 3_600.0
    assert stability["meets_decision_g"] is False
    assert verdict["verdict"] == "UNKNOWN"


def test_one_repeated_article_over_the_bar_among_short_ones_meets_it():
    observations = (
        repeated_at("a", WHEN, later(600), title="Headline a")
        + repeated_at("b", WHEN, later(1800), title="Headline b")
        + repeated_at("c", WHEN, later(4 * 3600), title="Headline c")
    )

    verdict = verdict_for(observations, span_seconds=4 * 3600)
    stability = verdict["stability_window"]

    assert stability["repeated_article_count"] == 3
    assert stability["articles_meeting_required_span"] == 1
    assert stability["longest_repeat_span_seconds"] == 14_400.0
    assert stability["shortest_repeat_span_seconds"] == 600.0
    assert stability["median_repeat_span_seconds"] == 1_800.0
    assert stability["meets_decision_g"] is True
    assert verdict["verdict"] == "SAFE TO IMPLEMENT"


def test_two_articles_are_never_added_up_into_one_stability_span():
    """A span belongs to the article it was measured on, and to no other."""

    observations = repeated_at("a", WHEN, later(300), title="Headline a") + repeated_at(
        "b", later(8 * 3600), later(8 * 3600 + 300), title="Headline b"
    )

    findings = observe.summarize_candidate(observations, "id")
    spans = {row["article_url"]: row["span_seconds"] for row in findings.repeat_spans}

    assert spans == {
        "https://www.fool.com/a": 300.0,
        "https://www.fool.com/b": 300.0,
    }
    assert findings.repeat_span_summary["longest_seconds"] == 300.0
    assert findings.stability_span_met is False


def test_an_article_that_changed_id_over_the_span_does_not_meet_the_bar():
    """Watched for three hours, and what it showed was the opposite of stability."""

    observations = repeated_at("a", WHEN, later(3 * 3600), ids=["first", "second"])

    findings = observe.summarize_candidate(observations, "id")

    assert findings.repeat_spans[0]["span_seconds"] == 10_800.0
    assert findings.repeat_spans[0]["one_identifier"] is False
    assert findings.repeat_span_summary["repeated_article_count"] == 1
    assert findings.repeat_span_summary["meeting_required_span"] == 0
    assert findings.stability_span_met is False
    assert findings.semantics == "unstable"


def test_the_run_span_is_reported_beside_the_verdict_and_never_folded_into_it():
    """The run's length may be published; it may not decide anything."""

    observations = repeated_at("a", WHEN, later(3 * 3600))

    short = verdict_for(observations, span_seconds=60)
    long = verdict_for(observations, span_seconds=86_400)

    assert short["verdict"] == long["verdict"] == "SAFE TO IMPLEMENT"
    assert short["field"] == long["field"]
    assert short["stability_window"]["meets_decision_g"] is True
    assert long["stability_window"]["meets_decision_g"] is True
    assert short["stability_window"]["observation_window_span_seconds"] == 60
    assert long["stability_window"]["observation_window_span_seconds"] == 86_400


# -- Candidate agreement -------------------------------------------------


def test_two_candidates_carrying_one_value_are_reported_as_one_fact():
    rows = observe.candidate_agreement(observed("AAPL", [current_item("a")]))
    pair = next(row for row in rows if row["fields"] == ["id", "content.id"])

    assert pair["both_present_count"] == 1
    assert pair["agreed_count"] == 1
    assert pair["disagreements"] == []


def test_two_candidates_that_diverge_show_the_divergence():
    payload = current_item("a")
    payload["content"]["id"] = "different"

    rows = observe.candidate_agreement(observed("AAPL", [payload]))
    pair = next(row for row in rows if row["fields"] == ["id", "content.id"])

    assert pair["agreed_count"] == 0
    assert pair["disagreements"][0]["id"] == "a"
    assert pair["disagreements"][0]["content.id"] == "different"


def test_candidates_never_both_present_are_reported_as_such():
    rows = observe.candidate_agreement(observed("AAPL", [legacy_item("u1")]))
    pair = next(row for row in rows if row["fields"] == ["id", "uuid"])

    assert pair["both_present_count"] == 0


def test_a_tie_between_candidates_recommends_the_field_the_code_already_reads():
    """`id` and `content.id` are equally clean here; only one is expected."""

    observations = observed("AAPL", [current_item("a")], attempt=1) + observed(
        "AAPL", [current_item("a")], attempt=2
    )
    findings = {
        field: observe.summarize_candidate(observations, field)
        for field in observe.CANDIDATE_FIELDS
    }

    assert findings["id"].semantics == findings["content.id"].semantics
    assert observe._external_id_verdict(findings)["field"] == "id"


def test_the_agreement_is_rendered_for_a_reviewer(fetcher, feed):
    artifact = artifact_from(fetcher, feed)

    markdown = observe.render_markdown(artifact)

    assert "Do the candidates agree?" in markdown
    assert "carried the same value" in markdown


# -- Candidate selection -------------------------------------------------


def split_item(article_id, *, top_level=True, content=True, url=None, title=None):
    """A payload that carries only some of the candidate identifiers.

    Yahoo has sent ``id`` and ``content.id`` identical in every window
    observed so far, so no real artifact separates them. That is exactly
    why both are observed, and why the ranking has to be right on the day
    a payload stops sending both.
    """

    item = current_item(
        article_id,
        url=url or f"https://www.fool.com/{article_id}",
        **({"title": title} if title else {}),
    )
    if not content:
        item["content"].pop("id")
    if not top_level:
        item.pop("id")
    return item


def article_seen(article, moments, *, top_level=True, content=True, ids=None):
    """One article, observed at each moment, carrying the named candidates."""

    identifiers = ids or [article] * len(moments)
    observations = []
    for attempt, (moment, identifier) in enumerate(zip(moments, identifiers), start=1):
        observations += observed(
            "AAPL",
            [
                split_item(
                    identifier,
                    top_level=top_level,
                    content=content,
                    url=f"https://www.fool.com/{article}",
                )
            ],
            attempt=attempt,
            when=moment,
        )
    return observations


def finding(
    field,
    *,
    semantics="article_scoped",
    presence=1.0,
    meets_bar=True,
    repeated=1,
    longest=10_800.0,
    collisions=0,
    unstable=0,
):
    """A hand-built summary, for candidate shapes observation cannot produce.

    Full coverage means a field was carried by every observation of every
    article, so it inherits the longest span any candidate has: a
    full-coverage candidate cannot miss decision G's bar while a narrower
    one clears it — ``test_a_candidate_on_every_item_cannot_miss_a_bar_a_
    narrower_one_clears`` holds that. The shape is specified anyway,
    because ranking is a claim about evidence rather than about Yahoo's
    current payload, and it has to hold when the two diverge.
    """

    spans = {
        "required_span_seconds": observe.DECISION_G_STABILITY_SECONDS,
        "repeated_article_count": repeated,
        "meeting_required_span": 1 if meets_bar else 0,
        "longest_seconds": longest,
        "shortest_seconds": longest,
        "median_seconds": longest,
    }
    return observe.CandidateFindings(
        field=field,
        valid_item_count=10,
        present_count=round(presence * 10),
        presence_fraction=presence,
        distinct_ids=repeated,
        articles_observed=repeated,
        articles_repeated=repeated,
        articles_at_multiple_positions=0,
        cross_ticker_articles=0,
        repeat_spans=(),
        repeat_span_summary=spans,
        stability_span_met=meets_bar,
        unstable_articles=({"article_url": "u"},) * unstable,
        position_varying_articles=(),
        cross_ticker_divergent_articles=(),
        colliding_ids=({"id": "x"},) * collisions,
        semantics=semantics,
        evidence=(),
    )


ABSENT = finding("uuid", semantics="absent", presence=0.0, meets_bar=False, repeated=0)


def test_full_coverage_without_the_stability_evidence_loses_to_partial_with_it():
    """Coverage is not evidence of stability, and cannot stand in for it."""

    verdict = observe._external_id_verdict(
        {
            "id": finding("id", presence=1.0, meets_bar=False),
            "content.id": finding("content.id", presence=0.6, meets_bar=True),
            "uuid": ABSENT,
        }
    )

    assert verdict["field"] == "content.id"
    assert verdict["verdict"] == "PARTIALLY SAFE"


def test_full_coverage_on_both_prefers_the_candidate_that_cleared_the_bar():
    verdict = observe._external_id_verdict(
        {
            "id": finding("id", presence=1.0, meets_bar=False),
            "content.id": finding("content.id", presence=1.0, meets_bar=True),
            "uuid": ABSENT,
        }
    )

    assert verdict["field"] == "content.id"
    assert verdict["verdict"] == "SAFE TO IMPLEMENT"


def test_a_candidate_on_every_item_cannot_miss_a_bar_a_narrower_one_clears():
    """Why the two tests above are hand-built rather than observed."""

    observations = article_seen("long", [WHEN, later(3 * 3600)]) + article_seen(
        "short", [WHEN, later(600)], content=False
    )

    everywhere = observe.summarize_candidate(observations, "id")
    narrower = observe.summarize_candidate(observations, "content.id")

    assert everywhere.presence_fraction == 1.0
    assert narrower.presence_fraction == 0.5
    assert narrower.stability_span_met
    assert everywhere.stability_span_met


def test_a_wider_candidate_that_proved_nothing_loses_to_a_narrower_one_that_did():
    """The failure shape, observed: the wider field is watched for minutes."""

    observations = article_seen(
        "short", [WHEN, later(200), later(400), later(600)], content=False
    ) + article_seen("long", [WHEN, later(3 * 3600)], top_level=False)

    verdict = verdict_for(observations)

    assert verdict["selection"]["considered"][0]["field"] == "content.id"
    assert verdict["field"] == "content.id"
    assert verdict["verdict"] == "PARTIALLY SAFE"


def test_between_two_qualified_candidates_the_wider_one_wins():
    observations = article_seen("long", [WHEN, later(3 * 3600)]) + article_seen(
        "short", [WHEN, later(600)], content=False
    )

    verdict = verdict_for(observations)

    assert verdict["field"] == "id"
    assert verdict["verdict"] == "SAFE TO IMPLEMENT"


def test_two_qualified_candidates_with_equal_coverage_break_toward_id():
    """Deterministic, and toward the field phase0 already reads."""

    verdict = verdict_for(article_seen("a", [WHEN, later(3 * 3600)]))

    assert verdict["field"] == "id"
    assert "tie-break" in verdict["selection"]["reason"]


def test_a_colliding_candidate_never_outranks_a_qualified_one():
    """100% coverage does not buy a shared identifier a recommendation."""

    shared = observed(
        "AAPL",
        [
            split_item("x", content=False, url="https://www.fool.com/a"),
            split_item("x", content=False, url="https://www.fool.com/b"),
        ],
        attempt=1,
    )
    observations = shared + article_seen("c", [WHEN, later(3 * 3600)])

    findings = {
        field: observe.summarize_candidate(observations, field)
        for field in observe.CANDIDATE_FIELDS
    }
    verdict = observe._external_id_verdict(findings)

    assert findings["id"].presence_fraction == 1.0
    assert findings["id"].semantics in COLLISION_SEMANTICS
    assert verdict["field"] == "content.id"
    assert verdict["verdict"] == "PARTIALLY SAFE"


def test_an_unqualified_candidate_with_no_qualified_alternative_stays_unknown():
    verdict = verdict_for(article_seen("a", [WHEN, later(600)]))

    assert verdict["field"] == "id"
    assert verdict["verdict"] == "UNKNOWN"
    assert all(
        row["meets_decision_g"] is False for row in verdict["selection"]["considered"]
    )


def test_when_no_candidate_is_article_scoped_the_verdict_is_never_safe():
    verdict = verdict_for(article_seen("a", [WHEN, later(3 * 3600)], ids=["1", "2"]))

    assert verdict["verdict"] == "UNSAFE"
    assert verdict["semantics"] == "unstable"


def test_the_verdict_reports_every_candidate_it_ranked_and_why_one_won():
    """A verdict that names only its winner cannot be audited."""

    observations = article_seen(
        "short", [WHEN, later(600)], content=False
    ) + article_seen("long", [WHEN, later(3 * 3600)], top_level=False)

    selection = verdict_for(observations)["selection"]
    rows = {row["field"]: row for row in selection["considered"]}

    assert [row["field"] for row in selection["considered"]] == [
        "content.id",
        "id",
        "uuid",
    ]
    assert rows["content.id"]["meets_decision_g"] is True
    assert rows["id"]["meets_decision_g"] is False
    assert rows["id"]["longest_repeat_span_seconds"] == 600.0
    assert rows["uuid"]["semantics"] == "absent"
    assert "decision G asks for" in selection["reason"]


def test_the_ranking_is_rendered_for_a_reviewer(fetcher, feed):
    artifact = artifact_from(fetcher, feed)
    verdict = artifact["yahoo"]["external_id_verdict"]

    markdown = observe.render_markdown(artifact)

    assert "Which candidate the verdict rests on" in markdown
    assert verdict["selection"]["reason"] in markdown
    for row in verdict["selection"]["considered"]:
        assert f"{row['rank'] + 1}. `{row['field']}`" in markdown


# -- What the selection reason is allowed to claim -----------------------


# Phrases that assert the two candidates were found to be the same thing.
# Ranking cannot establish any of them: it can only establish that no
# dimension it looked at put the runner-up first.
EQUALITY_CLAIMS = ("are both", "equally scoped", "indistinguishable")


def reason_for(**semantics_and_findings):
    """The selection decision for a hand-built pair, plus the absent third."""

    verdict = observe._external_id_verdict({**semantics_and_findings, "uuid": ABSENT})
    return verdict["selection"], verdict


def test_two_classifications_that_rank_alike_are_not_reported_as_one():
    """``response_position_scoped`` and ``ticker_scoped`` share a rank.

    They share it because neither can key a raw item, so ranking has no
    reason to prefer either — not because they are the same finding. A
    reason that read "both are response_position_scoped" off the tie would
    tell a reviewer something the observations never said.
    """

    selection, _ = reason_for(
        **{
            "id": finding("id", semantics="response_position_scoped", meets_bar=True),
            "content.id": finding(
                "content.id", semantics="ticker_scoped", meets_bar=False
            ),
        }
    )
    rows = {row["field"]: row for row in selection["considered"]}

    assert rows["id"]["semantics_rank"] == rows["content.id"]["semantics_rank"]
    assert rows["id"]["semantics"] != rows["content.id"]["semantics"]
    assert "response_position_scoped" in selection["reason"]
    assert "ticker_scoped" in selection["reason"]
    assert not any(claim in selection["reason"] for claim in EQUALITY_CLAIMS)
    assert selection["decided_by"] == "stability"


def test_a_shared_rank_still_reports_the_dimension_that_decided_it():
    """Naming both classifications does not excuse leaving the reason vague."""

    selection, _ = reason_for(
        **{
            "id": finding("id", semantics="response_position_scoped", presence=1.0),
            "content.id": finding(
                "content.id", semantics="ticker_scoped", presence=0.5
            ),
        }
    )

    assert selection["decided_by"] == "coverage"
    assert "100.0%" in selection["reason"] and "50.0%" in selection["reason"]
    assert not any(claim in selection["reason"] for claim in EQUALITY_CLAIMS)


def test_candidates_alike_in_every_dimension_but_classification_are_not_tied():
    """Same rank, same bar, same coverage — and still two different findings."""

    selection, _ = reason_for(
        **{
            "id": finding("id", semantics="response_position_scoped"),
            "content.id": finding("content.id", semantics="ticker_scoped"),
        }
    )

    assert selection["decided_by"] == "field_order"
    assert "field-order tie-break" in selection["reason"]
    assert "response_position_scoped" in selection["reason"]
    assert "ticker_scoped" in selection["reason"]
    assert "indistinguishable" not in selection["reason"]


def test_the_same_classification_twice_is_reported_as_the_same():
    """The reverse guard: equal labels may be, and are, stated as equal."""

    selection, _ = reason_for(
        **{
            "id": finding("id", meets_bar=True),
            "content.id": finding("content.id", meets_bar=False),
        }
    )

    assert selection["decided_by"] == "stability"
    assert "'id' and 'content.id' are both article_scoped" in selection["reason"]
    assert "decision G asks for" in selection["reason"]


def test_coverage_is_named_when_scope_and_the_bar_leave_nothing_to_choose():
    selection, _ = reason_for(
        **{
            "id": finding("id", presence=1.0),
            "content.id": finding("content.id", presence=0.4),
        }
    )

    assert selection["decided_by"] == "coverage"
    assert "both cleared decision G's 7200s bar" in selection["reason"]
    assert "100.0% of valid items against 40.0%" in selection["reason"]


def test_the_field_order_tie_break_is_named_when_it_is_what_decided():
    """The one branch where the reason is about the tool, not the provider."""

    selection, _ = reason_for(
        **{"id": finding("id"), "content.id": finding("content.id")}
    )

    assert selection["decided_by"] == "field_order"
    assert "indistinguishable on this evidence" in selection["reason"]
    assert "deterministic field-order tie-break" in selection["reason"]


def test_a_safer_classification_is_reported_with_both_classifications():
    selection, _ = reason_for(
        **{
            "id": finding("id", semantics="article_scoped"),
            "content.id": finding("content.id", semantics="publisher_scoped"),
        }
    )
    rows = {row["field"]: row for row in selection["considered"]}

    assert rows["id"]["semantics_rank"] < rows["content.id"]["semantics_rank"]
    assert selection["decided_by"] == "semantics"
    assert "article_scoped" in selection["reason"]
    assert "publisher_scoped" in selection["reason"]
    assert not any(claim in selection["reason"] for claim in EQUALITY_CLAIMS)


def test_the_only_candidate_is_reported_as_the_only_candidate():
    verdict = observe._external_id_verdict({"id": finding("id")})

    assert verdict["selection"]["decided_by"] == "only_candidate"
    assert "only candidate" in verdict["selection"]["reason"]


def test_the_rendered_reason_is_the_reason_the_artifact_recorded(fetcher, feed):
    """Markdown carries the decision through unedited, dimension and all."""

    artifact = artifact_from(fetcher, feed)
    selection = artifact["yahoo"]["external_id_verdict"]["selection"]

    markdown = observe.render_markdown(artifact)

    assert selection["reason"] in markdown
    assert f"decided on {selection['decided_by'].replace('_', ' ')}" in markdown
    for row in selection["considered"]:
        assert f"`{row['semantics']}`" in markdown


# -- Recomputing a committed artifact ------------------------------------


def test_a_recompute_reaches_the_findings_the_run_reached(fetcher, feed):
    """The retained records are enough to re-derive what they were used for."""

    artifact = artifact_from(fetcher, feed)

    again = observe.recompute_artifact(artifact, at="2026-08-24T00:00:00+00:00")

    assert again["yahoo"]["provider_id_candidates"] == (
        artifact["yahoo"]["provider_id_candidates"]
    )
    assert again["yahoo"]["external_id_verdict"] == (
        artifact["yahoo"]["external_id_verdict"]
    )
    assert again["yahoo"]["sources"] == artifact["yahoo"]["sources"]
    assert again["rss"] == artifact["rss"]
    assert again["attempts"] == artifact["attempts"]
    assert again["recomputed"]["from_records"] == "yahoo.observations"
    assert again["recomputed"]["at"] == "2026-08-24T00:00:00+00:00"


def test_a_recompute_refuses_a_record_set_that_lost_rows(fetcher, feed):
    """Numbers from a truncated record set would describe neither window."""

    artifact = artifact_from(fetcher, feed)
    artifact["yahoo"]["observations"].pop()

    with pytest.raises(ValueError, match="insufficient"):
        observe.recompute_artifact(artifact, at=WHEN)

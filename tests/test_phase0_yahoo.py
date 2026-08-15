"""Yahoo ingestion (#61) against the final Phase 0 persistence contract (#57).

The tests that used to reach into ``repository.connect()`` and assert on
``log_stage``/``insert_raw_items``/``set_source_state`` are gone: those
APIs no longer exist.  Reads here go through the public repository methods
and :class:`~phase0.repository.Phase0Reader`; the durable audit is checked
where I1 puts it, in ``run_log`` rows written inside the same transaction
as the data.
"""

from datetime import datetime, timedelta, timezone
import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from phase0.errors import Phase0RunContextError
from phase0.repository import Phase0Reader, Phase0Repository, StageRunContext
from phase0.tickers import TICKER_UNIVERSE
from phase0 import yahoo as yahoo_module
from phase0.yahoo import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    SHARED_PROVIDER_GATE,
    STAGE,
    TICKERS,
    YahooFinanceFetcher,
    YahooProviderBusyError,
    YahooProviderGate,
    effective_day,
    normalize_yahoo_item,
    partition_run_id,
)


#: Public repository attributes #57 removed.  Yahoo used every one of them.
REMOVED_REPOSITORY_APIS = (
    "connect",
    "insert_raw_items",
    "set_source_state",
    "log_stage",
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def today() -> str:
    return now().date().isoformat()


def epoch(moment: datetime) -> int:
    return int(moment.timestamp())


def legacy_item(ticker, **updates):
    """A yfinance record published *today*, so it settles on the fetch day."""

    item = {
        "title": f"{ticker} headline",
        "link": f"https://example.com/{ticker}?utm_source=yahoo",
        "publisher": "Example News",
        "providerPublishTime": epoch(now()),
    }
    item.update(updates)
    return item


def migrated(tmp_path, name="phase0.sqlite3"):
    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    return repository


def fetcher(repository, factory, **options):
    options.setdefault("max_retries", 0)
    # A private gate per fetcher: the production default is the process-wide
    # one, and tests that leave a request outstanding must not be able to
    # hand their provider's answer to the next test that asks for the same
    # ticker.  ``test_a_fetcher_built_without_a_gate_shares_the_process_one``
    # is what holds the default itself.
    options.setdefault("provider_gate", YahooProviderGate())
    return YahooFinanceFetcher(repository, ticker_factory=factory, **options)


@pytest.fixture(autouse=True)
def shared_gate_is_left_clean():
    """No test may abandon work on the process-wide gate."""

    yield
    assert SHARED_PROVIDER_GATE.outstanding == 0


def wait_for(predicate, timeout=5.0):
    """Poll *predicate* rather than sleeping a guessed interval."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def provider_threads():
    """Live threads actually running Yahoo provider work, by name."""

    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("yahoo-provider") and thread.is_alive()
    ]


class HangingProvider:
    """A provider whose calls block until the test lets them finish.

    This is the shape of the defect under repair: a call that outlives the
    caller's patience.  ``entered`` counts calls that really reached the
    provider body, so tests can distinguish "a second request was issued"
    from "a second caller joined the first request".
    """

    def __init__(self, *, raises=None, news=None):
        self.calls = []
        self.released = threading.Event()
        self._entered = threading.Semaphore(0)
        self._lock = threading.Lock()
        self._raises = raises
        self._news = news

    def __call__(self, ticker):
        with self._lock:
            self.calls.append(ticker)
        self._entered.release()
        if not self.released.wait(10):  # pragma: no cover - only on a hang
            raise AssertionError("hanging provider was never released")
        if self._raises is not None:
            raise self._raises
        news = self._news if self._news is not None else [legacy_item(ticker)]
        return SimpleNamespace(news=list(news))

    def wait_until_entered(self, count=1, timeout=5.0):
        for _ in range(count):
            assert self._entered.acquire(timeout=timeout), "provider was never called"


@pytest.fixture
def hanging():
    """Build hanging providers and guarantee their workers are let go."""

    built = []

    def build(**options):
        built.append(HangingProvider(**options))
        return built[-1]

    yield build
    for provider_ in built:
        provider_.released.set()
    assert wait_for(
        lambda: not provider_threads()
    ), "a provider worker outlived its test"


def provider(news_by_ticker=None, *, news=None, raises=None):
    """A fake ``yfinance.Ticker`` recording the symbols it was asked for."""

    calls: list[str] = []

    class FakeTicker:
        def __init__(self, ticker):
            calls.append(ticker)
            if raises is not None:
                raise raises
            if news_by_ticker is not None:
                self.news = news_by_ticker.get(ticker, [])
            else:
                self.news = list(news) if news is not None else [legacy_item(ticker)]

    FakeTicker.calls = calls
    return FakeTicker


def partitions(repository):
    """Every ``fetch_yahoo`` run log, as ``(ticker, day) -> row``."""

    rows = repository.run_log_entries(stage=STAGE)
    return {(row["ticker"], row["trading_day"]): row for row in rows}


# ---------------------------------------------------------------------------
# The five approved tickers
# ---------------------------------------------------------------------------


def test_the_default_universe_is_the_five_approved_symbols_in_spec_order():
    # Not ``sorted(SUPPORTED_TICKERS)``: alphabetical order would be an
    # ordering invented in this module rather than the shared one.
    assert TICKERS == tuple(TICKER_UNIVERSE)
    assert TICKERS == ("TSLA", "NVDA", "AMD", "AAPL", "META")


def test_every_approved_ticker_is_fetched_and_stored(tmp_path):
    repository = migrated(tmp_path)
    factory = provider()

    counts, errors = fetcher(repository, factory).fetch()

    assert factory.calls == list(TICKERS)
    assert counts["inserted"] == 5
    assert counts["tickers_succeeded"] == 5
    assert not errors
    stored = {row["ticker"] for row in repository.raw_items_for_day(today())}
    assert stored == set(TICKERS)


def test_an_unapproved_ticker_is_rejected_without_calling_the_provider(tmp_path):
    repository = migrated(tmp_path)
    factory = provider()

    counts, errors = fetcher(repository, factory).fetch(["GOOG"])

    assert factory.calls == []
    assert counts["tickers_rejected"] == 1
    assert errors == [{"ticker": "GOOG", "error": "unsupported Yahoo ticker"}]
    assert repository.source_state("yahoo:GOOG") is None
    assert repository.count("run_log") == 0


def test_a_ticker_is_normalized_before_the_provider_and_the_partition(tmp_path):
    repository = migrated(tmp_path)
    factory = provider()

    counts, errors = fetcher(repository, factory).fetch([" nvDa "])

    assert factory.calls == ["NVDA"]
    assert counts["inserted"] == 1
    assert not errors
    assert list(partitions(repository)) == [("NVDA", today())]


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------


def test_the_publisher_canonical_url_wins_over_the_yahoo_click_through():
    item = normalize_yahoo_item(
        "NVDA",
        {
            "link": "https://finance.yahoo.com/click-through",
            "content": {
                "title": "NVIDIA headline",
                "summary": "Company update",
                "canonicalUrl": {
                    "url": "https://publisher.example/story?utm_source=yahoo"
                },
                "provider": {"displayName": "Example News"},
                "pubDate": "2026-07-23T12:00:00Z",
            },
        },
    )

    assert item["ticker"] == "NVDA"
    assert item["source"] == "yahoo:Example News"
    assert item["url"] == "https://publisher.example/story"
    assert item["canonical_url"] == "https://publisher.example/story"
    assert item["published_at"] == "2026-07-23T12:00:00+00:00"


def test_a_yahoo_redirect_is_unwrapped():
    item = normalize_yahoo_item(
        "AMD",
        legacy_item(
            "AMD",
            link=(
                "https://r.search.yahoo.com/redirect?"
                "url=https%3A%2F%2Fpublisher.example%2Fstory%3Futm_source%3Dyahoo"
            ),
        ),
    )

    assert item["canonical_url"] == "https://publisher.example/story"


@pytest.mark.parametrize(
    "link,expected",
    [
        (
            "https://publisher.example/story?utm_source=yahoo&utm_medium=rss",
            "https://publisher.example/story",
        ),
        (
            "https://publisher.example/story?fbclid=abc&id=7",
            "https://publisher.example/story?id=7",
        ),
        ("HTTPS://Publisher.Example/Story/", "https://publisher.example/Story"),
        ("https://publisher.example:443/story", "https://publisher.example/story"),
        (
            "https://publisher.example/story?b=2&a=1",
            "https://publisher.example/story?a=1&b=2",
        ),
    ],
)
def test_tracking_and_ordering_noise_is_canonicalized_away(link, expected):
    item = normalize_yahoo_item("NVDA", legacy_item("NVDA", link=link))

    assert item["canonical_url"] == expected


def test_two_spellings_of_one_article_ingest_as_one_row(tmp_path):
    repository = migrated(tmp_path)
    factory = provider(
        news=[
            legacy_item("NVDA", link="https://publisher.example/story?utm_source=x"),
            legacy_item("NVDA", link="https://publisher.example/story/"),
        ]
    )

    counts, _ = fetcher(repository, factory).fetch(["NVDA"])

    assert counts["fetched"] == 2
    assert counts["inserted"] == 1
    assert counts["duplicates"] == 1
    assert repository.count("raw_items") == 1


# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 1_000, 1_000_000])
def test_second_millisecond_and_microsecond_epochs_agree(scale):
    item = normalize_yahoo_item(
        "NVDA", legacy_item("NVDA", providerPublishTime=1784808000 * scale)
    )

    assert (
        item["published_at"]
        == datetime.fromtimestamp(1784808000, tz=timezone.utc).isoformat()
    )


@pytest.mark.parametrize(
    "published",
    [
        "2026-07-23T12:00:00Z",
        "2026-07-23T12:00:00+00:00",
        "2026-07-23T14:00:00+02:00",
        "Thu, 23 Jul 2026 12:00:00 +0000",
        datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc),
    ],
)
def test_every_accepted_timestamp_shape_lands_on_the_same_utc_instant(published):
    item = normalize_yahoo_item(
        "NVDA", legacy_item("NVDA", providerPublishTime=published)
    )

    assert item["published_at"] == "2026-07-23T12:00:00+00:00"


def test_a_naive_timestamp_is_read_as_utc():
    item = normalize_yahoo_item(
        "NVDA", legacy_item("NVDA", providerPublishTime="2026-07-23T12:00:00")
    )

    assert item["published_at"] == "2026-07-23T12:00:00+00:00"


def test_a_missing_publication_timestamp_is_stored_as_null():
    item = normalize_yahoo_item("NVDA", legacy_item("NVDA", providerPublishTime=None))

    assert item["published_at"] is None


def test_an_undated_item_settles_on_the_fetch_day(tmp_path):
    repository = migrated(tmp_path)
    factory = provider(news=[legacy_item("NVDA", providerPublishTime=None)])

    fetcher(repository, factory).fetch(["NVDA"])

    assert list(partitions(repository)) == [("NVDA", today())]
    assert len(repository.raw_items_for_day(today(), "NVDA")) == 1


def test_one_response_shares_one_fetch_instant(tmp_path):
    # Reading the clock per item is what let a response straddling midnight
    # scatter undated articles across two partitions.
    items = [
        legacy_item("NVDA", providerPublishTime=None, link=f"https://e/{n}")
        for n in range(4)
    ]
    normalized = [
        normalize_yahoo_item("NVDA", item, fetched_at="2026-07-23T23:59:59+00:00")
        for item in items
    ]

    assert {item["fetched_at"] for item in normalized} == {"2026-07-23T23:59:59+00:00"}
    assert {effective_day(item) for item in normalized} == {"2026-07-23"}


# ---------------------------------------------------------------------------
# Retries, throttling, timeout
# ---------------------------------------------------------------------------


def test_a_transient_failure_retries_with_exponential_backoff(tmp_path):
    repository = migrated(tmp_path)
    calls = []
    delays = []

    class FlakyTicker:
        def __init__(self, ticker):
            calls.append(ticker)
            if len(calls) < 3:
                raise RuntimeError("transient")
            self.news = [legacy_item(ticker)]

    counts, errors = fetcher(
        repository,
        FlakyTicker,
        max_retries=2,
        retry_backoff_seconds=0.5,
        sleep=delays.append,
    ).fetch(["NVDA"])

    assert len(calls) == 3
    assert delays == [0.5, 1.0]
    assert counts["retries"] == 2
    assert counts["tickers_succeeded"] == 1
    assert not errors


def test_retry_exhaustion_settles_the_partition_through_the_run_lifecycle(tmp_path):
    repository = migrated(tmp_path)
    delays = []
    factory = provider(raises=RuntimeError("provider unavailable"))

    counts, errors = fetcher(
        repository,
        factory,
        max_retries=2,
        retry_backoff_seconds=0.25,
        sleep=delays.append,
    ).fetch(["NVDA"])

    assert factory.calls == ["NVDA", "NVDA", "NVDA"]
    assert delays == [0.25, 0.5]
    assert counts["retries"] == 2
    assert counts["tickers_failed"] == 1
    assert errors == [{"ticker": "NVDA", "error": "provider unavailable"}]

    state = repository.source_state("yahoo:NVDA")
    assert state["metadata"]["attempts"] == 3
    assert state["status"] == "failed"
    # The failure is durable because a run recorded it, not because a
    # separate ``log_stage`` call was made after the fact.
    run = partitions(repository)[("NVDA", today())]
    assert run["counts"]["source_state_status"] == "failed"


def test_throttling_waits_between_tickers_but_not_before_the_first(tmp_path):
    repository = migrated(tmp_path)
    delays = []

    fetcher(
        repository,
        provider(),
        throttle_seconds=0.75,
        sleep=delays.append,
    ).fetch(["TSLA", "NVDA", "AMD"])

    assert delays == [0.75, 0.75]


def test_a_rejected_ticker_does_not_consume_a_throttle_slot(tmp_path):
    repository = migrated(tmp_path)
    delays = []

    fetcher(
        repository,
        provider(),
        throttle_seconds=0.5,
        sleep=delays.append,
    ).fetch(["GOOG", "NVDA"])

    assert delays == []


def test_a_request_timeout_is_reported_and_checkpointed(tmp_path):
    repository = migrated(tmp_path)

    class SlowTicker:
        def __init__(self, ticker):
            time.sleep(0.05)
            self.news = [legacy_item(ticker)]

    counts, errors = fetcher(
        repository, SlowTicker, request_timeout_seconds=0.001
    ).fetch(["NVDA"])

    assert counts["timeouts"] == 1
    assert counts["tickers_failed"] == 1
    assert "exceeded" in errors[0]["error"]
    state = repository.source_state("yahoo:NVDA")
    assert state["metadata"]["error_type"] == "TimeoutError"
    assert state["last_success_at"] is None


# ---------------------------------------------------------------------------
# Malformed items and invalid evidence
# ---------------------------------------------------------------------------


def test_a_malformed_item_does_not_abort_later_items_and_is_durable(tmp_path):
    database = tmp_path / "phase0.sqlite3"
    repository = Phase0Repository(database)
    repository.migrate()
    factory = provider(news=[None, legacy_item("NVDA")])

    counts, errors = fetcher(repository, factory).fetch(["NVDA"])

    assert counts["fetched"] == 2
    assert counts["invalid"] == 1
    assert counts["inserted"] == 1
    assert counts["invalid_evidence_inserted"] == 1
    assert counts["tickers_partial"] == 1
    assert counts["tickers_succeeded"] == 0
    assert errors[0]["item_index"] == 0
    assert repository.count("raw_items") == 2

    reopened = Phase0Repository(database)
    assert reopened.count("raw_items") == 2
    assert reopened.source_state("yahoo:NVDA")["status"] == "partial"


@pytest.mark.parametrize(
    "updates,error_text",
    [
        ({"title": ""}, "missing title"),
        ({"link": ""}, "missing URL"),
        ({"providerPublishTime": "yesterday-ish"}, "timestamp is not valid"),
        ({"link": "ftp://publisher.example/story"}, "absolute HTTP(S)"),
    ],
)
def test_missing_or_invalid_fields_are_kept_as_invalid_evidence(
    tmp_path, updates, error_text
):
    repository = migrated(tmp_path)
    factory = provider(news=[legacy_item("NVDA", **updates)])

    counts, errors = fetcher(repository, factory).fetch(["NVDA"])

    assert counts["invalid"] == 1
    assert counts["tickers_failed"] == 1
    assert error_text in errors[0]["error"]
    rows = Phase0Reader(repository.database_path).raw_item_candidates()
    stored = repository.raw_items_for_day(today(), "NVDA")
    assert [row["ingest_status"] for row in stored] == ["invalid"]
    assert stored[0]["ticker"] == "NVDA"
    assert rows == [] or all(row["ticker"] == "NVDA" for row in rows)


def test_invalid_evidence_keeps_the_original_payload_and_the_reason(tmp_path):
    repository = migrated(tmp_path)
    factory = provider(news=[{"link": "https://publisher.example/x", "id": "abc"}])

    fetcher(repository, factory).fetch(["NVDA"])

    stored = repository.raw_items_for_day(today(), "NVDA")[0]
    assert stored["external_id"] == "abc"
    assert stored["canonical_url"].startswith("urn:yahoo:nvda:")
    assert "missing title" in stored["validation_errors"]
    assert "publisher.example" in stored["raw_json"]


def test_invalid_evidence_settles_on_the_fetch_day_even_beside_older_valid_items(
    tmp_path,
):
    repository = migrated(tmp_path)
    old = epoch(now() - timedelta(days=3))
    factory = provider(
        news=[
            legacy_item("NVDA", providerPublishTime=old),
            {"link": "https://publisher.example/broken"},
        ]
    )

    fetcher(repository, factory).fetch(["NVDA"])

    older_day = (now() - timedelta(days=3)).date().isoformat()
    assert set(partitions(repository)) == {("NVDA", older_day), ("NVDA", today())}
    assert [row["ingest_status"] for row in repository.raw_items_for_day(today())] == [
        "invalid"
    ]
    assert [
        row["ingest_status"] for row in repository.raw_items_for_day(older_day)
    ] == ["valid"]


# ---------------------------------------------------------------------------
# Duplicates and idempotent re-ingestion
# ---------------------------------------------------------------------------


def test_three_consecutive_fetches_are_idempotent_for_all_five_tickers(tmp_path):
    repository = migrated(tmp_path)
    factory = provider()
    engine = fetcher(repository, factory)

    runs = [engine.fetch() for _ in range(3)]

    assert runs[0][0]["inserted"] == 5
    assert runs[1][0]["duplicates"] == 5
    assert runs[2][0]["duplicates"] == 5
    assert all(not errors for _, errors in runs)
    assert repository.count("raw_items") == 5
    stored = {row["ticker"] for row in repository.raw_items_for_day(today())}
    assert stored == set(TICKERS)


def test_a_replay_writes_a_new_run_identity_rather_than_overwriting_one(tmp_path):
    repository = migrated(tmp_path)
    engine = fetcher(repository, provider())

    engine.fetch(["NVDA"])
    engine.fetch(["NVDA"])

    rows = repository.run_log_entries(stage=STAGE)
    assert len({row["run_id"] for row in rows}) == 2
    assert {row["trading_day"] for row in rows} == {today()}


def test_a_shared_article_keeps_both_ticker_associations(tmp_path):
    repository = migrated(tmp_path)
    shared = "https://publisher.example/shared"
    factory = provider(
        news_by_ticker={
            ticker: [legacy_item(ticker, title="Chip-sector headline", link=shared)]
            for ticker in ("NVDA", "AMD")
        }
    )

    counts, errors = fetcher(repository, factory).fetch(["NVDA", "AMD"])

    assert counts["inserted"] == 1
    assert counts["duplicates"] == 1
    assert not errors
    item_id = repository.raw_items_for_day(today())[0]["id"]
    assert repository.raw_item_tickers(item_id) == ["AMD", "NVDA"]


# ---------------------------------------------------------------------------
# Source-state persistence
# ---------------------------------------------------------------------------


def test_the_checkpoint_survives_a_reconnect(tmp_path):
    database = tmp_path / "phase0.sqlite3"
    repository = Phase0Repository(database)
    repository.migrate()

    fetcher(repository, provider()).fetch(["NVDA"])

    state = Phase0Repository(database).source_state("yahoo:NVDA")
    assert state["status"] == "success"
    assert state["metadata"]["status"] == "success"
    assert state["metadata"]["provider_count"] == 1
    assert state["last_success_at"] is not None


def test_the_checkpoint_rides_on_the_fetch_day_partition(tmp_path):
    repository = migrated(tmp_path)
    old = epoch(now() - timedelta(days=2))
    factory = provider(
        news=[
            legacy_item("NVDA", providerPublishTime=old, link="https://e/old"),
            legacy_item("NVDA", link="https://e/new"),
        ]
    )

    fetcher(repository, factory).fetch(["NVDA"])

    older_day = (now() - timedelta(days=2)).date().isoformat()
    rows = partitions(repository)
    # Source state is keyed by feed and answers "when did we last check
    # it", so it belongs to the fetch day and to no other partition.
    assert "source_states_recorded" not in rows[("NVDA", older_day)]["counts"]
    assert rows[("NVDA", today())]["counts"]["source_states_recorded"] == 1
    assert repository.source_state("yahoo:NVDA")["last_checked_at"][:10] == today()


def test_the_checkpoint_records_the_days_the_response_spanned(tmp_path):
    repository = migrated(tmp_path)
    old = epoch(now() - timedelta(days=1))
    factory = provider(
        news=[
            legacy_item("NVDA", providerPublishTime=old, link="https://e/old"),
            legacy_item("NVDA", link="https://e/new"),
        ]
    )

    fetcher(repository, factory).fetch(["NVDA"])

    yesterday = (now() - timedelta(days=1)).date().isoformat()
    metadata = repository.source_state("yahoo:NVDA")["metadata"]
    assert metadata["trading_days"] == sorted({yesterday, today()})


# ---------------------------------------------------------------------------
# Provider outcome semantics
# ---------------------------------------------------------------------------


def test_an_empty_response_is_checkpointed_as_empty(tmp_path):
    repository = migrated(tmp_path)

    counts, errors = fetcher(repository, provider(news=[])).fetch(["NVDA"])

    assert counts["tickers_empty"] == 1
    assert counts["tickers_succeeded"] == 0
    assert errors == [{"ticker": "NVDA", "error": "empty provider response"}]
    state = repository.source_state("yahoo:NVDA")
    assert state["status"] == "empty"
    assert repository.count("raw_items") == 0
    # An empty check still happened, and the run that recorded it says so.
    assert partitions(repository)[("NVDA", today())]["status"] == "success"


def test_a_partial_batch_is_checkpointed_as_partial(tmp_path):
    repository = migrated(tmp_path)
    factory = provider(news=[legacy_item("NVDA"), {"link": "https://e/broken"}])

    counts, _ = fetcher(repository, factory).fetch(["NVDA"])

    assert counts["tickers_partial"] == 1
    assert repository.source_state("yahoo:NVDA")["status"] == "partial"


def test_a_complete_provider_failure_is_settled_through_the_run_lifecycle(tmp_path):
    repository = migrated(tmp_path)

    counts, errors = fetcher(
        repository, provider(raises=RuntimeError("offline"))
    ).fetch(["NVDA"])

    assert counts["tickers_failed"] == 1
    assert counts["partitions"] == 1
    assert repository.count("raw_items") == 0
    state = repository.source_state("yahoo:NVDA")
    assert state["status"] == "failed"
    assert state["last_success_at"] is None
    assert state["last_error"] == "offline"
    # The old code reached for ``log_stage`` here.  Now the settlement is
    # an ordinary logged mutation, so the row exists for the same reason
    # every other row does.
    run = partitions(repository)[("NVDA", today())]
    assert run["pipeline_version"] == "phase0-v1"
    assert run["counts"] == {
        "source_states_recorded": 1,
        "source_state_status": "failed",
    }


@pytest.mark.parametrize(
    "news,raises,source_status,bucket",
    [
        ([legacy_item("NVDA")], None, "success", "tickers_succeeded"),
        ([None, legacy_item("NVDA")], None, "partial", "tickers_partial"),
        ([], None, "empty", "tickers_empty"),
        ([{"link": "https://e/x"}], None, "failed", "tickers_failed"),
        (None, RuntimeError("offline"), "failed", "tickers_failed"),
    ],
)
def test_the_run_outcome_and_the_checkpoint_never_contradict(
    tmp_path, news, raises, source_status, bucket
):
    repository = migrated(tmp_path)
    factory = provider(news=news, raises=raises)

    counts, _ = fetcher(repository, factory).fetch(["NVDA"])

    assert counts[bucket] == 1
    state = repository.source_state("yahoo:NVDA")
    assert state["status"] == source_status
    run = partitions(repository)[("NVDA", today())]
    # The contradiction that matters is a run claiming clean success over a
    # checkpoint that says the fetch failed.
    assert not (run["status"] == "success" and source_status == "failed")
    if source_status == "failed":
        assert run["status"] == "degraded"
        assert state["last_success_at"] is None
    else:
        assert run["status"] == "success"
        assert state["last_success_at"] is not None


# ---------------------------------------------------------------------------
# Cross-partition isolation
# ---------------------------------------------------------------------------


def test_one_ticker_failing_does_not_abort_the_others(tmp_path, caplog):
    repository = migrated(tmp_path)

    class MixedTicker:
        def __init__(self, ticker):
            if ticker == "TSLA":
                raise RuntimeError("provider unavailable")
            self.news = [legacy_item(ticker)]

    counts, errors = fetcher(repository, MixedTicker).fetch(["TSLA", "NVDA"])

    assert counts["tickers_failed"] == 1
    assert counts["tickers_succeeded"] == 1
    assert counts["inserted"] == 1
    assert errors[0]["ticker"] == "TSLA"
    assert repository.source_state("yahoo:TSLA")["metadata"] == {
        "attempts": 1,
        "error_type": "RuntimeError",
        "request_timeout_seconds": 30.0,
        "status": "failed",
    }
    assert "ticker=TSLA" in caplog.text
    assert set(partitions(repository)) == {("TSLA", today()), ("NVDA", today())}


def test_a_partition_never_holds_another_partitions_evidence(tmp_path):
    repository = migrated(tmp_path)
    factory = provider()

    fetcher(repository, factory).fetch(["NVDA", "AMD"])

    for ticker in ("NVDA", "AMD"):
        stored = repository.raw_items_for_day(today(), ticker)
        assert [row["ticker"] for row in stored] == [ticker]


def test_a_persistence_failure_in_one_ticker_leaves_the_others_settled(
    tmp_path, monkeypatch
):
    repository = migrated(tmp_path)
    real = Phase0Repository.ingest_raw_items

    def refuse_nvda(self, items, *, run, **kwargs):
        if run.ticker == "NVDA":
            raise RuntimeError("database unavailable")
        return real(self, items, run=run, **kwargs)

    monkeypatch.setattr(Phase0Repository, "ingest_raw_items", refuse_nvda)
    counts, errors = fetcher(repository, provider()).fetch(["NVDA", "AMD"])

    assert counts["tickers_failed"] == 1
    assert counts["tickers_succeeded"] == 1
    assert "persistence failed" in errors[0]["error"]
    assert [row["ticker"] for row in repository.raw_items_for_day(today())] == ["AMD"]
    assert repository.source_state("yahoo:AMD")["status"] == "success"


def test_a_failed_partition_does_not_stamp_the_feed_as_checked(tmp_path, monkeypatch):
    repository = migrated(tmp_path)

    def refuse(self, items, *, run, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(Phase0Repository, "ingest_raw_items", refuse)
    counts, errors = fetcher(repository, provider()).fetch(["NVDA"])

    assert counts["tickers_succeeded"] == 0
    assert counts["tickers_failed"] == 1
    assert "persistence failed" in errors[0]["error"]
    # No checkpoint at all, rather than one claiming a check that never
    # completed.  The failure is durable in the run log instead.
    assert repository.source_state("yahoo:NVDA") is None
    assert partitions(repository)[("NVDA", today())]["status"] == "failed"


# ---------------------------------------------------------------------------
# Per-ticker/day partitioning and multi-day splitting
# ---------------------------------------------------------------------------


def test_a_response_spanning_days_splits_into_one_run_per_day(tmp_path):
    repository = migrated(tmp_path)
    days = [now() - timedelta(days=offset) for offset in (2, 1, 0)]
    factory = provider(
        news=[
            legacy_item(
                "NVDA", providerPublishTime=epoch(moment), link=f"https://e/{index}"
            )
            for index, moment in enumerate(days)
        ]
    )

    counts, errors = fetcher(repository, factory).fetch(["NVDA"])

    expected = {("NVDA", moment.date().isoformat()) for moment in days}
    assert set(partitions(repository)) == expected
    assert counts["partitions"] == 3
    assert counts["inserted"] == 3
    assert not errors
    for _, day in expected:
        assert len(repository.raw_items_for_day(day, "NVDA")) == 1


def test_no_run_log_row_claims_more_than_one_ticker_or_day(tmp_path):
    repository = migrated(tmp_path)
    yesterday = epoch(now() - timedelta(days=1))
    factory = provider(
        news_by_ticker={
            ticker: [
                legacy_item(ticker, link=f"https://e/{ticker}/today"),
                legacy_item(
                    ticker,
                    providerPublishTime=yesterday,
                    link=f"https://e/{ticker}/yesterday",
                ),
            ]
            for ticker in TICKERS
        }
    )

    counts, _ = fetcher(repository, factory).fetch()

    rows = repository.run_log_entries(stage=STAGE)
    assert len(rows) == counts["partitions"] == 10
    # One row per identity, and the identity names exactly one partition.
    assert len({row["run_id"] for row in rows}) == 10
    for row in rows:
        assert row["ticker"] in TICKERS
        assert row["trading_day"] in {
            today(),
            (now() - timedelta(days=1)).date().isoformat(),
        }
        stored = repository.raw_items_for_day(row["trading_day"], row["ticker"])
        assert all(item["ticker"] == row["ticker"] for item in stored)


def test_the_run_identity_carries_the_partition(tmp_path):
    repository = migrated(tmp_path)
    yesterday = (now() - timedelta(days=1)).date().isoformat()
    factory = provider(
        news=[
            legacy_item("NVDA", link="https://e/today"),
            legacy_item(
                "NVDA",
                providerPublishTime=epoch(now() - timedelta(days=1)),
                link="https://e/yesterday",
            ),
        ]
    )

    fetcher(repository, factory).fetch(["NVDA"], run_id="fixed-run")

    rows = {row["run_id"]: row for row in repository.run_log_entries(stage=STAGE)}
    assert set(rows) == {
        partition_run_id("fixed-run", "NVDA", today()),
        partition_run_id("fixed-run", "NVDA", yesterday),
    }
    # ``run_log`` is UNIQUE(run_id, stage) and its upsert never moves a
    # row's day, so a shared identity would have collapsed these two.
    assert (
        rows[partition_run_id("fixed-run", "NVDA", today())]["trading_day"] == today()
    )


def test_a_bare_run_identity_is_refused_for_a_second_partition(tmp_path):
    # Why partition_run_id exists: I1 does not merely overwrite a reused
    # identity, it refuses it, so a fetcher that shared one across the days
    # of a single response could not persist the second day at all.
    repository = migrated(tmp_path)
    yesterday = (now() - timedelta(days=1)).date().isoformat()

    with repository.stage_run(
        run_id="shared",
        stage=STAGE,
        ticker="NVDA",
        trading_day=today(),
        pipeline_version="phase0-v1",
    ) as run:
        repository.ingest_raw_items([], run=run, terminal=True)

    with pytest.raises(Phase0RunContextError, match="names one partition"):
        with repository.stage_run(
            run_id="shared",
            stage=STAGE,
            ticker="NVDA",
            trading_day=yesterday,
            pipeline_version="phase0-v1",
        ) as run:
            repository.ingest_raw_items([], run=run, terminal=True)

    rows = repository.run_log_entries(stage=STAGE)
    assert [row["trading_day"] for row in rows] == [today()]


def test_an_item_may_not_be_ingested_into_a_foreign_partition(tmp_path):
    repository = migrated(tmp_path)
    item = normalize_yahoo_item("NVDA", legacy_item("NVDA"))

    with pytest.raises(Phase0RunContextError, match="AMD"):
        with repository.stage_run(
            run_id="cross",
            stage=STAGE,
            ticker="AMD",
            trading_day=today(),
            pipeline_version="phase0-v1",
        ) as run:
            repository.ingest_raw_items([item], run=run, terminal=True)

    assert repository.count("raw_items") == 0


# ---------------------------------------------------------------------------
# Terminal settlement and the absence of a logging bypass
# ---------------------------------------------------------------------------


def test_only_the_final_mutation_of_a_partition_is_terminal(tmp_path, monkeypatch):
    repository = migrated(tmp_path)
    calls = []
    real_ingest = Phase0Repository.ingest_raw_items
    real_state = Phase0Repository.record_source_state

    def watch_ingest(self, items, *, run, terminal=False, **kwargs):
        calls.append(("ingest", run.trading_day, terminal))
        return real_ingest(self, items, run=run, terminal=terminal, **kwargs)

    def watch_state(self, source, *, run, terminal=False, **kwargs):
        calls.append(("state", run.trading_day, terminal))
        return real_state(self, source, run=run, terminal=terminal, **kwargs)

    monkeypatch.setattr(Phase0Repository, "ingest_raw_items", watch_ingest)
    monkeypatch.setattr(Phase0Repository, "record_source_state", watch_state)

    yesterday = (now() - timedelta(days=1)).date().isoformat()
    factory = provider(
        news=[
            legacy_item("NVDA", link="https://e/today"),
            legacy_item(
                "NVDA",
                providerPublishTime=epoch(now() - timedelta(days=1)),
                link="https://e/yesterday",
            ),
        ]
    )
    fetcher(repository, factory).fetch(["NVDA"])

    assert calls == [
        ("ingest", yesterday, True),
        ("ingest", today(), False),
        ("state", today(), True),
    ]
    # Exactly one terminal mutation per partition.
    for day in (yesterday, today()):
        assert (
            sum(1 for _, run_day, terminal in calls if run_day == day and terminal) == 1
        )


def test_every_mutation_runs_under_a_stage_run_context(tmp_path, monkeypatch):
    repository = migrated(tmp_path)
    seen = []
    real_ingest = Phase0Repository.ingest_raw_items
    real_state = Phase0Repository.record_source_state

    def watch_ingest(self, items, *, run, **kwargs):
        seen.append(run)
        return real_ingest(self, items, run=run, **kwargs)

    def watch_state(self, source, *, run, **kwargs):
        seen.append(run)
        return real_state(self, source, run=run, **kwargs)

    monkeypatch.setattr(Phase0Repository, "ingest_raw_items", watch_ingest)
    monkeypatch.setattr(Phase0Repository, "record_source_state", watch_state)
    fetcher(repository, provider()).fetch(["NVDA"])

    assert seen
    assert all(isinstance(run, StageRunContext) for run in seen)
    assert all(run.stage == STAGE for run in seen)


def test_every_stored_item_has_a_run_log_row_for_its_own_partition(tmp_path):
    repository = migrated(tmp_path)
    yesterday = epoch(now() - timedelta(days=1))
    factory = provider(
        news_by_ticker={
            ticker: [
                legacy_item(ticker, link=f"https://e/{ticker}/a"),
                legacy_item(
                    ticker, providerPublishTime=yesterday, link=f"https://e/{ticker}/b"
                ),
            ]
            for ticker in TICKERS
        }
    )

    fetcher(repository, factory).fetch()

    logged = set(partitions(repository))
    for day in (today(), (now() - timedelta(days=1)).date().isoformat()):
        for row in repository.raw_items_for_day(day):
            assert (row["ticker"], day) in logged


def test_there_is_no_persist_run_log_switch(tmp_path):
    repository = migrated(tmp_path)

    with pytest.raises(TypeError, match="persist_run_log"):
        YahooFinanceFetcher(
            repository,
            ticker_factory=provider(),
            persist_run_log=False,
        )


def test_the_fetcher_takes_no_trading_day_override(tmp_path):
    # A run's day is a partition identity the evidence decides, not a label
    # the caller supplies; the repository would reject a mismatched batch.
    repository = migrated(tmp_path)

    with pytest.raises(TypeError, match="trading_day"):
        fetcher(repository, provider()).fetch(["NVDA"], trading_day="2026-07-23")


def test_the_removed_repository_apis_are_still_absent():
    for name in REMOVED_REPOSITORY_APIS:
        assert not hasattr(Phase0Repository, name), name


def test_the_module_never_reaches_for_a_removed_api():
    source = inspect.getsource(yahoo_module)
    for name in REMOVED_REPOSITORY_APIS:
        assert f".{name}(" not in source, name
    assert ".admin." not in source
    assert "stage_run(" in source
    assert "ingest_raw_items(" in source
    assert "record_source_state(" in source


def test_a_pipeline_version_reaches_every_partition(tmp_path):
    repository = migrated(tmp_path)

    fetcher(repository, provider(), pipeline_version="pipeline-test").fetch(
        ["NVDA"], run_id="pipeline-run"
    )

    rows = repository.run_log_entries(stage=STAGE)
    assert [row["pipeline_version"] for row in rows] == ["pipeline-test"]
    assert rows[0]["run_id"] == partition_run_id("pipeline-run", "NVDA", today())


# ---------------------------------------------------------------------------
# Credential redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_message,credential",
    [
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("Authorization: Bearer bearer-secret", "bearer-secret"),
        ("X-API-Key: api-secret", "api-secret"),
        ("Authorization: Custom opaque full value", "opaque full value"),
    ],
)
def test_provider_secrets_are_redacted_from_outputs_logs_and_state(
    tmp_path, caplog, provider_message, credential
):
    repository = migrated(tmp_path)

    counts, errors = fetcher(
        repository, provider(raises=RuntimeError(provider_message))
    ).fetch(["NVDA"])

    assert counts["tickers_failed"] == 1
    assert credential not in str(errors)
    assert credential not in caplog.text
    assert credential not in str(repository.source_state("yahoo:NVDA"))
    assert credential not in str(repository.run_log_entries(stage=STAGE))


def test_a_credential_in_an_item_error_is_redacted_before_it_is_stored(tmp_path):
    repository = migrated(tmp_path)
    factory = provider(
        news=[
            legacy_item(
                "NVDA",
                providerPublishTime="Authorization: Bearer leaked-secret",
            )
        ]
    )

    counts, errors = fetcher(repository, factory).fetch(["NVDA"])

    assert counts["invalid"] == 1
    assert "leaked-secret" not in str(errors)
    assert "leaked-secret" not in str(repository.source_state("yahoo:NVDA"))
    stored = repository.raw_items_for_day(today(), "NVDA")[0]
    assert "leaked-secret" not in stored["validation_errors"]


# ---------------------------------------------------------------------------
# Bounded provider work: what a timeout actually stops
# ---------------------------------------------------------------------------
#
# A timeout used to bound only the caller's wait.  The request it gave up on
# kept running, and every retry started another one, so a provider that hung
# cost one live request per attempt per scheduled fetch — without limit.
# yfinance cannot cancel a request in flight, so these tests do not assert
# that a timed-out call stops.  They assert the guarantee that is actually
# available: the work outstanding at any moment has a ceiling, and neither
# retries nor later fetches raise it.


def test_a_provider_that_answers_within_the_budget_is_unaffected(tmp_path):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = provider()

    counts, errors = fetcher(
        repository, factory, provider_gate=gate, request_timeout_seconds=5
    ).fetch(["NVDA"])

    assert counts["inserted"] == 1
    assert counts["timeouts"] == 0
    assert not errors
    assert repository.source_state("yahoo:NVDA")["status"] == "success"
    # The slot is handed back by the request, so a healthy fetch leaves none held.
    assert gate.outstanding == 0
    assert gate.live_workers == 0
    assert gate.joined == 0


def test_a_provider_that_exceeds_the_budget_times_out_while_its_call_runs_on(
    tmp_path, hanging
):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = hanging()

    counts, errors = fetcher(
        repository, factory, provider_gate=gate, request_timeout_seconds=0.05
    ).fetch(["NVDA"])

    assert counts["timeouts"] == 1
    assert counts["tickers_failed"] == 1
    assert "exceeded" in errors[0]["error"]
    # The honest part: the request is still running.  It is bounded, not stopped.
    assert gate.outstanding == 1
    assert gate.live_workers == 1
    assert gate.outstanding_keys == ("NVDA",)


def test_a_retry_joins_the_outstanding_request_instead_of_adding_one(tmp_path, hanging):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = hanging()

    counts, _ = fetcher(
        repository,
        factory,
        provider_gate=gate,
        request_timeout_seconds=0.05,
        max_retries=2,
        retry_backoff_seconds=0,
    ).fetch(["NVDA"])

    factory.wait_until_entered(1)
    assert counts["retries"] == 2  # three attempts were made
    # ...against one live request.  Occupancy, not invocation counting: the
    # gate accounts for the slot and the thread is really there.
    assert gate.outstanding == 1
    assert gate.live_workers == 1
    assert len(provider_threads()) == 1
    assert gate.joined == 2
    assert factory.calls == ["NVDA"]


def test_repeated_timeouts_never_exceed_the_configured_ceiling(tmp_path, hanging):
    repository = migrated(tmp_path)
    gate = YahooProviderGate(2)
    factory = hanging()

    counts, errors = fetcher(
        repository,
        factory,
        provider_gate=gate,
        request_timeout_seconds=0.05,
        max_retries=1,
        retry_backoff_seconds=0,
    ).fetch(["TSLA", "NVDA", "AMD", "AAPL", "META"])

    factory.wait_until_entered(2)
    assert gate.outstanding == 2 == gate.max_concurrent
    assert gate.live_workers == 2
    assert len(provider_threads()) == 2
    assert factory.calls == ["TSLA", "NVDA"]
    # Five tickers, ten attempts, two live requests: the rest were refused
    # outright rather than queued behind a provider that is already stuck.
    assert counts["timeouts"] == 2
    assert counts["provider_busy"] == 3
    assert counts["tickers_failed"] == 5
    assert {error["ticker"] for error in errors} == set(TICKERS)


def test_a_refused_request_names_the_ceiling_and_settles_as_a_failed_check(
    tmp_path, hanging
):
    repository = migrated(tmp_path)
    gate = YahooProviderGate(1)
    factory = hanging()

    counts, errors = fetcher(
        repository, factory, provider_gate=gate, request_timeout_seconds=0.05
    ).fetch(["TSLA", "NVDA"])

    assert counts["provider_busy"] == 1
    refusal = [error for error in errors if error["ticker"] == "NVDA"][0]["error"]
    assert "1 Yahoo provider slots" in refusal
    state = repository.source_state("yahoo:NVDA")
    assert state["status"] == "failed"
    assert state["metadata"]["error_type"] == "YahooProviderBusyError"
    assert partitions(repository)[("NVDA", today())]["status"] == "degraded"


def test_a_later_scheduled_fetch_during_a_hang_starts_no_second_worker(
    tmp_path, hanging
):
    """A scheduler builds a fresh fetcher per run; the ceiling is the process's."""

    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = hanging()

    fetcher(
        repository, factory, provider_gate=gate, request_timeout_seconds=0.05
    ).fetch(["NVDA"])
    factory.wait_until_entered(1)
    assert gate.live_workers == 1

    for _ in range(4):
        fetcher(
            repository, factory, provider_gate=gate, request_timeout_seconds=0.05
        ).fetch(["NVDA"])
        assert gate.outstanding == 1
        assert gate.live_workers == 1
        assert len(provider_threads()) == 1

    assert factory.calls == ["NVDA"]


def test_a_fetcher_built_without_a_gate_shares_the_process_one(tmp_path):
    repository = migrated(tmp_path)

    first = YahooFinanceFetcher(repository)
    second = YahooFinanceFetcher(repository)

    assert first.provider_gate is SHARED_PROVIDER_GATE
    assert second.provider_gate is first.provider_gate
    assert SHARED_PROVIDER_GATE.max_concurrent == DEFAULT_MAX_CONCURRENT_REQUESTS
    assert DEFAULT_MAX_CONCURRENT_REQUESTS == len(TICKERS)


def test_the_next_fetch_recovers_once_the_hung_request_finally_returns(
    tmp_path, hanging
):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = hanging()

    timed_out, _ = fetcher(
        repository, factory, provider_gate=gate, request_timeout_seconds=0.05
    ).fetch(["NVDA"])
    assert timed_out["timeouts"] == 1

    factory.released.set()
    assert wait_for(lambda: gate.outstanding == 0)
    assert gate.live_workers == 0

    recovered, errors = fetcher(
        repository, provider(), provider_gate=gate, request_timeout_seconds=5
    ).fetch(["NVDA"])

    assert recovered["inserted"] == 1
    assert not errors
    assert repository.source_state("yahoo:NVDA")["status"] == "success"
    assert gate.outstanding == 0


def test_an_ordinary_timeout_and_recovery_leaks_no_threads(tmp_path, hanging):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    baseline = threading.active_count()

    for _ in range(5):
        factory = hanging()
        fetcher(
            repository,
            factory,
            provider_gate=gate,
            request_timeout_seconds=0.05,
            max_retries=2,
            retry_backoff_seconds=0,
        ).fetch(["NVDA"])
        assert gate.live_workers == 1
        factory.released.set()
        assert wait_for(lambda: gate.outstanding == 0)

    assert gate.live_workers == 0
    assert not provider_threads()
    assert wait_for(lambda: threading.active_count() == baseline)


def test_provider_workers_are_daemons_so_a_hang_cannot_wedge_shutdown(
    tmp_path, hanging
):
    """Why this is not a ThreadPoolExecutor: its workers are joined at exit."""

    repository = migrated(tmp_path)
    gate = YahooProviderGate()

    fetcher(
        repository, hanging(), provider_gate=gate, request_timeout_seconds=0.05
    ).fetch(["NVDA"])

    workers = provider_threads()
    assert workers
    assert all(worker.daemon for worker in workers)


def test_retry_count_and_backoff_survive_the_bounded_design(tmp_path, hanging):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    delays = []
    factory = hanging()

    counts, _ = fetcher(
        repository,
        factory,
        provider_gate=gate,
        request_timeout_seconds=0.02,
        max_retries=3,
        retry_backoff_seconds=0.25,
        sleep=delays.append,
    ).fetch(["NVDA"])

    assert delays == [0.25, 0.5, 1.0]
    assert counts["retries"] == 3
    assert repository.source_state("yahoo:NVDA")["metadata"]["attempts"] == 4
    assert factory.calls == ["NVDA"]


def test_a_timed_out_ticker_still_settles_its_source_state_and_run_log(
    tmp_path, hanging
):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()

    fetcher(
        repository,
        hanging(),
        provider_gate=gate,
        request_timeout_seconds=0.05,
        max_retries=1,
        retry_backoff_seconds=0,
    ).fetch(["NVDA"])

    state = repository.source_state("yahoo:NVDA")
    assert state["status"] == "failed"
    assert state["metadata"]["attempts"] == 2
    assert state["metadata"]["error_type"] == "TimeoutError"
    assert state["last_success_at"] is None
    run = partitions(repository)[("NVDA", today())]
    assert run["status"] == "degraded"
    assert run["counts"]["source_state_status"] == "failed"
    assert run["completed_at"] is not None


def test_a_credential_in_a_joined_request_is_redacted_when_it_finally_raises(
    tmp_path, caplog, hanging
):
    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = hanging(raises=RuntimeError("Authorization: Bearer hung-secret"))
    delays = []

    def release_after_the_first_wait(_delay):
        delays.append(_delay)
        factory.released.set()

    counts, errors = fetcher(
        repository,
        factory,
        provider_gate=gate,
        request_timeout_seconds=0.05,
        max_retries=1,
        sleep=release_after_the_first_wait,
    ).fetch(["NVDA"])

    # Attempt one timed out; attempt two joined the same request and got its
    # exception.  One request, and the credential never escapes.
    assert factory.calls == ["NVDA"]
    assert gate.joined == 1
    assert counts["tickers_failed"] == 1
    assert "hung-secret" not in str(errors)
    assert "hung-secret" not in caplog.text
    assert "hung-secret" not in str(repository.source_state("yahoo:NVDA"))
    assert "hung-secret" not in str(repository.run_log_entries(stage=STAGE))


def test_a_provider_timeout_error_is_not_reported_as_our_budget_running_out(tmp_path):
    """``wait`` and ``result`` are separated so the two cannot be confused."""

    repository = migrated(tmp_path)

    counts, errors = fetcher(
        repository,
        provider(raises=TimeoutError("provider closed the connection")),
        request_timeout_seconds=5,
    ).fetch(["NVDA"])

    assert counts["timeouts"] == 1
    assert errors[0]["error"] == "provider closed the connection"


def test_the_gate_refuses_a_ceiling_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        YahooProviderGate(0)


def test_the_gate_reports_the_refusal_as_its_own_error_type(tmp_path, hanging):
    gate = YahooProviderGate(1)
    factory = hanging()

    gate.call("TSLA", lambda: factory("TSLA"))
    factory.wait_until_entered(1)

    with pytest.raises(YahooProviderBusyError):
        gate.call("NVDA", lambda: factory("NVDA"))
    assert gate.outstanding == 1


def test_a_finished_request_is_never_served_to_the_next_caller(tmp_path):
    """A retry must re-ask the provider, not inherit the last answer.

    The gate retires a request before it completes it, so a caller that has
    seen an outcome cannot still find that request outstanding.  Without
    that ordering a provider failing instantly is "retried" against a
    cached exception and called exactly once.
    """

    repository = migrated(tmp_path)
    gate = YahooProviderGate()
    factory = provider(raises=RuntimeError("instant failure"))

    counts, _ = fetcher(
        repository,
        factory,
        provider_gate=gate,
        max_retries=4,
        retry_backoff_seconds=0,
    ).fetch(["NVDA"])

    assert factory.calls == ["NVDA"] * 5
    assert counts["retries"] == 4
    assert gate.joined == 0
    assert gate.outstanding == 0


def test_a_request_is_retired_before_its_answer_reaches_the_caller():
    """The ordering that makes the previous test's guarantee airtight.

    Completion is the only thing a caller observes, so retiring the request
    first is what makes "I have an outcome" imply "that request is no longer
    outstanding".  Asserting the retry count alone would only catch this
    intermittently — the losing interleaving is a few microseconds wide —
    so this checks the invariant itself, on every one of many calls.
    """

    gate = YahooProviderGate()

    for index in range(200):
        answered = gate.call(f"KEY{index}", lambda: "answer")
        assert answered.result(timeout=5) == "answer"
        assert gate.outstanding == 0, "answered while still holding its slot"
        assert gate.live_workers == 0

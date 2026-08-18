"""Phase 0 pipeline orchestration (#68) against the final I1/I2/I3 contract.

The stale I4 branch orchestrated by writing: it minted one ``run_id`` and
one ``trading_day`` for the whole process, handed both to every fetcher,
and recorded each stage itself with ``repository.log_stage``. Every one of
those calls is gone from ``Phase0Repository`` now, and the assumption
underneath them -- that a process is a partition -- is what these tests
exist to keep from coming back.

Persistence assertions go through the final public surface:
``run_log_entries``/``source_state`` for the run lifecycle,
:class:`~phase0.repository.Phase0Reader` for evidence.
"""

import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

import pipeline
from pipeline import (
    ComponentResult,
    DOWNSTREAM_STAGES,
    EXIT_CODES,
    Stage,
    build_parser,
    component_status,
    database_report,
    execute_stage,
    invocation_status,
    main,
    new_invocation_id,
    replay_capabilities,
    run_live,
    run_replay,
    status_report,
)
from phase0.repository import Phase0Reader, Phase0Repository
from phase0.rss import RSSFetcher
from phase0.yahoo import TICKERS, YahooFinanceFetcher, YahooProviderGate


#: Write APIs #57 removed from ``Phase0Repository``. The stale pipeline
#: called ``log_stage`` directly and its tests used ``connect``; an
#: orchestrator has no business with any of them.
REMOVED_REPOSITORY_APIS = (
    "connect",
    "read_connection",
    "insert_raw_item",
    "insert_raw_items",
    "set_source_state",
    "log_stage",
    "clear_derived_for_day",
    "persist_run_log",
    "complete_stage_key",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def migrated(tmp_path, name="phase0.sqlite3"):
    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    return repository


def reader(repository):
    return Phase0Reader(repository.database_path)


def headline(ticker, *, published=None):
    moment = published or datetime.now(timezone.utc)
    return {
        "title": f"{ticker} headline",
        "link": f"https://example.com/{ticker.lower()}-{moment.date().isoformat()}",
        "publisher": "Example News",
        "providerPublishTime": int(moment.timestamp()),
    }


def provider(*, news_by_ticker=None, failing=()):
    """A fake ``yfinance.Ticker`` that can fail for named symbols only."""

    class FakeTicker:
        def __init__(self, ticker):
            if ticker in failing:
                raise RuntimeError(f"{ticker} provider unavailable")
            if news_by_ticker is not None:
                self.news = list(news_by_ticker.get(ticker, []))
            else:
                self.news = [headline(ticker)]

    return FakeTicker


FEED_BODY = (
    b"<rss><channel><item><title>Apple iPhone shipment update</title>"
    b"<link>https://example.com/apple-story</link>"
    b"<description>iPhone demand and Tim Cook commentary</description>"
    b"<pubDate>Tue, 18 Aug 2026 10:00:00 GMT</pubDate></item></channel></rss>"
)


def write_feeds(path, feed_ids):
    config = {"version": 1, "feeds": []}
    for feed_id in feed_ids:
        config["feeds"].append(
            {
                "id": feed_id,
                "name": f"Feed {feed_id}",
                "url": f"https://example.com/{feed_id}",
                "enabled": True,
                "format": "rss2",
                "intended_role": "test coverage",
                "expected_fields": {
                    "title": "title",
                    "url": "link",
                    "description": "description",
                    "published_at": "pubDate",
                },
                "polling": {
                    "interval_minutes": 30,
                    "conditional_get": True,
                    "timeout_seconds": 20,
                },
                "notes": ["Synthetic test fixture."],
            }
        )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def write_aliases(path):
    path.write_text(
        "tickers:\n  - ticker: AAPL\n    strong_aliases: [iPhone]\n",
        encoding="utf-8",
    )


@pytest.fixture
def config(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    write_feeds(feeds, ["alpha", "beta"])
    write_aliases(aliases)
    return {"feeds_path": feeds, "aliases_path": aliases}


def responder(*, failing=(), body=FEED_BODY):
    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        def __init__(self, url):
            self.url = url
            self.content = body

        def raise_for_status(self):
            return None

    def get(url, **kwargs):
        feed_id = str(url).rsplit("/", 1)[-1]
        if feed_id in failing:
            raise RuntimeError(f"{feed_id} unreachable")
        return Response(url)

    return get


def wire(monkeypatch, *, ticker_factory=None, get=None, yahoo=None, rss=None):
    """Point the orchestrator at test doubles without changing its wiring.

    The real fetchers are kept unless a replacement class is given, so the
    default is a genuine end-to-end run: real runs, real partitions, real
    ``run_log`` rows -- only the provider and the socket are fake.

    ``get`` is applied with ``setdefault``, so a caller that already chose
    one keeps it. That matters for replay, which passes its own refusing
    callable; overriding it here would quietly re-enable the network in the
    tests meant to prove it is off.
    """

    real_yahoo = pipeline.YahooFinanceFetcher
    real_rss = pipeline.RSSFetcher

    def build_yahoo(repository, **options):
        if yahoo is not None:
            return yahoo(repository, **options)
        options.setdefault("max_retries", 0)
        options.setdefault("provider_gate", YahooProviderGate())
        return real_yahoo(repository, ticker_factory=ticker_factory, **options)

    def build_rss(repository, **options):
        if rss is not None:
            return rss(repository, **options)
        options.setdefault("max_retries", 0)
        if get is not None:
            options.setdefault("get", get)
        return real_rss(repository, **options)

    monkeypatch.setattr(pipeline, "YahooFinanceFetcher", build_yahoo)
    monkeypatch.setattr(pipeline, "RSSFetcher", build_rss)


def component(result, name):
    return next(item for item in result.components if item.name == name)


def runs(repository, stage=None):
    rows = repository.run_log_entries(stage=stage)
    return [(r["stage"], r["ticker"], r["trading_day"], r["status"]) for r in rows]


# ---------------------------------------------------------------------------
# Architecture: what the orchestrator may not do
# ---------------------------------------------------------------------------


def test_pipeline_calls_no_repository_api_that_57_removed():
    source = inspect.getsource(pipeline)
    for name in REMOVED_REPOSITORY_APIS:
        assert f".{name}(" not in source, f"pipeline.py calls removed API {name}"


def test_pipeline_never_reaches_for_the_admin_surface():
    """``Phase0Admin`` is unlogged repair. Orchestration is not repair."""

    source = inspect.getsource(pipeline)
    assert ".admin" not in source
    assert "Phase0Admin" not in source


def test_pipeline_writes_no_run_log_row_of_its_own(tmp_path, config, monkeypatch):
    """The invocation is not a run, and leaves no row claiming to be one."""

    repository = migrated(tmp_path)

    class SilentFetcher:
        def __init__(self, repository, **options):
            pass

        def fetch(self, **kwargs):
            return {}, []

    wire(monkeypatch, yahoo=SilentFetcher, rss=SilentFetcher)
    run_live(repository, **config)

    assert repository.run_log_entries() == []
    assert repository.count("run_log") == 0


def test_pipeline_passes_no_trading_day_to_any_component(tmp_path, config):
    """Neither component accepts one, and the orchestrator does not offer one.

    I2 and I3 both dropped ``trading_day`` deliberately: a run's day is a
    partition identity the evidence decides. An orchestrator that supplied
    one would be overriding the only definition that can be right.
    """

    assert "trading_day" not in inspect.signature(YahooFinanceFetcher.fetch).parameters
    assert "trading_day" not in inspect.signature(RSSFetcher.fetch).parameters
    assert "trading_day" not in inspect.signature(run_live).parameters
    source = inspect.getsource(pipeline)
    assert "trading_day=" not in source


def test_the_cli_refuses_a_date_rather_than_ignoring_one(tmp_path):
    """``--date`` cannot be honoured, and silently dropping it would lie."""

    with pytest.raises(SystemExit, match="--date is not supported"):
        main(["--database", str(tmp_path / "db.sqlite3"), "--date", "2026-08-18"])


# ---------------------------------------------------------------------------
# Run identity: correlation is not a partition
# ---------------------------------------------------------------------------


def test_the_correlation_id_never_becomes_a_repository_run_id(
    tmp_path, config, monkeypatch
):
    """Every durable run id names a partition below the invocation id.

    The stale pipeline used one uuid as *the* run id for both sources, so
    two different partitions shared one identity. Here the invocation id is
    only ever a prefix: nothing opens a run under the bare base.
    """

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    invocation = "phase0-fixed-correlation"

    result = run_live(repository, **config, invocation_id=invocation)

    run_ids = {row["run_id"] for row in repository.run_log_entries()}
    assert run_ids, "the components recorded nothing to check"
    assert invocation not in run_ids
    assert f"{invocation}:yahoo" not in run_ids
    assert f"{invocation}:rss" not in run_ids
    for run_id in run_ids:
        base, _, partition = run_id.partition(":")
        assert run_id.startswith(f"{invocation}:")
        assert partition, f"{run_id} names no partition"
    assert result.invocation_id == invocation


def test_each_yahoo_ticker_day_keeps_its_own_run(tmp_path, config, monkeypatch):
    """Five tickers are five partitions, not one process-wide run."""

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    run_live(repository, **config, invocation_id="inv")

    rows = repository.run_log_entries(stage="fetch_yahoo")
    assert len(rows) == len(TICKERS)
    assert len({row["run_id"] for row in rows}) == len(TICKERS)
    for row in rows:
        assert row["run_id"] == f"inv:yahoo:{row['ticker']}:{row['trading_day']}"


def test_one_ticker_spanning_two_days_is_two_runs(tmp_path, config, monkeypatch):
    """A day boundary inside one response splits the partition, not the run."""

    repository = migrated(tmp_path)
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    wire(
        monkeypatch,
        ticker_factory=provider(
            news_by_ticker={
                "NVDA": [headline("NVDA"), headline("NVDA", published=yesterday)]
            }
        ),
        get=responder(),
    )

    run_live(repository, **config, invocation_id="inv")

    nvda = [
        row
        for row in repository.run_log_entries(stage="fetch_yahoo")
        if row["ticker"] == "NVDA"
    ]
    days = sorted(row["trading_day"] for row in nvda)
    assert days == sorted({now.date().isoformat(), yesterday.date().isoformat()})
    assert len({row["run_id"] for row in nvda}) == 2


def test_rss_feed_day_and_ticker_day_runs_stay_separate(tmp_path, config, monkeypatch):
    """Feed evidence and derived relevance are different partitions.

    ``fetch_rss``/``ingest_rss``/``checkpoint_rss`` belong to a feed and a
    day and carry no ticker; ``classify_rss`` belongs to a ticker and a day.
    Collapsing them under one identity would make ``UNIQUE(run_id, stage)``
    reject the second one outright.
    """

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    run_live(repository, **config, invocation_id="inv")

    feed_scoped = repository.run_log_entries(stage="fetch_rss")
    ticker_scoped = repository.run_log_entries(stage="classify_rss")
    assert feed_scoped and ticker_scoped
    assert all(row["ticker"] is None for row in feed_scoped)
    assert all(row["ticker"] is not None for row in ticker_scoped)
    assert not {row["run_id"] for row in feed_scoped} & {
        row["run_id"] for row in ticker_scoped
    }


def test_yahoo_and_rss_never_share_a_run_id(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    run_live(repository, **config, invocation_id="inv")

    rows = repository.run_log_entries()
    yahoo_ids = {row["run_id"] for row in rows if row["stage"] == "fetch_yahoo"}
    rss_ids = {row["run_id"] for row in rows if row["stage"].endswith("_rss")}
    assert yahoo_ids and rss_ids
    assert not yahoo_ids & rss_ids


def test_a_new_invocation_id_is_unique_per_execution():
    assert new_invocation_id() != new_invocation_id()


# ---------------------------------------------------------------------------
# Live orchestration
# ---------------------------------------------------------------------------


def test_a_clean_live_invocation_succeeds(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    result = run_live(repository, **config)

    assert result.status == "success"
    assert result.exit_code == 0
    assert [item.status for item in result.components] == ["success", "success"]
    assert component(result, "yahoo").counts["tickers_succeeded"] == len(TICKERS)
    assert component(result, "rss").counts["feeds_succeeded"] == 2


def test_yahoo_succeeds_while_rss_fails(tmp_path, config, monkeypatch):
    """A dead source does not cost the live one its evidence."""

    repository = migrated(tmp_path)
    wire(
        monkeypatch,
        ticker_factory=provider(),
        get=responder(failing={"alpha", "beta"}),
    )

    result = run_live(repository, **config)

    assert result.status == "degraded"
    assert result.exit_code == 1
    assert component(result, "yahoo").status == "success"
    assert component(result, "rss").status == "failed"
    yahoo_runs = repository.run_log_entries(stage="fetch_yahoo")
    assert len(yahoo_runs) == len(TICKERS)
    assert all(row["status"] == "success" for row in yahoo_runs)
    assert repository.count("raw_items") == len(TICKERS)


def test_rss_succeeds_while_yahoo_partially_fails(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(
        monkeypatch,
        ticker_factory=provider(failing={"NVDA"}),
        get=responder(),
    )

    result = run_live(repository, **config)

    assert result.status == "degraded"
    assert component(result, "yahoo").status == "degraded"
    assert component(result, "rss").status == "success"
    assert component(result, "yahoo").counts["tickers_succeeded"] == len(TICKERS) - 1
    assert component(result, "rss").counts["feeds_succeeded"] == 2


def test_one_failing_ticker_leaves_the_other_partitions_settled(
    tmp_path, config, monkeypatch
):
    """No cross-partition rollback: four tickers commit, one does not."""

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(failing={"NVDA"}), get=responder())

    run_live(repository, **config)

    by_ticker = {
        row["ticker"]: row["status"]
        for row in repository.run_log_entries(stage="fetch_yahoo")
    }
    assert by_ticker["NVDA"] == "degraded"
    assert all(
        status == "success" for ticker, status in by_ticker.items() if ticker != "NVDA"
    )
    stored = {row["ticker"] for row in reader(repository).raw_items()}
    assert "NVDA" not in stored
    assert stored >= {"AAPL", "AMD", "META", "TSLA"}


def test_one_failing_feed_leaves_the_other_feed_settled(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder(failing={"alpha"}))

    result = run_live(repository, **config)

    counts = component(result, "rss").counts
    assert counts["feeds_succeeded"] == 1
    assert counts["feeds_failed"] == 1
    assert counts["inserted"] == 1
    snapshots = reader(repository).feed_snapshots()
    assert [snapshot["feed_source"] for snapshot in snapshots] == ["rss:beta"]


def test_partial_evidence_is_degraded_and_never_a_clean_success(
    tmp_path, config, monkeypatch
):
    """The case the status rule exists for.

    Four tickers of five and one feed of two is a real, usable, incomplete
    result. Rounding it up hides an outage; rounding it down discards a
    day of evidence that is actually there.
    """

    repository = migrated(tmp_path)
    wire(
        monkeypatch,
        ticker_factory=provider(failing={"NVDA"}),
        get=responder(failing={"alpha"}),
    )

    result = run_live(repository, **config)

    assert result.status == "degraded"
    assert result.exit_code == 1
    assert {item.status for item in result.components} == {"degraded"}
    assert repository.count("raw_items") == len(TICKERS) - 1 + 1


def test_every_source_failing_is_a_failed_invocation(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(
        monkeypatch,
        ticker_factory=provider(failing=set(TICKERS)),
        get=responder(failing={"alpha", "beta"}),
    )

    result = run_live(repository, **config)

    assert result.status == "failed"
    assert result.exit_code == 2
    assert {item.status for item in result.components} == {"failed"}


@pytest.mark.parametrize("crashing", ["yahoo", "rss"])
def test_a_component_raising_does_not_stop_or_unwind_the_other(
    tmp_path, config, monkeypatch, crashing
):
    """Isolation in both directions, including the unexpected-exception path.

    Whichever source blows up, the other still runs and its evidence stays
    committed. There is no cross-source transaction that could take it back.
    """

    repository = migrated(tmp_path)

    class Exploding:
        def __init__(self, repository, **options):
            pass

        def fetch(self, **kwargs):
            raise RuntimeError("provider stack exploded")

    wire(
        monkeypatch,
        ticker_factory=provider(),
        get=responder(),
        **{crashing: Exploding},
    )

    result = run_live(repository, **config)

    assert result.status == "degraded"
    assert component(result, crashing).status == "failed"
    survivor = "rss" if crashing == "yahoo" else "yahoo"
    assert component(result, survivor).status == "success"
    assert repository.count("raw_items") > 0
    assert repository.count("run_log") > 0


def test_component_errors_are_aggregated_without_being_flattened(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    wire(
        monkeypatch,
        ticker_factory=provider(failing={"NVDA", "AMD"}),
        get=responder(failing={"alpha"}),
    )

    result = run_live(repository, **config)

    yahoo_errors = component(result, "yahoo").errors
    assert {error["ticker"] for error in yahoo_errors} == {"NVDA", "AMD"}
    rss_errors = component(result, "rss").errors
    assert [error["feed"] for error in rss_errors] == ["alpha"]
    payload = result.as_dict()
    assert [item["component"] for item in payload["components"]] == ["yahoo", "rss"]


def test_downstream_stages_are_registered_nowhere_yet(tmp_path, config, monkeypatch):
    """An empty registry is the honest state, and the CLI reports it."""

    assert DOWNSTREAM_STAGES == ()
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    result = run_live(repository, **config)

    assert [item.name for item in result.components] == ["yahoo", "rss"]


def test_a_registered_downstream_stage_is_orchestrated_in_order(
    tmp_path, config, monkeypatch
):
    """The extension point works, so M1--M5 need no orchestrator surgery."""

    repository = migrated(tmp_path)
    seen: list[str] = []

    def downstream(base_run_id):
        seen.append(base_run_id)
        return {"summarized": 2}, []

    monkeypatch.setattr(
        pipeline,
        "DOWNSTREAM_STAGES",
        (Stage("summarize", downstream, settled=("summarized",)),),
    )
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    result = run_live(repository, **config, invocation_id="inv")

    assert [item.name for item in result.components] == ["yahoo", "rss", "summarize"]
    assert seen == ["inv:summarize"]
    assert result.status == "success"


# ---------------------------------------------------------------------------
# Status rules
# ---------------------------------------------------------------------------


YAHOO_COUNTERS = {
    "settled": ("tickers_succeeded", "tickers_partial", "tickers_empty"),
    "unsettled": ("tickers_failed", "tickers_rejected"),
}


@pytest.mark.parametrize(
    "counts,errors,expected",
    [
        ({"tickers_succeeded": 5}, [], "success"),
        ({"tickers_empty": 5}, [], "success"),
        ({}, [], "success"),
        ({"tickers_succeeded": 4, "tickers_failed": 1}, [{"e": 1}], "degraded"),
        ({"tickers_partial": 1, "tickers_failed": 4}, [{"e": 1}], "degraded"),
        ({"tickers_failed": 5}, [{"e": 1}], "failed"),
        ({"tickers_rejected": 5}, [{"e": 1}], "failed"),
        # A counter can say a target was unsettled even when nothing was
        # appended to ``errors``; that is still not a success.
        ({"tickers_failed": 5}, [], "failed"),
    ],
)
def test_component_status_reads_settlement_not_error_presence(counts, errors, expected):
    assert component_status(counts, errors, **YAHOO_COUNTERS) == expected


def result_of(status, *, mandatory=True, name="x"):
    return ComponentResult(
        name=name,
        status=status,
        counts={},
        errors=[],
        duration_ms=0,
        mandatory=mandatory,
        run_id_base="base",
    )


@pytest.mark.parametrize(
    "statuses,expected",
    [
        (["success", "success"], "success"),
        (["success", "degraded"], "degraded"),
        (["success", "failed"], "degraded"),
        (["degraded", "failed"], "degraded"),
        (["failed", "failed"], "failed"),
        ([], "success"),
    ],
)
def test_invocation_status_keeps_component_nuance(statuses, expected):
    components = [result_of(status) for status in statuses]
    assert invocation_status(components) == expected


def test_an_optional_component_failing_alone_does_not_fail_the_invocation():
    components = [result_of("success"), result_of("failed", mandatory=False)]
    assert invocation_status(components) == "degraded"


def test_exit_codes_distinguish_all_three_outcomes():
    assert EXIT_CODES["success"] == 0
    assert EXIT_CODES["degraded"] == 1
    assert EXIT_CODES["failed"] == 2
    assert len(set(EXIT_CODES[key] for key in ("success", "degraded", "failed"))) == 3


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def seeded(tmp_path, config, monkeypatch):
    """A database with real Yahoo and RSS evidence already persisted."""

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    run_live(repository, **config, invocation_id="seed")
    return repository


def test_replay_touches_no_network_at_all(tmp_path, config, monkeypatch):
    """Structural, not documentary: the HTTP callable refuses to be called.

    ``run_replay`` builds its fetcher with a ``get`` that raises, and the
    test wiring deliberately does not override it. A replay that reached
    for a feed would fail here rather than quietly refetch.
    """

    repository = seeded(tmp_path, config, monkeypatch)
    calls: list[str] = []

    def exploding_get(url, **kwargs):
        calls.append(url)
        raise AssertionError("replay must not fetch")

    monkeypatch.setattr("requests.get", exploding_get)

    result = run_replay(repository, **config)

    assert result.status == "success"
    assert calls == []
    with pytest.raises(RuntimeError, match="must not fetch"):
        pipeline._refuse_network("https://example.com/alpha")


def test_replay_builds_its_fetcher_with_a_refusing_http_callable(
    tmp_path, config, monkeypatch
):
    """The guarantee lives in the wiring, not in a component's good manners.

    ``reclassify_persisted`` happens not to call ``get`` today. Asserting
    only that no request went out would therefore pass even if replay were
    handed a live session -- so what is asserted is the callable itself.
    """

    repository = seeded(tmp_path, config, monkeypatch)
    captured: dict[str, object] = {}
    real_rss = pipeline.RSSFetcher

    def spy(repository, **options):
        captured.update(options)
        return real_rss(repository, **options)

    monkeypatch.setattr(pipeline, "RSSFetcher", spy)

    run_replay(repository, **config)

    assert captured["get"] is pipeline._refuse_network


def test_replay_never_deletes_raw_evidence(tmp_path, config, monkeypatch):
    repository = seeded(tmp_path, config, monkeypatch)
    before = {
        table: repository.count(table)
        for table in ("raw_items", "feed_snapshots", "raw_item_feeds")
    }
    raw_before = {row["id"]: row["raw_json"] for row in reader(repository).raw_items()}

    run_replay(repository, **config)

    after = {table: repository.count(table) for table in before}
    assert after == before
    raw_after = {row["id"]: row["raw_json"] for row in reader(repository).raw_items()}
    assert raw_after == raw_before


def test_replay_leaves_unrelated_partitions_untouched(tmp_path, config, monkeypatch):
    """Yahoo's partitions are not the replay's to rewrite."""

    repository = seeded(tmp_path, config, monkeypatch)
    yahoo_before = runs(repository, stage="fetch_yahoo")
    source_state_before = reader(repository).source_state_rows()

    run_replay(repository, **config)

    assert runs(repository, stage="fetch_yahoo") == yahoo_before
    assert reader(repository).source_state_rows() == source_state_before


def test_replay_is_idempotent(tmp_path, config, monkeypatch):
    """Same stored bytes, same derived state, however many times it runs."""

    repository = seeded(tmp_path, config, monkeypatch)
    run_replay(repository, **config)
    first = {
        "associations": reader(repository).raw_item_associations(),
        "candidates": reader(repository).raw_item_candidates(),
        "evidence": reader(repository).raw_item_match_evidence(),
    }

    second_result = run_replay(repository, **config)
    second = {
        "associations": reader(repository).raw_item_associations(),
        "candidates": reader(repository).raw_item_candidates(),
        "evidence": reader(repository).raw_item_match_evidence(),
    }

    assert second == first
    assert second_result.status == "success"


def test_replay_reports_what_it_cannot_rebuild(tmp_path, config, monkeypatch):
    """No claim of "full pipeline replay" while nothing downstream exists."""

    capabilities = replay_capabilities()
    assert capabilities["supported"] == ["rss_relevance"]
    assert "dedup" in capabilities["unsupported"]
    assert "summarization" in capabilities["unsupported"]
    assert capabilities["downstream_stages_registered"] == 0
    assert capabilities["scoped_replay_available"] is False

    repository = seeded(tmp_path, config, monkeypatch)
    result = run_replay(repository, **config)
    assert result.as_dict()["replay"] == capabilities


def test_replay_does_not_invoke_yahoo(tmp_path, config, monkeypatch):
    repository = seeded(tmp_path, config, monkeypatch)

    class Forbidden:
        def __init__(self, repository, **options):
            raise AssertionError("replay must not build a Yahoo fetcher")

    monkeypatch.setattr(pipeline, "YahooFinanceFetcher", Forbidden)

    result = run_replay(repository, **config)

    assert [item.name for item in result.components] == ["rss_relevance_replay"]


def test_replay_records_its_own_runs_as_replay_partitions(
    tmp_path, config, monkeypatch
):
    repository = seeded(tmp_path, config, monkeypatch)

    run_replay(repository, **config, invocation_id="replay-inv")

    rows = repository.run_log_entries(stage="reclassify_rss")
    assert rows
    for row in rows:
        assert row["run_id"].startswith("replay-inv:rss_relevance_replay:")
        assert row["run_id"] != "replay-inv"


# ---------------------------------------------------------------------------
# Structured logging and redaction
# ---------------------------------------------------------------------------


def logged(caplog):
    return [json.loads(record.message) for record in caplog.records]


def test_invocation_logging_is_structured_and_correlated(
    tmp_path, config, monkeypatch, caplog
):
    caplog.set_level("INFO", logger="phase0.pipeline")
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    run_live(repository, **config, invocation_id="inv")

    events = logged(caplog)
    by_event = {payload["event"] for payload in events}
    assert by_event == {
        "invocation_started",
        "component_completed",
        "invocation_completed",
    }
    assert all(payload["invocation_id"] == "inv" for payload in events)
    completed = next(p for p in events if p["event"] == "invocation_completed")
    assert completed["status"] == "success"
    assert completed["mode"] == "live"
    started = next(p for p in events if p["event"] == "invocation_started")
    assert started["schema_version"] == repository.schema_version()
    assert started["stages"] == ["yahoo", "rss"]
    assert {item["component"] for item in completed["components"]} == {"yahoo", "rss"}
    assert all("duration_ms" in item for item in completed["components"])


def test_credentials_never_reach_the_structured_log(
    tmp_path, config, monkeypatch, caplog
):
    """Redaction covers what the pipeline builds, not only what it forwards.

    A component error arrives already redacted; an error made here out of
    an exception message has not been through ``redact_secrets`` yet, and
    an exception is exactly where a bearer token tends to end up.
    """

    caplog.set_level("INFO", logger="phase0.pipeline")
    repository = migrated(tmp_path)

    class Leaking:
        def __init__(self, repository, **options):
            pass

        def fetch(self, **kwargs):
            raise RuntimeError("upstream rejected token=abc123secret")

    class Chatty:
        def __init__(self, repository, **options):
            pass

        def fetch(self, **kwargs):
            return (
                {"feeds_succeeded": 1},
                [{"api_key": "top-secret-value", "error": "failed"}],
            )

    wire(monkeypatch, yahoo=Leaking, rss=Chatty)

    result = run_live(repository, **config)

    assert "abc123secret" not in caplog.text
    assert "top-secret-value" not in caplog.text
    assert component(result, "rss").errors[0]["api_key"] == "[REDACTED]"
    assert "abc123secret" not in json.dumps(result.as_dict())


def test_a_skipped_invocation_says_so(tmp_path, config, monkeypatch, caplog):
    caplog.set_level("INFO", logger="phase0.pipeline")
    database = tmp_path / "phase0.sqlite3"
    lock = tmp_path / "phase0.lock"
    wire(monkeypatch, ticker_factory=provider(), get=responder())

    with pipeline.single_instance(lock) as held:
        assert held is True
        code = main(
            [
                "--database",
                str(database),
                "--feeds",
                str(config["feeds_path"]),
                "--aliases",
                str(config["aliases_path"]),
                "--lock-file",
                str(lock),
            ]
        )

    assert code == 0
    skipped = [p for p in logged(caplog) if p["event"] == "invocation_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["mode"] == "live"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli(tmp_path, config, extra):
    database = tmp_path / "phase0.sqlite3"
    return main(
        [
            "--database",
            str(database),
            "--feeds",
            str(config["feeds_path"]),
            "--aliases",
            str(config["aliases_path"]),
            "--lock-file",
            str(tmp_path / "phase0.lock"),
            *extra,
        ]
    )


def test_cli_live_exits_zero_on_success(tmp_path, config, monkeypatch):
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    assert cli(tmp_path, config, []) == 0


def test_cli_live_exits_one_when_degraded(tmp_path, config, monkeypatch):
    wire(monkeypatch, ticker_factory=provider(failing={"NVDA"}), get=responder())
    assert cli(tmp_path, config, []) == 1


def test_cli_live_exits_two_when_every_source_fails(tmp_path, config, monkeypatch):
    wire(
        monkeypatch,
        ticker_factory=provider(failing=set(TICKERS)),
        get=responder(failing={"alpha", "beta"}),
    )
    assert cli(tmp_path, config, []) == 2


def test_cli_replay_exits_zero_after_a_live_run(tmp_path, config, monkeypatch):
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    assert cli(tmp_path, config, []) == 0
    assert cli(tmp_path, config, ["--replay"]) == 0


def test_cli_status_reports_durable_partition_rows(
    tmp_path, config, monkeypatch, capsys
):
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    cli(tmp_path, config, [])
    capsys.readouterr()

    assert cli(tmp_path, config, ["--status"]) == 0

    report = json.loads(capsys.readouterr().out)
    stages = {row["stage"] for row in report["stages"]}
    assert "fetch_yahoo" in stages and "fetch_rss" in stages
    assert report["data_as_of"] is not None
    assert report["replay"]["scoped_replay_available"] is False


def test_cli_database_info_reports_version_and_migrations(
    tmp_path, config, monkeypatch, capsys
):
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    cli(tmp_path, config, [])
    capsys.readouterr()

    assert cli(tmp_path, config, ["--database-info"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert (
        report["schema_version"]
        == Phase0Repository(tmp_path / "phase0.sqlite3").schema_version()
    )
    assert report["applied_migrations"][0] == "001_initial.sql"
    assert report["counts"]["raw_items"] > 0


def test_status_and_database_reports_take_no_lock(tmp_path, config, monkeypatch):
    """A read-only report must work while a live invocation is running."""

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=provider(), get=responder())
    with pipeline.single_instance(tmp_path / "phase0.lock") as held:
        assert held is True
        assert cli(tmp_path, config, ["--status"]) == 0
        assert cli(tmp_path, config, ["--database-info"]) == 0
    assert status_report(repository)["schema_version"] >= 15
    assert database_report(repository)["schema_version"] >= 15


def test_cli_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--status", "--replay"])


def test_database_environment_default_is_honored(tmp_path, monkeypatch):
    database = tmp_path / "configured.sqlite3"
    monkeypatch.setenv("PHASE0_DATABASE_PATH", str(database))

    args = build_parser().parse_args(["--status"])

    assert args.database == database


def test_execute_stage_reports_a_base_run_id_per_component():
    result = execute_stage(
        Stage("thing", lambda base: ({"settled": 1}, []), settled=("settled",)),
        invocation_id="inv",
    )
    assert result.run_id_base == "inv:thing"
    assert result.status == "success"
    assert result.duration_ms >= 0

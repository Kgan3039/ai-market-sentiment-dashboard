"""The live pipeline continues from ingestion into stories and themes.

These tests drive ``run_live`` end to end with fake providers, a fake
socket, and a deterministic fake encoder: real runs, real partitions, real
``run_log`` rows, real M2/M3/M5 -- only the outside world is stubbed.

What is pinned here is the *integration*: that ingestion's persisted
evidence becomes persisted stories and themes automatically, which days
are selected for that, how a failure or a degradation in that work is
reported at the top, and that none of it loads a model or reaches a
network.  The reconcilers and the coordinator have their own suites for
what they do inside a partition.
"""

from __future__ import annotations

import inspect
import json
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone

import pytest

import nlp.embeddings
import pipeline
from ai.summarization import GeminiClient
from nlp.embeddings import EmbeddingModelLoadError
from phase0.repository import (
    STAGE_DEGRADED,
    Phase0Repository,
    StageEpisode,
    StageOutcome,
)
from phase0.stories import DEGRADATION_REASON as M3_UNAVAILABLE
from phase0.yahoo import TICKERS
from pipeline import (
    DOWNSTREAM_STAGES,
    EVIDENCE_STAGES,
    INTELLIGENCE_STAGE,
    PIPELINE_VERSION,
    RETRY_COMPONENT,
    RETRY_HORIZON,
    IntelligenceSelection,
    episode_anchor,
    intelligence_stage,
    is_retry_run,
    needs_recovery,
    retryable,
    run_live,
    select_intelligence_days,
    unresolved,
)
from test_phase0_pipeline import (
    FakeEncoder,
    component,
    provider,
    responder,
    wire,
    write_aliases,
    write_feeds,
)

NOW = datetime(2026, 9, 15, 20, 0, tzinfo=timezone.utc)
TODAY = NOW.date().isoformat()


class ManualClock:
    """A clock a test advances by hand.

    Handed to the repository, it stamps every run; handed to ``run_live``
    as ``now``, it drives the retry cutoff.  One instant, two readers, so
    "is this failure recent" compares against the timestamp the failure
    was actually written with.
    """

    def __init__(self, start: datetime = NOW) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> datetime:
        self.current = self.current + delta
        return self.current


def clocked(tmp_path, clock: ManualClock) -> Phase0Repository:
    repository = Phase0Repository(tmp_path / "phase0.sqlite3", clock=clock)
    repository.migrate()
    return repository


def migrated(tmp_path) -> Phase0Repository:
    """A repository whose clock is frozen at ``NOW``.

    Every run it stamps and every cutoff the pipeline computes come from
    this one instant, so a test never races the wall clock.
    """

    return clocked(tmp_path, ManualClock())


def selection_for(repository, *, invocation_id: str, now: datetime = NOW):
    return select_intelligence_days(
        repository,
        invocation_id=invocation_id,
        pipeline_version=PIPELINE_VERSION,
        now=now,
    )


class RaisingEncoder(FakeEncoder):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    def embed_batch(self, texts):
        super().embed_batch(texts)
        raise self._error


@pytest.fixture
def config(tmp_path):
    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    write_feeds(feeds, ["alpha", "beta"])
    write_aliases(aliases)
    return {"feeds_path": feeds, "aliases_path": aliases}


#: Production A3b settings a developer's shell may export.  These tests
#: never exercise summaries, so every one is cleared and the provider
#: client's connection factory refuses: an ambient
#: ``PHASE0_SUMMARIES_ENABLED=1`` cannot change what they test or reach a
#: network.
_AMBIENT_SUMMARY_ENV = (
    "PHASE0_SUMMARIES_ENABLED",
    "PHASE0_SUMMARIES_MAX_PROVIDER_CALLS",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_MAX_OUTPUT_TOKENS",
    "GEMINI_TIMEOUT_MS",
)


@pytest.fixture(autouse=True)
def no_ambient_summary_config(monkeypatch):
    for name in _AMBIENT_SUMMARY_ENV:
        monkeypatch.delenv(name, raising=False)

    def refuse(*args, **kwargs):
        raise AssertionError("a real summary provider client was requested")

    monkeypatch.setattr(GeminiClient, "_get_client", refuse)


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch):
    """Belt and braces: the default service is a fake, the factory refuses."""

    fake = FakeEncoder()
    monkeypatch.setattr(nlp.embeddings, "get_default_service", lambda: fake)

    def refuse(*args, **kwargs):
        raise AssertionError("a real embedding model was requested in a test")

    monkeypatch.setattr(nlp.embeddings, "_default_encoder_factory", refuse)
    return fake


def live(repository, config, *, encoder=None, invocation_id=None, **kw):
    return run_live(
        repository,
        **config,
        encoder=encoder if encoder is not None else FakeEncoder(),
        invocation_id=invocation_id,
        **kw,
    )


def stage_rows(repository: Phase0Repository, stage: str) -> list[dict]:
    return [row for row in repository.read.run_log_rows() if row["stage"] == stage]


def theme_set(repository: Phase0Repository, ticker: str, day: str):
    return repository.theme_set(
        ticker=ticker, trading_day=day, pipeline_version=pipeline.PIPELINE_VERSION
    )


def article(ticker: str, moment: datetime, *, index: int = 0) -> dict:
    """One fake Yahoo item with a title distinct enough to cluster on."""

    return {
        "title": f"{ticker} topic {index % 2} variant {index}",
        "link": f"https://example.com/{ticker.lower()}-{moment.isoformat()}-{index}",
        "publisher": "Example News",
        "providerPublishTime": int(moment.timestamp()),
    }


def pinned_provider():
    """One article per configured ticker, published at the frozen ``NOW``.

    The pipeline module's default ``headline`` stamps the real wall clock,
    which can sit on a different UTC day from the frozen repository clock
    these tests run under; pinning every fixture to ``NOW`` keeps
    ``TODAY`` meaning the day the evidence actually lands on.
    """

    return provider(
        news_by_ticker={ticker: [article(ticker, NOW)] for ticker in TICKERS}
    )


def newsroom(**by_ticker: list[dict]) -> dict[str, list[dict]]:
    """A provider payload that a test can change *between* invocations.

    ``wire`` can only be applied once per test -- a second call wraps its
    own wrapper -- so the provider is wired once over this dict and the
    dict is edited to describe what the next fetch should return.
    """

    return {ticker: list(items) for ticker, items in by_ticker.items()}


def breaking_themes(monkeypatch) -> set[str]:
    """Make M5 fail for the tickers in the returned set, and only those.

    A failure injected into the encoder fails M3 first, which fails the
    stories and leaves themes unattempted.  Breaking ``cluster_themes``
    itself is how a test gets the other shape -- stories persisted, themes
    failed -- which is the one the retry policy is mostly about.
    """

    import phase0.themes as themes_module

    broken: set[str] = set()
    real = themes_module.cluster_themes

    def cluster(stories, *, ticker, **kwargs):
        if ticker in broken:
            raise RuntimeError(f"clustering unavailable for {ticker}")
        return real(stories, ticker=ticker, **kwargs)

    monkeypatch.setattr(themes_module, "cluster_themes", cluster)
    return broken


# ----------------------------------------------------------------------
# 1, 2, 15, 16 -- the normal path
# ----------------------------------------------------------------------


def test_a_live_run_produces_persisted_stories_and_themes(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())
    encoder = FakeEncoder()

    result = live(repository, config, encoder=encoder, invocation_id="inv")

    assert result.status == "success"
    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.status == "success"
    assert intelligence.counts["stories_succeeded"] >= len(TICKERS)
    assert intelligence.counts["themes_succeeded"] >= len(TICKERS)
    assert intelligence.errors == []

    # Every configured ticker got its own stories and theme run today.
    days = {row["trading_day"] for row in stage_rows(repository, "stories")}
    assert TODAY in days
    for ticker in TICKERS:
        assert repository.stories_for_day(TODAY, ticker)
        assert theme_set(repository, ticker, TODAY) is not None
        assert [
            row["status"]
            for row in stage_rows(repository, "themes")
            if row["ticker"] == ticker and row["trading_day"] == TODAY
        ] == ["success"]
    # The encoder was used -- and it was ours, not a downloaded model.
    assert encoder.calls
    assert repository.count("embeddings") == 0


def test_the_stage_is_registered_as_a_builder_bound_at_run_time():
    assert DOWNSTREAM_STAGES[0] is intelligence_stage
    assert not isinstance(intelligence_stage, pipeline.Stage)


def test_ingestion_only_behaviour_is_unchanged(tmp_path, config, monkeypatch):
    """Everything the ingestion components did, they still do."""

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())

    result = live(repository, config)

    yahoo = component(result, "yahoo")
    rss = component(result, "rss")
    assert yahoo.status == "success"
    assert yahoo.counts["tickers_succeeded"] == len(TICKERS)
    assert rss.status == "success"
    assert rss.counts["feeds_succeeded"] == 2
    assert [item.name for item in result.components] == [
        "yahoo",
        "rss",
        INTELLIGENCE_STAGE,
    ]


def test_no_summary_or_model_api_is_called(tmp_path, config, monkeypatch):
    """No LLM, no summarizer, no remote model -- only the injected encoder."""

    import nlp.themes.summarization as summarization

    called: list[str] = []
    for name in ("summarizer_inputs", "adapt_theme_set", "validate_theme_set"):
        original = getattr(summarization, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(summarization, name, spy)
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())

    result = live(repository, config)

    assert result.status == "success"
    assert called == []


# ----------------------------------------------------------------------
# 3, 4, 8 -- idempotency, late evidence, and days with nothing in them
# ----------------------------------------------------------------------


def test_an_unchanged_rerun_is_idempotent(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())
    live(repository, config, invocation_id="inv-1")
    before = {
        ticker: [
            (t["theme_key"], t["fingerprint"])
            for t in theme_set(repository, ticker, TODAY)["themes"]
        ]
        for ticker in TICKERS
    }
    stories_before = {
        ticker: [
            (row["id"], row["cluster_fingerprint"], row["content_hash"])
            for row in repository.stories_for_day(TODAY, ticker)
        ]
        for ticker in TICKERS
    }

    result = live(repository, config, invocation_id="inv-2")

    intelligence = component(result, INTELLIGENCE_STAGE)
    # The rerun changed no evidence -- every provider item was already
    # stored -- so no partition was touched and nothing was reconciled:
    # the component settles with nothing to do, not with a failure.
    assert intelligence.counts["partitions_touched"] == 0
    assert intelligence.counts["days_selected"] == 0
    assert intelligence.counts["partitions"] == 0
    assert intelligence.errors == []
    assert intelligence.status == "success"
    assert result.status == "success"
    for ticker in TICKERS:
        assert [
            (t["theme_key"], t["fingerprint"])
            for t in theme_set(repository, ticker, TODAY)["themes"]
        ] == before[ticker]
        assert [
            (row["id"], row["cluster_fingerprint"], row["content_hash"])
            for row in repository.stories_for_day(TODAY, ticker)
        ] == stories_before[ticker]


def test_late_evidence_rebuilds_its_historical_utc_partition(
    tmp_path, config, monkeypatch
):
    """A late article opens a run under its *published* day, and that day
    is what gets reconciled -- not the day the fetch happened."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", NOW)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    live(repository, config, invocation_id="inv-1")
    assert repository.stories_for_day(earlier_day, "NVDA") == []

    # The same fetch now also returns an article from two days ago.
    news["NVDA"].append(article("NVDA", earlier, index=1))
    result = live(repository, config, invocation_id="inv-2")

    intelligence = component(result, INTELLIGENCE_STAGE)
    selection = selection_for(repository, invocation_id="inv-2")
    assert ("NVDA", earlier_day) in selection.touched
    # Today's article was already stored: seen again, not new, not touched.
    assert ("NVDA", TODAY) not in selection.touched
    assert intelligence.counts["days_touched"] == 1
    assert repository.stories_for_day(earlier_day, "NVDA")
    assert theme_set(repository, "NVDA", earlier_day) is not None
    assert [
        row["run_id"]
        for row in stage_rows(repository, "stories")
        if row["trading_day"] == earlier_day and row["ticker"] == "NVDA"
    ] == [f"inv-2:{INTELLIGENCE_STAGE}:NVDA:{earlier_day}"]


def test_an_ingestion_run_with_no_evidence_produces_nothing(
    tmp_path, config, monkeypatch
):
    """Nothing changed, so nothing is selected: no fake story or theme."""

    repository = migrated(tmp_path)
    # Every ticker fails and both feeds are unreachable.  Yahoo still
    # settles a run per ticker for the fetch day -- a recorded failure --
    # but a run that inserted nothing changed nothing, and no partition
    # is touched by it.
    wire(
        monkeypatch,
        ticker_factory=provider(failing=set(TICKERS)),
        get=responder(failing={"alpha", "beta"}),
    )

    result = live(repository, config, invocation_id="inv")

    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.status == "success"
    assert stage_rows(repository, "fetch_yahoo")
    assert intelligence.counts["days_touched"] == 0
    assert intelligence.counts["days_selected"] == 0
    assert intelligence.counts["partitions"] == 0
    assert intelligence.counts.get("stories_succeeded", 0) == 0
    assert repository.count("stories") == 0
    assert repository.count("theme_sets") == 0
    assert stage_rows(repository, "stories") == []
    assert stage_rows(repository, "themes") == []
    assert result.status == "failed"


# ----------------------------------------------------------------------
# 5, 6, 7 -- the bounded retry
# ----------------------------------------------------------------------


def _age_intelligence_rows(repository: Phase0Repository, *, to: datetime) -> None:
    """Move every stories/themes completion into the past.

    The narrowest way to put an outcome outside the horizon without
    faking a clock inside the repository: the run happened, it settled,
    and only *when* is being changed.
    """

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE run_log SET started_at = ?, completed_at = ? "
            "WHERE stage IN ('stories', 'themes')",
            (to.isoformat(), to.isoformat()),
        )


def _seed_failed_partition(repository, config, monkeypatch, *, broken):
    """One NVDA article two days ago, whose themes fail.

    The failed partition is deliberately not today's: today is the fetch
    day, and the fetch-day checkpoint touches it on every invocation, so a
    failure there would be revisited as *touched* and prove nothing about
    the retry policy.
    """

    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    broken.add("NVDA")
    first = live(repository, config, invocation_id="a")
    assert component(first, INTELLIGENCE_STAGE).counts["themes_failed"] == 1
    assert theme_set(repository, "NVDA", earlier_day) is None
    assert [
        row["status"]
        for row in stage_rows(repository, "themes")
        if row["ticker"] == "NVDA"
    ] == ["failed"]
    # From here on the provider returns nothing for anyone.
    news.clear()
    return earlier_day


def test_a_recent_failed_partition_is_retried_without_new_evidence(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    broken = breaking_themes(monkeypatch)
    earlier_day = _seed_failed_partition(repository, config, monkeypatch, broken=broken)

    # M5 is back, and nothing new arrives for NVDA's day ...
    broken.clear()
    selection = selection_for(repository, invocation_id="b")
    assert earlier_day in selection.retried_days
    second = live(repository, config, invocation_id="b")

    # ... which is retried anyway, because its latest outcome was a recent
    # failure, and this time the themes land.  Afterwards it is not: the
    # newest outcome is a success.
    intelligence = component(second, INTELLIGENCE_STAGE)
    after = selection_for(repository, invocation_id="b")
    assert earlier_day not in after.touched_days
    assert earlier_day not in after.retried_days
    assert intelligence.counts["days_retried"] == 1
    assert theme_set(repository, "NVDA", earlier_day) is not None
    assert [
        row["status"]
        for row in stage_rows(repository, "themes")
        if row["ticker"] == "NVDA"
    ] == ["failed", "success"]


def test_a_failure_outside_the_horizon_is_not_swept(tmp_path, config, monkeypatch):
    """The retry is bounded: an old failure is left where it is."""

    repository = migrated(tmp_path)
    broken = breaking_themes(monkeypatch)
    earlier_day = _seed_failed_partition(repository, config, monkeypatch, broken=broken)
    broken.clear()

    # The failure is now just past the horizon.
    _age_intelligence_rows(repository, to=NOW - RETRY_HORIZON - timedelta(seconds=1))
    second = live(repository, config, invocation_id="b")

    intelligence = component(second, INTELLIGENCE_STAGE)
    assert intelligence.counts["days_retried"] == 0
    assert theme_set(repository, "NVDA", earlier_day) is None
    assert [
        row["run_id"]
        for row in stage_rows(repository, "themes")
        if row["ticker"] == "NVDA"
    ] == [f"a:{INTELLIGENCE_STAGE}:NVDA:{earlier_day}"]


def test_the_horizon_boundary_is_inclusive_of_the_edge(tmp_path, config, monkeypatch):
    """Exactly at the horizon still counts as recent."""

    repository = migrated(tmp_path)
    broken = breaking_themes(monkeypatch)
    earlier_day = _seed_failed_partition(repository, config, monkeypatch, broken=broken)
    _age_intelligence_rows(repository, to=NOW - RETRY_HORIZON)

    selection = selection_for(repository, invocation_id="none")

    assert selection.touched_days == ()
    assert selection.retried_days == (earlier_day,)
    assert selection.days == (earlier_day,)


def test_an_unchanged_replay_is_not_mistaken_for_a_retryable_failure(
    tmp_path, config, monkeypatch
):
    """Case 7: the ``partial`` quirk must not schedule a loop.

    A healthy identical rerun settles its stories run ``degraded`` with no
    errors, because unchanged rows count as partial work.  Nothing went
    wrong and nothing would change.  Selecting it would retry it every
    invocation for the whole horizon, for no reason.
    """

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    live(repository, config, invocation_id="a")
    # AMD's first article for the day brings the day back; NVDA, unchanged,
    # is replayed identically alongside it.
    news["AMD"] = [article("AMD", earlier)]
    live(repository, config, invocation_id="b")

    latest = {
        (row["ticker"], row["stage"]): row
        for row in repository.read.run_log_rows()
        if row["stage"] in ("stories", "themes") and row["trading_day"] == earlier_day
    }
    nvda_stories = latest[("NVDA", "stories")]
    assert nvda_stories["run_id"].startswith("b:")
    assert nvda_stories["status"] == "degraded"
    assert json.loads(nvda_stories["errors"]) == []

    news.clear()
    third = live(repository, config, invocation_id="c")

    assert component(third, INTELLIGENCE_STAGE).counts["days_retried"] == 0
    selection = selection_for(repository, invocation_id="c")
    assert earlier_day not in selection.days


def test_an_intentional_m3_degradation_is_retried_within_the_horizon(
    tmp_path, config, monkeypatch
):
    """Case 7, the other half: an explicit marker *is* worth retrying.

    M2-only stories are an honest intermediate generation, and the run
    says so with a ``stage_degraded`` marker.  A later run with the model
    back replaces them, so the marker selects the partition -- for as long
    as the horizon lasts, and no longer.
    """

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier, index=i) for i in range(3)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    first = live(
        repository,
        config,
        encoder=RaisingEncoder(EmbeddingModelLoadError("no model cache")),
        invocation_id="a",
    )
    intelligence = component(first, INTELLIGENCE_STAGE)
    assert intelligence.status == "degraded"
    assert intelligence.counts["stories_degraded"] >= 1
    assert {
        row["stage"] for row in repository.stories_for_day(earlier_day, "NVDA")
    } == {"m2.exact"}
    assert theme_set(repository, "NVDA", earlier_day) is None

    news.clear()
    selection = selection_for(repository, invocation_id="b")
    assert earlier_day in selection.retried_days
    second = live(repository, config, encoder=FakeEncoder(), invocation_id="b")

    # Every partition the unavailable model degraded is retried -- the
    # RSS-fed one included -- so the count is at least this day.
    assert component(second, INTELLIGENCE_STAGE).counts["days_retried"] >= 1
    assert {
        row["stage"] for row in repository.stories_for_day(earlier_day, "NVDA")
    } == {"m3.semantic"}
    assert theme_set(repository, "NVDA", earlier_day) is not None

    # A fresh degradation ages out of the horizon like any other outcome.
    live(
        repository,
        config,
        encoder=RaisingEncoder(EmbeddingModelLoadError("no model cache")),
        invocation_id="c",
    )
    _age_intelligence_rows(repository, to=NOW - RETRY_HORIZON - timedelta(hours=1))
    assert selection_for(repository, invocation_id="none").retried_days == ()


@pytest.mark.parametrize(
    "status, markers, expected",
    [
        ("failed", (), True),
        ("failed", (("stage_degraded", M3_UNAVAILABLE),), True),
        ("degraded", (("stage_degraded", M3_UNAVAILABLE),), True),
        ("degraded", (("stage_degraded", "m5_requires_semantic_stories"),), True),
        ("degraded", (), False),
        ("degraded", (("partition_error", None),), False),
        ("success", (), False),
        ("success", (("stage_degraded", M3_UNAVAILABLE),), False),
    ],
)
def test_unresolved_reads_the_marker_not_the_word(status, markers, expected):
    outcome = StageOutcome(
        run_id="x:intelligence:NVDA:" + TODAY,
        ticker="NVDA",
        trading_day=TODAY,
        stage="stories",
        pipeline_version=PIPELINE_VERSION,
        status=status,
        markers=markers,
        completed_at=NOW.isoformat(),
    )
    assert unresolved(outcome) is expected
    assert STAGE_DEGRADED == "stage_degraded"


def _attempt(run_id: str, *, at: datetime, status: str = "failed") -> StageOutcome:
    return StageOutcome(
        run_id=run_id,
        ticker="NVDA",
        trading_day=TODAY,
        stage="themes",
        pipeline_version=PIPELINE_VERSION,
        status=status,
        markers=(),
        completed_at=at.isoformat(),
    )


def _episode(*attempts: StageOutcome) -> StageEpisode:
    return StageEpisode(
        ticker="NVDA",
        trading_day=TODAY,
        stage="themes",
        pipeline_version=PIPELINE_VERSION,
        attempts=attempts,
    )


def test_the_anchor_is_the_newest_evidence_triggered_failure():
    """Retries never move it; a fresh evidence-triggered failure does."""

    first = _attempt(f"a:intelligence:NVDA:{TODAY}", at=NOW)
    retry_1 = _attempt(f"b:{RETRY_COMPONENT}:NVDA:{TODAY}", at=NOW + timedelta(hours=1))
    retry_2 = _attempt(f"c:{RETRY_COMPONENT}:NVDA:{TODAY}", at=NOW + timedelta(hours=2))
    evidence_again = _attempt(
        f"d:intelligence:NVDA:{TODAY}", at=NOW + timedelta(hours=3)
    )

    assert episode_anchor(_episode(first)) == NOW.isoformat()
    assert episode_anchor(_episode(first, retry_1, retry_2)) == NOW.isoformat()
    assert episode_anchor(_episode(first, retry_1, retry_2, evidence_again)) == (
        (NOW + timedelta(hours=3)).isoformat()
    )
    # An episode made only of retries cannot anchor anything.
    assert episode_anchor(_episode(retry_1, retry_2)) is None


def test_retryable_expires_from_the_anchor_not_from_the_latest_retry():
    """P2-A: additional failed retries do not renew the deadline."""

    first = _attempt(f"a:intelligence:NVDA:{TODAY}", at=NOW)
    late_retry = _attempt(
        f"z:{RETRY_COMPONENT}:NVDA:{TODAY}", at=NOW + RETRY_HORIZON + timedelta(days=1)
    )
    episode = _episode(first, late_retry)

    inside = (NOW - timedelta(hours=1)).isoformat()
    outside = (NOW + timedelta(seconds=1)).isoformat()
    assert retryable(episode, since=inside) is True
    # The newest attempt is far newer than ``since``; only the anchor counts.
    assert retryable(episode, since=outside) is False


def test_selection_unions_touched_and_retried_days(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    broken = breaking_themes(monkeypatch)
    earlier_day = _seed_failed_partition(repository, config, monkeypatch, broken=broken)
    broken.clear()
    other = NOW - timedelta(days=1)
    other_day = other.date().isoformat()

    selection_before = selection_for(repository, invocation_id="b")
    assert selection_before.touched_days == ()
    assert earlier_day in selection_before.retried_days

    selection = selection_for(repository, invocation_id="a")
    assert isinstance(selection, IntelligenceSelection)
    assert earlier_day in selection.touched_days
    assert earlier_day in selection.retried_days
    assert selection.days == tuple(
        sorted(
            set(selection.touched_days)
            | set(selection.retried_days)
            | set(selection.recovered_days)
        )
    )
    assert other_day not in selection.days


# ----------------------------------------------------------------------
# 9, 10, 11 -- the coordinator's guarantees survive the integration
# ----------------------------------------------------------------------


def test_one_partition_failing_does_not_stop_the_others(tmp_path, config, monkeypatch):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())
    breaking_themes(monkeypatch).add("AMD")

    result = live(repository, config)

    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.counts["themes_failed"] == 1
    assert intelligence.counts["themes_succeeded"] >= len(TICKERS) - 1
    assert theme_set(repository, "AMD", TODAY) is None
    for ticker in TICKERS:
        if ticker != "AMD":
            assert theme_set(repository, ticker, TODAY) is not None
    assert [error["ticker"] for error in intelligence.errors] == ["AMD"]


def test_previous_theme_identity_survives_the_live_path(tmp_path, config, monkeypatch):
    """Capture-before-story-write, through ``run_live``."""

    repository = migrated(tmp_path)
    news = newsroom(
        NVDA=[article("NVDA", NOW - timedelta(hours=h), index=h) for h in range(1, 7)]
    )
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    live(repository, config, invocation_id="a")
    before = {t["theme_key"] for t in theme_set(repository, "NVDA", TODAY)["themes"]}
    assert before

    # A seventh article moves membership, so story reconciliation deletes
    # the theme set before the themes stage opens.
    news["NVDA"].append(article("NVDA", NOW - timedelta(hours=7), index=7))
    result = live(repository, config, invocation_id="b")

    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.counts["previous_themes_captured"] >= len(before)
    after = theme_set(repository, "NVDA", TODAY)["themes"]
    carried = {t["theme_key"] for t in after if t["matched_previous_key"] is not None}
    assert carried, "no identity was carried across the invalidation"
    assert carried <= before


def test_m3_degradation_stays_truthful_through_the_live_path(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())

    result = live(
        repository,
        config,
        encoder=RaisingEncoder(EmbeddingModelLoadError("no model cache")),
        invocation_id="a",
    )

    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.status == "degraded"
    assert result.status == "degraded"
    assert result.exit_code == 1
    stories = [
        row for row in stage_rows(repository, "stories") if row["ticker"] == "NVDA"
    ]
    assert [row["status"] for row in stories] == ["degraded"]
    markers = [
        error
        for error in json.loads(stories[0]["errors"])
        if error.get("type") == STAGE_DEGRADED
    ]
    assert [marker["reason"] for marker in markers] == [M3_UNAVAILABLE]
    themes = [
        row for row in stage_rows(repository, "themes") if row["ticker"] == "NVDA"
    ]
    assert [row["status"] for row in themes] == ["degraded"]
    assert repository.count("theme_sets") == 0
    assert {row["stage"] for row in repository.stories_for_day(TODAY, "NVDA")} == {
        "m2.exact"
    }


# ----------------------------------------------------------------------
# 12, 13, 14 -- top-level status arithmetic
# ----------------------------------------------------------------------


def test_healthy_ingestion_with_failed_intelligence_is_degraded_never_success(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())
    breaking_themes(monkeypatch).update(TICKERS)

    result = live(repository, config)

    assert component(result, "yahoo").status == "success"
    assert component(result, "rss").status == "success"
    intelligence = component(result, INTELLIGENCE_STAGE)
    # Every theme run failed, but every story generation landed: the
    # component is degraded, not failed, and it is never success.
    assert intelligence.status == "degraded"
    assert intelligence.counts["themes_failed"] >= len(TICKERS)
    assert intelligence.counts["stories_succeeded"] >= len(TICKERS)
    assert result.status == "degraded"
    assert result.exit_code == 1


def test_healthy_ingestion_with_every_story_failing_is_degraded_never_success(
    tmp_path, config, monkeypatch
):
    """The other shape: M3 raises something nonrecoverable everywhere."""

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())

    result = live(repository, config, encoder=RaisingEncoder(RuntimeError("down")))

    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.status == "failed"
    assert intelligence.counts["stories_failed"] >= len(TICKERS)
    assert intelligence.counts["themes_not_attempted"] >= len(TICKERS)
    # Ingestion's evidence is durable and the invocation says so: degraded,
    # never success, and not failed either -- the mandatory work landed.
    assert result.status == "degraded"
    assert result.exit_code == 1


def test_healthy_ingestion_with_degraded_intelligence_is_degraded(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())
    breaking_themes(monkeypatch).add("NVDA")

    result = live(repository, config)

    assert component(result, INTELLIGENCE_STAGE).status == "degraded"
    assert result.status == "degraded"


def test_total_mandatory_ingestion_failure_remains_failed(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    wire(
        monkeypatch,
        ticker_factory=provider(failing=set(TICKERS)),
        get=responder(failing={"alpha", "beta"}),
    )

    result = live(repository, config)

    assert component(result, INTELLIGENCE_STAGE).status == "success"
    assert component(result, INTELLIGENCE_STAGE).mandatory is False
    assert result.status == "failed"
    assert result.exit_code == 2


def test_zero_eligible_partitions_do_not_degrade_a_healthy_invocation(
    tmp_path, config, monkeypatch
):
    """Nothing to reconcile is an honest success, and stays out of the way."""

    repository = migrated(tmp_path)
    # No invocation prefix matches, no recent failures: the selection is
    # empty and the component does nothing.
    selection = selection_for(repository, invocation_id="none")
    assert selection.days == ()

    stage = intelligence_stage(
        repository,
        pipeline_version=pipeline.PIPELINE_VERSION,
        invocation_id="none",
        encoder=FakeEncoder(),
    )
    counts, errors = stage.action("none:intelligence")
    assert counts["days_selected"] == 0
    assert counts["partitions"] == 0
    assert errors == []
    assert (
        pipeline.component_status(
            counts, errors, settled=stage.settled, unsettled=stage.unsettled
        )
        == "success"
    )


# ----------------------------------------------------------------------
# P1 -- a capture failure is one partition's failure
# ----------------------------------------------------------------------


def _breaking_capture(monkeypatch, ticker: str):
    """Make previous-theme capture raise for one ticker only."""

    import phase0.themes as themes_module

    real = themes_module.ThemeReconciler.capture_previous

    def capture(self, symbol, trading_day):
        if symbol == ticker:
            raise RuntimeError(f"capture exploded for {ticker}: api_key=sk-secret-42")
        return real(self, symbol, trading_day)

    monkeypatch.setattr(themes_module.ThemeReconciler, "capture_previous", capture)


def test_a_capture_failure_is_isolated_to_its_own_partition(
    tmp_path, config, monkeypatch
):
    """Five partitions; the middle one's capture raises.

    The earlier ones stay settled and counted, the later ones still run,
    the failed one leaves a durable ``themes`` failure a scheduler can find,
    and its stories are never touched.
    """

    repository = migrated(tmp_path)
    news = newsroom(**{ticker: [article(ticker, NOW)] for ticker in TICKERS})
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    # A healthy first pass so every partition has stories and themes.
    live(repository, config, invocation_id="a")
    for ticker in TICKERS:
        assert theme_set(repository, ticker, TODAY) is not None
    ordered = sorted(TICKERS)
    middle = ordered[2]
    stories_before = repository.stories_for_day(TODAY, middle)
    themes_before = theme_set(repository, middle, TODAY)
    story_runs_before = [
        row["run_id"]
        for row in stage_rows(repository, "stories")
        if row["ticker"] == middle
    ]

    # A second article for every ticker: all five partitions are touched.
    for ticker in TICKERS:
        news[ticker].append(article(ticker, NOW, index=1))
    _breaking_capture(monkeypatch, middle)
    result = live(repository, config, invocation_id="b")

    intelligence = component(result, INTELLIGENCE_STAGE)
    # Later partitions ran and settled.
    for ticker in ordered[3:]:
        assert [
            row["status"]
            for row in stage_rows(repository, "themes")
            if row["ticker"] == ticker and row["run_id"].startswith("b:")
        ] == ["success"]
    # Earlier ones too, and their counts are in the report.
    for ticker in ordered[:2]:
        assert any(
            row["run_id"].startswith("b:") and row["ticker"] == ticker
            for row in stage_rows(repository, "themes")
        )
    assert (
        intelligence.counts["stories_succeeded"]
        + intelligence.counts["stories_degraded"]
        >= len(TICKERS) - 1
    )
    assert intelligence.counts["stories_not_attempted"] == 1
    assert intelligence.counts["themes_failed"] == 1

    # The failed partition: no story run, no story mutation, a durable
    # themes failure attributed to the capture, redacted.
    assert [
        row["run_id"]
        for row in stage_rows(repository, "stories")
        if row["ticker"] == middle
    ] == story_runs_before
    assert repository.stories_for_day(TODAY, middle) == stories_before
    assert theme_set(repository, middle, TODAY) == themes_before
    failed = [
        row
        for row in stage_rows(repository, "themes")
        if row["ticker"] == middle and row["run_id"].startswith("b:")
    ]
    assert [row["status"] for row in failed] == ["failed"]
    assert "capture exploded" in failed[0]["errors"]
    assert "sk-secret-42" not in failed[0]["errors"]
    error = [e for e in intelligence.errors if e.get("ticker") == middle][0]
    assert error["type"] == "theme_partition_error"
    assert error["phase"] == "previous_theme_capture"
    assert "sk-secret-42" not in json.dumps(error)
    # And the component says what happened without pretending otherwise.
    assert intelligence.status == "degraded"
    assert result.status == "degraded"

    # A later invocation with no new evidence rediscovers it.
    monkeypatch.undo()
    selection = selection_for(repository, invocation_id="none")
    assert TODAY in selection.retried_days


def test_a_capture_failure_is_reported_through_the_coordinator(
    tmp_path, config, monkeypatch
):
    """The coordinator's own result says stories were not attempted."""

    from phase0.coordinator import PartitionCoordinator

    repository = migrated(tmp_path)
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())
    live(repository, config, invocation_id="a")
    _breaking_capture(monkeypatch, "NVDA")

    coordinator = PartitionCoordinator(
        repository, pipeline_version=PIPELINE_VERSION, encoder=FakeEncoder()
    )
    result = coordinator.run_partition("NVDA", TODAY, base_run_id="b:intelligence")

    assert result.stories.status == "not_attempted"
    assert result.stories.error["reason"] == "previous_theme_capture_failed"
    assert result.themes.status == "failed"
    assert result.themes.error["phase"] == "previous_theme_capture"
    assert result.previous_captured == 0
    # The next ticker is unaffected by the exception at all.
    other = coordinator.run_partition("AMD", TODAY, base_run_id="b:intelligence")
    assert other.themes.status == "success"


# ----------------------------------------------------------------------
# P2-A / P2-C -- the retry episode, under one controlled clock
# ----------------------------------------------------------------------


def _seed_failed_day(repository, config, monkeypatch, *, clock, broken):
    """A partition on a non-fetch day whose themes fail, then no new news."""

    earlier = clock.current - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    broken.add("NVDA")
    live(repository, config, invocation_id="seed")
    assert theme_set(repository, "NVDA", earlier_day) is None
    news.clear()
    return earlier_day, news


def test_a_permanent_failure_stops_being_retried_after_its_window(
    tmp_path, config, monkeypatch
):
    """P2-A/B and P2-C in one scenario, no timestamp rewriting.

    1. a failure opens an episode; 2. inside the window it is eligible;
    3. a failed retry is recorded; 4. it does *not* renew the deadline;
    5. past the original deadline the partition is left alone; 6. and a
    later success would have closed it.  The clock is the repository's,
    so every ``completed_at`` compared against is one this test advanced.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    broken = breaking_themes(monkeypatch)
    earlier_day, _ = _seed_failed_day(
        repository, config, monkeypatch, clock=clock, broken=broken
    )
    opened_at = clock.current

    # 2. Well inside the window: eligible.
    clock.advance(RETRY_HORIZON - timedelta(hours=6))
    assert (
        earlier_day
        in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )

    # 3. A retry runs and fails again (M5 is still broken).
    retried = live(repository, config, invocation_id="retry-1")
    assert component(retried, INTELLIGENCE_STAGE).counts["days_retried"] == 1
    newest = [
        row for row in stage_rows(repository, "themes") if row["ticker"] == "NVDA"
    ][-1]
    assert is_retry_run(newest["run_id"])
    assert newest["status"] == "failed"

    # 4. The failed retry did not move the anchor: just past the ORIGINAL
    #    deadline the episode is expired, even though the newest failure
    #    is only six hours old.
    clock.advance(timedelta(hours=6, seconds=1))
    assert clock.current > opened_at + RETRY_HORIZON
    assert clock.current < newest_completed(repository, "NVDA") + RETRY_HORIZON
    assert (
        earlier_day
        not in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )

    # 5. A no-new-evidence invocation leaves it alone.
    left = live(repository, config, invocation_id="later")
    assert component(left, INTELLIGENCE_STAGE).counts["days_retried"] == 0
    assert theme_set(repository, "NVDA", earlier_day) is None


def newest_completed(repository, ticker: str) -> datetime:
    rows = [row for row in stage_rows(repository, "themes") if row["ticker"] == ticker]
    return datetime.fromisoformat(rows[-1]["completed_at"])


def test_a_success_closes_the_episode(tmp_path, config, monkeypatch):
    """P2-A regression C."""

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    broken = breaking_themes(monkeypatch)
    earlier_day, _ = _seed_failed_day(
        repository, config, monkeypatch, clock=clock, broken=broken
    )
    assert (
        earlier_day
        in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )

    broken.clear()
    clock.advance(timedelta(hours=1))
    live(repository, config, invocation_id="heal")
    assert theme_set(repository, "NVDA", earlier_day) is not None

    assert (
        earlier_day
        not in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )
    assert (
        repository.read.stage_outcome_episodes(
            ("stories", "themes"),
            pipeline_version=PIPELINE_VERSION,
            completed_since="2000-01-01T00:00:00+00:00",
        )
        == []
    )


def test_new_evidence_after_an_expired_episode_opens_a_fresh_window(
    tmp_path, config, monkeypatch
):
    """P2-A regressions D and E.

    An expired episode does not block the touched-day path, and when the
    evidence-triggered attempt fails again the new window is anchored on
    *that* failure, not on the historical one.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    broken = breaking_themes(monkeypatch)
    earlier_day, news = _seed_failed_day(
        repository, config, monkeypatch, clock=clock, broken=broken
    )
    clock.advance(RETRY_HORIZON + timedelta(days=1))
    assert (
        earlier_day
        not in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )

    # D. New evidence for the old day arrives; it is processed as touched.
    earlier = datetime.fromisoformat(earlier_day + "T12:00:00+00:00")
    news["NVDA"] = [article("NVDA", earlier, index=5)]
    fresh = live(repository, config, invocation_id="fresh")
    intelligence = component(fresh, INTELLIGENCE_STAGE)
    assert (
        earlier_day
        in selection_for(
            repository, invocation_id="fresh", now=clock.current
        ).touched_days
    )
    assert intelligence.counts["themes_failed"] == 1  # M5 still broken
    reopened_at = clock.current
    newest = [
        row for row in stage_rows(repository, "themes") if row["ticker"] == "NVDA"
    ][-1]
    assert not is_retry_run(newest["run_id"])

    # E. The new window is measured from the fresh failure.
    news.clear()
    clock.advance(RETRY_HORIZON - timedelta(hours=1))
    assert (
        earlier_day
        in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )
    clock.advance(timedelta(hours=1, seconds=1))
    assert clock.current > reopened_at + RETRY_HORIZON
    assert (
        earlier_day
        not in selection_for(
            repository, invocation_id="none", now=clock.current
        ).retried_days
    )


def test_the_default_invocation_needs_no_clock(tmp_path, config, monkeypatch):
    """Production supplies no clock and no ``now`` and still works."""

    repository = Phase0Repository(tmp_path / "wall.sqlite3")
    repository.migrate()
    wire(monkeypatch, ticker_factory=pinned_provider(), get=responder())

    result = run_live(repository, **config, encoder=FakeEncoder())

    assert result.status == "success"
    assert repository.now().tzinfo is not None
    assert "now" not in inspect.signature(run_live).parameters
    assert "now" not in inspect.signature(intelligence_stage).parameters
    assert component(result, INTELLIGENCE_STAGE).counts["days_selected"] >= 1


# ----------------------------------------------------------------------
# P2-B -- one version never sees another's outcomes
# ----------------------------------------------------------------------


def _settle_theme_run(repository, *, run_id, version, day, status_ok: bool):
    with pytest.raises(RuntimeError) if not status_ok else nullcontext():
        with repository.stage_run(
            run_id=run_id,
            stage="themes",
            trading_day=day,
            pipeline_version=version,
            ticker="NVDA",
        ) as run:
            if status_ok:
                repository.clear_theme_set(
                    run=run,
                    ticker="NVDA",
                    trading_day=day,
                    pipeline_version=version,
                    terminal=True,
                )
            else:
                raise RuntimeError("themes failed")


def test_a_v2_success_does_not_hide_a_v1_failure(tmp_path):
    repository = migrated(tmp_path)
    _settle_theme_run(
        repository,
        run_id=f"a:intelligence:NVDA:{TODAY}",
        version="v1",
        day=TODAY,
        status_ok=False,
    )
    _settle_theme_run(
        repository,
        run_id=f"b:intelligence:NVDA:{TODAY}",
        version="v2",
        day=TODAY,
        status_ok=True,
    )

    v1 = select_intelligence_days(
        repository, invocation_id="none", pipeline_version="v1", now=NOW
    )
    v2 = select_intelligence_days(
        repository, invocation_id="none", pipeline_version="v2", now=NOW
    )
    assert v1.retried_days == (TODAY,)
    assert v2.retried_days == ()


def test_a_v2_failure_does_not_schedule_v1_work(tmp_path):
    repository = migrated(tmp_path)
    _settle_theme_run(
        repository,
        run_id=f"a:intelligence:NVDA:{TODAY}",
        version="v1",
        day=TODAY,
        status_ok=True,
    )
    _settle_theme_run(
        repository,
        run_id=f"b:intelligence:NVDA:{TODAY}",
        version="v2",
        day=TODAY,
        status_ok=False,
    )

    v1 = select_intelligence_days(
        repository, invocation_id="none", pipeline_version="v1", now=NOW
    )
    v2 = select_intelligence_days(
        repository, invocation_id="none", pipeline_version="v2", now=NOW
    )
    assert v1.retried_days == ()
    assert v2.retried_days == (TODAY,)


def _evidence(ticker: str, day: str, index: int) -> dict:
    return {
        "source": f"yahoo:{ticker}",
        "ticker": ticker,
        "title": f"{ticker} headline {index}",
        "url": f"https://example.com/{ticker.lower()}-{index}",
        "canonical_url": f"https://example.com/{ticker.lower()}-{index}",
        "published_at": f"{day}T12:00:00+00:00",
        "fetched_at": f"{day}T12:30:00+00:00",
        "raw_json": {"index": index},
    }


def test_touched_partitions_are_the_ones_evidence_stages_changed(tmp_path):
    """Observation A, at partition grain: inclusion by stage, and by change.

    A run counts only if it is an evidence stage *and* recorded that it
    inserted or re-associated something.  A run of the right stage that
    saw only what was already stored -- the fetch-day checkpoint, a
    duplicate -- is not a touch, and neither is any run of another stage.
    """

    repository = migrated(tmp_path)

    def run(run_id, *, stage, day, ticker=None, items=()):
        with repository.stage_run(
            run_id=run_id,
            stage=stage,
            trading_day=day,
            pipeline_version=PIPELINE_VERSION,
            ticker=ticker,
        ) as ctx:
            if items:
                repository.ingest_raw_items(list(items), run=ctx, terminal=True)

    # An insert under NVDA/09-13: touched.
    run(
        "inv:yahoo:NVDA:2026-09-13",
        stage="fetch_yahoo",
        day="2026-09-13",
        ticker="NVDA",
        items=[_evidence("NVDA", "2026-09-13", 1)],
    )
    # The same item seen again under AMD-less, later invocation: not new.
    run(
        "inv:yahoo:NVDA:2026-09-12",
        stage="fetch_yahoo",
        day="2026-09-12",
        ticker="NVDA",
        items=[_evidence("NVDA", "2026-09-12", 2)],
    )
    run(
        "inv-2:yahoo:NVDA:2026-09-12",
        stage="fetch_yahoo",
        day="2026-09-12",
        ticker="NVDA",
        items=[_evidence("NVDA", "2026-09-12", 2)],  # duplicate
    )
    # A fetch-day run with nothing to ingest: an evidence stage, no change.
    run("inv:yahoo:AMD:2026-09-15", stage="fetch_yahoo", day="2026-09-15", ticker="AMD")
    # Runs of other stages under the same prefix, whatever they did.
    for run_id, stage, day in (
        ("inv:rss:alpha:checkpoint:2026-09-15", "checkpoint_rss", "2026-09-15"),
        ("inv:rss:alpha:snapshot:2026-09-15", "fetch_rss", "2026-09-15"),
        ("inv:rss:alpha:observe:2026-09-14", "observe_rss", "2026-09-14"),
        ("inv:intelligence:NVDA:2026-09-11", "stories", "2026-09-11"),
    ):
        run(run_id, stage=stage, day=day, ticker="NVDA" if stage == "stories" else None)

    assert set(EVIDENCE_STAGES) == {
        "fetch_yahoo",
        "ingest_rss",
        "classify_rss",
        "reclassify_rss",
    }
    assert selection_for(repository, invocation_id="inv").touched == {
        ("NVDA", "2026-09-13"),
        ("NVDA", "2026-09-12"),
    }
    assert selection_for(repository, invocation_id="inv-2").touched == frozenset()


# ----------------------------------------------------------------------
# Retry identity is structural, not textual
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "run_id, expected",
    [
        # 1. A real retry run.
        (f"phase0-abc:{RETRY_COMPONENT}:NVDA:{TODAY}", True),
        # 2. The literal token inside a caller-chosen invocation id.
        (f"x:{RETRY_COMPONENT}:NVDA:{TODAY}:{INTELLIGENCE_STAGE}:NVDA:{TODAY}", False),
        (f":{RETRY_COMPONENT}::{INTELLIGENCE_STAGE}:AMD:{TODAY}", False),
        # 3. Similar-but-not-exact component text.
        (f"inv:{INTELLIGENCE_STAGE}-retry-2:NVDA:{TODAY}", False),
        (f"inv:{INTELLIGENCE_STAGE}:NVDA:{TODAY}", False),
        (f"inv:{RETRY_COMPONENT}x:NVDA:{TODAY}", False),
        (f"inv:{INTELLIGENCE_STAGE}{RETRY_COMPONENT}:NVDA:{TODAY}", False),
        # Malformed trailing components are not retries either.
        (f"inv:{RETRY_COMPONENT}:NVDA:not-a-day", False),
        (f"inv:{RETRY_COMPONENT}::{TODAY}", False),
        (RETRY_COMPONENT, False),
        ("", False),
    ],
)
def test_is_retry_run_reads_only_the_pipelines_own_components(run_id, expected):
    assert is_retry_run(run_id) is expected


def test_an_invocation_id_containing_the_token_is_evidence_triggered(
    tmp_path, config, monkeypatch
):
    """An adversarial invocation id changes nothing about how a run reads."""

    repository = migrated(tmp_path)
    broken = breaking_themes(monkeypatch)
    hostile = f"evil:{RETRY_COMPONENT}:NVDA:{TODAY}"
    news = newsroom(NVDA=[article("NVDA", NOW - timedelta(days=2))])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    broken.add("NVDA")

    live(repository, config, invocation_id=hostile)

    day = (NOW - timedelta(days=2)).date().isoformat()
    failed = [
        row
        for row in stage_rows(repository, "themes")
        if row["ticker"] == "NVDA" and row["trading_day"] == day
    ]
    assert len(failed) == 1
    # The run id contains the token text and is still not a retry, so it
    # anchors a window: the partition is retryable.
    assert RETRY_COMPONENT in failed[0]["run_id"]
    assert not is_retry_run(failed[0]["run_id"])
    assert day in selection_for(repository, invocation_id="none").retried_days


def test_a_day_both_touched_and_retried_runs_as_evidence_triggered(
    tmp_path, config, monkeypatch
):
    repository = migrated(tmp_path)
    broken = breaking_themes(monkeypatch)
    earlier_day, news = _seed_failed_day(
        repository, config, monkeypatch, clock=ManualClock(), broken=broken
    )
    # New evidence for the same day: touched *and* retried.
    earlier = datetime.fromisoformat(earlier_day + "T11:00:00+00:00")
    news["NVDA"] = [article("NVDA", earlier, index=9)]
    selection_before = selection_for(repository, invocation_id="none")
    assert earlier_day in selection_before.retried_days

    live(repository, config, invocation_id="both")

    selection = selection_for(repository, invocation_id="both")
    assert earlier_day in selection.touched_days
    assert selection.runs_as_retry("NVDA", earlier_day) is False
    newest = [
        row
        for row in stage_rows(repository, "themes")
        if row["ticker"] == "NVDA" and row["trading_day"] == earlier_day
    ][-1]
    assert newest["run_id"].startswith("both:")
    assert not is_retry_run(newest["run_id"])


# ----------------------------------------------------------------------
# One clock
# ----------------------------------------------------------------------


def test_the_repository_clock_stamps_runs_and_drives_the_cutoff(tmp_path):
    clock = ManualClock()
    repository = clocked(tmp_path, clock)

    with repository.stage_run(
        run_id=f"a:{INTELLIGENCE_STAGE}:NVDA:{TODAY}",
        stage="themes",
        trading_day=TODAY,
        pipeline_version=PIPELINE_VERSION,
        ticker="NVDA",
    ):
        pass
    row = stage_rows(repository, "themes")[0]
    assert datetime.fromisoformat(row["started_at"]) == NOW
    assert datetime.fromisoformat(row["completed_at"]) == NOW

    clock.advance(timedelta(hours=1))
    assert repository.now() == NOW + timedelta(hours=1)


def test_a_naive_clock_is_refused(tmp_path):
    from phase0.errors import Phase0ValidationError

    repository = Phase0Repository(
        tmp_path / "naive.sqlite3", clock=lambda: datetime(2026, 9, 15, 20, 0)
    )
    repository.migrate()

    with pytest.raises(Phase0ValidationError, match="timezone-aware"):
        repository.now()
    with pytest.raises(Phase0ValidationError, match="timezone-aware"):
        with repository.stage_run(
            run_id="x:themes:NVDA:" + TODAY,
            stage="themes",
            trading_day=TODAY,
            pipeline_version=PIPELINE_VERSION,
            ticker="NVDA",
        ):
            pass


def test_run_live_has_no_independent_clock():
    """The old ``now=`` override is gone: there is one clock, the repository's."""

    for function in (run_live, pipeline.run_replay, intelligence_stage):
        assert "now" not in inspect.signature(function).parameters, function


# ----------------------------------------------------------------------
# Crash between ingestion and intelligence
# ----------------------------------------------------------------------


def _ingest_only(repository, config, monkeypatch, *, news, invocation_id):
    """Run ingestion and die before intelligence, the way a crash would.

    The downstream registry is emptied for this one invocation, so the
    evidence-writing runs commit and nothing derived is ever opened.
    """

    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline, "DOWNSTREAM_STAGES", ())
        result = run_live(repository, **config, invocation_id=invocation_id)
    assert [item.name for item in result.components] == ["yahoo", "rss"]
    return result


def test_evidence_left_behind_by_a_crash_is_recovered(tmp_path, config, monkeypatch):
    """Regressions 1-6: recovered once, produced, then not again."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier, index=i) for i in range(2)])

    # 1-2. Evidence persisted; the process dies before intelligence.
    _ingest_only(repository, config, monkeypatch, news=news, invocation_id="crashed")
    assert repository.read.evidence_partition_tickers(earlier_day) == ["NVDA"]
    assert stage_rows(repository, "stories") == []
    assert stage_rows(repository, "themes") == []

    # 3-4. The next invocation brings no new evidence for that day, and
    #      still selects it -- as recovered, not touched, not retried.
    news.clear()
    selection = selection_for(repository, invocation_id="next")
    assert selection.touched_days == ()
    assert selection.retried_days == ()
    # The RSS fixture's article (2026-08-18) was ingested by the same
    # crashed invocation and is recovered alongside.
    assert earlier_day in selection.recovered_days
    assert selection.runs_as_retry("NVDA", earlier_day) is False

    # 5. Stories and themes are produced.
    result = live(repository, config, invocation_id="next")
    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.counts["days_recovered"] >= 1
    assert repository.stories_for_day(earlier_day, "NVDA")
    assert theme_set(repository, "NVDA", earlier_day) is not None
    newest = [
        row for row in stage_rows(repository, "themes") if row["ticker"] == "NVDA"
    ][-1]
    assert newest["run_id"] == f"next:{INTELLIGENCE_STAGE}:NVDA:{earlier_day}"
    assert not is_retry_run(newest["run_id"])

    # 6. Once processed, the day is not "never processed" again.
    after = selection_for(repository, invocation_id="after")
    assert after.recovered_days == ()
    again = live(repository, config, invocation_id="after")
    assert component(again, INTELLIGENCE_STAGE).counts["days_recovered"] == 0


def test_one_processed_ticker_does_not_mask_an_unprocessed_one(
    tmp_path, config, monkeypatch
):
    """Regression 7: the check is per partition even though the day reruns."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    # NVDA on that day is fully processed.
    live(repository, config, invocation_id="first")
    assert theme_set(repository, "NVDA", earlier_day) is not None

    # AMD's evidence for the same day lands, and the process dies before
    # intelligence.
    news.clear()
    news["AMD"] = [article("AMD", earlier)]
    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline, "DOWNSTREAM_STAGES", ())
        run_live(repository, **config, invocation_id="crashed")
    assert repository.read.evidence_partition_tickers(earlier_day) == ["AMD", "NVDA"]
    assert repository.read.attempted_partitions(
        "themes", earlier_day, pipeline_version=PIPELINE_VERSION
    ) == ["NVDA"]

    news.clear()
    selection = selection_for(repository, invocation_id="next")
    assert selection.recovered_days == (earlier_day,)

    live(repository, config, invocation_id="next")
    assert theme_set(repository, "AMD", earlier_day) is not None
    assert selection_for(repository, invocation_id="later").recovered_days == ()


def test_recovery_is_bounded_by_the_horizon(tmp_path, config, monkeypatch):
    """Regression 8: old unprocessed evidence is not swept indefinitely."""

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    _ingest_only(repository, config, monkeypatch, news=news, invocation_id="crashed")
    news.clear()

    assert (
        earlier_day
        in selection_for(
            repository, invocation_id="none", now=clock.current
        ).recovered_days
    )
    clock.advance(RETRY_HORIZON + timedelta(seconds=1))
    assert (
        selection_for(
            repository, invocation_id="none", now=clock.current
        ).recovered_days
        == ()
    )


def test_recovery_is_scoped_to_the_pipeline_version(tmp_path, config, monkeypatch):
    """Regression 9: v2 having processed a day does not make v1 look done."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    _ingest_only(repository, config, monkeypatch, news=news, invocation_id="crashed")
    # v2 processes the partition end to end.
    with repository.stage_run(
        run_id=f"v2:{INTELLIGENCE_STAGE}:NVDA:{earlier_day}",
        stage="themes",
        trading_day=earlier_day,
        pipeline_version="v2",
        ticker="NVDA",
    ):
        pass

    v1 = select_intelligence_days(
        repository, invocation_id="none", pipeline_version=PIPELINE_VERSION, now=NOW
    )
    v2 = select_intelligence_days(
        repository, invocation_id="none", pipeline_version="v2", now=NOW
    )
    assert earlier_day in v1.recovered_days
    assert earlier_day not in v2.recovered_days


def test_no_evidence_means_no_recovery(tmp_path, config, monkeypatch):
    """Regression 10: an evidence-writing run that landed nothing recovers nothing."""

    repository = migrated(tmp_path)
    # Every ticker fails; Yahoo still settles a run per ticker for the
    # fetch day, but no partition holds evidence.
    wire(
        monkeypatch,
        ticker_factory=provider(failing=set(TICKERS)),
        get=responder(failing={"alpha", "beta"}),
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline, "DOWNSTREAM_STAGES", ())
        run_live(repository, **config, invocation_id="crashed")
    assert stage_rows(repository, "fetch_yahoo")

    selection = selection_for(repository, invocation_id="none")
    assert selection.recovered_days == ()
    assert selection.days == ()


# ----------------------------------------------------------------------
# Recovery must not renew a story-failure episode
# ----------------------------------------------------------------------


def _seed_story_failed_day(repository, config, monkeypatch, *, clock):
    """A partition on a non-fetch day whose *stories* fail, then no news.

    The shape the retry window and crash recovery could both claim: a
    failed ``stories`` row, and -- because the coordinator never opens
    themes over a story failure -- no ``themes`` row at all.
    """

    earlier = clock.current - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    seeded = live(
        repository,
        config,
        encoder=RaisingEncoder(RuntimeError("down")),
        invocation_id="seed",
    )
    assert component(seeded, INTELLIGENCE_STAGE).counts["stories_failed"] >= 1
    assert [
        row["status"]
        for row in stage_rows(repository, "stories")
        if row["ticker"] == "NVDA" and row["trading_day"] == earlier_day
    ] == ["failed"]
    assert not [
        row for row in stage_rows(repository, "themes") if row["ticker"] == "NVDA"
    ]
    news.clear()
    return earlier_day, news


def story_rows(repository, ticker: str, day: str) -> list[dict]:
    return [
        row
        for row in stage_rows(repository, "stories")
        if row["ticker"] == ticker and row["trading_day"] == day
    ]


def story_anchor(repository, ticker: str, day: str) -> str | None:
    """The retry anchor of ``ticker``/``day``'s stories episode, from the ledger."""

    episodes = repository.read.stage_outcome_episodes(
        ("stories",),
        pipeline_version=PIPELINE_VERSION,
        completed_since="2000-01-01T00:00:00+00:00",
    )
    for episode in episodes:
        if episode.ticker == ticker and episode.trading_day == day:
            return episode_anchor(episode)
    return None


def test_a_permanent_story_failure_is_not_renewed_by_recovery(
    tmp_path, config, monkeypatch
):
    """Codex's reproduction, on the ledger, with the repository's clock.

    A failed ``stories`` row and no ``themes`` row is both an open
    episode and, read naively, a partition intelligence never reached.
    Recovery must leave it to the episode: it is retried inside the
    window under the retry identity, the anchor never moves, and past
    the original deadline nothing selects it any more.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier_day, _ = _seed_story_failed_day(
        repository, config, monkeypatch, clock=clock
    )
    opened_at = clock.current
    original = story_rows(repository, "NVDA", earlier_day)[-1]
    assert story_anchor(repository, "NVDA", earlier_day) == original["completed_at"]

    # Inside the window the retry window owns it; recovery does not name it.
    clock.advance(timedelta(hours=60))
    selection = selection_for(repository, invocation_id="none", now=clock.current)
    assert earlier_day in selection.retried_days
    assert earlier_day not in selection.recovered_days
    assert selection.runs_as_retry("NVDA", earlier_day) is True

    # The retry runs -- still broken -- and records under the retry identity.
    retried = live(
        repository,
        config,
        encoder=RaisingEncoder(RuntimeError("still down")),
        invocation_id="retry-1",
    )
    intelligence = component(retried, INTELLIGENCE_STAGE)
    # The RSS fixture's AAPL partition failed the same way and is retried
    # alongside; what matters is that nothing was *recovered*.
    assert intelligence.counts["days_retried"] >= 1
    assert intelligence.counts["days_recovered"] == 0
    newest = story_rows(repository, "NVDA", earlier_day)[-1]
    assert newest["status"] == "failed"
    assert is_retry_run(newest["run_id"])
    # The anchor is still the original failure.
    assert story_anchor(repository, "NVDA", earlier_day) == original["completed_at"]

    # Just past the ORIGINAL deadline: not retried, not recovered, left alone.
    clock.advance(timedelta(hours=13))
    assert clock.current > opened_at + RETRY_HORIZON
    assert (
        clock.current < datetime.fromisoformat(newest["completed_at"]) + RETRY_HORIZON
    )
    expired = selection_for(repository, invocation_id="none", now=clock.current)
    assert earlier_day not in expired.retried_days
    assert earlier_day not in expired.recovered_days
    left = live(repository, config, invocation_id="later")
    # (The RSS fixture's day is re-ingested -- touched -- by every
    # invocation, so it is not a witness here; NVDA's day is.)
    assert component(left, INTELLIGENCE_STAGE).counts["days_recovered"] == 0
    assert story_rows(repository, "NVDA", earlier_day) == [original, newest]
    assert theme_set(repository, "NVDA", earlier_day) is None


@pytest.mark.parametrize(
    ("status", "markers", "expected"),
    [
        (None, None, True),  # never attempted
        ("success", (), True),  # settled, themes never opened
        ("degraded", (), True),  # identical replay, themes never opened
        ("failed", (), False),  # the episode's
        ("degraded", ((STAGE_DEGRADED, M3_UNAVAILABLE),), False),  # the episode's
    ],
)
def test_needs_recovery_reads_the_newest_story_outcome(status, markers, expected):
    latest = (
        None
        if status is None
        else StageOutcome(
            run_id="x:intelligence:NVDA:2026-09-13",
            ticker="NVDA",
            trading_day="2026-09-13",
            stage="stories",
            pipeline_version=PIPELINE_VERSION,
            status=status,
            markers=markers,
            completed_at="2026-09-13T00:00:00+00:00",
        )
    )
    assert needs_recovery(latest) is expected


def test_a_never_attempted_partition_is_still_recovered(tmp_path, config, monkeypatch):
    """Regression 2: no stories row, no themes row -> recovered and produced."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    _ingest_only(repository, config, monkeypatch, news=news, invocation_id="crashed")
    news.clear()
    assert (
        repository.read.latest_partition_outcomes(
            "stories", earlier_day, pipeline_version=PIPELINE_VERSION
        )
        == {}
    )

    selection = selection_for(repository, invocation_id="next")
    assert earlier_day in selection.recovered_days
    assert earlier_day not in selection.retried_days
    assert selection.runs_as_retry("NVDA", earlier_day) is False
    live(repository, config, invocation_id="next")
    assert theme_set(repository, "NVDA", earlier_day) is not None
    assert selection_for(repository, invocation_id="after").recovered_days == ()


def _dying_before_themes(monkeypatch, ticker: str):
    """The process dies after the story run commits and before themes open."""

    import phase0.themes as themes_module

    real = themes_module.ThemeReconciler.run_partition

    def run_partition(self, symbol, trading_day, **kwargs):
        if symbol == ticker:
            raise RuntimeError(f"process died before themes for {ticker}")
        return real(self, symbol, trading_day, **kwargs)

    monkeypatch.setattr(themes_module.ThemeReconciler, "run_partition", run_partition)


def test_settled_stories_with_no_theme_attempt_are_recovered(
    tmp_path, config, monkeypatch
):
    """Regression 3: a crash between stories and themes is recovered."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    with monkeypatch.context() as scoped:
        _dying_before_themes(scoped, "NVDA")
        crashed = live(repository, config, invocation_id="crashed")
    assert component(crashed, INTELLIGENCE_STAGE).status == "failed"
    assert [row["status"] for row in story_rows(repository, "NVDA", earlier_day)] == [
        "success"
    ]
    assert repository.stories_for_day(earlier_day, "NVDA")
    assert "NVDA" not in repository.read.attempted_partitions(
        "themes", earlier_day, pipeline_version=PIPELINE_VERSION
    )

    news.clear()
    selection = selection_for(repository, invocation_id="next")
    assert earlier_day in selection.recovered_days
    assert earlier_day not in selection.retried_days
    result = live(repository, config, invocation_id="next")
    assert component(result, INTELLIGENCE_STAGE).counts["days_recovered"] >= 1
    assert theme_set(repository, "NVDA", earlier_day) is not None
    newest = [
        row for row in stage_rows(repository, "themes") if row["ticker"] == "NVDA"
    ][-1]
    assert not is_retry_run(newest["run_id"])
    assert selection_for(repository, invocation_id="after").recovered_days == ()


@pytest.mark.parametrize(
    "shape", ["success", "themes_failed", "capture_failed", "m2_only"]
)
def test_a_themes_attempt_is_never_missing_intelligence(
    tmp_path, config, monkeypatch, shape
):
    """Regression 4: any themes row, whatever its status, is not "never processed"."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier, index=i) for i in range(3)])
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    encoder = None
    if shape == "themes_failed":
        breaking_themes(monkeypatch).add("NVDA")
    elif shape == "capture_failed":
        _breaking_capture(monkeypatch, "NVDA")
    elif shape == "m2_only":
        encoder = RaisingEncoder(EmbeddingModelLoadError("no model cache"))
    live(repository, config, encoder=encoder, invocation_id="seed")
    themes = [
        row
        for row in stage_rows(repository, "themes")
        if row["ticker"] == "NVDA" and row["trading_day"] == earlier_day
    ]
    assert [row["status"] for row in themes] == [
        {
            "success": "success",
            "themes_failed": "failed",
            "capture_failed": "failed",
            "m2_only": "degraded",
        }[shape]
    ]

    news.clear()
    selection = selection_for(repository, invocation_id="next")
    assert earlier_day not in selection.recovered_days
    # The unresolved shapes are the retry window's; the healthy one is done.
    assert (earlier_day in selection.retried_days) is (shape != "success")


class FailingFor(FakeEncoder):
    """M3 fails for the named tickers only -- the article titles name them."""

    def __init__(self, *tickers: str) -> None:
        super().__init__()
        self.tickers = set(tickers)

    def embed_batch(self, texts):
        texts = list(texts)
        for text in texts:
            for ticker in self.tickers:
                if text.startswith(f"{ticker} "):
                    raise RuntimeError(f"down for {ticker}")
        return super().embed_batch(texts)


def _seed_story_failure(repository, config, monkeypatch, *, clock, ticker):
    """``_seed_story_failed_day`` for any ticker, returning its newsroom."""

    earlier = clock.current - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(**{ticker: [article(ticker, earlier)]})
    wire(monkeypatch, ticker_factory=provider(news_by_ticker=news), get=responder())
    live(
        repository,
        config,
        encoder=RaisingEncoder(RuntimeError("down")),
        invocation_id="seed",
    )
    assert [r["status"] for r in story_rows(repository, ticker, earlier_day)] == [
        "failed"
    ]
    news.clear()
    return earlier_day, news


def _crash_ingest(repository, config, monkeypatch, news, *, ticker, day):
    """Persist one article for ``ticker``/``day`` and die before intelligence."""

    earlier = datetime.fromisoformat(day + "T12:00:00+00:00")
    news[ticker] = [article(ticker, earlier)]
    with monkeypatch.context() as scoped:
        scoped.setattr(pipeline, "DOWNSTREAM_STAGES", ())
        run_live(repository, **config, invocation_id="crashed")
    news.clear()


@pytest.mark.parametrize(
    ("episode", "fresh"),
    [("NVDA", "AMD"), ("AMD", "NVDA")],
    ids=["episode-runs-second", "episode-runs-first"],
)
def test_a_first_attempt_beside_a_retry_gets_its_own_anchor(
    tmp_path, config, monkeypatch, episode, fresh
):
    """Mixed day: one partition retried, one never attempted, no new evidence.

    Each runs under its own identity.  The retry cannot move its anchor;
    the first attempt, failing, *is* an anchor -- so it is retried on its
    own account afterwards, whatever its neighbour does, and expires on
    its own original deadline.  Both ticker orders, so nothing here
    depends on which partition the coordinator reaches first.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier_day, news = _seed_story_failure(
        repository, config, monkeypatch, clock=clock, ticker=episode
    )
    episode_anchor_at = story_rows(repository, episode, earlier_day)[-1]["completed_at"]
    clock.advance(timedelta(hours=1))
    _crash_ingest(repository, config, monkeypatch, news, ticker=fresh, day=earlier_day)
    assert sorted(repository.read.evidence_partition_tickers(earlier_day)) == sorted(
        [episode, fresh]
    )

    clock.advance(timedelta(hours=1))
    selection = selection_for(repository, invocation_id="none", now=clock.current)
    assert (episode, earlier_day) in selection.retried
    assert (fresh, earlier_day) in selection.recovered
    assert selection.touched == frozenset()
    assert selection.runs_as_retry(episode, earlier_day) is True
    assert selection.runs_as_retry(fresh, earlier_day) is False

    # The episode's retry succeeds; the fresh partition's first attempt fails.
    first = live(repository, config, encoder=FailingFor(fresh), invocation_id="mixed")
    intelligence = component(first, INTELLIGENCE_STAGE)
    assert intelligence.counts["partitions_retried"] >= 1
    assert intelligence.counts["partitions_recovered"] == 1
    resolved = story_rows(repository, episode, earlier_day)[-1]
    assert resolved["status"] == "success"
    assert is_retry_run(resolved["run_id"])
    assert theme_set(repository, episode, earlier_day) is not None
    assert story_anchor(repository, episode, earlier_day) is None  # closed
    failed = story_rows(repository, fresh, earlier_day)[-1]
    assert failed["status"] == "failed"
    assert failed["run_id"] == f"mixed:{INTELLIGENCE_STAGE}:{fresh}:{earlier_day}"
    assert not is_retry_run(failed["run_id"])
    fresh_anchor_at = failed["completed_at"]
    assert story_anchor(repository, fresh, earlier_day) == fresh_anchor_at
    assert fresh_anchor_at > episode_anchor_at

    # One hour later the fresh partition is retried on its own account.
    clock.advance(timedelta(hours=1))
    selection = selection_for(repository, invocation_id="none", now=clock.current)
    assert (fresh, earlier_day) in selection.retried
    assert (episode, earlier_day) not in selection.retried
    assert (fresh, earlier_day) not in selection.recovered
    assert selection.runs_as_retry(fresh, earlier_day) is True
    assert selection.runs_as_retry(episode, earlier_day) is False

    # Its retry fails and does not move its anchor.
    live(repository, config, encoder=FailingFor(fresh), invocation_id="retry-1")
    newest = story_rows(repository, fresh, earlier_day)[-1]
    assert newest["status"] == "failed"
    assert is_retry_run(newest["run_id"])
    assert story_anchor(repository, fresh, earlier_day) == fresh_anchor_at
    # The neighbour, rerun as a settled partition, wrote an ordinary row.
    neighbour = story_rows(repository, episode, earlier_day)[-1]
    assert neighbour["run_id"].startswith("retry-1:")
    assert not is_retry_run(neighbour["run_id"])

    # It expires on its own original deadline.
    clock.current = datetime.fromisoformat(fresh_anchor_at) + RETRY_HORIZON
    assert (fresh, earlier_day) in selection_for(
        repository, invocation_id="none", now=clock.current
    ).retried
    clock.advance(timedelta(seconds=1))
    expired = selection_for(repository, invocation_id="none", now=clock.current)
    assert (fresh, earlier_day) not in expired.retried
    assert (fresh, earlier_day) not in expired.recovered
    assert earlier_day not in expired.days


def test_touched_retried_and_recovered_partitions_share_a_day(
    tmp_path, config, monkeypatch
):
    """AAPL new evidence, NVDA retry, AMD never attempted -- one day.

    All three fail.  AAPL's failure is a fresh evidence-triggered anchor,
    NVDA's retry leaves its anchor where it was, AMD's first failure is
    its own initial anchor.  No identity is inherited from a neighbour.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier_day, news = _seed_story_failure(
        repository, config, monkeypatch, clock=clock, ticker="NVDA"
    )
    nvda_anchor_at = story_rows(repository, "NVDA", earlier_day)[-1]["completed_at"]
    clock.advance(timedelta(hours=1))
    _crash_ingest(repository, config, monkeypatch, news, ticker="AMD", day=earlier_day)

    clock.advance(timedelta(hours=1))
    before = selection_for(repository, invocation_id="mixed", now=clock.current)
    assert before.touched == frozenset()
    assert ("NVDA", earlier_day) in before.retried
    assert before.recovered == {("AMD", earlier_day)}, before

    # AAPL's first article for that day arrives with this invocation.
    earlier = datetime.fromisoformat(earlier_day + "T12:00:00+00:00")
    news["AAPL"] = [article("AAPL", earlier)]
    result = live(
        repository,
        config,
        encoder=RaisingEncoder(RuntimeError("everything down")),
        invocation_id="mixed",
    )
    intelligence = component(result, INTELLIGENCE_STAGE)
    assert intelligence.counts["partitions_touched"] == 1
    assert intelligence.counts["partitions_recovered"] == 1
    assert intelligence.counts["partitions_retried"] >= 1
    after = selection_for(repository, invocation_id="mixed", now=clock.current)
    assert ("AAPL", earlier_day) in after.touched
    assert after.runs_as_retry("AAPL", earlier_day) is False
    assert after.runs_as_retry("NVDA", earlier_day) is True
    assert after.runs_as_retry("AMD", earlier_day) is True  # now it has an episode

    aapl = story_rows(repository, "AAPL", earlier_day)[-1]
    nvda = story_rows(repository, "NVDA", earlier_day)[-1]
    amd = story_rows(repository, "AMD", earlier_day)[-1]
    assert aapl["run_id"] == f"mixed:{INTELLIGENCE_STAGE}:AAPL:{earlier_day}"
    assert nvda["run_id"] == f"mixed:{RETRY_COMPONENT}:NVDA:{earlier_day}"
    assert amd["run_id"] == f"mixed:{INTELLIGENCE_STAGE}:AMD:{earlier_day}"
    assert story_anchor(repository, "AAPL", earlier_day) == aapl["completed_at"]
    assert story_anchor(repository, "NVDA", earlier_day) == nvda_anchor_at
    assert nvda["completed_at"] > nvda_anchor_at
    assert story_anchor(repository, "AMD", earlier_day) == amd["completed_at"]


def test_new_evidence_over_a_story_failure_opens_a_fresh_episode(
    tmp_path, config, monkeypatch
):
    """Regression 6: touched beats retried; a fresh failure re-anchors."""

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier_day, news = _seed_story_failed_day(
        repository, config, monkeypatch, clock=clock
    )
    original = story_rows(repository, "NVDA", earlier_day)[-1]
    clock.advance(timedelta(hours=12))

    earlier = datetime.fromisoformat(earlier_day + "T11:00:00+00:00")
    news["NVDA"] = [article("NVDA", earlier, index=9)]
    fresh = live(
        repository,
        config,
        encoder=RaisingEncoder(RuntimeError("still down")),
        invocation_id="fresh",
    )
    selection = selection_for(repository, invocation_id="fresh", now=clock.current)
    assert earlier_day in selection.touched_days
    assert earlier_day in selection.retried_days
    assert selection.runs_as_retry("NVDA", earlier_day) is False
    assert component(fresh, INTELLIGENCE_STAGE).counts["days_touched"] >= 1
    newest = story_rows(repository, "NVDA", earlier_day)[-1]
    assert newest["status"] == "failed"
    assert newest["run_id"].startswith("fresh:")
    assert not is_retry_run(newest["run_id"])
    # The window is now measured from the evidence-triggered failure.
    assert story_anchor(repository, "NVDA", earlier_day) == newest["completed_at"]
    assert newest["completed_at"] > original["completed_at"]


def test_another_versions_intelligence_does_not_suppress_this_ones(
    tmp_path, config, monkeypatch
):
    """Regression 7: v2 stories/themes rows decide nothing for v1."""

    repository = migrated(tmp_path)
    earlier = NOW - timedelta(days=2)
    earlier_day = earlier.date().isoformat()
    news = newsroom(NVDA=[article("NVDA", earlier)])
    _ingest_only(repository, config, monkeypatch, news=news, invocation_id="crashed")

    def settle(stage, *, version, boom=False):
        with pytest.raises(RuntimeError) if boom else nullcontext():
            with repository.stage_run(
                run_id=f"{version}:{INTELLIGENCE_STAGE}:NVDA:{earlier_day}",
                stage=stage,
                trading_day=earlier_day,
                pipeline_version=version,
                ticker="NVDA",
            ):
                if boom:
                    raise RuntimeError("boom")

    def selection(version):
        return select_intelligence_days(
            repository, invocation_id="none", pipeline_version=version, now=NOW
        )

    # v2 carried the partition all the way through: v1 is still recovered.
    settle("stories", version="v2")
    settle("themes", version="v2")
    assert earlier_day in selection(PIPELINE_VERSION).recovered_days
    assert earlier_day not in selection("v2").recovered_days

    # A v3 story failure is v3's episode, not v1's, and not v3's recovery.
    settle("stories", version="v3", boom=True)
    assert earlier_day in selection(PIPELINE_VERSION).recovered_days
    assert earlier_day not in selection(PIPELINE_VERSION).retried_days
    assert earlier_day in selection("v3").retried_days
    assert earlier_day not in selection("v3").recovered_days


# ----------------------------------------------------------------------
# A repeat sighting is not new evidence
# ----------------------------------------------------------------------


def _rss_item(title: str, link: str, day: str) -> bytes:
    return (
        f"<item><title>{title}</title><link>{link}</link>"
        f"<description>{title}</description>"
        f"<pubDate>{day}T10:00:00Z</pubDate></item>"
    ).encode()


def _feed(*items: bytes) -> bytes:
    return b"<rss><channel>" + b"".join(items) + b"</channel></rss>"


def mutable_responder(holder: dict[str, bytes]):
    """An RSS socket whose body is whatever ``holder["body"]`` is *now*."""

    class Response:
        status_code = 200
        headers: dict[str, str] = {}

        def __init__(self, url):
            self.url = url
            self.content = holder["body"]

        def raise_for_status(self):
            return None

    return lambda url, **kwargs: Response(url)


@pytest.fixture
def nvda_feed(tmp_path):
    """One feed, and an alias file that maps ``GeForce`` to NVDA."""

    feeds = tmp_path / "feeds.yaml"
    aliases = tmp_path / "aliases.yaml"
    write_feeds(feeds, ["alpha"])
    aliases.write_text(
        "tickers:\n  - ticker: NVDA\n    strong_aliases: [GeForce]\n",
        encoding="utf-8",
    )
    return {"feeds_path": feeds, "aliases_path": aliases}


def _classify_runs(repository, invocation_id: str, ticker: str, day: str):
    return [
        row
        for row in stage_rows(repository, "classify_rss")
        if row["run_id"].startswith(f"{invocation_id}:")
        and row["ticker"] == ticker
        and row["trading_day"] == day
    ]


def _projection(repository, ticker: str, day: str):
    evidence = repository.read.partition_evidence(ticker, day)
    return (
        tuple(item.item_id for item in evidence.items),
        tuple(sorted(e.raw_item_id for e in evidence.excluded)),
    )


def test_a_repeat_rss_sighting_does_not_touch_the_partition(
    tmp_path, nvda_feed, monkeypatch
):
    """Codex's second reproduction, through the real RSS path.

    The same article polled again is observed and re-classified, and both
    runs record that nothing changed -- so the partition is not touched,
    its failure runs only as a retry, and its window closes on the
    original anchor however many times the feed repeats itself.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier_day = (clock.current - timedelta(days=2)).date().isoformat()
    holder = {
        "body": _feed(
            _rss_item("GeForce supply update", "https://example.com/gf-1", earlier_day)
        )
    }
    wire(
        monkeypatch,
        ticker_factory=provider(news_by_ticker={}),
        get=mutable_responder(holder),
    )

    # 1-2. The article is persisted and associated to NVDA; its stories fail.
    seeded = live(
        repository,
        nvda_feed,
        encoder=RaisingEncoder(RuntimeError("down")),
        invocation_id="seed",
    )
    assert component(seeded, "rss").counts["inserted"] == 1
    [classified] = _classify_runs(repository, "seed", "NVDA", earlier_day)
    assert json.loads(classified["counts"])["relevance_changed"] == 1
    assert ("NVDA", earlier_day) in selection_for(
        repository, invocation_id="seed", now=clock.current
    ).touched
    original = story_rows(repository, "NVDA", earlier_day)[-1]
    assert original["status"] == "failed"
    anchor_at = original["completed_at"]
    projection = _projection(repository, "NVDA", earlier_day)
    raw_count = repository.count("raw_items")

    # 3-4. Sixty hours on, the feed still carries the same article.
    clock.advance(timedelta(hours=60))
    again = live(
        repository,
        nvda_feed,
        encoder=RaisingEncoder(RuntimeError("still down")),
        invocation_id="poll-2",
    )
    # 5. Nothing about the evidence changed ...
    assert component(again, "rss").counts["inserted"] == 0
    assert component(again, "rss").counts["duplicates"] == 1
    assert repository.count("raw_items") == raw_count
    assert _projection(repository, "NVDA", earlier_day) == projection
    [reclassified] = _classify_runs(repository, "poll-2", "NVDA", earlier_day)
    assert json.loads(reclassified["counts"])["relevance_assigned"] == 1
    assert json.loads(reclassified["counts"])["relevance_changed"] == 0
    # 6-7. ... so the partition was not touched, and ran only as a retry.
    selection = selection_for(repository, invocation_id="poll-2", now=clock.current)
    assert ("NVDA", earlier_day) not in selection.touched
    assert ("NVDA", earlier_day) in selection.retried
    retried = story_rows(repository, "NVDA", earlier_day)[-1]
    assert retried["run_id"] == f"poll-2:{RETRY_COMPONENT}:NVDA:{earlier_day}"
    assert retried["status"] == "failed"
    # 8. The anchor is the original failure.
    assert story_anchor(repository, "NVDA", earlier_day) == anchor_at

    # 9-12. Past the original deadline the same article is polled again:
    #       the day does not reopen and the failure stays expired.
    clock.advance(timedelta(hours=13))
    assert clock.current > datetime.fromisoformat(anchor_at) + RETRY_HORIZON
    third = live(
        repository,
        nvda_feed,
        encoder=RaisingEncoder(RuntimeError("still down")),
        invocation_id="poll-3",
    )
    assert component(third, "rss").counts["duplicates"] == 1
    selection = selection_for(repository, invocation_id="poll-3", now=clock.current)
    assert ("NVDA", earlier_day) not in selection.touched
    assert ("NVDA", earlier_day) not in selection.retried
    assert ("NVDA", earlier_day) not in selection.recovered
    assert earlier_day not in selection.days
    assert component(third, INTELLIGENCE_STAGE).counts["days_selected"] == 0
    assert story_rows(repository, "NVDA", earlier_day) == [original, retried]
    assert story_anchor(repository, "NVDA", earlier_day) == anchor_at

    # A genuinely new article for the same historical day is a touch: the
    # partition runs evidence-triggered, and its failure is a fresh anchor.
    clock.advance(timedelta(hours=1))
    holder["body"] = _feed(
        _rss_item("GeForce supply update", "https://example.com/gf-1", earlier_day),
        _rss_item("GeForce pricing memo", "https://example.com/gf-2", earlier_day),
    )
    fresh = live(
        repository,
        nvda_feed,
        encoder=RaisingEncoder(RuntimeError("still down")),
        invocation_id="fresh",
    )
    assert component(fresh, "rss").counts["inserted"] == 1
    [changed] = _classify_runs(repository, "fresh", "NVDA", earlier_day)
    assert json.loads(changed["counts"])["relevance_changed"] == 1
    assert _projection(repository, "NVDA", earlier_day) != projection
    selection = selection_for(repository, invocation_id="fresh", now=clock.current)
    assert ("NVDA", earlier_day) in selection.touched
    assert selection.runs_as_retry("NVDA", earlier_day) is False
    reopened = story_rows(repository, "NVDA", earlier_day)[-1]
    assert reopened["run_id"] == f"fresh:{INTELLIGENCE_STAGE}:NVDA:{earlier_day}"
    assert reopened["status"] == "failed"
    assert story_anchor(repository, "NVDA", earlier_day) == reopened["completed_at"]
    assert reopened["completed_at"] > anchor_at


def test_a_changed_association_on_a_stored_article_is_a_touch(
    tmp_path, nvda_feed, monkeypatch
):
    """The article is old; what changes is whether NVDA owns it.

    Gaining the association is a durable change to the partition's input
    and so is losing it -- each is a touch, and each reruns the partition
    evidence-triggered.  Re-deciding the same association in between is
    not.
    """

    clock = ManualClock()
    repository = clocked(tmp_path, clock)
    earlier_day = (clock.current - timedelta(days=2)).date().isoformat()
    holder = {
        "body": _feed(
            _rss_item("GeForce pricing memo", "https://example.com/gpu", earlier_day)
        )
    }
    wire(
        monkeypatch,
        ticker_factory=provider(news_by_ticker={}),
        get=mutable_responder(holder),
    )

    def aliases(*names: str) -> None:
        nvda_feed["aliases_path"].write_text(
            "tickers:\n  - ticker: NVDA\n    strong_aliases: [%s]\n" % ", ".join(names),
            encoding="utf-8",
        )

    # Stored, but the alias file does not know the word: no partition.
    aliases("Blackwell")
    seeded = live(repository, nvda_feed, invocation_id="seed")
    assert component(seeded, "rss").counts["inserted"] == 1
    assert repository.read.evidence_partition_tickers(earlier_day) == []
    assert selection_for(repository, invocation_id="seed").touched == frozenset()

    # The alias file learns the word: the same article now belongs to NVDA.
    aliases("Blackwell", "GeForce")
    clock.advance(timedelta(hours=1))
    gained = live(repository, nvda_feed, invocation_id="gained")
    assert component(gained, "rss").counts["inserted"] == 0
    assert repository.read.evidence_partition_tickers(earlier_day) == ["NVDA"]
    [run] = _classify_runs(repository, "gained", "NVDA", earlier_day)
    assert json.loads(run["counts"])["relevance_changed"] == 1
    selection = selection_for(repository, invocation_id="gained", now=clock.current)
    assert ("NVDA", earlier_day) in selection.touched
    assert theme_set(repository, "NVDA", earlier_day) is not None

    # Polled again, unchanged: re-decided, not changed, not touched.
    clock.advance(timedelta(hours=1))
    same = live(repository, nvda_feed, invocation_id="same")
    [run] = _classify_runs(repository, "same", "NVDA", earlier_day)
    assert json.loads(run["counts"])["relevance_assigned"] == 1
    assert json.loads(run["counts"])["relevance_changed"] == 0
    assert selection_for(repository, invocation_id="same").touched == frozenset()
    assert component(same, INTELLIGENCE_STAGE).counts["days_selected"] == 0

    # The alias file forgets the word: the association is withdrawn.
    aliases("Blackwell")
    clock.advance(timedelta(hours=1))
    lost = live(repository, nvda_feed, invocation_id="lost")
    assert repository.read.evidence_partition_tickers(earlier_day) == []
    [run] = _classify_runs(repository, "lost", "NVDA", earlier_day)
    assert json.loads(run["counts"])["relevance_assigned"] == 0
    assert json.loads(run["counts"])["relevance_changed"] == 1
    assert ("NVDA", earlier_day) in selection_for(
        repository, invocation_id="lost", now=clock.current
    ).touched
    assert component(lost, INTELLIGENCE_STAGE).counts["days_selected"] == 1

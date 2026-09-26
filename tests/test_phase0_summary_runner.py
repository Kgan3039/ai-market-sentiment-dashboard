"""A3b: the scheduled summary runner over a real migrated database.

Every partition here is written by the ordinary story and theme
reconcilers and read back through ``Phase0Reader.theme_population``; the
lifecycle, the run log, and the migrations are real.  Only the provider is
fake, and every fake counts its calls so "zero provider calls" is a
measured claim rather than an inferred one.
"""

from __future__ import annotations

import itertools
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ai.guarded_summary as guarded
import phase0.summary_runner as summary_runner
import pipeline
from ai.guarded_summary import resolve_generation_policy
from ai.summarization import (
    GeminiClient,
    ProviderRequestError,
)
from nlp.dedup.selection import cluster_fingerprint_for
from phase0.errors import Phase0RunContextError, Phase0ValidationError
from phase0.models import (
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import Phase0Repository
from phase0.summary_lifecycle import (
    STAGE,
    RetryPolicy,
    current_summary_artifact,
    ensure_summary,
)
from phase0.summary_runner import (
    DEFAULT_PROVIDER_CALL_BUDGET,
    PRODUCTION_MAX_ATTEMPTS,
    PRODUCTION_RETRY_POLICY,
    SummaryConfigError,
    SummaryRunner,
    production_generation_policy,
    provider_call_budget,
    run_scheduled_summaries,
    summaries_enabled,
)

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-07-23"
VERSION = "v1"
HORIZON = timedelta(days=3)
ID_LINE_RE = re.compile(r"- id: (\S+)")
CANARY = "sk-CANARY-SECRET-0123456789"
SUMMARY_TABLES = (
    "summary_artifacts",
    "summary_sentences",
    "summary_sentence_citations",
    "summary_generations",
    "summary_generation_attempts",
)
_ITEMS = itertools.count(1)
_RUNS = itertools.count(1)
_BASES = itertools.count(1)


# ----------------------------------------------------------------------
# A persisted world
# ----------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now = self.now + delta


#: Production A3b settings a developer's shell may export.  Every test here
#: starts from none of them and sets exactly what it exercises.
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


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def repository(tmp_path, clock):
    repo = Phase0Repository(tmp_path / "phase0.db", clock=clock)
    repo.migrate()
    return repo


def insert_items(repository, ticker, day, outlets):
    rows = []
    for outlet in outlets:
        index = next(_ITEMS)
        rows.append(
            {
                "source": f"yahoo:{outlet}",
                "ticker": ticker,
                "title": f"{outlet} headline {index}",
                "description": f"{outlet} standfirst {index}.",
                "url": f"https://{outlet.lower()}.example/{index}",
                "canonical_url": f"https://{outlet.lower()}.example/{index}",
                "published_at": f"{day}T10:0{index % 10}:00+00:00",
                "fetched_at": f"{day}T11:00:00+00:00",
                "raw_json": {"index": index},
            }
        )
    return [r.item_id for r in repository.admin.insert_raw_items(rows)]


def story(ticker, day, item_ids, title, *, stage="m3.semantic"):
    outlets = ["Reuters"] * len(item_ids)
    fingerprint = cluster_fingerprint_for(ticker, [str(i) for i in item_ids])
    return StoryRecord(
        cluster_fingerprint=fingerprint,
        canonical_title=title,
        members=tuple(
            StoryMemberRecord(
                raw_item_id=item,
                position=position,
                outlet=outlet,
                url=f"https://{outlet.lower()}.example/{item}",
                canonical_url=f"https://{outlet.lower()}.example/{item}",
            )
            for position, (item, outlet) in enumerate(zip(item_ids, outlets))
        ),
        canonical_item_id=item_ids[0],
        outlet=outlets[0],
        outlet_count=1,
        published_at=f"{day}T10:05:00+00:00",
        canonical_url=f"https://reuters.example/{item_ids[0]}",
        content_hash=f"h-{fingerprint[:8]}",
        stage=stage,
        member_story_keys=(fingerprint,),
        algorithm_version="m3.1",
        config_fingerprint="cfg",
        model_name="fake",
        model_revision="r1",
        embedding_dimension=4,
    )


def seed(
    repository,
    ticker="TSLA",
    *,
    day=DAY,
    themes=("Alpha", "Beta", "Gamma"),
    stage="m3.semantic",
    story_count_delta=0,
    with_theme_set=True,
):
    """One partition: one story per theme, themes ranked in the order given.

    Returns ``{label: theme_id}``.  ``story_count_delta`` records a wrong
    input story count on the set, which the A2/A3 health gate refuses.
    """

    items = {
        label: insert_items(repository, ticker, day, ["Reuters"]) for label in themes
    }
    records = [
        story(ticker, day, items[label], f"{ticker} {label} story", stage=stage)
        for label in themes
    ]
    with repository.stage_run(
        run_id=f"seed-{next(_RUNS)}",
        stage="stories",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        repository.reconcile_stories(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            stories=records,
        )
    story_ids = {
        row["canonical_title"]: row["id"]
        for row in repository.stories_for_day(day, ticker)
    }
    if not with_theme_set:
        return {}
    with repository.stage_run(
        run_id=f"seed-{next(_RUNS)}",
        stage="themes",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        repository.reconcile_themes(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            theme_set=ThemeSetRecord(
                method="hdbscan",
                method_reason="clustered",
                source_metadata={"story_count": len(themes) + story_count_delta},
                config_fingerprint="cfg",
                algorithm_version="m5.1",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            ),
            themes=[
                ThemeRecord(
                    fingerprint=f"fp-{ticker}-{label}-{items[label][0]}",
                    theme_key=f"key-{ticker}-{label}",
                    label=label,
                    label_source="representative_title",
                    story_ids=(story_ids[f"{ticker} {label} story"],),
                    citation_item_ids=tuple(items[label]),
                    status="ready",
                    salience_rank=rank,
                    story_count=1,
                )
                for rank, label in enumerate(themes, start=1)
            ],
            other_coverage=[],
            excluded=[],
            terminal=True,
        )
    population = repository.read.theme_population(ticker, day, VERSION)
    return {theme.label: theme.theme_id for theme in population.themes}


def seed_m2_only(repository, ticker="NVDA", day=DAY):
    """An M2-only partition as the theme stage leaves it: stories, no set."""

    seed(
        repository,
        ticker,
        day=day,
        themes=("Exact",),
        stage="m2.exact",
        with_theme_set=False,
    )
    with repository.stage_run(
        run_id=f"seed-{next(_RUNS)}",
        stage="themes",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        run.record_degradation("m5_requires_semantic_stories")
        repository.clear_theme_set(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            terminal=True,
        )


class Client:
    """A fake provider.  Behaviour is chosen per cited story title.

    ``fail`` titles answer with a provider request error on every attempt
    (a typed ``unavailable``); ``crash`` titles raise ``TypeError``, which
    A2 propagates as a defect; ``reject_first`` answers the first attempt
    with an uncited sentence so A2 regenerates once.
    """

    def __init__(
        self,
        *,
        model="fake-model",
        max_output_tokens=1024,
        fail=(),
        crash=(),
        reject_first=False,
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.fail = set(fail)
        self.crash = set(crash)
        self.reject_first = reject_first
        self.calls: list[str] = []
        self._seen: set[str] = set()

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls.append(user_prompt)
        if any(title in user_prompt for title in self.crash):
            raise TypeError(f"client defect api_key={CANARY}")
        if any(title in user_prompt for title in self.fail):
            raise ProviderRequestError("503 upstream unavailable")
        ids = ID_LINE_RE.findall(user_prompt)
        key = ",".join(sorted(ids))
        if self.reject_first and key not in self._seen:
            self._seen.add(key)
            return response_schema.model_validate(
                {
                    "label": "Coverage",
                    "sentences": [
                        {"text": "One sentence.", "citation_ids": []},
                        {"text": "Another sentence.", "citation_ids": ids},
                    ],
                }
            )
        return response_schema.model_validate(
            {
                "label": "Coverage summary",
                "sentences": [
                    {"text": "Coverage leads with this story.", "citation_ids": ids},
                    {"text": "Outlets repeat the report.", "citation_ids": ids[:1]},
                ],
            }
        )


def runner(
    repository,
    client,
    *,
    budget=DEFAULT_PROVIDER_CALL_BUDGET,
    retry=PRODUCTION_RETRY_POLICY,
):
    return SummaryRunner(
        repository,
        pipeline_version=VERSION,
        client=client,
        budget=budget,
        horizon=HORIZON,
        retry=retry,
    )


def sweep(repository, client, *, base=None, **kwargs):
    base = base or f"inv-{next(_BASES)}:summaries"
    return runner(repository, client, **kwargs).run(base_run_id=base)


def counts(repository):
    return {table: repository.read.count(table) for table in SUMMARY_TABLES}


def summary_runs(repository):
    return [row for row in repository.read.run_log_rows() if row["stage"] == STAGE]


def current(repository, client, ticker, theme_id):
    return current_summary_artifact(
        repository.read,
        ticker,
        DAY,
        VERSION,
        theme_id,
        production_generation_policy(client),
    )


def intelligence_rows(repository):
    with repository.admin.connect_writable() as connection:
        return {
            table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table}")]
            for table in (
                "stories",
                "story_members",
                "theme_sets",
                "themes",
                "theme_stories",
                "theme_citations",
            )
        }


# ----------------------------------------------------------------------
# A. Feature flag
# ----------------------------------------------------------------------


def build_stage(repository):
    return pipeline.summaries_stage(
        repository, pipeline_version=VERSION, invocation_id="inv", encoder=None
    )


def refuse_client():
    raise AssertionError("a provider client was constructed")


def test_flag_absent_builds_nothing(repository, monkeypatch):
    monkeypatch.delenv(summary_runner.ENABLED_ENV, raising=False)
    monkeypatch.setattr(summary_runner, "production_summary_client", refuse_client)
    assert summaries_enabled() is False
    assert build_stage(repository) is None


@pytest.mark.parametrize("value", ["", "  ", "0", "false", "FALSE", " no ", "off"])
def test_false_variants_build_nothing(repository, monkeypatch, value):
    monkeypatch.setenv(summary_runner.ENABLED_ENV, value)
    monkeypatch.setattr(summary_runner, "production_summary_client", refuse_client)
    assert build_stage(repository) is None


@pytest.mark.parametrize("value", ["1", "true", "Yes", " ON "])
def test_enabled_builds_a_non_mandatory_component(repository, monkeypatch, value):
    monkeypatch.setenv(summary_runner.ENABLED_ENV, value)
    monkeypatch.setattr(summary_runner, "production_summary_client", refuse_client)
    stage = build_stage(repository)
    assert stage is not None
    assert stage.name == "summaries"
    assert stage.mandatory is False
    # Building constructs no client; only running would.


def test_an_unrecognized_flag_is_reported_not_guessed(repository, monkeypatch):
    monkeypatch.setenv(summary_runner.ENABLED_ENV, "maybe")
    monkeypatch.setattr(summary_runner, "production_summary_client", refuse_client)
    with pytest.raises(SummaryConfigError):
        summaries_enabled()
    stage = build_stage(repository)
    result = pipeline.execute_stage(stage, invocation_id="inv")
    assert result.errors[0]["type"] == "summaries_misconfigured"
    assert "maybe" not in json.dumps(result.errors)
    assert result.counts["provider_calls"] == 0
    assert counts(repository) == dict.fromkeys(SUMMARY_TABLES, 0)


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_provider_key_makes_no_call_and_no_write(repository, monkeypatch, key):
    seed(repository)
    if key is None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GEMINI_API_KEY", key)
    before_runs = repository.read.run_log_rows()
    before_intelligence = intelligence_rows(repository)

    result_counts, errors = run_scheduled_summaries(
        repository,
        pipeline_version=VERSION,
        base_run_id="inv:summaries",
        horizon=HORIZON,
        client_factory=refuse_client,
    )

    assert errors == [
        {"type": "summaries_unconfigured", "reason": "provider_api_key_missing"}
    ]
    assert result_counts["provider_calls"] == 0
    assert result_counts["partitions_considered"] == 0
    assert counts(repository) == dict.fromkeys(SUMMARY_TABLES, 0)
    assert repository.read.run_log_rows() == before_runs
    assert intelligence_rows(repository) == before_intelligence


def test_invalid_budget_is_misconfigured_without_work(repository, monkeypatch):
    seed(repository)
    monkeypatch.setenv(summary_runner.BUDGET_ENV, "-1")
    result_counts, errors = run_scheduled_summaries(
        repository,
        pipeline_version=VERSION,
        base_run_id="inv:summaries",
        horizon=HORIZON,
        client_factory=refuse_client,
    )
    assert errors[0]["type"] == "summaries_misconfigured"
    assert result_counts["provider_calls"] == 0
    assert counts(repository) == dict.fromkeys(SUMMARY_TABLES, 0)


def test_a_bare_key_echoed_by_a_client_is_scrubbed(repository, monkeypatch):
    seed(repository)
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    monkeypatch.setattr(
        summary_runner, "provider_configuration_problem", lambda environ=None: None
    )

    class Echoing(Client):
        def generate(self, system_prompt, user_prompt, response_schema):
            raise TypeError(f"defect while holding {CANARY}")

    _, errors = run_scheduled_summaries(
        repository,
        pipeline_version=VERSION,
        base_run_id="inv:summaries",
        horizon=HORIZON,
        client_factory=Echoing,
    )
    assert errors and CANARY not in json.dumps(errors)


def test_configured_run_uses_the_injected_client(repository, monkeypatch):
    labels = seed(repository)
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    monkeypatch.setattr(
        summary_runner, "provider_configuration_problem", lambda environ=None: None
    )
    client = Client()
    result_counts, errors = run_scheduled_summaries(
        repository,
        pipeline_version=VERSION,
        base_run_id="inv:summaries",
        horizon=HORIZON,
        client_factory=lambda: client,
    )
    assert errors == []
    assert result_counts["generated"] == len(labels)
    assert result_counts["provider_calls"] == len(client.calls) == len(labels)
    assert CANARY not in json.dumps([result_counts, errors])


# ----------------------------------------------------------------------
# Production policy factory
# ----------------------------------------------------------------------


def test_policy_resolves_without_a_key_or_a_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "512")

    def no_network(*args, **kwargs):
        raise AssertionError("policy resolution reached the provider")

    monkeypatch.setattr(GeminiClient, "generate", no_network)
    monkeypatch.setattr(GeminiClient, "_get_client", no_network)
    policy = production_generation_policy()
    assert policy.model == "gemini-test-model"
    assert policy.max_output_tokens == 512
    assert policy.max_attempts == PRODUCTION_MAX_ATTEMPTS
    assert policy.rules == tuple(tuple(rule) for rule in guarded.load_copy_rules())
    assert policy == resolve_generation_policy(
        GeminiClient(), max_attempts=PRODUCTION_MAX_ATTEMPTS
    )


def test_policy_never_carries_the_credential(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    policy = production_generation_policy()
    assert CANARY not in repr(policy)
    assert CANARY not in json.dumps(
        {name: str(getattr(policy, name)) for name in policy.__dataclass_fields__}
    )


def test_runner_and_factory_agree_on_the_policy(repository):
    client = Client(model="m-1", max_output_tokens=777)
    assert runner(repository, client).policy == production_generation_policy(client)


# ----------------------------------------------------------------------
# B. Cache
# ----------------------------------------------------------------------


def test_current_artifacts_cost_nothing_and_open_no_run(repository):
    labels = seed(repository)
    client = Client()
    first, errors = sweep(repository, client)
    assert errors == []
    assert first["generated"] == 3
    runs_after_first = summary_runs(repository)
    assert len(runs_after_first) == 1
    tables_after_first = counts(repository)

    for _ in range(2):  # repeated scheduled invocations
        again, errors = sweep(repository, client)
        assert errors == []
        assert again["cache_hits"] == len(labels)
        assert again["provider_calls"] == 0
        assert again["partitions_current"] == 1
        assert again["partitions_with_work"] == 0
    assert len(client.calls) == 3
    assert summary_runs(repository) == runs_after_first
    assert counts(repository) == tables_after_first


# ----------------------------------------------------------------------
# C. Invalidation
# ----------------------------------------------------------------------


def test_changed_evidence_regenerates(repository):
    seed(repository)
    client = Client()
    sweep(repository, client)
    # A story change re-mints the whole partition's theme set.
    labels = seed(repository, themes=("Alpha", "Beta", "Gamma", "Delta"))
    result, _ = sweep(repository, client)
    assert result["generated"] == 4
    assert all(current(repository, client, "TSLA", i) for i in labels.values())


@pytest.mark.parametrize(
    "changed",
    [
        {"model": "fake-model-2"},
        {"max_output_tokens": 2048},
    ],
)
def test_changed_policy_regenerates(repository, changed):
    labels = seed(repository)
    old = Client()
    sweep(repository, old)
    new = Client(**changed)
    result, _ = sweep(repository, new)
    assert result["generated"] == 3
    assert result["cache_hits"] == 0
    assert all(current(repository, new, "TSLA", i) for i in labels.values())
    # Currentness is per policy: the old one still names its own artifact.
    assert all(current(repository, old, "TSLA", i) for i in labels.values())


def test_changed_copy_rules_regenerate(repository, monkeypatch):
    seed(repository)
    client = Client()
    sweep(repository, client)
    rules = guarded.load_copy_rules()
    monkeypatch.setattr(
        guarded,
        "load_copy_rules",
        lambda: tuple(rules) + (("advice", "phrase", "zz-never-used-zz"),),
    )
    result, _ = sweep(repository, client)
    assert result["generated"] == 3


# ----------------------------------------------------------------------
# D. Persistence
# ----------------------------------------------------------------------


def test_accepted_result_is_current_and_citations_resolve(repository):
    labels = seed(repository)
    client = Client()
    sweep(repository, client)
    for label, theme_id in labels.items():
        found = current(repository, client, "TSLA", theme_id)
        assert found is not None
        for sentence in found.artifact.sentences:
            for citation in sentence.citations:
                evidence = found.evidence_for(citation.story_id)
                assert evidence.title == f"TSLA {label} story"
                assert evidence.urls
                assert evidence.raw_item_ids
                assert repository.read.story(citation.story_id) is not None


def test_unavailable_records_accounting_and_no_artifact(repository):
    labels = seed(repository)
    client = Client(fail={"TSLA Alpha story", "TSLA Beta story", "TSLA Gamma story"})
    result, errors = sweep(repository, client)
    assert result["unavailable"] == 3
    assert result["provider_calls"] == 6
    tables = counts(repository)
    assert tables["summary_artifacts"] == 0
    assert tables["summary_generations"] == 3
    assert tables["summary_generation_attempts"] == 6
    assert {e["reason"] for e in errors if e["type"] == "summary_unavailable"} == {
        "provider_unavailable"
    }
    # No fabricated fallback: nothing is current.
    assert not any(current(repository, client, "TSLA", i) for i in labels.values())
    [run] = summary_runs(repository)
    assert run["status"] == "degraded"


# ----------------------------------------------------------------------
# E. Population health
# ----------------------------------------------------------------------


def test_m2_only_partition_is_refused_without_calls(repository):
    seed(repository, "TSLA")
    seed_m2_only(repository, "NVDA")
    client = Client()
    result, _ = sweep(repository, client)
    assert result["partitions_considered"] == 2
    assert result["partitions_refused"] == 1
    assert result["refused_theme_set_missing"] == 1
    assert all("NVDA" not in prompt for prompt in client.calls)
    assert repository.read.summary_generations("NVDA", DAY, VERSION) == []
    assert repository.read.summary_artifacts("NVDA", DAY, VERSION) == []
    assert all(run["ticker"] != "NVDA" for run in summary_runs(repository))


def test_inconsistent_population_is_refused_without_calls(repository):
    seed(repository, story_count_delta=1)
    client = Client()
    result, _ = sweep(repository, client)
    assert result["partitions_refused"] == 1
    assert result["refused_source_story_count_mismatch"] == 1
    assert client.calls == []
    assert counts(repository) == dict.fromkeys(SUMMARY_TABLES, 0)
    assert summary_runs(repository) == []


# ----------------------------------------------------------------------
# F. Failure isolation
# ----------------------------------------------------------------------


def test_unavailable_theme_does_not_stop_its_siblings(repository):
    labels = seed(repository)
    client = Client(fail={"TSLA Alpha story"})
    result, _ = sweep(repository, client)
    assert (result["unavailable"], result["generated"]) == (1, 2)
    assert current(repository, client, "TSLA", labels["Alpha"]) is None
    assert current(repository, client, "TSLA", labels["Beta"]) is not None
    assert current(repository, client, "TSLA", labels["Gamma"]) is not None
    [run] = summary_runs(repository)
    assert run["status"] == "degraded"


def test_an_exception_fails_only_its_partition(repository):
    tsla = seed(repository, "TSLA")
    nvda = seed(repository, "NVDA")
    before = intelligence_rows(repository)
    client = Client(crash={"TSLA Beta story"})
    result, errors = sweep(repository, client)

    assert result["partitions_failed"] == 1
    assert result["themes_not_attempted"] == 1  # TSLA Gamma
    failure = [e for e in errors if e["type"] == "summary_partition_failed"]
    assert [(e["ticker"], e["trading_day"]) for e in failure] == [("TSLA", DAY)]
    assert CANARY not in json.dumps(errors)
    # What committed before the crash stays; nothing after it was tried.
    assert current(repository, client, "TSLA", tsla["Alpha"]) is not None
    assert (
        repository.read.summary_generations(
            "TSLA", DAY, VERSION, theme_id=tsla["Gamma"]
        )
        == []
    )
    # The other partition was fully processed.
    assert all(current(repository, client, "NVDA", i) for i in nvda.values())
    statuses = {run["ticker"]: run["status"] for run in summary_runs(repository)}
    assert statuses == {"NVDA": "success", "TSLA": "failed"}
    # Evidence, stories and themes are exactly as they were.
    assert intelligence_rows(repository) == before


def test_summary_failures_leave_intelligence_untouched(repository):
    seed(repository)
    before = intelligence_rows(repository)
    sweep(repository, Client(fail={"TSLA Alpha story"}))
    sweep(repository, Client(crash={"TSLA Alpha story"}), retry=RetryPolicy())
    assert intelligence_rows(repository) == before


# ----------------------------------------------------------------------
# G. Retry
# ----------------------------------------------------------------------


def test_cooldown_suppresses_calls_until_it_passes(repository, clock):
    seed(repository)
    client = Client(fail={"TSLA Alpha story"})
    sweep(repository, client)
    calls = len(client.calls)
    runs = summary_runs(repository)

    suppressed, errors = sweep(repository, client)
    assert suppressed["cooldown"] == 1
    assert suppressed["provider_calls"] == 0
    assert len(client.calls) == calls
    assert summary_runs(repository) == runs  # no run for suppressed work
    assert any(e["type"] == "summaries_retry_suppressed" for e in errors)

    clock.advance(timedelta(minutes=61))
    retried, _ = sweep(repository, client)
    assert retried["unavailable"] == 1
    assert len(client.calls) == calls + 2


def test_exhausted_key_stops_and_a_new_key_escapes_it(repository, clock):
    labels = seed(repository, themes=("Alpha",))
    client = Client(fail={"TSLA Alpha story"})
    for _ in range(3):
        sweep(repository, client)
        clock.advance(timedelta(minutes=61))
    calls = len(client.calls)
    assert calls == 6

    exhausted, _ = sweep(repository, client)
    assert exhausted["exhausted"] == 1
    assert exhausted["provider_calls"] == 0
    assert len(client.calls) == calls

    # A changed policy is a new exact key and may generate at once.
    fresh = Client(model="fake-model-2")
    result, _ = sweep(repository, fresh)
    assert result["generated"] == 1
    assert current(repository, fresh, "TSLA", labels["Alpha"]) is not None


# ----------------------------------------------------------------------
# H. Budget
# ----------------------------------------------------------------------


def test_budget_default_and_validation():
    assert provider_call_budget({}) == DEFAULT_PROVIDER_CALL_BUDGET
    assert 0 < DEFAULT_PROVIDER_CALL_BUDGET < 1000
    assert provider_call_budget({summary_runner.BUDGET_ENV: " 7 "}) == 7
    assert provider_call_budget({summary_runner.BUDGET_ENV: "0"}) == 0
    for bad in ("-1", "1.5", "lots"):
        with pytest.raises(SummaryConfigError):
            provider_call_budget({summary_runner.BUDGET_ENV: bad})
    with pytest.raises(SummaryConfigError):
        runner(None, Client(), budget=-1)  # type: ignore[arg-type]


def test_zero_budget_makes_no_call_and_no_row(repository):
    seed(repository)
    client = Client()
    result, errors = sweep(repository, client, budget=0)
    assert result["deferred_budget"] == 3
    assert client.calls == []
    assert counts(repository) == dict.fromkeys(SUMMARY_TABLES, 0)
    assert summary_runs(repository) == []
    assert any(e["type"] == "summaries_budget_exhausted" for e in errors)


def test_most_salient_theme_spends_the_budget_first(repository):
    aapl = seed(repository, "AAPL", themes=("First", "Second"))
    seed(repository, "TSLA", themes=("Top", "Next"))
    client = Client()
    result, _ = sweep(repository, client, budget=PRODUCTION_MAX_ATTEMPTS)
    assert result["generated"] == 1
    assert result["deferred_budget"] == 1 + 2  # AAPL Second, both TSLA
    assert current(repository, client, "AAPL", aapl["First"]) is not None

    # Full order: days, then tickers, then salience rank.
    order_client = Client()
    sweep(repository, order_client)
    titles = [
        re.search(r"(AAPL|TSLA) \w+ story", prompt).group(0)
        for prompt in order_client.calls
    ]
    assert titles == ["AAPL Second story", "TSLA Top story", "TSLA Next story"]


@pytest.mark.parametrize("budget", range(0, 8))
def test_reservation_never_lets_real_calls_exceed_the_budget(repository, budget):
    seed(repository, themes=("A1", "A2", "A3", "A4"))
    always_two = Client(
        fail={"TSLA A1 story", "TSLA A2 story", "TSLA A3 story", "TSLA A4 story"}
    )
    result, _ = sweep(repository, always_two, budget=budget)
    assert len(always_two.calls) == result["provider_calls"] <= budget
    assert result["provider_calls"] == 2 * min(4, budget // 2)


def test_reservation_is_refunded_to_real_calls(repository):
    seed(repository, themes=("A1", "A2", "A3"))
    one_each = Client()
    result, _ = sweep(repository, one_each, budget=3)
    # 1 call (refund 1) -> 1 call (refund 1) -> 1 left < 2 reserved: defer.
    assert (result["generated"], result["deferred_budget"]) == (2, 1)
    assert result["provider_calls"] == 2


def test_regeneration_consumes_its_reservation(repository):
    seed(repository, themes=("A1", "A2"))
    client = Client(reject_first=True)
    result, _ = sweep(repository, client, budget=3)
    assert result["generated"] == 1
    assert result["deferred_budget"] == 1
    assert len(client.calls) == result["provider_calls"] == 2


def test_deferred_themes_leave_no_rows_and_run_next_time(repository):
    labels = seed(repository)
    client = Client()
    first, _ = sweep(repository, client, budget=2)
    assert (first["generated"], first["deferred_budget"]) == (1, 2)
    for label in ("Beta", "Gamma"):
        assert (
            repository.read.summary_generations(
                "TSLA", DAY, VERSION, theme_id=labels[label]
            )
            == []
        )
    [run] = summary_runs(repository)
    assert run["status"] == "degraded"

    second, _ = sweep(repository, client)
    assert (second["cache_hits"], second["generated"]) == (1, 2)
    assert all(current(repository, client, "TSLA", i) for i in labels.values())


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def test_selection_is_bounded_by_the_horizon(repository, clock):
    seed(repository, day="2026-07-15")
    clock.advance(timedelta(days=10))
    seed(repository, "NVDA", day="2026-07-30")
    pairs = runner(repository, Client()).partitions()
    assert pairs == [("2026-07-30", "NVDA")]


def test_selection_orders_days_then_tickers(repository):
    seed(repository, "TSLA", day="2026-07-22")
    seed(repository, "AAPL", day="2026-07-23")
    seed(repository, "NVDA", day="2026-07-22")
    assert runner(repository, Client()).partitions() == [
        ("2026-07-22", "NVDA"),
        ("2026-07-22", "TSLA"),
        ("2026-07-23", "AAPL"),
    ]


# ----------------------------------------------------------------------
# K. The run contract A3b builds on
# ----------------------------------------------------------------------


def test_one_run_carries_unavailable_then_accepted_generations(repository):
    """Typed ``unavailable`` is ordinary partial work, not a run failure."""

    labels = seed(repository)
    client = Client(fail={"TSLA Alpha story"})
    with repository.stage_run(
        run_id="contract:TSLA:" + DAY,
        stage=STAGE,
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run:
        outcomes = [
            ensure_summary(
                repository,
                run=run,
                ticker="TSLA",
                trading_day=DAY,
                pipeline_version=VERSION,
                theme_id=labels[label],
                client=client,
            )
            for label in ("Alpha", "Beta", "Gamma")
        ]
        assert not run.settled
    assert [o.source for o in outcomes] == ["unavailable", "generated", "generated"]
    [row] = summary_runs(repository)
    assert row["status"] == "degraded"
    assert counts(repository)["summary_generations"] == 3


def test_a_raising_mutation_settles_the_run_and_refuses_the_rest(repository):
    """A logged mutation that raises ends the run; what committed stays."""

    labels = seed(repository)
    client = Client()
    with pytest.raises(Phase0ValidationError):
        with repository.stage_run(
            run_id="contract-fail:TSLA:" + DAY,
            stage=STAGE,
            trading_day=DAY,
            pipeline_version=VERSION,
            ticker="TSLA",
        ) as run:
            first = ensure_summary(
                repository,
                run=run,
                ticker="TSLA",
                trading_day=DAY,
                pipeline_version=VERSION,
                theme_id=labels["Alpha"],
                client=client,
            )
            assert first.source == "generated"
            population = repository.read.theme_population("TSLA", DAY, VERSION)
            from phase0.summaries import build_generation_input

            generation_input = build_generation_input(population, labels["Beta"])
            policy = resolve_generation_policy(client)
            result = guarded.generate_guarded_summary(
                generation_input, client=client, policy=policy
            )
            wrong_policy = resolve_generation_policy(Client(model="other"))
            with pytest.raises(Phase0ValidationError):
                repository.persist_summary_generation(
                    run=run,
                    result=result,
                    generation_input=generation_input,
                    policy=wrong_policy,
                )
            assert run.settled
            with pytest.raises(Phase0RunContextError):
                repository.persist_summary_generation(
                    run=run,
                    result=result,
                    generation_input=generation_input,
                    policy=policy,
                )
            raise Phase0ValidationError("propagate as the runner does")
    [row] = summary_runs(repository)
    assert row["status"] == "failed"
    # The first theme's committed generation is durable.
    assert current(repository, client, "TSLA", labels["Alpha"]) is not None
    assert counts(repository)["summary_generations"] == 1


def test_a_reused_base_run_id_does_not_duplicate_accounting(repository):
    seed(repository)
    client = Client()
    sweep(repository, client, base="same:summaries")
    tables = counts(repository)
    again, errors = sweep(repository, client, base="same:summaries")
    assert errors == []
    assert again["cache_hits"] == 3
    assert counts(repository) == tables
    assert len(summary_runs(repository)) == 1


def test_a_reused_identity_after_cooldown_does_no_work_and_rewrites_nothing(
    repository, clock
):
    """The review's reproduction: the same base id, past the cooldown.

    Opening the same run again would rewrite its outcome and let A3's
    per-run idempotency key swallow new provider attempts, so the reused
    identity is refused before any provider work.
    """

    labels = seed(repository)
    client = Client(fail={"TSLA Alpha story"})
    first, _ = sweep(repository, client, base="fixed:summaries")
    assert first["unavailable"] == 1
    [before] = summary_runs(repository)
    assert before["status"] == "degraded"
    calls = len(client.calls)
    tables = counts(repository)

    clock.advance(timedelta(minutes=61))
    again, errors = sweep(repository, client, base="fixed:summaries")

    assert len(client.calls) == calls
    assert again["provider_calls"] == 0
    assert again["partitions_identity_reused"] == 1
    assert again["partitions_with_work"] == 0
    assert [e for e in errors if e["type"] == "summary_run_identity_reused"] == [
        {
            "type": "summary_run_identity_reused",
            "ticker": "TSLA",
            "trading_day": DAY,
            "pending": 1,
        }
    ]
    assert counts(repository) == tables
    assert summary_runs(repository) == [before]  # status, counts, errors intact
    assert current(repository, client, "TSLA", labels["Alpha"]) is None

    # A fresh identity retries at once, as the retry policy allows.
    fresh, _ = sweep(repository, client, base="fresh:summaries")
    assert fresh["unavailable"] == 1
    assert len(client.calls) == calls + 2
    assert counts(repository)["summary_generation_attempts"] == (
        tables["summary_generation_attempts"] + 2
    )


def test_a_reused_identity_over_current_artifacts_is_harmless(repository):
    seed(repository)
    client = Client()
    sweep(repository, client, base="fixed:summaries")
    [before] = summary_runs(repository)
    tables = counts(repository)
    again, errors = sweep(repository, client, base="fixed:summaries")
    assert errors == []
    assert again["cache_hits"] == 3
    assert again["partitions_identity_reused"] == 0
    assert counts(repository) == tables
    assert summary_runs(repository) == [before]


# ----------------------------------------------------------------------
# P1: the configured credential never becomes durable text
# ----------------------------------------------------------------------

SECRET = "synthetic-do-not-use-9f3c1e7a5b"


def secret_environ(**extra):
    return {"GEMINI_API_KEY": SECRET, **extra}


def every_durable_text(repository) -> str:
    """Every row of every table, as text: the whole database is searched."""

    with repository.admin.connect_writable() as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        ]
        return "\n".join(
            repr(tuple(row))
            for table in tables
            for row in connection.execute(f"SELECT * FROM {table}")
        )


def attempt_errors(repository):
    with repository.admin.connect_writable() as connection:
        return [
            (row[0], row[1])
            for row in connection.execute(
                "SELECT outcome, error FROM summary_generation_attempts ORDER BY rowid"
            )
        ]


class Transport:
    """Stands in for ``google.genai.Client``: its call raises what a real
    transport would.  No socket is ever opened."""

    def __init__(self, error):
        self.error = error
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        raise self.error


def real_client(monkeypatch, error):
    """The production ``GeminiClient``, its connection replaced by ``Transport``."""

    client = GeminiClient(api_key=SECRET, model="gemini-test")
    transport = Transport(error)
    monkeypatch.setattr(client, "_get_client", lambda: transport)
    return client, transport


def scheduled(repository, client, **environ):
    return run_scheduled_summaries(
        repository,
        pipeline_version=VERSION,
        base_run_id=f"inv-{next(_BASES)}:summaries",
        horizon=HORIZON,
        environ=secret_environ(**environ),
        client_factory=lambda: client,
    )


def test_a_bare_key_in_a_transport_error_is_never_persisted(repository, monkeypatch):
    import httpx

    seed(repository, themes=("Alpha",))
    client, transport = real_client(
        monkeypatch, httpx.ConnectError(f"connect failed for key {SECRET}")
    )
    result, errors = scheduled(repository, client)

    # Classification is unchanged: a request failure, retried once.
    assert result["unavailable"] == 1
    assert transport.calls == 2
    [generation] = repository.read.summary_generations("TSLA", DAY, VERSION)
    assert generation.reason == "provider_unavailable"
    recorded = attempt_errors(repository)
    assert [outcome for outcome, _ in recorded] == ["provider_error"] * 2
    for _, error in recorded:
        assert SECRET not in error
        assert "[REDACTED]" in error
        assert "provider request failed: connect failed for key" in error
    assert SECRET not in json.dumps([result, errors])
    assert SECRET not in every_durable_text(repository)


def test_a_bare_key_in_a_timeout_keeps_its_class(repository, monkeypatch):
    import httpx

    seed(repository, themes=("Alpha",))
    client, _ = real_client(monkeypatch, httpx.ReadTimeout(f"slow {SECRET}"))
    scheduled(repository, client)
    assert [o for o, _ in attempt_errors(repository)] == ["provider_timeout"] * 2
    assert SECRET not in every_durable_text(repository)


def test_a_bare_key_in_an_auth_rejection_keeps_its_class(repository, monkeypatch):
    from google.genai import errors as genai_errors

    seed(repository, themes=("Alpha",))
    rejection = genai_errors.APIError(
        401, {"error": {"message": f"bad key {SECRET}", "status": "UNAUTHENTICATED"}}
    )
    client, transport = real_client(monkeypatch, rejection)
    scheduled(repository, client)
    # Configuration failures are not retried, exactly as before.
    assert transport.calls == 1
    assert [o for o, _ in attempt_errors(repository)] == ["provider_unconfigured"]
    [generation] = repository.read.summary_generations("TSLA", DAY, VERSION)
    assert generation.reason == "provider_unconfigured"
    assert SECRET not in every_durable_text(repository)


def test_a_named_key_in_a_provider_error_is_never_persisted(repository, monkeypatch):
    import httpx

    seed(repository, themes=("Alpha",))
    client, _ = real_client(
        monkeypatch, httpx.ConnectError(f"refused: api_key={SECRET}")
    )
    result, errors = scheduled(repository, client)
    assert SECRET not in json.dumps([result, errors])
    assert all(SECRET not in error for _, error in attempt_errors(repository))
    assert SECRET not in every_durable_text(repository)


def test_a_bare_key_in_an_unexpected_exception_is_never_persisted(repository):
    seed(repository, "TSLA")
    seed(repository, "NVDA")

    class Leaking(Client):
        def generate(self, system_prompt, user_prompt, response_schema):
            if "TSLA" in user_prompt:
                raise TypeError(f"defect while holding {SECRET}")
            return super().generate(system_prompt, user_prompt, response_schema)

    result, errors = scheduled(repository, Leaking())

    assert result["partitions_failed"] == 1
    [failed] = [row for row in summary_runs(repository) if row["ticker"] == "TSLA"]
    assert failed["status"] == "failed"
    assert SECRET not in failed["errors"]
    assert "TypeError" in failed["errors"]  # still says what happened
    assert "[REDACTED]" in failed["errors"]
    assert SECRET not in json.dumps([result, errors])
    assert SECRET not in every_durable_text(repository)
    # The other partition was unaffected.
    assert result["generated"] == 3


def test_an_ordinary_provider_diagnostic_survives(repository, monkeypatch):
    import httpx

    seed(repository, themes=("Alpha",))
    client, _ = real_client(
        monkeypatch, httpx.ConnectError("503 upstream connect reset")
    )
    scheduled(repository, client)
    for _, error in attempt_errors(repository):
        assert "503 upstream connect reset" in error
        assert "[REDACTED]" not in error


def test_the_scrubbing_boundary_keeps_the_policy_and_hides_the_key(repository):
    client = Client(model="m-1", max_output_tokens=777)
    scrubbing = runner(repository, client)
    plain = SummaryRunner(
        repository,
        pipeline_version=VERSION,
        client=client,
        budget=2,
        horizon=HORIZON,
        secret=SECRET,
    )
    assert plain.policy == scrubbing.policy == production_generation_policy(client)
    assert SECRET not in repr(plain.client)
    assert SECRET not in repr(plain.policy)


def test_scrub_literal_secret_contract():
    scrub = summary_runner.scrub_literal_secret
    assert scrub(f"a {SECRET} b", SECRET) == "a [REDACTED] b"
    assert scrub({"k": [f"x{SECRET}", 3]}, SECRET) == {"k": ["x[REDACTED]", 3]}
    assert scrub(f"a {SECRET}", None) == f"a {SECRET}"
    assert scrub(f"a {SECRET}", "   ") == f"a {SECRET}"


def test_run_ids_are_partition_scoped_and_never_retry_shaped(repository):
    seed(repository, "TSLA")
    seed(repository, "NVDA")
    sweep(repository, Client(), base="inv-x:summaries")
    ids = sorted(run["run_id"] for run in summary_runs(repository))
    assert ids == [
        f"inv-x:summaries:NVDA:{DAY}",
        f"inv-x:summaries:TSLA:{DAY}",
    ]
    assert not any(pipeline.is_retry_run(run_id) for run_id in ids)


# ----------------------------------------------------------------------
# I/J. Placement in the pipeline
# ----------------------------------------------------------------------


def test_summaries_are_outside_the_intelligence_stages_and_coordinator():
    assert "summaries" not in pipeline.INTELLIGENCE_STAGES
    assert pipeline.DOWNSTREAM_STAGES.index(
        pipeline.summaries_stage
    ) > pipeline.DOWNSTREAM_STAGES.index(pipeline.intelligence_stage)
    coordinator = (ROOT / "phase0" / "coordinator.py").read_text("utf-8")
    assert "summary" not in coordinator.lower()
    for reviewed in ("ai/guarded_summary.py", "phase0/summary_lifecycle.py"):
        assert "summary_runner" not in (ROOT / reviewed).read_text("utf-8")


def test_replay_capability_is_explicit():
    capabilities = pipeline.replay_capabilities()
    assert "summarization" in capabilities["unsupported"]
    assert "summaries" in capabilities["live_only"]


# ----------------------------------------------------------------------
# L. Schema
# ----------------------------------------------------------------------


def test_no_new_migration(repository):
    migrations = sorted((ROOT / "phase0" / "migrations").glob("*.sql"))
    assert not list((ROOT / "phase0" / "migrations").glob("017*"))
    assert migrations[-1].name == "016_summary_artifacts.sql"
    assert repository.schema_version() == 16

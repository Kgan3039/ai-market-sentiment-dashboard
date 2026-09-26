"""A3: the persisted summary lifecycle over a real migrated database.

Every population here is what the ordinary story and theme reconcilers
wrote and what ``Phase0Reader.theme_population`` reads back, exactly as in
``tests/test_phase0_summary_input.py`` (whose day builder is reused).  The
provider is always a fake; the database, the migrations, the run log and
the reconcilers are real.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import itertools
import json
import re
import sqlite3
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ai.guarded_summary as guarded
import phase0.summary_lifecycle as lifecycle
from ai.guarded_summary import (
    AttemptRecord,
    GenerationPolicy,
    GuardedSummaryError,
    citation_id_for,
    generate_guarded_summary,
    load_copy_rules,
    resolve_generation_policy,
)
from ai.summarization import (
    GenerationUsage,
    ProviderConfigurationError,
    ProviderRequestError,
    Sentence,
    ThemeSummary,
)
from phase0.errors import Phase0RunContextError, Phase0ValidationError
from phase0.models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import (
    SUMMARY_DISCARD_INPUT_CHANGED,
    SUMMARY_ARTIFACT_ACCEPTED,
    SUMMARY_GENERATION_ACCEPTED,
    SUMMARY_GENERATION_DISCARDED_DUPLICATE,
    SUMMARY_GENERATION_DISCARDED_STALE,
    SUMMARY_GENERATION_UNAVAILABLE,
    SUMMARY_INVALIDATED_CORRUPT,
    Phase0Repository,
    summary_artifact_digest,
    summary_artifact_digest_of,
)
from phase0.summaries import (
    REFUSED_NO_THEME_SET,
    REFUSED_UNKNOWN_THEME,
    build_generation_input,
)
from phase0.schema import split_statements
from phase0.summary_lifecycle import (
    REJECT_DIGEST,
    REJECT_IDENTITY,
    REJECT_STRUCTURE,
    STAGE,
    RetryPolicy,
    SummaryLifecycleOutcome,
    current_summary_artifact,
    ensure_summary,
    validate_persisted_artifact,
)
from test_phase0_summary_input import (
    DAY,
    SUMMARY_TABLES,
    TICKER,
    VERSION,
    Day,
    EchoClient,
    build_day,
)
import test_phase0_summary_input as summary_input

ROOT = Path(__file__).resolve().parents[1]
ID_LINE_RE = re.compile(r"- id: (\S+)")
_RUN_IDS = itertools.count(1000)
NOW = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
CANARY = "sk-CANARY-SECRET-0123456789"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@contextmanager
def open_run(repository: Phase0Repository, *, run_id: str | None = None):
    with repository.stage_run(
        run_id=run_id or f"summaries-{next(_RUN_IDS)}",
        stage=STAGE,
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        yield run


def ensure(day: Day, client, *, theme="Deliveries", run_id=None, **kwargs):
    """One ``ensure_summary`` invocation under its own ``summaries`` run."""

    with open_run(day.repository, run_id=run_id) as run:
        return ensure_summary(
            day.repository,
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=VERSION,
            theme_id=day.theme_ids[theme],
            client=client,
            **kwargs,
        )


def current(day: Day, client, *, theme="Deliveries", theme_id=None):
    return current_summary_artifact(
        day.repository.read,
        TICKER,
        DAY,
        VERSION,
        day.theme_ids[theme] if theme_id is None else theme_id,
        resolve_generation_policy(client),
    )


def live_input(day: Day, theme="Deliveries"):
    """The frozen A2 input for the theme as the database has it right now."""

    return build_generation_input(
        day.repository.read.theme_population(TICKER, DAY, VERSION),
        day.theme_ids[theme],
    )


def stored_verdict(day: Day, client, *, theme="Deliveries"):
    """The central validator's verdict on the stored row for the live key."""

    generation_input = live_input(day, theme)
    policy = resolve_generation_policy(client)
    artifact = day.repository.read.summary_artifact(
        generation_input.theme.theme_id,
        generation_input.input_fingerprint,
        policy.fingerprint,
    )
    assert artifact is not None
    return validate_persisted_artifact(artifact, generation_input, policy)


def artifacts(day: Day, theme="Deliveries"):
    return day.repository.read.summary_artifacts(
        TICKER, DAY, VERSION, theme_id=day.theme_ids[theme]
    )


def generations(day: Day, theme="Deliveries"):
    return day.repository.read.summary_generations(
        TICKER, DAY, VERSION, theme_id=day.theme_ids[theme]
    )


def table_counts(repository: Phase0Repository) -> dict[str, int]:
    return {table: repository.read.count(table) for table in SUMMARY_TABLES}


def dump_all_rows(repository: Phase0Repository) -> str:
    with repository.admin.connect_writable() as connection:
        rows = []
        for table in SUMMARY_TABLES + ("run_log",):
            for row in connection.execute(f"SELECT * FROM {table}"):
                rows.append(repr(tuple(row)))
    return "\n".join(rows)


def reconcile_theme_set(
    repository: Phase0Repository,
    themes,
    *,
    other=(),
    excluded=(),
    story_count=None,
    fingerprint_suffix="",
):
    """Reconcile a theme set; ``themes`` is (label, story_ids, citations, key).

    The fixture's theme fingerprints are synthetic (``fp-<rank>``), so a
    membership change that M5 would fingerprint afresh has to say so with
    ``fingerprint_suffix`` to get the new theme rows M5 would produce.
    """

    with repository.stage_run(
        run_id=f"themes-{next(_RUN_IDS)}",
        stage="themes",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        return repository.reconcile_themes(
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=VERSION,
            theme_set=ThemeSetRecord(
                method="hdbscan",
                method_reason="clustered",
                source_metadata=(
                    None if story_count is None else {"story_count": story_count}
                ),
                config_fingerprint="cfg",
                algorithm_version="m5.1",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            ),
            themes=[
                ThemeRecord(
                    fingerprint=f"fp-{rank}{fingerprint_suffix}",
                    theme_key=key,
                    label=label,
                    label_source="representative_title",
                    story_ids=tuple(story_ids),
                    citation_item_ids=tuple(citations),
                    status="ready",
                    salience_rank=rank,
                    story_count=len(story_ids),
                )
                for rank, (label, story_ids, citations, key) in enumerate(
                    themes, start=1
                )
            ],
            other_coverage=[
                OtherCoverageRecord(story_id=s, reason="clustering_noise")
                for s in other
            ],
            excluded=[
                ExcludedStoryRecord(story_id=s, reason="no_encodable_text")
                for s in excluded
            ],
            terminal=True,
        )


def day_themes(day: Day, *, deliveries_key="key-1", robotaxi_key="key-2"):
    """The day's theme set as ``reconcile_theme_set`` takes it."""

    deliveries = [
        day.story_ids["Tesla Q2 deliveries top estimates"],
        day.story_ids["Tesla guidance"],
    ]
    robotaxi = [day.story_ids["Robotaxi expands"]]
    return [
        (
            "Deliveries",
            deliveries,
            day.items["deliveries"] + day.items["guidance"],
            deliveries_key,
        ),
        ("Robotaxi", robotaxi, day.items["robotaxi"], robotaxi_key),
    ]


def replay_theme_set(day: Day, **keys):
    """Reconcile the day's theme set again, byte-for-byte unless re-keyed."""

    return reconcile_theme_set(
        day.repository,
        day_themes(day, **keys),
        other=[day.story_ids["Weekend column"]],
        excluded=[day.story_ids["No text"]],
        story_count=len(day.story_ids),
    )


def clear_theme_set(day: Day) -> None:
    with day.repository.stage_run(
        run_id=f"themes-{next(_RUN_IDS)}",
        stage="themes",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        day.repository.clear_theme_set(
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=VERSION,
            terminal=True,
        )


_MIGRATION_016 = ROOT / "phase0" / "migrations" / "016_summary_artifacts.sql"
_SEALED_TRIGGER_RE = re.compile(
    r"CREATE TRIGGER IF NOT EXISTS (trg_summary_\w+_sealed_\w+)"
)


def _sealed_trigger_statements() -> list[tuple[str, str]]:
    """``(name, CREATE TRIGGER ...)`` for every sealing trigger in 016."""

    found = []
    for statement in split_statements(_MIGRATION_016.read_text(encoding="utf-8")):
        match = _SEALED_TRIGGER_RE.search(statement)  # a comment may lead
        if match:
            found.append((match.group(1), statement))
    assert len(found) == 4, [name for name, _ in found]
    return found


def corrupt(day: Day, statement: str, parameters=()) -> None:
    """Damage stored artifact rows the way only a broken database could.

    The sealing triggers refuse this through ordinary SQL, so they are
    dropped for the one statement and recreated exactly as migration 016
    defines them.  What remains is storage the triggers never saw damaged
    -- which is precisely what the read-side digest has to catch.
    """

    triggers = _sealed_trigger_statements()
    with day.repository.admin.connect_writable() as connection:
        for name, _ in triggers:
            connection.execute(f"DROP TRIGGER {name}")
        try:
            connection.execute(statement, parameters)
        finally:
            for _, create in triggers:
                connection.execute(create)


def sealed_sql_is_refused(day: Day, statement: str, parameters=()) -> None:
    """Ordinary SQL against a sealed artifact's children is refused."""

    with day.repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="sealed"):
            connection.execute(statement, parameters)


class ManualClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now = self.now + delta


def clocked_day(tmp_path, monkeypatch, clock: ManualClock) -> Day:
    def migrated(path):
        repository = Phase0Repository(path / "phase0.db", clock=clock)
        repository.migrate()
        return repository

    monkeypatch.setattr(summary_input, "migrated", migrated)
    return build_day(tmp_path)


# ----------------------------------------------------------------------
# Fake clients
# ----------------------------------------------------------------------


class NeverClient:
    """Fails the test if the provider is reached at all."""

    model = "fake-model"

    def generate(self, system_prompt, user_prompt, response_schema):
        raise AssertionError("the provider must not be called")


class UsageClient(EchoClient):
    def __init__(self, usage):
        super().__init__()
        self.last_usage = usage


class OrderedClient(EchoClient):
    """Cites the prompt's ids in reverse order, all of them, in each sentence."""

    def generate(self, system_prompt, user_prompt, response_schema):
        self.prompts.append(user_prompt)
        ids = ID_LINE_RE.findall(user_prompt)
        reverse = list(reversed(ids))
        return response_schema.model_validate(
            {
                "label": "Ordered coverage",
                "sentences": [
                    {"text": "Coverage, reversed.", "citation_ids": reverse},
                    {"text": "Coverage, forward.", "citation_ids": ids},
                    {"text": "Coverage, last only.", "citation_ids": reverse[:1]},
                ],
            }
        )


class FailingClient:
    """Every call fails at the provider: ``unavailable / provider_unavailable``."""

    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls += 1
        raise ProviderRequestError(f"provider request failed: api_key={CANARY}")


class UnconfiguredClient:
    model = "fake-model"

    def __init__(self):
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls += 1
        raise ProviderConfigurationError(
            f"credential refused: Authorization: Bearer {CANARY}"
        )


class BannedClient(EchoClient):
    """Always answers with advisory copy: ``unavailable / validation_exhausted``."""

    def generate(self, system_prompt, user_prompt, response_schema):
        self.prompts.append(user_prompt)
        ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": "Buy now",
                "sentences": [
                    {
                        "text": "Investors should buy the stock now.",
                        "citation_ids": ids,
                    },
                    {"text": "Coverage also notes guidance.", "citation_ids": ids},
                ],
            }
        )


class MidCallClient(EchoClient):
    """Runs a side effect while the provider is 'thinking', then answers."""

    def __init__(self, side_effect):
        super().__init__()
        self.side_effect = side_effect

    def generate(self, system_prompt, user_prompt, response_schema):
        self.side_effect()
        return super().generate(system_prompt, user_prompt, response_schema)


# ----------------------------------------------------------------------
# Persistence (1-9)
# ----------------------------------------------------------------------


def test_accepted_result_persists_exact_artifact(tmp_path):
    day = build_day(tmp_path)
    client = EchoClient()
    outcome = ensure(day, client)

    assert outcome.source == lifecycle.SOURCE_GENERATED
    assert outcome.provider_calls == 1 and len(client.prompts) == 1
    result = outcome.result
    artifact = outcome.artifact
    assert artifact is not None and result is not None and result.accepted
    assert artifact.status == "accepted"
    assert artifact.ticker == TICKER
    assert artifact.trading_day == DAY
    assert artifact.pipeline_version == VERSION
    assert artifact.theme_id == day.theme_ids["Deliveries"]
    assert artifact.theme_key == "key-1"
    assert artifact.input_fingerprint == result.input_fingerprint
    assert artifact.policy_fingerprint == result.policy_fingerprint
    assert artifact.policy_fingerprint == outcome.policy_fingerprint
    assert artifact.citation_convention == guarded.CITATION_CONVENTION
    assert artifact.prompt_version == guarded.PROMPT_VERSION
    assert artifact.model == "fake-model"
    assert artifact.guarantee == result.guarantee
    assert "semantic_faithfulness_not_established" in artifact.guarantee
    assert artifact.invalidated_at is None and artifact.invalidated_reason is None
    datetime.fromisoformat(artifact.created_at)
    # The one durable row is what the raw reader returns for the exact key.
    stored = day.repository.read.summary_artifact(
        artifact.theme_id, artifact.input_fingerprint, artifact.policy_fingerprint
    )
    assert stored == artifact
    assert table_counts(day.repository) == {
        "summary_artifacts": 1,
        "summary_sentences": 2,
        "summary_sentence_citations": 3,
        "summary_generations": 1,
        "summary_generation_attempts": 1,
    }


def test_generated_label_is_exact(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, EchoClient())
    assert outcome.artifact.label == "Coverage of deliveries"
    assert outcome.artifact.label == outcome.result.summary.label
    # And the theme row's own label is untouched, as is its summary column.
    theme = day.repository.read.theme(day.theme_ids["Deliveries"])
    assert theme["label"] == "Deliveries"
    assert theme["summary"] is None


def test_sentence_order_is_exact(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, OrderedClient())
    sentences = outcome.artifact.sentences
    assert [s.ordinal for s in sentences] == [1, 2, 3]
    assert [s.text for s in sentences] == [
        "Coverage, reversed.",
        "Coverage, forward.",
        "Coverage, last only.",
    ]
    assert [s.text for s in sentences] == [
        s.text for s in outcome.result.summary.sentences
    ]


def test_citation_order_and_story_ids_are_exact(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, OrderedClient())
    first = day.story_ids["Tesla Q2 deliveries top estimates"]
    second = day.story_ids["Tesla guidance"]
    assert first < second  # evidence order is story id order
    sentences = outcome.artifact.sentences
    assert [(c.position, c.story_id) for c in sentences[0].citations] == [
        (0, second),
        (1, first),
    ]
    assert [(c.position, c.story_id) for c in sentences[1].citations] == [
        (0, first),
        (1, second),
    ]
    assert [(c.position, c.story_id) for c in sentences[2].citations] == [(0, second)]
    # Round trip through the citation convention is exact.
    assert [[citation_id_for(c.story_id) for c in s.citations] for s in sentences] == [
        list(s.citation_ids) for s in outcome.result.summary.sentences
    ]
    assert outcome.artifact.story_ids == (second, first, first, second, second)


def test_unknown_usage_persists_as_null(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, EchoClient())  # no last_usage at all
    attempt = outcome.generation.attempts[0]
    assert (attempt.prompt_tokens, attempt.candidate_tokens, attempt.total_tokens) == (
        None,
        None,
        None,
    )
    with day.repository.admin.connect_writable() as connection:
        row = connection.execute(
            "SELECT prompt_tokens, candidate_tokens, total_tokens "
            "FROM summary_generation_attempts"
        ).fetchone()
    assert tuple(row) == (None, None, None)


def test_partially_known_usage_keeps_the_unknown_half_null(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, UsageClient(GenerationUsage(50, None, None)))
    attempt = outcome.generation.attempts[0]
    assert (attempt.prompt_tokens, attempt.candidate_tokens, attempt.total_tokens) == (
        50,
        None,
        None,
    )


def test_known_usage_is_exact(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, UsageClient(GenerationUsage(123, 45, 168)))
    attempt = outcome.generation.attempts[0]
    assert (attempt.prompt_tokens, attempt.candidate_tokens, attempt.total_tokens) == (
        123,
        45,
        168,
    )


def test_unavailable_creates_no_artifact_and_no_fallback(tmp_path):
    day = build_day(tmp_path)
    client = FailingClient()
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_UNAVAILABLE
    assert outcome.artifact is None and outcome.current is None
    assert outcome.provider_calls == 2 == client.calls
    assert outcome.generation.outcome == SUMMARY_GENERATION_UNAVAILABLE
    assert outcome.generation.reason == "provider_unavailable"
    assert outcome.generation.artifact_id is None
    counts = table_counts(day.repository)
    assert counts["summary_artifacts"] == 0
    assert counts["summary_sentences"] == 0
    assert counts["summary_generations"] == 1
    assert counts["summary_generation_attempts"] == 2
    assert current(day, NeverClient()) is None
    assert day.repository.read.theme(day.theme_ids["Deliveries"])["summary"] is None


# ----------------------------------------------------------------------
# Cache (10-15)
# ----------------------------------------------------------------------


def test_unchanged_input_and_policy_makes_zero_provider_calls(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    before = table_counts(day.repository)

    second = ensure(day, NeverClient())
    assert second.source == lifecycle.SOURCE_CACHE_HIT
    assert second.provider_calls == 0
    assert second.artifact == first.artifact
    assert second.input_fingerprint == first.input_fingerprint
    assert second.policy_fingerprint == first.policy_fingerprint
    assert table_counts(day.repository) == before  # nothing written on a hit
    assert current(day, NeverClient()).artifact == first.artifact


def test_changed_input_is_a_miss(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    # Same theme id, same membership: only the theme key moves, which is
    # part of A2's input identity.
    replay_theme_set(day, deliveries_key="key-1b")
    assert day.theme_ids["Deliveries"] == day.population().themes[0].theme_id
    client = EchoClient()
    second = ensure(day, client)
    assert second.source == lifecycle.SOURCE_GENERATED
    assert len(client.prompts) == 1
    assert second.input_fingerprint != first.input_fingerprint
    assert second.artifact.artifact_id != first.artifact.artifact_id
    assert current(day, NeverClient()).artifact == second.artifact
    assert len(artifacts(day)) == 2  # the old one is history, not gone


def test_changed_policy_is_a_miss(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())

    class OtherModel(EchoClient):
        model = "other-model"

    client = OtherModel()
    second = ensure(day, client)
    assert second.source == lifecycle.SOURCE_GENERATED and len(client.prompts) == 1
    assert second.policy_fingerprint != first.policy_fingerprint
    assert second.input_fingerprint == first.input_fingerprint
    # Each policy has its own current artifact; neither is stale.
    assert current(day, NeverClient()).artifact == first.artifact
    assert current(day, OtherModel()).artifact == second.artifact

    class Capped(EchoClient):
        max_output_tokens = 256

    third = ensure(day, Capped())
    assert third.source == lifecycle.SOURCE_GENERATED
    assert ensure(day, EchoClient(), max_attempts=1).source == (
        lifecycle.SOURCE_GENERATED
    )


def test_corrupt_cached_artifact_is_never_served(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        (first.artifact.artifact_id,),
    )
    # The pure read path reports None and writes nothing.
    before = table_counts(day.repository)
    assert current(day, NeverClient()) is None
    verdict = stored_verdict(day, NeverClient())
    assert not verdict.valid
    assert REJECT_STRUCTURE in verdict.codes and REJECT_DIGEST in verdict.codes
    assert "invalid_sentence_count" in verdict.codes
    assert table_counts(day.repository) == before
    # The raw reader still hands the row out under its exact key -- it is
    # the lifecycle that refuses to call it current.
    assert (
        day.repository.read.summary_artifact(
            first.artifact.theme_id,
            first.artifact.input_fingerprint,
            first.artifact.policy_fingerprint,
        ).status
        == "accepted"
    )


def test_cached_artifact_is_revalidated_with_a2s_pure_validator(tmp_path, monkeypatch):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    seen = []
    real = lifecycle.validate_candidate

    def spy(candidate, generation_input, *, rules=None):
        seen.append((candidate, generation_input, rules))
        return real(candidate, generation_input, rules=rules)

    monkeypatch.setattr(lifecycle, "validate_candidate", spy)
    second = ensure(day, NeverClient())
    assert second.source == lifecycle.SOURCE_CACHE_HIT
    assert len(seen) == 1
    candidate, generation_input, rules = seen[0]
    assert candidate == {
        "label": first.artifact.label,
        "sentences": [
            {
                "text": s.text,
                "citation_ids": [citation_id_for(c.story_id) for c in s.citations],
            }
            for s in first.artifact.sentences
        ],
    }
    assert generation_input.input_fingerprint == first.input_fingerprint
    assert rules == tuple(tuple(rule) for rule in load_copy_rules())

    # A validator that refuses is a miss, whatever the fingerprints say.
    monkeypatch.setattr(
        lifecycle,
        "validate_candidate",
        lambda *a, **k: guarded.ValidationVerdict(
            None, (guarded.ValidationFailure("banned_language", "sentence 1"),)
        ),
    )
    assert current(day, NeverClient()) is None


def test_copy_rule_change_is_a_miss(tmp_path):
    day = build_day(tmp_path)
    rules = tuple(load_copy_rules())
    first = ensure(day, EchoClient(), rules=rules)
    assert ensure(day, NeverClient(), rules=rules).source == lifecycle.SOURCE_CACHE_HIT

    tightened = rules + (("advisory", "literal", "guidance"),)
    client = EchoClient()
    second = ensure(day, client, rules=tightened)
    assert second.policy_fingerprint != first.policy_fingerprint
    # The old artifact's copy would trip the new rule, so under the new
    # policy there is no artifact at all and the provider is asked; its
    # answer trips the rule too and the result is honestly unavailable.
    assert second.source == lifecycle.SOURCE_UNAVAILABLE
    assert second.generation.reason == "validation_exhausted"
    assert len(client.prompts) == 2
    # The old policy's artifact is still current for the old policy.
    assert ensure(day, NeverClient(), rules=rules).source == lifecycle.SOURCE_CACHE_HIT


# ----------------------------------------------------------------------
# Policy consistency (16-18)
# ----------------------------------------------------------------------


def test_lookup_and_generation_use_one_resolved_policy_snapshot(tmp_path, monkeypatch):
    day = build_day(tmp_path)
    resolved = []
    real_resolve = lifecycle.resolve_generation_policy

    def resolve_spy(client, **kwargs):
        policy = real_resolve(client, **kwargs)
        resolved.append(policy)
        return policy

    looked_up = []
    real_lookup = lifecycle._reusable_artifact

    def lookup_spy(reader, generation_input, policy):
        looked_up.append(policy)
        return real_lookup(reader, generation_input, policy)

    generated = []
    real_generate = lifecycle.generate_guarded_summary

    def generate_spy(generation_input, **kwargs):
        generated.append(kwargs["policy"])
        return real_generate(generation_input, **kwargs)

    monkeypatch.setattr(lifecycle, "resolve_generation_policy", resolve_spy)
    monkeypatch.setattr(lifecycle, "_reusable_artifact", lookup_spy)
    monkeypatch.setattr(lifecycle, "generate_guarded_summary", generate_spy)

    # The rules file "changes" right after the policy is resolved: the
    # reloaded rules would reject everything the fake says.
    loads = []
    real_load = guarded.load_copy_rules

    def load_spy():
        loads.append(True)
        if len(loads) > 1:
            return real_load() + [("advisory", "literal", "coverage")]
        return real_load()

    monkeypatch.setattr(guarded, "load_copy_rules", load_spy)

    outcome = ensure(day, EchoClient())
    assert outcome.source == lifecycle.SOURCE_GENERATED
    assert len(resolved) == 1
    assert looked_up == [resolved[0]] and generated == [resolved[0]]
    assert looked_up[0] is generated[0]
    assert outcome.result.policy_fingerprint == resolved[0].fingerprint
    assert outcome.artifact.policy_fingerprint == resolved[0].fingerprint
    # Only the one resolution loaded rules; the generation used the snapshot.
    assert len(loads) == 1


def test_result_under_another_policy_is_refused_at_persist(tmp_path):
    day = build_day(tmp_path)
    repository = day.repository
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )

    class Capped(EchoClient):
        max_output_tokens = 512

    generated_under = resolve_generation_policy(Capped())
    result = generate_guarded_summary(
        generation_input, client=Capped(), policy=generated_under
    )
    looked_up_under = resolve_generation_policy(EchoClient())
    assert looked_up_under.fingerprint != generated_under.fingerprint

    with open_run(repository) as run:
        with pytest.raises(Phase0ValidationError, match="policy fingerprint"):
            repository.persist_summary_generation(
                run=run,
                result=result,
                generation_input=generation_input,
                policy=looked_up_under,
            )
    assert table_counts(repository) == {table: 0 for table in SUMMARY_TABLES}
    # The run recorded the refusal as its failure, durably.
    assert repository.read.run_log_rows(stage=STAGE)[-1]["status"] == "failed"

    # A2 itself refuses to generate under a policy its client does not match.
    with pytest.raises(GuardedSummaryError, match="no longer matches"):
        generate_guarded_summary(
            generation_input, client=EchoClient(), policy=generated_under
        )


def test_an_old_fingerprint_cannot_make_an_artifact_current(tmp_path):
    day = build_day(tmp_path)

    class OldModel(EchoClient):
        model = "old-model"

    old = ensure(day, OldModel())
    old_fingerprint = old.artifact.policy_fingerprint
    theme_id = day.theme_ids["Deliveries"]

    # No parameter takes a hash.
    with pytest.raises(GuardedSummaryError, match="not a fingerprint"):
        current_summary_artifact(
            day.repository.read, TICKER, DAY, VERSION, theme_id, old_fingerprint
        )
    # A policy cannot be assembled around a fingerprint it does not have.
    current_policy = resolve_generation_policy(EchoClient())
    with pytest.raises(GuardedSummaryError, match="does not match"):
        GenerationPolicy(
            model=current_policy.model,
            max_attempts=current_policy.max_attempts,
            rules=current_policy.rules,
            temperature=current_policy.temperature,
            max_output_tokens=current_policy.max_output_tokens,
            fingerprint=old_fingerprint,
        )
    # Under the active policy the old artifact is simply not current...
    assert current(day, EchoClient()) is None
    # ...although the raw reader still hands it out under its own exact key.
    assert (
        day.repository.read.summary_artifact(
            theme_id, old.input_fingerprint, old_fingerprint
        ).artifact_id
        == old.artifact.artifact_id
    )


# ----------------------------------------------------------------------
# Concurrency (19-24)
# ----------------------------------------------------------------------


def test_theme_change_during_the_call_is_discarded_stale(tmp_path):
    day = build_day(tmp_path)
    client = MidCallClient(lambda: replay_theme_set(day, deliveries_key="key-1b"))
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_DISCARDED_STALE
    assert outcome.provider_calls == 1
    assert outcome.artifact is None and outcome.current is None
    assert outcome.result.accepted  # A2 accepted; the write refused it
    assert outcome.generation.outcome == SUMMARY_GENERATION_DISCARDED_STALE
    assert outcome.generation.detail == SUMMARY_DISCARD_INPUT_CHANGED
    assert outcome.generation.input_fingerprint == outcome.input_fingerprint
    assert outcome.generation.provider_calls == 1
    assert table_counts(day.repository)["summary_artifacts"] == 0
    assert current(day, NeverClient()) is None


def test_theme_disappearing_during_the_call_is_discarded_stale(tmp_path):
    day = build_day(tmp_path)
    client = MidCallClient(lambda: clear_theme_set(day))
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_DISCARDED_STALE
    assert outcome.generation.detail == REFUSED_NO_THEME_SET
    assert table_counts(day.repository)["summary_artifacts"] == 0
    assert table_counts(day.repository)["summary_generations"] == 1


def test_theme_replaced_during_the_call_is_discarded_stale(tmp_path):
    day = build_day(tmp_path)

    def move_a_story():
        # Membership changes: new fingerprints, new theme ids.
        deliveries = [day.story_ids["Tesla Q2 deliveries top estimates"]]
        robotaxi = [day.story_ids["Robotaxi expands"], day.story_ids["Tesla guidance"]]
        reconcile_theme_set(
            day.repository,
            [
                ("Deliveries", deliveries, day.items["deliveries"], "key-1"),
                (
                    "Robotaxi",
                    robotaxi,
                    day.items["robotaxi"] + day.items["guidance"],
                    "key-2",
                ),
            ],
            other=[day.story_ids["Weekend column"]],
            excluded=[day.story_ids["No text"]],
            story_count=len(day.story_ids),
            fingerprint_suffix="-moved",
        )

    outcome = ensure(day, MidCallClient(move_a_story))
    assert outcome.source == lifecycle.SOURCE_DISCARDED_STALE
    assert outcome.generation.detail == REFUSED_UNKNOWN_THEME
    assert table_counts(day.repository)["summary_artifacts"] == 0


def test_same_key_sequential_completions_yield_one_accepted_artifact(tmp_path):
    day = build_day(tmp_path)
    repository = day.repository
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    policy = resolve_generation_policy(EchoClient())
    # Two workers both paid for a generation of the same frozen input.
    first = generate_guarded_summary(
        generation_input, client=EchoClient(), policy=policy
    )
    second = generate_guarded_summary(
        generation_input, client=EchoClient(), policy=policy
    )
    assert first.accepted and second.accepted

    with open_run(repository, run_id="worker-a") as run:
        winner = repository.persist_summary_generation(
            run=run, result=first, generation_input=generation_input, policy=policy
        )
    with open_run(repository, run_id="worker-b") as run:
        loser = repository.persist_summary_generation(
            run=run, result=second, generation_input=generation_input, policy=policy
        )
    assert winner.outcome == SUMMARY_GENERATION_ACCEPTED
    assert loser.outcome == SUMMARY_GENERATION_DISCARDED_DUPLICATE
    assert loser.artifact_id == winner.artifact_id
    assert loser.provider_calls == 1  # the spend is still accounted
    counts = table_counts(repository)
    assert counts["summary_artifacts"] == 1
    assert counts["summary_generations"] == 2
    assert counts["summary_generation_attempts"] == 2
    assert current(day, NeverClient()).artifact.artifact_id == winner.artifact_id
    # Through the lifecycle the duplicate reports the winner as current.
    outcome = ensure(day, NeverClient())
    assert outcome.source == lifecycle.SOURCE_CACHE_HIT
    assert outcome.artifact.artifact_id == winner.artifact_id


def test_a_repeated_persist_under_one_run_is_idempotent(tmp_path):
    day = build_day(tmp_path)
    repository = day.repository
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    policy = resolve_generation_policy(EchoClient())
    result = generate_guarded_summary(
        generation_input, client=EchoClient(), policy=policy
    )
    with open_run(repository, run_id="worker-a") as run:
        first = repository.persist_summary_generation(
            run=run, result=result, generation_input=generation_input, policy=policy
        )
        again = repository.persist_summary_generation(
            run=run, result=result, generation_input=generation_input, policy=policy
        )
    assert again == first
    assert table_counts(repository)["summary_generations"] == 1
    assert table_counts(repository)["summary_generation_attempts"] == 1


def test_different_fingerprints_only_the_live_input_activates(tmp_path):
    day = build_day(tmp_path)
    repository = day.repository
    theme_id = day.theme_ids["Deliveries"]
    policy = resolve_generation_policy(EchoClient())
    stale_input = build_generation_input(day.population(), theme_id)
    stale_result = generate_guarded_summary(
        stale_input, client=EchoClient(), policy=policy
    )
    replay_theme_set(day, deliveries_key="key-1b")
    live_input = build_generation_input(day.population(), theme_id)
    assert live_input.input_fingerprint != stale_input.input_fingerprint
    live_result = generate_guarded_summary(
        live_input, client=EchoClient(), policy=policy
    )

    with open_run(repository, run_id="worker-stale") as run:
        stale = repository.persist_summary_generation(
            run=run, result=stale_result, generation_input=stale_input, policy=policy
        )
    with open_run(repository, run_id="worker-live") as run:
        live = repository.persist_summary_generation(
            run=run, result=live_result, generation_input=live_input, policy=policy
        )
    assert stale.outcome == SUMMARY_GENERATION_DISCARDED_STALE
    assert stale.detail == SUMMARY_DISCARD_INPUT_CHANGED
    assert live.outcome == SUMMARY_GENERATION_ACCEPTED
    assert table_counts(repository)["summary_artifacts"] == 1
    assert current(day, NeverClient()).artifact.artifact_id == live.artifact_id


def test_no_database_lock_is_held_during_the_provider_call(tmp_path):
    day = build_day(tmp_path)
    path = day.repository.database_path
    observed = {}

    def probe():
        # Another process could take the write lock right now...
        connection = sqlite3.connect(path, timeout=0.2)
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO source_state (source, last_checked_at) "
                "VALUES ('probe', '2026-07-23T12:00:00+00:00')"
            )
            connection.commit()
            observed["wrote"] = True
            # ...and no read transaction of ours is pinning the WAL either:
            # a TRUNCATE checkpoint reports busy while any reader is open.
            busy, _, _ = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            observed["readers_open"] = bool(busy)
        finally:
            connection.close()

    outcome = ensure(day, MidCallClient(probe))
    assert observed == {"wrote": True, "readers_open": False}
    assert outcome.source == lifecycle.SOURCE_GENERATED


# ----------------------------------------------------------------------
# Corruption (25-28)
# ----------------------------------------------------------------------


def test_missing_sentence_is_not_a_hit(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 1",
        (first.artifact.artifact_id,),
    )
    client = EchoClient()
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_GENERATED and len(client.prompts) == 1
    assert "invalid_sentence_count" in outcome.cache_rejection_codes


def test_malformed_citation_is_not_a_hit(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    corrupt(
        day,
        "INSERT INTO summary_sentence_citations "
        "(artifact_id, sentence_ordinal, position, story_id) VALUES (?, 1, 5, 999999)",
        (first.artifact.artifact_id,),
    )
    assert current(day, NeverClient()) is None
    client = EchoClient()
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_GENERATED and len(client.prompts) == 1
    assert "unknown_citation" in outcome.cache_rejection_codes

    # And a sentence stripped of every citation.
    corrupt(
        day,
        "DELETE FROM summary_sentence_citations WHERE artifact_id = ?",
        (outcome.artifact.artifact_id,),
    )
    assert current(day, NeverClient()) is None
    verdict = stored_verdict(day, NeverClient())
    assert REJECT_STRUCTURE in verdict.codes and "missing_citation" in verdict.codes


def test_successful_replacement_invalidates_the_corrupt_artifact_atomically(
    tmp_path,
):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        (first.artifact.artifact_id,),
    )
    outcome = ensure(day, EchoClient())
    assert outcome.source == lifecycle.SOURCE_GENERATED
    replacement = outcome.artifact
    assert replacement.artifact_id != first.artifact.artifact_id
    assert replacement.status == "accepted"
    assert (
        replacement.theme_id,
        replacement.input_fingerprint,
        replacement.policy_fingerprint,
    ) == (
        first.artifact.theme_id,
        first.artifact.input_fingerprint,
        first.artifact.policy_fingerprint,
    )
    history = {a.artifact_id: a for a in artifacts(day)}
    old = history[first.artifact.artifact_id]
    assert old.status == "invalidated"
    assert old.invalidated_at is not None
    assert old.invalidated_reason.startswith(SUMMARY_INVALIDATED_CORRUPT)
    assert "invalid_sentence_count" in old.invalidated_reason
    # Same transaction: the invalidation timestamp is the replacement's.
    assert old.invalidated_at == replacement.created_at
    assert outcome.generation.completed_at == replacement.created_at
    assert current(day, NeverClient()).artifact == replacement
    assert ensure(day, NeverClient()).source == lifecycle.SOURCE_CACHE_HIT


def test_failed_replacement_leaves_the_corrupt_artifact_unservable(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        (first.artifact.artifact_id,),
    )
    outcome = ensure(day, FailingClient())
    assert outcome.source == lifecycle.SOURCE_UNAVAILABLE
    assert outcome.artifact is None
    assert "invalid_sentence_count" in outcome.cache_rejection_codes
    assert current(day, NeverClient()) is None
    history = artifacts(day)
    assert len(history) == 1  # no fake replacement
    # Still 'accepted' in the ledger -- originally accepted, never
    # explicitly invalidated -- and still not current.
    assert history[0].status == "accepted"
    assert history[0].artifact_id == first.artifact.artifact_id


# ----------------------------------------------------------------------
# Reconciliation (29-31)
# ----------------------------------------------------------------------


def test_identical_theme_replay_preserves_the_current_artifact(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    report = replay_theme_set(day)
    assert report.counts["unchanged"] == 2 and report.counts["updated"] == 0
    assert day.population().themes[0].theme_id == day.theme_ids["Deliveries"]
    assert current(day, NeverClient()).artifact == first.artifact
    assert ensure(day, NeverClient()).source == lifecycle.SOURCE_CACHE_HIT
    # Reconciliation neither wrote nor deleted A3 rows, and the theme's own
    # summary column stayed the theme stage's.
    assert table_counts(day.repository)["summary_artifacts"] == 1
    assert day.repository.read.theme(day.theme_ids["Deliveries"])["summary"] is None


def test_membership_change_makes_the_old_artifact_non_current(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    old_theme_id = day.theme_ids["Deliveries"]
    deliveries = [day.story_ids["Tesla Q2 deliveries top estimates"]]
    robotaxi = [day.story_ids["Robotaxi expands"], day.story_ids["Tesla guidance"]]
    reconcile_theme_set(
        day.repository,
        [
            ("Deliveries", deliveries, day.items["deliveries"], "key-1"),
            (
                "Robotaxi",
                robotaxi,
                day.items["robotaxi"] + day.items["guidance"],
                "key-2",
            ),
        ],
        other=[day.story_ids["Weekend column"]],
        excluded=[day.story_ids["No text"]],
        story_count=len(day.story_ids),
        fingerprint_suffix="-moved",
    )
    population = day.population()
    new_ids = {theme.label: theme.theme_id for theme in population.themes}
    assert new_ids["Deliveries"] != old_theme_id  # a new theme row
    assert current(day, NeverClient(), theme_id=new_ids["Deliveries"]) is None
    # And the old theme id names nothing live any more.
    assert current(day, NeverClient(), theme_id=old_theme_id) is None
    # The old artifact survived reconciliation as history, under its old id.
    old_history = day.repository.read.summary_artifacts(
        TICKER, DAY, VERSION, theme_id=old_theme_id
    )
    assert [a.artifact_id for a in old_history] == [first.artifact.artifact_id]
    assert old_history[0].status == "accepted"
    # A generation for the new theme is a fresh artifact; the old id is
    # never reused.
    day.theme_ids.update(new_ids)
    outcome = ensure(day, EchoClient())
    assert outcome.source == lifecycle.SOURCE_GENERATED
    assert outcome.artifact.theme_id == new_ids["Deliveries"]


def test_disappeared_theme_is_not_current(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, EchoClient())
    clear_theme_set(day)
    assert current(day, NeverClient()) is None
    outcome = ensure(day, NeverClient())
    assert outcome.source == lifecycle.SOURCE_REFUSED
    assert outcome.refusal_code == REFUSED_NO_THEME_SET
    assert outcome.provider_calls == 0
    assert [a.artifact_id for a in artifacts(day)] == [first.artifact.artifact_id]


# ----------------------------------------------------------------------
# Accounting (32-40)
# ----------------------------------------------------------------------


def test_attempt_count_outcomes_and_codes_are_exact(tmp_path):
    day = build_day(tmp_path)
    client = BannedClient()
    outcome = ensure(day, client)
    generation = outcome.generation
    assert outcome.source == lifecycle.SOURCE_UNAVAILABLE
    assert generation.reason == "validation_exhausted"
    assert generation.provider_calls == 2 == len(client.prompts)
    assert generation.max_attempts == 2
    assert generation.accepted_attempt is None
    assert [a.attempt for a in generation.attempts] == [1, 2]
    assert [a.outcome for a in generation.attempts] == ["rejected", "rejected"]
    assert [a.outcome for a in outcome.result.attempts] == ["rejected", "rejected"]
    for stored, observed in zip(generation.attempts, outcome.result.attempts):
        assert stored.failures == tuple((f.code, f.detail) for f in observed.failures)
        assert [code for code, _ in stored.failures] == list(observed.validation_codes)
    assert all(code == "banned_language" for code, _ in generation.attempts[0].failures)
    assert any(
        detail.startswith("sentence 1: ")
        for _, detail in generation.attempts[0].failures
    )
    # Structural detail only: the offending copy itself is never stored.
    assert "buy" not in dump_all_rows(day.repository).lower()
    assert generation.attempts[0].error is None


def test_accepted_attempt_and_latency_are_exact(tmp_path):
    day = build_day(tmp_path)
    ticks = iter([10.0, 10.25, 20.0, 20.5])
    client = summary_input.EchoClient()
    # First answer trips validation; the second is fine.
    banned = BannedClient()

    class TwoStep(EchoClient):
        def __init__(self):
            super().__init__()
            self.n = 0

        def generate(self, *args):
            self.n += 1
            if self.n == 1:
                return banned.generate(*args)
            return client.generate(*args)

    outcome = ensure(day, TwoStep(), clock=lambda: next(ticks))
    generation = outcome.generation
    assert outcome.source == lifecycle.SOURCE_GENERATED
    assert generation.accepted_attempt == 2
    assert generation.provider_calls == 2
    assert [a.latency_ms for a in generation.attempts] == [250.0, 500.0]
    assert generation.total_latency_ms == 750.0
    assert all(a.latency_ms >= 0 for a in generation.attempts)


@pytest.mark.parametrize(
    "client_type, reason, outcomes",
    [
        (UnconfiguredClient, "provider_unconfigured", ["provider_unconfigured"]),
        (FailingClient, "provider_unavailable", ["provider_error", "provider_error"]),
        (BannedClient, "validation_exhausted", ["rejected", "rejected"]),
    ],
)
def test_unavailable_reasons_are_preserved_distinctly(
    tmp_path, client_type, reason, outcomes
):
    day = build_day(tmp_path)
    outcome = ensure(day, client_type())
    assert outcome.source == lifecycle.SOURCE_UNAVAILABLE
    assert outcome.generation.reason == reason
    assert [a.outcome for a in outcome.generation.attempts] == outcomes
    assert outcome.generation.artifact_id is None
    stored = generations(day)[0]
    assert stored.reason == reason
    assert stored.outcome == SUMMARY_GENERATION_UNAVAILABLE


def test_no_secret_reaches_any_summary_table(tmp_path):
    day = build_day(tmp_path)
    for client in (FailingClient(), UnconfiguredClient()):
        outcome = ensure(day, client)
        assert outcome.source == lifecycle.SOURCE_UNAVAILABLE
        assert all(a.error for a in outcome.generation.attempts)
    everything = dump_all_rows(day.repository)
    assert CANARY not in everything
    assert "[REDACTED]" in everything


# ----------------------------------------------------------------------
# Retry policy (41-44)
# ----------------------------------------------------------------------


def test_without_a_retry_policy_nothing_implicit_suppresses_a_call(tmp_path):
    day = build_day(tmp_path)
    first = FailingClient()
    assert ensure(day, first).source == lifecycle.SOURCE_UNAVAILABLE
    for _ in range(3):
        client = FailingClient()
        assert ensure(day, client).source == lifecycle.SOURCE_UNAVAILABLE
        assert client.calls == 2
    assert len(generations(day)) == 4
    assert RetryPolicy().suppression(generations(day), now=NOW) is None


def test_explicit_cooldown_suppresses_calls_while_active(tmp_path, monkeypatch):
    clock = ManualClock()
    day = clocked_day(tmp_path, monkeypatch, clock)
    retry = RetryPolicy(cooldown=timedelta(minutes=30))
    assert ensure(day, FailingClient(), retry=retry).source == (
        lifecycle.SOURCE_UNAVAILABLE
    )
    before = table_counts(day.repository)

    clock.advance(timedelta(minutes=29))
    cooled = ensure(day, NeverClient(), retry=retry)
    assert cooled.source == lifecycle.SOURCE_COOLDOWN
    assert cooled.provider_calls == 0 and cooled.artifact is None
    assert table_counts(day.repository) == before

    clock.advance(timedelta(minutes=2))
    client = EchoClient()
    assert ensure(day, client, retry=retry).source == lifecycle.SOURCE_GENERATED
    assert len(client.prompts) == 1
    # A cooldown never outranks a cache hit.
    assert ensure(day, NeverClient(), retry=retry).source == (
        lifecycle.SOURCE_CACHE_HIT
    )


def test_explicit_cap_suppresses_calls_once_exhausted(tmp_path):
    day = build_day(tmp_path)
    retry = RetryPolicy(max_generations=2)
    assert ensure(day, FailingClient(), retry=retry).source == (
        lifecycle.SOURCE_UNAVAILABLE
    )
    assert ensure(day, FailingClient(), retry=retry).source == (
        lifecycle.SOURCE_UNAVAILABLE
    )
    exhausted = ensure(day, NeverClient(), retry=retry)
    assert exhausted.source == lifecycle.SOURCE_EXHAUSTED
    assert exhausted.provider_calls == 0
    assert len(generations(day)) == 2
    # The cap is per exact key: a changed input starts afresh.
    replay_theme_set(day, deliveries_key="key-1b")
    client = EchoClient()
    assert ensure(day, client, retry=retry).source == lifecycle.SOURCE_GENERATED
    # And another caller with no policy is not bound by this one's cap.
    assert ensure(day, EchoClient(), theme="Robotaxi").source == (
        lifecycle.SOURCE_GENERATED
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cooldown": 60},
        {"cooldown": timedelta(seconds=-1)},
        {"max_generations": 0},
        {"max_generations": True},
        {"max_generations": "3"},
    ],
)
def test_retry_policy_rejects_malformed_values(kwargs):
    with pytest.raises(Phase0ValidationError):
        RetryPolicy(**kwargs)


def test_readers_never_call_the_provider(tmp_path):
    day = build_day(tmp_path)
    ensure(day, FailingClient())  # an unavailable key on record
    # Every read surface answers without a client at all.
    assert current(day, NeverClient()) is None
    reader = day.repository.read
    assert reader.summary_artifacts(TICKER, DAY, VERSION) == []
    assert len(reader.summary_generations(TICKER, DAY, VERSION)) == 1
    assert (
        reader.summary_artifact(day.theme_ids["Deliveries"], "a" * 64, "b" * 64) is None
    )
    assert reader.count("summary_generations") == 1


# ----------------------------------------------------------------------
# Atomicity (45-47)
# ----------------------------------------------------------------------


def test_a_citation_insert_failure_rolls_back_the_artifact(tmp_path):
    day = build_day(tmp_path)
    corrupt(
        day,
        "CREATE TRIGGER injected BEFORE INSERT ON summary_sentence_citations "
        "WHEN NEW.sentence_ordinal = 2 BEGIN SELECT RAISE(ABORT, 'injected'); END",
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        ensure(day, EchoClient())
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}
    assert day.repository.read.run_log_rows(stage=STAGE)[-1]["status"] == "failed"
    corrupt(day, "DROP TRIGGER injected")
    assert ensure(day, EchoClient()).source == lifecycle.SOURCE_GENERATED


def test_an_attempt_insert_failure_rolls_back_generation_and_artifact(tmp_path):
    day = build_day(tmp_path)
    corrupt(
        day,
        "CREATE TRIGGER injected BEFORE INSERT ON summary_generation_attempts "
        "BEGIN SELECT RAISE(ABORT, 'injected'); END",
    )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        ensure(day, EchoClient())
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}
    assert current(day, NeverClient()) is None


def test_a_logged_mutation_failure_cannot_leave_a_current_artifact(
    tmp_path, monkeypatch
):
    day = build_day(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("run log unavailable")

    monkeypatch.setattr(Phase0Repository, "_write_final_run_log", boom)
    with pytest.raises(RuntimeError, match="run log unavailable"):
        ensure(day, EchoClient())
    monkeypatch.undo()
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}
    assert current(day, NeverClient()) is None


def test_persist_requires_the_run_to_cover_the_partition(tmp_path):
    day = build_day(tmp_path)
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    policy = resolve_generation_policy(EchoClient())
    result = generate_guarded_summary(
        generation_input, client=EchoClient(), policy=policy
    )
    with day.repository.stage_run(
        run_id="elsewhere",
        stage=STAGE,
        trading_day="2026-07-22",
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        with pytest.raises(Phase0RunContextError):
            day.repository.persist_summary_generation(
                run=run,
                result=result,
                generation_input=generation_input,
                policy=policy,
            )
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}


# ----------------------------------------------------------------------
# Reads (48-52)
# ----------------------------------------------------------------------


def test_current_excludes_stale_unavailable_and_invalidated(tmp_path):
    day = build_day(tmp_path)
    accepted = ensure(day, EchoClient())
    assert current(day, NeverClient()).artifact == accepted.artifact

    # Invalidated: corrupt, then replaced.
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        (accepted.artifact.artifact_id,),
    )
    replaced = ensure(day, EchoClient())
    assert current(day, NeverClient()).artifact == replaced.artifact
    invalidated = {a.artifact_id: a for a in artifacts(day)}[
        accepted.artifact.artifact_id
    ]
    assert invalidated.status == "invalidated"

    # Stale: the input moves on.
    replay_theme_set(day, deliveries_key="key-1b")
    assert current(day, NeverClient()) is None
    # Unavailable under the new input: still nothing current.
    assert ensure(day, FailingClient()).source == lifecycle.SOURCE_UNAVAILABLE
    assert current(day, NeverClient()) is None
    # History keeps all of it, newest first, statuses intact.
    history = artifacts(day)
    assert [a.artifact_id for a in history] == [
        replaced.artifact.artifact_id,
        accepted.artifact.artifact_id,
    ]
    assert [a.status for a in history] == ["accepted", "invalidated"]
    assert [g.outcome for g in generations(day)] == [
        SUMMARY_GENERATION_UNAVAILABLE,
        SUMMARY_GENERATION_ACCEPTED,
        SUMMARY_GENERATION_ACCEPTED,
    ]
    # The unhealthy-population read is None, not an exception.
    clear_theme_set(day)
    assert current(day, NeverClient()) is None


def test_history_readers_cover_the_partition_and_the_theme(tmp_path):
    day = build_day(tmp_path)
    deliveries = ensure(day, EchoClient())
    robotaxi = ensure(day, EchoClient(), theme="Robotaxi")
    reader = day.repository.read
    assert {a.artifact_id for a in reader.summary_artifacts(TICKER, DAY, VERSION)} == {
        deliveries.artifact.artifact_id,
        robotaxi.artifact.artifact_id,
    }
    assert [a.artifact_id for a in artifacts(day, "Robotaxi")] == [
        robotaxi.artifact.artifact_id
    ]
    assert reader.summary_artifacts("NVDA", DAY, VERSION) == []
    assert reader.summary_artifacts(TICKER, DAY, "v2") == []
    assert len(reader.summary_generations(TICKER, DAY, VERSION)) == 2
    assert [g.run_id for g in generations(day, "Robotaxi")] == [
        robotaxi.generation.run_id
    ]
    assert generations(day)[0].artifact_id == deliveries.artifact.artifact_id


def test_current_citations_resolve_to_frozen_evidence_provenance(tmp_path):
    day = build_day(tmp_path)
    ensure(day, EchoClient())
    shown = current(day, NeverClient())
    assert shown is not None
    first_story = day.story_ids["Tesla Q2 deliveries top estimates"]
    sentence = shown.artifact.sentences[0]
    assert [c.story_id for c in sentence.citations] == [first_story]
    evidence = shown.evidence_for(first_story)
    assert evidence.citation_id == citation_id_for(first_story)
    assert evidence.persisted_story_id == first_story
    assert evidence.title == "Tesla Q2 deliveries top estimates"
    assert evidence.outlet == "Reuters"
    assert evidence.raw_item_ids == tuple(day.items["deliveries"])
    assert evidence.urls == tuple(
        f"https://{outlet}.example/{item}"
        for outlet, item in zip(("reuters", "cnbc"), day.items["deliveries"])
    )
    # Every cited story resolves; nothing outside the input does.
    for story_id in shown.artifact.story_ids:
        assert shown.evidence_for(story_id).persisted_story_id == story_id
    with pytest.raises(GuardedSummaryError):
        shown.evidence_for(day.story_ids["Weekend column"])
    # The outcome exposes the same current view.
    hit = ensure(day, NeverClient())
    assert hit.current is not None and hit.current.artifact == shown.artifact


# ----------------------------------------------------------------------
# Boundary (53-58)
# ----------------------------------------------------------------------


def test_lifecycle_imports_no_backend_frontend_agent_or_provider_sdk():
    probe = (
        "import sys, phase0.summary_lifecycle; "
        "sys.exit(0 if not any(m.split('.')[0] in ('backend', 'frontend', "
        "'langchain', 'langgraph', 'google') for m in sys.modules) else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    tree = ast.parse((ROOT / "phase0" / "summary_lifecycle.py").read_text("utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {"backend", "frontend", "langchain", "langgraph", "google", "fastapi"}
    assert not {name for name in imported if name.split(".")[0] in forbidden}
    assert "nlp.eval.review" not in imported  # no A4b
    source = (ROOT / "phase0" / "summary_lifecycle.py").read_text("utf-8")
    assert "GeminiClient(" not in source  # never constructs a provider client


def test_live_registration_goes_through_the_runner_only():
    """A3b registers summaries through ``phase0.summary_runner`` alone.

    The lifecycle itself is still never imported by the orchestrator, is
    never driven by the coordinator, and replay still does not summarize.
    """

    import pipeline

    assert pipeline.DOWNSTREAM_STAGES == (
        pipeline.intelligence_stage,
        pipeline.summaries_stage,
    )
    assert "summarization" in pipeline.replay_capabilities()["unsupported"]
    source = (ROOT / "pipeline.py").read_text("utf-8")
    assert "summary_lifecycle" not in source
    assert "ensure_summary" not in source
    coordinator = (ROOT / "phase0" / "coordinator.py").read_text("utf-8")
    assert "summary" not in coordinator.lower()


def test_no_semantic_faithfulness_claim(tmp_path):
    day = build_day(tmp_path)
    outcome = ensure(day, EchoClient())
    assert "semantic_faithfulness_not_established" in outcome.artifact.guarantee
    assert not any(
        hasattr(outcome.artifact, name)
        for name in ("faithful", "faithfulness", "supported", "entailed")
    )
    assert not any(
        hasattr(SummaryLifecycleOutcome, name)
        for name in ("faithful", "faithfulness", "supported", "entailed")
    )
    with day.repository.admin.connect_writable() as connection:
        columns = {
            row["name"]
            for table in SUMMARY_TABLES
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
    assert not {c for c in columns if "faith" in c or "support" in c}


def test_a2_alone_still_writes_nothing(tmp_path):
    day = build_day(tmp_path)
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    result = generate_guarded_summary(generation_input, client=EchoClient())
    assert result.accepted
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}
    assert day.repository.read.run_log_rows(stage=STAGE) == []


# ----------------------------------------------------------------------
# Repository-specific
# ----------------------------------------------------------------------


def test_theme_stage_columns_are_never_written_by_the_lifecycle(tmp_path):
    day = build_day(tmp_path)
    theme_id = day.theme_ids["Deliveries"]
    before = day.repository.read.theme(theme_id)
    ensure(day, EchoClient())
    ensure(day, NeverClient())
    ensure(day, FailingClient(), theme="Robotaxi")
    after = day.repository.read.theme(theme_id)
    assert after == before
    assert after["summary"] is None and after["status"] == "ready"
    with day.repository.admin.connect_writable() as connection:
        theme_citations = connection.execute(
            "SELECT COUNT(*) FROM theme_citations WHERE theme_id = ?", (theme_id,)
        ).fetchone()[0]
    assert theme_citations == len(day.items["deliveries"] + day.items["guidance"])


def test_a_refused_population_makes_no_call_and_writes_nothing(tmp_path):
    day = build_day(tmp_path, story_count=99)  # recorded count disagrees
    outcome = ensure(day, NeverClient())
    assert outcome.source == lifecycle.SOURCE_REFUSED
    assert outcome.refusal_code == "source_story_count_mismatch"
    assert outcome.provider_calls == 0 and outcome.current is None
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}


def test_run_log_counts_describe_the_generation(tmp_path):
    day = build_day(tmp_path)
    ensure(day, EchoClient(), run_id="acct-generated")
    ensure(day, FailingClient(), theme="Robotaxi", run_id="acct-unavailable")
    rows = {
        row["run_id"]: dict(row, counts=json.loads(row["counts"]))
        for row in day.repository.read.run_log_rows(stage=STAGE)
    }
    generated = rows["acct-generated"]
    assert generated["counts"]["summary_generations"] == 1
    assert generated["counts"]["summary_provider_calls"] == 1
    assert generated["counts"]["summary_artifacts_inserted"] == 1
    assert generated["counts"]["summary_accepted"] == 1
    assert generated["success_count"] == 1
    unavailable = rows["acct-unavailable"]
    assert unavailable["counts"]["summary_provider_calls"] == 2
    assert unavailable["counts"]["summary_unavailable"] == 1
    assert unavailable["counts"]["summary_artifacts_inserted"] == 0
    assert unavailable["status"] == "degraded"


def test_persist_refuses_a_result_from_another_input(tmp_path):
    day = build_day(tmp_path)
    policy = resolve_generation_policy(EchoClient())
    deliveries = build_generation_input(day.population(), day.theme_ids["Deliveries"])
    robotaxi = build_generation_input(day.population(), day.theme_ids["Robotaxi"])
    result = generate_guarded_summary(deliveries, client=EchoClient(), policy=policy)
    with open_run(day.repository) as run:
        with pytest.raises(Phase0ValidationError, match="not generated from"):
            day.repository.persist_summary_generation(
                run=run, result=result, generation_input=robotaxi, policy=policy
            )
    assert table_counts(day.repository) == {table: 0 for table in SUMMARY_TABLES}


# ----------------------------------------------------------------------
# The A2 policy helper the lifecycle depends on
# ----------------------------------------------------------------------


def test_resolved_policy_is_exactly_what_a2_resolves_for_itself(tmp_path):
    day = build_day(tmp_path)
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )

    class Capped(EchoClient):
        max_output_tokens = 512

    policy = resolve_generation_policy(Capped(), max_attempts=1)
    assert policy.model == "fake-model"
    assert policy.max_attempts == 1
    assert policy.max_output_tokens == 512
    assert policy.rules == tuple(tuple(rule) for rule in load_copy_rules())
    assert policy.fingerprint == guarded.compute_policy_fingerprint(
        model="fake-model",
        max_attempts=1,
        rules=load_copy_rules(),
        max_output_tokens=512,
    )
    # Resolving is deterministic, and the two paths through A2 agree.
    assert resolve_generation_policy(Capped(), max_attempts=1) == policy
    implicit = generate_guarded_summary(
        generation_input, client=Capped(), max_attempts=1
    )
    explicit = generate_guarded_summary(
        generation_input, client=Capped(), max_attempts=1, policy=policy
    )
    assert implicit.policy_fingerprint == explicit.policy_fingerprint
    assert explicit.policy_fingerprint == policy.fingerprint


def test_a_resolved_policy_fixes_rules_and_attempts(tmp_path):
    day = build_day(tmp_path)
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    policy = resolve_generation_policy(EchoClient(), max_attempts=1)
    with pytest.raises(GuardedSummaryError, match="already fixes"):
        generate_guarded_summary(
            generation_input, client=EchoClient(), max_attempts=2, policy=policy
        )
    with pytest.raises(GuardedSummaryError, match="already fixes"):
        generate_guarded_summary(
            generation_input,
            client=EchoClient(),
            max_attempts=1,
            rules=load_copy_rules(),
            policy=policy,
        )
    with pytest.raises(GuardedSummaryError, match="must be a GenerationPolicy"):
        generate_guarded_summary(
            generation_input, client=EchoClient(), policy=policy.fingerprint
        )
    for bad in ({"max_attempts": 3}, {"max_attempts": True}):
        with pytest.raises(ValueError):
            resolve_generation_policy(EchoClient(), **bad)
    with pytest.raises(GuardedSummaryError):
        GenerationPolicy(
            model="m",
            max_attempts=2,
            rules=(("a", "literal", "b"),),
            temperature=0.1,
            max_output_tokens=0,
            fingerprint="x" * 64,
        )


# ----------------------------------------------------------------------
# Codex finding 1: currentness comes from the live database, never from a
# snapshot the caller kept
# ----------------------------------------------------------------------


def test_public_lifecycle_apis_take_no_population():
    import inspect

    for function in (current_summary_artifact, ensure_summary):
        parameters = inspect.signature(function).parameters
        assert "population" not in parameters, function.__name__
        assert {"ticker", "trading_day", "pipeline_version", "theme_id"} <= set(
            parameters
        )


def test_a_retained_snapshot_cannot_make_a_deleted_theme_current(tmp_path):
    day = build_day(tmp_path)
    accepted = ensure(day, EchoClient())
    theme_id = day.theme_ids["Deliveries"]
    old_population = day.population()
    old_input = build_generation_input(old_population, theme_id)
    policy = resolve_generation_policy(NeverClient())

    clear_theme_set(day)  # the ordinary logged repository API

    # The retained snapshot still projects, and the stored row is still a
    # valid artifact *for that old input* -- which is exactly why no public
    # API accepts either.
    assert validate_persisted_artifact(accepted.artifact, old_input, policy).valid
    assert current(day, NeverClient()) is None
    outcome = ensure(day, NeverClient())
    assert outcome.source == lifecycle.SOURCE_REFUSED
    assert outcome.refusal_code == REFUSED_NO_THEME_SET
    assert outcome.provider_calls == 0
    assert outcome.artifact is None and outcome.current is None
    # History is intact; nothing was served.
    assert [a.artifact_id for a in artifacts(day)] == [accepted.artifact.artifact_id]


def test_a_retained_snapshot_cannot_make_a_rekeyed_theme_current(tmp_path):
    day = build_day(tmp_path)
    accepted = ensure(day, EchoClient())
    theme_id = day.theme_ids["Deliveries"]
    old_input = build_generation_input(day.population(), theme_id)
    policy = resolve_generation_policy(NeverClient())

    replay_theme_set(day, deliveries_key="key-1b")  # same theme id, new key

    assert validate_persisted_artifact(accepted.artifact, old_input, policy).valid
    assert current(day, NeverClient()) is None
    client = EchoClient()
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_GENERATED  # never a cache hit
    assert len(client.prompts) == 1
    assert outcome.artifact.artifact_id != accepted.artifact.artifact_id
    assert outcome.input_fingerprint != accepted.input_fingerprint


def test_a_retained_snapshot_cannot_make_a_moved_theme_current(tmp_path):
    day = build_day(tmp_path)
    accepted = ensure(day, EchoClient())
    old_theme_id = day.theme_ids["Deliveries"]
    old_input = build_generation_input(day.population(), old_theme_id)
    policy = resolve_generation_policy(NeverClient())
    deliveries = [day.story_ids["Tesla Q2 deliveries top estimates"]]
    robotaxi = [day.story_ids["Robotaxi expands"], day.story_ids["Tesla guidance"]]
    reconcile_theme_set(
        day.repository,
        [
            ("Deliveries", deliveries, day.items["deliveries"], "key-1"),
            (
                "Robotaxi",
                robotaxi,
                day.items["robotaxi"] + day.items["guidance"],
                "key-2",
            ),
        ],
        other=[day.story_ids["Weekend column"]],
        excluded=[day.story_ids["No text"]],
        story_count=len(day.story_ids),
        fingerprint_suffix="-moved",
    )
    assert validate_persisted_artifact(accepted.artifact, old_input, policy).valid
    assert current(day, NeverClient(), theme_id=old_theme_id) is None
    outcome = ensure(day, NeverClient())  # asks for the old theme id
    assert outcome.source == lifecycle.SOURCE_REFUSED
    assert outcome.refusal_code == REFUSED_UNKNOWN_THEME
    assert outcome.provider_calls == 0


def test_a_recreated_partition_cannot_revive_an_old_artifact(tmp_path):
    day = build_day(tmp_path)
    accepted = ensure(day, EchoClient())
    old_theme_id = day.theme_ids["Deliveries"]
    clear_theme_set(day)
    replay_theme_set(day)  # the same logical theme set, written again
    population = day.population()
    new_ids = {theme.label: theme.theme_id for theme in population.themes}
    assert new_ids["Deliveries"] != old_theme_id  # AUTOINCREMENT: never reused
    assert current(day, NeverClient(), theme_id=old_theme_id) is None
    assert current(day, NeverClient(), theme_id=new_ids["Deliveries"]) is None
    day.theme_ids.update(new_ids)
    client = EchoClient()
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_GENERATED and len(client.prompts) == 1
    assert outcome.artifact.theme_id == new_ids["Deliveries"]
    assert {a.artifact_id for a in artifacts(day)} == {outcome.artifact.artifact_id}
    assert [
        a.artifact_id
        for a in day.repository.read.summary_artifacts(
            TICKER, DAY, VERSION, theme_id=old_theme_id
        )
    ] == [accepted.artifact.artifact_id]


# ----------------------------------------------------------------------
# Codex finding 2: a stored artifact is proved whole before it is reused
# ----------------------------------------------------------------------


def test_a_sealed_artifact_refuses_ordinary_child_sql(tmp_path):
    day = build_day(tmp_path)
    artifact_id = ensure(day, EchoClient()).artifact.artifact_id
    sealed_sql_is_refused(
        day, "DELETE FROM summary_sentences WHERE artifact_id = ?", (artifact_id,)
    )
    sealed_sql_is_refused(
        day,
        "DELETE FROM summary_sentence_citations WHERE artifact_id = ?",
        (artifact_id,),
    )
    sealed_sql_is_refused(
        day,
        "INSERT INTO summary_sentences (artifact_id, ordinal, text) "
        "VALUES (?, 3, 'Added.')",
        (artifact_id,),
    )
    sealed_sql_is_refused(
        day,
        "INSERT INTO summary_sentence_citations "
        "(artifact_id, sentence_ordinal, position, story_id) VALUES (?, 1, 9, 999)",
        (artifact_id,),
    )
    # The parent is held by its accepted generation row (RESTRICT) before
    # the child triggers would even fire.
    with day.repository.admin.connect_writable() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY|sealed"):
            connection.execute(
                "DELETE FROM summary_artifacts WHERE id = ?", (artifact_id,)
            )
    assert stored_verdict(day, NeverClient()).valid


def test_trailing_sentence_deletion_is_detected_by_the_digest(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, OrderedClient())  # three sentences
    assert len(first.artifact.sentences) == 3
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 3",
        (first.artifact.artifact_id,),
    )
    verdict = stored_verdict(day, NeverClient())
    # Two well-ordered sentences satisfy A2 and the structural checks;
    # only the digest knows a third one existed.
    assert verdict.codes == (REJECT_DIGEST,)
    assert current(day, NeverClient()) is None
    client = OrderedClient()
    outcome = ensure(day, client)
    assert outcome.source == lifecycle.SOURCE_GENERATED and len(client.prompts) == 1
    assert outcome.cache_rejection_codes == (REJECT_DIGEST,)
    history = {a.artifact_id: a for a in artifacts(day)}
    assert history[first.artifact.artifact_id].status == "invalidated"
    assert REJECT_DIGEST in history[first.artifact.artifact_id].invalidated_reason


def test_middle_sentence_deletion_is_detected(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, OrderedClient())
    corrupt(
        day,
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        (first.artifact.artifact_id,),
    )
    verdict = stored_verdict(day, NeverClient())
    assert REJECT_STRUCTURE in verdict.codes and REJECT_DIGEST in verdict.codes
    assert current(day, NeverClient()) is None


def test_citation_position_deletion_is_detected(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, OrderedClient())  # sentence 1 cites two stories
    corrupt(
        day,
        "DELETE FROM summary_sentence_citations WHERE artifact_id = ? "
        "AND sentence_ordinal = 1 AND position = 0",
        (first.artifact.artifact_id,),
    )
    remaining = day.repository.read.summary_artifact(
        first.artifact.theme_id,
        first.artifact.input_fingerprint,
        first.artifact.policy_fingerprint,
    )
    assert [c.position for c in remaining.sentences[0].citations] == [1]
    verdict = stored_verdict(day, NeverClient())
    assert REJECT_STRUCTURE in verdict.codes and REJECT_DIGEST in verdict.codes
    assert current(day, NeverClient()) is None


def test_citation_reorder_is_detected_by_the_digest(tmp_path):
    day = build_day(tmp_path)
    first = ensure(day, OrderedClient())
    artifact_id = first.artifact.artifact_id
    original = [(c.position, c.story_id) for c in first.artifact.sentences[0].citations]
    assert len(original) == 2
    corrupt(
        day,
        "DELETE FROM summary_sentence_citations WHERE artifact_id = ? "
        "AND sentence_ordinal = 1",
        (artifact_id,),
    )
    for position, (_, story_id) in enumerate(reversed(original)):
        corrupt(
            day,
            "INSERT INTO summary_sentence_citations "
            "(artifact_id, sentence_ordinal, position, story_id) VALUES (?, 1, ?, ?)",
            (artifact_id, position, story_id),
        )
    swapped = day.repository.read.summary_artifact(
        first.artifact.theme_id,
        first.artifact.input_fingerprint,
        first.artifact.policy_fingerprint,
    )
    assert [c.story_id for c in swapped.sentences[0].citations] == [
        story_id for _, story_id in reversed(original)
    ]
    verdict = stored_verdict(day, NeverClient())
    # Well-formed positions, distinct known stories: only the digest sees it.
    assert verdict.codes == (REJECT_DIGEST,)
    assert current(day, NeverClient()) is None


def _forge_artifact(day: Day, client, *, digest=None, **identity) -> int:
    """Insert an artifact under the live key with forged identity columns.

    The digest defaults to the one the forged content honestly has, so the
    identity check is what is under test; pass ``digest`` to test the
    digest check instead.
    """

    generation_input = live_input(day)
    policy = resolve_generation_policy(client)
    theme = generation_input.theme
    story_id = generation_input.evidence[0].persisted_story_id
    fields = {
        "ticker": TICKER,
        "trading_day": DAY,
        "pipeline_version": VERSION,
        "theme_id": theme.theme_id,
        "theme_key": theme.theme_key,
        "input_fingerprint": generation_input.input_fingerprint,
        "policy_fingerprint": policy.fingerprint,
        "citation_convention": guarded.CITATION_CONVENTION,
        "prompt_version": guarded.PROMPT_VERSION,
        "model": policy.model,
        "label": "Forged coverage",
        "guarantee": "structural_grounding_and_copy_policy_only",
    }
    fields.update(identity)
    sentences = [
        (1, "Coverage one.", [(0, story_id)]),
        (2, "Coverage two.", [(0, story_id)]),
    ]
    content_digest = digest or summary_artifact_digest(**fields, sentences=sentences)
    with day.repository.admin.connect_writable() as connection:
        columns = ", ".join(list(fields) + ["content_digest", "status", "created_at"])
        placeholders = ", ".join("?" for _ in range(len(fields) + 3))
        cursor = connection.execute(
            f"INSERT INTO summary_artifacts ({columns}) VALUES ({placeholders})",
            (*fields.values(), content_digest, "accepted", "2026-07-23T12:00:00+00:00"),
        )
        artifact_id = int(cursor.lastrowid)
        for ordinal, text, citations in sentences:
            connection.execute(
                "INSERT INTO summary_sentences (artifact_id, ordinal, text) "
                "VALUES (?, ?, ?)",
                (artifact_id, ordinal, text),
            )
            for position, cited in citations:
                connection.execute(
                    "INSERT INTO summary_sentence_citations "
                    "(artifact_id, sentence_ordinal, position, story_id) "
                    "VALUES (?, ?, ?, ?)",
                    (artifact_id, ordinal, position, cited),
                )
    return artifact_id


@pytest.mark.parametrize(
    "identity",
    [
        {"ticker": "NVDA"},
        {"trading_day": "2026-07-22"},
        {"pipeline_version": "v2"},
        {"theme_key": "some-other-key"},
        {"model": "other-model"},
        {"prompt_version": "a2.guarded.v0"},
        {"citation_convention": "persisted_story_id.v0"},
    ],
)
def test_a_forged_identity_column_is_detected(tmp_path, identity):
    day = build_day(tmp_path)
    forged = _forge_artifact(day, NeverClient(), **identity)
    # Found by its lookup key, refused on its stored identity.
    stored = day.repository.read.summary_artifact(
        day.theme_ids["Deliveries"],
        live_input(day).input_fingerprint,
        resolve_generation_policy(NeverClient()).fingerprint,
    )
    assert stored is not None and stored.artifact_id == forged
    verdict = stored_verdict(day, NeverClient())
    assert verdict.codes == (REJECT_IDENTITY,)
    assert current(day, NeverClient()) is None
    # A replacement generation invalidates the forgery and takes the key.
    outcome = ensure(day, EchoClient())
    assert outcome.source == lifecycle.SOURCE_GENERATED
    assert outcome.cache_rejection_codes == (REJECT_IDENTITY,)
    history = {
        a.artifact_id: a
        for a in day.repository.read.summary_artifacts(TICKER, DAY, VERSION)
    }
    if {"ticker", "trading_day", "pipeline_version"} & set(identity):
        # Forged partition columns keep it out of this partition's history...
        assert forged not in history
    else:
        assert history[forged].status == "invalidated"
        assert REJECT_IDENTITY in history[forged].invalidated_reason
    # ...but never out of the exact-key reader, where it is now invalidated.
    assert (
        day.repository.read.summary_artifact(
            day.theme_ids["Deliveries"],
            live_input(day).input_fingerprint,
            resolve_generation_policy(NeverClient()).fingerprint,
        ).artifact_id
        == outcome.artifact.artifact_id
    )


def test_a_digest_mismatch_alone_is_detected(tmp_path):
    day = build_day(tmp_path)
    _forge_artifact(day, NeverClient(), digest="f" * 64)
    verdict = stored_verdict(day, NeverClient())
    assert verdict.codes == (REJECT_DIGEST,)
    assert current(day, NeverClient()) is None
    # A forgery with an honest digest and honest identity is, structurally,
    # a valid artifact: the digest and identity checks are what they claim
    # to be, no more.
    day2 = build_day(tmp_path / "second")
    _forge_artifact(day2, NeverClient())
    assert stored_verdict(day2, NeverClient()).valid


def test_the_digest_covers_exactly_the_documented_material(tmp_path):
    day = build_day(tmp_path)
    artifact = ensure(day, OrderedClient()).artifact
    assert summary_artifact_digest_of(artifact) == artifact.content_digest
    # Lifecycle bookkeeping is outside the digest...
    bookkeeping = dataclasses.replace(
        artifact,
        artifact_id=999,
        status="invalidated",
        created_at="2027-01-01T00:00:00+00:00",
        invalidated_at="2027-01-01T00:00:00+00:00",
        invalidated_reason="x",
    )
    assert summary_artifact_digest_of(bookkeeping) == artifact.content_digest
    # ...and every content and identity field is inside it.
    for change in (
        {"label": "Other"},
        {"guarantee": "other"},
        {"ticker": "NVDA"},
        {"trading_day": "2026-07-22"},
        {"pipeline_version": "v2"},
        {"theme_id": artifact.theme_id + 1},
        {"theme_key": "k2"},
        {"input_fingerprint": "1" * 64},
        {"policy_fingerprint": "2" * 64},
        {"citation_convention": "c"},
        {"prompt_version": "p"},
        {"model": "m"},
        {"sentences": artifact.sentences[:2]},
        {"sentences": (artifact.sentences[0], artifact.sentences[2])},
        {
            "sentences": (dataclasses.replace(artifact.sentences[0], text="Edited."),)
            + artifact.sentences[1:]
        },
        {
            "sentences": (
                dataclasses.replace(
                    artifact.sentences[0],
                    citations=tuple(reversed(artifact.sentences[0].citations)),
                ),
            )
            + artifact.sentences[1:]
        },
        {
            "sentences": (
                dataclasses.replace(
                    artifact.sentences[0],
                    citations=artifact.sentences[0].citations[1:],
                ),
            )
            + artifact.sentences[1:]
        },
    ):
        changed = dataclasses.replace(artifact, **change)
        assert summary_artifact_digest_of(changed) != artifact.content_digest, change


def test_a_valid_duplicate_holder_is_a_duplicate_not_a_corruption(tmp_path):
    day = build_day(tmp_path)
    repository = day.repository
    generation_input = live_input(day)
    policy = resolve_generation_policy(EchoClient())
    first = generate_guarded_summary(
        generation_input, client=EchoClient(), policy=policy
    )
    second = generate_guarded_summary(
        generation_input, client=EchoClient(), policy=policy
    )
    with open_run(repository, run_id="w1") as run:
        winner = repository.persist_summary_generation(
            run=run, result=first, generation_input=generation_input, policy=policy
        )
    with open_run(repository, run_id="w2") as run:
        duplicate = repository.persist_summary_generation(
            run=run, result=second, generation_input=generation_input, policy=policy
        )
    assert duplicate.outcome == SUMMARY_GENERATION_DISCARDED_DUPLICATE
    assert duplicate.artifact_id == winner.artifact_id
    assert [a.status for a in artifacts(day)] == [SUMMARY_ARTIFACT_ACCEPTED]
    assert stored_verdict(day, NeverClient()).valid


# ----------------------------------------------------------------------
# Codex finding 3: an incoming accepted result is re-proved at the boundary
# ----------------------------------------------------------------------


def _real_result(day: Day, client=None, policy=None):
    generation_input = live_input(day)
    policy = policy or resolve_generation_policy(EchoClient())
    result = generate_guarded_summary(
        generation_input, client=client or EchoClient(), policy=policy
    )
    return generation_input, policy, result


def _persist_refused(day: Day, result, generation_input, policy, match: str):
    before = table_counts(day.repository)
    with open_run(day.repository, run_id=f"refused-{next(_RUN_IDS)}") as run:
        with pytest.raises(Phase0ValidationError, match=match):
            day.repository.persist_summary_generation(
                run=run, result=result, generation_input=generation_input, policy=policy
            )
        assert run.state == "terminal_failed"
    assert table_counts(day.repository) == before
    assert day.repository.read.run_log_rows(stage=STAGE)[-1]["status"] == "failed"


def _with_sentences(result, sentences):
    return dataclasses.replace(
        result, summary=ThemeSummary(label=result.summary.label, sentences=sentences)
    )


def _renumbered(result, numbers):
    attempts = tuple(
        dataclasses.replace(attempt, attempt=number)
        for attempt, number in zip(result.attempts, numbers)
    )
    return dataclasses.replace(result, attempts=attempts)


def _attempt(number, outcome, **fields):
    return AttemptRecord(attempt=number, outcome=outcome, **fields)


def test_a_mutated_accepted_summary_is_refused_atomically(tmp_path):
    day = build_day(tmp_path)
    generation_input, policy, result = _real_result(day)
    ids = list(result.summary.sentences[0].citation_ids)
    banned = _with_sentences(
        result,
        [
            Sentence(text="You should buy Tesla.", citation_ids=ids),
            Sentence(text="Coverage notes guidance.", citation_ids=ids),
        ],
    )
    _persist_refused(day, banned, generation_input, policy, "fails validation")
    unknown = _with_sentences(
        result,
        [
            Sentence(text="Coverage one.", citation_ids=["story:999999"]),
            Sentence(text="Coverage two.", citation_ids=ids),
        ],
    )
    _persist_refused(day, unknown, generation_input, policy, "fails validation")
    # Duplicate citations inside one sentence are A2's job to normalize; a
    # result that arrives un-normalized did not come from A2.
    doubled = _with_sentences(
        result,
        [
            Sentence(text="Coverage one.", citation_ids=ids + ids),
            Sentence(text="Coverage two.", citation_ids=ids),
        ],
    )
    _persist_refused(day, doubled, generation_input, policy, "normalized form")
    # The banned copy never reached any table, in any form.
    assert "buy" not in dump_all_rows(day.repository).lower()


@pytest.mark.parametrize(
    "label, mutate, match",
    [
        (
            "no attempts",
            lambda r: dataclasses.replace(r, attempts=()),
            "at least one attempt",
        ),
        (
            "numbering 1,3",
            lambda r: _renumbered(
                dataclasses.replace(
                    r,
                    attempts=(
                        _attempt(
                            1,
                            "rejected",
                            validation_codes=("blank_sentence",),
                            failures=(
                                guarded.ValidationFailure(
                                    "blank_sentence", "sentence 1"
                                ),
                            ),
                        ),
                        r.attempts[0],
                    ),
                    accepted_attempt=2,
                ),
                (1, 3),
            ),
            "1..2 in order",
        ),
        (
            "two accepted attempts",
            lambda r: dataclasses.replace(
                r,
                attempts=(_attempt(1, "accepted"), _attempt(2, "accepted")),
                accepted_attempt=2,
            ),
            "ended the generation",
        ),
        (
            "accepted attempt not last",
            lambda r: dataclasses.replace(
                r,
                attempts=(_attempt(1, "accepted"), _attempt(2, "provider_error")),
                accepted_attempt=1,
            ),
            "ended the generation",
        ),
        (
            "accepted_attempt metadata mismatch",
            lambda r: dataclasses.replace(r, accepted_attempt=2),
            "accepted_attempt does not name",
        ),
        (
            "accepted_attempt missing",
            lambda r: dataclasses.replace(r, accepted_attempt=None),
            "accepted_attempt does not name",
        ),
        (
            "more attempts than the policy allows",
            lambda r: dataclasses.replace(
                r,
                attempts=(
                    _attempt(1, "provider_error"),
                    _attempt(2, "provider_error"),
                    r.attempts[0],
                ),
                accepted_attempt=3,
            ),
            "more attempts than the policy allows",
        ),
        (
            "max_attempts not the policy's",
            lambda r: dataclasses.replace(r, max_attempts=1),
            "max_attempts does not match",
        ),
        (
            "accepted with a reason",
            lambda r: dataclasses.replace(r, reason="validation_exhausted"),
            "carries no reason",
        ),
        (
            "accepted without a summary",
            lambda r: dataclasses.replace(r, summary=None),
            "carries a summary",
        ),
        (
            "unknown attempt outcome",
            lambda r: dataclasses.replace(
                r, attempts=(_attempt(1, "mystery"),), accepted_attempt=None
            ),
            "unknown outcome",
        ),
        (
            "unknown status",
            lambda r: dataclasses.replace(r, status="pending"),
            "unknown status",
        ),
        (
            "failures without a rejection",
            lambda r: dataclasses.replace(
                r,
                attempts=(
                    _attempt(
                        1,
                        "accepted",
                        validation_codes=("blank_sentence",),
                        failures=(guarded.ValidationFailure("blank_sentence", "s"),),
                    ),
                ),
            ),
            "carries failures but was not rejected",
        ),
        (
            "rejection without codes",
            lambda r: dataclasses.replace(
                r,
                status="unavailable",
                summary=None,
                accepted_attempt=None,
                reason="validation_exhausted",
                attempts=(_attempt(1, "rejected"), _attempt(2, "rejected")),
            ),
            "rejected without known codes",
        ),
        (
            "unavailable containing an accepted attempt",
            lambda r: dataclasses.replace(
                r,
                status="unavailable",
                summary=None,
                accepted_attempt=None,
                reason="provider_unavailable",
                attempts=(_attempt(1, "provider_error"), _attempt(2, "accepted")),
            ),
            "no accepted attempt",
        ),
        (
            "unavailable with a summary",
            lambda r: dataclasses.replace(
                r,
                status="unavailable",
                accepted_attempt=None,
                reason="provider_unavailable",
                attempts=(_attempt(1, "provider_error"), _attempt(2, "provider_error")),
            ),
            "carries no summary",
        ),
        (
            "unavailable with the wrong reason",
            lambda r: dataclasses.replace(
                r,
                status="unavailable",
                summary=None,
                accepted_attempt=None,
                reason="validation_exhausted",
                attempts=(_attempt(1, "provider_error"), _attempt(2, "provider_error")),
            ),
            "does not follow from the final attempt",
        ),
        (
            "unavailable with an accepted_attempt",
            lambda r: dataclasses.replace(
                r,
                status="unavailable",
                summary=None,
                accepted_attempt=1,
                reason="provider_unavailable",
                attempts=(_attempt(1, "provider_error"), _attempt(2, "provider_error")),
            ),
            "no accepted_attempt",
        ),
        (
            "unconfigured attempt followed by another",
            lambda r: dataclasses.replace(
                r,
                status="unavailable",
                summary=None,
                accepted_attempt=None,
                reason="provider_unconfigured",
                attempts=(
                    _attempt(1, "provider_unconfigured"),
                    _attempt(2, "provider_error"),
                ),
            ),
            "ended the generation",
        ),
        (
            "another input's result",
            lambda r: dataclasses.replace(r, input_fingerprint="9" * 64),
            "not generated from",
        ),
        (
            "another policy's result",
            lambda r: dataclasses.replace(r, policy_fingerprint="9" * 64),
            "policy fingerprint",
        ),
    ],
)
def test_a_result_violating_the_a2_contract_is_refused_atomically(
    tmp_path, label, mutate, match
):
    day = build_day(tmp_path)
    generation_input, policy, result = _real_result(day)
    _persist_refused(day, mutate(result), generation_input, policy, match)


def test_a_mutated_result_is_refused_even_when_it_would_be_stale(tmp_path):
    day = build_day(tmp_path)
    generation_input, policy, result = _real_result(day)
    ids = list(result.summary.sentences[0].citation_ids)
    banned = _with_sentences(
        result,
        [
            Sentence(text="You should buy Tesla.", citation_ids=ids),
            Sentence(text="Coverage notes guidance.", citation_ids=ids),
        ],
    )
    clear_theme_set(day)  # the input is no longer live
    _persist_refused(day, banned, generation_input, policy, "fails validation")


def test_real_a2_results_still_persist_normally(tmp_path):
    day = build_day(tmp_path)
    for client, expected in (
        (EchoClient(), SUMMARY_GENERATION_ACCEPTED),
        (FailingClient(), SUMMARY_GENERATION_UNAVAILABLE),
        (UnconfiguredClient(), SUMMARY_GENERATION_UNAVAILABLE),
        (BannedClient(), SUMMARY_GENERATION_UNAVAILABLE),
    ):
        policy = resolve_generation_policy(client)
        generation_input = live_input(day, "Robotaxi")
        result = generate_guarded_summary(
            generation_input, client=client, policy=policy
        )
        with open_run(day.repository) as run:
            generation = day.repository.persist_summary_generation(
                run=run, result=result, generation_input=generation_input, policy=policy
            )
        assert generation.outcome == expected


# ----------------------------------------------------------------------
# Codex finding 4: a rolled-back operation contributes nothing to the run log
# ----------------------------------------------------------------------


class _CommitFails:
    """A connection whose commit fails once, after everything was written."""

    def __init__(self, real):
        self._real = real

    def commit(self):
        raise sqlite3.OperationalError("disk I/O error (injected at commit)")

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextmanager
def failing_commit(monkeypatch, repository: Phase0Repository):
    """Make the *next* writable connection this repository opens fail to commit."""

    real_open = Phase0Repository._open_connection
    state = {"armed": True}

    def open_connection(self):
        connection = real_open(self)
        if self is repository and state["armed"]:
            state["armed"] = False
            return _CommitFails(connection)
        return connection

    monkeypatch.setattr(Phase0Repository, "_open_connection", open_connection)
    try:
        yield
    finally:
        monkeypatch.undo()


def _run_log_for(day: Day, run_id: str) -> dict:
    rows = [
        row
        for row in day.repository.read.run_log_rows(stage=STAGE)
        if row["run_id"] == run_id
    ]
    assert len(rows) == 1
    row = dict(rows[0])
    row["counts"] = json.loads(row["counts"])
    return row


def _run_log_row(repository: Phase0Repository, run_id: str, *, stage=STAGE) -> dict:
    rows = repository.read.run_log_rows(run_id=run_id, stage=stage)
    assert len(rows) == 1
    return dict(rows[0])


def _record_markers(monkeypatch) -> list[str]:
    """Watch the markers ``_logged_mutation`` mints, without changing them.

    The real generator still runs -- the tests below need the ordinary
    write path, not a scripted one -- and every minted marker is appended
    here so a test can name "this operation's marker" afterwards.
    """

    minted: list[str] = []
    real = Phase0Repository._new_mutation_id

    def new_mutation_id():
        marker = real()
        minted.append(marker)
        return marker

    monkeypatch.setattr(
        Phase0Repository, "_new_mutation_id", staticmethod(new_mutation_id)
    )
    return minted


def _before_the_probe(monkeypatch, repository: Phase0Repository, action) -> None:
    """Run ``action`` once, right before ``repository`` reads durable state.

    The probe itself is the real one: it reads whatever ``action`` left on
    disk.  Nothing about its answer is scripted.
    """

    real_probe = Phase0Repository._open_probe_connection
    state = {"armed": True}

    def probe(self):
        if self is repository and state["armed"]:
            state["armed"] = False
            action()
        return real_probe(self)

    monkeypatch.setattr(Phase0Repository, "_open_probe_connection", probe)


def _spy_on_reconciliation(monkeypatch) -> list[dict]:
    """Record what ``_commit_landed`` was asked and what it answered."""

    calls: list[dict] = []
    real = Phase0Repository._commit_landed

    def commit_landed(self, context, before, intended, mutation_id):
        answer = real(self, context, before, intended, mutation_id)
        calls.append(
            {
                "before": before,
                "intended": intended,
                "mutation_id": mutation_id,
                "answer": answer,
            }
        )
        return answer

    monkeypatch.setattr(Phase0Repository, "_commit_landed", commit_landed)
    return calls


def _writer_b(day: Day, clock: ManualClock) -> Phase0Repository:
    """A second repository object over the same database and the same clock:
    another process, as far as the database can tell."""

    return Phase0Repository(day.repository.database_path, clock=clock)


class WriterBClient(EchoClient):
    """Echoes like ``EchoClient`` but signs its label, so the artifact it
    produces can be told from the one another writer would have."""

    def generate(self, system_prompt, user_prompt, response_schema):
        summary = super().generate(system_prompt, user_prompt, response_schema)
        return response_schema.model_validate(
            {
                "label": "Coverage of deliveries by writer B",
                "sentences": [
                    {"text": s.text, "citation_ids": list(s.citation_ids)}
                    for s in summary.sentences
                ],
            }
        )


def _shared_run(repository: Phase0Repository, *, attempt: int = 1):
    """A ``summaries`` run under the one run identity both writers share."""

    return repository.stage_run(
        run_id="shared",
        stage=STAGE,
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
        attempt=attempt,
    )


def _outcome_without_marker(outcome: tuple) -> tuple:
    index = Phase0Repository._RUN_LOG_MUTATION_ID_INDEX
    return outcome[:index] + outcome[index + 1 :]


def test_a_failed_commit_claims_no_committed_success(tmp_path, monkeypatch):
    day = build_day(tmp_path)
    with open_run(day.repository, run_id="commit-fails") as run:
        with failing_commit(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError, match="injected at commit"):
                ensure_summary(
                    day.repository,
                    run=run,
                    ticker=TICKER,
                    trading_day=DAY,
                    pipeline_version=VERSION,
                    theme_id=day.theme_ids["Deliveries"],
                    client=EchoClient(),
                )
        assert run.state == "terminal_failed"
        assert run.success_count == 0 and run.failure_count == 1
        assert not any(key.startswith("summary_") for key in run.counts)
    # Reconnect: nothing durable, and the durable failure claims nothing.
    fresh = Phase0Repository(day.repository.database_path)
    assert {t: fresh.read.count(t) for t in SUMMARY_TABLES} == {
        t: 0 for t in SUMMARY_TABLES
    }
    row = _run_log_for(day, "commit-fails")
    assert row["status"] == "failed"
    assert row["success_count"] == 0
    assert row["partial_count"] == 0
    assert row["failure_count"] == 1
    assert not any(key.startswith("summary_") for key in row["counts"])
    assert any("injected at commit" in json.dumps(e) for e in json.loads(row["errors"]))
    assert current(day, NeverClient()) is None


def test_a_prior_committed_operation_survives_a_later_rolled_back_one(
    tmp_path, monkeypatch
):
    day = build_day(tmp_path)
    with open_run(day.repository, run_id="two-ops") as run:
        first = ensure_summary(
            day.repository,
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=VERSION,
            theme_id=day.theme_ids["Deliveries"],
            client=EchoClient(),
        )
        assert first.source == lifecycle.SOURCE_GENERATED
        committed = dict(run.counts)
        assert committed["summary_generations"] == 1
        assert committed["summary_artifacts_inserted"] == 1
        assert committed["summary_accepted"] == 1
        assert run.success_count == 1
        with failing_commit(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError, match="injected at commit"):
                ensure_summary(
                    day.repository,
                    run=run,
                    ticker=TICKER,
                    trading_day=DAY,
                    pipeline_version=VERSION,
                    theme_id=day.theme_ids["Robotaxi"],
                    client=EchoClient(),
                )
        assert run.state == "terminal_failed"
        assert run.counts == committed  # exactly once, not zero, not twice
        assert run.success_count == 1 and run.failure_count == 1
    row = _run_log_for(day, "two-ops")
    assert row["status"] == "failed"
    assert row["success_count"] == 1
    assert row["failure_count"] == 1
    assert row["counts"] == committed
    fresh = Phase0Repository(day.repository.database_path)
    assert fresh.read.count("summary_artifacts") == 1
    assert fresh.read.count("summary_generations") == 1
    assert current(day, NeverClient()).artifact == first.artifact
    assert current(day, NeverClient(), theme="Robotaxi") is None


def test_a_rolled_back_theme_reconciliation_claims_no_success_either(
    tmp_path, monkeypatch
):
    """The staging is the logged mutation's, so every entrypoint gets it."""

    day = build_day(tmp_path)
    with day.repository.stage_run(
        run_id="themes-rollback",
        stage="themes",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        with failing_commit(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError):
                day.repository.clear_theme_set(
                    run=run,
                    ticker=TICKER,
                    trading_day=DAY,
                    pipeline_version=VERSION,
                    terminal=True,
                )
        assert run.success_count == 0 and run.failure_count == 1
        assert "cleared_themes" not in run.counts
    assert day.population().theme_set is not None  # nothing was cleared


# ----------------------------------------------------------------------
# Codex finding 5: real concurrency, one artifact
# ----------------------------------------------------------------------


def test_same_key_concurrent_completions_yield_one_accepted_artifact(tmp_path):
    day = build_day(tmp_path)
    path = day.repository.database_path
    generation_input = live_input(day)
    policy = resolve_generation_policy(EchoClient())
    results = [
        generate_guarded_summary(generation_input, client=EchoClient(), policy=policy)
        for _ in range(2)
    ]
    assert all(result.accepted for result in results)

    barrier = threading.Barrier(2)
    outcomes: dict[str, object] = {}

    def worker(name: str, result) -> None:
        repository = Phase0Repository(path)  # its own connections, own runs
        try:
            barrier.wait(timeout=10)
            with repository.stage_run(
                run_id=name,
                stage=STAGE,
                trading_day=DAY,
                pipeline_version=VERSION,
                ticker=TICKER,
            ) as run:
                outcomes[name] = repository.persist_summary_generation(
                    run=run,
                    result=result,
                    generation_input=generation_input,
                    policy=policy,
                )
        except BaseException as exc:  # noqa: BLE001 - reported below
            outcomes[name] = exc

    threads = [
        threading.Thread(target=worker, args=(f"worker-{index}", result))
        for index, result in enumerate(results)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads)
    assert not any(
        isinstance(value, BaseException) for value in outcomes.values()
    ), outcomes
    assert sorted(outcomes) == ["worker-0", "worker-1"]
    by_outcome = {g.outcome: g for g in outcomes.values()}
    assert set(by_outcome) == {
        SUMMARY_GENERATION_ACCEPTED,
        SUMMARY_GENERATION_DISCARDED_DUPLICATE,
    }
    winner = by_outcome[SUMMARY_GENERATION_ACCEPTED]
    loser = by_outcome[SUMMARY_GENERATION_DISCARDED_DUPLICATE]
    assert loser.artifact_id == winner.artifact_id
    assert winner.provider_calls == 1 and loser.provider_calls == 1
    counts = table_counts(day.repository)
    assert counts["summary_artifacts"] == 1
    assert counts["summary_generations"] == 2
    assert counts["summary_generation_attempts"] == 2
    assert [a.status for a in artifacts(day)] == [SUMMARY_ARTIFACT_ACCEPTED]
    assert current(day, NeverClient()).artifact.artifact_id == winner.artifact_id
    rows = {
        row["run_id"]: row["status"]
        for row in day.repository.read.run_log_rows(stage=STAGE)
    }
    assert rows == {"worker-0": "success", "worker-1": "success"}


# ----------------------------------------------------------------------
# Codex finding: commit() raising is not proof of a rollback.  What is on
# disk decides, and the run's accounting follows it.
# ----------------------------------------------------------------------


class _CommitsThenRaises:
    """A connection whose commit lands and *then* reports an error."""

    def __init__(self, real):
        self._real = real

    def commit(self):
        self._real.commit()
        raise sqlite3.OperationalError("disk I/O error (injected after commit)")

    def __getattr__(self, name):
        return getattr(self._real, name)


@contextmanager
def commit_then_raise(monkeypatch, repository: Phase0Repository):
    """Make the *next* writable connection this repository opens commit, then raise."""

    real_open = Phase0Repository._open_connection
    state = {"armed": True}

    def open_connection(self):
        connection = real_open(self)
        if self is repository and state["armed"]:
            state["armed"] = False
            return _CommitsThenRaises(connection)
        return connection

    monkeypatch.setattr(Phase0Repository, "_open_connection", open_connection)
    try:
        yield
    finally:
        monkeypatch.undo()


EXPECTED_ACCEPTED_COUNTS = {
    "summary_generations": 1,
    "summary_provider_calls": 1,
    "summary_accepted": 1,
    "summary_artifacts_inserted": 1,
    "summary_artifacts_invalidated": 0,
}


def _ensure_deliveries(day: Day, run, *, theme="Deliveries", terminal=False):
    return ensure_summary(
        day.repository,
        run=run,
        ticker=TICKER,
        trading_day=DAY,
        pipeline_version=VERSION,
        theme_id=day.theme_ids[theme],
        client=EchoClient(),
        terminal=terminal,
    )


def test_a_commit_that_lands_then_raises_keeps_its_accounting_nonterminal(
    tmp_path, monkeypatch
):
    day = build_day(tmp_path)
    with open_run(day.repository, run_id="landed") as run:
        with commit_then_raise(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError, match="after commit"):
                _ensure_deliveries(day, run)
        # Durable rows, exactly once.
        assert table_counts(day.repository) == {
            "summary_artifacts": 1,
            "summary_sentences": 2,
            "summary_sentence_citations": 3,
            "summary_generations": 1,
            "summary_generation_attempts": 1,
        }
        # The durable row is the one the transaction wrote: non-terminal,
        # with the committed counters, and no failure.
        durable = _run_log_for(day, "landed")
        assert durable["status"] == "degraded"
        assert durable["success_count"] == 1
        assert durable["partial_count"] == 0
        assert durable["failure_count"] == 0
        assert durable["counts"] == EXPECTED_ACCEPTED_COUNTS
        assert json.loads(durable["errors"]) == []
        # The context agrees with it, and the run is still open.
        assert run.state == "active"
        assert not run.settled
        assert (run.success_count, run.partial_count, run.failure_count) == (1, 0, 0)
        assert run.counts == EXPECTED_ACCEPTED_COUNTS
        assert run.errors == []
        # And the committed artifact is current: a retry is a cache hit.
        assert _ensure_deliveries(day, run).source == lifecycle.SOURCE_CACHE_HIT
    # The block ended normally, so the run settled on what it committed.
    final = _run_log_for(day, "landed")
    assert final["status"] == "success"
    assert final["success_count"] == 1 and final["failure_count"] == 0
    assert final["counts"] == EXPECTED_ACCEPTED_COUNTS
    assert run.state == "closed_without_terminal"
    assert current(day, NeverClient()) is not None


def test_a_commit_that_lands_then_raises_keeps_its_accounting_terminal(
    tmp_path, monkeypatch
):
    day = build_day(tmp_path)
    with open_run(day.repository, run_id="landed-terminal") as run:
        with commit_then_raise(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError, match="after commit"):
                _ensure_deliveries(day, run, terminal=True)
        assert table_counts(day.repository)["summary_artifacts"] == 1
        assert table_counts(day.repository)["summary_generations"] == 1
        assert table_counts(day.repository)["summary_generation_attempts"] == 1
        durable = _run_log_for(day, "landed-terminal")
        assert durable["status"] == "success"
        assert (
            durable["success_count"],
            durable["partial_count"],
            durable["failure_count"],
        ) == (1, 0, 0)
        assert durable["counts"] == EXPECTED_ACCEPTED_COUNTS
        assert json.loads(durable["errors"]) == []
        # The context reconciles to the durable terminal success, whole.
        assert run.state == "terminal_succeeded"
        assert run.settled
        assert (run.success_count, run.partial_count, run.failure_count) == (1, 0, 0)
        assert run.counts == EXPECTED_ACCEPTED_COUNTS
        assert run.errors == []
    # Leaving the block adds nothing: the outcome was written and is immutable.
    assert run.state == "terminal_succeeded"
    assert (run.success_count, run.partial_count, run.failure_count) == (1, 0, 0)
    assert _run_log_for(day, "landed-terminal") == durable
    assert current(day, NeverClient()) is not None


def test_a_prior_commit_plus_a_landed_late_error_count_each_exactly_once(
    tmp_path, monkeypatch
):
    day = build_day(tmp_path)
    with open_run(day.repository, run_id="two-landed") as run:
        first = _ensure_deliveries(day, run)
        assert first.source == lifecycle.SOURCE_GENERATED
        after_first = dict(run.counts)
        with commit_then_raise(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError, match="after commit"):
                _ensure_deliveries(day, run, theme="Robotaxi")
        both = {
            "summary_generations": 2,
            "summary_provider_calls": 2,
            "summary_accepted": 2,
            "summary_artifacts_inserted": 2,
            "summary_artifacts_invalidated": 0,
        }
        assert after_first == EXPECTED_ACCEPTED_COUNTS
        assert run.counts == both
        assert (run.success_count, run.partial_count, run.failure_count) == (2, 0, 0)
        assert run.state == "active"
        durable = _run_log_for(day, "two-landed")
        assert durable["status"] == "degraded"
        assert durable["success_count"] == 2 and durable["failure_count"] == 0
        assert durable["counts"] == both
        assert table_counts(day.repository)["summary_artifacts"] == 2
        assert table_counts(day.repository)["summary_generations"] == 2
    final = _run_log_for(day, "two-landed")
    assert final["status"] == "success" and final["counts"] == both
    assert current(day, NeverClient()) is not None
    assert current(day, NeverClient(), theme="Robotaxi") is not None


def test_a_genuine_rollback_is_still_told_apart_from_a_landed_commit(
    tmp_path, monkeypatch
):
    """The two helpers differ only in whether the data reached the disk;
    the repository answers each from the disk, not from the exception."""

    day = build_day(tmp_path)
    with open_run(day.repository, run_id="rolled-back") as run:
        with failing_commit(monkeypatch, day.repository):
            with pytest.raises(sqlite3.OperationalError, match="injected at commit"):
                _ensure_deliveries(day, run)
        assert run.state == "terminal_failed"
        assert (run.success_count, run.partial_count, run.failure_count) == (0, 0, 1)
        assert run.counts == {}
    assert table_counts(day.repository)["summary_artifacts"] == 0
    row = _run_log_for(day, "rolled-back")
    assert row["status"] == "failed" and row["success_count"] == 0
    assert row["counts"] == {}


def test_an_unprovable_commit_is_left_unknown_not_guessed(tmp_path, monkeypatch):
    """The commit landed, but the durable probe cannot run.

    Neither success nor rollback is assumed: nothing is written over the
    durable row, the counters are not rewritten, the run is left in the
    repository's existing "outcome unknown" state, and the exception
    still propagates.
    """

    day = build_day(tmp_path)

    def broken_probe(self):
        raise sqlite3.OperationalError("probe unavailable")

    with open_run(day.repository, run_id="unprovable") as run:
        with commit_then_raise(monkeypatch, day.repository):
            monkeypatch.setattr(
                Phase0Repository, "_open_probe_connection", broken_probe
            )
            with pytest.raises(sqlite3.OperationalError, match="after commit"):
                _ensure_deliveries(day, run)
        assert run.state == "settlement_failed"
        assert run.settled and run.terminated
        # Not fabricated either way: the accumulated counters stand, and the
        # error says the durable outcome is unknown.
        assert (run.success_count, run.partial_count, run.failure_count) == (1, 0, 0)
        assert any(error.get("durable_outcome") == "unknown" for error in run.errors)
        # The durable row is whatever the commit left -- here, the landed
        # write -- and no failure settlement was written over it.
        durable = _run_log_for(day, "unprovable")
        assert durable["status"] == "degraded"
        assert durable["success_count"] == 1 and durable["failure_count"] == 0
        assert durable["counts"] == EXPECTED_ACCEPTED_COUNTS
        assert json.loads(durable["errors"]) == []
    # Leaving the block writes nothing more either.
    assert _run_log_for(day, "unprovable") == durable
    assert table_counts(day.repository)["summary_artifacts"] == 1


@pytest.mark.parametrize("lands", [True, False], ids=["landed", "rolled_back"])
def test_an_identical_rewrite_is_told_apart_by_its_marker(tmp_path, monkeypatch, lands):
    """The operation would rewrite the run-log row to exactly what was
    there -- same counters, same clock.  The counters cannot say whether it
    landed; the marker can, because the rewrite carries a new one."""

    clock = ManualClock()
    day = clocked_day(tmp_path, monkeypatch, clock)
    repository = day.repository
    with repository.stage_run(
        run_id="identical",
        stage="themes",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        # First clear: the theme set goes, and the row records it.
        repository.clear_theme_set(
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=VERSION,
        )
        before = dict(run.counts)
        assert before["cleared_rows"] == 3
        first_marker = _run_log_row(repository, "identical", stage="themes")[
            "last_mutation_id"
        ]
        assert first_marker is not None
        # Second clear: nothing left to clear, so every counter it writes
        # is byte-identical to the first one under a frozen clock.
        sabotage = commit_then_raise if lands else failing_commit
        with sabotage(monkeypatch, repository):
            minted = _record_markers(monkeypatch)
            with pytest.raises(sqlite3.OperationalError):
                repository.clear_theme_set(
                    run=run,
                    ticker=TICKER,
                    trading_day=DAY,
                    pipeline_version=VERSION,
                )
        assert len(minted) == 1 and minted[0] != first_marker
        row = _run_log_row(repository, "identical", stage="themes")
        if lands:
            # Proved durable by its own marker: the run carries on.
            assert row["last_mutation_id"] == minted[0]
            assert run.state == "active" and not run.settled
            assert run.counts == before
            assert run.success_count == 3 and run.failure_count == 0
            assert run.errors == []
        else:
            # Proved rolled back: the row is exactly the first clear's,
            # old marker included, so the failure path runs as usual.
            assert row["last_mutation_id"] == first_marker
            assert run.state == "terminal_failed"
            assert run.counts == before
            assert run.success_count == 3 and run.failure_count == 1
    row = _run_log_row(repository, "identical", stage="themes")
    if lands:
        assert row["status"] == "success"
        assert row["success_count"] == 3 and row["failure_count"] == 0
        # Settlement wrote no marker of its own and kept the mutation's.
        assert row["last_mutation_id"] == minted[0]
    else:
        assert row["status"] == "failed"
        assert row["success_count"] == 3 and row["failure_count"] == 1
        assert row["last_mutation_id"] == first_marker


# ----------------------------------------------------------------------
# Codex P2: ``(run_id, stage)`` names a row, not the transaction that
# wrote it.  A second writer holding the same run identity can commit a
# row whose counters coincide with what this operation intended; only the
# per-mutation marker tells the two apart.
# ----------------------------------------------------------------------


def test_every_logged_mutation_writes_its_own_marker_and_settlement_keeps_it(
    tmp_path, monkeypatch
):
    day = build_day(tmp_path)
    minted = _record_markers(monkeypatch)
    with open_run(day.repository, run_id="marked") as run:
        _ensure_deliveries(day, run)
        first = _run_log_for(day, "marked")["last_mutation_id"]
        _ensure_deliveries(day, run, theme="Robotaxi")
        second = _run_log_for(day, "marked")["last_mutation_id"]
    # One fresh marker per invocation, written with the row it describes.
    assert minted == [first, second]
    assert first != second
    assert all(len(m) == 32 and int(m, 16) >= 0 for m in minted)
    # A read that mutates nothing mints nothing: the cache hit is not a
    # logged mutation and leaves the row -- and its marker -- alone.
    with open_run(day.repository, run_id="marked-again") as run:
        assert _ensure_deliveries(day, run).source == lifecycle.SOURCE_CACHE_HIT
    assert len(minted) == 2
    assert _run_log_for(day, "marked-again")["last_mutation_id"] is None
    # Another logged mutation, in another run and stage, mints anew: the
    # marker is not derived from the run, the stage or the data.
    with day.repository.stage_run(
        run_id="marked-again",
        stage="themes",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker=TICKER,
    ) as run:
        day.repository.clear_theme_set(
            run=run, ticker=TICKER, trading_day=DAY, pipeline_version=VERSION
        )
    assert len(minted) == 3 and len(set(minted)) == 3
    assert (
        _run_log_row(day.repository, "marked-again", stage="themes")["last_mutation_id"]
        == minted[2]
    )
    # The stage's own settlement is not a logged mutation and mints no
    # marker: the row keeps the last mutation's.
    final = _run_log_for(day, "marked")
    assert final["status"] == "success" and final["last_mutation_id"] == second


def test_a_competing_writer_with_a_coinciding_outcome_is_not_this_commit(
    tmp_path, monkeypatch
):
    """The exact collision.

    Writers A and B each hold an authorized run under the *same* run
    identity, on the same partition, over one database, under one frozen
    clock.  A's terminal mutation rolls back.  Before A's durability probe
    runs, B commits a terminal mutation of its own whose whole run-log
    outcome -- status, every counter, every count, the clock -- is equal
    to what A intended.  Compared on those columns, B's row *is* A's
    intended row, and A would conclude it had committed.  It had not: only
    B's data exists.
    """

    clock = ManualClock()
    day = clocked_day(tmp_path, monkeypatch, clock)
    a = day.repository
    b = _writer_b(day, clock)

    # The pre-operation state: an earlier attempt of this run identity
    # settled, and the row carries the marker of the mutation that wrote it.
    with _shared_run(a) as run:
        assert _ensure_deliveries(day, run).source == lifecycle.SOURCE_GENERATED
    before_row = _run_log_for(day, "shared")
    m0 = before_row["last_mutation_id"]
    assert m0 is not None and before_row["status"] == "success"
    robotaxi = day.theme_ids["Robotaxi"]
    b_client = WriterBClient()

    def writer_b_commits():
        # B's own, ordinary path: its own run, its own marker, a real
        # commit.  Terminal, like A's, so the durable status coincides too.
        with _shared_run(b, attempt=2) as run_b:
            outcome = ensure_summary(
                b,
                run=run_b,
                ticker=TICKER,
                trading_day=DAY,
                pipeline_version=VERSION,
                theme_id=robotaxi,
                client=b_client,
                terminal=True,
            )
            assert outcome.source == lifecycle.SOURCE_GENERATED
        assert run_b.state == "terminal_succeeded"

    with _shared_run(a, attempt=2) as run_a:
        with failing_commit(monkeypatch, a):
            minted = _record_markers(monkeypatch)
            calls = _spy_on_reconciliation(monkeypatch)
            _before_the_probe(monkeypatch, a, writer_b_commits)
            with pytest.raises(sqlite3.OperationalError, match="injected at commit"):
                ensure_summary(
                    a,
                    run=run_a,
                    ticker=TICKER,
                    trading_day=DAY,
                    pipeline_version=VERSION,
                    theme_id=robotaxi,
                    client=EchoClient(),
                    terminal=True,
                )
        # Two markers were minted, A's first, and they differ from each
        # other and from the pre-operation row's.
        assert len(minted) == 2
        ma, mb = minted
        assert len({ma, mb, m0}) == 3
        # The collision is real: apart from the marker, B's durable row is
        # exactly what A intended to write.
        [call] = calls
        assert call["mutation_id"] == ma
        durable = _run_log_row(a, "shared")
        assert durable["last_mutation_id"] == mb
        intended = call["intended"]
        assert intended[Phase0Repository._RUN_LOG_MUTATION_ID_INDEX] == ma
        with a.admin.connect_writable() as connection:
            durable_outcome = a._run_log_outcome(connection, run_a)
        assert _outcome_without_marker(durable_outcome) == _outcome_without_marker(
            intended
        )
        assert durable_outcome != intended
        # And A does not claim it.
        assert call["answer"] is None
        assert run_a.state == "settlement_failed"
        assert run_a.state != "terminal_succeeded"
        assert run_a.settled and run_a.terminated
        assert any(e.get("durable_outcome") == "unknown" for e in run_a.errors)
        # Nothing restored on the assumption of a rollback, and no failure
        # counted on the assumption of one either.
        assert (run_a.success_count, run_a.partial_count, run_a.failure_count) == (
            1,
            0,
            0,
        )
    # Only B's Robotaxi summary exists; A's is nowhere.
    robotaxi_artifacts = artifacts(day, "Robotaxi")
    assert [row.label for row in robotaxi_artifacts] == [
        "Coverage of deliveries by writer B"
    ]
    assert [g.outcome for g in generations(day, "Robotaxi")] == [
        SUMMARY_GENERATION_ACCEPTED
    ]
    assert table_counts(a)["summary_artifacts"] == 2
    # B's durable row was not written over: not by A's reconciliation and
    # not by A's block ending.
    assert _run_log_row(a, "shared") == durable
    assert durable["status"] == "success" and durable["success_count"] == 1
    assert b_client.prompts and len(b_client.prompts) == 1


def test_a_landed_commit_overwritten_by_a_competing_writer_is_unknown_not_rollback(
    tmp_path, monkeypatch
):
    """The complementary race.

    A's commit lands and then raises.  Before A probes, B commits a normal
    mutation under the same run identity, so the row A would have found
    with its own marker now carries B's.  The row can no longer prove
    A's transaction either way -- and in particular a marker that is not
    A's is *not* evidence that A rolled back.
    """

    clock = ManualClock()
    day = clocked_day(tmp_path, monkeypatch, clock)
    a = day.repository
    b = _writer_b(day, clock)
    b_client = WriterBClient()
    closing = contextlib.ExitStack()

    def writer_b_commits():
        # B's run stays open across A's probe so that what A reads is the
        # row as B's *mutation* left it; B settles after A has decided.
        run_b = closing.enter_context(_shared_run(b))
        outcome = ensure_summary(
            b,
            run=run_b,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=VERSION,
            theme_id=day.theme_ids["Robotaxi"],
            client=b_client,
        )
        assert outcome.source == lifecycle.SOURCE_GENERATED

    with _shared_run(a) as run_a:
        with commit_then_raise(monkeypatch, a):
            minted = _record_markers(monkeypatch)
            calls = _spy_on_reconciliation(monkeypatch)
            _before_the_probe(monkeypatch, a, writer_b_commits)
            with pytest.raises(sqlite3.OperationalError, match="after commit"):
                _ensure_deliveries(day, run_a)
        ma, mb = minted
        assert ma != mb
        durable = _run_log_row(a, "shared")
        assert durable["last_mutation_id"] == mb
        [call] = calls
        assert call["mutation_id"] == ma and call["before"] is None
        assert call["answer"] is None
        # Unknown, not rolled back: the counters A accumulated stand, no
        # failure is recorded against them, nothing is written.
        assert run_a.state == "settlement_failed"
        assert (run_a.success_count, run_a.partial_count, run_a.failure_count) == (
            1,
            0,
            0,
        )
        assert run_a.counts == EXPECTED_ACCEPTED_COUNTS
        assert any(e.get("durable_outcome") == "unknown" for e in run_a.errors)
    # A's block ended without touching B's row ...
    assert _run_log_row(a, "shared") == durable
    # ... and A's data did land: both summaries are durable, one each.
    assert [row.label for row in artifacts(day, "Deliveries")] == [
        "Coverage of deliveries"
    ]
    assert [row.label for row in artifacts(day, "Robotaxi")] == [
        "Coverage of deliveries by writer B"
    ]
    assert current(day, NeverClient()) is not None
    # B settles its own run on its own terms, keeping its marker.
    closing.close()
    settled = _run_log_row(a, "shared")
    assert settled["status"] == "success"
    assert settled["last_mutation_id"] == mb


def test_a_commit_that_raises_before_the_final_write_is_a_rollback(
    tmp_path, monkeypatch
):
    """An exception before the run-log row is written cannot have landed
    anything; no probe is needed and none is made."""

    day = build_day(tmp_path)
    probes = []
    real_probe = Phase0Repository._open_probe_connection

    def counting_probe(self):
        probes.append(True)
        return real_probe(self)

    monkeypatch.setattr(Phase0Repository, "_open_probe_connection", counting_probe)
    with open_run(day.repository, run_id="body-raised") as run:
        with pytest.raises(Phase0ValidationError):
            day.repository.persist_summary_generation(
                run=run,
                result=object(),
                generation_input=object(),
                policy=object(),
            )
        assert run.state == "terminal_failed"
        assert probes == []
    assert _run_log_for(day, "body-raised")["status"] == "failed"

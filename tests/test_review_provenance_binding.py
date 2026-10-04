"""A4c: review provenance binding, held adversarially.

Every world here is written through the real Phase 0 paths -- logged
ingestion, logged story and theme reconciliation, A3's ``ensure_summary``
with a fake provider -- and then attacked through the admin surface, raw
SQL, or an edited manifest.  The question each test asks is the one A4b
has to answer: does the persisted state *establish* that a reviewed
summary's evidence came through the logged pipeline, or does it only
*look* like it?

Content identity (a matching ``input_fingerprint``) is never enough;
several tests keep it intact on purpose while production origin fails.

The last test documents the trust boundary rather than a defect: a writer
with raw SQLite access who forges ``run_log`` and every binding consistently
is indistinguishable from the pipeline, and this milestone does not claim
otherwise.
"""

from __future__ import annotations

import itertools
import json
import re
import shutil
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import ai.summarization as summarization
import nlp.eval.faithfulness as g2
import nlp.eval.review as review
import phase0.rss as rss
import phase0.stories as stories_module
import phase0.summary_lifecycle as lifecycle
import phase0.themes as themes_module
import phase0.yahoo as yahoo
from ai.guarded_summary import generate_guarded_summary, resolve_generation_policy
from nlp.dedup.selection import cluster_fingerprint_for
from nlp.themes import theme_fingerprint_for
from phase0 import provenance
from phase0.models import (
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import (
    MIGRATIONS_PATH,
    Phase0Repository,
    StoryGenerationConflict,
)
from phase0.schema import load_migrations
from phase0.summaries import build_generation_input
from phase0.summary_lifecycle import (
    SOURCE_CACHE_HIT,
    SOURCE_GENERATED,
    current_summary_artifact,
    ensure_summary,
)
from phase0.summary_runner import PRODUCTION_MAX_ATTEMPTS, production_generation_policy

VERSION = "v1"
D1, D2 = "2026-07-20", "2026-07-21"
OUTLETS = ("Reuters", "Bloomberg", "CNBC")
GENERATED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
CODE = {"commit": "test", "dirty": False}
ID_LINE_RE = re.compile(r"- id: (\S+)")
_ITEMS = itertools.count(1)
_RUNS = itertools.count(1)


@pytest.fixture(autouse=True)
def no_ambient_provider_config(monkeypatch):
    for name in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_TIMEOUT_MS",
    ):
        monkeypatch.delenv(name, raising=False)


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


class Client:
    """A fake provider: one sentence citing every story, one citing the first."""

    def __init__(self) -> None:
        self.model = summarization.DEFAULT_MODEL
        self.max_output_tokens = summarization.DEFAULT_MAX_OUTPUT_TOKENS
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls += 1
        ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": "Coverage summary",
                "sentences": [
                    {"text": "Coverage leads with this story.", "citation_ids": ids},
                    {"text": "Outlets repeat the report.", "citation_ids": ids[:1]},
                ],
            }
        )


@pytest.fixture
def repository(tmp_path) -> Phase0Repository:
    repo = Phase0Repository(tmp_path / "phase0.db", clock=Clock())
    repo.migrate()
    return repo


def run_id(prefix: str) -> str:
    return f"{prefix}-{next(_RUNS)}"


# ----------------------------------------------------------------------
# Building a world through the real write paths
# ----------------------------------------------------------------------


def item_payload(ticker, day, n):
    index = next(_ITEMS)
    outlet = OUTLETS[n % len(OUTLETS)]
    return {
        "source": f"yahoo:{outlet}",
        "ticker": ticker,
        "title": f"{outlet} headline {index}",
        "description": f"{outlet} standfirst {index}.",
        "url": f"https://{outlet.lower()}.example/{index}",
        "canonical_url": f"https://{outlet.lower()}.example/{index}",
        "published_at": f"{day}T10:{n:02d}:00+00:00",
        "fetched_at": f"{day}T11:00:00+00:00",
        "raw_json": {"index": index},
    }


def ingest(repo, ticker, day, count, *, stage="fetch_yahoo", replay=False, rid=None):
    """``count`` raw items through one logged ingestion run."""

    items = [item_payload(ticker, day, n) for n in range(count)]
    with repo.stage_run(
        run_id=rid or run_id("ingest"),
        stage=stage,
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
        replay=replay,
    ) as run:
        results = repo.ingest_raw_items(items, run=run, terminal=True)
    return [r.item_id for r in results]


def admin_items(repo, ticker, day, count):
    items = [item_payload(ticker, day, n) for n in range(count)]
    return [r.item_id for r in repo.admin.insert_raw_items(items)]


def story_records(repo, ticker, day, item_ids, *, title="story"):
    with repo.admin.connect_writable() as connection:
        rows = {
            int(r["id"]): r
            for r in connection.execute(
                "SELECT * FROM raw_items WHERE id IN (%s)"
                % ",".join("?" * len(item_ids)),
                list(item_ids),
            )
        }
    records = []
    for n, item_id in enumerate(item_ids):
        row = rows[item_id]
        fingerprint = cluster_fingerprint_for(ticker, [str(item_id)])
        outlet = row["source"].split(":", 1)[1]
        records.append(
            StoryRecord(
                cluster_fingerprint=fingerprint,
                canonical_title=f"{ticker} {day} {title} {n}",
                members=(
                    StoryMemberRecord(
                        raw_item_id=item_id,
                        position=0,
                        outlet=outlet,
                        url=row["url"],
                        canonical_url=row["canonical_url"],
                    ),
                ),
                canonical_item_id=item_id,
                outlet=outlet,
                outlet_count=1,
                published_at=row["published_at"],
                canonical_url=row["canonical_url"],
                content_hash=f"h-{fingerprint[:8]}",
                stage="m3.semantic",
                member_story_keys=(fingerprint,),
                algorithm_version="m3.1",
                config_fingerprint="cfg",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            )
        )
    return records


def reconcile_stories(repo, ticker, day, records, *, rid=None):
    with repo.stage_run(
        run_id=rid or run_id("stories"),
        stage="stories",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        return repo.reconcile_stories(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            stories=records,
            terminal=True,
        )


def theme_payload(repo, ticker, day, groups):
    """ThemeRecords the way M5 writes them: real content fingerprints."""

    stored = repo.stories_for_day(day, ticker)
    ids = sorted(int(r["id"]) for r in stored)
    keys = {int(r["id"]): r["cluster_fingerprint"] for r in stored}
    with repo.admin.connect_writable() as connection:
        members = {
            story_id: [
                int(m["raw_item_id"])
                for m in connection.execute(
                    "SELECT raw_item_id FROM story_members WHERE story_id = ?",
                    (story_id,),
                )
            ]
            for story_id in ids
        }
    records, cursor = [], 0
    for n, size in enumerate(groups):
        chosen = ids[cursor : cursor + size]
        cursor += size
        fingerprint = theme_fingerprint_for(
            ticker, date.fromisoformat(day), [keys[s] for s in chosen]
        )
        records.append(
            ThemeRecord(
                fingerprint=fingerprint,
                theme_key=f"key-{ticker}-{day}-{n}",
                label=f"Theme {n}",
                label_source="canonical_story_title",
                story_ids=tuple(chosen),
                citation_item_ids=tuple(i for s in chosen for i in members[s]),
                status="ready",
                salience_rank=n + 1,
                story_count=len(chosen),
            )
        )
    return records


def theme_set(story_count):
    return ThemeSetRecord(
        method="hdbscan",
        method_reason="clustered",
        source_metadata={"story_count": story_count},
        config_fingerprint="cfg",
        algorithm_version="m5.1",
        model_name="fake",
        model_revision="r1",
        embedding_dimension=4,
    )


def reconcile_themes(repo, ticker, day, records, *, signature=True, rid=None):
    expected = (
        repo.story_generation(ticker, day, VERSION).signature
        if signature is True
        else signature
    )
    with repo.stage_run(
        run_id=rid or run_id("themes"),
        stage="themes",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        return repo.reconcile_themes(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            theme_set=theme_set(len(repo.stories_for_day(day, ticker))),
            themes=records,
            expected_story_signature=expected,
            terminal=True,
        )


def summarize(repo, ticker, day, theme_id, *, client=None, rid=None):
    with repo.stage_run(
        run_id=rid or run_id("sum"),
        stage="summaries",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        return ensure_summary(
            repo,
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            theme_id=theme_id,
            client=client or Client(),
            max_attempts=PRODUCTION_MAX_ATTEMPTS,
        )


def production_partition(repo, ticker="TSLA", day=D1, groups=(2, 1), *, admin=False):
    """One partition, every hop logged; returns theme ids by rank."""

    count = sum(groups)
    items = (
        admin_items(repo, ticker, day, count)
        if admin
        else ingest(repo, ticker, day, count)
    )
    reconcile_stories(repo, ticker, day, story_records(repo, ticker, day, items))
    reconcile_themes(repo, ticker, day, theme_payload(repo, ticker, day, groups))
    return theme_ids(repo, ticker, day)


def theme_ids(repo, ticker="TSLA", day=D1):
    population = repo.read.theme_population(ticker, day, VERSION)
    return [
        t.theme_id for t in sorted(population.themes, key=lambda t: t.salience_rank)
    ]


def chain(repo, theme_id, ticker="TSLA", day=D1):
    """``(origin problems, theme-build problem)`` for one current artifact.

    Built by the same functions A4b records a manifest with, then judged by
    the same functions it scores with.
    """

    reader = repo.read
    population = reader.theme_population(ticker, day, VERSION)
    current = current_summary_artifact(
        reader, ticker, day, VERSION, theme_id, production_generation_policy()
    )
    assert current is not None, "the artifact must be current: content identity holds"
    facts = g2._artifact_provenance(
        population,
        current,
        reader.summary_artifact_provenance(current.artifact.artifact_id),
    )
    record = {**g2._artifact_record(current, 1), "provenance": facts}
    build = g2._theme_build_facts(population)
    return (
        g2.artifact_origin_problems(record, build),
        provenance.verify_theme_build(build),
    )


def raw_sql(repo, statement, parameters=()):
    """A bypass: every trigger off for one statement, then restored."""

    with repo.admin.connect_writable() as connection:
        triggers = list(
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            )
        )
        for name, _ in triggers:
            connection.execute(f"DROP TRIGGER {name}")
        try:
            connection.execute(statement, parameters)
        finally:
            for _, create in triggers:
                connection.execute(create)


def direct_sql(repo, statement, parameters=()):
    """Ordinary raw SQL through the admin connection: triggers stay on."""

    with repo.admin.connect_writable() as connection:
        connection.execute(statement, parameters)


def one(repo, sql, parameters=()):
    with repo.admin.connect_writable() as connection:
        return connection.execute(sql, parameters).fetchone()


# ----------------------------------------------------------------------
# The stage names this module spells must be the owners'
# ----------------------------------------------------------------------


def test_provenance_stage_names_are_the_owning_modules_names():
    assert provenance.INGESTION_STAGES == {yahoo.STAGE, rss.STAGE_INGEST}
    assert provenance.STORIES_STAGE == stories_module.STAGE
    assert provenance.THEMES_STAGE == themes_module.STAGE
    assert provenance.SUMMARIES_STAGE == lifecycle.STAGE


# ----------------------------------------------------------------------
# A. The full logged chain verifies
# ----------------------------------------------------------------------


def test_a_full_logged_chain_verifies(repository):
    first, _ = production_partition(repository)
    artifact = summarize(repository, "TSLA", D1, first)
    assert artifact.source == SOURCE_GENERATED
    problems, build = chain(repository, first)
    assert problems == []
    assert build is None


def test_the_bindings_name_the_runs_that_wrote_each_hop(repository):
    items = ingest(repository, "TSLA", D1, 1, rid="ingest-x")
    reconcile_stories(
        repository,
        "TSLA",
        D1,
        story_records(repository, "TSLA", D1, items),
        rid="stories-x",
    )
    reconcile_themes(
        repository,
        "TSLA",
        D1,
        theme_payload(repository, "TSLA", D1, (1,)),
        rid="themes-x",
    )
    [theme] = theme_ids(repository)
    summarize(repository, "TSLA", D1, theme, rid="sum-x")
    population = repository.read.theme_population("TSLA", D1, VERSION)
    [member] = population.member_provenance
    assert (member.ingest_run_id, member.ingest_stage) == ("ingest-x", "fetch_yahoo")
    [story] = population.stories.stories
    assert story.build_run_id == "stories-x"
    assert story.build_run.recorded_mutation is True
    theme_set = population.theme_set
    assert theme_set.build_run_id == "themes-x"
    assert theme_set.build_story_signature == population.stories.signature
    assert theme_set.build_story_signature_version == provenance.STORY_SIGNATURE_VERSION
    artifact_id = repository.read.summary_generations("TSLA", D1, VERSION)[
        0
    ].artifact_id
    produced = repository.read.summary_artifact_provenance(artifact_id)
    assert produced.generation.run_id == "sum-x" and produced.run.stage == "summaries"


# ----------------------------------------------------------------------
# B / C. Correct content, untrusted origin
# ----------------------------------------------------------------------


def test_admin_raw_items_leave_origin_unverified_with_content_intact(repository):
    first, _ = production_partition(repository, admin=True)
    summarize(repository, "TSLA", D1, first)
    problems, build = chain(repository, first)  # asserts the artifact is current
    assert problems and all("has no ingestion provenance" in p for p in problems)
    assert build is None  # the theme build itself is fine; origin is not


def test_a_payload_cannot_supply_its_own_ingestion_provenance(repository):
    payload = {**item_payload("TSLA", D1, 0), "ingest_run_id": "forged"}
    payload["ingest_stage"] = "fetch_yahoo"
    [admin] = repository.admin.insert_raw_items([payload])
    row = one(repository, "SELECT * FROM raw_items WHERE id = ?", (admin.item_id,))
    assert row["ingest_run_id"] is None and row["ingest_stage"] is None
    payload = {**item_payload("TSLA", D1, 1), "ingest_run_id": "forged"}
    with repository.stage_run(
        run_id="ingest-real",
        stage="fetch_yahoo",
        trading_day=D1,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run:
        [logged] = repository.ingest_raw_items([payload], run=run, terminal=True)
    row = one(repository, "SELECT * FROM raw_items WHERE id = ?", (logged.item_id,))
    assert row["ingest_run_id"] == "ingest-real"


def test_admin_stories_leave_origin_unverified(repository):
    items = ingest(repository, "TSLA", D1, 2)
    repository.admin.reconcile_stories(
        ticker="TSLA",
        trading_day=D1,
        pipeline_version=VERSION,
        stories=story_records(repository, "TSLA", D1, items),
    )
    reconcile_themes(
        repository, "TSLA", D1, theme_payload(repository, "TSLA", D1, (2,))
    )
    [theme] = theme_ids(repository)
    summarize(repository, "TSLA", D1, theme)
    problems, build = chain(repository, theme)
    assert problems and all("no run is recorded" in p for p in problems)
    assert build is None


def test_an_admin_story_change_clears_the_binding_it_had(repository):
    items = ingest(repository, "TSLA", D1, 2)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    repository.admin.reconcile_stories(
        ticker="TSLA",
        trading_day=D1,
        pipeline_version=VERSION,
        stories=story_records(repository, "TSLA", D1, items, title="rewritten"),
    )
    stored = repository.stories_for_day(D1, "TSLA")
    assert [r["build_run_id"] for r in stored] == [None, None]


# ----------------------------------------------------------------------
# D / E / S. Untrusted theme writes cannot keep a trusted binding
# ----------------------------------------------------------------------


def test_an_admin_theme_set_is_never_bound(repository):
    items = ingest(repository, "TSLA", D1, 2)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    repository.admin.reconcile_themes(
        ticker="TSLA",
        trading_day=D1,
        pipeline_version=VERSION,
        theme_set=theme_set(len(repository.stories_for_day(D1, "TSLA"))),
        themes=theme_payload(repository, "TSLA", D1, (2,)),
    )
    [theme] = theme_ids(repository)
    summarize(repository, "TSLA", D1, theme)
    problems, build = chain(repository, theme)
    assert problems == []  # every hop was logged except the theme build...
    assert build == "the theme set carries no build binding"  # ...which is not


def _bound_partition(repository):
    first, second = production_partition(repository)
    assert chain_build(repository) is None
    return first, second


def chain_build(repository):
    return provenance.verify_theme_build(
        g2._theme_build_facts(repository.read.theme_population("TSLA", D1, VERSION))
    )


def test_an_admin_rewrite_of_membership_clears_the_binding(repository):
    _bound_partition(repository)
    stored = theme_payload(repository, "TSLA", D1, (1, 2))
    repository.admin.reconcile_themes(
        ticker="TSLA",
        trading_day=D1,
        pipeline_version=VERSION,
        theme_set=theme_set(len(repository.stories_for_day(D1, "TSLA"))),
        themes=stored,
    )
    assert chain_build(repository) == "the theme set carries no build binding"


def test_an_admin_theme_insert_clears_the_binding(repository):
    _bound_partition(repository)
    # A stray legacy theme written into the partition: the binding must not
    # survive a write it did not make.
    direct_sql(
        repository,
        "INSERT INTO themes (ticker, trading_day, label, salience_rank, status, "
        "content_hash, pipeline_version) VALUES ('TSLA', ?, 'x', 9, 'ready', 'h', ?)",
        (D1, VERSION),
    )
    assert chain_build(repository) == "the theme set carries no build binding"


@pytest.mark.parametrize(
    "statement",
    [
        # Rewriting a membership row, even to itself, as a repair would.
        "UPDATE theme_stories SET story_id = story_id",
        "UPDATE themes SET label = 'relabelled'",
        "UPDATE theme_sets SET method_reason = 'edited'",
        "DELETE FROM theme_citations WHERE raw_item_id = "
        "(SELECT min(raw_item_id) FROM theme_citations)",
    ],
)
def test_a_direct_theme_write_after_binding_clears_it(repository, statement):
    _bound_partition(repository)
    direct_sql(repository, statement)
    assert chain_build(repository) == "the theme set carries no build binding"


def test_a_membership_rewrite_that_bypasses_triggers_fails_the_fingerprint(repository):
    first, second = _bound_partition(repository)
    moved = one(
        repository,
        "SELECT min(story_id) AS id FROM theme_stories WHERE theme_id = ?",
        (first,),
    )["id"]
    raw_sql(repository, "DELETE FROM theme_citations WHERE theme_id = ?", (first,))
    raw_sql(
        repository,
        "UPDATE theme_stories SET theme_id = ? WHERE story_id = ?",
        (second, moved),
    )
    problem = chain_build(repository)
    assert problem is not None and "does not recompute from its membership" in problem


def test_a_story_content_write_after_binding_clears_it(repository):
    first, _ = _bound_partition(repository)
    summarize(repository, "TSLA", D1, first)
    direct_sql(repository, "UPDATE stories SET outlet_count = outlet_count")
    stored = repository.stories_for_day(D1, "TSLA")
    assert all(r["build_run_id"] is None for r in stored)


# ----------------------------------------------------------------------
# F. story_members / raw-item linkage tampered after the theme build
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bypass", [False, True])
def test_member_tampering_after_the_build_fails_deterministically(repository, bypass):
    first, _ = _bound_partition(repository)
    summarize(repository, "TSLA", D1, first)
    write = raw_sql if bypass else direct_sql
    write(repository, "UPDATE story_members SET outlet = 'Forged'")
    # The story signature moved, so the build is over a generation that is
    # gone -- whether or not the story bindings were cleared with it.
    assert chain_build(repository) == (
        "the theme set was built over a story generation that is no longer the "
        "partition's"
    )
    if not bypass:
        stored = repository.stories_for_day(D1, "TSLA")
        assert all(r["build_run_id"] is None for r in stored)


# ----------------------------------------------------------------------
# G. Forged or mismatched run identities
# ----------------------------------------------------------------------


def _log(repository, rid, stage, *, ticker="TSLA", day=D1, version=VERSION):
    """A run_log row written by the admin path: no mutation marker."""

    repository.admin.log_stage(
        run_id=rid,
        stage=stage,
        counts={},
        duration_ms=0,
        errors=[],
        started_at=f"{day}T12:00:00+00:00",
        completed_at=f"{day}T12:00:01+00:00",
        trading_day=day,
        pipeline_version=version,
        status="success",
        ticker=ticker,
    )


def _logged_run(repository, rid, stage, *, ticker="TSLA", day=D1, version=VERSION):
    """A run that recorded a real logged mutation (a source-state write)."""

    with repository.stage_run(
        run_id=rid,
        stage=stage,
        trading_day=day,
        pipeline_version=version,
        ticker=ticker,
    ) as run:
        repository.record_source_state(
            f"probe:{rid}",
            run=run,
            checked_at=f"{day}T12:00:00+00:00",
            status="success",
            terminal=True,
        )


@pytest.mark.parametrize(
    "setup, expected",
    [
        (lambda r: None, "has no stories run_log row"),
        (lambda r: _logged_run(r, "forged", "themes"), "has no stories run_log row"),
        (lambda r: _logged_run(r, "forged", "stories", ticker="NVDA"), "covers NVDA"),
        (lambda r: _logged_run(r, "forged", "stories", day=D2), f"covers {D2}"),
        (
            lambda r: _logged_run(r, "forged", "stories", version="v9"),
            "pipeline version v9",
        ),
        (lambda r: _log(r, "forged", "stories"), "recorded no repository mutation"),
    ],
    ids=[
        "missing",
        "wrong-stage",
        "wrong-ticker",
        "wrong-day",
        "wrong-version",
        "admin-log",
    ],
)
def test_a_story_bound_to_the_wrong_run_is_unverified(repository, setup, expected):
    first, _ = _bound_partition(repository)
    summarize(repository, "TSLA", D1, first)
    setup(repository)
    raw_sql(repository, "UPDATE stories SET build_run_id = 'forged'")
    problems, _ = chain(repository, first)
    assert problems and all(expected in p for p in problems)


def test_a_replay_ingestion_run_does_not_establish_origin(repository):
    items = ingest(repository, "TSLA", D1, 1, replay=True)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    reconcile_themes(
        repository, "TSLA", D1, theme_payload(repository, "TSLA", D1, (1,))
    )
    [theme] = theme_ids(repository)
    summarize(repository, "TSLA", D1, theme)
    problems, _ = chain(repository, theme)
    assert problems and all("was a replay" in p for p in problems)


def test_an_ingestion_run_of_a_non_ingestion_stage_is_unverified(repository):
    items = ingest(repository, "TSLA", D1, 1, stage="some_other_stage")
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    reconcile_themes(
        repository, "TSLA", D1, theme_payload(repository, "TSLA", D1, (1,))
    )
    [theme] = theme_ids(repository)
    summarize(repository, "TSLA", D1, theme)
    problems, _ = chain(repository, theme)
    assert problems == [f"raw item {items[0]} was not inserted by an ingestion stage"]


def test_a_theme_build_bound_to_the_wrong_run_is_unverified(repository):
    _bound_partition(repository)
    _logged_run(repository, "forged", "themes", ticker="NVDA")
    raw_sql(repository, "UPDATE theme_sets SET build_run_id = 'forged'")
    problem = chain_build(repository)
    assert problem is not None and "covers NVDA" in problem


def test_a_summary_generation_pointing_at_no_run_is_unverified(repository):
    first, _ = _bound_partition(repository)
    summarize(repository, "TSLA", D1, first)
    raw_sql(repository, "UPDATE summary_generations SET run_id = 'nobody'")
    problems, _ = chain(repository, first)
    assert any("has no summaries run_log row" in p for p in problems)


RUN = {
    "run_id": "r",
    "stage": "stories",
    "ticker": "TSLA",
    "trading_day": D1,
    "pipeline_version": VERSION,
    "replay": False,
    "status": "success",
    "recorded_mutation": True,
}
STORY = {
    "story_id": 1,
    "cluster_fingerprint": "k",
    "ticker": "TSLA",
    "trading_day": D1,
    "pipeline_version": VERSION,
    "build_run_id": "r",
    "run": RUN,
}


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, None),
        ({"build_run_id": None}, "no run is recorded"),
        ({"run": None}, "has no stories run_log row"),
        ({"run": {**RUN, "run_id": "other"}}, "is not a stories run"),
        ({"run": {**RUN, "recorded_mutation": False}}, "recorded no repository"),
        ({"run": {**RUN, "ticker": None}}, "names no ticker"),
        ({"run": {**RUN, "status": "failed"}}, None),  # health is not provenance
        ({"run": "not facts"}, "has no stories run_log row"),
        ({"trading_day": D2}, f"not {D2}"),
        ({"ticker": "NVDA"}, "not NVDA"),
    ],
)
def test_story_hop_rules(change, expected):
    problem = provenance.verify_story({**STORY, **change})
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and expected in problem


def test_raw_item_hop_rules():
    run = {**RUN, "stage": "ingest_rss", "ticker": None}
    item = {
        "raw_item_id": 7,
        "ticker": None,
        "effective_day": D1,
        "ingest_run_id": "r",
        "ingest_stage": "ingest_rss",
        "run": run,
    }
    # A ticker-less RSS slice is a legitimate ingestion run.
    assert provenance.verify_raw_item(item) is None
    # A failed run still admitted the row: health is not provenance.
    assert (
        provenance.verify_raw_item({**item, "run": {**run, "status": "failed"}}) is None
    )
    assert "was a replay" in provenance.verify_raw_item(
        {**item, "run": {**run, "replay": True}}
    )
    assert "not inserted by an ingestion stage" in provenance.verify_raw_item(
        {**item, "ingest_stage": "stories", "run": {**run, "stage": "stories"}}
    )
    assert "covers NVDA" in provenance.verify_raw_item(
        {**item, "ticker": "TSLA", "run": {**run, "ticker": "NVDA"}}
    )
    assert "not " + D2 in provenance.verify_raw_item({**item, "effective_day": D2})
    assert "malformed" in provenance.verify_raw_item({"raw_item_id": 7})


# ----------------------------------------------------------------------
# H / I. Historical rows stay unverified, and cannot be blessed
# ----------------------------------------------------------------------


def _v16_database(tmp_path):
    target = tmp_path / "migrations_v16"
    target.mkdir()
    for migration in load_migrations(MIGRATIONS_PATH):
        if migration.version <= 16:
            shutil.copy(MIGRATIONS_PATH / migration.name, target / migration.name)
    database = tmp_path / "historical.db"
    old = Phase0Repository(database, migrations_path=target)
    old.migrate()
    old.admin.insert_raw_items([item_payload("TSLA", D1, 0)])
    return database


def test_a_historical_raw_item_migrates_unverified(tmp_path):
    database = _v16_database(tmp_path)
    upgraded = Phase0Repository(database)
    assert upgraded.migrate() == ["017_review_provenance_binding.sql"]
    row = one(upgraded, "SELECT * FROM raw_items")
    assert row["ingest_run_id"] is None and row["ingest_stage"] is None
    facts = {
        "raw_item_id": row["id"],
        "ticker": row["ticker"],
        "effective_day": D1,
        "ingest_run_id": None,
        "ingest_stage": None,
        "run": None,
    }
    assert "has no ingestion provenance" in provenance.verify_raw_item(facts)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE raw_items SET ingest_run_id = 'r', ingest_stage = 'fetch_yahoo'",
        "UPDATE raw_items SET ingest_run_id = 'r'",
    ],
)
def test_historical_raw_item_provenance_cannot_be_blessed(tmp_path, statement):
    upgraded = Phase0Repository(_v16_database(tmp_path))
    upgraded.migrate()
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        direct_sql(upgraded, statement)


@pytest.mark.parametrize(
    "statement, message",
    [
        ("UPDATE raw_items SET ingest_run_id = 'other'", "immutable"),
        ("UPDATE raw_items SET ingest_run_id = NULL, ingest_stage = NULL", "immutable"),
        ("UPDATE raw_items SET ingest_stage = 'ingest_rss'", "immutable"),
        ("UPDATE raw_items SET title = 'forged'", "immutable"),
        ("UPDATE raw_items SET description = 'forged'", "immutable"),
        ("UPDATE raw_items SET raw_json = '{}'", "immutable"),
    ],
)
def test_bound_raw_item_provenance_and_content_are_immutable(
    repository, statement, message
):
    ingest(repository, "TSLA", D1, 1)
    with pytest.raises(sqlite3.IntegrityError, match=message):
        direct_sql(repository, statement)


def test_classification_columns_of_a_bound_raw_item_stay_writable(repository):
    [item] = ingest(repository, "TSLA", D1, 1)
    direct_sql(repository, "UPDATE raw_items SET ingest_status = 'ambiguous'")
    repository.admin.update_raw_item_ticker(item, None)
    row = one(repository, "SELECT * FROM raw_items")
    assert row["ingest_status"] == "ambiguous" and row["ingest_run_id"]


def test_an_insert_cannot_carry_half_a_provenance(repository):
    # Past the authorization trigger, the pairing rule still holds.
    with pytest.raises(sqlite3.IntegrityError, match="both run and stage"):
        with repository.admin.connect_writable() as connection:
            connection.execute("DROP TRIGGER trg_raw_item_provenance_authorized")
            connection.execute(
                "INSERT INTO raw_items (source, canonical_url, fetched_at, raw_json, "
                "ingest_status, ingest_run_id) VALUES ('yahoo:x', 'u', ?, '{}', "
                "'invalid', 'r')",
                (f"{D1}T11:00:00+00:00",),
            )


def test_a_duplicate_ingest_keeps_the_first_runs_provenance(repository):
    payload = item_payload("TSLA", D1, 0)
    for rid in ("ingest-first", "ingest-second"):
        with repository.stage_run(
            run_id=rid,
            stage="fetch_yahoo",
            trading_day=D1,
            pipeline_version=VERSION,
            ticker="TSLA",
        ) as run:
            repository.ingest_raw_items([payload], run=run, terminal=True)
    assert one(repository, "SELECT ingest_run_id FROM raw_items")[0] == "ingest-first"


def test_a_logged_refetch_does_not_bless_an_admin_row(repository):
    payload = item_payload("TSLA", D1, 0)
    repository.admin.insert_raw_items([payload])
    with repository.stage_run(
        run_id="ingest-later",
        stage="fetch_yahoo",
        trading_day=D1,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run:
        [result] = repository.ingest_raw_items([payload], run=run, terminal=True)
    assert result.inserted is False
    assert one(repository, "SELECT ingest_run_id FROM raw_items")[0] is None


# ----------------------------------------------------------------------
# J / K. Theme bindings need a verified signature of a known format
# ----------------------------------------------------------------------


def test_a_theme_build_without_an_expected_signature_is_not_bound(repository):
    items = ingest(repository, "TSLA", D1, 2)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    reconcile_themes(
        repository,
        "TSLA",
        D1,
        theme_payload(repository, "TSLA", D1, (2,)),
        signature=None,
    )
    row = one(repository, "SELECT * FROM theme_sets")
    assert row["build_run_id"] is None and row["build_story_signature"] is None
    assert chain_build(repository) == "the theme set carries no build binding"


@pytest.mark.parametrize("version", [2, 0, 99])
def test_an_unknown_signature_version_fails_closed(repository, version):
    _bound_partition(repository)
    # Ordinary SQL cannot touch a binding at all (A4c authorization) ...
    with pytest.raises(sqlite3.IntegrityError):
        direct_sql(
            repository,
            "UPDATE theme_sets SET build_story_signature_version = ?",
            (version,),
        )
    if version <= 0:
        # ... and the column refuses a non-positive version outright, even
        # with every trigger dropped.
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            raw_sql(
                repository,
                "UPDATE theme_sets SET build_story_signature_version = ?",
                (version,),
            )
        return
    # Planted past the triggers, an unknown version still fails closed.
    raw_sql(
        repository,
        "UPDATE theme_sets SET build_story_signature_version = ?",
        (version,),
    )
    assert chain_build(repository) == (
        f"story-signature version {version!r} is not recognized"
    )


@pytest.mark.parametrize("version", [True, "1", 1.0, None])
def test_a_malformed_signature_version_in_facts_fails_closed(repository, version):
    _bound_partition(repository)
    facts = g2._theme_build_facts(repository.read.theme_population("TSLA", D1, VERSION))
    assert provenance.verify_theme_build(facts) is None
    assert (
        provenance.verify_theme_build(
            {**facts, "build_story_signature_version": version}
        )
        is not None
    )


def test_a_partial_theme_binding_is_refused_by_the_schema(repository):
    _bound_partition(repository)
    with pytest.raises(sqlite3.IntegrityError, match="all three fields"):
        # Past the authorization triggers, the completeness rule still holds.
        with repository.admin.connect_writable() as connection:
            connection.execute("DROP TRIGGER trg_theme_set_binding_authorized_update")
            connection.execute("UPDATE theme_sets SET build_story_signature = NULL")


def test_a_stale_signature_is_rebound_by_the_next_verified_build(repository):
    first, _ = _bound_partition(repository)
    raw_sql(repository, "UPDATE theme_sets SET build_story_signature = ?", ("0" * 64,))
    assert "no longer the partition's" in chain_build(repository)
    reconcile_themes(
        repository, "TSLA", D1, theme_payload(repository, "TSLA", D1, (2, 1))
    )
    assert chain_build(repository) is None


# ----------------------------------------------------------------------
# L. Identical reruns: nothing about content moves
# ----------------------------------------------------------------------


def test_an_identical_rerun_changes_no_content_and_keeps_the_first_binding(repository):
    items = ingest(repository, "TSLA", D1, 3)
    records = story_records(repository, "TSLA", D1, items)
    reconcile_stories(repository, "TSLA", D1, records, rid="stories-1")
    reconcile_themes(
        repository,
        "TSLA",
        D1,
        theme_payload(repository, "TSLA", D1, (2, 1)),
        rid="themes-1",
    )
    signature = repository.story_generation("TSLA", D1, VERSION).signature
    theme_rows = one(repository, "SELECT group_concat(id) FROM themes")[0]

    story_report = reconcile_stories(repository, "TSLA", D1, records, rid="stories-2")
    theme_report = reconcile_themes(
        repository,
        "TSLA",
        D1,
        theme_payload(repository, "TSLA", D1, (2, 1)),
        rid="themes-2",
    )

    assert story_report.counts["unchanged"] == 3
    assert story_report.counts["updated"] == story_report.counts["inserted"] == 0
    assert story_report.counts["invalidated_themes"] == 0
    assert theme_report.counts["updated"] == theme_report.counts["inserted"] == 0
    assert theme_report.counts["changed_outputs"] == 0
    assert repository.story_generation("TSLA", D1, VERSION).signature == signature
    assert one(repository, "SELECT group_concat(id) FROM themes")[0] == theme_rows
    # The binding follows the run that wrote the content, not the last to look.
    stored = repository.stories_for_day(D1, "TSLA")
    assert {r["build_run_id"] for r in stored} == {"stories-1"}
    assert one(repository, "SELECT build_run_id FROM theme_sets")[0] == "themes-1"
    assert chain_build(repository) is None
    for rid in ("stories-2", "themes-2"):
        [row] = repository.read.run_log_rows(run_id=rid)
        assert row["success_count"] == 0


def test_a_logged_rerun_binds_unbound_stories_without_counting_a_change(repository):
    items = ingest(repository, "TSLA", D1, 2)
    records = story_records(repository, "TSLA", D1, items)
    repository.admin.reconcile_stories(
        ticker="TSLA", trading_day=D1, pipeline_version=VERSION, stories=records
    )
    reconcile_themes(
        repository, "TSLA", D1, theme_payload(repository, "TSLA", D1, (2,))
    )
    signature = repository.story_generation("TSLA", D1, VERSION).signature
    report = reconcile_stories(repository, "TSLA", D1, records, rid="stories-bless")
    assert report.counts["unchanged"] == 2 and report.counts["updated"] == 0
    assert report.counts["invalidated_themes"] == 0
    assert repository.story_generation("TSLA", D1, VERSION).signature == signature
    assert {r["build_run_id"] for r in repository.stories_for_day(D1, "TSLA")} == {
        "stories-bless"
    }
    assert one(repository, "SELECT count(*) FROM theme_sets")[0] == 1
    [row] = repository.read.run_log_rows(run_id="stories-bless")
    assert row["success_count"] == 0 and row["partial_count"] == 2


def test_a_changed_story_is_rebound_to_the_run_that_changed_it(repository):
    items = ingest(repository, "TSLA", D1, 2)
    reconcile_stories(
        repository,
        "TSLA",
        D1,
        story_records(repository, "TSLA", D1, items),
        rid="stories-1",
    )
    reconcile_stories(
        repository,
        "TSLA",
        D1,
        story_records(repository, "TSLA", D1, items, title="new title"),
        rid="stories-2",
    )
    assert {r["build_run_id"] for r in repository.stories_for_day(D1, "TSLA")} == {
        "stories-2"
    }


# ----------------------------------------------------------------------
# M. Concurrent theme workers
# ----------------------------------------------------------------------


def test_a_stale_theme_worker_cannot_overwrite_the_winners_binding(repository):
    items = ingest(repository, "TSLA", D1, 2)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    stale = repository.story_generation("TSLA", D1, VERSION).signature
    # Another story run replaces the generation; the winner builds over it.
    reconcile_stories(
        repository,
        "TSLA",
        D1,
        story_records(repository, "TSLA", D1, items, title="gen two"),
    )
    reconcile_themes(
        repository,
        "TSLA",
        D1,
        theme_payload(repository, "TSLA", D1, (2,)),
        rid="themes-winner",
    )
    with pytest.raises(StoryGenerationConflict):
        reconcile_themes(
            repository,
            "TSLA",
            D1,
            theme_payload(repository, "TSLA", D1, (1, 1)),
            signature=stale,
            rid="themes-loser",
        )
    assert one(repository, "SELECT build_run_id FROM theme_sets")[0] == "themes-winner"
    assert chain_build(repository) is None


def test_two_workers_writing_the_same_output_keep_the_first_writer(repository):
    items = ingest(repository, "TSLA", D1, 2)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    signature = repository.story_generation("TSLA", D1, VERSION).signature
    payload = theme_payload(repository, "TSLA", D1, (2,))
    reconcile_themes(repository, "TSLA", D1, payload, signature=signature, rid="w-a")
    reconcile_themes(repository, "TSLA", D1, payload, signature=signature, rid="w-b")
    assert one(repository, "SELECT build_run_id FROM theme_sets")[0] == "w-a"


# ----------------------------------------------------------------------
# N / O. The artifact's producer is its accepted generation, and only that
# ----------------------------------------------------------------------


def test_a_cache_hit_keeps_the_original_generations_provenance(repository):
    first, _ = _bound_partition(repository)
    generated = summarize(repository, "TSLA", D1, first, rid="sum-original")
    client = Client()
    hit = summarize(repository, "TSLA", D1, first, client=client, rid="sum-cached")
    assert hit.source == SOURCE_CACHE_HIT and client.calls == 0
    assert hit.artifact.artifact_id == generated.artifact.artifact_id
    produced = repository.read.summary_artifact_provenance(hit.artifact.artifact_id)
    assert produced.generation.run_id == "sum-original"
    problems, _ = chain(repository, first)
    assert problems == []


def test_a_discarded_duplicate_is_never_the_producer(repository):
    first, _ = _bound_partition(repository)
    population = repository.read.theme_population("TSLA", D1, VERSION)
    generation_input = build_generation_input(population, first)
    client = Client()
    policy = resolve_generation_policy(client, max_attempts=PRODUCTION_MAX_ATTEMPTS)
    result = generate_guarded_summary(
        generation_input,
        client=client,
        max_attempts=PRODUCTION_MAX_ATTEMPTS,
        policy=policy,
    )
    recorded = []
    for rid in ("sum-first", "sum-duplicate"):
        with repository.stage_run(
            run_id=rid,
            stage="summaries",
            trading_day=D1,
            pipeline_version=VERSION,
            ticker="TSLA",
        ) as run:
            recorded.append(
                repository.persist_summary_generation(
                    run=run,
                    result=result,
                    generation_input=generation_input,
                    policy=policy,
                    terminal=True,
                )
            )
    assert [g.outcome for g in recorded] == ["accepted", "discarded_duplicate"]
    assert recorded[0].artifact_id == recorded[1].artifact_id
    produced = repository.read.summary_artifact_provenance(recorded[0].artifact_id)
    assert produced.generation.generation_id == recorded[0].generation_id
    assert produced.generation.run_id == "sum-first"


GENERATION = {
    "generation_id": 1,
    "run_id": "s",
    "artifact_id": 5,
    "outcome": "accepted",
    "ticker": "TSLA",
    "trading_day": D1,
    "pipeline_version": VERSION,
    "theme_id": 3,
    "input_fingerprint": "i" * 64,
    "policy_fingerprint": "p" * 64,
}
ARTIFACT = {
    k: GENERATION[k]
    for k in GENERATION
    if k not in ("generation_id", "run_id", "outcome")
}
SUMMARY_RUN = {**RUN, "run_id": "s", "stage": "summaries"}


@pytest.mark.parametrize(
    "generation, run, expected",
    [
        (GENERATION, SUMMARY_RUN, None),
        (None, None, "has no accepted generation"),
        (
            {**GENERATION, "outcome": "discarded_duplicate"},
            SUMMARY_RUN,
            "not an accepted",
        ),
        ({**GENERATION, "outcome": "discarded_stale"}, SUMMARY_RUN, "not an accepted"),
        ({**GENERATION, "outcome": "unavailable"}, SUMMARY_RUN, "not an accepted"),
        (
            {**GENERATION, "input_fingerprint": "x" * 64},
            SUMMARY_RUN,
            "input_fingerprint",
        ),
        ({**GENERATION, "artifact_id": 6}, SUMMARY_RUN, "artifact_id"),
        (GENERATION, {**SUMMARY_RUN, "stage": "themes"}, "not a summaries run"),
        (GENERATION, None, "has no summaries run_log row"),
    ],
)
def test_summary_hop_rules(generation, run, expected):
    problem = provenance.verify_summary(
        {"generation": generation, "run": run}, ARTIFACT
    )
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and expected in problem


# ----------------------------------------------------------------------
# P / Q / R. Manifests: /1 stays readable and unproven; /2 is re-derived
# ----------------------------------------------------------------------


def production_world(repository):
    """Two eligible days, every hop logged, one summarized theme each."""

    for day in (D1, D2):
        [theme] = production_partition(repository, "TSLA", day, groups=(2,))
        summarize(repository, "TSLA", day, theme)


def write_manifest(repository, tmp_path, name="g2.csv"):
    population = g2.load_sentence_population(
        repository.database_path, candidate_days=(D1, D2)
    )
    drawn = g2.sample_sentences(population, seed="s")
    manifest = g2.build_manifest(
        drawn, csv_name=name, generated_at=GENERATED_AT, code=CODE
    )
    return g2.write_sample(drawn, tmp_path / name, manifest=manifest)


def rebind(payload):
    """Recompute every outer digest, as a careful forger would."""

    snapshot = payload["snapshot"]
    snapshot["sha256"] = g2.snapshot_digest(snapshot["rows"], snapshot["artifacts"])
    payload["population"]["digest"] = g2.population_digest(
        payload["population"]["partitions"], snapshot["artifacts"]
    )
    payload["binding"] = {
        "manifest_id": g2.manifest_identity(payload),
        "snapshot_sha256": snapshot["sha256"],
    }


def rewrite(path, change, *, rebound=True):
    payload = json.loads(Path(path).read_text())
    change(payload)
    if rebound:
        rebind(payload)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def complete(blank, out):
    import csv

    with Path(blank).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            reviewer_id="alice", reviewed_at="2026-07-26", reviewer_verdict="supported"
        )
    with Path(out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=g2.SENTENCE_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return Path(out)


@pytest.fixture
def ratified(monkeypatch):
    protocol = review.Protocol(
        id="k3-g2-test",
        positive_verdict="supported",
        negative_verdict="unsupported",
        adjudicated_states=frozenset({review.AdjudicationState.UNANIMOUS}),
    )
    monkeypatch.setattr(g2, "RATIFIED_G2_PROTOCOLS", {"k3-g2-test": protocol})


def test_a_v2_manifest_records_facts_and_derives_verified_provenance(
    repository, tmp_path
):
    production_world(repository)
    _, manifest_path = write_manifest(repository, tmp_path)
    manifest = g2.read_manifest(manifest_path)
    assert manifest["schema"] == g2.MANIFEST_SCHEMA
    assert manifest["origin"]["status"] == "verified_live"
    for artifact in manifest["snapshot"]["artifacts"]:
        assert set(artifact["provenance"]) == {"summary", "stories", "raw_items"}
    reviewed = [
        p for p in manifest["population"]["partitions"] if p["reviewed_artifact_ids"]
    ]
    assert reviewed and all(p["generation_binding"] == "verified" for p in reviewed)
    assert g2.classify_g2_origin(manifest)[0] is review.OriginStatus.VERIFIED_LIVE
    assert g2.reviewed_bindings(manifest)[0] == ["verified"]


def _as_v1(payload):
    """What an A4b /1 manifest of the same round looked like."""

    payload["schema"] = g2.MANIFEST_SCHEMA_V1
    payload["origin"] = {
        "status": "unverified",
        "detail": review.UNVERIFIED_DETAIL,
    }
    for artifact in payload["snapshot"]["artifacts"]:
        del artifact["provenance"]
    for partition in payload["population"]["partitions"]:
        facts = partition.pop("theme_build")
        if partition["generation_binding"] is not None or facts is not None:
            partition["generation_binding"] = "unverified"


@pytest.mark.parametrize("claimed", ["unverified", "verified"])
def test_a_v1_manifest_is_readable_and_never_proven(
    repository, tmp_path, ratified, claimed
):
    production_world(repository)
    csv_path, manifest_path = write_manifest(repository, tmp_path)

    def downgrade(payload):
        _as_v1(payload)
        # A /1 file may *say* anything; it carries nothing to derive from.
        for partition in payload["population"]["partitions"]:
            if partition["generation_binding"] is not None:
                partition["generation_binding"] = claimed
        if claimed == "verified":
            payload["origin"]["status"] = "verified_live"

    rewrite(manifest_path, downgrade)
    # The sheet's binding columns were cut against the /2 identity; re-cut.
    payload = json.loads(manifest_path.read_text())
    blank = g2.render_csv(
        [g2.SentenceRow(**row) for row in payload["snapshot"]["rows"]],
        payload["binding"],
    )
    csv_path.write_text(blank)

    manifest = g2.read_manifest(manifest_path)
    assert manifest["schema"] == g2.MANIFEST_SCHEMA_V1
    origin, detail = g2.classify_g2_origin(manifest)
    assert origin is review.OriginStatus.UNVERIFIED and detail == g2.V1_ORIGIN_DETAIL
    assert g2.reviewed_bindings(manifest)[0] == ["unverified"]
    sheet = complete(csv_path, tmp_path / "alice.csv")
    card = g2.score_g2(g2.score_sentence_round(manifest, [sheet]))
    assert card.gate_eligible is False
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE
    assert any(b.startswith("origin is unverified") for b in card.eligibility_blockers)
    assert any("theme-set build provenance" in b for b in card.eligibility_blockers)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["snapshot"]["artifacts"][0]["provenance"]["raw_items"][0].update(
            ingest_run_id="forged"
        ),
        lambda p: p["population"]["partitions"][0].update(generation_binding="x"),
        lambda p: p["snapshot"]["artifacts"][0]["provenance"]["summary"]["run"].update(
            replay=True
        ),
    ],
)
def test_v2_provenance_edited_without_rebinding_is_refused(
    repository, tmp_path, change
):
    production_world(repository)
    _, manifest_path = write_manifest(repository, tmp_path)
    rewrite(manifest_path, change, rebound=False)
    with pytest.raises(review.ReviewSamplingError):
        g2.read_manifest(manifest_path)


def _first_artifact(payload):
    return payload["snapshot"]["artifacts"][0]


def _its_partition(payload):
    artifact = _first_artifact(payload)
    return next(
        p
        for p in payload["population"]["partitions"]
        if p["trading_day"] == artifact["trading_day"]
        and p["ticker"] == artifact["ticker"]
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda p: _first_artifact(p)["provenance"]["stories"].reverse(),
            "story facts are not its evidence",
        ),
        (
            lambda p: _first_artifact(p)["provenance"]["raw_items"].pop(),
            "raw-item facts are not its evidence's members",
        ),
        (
            lambda p: _first_artifact(p)["provenance"]["stories"][0].update(
                trading_day=D2 if _first_artifact(p)["trading_day"] == D1 else D1
            ),
            "outside its partition",
        ),
        (
            lambda p: _first_artifact(p)["provenance"]["raw_items"][0].update(
                effective_day="2026-01-01"
            ),
            "outside its partition's day",
        ),
        (
            lambda p: _its_partition(p)["theme_build"]["themes"][0][
                "member_keys"
            ].reverse(),
            "not its theme's recorded membership",
        ),
        (
            lambda p: _its_partition(p)["theme_build"].update(ticker="NVDA"),
            "another partition's",
        ),
        (
            # A binding claimed over facts that do not support it.
            lambda p: (
                _its_partition(p)["theme_build"].update(
                    build_story_signature_version=9
                ),
            ),
            "binding do not follow",
        ),
        (
            # Origin claimed over facts that do not support it.
            lambda p: (
                _first_artifact(p)["provenance"]["raw_items"][0].update(
                    ingest_run_id=None, ingest_stage=None, run=None
                ),
            ),
            "recorded origin does not follow",
        ),
        (
            lambda p: _first_artifact(p).pop("provenance"),
            "provenance facts are malformed",
        ),
        (
            lambda p: _its_partition(p).pop("theme_build"),
            "records no theme_build",
        ),
    ],
)
def test_v2_contradictions_are_refused_even_with_outer_digests_recomputed(
    repository, tmp_path, change, message
):
    production_world(repository)
    _, manifest_path = write_manifest(repository, tmp_path)
    rewrite(manifest_path, change)
    with pytest.raises(review.ReviewSamplingError, match=message):
        g2.read_manifest(manifest_path)


def test_a_forged_run_fact_is_derived_unverified_not_trusted(repository, tmp_path):
    """A consistent-looking manifest whose facts do not verify stays unverified.

    The forger rewrites a story's run facts to a themes-stage run, then
    re-derives the recorded origin so the manifest reads cleanly: it is
    accepted, and it is unverified, because the verdict is recomputed.
    """

    production_world(repository)
    _, manifest_path = write_manifest(repository, tmp_path)

    def forge(payload):
        story = _first_artifact(payload)["provenance"]["stories"][0]
        story["run"]["stage"] = "themes"
        status, detail = g2.classify_g2_origin(payload)
        payload["origin"] = {"status": status.value, "detail": detail}

    rewrite(manifest_path, forge)
    manifest = g2.read_manifest(manifest_path)
    origin, detail = g2.classify_g2_origin(manifest)
    assert origin is review.OriginStatus.UNVERIFIED and "is not a stories run" in detail


def test_an_admin_ingested_round_samples_as_unverified(repository, tmp_path):
    for day in (D1, D2):
        [theme] = production_partition(repository, "TSLA", day, groups=(2,), admin=True)
        summarize(repository, "TSLA", day, theme)
    _, manifest_path = write_manifest(repository, tmp_path)
    manifest = g2.read_manifest(manifest_path)
    assert manifest["origin"]["status"] == "unverified"
    assert "has no ingestion provenance" in manifest["origin"]["detail"]
    # The theme builds were logged and verified: the two facts stay apart.
    assert g2.reviewed_bindings(manifest)[0] == ["verified"]


def test_review_sampling_writes_nothing_to_the_database(repository, tmp_path):
    production_world(repository)
    path = repository.database_path

    def dump():
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = [
                r[0]
                for r in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            return {
                t: sorted(map(repr, connection.execute(f'SELECT * FROM "{t}"')))
                for t in tables
            }
        finally:
            connection.close()

    before = dump()
    write_manifest(repository, tmp_path)
    assert dump() == before


# ----------------------------------------------------------------------
# T. The trust boundary, stated as a test
# ----------------------------------------------------------------------


def test_a_raw_sqlite_forger_can_fabricate_verified_provenance(repository):
    """Documented boundary, not a defect.

    Someone with raw write access to the database file can insert a
    ``run_log`` row with a mutation marker and point every binding at it,
    with the triggers dropped.  Nothing relational distinguishes that from
    the pipeline, and A4c does not claim otherwise: it proves linkage
    through repository-controlled logged writes and deterministic content
    digests, not cryptographic authenticity.
    """

    first, _ = production_partition(repository, admin=True)
    summarize(repository, "TSLA", D1, first)
    assert chain(repository, first)[0]  # honestly unverified
    raw_sql(
        repository,
        "INSERT INTO run_log (run_id, stage, counts, duration_ms, errors, "
        "started_at, completed_at, status, trading_day, pipeline_version, ticker, "
        "success_count, partial_count, failure_count, attempt, replay, "
        "last_mutation_id) VALUES ('forged', 'fetch_yahoo', '{}', 0, '[]', ?, ?, "
        "'success', ?, ?, 'TSLA', 1, 0, 0, 1, 0, ?)",
        (f"{D1}T12:00:00+00:00", f"{D1}T12:00:01+00:00", D1, VERSION, "f" * 32),
    )
    raw_sql(
        repository,
        "UPDATE raw_items SET ingest_run_id = 'forged', ingest_stage = 'fetch_yahoo'",
    )
    problems, build = chain(repository, first)
    assert problems == [] and build is None


# ======================================================================
# Codex review repairs
# ======================================================================
#
# P1-a: possession of a valid run id is not authority to write a binding.
# Only the repository's logged mutation paths hold the per-connection grant
# migration 017's ``*_authorized*`` triggers ask for.


def test_attack_a_restoring_a_borrowed_story_binding_is_refused(repository):
    """Codex attack A, exactly: mutate, restore the saved id, continue."""

    first, _ = production_partition(repository)
    story_id, saved = one(
        repository, "SELECT id, build_run_id FROM stories ORDER BY id LIMIT 1"
    )
    assert saved is not None
    direct_sql(
        repository,
        "UPDATE stories SET canonical_title = 'forged title' WHERE id = ?",
        (story_id,),
    )
    assert (
        one(repository, "SELECT build_run_id FROM stories WHERE id = ?", (story_id,))[0]
        is None
    )
    with pytest.raises(sqlite3.IntegrityError, match="logged story run"):
        direct_sql(
            repository,
            "UPDATE stories SET build_run_id = ? WHERE id = ?",
            (saved, story_id),
        )
    # The normal logged theme and summary pipeline carries on over it.
    reconcile_themes(
        repository, "TSLA", D1, theme_payload(repository, "TSLA", D1, (2, 1))
    )
    [theme, _] = theme_ids(repository)
    summarize(repository, "TSLA", D1, theme)
    problems, build = chain(repository, theme)
    assert build is None  # the theme build over the forged story is logged...
    assert any(f"story {story_id}: no run is recorded" in p for p in problems)


def test_a1_a_direct_raw_item_insert_cannot_borrow_ingest_provenance(repository):
    [item] = ingest(repository, "TSLA", D1, 1, rid="ingest-legit")
    payload = item_payload("TSLA", D1, 5)
    with pytest.raises(sqlite3.IntegrityError, match="logged ingestion run"):
        direct_sql(
            repository,
            "INSERT INTO raw_items (source, ticker, title, description, url, "
            "canonical_url, published_at, fetched_at, raw_json, ingest_run_id, "
            "ingest_stage) VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', 'ingest-legit', "
            "'fetch_yahoo')",
            tuple(
                payload[k]
                for k in (
                    "source",
                    "ticker",
                    "title",
                    "description",
                    "url",
                    "canonical_url",
                    "published_at",
                    "fetched_at",
                )
            ),
        )
    assert one(repository, "SELECT count(*) FROM raw_items")[0] == 1 and item


def test_a3_copying_a_legitimate_theme_binding_is_refused(repository):
    _bound_partition(repository)
    saved = tuple(
        one(
            repository,
            "SELECT build_run_id, build_story_signature, "
            "build_story_signature_version FROM theme_sets",
        )
    )
    direct_sql(repository, "UPDATE theme_sets SET method_reason = 'edited'")
    assert chain_build(repository) == "the theme set carries no build binding"
    with pytest.raises(sqlite3.IntegrityError, match="logged theme run"):
        direct_sql(
            repository,
            "UPDATE theme_sets SET build_run_id = ?, build_story_signature = ?, "
            "build_story_signature_version = ?",
            saved,
        )
    assert chain_build(repository) == "the theme set carries no build binding"


def test_a4_delete_and_reinsert_carrying_copied_provenance_is_refused(repository):
    ingest(repository, "TSLA", D1, 1, rid="ingest-legit")
    row = dict(one(repository, "SELECT * FROM raw_items"))
    direct_sql(repository, "DELETE FROM raw_items WHERE id = ?", (row["id"],))
    columns = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    with pytest.raises(sqlite3.IntegrityError, match="logged ingestion run"):
        direct_sql(
            repository,
            f"INSERT INTO raw_items ({columns}) VALUES ({marks})",
            tuple(row.values()),
        )
    # And a story carried back in the same way.
    items = ingest(repository, "TSLA", D1, 1)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    story = dict(one(repository, "SELECT * FROM stories"))
    assert story["build_run_id"] is not None
    direct_sql(repository, "UPDATE stories SET canonical_item_id = NULL")
    direct_sql(repository, "DELETE FROM story_members")
    direct_sql(repository, "DELETE FROM stories")
    columns = ", ".join(story)
    marks = ", ".join("?" for _ in story)
    with pytest.raises(sqlite3.IntegrityError, match="logged story run"):
        direct_sql(
            repository,
            f"INSERT INTO stories ({columns}) VALUES ({marks})",
            tuple(story.values()),
        )


def test_a5_a_concurrent_connection_cannot_borrow_the_grant(repository):
    from phase0.repository import (
        PROVENANCE_GRANT_FUNCTION,
        _trusted_provenance_write,
    )

    items = ingest(repository, "TSLA", D1, 1)
    reconcile_stories(
        repository, "TSLA", D1, story_records(repository, "TSLA", D1, items)
    )
    probe = f"SELECT {PROVENANCE_GRANT_FUNCTION}()"
    with repository.stage_run(
        run_id="grant-holder",
        stage="stories",
        trading_day=D1,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run:
        with repository.admin.connect_writable() as holder:
            with repository.admin.connect_writable() as other:
                with _trusted_provenance_write(holder, run):
                    assert holder.execute(probe).fetchone()[0] == 1
                    assert other.execute(probe).fetchone()[0] == 0
                    with pytest.raises(sqlite3.IntegrityError, match="logged story"):
                        other.execute("UPDATE stories SET build_run_id = 'stolen'")
                assert holder.execute(probe).fetchone()[0] == 0


def test_a5_the_grant_is_off_after_a_failure_and_not_reentrant(repository):
    from phase0.repository import (
        PROVENANCE_GRANT_FUNCTION,
        Phase0RunContextError,
        _trusted_provenance_write,
    )

    probe = f"SELECT {PROVENANCE_GRANT_FUNCTION}()"
    with repository.stage_run(
        run_id="grant-fail",
        stage="stories",
        trading_day=D1,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run:
        with repository.admin.connect_writable() as connection:
            with pytest.raises(RuntimeError):
                with _trusted_provenance_write(connection, run):
                    raise RuntimeError("boom")
            assert connection.execute(probe).fetchone()[0] == 0
            with _trusted_provenance_write(connection, run):
                with pytest.raises(Phase0RunContextError, match="already open"):
                    with _trusted_provenance_write(connection, run):
                        pass
            # Knowing a run id is not a run.
            with pytest.raises(Phase0RunContextError, match="logged stage run"):
                with _trusted_provenance_write(connection, "grant-fail"):
                    pass


def test_a5_a_bare_connection_cannot_write_provenance_at_all(repository):
    ingest(repository, "TSLA", D1, 1, rid="ingest-legit")
    connection = sqlite3.connect(repository.database_path)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function"):
            connection.execute(
                "INSERT INTO raw_items (source, canonical_url, fetched_at, raw_json, "
                "ingest_run_id, ingest_stage) VALUES ('yahoo:x', 'u', ?, '{}', "
                "'ingest-legit', 'fetch_yahoo')",
                (f"{D1}T11:00:00+00:00",),
            )
    finally:
        connection.close()


def test_a6_authorized_paths_still_bind_every_hop(repository):
    first, _ = production_partition(repository)
    summarize(repository, "TSLA", D1, first)
    assert chain(repository, first) == ([], None)


def test_a7_admin_writes_still_withhold_and_clear(repository):
    first, _ = production_partition(repository)
    repository.admin.reconcile_stories(
        ticker="TSLA",
        trading_day=D1,
        pipeline_version=VERSION,
        stories=story_records(
            repository,
            "TSLA",
            D1,
            [r["id"] for r in repository.raw_items_for_day(D1, "TSLA")],
            title="admin",
        ),
    )
    assert {r["build_run_id"] for r in repository.stories_for_day(D1, "TSLA")} == {None}
    assert one(repository, "SELECT count(*) FROM theme_sets")[0] == 0  # invalidated


# P1-b: every fact a partition's verdict rests on comes from ONE snapshot.
#
# The sampler reads each partition inside ``Phase0Reader.review_snapshot()``:
# one read-only connection, one read transaction, its snapshot established
# before the first partition read.  Concurrent commits are not seen by that
# pass.  Hooks below commit from other connections at exact points inside
# the read, so nothing here depends on timing.


def _unsummarized_partition(repository, day=D1, rid_prefix=""):
    items = ingest(repository, "TSLA", day, 2, rid=f"{rid_prefix}ingest-{day}")
    reconcile_stories(
        repository, "TSLA", day, story_records(repository, "TSLA", day, items)
    )
    reconcile_themes(
        repository, "TSLA", day, theme_payload(repository, "TSLA", day, (2,))
    )
    [theme] = theme_ids(repository, "TSLA", day)
    return theme


def _read(repository, day=D1):
    return g2._read_partition(
        repository.read, "TSLA", day, VERSION, production_generation_policy()
    )


def _hook_current(monkeypatch, action):
    """Run ``action`` (committing on another connection) once, inside the read."""

    real = g2.current_summary_artifact
    fired = []

    def hooked(snapshot, *args, **kwargs):
        if not fired:
            fired.append(snapshot)
            action()
        return real(snapshot, *args, **kwargs)

    monkeypatch.setattr(g2, "current_summary_artifact", hooked)
    return fired


def _problems(record, artifact):
    return g2.artifact_origin_problems(artifact, record["theme_build"])


# -- The Codex regression: two never-valid states, alternated -----------------


def _attack_world(repository):
    """D1: a summary whose chain has never been valid; D2: a clean partition.

    The D1 summary is generated while its ingestion run reads as a replay.
    State A repairs the ingestion run and breaks the producer (its summaries
    run names NVDA); state B repairs the producer and breaks the ingestion
    run again.  Each half is valid in exactly one state; no state holds both.
    """

    theme = _unsummarized_partition(repository, D1)
    direct_sql(
        repository, "UPDATE run_log SET replay = 1 WHERE run_id = 'ingest-%s'" % D1
    )
    summarize(repository, "TSLA", D1, theme, rid="sum-attacked")
    clean = _unsummarized_partition(repository, D2)
    summarize(repository, "TSLA", D2, clean, rid="sum-clean")
    return theme


def _set_state(repository, state):
    replay, ticker = (0, "NVDA") if state == "A" else (1, "TSLA")
    with repository.admin.connect_writable(immediate=True) as connection:
        connection.execute(
            "UPDATE run_log SET replay = ? WHERE run_id = ? AND stage = 'fetch_yahoo'",
            (replay, f"ingest-{D1}"),
        )
        connection.execute(
            "UPDATE run_log SET ticker = ? WHERE run_id = 'sum-attacked' "
            "AND stage = 'summaries'",
            (ticker,),
        )


def _alternate_on_every_read(monkeypatch, repository, start):
    """Commit the *other* state after every single partition read."""

    from phase0.repository import ReviewSnapshot

    state = {"now": start, "flips": 0}

    def flip():
        state["now"] = "B" if state["now"] == "A" else "A"
        state["flips"] += 1
        _set_state(repository, state["now"])

    for name in ("theme_population", "summary_artifact", "summary_artifact_provenance"):
        real = getattr(ReviewSnapshot, name)

        def wrapped(self, *args, _real=real, **kwargs):
            result = _real(self, *args, **kwargs)
            flip()
            return result

        monkeypatch.setattr(ReviewSnapshot, name, wrapped)
    return state


def test_neither_attack_state_holds_a_valid_chain(repository):
    theme = _attack_world(repository)
    _set_state(repository, "A")
    problems, _ = chain(repository, theme)
    assert any("covers NVDA" in p for p in problems)
    assert not any("was a replay" in p for p in problems)
    _set_state(repository, "B")
    problems, _ = chain(repository, theme)
    assert any("was a replay" in p for p in problems)
    assert not any("covers NVDA" in p for p in problems)


def test_composing_reads_from_two_states_would_falsely_verify(repository):
    """The attack is real: population from A plus producer from B verifies.

    This is what separate per-read snapshots allowed.  The sampler no longer
    composes reads that way; the next tests show it cannot.
    """

    theme = _attack_world(repository)
    reader = repository.read
    _set_state(repository, "A")
    population = reader.theme_population("TSLA", D1, VERSION)
    current = current_summary_artifact(
        reader, "TSLA", D1, VERSION, theme, production_generation_policy()
    )
    _set_state(repository, "B")
    producer = reader.summary_artifact_provenance(current.artifact.artifact_id)
    record = {
        **g2._artifact_record(current, 1),
        "provenance": g2._artifact_provenance(population, current, producer),
    }
    assert g2.artifact_origin_problems(record, g2._theme_build_facts(population)) == []


@pytest.mark.parametrize("start", ["A", "B"])
def test_alternating_states_never_yield_a_verified_artifact(
    repository, monkeypatch, start
):
    theme = _attack_world(repository)
    _set_state(repository, start)
    state = _alternate_on_every_read(monkeypatch, repository, start)
    record, artifacts = _read(repository, D1)
    assert state["flips"] >= 3  # population, artifact, producer -- all flipped
    assert record["outcome"] == g2.PARTITION_ENUMERATED
    [artifact] = artifacts
    assert artifact["theme_id"] == theme
    problems = _problems(record, artifact)
    expected = "covers NVDA" if start == "A" else "was a replay"
    other = "was a replay" if start == "A" else "covers NVDA"
    # One coherent state: exactly the starting state's broken half, never
    # the other state's repair of it.
    assert problems and all(expected in p for p in problems)
    assert not any(other in p for p in problems)


@pytest.mark.parametrize("start", ["A", "B"])
def test_alternating_states_never_write_a_verified_manifest(
    repository, monkeypatch, tmp_path, start
):
    _attack_world(repository)
    _set_state(repository, start)
    _alternate_on_every_read(monkeypatch, repository, start)
    _, manifest_path = write_manifest(repository, tmp_path)
    manifest = g2.read_manifest(manifest_path)
    assert manifest["schema"] == g2.MANIFEST_SCHEMA
    assert manifest["origin"]["status"] == "unverified"
    origin, detail = g2.classify_g2_origin(manifest)
    assert origin is review.OriginStatus.UNVERIFIED
    attacked = [a for a in manifest["snapshot"]["artifacts"] if a["trading_day"] == D1]
    assert attacked
    build = next(
        p["theme_build"]
        for p in manifest["population"]["partitions"]
        if (p["trading_day"], p["ticker"]) == (D1, "TSLA")
    )
    assert g2.artifact_origin_problems(attacked[0], build)


# -- Snapshot semantics ---------------------------------------------------------


def test_s1_a_stable_valid_partition_samples_verified(repository):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)
    record, [artifact] = _read(repository)
    assert record["outcome"] == g2.PARTITION_ENUMERATED
    assert record["generation_binding"] == "verified"
    assert _problems(record, artifact) == []


def test_s2_a_summary_committed_during_the_read_is_not_seen(repository, monkeypatch):
    theme = _unsummarized_partition(repository)
    _hook_current(monkeypatch, lambda: summarize(repository, "TSLA", D1, theme))
    record, artifacts = _read(repository)
    assert record["outcome"] == g2.PARTITION_ENUMERATED
    assert artifacts == []
    assert [t["outcome"] for t in record["themes"]] == [g2.THEME_NO_CURRENT]
    monkeypatch.undo()
    record, [artifact] = _read(repository)  # the next pass sees it
    assert _problems(record, artifact) == []


def test_s3_breaking_ingestion_during_the_read_is_not_seen(repository, monkeypatch):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)
    _hook_current(
        monkeypatch,
        lambda: direct_sql(
            repository, f"UPDATE run_log SET replay = 1 WHERE run_id = 'ingest-{D1}'"
        ),
    )
    record, [artifact] = _read(repository)
    assert _problems(record, artifact) == []  # the snapshot it opened
    assert all(r["run"]["replay"] is False for r in artifact["provenance"]["raw_items"])
    monkeypatch.undo()
    record, [artifact] = _read(repository)
    assert all("was a replay" in p for p in _problems(record, artifact))


def test_s4_repairing_ingestion_during_the_read_is_not_seen(repository, monkeypatch):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)
    direct_sql(
        repository, f"UPDATE run_log SET replay = 1 WHERE run_id = 'ingest-{D1}'"
    )
    _hook_current(
        monkeypatch,
        lambda: direct_sql(
            repository, f"UPDATE run_log SET replay = 0 WHERE run_id = 'ingest-{D1}'"
        ),
    )
    record, [artifact] = _read(repository)
    problems = _problems(record, artifact)
    assert problems and all("was a replay" in p for p in problems)
    monkeypatch.undo()
    record, [artifact] = _read(repository)
    assert _problems(record, artifact) == []


@pytest.mark.parametrize(
    "statement, check",
    [
        (
            "UPDATE run_log SET pipeline_version = 'v9' WHERE stage = 'stories'",
            lambda record, artifact: all(
                s["run"]["pipeline_version"] == VERSION
                for s in artifact["provenance"]["stories"]
            ),
        ),
        (
            "UPDATE run_log SET ticker = 'NVDA' WHERE stage = 'themes'",
            lambda record, artifact: record["theme_build"]["run"]["ticker"] == "TSLA",
        ),
        (
            "UPDATE run_log SET ticker = 'NVDA' WHERE stage = 'summaries'",
            lambda record, artifact: artifact["provenance"]["summary"]["run"]["ticker"]
            == "TSLA",
        ),
    ],
    ids=["s5-story-run", "s6-theme-run", "s7-summary-producer"],
)
def test_s5_s6_s7_concurrent_run_mutations_are_not_mixed_in(
    repository, monkeypatch, statement, check
):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)
    _hook_current(monkeypatch, lambda: direct_sql(repository, statement))
    record, [artifact] = _read(repository)
    assert check(record, artifact)  # every fact is the pre-change state's
    assert record["generation_binding"] == "verified"
    assert _problems(record, artifact) == []
    monkeypatch.undo()
    record, [artifact] = _read(repository)  # the next pass: wholly after
    assert not check(record, artifact)


def test_s6_a_concurrent_theme_rebuild_is_not_mixed_in(repository, monkeypatch):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)

    def rebuild():
        payload = theme_payload(repository, "TSLA", D1, (1, 1))
        reconcile_themes(repository, "TSLA", D1, payload, rid="themes-rebuild")

    _hook_current(monkeypatch, rebuild)
    record, [artifact] = _read(repository)
    assert [t["theme_id"] for t in record["theme_build"]["themes"]] == [theme]
    assert record["theme_build"]["build_run_id"] != "themes-rebuild"
    assert _problems(record, artifact) == []


def _checkpoint_busy(repository):
    with repository.admin.connect_writable() as connection:
        return connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]


def test_s8_s9_an_exception_mid_read_releases_the_snapshot(repository, monkeypatch):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)

    def boom():
        raise RuntimeError("mid-read failure")

    seen = _hook_current(monkeypatch, boom)
    with pytest.raises(RuntimeError, match="mid-read failure"):
        _read(repository)
    [snapshot] = seen
    with pytest.raises(Exception, match="closed"):
        snapshot.theme_population("TSLA", D1, VERSION)
    # No reader is left holding an old snapshot: a full checkpoint succeeds.
    direct_sql(repository, "UPDATE run_log SET status = status")
    assert _checkpoint_busy(repository) == 0
    monkeypatch.undo()
    record, [artifact] = _read(repository)  # S9: sampling works normally
    assert _problems(record, artifact) == []


def test_s10_a_wal_writer_commits_while_the_snapshot_holds_its_state(repository):
    theme = _unsummarized_partition(repository)
    policy = production_generation_policy()
    assert one(repository, "PRAGMA journal_mode")[0] == "wal"
    with repository.read.review_snapshot() as snapshot:
        before = snapshot.theme_population("TSLA", D1, VERSION)
        # A writer commits while the read transaction is open.
        summarize(repository, "TSLA", D1, theme)
        assert repository.read.summary_generations("TSLA", D1, VERSION)
        # ...and is busy-free for writers, but this snapshot still sees none.
        assert (
            current_summary_artifact(snapshot, "TSLA", D1, VERSION, theme, policy)
            is None
        )
        assert snapshot.theme_population("TSLA", D1, VERSION) == before
        # A full checkpoint cannot pass a reader still on the old snapshot.
        assert _checkpoint_busy(repository) == 1
    assert _checkpoint_busy(repository) == 0
    with repository.read.review_snapshot() as snapshot:
        assert current_summary_artifact(snapshot, "TSLA", D1, VERSION, theme, policy)


def test_b6_an_unrelated_run_log_change_does_not_disturb_the_read(
    repository, monkeypatch
):
    theme = _unsummarized_partition(repository)
    summarize(repository, "TSLA", D1, theme)
    ingest(repository, "NVDA", D1, 1, rid="ingest-nvda")
    _hook_current(
        monkeypatch,
        lambda: direct_sql(
            repository, "UPDATE run_log SET replay = 1 WHERE run_id = 'ingest-nvda'"
        ),
    )
    record, artifacts = _read(repository)
    assert record["outcome"] == g2.PARTITION_ENUMERATED and len(artifacts) == 1


def test_a_review_snapshot_cannot_be_built_outside_the_reader():
    from phase0.repository import ReviewSnapshot

    with pytest.raises(Exception, match="review_snapshot"):
        ReviewSnapshot(object(), None)


# P2: a signature version is an exact stored INTEGER or nothing.


def test_attack_c_a_real_version_is_refused_by_the_schema(repository):
    _bound_partition(repository)
    with repository.admin.connect_writable() as connection:
        connection.execute("DROP TRIGGER trg_theme_set_binding_authorized_update")
        with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
            connection.execute(
                "UPDATE theme_sets SET build_story_signature_version = 1.5"
            )


def test_attack_c_a_planted_real_version_fails_closed_through_the_reader(repository):
    """Database -> reader -> verifier: REAL 1.5 is never read as version 1."""

    _bound_partition(repository)
    with repository.admin.connect_writable() as connection:
        create = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE name = 'trg_theme_set_binding_authorized_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER trg_theme_set_binding_authorized_update")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE theme_sets SET build_story_signature_version = 1.5")
        connection.execute("PRAGMA ignore_check_constraints = OFF")
        connection.execute(create)
    assert (
        one(repository, "SELECT typeof(build_story_signature_version) FROM theme_sets")[
            0
        ]
        == "real"
    )
    theme_set = repository.read.theme_population("TSLA", D1, VERSION).theme_set
    assert theme_set.build_story_signature_version == 1.5
    assert chain_build(repository) == "story-signature version 1.5 is not recognized"


@pytest.mark.parametrize("value", [1.0, "1", True])
def test_whole_valued_versions_are_stored_as_the_exact_integer(repository, value):
    """INTEGER affinity stores these losslessly as integer 1 -- not a coercion
    by any reader, and nothing else reaches the column."""

    _bound_partition(repository)
    with repository.admin.connect_writable() as connection:
        connection.execute("DROP TRIGGER trg_theme_set_binding_authorized_update")
        # Off 1 first: rewriting the stored value to itself is a write that
        # leaves the binding as it was, which (rightly) clears it.
        connection.execute("UPDATE theme_sets SET build_story_signature_version = 2")
        connection.execute(
            "UPDATE theme_sets SET build_story_signature_version = ?", (value,)
        )
    row = one(
        repository,
        "SELECT build_story_signature_version, "
        "typeof(build_story_signature_version) FROM theme_sets",
    )
    assert tuple(row) == (1, "integer")


# Optional P3: a golden vector for story-signature format 1.


GOLDEN_STORY_SIGNATURE_V1 = (
    "640f74db61bae90d5bdddc7341ab6902194656c59eb7577b0dcb3a4da31234b4"
)


def test_story_signature_format_1_golden_vector(tmp_path):
    """If this moves, the signature encoding moved: bump STORY_SIGNATURE_VERSION."""

    repo = Phase0Repository(tmp_path / "golden.db", clock=Clock())
    repo.migrate()
    item = {
        "source": "yahoo:Reuters",
        "ticker": "TSLA",
        "title": "Golden headline",
        "description": "Golden standfirst.",
        "url": "https://reuters.example/golden",
        "canonical_url": "https://reuters.example/golden",
        "published_at": f"{D1}T10:00:00+00:00",
        "fetched_at": f"{D1}T11:00:00+00:00",
        "raw_json": {"golden": True},
    }
    with repo.stage_run(
        run_id="golden-ingest",
        stage="fetch_yahoo",
        trading_day=D1,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run:
        [result] = repo.ingest_raw_items([item], run=run, terminal=True)
    reconcile_stories(
        repo,
        "TSLA",
        D1,
        story_records(repo, "TSLA", D1, [result.item_id]),
        rid="golden-stories",
    )
    assert provenance.STORY_SIGNATURE_VERSION == 1
    assert repo.story_generation("TSLA", D1, VERSION).signature == (
        GOLDEN_STORY_SIGNATURE_V1
    )


def test_closed_connections_leave_no_grant_behind(repository):
    from phase0.repository import _GRANTS

    before = len(_GRANTS)
    first, _ = production_partition(repository)
    summarize(repository, "TSLA", D1, first)
    with pytest.raises(StoryGenerationConflict):
        reconcile_themes(
            repository,
            "TSLA",
            D1,
            theme_payload(repository, "TSLA", D1, (2, 1)),
            signature="0" * 64,
        )
    assert len(_GRANTS) == before

"""B1: the narrative read API over persisted SQLite state.

Every database here is written through the real Phase 0 write paths --
story and theme reconciliation, and A3's ``ensure_summary`` with a fake
provider -- and read back through the public routes.  The read path itself
is never mocked except where a test says why.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import itertools
import os
import re
import socket
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

import ai.guarded_summary as guarded  # noqa: E402
import ai.summarization as summarization  # noqa: E402
import phase0.repository as phase0_repository  # noqa: E402
import phase0.summary_lifecycle as lifecycle  # noqa: E402
import phase0.summary_runner as summary_runner  # noqa: E402
from nlp.dedup.selection import cluster_fingerprint_for  # noqa: E402
from phase0.models import (  # noqa: E402
    ExcludedStoryRecord,
    OtherCoverageRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import DEFAULT_DATABASE_PATH, Phase0Repository  # noqa: E402
from phase0.summary_lifecycle import (  # noqa: E402
    SOURCE_GENERATED,
    current_summary_artifact,
    ensure_summary,
)
from phase0.summary_runner import (  # noqa: E402
    PRODUCTION_MAX_ATTEMPTS,
    production_generation_policy,
)

app = importlib.import_module("main").app
config = importlib.import_module("app.config")
routes = importlib.import_module("app.phase0.routes")
narrative = importlib.import_module("app.phase0.repository")
sqlite_module = importlib.import_module("app.phase0.sqlite_repository")
schemas = importlib.import_module("app.phase0.schemas")
SqliteNarrativeRepository = sqlite_module.SqliteNarrativeRepository

DAY = "2026-07-23"
EARLIER = "2026-07-22"
VERSION = "v1"
#: 11:00 ET on a Thursday: inside the market session.
NOW = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
OUTLETS = ("Reuters", "Bloomberg", "CNBC")
UNIVERSE = ["TSLA", "NVDA", "AMD", "AAPL", "META"]
GENERATED_LABEL = "Coverage summary"
SUMMARY_TABLES = (
    "summary_artifacts",
    "summary_sentences",
    "summary_sentence_citations",
    "summary_generations",
    "summary_generation_attempts",
)
_ITEMS = itertools.count(1)
_RUNS = itertools.count(1)
ID_LINE_RE = re.compile(r"- id: (\S+)")


# ----------------------------------------------------------------------
# A persisted world
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_ambient_provider_config(monkeypatch):
    """The API resolves the production policy from GEMINI_*; start from none."""

    for name in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_TIMEOUT_MS",
        "PHASE0_NARRATIVE_SOURCE",
        "PHASE0_DATABASE_PATH",
        "PHASE0_PIPELINE_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


class Clock:
    def __init__(self) -> None:
        self.now = NOW - timedelta(minutes=30)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now = self.now + timedelta(minutes=minutes)


@dataclasses.dataclass
class World:
    path: Path
    repository: Phase0Repository
    clock: Clock


@pytest.fixture
def world(tmp_path) -> World:
    clock = Clock()
    repository = Phase0Repository(tmp_path / "phase0.db", clock=clock)
    repository.migrate()
    return World(tmp_path / "phase0.db", repository, clock)


class Client:
    """A fake provider with the production client's model and output cap.

    The first sentence cites every evidence story in *reverse* order, so a
    reader that re-sorted citations would be caught.
    """

    def __init__(self, *, model=summarization.DEFAULT_MODEL) -> None:
        self.model = model
        self.max_output_tokens = summarization.DEFAULT_MAX_OUTPUT_TOKENS
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls += 1
        ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": GENERATED_LABEL,
                "sentences": [
                    {
                        "text": "Coverage leads with this story.",
                        "citation_ids": list(reversed(ids)),
                    },
                    {"text": "Outlets repeat the report.", "citation_ids": ids[:1]},
                ],
            }
        )


@dataclasses.dataclass
class Seeded:
    theme_ids: dict[str, int]
    theme_keys: dict[str, str]
    #: label -> member story ids; "other" and "excluded" too.
    story_ids: dict[str, list[int]]


def _insert_item(repository, ticker, day, outlet, published_at):
    index = next(_ITEMS)
    [result] = repository.admin.insert_raw_items(
        [
            {
                "source": f"yahoo:{outlet}",
                "ticker": ticker,
                "title": f"{outlet} headline {index}",
                "description": f"{outlet} standfirst {index}.",
                "url": f"https://{outlet.lower()}.example/{index}",
                "canonical_url": f"https://{outlet.lower()}.example/{index}",
                "published_at": published_at or f"{day}T09:00:00+00:00",
                "fetched_at": f"{day}T11:00:00+00:00",
                "raw_json": {"index": index},
            }
        ]
    )
    return result.item_id


def _story(ticker, day, item_id, title, outlet, stage, published_at):
    fingerprint = cluster_fingerprint_for(ticker, [str(item_id)])
    return StoryRecord(
        cluster_fingerprint=fingerprint,
        canonical_title=title,
        members=(
            StoryMemberRecord(
                raw_item_id=item_id,
                position=0,
                outlet=outlet,
                url=f"https://{outlet.lower()}.example/{item_id}",
                canonical_url=f"https://{outlet.lower()}.example/{item_id}",
            ),
        ),
        canonical_item_id=item_id,
        outlet=outlet,
        outlet_count=1,
        published_at=published_at,
        canonical_url=f"https://{outlet.lower()}.example/{item_id}",
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
    world,
    ticker="TSLA",
    day=DAY,
    *,
    themes=(("Alpha", 2, 1), ("Beta", 1, 2)),
    other=0,
    excluded=0,
    stage="m3.semantic",
    with_theme_set=True,
    story_count_delta=0,
    published_at=None,
) -> Seeded:
    """One partition, through the real reconciliation paths.

    ``themes`` is ``(label, story count, salience rank)``.  Every story has
    one raw item; outlets rotate.  ``published_at`` maps a story title to a
    timestamp (``None`` for absent); otherwise stories are stamped in
    creation order.
    """

    repository = world.repository
    published_at = published_at or {}
    groups: list[tuple[str, list[str]]] = [
        (label, [f"{ticker} {day} t{n} story {i}" for i in range(count)])
        for n, (label, count, _) in enumerate(themes)
    ]
    groups.append(("other", [f"{ticker} {day} other {i}" for i in range(other)]))
    groups.append(
        ("excluded", [f"{ticker} {day} excluded {i}" for i in range(excluded)])
    )
    records = []
    sequence = 0
    for _, titles in groups:
        for title in titles:
            outlet = OUTLETS[sequence % len(OUTLETS)]
            stamp = published_at.get(title, f"{day}T10:{sequence:02d}:00+00:00")
            item = _insert_item(repository, ticker, day, outlet, stamp)
            records.append(_story(ticker, day, item, title, outlet, stage, stamp))
            sequence += 1
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
    rows = repository.stories_for_day(day, ticker)
    by_title = {row["canonical_title"]: row["id"] for row in rows}
    story_ids = {label: [by_title[t] for t in titles] for label, titles in groups}
    if not with_theme_set:
        return Seeded({}, {}, story_ids)
    items = {
        row["id"]: [member["raw_item_id"] for member in _members(world, row["id"])]
        for row in rows
    }
    theme_records = []
    for n, (label, _, rank) in enumerate(themes):
        members = story_ids[label]
        theme_records.append(
            ThemeRecord(
                fingerprint=f"fp-{ticker}-{day}-{n}-{members[0]}",
                theme_key=f"key-{ticker}-{day}-{n}",
                label=label,
                label_source="canonical_story_title",
                story_ids=tuple(members),
                citation_item_ids=tuple(i for s in members for i in items[s]),
                status="ready",
                salience_rank=rank,
                story_count=len(members),
            )
        )
    # Positions run opposite to story ids, so ordering by position is visible.
    other_ids = story_ids["other"]
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
                source_metadata={"story_count": len(rows) + story_count_delta},
                config_fingerprint="cfg",
                algorithm_version="m5.1",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            ),
            themes=theme_records,
            other_coverage=[
                OtherCoverageRecord(
                    story_id=story_id,
                    reason="clustering_noise",
                    position=len(other_ids) - 1 - index,
                )
                for index, story_id in enumerate(other_ids)
            ],
            excluded=[
                ExcludedStoryRecord(story_id=story_id, reason="no_encodable_text")
                for story_id in story_ids["excluded"]
            ],
            terminal=True,
        )
    population = repository.read.theme_population(ticker, day, VERSION)
    return Seeded(
        {theme.label: theme.theme_id for theme in population.themes},
        {theme.label: theme.theme_key for theme in population.themes},
        story_ids,
    )


def seed_m2_only(world, ticker="NVDA", day=DAY) -> Seeded:
    """Stories at m2.exact and no theme set, as the theme stage leaves them."""

    seeded = seed(
        world,
        ticker,
        day,
        themes=(("Exact", 3, 1),),
        stage="m2.exact",
        with_theme_set=False,
    )
    with world.repository.stage_run(
        run_id=f"seed-{next(_RUNS)}",
        stage="themes",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        run.record_degradation("m5_requires_semantic_stories")
        world.repository.clear_theme_set(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            terminal=True,
        )
    return seeded


def _members(world, story_id):
    with world.repository.admin.connect_writable() as connection:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM story_members WHERE story_id = ? ORDER BY position",
                (story_id,),
            )
        ]


def summarize(world, theme_id, ticker="TSLA", day=DAY, client=None):
    client = client or Client()
    with world.repository.stage_run(
        run_id=f"sum-{next(_RUNS)}",
        stage="summaries",
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        outcome = ensure_summary(
            world.repository,
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=VERSION,
            theme_id=theme_id,
            client=client,
            max_attempts=PRODUCTION_MAX_ATTEMPTS,
        )
    assert outcome.source == SOURCE_GENERATED
    return outcome


def raw_sql(world, statement, parameters=()):
    """Damage the database the way only broken storage or a bypass could.

    Every trigger is dropped for the one statement and recreated exactly
    from ``sqlite_master``: what remains is state the write path's guards
    never saw, which is exactly what the read side must refuse to serve.
    """

    with world.repository.admin.connect_writable() as connection:
        triggers = [
            (row[0], row[1])
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            )
        ]
        for name, _ in triggers:
            connection.execute(f"DROP TRIGGER {name}")
        try:
            connection.execute(statement, parameters)
        finally:
            for _, create in triggers:
                connection.execute(create)


def read_api(world, **kwargs) -> SqliteNarrativeRepository:
    kwargs.setdefault("now_provider", lambda: NOW)
    return SqliteNarrativeRepository(
        database_path=world.path, pipeline_version=VERSION, **kwargs
    )


@pytest.fixture
def api():
    def make(repository) -> TestClient:
        app.dependency_overrides[routes.repository_dependency] = lambda: repository
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def themes_of(client, ticker="TSLA", day=None):
    path = f"/api/v1/tickers/{ticker}/themes" + (f"?date={day}" if day else "")
    response = client.get(path)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert_contract(payload)
    return payload


def assert_contract(payload):
    """The response invariants the frontend relies on, checked on JSON."""

    locations = []
    for theme in payload["themes"]:
        stories = {story["id"]: story for story in theme["stories"]}
        assert len(stories) == len(theme["stories"])
        for citation in theme["citations"]:
            assert stories[citation["id"]] == citation
        cited = {citation["id"] for citation in theme["citations"]}
        for sentence in theme["sentences"]:
            assert set(sentence["citation_ids"]) <= cited
        if theme["degraded"]:
            assert theme["sentences"] == []
        else:
            assert 2 <= len(theme["sentences"]) <= 4
        locations += list(stories)
    locations += [story["id"] for story in payload["other_coverage"]["stories"]]
    assert len(locations) == len(set(locations)), "a story is placed twice"
    for theme in payload["themes"]:
        for story in theme["stories"]:
            assert story["url"].startswith("https://")
    for story in payload["other_coverage"]["stories"]:
        assert story["url"].startswith("https://")


def by_label(payload):
    return {theme["label"]: theme for theme in payload["themes"]}


def sid(story_id):
    return f"story:{story_id}"


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_default_source_is_the_fixture():
    assert config.Settings().PHASE0_NARRATIVE_SOURCE == "fixture"
    narrative.get_narrative_repository.cache_clear()
    try:
        repository = narrative.get_narrative_repository()
        assert isinstance(repository, narrative.FixtureNarrativeRepository)
    finally:
        narrative.get_narrative_repository.cache_clear()


def test_sqlite_is_selected_only_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASE0_NARRATIVE_SOURCE", "sqlite")
    monkeypatch.setenv("PHASE0_DATABASE_PATH", str(tmp_path / "x.db"))
    monkeypatch.setenv("PHASE0_PIPELINE_VERSION", "v9")
    settings = config.Settings()
    assert settings.PHASE0_NARRATIVE_SOURCE == "sqlite"
    monkeypatch.setattr(config, "settings", settings)
    narrative.get_narrative_repository.cache_clear()
    try:
        repository = narrative.get_narrative_repository()
    finally:
        narrative.get_narrative_repository.cache_clear()
    assert isinstance(repository, SqliteNarrativeRepository)
    assert repository.pipeline_version == "v9"
    assert repository.reader._database_path == tmp_path / "x.db"

    monkeypatch.setenv("PHASE0_NARRATIVE_SOURCE", "database")
    with pytest.raises(ValueError):
        config.Settings()
    with pytest.raises(ValueError, match="unknown narrative source"):
        narrative.build_narrative_repository("auto")


def test_sqlite_defaults_match_the_pipeline():
    default = narrative.build_narrative_repository("sqlite")
    assert default.reader._database_path == DEFAULT_DATABASE_PATH
    pipeline_source = (PROJECT_ROOT / "pipeline.py").read_text(encoding="utf-8")
    assert (
        'os.getenv("PHASE0_PIPELINE_VERSION", "phase0-v1")' in pipeline_source
    ), "pipeline default moved; update Settings.PHASE0_PIPELINE_VERSION"
    assert config.Settings().PHASE0_PIPELINE_VERSION == "phase0-v1"


def test_sqlite_mode_never_falls_back_to_the_fixture(api, tmp_path):
    client = api(
        narrative.build_narrative_repository(
            "sqlite", database_path=tmp_path / "missing.db", pipeline_version=VERSION
        )
    )
    for path in (
        "/api/v1/tickers",
        "/api/v1/meta/status",
        "/api/v1/tickers/NVDA/themes",
    ):
        response = client.get(path)
        assert response.status_code == 503
        assert response.json() == {"detail": routes.UNAVAILABLE_DETAIL}


def test_the_default_policy_resolver_is_the_schedulers():
    signature = SqliteNarrativeRepository.__init__.__kwdefaults__
    assert signature["policy_resolver"] is summary_runner.production_generation_policy


# ----------------------------------------------------------------------
# Tickers and dates
# ----------------------------------------------------------------------


def test_tickers_are_the_fixed_universe_in_order(world, api):
    seed(world, "NVDA")
    seed(world, "TSLA", themes=(("Alpha", 1, 1),))
    payload = api(read_api(world)).get("/api/v1/tickers").json()
    assert [item["ticker"] for item in payload["tickers"]] == UNIVERSE
    names = {item["ticker"]: item["company_name"] for item in payload["tickers"]}
    assert names["NVDA"] == "NVIDIA" and names["META"] == "Meta Platforms"
    counts = {item["ticker"]: item["theme_count"] for item in payload["tickers"]}
    assert counts == {"TSLA": 1, "NVDA": 2, "AMD": 0, "AAPL": 0, "META": 0}


def test_multiple_tickers_are_served_separately(world, api):
    tsla = seed(world, "TSLA")
    nvda = seed(world, "NVDA", themes=(("Gamma", 1, 1),), other=1)
    client = api(read_api(world))
    tsla_payload = themes_of(client, "TSLA")
    nvda_payload = themes_of(client, "nvda")
    assert nvda_payload["ticker"] == "NVDA"
    assert [t["id"] for t in tsla_payload["themes"]] == [
        tsla.theme_keys["Alpha"],
        tsla.theme_keys["Beta"],
    ]
    assert [t["id"] for t in nvda_payload["themes"]] == [nvda.theme_keys["Gamma"]]
    assert [s["id"] for s in nvda_payload["other_coverage"]["stories"]] == [
        sid(nvda.story_ids["other"][0])
    ]


def test_explicit_date_selects_that_day(world, api):
    older = seed(world, day=EARLIER)
    summarize(world, older.theme_ids["Alpha"], day=EARLIER)
    seed(world, day=DAY)
    payload = themes_of(api(read_api(world)), day=EARLIER)
    assert payload["date"] == EARLIER
    assert by_label(payload)[GENERATED_LABEL]["degraded"] is False


def test_latest_real_coverage_wins_over_an_older_summarized_day(world, api):
    older = seed(world, day=EARLIER)
    summarize(world, older.theme_ids["Alpha"], day=EARLIER)
    seed(world, day=DAY)
    payload = themes_of(api(read_api(world)))
    assert payload["date"] == DAY
    assert all(theme["degraded"] for theme in payload["themes"])
    assert [theme["label"] for theme in payload["themes"]] == ["Alpha", "Beta"]


def test_latest_day_ignores_invalidated_stories(world, api):
    seed(world, day=EARLIER)
    later = seed(world, day=DAY, themes=(("Alpha", 1, 1),), with_theme_set=False)
    raw_sql(
        world,
        "UPDATE stories SET invalidated_at = '2026-07-23T14:00:00+00:00' "
        "WHERE id = ?",
        (later.story_ids["Alpha"][0],),
    )
    reader = world.repository.read
    assert reader.latest_story_day("TSLA", VERSION) == EARLIER
    assert reader.latest_story_day("TSLA", "other-version") is None
    assert themes_of(api(read_api(world)))["date"] == EARLIER


def test_run_completion_reader_refuses_unknown_statuses(world):
    with pytest.raises(phase0_repository.Phase0ValidationError):
        world.repository.read.latest_run_completion(VERSION, ("running",))
    with pytest.raises(phase0_repository.Phase0ValidationError):
        world.repository.read.latest_run_completion(VERSION, ())


def test_unknown_ticker_is_404_and_known_empty_date_is_200(world, api):
    seed(world)
    client = api(read_api(world))
    assert client.get("/api/v1/tickers/GME/themes").status_code == 404
    payload = themes_of(client, "TSLA", day="2026-07-01")
    assert payload["date"] == "2026-07-01"
    assert payload["themes"] == []
    assert payload["other_coverage"] == {
        "outlet_count": 0,
        "story_count": 0,
        "stories": [],
    }
    assert client.get("/api/v1/tickers/TSLA/themes?date=nope").status_code == 422


# ----------------------------------------------------------------------
# Currentness
# ----------------------------------------------------------------------


def test_current_artifact_is_served_exactly(world, api):
    seeded = seed(world)
    outcome = summarize(world, seeded.theme_ids["Alpha"])
    payload = themes_of(api(read_api(world)))
    alpha, beta = payload["themes"]
    assert alpha["id"] == seeded.theme_keys["Alpha"]
    assert alpha["rank"] == 1 and alpha["degraded"] is False
    assert alpha["label"] == outcome.artifact.label == GENERATED_LABEL
    members = [sid(i) for i in seeded.story_ids["Alpha"]]
    assert alpha["sentences"] == [
        {"text": "Coverage leads with this story.", "citation_ids": members[::-1]},
        {"text": "Outlets repeat the report.", "citation_ids": members[:1]},
    ]
    assert [s["text"] for s in alpha["sentences"]] == [
        sentence.text for sentence in outcome.artifact.sentences
    ]
    assert beta["degraded"] is True and beta["label"] == "Beta"


def test_current_theme_stories_are_the_frozen_evidence(world, api):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    current = current_summary_artifact(
        world.repository.read,
        "TSLA",
        DAY,
        VERSION,
        seeded.theme_ids["Alpha"],
        production_generation_policy(),
    )
    alpha = themes_of(api(read_api(world)))["themes"][0]
    expected = [
        {
            "id": evidence.citation_id,
            "headline": evidence.title,
            "outlet": evidence.outlet,
            "url": evidence.urls[0],
            "published_at": evidence.published_at.replace("+00:00", "Z"),
        }
        for evidence in current.generation_input.evidence
    ]
    assert alpha["stories"] == expected
    assert alpha["citations"] == expected
    assert alpha["story_count"] == 2 and alpha["outlet_count"] == 2


def test_degraded_projection_agrees_with_the_frozen_evidence(world, api):
    """Persisted-story mapping and evidence mapping cannot drift apart."""

    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    current = themes_of(api(read_api(world)))["themes"][0]

    def unresolvable():
        raise summarization.ProviderConfigurationError("bad setting")

    degraded = themes_of(api(read_api(world, policy_resolver=unresolvable)))
    degraded = degraded["themes"][0]
    assert current["degraded"] is False and degraded["degraded"] is True
    assert degraded["stories"] == current["stories"]
    assert degraded["label"] == "Alpha"


def test_changed_membership_retires_the_old_artifact_and_its_label(world, api):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    reseeded = seed(world, themes=(("Alpha", 2, 1), ("Beta", 1, 2), ("Delta", 1, 3)))
    assert world.repository.read.summary_artifacts("TSLA", DAY, VERSION)
    payload = themes_of(api(read_api(world)))
    assert [theme["label"] for theme in payload["themes"]] == ["Alpha", "Beta", "Delta"]
    assert all(theme["degraded"] for theme in payload["themes"])
    assert payload["themes"][0]["id"] == reseeded.theme_keys["Alpha"]


def test_changed_model_retires_the_artifact(world, api, monkeypatch):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    payload = themes_of(api(read_api(world)))
    assert payload["themes"][0]["degraded"] is True
    assert payload["themes"][0]["label"] == "Alpha"


def test_changed_output_cap_retires_the_artifact(world, api, monkeypatch):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")
    assert themes_of(api(read_api(world)))["themes"][0]["degraded"] is True


def test_changed_copy_rules_retire_the_artifact(world, api, monkeypatch):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    rules = guarded.load_copy_rules()
    monkeypatch.setattr(
        guarded,
        "load_copy_rules",
        lambda: tuple(rules) + (("advice", "phrase", "zz-never-used-zz"),),
    )
    assert themes_of(api(read_api(world)))["themes"][0]["degraded"] is True


def test_invalidated_artifact_is_not_served(world, api):
    seeded = seed(world)
    outcome = summarize(world, seeded.theme_ids["Alpha"])
    raw_sql(
        world,
        "UPDATE summary_artifacts SET status = 'invalidated', "
        "invalidated_at = ?, invalidated_reason = 'test' WHERE id = ?",
        ("2026-07-23T14:50:00+00:00", outcome.artifact.artifact_id),
    )
    alpha = themes_of(api(read_api(world)))["themes"][0]
    assert alpha["degraded"] is True and alpha["label"] == "Alpha"


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        "UPDATE summary_sentences SET text = 'Edited after the fact.' "
        "WHERE artifact_id = ? AND ordinal = 1",
    ],
    ids=["missing-sentence", "digest-mismatch"],
)
def test_corrupt_artifact_is_not_served(world, api, statement):
    seeded = seed(world)
    outcome = summarize(world, seeded.theme_ids["Alpha"])
    raw_sql(world, statement, (outcome.artifact.artifact_id,))
    alpha = themes_of(api(read_api(world)))["themes"][0]
    assert alpha["degraded"] is True
    assert alpha["sentences"] == [] and alpha["label"] == "Alpha"


def test_population_race_guard_degrades_instead_of_mixing(world, api, monkeypatch):
    """If P2's frozen input differs from the P1 layout, the theme degrades.

    Theme ids are never reused, so a same-id theme with different evidence
    cannot be staged through the write paths; the currentness reader is
    wrapped to hand back a CurrentSummary for another theme's input.
    """

    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    population = world.repository.read.theme_population("TSLA", DAY, VERSION)
    from phase0.summaries import build_generation_input

    other_input = build_generation_input(population, seeded.theme_ids["Beta"])
    real = sqlite_module.current_summary_artifact

    def drifted(*args, **kwargs):
        found = real(*args, **kwargs)
        if found is None:
            return None
        return dataclasses.replace(found, generation_input=other_input)

    monkeypatch.setattr(sqlite_module, "current_summary_artifact", drifted)
    alpha = themes_of(api(read_api(world)))["themes"][0]
    assert alpha["degraded"] is True and alpha["label"] == "Alpha"
    assert [s["id"] for s in alpha["stories"]] == [
        sid(i) for i in seeded.story_ids["Alpha"]
    ]


def test_unresolvable_policy_serves_degraded_and_generates_nothing(world, api):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])

    def unresolvable():
        raise summarization.ProviderConfigurationError("GEMINI_MAX_OUTPUT_TOKENS")

    payload = themes_of(api(read_api(world, policy_resolver=unresolvable)))
    assert all(theme["degraded"] for theme in payload["themes"])


# ----------------------------------------------------------------------
# Population health and placement
# ----------------------------------------------------------------------


def test_m2_only_day_has_no_themes_and_keeps_every_story(world, api):
    seeded = seed_m2_only(world, "NVDA")
    client = api(read_api(world))
    payload = themes_of(client, "NVDA")
    assert payload["themes"] == []
    assert [s["id"] for s in payload["other_coverage"]["stories"]] == [
        sid(i) for i in sorted(seeded.story_ids["Exact"])
    ]
    assert payload["other_coverage"]["story_count"] == 3
    tickers = client.get("/api/v1/tickers").json()["tickers"]
    assert {t["ticker"]: t["theme_count"] for t in tickers}["NVDA"] == 0


def test_inconsistent_population_shows_no_themes_even_with_an_artifact(world, api):
    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    # The theme set now records a story count the live generation lacks.
    raw_sql(
        world,
        "UPDATE theme_sets SET source_metadata = json_set(source_metadata, "
        "'$.story_count', 99) WHERE ticker = 'TSLA'",
    )
    payload = themes_of(api(read_api(world)))
    assert payload["themes"] == []
    live = sorted(i for ids in seeded.story_ids.values() for i in ids)
    assert [s["id"] for s in payload["other_coverage"]["stories"]] == [
        sid(i) for i in live
    ]


def test_other_coverage_then_excluded_in_deterministic_order(world, api):
    seeded = seed(world, other=3, excluded=2)
    payload = themes_of(api(read_api(world)))
    other_ids = seeded.story_ids["other"]
    expected = [sid(i) for i in reversed(other_ids)] + [
        sid(i) for i in sorted(seeded.story_ids["excluded"])
    ]
    coverage = payload["other_coverage"]
    assert [s["id"] for s in coverage["stories"]] == expected
    assert coverage["story_count"] == 5
    assert coverage["outlet_count"] == len({s["outlet"] for s in coverage["stories"]})
    # No live story is lost.
    served = [s["id"] for t in payload["themes"] for s in t["stories"]]
    served += [s["id"] for s in coverage["stories"]]
    live = [sid(i) for ids in seeded.story_ids.values() for i in ids]
    assert sorted(served) == sorted(live)


def test_themes_follow_salience_rank_not_storage_order(world, api):
    seeded = seed(world, themes=(("Low", 1, 3), ("High", 1, 1), ("Mid", 1, 2)))
    payload = themes_of(api(read_api(world)))
    assert [(t["rank"], t["label"]) for t in payload["themes"]] == [
        (1, "High"),
        (2, "Mid"),
        (3, "Low"),
    ]
    assert payload["themes"][0]["id"] == seeded.theme_keys["High"]


# ----------------------------------------------------------------------
# Field edge cases
# ----------------------------------------------------------------------


def test_missing_publication_time_is_null_in_current_and_other(world, api):
    theme_title = f"TSLA {DAY} t0 story 0"
    other_title = f"TSLA {DAY} other 0"
    seeded = seed(
        world,
        themes=(("Alpha", 1, 1),),
        other=1,
        published_at={theme_title: None, other_title: None},
    )
    summarize(world, seeded.theme_ids["Alpha"])
    payload = themes_of(api(read_api(world)))
    alpha = payload["themes"][0]
    assert alpha["degraded"] is False
    assert alpha["stories"][0]["published_at"] is None
    assert "published_at" in alpha["citations"][0]
    assert payload["other_coverage"]["stories"][0]["published_at"] is None


def test_timestamp_without_offset_fails_closed(world, api):
    seeded = seed(world)
    raw_sql(
        world,
        "UPDATE stories SET published_at = '2026-07-23 10:00:00' WHERE id = ?",
        (seeded.story_ids["Beta"][0],),
    )
    response = api(read_api(world)).get("/api/v1/tickers/TSLA/themes")
    assert response.status_code == 503
    assert response.json() == {"detail": routes.UNAVAILABLE_DETAIL}


def test_story_without_any_url_fails_closed(world, api):
    seeded = seed(world)
    story_id = seeded.story_ids["Beta"][0]
    raw_sql(world, "UPDATE stories SET canonical_url = NULL WHERE id = ?", (story_id,))
    raw_sql(
        world,
        "UPDATE story_members SET url = NULL, canonical_url = NULL "
        "WHERE story_id = ?",
        (story_id,),
    )
    response = api(read_api(world)).get("/api/v1/tickers/TSLA/themes")
    assert response.status_code == 503


def test_long_persisted_m5_label_is_served_verbatim(world, api):
    """D5: M5 labels are real headlines and nothing bounds their length.

    Yahoo and RSS titles are only stripped, ``display_text`` only
    normalizes whitespace and typography, M5 copies the representative
    story's title into ``themes.label``, and the column has no CHECK on
    length.  The API therefore carries no 120-character cap, and nothing
    is truncated.
    """

    label = "Tesla " + "headline words that keep going " * 8
    label = label.strip()
    assert len(label) > 120
    seed(world, themes=((label, 1, 1),))
    theme = themes_of(api(read_api(world)))["themes"][0]
    assert theme["label"] == label
    assert theme["degraded"] is True
    schemas.Theme(
        id="k", label="x" * 500, rank=1, degraded=True, outlet_count=0, story_count=0
    )
    with pytest.raises(ValueError):
        schemas.Theme(
            id="k", label="", rank=1, degraded=True, outlet_count=0, story_count=0
        )


def test_theme_without_a_stable_key_fails_closed(world, api):
    seed(world)
    raw_sql(world, "UPDATE themes SET theme_key = NULL WHERE label = 'Beta'")
    response = api(read_api(world)).get("/api/v1/tickers/TSLA/themes")
    assert response.status_code == 503


# ----------------------------------------------------------------------
# Database states
# ----------------------------------------------------------------------


def test_empty_database_is_unavailable_not_fabricated(world, api):
    client = api(read_api(world))
    for path in (
        "/api/v1/meta/status",
        "/api/v1/tickers",
        "/api/v1/tickers/TSLA/themes",
    ):
        response = client.get(path)
        assert response.status_code == 503, path
        assert response.json() == {"detail": routes.UNAVAILABLE_DETAIL}
    assert client.get("/api/v1/tickers/GME/themes").status_code == 404


def test_runs_without_stories(world, api):
    with world.repository.stage_run(
        run_id="fetch-1",
        stage="fetch_yahoo",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker="TSLA",
    ):
        pass
    client = api(read_api(world))
    tickers = client.get("/api/v1/tickers").json()["tickers"]
    assert [t["theme_count"] for t in tickers] == [0] * 5
    payload = themes_of(client, "TSLA")
    assert payload["date"] == DAY
    assert payload["themes"] == [] and payload["other_coverage"]["stories"] == []


def test_missing_database_is_unavailable_and_never_created(tmp_path, api):
    path = tmp_path / "absent" / "phase0.db"
    client = api(
        SqliteNarrativeRepository(database_path=path, pipeline_version=VERSION)
    )
    assert client.get("/api/v1/meta/status").status_code == 503
    assert client.get("/api/v1/tickers/TSLA/themes").status_code == 503
    assert not path.exists() and not path.parent.exists()


def test_wrong_schema_version_is_unavailable(world, api):
    seed(world)
    with world.repository.admin.connect_writable() as connection:
        connection.execute("PRAGMA user_version = 15")
    response = api(read_api(world)).get("/api/v1/tickers/TSLA/themes")
    assert response.status_code == 503


def test_corrupt_database_is_unavailable(tmp_path, api):
    path = tmp_path / "phase0.db"
    path.write_bytes(b"this is not a sqlite database" * 64)
    client = api(
        SqliteNarrativeRepository(database_path=path, pipeline_version=VERSION)
    )
    for route in ("/api/v1/meta/status", "/api/v1/tickers/TSLA/themes"):
        response = client.get(route)
        assert response.status_code == 503
        assert str(path) not in response.text


# ----------------------------------------------------------------------
# Status and freshness
# ----------------------------------------------------------------------


def _run(world, stage, *, status="success", ticker="TSLA", day=DAY, minutes=5):
    world.clock.advance(minutes)
    with world.repository.stage_run(
        run_id=f"run-{next(_RUNS)}",
        stage=stage,
        trading_day=day,
        pipeline_version=VERSION,
        ticker=ticker,
    ) as run:
        if status == "degraded":
            run.record_degradation("stage_degraded", detail="SECRET-DETAIL-TEXT")
    return world.clock.now


def test_status_reports_the_latest_run_of_each_stage(world, api):
    _run(world, "fetch_yahoo")
    _run(world, "fetch_yahoo", status="degraded")
    _run(world, "themes")
    response = api(read_api(world)).get("/api/v1/meta/status")
    assert response.status_code == 200
    payload = response.json()
    runs = {run["stage"]: run for run in payload["last_runs"]}
    assert list(runs) == ["fetch_yahoo", "themes"]
    assert runs["fetch_yahoo"]["status"] == "degraded"
    assert runs["fetch_yahoo"]["error_count"] == 1
    assert runs["themes"] == {
        **runs["themes"],
        "status": "success",
        "error_count": 0,
    }
    assert "SECRET-DETAIL-TEXT" not in response.text
    assert "stage_degraded" not in response.text


def test_status_ignores_other_pipeline_versions(world, api):
    _run(world, "themes")
    world.clock.advance(5)
    with world.repository.stage_run(
        run_id="other-version",
        stage="summaries",
        trading_day=DAY,
        pipeline_version="v2",
        ticker="TSLA",
    ):
        pass
    runs = api(read_api(world)).get("/api/v1/meta/status").json()["last_runs"]
    assert [run["stage"] for run in runs] == ["themes"]


def test_data_as_of_prefers_success_then_falls_back_to_degraded(world, api):
    degraded_at = _run(world, "fetch_yahoo", status="degraded")
    client = api(read_api(world))
    payload = client.get("/api/v1/meta/status").json()
    assert datetime.fromisoformat(payload["data_as_of"]) == degraded_at
    success_at = _run(world, "themes")
    _run(world, "fetch_yahoo", status="degraded")
    payload = client.get("/api/v1/meta/status").json()
    assert datetime.fromisoformat(payload["data_as_of"]) == success_at


def test_partition_data_as_of_comes_from_its_own_runs(world, api):
    seed(world, "TSLA")
    tsla_at = world.clock.now
    world.clock.advance(20)
    seed(world, "NVDA")
    nvda_at = world.clock.now
    client = api(read_api(world))
    tsla = themes_of(client, "TSLA")
    assert datetime.fromisoformat(tsla["data_as_of"]) == tsla_at
    status = client.get("/api/v1/meta/status").json()
    assert datetime.fromisoformat(status["data_as_of"]) == nvda_at
    # A day this ticker never ran falls back to the global timestamp.
    empty = themes_of(client, "TSLA", day="2026-07-01")
    assert empty["data_as_of"] == status["data_as_of"]
    tickers = {t["ticker"]: t for t in client.get("/api/v1/tickers").json()["tickers"]}
    assert datetime.fromisoformat(tickers["TSLA"]["data_as_of"]) == tsla_at
    assert tickers["AMD"]["data_as_of"] == status["data_as_of"]


@pytest.mark.parametrize(
    "now, stale",
    [
        (datetime(2026, 7, 23, 15, 15, tzinfo=timezone.utc), False),
        (datetime(2026, 7, 23, 17, 0, tzinfo=timezone.utc), True),
        (datetime(2026, 7, 23, 21, 0, tzinfo=timezone.utc), False),
        (datetime(2026, 7, 25, 17, 0, tzinfo=timezone.utc), False),
    ],
    ids=["fresh", "stale-in-session", "after-close", "weekend"],
)
def test_staleness_uses_the_market_hours_rule(world, api, now, stale):
    _run(world, "themes")  # completes at 14:35 UTC
    client = api(read_api(world, now_provider=lambda: now))
    assert client.get("/api/v1/meta/status").json()["is_stale"] is stale
    tickers = client.get("/api/v1/tickers").json()["tickers"]
    assert {t["is_stale"] for t in tickers} == {stale}


# ----------------------------------------------------------------------
# GET never generates, never writes, never reaches the network
# ----------------------------------------------------------------------


def _dump(path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def _explode(name):
    def refuse(*args, **kwargs):
        raise AssertionError(f"GET reached {name}")

    return refuse


@pytest.fixture
def persisted_world(world):
    """Current, degraded, Other Coverage and M2-only state, all at once."""

    seeded = seed(world, other=2, excluded=1)
    summarize(world, seeded.theme_ids["Alpha"])
    seed_m2_only(world, "NVDA")
    return world


@pytest.fixture
def generation_forbidden(monkeypatch):
    for owner, name in (
        (lifecycle, "ensure_summary"),
        (summary_runner, "ensure_summary"),
        (lifecycle, "generate_guarded_summary"),
        (guarded, "generate_guarded_summary"),
        (summarization.GeminiClient, "generate"),
        (summarization.GeminiClient, "_get_client"),
        (phase0_repository.Phase0Repository, "__init__"),
        (phase0_repository.Phase0Repository, "migrate"),
        (phase0_repository.Phase0Repository, "stage_run"),
        (phase0_repository.Phase0Repository, "persist_summary_generation"),
        (phase0_repository.Phase0Repository, "_open_connection"),
        (phase0_repository.Phase0Admin, "connect_writable"),
        (summary_runner, "run_scheduled_summaries"),
    ):
        monkeypatch.setattr(owner, name, _explode(f"{owner!r}.{name}"))
    real_connect = sqlite3.connect

    def read_only_connect(database, *args, **kwargs):
        if "mode=ro" not in str(database):
            raise AssertionError(f"GET opened a writable connection: {database}")
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", read_only_connect)


@pytest.fixture
def network_forbidden(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _explode("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", _explode("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", _explode("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", _explode("getaddrinfo"))
    for module in ("google.genai", "yfinance", "feedparser"):
        monkeypatch.setitem(sys.modules, module, None)


def _exercise(client):
    responses = [
        client.get("/api/v1/tickers"),
        client.get("/api/v1/meta/status"),
        client.get("/api/v1/tickers/TSLA/themes"),
        client.get(f"/api/v1/tickers/TSLA/themes?date={DAY}"),
        client.get("/api/v1/tickers/NVDA/themes"),
        client.get("/api/v1/tickers/AMD/themes"),
    ]
    assert [r.status_code for r in responses] == [200] * len(responses)
    for response in responses[2:]:
        assert_contract(response.json())
    return responses


def test_get_never_generates_or_writes_rows(persisted_world, api, request):
    """No generation and no row change.

    This proves the *database content* is untouched -- rows, and the main
    database file's bytes.  It does not claim the directory is untouched:
    SQLite may create its WAL coordination files, which the filesystem
    tests below pin down separately.
    """

    main_file_before = _main_file_digest(persisted_world.path)
    before = _dump(persisted_world.path)
    counts_before = {
        table: persisted_world.repository.read.count(table)
        for table in (*SUMMARY_TABLES, "run_log", "pipeline_stage_keys")
    }
    repository = read_api(persisted_world)
    request.getfixturevalue("generation_forbidden")
    responses = _exercise(api(repository))
    tsla = responses[2].json()
    assert tsla["themes"][0]["degraded"] is False  # the current summary served
    assert tsla["themes"][1]["degraded"] is True
    assert _dump(persisted_world.path) == before
    assert _main_file_digest(persisted_world.path) == main_file_before
    assert {
        table: repository.reader.count(table) for table in counts_before
    } == counts_before


def test_get_needs_no_network_and_no_provider_sdk(persisted_world, api, request):
    repository = read_api(persisted_world)
    request.getfixturevalue("network_forbidden")
    tsla = _exercise(api(repository))[2].json()
    assert tsla["themes"][0]["label"] == GENERATED_LABEL


def test_get_needs_no_credential(persisted_world, api, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    tsla = _exercise(api(read_api(persisted_world)))[2].json()
    assert tsla["themes"][0]["degraded"] is False


def _main_file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ----------------------------------------------------------------------
# Independent review repairs (P2-1 .. P2-5)
# ----------------------------------------------------------------------

# -- P2-1: each ticker is stale by its own freshness -----------------------


def _seed_at(world, ticker, hour, minute):
    world.clock.now = datetime(2026, 7, 23, hour, minute, tzinfo=timezone.utc)
    seed(world, ticker, themes=(("Alpha", 1, 1),))
    return world.clock.now


def test_each_ticker_is_stale_by_its_own_freshness(world, api):
    """The independent reproduction: TSLA 12:00, NVDA 14:55, now 15:00 UTC."""

    tsla_at = _seed_at(world, "TSLA", 12, 0)
    nvda_at = _seed_at(world, "NVDA", 14, 55)
    now = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)  # 11:00 ET
    client = api(read_api(world, now_provider=lambda: now))
    status = client.get("/api/v1/meta/status").json()
    assert status["is_stale"] is False  # the global state is fresh
    tickers = {t["ticker"]: t for t in client.get("/api/v1/tickers").json()["tickers"]}
    assert datetime.fromisoformat(tickers["TSLA"]["data_as_of"]) == tsla_at
    assert datetime.fromisoformat(tickers["NVDA"]["data_as_of"]) == nvda_at
    assert tickers["TSLA"]["is_stale"] is True
    assert tickers["NVDA"]["is_stale"] is False
    # A ticker with no partition inherits the global timestamp and verdict.
    assert tickers["AMD"]["is_stale"] is False


@pytest.mark.parametrize(
    "now",
    [
        datetime(2026, 7, 23, 21, 0, tzinfo=timezone.utc),  # after the close
        datetime(2026, 7, 25, 15, 0, tzinfo=timezone.utc),  # Saturday
    ],
    ids=["after-close", "weekend"],
)
def test_per_ticker_staleness_keeps_the_closed_market_rule(world, api, now):
    _seed_at(world, "TSLA", 12, 0)
    _seed_at(world, "NVDA", 14, 55)
    client = api(read_api(world, now_provider=lambda: now))
    tickers = client.get("/api/v1/tickers").json()["tickers"]
    assert {t["is_stale"] for t in tickers} == {False}


def test_one_reference_instant_per_ticker_list(world):
    _seed_at(world, "TSLA", 12, 0)
    calls = []

    def now():
        calls.append(1)
        # Each call would land later; only one may be consulted.
        return datetime(2026, 7, 23, 13, 0, tzinfo=timezone.utc) + timedelta(
            hours=len(calls)
        )

    items = read_api(world, now_provider=now).list_tickers()
    assert len(calls) == 1
    assert len(items) == 5


# -- P2-2: the latest stage outcome is the latest completion ---------------


def _stories_records(world, ticker):
    item = _insert_item(world.repository, ticker, DAY, "Reuters", None)
    return [
        _story(
            ticker,
            DAY,
            item,
            f"{ticker} story",
            "Reuters",
            "m3.semantic",
            f"{DAY}T10:00:00+00:00",
        )
    ]


def test_lower_id_run_that_completes_later_is_the_latest_outcome(world, api):
    """Run A opens first (lower id) and settles after run B."""

    repository = world.repository
    world.clock.now = datetime(2026, 7, 23, 14, 30, tzinfo=timezone.utc)
    with repository.stage_run(
        run_id="run-a",
        stage="stories",
        trading_day=DAY,
        pipeline_version=VERSION,
        ticker="TSLA",
    ) as run_a:
        repository.reconcile_stories(
            run=run_a,
            ticker="TSLA",
            trading_day=DAY,
            pipeline_version=VERSION,
            stories=_stories_records(world, "TSLA"),
        )
        # A's logged mutation has already written its run_log row.
        [row_a] = repository.read.run_log_rows(run_id="run-a")
        world.clock.now = datetime(2026, 7, 23, 14, 35, tzinfo=timezone.utc)
        with repository.stage_run(
            run_id="run-b",
            stage="stories",
            trading_day=DAY,
            pipeline_version=VERSION,
            ticker="NVDA",
        ) as run_b:
            repository.reconcile_stories(
                run=run_b,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version=VERSION,
                stories=_stories_records(world, "NVDA"),
            )
        world.clock.now = datetime(2026, 7, 23, 14, 40, tzinfo=timezone.utc)
        run_a.record_degradation("stage_degraded", detail="SECRET-DETAIL-TEXT")
    [row_a] = repository.read.run_log_rows(run_id="run-a")
    [row_b] = repository.read.run_log_rows(run_id="run-b")
    assert row_a["id"] < row_b["id"]
    assert (row_a["status"], row_b["status"]) == ("degraded", "success")

    [latest] = repository.read.latest_stage_runs(VERSION)
    assert latest["stage"] == "stories"
    assert latest["status"] == "degraded"
    assert latest["completed_at"] == row_a["completed_at"]
    assert latest["error_count"] == 1
    assert set(latest) == {
        "stage",
        "status",
        "started_at",
        "completed_at",
        "duration_ms",
        "error_count",
    }
    response = api(read_api(world)).get("/api/v1/meta/status")
    [served] = response.json()["last_runs"]
    assert served["status"] == "degraded" and served["error_count"] == 1
    assert "SECRET-DETAIL-TEXT" not in response.text
    assert repository.read.latest_stage_runs("v2") == []


def test_equal_completions_resolve_to_the_newer_row_deterministically(world):
    world.clock.now = datetime(2026, 7, 23, 14, 30, tzinfo=timezone.utc)
    _run(world, "themes", minutes=0)
    _run(world, "themes", status="degraded", minutes=0)
    rows = world.repository.read.run_log_rows(stage="themes")
    assert rows[0]["completed_at"] == rows[1]["completed_at"]
    for _ in range(3):
        [latest] = world.repository.read.latest_stage_runs(VERSION)
        assert latest["status"] == "degraded"  # the newer row, every time


def test_a_row_without_a_real_completion_never_wins(world):
    completed = _run(world, "themes")
    with world.repository.admin.connect_writable() as connection:
        # NULL is refused outright by the schema.
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO run_log (run_id, stage, duration_ms, started_at, "
                "completed_at, status, trading_day, pipeline_version) "
                "VALUES ('bad-null', 'themes', 0, ?, NULL, 'failed', ?, ?)",
                (f"{DAY}T15:00:00+00:00", DAY, VERSION),
            )
        # A value that is not a timestamp gets in only past the CHECK.
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            "INSERT INTO run_log (run_id, stage, duration_ms, started_at, "
            "completed_at, status, trading_day, pipeline_version) "
            "VALUES ('bad-text', 'themes', 0, ?, 'zzzz-not-a-time', 'failed', ?, ?)",
            (f"{DAY}T15:00:00+00:00", DAY, VERSION),
        )
        connection.execute("PRAGMA ignore_check_constraints = OFF")
    reader = world.repository.read
    [latest] = reader.latest_stage_runs(VERSION)
    assert latest["status"] == "success"
    assert datetime.fromisoformat(latest["completed_at"]) == completed
    anchor = reader.latest_run_completion(VERSION, ("success", "degraded", "failed"))
    assert datetime.fromisoformat(anchor["completed_at"]) == completed


# -- P2-3: a blank provenance URL fails closed ------------------------------


def _assert_fixed_503(response):
    assert response.status_code == 503
    assert response.json() == {"detail": routes.UNAVAILABLE_DETAIL}


@pytest.mark.parametrize("blank", ["   ", "\t"], ids=["spaces", "tab"])
def test_blank_url_on_a_degraded_theme_fails_closed(world, api, blank):
    seeded = seed(world)
    raw_sql(
        world,
        "UPDATE stories SET canonical_url = ? WHERE id = ?",
        (blank, seeded.story_ids["Beta"][0]),
    )
    _assert_fixed_503(api(read_api(world)).get("/api/v1/tickers/TSLA/themes"))


def test_blank_url_in_frozen_evidence_fails_closed(world, api):
    """URLs are provenance, outside the input fingerprint: the artifact stays
    current, so the blank URL reaches the frozen-evidence projection."""

    seeded = seed(world)
    summarize(world, seeded.theme_ids["Alpha"])
    raw_sql(
        world,
        "UPDATE stories SET canonical_url = '   ' WHERE id = ?",
        (seeded.story_ids["Alpha"][0],),
    )
    current = current_summary_artifact(
        world.repository.read,
        "TSLA",
        DAY,
        VERSION,
        seeded.theme_ids["Alpha"],
        production_generation_policy(),
    )
    assert current is not None and current.generation_input.evidence[0].urls[0] == (
        "   "
    )
    _assert_fixed_503(api(read_api(world)).get("/api/v1/tickers/TSLA/themes"))


def test_blank_url_in_other_coverage_fails_closed(world, api):
    seeded = seed(world, other=1)
    raw_sql(
        world,
        "UPDATE stories SET canonical_url = '   ' WHERE id = ?",
        (seeded.story_ids["other"][0],),
    )
    _assert_fixed_503(api(read_api(world)).get("/api/v1/tickers/TSLA/themes"))


def test_all_empty_urls_fail_closed(world, api):
    seeded = seed(world)
    story_id = seeded.story_ids["Beta"][0]
    raw_sql(world, "UPDATE stories SET canonical_url = '' WHERE id = ?", (story_id,))
    raw_sql(
        world,
        "UPDATE story_members SET url = '', canonical_url = '' WHERE story_id = ?",
        (story_id,),
    )
    _assert_fixed_503(api(read_api(world)).get("/api/v1/tickers/TSLA/themes"))


def test_empty_canonical_url_keeps_the_existing_member_url_selection(world, api):
    """An empty canonical URL is absent, exactly as ``evidence_for`` treats it:
    the member URL it already selects next is served unchanged."""

    seeded = seed(world)
    story_id = seeded.story_ids["Beta"][0]
    [member] = _members(world, story_id)
    raw_sql(world, "UPDATE stories SET canonical_url = '' WHERE id = ?", (story_id,))
    beta = themes_of(api(read_api(world)))["themes"][1]
    assert beta["stories"][0]["url"] == member["canonical_url"]


# -- P2-4: public theme ids are unique --------------------------------------


def test_response_model_rejects_duplicate_theme_ids():
    def theme(rank):
        return schemas.Theme(
            id="same-key",
            label=f"Theme {rank}",
            rank=rank,
            degraded=True,
            outlet_count=0,
            story_count=0,
        )

    empty = schemas.OtherCoverage(outlet_count=0, story_count=0, stories=[])
    with pytest.raises(ValueError, match="Theme ids must be unique"):
        schemas.TickerThemesResponse(
            ticker="TSLA",
            date=DAY,
            data_as_of=NOW,
            themes=[theme(1), theme(2)],
            other_coverage=empty,
        )


def test_persisted_duplicate_theme_keys_fail_closed(world, api):
    seeded = seed(world)
    client = api(read_api(world))
    ids = [theme["id"] for theme in themes_of(client)["themes"]]
    assert ids == [seeded.theme_keys["Alpha"], seeded.theme_keys["Beta"]]
    with world.repository.admin.connect_writable() as connection:
        connection.execute("DROP INDEX idx_themes_theme_key")
    raw_sql(
        world,
        "UPDATE themes SET theme_key = ? WHERE label = 'Beta'",
        (seeded.theme_keys["Alpha"],),
    )
    _assert_fixed_503(client.get("/api/v1/tickers/TSLA/themes"))


# -- P2-5: SQLite WAL sidecars, separately from database content -------------

_WAL_SIDECARS = ("-wal", "-shm")


def _entries(path: Path) -> set[str]:
    return set(os.listdir(path.parent))


def _sidecars(path: Path) -> set[str]:
    return {path.name + suffix for suffix in _WAL_SIDECARS}


def _is_wal(path: Path) -> bool:
    header = path.read_bytes()[:20]
    return header[18] == 2 and header[19] == 2


needs_permissions = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores directory permissions",
)


def test_checkpointed_wal_read_writes_no_database_content(persisted_world, api):
    """What a GET may leave behind when the directory lets SQLite write.

    The live database is WAL and, with no writer connected, checkpointed:
    only the main file exists.  A ``mode=ro`` read cannot run without the
    WAL index, so SQLite creates its coordination files -- an empty ``-wal``
    and a ``-shm`` index -- with the database file's permissions.  Nothing
    else appears, and the main database file's bytes do not change.
    """

    path = persisted_world.path
    assert _is_wal(path)
    before = _entries(path)
    assert not before & _sidecars(path), "precondition: checkpointed, no sidecars"
    digest = _main_file_digest(path)

    _exercise(api(read_api(persisted_world)))

    created = _entries(path) - before
    assert created <= _sidecars(path)
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        assert wal.stat().st_size == 0  # no content was logged
    assert _main_file_digest(path) == digest


@needs_permissions
def test_without_provisioned_wal_state_a_read_only_directory_fails_closed(
    persisted_world, api
):
    """No sidecars and a directory the API cannot write: fixed 503, no files."""

    path = persisted_world.path
    before = _entries(path)
    assert not before & _sidecars(path)
    digest = _main_file_digest(path)
    client = api(read_api(persisted_world))
    os.chmod(path.parent, 0o555)
    try:
        for route in (
            "/api/v1/meta/status",
            "/api/v1/tickers",
            "/api/v1/tickers/TSLA/themes",
        ):
            response = client.get(route)
            _assert_fixed_503(response)
            assert str(path.parent) not in response.text
        assert _entries(path) == before
    finally:
        os.chmod(path.parent, 0o755)
    assert _main_file_digest(path) == digest


@needs_permissions
def test_writer_provisioned_wal_state_serves_a_read_only_api(persisted_world, api):
    """The supported deployment: the writer keeps the WAL state live.

    A writer-owned connection held open keeps ``-wal`` and ``-shm`` in
    place across the pipeline's own open/write/close cycles.  With them
    provisioned, the API reads from a directory it cannot write, creates no
    filesystem entry, and sees the writer's new commits.
    """

    path = persisted_world.path
    keeper = sqlite3.connect(path)  # writer-side, not the API
    try:
        keeper.execute("SELECT 1").fetchall()
        assert _sidecars(path) <= _entries(path)
        client = api(read_api(persisted_world))
        os.chmod(path.parent, 0o555)
        try:
            before = _entries(path)
            first = client.get("/api/v1/meta/status").json()
            _exercise(client)
            # The scheduled writer runs again while the API is serving.
            _run(persisted_world, "summaries", minutes=10)
            second = client.get("/api/v1/meta/status").json()
            assert _entries(path) == before
        finally:
            os.chmod(path.parent, 0o755)
        assert second["data_as_of"] > first["data_as_of"]
    finally:
        keeper.close()

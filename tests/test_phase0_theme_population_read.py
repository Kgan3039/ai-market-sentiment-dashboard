"""``Phase0Reader.partition_generations`` and ``theme_population``: the read
A4's review sampler is built on.

One snapshot per partition, enumeration that does not lose a theme set
whose stories have gone, and the current story generation reported beside
the set so a consumer can judge whether the set is a valid view of it.
"""

from __future__ import annotations

import itertools
import sqlite3
from pathlib import Path

import pytest

from phase0.errors import Phase0ValidationError
from phase0.models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import (
    PartitionGeneration,
    Phase0Repository,
    ThemePopulation,
)

DAY = "2026-07-23"
_RUN_IDS = itertools.count(1)
_ITEMS = itertools.count(1)


def migrated(tmp_path: Path) -> Phase0Repository:
    repository = Phase0Repository(tmp_path / "phase0.db")
    repository.migrate()
    return repository


def items(repository, ticker, count):
    rows = []
    for _ in range(count):
        index = next(_ITEMS)
        rows.append(
            {
                "source": "yahoo:Outlet",
                "ticker": ticker,
                "title": f"{ticker} headline {index}",
                "description": f"Body {index}",
                "url": f"https://publisher.example/{index}",
                "canonical_url": f"https://publisher.example/{index}",
                "published_at": f"{DAY}T10:00:00+00:00",
                "fetched_at": f"{DAY}T11:00:00+00:00",
                "raw_json": {"index": index},
            }
        )
    return [r.item_id for r in repository.admin.insert_raw_items(rows)]


def stories(repository, ticker, count, *, stage="m3.semantic", version="v1"):
    item_ids = items(repository, ticker, count)
    with repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage="stories",
        trading_day=DAY,
        pipeline_version=version,
        ticker=ticker,
    ) as run:
        repository.reconcile_stories(
            run=run,
            ticker=ticker,
            trading_day=DAY,
            pipeline_version=version,
            stories=[
                StoryRecord(
                    cluster_fingerprint=f"{ticker}-{version}-{n}",
                    canonical_title=f"Story {n}",
                    members=(
                        StoryMemberRecord(raw_item_id=item, position=0, outlet="o"),
                    ),
                    canonical_item_id=item,
                    content_hash=f"h{n}",
                    stage=stage,
                )
                for n, item in enumerate(item_ids, start=1)
            ],
        )
    return [row["id"] for row in repository.stories_for_day(DAY, ticker)]


def theme_set(repository, ticker, story_ids, *, version="v1"):
    with repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage="themes",
        trading_day=DAY,
        pipeline_version=version,
        ticker=ticker,
    ) as run:
        repository.reconcile_themes(
            run=run,
            ticker=ticker,
            trading_day=DAY,
            pipeline_version=version,
            theme_set=ThemeSetRecord(
                method="hdbscan",
                method_reason="clustered",
                config_fingerprint="cfg",
                algorithm_version="m5.1",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            ),
            themes=[
                ThemeRecord(
                    fingerprint="fp",
                    theme_key="k",
                    label="Label",
                    label_source="representative_title",
                    story_ids=tuple(story_ids[:2]),
                    status="ready",
                    salience_rank=1,
                    story_count=2,
                )
            ],
            other_coverage=[
                OtherCoverageRecord(story_id=story_ids[2], reason="clustering_noise")
            ],
            excluded=[
                ExcludedStoryRecord(story_id=s, reason="no_encodable_text")
                for s in story_ids[3:]
            ],
            terminal=True,
        )


def test_theme_population_reads_the_set_its_membership_and_the_generation(tmp_path):
    repository = migrated(tmp_path)
    ids = stories(repository, "NVDA", 4)
    theme_set(repository, "NVDA", ids)

    population = repository.read.theme_population("NVDA", DAY, "v1")
    assert isinstance(population, ThemePopulation)
    assert population.theme_set.method == "hdbscan"
    assert population.theme_set.config_fingerprint == "cfg"
    [theme] = population.themes
    assert (theme.theme_key, theme.label, theme.story_ids) == (
        "k",
        "Label",
        tuple(ids[:2]),
    )
    assert [(o.story_id, o.reason) for o in population.other_coverage] == [
        (ids[2], "clustering_noise")
    ]
    assert [(e.story_id, e.reason) for e in population.excluded] == [
        (ids[3], "no_encodable_text")
    ]
    assert [s.story_id for s in population.stories.stories] == ids
    assert population.stories.stages == {"m3.semantic"}
    assert len(population.stories.signature) == 64
    assert (
        population.stories.signature
        == repository.story_generation("NVDA", DAY, "v1").signature
    )
    assert {m.raw_item_id for m in population.member_provenance} == {
        m.raw_item_id for s in population.stories.stories for m in s.members
    }
    assert all(m.has_payload and m.fetched_at for m in population.member_provenance)


def test_a_partition_without_a_theme_set_reports_its_stories_alone(tmp_path):
    repository = migrated(tmp_path)
    ids = stories(repository, "NVDA", 2, stage="m2.exact")
    population = repository.read.theme_population("NVDA", DAY, "v1")
    assert population.theme_set is None
    assert population.themes == () and population.other_coverage == ()
    assert [s.story_id for s in population.stories.stories] == ids
    assert population.stories.stages == {"m2.exact"}


def test_partition_generations_lists_story_and_theme_only_partitions(tmp_path):
    repository = migrated(tmp_path)
    nvda = stories(repository, "NVDA", 4)
    theme_set(repository, "NVDA", nvda)
    stories(repository, "AMD", 2, stage="m2.exact")
    assert repository.read.partition_generations(DAY) == [
        PartitionGeneration("AMD", DAY, "v1", 2, False),
        PartitionGeneration("NVDA", DAY, "v1", 4, True),
    ]

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET invalidated_at = ? WHERE ticker = 'NVDA'",
            (f"{DAY}T23:00:00+00:00",),
        )
    assert repository.read.partition_generations(DAY)[1] == PartitionGeneration(
        "NVDA", DAY, "v1", 0, True
    )
    population = repository.read.theme_population("NVDA", DAY, "v1")
    assert population.theme_set is not None and population.stories.is_empty


def test_partition_generations_keeps_pipeline_versions_apart(tmp_path):
    repository = migrated(tmp_path)
    stories(repository, "NVDA", 2, version="v1")
    stories(repository, "NVDA", 3, version="v2")
    assert [
        (g.pipeline_version, g.live_story_count)
        for g in repository.read.partition_generations(DAY)
    ] == [
        ("v1", 2),
        ("v2", 3),
    ]
    assert len(repository.read.theme_population("NVDA", DAY, "v2").stories.stories) == 3


def test_the_snapshot_read_is_read_only_and_needs_an_existing_database(tmp_path):
    repository = migrated(tmp_path)
    stories(repository, "NVDA", 1)
    before = repository.database_path.read_bytes()
    repository.read.theme_population("NVDA", DAY, "v1")
    repository.read.partition_generations(DAY)
    assert repository.database_path.read_bytes() == before
    with pytest.raises(Phase0ValidationError):
        Phase0Repository(tmp_path / "absent.db").read.partition_generations(DAY)
    assert not (tmp_path / "absent.db").exists()


def test_the_snapshot_connection_never_leaves_the_reader(tmp_path):
    repository = migrated(tmp_path)
    stories(repository, "NVDA", 1)
    population = repository.read.theme_population("NVDA", DAY, "v1")
    for value in vars(population).values():
        assert not isinstance(value, (sqlite3.Connection, sqlite3.Cursor))

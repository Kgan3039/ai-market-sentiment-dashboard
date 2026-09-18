"""A2: the persisted theme population -> frozen generation input adapter.

Every population here is read back through
``Phase0Reader.theme_population`` from a temporary migrated database that
the ordinary story and theme reconcilers wrote, so what the adapter sees is
what production would see.  The generator is exercised against that input
with a fake client to prove the boundary: nothing is written.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path

import pytest

from ai.guarded_summary import (
    SummaryGenerationInput,
    build_prompt,
    compute_input_fingerprint,
    generate_guarded_summary,
    load_copy_rules,
)
from nlp.dedup.selection import cluster_fingerprint_for
from phase0.models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import Phase0Repository, ThemePopulation
from phase0.summaries import (
    REFUSED_DEGRADED_GENERATION,
    REFUSED_DUPLICATE_MEMBERSHIP,
    REFUSED_MIXED_STAGES,
    REFUSED_NO_LIVE_STORIES,
    REFUSED_NO_THEME_SET,
    REFUSED_SET_INCONSISTENT,
    REFUSED_SOURCE_COUNT,
    REFUSED_SOURCE_COUNT_MALFORMED,
    REFUSED_THEME_STORY_NOT_LIVE,
    REFUSED_UNKNOWN_THEME,
    SummaryInputError,
    assess_population,
    build_generation_input,
)
from phase0.themes import story_description

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-07-23"
TICKER = "TSLA"
VERSION = "v1"
ID_LINE_RE = re.compile(r"- id: (\S+)")
_RUN_IDS = itertools.count(1)
_ITEMS = itertools.count(1)


# ----------------------------------------------------------------------
# Building a realistic persisted day
# ----------------------------------------------------------------------


def migrated(tmp_path: Path) -> Phase0Repository:
    repository = Phase0Repository(tmp_path / "phase0.db")
    repository.migrate()
    return repository


def insert_items(repository, specs):
    """``specs`` is a list of (outlet, description) pairs; returns raw item ids."""

    rows = []
    for outlet, description in specs:
        index = next(_ITEMS)
        rows.append(
            {
                "source": f"yahoo:{outlet}",
                "ticker": TICKER,
                "title": f"{outlet} headline {index}",
                "description": description,
                "url": f"https://{outlet.lower()}.example/{index}",
                "canonical_url": f"https://{outlet.lower()}.example/{index}",
                "published_at": f"{DAY}T10:0{index % 10}:00+00:00",
                "fetched_at": f"{DAY}T11:00:00+00:00",
                "raw_json": {"index": index},
            }
        )
    return [r.item_id for r in repository.admin.insert_raw_items(rows)]


def story_record(
    item_ids, *, title, stage="m3.semantic", outlets=None, published_at=None
):
    """One story whose fingerprint proves its own membership (M2's contract)."""

    outlets = outlets or ["Reuters"] * len(item_ids)
    fingerprint = cluster_fingerprint_for(TICKER, [str(i) for i in item_ids])
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
        outlet_count=len(set(outlets)),
        published_at=published_at or f"{DAY}T10:05:00+00:00",
        canonical_url=f"https://{outlets[0].lower()}.example/{item_ids[0]}",
        content_hash=f"h-{fingerprint[:8]}",
        stage=stage,
        member_story_keys=(fingerprint,),
        algorithm_version="m3.1",
        config_fingerprint="cfg",
        model_name="fake",
        model_revision="r1",
        embedding_dimension=4,
    )


def persist_stories(repository, records, *, version=VERSION):
    with repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage="stories",
        trading_day=DAY,
        pipeline_version=version,
        ticker=TICKER,
    ) as run:
        repository.reconcile_stories(
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=version,
            stories=list(records),
        )
    rows = repository.stories_for_day(DAY, TICKER)
    return {row["canonical_title"]: row["id"] for row in rows}


def persist_theme_set(
    repository,
    themes,
    *,
    other=(),
    excluded=(),
    version=VERSION,
    story_count=None,
):
    """``themes`` is a list of (label, story_ids, citation_item_ids)."""

    with repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage="themes",
        trading_day=DAY,
        pipeline_version=version,
        ticker=TICKER,
    ) as run:
        repository.reconcile_themes(
            run=run,
            ticker=TICKER,
            trading_day=DAY,
            pipeline_version=version,
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
                    fingerprint=f"fp-{rank}",
                    theme_key=f"key-{rank}",
                    label=label,
                    label_source="representative_title",
                    story_ids=tuple(story_ids),
                    citation_item_ids=tuple(citations),
                    status="ready",
                    salience_rank=rank,
                    story_count=len(story_ids),
                )
                for rank, (label, story_ids, citations) in enumerate(themes, start=1)
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


@dataclasses.dataclass
class Day:
    repository: Phase0Repository
    items: dict[str, list[int]]
    story_ids: dict[str, int]
    theme_ids: dict[str, int]

    def population(self, version=VERSION) -> ThemePopulation:
        return self.repository.read.theme_population(TICKER, DAY, version)


def build_day(tmp_path, *, story_count=None, record_story_count=True) -> Day:
    """Two themes, one Other-coverage story, one exclusion, all persisted.

    Theme "Deliveries" holds a two-outlet story (with a description) and a
    single-outlet story; theme "Robotaxi" holds one story.  A fifth story
    sits under Other coverage and a sixth is excluded.
    """

    repository = migrated(tmp_path)
    items = {
        "deliveries": insert_items(
            repository,
            [
                ("Reuters", "Tesla delivered 462,000 vehicles."),
                ("CNBC", "Beat consensus."),
            ],
        ),
        "guidance": insert_items(repository, [("Bloomberg", "")]),
        "robotaxi": insert_items(repository, [("TheVerge", "Robotaxi pilot expands.")]),
        "noise": insert_items(repository, [("Barrons", "Unrelated column.")]),
        "excluded": insert_items(repository, [("Forbes", "")]),
    }
    story_ids = persist_stories(
        repository,
        [
            story_record(
                items["deliveries"],
                title="Tesla Q2 deliveries top estimates",
                outlets=["Reuters", "CNBC"],
            ),
            story_record(
                items["guidance"], title="Tesla guidance", outlets=["Bloomberg"]
            ),
            story_record(
                items["robotaxi"], title="Robotaxi expands", outlets=["TheVerge"]
            ),
            story_record(items["noise"], title="Weekend column", outlets=["Barrons"]),
            story_record(items["excluded"], title="No text", outlets=["Forbes"]),
        ],
    )
    deliveries = [
        story_ids["Tesla Q2 deliveries top estimates"],
        story_ids["Tesla guidance"],
    ]
    robotaxi = [story_ids["Robotaxi expands"]]
    persist_theme_set(
        repository,
        [
            ("Deliveries", deliveries, items["deliveries"] + items["guidance"]),
            ("Robotaxi", robotaxi, items["robotaxi"]),
        ],
        other=[story_ids["Weekend column"]],
        excluded=[story_ids["No text"]],
        story_count=(
            None
            if not record_story_count
            else story_count
            if story_count is not None
            else len(story_ids)
        ),
    )
    population = repository.read.theme_population(TICKER, DAY, VERSION)
    theme_ids = {theme.label: theme.theme_id for theme in population.themes}
    return Day(repository, items, story_ids, theme_ids)


class EchoClient:
    """Cites whatever ids were in the prompt; records every prompt."""

    model = "fake-model"

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, system_prompt, user_prompt, response_schema):
        self.prompts.append(user_prompt)
        ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": "Coverage of deliveries",
                "sentences": [
                    {
                        "text": "Coverage leads with the delivery figure.",
                        "citation_ids": ids[:1],
                    },
                    {"text": "Outlets also carry guidance.", "citation_ids": ids},
                ],
            }
        )


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


def test_persisted_theme_maps_to_frozen_input(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    generation_input = build_generation_input(population, day.theme_ids["Deliveries"])

    assert isinstance(generation_input, SummaryGenerationInput)
    assert generation_input.ticker == TICKER
    assert generation_input.trading_day == DAY
    assert generation_input.theme.theme_id == day.theme_ids["Deliveries"]
    assert generation_input.theme.theme_key == "key-1"
    assert generation_input.theme.label == "Deliveries"
    assert generation_input.theme.pipeline_version == VERSION

    first, second = generation_input.evidence
    assert (
        first.persisted_story_id == day.story_ids["Tesla Q2 deliveries top estimates"]
    )
    assert first.title == "Tesla Q2 deliveries top estimates"
    assert first.description == "Tesla delivered 462,000 vehicles."
    assert first.outlet == "Reuters"
    assert first.published_at == f"{DAY}T10:05:00+00:00"
    assert second.title == "Tesla guidance"
    assert second.description == ""  # the only member has no standfirst
    assert second.outlet == "Bloomberg"


def test_citation_ids_are_namespaced_persisted_story_ids(tmp_path):
    day = build_day(tmp_path)
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    expected = {
        f"story:{day.story_ids['Tesla Q2 deliveries top estimates']}",
        f"story:{day.story_ids['Tesla guidance']}",
    }
    assert generation_input.evidence_ids == expected
    for story in generation_input.evidence:
        assert story.citation_id == f"story:{story.persisted_story_id}"
        assert re.fullmatch(r"story:[1-9]\d*", story.citation_id)
    # Never the non-durable cluster fingerprint.
    fingerprints = {s.cluster_fingerprint for s in day.population().stories.stories}
    assert not generation_input.evidence_ids & fingerprints
    assert not any(fp in build_prompt(generation_input) for fp in fingerprints)


def test_only_the_selected_theme_reaches_the_model(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    prompt = build_prompt(build_generation_input(population, day.theme_ids["Robotaxi"]))

    assert ID_LINE_RE.findall(prompt) == [f"story:{day.story_ids['Robotaxi expands']}"]
    for absent in (
        "Tesla Q2 deliveries",
        "Tesla guidance",
        "Weekend column",
        "No text",
        "Unrelated column",
    ):
        assert absent not in prompt


def test_other_coverage_and_exclusions_never_become_evidence(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    reachable = set()
    for theme in population.themes:
        generation_input = build_generation_input(population, theme.theme_id)
        reachable |= {s.persisted_story_id for s in generation_input.evidence}
    assert reachable == {
        day.story_ids["Tesla Q2 deliveries top estimates"],
        day.story_ids["Tesla guidance"],
        day.story_ids["Robotaxi expands"],
    }
    assert day.story_ids["Weekend column"] not in reachable
    assert day.story_ids["No text"] not in reachable


def test_description_reuses_the_story_description_policy(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    by_id = {s.story_id: s for s in population.stories.stories}
    generation_input = build_generation_input(population, day.theme_ids["Deliveries"])
    for story in generation_input.evidence:
        assert story.description == (
            story_description(by_id[story.persisted_story_id]) or ""
        )


def test_provenance_maps_outlet_timestamp_urls_and_raw_items(tmp_path):
    day = build_day(tmp_path)
    generation_input = build_generation_input(
        day.population(), day.theme_ids["Deliveries"]
    )
    first = generation_input.evidence[0]
    reuters, cnbc = day.items["deliveries"]

    assert first.raw_item_ids == (reuters, cnbc)
    assert first.urls == (
        f"https://reuters.example/{reuters}",
        f"https://cnbc.example/{cnbc}",
    )
    assert first.published_at.endswith("+00:00")
    # Provenance travels beside the evidence, never inside the prompt.
    prompt = build_prompt(generation_input)
    assert "reuters.example" not in prompt and str(reuters) not in ID_LINE_RE.findall(
        prompt
    )


def test_source_population_mutation_cannot_reach_the_frozen_input(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    generation_input = build_generation_input(population, day.theme_ids["Deliveries"])
    before = generation_input.input_fingerprint
    snapshot = dataclasses.replace(generation_input)

    # The population is frozen too; every route to mutation is closed.
    with pytest.raises(dataclasses.FrozenInstanceError):
        population.themes[0].label = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        population.stories.stories[0].canonical_title = "x"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        population.themes[0].story_ids.append(999)  # type: ignore[attr-defined]

    # Re-reading after the database changes gives a *new* population; the
    # input already built keeps describing what it was built from.
    persist_stories(
        day.repository,
        [
            story_record(
                day.items["robotaxi"], title="Robotaxi expands", outlets=["TheVerge"]
            )
        ],
    )
    assert generation_input == snapshot
    assert generation_input.input_fingerprint == before


def test_fingerprint_changes_when_model_visible_evidence_changes(tmp_path):
    day = build_day(tmp_path)
    original = build_generation_input(day.population(), day.theme_ids["Deliveries"])

    # Rewrite the day's stories with one changed title, then re-cluster.
    records = [
        story_record(
            day.items["deliveries"],
            title="Tesla Q2 deliveries top estimates",
            outlets=["Reuters", "CNBC"],
        ),
        story_record(
            day.items["guidance"], title="Tesla raises guidance", outlets=["Bloomberg"]
        ),
        story_record(
            day.items["robotaxi"], title="Robotaxi expands", outlets=["TheVerge"]
        ),
        story_record(day.items["noise"], title="Weekend column", outlets=["Barrons"]),
        story_record(day.items["excluded"], title="No text", outlets=["Forbes"]),
    ]
    story_ids = persist_stories(day.repository, records)
    persist_theme_set(
        day.repository,
        [
            (
                "Deliveries",
                [
                    story_ids["Tesla Q2 deliveries top estimates"],
                    story_ids["Tesla raises guidance"],
                ],
                day.items["deliveries"] + day.items["guidance"],
            ),
            ("Robotaxi", [story_ids["Robotaxi expands"]], day.items["robotaxi"]),
        ],
        other=[story_ids["Weekend column"]],
        excluded=[story_ids["No text"]],
        story_count=5,
    )
    population = day.population()
    theme_id = next(t.theme_id for t in population.themes if t.label == "Deliveries")
    changed = build_generation_input(population, theme_id)
    assert changed.input_fingerprint != original.input_fingerprint
    # Story reconciliation replaced the theme set, so the durable theme row is
    # new; continuity is the theme key.  The evidence alone moves the
    # fingerprint even with the original theme identity held fixed.
    assert changed.theme.theme_key == original.theme.theme_key
    assert (
        compute_input_fingerprint(TICKER, DAY, original.theme, changed.evidence)
        != original.input_fingerprint
    )


# ----------------------------------------------------------------------
# The health gate
# ----------------------------------------------------------------------


def refusal(population, theme_id):
    with pytest.raises(SummaryInputError) as info:
        build_generation_input(population, theme_id)
    return info.value.code


def test_unknown_theme_is_refused(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    assert assess_population(population) is None
    assert refusal(population, 9999) == REFUSED_UNKNOWN_THEME


def test_partition_without_a_theme_set_is_refused(tmp_path):
    repository = migrated(tmp_path)
    items = insert_items(repository, [("Reuters", "text")])
    persist_stories(repository, [story_record(items, title="Only story")])
    population = repository.read.theme_population(TICKER, DAY, VERSION)
    assert assess_population(population)[0] == REFUSED_NO_THEME_SET
    assert refusal(population, 1) == REFUSED_NO_THEME_SET


def test_stale_set_over_a_replaced_generation_is_refused(tmp_path):
    day = build_day(tmp_path)
    theme_id = day.theme_ids["Deliveries"]
    # Story reconciliation with a changed representation drops the theme set
    # (there is no invalidated_at on themes), so the refusal is "no set".
    persist_stories(
        day.repository,
        [
            story_record(
                day.items["robotaxi"], title="Robotaxi expands", outlets=["TheVerge"]
            )
        ],
    )
    population = day.population()
    assert assess_population(population) is not None
    assert refusal(population, theme_id) in {
        REFUSED_NO_THEME_SET,
        REFUSED_SET_INCONSISTENT,
    }


def test_set_that_places_stories_other_than_the_live_ones_is_refused(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    # Same snapshot, but with one live story missing from the accounting.
    trimmed = dataclasses.replace(population, other_coverage=())
    assert assess_population(trimmed)[0] == REFUSED_SET_INCONSISTENT
    assert refusal(trimmed, day.theme_ids["Deliveries"]) == REFUSED_SET_INCONSISTENT
    # And with a theme whose story is not live at all.
    stale_theme = dataclasses.replace(
        population.themes[0], story_ids=population.themes[0].story_ids + (424242,)
    )
    stale = dataclasses.replace(
        population, themes=(stale_theme,) + population.themes[1:]
    )
    assert refusal(stale, stale_theme.theme_id) == REFUSED_SET_INCONSISTENT


def test_theme_story_not_in_live_generation_is_refused(tmp_path, monkeypatch):
    day = build_day(tmp_path)
    population = day.population()
    theme = population.themes[0]
    gone = theme.story_ids[-1]
    # Drop the story from the live generation but leave it on the theme.
    missing_live = dataclasses.replace(
        population,
        stories=dataclasses.replace(
            population.stories,
            stories=tuple(s for s in population.stories.stories if s.story_id != gone),
        ),
    )
    # The accounting gate catches it first: the set places a story the
    # generation does not hold.
    assert assess_population(missing_live)[0] == REFUSED_SET_INCONSISTENT
    assert refusal(missing_live, theme.theme_id) == REFUSED_SET_INCONSISTENT

    # The projection checks membership again on its own, so a gate that
    # somehow passed could still not summarize a story that is not live.
    monkeypatch.setattr(
        "phase0.summaries.require_healthy_population", lambda population: None
    )
    assert refusal(missing_live, theme.theme_id) == REFUSED_THEME_STORY_NOT_LIVE
    # And the other theme, whose members are all live, still builds.
    other = population.themes[1]
    assert (
        build_generation_input(missing_live, other.theme_id).theme.theme_id
        == other.theme_id
    )


def test_m2_only_and_mixed_generations_are_refused(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    degraded = dataclasses.replace(
        population,
        stories=dataclasses.replace(population.stories, stages=frozenset({"m2.exact"})),
    )
    assert assess_population(degraded)[0] == REFUSED_DEGRADED_GENERATION
    assert refusal(degraded, day.theme_ids["Deliveries"]) == REFUSED_DEGRADED_GENERATION

    mixed = dataclasses.replace(
        population,
        stories=dataclasses.replace(
            population.stories, stages=frozenset({"m2.exact", "m3.semantic"})
        ),
    )
    assert assess_population(mixed)[0] == REFUSED_MIXED_STAGES

    # A real M2-only partition, written by the reconciler itself.
    repository = migrated(tmp_path / "m2")
    items = insert_items(repository, [("Reuters", "text")])
    persist_stories(
        repository, [story_record(items, title="Exact only", stage="m2.exact")]
    )
    real = repository.read.theme_population(TICKER, DAY, VERSION)
    assert (
        assess_population(real)[0] == REFUSED_NO_THEME_SET
    )  # decision H: no set exists


def test_set_without_live_stories_is_refused(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    emptied = dataclasses.replace(
        population, stories=dataclasses.replace(population.stories, stories=())
    )
    assert assess_population(emptied)[0] == REFUSED_NO_LIVE_STORIES


def test_recorded_source_story_count_must_match(tmp_path):
    day = build_day(tmp_path, story_count=4)  # the set clustered five
    population = day.population()
    assert assess_population(population)[0] == REFUSED_SOURCE_COUNT
    assert refusal(population, day.theme_ids["Deliveries"]) == REFUSED_SOURCE_COUNT


def test_duplicate_membership_is_refused(tmp_path):
    day = build_day(tmp_path)
    population = day.population()
    first, second = population.themes
    shared = first.story_ids[0]
    doubled = dataclasses.replace(
        population,
        themes=(
            first,
            dataclasses.replace(second, story_ids=second.story_ids + (shared,)),
        ),
    )
    assert assess_population(doubled)[0] == REFUSED_DUPLICATE_MEMBERSHIP
    assert refusal(doubled, first.theme_id) == REFUSED_DUPLICATE_MEMBERSHIP
    # The database refuses the same thing on the way in (migration 014).
    with pytest.raises(Exception):
        persist_theme_set(
            day.repository,
            [
                ("A", [shared], day.items["deliveries"][:1]),
                ("B", [shared], day.items["deliveries"][1:]),
            ],
            other=[
                day.story_ids["Tesla guidance"],
                day.story_ids["Robotaxi expands"],
                day.story_ids["Weekend column"],
            ],
            excluded=[day.story_ids["No text"]],
        )


# ----------------------------------------------------------------------
# The A2/A3 boundary: nothing is written
# ----------------------------------------------------------------------


def test_generation_writes_nothing(tmp_path):
    day = build_day(tmp_path)
    repository = day.repository
    theme_id = day.theme_ids["Deliveries"]

    def snapshot():
        theme = repository.read.theme(theme_id)
        return {
            "summary": theme["summary"],
            "status": theme["status"],
            "citations": theme["citations"],
            "updated_at": theme["updated_at"],
            "run_log": repository.count("run_log"),
            "stage_keys": repository.count("pipeline_stage_keys"),
            "themes": repository.read.count("themes"),
            "stories": repository.read.count("stories"),
            "tables": sorted(repository.read.table_names()),
            "schema_version": repository.schema_version(),
        }

    before = snapshot()
    assert before["summary"] is None and before["status"] == "ready"

    client = EchoClient()
    population = day.population()
    generation_input = build_generation_input(population, theme_id)
    result = generate_guarded_summary(
        generation_input, client=client, rules=load_copy_rules()
    )
    assert result.accepted
    assert result.summary is not None

    assert snapshot() == before
    assert not any("summar" in table for table in before["tables"])
    # The persisted theme row still carries no summary; the result is a value.
    assert repository.read.theme(theme_id)["summary"] is None


def test_no_summary_migration_exists():
    migrations = sorted((ROOT / "phase0" / "migrations").glob("*.sql"))
    assert migrations, "migrations directory is where it was"
    for path in migrations:
        text = path.read_text(encoding="utf-8").lower()
        assert "summaries" not in text and "summary_" not in text, path.name


# ----------------------------------------------------------------------
# P2-4: source_metadata.story_count is optional, and exact when present
# ----------------------------------------------------------------------


def test_absent_story_count_is_allowed_for_compatibility(tmp_path):
    """No ``source_metadata`` block at all: judged on placement alone."""

    day = build_day(tmp_path, record_story_count=False)
    population = day.population()
    assert population.theme_set.source_metadata is None
    assert assess_population(population) is None
    built = build_generation_input(population, day.theme_ids["Deliveries"])
    assert len(built.evidence) == 2


def test_absent_story_count_key_inside_metadata_is_allowed(tmp_path):
    day = build_day(tmp_path)
    with day.repository.admin.connect_writable() as connection:
        connection.execute(
            'UPDATE theme_sets SET source_metadata = \'{"stage": "m3.semantic"}\''
        )
    population = day.population()
    assert "story_count" not in population.theme_set.source_metadata
    assert assess_population(population) is None


def test_correct_integer_story_count_is_accepted(tmp_path):
    day = build_day(tmp_path, story_count=5)
    assert assess_population(day.population()) is None


def test_wrong_integer_story_count_is_refused(tmp_path):
    day = build_day(tmp_path, story_count=4)
    population = day.population()
    assert assess_population(population)[0] == REFUSED_SOURCE_COUNT
    assert refusal(population, day.theme_ids["Deliveries"]) == REFUSED_SOURCE_COUNT


@pytest.mark.parametrize("value", ["5", "999", True, 5.0, [5], {"n": 5}])
def test_malformed_story_count_is_refused_not_coerced(tmp_path, value):
    """Persisted for real, read back for real: ``"5"`` is not ``5``."""

    day = build_day(tmp_path, story_count=value)
    population = day.population()
    assert population.theme_set.source_metadata["story_count"] == value
    code, detail = assess_population(population)
    assert code == REFUSED_SOURCE_COUNT_MALFORMED
    assert type(value).__name__ in detail
    assert (
        refusal(population, day.theme_ids["Deliveries"])
        == REFUSED_SOURCE_COUNT_MALFORMED
    )

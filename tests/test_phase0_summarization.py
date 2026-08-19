"""Tests for cache-aware theme summarization into Phase 0 (issue #73 / A3).

Pytest-style, matching the rest of the ``phase0`` test suite (see
``tests/test_phase0_repository.py``), not ``tests/test_ai_summarization.py``'s
``unittest.TestCase`` style.
"""

from __future__ import annotations

import itertools
import re

import pytest

from ai.summarization import GenerationUsage
from nlp.themes.models import ClusteringMethod
from phase0.models import StoryMemberRecord, StoryRecord
from phase0.repository import Phase0Repository
from phase0.summarization import compute_content_hash, summarize_theme_set
from tests.test_theme_clustering import AngleEncoder, run, story, three_strand_day

TICKER = "NVDA"
TRADING_DAY = "2026-03-05"
PIPELINE_VERSION = "v1"
ID_LINE_RE = re.compile(r"- id: (\S+)")

_RUN_IDS = itertools.count(1)


def _next_run_id() -> str:
    return f"run-{next(_RUN_IDS)}"


def migrated(tmp_path) -> Phase0Repository:
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    return repository


def seed_phase0_from_theme_stories(repository: Phase0Repository, theme_stories) -> None:
    """Insert raw_items + stories matching M5's ThemeStory fixtures exactly.

    ``cluster_fingerprint`` is set to each story's ``story_key`` - the
    field `phase0.summarization.summarize_theme_set` resolves through
    `Phase0Repository.story_ids_for_fingerprints`.
    """

    raw_item_id_by_item_id: dict[str, int] = {}
    for theme_story in theme_stories:
        for item_id in theme_story.item_ids:
            result = repository.admin.insert_raw_item(
                {
                    "source": f"test:{item_id}",
                    "ticker": TICKER,
                    "title": theme_story.title,
                    "description": theme_story.description or "",
                    "url": f"https://example.com/{item_id}",
                    "canonical_url": f"https://example.com/{item_id}",
                    "published_at": theme_story.published_at.isoformat(),
                    "fetched_at": theme_story.published_at.isoformat(),
                    "raw_json": {"title": theme_story.title},
                }
            )
            raw_item_id_by_item_id[item_id] = result.item_id

    stories = []
    for theme_story in theme_stories:
        member_raw_ids = [raw_item_id_by_item_id[item_id] for item_id in theme_story.item_ids]
        stories.append(
            StoryRecord(
                cluster_fingerprint=theme_story.story_key,
                canonical_title=theme_story.title,
                members=tuple(
                    StoryMemberRecord(
                        raw_item_id=raw_id,
                        position=position,
                        outlet=theme_story.outlets[0] if theme_story.outlets else None,
                    )
                    for position, raw_id in enumerate(member_raw_ids)
                ),
                outlet_count=len(theme_story.outlets) or 1,
                content_hash=f"hash-{theme_story.story_key}",
            )
        )
    repository.admin.reconcile_stories(
        ticker=TICKER,
        trading_day=TRADING_DAY,
        pipeline_version=PIPELINE_VERSION,
        stories=stories,
    )


class FakeGeminiClient:
    """Deterministic stand-in, same prompt-id-extraction trick as
    tests/test_ai_summarization.py's fake, plus a fixed `.usage` so the
    run_log token accounting has something real to assert on."""

    def __init__(self) -> None:
        self.calls = 0
        self.usage = GenerationUsage(input_tokens=42, output_tokens=7)

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        self.calls += 1
        story_ids = ID_LINE_RE.findall(user_prompt)
        assert story_ids
        sentences = [
            {"text": f"Coverage sentence {index + 1} about the theme.", "citation_ids": [story_id]}
            for index, story_id in enumerate(story_ids[:2])
        ]
        if len(sentences) < 2:
            sentences.append({"text": "Additional coverage.", "citation_ids": [story_ids[0]]})
        return response_schema.model_validate({"label": "Coverage of recent developments", "sentences": sentences})


class NeverCallGeminiClient:
    """Fails the test immediately if the caching layer calls it at all."""

    def generate(self, system_prompt: str, user_prompt: str, response_schema):
        raise AssertionError("summarize() must not be called for an unchanged theme")


@pytest.fixture
def seeded_repository_and_theme_set(tmp_path):
    stories, encoder = three_strand_day()
    repository = migrated(tmp_path)
    seed_phase0_from_theme_stories(repository, stories)
    theme_set = run(stories, encoder)
    assert len(theme_set.themes) == 3  # a1/a2, b1/b2, c1/c2
    return repository, theme_set


def test_first_run_calls_the_llm_for_every_theme(seeded_repository_and_theme_set):
    repository, theme_set = seeded_repository_and_theme_set

    report = summarize_theme_set(
        theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )

    assert report.cache_misses == 3
    assert report.cache_hits == 0
    assert len(report.attempts) == 3
    assert repository.count("themes") == 3
    stored = repository.theme_set(
        ticker=TICKER, trading_day=TRADING_DAY, pipeline_version=PIPELINE_VERSION
    )
    for row in stored["themes"]:
        assert row["summary"]
        assert row["content_hash"]
        assert row["label"]


def test_rerun_with_unchanged_data_makes_zero_llm_calls(seeded_repository_and_theme_set):
    """The literal issue #73 DoD, proven against the real repository."""

    repository, theme_set = seeded_repository_and_theme_set

    first = summarize_theme_set(
        theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )
    assert first.cache_misses == 3

    second = summarize_theme_set(
        theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=NeverCallGeminiClient(),
    )

    assert second.cache_hits == 3
    assert second.cache_misses == 0
    assert second.attempts == ()


def test_a_changed_story_only_invalidates_its_own_theme(seeded_repository_and_theme_set):
    repository, theme_set = seeded_repository_and_theme_set
    summarize_theme_set(
        theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )

    # Rebuild the "a" theme's stories with a1's title changed; b/c untouched.
    stories, encoder = three_strand_day()
    changed_stories = [
        story("a1", "earnings one but rewritten", 0) if s.story_key == "a1" else s
        for s in stories
    ]
    encoder.angles["earnings one but rewritten"] = encoder.angles["earnings one"]
    changed_theme_set = run(changed_stories, encoder)
    assert len(changed_theme_set.themes) == 3

    report = summarize_theme_set(
        changed_theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )

    assert report.cache_misses == 1
    assert report.cache_hits == 2


def test_content_hash_changes_with_story_text_not_just_membership():
    stories, _ = three_strand_day()
    from nlp.themes.summarization import theme_to_summarizer_input

    theme_set = run(stories, AngleEncoder({s.title: 0.0 for s in stories}))
    # Membership-only fingerprint would be identical; content_hash must not be.
    hashes = set()
    for theme in theme_set.themes:
        hashes.add(compute_content_hash(theme_to_summarizer_input(theme)))
    assert len(hashes) == len(theme_set.themes)


def test_run_log_records_cache_and_token_accounting(seeded_repository_and_theme_set):
    repository, theme_set = seeded_repository_and_theme_set

    summarize_theme_set(
        theme_set,
        repository=repository,
        run_id="run-log-check",
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )

    entries = repository.run_log_entries(run_id="run-log-check", stage="summarize")
    assert len(entries) == 1
    counts = entries[0]["counts"]
    assert counts["cache_misses"] == 3
    assert counts["cache_hits"] == 0
    assert counts["llm_calls"] == 3
    assert counts["input_size"] == 3 * 42
    assert counts["output_size"] == 3 * 7
    assert len(counts["llm_call_log"]) == 3
    assert entries[0]["status"] == "success"


def test_other_coverage_and_excluded_stories_are_persisted_too(tmp_path):
    stories, encoder = three_strand_day()
    # Add a lone story that will not cluster with anything (an outlier).
    outlier = story("lonely", "an unrelated wire story", 20)
    encoder.angles["an unrelated wire story"] = 300.0
    all_stories = stories + [outlier]

    repository = migrated(tmp_path)
    seed_phase0_from_theme_stories(repository, all_stories)
    theme_set = run(all_stories, encoder)
    assert theme_set.other_coverage or theme_set.excluded

    summarize_theme_set(
        theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )

    stored = repository.theme_set(
        ticker=TICKER, trading_day=TRADING_DAY, pipeline_version=PIPELINE_VERSION
    )
    assert len(stored["other_coverage"]) + len(stored["excluded"]) == (
        len(theme_set.other_coverage) + len(theme_set.excluded)
    )


def test_stored_theme_method_matches_clustering_method(seeded_repository_and_theme_set):
    repository, theme_set = seeded_repository_and_theme_set
    assert theme_set.method in (ClusteringMethod.HDBSCAN, ClusteringMethod.AGGLOMERATIVE)

    summarize_theme_set(
        theme_set,
        repository=repository,
        run_id=_next_run_id(),
        pipeline_version=PIPELINE_VERSION,
        client=FakeGeminiClient(),
    )

    stored = repository.theme_set(
        ticker=TICKER, trading_day=TRADING_DAY, pipeline_version=PIPELINE_VERSION
    )
    for row in stored["themes"]:
        assert row["method"] == theme_set.method.value

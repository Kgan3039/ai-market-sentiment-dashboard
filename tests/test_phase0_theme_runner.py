"""The ``themes`` stage: persisted stories in, one theme generation out.

Every test drives the real repository, the real story stage, and the real
M5 clustering.  Only the encoder is a fake, and it is deterministic, so no
test here loads a model or touches the sentence-transformers cache.

What is pinned down is the boundary, not the clustering: which generation
is authoritative afterwards, what continuity survives, and what a refusal
looks like from outside.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nlp.embeddings import EMBEDDING_DTYPE, serialize_vector
from nlp.themes import PreviousTheme, ThemeConfig
from nlp.themes.errors import ThemeCapacityError
from nlp.dedup.selection import cluster_fingerprint_for
from phase0.errors import Phase0IntegrityError
from phase0.models import ThemeRecord, ThemeSetRecord
from phase0.repository import (
    STAGE_DEGRADED,
    PersistedStory,
    PersistedStoryMember,
    Phase0Repository,
    PreviousThemeGeneration,
    StoryGenerationConflict,
    ThemeIdentity,
)
from phase0.stories import StoryReconciler
from phase0.themes import (
    CanonicalClusterUnrecoverable,
    DEGRADATION_REASON,
    canonical_cluster_members,
    DESCRIPTION_POLICY,
    GENERATION_EMPTY,
    GENERATION_EXACT,
    GENERATION_SEMANTIC,
    STAGE,
    ThemeGenerationError,
    ThemePartitionOutcome,
    ThemeReconciler,
    classify_generation,
    evaluate_previous_themes,
    expected_provenance,
    story_description,
    theme_story,
)

DAY = "2026-07-23"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class FakeEncoder:
    """Deterministic stand-in for :class:`nlp.embeddings.EmbeddingService`.

    Vectors are derived from a digest of the text, so identical headlines
    encode identically and no similarity is accidental.  Texts are kept so
    a test can assert what M5 was actually asked to embed.
    """

    model_name = "fake-encoder"
    model_revision = "rev-1"
    dimension = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    @property
    def texts(self) -> list[str]:
        return [text for call in self.calls for text in call]

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        return [
            [byte / 255.0 for byte in hashlib.sha256(text.encode()).digest()[:8]]
            for text in texts
        ]


class RaisingEncoder(FakeEncoder):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    def embed_batch(self, texts):
        super().embed_batch(texts)
        raise self._error


class DifferentEncoder(FakeEncoder):
    model_name = "other-encoder"
    model_revision = "rev-9"


def _member(**overrides) -> PersistedStoryMember:
    defaults = {
        "raw_item_id": 1,
        "position": 0,
        "outlet": "yahoo barrons",
        "url": None,
        "canonical_url": None,
        "match_reason": "canonical",
        "quarantined": False,
        "description": None,
    }
    defaults.update(overrides)
    return PersistedStoryMember(**defaults)


def migrated(tmp_path: Path, name: str = "phase0.sqlite3") -> Phase0Repository:
    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    return repository


def evidence(
    repository: Phase0Repository,
    index: int,
    *,
    ticker: str = "NVDA",
    title: str | None = None,
    description: str | None = None,
    published_at: str | None = None,
    day: str = DAY,
) -> int:
    link = f"https://publisher.example/{ticker.lower()}/{index}"
    return repository.admin.insert_raw_items(
        [
            {
                "source": "yahoo:Barron's",
                "ticker": ticker,
                "title": title or f"{ticker} headline {index}",
                "description": description,
                "url": link,
                "canonical_url": link,
                "published_at": published_at or f"{day}T{index % 24:02d}:00:00+00:00",
                "fetched_at": f"{day}T23:30:00+00:00",
                "ingest_status": "valid",
                "external_id": f"prov-{ticker}-{index}",
                "raw_json": json.dumps({"index": index}),
            }
        ]
    )[0].item_id


def seed_stories(
    repository: Phase0Repository,
    *,
    ticker: str = "NVDA",
    count: int = 6,
    encoder: FakeEncoder | None = None,
) -> StoryReconciler:
    """Persist a healthy ``m3.semantic`` generation through the real stage."""

    for index in range(1, count + 1):
        evidence(
            repository,
            index,
            ticker=ticker,
            title=f"{ticker} topic {index % 2} variant {index}",
            description=f"Body {index}",
        )
    runner = StoryReconciler(
        repository,
        pipeline_version="v1",
        encoder=encoder if encoder is not None else FakeEncoder(),
    )
    runner.run_partition(ticker, DAY, base_run_id="stories-seed")
    return runner


def themes(
    repository: Phase0Repository, *, encoder=None, config: ThemeConfig | None = None
) -> ThemeReconciler:
    return ThemeReconciler(
        repository,
        pipeline_version="v1",
        encoder=encoder if encoder is not None else FakeEncoder(),
        config=config,
    )


def theme_rows(repository: Phase0Repository, ticker: str = "NVDA"):
    stored = repository.theme_set(ticker=ticker, trading_day=DAY, pipeline_version="v1")
    return stored


def run_rows(repository: Phase0Repository, stage: str = STAGE) -> list[dict]:
    return [row for row in repository.read.run_log_rows() if row["stage"] == stage]


def markers(row) -> list[dict]:
    return [
        error
        for error in json.loads(row["errors"])
        if isinstance(error, dict) and error.get("type") == STAGE_DEGRADED
    ]


def seed_theme_set(
    repository: Phase0Repository,
    *,
    ticker: str = "NVDA",
    algorithm_version: str = "m5.themes.v1",
    config_fingerprint: str = "cfg-1",
    model_name: str = "fake-encoder",
    model_revision: str | None = "rev-1",
    embedding_dimension: int | None = 8,
    centroid: bytes | None = None,
    theme_key: str = "carried-over",
) -> int:
    """Write one theme set with chosen provenance, through the real path."""

    stored = repository.stories_for_day(DAY, ticker)
    assert stored, "seed stories before seeding a theme set"
    vector = (
        centroid
        if centroid is not None
        else serialize_vector(np.ones(embedding_dimension or 8, dtype=EMBEDDING_DTYPE))
    )
    with repository.stage_run(
        run_id=f"seed-themes-{ticker}",
        stage=STAGE,
        trading_day=DAY,
        pipeline_version="v1",
        ticker=ticker,
    ) as run:
        repository.reconcile_themes(
            run=run,
            ticker=ticker,
            trading_day=DAY,
            pipeline_version="v1",
            theme_set=ThemeSetRecord(
                method="hdbscan",
                method_reason="seeded",
                config_fingerprint=config_fingerprint,
                algorithm_version=algorithm_version,
                model_name=model_name,
                model_revision=model_revision,
                embedding_dimension=embedding_dimension,
            ),
            themes=[
                ThemeRecord(
                    fingerprint="seeded",
                    theme_key=theme_key,
                    label="Seeded theme",
                    story_ids=(stored[0]["id"],),
                    citation_item_ids=(json.loads(stored[0]["member_ids"])[0],),
                    method="hdbscan",
                    salience_rank=1,
                    centroid=vector,
                    algorithm_version=algorithm_version,
                    config_fingerprint=config_fingerprint,
                    model_name=model_name,
                    model_revision=model_revision,
                    embedding_dimension=embedding_dimension,
                )
            ],
            terminal=True,
        )
    return stored[0]["id"]


def identity(**overrides) -> ThemeIdentity:
    defaults = {
        "theme_key": "key-1",
        "algorithm_version": "m5.themes.v1",
        "config_fingerprint": "cfg-1",
        "model_name": "fake-encoder",
        "model_revision": "rev-1",
        "embedding_dimension": 8,
        "centroid": serialize_vector(np.ones(8, dtype=EMBEDDING_DTYPE)),
    }
    defaults.update(overrides)
    return ThemeIdentity(**defaults)


def generation(*identities, **overrides) -> PreviousThemeGeneration:
    """A stored theme generation with chosen set-level provenance."""

    defaults = {
        "ticker": "NVDA",
        "trading_day": DAY,
        "pipeline_version": "v1",
        "algorithm_version": "m5.themes.v1",
        "config_fingerprint": "cfg-1",
        "model_name": "fake-encoder",
        "model_revision": "rev-1",
        "embedding_dimension": 8,
    }
    defaults.update(overrides)
    defaults["identities"] = tuple(identities)
    return PreviousThemeGeneration(**defaults)


def matching_generation(expected, *identities, **overrides) -> PreviousThemeGeneration:
    """A generation whose set-level provenance equals ``expected``."""

    fields = {
        "algorithm_version": expected.algorithm_version,
        "config_fingerprint": expected.config_fingerprint,
        "model_name": expected.model_name,
        "model_revision": expected.model_revision,
        "embedding_dimension": expected.embedding_dimension,
    }
    fields.update(overrides)
    return generation(*identities, **fields)


def child(expected, **overrides) -> ThemeIdentity:
    """A child identity agreeing with ``expected`` unless overridden."""

    fields = {
        "algorithm_version": expected.algorithm_version,
        "config_fingerprint": expected.config_fingerprint,
        "model_name": expected.model_name,
        "model_revision": expected.model_revision,
        "embedding_dimension": expected.embedding_dimension,
    }
    fields.update(overrides)
    return identity(**fields)


def expectation(**overrides):
    encoder = FakeEncoder()
    expected = expected_provenance(
        ThemeConfig(supported_tickers=["NVDA", "AMD", "AAPL"]), encoder
    )
    return dataclasses.replace(expected, **overrides) if overrides else expected


# ----------------------------------------------------------------------
# T1, T24, T26 -- the healthy path
# ----------------------------------------------------------------------


def test_a_healthy_generation_produces_a_persisted_theme_set(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository)

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "success"
    assert outcome.generation == GENERATION_SEMANTIC
    assert outcome.theme_count >= 1

    stored = theme_rows(repository)
    assert stored is not None
    assert stored["algorithm_version"] == "m5.themes.v1"
    assert stored["config_fingerprint"]
    assert stored["model_name"] == "fake-encoder"
    assert stored["model_revision"] == "rev-1"
    assert stored["embedding_dimension"] == 8
    assert stored["source_metadata"]["stage"] == "m3.semantic"
    for theme in stored["themes"]:
        assert theme["centroid"] is not None
        assert theme["algorithm_version"] == "m5.themes.v1"
        assert theme["status"] == "ready"
        assert theme["method"]

    ledger = run_rows(repository)
    assert [row["status"] for row in ledger] == ["success"]
    assert markers(ledger[0]) == []


def test_the_stage_persists_no_embeddings(tmp_path):
    """Decision A: vectors are computed in memory and used, not stored."""

    repository = migrated(tmp_path)
    seed_stories(repository)
    themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert repository.count("embeddings") == 0


def test_an_idempotent_replay_is_deterministic(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository)
    runner = themes(repository)
    runner.run_partition("NVDA", DAY, base_run_id="themes-1")
    before = theme_rows(repository)

    runner.run_partition("NVDA", DAY, base_run_id="themes-2")
    after = theme_rows(repository)

    assert [theme["fingerprint"] for theme in after["themes"]] == [
        theme["fingerprint"] for theme in before["themes"]
    ]
    assert [theme["theme_key"] for theme in after["themes"]] == [
        theme["theme_key"] for theme in before["themes"]
    ]
    assert after["config_fingerprint"] == before["config_fingerprint"]


def test_a_complete_reconcile_removes_obsolete_themes(tmp_path):
    """The theme set is authoritative: what is not in it is deleted."""

    repository = migrated(tmp_path)
    seed_stories(repository)
    seed_theme_set(repository, theme_key="stale-key")
    assert [t["theme_key"] for t in theme_rows(repository)["themes"]] == ["stale-key"]

    themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    keys = {theme["theme_key"] for theme in theme_rows(repository)["themes"]}
    assert "stale-key" not in keys
    with repository.admin.connect_writable() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM themes WHERE fingerprint = 'seeded'"
            ).fetchone()[0]
            == 0
        )


# ----------------------------------------------------------------------
# T4 -- T10 -- the compatibility gate
# ----------------------------------------------------------------------


#: Set-level provenance that refuses the whole stored generation.
GENERATION_MISMATCHES = {
    "algorithm": {"algorithm_version": "m5.themes.v0"},
    "config": {"config_fingerprint": "cfg-moved"},
    "model": {"model_name": "other-encoder"},
    "revision": {"model_revision": "rev-2"},
    "revision_null": {"model_revision": None},
    "dimension": {"embedding_dimension": 384},
}


@pytest.mark.parametrize("case", sorted(GENERATION_MISMATCHES))
def test_an_incompatible_generation_is_refused_whole(case):
    """The set decides, and it refuses every child with itself."""

    expected = expectation()
    override = GENERATION_MISMATCHES[case]
    reason = case.split("_")[0]
    # Children agree with the *set*, so nothing is wrong with them alone.
    stored = matching_generation(
        expected,
        child(expected, theme_key="a", **override),
        child(expected, theme_key="b", **override),
        **override,
    )

    capture = evaluate_previous_themes(stored, expected)

    assert capture.compatible == ()
    assert capture.generation_rejected == reason
    assert capture.has_incompatible is True
    assert capture.counts[f"previous_themes_rejected_{reason}"] == 2
    assert capture.counts["previous_themes_rejected"] == 2
    assert capture.counts["previous_theme_generation_rejected"] == 1
    assert capture.counts["previous_theme_generation_seen"] == 1


def test_an_incompatible_zero_theme_generation_is_still_refused():
    """P1-1: a set with no themes is still a generation.

    A day below the clustering floor stores exactly this -- provenance and
    no children.  Judging it by its themes would find an empty list and
    conclude there was nothing incompatible to clear.
    """

    expected = expectation()
    stored = matching_generation(expected, model_name="an-older-encoder")

    capture = evaluate_previous_themes(stored, expected)

    assert capture.identities == ()
    assert capture.generation_present is True
    assert capture.generation_rejected == "model"
    assert capture.has_incompatible is True
    assert capture.counts["previous_theme_generation_seen"] == 1
    assert capture.counts["previous_theme_generation_rejected"] == 1
    # No children, so nothing to count as an identity rejection.
    assert capture.counts["previous_themes_rejected"] == 0


def test_a_compatible_zero_theme_generation_is_left_alone():
    """Themeless is not incompatible; there is nothing to clear."""

    expected = expectation()
    capture = evaluate_previous_themes(matching_generation(expected), expected)

    assert capture.generation_present is True
    assert capture.generation_rejected is None
    assert capture.has_incompatible is False
    assert capture.compatible == ()
    assert capture.counts["previous_theme_generation_rejected"] == 0


def test_set_level_provenance_overrides_apparently_compatible_children():
    """P1-1: children matching the runtime cannot rescue their own set."""

    expected = expectation()
    stored = matching_generation(
        expected,
        # These agree with the *upcoming run* exactly...
        child(expected, theme_key="a"),
        child(expected, theme_key="b"),
        # ...while the set that holds them does not.
        model_name="an-older-encoder",
    )

    capture = evaluate_previous_themes(stored, expected)

    assert capture.compatible == ()
    assert capture.generation_rejected == "model"
    assert capture.has_incompatible is True
    assert capture.counts["previous_themes_rejected_model"] == 2


def test_mixed_child_provenance_never_partially_reuses_identities():
    """P1-1: half a generation is not a generation."""

    expected = expectation()
    stored = matching_generation(
        expected,
        child(expected, theme_key="agrees"),
        child(expected, theme_key="disagrees", config_fingerprint="cfg-other"),
    )

    capture = evaluate_previous_themes(stored, expected)

    assert capture.compatible == ()
    assert capture.generation_rejected == "inconsistent"
    assert capture.has_incompatible is True
    assert capture.counts["previous_themes_rejected_inconsistent"] == 2


def test_a_compatible_identity_becomes_a_previous_theme():
    expected = expectation()
    stored = matching_generation(expected, child(expected, theme_key="key-1"))

    capture = evaluate_previous_themes(stored, expected)

    assert len(capture.compatible) == 1
    carried = capture.compatible[0]
    assert isinstance(carried, PreviousTheme)
    assert carried.theme_key == "key-1"
    assert len(carried.centroid) == 8
    assert capture.rejected == {}
    assert capture.generation_rejected is None
    assert capture.has_incompatible is False


@pytest.mark.parametrize(
    "centroid",
    [
        None,
        b"not-a-vector",
        serialize_vector(np.ones(4, dtype=EMBEDDING_DTYPE)),
        serialize_vector(np.ones(8, dtype=EMBEDDING_DTYPE))[:-4],
    ],
    ids=["missing", "malformed", "wrong_width", "truncated"],
)
def test_a_bad_centroid_costs_its_own_identity_only(centroid):
    """It never crashes, never reaches M5, and never refuses the set."""

    expected = expectation()
    stored = matching_generation(
        expected,
        child(expected, theme_key="good"),
        child(expected, theme_key="bad", centroid=centroid),
    )

    capture = evaluate_previous_themes(stored, expected)

    assert [entry.theme_key for entry in capture.compatible] == ["good"]
    assert capture.rejected == {"centroid": 1}
    # A corrupt centroid says nothing about the generation's provenance.
    assert capture.generation_rejected is None
    assert capture.has_incompatible is False


def test_no_previous_generation_is_not_an_incompatibility():
    capture = evaluate_previous_themes(None, expectation())

    assert capture.counts["previous_theme_generation_seen"] == 0
    assert capture.counts["previous_themes_seen"] == 0
    assert capture.counts["previous_themes_rejected"] == 0
    # Nothing stored is not the same fact as nothing reusable.
    assert capture.has_incompatible is False


def test_rejection_counts_are_reported_without_degrading_the_run(tmp_path):
    """A config transition is not itself a degraded current run."""

    repository = migrated(tmp_path)
    seed_stories(repository)
    seed_theme_set(repository, config_fingerprint="a-previous-config")
    previous = themes(repository).capture_previous("NVDA", DAY)
    assert previous is not None
    assert len(previous.identities) == 1

    outcome = themes(repository).run_partition(
        "NVDA", DAY, base_run_id="themes-1", previous=previous
    )

    assert outcome.status == "success"
    assert outcome.degradation_reason is None
    assert outcome.counts["previous_themes_seen"] == 1
    assert outcome.counts["previous_themes_rejected"] == 1
    assert outcome.counts["previous_themes_rejected_config"] == 1
    assert outcome.counts["previous_themes_compatible"] == 0
    assert outcome.counts["previous_theme_generation_rejected"] == 1

    ledger = run_rows(repository)
    settled = [row for row in ledger if "themes-1" in row["run_id"]][0]
    assert settled["status"] == "success"
    assert markers(settled) == []


# ----------------------------------------------------------------------
# T3 -- continuity
# ----------------------------------------------------------------------


def test_a_compatible_identity_carries_the_theme_key_across_a_change(tmp_path):
    """AC-4: a theme that gained a story keeps the name it had."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=6)
    runner = themes(repository)
    runner.run_partition("NVDA", DAY, base_run_id="themes-1")
    before = theme_rows(repository)
    carried = {theme["theme_key"]: theme["fingerprint"] for theme in before["themes"]}
    assert carried

    # A seventh article joins the day, so at least one theme's membership
    # -- and therefore its fingerprint -- moves.
    evidence(
        repository,
        7,
        title="NVDA topic 1 variant 7",
        description="Body 7",
    )
    previous = runner.capture_previous("NVDA", DAY)
    StoryReconciler(
        repository, pipeline_version="v1", encoder=FakeEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-2")
    outcome = runner.run_partition(
        "NVDA", DAY, base_run_id="themes-2", previous=previous
    )

    assert outcome.status == "success"
    assert outcome.counts["previous_themes_compatible"] == len(previous.identities)
    after = theme_rows(repository)
    matched = [
        theme for theme in after["themes"] if theme["matched_previous_key"] is not None
    ]
    assert matched, "no theme carried a previous identity over"
    for theme in matched:
        assert theme["theme_key"] == theme["matched_previous_key"]
        assert theme["theme_key"] in carried


# ----------------------------------------------------------------------
# T11 -- T13 -- incompatible live sets and failure
# ----------------------------------------------------------------------


def test_an_incompatible_live_set_is_cleared_before_m5_runs(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository)
    seed_theme_set(repository, model_name="an-older-encoder", theme_key="old-key")
    previous = themes(repository).capture_previous("NVDA", DAY)

    outcome = themes(repository).run_partition(
        "NVDA", DAY, base_run_id="themes-1", previous=previous
    )

    assert outcome.status == "success"
    assert outcome.cleared is True
    keys = {theme["theme_key"] for theme in theme_rows(repository)["themes"]}
    assert "old-key" not in keys
    assert theme_rows(repository)["model_name"] == "fake-encoder"


def test_an_incompatible_clear_survives_a_later_m5_failure(tmp_path):
    """Intentional: the incompatible generation does not come back."""

    repository = migrated(tmp_path)
    seed_stories(repository)
    seed_theme_set(repository, model_name="an-older-encoder", theme_key="old-key")
    previous = themes(repository).capture_previous("NVDA", DAY)
    assert repository.count("theme_sets") == 1

    outcome = themes(
        repository, encoder=RaisingEncoder(RuntimeError("M5 fell over"))
    ).run_partition("NVDA", DAY, base_run_id="themes-1", previous=previous)

    assert outcome.status == "failed"
    assert repository.count("themes") == 0
    assert repository.count("theme_sets") == 0
    settled = [row for row in run_rows(repository) if "themes-1" in row["run_id"]][0]
    assert settled["status"] == "failed"


def test_a_compatible_set_survives_an_m5_failure(tmp_path):
    """A failed refresh is not a reason to destroy a valid generation."""

    repository = migrated(tmp_path)
    seed_stories(repository)
    runner = themes(repository)
    runner.run_partition("NVDA", DAY, base_run_id="themes-1")
    before = theme_rows(repository)
    previous = runner.capture_previous("NVDA", DAY)

    outcome = themes(
        repository, encoder=RaisingEncoder(RuntimeError("M5 fell over"))
    ).run_partition("NVDA", DAY, base_run_id="themes-2", previous=previous)

    assert outcome.status == "failed"
    assert outcome.cleared is False
    after = theme_rows(repository)
    assert [theme["theme_key"] for theme in after["themes"]] == [
        theme["theme_key"] for theme in before["themes"]
    ]
    settled = [row for row in run_rows(repository) if "themes-2" in row["run_id"]][0]
    assert settled["status"] == "failed"


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("something nobody predicted"),
        ThemeCapacityError("NVDA", 999, 250),
    ],
    ids=["unexpected", "capacity"],
)
def test_every_m5_failure_fails_the_partition(tmp_path, error):
    """There is no lower theme tier to fall back to."""

    repository = migrated(tmp_path)
    seed_stories(repository)

    outcome = themes(repository, encoder=RaisingEncoder(error)).run_partition(
        "NVDA", DAY, base_run_id="themes-1"
    )

    assert outcome.status == "failed"
    assert outcome.degradation_reason is None
    assert theme_rows(repository) is None
    settled = run_rows(repository)[0]
    assert settled["status"] == "failed"
    assert markers(settled) == []


# ----------------------------------------------------------------------
# T14 -- T20 -- the story generation guard
# ----------------------------------------------------------------------


def test_zero_stories_clears_stale_themes_and_settles_success(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    seed_theme_set(repository)
    assert repository.count("theme_sets") == 1

    with repository.admin.connect_writable() as connection:
        connection.execute("DELETE FROM theme_citations")
        connection.execute("DELETE FROM theme_stories")
        connection.execute("DELETE FROM themes")
        connection.execute("DELETE FROM theme_other_coverage")
        connection.execute("DELETE FROM theme_excluded_stories")
        connection.execute("UPDATE stories SET canonical_item_id = NULL")
        connection.execute("DELETE FROM story_members")
        connection.execute("DELETE FROM stories")

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "success"
    assert outcome.generation == GENERATION_EMPTY
    assert outcome.degradation_reason is None
    assert repository.count("theme_sets") == 0
    settled = [row for row in run_rows(repository) if "themes-1" in row["run_id"]][0]
    assert settled["status"] == "success"
    assert markers(settled) == []


def test_an_already_empty_partition_is_a_successful_no_op(tmp_path):
    repository = migrated(tmp_path)

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "success"
    assert outcome.generation == GENERATION_EMPTY
    assert outcome.cleared is False
    assert repository.count("theme_sets") == 0
    settled = run_rows(repository)[0]
    assert settled["status"] == "success"
    assert settled["partial_count"] == 0
    assert json.loads(settled["errors"]) == []


def test_an_m2_only_generation_is_degraded_and_ships_no_themes(tmp_path):
    repository = migrated(tmp_path)
    from nlp.embeddings import EmbeddingModelLoadError

    for index in range(1, 4):
        evidence(repository, index, description=f"Body {index}")
    StoryReconciler(
        repository,
        pipeline_version="v1",
        encoder=RaisingEncoder(EmbeddingModelLoadError("no model cache")),
    ).run_partition("NVDA", DAY, base_run_id="stories-1")
    assert {row["stage"] for row in repository.stories_for_day(DAY, "NVDA")} == {
        "m2.exact"
    }
    seed_theme_set(repository)

    encoder = FakeEncoder()
    outcome = themes(repository, encoder=encoder).run_partition(
        "NVDA", DAY, base_run_id="themes-1"
    )

    assert outcome.status == "degraded"
    assert outcome.generation == GENERATION_EXACT
    assert outcome.degradation_reason == DEGRADATION_REASON
    assert encoder.calls == [], "M5 must not be attempted over M2 output"
    assert repository.count("theme_sets") == 0
    assert repository.count("themes") == 0

    settled = [row for row in run_rows(repository) if "themes-1" in row["run_id"]][0]
    assert settled["status"] == "degraded"
    assert [marker["reason"] for marker in markers(settled)] == [DEGRADATION_REASON]


def _break_stage(repository: Phase0Repository, sql: str) -> None:
    with repository.admin.connect_writable() as connection:
        connection.execute(sql)


@pytest.mark.parametrize(
    "sql, fragment",
    [
        (
            "UPDATE stories SET stage = 'm2.exact' WHERE id = "
            "(SELECT MIN(id) FROM stories)",
            "mixes story stages",
        ),
        (
            "UPDATE stories SET stage = NULL WHERE id = (SELECT MIN(id) FROM stories)",
            "stage was never set",
        ),
        (
            "UPDATE stories SET model_revision = 'rev-9' WHERE id = "
            "(SELECT MIN(id) FROM stories)",
            "different model identities",
        ),
    ],
    ids=["mixed", "null_stage", "inconsistent_model"],
)
def test_a_broken_generation_fails_the_partition(tmp_path, sql, fragment):
    repository = migrated(tmp_path)
    seed_stories(repository, count=4)
    _break_stage(repository, sql)

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "failed"
    assert fragment in outcome.error["error"]
    assert theme_rows(repository) is None
    assert run_rows(repository)[0]["status"] == "failed"


def test_an_encoder_that_did_not_produce_the_stories_fails_the_partition(tmp_path):
    """Two embedding spaces produce plausible numbers about nothing."""

    repository = migrated(tmp_path)
    seed_stories(repository)

    encoder = DifferentEncoder()
    outcome = themes(repository, encoder=encoder).run_partition(
        "NVDA", DAY, base_run_id="themes-1"
    )

    assert outcome.status == "failed"
    assert "embedding spaces" in outcome.error["error"]
    assert encoder.calls == []
    assert theme_rows(repository) is None
    assert run_rows(repository)[0]["status"] == "failed"


def test_classify_generation_names_every_case(tmp_path):
    """The guard's own vocabulary, exercised directly."""

    repository = migrated(tmp_path)
    empty = repository.story_generation("NVDA", DAY, "v1")
    assert classify_generation(empty) == GENERATION_EMPTY

    seed_stories(repository, count=3)
    healthy = repository.story_generation("NVDA", DAY, "v1")
    assert classify_generation(healthy) == GENERATION_SEMANTIC

    _break_stage(repository, "UPDATE stories SET stage = 'm2.exact'")
    exact = repository.story_generation("NVDA", DAY, "v1")
    assert classify_generation(exact) == GENERATION_EXACT

    _break_stage(
        repository,
        "UPDATE stories SET stage = 'm3.semantic' WHERE id = "
        "(SELECT MIN(id) FROM stories)",
    )
    mixed = repository.story_generation("NVDA", DAY, "v1")
    with pytest.raises(ThemeGenerationError, match="mixes story stages"):
        classify_generation(mixed)
    assert isinstance(ThemeGenerationError("x"), Phase0IntegrityError)


# ----------------------------------------------------------------------
# T21 -- T23, T33 -- reconstruction from persisted state
# ----------------------------------------------------------------------


def test_the_description_is_the_canonical_member_s(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository, count=3)
    generation = repository.story_generation("NVDA", DAY, "v1")

    for story in generation.stories:
        canonical = [
            member
            for member in story.members
            if member.raw_item_id == story.canonical_item_id
        ][0]
        assert story_description(story) == canonical.description
        assert theme_story(story).description == canonical.description


def _clustered(ticker: str, *groups) -> PersistedStory:
    """A persisted story assembled from real M2 cluster fingerprints.

    ``groups`` are ``(item_id, description)`` sequences, one per M2
    cluster, canonical cluster first -- the shape ``phase0.stories``
    persists for a semantically merged story.
    """

    members = []
    keys = []
    position = 0
    for group in groups:
        keys.append(cluster_fingerprint_for(ticker, [str(item) for item, _ in group]))
        for item, description in group:
            members.append(
                _member(raw_item_id=item, position=position, description=description)
            )
            position += 1
    return PersistedStory(
        story_id=1,
        cluster_fingerprint="fp",
        ticker=ticker,
        trading_day=DAY,
        pipeline_version="v1",
        stage="m3.semantic",
        canonical_title="Nvidia shares climb after strong results",
        canonical_item_id=groups[0][0][0],
        canonical_url=None,
        source="yahoo barrons",
        outlet="yahoo barrons",
        outlet_count=1,
        published_at=f"{DAY}T09:00:00+00:00",
        content_hash="ch",
        algorithm_version="m3.semantic.v1",
        config_fingerprint="cfg",
        model_name="fake-encoder",
        model_revision="rev-1",
        embedding_dimension=8,
        quarantined=False,
        semantic_skip_reason=None,
        member_story_keys=tuple(keys),
        members=tuple(members),
        provider_conflicts=(),
        semantic_merges=(),
    )


def test_a_merged_story_never_borrows_another_clusters_description():
    """P1-3: M3 embedded the canonical cluster's text and nothing else.

    Two M2 clusters merged semantically.  The canonical one carries no
    standfirst, so M3 encoded a title alone.  Reaching into the *other*
    cluster for a description would hand M5 a title-and-description pair
    no run ever encoded and call it a reconstruction.
    """

    story = _clustered(
        "NVDA",
        [(1, None)],  # canonical cluster: title only
        [(2, "Cluster B standfirst")],  # merged in, contributed no text
    )

    assert [member.raw_item_id for member in canonical_cluster_members(story)] == [1]
    assert story_description(story) is None
    assert theme_story(story).description is None


def test_a_merged_story_keeps_its_own_clusters_description():
    """The restriction is to the canonical cluster, not to the canonical row."""

    story = _clustered(
        "NVDA",
        [(1, None), (3, "Canonical cluster standfirst")],
        [(2, "Other cluster standfirst")],
    )

    members = canonical_cluster_members(story)
    assert [member.raw_item_id for member in members] == [1, 3]
    assert story_description(story) == "Canonical cluster standfirst"


def test_a_merged_story_prefers_its_canonical_item(tmp_path):
    story = _clustered(
        "NVDA",
        [(1, "Canonical item standfirst"), (3, "Same cluster, later position")],
        [(2, "Other cluster standfirst")],
    )

    assert story_description(story) == "Canonical item standfirst"


def test_an_unrecoverable_cluster_boundary_is_refused(tmp_path):
    """Fail closed.  There is no safe reading of broken provenance.

    Returning the canonical row alone would drop a standfirst that
    belonged to the canonical cluster; reading past it would borrow one
    that did not.  Both put text into M5 that no run ever encoded, so the
    reconstruction says it cannot and the partition fails.
    """

    story = _clustered(
        "NVDA",
        [(1, None), (3, "Canonical cluster standfirst")],
        [(2, "Other cluster standfirst")],
    )
    broken = dataclasses.replace(
        story, member_story_keys=("not-a-fingerprint", "nor-this-one")
    )

    with pytest.raises(CanonicalClusterUnrecoverable, match="no prefix"):
        canonical_cluster_members(broken)
    with pytest.raises(CanonicalClusterUnrecoverable):
        story_description(broken)
    with pytest.raises(CanonicalClusterUnrecoverable):
        theme_story(broken)


def test_a_canonical_description_does_not_excuse_broken_provenance():
    """Enough text being available is not the same as provenance holding.

    The canonical row here has its own standfirst, so a reconstruction
    could answer without ever looking at the boundary.  It still refuses:
    the rule is about whether the persisted claim can be believed.
    """

    story = _clustered(
        "NVDA",
        [(1, "Canonical item standfirst")],
        [(2, "Other cluster standfirst")],
    )
    broken = dataclasses.replace(story, member_story_keys=("bogus", "also-bogus"))

    with pytest.raises(CanonicalClusterUnrecoverable):
        story_description(broken)


def test_an_ambiguous_cluster_boundary_is_refused(monkeypatch):
    """Two provable boundaries is not better than none.

    Distinct member ids make two prefixes digest alike only by SHA-256
    collision, so the ambiguity is forced here rather than constructed:
    what is under test is that the guard exists and refuses, not that
    fingerprints collide.
    """

    story = _clustered("NVDA", [(1, None), (3, "a")], [(2, "b")])
    monkeypatch.setattr(
        "phase0.themes.cluster_fingerprint_for",
        lambda ticker, ids: story.member_story_keys[0],
    )

    with pytest.raises(CanonicalClusterUnrecoverable, match="ambiguous"):
        canonical_cluster_members(story)


def test_a_provable_boundary_is_used(tmp_path):
    """The valid case still recovers exactly one prefix."""

    story = _clustered(
        "NVDA",
        [(1, None), (3, "Canonical cluster standfirst")],
        [(2, "Other cluster standfirst")],
    )

    members = canonical_cluster_members(story)

    assert [member.raw_item_id for member in members] == [1, 3]
    assert story_description(story) == "Canonical cluster standfirst"


def test_a_single_cluster_story_uses_its_whole_member_list():
    """The unmerged case is unchanged: position order within one cluster."""

    story = _clustered("NVDA", [(30, None), (10, "Position one wins"), (20, "loses")])

    assert [member.raw_item_id for member in canonical_cluster_members(story)] == [
        30,
        10,
        20,
    ]
    assert story_description(story) == "Position one wins"
    # And it is not what raw_item_id ordering would have produced.
    by_id = next(
        member.description
        for member in sorted(story.members, key=lambda entry: entry.raw_item_id)
        if (member.description or "").strip()
    )
    assert by_id == "Position one wins"
    assert isinstance(DESCRIPTION_POLICY, str) and "canonical_m2_cluster" in (
        DESCRIPTION_POLICY
    )


def test_the_description_matches_what_m3_chose_for_a_real_story(tmp_path):
    """For a story M3 built from one cluster, the two choices agree.

    The reconstruction is checked against ``nlp.semdedup.bridge``'s own
    selection rather than against a restatement of it, so a change to
    either side shows up here.
    """

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.semdedup.bridge import stories_from_dedup

    repository = migrated(tmp_path)
    title = "NVDA reports quarterly results"
    # The earliest-published article becomes canonical and carries no
    # standfirst, so the fallback is what answers.
    evidence(
        repository,
        1,
        title=title,
        description=None,
        published_at=f"{DAY}T09:00:00+00:00",
    )
    evidence(
        repository,
        2,
        title=title,
        description="The only standfirst in this cluster",
        published_at=f"{DAY}T18:00:00+00:00",
    )

    StoryReconciler(
        repository, pipeline_version="v1", encoder=FakeEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")
    generation = repository.story_generation("NVDA", DAY, "v1")
    assert len(generation.stories) == 1
    story = generation.stories[0]
    assert len(story.members) == 2

    projected = repository.read.partition_evidence("NVDA", DAY)
    exact = deduplicate(projected.items, config=DedupConfig(supported_tickers=["NVDA"]))
    m3_inputs = stories_from_dedup(exact, projected.items)
    assert len(m3_inputs) == 1

    assert story_description(story) == m3_inputs[0].description
    assert story_description(story) == "The only standfirst in this cluster"


def test_m5_sees_the_persisted_generation_not_an_in_memory_result(tmp_path):
    """Edit the stored rows; what M5 embeds changes to match.

    The story stage's own in-memory result still holds the original
    headline, so a runner that reused it would embed that instead.
    """

    repository = migrated(tmp_path)
    seed_stories(repository, count=4)
    rewritten = "A REWRITTEN PERSISTED HEADLINE"
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET canonical_title = ? "
            "WHERE id = (SELECT MIN(id) FROM stories)",
            (rewritten,),
        )

    encoder = FakeEncoder()
    outcome = themes(repository, encoder=encoder).run_partition(
        "NVDA", DAY, base_run_id="themes-1"
    )

    assert outcome.status == "success"
    assert any(rewritten in text for text in encoder.texts)
    assert not any("NVDA topic" in text and rewritten in text for text in encoder.texts)


def test_trust_bearing_state_survives_the_reconstruction(tmp_path):
    """Quarantine, conflicts, skip reason, and merges all travel."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    story_id = repository.stories_for_day(DAY, "NVDA")[0]["id"]
    with repository.admin.connect_writable() as connection:
        # A real single-cluster key: reconstruction now proves the claim,
        # so a placeholder here would be testing the wrong refusal.
        member_ids = [
            str(row["raw_item_id"])
            for row in connection.execute(
                "SELECT raw_item_id FROM story_members WHERE story_id = ?",
                (story_id,),
            )
        ]
        connection.execute(
            "UPDATE stories SET quarantined = 1, "
            "semantic_skip_reason = 'provider_quarantine', "
            "member_story_keys = ? WHERE id = ?",
            (json.dumps([cluster_fingerprint_for("NVDA", member_ids)]), story_id),
        )
        connection.execute(
            "UPDATE story_members SET quarantined = 1 WHERE story_id = ?", (story_id,)
        )
        connection.execute(
            "INSERT INTO story_provider_conflicts "
            "(story_id, provider_namespace, provider_item_id, item_ids, fields) "
            "VALUES (?, 'yahoo barrons', 'prov-1', '[]', '[]')",
            (story_id,),
        )
        connection.execute(
            "INSERT INTO story_semantic_merges "
            "(story_id, left_story_key, right_story_key, similarity, reason) "
            "VALUES (?, 'm2-a', 'm2-b', 0.91, 'semantic_similarity')",
            (story_id,),
        )

    generation = repository.story_generation("NVDA", DAY, "v1")
    story = [entry for entry in generation.stories if entry.story_id == story_id][0]
    projected = theme_story(story)

    assert projected.semantic_skip_reason == "provider_quarantine"
    assert projected.is_quarantined is True
    assert projected.quarantined_member_ids == projected.item_ids
    assert projected.provider_conflicts == (("yahoo barrons", "prov-1"),)
    assert projected.merge_evidence == (("m2-a", "m2-b", 0.91, "semantic_similarity"),)
    assert len(projected.member_story_keys) == 1
    assert projected.content_hash == story.content_hash
    assert projected.published_at is not None
    assert projected.published_at.tzinfo is not None


# ----------------------------------------------------------------------
# T27 -- T29, T34, T35 -- sweeps, isolation, scoping, redaction
# ----------------------------------------------------------------------


def test_a_theme_only_partition_is_cleared(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    seed_theme_set(repository)
    with repository.admin.connect_writable() as connection:
        connection.execute("DELETE FROM theme_citations")
        connection.execute("DELETE FROM theme_stories")
        connection.execute("UPDATE stories SET canonical_item_id = NULL")
        connection.execute("DELETE FROM story_members")
        connection.execute("DELETE FROM stories")
    assert repository.read.theme_partitions(DAY, pipeline_version="v1") == ["NVDA"]

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "success"
    assert outcome.generation == GENERATION_EMPTY
    assert outcome.cleared is True
    assert repository.read.theme_partitions(DAY, pipeline_version="v1") == []


def test_two_tickers_are_settled_independently(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository, ticker="NVDA", count=3)
    seed_stories(repository, ticker="AMD", count=3)
    runner = themes(repository)

    runner.run_partition("NVDA", DAY, base_run_id="themes-1")
    runner.run_partition("AMD", DAY, base_run_id="themes-1")

    for ticker in ("NVDA", "AMD"):
        stored = theme_rows(repository, ticker)
        assert stored is not None
        assert stored["ticker"] == ticker
    assert len(run_rows(repository)) == 2


def test_a_failing_partition_does_not_raise_out_of_the_runner(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    _break_stage(repository, "UPDATE stories SET stage = NULL")

    runner = themes(repository)
    first = runner.run_partition("NVDA", DAY, base_run_id="themes-1")
    assert first.status == "failed"

    # The runner is still usable for the next partition.
    seed_stories(repository, ticker="AMD", count=3)
    second = runner.run_partition("AMD", DAY, base_run_id="themes-1")
    assert second.status == "success"


def test_returned_theme_diagnostics_are_redacted(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository)
    secret = "sk-live-7777-secret"
    message = f"upstream refused: Authorization: Bearer {secret}"

    outcome = themes(
        repository, encoder=RaisingEncoder(RuntimeError(message))
    ).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "failed"
    assert secret not in json.dumps(dict(outcome.error))
    assert "[REDACTED]" in outcome.error["error"]
    assert outcome.error["error"].startswith("RuntimeError: ")
    assert secret not in run_rows(repository)[0]["errors"]


def test_a_theme_partition_outcome_carries_no_run_state(tmp_path):
    repository = migrated(tmp_path)
    seed_stories(repository, count=3)

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert isinstance(outcome, ThemePartitionOutcome)
    assert dataclasses.is_dataclass(outcome)
    assert outcome.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "failed"
    from phase0.repository import StageRunContext

    reachable = [getattr(outcome, field.name) for field in dataclasses.fields(outcome)]
    assert not any(isinstance(value, StageRunContext) for value in reachable)
    assert not any(isinstance(value, Phase0Repository) for value in reachable)
    assert outcome.attempted is True


def test_a_day_below_the_clustering_floor_still_persists_its_coverage(tmp_path):
    """Themeless is not the same fact as storyless.

    Under M5's clustering floor every story is listed individually under
    Other Coverage.  The day has coverage and says so: a ``theme_sets``
    row exists with zero themes.  Clearing it instead would be the same
    durable state as a day with no stories at all, and those are different
    days.
    """

    repository = migrated(tmp_path)
    seed_stories(repository, count=2)

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "success"
    assert outcome.generation == GENERATION_SEMANTIC
    assert outcome.theme_count == 0
    stored = theme_rows(repository)
    assert stored is not None
    assert stored["method"] == "small_n_fallback"
    assert stored["themes"] == []
    assert len(stored["other_coverage"]) == 2
    assert {entry["reason"] for entry in stored["other_coverage"]} == {
        "below_clustering_floor"
    }


# ----------------------------------------------------------------------
# P1-2 -- a theme set may not be written over stories it never saw
# ----------------------------------------------------------------------


def test_themes_are_refused_when_the_story_generation_moved(tmp_path):
    """Clustering takes long enough for the stories to be replaced.

    Both generations are ``m3.semantic``, so a stage check sees nothing
    wrong.  The signature read with the stories is what notices, and it is
    recompared inside the writing transaction rather than before it.
    """

    repository = migrated(tmp_path)
    seed_stories(repository, count=4)
    before = repository.story_generation("NVDA", DAY, "v1")

    replaced: dict[str, object] = {}

    class ReplacesTheStoriesMidRun(FakeEncoder):
        """Commits a different story generation while M5 is running."""

        def embed_batch(self, texts):
            if not replaced:
                evidence(
                    repository,
                    99,
                    title="NVDA topic 0 variant 99",
                    description="Body 99",
                )
                StoryReconciler(
                    repository, pipeline_version="v1", encoder=FakeEncoder()
                ).run_partition("NVDA", DAY, base_run_id="stories-racer")
                replaced["signature"] = repository.story_generation(
                    "NVDA", DAY, "v1"
                ).signature
            return super().embed_batch(texts)

    outcome = themes(repository, encoder=ReplacesTheStoriesMidRun()).run_partition(
        "NVDA", DAY, base_run_id="themes-1"
    )

    assert replaced["signature"] != before.signature
    assert outcome.status == "failed"
    assert "no longer the partition's" in outcome.error["error"]
    # No stale theme set was written.
    assert theme_rows(repository) is None
    # The generation that won the race is untouched.
    after = repository.story_generation("NVDA", DAY, "v1")
    assert after.signature == replaced["signature"]
    assert len(after.stories) > len(before.stories)
    settled = [row for row in run_rows(repository) if "themes-1" in row["run_id"]][0]
    assert settled["status"] == "failed"


def test_an_unchanged_generation_still_commits(tmp_path):
    """The check refuses a moved generation, not an ordinary one."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=4)
    generation = repository.story_generation("NVDA", DAY, "v1")

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")

    assert outcome.status == "success"
    assert theme_rows(repository) is not None
    # Nothing about the stories moved, so the signature still describes them.
    assert repository.story_generation("NVDA", DAY, "v1").signature == (
        generation.signature
    )


def test_the_generation_signature_tracks_story_representation(tmp_path):
    """It moves exactly when a story change should invalidate themes.

    Reused from ``_stored_story_signature`` rather than invented, so a
    change that makes ``reconcile_stories`` drop the day's themes also
    moves this digest -- including changes that leave membership alone.
    """

    repository = migrated(tmp_path)
    seed_stories(repository, count=3)
    first = repository.story_generation("NVDA", DAY, "v1").signature

    assert repository.story_generation("NVDA", DAY, "v1").signature == first

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET outlet_count = outlet_count + 5 "
            "WHERE id = (SELECT MIN(id) FROM stories)"
        )
    assert repository.story_generation("NVDA", DAY, "v1").signature != first

    # A different partition has its own signature.
    assert repository.story_generation("AMD", DAY, "v1").signature != first


def test_the_signature_check_is_optional_for_other_callers(tmp_path):
    """Existing theme writers keep working without one."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    seed_theme_set(repository)

    assert theme_rows(repository) is not None


def test_a_stale_signature_is_refused_by_the_repository(tmp_path):
    """The refusal is the repository's, inside the writing transaction."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    stored = repository.stories_for_day(DAY, "NVDA")

    with pytest.raises(StoryGenerationConflict, match="no longer the partition"):
        with repository.stage_run(
            run_id="stale-1",
            stage=STAGE,
            trading_day=DAY,
            pipeline_version="v1",
            ticker="NVDA",
        ) as run:
            repository.reconcile_themes(
                run=run,
                ticker="NVDA",
                trading_day=DAY,
                pipeline_version="v1",
                theme_set=ThemeSetRecord(method="hdbscan"),
                themes=[
                    ThemeRecord(
                        fingerprint="T",
                        theme_key="T",
                        label="Theme",
                        story_ids=(stored[0]["id"],),
                        citation_item_ids=(json.loads(stored[0]["member_ids"])[0],),
                        method="hdbscan",
                        salience_rank=1,
                    )
                ],
                expected_story_signature="a-signature-from-another-moment",
                terminal=True,
            )

    assert theme_rows(repository) is None
    assert run_rows(repository)[0]["status"] == "failed"


# ----------------------------------------------------------------------
# P1-3 integration -- the merged-cluster reproduction, end to end
# ----------------------------------------------------------------------


class MergingEncoder(FakeEncoder):
    """Encodes everything near-identically so M3 merges compatible pairs."""

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            base = np.ones(8, dtype=float)
            base[0] += 0.0001 * (hashlib.sha256(text.encode()).digest()[0] / 255.0)
            vectors.append((base / np.linalg.norm(base)).tolist())
        return vectors


def test_a_real_semantic_merge_reproduces_m3s_canonical_input(tmp_path):
    """P1-3, through the real stages rather than a constructed record."""

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.semdedup import SemanticDedupConfig, merge_semantic_duplicates
    from nlp.semdedup.bridge import stories_from_dedup
    from nlp.semdedup.encoding import story_text

    repository = migrated(tmp_path)
    # Two distinct exact titles -> two M2 clusters; paraphrases that clear
    # M3's contradiction guards -> one semantic story.
    evidence(
        repository,
        1,
        title="Nvidia shares climb after strong results",
        description=None,
        published_at=f"{DAY}T08:00:00+00:00",
    )
    evidence(
        repository,
        2,
        title="Investors cheer a record chip quarter",
        description="Cluster B standfirst",
        published_at=f"{DAY}T10:00:00+00:00",
    )
    StoryReconciler(
        repository, pipeline_version="v1", encoder=MergingEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")

    generation = repository.story_generation("NVDA", DAY, "v1")
    assert len(generation.stories) == 1, "the two clusters did not merge"
    story = generation.stories[0]
    assert len(story.member_story_keys) == 2
    assert len(story.members) == 2

    # What M3 actually encoded for the canonical semantic input.
    projected = repository.read.partition_evidence("NVDA", DAY)
    exact = deduplicate(projected.items, config=DedupConfig(supported_tickers=["NVDA"]))
    inputs = {
        entry.story_key: entry for entry in stories_from_dedup(exact, projected.items)
    }
    semantic = merge_semantic_duplicates(
        list(inputs.values()),
        config=SemanticDedupConfig(supported_tickers=["NVDA"]),
        encoder=MergingEncoder(),
    )
    canonical_input = inputs[semantic.stories[0].canonical_story_key]
    assert canonical_input.description is None
    assert story_text(canonical_input) == "Nvidia shares climb after strong results"

    # The reconstruction must not attach the other cluster's standfirst.
    assert story_description(story) is None
    projected_story = theme_story(story)
    assert projected_story.description is None
    assert projected_story.title == canonical_input.title


def test_a_merged_story_with_a_canonical_description_reproduces_it(tmp_path):
    """The other half of the contract: real text is kept, not dropped."""

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.semdedup import SemanticDedupConfig, merge_semantic_duplicates
    from nlp.semdedup.bridge import stories_from_dedup

    repository = migrated(tmp_path)
    evidence(
        repository,
        1,
        title="Nvidia shares climb after strong results",
        description="Canonical cluster standfirst",
        published_at=f"{DAY}T08:00:00+00:00",
    )
    evidence(
        repository,
        2,
        title="Investors cheer a record chip quarter",
        description="Cluster B standfirst",
        published_at=f"{DAY}T10:00:00+00:00",
    )
    StoryReconciler(
        repository, pipeline_version="v1", encoder=MergingEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")

    generation = repository.story_generation("NVDA", DAY, "v1")
    story = generation.stories[0]
    assert len(story.member_story_keys) == 2

    projected = repository.read.partition_evidence("NVDA", DAY)
    exact = deduplicate(projected.items, config=DedupConfig(supported_tickers=["NVDA"]))
    inputs = {
        entry.story_key: entry for entry in stories_from_dedup(exact, projected.items)
    }
    semantic = merge_semantic_duplicates(
        list(inputs.values()),
        config=SemanticDedupConfig(supported_tickers=["NVDA"]),
        encoder=MergingEncoder(),
    )
    canonical_input = inputs[semantic.stories[0].canonical_story_key]

    assert story_description(story) == canonical_input.description
    assert story_description(story) == "Canonical cluster standfirst"


# ----------------------------------------------------------------------
# P2 -- committed work survives a later failure in the same attempt
# ----------------------------------------------------------------------


def test_a_failed_attempt_reports_the_clear_it_committed(tmp_path):
    """The clear is durable, so the returned outcome has to say so."""

    repository = migrated(tmp_path)
    seed_stories(repository)
    seed_theme_set(repository, model_name="an-older-encoder", theme_key="old-key")
    previous = themes(repository).capture_previous("NVDA", DAY)
    assert previous is not None

    outcome = themes(
        repository, encoder=RaisingEncoder(RuntimeError("M5 fell over"))
    ).run_partition("NVDA", DAY, base_run_id="themes-1", previous=previous)

    assert outcome.status == "failed"
    # One theme plus the theme_sets row.
    assert outcome.counts["cleared_rows"] == 2
    assert outcome.cleared is True
    assert outcome.counts["previous_theme_generation_rejected"] == 1
    assert repository.count("theme_sets") == 0

    # The durable ledger and the returned accounting agree.
    settled = [row for row in run_rows(repository) if "themes-1" in row["run_id"]][0]
    assert settled["status"] == "failed"
    assert json.loads(settled["counts"])["cleared_rows"] == 2
    # And the diagnostic is still redacted.
    assert outcome.error["error"].startswith("RuntimeError: ")


def test_a_real_merge_with_broken_provenance_fails_the_partition(tmp_path):
    """Codex's reproduction, through the production path.

    Two real M2 clusters merged into one M3 story.  The canonical
    cluster's canonical item carries no standfirst, so the text M3 encoded
    came from *another member of that same cluster*; a second cluster was
    merged in and contributed none.  With the cluster provenance broken,
    neither the canonical row nor the flattened member list can say which
    text that was -- so the stage refuses rather than persisting themes
    over a description M3 never saw.
    """

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.semdedup import SemanticDedupConfig, merge_semantic_duplicates
    from nlp.semdedup.bridge import stories_from_dedup

    repository = migrated(tmp_path)
    title = "Nvidia shares climb after strong results"
    # Canonical M2 cluster: two members, the earlier one (canonical) blank.
    evidence(
        repository,
        1,
        title=title,
        description=None,
        published_at=f"{DAY}T08:00:00+00:00",
    )
    evidence(
        repository,
        3,
        title=title,
        description="Canonical cluster standfirst",
        published_at=f"{DAY}T09:00:00+00:00",
    )
    # A second M2 cluster, merged in semantically, with its own standfirst.
    evidence(
        repository,
        2,
        title="Investors cheer a record chip quarter",
        description="Other cluster standfirst",
        published_at=f"{DAY}T10:00:00+00:00",
    )
    StoryReconciler(
        repository, pipeline_version="v1", encoder=MergingEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")

    generation = repository.story_generation("NVDA", DAY, "v1")
    assert len(generation.stories) == 1, "the two clusters did not merge"
    story = generation.stories[0]
    assert len(story.member_story_keys) == 2
    assert len(story.members) == 3

    # The text M3 really embedded came from inside the canonical cluster.
    projected = repository.read.partition_evidence("NVDA", DAY)
    exact = deduplicate(projected.items, config=DedupConfig(supported_tickers=["NVDA"]))
    inputs = {
        entry.story_key: entry for entry in stories_from_dedup(exact, projected.items)
    }
    semantic = merge_semantic_duplicates(
        list(inputs.values()),
        config=SemanticDedupConfig(supported_tickers=["NVDA"]),
        encoder=MergingEncoder(),
    )
    canonical_input = inputs[semantic.stories[0].canonical_story_key]
    assert canonical_input.description == "Canonical cluster standfirst"
    # Intact, the reconstruction reproduces it.
    assert story_description(story) == canonical_input.description

    # A healthy theme set exists and is compatible with the coming run.
    healthy = themes(repository, encoder=MergingEncoder()).run_partition(
        "NVDA", DAY, base_run_id="themes-healthy"
    )
    assert healthy.status == "success"
    before = theme_rows(repository)
    assert before is not None
    previous = themes(repository, encoder=MergingEncoder()).capture_previous(
        "NVDA", DAY
    )

    # Now break the cluster provenance -- the narrowest legitimate way to
    # make the boundary unprovable without touching anything else.
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET member_story_keys = ? WHERE id = ?",
            (json.dumps(["corrupted-key", "second-corrupted-key"]), story.story_id),
        )

    encoder = MergingEncoder()
    outcome = themes(repository, encoder=encoder).run_partition(
        "NVDA", DAY, base_run_id="themes-broken", previous=previous
    )

    assert outcome.status == "failed"
    assert outcome.degradation_reason is None
    assert "cannot be identified" in outcome.error["error"]
    assert outcome.error["error"].startswith("CanonicalClusterUnrecoverable: ")
    # M5 was never asked to embed a fabricated pair.
    assert encoder.calls == []
    # The compatible generation is preserved -- reconstruction failing is
    # not a reason to destroy a theme set that still describes the day.
    after = theme_rows(repository)
    assert after is not None
    assert [theme["theme_key"] for theme in after["themes"]] == [
        theme["theme_key"] for theme in before["themes"]
    ]
    assert outcome.cleared is False

    settled = [row for row in run_rows(repository) if "themes-broken" in row["run_id"]][
        0
    ]
    assert settled["status"] == "failed"
    assert markers(settled) == []


def test_an_incompatible_zero_theme_generation_reports_its_reason(tmp_path):
    """Observability: the refusal says which dimension moved.

    A refused generation with no themes has no identity rows to count, so
    the per-identity counters are all zero and only the generation-level
    ones can carry the reason.
    """

    expected = expectation()
    capture = evaluate_previous_themes(
        matching_generation(expected, model_name="an-older-encoder"), expected
    )

    assert capture.counts["previous_theme_generation_rejected"] == 1
    assert capture.counts["previous_theme_generation_rejected_model"] == 1
    assert capture.counts["previous_theme_generation_rejected_config"] == 0
    # No rows, so the identity counters stay silent rather than guessing.
    assert capture.counts["previous_themes_rejected"] == 0
    assert capture.counts["previous_themes_rejected_model"] == 0


def test_a_healthy_run_reports_why_it_could_not_reuse_the_old_generation(tmp_path):
    """Reported on the outcome, and still not a degradation."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=2)
    # Two stories sit below the clustering floor, so this stores a theme
    # set with no themes -- and an incompatible model.
    themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")
    with repository.admin.connect_writable() as connection:
        connection.execute("UPDATE theme_sets SET model_name = 'an-older-encoder'")
    previous = themes(repository).capture_previous("NVDA", DAY)
    assert previous is not None and previous.identities == ()

    outcome = themes(repository).run_partition(
        "NVDA", DAY, base_run_id="themes-2", previous=previous
    )

    assert outcome.status == "success"
    assert outcome.degradation_reason is None
    assert outcome.previous_generation_rejected == "model"
    assert outcome.counts["previous_theme_generation_rejected_model"] == 1
    assert outcome.cleared is True
    settled = [row for row in run_rows(repository) if "themes-2" in row["run_id"]][0]
    assert settled["status"] == "success"
    assert markers(settled) == []


# ----------------------------------------------------------------------
# Single-key provenance is a claim, and is proved like any other
# ----------------------------------------------------------------------


def _merged_partition(repository: Phase0Repository) -> tuple[int, str]:
    """Two real M2 clusters merged into one real M3 semantic story.

    The canonical cluster carries no standfirst and the other one does, so
    the text M3 embedded is title-only and any leak across the boundary is
    visible in the reconstruction.
    """

    evidence(
        repository,
        1,
        title="Nvidia shares climb after strong results",
        description=None,
        published_at=f"{DAY}T08:00:00+00:00",
    )
    evidence(
        repository,
        2,
        title="Investors cheer a record chip quarter",
        description="Other cluster only",
        published_at=f"{DAY}T10:00:00+00:00",
    )
    StoryReconciler(
        repository, pipeline_version="v1", encoder=MergingEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")

    generation = repository.story_generation("NVDA", DAY, "v1")
    assert len(generation.stories) == 1, "the two clusters did not merge"
    story = generation.stories[0]
    assert len(story.member_story_keys) == 2
    assert len(story.members) == 2
    # Intact, the reconstruction reproduces M3's title-only input.
    assert story_description(story) is None
    return story.story_id, story.member_story_keys[0]


def test_a_truncated_key_list_does_not_launder_a_merged_story(tmp_path):
    """The Codex reproduction, through the production path.

    One key does not prove one cluster.  Losing the second key leaves a
    record claiming a single M2 cluster while still holding both clusters'
    members -- and the surviving key does not describe them.  Accepting it
    would hand M5 the other cluster's standfirst and call it M3's input.
    """

    repository = migrated(tmp_path)
    story_id, canonical_key = _merged_partition(repository)

    # Corrupt only member_story_keys; every member row stays as it was.
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET member_story_keys = ? WHERE id = ?",
            (json.dumps([canonical_key]), story_id),
        )
    corrupted = repository.story_generation("NVDA", DAY, "v1").stories[0]
    assert corrupted.member_story_keys == (canonical_key,)
    assert len(corrupted.members) == 2

    encoder = MergingEncoder()
    outcome = themes(repository, encoder=encoder).run_partition(
        "NVDA", DAY, base_run_id="themes-broken"
    )

    assert outcome.status == "failed"
    assert outcome.error["error"].startswith("CanonicalClusterUnrecoverable: ")
    assert "describe different clusters" in outcome.error["error"]
    assert outcome.degradation_reason is None
    # M5 was never asked to embed the laundered pair.
    assert encoder.calls == []
    assert theme_rows(repository) is None

    settled = [row for row in run_rows(repository) if "themes-broken" in row["run_id"]][
        0
    ]
    assert settled["status"] == "failed"
    assert markers(settled) == []
    # Both the returned diagnostic and the durable one are redacted.
    assert "[REDACTED]" not in outcome.error["error"]
    assert "CanonicalClusterUnrecoverable" in settled["errors"]


def test_a_real_single_cluster_story_is_accepted(tmp_path):
    """The legitimate one-key case still reconstructs and clusters."""

    repository = migrated(tmp_path)
    seed_stories(repository, count=4)
    generation = repository.story_generation("NVDA", DAY, "v1")

    for story in generation.stories:
        assert len(story.member_story_keys) == 1
        members = canonical_cluster_members(story)
        # One cluster, so the whole member set is it -- proved, not assumed.
        assert members == tuple(sorted(story.members, key=lambda entry: entry.position))

    outcome = themes(repository).run_partition("NVDA", DAY, base_run_id="themes-1")
    assert outcome.status == "success"
    assert theme_rows(repository) is not None


def test_a_single_cluster_canonical_description_is_reproduced(tmp_path):
    """One key, valid fingerprint, canonical row carries the text."""

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.semdedup.bridge import stories_from_dedup

    repository = migrated(tmp_path)
    evidence(
        repository,
        1,
        title="Nvidia posts record quarterly revenue",
        description="Canonical item standfirst",
        published_at=f"{DAY}T08:00:00+00:00",
    )
    StoryReconciler(
        repository, pipeline_version="v1", encoder=FakeEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")

    story = repository.story_generation("NVDA", DAY, "v1").stories[0]
    assert len(story.member_story_keys) == 1

    projected = repository.read.partition_evidence("NVDA", DAY)
    exact = deduplicate(projected.items, config=DedupConfig(supported_tickers=["NVDA"]))
    m3_input = stories_from_dedup(exact, projected.items)[0]

    assert story_description(story) == m3_input.description
    assert story_description(story) == "Canonical item standfirst"


def test_a_single_cluster_fallback_reproduces_m3s_choice(tmp_path):
    """One key, canonical row blank, a same-cluster member supplies it."""

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.semdedup.bridge import stories_from_dedup

    repository = migrated(tmp_path)
    title = "Nvidia posts record quarterly revenue"
    evidence(
        repository,
        1,
        title=title,
        description=None,
        published_at=f"{DAY}T08:00:00+00:00",
    )
    evidence(
        repository,
        2,
        title=title,
        description="Same cluster standfirst",
        published_at=f"{DAY}T09:00:00+00:00",
    )
    StoryReconciler(
        repository, pipeline_version="v1", encoder=FakeEncoder()
    ).run_partition("NVDA", DAY, base_run_id="stories-1")

    story = repository.story_generation("NVDA", DAY, "v1").stories[0]
    assert len(story.member_story_keys) == 1
    assert len(story.members) == 2

    projected = repository.read.partition_evidence("NVDA", DAY)
    exact = deduplicate(projected.items, config=DedupConfig(supported_tickers=["NVDA"]))
    m3_input = stories_from_dedup(exact, projected.items)[0]

    assert story_description(story) == m3_input.description
    assert story_description(story) == "Same cluster standfirst"


def test_a_single_key_that_does_not_describe_its_members_is_refused():
    """The claim is checked even when only one cluster is claimed."""

    story = _clustered("NVDA", [(1, None), (3, "Same cluster standfirst")])
    assert canonical_cluster_members(story)  # intact, it is accepted

    broken = dataclasses.replace(story, member_story_keys=("not-a-fingerprint",))
    with pytest.raises(CanonicalClusterUnrecoverable, match="different clusters"):
        canonical_cluster_members(broken)
    with pytest.raises(CanonicalClusterUnrecoverable):
        story_description(broken)


def test_a_single_key_naming_a_subset_is_refused():
    """A key for part of the membership does not prove the whole of it."""

    story = _clustered("NVDA", [(1, None), (3, "Same cluster standfirst")])
    subset_key = cluster_fingerprint_for("NVDA", ["1"])
    broken = dataclasses.replace(story, member_story_keys=(subset_key,))

    with pytest.raises(CanonicalClusterUnrecoverable, match="different clusters"):
        canonical_cluster_members(broken)


def test_a_story_naming_no_cluster_at_all_is_refused():
    """M3 always names the cluster its story is called after."""

    story = _clustered("NVDA", [(1, "Canonical item standfirst")])
    broken = dataclasses.replace(story, member_story_keys=())

    with pytest.raises(CanonicalClusterUnrecoverable, match="names no M2 cluster"):
        canonical_cluster_members(broken)
    with pytest.raises(CanonicalClusterUnrecoverable):
        theme_story(broken)

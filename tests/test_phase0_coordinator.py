"""Cross-stage ordering: capture, then stories, then themes.

The coordinator owns one fact that neither stage can own for itself --
that theme identities must be read before story reconciliation can delete
them -- and these tests are about that fact and its consequences, not
about clustering.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from nlp.embeddings import EmbeddingModelLoadError
from phase0.coordinator import PartitionCoordinator, PartitionResult, summarize
from phase0.repository import Phase0Repository, StageRunContext
from phase0.stories import StoryReconciler
from phase0.themes import ThemeReconciler

DAY = "2026-07-23"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class FakeEncoder:
    model_name = "fake-encoder"
    model_revision = "rev-1"
    dimension = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

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


def migrated(tmp_path: Path) -> Phase0Repository:
    repository = Phase0Repository(tmp_path / "phase0.sqlite3")
    repository.migrate()
    return repository


def evidence(
    repository: Phase0Repository,
    index: int,
    *,
    ticker: str = "NVDA",
    title: str | None = None,
) -> int:
    link = f"https://publisher.example/{ticker.lower()}/{index}"
    return repository.admin.insert_raw_items(
        [
            {
                "source": "yahoo:Barron's",
                "ticker": ticker,
                "title": title or f"{ticker} topic {index % 2} variant {index}",
                "description": f"Body {index}",
                "url": link,
                "canonical_url": link,
                "published_at": f"{DAY}T{index % 24:02d}:00:00+00:00",
                "fetched_at": f"{DAY}T23:30:00+00:00",
                "ingest_status": "valid",
                "external_id": f"prov-{ticker}-{index}",
                "raw_json": json.dumps({"index": index}),
            }
        ]
    )[0].item_id


def seed(repository: Phase0Repository, *, ticker: str = "NVDA", count: int = 6):
    return [evidence(repository, index, ticker=ticker) for index in range(1, count + 1)]


def coordinator(repository: Phase0Repository, *, encoder=None, **overrides):
    return PartitionCoordinator(
        repository,
        pipeline_version="v1",
        encoder=encoder if encoder is not None else FakeEncoder(),
        **overrides,
    )


def rows(repository: Phase0Repository, stage: str) -> list[dict]:
    return [row for row in repository.read.run_log_rows() if row["stage"] == stage]


def theme_set(repository: Phase0Repository, ticker: str = "NVDA"):
    return repository.theme_set(ticker=ticker, trading_day=DAY, pipeline_version="v1")


# ----------------------------------------------------------------------
# The ordering itself
# ----------------------------------------------------------------------


def test_the_coordinator_runs_both_stages_in_order(tmp_path):
    repository = migrated(tmp_path)
    seed(repository)

    counts, errors = coordinator(repository).run(DAY, run_id="run-1")

    assert errors == []
    assert counts["partitions"] == 1
    assert counts["stories_succeeded"] == 1
    assert counts["themes_succeeded"] == 1
    assert counts["themes_persisted"] >= 1
    assert [row["stage"] for row in repository.read.run_log_rows()] == [
        "stories",
        "themes",
    ]
    assert theme_set(repository) is not None


def test_each_partition_runs_under_the_identity_it_is_given(tmp_path):
    """The day is the unit of execution; the run id is per partition."""

    repository = migrated(tmp_path)
    seed(repository, ticker="NVDA")
    seed(repository, ticker="AMD")

    counts, errors = coordinator(repository).run(
        DAY,
        run_id="inv:intelligence",
        identity=lambda ticker: (
            "inv:intelligence-retry" if ticker == "NVDA" else "inv:intelligence"
        ),
    )

    assert errors == []
    assert counts["partitions"] == 2
    by_ticker = {
        (row["ticker"], row["stage"]): row["run_id"]
        for row in repository.read.run_log_rows()
    }
    assert by_ticker[("NVDA", "stories")] == f"inv:intelligence-retry:NVDA:{DAY}"
    assert by_ticker[("NVDA", "themes")] == f"inv:intelligence-retry:NVDA:{DAY}"
    assert by_ticker[("AMD", "stories")] == f"inv:intelligence:AMD:{DAY}"
    assert by_ticker[("AMD", "themes")] == f"inv:intelligence:AMD:{DAY}"


def test_previous_identities_are_captured_before_stories_delete_them(tmp_path):
    """The whole reason this module exists.

    PR #101 made story reconciliation invalidate the theme set on *any*
    changed story representation, and invalidation there means deletion --
    ``themes`` has no ``invalidated_at``.  A theme runner that read its own
    continuity would find nothing on precisely the runs continuity is for.
    """

    repository = migrated(tmp_path)
    seed(repository)
    runner = coordinator(repository)
    runner.run_partition("NVDA", DAY, base_run_id="run-1")
    before = {theme["theme_key"] for theme in theme_set(repository)["themes"]}
    assert before

    # A new article changes story membership, so reconcile_stories will
    # delete the theme set before the themes stage opens.
    evidence(repository, 7, title="NVDA topic 1 variant 7")

    observed: dict[str, object] = {}
    real_capture = runner.themes.capture_previous
    real_stories = runner.stories.run_partition

    def watched_capture(ticker, trading_day):
        captured = real_capture(ticker, trading_day)
        observed["captured"] = tuple(entry.theme_key for entry in captured.identities)
        observed["generation_model"] = captured.model_name
        observed["theme_sets_at_capture"] = repository.count("theme_sets")
        return captured

    def watched_stories(ticker, trading_day, *, base_run_id):
        outcome = real_stories(ticker, trading_day, base_run_id=base_run_id)
        observed["theme_sets_after_stories"] = repository.count("theme_sets")
        return outcome

    runner.themes.capture_previous = watched_capture
    runner.stories.run_partition = watched_stories
    result = runner.run_partition("NVDA", DAY, base_run_id="run-2")

    assert observed["theme_sets_at_capture"] == 1
    # Story reconciliation destroyed it, exactly as PR #101 intends.
    assert observed["theme_sets_after_stories"] == 0
    assert set(observed["captured"]) == before
    assert observed["generation_model"] == "fake-encoder"
    assert result.previous_captured == len(before)
    # And the identities still reached M5.
    assert result.themes.counts["previous_themes_seen"] == len(before)


def test_captured_identities_preserve_theme_keys_across_invalidation(tmp_path):
    """T30: continuity survives PR #101's broad invalidation."""

    repository = migrated(tmp_path)
    seed(repository)
    runner = coordinator(repository)
    runner.run_partition("NVDA", DAY, base_run_id="run-1")
    before = {theme["theme_key"] for theme in theme_set(repository)["themes"]}

    evidence(repository, 7, title="NVDA topic 1 variant 7")
    result = runner.run_partition("NVDA", DAY, base_run_id="run-2")

    assert result.themes.status == "success"
    assert result.themes.counts["previous_themes_compatible"] == len(before)
    after = theme_set(repository)["themes"]
    carried = {
        theme["theme_key"]
        for theme in after
        if theme["matched_previous_key"] is not None
    }
    assert carried, "no identity was carried across the invalidation"
    assert carried <= before


# ----------------------------------------------------------------------
# Upstream story failure
# ----------------------------------------------------------------------


def test_a_failed_story_stage_leaves_themes_untouched(tmp_path):
    """T31: no themes run, no theme mutation, and it is reported."""

    repository = migrated(tmp_path)
    seed(repository)
    runner = coordinator(repository)
    runner.run_partition("NVDA", DAY, base_run_id="run-1")
    before = theme_set(repository)
    assert before is not None

    # Break the story stage only: a NULL stage is refused by the story
    # projection path long before anything is written.
    broken = coordinator(repository)
    broken.stories = StoryReconciler(
        repository,
        pipeline_version="v1",
        encoder=RaisingEncoder(RuntimeError("stories exploded")),
    )

    def always_fails(ticker, trading_day, *, base_run_id):
        from phase0.stories import PartitionOutcome

        with repository.stage_run(
            run_id=f"{base_run_id}:{ticker}:{trading_day}",
            stage="stories",
            ticker=ticker,
            trading_day=trading_day,
            pipeline_version="v1",
        ) as run:
            assert run is not None
            raise RuntimeError("stories exploded")
        return PartitionOutcome  # pragma: no cover

    def failing(ticker, trading_day, *, base_run_id):
        try:
            always_fails(ticker, trading_day, base_run_id=base_run_id)
        except RuntimeError as exc:
            from phase0.stories import PartitionOutcome

            return PartitionOutcome(
                ticker=ticker,
                trading_day=trading_day,
                status="failed",
                story_stage=None,
                degradation_reason=None,
                story_count=0,
                error={"type": "partition_error", "error": str(exc)},
            )

    broken.stories.run_partition = failing
    result = broken.run_partition("NVDA", DAY, base_run_id="run-2")

    assert result.stories.status == "failed"
    assert result.themes.status == "not_attempted"
    assert result.themes_attempted is False
    assert result.themes.error["reason"] == "upstream_story_failure"
    assert result.previous_captured == len(before["themes"])

    # No themes run was opened for run-2.
    assert [row["run_id"] for row in rows(repository, "themes")] == [
        "run-1:NVDA:2026-07-23"
    ]
    after = theme_set(repository)
    assert [theme["theme_key"] for theme in after["themes"]] == [
        theme["theme_key"] for theme in before["themes"]
    ]


def test_a_degraded_story_stage_still_runs_themes(tmp_path):
    """T32: M2-only stories still get a themes attempt, which degrades."""

    repository = migrated(tmp_path)
    seed(repository, count=3)
    runner = coordinator(repository)
    runner.run_partition("NVDA", DAY, base_run_id="run-1")
    assert theme_set(repository) is not None

    degraded = coordinator(repository)
    degraded.stories = StoryReconciler(
        repository,
        pipeline_version="v1",
        encoder=RaisingEncoder(EmbeddingModelLoadError("no model cache")),
    )
    result = degraded.run_partition("NVDA", DAY, base_run_id="run-2")

    assert result.stories.status == "degraded"
    assert result.themes.status == "degraded"
    assert result.themes.degradation_reason == "m5_requires_semantic_stories"
    assert theme_set(repository) is None
    assert {row["stage"] for row in repository.stories_for_day(DAY, "NVDA")} == {
        "m2.exact"
    }
    settled = [row for row in rows(repository, "themes") if "run-2" in row["run_id"]]
    assert [row["status"] for row in settled] == ["degraded"]


# ----------------------------------------------------------------------
# Enumeration and isolation
# ----------------------------------------------------------------------


def test_enumeration_unions_evidence_stories_and_themes(tmp_path):
    repository = migrated(tmp_path)
    seed(repository, ticker="NVDA", count=2)
    seed(repository, ticker="AMD", count=2)
    runner = coordinator(repository)
    runner.run(DAY, run_id="run-1")

    # AAPL gets evidence only; AMD keeps stories and themes; then NVDA's
    # evidence associations are withdrawn so only its stories remain.
    evidence(repository, 1, ticker="AAPL")
    with repository.admin.connect_writable() as connection:
        connection.execute("DELETE FROM raw_item_tickers WHERE ticker = 'NVDA'")

    assert "NVDA" not in repository.read.evidence_partition_tickers(DAY)
    assert runner.partitions(DAY) == ["AAPL", "AMD", "NVDA"]


def test_one_failing_partition_does_not_stop_the_others(tmp_path):
    repository = migrated(tmp_path)
    for ticker in ("AAPL", "AMD", "NVDA"):
        seed(repository, ticker=ticker, count=3)

    class FailsOneTicker(FakeEncoder):
        def embed_batch(self, texts):
            if any("AMD" in text for text in texts):
                raise RuntimeError("this partition only")
            return super().embed_batch(texts)

    counts, errors = coordinator(repository, encoder=FailsOneTicker()).run(
        DAY, run_id="run-1"
    )

    assert counts["partitions"] == 3
    assert counts["stories_failed"] == 1
    assert counts["themes_not_attempted"] == 1
    assert counts["stories_succeeded"] == 2
    assert counts["themes_succeeded"] == 2
    assert theme_set(repository, "AAPL") is not None
    assert theme_set(repository, "NVDA") is not None
    assert theme_set(repository, "AMD") is None
    assert [error["ticker"] for error in errors] == ["AMD"]


def test_a_stale_theme_only_partition_is_swept(tmp_path):
    """T27 through the coordinator: no stories left, themes cleared."""

    repository = migrated(tmp_path)
    seed(repository, count=2)
    runner = coordinator(repository)
    runner.run(DAY, run_id="run-1")
    assert theme_set(repository) is not None

    with repository.admin.connect_writable() as connection:
        connection.execute("DELETE FROM raw_item_tickers")
        connection.execute("DELETE FROM theme_citations")
        connection.execute("DELETE FROM theme_stories")
        connection.execute("DELETE FROM theme_other_coverage")
        connection.execute("DELETE FROM theme_excluded_stories")
        connection.execute("UPDATE stories SET canonical_item_id = NULL")
        connection.execute("DELETE FROM story_members")
        connection.execute("DELETE FROM stories")

    assert repository.read.theme_partitions(DAY, pipeline_version="v1") == ["NVDA"]
    counts, errors = runner.run(DAY, run_id="run-2")

    assert counts["partitions"] == 1
    assert counts["themes_succeeded"] == 1
    assert counts["theme_sets_cleared"] == 1
    assert errors == []
    assert theme_set(repository) is None


def test_multi_ticker_partitions_are_independent(tmp_path):
    repository = migrated(tmp_path)
    shared = evidence(repository, 1, ticker="NVDA")
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO raw_item_tickers "
            "(raw_item_id, ticker, association_type) VALUES (?, 'AMD', 'relevance')",
            (shared,),
        )

    counts, _ = coordinator(repository).run(DAY, run_id="run-1")

    assert counts["partitions"] == 2
    assert counts["stories_succeeded"] == 2
    assert counts["themes_succeeded"] == 2
    for ticker in ("NVDA", "AMD"):
        stored = repository.stories_for_day(DAY, ticker)
        assert len(stored) == 1
    assert len(rows(repository, "stories")) == 2
    assert len(rows(repository, "themes")) == 2


# ----------------------------------------------------------------------
# The result record
# ----------------------------------------------------------------------


def test_a_partition_result_is_frozen_and_carries_no_run_state(tmp_path):
    repository = migrated(tmp_path)
    seed(repository, count=2)

    result = coordinator(repository).run_partition("NVDA", DAY, base_run_id="run-1")

    assert isinstance(result, PartitionResult)
    assert result.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ticker = "AMD"
    reachable = [getattr(result, field.name) for field in dataclasses.fields(result)]
    assert not any(isinstance(value, StageRunContext) for value in reachable)
    assert not any(isinstance(value, Phase0Repository) for value in reachable)
    assert result.stories.__dataclass_params__.frozen
    assert result.themes.__dataclass_params__.frozen


def test_summarize_counts_both_stages_apart(tmp_path):
    repository = migrated(tmp_path)
    seed(repository, count=2)
    result = coordinator(repository).run_partition("NVDA", DAY, base_run_id="run-1")

    counts, errors = summarize([result])

    assert counts["partitions"] == 1
    assert counts["stories_succeeded"] == 1
    assert counts["themes_succeeded"] == 1
    assert counts["themes_not_attempted"] == 0
    assert errors == []


def test_the_coordinator_holds_no_stage_logic():
    """It orders the two runners; it does not reimplement them."""

    import inspect

    from phase0 import coordinator as module

    source = inspect.getsource(module)
    for forbidden in (
        "deduplicate(",
        "merge_semantic_duplicates(",
        "cluster_themes(",
        "reconcile_stories(",
        "reconcile_themes(",
        "clear_theme_set(",
    ):
        assert forbidden not in source, forbidden


def test_story_reconciler_knows_nothing_about_themes():
    """The dependency stays one-way."""

    import inspect

    from phase0 import stories as module

    # Prose may mention themes -- the module explains why it ships none.
    # What must not exist is a dependency or a hook.
    source = inspect.getsource(module)
    assert "from .themes import" not in source
    assert "from .coordinator import" not in source
    assert "phase0.themes" not in source
    assert "phase0.coordinator" not in source
    assert not hasattr(StoryReconciler, "capture_previous")
    parameters = inspect.signature(StoryReconciler.run_partition).parameters
    assert set(parameters) == {"self", "ticker", "trading_day", "base_run_id"}
    assert "previous" not in parameters
    public = {name for name in dir(StoryReconciler) if not name.startswith("_")}
    assert not {name for name in public if "callback" in name or "hook" in name}


def test_the_theme_runner_is_reusable_across_partitions(tmp_path):
    repository = migrated(tmp_path)
    seed(repository, ticker="NVDA", count=3)
    seed(repository, ticker="AMD", count=3)
    runner = ThemeReconciler(repository, pipeline_version="v1", encoder=FakeEncoder())
    stories = StoryReconciler(repository, pipeline_version="v1", encoder=FakeEncoder())
    for ticker in ("AMD", "NVDA"):
        stories.run_partition(ticker, DAY, base_run_id="stories-1")

    outcomes = [
        runner.run_partition(ticker, DAY, base_run_id="themes-1")
        for ticker in ("AMD", "NVDA")
    ]

    assert [outcome.status for outcome in outcomes] == ["success", "success"]
    assert {outcome.ticker for outcome in outcomes} == {"AMD", "NVDA"}


def test_the_pipeline_registers_through_the_coordinator_only():
    """The live pipeline drives this module; it does not reimplement it.

    The orchestrator's downstream registry names one builder, and that
    builder calls ``PartitionCoordinator``.  ``StoryReconciler`` and
    ``ThemeReconciler`` are never constructed from ``pipeline.py``, because
    doing so would put the capture-before-story-write ordering back in a
    second place where it could be gotten wrong.
    """

    import inspect

    import pipeline

    assert pipeline.DOWNSTREAM_STAGES == (pipeline.intelligence_stage,)
    source = inspect.getsource(pipeline)
    assert "PartitionCoordinator(" in source
    assert "StoryReconciler(" not in source
    assert "ThemeReconciler(" not in source
    assert "reconcile_stories(" not in source
    assert "reconcile_themes(" not in source


def test_a_zero_theme_generation_is_still_captured(tmp_path):
    """P1-1 through the coordinator: the set is the unit, not its themes.

    Two stories sit below M5's clustering floor, so the day stores a theme
    set with no themes in it.  A capture that read only ``themes`` would
    report nothing and the runner would never notice that what is stored
    was built by a model it can no longer reproduce.
    """

    repository = migrated(tmp_path)
    seed(repository, count=2)
    runner = coordinator(repository)
    runner.run_partition("NVDA", DAY, base_run_id="run-1")

    stored = theme_set(repository)
    assert stored is not None
    assert stored["themes"] == []

    captured = runner.themes.capture_previous("NVDA", DAY)
    assert captured is not None
    assert captured.identities == ()
    assert captured.model_name == "fake-encoder"
    assert captured.embedding_dimension == 8


def test_the_aggregate_preserves_a_committed_clear_after_a_failure(tmp_path):
    """P2 at the coordinator level: durable work is reported, not lost."""

    repository = migrated(tmp_path)
    seed(repository, count=6)
    runner = coordinator(repository)
    runner.run_partition("NVDA", DAY, base_run_id="run-1")
    assert theme_set(repository) is not None

    # A runtime whose provenance no longer matches the stored generation,
    # and whose encoder then fails after the clear has committed.
    failing = coordinator(repository)
    failing.themes = ThemeReconciler(
        repository,
        pipeline_version="v1",
        encoder=RaisingEncoder(RuntimeError("M5 fell over")),
    )

    class OtherModel(FakeEncoder):
        model_name = "an-older-encoder"

    # Rewrite the stored generation's provenance so the gate refuses it.
    with repository.admin.connect_writable() as connection:
        connection.execute("UPDATE theme_sets SET model_name = 'an-older-encoder'")
        connection.execute("UPDATE themes SET model_name = 'an-older-encoder'")

    counts, errors = failing.run(DAY, run_id="run-2")

    assert counts["themes_failed"] == 1
    assert counts["theme_rows_cleared"] >= 1
    assert counts["theme_sets_cleared"] == 1
    assert repository.count("theme_sets") == 0
    assert [error["type"] for error in errors] == ["theme_partition_error"]

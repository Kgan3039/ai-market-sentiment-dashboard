"""The ``stories`` stage: evidence in, one authoritative generation out.

Every test drives :class:`~phase0.stories.StoryReconciler` through the real
repository, the real M2 core, and the real M3 stage.  Only the encoder is a
fake, and it is deterministic: no test here loads a model, reaches the
network, or touches the sentence-transformers cache.

What is being pinned down is the *contract between the stages*, not the
clustering itself -- M2 and M3 have their own suites.  Specifically: which
generation is authoritative after a run, what a degradation looks like from
the outside, and what survives a failure.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from nlp.dedup import DedupConfig
from nlp.embeddings import (
    EmbeddingEncodingError,
    EmbeddingInputError,
    EmbeddingModelLoadError,
)
from nlp.semdedup import (
    SemanticDedupCapacityError,
    SemanticDedupConfig,
    SemanticDedupConfigError,
    SemanticDedupEncodingError,
    SemanticDedupInputError,
)
from phase0.models import ThemeRecord, ThemeSetRecord
from phase0.repository import STAGE_DEGRADED, Phase0Repository, StageRunContext
from phase0.stories import (
    DEGRADATION_REASON,
    STAGE,
    PartitionOutcome,
    StoryReconciler,
)

DAY = "2026-07-23"
OTHER_DAY = "2026-07-24"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class FakeEncoder:
    """A deterministic stand-in for :class:`nlp.embeddings.EmbeddingService`.

    Satisfies the ``StoryEncoder`` protocol exactly, counts its calls so a
    test can prove an empty partition never asks for a vector, and derives
    each vector from a digest of the text so identical headlines encode
    identically without any similarity being accidental.
    """

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
    """An encoder that fails the way a real one fails."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    def embed_batch(self, texts):
        super().embed_batch(texts)
        raise self._error


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
    source: str = "yahoo:Barron's",
    day: str = DAY,
    external_id: str | None = None,
    ingest_status: str = "valid",
    url: str | None = None,
    unowned: bool = False,
) -> int:
    """Persist one raw item and return its id.

    Written through ``admin`` on purpose: these tests are about what the
    story stage does with evidence, and routing every fixture through a
    Yahoo run would make the setup longer than the assertion.

    ``unowned`` stores the item with no primary ticker.  Ingestion writes a
    ``source`` association for whatever ``raw_items.ticker`` says, so an
    item seeded with a ticker is *already* authoritatively associated --
    which is the right default here, and useless for testing what happens
    to evidence nothing claims.
    """

    link = url or f"https://publisher.example/{ticker.lower()}/{index}"
    item = {
        "source": source,
        "ticker": None if unowned else ticker,
        "title": title if title is not None else f"{ticker} headline {index}",
        "description": f"Body {index}",
        "url": link,
        "canonical_url": link,
        "published_at": f"{day}T1{index % 10}:00:00+00:00",
        "fetched_at": f"{day}T23:00:00+00:00",
        "ingest_status": ingest_status,
        "external_id": external_id if external_id is not None else f"prov-{index}",
        "raw_json": json.dumps({"index": index}),
    }
    if ingest_status != "valid":
        item["validation_errors"] = ["seeded as non-valid evidence"]
    return repository.admin.insert_raw_items([item])[0].item_id


def associate(repository: Phase0Repository, item_id: int, ticker: str) -> None:
    """Give a raw item an accepted association -- the authoritative link."""

    with repository.admin.connect_writable() as connection:
        connection.execute(
            "INSERT OR IGNORE INTO raw_item_tickers "
            "(raw_item_id, ticker, association_type) VALUES (?, ?, 'relevance')",
            (item_id, ticker),
        )


def runner(
    repository: Phase0Repository,
    *,
    encoder=None,
    dedup_config: DedupConfig | None = None,
    semantic_config: SemanticDedupConfig | None = None,
) -> StoryReconciler:
    return StoryReconciler(
        repository,
        pipeline_version="v1",
        encoder=encoder if encoder is not None else FakeEncoder(),
        dedup_config=dedup_config,
        semantic_config=semantic_config,
    )


def stories(repository: Phase0Repository, ticker: str = "NVDA", **kwargs):
    return repository.stories_for_day(DAY, ticker, **kwargs)


def run_rows(repository: Phase0Repository) -> list[dict]:
    return [row for row in repository.read.run_log_rows() if row["stage"] == STAGE]


def degradation_markers(row) -> list[dict]:
    return [
        error
        for error in json.loads(row["errors"])
        if isinstance(error, dict) and error.get("type") == STAGE_DEGRADED
    ]


def seed_partition(repository: Phase0Repository, count: int = 3, ticker="NVDA"):
    """A partition of associated, valid, projectable evidence."""

    item_ids = []
    for index in range(1, count + 1):
        item_id = evidence(repository, index, ticker=ticker)
        associate(repository, item_id, ticker)
        item_ids.append(item_id)
    return item_ids


def seed_theme(repository: Phase0Repository, ticker: str = "NVDA") -> int:
    """Build a theme set over whatever stories the partition currently has."""

    persisted = stories(repository, ticker)
    with repository.stage_run(
        run_id=f"themes-{ticker}",
        stage="themes",
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
                method_reason="clustered",
                config_fingerprint="cfg",
                algorithm_version="m5.1",
                model_name="fake-encoder",
                model_revision="rev-1",
                embedding_dimension=8,
            ),
            themes=[
                ThemeRecord(
                    fingerprint="T",
                    theme_key="T",
                    label="Theme",
                    story_ids=(persisted[0]["id"],),
                    citation_item_ids=(json.loads(persisted[0]["member_ids"])[0],),
                    method="hdbscan",
                    salience_rank=1,
                )
            ],
            terminal=True,
        )
    return persisted[0]["id"]


# ----------------------------------------------------------------------
# T1 -- the healthy path
# ----------------------------------------------------------------------


def test_a_healthy_run_persists_the_semantic_generation(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository)

    counts, errors = runner(repository).run(DAY, run_id="run-1")

    persisted = stories(repository)
    assert len(persisted) == 3
    assert {row["stage"] for row in persisted} == {"m3.semantic"}
    for row in persisted:
        assert row["algorithm_version"] == "m3.semantic.v1"
        assert row["config_fingerprint"]
        assert row["model_name"] == "fake-encoder"
        assert row["model_revision"] == "rev-1"
        assert row["embedding_dimension"] == 8
        assert row["pipeline_version"] == "v1"

    ledger = run_rows(repository)
    assert [row["stage"] for row in ledger] == [STAGE]
    assert ledger[0]["status"] == "success"
    assert degradation_markers(ledger[0]) == []
    assert errors == []
    assert counts["partitions_succeeded"] == 1
    assert counts["stories_inserted"] == 3


# ----------------------------------------------------------------------
# T2 -- the four recoverable M3 failures
# ----------------------------------------------------------------------


RECOVERABLE = {
    "capacity": SemanticDedupCapacityError("NVDA", 999, 250),
    "semantic_encoding": SemanticDedupEncodingError("encoder returned junk"),
    "model_load": EmbeddingModelLoadError("no model cache on this host"),
    "model_encoding": EmbeddingEncodingError("failed to encode 3 text item(s)"),
}


@pytest.mark.parametrize("name", sorted(RECOVERABLE))
def test_a_recoverable_m3_failure_ships_the_exact_generation(tmp_path, name):
    """M2's answer, said out loud -- not M3's answer quietly downgraded."""

    repository = migrated(tmp_path)
    seed_partition(repository)

    counts, errors = runner(repository, encoder=RaisingEncoder(RECOVERABLE[name])).run(
        DAY, run_id="run-1"
    )

    persisted = stories(repository)
    assert len(persisted) == 3
    assert {row["stage"] for row in persisted} == {"m2.exact"}
    for row in persisted:
        assert row["algorithm_version"] == "m2.core.v1"
        assert row["config_fingerprint"]
        # M3 did not run, and a NULL says so where a plausible value would
        # not.
        assert row["model_name"] is None
        assert row["model_revision"] is None
        assert row["embedding_dimension"] is None
        assert json.loads(row["member_story_keys"]) == []

    ledger = run_rows(repository)
    assert ledger[0]["status"] == "degraded"
    assert degradation_markers(ledger[0]) == [
        {
            "type": STAGE_DEGRADED,
            "reason": DEGRADATION_REASON,
            "detail": f"{type(RECOVERABLE[name]).__name__}: {RECOVERABLE[name]}",
        }
    ]

    # Decision H: a degraded generation ships no themes at all.
    assert repository.count("theme_sets") == 0
    assert repository.count("themes") == 0
    assert counts["partitions_degraded"] == 1
    assert [error["reason"] for error in errors] == [DEGRADATION_REASON]


# ----------------------------------------------------------------------
# T3 / T4 -- what must never degrade
# ----------------------------------------------------------------------


NONRECOVERABLE = {
    "semantic_input": SemanticDedupInputError("stories[0] has a blank story_key"),
    "semantic_config": SemanticDedupConfigError("window_hours must be positive"),
    "embedding_input": EmbeddingInputError("texts[0] is invalid"),
    "unexpected": RuntimeError("something nobody predicted"),
}


@pytest.mark.parametrize("name", sorted(NONRECOVERABLE))
def test_a_nonrecoverable_m3_failure_fails_the_partition(tmp_path, name):
    """A defect in this stage is not a condition in the world.

    Falling back would ship M2's answer while burying the bug -- and the
    bug is in the code that built M3's input, which casts doubt on the M2
    half of the same run.
    """

    repository = migrated(tmp_path)
    seed_partition(repository)
    runner(repository).run(DAY, run_id="run-healthy")
    before = stories(repository, include_invalidated=True)

    counts, errors = runner(
        repository, encoder=RaisingEncoder(NONRECOVERABLE[name])
    ).run(DAY, run_id="run-broken")

    assert stories(repository, include_invalidated=True) == before
    failed = [
        row for row in run_rows(repository) if row["run_id"].startswith("run-broken")
    ]
    assert [row["status"] for row in failed] == ["failed"]
    assert degradation_markers(failed[0]) == []
    assert counts["partitions_failed"] == 1
    assert errors and errors[0]["type"] == "partition_error"


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            DedupConfig(supported_tickers=["NVDA"], max_partition_items=1),
            id="capacity",
        ),
        pytest.param(
            DedupConfig(supported_tickers=["AMD"]),
            id="input",
        ),
    ],
)
def test_an_m2_failure_ships_no_fallback_generation(tmp_path, config):
    """There is no M2 output to retain, so there is nothing to degrade to."""

    repository = migrated(tmp_path)
    seed_partition(repository, 3)
    runner(repository).run(DAY, run_id="run-healthy")
    before = stories(repository, include_invalidated=True)
    assert before

    counts, _ = runner(repository, dedup_config=config).run(DAY, run_id="run-broken")

    assert stories(repository, include_invalidated=True) == before
    failed = [
        row for row in run_rows(repository) if row["run_id"].startswith("run-broken")
    ]
    assert [row["status"] for row in failed] == ["failed"]
    assert counts["partitions_failed"] == 1


def test_the_recoverable_tuple_excludes_every_parent(tmp_path):
    """Catching ``SemanticDedupError`` would swallow three real defects."""

    from phase0.stories import _RECOVERABLE_M3

    for error in RECOVERABLE.values():
        assert isinstance(error, _RECOVERABLE_M3)
    for error in NONRECOVERABLE.values():
        assert not isinstance(error, _RECOVERABLE_M3)


# ----------------------------------------------------------------------
# T5 / T6 / T7 -- generation replacement
# ----------------------------------------------------------------------


def test_a_recovered_run_leaves_no_degraded_residue(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository)
    runner(repository, encoder=RaisingEncoder(RECOVERABLE["model_load"])).run(
        DAY, run_id="run-degraded"
    )
    assert {row["stage"] for row in stories(repository)} == {"m2.exact"}

    runner(repository).run(DAY, run_id="run-healthy")

    persisted = stories(repository, include_invalidated=True)
    assert {row["stage"] for row in persisted} == {"m3.semantic"}
    assert all(row["invalidated_at"] is None for row in persisted)


def test_a_degraded_replay_clears_the_healthy_generation_and_its_themes(tmp_path):
    """The direction that matters most: no mixed generation, no stale set."""

    repository = migrated(tmp_path)
    seed_partition(repository)
    runner(repository).run(DAY, run_id="run-healthy")
    seed_theme(repository)
    assert repository.count("theme_sets") == 1

    runner(repository, encoder=RaisingEncoder(RECOVERABLE["model_load"])).run(
        DAY, run_id="run-degraded"
    )

    persisted = stories(repository, include_invalidated=True)
    assert {row["stage"] for row in persisted} == {"m2.exact"}
    assert repository.count("themes") == 0
    assert repository.count("theme_stories") == 0
    assert repository.count("theme_sets") == 0


def test_an_idempotent_replay_rewrites_nothing(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository)
    runner(repository).run(DAY, run_id="run-1")
    before = stories(repository)

    counts, _ = runner(repository).run(DAY, run_id="run-2")

    assert stories(repository) == before
    assert counts["stories_unchanged"] == len(before)
    assert counts["stories_updated"] == 0
    assert counts["stories_inserted"] == 0

    replay = [row for row in run_rows(repository) if row["run_id"].startswith("run-2")]
    # Unchanged counts as `partial`, so the run resolves `degraded` -- with
    # no errors at all.  That is the fact a consumer must not read as an
    # outage, and the absence of the marker is what says so.
    assert json.loads(replay[0]["errors"]) == []
    assert degradation_markers(replay[0]) == []


# ----------------------------------------------------------------------
# T8 -- eligibility
# ----------------------------------------------------------------------


def test_only_eligible_evidence_reaches_the_dedup_core(tmp_path):
    repository = migrated(tmp_path)
    good = seed_partition(repository, 1)

    invalid = evidence(repository, 50, ingest_status="invalid")
    associate(repository, invalid, "NVDA")
    unprojectable = evidence(repository, 51, source="Example News")
    associate(repository, unprojectable, "NVDA")
    unassociated = evidence(repository, 52, unowned=True)

    outcome = runner(repository).run_partition("NVDA", DAY, base_run_id="run-1")

    members = {
        member
        for row in stories(repository)
        for member in json.loads(row["member_ids"])
    }
    assert members == set(good)
    assert invalid not in members
    assert unprojectable not in members
    assert unassociated not in members

    reasons = sorted(entry.outcome for entry in outcome.excluded)
    assert reasons == ["invalid", "unprojectable"]


# ----------------------------------------------------------------------
# T9 / T12 -- identity across partitions and providers
# ----------------------------------------------------------------------


def test_one_article_can_become_a_story_in_two_partitions(tmp_path):
    repository = migrated(tmp_path)
    shared = evidence(repository, 1, ticker="NVDA")
    associate(repository, shared, "NVDA")
    associate(repository, shared, "AMD")

    counts, _ = runner(repository).run(DAY, run_id="run-1")

    assert counts["partitions"] == 2
    for ticker in ("NVDA", "AMD"):
        persisted = stories(repository, ticker)
        assert len(persisted) == 1
        assert json.loads(persisted[0]["member_ids"]) == [shared]
    # Two partitions, two runs: neither speaks for the other.
    assert len(run_rows(repository)) == 2


def test_one_bare_provider_id_from_two_providers_stays_two_stories(tmp_path):
    """The scheme is what keeps the two provider id spaces apart."""

    repository = migrated(tmp_path)
    yahoo = evidence(
        repository,
        1,
        source="yahoo:Barron's",
        external_id="shared-id",
        url="https://publisher.example/from-yahoo",
        title="A completely unrelated Yahoo headline",
    )
    rss = evidence(
        repository,
        2,
        source="rss:example-news.com",
        external_id="shared-id",
        url="https://publisher.example/from-rss",
        title="An entirely different RSS headline about other matters",
    )
    for item_id in (yahoo, rss):
        associate(repository, item_id, "NVDA")

    runner(repository).run(DAY, run_id="run-1")

    persisted = stories(repository)
    assert len(persisted) == 2
    members = sorted(json.loads(row["member_ids"]) for row in persisted)
    assert members == [[yahoo], [rss]]


# ----------------------------------------------------------------------
# T10 / T11 -- sweeping and emptiness
# ----------------------------------------------------------------------


def test_a_partition_with_no_remaining_evidence_is_swept(tmp_path):
    """An unvisited partition would keep an authoritative-looking generation."""

    repository = migrated(tmp_path)
    item_ids = seed_partition(repository, 2)
    runner(repository).run(DAY, run_id="run-1")
    seed_theme(repository)
    assert stories(repository)

    # The association is withdrawn: the evidence is no longer NVDA's, so
    # NVDA returns no evidence partition at all.
    with repository.admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM raw_item_tickers WHERE raw_item_id IN (?, ?)", item_ids
        )
    assert repository.read.evidence_partitions(trading_day=DAY) == []
    assert repository.read.story_partitions(DAY, pipeline_version="v1") == ["NVDA"]

    counts, _ = runner(repository).run(DAY, run_id="run-2")

    assert counts["partitions"] == 1
    assert stories(repository, include_invalidated=True) == []
    assert repository.count("themes") == 0
    assert repository.count("theme_sets") == 0
    swept = [row for row in run_rows(repository) if row["run_id"].startswith("run-2")]
    assert [row["ticker"] for row in swept] == ["NVDA"]


def test_an_all_invalid_partition_clears_without_asking_for_a_vector(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository, 2)
    runner(repository).run(DAY, run_id="run-1")
    assert len(stories(repository)) == 2

    with repository.admin.connect_writable() as connection:
        connection.execute("UPDATE raw_items SET ingest_status = 'invalid'")

    encoder = FakeEncoder()
    counts, errors = runner(repository, encoder=encoder).run(DAY, run_id="run-2")

    assert stories(repository, include_invalidated=True) == []
    assert encoder.calls == []
    assert errors == []
    assert counts["partitions_succeeded"] == 1
    assert counts["stories_deleted"] == 2


# ----------------------------------------------------------------------
# T13 / T14 / T15 -- decision A, isolation, and the outcome record
# ----------------------------------------------------------------------


def test_the_stage_persists_no_embeddings(tmp_path):
    """Decision A: vectors are computed in memory and used, not stored."""

    repository = migrated(tmp_path)
    seed_partition(repository)

    runner(repository).run(DAY, run_id="run-1")

    assert repository.count("embeddings") == 0
    assert all(row["embedding"] is None for row in stories(repository))


def test_one_failed_partition_does_not_stop_the_others(tmp_path):
    repository = migrated(tmp_path)
    for ticker in ("AAPL", "AMD", "NVDA"):
        seed_partition(repository, 1, ticker)

    class FailsOneTicker(FakeEncoder):
        def embed_batch(self, texts):
            if any("AMD" in text for text in texts):
                raise RuntimeError("this partition only")
            return super().embed_batch(texts)

    counts, errors = runner(repository, encoder=FailsOneTicker()).run(
        DAY, run_id="run-1"
    )

    assert counts["partitions"] == 3
    assert counts["partitions_failed"] == 1
    assert counts["partitions_succeeded"] == 2
    # AAPL sorts before AMD and NVDA after it: the failure interrupted
    # neither its predecessor nor its successor.
    assert len(stories(repository, "AAPL")) == 1
    assert len(stories(repository, "NVDA")) == 1
    assert stories(repository, "AMD") == []

    ledger = {row["ticker"]: row["status"] for row in run_rows(repository)}
    assert ledger == {"AAPL": "success", "AMD": "failed", "NVDA": "success"}
    assert [error["ticker"] for error in errors] == ["AMD"]


def test_a_partition_outcome_carries_no_run_state(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository, 1)

    outcome = runner(repository).run_partition("NVDA", DAY, base_run_id="run-1")

    assert dataclasses.is_dataclass(outcome)
    assert outcome.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.status = "failed"

    reachable = [getattr(outcome, field.name) for field in dataclasses.fields(outcome)]
    assert not any(isinstance(value, StageRunContext) for value in reachable)
    assert not any(isinstance(value, Phase0Repository) for value in reachable)

    # The counts are this record's own; mutating them reaches nothing.
    assert outcome.counts["inserted"] == 1
    assert outcome.status == "success"
    assert outcome.story_stage == "m3.semantic"
    assert outcome.degradation_reason is None
    assert outcome.degraded is False


def test_a_degraded_outcome_names_its_reason(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository, 1)

    outcome = runner(
        repository, encoder=RaisingEncoder(RECOVERABLE["model_load"])
    ).run_partition("NVDA", DAY, base_run_id="run-1")

    assert isinstance(outcome, PartitionOutcome)
    assert outcome.status == "degraded"
    assert outcome.degraded is True
    assert outcome.degradation_reason == DEGRADATION_REASON
    assert outcome.story_stage == "m2.exact"


def test_a_failed_partition_outcome_reports_without_raising(tmp_path):
    repository = migrated(tmp_path)
    seed_partition(repository, 1)

    outcome = runner(
        repository, encoder=RaisingEncoder(RuntimeError("boom"))
    ).run_partition("NVDA", DAY, base_run_id="run-1")

    assert outcome.status == "failed"
    assert outcome.story_stage is None
    assert outcome.error["error"] == "RuntimeError: boom"
    assert stories(repository) == []


# ----------------------------------------------------------------------
# Projection failure belongs inside the partition's own run
# ----------------------------------------------------------------------


def test_a_projection_failure_is_isolated_to_its_own_partition(tmp_path, monkeypatch):
    """Discovery must not project, or one bad partition takes the day.

    Enumerating partitions through ``evidence_partitions`` ran
    ``classify_evidence`` over every associated row *before* the first
    ``stage_run`` opened.  A projection defect in one ticker therefore
    raised out of enumeration: no partition was attempted, and the ledger
    could not even say the day had been tried.  Discovery is now
    association-only, so the same defect fails exactly one run -- durably,
    where an operator can see whose it was.
    """

    repository = migrated(tmp_path)
    for ticker in ("AAPL", "AMD", "NVDA"):
        seed_partition(repository, 1, ticker)

    # Every ticker is discoverable without projecting anything.
    assert repository.read.evidence_partition_tickers(DAY) == ["AAPL", "AMD", "NVDA"]

    import phase0.repository as repository_module

    healthy = repository_module.classify_evidence

    def defective(row, ticker, trading_day):
        if ticker == "AMD":
            raise RuntimeError("projection defect reaching this partition only")
        return healthy(row, ticker, trading_day)

    monkeypatch.setattr(repository_module, "classify_evidence", defective)

    counts, errors = runner(repository).run(DAY, run_id="run-1")

    # Three attempts, each recorded, and only AMD's failed.
    ledger = {row["ticker"]: row["status"] for row in run_rows(repository)}
    assert ledger == {"AAPL": "success", "AMD": "failed", "NVDA": "success"}
    assert len(run_rows(repository)) == 3

    assert counts["partitions"] == 3
    assert counts["partitions_succeeded"] == 2
    assert counts["partitions_failed"] == 1
    assert len(stories(repository, "AAPL")) == 1
    assert len(stories(repository, "NVDA")) == 1
    assert stories(repository, "AMD") == []
    assert [error["ticker"] for error in errors] == ["AMD"]


def test_partition_discovery_projects_nothing(tmp_path, monkeypatch):
    """Stated as a rule, not inferred from one defect's blast radius."""

    repository = migrated(tmp_path)
    seed_partition(repository, 1)

    import phase0.repository as repository_module

    def refuse(*args, **kwargs):
        raise AssertionError("discovery must not classify or project evidence")

    monkeypatch.setattr(repository_module, "classify_evidence", refuse)

    assert runner(repository).partitions(DAY) == ["NVDA"]


# ----------------------------------------------------------------------
# Failure diagnostics are returned redacted, not only stored redacted
# ----------------------------------------------------------------------


#: Recognized by ``phase0.redaction`` -- an Authorization header value and
#: a bare ``api_key=`` assignment.
LEAKY_MESSAGE = (
    "upstream refused: Authorization: Bearer sk-live-4242-secret "
    "(api_key=sk-live-4242-secret)"
)
SECRET = "sk-live-4242-secret"


def test_a_failed_partition_outcome_is_redacted(tmp_path):
    """The outcome is *returned*, so it leaves redacted or it leaks."""

    repository = migrated(tmp_path)
    seed_partition(repository, 1)

    outcome = runner(
        repository, encoder=RaisingEncoder(RuntimeError(LEAKY_MESSAGE))
    ).run_partition("NVDA", DAY, base_run_id="run-1")

    assert outcome.status == "failed"
    assert SECRET not in json.dumps(dict(outcome.error))
    assert "[REDACTED]" in outcome.error["error"]
    # The useful half survives: a class name is not where a secret lives.
    assert outcome.error["error"].startswith("RuntimeError: ")


def test_the_aggregate_report_is_redacted_too(tmp_path):
    """Redacting at the source means every copy downstream is safe."""

    repository = migrated(tmp_path)
    seed_partition(repository, 1)

    counts, errors = runner(
        repository, encoder=RaisingEncoder(RuntimeError(LEAKY_MESSAGE))
    ).run(DAY, run_id="run-1")

    assert counts["partitions_failed"] == 1
    assert SECRET not in json.dumps(errors)
    assert "[REDACTED]" in errors[0]["error"]

    # The durable row was already safe and stays so; the two now agree
    # rather than the returned copy being the loose one.
    ledger = run_rows(repository)
    assert SECRET not in ledger[0]["errors"]

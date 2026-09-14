"""A4a (issue #74): G1 review sampling from persisted themes, and the scorecard.

The population tests drive the real repository: raw items, a story
generation, and a theme set written through the logged reconciliation
entrypoints, then read back by the sampler through the reader's snapshot.
Nothing here clusters; the theme sets are stated, so what the sampler must
reproduce is known exactly.

The scoring tests pin the four facts apart -- ``threshold_met``,
``review_complete``, ``gate_eligible``, ``gate_result`` -- and hold the
tool to its fail-closed rules: nothing serialized is believed about
origin, ratification, reviewers, adjudication, population, or thresholds.
The only way to reach ``PASS`` in this file is to patch the origin
classifier and the protocol registry in code, which is the point.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

from nlp.eval import review
from nlp.eval.review import (
    ADJUDICATION_FIELDNAMES,
    ASSIGNMENT_FIELDNAMES,
    CONTEXT_FIELDS,
    IDENTITY_FIELDS,
    RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS,
    RELEASE_G1_THRESHOLD,
    SKIP_M2_ONLY,
    SKIP_NO_STORY_OUTPUT,
    SKIP_SET_INCONSISTENT,
    SKIP_STALE_SET_DEGRADED,
    SKIP_STALE_SET_NO_STORIES,
    SKIP_THEMES_NOT_GENERATED,
    SNAPSHOT_FIELDS,
    UNRATIFIED_PROTOCOL,
    AdjudicationState,
    DevelopmentOverrides,
    GateResult,
    OperatorAttestation,
    OriginStatus,
    Protocol,
    ReviewSamplingError,
    RoundResult,
    build_manifest,
    derive_gate_result,
    load_fixture_population,
    load_persisted_population,
    manifest_path_for,
    read_manifest,
    render_scorecard,
    resolve_protocol,
    row_id_for,
    sample_assignments,
    score_gate,
    score_round,
    write_sample,
)
from nlp.eval.trust import DatasetKind, LabelingStatus, MetricsPurpose
from phase0.models import (
    ExcludedStoryRecord,
    OtherCoverageRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import Phase0Repository
from phase0.stories import STAGE as STORIES_STAGE
from phase0.themes import STAGE as THEMES_STAGE
from phase0.yahoo import STAGE as YAHOO_FETCH_STAGE
from tools import make_review_sheets

DAY = "2026-07-23"
VERSION = "v1"
_RUN_IDS = itertools.count(1)
_ITEM_INDEX = itertools.count(1)


# ----------------------------------------------------------------------
# Building a persisted day
# ----------------------------------------------------------------------


def migrated(tmp_path: Path, name: str = "phase0.db") -> Phase0Repository:
    repository = Phase0Repository(tmp_path / name)
    repository.migrate()
    return repository


def raw_items(
    repository: Phase0Repository,
    ticker: str,
    count: int,
    day: str = DAY,
    *,
    description=None,
    title=None,
):
    rows = []
    for _ in range(count):
        index = next(_ITEM_INDEX)
        rows.append(
            {
                "source": f"yahoo:Outlet {index % 3}",
                "ticker": ticker,
                "title": title or f"{ticker} headline {index}",
                "description": description or f"Standfirst for {ticker} item {index}",
                "url": f"https://publisher.example/{ticker.lower()}/{index}",
                "canonical_url": f"https://publisher.example/{ticker.lower()}/{index}",
                "published_at": f"{day}T{index % 24:02d}:00:00+00:00",
                "fetched_at": f"{day}T23:30:00+00:00",
                "raw_json": {"index": index},
            }
        )
    return [r.item_id for r in repository.admin.insert_raw_items(rows)]


def story(fingerprint: str, item_ids, *, stage: str = "m3.semantic", title=None):
    return StoryRecord(
        cluster_fingerprint=fingerprint,
        canonical_title=title or f"Story {fingerprint}",
        members=tuple(
            StoryMemberRecord(
                raw_item_id=item_id,
                position=position,
                outlet=f"outlet-{item_id % 3}",
                url=f"https://publisher.example/item/{item_id}",
                canonical_url=f"https://publisher.example/item/{item_id}",
            )
            for position, item_id in enumerate(item_ids)
        ),
        canonical_item_id=item_ids[0],
        canonical_url=f"https://publisher.example/item/{item_ids[0]}",
        outlet_count=len(item_ids),
        content_hash=f"hash-{fingerprint}",
        stage=stage,
        algorithm_version="m3.1",
        config_fingerprint="cfg",
    )


def run_stage(repository, stage, ticker, day=DAY, version=VERSION):
    return repository.stage_run(
        run_id=f"run-{next(_RUN_IDS)}",
        stage=stage,
        trading_day=day,
        pipeline_version=version,
        ticker=ticker,
    )


def persist_stories(
    repository,
    ticker,
    count,
    *,
    day=DAY,
    stage="m3.semantic",
    version=VERSION,
    description=None,
    title=None,
) -> list[int]:
    items = raw_items(
        repository, ticker, count, day, description=description, title=title
    )
    with run_stage(repository, STORIES_STAGE, ticker, day, version) as run:
        repository.reconcile_stories(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=version,
            stories=[
                story(f"{ticker}-{day}-{n}", [item], stage=stage, title=title)
                for n, item in enumerate(items, start=1)
            ],
        )
    return [row["id"] for row in repository.stories_for_day(day, ticker)]


def theme_record(key: str, label: str, story_ids, rank: int) -> ThemeRecord:
    return ThemeRecord(
        fingerprint=f"fp-{key}",
        theme_key=key,
        label=label,
        label_source="representative_title",
        story_ids=tuple(story_ids),
        status="ready",
        salience_rank=rank,
        story_count=len(story_ids),
        method="hdbscan",
        content_hash=f"fp-{key}",
        algorithm_version="m5.1",
        config_fingerprint="cfg-m5",
        model_name="fake-encoder",
        model_revision="r1",
        embedding_dimension=8,
    )


def persist_theme_set(
    repository,
    ticker,
    *,
    themes: dict[str, list[int]],
    other: list[int] = (),
    excluded: list[int] = (),
    day=DAY,
    version=VERSION,
    label_prefix="Theme",
) -> None:
    with run_stage(repository, THEMES_STAGE, ticker, day, version) as run:
        repository.reconcile_themes(
            run=run,
            ticker=ticker,
            trading_day=day,
            pipeline_version=version,
            theme_set=ThemeSetRecord(
                method="hdbscan",
                method_reason="clustered",
                quality={"theme_count": len(themes)},
                config_fingerprint="cfg-m5",
                algorithm_version="m5.1",
                model_name="fake-encoder",
                model_revision="r1",
                embedding_dimension=8,
            ),
            themes=[
                theme_record(key, f"{label_prefix} {key}", ids, rank)
                for rank, (key, ids) in enumerate(themes.items(), start=1)
            ],
            other_coverage=[
                OtherCoverageRecord(story_id=s, reason="clustering_noise", position=p)
                for p, s in enumerate(other)
            ],
            excluded=[
                ExcludedStoryRecord(story_id=s, reason="no_encodable_text")
                for s in excluded
            ],
            terminal=True,
        )


def ingestion_run(repository, ticker, day=DAY):
    """A logged fetch run, so the ledger says evidence was fetched."""

    with run_stage(repository, YAHOO_FETCH_STAGE, ticker, day):
        pass


def seed_wide(repository) -> None:
    for ticker in ("NVDA", "AMD"):
        ids = persist_stories(repository, ticker, 50)
        persist_theme_set(
            repository,
            ticker,
            themes={f"{ticker}-a": ids[0:20], f"{ticker}-b": ids[20:40]},
            other=ids[40:48],
            excluded=ids[48:50],
        )
        ingestion_run(repository, ticker)


@pytest.fixture
def day_db(tmp_path):
    """NVDA: 2 themes (3+2 stories), 2 other coverage, 1 excluded = 8 rows."""

    repository = migrated(tmp_path)
    ids = persist_stories(repository, "NVDA", 8)
    persist_theme_set(
        repository,
        "NVDA",
        themes={"t-alpha": ids[0:3], "t-beta": ids[3:5]},
        other=ids[5:7],
        excluded=ids[7:8],
    )
    ingestion_run(repository, "NVDA")
    return repository


@pytest.fixture
def wide_db(tmp_path):
    """Two tickers, 100 placements, for multi-round tests."""

    repository = migrated(tmp_path, "wide.db")
    seed_wide(repository)
    return repository


def population_of(repository, **kwargs):
    kwargs.setdefault("trading_days", [DAY])
    return load_persisted_population(repository.database_path, **kwargs)


# ----------------------------------------------------------------------
# Sheets and rounds
# ----------------------------------------------------------------------


def write_round(population, directory, *, seed="s", size=40, name="r1", priors=()):
    sample = sample_assignments(
        population, seed=seed, size=size, round_id=name, prior_manifests=priors
    )
    manifest = build_manifest(
        sample, csv_name=f"{name}.csv", code={"commit": "x", "dirty": False}
    )
    return write_sample(sample, directory / f"{name}.csv", manifest=manifest)


def fill(csv_path: Path, out: Path, verdict, reviewer="alice") -> Path:
    """Complete a blank sheet; ``verdict`` is a value or a function of the row."""

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        value = verdict(row) if callable(verdict) else verdict
        row["reviewer_verdict"] = value
        row["reviewer_id"] = reviewer if value else ""
        row["reviewed_at"] = "2026-09-12T10:00:00+00:00" if value else ""
    write_rows(out, rows)
    return out


def write_rows(out: Path, rows) -> None:
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ASSIGNMENT_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def adjudicate(out: Path, finals: dict[str, str], adjudicator="carol") -> Path:
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ADJUDICATION_FIELDNAMES, lineterminator="\n"
        )
        writer.writeheader()
        for row_id, verdict in finals.items():
            writer.writerow(
                {
                    "row_id": row_id,
                    "final_verdict": verdict,
                    "adjudicator_id": adjudicator,
                    "adjudicated_at": "2026-09-12T11:00:00+00:00",
                    "adjudication_notes": "",
                }
            )
    return out


def two_reviewer_round(
    csv_path, manifest_path, directory, *, bob=None, adjudicated=None
):
    manifest = read_manifest(manifest_path)
    name = manifest["sample"]["round_id"]
    a = fill(csv_path, directory / f"{name}.a.csv", "correct", "alice")
    b = fill(csv_path, directory / f"{name}.b.csv", bob or "correct", "bob")
    return score_round(manifest, [a, b], adjudicated=adjudicated)


def two_linked_rounds(population, directory, **kwargs):
    """Two 40-row rounds, the second drawn excluding the first."""

    csv1, m1 = write_round(population, directory, seed="one", name="r1")
    csv2, m2 = write_round(
        population, directory, seed="two", name="r2", priors=[read_manifest(m1)]
    )
    return [
        two_reviewer_round(csv1, m1, directory, **kwargs),
        two_reviewer_round(csv2, m2, directory, **kwargs),
    ]


TEST_PROTOCOL = Protocol(
    id="k3-test",
    positive_verdict="correct",
    negative_verdict="incorrect",
    adjudicated_states=frozenset(
        {AdjudicationState.UNANIMOUS, AdjudicationState.RESOLVED}
    ),
)


@pytest.fixture
def eligible_world(monkeypatch):
    """The only route to PASS: ratified protocol, verified origin, and a verified
    theme-set build binding -- all three patched in code, none reachable from
    an artifact."""

    monkeypatch.setattr(review, "RATIFIED_PROTOCOLS", {TEST_PROTOCOL.id: TEST_PROTOCOL})
    monkeypatch.setattr(
        review,
        "classify_origin",
        lambda source: (OriginStatus.VERIFIED_LIVE, "patched for the test"),
    )
    monkeypatch.setattr(
        review,
        "classify_generation_binding",
        lambda population: (
            review.GENERATION_BINDING_VERIFIED,
            population.stories.signature,
        ),
    )


def reauthor(manifest_path: Path, csv_path: Path, mutate) -> None:
    """Rewrite a manifest as an attacker with full knowledge would: apply
    ``mutate``, recompute every digest the reader checks, and recut the blank
    sheet with the new binding.  Self-consistent substitution *before* review
    is outside the threat model; what must hold is that nothing re-authored
    can gain eligibility, and that sheets completed against the original
    cannot be handed in against the re-authored one."""

    payload = json.loads(manifest_path.read_text())
    mutate(payload)
    snapshot = payload["snapshot"]
    snapshot["sha256"] = review.snapshot_digest(
        snapshot["rows"], snapshot["theme_sets"]
    )
    selection = payload["selection"]
    selection["theme_set_ids"] = [s["theme_set_id"] for s in snapshot["theme_sets"]]
    selection["digest"] = review.selection_digest(
        trading_days=selection["trading_days"],
        tickers=selection["tickers"],
        pipeline_versions=selection["pipeline_versions"],
        theme_set_ids=selection["theme_set_ids"],
        source_mode=payload["source"].get("mode"),
        theme_sets=snapshot["theme_sets"],
        skipped_partitions=selection["skipped_partitions"],
    )
    payload["binding"] = {
        "manifest_id": review.manifest_identity(payload),
        "snapshot_sha256": snapshot["sha256"],
    }
    rows = read_rows(csv_path)
    for row in rows:
        row.update(payload["binding"])
    write_rows(csv_path, rows)
    payload["sheet"]["blank_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ratified_manifest(population, directory, name, seed="s", priors=()):
    sample = sample_assignments(
        population, seed=seed, size=40, round_id=name, prior_manifests=priors
    )
    manifest = build_manifest(
        sample,
        protocol_id=TEST_PROTOCOL.id,
        csv_name=f"{name}.csv",
        code={"commit": "x", "dirty": False},
    )
    return write_sample(sample, directory / f"{name}.csv", manifest=manifest)


# ----------------------------------------------------------------------
# The persisted population, exactly once
# ----------------------------------------------------------------------


def test_population_is_every_placement_of_the_persisted_theme_set(day_db):
    population = population_of(day_db)
    assert population.by_assignment_type == {
        "theme": 5,
        "other_coverage": 2,
        "excluded": 1,
    }
    assert len({row.story_id for row in population.rows}) == 8
    assert len({row.row_id for row in population.rows}) == 8
    assert population.skipped == ()
    [theme_set] = population.theme_sets
    assert (
        theme_set.theme_count,
        theme_set.other_coverage_count,
        theme_set.excluded_count,
    ) == (
        2,
        2,
        1,
    )
    assert theme_set.config_fingerprint == "cfg-m5"
    assert theme_set.model_name == "fake-encoder"
    assert len(theme_set.story_generation_signature_current) == 64
    assert theme_set.theme_build_story_generation_signature is None
    assert theme_set.generation_binding == "unverified"


def test_population_reads_what_was_stored_not_a_reclustering(day_db):
    population = population_of(day_db)
    alpha = [r for r in population.rows if r.theme_key == "t-alpha"]
    assert len(alpha) == 3
    assert {r.theme_label for r in alpha} == {"Theme t-alpha"}
    assert {r.theme_story_count for r in alpha} == {"3"}
    assert {r.theme_label_source for r in alpha} == {"representative_title"}
    other = [r for r in population.rows if r.assignment_type == "other_coverage"]
    assert {r.placement_reason for r in other} == {"clustering_noise"}
    excluded = [r for r in population.rows if r.assignment_type == "excluded"]
    assert {r.placement_reason for r in excluded} == {"no_encodable_text"}


def test_population_reflects_only_the_named_pipeline_version(day_db):
    amd = persist_stories(day_db, "AMD", 3, version="v2")
    persist_theme_set(day_db, "AMD", themes={"amd-a": amd}, version="v2")
    with pytest.raises(ReviewSamplingError, match="span pipeline versions"):
        population_of(day_db)
    assert population_of(day_db, pipeline_version="v1").tickers == ("NVDA",)
    assert population_of(day_db, pipeline_version="v2").tickers == ("AMD",)


def test_an_empty_database_refuses_rather_than_using_the_fixture(tmp_path):
    repository = migrated(tmp_path)
    with pytest.raises(ReviewSamplingError, match="no fixture is substituted"):
        population_of(repository)


def test_the_sampler_does_not_create_a_database_that_is_not_there(tmp_path):
    missing = tmp_path / "typo.db"
    with pytest.raises(ReviewSamplingError, match="no Phase 0 database"):
        load_persisted_population(missing, trading_days=[DAY])
    assert not missing.exists()


# ----------------------------------------------------------------------
# Population enumeration regressions (Codex 11, 12)
# ----------------------------------------------------------------------


def test_a_theme_only_partition_is_enumerated_and_named_stale(day_db):
    """Codex 11: a theme set whose stories were invalidated must not vanish."""

    with day_db.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET invalidated_at = ? WHERE ticker = 'NVDA'",
            (f"{DAY}T23:59:00+00:00",),
        )
    population = population_of(day_db)
    assert population.rows == ()
    [skipped] = population.skipped
    assert (skipped.ticker, skipped.reason) == ("NVDA", SKIP_STALE_SET_NO_STORIES)
    assert skipped.reason != SKIP_NO_STORY_OUTPUT


def test_a_theme_set_over_an_m2_only_generation_is_not_a_review_population(tmp_path):
    """Codex 12: a persisted set over degraded stories is stale, not healthy M5."""

    repository = migrated(tmp_path)
    ids = persist_stories(repository, "NVDA", 4, stage="m2.exact")
    persist_theme_set(repository, "NVDA", themes={"t": ids[:3]}, other=ids[3:])
    population = population_of(repository)
    assert population.rows == ()
    [skipped] = population.skipped
    assert skipped.reason == SKIP_STALE_SET_DEGRADED
    assert "m2.exact" in skipped.detail


def test_an_m2_only_partition_without_a_set_is_recorded_as_skipped(day_db):
    persist_stories(day_db, "AMD", 3, stage="m2.exact")
    population = population_of(day_db)
    assert population.tickers == ("AMD", "NVDA")
    [skipped] = population.skipped
    assert (skipped.ticker, skipped.reason) == ("AMD", SKIP_M2_ONLY)
    assert "decision H" in skipped.detail
    assert all(row.ticker == "NVDA" for row in population.rows)


def test_healthy_stories_without_a_theme_set_are_skipped_with_their_own_reason(day_db):
    persist_stories(day_db, "AMD", 3)
    [skipped] = population_of(day_db).skipped
    assert skipped.reason == SKIP_THEMES_NOT_GENERATED


def test_a_theme_set_that_does_not_account_for_the_live_generation_is_skipped(day_db):
    """A set that places fewer stories than exist is inconsistent, not reviewable."""

    with day_db.admin.connect_writable() as connection:
        connection.execute(
            "DELETE FROM theme_excluded_stories WHERE theme_set_id = "
            "(SELECT id FROM theme_sets WHERE ticker = 'NVDA')"
        )
    [skipped] = population_of(day_db).skipped
    assert skipped.reason == SKIP_SET_INCONSISTENT


def test_a_requested_ticker_with_nothing_persisted_is_named(day_db):
    population = population_of(day_db, tickers=["NVDA", "TSLA"])
    [skipped] = population.skipped
    assert (skipped.ticker, skipped.reason) == ("TSLA", SKIP_NO_STORY_OUTPUT)


def test_skipped_partitions_travel_in_the_manifest(day_db, tmp_path):
    persist_stories(day_db, "AMD", 3, stage="m2.exact")
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    assert manifest["selection"]["skipped_partitions"][0]["reason"] == SKIP_M2_ONLY


# ----------------------------------------------------------------------
# Deterministic seeds, no replacement, visible shortfall, identity
# ----------------------------------------------------------------------


def test_the_same_seed_draws_the_same_rows(day_db):
    population = population_of(day_db)
    first = sample_assignments(population, seed="s", size=4)
    second = sample_assignments(population, seed="s", size=4)
    assert [r.row_id for r in first.rows] == [r.row_id for r in second.rows]
    assert len(first.rows) == 4


def test_a_different_seed_changes_the_draw(wide_db):
    population = population_of(wide_db)
    a = {r.row_id for r in sample_assignments(population, seed="a", size=40).rows}
    b = {r.row_id for r in sample_assignments(population, seed="b", size=40).rows}
    assert a != b


def test_requesting_more_than_the_population_takes_it_whole_and_says_so(day_db):
    sample = sample_assignments(population_of(day_db), seed="s", size=40)
    assert (sample.actual_size, sample.requested_size, sample.shortfall) == (8, 40, 32)
    assert len({r.row_id for r in sample.rows}) == 8


def test_a_seed_and_a_positive_size_are_required(day_db):
    population = population_of(day_db)
    with pytest.raises(ReviewSamplingError, match="seed is required"):
        sample_assignments(population, seed="", size=4)
    with pytest.raises(ReviewSamplingError, match="positive integer"):
        sample_assignments(population, seed="s", size=0)


def test_row_identity_is_the_persisted_placement_with_a_full_sha256(day_db):
    row = next(r for r in population_of(day_db).rows if r.theme_key == "t-alpha")
    identity = row.identity()
    assert set(identity) == {"gate", *IDENTITY_FIELDS}
    assert identity["theme_set_id"].isdigit() and identity["story_id"].isdigit()
    assert row.row_id == row_id_for(identity)
    assert row.row_id.startswith("g1-") and len(row.row_id) == 3 + 64


def test_moving_a_story_to_another_theme_changes_its_row_id(day_db):
    before = {r.story_id: r.row_id for r in population_of(day_db).rows}
    ids = [int(r.story_id) for r in population_of(day_db).rows]
    persist_theme_set(
        day_db,
        "NVDA",
        themes={"t-alpha": ids[0:2], "t-beta": ids[2:5]},
        other=ids[5:7],
        excluded=ids[7:8],
    )
    after = {r.story_id: r.row_id for r in population_of(day_db).rows}
    assert before[str(ids[2])] != after[str(ids[2])]
    assert before[str(ids[0])] == after[str(ids[0])]


# ----------------------------------------------------------------------
# The manifest snapshot is the review boundary (Codex 6, 7, 8)
# ----------------------------------------------------------------------


def test_manifest_captures_the_full_review_context_and_binds_it(day_db, tmp_path):
    population = population_of(day_db)
    csv_path, manifest_path = write_round(population, tmp_path, size=8)
    assert manifest_path == manifest_path_for(csv_path)
    manifest = read_manifest(manifest_path)
    assert manifest["schema"] == "a4a-review-sample/3"
    assert manifest["gate"] == "G1"
    assert manifest["origin"]["status"] == "unverified"
    assert manifest["operator_attestation"] is None
    assert "database_sha256" not in manifest["source"]
    rows = manifest["snapshot"]["rows"]
    assert [r["row_id"] for r in rows] == manifest["sample"]["row_ids"]
    assert set(rows[0]) == set(SNAPSHOT_FIELDS)
    assert set(CONTEXT_FIELDS) >= {
        "story_title",
        "story_description",
        "story_canonical_url",
        "story_outlets",
        "story_stage",
        "theme_label",
        "theme_label_source",
        "theme_story_count",
        "sibling_story_titles",
    }
    provenance = manifest["snapshot"]["theme_sets"][0]
    assert {
        "config_fingerprint",
        "algorithm_version",
        "model_name",
        "model_revision",
        "embedding_dimension",
        "method",
        "story_generation_signature_current",
        "theme_build_story_generation_signature",
        "generation_binding",
    } <= set(provenance)
    assert len(manifest["snapshot"]["sha256"]) == 64
    assert manifest["population"]["digest"] == population.digest
    assert manifest["selection"]["digest"] == population.selection_digest
    assert manifest["labeling_protocol"] == {
        "id": UNRATIFIED_PROTOCOL,
        "vocabulary_at_sampling": {"positive": "correct", "negative": "incorrect"},
        "note": manifest["labeling_protocol"]["note"],
    }
    assert "ratified" not in manifest["labeling_protocol"]


@pytest.mark.parametrize(
    "column",
    ["story_title", "theme_label", "story_description", "sibling_story_titles"],
)
def test_editing_review_context_in_a_completed_sheet_is_rejected(
    day_db, tmp_path, column
):
    """Codex 6."""

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    rows = read_rows(sheet)
    target = next(r for r in rows if r["assignment_type"] == "theme")
    target[column] = target[column] + " (edited)"
    write_rows(sheet, rows)
    with pytest.raises(ReviewSamplingError, match=f"column {column!r} differs"):
        score_round(manifest, [sheet])


def test_a_sheet_missing_a_manifest_row_is_rejected(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    write_rows(sheet, read_rows(sheet)[:-1])
    with pytest.raises(ReviewSamplingError, match="missing manifest rows"):
        score_round(manifest, [sheet])


def test_a_sheet_with_an_extra_row_is_rejected(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    rows = read_rows(sheet)
    write_rows(sheet, rows + [dict(rows[0], row_id="g1-" + "0" * 64)])
    with pytest.raises(ReviewSamplingError, match="not in the manifest"):
        score_round(manifest, [sheet])


def test_a_manifest_whose_snapshot_was_altered_is_rejected(day_db, tmp_path):
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    payload = json.loads(manifest_path.read_text())
    payload["snapshot"]["rows"][0]["story_title"] = "something else"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ReviewSamplingError, match="review context was altered"):
        read_manifest(manifest_path)


def test_a_manifest_whose_identity_was_altered_is_rejected(day_db, tmp_path):
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    payload = json.loads(manifest_path.read_text())
    payload["snapshot"]["rows"][0]["story_id"] = "999999"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ReviewSamplingError, match="does not match its own identity"):
        read_manifest(manifest_path)


def test_a_manifest_missing_a_context_column_is_rejected(day_db, tmp_path):
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    payload = json.loads(manifest_path.read_text())
    for row in payload["snapshot"]["rows"]:
        del row["story_description"]
    payload["snapshot"]["sha256"] = review.snapshot_digest(
        payload["snapshot"]["rows"], payload["snapshot"]["theme_sets"]
    )
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ReviewSamplingError, match="captured review columns"):
        read_manifest(manifest_path)


def test_database_mutation_after_sampling_does_not_alter_the_captured_review(
    day_db, tmp_path
):
    """Codex 7: scoring is of the snapshot; the live database is not consulted."""

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    before = score_gate([score_round(read_manifest(manifest_path), [sheet])]).as_dict()

    ids = [int(r.story_id) for r in population_of(day_db).rows]
    persist_theme_set(
        day_db,
        "NVDA",
        themes={"t-alpha": ids[0:1], "t-beta": ids[1:5]},
        other=ids[5:8],
        label_prefix="Renamed",
    )
    assert (
        population_of(day_db).digest
        != read_manifest(manifest_path)["population"]["digest"]
    )

    after = score_gate([score_round(read_manifest(manifest_path), [sheet])]).as_dict()
    assert after == before
    assert (
        after["rounds"][0]["snapshot_sha256"] == before["rounds"][0]["snapshot_sha256"]
    )


def test_wal_writes_cannot_undermine_the_logical_snapshot(day_db, tmp_path):
    """Codex 8: integrity is the logical snapshot, not the main file's bytes."""

    with day_db.admin.connect_writable() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    main_before = hashlib.sha256(day_db.database_path.read_bytes()).hexdigest()

    ids = [int(r.story_id) for r in population_of(day_db).rows]
    persist_theme_set(day_db, "NVDA", themes={"t-alpha": ids[0:5]}, other=ids[5:8])

    # The main file's bytes are not what the tool relies on; whatever WAL
    # did to them, the logical population moved and the snapshot did not.
    assert "database_sha256" not in manifest["source"]
    assert population_of(day_db).digest != manifest["population"]["digest"]
    assert (
        read_manifest(manifest_path)["snapshot"]["sha256"]
        == manifest["snapshot"]["sha256"]
    )
    _ = main_before


# ----------------------------------------------------------------------
# Origin (Codex 1, 2, 15)
# ----------------------------------------------------------------------


def test_persisted_rows_are_unverified_and_no_dataset_kind_is_claimed(day_db, tmp_path):
    population = population_of(day_db)
    assert review.classify_origin(population.source)[0] is OriginStatus.UNVERIFIED
    csv_path, manifest_path = write_round(population, tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    card = score_gate([score_round(read_manifest(manifest_path), [sheet])])
    assert card.origin_status is OriginStatus.UNVERIFIED
    assert card.trust_contract is None
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert "Origin unverifiable" in card.banner
    assert "synthetic" not in card.banner.lower()


def test_fake_evidence_with_a_ledger_and_an_attestation_cannot_upgrade(
    day_db, tmp_path, monkeypatch
):
    """Codex 1: manually inserted rows + timestamps + payload + ingestion run
    + attestation stays NOT_ELIGIBLE, even under a ratified protocol."""

    monkeypatch.setattr(review, "RATIFIED_PROTOCOLS", {TEST_PROTOCOL.id: TEST_PROTOCOL})
    population = population_of(day_db)
    indicators = population.indicators["member_raw_items"]
    assert indicators["with_fetched_at"] == indicators["with_provider_payload"] == 8
    assert population.indicators["runs_by_day"][DAY][YAHOO_FETCH_STAGE] == {
        "success": 1
    }

    sample = sample_assignments(population, seed="s", size=8)
    manifest = build_manifest(
        sample,
        protocol_id=TEST_PROTOCOL.id,
        csv_name="r.csv",
        code={"commit": "x", "dirty": False},
        operator_attestation=OperatorAttestation(
            "ops-kartik", "soak window day 3, run_log checked", "2026-09-12"
        ),
    )
    assert manifest["origin"]["status"] == "unverified"
    assert manifest["operator_attestation"]["effect"].startswith("none")
    csv_path, manifest_path = write_sample(
        sample, tmp_path / "r.csv", manifest=manifest
    )
    adj = adjudicate(tmp_path / "adj.csv", {})
    card = score_gate(
        [two_reviewer_round(csv_path, manifest_path, tmp_path, adjudicated=adj)],
        development=DevelopmentOverrides(required_unique_assignments=8),
    )
    assert card.threshold_met is True
    assert card.origin_status is OriginStatus.UNVERIFIED
    assert card.gate_eligible is False
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert any("origin is unverified" in b for b in card.eligibility_blockers)


def test_an_attestation_is_recorded_but_changes_nothing_about_origin(day_db, tmp_path):
    sample = sample_assignments(population_of(day_db), seed="s", size=8)
    with_it = build_manifest(
        sample,
        csv_name="r.csv",
        code={"commit": "x", "dirty": False},
        operator_attestation=OperatorAttestation("ops", "checked", "2026-09-12"),
    )
    without = build_manifest(
        sample, csv_name="r.csv", code={"commit": "x", "dirty": False}
    )
    assert with_it["origin"] == without["origin"]
    assert with_it["snapshot"]["sha256"] == without["snapshot"]["sha256"]


def test_an_edited_manifest_claiming_a_ratified_protocol_stays_unratified(
    day_db, tmp_path, eligible_world
):
    """Codex 2 and 15: the artifact's say-so about ratification is ignored."""

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    payload = json.loads(manifest_path.read_text())
    payload["labeling_protocol"]["id"] = "invented-k3"
    payload["labeling_protocol"]["ratified"] = True
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ReviewSamplingError, match="altered after its sheet was cut"):
        read_manifest(manifest_path)

    # Re-authored with every digest recomputed and the sheet recut: readable,
    # but the registry, not the file, decides ratification.
    def claim_ratified(payload):
        payload["labeling_protocol"]["id"] = "invented-k3"
        payload["labeling_protocol"]["ratified"] = True

    reauthor(manifest_path, csv_path, claim_ratified)
    manifest = read_manifest(manifest_path)
    assert resolve_protocol(manifest["labeling_protocol"]["id"])[1] is False
    adj = adjudicate(tmp_path / "adj.csv", {})
    card = score_gate(
        [two_reviewer_round(csv_path, manifest_path, tmp_path, adjudicated=adj)],
        development=DevelopmentOverrides(required_unique_assignments=8),
    )
    assert card.protocol_ratified is False
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert any(
        "'invented-k3' is not in the ratified registry" in b
        for b in card.eligibility_blockers
    )


def test_sampling_refuses_an_unknown_protocol_id(day_db):
    sample = sample_assignments(population_of(day_db), seed="s", size=8)
    with pytest.raises(ReviewSamplingError, match="unknown labeling protocol"):
        build_manifest(sample, protocol_id="k3-v1", csv_name="r.csv", code={})


# ----------------------------------------------------------------------
# Serialized reports carry no authority (Codex 3)
# ----------------------------------------------------------------------


def test_a_hand_built_round_result_is_not_scored(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    with pytest.raises(ReviewSamplingError, match="must come from score_round"):
        score_gate([{"round_id": "forged"}])


@pytest.mark.parametrize(
    "field, value",
    [
        ("reviewer_ids", ("alice", "bob")),
        ("adjudication_state", AdjudicationState.RESOLVED),
        ("ratified", True),
        ("adjudicator_ids", ("carol",)),
    ],
)
def test_tampering_with_a_round_result_is_detected_against_its_artifacts(
    day_db, tmp_path, field, value
):
    """Codex 3: a RoundResult is re-derived from the artifacts it names."""

    import dataclasses

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    honest = score_round(read_manifest(manifest_path), [sheet])
    forged = dataclasses.replace(honest, **{field: value})
    with pytest.raises(
        ReviewSamplingError, match="does not match its source artifacts"
    ):
        score_gate([forged])


def test_tampering_with_outcomes_or_provenance_in_a_round_result_is_detected(
    day_db, tmp_path
):
    import dataclasses

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "incorrect")
    honest = score_round(read_manifest(manifest_path), [sheet])
    flipped = tuple(
        dataclasses.replace(o, resolved=True, final_verdict="correct")
        for o in honest.outcomes
    )
    with pytest.raises(
        ReviewSamplingError, match="does not match its source artifacts"
    ):
        score_gate([dataclasses.replace(honest, outcomes=flipped)])
    edited_manifest = dict(honest.manifest)
    edited_manifest["source"] = {"mode": "fixture"}
    with pytest.raises(ReviewSamplingError, match="manifest on disk"):
        score_gate([dataclasses.replace(honest, manifest=edited_manifest)])


def test_a_sheet_changed_after_scoring_is_detected(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "incorrect")
    honest = score_round(read_manifest(manifest_path), [sheet])
    fill(csv_path, sheet, "correct")
    with pytest.raises(
        ReviewSamplingError, match="does not match its source artifacts"
    ):
        score_gate([honest])


# ----------------------------------------------------------------------
# Cross-round compatibility (Codex 4, 5, 16)
# ----------------------------------------------------------------------


def test_two_rounds_from_different_populations_cannot_combine(
    day_db, wide_db, tmp_path
):
    """Codex 4."""

    csv1, m1 = write_round(population_of(day_db), tmp_path, name="r1", size=8)
    csv2, m2 = write_round(population_of(wide_db), tmp_path, name="r2")
    a = score_round(read_manifest(m1), [fill(csv1, tmp_path / "a1.csv", "correct")])
    b = score_round(read_manifest(m2), [fill(csv2, tmp_path / "a2.csv", "correct")])
    with pytest.raises(ReviewSamplingError, match="population digest"):
        score_gate([a, b])


def test_a_placement_changed_between_rounds_makes_the_populations_incompatible(
    wide_db, tmp_path
):
    """Codex 5: a re-placed story is a different population, not another row."""

    csv1, m1 = write_round(population_of(wide_db), tmp_path, seed="one", name="r1")
    first = read_manifest(m1)
    nvda = [int(r.story_id) for r in population_of(wide_db).rows if r.ticker == "NVDA"]
    persist_theme_set(
        wide_db,
        "NVDA",
        themes={"NVDA-a": nvda[0:19], "NVDA-b": nvda[19:40]},
        other=nvda[40:48],
        excluded=nvda[48:50],
    )
    changed = population_of(wide_db)
    with pytest.raises(ReviewSamplingError, match="different population"):
        sample_assignments(changed, seed="two", size=40, prior_manifests=[first])
    csv2, m2 = write_round(changed, tmp_path, seed="two", name="r2")
    a = score_round(first, [fill(csv1, tmp_path / "a1.csv", "correct")])
    b = score_round(read_manifest(m2), [fill(csv2, tmp_path / "a2.csv", "correct")])
    with pytest.raises(ReviewSamplingError, match="population digest"):
        score_gate([a, b])


def test_unlinked_rounds_from_one_population_cannot_combine(wide_db, tmp_path):
    """Two draws that did not exclude each other cannot prove they differ."""

    population = population_of(wide_db)
    csv1, m1 = write_round(population, tmp_path, seed="one", name="r1")
    csv2, m2 = write_round(population, tmp_path, seed="two", name="r2")
    a = score_round(read_manifest(m1), [fill(csv1, tmp_path / "a1.csv", "correct")])
    b = score_round(read_manifest(m2), [fill(csv2, tmp_path / "a2.csv", "correct")])
    with pytest.raises(
        ReviewSamplingError, match="every earlier round must have been excluded"
    ):
        score_gate([a, b])


def test_the_same_forty_scored_twice_cannot_combine(wide_db, tmp_path):
    population = population_of(wide_db)
    csv1, m1 = write_round(population, tmp_path, seed="one", name="r1")
    a = score_round(read_manifest(m1), [fill(csv1, tmp_path / "a1.csv", "correct")])
    b = score_round(
        read_manifest(m1), [fill(csv1, tmp_path / "a2.csv", "correct", "bob")]
    )
    with pytest.raises(
        ReviewSamplingError, match="every earlier round must have been excluded"
    ):
        score_gate([a, b])


def test_a_round_excluding_a_manifest_that_is_not_scored_cannot_combine(
    wide_db, tmp_path
):
    population = population_of(wide_db)
    csv1, m1 = write_round(population, tmp_path, seed="one", name="r1")
    csv2, m2 = write_round(
        population, tmp_path, seed="two", name="r2", priors=[read_manifest(m1)]
    )
    csv3, m3 = write_round(population, tmp_path, seed="three", name="r3")
    b = score_round(read_manifest(m2), [fill(csv2, tmp_path / "a2.csv", "correct")])
    c = score_round(read_manifest(m3), [fill(csv3, tmp_path / "a3.csv", "correct")])
    with pytest.raises(ReviewSamplingError):
        score_gate([b, c])


def test_fixture_and_persisted_rounds_cannot_combine(day_db, tmp_path):
    """Codex 16."""

    csv1, m1 = write_round(population_of(day_db), tmp_path, name="r1", size=8)
    csv2, m2 = write_round(load_fixture_population(), tmp_path, name="fx", size=8)
    a = score_round(read_manifest(m1), [fill(csv1, tmp_path / "a1.csv", "correct")])
    b = score_round(read_manifest(m2), [fill(csv2, tmp_path / "a2.csv", "correct")])
    with pytest.raises(ReviewSamplingError, match="population digest|source mode"):
        score_gate([a, b])


def test_two_linked_rounds_of_forty_reach_the_floor_together(wide_db, tmp_path):
    rounds = two_linked_rounds(population_of(wide_db), tmp_path)
    r1, r2 = rounds
    assert not set(o.row_id for o in r1.outcomes) & set(o.row_id for o in r2.outcomes)
    assert (
        r2.manifest["sample"]["prior_rounds"][0]["manifest_sha256"]
        == r1.manifest_sha256
    )
    card = score_gate(rounds)
    assert card.unique_assignments == RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS
    assert card.review_complete is True
    assert card.threshold_met is True
    assert card.gate_result is GateResult.NOT_ELIGIBLE  # unverified, unratified


# ----------------------------------------------------------------------
# Locked release requirements (Codex 9, 10)
# ----------------------------------------------------------------------


def test_release_requirements_are_the_specs():
    assert RELEASE_G1_THRESHOLD == 0.75
    assert RELEASE_G1_REQUIRED_UNIQUE_ASSIGNMENTS == 80


def test_lowering_threshold_or_floor_forces_a_development_evaluation(
    wide_db, tmp_path, eligible_world
):
    """Codex 9: even a world that could PASS cannot PASS with overrides."""

    csv1, m1 = ratified_manifest(population_of(wide_db), tmp_path, "r1")
    adj = adjudicate(tmp_path / "adj.csv", {})
    single = [two_reviewer_round(csv1, m1, tmp_path, adjudicated=adj)]
    assert score_gate(single).gate_result is GateResult.INCOMPLETE  # 40 < 80
    lowered = score_gate(
        single,
        development=DevelopmentOverrides(threshold=0.0, required_unique_assignments=1),
    )
    assert lowered.evaluation_mode == "development"
    assert lowered.threshold_met is True and lowered.review_complete is True
    assert lowered.gate_result is GateResult.NOT_ELIGIBLE
    assert any("development overrides" in b for b in lowered.eligibility_blockers)


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf, -0.1, 1.5, "0.5"])
def test_invalid_thresholds_are_rejected(threshold):
    """Codex 10."""

    with pytest.raises(ReviewSamplingError):
        DevelopmentOverrides(threshold=threshold)


@pytest.mark.parametrize("count", [0, -1, 2.5, True, "80"])
def test_invalid_required_counts_are_rejected(count):
    with pytest.raises(ReviewSamplingError):
        DevelopmentOverrides(required_unique_assignments=count)


def test_the_cli_rejects_invalid_development_values(day_db, tmp_path, capsys):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    for bad in (
        ["--development-threshold", "nan"],
        ["--development-threshold", "inf"],
        ["--development-threshold", "-1"],
        ["--development-required-unique", "0"],
    ):
        code = make_review_sheets.main(
            ["score-assignments", "--round", str(manifest_path), str(sheet), *bad]
        )
        assert code == 2, bad


# ----------------------------------------------------------------------
# The gate-result matrix, reachable only by code
# ----------------------------------------------------------------------


def test_derive_gate_result_follows_the_locked_precedence():
    assert (
        derive_gate_result(
            gate_eligible=False, review_complete=True, threshold_met=True
        )
        is GateResult.NOT_ELIGIBLE
    )
    assert (
        derive_gate_result(
            gate_eligible=True, review_complete=False, threshold_met=True
        )
        is GateResult.INCOMPLETE
    )
    assert (
        derive_gate_result(gate_eligible=True, review_complete=True, threshold_met=True)
        is GateResult.PASS
    )
    assert (
        derive_gate_result(
            gate_eligible=True, review_complete=True, threshold_met=False
        )
        is GateResult.FAIL
    )
    assert (
        derive_gate_result(gate_eligible=True, review_complete=True, threshold_met=None)
        is GateResult.FAIL
    )


def eligible_rounds(population, directory, *, bob=None):
    csv1, m1 = ratified_manifest(population, directory, "r1", seed="one")
    csv2, m2 = ratified_manifest(
        population, directory, "r2", seed="two", priors=[read_manifest(m1)]
    )
    adj1 = adjudicate(directory / "adj1.csv", {})
    adj2 = adjudicate(directory / "adj2.csv", {})
    return [
        two_reviewer_round(csv1, m1, directory, bob=bob, adjudicated=adj1),
        two_reviewer_round(csv2, m2, directory, bob=bob, adjudicated=adj2),
    ]


def test_pass_needs_verified_origin_ratified_protocol_two_reviewers_and_the_floor(
    wide_db, tmp_path, eligible_world
):
    card = score_gate(eligible_rounds(population_of(wide_db), tmp_path))
    assert card.origin_status is OriginStatus.VERIFIED_LIVE
    assert card.protocol_ratified is True
    assert card.adjudication_state is AdjudicationState.UNANIMOUS
    assert (card.threshold_met, card.review_complete, card.gate_eligible) == (
        True,
        True,
        True,
    )
    assert card.gate_result is GateResult.PASS
    assert card.trust_contract.dataset_kind is DatasetKind.SAMPLED_PRODUCTION
    assert (
        card.trust_contract.labeling_status is LabelingStatus.MULTI_REVIEWER_ADJUDICATED
    )
    assert card.trust_contract.metrics_purpose is MetricsPurpose.GATE_ACCEPTANCE


def test_fail_is_an_eligible_complete_review_below_the_threshold(
    wide_db, tmp_path, eligible_world
):
    rounds = eligible_rounds(population_of(wide_db), tmp_path)
    # Both reviewers unanimous on "incorrect" for every theme row.
    csv1, m1 = ratified_manifest(population_of(wide_db), tmp_path, "s1", seed="one")
    csv2, m2 = ratified_manifest(
        population_of(wide_db), tmp_path, "s2", seed="two", priors=[read_manifest(m1)]
    )

    def harsh(row):
        return "incorrect" if row["assignment_type"] == "theme" else "correct"

    rounds = []
    for csv_path, manifest_path, adj in ((csv1, m1, "x1"), (csv2, m2, "x2")):
        manifest = read_manifest(manifest_path)
        a = fill(csv_path, tmp_path / f"{adj}.a.csv", harsh, "alice")
        b = fill(csv_path, tmp_path / f"{adj}.b.csv", harsh, "bob")
        rounds.append(
            score_round(
                manifest, [a, b], adjudicated=adjudicate(tmp_path / f"{adj}.csv", {})
            )
        )
    card = score_gate(rounds)
    assert card.gate_eligible is True and card.review_complete is True
    assert card.threshold_met is False
    assert card.gate_result is GateResult.FAIL


def test_incomplete_when_rows_are_unresolved(wide_db, tmp_path, eligible_world):
    population = population_of(wide_db)
    csv1, m1 = ratified_manifest(population, tmp_path, "r1", seed="one")
    csv2, m2 = ratified_manifest(
        population, tmp_path, "r2", seed="two", priors=[read_manifest(m1)]
    )

    def blanks(row):
        return "" if row["assignment_type"] == "excluded" else "correct"

    rounds = []
    for csv_path, manifest_path, tag in ((csv1, m1, "1"), (csv2, m2, "2")):
        a = fill(csv_path, tmp_path / f"a{tag}.csv", blanks, "alice")
        b = fill(csv_path, tmp_path / f"b{tag}.csv", blanks, "bob")
        rounds.append(
            score_round(
                read_manifest(manifest_path),
                [a, b],
                adjudicated=adjudicate(tmp_path / f"adj{tag}.csv", {}),
            )
        )
    card = score_gate(rounds)
    assert card.gate_eligible is True
    assert card.unresolved_count > 0
    assert card.review_complete is False
    assert card.gate_result is GateResult.INCOMPLETE


def test_a_round_with_a_shortfall_is_not_release_complete(
    day_db, tmp_path, eligible_world
):
    csv_path, manifest_path = ratified_manifest(population_of(day_db), tmp_path, "r1")
    card = score_gate(
        [
            two_reviewer_round(
                csv_path,
                manifest_path,
                tmp_path,
                adjudicated=adjudicate(tmp_path / "adj.csv", {}),
            )
        ]
    )
    assert card.gate_eligible is True
    assert any("population exhausted" in r for r in card.incompleteness)
    assert card.gate_result is GateResult.INCOMPLETE


# ----------------------------------------------------------------------
# Reviewers and adjudication (Codex 13)
# ----------------------------------------------------------------------


def test_a_perfect_single_reviewer_review_is_provisional_and_not_eligible(
    day_db, tmp_path
):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    card = score_gate(
        [score_round(read_manifest(manifest_path), [sheet])],
        development=DevelopmentOverrides(required_unique_assignments=8),
    )
    assert card.rate == 1.0 and card.threshold_met is True
    assert card.adjudication_state is AdjudicationState.NOT_APPLICABLE
    assert card.reviewer_count == 1
    assert card.gate_result is GateResult.NOT_ELIGIBLE


def test_an_empty_adjudication_file_does_not_make_a_round_adjudicated(
    wide_db, tmp_path, eligible_world, monkeypatch
):
    """Codex 13: unanimous is unanimous; what counts as adjudicated is K3's."""

    strict = Protocol(
        id=TEST_PROTOCOL.id,
        positive_verdict="correct",
        negative_verdict="incorrect",
        adjudicated_states=frozenset({AdjudicationState.RESOLVED}),
    )
    monkeypatch.setattr(review, "RATIFIED_PROTOCOLS", {strict.id: strict})
    card = score_gate(eligible_rounds(population_of(wide_db), tmp_path))
    assert card.adjudication_state is AdjudicationState.UNANIMOUS
    assert (
        card.trust_contract.labeling_status
        is LabelingStatus.MULTI_REVIEWER_UNADJUDICATED
    )
    assert card.gate_eligible is False
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert any(
        "'unanimous' is not one the protocol counts" in b
        for b in card.eligibility_blockers
    )


def test_a_disagreement_is_open_until_adjudicated_and_then_resolved(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    a = fill(csv_path, tmp_path / "a.csv", "correct", "alice")
    b = fill(
        csv_path,
        tmp_path / "b.csv",
        lambda row: "incorrect" if row["theme_key"] == "t-beta" else "correct",
        "bob",
    )
    open_round = score_round(manifest, [a, b])
    assert open_round.adjudication_state is AdjudicationState.OPEN
    disputed = [o.row_id for o in open_round.unresolved]
    assert len(disputed) == 2
    assert open_round.agreement_rate == pytest.approx(6 / 8)
    assert open_round.adjudication_state is AdjudicationState.OPEN

    adj = adjudicate(
        tmp_path / "adj.csv", {disputed[0]: "incorrect", disputed[1]: "correct"}
    )
    closed = score_round(manifest, [a, b], adjudicated=adj)
    assert closed.adjudication_state is AdjudicationState.RESOLVED
    assert closed.adjudicator_ids == ("carol",)
    card = score_gate(
        [closed], development=DevelopmentOverrides(required_unique_assignments=8)
    )
    assert card.rate == pytest.approx(7 / 8)
    assert card.adjudication_state is AdjudicationState.RESOLVED
    assert card.gate_result is GateResult.NOT_ELIGIBLE


def test_blank_verdicts_are_unresolved_and_left_out_of_the_rate(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(
        csv_path,
        tmp_path / "a.csv",
        lambda row: "" if row["assignment_type"] == "excluded" else "correct",
    )
    result = score_round(read_manifest(manifest_path), [sheet])
    assert (
        len(result.unresolved) == 1
        and result.unresolved[0].unresolved_reason == "blank"
    )
    card = score_gate([result])
    assert card.resolved_count == 7 and card.rate == 1.0
    assert card.review_complete is False


def test_two_sheets_from_one_reviewer_are_not_two_reviewers(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    a = fill(csv_path, tmp_path / "a.csv", "correct", reviewer="alice")
    b = fill(csv_path, tmp_path / "b.csv", "correct", reviewer="alice")
    with pytest.raises(ReviewSamplingError, match="not two reviewers"):
        score_round(manifest, [a, b])


def test_a_verdict_outside_the_vocabulary_or_without_a_reviewer_is_rejected(
    day_db, tmp_path
):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    with pytest.raises(
        ReviewSamplingError, match="is not 'correct', 'incorrect', or blank"
    ):
        score_round(manifest, [fill(csv_path, tmp_path / "a.csv", "mostly")])
    with pytest.raises(ReviewSamplingError, match="no reviewer_id"):
        score_round(
            manifest, [fill(csv_path, tmp_path / "b.csv", "correct", reviewer="")]
        )


# ----------------------------------------------------------------------
# Redaction (Codex 14)
# ----------------------------------------------------------------------


def test_credentials_in_source_text_never_reach_the_sheet_or_manifest(tmp_path):
    repository = migrated(tmp_path)
    ids = persist_stories(
        repository,
        "NVDA",
        3,
        description="Standfirst. Authorization: Bearer source-secret trailing text",
        title="Headline api_key=abc123secret for NVDA",
    )
    persist_theme_set(repository, "NVDA", themes={"t": ids[:2]}, other=ids[2:])
    population = population_of(repository)
    csv_path, manifest_path = write_round(population, tmp_path, size=3)
    for text in (csv_path.read_text(), manifest_path.read_text()):
        assert "source-secret" not in text
        assert "abc123secret" not in text
        assert "[REDACTED]" in text
    row = read_rows(csv_path)[0]
    assert row["story_description"].startswith("Standfirst. Authorization: [REDACTED]")
    assert (
        "REDACTED" in row["sibling_story_titles"] or row["sibling_story_titles"] == ""
    )


def test_credentials_in_an_attestation_are_redacted(day_db, tmp_path):
    sample = sample_assignments(population_of(day_db), seed="s", size=8)
    manifest = build_manifest(
        sample,
        csv_name="r.csv",
        code={"commit": "x", "dirty": False},
        operator_attestation=OperatorAttestation(
            "ops", "checked with token=tok-secret-value", "2026-09-12"
        ),
    )
    assert "tok-secret-value" not in json.dumps(manifest)


# ----------------------------------------------------------------------
# Fixture mode, CSV context, scorecard shape, CLI
# ----------------------------------------------------------------------


def test_fixture_mode_is_synthetic_by_the_fixtures_own_declaration(tmp_path):
    population = load_fixture_population()
    assert population.source["mode"] == "fixture"
    assert population.indicators is None
    assert review.classify_origin(population.source)[0] is OriginStatus.SYNTHETIC
    csv_path, manifest_path = write_round(population, tmp_path, name="fx", size=30)
    adj = adjudicate(tmp_path / "adj.csv", {})
    card = score_gate(
        [two_reviewer_round(csv_path, manifest_path, tmp_path, adjudicated=adj)],
        development=DevelopmentOverrides(required_unique_assignments=30),
    )
    assert card.threshold_met is True and card.review_complete is True
    assert card.trust_contract is not None and card.trust_contract.is_synthetic
    assert card.gate_result is GateResult.NOT_ELIGIBLE


def test_a_perfect_fixture_review_cannot_pass_even_under_a_ratified_protocol(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(review, "RATIFIED_PROTOCOLS", {TEST_PROTOCOL.id: TEST_PROTOCOL})
    population = load_fixture_population()
    csv_path, manifest_path = ratified_manifest(population, tmp_path, "fx")
    adj = adjudicate(tmp_path / "adj.csv", {})
    card = score_gate(
        [two_reviewer_round(csv_path, manifest_path, tmp_path, adjudicated=adj)],
        development=DevelopmentOverrides(required_unique_assignments=30),
    )
    assert card.origin_status is OriginStatus.SYNTHETIC
    assert card.gate_result is GateResult.NOT_ELIGIBLE


def test_the_cli_only_enters_fixture_mode_when_told(tmp_path, capsys):
    out = tmp_path / "fx.csv"
    assert (
        make_review_sheets.main(
            ["sample-assignments", "--seed", "s", "--out", str(out)]
        )
        == 2
    )
    assert "needs --database" in capsys.readouterr().err
    assert (
        make_review_sheets.main(
            ["sample-assignments", "--fixture", "--seed", "s", "--out", str(out)]
        )
        == 0
    )
    assert read_manifest(manifest_path_for(out))["origin"]["status"] == "synthetic"


def test_the_sheet_carries_siblings_source_text_and_links(day_db, tmp_path):
    csv_path, _ = write_round(population_of(day_db), tmp_path, size=8)
    rows = read_rows(csv_path)
    assert tuple(rows[0].keys()) == ASSIGNMENT_FIELDNAMES
    alpha = [r for r in rows if r["theme_key"] == "t-alpha"]
    assert len(alpha) == 3
    for row in alpha:
        siblings = row["sibling_story_titles"].split(" | ")
        assert len(siblings) == 2 and row["story_title"] not in siblings
        assert row["story_description"].startswith("Standfirst for NVDA")
        assert row["story_canonical_url"].startswith("https://publisher.example/")
        assert row["story_outlets"]
        assert row["story_stage"] == "m3.semantic"
        assert row["theme_label"] == "Theme t-alpha"
        assert row["theme_story_count"] == "3"
    other = next(r for r in rows if r["assignment_type"] == "other_coverage")
    assert other["placement_reason"] == "clustering_noise"
    assert other["sibling_story_titles"] == "" and other["theme_key"] == ""
    for row in rows:
        assert row["reviewer_id"] == row["reviewed_at"] == row["reviewer_verdict"] == ""


def test_the_scorecard_never_exposes_a_meets_gate_boolean(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    payload = score_gate([score_round(read_manifest(manifest_path), [sheet])]).as_dict()
    assert "meets_gate" not in payload
    assert {"threshold_met", "review_complete", "gate_eligible", "gate_result"} <= set(
        payload
    )
    assert payload["trust_contract"] is None
    assert payload["origin"]["status"] == "unverified"
    assert "K4" in payload["note"]
    text = render_scorecard(
        score_gate([score_round(read_manifest(manifest_path), [sheet])])
    )
    assert text.startswith("WARNING: Origin unverifiable")
    assert "gate_result        NOT_ELIGIBLE" in text


def test_cli_scores_from_artifacts_and_exits_three_when_not_eligible(
    wide_db, tmp_path, capsys
):
    db = str(wide_db.database_path)
    out1, out2 = tmp_path / "r1.csv", tmp_path / "r2.csv"
    base = ["sample-assignments", "--database", db, "--day", DAY]
    assert (
        make_review_sheets.main(
            [*base, "--seed", "one", "--round-id", "r1", "--out", str(out1)]
        )
        == 0
    )
    assert (
        make_review_sheets.main(
            [
                *base,
                "--seed",
                "two",
                "--round-id",
                "r2",
                "--exclude-manifest",
                str(manifest_path_for(out1)),
                "--out",
                str(out2),
            ]
        )
        == 0
    )
    capsys.readouterr()
    sheets = {}
    for out in (out1, out2):
        sheets[out] = (
            fill(out, out.with_suffix(".a.csv"), "correct", "alice"),
            fill(out, out.with_suffix(".b.csv"), "correct", "bob"),
        )
    report = tmp_path / "scorecard.json"
    code = make_review_sheets.main(
        [
            "score-assignments",
            "--round",
            str(manifest_path_for(out1)),
            str(sheets[out1][0]),
            str(sheets[out1][1]),
            "--round",
            str(manifest_path_for(out2)),
            str(sheets[out2][0]),
            str(sheets[out2][1]),
            "--report",
            str(report),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 3
    assert payload["gate_result"] == "NOT_ELIGIBLE"
    assert payload["unique_assignments"] == 80
    assert payload["review_complete"] is True
    assert payload["threshold_met"] is True
    assert "meets_gate" not in payload
    assert json.loads(report.read_text())["round_reports"][0]["note"].startswith(
        "report only"
    )


def test_cli_records_an_attestation_as_metadata_only(day_db, tmp_path, capsys):
    out = tmp_path / "r.csv"
    code = make_review_sheets.main(
        [
            "sample-assignments",
            "--database",
            str(day_db.database_path),
            "--day",
            DAY,
            "--seed",
            "s",
            "--size",
            "8",
            "--attested-by",
            "ops",
            "--attestation",
            "checked",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert "does not change origin" in capsys.readouterr().err
    manifest = read_manifest(manifest_path_for(out))
    assert manifest["origin"]["status"] == "unverified"
    assert manifest["operator_attestation"]["attested_by"] == "ops"


def test_cli_exit_codes_follow_the_gate_result():
    assert make_review_sheets.EXIT_BY_RESULT[GateResult.PASS] == 0
    assert make_review_sheets.EXIT_BY_RESULT[GateResult.FAIL] == 1
    assert make_review_sheets.EXIT_BY_RESULT[GateResult.INCOMPLETE] == 3
    assert make_review_sheets.EXIT_BY_RESULT[GateResult.NOT_ELIGIBLE] == 3
    assert make_review_sheets.EXIT_USAGE == 2


def test_round_result_is_a_report_not_an_input():
    assert not hasattr(review, "read_round_result")
    assert "never read back from disk" in RoundResult.__doc__


def test_a_linked_round_can_be_scored_on_its_own(wide_db, tmp_path):
    """Its own development metrics are legitimate; only combination needs the chain."""

    population = population_of(wide_db)
    _, m1 = write_round(population, tmp_path, seed="one", name="r1")
    csv2, m2 = write_round(
        population, tmp_path, seed="two", name="r2", priors=[read_manifest(m1)]
    )
    alone = score_round(read_manifest(m2), [fill(csv2, tmp_path / "a2.csv", "correct")])
    card = score_gate([alone])
    assert card.unique_assignments == 40
    assert card.gate_result is GateResult.NOT_ELIGIBLE


# ----------------------------------------------------------------------
# Re-review finding 1: a completed sheet is bound to its exact snapshot
# ----------------------------------------------------------------------


def test_blank_sheets_carry_the_manifest_and_snapshot_they_were_cut_from(
    day_db, tmp_path
):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    rows = read_rows(csv_path)
    assert tuple(rows[0].keys()) == ASSIGNMENT_FIELDNAMES
    assert set(review.BINDING_FIELDS) <= set(rows[0])
    assert {r["manifest_id"] for r in rows} == {manifest["binding"]["manifest_id"]}
    assert {r["snapshot_sha256"] for r in rows} == {manifest["snapshot"]["sha256"]}
    assert manifest["binding"]["manifest_id"] == review.manifest_identity(manifest)
    assert manifest["sheet"]["binding_columns"] == list(review.BINDING_FIELDS)
    assert "manifest_id" not in manifest["snapshot"]["rows"][0]


def test_a_completed_sheet_cannot_be_rebound_to_a_reauthored_snapshot(day_db, tmp_path):
    """Codex's exact exploit: alter theme-set provenance, recompute every
    digest, hand in the unchanged completed sheets against the replacement."""

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    original = read_manifest(manifest_path)
    a = fill(csv_path, tmp_path / "a.csv", "correct", "alice")
    b = fill(csv_path, tmp_path / "b.csv", "correct", "bob")
    assert (
        score_round(original, [a, b]).adjudication_state is AdjudicationState.UNANIMOUS
    )

    payload = json.loads(manifest_path.read_text())
    payload["snapshot"]["theme_sets"][0]["model_name"] = "some-other-model"
    snapshot = payload["snapshot"]
    snapshot["sha256"] = review.snapshot_digest(
        snapshot["rows"], snapshot["theme_sets"]
    )
    selection = payload["selection"]
    selection["digest"] = review.selection_digest(
        trading_days=selection["trading_days"],
        tickers=selection["tickers"],
        pipeline_versions=selection["pipeline_versions"],
        theme_set_ids=selection["theme_set_ids"],
        source_mode=payload["source"]["mode"],
        theme_sets=snapshot["theme_sets"],
        skipped_partitions=selection["skipped_partitions"],
    )
    payload["population"]["digest"] = "0" * 64
    payload["binding"] = {
        "manifest_id": review.manifest_identity(payload),
        "snapshot_sha256": snapshot["sha256"],
    }
    replacement_path = tmp_path / "replacement.manifest.json"
    replacement_path.write_text(json.dumps(payload))
    replacement = read_manifest(replacement_path)
    assert replacement["binding"] != original["binding"]

    with pytest.raises(ReviewSamplingError, match="cannot be rebound"):
        score_round(replacement, [a, b])
    with pytest.raises(ReviewSamplingError, match="cannot be rebound"):
        score_round(replacement, [a])


def test_a_sheet_whose_binding_columns_were_edited_is_rejected(day_db, tmp_path):
    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    rows = read_rows(sheet)
    rows[3]["snapshot_sha256"] = "f" * 64
    write_rows(sheet, rows)
    with pytest.raises(ReviewSamplingError, match="'snapshot_sha256' does not name"):
        score_round(manifest, [sheet])
    rows = read_rows(sheet)
    rows[3]["snapshot_sha256"] = manifest["snapshot"]["sha256"]
    rows[0]["manifest_id"] = ""
    write_rows(sheet, rows)
    with pytest.raises(ReviewSamplingError, match="'manifest_id' does not name"):
        score_round(manifest, [sheet])


def test_a_manifest_edited_anywhere_after_its_sheet_was_cut_is_rejected(
    day_db, tmp_path
):
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    for mutate in (
        lambda p: p["sample"].__setitem__("round_id", "renamed"),
        lambda p: p["source"].__setitem__("mode", "fixture"),
        lambda p: p["origin"].__setitem__("status", "verified_live"),
        lambda p: p["sample"].__setitem__("seed", "another-seed"),
    ):
        payload = json.loads(manifest_path.read_text())
        mutate(payload)
        target = tmp_path / "edited.manifest.json"
        target.write_text(json.dumps(payload))
        with pytest.raises(ReviewSamplingError):
            read_manifest(target)


# ----------------------------------------------------------------------
# Re-review finding 2: the build binding is unknown, and said so
# ----------------------------------------------------------------------


def test_a_normal_persisted_theme_set_reports_current_and_build_signatures_apart(
    day_db, tmp_path
):
    """A: the current generation is reported as current; the build is unknown."""

    population = population_of(day_db)
    [provenance] = population.theme_sets
    current = day_db.story_generation("NVDA", DAY, VERSION).signature
    assert provenance.story_generation_signature_current == current
    assert provenance.theme_build_story_generation_signature is None
    assert provenance.generation_binding == review.GENERATION_BINDING_UNVERIFIED
    csv_path, manifest_path = write_round(population, tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    card = score_gate([score_round(read_manifest(manifest_path), [sheet])])
    assert any(
        "build provenance is ['unverified']" in b for b in card.eligibility_blockers
    )
    assert card.gate_result is GateResult.NOT_ELIGIBLE


def test_stories_mutated_in_place_do_not_become_the_generation_that_built_the_set(
    day_db, tmp_path
):
    """B: same ids, same stages, same memberships; different titles and hashes."""

    before = population_of(day_db)
    with day_db.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE stories SET canonical_title = canonical_title || ' (rewritten)', "
            "content_hash = content_hash || '-x' WHERE ticker = 'NVDA'"
        )
    after = population_of(day_db)
    assert after.rows and after.skipped == ()
    [was], [now] = before.theme_sets, after.theme_sets
    assert (
        now.story_generation_signature_current != was.story_generation_signature_current
    )
    # The set is still not claimed to have been built over what is there now.
    assert now.theme_build_story_generation_signature is None
    assert now.generation_binding == review.GENERATION_BINDING_UNVERIFIED
    assert after.digest != before.digest
    csv_path, manifest_path = write_round(after, tmp_path, size=8)
    sheet = fill(csv_path, tmp_path / "a.csv", "correct")
    card = score_gate([score_round(read_manifest(manifest_path), [sheet])])
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert any("proves nothing about the build" in b for b in card.eligibility_blockers)


def test_an_unverified_build_binding_blocks_an_otherwise_eligible_world(
    wide_db, tmp_path, monkeypatch
):
    """C: no historical binding is available; nothing else can stand in for it."""

    monkeypatch.setattr(review, "RATIFIED_PROTOCOLS", {TEST_PROTOCOL.id: TEST_PROTOCOL})
    monkeypatch.setattr(
        review,
        "classify_origin",
        lambda source: (OriginStatus.VERIFIED_LIVE, "patched for the test"),
    )
    card = score_gate(eligible_rounds(population_of(wide_db), tmp_path))
    assert card.origin_status is OriginStatus.VERIFIED_LIVE and card.protocol_ratified
    assert card.gate_eligible is False
    assert card.gate_result is GateResult.NOT_ELIGIBLE
    assert card.eligibility_blockers == (
        next(b for b in card.eligibility_blockers if "build provenance" in b),
    )


def test_a_theme_set_recording_a_different_input_count_is_inconsistent(day_db):
    with day_db.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE theme_sets SET source_metadata = '{\"story_count\": 99}' "
            "WHERE ticker = 'NVDA'"
        )
    [skipped] = population_of(day_db).skipped
    assert skipped.reason == SKIP_SET_INCONSISTENT
    assert "recorded 99 input stories" in skipped.detail


def test_the_fixture_binding_is_in_process_and_never_verified(tmp_path):
    [first, *_] = load_fixture_population().theme_sets
    assert first.generation_binding == review.GENERATION_BINDING_IN_PROCESS
    assert (
        first.theme_build_story_generation_signature
        == first.story_generation_signature_current
    )


# ----------------------------------------------------------------------
# Re-review finding 3: the selection digest is recomputed, never trusted
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s, p: s.__setitem__("tickers", ["TSLA"]), "selection was altered"),
        (
            lambda s, p: s.__setitem__("trading_days", ["2026-07-24"]),
            "selection was altered",
        ),
        (
            lambda s, p: s.__setitem__("pipeline_versions", ["v9"]),
            "selection was altered",
        ),
        (lambda s, p: s.__setitem__("theme_set_ids", ["999"]), "selection was altered"),
        (
            lambda s, p: p["source"].__setitem__("mode", "fixture"),
            "selection was altered",
        ),
        (lambda s, p: s.__setitem__("skipped_partitions", []), "selection was altered"),
    ],
    ids=["ticker", "day", "pipeline_version", "theme_set_id", "source_mode", "skipped"],
)
def test_editing_a_selection_field_without_its_digest_is_rejected(
    day_db, tmp_path, mutate, message
):
    persist_stories(day_db, "AMD", 2, stage="m2.exact")  # one skipped partition
    _, manifest_path = write_round(population_of(day_db), tmp_path, size=8)
    payload = json.loads(manifest_path.read_text())
    mutate(payload["selection"], payload)
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ReviewSamplingError, match=message):
        read_manifest(manifest_path)


def test_a_selection_recomputed_to_claim_unrelated_partitions_is_rejected(
    day_db, tmp_path
):
    """Digest recomputed honestly, but the selection no longer matches the snapshot."""

    csv_path, manifest_path = write_round(population_of(day_db), tmp_path, size=8)

    def claim(field, value):
        return lambda p: p["selection"].__setitem__(field, value)

    def rename_theme_set(p):
        p["snapshot"]["theme_sets"][0]["theme_set_id"] = "999"

    for name, mutate in (
        ("tickers", claim("tickers", ["TSLA"])),
        ("trading_days", claim("trading_days", ["2026-07-24"])),
        ("pipeline_versions", claim("pipeline_versions", ["v9"])),
        ("theme_set_id", rename_theme_set),
    ):
        target_csv = tmp_path / f"{name}.csv"
        target = manifest_path_for(target_csv)
        target.write_text(manifest_path.read_text())
        target_csv.write_text(csv_path.read_text())
        reauthor(target, target_csv, mutate)
        with pytest.raises(ReviewSamplingError, match="lies outside the selection"):
            read_manifest(target)


def test_selection_digest_is_one_function_used_at_write_and_read(day_db, tmp_path):
    population = population_of(day_db)
    _, manifest_path = write_round(population, tmp_path, size=8)
    manifest = read_manifest(manifest_path)
    assert manifest["selection"]["digest"] == population.selection_digest
    assert manifest["selection"]["digest"] == review.selection_digest(
        trading_days=manifest["selection"]["trading_days"],
        tickers=manifest["selection"]["tickers"],
        pipeline_versions=manifest["selection"]["pipeline_versions"],
        theme_set_ids=manifest["selection"]["theme_set_ids"],
        source_mode=manifest["source"]["mode"],
        theme_sets=manifest["snapshot"]["theme_sets"],
        skipped_partitions=manifest["selection"]["skipped_partitions"],
    )


# ----------------------------------------------------------------------
# Re-review finding 4: credential-like provenance identifiers are refused
# ----------------------------------------------------------------------


def _set_theme_set_field(repository, ticker, column, value):
    with repository.admin.connect_writable() as connection:
        connection.execute(
            f"UPDATE theme_sets SET {column} = ? WHERE ticker = ?", (value, ticker)
        )


@pytest.mark.parametrize(
    "column, value",
    [
        ("model_name", "api_key=provenance-secret"),
        ("model_revision", "token=revision-secret"),
        ("algorithm_version", "auth_token=algo-secret"),
        ("config_fingerprint", "password=config-secret"),
        ("method", "hdbscan"),  # control: a legitimate value stays usable
    ],
)
def test_a_credential_bearing_provenance_identifier_never_leaks(
    day_db, tmp_path, column, value
):
    _set_theme_set_field(day_db, "NVDA", column, value)
    population = population_of(day_db)
    secret = value.split("=")[-1].split(" ")[-1]
    if column == "method":
        assert population.rows and population.skipped == ()
        return
    assert population.rows == ()
    [skipped] = population.skipped
    assert skipped.reason == review.SKIP_PROVENANCE_CREDENTIAL
    assert column in skipped.detail
    assert secret not in skipped.detail
    assert secret not in json.dumps([s.as_dict() for s in population.theme_sets])
    with pytest.raises(ReviewSamplingError):
        sample_assignments(population, seed="s", size=4)
    manifest_text = json.dumps(
        {
            "selection": [s.as_dict() for s in population.skipped],
            "indicators": population.indicators,
        }
    )
    assert secret not in manifest_text


def test_a_credential_bearing_theme_key_is_refused_without_being_shown(day_db):
    with day_db.admin.connect_writable() as connection:
        connection.execute(
            "UPDATE themes SET theme_key = 'secret=key-secret' "
            "WHERE theme_key = 't-alpha'"
        )
    population = population_of(day_db)
    [skipped] = population.skipped
    assert skipped.reason == review.SKIP_PROVENANCE_CREDENTIAL
    assert "theme_key" in skipped.detail and "key-secret" not in skipped.detail


def test_a_credential_bearing_pipeline_version_argument_is_refused(day_db):
    with pytest.raises(
        ReviewSamplingError, match="pipeline_version carries credential"
    ):
        population_of(day_db, pipeline_version="token=abc")


def test_free_form_context_is_still_redacted_not_refused(tmp_path):
    repository = migrated(tmp_path)
    ids = persist_stories(
        repository, "NVDA", 2, description="Authorization: Bearer s3cret"
    )
    persist_theme_set(repository, "NVDA", themes={"t": ids})
    population = population_of(repository)
    assert population.rows and population.skipped == ()
    assert all("s3cret" not in r.story_description for r in population.rows)


# ----------------------------------------------------------------------
# Micro-review: discovered pipeline versions are checked before they are kept
# ----------------------------------------------------------------------

SECRET_VERSION = "api_key=provenance-secret"


def _no_secret(*texts: str) -> None:
    for text in texts:
        assert "provenance-secret" not in text


def test_a_credential_bearing_discovered_pipeline_version_is_never_retained(
    tmp_path,
):
    """Codex's reproduction, alone: the version is refused at discovery."""

    repository = migrated(tmp_path)
    ids = persist_stories(repository, "NVDA", 3, version=SECRET_VERSION)
    persist_theme_set(repository, "NVDA", themes={"t": ids}, version=SECRET_VERSION)
    population = population_of(repository)
    assert population.rows == ()
    assert population.pipeline_versions == ()
    [skipped] = population.skipped
    assert skipped.reason == review.SKIP_PROVENANCE_CREDENTIAL
    assert skipped.pipeline_version == review.REJECTED_IDENTIFIER
    assert "pipeline_version" in skipped.detail
    _no_secret(
        json.dumps(skipped.as_dict()), json.dumps(list(population.pipeline_versions))
    )
    sample_error = ""
    try:
        sample_assignments(population, seed="s", size=4)
    except ReviewSamplingError as exc:
        sample_error = str(exc)
    _no_secret(sample_error)


def test_a_credential_version_beside_a_normal_partition_does_not_leak_in_the_mixed_path(
    day_db, tmp_path
):
    """Codex's reproduction, mixed: the secret version never reaches the
    mixed-version comparison, its error text, the manifest, or the CSV."""

    amd = persist_stories(day_db, "AMD", 3, version=SECRET_VERSION)
    persist_theme_set(day_db, "AMD", themes={"amd-a": amd}, version=SECRET_VERSION)
    population = population_of(day_db)  # no mixed-version error: v1 is alone
    assert population.pipeline_versions == (VERSION,)
    assert {r.ticker for r in population.rows} == {"NVDA"}
    [skipped] = population.skipped
    assert (skipped.ticker, skipped.reason) == (
        "AMD",
        review.SKIP_PROVENANCE_CREDENTIAL,
    )
    csv_path, manifest_path = write_round(population, tmp_path, size=8)
    _no_secret(csv_path.read_text(), manifest_path.read_text())
    manifest = read_manifest(manifest_path)
    assert (
        manifest["selection"]["skipped_partitions"][0]["pipeline_version"]
        == "<rejected>"
    )


def test_safe_mixed_pipeline_versions_still_require_a_choice(day_db):
    amd = persist_stories(day_db, "AMD", 3, version="v2")
    persist_theme_set(day_db, "AMD", themes={"amd-a": amd}, version="v2")
    with pytest.raises(
        ReviewSamplingError, match=r"span pipeline versions \['v1', 'v2'\]"
    ):
        population_of(day_db)
    assert population_of(day_db, pipeline_version="v2").tickers == ("AMD",)


def test_a_supplied_credential_bearing_pipeline_version_is_still_refused_by_name(
    day_db,
):
    error = ""
    try:
        population_of(day_db, pipeline_version=SECRET_VERSION)
    except ReviewSamplingError as exc:
        error = str(exc)
    assert "pipeline_version carries credential" in error
    _no_secret(error)


# ----------------------------------------------------------------------
# Micro-review: the rejection sentinel is reserved on both boundaries
# ----------------------------------------------------------------------


def test_a_persisted_pipeline_version_equal_to_the_sentinel_is_never_healthy(
    day_db,
):
    amd = persist_stories(day_db, "AMD", 3, version=review.REJECTED_IDENTIFIER)
    persist_theme_set(
        day_db, "AMD", themes={"amd-a": amd}, version=review.REJECTED_IDENTIFIER
    )
    population = population_of(day_db)
    assert {r.ticker for r in population.rows} == {"NVDA"}
    assert population.pipeline_versions == (VERSION,)
    [skipped] = population.skipped
    assert (skipped.ticker, skipped.reason) == ("AMD", review.SKIP_RESERVED_IDENTIFIER)
    assert skipped.reason != review.SKIP_PROVENANCE_CREDENTIAL
    assert "reserved sentinel" in skipped.detail


def test_a_supplied_pipeline_version_equal_to_the_sentinel_is_refused(day_db):
    amd = persist_stories(day_db, "AMD", 3, version=review.REJECTED_IDENTIFIER)
    persist_theme_set(
        day_db, "AMD", themes={"amd-a": amd}, version=review.REJECTED_IDENTIFIER
    )
    with pytest.raises(ReviewSamplingError, match="is reserved"):
        population_of(day_db, pipeline_version=review.REJECTED_IDENTIFIER)
    with pytest.raises(ReviewSamplingError, match="is reserved"):
        population_of(day_db, pipeline_version=f"  {review.REJECTED_IDENTIFIER} ")


def test_the_sentinel_alone_in_a_database_leaves_nothing_reviewable(tmp_path):
    repository = migrated(tmp_path)
    ids = persist_stories(repository, "NVDA", 3, version=review.REJECTED_IDENTIFIER)
    persist_theme_set(
        repository, "NVDA", themes={"t": ids}, version=review.REJECTED_IDENTIFIER
    )
    population = population_of(repository)
    assert population.rows == () and population.pipeline_versions == ()
    assert [s.reason for s in population.skipped] == [review.SKIP_RESERVED_IDENTIFIER]


def test_a_credential_bearing_discovered_version_still_maps_to_the_sentinel(day_db):
    amd = persist_stories(day_db, "AMD", 3, version=SECRET_VERSION)
    persist_theme_set(day_db, "AMD", themes={"amd-a": amd}, version=SECRET_VERSION)
    [skipped] = population_of(day_db).skipped
    assert skipped.reason == review.SKIP_PROVENANCE_CREDENTIAL
    assert skipped.pipeline_version == review.REJECTED_IDENTIFIER
    _no_secret(json.dumps(skipped.as_dict()))


def test_safe_versions_and_safe_mixed_versions_are_unaffected_by_the_reservation(
    day_db,
):
    assert population_of(day_db).pipeline_versions == (VERSION,)
    amd = persist_stories(day_db, "AMD", 3, version="v2")
    persist_theme_set(day_db, "AMD", themes={"amd-a": amd}, version="v2")
    with pytest.raises(
        ReviewSamplingError, match=r"span pipeline versions \['v1', 'v2'\]"
    ):
        population_of(day_db)
    assert population_of(day_db, pipeline_version="v1").tickers == ("NVDA",)
    assert population_of(day_db, pipeline_version="v2").tickers == ("AMD",)

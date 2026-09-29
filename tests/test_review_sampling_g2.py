"""A4b: G2 sentence-faithfulness review sampling and scoring.

Every database here is built through the real Phase 0 write paths -- story
and theme reconciliation, then A3's ``ensure_summary`` with a fake provider
-- and is then only *read* by the code under test.  Nothing here asserts
what "supported" means: that is K3's protocol.
"""

from __future__ import annotations

import csv
import errno
import dataclasses
import hashlib
import itertools
import json
import os
import re
import socket
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ai.guarded_summary as guarded
import ai.summarization as summarization
import nlp.eval.faithfulness as g2
import nlp.eval.review as review
import phase0.summary_lifecycle as lifecycle
from nlp.dedup.selection import cluster_fingerprint_for
from phase0.models import (
    OtherCoverageRecord,
    StoryMemberRecord,
    StoryRecord,
    ThemeRecord,
    ThemeSetRecord,
)
from phase0.repository import Phase0Reader, Phase0Repository
from phase0.summary_lifecycle import SOURCE_GENERATED, ensure_summary
from phase0.summary_runner import PRODUCTION_MAX_ATTEMPTS, production_generation_policy
from tools import make_review_sheets

VERSION = "v1"
D1, D2, D3, D4, D5 = (
    "2026-07-20",
    "2026-07-21",
    "2026-07-22",
    "2026-07-23",
    "2026-07-24",
)
OUTLETS = ("Reuters", "Bloomberg", "CNBC")
GENERATED_AT = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
CODE = {"commit": "test", "dirty": False}
CREDENTIAL = "api_key=abcd1234efgh5678"
ID_LINE_RE = re.compile(r"- id: (\S+)")
_ITEMS = itertools.count(1)
_RUNS = itertools.count(1)


@pytest.fixture(autouse=True)
def no_ambient_provider_config(monkeypatch):
    """The production policy is resolved from GEMINI_*; start from none."""

    for name in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_TIMEOUT_MS",
    ):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------------
# A persisted world
# ----------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 25, 10, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


@dataclasses.dataclass
class World:
    path: Path
    repository: Phase0Repository


@pytest.fixture
def world(tmp_path) -> World:
    repository = Phase0Repository(tmp_path / "phase0.db", clock=Clock())
    repository.migrate()
    return World(tmp_path / "phase0.db", repository)


class Client:
    """A fake provider with the production client's model and output cap.

    Sentence 1 cites every evidence story in *reverse* order, so anything
    that re-sorted citations would be caught; sentence 2 cites the first.
    """

    def __init__(self, *, first_sentence="Coverage leads with this story.") -> None:
        self.model = summarization.DEFAULT_MODEL
        self.max_output_tokens = summarization.DEFAULT_MAX_OUTPUT_TOKENS
        self.first_sentence = first_sentence
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls += 1
        ids = ID_LINE_RE.findall(user_prompt)
        return response_schema.model_validate(
            {
                "label": "Coverage summary",
                "sentences": [
                    {"text": self.first_sentence, "citation_ids": list(reversed(ids))},
                    {"text": "Outlets repeat the report.", "citation_ids": ids[:1]},
                ],
            }
        )


def _insert_item(repository, ticker, day, outlet, stamp):
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
                "published_at": stamp,
                "fetched_at": f"{day}T11:00:00+00:00",
                "raw_json": {"index": index},
            }
        ]
    )
    return result.item_id


def _story(ticker, item_id, title, outlet, stamp):
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
        published_at=stamp,
        canonical_url=f"https://{outlet.lower()}.example/{item_id}",
        content_hash=f"h-{fingerprint[:8]}",
        stage="m3.semantic",
        member_story_keys=(fingerprint,),
        algorithm_version="m3.1",
        config_fingerprint="cfg",
        model_name="fake",
        model_revision="r1",
        embedding_dimension=4,
    )


def seed(world, ticker, day, themes=(2, 1), other=1, generation=0):
    """One partition through the real reconciliation paths; theme ids by rank."""

    repository = world.repository
    groups = [
        [f"{ticker} {day} g{generation} t{n} story {i}" for i in range(count)]
        for n, count in enumerate(themes)
    ]
    others = [f"{ticker} {day} g{generation} other {i}" for i in range(other)]
    records, sequence = [], 0
    for title in [t for group in groups for t in group] + others:
        outlet = OUTLETS[sequence % len(OUTLETS)]
        stamp = f"{day}T10:{sequence:02d}:00+00:00"
        item = _insert_item(repository, ticker, day, outlet, stamp)
        records.append(_story(ticker, item, title, outlet, stamp))
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
    with repository.admin.connect_writable() as connection:
        items = {
            row["id"]: [
                m["raw_item_id"]
                for m in connection.execute(
                    "SELECT raw_item_id FROM story_members WHERE story_id = ? "
                    "ORDER BY position",
                    (row["id"],),
                )
            ]
            for row in rows
        }
    theme_records = []
    for n, group in enumerate(groups):
        members = [by_title[t] for t in group]
        theme_records.append(
            ThemeRecord(
                fingerprint=f"fp-{ticker}-{day}-{generation}-{n}-{members[0]}",
                theme_key=f"key-{ticker}-{day}-{n}",
                label=f"Theme {n}",
                label_source="canonical_story_title",
                story_ids=tuple(members),
                citation_item_ids=tuple(i for s in members for i in items[s]),
                status="ready",
                salience_rank=n + 1,
                story_count=len(members),
            )
        )
    other_ids = [by_title[t] for t in others]
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
                source_metadata={"story_count": len(rows)},
                config_fingerprint="cfg",
                algorithm_version="m5.1",
                model_name="fake",
                model_revision="r1",
                embedding_dimension=4,
            ),
            themes=theme_records,
            other_coverage=[
                OtherCoverageRecord(
                    story_id=story_id, reason="clustering_noise", position=index
                )
                for index, story_id in enumerate(other_ids)
            ],
            excluded=[],
            terminal=True,
        )
    population = repository.read.theme_population(ticker, day, VERSION)
    return [
        t.theme_id for t in sorted(population.themes, key=lambda t: t.salience_rank)
    ]


def summarize(world, ticker, day, theme_id, client=None):
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
            client=client or Client(),
            max_attempts=PRODUCTION_MAX_ATTEMPTS,
        )
    assert outcome.source == SOURCE_GENERATED
    return outcome.artifact


def raw_sql(world, statement, parameters=()):
    """Damage storage the way only a bypass could: triggers off for one statement."""

    with world.repository.admin.connect_writable() as connection:
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


def standard_world(world):
    """D1-D3 summarized, D4 themes without summaries, D5 nothing at all.

    Per summarized day: TSLA has two themes, the first summarized and the
    second degraded; NVDA has one summarized theme; the other tickers hold
    nothing.
    """

    artifacts = {}
    for day in (D1, D2, D3):
        tsla = seed(world, "TSLA", day)
        nvda = seed(world, "NVDA", day, themes=(3,))
        artifacts[day] = [
            summarize(world, "TSLA", day, tsla[0]),
            summarize(world, "NVDA", day, nvda[0]),
        ]
    seed(world, "TSLA", D4)
    return artifacts


def population(world, days=(D1, D2, D3, D4, D5), **kwargs):
    return g2.load_sentence_population(world.path, candidate_days=days, **kwargs)


def sample(world, tmp_path, *, seed_text="s1", name="g2.csv", pop=None, **kwargs):
    pop = pop or population(world)
    drawn = g2.sample_sentences(pop, seed=seed_text, **kwargs)
    manifest = g2.build_manifest(
        drawn, csv_name=name, generated_at=GENERATED_AT, code=CODE
    )
    csv_path, manifest_path = g2.write_sample(drawn, tmp_path / name, manifest=manifest)
    return drawn, csv_path, manifest_path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames=g2.SENTENCE_FIELDNAMES):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return Path(path)


def complete(blank, out, reviewer, verdicts):
    """A reviewer's sheet: ``verdicts`` is a list, one per row, or a callable."""

    rows = read_rows(blank)
    for index, row in enumerate(rows):
        verdict = verdicts(index) if callable(verdicts) else verdicts[index]
        row.update(
            reviewer_id=reviewer if verdict else "",
            reviewed_at="2026-07-26" if verdict else "",
            reviewer_verdict=verdict,
        )
    return write_rows(out, rows)


def rewrite_manifest(path, change, *, rebind=False):
    payload = json.loads(Path(path).read_text())
    change(payload)
    if rebind:
        rebind_manifest(payload)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rebind_manifest(payload):
    """Recompute every *outer* digest, as a careful forger would."""

    snapshot = payload["snapshot"]
    snapshot["sha256"] = g2.snapshot_digest(snapshot["rows"], snapshot["artifacts"])
    payload["population"]["digest"] = g2.population_digest(
        payload["population"]["partitions"], snapshot["artifacts"]
    )
    payload["selection"]["digest"] = g2.selection_digest(payload["selection"])
    payload["sample"]["row_ids"] = [r["row_id"] for r in snapshot["rows"]]
    payload["binding"] = {
        "manifest_id": g2.manifest_identity(payload),
        "snapshot_sha256": snapshot["sha256"],
    }


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_dump(path):
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


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------


def test_candidate_days_are_accounted_and_only_eligible_days_can_be_drawn(world):
    standard_world(world)
    pop = population(world)
    accounting = {d["trading_day"]: d for d in pop.day_accounting()}
    assert pop.eligible_days == [D1, D2, D3]
    assert accounting[D4] == {
        "trading_day": D4,
        "reviewable_artifacts": 0,
        "eligible": False,
        "reason": g2.DAY_NO_CURRENT_SUMMARY,
    }
    assert accounting[D5]["reason"] == g2.DAY_NO_PARTITIONS
    for n in range(40):
        drawn = g2.sample_sentences(pop, seed=f"seed-{n}")
        assert len(drawn.selected_days) == 2
        assert set(drawn.selected_days) <= {D1, D2, D3}


def test_the_draw_is_a_pure_function_of_the_seed(world):
    standard_world(world)
    pop = population(world)
    first = g2.sample_sentences(pop, seed="phase0-g2")
    again = g2.sample_sentences(population(world), seed="phase0-g2")
    assert first.selected_days == again.selected_days
    seen = {g2.sample_sentences(pop, seed=f"k{n}").selected_days for n in range(30)}
    assert len(seen) > 1


def test_manifest_records_the_whole_selection(world, tmp_path):
    standard_world(world)
    drawn, _, manifest_path = sample(world, tmp_path)
    selection = g2.read_manifest(manifest_path)["selection"]
    assert selection["candidate_input"] == {
        "days": [D1, D2, D3, D4, D5],
        "window": None,
    }
    assert selection["candidate_days"] == [D1, D2, D3, D4, D5]
    assert selection["eligible_days"] == [D1, D2, D3]
    assert selection["excluded_days"] == [
        {"trading_day": D4, "reason": g2.DAY_NO_CURRENT_SUMMARY},
        {"trading_day": D5, "reason": g2.DAY_NO_PARTITIONS},
    ]
    assert selection["draw"] == {
        "method": g2.DRAW_METHOD,
        "seed": "s1",
        "size": 2,
        "required_days": 2,
        "development_override": False,
    }
    assert selection["selected_days"] == list(drawn.selected_days)


def test_a_window_expands_to_calendar_days(world, tmp_path):
    standard_world(world)
    pop = g2.load_sentence_population(world.path, window=(D1, D5))
    assert pop.candidate_days == (D1, D2, D3, D4, D5)
    drawn = g2.sample_sentences(pop, seed="w")
    manifest = g2.build_manifest(drawn, csv_name="w.csv", code=CODE)
    g2.write_sample(drawn, tmp_path / "w.csv", manifest=manifest)
    read = g2.read_manifest(tmp_path / "w.manifest.json")
    assert read["selection"]["candidate_input"]["window"] == {"start": D1, "end": D5}


def test_fewer_than_two_eligible_days_is_refused(world):
    standard_world(world)
    pop = population(world, days=(D1, D4, D5))
    with pytest.raises(review.ReviewSamplingError, match="1 eligible day"):
        g2.sample_sentences(pop, seed="s")


def test_a_development_draw_size_is_recorded_and_ineligible(world, tmp_path):
    standard_world(world)
    drawn, csv_path, manifest_path = sample(world, tmp_path, draw_size=1)
    assert len(drawn.selected_days) == 1
    manifest = g2.read_manifest(manifest_path)
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    card = g2.score_g2(g2.score_sentence_round(manifest, [a]))
    assert card.evaluation_mode == "development"
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE
    assert any("exactly 2" in b for b in card.eligibility_blockers)


# ----------------------------------------------------------------------
# Census
# ----------------------------------------------------------------------


def test_every_sentence_of_every_current_artifact_is_one_row(world, tmp_path):
    artifacts = standard_world(world)
    drawn, csv_path, _ = sample(world, tmp_path)
    expected = [
        (a.artifact_id, s.ordinal)
        for day in drawn.selected_days
        for a in artifacts[day]
        for s in a.sentences
    ]
    rows = read_rows(csv_path)
    assert sorted(
        (int(r["artifact_id"]), int(r["sentence_ordinal"])) for r in rows
    ) == (sorted(expected))
    assert len({r["row_id"] for r in rows}) == len(rows)
    assert all(r["row_id"].startswith("g2-") for r in rows)


def test_every_ticker_is_accounted_on_every_candidate_day(world, tmp_path):
    standard_world(world)
    _, _, manifest_path = sample(world, tmp_path)
    partitions = g2.read_manifest(manifest_path)["population"]["partitions"]
    assert [(p["trading_day"], p["ticker"]) for p in partitions] == [
        (day, ticker)
        for day in (D1, D2, D3, D4, D5)
        for ticker in ("TSLA", "NVDA", "AMD", "AAPL", "META")
    ]
    amd = next(p for p in partitions if (p["trading_day"], p["ticker"]) == (D1, "AMD"))
    assert amd["outcome"] == review.SKIP_NO_STORY_OUTPUT


def test_degraded_themes_are_counted_not_reviewed(world, tmp_path):
    standard_world(world)
    drawn, csv_path, manifest_path = sample(world, tmp_path)
    manifest = g2.read_manifest(manifest_path)
    tsla = next(
        p
        for p in manifest["population"]["partitions"]
        if p["ticker"] == "TSLA" and p["trading_day"] == drawn.selected_days[0]
    )
    assert [t["outcome"] for t in tsla["themes"]] == [
        g2.THEME_CURRENT,
        g2.THEME_NO_CURRENT,
    ]
    assert tsla["degraded_theme_count"] == 1
    assert manifest["population"]["selected_days"]["degraded_theme_count"] == 2
    degraded_id = str(tsla["themes"][1]["theme_id"])
    assert all(r["theme_id"] != degraded_id for r in read_rows(csv_path))


def test_rows_carry_the_exact_persisted_artifact_identity(world, tmp_path):
    artifacts = standard_world(world)
    drawn, csv_path, _ = sample(world, tmp_path)
    stored = {
        a.artifact_id: a
        for day in drawn.selected_days
        for a in Phase0Reader(world.path).summary_artifacts("TSLA", day, VERSION)
        + Phase0Reader(world.path).summary_artifacts("NVDA", day, VERSION)
    }
    assert set(stored) == {
        a.artifact_id for d in drawn.selected_days for a in artifacts[d]
    }
    for row in read_rows(csv_path):
        artifact = stored[int(row["artifact_id"])]
        sentence = artifact.sentences[int(row["sentence_ordinal"]) - 1]
        assert row["theme_id"] == str(artifact.theme_id)
        assert row["theme_key"] == artifact.theme_key
        assert row["input_fingerprint"] == artifact.input_fingerprint
        assert row["policy_fingerprint"] == artifact.policy_fingerprint
        assert row["content_digest"] == artifact.content_digest
        assert row["sentence_text"] == sentence.text
        assert row["summary_label"] == artifact.label
        assert row["sentence_count"] == str(len(artifact.sentences))


def test_multiple_citations_keep_their_persisted_order(world, tmp_path):
    standard_world(world)
    _, csv_path, _ = sample(world, tmp_path)
    nvda = next(
        r
        for r in read_rows(csv_path)
        if r["ticker"] == "NVDA" and r["sentence_ordinal"] == "1"
    )
    ids = nvda["citation_story_ids"].split(" | ")
    assert len(ids) == 3
    numbers = [int(i.split(":")[1]) for i in ids]
    assert numbers == sorted(numbers, reverse=True)
    positions = re.findall(r"^\[(\d+)\] (story:\d+)", nvda["cited_evidence"], re.M)
    assert [p for p, _ in positions] == ["0", "1", "2"]
    assert [s for _, s in positions] == ids


def test_frozen_evidence_is_what_the_model_saw(world, tmp_path):
    standard_world(world)
    drawn, csv_path, manifest_path = sample(world, tmp_path)
    manifest = g2.read_manifest(manifest_path)
    policy = production_generation_policy()
    for artifact in manifest["snapshot"]["artifacts"]:
        current = lifecycle.current_summary_artifact(
            Phase0Reader(world.path),
            artifact["ticker"],
            artifact["trading_day"],
            VERSION,
            artifact["theme_id"],
            policy,
        )
        assert [
            {
                k: e[k]
                for k in (
                    "citation_id",
                    "title",
                    "description",
                    "outlet",
                    "published_at",
                )
            }
            for e in artifact["evidence"]
        ] == [s.model_visible() for s in current.generation_input.evidence]
    row = read_rows(csv_path)[0]
    assert (
        "Title: " in row["cited_evidence"] and "Description: " in row["cited_evidence"]
    )


# ----------------------------------------------------------------------
# Currentness
# ----------------------------------------------------------------------


def test_a_superseded_artifact_is_not_reviewed(world):
    first = standard_world(world)
    old = first[D1][0]
    reseeded = seed(world, "TSLA", D1, themes=(2, 1), generation=1)
    pop = population(world)
    ids = {a["artifact_id"] for a in pop.artifacts}
    assert old.artifact_id not in ids
    new = summarize(world, "TSLA", D1, reseeded[0])
    assert new.artifact_id in {a["artifact_id"] for a in population(world).artifacts}


def test_a_policy_change_after_generation_leaves_nothing_current(world):
    standard_world(world)
    other = guarded.resolve_generation_policy(
        type("C", (), {"model": "gemini-other", "max_output_tokens": 1024})(),
        max_attempts=PRODUCTION_MAX_ATTEMPTS,
    )
    pop = population(world, policy_resolver=lambda: other)
    assert pop.artifacts == ()
    assert pop.eligible_days == []
    assert pop.policy["fingerprint"] == other.fingerprint
    with pytest.raises(review.ReviewSamplingError, match="0 eligible"):
        g2.sample_sentences(pop, seed="s")


def test_the_same_model_env_change_is_seen_through_the_default_policy(
    world, monkeypatch
):
    standard_world(world)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-other")
    assert population(world).artifacts == ()


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM summary_sentences WHERE artifact_id = ? AND ordinal = 2",
        "UPDATE summary_sentences SET text = 'Edited after the fact.' "
        "WHERE artifact_id = ? AND ordinal = 1",
    ],
    ids=["missing-sentence", "digest-mismatch"],
)
def test_a_corrupt_artifact_is_not_reviewed(world, statement):
    artifacts = standard_world(world)
    bad = artifacts[D1][0]
    raw_sql(world, statement, (bad.artifact_id,))
    pop = population(world)
    assert bad.artifact_id not in {a["artifact_id"] for a in pop.artifacts}
    tsla = next(
        p for p in pop.partitions if (p["trading_day"], p["ticker"]) == (D1, "TSLA")
    )
    assert tsla["themes"][0]["outcome"] == g2.THEME_NO_CURRENT


def test_an_artifact_current_for_another_input_skips_the_partition(world, monkeypatch):
    standard_world(world)
    reader = Phase0Reader(world.path)
    tsla_pop = reader.theme_population("TSLA", D1, VERSION)
    from phase0.summaries import build_generation_input

    other_input = build_generation_input(
        tsla_pop, sorted(tsla_pop.themes, key=lambda t: t.salience_rank)[1].theme_id
    )
    real = g2.current_summary_artifact

    def drifted(reader, ticker, day, *args, **kwargs):
        found = real(reader, ticker, day, *args, **kwargs)
        if found is None or (ticker, day) != ("TSLA", D1):
            return found
        return dataclasses.replace(found, generation_input=other_input)

    monkeypatch.setattr(g2, "current_summary_artifact", drifted)
    pop = population(world)
    tsla = next(
        p for p in pop.partitions if (p["trading_day"], p["ticker"]) == (D1, "TSLA")
    )
    assert tsla["outcome"] == g2.PARTITION_POPULATION_CHANGED
    assert tsla["reviewed_artifact_ids"] == [] and tsla["themes"] == []
    assert all((a["trading_day"], a["ticker"]) != (D1, "TSLA") for a in pop.artifacts)


def test_a_population_that_moves_between_reads_is_skipped(world, monkeypatch, tmp_path):
    standard_world(world)
    real = Phase0Reader.theme_population
    reads: dict[tuple, int] = {}

    def moving(self, ticker, day, version):
        found = real(self, ticker, day, version)
        key = (ticker, str(day))
        reads[key] = reads.get(key, 0) + 1
        if key == ("NVDA", D2) and reads[key] > 1:
            return dataclasses.replace(found, themes=())
        return found

    monkeypatch.setattr(Phase0Reader, "theme_population", moving)
    pop = population(world)
    nvda = next(
        p for p in pop.partitions if (p["trading_day"], p["ticker"]) == (D2, "NVDA")
    )
    assert nvda["outcome"] == g2.PARTITION_POPULATION_CHANGED
    # A selected day with a skipped partition is an incomplete census.
    drawn = next(
        d
        for n in range(50)
        if D2 in (d := g2.sample_sentences(pop, seed=f"m{n}")).selected_days
    )
    manifest = g2.build_manifest(drawn, csv_name="m.csv", code=CODE)
    csv_path, manifest_path = g2.write_sample(
        drawn, tmp_path / "m.csv", manifest=manifest
    )
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    card = g2.score_g2(g2.score_sentence_round(g2.read_manifest(manifest_path), [a]))
    assert card.review_complete is False
    assert any("changed while sampling" in i for i in card.incompleteness)


# ----------------------------------------------------------------------
# Security
# ----------------------------------------------------------------------


def test_a_credential_like_sentence_withholds_its_artifact(world, tmp_path):
    standard_world(world)
    ids = seed(world, "AMD", D1, themes=(1,))
    summarize(
        world,
        "AMD",
        D1,
        ids[0],
        client=Client(first_sentence=f"A filing shows {CREDENTIAL}."),
    )
    pop = population(world)
    amd = next(
        p for p in pop.partitions if (p["trading_day"], p["ticker"]) == (D1, "AMD")
    )
    assert amd["themes"][0]["outcome"] == g2.THEME_WITHHELD
    assert amd["themes"][0]["reason"] == g2.WITHHELD_SENTENCE
    assert amd["themes"][0]["field"] == "sentence 1"
    drawn = next(
        d
        for n in range(50)
        if D1 in (d := g2.sample_sentences(pop, seed=f"c{n}")).selected_days
    )
    manifest = g2.build_manifest(drawn, csv_name="c.csv", code=CODE)
    csv_path, manifest_path = g2.write_sample(
        drawn, tmp_path / "c.csv", manifest=manifest
    )
    for path in (csv_path, manifest_path):
        assert CREDENTIAL not in path.read_text()
        assert "abcd1234efgh5678" not in path.read_text()
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    card = g2.score_g2(g2.score_sentence_round(g2.read_manifest(manifest_path), [a]))
    assert any("withheld" in i for i in card.incompleteness)


def test_the_manifest_records_only_the_database_basename(world, tmp_path):
    standard_world(world)
    _, _, manifest_path = sample(world, tmp_path)
    text = manifest_path.read_text()
    assert json.loads(text)["source"] == {"mode": "persisted", "database": "phase0.db"}
    assert str(world.path.parent) not in text


# ----------------------------------------------------------------------
# Tamper detection
# ----------------------------------------------------------------------


@pytest.fixture
def sampled(world, tmp_path):
    standard_world(world)
    return sample(world, tmp_path)


def _first_artifact(payload):
    return payload["snapshot"]["artifacts"][0]


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda p: _first_artifact(p)["sentences"][0].update(text="Altered."),
            "content digest",
        ),
        (
            lambda p: _first_artifact(p)["sentences"][0]["citations"].reverse(),
            "content digest",
        ),
        (
            lambda p: _first_artifact(p)["sentences"][0]["citations"][0].update(
                story_id=999999
            ),
            "content digest",
        ),
        (
            lambda p: _first_artifact(p)["evidence"][0].update(description="Invented."),
            "input fingerprint",
        ),
        (
            lambda p: _first_artifact(p)["evidence"][0].update(title="Invented."),
            "input fingerprint",
        ),
    ],
    ids=[
        "sentence",
        "citation-order",
        "citation-id",
        "evidence-description",
        "evidence-title",
    ],
)
def test_content_tampering_is_refused_even_with_every_outer_digest_recomputed(
    sampled, change, message
):
    _, _, manifest_path = sampled
    rewrite_manifest(manifest_path, change, rebind=True)
    with pytest.raises(review.ReviewSamplingError, match=message):
        g2.read_manifest(manifest_path)


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["snapshot"]["rows"][0].update(sentence_text="Altered."),
        lambda p: p["snapshot"]["rows"][0].update(artifact_id="424242"),
        lambda p: p["snapshot"]["rows"].pop(),
        lambda p: p["snapshot"]["rows"].append(dict(p["snapshot"]["rows"][0])),
        lambda p: _first_artifact(p)["evidence"][0]["urls"].insert(
            0, "https://x.example"
        ),
    ],
    ids=[
        "row-context",
        "row-identity",
        "row-removed",
        "row-duplicated",
        "evidence-url",
    ],
)
def test_snapshot_tampering_is_refused(sampled, change):
    _, _, manifest_path = sampled
    rewrite_manifest(manifest_path, change)
    with pytest.raises(review.ReviewSamplingError):
        g2.read_manifest(manifest_path)


def test_row_tampering_is_refused_even_when_rebound(sampled):
    _, _, manifest_path = sampled

    def change(payload):
        payload["snapshot"]["rows"][0]["sentence_text"] = "Altered."

    rewrite_manifest(manifest_path, change, rebind=True)
    with pytest.raises(review.ReviewSamplingError, match="exact sentence census"):
        g2.read_manifest(manifest_path)


def _other_seed(payload):
    """A seed whose draw differs from the recorded selection."""

    selection = payload["selection"]
    return next(
        seed
        for seed in (f"other-{n}" for n in range(100))
        if g2.draw_days(selection["eligible_days"], seed=seed, size=2)
        != selection["selected_days"]
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda p: p["selection"].update(selected_days=["2026-07-20", "2026-07-23"]),
            "seeded draw",
        ),
        (lambda p: p["selection"]["draw"].update(seed=_other_seed(p)), "seeded draw"),
        (
            lambda p: p["selection"].update(
                eligible_days=p["selection"]["eligible_days"] + ["2026-07-23"]
            ),
            "eligible days",
        ),
        (lambda p: p["population"]["partitions"].pop(), "day accounting|partitions"),
    ],
    ids=["hand-picked-days", "seed", "eligible-set", "partition-removed"],
)
def test_selection_tampering_is_refused_even_when_rebound(sampled, change, message):
    _, _, manifest_path = sampled
    rewrite_manifest(manifest_path, change, rebind=True)
    with pytest.raises(review.ReviewSamplingError, match=message):
        g2.read_manifest(manifest_path)


def test_an_edited_manifest_breaks_its_binding(sampled):
    _, _, manifest_path = sampled
    rewrite_manifest(manifest_path, lambda p: p.update(claim="everything was served"))
    with pytest.raises(review.ReviewSamplingError, match="binding"):
        g2.read_manifest(manifest_path)


def _tamper_sheet(blank, out, change):
    rows = read_rows(blank)
    for row in rows:
        row.update(reviewer_id="alice", reviewer_verdict="supported")
    rows = change(rows)
    return write_rows(out, rows)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda rows: [dict(rows[0], sentence_text="Edited.")] + rows[1:], "altered"),
        (
            lambda rows: [dict(rows[0], cited_evidence=rows[0]["cited_evidence"] + "!")]
            + rows[1:],
            "altered",
        ),
        (
            lambda rows: [dict(rows[0], citation_story_ids="story:1")] + rows[1:],
            "altered",
        ),
        (lambda rows: rows[1:], "missing manifest rows"),
        (lambda rows: rows + [rows[0]], "duplicate row_id"),
        (
            lambda rows: rows + [dict(rows[0], row_id="g2-" + "0" * 64)],
            "not in the manifest",
        ),
        (
            lambda rows: [dict(rows[0], reviewer_verdict="correct")] + rows[1:],
            "verdict",
        ),
    ],
    ids=[
        "sentence",
        "evidence",
        "citation",
        "missing-row",
        "duplicate-row",
        "added-row",
        "g1-vocabulary",
    ],
)
def test_completed_sheet_tampering_is_refused(sampled, tmp_path, change, message):
    _, csv_path, manifest_path = sampled
    sheet = _tamper_sheet(csv_path, tmp_path / "t.csv", change)
    with pytest.raises(review.ReviewSamplingError, match=message):
        g2.score_sentence_round(g2.read_manifest(manifest_path), [sheet])


def test_a_sheet_cannot_be_scored_against_another_manifest(world, tmp_path):
    standard_world(world)
    _, csv_a, _ = sample(world, tmp_path / "a", seed_text="x1")
    for n in range(2, 50):
        drawn_b, _, manifest_b = sample(world, tmp_path / f"b{n}", seed_text=f"x{n}")
        if (
            drawn_b.selected_days
            != g2.read_manifest(tmp_path / "a" / "g2.manifest.json")["selection"][
                "selected_days"
            ]
        ):
            break
    sheet = complete(csv_a, tmp_path / "s.csv", "alice", lambda i: "supported")
    with pytest.raises(review.ReviewSamplingError):
        g2.score_sentence_round(g2.read_manifest(manifest_b), [sheet])


def test_g1_and_g2_manifests_never_cross(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    with pytest.raises(review.ReviewSamplingError, match="a4a-review-sample"):
        review.read_manifest(manifest_path)
    g1_population = review.load_fixture_population()
    g1_sample = review.sample_assignments(g1_population, seed="g1", size=5)
    g1_manifest = review.build_manifest(g1_sample, csv_name="g1.csv", code=CODE)
    _, g1_path = review.write_sample(
        g1_sample, tmp_path / "g1.csv", manifest=g1_manifest
    )
    with pytest.raises(review.ReviewSamplingError, match=g2.MANIFEST_SCHEMA):
        g2.read_manifest(g1_path)
    with pytest.raises(review.ReviewSamplingError, match="G2 round"):
        g2.score_sentence_round(review.read_manifest(g1_path), [csv_path])
    sheet = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    result = g2.score_sentence_round(g2.read_manifest(manifest_path), [sheet])
    with pytest.raises(review.ReviewSamplingError):
        review.score_gate([result])


def test_a_g1_protocol_is_never_a_g2_protocol(sampled, monkeypatch):
    k3_g1 = review.Protocol(
        id="k3-g1-v1",
        positive_verdict="correct",
        negative_verdict="incorrect",
        adjudicated_states=frozenset({review.AdjudicationState.UNANIMOUS}),
    )
    monkeypatch.setattr(review, "RATIFIED_PROTOCOLS", {"k3-g1-v1": k3_g1})
    assert review.resolve_protocol("k3-g1-v1") == (k3_g1, True)
    assert g2.resolve_g2_protocol("k3-g1-v1") == (g2.PROVISIONAL_G2_PROTOCOL, False)
    with pytest.raises(
        review.ReviewSamplingError, match="unknown G2 labeling protocol"
    ):
        g2.require_known_g2_protocol("k3-g1-v1")


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def _scored(sampled, tmp_path, a, b=None, adjudication=None):
    _, csv_path, manifest_path = sampled
    sheets = [complete(csv_path, tmp_path / "alice.csv", "alice", a)]
    if b is not None:
        sheets.append(complete(csv_path, tmp_path / "bob.csv", "bob", b))
    manifest = g2.read_manifest(manifest_path)
    result = g2.score_sentence_round(manifest, sheets, adjudicated=adjudication)
    return result, g2.score_g2(result)


def test_two_agreeing_reviewers_measure_a_rate_that_is_not_a_verdict(sampled, tmp_path):
    result, card = _scored(
        sampled, tmp_path, lambda i: "supported", lambda i: "supported"
    )
    assert result.adjudication_state is review.AdjudicationState.UNANIMOUS
    assert card.rate == 1.0 and card.threshold_met is True
    assert card.review_complete is True
    assert card.gate_eligible is False
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE
    blockers = " ".join(card.eligibility_blockers)
    assert "origin is unverified" in blockers
    assert "not in the ratified G2 registry" in blockers
    assert card.origin_status is review.OriginStatus.UNVERIFIED
    text = g2.render_scorecard(card)
    assert "gate_result        NOT_ELIGIBLE" in text
    assert "a measurement, not a verdict" in text


def test_an_unadjudicated_disagreement_leaves_the_review_incomplete(sampled, tmp_path):
    result, card = _scored(
        sampled,
        tmp_path,
        lambda i: "supported",
        lambda i: "unsupported" if i == 0 else "supported",
    )
    assert result.adjudication_state is review.AdjudicationState.OPEN
    assert card.unresolved_count == 1 and card.review_complete is False
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE


def test_adjudication_resolves_a_disagreement(sampled, tmp_path):
    _, csv_path, _ = sampled
    first = read_rows(csv_path)[0]["row_id"]
    adjudication = tmp_path / "adj.csv"
    with adjudication.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.ADJUDICATION_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "row_id": first,
                "final_verdict": "unsupported",
                "adjudicator_id": "carol",
                "adjudicated_at": "2026-07-27",
                "adjudication_notes": "",
            }
        )
    result, card = _scored(
        sampled,
        tmp_path,
        lambda i: "supported",
        lambda i: "unsupported" if i == 0 else "supported",
        adjudication=adjudication,
    )
    assert result.adjudication_state is review.AdjudicationState.RESOLVED
    assert card.review_complete is True and card.unresolved_count == 0
    assert card.positive_count == card.sentence_count - 1
    assert card.adjudicator_ids == ("carol",)


def test_a_blank_verdict_is_unresolved(sampled, tmp_path):
    _, card = _scored(sampled, tmp_path, lambda i: "" if i == 0 else "supported")
    assert card.review_complete is False
    assert any("unresolved" in i for i in card.incompleteness)
    assert any("fewer than two reviewers" in b for b in card.eligibility_blockers)


@pytest.mark.parametrize(
    "positive, resolved, met",
    [(19, 20, True), (18, 20, False), (95, 100, True), (94, 100, False), (0, 0, None)],
)
def test_the_threshold_is_compared_exactly(positive, resolved, met):
    rate, threshold_met = g2.faithfulness_rate(
        positive, resolved, g2.RELEASE_G2_THRESHOLD
    )
    assert threshold_met is met
    if resolved:
        assert rate == positive / resolved


def _twenty_sentence_world(world):
    """Exactly two eligible days, ten sentences each: five two-sentence summaries."""

    for day in (D1, D2):
        for ticker, count in (("TSLA", 3), ("NVDA", 2)):
            ids = seed(world, ticker, day, themes=(1,) * count, other=0)
            for theme_id in ids:
                summarize(world, ticker, day, theme_id)


def test_nineteen_of_twenty_meets_the_threshold_and_is_still_not_a_pass(
    world, tmp_path
):
    _twenty_sentence_world(world)
    _, csv_path, manifest_path = sample(
        world, tmp_path, pop=population(world, days=(D1, D2))
    )
    verdicts = lambda i: "unsupported" if i == 0 else "supported"  # noqa: E731
    a = complete(csv_path, tmp_path / "a.csv", "alice", verdicts)
    b = complete(csv_path, tmp_path / "b.csv", "bob", verdicts)
    card = g2.score_g2(g2.score_sentence_round(g2.read_manifest(manifest_path), [a, b]))
    assert card.sentence_count == 20 and card.positive_count == 19
    assert card.rate == 0.95 and card.threshold_met is True
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE


@pytest.fixture
def eligible_machinery(monkeypatch):
    """What a later reviewed change would have to supply, stood in for.

    Nothing produces verified origin or build binding today, and no G2
    protocol is ratified; these patches show the precedence would work once
    all three exist, and that each alone still blocks.
    """

    protocol = review.Protocol(
        id="k3-g2-test",
        positive_verdict="supported",
        negative_verdict="unsupported",
        adjudicated_states=frozenset({review.AdjudicationState.UNANIMOUS}),
    )
    monkeypatch.setattr(g2, "RATIFIED_G2_PROTOCOLS", {"k3-g2-test": protocol})
    monkeypatch.setattr(
        g2,
        "classify_generation_binding",
        lambda population: (review.GENERATION_BINDING_VERIFIED, "sig"),
    )
    return monkeypatch


def _twenty(world, tmp_path, protocol, verdicts):
    _twenty_sentence_world(world)
    drawn = g2.sample_sentences(population(world, days=(D1, D2)), seed="t")
    manifest = g2.build_manifest(
        drawn, csv_name="t.csv", protocol_id=protocol, code=CODE
    )
    csv_path, manifest_path = g2.write_sample(
        drawn, tmp_path / "t.csv", manifest=manifest
    )
    a = complete(csv_path, tmp_path / "a.csv", "alice", verdicts)
    b = complete(csv_path, tmp_path / "b.csv", "bob", verdicts)
    return g2.score_sentence_round(g2.read_manifest(manifest_path), [a, b])


def test_unverified_origin_alone_prevents_eligibility(
    world, tmp_path, eligible_machinery
):
    card = g2.score_g2(_twenty(world, tmp_path, "k3-g2-test", lambda i: "supported"))
    assert card.eligibility_blockers == (
        f"origin is unverified: {review.UNVERIFIED_DETAIL}",
    )
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE


def test_an_unratified_protocol_alone_prevents_eligibility(
    world, tmp_path, eligible_machinery
):
    eligible_machinery.setattr(
        g2, "classify_origin", lambda source: (review.OriginStatus.VERIFIED_LIVE, "t")
    )
    card = g2.score_g2(_twenty(world, tmp_path, "unratified", lambda i: "supported"))
    # Unratified: the protocol is refused, and so is its adjudication state,
    # because an unratified protocol counts nothing as adjudicated.
    assert len(card.eligibility_blockers) == 2
    assert "not in the ratified G2 registry" in card.eligibility_blockers[0]
    assert "not one the protocol counts" in card.eligibility_blockers[1]
    assert card.threshold_met is True
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE


@pytest.mark.parametrize(
    "unsupported, result",
    [(1, review.GateResult.PASS), (2, review.GateResult.FAIL)],
)
def test_the_precedence_reaches_pass_or_fail_only_when_everything_holds(
    world, tmp_path, eligible_machinery, unsupported, result
):
    eligible_machinery.setattr(
        g2, "classify_origin", lambda source: (review.OriginStatus.VERIFIED_LIVE, "t")
    )
    card = g2.score_g2(
        _twenty(
            world,
            tmp_path,
            "k3-g2-test",
            lambda i: "unsupported" if i < unsupported else "supported",
        )
    )
    assert card.eligibility_blockers == ()
    assert card.gate_result is result


def test_a_development_threshold_is_never_eligible(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    result = g2.score_sentence_round(g2.read_manifest(manifest_path), [a])
    card = g2.score_g2(result, development=g2.G2DevelopmentOverrides(threshold=0.5))
    assert card.evaluation_mode == "development"
    assert any("development threshold" in b for b in card.eligibility_blockers)


def test_a_hand_edited_round_report_is_refused(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    result = g2.score_sentence_round(g2.read_manifest(manifest_path), [a])
    forged = dataclasses.replace(result, reviewer_ids=("alice", "bob"))
    with pytest.raises(review.ReviewSamplingError, match="does not match"):
        g2.score_g2(forged)


# ----------------------------------------------------------------------
# Isolation
# ----------------------------------------------------------------------


@pytest.fixture
def no_provider_no_network(monkeypatch):
    """Arm after seeding: from then on, any generation, provider or socket fails."""

    def refuse(*args, **kwargs):
        raise AssertionError("A4b must not generate, call a provider, or connect")

    def arm():
        monkeypatch.setattr(summarization.GeminiClient, "generate", refuse)
        monkeypatch.setattr(summarization.GeminiClient, "_get_client", refuse)
        monkeypatch.setattr(guarded, "generate_guarded_summary", refuse)
        monkeypatch.setattr(lifecycle, "generate_guarded_summary", refuse)
        monkeypatch.setattr(lifecycle, "ensure_summary", refuse)
        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)

    return arm


def test_sampling_and_scoring_need_no_key_no_provider_and_no_network(
    world, tmp_path, no_provider_no_network
):
    standard_world(world)
    before_sha = file_sha(world.path)
    before_rows = table_dump(world.path)
    before_entries = set(os.listdir(world.path.parent))
    no_provider_no_network()
    assert "GEMINI_API_KEY" not in os.environ

    _, csv_path, manifest_path = sample(world, tmp_path)
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    card = g2.score_g2(g2.score_sentence_round(g2.read_manifest(manifest_path), [a]))
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE

    assert file_sha(world.path) == before_sha
    assert table_dump(world.path) == before_rows
    # Read-only access, not zero filesystem activity: in a writable directory
    # SQLite may create its WAL coordination files, and nothing else appears
    # beside the database.
    created = set(os.listdir(world.path.parent)) - before_entries
    created -= {"g2.csv", "g2.manifest.json", "a.csv"}
    assert created <= {"phase0.db-wal", "phase0.db-shm"}


def test_scoring_never_consults_the_database(world, tmp_path):
    standard_world(world)
    _, csv_path, manifest_path = sample(world, tmp_path)
    for suffix in ("", "-wal", "-shm"):
        Path(f"{world.path}{suffix}").unlink(missing_ok=True)
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    card = g2.score_g2(g2.score_sentence_round(g2.read_manifest(manifest_path), [a]))
    assert card.sentence_count == len(read_rows(csv_path))


def test_resampling_an_unchanged_database_is_byte_identical(world, tmp_path):
    standard_world(world)
    _, csv_a, manifest_a = sample(world, tmp_path / "a")
    _, csv_b, manifest_b = sample(world, tmp_path / "b")
    assert csv_a.read_bytes() == csv_b.read_bytes()
    assert manifest_a.read_bytes() == manifest_b.read_bytes()


# ----------------------------------------------------------------------
# The CLI
# ----------------------------------------------------------------------


def test_the_cli_samples_and_scores_without_calling_a_verdict_a_pass(
    world, tmp_path, capsys
):
    _twenty_sentence_world(world)
    out = tmp_path / "cli" / "g2.csv"
    code = make_review_sheets.main(
        [
            "sample-sentences",
            "--database",
            str(world.path),
            "--window-start",
            D1,
            "--window-end",
            D3,
            "--seed",
            "cli",
            "--out",
            str(out),
        ]
    )
    assert code == make_review_sheets.EXIT_PASS
    assert "excluded 2026-07-22: no_partitions" in capsys.readouterr().err
    a = complete(out, tmp_path / "a.csv", "alice", lambda i: "supported")
    b = complete(out, tmp_path / "b.csv", "bob", lambda i: "supported")
    report = tmp_path / "cli" / "scorecard.json"
    code = make_review_sheets.main(
        [
            "score-sentences",
            "--round",
            str(out.with_name("g2.manifest.json")),
            str(a),
            str(b),
            "--report",
            str(report),
        ]
    )
    printed = capsys.readouterr().out
    assert code == make_review_sheets.EXIT_NO_VERDICT
    assert "gate_result        NOT_ELIGIBLE" in printed
    assert "measured rate      1.0000" in printed
    payload = json.loads(report.read_text())
    assert payload["gate"] == "G2" and payload["gate_result"] == "NOT_ELIGIBLE"
    assert payload["threshold_met"] is True and payload["gate_eligible"] is False


def test_the_cli_refuses_a_g1_manifest_for_g2_scoring(world, tmp_path, capsys):
    g1_sample = review.sample_assignments(
        review.load_fixture_population(), seed="g1", size=3
    )
    manifest = review.build_manifest(g1_sample, csv_name="g1.csv", code=CODE)
    csv_path, manifest_path = review.write_sample(
        g1_sample, tmp_path / "g1.csv", manifest=manifest
    )
    code = make_review_sheets.main(
        ["score-sentences", "--round", str(manifest_path), str(csv_path)]
    )
    assert code == make_review_sheets.EXIT_USAGE
    assert g2.MANIFEST_SCHEMA in capsys.readouterr().err


# ======================================================================
# Review repairs: adversarial regressions
# ======================================================================


def _raw_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _write_raw(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    return Path(path)


def _prepend_column(blank, out, name, value_for, fill=None):
    """A sheet with ``name`` repeated as a new first column (the P1 attack)."""

    header, *body = _raw_rows(blank)
    verdict = header.index("reviewer_verdict")
    reviewer = header.index("reviewer_id")
    rows = [[name] + header]
    for index, row in enumerate(body):
        if fill:
            row[verdict], row[reviewer] = fill, "alice"
        rows.append([value_for(index, row)] + row)
    return _write_raw(out, rows)


# -- P1: duplicate headers ---------------------------------------------------


def test_a_duplicate_sentence_text_column_is_refused(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    sheet = _prepend_column(
        csv_path,
        tmp_path / "dup.csv",
        "sentence_text",
        lambda i, r: "Invented.",
        fill="supported",
    )
    with pytest.raises(
        review.ReviewSamplingError, match="more than once.*sentence_text"
    ):
        g2.score_sentence_round(g2.read_manifest(manifest_path), [sheet])


def test_a_duplicate_verdict_column_is_refused(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    sheet = _prepend_column(
        csv_path,
        tmp_path / "dup.csv",
        "reviewer_verdict",
        lambda i, r: "unsupported",
        fill="supported",
    )
    with pytest.raises(review.ReviewSamplingError, match="reviewer_verdict"):
        g2.score_sentence_round(g2.read_manifest(manifest_path), [sheet])


def _g1_round(tmp_path):
    g1_sample = review.sample_assignments(
        review.load_fixture_population(), seed="g1", size=5
    )
    manifest = review.build_manifest(g1_sample, csv_name="g1.csv", code=CODE)
    return review.write_sample(g1_sample, tmp_path / "g1.csv", manifest=manifest)


@pytest.mark.parametrize("column", ["story_title", "reviewer_verdict", "manifest_id"])
def test_a_duplicate_g1_column_is_refused(tmp_path, column):
    csv_path, manifest_path = _g1_round(tmp_path)
    sheet = _prepend_column(
        csv_path, tmp_path / "dup.csv", column, lambda i, r: "correct", fill="correct"
    )
    with pytest.raises(review.ReviewSamplingError, match=f"more than once.*{column}"):
        review.score_round(review.read_manifest(manifest_path), [sheet])


def test_a_valid_g1_sheet_still_scores(tmp_path):
    csv_path, manifest_path = _g1_round(tmp_path)
    rows = read_rows(csv_path)
    for row in rows:
        row.update(reviewer_id="alice", reviewer_verdict="correct")
    sheet = write_rows(
        tmp_path / "ok.csv", rows, fieldnames=review.ASSIGNMENT_FIELDNAMES
    )
    result = review.score_round(review.read_manifest(manifest_path), [sheet])
    assert all(o.resolved for o in result.outcomes)


def test_a_duplicate_adjudication_column_is_refused(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    first = read_rows(csv_path)[0]["row_id"]
    adjudication = _write_raw(
        tmp_path / "adj.csv",
        [
            ["final_verdict"] + list(review.ADJUDICATION_FIELDNAMES),
            ["supported", first, "unsupported", "carol", "2026-07-27", ""],
        ],
    )
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    b = complete(
        csv_path,
        tmp_path / "b.csv",
        "bob",
        lambda i: "unsupported" if i == 0 else "supported",
    )
    with pytest.raises(review.ReviewSamplingError, match="final_verdict"):
        g2.score_sentence_round(
            g2.read_manifest(manifest_path), [a, b], adjudicated=adjudication
        )


def test_a_row_with_more_cells_than_its_header_is_refused(sampled, tmp_path):
    _, csv_path, manifest_path = sampled
    header, *body = _raw_rows(csv_path)
    body[0] = body[0] + ["smuggled"]
    sheet = _write_raw(tmp_path / "extra.csv", [header] + body)
    with pytest.raises(review.ReviewSamplingError, match="more cells than its header"):
        g2.score_sentence_round(g2.read_manifest(manifest_path), [sheet])


# -- P2: aggregate accounting is derived, never believed ---------------------


@pytest.fixture
def withheld_round(world, tmp_path):
    """A selected day that genuinely holds a withheld artifact."""

    standard_world(world)
    ids = seed(world, "AMD", D1, themes=(1,))
    summarize(
        world,
        "AMD",
        D1,
        ids[0],
        client=Client(first_sentence=f"A filing shows {CREDENTIAL}."),
    )
    pop = population(world)
    seed_text = next(
        f"w{n}"
        for n in range(100)
        if D1 in g2.sample_sentences(pop, seed=f"w{n}").selected_days
    )
    return sample(world, tmp_path, seed_text=seed_text, pop=pop)


def _amd_d1(payload):
    return next(
        p
        for p in payload["population"]["partitions"]
        if (p["trading_day"], p["ticker"]) == (D1, "AMD")
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["population"]["selected_days"].update(withheld_artifact_count=0),
        lambda p: p["population"]["selected_days"].update(degraded_theme_count=0),
        lambda p: p["population"]["selected_days"].update(artifact_count=1),
        lambda p: p["population"]["selected_days"].update(sentence_count=1),
        lambda p: p["population"]["selected_days"].update(
            population_changed_partition_count=1
        ),
        lambda p: _amd_d1(p).update(withheld_artifact_count=0),
        lambda p: _amd_d1(p).update(degraded_theme_count=1),
        lambda p: _amd_d1(p).update(theme_count=0),
        lambda p: _amd_d1(p).update(reviewed_artifact_ids=[424242]),
        lambda p: _amd_d1(p)["themes"][0].update(reason=""),
        lambda p: _amd_d1(p)["themes"][0].update(artifact_id=424242),
        lambda p: _amd_d1(p).update(outcome="no_story_output"),
    ],
    ids=[
        "selected-withheld",
        "selected-degraded",
        "selected-artifacts",
        "selected-sentences",
        "selected-population-changed",
        "partition-withheld",
        "partition-degraded",
        "partition-theme-count",
        "partition-reviewed-ids",
        "theme-withheld-reason",
        "theme-artifact-id",
        "partition-outcome",
    ],
)
def test_forged_accounting_is_refused_even_with_outer_digests_recomputed(
    withheld_round, change
):
    _, _, manifest_path = withheld_round
    rewrite_manifest(manifest_path, change, rebind=True)
    with pytest.raises(
        review.ReviewSamplingError,
        match=(
            "accounting was altered|theme outcome|names an artifact"
            "|only an enumerated partition"
        ),
    ):
        g2.read_manifest(manifest_path)


def test_the_codex_withheld_count_forgery_is_refused(withheld_round, tmp_path):
    """Reproduction B: zero the withheld total, rebind, recut the sheet."""

    _, csv_path, manifest_path = withheld_round
    assert (
        json.loads(manifest_path.read_text())["population"]["selected_days"][
            "withheld_artifact_count"
        ]
        == 1
    )
    rewrite_manifest(
        manifest_path,
        lambda p: p["population"]["selected_days"].update(withheld_artifact_count=0),
        rebind=True,
    )
    with pytest.raises(review.ReviewSamplingError, match="accounting was altered"):
        g2.read_manifest(manifest_path)


# -- P2: development provenance is a bound fact ------------------------------


def _cli_sample(world, out, *extra):
    return make_review_sheets.main(
        [
            "sample-sentences",
            "--database",
            str(world.path),
            "--window-start",
            D1,
            "--window-end",
            D5,
            "--seed",
            "dev",
            "--out",
            str(out),
            *extra,
        ]
    )


def test_a_two_day_development_draw_stays_a_development_draw(world, tmp_path, capsys):
    standard_world(world)
    out = tmp_path / "dev" / "g2.csv"
    assert _cli_sample(world, out, "--development-days", "2") == 0
    manifest_path = out.with_name("g2.manifest.json")
    manifest = g2.read_manifest(manifest_path)
    assert manifest["selection"]["draw"]["size"] == 2
    assert manifest["selection"]["draw"]["development_override"] is True
    a = complete(out, tmp_path / "a.csv", "alice", lambda i: "supported")
    b = complete(out, tmp_path / "b.csv", "bob", lambda i: "supported")
    card = g2.score_g2(g2.score_sentence_round(manifest, [a, b]))
    assert card.evaluation_mode == "development"
    assert card.gate_result is review.GateResult.NOT_ELIGIBLE
    assert any("development draw of 2 day(s)" in x for x in card.eligibility_blockers)


def test_a_release_invocation_records_no_override(world, tmp_path):
    standard_world(world)
    out = tmp_path / "rel" / "g2.csv"
    assert _cli_sample(world, out) == 0
    draw = g2.read_manifest(out.with_name("g2.manifest.json"))["selection"]["draw"]
    assert draw["development_override"] is False and draw["size"] == 2


def test_development_and_release_rows_have_different_ids(world):
    standard_world(world)
    pop = population(world)
    release = g2.sample_sentences(pop, seed="same")
    development = g2.sample_sentences(pop, seed="same", draw_size=2)
    assert release.selected_days == development.selected_days
    assert {r.row_id for r in release.rows}.isdisjoint(
        {r.row_id for r in development.rows}
    )


def test_erasing_the_override_is_refused_with_outer_digests_recomputed(world, tmp_path):
    standard_world(world)
    _, csv_path, manifest_path = sample(world, tmp_path, draw_size=2)
    rewrite_manifest(
        manifest_path,
        lambda p: p["selection"]["draw"].update(development_override=False),
        rebind=True,
    )
    with pytest.raises(review.ReviewSamplingError, match="exact sentence census"):
        g2.read_manifest(manifest_path)


def test_erasing_the_override_and_reauthoring_rows_orphans_every_sheet(world, tmp_path):
    standard_world(world)
    _, csv_path, manifest_path = sample(world, tmp_path, draw_size=2)
    reviewed = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")

    def reauthor(payload):
        payload["selection"]["draw"]["development_override"] = False
        payload["snapshot"]["rows"] = [
            row.snapshot()
            for a in payload["snapshot"]["artifacts"]
            for row in g2.rows_for_artifact(a, development_override=False)
        ]

    rewrite_manifest(manifest_path, reauthor, rebind=True)
    forged = g2.read_manifest(manifest_path)  # a different, re-authored artifact
    with pytest.raises(review.ReviewSamplingError):
        g2.score_sentence_round(forged, [reviewed])


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_a_malformed_override_flag_is_refused(world, tmp_path, value):
    standard_world(world)
    _, _, manifest_path = sample(world, tmp_path)
    rewrite_manifest(
        manifest_path,
        lambda p: p["selection"]["draw"].update(development_override=value),
        rebind=True,
    )
    with pytest.raises(review.ReviewSamplingError, match="development_override"):
        g2.read_manifest(manifest_path)


def test_a_non_two_day_draw_without_an_override_is_refused(world, tmp_path):
    standard_world(world)
    _, _, manifest_path = sample(world, tmp_path, draw_size=1)
    rewrite_manifest(
        manifest_path,
        lambda p: p["selection"]["draw"].update(development_override=False),
        rebind=True,
    )
    with pytest.raises(review.ReviewSamplingError, match="must record its development"):
        g2.read_manifest(manifest_path)


# -- P2: configuration and database errors are input errors ----------------


@pytest.mark.parametrize(
    "name, value",
    [
        ("GEMINI_MAX_OUTPUT_TOKENS", "invalid"),
        ("GEMINI_MAX_OUTPUT_TOKENS", "0"),
        ("GEMINI_TIMEOUT_MS", "invalid"),
        ("GEMINI_TIMEOUT_MS", "-5"),
    ],
)
def test_malformed_policy_configuration_is_a_safe_usage_error(
    world, tmp_path, capsys, monkeypatch, name, value
):
    standard_world(world)
    monkeypatch.setenv(name, value)
    out = tmp_path / "cfg" / "g2.csv"
    assert _cli_sample(world, out) == make_review_sheets.EXIT_USAGE
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err + captured.out
    assert f"error: the production summary policy cannot be resolved: {name}" in (
        captured.err
    )
    assert not out.parent.exists() or not any(out.parent.iterdir())


def test_an_unexpected_programmer_error_is_not_disguised(world, tmp_path, monkeypatch):
    standard_world(world)

    def broken():
        raise TypeError("a real bug")

    monkeypatch.setattr(g2, "_default_policy", broken)
    with pytest.raises(TypeError, match="a real bug"):
        _cli_sample(world, tmp_path / "bug" / "g2.csv")


def test_a_file_that_is_not_sqlite_is_a_safe_usage_error(tmp_path, capsys):
    bogus = tmp_path / "phase0.db"
    bogus.write_bytes(b"this is not a database" * 100)
    code = make_review_sheets.main(
        [
            "sample-sentences",
            "--database",
            str(bogus),
            "--candidate-day",
            D1,
            "--seed",
            "s",
            "--out",
            str(tmp_path / "o" / "g2.csv"),
        ]
    )
    captured = capsys.readouterr()
    assert code == make_review_sheets.EXIT_USAGE
    assert "cannot read the Phase 0 database phase0.db: DatabaseError" in captured.err
    assert "Traceback" not in captured.err


# -- P2: outputs never overwrite anything -------------------------------------


@pytest.fixture
def cli_world(world, tmp_path, monkeypatch):
    standard_world(world)
    monkeypatch.chdir(tmp_path)
    return world


def _db_state(world):
    return file_sha(world.path), table_dump(world.path)


@pytest.mark.parametrize(
    "alias",
    [
        lambda w: str(w.path),
        lambda w: "phase0.db",
        lambda w: str(w.path.parent / "sub" / ".." / "phase0.db"),
        lambda w: str(w.path) + "-wal",
        lambda w: str(w.path.parent / "phase0.db.manifest.json"),
    ],
    ids=["same-path", "relative", "dotdot", "wal-sidecar", "not-an-alias-control"],
)
def test_out_can_never_be_the_database(cli_world, capsys, alias):
    before = _db_state(cli_world)
    target = alias(cli_world)
    (cli_world.path.parent / "sub").mkdir(exist_ok=True)
    code = _cli_sample(cli_world, target)
    if target.endswith("manifest.json"):
        # Control: a genuinely new path is written, and the database is intact.
        assert code == 0
    else:
        assert code == make_review_sheets.EXIT_USAGE
        assert "refusing to write over it" in capsys.readouterr().err
    assert _db_state(cli_world) == before


def test_a_symlink_to_the_database_is_refused(cli_world, tmp_path, capsys):
    before = _db_state(cli_world)
    link = tmp_path / "link.csv"
    link.symlink_to(cli_world.path)
    assert _cli_sample(cli_world, link) == make_review_sheets.EXIT_USAGE
    assert "refusing to write over it" in capsys.readouterr().err
    assert _db_state(cli_world) == before


def test_a_manifest_path_aliasing_the_database_is_refused(cli_world, tmp_path, capsys):
    before = _db_state(cli_world)
    (tmp_path / "evil.manifest.json").symlink_to(cli_world.path)
    assert (
        _cli_sample(cli_world, tmp_path / "evil.csv") == make_review_sheets.EXIT_USAGE
    )
    assert "refusing to write over it" in capsys.readouterr().err
    assert not (tmp_path / "evil.csv").exists()
    assert _db_state(cli_world) == before


@pytest.mark.parametrize("existing", ["g2.csv", "g2.manifest.json"])
def test_existing_review_outputs_are_never_overwritten(
    cli_world, tmp_path, capsys, existing
):
    out = tmp_path / "keep" / "g2.csv"
    out.parent.mkdir()
    (out.parent / existing).write_text("keep me")
    assert _cli_sample(cli_world, out) == make_review_sheets.EXIT_USAGE
    assert "already exists" in capsys.readouterr().err
    assert (out.parent / existing).read_text() == "keep me"
    assert sorted(os.listdir(out.parent)) == [existing]


def test_csv_and_manifest_can_never_be_one_file(tmp_path):
    with pytest.raises(review.ReviewSamplingError, match="are the same file"):
        g2.check_output_paths([tmp_path / "x.csv", tmp_path / "." / "x.csv"])
    (tmp_path / "g2.manifest.json").symlink_to(tmp_path / "g2.csv")
    with pytest.raises(review.ReviewSamplingError, match="same file|already exists"):
        g2.check_output_paths([tmp_path / "g2.csv", tmp_path / "g2.manifest.json"])


def test_write_sample_leaves_no_half_pair(world, tmp_path):
    standard_world(world)
    drawn = g2.sample_sentences(population(world), seed="s")
    manifest = g2.build_manifest(drawn, csv_name="g2.csv", code=CODE)
    (tmp_path / "g2.manifest.json").write_text("occupied")
    with pytest.raises(review.ReviewSamplingError, match="already exists"):
        g2.write_sample(drawn, tmp_path / "g2.csv", manifest=manifest)
    assert not (tmp_path / "g2.csv").exists()
    assert (tmp_path / "g2.manifest.json").read_text() == "occupied"


@pytest.fixture
def scored_inputs(world, tmp_path):
    standard_world(world)
    _, csv_path, manifest_path = sample(world, tmp_path)
    a = complete(csv_path, tmp_path / "a.csv", "alice", lambda i: "supported")
    return manifest_path, a


def _score(manifest_path, sheet, report):
    return make_review_sheets.main(
        [
            "score-sentences",
            "--round",
            str(manifest_path),
            str(sheet),
            "--report",
            str(report),
        ]
    )


@pytest.mark.parametrize(
    "target",
    [lambda m, a: a, lambda m, a: m, lambda m, a: a.parent / "x" / ".." / a.name],
    ids=["report-is-sheet", "report-is-manifest", "report-dotdot-alias"],
)
def test_the_report_can_never_be_an_input(scored_inputs, capsys, target):
    manifest_path, sheet = scored_inputs
    (sheet.parent / "x").mkdir(exist_ok=True)
    before = {p: file_sha(p) for p in (manifest_path, sheet)}
    assert _score(manifest_path, sheet, target(manifest_path, sheet)) == 2
    assert "refusing to write over it" in capsys.readouterr().err
    assert {p: file_sha(p) for p in before} == before


def test_an_existing_report_is_never_overwritten(scored_inputs, tmp_path, capsys):
    manifest_path, sheet = scored_inputs
    report = tmp_path / "scorecard.json"
    report.write_text("keep me")
    assert _score(manifest_path, sheet, report) == 2
    assert "already exists" in capsys.readouterr().err
    assert report.read_text() == "keep me"


# -- P2: credential-like operator identifiers --------------------------------


@pytest.mark.parametrize(
    "flag", ["--seed", "--round-id", "--protocol"], ids=["seed", "round-id", "protocol"]
)
def test_a_credential_like_operator_identifier_is_refused_before_output(
    cli_world, tmp_path, capsys, flag
):
    out = tmp_path / "cred" / "g2.csv"
    args = [
        "sample-sentences",
        "--database",
        str(cli_world.path),
        "--window-start",
        D1,
        "--window-end",
        D5,
        "--seed",
        "fine",
        "--out",
        str(out),
    ]
    if flag == "--seed":
        args[args.index("--seed") + 1] = CREDENTIAL
    else:
        args += [flag, CREDENTIAL]
    code = make_review_sheets.main(args)
    captured = capsys.readouterr()
    assert code == make_review_sheets.EXIT_USAGE
    assert "credential-like" in captured.err
    for secret in (CREDENTIAL, "abcd1234efgh5678"):
        assert secret not in captured.err and secret not in captured.out
    assert not out.parent.exists()


def test_a_credential_like_output_name_is_refused(cli_world, tmp_path, capsys):
    out = tmp_path / "cred" / "token=abcd1234efgh5678.csv"
    assert _cli_sample(cli_world, out) == make_review_sheets.EXIT_USAGE
    captured = capsys.readouterr()
    assert "abcd1234efgh5678" not in captured.err + captured.out
    assert not out.parent.exists()


@pytest.mark.parametrize("field", ["seed", "round_id"])
def test_the_library_refuses_credential_like_values_without_echoing(world, field):
    standard_world(world)
    pop = population(world)
    kwargs = {"seed": "fine", "round_id": "fine", field: CREDENTIAL}
    with pytest.raises(review.ReviewSamplingError) as caught:
        g2.sample_sentences(pop, **kwargs)
    assert field in str(caught.value)
    assert "abcd1234efgh5678" not in str(caught.value)
    with pytest.raises(review.ReviewSamplingError) as caught:
        g2.require_known_g2_protocol(CREDENTIAL)
    assert "abcd1234efgh5678" not in str(caught.value)


# ======================================================================
# Closure repairs
# ======================================================================


# -- Companions of the resolved database target ------------------------------


@pytest.fixture
def aliased_db(cli_world, tmp_path):
    """``alias.db -> real.db``: a closed copy with no companion files present.

    The copy is made with SQLite's backup API and closed, so ``real.db-wal``,
    ``-shm`` and ``-journal`` do not exist: an output there can only be
    refused because it *is* a companion, not because something is in the way.
    """

    directory = tmp_path / "db"
    directory.mkdir()
    real = directory / "real.db"
    source = sqlite3.connect(f"file:{cli_world.path}?mode=ro", uri=True)
    target = sqlite3.connect(real)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    alias = directory / "alias.db"
    alias.symlink_to(real)
    assert sorted(os.listdir(directory)) == ["alias.db", "real.db"]
    return real, alias


def test_resolved_target_companions_are_protected(aliased_db):
    real, alias = aliased_db
    protected = {p.name for p in g2.protected_database_paths(alias)}
    for base in ("alias.db", "real.db"):
        for suffix in ("", "-wal", "-shm", "-journal"):
            assert base + suffix in protected


def _sample_through_alias(alias, out):
    return make_review_sheets.main(
        [
            "sample-sentences",
            "--database",
            str(alias),
            "--window-start",
            D1,
            "--window-end",
            D5,
            "--seed",
            "s",
            "--out",
            str(out),
        ]
    )


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_out_can_never_be_a_companion_of_the_resolved_database(
    aliased_db, capsys, suffix
):
    real, alias = aliased_db
    target = real.with_name(real.name + suffix)
    before = file_sha(real)
    assert _sample_through_alias(alias, target) == make_review_sheets.EXIT_USAGE
    assert "refusing to write over it" in capsys.readouterr().err
    assert sorted(os.listdir(real.parent)) == ["alias.db", "real.db"]
    assert file_sha(real) == before


def test_supplied_path_companions_stay_protected(aliased_db, capsys):
    real, alias = aliased_db
    target = alias.with_name("alias.db-wal")
    assert _sample_through_alias(alias, target) == make_review_sheets.EXIT_USAGE
    assert "refusing to write over it" in capsys.readouterr().err
    assert sorted(os.listdir(real.parent)) == ["alias.db", "real.db"]


def test_sampling_through_the_alias_still_works(aliased_db, tmp_path):
    real, alias = aliased_db
    out = tmp_path / "ok" / "g2.csv"
    assert _sample_through_alias(alias, out) == 0
    assert g2.read_manifest(out.with_name("g2.manifest.json"))["source"] == {
        "mode": "persisted",
        "database": "alias.db",
    }


# -- No partial output survives a failed write --------------------------------


class _FailingHandle:
    """A real file that fails mid-write (or at close), as a full disk would."""

    def __init__(self, real, stage):
        self.real, self.stage = real, stage

    def write(self, text):
        if self.stage == "write":
            self.real.write(text[: len(text) // 2])
            self.real.flush()
            raise OSError(errno.ENOSPC, "No space left on device")
        return self.real.write(text)

    def flush(self):
        return self.real.flush()

    def fileno(self):
        return self.real.fileno()

    def close(self):
        self.real.close()
        if self.stage == "close":
            raise OSError(errno.ENOSPC, "No space left on device")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _fail_when_creating(monkeypatch, suffix, stage):
    real_open = Path.open

    def opener(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if mode == "x" and self.name.endswith(suffix):
            return _FailingHandle(handle, stage)
        return handle

    monkeypatch.setattr(Path, "open", opener)


@pytest.mark.parametrize("stage", ["write", "close"])
@pytest.mark.parametrize(
    "failing", [".csv", ".manifest.json"], ids=["csv-fails", "manifest-fails"]
)
def test_a_failed_sample_write_leaves_neither_file(
    cli_world, tmp_path, capsys, monkeypatch, failing, stage
):
    before = _db_state(cli_world)
    out = tmp_path / "full" / "g2.csv"
    _fail_when_creating(monkeypatch, failing, stage)
    assert _cli_sample(cli_world, out) == make_review_sheets.EXIT_USAGE
    captured = capsys.readouterr()
    assert "No space left on device" in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()
    assert not out.with_name("g2.manifest.json").exists()
    assert _db_state(cli_world) == before


@pytest.mark.parametrize("stage", ["write", "close"])
def test_a_failed_report_write_leaves_no_report(
    scored_inputs, tmp_path, capsys, monkeypatch, stage
):
    manifest_path, sheet = scored_inputs
    before = {p: file_sha(p) for p in (manifest_path, sheet)}
    report = tmp_path / "reports" / "scorecard.json"
    _fail_when_creating(monkeypatch, "scorecard.json", stage)
    assert _score(manifest_path, sheet, report) == make_review_sheets.EXIT_USAGE
    assert "No space left on device" in capsys.readouterr().err
    assert not report.exists()
    assert {p: file_sha(p) for p in before} == before


def test_cleanup_never_removes_a_file_that_already_existed(tmp_path):
    existing = tmp_path / "occupied.csv"
    existing.write_text("keep me")
    with pytest.raises(review.ReviewSamplingError, match="already exists"):
        g2.create_new_file(existing, "new content")
    assert existing.read_text() == "keep me"


def test_a_non_io_failure_mid_write_still_removes_the_file(tmp_path, monkeypatch):
    target = tmp_path / "x.csv"
    real_open = Path.open

    class Boom(_FailingHandle):
        def write(self, text):
            self.real.write(text[:3])
            raise KeyboardInterrupt

    def opener(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        return Boom(handle, "") if mode == "x" else handle

    monkeypatch.setattr(Path, "open", opener)
    with pytest.raises(KeyboardInterrupt):
        g2.create_new_file(target, "content")
    assert not target.exists()

"""Sampling and scorecard logic for issue #74 (A4): human-review sheets.

Section 8 of ``docs/PHASE_0_SPEC.md`` needs two review-sheet kinds to compute
its go/no-go gates:

    G1  story -> theme assignment correctness (>=75%)
    G2  summary-sentence faithfulness (>=95%)

**No real soak-window data exists yet.** #57 (I1, the real SQLite pipeline)
is still open, so there is nothing to sample story->theme assignments or
summaries *from* except M5's own committed eval fixture
(``nlp/themes/data/ticker_days.json`` + ``story_vectors.json``): 3
ticker-days, 30 stories, explicitly ``dataset_kind: synthetic_development``.
K3 (#60, reviewer guidelines) is also still open, so there is no committed
sheet format to match either. This module follows the same posture M4 and M5
already established for exactly this situation (see ``nlp/eval/trust.py``):
proceed on the synthetic fixture, and make every output honestly declare
that it is not gate-eligible, rather than block.

This is also the **first code path that calls the real summarizer
(``ai.summarization.summarize``) against real M5 output** — until now,
nothing wired M5's ``ThemeSet`` to A1's ``summarize()``.

Assignment sampling is fully offline (clustering replays the committed
vector fixture, same as ``tools.eval_themes``). Summary-sentence sampling
calls ``ai.summarization.summarize`` and therefore needs a client: pass a
real ``ai.summarization.GeminiClient`` to actually call Gemini, or an
injectable fake in tests -- there is no offline substitute for what a live
model says, so this module never fabricates one itself.
"""

from __future__ import annotations

import csv
import dataclasses
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ai.summarization import summarize
from nlp.embeddings import compose_embedding_text
from nlp.eval.trust import DatasetKind, LabelingStatus, MetricsPurpose, TrustContract
from nlp.themes.config import ThemeConfig
from nlp.themes.dataset import DEFAULT_FIXTURE_PATH, TickerDaySet, load_ticker_days, tickers_of
from nlp.themes.models import ThemeSet
from nlp.themes.service import cluster_themes
from nlp.themes.summarization import theme_to_summarizer_input
from nlp.themes.vectors import DEFAULT_VECTOR_PATH, FixtureEncoder, load_story_vectors

DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SUMMARY_DAYS = 2
DEFAULT_ASSIGNMENT_SEED = "phase0-a4-assignments"
DEFAULT_SUMMARY_SEED = "phase0-a4-summaries"
DEFAULT_ASSIGNMENT_THRESHOLD = 0.75  # gate G1
DEFAULT_SUMMARY_THRESHOLD = 0.95  # gate G2

POSITIVE_ASSIGNMENT_VERDICT = "correct"
NEGATIVE_ASSIGNMENT_VERDICT = "incorrect"
POSITIVE_SUMMARY_VERDICT = "supported"
NEGATIVE_SUMMARY_VERDICT = "not_supported"

DayKey = tuple[str, str]  # (ticker, trading_day.isoformat())


class ReviewSamplingError(ValueError):
    """A review sheet, or a completed sheet being scored, is not usable."""


# --------------------------------------------------------------------------
# Loading: offline by default, same posture as tools.eval_themes.
# --------------------------------------------------------------------------


def load_theme_sets(
    fixture_path: str | Path = DEFAULT_FIXTURE_PATH,
    vectors_path: str | Path = DEFAULT_VECTOR_PATH,
) -> tuple[TickerDaySet, dict[DayKey, ThemeSet]]:
    """Cluster every ticker-day in the committed M5 fixture, offline.

    Vectors replay the committed fixture rather than loading a model, so
    this is deterministic and network-free by default -- the same contract
    ``tools.eval_themes`` makes for AC-4 evaluation.
    """

    day_set = load_ticker_days(fixture_path)
    fixture_encoder = FixtureEncoder(load_story_vectors(vectors_path))
    fixture_encoder.bind(
        {
            story.story_key: compose_embedding_text(story.title, story.description)
            for day in day_set.days
            for story in day.stories
        }
    )
    config = ThemeConfig(supported_tickers=tickers_of(day_set))
    theme_sets: dict[DayKey, ThemeSet] = {}
    for day in day_set.days:
        theme_sets[(day.ticker, day.trading_day.isoformat())] = cluster_themes(
            day.stories,
            ticker=day.ticker,
            trading_day=day.trading_day,
            config=config,
            encoder=fixture_encoder,
        )
    return day_set, theme_sets


# --------------------------------------------------------------------------
# Part (a): story -> theme assignment sampling (gate G1).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AssignmentRow:
    row_id: str
    ticker: str
    trading_day: str
    story_key: str
    story_title: str
    story_outlets: str
    assignment_type: str  # "theme" | "other_coverage" | "excluded"
    theme_key: str
    theme_label: str
    reason: str
    reviewer_verdict: str = ""
    reviewer_notes: str = ""


ASSIGNMENT_FIELDNAMES = tuple(field.name for field in dataclasses.fields(AssignmentRow))


@dataclass(frozen=True)
class AssignmentSample:
    population_size: int
    requested_sample_size: int
    actual_sample_size: int
    seed: str
    rows: tuple[AssignmentRow, ...]


def _all_assignment_rows(theme_sets: Mapping[DayKey, ThemeSet]) -> list[AssignmentRow]:
    """Every story's assignment outcome, across every loaded ticker-day.

    ``ThemeSet.accounted_story_keys`` guarantees each story appears in
    exactly one of themes/other_coverage/excluded, so this is the whole
    population with no story missed or double-counted.
    """

    rows: list[AssignmentRow] = []
    for (ticker, trading_day), theme_set in theme_sets.items():
        for theme in theme_set.themes:
            for entry in theme.evidence:
                rows.append(
                    AssignmentRow(
                        row_id=f"{ticker}:{trading_day}:{entry.story_key}",
                        ticker=ticker,
                        trading_day=trading_day,
                        story_key=entry.story_key,
                        story_title=entry.title,
                        story_outlets=";".join(entry.outlets),
                        assignment_type="theme",
                        theme_key=theme.theme_key,
                        theme_label=theme.label,
                        reason="",
                    )
                )
        for entry in theme_set.other_coverage:
            rows.append(
                AssignmentRow(
                    row_id=f"{ticker}:{trading_day}:{entry.story_key}",
                    ticker=ticker,
                    trading_day=trading_day,
                    story_key=entry.story_key,
                    story_title=entry.evidence.title,
                    story_outlets=";".join(entry.evidence.outlets),
                    assignment_type="other_coverage",
                    theme_key="",
                    theme_label="",
                    reason=entry.reason.value,
                )
            )
        for entry in theme_set.excluded:
            rows.append(
                AssignmentRow(
                    row_id=f"{ticker}:{trading_day}:{entry.story_key}",
                    ticker=ticker,
                    trading_day=trading_day,
                    story_key=entry.story_key,
                    story_title="",
                    story_outlets="",
                    assignment_type="excluded",
                    theme_key="",
                    theme_label="",
                    reason=entry.reason.value,
                )
            )
    return rows


def sample_assignments(
    theme_sets: Mapping[DayKey, ThemeSet],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: str = DEFAULT_ASSIGNMENT_SEED,
) -> AssignmentSample:
    """Deterministically sample story->theme assignments for G1 review.

    Takes ``min(sample_size, population)`` rather than sampling with
    replacement: the committed fixture holds only 30 stories today, fewer
    than the spec's 40, and a sample padded with repeats would overstate
    how much was actually reviewed. Callers can see the shortfall in
    ``population_size`` vs. ``requested_sample_size``.
    """

    population = sorted(_all_assignment_rows(theme_sets), key=lambda row: row.row_id)
    if not population:
        raise ReviewSamplingError("no story->theme assignments to sample from")
    rng = random.Random(seed)
    actual_size = min(sample_size, len(population))
    sampled = sorted(rng.sample(population, actual_size), key=lambda row: row.row_id)
    return AssignmentSample(
        population_size=len(population),
        requested_sample_size=sample_size,
        actual_sample_size=actual_size,
        seed=seed,
        rows=tuple(sampled),
    )


def write_assignment_csv(sample: AssignmentSample, path: str | Path) -> Path:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ASSIGNMENT_FIELDNAMES)
        writer.writeheader()
        for row in sample.rows:
            writer.writerow(dataclasses.asdict(row))
    return location


# --------------------------------------------------------------------------
# Part (b): summary-sentence sampling (gate G2). Calls the real summarizer.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SentenceRow:
    row_id: str
    ticker: str
    trading_day: str
    theme_key: str
    theme_label: str
    sentence_index: int
    sentence_text: str
    citation_ids: str
    cited_story_titles: str
    cited_outlets: str
    reviewer_verdict: str = ""
    reviewer_notes: str = ""


SENTENCE_FIELDNAMES = tuple(field.name for field in dataclasses.fields(SentenceRow))


@dataclass(frozen=True)
class SentenceSample:
    day_population_size: int
    requested_days: int
    actual_days: int
    sampled_days: tuple[DayKey, ...]
    seed: str
    rows: tuple[SentenceRow, ...]


def sample_summary_sentences(
    theme_sets: Mapping[DayKey, ThemeSet],
    *,
    client: Any,
    days: int = DEFAULT_SUMMARY_DAYS,
    seed: str = DEFAULT_SUMMARY_SEED,
) -> SentenceSample:
    """Sample days, summarize their themes for real, and list every sentence.

    ``client`` is required and never defaulted: this is the one place in the
    tool that makes a real model call, so the caller must say explicitly
    which client answers it (a real ``ai.summarization.GeminiClient`` to
    actually call Gemini, a fake one in tests).

    ``ai.summarization.summarize`` already enforces that every citation_id
    resolves to a real member story before returning (see issue #65's
    citation-integrity fix), so every row emitted here is guaranteed to cite
    a real story in its own theme -- never an invented or foreign id.
    """

    all_days = sorted(theme_sets.keys())
    if not all_days:
        raise ReviewSamplingError("no ticker-days to sample summaries from")
    rng = random.Random(seed)
    actual_days = min(days, len(all_days))
    sampled_days = tuple(sorted(rng.sample(all_days, actual_days)))

    rows: list[SentenceRow] = []
    for ticker, trading_day in sampled_days:
        theme_set = theme_sets[(ticker, trading_day)]
        for theme in theme_set.themes:
            evidence_by_key = {entry.story_key: entry for entry in theme.evidence}
            theme_input = theme_to_summarizer_input(theme)
            summary = summarize(theme_input, client=client)
            for index, sentence in enumerate(summary.sentences):
                titles = []
                outlets = []
                for citation_id in sentence.citation_ids:
                    entry = evidence_by_key.get(citation_id)
                    titles.append(entry.title if entry is not None else "")
                    outlets.append(", ".join(entry.outlets) if entry is not None else "")
                rows.append(
                    SentenceRow(
                        row_id=f"{ticker}:{trading_day}:{theme.theme_key}:{index}",
                        ticker=ticker,
                        trading_day=trading_day,
                        theme_key=theme.theme_key,
                        theme_label=summary.label,
                        sentence_index=index,
                        sentence_text=sentence.text,
                        citation_ids=";".join(sentence.citation_ids),
                        cited_story_titles=";".join(titles),
                        cited_outlets=";".join(outlets),
                    )
                )
    return SentenceSample(
        day_population_size=len(all_days),
        requested_days=days,
        actual_days=actual_days,
        sampled_days=sampled_days,
        seed=seed,
        rows=tuple(rows),
    )


def write_sentence_csv(sample: SentenceSample, path: str | Path) -> Path:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SENTENCE_FIELDNAMES)
        writer.writeheader()
        for row in sample.rows:
            writer.writerow(dataclasses.asdict(row))
    return location


# --------------------------------------------------------------------------
# Reading completed sheets and computing the scorecard.
# --------------------------------------------------------------------------


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class Scorecard:
    """G1/G2 numbers computed from one or two completed review sheets.

    ``trust_contract`` always declares ``gate_eligible=False``: the
    population sampled from is M5's synthetic fixture, and
    ``nlp.eval.trust.check_trust_invariants`` structurally refuses to let a
    ``dataset_kind=synthetic_development`` contract claim gate eligibility,
    so this stays true by construction, not by convention, until real
    soak-window data and K3's guidelines replace the fixture this was
    sampled from.
    """

    sample_size: int
    reviewer_count: int
    agreement_rate: Optional[float]
    resolved_count: int
    unresolved_count: int
    unresolved_row_ids: tuple[str, ...]
    rate: Optional[float]
    gate_threshold: float
    meets_gate: bool
    trust_contract: TrustContract

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "reviewer_count": self.reviewer_count,
            "agreement_rate": self.agreement_rate,
            "resolved_count": self.resolved_count,
            "unresolved_count": self.unresolved_count,
            "unresolved_row_ids": list(self.unresolved_row_ids),
            "rate": self.rate,
            "gate_threshold": self.gate_threshold,
            "meets_gate": self.meets_gate,
            "trust_contract": self.trust_contract.as_dict(),
        }


def _read_verdicts(path: str | Path, *, verdict_field: str) -> dict[str, str]:
    location = Path(path)
    rows = read_csv_rows(location)
    verdicts: dict[str, str] = {}
    for row in rows:
        if "row_id" not in row:
            raise ReviewSamplingError(f"{location}: sheet has no row_id column")
        row_id = row["row_id"]
        if row_id in verdicts:
            raise ReviewSamplingError(f"{location}: duplicate row_id {row_id!r}")
        if verdict_field not in row:
            raise ReviewSamplingError(f"{location}: sheet has no {verdict_field!r} column")
        verdicts[row_id] = row[verdict_field].strip()
    return verdicts


def _score(
    sheets: Sequence[str | Path],
    *,
    adjudicated: str | Path | None,
    threshold: float,
    positive_value: str,
    negative_value: str,
) -> Scorecard:
    if not sheets:
        raise ReviewSamplingError("at least one completed sheet is required")
    if len(sheets) > 2:
        raise ReviewSamplingError("at most two reviewer sheets are supported per round")

    sheet_verdicts = [_read_verdicts(sheet, verdict_field="reviewer_verdict") for sheet in sheets]
    row_ids = sorted(sheet_verdicts[0].keys())
    for sheet, verdicts in zip(sheets[1:], sheet_verdicts[1:]):
        if set(verdicts.keys()) != set(row_ids):
            raise ReviewSamplingError(
                f"{sheet}: does not cover the same row_ids as {sheets[0]}; "
                "reviewer sheets must review the same sample"
            )

    adjudicated_verdicts: dict[str, str] = {}
    if adjudicated is not None:
        adjudicated_verdicts = _read_verdicts(adjudicated, verdict_field="final_verdict")

    def _checked(value: str, where: str) -> str:
        if value not in (positive_value, negative_value, ""):
            raise ReviewSamplingError(
                f"{where}: verdict {value!r} must be {positive_value!r}, "
                f"{negative_value!r}, or blank"
            )
        return value

    outcomes: list[bool] = []
    unresolved_ids: list[str] = []
    for row_id in row_ids:
        verdicts = [
            _checked(sheet[row_id], f"{sheets[index]} row {row_id}")
            for index, sheet in enumerate(sheet_verdicts)
        ]
        if any(verdict == "" for verdict in verdicts):
            unresolved_ids.append(row_id)
            continue
        if len(verdicts) == 1 or verdicts[0] == verdicts[1]:
            outcomes.append(verdicts[0] == positive_value)
            continue
        final = adjudicated_verdicts.get(row_id, "")
        if final == "":
            unresolved_ids.append(row_id)
            continue
        outcomes.append(_checked(final, f"{adjudicated} row {row_id}") == positive_value)

    reviewer_count = len(sheets)
    agreement_rate: Optional[float] = None
    if reviewer_count == 2:
        comparable = [
            row_id
            for row_id in row_ids
            if sheet_verdicts[0][row_id] != "" and sheet_verdicts[1][row_id] != ""
        ]
        if comparable:
            matches = sum(
                1 for row_id in comparable if sheet_verdicts[0][row_id] == sheet_verdicts[1][row_id]
            )
            agreement_rate = matches / len(comparable)

    resolved_count = len(outcomes)
    unresolved_count = len(unresolved_ids)
    rate = (sum(outcomes) / resolved_count) if resolved_count else None

    if reviewer_count == 1:
        labeling_status = LabelingStatus.SINGLE_AUTHOR_UNADJUDICATED
        is_adjudicated = False
    elif adjudicated is not None:
        labeling_status = LabelingStatus.MULTI_REVIEWER_ADJUDICATED
        is_adjudicated = True
    else:
        labeling_status = LabelingStatus.MULTI_REVIEWER_UNADJUDICATED
        is_adjudicated = False

    trust_contract = TrustContract(
        dataset_kind=DatasetKind.SYNTHETIC_DEVELOPMENT,
        real_ingested_evidence=False,
        labeling_status=labeling_status,
        reviewer_count=reviewer_count,
        adjudicated=is_adjudicated,
        gate_eligible=False,
        metrics_purpose=MetricsPurpose.DEVELOPMENT_REGRESSION_ONLY,
    )

    meets_gate = rate is not None and unresolved_count == 0 and rate >= threshold

    return Scorecard(
        sample_size=len(row_ids),
        reviewer_count=reviewer_count,
        agreement_rate=agreement_rate,
        resolved_count=resolved_count,
        unresolved_count=unresolved_count,
        unresolved_row_ids=tuple(sorted(unresolved_ids)),
        rate=rate,
        gate_threshold=threshold,
        meets_gate=meets_gate,
        trust_contract=trust_contract,
    )


def score_assignments(
    sheets: Sequence[str | Path],
    *,
    adjudicated: str | Path | None = None,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> Scorecard:
    """Compute gate G1 (theme-assignment agreement) from completed sheets."""

    return _score(
        sheets,
        adjudicated=adjudicated,
        threshold=threshold,
        positive_value=POSITIVE_ASSIGNMENT_VERDICT,
        negative_value=NEGATIVE_ASSIGNMENT_VERDICT,
    )


def score_summaries(
    sheets: Sequence[str | Path],
    *,
    adjudicated: str | Path | None = None,
    threshold: float = DEFAULT_SUMMARY_THRESHOLD,
) -> Scorecard:
    """Compute gate G2 (summary-sentence faithfulness) from completed sheets."""

    return _score(
        sheets,
        adjudicated=adjudicated,
        threshold=threshold,
        positive_value=POSITIVE_SUMMARY_VERDICT,
        negative_value=NEGATIVE_SUMMARY_VERDICT,
    )

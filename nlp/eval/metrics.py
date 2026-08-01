"""Deterministic scoring for labeled pair predictions.

This module counts; it never decides.  A predictor hands back one
:class:`PairPrediction` per pair and everything here is arithmetic over
those, so a metric cannot disagree with the behaviour it claims to measure.

Two deliberate choices:

* **Undefined is ``None``, not zero.**  Precision over zero predicted
  merges is not "0% precise", it is unmeasured.  Returning 0.0 would let a
  stage that merges nothing look like a failing stage rather than an
  unevaluated one, and would silently satisfy a "recall is low" narrative
  with a number nobody computed.
* **Ambiguous pairs are excluded from the headline numbers** and reported
  separately.  Scoring against a label the author already marked arguable
  measures the coin flip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from .dataset import LabeledPair, PairSet


@dataclass(frozen=True)
class PairPrediction:
    """What a stage did with one labeled pair.

    ``candidate`` answers a different question from ``merged``: did the
    stage *consider* the pair at all?  The gap between candidate recall and
    merge recall separates "the generator never proposed it" from "the
    predicate refused it", which is the only way to tune a threshold
    without guessing.
    """

    merged: bool
    #: Which stage merged it, when one did.
    stage: str | None = None
    #: The stage's own similarity/confidence value, when it has one.
    score: float | None = None
    #: Whether the pair reached the merge predicate at all.
    candidate: bool = False
    #: Free-form, deterministic explanation for the confusion report.
    detail: str = ""


class PairPredictor(Protocol):
    """Anything that can label one pair."""

    def __call__(self, pair: LabeledPair) -> PairPrediction:
        ...


@dataclass(frozen=True)
class Confusion:
    """Which pairs landed in each cell, by pair id, sorted."""

    true_positives: tuple[str, ...] = ()
    false_positives: tuple[str, ...] = ()
    true_negatives: tuple[str, ...] = ()
    false_negatives: tuple[str, ...] = ()

    @property
    def tp(self) -> int:
        return len(self.true_positives)

    @property
    def fp(self) -> int:
        return len(self.false_positives)

    @property
    def tn(self) -> int:
        return len(self.true_negatives)

    @property
    def fn(self) -> int:
        return len(self.false_negatives)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return the ratio, or ``None`` when the denominator is zero."""

    if denominator == 0:
        return None
    return numerator / denominator


@dataclass(frozen=True)
class Metrics:
    """Precision, recall, F1, accuracy; ``None`` where undefined."""

    confusion: Confusion
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None

    @classmethod
    def from_confusion(cls, confusion: Confusion) -> "Metrics":
        precision = _ratio(confusion.tp, confusion.tp + confusion.fp)
        recall = _ratio(confusion.tp, confusion.tp + confusion.fn)
        if precision is None or recall is None or precision + recall == 0.0:
            f1: float | None = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        accuracy = _ratio(confusion.tp + confusion.tn, confusion.total)
        return cls(
            confusion=confusion,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
        )

    def meets(self, *, precision_floor: float, recall_floor: float) -> bool:
        """True only when both metrics are defined and clear their floor.

        An undefined metric never passes a gate.  A gate that a stage
        cannot be measured against has not been satisfied.
        """

        if self.precision is None or self.recall is None:
            return False
        return self.precision >= precision_floor and self.recall >= recall_floor


@dataclass(frozen=True)
class CategoryBreakdown:
    """One slice of the set, scored on its own."""

    key: str
    metrics: Metrics


@dataclass(frozen=True)
class EvaluationReport:
    """The full result of scoring one predictor against one pair set."""

    dataset_id: str
    predictor: str
    overall: Metrics
    by_category: tuple[CategoryBreakdown, ...]
    by_expected_stage: tuple[CategoryBreakdown, ...]
    by_ticker: tuple[CategoryBreakdown, ...]
    #: Recall computed over pairs the stage even considered, versus recall
    #: over every positive.  ``candidate_recall`` bounds ``recall``.
    candidate_recall: float | None
    #: Ambiguous pairs the stage merged, by pair id.  Not scored.
    ambiguous_merged: tuple[str, ...]
    ambiguous_count: int
    #: Per-pair detail keyed by pair id, for the confusion listing.
    details: Mapping[str, str] = field(default_factory=dict)
    #: The threshold this run used, when the predictor has one.
    threshold: float | None = None
    #: Predicted scores keyed by pair id, when the predictor emits them.
    scores: Mapping[str, float] = field(default_factory=dict)

    @property
    def merge_recall(self) -> float | None:
        """Alias for ``overall.recall``, named for the candidate contrast."""

        return self.overall.recall


def _score_slice(pairs: Sequence[tuple[LabeledPair, PairPrediction]]) -> Metrics:
    true_positives: list[str] = []
    false_positives: list[str] = []
    true_negatives: list[str] = []
    false_negatives: list[str] = []
    for pair, prediction in pairs:
        if pair.is_positive and prediction.merged:
            true_positives.append(pair.pair_id)
        elif not pair.is_positive and prediction.merged:
            false_positives.append(pair.pair_id)
        elif not pair.is_positive:
            true_negatives.append(pair.pair_id)
        else:
            false_negatives.append(pair.pair_id)
    return Metrics.from_confusion(
        Confusion(
            true_positives=tuple(sorted(true_positives)),
            false_positives=tuple(sorted(false_positives)),
            true_negatives=tuple(sorted(true_negatives)),
            false_negatives=tuple(sorted(false_negatives)),
        )
    )


def _breakdown(
    scored: Sequence[tuple[LabeledPair, PairPrediction]],
    key: Callable[[LabeledPair], str],
) -> tuple[CategoryBreakdown, ...]:
    groups: dict[str, list[tuple[LabeledPair, PairPrediction]]] = {}
    for pair, prediction in scored:
        groups.setdefault(key(pair), []).append((pair, prediction))
    return tuple(
        CategoryBreakdown(key=name, metrics=_score_slice(groups[name]))
        for name in sorted(groups)
    )


def evaluate(
    pair_set: PairSet,
    predictor: PairPredictor,
    *,
    name: str,
    threshold: float | None = None,
) -> EvaluationReport:
    """Run ``predictor`` over every pair and score the scored ones.

    The predictor is called once per pair, in dataset order, and the
    dataset is ordered by ``pair_id``, so two runs of the same predictor
    over the same set produce identical reports in any process.
    """

    predictions = [(pair, predictor(pair)) for pair in pair_set.pairs]
    for pair, prediction in predictions:
        if not isinstance(prediction, PairPrediction):
            raise TypeError(
                f"{name}: predictor returned {type(prediction).__name__} "
                f"for {pair.pair_id}, expected PairPrediction"
            )
    scored = [(pair, prediction) for pair, prediction in predictions if pair.is_scored]
    positives = [(pair, prediction) for pair, prediction in scored if pair.is_positive]
    considered = sum(
        1 for _, prediction in positives if prediction.candidate or prediction.merged
    )
    return EvaluationReport(
        dataset_id=pair_set.dataset_id,
        predictor=name,
        overall=_score_slice(scored),
        by_category=_breakdown(scored, lambda pair: pair.category),
        by_expected_stage=_breakdown(scored, lambda pair: pair.expected_stage),
        by_ticker=_breakdown(scored, lambda pair: pair.ticker),
        candidate_recall=_ratio(considered, len(positives)),
        ambiguous_merged=tuple(
            sorted(
                pair.pair_id
                for pair, prediction in predictions
                if pair.label == "ambiguous" and prediction.merged
            )
        ),
        ambiguous_count=len(pair_set.ambiguous),
        details={
            pair.pair_id: prediction.detail
            for pair, prediction in predictions
            if prediction.detail
        },
        threshold=threshold,
        scores={
            pair.pair_id: prediction.score
            for pair, prediction in predictions
            if prediction.score is not None
        },
    )


@dataclass(frozen=True)
class ThresholdPoint:
    """One point of a threshold sweep."""

    threshold: float
    report: EvaluationReport

    @property
    def precision(self) -> float | None:
        return self.report.overall.precision

    @property
    def recall(self) -> float | None:
        return self.report.overall.recall

    @property
    def f1(self) -> float | None:
        return self.report.overall.f1


def sweep_thresholds(
    pair_set: PairSet,
    predictor_factory: Callable[[float], PairPredictor],
    thresholds: Sequence[float],
    *,
    name: str,
) -> tuple[ThresholdPoint, ...]:
    """Score one predictor family across thresholds, low to high.

    Points are returned in ascending threshold order regardless of the
    order supplied, so a sweep reads the same way however it was requested.
    Repeated thresholds are rejected: silently scoring one twice would put
    two different-looking rows with identical numbers in a tuning table.
    """

    values = [float(threshold) for threshold in thresholds]
    if not values:
        raise ValueError("a sweep needs at least one threshold")
    if len(set(values)) != len(values):
        raise ValueError("sweep thresholds must be distinct")
    return tuple(
        ThresholdPoint(
            threshold=threshold,
            report=evaluate(
                pair_set,
                predictor_factory(threshold),
                name=f"{name}@{threshold:g}",
                threshold=threshold,
            ),
        )
        for threshold in sorted(values)
    )

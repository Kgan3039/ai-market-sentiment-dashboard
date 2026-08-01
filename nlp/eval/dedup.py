"""Scoring the deduplication stages against the labeled set.

The predictor here calls :func:`nlp.dedup.deduplicate` and reads its result.
It reimplements no identity rule, no window, and no threshold, so the
measured precision is the precision of the shipped stage rather than of a
parallel model of it.

Each pair is scored on its own two records.  That is faithful because M2
guarantees a record's clusters do not depend on which other records share
the batch (``nlp/dedup/service.py``), and it keeps a false merge attributable
to exactly one labeled pair.
"""

from __future__ import annotations

from typing import Sequence

from nlp.dedup import DedupConfig, MatchReason, RawItem, deduplicate

from .dataset import LabeledItem, LabeledPair, PairSet, default_pair_set
from .metrics import EvaluationReport, PairPrediction, PairPredictor, evaluate

#: Every accepted reason that means "these two are one story".  ``CANONICAL``
#: is excluded: it names the representative member, not a merge signal.
_MERGE_REASONS = frozenset(
    reason for reason in MatchReason if reason is not MatchReason.CANONICAL
)


def to_raw_item(item: LabeledItem) -> RawItem:
    """Project one labeled item onto the dedup core's input model."""

    return RawItem(
        item_id=item.item_id,
        ticker=item.ticker,
        title=item.title,
        description=item.description,
        url=item.url,
        canonical_url=item.canonical_url,
        source=item.source,
        published_at=item.published_at,
        provider_item_id=item.provider_item_id,
    )


def to_raw_items(pair: LabeledPair) -> tuple[RawItem, RawItem]:
    """Project both sides of a labeled pair."""

    return to_raw_item(pair.item_a), to_raw_item(pair.item_b)


def config_for(pair_set: PairSet) -> DedupConfig:
    """Build the dedup configuration the set declares.

    The ticker universe comes from the dataset manifest rather than a
    constant, so a set covering different symbols cannot be scored under a
    universe that silently rejects half of it.
    """

    return DedupConfig(supported_tickers=tuple(pair_set.metadata["tickers"]))


def m2_predictor(config: DedupConfig) -> PairPredictor:
    """Return a predictor that runs the M2 core over one pair at a time."""

    def predict(pair: LabeledPair) -> PairPrediction:
        left, right = to_raw_items(pair)
        result = deduplicate([left, right], config=config)
        merged_cluster = next(
            (cluster for cluster in result.clusters if len(cluster.member_ids) == 2),
            None,
        )
        veto_total = sum(count for _, count in result.stats.veto_counts)
        merged = merged_cluster is not None
        if merged_cluster is not None:
            reasons = ",".join(
                reason.value
                for reason in merged_cluster.match_reasons
                if reason in _MERGE_REASONS
            )
            detail = f"merged: {reasons or 'unattributed'}"
        elif result.provider_conflicts:
            detail = "not merged: provider identity quarantined"
        elif veto_total:
            detail = "not merged: " + ",".join(
                f"{reason}x{count}" for reason, count in result.stats.veto_counts
            )
        else:
            detail = "not merged: no signal"
        return PairPrediction(
            merged=merged,
            stage="m2" if merged else None,
            score=None,
            # M2 has no similarity score, so "considered" is whatever the
            # public counters expose: an edge that reached the gate, or a
            # MinHash candidate that was generated.
            candidate=bool(merged or veto_total or result.stats.candidate_pair_count),
            detail=detail,
        )

    return predict


def evaluate_m2(
    pair_set: PairSet | None = None,
    *,
    config: DedupConfig | None = None,
) -> EvaluationReport:
    """Score the merged M2 core against the labeled set."""

    pairs = pair_set if pair_set is not None else default_pair_set()
    settings = config if config is not None else config_for(pairs)
    return evaluate(pairs, m2_predictor(settings), name="m2")


def merged_pair_ids(reports: Sequence[EvaluationReport]) -> tuple[str, ...]:
    """Union of the pair ids any of ``reports`` merged, sorted."""

    merged: set[str] = set()
    for report in reports:
        merged.update(report.overall.confusion.true_positives)
        merged.update(report.overall.confusion.false_positives)
        merged.update(report.ambiguous_merged)
    return tuple(sorted(merged))

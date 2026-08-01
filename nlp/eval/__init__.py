"""Phase 0 evaluation tooling (M4, issue #67).

The labeled deduplication pair set and the deterministic evaluator that
measures a stage against it.  Nothing here decides whether two records are
duplicates: the evaluator calls the public stage APIs
(:func:`nlp.dedup.deduplicate`, and M3 once it exists) and only counts what
they return, so a metric can never drift away from shipped behaviour.

    from nlp.eval import default_pair_set, evaluate_m2

    pairs = default_pair_set()
    report = evaluate_m2(pairs)
    report.overall.precision, report.overall.recall

The dataset is **synthetic and clearly marked as such**; see
``nlp/eval/data/dedup_pairs.meta.json`` for why, and ``nlp/README.md`` for
what that means for the AC-3 numbers.
"""

from __future__ import annotations

from .dataset import (
    DEFAULT_META_PATH,
    SUPPORTED_SCHEMA_VERSION,
    EvalDatasetError,
    LabeledItem,
    LabeledPair,
    PairSet,
    default_pair_set,
    load_pair_set,
)
from .dedup import (
    PairPrediction,
    evaluate_m2,
    m2_predictor,
    to_raw_items,
)
from .metrics import (
    CategoryBreakdown,
    Confusion,
    EvaluationReport,
    Metrics,
    ThresholdPoint,
    evaluate,
    sweep_thresholds,
)

__all__ = [
    "DEFAULT_META_PATH",
    "SUPPORTED_SCHEMA_VERSION",
    "CategoryBreakdown",
    "Confusion",
    "EvalDatasetError",
    "EvaluationReport",
    "LabeledItem",
    "LabeledPair",
    "Metrics",
    "PairPrediction",
    "PairSet",
    "ThresholdPoint",
    "default_pair_set",
    "evaluate",
    "evaluate_m2",
    "load_pair_set",
    "m2_predictor",
    "sweep_thresholds",
    "to_raw_items",
]

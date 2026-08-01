"""Phase 0 evaluation tooling (M4, issue #67).

The labeled deduplication datasets and the deterministic evaluators that
measure a stage against them.  Nothing here decides whether two records are
duplicates: the evaluators call the public stage APIs
(:func:`nlp.dedup.deduplicate`, and M3 once it exists) and only count what
they return, so a metric can never drift away from shipped behaviour.

There are **two** measurements, and they are not interchangeable:

``isolated_pair_metrics``
    One pair at a time, two records per invocation.  Answers "does a
    two-item call merge this pair".

``multi_item_cluster_metrics``
    A whole group in one invocation, scored on the partition returned.
    This is the only one that can see cluster-wide compatibility,
    transitivity, provider quarantine, and window-on-span behaviour.

    from nlp.eval import default_pair_set, evaluate_m2_isolated_pairs
    from nlp.eval import default_cluster_cases, evaluate_m2_clusters

    report = evaluate_m2_isolated_pairs(default_pair_set())
    report.isolated_pair_metrics.precision

    clusters = evaluate_m2_clusters(default_cluster_cases())
    clusters.exact_partition_rate

Both datasets are **synthetic, single-author, unadjudicated development
sets**.  Their trust contract is enforced at load time and travels with
every report; see :mod:`nlp.eval.trust`.  The numbers are development
regression signals and are not valid for K3/G4 or final AC-3 acceptance.
"""

from __future__ import annotations

from .clusters import (
    DEFAULT_META_PATH as DEFAULT_CLUSTER_META_PATH,
    MULTI_ITEM_LIMITATION,
    AccountingViolation,
    CaseOutcome,
    ClusterCase,
    ClusterCaseSet,
    ClusterEvaluationReport,
    ClusterItem,
    CrossFixtureClaim,
    PartitionAccountingError,
    canonical_partition,
    check_cross_fixture_claims,
    default_cluster_cases,
    evaluate_clusters,
    evaluate_m2_clusters,
    load_cluster_cases,
    m2_cluster_predictor,
    permutations_of,
    validate_predicted_partition,
)
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
    evaluate_m2_isolated_pairs,
    m2_isolated_pair_predictor,
    m2_predictor,
    to_raw_items,
)
from .metrics import (
    ISOLATED_PAIR_LIMITATION,
    CategoryBreakdown,
    Confusion,
    EvaluationFailure,
    EvaluationReport,
    Metrics,
    ThresholdPoint,
    evaluate,
    evaluate_isolated_pairs,
    sweep_thresholds,
)
from .trust import (
    WARNING_BANNER,
    DatasetKind,
    LabelingStatus,
    MetricsPurpose,
    TrustContract,
    TrustContractError,
)
from .validation import (
    GateValueError,
    validate_optional_unit_interval,
    validate_thresholds,
    validate_unit_interval,
)

__all__ = [
    "AccountingViolation",
    "CaseOutcome",
    "CategoryBreakdown",
    "ClusterCase",
    "ClusterCaseSet",
    "ClusterEvaluationReport",
    "ClusterItem",
    "Confusion",
    "CrossFixtureClaim",
    "DEFAULT_CLUSTER_META_PATH",
    "DEFAULT_META_PATH",
    "DatasetKind",
    "EvalDatasetError",
    "EvaluationFailure",
    "EvaluationReport",
    "GateValueError",
    "ISOLATED_PAIR_LIMITATION",
    "LabeledItem",
    "LabeledPair",
    "LabelingStatus",
    "MULTI_ITEM_LIMITATION",
    "Metrics",
    "MetricsPurpose",
    "PairPrediction",
    "PairSet",
    "PartitionAccountingError",
    "SUPPORTED_SCHEMA_VERSION",
    "ThresholdPoint",
    "TrustContract",
    "TrustContractError",
    "WARNING_BANNER",
    "canonical_partition",
    "check_cross_fixture_claims",
    "default_cluster_cases",
    "default_pair_set",
    "evaluate",
    "evaluate_clusters",
    "evaluate_isolated_pairs",
    "evaluate_m2",
    "evaluate_m2_clusters",
    "evaluate_m2_isolated_pairs",
    "load_cluster_cases",
    "load_pair_set",
    "m2_cluster_predictor",
    "m2_isolated_pair_predictor",
    "m2_predictor",
    "permutations_of",
    "sweep_thresholds",
    "to_raw_items",
    "validate_optional_unit_interval",
    "validate_predicted_partition",
    "validate_thresholds",
    "validate_unit_interval",
]

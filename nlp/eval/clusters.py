"""Multi-item cluster evaluation: what pairwise scoring cannot see.

An isolated-pair metric invokes a stage on two records.  Production invokes
it on a day's batch, and several of M2's rules only exist in a batch:

* the compatibility gate is applied to the **whole prospective cluster**, so
  a third record can change whether two others merge;
* merges are **transitive**, so a chain can reach further than any single
  edge;
* **provider-conflict quarantine** looks at every record sharing a provider
  item id, so a third record under that id can suppress a merge the pair
  alone would make;
* the merge **window applies to the cluster's span**, not to a pair;
* **capacity** is a property of a partition, not of a pair.

None of that is observable two records at a time.  So each case here is run
as one group, in one call, and scored on the partition that comes back.

Two expectations are recorded per case and both are measured:

``expected_partition``      ground truth - how a reader groups the items
``exact_stage_partition``   what the exact stage alone should produce

They differ where the grouping needs semantics, and reporting both is what
keeps "M2 did not merge these two rewrites" legible as correct behaviour
rather than a defect.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from nlp.dedup import DedupConfig, RawItem, deduplicate

from .dataset import EvalDatasetError, validate_url
from .metrics import EvaluationFailure, Metrics, _ratio
from .trust import (
    TrustContract,
    parse_trust_contract,
    validate_labeling,
    validate_provenance,
)

SUPPORTED_SCHEMA_VERSION = "phase0.cluster_eval.v1"
DEFAULT_META_PATH = Path(__file__).resolve().parent / "data" / "cluster_cases.meta.json"

#: What a :class:`ClusterEvaluationReport` measures.
MULTI_ITEM_SCOPE = "multi_item_clusters"

MULTI_ITEM_LIMITATION = (
    "Multi-item cluster metrics run each case as one batch, so they do "
    "observe cluster-wide compatibility, transitivity, quarantine, and "
    "window-on-span behaviour that isolated-pair metrics cannot. They are "
    "still nine authored cases, not a sample of production traffic, and "
    "they are not evidence of production acceptance."
)

_REQUIRED_META_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "cases_file",
        "tickers",
        "statuses",
        "categories",
        "trust_contract",
        "provenance",
        "labeling",
    }
)
_REQUIRED_CASE_KEYS = frozenset(
    {
        "case_id",
        "ticker",
        "category",
        "status",
        "resolvable_by",
        "rationale",
        "expected_partition",
        "exact_stage_partition",
        "items",
    }
)
_ITEM_KEYS = frozenset(
    {
        "item_id",
        "ticker",
        "title",
        "description",
        "url",
        "canonical_url",
        "source",
        "published_at",
        "provider_item_id",
    }
)
_REQUIRED_ITEM_KEYS = frozenset({"item_id", "canonical_url"})


Partition = tuple[frozenset[str], ...]


def _as_partition(groups: Sequence[Sequence[str]], *, where: str) -> Partition:
    """Validate and canonicalize a list of groups into a partition."""

    if not isinstance(groups, list) or not groups:
        raise EvalDatasetError(f"{where}: a partition must be a non-empty list")
    seen: set[str] = set()
    result: list[frozenset[str]] = []
    for group in groups:
        if not isinstance(group, list) or not group:
            raise EvalDatasetError(f"{where}: every group must be a non-empty list")
        members = set()
        for item_id in group:
            if not isinstance(item_id, str) or not item_id.strip():
                raise EvalDatasetError(f"{where}: item ids must be non-blank strings")
            if item_id in seen:
                raise EvalDatasetError(
                    f"{where}: {item_id!r} appears in more than one group; a "
                    "partition places every item exactly once"
                )
            seen.add(item_id)
            members.add(item_id)
        result.append(frozenset(members))
    return canonical_partition(result)


def canonical_partition(groups: Sequence[frozenset[str]]) -> Partition:
    """Order a partition so two equal partitions compare equal."""

    return tuple(sorted(groups, key=lambda group: (len(group), sorted(group))))


@dataclass(frozen=True)
class ClusterItem:
    """One record inside a multi-item case."""

    item_id: str
    ticker: str
    title: str | None
    description: str | None
    url: str | None
    canonical_url: str | None
    source: str | None
    published_at: str | None
    provider_item_id: str | None
    synthetic: bool = True


@dataclass(frozen=True)
class ClusterCase:
    """One batch of records with the partition a reader would produce."""

    case_id: str
    ticker: str
    category: str
    #: ``decidable`` | ``ambiguous``; ambiguous cases are excluded from the
    #: headline partition metrics and reported separately.
    status: str
    #: ``m2`` | ``m3``: whose responsibility the ground-truth grouping is.
    resolvable_by: str
    rationale: str
    expected_partition: Partition
    exact_stage_partition: Partition
    items: tuple[ClusterItem, ...]
    synthetic: bool = True

    @property
    def item_ids(self) -> frozenset[str]:
        return frozenset(item.item_id for item in self.items)

    @property
    def is_scored(self) -> bool:
        return self.status == "decidable"


@dataclass(frozen=True)
class ClusterCaseSet:
    """A validated multi-item case set plus its manifest."""

    dataset_id: str
    schema_version: str
    trust: TrustContract
    metadata: Mapping[str, Any]
    cases: tuple[ClusterCase, ...]
    source_path: Path

    def __iter__(self) -> Iterator[ClusterCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def by_id(self, case_id: str) -> ClusterCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    def composition(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for field_name, getter in (
            ("category", lambda case: case.category),
            ("status", lambda case: case.status),
            ("resolvable_by", lambda case: case.resolvable_by),
            ("ticker", lambda case: case.ticker),
        ):
            tally: dict[str, int] = {}
            for case in self.cases:
                key = getter(case)
                tally[key] = tally.get(key, 0) + 1
            counts[field_name] = dict(sorted(tally.items()))
        counts["items"] = {"total": sum(len(case.items) for case in self.cases)}
        return counts


def _require_str(value: Any, field: str, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalDatasetError(f"{where}: {field} must be a non-blank string")
    return value


def _optional_str(value: Any, field: str, *, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise EvalDatasetError(f"{where}: {field} must be a non-blank string or null")
    return value


def _parse_item(
    payload: Any, *, case_id: str, ticker: str, synthetic: bool
) -> ClusterItem:
    where = f"{case_id}.{payload.get('item_id', '?')}"
    if not isinstance(payload, dict):
        raise EvalDatasetError(f"{case_id}: every item must be an object")
    unknown = sorted(set(payload) - _ITEM_KEYS)
    if unknown:
        raise EvalDatasetError(f"{where}: unknown field(s) {unknown}")
    missing = sorted(_REQUIRED_ITEM_KEYS - set(payload))
    if missing:
        raise EvalDatasetError(
            f"{where}: missing field(s) {missing}; canonical_url must be "
            "present even when null"
        )
    published_at = _optional_str(
        payload.get("published_at"), "published_at", where=where
    )
    if published_at is not None:
        from .dataset import _validate_timestamp

        _validate_timestamp(published_at, where=where)
    return ClusterItem(
        item_id=_require_str(payload.get("item_id"), "item_id", where=where),
        ticker=str(payload.get("ticker") or ticker),
        title=_optional_str(payload.get("title"), "title", where=where),
        description=_optional_str(
            payload.get("description"), "description", where=where
        ),
        url=validate_url(
            _optional_str(payload.get("url"), "url", where=where), "url", where=where
        ),
        canonical_url=validate_url(
            _optional_str(payload.get("canonical_url"), "canonical_url", where=where),
            "canonical_url",
            where=where,
        ),
        source=_optional_str(payload.get("source"), "source", where=where),
        published_at=published_at,
        provider_item_id=_optional_str(
            payload.get("provider_item_id"), "provider_item_id", where=where
        ),
        synthetic=synthetic,
    )


def _parse_case(
    payload: Any, metadata: Mapping[str, Any], *, line: int, synthetic: bool
) -> ClusterCase:
    if not isinstance(payload, dict):
        raise EvalDatasetError(f"line {line}: each row must be a JSON object")
    missing = sorted(_REQUIRED_CASE_KEYS - set(payload))
    if missing:
        raise EvalDatasetError(f"line {line}: missing field(s) {missing}")
    unknown = sorted(set(payload) - _REQUIRED_CASE_KEYS)
    if unknown:
        raise EvalDatasetError(f"line {line}: unknown field(s) {unknown}")
    case_id = _require_str(payload["case_id"], "case_id", where=f"line {line}")
    ticker = _require_str(payload["ticker"], "ticker", where=case_id)
    if ticker not in metadata["tickers"]:
        raise EvalDatasetError(f"{case_id}: ticker={ticker!r} is not supported")
    for field_name, vocabulary in (
        ("status", metadata["statuses"]),
        ("category", metadata["categories"]),
    ):
        if payload[field_name] not in vocabulary:
            raise EvalDatasetError(
                f"{case_id}: {field_name}={payload[field_name]!r} is not one of "
                f"{sorted(vocabulary)}"
            )
    if payload["resolvable_by"] not in {"m2", "m3"}:
        raise EvalDatasetError(
            f"{case_id}: resolvable_by must be 'm2' or 'm3', not "
            f"{payload['resolvable_by']!r}"
        )
    items = tuple(
        _parse_item(entry, case_id=case_id, ticker=ticker, synthetic=synthetic)
        for entry in payload["items"]
    )
    if len(items) < 3:
        raise EvalDatasetError(
            f"{case_id}: a multi-item case needs at least three items; a "
            "two-item case is a pair and belongs in the pair set"
        )
    seen: set[str] = set()
    for item in items:
        if item.item_id in seen:
            raise EvalDatasetError(f"{case_id}: duplicate item_id {item.item_id}")
        seen.add(item.item_id)

    expected = _as_partition(
        payload["expected_partition"], where=f"{case_id}.expected_partition"
    )
    exact = _as_partition(
        payload["exact_stage_partition"], where=f"{case_id}.exact_stage_partition"
    )
    for name, partition in (("expected", expected), ("exact_stage", exact)):
        covered = frozenset(itertools.chain.from_iterable(partition))
        if covered != frozenset(seen):
            raise EvalDatasetError(
                f"{case_id}: {name}_partition covers {sorted(covered)} but the "
                f"case holds {sorted(seen)}"
            )
    return ClusterCase(
        case_id=case_id,
        ticker=ticker,
        category=payload["category"],
        status=payload["status"],
        resolvable_by=payload["resolvable_by"],
        rationale=_require_str(payload["rationale"], "rationale", where=case_id),
        expected_partition=expected,
        exact_stage_partition=exact,
        items=items,
        synthetic=synthetic,
    )


def load_cluster_cases(
    meta_path: str | Path = DEFAULT_META_PATH,
) -> ClusterCaseSet:
    """Load and fully validate the multi-item cluster case set."""

    path = Path(meta_path).resolve()
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalDatasetError(f"{path}: manifest not found") from exc
    except json.JSONDecodeError as exc:
        raise EvalDatasetError(f"{path}: manifest is not valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise EvalDatasetError(f"{path}: manifest must be a JSON object")
    missing = sorted(_REQUIRED_META_KEYS - set(metadata))
    if missing:
        raise EvalDatasetError(f"{path}: manifest is missing {missing}")
    if metadata["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise EvalDatasetError(
            f"{path}: unsupported schema_version {metadata['schema_version']!r}"
        )
    trust = parse_trust_contract(metadata, where=str(path))
    validate_provenance(metadata, trust, where=str(path))
    validate_labeling(metadata, trust, where=str(path))

    cases_path = path.parent / str(metadata["cases_file"])
    try:
        body = cases_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvalDatasetError(f"{cases_path}: cases file not found") from exc

    cases: list[ClusterCase] = []
    seen_case_ids: set[str] = set()
    seen_item_ids: set[str] = set()
    for line_number, raw_line in enumerate(body.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(
                f"{cases_path} line {line_number}: not valid JSON: {exc}"
            ) from exc
        case = _parse_case(
            payload, metadata, line=line_number, synthetic=trust.is_synthetic
        )
        if case.case_id in seen_case_ids:
            raise EvalDatasetError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
        for item in case.items:
            if item.item_id in seen_item_ids:
                raise EvalDatasetError(f"duplicate item_id: {item.item_id}")
            seen_item_ids.add(item.item_id)
        cases.append(case)

    if not cases:
        raise EvalDatasetError(f"{cases_path}: holds no cases")
    identifiers = [case.case_id for case in cases]
    if identifiers != sorted(identifiers):
        raise EvalDatasetError(f"{cases_path}: rows must be sorted by case_id")
    return ClusterCaseSet(
        dataset_id=str(metadata["dataset_id"]),
        schema_version=str(metadata["schema_version"]),
        trust=trust,
        metadata=metadata,
        cases=tuple(cases),
        source_path=path,
    )


def default_cluster_cases() -> ClusterCaseSet:
    """Load the committed multi-item cluster case set."""

    return load_cluster_cases(DEFAULT_META_PATH)


def to_raw_items(case: ClusterCase) -> list[RawItem]:
    """Project a case's records onto the dedup core's input model."""

    return [
        RawItem(
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
        for item in case.items
    ]


#: A clusterer takes a whole case and returns the partition it produced.
ClusterPredictor = Callable[[ClusterCase], Partition]


def m2_cluster_predictor(config: DedupConfig) -> ClusterPredictor:
    """Return a predictor that runs the M2 core over a whole case at once."""

    def predict(case: ClusterCase) -> Partition:
        result = deduplicate(to_raw_items(case), config=config)
        return canonical_partition(
            [frozenset(cluster.member_ids) for cluster in result.clusters]
        )

    return predict


def _pair_set_of(partition: Partition) -> set[frozenset[str]]:
    """Every co-clustered pair implied by a partition."""

    return {
        frozenset(pair)
        for group in partition
        for pair in itertools.combinations(sorted(group), 2)
    }


@dataclass(frozen=True)
class CaseOutcome:
    """One case, scored against one of its two expectations."""

    case_id: str
    category: str
    status: str
    resolvable_by: str
    expected: Partition
    predicted: Partition
    exact_match: bool
    #: Predicted co-clusterings the expectation does not contain.
    over_merged_pairs: tuple[tuple[str, str], ...]
    #: Expected co-clusterings the prediction does not contain.
    under_merged_pairs: tuple[tuple[str, str], ...]
    missing_items: tuple[str, ...]
    duplicated_items: tuple[str, ...]
    permutation_stable: bool

    @property
    def accounted(self) -> bool:
        return not self.missing_items and not self.duplicated_items


@dataclass(frozen=True)
class ClusterEvaluationReport:
    """Multi-item scoring of one clusterer against the case set."""

    dataset_id: str
    predictor: str
    trust: TrustContract
    #: Which expectation this report scored against: ``expected_partition``
    #: (ground truth) or ``exact_stage_partition``.
    target: str
    outcomes: tuple[CaseOutcome, ...]
    #: Co-clustering precision/recall/F1 over every scored case's pairs.
    pairwise: Metrics
    exact_partition_matches: int
    scored_case_count: int
    ambiguous_case_count: int
    permutation_failures: tuple[str, ...]
    accounting_failures: tuple[str, ...]
    failures: tuple[EvaluationFailure, ...] = ()
    scope: str = MULTI_ITEM_SCOPE
    limitation: str = MULTI_ITEM_LIMITATION

    @property
    def exact_partition_rate(self) -> float | None:
        return _ratio(self.exact_partition_matches, self.scored_case_count)

    @property
    def over_merge_case_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                outcome.case_id
                for outcome in self.outcomes
                if outcome.status == "decidable" and outcome.over_merged_pairs
            )
        )

    @property
    def under_merge_case_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                outcome.case_id
                for outcome in self.outcomes
                if outcome.status == "decidable" and outcome.under_merged_pairs
            )
        )

    @property
    def evaluated_case_count(self) -> int:
        return len(self.outcomes)

    @property
    def failed_case_count(self) -> int:
        return len(self.failures)

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(failure.case_id for failure in self.failures))

    @property
    def complete(self) -> bool:
        return not self.failures


def _permutations_of(case: ClusterCase) -> list[ClusterCase]:
    """A reversed and a rotated ordering — never a random one."""

    items = list(case.items)
    orderings = [list(reversed(items))]
    if len(items) > 2:
        orderings.append(items[1:] + items[:1])
    from dataclasses import replace

    return [replace(case, items=tuple(ordering)) for ordering in orderings]


def evaluate_clusters(
    case_set: ClusterCaseSet,
    predictor: ClusterPredictor,
    *,
    name: str,
    target: str = "exact_stage_partition",
) -> ClusterEvaluationReport:
    """Run every case as one batch and score the partitions that come back.

    ``target`` selects which committed expectation to score against:
    ``exact_stage_partition`` for what M2 alone should do, or
    ``expected_partition`` for the ground truth a reader would produce.

    A case whose stage raises is recorded in ``failures``, excluded from
    every denominator, and marks the report incomplete.
    """

    if target not in {"expected_partition", "exact_stage_partition"}:
        raise ValueError(f"unknown target: {target!r}")

    outcomes: list[CaseOutcome] = []
    failures: list[EvaluationFailure] = []
    for case in case_set.cases:
        expected = getattr(case, target)
        try:
            predicted = canonical_partition(predictor(case))
            stable = all(
                canonical_partition(predictor(shuffled)) == predicted
                for shuffled in _permutations_of(case)
            )
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            failures.append(EvaluationFailure.of(case.case_id, error))
            continue

        predicted_items = [item_id for group in predicted for item_id in sorted(group)]
        counts: dict[str, int] = {}
        for item_id in predicted_items:
            counts[item_id] = counts.get(item_id, 0) + 1
        expected_pairs = _pair_set_of(expected)
        predicted_pairs = _pair_set_of(predicted)
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                category=case.category,
                status=case.status,
                resolvable_by=case.resolvable_by,
                expected=expected,
                predicted=predicted,
                exact_match=predicted == expected,
                over_merged_pairs=tuple(
                    tuple(sorted(pair))  # type: ignore[misc]
                    for pair in sorted(
                        predicted_pairs - expected_pairs, key=lambda p: sorted(p)
                    )
                ),
                under_merged_pairs=tuple(
                    tuple(sorted(pair))  # type: ignore[misc]
                    for pair in sorted(
                        expected_pairs - predicted_pairs, key=lambda p: sorted(p)
                    )
                ),
                missing_items=tuple(sorted(case.item_ids - set(counts))),
                duplicated_items=tuple(
                    sorted(item_id for item_id, count in counts.items() if count > 1)
                ),
                permutation_stable=stable,
            )
        )

    scored = [outcome for outcome in outcomes if outcome.status == "decidable"]
    true_positives: list[str] = []
    false_positives: list[str] = []
    false_negatives: list[str] = []
    true_negatives: list[str] = []
    for outcome in scored:
        expected_pairs = _pair_set_of(outcome.expected)
        predicted_pairs = _pair_set_of(outcome.predicted)
        all_pairs = {
            frozenset(pair)
            for pair in itertools.combinations(
                sorted(itertools.chain.from_iterable(outcome.expected)), 2
            )
        }
        for pair in sorted(all_pairs, key=lambda p: sorted(p)):
            label = "|".join(sorted(pair))
            if pair in expected_pairs and pair in predicted_pairs:
                true_positives.append(f"{outcome.case_id}:{label}")
            elif pair in predicted_pairs:
                false_positives.append(f"{outcome.case_id}:{label}")
            elif pair in expected_pairs:
                false_negatives.append(f"{outcome.case_id}:{label}")
            else:
                true_negatives.append(f"{outcome.case_id}:{label}")

    from .metrics import Confusion

    return ClusterEvaluationReport(
        dataset_id=case_set.dataset_id,
        predictor=name,
        trust=case_set.trust,
        target=target,
        outcomes=tuple(outcomes),
        pairwise=Metrics.from_confusion(
            Confusion(
                true_positives=tuple(sorted(true_positives)),
                false_positives=tuple(sorted(false_positives)),
                true_negatives=tuple(sorted(true_negatives)),
                false_negatives=tuple(sorted(false_negatives)),
            )
        ),
        exact_partition_matches=sum(1 for outcome in scored if outcome.exact_match),
        scored_case_count=len(scored),
        ambiguous_case_count=sum(
            1 for outcome in outcomes if outcome.status == "ambiguous"
        ),
        permutation_failures=tuple(
            sorted(
                outcome.case_id
                for outcome in outcomes
                if not outcome.permutation_stable
            )
        ),
        accounting_failures=tuple(
            sorted(outcome.case_id for outcome in outcomes if not outcome.accounted)
        ),
        failures=tuple(failures),
    )


def evaluate_m2_clusters(
    case_set: ClusterCaseSet | None = None,
    *,
    config: DedupConfig | None = None,
    target: str = "exact_stage_partition",
) -> ClusterEvaluationReport:
    """Score the merged M2 core against the multi-item case set."""

    cases = case_set if case_set is not None else default_cluster_cases()
    settings = config or DedupConfig(supported_tickers=tuple(cases.metadata["tickers"]))
    return evaluate_clusters(
        cases, m2_cluster_predictor(settings), name="m2", target=target
    )

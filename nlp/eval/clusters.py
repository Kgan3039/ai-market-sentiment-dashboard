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

Three expectations are recorded per case:

``expected_partition``      ground truth over the *determinate* items
``indeterminate_item_ids``  items the records do not place; never scored
``exact_stage_partition``   what the exact stage alone should produce

Keeping them apart is the point.  Ground truth is what a reader says about
the articles.  ``exact_stage_partition`` is what an implementation does with
them, including where a policy deliberately trades recall for safety, and
including where an item has to be put somewhere even though the evidence
does not say where.  Recording an implementation's traversal order as human
truth would make the fixture agree with the code by construction.

Accounting is checked **before** any metric.  A partition that invents an
item id, drops one, repeats one across groups, or contains an empty group
is not a worse answer to score — it is not an answer to the question, and
the case is failed rather than given credit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterator, Mapping, Sequence

from nlp.dedup import DedupConfig, RawItem, deduplicate

from .dataset import EvalDatasetError, PairSet, default_pair_set, validate_url
from .metrics import Confusion, EvaluationFailure, Metrics, _ratio
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

#: Run every ordering when the case is small enough that the factorial is
#: affordable; above it, fall back to a documented representative set.
MAX_EXHAUSTIVE_PERMUTATIONS = 120
#: Deterministic shuffles added to the representative set for larger cases.
REPRESENTATIVE_SHUFFLES = 8

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
        "indeterminate_item_ids",
        "expected_partition",
        "exact_stage_partition",
        "cross_fixture_claims",
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
_CLAIM_KEYS = frozenset(
    {"pair_id", "item_ids", "relationship", "note", "divergence_reason"}
)
_RELATIONSHIPS = frozenset({"same_story", "different_story"})
#: A pair label maps onto exactly one claimable relationship; an ambiguous
#: pair maps onto none, so a cluster case may not borrow its authority.
_RELATIONSHIP_BY_LABEL = {
    "duplicate": "same_story",
    "distinct": "different_story",
}


Partition = tuple[frozenset[str], ...]


class PartitionAccountingError(ValueError):
    """A predicted partition is not a partition of the case's items.

    Carries the three id sets so a report can say exactly what went wrong
    rather than only that something did.
    """

    def __init__(
        self,
        message: str,
        *,
        missing_item_ids: tuple[str, ...] = (),
        duplicated_item_ids: tuple[str, ...] = (),
        unexpected_item_ids: tuple[str, ...] = (),
        empty_cluster_count: int = 0,
    ) -> None:
        super().__init__(message)
        self.missing_item_ids = missing_item_ids
        self.duplicated_item_ids = duplicated_item_ids
        self.unexpected_item_ids = unexpected_item_ids
        self.empty_cluster_count = empty_cluster_count


def canonical_partition(groups: Sequence[frozenset[str]]) -> Partition:
    """Order a partition so two equal partitions compare equal."""

    return tuple(sorted(groups, key=lambda group: (len(group), sorted(group))))


def validate_predicted_partition(
    groups: Any, expected_item_ids: frozenset[str], *, where: str
) -> Partition:
    """Check a predicted partition against the case's item universe.

    Raises :class:`PartitionAccountingError` for a group that is not a
    collection, an empty group, a blank id, an id the case does not
    contain, an id in two groups, or an id the case contains that the
    partition does not.  Nothing downstream sees a partition that failed
    here, so an invented id can never earn exact-partition credit or a
    perfect co-clustering score.
    """

    if isinstance(groups, (str, bytes)) or not isinstance(groups, (list, tuple)):
        raise PartitionAccountingError(
            f"{where}: a partition must be a sequence of groups, got "
            f"{type(groups).__name__}"
        )
    seen: dict[str, int] = {}
    empty_clusters = 0
    normalized: list[frozenset[str]] = []
    for group in groups:
        if isinstance(group, (str, bytes)) or not isinstance(
            group, (list, tuple, set, frozenset)
        ):
            raise PartitionAccountingError(
                f"{where}: every group must be a collection of item ids, got "
                f"{type(group).__name__}"
            )
        members = list(group)
        if not members:
            empty_clusters += 1
            continue
        for item_id in members:
            if not isinstance(item_id, str) or not item_id.strip():
                raise PartitionAccountingError(
                    f"{where}: item ids must be non-blank strings, got {item_id!r}"
                )
            seen[item_id] = seen.get(item_id, 0) + 1
        normalized.append(frozenset(members))

    duplicated = tuple(sorted(item for item, count in seen.items() if count > 1))
    unexpected = tuple(sorted(set(seen) - expected_item_ids))
    missing = tuple(sorted(expected_item_ids - set(seen)))
    if empty_clusters or duplicated or unexpected or missing:
        parts = []
        if missing:
            parts.append(f"missing {list(missing)}")
        if duplicated:
            parts.append(f"in more than one group {list(duplicated)}")
        if unexpected:
            parts.append(f"not in the case {list(unexpected)}")
        if empty_clusters:
            parts.append(f"{empty_clusters} empty group(s)")
        raise PartitionAccountingError(
            f"{where}: predicted partition does not account for the case's "
            f"items: {'; '.join(parts)}",
            missing_item_ids=missing,
            duplicated_item_ids=duplicated,
            unexpected_item_ids=unexpected,
            empty_cluster_count=empty_clusters,
        )
    return canonical_partition(normalized)


def _as_expected_partition(
    groups: Any, universe: frozenset[str], *, where: str
) -> Partition:
    """Validate a *committed* partition from the fixture."""

    try:
        return validate_predicted_partition(groups, universe, where=where)
    except PartitionAccountingError as exc:
        raise EvalDatasetError(str(exc)) from exc


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
class CrossFixtureClaim:
    """A relationship this case shares with a pair in the pair set."""

    pair_id: str
    item_ids: tuple[str, str]
    relationship: str
    note: str = ""
    #: Set only when the case deliberately disagrees with the pair. Without
    #: it a disagreement is a defect, not a decision.
    divergence_reason: str = ""


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
    #: Items the records do not place. Excluded from ground-truth scoring.
    indeterminate_item_ids: frozenset[str]
    expected_partition: Partition
    exact_stage_partition: Partition
    cross_fixture_claims: tuple[CrossFixtureClaim, ...]
    items: tuple[ClusterItem, ...]
    synthetic: bool = True

    @property
    def item_ids(self) -> frozenset[str]:
        return frozenset(item.item_id for item in self.items)

    @property
    def determinate_item_ids(self) -> frozenset[str]:
        return self.item_ids - self.indeterminate_item_ids

    @property
    def is_scored(self) -> bool:
        return self.status == "decidable"

    def universe_for(self, target: str) -> frozenset[str]:
        """Which items a partition for ``target`` is expected to cover."""

        return (
            self.determinate_item_ids
            if target == "expected_partition"
            else self.item_ids
        )


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
        counts["items"] = {
            "total": sum(len(case.items) for case in self.cases),
            "indeterminate": sum(
                len(case.indeterminate_item_ids) for case in self.cases
            ),
        }
        counts["cross_fixture_claims"] = {
            "total": sum(len(case.cross_fixture_claims) for case in self.cases)
        }
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
    if not isinstance(payload, dict):
        raise EvalDatasetError(f"{case_id}: every item must be an object")
    where = f"{case_id}.{payload.get('item_id', '?')}"
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


def _parse_claim(payload: Any, *, case_id: str, item_ids: frozenset[str]):
    if not isinstance(payload, dict):
        raise EvalDatasetError(
            f"{case_id}: every cross_fixture_claim must be an object"
        )
    unknown = sorted(set(payload) - _CLAIM_KEYS)
    if unknown:
        raise EvalDatasetError(f"{case_id}: unknown claim field(s) {unknown}")
    pair_id = _require_str(payload.get("pair_id"), "pair_id", where=case_id)
    members = payload.get("item_ids")
    if not isinstance(members, list) or len(members) != 2:
        raise EvalDatasetError(
            f"{case_id}: claim for {pair_id} must name exactly two item ids"
        )
    for item_id in members:
        if item_id not in item_ids:
            raise EvalDatasetError(
                f"{case_id}: claim for {pair_id} names {item_id!r}, which is "
                "not in the case"
            )
    relationship = _require_str(
        payload.get("relationship"), "relationship", where=case_id
    )
    if relationship not in _RELATIONSHIPS:
        raise EvalDatasetError(
            f"{case_id}: claim relationship={relationship!r} is not one of "
            f"{sorted(_RELATIONSHIPS)}"
        )
    return CrossFixtureClaim(
        pair_id=pair_id,
        item_ids=(str(members[0]), str(members[1])),
        relationship=relationship,
        note=str(payload.get("note") or ""),
        divergence_reason=str(payload.get("divergence_reason") or ""),
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
    universe = frozenset(seen)

    indeterminate = payload["indeterminate_item_ids"]
    if not isinstance(indeterminate, list) or any(
        item_id not in universe for item_id in indeterminate
    ):
        raise EvalDatasetError(
            f"{case_id}: indeterminate_item_ids must be a list of the case's "
            f"own item ids, got {indeterminate!r}"
        )
    indeterminate_ids = frozenset(indeterminate)
    determinate = universe - indeterminate_ids
    if not determinate:
        raise EvalDatasetError(
            f"{case_id}: every item is indeterminate; the case asserts nothing"
        )

    expected = _as_expected_partition(
        payload["expected_partition"],
        determinate,
        where=f"{case_id}.expected_partition",
    )
    exact = _as_expected_partition(
        payload["exact_stage_partition"],
        universe,
        where=f"{case_id}.exact_stage_partition",
    )
    claims = payload["cross_fixture_claims"]
    if not isinstance(claims, list):
        raise EvalDatasetError(f"{case_id}: cross_fixture_claims must be a list")
    return ClusterCase(
        case_id=case_id,
        ticker=ticker,
        category=payload["category"],
        status=payload["status"],
        resolvable_by=payload["resolvable_by"],
        rationale=_require_str(payload["rationale"], "rationale", where=case_id),
        indeterminate_item_ids=indeterminate_ids,
        expected_partition=expected,
        exact_stage_partition=exact,
        cross_fixture_claims=tuple(
            _parse_claim(claim, case_id=case_id, item_ids=universe) for claim in claims
        ),
        items=items,
        synthetic=synthetic,
    )


def _together(partition: Partition, left: str, right: str) -> bool:
    return any({left, right} <= group for group in partition)


def check_cross_fixture_claims(case: ClusterCase, pair_set: PairSet) -> None:
    """Refuse a cluster case that contradicts a pair without saying so.

    Two fixtures describing the same relationship must give it the same
    answer.  Where a case genuinely has to differ, ``divergence_reason``
    makes the disagreement a documented decision instead of an accident
    nobody sees.
    """

    for claim in case.cross_fixture_claims:
        try:
            pair = pair_set.by_id(claim.pair_id)
        except KeyError as exc:
            raise EvalDatasetError(
                f"{case.case_id}: cross-fixture claim names {claim.pair_id}, "
                "which is not in the pair set"
            ) from exc
        left, right = claim.item_ids
        if left in case.indeterminate_item_ids or right in case.indeterminate_item_ids:
            raise EvalDatasetError(
                f"{case.case_id}: cross-fixture claim for {claim.pair_id} names "
                "an indeterminate item; a claim must be about items the case "
                "actually places"
            )
        implied = (
            "same_story"
            if _together(case.expected_partition, left, right)
            else ("different_story")
        )
        if implied != claim.relationship:
            raise EvalDatasetError(
                f"{case.case_id}: claim for {claim.pair_id} says "
                f"{claim.relationship!r} but expected_partition places "
                f"{left} and {right} as {implied!r}"
            )
        expected_relationship = _RELATIONSHIP_BY_LABEL.get(pair.label)
        if expected_relationship is None:
            raise EvalDatasetError(
                f"{case.case_id}: {claim.pair_id} is labelled {pair.label!r}; a "
                "cluster case may not claim a decided relationship from a pair "
                "the records do not decide"
            )
        if expected_relationship != claim.relationship and not claim.divergence_reason:
            raise EvalDatasetError(
                f"{case.case_id}: claim for {claim.pair_id} says "
                f"{claim.relationship!r} but the pair is labelled {pair.label!r} "
                f"({expected_relationship!r}); a cluster fixture may not "
                "contradict a pair fixture without a divergence_reason"
            )


def load_cluster_cases(
    meta_path: str | Path = DEFAULT_META_PATH,
    *,
    pair_set: PairSet | None = None,
) -> ClusterCaseSet:
    """Load and fully validate the multi-item cluster case set.

    ``pair_set`` supplies the fixtures the cross-fixture claims are checked
    against; pass ``None`` to skip that check only when there is no pair
    set to check against.
    """

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

    if pair_set is not None:
        for case in cases:
            check_cross_fixture_claims(case, pair_set)

    return ClusterCaseSet(
        dataset_id=str(metadata["dataset_id"]),
        schema_version=str(metadata["schema_version"]),
        trust=trust,
        metadata=metadata,
        cases=tuple(cases),
        source_path=path,
    )


def default_cluster_cases() -> ClusterCaseSet:
    """Load the committed cluster cases, cross-checked against the pair set."""

    return load_cluster_cases(DEFAULT_META_PATH, pair_set=default_pair_set())


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
ClusterPredictor = Callable[[ClusterCase], Any]


def m2_cluster_predictor(config: DedupConfig) -> ClusterPredictor:
    """Return a predictor that runs the M2 core over a whole case at once."""

    def predict(case: ClusterCase) -> Partition:
        result = deduplicate(to_raw_items(case), config=config)
        return tuple(frozenset(cluster.member_ids) for cluster in result.clusters)

    return predict


def permutations_of(case: ClusterCase) -> tuple[tuple[ClusterItem, ...], ...]:
    """Return the orderings this case is checked under.

    Exhaustive while the factorial stays under
    :data:`MAX_EXHAUSTIVE_PERMUTATIONS`, which covers every Phase 0 fixture.
    Above that the set is the original, the reverse, every cyclic rotation,
    and :data:`REPRESENTATIVE_SHUFFLES` shuffles seeded on the case id — so
    it is documented, deterministic, and reproducible from the case alone
    rather than from a clock or a global random state.
    """

    items = tuple(case.items)
    if math.factorial(len(items)) <= MAX_EXHAUSTIVE_PERMUTATIONS:
        return tuple(itertools.permutations(items))
    orderings: list[tuple[ClusterItem, ...]] = [items, tuple(reversed(items))]
    orderings += [items[offset:] + items[:offset] for offset in range(1, len(items))]
    generator = random.Random(case.case_id)
    for _ in range(REPRESENTATIVE_SHUFFLES):
        shuffled = list(items)
        generator.shuffle(shuffled)
        orderings.append(tuple(shuffled))
    return tuple(dict.fromkeys(orderings))


def _pair_set_of(partition: Partition, universe: frozenset[str]) -> set[frozenset[str]]:
    """Every co-clustered pair implied by a partition, restricted to ``universe``."""

    return {
        frozenset(pair)
        for group in partition
        for pair in itertools.combinations(sorted(group & universe), 2)
    }


def _restrict(partition: Partition, universe: frozenset[str]) -> Partition:
    return canonical_partition(
        [group & universe for group in partition if group & universe]
    )


@dataclass(frozen=True)
class CaseOutcome:
    """One case, scored against one of its expectations."""

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
    indeterminate_item_ids: tuple[str, ...]
    permutation_count: int
    permutation_stable: bool
    #: Orderings whose partition differed from the canonical run.
    unstable_permutation_count: int


@dataclass(frozen=True)
class AccountingViolation:
    """A case whose predicted partition was not a partition of its items."""

    case_id: str
    missing_item_ids: tuple[str, ...]
    duplicated_item_ids: tuple[str, ...]
    unexpected_item_ids: tuple[str, ...]
    empty_cluster_count: int
    message: str


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
    accounting_violations: tuple[AccountingViolation, ...]
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
    def accounting_failure_ids(self) -> tuple[str, ...]:
        return tuple(sorted(entry.case_id for entry in self.accounting_violations))

    @property
    def missing_item_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item_id
                    for entry in self.accounting_violations
                    for item_id in entry.missing_item_ids
                }
            )
        )

    @property
    def duplicated_item_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item_id
                    for entry in self.accounting_violations
                    for item_id in entry.duplicated_item_ids
                }
            )
        )

    @property
    def unexpected_item_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item_id
                    for entry in self.accounting_violations
                    for item_id in entry.unexpected_item_ids
                }
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

    A case whose stage raises, or whose predicted partition does not
    account for exactly the case's items, is recorded in ``failures``,
    excluded from every denominator, and marks the report incomplete.
    """

    if target not in {"expected_partition", "exact_stage_partition"}:
        raise ValueError(f"unknown target: {target!r}")

    outcomes: list[CaseOutcome] = []
    failures: list[EvaluationFailure] = []
    violations: list[AccountingViolation] = []
    for case in case_set.cases:
        expected = getattr(case, target)
        universe = case.universe_for(target)
        try:
            orderings = permutations_of(case)
            partitions = [
                validate_predicted_partition(
                    predictor(replace(case, items=ordering)),
                    case.item_ids,
                    where=f"{case.case_id}[{index}]",
                )
                for index, ordering in enumerate(orderings)
            ]
        except PartitionAccountingError as error:
            violations.append(
                AccountingViolation(
                    case_id=case.case_id,
                    missing_item_ids=error.missing_item_ids,
                    duplicated_item_ids=error.duplicated_item_ids,
                    unexpected_item_ids=error.unexpected_item_ids,
                    empty_cluster_count=error.empty_cluster_count,
                    message=str(error),
                )
            )
            failures.append(EvaluationFailure.of(case.case_id, error))
            continue
        except Exception as error:  # noqa: BLE001 - reported, never swallowed
            failures.append(EvaluationFailure.of(case.case_id, error))
            continue

        predicted = partitions[0]
        unstable = sum(1 for other in partitions[1:] if other != predicted)
        restricted = _restrict(predicted, universe)
        expected_pairs = _pair_set_of(expected, universe)
        predicted_pairs = _pair_set_of(predicted, universe)
        outcomes.append(
            CaseOutcome(
                case_id=case.case_id,
                category=case.category,
                status=case.status,
                resolvable_by=case.resolvable_by,
                expected=expected,
                predicted=restricted,
                exact_match=restricted == expected,
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
                indeterminate_item_ids=tuple(sorted(case.indeterminate_item_ids)),
                permutation_count=len(orderings),
                permutation_stable=unstable == 0,
                unstable_permutation_count=unstable,
            )
        )

    scored = [outcome for outcome in outcomes if outcome.status == "decidable"]
    true_positives: list[str] = []
    false_positives: list[str] = []
    false_negatives: list[str] = []
    true_negatives: list[str] = []
    for outcome in scored:
        universe = frozenset(itertools.chain.from_iterable(outcome.expected))
        expected_pairs = _pair_set_of(outcome.expected, universe)
        predicted_pairs = _pair_set_of(outcome.predicted, universe)
        for pair in sorted(
            (frozenset(pair) for pair in itertools.combinations(sorted(universe), 2)),
            key=lambda p: sorted(p),
        ):
            label = "|".join(sorted(pair))
            if pair in expected_pairs and pair in predicted_pairs:
                true_positives.append(f"{outcome.case_id}:{label}")
            elif pair in predicted_pairs:
                false_positives.append(f"{outcome.case_id}:{label}")
            elif pair in expected_pairs:
                false_negatives.append(f"{outcome.case_id}:{label}")
            else:
                true_negatives.append(f"{outcome.case_id}:{label}")

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
        accounting_violations=tuple(violations),
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

"""Duplicate detection: identity signals, one compatibility gate, union-find.

Issue #64 splits M2 into stage 1 (canonical-URL and normalized-title exact
match) and stage 2 (MinHash over title shingles).  Both are here.

====================================================  ===========  ========
Signal                                                Window       Gated
====================================================  ===========  ========
provider namespace + provider item id (unconflicted)  none         no
URL identity key, corroborated, both dated            url          yes
identical normalized title *and* description          content      yes
identical normalized title                            exact_title  yes
MinHash candidate verified structurally identical     near_exact   yes
====================================================  ===========  ========

Provider identity is the only *authoritative* signal: one feed asserting
that two records are its same item.  It is not time-bounded and not gated —
but it applies only while the identity is consistent.  A provider item id
that carried conflicting payloads quarantines every item under it, and
those items then merge through no signal at all.

Every other signal is circumstantial, so every other edge passes the shared
compatibility gate before it is unioned.  The gate is applied once,
centrally, inside :meth:`_ConstrainedDisjointSet.union` — never inside an
individual signal — so no tier can drift into a weaker check of its own,
and it is applied to the **whole prospective cluster** rather than the two
endpoints of the edge.  Endpoint-only checking let a record carrying no
description bridge two records whose descriptions contradicted each other.

Merges never cross a ticker partition, and every window is applied to the
*span of the resulting cluster*, so transitive chaining cannot widen a
merge beyond the configured window.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import itertools
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from .compatibility import (
    PROVIDER_EXTRA_CHECKS,
    VETO_REASONS,
    EvidenceSummary,
    combine,
    summarize,
)
from .config import DedupConfig
from .errors import DedupCapacityError
from .minhash import estimate_similarity, shingles, signature
from .models import MatchReason, NormalizedItem, ProviderConflict, REASON_RANK

_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)


KeyGetter = Callable[[NormalizedItem], str | None]


@dataclass(frozen=True)
class CandidateEdge:
    """One proposed merge between two items in the same partition."""

    left: int
    right: int
    reason: MatchReason


@dataclass(frozen=True)
class DuplicateGroup:
    """One detected cluster: member item indices plus why they joined."""

    ticker: str
    #: Input indices ordered by (published_at, item_id), unknown last.
    member_indices: tuple[int, ...]
    #: Strongest accepted merge reason per member index.
    member_reasons: tuple[tuple[int, MatchReason], ...]
    #: Every distinct reason that merged this group, strongest first.
    reasons: tuple[MatchReason, ...]


@dataclass(frozen=True)
class DetectionReport:
    """Detection output plus counters."""

    groups: tuple[DuplicateGroup, ...]
    reason_counts: tuple[tuple[str, int], ...]
    veto_counts: tuple[tuple[str, int], ...]
    provider_conflicts: tuple[ProviderConflict, ...]
    quarantined_item_ids: tuple[str, ...]
    candidate_pair_count: int
    verified_candidate_pair_count: int


# --------------------------------------------------------------------------
# Provider-identity conflicts
# --------------------------------------------------------------------------


def _conflicting_fields(
    members: Sequence[NormalizedItem], config: DedupConfig
) -> tuple[str, ...]:
    """Return the sorted reasons the records under one provider id disagree.

    Text evidence is compared with the *same* primitive the compatibility
    gate uses, so provider quarantine can never be the weaker of the two.
    Every pair is compared, so a three-record group where only one pair
    conflicts is still quarantined as a whole.  On top of that, provider
    identity applies three checks ordinary compatibility does not: an item
    id must not move between URLs, between tickers, or across time.
    """

    fields: set[str] = set()
    for left, right in itertools.combinations(members, 2):
        reason = combine(summarize(left), summarize(right))[1]
        if reason is not None:
            fields.add(reason)
    # The stricter-than-compatibility checks, named in PROVIDER_EXTRA_CHECKS.
    urls = {item.url_key for item in members if item.url_key is not None}
    if len(urls) > 1:
        fields.add("url")
    if len({item.ticker for item in members}) > 1:
        fields.add("ticker")
    stamps = sorted(item.published_at for item in members if item.published_at)
    if stamps and stamps[-1] - stamps[0] > config.provider_timestamp_tolerance:
        fields.add("published_at")
    assert fields <= set(PROVIDER_EXTRA_CHECKS) | set(VETO_REASONS)
    return tuple(sorted(fields))


def provider_conflicts(
    items: Sequence[NormalizedItem], config: DedupConfig
) -> tuple[tuple[ProviderConflict, ...], frozenset[int]]:
    """Return provider conflicts and the item indices they quarantine.

    When one ``(source, provider_item_id)`` pair describes two different
    articles the feed is wrong about something, and the core cannot tell
    which payload is right.  Rather than guess, every item under that
    identity is quarantined: it merges through no signal, and is emitted as
    its own cluster so no coverage is lost.
    """

    grouped: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        if item.provider_key is not None:
            grouped.setdefault(item.provider_key, []).append(index)
    conflicts: list[ProviderConflict] = []
    quarantined: set[int] = set()
    for key in sorted(grouped):
        indices = grouped[key]
        if len(indices) < 2:
            continue
        members = [items[index] for index in indices]
        fields = _conflicting_fields(members, config)
        if not fields:
            continue
        quarantined.update(indices)
        conflicts.append(
            ProviderConflict(
                provider_namespace=members[0].provider_namespace or "",
                provider_item_id=members[0].provider_item_id or "",
                item_ids=tuple(sorted(item.item_id for item in members)),
                fields=fields,
            )
        )
    conflicts.sort(key=lambda entry: (entry.provider_namespace, entry.provider_item_id))
    return tuple(conflicts), frozenset(quarantined)


# --------------------------------------------------------------------------
# Cluster-span constrained union-find
# --------------------------------------------------------------------------


def _order_key(item: NormalizedItem) -> tuple[bool, datetime, str]:
    """Total, permutation-invariant ordering: earliest first, id breaks ties."""

    published_at = item.published_at
    return (published_at is None, published_at or _EPOCH, item.item_id)


class _ConstrainedDisjointSet:
    """Union-find that enforces the window *and* cluster-wide compatibility.

    Each component carries the evidence summary of everything in it, so a
    union is checked against the whole prospective cluster rather than the
    two endpoints of one edge.  Without that, a record carrying no
    description could bridge two records whose descriptions flatly
    contradict each other.
    """

    def __init__(
        self,
        timestamps: Sequence[datetime | None],
        summaries: Sequence[EvidenceSummary],
    ) -> None:
        self._parent = list(range(len(timestamps)))
        self._size = [1] * len(timestamps)
        self._earliest: list[datetime | None] = list(timestamps)
        self._latest: list[datetime | None] = list(timestamps)
        self._summary: list[EvidenceSummary] = list(summaries)

    def find(self, index: int) -> int:
        root = index
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[index] != root:
            self._parent[index], index = root, self._parent[index]
        return root

    def union(
        self, left: int, right: int, window: timedelta | None, *, gated: bool
    ) -> tuple[bool, str | None]:
        """Merge the two components.

        Returns ``(merged, veto_reason)``.  ``gated`` is False only for the
        authoritative provider signal, whose groups have already survived
        conflict quarantine.
        """

        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False, None
        merged_summary, veto = combine(
            self._summary[left_root], self._summary[right_root]
        )
        if gated and veto is not None:
            return False, veto
        earliest = _minimum(self._earliest[left_root], self._earliest[right_root])
        latest = _maximum(self._latest[left_root], self._latest[right_root])
        if (
            window is not None
            and earliest is not None
            and latest is not None
            and latest - earliest > window
        ):
            return False, None
        # Merge the smaller cluster into the larger one; the lower root index
        # wins ties so the structure never depends on input order.
        if self._size[left_root] < self._size[right_root] or (
            self._size[left_root] == self._size[right_root] and right_root < left_root
        ):
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]
        self._earliest[left_root] = earliest
        self._latest[left_root] = latest
        if merged_summary is not None:
            self._summary[left_root] = merged_summary
        return True, None


def _minimum(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    return left if right is None else min(left, right)


def _maximum(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    return left if right is None else max(left, right)


# --------------------------------------------------------------------------
# Stage 1: exact identity signals
# --------------------------------------------------------------------------


def _bucket_edges(
    items: Sequence[NormalizedItem],
    ordered: Sequence[int],
    key: KeyGetter,
    reason: MatchReason,
    *,
    requires_timestamp: bool,
) -> list[CandidateEdge]:
    """Propose every unique pair inside each exact-key bucket.

    An earlier version chained only time-adjacent members, on the argument
    that the window applies to the cluster span so skipped pairs were
    already covered.  That reasoning fails once an edge can be *vetoed*: if
    A and C share a key and are compatible, but an incompatible B sorts
    between them, chaining proposes only A-B and B-C and never considers
    A-C.  Buckets are bounded by the per-ticker capacity contract, so all
    unique pairs is affordable and complete.
    """

    groups: dict[str, list[int]] = {}
    for index in ordered:
        item = items[index]
        if requires_timestamp and item.published_at is None:
            continue
        group_key = key(item)
        if group_key is not None:
            groups.setdefault(group_key, []).append(index)
    return [
        CandidateEdge(left, right, reason)
        for group_key in sorted(groups)
        for left, right in itertools.combinations(groups[group_key], 2)
    ]


def url_corroborated(
    left: NormalizedItem, right: NormalizedItem, config: DedupConfig
) -> bool:
    """Return True when two same-URL items carry independent agreement.

    A URL can be reused: slugs are recycled, evergreen pages rewritten, live
    blogs served from one address.  Equality of the key is therefore a hint,
    and something else must also agree.  This runs *after* the compatibility
    gate, so it answers "is there positive evidence?", not "is there no
    objection?".
    """

    if left.provider_key is not None and left.provider_key == right.provider_key:
        return True
    if (
        left.content_fingerprint is not None
        and left.content_fingerprint == right.content_fingerprint
    ):
        return True
    if left.title_key is not None and left.title_key == right.title_key:
        return True
    if left.published_at is not None and right.published_at is not None:
        return abs(right.published_at - left.published_at) <= config.url_window
    return False


# --------------------------------------------------------------------------
# Stage 2: MinHash candidate generation plus exact verification
# --------------------------------------------------------------------------


def minhash_candidates(
    items: Sequence[NormalizedItem],
    ordered: Sequence[int],
    config: DedupConfig,
) -> list[tuple[int, int]]:
    """Return MinHash-proposed candidate pairs, in deterministic order.

    Candidates form only between timestamped items inside
    ``near_exact_window`` — a hard requirement of the signal anyway, and
    what keeps pairwise signature comparison affordable at Phase 0 volumes.
    """

    eligible = [
        index
        for index in ordered
        if items[index].published_at is not None and items[index].title_key
    ]
    signatures = {
        index: signature(
            shingles(items[index].title_key or "", config.minhash_shingle_size),
            permutations=config.minhash_permutations,
            seed=config.minhash_seed,
        )
        for index in eligible
    }
    window = config.near_exact_window
    pairs: list[tuple[int, int]] = []
    for offset, left in enumerate(eligible):
        left_time = items[left].published_at
        assert left_time is not None
        for right in eligible[offset + 1 :]:
            right_time = items[right].published_at
            assert right_time is not None
            if right_time - left_time > window:
                # `eligible` is time-ordered, so no later item qualifies.
                break
            if (
                estimate_similarity(signatures[left], signatures[right])
                >= config.candidate_min_similarity
            ):
                pairs.append((left, right))
    return pairs


def verify_candidate(left: NormalizedItem, right: NormalizedItem) -> bool:
    """Return True when a MinHash candidate is a structural duplicate.

    Verification is exact and conservative: the two token sequences must be
    identical, together with their ordered protected expressions.  It reads
    nothing but the two items, so no third record can change the answer, and
    a title the core declined to parse never verifies.
    """

    if left.uncertain_title or right.uncertain_title:
        return False
    return (
        left.structural_key is not None and left.structural_key == right.structural_key
    )


# --------------------------------------------------------------------------
# Partition detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Acceptance:
    """What survived the gate, and the structure that recorded it."""

    accepted: tuple[CandidateEdge, ...]
    reason_counts: Counter
    veto_counts: Counter
    disjoint_set: "_ConstrainedDisjointSet"
    position: dict[int, int]


def _accept_edges(
    items: Sequence[NormalizedItem],
    edges: Sequence[CandidateEdge],
    ordered: Sequence[int],
    config: DedupConfig,
) -> _Acceptance:
    """Apply edges strongest-first through the single compatibility gate."""

    windows = {
        MatchReason.PROVIDER_ITEM: None,
        MatchReason.CANONICAL_URL: config.url_window,
        MatchReason.EXACT_CONTENT: config.content_window,
        MatchReason.EXACT_TITLE: config.exact_title_window,
        MatchReason.NEAR_EXACT_TITLE: config.near_exact_window,
    }
    position = {index: offset for offset, index in enumerate(ordered)}
    disjoint_set = _ConstrainedDisjointSet(
        [items[index].published_at for index in ordered],
        [summarize(items[index]) for index in ordered],
    )
    ordered_edges = sorted(
        edges,
        key=lambda edge: (
            REASON_RANK[edge.reason],
            position[edge.left],
            position[edge.right],
        ),
    )
    accepted: list[CandidateEdge] = []
    reason_counts: Counter = Counter()
    veto_counts: Counter = Counter()
    for edge in ordered_edges:
        # One gate, one place, applied to the whole prospective cluster.
        # Provider identity is authoritative and is the only signal that
        # bypasses it, having already survived conflict quarantine.
        merged, veto = disjoint_set.union(
            position[edge.left],
            position[edge.right],
            windows[edge.reason],
            gated=edge.reason is not MatchReason.PROVIDER_ITEM,
        )
        if veto is not None:
            veto_counts[veto] += 1
            continue
        if merged:
            accepted.append(edge)
            reason_counts[edge.reason.value] += 1
    return _Acceptance(
        accepted=tuple(accepted),
        reason_counts=reason_counts,
        veto_counts=veto_counts,
        disjoint_set=disjoint_set,
        position=position,
    )


def _detect_partition(
    ticker: str,
    items: Sequence[NormalizedItem],
    ordered: Sequence[int],
    mergeable: Sequence[int],
    config: DedupConfig,
) -> tuple[list[DuplicateGroup], Counter, Counter, int, int]:
    edges: list[CandidateEdge] = []
    edges.extend(
        _bucket_edges(
            items,
            mergeable,
            lambda item: item.provider_key,
            MatchReason.PROVIDER_ITEM,
            requires_timestamp=False,
        )
    )
    # A repeated URL only counts as evidence between records that can be
    # placed in time: without both timestamps the configured window cannot
    # be enforced, and an unbounded URL merge is exactly what a recycled
    # slug exploits.  An undated pair may still merge on provider identity.
    url_edges = _bucket_edges(
        items,
        mergeable,
        lambda item: item.url_key,
        MatchReason.CANONICAL_URL,
        requires_timestamp=True,
    )
    edges.extend(
        edge
        for edge in url_edges
        if url_corroborated(items[edge.left], items[edge.right], config)
    )
    edges.extend(
        _bucket_edges(
            items,
            mergeable,
            lambda item: item.content_key,
            MatchReason.EXACT_CONTENT,
            requires_timestamp=True,
        )
    )
    edges.extend(
        _bucket_edges(
            items,
            mergeable,
            lambda item: item.title_key,
            MatchReason.EXACT_TITLE,
            requires_timestamp=True,
        )
    )
    candidates = minhash_candidates(items, mergeable, config)
    verified = 0
    for left, right in candidates:
        if verify_candidate(items[left], items[right]):
            verified += 1
            edges.append(CandidateEdge(left, right, MatchReason.NEAR_EXACT_TITLE))

    outcome = _accept_edges(items, edges, ordered, config)
    disjoint_set, position = outcome.disjoint_set, outcome.position

    clusters: dict[int, list[int]] = {}
    for index in ordered:
        clusters.setdefault(disjoint_set.find(position[index]), []).append(index)
    member_reasons: dict[int, MatchReason] = {}
    cluster_reasons: dict[int, set[MatchReason]] = {}
    for edge in outcome.accepted:
        root = disjoint_set.find(position[edge.left])
        cluster_reasons.setdefault(root, set()).add(edge.reason)
        for index in (edge.left, edge.right):
            current = member_reasons.get(index)
            if current is None or REASON_RANK[edge.reason] < REASON_RANK[current]:
                member_reasons[index] = edge.reason

    groups = [
        DuplicateGroup(
            ticker=ticker,
            member_indices=tuple(clusters[root]),
            member_reasons=tuple(
                (index, member_reasons[index])
                for index in clusters[root]
                if index in member_reasons
            ),
            reasons=tuple(
                sorted(cluster_reasons.get(root, set()), key=lambda r: REASON_RANK[r])
            ),
        )
        for root in sorted(clusters, key=lambda value: position[clusters[value][0]])
    ]
    return (
        groups,
        outcome.reason_counts,
        outcome.veto_counts,
        len(candidates),
        verified,
    )


def detect_duplicates(
    items: Sequence[NormalizedItem], config: DedupConfig
) -> DetectionReport:
    """Group items into clusters using exact identity signals only.

    Raises :class:`~nlp.dedup.errors.DedupCapacityError` before producing
    anything if a ticker partition exceeds ``max_partition_items``.
    """

    partitions: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        partitions.setdefault(item.ticker, []).append(index)
    for ticker in sorted(partitions):
        if len(partitions[ticker]) > config.max_partition_items:
            raise DedupCapacityError(
                ticker, len(partitions[ticker]), config.max_partition_items
            )

    conflicts, quarantined = provider_conflicts(items, config)
    groups: list[DuplicateGroup] = []
    reason_counts: Counter = Counter()
    veto_counts: Counter = Counter()
    candidate_pairs = 0
    verified_pairs = 0
    for ticker in sorted(partitions):
        ordered = sorted(partitions[ticker], key=lambda index: _order_key(items[index]))
        mergeable = [index for index in ordered if index not in quarantined]
        (
            partition_groups,
            partition_reasons,
            partition_vetoes,
            partition_candidates,
            partition_verified,
        ) = _detect_partition(ticker, items, ordered, mergeable, config)
        groups.extend(partition_groups)
        reason_counts.update(partition_reasons)
        veto_counts.update(partition_vetoes)
        candidate_pairs += partition_candidates
        verified_pairs += partition_verified
    return DetectionReport(
        groups=tuple(groups),
        reason_counts=tuple(sorted(reason_counts.items())),
        veto_counts=tuple(sorted(veto_counts.items())),
        provider_conflicts=conflicts,
        quarantined_item_ids=tuple(
            sorted(items[index].item_id for index in quarantined)
        ),
        candidate_pair_count=candidate_pairs,
        verified_candidate_pair_count=verified_pairs,
    )

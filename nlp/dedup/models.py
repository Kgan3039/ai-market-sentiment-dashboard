"""Immutable data model for the M2 dedup core.

Field names follow the Phase 0 ``raw_items`` columns in
``docs/PHASE_0_SPEC.md`` section 5 so an adapter is a straight projection.
The core itself is a pure transformation: raw items in, clusters out.  It
assigns no durable identifier and writes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MatchReason(str, Enum):
    """Why a member belongs to a cluster, strongest signal first."""

    #: Same provider namespace and provider item id, unconflicted.  The one
    #: authoritative signal: it is not time-bounded.
    PROVIDER_ITEM = "provider_item"
    #: Identical conservative URL identity key, corroborated.
    CANONICAL_URL = "canonical_url"
    #: Identical normalized title *and* description, windowed.
    EXACT_CONTENT = "exact_content"
    #: Byte-identical normalized title, windowed.
    EXACT_TITLE = "exact_title"
    #: MinHash candidate verified as structurally identical, windowed.
    #: With per-item normalization this predicate coincides with
    #: :attr:`EXACT_TITLE`, so the reason is not currently emitted; see
    #: ``nlp/README.md``.
    NEAR_EXACT_TITLE = "near_exact_title"
    #: The member the cluster is named after; never a merge signal.
    CANONICAL = "canonical"


#: Lower rank wins when a member is attached by more than one signal, and
#: orders edge application so stronger identity merges first.
REASON_RANK: dict[MatchReason, int] = {
    MatchReason.PROVIDER_ITEM: 0,
    MatchReason.CANONICAL_URL: 1,
    MatchReason.EXACT_CONTENT: 2,
    MatchReason.EXACT_TITLE: 3,
    MatchReason.NEAR_EXACT_TITLE: 4,
    MatchReason.CANONICAL: 5,
}


@dataclass(frozen=True)
class RawItem:
    """One ingested headline.

    ``ticker`` is **required** to be a supported symbol: the core does not
    filter unassigned items, because Phase 0's relevance filter already
    excludes them from processing (spec section 2).  Orchestration drops
    them before calling.

    ``published_at`` accepts a timezone-aware :class:`datetime` or an
    ISO-8601 string carrying an offset.  Naive values are rejected: no
    Phase 0 contract states what wall clock they are in, and guessing would
    silently shift every merge window.

    ``provider_item_id`` is the feed's own stable identifier.  Everything
    else is optional; an item with no usable title and no usable URL still
    survives as its own cluster.
    """

    item_id: str | int
    ticker: str
    title: str | None = None
    description: str | None = None
    url: str | None = None
    canonical_url: str | None = None
    source: str | None = None
    published_at: datetime | str | None = None
    provider_item_id: str | None = None


@dataclass(frozen=True)
class NormalizedItem:
    """Every deterministic signal the core derives from one raw item."""

    item_id: str
    ticker: str
    display_title: str
    #: Exact-title identity, or ``None`` when the title is empty or its
    #: numeric notation could not be parsed with confidence.
    title_key: str | None
    #: Digest of the token sequence and its protected expressions.
    structural_key: str | None
    #: Ordered numeric expressions of the title, never sorted.
    title_protected: tuple[str, ...]
    #: True when the title used numeric notation the core cannot represent.
    uncertain_title: bool
    normalized_description: str
    normalized_source: str
    #: Outlet identity used for ``outlet_count``.
    outlet: str
    provider_namespace: str | None
    provider_item_id: str | None
    #: Authoritative-tier key, or ``None`` without a provider identity.
    provider_key: str | None
    #: Followable, storable URL (never used for identity).
    canonical_url: str | None
    #: URL identity key, or ``None`` when identity cannot safely be claimed.
    url_key: str | None
    #: Identical title-and-description key, or ``None`` without both.
    content_key: str | None
    #: Fingerprint of all available text; ``None`` when there is none.
    content_fingerprint: str | None
    published_at: datetime | None


@dataclass(frozen=True)
class ProviderConflict:
    """One provider item id that described more than one article."""

    provider_namespace: str
    provider_item_id: str
    #: Every raw item id published under this provider identity, sorted.
    item_ids: tuple[str, ...]
    #: Which fields disagreed, sorted.
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ClusterMember:
    """One retained member of a cluster, with its source link."""

    item_id: str
    title: str
    outlet: str
    source: str | None
    url: str | None
    canonical_url: str | None
    published_at: datetime | None
    match_reason: MatchReason


@dataclass(frozen=True)
class DeduplicatedCluster:
    """One canonical story: a representative headline plus every member.

    There is no ``story_id`` here on purpose.  A durable identifier belongs
    to the ``stories`` table (issue #57) and to whatever reconciles a run
    against it; see :attr:`cluster_fingerprint`.
    """

    #: Collision-safe digest of the ticker and the sorted unique member set,
    #: length-prefix encoded so no item id can forge a boundary.  It is a
    #: *change-detection* handle, not a durable key: adding a syndicated
    #: member changes it, which is precisely what a reconciler needs to see.
    cluster_fingerprint: str
    ticker: str
    canonical_item_id: str
    canonical_title: str
    normalized_title: str
    canonical_url: str | None
    source: str | None
    outlet: str
    published_at: datetime | None
    outlet_count: int
    member_ids: tuple[str, ...]
    members: tuple[ClusterMember, ...]
    match_reasons: tuple[MatchReason, ...]
    #: Digest of the whole projection plus every version and policy marker.
    content_hash: str
    algorithm_version: str

    @property
    def member_count(self) -> int:
        """Number of raw items collapsed into this cluster."""

        return len(self.member_ids)

    @property
    def is_merged(self) -> bool:
        """True when this cluster collapsed more than one raw item."""

        return len(self.member_ids) > 1

    @property
    def is_syndicated(self) -> bool:
        """True when at least two distinct outlets carried the story."""

        return self.outlet_count >= 2


@dataclass(frozen=True)
class DedupStats:
    """Counters for the caller's own logging and the AC-3 spot check."""

    input_count: int
    cluster_count: int
    merged_cluster_count: int
    duplicate_member_count: int
    syndicated_cluster_count: int
    reason_counts: tuple[tuple[str, int], ...] = ()
    #: Distinct provider item ids that carried conflicting payloads.
    provider_conflict_count: int = 0
    #: Items quarantined by those conflicts; they merge through no signal.
    quarantined_item_count: int = 0
    #: Titles whose numeric notation the core declined to parse.
    uncertain_title_count: int = 0
    #: Edges refused by the compatibility gate, by reason.
    veto_counts: tuple[tuple[str, int], ...] = ()
    #: MinHash telemetry.
    candidate_pair_count: int = 0
    verified_candidate_pair_count: int = 0

    def reason_count(self, reason: MatchReason | str) -> int:
        """Number of accepted merge edges attributed to ``reason``."""

        key = reason.value if isinstance(reason, MatchReason) else str(reason)
        return dict(self.reason_counts).get(key, 0)

    def veto_count(self, reason: str) -> int:
        """Number of candidate edges refused for this incompatibility."""

        return dict(self.veto_counts).get(reason, 0)


@dataclass(frozen=True)
class DedupResult:
    """Deterministic output of one dedup run."""

    clusters: tuple[DeduplicatedCluster, ...]
    stats: DedupStats
    #: Item ids quarantined by a provider-identity conflict.  They are
    #: still emitted as single-member clusters, so no coverage is lost.
    quarantined_item_ids: tuple[str, ...]
    provider_conflicts: tuple[ProviderConflict, ...]
    config_fingerprint: str
    algorithm_version: str

    @property
    def cluster_by_member(self) -> dict[str, DeduplicatedCluster]:
        """Map every input item id onto the cluster that absorbed it."""

        return {
            member_id: cluster
            for cluster in self.clusters
            for member_id in cluster.member_ids
        }

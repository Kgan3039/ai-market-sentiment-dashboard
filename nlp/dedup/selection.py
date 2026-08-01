"""Canonical member selection and cluster assembly.

Merge policy (issue #64): the earliest-published member is canonical, every
member is retained with its source link, and ``outlet_count`` tracks the
number of distinct outlets carrying the cluster.

The core assigns no durable identifier.  Issue #64 does not ask for one,
and the ``stories`` table that owns ``id`` belongs to issue #57.  What the
core returns instead is a *cluster fingerprint*: a collision-safe digest of
the ticker and the sorted unique member set, for change detection by
whoever reconciles a run against storage.  See
:func:`cluster_fingerprint_for`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from typing import Sequence

from .config import ALGORITHM_VERSION
from .errors import DedupInputError
from .detection import DuplicateGroup
from .models import (
    DeduplicatedCluster,
    MatchReason,
    NormalizedItem,
    RawItem,
    ClusterMember,
)

_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
#: Namespace for cluster fingerprints.
CLUSTER_NAMESPACE = "m2.cluster.v1"


@dataclass(frozen=True)
class _MemberView:
    """One group member paired with the raw item it came from."""

    index: int
    raw: RawItem
    normalized: NormalizedItem


def _canonical_sort_key(member: _MemberView) -> tuple[bool, datetime, str, str]:
    """Earliest publication wins; outlet then item id break exact ties."""

    published_at = member.normalized.published_at
    return (
        published_at is None,
        published_at or _EPOCH,
        member.normalized.outlet,
        member.normalized.item_id,
    )


def _member_sort_key(member: _MemberView) -> tuple[bool, datetime, str]:
    published_at = member.normalized.published_at
    return (published_at is None, published_at or _EPOCH, member.normalized.item_id)


def encode_fields(fields: Sequence[str]) -> bytes:
    """Length-prefix every field so no value can forge a boundary.

    ``["a", "b"]`` and ``["a\x1fb"]`` produce different bytes, so two
    different member sets cannot collide through separator ambiguity no
    matter what characters an upstream item id contains.
    """

    encoded = bytearray()
    for value in fields:
        payload = value.encode("utf-8")
        encoded += str(len(payload)).encode("ascii") + b":" + payload
    return bytes(encoded)


def cluster_fingerprint_for(ticker: str, member_ids: Sequence[str]) -> str:
    """Return the change-detection fingerprint of one cluster.

    A full-width SHA-256 digest of the ticker and the *sorted unique member
    set*, length-prefix encoded.  The public path already guarantees
    well-formed input; the checks here exist so a future caller cannot
    quietly produce a fingerprint for an empty, blank, or duplicated member
    set that would collide with a real one.  It is stable under input permutation and
    independent of which member is currently canonical, and it changes when
    membership changes — which is exactly what a reconciler needs to notice
    that a syndicated copy joined.  It is **not** a durable id: the layer
    that owns the ``stories`` table assigns and carries that, joining a
    run's clusters to stored rows on ``member_ids``.
    """

    if not isinstance(CLUSTER_NAMESPACE, str) or not CLUSTER_NAMESPACE.strip():
        raise DedupInputError("cluster fingerprint needs a non-blank namespace")
    if not isinstance(ticker, str) or not ticker.strip():
        raise DedupInputError("cluster fingerprint needs a non-blank ticker")
    identifiers = list(member_ids)
    if not identifiers:
        raise DedupInputError("cluster fingerprint needs at least one member")
    unique = set()
    for identifier in identifiers:
        if not isinstance(identifier, str) or not identifier.strip():
            raise DedupInputError("cluster member ids must be non-blank strings")
        if identifier in unique:
            raise DedupInputError(f"duplicate cluster member id: {identifier!r}")
        unique.add(identifier)
    payload = encode_fields((CLUSTER_NAMESPACE, ticker, *sorted(unique)))
    return hashlib.sha256(payload).hexdigest()


def _isoformat(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _cluster_content_hash(
    *,
    cluster_fingerprint: str,
    ticker: str,
    canonical_item_id: str,
    normalized_title: str,
    canonical_url: str | None,
    published_at: datetime | None,
    outlet_count: int,
    member_ids: Sequence[str],
    config_fingerprint: str,
) -> str:
    """Hash everything a stored row derives from, including the settings.

    A caller can therefore skip an unchanged cluster on replay, while a
    changed algorithm version or configuration forces a rewrite.
    """

    fields = [
        ALGORITHM_VERSION,
        config_fingerprint,
        cluster_fingerprint,
        ticker,
        canonical_item_id,
        normalized_title,
        canonical_url or "",
        _isoformat(published_at),
        str(outlet_count),
        *member_ids,
    ]
    return hashlib.sha256(encode_fields(fields)).hexdigest()


def _canonical_display_title(
    canonical: _MemberView, ordered: Sequence[_MemberView]
) -> str:
    if canonical.normalized.display_title:
        return canonical.normalized.display_title
    for member in ordered:
        if member.normalized.display_title:
            return member.normalized.display_title
    return ""


def build_cluster(
    group: DuplicateGroup,
    raw_items: Sequence[RawItem],
    normalized_items: Sequence[NormalizedItem],
    config_fingerprint: str,
) -> DeduplicatedCluster:
    """Assemble one canonical cluster from a detected duplicate group."""

    members = [
        _MemberView(index, raw_items[index], normalized_items[index])
        for index in group.member_indices
    ]
    canonical = min(members, key=_canonical_sort_key)
    ordered = [canonical] + sorted(
        (member for member in members if member.index != canonical.index),
        key=_member_sort_key,
    )
    reasons = dict(group.member_reasons)
    cluster_members = tuple(
        ClusterMember(
            item_id=member.normalized.item_id,
            title=member.normalized.display_title,
            outlet=member.normalized.outlet,
            source=member.raw.source,
            url=member.raw.url,
            canonical_url=member.normalized.canonical_url,
            published_at=member.normalized.published_at,
            match_reason=(
                MatchReason.CANONICAL
                if member.index == canonical.index
                else reasons.get(member.index, MatchReason.CANONICAL)
            ),
        )
        for member in ordered
    )
    member_ids = tuple(member.item_id for member in cluster_members)
    outlet_count = len({member.outlet for member in cluster_members if member.outlet})
    fingerprint = cluster_fingerprint_for(group.ticker, member_ids)
    return DeduplicatedCluster(
        cluster_fingerprint=fingerprint,
        ticker=group.ticker,
        canonical_item_id=canonical.normalized.item_id,
        canonical_title=_canonical_display_title(canonical, ordered),
        normalized_title=canonical.normalized.title_key or "",
        canonical_url=canonical.normalized.canonical_url,
        source=canonical.raw.source,
        outlet=canonical.normalized.outlet,
        published_at=canonical.normalized.published_at,
        outlet_count=outlet_count,
        member_ids=member_ids,
        members=cluster_members,
        match_reasons=group.reasons,
        content_hash=_cluster_content_hash(
            cluster_fingerprint=fingerprint,
            ticker=group.ticker,
            canonical_item_id=canonical.normalized.item_id,
            normalized_title=canonical.normalized.title_key or "",
            canonical_url=canonical.normalized.canonical_url,
            published_at=canonical.normalized.published_at,
            outlet_count=outlet_count,
            member_ids=member_ids,
            config_fingerprint=config_fingerprint,
        ),
        algorithm_version=ALGORITHM_VERSION,
    )


def build_clusters(
    groups: Sequence[DuplicateGroup],
    raw_items: Sequence[RawItem],
    normalized_items: Sequence[NormalizedItem],
    config_fingerprint: str,
) -> tuple[DeduplicatedCluster, ...]:
    """Build every cluster, ordered deterministically for stable output."""

    clusters = [
        build_cluster(group, raw_items, normalized_items, config_fingerprint)
        for group in groups
    ]
    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                cluster.ticker,
                cluster.published_at is None,
                cluster.published_at or _EPOCH,
                cluster.cluster_fingerprint,
            ),
        )
    )

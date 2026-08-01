"""The M2 dedup core entry point.

:func:`deduplicate` is a pure, deterministic transformation: raw items in,
canonical clusters out.  It touches no database, no clock, no network, no
random source, no filesystem, and no model, and it holds no state between
calls.  Everything about a run comes from its arguments.

What it deliberately does *not* do: assign durable story identifiers,
persist anything, read configuration from disk, project database rows, or
report partial success.  Those belong to the Phase 0 repository (issue #57)
and the pipeline runner (issue #68), neither of which exists yet.
"""

from __future__ import annotations

from typing import Sequence

from .config import ALGORITHM_VERSION, DedupConfig
from .detection import detect_duplicates
from .errors import DedupInputError
from .models import DedupResult, DedupStats, NormalizedItem, RawItem
from .normalization import normalize_item
from .selection import build_clusters


def _normalize_items(
    items: Sequence[RawItem], config: DedupConfig
) -> tuple[NormalizedItem, ...]:
    """Normalize every raw item, rejecting anything the core cannot process.

    A repeated ``item_id`` means the caller lost track of identity upstream;
    silently collapsing it would hide an ingestion bug and corrupt
    ``outlet_count``.  A missing or unsupported ticker is likewise the
    caller's to fix before calling.
    """

    universe = config.ticker_universe
    normalized: list[NormalizedItem] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        try:
            entry = normalize_item(item, universe)
        except DedupInputError as exc:
            raise DedupInputError(f"items[{index}] is invalid: {exc}") from exc
        if entry.item_id in seen:
            raise DedupInputError(f"duplicate item_id: {entry.item_id}")
        seen.add(entry.item_id)
        normalized.append(entry)
    return tuple(normalized)


def deduplicate(items: Sequence[RawItem], *, config: DedupConfig) -> DedupResult:
    """Collapse raw items into canonical clusters.

    Every item must carry a ticker inside ``config.supported_tickers``;
    unassigned items are the caller's to filter out, because Phase 0's
    relevance filter already excludes them from processing.

    Deterministic end to end: the same input set produces byte-identical
    output regardless of the order it is supplied in, and no record's
    clusters depend on which *other* records share the batch.  The returned
    value shares no mutable state with the caller's input container.

    Raises :class:`~nlp.dedup.errors.DedupCapacityError` — before producing
    any output — when one ticker holds more than
    ``config.max_partition_items`` items.  The core never returns a result
    that looks complete but skipped work.
    """

    if not isinstance(config, DedupConfig):
        raise DedupInputError("config must be a DedupConfig")
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        raise DedupInputError("items must be a sequence of RawItem instances")
    # Snapshot before any work: a later mutation of the caller's list must
    # not be able to reach into the result.
    snapshot = tuple(items)
    fingerprint = config.fingerprint()
    normalized = _normalize_items(snapshot, config)

    detection = detect_duplicates(normalized, config)
    clusters = build_clusters(detection.groups, snapshot, normalized, fingerprint)
    merged = [cluster for cluster in clusters if cluster.is_merged]
    stats = DedupStats(
        input_count=len(normalized),
        cluster_count=len(clusters),
        merged_cluster_count=len(merged),
        duplicate_member_count=sum(cluster.member_count - 1 for cluster in merged),
        syndicated_cluster_count=sum(
            1 for cluster in clusters if cluster.is_syndicated
        ),
        reason_counts=detection.reason_counts,
        provider_conflict_count=len(detection.provider_conflicts),
        quarantined_item_count=len(detection.quarantined_item_ids),
        uncertain_title_count=sum(1 for entry in normalized if entry.uncertain_title),
        veto_counts=detection.veto_counts,
        candidate_pair_count=detection.candidate_pair_count,
        verified_candidate_pair_count=detection.verified_candidate_pair_count,
    )
    return DedupResult(
        clusters=clusters,
        stats=stats,
        quarantined_item_ids=detection.quarantined_item_ids,
        provider_conflicts=detection.provider_conflicts,
        config_fingerprint=fingerprint,
        algorithm_version=ALGORITHM_VERSION,
    )

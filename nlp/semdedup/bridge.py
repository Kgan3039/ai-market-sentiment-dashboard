"""The one place M3 knows what an M2 cluster looks like.

M2's :class:`~nlp.dedup.DeduplicatedCluster` carries a canonical title but
no description — descriptions are per-member and M2 has no reason to pick
one.  M3 embeds title *and* description, so the projection needs the raw
items too, and the caller already has both.  Keeping that join here means
the rest of :mod:`nlp.semdedup` never imports :mod:`nlp.dedup`'s models and
can be driven by anything that produces :class:`StoryInput`.

The description chosen is the canonical member's, falling back to the first
member that has one in cluster order.  Deterministic and explainable: the
story is represented by the article it is named after.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from nlp.dedup import DedupResult, DeduplicatedCluster, RawItem

from .models import SourceLink, StoryInput


def _descriptions(raw_items: Iterable[RawItem]) -> Mapping[str, str | None]:
    return {str(item.item_id).strip(): item.description for item in raw_items}


def story_from_cluster(
    cluster: DeduplicatedCluster, descriptions: Mapping[str, str | None]
) -> StoryInput:
    """Project one M2 cluster onto M3's input type."""

    description = descriptions.get(cluster.canonical_item_id)
    if not description:
        for member in cluster.members:
            candidate = descriptions.get(member.item_id)
            if candidate:
                description = candidate
                break
    return StoryInput(
        story_key=cluster.cluster_fingerprint,
        ticker=cluster.ticker,
        title=cluster.canonical_title,
        description=description or None,
        published_at=cluster.published_at,
        outlets=tuple(
            sorted({member.outlet for member in cluster.members if member.outlet})
        ),
        member_ids=tuple(cluster.member_ids),
        source_links=tuple(
            SourceLink(
                item_id=member.item_id,
                outlet=member.outlet,
                url=member.canonical_url or member.url,
            )
            for member in cluster.members
        ),
    )


def stories_from_dedup(
    result: DedupResult, raw_items: Sequence[RawItem]
) -> tuple[StoryInput, ...]:
    """Project a whole M2 run onto M3's input, in M2's cluster order."""

    descriptions = _descriptions(raw_items)
    return tuple(
        story_from_cluster(cluster, descriptions) for cluster in result.clusters
    )

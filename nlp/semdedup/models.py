"""Immutable data model for the M3 semantic dedup stage.

M3's input is deliberately *not* :class:`nlp.dedup.DeduplicatedCluster`.
The stage needs a canonical title, a description, a time, and enough member
bookkeeping to preserve retention — nothing about how M2 arrived at the
cluster.  Taking a small neutral type keeps M3 testable without M2 and
means a change to M2's cluster shape cannot reach in here; the bridge in
:mod:`nlp.semdedup.bridge` does the projection in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SemanticSkipReason(str, Enum):
    """Why a story was kept out of semantic candidate generation."""

    #: Every member is under an M2 provider-identity quarantine.  M2 found
    #: one feed identifier describing two different articles and could not
    #: tell which payload was right; a cosine score is not evidence about
    #: which one is, so M3 does not get to overrule it.
    PROVIDER_QUARANTINE = "provider_quarantine"


class SemanticMergeReason(str, Enum):
    """Why two canonical stories were merged."""

    #: Cosine similarity cleared the threshold against *every* member of the
    #: prospective story, and no contradiction guard objected.
    SEMANTIC_SIMILARITY = "semantic_similarity"


@dataclass(frozen=True)
class SourceLink:
    """One retained source link, carried through unchanged."""

    item_id: str
    outlet: str
    url: str | None


@dataclass(frozen=True)
class StoryInput:
    """One canonical story arriving from M2.

    ``story_key`` is whatever stable handle the caller uses for the cluster
    — M2's ``cluster_fingerprint`` in the Phase 0 pipeline.  M3 treats it as
    opaque and never parses it.
    """

    story_key: str
    ticker: str
    title: str
    description: str | None = None
    published_at: datetime | None = None
    #: Distinct outlets carrying this story, in the caller's order.
    outlets: tuple[str, ...] = ()
    #: Every raw item collapsed into this story.
    member_ids: tuple[str, ...] = ()
    source_links: tuple[SourceLink, ...] = ()
    #: Member ids M2 quarantined under a provider-identity conflict, read
    #: from ``DedupResult.quarantined_item_ids``.  Never inferred.
    quarantined_member_ids: tuple[str, ...] = ()
    #: ``(namespace, provider_item_id)`` of every conflict touching this
    #: story, from ``DedupResult.provider_conflicts``.
    provider_conflicts: tuple[tuple[str, str], ...] = ()

    @property
    def is_quarantined(self) -> bool:
        """True when M2 isolated this story's members on a conflict.

        A story is quarantined when *every* member is, which is what M2
        produces: quarantine applies to all items under a conflicting
        provider identity, and such a cluster is always a singleton.
        """

        return bool(self.member_ids) and set(self.quarantined_member_ids) == set(
            self.member_ids
        )


@dataclass(frozen=True)
class SemanticMerge:
    """Evidence for one accepted semantic merge, for the audit trail."""

    left_story_key: str
    right_story_key: str
    similarity: float
    reason: SemanticMergeReason = SemanticMergeReason.SEMANTIC_SIMILARITY


@dataclass(frozen=True)
class RejectedPair:
    """A candidate pair the stage refused, and why.

    Every refusal is recorded, not just counted: "why did these two not
    merge" is the question a reviewer asks about a clustering stage, and a
    counter cannot answer it.
    """

    left_story_key: str
    right_story_key: str
    similarity: float | None
    reason: str


@dataclass(frozen=True)
class SemanticStory:
    """One story after semantic merging.

    Carries no durable identifier.  ``stories.id`` belongs to issue #57;
    what is here is a change-detection fingerprint over the ticker and the
    sorted member set, exactly as M2 does for clusters.
    """

    story_fingerprint: str
    ticker: str
    #: The ``story_key`` of the member chosen as canonical.
    canonical_story_key: str
    canonical_title: str
    published_at: datetime | None
    #: Input story keys collapsed into this story, canonical first.
    member_story_keys: tuple[str, ...]
    #: Every raw item id, sorted; the union of the members' items.
    member_ids: tuple[str, ...]
    outlets: tuple[str, ...]
    outlet_count: int
    source_links: tuple[SourceLink, ...]
    #: Accepted merge edges, sorted; empty for a story M3 did not touch.
    merges: tuple[SemanticMerge, ...]
    #: Quarantined member ids carried through unchanged from M2.
    quarantined_member_ids: tuple[str, ...]
    #: Provider conflicts touching this story, carried through from M2.
    provider_conflicts: tuple[tuple[str, str], ...]
    #: Set when the story was held out of candidate generation.
    semantic_skip_reason: SemanticSkipReason | None
    content_hash: str
    algorithm_version: str

    @property
    def is_quarantined(self) -> bool:
        """True when this story carries an unresolved M2 provider conflict."""

        return self.semantic_skip_reason is SemanticSkipReason.PROVIDER_QUARANTINE

    @property
    def member_count(self) -> int:
        """Number of M2 stories collapsed into this one."""

        return len(self.member_story_keys)

    @property
    def is_merged(self) -> bool:
        """True when M3 collapsed more than one M2 story."""

        return len(self.member_story_keys) > 1

    @property
    def is_syndicated(self) -> bool:
        """True when at least two distinct outlets carried the story."""

        return self.outlet_count >= 2


@dataclass(frozen=True)
class SemanticDedupStats:
    """Counters for the caller's logging and the AC-3 spot check."""

    input_story_count: int
    story_count: int
    merged_story_count: int
    collapsed_story_count: int
    #: Pairs the candidate generator proposed inside a ticker and window.
    candidate_pair_count: int
    #: Candidates that cleared the similarity threshold.
    above_threshold_count: int
    #: Edges accepted into a story after every guard and the clique rule.
    accepted_pair_count: int
    #: Refusals by reason, sorted.
    veto_counts: tuple[tuple[str, int], ...] = ()
    #: Stories with no usable text, which cannot be compared at all.
    unencodable_story_count: int = 0
    #: Stories held out of candidate generation, by reason, sorted.
    skipped_story_counts: tuple[tuple[str, int], ...] = ()

    def skipped_count(self, reason: str) -> int:
        """Number of stories held out for this reason."""

        return dict(self.skipped_story_counts).get(reason, 0)

    def veto_count(self, reason: str) -> int:
        """Number of candidate pairs refused for this reason."""

        return dict(self.veto_counts).get(reason, 0)


@dataclass(frozen=True)
class SemanticDedupResult:
    """Deterministic output of one semantic dedup run."""

    stories: tuple[SemanticStory, ...]
    stats: SemanticDedupStats
    rejected_pairs: tuple[RejectedPair, ...]
    config_fingerprint: str
    algorithm_version: str
    #: Which encoder produced the vectors, recorded so a stored result can
    #: be invalidated when the model moves.
    model_name: str
    model_revision: str | None
    #: Vector width the run actually used; ``None`` when nothing encodable.
    embedding_dimension: int | None = None

    @property
    def story_by_member(self) -> dict[str, SemanticStory]:
        """Map every input story key onto the story that absorbed it."""

        return {key: story for story in self.stories for key in story.member_story_keys}

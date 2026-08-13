"""Transport records for the Phase 0 reconciliation boundaries.

These are deliberately plain: the pipeline projects M2/M3/M5 results onto
them, and the repository writes one ticker/trading-day atomically.  They
carry no algorithm and no durable identifier — ``stories.id`` and
``themes.id`` belong to the database, and the upstream fingerprints stay
what they are upstream: change-detection handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class StoryMemberRecord:
    """One raw item retained inside a canonical story, with its link."""

    raw_item_id: int
    position: int = 0
    outlet: str | None = None
    url: str | None = None
    canonical_url: str | None = None
    match_reason: str | None = None
    quarantined: bool = False


@dataclass(frozen=True)
class ProviderConflictRecord:
    """One M2 provider-identity conflict touching a story."""

    provider_namespace: str
    provider_item_id: str
    item_ids: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticMergeRecord:
    """One accepted M3 merge edge, kept as audit evidence."""

    left_story_key: str
    right_story_key: str
    similarity: float
    reason: str = "semantic_similarity"


@dataclass(frozen=True)
class StoryRecord:
    """One canonical story for a ticker/trading-day."""

    cluster_fingerprint: str
    canonical_title: str
    members: tuple[StoryMemberRecord, ...]
    canonical_item_id: int | None = None
    outlet_count: int = 1
    published_at: str | datetime | None = None
    canonical_url: str | None = None
    source: str | None = None
    outlet: str | None = None
    content_hash: str | None = None
    algorithm_version: str | None = None
    config_fingerprint: str | None = None
    stage: str = "m2.exact"
    model_name: str | None = None
    model_revision: str | None = None
    embedding_dimension: int | None = None
    quarantined: bool = False
    semantic_skip_reason: str | None = None
    member_story_keys: tuple[str, ...] = ()
    provider_conflicts: tuple[ProviderConflictRecord, ...] = ()
    semantic_merges: tuple[SemanticMergeRecord, ...] = ()
    embedding: bytes | None = None


@dataclass(frozen=True)
class ThemeRecord:
    """One theme of a ticker/trading-day theme set."""

    fingerprint: str
    label: str
    story_ids: tuple[int, ...]
    citation_item_ids: tuple[int, ...] = ()
    theme_key: str | None = None
    label_source: str | None = None
    summary: str | None = None
    status: str = "pending"
    salience: float | None = None
    salience_rank: int = 1
    cohesion: float | None = None
    min_pairwise_cohesion: float | None = None
    story_count: int | None = None
    outlet_count: int | None = None
    latest_published_at: str | datetime | None = None
    salience_story_component: float | None = None
    salience_outlet_component: float | None = None
    salience_recency_component: float | None = None
    centroid: bytes | None = None
    matched_previous_key: str | None = None
    method: str | None = None
    content_hash: str | None = None
    algorithm_version: str | None = None
    config_fingerprint: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    embedding_dimension: int | None = None


@dataclass(frozen=True)
class OtherCoverageRecord:
    """One story shown under "Other coverage", with its stated reason."""

    story_id: int
    reason: str
    position: int = 0


@dataclass(frozen=True)
class ExcludedStoryRecord:
    """One story the stage could not place, and why."""

    story_id: int
    reason: str


@dataclass(frozen=True)
class ThemeSetRecord:
    """Ticker/trading-day metadata shared by every theme in one run."""

    method: str
    method_reason: str = ""
    quality: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] | None = None
    trust_metadata: Mapping[str, Any] = field(default_factory=dict)
    config_fingerprint: str = ""
    algorithm_version: str = ""
    model_name: str | None = None
    model_revision: str | None = None
    embedding_dimension: int | None = None


@dataclass(frozen=True)
class ReconciliationReport:
    """What one atomic reconciliation actually changed.

    The id tuples cover the rows a reconciliation keys by fingerprint —
    stories, or themes.  They are not everything an operation writes:
    ``reconcile_themes`` also owns the ``theme_sets`` row and the day's
    Other Coverage and exclusion lists, none of which are themes and none
    of which have an id a caller could match against these tuples.  Those
    land in :attr:`changed_outputs`, and :attr:`changed` counts them,
    because a report that says "nothing changed" while the stored output
    is different is worse than no report at all.
    """

    inserted: tuple[int, ...] = ()
    updated: tuple[int, ...] = ()
    unchanged: tuple[int, ...] = ()
    deleted: tuple[int, ...] = ()
    invalidated: tuple[int, ...] = ()
    removed_members: int = 0
    invalidated_theme_ids: tuple[int, ...] = ()
    #: Names of the non-row-keyed outputs this reconciliation rewrote,
    #: sorted.  See :data:`AUXILIARY_OUTPUTS` for the vocabulary.
    changed_outputs: tuple[str, ...] = ()

    @property
    def counts(self) -> dict[str, int]:
        """Per-outcome counts, for ``run_log`` and the status endpoint."""

        return {
            "inserted": len(self.inserted),
            "updated": len(self.updated),
            "unchanged": len(self.unchanged),
            "deleted": len(self.deleted),
            "invalidated": len(self.invalidated),
            "removed_members": self.removed_members,
            "invalidated_themes": len(self.invalidated_theme_ids),
            "changed_outputs": len(self.changed_outputs),
        }

    @property
    def changed(self) -> bool:
        """True when this reconciliation wrote anything different.

        False therefore means a genuine replay: every owned table already
        holds exactly what this settlement would have written, and nothing
        was written at all.
        """

        return bool(
            self.inserted
            or self.updated
            or self.deleted
            or self.invalidated
            or self.changed_outputs
        )


#: The outputs a reconciliation owns that are not keyed by a row id, and so
#: cannot be reported as one.  ``theme_set`` is the ``theme_sets`` row's own
#: metadata (method, quality, trust, and the model/config fingerprints);
#: the other two are the day's Other Coverage and exclusion lists.
AUXILIARY_OUTPUTS = ("excluded", "other_coverage", "theme_set")


__all__ = [
    "AUXILIARY_OUTPUTS",
    "ExcludedStoryRecord",
    "OtherCoverageRecord",
    "ProviderConflictRecord",
    "ReconciliationReport",
    "SemanticMergeRecord",
    "StoryMemberRecord",
    "StoryRecord",
    "ThemeRecord",
    "ThemeSetRecord",
]

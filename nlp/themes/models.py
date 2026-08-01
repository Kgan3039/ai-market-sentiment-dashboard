"""Immutable data model for the M5 theme clustering stage.

Two invariants shape everything here.

**No story disappears.**  Every canonical story handed in comes back in
exactly one of three places: a theme, ``other_coverage``, or
``excluded`` with a stated reason.  :meth:`ThemeSet.accounted_story_keys`
is the partition, and the service asserts it before returning.

**A theme carries exactly the evidence a summarizer may cite.**  The
summarizer (issue #65/#80) must be able to take a theme and produce cited
sentences without reaching for anything else, and must not be able to cite
a story that is not in the theme.  :class:`ThemeEvidence` is that closed
set; :meth:`Theme.citable_item_ids` is what a citation id may resolve to.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class ClusteringMethod(str, Enum):
    """How a ticker-day's themes were produced."""

    #: HDBSCAN over story embeddings, the issue's primary algorithm.
    HDBSCAN = "hdbscan"
    #: Agglomerative fallback, used when HDBSCAN is unstable at this n.
    AGGLOMERATIVE = "agglomerative"
    #: Fewer stories than the clustering floor; stories are listed
    #: individually rather than forced into unstable groups.
    SMALL_N_FALLBACK = "small_n_fallback"


class ExclusionReason(str, Enum):
    """Why a story is in neither a theme nor ordinary other coverage."""

    #: No usable title or description, so it could not be embedded.
    NO_ENCODABLE_TEXT = "no_encodable_text"


@dataclass(frozen=True)
class ThemeStory:
    """One canonical story arriving from M3 (or from M2 when M3 is off).

    ``story_key`` is opaque: M3's ``story_fingerprint`` in the Phase 0
    pipeline.  ``item_ids`` are the raw items a citation may resolve to.
    """

    story_key: str
    ticker: str
    title: str
    description: str | None = None
    published_at: datetime | None = None
    outlets: tuple[str, ...] = ()
    item_ids: tuple[str, ...] = ()
    #: ``(item_id, outlet, url)`` for every retained source link.
    source_links: tuple[tuple[str, str, str | None], ...] = ()


@dataclass(frozen=True)
class ThemeEvidence:
    """Everything a summarizer is permitted to use about one member story.

    Deliberately a projection rather than the story itself: it carries no
    embedding, no cluster internals, and no field the citation contract in
    ``docs/PHASE_0_SPEC.md`` section 7 does not resolve.
    """

    story_key: str
    title: str
    description: str | None
    outlets: tuple[str, ...]
    published_at: datetime | None
    item_ids: tuple[str, ...]
    source_links: tuple[tuple[str, str, str | None], ...]


@dataclass(frozen=True)
class SalienceFeatures:
    """The inputs behind a theme's salience, kept for review.

    Reported rather than folded away, because "why is this theme first" is
    a question a reviewer asks about every ranked list.
    """

    story_count: int
    outlet_count: int
    #: Publication time of the theme's most recent story.
    latest_published_at: datetime | None
    #: Each component after normalization within the ticker-day, in [0, 1].
    story_component: float
    outlet_component: float
    recency_component: float


@dataclass(frozen=True)
class Theme:
    """One coherent group of canonical stories for a ticker-day."""

    #: Stable across runs while the theme's membership matches a previous
    #: run's theme closely enough; otherwise equal to ``fingerprint``.
    theme_key: str
    #: Content digest of ticker, trading day, and the sorted member set.
    fingerprint: str
    ticker: str
    trading_day: date
    #: Deterministic representative headline: the highest-salience member's
    #: title.  The LLM label from issue #65 replaces this downstream; M5
    #: never calls a model.
    label: str
    label_source: str
    #: Member story keys, canonical first, then by publication time.
    member_story_keys: tuple[str, ...]
    evidence: tuple[ThemeEvidence, ...]
    salience: float
    salience_rank: int
    salience_features: SalienceFeatures
    #: Mean pairwise cosine between members; 1.0 for a single-member theme.
    cohesion: float
    #: Unit-length mean of the member vectors, for the next run's stability
    #: matching.  Rounded so a stored value round-trips exactly.
    centroid: tuple[float, ...]
    #: The previous run's ``theme_key`` this theme was matched to.
    matched_previous_key: str | None
    method: ClusteringMethod

    @property
    def story_count(self) -> int:
        return len(self.member_story_keys)

    @property
    def outlet_count(self) -> int:
        return self.salience_features.outlet_count

    @property
    def citable_item_ids(self) -> tuple[str, ...]:
        """Every raw item id a citation in this theme may resolve to."""

        return tuple(
            sorted({item_id for entry in self.evidence for item_id in entry.item_ids})
        )


@dataclass(frozen=True)
class ExcludedStory:
    """A story the stage could not place, and why.  Never silent."""

    story_key: str
    reason: ExclusionReason


@dataclass(frozen=True)
class ThemeQuality:
    """Deterministic quality signals for one ticker-day.

    Unsupervised clustering has no objective ground truth, and none of
    these numbers pretends otherwise.  They are the things that *can* be
    checked mechanically: did every story land somewhere, are the themes
    internally closer than they are to each other, did the day land inside
    AC-4's shape.
    """

    story_count: int
    theme_count: int
    other_coverage_count: int
    excluded_count: int
    singleton_theme_count: int
    #: Mean of the themes' cohesion values; ``None`` with no themes.
    mean_cohesion: float | None
    #: Highest cosine between two different themes' centroids; ``None``
    #: with fewer than two themes.  A high value means the split is thin.
    max_inter_theme_similarity: float | None
    #: Fraction of stories placed in a theme rather than other coverage.
    theme_coverage: float
    #: Whether the day satisfies AC-4's shape for its story count.
    meets_ac4_shape: bool


@dataclass(frozen=True)
class ThemeSet:
    """One ticker-day's themes, plus everything that is not in one."""

    ticker: str
    trading_day: date
    themes: tuple[Theme, ...]
    #: Stories deliberately not in any theme: clustering noise, incoherent
    #: outliers, or every story on a day below the clustering floor.  These
    #: are shown to the reader under "Other coverage", not dropped.
    other_coverage: tuple[ThemeEvidence, ...]
    excluded: tuple[ExcludedStory, ...]
    method: ClusteringMethod
    #: Why the method is what it is, in one auditable phrase.
    method_reason: str
    quality: ThemeQuality
    config_fingerprint: str
    algorithm_version: str
    model_name: str
    model_revision: str | None

    @property
    def accounted_story_keys(self) -> tuple[str, ...]:
        """Every input story key, exactly once, sorted.

        The service asserts this equals the input set: a story that is in
        no theme, no other coverage, and no exclusion has been lost, and
        losing a story silently is the failure AC-4 is written against.
        """

        return tuple(
            sorted(
                [key for theme in self.themes for key in theme.member_story_keys]
                + [entry.story_key for entry in self.other_coverage]
                + [entry.story_key for entry in self.excluded]
            )
        )

    @property
    def is_clustered(self) -> bool:
        """True when the day was clustered rather than listed individually."""

        return self.method is not ClusteringMethod.SMALL_N_FALLBACK


@dataclass(frozen=True)
class PreviousTheme:
    """A theme from an earlier run of the same ticker-day.

    Only the identity and the centroid are needed: matching is by shape,
    so the previous run's membership and labels are irrelevant and are
    deliberately not accepted here.
    """

    theme_key: str
    centroid: tuple[float, ...]

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
    #: The day had enough stories but no partition worth shipping: the
    #: vectors were degenerate, or nothing cleared the quality floors.  No
    #: theme is invented to satisfy a count.
    NO_SEPARABLE_STRUCTURE = "no_separable_structure"


class ExclusionReason(str, Enum):
    """Why a story is in neither a theme nor ordinary other coverage."""

    #: No usable title or description, so it could not be embedded.
    NO_ENCODABLE_TEXT = "no_encodable_text"


class OtherCoverageReason(str, Enum):
    """Why a story is outside every theme but still shown to the reader.

    Recorded per story rather than inferred from the theme set.  "Other
    coverage" is four different situations wearing one name, and a reviewer
    asking why a story is not in a theme is asking which of them applies.
    """

    #: The day is below the clustering floor, so nothing was clustered.
    BELOW_CLUSTERING_FLOOR = "below_clustering_floor"
    #: The clustering algorithm called this story noise.
    CLUSTERING_NOISE = "clustering_noise"
    #: Its cluster held fewer stories than a theme must.
    BELOW_THEME_SIZE_FLOOR = "below_theme_size_floor"
    #: Its cluster was looser than a theme may be, so it was dissolved.
    BELOW_COHESION_FLOOR = "below_cohesion_floor"
    #: It contradicted the theme it would otherwise have joined.
    THEME_INCOMPATIBLE = "theme_incompatible"
    #: M2 could not settle which article this feed identity described, and
    #: M3 held it out of semantic merging.  A story whose payload identity
    #: is disputed is never grouped or handed to a summarizer as part of a
    #: narrative, but it is not dropped either.
    PROVIDER_QUARANTINE = "provider_quarantine"
    #: M3 held it out of candidate generation for some other stated reason.
    SEMANTIC_SKIP = "semantic_skip"
    #: Every story in the day sits at the same point in the embedding
    #: space, so there is no structure to cluster and no theme is invented.
    DEGENERATE_EMBEDDING_GEOMETRY = "degenerate_embedding_geometry"
    #: No partition anywhere in AC-4's band produced a theme clearing the
    #: mandatory quality floors.
    INSUFFICIENT_THEME_STRUCTURE = "insufficient_theme_structure"


@dataclass(frozen=True)
class ThemeStory:
    """One canonical story arriving from M3 (or from M2 when M3 is off).

    ``story_key`` is opaque: M3's ``story_fingerprint`` in the Phase 0
    pipeline.  ``item_ids`` are the raw items a citation may resolve to.

    The trust-bearing fields below are carried through from M2 and M3
    unchanged.  M5 never infers them: a quarantine that M5 guessed at would
    be a quarantine nobody upstream asserted.
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
    #: Distinct outlets as M3 counted them.  Kept alongside ``outlets``
    #: because M3's count is the one the salience number must agree with.
    outlet_count: int | None = None
    #: The M2 story keys M3 collapsed into this one, canonical first.
    member_story_keys: tuple[str, ...] = ()
    #: Raw item ids M2 quarantined under a provider-identity conflict.
    quarantined_member_ids: tuple[str, ...] = ()
    #: ``(namespace, provider_item_id)`` of every conflict touching this
    #: story, carried through from M2 by way of M3.
    provider_conflicts: tuple[tuple[str, str], ...] = ()
    #: M3's ``SemanticSkipReason`` value, or ``None``.
    semantic_skip_reason: str | None = None
    #: ``(left_key, right_key, similarity, reason)`` for each merge M3
    #: accepted into this story.  Audit evidence, never a merge input.
    merge_evidence: tuple[tuple[str, str, float, str], ...] = ()
    #: M3's content hash for the story, when it supplied one.
    content_hash: str | None = None

    @property
    def is_quarantined(self) -> bool:
        """True when M3 held this story out under an M2 provider conflict."""

        return self.semantic_skip_reason == "provider_quarantine"

    @property
    def is_semantically_skipped(self) -> bool:
        """True when M3 held this story out of candidate generation at all."""

        return self.semantic_skip_reason is not None


@dataclass(frozen=True)
class ThemeSourceMetadata:
    """What produced the stories M5 was handed.

    Recorded so a theme set can be traced to the dedup run behind it.  A
    theme set whose stories came from a different encoder, or from a
    differently configured M3, is a different result even when the member
    keys look the same.
    """

    #: ``"m3.semantic"`` or ``"m2.exact"`` — which stage produced the input.
    stage: str
    algorithm_version: str
    config_fingerprint: str
    model_name: str
    model_revision: str | None
    embedding_dimension: int | None
    story_count: int
    quarantined_story_count: int
    semantically_skipped_story_count: int
    merged_story_count: int


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
    #: Lowest cosine between any two members; 1.0 for a single-member theme.
    #: Reported beside the mean because a mean hides the loosest pair, and
    #: the loosest pair is what a reader notices first in a theme.
    min_pairwise_cohesion: float
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
class OtherCoverageEntry:
    """One story shown under "Other coverage", with the reason it is there."""

    evidence: ThemeEvidence
    reason: OtherCoverageReason

    @property
    def story_key(self) -> str:
        return self.evidence.story_key


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
    #: Why, in one phrase.  A day that honestly produced no theme reads as
    #: a stated degradation rather than an unexplained ``False``.
    ac4_shape_detail: str = ""
    #: Lowest ``min_pairwise_cohesion`` across the themes; ``None`` with no
    #: themes.  The day's weakest link, which a mean cannot show.
    min_pairwise_cohesion: float | None = None


@dataclass(frozen=True)
class ThemeSet:
    """One ticker-day's themes, plus everything that is not in one."""

    ticker: str
    trading_day: date
    themes: tuple[Theme, ...]
    #: Stories deliberately not in any theme: clustering noise, incoherent
    #: outliers, quarantined stories, or every story on a day below the
    #: clustering floor.  Shown to the reader under "Other coverage" with a
    #: stated reason each, not dropped.
    other_coverage: tuple[OtherCoverageEntry, ...]
    excluded: tuple[ExcludedStory, ...]
    method: ClusteringMethod
    #: Why the method is what it is, in one auditable phrase.
    method_reason: str
    quality: ThemeQuality
    config_fingerprint: str
    algorithm_version: str
    model_name: str
    model_revision: str | None
    #: Vector width the run used; ``None`` when nothing was encodable.
    embedding_dimension: int | None = None
    #: What produced the input stories, when the caller came through the
    #: bridge.  ``None`` when stories were constructed directly.
    source_metadata: ThemeSourceMetadata | None = None
    #: Every story key handed in, sorted.  Kept so the accounting below is
    #: checkable against the input without the caller holding it.
    input_story_keys: tuple[str, ...] = ()
    #: Input keys nothing accounted for.  Non-empty means a story was lost.
    missing_story_keys: tuple[str, ...] = ()
    #: Accounted keys that were never handed in.  Non-empty means one was
    #: invented, which a count of accounted stories could not distinguish
    #: from a loss.
    unexpected_story_keys: tuple[str, ...] = ()
    #: Keys appearing in more than one theme.  Non-empty means a raw item
    #: could be cited from two themes.
    duplicate_membership_keys: tuple[str, ...] = ()

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
    def complete(self) -> bool:
        """True only when the partition is exactly the input, once each.

        The production path computes the three diagnostics above and this
        reads them, so a result can never report itself complete while one
        of them is non-empty.
        """

        return not (
            self.missing_story_keys
            or self.unexpected_story_keys
            or self.duplicate_membership_keys
        )

    @property
    def other_coverage_evidence(self) -> tuple[ThemeEvidence, ...]:
        """The other-coverage entries' evidence, in order."""

        return tuple(entry.evidence for entry in self.other_coverage)

    def other_coverage_by_reason(self) -> dict[str, tuple[str, ...]]:
        """Story keys under other coverage, grouped by stated reason."""

        grouped: dict[str, list[str]] = {}
        for entry in self.other_coverage:
            grouped.setdefault(entry.reason.value, []).append(entry.story_key)
        return {reason: tuple(sorted(keys)) for reason, keys in sorted(grouped.items())}

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

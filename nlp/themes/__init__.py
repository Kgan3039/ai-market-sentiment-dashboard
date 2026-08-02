"""Phase 0 M5: theme clustering with a small-n fallback (issue #72).

Groups a ticker-day's canonical stories into 2-6 salience-ranked themes
plus "Other coverage", using HDBSCAN over M1 embeddings with the
agglomerative fallback the issue allows, and skipping clustering entirely
below four stories.

    from nlp.embeddings import EmbeddingService
    from nlp.themes import ThemeConfig, cluster_themes

    themes = cluster_themes(
        stories,
        ticker="NVDA",
        trading_day=date(2026, 3, 5),
        config=ThemeConfig(supported_tickers=TICKERS),
        encoder=EmbeddingService(),
    )

**No story disappears.** Every input comes back in exactly one theme, in
``other_coverage``, or in ``excluded`` with a stated reason, and the
function asserts that partition before returning.

M5 calls no LLM and adds no retrieval framework. It prepares the closed
evidence set the citation-safe summarizer (#65/#80) consumes — see
:class:`~nlp.themes.models.ThemeEvidence` — and stops there.
"""

from __future__ import annotations

from .bridge import (
    descriptions_from_semantic,
    source_metadata_from_exact,
    source_metadata_from_semantic,
    theme_stories_from_exact,
    theme_stories_from_semantic,
)
from .clustering import ClusterAssignment, assign_clusters
from .compatibility import (
    COMPATIBILITY_POLICY_VERSION,
    PERMITTED_DIFFERENCES,
    incompatible_members,
    story_claims,
)
from .config import ALGORITHM_VERSION, ThemeConfig
from .errors import (
    ThemeCapacityError,
    ThemeClusteringError,
    ThemeConfigError,
    ThemeEncodingError,
    ThemeError,
    ThemeInputError,
)
from .models import (
    ClusteringMethod,
    ExcludedStory,
    ExclusionReason,
    OtherCoverageEntry,
    OtherCoverageReason,
    PreviousTheme,
    SalienceFeatures,
    Theme,
    ThemeEvidence,
    ThemeQuality,
    ThemeSet,
    ThemeSourceMetadata,
    ThemeStory,
)
from .service import cluster_themes, encoder_identity, theme_fingerprint_for

__all__ = [
    "ALGORITHM_VERSION",
    "COMPATIBILITY_POLICY_VERSION",
    "PERMITTED_DIFFERENCES",
    "OtherCoverageEntry",
    "OtherCoverageReason",
    "ThemeClusteringError",
    "ThemeSourceMetadata",
    "encoder_identity",
    "incompatible_members",
    "source_metadata_from_exact",
    "source_metadata_from_semantic",
    "story_claims",
    "ClusterAssignment",
    "ClusteringMethod",
    "ExcludedStory",
    "ExclusionReason",
    "PreviousTheme",
    "SalienceFeatures",
    "Theme",
    "ThemeCapacityError",
    "ThemeConfig",
    "ThemeConfigError",
    "ThemeEncodingError",
    "ThemeError",
    "ThemeEvidence",
    "ThemeInputError",
    "ThemeQuality",
    "ThemeSet",
    "ThemeStory",
    "assign_clusters",
    "cluster_themes",
    "descriptions_from_semantic",
    "theme_fingerprint_for",
    "theme_stories_from_exact",
    "theme_stories_from_semantic",
]

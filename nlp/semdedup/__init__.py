"""Phase 0 M3: semantic deduplication (issue #70).

Merges canonical stories that describe one event in different words, using
M1 embeddings and a threshold selected on M4's labelled set.  It runs
*after* M2 and never changes it: M2 stays the deterministic, precision-first
exact stage, and M3 is the only place a similarity value participates in a
merge decision.

    from nlp.dedup import DedupConfig, deduplicate
    from nlp.embeddings import EmbeddingService
    from nlp.semdedup import (
        SemanticDedupConfig, merge_semantic_duplicates, stories_from_dedup,
    )

    exact = deduplicate(raw_items, config=DedupConfig(supported_tickers=TICKERS))
    stories = stories_from_dedup(exact, raw_items)
    result = merge_semantic_duplicates(
        stories,
        config=SemanticDedupConfig(supported_tickers=TICKERS),
        encoder=EmbeddingService(),
    )

The encoder is injected, never constructed here, so this package loads no
model and its tests need no network.

**Precision is favoured over recall, and the guards outrank the score.**
Measured on the labelled set, cosine similarity alone cannot separate
same-story rewrites from hard negatives at any threshold; see
:mod:`nlp.semdedup.evidence` for what the guards refuse and why.
"""

from __future__ import annotations

from .bridge import conflicts_by_item, stories_from_dedup, story_from_cluster
from .config import (
    ALGORITHM_VERSION,
    DEFAULT_FRAME_OVERLAP,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_WINDOW_HOURS,
    SemanticDedupConfig,
)
from .encoding import (
    StoryEncoder,
    story_text,
    validate_dimension,
    validate_model_metadata,
)
from .errors import (
    SemanticDedupCapacityError,
    SemanticDedupConfigError,
    SemanticDedupEncodingError,
    SemanticDedupError,
    SemanticDedupInputError,
)
from .evidence import VETO_REASONS
from .models import (
    RejectedPair,
    SemanticDedupResult,
    SemanticDedupStats,
    SemanticMerge,
    SemanticMergeReason,
    SemanticSkipReason,
    SemanticStory,
    SourceLink,
    StoryInput,
)
from .service import merge_semantic_duplicates, story_fingerprint_for

__all__ = [
    "ALGORITHM_VERSION",
    "DEFAULT_FRAME_OVERLAP",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DEFAULT_WINDOW_HOURS",
    "VETO_REASONS",
    "RejectedPair",
    "SemanticDedupCapacityError",
    "SemanticDedupConfig",
    "SemanticDedupConfigError",
    "SemanticDedupEncodingError",
    "SemanticDedupError",
    "SemanticDedupInputError",
    "SemanticDedupResult",
    "SemanticDedupStats",
    "SemanticMerge",
    "SemanticMergeReason",
    "SemanticSkipReason",
    "SemanticStory",
    "SourceLink",
    "StoryEncoder",
    "StoryInput",
    "merge_semantic_duplicates",
    "conflicts_by_item",
    "stories_from_dedup",
    "story_fingerprint_for",
    "story_from_cluster",
    "story_text",
    "validate_dimension",
    "validate_model_metadata",
]

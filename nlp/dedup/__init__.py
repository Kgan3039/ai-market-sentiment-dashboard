"""Phase 0 M2: the exact / near-exact deduplication core (issue #64).

A pure function that collapses identical articles and syndicated copies
into canonical clusters *before* M3's semantic merge, using the two stages
issue #64 specifies: stage 1 canonical-URL and normalized-title exact
match, stage 2 MinHash over normalized title shingles.

The core is precision-first.  One authoritative signal merges on its own —
a feed asserting that two records are its same item, while that identity is
consistent.  Every other signal (URL, exact content, exact title, MinHash
candidate) is circumstantial and passes a single compatibility gate before
any merge, so explicit disagreement always vetoes and missing information
never does.

No embeddings, no similarity threshold in any accept decision, no
randomness, no clock, no filesystem, no database.  Anything that needs an
understanding that two differently worded headlines describe one event is
M3's job: false negatives here are acceptable, false positives are not.

    from nlp.dedup import DedupConfig, RawItem, deduplicate

    config = DedupConfig(supported_tickers=["TSLA", "NVDA", "AMD", "AAPL", "META"])
    result = deduplicate(raw_items, config=config)
    result.clusters             # deterministically ordered
    result.stats                # counters, including gate vetoes
    result.provider_conflicts   # quarantined provider identities

Normalization internals, MinHash helpers, tokenizers, and signal builders
live in the submodules and are intentionally not re-exported.
"""

from __future__ import annotations

from .config import ALGORITHM_VERSION, DedupConfig
from .errors import (
    DedupCapacityError,
    DedupConfigError,
    DedupError,
    DedupInputError,
)
from .models import (
    ClusterMember,
    DedupResult,
    DedupStats,
    DeduplicatedCluster,
    MatchReason,
    ProviderConflict,
    RawItem,
)
from .service import deduplicate

__all__ = [
    "ALGORITHM_VERSION",
    "ClusterMember",
    "DedupCapacityError",
    "DedupConfig",
    "DedupConfigError",
    "DedupError",
    "DedupInputError",
    "DedupResult",
    "DedupStats",
    "DeduplicatedCluster",
    "MatchReason",
    "ProviderConflict",
    "RawItem",
    "deduplicate",
]

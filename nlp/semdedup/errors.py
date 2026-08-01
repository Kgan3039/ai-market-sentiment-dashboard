"""Error hierarchy for the M3 semantic dedup stage."""

from __future__ import annotations


class SemanticDedupError(RuntimeError):
    """Base error for semantic deduplication failures."""


class SemanticDedupInputError(ValueError, SemanticDedupError):
    """A story is structurally invalid and cannot be compared."""


class SemanticDedupConfigError(ValueError, SemanticDedupError):
    """The semantic deduplication configuration is not usable."""


class SemanticDedupEncodingError(SemanticDedupError):
    """The encoder returned something the stage cannot trust.

    Raised for a wrong vector count, a wrong or inconsistent dimension, or a
    vector the similarity function refuses.  The stage never falls back to a
    lexical comparison when embeddings fail: silently changing which
    algorithm produced a merge would make the run unexplainable.
    """


class SemanticDedupCapacityError(SemanticDedupError):
    """A ticker partition is larger than the stage will process.

    Raised *before* any output exists.  Candidate generation is exhaustive
    by design, so a caller that hits this must split the batch or raise the
    limit deliberately rather than receive a result that skipped work.
    """

    def __init__(self, ticker: str, story_count: int, limit: int) -> None:
        super().__init__(
            f"partition {ticker!r} holds {story_count} stories, above the "
            f"configured max_partition_stories={limit}"
        )
        self.ticker = ticker
        self.story_count = story_count
        self.limit = limit

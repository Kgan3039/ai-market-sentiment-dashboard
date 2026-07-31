"""Error hierarchy for the M2 dedup core."""

from __future__ import annotations


class DedupError(RuntimeError):
    """Base error for deduplication failures."""


class DedupInputError(ValueError, DedupError):
    """A raw item is structurally invalid and cannot be deduplicated."""


class DedupConfigError(ValueError, DedupError):
    """The deduplication configuration is not usable."""


class DedupCapacityError(DedupError):
    """A partition is larger than the core will process.

    Raised *before* any output is produced.  The core never returns a
    result that looks complete while silently skipping work; a caller that
    hits this must split the batch or raise the configured limit
    deliberately.
    """

    def __init__(self, ticker: str, item_count: int, limit: int) -> None:
        super().__init__(
            f"partition {ticker!r} holds {item_count} items, above the "
            f"configured max_partition_items={limit}"
        )
        self.ticker = ticker
        self.item_count = item_count
        self.limit = limit

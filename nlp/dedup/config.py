"""Strict, fingerprintable configuration for the M2 dedup core.

There is no default construction.  ``supported_tickers`` must be supplied
by the caller, because the core has no business reading files to discover
which symbols Phase 0 covers: that belongs to whatever orchestrates it.
Every setting, plus the static policies in :mod:`nlp.dedup.structural` and
:mod:`nlp.dedup.minhash`, is folded into :meth:`DedupConfig.fingerprint`,
so editing a policy invalidates cached output without anyone remembering to
bump a version constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from typing import Collection

from .content import DEDUP_CONTENT_VERSION
from .errors import DedupConfigError
from .minhash import (
    DEFAULT_CANDIDATE_SIMILARITY,
    DEFAULT_PERMUTATIONS,
    DEFAULT_SEED,
    DEFAULT_SHINGLE_SIZE,
)
from .minhash import policy_fingerprint as minhash_policy_fingerprint
from .structural import STRUCTURE_VERSION
from .structural import policy_fingerprint as structural_policy_fingerprint

#: Bumped whenever a change to normalization, identity, detection, or
#: selection can change the clusters produced from identical input.
ALGORITHM_VERSION = "m2.core.v1"

#: No text window may exceed two weeks.  Phase 0 works on trading days; a
#: longer textual reach would let a recurring headline collapse across
#: reporting periods regardless of what a caller passes.
MAX_WINDOW_HOURS = 336.0

#: Default ceiling on one ticker partition.  Candidate generation is
#: quadratic, and 2,000 items in one window measured ~54 s on the dev box
#: against a Phase 0 reality of a few dozen headlines per ticker per run.
DEFAULT_MAX_PARTITION_ITEMS = 250

_SYMBOL_SEPARATORS = frozenset(" ,;/\t\n")


def _validate_window(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DedupConfigError(f"{field_name} must be a number of hours")
    hours = float(value)
    if not math.isfinite(hours):
        raise DedupConfigError(f"{field_name} must be finite")
    if hours <= 0:
        raise DedupConfigError(f"{field_name} must be greater than zero")
    if hours > MAX_WINDOW_HOURS:
        raise DedupConfigError(f"{field_name} must not exceed {MAX_WINDOW_HOURS} hours")
    return hours


def _validate_universe(value: Collection[str]) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise DedupConfigError("supported_tickers must be a collection of symbols")
    symbols: set[str] = set()
    for symbol in value:
        if not isinstance(symbol, str):
            raise DedupConfigError("supported_tickers must contain strings")
        stripped = symbol.strip()
        if not stripped or any(
            character in _SYMBOL_SEPARATORS for character in stripped
        ):
            raise DedupConfigError(
                f"supported_tickers has an invalid symbol: {symbol!r}"
            )
        symbols.add(stripped.upper())
    if not symbols:
        raise DedupConfigError("supported_tickers must not be empty")
    return frozenset(symbols)


@dataclass(frozen=True)
class DedupConfig:
    """Immutable settings for one deduplication run.

    ``supported_tickers`` is required and has no fallback: the core rejects
    any item outside it, and there is no configuration in which the check
    is skipped.

    Windows are compared against the *span of the resulting cluster*, never
    a single pair, so transitive chaining cannot widen a merge.  They must
    satisfy ``near_exact <= exact_title <= content`` and ``url <= content``:
    a weaker signal may never reach further than a stronger one.
    """

    #: Approved Phase 0 symbols.  Required.
    supported_tickers: Collection[str]
    #: Identical normalized title *and* description.
    content_window_hours: float = 72.0
    #: Byte-identical normalized title.
    exact_title_window_hours: float = 72.0
    #: MinHash candidate verified as structurally identical.
    near_exact_window_hours: float = 36.0
    #: How far a repeated URL may reach when the only corroborating
    #: evidence is proximity in time.
    url_window_hours: float = 72.0
    #: Timestamp disagreement tolerated within one provider item id.
    provider_timestamp_tolerance_hours: float = 1.0
    #: MinHash parameters (issue #64 stage 2); see :mod:`nlp.dedup.minhash`.
    minhash_permutations: int = DEFAULT_PERMUTATIONS
    minhash_shingle_size: int = DEFAULT_SHINGLE_SIZE
    minhash_seed: str = DEFAULT_SEED
    #: Estimated-Jaccard floor for proposing a candidate pair.  A prefilter,
    #: never a merge decision.
    candidate_min_similarity: float = DEFAULT_CANDIDATE_SIMILARITY
    #: Largest ticker partition the core will process.  Above it,
    #: :class:`~nlp.dedup.errors.DedupCapacityError` is raised before any
    #: output exists.
    max_partition_items: int = DEFAULT_MAX_PARTITION_ITEMS

    def __post_init__(self) -> None:
        content = _validate_window(self.content_window_hours, "content_window_hours")
        exact = _validate_window(
            self.exact_title_window_hours, "exact_title_window_hours"
        )
        near = _validate_window(self.near_exact_window_hours, "near_exact_window_hours")
        url = _validate_window(self.url_window_hours, "url_window_hours")
        _validate_window(
            self.provider_timestamp_tolerance_hours,
            "provider_timestamp_tolerance_hours",
        )
        if near > exact:
            raise DedupConfigError(
                "near_exact_window_hours must not exceed exact_title_window_hours"
            )
        if exact > content:
            raise DedupConfigError(
                "exact_title_window_hours must not exceed content_window_hours"
            )
        if url > content:
            raise DedupConfigError(
                "url_window_hours must not exceed content_window_hours"
            )
        for name in (
            "minhash_permutations",
            "minhash_shingle_size",
            "max_partition_items",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DedupConfigError(f"{name} must be a positive integer")
        if self.minhash_permutations > 4096:
            raise DedupConfigError("minhash_permutations must not exceed 4096")
        if not isinstance(self.minhash_seed, str) or not self.minhash_seed.strip():
            raise DedupConfigError("minhash_seed must be a non-empty string")
        similarity = self.candidate_min_similarity
        if (
            isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(float(similarity))
            or not 0.0 <= float(similarity) <= 1.0
        ):
            raise DedupConfigError("candidate_min_similarity must be in [0.0, 1.0]")
        object.__setattr__(
            self, "supported_tickers", _validate_universe(self.supported_tickers)
        )

    @property
    def ticker_universe(self) -> frozenset[str]:
        """The enforced symbols, always resolved."""

        assert isinstance(self.supported_tickers, frozenset)  # set in __post_init__
        return self.supported_tickers

    @property
    def content_window(self) -> timedelta:
        """Window for identical title-and-description merges."""

        return timedelta(hours=float(self.content_window_hours))

    @property
    def exact_title_window(self) -> timedelta:
        """Window for identical normalized-title merges."""

        return timedelta(hours=float(self.exact_title_window_hours))

    @property
    def near_exact_window(self) -> timedelta:
        """Window for verified MinHash candidates."""

        return timedelta(hours=float(self.near_exact_window_hours))

    @property
    def url_window(self) -> timedelta:
        """How far temporal proximity alone may corroborate a repeated URL."""

        return timedelta(hours=float(self.url_window_hours))

    @property
    def provider_timestamp_tolerance(self) -> timedelta:
        """Timestamp disagreement tolerated within one provider item id."""

        return timedelta(hours=float(self.provider_timestamp_tolerance_hours))

    def fingerprint(self) -> str:
        """Return a stable digest of the settings and every static policy."""

        payload = {
            "algorithm_version": ALGORITHM_VERSION,
            "content_version": DEDUP_CONTENT_VERSION,
            "structure_version": STRUCTURE_VERSION,
            "structural_policy": structural_policy_fingerprint(),
            "minhash_policy": minhash_policy_fingerprint(),
            "content_window_hours": float(self.content_window_hours),
            "exact_title_window_hours": float(self.exact_title_window_hours),
            "near_exact_window_hours": float(self.near_exact_window_hours),
            "url_window_hours": float(self.url_window_hours),
            "provider_timestamp_tolerance_hours": float(
                self.provider_timestamp_tolerance_hours
            ),
            "minhash_permutations": self.minhash_permutations,
            "minhash_shingle_size": self.minhash_shingle_size,
            "minhash_seed": self.minhash_seed,
            "candidate_min_similarity": float(self.candidate_min_similarity),
            "max_partition_items": self.max_partition_items,
            "supported_tickers": sorted(self.ticker_universe),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

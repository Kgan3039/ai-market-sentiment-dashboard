"""Strict, fingerprintable configuration for the M3 semantic dedup stage.

As in M2 there is no default construction: ``supported_tickers`` must be
supplied, because a stage has no business reading files to discover which
symbols Phase 0 covers.  Every setting and every static policy is folded
into :meth:`SemanticDedupConfig.fingerprint`, so editing a guard lexicon
invalidates cached output without anyone remembering to bump a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
from typing import Collection

from nlp.dedup.text import TICKER_PATTERN

from .errors import SemanticDedupConfigError
from .evidence import policy_components as evidence_policy_components
from .evidence import policy_fingerprint as evidence_policy_fingerprint

#: Bumped when a change to candidate generation, the predicate, or story
#: assembly can change the stories produced from identical input.
ALGORITHM_VERSION = "m3.semantic.v1"

#: Issue #70 specifies a (ticker, +/-36h) comparison window.  A longer
#: reach would let a recurring headline collapse across reporting periods,
#: so the configured value may widen for an experiment but never past a
#: week.
DEFAULT_WINDOW_HOURS = 36.0
MAX_WINDOW_HOURS = 168.0

#: Selected on the M4 labelled set, not chosen by intuition, and
#: re-derived from scratch whenever the guards change.
#:
#: The authoritative numbers are the generated ``selection`` block in
#: ``nlp/eval/data/results/m3_threshold_sweep.json``; no metric is restated
#: here, because a comment cannot be regenerated and drifts.  The *policy*
#: is what this constant records: take the lowest tested threshold at which
#: no false merge survives, then step above it far enough that the choice
#: does not rest on a single observation, provided recall still clears
#: AC-3's floor.  The F1 maximum is never taken - it buys F1 with false
#: merges, and a false merge is the failure this stage exists to prevent.
#:
#: **Provisional.**  Selected on a synthetic, single-author, unadjudicated
#: development dataset that is not gate eligible.  A development default,
#: not an accepted operating point.
#:
#: It is the *floor* under a merge that has already survived every
#: contradiction guard, never a merge rule on its own.
DEFAULT_SIMILARITY_THRESHOLD = 0.70

#: Lexical overlap at or above which two headlines are treated as the same
#: frame, so that a substitution inside it is a different event.
DEFAULT_FRAME_OVERLAP = 0.5

#: Candidate generation is exhaustive within a ticker window; Phase 0 sees
#: a few dozen canonical stories per ticker per day.
DEFAULT_MAX_PARTITION_STORIES = 250

#: The exact text handed to the encoder, named so a change to it moves the
#: fingerprint: M1's ``compose_embedding_text`` over the canonical title and
#: the chosen standfirst.
SEMANTIC_INPUT_COMPOSITION = "m1.compose_embedding_text(title, description)"

_POLICY_FINGERPRINTS = {
    "evidence_policy": evidence_policy_fingerprint,
}


def _validate_unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticDedupConfigError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise SemanticDedupConfigError(f"{field} must be in [0.0, 1.0]")
    return number


def _validate_universe(value: Collection[str]) -> frozenset[str]:
    """Validate a ticker universe against the same syntax M2 enforces."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise SemanticDedupConfigError(
            "supported_tickers must be a collection of symbols"
        )
    symbols: set[str] = set()
    for symbol in value:
        if not isinstance(symbol, str):
            raise SemanticDedupConfigError("supported_tickers must contain strings")
        stripped = symbol.strip()
        if not stripped:
            raise SemanticDedupConfigError(
                "supported_tickers must not contain blank symbols"
            )
        normalized = stripped.upper()
        if not TICKER_PATTERN.match(normalized):
            raise SemanticDedupConfigError(
                f"supported_tickers has an invalid symbol: {symbol!r}"
            )
        if normalized in symbols:
            raise SemanticDedupConfigError(
                f"supported_tickers contains a duplicate symbol: {symbol!r}"
            )
        symbols.add(normalized)
    if not symbols:
        raise SemanticDedupConfigError("supported_tickers must not be empty")
    return frozenset(symbols)


@dataclass(frozen=True)
class SemanticDedupConfig:
    """Immutable settings for one semantic deduplication run."""

    #: Approved Phase 0 symbols.  Required; merges never cross a ticker.
    supported_tickers: Collection[str]
    #: Cosine floor applied to every pair of a prospective story.
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    #: Comparison window between any two members, issue #70's +/-36h.
    window_hours: float = DEFAULT_WINDOW_HOURS
    #: Jaccard overlap at which the ``same_frame`` guard engages.
    frame_overlap_threshold: float = DEFAULT_FRAME_OVERLAP
    #: Largest ticker partition the stage will process.
    max_partition_stories: int = DEFAULT_MAX_PARTITION_STORIES
    #: Whether an undated story may merge.  Off: without a timestamp the
    #: +/-36h window cannot be enforced, and an unbounded semantic merge is
    #: exactly how a recurring headline collapses across quarters.
    allow_undated_merges: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "similarity_threshold",
            _validate_unit_interval(self.similarity_threshold, "similarity_threshold"),
        )
        object.__setattr__(
            self,
            "frame_overlap_threshold",
            _validate_unit_interval(
                self.frame_overlap_threshold, "frame_overlap_threshold"
            ),
        )
        if isinstance(self.window_hours, bool) or not isinstance(
            self.window_hours, (int, float)
        ):
            raise SemanticDedupConfigError("window_hours must be a number of hours")
        hours = float(self.window_hours)
        if not math.isfinite(hours) or hours <= 0:
            raise SemanticDedupConfigError("window_hours must be greater than zero")
        if hours > MAX_WINDOW_HOURS:
            raise SemanticDedupConfigError(
                f"window_hours must not exceed {MAX_WINDOW_HOURS} hours"
            )
        object.__setattr__(self, "window_hours", hours)
        if (
            isinstance(self.max_partition_stories, bool)
            or not isinstance(self.max_partition_stories, int)
            or self.max_partition_stories <= 0
        ):
            raise SemanticDedupConfigError(
                "max_partition_stories must be a positive integer"
            )
        if not isinstance(self.allow_undated_merges, bool):
            raise SemanticDedupConfigError("allow_undated_merges must be a boolean")
        object.__setattr__(
            self, "supported_tickers", _validate_universe(self.supported_tickers)
        )

    @property
    def ticker_universe(self) -> frozenset[str]:
        """The enforced symbols, always resolved."""

        assert isinstance(self.supported_tickers, frozenset)  # set in __post_init__
        return self.supported_tickers

    @property
    def window(self) -> timedelta:
        """How far apart two stories may be published and still compare."""

        return timedelta(hours=float(self.window_hours))

    def fingerprint_components(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_dimension: int | None = None,
    ) -> dict[str, object]:
        """Return every behaviour-changing input, named.

        The fingerprint is a digest of exactly this map, and every guard
        lexicon, pattern, scope and ordering reaches it through
        :func:`nlp.semdedup.evidence.policy_components`.  A new rule
        registered there changes the digest without anyone bumping a
        constant by hand.
        """

        components: dict[str, object] = {
            "algorithm_version": ALGORITHM_VERSION,
            "semantic_input_composition": SEMANTIC_INPUT_COMPOSITION,
            "similarity_threshold": float(self.similarity_threshold),
            "window_hours": float(self.window_hours),
            "frame_overlap_threshold": float(self.frame_overlap_threshold),
            "max_partition_stories": self.max_partition_stories,
            "allow_undated_merges": self.allow_undated_merges,
            "supported_tickers": sorted(self.ticker_universe),
            "model_name": model_name,
            "model_revision": model_revision,
            "embedding_dimension": embedding_dimension,
        }
        components.update(
            {
                f"evidence.{name}": value
                for name, value in evidence_policy_components().items()
            }
        )
        # Imported here rather than at module scope: service imports config.
        from .service import cluster_compatibility_components

        components.update(
            {
                f"cluster_compatibility.{name}": value
                for name, value in cluster_compatibility_components().items()
            }
        )
        return components

    def fingerprint(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_dimension: int | None = None,
    ) -> str:
        """Return a stable digest of the settings, the policies, and the model.

        The encoder identity and vector width are part of it: the same
        stories under the same settings but a different model are a
        different result, and a cache that ignored that would serve merges
        nobody can reproduce.
        """

        payload = self.fingerprint_components(
            model_name=model_name,
            model_revision=model_revision,
            embedding_dimension=embedding_dimension,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

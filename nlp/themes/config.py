"""Strict, fingerprintable configuration for M5 theme clustering.

Defaults come from AC-4 and issue #72 rather than from taste: 2-6 themes,
a clustering floor of 4 stories, noise into "Other coverage".
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Collection

from nlp.dedup.text import TICKER_PATTERN
from nlp.semdedup.evidence import policy_fingerprint as evidence_policy_fingerprint

from .errors import ThemeConfigError

#: Bumped when a change to clustering, salience, or theme assembly can
#: change the themes produced from identical input.
ALGORITHM_VERSION = "m5.themes.v1"

#: AC-4: "ticker-days with <4 stories skip clustering and list stories
#: individually".
DEFAULT_MIN_STORIES = 4
#: AC-4: "each ticker-day with >=4 canonical stories produces 2-6 themes".
DEFAULT_MIN_THEMES = 2
DEFAULT_MAX_THEMES = 6

#: Phase 0 sees a few dozen canonical stories per ticker per day.  Beyond
#: this the stage refuses rather than returning a partial day.
DEFAULT_MAX_STORIES_PER_DAY = 250


def _unit_interval(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThemeConfigError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ThemeConfigError(f"{field} must be in [0.0, 1.0]")
    return number


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ThemeConfigError(f"{field} must be a positive integer")
    return value


def _validate_universe(value: Collection[str]) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise ThemeConfigError("supported_tickers must be a collection of symbols")
    symbols: set[str] = set()
    for symbol in value:
        if not isinstance(symbol, str):
            raise ThemeConfigError("supported_tickers must contain strings")
        normalized = symbol.strip().upper()
        if not normalized:
            raise ThemeConfigError("supported_tickers must not contain blank symbols")
        if not TICKER_PATTERN.match(normalized):
            raise ThemeConfigError(
                f"supported_tickers has an invalid symbol: {symbol!r}"
            )
        if normalized in symbols:
            raise ThemeConfigError(
                f"supported_tickers contains a duplicate symbol: {symbol!r}"
            )
        symbols.add(normalized)
    if not symbols:
        raise ThemeConfigError("supported_tickers must not be empty")
    return frozenset(symbols)


@dataclass(frozen=True)
class ThemeConfig:
    """Immutable settings for one ticker-day clustering run."""

    supported_tickers: Collection[str]
    #: Below this many stories the day is listed individually (AC-4).
    min_stories_for_clustering: int = DEFAULT_MIN_STORIES
    #: AC-4's theme-count band.  A clustering outside it is "unstable" and
    #: triggers the agglomerative fallback issue #72 allows.
    min_themes: int = DEFAULT_MIN_THEMES
    max_themes: int = DEFAULT_MAX_THEMES
    #: HDBSCAN parameters.  The library default of 5 is far too large for a
    #: Phase 0 ticker-day, where a real theme is often two or three stories.
    min_cluster_size: int = 2
    min_samples: int = 1
    #: The looseness at which a group stops being a theme.  It does two
    #: jobs: an HDBSCAN clustering with a cluster below it is treated as
    #: unstable and re-run agglomeratively, and a cluster still below it
    #: afterwards is dissolved into other coverage.  Together these are what
    #: stop a giant catch-all of unrelated market language from being
    #: presented to a reader as a theme.  Calibrated on the committed
    #: ticker-days: real strands there hold 0.42-0.72, the grab-bag 0.35.
    min_theme_cohesion: float = 0.40
    #: Fewest stories a group must hold to be shown as a theme.  A lone
    #: story is coverage, not a theme; it goes under "Other coverage" where
    #: a reader can still see it.
    min_theme_stories: int = 2
    #: Centroid cosine at which a theme is treated as the same theme as one
    #: from the previous run, so an unchanged theme keeps its identity.
    stability_threshold: float = 0.90
    #: Salience weights; normalized internally, so only their ratio matters.
    story_weight: float = 0.4
    outlet_weight: float = 0.3
    recency_weight: float = 0.3
    #: Half-life of the recency component, in hours before the day's most
    #: recent story.  Measured against the data, never against a clock, so
    #: replaying a stored day (AC-8) ranks it the same way next year.
    recency_half_life_hours: float = 6.0
    max_stories_per_day: int = DEFAULT_MAX_STORIES_PER_DAY

    def __post_init__(self) -> None:
        for field in (
            "min_stories_for_clustering",
            "min_themes",
            "max_themes",
            "min_cluster_size",
            "min_samples",
            "min_theme_stories",
            "max_stories_per_day",
        ):
            _positive_int(getattr(self, field), field)
        if self.min_themes > self.max_themes:
            raise ThemeConfigError("min_themes must not exceed max_themes")
        if self.min_cluster_size < 2:
            raise ThemeConfigError("min_cluster_size must be at least 2")
        object.__setattr__(
            self,
            "min_theme_cohesion",
            _unit_interval(self.min_theme_cohesion, "min_theme_cohesion"),
        )
        object.__setattr__(
            self,
            "stability_threshold",
            _unit_interval(self.stability_threshold, "stability_threshold"),
        )
        weights = []
        for field in ("story_weight", "outlet_weight", "recency_weight"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ThemeConfigError(f"{field} must be a number")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ThemeConfigError(f"{field} must be finite and non-negative")
            object.__setattr__(self, field, number)
            weights.append(number)
        if sum(weights) <= 0:
            raise ThemeConfigError("at least one salience weight must be positive")
        half_life = self.recency_half_life_hours
        if (
            isinstance(half_life, bool)
            or not isinstance(half_life, (int, float))
            or not math.isfinite(float(half_life))
            or float(half_life) <= 0
        ):
            raise ThemeConfigError("recency_half_life_hours must be positive")
        object.__setattr__(self, "recency_half_life_hours", float(half_life))
        object.__setattr__(
            self, "supported_tickers", _validate_universe(self.supported_tickers)
        )

    @property
    def ticker_universe(self) -> frozenset[str]:
        assert isinstance(self.supported_tickers, frozenset)  # set in __post_init__
        return self.supported_tickers

    @property
    def salience_weights(self) -> tuple[float, float, float]:
        """The three weights normalized to sum to one."""

        total = self.story_weight + self.outlet_weight + self.recency_weight
        return (
            self.story_weight / total,
            self.outlet_weight / total,
            self.recency_weight / total,
        )

    def fingerprint(self, *, model_name: str, model_revision: str | None) -> str:
        """Return a stable digest of the settings, policies, and encoder."""

        payload = {
            "algorithm_version": ALGORITHM_VERSION,
            "coherence_policy": evidence_policy_fingerprint(),
            "min_stories_for_clustering": self.min_stories_for_clustering,
            "min_themes": self.min_themes,
            "max_themes": self.max_themes,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "min_theme_cohesion": self.min_theme_cohesion,
            "min_theme_stories": self.min_theme_stories,
            "stability_threshold": self.stability_threshold,
            "salience_weights": list(self.salience_weights),
            "recency_half_life_hours": self.recency_half_life_hours,
            "max_stories_per_day": self.max_stories_per_day,
            "supported_tickers": sorted(self.ticker_universe),
            "model_name": model_name,
            "model_revision": model_revision,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

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

#: The exact text handed to the encoder, named so a change to it moves the
#: fingerprint.  The same composition M1 and M3 use, so a ticker-day costs
#: no extra model calls when both stages run.
SEMANTIC_INPUT_COMPOSITION = "m1.compose_embedding_text(title, description)"

#: Namespace prefix for theme fingerprints, so a digest computed here can
#: never collide with one computed by another stage over the same fields.
THEME_NAMESPACE = "m5.theme.v1"

#: Static policies, stated rather than implied, and fingerprinted.  Each
#: names a decision a reader of the output would otherwise have to infer
#: from the code.
LABEL_POLICY = (
    "leading_member_title; deterministic; no model call; label never "
    "participates in membership identity"
)
OUTLIER_POLICY = "clustering_noise_to_other_coverage_never_dropped"
OTHER_COVERAGE_POLICY = (
    "explicit_entry_per_story_with_stated_reason; never a narrative theme"
)
RANKING_POLICY = "salience = weighted(story_count, outlet_diversity, recency)"
TIE_BREAK_POLICY = "salience_desc, earliest_member, theme_fingerprint"
STORY_ORDERING_POLICY = "undated_last, published_at_asc, story_key_asc"
QUARANTINE_POLICY = (
    "upstream_quarantined_and_semantically_skipped_stories_held_out_of "
    "_clustering_and_shown_under_other_coverage"
)

#: Decimal places every reported score is rounded to, on one code path, so
#: a committed artifact is byte-stable across machines.
SCORE_PRECISION = 6
#: Centroids round to the same precision, so a stored centroid read back
#: matches the one the next run computes - which theme identity depends on.
CENTROID_PRECISION = 6
#: Precision of the committed story vectors the offline evaluation reads.
VECTOR_PRECISION = 6

#: Below this maximum pairwise cosine *distance* the day's vectors carry no
#: separable structure at all: every story is the same point.
DEGENERATE_GEOMETRY_EPSILON = 1e-9

DEGENERATE_GEOMETRY_POLICY = (
    "no_separable_structure_produces_no_theme; every story goes to other "
    "coverage with a stated reason; meets_ac4_shape reports false rather "
    "than inventing a split to satisfy the band"
)
SUBSET_EXTRACTION_POLICY = (
    "a cluster failing a floor sheds its least-central member repeatedly, "
    "ties by story order, until both floors hold or it falls below the size "
    "floor; shed members go to other coverage with a stated reason"
)
LABEL_GENERICITY_POLICY = (
    "representative = member title scored by distinct informative tokens, "
    "entity and numeral evidence, and repetition penalty; then outlet_count, "
    "recency, story_key. Verbatim member title only; never synthesized"
)
SUMMARIZATION_ADAPTER_POLICY = (
    "theme members only; one MemberStory per member story_key; citations "
    "closed over theme membership; other coverage and excluded rejected"
)
ACCOUNTING_CONTRACT = (
    "input == themes + other_coverage + excluded, exactly once each; "
    "complete is false when any diagnostic is non-empty"
)
OUTLET_COUNT_POLICY = (
    "upstream ThemeStory.outlet_count is authoritative when present; "
    "len(outlets) is a fallback, never a substitute"
)
PERTURBATION_SELECTION_POLICY = "drop_the_least_recent_story"
PERMUTATION_SELECTION_POLICY = "reverse_then_interleave_never_random"
STABILITY_MATCHING_ALGORITHM = (
    "greedy_one_to_one_centroid_cosine_descending_ties_by_previous_key"
)
STABILITY_MEMBERSHIP_FORMULA = "|exact member sets in both| / |baseline theme count|"
STABILITY_IDENTITY_FORMULA = (
    "|baseline theme_keys carried into the new run| / |baseline theme count|"
)
STABILITY_MATCHED_OF_NEW_FORMULA = (
    "|new themes with any match| / |new theme count|  (the weak denominator)"
)
STABILITY_STORY_RETENTION_FORMULA = (
    "|baseline in-theme stories still in a theme| / "
    "|baseline in-theme stories still present|"
)


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
    #: presented to a reader as a theme.
    #:
    #: **Not independently validated.**  The value was chosen by looking at
    #: the committed ticker-days - where authored strands hold 0.42-0.72 and
    #: the grab-bag holds 0.35 - and those are the same days it is then
    #: evaluated on, which makes every cohesion number the fixtures produce
    #: partly a restatement of the threshold.  It cannot be calibrated
    #: honestly until real ingested days exist (#57/#68) and a human review
    #: (#60/K3) says which groups a reader would accept.  Until then treat
    #: it as a development default, and read any theme sitting within a few
    #: hundredths of it as unadjudicated.
    min_theme_cohesion: float = 0.40
    #: The *weakest pair* a theme may contain.  Separate from the mean
    #: because a mean hides its worst link: the six-story TSLA theme held a
    #: mean of 0.4205 over a pair at 0.2676, and it was the pair a reader
    #: would have noticed.  A theme must clear **both** floors.
    #:
    #: Unlike the mean floor this one is not read off M5's own fixture. M4's
    #: labelled pair set - a different dataset, built for a different stage -
    #: measured that genuine rewrites of one *event* score 0.42-0.73 under
    #: this encoder. A theme is coarser than an event, so its weakest link
    #: belongs below that range; 0.30 sits under the loosest genuine
    #: same-event pair while still excluding pairs that share almost no
    #: semantic content. That is a cross-reference, not a calibration: it
    #: still needs real days (#57/#68) and the #60/K3 review.
    min_theme_pairwise_cohesion: float = 0.30
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
    #: A theme whose cohesion clears the floor by less than this is reported
    #: ``near_cohesion_floor``: it survives on a threshold that is not
    #: independently calibrated, and saying so is more use than a pass.
    near_cohesion_floor_margin: float = 0.05
    #: Maximum pairwise cosine distance at which a day counts as having no
    #: separable structure at all.
    degenerate_geometry_epsilon: float = DEGENERATE_GEOMETRY_EPSILON

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
            "min_theme_pairwise_cohesion",
            _unit_interval(
                self.min_theme_pairwise_cohesion, "min_theme_pairwise_cohesion"
            ),
        )
        if self.min_theme_pairwise_cohesion > self.min_theme_cohesion:
            raise ThemeConfigError(
                "min_theme_pairwise_cohesion must not exceed min_theme_cohesion; "
                "a mean cannot be below its own minimum"
            )
        object.__setattr__(
            self,
            "degenerate_geometry_epsilon",
            _unit_interval(
                self.degenerate_geometry_epsilon, "degenerate_geometry_epsilon"
            ),
        )
        object.__setattr__(
            self,
            "near_cohesion_floor_margin",
            _unit_interval(
                self.near_cohesion_floor_margin, "near_cohesion_floor_margin"
            ),
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

    def fingerprint_components(
        self,
        *,
        model_name: str,
        model_revision: str | None,
        embedding_dimension: int | None = None,
    ) -> dict[str, object]:
        """Return every behaviour-changing input, named.

        The fingerprint is a digest of exactly this map.  Static policies —
        the compatibility contract, the fallback objective, the label rule,
        the other-coverage vocabulary, the ordering rules — reach it
        through the modules that own them, so adding a term to a lexicon or
        a reason to an enum invalidates cached themes without anyone
        bumping a constant by hand.
        """

        # Imported here rather than at module scope: both import config.
        from .bridge import DESCRIPTION_SELECTION_POLICY
        from .clustering import FALLBACK_SELECTION_POLICY
        from .summarization import adapter_policy_components
        from .trust import STAGE_TRUST_VERSION
        from .compatibility import policy_components as compatibility_components
        from .models import ExclusionReason, OtherCoverageReason

        components: dict[str, object] = {
            "algorithm_version": ALGORITHM_VERSION,
            "semantic_input_composition": SEMANTIC_INPUT_COMPOSITION,
            "min_stories_for_clustering": self.min_stories_for_clustering,
            "min_themes": self.min_themes,
            "max_themes": self.max_themes,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "min_theme_cohesion": self.min_theme_cohesion,
            "min_theme_pairwise_cohesion": self.min_theme_pairwise_cohesion,
            "near_cohesion_floor_margin": self.near_cohesion_floor_margin,
            "min_theme_stories": self.min_theme_stories,
            "stability_threshold": self.stability_threshold,
            "salience_weights": list(self.salience_weights),
            "recency_half_life_hours": self.recency_half_life_hours,
            "max_stories_per_day": self.max_stories_per_day,
            "supported_tickers": sorted(self.ticker_universe),
            "model_name": model_name,
            "model_revision": model_revision,
            "embedding_dimension": embedding_dimension,
            "label_policy": LABEL_POLICY,
            "outlier_policy": OUTLIER_POLICY,
            "other_coverage_policy": OTHER_COVERAGE_POLICY,
            "other_coverage_reasons": ",".join(
                sorted(reason.value for reason in OtherCoverageReason)
            ),
            "exclusion_reasons": ",".join(
                sorted(reason.value for reason in ExclusionReason)
            ),
            "ranking_policy": RANKING_POLICY,
            "tie_break_policy": TIE_BREAK_POLICY,
            "story_ordering_policy": STORY_ORDERING_POLICY,
            "theme_namespace": THEME_NAMESPACE,
            "quarantine_policy": QUARANTINE_POLICY,
            "score_precision": SCORE_PRECISION,
            "centroid_precision": CENTROID_PRECISION,
            "vector_precision": VECTOR_PRECISION,
            "degenerate_geometry_policy": DEGENERATE_GEOMETRY_POLICY,
            "degenerate_geometry_epsilon": self.degenerate_geometry_epsilon,
            "subset_extraction_policy": SUBSET_EXTRACTION_POLICY,
            "label_genericity_policy": LABEL_GENERICITY_POLICY,
            "summarization_adapter_policy": SUMMARIZATION_ADAPTER_POLICY,
            "stage_trust_version": STAGE_TRUST_VERSION,
            "accounting_contract": ACCOUNTING_CONTRACT,
            "description_selection_policy": DESCRIPTION_SELECTION_POLICY,
            "outlet_count_policy": OUTLET_COUNT_POLICY,
            "perturbation_selection_policy": PERTURBATION_SELECTION_POLICY,
            "permutation_selection_policy": PERMUTATION_SELECTION_POLICY,
            "stability_matching_algorithm": STABILITY_MATCHING_ALGORITHM,
            "stability_membership_formula": STABILITY_MEMBERSHIP_FORMULA,
            "stability_identity_formula": STABILITY_IDENTITY_FORMULA,
            "stability_matched_of_new_formula": STABILITY_MATCHED_OF_NEW_FORMULA,
            "stability_story_retention_formula": STABILITY_STORY_RETENTION_FORMULA,
        }
        components.update(
            {
                f"fallback.{name}": value
                for name, value in FALLBACK_SELECTION_POLICY.items()
            }
        )
        components.update(
            {
                f"summarization_adapter.{name}": value
                for name, value in adapter_policy_components().items()
            }
        )
        components.update(
            {
                f"compatibility.{name}": value
                for name, value in compatibility_components().items()
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
        """Return a stable digest of the settings, policies, and encoder."""

        payload = self.fingerprint_components(
            model_name=model_name,
            model_revision=model_revision,
            embedding_dimension=embedding_dimension,
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

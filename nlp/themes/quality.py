"""Deterministic theme-quality evaluation for issue #72.

Unsupervised clustering has no objective ground truth and nothing here
pretends otherwise.  What *can* be checked mechanically is checked, and the
rest is left to the human review K3 (#60) exists for:

* **coverage** — did every story land somewhere, and how many are in a theme
  rather than in "Other coverage"?
* **shape** — does the day satisfy AC-4's band for its story count?
* **coherence** — are themes internally closer than they are to each other?
  A high inter-theme similarity means the split is thin, not that it is
  wrong.
* **stability** — do the themes survive a permutation of the input, and a
  small perturbation of the day (one story added or removed)?
* **runtime** — measured, and reported as a measurement rather than a bound.

None of these says a theme is *right*.  A day can score perfectly here and
still group two stories a reader would separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import time
from typing import Any, Sequence

from .config import ThemeConfig, clears
from .models import PreviousTheme, ThemeSet, ThemeStory
from .service import cluster_themes


@dataclass(frozen=True)
class StabilityReport:
    """How the day's themes held up under a change.

    **The two retention numbers measure different things and one is much
    weaker than the other.**  Reported together, and named, because the
    earlier pairing of a 0.20 membership retention with a 1.00 identity
    retention read as "stable" and meant almost the opposite.

    ``membership_retained`` is the strong one: the fraction of the
    baseline's themes whose *exact member set* still exists afterwards.
    0.20 means four themes in five were restructured.

    ``identity_retained`` is the weak one: the fraction of the baseline's
    themes whose ``theme_key`` still appears afterwards.  It says
    identifiers were carried across, not that the themes behind them held
    their stories.  It is computed over the **baseline's** themes on
    purpose — dividing by the perturbed run's themes made a collapse from
    five themes to one score 1.00, because the one survivor matched.

    ``matched_fraction_of_new`` is that weaker denominator, kept only so
    the difference is visible rather than hidden.
    """

    membership_retained: float
    identity_retained: float
    matched_fraction_of_new: float
    #: Fraction of the baseline's in-theme stories still in some theme.
    stories_retained_in_themes: float
    theme_count_before: int
    theme_count_after: int

    @property
    def interpretation(self) -> str:
        """One sentence a reader cannot misread as overall stability."""

        if self.membership_retained >= self.identity_retained:
            return (
                f"{self.membership_retained:.2f} of the baseline's themes kept "
                "their exact membership."
            )
        return (
            f"only {self.membership_retained:.2f} of the baseline's themes kept "
            f"their exact membership, while {self.identity_retained:.2f} kept an "
            "identifier; stable identifiers over restructured membership is not "
            "stable clustering."
        )


@dataclass(frozen=True)
class ThemeDetail:
    """Everything checkable about one theme, per the issue's audit list."""

    rank: int
    label: str
    label_source: str
    fingerprint: str
    theme_key: str
    member_count: int
    #: Exact membership, sorted, so the partition is in the record.
    member_story_keys: tuple[str, ...]
    outlet_count: int
    outlets: tuple[str, ...]
    representative_story_key: str
    salience: float
    #: Mean pairwise cosine between members.
    cohesion: float
    #: Lowest cosine between any two members: what the loosest pair looks
    #: like, which a mean can hide.
    min_pairwise_cohesion: float
    #: ``cohesion - min_theme_cohesion``.  A small positive margin means the
    #: theme survives only because of a threshold that is not independently
    #: calibrated; see ``ThemeConfig.min_theme_cohesion``.
    cohesion_margin: float
    near_cohesion_floor: bool
    method: str
    matched_previous_key: str | None


@dataclass(frozen=True)
class TickerDayReport:
    """One ticker-day, measured."""

    ticker: str
    trading_day: date
    volume: str
    story_count: int
    theme_count: int
    other_coverage_count: int
    excluded_count: int
    singleton_theme_count: int
    method: str
    method_reason: str
    mean_cohesion: float | None
    #: The day's weakest link: the lowest ``min_pairwise_cohesion`` across
    #: its themes.  A mean cannot show it, and it is the pair a reader
    #: notices first.
    min_pairwise_cohesion: float | None
    max_inter_theme_similarity: float | None
    theme_coverage: float
    meets_ac4_shape: bool
    #: Why, in one phrase, so an honest ``False`` is not an unexplained one.
    ac4_shape_detail: str
    no_story_lost: bool
    #: Ranked themes as ``(rank, label, story_count, outlet_count, salience)``.
    themes: tuple[tuple[int, str, int, int, float], ...]
    #: The full per-theme audit record.
    theme_details: tuple[ThemeDetail, ...]
    other_coverage: tuple[str, ...]
    other_coverage_by_reason: dict[str, tuple[str, ...]]
    excluded_by_reason: dict[str, tuple[str, ...]]
    #: The exact partition: sorted member sets, sorted.
    partition: tuple[tuple[str, ...], ...]
    #: Input keys minus accounted keys, and the reverse.  Both empty is the
    #: no-story-loss proof; a count alone could not distinguish a lost story
    #: from an invented one.
    missing_story_keys: tuple[str, ...]
    unexpected_story_keys: tuple[str, ...]
    duplicate_membership_keys: tuple[str, ...]
    permutation_stable: bool
    #: Membership under a permuted input, so an unstable permutation shows
    #: what changed rather than only that something did.
    permutation: StabilityReport
    perturbation: StabilityReport
    rerun_keeps_identity: bool
    config_fingerprint: str
    algorithm_version: str
    model_name: str
    model_revision: str | None
    embedding_dimension: int | None
    elapsed_seconds: float


def _membership(theme_set: ThemeSet) -> set[frozenset[str]]:
    return {frozenset(theme.member_story_keys) for theme in theme_set.themes}


def _previous(theme_set: ThemeSet) -> tuple[PreviousTheme, ...]:
    return tuple(
        PreviousTheme(theme_key=theme.theme_key, centroid=theme.centroid)
        for theme in theme_set.themes
    )


def _permuted(stories: Sequence[ThemeStory]) -> list[ThemeStory]:
    """A reversed, then interleaved, ordering — never a random one."""

    reversed_stories = list(reversed(stories))
    front = reversed_stories[: len(reversed_stories) // 2]
    back = reversed_stories[len(reversed_stories) // 2 :]
    return [story for pair in zip(back, front) for story in pair] + (
        back[len(front) :] if len(back) > len(front) else []
    )


def _stability(baseline: ThemeSet, perturbed: ThemeSet) -> StabilityReport:
    before = _membership(baseline)
    after = _membership(perturbed)
    retained = len(before & after) / len(before) if before else 1.0
    carried = {
        theme.matched_previous_key
        for theme in perturbed.themes
        if theme.matched_previous_key
    }
    baseline_keys = {theme.theme_key for theme in baseline.themes}
    identity = (
        len(carried & baseline_keys) / len(baseline_keys) if baseline_keys else 1.0
    )
    matched_of_new = (
        sum(1 for theme in perturbed.themes if theme.matched_previous_key)
        / len(perturbed.themes)
        if perturbed.themes
        else 1.0
    )
    baseline_stories = {
        key for theme in baseline.themes for key in theme.member_story_keys
    }
    after_stories = {
        key for theme in perturbed.themes for key in theme.member_story_keys
    }
    # A story the perturbation removed cannot be retained; only the ones
    # still present are asked about.
    present = baseline_stories & set(perturbed.accounted_story_keys)
    stories_retained = len(present & after_stories) / len(present) if present else 1.0
    return StabilityReport(
        membership_retained=retained,
        identity_retained=identity,
        matched_fraction_of_new=matched_of_new,
        stories_retained_in_themes=stories_retained,
        theme_count_before=len(baseline.themes),
        theme_count_after=len(perturbed.themes),
    )


def evaluate_ticker_day(
    stories: Sequence[ThemeStory],
    *,
    ticker: str,
    trading_day: date,
    volume: str,
    config: ThemeConfig,
    encoder: Any,
) -> TickerDayReport:
    """Cluster one ticker-day and measure everything checkable about it."""

    started = time.perf_counter()
    baseline = cluster_themes(
        stories,
        ticker=ticker,
        trading_day=trading_day,
        config=config,
        encoder=encoder,
    )
    elapsed = time.perf_counter() - started

    # Told what the baseline produced, so the permutation's identity
    # retention is measurable rather than vacuously zero.
    permuted = cluster_themes(
        _permuted(stories),
        ticker=ticker,
        trading_day=trading_day,
        config=config,
        encoder=encoder,
        previous_themes=_previous(baseline),
    )
    # A re-run of the identical day, told what the last run produced: AC-4's
    # "re-running the pipeline within a day does not rename an unchanged
    # theme".
    rerun = cluster_themes(
        stories,
        ticker=ticker,
        trading_day=trading_day,
        config=config,
        encoder=encoder,
        previous_themes=_previous(baseline),
    )
    # A small perturbation: drop the day's least recent story.
    trimmed = sorted(
        stories, key=lambda story: (story.published_at is None, story.published_at)
    )[1:]
    perturbed = cluster_themes(
        trimmed,
        ticker=ticker,
        trading_day=trading_day,
        config=config,
        encoder=encoder,
        previous_themes=_previous(baseline),
    )

    expected = tuple(sorted(story.story_key for story in stories))
    accounted = baseline.accounted_story_keys
    membership: list[str] = [
        key for theme in baseline.themes for key in theme.member_story_keys
    ]
    duplicates = tuple(sorted({key for key in membership if membership.count(key) > 1}))
    details = tuple(
        ThemeDetail(
            rank=theme.salience_rank,
            label=theme.label,
            label_source=theme.label_source,
            fingerprint=theme.fingerprint,
            theme_key=theme.theme_key,
            member_count=theme.story_count,
            member_story_keys=tuple(sorted(theme.member_story_keys)),
            outlet_count=theme.outlet_count,
            outlets=tuple(
                sorted({outlet for entry in theme.evidence for outlet in entry.outlets})
            ),
            representative_story_key=theme.member_story_keys[0],
            salience=round(theme.salience, 6),
            cohesion=round(theme.cohesion, 6),
            min_pairwise_cohesion=round(theme.min_pairwise_cohesion, 6),
            cohesion_margin=round(theme.cohesion - config.min_theme_cohesion, 6),
            near_cohesion_floor=not clears(
                theme.cohesion - config.min_theme_cohesion,
                config.near_cohesion_floor_margin,
            ),
            method=theme.method.value,
            matched_previous_key=theme.matched_previous_key,
        )
        for theme in baseline.themes
    )
    return TickerDayReport(
        ticker=ticker,
        trading_day=trading_day,
        volume=volume,
        story_count=len(stories),
        theme_count=len(baseline.themes),
        other_coverage_count=len(baseline.other_coverage),
        excluded_count=len(baseline.excluded),
        singleton_theme_count=baseline.quality.singleton_theme_count,
        method=baseline.method.value,
        method_reason=baseline.method_reason,
        mean_cohesion=baseline.quality.mean_cohesion,
        min_pairwise_cohesion=baseline.quality.min_pairwise_cohesion,
        max_inter_theme_similarity=baseline.quality.max_inter_theme_similarity,
        theme_coverage=baseline.quality.theme_coverage,
        meets_ac4_shape=baseline.quality.meets_ac4_shape,
        ac4_shape_detail=baseline.quality.ac4_shape_detail,
        no_story_lost=baseline.accounted_story_keys == expected,
        themes=tuple(
            (
                theme.salience_rank,
                theme.label,
                theme.story_count,
                theme.outlet_count,
                round(theme.salience, 4),
            )
            for theme in baseline.themes
        ),
        theme_details=details,
        other_coverage=tuple(entry.story_key for entry in baseline.other_coverage),
        other_coverage_by_reason=baseline.other_coverage_by_reason(),
        excluded_by_reason={
            reason: tuple(
                sorted(
                    entry.story_key
                    for entry in baseline.excluded
                    if entry.reason.value == reason
                )
            )
            for reason in sorted({entry.reason.value for entry in baseline.excluded})
        },
        partition=tuple(
            sorted(tuple(sorted(theme.member_story_keys)) for theme in baseline.themes)
        ),
        missing_story_keys=tuple(sorted(set(expected) - set(accounted))),
        unexpected_story_keys=tuple(sorted(set(accounted) - set(expected))),
        duplicate_membership_keys=duplicates,
        permutation_stable=(
            _membership(permuted) == _membership(baseline)
            and [theme.fingerprint for theme in permuted.themes]
            == [theme.fingerprint for theme in baseline.themes]
        ),
        permutation=_stability(baseline, permuted),
        perturbation=_stability(baseline, perturbed),
        rerun_keeps_identity=all(
            theme.theme_key == original.theme_key
            for theme, original in zip(rerun.themes, baseline.themes)
        )
        and len(rerun.themes) == len(baseline.themes),
        config_fingerprint=baseline.config_fingerprint,
        algorithm_version=baseline.algorithm_version,
        model_name=baseline.model_name,
        model_revision=baseline.model_revision,
        embedding_dimension=baseline.embedding_dimension,
        elapsed_seconds=round(elapsed, 4),
    )

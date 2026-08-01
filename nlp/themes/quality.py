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

from .config import ThemeConfig
from .models import PreviousTheme, ThemeSet, ThemeStory
from .service import cluster_themes


@dataclass(frozen=True)
class StabilityReport:
    """How the day's themes held up under a change that should not matter."""

    #: Themes whose exact membership survived, as a fraction of the run's
    #: themes.  1.0 means the perturbation changed nothing structurally.
    membership_retained: float
    #: Themes that kept their identity through centroid matching.
    identity_retained: float
    theme_count_before: int
    theme_count_after: int


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
    max_inter_theme_similarity: float | None
    theme_coverage: float
    meets_ac4_shape: bool
    no_story_lost: bool
    #: Ranked themes as ``(rank, label, story_count, outlet_count, salience)``.
    themes: tuple[tuple[int, str, int, int, float], ...]
    other_coverage: tuple[str, ...]
    permutation_stable: bool
    perturbation: StabilityReport
    rerun_keeps_identity: bool
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
    matched = sum(1 for theme in perturbed.themes if theme.matched_previous_key)
    identity = matched / len(perturbed.themes) if perturbed.themes else 1.0
    return StabilityReport(
        membership_retained=retained,
        identity_retained=identity,
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

    permuted = cluster_themes(
        _permuted(stories),
        ticker=ticker,
        trading_day=trading_day,
        config=config,
        encoder=encoder,
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
        max_inter_theme_similarity=baseline.quality.max_inter_theme_similarity,
        theme_coverage=baseline.quality.theme_coverage,
        meets_ac4_shape=baseline.quality.meets_ac4_shape,
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
        other_coverage=tuple(entry.story_key for entry in baseline.other_coverage),
        permutation_stable=(
            _membership(permuted) == _membership(baseline)
            and [theme.fingerprint for theme in permuted.themes]
            == [theme.fingerprint for theme in baseline.themes]
        ),
        perturbation=_stability(baseline, perturbed),
        rerun_keeps_identity=all(
            theme.theme_key == original.theme_key
            for theme, original in zip(rerun.themes, baseline.themes)
        )
        and len(rerun.themes) == len(baseline.themes),
        elapsed_seconds=round(elapsed, 4),
    )

"""Salience ranking and theme-identity matching across runs.

Salience is ``f(story count, outlet diversity, recency)`` exactly as issue
#72 specifies, with each component normalized *within the ticker-day* so a
quiet day's top theme is not penalised against a busy one's.

Recency is measured against the day's own most recent story, never against
a wall clock.  Replaying a stored day (AC-8) must rank it the same way next
year as it does today, and a clock comparison would quietly break that.
"""

from __future__ import annotations

from datetime import datetime
import math
from typing import Sequence

from .config import ThemeConfig
from .models import PreviousTheme, SalienceFeatures


def _ratio(value: int, maximum: int) -> float:
    return 0.0 if maximum <= 0 else value / maximum


def recency_component(
    latest: datetime | None, reference: datetime | None, half_life_hours: float
) -> float:
    """Return a decay in ``(0, 1]``, or 0.0 for an undated theme.

    A theme whose newest story *is* the day's newest scores 1.0, and the
    score halves for every ``half_life_hours`` it lags behind.
    """

    if latest is None or reference is None:
        return 0.0
    hours = max(0.0, (reference - latest).total_seconds() / 3600.0)
    return float(0.5 ** (hours / half_life_hours))


def salience_features(
    *,
    story_count: int,
    outlet_count: int,
    latest_published_at: datetime | None,
    max_story_count: int,
    max_outlet_count: int,
    reference_time: datetime | None,
    config: ThemeConfig,
) -> SalienceFeatures:
    """Return one theme's normalized salience components."""

    return SalienceFeatures(
        story_count=story_count,
        outlet_count=outlet_count,
        latest_published_at=latest_published_at,
        story_component=_ratio(story_count, max_story_count),
        outlet_component=_ratio(outlet_count, max_outlet_count),
        recency_component=recency_component(
            latest_published_at, reference_time, config.recency_half_life_hours
        ),
    )


def salience_of(features: SalienceFeatures, config: ThemeConfig) -> float:
    """Return the weighted salience of one theme, in ``[0, 1]``."""

    story_weight, outlet_weight, recency_weight = config.salience_weights
    return (
        story_weight * features.story_component
        + outlet_weight * features.outlet_component
        + recency_weight * features.recency_component
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def match_previous_themes(
    centroids: Sequence[tuple[float, ...]],
    previous: Sequence[PreviousTheme],
    threshold: float,
) -> dict[int, str]:
    """Match this run's themes to the previous run's by centroid similarity.

    One-to-one and greedy from the most similar pair down, so a theme
    cannot claim an identity another theme matches better.  Ties break on
    the previous key then the current index, so the assignment is a
    function of the data and not of iteration order.

    AC-4 requires that re-running within a day does not rename an unchanged
    theme.  Identical input already produces an identical fingerprint; this
    handles the case the requirement is really about — a theme that gained
    or lost a story since the last run and is still the same theme.
    """

    scored: list[tuple[float, str, int]] = []
    for index, centroid in enumerate(centroids):
        for entry in previous:
            similarity = _cosine(centroid, entry.centroid)
            if similarity >= threshold:
                scored.append((similarity, entry.theme_key, index))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched: dict[int, str] = {}
    claimed: set[str] = set()
    for _, theme_key, index in scored:
        if index in matched or theme_key in claimed:
            continue
        matched[index] = theme_key
        claimed.add(theme_key)
    return matched

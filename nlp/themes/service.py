"""The M5 theme clustering entry point.

:func:`cluster_themes` takes one ticker-day's canonical stories and returns
its themes plus everything that is not in one.  It is deterministic: the
same stories, configuration, and encoder produce byte-identical output
regardless of the order they are supplied in, because the first thing it
does is sort them.

It loads no model, reads no clock, and touches no database.  ``trading_day``
is an argument rather than something derived from a timestamp: which trading
day a story belongs to is a calendar question that belongs to the pipeline
(#57/#68), and guessing it here would put a story in the wrong day silently.

**No story disappears.**  Every input comes back in exactly one theme, in
``other_coverage``, or in ``excluded`` with a reason, and the function
asserts that partition before returning.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
from typing import Any, Sequence

import numpy as np

from nlp.embeddings import EmbeddingError, compose_embedding_text

from .clustering import NOISE, assign_clusters, centroid_of, mean_pairwise_similarity
from .coherence import conflicting_members
from .config import ALGORITHM_VERSION, ThemeConfig
from .errors import (
    ThemeCapacityError,
    ThemeEncodingError,
    ThemeInputError,
)
from .models import (
    ClusteringMethod,
    ExcludedStory,
    ExclusionReason,
    PreviousTheme,
    SalienceFeatures,
    Theme,
    ThemeEvidence,
    ThemeQuality,
    ThemeSet,
    ThemeStory,
)
from .salience import match_previous_themes, salience_features, salience_of

_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
#: Namespace for theme fingerprints.
THEME_NAMESPACE = "m5.theme.v1"


def _encode_fields(fields: Sequence[str]) -> bytes:
    encoded = bytearray()
    for value in fields:
        payload = value.encode("utf-8")
        encoded += str(len(payload)).encode("ascii") + b":" + payload
    return bytes(encoded)


def theme_fingerprint_for(
    ticker: str, trading_day: date, member_keys: Sequence[str]
) -> str:
    """Return the content digest of one theme."""

    if not isinstance(THEME_NAMESPACE, str) or not THEME_NAMESPACE.strip():
        raise ThemeInputError("theme fingerprint needs a non-blank namespace")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ThemeInputError("theme fingerprint needs a non-blank ticker")
    if not isinstance(trading_day, date):
        raise ThemeInputError("theme fingerprint needs a trading day")
    keys = list(member_keys)
    if not keys:
        raise ThemeInputError("theme fingerprint needs at least one member")
    unique: set[str] = set()
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ThemeInputError("theme member keys must be non-blank strings")
        if key in unique:
            raise ThemeInputError(f"duplicate theme member key: {key!r}")
        unique.add(key)
    payload = _encode_fields(
        (THEME_NAMESPACE, ticker, trading_day.isoformat(), *sorted(unique))
    )
    return hashlib.sha256(payload).hexdigest()


def _order_key(story: ThemeStory) -> tuple[bool, datetime, str]:
    stamp = story.published_at
    return (stamp is None, stamp or _EPOCH, story.story_key)


def _validate(
    stories: Sequence[ThemeStory],
    ticker: str,
    trading_day: date,
    config: ThemeConfig,
) -> None:
    if not isinstance(ticker, str) or ticker.strip().upper() not in (
        config.ticker_universe
    ):
        raise ThemeInputError(f"ticker is outside the supported universe: {ticker!r}")
    if not isinstance(trading_day, date) or isinstance(trading_day, datetime):
        raise ThemeInputError("trading_day must be a datetime.date, not a datetime")
    seen: set[str] = set()
    for index, story in enumerate(stories):
        if not isinstance(story, ThemeStory):
            raise ThemeInputError("stories must be ThemeStory instances")
        if not isinstance(story.story_key, str) or not story.story_key.strip():
            raise ThemeInputError(f"stories[{index}] has a blank story_key")
        if story.story_key in seen:
            raise ThemeInputError(f"duplicate story_key: {story.story_key}")
        seen.add(story.story_key)
        if story.ticker != ticker:
            raise ThemeInputError(
                f"stories[{index}] is for {story.ticker}, not the requested {ticker}; "
                "themes are built one ticker-day at a time"
            )
        stamp = story.published_at
        if stamp is not None and (
            not isinstance(stamp, datetime)
            or stamp.tzinfo is None
            or stamp.tzinfo.utcoffset(stamp) is None
        ):
            raise ThemeInputError(
                f"stories[{index}] published_at must be a timezone-aware datetime"
            )


def _story_text(story: ThemeStory) -> str | None:
    """Return the exact text M5 embeds, or ``None`` when there is none.

    The representation is the canonical story's headline and standfirst
    joined by M1's composition — the same input M3 compares and the same
    one the embedding cache is keyed on, so a ticker-day costs no extra
    model calls when both stages run.
    """

    try:
        return compose_embedding_text(story.title, story.description)
    except EmbeddingError:
        return None


def _encode(
    stories: Sequence[ThemeStory], encoder: Any
) -> tuple[np.ndarray, list[int]]:
    positions = [index for index, story in enumerate(stories) if _story_text(story)]
    texts = [_story_text(stories[index]) or "" for index in positions]
    if not texts:
        return np.zeros((0, 0)), positions
    vectors = encoder.embed_batch(texts)
    if len(vectors) != len(texts):
        raise ThemeEncodingError(
            f"encoder returned {len(vectors)} vectors for {len(texts)} stories"
        )
    try:
        matrix = np.asarray(vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ThemeEncodingError("encoder returned non-numeric vectors") from exc
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ThemeEncodingError("encoder output shape does not match the input")
    if not np.all(np.isfinite(matrix)):
        raise ThemeEncodingError("encoder returned a non-finite vector")
    if np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ThemeEncodingError(
            "encoder returned a zero vector; cosine similarity is undefined for it"
        )
    return matrix, positions


def _evidence(story: ThemeStory) -> ThemeEvidence:
    return ThemeEvidence(
        story_key=story.story_key,
        title=story.title,
        description=story.description,
        outlets=tuple(sorted({outlet for outlet in story.outlets if outlet})),
        published_at=story.published_at,
        item_ids=tuple(sorted(set(story.item_ids))),
        source_links=tuple(sorted(story.source_links, key=lambda link: link[0])),
    )


def _latest(stories: Sequence[ThemeStory]) -> datetime | None:
    stamps = [story.published_at for story in stories if story.published_at]
    return max(stamps) if stamps else None


def cluster_themes(
    stories: Sequence[ThemeStory],
    *,
    ticker: str,
    trading_day: date,
    config: ThemeConfig,
    encoder: Any,
    previous_themes: Sequence[PreviousTheme] = (),
) -> ThemeSet:
    """Group one ticker-day's canonical stories into salience-ranked themes.

    Raises :class:`~nlp.themes.errors.ThemeCapacityError` before producing
    anything when the day holds more than ``config.max_stories_per_day``
    stories.
    """

    if not isinstance(config, ThemeConfig):
        raise ThemeInputError("config must be a ThemeConfig")
    if isinstance(stories, (str, bytes)) or not isinstance(stories, Sequence):
        raise ThemeInputError("stories must be a sequence of ThemeStory")
    if not hasattr(encoder, "embed_batch"):
        raise ThemeInputError("encoder must implement embed_batch")
    symbol = ticker.strip().upper() if isinstance(ticker, str) else ticker
    _validate(stories, symbol, trading_day, config)
    if len(stories) > config.max_stories_per_day:
        raise ThemeCapacityError(symbol, len(stories), config.max_stories_per_day)

    ordered = sorted(stories, key=_order_key)
    fingerprint = config.fingerprint(
        model_name=str(getattr(encoder, "model_name", "") or ""),
        model_revision=getattr(encoder, "model_revision", None),
    )
    vectors, encodable = _encode(ordered, encoder)
    excluded = tuple(
        ExcludedStory(
            story_key=ordered[index].story_key,
            reason=ExclusionReason.NO_ENCODABLE_TEXT,
        )
        for index in range(len(ordered))
        if index not in set(encodable)
    )
    usable = [ordered[index] for index in encodable]

    if len(usable) < config.min_stories_for_clustering:
        return _assemble(
            symbol,
            trading_day,
            ordered,
            usable,
            vectors,
            groups=[],
            other_positions=list(range(len(usable))),
            excluded=excluded,
            method=ClusteringMethod.SMALL_N_FALLBACK,
            method_reason=(
                f"{len(usable)} stories, below the clustering floor of "
                f"{config.min_stories_for_clustering}; listed individually"
            ),
            config=config,
            fingerprint=fingerprint,
            previous_themes=previous_themes,
            encoder=encoder,
        )

    assignment = assign_clusters(vectors, config)
    grouped: dict[int, list[int]] = {}
    other: list[int] = []
    for position, label in enumerate(assignment.labels):
        if label == NOISE:
            other.append(position)
        else:
            grouped.setdefault(label, []).append(position)

    notes: list[str] = [assignment.reason]
    groups: list[list[int]] = []
    for label in sorted(grouped):
        members = grouped[label]
        if len(members) < config.min_theme_stories:
            # A lone story is coverage, not a theme.  It stays visible under
            # other coverage rather than padding the theme list.
            notes.append(
                f"moved a {len(members)}-story cluster to other coverage, "
                f"below the {config.min_theme_stories}-story theme floor"
            )
            other.extend(members)
            continue
        cohesion = mean_pairwise_similarity(vectors, members)
        if cohesion < config.min_theme_cohesion:
            # A cluster this loose is a catch-all, not a theme.  Its stories
            # go to other coverage rather than being presented as a group a
            # reader could not recognise.
            notes.append(
                f"dissolved a {len(members)}-story cluster with cohesion "
                f"{cohesion:.4f} into other coverage"
            )
            other.extend(members)
            continue
        ejected = conflicting_members(
            usable, sorted(members, key=lambda p: -_position_salience(usable, p))
        )
        if ejected:
            notes.append(
                f"moved {len(ejected)} contradicting story(ies) out of a "
                f"{len(members)}-story theme"
            )
            other.extend(ejected)
            members = [position for position in members if position not in set(ejected)]
        if members:
            groups.append(sorted(members))

    return _assemble(
        symbol,
        trading_day,
        ordered,
        usable,
        vectors,
        groups=groups,
        other_positions=sorted(other),
        excluded=excluded,
        method=(
            ClusteringMethod.HDBSCAN
            if assignment.method == "hdbscan"
            else ClusteringMethod.AGGLOMERATIVE
        ),
        method_reason="; ".join(notes),
        config=config,
        fingerprint=fingerprint,
        previous_themes=previous_themes,
        encoder=encoder,
    )


def _position_salience(usable: Sequence[ThemeStory], position: int) -> float:
    """A cheap within-cluster ordering: more outlets, then more recent."""

    story = usable[position]
    stamp = story.published_at or _EPOCH
    return len(story.outlets) + stamp.timestamp() / 1e12


def _assemble(
    ticker: str,
    trading_day: date,
    ordered: Sequence[ThemeStory],
    usable: Sequence[ThemeStory],
    vectors: np.ndarray,
    *,
    groups: Sequence[Sequence[int]],
    other_positions: Sequence[int],
    excluded: tuple[ExcludedStory, ...],
    method: ClusteringMethod,
    method_reason: str,
    config: ThemeConfig,
    fingerprint: str,
    previous_themes: Sequence[PreviousTheme],
    encoder: Any,
) -> ThemeSet:
    reference_time = _latest(usable)
    counts = [len(group) for group in groups]
    outlet_counts = [
        len({outlet for position in group for outlet in usable[position].outlets})
        for group in groups
    ]
    max_stories = max(counts) if counts else 0
    max_outlets = max(outlet_counts) if outlet_counts else 0

    drafts: list[
        tuple[float, SalienceFeatures, list[int], float, tuple[float, ...]]
    ] = []
    for group, outlet_count in zip(groups, outlet_counts):
        members = list(group)
        features = salience_features(
            story_count=len(members),
            outlet_count=outlet_count,
            latest_published_at=_latest([usable[position] for position in members]),
            max_story_count=max_stories,
            max_outlet_count=max_outlets,
            reference_time=reference_time,
            config=config,
        )
        drafts.append(
            (
                salience_of(features, config),
                features,
                members,
                mean_pairwise_similarity(vectors, members),
                centroid_of(vectors, members),
            )
        )

    # Rank by salience, then by the earliest story, then by fingerprint, so
    # equal-salience themes never swap places between runs.
    drafts.sort(
        key=lambda draft: (
            -draft[0],
            _order_key(usable[min(draft[2], key=lambda p: _order_key(usable[p]))]),
            theme_fingerprint_for(
                ticker, trading_day, [usable[p].story_key for p in draft[2]]
            ),
        )
    )
    matched = match_previous_themes(
        [draft[4] for draft in drafts], previous_themes, config.stability_threshold
    )

    themes: list[Theme] = []
    for rank, (salience, features, members, cohesion, centroid) in enumerate(drafts, 1):
        leading = max(members, key=lambda p: _position_salience(usable, p))
        ordered_members = [leading] + sorted(
            (position for position in members if position != leading),
            key=lambda p: _order_key(usable[p]),
        )
        member_keys = tuple(usable[position].story_key for position in ordered_members)
        digest = theme_fingerprint_for(ticker, trading_day, member_keys)
        previous_key = matched.get(rank - 1)
        themes.append(
            Theme(
                theme_key=previous_key or digest,
                fingerprint=digest,
                ticker=ticker,
                trading_day=trading_day,
                label=usable[leading].title,
                label_source="canonical_story_title",
                member_story_keys=member_keys,
                evidence=tuple(
                    _evidence(usable[position]) for position in ordered_members
                ),
                salience=salience,
                salience_rank=rank,
                salience_features=features,
                cohesion=cohesion,
                centroid=centroid,
                matched_previous_key=previous_key,
                method=method,
            )
        )

    other = tuple(
        _evidence(usable[position])
        for position in sorted(
            set(other_positions), key=lambda p: _order_key(usable[p])
        )
    )
    result = ThemeSet(
        ticker=ticker,
        trading_day=trading_day,
        themes=tuple(themes),
        other_coverage=other,
        excluded=excluded,
        method=method,
        method_reason=method_reason,
        quality=_quality(len(ordered), themes, other, excluded, config),
        config_fingerprint=fingerprint,
        algorithm_version=ALGORITHM_VERSION,
        model_name=str(getattr(encoder, "model_name", "") or ""),
        model_revision=getattr(encoder, "model_revision", None),
    )
    expected = tuple(sorted(story.story_key for story in ordered))
    if result.accounted_story_keys != expected:
        raise AssertionError(
            "theme assembly lost or duplicated a story; "
            f"{len(expected)} in, {len(result.accounted_story_keys)} accounted"
        )
    return result


def _quality(
    story_count: int,
    themes: Sequence[Theme],
    other: Sequence[ThemeEvidence],
    excluded: Sequence[ExcludedStory],
    config: ThemeConfig,
) -> ThemeQuality:
    in_themes = sum(theme.story_count for theme in themes)
    inter = None
    if len(themes) >= 2:
        from .salience import _cosine

        inter = max(
            _cosine(left.centroid, right.centroid)
            for index, left in enumerate(themes)
            for right in themes[index + 1 :]
        )
    small_day = story_count < config.min_stories_for_clustering
    meets = (
        (not themes and len(other) == story_count)
        if small_day
        else config.min_themes <= len(themes) <= config.max_themes
    )
    return ThemeQuality(
        story_count=story_count,
        theme_count=len(themes),
        other_coverage_count=len(other),
        excluded_count=len(excluded),
        singleton_theme_count=sum(1 for theme in themes if theme.story_count == 1),
        mean_cohesion=(
            sum(theme.cohesion for theme in themes) / len(themes) if themes else None
        ),
        max_inter_theme_similarity=inter,
        theme_coverage=(in_themes / story_count) if story_count else 0.0,
        meets_ac4_shape=meets,
    )

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

from .clustering import (
    NOISE,
    assign_clusters,
    centroid_of,
    mean_pairwise_similarity,
    min_pairwise_similarity,
)
from .compatibility import incompatible_members
from .config import ALGORITHM_VERSION, THEME_NAMESPACE, ThemeConfig
from .errors import (
    ThemeCapacityError,
    ThemeEncodingError,
    ThemeInputError,
)
from .models import (
    ClusteringMethod,
    ExcludedStory,
    ExclusionReason,
    OtherCoverageEntry,
    OtherCoverageReason,
    PreviousTheme,
    SalienceFeatures,
    Theme,
    ThemeEvidence,
    ThemeQuality,
    ThemeSet,
    ThemeSourceMetadata,
    ThemeStory,
)
from .salience import match_previous_themes, salience_features, salience_of

_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)


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


def encoder_identity(encoder: Any) -> tuple[str, str | None, int | None]:
    """Return the encoder's validated ``(name, revision, dimension)``.

    A theme set records which model produced it so a stored day can be
    invalidated when the model moves.  A blank name would make that record
    useless and the failure silent, so it is refused here rather than
    written into an artifact as an empty string.
    """

    name = getattr(encoder, "model_name", None)
    if not isinstance(name, str) or not name.strip():
        raise ThemeEncodingError(
            "encoder must expose a non-blank model_name; a theme set that "
            "cannot name its encoder cannot be invalidated when the model moves"
        )
    revision = getattr(encoder, "model_revision", None)
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ThemeEncodingError("encoder model_revision must be a non-blank string")
    dimension = getattr(encoder, "dimension", None)
    if dimension is not None and (
        isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
    ):
        raise ThemeEncodingError("encoder dimension must be a positive integer")
    return name.strip(), revision, dimension


def _encode(
    stories: Sequence[ThemeStory], encoder: Any, declared_dimension: int | None
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
    if declared_dimension is not None and matrix.shape[1] != declared_dimension:
        raise ThemeEncodingError(
            f"encoder returned {matrix.shape[1]}-dimensional vectors but "
            f"declares dimension {declared_dimension}"
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
    source_metadata: ThemeSourceMetadata | None = None,
) -> ThemeSet:
    """Group one ticker-day's canonical stories into salience-ranked themes.

    Raises :class:`~nlp.themes.errors.ThemeCapacityError` before producing
    anything when the day holds more than ``config.max_stories_per_day``
    stories.

    ``source_metadata`` records which dedup run produced the stories; pass
    the bridge's projection so the theme set can be traced back to it.
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
    model_name, model_revision, declared_dimension = encoder_identity(encoder)
    fingerprint = config.fingerprint(
        model_name=model_name,
        model_revision=model_revision,
        embedding_dimension=declared_dimension,
    )
    vectors, encodable = _encode(ordered, encoder, declared_dimension)
    dimension = int(vectors.shape[1]) if vectors.size else declared_dimension
    excluded = tuple(
        ExcludedStory(
            story_key=ordered[index].story_key,
            reason=ExclusionReason.NO_ENCODABLE_TEXT,
        )
        for index in range(len(ordered))
        if index not in set(encodable)
    )
    usable = [ordered[index] for index in encodable]

    # A story M2 quarantined and M3 held out never joins a theme.  Its
    # payload identity is disputed upstream: nothing downstream is entitled
    # to fold it into a narrative or hand it to a summarizer as part of
    # one.  It is still shown, with the reason on it.
    held: dict[int, OtherCoverageReason] = {
        position: (
            OtherCoverageReason.PROVIDER_QUARANTINE
            if story.is_quarantined
            else OtherCoverageReason.SEMANTIC_SKIP
        )
        for position, story in enumerate(usable)
        if story.is_semantically_skipped
    }
    clusterable = [position for position in range(len(usable)) if position not in held]

    if len(clusterable) < config.min_stories_for_clustering:
        reasons = dict(held)
        for position in clusterable:
            reasons[position] = OtherCoverageReason.BELOW_CLUSTERING_FLOOR
        note = (
            f"{len(clusterable)} clusterable stories, below the clustering "
            f"floor of {config.min_stories_for_clustering}; listed individually"
        )
        return _assemble(
            symbol,
            trading_day,
            ordered,
            usable,
            vectors,
            groups=[],
            other_reasons=reasons,
            excluded=excluded,
            method=ClusteringMethod.SMALL_N_FALLBACK,
            method_reason=_with_held(note, held),
            config=config,
            fingerprint=fingerprint,
            previous_themes=previous_themes,
            model_name=model_name,
            model_revision=model_revision,
            dimension=dimension,
            source_metadata=source_metadata,
        )

    subset = vectors[clusterable]
    assignment = assign_clusters(subset, config)
    grouped: dict[int, list[int]] = {}
    reasons = dict(held)
    for index, label in enumerate(assignment.labels):
        position = clusterable[index]
        if label == NOISE:
            reasons[position] = OtherCoverageReason.CLUSTERING_NOISE
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
            for position in members:
                reasons[position] = OtherCoverageReason.BELOW_THEME_SIZE_FLOOR
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
            for position in members:
                reasons[position] = OtherCoverageReason.BELOW_COHESION_FLOOR
            continue
        ejected = incompatible_members(
            usable, sorted(members, key=lambda p: -_position_salience(usable, p))
        )
        if ejected:
            notes.append(
                f"moved {len(ejected)} contradicting story(ies) out of a "
                f"{len(members)}-story theme"
            )
            for position in ejected:
                reasons[position] = OtherCoverageReason.THEME_INCOMPATIBLE
            members = [position for position in members if position not in set(ejected)]
        if len(members) >= config.min_theme_stories:
            groups.append(sorted(members))
        else:
            # Ejecting the contradiction left too few to be a theme.
            for position in members:
                reasons[position] = OtherCoverageReason.BELOW_THEME_SIZE_FLOOR

    return _assemble(
        symbol,
        trading_day,
        ordered,
        usable,
        vectors,
        groups=groups,
        other_reasons=reasons,
        excluded=excluded,
        method=(
            ClusteringMethod.HDBSCAN
            if assignment.method == "hdbscan"
            else ClusteringMethod.AGGLOMERATIVE
        ),
        method_reason=_with_held("; ".join(notes), held),
        config=config,
        fingerprint=fingerprint,
        previous_themes=previous_themes,
        model_name=model_name,
        model_revision=model_revision,
        dimension=dimension,
        source_metadata=source_metadata,
    )


def _with_held(note: str, held: dict[int, OtherCoverageReason]) -> str:
    """Append the upstream hold-outs to the method reason, when there are any."""

    if not held:
        return note
    counts: dict[str, int] = {}
    for reason in held.values():
        counts[reason.value] = counts.get(reason.value, 0) + 1
    detail = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    return f"{note}; held out of clustering by upstream state: {detail}"


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
    other_reasons: dict[int, OtherCoverageReason],
    excluded: tuple[ExcludedStory, ...],
    method: ClusteringMethod,
    method_reason: str,
    config: ThemeConfig,
    fingerprint: str,
    previous_themes: Sequence[PreviousTheme],
    model_name: str,
    model_revision: str | None,
    dimension: int | None,
    source_metadata: ThemeSourceMetadata | None,
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
        tuple[float, SalienceFeatures, list[int], float, tuple[float, ...], float]
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
                min_pairwise_similarity(vectors, members),
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
    for rank, (
        salience,
        features,
        members,
        cohesion,
        centroid,
        min_cohesion,
    ) in enumerate(drafts, 1):
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
                min_pairwise_cohesion=min_cohesion,
                centroid=centroid,
                matched_previous_key=previous_key,
                method=method,
            )
        )

    other = tuple(
        OtherCoverageEntry(
            evidence=_evidence(usable[position]), reason=other_reasons[position]
        )
        for position in sorted(other_reasons, key=lambda p: _order_key(usable[p]))
    )
    result = ThemeSet(
        ticker=ticker,
        trading_day=trading_day,
        themes=tuple(themes),
        other_coverage=other,
        excluded=excluded,
        method=method,
        method_reason=method_reason,
        quality=_quality(len(ordered), themes, other, excluded, method, config),
        config_fingerprint=fingerprint,
        algorithm_version=ALGORITHM_VERSION,
        model_name=model_name,
        model_revision=model_revision,
        embedding_dimension=dimension,
        source_metadata=source_metadata,
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
    other: Sequence[OtherCoverageEntry],
    excluded: Sequence[ExcludedStory],
    method: ClusteringMethod,
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
    # AC-4's band applies to a day that was *clustered*.  Which branch ran
    # is the method, not the raw input count: a five-story day with three
    # encodable stories, or with two quarantined, is legitimately below the
    # floor, and judging it against "2-6 themes" because five arrived
    # reported a correct degradation as a failure.
    if method is ClusteringMethod.SMALL_N_FALLBACK:
        meets = not themes and len(other) + len(excluded) == story_count
    else:
        meets = config.min_themes <= len(themes) <= config.max_themes
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

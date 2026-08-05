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

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from typing import Any, Sequence

import numpy as np

from nlp.dedup.structural import tokenize
from nlp.dedup.text import display_text
from nlp.embeddings import EmbeddingError, compose_embedding_text

from .clustering import (
    BELOW_FLOOR,
    NARRATIVE_MISMATCH,
    NO_STRUCTURE,
    SURPLUS_TO_CAP,
    NOISE,
    assign_clusters,
    centroid_of,
    coherent_subset,
    mean_pairwise_similarity,
    min_pairwise_similarity,
    surviving_themes,
)
from .compatibility import incompatible_members
from .config import ALGORITHM_VERSION, THEME_NAMESPACE, ThemeConfig
from .errors import (
    ThemeCapacityError,
    ThemeEncodingError,
    ThemeInputError,
    ThemePartitionError,
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

#: The clustering layer's dissolution reasons, mapped onto the reasons a
#: reader sees.  One table, so a new reason cannot reach other coverage
#: wearing another one's name.
_DISSOLUTION_REASONS = {
    BELOW_FLOOR: OtherCoverageReason.BELOW_COHESION_FLOOR,
    NARRATIVE_MISMATCH: OtherCoverageReason.NARRATIVE_MISMATCH,
    SURPLUS_TO_CAP: OtherCoverageReason.SURPLUS_TO_THEME_CAP,
}


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
    duplicated: set[str] = set()
    owner: dict[str, str] = {}
    contested: dict[str, set[str]] = {}
    for index, story in enumerate(stories):
        if not isinstance(story, ThemeStory):
            raise ThemeInputError("stories must be ThemeStory instances")
        if not isinstance(story.story_key, str) or not story.story_key.strip():
            raise ThemeInputError(f"stories[{index}] has a blank story_key")
        if story.story_key.strip() != story.story_key:
            raise ThemeInputError(
                f"stories[{index}] story_key is padded: {story.story_key!r}; a "
                "fingerprint is compared verbatim and whitespace makes two "
                "handles for one story"
            )
        if story.story_key in seen:
            duplicated.add(story.story_key)
        seen.add(story.story_key)
        # One raw item, one owning story.  Checked here rather than only in
        # the bridge, because the guarantee is about what is citable and
        # must not depend on which door the caller came through.
        member_seen: set[str] = set()
        for item_id in story.item_ids:
            if not isinstance(item_id, str) or not item_id.strip():
                raise ThemeInputError(
                    f"stories[{index}] ({story.story_key}) has a blank member id"
                )
            if item_id.strip() != item_id:
                raise ThemeInputError(
                    f"stories[{index}] ({story.story_key}) member id is padded: "
                    f"{item_id!r}"
                )
            if item_id in member_seen:
                raise ThemePartitionError(
                    f"story {story.story_key!r} lists member id {item_id!r} twice",
                    overlapping_item_ids=(item_id,),
                    affected_story_keys=(story.story_key,),
                )
            member_seen.add(item_id)
            previous = owner.get(item_id)
            if previous is not None and previous != story.story_key:
                contested.setdefault(item_id, set()).update({previous, story.story_key})
            owner.setdefault(item_id, story.story_key)
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
        count = story.outlet_count
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ThemeInputError(
                    f"stories[{index}] ({story.story_key}) outlet_count must be a "
                    "positive integer"
                )
            distinct = len({outlet for outlet in story.outlets if outlet})
            if count < distinct:
                raise ThemeInputError(
                    f"stories[{index}] ({story.story_key}) declares outlet_count "
                    f"{count} but lists {distinct} distinct outlets; the "
                    "authoritative count may exceed the projected set, never "
                    "fall below it"
                )
    if duplicated:
        keys = tuple(sorted(duplicated))
        raise ThemePartitionError(
            f"story key(s) appear more than once: {list(keys)}",
            overlapping_story_keys=keys,
            affected_story_keys=keys,
        )
    if contested:
        items = tuple(sorted(contested))
        affected = tuple(sorted({key for keys in contested.values() for key in keys}))
        raise ThemePartitionError(
            f"raw item(s) {list(items)} are claimed by more than one story "
            f"({list(affected)}); one raw item must be citable from exactly "
            "one theme",
            overlapping_item_ids=items,
            affected_story_keys=affected,
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
    clusterable_stories = [usable[position] for position in clusterable]
    assignment = assign_clusters(subset, config, clusterable_stories)
    reasons = dict(held)
    labels: list[int] = []
    for index, label in enumerate(assignment.labels):
        position = clusterable[index]
        if label == NOISE:
            reasons[position] = OtherCoverageReason.CLUSTERING_NOISE
        labels.append(label)

    notes: list[str] = [assignment.reason]
    groups: list[list[int]] = []
    structureless = assignment.method == NO_STRUCTURE
    if structureless:
        degenerate = "degenerate-geometry" in assignment.reason
        for position in clusterable:
            reasons[position] = (
                OtherCoverageReason.DEGENERATE_EMBEDDING_GEOMETRY
                if degenerate
                else OtherCoverageReason.INSUFFICIENT_THEME_STRUCTURE
            )
    else:
        # The same function the fallback scored candidates with, so the day
        # cannot be assembled into a shape the objective never evaluated.
        # Positions here index ``clusterable``; map them back once.
        kept, dissolved = surviving_themes(subset, labels, config, clusterable_stories)
        for local, why in dissolved.items():
            position = clusterable[local]
            if position not in reasons:
                reasons[position] = _DISSOLUTION_REASONS[why]
        if dissolved:
            counts: dict[str, int] = {}
            for why in dissolved.values():
                counts[why] = counts.get(why, 0) + 1
            detail = ", ".join(
                f"{count} {why}" for why, count in sorted(counts.items())
            )
            notes.append(
                f"{len(dissolved)} story(ies) moved to other coverage: {detail}"
            )
        for members_local in kept:
            members = [clusterable[local] for local in members_local]
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
                members = [
                    position for position in members if position not in set(ejected)
                ]
            # Ejecting a contradiction can break the floors the subset cleared.
            survivors = (
                [
                    members[local]
                    for local in coherent_subset(
                        vectors[members], range(len(members)), config
                    )
                ]
                if members
                else []
            )
            for position in members:
                if position not in survivors and position not in reasons:
                    reasons[position] = OtherCoverageReason.BELOW_COHESION_FLOOR
            if survivors:
                groups.append(sorted(survivors))

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
            ClusteringMethod.NO_SEPARABLE_STRUCTURE
            if structureless
            else ClusteringMethod.HDBSCAN
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


def outlet_count_of(story: ThemeStory) -> int:
    """Return the authoritative distinct-outlet count for one story.

    M3's ``outlet_count`` wins when it is present: it counts the outlets the
    dedup stage actually saw, which can exceed the outlets M5 was handed,
    and substituting ``len(story.outlets)`` would silently under-rank a
    widely syndicated story against a narrowly carried one.  The projected
    set is the fallback, never a replacement.
    """

    return story.authoritative_outlet_count


@dataclass(frozen=True)
class OutletCoverage:
    """What is actually known about how widely a theme was carried.

    Three numbers instead of one, because one number had to lie.  Summing
    each story's unnamed excess treated every unknown carrier as distinct -
    two stories each syndicated to eight unnamed outlets became sixteen,
    when they may well have been the same eight - and that inflated
    salience for whichever theme happened to hold the most unnamed
    coverage.
    """

    #: Outlets named explicitly by at least one member.  Exact.
    named_outlet_count: int
    #: The largest authoritative per-story count in the theme.  A lower
    #: bound on the theme's true distinct outlets, and the number used for
    #: ranking: it cannot double-count carriers that may be shared.
    bounded_outlet_count: int
    #: Whether any member's authoritative count exceeded the names it
    #: carried, so the true distinct total is unknown and at least this.
    has_unresolved_outlet_count: bool

    @property
    def ranking_count(self) -> int:
        """The number salience uses: never an estimate, never a sum."""

        return max(self.named_outlet_count, self.bounded_outlet_count)


def outlet_coverage(stories: Sequence[ThemeStory]) -> OutletCoverage:
    """Summarize a group's outlet coverage without inventing distinctness.

    Named outlets are unioned, which is exact.  Unnamed excess is *not*
    summed across stories: nothing in the projection says two stories'
    unnamed carriers are different outlets, and assuming they are inflates
    every theme that holds syndicated coverage.  What can be asserted is a
    lower bound - the theme has at least as many distinct outlets as its
    most widely carried member - and that is what ranking uses.
    """

    named = {outlet for story in stories for outlet in story.outlets if outlet}
    counts = [outlet_count_of(story) for story in stories]
    unresolved = any(
        outlet_count_of(story) > len({o for o in story.outlets if o})
        for story in stories
    )
    return OutletCoverage(
        named_outlet_count=len(named),
        bounded_outlet_count=max(counts) if counts else 0,
        has_unresolved_outlet_count=unresolved,
    )


def _position_salience(usable: Sequence[ThemeStory], position: int) -> float:
    """A cheap within-cluster ordering: more outlets, then more recent."""

    story = usable[position]
    stamp = story.published_at or _EPOCH
    return outlet_count_of(story) + stamp.timestamp() / 1e12


#: Words that carry no discriminating content in a market headline, so a
#: title made mostly of them is not a label a reader learns anything from.
_LOW_CONTENT_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "s",
        "the",
        "to",
        "update",
        "updates",
        "news",
        "report",
        "reports",
        "results",
        "story",
        "stories",
    }
)


def title_informativeness(title: str) -> tuple[int, int, int]:
    """Score a headline's usefulness as a label, extractively.

    Returns ``(distinct content tokens, entity-or-numeral tokens, negative
    repetition penalty)``.  Nothing is synthesized and nothing is dropped:
    the label is still the member's own title, this only decides *which*
    member's title represents the theme.  "results results results results"
    scores one content token and a repetition penalty of -3; an informative
    headline naming a company and a figure outscores it regardless of how
    recent or widely carried the generic one is.
    """

    tokens = tokenize(display_text(title))
    content = [token for token in tokens if token not in _LOW_CONTENT_TOKENS]
    distinct = len(set(content))
    specific = sum(
        1
        for token in set(content)
        if any(character.isdigit() for character in token)
        or (token and token not in _LOW_CONTENT_TOKENS and len(token) > 3)
    )
    repetition = distinct - len(content)
    return distinct, specific, repetition


def _representative_of(usable: Sequence[ThemeStory], members: Sequence[int]) -> int:
    """The member whose title labels the theme.

    Chosen from the theme's **final** membership, so a story ejected by the
    cohesion or narrative gate can never label the theme it left.  Among
    members carrying the theme's dominant narrative family, the most
    informative title wins; a theme whose family is unknown falls back to
    informativeness over all its members.  A label that named a family the
    rest of the theme is not about would imply a narrower story than the
    evidence supports.
    """

    from .narrative import dominant_family, narrative_families

    family = dominant_family(usable, members)
    candidates = [
        position
        for position in members
        if family is not None and family in narrative_families(usable[position])
    ]
    return max(candidates or list(members), key=lambda p: _label_rank(usable, p))


def _label_rank(usable: Sequence[ThemeStory], position: int) -> tuple:
    """Ordering for the member whose title labels the theme.

    Informativeness first, then the upstream outlet count, then recency,
    then the story key.  Ranking on outlets and recency alone let a generic
    headline label a theme whose other members said something.
    """

    story = usable[position]
    distinct, specific, repetition = title_informativeness(story.title)
    stamp = story.published_at or _EPOCH
    return (
        distinct + specific + repetition,
        specific,
        outlet_count_of(story),
        stamp,
        story.story_key,
    )


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
    coverages = [
        outlet_coverage([usable[position] for position in group]) for group in groups
    ]
    outlet_counts = [coverage.ranking_count for coverage in coverages]
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
        leading = _representative_of(usable, members)
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
    expected = tuple(sorted(story.story_key for story in ordered))
    membership = [key for theme in themes for key in theme.member_story_keys]
    accounted = sorted(
        membership
        + [entry.story_key for entry in other]
        + [entry.story_key for entry in excluded]
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
        input_story_keys=expected,
        missing_story_keys=tuple(sorted(set(expected) - set(accounted))),
        unexpected_story_keys=tuple(sorted(set(accounted) - set(expected))),
        duplicate_membership_keys=tuple(
            sorted({key for key in membership if membership.count(key) > 1})
        ),
    )
    if result.accounted_story_keys != expected or not result.complete:
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
        detail = (
            "below the clustering floor; AC-4 asks for individual listing and "
            "that is what happened"
        )
    elif method is ClusteringMethod.NO_SEPARABLE_STRUCTURE:
        # Honestly false.  The day had enough stories for AC-4's band and
        # produced no theme, and saying so is worth more than a split drawn
        # to satisfy a count - AC-4's other half is that no story is
        # dropped, and every one of them is listed.
        meets = False
        detail = (
            "enough stories for AC-4's band but no partition cleared the "
            "quality floors; no theme was invented and every story is listed"
        )
    else:
        meets = config.min_themes <= len(themes) <= config.max_themes
        detail = (
            f"{len(themes)} theme(s) against the {config.min_themes}-"
            f"{config.max_themes} band"
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
        ac4_shape_detail=detail,
        min_pairwise_cohesion=(
            min(theme.min_pairwise_cohesion for theme in themes) if themes else None
        ),
    )

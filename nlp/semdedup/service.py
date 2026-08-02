"""The M3 semantic dedup entry point.

:func:`merge_semantic_duplicates` is deterministic: the same stories under
the same configuration and encoder produce byte-identical output regardless
of the order they are supplied in.  It touches no database, no clock, no
network, and no filesystem; the only outside dependency is the injected
encoder, which is why the encoder identity is part of the fingerprint.

Cluster construction is deliberately conservative:

* candidates are exhaustive inside one ticker and one window, never across;
* every candidate passes the *whole prospective story's* evidence summary,
  not just the two endpoints, so a vague story cannot bridge two that
  contradict each other;
* a story is a **clique**: every pair inside it independently cleared the
  threshold, the guards, and the window.  Transitive chaining therefore
  cannot bridge incompatible endpoints even in principle.

Precision is favoured everywhere the two conflict.  A missed rewrite costs
a duplicate card; a false merge attributes one company's story to another
event and is the failure the whole trust-first design exists to prevent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
from typing import Any, Sequence

from nlp.embeddings import EmbeddingInputError, cosine_similarity

from .config import ALGORITHM_VERSION, SemanticDedupConfig
from .encoding import (
    StoryEncoder,
    encode_stories,
    validate_dimension,
    validate_model_metadata,
)
from .errors import (
    SemanticDedupCapacityError,
    SemanticDedupEncodingError,
    SemanticDedupInputError,
)
from .evidence import StoryEvidence, combine, summarize
from .models import (
    RejectedPair,
    SemanticSkipReason,
    SemanticDedupResult,
    SemanticDedupStats,
    SemanticMerge,
    SemanticStory,
    SourceLink,
    StoryInput,
)

_EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
#: Namespace for semantic story fingerprints.
STORY_NAMESPACE = "m3.story.v1"


def _encode_fields(fields: Sequence[str]) -> bytes:
    """Length-prefix every field so no value can forge a boundary."""

    encoded = bytearray()
    for value in fields:
        payload = value.encode("utf-8")
        encoded += str(len(payload)).encode("ascii") + b":" + payload
    return bytes(encoded)


def story_fingerprint_for(ticker: str, member_keys: Sequence[str]) -> str:
    """Return the change-detection fingerprint of one semantic story."""

    if not isinstance(STORY_NAMESPACE, str) or not STORY_NAMESPACE.strip():
        raise SemanticDedupInputError("story fingerprint needs a non-blank namespace")
    if not isinstance(ticker, str) or not ticker.strip():
        raise SemanticDedupInputError("story fingerprint needs a non-blank ticker")
    keys = list(member_keys)
    if not keys:
        raise SemanticDedupInputError("story fingerprint needs at least one member")
    unique: set[str] = set()
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise SemanticDedupInputError("story member keys must be non-blank strings")
        if key in unique:
            raise SemanticDedupInputError(f"duplicate story member key: {key!r}")
        unique.add(key)
    payload = _encode_fields((STORY_NAMESPACE, ticker, *sorted(unique)))
    return hashlib.sha256(payload).hexdigest()


def _validate(stories: Sequence[StoryInput], config: SemanticDedupConfig) -> None:
    seen: set[str] = set()
    for index, story in enumerate(stories):
        if not isinstance(story, StoryInput):
            raise SemanticDedupInputError("stories must be StoryInput instances")
        if not isinstance(story.story_key, str) or not story.story_key.strip():
            raise SemanticDedupInputError(f"stories[{index}] has a blank story_key")
        if story.story_key in seen:
            raise SemanticDedupInputError(f"duplicate story_key: {story.story_key}")
        seen.add(story.story_key)
        if story.ticker not in config.ticker_universe:
            raise SemanticDedupInputError(
                f"stories[{index}] ticker is outside the supported universe: "
                f"{story.ticker}"
            )
        stamp = story.published_at
        if stamp is not None:
            if not isinstance(stamp, datetime):
                raise SemanticDedupInputError(
                    f"stories[{index}] published_at must be a datetime or None"
                )
            if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
                raise SemanticDedupInputError(
                    f"stories[{index}] published_at must carry a timezone offset"
                )
        if len(set(story.member_ids)) != len(story.member_ids):
            raise SemanticDedupInputError(f"stories[{index}] repeats a member_id")
        unknown = set(story.quarantined_member_ids) - set(story.member_ids)
        if unknown:
            raise SemanticDedupInputError(
                f"stories[{index}] quarantines {sorted(unknown)}, which are not "
                "its own members"
            )
    # A raw item belongs to exactly one canonical story. Two input stories
    # sharing a member mean the caller's upstream partition is broken, and
    # merging or silently duplicating them would carry that corruption into
    # the output instead of surfacing it.
    owner: dict[str, str] = {}
    overlaps: set[str] = set()
    for story in stories:
        for item_id in story.member_ids:
            if item_id in owner and owner[item_id] != story.story_key:
                overlaps.add(f"{item_id} in {owner[item_id]!r} and {story.story_key!r}")
            else:
                owner.setdefault(item_id, story.story_key)
    if overlaps:
        raise SemanticDedupInputError(
            "input stories overlap; a member id belongs to exactly one "
            "canonical story: " + "; ".join(sorted(overlaps))
        )


def _order_key(story: StoryInput) -> tuple[bool, datetime, str]:
    """Total, permutation-invariant ordering: earliest first, key breaks ties."""

    stamp = story.published_at
    return (stamp is None, stamp or _EPOCH, story.story_key)


def _similarity(left: Any, right: Any) -> float:
    try:
        return cosine_similarity(left, right)
    except EmbeddingInputError as exc:
        raise SemanticDedupEncodingError(
            f"cosine similarity is undefined for these vectors: {exc}"
        ) from exc


@dataclass
class _Component:
    """One prospective story under construction."""

    members: list[int]
    evidence: StoryEvidence


def _candidate_pairs(
    partition: Sequence[StoryInput], config: SemanticDedupConfig
) -> list[tuple[int, int]]:
    """Return every comparable pair of a time-ordered partition.

    Positions index ``partition``, which is sorted by publication time, so
    every pair is ``(earlier, later)`` and every downstream ordering is a
    function of the data rather than of the caller's list order.

    Exhaustive within the window rather than approximate: Phase 0 slices are
    small, and an approximate generator would make a missing merge
    unattributable — you could not tell a refused pair from a pair the
    generator never proposed.

    A story M2 quarantined under a provider-identity conflict is not
    eligible at all.  M2 found one feed identifier describing two different
    articles and could not tell which payload was right; cosine similarity
    is not evidence about which one is, so an authoritative conflict is
    never overruled by a score.  The story is retained unchanged in the
    output with its skip reason recorded.
    """

    eligible = [
        position
        for position, story in enumerate(partition)
        if not story.is_quarantined
        and (story.published_at is not None or config.allow_undated_merges)
    ]
    window = config.window
    pairs: list[tuple[int, int]] = []
    for left, right in itertools.combinations(eligible, 2):
        left_time = partition[left].published_at
        right_time = partition[right].published_at
        if left_time is not None and right_time is not None:
            if abs(right_time - left_time) > window:
                continue
        elif not config.allow_undated_merges:
            continue
        pairs.append((left, right))
    return pairs


def _merge_partition(
    partition: Sequence[StoryInput],
    vectors: Sequence[Any | None],
    config: SemanticDedupConfig,
) -> tuple[list[list[int]], list[SemanticMerge], list[RejectedPair], Counter, int, int]:
    """Cluster one time-ordered ticker partition into cliques.

    Everything below works in *position* space over the time-ordered
    partition, never in the caller's input order, which is what makes the
    output invariant under permutation of the input.
    """

    evidence = {
        position: summarize(story.title, story.description)
        for position, story in enumerate(partition)
    }
    components: dict[int, _Component] = {
        position: _Component(members=[position], evidence=summary)
        for position, summary in evidence.items()
    }
    owner = {position: position for position in evidence}
    accepted: dict[tuple[int, int], float] = {}
    merges: list[SemanticMerge] = []
    rejected: list[RejectedPair] = []
    vetoes: Counter = Counter()
    above_threshold = 0

    candidates = _candidate_pairs(partition, config)
    for left, right in candidates:
        left_vector, right_vector = vectors[left], vectors[right]
        if left_vector is None or right_vector is None:
            rejected.append(
                _rejection(partition, left, right, None, "no_encodable_text")
            )
            vetoes["no_encodable_text"] += 1
            continue
        score = _similarity(left_vector, right_vector)
        if score < config.similarity_threshold:
            rejected.append(
                _rejection(partition, left, right, score, "below_threshold")
            )
            vetoes["below_threshold"] += 1
            continue
        above_threshold += 1
        # The guards run pairwise first so the recorded reason names the
        # actual objection between these two, then again across the whole
        # prospective story below.
        veto = combine(
            evidence[left],
            evidence[right],
            frame_overlap=config.frame_overlap_threshold,
        )[1]
        if veto is not None:
            rejected.append(_rejection(partition, left, right, score, veto))
            vetoes[veto] += 1
            continue
        accepted[(left, right)] = score

    # Apply accepted edges strongest first, so the most confident merge
    # decides a story's shape before a marginal one can.  Ties break on the
    # time-ordered positions, so the order is a function of the data.
    for (left, right), score in sorted(
        accepted.items(), key=lambda entry: (-entry[1], entry[0])
    ):
        left_root, right_root = owner[left], owner[right]
        if left_root == right_root:
            continue
        merged_evidence, veto = combine(
            components[left_root].evidence,
            components[right_root].evidence,
            frame_overlap=config.frame_overlap_threshold,
        )
        if veto is not None:
            rejected.append(
                _rejection(partition, left, right, score, f"cluster_{veto}")
            )
            vetoes[f"cluster_{veto}"] += 1
            continue
        if not _is_clique(
            components[left_root].members, components[right_root].members, accepted
        ):
            rejected.append(
                _rejection(partition, left, right, score, "cluster_not_complete")
            )
            vetoes["cluster_not_complete"] += 1
            continue
        assert merged_evidence is not None
        survivor, absorbed = sorted((left_root, right_root))
        components[survivor].members.extend(components[absorbed].members)
        components[survivor].evidence = merged_evidence
        for position in components[absorbed].members:
            owner[position] = survivor
        del components[absorbed]
        merges.append(
            SemanticMerge(
                left_story_key=partition[left].story_key,
                right_story_key=partition[right].story_key,
                similarity=score,
            )
        )

    groups = [sorted(component.members) for _, component in sorted(components.items())]
    return groups, merges, rejected, vetoes, len(candidates), above_threshold


def _is_clique(
    left_members: Sequence[int],
    right_members: Sequence[int],
    accepted: dict[tuple[int, int], float],
) -> bool:
    """True when every cross pair between two components was accepted.

    This is what stops transitive drift: a story is a set in which *every*
    member independently cleared the threshold and every guard against
    *every* other member.  Single-link chaining would let A-B and B-C place
    A and C together without anything ever comparing them.
    """

    for left in left_members:
        for right in right_members:
            key = (left, right) if left < right else (right, left)
            if key not in accepted:
                return False
    return True


def _rejection(
    partition: Sequence[StoryInput],
    left: int,
    right: int,
    similarity: float | None,
    reason: str,
) -> RejectedPair:
    return RejectedPair(
        left_story_key=partition[left].story_key,
        right_story_key=partition[right].story_key,
        similarity=similarity,
        reason=reason,
    )


def _canonical_sort_key(story: StoryInput) -> tuple[bool, datetime, str, str]:
    """Earliest publication wins; outlet then story key break exact ties."""

    stamp = story.published_at
    return (
        stamp is None,
        stamp or _EPOCH,
        story.outlets[0] if story.outlets else "",
        story.story_key,
    )


def _build_story(
    ticker: str,
    members: Sequence[StoryInput],
    merges: Sequence[SemanticMerge],
    config_fingerprint: str,
) -> SemanticStory:
    canonical = min(members, key=_canonical_sort_key)
    ordered = [canonical] + sorted(
        (story for story in members if story.story_key != canonical.story_key),
        key=_canonical_sort_key,
    )
    member_keys = tuple(story.story_key for story in ordered)
    member_ids = tuple(
        sorted({item_id for story in members for item_id in story.member_ids})
    )
    # Every outlet the story declares *and* every outlet a retained link
    # names.  Taking the union means ``outlet_count`` can never undercount
    # the links AC-3 requires to be kept.
    outlets = tuple(
        sorted(
            {outlet for story in members for outlet in story.outlets if outlet}
            | {
                link.outlet
                for story in members
                for link in story.source_links
                if link.outlet
            }
        )
    )
    links = sorted(
        {
            (link.item_id, link.outlet, link.url)
            for story in members
            for link in story.source_links
        },
        key=lambda link: (link[0], link[1], link[2] or ""),
    )
    quarantined = tuple(
        sorted({item for story in members for item in story.quarantined_member_ids})
    )
    conflicts = tuple(
        sorted({entry for story in members for entry in story.provider_conflicts})
    )
    skip = (
        SemanticSkipReason.PROVIDER_QUARANTINE
        if any(story.is_quarantined for story in members)
        else None
    )
    fingerprint = story_fingerprint_for(ticker, member_keys)
    relevant = tuple(
        merge
        for merge in merges
        if merge.left_story_key in set(member_keys)
        and merge.right_story_key in set(member_keys)
    )
    return SemanticStory(
        story_fingerprint=fingerprint,
        ticker=ticker,
        canonical_story_key=canonical.story_key,
        canonical_title=canonical.title,
        published_at=canonical.published_at,
        member_story_keys=member_keys,
        member_ids=member_ids,
        outlets=outlets,
        outlet_count=len(outlets),
        source_links=tuple(
            SourceLink(item_id=item_id, outlet=outlet, url=url)
            for item_id, outlet, url in links
        ),
        merges=relevant,
        quarantined_member_ids=quarantined,
        provider_conflicts=conflicts,
        semantic_skip_reason=skip,
        content_hash=hashlib.sha256(
            _encode_fields(
                [
                    ALGORITHM_VERSION,
                    config_fingerprint,
                    fingerprint,
                    ticker,
                    canonical.story_key,
                    canonical.title,
                    canonical.published_at.isoformat()
                    if canonical.published_at
                    else "",
                    str(len(outlets)),
                    skip.value if skip is not None else "",
                    *member_keys,
                    *member_ids,
                    *quarantined,
                ]
            )
        ).hexdigest(),
        algorithm_version=ALGORITHM_VERSION,
    )


def merge_semantic_duplicates(
    stories: Sequence[StoryInput],
    *,
    config: SemanticDedupConfig,
    encoder: StoryEncoder,
) -> SemanticDedupResult:
    """Merge canonical stories that describe one event in different words.

    Raises :class:`~nlp.semdedup.errors.SemanticDedupCapacityError` — before
    producing any output — when one ticker holds more than
    ``config.max_partition_stories`` stories.
    """

    if not isinstance(config, SemanticDedupConfig):
        raise SemanticDedupInputError("config must be a SemanticDedupConfig")
    if isinstance(stories, (str, bytes)) or not isinstance(stories, Sequence):
        raise SemanticDedupInputError("stories must be a sequence of StoryInput")
    if not isinstance(encoder, StoryEncoder):
        raise SemanticDedupInputError(
            "encoder must implement embed_batch, model_name, and model_revision"
        )
    snapshot = tuple(stories)
    _validate(snapshot, config)

    partitions: dict[str, list[int]] = {}
    for index, story in enumerate(snapshot):
        partitions.setdefault(story.ticker, []).append(index)
    for ticker in sorted(partitions):
        if len(partitions[ticker]) > config.max_partition_stories:
            raise SemanticDedupCapacityError(
                ticker, len(partitions[ticker]), config.max_partition_stories
            )

    model_name, model_revision = validate_model_metadata(encoder)
    declared_dimension = validate_dimension(getattr(encoder, "dimension", None))
    vectors, unencodable = encode_stories(snapshot, encoder)
    observed = next(
        (len(list(vector)) for vector in vectors if vector is not None), None
    )
    dimension = declared_dimension if declared_dimension is not None else observed
    fingerprint = config.fingerprint(
        model_name=model_name,
        model_revision=model_revision,
        embedding_dimension=dimension,
    )

    built: list[SemanticStory] = []
    rejected: list[RejectedPair] = []
    vetoes: Counter = Counter()
    candidate_count = 0
    above_threshold = 0
    accepted_edges = 0
    for ticker in sorted(partitions):
        ordered = sorted(
            partitions[ticker], key=lambda index: _order_key(snapshot[index])
        )
        partition = [snapshot[index] for index in ordered]
        (
            groups,
            merges,
            partition_rejections,
            partition_vetoes,
            partition_candidates,
            partition_above,
        ) = _merge_partition(partition, [vectors[index] for index in ordered], config)
        candidate_count += partition_candidates
        above_threshold += partition_above
        accepted_edges += len(merges)
        rejected.extend(partition_rejections)
        vetoes.update(partition_vetoes)
        for group in groups:
            built.append(
                _build_story(
                    ticker,
                    [partition[position] for position in group],
                    merges,
                    fingerprint,
                )
            )

    built.sort(
        key=lambda story: (
            story.ticker,
            story.published_at is None,
            story.published_at or _EPOCH,
            story.story_fingerprint,
        )
    )
    merged = [story for story in built if story.is_merged]
    return SemanticDedupResult(
        stories=tuple(built),
        stats=SemanticDedupStats(
            input_story_count=len(snapshot),
            story_count=len(built),
            merged_story_count=len(merged),
            collapsed_story_count=sum(story.member_count - 1 for story in merged),
            candidate_pair_count=candidate_count,
            above_threshold_count=above_threshold,
            accepted_pair_count=accepted_edges,
            veto_counts=tuple(sorted(vetoes.items())),
            unencodable_story_count=unencodable,
            skipped_story_counts=tuple(
                sorted(
                    {
                        SemanticSkipReason.PROVIDER_QUARANTINE.value: sum(
                            1 for story in built if story.is_quarantined
                        )
                    }.items()
                )
            ),
        ),
        rejected_pairs=tuple(
            sorted(
                rejected,
                key=lambda entry: (entry.left_story_key, entry.right_story_key),
            )
        ),
        config_fingerprint=fingerprint,
        algorithm_version=ALGORITHM_VERSION,
        model_name=model_name,
        model_revision=model_revision,
        embedding_dimension=dimension,
    )

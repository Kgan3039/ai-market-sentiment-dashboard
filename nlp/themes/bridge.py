"""The one place M5 knows what an M3 story looks like.

M5 clusters *canonical* stories — one per event, after M2 collapsed exact
copies and M3 collapsed rewrites. Feeding it raw duplicates would let the
size of a syndication burst decide the shape of the day's themes.

The bridge also accepts M2 output directly, for the case the spec's Section
9 anticipates: "if semantic dedup tuning slips, M5 proceeds on stage-1/2
dedup output". That path is a documented degradation, not a default.

**Only M3's public result is read.**  Every field below comes off
:class:`~nlp.semdedup.SemanticDedupResult` or
:class:`~nlp.semdedup.SemanticStory`; nothing reaches into M3's evidence,
guard, or normalization helpers, and M5 never re-derives a value M3 already
published.

**Nothing trust-bearing is dropped.**  A quarantine M2 raised and M3
honoured has to survive one more stage to reach a reader, and a bridge that
projected only the title and the timestamps would silently launder it.  The
projection therefore carries quarantine state, provider conflicts, the
semantic skip reason, the accepted merge evidence, M3's outlet count, and
the model and configuration identity of the run that produced it.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from nlp.dedup import DedupResult, RawItem
from nlp.semdedup import SemanticDedupResult
from nlp.semdedup.bridge import stories_from_dedup

from .errors import ThemeInputError
from .models import ThemeSourceMetadata, ThemeStory


def _links(source_links) -> tuple[tuple[str, str, str | None], ...]:
    return tuple(sorted((link.item_id, link.outlet, link.url) for link in source_links))


def _merge_evidence(merges) -> tuple[tuple[str, str, float, str], ...]:
    """Project M3's accepted merges onto plain, sorted tuples."""

    return tuple(
        sorted(
            (
                merge.left_story_key,
                merge.right_story_key,
                float(merge.similarity),
                getattr(merge.reason, "value", str(merge.reason)),
            )
            for merge in merges
        )
    )


def _reject_overlapping_items(stories: Sequence[ThemeStory]) -> None:
    """Refuse input where one raw item belongs to two canonical stories.

    Two stories claiming the same item means the dedup partition upstream
    is broken.  Clustering it anyway would let one article be cited from
    two themes, which is precisely what "no story appears twice" exists to
    prevent — and by then the damage is invisible.
    """

    owner: dict[str, str] = {}
    for story in stories:
        for item_id in story.item_ids:
            previous = owner.get(item_id)
            if previous is not None and previous != story.story_key:
                raise ThemeInputError(
                    f"item {item_id!r} belongs to both story {previous!r} and "
                    f"story {story.story_key!r}; the upstream partition overlaps"
                )
            owner[item_id] = story.story_key


def theme_stories_from_semantic(
    result: SemanticDedupResult, descriptions: Mapping[str, str | None] | None = None
) -> tuple[ThemeStory, ...]:
    """Project an M3 run onto M5's input, in M3's story order.

    ``descriptions`` maps a *story key* to the standfirst M5 should embed.
    M3's output carries the canonical title but not the description it
    compared, so the caller — which has the raw items — supplies it. Without
    it M5 embeds titles alone, which is a weaker representation and is
    reported rather than silently assumed.
    """

    lookup = descriptions or {}
    stories = tuple(
        ThemeStory(
            story_key=story.story_fingerprint,
            ticker=story.ticker,
            title=story.canonical_title,
            description=lookup.get(story.story_fingerprint),
            published_at=story.published_at,
            outlets=tuple(story.outlets),
            item_ids=tuple(story.member_ids),
            source_links=_links(story.source_links),
            outlet_count=story.outlet_count,
            member_story_keys=tuple(story.member_story_keys),
            quarantined_member_ids=tuple(story.quarantined_member_ids),
            provider_conflicts=tuple(story.provider_conflicts),
            semantic_skip_reason=(
                story.semantic_skip_reason.value
                if story.semantic_skip_reason is not None
                else None
            ),
            merge_evidence=_merge_evidence(story.merges),
            content_hash=story.content_hash,
        )
        for story in result.stories
    )
    _reject_overlapping_items(stories)
    return stories


def source_metadata_from_semantic(result: SemanticDedupResult) -> ThemeSourceMetadata:
    """Record which M3 run produced the stories M5 is about to cluster."""

    return ThemeSourceMetadata(
        stage="m3.semantic",
        algorithm_version=result.algorithm_version,
        config_fingerprint=result.config_fingerprint,
        model_name=result.model_name,
        model_revision=result.model_revision,
        embedding_dimension=result.embedding_dimension,
        story_count=len(result.stories),
        quarantined_story_count=sum(
            1 for story in result.stories if story.is_quarantined
        ),
        semantically_skipped_story_count=sum(
            1 for story in result.stories if story.semantic_skip_reason is not None
        ),
        merged_story_count=sum(1 for story in result.stories if story.is_merged),
    )


def theme_stories_from_exact(
    result: DedupResult, raw_items: Sequence[RawItem]
) -> tuple[ThemeStory, ...]:
    """Project an M2-only run onto M5's input.

    The spec's documented degradation path. Themes built this way group
    *near-duplicate* stories, because nothing has merged the rewrites yet;
    the caller should record that it took this path.
    """

    stories = tuple(
        ThemeStory(
            story_key=story.story_key,
            ticker=story.ticker,
            title=story.title,
            description=story.description,
            published_at=story.published_at,
            outlets=tuple(story.outlets),
            item_ids=tuple(story.member_ids),
            source_links=_links(story.source_links),
            outlet_count=len({outlet for outlet in story.outlets if outlet}),
            member_story_keys=(story.story_key,),
            quarantined_member_ids=tuple(story.quarantined_member_ids),
            provider_conflicts=tuple(story.provider_conflicts),
            # M3 did not run, so there is no semantic skip to report; the
            # quarantine itself still travels on its own fields.
            semantic_skip_reason=(
                "provider_quarantine" if story.is_quarantined else None
            ),
            merge_evidence=(),
        )
        for story in stories_from_dedup(result, raw_items)
    )
    _reject_overlapping_items(stories)
    return stories


def source_metadata_from_exact(result: DedupResult) -> ThemeSourceMetadata:
    """Record an M2-only run as the source, so the degradation is visible."""

    return ThemeSourceMetadata(
        stage="m2.exact",
        algorithm_version=result.algorithm_version,
        config_fingerprint=result.config_fingerprint,
        # No encoder ran: M2 makes no vectors, and claiming one here would
        # attribute the themes to a model that took no part in producing
        # the stories they were built from.
        model_name="",
        model_revision=None,
        embedding_dimension=None,
        story_count=len(result.clusters),
        quarantined_story_count=len(set(result.quarantined_item_ids)),
        semantically_skipped_story_count=0,
        merged_story_count=0,
    )


#: How :func:`first_available_descriptions` picks, named so the choice is
#: fingerprinted and so nobody has to read the loop to know what it did.
DESCRIPTION_SELECTION_POLICY = (
    "first_non_empty_description_in_member_id_order; not canonical "
    "provenance, because M3 publishes no canonical raw member"
)


def first_available_descriptions(
    result: SemanticDedupResult, raw_descriptions: Mapping[str, str | None]
) -> dict[str, str | None]:
    """Pick one standfirst per M3 story, by member-id order.

    **This is not the canonical member's description, and does not claim to
    be.**  M3's public result exposes ``canonical_story_key`` — which M2
    *story* was chosen — but no canonical *raw item*: ``member_ids`` is the
    sorted union of the members' items, so its first element is whichever
    id sorts first, not the article the story is named after. The previous
    name and docstring said "the canonical member's standfirst wins", which
    inferred canonical ownership from a sort order that carries none.

    So the rule is stated for what it is: the first non-empty description in
    ``member_ids`` order. Deterministic and reproducible; not provenance.

    *Missing public contract:* M3 would need to publish the raw item id
    behind ``canonical_story_key`` for a caller to recover the canonical
    standfirst. It still does not, and the gap is in M3's public result
    rather than in persistence or orchestration, both of which landed with
    I1 and I4. A caller that needs true canonical provenance should pass its
    own mapping rather than use this helper; I5's projection resolves the
    canonical member through M2's cluster index for exactly this reason.
    """

    chosen: dict[str, str | None] = {}
    for story in result.stories:
        description = None
        for item_id in story.member_ids:
            candidate = raw_descriptions.get(item_id)
            if candidate:
                description = candidate
                break
        chosen[story.story_fingerprint] = description
    return chosen


#: Former name.  It claimed a provenance the public M3 model cannot
#: support; kept so callers do not break, documented so nobody believes it.
descriptions_from_semantic = first_available_descriptions

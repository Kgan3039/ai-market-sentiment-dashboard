"""The one place M5 knows what an M3 story looks like.

M5 clusters *canonical* stories — one per event, after M2 collapsed exact
copies and M3 collapsed rewrites. Feeding it raw duplicates would let the
size of a syndication burst decide the shape of the day's themes.

The bridge also accepts M2 output directly, for the case the spec's Section
9 anticipates: "if semantic dedup tuning slips, M5 proceeds on stage-1/2
dedup output". That path is a documented degradation, not a default.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from nlp.dedup import DedupResult, RawItem
from nlp.semdedup import SemanticDedupResult
from nlp.semdedup.bridge import stories_from_dedup

from .models import ThemeStory


def _links(source_links) -> tuple[tuple[str, str, str | None], ...]:
    return tuple(sorted((link.item_id, link.outlet, link.url) for link in source_links))


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
    return tuple(
        ThemeStory(
            story_key=story.story_fingerprint,
            ticker=story.ticker,
            title=story.canonical_title,
            description=lookup.get(story.story_fingerprint),
            published_at=story.published_at,
            outlets=tuple(story.outlets),
            item_ids=tuple(story.member_ids),
            source_links=_links(story.source_links),
        )
        for story in result.stories
    )


def theme_stories_from_exact(
    result: DedupResult, raw_items: Sequence[RawItem]
) -> tuple[ThemeStory, ...]:
    """Project an M2-only run onto M5's input.

    The spec's documented degradation path. Themes built this way group
    *near-duplicate* stories, because nothing has merged the rewrites yet;
    the caller should record that it took this path.
    """

    return tuple(
        ThemeStory(
            story_key=story.story_key,
            ticker=story.ticker,
            title=story.title,
            description=story.description,
            published_at=story.published_at,
            outlets=tuple(story.outlets),
            item_ids=tuple(story.member_ids),
            source_links=_links(story.source_links),
        )
        for story in stories_from_dedup(result, raw_items)
    )


def descriptions_from_semantic(
    result: SemanticDedupResult, raw_descriptions: Mapping[str, str | None]
) -> dict[str, str | None]:
    """Pick each M3 story's description from its member raw items.

    The canonical member's standfirst wins; failing that, the first member
    in the story's own order that has one. Deterministic and explainable:
    the story is represented by the article it is named after.
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

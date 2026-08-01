"""The one contradiction a theme may not contain.

M3's guards decide whether two records are the *same story*.  A theme is a
coarser thing — "today's coverage of the earnings report" legitimately
holds stories about different quarters, different magnitudes, and different
people — so applying M3's full guard set here would shred every theme into
singletons.

Exactly one family transfers: **opposing claims**.  A theme whose members
say both "raised guidance" and "cut guidance" cannot be summarized in two
to four faithful sentences; whichever the summarizer picks, the theme
contains its contradiction.  Nothing else in M3's set is a theme-level
problem.

When a theme does contain both polarities the minority is moved to "Other
coverage" rather than dropped, and every move is reported.  Majority is by
story count, ties broken by salience order then story key, so the outcome
is a function of the data.
"""

from __future__ import annotations

from typing import Sequence

from nlp.dedup.structural import tokenize
from nlp.dedup.text import display_text
from nlp.semdedup.evidence import contrasts

from .models import ThemeStory


def story_polarities(story: ThemeStory) -> frozenset[tuple[str, str]]:
    """Return the ``(family, polarity)`` claims one story makes."""

    text = display_text(
        " ".join(part for part in (story.title, story.description or "") if part)
    )
    return frozenset(contrasts(tokenize(text)))


def conflicting_members(
    stories: Sequence[ThemeStory], ordered_positions: Sequence[int]
) -> tuple[int, ...]:
    """Return the positions to eject so the theme stops contradicting itself.

    ``ordered_positions`` must already be in the order the caller considers
    authoritative (highest salience first); the majority tie-break follows
    it, so a theme's identity survives the ejection rather than flipping to
    whichever half happened to be larger by one.
    """

    polarities = {
        position: story_polarities(stories[position]) for position in ordered_positions
    }
    families: dict[str, dict[str, list[int]]] = {}
    for position in ordered_positions:
        for family, polarity in sorted(polarities[position]):
            families.setdefault(family, {}).setdefault(polarity, []).append(position)

    ejected: set[int] = set()
    for family in sorted(families):
        sides = families[family]
        if len(sides) < 2:
            continue
        # Larger side wins; on a tie the side holding the theme's leading
        # story wins, which keeps the theme about what it was about.
        winner = max(
            sorted(sides),
            key=lambda polarity: (
                len(sides[polarity]),
                -ordered_positions.index(
                    min(sides[polarity], key=ordered_positions.index)
                ),
            ),
        )
        for polarity, positions in sides.items():
            if polarity != winner:
                ejected.update(positions)
    return tuple(sorted(ejected))

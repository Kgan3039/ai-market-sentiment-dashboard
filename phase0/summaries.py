"""A2: from a persisted theme population to a frozen generation input.

The production source of a summary's evidence is what the ``themes`` stage
*stored*, read back in one snapshot by
:meth:`phase0.repository.Phase0Reader.theme_population`.  This module takes
that :class:`~phase0.repository.ThemePopulation` **value** -- never a
connection, never the in-memory M5 ``Theme`` the stage clustered -- and
projects one of its themes onto :class:`ai.guarded_summary.SummaryGenerationInput`.

It is read-only in every sense: it opens nothing, writes nothing, logs
nothing, and holds no repository.  A caller reads a population, hands it
here, and hands the frozen input to :func:`ai.guarded_summary.generate_guarded_summary`.

**The population is judged before anything is built.**  A theme set is only
a valid view of the day if the current story generation is healthy
``m3.semantic`` output, the set places exactly the live stories, and the set
recorded the number of stories it actually clustered.  Those are the same
persisted invariants A4a's review sampler applies before it samples; they
are restated here, locally and minimally, rather than imported from the
review tooling, because production input construction must not depend on a
private evaluation helper.  A population that fails them is refused with a
stable code, never summarized as if it were healthy.

**Citation identity is the durable row.**  A citation id is
``story:<stories.id>``.  ``cluster_fingerprint`` is a change-detection
handle by its own contract and is not used.  The id names a canonical
story; resolving it to concrete raw items and URLs is provenance that
travels *beside* the evidence (``raw_item_ids``, ``urls``), not a claim
that the sentence citing it is semantically supported.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from ai.guarded_summary import (
    EvidenceStory,
    SummaryGenerationInput,
    ThemeReference,
    citation_id_for,
)

from .errors import Phase0IntegrityError
from .repository import PersistedStory, ThemeMembership, ThemePopulation
from .themes import CanonicalClusterUnrecoverable, story_description

#: The only story stage a theme set may be summarized over.
HEALTHY_STORY_STAGE = "m3.semantic"
#: The one deliberate degradation the story stage records (decision H:
#: no theme set exists over it, and none may be summarized).
DEGRADED_STORY_STAGE = "m2.exact"

#: Stable refusal codes.  Each names one persisted invariant.
REFUSED_NO_THEME_SET = "theme_set_missing"
REFUSED_NO_LIVE_STORIES = "no_live_stories"
REFUSED_DEGRADED_GENERATION = "degraded_story_generation"
REFUSED_MIXED_STAGES = "mixed_story_stages"
REFUSED_SET_INCONSISTENT = "theme_set_inconsistent_with_stories"
REFUSED_SOURCE_COUNT = "source_story_count_mismatch"
REFUSED_SOURCE_COUNT_MALFORMED = "source_story_count_malformed"
REFUSED_DUPLICATE_MEMBERSHIP = "duplicate_theme_membership"
REFUSED_UNKNOWN_THEME = "unknown_theme"
REFUSED_THEME_STORY_NOT_LIVE = "theme_story_not_live"
REFUSED_EMPTY_THEME = "empty_theme"
REFUSED_STORY_TEXT = "story_text_unrecoverable"
REFUSED_TIMESTAMP = "invalid_published_at"

REFUSAL_CODES: tuple[str, ...] = (
    REFUSED_NO_THEME_SET,
    REFUSED_NO_LIVE_STORIES,
    REFUSED_DEGRADED_GENERATION,
    REFUSED_MIXED_STAGES,
    REFUSED_SET_INCONSISTENT,
    REFUSED_SOURCE_COUNT,
    REFUSED_SOURCE_COUNT_MALFORMED,
    REFUSED_DUPLICATE_MEMBERSHIP,
    REFUSED_UNKNOWN_THEME,
    REFUSED_THEME_STORY_NOT_LIVE,
    REFUSED_EMPTY_THEME,
    REFUSED_STORY_TEXT,
    REFUSED_TIMESTAMP,
)


class SummaryInputError(Phase0IntegrityError):
    """The population cannot be summarized; ``code`` says why."""

    def __init__(self, code: str, message: str) -> None:
        if code not in REFUSAL_CODES:
            raise ValueError(f"unknown refusal code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


# ----------------------------------------------------------------------
# The health gate
# ----------------------------------------------------------------------


def assess_population(population: ThemePopulation) -> tuple[str, str] | None:
    """Why a population is not summarizable, or ``None`` when it is.

    Checked in this order, and the first failure is the answer:

    1. a theme set exists;
    2. the partition holds live stories;
    3. every live story is ``m3.semantic`` (an all-``m2.exact`` generation
       is the recorded degradation; a mix is corruption);
    4. the set places exactly the live story ids -- across its themes,
       Other coverage, and exclusions -- with no story placed twice;
    5. the set's recorded input story count, when the set recorded one,
       is a real integer equal to the live count.

    ``source_metadata.story_count`` is optional for compatibility: the
    theme stage has always written it, but a set persisted without a
    ``source_metadata`` block at all (or without that key) is judged on
    the placement check alone, which already requires the set to account
    for exactly the live stories.  When the key *is* present it is held
    to its meaning: a bool, a string, a float, or anything else that is
    not an ``int`` is malformed metadata and refuses the population --
    ``"5"`` is not coerced to ``5``, because persisted metadata that has
    to be repaired on the way in is not verified input.
    """

    theme_set = population.theme_set
    live = population.stories.stories
    if theme_set is None:
        return REFUSED_NO_THEME_SET, "no theme set is persisted for this partition"
    if not live:
        return (
            REFUSED_NO_LIVE_STORIES,
            f"theme set {theme_set.theme_set_id} persists but the partition holds "
            "no live stories; the set is stale",
        )
    stages = sorted(population.stories.stages)
    if stages == [DEGRADED_STORY_STAGE]:
        return (
            REFUSED_DEGRADED_GENERATION,
            f"{len(live)} stories carry stage={DEGRADED_STORY_STAGE}; M3 did not "
            "complete and the generation may not be summarized",
        )
    if stages != [HEALTHY_STORY_STAGE]:
        return (
            REFUSED_MIXED_STAGES,
            f"live stories carry stages {stages}; only {HEALTHY_STORY_STAGE} "
            "output may be summarized",
        )
    placed: list[int] = [
        story_id for theme in population.themes for story_id in theme.story_ids
    ]
    placed += [entry.story_id for entry in population.other_coverage]
    placed += [entry.story_id for entry in population.excluded]
    if len(placed) != len(set(placed)):
        duplicates = sorted({s for s in placed if placed.count(s) > 1})
        return (
            REFUSED_DUPLICATE_MEMBERSHIP,
            f"stories {duplicates} are placed more than once in theme set "
            f"{theme_set.theme_set_id}",
        )
    live_ids = sorted(story.story_id for story in live)
    if sorted(placed) != live_ids:
        return (
            REFUSED_SET_INCONSISTENT,
            f"theme set {theme_set.theme_set_id} places {sorted(placed)} but the "
            f"live generation is {live_ids}",
        )
    metadata = theme_set.source_metadata or {}
    if "story_count" in metadata:
        recorded = metadata["story_count"]
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            return (
                REFUSED_SOURCE_COUNT_MALFORMED,
                f"theme set {theme_set.theme_set_id} recorded a story_count of type "
                f"{type(recorded).__name__}, not an integer; the metadata is malformed",
            )
        if recorded != len(live):
            return (
                REFUSED_SOURCE_COUNT,
                f"theme set {theme_set.theme_set_id} recorded {recorded} input "
                f"stories but the live generation holds {len(live)}",
            )
    return None


def require_healthy_population(population: ThemePopulation) -> None:
    """Raise :class:`SummaryInputError` unless :func:`assess_population` passes."""

    verdict = assess_population(population)
    if verdict is not None:
        code, message = verdict
        raise SummaryInputError(code, message)


# ----------------------------------------------------------------------
# Projection
# ----------------------------------------------------------------------


def _find_theme(population: ThemePopulation, theme_id: int) -> ThemeMembership:
    matches = [theme for theme in population.themes if theme.theme_id == theme_id]
    if not matches:
        raise SummaryInputError(
            REFUSED_UNKNOWN_THEME,
            f"theme {theme_id} is not in the persisted theme set for "
            f"{population.ticker} {population.trading_day} "
            f"{population.pipeline_version}",
        )
    if len(matches) > 1:
        raise SummaryInputError(
            REFUSED_DUPLICATE_MEMBERSHIP, f"theme {theme_id} appears more than once"
        )
    return matches[0]


def utc_published_at(value: str | None, story_id: int) -> str:
    """Render a persisted timestamp in UTC, or refuse it.

    Persistence validates the column as ISO-8601 on the way in, so a value
    that will not parse, or that carries no offset, is corruption.  An
    absent timestamp is rendered as the empty string: the model is told
    nothing rather than something invented.
    """

    if value is None:
        return ""
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise SummaryInputError(
            REFUSED_TIMESTAMP, f"story {story_id} published_at is not ISO-8601"
        ) from exc
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        raise SummaryInputError(
            REFUSED_TIMESTAMP, f"story {story_id} published_at carries no UTC offset"
        )
    return stamp.astimezone(timezone.utc).isoformat()


def evidence_for(story: PersistedStory) -> EvidenceStory:
    """Project one live persisted story onto the model-visible evidence shape.

    Title is the canonical title.  Description is what
    :func:`phase0.themes.story_description` recovers -- the standfirst M3
    embedded and M5 clustered, or nothing -- never a member's text the
    story was not encoded from.  Outlet is the story's own, else the
    lexicographically first member outlet.  Provenance is every member's
    raw item id in persisted position order, and the distinct URLs in that
    order with the canonical URL first.
    """

    try:
        description = story_description(story)
    except CanonicalClusterUnrecoverable as exc:
        raise SummaryInputError(
            REFUSED_STORY_TEXT,
            f"story {story.story_id}: its embedded text cannot be recovered",
        ) from exc
    ordered = sorted(story.members, key=lambda member: member.position)
    outlet = story.outlet or ""
    if not outlet:
        named = sorted({member.outlet for member in ordered if member.outlet})
        outlet = named[0] if named else ""
    urls: list[str] = []
    for candidate in [story.canonical_url] + [
        member.canonical_url or member.url for member in ordered
    ]:
        if candidate and candidate not in urls:
            urls.append(candidate)
    return EvidenceStory(
        citation_id=citation_id_for(story.story_id),
        persisted_story_id=story.story_id,
        title=story.canonical_title,
        description=description or "",
        outlet=outlet,
        published_at=utc_published_at(story.published_at, story.story_id),
        raw_item_ids=tuple(member.raw_item_id for member in ordered),
        urls=tuple(urls),
    )


def build_generation_input(
    population: ThemePopulation, theme_id: int
) -> SummaryGenerationInput:
    """The frozen A2 input for one theme of one persisted population.

    Refuses an unhealthy population first (:func:`assess_population`),
    then a theme the set does not hold, then a theme whose members are
    not all live stories.  The evidence is exactly the theme's member
    stories in persisted membership order; Other coverage and exclusions
    are not stories the theme is made of and cannot appear.
    """

    require_healthy_population(population)
    theme = _find_theme(population, int(theme_id))
    if not theme.story_ids:
        raise SummaryInputError(
            REFUSED_EMPTY_THEME, f"theme {theme.theme_id} has no member stories"
        )
    if len(theme.story_ids) != len(set(theme.story_ids)):
        raise SummaryInputError(
            REFUSED_DUPLICATE_MEMBERSHIP,
            f"theme {theme.theme_id} lists a story more than once",
        )
    live = {story.story_id: story for story in population.stories.stories}
    missing = [story_id for story_id in theme.story_ids if story_id not in live]
    if missing:
        raise SummaryInputError(
            REFUSED_THEME_STORY_NOT_LIVE,
            f"theme {theme.theme_id} places stories {missing} that are not in "
            "the live generation",
        )
    evidence: Sequence[EvidenceStory] = [
        evidence_for(live[story_id]) for story_id in theme.story_ids
    ]
    return SummaryGenerationInput.compose(
        ticker=population.ticker,
        trading_day=population.trading_day,
        theme=ThemeReference(
            theme_id=theme.theme_id,
            theme_key=theme.theme_key or "",
            label=theme.label,
            pipeline_version=population.pipeline_version,
        ),
        evidence=evidence,
    )


__all__ = [
    "DEGRADED_STORY_STAGE",
    "HEALTHY_STORY_STAGE",
    "REFUSAL_CODES",
    "SummaryInputError",
    "assess_population",
    "build_generation_input",
    "evidence_for",
    "require_healthy_population",
    "utc_published_at",
]

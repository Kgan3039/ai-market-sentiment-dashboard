"""The narrow adapter from an M5 theme to the citation-safe summarizer.

:mod:`ai.summarization` already fixes the shape it consumes — a
``ThemeInput`` of ``MemberStory`` records with ``id``, ``title``,
``description``, ``outlet`` and ``published_at`` — and
:class:`~nlp.themes.models.ThemeEvidence` is not that shape.  Something has
to translate, and until now nothing did: "M5 prepares the summarizer's
evidence" was true about the *contents* and untested about the *contract*.

This is the translation and nothing more.  The summarizer's behaviour is
untouched; no model is called from here and no network is reached.

**The citation contract is the point.**  ``story_key`` becomes the
summarizer's story ``id`` verbatim, so a citation the model emits resolves
back to exactly one member story and to that story's raw items.  A theme is
the whole permitted universe: other-coverage and excluded stories are
refused rather than quietly filtered, because a caller passing one is
asking for a citation the theme cannot support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from .errors import ThemeInputError
from .models import Theme, ThemeSet

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ai.summarization import MemberStory, ThemeInput

#: How a member story becomes a summarizer record.  Fingerprinted, so a
#: change to the mapping invalidates cached summaries.
ADAPTER_POLICY: dict[str, str] = {
    "version": "m5.summarization_adapter.v1",
    "story_id": "theme_evidence.story_key_verbatim",
    "outlet": "single_lexicographically_first_of_the_story_outlets",
    "multi_outlet": "extra_outlets_appended_to_description_never_dropped",
    "published_at": "isoformat_utc",
    "description": "story_description_or_empty_string",
    "member_scope": "theme_membership_only",
    "rejects": "other_coverage, excluded, unknown story keys, duplicates",
    "ordering": "theme_evidence_order_preserved",
}

#: Written into the description when a story carries more than one outlet,
#: so the summarizer sees every carrier even though its record holds one.
MULTI_OUTLET_PREFIX = "Also carried by: "


def adapter_policy_components() -> dict[str, str]:
    """The adapter policy, sorted, for the configuration fingerprint."""

    return dict(sorted(ADAPTER_POLICY.items()))


def _member_story(entry, ticker: str) -> "MemberStory":
    from ai.summarization import MemberStory

    outlets = tuple(sorted({outlet for outlet in entry.outlets if outlet}))
    description = entry.description or ""
    if len(outlets) > 1:
        # The summarizer's record holds one outlet.  Dropping the rest would
        # lose exactly the syndication evidence a reader uses to judge how
        # widely a story was carried, so they travel in the description
        # rather than disappearing.
        extra = ", ".join(outlets[1:])
        description = (
            f"{description} {MULTI_OUTLET_PREFIX}{extra}."
            if description
            else f"{MULTI_OUTLET_PREFIX}{extra}."
        ).strip()
    return MemberStory(
        id=entry.story_key,
        title=entry.title,
        description=description,
        outlet=outlets[0] if outlets else "",
        published_at=(
            entry.published_at.isoformat() if entry.published_at is not None else ""
        ),
    )


def theme_to_summarizer_input(theme: Theme) -> "ThemeInput":
    """Convert one theme into the summarizer's public input.

    Raises :class:`~nlp.themes.errors.ThemeInputError` when the theme's own
    evidence is not a clean set — a duplicate member would give the model
    two records with one id, and every citation to it would be ambiguous.
    """

    from ai.summarization import ThemeInput

    keys = [entry.story_key for entry in theme.evidence]
    if len(keys) != len(set(keys)):
        raise ThemeInputError(
            f"theme {theme.fingerprint} carries duplicate member story keys; "
            "a summarizer id must resolve to exactly one story"
        )
    if set(keys) != set(theme.member_story_keys):
        raise ThemeInputError(
            f"theme {theme.fingerprint} evidence does not match its membership; "
            "the summarizer may only see what the theme is made of"
        )
    return ThemeInput(
        ticker=theme.ticker,
        member_stories=[_member_story(entry, theme.ticker) for entry in theme.evidence],
        trading_day=theme.trading_day.isoformat(),
    )


def summarizer_inputs(theme_set: ThemeSet) -> dict[str, "ThemeInput"]:
    """Convert every *normal* theme, keyed by ``theme_key``.

    Other coverage and excluded stories are absent by construction: they are
    not themes, and there is nothing for a summarizer to say about them that
    a citation could support.
    """

    return {
        theme.theme_key: theme_to_summarizer_input(theme) for theme in theme_set.themes
    }


def citable_item_ids(theme: Theme) -> Mapping[str, tuple[str, ...]]:
    """Map each summarizer story id to the raw items a citation resolves to."""

    return {
        entry.story_key: tuple(sorted(set(entry.item_ids))) for entry in theme.evidence
    }


def unresolved_citations(theme: Theme, citation_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the citation ids this theme cannot support, sorted.

    The mirror of :func:`ai.summarization.resolve_citations`, expressed
    against a theme so a caller can check a summary without rebuilding the
    input.
    """

    permitted = {entry.story_key for entry in theme.evidence}
    return tuple(sorted({str(value) for value in citation_ids} - permitted))

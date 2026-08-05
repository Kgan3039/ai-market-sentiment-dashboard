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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Mapping, Sequence

from .errors import ThemeInputError
from .models import Theme, ThemeSet, validate_theme_set_invariants

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ai.summarization import MemberStory, ThemeInput

#: How a member story becomes a summarizer record.  Fingerprinted, so a
#: change to the mapping invalidates cached summaries.
ADAPTER_POLICY: dict[str, str] = {
    "version": "m5.summarization_adapter.v2",
    "story_id": "theme_evidence.story_key_verbatim",
    "outlet": "single_deterministic_primary_lexicographically_first",
    "carrier_metadata": (
        "full outlet list travels in AdaptedTheme.carriers, outside "
        "MemberStory; never inserted into title or description"
    ),
    "publisher_text": "title_and_description_verbatim_never_modified",
    "published_at": "utc_normalized_isoformat",
    "naive_timestamp": "rejected",
    "description": "story_description_verbatim_or_empty_string",
    "member_scope": "theme_membership_only",
    "rejects": (
        "other_coverage, excluded, unknown story keys, duplicates, "
        "incomplete_or_contradictory_theme_sets"
    ),
    "ordering": "theme_evidence_order_preserved",
    "validation": "central_theme_set_invariants_revalidated_before_adapting",
}


def adapter_policy_components() -> dict[str, str]:
    """The adapter policy, sorted, for the configuration fingerprint."""

    return dict(sorted(ADAPTER_POLICY.items()))


def utc_isoformat(stamp: datetime | None) -> str:
    """Render a timestamp in UTC, or refuse it.

    The adapter advertises UTC.  ``datetime.isoformat`` renders whatever
    offset the value carries, so a +05:00 story went out labelled UTC and
    five hours wrong.  Aware values are converted; naive ones are refused,
    because the upstream contract requires awareness and guessing a zone
    here would put a story on the wrong trading day silently.
    """

    if stamp is None:
        return ""
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        raise ThemeInputError(
            "the summarization adapter needs timezone-aware timestamps; "
            f"{stamp!r} is naive and its UTC instant cannot be recovered"
        )
    return stamp.astimezone(timezone.utc).isoformat()


def primary_outlet(outlets: Sequence[str]) -> str:
    """The one outlet the summarizer record names, chosen deterministically."""

    named = sorted({outlet for outlet in outlets if outlet})
    return named[0] if named else ""


def _member_story(entry) -> "MemberStory":
    """Project one member story, **without touching publisher text**.

    Title and description travel verbatim.  The earlier adapter appended
    "Also carried by: …" to the description so the extra outlets survived,
    which made adapter metadata indistinguishable from something a
    publisher wrote — and the summarizer's whole job is to quote publisher
    text faithfully.  The carrier list now travels beside the records, in
    :class:`AdaptedTheme`, where nothing can mistake it for evidence.
    """

    from ai.summarization import MemberStory

    return MemberStory(
        id=entry.story_key,
        title=entry.title,
        description=entry.description or "",
        outlet=primary_outlet(entry.outlets),
        published_at=utc_isoformat(entry.published_at),
    )


@dataclass(frozen=True)
class AdaptedTheme:
    """One theme in the summarizer's shape, plus the metadata it cannot hold.

    ``theme_input`` is exactly what :func:`ai.summarization.summarize`
    consumes.  Everything else is M5's own record, kept outside the model's
    text so it can never be quoted back as though a publisher had written
    it.
    """

    theme_key: str
    fingerprint: str
    theme_input: "ThemeInput"
    #: ``story_key -> every outlet that carried it``, sorted.
    carriers: Mapping[str, tuple[str, ...]]
    #: ``story_key -> the raw items a citation may resolve to``, sorted.
    citable_items: Mapping[str, tuple[str, ...]]


def adapt_theme(theme: Theme) -> AdaptedTheme:
    """Convert one theme, with its carrier and citation metadata."""

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
    return AdaptedTheme(
        theme_key=theme.theme_key,
        fingerprint=theme.fingerprint,
        theme_input=ThemeInput(
            ticker=theme.ticker,
            member_stories=[_member_story(entry) for entry in theme.evidence],
            trading_day=theme.trading_day.isoformat(),
        ),
        carriers={
            entry.story_key: tuple(sorted({o for o in entry.outlets if o}))
            for entry in theme.evidence
        },
        citable_items={
            entry.story_key: tuple(sorted(set(entry.item_ids)))
            for entry in theme.evidence
        },
    )


def theme_to_summarizer_input(theme: Theme) -> "ThemeInput":
    """Convert one theme into the summarizer's public input alone."""

    return adapt_theme(theme).theme_input


def validate_theme_set(theme_set: ThemeSet) -> None:
    """Re-check a theme set's partition before anything is adapted.

    ``ThemeSet.__post_init__`` enforces these too, and this calls the same
    validator: a set can reach here having been unpickled, reconstructed
    field by field, or built by a future caller, and "it must have been
    validated on the way in" is not something the citation contract can
    afford to assume.
    """

    validate_theme_set_invariants(theme_set)
    if not theme_set.complete:
        raise ThemeInputError(
            f"theme set for {theme_set.ticker} {theme_set.trading_day} is not "
            f"complete: missing={list(theme_set.missing_story_keys)}, "
            f"unexpected={list(theme_set.unexpected_story_keys)}, "
            f"duplicated={list(theme_set.duplicate_membership_keys)}"
        )


def adapt_theme_set(theme_set: ThemeSet) -> dict[str, AdaptedTheme]:
    """Adapt every *normal* theme, keyed by ``theme_key``, after validating.

    Validation runs **before** the dictionary is built, and the result is
    checked afterwards: a comprehension keyed on a duplicate identity
    overwrites silently and returns a map shorter than the theme list,
    which is a theme lost with no error anywhere.
    """

    validate_theme_set(theme_set)
    adapted = {theme.theme_key: adapt_theme(theme) for theme in theme_set.themes}
    if len(adapted) != len(theme_set.themes):
        raise ThemeInputError(
            f"adapting {len(theme_set.themes)} themes produced {len(adapted)} "
            "entries; a theme identity was reused and a theme would have been "
            "dropped silently"
        )
    return adapted


def summarizer_inputs(theme_set: ThemeSet) -> dict[str, "ThemeInput"]:
    """Convert every *normal* theme, keyed by ``theme_key``.

    Validates the set first rather than trusting it.  Other coverage and
    excluded stories are absent by construction: they are not themes, and
    there is nothing for a summarizer to say about them that a citation
    could support.
    """

    return {
        key: adapted.theme_input for key, adapted in adapt_theme_set(theme_set).items()
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

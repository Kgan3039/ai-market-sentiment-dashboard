"""Outlet identity, canonicalized at the projection boundary (decision B).

A publisher carried by both providers would reach us twice — once as
``yahoo:<display name>``, the provider's own byline for it, and once as
``rss:<host>``, an article's resolved host — and M2 counts distinct outlets
to decide whether a story is syndicated, which M5 then spends 30% of its
salience weight on.  Uncorrected, one such publisher would read as two.
No such pair has been observed yet, which is why the mapping below is
empty; this module is where the correction goes if one is.

Correcting it inside :mod:`nlp.dedup.text` would move the ``text_policy``
fingerprint and invalidate every committed M2/M3/M4 result for a change
that has nothing to do with those algorithms.  So it happens here, in the
layer that already translates between the persisted world and the stage
world, and ``normalize_source`` is left exactly as it is.

Two rules make the result safe to hand to M2:

* **The fixed point.**  Every emitted value satisfies
  ``normalize_source(value) == value``.  It is established by *calling*
  :func:`nlp.dedup.text.normalize_source` — never by predicting what that
  function would do — and a value that is not already its own image is
  refused rather than repaired.  ``NormalizedItem.outlet`` then derives the
  outlet from this exact string, so there is one outlet representation and
  not two whose equality is assumed.
* **The reserved namespace.**  An unmapped source emits ``<scheme> <token>``
  or a bare ``<scheme>``, so a mapped publisher id may not be ``yahoo`` or
  ``rss`` and may not begin with ``yahoo `` or ``rss ``.  This is why the
  Yahoo Finance house byline would map to ``yahoofinance``: an unmapped
  ``yahoo:Finance`` emits exactly ``yahoo finance``, and the two would
  silently unify.

The mapping is deliberately empty.  Decision B's stop condition rests on
``docs/observations/i5-provider-observation-2026-08-23.md``, which found no
Yahoo/RSS equivalence at all in its window — neither MarketWatch nor
TechCrunch appeared as a Yahoo publisher — so there is no evidence-backed
entry to write, and inventing one would be the inference this whole module
exists to avoid.  Shipping empty is safe by construction: the fallback
preserves identity, so the two copies simply count as separate outlets,
which is today's behaviour.
"""

from __future__ import annotations

from typing import Mapping

from nlp.dedup.text import normalize_source, text_key

from .errors import Phase0ValidationError

#: Bumped when the mapping or the fallback changes.  Persisted outlets are
#: copied from M2's result, so a policy change is a re-baseline, not a
#: silent correction.
PUBLISHER_POLICY_VERSION = "i5.publisher.v1"

#: Recognized persisted source schemes.  ``RawItem.source`` is written as
#: ``yahoo:<publisher display name>`` or ``rss:<resolved host>``; nothing
#: else is a scheme, and a value that merely contains a colon is not
#: promoted into one.
PROVIDER_SCHEMES = frozenset({"yahoo", "rss"})

#: Explicit, reviewed publisher equivalences, keyed on the exact stored
#: source string.  Empty on purpose -- see the module docstring.  Entries
#: are evidence-backed only: two spellings unify here because an
#: observation artifact showed they are one publisher, never because they
#: look alike.
PUBLISHER_MAPPING: Mapping[str, str] = {}


class PublisherPolicyError(Phase0ValidationError):
    """A stored source the publisher policy cannot represent.

    Raised rather than repaired.  The projection turns this into
    *unprojectable* evidence, which stays counted and inspectable, and the
    only rescue is an explicit reviewed mapping entry.
    """


def _reserved(identifier: str) -> bool:
    """Would this mapped id land inside an unmapped scheme's namespace?"""

    return any(
        identifier == scheme or identifier.startswith(f"{scheme} ")
        for scheme in PROVIDER_SCHEMES
    )


def _validate_mapping(mapping: Mapping[str, str]) -> None:
    """Check the reviewed table at import, not at the first article."""

    for source, identifier in mapping.items():
        if not source or not source.strip():
            raise PublisherPolicyError("publisher mapping key must be non-empty")
        if not identifier or identifier != identifier.strip():
            raise PublisherPolicyError(
                f"publisher id for {source!r} must be non-empty and unpadded"
            )
        if _reserved(identifier):
            raise PublisherPolicyError(
                f"publisher id {identifier!r} for {source!r} lies inside the "
                f"reserved unmapped namespace; an unmapped source emits that "
                f"same value and the two would silently unify"
            )
        if normalize_source(identifier) != identifier:
            raise PublisherPolicyError(
                f"publisher id {identifier!r} for {source!r} is not a fixed "
                f"point of normalize_source"
            )


_validate_mapping(PUBLISHER_MAPPING)


def source_scheme(source: str | None) -> str | None:
    """The persisted provider scheme, or ``None`` when there is not one.

    Only :data:`PROVIDER_SCHEMES` count.  The scheme is what keeps the two
    provider-id spaces apart once a publisher has been unified, so it is
    read from the *stored* source and never from a canonical publisher id.
    """

    text = str(source or "").strip()
    scheme, separator, _ = text.partition(":")
    if not separator:
        return None
    scheme = scheme.strip().casefold()
    return scheme if scheme in PROVIDER_SCHEMES else None


def _squash(remainder: str) -> str:
    """Collapse a source remainder into one alphanumeric token.

    This is not public-suffix knowledge: nothing is classified, dropped, or
    interpreted.  The whole remainder is kept with its separators removed,
    which separates more than a space-preserving form would --
    ``example-news.com`` and ``example-news.co`` stay distinct where a
    space-preserving fallback would have collapsed both to ``examplenews``.
    """

    return "".join(text_key(remainder).split())


def canonical_publisher(source: str | None) -> str:
    """Return the outlet identity for one stored ``RawItem.source``.

    ``source`` is the sole input.  The URL host, Yahoo's nested provider
    fields, RSS feed metadata, and the title are all deliberately not
    consulted: inferring that two spellings are one publisher is exactly
    what decision B forbids outside the reviewed mapping.
    """

    text = str(source or "").strip()
    if not text:
        raise PublisherPolicyError("source is required to identify an outlet")

    mapped = PUBLISHER_MAPPING.get(text)
    if mapped is not None:
        return mapped

    scheme = source_scheme(text)
    if scheme is None:
        raise PublisherPolicyError(
            f"source {source!r} carries no recognized provider scheme; the "
            f"fallback preserves scheme identity and cannot invent one"
        )

    _, _, remainder = text.partition(":")
    token = _squash(remainder)
    identifier = f"{scheme} {token}" if token else scheme
    if normalize_source(identifier) != identifier:
        # Reachable: a remainder that squashes to a single noise suffix --
        # ``yahoo:Inc`` emits ``yahoo inc``, which normalize_source folds
        # back to ``yahoo`` and so collides with the bare scheme.  Refusing
        # is the honest answer; an explicit mapping entry is the rescue.
        raise PublisherPolicyError(
            f"source {source!r} canonicalizes to {identifier!r}, which is not "
            f"a fixed point of normalize_source; it needs an explicit "
            f"reviewed mapping entry"
        )
    return identifier


__all__ = [
    "PROVIDER_SCHEMES",
    "PUBLISHER_MAPPING",
    "PUBLISHER_POLICY_VERSION",
    "PublisherPolicyError",
    "canonical_publisher",
    "source_scheme",
]

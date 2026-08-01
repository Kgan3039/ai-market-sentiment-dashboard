"""Neutral text utilities shared by the M2 dedup stage.

Nothing here is statistical, model-driven, or embedding-specific: identical
input always produces identical output, in this process or any other.  M2
owns these definitions outright so that a change to M1's encoder input
composition can never silently move M2's identity keys.
"""

from __future__ import annotations

import html
import unicodedata

import hashlib
import re

from .errors import DedupInputError

#: Bumped when source/provider normalization or the ticker syntax changes.
#: Folded into the configuration fingerprint by :mod:`nlp.dedup.config`.
TEXT_POLICY_VERSION = "m2.text.v2"

#: Phase 0 symbols are plain equity tickers, optionally with a one-to-three
#: letter class or exchange suffix ("BRK.B").  One definition, used both to
#: validate a configured universe and to validate a record's ticker, so a
#: configuration can never hold a symbol no record could satisfy.
TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,9}(?:[.\-][A-Z]{1,3})?$")

#: Typographic variants folded before matching so a curly apostrophe or an em
#: dash never splits one story into two.
_TYPOGRAPHIC_MAP = str.maketrans(
    {
        " ": " ",  # no-break space
        "­": "",  # soft hyphen
        "‐": "-",  # hyphen
        "‑": "-",  # non-breaking hyphen
        "‒": "-",  # figure dash
        "–": "-",  # en dash
        "—": "-",  # em dash
        "―": "-",  # horizontal bar
        "‘": "'",  # left single quote
        "’": "'",  # right single quote
        "‚": "'",  # single low quote
        "‛": "'",  # single high-reversed quote
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "„": '"',  # double low quote
        "…": "...",  # ellipsis
        "′": "'",  # prime
        "″": '"',  # double prime
        "​": "",  # zero-width space
        "‌": "",  # zero-width non-joiner
        "‍": "",  # zero-width joiner
        "−": "-",  # minus sign
        "﻿": "",  # byte-order mark
    }
)

_SOURCE_NOISE_SUFFIXES = frozenset(
    {"co", "com", "inc", "io", "llc", "ltd", "net", "org"}
)


def require_optional_str(value: object, field: str) -> str | None:
    """Return ``value`` when it is a string or ``None``, else raise."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise DedupInputError(f"{field} must be a string or None")
    return value


def display_text(value: str | None) -> str:
    """Return human-readable text with markup and typography normalized.

    HTML entities are unescaped, compatibility characters folded, curly
    quotes and dashes flattened, and whitespace collapsed.  Casing, wording,
    and punctuation survive: this is what the UI shows, not a matching key.
    """

    if value is None:
        return ""
    if not isinstance(value, str):
        raise DedupInputError("text must be a string or None")
    unescaped = html.unescape(value)
    folded = unicodedata.normalize("NFKC", unescaped).translate(_TYPOGRAPHIC_MAP)
    printable = "".join(
        character if character.isprintable() else " " for character in folded
    )
    return " ".join(printable.split())


def strip_accents(value: str) -> str:
    """Return ``value`` with Latin combining marks removed.

    Folding is applied only where the decomposed base character is ASCII, so
    ``café`` becomes ``cafe`` while Cyrillic ``й``, Arabic, Hebrew, and
    Devanagari text keep every mark.  Stripping marks from those scripts can
    turn two different words into one key, which is exactly the false merge
    this module exists to prevent.
    """

    folded: list[str] = []
    for character in value:
        decomposed = unicodedata.normalize("NFKD", character)
        if decomposed[0].isascii():
            folded.append(
                "".join(part for part in decomposed if not unicodedata.combining(part))
            )
        else:
            folded.append(character)
    return "".join(folded)


def text_key(value: str | None) -> str:
    """Return a case-, accent-, and punctuation-insensitive matching key.

    Used only for outlet and publisher names, which have no numeric
    structure to protect.  Titles and descriptions go through
    :func:`nlp.dedup.structural.tokenize` instead, which preserves the
    currency, magnitude, and percentage markers this key would flatten.
    """

    text = display_text(value)
    if not text:
        return ""
    folded = strip_accents(text.casefold()).replace("'", "")
    spaced = "".join(character if character.isalnum() else " " for character in folded)
    return " ".join(spaced.split())


def provider_namespace(value: str | None) -> str:
    """Return the conservative namespace for a provider item id.

    Deliberately *not* :func:`normalize_source`: that folds away legal and
    domain suffixes, which would make ``Acme Inc`` and ``Acme LLC`` share an
    authoritative identity and let one feed's item id silently merge another
    company's article.  Only formatting is cleaned here — case, accents,
    punctuation, and whitespace — so two spellings share a namespace only
    when they really are the same string modulo formatting.  Being too
    strict costs at most a tier-1 merge, which the weaker signals can still
    make; being too loose merges unrelated records on a stranger's id.
    """

    text = display_text(value)
    if not text:
        return ""
    folded = strip_accents(text.casefold())
    spaced = "".join(character if character.isalnum() else " " for character in folded)
    return " ".join(spaced.split())


def normalize_source(value: str | None) -> str:
    """Return the display/outlet key (``"Reuters.com"`` becomes ``"reuters"``).

    Used for outlet counting and title-attribution stripping, never for
    provider identity — see :func:`provider_namespace`.
    """

    key = text_key(value)
    if not key:
        return ""
    tokens = key.split()
    if tokens and tokens[0] == "www":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in _SOURCE_NOISE_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def policy_fingerprint() -> str:
    """Return a digest of this module's static text policy."""

    payload = "|".join(
        (
            TEXT_POLICY_VERSION,
            TICKER_PATTERN.pattern,
            "".join(sorted(_SOURCE_NOISE_SUFFIXES)),
            repr(sorted(_TYPOGRAPHIC_MAP.items())),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

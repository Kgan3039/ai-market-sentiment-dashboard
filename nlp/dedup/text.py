"""Neutral text utilities shared by the M2 dedup stage.

Nothing here is statistical, model-driven, or embedding-specific: identical
input always produces identical output, in this process or any other.  M2
owns these definitions outright so that a change to M1's encoder input
composition can never silently move M2's identity keys.
"""

from __future__ import annotations

import html
import unicodedata

from .errors import DedupInputError

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


def normalize_source(value: str | None) -> str:
    """Return the publisher key (``"Reuters.com"`` becomes ``"reuters"``)."""

    key = text_key(value)
    if not key:
        return ""
    tokens = key.split()
    if tokens and tokens[0] == "www":
        tokens = tokens[1:]
    while len(tokens) > 1 and tokens[-1] in _SOURCE_NOISE_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)

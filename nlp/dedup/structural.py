"""Structural analysis of a headline.

M2's title matching is *structural*, never semantic.  After narrow,
auditable wire-service and publisher-attribution metadata is stripped, two
titles match only when their normalized token sequences are **identical**.
There is no filler allowlist, no stopword removal, and no similarity score
in the accept decision, so ordinary words stay meaningful::

    "The Who tour dates"      !=  "Who tour dates"
    "The A-Team returns"      !=  "A Team returns"
    "Acme Inc names a CEO"    !=  "Acme names a CEO"

and every meaning-bearing substitution is rejected by construction::

    "Apple names new CFO"        !=  "Apple names new COO"
    "Broadcom raises guidance"   !=  "Broadcom cuts guidance"
    "Tesla recalls 5 million"    !=  "Tesla recalls 5 billion"
    "Margin fell -5%"            !=  "Margin fell 5%"

What *is* stripped is metadata that is delimited and identifiable:

* leading wire decorations (``UPDATE 2-``, ``EXCLUSIVE:``, ``REFILE-``);
* a leading or trailing separator segment naming **this item's own
  source**, or an outlet in the versioned static policy
  :data:`KNOWN_PUBLISHER_AFFIXES`.

Nothing here ever learns removable names from other records.  A title's
normalization depends only on that title, its own ``source``, and the
static policy, so adding an unrelated record to a batch can never change
whether two other records merge, and a publisher first seen in a later run
cannot reinterpret headlines seen earlier.

Tokenization keeps the entire protected numeric expression intact: sign,
approximation marker, currency symbol, thousands separators, decimals,
ranges, percentages, and basis points all survive as one token.  When the
title contains numeric notation this module cannot represent faithfully,
the title is marked *uncertain* and produces no matching key at all, so an
unparseable number can never be silently equated with a different one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import re
import unicodedata

from .text import display_text, normalize_source, strip_accents

#: Bumped whenever tokenization, metadata stripping, or the protected
#: expression rules change.  It is folded into the configuration
#: fingerprint together with :func:`policy_fingerprint`, so a policy edit
#: invalidates cached output even if nobody remembers to bump this string.
STRUCTURE_VERSION = "m2.structure.v3"

#: Wire-service decorations that carry no story identity.  Reuters and AP
#: reuse these on updated copies of one headline.
_NEWSWIRE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:update|refile|corrected|correction|exclusive|breaking|analysis"
    r"|factbox|explainer|timeline|watch|live|video|photos?|column|opinion"
    r"|preview|recap|graphic|instant view|wrapup|wrap|table|press release"
    r"|just in|alert|developing|repeat)\s*\d*\s*[-:|>]+\s*",
    re.IGNORECASE,
)
_MAX_PREFIX_STRIPS = 3

#: Splits " - ", " | ", " -- " style separators without touching hyphenated
#: words such as "AI-driven".
_TITLE_SEGMENT_PATTERN = re.compile(r"\s+(?:\||-{1,2}|::|·|•|>)\s+|\s*\|\s*")

#: Publisher names stripped from a title's leading or trailing separator
#: segment even when they differ from ``source``.  Aggregators keep the
#: originating wire in the headline ("... - Reuters" served by Yahoo
#: Finance), which is exactly the syndication case M2 has to collapse.
#: Values are :func:`normalize_source` keys.  Ambiguous common words (for
#: example "Time", "Live", "Watch", "Target", "Apple") are deliberately
#: excluded: they are ordinary headline words far more often than they are
#: attribution.  This list is static and versioned through
#: :func:`policy_fingerprint`; it is never extended from the batch being
#: processed.
KNOWN_PUBLISHER_AFFIXES = frozenset(
    {
        "abc news",
        "ap",
        "associated press",
        "axios",
        "barrons",
        "bbc",
        "bbc news",
        "benzinga",
        "bloomberg",
        "business insider",
        "cbs news",
        "cnbc",
        "cnn",
        "cnn business",
        "financial times",
        "forbes",
        "fortune",
        "fox business",
        "ft",
        "insider monkey",
        "investing",
        "investors business daily",
        "marketwatch",
        "nbc news",
        "new york times",
        "npr",
        "nytimes",
        "politico",
        "quartz",
        "reuters",
        "seeking alpha",
        "techcrunch",
        "the economist",
        "the guardian",
        "the information",
        "the motley fool",
        "the new york times",
        "the verge",
        "the wall street journal",
        "the washington post",
        "usa today",
        "wall street journal",
        "washington post",
        "wsj",
        "yahoo",
        "yahoo finance",
        "zacks",
    }
)

#: Every Unicode currency sign, so "₹5" and "$5" are different tokens
#: without maintaining a hand-written symbol list.
_CURRENCY_SYMBOLS = "".join(
    chr(code)
    for code in range(0x0024, 0x20D0)
    if unicodedata.category(chr(code)) == "Sc"
)
_CURRENCY_CLASS = f"[{re.escape(_CURRENCY_SYMBOLS)}]"
_APPROX_CLASS = "[~≈∼]"
_MAGNITUDE = rf"[+\-]?{_CURRENCY_CLASS}?\d[\d,]*(?:\.\d+)?"
#: date | signed/approximate/ranged number with units | Unicode word.
#: Ordering matters: the number alternatives must win over the word rule.
_TOKEN_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    rf"|{_APPROX_CLASS}?{_MAGNITUDE}(?:\s*-\s*{_MAGNITUDE})?(?:%|bps|bp)?"
    r"|[^\W_]+",
    re.UNICODE,
)
_THOUSANDS_PATTERN = re.compile(r"^\d{1,3}(?:,\d{3})+$")
_POSSESSIVE_PATTERN = re.compile(r"'s(?=\b)")
#: Numeric notation this module does not represent faithfully.  Its
#: presence makes the whole title uncertain rather than silently lossy.
_UNSUPPORTED_NUMERIC = frozenset("‰‱٪﹪⁄")


@dataclass(frozen=True)
class TitleStructure:
    """Everything M2 derives from one headline."""

    #: Human-readable title, typography folded, for the UI.
    display: str
    #: Separator-delimited segments after wire-prefix stripping, kept for
    #: diagnostics.  They are never re-interpreted against other records.
    segments: tuple[str, ...]
    #: Normalized tokens in order, numeric expressions kept whole.
    tokens: tuple[str, ...]
    #: ``" ".join(tokens)``; ``None`` when empty or uncertain.
    title_key: str | None
    #: Ordered ``expression>following-token`` pairs, never sorted.
    protected: tuple[str, ...]
    #: Digest of tokens and protected expressions; ``None`` when the title
    #: carries no token or could not be parsed with confidence.
    structural_key: str | None
    #: True when the title used numeric notation M2 cannot represent.
    uncertain: bool


def strip_newswire_prefixes(title: str) -> str:
    """Remove leading ``UPDATE 2-`` / ``EXCLUSIVE:`` style decorations."""

    stripped = title
    for _ in range(_MAX_PREFIX_STRIPS):
        candidate = _NEWSWIRE_PREFIX_PATTERN.sub("", stripped, count=1)
        if candidate == stripped or not candidate.strip():
            break
        stripped = candidate
    return stripped.strip()


def split_segments(title: str) -> tuple[str, ...]:
    """Split a title on publisher-attribution separators."""

    return tuple(
        segment.strip()
        for segment in _TITLE_SEGMENT_PATTERN.split(title)
        if segment and segment.strip()
    )


def strip_attribution(
    segments: tuple[str, ...], publishers: frozenset[str]
) -> tuple[str, ...]:
    """Drop leading and trailing segments that name a publisher.

    Unrecognized segments (" - analysts", " - sources") are story content
    and always survive.  The last remaining segment is never removed, so a
    title cannot be normalized away entirely.
    """

    if len(segments) < 2:
        return segments
    remaining = list(segments)
    while len(remaining) > 1 and normalize_source(remaining[-1]) in publishers:
        remaining.pop()
    while len(remaining) > 1 and normalize_source(remaining[0]) in publishers:
        remaining.pop(0)
    return tuple(remaining)


def _normalize_number(token: str) -> str:
    """Collapse thousands separators inside an otherwise intact number."""

    def collapse(match: re.Match[str]) -> str:
        digits = match.group(0)
        return digits.replace(",", "") if _THOUSANDS_PATTERN.match(digits) else digits

    return re.sub(r"\d[\d,]*", collapse, token)


def has_unsupported_numeric_notation(text: str) -> bool:
    """Return True when ``text`` uses numeric notation M2 cannot parse."""

    for character in text:
        if character in _UNSUPPORTED_NUMERIC:
            return True
        if character.isdigit() and not character.isascii():
            return True
        try:
            unicodedata.numeric(character)
        except (TypeError, ValueError):
            continue
        if not (character.isascii() and character.isdigit()):
            return True
    return False


def tokenize(text: str) -> tuple[str, ...]:
    """Return the ordered tokens of an already-displayable text.

    Case, accents (Latin only), possessives, and punctuation are folded
    away.  A numeric expression keeps its approximation marker, sign,
    currency symbol, decimal point, thousands grouping, range, percent, and
    basis-point suffix, so ``~-5.5%``, ``5-10%``, ``₹5``, ``$5``, and ``5``
    are five different tokens.  Non-Latin scripts tokenize as ordinary
    words rather than disappearing.
    """

    folded = _POSSESSIVE_PATTERN.sub("", strip_accents(text.casefold()))
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(folded):
        token = "".join(match.group(0).split())
        if any(character.isdigit() for character in token):
            token = _normalize_number(token)
        tokens.append(token)
    return tuple(tokens)


def protected_expressions(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return every numeric expression bound to the token that follows it.

    Binding the neighbour keeps "5 million" distinct from "5 billion" and
    "$5 million" distinct from "5 million units".  Order is preserved and
    never sorted, so "up 5% to $10" cannot equal "up 10% to $5".
    """

    expressions: list[str] = []
    for index, token in enumerate(tokens):
        if any(character.isdigit() for character in token):
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            expressions.append(f"{token}>{following}")
    return tuple(expressions)


def _encode(fields: tuple[str, ...]) -> bytes:
    """Length-prefix every field so no value can forge a boundary."""

    encoded = bytearray()
    for field in fields:
        payload = field.encode("utf-8")
        encoded += str(len(payload)).encode("ascii") + b":" + payload
    return bytes(encoded)


def structural_key(tokens: tuple[str, ...], protected: tuple[str, ...]) -> str | None:
    """Return the identity of a token sequence, or ``None`` when empty."""

    if not tokens:
        return None
    payload = _encode((STRUCTURE_VERSION,) + tokens + ("\x00protected",) + protected)
    return hashlib.sha256(payload).hexdigest()


def analyze_title(title: str | None, source: str | None = None) -> TitleStructure:
    """Derive every title-based signal for one item."""

    display = display_text(title)
    uncertain = has_unsupported_numeric_notation(display)
    segments = split_segments(strip_newswire_prefixes(display))
    publishers = set(KNOWN_PUBLISHER_AFFIXES)
    source_key = normalize_source(source)
    if source_key:
        publishers.add(source_key)
    tokens = tokenize(" - ".join(strip_attribution(segments, frozenset(publishers))))
    protected = protected_expressions(tokens)
    if uncertain:
        return TitleStructure(
            display=display,
            segments=segments,
            tokens=tokens,
            title_key=None,
            protected=protected,
            structural_key=None,
            uncertain=True,
        )
    return TitleStructure(
        display=display,
        segments=segments,
        tokens=tokens,
        title_key=" ".join(tokens) if tokens else None,
        protected=protected,
        structural_key=structural_key(tokens, protected),
        uncertain=False,
    )


@lru_cache(maxsize=1)
def policy_fingerprint() -> str:
    """Return a digest of every static policy this module applies.

    Folded into :meth:`nlp.dedup.config.DedupConfig.fingerprint`, so
    editing the publisher list, the wire-prefix pattern, the separator
    pattern, or the tokenizer changes the configuration fingerprint
    automatically instead of relying on someone bumping a constant.
    """

    payload = _encode(
        (
            STRUCTURE_VERSION,
            _NEWSWIRE_PREFIX_PATTERN.pattern,
            str(_MAX_PREFIX_STRIPS),
            _TITLE_SEGMENT_PATTERN.pattern,
            _TOKEN_PATTERN.pattern,
            "".join(sorted(_UNSUPPORTED_NUMERIC)),
        )
        + tuple(sorted(KNOWN_PUBLISHER_AFFIXES))
    )
    return hashlib.sha256(payload).hexdigest()

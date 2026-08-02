"""Safety guards: what a high cosine similarity is not allowed to override.

Measured on the M4 labelled set with the Phase 0 encoder, cosine similarity
alone **cannot** separate same-story rewrites from hard negatives at any
threshold.  Genuine rewrites of one event score 0.42-0.73, because they
share almost no wording; template negatives that differ only in a date, a
magnitude, a role, or a sign score 0.97-0.996, because they share almost
all of it.  The classes are inverted with respect to similarity.

So M3 is not a threshold with guards bolted on: it is a set of guards with
a threshold behind them.  Each guard is a static, versioned, auditable
policy over the same tokenizer M2 uses, and each answers one question a
reader would ask:

``numeric``          do the two claim different magnitudes, currencies,
                     units, ranges, or signs?
``temporal``         do they describe different quarters, years, months, or
                     dates?
``role``             do they name different named roles (CFO vs COO)?
``contrast``         do they make opposing claims (raised vs cut, approved
                     vs rejected, beat vs missed, profit vs loss)?
``negation``         does one explicitly negate where the other does not?
``subject_shift``    is the *headline* about a supplier, reseller, or
                     agency rather than the company itself?
``entity``           do they name different explicit organisations or
                     people in the same slot?
``article_type``     is one a live blog, an analysis, a rumour, a
                     confirmation, an interview, or a hands-on where the
                     other is a plain report?
``same_frame``       do they share most of their wording and differ only in
                     the slot that carries the event?

The last one is the load-bearing guard for "same template, different
event".  When two headlines overlap heavily, the tokens they do *not* share
are the story, so a substitution in that slot is a different story however
close the vectors are.  A real rewrite has the opposite shape: low lexical
overlap, high semantic similarity, and it never triggers this guard.

``same_frame`` and ``subject_shift`` read the **headline only**.  A frame is
a headline template and a subject is a headline subject; running either over
the standfirst as well was a defect.  Two paraphrased standfirsts of one
briefing inflate the overlap until ordinary synonym choices look like a
swapped slot, and an attribution clause ("suppliers told the paper") is not
the article being about a supplier.  Both cost real rewrites: P054 and P068
respectively.

The asymmetry from M2 is preserved throughout: **explicit disagreement
vetoes, missing information does not.**  A story with no numbers does not
contradict one that has numbers.  And, as in M2, comparison works on a
summary of a *set* of stories, so a sparse story can never bridge two that
contradict each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence
import hashlib
import re

from nlp.dedup.compatibility import NEGATION_TOKENS
from nlp.dedup.structural import KNOWN_PUBLISHER_AFFIXES
from nlp.dedup.structural import policy_fingerprint as structural_policy_fingerprint
from nlp.dedup.structural import tokenize
from nlp.dedup.text import display_text

#: Bumped whenever a guard, a lexicon, or the comparison changes.
EVIDENCE_POLICY_VERSION = "m3.evidence.v5"

#: Bumped when the article-type classifier changes shape.
ARTICLE_TYPE_POLICY_VERSION = "m3.article_type.v3"
#: Bumped when cardinal normalization changes.
CARDINAL_POLICY_VERSION = "m3.quantity.v2"
#: Bumped when explicit-entity extraction changes.
ENTITY_POLICY_VERSION = "m3.entity.v3"

#: A numeric token as M2's tokenizer emits it: optional sign, optional
#: currency symbol, then a digit.  Deliberately stricter than "contains a
#: digit", which would treat a model name such as ``mi400`` as a magnitude.
_NUMERIC_TOKEN = re.compile(r"^[+\-]?[^\w\s]?\d")

#: Words that change what a bare number means, so they bind to it.  "5
#: million" and "5 billion" must not compare equal; "1,000 roles" and
#: "1,000 under the plan" must.
_MAGNITUDE_WORDS = frozenset(
    {
        "hundred",
        "thousand",
        "million",
        "billion",
        "trillion",
        "bn",
        "mn",
        "tn",
    }
)
_UNIT_WORDS = frozenset(
    {
        "bp",
        "bps",
        "cent",
        "cents",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "gb",
        "gw",
        "gwh",
        "kw",
        "kwh",
        "mb",
        "mw",
        "mwh",
        "pence",
        "percent",
        "percentage",
        "points",
        "pounds",
        "rupees",
        "tb",
        "tw",
        "twh",
        "yen",
        "yuan",
    }
)
_BOUND_WORDS = _MAGNITUDE_WORDS | _UNIT_WORDS

#: Counts a newsroom spells out rather than digitising.  "eleven European
#: markets" and "nine European markets" are the same disagreement as
#: "11" and "9", and a magnitude guard that only read digits could not see
#: it.  The multipliers (million, billion) stay out: they already bind to a
#: preceding number, and treating a bare one as a count would double-count
#: "5 million".
_CARDINAL_VALUES: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "dozen": 12,
}
#: Tens that can take a following unit ("twenty-one" tokenizes as two).
_TENS_WORDS = frozenset(
    {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
)
_UNIT_CARDINALS = frozenset(
    {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
)
_CARDINAL_WORDS = frozenset(
    {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "dozen",
    }
)

_QUARTER_WORDS = {
    "first": "q1",
    "second": "q2",
    "third": "q3",
    "fourth": "q4",
}
_QUARTER_TOKENS = {"q1": "q1", "q2": "q2", "q3": "q3", "q4": "q4"}
_MONTHS = {
    "january": "m01",
    "february": "m02",
    "march": "m03",
    "april": "m04",
    "may": "m05",
    "june": "m06",
    "july": "m07",
    "august": "m08",
    "september": "m09",
    "october": "m10",
    "november": "m11",
    "december": "m12",
}
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR = re.compile(r"^(19|20|21)\d{2}$")

#: Named roles, longest phrase first so "chief financial officer" is not
#: read as a bare "chief".  Only fires when *both* stories name a role.
_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cfo", re.compile(r"\b(?:cfo|chief financial officer|finance chief)\b")),
    ("coo", re.compile(r"\b(?:coo|chief operating officer|operations chief)\b")),
    ("cto", re.compile(r"\b(?:cto|chief technology officer|technology chief)\b")),
    ("cmo", re.compile(r"\b(?:cmo|chief marketing officer|marketing chief)\b")),
    ("ceo", re.compile(r"\b(?:ceo|chief executive officer|chief executive)\b")),
    ("chair", re.compile(r"\b(?:chair|chairman|chairwoman|chairperson)\b")),
)

#: Mutually exclusive claim families.  A veto needs a token from the same
#: family on *both* sides with *different* polarity, so a story that simply
#: does not take a position never blocks a merge.
_CONTRAST_GROUPS: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        "direction",
        frozenset(
            {
                "raises",
                "raise",
                "raised",
                "lifts",
                "lift",
                "lifted",
                "boosts",
                "boosted",
                "increases",
                "increase",
                "increased",
                "climbs",
                "climbed",
                "jumps",
                "jumped",
                "rises",
                "rise",
                "rose",
                "grows",
                "grew",
                "expands",
                "expanded",
                "higher",
                "stronger",
                "widens",
                "widened",
                "accelerates",
                "accelerated",
                "accelerate",
                "improves",
                "improved",
            }
        ),
        frozenset(
            {
                "cuts",
                "cut",
                "lowers",
                "lower",
                "lowered",
                "reduces",
                "reduce",
                "reduced",
                "trims",
                "trimmed",
                "falls",
                "fall",
                "fell",
                "drops",
                "drop",
                "dropped",
                "declines",
                "decline",
                "declined",
                "contracts",
                "contracted",
                "slows",
                "slowed",
                "weaker",
                "slips",
                "slipped",
                "narrows",
                "narrowed",
                "shrinks",
                "shrank",
            }
        ),
    ),
    (
        "outcome",
        frozenset(
            {
                "approves",
                "approved",
                "approval",
                "clears",
                "cleared",
                "grants",
                "granted",
                "upholds",
                "upheld",
                "accepts",
                "accepted",
                "allows",
                "allowed",
                "wins",
                "won",
                "authorises",
                "authorised",
                "authorizes",
            }
        ),
        frozenset(
            {
                "rejects",
                "rejected",
                "blocks",
                "blocked",
                "denies",
                "denied",
                "overturns",
                "overturned",
                "refuses",
                "refused",
                "loses",
                "lost",
                "bars",
                "barred",
                "vetoes",
                "vetoed",
            }
        ),
    ),
    (
        "performance",
        frozenset(
            {
                "beats",
                "beat",
                "tops",
                "topped",
                "exceeds",
                "exceed",
                "exceeded",
                "above",
                "outperformed",
                "outperforms",
                "ahead",
                "surpasses",
                "surpassed",
            }
        ),
        frozenset(
            {
                "misses",
                "miss",
                "missed",
                "trails",
                "trail",
                "trailed",
                "below",
                "short",
                "lags",
                "lagged",
                "behind",
                "underperforms",
                "underperformed",
            }
        ),
    ),
    (
        "result",
        frozenset({"profit", "profits", "profitable", "surplus", "gain", "gains"}),
        frozenset({"loss", "losses", "deficit", "shortfall"}),
    ),
    (
        "commitment",
        frozenset({"maintains", "maintained", "reaffirms", "reaffirmed", "keeps"}),
        frozenset({"withdraws", "withdrawn", "withdrew", "suspends", "suspended"}),
    ),
)

#: Words that turn a preceding phrase into a company name.  "First Look"
#: is a genre; "First Look Capital" is a fund.  Suppression keys on this
#: list rather than on capitalisation, because a Title Case headline
#: capitalises every word and would otherwise hide every genre phrase.
CORPORATE_DESIGNATORS = frozenset(
    {
        "advisors",
        "bank",
        "capital",
        "corp",
        "corporation",
        "group",
        "holdings",
        "inc",
        "labs",
        "llc",
        "ltd",
        "media",
        "networks",
        "partners",
        "plc",
        "securities",
        "studios",
        "systems",
        "technologies",
        "ventures",
    }
)

#: Delimiters a newsroom uses to separate a genre label from the headline.
_DELIMITERS = ":-|\u2013\u2014"

#: What *kind* of article a record is.  Two records can cover one event
#: and still be different stories: a rolling live blog is not the article
#: about one announcement inside it, a follow-up analysis is not the report
#: it analyses, an interview is not the release it discusses, and a
#: hands-on is not the launch it follows.  Merging across those loses
#: reporting that exists in only one of them, and lets a citation to one
#: resolve to the other.
#:
#: Every record has a type; the default is a plain ``report``.  A veto
#: fires on a marker present on one side and absent on the other, because
#: "plain report" is itself a type.
#:
#: Two match modes, because the evidence differs in strength:
#:
#: ``anywhere``
#:     A phrase that identifies the genre wherever it appears - "live
#:     updates", "hands on", "what to expect", "is said to".
#: ``anchored``
#:     A single word that is only a genre label in a headline *position*:
#:     at the start before a delimiter, immediately before a delimiter, or
#:     at the end.  "Tesla Earnings Preview" and "Nvidia Interview: ..."
#:     are genre labels; "analysts review results" and "the review board"
#:     are ordinary prose, and "live operations" is neither.
#:
#: Either way a match adjacent to a :data:`CORPORATE_DESIGNATORS` word is
#: discarded, which is what keeps "First Look Capital", "Interview Corp",
#: "Preview Networks" and "Recap Media" ordinary reports without needing
#: to guess at capitalisation.
DEFAULT_ARTICLE_TYPE = "report"

ARTICLE_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "live_blog",
        re.compile(
            r"\b(?:live (?:updates?|blog|coverage)|liveblog|as it happened)\b",
            re.IGNORECASE,
        ),
        "anywhere",
    ),
    (
        "analysis",
        re.compile(
            r"\b(?:what (?:it|this|that|the \w+) means?\b"
            r"|means? for (?:the |his |her |its )?\w+"
            r"|what to know\b"
            r"|key takeaways\b"
            r"|deep dive\b"
            r"|breaking down\b)",
            re.IGNORECASE,
        ),
        "anywhere",
    ),
    ("analysis", re.compile(r"\b(?:analysis|explainer)\b", re.IGNORECASE), "anchored"),
    (
        "interview",
        re.compile(
            r"\b(?:in an interview\b|interview with\b|q&a with\b"
            r"|in conversation with\b|speaks to\b|sits down with\b"
            r"|tells (?:cnbc|reuters|bloomberg|the ft)\b"
            r"|explains (?:the|how|why)\b)",
            re.IGNORECASE,
        ),
        "anywhere",
    ),
    ("interview", re.compile(r"\b(?:interview|q&a)\b", re.IGNORECASE), "anchored"),
    (
        "hands_on",
        re.compile(
            r"\b(?:hands[- ]on\b|first look at\b|we tried\b|road test\b)",
            re.IGNORECASE,
        ),
        "anywhere",
    ),
    (
        "hands_on",
        re.compile(r"\b(?:review|first look|hands[- ]on)\b", re.IGNORECASE),
        "anchored_end",
    ),
    (
        "rumour",
        re.compile(
            r"\b(?:is said to\b|are said to\b|reportedly\b|rumou?red\b"
            r"|sources say\b|people familiar\b|is expected to\b)",
            re.IGNORECASE,
        ),
        "anywhere",
    ),
    (
        "confirmation",
        re.compile(
            r"\b(?:officially confirms?\b|confirms? (?:the |earlier )?reports?\b"
            r"|confirms? plans to\b)",
            re.IGNORECASE,
        ),
        "anywhere",
    ),
    (
        "opinion",
        re.compile(r"\b(?:opinion|column|commentary)\b", re.IGNORECASE),
        "anchored",
    ),
    (
        "preview",
        re.compile(r"\b(?:what to expect\b|ahead of the\b)", re.IGNORECASE),
        "anywhere",
    ),
    ("preview", re.compile(r"\bpreview\b", re.IGNORECASE), "anchored"),
    (
        "recap",
        re.compile(r"\b(?:wrap[- ]up\b|round[- ]up\b|the week in\b)", re.IGNORECASE),
        "anywhere",
    ),
    ("recap", re.compile(r"\brecap\b", re.IGNORECASE), "anchored"),
)

_WORD_BEFORE = re.compile(r"([\w&'-]+)\W*$")
_WORD_AFTER = re.compile(r"^\W*([\w&'-]+)")


def _neighbour_words(text: str, start: int, end: int) -> tuple[str, str]:
    """Return the words immediately before and after a span, lower-cased."""

    before = _WORD_BEFORE.search(text[:start])
    after = _WORD_AFTER.search(text[end:])
    return (
        before.group(1).casefold() if before else "",
        after.group(1).casefold() if after else "",
    )


def _is_anchored(text: str, start: int, end: int, *, end_only: bool = False) -> bool:
    """True when a span sits in a headline position a genre label occupies."""

    tail = text[end:].strip()
    at_end = not tail or all(character in _DELIMITERS + ".," for character in tail)
    if end_only:
        return at_end
    head = text[:start].strip()
    at_start = not head or head[-1] in _DELIMITERS
    before_delimiter = bool(tail) and tail[0] in _DELIMITERS
    return at_start or at_end or before_delimiter


def article_types(text: str) -> tuple[str, ...]:
    """Return the sorted article-type markers a text carries.

    Empty means the default :data:`DEFAULT_ARTICLE_TYPE`; the comparison in
    :func:`combine` treats the empty tuple as its own value, so a marked
    record never merges with an unmarked one.
    """

    found: set[str] = set()
    for name, pattern, mode in ARTICLE_TYPE_PATTERNS:
        if name in found:
            continue
        for match in pattern.finditer(text):
            before, after = _neighbour_words(text, match.start(), match.end())
            if before in CORPORATE_DESIGNATORS or after in CORPORATE_DESIGNATORS:
                continue
            if mode == "anchored" and not _is_anchored(
                text, match.start(), match.end()
            ):
                continue
            if mode == "anchored_end" and not _is_anchored(
                text, match.start(), match.end(), end_only=True
            ):
                continue
            found.add(name)
            break
    return tuple(sorted(found))


#: A capitalised run of two or more words.  Used to protect number words
#: inside a name ("One Medical") from the magnitude signature; it is *not*
#: on its own evidence of an entity, because Title Case capitalises
#: everything.
_PROPER_RUN = re.compile(r"(?:\b[A-Z][\w&.'-]*(?:\s+|$)){2,}")

#: Capitalised words that begin a great many headlines but name nothing.
_NON_ENTITY_CAPITALS = frozenset(
    {
        "a",
        "an",
        "the",
        "what",
        "why",
        "how",
        "first",
        "second",
        "third",
        "fourth",
        "new",
        "q1",
        "q2",
        "q3",
        "q4",
        "update",
        "exclusive",
        "breaking",
        "refile",
        "wrapup",
    }
)


def _proper_noun_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return the character spans of capitalised multi-word runs."""

    return tuple((match.start(), match.end()) for match in _PROPER_RUN.finditer(text))


#: Outlet names are never story entities.  Reused from M2's versioned
#: publisher list rather than restated, so "New York Times reports X" and
#: "Wall Street Journal reports X" are one story reported twice, not two
#: organisations in conflict.
_OUTLET_NAMES = frozenset(
    name.casefold() for name in KNOWN_PUBLISHER_AFFIXES
) | frozenset({"ft", "the ft"})

#: Verbs and scaffolding that a Title Case headline capitalises but which
#: name nobody.  A backstop: extraction is context-anchored, not
#: capitalisation-anchored.
_HEADLINE_SCAFFOLDING = frozenset(
    {
        "adds",
        "announces",
        "beats",
        "buys",
        "company",
        "cuts",
        "delivers",
        "expands",
        "hits",
        "lifts",
        "misses",
        "opens",
        "posts",
        "raises",
        "reports",
        "results",
        "revenue",
        "rises",
        "says",
        "shares",
        "strong",
        "wins",
    }
)

#: Where an entity may be read from, and under which role.  Extraction is
#: **context-anchored**: a capitalised run only counts when one of these
#: patterns puts it in a named slot, or when it carries a corporate
#: designator of its own.  A Title Case headline capitalises every word, so
#: capitalisation alone is evidence of nothing.
#:
#: Roles are compared separately: an appointee conflicts with an appointee
#: and a counterparty with a counterparty, never across.
_ENTITY_CONTEXTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "person",
        re.compile(
            r"\b(?:appoints?|appointed|names?|named|promotes?|elects?|hires?)\s+"
            r"(?P<name>(?:[A-Z][\w.'-]*\s+){1,3}[A-Z][\w.'-]*)"
        ),
    ),
    (
        "person",
        re.compile(
            r"(?P<name>(?:[A-Z][\w.'-]*\s+){1,3}[A-Z][\w.'-]*)\s+"
            r"(?:is\s+)?(?:appointed|named|promoted|elected)"
        ),
    ),
    (
        "person",
        # The flag is scoped to the title, never to the name: a whole-pattern
        # re.IGNORECASE would make [A-Z] match lowercase too, and "Meta
        # finance chief explains the advertising acceleration" would name a
        # person called "explains the advertising acceleration" (P152).
        re.compile(
            r"\b(?i:ceo|cfo|coo|cto|cmo|chief executive|chief financial officer|"
            r"finance chief|chair(?:man|woman)?)\s+"
            r"(?P<name>(?:[A-Z][\w.'-]*\s+){1,3}[A-Z][\w.'-]*)"
        ),
    ),
    (
        "counterparty",
        re.compile(
            r"\b(?:partnership with|deal with|agreement with|acquires?|"
            r"acquisition of|buys|merges with|invests in)\s+"
            r"(?P<name>(?:[A-Z][\w.'-]*\s+){0,3}[A-Z][\w.'-]*)"
        ),
    ),
    (
        "analyst",
        re.compile(
            r"(?P<name>(?:[A-Z][\w.'-]*\s+){1,3}[A-Z][\w.'-]*)\s+"
            r"(?:raised|lifted|cut|trimmed|reiterated)\s+its"
        ),
    ),
    (
        "organisation",
        re.compile(
            r"(?P<name>(?:[A-Z][\w.'-]*\s+){0,3}[A-Z][\w.'-]*\s+"
            r"(?:Advisors|Bank|Capital|Corp|Corporation|Group|Holdings|Inc|"
            r"Labs|LLC|Ltd|Media|Motors|Networks|Partners|Plc|Securities|"
            r"Studios|Systems|Technologies|Ventures|Packaging))\b"
        ),
    ),
)


def _normalize_entity(raw: str) -> str | None:
    """Fold an extracted name, or reject it as not-a-name."""

    words = [word for word in raw.split() if word]

    def folded_of(parts: list[str]) -> str:
        return " ".join(word.casefold().strip(".,'") for word in parts)

    # Check the outlet list *before* trimming: "New York Times" loses its
    # "New" to the non-entity trim and would no longer match the publisher
    # list it belongs to.
    if folded_of(words) in _OUTLET_NAMES:
        return None
    while words and words[0].casefold().endswith("'s"):
        words.pop(0)
    while words and words[0].casefold().strip(".,'") in _NON_ENTITY_CAPITALS:
        words.pop(0)
    while words and words[-1].casefold().strip(".,'") in _NON_ENTITY_CAPITALS:
        words.pop()
    if not words:
        return None
    folded = " ".join(word.casefold().strip(".,'") for word in words)
    if folded in _OUTLET_NAMES:
        return None
    if all(word in _HEADLINE_SCAFFOLDING for word in folded.split()):
        return None
    return folded


def role_entities(text: str) -> tuple[tuple[str, str], ...]:
    """Return the sorted ``(role, name)`` pairs a text explicitly names.

    Context-anchored: a capitalised run only counts when an appointment
    verb, a counterparty preposition, a role word, an analyst-action verb,
    or a corporate designator of its own puts it in a named slot.  Ordinary
    Title Case ("Company Reports Strong Results") therefore names nobody,
    and a publisher in the headline is excluded outright.
    """

    found: set[tuple[str, str]] = set()
    for role, pattern in _ENTITY_CONTEXTS:
        for match in pattern.finditer(text):
            name = _normalize_entity(match.group("name"))
            if name and " " in name:
                found.add((role, name))
    return tuple(sorted(found))


def explicit_entities(text: str) -> tuple[str, ...]:
    """Return just the names :func:`role_entities` found, sorted."""

    return tuple(sorted({name for _, name in role_entities(text)}))


#: Nouns that make a headline about somebody other than the company the
#: ticker names.  Deliberately narrow: "partner", "rival", and "customer"
#: are modifiers far more often than they are subjects, and including them
#: would split legitimate rewrites.
_SUBJECT_SHIFT_TOKENS = frozenset(
    {
        "agencies",
        "agency",
        "contractor",
        "contractors",
        "distributor",
        "distributors",
        "reseller",
        "resellers",
        "subcontractor",
        "supplier",
        "suppliers",
        "vendor",
        "vendors",
    }
)

#: Words that carry no event identity, so a difference in them is not a
#: difference in the story.  Used only by the ``same_frame`` guard.
#: Includes the reporting verbs a newsroom swaps freely - "Company reports
#: strong results" and "Company posts strong results" are one event - so
#: the guard does not read a synonym as a substituted slot.
_FUNCTION_WORDS = frozenset(
    {
        "announce",
        "announces",
        "post",
        "posts",
        "report",
        "reports",
        "show",
        "shows",
        "a",
        "about",
        "after",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "new",
        "of",
        "on",
        "or",
        "over",
        "s",
        "said",
        "says",
        "that",
        "the",
        "their",
        "then",
        "there",
        "this",
        "to",
        "under",
        "up",
        "was",
        "were",
        "will",
        "with",
    }
)


def _text_of(title: str, description: str | None) -> str:
    """Return the displayable text both guards and overlap read."""

    parts = [part for part in (title, description or "") if part]
    return display_text(" ".join(parts))


#: Qualifiers that change what a quantity asserts.  "about 5" is not "5",
#: and "at least 5" is not "up to 5".  Absent means the record stated an
#: exact figure, which is itself a claim, so absence compares as ``exact``.
_APPROXIMATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("at_least", ("at least", "no fewer than", "or more")),
    ("at_most", ("up to", "no more than", "as many as")),
    ("more_than", ("more than", "over", "above", "north of")),
    ("less_than", ("less than", "fewer than", "under", "below")),
    ("about", ("about", "approximately", "roughly", "nearly", "around", "some")),
)

#: Trailing nouns that say *what* is being counted.  A closed list on
#: purpose: taking whatever word follows would make "495,000 vehicles" and
#: "495,000 cars" disagree, which is a paraphrase, not a contradiction.
#: A unit named on only one side stays unknown.
_COUNTED_UNITS = frozenset(
    {
        "chip",
        "chips",
        "dollar",
        "dollars",
        "euro",
        "euros",
        "job",
        "jobs",
        "share",
        "shares",
        "tonne",
        "tonnes",
        "unit",
        "units",
        "user",
        "users",
        "vehicle",
        "vehicles",
    }
)

#: Words that make a following digit part of a product name rather than a
#: quantity: "Model 3", "Series 7", "Gen 5".
_PRODUCT_QUALIFIERS = frozenset(
    {"model", "series", "generation", "gen", "mark", "type", "class", "version"}
)

_PERCENT_SUFFIX = re.compile(r"(%|bps|bp)$")
_RANGE_TOKEN = re.compile(r"\d[\d,.]*\s*-\s*[+\-]?[^\w\s]?\d")


class Quantity(NamedTuple):
    """One magnitude claim, decomposed so each part compares on its own.

    A flat string could only be equal or unequal.  Separating the parts is
    what lets a *unit named on one side only* stay unknown while a unit
    named on both sides and differing is a contradiction.
    """

    approximation: str
    sign: str
    value: str
    is_range: bool
    magnitude: str
    currency: str
    percent_kind: str
    unit: str

    def render(self) -> str:
        """A stable string form, for fingerprints and reports."""

        return "|".join(
            (
                self.approximation,
                self.sign,
                self.value,
                "range" if self.is_range else "point",
                self.magnitude,
                self.currency,
                self.percent_kind,
                self.unit,
            )
        )


#: Fields compared only when *both* sides state them.  Everything else is
#: an assertion whose absence is itself a claim.
_UNKNOWN_IF_ABSENT = ("unit",)


def _approximation_before(tokens: Sequence[str], index: int) -> str:
    """Return the qualifier immediately preceding a number, or ``exact``.

    Matched on **token boundaries**, never as a substring: "over" lives
    inside "handovers", and a substring match there turned an exact
    delivery figure into "more than", which refused a real rewrite.
    """

    window = list(tokens[max(0, index - 3) : index])
    for name, phrases in _APPROXIMATIONS:
        for phrase in phrases:
            words = phrase.split()
            span = len(words)
            if any(
                window[offset : offset + span] == words
                for offset in range(len(window) - span + 1)
            ):
                return name
    return "exact"


def _decompose(token: str) -> tuple[str, str, str, bool]:
    """Split a numeric token into sign, currency, value and range-ness."""

    sign = ""
    body = token
    if body and body[0] in "+-":
        sign, body = body[0], body[1:]
    currency = ""
    if body and not body[0].isdigit():
        currency, body = body[0], body[1:]
    is_range = bool(_RANGE_TOKEN.search(token))
    return sign, currency, body, is_range


def numeric_signature(
    tokens: tuple[str, ...], protected: frozenset[str] = frozenset()
) -> tuple[Quantity, ...]:
    """Return the ordered, structured magnitude claims of a token sequence.

    Each claim keeps its approximation qualifier, sign, value, range-ness,
    magnitude word, currency, percent/basis-point kind and counted unit as
    separate fields, so "about 5 units" and "5 units", "5 million units"
    and "5 million dollars", "5%" and "5 basis points" are all different
    claims while "eleven units" and "11 units" are the same one.

    Counts spelled out in words normalize to their digit value.  A number
    word inside a capitalised name (One Medical, Formula One) or after a
    product qualifier (Model 3) is not a quantity; ``protected`` carries
    the first case and :data:`_PRODUCT_QUALIFIERS` the second.
    """

    claims: list[Quantity] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        previous = tokens[index - 1] if index else ""
        if _NUMERIC_TOKEN.match(token):
            if previous in _PRODUCT_QUALIFIERS:
                index += 1
                continue
            sign, currency, body, is_range = _decompose(token)
            percent = ""
            match = _PERCENT_SUFFIX.search(body)
            if match:
                percent = "%" if match.group(1) == "%" else "bps"
                body = body[: match.start()]
            consumed = 1
            nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
            magnitude = ""
            if nxt in _MAGNITUDE_WORDS:
                magnitude = nxt
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            if not percent and nxt in {"basis", "percentage"}:
                after = (
                    tokens[index + consumed + 1]
                    if index + consumed + 1 < len(tokens)
                    else ""
                )
                if after in {"points", "point"}:
                    percent = "bps" if nxt == "basis" else "%"
                    consumed += 2
                    nxt = (
                        tokens[index + consumed]
                        if index + consumed < len(tokens)
                        else ""
                    )
            if not percent and nxt == "percent":
                percent = "%"
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            unit = nxt if nxt in _COUNTED_UNITS else ""
            claims.append(
                Quantity(
                    approximation=_approximation_before(tokens, index),
                    sign=sign,
                    value=body,
                    is_range=is_range,
                    magnitude=magnitude,
                    currency=currency,
                    percent_kind=percent,
                    unit=unit,
                )
            )
            index += consumed
            continue
        if token in _CARDINAL_VALUES and token not in protected:
            value = _CARDINAL_VALUES[token]
            consumed = 1
            nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
            if token in _TENS_WORDS and nxt in _UNIT_CARDINALS and nxt not in protected:
                value += _CARDINAL_VALUES[nxt]
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            magnitude = ""
            if nxt == "hundred":
                value *= 100
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            if nxt in _MAGNITUDE_WORDS:
                magnitude = nxt
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            claims.append(
                Quantity(
                    approximation=_approximation_before(tokens, index),
                    sign="",
                    value=str(value),
                    is_range=False,
                    magnitude=magnitude,
                    currency="",
                    percent_kind="",
                    unit=nxt if nxt in _COUNTED_UNITS else "",
                )
            )
            index += consumed
            continue
        index += 1
    return tuple(claims)


def quantities_conflict(
    left: tuple[Quantity, ...], right: tuple[Quantity, ...]
) -> bool:
    """True when two ordered quantity sequences make incompatible claims.

    Missing information does not conflict: a record that names no unit does
    not contradict one that does.  Everything else - approximation, sign,
    value, range-ness, magnitude, currency, percent kind - is an assertion,
    and a difference in any of them is a contradiction.
    """

    if not left or not right:
        return False
    if len(left) != len(right):
        return True
    for first, second in zip(left, right):
        for field in Quantity._fields:
            one, other = getattr(first, field), getattr(second, field)
            if field in _UNKNOWN_IF_ABSENT and (not one or not other):
                continue
            if one != other:
                return True
    return False


def temporal_markers(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return the sorted reporting-period markers of a token sequence."""

    markers: set[str] = set()
    for index, token in enumerate(tokens):
        if token in _QUARTER_TOKENS:
            markers.add(_QUARTER_TOKENS[token])
        elif token in _MONTHS:
            markers.add(_MONTHS[token])
        elif _ISO_DATE.match(token):
            markers.add(f"d{token}")
        elif _YEAR.match(token):
            markers.add(f"y{token}")
        elif token in _QUARTER_WORDS and index + 1 < len(tokens):
            if tokens[index + 1] == "quarter":
                markers.add(_QUARTER_WORDS[token])
    return tuple(sorted(markers))


def roles(text: str) -> tuple[str, ...]:
    """Return the sorted named roles a text mentions."""

    lowered = text.casefold()
    found: set[str] = set()
    for key, pattern in _ROLE_PATTERNS:
        if pattern.search(lowered):
            found.add(key)
    # "chief financial officer" also matches nothing else; a bare "chair"
    # inside "chairman" is handled by the alternation above.
    return tuple(sorted(found))


def contrasts(tokens: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Return the sorted ``(family, polarity)`` claims a text makes."""

    present = set(tokens)
    found: set[tuple[str, str]] = set()
    for family, positive, negative in _CONTRAST_GROUPS:
        if present & positive:
            found.add((family, "positive"))
        if present & negative:
            found.add((family, "negative"))
    return tuple(sorted(found))


#: Surface forms folded onto one lemma, so "supplier" and "suppliers" are
#: the same subject rather than two.
_SUBJECT_LEMMAS = {
    "agencies": "agency",
    "agency": "agency",
    "contractor": "contractor",
    "contractors": "contractor",
    "distributor": "distributor",
    "distributors": "distributor",
    "reseller": "reseller",
    "resellers": "reseller",
    "subcontractor": "contractor",
    "supplier": "supplier",
    "suppliers": "supplier",
    "vendor": "vendor",
    "vendors": "vendor",
}


#: A trailing clause that attributes the headline to somebody rather than
#: making the headline about them: ", Apple suppliers say", ", sources
#: said", ", according to two dealers".  Stripped before subjects are read,
#: because an attribution is not a subject - reading one as a subject cost
#: a real rewrite (P068).
_ATTRIBUTION_CLAUSE = re.compile(
    r",\s*[^,]*\b(?:says?|said|sources|according to|tells?|told)\b[^,]*$",
    re.IGNORECASE,
)


def strip_attribution_clause(headline: str) -> str:
    """Return the headline without a trailing attribution clause."""

    return _ATTRIBUTION_CLAUSE.sub("", headline).strip()


def subject_markers(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return the sorted third-party subjects a *headline* names.

    Lemma-folded, so a plural and a singular are one subject.  The caller
    passes headline tokens only: a standfirst that merely attributes a
    quote ("suppliers told the paper") is not the article being about a
    supplier, and treating it as one cost a real rewrite.
    """

    return tuple(
        sorted({_SUBJECT_LEMMAS[token] for token in tokens if token in _SUBJECT_LEMMAS})
    )


@dataclass(frozen=True)
class StoryEvidence:
    """What a *set* of stories collectively asserts.

    Every field holds the known values across the set, so combining two
    summaries is exactly equivalent to comparing every story on one side
    against every story on the other — at constant cost.  That is what
    makes prospective-cluster checking affordable, and it is why a story
    carrying no numbers cannot bridge two that disagree about them.
    """

    numeric: frozenset[tuple[str, ...]]
    temporal: frozenset[tuple[str, ...]]
    roles: frozenset[tuple[str, ...]]
    contrasts: frozenset[tuple[str, str]]
    negations: frozenset[bool]
    #: Third-party subjects named in the *headline*.
    subjects: frozenset[tuple[str, ...]]
    article_types: frozenset[tuple[str, ...]]
    #: Explicit ``(role, name)`` pairs, from headline and standfirst.
    entities: frozenset[tuple[tuple[str, str], ...]]
    #: Token sets of each member, kept for the ``same_frame`` guard, which
    #: is pairwise by nature: it asks whether *these two* share a frame.
    token_sets: frozenset[frozenset[str]]


def summarize(title: str, description: str | None = None) -> StoryEvidence:
    """Return the evidence one story carries.

    Scope matters and is deliberate.  Magnitudes, periods, roles,
    polarity, negation and named entities are read from the **headline and
    the standfirst**, because a contradiction anywhere in the record is a
    contradiction.  The frame and the subject are read from the **headline
    alone**, because a frame is a headline template and a subject is a
    headline subject.
    """

    combined = _text_of(title, description)
    headline = display_text(title or "")
    tokens = tokenize(combined)
    if not tokens:
        return StoryEvidence(
            numeric=frozenset(),
            temporal=frozenset(),
            roles=frozenset(),
            contrasts=frozenset(),
            negations=frozenset(),
            subjects=frozenset(),
            article_types=frozenset({()}),
            entities=frozenset(),
            token_sets=frozenset(),
        )
    protected = frozenset(
        word.casefold()
        for start, end in _proper_noun_spans(combined)
        for word in combined[start:end].split()
    )
    numeric = numeric_signature(tokens, protected)
    temporal = temporal_markers(tokens)
    role_keys = roles(combined)
    headline_tokens = tokenize(headline)
    subjects = subject_markers(tokenize(strip_attribution_clause(headline)))
    names = role_entities(combined)
    return StoryEvidence(
        numeric=frozenset({numeric}) if numeric else frozenset(),
        temporal=frozenset({temporal}) if temporal else frozenset(),
        roles=frozenset({role_keys}) if role_keys else frozenset(),
        contrasts=frozenset(contrasts(tokens)),
        negations=frozenset({bool(set(tokens) & NEGATION_TOKENS)}),
        subjects=frozenset({subjects}),
        article_types=frozenset({article_types(combined)}),
        entities=frozenset({names}) if names else frozenset(),
        token_sets=frozenset({frozenset(headline_tokens)}),
    )


#: Fields compared as "at most one known value".  Split around the
#: polarity check because the *order* of all of these is load-bearing: a
#: recorded veto should name the most specific true objection, and several
#: guards fire together on a well-formed hard negative.
#:
#: Temporal precedes numeric because a year or a quarter is also a number,
#: and "these describe different periods" explains a refusal better than
#: "the numbers differ".  Polarity precedes negation because several of the
#: opposing verbs ("rejects", "denies", "refused") are negation tokens too,
#: and "these take opposite positions" is the better answer.
_BEFORE_POLARITY: tuple[tuple[str, str], ...] = (
    ("article_type", "article_types"),
    ("temporal_disagreement", "temporal"),
    ("role_disagreement", "roles"),
    ("subject_shift", "subjects"),
)
_AFTER_POLARITY: tuple[tuple[str, str], ...] = (("negation", "negations"),)


def _entity_conflict(
    left: frozenset[tuple[tuple[str, str], ...]],
    right: frozenset[tuple[tuple[str, str], ...]],
) -> bool:
    """True when both sides name entities and each names one the other does not.

    Missing entity evidence is *unknown*, not contradictory: a record that
    names nobody never blocks a merge.  A shared entity plus an extra one
    on a single side is elaboration, not disagreement.  Only a genuine
    substitution - Alice Smith against Bob Jones, Northfield Securities
    against Calder Bank Markets - is refused.
    """

    if not left or not right:
        return False
    left_pairs = {entry for group in left for entry in group}
    right_pairs = {entry for group in right for entry in group}
    shared_roles = {role for role, _ in left_pairs} & {role for role, _ in right_pairs}
    for role in sorted(shared_roles):
        left_names = {name for slot, name in left_pairs if slot == role}
        right_names = {name for slot, name in right_pairs if slot == role}
        if (left_names - right_names) and (right_names - left_names):
            return True
    return False


_VALUE_CHECKS = _BEFORE_POLARITY + _AFTER_POLARITY

#: Every reason :func:`combine` can return, in the order it tries them.
VETO_REASONS = (
    tuple(reason for reason, _ in _BEFORE_POLARITY)
    + ("numeric_disagreement", "contrast_polarity")
    + tuple(reason for reason, _ in _AFTER_POLARITY)
    + ("entity_conflict", "same_frame_different_event")
)


def _same_frame(left: frozenset[str], right: frozenset[str], overlap: float) -> bool:
    """True when two token sets share a frame but swap its content slot.

    Requires distinguishing content words on *both* sides.  A title that is
    a strict elaboration of another ("Apple opens a store" against "Apple
    opens a store in Riyadh") adds detail rather than substituting an
    event, so it is not caught here.
    """

    if not left or not right:
        return False
    union = left | right
    if len(left & right) / len(union) < overlap:
        return False
    left_only = (left - right) - _FUNCTION_WORDS
    right_only = (right - left) - _FUNCTION_WORDS
    return bool(left_only) and bool(right_only)


def combine(
    left: StoryEvidence, right: StoryEvidence, *, frame_overlap: float
) -> tuple[StoryEvidence | None, str | None]:
    """Merge two summaries, or report why the sets contradict each other."""

    merged = StoryEvidence(
        numeric=left.numeric | right.numeric,
        temporal=left.temporal | right.temporal,
        roles=left.roles | right.roles,
        contrasts=left.contrasts | right.contrasts,
        negations=left.negations | right.negations,
        subjects=left.subjects | right.subjects,
        article_types=left.article_types | right.article_types,
        entities=left.entities | right.entities,
        token_sets=left.token_sets | right.token_sets,
    )
    for reason, field in _BEFORE_POLARITY:
        if len(getattr(merged, field)) > 1:
            return None, reason
    for first in left.numeric:
        for second in right.numeric:
            if quantities_conflict(first, second):
                return None, "numeric_disagreement"
    families = [family for family, _ in merged.contrasts]
    if len(families) != len(set(families)):
        return None, "contrast_polarity"
    for reason, field in _AFTER_POLARITY:
        if len(getattr(merged, field)) > 1:
            return None, reason
    if _entity_conflict(left.entities, right.entities):
        return None, "entity_conflict"
    for left_tokens in left.token_sets:
        for right_tokens in right.token_sets:
            if _same_frame(left_tokens, right_tokens, frame_overlap):
                return None, "same_frame_different_event"
    return merged, None


def contradiction(
    left: StoryEvidence, right: StoryEvidence, *, frame_overlap: float
) -> str | None:
    """Return why two summaries may not merge, or ``None`` when they may."""

    return combine(left, right, frame_overlap=frame_overlap)[1]


#: Every behaviour-changing policy in this module, named once.  The
#: fingerprint walks this map, so adding a rule here is all it takes for it
#: to invalidate cached M3 output - no manual version bump to forget.
POLICY_COMPONENTS: dict[str, "object"] = {}


def _register(name: str, value: object) -> None:
    POLICY_COMPONENTS[name] = value


def _component_values() -> dict[str, str]:
    """Render every registered component as a stable string."""

    return {
        "evidence_version": EVIDENCE_POLICY_VERSION,
        "article_type_version": ARTICLE_TYPE_POLICY_VERSION,
        "cardinal_version": CARDINAL_POLICY_VERSION,
        "entity_version": ENTITY_POLICY_VERSION,
        "tokenizer": structural_policy_fingerprint(),
        "numeric_token": _NUMERIC_TOKEN.pattern,
        "bound_words": ",".join(sorted(_BOUND_WORDS)),
        "cardinal_values": ",".join(
            f"{word}={value}" for word, value in sorted(_CARDINAL_VALUES.items())
        ),
        "tens_words": ",".join(sorted(_TENS_WORDS)),
        "unit_cardinals": ",".join(sorted(_UNIT_CARDINALS)),
        "months": ",".join(sorted(_MONTHS)),
        "quarters": ",".join(sorted(_QUARTER_WORDS)),
        "iso_date": _ISO_DATE.pattern,
        "year": _YEAR.pattern,
        "roles": ";".join(f"{key}:{p.pattern}" for key, p in _ROLE_PATTERNS),
        "contrasts": ";".join(
            f"{family}:{','.join(sorted(positive))}|{','.join(sorted(negative))}"
            for family, positive, negative in _CONTRAST_GROUPS
        ),
        "negation_tokens": ",".join(sorted(NEGATION_TOKENS)),
        "subject_lemmas": ",".join(
            f"{k}={v}" for k, v in sorted(_SUBJECT_LEMMAS.items())
        ),
        "article_type_patterns": ";".join(
            f"{name}:{pattern.pattern}" for name, pattern, _ in ARTICLE_TYPE_PATTERNS
        ),
        "article_type_modes": ",".join(
            f"{name}:{mode}" for name, _, mode in ARTICLE_TYPE_PATTERNS
        ),
        "delimiters": _DELIMITERS,
        "corporate_designators": ",".join(sorted(CORPORATE_DESIGNATORS)),
        "entity_contexts": ";".join(
            f"{role}:{pattern.pattern}" for role, pattern in _ENTITY_CONTEXTS
        ),
        "headline_scaffolding": ",".join(sorted(_HEADLINE_SCAFFOLDING)),
        "outlet_names": ",".join(sorted(_OUTLET_NAMES)),
        "proper_run": _PROPER_RUN.pattern,
        "non_entity_capitals": ",".join(sorted(_NON_ENTITY_CAPITALS)),
        "function_words": ",".join(sorted(_FUNCTION_WORDS)),
        "guard_order": ",".join(VETO_REASONS),
        "same_frame_scope": "headline_only",
        "subject_scope": "headline_only_minus_attribution",
        "attribution_clause": _ATTRIBUTION_CLAUSE.pattern,
    }


def policy_components() -> dict[str, str]:
    """Return every registered policy component and its rendered value."""

    return dict(sorted(_component_values().items()))


def policy_fingerprint() -> str:
    """Return a digest of every static policy this module applies.

    Folded into :meth:`nlp.semdedup.config.SemanticDedupConfig.fingerprint`,
    so editing any lexicon, pattern, guard order or scope changes the
    configuration fingerprint automatically.  M2's structural fingerprint is
    one of the components, because M3 reads its tokens through M2's
    tokenizer.
    """

    payload = "|".join(
        f"{name}={value}" for name, value in sorted(_component_values().items())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
import hashlib
import re

from nlp.dedup.compatibility import NEGATION_TOKENS
from nlp.dedup.structural import policy_fingerprint as structural_policy_fingerprint
from nlp.dedup.structural import tokenize
from nlp.dedup.text import display_text

#: Bumped whenever a guard, a lexicon, or the comparison changes.
EVIDENCE_POLICY_VERSION = "m3.evidence.v3"

#: Bumped when the article-type classifier changes shape.
ARTICLE_TYPE_POLICY_VERSION = "m3.article_type.v2"
#: Bumped when cardinal normalization changes.
CARDINAL_POLICY_VERSION = "m3.cardinal.v1"
#: Bumped when explicit-entity extraction changes.
ENTITY_POLICY_VERSION = "m3.entity.v1"

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

#: What *kind* of article a record is.  Two records can cover one event
#: and still be different stories: a rolling live blog is not the article
#: about one announcement inside it, a follow-up analysis is not the report
#: it analyses, a confirmation is not the rumour it confirms, an interview
#: is not the release it discusses, and a hands-on is not the launch it
#: follows.  Merging across those loses reporting that exists in only one
#: of them, and lets a citation to one resolve to the other.
#:
#: Every record has a type; the default is a plain ``report``.  A veto
#: therefore fires on a marker present on one side and absent on the other,
#: not only on two different markers, because "plain report" is itself a
#: type.
#:
#: Classification is deliberately hard to trigger.  Each marker is an
#: anchored regular expression with word boundaries, not a substring, and a
#: match is discarded when it sits inside a capitalised proper-noun run, so
#: "First Look Capital", "Interview Corp", "Preview Networks" and
#: "Recap Media" are ordinary reports.  Bare verbs are not markers at all:
#: "confirms", "review", "says" and "live" appear in ordinary copy
#: ("Company confirms earnings date", "analysts review results", "live
#: operations"), so a genre is only claimed when the phrase identifies the
#: genre on its own.  Uncertainty yields ``report``, never a veto.
DEFAULT_ARTICLE_TYPE = "report"

ARTICLE_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "live_blog",
        re.compile(
            r"\b(?:live (?:updates?|blog|coverage)|liveblog|as it happened)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "analysis",
        re.compile(
            r"\b(?:what (?:it|this|that|the \w+) means?\b"
            r"|means? for (?:the |his |her |its )?\w+"
            r"|explainer\b"
            r"|what to know\b"
            r"|key takeaways\b"
            r"|deep dive\b"
            r"|breaking down\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "interview",
        re.compile(
            r"\b(?:in an interview\b|interview with\b|q&a with\b"
            r"|in conversation with\b|speaks to\b|sits down with\b"
            r"|tells (?:cnbc|reuters|bloomberg|the ft)\b"
            r"|explains (?:the|how|why)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "hands_on",
        re.compile(
            r"\b(?:hands[- ]on\b|first look at\b|we tried\b|road test\b"
            r"|hands[- ]on review\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "rumour",
        re.compile(
            r"\b(?:is said to\b|are said to\b|reportedly\b|rumou?red\b"
            r"|sources say\b|people familiar\b|is expected to\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "confirmation",
        re.compile(
            r"\b(?:officially confirms?\b|confirms? (?:the |earlier )?reports?\b"
            r"|confirms? plans to\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "opinion",
        re.compile(r"^\s*(?:opinion|column|commentary)\s*[:\-]", re.IGNORECASE),
    ),
    (
        "preview",
        re.compile(r"\b(?:what to expect\b|ahead of the\b)", re.IGNORECASE),
    ),
    (
        "recap",
        re.compile(r"\b(?:wrap[- ]up\b|round[- ]up\b|the week in\b)", re.IGNORECASE),
    ),
)

#: A capitalised run of two or more words: the shape of an organisation or
#: a person's name.  Used to keep genre markers out of entity names and to
#: keep entity words out of the magnitude signature.
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


def _inside_span(spans: tuple[tuple[int, int], ...], start: int, end: int) -> bool:
    return any(begin <= start and end <= finish for begin, finish in spans)


def article_types(text: str) -> tuple[str, ...]:
    """Return the sorted article-type markers a text carries.

    Empty means the default :data:`DEFAULT_ARTICLE_TYPE`; the comparison in
    :func:`combine` treats the empty tuple as its own value, so a marked
    record never merges with an unmarked one.

    A match inside a capitalised proper-noun run is discarded: "First Look
    Capital" names a fund, not a genre.
    """

    spans = _proper_noun_spans(text)
    found: set[str] = set()
    for name, pattern in ARTICLE_TYPE_PATTERNS:
        for match in pattern.finditer(text):
            if not _inside_span(spans, match.start(), match.end()):
                found.add(name)
                break
    return tuple(sorted(found))


def explicit_entities(text: str) -> tuple[str, ...]:
    """Return the sorted explicit named entities a text carries.

    Deliberately narrow: only a **capitalised run of two or more words**
    counts, because a single capitalised token in a headline is far more
    often ordinary headline casing than a name.  That keeps "Alice Smith"
    and "Wolfsberg Motors" in and leaves "Acme acquires Beta" out; the
    limitation is documented rather than papered over with heuristics.

    A run that is entirely non-entity capitals ("The New") is dropped, a
    leading non-entity word is trimmed so "The Acme Group" yields ``acme
    group``, and a leading possessive is dropped so "Apple's App Store" and
    "App Store" are the same entity rather than two.
    """

    entities: set[str] = set()
    for match in _PROPER_RUN.finditer(text):
        words = [word for word in match.group(0).split() if word]
        # A leading possessive is a *relation*, not part of the name:
        # "Apple's App Store" and "App Store" are one entity, and
        # "Tesla's Berlin" is Tesla plus a place word rather than a name.
        # Keeping it made the guard reject two real rewrites.
        while words and words[0].casefold().endswith("'s"):
            words.pop(0)
        while words and words[0].casefold().strip(".,'") in _NON_ENTITY_CAPITALS:
            words.pop(0)
        while words and words[-1].casefold().strip(".,'") in _NON_ENTITY_CAPITALS:
            words.pop()
        if len(words) < 2:
            continue
        entities.add(" ".join(word.casefold().strip(".,'") for word in words))
    return tuple(sorted(entities))


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
_FUNCTION_WORDS = frozenset(
    {
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


def numeric_signature(
    tokens: tuple[str, ...], protected: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Return the ordered magnitude claims of a token sequence.

    Each numeric token is bound to the following token only when that token
    changes what the number means (a magnitude or a unit).  Order is
    preserved and never sorted, so "up 5% to $10" cannot equal "up 10% to
    $5".

    Counts spelled out in words are **normalized to their digit form**, so
    "eleven" and "11", "twenty-one" and "21", "a dozen" and "12", "one
    hundred" and "100", and "five million" and "5 million" are the same
    claim.  What is *not* normalized: ordinals ("first quarter" stays a
    period marker), ranges, approximation markers, currency symbols and
    units, all of which stay attached to the token they qualify.

    ``protected`` holds tokens that sit inside a capitalised proper-noun
    run.  A number word there is part of a name - One Medical, Big Four -
    and is left alone rather than turned into a quantity.
    """

    claims: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _NUMERIC_TOKEN.match(token):
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            claims.append(
                f"{token} {following}" if following in _BOUND_WORDS else token
            )
            index += 1
            continue
        if token in _CARDINAL_VALUES and token not in protected:
            value = _CARDINAL_VALUES[token]
            consumed = 1
            nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
            if token in _TENS_WORDS and nxt in _UNIT_CARDINALS and nxt not in protected:
                value += _CARDINAL_VALUES[nxt]
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            if nxt == "hundred":
                value *= 100
                consumed += 1
                nxt = tokens[index + consumed] if index + consumed < len(tokens) else ""
            if nxt in _BOUND_WORDS:
                claims.append(f"{value} {nxt}")
                consumed += 1
            else:
                claims.append(str(value))
            index += consumed
            continue
        index += 1
    return tuple(claims)


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
    #: Explicit multi-word named entities, from headline and standfirst.
    entities: frozenset[tuple[str, ...]]
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
    names = explicit_entities(combined)
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
    ("numeric_disagreement", "numeric"),
    ("role_disagreement", "roles"),
    ("subject_shift", "subjects"),
)
_AFTER_POLARITY: tuple[tuple[str, str], ...] = (("negation", "negations"),)


def _entity_conflict(
    left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]
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
    left_names = {name for group in left for name in group}
    right_names = {name for group in right for name in group}
    if not left_names or not right_names:
        return False
    return bool(left_names - right_names) and bool(right_names - left_names)


_VALUE_CHECKS = _BEFORE_POLARITY + _AFTER_POLARITY

#: Every reason :func:`combine` can return, in the order it tries them.
VETO_REASONS = (
    tuple(reason for reason, _ in _BEFORE_POLARITY)
    + ("contrast_polarity",)
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
            f"{name}:{pattern.pattern}" for name, pattern in ARTICLE_TYPE_PATTERNS
        ),
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

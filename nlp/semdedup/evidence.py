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
``subject_shift``    is one about a supplier, reseller, or agency rather
                     than the company itself?
``same_frame``       do they share most of their wording and differ only in
                     the slot that carries the event?

The last one is the load-bearing guard for "same template, different
event".  When two headlines overlap heavily, the tokens they do *not* share
are the story, so a substitution in that slot is a different story however
close the vectors are.  A real rewrite has the opposite shape: low lexical
overlap, high semantic similarity, and it never triggers this guard.

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
EVIDENCE_POLICY_VERSION = "m3.evidence.v1"

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


def numeric_signature(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return the ordered magnitude claims of a token sequence.

    Each numeric token is bound to the following token only when that token
    changes what the number means (a magnitude or a unit).  Order is
    preserved and never sorted, so "up 5% to $10" cannot equal "up 10% to
    $5".
    """

    claims: list[str] = []
    for index, token in enumerate(tokens):
        if not _NUMERIC_TOKEN.match(token):
            continue
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        claims.append(f"{token} {following}" if following in _BOUND_WORDS else token)
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


def subject_markers(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """Return the sorted third-party subject nouns a text mentions."""

    return tuple(sorted(set(tokens) & _SUBJECT_SHIFT_TOKENS))


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
    subjects: frozenset[tuple[str, ...]]
    #: Token sets of each member, kept for the ``same_frame`` guard, which
    #: is pairwise by nature: it asks whether *these two* share a frame.
    token_sets: frozenset[frozenset[str]]


def summarize(title: str, description: str | None = None) -> StoryEvidence:
    """Return the evidence one story carries."""

    text = _text_of(title, description)
    tokens = tokenize(text)
    if not tokens:
        return StoryEvidence(
            numeric=frozenset(),
            temporal=frozenset(),
            roles=frozenset(),
            contrasts=frozenset(),
            negations=frozenset(),
            subjects=frozenset(),
            token_sets=frozenset(),
        )
    numeric = numeric_signature(tokens)
    temporal = temporal_markers(tokens)
    role_keys = roles(text)
    subjects = subject_markers(tokens)
    return StoryEvidence(
        numeric=frozenset({numeric}) if numeric else frozenset(),
        temporal=frozenset({temporal}) if temporal else frozenset(),
        roles=frozenset({role_keys}) if role_keys else frozenset(),
        contrasts=frozenset(contrasts(tokens)),
        negations=frozenset({bool(set(tokens) & NEGATION_TOKENS)}),
        subjects=frozenset({subjects}),
        token_sets=frozenset({frozenset(tokens)}),
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
    ("temporal_disagreement", "temporal"),
    ("numeric_disagreement", "numeric"),
    ("role_disagreement", "roles"),
    ("subject_shift", "subjects"),
)
_AFTER_POLARITY: tuple[tuple[str, str], ...] = (("negation", "negations"),)
_VALUE_CHECKS = _BEFORE_POLARITY + _AFTER_POLARITY

#: Every reason :func:`combine` can return, in the order it tries them.
VETO_REASONS = (
    tuple(reason for reason, _ in _BEFORE_POLARITY)
    + ("contrast_polarity",)
    + tuple(reason for reason, _ in _AFTER_POLARITY)
    + ("same_frame_different_event",)
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


def policy_fingerprint() -> str:
    """Return a digest of every static policy this module applies.

    Folds in M2's structural fingerprint as well: M3 reads its tokens
    through M2's tokenizer, so a tokenizer change moves M3's guards too and
    must invalidate M3's cached output.
    """

    payload = "|".join(
        (
            EVIDENCE_POLICY_VERSION,
            structural_policy_fingerprint(),
            _NUMERIC_TOKEN.pattern,
            ",".join(sorted(_BOUND_WORDS)),
            ",".join(sorted(_MONTHS)),
            ",".join(sorted(_QUARTER_WORDS)),
            ",".join(f"{key}:{pattern.pattern}" for key, pattern in _ROLE_PATTERNS),
            ";".join(
                f"{family}:{','.join(sorted(positive))}|{','.join(sorted(negative))}"
                for family, positive, negative in _CONTRAST_GROUPS
            ),
            ",".join(sorted(_SUBJECT_SHIFT_TOKENS)),
            ",".join(sorted(_FUNCTION_WORDS)),
            ",".join(sorted(NEGATION_TOKENS)),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

"""M5's theme-compatibility contract: what a theme may not contain.

M3 answers "are these the *same story*".  M5 asks a different and much
coarser question — "does a reader recognise these as one narrative" — and
the two need different answers.  A theme called "today's earnings coverage"
legitimately holds stories about different quarters, different magnitudes,
different people, and different article types; applying M3's guard set here
would shred every theme into singletons.  Exactly one family transfers:
**a theme may not assert both sides of the same claim.**

**Why this policy lives in M5.**  M3's guard lexicons are private to
:mod:`nlp.semdedup.evidence` and are versioned against M3's question, not
this one.  Importing them coupled every theme to a guard change that has
nothing to do with themes, and read a private module besides.  So this is a
*narrow public theme-compatibility contract* that M5 owns, states, versions
and fingerprints:

* it reads only public M5 input — the canonical title and standfirst on
  :class:`~nlp.themes.models.ThemeStory`, which the bridge projects from
  public M3 output fields;
* it names four opposing-claim families and nothing else;
* it is deliberately *narrower* than M3's guard set, and the families it
  does not veto on are listed in :data:`PERMITTED_DIFFERENCES` with the
  reason, so "M5 allows this" is a decision on the record rather than an
  omission.

**Integration requirement.**  M3 does not currently expose per-story
compatibility evidence on its public result, so M5 derives polarity from
text.  When #57/#68 land and M3 publishes a public evidence projection, this
module should consume it instead of re-deriving from the headline; the
contract below is the shape that projection needs to satisfy.

**The check is cluster-wide.**  :func:`incompatible_members` is handed a
whole prospective theme and compares every member against the theme's
combined claims, exactly as M2 and M3 apply compatibility to a whole
prospective cluster rather than to pair endpoints.  A story that opposes any
member opposes the theme.
"""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from nlp.dedup.structural import tokenize
from nlp.dedup.text import display_text

from .models import ThemeStory

#: Bumped when a family, a lexicon, or the comparison changes.
COMPATIBILITY_POLICY_VERSION = "m5.compatibility.v1"

#: The claim families a *theme* may not hold both sides of, as
#: ``(family, positive terms, negative terms)``.  Small on purpose: each
#: entry has to survive the question "would a reader call a group holding
#: both of these one story about one thing?"
OPPOSING_CLAIM_FAMILIES: tuple[tuple[str, frozenset[str], frozenset[str]], ...] = (
    (
        # Guidance, prices, output, headcount: which way the number moved.
        "direction",
        frozenset(
            {
                "raises",
                "raised",
                "lifts",
                "lifted",
                "boosts",
                "boosted",
                "increases",
                "increased",
                "climbs",
                "climbed",
                "jumps",
                "jumped",
                "rises",
                "rose",
                "grows",
                "grew",
                "expands",
                "expanded",
                "adds",
                "added",
                "higher",
                "stronger",
                "upgrade",
                "upgrades",
                "upgraded",
                "raise",
                "lift",
                "boost",
                "increase",
                "climb",
                "jump",
                "rise",
                "grow",
                "expand",
                "add",
                "upgrade",
            }
        ),
        frozenset(
            {
                "cuts",
                "cut",
                "lowers",
                "lowered",
                "reduces",
                "reduced",
                "trims",
                "trimmed",
                "slashes",
                "slashed",
                "falls",
                "fell",
                "drops",
                "dropped",
                "declines",
                "declined",
                "shrinks",
                "shrank",
                "slows",
                "slowed",
                "lower",
                "weaker",
                "downgrade",
                "downgrades",
                "downgraded",
                "lower",
                "reduce",
                "trim",
                "slash",
                "fall",
                "drop",
                "decline",
                "shrink",
                "slow",
                "downgrade",
            }
        ),
    ),
    (
        # Against an expectation: a theme cannot both beat and miss.
        "performance",
        frozenset(
            {
                "beats",
                "beat",
                "tops",
                "top",
                "topped",
                "exceeds",
                "exceed",
                "exceeded",
                "outperform",
                "outperforms",
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
                "lags",
                "lag",
                "lagged",
                "underperform",
                "underperforms",
            }
        ),
    ),
    (
        # A decision that went one way or the other.
        "decision",
        frozenset(
            {
                "approves",
                "approved",
                "approval",
                "clears",
                "cleared",
                "authorises",
                "authorises",
                "authorizes",
                "authorized",
                "grants",
                "granted",
                "wins",
                "won",
                "upholds",
                "uphold",
                "upheld",
                "approve",
                "clear",
                "authorise",
                "authorize",
                "grant",
                "win",
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
                "refuses",
                "refused",
                "bans",
                "banned",
                "loses",
                "lost",
                "overturns",
                "overturn",
                "overturned",
                "reject",
                "block",
                "deny",
                "refuse",
                "ban",
                "lose",
            }
        ),
    ),
    (
        # Whether the thing is happening at all.  Paired with the negation
        # scan below, this is what separates "will build" from "will not
        # build" and "confirms" from "denies".
        "commitment",
        frozenset(
            {
                "confirms",
                "confirmed",
                "proceeds",
                "proceeding",
                "launches",
                "launched",
                "opens",
                "opened",
                "starts",
                "started",
                "resumes",
                "resumed",
                "signs",
                "sign",
                "signed",
                "confirm",
                "proceed",
                "launch",
                "open",
                "start",
                "resume",
            }
        ),
        frozenset(
            {
                "denies",
                "denied",
                "cancels",
                "cancelled",
                "canceled",
                "scraps",
                "scrapped",
                "abandons",
                "abandoned",
                "halts",
                "halted",
                "suspends",
                "suspended",
                "delays",
                "delayed",
                "postpones",
                "postpone",
                "postponed",
                "cancel",
                "scrap",
                "abandon",
                "halt",
                "suspend",
                "delay",
            }
        ),
    ),
)

#: Tokens that flip the polarity of the claim in the same headline.  "Tesla
#: will not open the plant" makes a negative commitment claim out of a
#: positive verb, and a theme holding it beside "Tesla opens the plant"
#: contradicts itself just as plainly as one holding "opens"/"halts".
NEGATION_MARKERS = frozenset(
    {"not", "no", "never", "wont", "cannot", "denies", "denied"}
)

#: Differences M5 deliberately allows inside one theme, with the reason.
#: Listed so a reviewer can see that each was decided rather than missed;
#: every one of these vetoes in M3, where the question is a narrower one.
PERMITTED_DIFFERENCES: tuple[tuple[str, str], ...] = (
    (
        "reporting_period",
        "A day's coverage of an earnings report routinely spans the quarter "
        "reported and the quarter guided; splitting on it would separate a "
        "story from its own follow-up.",
    ),
    (
        "named_entities",
        "One narrative names several people and counterparties - the "
        "acquirer, the target, the regulator - and a theme is about the "
        "event, not about one name in it.",
    ),
    (
        "named_roles",
        "A management story legitimately mentions the outgoing CFO and the "
        "incoming one in the same theme.",
    ),
    (
        "article_type",
        "A live blog, a preview, and a recap of the same event belong to "
        "the same narrative even though none of them is the same story.",
    ),
    (
        "quantities_and_units",
        "Different figures usually mean different facets of one event - "
        "deliveries, revenue, headcount - rather than a contradiction.",
    ),
    (
        "repeated_distinct_events",
        "Two recalls in one day are two events and one narrative; the "
        "cohesion floor, not this contract, decides whether they cohere.",
    ),
)


def _claim_tokens(story: ThemeStory) -> tuple[str, ...]:
    text = display_text(
        " ".join(part for part in (story.title, story.description or "") if part)
    )
    return tokenize(text)


def story_claims(story: ThemeStory) -> frozenset[tuple[str, str]]:
    """Return the ``(family, polarity)`` claims one story makes.

    A negation marker anywhere in the text inverts that story's polarity
    for every family it asserts.  Blunt on purpose: scoping negation to a
    clause is M3's problem, and getting it wrong here costs a theme member,
    not a merge.
    """

    tokens = set(_claim_tokens(story))
    negated = bool(tokens & NEGATION_MARKERS)
    found: set[tuple[str, str]] = set()
    for family, positive, negative in OPPOSING_CLAIM_FAMILIES:
        if tokens & positive:
            found.add((family, "negative" if negated else "positive"))
        if tokens & negative:
            found.add((family, "positive" if negated else "negative"))
    return frozenset(found)


def claims_conflict(
    left: frozenset[tuple[str, str]], right: frozenset[tuple[str, str]]
) -> bool:
    """True when two claim sets take opposite sides of the same family."""

    for family, polarity in left:
        if (family, "positive" if polarity == "negative" else "negative") in right:
            return True
    return False


def incompatible_members(
    stories: Sequence[ThemeStory], ordered_positions: Sequence[int]
) -> tuple[int, ...]:
    """Return the positions to move out so the theme stops contradicting itself.

    Cluster-wide, not pairwise: each family's members are split by polarity
    across the *whole* prospective theme and the minority side leaves
    together, so a theme cannot keep a contradiction just because no single
    pair was examined.

    ``ordered_positions`` must already be in the order the caller considers
    authoritative (highest salience first).  Majority is by story count;
    ties go to the side holding the theme's leading story, so the theme
    stays about what it was about instead of flipping to whichever half was
    larger by one.
    """

    claims = {
        position: story_claims(stories[position]) for position in ordered_positions
    }
    families: dict[str, dict[str, list[int]]] = {}
    for position in ordered_positions:
        for family, polarity in sorted(claims[position]):
            families.setdefault(family, {}).setdefault(polarity, []).append(position)

    ejected: set[int] = set()
    for family in sorted(families):
        sides = families[family]
        if len(sides) < 2:
            continue
        winner = max(
            sorted(sides),
            key=lambda polarity: (
                len(sides[polarity]),
                -ordered_positions.index(
                    min(sides[polarity], key=ordered_positions.index)
                ),
            ),
        )
        for polarity, positions in sides.items():
            if polarity != winner:
                ejected.update(positions)
    return tuple(sorted(ejected))


def policy_components() -> dict[str, str]:
    """Every behaviour-changing part of this contract, named.

    Reaches the configuration fingerprint automatically, so adding a term
    to a lexicon invalidates cached themes without anyone bumping a
    constant by hand.
    """

    components: dict[str, str] = {
        "version": COMPATIBILITY_POLICY_VERSION,
        "families": ",".join(family for family, _, _ in OPPOSING_CLAIM_FAMILIES),
        "negation_markers": ",".join(sorted(NEGATION_MARKERS)),
        "scope": "whole_prospective_theme",
        "text_scope": "title_and_description",
        "tie_break": "majority_then_leading_story",
        "permitted_differences": ",".join(name for name, _ in PERMITTED_DIFFERENCES),
        "tokenizer": "nlp.dedup.structural.tokenize",
    }
    for family, positive, negative in OPPOSING_CLAIM_FAMILIES:
        components[f"family.{family}.positive"] = ",".join(sorted(positive))
        components[f"family.{family}.negative"] = ",".join(sorted(negative))
    return dict(sorted(components.items()))


def policy_fingerprint() -> str:
    """A stable digest of the whole contract."""

    encoded = json.dumps(policy_components(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

"""What a theme is *about*, beyond how close its vectors are.

Cosine floors answer "are these stories written similarly".  They do not
answer "would a reader call these one narrative", and on the committed
eighteen-story day the difference was the whole problem: the quarterly
delivery number and the quarterly grid-storage contract score 0.39 against
each other — comfortably above any floor a theme could carry — because both
are Tesla stories about a quarterly record.  A reader separates them
instantly.

So M5 reads a second, coarse signal off the public story text: **which
narrative family is this about**.  The rule is deliberately blunt and
trust-first:

* a family is only assigned on **explicit, high-confidence evidence** — a
  phrase a newsroom would have to have written on purpose;
* **two different explicit families cannot share a normal theme**;
* **missing evidence is unknown**, and unknown never blocks anything, so a
  story the classifier cannot read still clusters on geometry alone;
* one story may carry several families, and it is compatible with anything
  it shares a family with.

This is not a reimplementation of M3's guards.  M3 asks whether two records
are the same *story*; this asks whether two stories are the same *subject*,
which is a coarser question with a coarser answer.  It uses no ticker, no
item id, and no per-fixture exception: every rule below is a phrase list
that would fire the same way on any company's coverage.

**Recalls carry their product.**  Two recalls in one day are two events, and
the family key includes the product so a Cybertruck wiper recall and a
Model Y seatbelt recall are ``recall:cybertruck`` and ``recall:model_y`` —
different subjects, not one "recalls" theme.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

from nlp.dedup.text import display_text

from .models import ThemeStory

#: Bumped when a family, a phrase list, or the comparison changes.
NARRATIVE_POLICY_VERSION = "m5.narrative.v1"

#: ``(family, pattern)``.  Every pattern is a phrase a newsroom writes on
#: purpose; none is a bare noun that could appear in passing.  Ordered only
#: for readability — a story collects every family that matches.
NARRATIVE_FAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "vehicle_deliveries",
        # Anchored to vehicles or to a reporting period on purpose: a bare
        # "deliveries" also describes a storage contract's shipments, and
        # reading that as a quarterly vehicle number put a grid-storage
        # story in the delivery theme.
        re.compile(
            r"\b(?:deliver(?:s|ed|ing)\s+(?:[\d,.]+\s+)?(?:vehicles?|cars?|"
            r"units?|trucks?)|vehicle deliver(?:y|ies)|car deliver(?:y|ies)|"
            r"(?:quarterly|annual|record|first[- ]quarter|q[1-4])\s+"
            r"deliver(?:y|ies)|deliveries\s+(?:top|beat|miss|rose|fell|"
            r"climbed|dropped)|handovers?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "energy_storage",
        re.compile(
            r"\b(?:energy storage|grid storage|battery storage|storage "
            r"deployments?|megapack\w*|powerwall\w*|storage contract|"
            r"utility[- ]scale storage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "charging_infrastructure",
        re.compile(
            r"\b(?:supercharger\w*|charging (?:network|corridor|station\w*|"
            r"hub\w*|point\w*)|fast[- ]charg\w+|charging infrastructure)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "battery_manufacturing",
        re.compile(
            r"\b(?:battery (?:line|plant|factory|cell production|"
            r"manufacturing)|cell (?:line|plant)|gigafactory cell\w*|"
            r"battery production)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "factory_operations",
        re.compile(
            r"\b(?:factory|plant|assembly line|production line|gigafactory)\b"
            r"(?=.*\b(?:resumes?|resumed|restart\w*|halt\w*|shift\w*|output|"
            r"idle\w*|shutdown|reopen\w*|expand\w*|hiring|workers?)\b)"
            r"|\b(?:adds? a (?:second|third|fourth) shift|production halt\w*|"
            r"line stoppage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "investor_event",
        re.compile(
            r"\b(?:investor day|analyst day|shareholder meeting|annual "
            r"meeting|capital markets day|earnings call date|investor "
            r"conference)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "regulatory_permit",
        re.compile(
            r"\b(?:permit\w*|licen[cs]e application|applies? for|"
            r"application for|regulatory (?:approval|filing)|hearing date|"
            r"files? with the\b|petition\w*)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "product_trial",
        re.compile(
            r"\b(?:pilot (?:programme|program)|trial (?:to|in|across)|"
            r"supervised (?:self[- ]driving|driving|autonom\w+)|"
            r"beta (?:release|programme|program)|test(?:ing)? (?:fleet|"
            r"programme|program)|expands? its .{0,24}trial)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pricing",
        re.compile(
            r"\b(?:cuts? .{0,20}prices?|raises? .{0,20}prices?|price cut\w*|"
            r"price (?:increase|rise|reduction)s?|sticker price|"
            r"lower(?:s|ed)? .{0,16}price|discount\w*|price war)\b",
            re.IGNORECASE,
        ),
    ),
)

#: A recall is a family per *product*, because two recalls in one day are
#: two events.  The product is read from a short list of explicit model
#: words; a recall naming none is ``recall:unspecified``.
_RECALL = re.compile(r"\brecall(?:s|ed|ing)?\b", re.IGNORECASE)
_PRODUCT = re.compile(
    r"\b(model\s+[a-z0-9]+|cybertruck\w*|semi\b|roadster\w*|powerwall\w*|"
    r"megapack\w*|solar roof)\b",
    re.IGNORECASE,
)

#: Families that may share a theme despite being different, with the reason.
#: Deliberately tiny: the default is that two explicit subjects are two
#: subjects, and every entry here has to survive "would a reader call this
#: one story about one thing?"
#: Empty on purpose.  Every candidate exception - deliveries with pricing,
#: permits with trials - reads plausibly on one day's coverage and wrongly
#: on the next, and an exception justified by a fixture with one author is
#: not an exception, it is a fitted parameter.  Unknown families already
#: block nothing, which is where the flexibility belongs.
COMPATIBLE_FAMILY_PAIRS: tuple[tuple[str, str, str], ...] = ()

_COMPATIBLE = frozenset(
    frozenset((left, right)) for left, right, _ in COMPATIBLE_FAMILY_PAIRS
)


def _normalize_product(raw: str) -> str:
    """Fold a product name onto one key, so a plural is the same product."""

    token = re.sub(r"\s+", "_", raw.casefold().strip())
    return token[:-1] if token.endswith("s") and not token.endswith("ss") else token


def _text_of(story: ThemeStory) -> str:
    return display_text(
        " ".join(part for part in (story.title, story.description or "") if part)
    )


def narrative_families(story: ThemeStory) -> frozenset[str]:
    """Return the narrative families one story explicitly asserts.

    Empty means unknown, not "no family": a story the phrase lists cannot
    read blocks nothing.
    """

    text = _text_of(story)
    found = {family for family, pattern in NARRATIVE_FAMILIES if pattern.search(text)}
    if _RECALL.search(text):
        product = _PRODUCT.search(text)
        token = _normalize_product(product.group(1)) if product else "unspecified"
        found.add(f"recall:{token}")
    return frozenset(found)


def families_compatible(left: frozenset[str], right: frozenset[str]) -> bool:
    """True when two stories may share a normal theme.

    Unknown on either side is compatible with everything — the classifier
    declining to read a story is not evidence about it.  Otherwise they must
    share a family, or their families must be declared compatible.
    """

    if not left or not right:
        return True
    if left & right:
        return True
    return any(
        frozenset((one, other)) in _COMPATIBLE for one in left for other in right
    )


def narratively_incompatible(
    stories: Sequence[ThemeStory], ordered_positions: Sequence[int]
) -> tuple[int, ...]:
    """Return the positions to move out so a theme is about one subject.

    Cluster-wide and deterministic.  The theme keeps the largest group of
    mutually compatible members, ties broken by the caller's order — which
    is salience order, so a theme stays about what it was mostly about.
    Everything outside that group leaves together.
    """

    families = {
        position: narrative_families(stories[position])
        for position in ordered_positions
    }
    best: list[int] = []
    for anchor in ordered_positions:
        group = [anchor]
        for position in ordered_positions:
            if position == anchor:
                continue
            if all(
                families_compatible(families[position], families[member])
                for member in group
            ):
                group.append(position)
        if len(group) > len(best):
            best = group
    keep = set(best)
    return tuple(
        sorted(position for position in ordered_positions if position not in keep)
    )


def dominant_family(
    stories: Sequence[ThemeStory], positions: Sequence[int]
) -> str | None:
    """The family the most members assert, or ``None`` when unknown.

    Reported beside a theme so a reviewer can see what it claims to be
    about without re-reading its headlines.
    """

    counts: dict[str, int] = {}
    for position in positions:
        for family in narrative_families(stories[position]):
            counts[family] = counts.get(family, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda family: (-counts[family], family))


def policy_components() -> dict[str, str]:
    """Every behaviour-changing part of this layer, named."""

    components: dict[str, str] = {
        "version": NARRATIVE_POLICY_VERSION,
        "families": ",".join(family for family, _ in NARRATIVE_FAMILIES) + ",recall:*",
        "unknown_rule": "empty_family_set_blocks_nothing",
        "default_rule": "two_distinct_explicit_families_cannot_share_a_theme",
        "scope": "whole_prospective_theme_after_subset_extraction",
        "selection": "largest_mutually_compatible_group_ties_by_salience_order",
        "recall_key": "recall:<product>_or_recall:unspecified",
        "recall_pattern": _RECALL.pattern,
        "product_pattern": _PRODUCT.pattern,
        "compatible_pairs": ";".join(
            f"{left}+{right}" for left, right, _ in COMPATIBLE_FAMILY_PAIRS
        ),
        "text_scope": "title_and_description",
    }
    for family, pattern in NARRATIVE_FAMILIES:
        components[f"family.{family}"] = pattern.pattern
    return dict(sorted(components.items()))


def policy_fingerprint() -> str:
    """A stable digest of the whole layer."""

    encoded = json.dumps(policy_components(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

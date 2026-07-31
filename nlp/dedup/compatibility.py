"""The one compatibility gate every non-authoritative merge passes through.

Earlier versions checked contradictions per tier, and each tier checked
something slightly different, so the same false merge kept reappearing
through whichever tier had the weakest check.  There is now exactly one
gate: :func:`incompatibility`.  :mod:`nlp.dedup.detection` calls it before
unioning **any** edge that is not authoritative provider identity, so URL,
exact-content, exact-title, and MinHash candidates are all held to the same
rule and none of them can be widened independently.

The rule is asymmetric on purpose:

* **explicit disagreement vetoes.**  Two records that say different things
  are different records, however strongly their URL, headline, or timing
  agree.
* **missing information does not.**  A record without a description does not
  contradict one that has a description; it just says less.  Vetoing there
  would refuse ordinary syndication where one outlet omits the standfirst.

That is what keeps this pair apart even though the titles normalize
identically::

    title:       "Quarterly results"
    description: "Profit rose sharply."   vs   "Profit fell sharply."

and what still lets a Reuters original merge with a Yahoo copy that carries
the same headline and no standfirst at all.
"""

from __future__ import annotations

from .models import NormalizedItem
from .structural import protected_expressions

#: Tokens that flip the meaning of a sentence.  A disagreement in how many
#: of these appear, over fields both records actually have, is treated as an
#: explicit contradiction rather than a wording difference.
NEGATION_TOKENS = frozenset(
    {
        "no",
        "not",
        "never",
        "without",
        "denies",
        "denied",
        "deny",
        "rejects",
        "rejected",
        "reject",
        "refuses",
        "refused",
        "halts",
        "halted",
        "cancels",
        "cancelled",
        "canceled",
        "drops",
        "dropped",
        "abandons",
        "abandoned",
        "fails",
        "failed",
        "unable",
        "neither",
        "nor",
    }
)

#: Ordered reasons, most specific first, so the recorded veto explains the
#: real objection instead of a generic "the text differs".
VETO_REASONS = (
    "uncertain_numeric_notation",
    "text_presence",
    "protected_expression",
    "negation",
    "title",
    "description",
)


def _negation_count(tokens: tuple[str, ...]) -> int:
    return sum(1 for token in tokens if token in NEGATION_TOKENS)


def _description_tokens(item: NormalizedItem) -> tuple[str, ...]:
    return tuple(item.normalized_description.split())


def incompatibility(left: NormalizedItem, right: NormalizedItem) -> str | None:
    """Return why two records may not merge, or ``None`` when they may.

    Only fields *both* records carry are compared, so absence is never a
    contradiction.  The returned string is one of :data:`VETO_REASONS` and
    is counted in :attr:`~nlp.dedup.models.DedupStats.veto_counts`.
    """

    # A title whose numeric notation the core could not represent cannot be
    # compared safely at all.
    if left.uncertain_title or right.uncertain_title:
        return "uncertain_numeric_notation"

    # One record carries no usable text and the other carries some: there is
    # no evidence they describe the same thing.
    if (left.content_fingerprint is None) != (right.content_fingerprint is None):
        return "text_presence"

    comparable_titles = left.title_key is not None and right.title_key is not None
    comparable_descriptions = bool(
        left.normalized_description and right.normalized_description
    )

    # Numbers, currencies, units, ranges, signs, quarters, years, and dates
    # travel inside the protected expressions of whichever fields exist on
    # both sides.
    if comparable_titles and left.title_protected != right.title_protected:
        return "protected_expression"
    if comparable_descriptions and protected_expressions(
        _description_tokens(left)
    ) != protected_expressions(_description_tokens(right)):
        return "protected_expression"

    if comparable_titles and _negation_count(
        tuple((left.title_key or "").split())
    ) != _negation_count(tuple((right.title_key or "").split())):
        return "negation"
    if comparable_descriptions and _negation_count(
        _description_tokens(left)
    ) != _negation_count(_description_tokens(right)):
        return "negation"

    if comparable_titles and left.title_key != right.title_key:
        return "title"
    if comparable_descriptions and (
        left.normalized_description != right.normalized_description
    ):
        return "description"
    return None

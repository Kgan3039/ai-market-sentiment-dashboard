"""The shared evidence primitive: one definition of "these disagree".

Two things need to decide whether records contradict each other — the
compatibility gate every circumstantial merge passes, and provider-conflict
quarantine — and earlier revisions gave them separate definitions that
drifted apart.  Both now go through :func:`combine`, so provider quarantine
can be *stricter* than ordinary compatibility but never weaker.

The rule is asymmetric on purpose:

* **explicit disagreement vetoes.**  Two records that say different things
  are different records, however strongly their URL, headline, or timing
  agree.
* **missing information does not.**  A record without a description does not
  contradict one that has a description; it just says less.

That keeps this pair apart even though the titles normalize identically::

    title:       "Quarterly results"
    description: "Profit rose sharply."   vs   "Profit fell sharply."

while a Reuters original still merges with a copy carrying the same headline
and no standfirst at all.

Comparison works on :class:`EvidenceSummary`, which describes a *set* of
records rather than one.  That is what lets the gate reason about whole
clusters: a sparse record may bridge two others only when the values they
do carry agree, so A("profit rose") — B(no description) — C("profit fell")
can never end up in one cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .models import NormalizedItem
from .structural import protected_expressions

#: Bumped when the tokens, the reason order, or the comparison changes.
COMPATIBILITY_POLICY_VERSION = "m2.compatibility.v2"

#: Bumped when provider-conflict quarantine changes what it catches.
PROVIDER_CONFLICT_POLICY_VERSION = "m2.provider.v2"

#: The checks provider identity applies *on top of* the shared primitive.
#: Provider quarantine may be stricter than ordinary compatibility, never
#: weaker, so this is an addition to :data:`VETO_REASONS`, not a
#: replacement for it.
PROVIDER_EXTRA_CHECKS = ("url", "ticker", "published_at")

#: Tokens that flip the meaning of a sentence.  A disagreement in how many
#: appear, over fields the records actually share, is an explicit
#: contradiction rather than a wording difference.
NEGATION_TOKENS = frozenset(
    {
        "abandoned",
        "abandons",
        "cancelled",
        "canceled",
        "cancels",
        "denied",
        "denies",
        "deny",
        "dropped",
        "drops",
        "failed",
        "fails",
        "halted",
        "halts",
        "neither",
        "never",
        "no",
        "nor",
        "not",
        "refused",
        "refuses",
        "reject",
        "rejected",
        "rejects",
        "unable",
        "without",
    }
)


@dataclass(frozen=True)
class EvidenceSummary:
    """What a set of records collectively asserts.

    Every field holds the *known* values across the set.  A consistent set
    has at most one known value per field; two values means the set already
    contradicts itself, which only happens if a caller combines summaries
    that were never gated.
    """

    uncertain: bool
    text_bearing: frozenset[bool]
    titles: frozenset[str]
    descriptions: frozenset[str]
    title_protected: frozenset[tuple[str, ...]]
    description_protected: frozenset[tuple[str, ...]]
    title_negations: frozenset[int]
    description_negations: frozenset[int]

    @property
    def is_consistent(self) -> bool:
        """True when no field carries two different known values."""

        return not self.uncertain and all(
            len(values) <= 1
            for values in (
                self.text_bearing,
                self.titles,
                self.descriptions,
                self.title_protected,
                self.description_protected,
                self.title_negations,
                self.description_negations,
            )
        )


#: Field checks in order, most specific first, so a recorded veto explains
#: the real objection rather than a generic "the text differs".  The order
#: is load-bearing: :func:`combine` walks it.
_CHECKS: tuple[tuple[str, str], ...] = (
    ("text_presence", "text_bearing"),
    ("protected_expression", "title_protected"),
    ("protected_expression", "description_protected"),
    ("negation", "title_negations"),
    ("negation", "description_negations"),
    ("title", "titles"),
    ("description", "descriptions"),
)

#: Every reason :func:`combine` can return, for callers that enumerate them.
VETO_REASONS = ("uncertain_numeric_notation",) + tuple(
    dict.fromkeys(reason for reason, _ in _CHECKS)
)


def _negations(tokens: tuple[str, ...]) -> int:
    return sum(1 for token in tokens if token in NEGATION_TOKENS)


def summarize(item: NormalizedItem) -> EvidenceSummary:
    """Return the evidence one record carries."""

    description_tokens = tuple(item.normalized_description.split())
    has_description = bool(item.normalized_description)
    has_title = item.title_key is not None
    return EvidenceSummary(
        uncertain=item.uncertain_title,
        text_bearing=frozenset({item.content_fingerprint is not None}),
        titles=frozenset({item.title_key} if has_title else ()),
        descriptions=(
            frozenset({item.normalized_description}) if has_description else frozenset()
        ),
        title_protected=(
            frozenset({item.title_protected}) if has_title else frozenset()
        ),
        description_protected=(
            frozenset({protected_expressions(description_tokens)})
            if has_description
            else frozenset()
        ),
        title_negations=(
            frozenset({_negations(tuple((item.title_key or "").split()))})
            if has_title
            else frozenset()
        ),
        description_negations=(
            frozenset({_negations(description_tokens)})
            if has_description
            else frozenset()
        ),
    )


def combine(
    left: EvidenceSummary, right: EvidenceSummary
) -> tuple[EvidenceSummary | None, str | None]:
    """Merge two summaries, or report why the sets contradict each other.

    Because each side is internally consistent, a field holds at most one
    known value per side; the union may therefore hold at most one for the
    merged set to be admissible.  That is exactly equivalent to comparing
    every record on the left against every record on the right, at constant
    cost — which is what makes cluster-wide checking affordable.
    """

    if left.uncertain or right.uncertain:
        return None, "uncertain_numeric_notation"
    merged = EvidenceSummary(
        uncertain=False,
        text_bearing=left.text_bearing | right.text_bearing,
        titles=left.titles | right.titles,
        descriptions=left.descriptions | right.descriptions,
        title_protected=left.title_protected | right.title_protected,
        description_protected=(
            left.description_protected | right.description_protected
        ),
        title_negations=left.title_negations | right.title_negations,
        description_negations=left.description_negations | right.description_negations,
    )
    for reason, field in _CHECKS:
        if len(getattr(merged, field)) > 1:
            return None, reason
    return merged, None


def incompatibility(left: NormalizedItem, right: NormalizedItem) -> str | None:
    """Return why two records may not merge, or ``None`` when they may."""

    return combine(summarize(left), summarize(right))[1]


def policy_fingerprint() -> str:
    """Return a digest of this module's static compatibility policy."""

    payload = "|".join(
        (
            COMPATIBILITY_POLICY_VERSION,
            ",".join(sorted(NEGATION_TOKENS)),
            ",".join(f"{reason}:{field}" for reason, field in _CHECKS),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_policy_fingerprint() -> str:
    """Return a digest of the static provider-conflict policy."""

    payload = "|".join((PROVIDER_CONFLICT_POLICY_VERSION,) + PROVIDER_EXTRA_CHECKS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

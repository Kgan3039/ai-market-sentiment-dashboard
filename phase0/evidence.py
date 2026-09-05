"""The evidence read/projection boundary between Phase 0 and M1-M5.

This module answers two questions and deliberately nothing else: *which
persisted evidence may a downstream stage process*, and *what does one row
look like once it reaches that stage*.  It runs no stage: M1 embeddings,
M2/M3 dedup, and M5 clustering are wired elsewhere.

**Eligibility (decision F).**  A raw item enters ticker ``T``'s processing
on day ``D`` when it is ``valid``, authoritatively associated with ``T`` in
``raw_item_tickers``, derived onto ``D``, and projectable.
``raw_item_candidates`` and ``raw_item_match_evidence`` are observability:
they may explain why ownership was withheld, and they never confer it.

**Nothing disappears quietly.**  Evidence that cannot be processed is
counted and inspectable rather than dropped -- an invalid payload, an item
whose timestamp the dedup core refuses, a source the publisher policy
cannot represent.  Those are the three exclusion outcomes below, and the
counters in :class:`EvidencePartition` are produced by *this* logic rather
than by a second definition written in SQL, so what a partition reports and
what it can emit cannot drift apart.

**Projectability is the downstream contract itself.**  An item is
projectable when :func:`nlp.dedup.normalization.normalize_item` accepts it,
plus decision F's rule 4 -- a non-empty title or a non-empty URL.  Rule 4 is
defence behind rule 1: the ``raw_items`` CHECK already guarantees a valid
row has both, so it fires only if that invariant is ever relaxed, and then
it fails loudly instead of producing an empty-headline story.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nlp.dedup.errors import DedupInputError
from nlp.dedup.models import RawItem
from nlp.dedup.normalization import normalize_item

from .errors import Phase0IntegrityError
from .publishers import (
    PUBLISHER_POLICY_VERSION,
    PublisherPolicyError,
    canonical_publisher,
    source_scheme,
)
from .tickers import SUPPORTED_TICKERS

#: Bumped when the projection's own shape changes.  Separate from
#: :data:`~phase0.publishers.PUBLISHER_POLICY_VERSION`: the outlet policy
#: can move without the projection moving, and a reader comparing two runs
#: needs to know which one did.
PROJECTION_VERSION = "i5.projection.v1"

#: The three exclusion outcomes.  ``invalid`` covers every ``ingest_status``
#: that is not ``valid`` -- ``'invalid'`` and ``'ambiguous'`` both -- because
#: decision A1's partition invariant has exactly three terms and an
#: ambiguous row is, for downstream purposes, evidence we may not process.
#: :attr:`EvidencePartition.excluded_ambiguous` names the ambiguous share of
#: it without becoming a fourth term.
EXCLUDED_INVALID = "invalid"
EXCLUDED_UNPROJECTABLE = "unprojectable"
ELIGIBLE = "eligible"


@dataclass(frozen=True)
class ExcludedEvidence:
    """One associated item that may not be processed, and why."""

    raw_item_id: int
    ticker: str
    trading_day: str
    #: :data:`EXCLUDED_INVALID` or :data:`EXCLUDED_UNPROJECTABLE`.
    outcome: str
    ingest_status: str
    detail: str


@dataclass(frozen=True)
class PartitionEvidence:
    """Everything one ticker-day partition may hand to a downstream stage.

    ``items`` is what M2 receives.  ``excluded`` is the rest of the
    partition's associated evidence, kept beside it so a caller reporting
    "42 stories" can also say what the other 3 rows were.
    """

    ticker: str
    trading_day: str
    items: tuple[RawItem, ...]
    excluded: tuple[ExcludedEvidence, ...]
    projection_version: str = PROJECTION_VERSION
    publisher_policy_version: str = PUBLISHER_POLICY_VERSION

    @property
    def associated_item_count(self) -> int:
        return len(self.items) + len(self.excluded)


@dataclass(frozen=True)
class EvidencePartition:
    """Evidence accounting for one ``(ticker, trading_day)``.

    The population is only what is authoritatively associated with this
    ticker, so there is no ``total_item_count`` here and no
    ``excluded_unassociated``: the day's total includes items this ticker
    has no claim on, and unassociated evidence belongs to no ticker at all.
    Both live on :class:`EvidenceDay`.

    Under decision E one article may be associated with several tickers and
    is counted once in each of their partitions.  **These rows are therefore
    not a partition of the day's evidence** and their counts must never be
    summed against :attr:`EvidenceDay.associated_any_ticker`.
    """

    ticker: str
    trading_day: str
    associated_item_count: int
    eligible_item_count: int
    excluded_invalid: int
    excluded_unprojectable: int
    #: Diagnostic only: the ``ingest_status = 'ambiguous'`` share of
    #: :attr:`excluded_invalid`, already counted there.  Never a fourth term.
    excluded_ambiguous: int
    latest_fetched_at: str | None

    def __post_init__(self) -> None:
        total = (
            self.eligible_item_count
            + self.excluded_invalid
            + self.excluded_unprojectable
        )
        if total != self.associated_item_count:
            raise Phase0IntegrityError(
                f"evidence partition {self.ticker}/{self.trading_day} does not "
                f"account for its own population: {self.eligible_item_count} "
                f"eligible + {self.excluded_invalid} invalid + "
                f"{self.excluded_unprojectable} unprojectable != "
                f"{self.associated_item_count} associated"
            )
        if self.excluded_ambiguous > self.excluded_invalid:
            raise Phase0IntegrityError(
                f"evidence partition {self.ticker}/{self.trading_day} reports "
                f"more ambiguous rows than invalid ones, but ambiguous is a "
                f"subset of invalid"
            )


@dataclass(frozen=True)
class EvidenceDay:
    """Evidence accounting for one trading day, ownership included or not.

    The only invariant here is
    ``associated_any_ticker + unassociated_item_count == total_item_count``.

    The four unassociated signals below are **independent counts, not a
    cause partition**: one item can be ambiguous, carry candidate rows, and
    carry match evidence at the same time, and forcing that into a single
    cause would invent a fact.  Only ``unassociated_without_evidence`` is
    complementary — it counts the items neither of the evidence signals
    named.  For the whole story about one item, read
    :meth:`~phase0.repository.Phase0Reader.unassociated_items`.
    """

    trading_day: str
    total_item_count: int
    associated_any_ticker: int
    unassociated_item_count: int
    unassociated_invalid: int
    unassociated_ambiguous: int
    unassociated_with_candidates: int
    unassociated_with_match_evidence: int
    unassociated_without_evidence: int
    latest_fetched_at: str | None

    def __post_init__(self) -> None:
        if (
            self.associated_any_ticker + self.unassociated_item_count
            != self.total_item_count
        ):
            raise Phase0IntegrityError(
                f"evidence day {self.trading_day} does not account for its own "
                f"population: {self.associated_any_ticker} associated + "
                f"{self.unassociated_item_count} unassociated != "
                f"{self.total_item_count} total"
            )


@dataclass(frozen=True)
class CandidateEvidence:
    """A ticker something suggested for an item, and nothing accepted."""

    ticker: str
    reason: str


@dataclass(frozen=True)
class MatchEvidence:
    """Why one ticker was matched or excluded for an item, as stored."""

    ticker: str
    decision: str
    evidence: tuple[Any, ...]


@dataclass(frozen=True)
class UnassociatedItem:
    """One raw item no ticker holds, with the evidence that explains it.

    Whether the withheld ticker was right is a question for a reader, which
    is why the candidate and match rows travel with the item rather than
    being reduced to a count.
    """

    raw_item_id: int
    trading_day: str
    source: str
    ingest_status: str
    title: str | None
    url: str | None
    canonical_url: str
    external_id: str | None
    published_at: str | None
    fetched_at: str
    validation_errors: tuple[Any, ...]
    candidates: tuple[CandidateEvidence, ...]
    match_evidence: tuple[MatchEvidence, ...]


@dataclass(frozen=True)
class WithheldMatch:
    """Match evidence naming a ticker that holds no association.

    Keyed on ``(raw_item_id, ticker)``: under the current relevance policy
    (``ambiguous_match_action: flag_do_not_assign``) an item matching two
    tickers is flagged and assigned to neither, and an item legitimately
    associated with AMD can still have a withheld NVDA match.  An
    association to *another* ticker therefore never hides this signal.

    Its denominator is items whose evidence names this ticker.  It is not an
    eligibility counter and must never be summed with one.
    """

    raw_item_id: int
    ticker: str
    trading_day: str
    ingest_status: str
    source: str
    title: str | None
    evidence: tuple[Any, ...]


def provider_item_id(source: str | None, external_id: str | None) -> str | None:
    """Qualify a bare provider id with the scheme that issued it.

    ``raw_items.external_id`` holds the provider-native value -- Yahoo's own
    article id, an RSS guid -- and the scheme lives on ``source``.  M2 keys
    its authoritative tier on ``provider_namespace(source) + provider id``,
    and canonicalization can make that namespace *identical* for a unified
    publisher, so the scheme qualifier here is the only thing keeping the
    two id spaces apart.  It is read from the persisted source and never
    from the canonical publisher id, which by then may name both.

    ``None`` in, ``None`` out: an absent or blank id is no identifier, and
    ``yahoo:None`` would be a fabricated one.
    """

    identifier = str(external_id or "").strip()
    if not identifier:
        return None
    scheme = source_scheme(source)
    if scheme is None:
        raise PublisherPolicyError(
            f"source {source!r} carries no recognized provider scheme, so its "
            f"provider id cannot be qualified; an unqualified id would let the "
            f"Yahoo and RSS id spaces collide"
        )
    return f"{scheme}:{identifier}"


def project_raw_item(row: Mapping[str, Any], ticker: str) -> RawItem:
    """Project one persisted row onto the downstream input contract.

    Raises :class:`~nlp.dedup.errors.DedupInputError` or
    :class:`~phase0.publishers.PublisherPolicyError` when the row cannot be
    projected.  Callers that are counting evidence use
    :func:`classify_evidence`, which turns that into a measurable exclusion
    instead of an error.

    Nothing is inferred: ``source`` is the sole outlet input, the ticker is
    the caller's authoritative association rather than ``raw_items.ticker``,
    and no field is substituted for another.
    """

    title = str(row.get("title") or "").strip()
    url = str(row.get("url") or "").strip()
    canonical_url = str(row.get("canonical_url") or "").strip()
    if not title and not url:
        # Decision F rule 4.  The CHECK on `raw_items` already promises a
        # valid row has both; this is what catches the day that promise is
        # relaxed, instead of M2 emitting an empty-headline story.
        raise DedupInputError(
            f"raw item {row.get('id')} has neither a title nor a URL and "
            f"cannot be projected"
        )

    source = str(row.get("source") or "")
    item = RawItem(
        item_id=str(row["id"]),
        ticker=ticker,
        title=title or None,
        description=str(row.get("description") or "").strip() or None,
        url=url or None,
        canonical_url=canonical_url or None,
        source=canonical_publisher(source),
        published_at=row.get("published_at"),
        provider_item_id=provider_item_id(source, row.get("external_id")),
    )
    # Proving the contract by calling it: the core is the authority on what
    # it accepts, and predicting its answer here is how the two definitions
    # would drift.  Pure and deterministic, so this costs nothing but time.
    normalize_item(item, SUPPORTED_TICKERS)
    return item


def classify_evidence(
    row: Mapping[str, Any], ticker: str, trading_day: str
) -> tuple[str, RawItem | None, ExcludedEvidence | None]:
    """Decide one associated row's fate: eligible, or excluded and why.

    This is the single definition the partition counters and the projected
    evidence both come from.
    """

    status = str(row.get("ingest_status") or "")
    raw_item_id = int(row["id"])
    if status != "valid":
        return (
            EXCLUDED_INVALID,
            None,
            ExcludedEvidence(
                raw_item_id=raw_item_id,
                ticker=ticker,
                trading_day=trading_day,
                outcome=EXCLUDED_INVALID,
                ingest_status=status,
                detail=f"ingest_status is {status!r}, not 'valid'",
            ),
        )
    try:
        item = project_raw_item(row, ticker)
    except (DedupInputError, PublisherPolicyError) as exc:
        return (
            EXCLUDED_UNPROJECTABLE,
            None,
            ExcludedEvidence(
                raw_item_id=raw_item_id,
                ticker=ticker,
                trading_day=trading_day,
                outcome=EXCLUDED_UNPROJECTABLE,
                ingest_status=status,
                detail=str(exc),
            ),
        )
    return ELIGIBLE, item, None


__all__ = [
    "CandidateEvidence",
    "ELIGIBLE",
    "EXCLUDED_INVALID",
    "EXCLUDED_UNPROJECTABLE",
    "EvidenceDay",
    "EvidencePartition",
    "ExcludedEvidence",
    "MatchEvidence",
    "PROJECTION_VERSION",
    "PartitionEvidence",
    "UnassociatedItem",
    "WithheldMatch",
    "classify_evidence",
    "project_raw_item",
    "provider_item_id",
]

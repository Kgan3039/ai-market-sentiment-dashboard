"""``observe_phase0_providers.py``: look at what the providers actually send.

I5 needs two facts before it can be implemented, and neither is settled by
reading the code:

1. Does Yahoo hand out a stable, article-scoped identifier that
   ``raw_items.external_id`` could safely carry?  ``phase0/yahoo.py`` reads
   ``item["id"]`` and ``item["uuid"]`` only on the *invalid* evidence path;
   a valid item is persisted today with no ``external_id`` at all.
2. Which exact ``yahoo:<publisher>`` and ``rss:<host>`` strings does
   ingestion really produce?  I5's explicit publisher map has to be built
   from observed strings, not from guesses about publisher names.

This tool answers both by observation and writes a reviewable artifact.  It
is diagnostic only: it never constructs a :class:`~phase0.repository.Phase0Repository`,
never writes a row, never reads or mutates ``source_state``, and nothing in
``pipeline.py`` reaches it.  Network access lives behind injected callables,
so every parser, summary, and renderer below is exercised offline.

**Whose code produced a string.**  Where a *stored* value is being observed,
the production function that produces it is called rather than re-derived:
:func:`~phase0.yahoo.normalize_yahoo_item` for the Yahoo source string and
``RSSFetcher._normalized_entries`` for the RSS one.  That RSS seam is private
on purpose -- the public ``fetch`` persists -- and reaching for it is what
makes "what the code actually stores" a measurement instead of a second
implementation free to drift from the first.

**What conditional GET means here.**  Observation issues an unconditional
request every time, because reading a feed's stored ETag would mean reading
``source_state``.  So a 304 never appears in these numbers, and the fetch
this tool performs is not identical to a scheduled one.  That is a
limitation, recorded as one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import requests

from phase0.redaction import redact_secrets
from phase0.repository import utc_now
from phase0.rss import RSSFetcher, _read_response_body, load_feed_config, parse_feed
from phase0.yahoo import normalize_yahoo_item


#: Bumped when the artifact's shape changes, so a committed observation can
#: always be read by the code that claims to understand it.
ARTIFACT_SCHEMA = "i5-provider-observation/2"

#: The candidate provider-identifier fields, in the order the artifact
#: reports them.  ``uuid`` is the legacy shape ``phase0/yahoo.py`` still
#: reads; ``id`` is the current one; ``content.id`` is observed because a
#: current payload carries it too, and a field that merely *repeats* the
#: top-level id is worth proving rather than assuming.
CANDIDATE_FIELDS = ("id", "uuid", "content.id")

#: The header a scheduled RSS fetch sends, copied so the response this tool
#: observes is the response ingestion would have parsed.
RSS_USER_AGENT = "TickerNarrativesPhase0/1.0"

DEFAULT_TICKERS = ("AAPL", "AMD", "META", "NVDA", "TSLA")

#: Decision G asks that the same article be seen to carry the same
#: identifier "across fetches hours apart".  Two hours is the smallest span
#: that honestly reads as "hours"; the number is here rather than inline so
#: a reviewer can see what the artifact's claim was measured against.
#:
#: It is measured *per article*.  How long a run lasted says nothing about
#: an identifier: a twenty-hour run whose articles were each seen twice ten
#: minutes apart tested a ten-minute claim twenty hours late.
DECISION_G_STABILITY_SECONDS = 2 * 60 * 60


# -- Observation records -------------------------------------------------


@dataclass(frozen=True)
class YahooItemObservation:
    """One provider record, reduced to the fields that settle a question.

    Deliberately narrow: the raw payload carries thumbnails, resolutions,
    premium-finance flags, and storyline objects that establish nothing here
    and would bloat a committed artifact.
    """

    attempt: int
    observed_at: str
    ticker: str
    position: int
    candidate_ids: Mapping[str, str | None]
    valid: bool
    validation_error: str | None
    stored_source: str | None
    raw_publisher: str | None
    publisher_field: str | None
    provider_display_name: str | None
    provider_source_id: str | None
    provider_url: str | None
    canonical_url: str | None
    title: str


@dataclass(frozen=True)
class RSSEntryObservation:
    """One feed entry, as ``RSSFetcher`` would have persisted it."""

    attempt: int
    observed_at: str
    feed_id: str
    entry_link: str | None
    resolved_host: str | None
    stored_source: str
    canonical_url: str
    external_id: str
    title: str | None
    ingest_status: str


@dataclass(frozen=True)
class AttemptRecord:
    """What one round of requests did, whether or not it produced items."""

    attempt: int
    started_at: str
    yahoo: Mapping[str, Any]
    rss: Mapping[str, Any]


# -- Yahoo observation ---------------------------------------------------


def _text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def candidate_ids(payload: Any) -> dict[str, str | None]:
    """Read every candidate provider identifier off one payload.

    Reads the fields, and only the fields.  Nothing here decides what an id
    *means* -- that is the summary's job, and it decides it from repeated
    observations rather than from the name of the key.
    """

    mapping = payload if isinstance(payload, Mapping) else {}
    content = mapping.get("content")
    content = content if isinstance(content, Mapping) else {}
    return {
        "id": _text(mapping.get("id")),
        "uuid": _text(mapping.get("uuid")),
        "content.id": _text(content.get("id")),
    }


def _publisher_fields(
    payload: Any,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Mirror ``normalize_yahoo_item``'s publisher precedence, and say which won.

    The *values* come straight off the payload; only the name of the branch
    that fired is derived, and the raw values are reported beside it so the
    derivation can be checked rather than trusted.
    """

    mapping = payload if isinstance(payload, Mapping) else {}
    content = mapping.get("content")
    content = content if isinstance(content, Mapping) else {}
    provider = content.get("provider")
    provider = provider if isinstance(provider, Mapping) else {}
    display_name = _text(provider.get("displayName"))
    source_id = _text(provider.get("sourceId"))
    provider_url = _text(provider.get("url"))
    legacy = _text(mapping.get("publisher"))
    if legacy:
        field = "publisher"
    elif display_name:
        field = "content.provider.displayName"
    else:
        field = "fallback:Yahoo Finance"
    return field, legacy or display_name, display_name, source_id, provider_url


def observe_yahoo_item(
    ticker: str,
    payload: Any,
    *,
    attempt: int,
    observed_at: str,
) -> YahooItemObservation:
    """Reduce one provider record to an observation, valid or not.

    An unusable record is kept: ``phase0/yahoo.py`` keeps it too, as invalid
    evidence under ``yahoo:<ticker>``, and a coverage fraction that quietly
    dropped the records the provider got wrong would overstate itself.
    """

    field, raw_publisher, display_name, source_id, provider_url = _publisher_fields(
        payload
    )
    try:
        normalized = normalize_yahoo_item(ticker, payload, fetched_at=observed_at)
    except Exception as exc:  # the provider's shape is the thing under test
        mapping = payload if isinstance(payload, Mapping) else {}
        content = mapping.get("content")
        content = content if isinstance(content, Mapping) else {}
        return YahooItemObservation(
            attempt=attempt,
            observed_at=observed_at,
            ticker=ticker,
            position=-1,
            candidate_ids=candidate_ids(payload),
            valid=False,
            validation_error=str(redact_secrets(str(exc))),
            stored_source=None,
            raw_publisher=_redact(raw_publisher),
            publisher_field=field,
            provider_display_name=_redact(display_name),
            provider_source_id=_redact(source_id),
            provider_url=_redact(provider_url),
            canonical_url=None,
            title=_redact(str(mapping.get("title") or content.get("title") or ""))
            or "",
        )
    return YahooItemObservation(
        attempt=attempt,
        observed_at=observed_at,
        ticker=ticker,
        position=-1,
        candidate_ids=candidate_ids(payload),
        valid=True,
        validation_error=None,
        stored_source=_redact(normalized["source"]),
        raw_publisher=_redact(raw_publisher),
        publisher_field=field,
        provider_display_name=_redact(display_name),
        provider_source_id=_redact(source_id),
        provider_url=_redact(provider_url),
        canonical_url=_redact(normalized["canonical_url"]),
        title=_redact(normalized["title"]),
    )


def _redact(value: str | None) -> str | None:
    if value is None:
        return None
    return str(redact_secrets(value))


def observe_yahoo_response(
    ticker: str,
    items: Sequence[Any],
    *,
    attempt: int,
    observed_at: str,
) -> list[YahooItemObservation]:
    """Observe a whole response, keeping each record's position in it.

    Position is retained because "the identifier is really a slot in this
    response" is one of the semantics that has to be ruled out, and it
    cannot be ruled out without it.
    """

    observations = []
    for position, payload in enumerate(items):
        entry = observe_yahoo_item(
            ticker, payload, attempt=attempt, observed_at=observed_at
        )
        observations.append(
            YahooItemObservation(
                attempt=entry.attempt,
                observed_at=entry.observed_at,
                ticker=entry.ticker,
                position=position,
                candidate_ids=entry.candidate_ids,
                valid=entry.valid,
                validation_error=entry.validation_error,
                stored_source=entry.stored_source,
                raw_publisher=entry.raw_publisher,
                publisher_field=entry.publisher_field,
                provider_display_name=entry.provider_display_name,
                provider_source_id=entry.provider_source_id,
                provider_url=entry.provider_url,
                canonical_url=entry.canonical_url,
                title=entry.title,
            )
        )
    return observations


# -- Yahoo provider-id findings ------------------------------------------

#: Every semantics verdict this tool can reach, with what each one means.
#: A verdict is a *conclusion from repeated observation*; none of them can
#: be reached from a field name.
SEMANTICS = {
    "absent": "no valid item carried the field",
    "article_scoped": (
        "one identifier per article, unchanged across attempts, positions, "
        "and tickers, and never shared by two articles"
    ),
    "article_scoped_unconfirmed": (
        "consistent with an article-scoped identifier, but no article was "
        "observed twice, so stability was never actually tested"
    ),
    "ticker_scoped": (
        "the same article carried different identifiers under different tickers"
    ),
    "response_position_scoped": (
        "the same article carried a different identifier at a different "
        "position in the response"
    ),
    "publisher_scoped": "different articles from one publisher shared an identifier",
    "colliding": "demonstrably different articles shared an identifier",
    "unstable": "the same article carried different identifiers for no observed reason",
}


@dataclass(frozen=True)
class CandidateFindings:
    """What repeated observation established about one candidate field."""

    field: str
    valid_item_count: int
    present_count: int
    presence_fraction: float
    distinct_ids: int
    articles_observed: int
    articles_repeated: int
    articles_at_multiple_positions: int
    cross_ticker_articles: int
    repeat_spans: tuple[Mapping[str, Any], ...]
    repeat_span_summary: Mapping[str, Any]
    stability_span_met: bool
    unstable_articles: tuple[Mapping[str, Any], ...]
    position_varying_articles: tuple[Mapping[str, Any], ...]
    cross_ticker_divergent_articles: tuple[Mapping[str, Any], ...]
    colliding_ids: tuple[Mapping[str, Any], ...]
    semantics: str
    evidence: tuple[str, ...]


def _article_key(observation: YahooItemObservation) -> str | None:
    """Identify an article independently of the field under test.

    The canonical URL is the only article identity available that does not
    come from the candidate itself; using the candidate would make every
    stability question answer itself.
    """

    return observation.canonical_url


def summarize_candidate(
    observations: Sequence[YahooItemObservation],
    field: str,
    *,
    required_span_seconds: float = DECISION_G_STABILITY_SECONDS,
) -> CandidateFindings:
    """Establish presence, stability, scope, and collisions for one field.

    Stability is measured per article, not per run.  ``required_span_seconds``
    is the bar each *repeated article* has to clear on its own, and the
    spans behind that count are reported rather than reduced to a flag.
    """

    valid = [entry for entry in observations if entry.valid and _article_key(entry)]
    present = [entry for entry in valid if entry.candidate_ids.get(field)]

    by_article: dict[str, list[YahooItemObservation]] = defaultdict(list)
    for entry in valid:
        by_article[_article_key(entry)].append(entry)
    by_id: dict[str, set[str]] = defaultdict(set)
    for entry in present:
        by_id[entry.candidate_ids[field]].add(_article_key(entry))

    unstable: list[Mapping[str, Any]] = []
    position_varying: list[Mapping[str, Any]] = []
    cross_ticker_divergent: list[Mapping[str, Any]] = []
    repeat_spans: list[Mapping[str, Any]] = []
    repeated = 0
    multi_position = 0
    cross_ticker = 0
    for url, entries in sorted(by_article.items()):
        ids = sorted(
            {
                entry.candidate_ids[field]
                for entry in entries
                if entry.candidate_ids.get(field)
            }
        )
        attempts = sorted({entry.attempt for entry in entries})
        positions = sorted({entry.position for entry in entries})
        tickers = sorted({entry.ticker for entry in entries})
        if len(attempts) > 1:
            repeated += 1
        if len(positions) > 1:
            multi_position += 1
        if len(tickers) > 1:
            cross_ticker += 1
        # How long *this article* was watched.  Only observations that
        # carried the field count: an observation with no identifier is no
        # evidence that the identifier held, and folding it in would stretch
        # the span with time nobody was measuring.  Timestamps never leave
        # the article they belong to, so two different articles can never be
        # added up into one stability claim.
        carrying = [entry for entry in entries if entry.candidate_ids.get(field)]
        if len(carrying) > 1:
            moments = sorted(
                datetime.fromisoformat(entry.observed_at) for entry in carrying
            )
            span = round((moments[-1] - moments[0]).total_seconds(), 3)
            repeat_spans.append(
                {
                    "article_url": url,
                    "first_observed_at": moments[0].isoformat(),
                    "last_observed_at": moments[-1].isoformat(),
                    "span_seconds": span,
                    "observation_count": len(carrying),
                    "attempts": attempts,
                    "ids": ids,
                    "one_identifier": len(ids) == 1,
                    "meets_required_span": span >= required_span_seconds
                    and len(ids) == 1,
                }
            )
        if len(ids) <= 1:
            continue
        detail = {
            "article_url": url,
            "titles": sorted({entry.title for entry in entries}),
            "ids": ids,
            "attempts": attempts,
            "positions": positions,
            "tickers": tickers,
        }
        unstable.append(detail)
        # "Ticker-scoped" is a specific claim: one identifier per ticker,
        # and a different one per ticker.  An article that is unstable
        # *within* one ticker is not evidence for it.
        per_ticker = {
            ticker: {
                entry.candidate_ids[field]
                for entry in entries
                if entry.ticker == ticker and entry.candidate_ids.get(field)
            }
            for ticker in tickers
        }
        settled = [values for values in per_ticker.values() if len(values) == 1]
        if (
            len(tickers) > 1
            and len(settled) == len(per_ticker)
            and len({next(iter(values)) for values in settled}) > 1
        ):
            cross_ticker_divergent.append(detail)
        # Likewise for position: one identifier per slot, a different one
        # per slot.
        per_position = {
            position: {
                entry.candidate_ids[field]
                for entry in entries
                if entry.position == position and entry.candidate_ids.get(field)
            }
            for position in positions
        }
        placed = [values for values in per_position.values() if len(values) == 1]
        if (
            len(positions) > 1
            and len(placed) == len(per_position)
            and len({next(iter(values)) for values in placed}) == len(positions)
        ):
            position_varying.append(detail)

    colliding: list[Mapping[str, Any]] = []
    for identifier, urls in sorted(by_id.items()):
        if len(urls) <= 1:
            continue
        titles = sorted(
            {
                entry.title
                for entry in present
                if entry.candidate_ids[field] == identifier
            }
        )
        publishers = sorted(
            {
                entry.stored_source
                for entry in present
                if entry.candidate_ids[field] == identifier and entry.stored_source
            }
        )
        colliding.append(
            {
                "id": identifier,
                "article_urls": sorted(urls),
                "titles": titles,
                "distinct_titles": len(titles) > 1,
                "publishers": publishers,
            }
        )

    repeat_spans.sort(key=lambda row: (-row["span_seconds"], row["article_url"]))
    span_summary = _repeat_span_summary(repeat_spans, required_span_seconds)
    stability_span_met = span_summary["meeting_required_span"] > 0

    semantics, evidence = _classify_semantics(
        field=field,
        span_summary=span_summary,
        present_count=len(present),
        valid_count=len(valid),
        articles=len(by_article),
        repeated=repeated,
        multi_position=multi_position,
        cross_ticker=cross_ticker,
        unstable=unstable,
        position_varying=position_varying,
        cross_ticker_divergent=cross_ticker_divergent,
        colliding=colliding,
    )
    return CandidateFindings(
        field=field,
        valid_item_count=len(valid),
        present_count=len(present),
        presence_fraction=(round(len(present) / len(valid), 4) if valid else 0.0),
        distinct_ids=len(by_id),
        articles_observed=len(by_article),
        articles_repeated=repeated,
        articles_at_multiple_positions=multi_position,
        cross_ticker_articles=cross_ticker,
        repeat_spans=tuple(repeat_spans),
        repeat_span_summary=span_summary,
        stability_span_met=stability_span_met,
        unstable_articles=tuple(unstable),
        position_varying_articles=tuple(position_varying),
        cross_ticker_divergent_articles=tuple(cross_ticker_divergent),
        colliding_ids=tuple(colliding),
        semantics=semantics,
        evidence=tuple(evidence),
    )


def _repeat_span_summary(
    repeat_spans: Sequence[Mapping[str, Any]],
    required_span_seconds: float,
) -> dict[str, Any]:
    """Describe the repeated articles' spans, rather than just counting them.

    The longest span is what the strongest single piece of evidence is
    worth; the median says whether that piece was typical or lucky.  Both
    are ``None`` when nothing repeated, because zero would read as a
    measured span of no time rather than as an absence of measurement.

    The distribution covers every repeated article.  ``meeting_required_span``
    counts only those that held *one* identifier across the span: an article
    that carried two identifiers three hours apart was watched for three
    hours and demonstrated the opposite of stability.
    """

    spans = sorted(float(row["span_seconds"]) for row in repeat_spans)
    return {
        "required_span_seconds": required_span_seconds,
        "repeated_article_count": len(spans),
        "meeting_required_span": sum(
            1 for row in repeat_spans if row["meets_required_span"]
        ),
        "longest_seconds": spans[-1] if spans else None,
        "shortest_seconds": spans[0] if spans else None,
        "median_seconds": round(statistics.median(spans), 3) if spans else None,
    }


def _classify_semantics(
    *,
    field: str,
    span_summary: Mapping[str, Any],
    present_count: int,
    valid_count: int,
    articles: int,
    repeated: int,
    multi_position: int,
    cross_ticker: int,
    unstable: Sequence[Mapping[str, Any]],
    position_varying: Sequence[Mapping[str, Any]],
    cross_ticker_divergent: Sequence[Mapping[str, Any]],
    colliding: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    """Name the strongest verdict the observations actually support.

    The order matters.  Absence beats everything; instability beats
    stability; a collision beats a clean stability record, because an
    identifier that is stable *and* shared is worse than one that is
    merely unstable.  "Unconfirmed" is a real verdict and not a polite way
    of saying "fine": it is what an identifier gets when nothing repeated
    and so nothing was tested.

    A collision is never explained away here.  Article identity in this
    study is the canonical URL, so one identifier on two canonical URLs is
    two articles sharing an identifier, and the headlines have no vote:
    recurring and templated headlines are exactly how a real collision
    looks.  Deciding that two URLs are one article would take a URL-alias
    rule defined and defended on its own evidence, and there is no such
    rule here.
    """

    evidence = [
        f"{present_count}/{valid_count} valid items carried {field}",
        f"{articles} distinct articles observed",
        f"{repeated} articles observed in more than one attempt",
        f"{span_summary['repeated_article_count']} articles observed more than "
        f"once carrying {field}, "
        f"{span_summary['meeting_required_span']} of them at least "
        f"{span_summary['required_span_seconds']:g}s apart",
        f"{multi_position} articles observed at more than one response position",
        f"{cross_ticker} articles observed under more than one ticker",
    ]
    if present_count == 0:
        return "absent", evidence
    if unstable:
        evidence.append(f"{len(unstable)} articles carried more than one {field}")
        if len(cross_ticker_divergent) == len(unstable):
            return "ticker_scoped", evidence
        if len(position_varying) == len(unstable):
            return "response_position_scoped", evidence
        return "unstable", evidence
    if colliding:
        shared_publisher = all(len(entry["publishers"]) == 1 for entry in colliding)
        differing_titles = any(entry["distinct_titles"] for entry in colliding)
        evidence.append(
            f"{len(colliding)} identifiers were shared by more than one "
            "canonical URL, which is this study's article identity"
        )
        # "Publisher-scoped" is the narrower claim, and it needs headlines
        # that differ to establish that the articles differ.  Without them
        # the finding is still a collision -- it is simply a collision this
        # observation cannot attribute to a publisher's numbering.
        if differing_titles and shared_publisher:
            return "publisher_scoped", evidence
        if not differing_titles:
            evidence.append(
                "the shared identifiers carried one headline across those "
                "URLs, which recurring and templated coverage does too; it "
                "is not evidence that the URLs are one article"
            )
        return "colliding", evidence
    if repeated == 0 and cross_ticker == 0 and multi_position == 0:
        return "article_scoped_unconfirmed", evidence
    return "article_scoped", evidence


# -- Yahoo source strings ------------------------------------------------


def summarize_yahoo_sources(
    observations: Sequence[YahooItemObservation],
) -> list[dict[str, Any]]:
    """Group observations by the exact string ingestion would have stored.

    Grouped by the *stored* string, not by the raw publisher name: two raw
    names that normalize to one stored source are one source as far as the
    database is concerned, and I5's map keys off the stored string.
    """

    grouped: dict[str, list[YahooItemObservation]] = defaultdict(list)
    for entry in observations:
        if entry.stored_source:
            grouped[entry.stored_source].append(entry)
    summary = []
    for source, entries in sorted(grouped.items()):
        example = min(
            entries, key=lambda item: (item.attempt, item.ticker, item.position)
        )
        summary.append(
            {
                "stored_source": source,
                "raw_publisher": example.raw_publisher,
                "publisher_field": example.publisher_field,
                "provider_display_names": sorted(
                    {
                        e.provider_display_name
                        for e in entries
                        if e.provider_display_name
                    }
                ),
                "provider_source_ids": sorted(
                    {e.provider_source_id for e in entries if e.provider_source_id}
                ),
                "provider_urls": sorted(
                    {e.provider_url for e in entries if e.provider_url}
                ),
                "article_hosts": sorted(
                    {host for host in (_host(e.canonical_url) for e in entries) if host}
                ),
                "observation_count": len(entries),
                "distinct_article_count": len(
                    {e.canonical_url for e in entries if e.canonical_url}
                ),
                "tickers": sorted({e.ticker for e in entries}),
                "example": {
                    "ticker": example.ticker,
                    "title": example.title,
                    "canonical_url": example.canonical_url,
                    "candidate_ids": dict(example.candidate_ids),
                },
            }
        )
    return summary


def summarize_invalid_yahoo(
    observations: Sequence[YahooItemObservation],
) -> list[dict[str, Any]]:
    """Report the records that would have been stored as invalid evidence.

    These carry ``yahoo:<ticker>`` rather than a publisher, which is the
    one place ``phase0/yahoo.py`` already writes an ``external_id`` -- so
    an unusable record is worth reporting even when the count is zero.
    """

    grouped: dict[str, list[YahooItemObservation]] = defaultdict(list)
    for entry in observations:
        if not entry.valid:
            grouped[entry.ticker].append(entry)
    return [
        {
            "stored_source": f"yahoo:{ticker}",
            "count": len(entries),
            "errors": sorted(
                {e.validation_error for e in entries if e.validation_error}
            ),
            "candidate_ids_present": sorted(
                {
                    field
                    for e in entries
                    for field in CANDIDATE_FIELDS
                    if e.candidate_ids.get(field)
                }
            ),
        }
        for ticker, entries in sorted(grouped.items())
    ]


# -- RSS observation -----------------------------------------------------


class _StubResponse:
    """The two attributes ``_normalized_entries`` reads off a response.

    Feeding the real response through would work too; this exists so the
    offline tests can drive the production projection with no network and
    no ``requests`` object at all.
    """

    def __init__(self, status_code: int, headers: Mapping[str, str]) -> None:
        self.status_code = status_code
        self.headers = dict(headers)


def observation_fetcher(feeds_path: str | Path, aliases_path: str | Path) -> RSSFetcher:
    """Build an ``RSSFetcher`` that has no repository to write to.

    ``None`` is deliberate rather than a mock: only ``fetch`` and its
    settlement helpers touch ``self.repository``, so a tool that reaches
    ``_normalized_entries`` and nothing else cannot silently start
    persisting -- it would raise instead.
    """

    return RSSFetcher(
        None,  # type: ignore[arg-type]
        feeds_path=feeds_path,
        aliases_path=aliases_path,
        get=_refuse_network,
    )


def _refuse_network(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("the observation fetcher must not issue its own requests")


def _host(url: str | None) -> str | None:
    if not url:
        return None
    return (urlsplit(url).hostname or "").lower() or None


def observe_rss_response(
    fetcher: RSSFetcher,
    feed: Mapping[str, Any],
    body: bytes,
    *,
    response_url: str,
    status_code: int = 200,
    headers: Mapping[str, str] | None = None,
    attempt: int,
    observed_at: str,
) -> list[RSSEntryObservation]:
    """Observe the exact source strings ``RSSFetcher`` would have stored.

    ``parse_feed`` and ``_normalized_entries`` are the production path; the
    only thing this function adds is the reduction to a committable record.
    In particular the resolved host is read back *off the stored string*
    rather than recomputed, so the artifact cannot disagree with itself.
    """

    parsed = parse_feed(
        body,
        feed_url=response_url,
        expected_format=feed.get("format"),
    )
    entries = fetcher._normalized_entries(
        parsed,
        feed=feed,
        source=f"rss:{feed['id']}",
        snapshot_id=0,
        response=_StubResponse(status_code, headers or {}),
        response_url=response_url,
        checked_at=observed_at,
    )
    observations = []
    for entry in entries:
        stored_source = str(entry["source"])
        # ``_normalized_entries`` falls back to the *feed* source for
        # unusable evidence.  Detected by the status it set, not by
        # comparing the string to the feed id: a publisher host that
        # happened to equal a feed id would otherwise be misreported.
        resolved = (
            None
            if entry["ingest_status"] == "invalid"
            else stored_source.split("rss:", 1)[1]
        )
        observations.append(
            RSSEntryObservation(
                attempt=attempt,
                observed_at=observed_at,
                feed_id=str(feed["id"]),
                entry_link=_redact(entry.get("url")),
                resolved_host=resolved,
                stored_source=stored_source,
                canonical_url=_redact(entry["canonical_url"]) or "",
                external_id=_redact(str(entry["external_id"])) or "",
                title=_redact(entry.get("title")),
                ingest_status=str(entry["ingest_status"]),
            )
        )
    return observations


def summarize_rss_sources(
    observations: Sequence[RSSEntryObservation],
) -> list[dict[str, Any]]:
    """Group feed entries by the exact string ingestion would have stored."""

    grouped: dict[str, list[RSSEntryObservation]] = defaultdict(list)
    for entry in observations:
        grouped[entry.stored_source].append(entry)
    summary = []
    for source, entries in sorted(grouped.items()):
        example = min(
            entries, key=lambda item: (item.attempt, item.feed_id, item.title or "")
        )
        summary.append(
            {
                "stored_source": source,
                "resolved_host": example.resolved_host,
                "is_feed_scoped_fallback": example.resolved_host is None,
                "feed_ids": sorted({e.feed_id for e in entries}),
                "observation_count": len(entries),
                "distinct_article_count": len({e.canonical_url for e in entries}),
                "ingest_statuses": sorted({e.ingest_status for e in entries}),
                "example": {
                    "feed_id": example.feed_id,
                    "title": example.title,
                    "entry_link": example.entry_link,
                    "canonical_url": example.canonical_url,
                    "external_id": example.external_id,
                },
            }
        )
    return summary


# -- Cross-source equivalence --------------------------------------------

#: The four verdicts, and what each one is allowed to rest on.
EQUIVALENCE_VERDICTS = {
    "CONFIRMED": (
        "the same article, or the same publisher host, was observed under "
        "both source strings"
    ),
    "LIKELY_BUT_NOT_PROVEN": (
        "the names reduce to the same comparable token, but no article and "
        "no provider-declared host connects them"
    ),
    "NOT_EQUIVALENT": (
        "the names look related, but the provider itself declares a " "different host"
    ),
    "UNKNOWN": "no observed signal relates the two strings",
}


def comparable_token(value: str) -> str:
    """Reduce a publisher name or host to a token for *proposing* a pair.

    This is candidate generation for human review and nothing else.  It
    never produces a mapping, never appears on the CONFIRMED path, and is
    not the publisher canonicalization I5 will implement -- that one is an
    explicit reviewed table, and this is the thing that tells a reviewer
    which rows are worth writing.
    """

    return "".join(character for character in value.lower() if character.isalnum())


def _host_tokens(host: str) -> set[str]:
    """Every token a host could plausibly be recognized by.

    Both the whole host and the host minus its final label are offered,
    because "marketwatch.com" and "MarketWatch" are the same publisher
    while "marketwatch" and "marketwatch.com" are not the same *string*.
    Offering both is what keeps the suggestion honest: it widens what a
    reviewer is asked about, and narrows nothing automatically.
    """

    labels = [label for label in host.split(".") if label and label != "www"]
    tokens = {comparable_token("".join(labels))}
    if len(labels) > 1:
        tokens.add(comparable_token("".join(labels[:-1])))
    return {token for token in tokens if token}


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def equivalence_findings(
    yahoo_sources: Sequence[Mapping[str, Any]],
    rss_sources: Sequence[Mapping[str, Any]],
    rss_observations: Sequence[RSSEntryObservation],
) -> tuple[list[dict[str, Any]], int]:
    """Classify every Yahoo/RSS source pair the observations say something about.

    Returns the pairs carrying a verdict and the number of pairs left
    ``UNKNOWN``; the unknown pairs are counted rather than listed because
    they are the whole cross product minus the interesting part, and a
    findings table nobody reads is not a finding.
    """

    rss_articles: dict[str, set[str]] = defaultdict(set)
    for entry in rss_observations:
        if entry.resolved_host:
            rss_articles[entry.stored_source].add(entry.canonical_url)

    findings: list[dict[str, Any]] = []
    unknown = 0
    for yahoo in yahoo_sources:
        yahoo_source = str(yahoo["stored_source"])
        publisher = (
            yahoo_source.split("yahoo:", 1)[1] if ":" in yahoo_source else yahoo_source
        )
        yahoo_token = comparable_token(publisher)
        yahoo_hosts = {_strip_www(host) for host in yahoo["article_hosts"]}
        declared = {_strip_www(host) for host in yahoo["provider_source_ids"]}
        for rss in rss_sources:
            if rss["is_feed_scoped_fallback"]:
                continue
            rss_source = str(rss["stored_source"])
            host = _strip_www(str(rss["resolved_host"]))
            evidence: list[str] = []
            verdict = "UNKNOWN"

            shared_articles = sorted(
                rss_articles[rss_source] & _yahoo_article_urls(yahoo)
            )
            if shared_articles:
                verdict = "CONFIRMED"
                evidence.append(
                    f"the same article was observed under both sources: "
                    f"{shared_articles[0]}"
                )
            elif host in yahoo_hosts:
                verdict = "CONFIRMED"
                evidence.append(
                    f"a Yahoo article stored under {yahoo_source!r} has its "
                    f"canonical URL on {host}, which is the host "
                    f"{rss_source!r} is keyed by"
                )
            elif host in declared:
                verdict = "CONFIRMED"
                evidence.append(
                    f"Yahoo declares content.provider.sourceId={host!r} for "
                    f"{yahoo_source!r}, matching {rss_source!r}"
                )
            elif yahoo_token and yahoo_token in _host_tokens(host):
                if declared and not (declared & {host}):
                    verdict = "NOT_EQUIVALENT"
                    evidence.append(
                        f"the names reduce to {yahoo_token!r}, but Yahoo "
                        f"declares sourceId {sorted(declared)} for "
                        f"{yahoo_source!r}, not {host!r}"
                    )
                else:
                    verdict = "LIKELY_BUT_NOT_PROVEN"
                    evidence.append(
                        f"both names reduce to {yahoo_token!r}; no shared "
                        f"article and no provider-declared host was observed"
                    )

            if verdict == "UNKNOWN":
                unknown += 1
                continue
            findings.append(
                {
                    "yahoo_source": yahoo_source,
                    "rss_source": rss_source,
                    "verdict": verdict,
                    "evidence": evidence,
                }
            )
    findings.sort(
        key=lambda row: (row["verdict"], row["yahoo_source"], row["rss_source"])
    )
    return findings, unknown


def _yahoo_article_urls(yahoo: Mapping[str, Any]) -> set[str]:
    urls = yahoo.get("article_urls")
    if urls:
        return {str(url) for url in urls}
    example = str(yahoo["example"].get("canonical_url") or "")
    return {example} if example else set()


# -- Live collection -----------------------------------------------------


def default_yahoo_news(ticker: str) -> list[Any]:
    """Ask yfinance for one ticker's news, the way ingestion does.

    ``YahooFinanceFetcher._news`` additionally points yfinance's timezone
    cache at the database's directory.  There is no database here, so the
    library's default location is used instead; it changes where a cache
    file lands and nothing about the payload under observation.
    """

    import yfinance as yf

    return list(yf.Ticker(ticker).news or [])


def default_feed_get(url: str, *, timeout: float) -> Any:
    """Fetch a feed unconditionally, with ingestion's own User-Agent.

    No ``If-None-Match``/``If-Modified-Since``: those come from
    ``source_state``, and this tool does not read it.
    """

    return requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": RSS_USER_AGENT},
        stream=True,
    )


def enabled_feeds(feed_config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [feed for feed in feed_config["feeds"] if feed.get("enabled", True)]


def collect_attempt(
    *,
    attempt: int,
    tickers: Sequence[str],
    feeds: Sequence[Mapping[str, Any]],
    fetcher: RSSFetcher,
    news_for: Callable[[str], list[Any]],
    feed_get: Callable[..., Any],
    clock: Callable[[], str],
) -> tuple[AttemptRecord, list[YahooItemObservation], list[RSSEntryObservation]]:
    """Run one round of observations against both providers.

    A provider failure is recorded and moved past rather than raised: an
    attempt that reached one source and not the other is still evidence,
    and losing the whole window because one feed 500'd would be a poor
    trade.
    """

    started_at = clock()
    yahoo_result: dict[str, Any] = {}
    rss_result: dict[str, Any] = {}
    yahoo_observations: list[YahooItemObservation] = []
    rss_observations: list[RSSEntryObservation] = []

    for ticker in tickers:
        observed_at = clock()
        try:
            items = news_for(ticker)
        except Exception as exc:
            yahoo_result[ticker] = {"error": str(redact_secrets(str(exc)))}
            continue
        observations = observe_yahoo_response(
            ticker, items, attempt=attempt, observed_at=observed_at
        )
        yahoo_observations.extend(observations)
        yahoo_result[ticker] = {
            "item_count": len(observations),
            "valid_count": sum(1 for entry in observations if entry.valid),
            "observed_at": observed_at,
        }

    for feed in feeds:
        feed_id = str(feed["id"])
        observed_at = clock()
        timeout = float((feed.get("polling") or {}).get("timeout_seconds", 20))
        try:
            response = feed_get(feed["url"], timeout=timeout)
            response.raise_for_status()
            body = _read_response_body(response)
            response_url = str(getattr(response, "url", None) or feed["url"])
            observations = observe_rss_response(
                fetcher,
                feed,
                body,
                response_url=response_url,
                status_code=response.status_code,
                headers=response.headers,
                attempt=attempt,
                observed_at=observed_at,
            )
        except Exception as exc:
            rss_result[feed_id] = {"error": str(redact_secrets(str(exc)))}
            continue
        rss_observations.extend(observations)
        rss_result[feed_id] = {
            "status_code": response.status_code,
            "entry_count": len(observations),
            "body_bytes": len(body),
            "observed_at": observed_at,
        }

    record = AttemptRecord(
        attempt=attempt,
        started_at=started_at,
        yahoo=yahoo_result,
        rss=rss_result,
    )
    return record, yahoo_observations, rss_observations


def collect(
    *,
    tickers: Sequence[str],
    feeds: Sequence[Mapping[str, Any]],
    fetcher: RSSFetcher,
    attempts: int,
    interval_seconds: float,
    news_for: Callable[[str], list[Any]] = default_yahoo_news,
    feed_get: Callable[..., Any] = default_feed_get,
    clock: Callable[[], str] = utc_now,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[AttemptRecord], list[YahooItemObservation], list[RSSEntryObservation]]:
    """Observe both providers ``attempts`` times, separated in time.

    Separation is the point.  Two back-to-back requests can return one
    cached response, and an identifier that never had a chance to change
    proves nothing about stability.
    """

    records: list[AttemptRecord] = []
    yahoo_observations: list[YahooItemObservation] = []
    rss_observations: list[RSSEntryObservation] = []
    for attempt in range(1, attempts + 1):
        if attempt > 1 and interval_seconds > 0:
            sleep(interval_seconds)
        record, yahoo, rss = collect_attempt(
            attempt=attempt,
            tickers=tickers,
            feeds=feeds,
            fetcher=fetcher,
            news_for=news_for,
            feed_get=feed_get,
            clock=clock,
        )
        records.append(record)
        yahoo_observations.extend(yahoo)
        rss_observations.extend(rss)
    return records, yahoo_observations, rss_observations


# -- Artifact ------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _yfinance_version() -> str | None:
    try:
        import yfinance

        return str(getattr(yfinance, "__version__", None) or "unknown")
    except Exception:
        return None


def candidate_agreement(
    observations: Sequence[YahooItemObservation],
) -> list[dict[str, Any]]:
    """Report whether two candidate fields ever disagree.

    Two fields that carry the same value on every payload are one fact, not
    two, and wiring either would be the same change.  Two that diverge are
    a finding on their own -- and the divergence, not the agreement, is
    what has to be visible.
    """

    rows = []
    for left_index, left in enumerate(CANDIDATE_FIELDS):
        for right in CANDIDATE_FIELDS[left_index + 1 :]:
            both = [
                entry
                for entry in observations
                if entry.candidate_ids.get(left) and entry.candidate_ids.get(right)
            ]
            disagreed = [
                {
                    "title": entry.title,
                    left: entry.candidate_ids[left],
                    right: entry.candidate_ids[right],
                }
                for entry in both
                if entry.candidate_ids[left] != entry.candidate_ids[right]
            ]
            rows.append(
                {
                    "fields": [left, right],
                    "both_present_count": len(both),
                    "agreed_count": len(both) - len(disagreed),
                    "disagreements": disagreed[:5],
                }
            )
    return rows


def observation_span_seconds(records: Sequence[AttemptRecord]) -> float:
    """How far apart the first and last rounds of observation really were.

    Not the configured interval: an attempt that failed, was slow, or was
    interrupted moves the real span.

    This is context, not evidence.  It says how long the *run* lasted, and
    a run's length says nothing about an identifier -- a twenty-hour run
    can consist entirely of articles seen twice ten minutes apart.  The
    stability bar is decided per article, in
    :func:`summarize_candidate`, and this number is reported beside the
    verdict as ``observation_window_span_seconds`` without entering it.
    """

    started = sorted(record.started_at for record in records)
    if len(started) < 2:
        return 0.0
    first = datetime.fromisoformat(started[0])
    last = datetime.fromisoformat(started[-1])
    return round((last - first).total_seconds(), 3)


def _external_id_verdict(
    findings: Mapping[str, CandidateFindings],
    *,
    span_seconds: float = 0.0,
) -> dict[str, Any]:
    """Turn the per-field findings into the one decision I5 is waiting on.

    The verdict is about ``external_id`` as a whole, so it is the *best*
    candidate that decides it -- and the reasoning names which field that
    was, because "safe to implement" with no field named is not a decision
    anyone can act on.

    Decision G's stability bar is a gate on this verdict, and it is cleared
    only by a *repeated article* whose own observations were far enough
    apart.  ``span_seconds`` -- how long the run lasted -- is reported and
    never gated on.
    """

    order = {
        "article_scoped": 0,
        "article_scoped_unconfirmed": 1,
        "publisher_scoped": 2,
        "response_position_scoped": 3,
        "ticker_scoped": 3,
        "unstable": 3,
        "colliding": 3,
        "absent": 4,
    }
    # Ties break toward the field the codebase already reads.  ``id`` and
    # ``content.id`` can carry the same value on every payload, and
    # recommending the one that merely sorts first would be an arbitrary
    # choice dressed up as a finding -- ``_invalid_evidence`` already
    # reaches for ``id``, so that is the one a reader will expect.
    ranked = sorted(
        findings.values(),
        key=lambda entry: (
            order[entry.semantics],
            -entry.presence_fraction,
            CANDIDATE_FIELDS.index(entry.field),
        ),
    )
    best = ranked[0]
    if best.semantics == "absent":
        verdict = "UNKNOWN"
        reason = "no candidate field was present on any valid item"
    elif best.semantics in {
        "ticker_scoped",
        "response_position_scoped",
        "unstable",
        "colliding",
    }:
        verdict = "UNSAFE"
        reason = f"the best candidate {best.field!r} is {best.semantics}"
    elif best.semantics == "publisher_scoped":
        verdict = "UNSAFE"
        reason = (
            f"{best.field!r} identifies a publisher rather than an article, "
            "so it cannot key a raw item"
        )
    elif best.semantics == "article_scoped_unconfirmed":
        verdict = "UNKNOWN"
        reason = (
            f"{best.field!r} looks article-scoped, but no article repeated, "
            "so stability was never tested"
        )
    elif not best.stability_span_met:
        # Repeated, but not over enough time to have tested what decision G
        # asks.  UNKNOWN rather than PARTIALLY SAFE: the gap is missing
        # evidence, not a known limit.
        verdict = "UNKNOWN"
        longest = best.repeat_span_summary["longest_seconds"]
        if longest is None:
            measured = f"no article carried {best.field!r} more than once"
        else:
            measured = (
                f"the longest any one article was watched carrying it was "
                f"{longest:g}s"
            )
        reason = (
            f"{best.field!r} is article-scoped where it was seen, but "
            f"{measured}, short of the "
            f"{best.repeat_span_summary['required_span_seconds']:g}s apart "
            "decision G asks for, so stability over hours was never tested"
        )
    elif best.presence_fraction < 1.0:
        verdict = "PARTIALLY SAFE"
        reason = (
            f"{best.field!r} is stable and article-scoped where present, but "
            f"only {best.presence_fraction:.1%} of valid items carried it"
        )
    else:
        verdict = "SAFE TO IMPLEMENT"
        reason = (
            f"{best.field!r} was present on every valid item, unchanged "
            "across attempts, positions, and tickers, never shared by two "
            "canonical URLs, and held by "
            f"{best.repeat_span_summary['meeting_required_span']} articles "
            "across observations at least "
            f"{best.repeat_span_summary['required_span_seconds']:g}s apart"
        )
    return {
        "verdict": verdict,
        "field": best.field,
        "reason": reason,
        "semantics": best.semantics,
        "presence_fraction": best.presence_fraction,
        # What decision G's bar was actually measured against: the
        # repeated articles, one at a time.  ``observation_window_span``
        # sits alongside as context and is deliberately not what
        # ``meets_decision_g`` reads.
        "stability_window": {
            # Read off the finding, so the bar reported is the bar the
            # articles were actually measured against.
            "required_span_seconds": best.repeat_span_summary["required_span_seconds"],
            "repeated_article_count": best.repeat_span_summary[
                "repeated_article_count"
            ],
            "articles_meeting_required_span": best.repeat_span_summary[
                "meeting_required_span"
            ],
            "longest_repeat_span_seconds": best.repeat_span_summary["longest_seconds"],
            "median_repeat_span_seconds": best.repeat_span_summary["median_seconds"],
            "shortest_repeat_span_seconds": best.repeat_span_summary[
                "shortest_seconds"
            ],
            "meets_decision_g": best.stability_span_met,
            "observation_window_span_seconds": span_seconds,
        },
    }


DEFAULT_LIMITATIONS = (
    "Stability is only as strong as the span each repeated article was "
    "watched over, which is what the verdict is gated on. How long the run "
    "lasted is reported beside it as context and decides nothing: a long "
    "run of closely spaced repeats tests a short claim.",
    "Observation issues an unconditional GET, because reading a feed's "
    "stored ETag would mean reading source_state; a scheduled fetch sends "
    "conditional headers and can receive 304, which never appears here.",
    "Article identity is proxied by the canonical URL phase0 stores. Two "
    "URLs for one article would read as two articles, and one URL reused "
    "for two articles would read as one. One identifier on two canonical "
    "URLs is therefore counted as a collision, headlines included: "
    "suppressing it would need a URL-alias rule defined on its own "
    "evidence, and none is defined here.",
    "Only the five approved tickers and the enabled feeds were observed. "
    "A publisher that never appeared in this window is not evidence of "
    "absence.",
    "Every number here describes one observation window. Provider payload "
    "shapes have changed before -- the legacy uuid shape is why the code "
    "reads two -- so a verdict is a statement about now, not forever.",
)


def build_artifact(
    *,
    records: Sequence[AttemptRecord],
    yahoo_observations: Sequence[YahooItemObservation],
    rss_observations: Sequence[RSSEntryObservation],
    tickers: Sequence[str],
    feeds: Sequence[Mapping[str, Any]],
    feeds_path: str | Path,
    interval_seconds: float,
    generated_at: str,
    commit: str | None = None,
    dirty: bool | None = None,
    yfinance_version: str | None = None,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Assemble the committed record. Deterministic given its inputs."""

    findings = {
        field: summarize_candidate(yahoo_observations, field)
        for field in CANDIDATE_FIELDS
    }
    yahoo_sources = summarize_yahoo_sources(yahoo_observations)
    rss_sources = summarize_rss_sources(rss_observations)
    equivalences, unknown_pairs = equivalence_findings(
        yahoo_sources, rss_sources, rss_observations
    )
    feeds_bytes = Path(feeds_path).read_bytes()
    started = [record.started_at for record in records]
    span = observation_span_seconds(records)
    return {
        "schema": ARTIFACT_SCHEMA,
        "generated_at": generated_at,
        "window": {
            "started_at": min(started) if started else None,
            "ended_at": max(started) if started else None,
            "attempts": len(records),
            "interval_seconds": interval_seconds,
            "observed_span_seconds": span,
        },
        "code": {"commit": commit, "dirty": dirty},
        "environment": {
            "yfinance_version": yfinance_version,
            "python_version": python_version,
        },
        "feeds": {
            "config_path": str(feeds_path),
            "config_sha256": hashlib.sha256(feeds_bytes).hexdigest(),
            "enabled_feed_ids": sorted(str(feed["id"]) for feed in feeds),
        },
        "tickers": list(tickers),
        "attempts": [
            {
                "attempt": record.attempt,
                "started_at": record.started_at,
                "yahoo": dict(record.yahoo),
                "rss": dict(record.rss),
            }
            for record in records
        ],
        "yahoo": {
            "observations": [_observation_json(entry) for entry in yahoo_observations],
            "item_observation_count": len(yahoo_observations),
            "valid_item_count": sum(1 for e in yahoo_observations if e.valid),
            "invalid_item_count": sum(1 for e in yahoo_observations if not e.valid),
            "provider_id_candidates": {
                field: _findings_json(entry) for field, entry in findings.items()
            },
            "candidate_agreement": candidate_agreement(yahoo_observations),
            "external_id_verdict": _external_id_verdict(findings, span_seconds=span),
            "sources": yahoo_sources,
            "invalid_sources": summarize_invalid_yahoo(yahoo_observations),
        },
        "rss": {
            "entry_observation_count": len(rss_observations),
            "sources": rss_sources,
        },
        "equivalence": {
            "verdict_meanings": dict(EQUIVALENCE_VERDICTS),
            "findings": equivalences,
            "unknown_pair_count": unknown_pairs,
        },
        "semantics_meanings": dict(SEMANTICS),
        "limitations": list(DEFAULT_LIMITATIONS),
    }


def _observation_json(entry: YahooItemObservation) -> dict[str, Any]:
    """The minimal record a reader needs to recompute the conclusions.

    Committed because a verdict nobody can recompute is a verdict nobody
    can correct: every number under ``provider_id_candidates`` follows from
    these rows and the article identity (the canonical URL), and a
    methodology error found later can be re-run against them instead of
    re-run against the providers, which will have moved on.
    """

    return {
        "attempt": entry.attempt,
        "observed_at": entry.observed_at,
        "ticker": entry.ticker,
        "position": entry.position,
        "valid": entry.valid,
        "candidate_ids": dict(entry.candidate_ids),
        "canonical_url": entry.canonical_url,
        "stored_source": entry.stored_source,
        "title": entry.title,
    }


def _findings_json(entry: CandidateFindings) -> dict[str, Any]:
    return {
        "field": entry.field,
        "valid_item_count": entry.valid_item_count,
        "present_count": entry.present_count,
        "presence_fraction": entry.presence_fraction,
        "distinct_ids": entry.distinct_ids,
        "articles_observed": entry.articles_observed,
        "articles_repeated": entry.articles_repeated,
        "articles_at_multiple_positions": entry.articles_at_multiple_positions,
        "cross_ticker_articles": entry.cross_ticker_articles,
        "repeat_span_summary": dict(entry.repeat_span_summary),
        "stability_span_met": entry.stability_span_met,
        "repeat_spans": [dict(row) for row in entry.repeat_spans],
        "unstable_articles": [dict(row) for row in entry.unstable_articles],
        "position_varying_articles": [
            dict(row) for row in entry.position_varying_articles
        ],
        "cross_ticker_divergent_articles": [
            dict(row) for row in entry.cross_ticker_divergent_articles
        ],
        "colliding_ids": [dict(row) for row in entry.colliding_ids],
        "semantics": entry.semantics,
        "evidence": list(entry.evidence),
    }


# -- Rendering -----------------------------------------------------------


def _hours(seconds: float | None) -> str:
    return "n/a" if seconds is None else f"{seconds / 3600:.2f}h"


def _stability_lines(stability: Mapping[str, Any]) -> list[str]:
    """Say what the stability claim was measured on, article by article.

    A reader who is told only "watched over 20h" cannot tell whether any
    article was watched for twenty hours or whether every article was
    watched for ten minutes, twenty hours apart. So the repeated articles
    are reported, and the run's own span is reported separately and named
    as context.
    """

    required_hours = stability["required_span_seconds"] / 3600
    repeated = stability["repeated_article_count"]
    meeting = stability["articles_meeting_required_span"]
    lines = []
    if stability["meets_decision_g"]:
        lines.append(
            f"Decision G's bar is met by article, not by run: {meeting} of "
            f"{repeated} repeated articles carried the identifier across "
            f"observations at least {required_hours:.0f}h apart. The longest "
            f"was {_hours(stability['longest_repeat_span_seconds'])}, the "
            f"median repeated article "
            f"{_hours(stability['median_repeat_span_seconds'])}, the shortest "
            f"{_hours(stability['shortest_repeat_span_seconds'])}."
        )
    elif repeated == 0:
        lines.append(
            "**This verdict is provisional.** No article was observed twice "
            "carrying the identifier, so nothing was measured against "
            f"decision G's {required_hours:.0f}h bar."
        )
    else:
        lines.append(
            f"**This verdict is provisional.** {repeated} articles repeated, "
            "and none of them was watched for long enough: the longest was "
            f"{_hours(stability['longest_repeat_span_seconds'])} and the "
            f"median {_hours(stability['median_repeat_span_seconds'])}, "
            f"against decision G's {required_hours:.0f}h. Nothing here "
            "contradicts the verdict; no repeated article was watched long "
            "enough to have earned it."
        )
    lines.append("")
    lines.append(
        "The observation window itself spanned "
        f"{_hours(stability['observation_window_span_seconds'])}. That is "
        "context, not evidence: a run's length says nothing about an "
        "identifier, and it does not enter the verdict."
    )
    lines.append("")
    return lines


def render_markdown(artifact: Mapping[str, Any]) -> str:
    """Render the artifact for a reviewer, from the artifact and nothing else.

    Every claim below is read out of the JSON, so the prose and the data
    cannot drift apart: there is no second summary to keep in step.
    """

    window = artifact["window"]
    lines: list[str] = []
    lines.append("# I5 provider observation")
    lines.append("")
    lines.append(
        "Observational only. No ingestion behavior was changed, nothing was "
        "written to the Phase 0 database, and no publisher mapping is "
        "implemented here."
    )
    lines.append("")
    lines.append("## Window")
    lines.append("")
    lines.append(f"- Generated at: `{artifact['generated_at']}`")
    lines.append(
        f"- Observed from `{window['started_at']}` to `{window['ended_at']}` "
        f"across {window['attempts']} attempts "
        f"{window['interval_seconds']:g}s apart, a real span of "
        f"{window['observed_span_seconds']:g}s"
    )
    lines.append(f"- Code commit: `{artifact['code']['commit']}`")
    lines.append(f"- Working tree dirty: `{artifact['code']['dirty']}`")
    lines.append(
        f"- yfinance `{artifact['environment']['yfinance_version']}`, "
        f"Python `{artifact['environment']['python_version']}`"
    )
    lines.append(f"- Tickers: {', '.join(f'`{t}`' for t in artifact['tickers'])}")
    feed_list = ", ".join(f"`{feed}`" for feed in artifact["feeds"]["enabled_feed_ids"])
    lines.append(
        f"- Feeds: {feed_list} (`{artifact['feeds']['config_path']}` "
        f"sha256 `{artifact['feeds']['config_sha256'][:16]}`)"
    )
    lines.append("")

    verdict = artifact["yahoo"]["external_id_verdict"]
    lines.append("## Yahoo `external_id` verdict")
    lines.append("")
    lines.append(f"**{verdict['verdict']}** — field `{verdict['field']}`")
    lines.append("")
    lines.append(f"{verdict['reason']}.")
    lines.append("")
    lines.extend(_stability_lines(verdict["stability_window"]))
    lines.append("### Candidate fields")
    lines.append("")
    lines.append(
        "| field | present | coverage | distinct ids | articles | repeated | "
        "repeats over the bar | longest repeat | cross-ticker | unstable | "
        "collisions | semantics |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for field in CANDIDATE_FIELDS:
        row = artifact["yahoo"]["provider_id_candidates"][field]
        spans = row["repeat_span_summary"]
        lines.append(
            f"| `{field}` | {row['present_count']}/{row['valid_item_count']} | "
            f"{row['presence_fraction']:.0%} | {row['distinct_ids']} | "
            f"{row['articles_observed']} | {row['articles_repeated']} | "
            f"{spans['meeting_required_span']}/{spans['repeated_article_count']} | "
            f"{_hours(spans['longest_seconds'])} | "
            f"{row['cross_ticker_articles']} | {len(row['unstable_articles'])} | "
            f"{len(row['colliding_ids'])} | `{row['semantics']}` |"
        )
    lines.append("")
    lines.append("### Do the candidates agree?")
    lines.append("")
    for row in artifact["yahoo"]["candidate_agreement"]:
        left, right = row["fields"]
        if row["both_present_count"] == 0:
            lines.append(f"- `{left}` and `{right}` were never both present.")
            continue
        if row["disagreements"]:
            lines.append(
                f"- `{left}` and `{right}` disagreed on "
                f"{row['both_present_count'] - row['agreed_count']} of "
                f"{row['both_present_count']} items, e.g. "
                f"{row['disagreements'][0]}"
            )
        else:
            lines.append(
                f"- `{left}` and `{right}` carried the same value on all "
                f"{row['both_present_count']} items that had both."
            )
    lines.append("")
    for field in CANDIDATE_FIELDS:
        row = artifact["yahoo"]["provider_id_candidates"][field]
        lines.append(
            f"**`{field}`** — {artifact['semantics_meanings'][row['semantics']]}."
        )
        lines.append("")
        for item in row["evidence"]:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Yahoo source strings")
    lines.append("")
    lines.append(
        "| stored source | raw publisher | field | provider sourceId | "
        "article hosts | observations | tickers |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in artifact["yahoo"]["sources"]:
        lines.append(
            f"| `{row['stored_source']}` | {row['raw_publisher']} | "
            f"`{row['publisher_field']}` | "
            f"{', '.join(f'`{v}`' for v in row['provider_source_ids']) or '—'} | "
            f"{', '.join(f'`{v}`' for v in row['article_hosts']) or '—'} | "
            f"{row['observation_count']} | {', '.join(row['tickers'])} |"
        )
    lines.append("")
    invalid = artifact["yahoo"]["invalid_sources"]
    if invalid:
        lines.append("### Invalid Yahoo evidence")
        lines.append("")
        for row in invalid:
            lines.append(
                f"- `{row['stored_source']}` — {row['count']} records; "
                f"errors: {row['errors']}"
            )
        lines.append("")
    else:
        lines.append("No Yahoo record failed normalization in this window.")
        lines.append("")

    lines.append("## RSS source strings")
    lines.append("")
    lines.append(
        "| stored source | resolved host | feeds | observations | articles | statuses |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in artifact["rss"]["sources"]:
        host = (
            f"`{row['resolved_host']}`"
            if row["resolved_host"]
            else "feed-scoped fallback"
        )
        lines.append(
            f"| `{row['stored_source']}` | {host} | "
            f"{', '.join(row['feed_ids'])} | {row['observation_count']} | "
            f"{row['distinct_article_count']} | {', '.join(row['ingest_statuses'])} |"
        )
    lines.append("")
    lines.append("Example entries, showing that a feed's host is not the article's:")
    lines.append("")
    for row in artifact["rss"]["sources"]:
        example = row["example"]
        lines.append(
            f"- feed `{example['feed_id']}` → `{row['stored_source']}`  \n"
            f"  link: `{example['entry_link']}`  \n"
            f"  stored canonical URL: `{example['canonical_url']}`  \n"
            f"  external_id: `{example['external_id']}`"
        )
    lines.append("")

    lines.append("## Cross-source publisher equivalence")
    lines.append("")
    lines.append("| Yahoo source | RSS source | verdict | evidence |")
    lines.append("| --- | --- | --- | --- |")
    for row in artifact["equivalence"]["findings"]:
        lines.append(
            f"| `{row['yahoo_source']}` | `{row['rss_source']}` | "
            f"**{row['verdict']}** | {' '.join(row['evidence'])} |"
        )
    if not artifact["equivalence"]["findings"]:
        lines.append("| — | — | — | no pair carried any observed signal |")
    lines.append("")
    lines.append(
        f"{artifact['equivalence']['unknown_pair_count']} further pairs are "
        "`UNKNOWN`: nothing observed relates them, and they are counted "
        "rather than listed."
    )
    lines.append("")
    lines.append(
        "**No mapping is implemented from this table.** It is the input "
        "to I5's explicit reviewed publisher map, not the map."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in artifact["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="observe_phase0_providers",
        description=(
            "Observe what Yahoo and the configured RSS feeds actually send, "
            "and write a reviewable artifact. Read-only: nothing is persisted "
            "to the Phase 0 database."
        ),
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="comma-separated tickers to observe",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="how many separated rounds of observation to run",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=240.0,
        help="delay between rounds; back-to-back rounds prove nothing",
    )
    parser.add_argument("--feeds", default="config/feeds.yaml")
    parser.add_argument("--aliases", default="config/aliases.yaml")
    parser.add_argument("--out-dir", default="docs/observations")
    parser.add_argument(
        "--date",
        default=None,
        help="artifact date stamp; defaults to the UTC date of the run",
    )
    parser.add_argument("--skip-yahoo", action="store_true")
    parser.add_argument("--skip-rss", action="store_true")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the markdown instead of writing the artifact files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.attempts < 1:
        print("--attempts must be at least 1", file=sys.stderr)
        return 2

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.skip_yahoo:
        tickers = []
    feed_config = load_feed_config(args.feeds)
    feeds = [] if args.skip_rss else enabled_feeds(feed_config)
    fetcher = observation_fetcher(args.feeds, args.aliases)

    records, yahoo_observations, rss_observations = collect(
        tickers=tickers,
        feeds=feeds,
        fetcher=fetcher,
        attempts=args.attempts,
        interval_seconds=args.interval_seconds,
    )
    generated_at = utc_now()
    artifact = build_artifact(
        records=records,
        yahoo_observations=yahoo_observations,
        rss_observations=rss_observations,
        tickers=tickers,
        feeds=feeds,
        feeds_path=args.feeds,
        interval_seconds=args.interval_seconds,
        generated_at=generated_at,
        commit=_git("rev-parse", "HEAD"),
        dirty=bool(_git("status", "--porcelain")),
        yfinance_version=_yfinance_version(),
        python_version=platform.python_version(),
    )
    markdown = render_markdown(artifact)
    if args.stdout:
        print(markdown)
        return 0

    stamp = args.date or generated_at[:10]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"i5-provider-observation-{stamp}.json"
    md_path = out_dir / f"i5-provider-observation-{stamp}.md"
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(markdown, encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(artifact["yahoo"]["external_id_verdict"]["verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

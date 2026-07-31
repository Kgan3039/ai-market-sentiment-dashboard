"""Turn a :class:`~nlp.dedup.models.RawItem` into its comparison surface.

This module wires the neutral text utilities, the structural title
analyzer, the conservative URL rules, and the content fingerprint together.
It also carries the policies for the two fields that are easy to get
silently wrong: timestamps and tickers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re

from .content import content_identity_key, dedup_content_fingerprint
from .errors import DedupInputError
from .models import NormalizedItem, RawItem
from .structural import analyze_title, tokenize
from .text import display_text, normalize_source, require_optional_str
from .urls import clean_url, url_host, url_identity_key

#: Phase 0 symbols are plain equity tickers, optionally with a one-to-three
#: letter class or exchange suffix ("BRK.B").  Anything else — a
#: comma-separated list, a sentence, a symbol with whitespace — is an
#: upstream bug that must not silently become a partition of its own.
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{0,9}(?:[.\-][A-Z]{1,3})?$")

#: Absolute plausibility bounds for a publication timestamp.  They are
#: deliberately *not* relative to the current clock: a replay of a stored
#: day (AC-8) must produce the same stories next year as it does today, and
#: a wall-clock comparison would break that.  Clock-skew policy against
#: "now" belongs to ingestion, which knows its own fetch time.
MIN_PLAUSIBLE_PUBLISHED_AT = datetime(1990, 1, 1, tzinfo=timezone.utc)
MAX_PLAUSIBLE_PUBLISHED_AT = datetime(2100, 1, 1, tzinfo=timezone.utc)

_ZULU_PATTERN = re.compile(r"[Zz]$")


def normalize_item_id(value: str | int) -> str:
    """Return the string item identifier, rejecting empty or wrong types."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DedupInputError("item_id must be a string or an integer")
    text = str(value).strip()
    if not text:
        raise DedupInputError("item_id must be non-empty")
    return text


def normalize_ticker(value: str | None) -> str | None:
    """Return the validated upper-case symbol, or ``None`` when unassigned.

    Raises :class:`DedupInputError` for anything that is not a single
    well-formed symbol, including whitespace-containing and multi-ticker
    strings.  An empty or missing value means "the relevance filter did not
    assign this item", which is a legitimate state, not an error.
    """

    if value is None:
        return None
    if not isinstance(value, str):
        raise DedupInputError("ticker must be a string or None")
    stripped = value.strip()
    if not stripped:
        return None
    if any(character.isspace() for character in stripped):
        raise DedupInputError(f"ticker must be a single symbol: {value!r}")
    symbol = stripped.upper()
    if not _TICKER_PATTERN.match(symbol):
        raise DedupInputError(f"ticker is not a valid symbol: {value!r}")
    return symbol


def normalize_provider_item_id(value: str | None) -> str | None:
    """Return the feed's own item identifier, or ``None`` when absent."""

    text = require_optional_str(value, "provider_item_id")
    if text is None:
        return None
    collapsed = " ".join(text.split())
    return collapsed or None


def normalize_timestamp(
    value: datetime | str | None, *, field: str = "published_at"
) -> datetime | None:
    """Return a UTC-aware timestamp, or ``None`` when absent.

    Timezone-aware values are converted to UTC.  **Naive values are
    rejected**: no Phase 0 ingestion contract states which wall clock they
    would be in, and reading them as UTC would shift every merge window by
    the publisher's offset without anything failing loudly.  Timestamps
    outside :data:`MIN_PLAUSIBLE_PUBLISHED_AT` ..
    :data:`MAX_PLAUSIBLE_PUBLISHED_AT` are rejected as corrupt.
    """

    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(_ZULU_PATTERN.sub("+00:00", text))
        except ValueError as exc:
            raise DedupInputError(
                f"{field} is not a valid ISO-8601 timestamp: {value!r}"
            ) from exc
    else:
        raise DedupInputError(
            f"{field} must be a timezone-aware datetime, an ISO-8601 string "
            "with an offset, or None"
        )
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise DedupInputError(
            f"{field} must carry a timezone offset; naive timestamps are "
            f"rejected because Phase 0 does not define their wall clock: {value!r}"
        )
    utc = parsed.astimezone(timezone.utc)
    if not MIN_PLAUSIBLE_PUBLISHED_AT <= utc < MAX_PLAUSIBLE_PUBLISHED_AT:
        raise DedupInputError(f"{field} is outside the plausible range: {value!r}")
    return utc


def normalize_item(item: RawItem, universe: frozenset[str]) -> NormalizedItem:
    """Normalize one raw item into its deterministic comparison surface.

    A missing, malformed, or unsupported ticker is an error: Phase 0's
    relevance filter already excludes unassigned items from processing
    (spec section 2), so the core never has to invent a partition for them.
    """

    if not isinstance(item, RawItem):
        raise DedupInputError("items must be RawItem instances")
    item_id = normalize_item_id(item.item_id)
    title = require_optional_str(item.title, "title")
    description = require_optional_str(item.description, "description")
    source = require_optional_str(item.source, "source")
    url = require_optional_str(item.url, "url")
    raw_canonical_url = require_optional_str(item.canonical_url, "canonical_url")
    ticker = normalize_ticker(item.ticker)
    if ticker is None:
        raise DedupInputError(
            "ticker is required: the core does not process unassigned items"
        )
    if ticker not in universe:
        raise DedupInputError(f"ticker is outside the supported universe: {ticker}")

    structure = analyze_title(title, source)
    # The body goes through the same tokenizer as the title so that "$5
    # million" and "5 million" cannot collapse into one description key.
    normalized_description = " ".join(tokenize(display_text(description)))
    normalized_source = normalize_source(source)
    provider_item_id = normalize_provider_item_id(item.provider_item_id)
    provider_key = (
        f"{len(normalized_source)}:{normalized_source}\x1f{provider_item_id}"
        if normalized_source and provider_item_id
        else None
    )
    # A canonical URL is preferred, but a value that yields no identity key
    # (a non-HTTP scheme, an unparseable host) must not shadow a usable
    # ``url``.
    identity_key = url_identity_key(raw_canonical_url) or url_identity_key(url)
    title_key = structure.title_key
    return NormalizedItem(
        item_id=item_id,
        ticker=ticker,
        display_title=structure.display,
        title_key=title_key,
        structural_key=structure.structural_key,
        title_protected=structure.protected,
        uncertain_title=structure.uncertain,
        normalized_description=normalized_description,
        normalized_source=normalized_source,
        outlet=normalized_source or url_host(raw_canonical_url) or url_host(url),
        provider_namespace=normalized_source or None,
        provider_item_id=provider_item_id,
        provider_key=provider_key,
        canonical_url=clean_url(raw_canonical_url) or clean_url(url),
        url_key=identity_key,
        content_key=content_identity_key(title_key or "", normalized_description),
        content_fingerprint=dedup_content_fingerprint(
            title_key or "", normalized_description
        ),
        published_at=normalize_timestamp(item.published_at),
    )

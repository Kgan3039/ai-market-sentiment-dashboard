"""The approved Phase 0 ticker universe.

``docs/PHASE_0_SPEC.md`` section 2 fixes the universe at five symbols and
says explicitly that it is hardcoded.  This module is the single Python
statement of that fact; migration ``004`` mirrors it into the
``supported_tickers`` table so the database enforces it too and downstream
consumers can read the universe without importing this package.
"""

from __future__ import annotations

import re
from typing import Any

from .errors import Phase0ValidationError, UnsupportedTickerError


#: Symbol to display name, in the order the spec lists them.
TICKER_UNIVERSE: dict[str, str] = {
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "AAPL": "Apple",
    "META": "Meta Platforms",
}

SUPPORTED_TICKERS = frozenset(TICKER_UNIVERSE)

#: A single symbol and nothing else: no separators, no lists, no spaces.
_SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def normalize_ticker(
    value: Any,
    *,
    field: str = "ticker",
    optional: bool = False,
) -> str | None:
    """Return the canonical symbol, or raise.

    Normalization is deterministic: surrounding whitespace is stripped and
    the symbol is upper-cased.  Anything that is not then a single approved
    symbol — blank, malformed, or a multi-symbol string — is rejected.
    """

    if value is None:
        if optional:
            return None
        raise Phase0ValidationError(f"{field} is required")
    if isinstance(value, (bytes, bytearray)):
        raise Phase0ValidationError(f"{field} must be a string")
    normalized = str(value).strip().upper()
    if not normalized:
        # ``None`` means "this item matches no ticker", which the spec
        # allows.  A blank string is a malformed value, not that statement,
        # so it is rejected even where a ticker is optional.
        raise Phase0ValidationError(f"{field} must not be blank")
    if not _SYMBOL_PATTERN.match(normalized):
        raise UnsupportedTickerError(f"malformed {field}: {normalized!r}")
    if normalized not in SUPPORTED_TICKERS:
        raise UnsupportedTickerError(f"unsupported Phase 0 {field}: {normalized}")
    return normalized


def is_supported_ticker(value: Any) -> bool:
    """True when ``value`` normalizes to an approved Phase 0 symbol."""

    try:
        return normalize_ticker(value) is not None
    except Phase0ValidationError:
        return False


__all__ = [
    "SUPPORTED_TICKERS",
    "TICKER_UNIVERSE",
    "is_supported_ticker",
    "normalize_ticker",
]

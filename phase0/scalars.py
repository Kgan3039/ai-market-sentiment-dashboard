"""How a scalar column is protected depends on what the column means.

Redaction is the right answer for text that only explains something and the
wrong answer for text that *identifies* something.  Rewriting a provider
item id to ``[REDACTED]`` does not protect a credential that should never
have been there — it silently repoints the row, and the damage outlives
the incident.  So Phase 0 runs two policies, and every scalar that can
carry caller-supplied text is assigned to one of them explicitly.

**Policy A — diagnostics** (:func:`sanitize_diagnostic_scalar`): redact and
keep going.  A merge reason, a skip reason, a match reason, a method
rationale.  Nothing keys off these, so losing a substring costs nothing.

**Policy B — identity and configuration**
(:func:`validate_safe_identifier_scalar`): refuse the write.  A provider
namespace, a provider item id, a story key, a model name or revision.
These are joined on, cached against, and compared across runs; a quietly
altered one is worse than a rejected batch.
"""

from __future__ import annotations

from typing import Any

from .errors import Phase0ValidationError
from .redaction import contains_credential, redact_text


def sanitize_diagnostic_scalar(value: Any, field: str) -> str | None:
    """Policy A: redact credentials out of free-form diagnostic text."""

    if value is None:
        return None
    return redact_text(str(value))


def validate_safe_identifier_scalar(
    value: Any,
    field: str,
    *,
    max_length: int = 512,
) -> str | None:
    """Policy B: reject an identifier that carries credential material."""

    if value is None:
        return None
    text = str(value)
    if len(text) > max_length:
        raise Phase0ValidationError(f"{field} is longer than {max_length} characters")
    if "\n" in text or "\r" in text:
        raise Phase0ValidationError(f"{field} must be a single line")
    if contains_credential(text):
        raise Phase0ValidationError(
            f"{field} must not contain credential material; it identifies a "
            "row, so it is rejected rather than redacted"
        )
    return text


def require_safe_identifier_scalar(
    value: Any,
    field: str,
    *,
    max_length: int = 512,
) -> str:
    """Policy B for a column that is also ``NOT NULL``."""

    checked = validate_safe_identifier_scalar(value, field, max_length=max_length)
    if checked is None or not checked.strip():
        raise Phase0ValidationError(f"{field} is required")
    return checked


__all__ = [
    "require_safe_identifier_scalar",
    "sanitize_diagnostic_scalar",
    "validate_safe_identifier_scalar",
]

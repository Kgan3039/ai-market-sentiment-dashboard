"""Typed errors raised by the Phase 0 persistence layer.

Every message is redacted on construction, so a credential that reached an
input payload cannot escape through an exception string either.
"""

from __future__ import annotations

from .redaction import redact_secrets


class Phase0Error(RuntimeError):
    """Base error for the Phase 0 persistence layer."""

    def __init__(self, message: str = "", *args: object) -> None:
        super().__init__(redact_secrets(str(message)), *args)


class Phase0ValidationError(Phase0Error, ValueError):
    """Input rejected at the repository boundary.

    Subclasses :class:`ValueError` so existing callers that catch
    ``ValueError`` keep working after the typed hierarchy landed.
    """


class UnsupportedTickerError(Phase0ValidationError):
    """A ticker outside the approved Phase 0 universe."""


class Phase0IntegrityError(Phase0Error):
    """A relationship or lifecycle contract was violated."""


class Phase0MigrationError(Phase0Error):
    """A migration could not be applied, or history was rewritten."""


class StageKeyError(Phase0Error):
    """A stage key was not owned by the caller, or does not exist."""


__all__ = [
    "Phase0Error",
    "Phase0IntegrityError",
    "Phase0MigrationError",
    "Phase0ValidationError",
    "StageKeyError",
    "UnsupportedTickerError",
]

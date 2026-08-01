"""One validator for every number that acts as a quality gate.

Thresholds and floors are compared with ``<`` and ``>=``, and **NaN loses
every one of those comparisons silently**.  A gate checked against NaN does
not fail loudly; it passes, or it fails, depending on which way the
comparison happens to be written, and either way nobody is told.  Infinity
is the same problem with the opposite sign: ``--precision-floor inf`` fails
every run for a reason the message never explains, and ``-inf`` passes every
run for a reason nobody sees.

So no code path anywhere in :mod:`nlp.eval` may compare against a number it
has not put through :func:`validate_unit_interval` first.  ``argparse``'s
``type=float`` is not that check: it accepts ``nan``, ``inf`` and ``-inf``
happily.
"""

from __future__ import annotations

import math
from typing import Sequence


class GateValueError(ValueError):
    """A threshold, floor, or sweep point is not a usable probability."""


def validate_unit_interval(value: object, field: str) -> float:
    """Return ``value`` as a float in ``[0.0, 1.0]``, or raise.

    Rejects, in this order and with a distinct message for each: booleans
    (Python would otherwise treat ``True`` as ``1``), non-numbers, NaN,
    positive and negative infinity, and anything outside the closed unit
    interval.
    """

    if isinstance(value, bool):
        raise GateValueError(
            f"{field} must be a number, not a boolean: {value!r}; "
            "True would otherwise be read as 1.0"
        )
    if not isinstance(value, (int, float)):
        raise GateValueError(f"{field} must be a number, got {value!r}")
    number = float(value)
    if math.isnan(number):
        raise GateValueError(
            f"{field} must be a real number, got NaN; NaN loses every "
            "comparison silently, so a gate checked against it never fails "
            "for the reason you think"
        )
    if math.isinf(number):
        raise GateValueError(
            f"{field} must be finite, got {number}; an infinite gate passes "
            "or fails every run without explaining why"
        )
    if not 0.0 <= number <= 1.0:
        raise GateValueError(
            f"{field} must lie in [0.0, 1.0], got {number}; precision, "
            "recall and cosine similarity are all probabilities"
        )
    return number


def validate_optional_unit_interval(value: object, field: str) -> float | None:
    """Return ``None`` unchanged, otherwise validate as a unit interval."""

    if value is None:
        return None
    return validate_unit_interval(value, field)


def validate_thresholds(thresholds: Sequence[object]) -> tuple[float, ...]:
    """Return the sorted, validated, distinct sweep points.

    Every value goes through :func:`validate_unit_interval`, so a sweep
    cannot contain a row that was scored against a comparison nobody can
    reason about.
    """

    values = [
        validate_unit_interval(entry, f"sweep threshold[{index}]")
        for index, entry in enumerate(thresholds)
    ]
    if not values:
        raise GateValueError("a sweep needs at least one threshold")
    if len(set(values)) != len(values):
        raise GateValueError("sweep thresholds must be distinct")
    return tuple(sorted(values))

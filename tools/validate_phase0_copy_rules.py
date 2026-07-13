#!/usr/bin/env python3
"""Validate Phase 0 copy-deck guardrail files without implementing a linter."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPY_DECK_PATH = ROOT / "docs" / "phase0_copy_deck.md"
BANNED_PHRASES_PATH = ROOT / "config" / "banned_phrases.txt"
REQUIRED_COPY = (
    "Ticker Narratives",
    "Themes dominating current coverage",
    "Key narratives around today’s move",
    "Data as of HH:MM",
    "Summary unavailable — source stories are still available",
    "AI-generated from cited sources. Informational only — not investment advice.",
)
REQUIRED_CATEGORIES = {
    "causal_price",
    "prediction",
    "advisory",
    "unsupported_certainty",
    "model_confidence",
}
VALID_MATCH_TYPES = {"literal", "regex"}


def load_copy_deck() -> str:
    copy_deck = COPY_DECK_PATH.read_text(encoding="utf-8").strip()
    if not copy_deck:
        raise ValueError("copy deck must be non-empty")
    return copy_deck


def load_rules() -> list[tuple[str, str, str]]:
    rules: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(BANNED_PHRASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"rule on line {line_number} must have three tab-separated columns")
        category, match_type, pattern = parts
        if not category or match_type not in VALID_MATCH_TYPES or not pattern:
            raise ValueError(f"invalid rule on line {line_number}")
        if match_type == "regex":
            re.compile(pattern, re.IGNORECASE)
        rules.append((category, match_type, pattern))
    if not rules:
        raise ValueError("banned phrase file must contain at least one rule")
    return rules


def rule_matches(text: str, match_type: str, pattern: str) -> bool:
    if match_type == "literal":
        expression = rf"(?<![A-Za-z0-9]){re.escape(pattern)}(?![A-Za-z0-9])"
    else:
        expression = pattern
    return bool(re.search(expression, text, re.IGNORECASE))


def detected_categories(text: str, rules: list[tuple[str, str, str]]) -> set[str]:
    return {category for category, match_type, pattern in rules if rule_matches(text, match_type, pattern)}


def validate() -> None:
    copy_deck = load_copy_deck()
    missing_copy = [phrase for phrase in REQUIRED_COPY if phrase not in copy_deck]
    if missing_copy:
        raise ValueError(f"copy deck is missing required phrases: {missing_copy}")
    rules = load_rules()
    categories = {category for category, _, _ in rules}
    missing_categories = REQUIRED_CATEGORIES - categories
    if missing_categories:
        raise ValueError(f"banned phrase file is missing categories: {sorted(missing_categories)}")


if __name__ == "__main__":
    try:
        validate()
    except (OSError, ValueError, re.error) as error:
        print(f"Phase 0 copy-rule validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 0 copy deck and banned-language rules are valid.")

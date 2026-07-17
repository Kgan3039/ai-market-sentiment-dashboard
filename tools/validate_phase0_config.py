#!/usr/bin/env python3
"""Validate Phase 0 RSS and ticker-mapping configuration.

This is intentionally a configuration check, not the RSS fetcher or production
relevance filter owned by issue #62. It also exercises representative negative
examples so future config edits do not turn known ambiguous language into a
silent ticker assignment.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FEEDS_PATH = ROOT / "config" / "feeds.yaml"
ALIASES_PATH = ROOT / "config" / "aliases.yaml"
REQUIRED_TICKERS = {"TSLA", "NVDA", "AMD", "AAPL", "META"}
REQUIRED_FEED_FIELDS = {"id", "name", "url", "enabled", "intended_role", "expected_fields", "polling", "notes"}
REQUIRED_EXPECTED_FIELDS = {"title", "url", "description", "published_at"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return document


def phrase_present(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", text, re.IGNORECASE))


def matched_tickers(text: str, aliases: dict[str, Any]) -> set[str]:
    """Reference the documented mapping rules for negative validation cases."""
    matches: set[str] = set()
    for entry in aliases["tickers"]:
        ticker = entry["ticker"]
        if any(phrase_present(text, excluded) for excluded in entry["exclusion_terms"]):
            continue
        strong_terms = [ticker, entry["cashtag"], entry["official_company_name"], *entry["strong_aliases"]]
        direct_terms = [term for term in strong_terms if term not in entry["context_required_aliases"]]
        # Case-insensitive matching makes META and AMD indistinguishable from
        # ordinary words/acronyms. Treat their bare symbols as contextual when
        # the config marks the same spelling as a contextual alias.
        contextual_spellings = {term.casefold() for term in entry["context_required_aliases"]}
        direct_terms = [term for term in direct_terms if term.casefold() not in contextual_spellings]
        if any(phrase_present(text, term) for term in direct_terms):
            matches.add(ticker)
            continue
        if any(phrase_present(text, term) for term in entry["context_required_aliases"]):
            if any(phrase_present(text, context) for context in entry["context_terms"]):
                matches.add(ticker)
    return matches


def match_outcome(text: str, aliases: dict[str, Any]) -> tuple[str, set[str]]:
    """Classify a reference match without making an ambiguous assignment."""
    matches = matched_tickers(text, aliases)
    if len(matches) > 1:
        return "ambiguous", matches
    if matches:
        return "assigned", matches
    return "unmatched", matches


def validate() -> None:
    feeds = load_yaml(FEEDS_PATH)
    aliases = load_yaml(ALIASES_PATH)

    feed_entries = feeds.get("feeds")
    if not isinstance(feed_entries, list) or not feed_entries:
        raise ValueError("feeds.yaml must contain a non-empty feeds list")
    feed_ids = [entry.get("id") for entry in feed_entries]
    if len(feed_ids) != len(set(feed_ids)):
        raise ValueError("feed IDs must be unique")
    for entry in feed_entries:
        missing = REQUIRED_FEED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"feed {entry.get('id', '<unknown>')} missing fields: {sorted(missing)}")
        expected_fields = entry["expected_fields"]
        missing_expected = REQUIRED_EXPECTED_FIELDS - expected_fields.keys()
        if missing_expected:
            raise ValueError(f"feed {entry['id']} missing expected fields: {sorted(missing_expected)}")
        if not entry["url"].startswith("https://"):
            raise ValueError(f"feed {entry['id']} must use HTTPS")
        if not isinstance(entry["intended_role"], str) or not entry["intended_role"].strip():
            raise ValueError(f"feed {entry['id']} must state an intended role")

    ticker_entries = aliases.get("tickers")
    if not isinstance(ticker_entries, list):
        raise ValueError("aliases.yaml must contain a tickers list")
    ticker_names = {entry.get("ticker") for entry in ticker_entries}
    if ticker_names != REQUIRED_TICKERS:
        raise ValueError(f"expected exactly {sorted(REQUIRED_TICKERS)}, got {sorted(ticker_names)}")
    for entry in ticker_entries:
        for key in ("cashtag", "official_company_name", "strong_aliases"):
            if not entry.get(key):
                raise ValueError(f"ticker {entry['ticker']} has empty {key}")
        if not isinstance(entry.get("exclusion_terms"), list):
            raise ValueError(f"ticker {entry['ticker']} must provide an exclusion_terms list")

    false_positive_cases = {
        "A recipe for apple pie and fresh apple cider.": set(),
        "Researchers completed a meta-analysis of clinical trials.": set(),
        "The clinic treats age-related macular degeneration.": set(),
        "Students built a Tesla coil for the physics fair.": set(),
        "Elon Musk discussed his other companies.": set(),
        "The apple harvest is underway.": set(),
        "A meta framework improves documentation.": set(),
        "The AMD department published its annual report.": set(),
    }
    for text, expected in false_positive_cases.items():
        actual = matched_tickers(text, aliases)
        if actual != expected:
            raise ValueError(f"false-positive check failed for {text!r}: expected {expected}, got {actual}")


if __name__ == "__main__":
    try:
        validate()
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"Phase 0 configuration validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Phase 0 RSS and ticker-alias configuration is valid.")

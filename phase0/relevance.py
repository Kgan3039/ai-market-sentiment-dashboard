"""Ticker relevance matching for general RSS items."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


GENERIC_CONTEXT_ONLY = {
    "earnings",
    "revenue",
    "share",
    "shares",
    "stock",
    "stocks",
}


@dataclass(frozen=True)
class RelevanceResult:
    ticker: str | None
    matches: tuple[str, ...]
    ambiguous: bool


def load_alias_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("tickers"), list):
        raise ValueError("alias config must contain a tickers list")
    return config


def _contains_phrase(text: str, phrase: str) -> bool:
    # Cashtags need a boundary before '$'; ordinary phrases use alphanumeric
    # boundaries so AMD does not match metadata and Meta does not match metaverse.
    escaped = re.escape(str(phrase).strip())
    return bool(
        re.search(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", text, re.IGNORECASE)
    )


def _matches_ticker(text: str, rule: Mapping[str, Any]) -> bool:
    exclusions = rule.get("exclusion_terms") or []
    if any(_contains_phrase(text, phrase) for phrase in exclusions):
        return False

    contextual_spellings = {
        str(alias).casefold() for alias in (rule.get("context_required_aliases") or [])
    }
    strong = [
        rule.get("ticker"),
        rule.get("cashtag"),
        rule.get("official_company_name"),
        *(rule.get("strong_aliases") or []),
    ]
    strong = [
        phrase
        for phrase in strong
        if phrase
        and str(phrase).casefold() not in contextual_spellings
        and str(phrase).casefold() not in GENERIC_CONTEXT_ONLY
    ]
    if any(_contains_phrase(text, phrase) for phrase in strong):
        return True

    context_terms = rule.get("context_terms") or []
    has_context = any(_contains_phrase(text, term) for term in context_terms)
    return has_context and any(
        _contains_phrase(text, alias)
        for alias in (rule.get("context_required_aliases") or [])
    )


def match_ticker(
    title: str, description: str, alias_config: Mapping[str, Any]
) -> RelevanceResult:
    text = f"{title or ''}\n{description or ''}"
    matches = tuple(
        str(rule["ticker"]).upper()
        for rule in alias_config.get("tickers", [])
        if _matches_ticker(text, rule)
    )
    return RelevanceResult(
        ticker=matches[0] if len(matches) == 1 else None,
        matches=matches,
        ambiguous=len(matches) > 1,
    )

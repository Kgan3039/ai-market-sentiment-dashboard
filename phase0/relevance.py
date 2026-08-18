"""Ticker relevance matching for general RSS items."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .repository import SUPPORTED_TICKERS


GENERIC_CONTEXT_ONLY = {
    "earnings",
    "revenue",
    "share",
    "shares",
    "stock",
    "stocks",
}
ALIAS_LIST_FIELDS = {
    "strong_aliases",
    "context_required_aliases",
    "context_terms",
    "exclusion_terms",
}


@dataclass(frozen=True)
class RelevanceResult:
    ticker: str | None
    matches: tuple[str, ...]
    ambiguous: bool
    evidence: tuple[dict[str, Any], ...]


def load_alias_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("tickers"), list):
        raise ValueError("alias config must contain a tickers list")
    seen: set[str] = set()
    for index, rule in enumerate(config["tickers"]):
        if not isinstance(rule, dict):
            raise ValueError(f"tickers[{index}] must be an object")
        ticker = rule.get("ticker")
        if not isinstance(ticker, str) or ticker.upper() not in SUPPORTED_TICKERS:
            raise ValueError(f"tickers[{index}].ticker is unsupported")
        ticker = ticker.upper()
        if ticker in seen:
            raise ValueError(f"duplicate alias ticker: {ticker}")
        seen.add(ticker)
        for field in ALIAS_LIST_FIELDS:
            value = rule.get(field, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise ValueError(f"tickers[{index}].{field} must be a string list")
        for field in {"cashtag", "official_company_name"}:
            value = rule.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"tickers[{index}].{field} must be a string")
    return config


def _contains_phrase(text: str, phrase: str) -> bool:
    # Cashtags need a boundary before '$'; ordinary phrases use alphanumeric
    # and hyphen boundaries so AMD does not match metadata and an "Nvidia-like"
    # comparison does not become evidence about NVIDIA.
    escaped = re.escape(str(phrase).strip())
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9-]){escaped}(?![A-Za-z0-9-])",
            text,
            re.IGNORECASE,
        )
    )


def _matched_fields(title: str, description: str, phrase: str) -> list[str]:
    return [
        field
        for field, text in (("title", title), ("description", description))
        if _contains_phrase(text, phrase)
    ]


def _evidence(
    *,
    kind: str,
    phrase: str,
    title: str,
    description: str,
) -> list[dict[str, str]]:
    return [
        {
            "rule": kind,
            "term": str(phrase),
            "field": field,
        }
        for field in _matched_fields(title, description, str(phrase))
    ]


def _evaluate_ticker(
    title: str,
    description: str,
    rule: Mapping[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    ticker = str(rule["ticker"]).upper()
    exclusion_evidence = [
        evidence
        for phrase in (rule.get("exclusion_terms") or [])
        for evidence in _evidence(
            kind="exclusion",
            phrase=str(phrase),
            title=title,
            description=description,
        )
    ]
    if exclusion_evidence:
        return False, {
            "ticker": ticker,
            "decision": "excluded",
            "evidence": exclusion_evidence,
        }

    contextual_spellings = {
        str(alias).casefold() for alias in (rule.get("context_required_aliases") or [])
    }
    strong_rules = [
        ("ticker_symbol", rule.get("ticker")),
        ("cashtag", rule.get("cashtag")),
        ("official_company_name", rule.get("official_company_name")),
        *[("strong_alias", alias) for alias in (rule.get("strong_aliases") or [])],
    ]
    strong_rules = [
        (kind, phrase)
        for kind, phrase in strong_rules
        if phrase
        and str(phrase).casefold() not in contextual_spellings
        and str(phrase).casefold() not in GENERIC_CONTEXT_ONLY
    ]
    strong_evidence = [
        evidence
        for kind, phrase in strong_rules
        for evidence in _evidence(
            kind=kind,
            phrase=str(phrase),
            title=title,
            description=description,
        )
    ]
    if strong_evidence:
        return True, {
            "ticker": ticker,
            "decision": "matched",
            "evidence": strong_evidence,
        }

    alias_evidence = [
        evidence
        for alias in (rule.get("context_required_aliases") or [])
        for evidence in _evidence(
            kind="context_alias",
            phrase=str(alias),
            title=title,
            description=description,
        )
    ]
    context_evidence = [
        evidence
        for term in (rule.get("context_terms") or [])
        for evidence in _evidence(
            kind="context_term",
            phrase=str(term),
            title=title,
            description=description,
        )
    ]
    if alias_evidence and context_evidence:
        return True, {
            "ticker": ticker,
            "decision": "matched",
            "evidence": alias_evidence + context_evidence,
        }
    return False, None


def match_ticker(
    title: str, description: str, alias_config: Mapping[str, Any]
) -> RelevanceResult:
    normalized_title = str(title or "")
    normalized_description = str(description or "")
    evaluated = [
        _evaluate_ticker(normalized_title, normalized_description, rule)
        for rule in alias_config.get("tickers", [])
    ]
    matches = tuple(
        str(evidence["ticker"])
        for matched, evidence in evaluated
        if matched and evidence is not None
    )
    evidence = tuple(item for _, item in evaluated if item is not None)
    return RelevanceResult(
        ticker=matches[0] if len(matches) == 1 else None,
        matches=matches,
        ambiguous=len(matches) > 1,
        evidence=evidence,
    )

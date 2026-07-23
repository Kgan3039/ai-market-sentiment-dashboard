"""RSS fetching, parsing, relevance classification, and persistence."""

from __future__ import annotations

import email.utils
import xml.etree.ElementTree as ET
from datetime import timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
import yaml

from .relevance import load_alias_config, match_ticker
from .repository import Phase0Repository, utc_now
from .urls import canonicalize_url


def load_feed_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    feeds = config.get("feeds") if isinstance(config, dict) else None
    if not isinstance(feeds, list) or len(feeds) > 3:
        raise ValueError("feed config must contain no more than three feeds")
    return config


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ET.Element, names: set[str]) -> str:
    for child in element:
        if _local_name(child.tag) in names and child.text:
            return child.text.strip()
    return ""


def _published_iso(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return value


def parse_feed(xml_body: bytes | str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_body)
    entries = [
        node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}
    ]
    parsed: list[dict[str, str]] = []
    for entry in entries:
        link = _child_text(entry, {"link"})
        if not link:
            for child in entry:
                if _local_name(child.tag) == "link":
                    link = child.attrib.get("href", "")
                    if link:
                        break
        parsed.append(
            {
                "title": _child_text(entry, {"title"}),
                "description": _child_text(
                    entry, {"description", "summary", "content"}
                ),
                "url": link,
                "published_at": _published_iso(
                    _child_text(entry, {"pubdate", "published", "updated"})
                ),
            }
        )
    return [item for item in parsed if item["title"] and item["url"]]


class RSSFetcher:
    def __init__(
        self,
        repository: Phase0Repository,
        *,
        feeds_path: str | Path,
        aliases_path: str | Path,
        get: Callable[..., Any] = requests.get,
    ) -> None:
        self.repository = repository
        self.feed_config = load_feed_config(feeds_path)
        self.alias_config = load_alias_config(aliases_path)
        self._get = get

    def fetch(self) -> tuple[dict[str, int], list[dict[str, Any]]]:
        counts = {
            "fetched": 0,
            "inserted": 0,
            "duplicates": 0,
            "assigned": 0,
            "unmatched": 0,
            "ambiguous": 0,
            "feeds_succeeded": 0,
            "feeds_not_modified": 0,
            "invalid": 0,
        }
        errors: list[dict[str, Any]] = []
        feeds: Iterable[dict[str, Any]] = self.feed_config["feeds"]
        for feed in feeds:
            if not feed.get("enabled", True):
                continue
            feed_id = str(feed.get("id") or feed.get("name") or "rss")
            timeout = int((feed.get("polling") or {}).get("timeout_seconds", 20))
            try:
                source = f"rss:{feed_id}"
                state = self.repository.source_state(source)
                headers = {"User-Agent": "TickerNarrativesPhase0/1.0"}
                if state and state.get("etag"):
                    headers["If-None-Match"] = state["etag"]
                if state and state.get("last_modified"):
                    headers["If-Modified-Since"] = state["last_modified"]
                response = self._get(
                    feed["url"],
                    timeout=timeout,
                    headers=headers,
                )
                checked_at = utc_now()
                if response.status_code == 304:
                    counts["feeds_succeeded"] += 1
                    counts["feeds_not_modified"] += 1
                    self.repository.set_source_state(
                        source,
                        etag=response.headers.get("ETag"),
                        last_modified=response.headers.get("Last-Modified"),
                        checked_at=checked_at,
                        successful=True,
                        metadata={"status_code": 304},
                    )
                    continue
                response.raise_for_status()
                items = parse_feed(response.content)
                counts["feeds_succeeded"] += 1
                counts["fetched"] += len(items)
                self.repository.set_source_state(
                    source,
                    etag=response.headers.get("ETag"),
                    last_modified=response.headers.get("Last-Modified"),
                    checked_at=checked_at,
                    successful=True,
                    metadata={
                        "status_code": response.status_code,
                        "item_count": len(items),
                    },
                )
                for item in items:
                    canonical_url = canonicalize_url(item["url"])
                    if not canonical_url:
                        counts["invalid"] += 1
                        continue
                    # Preserve raw input before running relevance classification.
                    result = self.repository.insert_raw_item(
                        {
                            "source": source,
                            "ticker": None,
                            "title": item["title"],
                            "description": item["description"],
                            "url": item["url"],
                            "canonical_url": canonical_url,
                            "published_at": item["published_at"],
                            "fetched_at": checked_at,
                            "raw_json": {**item, "feed_id": feed_id},
                        }
                    )
                    relevance = match_ticker(
                        item["title"], item["description"], self.alias_config
                    )
                    if relevance.ambiguous:
                        counts["ambiguous"] += 1
                        errors.append(
                            {
                                "feed": feed_id,
                                "type": "ambiguous_ticker",
                                "url": item["url"],
                                "matches": list(relevance.matches),
                            }
                        )
                    elif relevance.ticker:
                        counts["assigned"] += 1
                        self.repository.update_raw_item_ticker(
                            result.item_id, relevance.ticker
                        )
                    else:
                        counts["unmatched"] += 1
                    counts["inserted" if result.inserted else "duplicates"] += 1
            except Exception as exc:
                self.repository.set_source_state(
                    f"rss:{feed_id}",
                    etag=None,
                    last_modified=None,
                    checked_at=utc_now(),
                    successful=False,
                    metadata={"error": str(exc)},
                )
                errors.append(
                    {"feed": feed_id, "type": "fetch_error", "error": str(exc)}
                )
        return counts, errors

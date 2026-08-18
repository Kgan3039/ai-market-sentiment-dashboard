"""RSS fetching, evidence preservation, and relevance classification.

Ingestion is deliberately split across several runs per feed rather than
gathered into one, because the guarantees #62 asks for are not all the same
guarantee:

``fetch_rss``
    The exact response bytes, and nothing derived from them.  It commits
    first and alone, so a parser or classifier that fails afterwards still
    leaves on disk the response that broke it.

``ingest_rss``
    The normalized entries for one trading day, unclassified.  Raw evidence
    becomes durable here, before any relevance decision exists, which is
    what lets a classifier failure cost the classification and not the
    evidence.

``observe_rss``
    A story already stored, seen again.  It adds provenance under the day
    the item has always belonged to and changes nothing about the row --
    an undated entry a feed repeats tomorrow is still yesterday's evidence.

``classify_rss``
    One run per ``(ticker, trading_day)`` partition, holding only derived
    state: the association, the candidate row, and the match evidence for
    that one ticker.  An article about two tickers is written by two runs,
    each staying inside its own partition.

``checkpoint_rss``
    The feed's own checkpoint, last, so a conditional-request marker is
    never advanced over evidence that failed to persist.

The split is the honest shape.  Claiming one transaction would mean either
losing the evidence when classification fails, or letting one run write
across every partition a feed happens to mention.
"""

from __future__ import annotations

from datetime import datetime, timezone
import email.utils
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urljoin, urlsplit
import uuid

import requests
import yaml

from .relevance import load_alias_config, match_ticker
from .repository import Phase0Repository, redact_secrets, utc_now
from .urls import canonicalize_url


#: The four stages one feed moves through; see the module docstring for
#: why they are four and not one.
STAGE_FETCH = "fetch_rss"
STAGE_INGEST = "ingest_rss"
STAGE_OBSERVE = "observe_rss"
STAGE_CLASSIFY = "classify_rss"
STAGE_CHECKPOINT = "checkpoint_rss"
STAGE_RECLASSIFY = "reclassify_rss"


MAX_FEED_BYTES = 5_000_000
MAX_FEED_ENTRIES = 5_000
SUPPORTED_FEED_FORMATS = {"atom", "rdf", "rss", "rss2"}
FEED_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
XML_BASE = "{http://www.w3.org/XML/1998/namespace}base"
LOGGER = logging.getLogger(__name__)
REQUIRED_FEED_FIELDS = {
    "id",
    "name",
    "url",
    "enabled",
    "format",
    "intended_role",
    "expected_fields",
    "polling",
    "notes",
}
REQUIRED_EXPECTED_FIELDS = {"title", "url", "description", "published_at"}
REQUIRED_POLLING_FIELDS = {
    "interval_minutes",
    "conditional_get",
    "timeout_seconds",
}


class RSSRequestError(RuntimeError):
    """A feed request that failed after all configured attempts."""

    def __init__(self, attempts: int, original: Exception) -> None:
        super().__init__(str(original))
        self.attempts = attempts
        self.original = original


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number")
    if value <= 0:
        raise ValueError(f"{field} must be a positive number")
    return float(value)


def load_feed_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    feeds = config.get("feeds") if isinstance(config, dict) else None
    if not isinstance(feeds, list) or len(feeds) > 3:
        raise ValueError("feed config must contain a list of no more than three feeds")

    feed_ids: set[str] = set()
    for index, feed in enumerate(feeds):
        prefix = f"feeds[{index}]"
        if not isinstance(feed, dict):
            raise ValueError(f"{prefix} must be an object")
        missing_fields = REQUIRED_FEED_FIELDS - set(feed)
        if missing_fields:
            raise ValueError(
                f"{prefix} is missing required fields: "
                + ", ".join(sorted(missing_fields))
            )
        feed_id = feed.get("id")
        if not isinstance(feed_id, str) or not FEED_ID_PATTERN.fullmatch(feed_id):
            raise ValueError(f"{prefix}.id must be a lowercase slug")
        if feed_id in feed_ids:
            raise ValueError(f"duplicate feed id: {feed_id}")
        feed_ids.add(feed_id)

        url = feed.get("url")
        parsed_url = urlsplit(url) if isinstance(url, str) else None
        if (
            parsed_url is None
            or parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username
            or parsed_url.password
        ):
            raise ValueError(f"{prefix}.url must be an absolute HTTPS URL")
        if not isinstance(feed["enabled"], bool):
            raise ValueError(f"{prefix}.enabled must be a boolean")
        feed_format = feed["format"]
        if (
            not isinstance(feed_format, str)
            or feed_format.lower() not in SUPPORTED_FEED_FORMATS
        ):
            raise ValueError(f"{prefix}.format is unsupported")
        for field in {"name", "intended_role"}:
            if not isinstance(feed[field], str) or not feed[field].strip():
                raise ValueError(f"{prefix}.{field} must be a non-empty string")
        notes = feed["notes"]
        if (
            not isinstance(notes, list)
            or not notes
            or not all(isinstance(note, str) and note.strip() for note in notes)
        ):
            raise ValueError(f"{prefix}.notes must be a non-empty string list")
        expected_fields = feed["expected_fields"]
        if not isinstance(expected_fields, dict):
            raise ValueError(f"{prefix}.expected_fields must be an object")
        missing_expected = REQUIRED_EXPECTED_FIELDS - set(expected_fields)
        if missing_expected:
            raise ValueError(
                f"{prefix}.expected_fields is missing required fields: "
                + ", ".join(sorted(missing_expected))
            )
        if not all(
            isinstance(expected_fields[field], str) and expected_fields[field].strip()
            for field in REQUIRED_EXPECTED_FIELDS
        ):
            raise ValueError(
                f"{prefix}.expected_fields values must be non-empty strings"
            )

        polling = feed["polling"]
        if not isinstance(polling, dict):
            raise ValueError(f"{prefix}.polling must be an object")
        missing_polling = REQUIRED_POLLING_FIELDS - set(polling)
        if missing_polling:
            raise ValueError(
                f"{prefix}.polling is missing required fields: "
                + ", ".join(sorted(missing_polling))
            )
        _positive_number(
            polling["timeout_seconds"],
            f"{prefix}.polling.timeout_seconds",
        )
        _positive_number(
            polling["interval_minutes"],
            f"{prefix}.polling.interval_minutes",
        )
        if not isinstance(polling["conditional_get"], bool):
            raise ValueError(f"{prefix}.polling.conditional_get must be a boolean")
        if "max_retries" in polling and (
            isinstance(polling["max_retries"], bool)
            or not isinstance(polling["max_retries"], int)
            or polling["max_retries"] < 0
        ):
            raise ValueError(
                f"{prefix}.polling.max_retries must be a non-negative integer"
            )
        if "retry_backoff_seconds" in polling:
            backoff = polling["retry_backoff_seconds"]
            if isinstance(backoff, bool) or not isinstance(backoff, (int, float)):
                raise ValueError(
                    f"{prefix}.polling.retry_backoff_seconds must be non-negative"
                )
            if backoff < 0:
                raise ValueError(
                    f"{prefix}.polling.retry_backoff_seconds must be non-negative"
                )
    return config


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _element_text(element: ET.Element) -> str:
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def _child_text(element: ET.Element, names: set[str]) -> str:
    for child in element:
        if _local_name(child.tag) in names:
            value = _element_text(child)
            if value:
                return value
    return ""


def _base_chain(root: ET.Element, feed_url: str | None) -> dict[ET.Element, str]:
    """Resolve every element's XML Base, walking down from the root.

    XML Base is *cumulative and relative*: each ``xml:base`` is resolved
    against the base already in effect, and the base in effect at the root
    is the document's own URL.  Resolving a root ``xml:base="articles/"``
    as if it were already absolute produced a relative link that then
    failed validation as "not absolute HTTP(S)" -- the feed's own URL is
    what it was always meant to be relative to.

    ElementTree has no parent pointers, so the chain is built in one
    top-down pass rather than by walking up from each entry.
    """

    document_base = str(feed_url or "")
    bases: dict[ET.Element, str] = {}
    stack = [(root, document_base)]
    while stack:
        element, inherited = stack.pop()
        declared = str(element.attrib.get(XML_BASE) or "").strip()
        resolved = urljoin(inherited, declared) if declared else inherited
        bases[element] = resolved
        for child in element:
            stack.append((child, resolved))
    return bases


def _entry_link(entry: ET.Element) -> tuple[str, ET.Element | None]:
    links = [child for child in entry if _local_name(child.tag) == "link"]
    for child in links:
        href = str(child.attrib.get("href") or "").strip()
        rel = str(child.attrib.get("rel") or "alternate").lower()
        if href and rel in {"", "alternate"}:
            return href, child
    for child in links:
        value = _element_text(child)
        if value:
            return value, child
    guid_element = next(
        (child for child in entry if _local_name(child.tag) == "guid"),
        None,
    )
    if guid_element is not None:
        guid = _element_text(guid_element)
        is_permalink = str(guid_element.attrib.get("isPermaLink", "true")).lower()
        if is_permalink != "false" and urlsplit(guid).scheme in {"http", "https"}:
            return guid, guid_element
    return "", None


def _published_iso(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("publication timestamp is not valid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def parse_feed(
    xml_body: bytes | str,
    *,
    feed_url: str | None = None,
    expected_format: str | None = None,
) -> list[dict[str, Any]]:
    """Parse every entry, retaining malformed entry evidence and identifiers."""
    body = xml_body.encode("utf-8") if isinstance(xml_body, str) else xml_body
    if len(body) > MAX_FEED_BYTES:
        raise ValueError("feed XML exceeds the configured size limit")
    if b"<!DOCTYPE" in body.upper():
        raise ValueError("feed XML document types are not allowed")
    root = ET.fromstring(body)
    root_name = _local_name(root.tag)
    if root_name not in {"feed", "rdf", "rss"}:
        raise ValueError(f"unsupported feed root: {root_name}")
    if expected_format:
        normalized_format = expected_format.lower()
        if normalized_format == "atom" and root_name != "feed":
            raise ValueError("feed content does not match configured Atom format")
        if normalized_format in {"rss", "rss2", "rdf"} and root_name == "feed":
            raise ValueError("feed content does not match configured RSS format")

    entries = [
        node for node in root.iter() if _local_name(node.tag) in {"item", "entry"}
    ]
    if len(entries) > MAX_FEED_ENTRIES:
        raise ValueError("feed contains too many entries")
    parsed: list[dict[str, Any]] = []
    bases = _base_chain(root, feed_url)
    feed_digest = hashlib.sha256(body).digest()
    for entry_index, entry in enumerate(entries):
        entry_digest = hashlib.sha256(
            feed_digest + entry_index.to_bytes(8, "big")
        ).hexdigest()
        guid = _child_text(entry, {"guid", "id"})
        raw_link, link_element = _entry_link(entry)
        # The base in effect where the href was *written* -- a nested
        # ``xml:base`` on the link element itself applies to it, not the
        # entry's.
        link_base = bases.get(link_element if link_element is not None else entry, "")
        resolved_link = urljoin(link_base, raw_link) if raw_link else ""
        title = _child_text(entry, {"title"})
        description = _child_text(
            entry,
            {"content", "description", "summary"},
        )
        published_raw = _child_text(
            entry,
            {"pubdate", "published", "updated"},
        )
        validation_errors = []
        if not title:
            validation_errors.append("missing title")
        if not resolved_link:
            validation_errors.append("missing link")
        elif (
            urlsplit(resolved_link).scheme not in {"http", "https"}
            or not urlsplit(resolved_link).netloc
        ):
            validation_errors.append("link must resolve to absolute HTTP(S)")
        try:
            published_at = _published_iso(published_raw)
        except ValueError as exc:
            published_at = None
            validation_errors.append(str(exc))
        parsed.append(
            {
                "title": title,
                "description": description,
                "url": resolved_link,
                "published_at": published_at,
                "published_raw": published_raw,
                "external_id": guid or entry_digest,
                "guid": guid or None,
                "entry_digest": entry_digest,
                "entry_index": entry_index,
                "entry_attributes": dict(entry.attrib),
                "validation_errors": validation_errors,
            }
        )
    return parsed


def _retryable_request_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return True
    return status_code in {408, 429} or status_code >= 500


def _read_response_body(response: Any) -> bytes:
    """Read a bounded response without first materializing an oversized body."""
    content_length = response.headers.get("Content-Length")
    if content_length not in (None, ""):
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("feed Content-Length is invalid") from exc
        if declared_length < 0:
            raise ValueError("feed Content-Length is invalid")
        if declared_length > MAX_FEED_BYTES:
            raise ValueError("feed response exceeds the configured size limit")

    iter_content = getattr(response, "iter_content", None)
    if not callable(iter_content):
        body = bytes(response.content)
        if len(body) > MAX_FEED_BYTES:
            raise ValueError("feed response exceeds the configured size limit")
        return body

    chunks = bytearray()
    for chunk in iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        chunks.extend(chunk)
        if len(chunks) > MAX_FEED_BYTES:
            raise ValueError("feed response exceeds the configured size limit")
    return bytes(chunks)


def partition_run_id(base_run_id: str, feed_id: str, scope: str, day: str) -> str:
    """The run identity for one feed's slice of one partition.

    ``run_log`` is ``UNIQUE(run_id, stage)`` and I1 refuses to record one
    identity against a second partition, so an identity has to name the
    partition it belongs to.  A feed spanning two days, or mentioning two
    tickers, would otherwise not merely muddle the audit — the second
    partition would fail to persist at all.
    """

    return f"{base_run_id}:{feed_id}:{scope}:{day}"


def effective_day(item: Mapping[str, Any]) -> str:
    """The trading day a raw item belongs to.

    Derived exactly as I1 derives it — ``published_at`` falling back to
    ``fetched_at`` — because a run whose day disagrees with the row's is
    refused, and two definitions would drift.
    """

    stamp = item.get("published_at") or item.get("fetched_at")
    return str(stamp)[:10]


class RSSFetcher:
    """Fetch approved feeds and settle every partition they produce."""

    def __init__(
        self,
        repository: Phase0Repository,
        *,
        feeds_path: str | Path,
        aliases_path: str | Path,
        get: Callable[..., Any] = requests.get,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        throttle_seconds: float = 0,
        sleep: Callable[[float], None] = time.sleep,
        pipeline_version: str = "phase0-v1",
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if retry_backoff_seconds < 0 or throttle_seconds < 0:
            raise ValueError("retry and throttle delays cannot be negative")
        if not pipeline_version.strip():
            raise ValueError("pipeline_version is required")
        self.repository = repository
        self.feed_config = load_feed_config(feeds_path)
        self.alias_config = load_alias_config(aliases_path)
        self._get = get
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.throttle_seconds = float(throttle_seconds)
        self._sleep = sleep
        self.pipeline_version = pipeline_version.strip()

    # -- HTTP ------------------------------------------------------------

    def _request(
        self,
        feed: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> tuple[Any, int]:
        polling = feed.get("polling") or {}
        timeout = float(polling.get("timeout_seconds", 20))
        max_retries = int(polling.get("max_retries", self.max_retries))
        backoff = float(
            polling.get("retry_backoff_seconds", self.retry_backoff_seconds)
        )
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._get(
                    feed["url"],
                    timeout=timeout,
                    headers=dict(headers),
                    stream=True,
                )
                if response.status_code != 304:
                    response.raise_for_status()
                return response, attempt
            except Exception as exc:
                if attempt == attempts or not _retryable_request_error(exc):
                    raise RSSRequestError(attempt, exc) from exc
                self._sleep(backoff * (2 ** (attempt - 1)))
        raise AssertionError("unreachable")

    # -- Relevance -------------------------------------------------------

    def _classify(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Decide one item's relevance, in memory, touching no database.

        Returned whole so the caller can fan it out across partitions: the
        decision for each ticker is written by that ticker's own run.
        """

        if item.get("ingest_status") == "invalid":
            return {
                "ticker": None,
                "ingest_status": "invalid",
                "matches": (),
                "evidence": (),
                "outcome": "invalid",
            }
        relevance = match_ticker(
            str(item.get("title") or ""),
            str(item.get("description") or ""),
            self.alias_config,
        )
        if relevance.ambiguous:
            outcome, status, ticker = "ambiguous", "ambiguous", None
        elif relevance.ticker:
            outcome, status, ticker = "assigned", "valid", relevance.ticker
        else:
            outcome, status, ticker = "unmatched", "valid", None
        return {
            "ticker": ticker,
            "ingest_status": status,
            "matches": relevance.matches,
            "evidence": relevance.evidence,
            "outcome": outcome,
        }

    @staticmethod
    def _decisions_by_partition(
        classified: Sequence[tuple[int, str, Mapping[str, Any]]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        """Group per-item decisions into ``(ticker, day)`` partitions.

        One item yields one decision per ticker it has anything to say
        about — a match or an exclusion — and each lands in that ticker's
        partition.  Nothing is written across partitions, which is the
        rule RSS makes easy to break: a single feed routinely carries
        several tickers and several days at once.
        """

        partitions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item_id, day, classification in classified:
            for entry in classification["evidence"]:
                ticker = str(entry["ticker"])
                assigned = classification["ticker"] == ticker
                is_candidate = (
                    classification["outcome"] == "ambiguous"
                    and ticker in classification["matches"]
                )
                partitions[(ticker, day)].append(
                    {
                        "raw_item_id": item_id,
                        # Only the ticker that owns the assignment writes it;
                        # every other partition leaves the column alone.
                        "ticker": ticker if assigned else None,
                        "ingest_status": classification["ingest_status"],
                        "candidate_reason": (
                            json.dumps(entry, sort_keys=True, separators=(",", ":"))
                            if is_candidate
                            else None
                        ),
                        "evidence": {
                            "decision": entry["decision"],
                            "evidence": entry["evidence"],
                        },
                    }
                )
        return partitions

    # -- One feed --------------------------------------------------------

    def _normalized_entries(
        self,
        parsed_items: Sequence[Mapping[str, Any]],
        *,
        feed: Mapping[str, Any],
        source: str,
        snapshot_id: int,
        response: Any,
        response_url: str,
        checked_at: str,
    ) -> list[dict[str, Any]]:
        """Turn parsed entries into raw items, carrying their provenance.

        No relevance decision is taken here and no ticker is asserted: the
        batch this produces is evidence, and it has to be storable before
        anything derived from it exists.
        """

        feed_id = feed["id"]
        feed_metadata = {
            "id": feed_id,
            "name": feed.get("name"),
            "configured_url": feed["url"],
            "response_url": response_url,
            "format": feed.get("format", "rss"),
            "http": {
                "status_code": response.status_code,
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "checked_at": checked_at,
            },
        }
        entries = []
        for item in parsed_items:
            item_errors = list(item["validation_errors"])
            canonical_url = canonicalize_url(item["url"])
            parsed_url = urlsplit(canonical_url)
            if item["url"] and (
                parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc
            ):
                if "link must resolve to absolute HTTP(S)" not in item_errors:
                    item_errors.append("link must resolve to absolute HTTP(S)")
            if not canonical_url or item_errors:
                # Unusable evidence is still evidence: it is kept under a
                # urn keyed by the feed and the entry so it cannot collide.
                canonical_url = f"urn:rss:{feed_id}:{item['entry_digest']}"
                item_source = source
            else:
                # Keyed by publisher, not by feed: the same story on two
                # approved feeds is one row with two provenance records.
                item_source = f"rss:{parsed_url.hostname.lower()}"
            entries.append(
                {
                    "source": item_source,
                    "ticker": None,
                    "title": item["title"] or None,
                    "description": item["description"] or None,
                    "url": item["url"] or None,
                    "canonical_url": canonical_url,
                    "external_id": item["external_id"],
                    "published_at": item["published_at"],
                    "fetched_at": checked_at,
                    "ingest_status": "invalid" if item_errors else "valid",
                    "validation_errors": item_errors,
                    "raw_json": {
                        "feed": feed_metadata,
                        "feed_snapshot_id": snapshot_id,
                        "guid": item["guid"],
                        "entry_index": item["entry_index"],
                        "published_raw": item["published_raw"],
                        "entry_attributes": item["entry_attributes"],
                        "parsed": {
                            "title": item["title"],
                            "description": item["description"],
                            "url": item["url"],
                            "published_at": item["published_at"],
                        },
                    },
                    "feed_provenance": [
                        {
                            "feed_source": source,
                            "external_id": item["external_id"],
                            "snapshot_id": snapshot_id,
                            "entry_digest": item["entry_digest"],
                        }
                    ],
                }
            )
        return entries

    def _checkpoint(
        self,
        source: str,
        *,
        base_run_id: str,
        feed_id: str,
        fetch_day: str,
        checked_at: str,
        status: str,
        metadata: Mapping[str, Any],
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Settle one feed's checkpoint in its own terminal run.

        Last of the feed's runs on purpose.  A conditional-request marker
        is a promise that everything up to it is already stored, so
        advancing one over evidence that failed to persist would make the
        next fetch skip a response nobody kept.
        """

        with self.repository.stage_run(
            run_id=partition_run_id(base_run_id, feed_id, "checkpoint", fetch_day),
            stage=STAGE_CHECKPOINT,
            trading_day=fetch_day,
            pipeline_version=self.pipeline_version,
        ) as run:
            self.repository.record_source_state(
                source,
                run=run,
                etag=etag,
                last_modified=last_modified,
                checked_at=checked_at,
                status=status,
                metadata=dict(metadata),
                terminal=True,
            )

    def _fetch_feed(
        self,
        feed: Mapping[str, Any],
        *,
        base_run_id: str,
        counts: dict[str, int],
        errors: list[dict[str, Any]],
    ) -> None:
        feed_id = feed["id"]
        source = f"rss:{feed_id}"
        checked_at = utc_now()
        fetch_day = checked_at[:10]
        attempts = 1
        try:
            state = self.repository.source_state(source)
            headers = {"User-Agent": "TickerNarrativesPhase0/1.0"}
            conditional = (feed.get("polling") or {}).get("conditional_get", True)
            if conditional and state and state.get("etag"):
                headers["If-None-Match"] = state["etag"]
            if conditional and state and state.get("last_modified"):
                headers["If-Modified-Since"] = state["last_modified"]

            response, attempts = self._request(feed, headers)
            counts["retries"] += attempts - 1
            checked_at = utc_now()
            fetch_day = checked_at[:10]

            if response.status_code == 304:
                self._settle_not_modified(
                    source,
                    feed_id=feed_id,
                    state=state,
                    response=response,
                    base_run_id=base_run_id,
                    fetch_day=fetch_day,
                    checked_at=checked_at,
                    attempts=attempts,
                    counts=counts,
                    errors=errors,
                )
                return

            response_url = str(getattr(response, "url", None) or feed["url"])
            response_body = _read_response_body(response)

            # 1. The bytes, alone, before anything derived from them.
            with self.repository.stage_run(
                run_id=partition_run_id(base_run_id, feed_id, "snapshot", fetch_day),
                stage=STAGE_FETCH,
                trading_day=fetch_day,
                pipeline_version=self.pipeline_version,
            ) as run:
                snapshot_id = self.repository.record_feed_snapshot(
                    feed_source=source,
                    response_url=response_url,
                    body=response_body,
                    fetched_at=checked_at,
                    content_type=response.headers.get("Content-Type"),
                    content_encoding=response.headers.get("Content-Encoding"),
                    run=run,
                    terminal=True,
                )

            parsed_items = parse_feed(
                response_body,
                feed_url=response_url,
                expected_format=feed.get("format"),
            )
            counts["fetched"] += len(parsed_items)
            entries = self._normalized_entries(
                parsed_items,
                feed=feed,
                source=source,
                snapshot_id=snapshot_id,
                response=response,
                response_url=response_url,
                checked_at=checked_at,
            )

            local = dict.fromkeys(("assigned", "unmatched", "ambiguous", "invalid"), 0)
            local_errors: list[dict[str, Any]] = []
            for entry in entries:
                if entry["ingest_status"] == "invalid":
                    local_errors.append(
                        {
                            "feed": feed_id,
                            "type": "invalid_entry",
                            "external_id": entry["external_id"],
                            "errors": entry["validation_errors"],
                        }
                    )

            # 2. Raw evidence.  A first sighting is ingested under its own
            #    day; a story already stored is *observed*, which adds
            #    provenance under the day it has always belonged to and
            #    changes nothing about the row.  Re-ingesting a repeat is
            #    what used to wedge an undated entry's feed for good.
            stored = self.repository.stored_raw_items(
                [(entry["source"], entry["canonical_url"]) for entry in entries]
            )
            fresh: dict[str, list[dict[str, Any]]] = defaultdict(list)
            seen: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entry in entries:
                known = stored.get((entry["source"], entry["canonical_url"]))
                if known is None:
                    fresh[effective_day(entry)].append(entry)
                else:
                    seen[known["trading_day"]].append(
                        {
                            "raw_item_id": known["id"],
                            "feed_provenance": entry["feed_provenance"],
                        }
                    )
            item_ids: list[int] = [
                observation["raw_item_id"]
                for group in seen.values()
                for observation in group
            ]
            # An entry that was already stored *before* this response is a
            # duplicate the lookup above can see.
            counts["duplicates"] += len(item_ids)
            for day in sorted(fresh):
                with self.repository.stage_run(
                    run_id=partition_run_id(base_run_id, feed_id, "ingest", day),
                    stage=STAGE_INGEST,
                    trading_day=day,
                    pipeline_version=self.pipeline_version,
                ) as run:
                    results = self.repository.ingest_raw_items(
                        fresh[day], run=run, terminal=True
                    )
                # ...and one that duplicates an entry from *this* response is
                # a duplicate only the insert can see.  The lookup ran before
                # any of them was written, so a feed listing the same story
                # twice arrives here as two first sightings; ``inserted`` on
                # the result is the authoritative answer to which of them
                # actually created the row, so both buckets come from it
                # rather than from a second dedup guess in this module.
                counts["inserted"] += sum(result.inserted for result in results)
                counts["duplicates"] += sum(not result.inserted for result in results)
                item_ids.extend(result.item_id for result in results)
            for day in sorted(seen):
                with self.repository.stage_run(
                    run_id=partition_run_id(base_run_id, feed_id, "observe", day),
                    stage=STAGE_OBSERVE,
                    trading_day=day,
                    pipeline_version=self.pipeline_version,
                ) as run:
                    self.repository.record_feed_observations(
                        seen[day], run=run, terminal=True
                    )

            # 3. Derived relevance, computed from what is *persisted*.
            #    Reading the rows back is the whole fix for a syndicated
            #    variant: feed B's title for a story feed A already stored
            #    is not what the row holds, so classifying B's text would
            #    write decisions about words no reader can find -- and the
            #    next offline replay, reading the row, would reverse them.
            self._classify_persisted(
                self.repository.rss_raw_items(item_ids),
                base_run_id=base_run_id,
                scope=feed_id,
                stage=STAGE_CLASSIFY,
                counts=local,
                errors=local_errors,
                context={"feed": feed_id},
            )

            for key, value in local.items():
                counts[key] += value
            errors.extend(local_errors)
            status = "partial" if local["invalid"] else "success"

            # 4. Only now may the feed's checkpoint move.
            self._checkpoint(
                source,
                base_run_id=base_run_id,
                feed_id=feed_id,
                fetch_day=fetch_day,
                checked_at=checked_at,
                status=status,
                etag=(response.headers.get("ETag") if status == "success" else None),
                last_modified=(
                    response.headers.get("Last-Modified")
                    if status == "success"
                    else None
                ),
                metadata={
                    "status": status,
                    "status_code": response.status_code,
                    "item_count": len(entries),
                    "assigned": local["assigned"],
                    "unmatched": local["unmatched"],
                    "ambiguous": local["ambiguous"],
                    "invalid": local["invalid"],
                    "attempts": attempts,
                    "response_url": response_url,
                    "response_etag": response.headers.get("ETag"),
                    "response_last_modified": response.headers.get("Last-Modified"),
                },
            )
            if local["invalid"]:
                counts["feeds_partial"] += 1
            else:
                counts["feeds_succeeded"] += 1
        except RSSRequestError as exc:
            counts["retries"] += exc.attempts - 1
            counts["feeds_failed"] += 1
            self._settle_failed_feed(
                source,
                base_run_id=base_run_id,
                feed_id=feed_id,
                fetch_day=fetch_day,
                checked_at=checked_at,
                attempts=exc.attempts,
                error_type=type(exc.original).__name__,
                message=redact_secrets(str(exc.original)),
                errors=errors,
                error_kind="fetch_error",
            )
        except Exception as exc:
            counts["feeds_failed"] += 1
            self._settle_failed_feed(
                source,
                base_run_id=base_run_id,
                feed_id=feed_id,
                fetch_day=fetch_day,
                checked_at=checked_at,
                attempts=attempts,
                error_type=type(exc).__name__,
                message=redact_secrets(str(exc)),
                errors=errors,
                error_kind="processing_error",
            )

    def _settle_failed_feed(
        self,
        source: str,
        *,
        base_run_id: str,
        feed_id: str,
        fetch_day: str,
        checked_at: str,
        attempts: int,
        error_type: str,
        message: str,
        errors: list[dict[str, Any]],
        error_kind: str,
    ) -> None:
        """Record a failed feed without advancing anything it did not earn.

        No ``etag`` and no ``last_modified``: whatever evidence did land is
        already durable under its own runs, and a marker moved here would
        tell the next fetch that a response it never stored can be skipped.
        """

        errors.append({"feed": feed_id, "type": error_kind, "error": message})
        LOGGER.error(
            "RSS feed=%s failed during %s error_type=%s",
            feed_id,
            error_kind.removesuffix("_error"),
            error_type,
        )
        try:
            self._checkpoint(
                source,
                base_run_id=base_run_id,
                feed_id=feed_id,
                fetch_day=fetch_day,
                checked_at=checked_at,
                status="failed",
                metadata={
                    "status": "failed",
                    "attempts": attempts,
                    "error_type": error_type,
                },
            )
        except Exception as settle_error:  # settlement must not mask the cause
            errors.append(
                {
                    "feed": feed_id,
                    "type": "settlement_error",
                    "error": redact_secrets(f"checkpoint failed: {settle_error}"),
                }
            )

    def _settle_not_modified(
        self,
        source: str,
        *,
        feed_id: str,
        state: Mapping[str, Any] | None,
        response: Any,
        base_run_id: str,
        fetch_day: str,
        checked_at: str,
        attempts: int,
        counts: dict[str, int],
        errors: list[dict[str, Any]],
    ) -> None:
        """A 304 is only good news when there is a stored response behind it."""

        has_baseline = bool(
            state
            and state.get("last_success_at")
            and (state.get("metadata") or {}).get("status")
            in {"success", "not_modified"}
        )
        counts["feeds_not_modified"] += 1
        if has_baseline:
            counts["feeds_succeeded"] += 1
        else:
            counts["feeds_partial"] += 1
            errors.append(
                {
                    "feed": feed_id,
                    "type": "not_modified_without_baseline",
                    "error": "received 304 before a successful feed snapshot",
                }
            )
        self._checkpoint(
            source,
            base_run_id=base_run_id,
            feed_id=feed_id,
            fetch_day=fetch_day,
            checked_at=checked_at,
            status="success" if has_baseline else "partial",
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
            metadata={
                "status": (
                    "not_modified" if has_baseline else "not_modified_without_baseline"
                ),
                "status_code": 304,
                "attempts": attempts,
            },
        )

    def _classify_persisted(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        base_run_id: str,
        scope: str,
        stage: str,
        counts: dict[str, int],
        errors: list[dict[str, Any]],
        context: Mapping[str, Any] | None = None,
    ) -> int:
        """Classify persisted rows and settle each partition they fall in.

        One routine for both paths on purpose.  A live fetch and an offline
        replay reading the same stored evidence must reach the same answer,
        and the surest way to guarantee that is for them to run the same
        code over the same input -- rows read back from the database, never
        a parser object that happens to be in hand.
        """

        classified: list[tuple[int, str, Mapping[str, Any]]] = []
        for item in items:
            day = str(item["trading_day"])
            try:
                classification = self._classify(item)
            except BaseException:
                # Attribute the failure to the day whose evidence broke the
                # classifier.  There is no ticker to attribute it to,
                # because deciding the ticker is precisely what failed -- so
                # this run is ticker-less, and it is recorded before the
                # exception leaves rather than the work failing silently.
                with self.repository.stage_run(
                    run_id=partition_run_id(base_run_id, scope, "scan", day),
                    stage=stage,
                    trading_day=day,
                    pipeline_version=self.pipeline_version,
                ):
                    raise
            counts[classification["outcome"]] += 1
            if classification["outcome"] == "ambiguous":
                errors.append(
                    {
                        **(context or {}),
                        "type": "ambiguous_ticker",
                        "raw_item_id": int(item["id"]),
                        "matches": list(classification["matches"]),
                    }
                )
            classified.append((int(item["id"]), day, classification))

        partitions = self._decisions_by_partition(classified)
        # A ticker that has stopped matching produces no decision above, so
        # nothing would clear the rows it left behind -- and only that
        # ticker's own run is allowed to.  Withdrawing is therefore an
        # explicit empty decision in that partition, not an omission.
        for item, (item_id, day, classification) in zip(items, classified):
            deciding = {str(entry["ticker"]) for entry in classification["evidence"]}
            for ticker in json.loads(item["derived_tickers"] or "[]"):
                if ticker in deciding:
                    continue
                partitions[(str(ticker), day)].append(
                    {
                        "raw_item_id": item_id,
                        "ticker": None,
                        "ingest_status": classification["ingest_status"],
                        "candidate_reason": None,
                        "evidence": None,
                    }
                )

        written = 0
        for (ticker, day), decisions in sorted(partitions.items()):
            with self.repository.stage_run(
                run_id=partition_run_id(base_run_id, scope, ticker.lower(), day),
                stage=stage,
                trading_day=day,
                pipeline_version=self.pipeline_version,
                ticker=ticker,
            ) as run:
                written += self.repository.replace_relevance_classifications(
                    decisions, run=run, terminal=True
                )
        return written

    # -- Entry points ----------------------------------------------------

    def fetch(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Fetch every enabled feed and settle each partition it produces.

        There is no aggregate run.  Each feed's evidence, each day it spans,
        and each ticker it mentions is its own run, because a run identity
        names one partition; the counters returned here are a summary for
        the caller, not a second audit record.

        There is no ``trading_day`` argument, for the reason I2 gives: a
        run's day is a partition identity the evidence decides, not a label
        a caller may put on it.  The repository derives each item's day and
        refuses a batch that disagrees with its run, so an override could
        only ever be ignored or fatal -- and for RSS it would be fatal,
        because the snapshot and the checkpoint carry real fetch
        timestamps that a declared day would contradict.

        **The counters are two independent partitions, not one.**

        * What happened to each parsed *entry*:
          ``fetched == inserted + duplicates``.  An entry is a duplicate
          whether the row it resolved to was stored by an earlier fetch or
          by an earlier entry of this same response.
        * What was decided about each distinct persisted *item*:
          ``assigned + unmatched + ambiguous + invalid``.

        The two do not sum together and are not meant to.  A malformed
        entry is stored evidence like any other, so it counts once as
        ``inserted`` and once as ``invalid`` -- those are answers to
        different questions, and collapsing them would lose one of them.
        """

        base_run_id = self._resolved_run_id("rss", run_id)
        counts = {
            "fetched": 0,
            "inserted": 0,
            "duplicates": 0,
            "assigned": 0,
            "unmatched": 0,
            "ambiguous": 0,
            "feeds_succeeded": 0,
            "feeds_partial": 0,
            "feeds_not_modified": 0,
            "feeds_failed": 0,
            "invalid": 0,
            "retries": 0,
        }
        errors: list[dict[str, Any]] = []
        processed = 0
        for feed in self.feed_config["feeds"]:
            if not feed.get("enabled", True):
                continue
            if processed and self.throttle_seconds:
                self._sleep(self.throttle_seconds)
            processed += 1
            self._fetch_feed(
                feed,
                base_run_id=base_run_id,
                counts=counts,
                errors=errors,
            )
        return counts, redact_secrets(errors)

    def reclassify_persisted(
        self,
        *,
        run_id: str | None = None,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        """Rebuild relevance from stored evidence, with no network at all.

        **Reads only persisted evidence.**  The input is
        :meth:`Phase0Repository.rss_raw_items` — rows that already carry RSS
        provenance — so nothing here consults a feed, a session, or a clock
        that matters.  ``RSSFetcher`` needs no working ``get`` to run this.

        **Replacement is atomic per partition.**  Each ``(ticker, day)``
        partition's derived state is replaced inside that partition's own
        terminal run: its association, candidate row, and match evidence go
        together or not at all.  A partition that fails leaves its *previous*
        derived state exactly as it was — this replaces, it never clears
        first and rebuilds after.

        **Raw evidence is never touched.**  Snapshots, provenance rows,
        ``raw_json``, and the parser's own ``invalid`` verdict all survive
        replay unchanged, which is what makes replay idempotent: the same
        stored bytes produce the same decisions, and writing them twice is
        the same as writing them once.

        A failure propagates after its run is recorded ``failed``; earlier
        partitions that already committed stay committed, and nothing is
        left half-replaced inside any one of them.
        """

        base_run_id = self._resolved_run_id("rss-reclassify", run_id)
        counts = {
            "scanned": 0,
            "updated": 0,
            "assigned": 0,
            "unmatched": 0,
            "ambiguous": 0,
            "invalid": 0,
        }
        errors: list[dict[str, Any]] = []
        items = self.repository.rss_raw_items()
        counts["scanned"] = len(items)
        counts["updated"] = self._classify_persisted(
            items,
            base_run_id=base_run_id,
            scope="replay",
            stage=STAGE_RECLASSIFY,
            counts=counts,
            errors=errors,
        )
        return counts, redact_secrets(errors)

    @staticmethod
    def _resolved_run_id(prefix: str, run_id: str | None) -> str:
        resolved = f"{prefix}-{uuid.uuid4()}" if run_id is None else str(run_id).strip()
        if not resolved:
            raise ValueError("run_id is required")
        return resolved

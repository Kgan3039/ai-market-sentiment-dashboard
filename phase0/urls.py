"""URL normalization used before immutable raw-item insertion."""

from __future__ import annotations

import re
from urllib.parse import (
    parse_qsl,
    unquote,
    urlencode,
    urlsplit,
    urlunsplit,
)


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "source",
}
REDIRECT_PARAMETERS = {
    "dest",
    "destination",
    "redirect",
    "target",
    "u",
    "url",
}
YAHOO_RU_PATTERN = re.compile(r"/RU=([^/]+)", re.IGNORECASE)


def _http_url(value: str) -> str | None:
    decoded = value
    for _ in range(3):
        candidate = unquote(decoded)
        if candidate == decoded:
            break
        decoded = candidate
    parts = urlsplit(decoded)
    if parts.scheme.lower() in {"http", "https"} and parts.netloc:
        return decoded
    return None


def _unwrap_yahoo_redirect(url: str) -> str:
    current = url
    for _ in range(3):
        parts = urlsplit(current)
        hostname = (parts.hostname or "").lower()
        if hostname != "yahoo.com" and not hostname.endswith(".yahoo.com"):
            break
        target = next(
            (
                _http_url(value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if key.lower() in REDIRECT_PARAMETERS and _http_url(value)
            ),
            None,
        )
        if target is None:
            match = YAHOO_RU_PATTERN.search(parts.path)
            target = _http_url(match.group(1)) if match else None
        if target is None or target == current:
            break
        current = target
    return current


def canonicalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    value = _unwrap_yahoo_redirect(value)
    parts = urlsplit(value)
    filtered_query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMETERS
    ]
    hostname = (parts.hostname or "").lower()
    port = parts.port
    netloc = hostname
    if port and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        netloc = f"{hostname}:{port}"
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (parts.scheme.lower(), netloc, path, urlencode(sorted(filtered_query)), "")
    )

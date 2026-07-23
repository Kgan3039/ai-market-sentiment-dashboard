"""URL normalization used before immutable raw-item insertion."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "source",
}


def canonicalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
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

"""Conservative URL handling for the M2 dedup stage.

Two deliberately different operations live here:

``clean_url``
    Cleanup for *display and storage*.  Produces a link a user can follow.
    It is never used as an identity key.

``url_identity_key``
    A key asserting "these two links are the same document".  A false claim
    here merges unrelated articles, so this function only applies
    transformations that are safe for *every* publisher, and returns
    ``None`` whenever it is not certain.

Specifically, the identity key does **not** strip ``www``/``m.``/``amp.``
host prefixes, does **not** strip ``/amp`` path suffixes, does **not**
collapse duplicate slashes, does **not** normalize trailing slashes, does
**not** equate ``http`` with ``https``, does **not** unwrap redirect
wrappers, and does **not** drop broad parameters such as ``ref`` or ``src``.
Any of those can address a different document at some publisher.  What it
does apply is limited to transformations defined as equivalent by RFC 3986
plus an extremely narrow allowlist of click/analytics parameters.
"""

from __future__ import annotations

import encodings.idna
import hashlib
import re
from urllib.parse import urlsplit, urlunsplit

from .text import display_text

#: Bumped when the tracking allowlist or the identity rules change.
URL_POLICY_VERSION = "m2.url.v1"

_HTTP_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": "80", "https": "443"}
_ASCII_HOST_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9\-._]*[a-z0-9])?$")
_MAX_PORT = 65535

#: The only query parameters removed before comparing two URLs.  Every entry
#: is a click or analytics identifier that no publisher routes on.  Broad
#: names such as ``ref``, ``src``, ``partner``, or ``cmpid`` are deliberately
#: absent: they are meaning-bearing at some publishers.
TRACKING_PARAMS = frozenset(
    {
        "_hsenc",
        "_hsmi",
        "dclid",
        "fbclid",
        "gbraid",
        "gclid",
        "igshid",
        "li_fat_id",
        "mc_cid",
        "mc_eid",
        "mkt_tok",
        "msclkid",
        "ttclid",
        "twclid",
        "wbraid",
        "yclid",
    }
)
#: Prefix families of the same nature.  Kept to one entry on purpose.
TRACKING_PARAM_PREFIXES = ("utm_",)


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    if lowered in TRACKING_PARAMS:
        return True
    return any(lowered.startswith(prefix) for prefix in TRACKING_PARAM_PREFIXES)


def _idna_host(hostname: str) -> str | None:
    """Return the lower-case punycode form of a host, or ``None``.

    A Unicode IDN and its already-encoded punycode spelling therefore map
    onto the same identity, which is the one host transformation that is
    unambiguously safe.
    """

    host = hostname.strip().rstrip(".")
    if not host or ".." in host:
        return None
    if host.isascii():
        encoded = host.lower()
    else:
        labels: list[str] = []
        for label in host.split("."):
            if not label:
                return None
            try:
                labels.append(encodings.idna.ToASCII(label).decode("ascii").lower())
            except (UnicodeError, ValueError):
                return None
        encoded = ".".join(labels)
    if "." not in encoded or not _ASCII_HOST_PATTERN.match(encoded):
        return None
    return encoded


def _split_authority(netloc: str) -> tuple[str | None, str | None, bool]:
    """Return ``(host, port, has_credentials)`` for a URL authority."""

    userinfo, separator, hostport = netloc.rpartition("@")
    has_credentials = bool(separator)
    if hostport.startswith("["):
        # IPv6 literal: never a syndicated news host, and bracket handling
        # is not worth the identity risk.
        return None, None, has_credentials
    hostname, _, port = hostport.partition(":")
    host = _idna_host(hostname)
    if host is None:
        return None, None, has_credentials
    if port:
        if not port.isdigit() or not 0 < int(port) <= _MAX_PORT:
            # 99999 is not a port; treating it as one would invent an
            # identity for a URL no client can even fetch.
            return None, None, has_credentials
        port = str(int(port))
    return host, port or None, has_credentials


def _parts(value: str | None) -> tuple[str, str, str | None, str, str, bool] | None:
    """Return ``(scheme, host, port, path, query, has_credentials)``."""

    text = display_text(value)
    if not text or " " in text:
        return None
    try:
        split = urlsplit(text)
    except ValueError:
        return None
    scheme = split.scheme.lower()
    if scheme not in _HTTP_SCHEMES:
        return None
    host, port, has_credentials = _split_authority(split.netloc)
    if host is None:
        return None
    if port == _DEFAULT_PORTS[scheme]:
        port = None
    return scheme, host, port, split.path, split.query, has_credentials


def _strip_tracking(query: str) -> str:
    """Drop allowlisted tracking parameters without decoding anything else.

    The query is split on ``&`` and rejoined verbatim.  Parsing and
    re-encoding would change identity semantics in three ways M2 must not
    accept: ``a+b`` and ``a%20b`` would collapse onto one another, every
    percent-escape would be rewritten into a canonical spelling the
    publisher never used, and repeated parameters would be reordered.  All
    three can address different documents, so the surviving parameters keep
    their original spelling *and* their original order.
    """

    if not query:
        return ""
    kept = [
        part
        for part in query.split("&")
        if part and not _is_tracking_param(part.split("=", 1)[0])
    ]
    return "&".join(kept)


def clean_url(value: str | None) -> str | None:
    """Return a followable, storable form of an article URL, or ``None``.

    Lower-cases the scheme and host, IDNA-normalizes the host, drops a
    default port, the fragment, any credentials, and allowlisted click
    parameters.  Path and remaining query are preserved as given, including
    their order: this value is shown to users, not compared.
    """

    parsed = _parts(value)
    if parsed is None:
        return None
    scheme, host, port, path, query, _ = parsed
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((scheme, netloc, path, _strip_tracking(query), ""))


def url_identity_key(value: str | None) -> str | None:
    """Return a conservative "same document" key, or ``None``.

    ``None`` means M2 refuses to claim identity for this link: it is
    unparseable, not HTTP(S), carries credentials (so the bare URL is not
    the whole request), or has a host it cannot normalize.  Callers must
    treat ``None`` as "no URL evidence", never as a key of its own.
    """

    parsed = _parts(value)
    if parsed is None:
        return None
    scheme, host, port, path, query, has_credentials = parsed
    if has_credentials:
        return None
    authority = f"{host}:{port}" if port else host
    key = f"{scheme}://{authority}{path}"
    cleaned_query = _strip_tracking(query)
    return f"{key}?{cleaned_query}" if cleaned_query else key


def url_host(value: str | None) -> str:
    """Return the normalized host of a URL, or the empty string."""

    parsed = _parts(value)
    return "" if parsed is None else parsed[1]


def policy_fingerprint() -> str:
    """Return a digest of this module's static URL-identity policy."""

    payload = "|".join(
        (
            URL_POLICY_VERSION,
            ",".join(sorted(TRACKING_PARAMS)),
            ",".join(TRACKING_PARAM_PREFIXES),
            _ASCII_HOST_PATTERN.pattern,
            str(_MAX_PORT),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

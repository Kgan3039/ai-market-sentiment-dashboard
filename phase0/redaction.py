"""Centralized secret redaction for everything Phase 0 persists or raises.

Every string, mapping, and sequence that can reach ``run_log.errors``,
``source_state.metadata``, or a repository exception message passes through
:func:`redact_secrets`.  The rule is deliberately blunt: when a credential
is recognized, the *whole* credential value is removed, never just the
scheme that introduces it.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


REDACTED = "[REDACTED]"

#: Names that introduce a credential.  Matched as a whole word inside
#: strings and as a substring inside mapping keys, because a key called
#: ``upstream_api_key`` is exactly as sensitive as ``api_key``.
_SECRET_NAME = (
    r"(?:proxy[-_]?authorization|authorization|auth[-_]?header"
    r"|set[-_]?cookie|cookie"
    r"|client[-_]?secret|secret[-_]?key|secret"
    r"|passwd|password|pwd|passphrase"
    r"|(?:access|refresh|id|bearer|auth|session|csrf)[-_]?token|token"
    r"|x[-_]?api[-_]?key|api[-_]?key|apikey"
    r"|private[-_]?key|session[-_]?id|sig|signature)"
)

#: Mapping keys whose value is dropped entirely, whatever its type.
SECRET_KEY_PATTERN = re.compile(
    r"(authorization|cookie|credential|password|passwd|pwd|passphrase"
    r"|secret|token|api[_-]?key|apikey|private[_-]?key|session[_-]?id"
    r"|signature)",
    re.IGNORECASE,
)

#: ``Authorization: <anything>`` — the entire header value disappears, so
#: ``Bearer abc``, ``Basic dXNlcjpwYXNz``, and a bare opaque token are all
#: removed rather than merely losing their scheme word.
_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?i)(\b(?:proxy[-_]?)?authorization\b\s*[:=](?!\s*\[REDACTED\])\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\r\n,;}\]]+)"
)

#: A credential introduced by its scheme anywhere else in a string —
#: ``Bearer abc123`` and ``Basic dXNlcjpwYXNz`` standing on their own, with
#: no ``Authorization:`` in front of them.
_SCHEME_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(Bearer|Basic|Digest)\s+"
    # A credential, not the next English word.  Ordinary prose after the
    # scheme word is either all lower-case ("Basic understanding") or
    # capitalized ("Basic Auth"); a credential is neither, because base64
    # and opaque tokens mix case and digits.  Three characters is the floor
    # so short tokens such as ``Bearer abc123`` are still caught.
    # The two lookaheads must stay case-*sensitive* while the scheme word
    # above is matched case-insensitively, hence the explicit ``(?-i:…)``.
    r"(?-i:(?![a-z]+\b)(?![A-Z][a-z]*\b))"
    r"[A-Za-z0-9._~+/=-]{3,}"
)

#: ``"api_key": "abc"`` and ``'password' = 'abc'`` keep their quoting so a
#: serialized payload stays parseable after redaction.
_QUOTED_SECRET_PATTERN = re.compile(
    r"(?i)([\"']?\b"
    + _SECRET_NAME
    + r"\b[\"']?\s*[:=]\s*)([\"'])(?!\[REDACTED\]\2)(?:\\.|(?!\2).)*\2"
)

#: ``api_key=abc``, ``password: abc`` outside quotes.
_BARE_SECRET_PATTERN = re.compile(
    r"(?i)(\b" + _SECRET_NAME + r"\b\s*[:=](?!\s*\[REDACTED\])\s*)([^\s,;&#\"'}\])]+)"
)

#: ``?api_key=abc&access_token=def``
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&]" + _SECRET_NAME + r"=)(?!\[REDACTED\])([^&\s#\"']*)"
)

#: ``https://user:password@host`` userinfo credentials.
_URL_USERINFO_PATTERN = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://)([^/\s:@\"']+):([^/\s@\"']*)@"
)


def redact_text(value: str) -> str:
    """Remove every recognized credential value from one string."""

    redacted = _AUTHORIZATION_HEADER_PATTERN.sub(rf"\1{REDACTED}", value)
    redacted = _SCHEME_CREDENTIAL_PATTERN.sub(rf"\1 {REDACTED}", redacted)
    redacted = _QUOTED_SECRET_PATTERN.sub(rf"\1\g<2>{REDACTED}\g<2>", redacted)
    redacted = _BARE_SECRET_PATTERN.sub(rf"\1{REDACTED}", redacted)
    redacted = _QUERY_SECRET_PATTERN.sub(rf"\1{REDACTED}", redacted)
    redacted = _URL_USERINFO_PATTERN.sub(rf"\1{REDACTED}:{REDACTED}@", redacted)
    return redacted


def redact_secrets(value: Any) -> Any:
    """Recursively redact credentials in strings, mappings, and sequences."""

    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if SECRET_KEY_PATTERN.search(str(key))
                else redact_secrets(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value

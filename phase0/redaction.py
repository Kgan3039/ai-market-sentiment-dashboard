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
#: ``Bearer abc``, ``Basic a``, ``Basic dXNlcjpwYXNz`` standing on their
#: own, with no ``Authorization:`` in front of them.
_SCHEME_CREDENTIAL_PATTERN = re.compile(
    r"(?i)\b(Bearer|Basic|Digest)\s+"
    # Redact the token unless it is confidently ordinary prose.  "Prose"
    # is narrow on purpose: an all-lower-case or Capitalized run of at
    # least four letters, which is what "basic understanding" and "bearer
    # instrument" look like.  Everything else goes, including one-character
    # tokens — `Bearer a` is a credential, `bearer instrument` is not, and
    # nothing in between is worth leaking to find out.
    #
    # The lookaheads must stay case-*sensitive* while the scheme word above
    # is matched case-insensitively, hence the explicit ``(?-i:…)``: with
    # the outer flag applied, ``[A-Z][a-z]*`` would happily match base64
    # such as ``dXNlcjpwYXNz``.
    r"(?-i:(?![a-z]{4,}\b)(?![A-Z][a-z]{3,}\b))"
    # A token, not its trailing punctuation: end on something that can end
    # a credential so `Bearer abc.` keeps its sentence-ending period.
    r"[A-Za-z0-9._~+/=:-]*[A-Za-z0-9=]"
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


def contains_credential(value: Any) -> bool:
    """True when redaction would change ``value``.

    Used where replacing a credential would be *worse* than refusing it:
    an identifier silently rewritten to ``[REDACTED]`` still names a row,
    a cache entry, or a model — just the wrong one, and permanently.
    """

    if not isinstance(value, str):
        return False
    return redact_text(value) != value


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

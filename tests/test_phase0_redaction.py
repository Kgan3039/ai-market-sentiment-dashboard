"""The redaction contract, and the one narrow hole cut in it.

``SECRET_KEY_PATTERN`` matches ``token`` as a substring, which is correct
for ``access_token`` and wrong for ``prompt_tokens``: a token *count* is
not a token.  LLM usage telemetry that reaches operational metadata --
run-log counts, source-state metadata, a diagnostics mapping -- was
therefore stored as ``[REDACTED]``, losing a number that is not a secret
in the first place.

The exemption cut for it is deliberately the smallest one that works, and
most of this module exists to hold it there: an *exact* approved key name,
carrying a value that is actually a count.  Everything else about
redaction -- credential keys, header and URL handling, nesting, the string
rules -- must be exactly what it was, and is asserted here too.
"""

from __future__ import annotations

import json

import pytest

from phase0.redaction import (
    REDACTED,
    SAFE_TELEMETRY_KEYS,
    contains_credential,
    redact_secrets,
    redact_text,
)


#: The keys the providers report usage under.  Named here rather than
#: imported into the assertions, so a change to the module's own frozenset
#: has to be made here too, on purpose, rather than silently agreeing with
#: itself.
APPROVED_KEYS = (
    "prompt_tokens",
    "candidate_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "completion_tokens",
)


def test_the_approved_set_is_exactly_these_six_names():
    """Widening the allowlist is a security decision, not a refactor."""

    assert SAFE_TELEMETRY_KEYS == frozenset(APPROVED_KEYS)


# ----------------------------------------------------------------------
# 1-4: what survives
# ----------------------------------------------------------------------


@pytest.mark.parametrize("key", APPROVED_KEYS)
def test_each_approved_key_keeps_a_nonnegative_count(key):
    assert redact_secrets({key: 120}) == {key: 120}


@pytest.mark.parametrize("key", APPROVED_KEYS)
@pytest.mark.parametrize(
    "spelling", [str.upper, str.title, lambda name: name.capitalize()]
)
def test_the_approved_names_are_matched_case_insensitively(key, spelling):
    written = spelling(key)
    assert redact_secrets({written: 7}) == {written: 7}


@pytest.mark.parametrize("key", APPROVED_KEYS)
def test_unknown_usage_survives_as_none(key):
    """``None`` is the providers' "not reported", and the A3 attempt
    columns store it as NULL rather than as zero; redaction must not turn
    that distinction into ``[REDACTED]`` either."""

    assert redact_secrets({key: None}) == {key: None}


@pytest.mark.parametrize("key", APPROVED_KEYS)
def test_zero_survives_and_is_not_confused_with_absence(key):
    assert redact_secrets({key: 0}) == {key: 0}


def test_a_whole_usage_block_survives_intact():
    usage = {"prompt_tokens": 120, "candidate_tokens": 40, "total_tokens": 160}

    assert redact_secrets(usage) == usage


# ----------------------------------------------------------------------
# 5-9: an approved key is not a blank cheque for its value
# ----------------------------------------------------------------------


#: Values that are not counts.  Under an approved key each one is still a
#: value a credential could be hiding in, so each one still goes.
NON_COUNT_VALUES = [
    pytest.param(True, id="bool-true"),
    pytest.param(False, id="bool-false"),
    pytest.param(-1, id="negative"),
    pytest.param(1.5, id="float"),
    pytest.param(0.0, id="float-zero"),
    pytest.param("abc", id="string"),
    pytest.param("Bearer secret", id="credential-string"),
    pytest.param("120", id="numeric-string"),
    pytest.param({"secret": "abc"}, id="mapping"),
    pytest.param(["secret"], id="list"),
    pytest.param(("secret",), id="tuple"),
    pytest.param(object(), id="object"),
]


@pytest.mark.parametrize("key", APPROVED_KEYS)
@pytest.mark.parametrize("value", NON_COUNT_VALUES)
def test_an_approved_key_carrying_anything_but_a_count_is_redacted(key, value):
    assert redact_secrets({key: value}) == {key: REDACTED}


def test_a_bool_is_not_an_integer_here():
    """``bool`` subclasses ``int``, so ``True`` would pass an ``isinstance``
    check as the count ``1``.  It is excluded explicitly, and this is the
    assertion that says so out loud."""

    assert isinstance(True, int)  # the trap
    assert redact_secrets({"prompt_tokens": True}) == {"prompt_tokens": REDACTED}
    assert redact_secrets({"prompt_tokens": 1}) == {"prompt_tokens": 1}


def test_a_credential_string_under_an_approved_key_is_removed_whole():
    """Not merely redacted *within* the string: the value is dropped."""

    redacted = redact_secrets({"total_tokens": "Authorization: Bearer sk-live-9999"})

    assert redacted == {"total_tokens": REDACTED}
    assert "sk-live-9999" not in json.dumps(redacted)


# ----------------------------------------------------------------------
# 10-11: credential keys and near misses
# ----------------------------------------------------------------------


CREDENTIAL_KEYS = [
    "access_token",
    "refresh_token",
    "api_token",
    "secret_token",
    "id_token",
    "bearer_token",
    "session_token",
    "csrf_token",
    "token",
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "password",
    "passphrase",
    "client_secret",
    "api_key",
    "x-api-key",
    "apikey",
    "private_key",
    "session_id",
    "signature",
    "credential",
]


@pytest.mark.parametrize("key", CREDENTIAL_KEYS)
def test_credential_keys_are_still_dropped_whatever_they_carry(key):
    assert redact_secrets({key: "sk-live-9999"}) == {key: REDACTED}
    # And not rescued by wearing a count-shaped value, either.
    assert redact_secrets({key: 120}) == {key: REDACTED}


#: Names that merely *resemble* an approved key.  The allowlist is an
#: exact-match set precisely so these keep no exemption: a substring rule
#: would hand one to every one of them.
NEAR_MISS_KEYS = [
    "my_prompt_tokens",
    "prompt_tokens_secret",
    "prompt_token",
    "token",
    "tokens",
    "prompt_tokens_v2",
    "x_total_tokens",
    "total_tokens_signature",
    "prompt tokens",
    "prompt_tokens ",
    " prompt_tokens",
    "prompt-tokens",
    "promptTokens",
]


@pytest.mark.parametrize("key", NEAR_MISS_KEYS)
def test_a_near_miss_key_gets_no_exemption(key):
    assert redact_secrets({key: 120}) == {key: REDACTED}


def test_a_key_that_is_not_a_secret_at_all_is_untouched():
    """The exemption narrows the secret-key rule; it does not widen it.
    A key redaction never cared about still reaches ordinary recursion."""

    assert redact_secrets({"latency_ms": 12.5}) == {"latency_ms": 12.5}
    assert redact_secrets({"model": "fake-model"}) == {"model": "fake-model"}
    assert redact_secrets({"note": "Authorization: Bearer abc123XYZ"}) == {
        "note": f"Authorization: {REDACTED}"
    }


# ----------------------------------------------------------------------
# 12-13: nesting, in both directions
# ----------------------------------------------------------------------


def test_nested_telemetry_survives_at_every_depth():
    payload = {
        "attempts": [
            {"attempt": 1, "usage": {"prompt_tokens": 120, "total_tokens": 160}},
            {"attempt": 2, "usage": {"prompt_tokens": None, "total_tokens": 0}},
        ],
        "totals": {"nested": {"candidate_tokens": 40}},
    }

    assert redact_secrets(payload) == payload


def test_nested_credential_material_beside_telemetry_still_goes():
    payload = {
        "usage": {"prompt_tokens": 120, "api_key": "sk-live-9999"},
        "headers": {"Authorization": "Basic dXNlcjpwYXNz"},
        "trace": [
            "x-api-key: sk-live-8888",
            {"access_token": "tok-abcdef", "total_tokens": 160},
        ],
    }

    redacted = redact_secrets(payload)
    serialized = json.dumps(redacted)

    # The counts are there ...
    assert redacted["usage"]["prompt_tokens"] == 120
    assert redacted["trace"][1]["total_tokens"] == 160
    # ... and not one credential is.
    for credential in ("sk-live-9999", "dXNlcjpwYXNz", "sk-live-8888", "tok-abcdef"):
        assert credential not in serialized


def test_telemetry_under_a_credential_key_does_not_come_back():
    """The parent key decides first: a whole block filed under
    ``credentials`` is dropped before its contents are ever examined."""

    redacted = redact_secrets({"credential": {"prompt_tokens": 120}})

    assert redacted == {"credential": REDACTED}


# ----------------------------------------------------------------------
# 14-15: everything that was already true, still true
# ----------------------------------------------------------------------


#: The cases the persistence contracts already hold ``redact_text`` to,
#: restated here so this module fails on its own if the allowlist ever
#: reaches the string rules.
STRING_SECRET_CASES = [
    ("Authorization: Bearer abc123XYZ", "abc123XYZ"),
    ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ("proxy-authorization=Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
    ('{"Authorization": "Basic dXNlcjpwYXNz"}', "dXNlcjpwYXNz"),
    ("x-api-key: sk-live-9999", "sk-live-9999"),
    ("api_key=SUPERSECRET&page=2", "SUPERSECRET"),
    ("https://h/f?api_key=SUPERSECRET&access_token=TOK123", "SUPERSECRET"),
    ("https://h/f?access_token=TOK123", "TOK123"),
    ("password: hunter2", "hunter2"),
    ('{"access_token": "tok-abcdef"}', "tok-abcdef"),
    ("https://user:pa55w0rd@example.com/feed", "pa55w0rd"),
    ("client_secret=shhh1", "shhh1"),
]


@pytest.mark.parametrize("text, credential", STRING_SECRET_CASES)
def test_string_redaction_is_unchanged(text, credential):
    redacted = redact_text(text)

    assert credential not in redacted
    assert REDACTED in redacted


@pytest.mark.parametrize("text, credential", STRING_SECRET_CASES)
def test_contains_credential_is_unchanged(text, credential):
    assert contains_credential(text) is True


def test_a_telemetry_string_is_not_a_credential():
    """``contains_credential`` takes strings only and is untouched by the
    allowlist, but the sentence a usage log would write must still read as
    ordinary text rather than as a credential."""

    assert contains_credential("prompt_tokens=120") is False
    assert redact_text("prompt_tokens=120") == "prompt_tokens=120"
    # The string rules are separate from the key rules: a bare `token=`
    # in free text is still a credential introduction, as it always was.
    assert contains_credential("token=abc") is True


def test_no_raw_credential_survives_a_serialized_mixed_payload():
    """The end-to-end shape: telemetry and credentials in one operational
    blob, serialized the way the repository stores it."""

    payload = {
        "usage": {
            "prompt_tokens": 120,
            "candidate_tokens": 40,
            "total_tokens": 160,
            "completion_tokens": None,
        },
        "request": {
            "headers": {
                "Authorization": "Bearer sk-live-9999",
                "x-api-key": "sk-live-8888",
            },
            "url": "https://user:pa55w0rd@h/f?api_key=SUPERSECRET",
        },
        "errors": ["Authorization: Basic dXNlcjpwYXNz", {"password": "hunter2"}],
    }

    serialized = json.dumps(redact_secrets(payload))

    for credential in (
        "sk-live-9999",
        "sk-live-8888",
        "pa55w0rd",
        "SUPERSECRET",
        "dXNlcjpwYXNz",
        "hunter2",
    ):
        assert credential not in serialized
    assert '"prompt_tokens": 120' in serialized
    assert '"completion_tokens": null' in serialized


def test_redaction_still_copies_rather_than_mutating():
    """An exempt value is returned as-is; the caller's container is not."""

    payload = {"usage": {"prompt_tokens": 120, "api_key": "sk-live-9999"}}
    redacted = redact_secrets(payload)

    assert payload["usage"]["api_key"] == "sk-live-9999"
    assert redacted is not payload
    assert redacted["usage"] is not payload["usage"]


def test_redaction_is_idempotent():
    payload = {"prompt_tokens": 120, "api_key": "sk-live-9999"}
    once = redact_secrets(payload)

    assert redact_secrets(once) == once

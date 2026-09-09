"""Tests for phase0.redaction's LLM-telemetry allowlist (issue #73 / A3).

The rest of phase0.redaction's behavior is exercised indirectly by
tests/test_phase0_pipeline.py, tests/test_phase0_persistence_contracts.py,
and tests/test_phase0_remote_v4_compat.py; this file covers only the
allowlist addition, in isolation.
"""

from __future__ import annotations

from phase0.redaction import REDACTED, SAFE_TELEMETRY_KEYS, redact_secrets


def test_allowlisted_telemetry_keys_survive_redaction():
    payload = {
        "input_tokens": 126,
        "output_tokens": 21,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert redact_secrets(payload) == payload


def test_allowlist_is_exact_not_a_widened_pattern():
    # "api_token" contains "token" and is NOT on the allowlist - it must
    # still be redacted. The allowlist exempts specific names, never a
    # pattern, so this stays true regardless of what SAFE_TELEMETRY_KEYS
    # ever grows to contain.
    payload = {"api_token": "abc123", "session_token": "def456", "input_tokens": 10}
    result = redact_secrets(payload)
    assert result["api_token"] == REDACTED
    assert result["session_token"] == REDACTED
    assert result["input_tokens"] == 10


def test_allowlist_match_is_case_insensitive_on_the_key():
    assert redact_secrets({"Input_Tokens": 5}) == {"Input_Tokens": 5}


def test_allowlisted_key_values_are_still_recursively_redacted():
    # The allowlist exempts the key from the blanket wipe, not the value
    # from ordinary text redaction - a telemetry field holding a
    # credential-shaped string is still caught by redact_text.
    payload = {"input_tokens": "Bearer abc123XYZ"}
    assert redact_secrets(payload) == {"input_tokens": f"Bearer {REDACTED}"}


def test_safe_telemetry_keys_are_exactly_the_documented_set():
    assert SAFE_TELEMETRY_KEYS == {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }

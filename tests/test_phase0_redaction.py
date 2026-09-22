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
        "prompt_tokens": 100,
        "candidate_tokens": 20,
        "total_tokens": 120,
        "input_tokens": 100,
        "output_tokens": 20,
        "completion_tokens": 20,
    }
    assert redact_secrets(payload) == payload


def test_allowlist_is_exact_not_a_widened_pattern():
    # "api_token" and "session_token" contain "token" and are NOT on the
    # allowlist - they must still be redacted. The allowlist exempts
    # specific names, never a pattern, so this stays true regardless of
    # what SAFE_TELEMETRY_KEYS ever grows to contain.
    payload = {"api_token": "abc123", "session_token": "def456", "prompt_tokens": 10}
    result = redact_secrets(payload)
    assert result["api_token"] == REDACTED
    assert result["session_token"] == REDACTED
    assert result["prompt_tokens"] == 10


def test_allowlist_match_is_case_insensitive_on_the_key():
    assert redact_secrets({"Prompt_Tokens": 5}) == {"Prompt_Tokens": 5}


def test_allowlisted_key_values_are_still_recursively_redacted():
    # The allowlist exempts the key from the blanket wipe, not the value
    # from ordinary text redaction - a telemetry field holding a
    # credential-shaped string is still caught by redact_text.
    payload = {"prompt_tokens": "Bearer abc123XYZ"}
    assert redact_secrets(payload) == {"prompt_tokens": f"Bearer {REDACTED}"}


def test_safe_telemetry_keys_matches_generation_usage_field_names():
    # ai.summarization.GenerationUsage's actual fields (prompt_tokens,
    # candidate_tokens, total_tokens) must be covered; the rest are kept
    # for forward/backward naming tolerance.
    assert {"prompt_tokens", "candidate_tokens", "total_tokens"} <= SAFE_TELEMETRY_KEYS

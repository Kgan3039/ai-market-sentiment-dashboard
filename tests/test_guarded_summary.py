"""A2: guarded summary generation over a frozen input (``ai.guarded_summary``).

Every test is offline.  ``google.genai`` is never imported: the clients here
are fakes honouring ``generate(system_prompt, user_prompt, response_schema)``,
and one test asserts that importing the module under test does not pull the
SDK in.
"""

from __future__ import annotations

import ast
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai import guarded_summary as gs
from ai.guarded_summary import (
    CODE_BANNED_LANGUAGE,
    CODE_BLANK_SENTENCE,
    CODE_INVALID_LABEL,
    CODE_INVALID_SENTENCE_COUNT,
    CODE_MALFORMED_OUTPUT,
    CODE_MISSING_CITATION,
    CODE_UNKNOWN_CITATION,
    MAX_ATTEMPTS,
    OUTCOME_ACCEPTED,
    OUTCOME_PROVIDER_ERROR,
    OUTCOME_PROVIDER_TIMEOUT,
    OUTCOME_PROVIDER_UNCONFIGURED,
    OUTCOME_REJECTED,
    REASON_PROVIDER_UNAVAILABLE,
    REASON_PROVIDER_UNCONFIGURED,
    REASON_VALIDATION_EXHAUSTED,
    STATUS_ACCEPTED,
    STATUS_UNAVAILABLE,
    SYSTEM_PROMPT_A2,
    VALIDATION_CODES,
    EvidenceStory,
    GuardedSummaryError,
    SummaryGenerationInput,
    ThemeReference,
    build_prompt,
    compute_policy_fingerprint,
    generate_guarded_summary,
    load_copy_rules,
    validate_candidate,
)
from ai.summarization import (
    GeminiClient,
    GenerationUsage,
    MalformedOutputError,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderTimeoutError,
    ThemeSummary,
)

ROOT = Path(__file__).resolve().parents[1]
ID_LINE_RE = re.compile(r"- id: (\S+)")

# A phrase the banned-language rules flag as advisory in generated copy.
BANNED = "Investors should buy the stock now."


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def evidence(*ids: int, description: str = "A standfirst.") -> list[EvidenceStory]:
    return [
        EvidenceStory(
            citation_id=f"story:{identifier}",
            persisted_story_id=identifier,
            title=f"Headline {identifier}",
            description=description,
            outlet="Reuters",
            published_at="2026-07-13T09:05:00+00:00",
            raw_item_ids=(identifier * 10, identifier * 10 + 1),
            urls=(f"https://publisher.example/{identifier}",),
        )
        for identifier in ids
    ]


def theme() -> ThemeReference:
    return ThemeReference(
        theme_id=3, theme_key="k3", label="Label", pipeline_version="v1"
    )


@pytest.fixture
def generation_input() -> SummaryGenerationInput:
    return SummaryGenerationInput.compose(
        ticker="TSLA",
        trading_day="2026-07-13",
        theme=theme(),
        evidence=evidence(17, 204),
    )


@pytest.fixture(scope="module")
def rules():
    return load_copy_rules()


def good(ids, *, label="Delivery coverage", texts=("First.", "Second.")):
    return {
        "label": label,
        "sentences": [{"text": text, "citation_ids": list(ids)} for text in texts],
    }


class ScriptedClient:
    """Returns (or raises) one scripted answer per call, in order."""

    model = "fake-model"
    max_output_tokens = 512

    def __init__(self, *answers, usage=None):
        self.answers = list(answers)
        self.calls: list[tuple[str, str, type]] = []
        self.last_usage = usage

    def generate(self, system_prompt, user_prompt, response_schema):
        self.calls.append((system_prompt, user_prompt, response_schema))
        if not self.answers:
            raise AssertionError("client called more times than scripted")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            return answer(user_prompt, response_schema)
        if isinstance(answer, dict):
            return response_schema.model_validate(answer)
        return answer


def echo_valid(user_prompt, response_schema):
    """A schema-valid answer citing ids that were actually in the prompt."""

    ids = ID_LINE_RE.findall(user_prompt)
    return response_schema.model_validate(good(ids[:1]))


# ----------------------------------------------------------------------
# The frozen input
# ----------------------------------------------------------------------


def test_input_is_frozen_and_fingerprint_is_intrinsic(generation_input):
    with pytest.raises(dataclasses.FrozenInstanceError):
        generation_input.ticker = "NVDA"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        generation_input.evidence[0].title = "x"  # type: ignore[misc]
    assert isinstance(generation_input.evidence, tuple)
    assert isinstance(generation_input.evidence_ids, frozenset)
    assert generation_input.evidence_ids == {"story:17", "story:204"}
    # A hand-built instance claiming a fingerprint it does not have is refused.
    with pytest.raises(GuardedSummaryError):
        SummaryGenerationInput(
            ticker="TSLA",
            trading_day="2026-07-13",
            theme=theme(),
            evidence=generation_input.evidence,
            evidence_ids=generation_input.evidence_ids,
            input_fingerprint="0" * 64,
        )
    with pytest.raises(GuardedSummaryError):
        SummaryGenerationInput(
            ticker="TSLA",
            trading_day="2026-07-13",
            theme=theme(),
            evidence=generation_input.evidence,
            evidence_ids=frozenset({"story:17"}),
            input_fingerprint=generation_input.input_fingerprint,
        )


def test_fingerprint_tracks_model_visible_evidence_only():
    base = SummaryGenerationInput.compose(
        ticker="TSLA", trading_day="2026-07-13", theme=theme(), evidence=evidence(1, 2)
    )
    same = SummaryGenerationInput.compose(
        ticker="TSLA", trading_day="2026-07-13", theme=theme(), evidence=evidence(1, 2)
    )
    assert base.input_fingerprint == same.input_fingerprint

    changed_text = SummaryGenerationInput.compose(
        ticker="TSLA",
        trading_day="2026-07-13",
        theme=theme(),
        evidence=evidence(1, 2, description="Different standfirst."),
    )
    assert changed_text.input_fingerprint != base.input_fingerprint

    reordered = SummaryGenerationInput.compose(
        ticker="TSLA", trading_day="2026-07-13", theme=theme(), evidence=evidence(2, 1)
    )
    assert reordered.input_fingerprint != base.input_fingerprint

    other_theme = SummaryGenerationInput.compose(
        ticker="TSLA",
        trading_day="2026-07-13",
        theme=dataclasses.replace(theme(), theme_id=4),
        evidence=evidence(1, 2),
    )
    assert other_theme.input_fingerprint != base.input_fingerprint

    # Provenance is carried beside the evidence, not shown to the model, and
    # so does not move the fingerprint.
    provenance_only = [
        dataclasses.replace(s, urls=("https://x/",)) for s in evidence(1, 2)
    ]
    same_visible = SummaryGenerationInput.compose(
        ticker="TSLA", trading_day="2026-07-13", theme=theme(), evidence=provenance_only
    )
    assert same_visible.input_fingerprint == base.input_fingerprint


def test_every_model_visible_input_change_changes_the_fingerprint():
    """Field by field: visible in the prompt <=> moves the fingerprint.

    Each case changes exactly one field to a distinctive value, then asks
    two questions of the *actual* prompt builder and the *actual*
    fingerprint: is the new value rendered, and did the fingerprint move?
    The two answers must agree, so the fingerprint can never silently
    drift from what the model is shown.
    """

    def compose(ticker="TSLA", trading_day="2026-07-13", theme_ref=None, stories=None):
        return SummaryGenerationInput.compose(
            ticker=ticker,
            trading_day=trading_day,
            theme=theme_ref or theme(),
            evidence=stories or evidence(1, 2),
        )

    base = compose()
    base_prompt = build_prompt(base)

    def story_with(**changes):
        first, second = evidence(1, 2)
        return [dataclasses.replace(first, **changes), second]

    # (name, variant input, the distinctive value, expected to be visible)
    cases = [
        ("ticker", compose(ticker="NVDA"), "NVDA", True),
        ("trading_day", compose(trading_day="2031-01-02"), "2031-01-02", True),
        ("citation_id", compose(stories=evidence(1, 7)), "story:7", True),
        ("title", compose(stories=story_with(title="ZZ-TITLE")), "ZZ-TITLE", True),
        (
            "description",
            compose(stories=story_with(description="ZZ-DESC")),
            "ZZ-DESC",
            True,
        ),
        ("outlet", compose(stories=story_with(outlet="ZZ-OUTLET")), "ZZ-OUTLET", True),
        (
            "published_at",
            compose(stories=story_with(published_at="2031-01-02T03:04:05+00:00")),
            "2031-01-02T03:04:05+00:00",
            True,
        ),
        (
            "theme_label",
            compose(theme_ref=dataclasses.replace(theme(), label="ZZ-LABEL")),
            "ZZ-LABEL",
            False,
        ),
        (
            "raw_item_ids",
            compose(stories=story_with(raw_item_ids=(987654,))),
            "987654",
            False,
        ),
        (
            "urls",
            compose(stories=story_with(urls=("https://zz.example/",))),
            "zz.example",
            False,
        ),
    ]
    for name, variant, marker, visible in cases:
        prompt = build_prompt(variant)
        assert marker not in base_prompt, name
        assert (marker in prompt) is visible, f"{name}: visibility"
        moved = variant.input_fingerprint != base.input_fingerprint
        if visible:
            assert moved, f"{name} is model-visible but did not move the fingerprint"
        elif name == "theme_label":
            # Not shown to the model, and not part of the fingerprint either.
            assert not moved, f"{name} is not model-visible but moved the fingerprint"
        else:
            assert not moved, f"{name} is provenance and must not move the fingerprint"

    # Evidence order is model-visible (the prompt lists stories in order).
    reordered = compose(stories=evidence(2, 1))
    assert ID_LINE_RE.findall(build_prompt(reordered)) == ["story:2", "story:1"]
    assert ID_LINE_RE.findall(base_prompt) == ["story:1", "story:2"]
    assert reordered.input_fingerprint != base.input_fingerprint

    # The identity fields that are hashed but not rendered are exactly the
    # theme's durable id, key, and pipeline version -- and nothing else.
    for field_name, value in (
        ("theme_id", 99),
        ("theme_key", "ZZ-KEY"),
        ("pipeline_version", "ZZ-V"),
    ):
        variant = compose(theme_ref=dataclasses.replace(theme(), **{field_name: value}))
        assert str(value) not in build_prompt(variant), field_name
        assert variant.input_fingerprint != base.input_fingerprint, field_name


def test_evidence_and_input_refuse_broken_shapes():
    with pytest.raises(GuardedSummaryError):
        EvidenceStory("story:2", 1, "t", "d", "o", "p")  # id names another story
    with pytest.raises(GuardedSummaryError):
        EvidenceStory("story:1", 1, "   ", "d", "o", "p")  # no title
    with pytest.raises(GuardedSummaryError):
        SummaryGenerationInput.compose(
            ticker="TSLA", trading_day="d", theme=theme(), evidence=[]
        )
    with pytest.raises(GuardedSummaryError):
        SummaryGenerationInput.compose(
            ticker="TSLA",
            trading_day="d",
            theme=theme(),
            evidence=evidence(1) + evidence(1),
        )


def test_prompt_contains_only_the_frozen_evidence(generation_input):
    prompt = build_prompt(generation_input)
    assert ID_LINE_RE.findall(prompt) == ["story:17", "story:204"]
    assert "Headline 17" in prompt and "Headline 204" in prompt
    assert "raw_item" not in prompt and "publisher.example" not in prompt
    assert "Validation feedback" not in prompt


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_valid_candidate_is_accepted(generation_input, rules):
    verdict = validate_candidate(good(["story:17"]), generation_input, rules=rules)
    assert verdict.accepted
    assert isinstance(verdict.summary, ThemeSummary)
    assert [s.citation_ids for s in verdict.summary.sentences] == [["story:17"]] * 2


def test_unknown_citation_rejects_the_whole_attempt(generation_input, rules):
    candidate = good(["story:17"])
    candidate["sentences"][1]["citation_ids"] = ["story:17", "story:999"]
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.summary is None
    assert verdict.codes == (CODE_UNKNOWN_CITATION,)
    assert verdict.failures[0].detail == "sentence 2: 1 unknown citation"
    assert "999" not in verdict.failures[0].detail
    # A bare number, or an id from another theme, is just as unknown.
    for bad in ("17", "story:9999", "story:17 "):
        candidate["sentences"][1]["citation_ids"] = [bad]
        assert validate_candidate(candidate, generation_input, rules=rules).codes == (
            CODE_UNKNOWN_CITATION,
        )


def test_missing_citation_is_rejected(generation_input, rules):
    candidate = good(["story:17"])
    candidate["sentences"][0]["citation_ids"] = []
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.codes == (CODE_MISSING_CITATION,)
    assert verdict.failures[0].detail == "sentence 1"


@pytest.mark.parametrize(
    "candidate",
    [
        "not json at all",
        '["a", "list"]',
        {"label": "x"},
        {"label": 5, "sentences": []},
        {"label": "x", "sentences": ["not an object"]},
        {"label": "x", "sentences": [{"text": "t", "citation_ids": "story:17"}]},
        {"label": "x", "sentences": [{"text": "t", "citation_ids": [17]}]},
        None,
        42,
    ],
)
def test_malformed_candidate_is_rejected(candidate, generation_input, rules):
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.summary is None
    assert verdict.codes == (CODE_MALFORMED_OUTPUT,)


def test_blank_and_empty_outputs_are_rejected(generation_input, rules):
    empty = validate_candidate(
        {"label": "", "sentences": []}, generation_input, rules=rules
    )
    assert empty.codes == (CODE_INVALID_LABEL, CODE_INVALID_SENTENCE_COUNT)

    blank = good(["story:17"], texts=("   ", "Second."))
    verdict = validate_candidate(blank, generation_input, rules=rules)
    assert verdict.codes == (CODE_BLANK_SENTENCE,)
    assert verdict.failures[0].detail == "sentence 1"


@pytest.mark.parametrize("count", [0, 1, 5, 6])
def test_sentence_count_outside_two_to_four_is_rejected(count, generation_input, rules):
    candidate = good(["story:17"], texts=tuple(f"Sentence {n}." for n in range(count)))
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert CODE_INVALID_SENTENCE_COUNT in verdict.codes
    assert verdict.summary is None


@pytest.mark.parametrize("count", [2, 3, 4])
def test_sentence_count_inside_bounds_is_accepted(count, generation_input, rules):
    candidate = good(["story:204"], texts=tuple(f"Sentence {n}." for n in range(count)))
    assert validate_candidate(candidate, generation_input, rules=rules).accepted


def test_label_bounds(generation_input, rules):
    nine = good(["story:17"], label="one two three four five six seven eight nine")
    assert validate_candidate(nine, generation_input, rules=rules).codes == (
        CODE_INVALID_LABEL,
    )
    eight = good(["story:17"], label="one two three four five six seven eight")
    assert validate_candidate(eight, generation_input, rules=rules).accepted


def test_duplicate_valid_citations_are_deduped_preserving_order(
    generation_input, rules
):
    candidate = good(["story:204", "story:17", "story:204", "story:17", "story:204"])
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.accepted
    for sentence in verdict.summary.sentences:
        assert sentence.citation_ids == ["story:204", "story:17"]


def test_duplicates_never_hide_an_unknown_citation(generation_input, rules):
    candidate = good(["story:17", "story:17", "story:5", "story:5"])
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.summary is None
    assert verdict.codes == (CODE_UNKNOWN_CITATION, CODE_UNKNOWN_CITATION)
    assert [f.detail for f in verdict.failures] == [
        "sentence 1: 1 unknown citation",
        "sentence 2: 1 unknown citation",
    ]


def test_banned_generated_sentence_is_rejected(generation_input, rules):
    candidate = good(["story:17"], texts=("Coverage is broad.", BANNED))
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.codes == (CODE_BANNED_LANGUAGE,)
    assert verdict.failures[0].detail == "sentence 2: advisory"
    assert BANNED not in verdict.failures[0].detail


def test_banned_generated_label_is_rejected(generation_input, rules):
    candidate = good(["story:17"], label="Shares will rise")
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.codes == (CODE_BANNED_LANGUAGE,)
    assert verdict.failures[0].detail == "label: prediction"


def test_banned_phrase_inside_publisher_evidence_is_not_linted(rules):
    """The rules apply to generated copy; publisher text is evidence."""

    tainted = SummaryGenerationInput.compose(
        ticker="TSLA",
        trading_day="2026-07-13",
        theme=theme(),
        evidence=[
            EvidenceStory(
                citation_id="story:1",
                persisted_story_id=1,
                title="Analyst: the stock fell because of tariffs",
                description=BANNED,
                outlet="Outlet",
                published_at="2026-07-13T09:05:00+00:00",
            )
        ],
    )
    # The input builds, the prompt carries the publisher text verbatim, and a
    # clean generated summary over it is accepted.
    assert BANNED in build_prompt(tainted)
    verdict = validate_candidate(
        good(["story:1"], texts=("Coverage leads with tariffs.", "Analysts weigh in.")),
        tainted,
        rules=rules,
    )
    assert verdict.accepted


def test_failures_are_reported_in_canonical_order(generation_input, rules):
    candidate = {
        "label": "",
        "sentences": [
            {"text": " ", "citation_ids": []},
            {"text": BANNED, "citation_ids": ["story:2"]},
        ],
    }
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert verdict.codes == (
        CODE_INVALID_LABEL,
        CODE_BLANK_SENTENCE,
        CODE_MISSING_CITATION,
        CODE_UNKNOWN_CITATION,
        CODE_BANNED_LANGUAGE,
    )
    ranks = [VALIDATION_CODES.index(code) for code in verdict.codes]
    assert ranks == sorted(ranks)


# ----------------------------------------------------------------------
# The guarded loop
# ----------------------------------------------------------------------


def test_valid_result_accepted_in_one_attempt(generation_input, rules):
    usage = GenerationUsage(prompt_tokens=120, candidate_tokens=40, total_tokens=160)
    client = ScriptedClient(good(["story:17"]), usage=usage)
    result = generate_guarded_summary(generation_input, client=client, rules=rules)

    assert result.status == STATUS_ACCEPTED and result.accepted
    assert result.reason is None
    assert result.accepted_attempt == 1
    assert result.provider_calls == 1 == len(client.calls)
    assert isinstance(result.summary, ThemeSummary)
    assert result.input_fingerprint == generation_input.input_fingerprint
    assert result.theme == generation_input.theme
    assert result.max_attempts == MAX_ATTEMPTS == 2
    (attempt,) = result.attempts
    assert attempt.outcome == OUTCOME_ACCEPTED
    assert attempt.usage == usage
    assert attempt.latency_ms is not None and attempt.latency_ms >= 0
    assert "semantic_faithfulness_not_established" in result.guarantee
    # The client saw the A2 prompt and the permissive candidate schema.
    system_prompt, user_prompt, schema = client.calls[0]
    assert system_prompt == SYSTEM_PROMPT_A2
    assert schema is gs.CandidateSummary
    assert ID_LINE_RE.findall(user_prompt) == ["story:17", "story:204"]


def test_missing_usage_stays_none(generation_input, rules):
    client = ScriptedClient(good(["story:17"]))  # no last_usage attribute value
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.attempts[0].usage is None

    class Bare:
        def generate(self, system_prompt, user_prompt, response_schema):
            return response_schema.model_validate(good(["story:17"]))

    result = generate_guarded_summary(generation_input, client=Bare(), rules=rules)
    assert result.attempts[0].usage is None


def test_first_invalid_then_valid_is_accepted_with_feedback(generation_input, rules):
    client = ScriptedClient(good(["story:999"]), echo_valid)
    result = generate_guarded_summary(generation_input, client=client, rules=rules)

    assert result.accepted and result.accepted_attempt == 2
    assert [a.outcome for a in result.attempts] == [OUTCOME_REJECTED, OUTCOME_ACCEPTED]
    # Both sentences cited the unknown id: one failure per sentence.
    assert result.attempts[0].validation_codes == (CODE_UNKNOWN_CITATION,) * 2
    first_prompt, second_prompt = (call[1] for call in client.calls)
    assert "Validation feedback" not in first_prompt
    assert "Validation feedback" in second_prompt
    assert "- unknown_citation:" in second_prompt


def test_retry_feedback_carries_only_safe_information(generation_input, rules):
    """Nothing the model said, and nothing the provider said, is echoed."""

    poisoned = good(["story:SECRET-LEAK"], label="LEAKED LABEL", texts=(BANNED, "x"))
    client = ScriptedClient(poisoned, echo_valid)
    generate_guarded_summary(generation_input, client=client, rules=rules)
    second_prompt = client.calls[1][1]
    feedback = second_prompt.split("Validation feedback", 1)[1]
    for leaked in ("SECRET-LEAK", "LEAKED LABEL", BANNED):
        assert leaked not in feedback
    for line in feedback.splitlines()[1:]:
        code = line[2:].split(":", 1)[0]
        assert code in VALIDATION_CODES
        assert line == f"- {code}: {gs.FEEDBACK_MESSAGES[code]}"


def test_retry_uses_identical_evidence_and_fingerprint(generation_input, rules):
    client = ScriptedClient(good([]), good(["story:17"]))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.accepted
    first, second = (call[1] for call in client.calls)
    evidence_block = build_prompt(generation_input)
    assert first == evidence_block
    assert second.startswith(evidence_block)
    assert ID_LINE_RE.findall(first) == ID_LINE_RE.findall(second)
    assert result.input_fingerprint == generation_input.input_fingerprint
    # Both calls carried the same input identity; nothing new was introduced.
    assert second[len(evidence_block) :].strip().startswith("Validation feedback")


def test_all_attempts_invalid_is_unavailable_with_no_summary(generation_input, rules):
    client = ScriptedClient(good(["story:1"]), good(["story:2"]), good(["story:17"]))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)

    assert result.status == STATUS_UNAVAILABLE and not result.accepted
    assert result.summary is None
    assert result.reason == REASON_VALIDATION_EXHAUSTED
    assert result.accepted_attempt is None
    assert len(client.calls) == MAX_ATTEMPTS == result.provider_calls
    assert client.answers == [
        good(["story:17"])
    ]  # the third answer was never asked for
    assert all(a.outcome == OUTCOME_REJECTED for a in result.attempts)


@pytest.mark.parametrize("max_attempts", [1, 2])
def test_provider_calls_are_bounded_by_max_attempts(
    max_attempts, generation_input, rules
):
    client = ScriptedClient(*([good(["story:0"])] * 10))
    result = generate_guarded_summary(
        generation_input, client=client, max_attempts=max_attempts, rules=rules
    )
    assert len(client.calls) == max_attempts == result.provider_calls
    assert result.status == STATUS_UNAVAILABLE
    assert result.max_attempts == max_attempts


def test_transient_provider_failure_then_valid_result(generation_input, rules):
    client = ScriptedClient(ProviderRequestError("503 upstream"), good(["story:204"]))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.accepted and result.accepted_attempt == 2
    assert result.attempts[0].outcome == OUTCOME_PROVIDER_ERROR
    assert result.attempts[0].error == "503 upstream"
    # A provider failure is not model feedback: the second prompt is plain.
    assert "Validation feedback" not in client.calls[1][1]


def test_provider_exhaustion_is_unavailable(generation_input, rules):
    client = ScriptedClient(ProviderRequestError("a"), ProviderRequestError("b"))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.status == STATUS_UNAVAILABLE
    assert result.reason == REASON_PROVIDER_UNAVAILABLE
    assert result.summary is None
    assert [a.outcome for a in result.attempts] == [OUTCOME_PROVIDER_ERROR] * 2


def test_missing_configuration_fails_without_a_second_call(generation_input, rules):
    client = ScriptedClient(
        ProviderConfigurationError("GEMINI_API_KEY is not configured")
    )
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.status == STATUS_UNAVAILABLE
    assert result.reason == REASON_PROVIDER_UNCONFIGURED
    assert len(client.calls) == 1
    assert result.attempts[0].outcome == OUTCOME_PROVIDER_UNCONFIGURED


def test_timeout_is_controlled(generation_input, rules):
    client = ScriptedClient(ProviderTimeoutError("timed out"), good(["story:17"]))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.accepted and result.attempts[0].outcome == OUTCOME_PROVIDER_TIMEOUT

    client = ScriptedClient(ProviderTimeoutError("t1"), ProviderTimeoutError("t2"))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.status == STATUS_UNAVAILABLE
    assert result.reason == REASON_PROVIDER_UNAVAILABLE
    assert len(client.calls) == 2


def test_malformed_provider_answer_is_a_retryable_rejection(generation_input, rules):
    def bad_schema(user_prompt, response_schema):
        return response_schema.model_validate({"label": 5})

    for failure in (MalformedOutputError("not json"), bad_schema):
        client = ScriptedClient(failure, good(["story:17"]))
        result = generate_guarded_summary(generation_input, client=client, rules=rules)
        assert result.accepted and result.accepted_attempt == 2
        assert result.attempts[0].outcome == OUTCOME_REJECTED
        assert result.attempts[0].validation_codes == (CODE_MALFORMED_OUTPUT,)
        assert "- malformed_output:" in client.calls[1][1]


def test_programmer_defects_propagate(generation_input, rules):
    class BrokenContract:
        def generate(self, system_prompt):  # wrong call shape
            return None

    with pytest.raises(TypeError):
        generate_guarded_summary(generation_input, client=BrokenContract(), rules=rules)

    client = ScriptedClient(KeyError("boom"))
    with pytest.raises(KeyError):
        generate_guarded_summary(generation_input, client=client, rules=rules)

    with pytest.raises(ValueError):
        generate_guarded_summary(
            generation_input, client=ScriptedClient(), max_attempts=0, rules=rules
        )
    with pytest.raises(GuardedSummaryError):
        generate_guarded_summary(
            {"not": "an input"}, client=ScriptedClient(), rules=rules
        )


def test_provider_errors_are_redacted_before_they_reach_the_result(
    generation_input, rules
):
    leak = ProviderRequestError("401 for key api_key=AIzaSyA-secret-value-1234567890")
    client = ScriptedClient(leak, good(["story:17"]))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert "AIzaSyA" not in str(leak)
    assert "AIzaSyA" not in result.attempts[0].error
    assert "[REDACTED]" in result.attempts[0].error


def test_latency_is_measured_with_the_injected_clock(generation_input, rules):
    ticks = iter([10.0, 10.25, 20.0, 20.5])
    client = ScriptedClient(good(["story:0"]), good(["story:17"]))
    result = generate_guarded_summary(
        generation_input, client=client, rules=rules, clock=lambda: next(ticks)
    )
    assert [a.latency_ms for a in result.attempts] == [250.0, 500.0]


# ----------------------------------------------------------------------
# Policy fingerprint
# ----------------------------------------------------------------------


def test_policy_fingerprint_is_deterministic_and_sensitive(rules):
    base = compute_policy_fingerprint(model="m", max_attempts=2, rules=rules)
    assert base == compute_policy_fingerprint(model="m", max_attempts=2, rules=rules)
    assert re.fullmatch(r"[0-9a-f]{64}", base)
    assert base != compute_policy_fingerprint(model="m2", max_attempts=2, rules=rules)
    assert base != compute_policy_fingerprint(model="m", max_attempts=3, rules=rules)
    assert base != compute_policy_fingerprint(
        model="m", max_attempts=2, rules=rules, max_output_tokens=2
    )
    assert base != compute_policy_fingerprint(
        model="m", max_attempts=2, rules=rules, temperature=0.9
    )
    fewer_rules = tuple(rules)[:-1]
    assert base != compute_policy_fingerprint(
        model="m", max_attempts=2, rules=fewer_rules
    )

    client = ScriptedClient(good(["story:17"]))
    generation_input = SummaryGenerationInput.compose(
        ticker="TSLA", trading_day="2026-07-13", theme=theme(), evidence=evidence(17)
    )
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.policy_fingerprint == compute_policy_fingerprint(
        model="fake-model", max_attempts=2, rules=rules, max_output_tokens=512
    )


# ----------------------------------------------------------------------
# No network, no SDK, no forbidden dependencies
# ----------------------------------------------------------------------


def test_module_never_imports_the_provider_sdk_or_forbidden_layers():
    # A fresh interpreter, so other tests loading the SDK cannot mask it.
    probe = (
        "import sys, ai.guarded_summary, phase0.summaries; "
        "sys.exit(0 if 'google.genai' not in sys.modules and "
        "'backend' not in sys.modules else 1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    for path in (ROOT / "ai" / "guarded_summary.py", ROOT / "phase0" / "summaries.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        forbidden = {"google", "google.genai", "backend", "frontend", "langchain"}
        assert not {name for name in imported if name.split(".")[0] in forbidden}
        assert "nlp.themes.summarization" not in imported
        assert "nlp.eval.review" not in imported


def test_default_rules_load_from_the_phase0_source_of_truth(generation_input):
    """No ``rules`` argument means the committed banned-phrase file."""

    client = ScriptedClient(good(["story:17"], texts=(BANNED, "x")), good(["story:17"]))
    result = generate_guarded_summary(generation_input, client=client)
    assert result.accepted_attempt == 2
    assert result.attempts[0].validation_codes == (CODE_BANNED_LANGUAGE,)


def test_pydantic_validation_error_from_a_fake_is_treated_as_malformed(
    generation_input, rules
):
    """A fake that validates through the response schema raises ValidationError."""

    def raises(user_prompt, response_schema):
        return response_schema.model_validate({"sentences": "nope"})

    with pytest.raises(ValidationError):
        raises("", gs.CandidateSummary)
    client = ScriptedClient(raises, good(["story:17"]))
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.accepted_attempt == 2
    assert result.attempts[0].validation_codes == (CODE_MALFORMED_OUTPUT,)


# ----------------------------------------------------------------------
# P2-1: provider errors are classified by their real types
# ----------------------------------------------------------------------


class _RaisingModels:
    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        raise self._exc


class _RaisingSdkClient:
    def __init__(self, exc):
        self.models = _RaisingModels(exc)


def _client_raising(exc) -> GeminiClient:
    """A real GeminiClient whose (fake) SDK client raises ``exc``."""

    client = GeminiClient(model="fake-model", api_key="k-not-a-real-key")
    client._client = _RaisingSdkClient(exc)
    return client


def _api_error(code: int, status: str):
    from google.genai import errors

    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "status": status, "message": "m"}})


@pytest.mark.parametrize("name", ["HTTPError", "TimeoutException", "TransportError"])
def test_an_application_exception_named_like_a_transport_error_propagates(name):
    fake = type(name, (Exception,), {})
    with pytest.raises(fake):
        _client_raising(fake("not the transport")).generate("s", "u", ThemeSummary)


@pytest.mark.parametrize("exc_type", [TypeError, KeyError, RuntimeError])
def test_programmer_exceptions_from_the_provider_call_propagate(exc_type):
    with pytest.raises(exc_type):
        _client_raising(exc_type("boom")).generate("s", "u", ThemeSummary)


def test_a_real_httpx_timeout_normalizes_to_a_timeout():
    import httpx

    with pytest.raises(ProviderTimeoutError):
        _client_raising(httpx.ReadTimeout("read timed out")).generate(
            "s", "u", ThemeSummary
        )


def test_a_real_httpx_transport_failure_normalizes_to_a_request_error():
    import httpx

    for exc in (httpx.ConnectError("refused"), httpx.RemoteProtocolError("reset")):
        with pytest.raises(ProviderRequestError) as info:
            _client_raising(exc).generate("s", "u", ThemeSummary)
        assert not isinstance(info.value, ProviderTimeoutError)


@pytest.mark.parametrize(
    "code, status", [(401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")]
)
def test_a_real_sdk_authentication_error_is_configuration_not_a_retry(
    code, status, generation_input, rules
):
    sdk_error = _api_error(code, status)
    client = _client_raising(sdk_error)
    with pytest.raises(ProviderAuthenticationError) as info:
        client.generate("s", "u", ThemeSummary)
    assert isinstance(info.value, ProviderConfigurationError)
    assert str(code) in str(info.value)

    # Through the guarded loop: one call, immediately unavailable, no retry.
    client._client.models.calls = 0
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert client._client.models.calls == 1
    assert result.provider_calls == 1
    assert result.status == STATUS_UNAVAILABLE
    assert result.reason == REASON_PROVIDER_UNCONFIGURED
    assert result.attempts[0].outcome == OUTCOME_PROVIDER_UNCONFIGURED


@pytest.mark.parametrize(
    "code, status", [(429, "RESOURCE_EXHAUSTED"), (503, "UNAVAILABLE")]
)
def test_a_real_sdk_transient_error_is_retryable_and_bounded(
    code, status, generation_input, rules
):
    client = _client_raising(_api_error(code, status))
    with pytest.raises(ProviderRequestError) as info:
        client.generate("s", "u", ThemeSummary)
    assert not isinstance(info.value, ProviderConfigurationError)
    client._client.models.calls = 0
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert client._client.models.calls == 2 == result.provider_calls
    assert result.status == STATUS_UNAVAILABLE
    assert result.reason == REASON_PROVIDER_UNAVAILABLE
    assert {a.outcome for a in result.attempts} == {OUTCOME_PROVIDER_ERROR}


def test_a_missing_api_key_is_a_configuration_error_before_any_call(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderConfigurationError) as info:
        GeminiClient(model="fake-model").generate("s", "u", ThemeSummary)
    assert not isinstance(info.value, ProviderAuthenticationError)


def test_provider_error_messages_are_redacted():
    import httpx

    exc = httpx.ConnectError("dial failed for api_key=AIzaSyA-secret-value-1234567890")
    with pytest.raises(ProviderRequestError) as info:
        _client_raising(exc).generate("s", "u", ThemeSummary)
    assert "AIzaSyA-secret-value-1234567890" not in str(info.value)
    assert "[REDACTED]" in str(info.value)


# ----------------------------------------------------------------------
# P2-2: unknown-citation diagnostics never carry model text
# ----------------------------------------------------------------------


CANARY = "api_key=CANARY_SECRET_123"


def test_an_unknown_citation_value_never_appears_in_diagnostics(
    generation_input, rules
):
    poisoned = good(["story:17"])
    poisoned["sentences"][1]["citation_ids"] = [CANARY, "story:17"]
    verdict = validate_candidate(poisoned, generation_input, rules=rules)
    assert verdict.summary is None
    assert verdict.codes == (CODE_UNKNOWN_CITATION,)
    assert verdict.failures[0].detail == "sentence 2: 1 unknown citation"
    assert "CANARY" not in repr(verdict)

    client = ScriptedClient(poisoned, poisoned)
    result = generate_guarded_summary(generation_input, client=client, rules=rules)
    assert result.status == STATUS_UNAVAILABLE
    assert result.reason == REASON_VALIDATION_EXHAUSTED
    serialized = repr(result) + str(result) + repr(dataclasses.asdict(result))
    assert "CANARY" not in serialized
    for attempt in result.attempts:
        assert "CANARY" not in repr(attempt.failures)
        assert "CANARY" not in str(attempt.error)
    retry_prompt = client.calls[1][1]
    assert "CANARY" not in retry_prompt
    assert "Validation feedback" in retry_prompt


def test_several_unknown_citations_are_counted_not_quoted(generation_input, rules):
    candidate = good(["story:17"])
    candidate["sentences"][0]["citation_ids"] = ["story:17", "zz-one", "zz-two"]
    verdict = validate_candidate(candidate, generation_input, rules=rules)
    assert [f.detail for f in verdict.failures] == ["sentence 1: 2 unknown citations"]


# ----------------------------------------------------------------------
# P2-3: frozen inputs are exact -- no mutable model-visible values
# ----------------------------------------------------------------------


def _compose(**overrides):
    fields = dict(
        ticker="TSLA", trading_day="2026-07-13", theme=theme(), evidence=evidence(1, 2)
    )
    fields.update(overrides)
    return SummaryGenerationInput.compose(**fields)


def test_mutable_or_wrong_typed_input_fields_are_rejected():
    with pytest.raises(GuardedSummaryError, match="ticker must be a str"):
        _compose(ticker=["TSLA"])
    with pytest.raises(GuardedSummaryError, match="trading_day must be a str"):
        _compose(trading_day=["2026-07-13"])
    with pytest.raises(GuardedSummaryError, match="theme must be a ThemeReference"):
        _compose(theme={"theme_id": 1})
    with pytest.raises(GuardedSummaryError):
        _compose(evidence=[{"citation_id": "story:1"}])


def test_evidence_must_be_a_tuple_when_built_directly():
    stories = tuple(evidence(1, 2))
    with pytest.raises(GuardedSummaryError, match="evidence must be a tuple"):
        SummaryGenerationInput(
            ticker="TSLA",
            trading_day="2026-07-13",
            theme=theme(),
            evidence=list(stories),
            evidence_ids=frozenset(s.citation_id for s in stories),
            input_fingerprint="x",
        )
    with pytest.raises(GuardedSummaryError, match="evidence_ids"):
        SummaryGenerationInput(
            ticker="TSLA",
            trading_day="2026-07-13",
            theme=theme(),
            evidence=stories,
            evidence_ids={s.citation_id for s in stories},  # a set, not frozenset
            input_fingerprint="x",
        )


def test_evidence_story_fields_are_exact():
    first = evidence(1)[0]
    for field, value in (
        ("title", ["Headline"]),
        ("description", ["d"]),
        ("outlet", ["o"]),
        ("published_at", ["p"]),
        ("citation_id", ["story:1"]),
        ("raw_item_ids", [1, 2]),
        ("urls", ["https://x/"]),
        ("raw_item_ids", (True,)),
        ("raw_item_ids", ("7",)),
        ("urls", (7,)),
        ("persisted_story_id", True),
    ):
        with pytest.raises(GuardedSummaryError):
            dataclasses.replace(first, **{field: value})


def test_theme_reference_fields_are_exact():
    for field, value in (
        ("theme_id", True),
        ("theme_id", "3"),
        ("theme_key", ["k"]),
        ("label", ["l"]),
        ("pipeline_version", ["v"]),
    ):
        with pytest.raises(GuardedSummaryError):
            dataclasses.replace(theme(), **{field: value})
        with pytest.raises(GuardedSummaryError):
            ThemeReference(**{**dataclasses.asdict(theme()), field: value})


def test_replace_cannot_bypass_the_checks_on_a_composed_input():
    base = _compose()
    with pytest.raises(GuardedSummaryError):
        dataclasses.replace(base, ticker=["NVDA"])
    with pytest.raises(GuardedSummaryError):
        dataclasses.replace(base, input_fingerprint=["x"])
    with pytest.raises(GuardedSummaryError):
        dataclasses.replace(base, evidence=list(base.evidence))
    assert dataclasses.replace(base).input_fingerprint == base.input_fingerprint


def test_a_normally_built_input_still_composes():
    built = _compose()
    assert built.ticker == "TSLA" and len(built.evidence) == 2
    assert build_prompt(built).startswith("Ticker: TSLA")


def test_source_values_mutated_after_composition_cannot_reach_prompt_or_fingerprint():
    ids = [1, 2]
    urls = ["https://a/"]
    stories = [
        dataclasses.replace(evidence(1)[0], raw_item_ids=tuple(ids), urls=tuple(urls))
    ]
    built = _compose(evidence=stories)
    before_prompt, before_fp = build_prompt(built), built.input_fingerprint
    ids.append(99)
    urls.append("https://b/")
    stories.append(evidence(2)[0])
    assert build_prompt(built) == before_prompt
    assert built.input_fingerprint == before_fp
    assert len(built.evidence) == 1


# ----------------------------------------------------------------------
# P2-5: at most one regeneration, refused before any provider call
# ----------------------------------------------------------------------


@pytest.mark.parametrize("max_attempts", [1, 2])
def test_allowed_attempt_counts_are_accepted(max_attempts, generation_input, rules):
    client = ScriptedClient(good(["story:17"]))
    result = generate_guarded_summary(
        generation_input, client=client, max_attempts=max_attempts, rules=rules
    )
    assert result.accepted and result.max_attempts == max_attempts
    assert result.policy_fingerprint == compute_policy_fingerprint(
        model="fake-model",
        max_attempts=max_attempts,
        rules=rules,
        max_output_tokens=512,
    )


@pytest.mark.parametrize("max_attempts", [0, 3, -1, 1.5, True, False, "2", None])
def test_disallowed_attempt_counts_are_refused_before_any_call(
    max_attempts, generation_input, rules
):
    client = ScriptedClient(good(["story:17"]))
    with pytest.raises(ValueError):
        generate_guarded_summary(
            generation_input, client=client, max_attempts=max_attempts, rules=rules
        )
    assert client.calls == []

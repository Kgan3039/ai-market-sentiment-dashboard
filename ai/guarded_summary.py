"""A2: guarded summary generation over one frozen evidence set.

One immutable :class:`SummaryGenerationInput` in; one typed
:class:`SummaryGenerationResult` out.  In between: a prompt built from that
input and nothing else, at most :data:`MAX_ATTEMPTS` provider calls, a pure
structural validator run against the very same input, and a bounded
regeneration whose only new material is a list of stable error codes.

**What "accepted" means, exactly.**  An accepted summary is *structurally
grounded* and *policy-valid*: it parsed, it has 2-4 non-blank sentences, a
1-8 word label, every sentence cites at least one id, every id names a
story in the frozen input, and the generated copy trips none of the Phase
0 banned-language rules.  Citation ids are ``story:<persisted story id>``,
so a citation resolves to one durable canonical story.

**What it does not mean.**  Nothing here establishes that a sentence is
*entailed* by the story it cites.  That is semantic faithfulness (gate G2),
and it is measured by human review later, not by this module.

**What this module never does.**  It never reads or writes the database,
never persists a result, never caches, never registers as a stage, and
never fabricates a fallback summary.  A generation that cannot be accepted
is reported as ``unavailable`` with the attempts that were made.
Persistence, caching, invalidation, cost accounting and serving belong to
A3 and B1.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, ValidationError

from .summarization import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_TEMPERATURE,
    MAX_LABEL_WORDS,
    MAX_SENTENCES,
    MIN_SENTENCES,
    SYSTEM_PROMPT,
    GenerationUsage,
    MalformedOutputError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderTimeoutError,
    Sentence,
    ThemeSummary,
)

# ----------------------------------------------------------------------
# Policy constants.  Every one of these is folded into the policy
# fingerprint, so changing any of them changes the fingerprint.
# ----------------------------------------------------------------------

#: The citation identity convention this module emits and validates.
CITATION_CONVENTION = "persisted_story_id.v1"
CITATION_PREFIX = "story:"

#: Attempt 1 is the plain prompt; attempt 2 is the one bounded regeneration
#: with validation feedback.  There is no third.
MAX_ATTEMPTS = 2
#: The only attempt counts a caller may choose: no regeneration, or one.
ALLOWED_MAX_ATTEMPTS = frozenset({1, MAX_ATTEMPTS})

PROMPT_VERSION = "a2.guarded.v1"

#: The A1 rules, plus the two things A1 could not say because it did not
#: know its ids would be namespaced or that a second attempt existed.
SYSTEM_PROMPT_A2 = (
    SYSTEM_PROMPT
    + """11. Citation ids look like "story:17". Copy them exactly as given in the
    evidence; never shorten, renumber, or invent one.
12. If validation feedback is provided, it describes what was wrong with a
    previous attempt. Fix those problems using the same evidence only.
"""
)

#: Result statuses.
STATUS_ACCEPTED = "accepted"
STATUS_UNAVAILABLE = "unavailable"

#: Per-attempt outcomes.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"  # answered, failed structural/policy validation
OUTCOME_PROVIDER_ERROR = "provider_error"
OUTCOME_PROVIDER_TIMEOUT = "provider_timeout"
OUTCOME_PROVIDER_UNCONFIGURED = "provider_unconfigured"

#: Why a result is unavailable.
REASON_VALIDATION_EXHAUSTED = "validation_exhausted"
REASON_PROVIDER_UNAVAILABLE = "provider_unavailable"
REASON_PROVIDER_UNCONFIGURED = "provider_unconfigured"

#: Stable validation codes, in the order the validator reports them.
CODE_MALFORMED_OUTPUT = "malformed_output"
CODE_INVALID_LABEL = "invalid_label"
CODE_INVALID_SENTENCE_COUNT = "invalid_sentence_count"
CODE_BLANK_SENTENCE = "blank_sentence"
CODE_MISSING_CITATION = "missing_citation"
CODE_UNKNOWN_CITATION = "unknown_citation"
CODE_BANNED_LANGUAGE = "banned_language"

VALIDATION_CODES: tuple[str, ...] = (
    CODE_MALFORMED_OUTPUT,
    CODE_INVALID_LABEL,
    CODE_INVALID_SENTENCE_COUNT,
    CODE_BLANK_SENTENCE,
    CODE_MISSING_CITATION,
    CODE_UNKNOWN_CITATION,
    CODE_BANNED_LANGUAGE,
)

#: What the model is told about each code on the second attempt.  Fixed
#: text, keyed by code: the feedback block is assembled from these and
#: from nothing the model or the provider said.
FEEDBACK_MESSAGES: Mapping[str, str] = {
    CODE_MALFORMED_OUTPUT: "the output was not valid JSON matching the schema",
    CODE_INVALID_LABEL: f"the label must be 1 to {MAX_LABEL_WORDS} words",
    CODE_INVALID_SENTENCE_COUNT: (
        f"there must be between {MIN_SENTENCES} and {MAX_SENTENCES} sentences"
    ),
    CODE_BLANK_SENTENCE: "every sentence must contain text",
    CODE_MISSING_CITATION: "every sentence must cite at least one story id",
    CODE_UNKNOWN_CITATION: (
        "a citation id was not one of the supplied story ids; use only the ids "
        "listed in the evidence, exactly as written"
    ),
    CODE_BANNED_LANGUAGE: (
        "the text used advisory, predictive, causal, or model-confidence "
        "framing; describe what the coverage says instead"
    ),
}

#: The content scopes the copy rules apply to.  Evidence text is never
#: linted: it is publisher copy, and the rules say so themselves.
LABEL_SCOPE = "generated_label"
SENTENCE_SCOPE = "generated_summary"

Rules = Sequence[tuple[str, str, str]]


class GuardedSummaryError(ValueError):
    """A frozen input could not be built, or a caller broke the contract."""


# ----------------------------------------------------------------------
# The frozen input
# ----------------------------------------------------------------------


def citation_id_for(persisted_story_id: int) -> str:
    """The citation id of one persisted story: ``story:<id>``."""

    identifier = int(persisted_story_id)
    if identifier <= 0:
        raise GuardedSummaryError("a persisted story id is a positive integer")
    return f"{CITATION_PREFIX}{identifier}"


@dataclass(frozen=True)
class EvidenceStory:
    """One canonical story the model may summarize and cite.

    ``title``, ``description``, ``outlet`` and ``published_at`` are what
    the model sees.  ``raw_item_ids`` and ``urls`` are provenance carried
    beside the evidence so a later resolver can turn a ``story:<id>``
    citation into concrete raw items; they are not shown to the model.
    """

    citation_id: str
    persisted_story_id: int
    title: str
    description: str
    outlet: str
    published_at: str
    raw_item_ids: tuple[int, ...] = ()
    urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_str(self.citation_id, "citation_id")
        _require_int(self.persisted_story_id, "persisted_story_id")
        if self.citation_id != citation_id_for(self.persisted_story_id):
            raise GuardedSummaryError(
                f"citation id {self.citation_id!r} does not name persisted story "
                f"{self.persisted_story_id}"
            )
        for name in ("title", "description", "outlet", "published_at"):
            _require_str(getattr(self, name), f"{self.citation_id}.{name}")
        if not self.title.strip():
            raise GuardedSummaryError(f"{self.citation_id} has no title")
        _require_tuple_of(self.raw_item_ids, "raw_item_ids", _require_int)
        _require_tuple_of(self.urls, "urls", _require_str)

    def model_visible(self) -> dict[str, str]:
        """Exactly the fields the prompt renders, in a fixed key order."""

        return {
            "citation_id": self.citation_id,
            "title": self.title,
            "description": self.description,
            "outlet": self.outlet,
            "published_at": self.published_at,
        }


def _require_str(value: Any, field: str) -> None:
    if not isinstance(value, str):
        raise GuardedSummaryError(f"{field} must be a str, not {type(value).__name__}")


def _require_int(value: Any, field: str) -> None:
    # bool is an int in Python; an identity of ``True`` is a defect.
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuardedSummaryError(f"{field} must be an int, not {type(value).__name__}")


def _require_tuple_of(
    value: Any, field: str, check: Callable[[Any, str], None]
) -> None:
    if not isinstance(value, tuple):
        raise GuardedSummaryError(
            f"{field} must be a tuple, not {type(value).__name__}"
        )
    for position, item in enumerate(value):
        check(item, f"{field}[{position}]")


@dataclass(frozen=True)
class ThemeReference:
    """Which persisted theme this input was built for.

    Frozen *and* exact: every field is checked for its declared immutable
    type on construction, and ``dataclasses.replace`` re-runs the check,
    so an instance cannot carry a list where a str belongs and then be
    changed from outside.
    """

    theme_id: int
    theme_key: str
    label: str
    pipeline_version: str

    def __post_init__(self) -> None:
        _require_int(self.theme_id, "theme_id")
        _require_str(self.theme_key, "theme_key")
        _require_str(self.label, "label")
        _require_str(self.pipeline_version, "pipeline_version")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_input_fingerprint(
    ticker: str,
    trading_day: str,
    theme: ThemeReference,
    evidence: Sequence[EvidenceStory],
) -> str:
    """SHA-256 over everything the model sees plus the input's identity.

    Model-visible: ticker, trading day, and each evidence story's rendered
    fields in order.  Identity: the persisted theme and pipeline version,
    and the citation convention.  Provenance (raw item ids, urls) is *not*
    in here: it is not shown to the model and changing it does not change
    what was summarized.  Not a cache key -- A3 decides what a cache key
    is -- just the identity of what this result was generated from.
    """

    payload = {
        "citation_convention": CITATION_CONVENTION,
        "ticker": ticker,
        "trading_day": trading_day,
        "theme": {
            "theme_id": theme.theme_id,
            "theme_key": theme.theme_key,
            "pipeline_version": theme.pipeline_version,
        },
        "evidence": [story.model_visible() for story in evidence],
    }
    return _sha256(_canonical_json(payload))


@dataclass(frozen=True)
class SummaryGenerationInput:
    """The whole permitted universe for one generation, frozen.

    Built through :meth:`compose`; ``__post_init__`` recomputes the
    derived fields and refuses an instance whose ``evidence_ids`` or
    ``input_fingerprint`` disagree with its evidence, so a hand-assembled
    input cannot claim a fingerprint it does not have.
    """

    ticker: str
    trading_day: str
    theme: ThemeReference
    evidence: tuple[EvidenceStory, ...]
    evidence_ids: frozenset[str]
    input_fingerprint: str

    @classmethod
    def compose(
        cls,
        *,
        ticker: str,
        trading_day: str,
        theme: ThemeReference,
        evidence: Sequence[EvidenceStory],
    ) -> "SummaryGenerationInput":
        if not isinstance(theme, ThemeReference):
            raise GuardedSummaryError("theme must be a ThemeReference")
        stories = tuple(evidence)
        if not all(isinstance(story, EvidenceStory) for story in stories):
            raise GuardedSummaryError("evidence must be EvidenceStory records")
        return cls(
            ticker=ticker,
            trading_day=trading_day,
            theme=theme,
            evidence=stories,
            evidence_ids=frozenset(story.citation_id for story in stories),
            input_fingerprint=compute_input_fingerprint(
                ticker, trading_day, theme, stories
            ),
        )

    def __post_init__(self) -> None:
        _require_str(self.ticker, "ticker")
        _require_str(self.trading_day, "trading_day")
        if not isinstance(self.theme, ThemeReference):
            raise GuardedSummaryError("theme must be a ThemeReference")
        if not isinstance(self.evidence, tuple):
            raise GuardedSummaryError(
                f"evidence must be a tuple, not {type(self.evidence).__name__}"
            )
        if not self.evidence:
            raise GuardedSummaryError("a generation input needs at least one story")
        if not all(isinstance(story, EvidenceStory) for story in self.evidence):
            raise GuardedSummaryError("evidence must be EvidenceStory records")
        _require_str(self.input_fingerprint, "input_fingerprint")
        ids = [story.citation_id for story in self.evidence]
        if len(ids) != len(set(ids)):
            raise GuardedSummaryError("evidence carries a duplicate citation id")
        if not isinstance(self.evidence_ids, frozenset) or self.evidence_ids != set(
            ids
        ):
            raise GuardedSummaryError("evidence_ids does not match the evidence")
        expected = compute_input_fingerprint(
            self.ticker, self.trading_day, self.theme, self.evidence
        )
        if self.input_fingerprint != expected:
            raise GuardedSummaryError("input_fingerprint does not match the evidence")


# ----------------------------------------------------------------------
# The prompt
# ----------------------------------------------------------------------


def build_prompt(
    generation_input: SummaryGenerationInput,
    feedback_codes: Sequence[str] = (),
) -> str:
    """Render the frozen input, plus an optional feedback block.

    The evidence section is a pure function of the input and is
    byte-identical between attempts.  The feedback block, when present,
    is assembled from :data:`FEEDBACK_MESSAGES` by code: nothing the model
    produced, and nothing the provider said, is echoed back.
    """

    lines = [
        f"Ticker: {generation_input.ticker}",
        f"Trading day: {generation_input.trading_day}",
        "Evidence stories (cite by id, exactly as written):",
    ]
    for story in generation_input.evidence:
        lines.append(
            f"- id: {story.citation_id}\n"
            f"  title: {story.title}\n"
            f"  description: {story.description}\n"
            f"  outlet: {story.outlet}\n"
            f"  time: {story.published_at}"
        )
    if feedback_codes:
        lines.append("")
        lines.append(
            "Validation feedback on the previous attempt (regenerate from the "
            "same evidence above and fix every point):"
        )
        for code in _ordered_codes(feedback_codes):
            lines.append(f"- {code}: {FEEDBACK_MESSAGES[code]}")
    return "\n".join(lines)


def _ordered_codes(codes: Sequence[str]) -> tuple[str, ...]:
    """Distinct codes, in the validator's canonical order."""

    unknown = sorted(set(codes) - set(VALIDATION_CODES))
    if unknown:
        raise GuardedSummaryError(f"unknown validation code(s): {unknown}")
    present = set(codes)
    return tuple(code for code in VALIDATION_CODES if code in present)


# ----------------------------------------------------------------------
# What the provider is asked for
# ----------------------------------------------------------------------


class CandidateSentence(BaseModel):
    """One sentence as the provider returns it: shape only, no bounds."""

    text: str
    citation_ids: list[str]


class CandidateSummary(BaseModel):
    """The provider's response schema.

    Deliberately looser than :class:`~ai.summarization.ThemeSummary`: it
    pins the JSON shape (so structured output is used) but carries no
    length bounds, so an out-of-range answer reaches the validator and is
    reported under its own code rather than as a generic parse failure.
    """

    label: str
    sentences: list[CandidateSentence]


# ----------------------------------------------------------------------
# Validation: pure, and only ever against the frozen input
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationFailure:
    """One reason an attempt was refused.

    ``detail`` is safe text composed by this module -- positions,
    counts, rule categories -- never a quote of the candidate.
    """

    code: str
    detail: str


@dataclass(frozen=True)
class ValidationVerdict:
    summary: Optional[ThemeSummary]
    failures: tuple[ValidationFailure, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.summary is not None and not self.failures

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(failure.code for failure in self.failures)


def load_copy_rules() -> tuple[tuple[str, str, str], ...]:
    """The Phase 0 banned-language rules, from their one source of truth."""

    from tools.validate_phase0_copy_rules import load_rules

    return tuple(load_rules())


def _detected_categories(text: str, rules: Rules, *, content_scope: str) -> list[str]:
    from tools.validate_phase0_copy_rules import detected_categories

    return sorted(detected_categories(text, list(rules), content_scope=content_scope))


def _coerce_candidate(candidate: Any) -> Optional[dict[str, Any]]:
    """Turn whatever the client returned into a plain mapping, or ``None``."""

    if isinstance(candidate, BaseModel):
        return candidate.model_dump()
    if isinstance(candidate, str):
        try:
            decoded = json.loads(candidate)
        except ValueError:
            return None
        return decoded if isinstance(decoded, dict) else None
    if isinstance(candidate, Mapping):
        return dict(candidate)
    return None


def validate_candidate(
    candidate: Any,
    generation_input: SummaryGenerationInput,
    *,
    rules: Optional[Rules] = None,
) -> ValidationVerdict:
    """Judge one candidate against the frozen input; make no provider call.

    Reports *every* failure it finds, in canonical code order and then
    sentence order, so the feedback block is complete after one pass.
    A candidate is accepted only when there are none.  Duplicate valid
    citations inside one sentence are normalized (first occurrence kept,
    order preserved) and are not a failure.
    """

    failures: list[ValidationFailure] = []
    payload = _coerce_candidate(candidate)
    if payload is None:
        return ValidationVerdict(
            None, (ValidationFailure(CODE_MALFORMED_OUTPUT, "not an object"),)
        )

    label = payload.get("label")
    sentences = payload.get("sentences")
    if not isinstance(label, str) or not isinstance(sentences, list):
        return ValidationVerdict(
            None,
            (ValidationFailure(CODE_MALFORMED_OUTPUT, "label or sentences missing"),),
        )
    normalized: list[tuple[str, list[str]]] = []
    for index, entry in enumerate(sentences, start=1):
        if not isinstance(entry, Mapping):
            return ValidationVerdict(
                None,
                (
                    ValidationFailure(
                        CODE_MALFORMED_OUTPUT, f"sentence {index} not an object"
                    ),
                ),
            )
        text = entry.get("text")
        ids = entry.get("citation_ids")
        if (
            not isinstance(text, str)
            or not isinstance(ids, list)
            or not all(isinstance(value, str) for value in ids)
        ):
            return ValidationVerdict(
                None,
                (
                    ValidationFailure(
                        CODE_MALFORMED_OUTPUT, f"sentence {index} malformed"
                    ),
                ),
            )
        normalized.append((text, list(ids)))

    # -- label ------------------------------------------------------------
    word_count = len(label.split())
    if word_count == 0 or word_count > MAX_LABEL_WORDS:
        failures.append(
            ValidationFailure(CODE_INVALID_LABEL, f"label has {word_count} words")
        )

    # -- sentence count ---------------------------------------------------
    count = len(normalized)
    if count < MIN_SENTENCES or count > MAX_SENTENCES:
        failures.append(
            ValidationFailure(CODE_INVALID_SENTENCE_COUNT, f"{count} sentences")
        )

    # -- per-sentence: blank, missing, unknown ---------------------------
    deduped: list[tuple[str, list[str]]] = []
    blank: list[int] = []
    missing: list[int] = []
    unknown: list[tuple[int, str]] = []
    for index, (text, ids) in enumerate(normalized, start=1):
        if not text.strip():
            blank.append(index)
        seen: list[str] = []
        for value in ids:
            if value not in seen:
                seen.append(value)
        if not seen:
            missing.append(index)
        for value in seen:
            if value not in generation_input.evidence_ids:
                unknown.append((index, value))
        deduped.append((text, seen))
    for index in blank:
        failures.append(ValidationFailure(CODE_BLANK_SENTENCE, f"sentence {index}"))
    for index in missing:
        failures.append(ValidationFailure(CODE_MISSING_CITATION, f"sentence {index}"))
    unknown_per_sentence: dict[int, int] = {}
    for index, _value in unknown:
        unknown_per_sentence[index] = unknown_per_sentence.get(index, 0) + 1
    for index in sorted(unknown_per_sentence):
        # Structural metadata only.  The offending value is model output --
        # arbitrary text that could be anything, a credential included --
        # and it is never echoed into a diagnostic.
        count = unknown_per_sentence[index]
        failures.append(
            ValidationFailure(
                CODE_UNKNOWN_CITATION,
                f"sentence {index}: {count} unknown citation"
                f"{'' if count == 1 else 's'}",
            )
        )

    # -- generated-copy policy -------------------------------------------
    active_rules = load_copy_rules() if rules is None else rules
    for category in _detected_categories(
        label, active_rules, content_scope=LABEL_SCOPE
    ):
        failures.append(ValidationFailure(CODE_BANNED_LANGUAGE, f"label: {category}"))
    for index, (text, _) in enumerate(deduped, start=1):
        for category in _detected_categories(
            text, active_rules, content_scope=SENTENCE_SCOPE
        ):
            failures.append(
                ValidationFailure(CODE_BANNED_LANGUAGE, f"sentence {index}: {category}")
            )

    if failures:
        return ValidationVerdict(None, tuple(failures))
    summary = ThemeSummary(
        label=label,
        sentences=[Sentence(text=text, citation_ids=ids) for text, ids in deduped],
    )
    return ValidationVerdict(summary, ())


# ----------------------------------------------------------------------
# Policy fingerprint
# ----------------------------------------------------------------------


def rules_digest(rules: Rules) -> str:
    """SHA-256 of the banned-language rules as loaded."""

    return _sha256(_canonical_json([list(rule) for rule in rules]))


def compute_policy_fingerprint(
    *,
    model: str,
    max_attempts: int,
    rules: Rules,
    temperature: float = DEFAULT_TEMPERATURE,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> str:
    """SHA-256 over everything that decides what this module would accept.

    Prompt text and version, both output contracts, the model and its
    generation settings, the attempt bound, the copy rules, the feedback
    wording and the citation convention.  Read-only provenance for the
    result; A3 decides what to do with it.
    """

    payload = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT_A2,
        "candidate_schema": CandidateSummary.model_json_schema(),
        "summary_schema": ThemeSummary.model_json_schema(),
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "max_attempts": max_attempts,
        "rules_digest": rules_digest(rules),
        "feedback_messages": dict(FEEDBACK_MESSAGES),
        "citation_convention": CITATION_CONVENTION,
        "sentence_bounds": [MIN_SENTENCES, MAX_SENTENCES],
        "label_max_words": MAX_LABEL_WORDS,
    }
    return _sha256(_canonical_json(payload))


def _require_allowed_attempts(max_attempts: Any) -> None:
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts not in ALLOWED_MAX_ATTEMPTS
    ):
        raise ValueError(
            f"max_attempts must be one of {sorted(ALLOWED_MAX_ATTEMPTS)} (at most "
            f"one regeneration), not {max_attempts!r}"
        )


@dataclass(frozen=True)
class GenerationPolicy:
    """Everything that decides what this module would accept, resolved once.

    The policy is *resolved* from a client and a rule set by
    :func:`resolve_generation_policy` and then carried, unchanged, through
    every step that has to agree on it: a cache lookup keyed by
    ``fingerprint``, the generation itself, and the write that records
    which policy produced a result.  The rules travel inside it so the
    validator runs against the rules the fingerprint was computed over --
    never against a file that may have been reloaded in between.

    Frozen and exact, like :class:`SummaryGenerationInput`:
    ``__post_init__`` recomputes ``fingerprint`` from the other fields and
    refuses an instance whose fingerprint disagrees with them, so a
    hand-assembled policy cannot claim a fingerprint it does not have.
    """

    model: str
    max_attempts: int
    rules: tuple[tuple[str, str, str], ...]
    temperature: float
    max_output_tokens: int
    fingerprint: str

    def __post_init__(self) -> None:
        _require_str(self.model, "model")
        _require_allowed_attempts(self.max_attempts)
        if not isinstance(self.rules, tuple) or not all(
            isinstance(rule, tuple) and len(rule) == 3 for rule in self.rules
        ):
            raise GuardedSummaryError(
                "rules must be a tuple of (category, kind, pattern)"
            )
        for rule in self.rules:
            for value in rule:
                _require_str(value, "rule")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise GuardedSummaryError("temperature must be a number")
        _require_int(self.max_output_tokens, "max_output_tokens")
        if self.max_output_tokens <= 0:
            raise GuardedSummaryError("max_output_tokens must be positive")
        _require_str(self.fingerprint, "fingerprint")
        expected = compute_policy_fingerprint(
            model=self.model,
            max_attempts=self.max_attempts,
            rules=self.rules,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        if self.fingerprint != expected:
            raise GuardedSummaryError("policy fingerprint does not match the policy")


def resolve_generation_policy(
    client: Any,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    rules: Optional[Rules] = None,
) -> GenerationPolicy:
    """The effective policy one generation over ``client`` would run under.

    Exactly what :func:`generate_guarded_summary` resolves for itself when
    it is given no policy: the client's model and output cap, the attempt
    bound, and the copy rules -- loaded from their source of truth when
    ``rules`` is ``None``, and snapshotted here so the same rules reach
    the validator.  A caller that must agree with a generation about its
    policy (a cache lookup made *before* the call) resolves once and hands
    the same object to both.
    """

    _require_allowed_attempts(max_attempts)
    active_rules = tuple(
        tuple(rule) for rule in (load_copy_rules() if rules is None else rules)
    )
    model = str(getattr(client, "model", "unspecified"))
    max_output_tokens = int(
        getattr(client, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
    )
    return GenerationPolicy(
        model=model,
        max_attempts=max_attempts,
        rules=active_rules,
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=max_output_tokens,
        fingerprint=compute_policy_fingerprint(
            model=model,
            max_attempts=max_attempts,
            rules=active_rules,
            temperature=DEFAULT_TEMPERATURE,
            max_output_tokens=max_output_tokens,
        ),
    )


# ----------------------------------------------------------------------
# The typed result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptRecord:
    """What one provider call did.  Observed, never persisted here."""

    attempt: int
    outcome: str
    validation_codes: tuple[str, ...] = ()
    failures: tuple[ValidationFailure, ...] = ()
    latency_ms: Optional[float] = None
    usage: Optional[GenerationUsage] = None
    #: Redacted provider error text for provider outcomes; ``None`` otherwise.
    error: Optional[str] = None


@dataclass(frozen=True)
class SummaryGenerationResult:
    """The outcome of one guarded generation over one frozen input."""

    status: str
    summary: Optional[ThemeSummary]
    reason: Optional[str]
    attempts: tuple[AttemptRecord, ...]
    input_fingerprint: str
    policy_fingerprint: str
    theme: ThemeReference
    max_attempts: int = MAX_ATTEMPTS
    accepted_attempt: Optional[int] = None
    #: Plain statement of what acceptance claims, carried on the value so
    #: no consumer can mistake it for a faithfulness verdict.
    guarantee: str = field(
        default=(
            "structural_grounding_and_copy_policy_only; "
            "semantic_faithfulness_not_established"
        )
    )

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    @property
    def provider_calls(self) -> int:
        return len(self.attempts)


# ----------------------------------------------------------------------
# The guarded loop
# ----------------------------------------------------------------------


def generate_guarded_summary(
    generation_input: SummaryGenerationInput,
    *,
    client: Any,
    max_attempts: int = MAX_ATTEMPTS,
    rules: Optional[Rules] = None,
    clock: Callable[[], float] = time.perf_counter,
    policy: Optional[GenerationPolicy] = None,
) -> SummaryGenerationResult:
    """Generate, validate, regenerate once with feedback, or give up -- typed.

    ``client`` is anything with ``generate(system_prompt, user_prompt,
    response_schema)``; it is called exactly once per attempt and never
    more than ``max_attempts`` times.  The evidence half of the prompt is
    identical on every attempt.

    ``policy`` is an already-resolved :class:`GenerationPolicy`, for a
    caller that looked something up under a fingerprint before calling
    and needs this generation to run under exactly that policy.  When it
    is given, ``rules`` must be left unset and ``max_attempts`` must equal
    the policy's -- the policy already fixes both -- and the client must
    still be the one the policy was resolved from.  Without it the policy
    is resolved here, from the same helper, with the same result.

    Handled and recorded: a validation rejection or malformed answer
    (retryable with feedback), a provider request failure or timeout
    (retryable without feedback -- the model never saw anything), and a
    configuration failure (not retryable; a second call would fail the
    same way).  Everything else -- a ``TypeError`` from a client that does
    not honour the call shape, say -- propagates, because it is a defect
    and not a provider outcome.
    """

    if not isinstance(generation_input, SummaryGenerationInput):
        raise GuardedSummaryError(
            "generate_guarded_summary needs a SummaryGenerationInput"
        )
    _require_allowed_attempts(max_attempts)
    if policy is None:
        policy = resolve_generation_policy(
            client, max_attempts=max_attempts, rules=rules
        )
    else:
        if not isinstance(policy, GenerationPolicy):
            raise GuardedSummaryError("policy must be a GenerationPolicy")
        if rules is not None or max_attempts != policy.max_attempts:
            raise GuardedSummaryError(
                "a resolved policy already fixes the rules and max_attempts; "
                "pass max_attempts=policy.max_attempts and no rules"
            )
        observed = resolve_generation_policy(
            client, max_attempts=policy.max_attempts, rules=policy.rules
        )
        if observed.fingerprint != policy.fingerprint:
            raise GuardedSummaryError(
                "the client no longer matches the resolved policy; resolve the "
                "policy again from the client that will generate"
            )
    active_rules = policy.rules
    policy_fingerprint = policy.fingerprint

    def finish(
        status: str,
        summary: Optional[ThemeSummary],
        reason: Optional[str],
        attempts: Sequence[AttemptRecord],
        accepted_attempt: Optional[int] = None,
    ) -> SummaryGenerationResult:
        return SummaryGenerationResult(
            status=status,
            summary=summary,
            reason=reason,
            attempts=tuple(attempts),
            input_fingerprint=generation_input.input_fingerprint,
            policy_fingerprint=policy_fingerprint,
            theme=generation_input.theme,
            max_attempts=max_attempts,
            accepted_attempt=accepted_attempt,
        )

    attempts: list[AttemptRecord] = []
    feedback: tuple[str, ...] = ()
    for number in range(1, max_attempts + 1):
        prompt = build_prompt(generation_input, feedback)
        started = clock()
        try:
            candidate = client.generate(SYSTEM_PROMPT_A2, prompt, CandidateSummary)
        except ProviderConfigurationError as exc:
            attempts.append(
                AttemptRecord(
                    attempt=number,
                    outcome=OUTCOME_PROVIDER_UNCONFIGURED,
                    latency_ms=_elapsed_ms(clock, started),
                    error=str(exc),
                )
            )
            return finish(
                STATUS_UNAVAILABLE, None, REASON_PROVIDER_UNCONFIGURED, attempts
            )
        except ProviderTimeoutError as exc:
            attempts.append(
                AttemptRecord(
                    attempt=number,
                    outcome=OUTCOME_PROVIDER_TIMEOUT,
                    latency_ms=_elapsed_ms(clock, started),
                    error=str(exc),
                )
            )
            continue
        except ProviderRequestError as exc:
            attempts.append(
                AttemptRecord(
                    attempt=number,
                    outcome=OUTCOME_PROVIDER_ERROR,
                    latency_ms=_elapsed_ms(clock, started),
                    error=str(exc),
                )
            )
            continue
        except (MalformedOutputError, ValidationError):
            # The provider answered with something that is not the schema.
            # Recorded as a validation rejection so the second attempt can
            # be told what was wrong; the raw error is not carried.
            verdict = ValidationVerdict(
                None, (ValidationFailure(CODE_MALFORMED_OUTPUT, "unparseable"),)
            )
            latency = _elapsed_ms(clock, started)
            usage = getattr(client, "last_usage", None)
        else:
            latency = _elapsed_ms(clock, started)
            usage = getattr(client, "last_usage", None)
            verdict = validate_candidate(
                candidate, generation_input, rules=active_rules
            )

        if verdict.accepted:
            attempts.append(
                AttemptRecord(
                    attempt=number,
                    outcome=OUTCOME_ACCEPTED,
                    latency_ms=latency,
                    usage=usage,
                )
            )
            return finish(STATUS_ACCEPTED, verdict.summary, None, attempts, number)
        attempts.append(
            AttemptRecord(
                attempt=number,
                outcome=OUTCOME_REJECTED,
                validation_codes=verdict.codes,
                failures=verdict.failures,
                latency_ms=latency,
                usage=usage,
            )
        )
        feedback = _ordered_codes(verdict.codes)

    last = attempts[-1]
    if last.outcome in (OUTCOME_PROVIDER_ERROR, OUTCOME_PROVIDER_TIMEOUT):
        reason = REASON_PROVIDER_UNAVAILABLE
    else:
        reason = REASON_VALIDATION_EXHAUSTED
    return finish(STATUS_UNAVAILABLE, None, reason, attempts)


def _elapsed_ms(clock: Callable[[], float], started: float) -> float:
    return max(0.0, (clock() - started) * 1000.0)


__all__ = [
    "AttemptRecord",
    "CITATION_CONVENTION",
    "CITATION_PREFIX",
    "CandidateSentence",
    "CandidateSummary",
    "CODE_BANNED_LANGUAGE",
    "CODE_BLANK_SENTENCE",
    "CODE_INVALID_LABEL",
    "CODE_INVALID_SENTENCE_COUNT",
    "CODE_MALFORMED_OUTPUT",
    "CODE_MISSING_CITATION",
    "CODE_UNKNOWN_CITATION",
    "EvidenceStory",
    "FEEDBACK_MESSAGES",
    "GenerationPolicy",
    "GuardedSummaryError",
    "ALLOWED_MAX_ATTEMPTS",
    "MAX_ATTEMPTS",
    "OUTCOME_ACCEPTED",
    "OUTCOME_PROVIDER_ERROR",
    "OUTCOME_PROVIDER_TIMEOUT",
    "OUTCOME_PROVIDER_UNCONFIGURED",
    "OUTCOME_REJECTED",
    "PROMPT_VERSION",
    "REASON_PROVIDER_UNAVAILABLE",
    "REASON_PROVIDER_UNCONFIGURED",
    "REASON_VALIDATION_EXHAUSTED",
    "STATUS_ACCEPTED",
    "STATUS_UNAVAILABLE",
    "SYSTEM_PROMPT_A2",
    "SummaryGenerationInput",
    "SummaryGenerationResult",
    "ThemeReference",
    "VALIDATION_CODES",
    "ValidationFailure",
    "ValidationVerdict",
    "build_prompt",
    "citation_id_for",
    "compute_input_fingerprint",
    "compute_policy_fingerprint",
    "generate_guarded_summary",
    "load_copy_rules",
    "resolve_generation_policy",
    "rules_digest",
    "validate_candidate",
]

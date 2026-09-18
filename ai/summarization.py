"""Cited theme summarization module.

Author: Abhi
Responsibility: Turn a theme's member stories into a strict-JSON label plus a
2-4 sentence summary where every sentence carries citations to the member
stories it draws from.

Dataset Format Contract:
- Input: member stories (id, title, description, outlet, published_at) for
  one theme, matching the story fields described in
  docs/PHASE_0_SPEC.md Section 3 (I1's raw_items/stories tables) and the
  citation shape B1's read API resolves them to.
- Output: {label: <=8 words, sentences: [{text, citation_ids: [...]}]}
- Fixture-first (see ai/fixtures/theme_fixtures.json): this module has no
  dependency on I1's real persistence layer or M5's clustering.
  `summarize()` only retries on structurally malformed provider output.

The guarded production path is :mod:`ai.guarded_summary` (A2): it reuses
the prompt, schema and client defined here, calls the client once per
bounded attempt, validates against a frozen input, and returns a typed
accepted/unavailable result instead of raising.  `summarize()` stays as the
fixture-first A1 entry point and is not called by that path.

Provider: Gemini (google-genai), low temperature, structured JSON output via
response_schema so parsing failures are rare by construction.  Provider
failures are normalized to the project-owned exception types below;
messages are redacted so an API key never travels in an error string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from phase0.redaction import redact_text

MAX_LABEL_WORDS = 8
MIN_SENTENCES = 2
MAX_SENTENCES = 4
MAX_RETRIES = 1  # one retry on structurally malformed provider output

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.1
#: A 2-4 sentence cited summary is a few hundred tokens; the cap exists so a
#: runaway response is cut off rather than paid for.
DEFAULT_MAX_OUTPUT_TOKENS = 1024
#: Wall-clock bound on one provider request.  Without it a hung connection
#: blocks whatever stage is generating for as long as the socket lives.
DEFAULT_TIMEOUT_MS = 30_000

SYSTEM_PROMPT = (
    "You are generating a short, neutral summary of news coverage for a stock "
    "ticker theme.\n"
    "\n"
    "Rules (must all be followed exactly):\n"
    "1. Output strict JSON matching the provided schema. No prose outside the JSON.\n"
    '2. "label" is a short theme title of at most 8 words. No punctuation-only '
    "labels.\n"
    '3. "sentences" has between 2 and 4 entries.\n'
    '4. Every sentence\'s "citation_ids" must contain at least one id, and '
    "every id must be\n"
    "   one of the story ids given in the input. Never invent an id.\n"
    "5. Every factual claim must be attributed to at least one cited story. Do "
    "not state\n"
    "   anything that is not supported by the cited stories.\n"
    '6. Never give financial advice or a trading recommendation (e.g. "buy", '
    '"sell", "hold").\n'
    '7. Never predict future price or stock movement (e.g. "will rise", '
    '"expected to fall").\n'
    '8. Never claim a causal explanation for a price move (e.g. "fell because",\n'
    '   "this move was driven by", "explains today\'s decline").\n'
    "9. Describe what the coverage says, not what will happen next. Use "
    "framing like\n"
    '   "coverage today is dominated by..." or "the most-covered storyline is...",\n'
    "   never a causal or predictive framing, even if the source headlines use one.\n"
    "10. Do not include any commentary, caveats, or meta-text about these "
    "rules in the output.\n"
)


@dataclass
class MemberStory:
    """One story feeding a theme; mirrors I1's raw_items/stories fields."""

    id: str
    title: str
    description: str
    outlet: str
    published_at: str


@dataclass
class ThemeInput:
    """A theme's member stories, the unit `summarize()` operates on."""

    ticker: str
    member_stories: list[MemberStory]
    trading_day: Optional[str] = None


class Sentence(BaseModel):
    text: str
    citation_ids: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("sentence text must not be blank")
        return value


class ThemeSummary(BaseModel):
    label: str
    sentences: list[Sentence] = Field(
        min_length=MIN_SENTENCES, max_length=MAX_SENTENCES
    )

    @field_validator("label")
    @classmethod
    def _label_word_count(cls, value: str) -> str:
        word_count = len(value.split())
        if word_count == 0 or word_count > MAX_LABEL_WORDS:
            raise ValueError(
                f"label must be 1-{MAX_LABEL_WORDS} words, got {word_count}"
            )
        return value


class SummarizationError(RuntimeError):
    """Raised when the provider fails to produce a schema-valid summary after retrying.

    Every message is redacted on construction so a credential that reached
    a provider error string cannot escape through the exception either.
    """

    def __init__(self, message: str = "", *args: object) -> None:
        super().__init__(redact_text(str(message)), *args)


class ProviderConfigurationError(SummarizationError):
    """The provider cannot be called at all (no API key, no client)."""


class ProviderAuthenticationError(ProviderConfigurationError):
    """The provider refused the credential (HTTP 401/403).

    A configuration failure, not a request failure: a second call with the
    same credential fails the same way, so nothing retries it.
    """


#: HTTP statuses that mean the credential, not the request, was refused.
AUTHENTICATION_STATUS_CODES = frozenset({401, 403})


class ProviderRequestError(SummarizationError):
    """The provider was called and the request failed (API error, network)."""


class ProviderTimeoutError(ProviderRequestError):
    """The provider did not answer within the configured timeout."""


class MalformedOutputError(SummarizationError):
    """The provider answered, but not with output matching the schema."""


@dataclass(frozen=True)
class GenerationUsage:
    """Token counts for one provider call, exactly as the provider reported them.

    ``None`` means the provider did not report that figure.  It is never
    zero by default: an unknown count and a zero count are different facts.
    """

    prompt_tokens: Optional[int]
    candidate_tokens: Optional[int]
    total_tokens: Optional[int]


def _usage_from_response(response: object) -> Optional[GenerationUsage]:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None
    return GenerationUsage(
        prompt_tokens=getattr(metadata, "prompt_token_count", None),
        candidate_tokens=getattr(metadata, "candidates_token_count", None),
        total_tokens=getattr(metadata, "total_token_count", None),
    )


def build_user_prompt(theme: ThemeInput) -> str:
    """Serialize a theme's member stories for the model prompt."""
    lines = [f"Ticker: {theme.ticker}"]
    if theme.trading_day:
        lines.append(f"Trading day: {theme.trading_day}")
    lines.append("Member stories:")
    for story in theme.member_stories:
        lines.append(
            f"- id: {story.id}\n"
            f"  title: {story.title}\n"
            f"  description: {story.description}\n"
            f"  outlet: {story.outlet}\n"
            f"  time: {story.published_at}"
        )
    return "\n".join(lines)


def build_generation_config_kwargs(
    system_prompt: str,
    response_schema: type[BaseModel],
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict:
    """Pure builder for the Gemini generation config.

    Kept separate from the network call so the request shape (guardrail
    prompt, JSON mode, low temperature, token cap, timeout) is unit-testable
    without importing google.genai.  ``http_options`` is a plain mapping
    here; ``GenerateContentConfig`` coerces it to ``types.HttpOptions``.
    """
    return {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "temperature": DEFAULT_TEMPERATURE,
        "max_output_tokens": int(max_output_tokens),
        "http_options": {"timeout": int(timeout_ms)},
    }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ProviderConfigurationError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ProviderConfigurationError(f"{name} must be positive")
    return value


class GeminiClient:
    """Thin wrapper around google.genai for structured JSON generation.

    Kept injectable behind `summarize(theme, client=...)` so tests never need
    network access or an API key.

    Provider failures are normalized into the project-owned exception
    types above so callers never have to import ``google.genai`` to tell a
    timeout from a bad key from a schema miss.  Programmer defects (a
    ``TypeError`` from a wrong call shape, say) are not normalized: they
    are not provider failures and must surface as what they are.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        timeout_ms: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.timeout_ms = timeout_ms or _env_int(
            "GEMINI_TIMEOUT_MS", DEFAULT_TIMEOUT_MS
        )
        self.max_output_tokens = max_output_tokens or _env_int(
            "GEMINI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS
        )
        self._client = None
        #: Usage reported by the most recent ``generate()`` call, or ``None``
        #: when the provider reported nothing (or no call completed).
        self.last_usage: Optional[GenerationUsage] = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise ProviderConfigurationError("GEMINI_API_KEY is not configured")
            from google import genai  # lazy import: tests never need this installed

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(
        self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel]
    ) -> BaseModel:
        from google.genai import types

        self.last_usage = None
        client = self._get_client()
        config = types.GenerateContentConfig(
            **build_generation_config_kwargs(
                system_prompt,
                response_schema,
                max_output_tokens=self.max_output_tokens,
                timeout_ms=self.timeout_ms,
            )
        )
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
        except Exception as exc:
            # Only the SDK's and the transport's own exception types are
            # normalized, matched by ``isinstance`` against the real classes.
            # Anything else is not a provider outcome and is re-raised
            # exactly as it came.
            normalized = _normalize_provider_error(exc)
            if normalized is None:
                raise
            raise normalized from exc
        self.last_usage = _usage_from_response(response)
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise MalformedOutputError("provider returned no text and no parsed output")
        try:
            return response_schema.model_validate_json(text)
        except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
            raise MalformedOutputError(
                f"provider output did not match schema: {exc}"
            ) from exc


def _status_code(value: object) -> Optional[int]:
    """An HTTP status as an int, or ``None`` -- never a bool read as one."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _normalize_provider_error(exc: BaseException) -> Optional[SummarizationError]:
    """Map a real SDK or transport exception onto this module's types.

    Imports are deferred to the provider-call path so offline callers never
    load the SDK.  Classification is by the actual classes and their
    structured fields: ``google.genai.errors.APIError.code`` for the HTTP
    status, ``httpx``'s exception hierarchy for timeouts and transport
    failures.  A class is never recognized by its name.  ``None`` means
    "not a provider failure": the caller re-raises the original.
    """

    import httpx
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        code = _status_code(getattr(exc, "code", None))
        if code in AUTHENTICATION_STATUS_CODES:
            return ProviderAuthenticationError(
                f"provider rejected the credential (HTTP {code} "
                f"{getattr(exc, 'status', None) or ''})".rstrip()
            )
        return ProviderRequestError(f"provider request failed: {exc}")
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return ProviderTimeoutError(f"provider timed out: {exc}")
    if isinstance(exc, httpx.HTTPStatusError):
        code = _status_code(getattr(exc.response, "status_code", None))
        if code in AUTHENTICATION_STATUS_CODES:
            return ProviderAuthenticationError(
                f"provider rejected the credential (HTTP {code})"
            )
        return ProviderRequestError(f"provider request failed: {exc}")
    if isinstance(exc, (httpx.HTTPError, OSError)):
        return ProviderRequestError(f"provider request failed: {exc}")
    return None


def summarize(
    theme: ThemeInput, *, client: Optional[GeminiClient] = None
) -> ThemeSummary:
    """Generate a cited ThemeSummary for `theme`.

    Retries once on a structurally malformed provider response - including a
    response whose citation_ids don't all resolve to a real member story,
    which is treated as invalid output rather than allowed through - then
    raises SummarizationError. Does not retry or degrade on banned-language
    guardrail failures; that loop belongs to A2's guardrail chain, which
    wraps this function.
    """
    if not theme.member_stories:
        raise SummarizationError("theme has no member stories to summarize")

    active_client = client or GeminiClient()
    user_prompt = build_user_prompt(theme)

    last_error: Optional[Exception] = None
    for _ in range(MAX_RETRIES + 1):
        try:
            result = active_client.generate(SYSTEM_PROMPT, user_prompt, ThemeSummary)
            if not isinstance(result, ThemeSummary):
                result = ThemeSummary.model_validate(result)
            unresolved = resolve_citations(theme, result)
            if unresolved:
                raise SummarizationError(
                    f"summary cited unknown story id(s): {sorted(unresolved)}"
                )
            return result
        except Exception as exc:
            # Provider/parse/citation failure - retried once, then raised below.
            last_error = exc

    raise SummarizationError(
        f"failed to produce a valid theme summary after {MAX_RETRIES + 1} "
        f"attempt(s): {last_error}"
    ) from last_error


def resolve_citations(theme: ThemeInput, summary: ThemeSummary) -> set[str]:
    """Return citation ids used in `summary` that do not match any member story id.

    Pure helper reused by A2's guardrail chain; `summarize()` does not enforce
    this itself.
    """
    known_ids = {story.id for story in theme.member_stories}
    used_ids = {
        citation_id
        for sentence in summary.sentences
        for citation_id in sentence.citation_ids
    }
    return used_ids - known_ids

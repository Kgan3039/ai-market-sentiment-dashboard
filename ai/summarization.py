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
  dependency on I1's real persistence layer, M5's clustering, or A2's
  banned-phrase linter / regenerate-with-feedback / degrade chain (a
  separate, later issue). `summarize()` only retries on structurally
  malformed provider output.

Provider: Gemini (google-genai), low temperature, structured JSON output via
response_schema so parsing failures are rare by construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, field_validator

MAX_LABEL_WORDS = 8
MIN_SENTENCES = 2
MAX_SENTENCES = 4
MAX_RETRIES = 1  # one retry on structurally malformed provider output

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.1

SYSTEM_PROMPT = """You are generating a short, neutral summary of news coverage for a stock ticker theme.

Rules (must all be followed exactly):
1. Output strict JSON matching the provided schema. No prose outside the JSON.
2. "label" is a short theme title of at most 8 words. No punctuation-only labels.
3. "sentences" has between 2 and 4 entries.
4. Every sentence's "citation_ids" must contain at least one id, and every id must be
   one of the story ids given in the input. Never invent an id.
5. Every factual claim must be attributed to at least one cited story. Do not state
   anything that is not supported by the cited stories.
6. Never give financial advice or a trading recommendation (e.g. "buy", "sell", "hold").
7. Never predict future price or stock movement (e.g. "will rise", "expected to fall").
8. Never claim a causal explanation for a price move (e.g. "fell because",
   "this move was driven by", "explains today's decline").
9. Describe what the coverage says, not what will happen next. Use framing like
   "coverage today is dominated by..." or "the most-covered storyline is...",
   never a causal or predictive framing, even if the source headlines use one.
10. Do not include any commentary, caveats, or meta-text about these rules in the output.
"""


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


class ThemeSummary(BaseModel):
    label: str
    sentences: list[Sentence] = Field(min_length=MIN_SENTENCES, max_length=MAX_SENTENCES)

    @field_validator("label")
    @classmethod
    def _label_word_count(cls, value: str) -> str:
        word_count = len(value.split())
        if word_count == 0 or word_count > MAX_LABEL_WORDS:
            raise ValueError(f"label must be 1-{MAX_LABEL_WORDS} words, got {word_count}")
        return value


class SummarizationError(RuntimeError):
    """Raised when the provider fails to produce a schema-valid summary after retrying."""


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


def build_generation_config_kwargs(system_prompt: str, response_schema: type[BaseModel]) -> dict:
    """Pure builder for the Gemini generation config.

    Kept separate from the network call so the request shape (guardrail
    prompt, JSON mode, low temperature) is unit-testable without importing
    google.genai.
    """
    return {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "temperature": DEFAULT_TEMPERATURE,
    }


class GeminiClient:
    """Thin wrapper around google.genai for structured JSON generation.

    Kept injectable behind `summarize(theme, client=...)` so tests never need
    network access or an API key.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise SummarizationError("GEMINI_API_KEY is not configured")
            from google import genai  # lazy import: tests never need this installed

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(self, system_prompt: str, user_prompt: str, response_schema: type[BaseModel]) -> BaseModel:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            **build_generation_config_kwargs(system_prompt, response_schema)
        )
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed
        return response_schema.model_validate_json(response.text)


def summarize(theme: ThemeInput, *, client: Optional[GeminiClient] = None) -> ThemeSummary:
    """Generate a cited ThemeSummary for `theme`.

    Retries once on a structurally malformed provider response, then raises
    SummarizationError. Does not retry or degrade on semantic guardrail
    failures (banned language, citation resolution) - that loop belongs to
    A2's guardrail chain, which wraps this function.
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
            return result
        except Exception as exc:  # noqa: BLE001 - provider/parse failure, retried once
            last_error = exc

    raise SummarizationError(
        f"failed to produce a valid theme summary after {MAX_RETRIES + 1} attempt(s): {last_error}"
    ) from last_error


def resolve_citations(theme: ThemeInput, summary: ThemeSummary) -> set[str]:
    """Return citation ids used in `summary` that do not match any member story id.

    Pure helper reused by A2's guardrail chain; `summarize()` does not enforce
    this itself.
    """
    known_ids = {story.id for story in theme.member_stories}
    used_ids = {citation_id for sentence in summary.sentences for citation_id in sentence.citation_ids}
    return used_ids - known_ids

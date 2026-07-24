"""Stable request-free response models for the Phase 0 read API."""

from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """A publisher source that a generated sentence can cite."""

    id: str
    headline: str
    outlet: str
    url: str
    published_at: datetime


class Story(Citation):
    """A canonical story included in a theme or Other coverage."""


class CitedSentence(BaseModel):
    """One generated sentence and the story identifiers supporting it."""

    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class Theme(BaseModel):
    """A ranked coverage theme with traceable summary content."""

    id: str
    label: str = Field(min_length=1, max_length=120)
    rank: int = Field(ge=1)
    # Mirrors ai.summarization.ThemeSummary so pipeline output can pass through unchanged.
    sentences: list[CitedSentence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    stories: list[Story] = Field(default_factory=list)
    outlet_count: int = Field(ge=0)
    story_count: int = Field(ge=0)
    degraded: bool = False

    @model_validator(mode="after")
    def validate_summary_references(self):
        if self.degraded and self.sentences:
            raise ValueError("Degraded themes must not include generated summary sentences")

        citation_ids = {citation.id for citation in self.citations}
        for sentence in self.sentences:
            unknown_ids = set(sentence.citation_ids) - citation_ids
            if unknown_ids:
                raise ValueError(f"Summary references unknown citations: {sorted(unknown_ids)}")
        return self


class OtherCoverage(BaseModel):
    """Stories intentionally left outside the ranked theme cards."""

    outlet_count: int = Field(ge=0)
    story_count: int = Field(ge=0)
    stories: list[Story] = Field(default_factory=list)


class TickerThemesResponse(BaseModel):
    """All presentation data for one ticker on one trading day."""

    ticker: str
    date: date
    data_as_of: datetime
    themes: list[Theme] = Field(default_factory=list)
    other_coverage: OtherCoverage


class TickerListItem(BaseModel):
    """The lightweight data required to render a ticker tab."""

    ticker: str
    company_name: str
    data_as_of: datetime
    theme_count: int = Field(ge=0)
    is_stale: bool


class TickerListResponse(BaseModel):
    """The fixed Phase 0 ticker universe."""

    data_as_of: datetime
    tickers: list[TickerListItem]


class StageRunStatus(BaseModel):
    """Latest pipeline result for one stage."""

    stage: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    error_count: int = Field(default=0, ge=0)


class MetaStatusResponse(BaseModel):
    """Freshness data consumed by the page header and stale-state banner."""

    data_as_of: datetime
    is_stale: bool
    last_runs: list[StageRunStatus]


class HealthResponse(BaseModel):
    """Small operational check retained outside the Phase 0 product surface."""

    status: str
    version: str
    message: str

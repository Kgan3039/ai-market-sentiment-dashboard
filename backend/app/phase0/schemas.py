"""Stable request-free response models for the Phase 0 read API."""

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("citation_ids")
    @classmethod
    def validate_unique_citation_ids(cls, citation_ids: list[str]) -> list[str]:
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Sentence citation ids must be unique")
        return citation_ids


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
    def validate_theme_integrity(self):
        if self.degraded and self.sentences:
            raise ValueError("Degraded themes must not include generated summary sentences")
        if not self.degraded and not 2 <= len(self.sentences) <= 4:
            raise ValueError("Non-degraded themes must include 2 to 4 cited summary sentences")

        story_ids = [story.id for story in self.stories]
        citation_ids = [citation.id for citation in self.citations]
        if len(story_ids) != len(set(story_ids)):
            raise ValueError("Theme stories must have unique ids")
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Theme citations must have unique ids")
        if self.story_count != len(self.stories):
            raise ValueError("Theme story_count must match the returned story list")
        if self.outlet_count != len({story.outlet for story in self.stories}):
            raise ValueError("Theme outlet_count must match the returned story list")

        story_by_id = {story.id: story for story in self.stories}
        for citation in self.citations:
            story = story_by_id.get(citation.id)
            if story is None:
                raise ValueError("Theme citations must belong to returned member stories")
            if citation.model_dump() != story.model_dump():
                raise ValueError("Theme citations must match their returned member stories")

        citation_id_set = set(citation_ids)
        for sentence in self.sentences:
            unknown_ids = set(sentence.citation_ids) - citation_id_set
            if unknown_ids:
                raise ValueError(f"Summary references unknown citations: {sorted(unknown_ids)}")
        return self


class OtherCoverage(BaseModel):
    """Stories intentionally left outside the ranked theme cards."""

    outlet_count: int = Field(ge=0)
    story_count: int = Field(ge=0)
    stories: list[Story] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self):
        story_ids = [story.id for story in self.stories]
        if len(story_ids) != len(set(story_ids)):
            raise ValueError("Other coverage stories must have unique ids")
        if self.story_count != len(self.stories):
            raise ValueError("Other coverage story_count must match the returned story list")
        if self.outlet_count != len({story.outlet for story in self.stories}):
            raise ValueError("Other coverage outlet_count must match the returned story list")
        return self


class TickerThemesResponse(BaseModel):
    """All presentation data for one ticker on one trading day."""

    ticker: str
    date: date
    data_as_of: datetime
    themes: list[Theme] = Field(default_factory=list)
    other_coverage: OtherCoverage

    @model_validator(mode="after")
    def validate_story_placement(self):
        ranks = [theme.rank for theme in self.themes]
        if len(ranks) != len(set(ranks)):
            raise ValueError("Theme ranks must be unique for a ticker day")

        story_ids = [
            story.id
            for theme in self.themes
            for story in theme.stories
        ] + [story.id for story in self.other_coverage.stories]
        if len(story_ids) != len(set(story_ids)):
            raise ValueError("A story can appear in only one theme or Other coverage")
        return self


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

"""B1: the narrative read API over the pipeline's persisted SQLite state.

**A GET only reads.**  This module holds a :class:`phase0.repository.Phase0Reader`
-- a path, never a connection, and every query it runs opens SQLite
``mode=ro`` with ``query_only`` and a write-denying authorizer -- and never a
:class:`~phase0.repository.Phase0Repository`, so no run, no stage key and no
summary can be written from here.  No summary is generated here either:
nothing in this module calls ``ensure_summary``, A2, or a provider client's
``generate``.  A theme without a current summary is served degraded.

**Currentness is A3's.**  A theme's generated label and sentences are
served only when :func:`phase0.summary_lifecycle.current_summary_artifact`
returns a :class:`~phase0.summary_lifecycle.CurrentSummary` under the policy
production generates under (:func:`phase0.summary_runner.production_generation_policy`,
the same boundary the scheduler uses; resolving it makes no provider call
and needs no credential).  Nothing here re-derives currentness.

**One snapshot per response.**  The layout -- themes, Other Coverage, which
stories exist -- comes from one ``theme_population`` read (P1).
``current_summary_artifact`` reads its own fresh snapshot (P2).  A summary is
combined with the P1 layout only if P1 projects onto the exact frozen input
the artifact is current for; otherwise the theme is served degraded rather
than mixing two states.

**Current themes are shown from their frozen evidence.**  Their stories and
citations are the :class:`~ai.guarded_summary.EvidenceStory` records the
artifact was validated against, not a fresh read of story rows.  Degraded
themes and Other Coverage are projected from the persisted stories with the
same rules :func:`phase0.summaries.evidence_for` applies.

**Population health is A2's gate.**  A day
:func:`phase0.summaries.assess_population` refuses (no theme set, M2-only,
mixed or inconsistent) has no themes; every live story is Other Coverage.

Anything the public contract cannot represent without inventing data -- a
story without a URL, a timestamp without an offset, a theme without its
stable key -- is a :class:`NarrativeUnavailableError`, never a guess and
never a fixture.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from ai.guarded_summary import GenerationPolicy, citation_id_for
from ai.summarization import SummarizationError
from phase0.errors import Phase0Error
from phase0.repository import (
    DATABASE_READ_ERRORS,
    MIGRATIONS_PATH,
    PersistedStory,
    Phase0Reader,
    ThemeMembership,
    ThemePopulation,
)
from phase0.schema import latest_version, load_migrations
from phase0.summaries import (
    SummaryInputError,
    assess_population,
    build_generation_input,
    utc_published_at,
)
from phase0.summary_lifecycle import CurrentSummary, current_summary_artifact
from phase0.summary_runner import production_generation_policy
from phase0.tickers import TICKER_UNIVERSE

from .repository import (
    NarrativeUnavailableError,
    _current_utc_time,
    is_stale_during_market_hours,
)
from .schemas import (
    CitedSentence,
    MetaStatusResponse,
    OtherCoverage,
    StageRunStatus,
    Story,
    Theme,
    TickerListItem,
    TickerThemesResponse,
)

logger = logging.getLogger(__name__)

#: Runs whose completion can stand as "data as of": a success, else a
#: degraded run, which persisted real evidence with some source incomplete.
_FRESHNESS_STATUSES: tuple[tuple[str, ...], ...] = (("success",), ("degraded",))

#: Failures that mean "the persisted state cannot be served", as opposed to
#: programming defects, which are left to surface as themselves.  ValueError
#: covers pydantic's ValidationError and A2's GuardedSummaryError; LookupError
#: keeps an internal KeyError from being reported as an unknown ticker.
_UNAVAILABLE_ERRORS = (
    *DATABASE_READ_ERRORS,
    Phase0Error,
    ValueError,
    LookupError,
    OSError,
)


class _IntegrityFailure(Phase0Error):
    """Persisted rows break an invariant the public contract relies on."""


@contextmanager
def _serving(what: str) -> Iterator[None]:
    """Turn an unservable persisted state into ``NarrativeUnavailableError``.

    The cause is logged here and chained for the server; the route answers
    with a fixed message and no detail.
    """

    try:
        yield
    except _UNAVAILABLE_ERRORS as exc:
        logger.warning(
            "Narrative %s unavailable: %s: %s", what, type(exc).__name__, exc
        )
        raise NarrativeUnavailableError(what) from exc


class SqliteNarrativeRepository:
    """The persisted narrative read model.  Implements ``NarrativeReadRepository``."""

    def __init__(
        self,
        *,
        database_path: str | Path,
        pipeline_version: str,
        policy_resolver: Callable[[], GenerationPolicy] = production_generation_policy,
        now_provider: Callable[[], datetime] = _current_utc_time,
    ) -> None:
        version = str(pipeline_version or "").strip()
        if not version:
            raise ValueError("pipeline_version is required")
        self.pipeline_version = version
        self.reader = Phase0Reader(Path(database_path))
        self._policy_resolver = policy_resolver
        self._now_provider = now_provider
        self._schema_version = latest_version(load_migrations(MIGRATIONS_PATH))

    # -- NarrativeReadRepository -------------------------------------------

    def get_status(self) -> MetaStatusResponse:
        with _serving("status"):
            return self._status(self._now_provider())[0]

    def list_tickers(self) -> list[TickerListItem]:
        with _serving("tickers"):
            # One reference instant for the whole list, so two tickers can
            # never be judged across a clock boundary within one response.
            now = self._now_provider()
            status, _ = self._status(now)
            items: list[TickerListItem] = []
            for ticker, company_name in TICKER_UNIVERSE.items():
                day = self.reader.latest_story_day(ticker, self.pipeline_version)
                theme_count = 0
                data_as_of = status.data_as_of
                if day is not None:
                    population = self._population(ticker, day)
                    if assess_population(population) is None:
                        theme_count = len(population.themes)
                    data_as_of = self._partition_data_as_of(ticker, day, status)
                items.append(
                    TickerListItem(
                        ticker=ticker,
                        company_name=company_name,
                        data_as_of=data_as_of,
                        theme_count=theme_count,
                        # This ticker's own freshness, not the global one.
                        is_stale=is_stale_during_market_hours(data_as_of, now),
                    )
                )
            return items

    def get_themes(
        self, ticker: str, requested_date: str | None
    ) -> TickerThemesResponse:
        symbol = str(ticker).upper()
        if symbol not in TICKER_UNIVERSE:
            raise KeyError(symbol)
        with _serving("themes"):
            status, anchor_day = self._status(self._now_provider())
            day = (
                requested_date
                or self.reader.latest_story_day(symbol, self.pipeline_version)
                or anchor_day
            )
            population = self._population(symbol, day)
            themes, other = self._layout(population)
            return TickerThemesResponse(
                ticker=symbol,
                date=day,
                data_as_of=self._partition_data_as_of(symbol, day, status),
                themes=themes,
                other_coverage=other,
            )

    # -- Status and freshness ------------------------------------------------

    def _check_schema(self) -> None:
        found = self.reader.schema_version()
        if found != self._schema_version:
            raise _IntegrityFailure(
                f"database schema version {found}, expected {self._schema_version}"
            )

    def _status(self, now: datetime) -> tuple[MetaStatusResponse, str]:
        """The status response and the trading day its ``data_as_of`` came from."""

        self._check_schema()
        anchor = None
        for statuses in _FRESHNESS_STATUSES:
            anchor = self.reader.latest_run_completion(self.pipeline_version, statuses)
            if anchor is not None:
                break
        if anchor is None:
            raise _IntegrityFailure("no completed pipeline run is recorded")
        data_as_of = _aware_timestamp(anchor["completed_at"], "run completed_at")
        last_runs = [
            StageRunStatus(
                stage=row["stage"],
                status=row["status"],
                started_at=_aware_timestamp(row["started_at"], "run started_at"),
                completed_at=_aware_timestamp(row["completed_at"], "run completed_at"),
                duration_ms=row["duration_ms"],
                error_count=row["error_count"],
            )
            for row in self.reader.latest_stage_runs(self.pipeline_version)
        ]
        status = MetaStatusResponse(
            data_as_of=data_as_of,
            is_stale=is_stale_during_market_hours(data_as_of, now),
            last_runs=last_runs,
        )
        return status, str(anchor["trading_day"])

    def _partition_data_as_of(
        self, ticker: str, day: str, status: MetaStatusResponse
    ) -> datetime:
        row = self.reader.latest_run_completion(
            self.pipeline_version,
            ("success", "degraded"),
            ticker=ticker,
            trading_day=day,
        )
        if row is None:
            return status.data_as_of
        return _aware_timestamp(row["completed_at"], "run completed_at")

    # -- One partition -------------------------------------------------------

    def _population(self, ticker: str, day: str) -> ThemePopulation:
        return self.reader.theme_population(ticker, day, self.pipeline_version)

    def _layout(self, population: ThemePopulation) -> tuple[list[Theme], OtherCoverage]:
        live = {story.story_id: story for story in population.stories.stories}
        if assess_population(population) is not None:
            # Not a valid view of the day: no themes, and nothing hidden.
            return [], _other_coverage(
                [_persisted_story(live[story_id]) for story_id in sorted(live)]
            )

        policy = self._policy() if population.themes else None
        themes = [
            self._theme(population, membership, live, policy)
            for membership in sorted(
                population.themes,
                key=lambda theme: (theme.salience_rank, theme.theme_id),
            )
        ]
        placed = [
            entry.story_id
            for entry in sorted(
                population.other_coverage,
                key=lambda entry: (entry.position, entry.story_id),
            )
        ] + sorted(entry.story_id for entry in population.excluded)
        return themes, _other_coverage([_persisted_story(live[i]) for i in placed])

    def _policy(self) -> Optional[GenerationPolicy]:
        """The production policy, or ``None`` when it cannot be resolved.

        ``None`` serves every theme degraded: with no policy there is no
        currentness, so no summary may be shown -- and none is generated.
        """

        try:
            return self._policy_resolver()
        except (SummarizationError, ValueError, OSError) as exc:
            logger.warning(
                "Summary policy unresolvable; serving themes degraded (%s)",
                type(exc).__name__,
            )
            return None

    def _theme(
        self,
        population: ThemePopulation,
        membership: ThemeMembership,
        live: dict[int, PersistedStory],
        policy: Optional[GenerationPolicy],
    ) -> Theme:
        if not membership.theme_key or not membership.theme_key.strip():
            raise _IntegrityFailure(f"theme {membership.theme_id} has no theme_key")
        current = self._current(population, membership, policy)
        if current is None:
            stories = [_persisted_story(live[i]) for i in membership.story_ids]
            label = membership.label
            sentences: list[CitedSentence] = []
        else:
            stories = [
                _evidence_story(story) for story in current.generation_input.evidence
            ]
            label = current.artifact.label
            sentences = [
                CitedSentence(
                    text=sentence.text,
                    citation_ids=[
                        citation_id_for(citation.story_id)
                        for citation in sentence.citations
                    ],
                )
                for sentence in current.artifact.sentences
            ]
        return Theme(
            id=membership.theme_key,
            label=label,
            rank=membership.salience_rank,
            sentences=sentences,
            citations=list(stories),
            stories=stories,
            outlet_count=len({story.outlet for story in stories}),
            story_count=len(stories),
            degraded=current is None,
        )

    def _current(
        self,
        population: ThemePopulation,
        membership: ThemeMembership,
        policy: Optional[GenerationPolicy],
    ) -> Optional[CurrentSummary]:
        """A3's current summary for this theme, if it matches the layout snapshot."""

        if policy is None:
            return None
        current = current_summary_artifact(
            self.reader,
            population.ticker,
            population.trading_day,
            population.pipeline_version,
            membership.theme_id,
            policy,
        )
        if current is None:
            return None
        try:
            layout_input = build_generation_input(population, membership.theme_id)
        except SummaryInputError:
            layout_input = None
        if (
            layout_input is None
            or layout_input.input_fingerprint
            != current.generation_input.input_fingerprint
        ):
            logger.info(
                "Theme %s changed between reads; serving it degraded",
                membership.theme_id,
            )
            return None
        return current


# ----------------------------------------------------------------------
# Projections onto the public contract
# ----------------------------------------------------------------------


def _aware_timestamp(value: Any, field: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise _IntegrityFailure(f"{field} is not ISO-8601") from exc
    if stamp.tzinfo is None or stamp.tzinfo.utcoffset(stamp) is None:
        raise _IntegrityFailure(f"{field} carries no UTC offset")
    return stamp.astimezone(timezone.utc)


def _api_story(
    *,
    citation_id: str,
    headline: str,
    outlet: str,
    published_at: str,
    urls: Sequence[str],
) -> Story:
    # The selected provenance URL is served as stored or not at all: a blank
    # one is neither stripped into shape nor replaced by another candidate.
    if not urls or not urls[0].strip():
        raise _IntegrityFailure(f"{citation_id} has no usable source URL")
    return Story(
        id=citation_id,
        headline=headline,
        outlet=outlet,
        url=urls[0],
        published_at=(
            None
            if published_at == ""
            else _aware_timestamp(published_at, f"{citation_id} published_at")
        ),
    )


def _evidence_story(evidence: Any) -> Story:
    """A current theme's story, exactly as its frozen evidence records it."""

    return _api_story(
        citation_id=evidence.citation_id,
        headline=evidence.title,
        outlet=evidence.outlet,
        published_at=evidence.published_at,
        urls=evidence.urls,
    )


def _persisted_story(story: PersistedStory) -> Story:
    """A persisted story, with :func:`phase0.summaries.evidence_for`'s rules.

    Title is the canonical title; outlet is the story's own, else the
    lexicographically first member outlet; URLs are the canonical URL then
    each member's, in member position order; the timestamp is rendered in
    UTC by the same helper, and absent stays absent.  The description,
    which only the model sees, is not needed and is not recovered.
    """

    ordered = sorted(story.members, key=lambda member: member.position)
    outlet = story.outlet or ""
    if not outlet:
        named = sorted({member.outlet for member in ordered if member.outlet})
        outlet = named[0] if named else ""
    urls: list[str] = []
    for candidate in [story.canonical_url] + [
        member.canonical_url or member.url for member in ordered
    ]:
        if candidate and candidate not in urls:
            urls.append(candidate)
    return _api_story(
        citation_id=citation_id_for(story.story_id),
        headline=story.canonical_title,
        outlet=outlet,
        published_at=utc_published_at(story.published_at, story.story_id),
        urls=urls,
    )


def _other_coverage(stories: list[Story]) -> OtherCoverage:
    return OtherCoverage(
        outlet_count=len({story.outlet for story in stories}),
        story_count=len(stories),
        stories=stories,
    )


__all__ = ["SqliteNarrativeRepository"]

"""Read-model boundary for fixture data now and SQLite data after I1 lands."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .schemas import (
    MetaStatusResponse,
    OtherCoverage,
    TickerListItem,
    TickerThemesResponse,
)


class NarrativeReadRepository(Protocol):
    """Minimal read contract the API needs from the Phase 0 persistence layer."""

    def list_tickers(self) -> list[TickerListItem]: ...

    def get_themes(self, ticker: str, requested_date: str | None) -> TickerThemesResponse: ...

    def get_status(self) -> MetaStatusResponse: ...


class FixtureNarrativeRepository:
    """Deterministic source used until the pipeline's SQLite repository is merged."""

    def __init__(self, fixture: dict):
        self.fixture = fixture

    @classmethod
    def from_default_fixture(cls) -> "FixtureNarrativeRepository":
        return cls(_load_fixture())

    def list_tickers(self) -> list[TickerListItem]:
        status = self.get_status()
        tickers: list[TickerListItem] = []
        for ticker_data in self.fixture["tickers"]:
            ticker = ticker_data["ticker"]
            latest_day = self._latest_day(ticker)
            tickers.append(
                TickerListItem(
                    ticker=ticker,
                    company_name=ticker_data["company_name"],
                    data_as_of=latest_day["data_as_of"],
                    theme_count=len(latest_day["themes"]),
                    is_stale=status.is_stale,
                )
            )
        return tickers

    def get_themes(self, ticker: str, requested_date: str | None) -> TickerThemesResponse:
        ticker = ticker.upper()
        if ticker not in self.fixture["ticker_days"]:
            raise KeyError(ticker)

        days = self.fixture["ticker_days"][ticker]
        if requested_date and requested_date in days:
            payload = days[requested_date]
            selected_date = requested_date
        elif requested_date:
            # A valid ticker with no coverage on a requested date is an empty state, not an error.
            return TickerThemesResponse(
                ticker=ticker,
                date=requested_date,
                data_as_of=self.get_status().data_as_of,
                themes=[],
                other_coverage=OtherCoverage(outlet_count=0, story_count=0, stories=[]),
            )
        else:
            selected_date = max(days)
            payload = days[selected_date]

        return TickerThemesResponse.model_validate(
            {
                "ticker": ticker,
                "date": selected_date,
                **payload,
            }
        )

    def get_status(self) -> MetaStatusResponse:
        return MetaStatusResponse.model_validate(self.fixture["status"])

    def _latest_day(self, ticker: str) -> dict:
        days = self.fixture["ticker_days"][ticker]
        return days[max(days)]


@lru_cache(maxsize=1)
def _load_fixture() -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / "phase0_narratives.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


@lru_cache(maxsize=1)
def get_narrative_repository() -> NarrativeReadRepository:
    """Return the current read source; replace this factory with I1's SQLite adapter later."""
    return FixtureNarrativeRepository.from_default_fixture()

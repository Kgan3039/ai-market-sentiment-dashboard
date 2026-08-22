"""Read-model boundary for the Phase 0 narrative API.

Still fixture-backed. I1's SQLite persistence layer
(:mod:`phase0.repository`) is on ``main``, but nothing yet writes the
``stories`` and ``themes`` it would read: the downstream stages are
implemented in :mod:`nlp` and not registered in ``pipeline.py``. The
adapter from that repository to this read model is deliberately
deferred until the stored output is stable -- see
``docs/decisions/I5-decisions.md``.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol
from zoneinfo import ZoneInfo

from .schemas import (
    MetaStatusResponse,
    OtherCoverage,
    TickerListItem,
    TickerThemesResponse,
)

EASTERN_TIME = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16)


def _current_utc_time() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_stale_during_market_hours(data_as_of: object, now: datetime) -> bool:
    """Return stale only during the regular U.S. market session.

    The session is evaluated in U.S. Eastern time.
    """
    timestamp = _parse_timestamp(data_as_of)
    if timestamp is None:
        return False

    current_time = _parse_timestamp(now)
    if current_time is None:
        return False

    eastern_now = current_time.astimezone(EASTERN_TIME)
    market_time = eastern_now.replace(tzinfo=None).time()
    is_market_weekday = eastern_now.weekday() < 5
    is_market_session = MARKET_OPEN <= market_time < MARKET_CLOSE
    if not is_market_weekday or not is_market_session:
        return False

    return current_time - timestamp > timedelta(hours=1)


class NarrativeReadRepository(Protocol):
    """Minimal read contract for the Phase 0 persistence layer."""

    def list_tickers(self) -> list[TickerListItem]:
        ...

    def get_themes(
        self, ticker: str, requested_date: str | None
    ) -> TickerThemesResponse:
        ...

    def get_status(self) -> MetaStatusResponse:
        ...


class FixtureNarrativeRepository:
    """Fixture source until the pipeline SQLite repository merges."""

    def __init__(
        self,
        fixture: dict,
        now_provider: Callable[[], datetime] = _current_utc_time,
    ):
        self.fixture = fixture
        self._now_provider = now_provider

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

    def get_themes(
        self, ticker: str, requested_date: str | None
    ) -> TickerThemesResponse:
        ticker = ticker.upper()
        if ticker not in self.fixture["ticker_days"]:
            raise KeyError(ticker)

        days = self.fixture["ticker_days"][ticker]
        if requested_date and requested_date in days:
            payload = days[requested_date]
            selected_date = requested_date
        elif requested_date:
            # A valid ticker with no coverage on a requested date is an empty
            # state, not an error.
            empty_coverage = OtherCoverage(
                outlet_count=0,
                story_count=0,
                stories=[],
            )
            return TickerThemesResponse(
                ticker=ticker,
                date=requested_date,
                data_as_of=self.get_status().data_as_of,
                themes=[],
                other_coverage=empty_coverage,
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
        status_payload = dict(self.fixture["status"])
        data_as_of = _parse_timestamp(status_payload.get("data_as_of"))
        if data_as_of is None:
            raise ValueError("Fixture status data_as_of must be valid")

        status_payload["data_as_of"] = data_as_of
        status_payload["is_stale"] = is_stale_during_market_hours(
            data_as_of,
            self._now_provider(),
        )
        return MetaStatusResponse.model_validate(status_payload)

    def _latest_day(self, ticker: str) -> dict:
        days = self.fixture["ticker_days"][ticker]
        return days[max(days)]


@lru_cache(maxsize=1)
def _load_fixture() -> dict:
    fixture_directory = Path(__file__).parent / "fixtures"
    fixture_path = fixture_directory / "phase0_narratives.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


@lru_cache(maxsize=1)
def get_narrative_repository() -> NarrativeReadRepository:
    """Return the read source.

    Fixture-backed until a SQLite adapter is built on top of
    :class:`phase0.repository.Phase0Repository`; that waits on the
    downstream stages actually producing stored output.
    """
    return FixtureNarrativeRepository.from_default_fixture()

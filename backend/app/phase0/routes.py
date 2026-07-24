"""Phase 0 public read endpoints for the Ticker Narratives page."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from .repository import NarrativeReadRepository, get_narrative_repository
from .schemas import MetaStatusResponse, TickerListResponse, TickerThemesResponse

router = APIRouter(prefix="/api/v1", tags=["Ticker Narratives"])


def repository_dependency() -> NarrativeReadRepository:
    return get_narrative_repository()


@router.get("/tickers", response_model=TickerListResponse)
def list_tickers(
    repository: NarrativeReadRepository = Depends(repository_dependency),
) -> TickerListResponse:
    """Return the fixed Phase 0 ticker universe and its current coverage counts."""
    status = repository.get_status()
    return TickerListResponse(data_as_of=status.data_as_of, tickers=repository.list_tickers())


@router.get("/tickers/{ticker}/themes", response_model=TickerThemesResponse)
def get_ticker_themes(
    ticker: str,
    date_value: date | None = Query(default=None, alias="date"),
    repository: NarrativeReadRepository = Depends(repository_dependency),
) -> TickerThemesResponse:
    """Return ranked, cited coverage themes for one ticker and optional trading day."""
    try:
        return repository.get_themes(ticker, date_value.isoformat() if date_value else None)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Ticker is not part of the Phase 0 universe.",
        ) from exc


@router.get("/meta/status", response_model=MetaStatusResponse)
def get_meta_status(
    repository: NarrativeReadRepository = Depends(repository_dependency),
) -> MetaStatusResponse:
    """Return the latest pipeline stage status and page freshness timestamp."""
    return repository.get_status()

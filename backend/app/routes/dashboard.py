"""Dashboard routes for aggregated frontend summary data."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import AvailabilityStatus, DashboardAvailability, DashboardSummary
from app.services.data_service import DataService
from app.services.prediction_service import PredictionService
from app.services.sentiment_service import SentimentService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def _availability_status(available: bool, source: str, detail: str, item_count: int | None = None) -> AvailabilityStatus:
    return AvailabilityStatus(
        available=available,
        source=source,
        detail=detail,
        item_count=item_count,
    )


@router.get("/summary/{ticker}", response_model=DashboardSummary)
async def get_dashboard_summary(ticker: str):
    """
    Get a comprehensive dashboard summary for a stock ticker.

    Existing contract fields (`ticker`, `date`, `sentiment`, `market_data`, `prediction`)
    remain unchanged. Additional fields extend the response with headline items,
    availability metadata, and fundamentals.
    """
    if not ticker or len(ticker) < 1:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        ticker = ticker.upper()

        sentiment_data = SentimentService.get_sentiment_for_ticker(ticker)
        market_data = DataService.get_market_data(ticker)
        prediction_data = PredictionService.predict_for_ticker(ticker)
        headline_data = DataService.get_headlines_for_ticker(ticker)
        fundamentals_data = DataService.get_fundamentals_for_ticker(ticker)

        if not sentiment_data or "overall_sentiment" not in sentiment_data:
            raise ValueError("Invalid sentiment data")
        if not market_data or not hasattr(market_data, "date"):
            raise ValueError("Invalid market data")
        if not prediction_data or "prediction" not in prediction_data:
            raise ValueError("Invalid prediction data")

        headlines = headline_data.get("headlines", [])
        fundamentals = fundamentals_data.get("fundamentals")

        availability = DashboardAvailability(
            sentiment=_availability_status(
                available=sentiment_data.get("overall_sentiment") is not None,
                source=sentiment_data.get("source", "unknown"),
                item_count=sentiment_data.get("item_count"),
                detail=f"Scored {sentiment_data.get('item_count', 0)} text items",
            ),
            prediction=_availability_status(
                available=prediction_data.get("prediction") is not None,
                source=prediction_data.get("source", "unknown"),
                detail="Generated from current market and sentiment inputs",
            ),
            headlines=_availability_status(
                available=len(headlines) > 0,
                source=headline_data.get("source", "unknown"),
                item_count=len(headlines),
                detail=f"{len(headlines)} headline item(s) available",
            ),
            fundamentals=_availability_status(
                available=fundamentals is not None,
                source=fundamentals_data.get("source", "unknown"),
                detail="Fundamentals source available" if fundamentals is not None else "No fundamentals source available",
            ),
        )

        return DashboardSummary(
            ticker=ticker,
            date=market_data.date,
            sentiment=sentiment_data["overall_sentiment"],
            market_data=market_data,
            prediction=prediction_data["prediction"],
            headlines=headlines,
            availability=availability,
            fundamentals=fundamentals,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard data: {str(e)}")


@router.get("/summary-batch")
async def get_dashboard_summary_batch(tickers: List[str] = Query(...)):
    """Get dashboard summaries for multiple stocks."""
    if not tickers or len(tickers) == 0:
        raise HTTPException(status_code=400, detail="At least one ticker required")

    if len(tickers) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 tickers per request")

    try:
        summaries = []
        for ticker in tickers:
            summaries.append(await get_dashboard_summary(ticker.upper()))
        return summaries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard summaries: {str(e)}")

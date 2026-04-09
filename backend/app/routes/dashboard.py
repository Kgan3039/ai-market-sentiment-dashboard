"""
Dashboard Routes

Endpoints for the frontend dashboard to retrieve aggregated data.

Author: Mihir (with aggregation from Matthew NLP, Isaac data, Abhi ML)
Status: Placeholder implementation - awaiting service integrations

Endpoints:
- GET /dashboard/summary/{ticker} - Get comprehensive dashboard summary for one stock
- GET /dashboard/summary-batch - Get dashboard summaries for multiple stocks

TODO (Mihir): Add in-memory caching with 1 hour TTL
TODO (Srish): Add real-time WebSocket or Server-Sent Events (SSE) for live updates
TODO (Srish): Add historical data support (date range filtering)
TODO (Mihir): Add performance metrics (query execution time, cache hit rate)
TODO (Srish): Add pagination support for large watchlists
TODO (Srish): Add portfolio aggregation (summed sentiment, average prediction)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.models.schemas import DashboardSummary
from app.services.sentiment_service import SentimentService
from app.services.prediction_service import PredictionService
from app.services.data_service import DataService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary/{ticker}", response_model=DashboardSummary)
async def get_dashboard_summary(ticker: str):
    """
    Get comprehensive dashboard summary for a stock.

    Combines sentiment, market data, and predictions into a single response
    for efficient frontend rendering.

    Args:
        ticker (str): Stock ticker symbol

    Returns:
        DashboardSummary: Aggregated sentiment, market data, and prediction

    Raises:
        HTTPException: 404 if ticker not found

    Example:
        GET /dashboard/summary/NVDA
        Response: {
            "ticker": "NVDA",
            "date": "2026-04-01",
            "sentiment": {...},
            "market_data": {...},
            "prediction": {...}
        }

    TODO (Mihir): Add @lru_cache to cache dashboard data (1 hour TTL)
    TODO (Srish): Add optional date parameter to get historical dashboard
    TODO (Srish): Add sentiment_source_breakdown (Reddit vs News sentiment)
    TODO (Srish): Add technical indicators (RSI, MACD, Bollinger Bands)
    TODO (Srish): Add analyst ratings (if available from market data API)
    """
    if not ticker or len(ticker) < 1:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        ticker = ticker.upper()

        # Get data from all services
        sentiment_data = SentimentService.get_sentiment_for_ticker(ticker)
        market_data = DataService.get_market_data(ticker)
        prediction_data = PredictionService.predict_for_ticker(ticker)

        # Combine into dashboard summary
        return DashboardSummary(
            ticker=ticker,
            date=market_data.date,
            sentiment=sentiment_data["overall_sentiment"],
            market_data=market_data,
            prediction=prediction_data["prediction"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dashboard data: {str(e)}"
        )


@router.get("/summary-batch")
async def get_dashboard_summary_batch(tickers: List[str] = Query(...)):
    """
    Get dashboard summaries for multiple stocks.

    Args:
        tickers (List[str]): List of stock ticker symbols

    Returns:
        List[DashboardSummary]: Dashboard data for each ticker

    Example:
        GET /dashboard/summary-batch?tickers=NVDA&tickers=TSLA
        Response: [
            {"ticker": "NVDA", "sentiment": {...}, ...},
            {"ticker": "TSLA", "sentiment": {...}, ...}
        ]

    TODO (Mihir): Add @lru_cache with composite cache key of all tickers
    TODO (Srish): Implement Server-Sent Events (SSE) for live dashboard updates
    TODO (Srish): Add portfolio-level aggregation (average sentiment, combined signals)
    TODO (Mihir): Implement streaming response for large ticker lists
    TODO (Srish): Add sorting/filtering options (by sentiment, prediction, risk)
    TODO (Srish): Add portfolio performance tracking (aggregate P&L)
    """
    if not tickers or len(tickers) == 0:
        raise HTTPException(status_code=400, detail="At least one ticker required")
    
    if len(tickers) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 tickers per request"
        )

    try:
        summaries = []
        for ticker in tickers:
            data = await get_dashboard_summary(ticker.upper())
            summaries.append(data)
        return summaries
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dashboard summaries: {str(e)}"
        )

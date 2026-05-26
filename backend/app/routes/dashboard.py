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

from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    ComponentAvailability,
    DashboardAvailability,
    DashboardSummary,
)
from app.services.sentiment_service import SentimentService
from app.services.prediction_service import PredictionService
from app.services.data_service import DataService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


def _availability(
    available: bool,
    status: str,
    source: str,
    message: str,
    count: int = None,
) -> ComponentAvailability:
    return ComponentAvailability(
        available=available,
        status=status,
        source=source,
        message=message,
        count=count,
    )


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
            "sentiment": {...},
            "market_data": {...},
            "prediction": {...},
            "updated_at": "2026-04-01T10:30:00"
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
        headlines = DataService.get_headlines(ticker)
        headline_status = DataService.get_headlines_status(ticker)
        social_posts = DataService.get_social_posts(ticker)
        fundamentals = DataService.get_fundamentals(ticker)
        fundamentals_status = DataService.get_fundamentals_status(ticker)

        for headline in headlines:
            try:
                headline.sentiment = SentimentService.get_sentiment_for_text(headline.headline)
            except Exception:
                headline.sentiment = None
        for post in social_posts:
            try:
                post.sentiment = SentimentService.get_sentiment_for_text(post.text)
            except Exception:
                post.sentiment = None

        overall_sentiment = sentiment_data["overall_sentiment"]
        prediction = prediction_data["prediction"]

        availability = DashboardAvailability(
            sentiment=_availability(
                available=overall_sentiment is not None,
                status="ready" if overall_sentiment is not None else "unavailable",
                source="Sentiment model",
                message=(
                    "Sentiment is available."
                    if overall_sentiment is not None
                    else "Aggregate sentiment unavailable until enough validated text data is available."
                ),
            ),
            market_data=_availability(
                available=market_data.price > 0,
                status="ready" if market_data.price > 0 else "fallback",
                source="Market data provider",
                message=(
                    "Market data is available."
                    if market_data.price > 0
                    else "Market data is not available for this ticker."
                ),
            ),
            prediction=_availability(
                available=prediction is not None,
                status="ready" if prediction is not None else "unavailable",
                source="Prediction model",
                message=(
                    "Prediction is available."
                    if prediction is not None
                    else "Prediction unavailable until enough validated input data is available."
                ),
            ),
            headlines=_availability(
                available=headline_status.get("available", len(headlines) > 0),
                status=headline_status.get("status", "ready" if headlines else "unavailable"),
                source=headline_status.get("source", "Yahoo Finance via yfinance"),
                message=headline_status.get(
                    "message",
                    (
                        f"{len(headlines)} headline items are available."
                        if headlines
                        else "Headline provider is unavailable or returned no articles."
                    ),
                ),
                count=headline_status.get("count", len(headlines)),
            ),
            social_posts=_availability(
                available=len(social_posts) > 0,
                status="ready" if social_posts else "unavailable",
                source="Social posts",
                message=(
                    f"{len(social_posts)} social posts are available."
                    if social_posts
                    else "No social posts are available for this ticker yet."
                ),
                count=len(social_posts),
            ),
            fundamentals=_availability(
                available=fundamentals_status.get("available", fundamentals is not None),
                status=fundamentals_status.get(
                    "status", "ready" if fundamentals is not None else "unavailable"
                ),
                source=fundamentals_status.get("source", "Yahoo Finance via yfinance"),
                message=fundamentals_status.get(
                    "message",
                    (
                        "Company fundamentals are available."
                        if fundamentals is not None
                        else "Fundamentals provider is unavailable or returned no usable fields."
                    ),
                ),
            ),
        )

        # Combine into dashboard summary
        return DashboardSummary(
            ticker=ticker,
            sentiment=overall_sentiment,
            market_data=market_data,
            prediction=prediction,
            headlines=headlines,
            social_posts=social_posts,
            fundamentals=fundamentals,
            availability=availability,
            status=availability.model_dump(),
            updated_at=datetime.now(),
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

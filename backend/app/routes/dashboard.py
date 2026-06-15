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
import re
from typing import List

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    ComponentAvailability,
    DashboardAvailability,
    DashboardSummary,
)
from app.services.sentiment_service import SentimentService
from app.services.prediction_service import PredictionService, TickerNotFoundError
from app.services.data_service import DataService

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
VALID_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _prediction_availability_status(prediction) -> str:
    if prediction is None:
        return "unavailable"
    if not (prediction.model_info or {}).get("real_training_data"):
        return "experimental"
    return "ready"


def _prediction_availability_message(prediction) -> str:
    if prediction is None:
        return "Signal unavailable until enough validated input data is available."
    if not (prediction.model_info or {}).get("real_training_data"):
        return (
            "Experimental signal only — trained on synthetic data, "
            "not validated against real historical outcomes."
        )
    return "Experimental signal available (validated model)."


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


def _social_posts_availability(social_posts) -> ComponentAvailability:
    if not social_posts:
        return _availability(
            available=False,
            status="unavailable",
            source="Social posts",
            message="No social posts are available for this ticker yet.",
            count=0,
        )

    social_count = sum(
        1 for post in social_posts if getattr(post, "content_type", None) == "social_post"
    )
    publisher_count = len(social_posts) - social_count

    if social_count == 0 and publisher_count > 0:
        return _availability(
            available=True,
            status="cached",
            source="Publisher headlines",
            message=(
                f"{publisher_count} publisher headline items are available. "
                "These are not social chatter."
            ),
            count=publisher_count,
        )

    if publisher_count > 0:
        return _availability(
            available=True,
            status="partial",
            source="Social posts and publisher headlines",
            message=(
                f"{social_count} social posts and {publisher_count} publisher headline items are available."
            ),
            count=len(social_posts),
        )

    return _availability(
        available=True,
        status="ready",
        source="Social posts",
        message=f"{social_count} social posts are available.",
        count=social_count,
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
    ticker = (ticker or "").strip().upper()
    if not ticker or not VALID_TICKER_RE.match(ticker):
        raise HTTPException(
            status_code=400,
            detail="Enter a valid ticker symbol using 1-5 letters.",
        )

    try:
        # Get data from all services
        sentiment_data = SentimentService.get_sentiment_for_ticker(ticker)
        market_data = DataService.get_market_data(ticker)
        market_history = DataService.get_market_history(ticker)
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
        market_status = market_data.status or ("ready" if market_data.price > 0 else "unavailable")
        market_source = market_data.source or "source not reported"
        market_available = market_data.price > 0 and market_status != "unavailable"
        market_message = (
            f"Market snapshot is available from {market_source}."
            if market_available
            else "Market snapshot unavailable from configured providers."
        )
        if market_available and market_data.volume_delta is not None:
            market_message = f"{market_message} Volume delta {market_data.volume_delta:+.1%}."

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
                available=market_available,
                status=market_status,
                source=market_source,
                message=market_message,
            ),
            prediction=_availability(
                available=False,
                status=_prediction_availability_status(prediction),
                source="Experimental signal",
                message=_prediction_availability_message(prediction),
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
            social_posts=_social_posts_availability(social_posts),
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
            market_history=market_history,
            prediction=prediction,
            headlines=headlines,
            social_posts=social_posts,
            fundamentals=fundamentals,
            availability=availability,
            status=availability.model_dump(),
            updated_at=datetime.now(),
        )
    except HTTPException:
        raise
    except TickerNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"No dashboard data found for {ticker}. Try NVDA or TSLA for the local demo.",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dashboard summaries: {str(e)}"
        )

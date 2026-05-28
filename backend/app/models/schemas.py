"""
Pydantic schemas for API request/response validation.

These models define the contract between backend and frontend.
All endpoints return data matching these schemas.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional, Dict, List
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Status of the API")
    version: str = Field(..., description="API version")
    message: str = Field(..., description="Additional message")

    class Config:
        example = {
            "status": "ok",
            "version": "0.1.0",
            "message": "API is running and ready to serve requests",
        }


class SentimentScores(BaseModel):
    """Sentiment analysis scores for a text or ticker."""

    positive_prob: float = Field(..., ge=0, le=1, description="Positive sentiment probability")
    negative_prob: float = Field(..., ge=0, le=1, description="Negative sentiment probability")
    neutral_prob: float = Field(..., ge=0, le=1, description="Neutral sentiment probability")
    sentiment_score: float = Field(
        ..., ge=-1, le=1, description="Computed sentiment score (positive_prob - negative_prob)"
    )
    sentiment_label: str = Field(..., description="Label: 'positive', 'negative', or 'neutral'")
    sentiment_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score (highest probability among positive/negative/neutral)"
    )

class MarketData(BaseModel):
    """Market data for a stock."""

    symbol: str = Field(..., description="Stock ticker symbol")
    price: float = Field(..., description="Current stock price")
    day_high: float = Field(..., description="Day high price")
    volume: int = Field(..., description="Trading volume")
    timestamp: datetime = Field(..., description="Data timestamp")

    class Config:
        example = {
            "symbol": "NVDA",
            "price": 875.50,
            "day_high": 885.00,
            "volume": 45000000,
            "timestamp": "2026-04-01T10:30:00",
        }


class MarketHistoryPoint(BaseModel):
    """Historical market point for compact dashboard charts."""

    date: str = Field(..., description="Trading date")
    close: float = Field(..., description="Close price")
    volume: Optional[int] = Field(None, description="Trading volume when available")


class PredictionResponse(BaseModel):
    """Stock movement prediction response."""

    model_config = ConfigDict(protected_namespaces=())

    symbol: str = Field(..., description="Stock ticker symbol")
    predicted_movement: str = Field(
        ..., description="Predicted movement: 'up', 'down', or 'neutral'"
    )
    probability: float = Field(
        ..., ge=0, le=1, description="Confidence score for prediction"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence score"
    )
    model_info: Optional[Dict[str, Any]] = Field(
        None, description="Optional model provenance for the serving predictor"
    )

class TextAnalysisRequest(BaseModel):
    """Request model for text sentiment analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze for sentiment")

    class Config:
        example = {
            "text": "NVDA earnings beat expectations by 15% this quarter",
        }


class HeadlineItem(BaseModel):
    """Normalized news/headline item for Market Pulse."""

    id: str = Field(..., description="Stable headline identifier")
    ticker: str = Field(..., description="Stock ticker symbol")
    headline: str = Field(..., description="Headline text for frontend display")
    title: str = Field(..., description="Alias for headline text")
    source: str = Field(..., description="Publisher or source name")
    url: Optional[str] = Field(None, description="Canonical article URL")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    time: Optional[str] = Field(None, description="Display-friendly publication date")
    summary: Optional[str] = Field(None, description="Short article summary when available")
    sentiment: Optional[SentimentScores] = Field(
        None, description="Optional sentiment score for the headline text"
    )


class SocialPostItem(BaseModel):
    """Normalized social or pipeline post item for Market Pulse."""

    id: Optional[str] = Field(None, description="Stable post identifier when available")
    ticker: str = Field(..., description="Stock ticker symbol")
    text: str = Field(..., description="Post text for frontend display")
    source: str = Field(..., description="Post source name")
    date: Optional[str] = Field(None, description="Post date when available")
    post_score: Optional[float] = Field(None, description="Source engagement score when available")
    sentiment: Optional[SentimentScores] = Field(
        None, description="Optional sentiment score for the post text"
    )


class Fundamentals(BaseModel):
    """Company fundamentals and metadata for Financials & Ratios."""

    source: str = Field(..., description="Provider used for fundamentals")
    company_name: Optional[str] = Field(None, description="Company display name")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    trailing_pe: Optional[float] = Field(None, description="Trailing P/E ratio")
    forward_pe: Optional[float] = Field(None, description="Forward P/E ratio")
    price_to_book: Optional[float] = Field(None, description="Price/book ratio")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    beta: Optional[float] = Field(None, description="Beta")
    eps: Optional[float] = Field(None, description="Trailing EPS")
    revenue: Optional[float] = Field(None, description="Total revenue")
    net_income: Optional[float] = Field(None, description="Net income")
    operating_cash_flow: Optional[float] = Field(None, description="Operating cash flow")
    debt_to_equity: Optional[float] = Field(None, description="Debt/equity ratio")
    currency: Optional[str] = Field(None, description="Currency for monetary values")


class ComponentAvailability(BaseModel):
    """Availability details for a dashboard data component."""

    available: bool = Field(..., description="Whether useful data is present")
    status: str = Field(..., description="ready, partial, unavailable, or fallback")
    source: Optional[str] = Field(None, description="Provider or subsystem name")
    message: Optional[str] = Field(None, description="Human-readable availability note")
    count: Optional[int] = Field(None, description="Number of items available, if relevant")


class DashboardAvailability(BaseModel):
    """Availability map for dashboard sections."""

    sentiment: ComponentAvailability
    market_data: ComponentAvailability
    prediction: ComponentAvailability
    headlines: ComponentAvailability
    social_posts: Optional[ComponentAvailability] = None
    fundamentals: ComponentAvailability


class DashboardSummary(BaseModel):
    """Summary data for dashboard display."""

    ticker: str = Field(..., description="Stock ticker symbol")
    sentiment: Optional[SentimentScores] = Field(None, description="Current sentiment scores")
    market_data: MarketData = Field(..., description="Current market data")
    market_history: List[MarketHistoryPoint] = Field(
        default_factory=list, description="Recent historical close prices"
    )
    prediction: Optional[PredictionResponse] = Field(None, description="Stock movement prediction")
    headlines: List[HeadlineItem] = Field(
        default_factory=list, description="Normalized headline items for Market Pulse"
    )
    social_posts: List[SocialPostItem] = Field(
        default_factory=list, description="Normalized social or pipeline posts for Market Pulse"
    )
    fundamentals: Optional[Fundamentals] = Field(
        None, description="Company fundamentals and metadata when available"
    )
    availability: Optional[DashboardAvailability] = Field(
        None, description="Per-section availability details"
    )
    status: Dict[str, Any] = Field(
        default_factory=dict, description="Backend-friendly status alias for availability"
    )
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        example = {
            "ticker": "NVDA",
            "sentiment": None,
            "market_data": {
                "symbol": "NVDA",
                "price": 875.50,
                "day_high": 885.00,
                "volume": 45000000,
                "timestamp": "2026-04-01T10:30:00",
            },
            "market_history": [],
            "prediction": None,
            "headlines": [],
            "fundamentals": None,
            "availability": None,
            "status": {},
            "updated_at": "2026-04-01T10:30:00",
        }

"""
Pydantic schemas for API request/response validation.

These models define the contract between backend and frontend.
All endpoints return data matching these schemas.
"""

from typing import Optional

from pydantic import BaseModel, Field


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

    class Config:
        example = {
            "positive_prob": 0.75,
            "negative_prob": 0.15,
            "neutral_prob": 0.10,
            "sentiment_score": 0.60,
            "sentiment_label": "positive",
            "sentiment_confidence": 0.75,
        }


class MarketData(BaseModel):
    """Market data for a stock."""

    ticker: str = Field(..., description="Stock ticker symbol")
    price: float = Field(..., description="Current stock price")
    day_high: float = Field(..., description="Day high price")
    volume: int = Field(..., description="Trading volume")
    date: str = Field(..., description="ISO date string for the market data snapshot")

    class Config:
        example = {
            "ticker": "NVDA",
            "price": 875.50,
            "day_high": 885.00,
            "volume": 45000000,
            "date": "2026-04-01",
        }


class PredictionResponse(BaseModel):
    """Stock movement prediction response."""

    ticker: str = Field(..., description="Stock ticker symbol")
    date: str = Field(..., description="ISO date string for the prediction")
    label: str = Field(..., description="Predicted movement label: 'up', 'down', or 'neutral'")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")

    class Config:
        example = {
            "ticker": "NVDA",
            "date": "2026-04-01",
            "label": "up",
            "confidence": 0.85,
        }


class TextAnalysisRequest(BaseModel):
    """Request model for text sentiment analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze for sentiment")

    class Config:
        example = {
            "text": "NVDA earnings beat expectations by 15% this quarter",
        }


class HeadlineItem(BaseModel):
    """Headline/news item for dashboard market pulse."""

    id: str = Field(..., description="Stable identifier for the headline item")
    ticker: str = Field(..., description="Ticker associated with the headline")
    headline: str = Field(..., description="Headline text")
    source: str = Field(..., description="Headline source or publisher")
    url: Optional[str] = Field(None, description="Source URL for the headline")
    published_at: Optional[str] = Field(None, description="ISO timestamp/date when available")
    sentiment_label: Optional[str] = Field(None, description="Optional sentiment label for the headline")
    sentiment_score: Optional[float] = Field(None, description="Optional sentiment score for the headline")

    class Config:
        example = {
            "id": "NVDA-0",
            "ticker": "NVDA",
            "headline": "Nvidia expands enterprise AI partnerships",
            "source": "Reuters",
            "url": "https://example.com/article",
            "published_at": "2026-04-23T10:00:00Z",
            "sentiment_label": "positive",
            "sentiment_score": 0.42,
        }


class AvailabilityStatus(BaseModel):
    """Availability metadata for one dashboard section."""

    available: bool = Field(..., description="Whether the section has usable data")
    source: str = Field(..., description="Primary source used for the section")
    item_count: Optional[int] = Field(None, description="Number of records contributing to the section")
    detail: Optional[str] = Field(None, description="Short explanatory detail")

    class Config:
        example = {
            "available": True,
            "source": "pipeline_posts",
            "item_count": 5,
            "detail": "Generated from 5 pipeline headline items",
        }


class DashboardAvailability(BaseModel):
    """Availability metadata for dashboard sections."""

    sentiment: AvailabilityStatus
    prediction: AvailabilityStatus
    headlines: AvailabilityStatus
    fundamentals: AvailabilityStatus


class FinancialSnapshot(BaseModel):
    """Financial statement values for one reporting period."""

    revenue: Optional[float] = Field(None, description="Revenue value for the period")
    net_income: Optional[float] = Field(None, description="Net income value for the period")
    operating_cash_flow: Optional[float] = Field(None, description="Operating cash flow for the period")
    eps: Optional[float] = Field(None, description="Earnings per share for the period")


class FundamentalRatios(BaseModel):
    """Key financial ratios for the ticker."""

    pe: Optional[float] = Field(None, description="Price-to-earnings ratio")
    eps: Optional[float] = Field(None, description="Trailing EPS")
    roe: Optional[float] = Field(None, description="Return on equity")
    debt_to_equity: Optional[float] = Field(None, description="Debt-to-equity ratio")
    revenue_growth_yoy: Optional[float] = Field(None, description="Revenue growth year-over-year")
    gross_margin: Optional[float] = Field(None, description="Gross margin ratio")


class FundamentalsData(BaseModel):
    """Company fundamentals and metadata for dashboard display."""

    ticker: str = Field(..., description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Company name")
    exchange: Optional[str] = Field(None, description="Primary exchange")
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    currency: Optional[str] = Field(None, description="Reporting currency")
    annual: FinancialSnapshot = Field(default_factory=FinancialSnapshot, description="Annual financial values")
    quarterly: FinancialSnapshot = Field(default_factory=FinancialSnapshot, description="Quarterly financial values")
    ratios: FundamentalRatios = Field(default_factory=FundamentalRatios, description="Key financial ratios")
    source: str = Field(..., description="Source used to populate fundamentals")

    class Config:
        example = {
            "ticker": "NVDA",
            "company_name": "NVIDIA Corporation",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Semiconductors",
            "market_cap": 2200000000000,
            "currency": "USD",
            "annual": {
                "revenue": 60922000000,
                "net_income": 29760000000,
                "operating_cash_flow": 28090000000,
                "eps": 12.05,
            },
            "quarterly": {
                "revenue": 26044000000,
                "net_income": 14881000000,
                "operating_cash_flow": 15000000000,
                "eps": 5.98,
            },
            "ratios": {
                "pe": 68.1,
                "eps": 12.05,
                "roe": 0.89,
                "debt_to_equity": 0.22,
                "revenue_growth_yoy": 1.26,
                "gross_margin": 0.76,
            },
            "source": "yfinance",
        }


class DashboardSummary(BaseModel):
    """Summary data for dashboard display."""

    ticker: str = Field(..., description="Stock ticker symbol")
    date: str = Field(..., description="ISO date string for the dashboard snapshot")
    sentiment: SentimentScores = Field(..., description="Current sentiment scores")
    market_data: MarketData = Field(..., description="Current market data")
    prediction: PredictionResponse = Field(..., description="Stock movement prediction")
    headlines: list[HeadlineItem] = Field(default_factory=list, description="Headline items for Market Pulse")
    availability: DashboardAvailability = Field(..., description="Availability metadata for dashboard sections")
    fundamentals: Optional[FundamentalsData] = Field(None, description="Company fundamentals when available")

    class Config:
        example = {
            "ticker": "NVDA",
            "date": "2026-04-01",
            "sentiment": {
                "positive_prob": 0.75,
                "negative_prob": 0.15,
                "neutral_prob": 0.10,
                "sentiment_score": 0.60,
                "sentiment_label": "positive",
                "sentiment_confidence": 0.75,
            },
            "market_data": {
                "ticker": "NVDA",
                "price": 875.50,
                "day_high": 885.00,
                "volume": 45000000,
                "date": "2026-04-01",
            },
            "prediction": {
                "ticker": "NVDA",
                "date": "2026-04-01",
                "label": "up",
                "confidence": 0.85,
            },
            "headlines": [
                {
                    "id": "NVDA-0",
                    "ticker": "NVDA",
                    "headline": "Nvidia expands enterprise AI partnerships",
                    "source": "Reuters",
                    "url": "https://example.com/article",
                    "published_at": "2026-04-01T10:00:00Z",
                    "sentiment_label": "positive",
                    "sentiment_score": 0.42,
                }
            ],
            "availability": {
                "sentiment": {"available": True, "source": "pipeline_posts", "item_count": 8, "detail": "Scored 8 text items"},
                "prediction": {"available": True, "source": "prediction_model", "detail": "Generated from current market and sentiment inputs"},
                "headlines": {"available": True, "source": "pipeline_posts", "item_count": 5, "detail": "5 recent headline items"},
                "fundamentals": {"available": True, "source": "yfinance", "detail": "Company profile and financial ratios loaded"},
            },
            "fundamentals": {
                "ticker": "NVDA",
                "company_name": "NVIDIA Corporation",
                "exchange": "NASDAQ",
                "sector": "Technology",
                "industry": "Semiconductors",
                "market_cap": 2200000000000,
                "currency": "USD",
                "annual": {"revenue": 60922000000, "net_income": 29760000000, "operating_cash_flow": 28090000000, "eps": 12.05},
                "quarterly": {"revenue": 26044000000, "net_income": 14881000000, "operating_cash_flow": 15000000000, "eps": 5.98},
                "ratios": {"pe": 68.1, "eps": 12.05, "roe": 0.89, "debt_to_equity": 0.22, "revenue_growth_yoy": 1.26, "gross_margin": 0.76},
                "source": "yfinance",
            },
        }

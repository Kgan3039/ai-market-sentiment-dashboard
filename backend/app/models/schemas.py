"""
Pydantic schemas for API request/response validation.

These models define the contract between backend and frontend.
All endpoints return data matching these schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List


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
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence score"
    )

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

class DashboardSummary(BaseModel):
    """Summary data for dashboard display."""

    ticker: str = Field(..., description="Stock ticker symbol")
    date: str = Field(..., description="ISO date string for the dashboard snapshot")
    sentiment: SentimentScores = Field(..., description="Current sentiment scores")
    market_data: MarketData = Field(..., description="Current market data")
    prediction: PredictionResponse = Field(..., description="Stock movement prediction")

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
        }

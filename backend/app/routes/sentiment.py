"""
Sentiment Analysis Routes

Endpoints for retrieving sentiment analysis for stocks and social media data.

Author: Mihir (with data from Matthew NLP module)
Status: Placeholder implementation - awaiting NLP integration

Endpoints:
- GET /sentiment/{ticker} - Get aggregated sentiment for a stock
- POST /sentiment/analyze-text - Analyze sentiment of arbitrary text

TODO (Mihir + Matthew): Integrate with real FinBERT sentiment analysis
TODO (Mihir): Add caching layer for sentiment scores
TODO (Srish): Update frontend to display sentiment with visual indicators
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import SentimentScores, TextAnalysisRequest
from app.services.sentiment_service import SentimentService

router = APIRouter(prefix="/sentiment", tags=["Sentiment"])


@router.get("/{ticker}", response_model=SentimentScores)
async def get_sentiment(ticker: str):
    """
    Get aggregated sentiment scores for a stock ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'NVDA', 'TSLA')

    Returns:
        SentimentScores: Sentiment analysis with probabilities

    Raises:
        HTTPException: 404 if ticker data not found

    Example:
        GET /sentiment/NVDA
        Response: {
            "positive_prob": 0.75,
            "negative_prob": 0.15,
            "neutral_prob": 0.10,
            "sentiment_score": 0.60,
            "sentiment_label": "positive",
            "sentiment_confidence": 0.75
        }

    TODO (Mihir): Add @cache decorator to cache sentiment for 1 hour
    TODO (Matthew + Mihir): Once NLP is integrated, sentiment will update daily
    TODO (Srish): Add optional date parameter to get historical sentiment
    TODO (Srish): Add option to filter by source (reddit, news, twitter)
    """
    if not ticker or len(ticker) < 1:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        result = SentimentService.get_sentiment_for_ticker(ticker.upper())
        if result["overall_sentiment"] is None:
            raise HTTPException(
                status_code=404,
                detail="Aggregate sentiment unavailable until enough validated text data is available.",
            )
        return result["overall_sentiment"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sentiment data: {str(e)}"
        )


@router.post("/analyze-text", response_model=SentimentScores)
async def analyze_text(request: TextAnalysisRequest):
    """
    Analyze sentiment of arbitrary text.

    Args:
        request (TextAnalysisRequest): Request containing text to analyze

    Returns:
        SentimentScores: Sentiment analysis results

    Example:
        POST /sentiment/analyze-text
        Body: {"text": "NVDA earnings beat expectations!"}

        Response: {
            "positive": 0.85,
            "negative": 0.08,
            "neutral": 0.07,
            "sentiment_score": 0.77,
            "sentiment_label": "positive"
        }
    """
    try:
        return SentimentService.get_sentiment_for_text(request.text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing text: {str(e)}"
        )

"""
Stock Prediction Routes

Endpoints for retrieving stock movement predictions based on sentiment and market data.

Author: Mihir (with model from Abhi ML module)
Status: Placeholder implementation - awaiting ML model integration

Endpoints:
- GET /prediction/{ticker} - Get movement prediction for a stock

TODO (Mihir + Abhi): Integrate with trained RandomForest/LogisticRegression model
TODO (Abhi): Export trained model for backend to load
TODO (Srish): Update frontend to display predictions with confidence indicators
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(ticker: str):
    """
    Get stock movement prediction for a ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'NVDA', 'TSLA')

    Returns:
        PredictionResponse: Predicted movement label and confidence

    Raises:
        HTTPException: 404 if ticker data not found

    Example:
        GET /prediction/NVDA
        Response: {
            "ticker": "NVDA",
            "date": "2026-04-01",
            "label": "up",
            "confidence": 0.85
        }

    TODO (Mihir): Add @cache decorator to cache predictions for 1 hour
    TODO (Abhi + Mihir): Once models are integrated, predictions will auto-update
    TODO (Srish): Add optional date parameter to predict for different timeframes
    TODO (Abhi): Add model confidence intervals to response
    TODO (Abhi): Add feature attribution to explain which factors drive prediction
    """
    if not ticker or len(ticker) < 1:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        result = PredictionService.predict_for_ticker(ticker.upper())
        return result["prediction"]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating prediction: {str(e)}"
        )

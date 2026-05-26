"""
Stock Prediction Routes
...
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictionResponse
from app.services.prediction_service import PredictionService, TickerNotFoundError

router = APIRouter(prefix="/prediction", tags=["Prediction"])


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(ticker: str):
    """
    Get stock movement prediction for a ticker.
    ...
    """
    if not ticker or len(ticker) < 1:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        result = PredictionService.predict_for_ticker(ticker.upper())
        if result["prediction"] is None:
            raise HTTPException(
                status_code=404,
                detail="Prediction unavailable until enough validated input data is available.",
            )
        return result["prediction"]
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")

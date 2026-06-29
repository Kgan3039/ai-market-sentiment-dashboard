"""
Experimental signal routes.

The /prediction path is kept for API compatibility, but it only returns a
signal when the serving artifact is trained and evaluated on real outcomes.
Synthetic-only artifacts are intentionally withheld.
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import PredictionResponse
from app.services.prediction_service import PredictionService, TickerNotFoundError

router = APIRouter(prefix="/prediction", tags=["Experimental Signal"])


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(ticker: str):
    """
    Get a validated experimental signal for a ticker when available.
    """
    if not ticker or len(ticker) < 1:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    try:
        result = PredictionService.predict_for_ticker(ticker.upper())
        if result["prediction"] is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    result.get("model_info", {}).get("reason")
                    or "Experimental signal unavailable until a model is trained and evaluated on real historical outcomes."
                ),
            )
        return result["prediction"]
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating experimental signal: {str(e)}")

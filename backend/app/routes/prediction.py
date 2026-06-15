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
        prediction = result["prediction"]
        if prediction is None:
            raise HTTPException(
                status_code=404,
                detail="Signal unavailable until enough validated input data is available.",
            )
        if not (prediction.model_info or {}).get("real_training_data"):
            raise HTTPException(
                status_code=404,
                detail=(
                    "Experimental signal only — not available until the model is trained "
                    "and evaluated on real historical outcomes."
                ),
            )
        return prediction
    except TickerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")

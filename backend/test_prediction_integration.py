"""End-to-end integration test for the prediction path.
 
Verifies that PredictionService.predict_for_ticker() returns the correct
API contract keys and that the ML model path is not silently falling back
to the rule-based fallback.
 
Run from the backend/ directory:
    python test_prediction_integration.py
"""
 
import os
import sys
 
# Make sure backend/ and prediction/ are on the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prediction")))
 
from datetime import datetime

from app.models.schemas import HeadlineItem, MarketData
from app.services.data_service import DataService
from app.services.prediction_service import PredictionService
 
 
def test_predict_for_ticker_contract():
    original_fetch = DataService._fetch_yfinance_headlines
    original_market_data = DataService.get_market_data
    try:
        DataService._fetch_yfinance_headlines = staticmethod(
            lambda ticker, limit: [
                HeadlineItem(
                    id="nvda-provider-1",
                    ticker=ticker,
                    headline="Nvidia shares rise after strong AI demand beats expectations",
                    title="Nvidia shares rise after strong AI demand beats expectations",
                    source="Test News",
                    published_at=datetime(2026, 5, 18),
                )
            ]
        )
        DataService.get_market_data = staticmethod(
            lambda ticker: MarketData(
                symbol=ticker,
                price=100.0,
                day_high=105.0,
                volume=1_000_000,
                timestamp=datetime(2026, 5, 18),
            )
        )

        result = PredictionService.predict_for_ticker("NVDA")
        prediction = result["prediction"]
    finally:
        DataService._fetch_yfinance_headlines = original_fetch
        DataService.get_market_data = original_market_data
        DataService._PROVIDER_CACHE.clear()
        DataService._PROVIDER_STATUS.clear()
 
    assert hasattr(prediction, "predicted_movement"), "Missing field: predicted_movement"
    assert hasattr(prediction, "probability"), "Missing field: probability"
    assert hasattr(prediction, "confidence"), "Missing field: confidence"
    assert hasattr(prediction, "model_info"), "Missing field: model_info"
 
    assert prediction.predicted_movement in ("up", "down", "neutral"), \
        f"Unexpected predicted_movement value: {prediction.predicted_movement}"
    assert 0.0 <= prediction.probability <= 1.0, \
        f"probability out of range: {prediction.probability}"
    assert 0.0 <= prediction.confidence <= 1.0, \
        f"confidence out of range: {prediction.confidence}"
 
    assert prediction.predicted_movement != "neutral", \
        "ML model path is being bypassed — endpoint is falling back to rule-based. " \
        "Check that prediction module loads correctly and predict() is being called."
    assert prediction.model_info["name"] == "RandomForestClassifier"
    assert prediction.model_info["artifact_source"] in (
        "disk",
        "trained_and_persisted",
    )

    print(f"✓ predicted_movement : {prediction.predicted_movement}")
    print(f"✓ probability        : {prediction.probability}")
    print(f"✓ confidence         : {prediction.confidence}")
    print("predict_for_ticker('NVDA') end-to-end check passed.")
 
 
if __name__ == "__main__":
    test_predict_for_ticker_contract()

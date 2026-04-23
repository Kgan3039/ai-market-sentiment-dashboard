"""Prediction Service - backend integration layer for stock movement predictions."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

from app.models.schemas import PredictionResponse


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load_prediction_module():
    """Load the shared prediction module dynamically."""
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    try:
        from prediction import prediction as prediction_module
        return prediction_module
    except Exception:
        prediction_dir = os.path.join(repo_root, "prediction")
        if prediction_dir not in sys.path:
            sys.path.append(prediction_dir)
        try:
            import prediction as prediction_module
            return prediction_module
        except Exception:
            return None


class PredictionService:
    """Service for managing stock movement predictions."""

    @staticmethod
    def _normalize_prediction_payload(
        prediction: PredictionResponse | Dict[str, Any],
        ticker: str,
    ) -> PredictionResponse:
        """Return a canonical prediction object that matches the API schema."""
        if isinstance(prediction, PredictionResponse):
            return prediction

        if isinstance(prediction, dict):
            if {"ticker", "date", "label", "confidence"}.issubset(prediction):
                return PredictionResponse(**prediction)

            return PredictionResponse(
                ticker=ticker,
                date=prediction.get("date", datetime.now().date().isoformat()),
                label=prediction.get("label", prediction.get("direction", prediction.get("predicted_movement", "neutral"))),
                confidence=float(prediction.get("confidence", prediction.get("probability", 0.0))),
            )

        raise ValueError("Prediction payload must be a PredictionResponse or dict")

    @staticmethod
    def predict_movement(
        ticker: str,
        sentiment_score: float,
        market_features: Dict[str, float],
    ) -> Tuple[PredictionResponse, str]:
        """Generate a prediction using the shared ML module when available."""
        prediction_module = _load_prediction_module()

        if prediction_module is not None and hasattr(prediction_module, "predict"):
            try:
                result = prediction_module.predict(
                    sentiment_score=sentiment_score,
                    sentiment_confidence=market_features.get("sentiment_confidence", 0.75),
                    price_delta_24h=market_features.get("price_delta_24h", 0.0),
                    volume_delta=market_features.get("volume_delta", 0.0),
                    model="rf",
                )
                return (
                    PredictionService._normalize_prediction_payload(
                        {
                            "ticker": ticker,
                            "date": datetime.now().date().isoformat(),
                            "label": result.get("label", result.get("direction", result.get("predicted_movement", "neutral"))),
                            "confidence": result.get("confidence", result.get("probability", 0.0)),
                        },
                        ticker,
                    ),
                    "prediction_model",
                )
            except Exception:
                pass

        movement = "up" if sentiment_score > 0.2 else ("down" if sentiment_score < -0.2 else "neutral")
        confidence = 0.75 if movement == "up" else 0.65 if movement == "down" else 0.5
        return (
            PredictionResponse(
                ticker=ticker,
                date=datetime.now().date().isoformat(),
                label=movement,
                confidence=confidence,
            ),
            "fallback_rule",
        )

    @staticmethod
    def predict_for_ticker(ticker: str) -> Dict[str, Any]:
        """Generate full prediction metadata for a ticker."""
        from app.services.data_service import DataService
        from app.services.sentiment_service import SentimentService

        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        sentiment_info = SentimentService.get_sentiment_for_ticker(ticker)
        overall_sentiment = sentiment_info.get("overall_sentiment")

        if isinstance(overall_sentiment, dict):
            sentiment_score = float(overall_sentiment.get("sentiment_score", 0.0))
            sentiment_confidence = float(overall_sentiment.get("sentiment_confidence", 0.0))
        elif overall_sentiment is not None:
            sentiment_score = getattr(overall_sentiment, "sentiment_score", 0.0)
            sentiment_confidence = getattr(overall_sentiment, "sentiment_confidence", 0.0)
        else:
            sentiment_score = 0.0
            sentiment_confidence = 0.0

        feature_snapshot = DataService.get_feature_snapshot(ticker)
        market_features = {
            "price_delta_24h": feature_snapshot.get("price_delta_24h", 0.0),
            "volume_delta": feature_snapshot.get("volume_delta", 0.0),
            "sentiment_confidence": sentiment_confidence,
        }

        prediction_payload, source = PredictionService.predict_movement(
            ticker=ticker,
            sentiment_score=sentiment_score,
            market_features=market_features,
        )

        return {
            "ticker": ticker,
            "date": prediction_payload.date,
            "prediction": prediction_payload,
            "source": source,
            "model_info": {
                "name": "RandomForestClassifier" if source == "prediction_model" else "FallbackRule",
                "version": "0.1.0",
                "features_used": [
                    "sentiment_score",
                    "sentiment_confidence",
                    "price_delta_24h",
                    "volume_delta",
                ],
            },
        }

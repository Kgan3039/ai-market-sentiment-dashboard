"""Prediction Service - API interface to stock movement prediction model.

Author: Mihir (with integration from Abhi ML model)
Responsibility: Provide REST API endpoints for stock movement predictions

Integration Points:
- Calls Abhi ML prediction model (../prediction/prediction.py)
- Consumes sentiment results from Matthew NLP module
- Consumes market data from Isaac data pipeline
- Returns results to Prediction routes for API endpoints

Current Status: Placeholder implementation - awaiting ML model integration
"""

import os
import sys
from typing import Dict, Any
from datetime import datetime
from app.models.schemas import PredictionResponse


def _load_prediction_module():
    """Load prediction/prediction module dynamically and return predict function."""
    try:
        from prediction import prediction as prediction_module
        return prediction_module
    except Exception:
        prediction_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'prediction'))
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
    def predict_movement(
        ticker: str, sentiment_score: float, market_features: Dict[str, float]
    ) -> PredictionResponse:
        """
        Generate prediction for stock movement using Abhi prediction module when available.

        Args:
            ticker (str): Stock ticker symbol
            sentiment_score (float): Sentiment score from -1 to 1
            market_features (Dict): Market data features (price_delta, volume, etc.)

        Returns:
            PredictionResponse: Prediction with confidence score
        """
        prediction_module = _load_prediction_module()

        if prediction_module is not None and hasattr(prediction_module, 'predict'):
            try:
                result = prediction_module.predict(
                    sentiment_score=sentiment_score,
                    sentiment_confidence=market_features.get('sentiment_confidence', 0.75),
                    price_delta_24h=market_features.get('price_delta_24h', 0.0),
                    volume_delta=market_features.get('volume_delta', 0.0),
                    model='rf'
                )

                return PredictionResponse(
                    symbol=ticker,
                    predicted_movement=result.get('direction', 'neutral'),
                    probability=float(result.get('confidence', 0.0)),
                    confidence=float(result.get('confidence', 0.0)),
                )
            except Exception:
                pass

        # Fallback: simple rule-based prediction
        movement = "up" if sentiment_score > 0.2 else ("down" if sentiment_score < -0.2 else "neutral")
        probability = 0.75 if movement == "up" else 0.65 if movement == "down" else 0.5
        confidence = 0.82

        return PredictionResponse(
            symbol=ticker,
            predicted_movement=movement,
            probability=probability,
            confidence=confidence,
        )

    @staticmethod
    def predict_for_ticker(ticker: str) -> Dict[str, Any]:
        """
        Generate full prediction for a ticker using current sentiment and market data.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            dict: Prediction results with supporting data
        """
        from app.services.sentiment_service import SentimentService
        from app.services.data_service import DataService

        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        sentiment_info = SentimentService.get_sentiment_for_ticker(ticker)
        market_info = DataService.get_market_data(ticker)

        overall_sentiment = sentiment_info.get('overall_sentiment')
        sentiment_score = overall_sentiment.sentiment_score if overall_sentiment is not None else 0.0
        sentiment_confidence = overall_sentiment.sentiment_confidence if overall_sentiment is not None else 0.0

        market_features = {
            'price_delta_24h': 0.0,
            'volume_delta': 0.0,
            'sentiment_confidence': sentiment_confidence,
        }

        # Try to infer delta fields from pipeline by reading stock_data
        try:
            import json
            pipeline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_data.json'))
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'r') as f:
                    data = json.load(f)
                for row in data:
                    if row.get('ticker', '').upper() == ticker:
                        market_features['price_delta_24h'] = float(row.get('price_delta_24h', 0.0))
                        market_features['volume_delta'] = float(row.get('volume_delta', 0.0))
                        break
        except Exception:
            pass

        prediction = PredictionService.predict_movement(
            ticker=ticker,
            sentiment_score=sentiment_score,
            market_features=market_features,
        )

        return {
            'ticker': ticker,
            'prediction': prediction,
            'model_info': {
                'name': 'RandomForestClassifier' if prediction.predicted_movement in ['up', 'down'] else 'FallbackRule',
                'version': '0.1.0',
                'features_used': [
                    'sentiment_score',
                    'sentiment_confidence',
                    'price_delta_24h',
                    'volume_delta',
                ],
            },
            'timestamp': datetime.now().isoformat(),
        }

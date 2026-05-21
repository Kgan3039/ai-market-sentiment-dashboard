"""Prediction Service - API interface to stock movement prediction model.

Author: Mihir (with integration from Abhi ML model)

Assumptions:
- sentiment_score: float in [-1, 1], derived as avg_positive_prob - avg_negative_prob
- sentiment_confidence: max(positive, negative, neutral) prob from NLP model
- price_delta_24h: (close - open) / open for last 24h; clipped to ±8% in predict()
- volume_delta: (today_vol - avg_vol) / avg_vol; clipped to ±50% in predict()
- ML model: RandomForestClassifier trained on synthetic data (see prediction.py)
- Confidence output is soft-clipped to [0.52, 0.84] in ML module for demo plausibility
"""

import os
import sys
from typing import Dict, Any
from datetime import datetime
from app.models.schemas import PredictionResponse
import re


class TickerNotFoundError(Exception):
    pass


def _load_prediction_module():
    """Load prediction/prediction module dynamically and return predict function."""
    try:
        from prediction import prediction as prediction_module
        return prediction_module
    except Exception:
        prediction_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "prediction")
        )
        if prediction_dir not in sys.path:
            sys.path.append(prediction_dir)
        try:
            import prediction as prediction_module
            return prediction_module
        except Exception:
            return None


class PredictionService:

    @staticmethod
    def predict_movement(
        ticker: str, sentiment_score: float, market_features: Dict[str, float]
    ) -> PredictionResponse:
        prediction_module = _load_prediction_module()
        used_ml_model = False

        if prediction_module is not None and hasattr(prediction_module, 'predict'):
            try:
                result = prediction_module.predict(
                    sentiment_score=sentiment_score,
                    sentiment_confidence=market_features.get('sentiment_confidence', 0.5),
                    price_delta_24h=market_features.get('price_delta_24h', 0.0),
                    volume_delta=market_features.get('volume_delta', 0.0),
                    model='rf'
                )
                used_ml_model = True
                return PredictionResponse(
                    symbol=ticker,
                    predicted_movement=result.get('predicted_movement', 'neutral'),
                    probability=float(result.get('probability', 0.5)),
                    confidence=float(result.get('confidence', 0.5)),
                ), used_ml_model
            except Exception:
                pass

        # Fallback: rule-based, clearly marked
        movement = "up" if sentiment_score > 0.2 else ("down" if sentiment_score < -0.2 else "neutral")
        # Derive soft probability from sentiment magnitude rather than hardcoding
        magnitude = min(abs(sentiment_score), 1.0)
        probability = round(0.5 + 0.2 * magnitude, 4)
        confidence = round(0.5 + 0.15 * magnitude, 4)

        return PredictionResponse(
            symbol=ticker,
            predicted_movement=movement,
            probability=probability,
            confidence=confidence,
        ), used_ml_model

    @staticmethod
    def predict_for_ticker(ticker: str) -> Dict[str, Any]:
        from app.services.sentiment_service import SentimentService
        from app.services.data_service import DataService

        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        VALID_TICKER = re.compile(r'^[A-Z]{1,5}$')
        if not VALID_TICKER_RE.match(ticker):
            raise TickerNotFoundError(f"Unknown ticker symbol: {ticker}")
        sentiment_info = SentimentService.get_sentiment_for_ticker(ticker)

        if sentiment_info is None:
            raise TickerNotFoundError(f"No sentiment data found for ticker: {ticker}")

        market_info = DataService.get_market_data(ticker)

        if market_info is None:
            raise TickerNotFoundError(f"No market data found for ticker: {ticker}")

        overall_sentiment = sentiment_info.get('overall_sentiment')
        sentiment_score = overall_sentiment.sentiment_score if overall_sentiment is not None else 0.0
        sentiment_confidence = overall_sentiment.sentiment_confidence if overall_sentiment is not None else 0.5

        market_features = {
            'price_delta_24h': 0.0,
            'volume_delta': 0.0,
            'sentiment_confidence': sentiment_confidence,
        }

        try:
            import json
            pipeline_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_data.json")
            )
            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'r') as f:
                    data = json.load(f)
                for row in data:
                    if row.get('ticker', '').upper() == ticker:
                        market_snapshot = row.get('market_data', {}) or {}
                        market_features['price_delta_24h'] = float(market_snapshot.get('price_delta_24h', 0.0) or 0.0)
                        market_features['volume_delta'] = 0.0
                        break
        except Exception:
            pass

        prediction, used_ml_model = PredictionService.predict_movement(
            ticker=ticker,
            sentiment_score=sentiment_score,
            market_features=market_features,
        )

        return {
            'ticker': ticker,
            'prediction': prediction,
            'model_info': {
                'name': 'RandomForestClassifier' if used_ml_model else 'FallbackRule',
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
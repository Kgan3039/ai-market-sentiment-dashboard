"""Experimental signal service for stock movement research.

Author: Mihir (with integration from Abhi ML model)

Assumptions:
- sentiment_score: float in [-1, 1], derived as avg_positive_prob - avg_negative_prob
- sentiment_confidence: max(positive, negative, neutral) prob from NLP model
- price_delta_24h: (close - open) / open for last 24h; clipped to ±8% in predict()
- volume_delta: (today_vol - avg_vol) / avg_vol; clipped to ±50% in predict()
- Synthetic model artifacts are not exposed as actionable prediction outputs.
- A real signal may be exposed only after training/evaluation on historical outcomes.
"""

import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from typing import Any, Dict, Tuple

from app.models.schemas import PredictionResponse
import re


VALID_TICKER_RE = re.compile(r'^[A-Z]{1,5}$')


class TickerNotFoundError(Exception):
    pass


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
    SYNTHETIC_UNAVAILABLE_REASON = (
        "Experimental signal unavailable: current artifacts are trained on synthetic data "
        "and have not been evaluated against real historical outcomes."
    )

    @staticmethod
    def _unavailable_model_info(reason: str, model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info = dict(model_info or {})
        info.update(
            {
                'status': 'unavailable',
                'reason': reason,
                'real_training_data': False,
                'exposes_actionable_output': False,
            }
        )
        info.setdefault('features_used', [])
        return info

    @staticmethod
    def prewarm_model_artifacts() -> Dict[str, Any]:
        """Load persisted experimental signal artifacts into memory before first request."""
        prediction_module = _load_prediction_module()

        if prediction_module is None:
            return {
                'status': 'unavailable',
                'reason': 'Experimental signal module could not be loaded.',
            }

        if hasattr(prediction_module, 'bootstrap_model_artifacts'):
            artifacts = prediction_module.bootstrap_model_artifacts(force_retrain=False)
        elif hasattr(prediction_module, 'get_model_artifacts'):
            artifacts = prediction_module.get_model_artifacts()
        else:
            return {
                'status': 'unavailable',
                'reason': 'Experimental signal module does not expose artifact loading.',
            }

        provenance = getattr(artifacts, 'provenance', {}) or {}
        return {
            'status': 'ready',
            'artifact_source': provenance.get('artifact_source'),
            'artifact_path': provenance.get('artifact_path'),
            'version': provenance.get('artifact_version'),
            'trained_at': provenance.get('trained_at'),
        }

    @staticmethod
    def _fractional_price_delta(*sources: Dict[str, Any]) -> float:
        for source in sources:
            if not source:
                continue
            percent_change = source.get('percent_change_24h')
            if percent_change is not None:
                try:
                    return float(percent_change) / 100
                except (TypeError, ValueError):
                    pass

        for source in sources:
            if not source:
                continue
            price_delta = source.get('price_delta_24h')
            if price_delta is not None:
                try:
                    return float(price_delta)
                except (TypeError, ValueError):
                    pass

        return 0.0

    @staticmethod
    def predict_movement(
        ticker: str, sentiment_score: float, market_features: Dict[str, float]
    ) -> tuple[Optional[PredictionResponse], Dict[str, Any]]:
        prediction_module = _load_prediction_module()

        if prediction_module is not None and hasattr(prediction_module, "predict"):
            try:
                model_key = 'rf'
                result = prediction_module.predict(
                    sentiment_score=sentiment_score,
                    sentiment_confidence=market_features.get('sentiment_confidence', 0.5),
                    price_delta_24h=market_features.get('price_delta_24h', 0.0),
                    volume_delta=market_features.get('volume_delta', 0.0),
                    model=model_key
                )
                if hasattr(prediction_module, 'get_model_provenance'):
                    model_info = prediction_module.get_model_provenance(model_key)
                else:
                    model_info = {
                        'name': 'RandomForestClassifier',
                        'version': '0.1.0',
                        'artifact_source': 'unknown',
                    }
                if model_info.get('real_training_data') is not True:
                    return None, PredictionService._unavailable_model_info(
                        PredictionService.SYNTHETIC_UNAVAILABLE_REASON,
                        model_info,
                    )

                model_info['status'] = 'ready'
                model_info['exposes_actionable_output'] = True

                return PredictionResponse(
                    symbol=ticker,
                    predicted_movement=result.get('predicted_movement', 'neutral'),
                    probability=float(result.get('probability', 0.5)),
                    confidence=float(result.get('confidence', 0.5)),
                    model_info=model_info,
                ), model_info
            except Exception:
                pass

        model_info = {
            'name': 'FallbackRule',
            'version': '0.1.0',
            'artifact_source': 'none',
        }

        return None, PredictionService._unavailable_model_info(
            "Experimental signal unavailable: no validated model artifact is available.",
            model_info,
        )

    @staticmethod
    def predict_for_ticker(ticker: str) -> Dict[str, Any]:
        from app.services.sentiment_service import SentimentService

        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        if not VALID_TICKER_RE.match(ticker):
            raise TickerNotFoundError(f"Unknown ticker symbol: {ticker}")
        sentiment_info = SentimentService.get_sentiment_for_ticker(ticker)

        if sentiment_info is None:
            raise TickerNotFoundError(f"No sentiment data found for ticker: {ticker}")

        market_info = DataService.get_market_data(ticker)
        ticker_records = DataService._get_ticker_records(ticker)

        if market_info is None or (market_info.price <= 0 and not ticker_records):
            raise TickerNotFoundError(f"No market data found for ticker: {ticker}")

        overall_sentiment = sentiment_info.get('overall_sentiment')
        if overall_sentiment is None:
            return {
                'ticker': ticker,
                'prediction': None,
                'model_info': {
                    'name': None,
                    'version': '0.1.0',
                    'status': 'unavailable',
                    'reason': 'Validated aggregate sentiment is unavailable.',
                    'real_training_data': False,
                    'exposes_actionable_output': False,
                    'features_used': [],
                },
                'timestamp': datetime.now().isoformat(),
            }

        sentiment_score = overall_sentiment.sentiment_score
        sentiment_confidence = overall_sentiment.sentiment_confidence
        market_info_features = market_info.model_dump()

        feature_snapshot = DataService.get_feature_snapshot(ticker)
        market_features = {
            'price_delta_24h': PredictionService._fractional_price_delta(market_info_features),
            'volume_delta': float(market_info.volume_delta or 0.0),
            'sentiment_confidence': sentiment_confidence,
        }

        for row in ticker_records:
            market_snapshot = row.get('market_data', {}) or {}
            market_features['price_delta_24h'] = PredictionService._fractional_price_delta(
                market_snapshot,
                row,
            )
            market_features['volume_delta'] = float(
                market_snapshot.get('volume_delta', row.get('volume_delta', 0.0)) or 0.0
            )
            break

        prediction, model_info = PredictionService.predict_movement(
            ticker=ticker,
            sentiment_score=sentiment_score,
            market_features=market_features,
        )

        model_info.setdefault('features_used', [
            'sentiment_score',
            'sentiment_confidence',
            'price_delta_24h',
            'volume_delta',
        ])

        return {
            'ticker': ticker,
            'prediction': prediction,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat(),
        }

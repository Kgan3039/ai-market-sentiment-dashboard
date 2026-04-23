import os

from fastapi.testclient import TestClient

os.environ["DEBUG"] = "false"

from main import app
from app.models.schemas import FundamentalsData, HeadlineItem, MarketData, PredictionResponse, SentimentScores
from app.services.data_service import DataService
from app.services.prediction_service import PredictionService
from app.services.sentiment_service import SentimentService


client = TestClient(app)


def _sentiment_payload() -> SentimentScores:
    return SentimentScores(
        positive_prob=0.72,
        negative_prob=0.10,
        neutral_prob=0.18,
        sentiment_score=0.62,
        sentiment_label="positive",
        sentiment_confidence=0.72,
    )


def _market_payload(ticker: str) -> MarketData:
    return MarketData(
        ticker=ticker,
        price=875.50,
        day_high=885.00,
        volume=45000000,
        date="2026-04-01",
    )


def _prediction_payload(ticker: str) -> PredictionResponse:
    return PredictionResponse(
        ticker=ticker,
        date="2026-04-01",
        label="up",
        confidence=0.78,
    )


def _headline_payload(ticker: str) -> HeadlineItem:
    return HeadlineItem(
        id=f"{ticker}-0",
        ticker=ticker,
        headline=f"{ticker} expands AI partnerships",
        source="Reuters",
        url="https://example.com/article",
        published_at="2026-04-01T10:00:00Z",
        sentiment_label="positive",
        sentiment_score=0.42,
    )


def _fundamentals_payload(ticker: str) -> FundamentalsData:
    return FundamentalsData(
        ticker=ticker,
        company_name=f"{ticker} Corporation",
        exchange="NASDAQ",
        sector="Technology",
        industry="Semiconductors",
        market_cap=2200000000000,
        currency="USD",
        source="yfinance",
    )


def test_sentiment_response_uses_dataset_field_names(monkeypatch):
    monkeypatch.setattr(
        SentimentService,
        "get_sentiment_for_ticker",
        staticmethod(
            lambda ticker: {
                "ticker": ticker,
                "date": "2026-04-01",
                "overall_sentiment": _sentiment_payload(),
                "source_breakdown": {},
            }
        ),
    )

    response = client.get("/sentiment/NVDA")

    assert response.status_code == 200
    assert set(response.json().keys()) == {
        "positive_prob",
        "negative_prob",
        "neutral_prob",
        "sentiment_score",
        "sentiment_label",
        "sentiment_confidence",
    }


def test_market_response_uses_ticker_and_date(monkeypatch):
    monkeypatch.setattr(
        DataService,
        "get_market_data",
        staticmethod(lambda ticker: _market_payload(ticker)),
    )

    response = client.get("/market/NVDA")
    payload = response.json()

    assert response.status_code == 200
    assert set(payload.keys()) == {"ticker", "price", "day_high", "volume", "date"}
    assert "symbol" not in payload
    assert "timestamp" not in payload


def test_market_batch_route_uses_batch_handler(monkeypatch):
    monkeypatch.setattr(
        DataService,
        "get_market_data_multiple",
        staticmethod(
            lambda tickers: {
                ticker: _market_payload(ticker)
                for ticker in tickers
            }
        ),
    )

    response = client.get("/market/batch?tickers=NVDA&tickers=TSLA")
    payload = response.json()

    assert response.status_code == 200
    assert isinstance(payload, list)
    assert [item["ticker"] for item in payload] == ["NVDA", "TSLA"]


def test_prediction_response_uses_label_contract(monkeypatch):
    monkeypatch.setattr(
        PredictionService,
        "predict_for_ticker",
        staticmethod(
            lambda ticker: {
                "ticker": ticker,
                "date": "2026-04-01",
                "prediction": _prediction_payload(ticker),
                "model_info": {},
            }
        ),
    )

    response = client.get("/prediction/NVDA")
    payload = response.json()

    assert response.status_code == 200
    assert set(payload.keys()) == {"ticker", "date", "label", "confidence"}
    assert "symbol" not in payload
    assert "predicted_movement" not in payload
    assert "probability" not in payload


def test_dashboard_summary_uses_canonical_names(monkeypatch):
    monkeypatch.setattr(
        SentimentService,
        "get_sentiment_for_ticker",
        staticmethod(
            lambda ticker: {
                "ticker": ticker,
                "date": "2026-04-01",
                "overall_sentiment": _sentiment_payload(),
                "source_breakdown": {},
            }
        ),
    )
    monkeypatch.setattr(
        DataService,
        "get_market_data",
        staticmethod(lambda ticker: _market_payload(ticker)),
    )
    monkeypatch.setattr(
        PredictionService,
        "predict_for_ticker",
        staticmethod(
            lambda ticker: {
                "ticker": ticker,
                "date": "2026-04-01",
                "prediction": _prediction_payload(ticker),
                "model_info": {},
            }
        ),
    )
    monkeypatch.setattr(
        DataService,
        "get_headlines_for_ticker",
        staticmethod(
            lambda ticker: {
                "ticker": ticker,
                "source": "pipeline",
                "headlines": [_headline_payload(ticker)],
            }
        ),
    )
    monkeypatch.setattr(
        DataService,
        "get_fundamentals_for_ticker",
        staticmethod(
            lambda ticker: {
                "ticker": ticker,
                "source": "yfinance",
                "fundamentals": _fundamentals_payload(ticker),
            }
        ),
    )

    response = client.get("/dashboard/summary/NVDA")
    payload = response.json()

    assert response.status_code == 200
    assert set(payload.keys()) == {
        "ticker",
        "date",
        "sentiment",
        "market_data",
        "prediction",
        "headlines",
        "availability",
        "fundamentals",
    }
    assert "updated_at" not in payload
    assert set(payload["market_data"].keys()) == {"ticker", "price", "day_high", "volume", "date"}
    assert set(payload["prediction"].keys()) == {"ticker", "date", "label", "confidence"}
    assert set(payload["headlines"][0].keys()) == {
        "id",
        "ticker",
        "headline",
        "source",
        "url",
        "published_at",
        "sentiment_label",
        "sentiment_score",
    }
    assert set(payload["availability"].keys()) == {
        "sentiment",
        "prediction",
        "headlines",
        "fundamentals",
    }
    assert payload["availability"]["headlines"]["available"] is True
    assert payload["fundamentals"]["ticker"] == "NVDA"

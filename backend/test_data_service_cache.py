#!/usr/bin/env python3
"""Regression checks for dashboard data provider caching."""

import asyncio
import os
import sys

from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.models.schemas import Fundamentals, HeadlineItem, MarketData, PredictionResponse, SentimentScores
from app.routes.dashboard import get_dashboard_summary
from app.services.data_service import DataService
from app.services.prediction_service import PredictionService
from app.services import sentiment_service
from app.services.sentiment_service import SentimentService


def _headline() -> HeadlineItem:
    return HeadlineItem(
        id="nvda-test-1",
        ticker="NVDA",
        headline="Nvidia shares rise after stronger data center demand",
        title="Nvidia shares rise after stronger data center demand",
        source="Test News",
        published_at=datetime(2026, 5, 18),
    )


def _fundamentals() -> Fundamentals:
    return Fundamentals(
        source="Test provider",
        company_name="NVIDIA Corporation",
        sector="Technology",
        market_cap=3_000_000_000_000,
        trailing_pe=50.0,
        currency="USD",
    )


def _sentiment() -> SentimentScores:
    return SentimentScores(
        positive_prob=0.72,
        negative_prob=0.08,
        neutral_prob=0.20,
        sentiment_score=0.64,
        sentiment_label="positive",
        sentiment_confidence=0.72,
    )


def test_headline_cache_falls_back_to_stale_data() -> None:
    DataService._PROVIDER_CACHE.clear()
    DataService._PROVIDER_STATUS.clear()

    original_fetch = DataService._fetch_yfinance_headlines
    original_ttl = DataService._CACHE_TTL_SECONDS
    try:
        DataService._fetch_yfinance_headlines = staticmethod(lambda ticker, limit: [_headline()])
        assert len(DataService.get_headlines("NVDA")) == 1
        assert DataService.get_headlines_status("NVDA")["status"] == "ready"

        DataService._CACHE_TTL_SECONDS = -1

        def fail_fetch(ticker: str, limit: int):
            raise RuntimeError("provider unavailable")

        DataService._fetch_yfinance_headlines = staticmethod(fail_fetch)
        stale_headlines = DataService.get_headlines("NVDA")
        status = DataService.get_headlines_status("NVDA")

        assert len(stale_headlines) == 1
        assert status["status"] == "fallback"
        assert status["available"] is True
        assert status["count"] == 1
    finally:
        DataService._fetch_yfinance_headlines = original_fetch
        DataService._CACHE_TTL_SECONDS = original_ttl
        DataService._PROVIDER_CACHE.clear()
        DataService._PROVIDER_STATUS.clear()


def test_fundamentals_cache_falls_back_to_stale_data() -> None:
    DataService._PROVIDER_CACHE.clear()
    DataService._PROVIDER_STATUS.clear()

    original_fetch = DataService._fetch_yfinance_fundamentals
    original_ttl = DataService._CACHE_TTL_SECONDS
    try:
        DataService._fetch_yfinance_fundamentals = staticmethod(lambda ticker: _fundamentals())
        assert DataService.get_fundamentals("NVDA") is not None
        assert DataService.get_fundamentals_status("NVDA")["status"] == "ready"

        DataService._CACHE_TTL_SECONDS = -1

        def fail_fetch(ticker: str):
            raise RuntimeError("provider unavailable")

        DataService._fetch_yfinance_fundamentals = staticmethod(fail_fetch)
        stale_fundamentals = DataService.get_fundamentals("NVDA")
        status = DataService.get_fundamentals_status("NVDA")

        assert stale_fundamentals is not None
        assert status["status"] == "fallback"
        assert status["available"] is True
    finally:
        DataService._fetch_yfinance_fundamentals = original_fetch
        DataService._CACHE_TTL_SECONDS = original_ttl
        DataService._PROVIDER_CACHE.clear()
        DataService._PROVIDER_STATUS.clear()


def test_dashboard_summary_uses_provider_status() -> None:
    DataService._PROVIDER_STATUS.clear()

    original_sentiment_for_ticker = SentimentService.get_sentiment_for_ticker
    original_sentiment_for_text = SentimentService.get_sentiment_for_text
    original_market_data = DataService.get_market_data
    original_market_history = DataService.get_market_history
    original_prediction = PredictionService.predict_for_ticker
    original_headlines = DataService.get_headlines
    original_headlines_status = DataService.get_headlines_status
    original_fundamentals = DataService.get_fundamentals
    original_fundamentals_status = DataService.get_fundamentals_status
    try:
        SentimentService.get_sentiment_for_ticker = staticmethod(
            lambda ticker: {"overall_sentiment": _sentiment()}
        )
        SentimentService.get_sentiment_for_text = staticmethod(lambda text: _sentiment())
        DataService.get_market_data = staticmethod(
            lambda ticker: MarketData(
                symbol=ticker,
                price=100.0,
                day_high=105.0,
                volume=1_000_000,
                price_delta_24h=2.5,
                percent_change_24h=2.56,
                volume_delta=0.2,
                source="Yahoo Finance via yfinance",
                status="ready",
                timestamp=datetime(2026, 5, 18),
            )
        )
        DataService.get_market_history = staticmethod(lambda ticker: [])
        PredictionService.predict_for_ticker = staticmethod(
            lambda ticker: {
                "prediction": PredictionResponse(
                    symbol=ticker,
                    predicted_movement="up",
                    probability=0.62,
                    confidence=0.68,
                ),
                "model_info": {"name": "Test model"},
            }
        )
        DataService.get_headlines = staticmethod(lambda ticker: [_headline()])
        DataService.get_headlines_status = staticmethod(
            lambda ticker: {
                "available": True,
                "status": "fallback",
                "source": "Yahoo Finance via yfinance (stale cache)",
                "message": "Headline provider failed; using 1 cached items.",
                "count": 1,
            }
        )
        DataService.get_fundamentals = staticmethod(lambda ticker: _fundamentals())
        DataService.get_fundamentals_status = staticmethod(
            lambda ticker: {
                "available": True,
                "status": "fallback",
                "source": "Yahoo Finance via yfinance (stale cache)",
                "message": "Fundamentals provider failed; using cached company fundamentals.",
                "count": None,
            }
        )

        summary = asyncio.run(get_dashboard_summary("NVDA"))

        assert summary.availability.headlines.status == "fallback"
        assert summary.availability.headlines.count == 1
        assert summary.availability.fundamentals.status == "fallback"
        assert summary.availability.market_data.status == "ready"
        assert summary.availability.market_data.source == "Yahoo Finance via yfinance"
        assert "Volume delta +20.0%" in summary.availability.market_data.message
        assert summary.status["headlines"]["status"] == "fallback"
        assert summary.headlines[0].sentiment is not None
    finally:
        SentimentService.get_sentiment_for_ticker = original_sentiment_for_ticker
        SentimentService.get_sentiment_for_text = original_sentiment_for_text
        DataService.get_market_data = original_market_data
        DataService.get_market_history = original_market_history
        PredictionService.predict_for_ticker = original_prediction
        DataService.get_headlines = original_headlines
        DataService.get_headlines_status = original_headlines_status
        DataService.get_fundamentals = original_fundamentals
        DataService.get_fundamentals_status = original_fundamentals_status
        DataService._PROVIDER_STATUS.clear()


def test_dashboard_summary_uses_headlines_for_demo_sentiment_and_prediction() -> None:
    DataService._PROVIDER_CACHE.clear()
    DataService._PROVIDER_STATUS.clear()

    original_fetch = DataService._fetch_yfinance_headlines
    original_market_data = DataService.get_market_data
    original_market_history = DataService.get_market_history
    original_fundamentals = DataService.get_fundamentals
    original_fundamentals_status = DataService.get_fundamentals_status
    original_pipeline_file_path = sentiment_service._pipeline_file_path
    try:
        sentiment_service._pipeline_file_path = lambda: os.path.join(
            os.path.dirname(__file__), "missing_stock_data.json"
        )
        DataService._fetch_yfinance_headlines = staticmethod(
            lambda ticker, limit: [
                HeadlineItem(
                    id=f"{ticker}-real-1",
                    ticker=ticker,
                    headline="Nvidia shares rise as data center demand beats expectations",
                    title="Nvidia shares rise as data center demand beats expectations",
                    source="Test News",
                    published_at=datetime(2026, 5, 18),
                ),
                HeadlineItem(
                    id=f"{ticker}-real-2",
                    ticker=ticker,
                    headline="Nvidia profit growth offsets investor concerns",
                    title="Nvidia profit growth offsets investor concerns",
                    source="Test News",
                    published_at=datetime(2026, 5, 18),
                ),
            ]
        )
        DataService.get_market_data = staticmethod(
            lambda ticker: MarketData(
                symbol=ticker,
                price=100.0,
                day_high=105.0,
                volume=1_000_000,
                source="Yahoo Finance via yfinance",
                status="ready",
                timestamp=datetime(2026, 5, 18),
            )
        )
        DataService.get_market_history = staticmethod(lambda ticker: [])
        DataService.get_fundamentals = staticmethod(lambda ticker: None)
        DataService.get_fundamentals_status = staticmethod(
            lambda ticker: {
                "available": False,
                "status": "unavailable",
                "source": "Yahoo Finance via yfinance",
                "message": "Fundamentals provider returned no usable fields.",
            }
        )

        sentiment = SentimentService.get_sentiment_for_ticker("NVDA")
        summary = asyncio.run(get_dashboard_summary("NVDA"))

        assert sentiment["overall_sentiment"] is not None
        assert sentiment["source_breakdown"]["Test News"]["count"] == 2
        assert summary.sentiment is not None
        assert summary.sentiment.sentiment_label in ("positive", "neutral", "negative")
        assert summary.prediction is not None
        assert summary.prediction.predicted_movement in ("up", "down", "neutral")
        assert summary.availability.sentiment.available is True
        assert summary.availability.prediction.available is True
    finally:
        DataService._fetch_yfinance_headlines = original_fetch
        DataService.get_market_data = original_market_data
        DataService.get_market_history = original_market_history
        DataService.get_fundamentals = original_fundamentals
        DataService.get_fundamentals_status = original_fundamentals_status
        sentiment_service._pipeline_file_path = original_pipeline_file_path
        DataService._PROVIDER_CACHE.clear()
        DataService._PROVIDER_STATUS.clear()


def test_market_data_preserves_pipeline_provenance() -> None:
    original_records = DataService._get_ticker_records
    original_load = DataService._load_pipeline_records
    try:
        DataService._get_ticker_records = staticmethod(
            lambda ticker: [
                {
                    "ticker": ticker,
                    "date": "2026-05-29",
                    "market_data": {
                        "price": 135.13,
                        "day_high": 139.35,
                        "volume": 313_566_600,
                        "price_delta_24h": -3.11,
                        "percent_change_24h": -2.2497,
                        "volume_delta": 0.3223,
                        "source": "Yahoo Finance via yfinance",
                        "status": "ready",
                    },
                }
            ]
        )
        DataService._load_pipeline_records = staticmethod(lambda: [])

        market_data = DataService.get_market_data("NVDA")

        assert market_data.source == "Yahoo Finance via yfinance"
        assert market_data.status == "ready"
        assert market_data.price_delta_24h == -3.11
        assert market_data.volume_delta == 0.3223
    finally:
        DataService._get_ticker_records = original_records
        DataService._load_pipeline_records = original_load


if __name__ == "__main__":
    test_headline_cache_falls_back_to_stale_data()
    test_fundamentals_cache_falls_back_to_stale_data()
    test_dashboard_summary_uses_provider_status()
    test_dashboard_summary_uses_headlines_for_demo_sentiment_and_prediction()
    print("Data provider cache tests passed.")

"""Data Service - API interface to data pipeline.

Author: Mihir (with integration from Isaac data pipeline)
Responsibility: Provide data access layer for market and social media data

Integration Points:
- Loads data from Isaac data pipeline (../data/app.py)
- Reads from stock_data.json containing raw posts with market features
- Provides market data and social media data to other services

Current Status: Active integration with pipeline data when available
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
from app.models.schemas import MarketData


class DataService:
    """Service for managing data retrieval from external sources."""

    @staticmethod
    def _pipeline_file_path() -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_data.json")
        )

    @staticmethod
    def _load_pipeline_records() -> List[Dict[str, Any]]:
        pipeline_path = DataService._pipeline_file_path()
        if not os.path.exists(pipeline_path):
            return []

        try:
            with open(pipeline_path, "r") as file:
                records = json.load(file)
            return records if isinstance(records, list) else []
        except Exception:
            return []

    @staticmethod
    def _get_ticker_records(ticker: str) -> List[Dict[str, Any]]:
        return [
            record
            for record in DataService._load_pipeline_records()
            if record.get("ticker", "").upper() == ticker
        ]

    @staticmethod
    def get_market_data(ticker: str) -> MarketData:
        """
        Get current market data for a stock.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            MarketData: Current market data

        The service attempts to use data from the pipeline JSON then falls back to yfinance.
        """
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        market_data = None
        ticker_records = DataService._get_ticker_records(ticker)
        if ticker_records:
            latest = sorted(ticker_records, key=lambda r: r.get("date", ""), reverse=True)[0]
            market_snapshot = latest.get("market_data", {}) or {}
            price = float(market_snapshot.get("price", 0.0) or 0.0)
            market_data = {
                "symbol": ticker,
                "price": price,
                "day_high": price,
                "volume": int(market_snapshot.get("volume", 0) or 0),
                "timestamp": datetime.now(),
            }

        # Fallback to yfinance if pipeline data is unavailable or incomplete
        if market_data is None or market_data['price'] <= 0:
            try:
                import yfinance as yf

                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    day_high = float(hist['High'].iloc[-1])
                    volume = int(hist['Volume'].iloc[-1])

                    market_data = {
                        'ticker': ticker,
                        'price': price,
                        'day_high': day_high,
                        'volume': volume,
                        'date': datetime.now().date().isoformat(),
                    }
            except Exception:
                market_data = None

        if market_data is None:
            market_data = {
                'ticker': ticker,
                'price': 0.0,
                'day_high': 0.0,
                'volume': 0,
                'date': datetime.now().date().isoformat(),
            }

        return MarketData(**market_data)

    @staticmethod
    def get_market_data_multiple(tickers: List[str]) -> Dict[str, MarketData]:
        """
        Get market data for multiple stocks.

        Args:
            tickers (List[str]): List of stock ticker symbols

        Returns:
            Dict[str, MarketData]: Market data for each ticker
        """
        return {ticker.upper(): DataService.get_market_data(ticker.upper()) for ticker in tickers}

    @staticmethod
    def get_social_media_data(ticker: str) -> Dict[str, Any]:
        """
        Retrieve social media data (Reddit, Twitter, etc.) for a ticker from pipeline.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            Dict: Social media posts and engagement metrics
        """
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        posts = []
        for record in DataService._get_ticker_records(ticker):
            record_posts = record.get("posts", []) or []
            for post in record_posts:
                posts.append(
                    {
                        "ticker": ticker,
                        "date": record.get("date", datetime.now().date().isoformat()),
                        "text": post.get("text", ""),
                        "source": post.get("source", "unknown"),
                        "post_score": post.get("post_score", 0),
                    }
                )

        if not posts:
            posts = [
                {
                    "ticker": ticker,
                    "date": datetime.now().date().isoformat(),
                    "text": f"Sample fallback post for {ticker}",
                    "source": "mock_social",
                    "post_score": 1,
                }
            ]

        return {
            "ticker": ticker,
            "post_count": len(posts),
            "posts": posts,
            "avg_post_score": sum([p.get("post_score", 0) for p in posts]) / max(1, len(posts)),
        }

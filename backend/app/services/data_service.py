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
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'stock_data.json'))

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
        pipeline_path = DataService._pipeline_file_path()

        # Try reading pipeline JSON first
        market_data = None
        if os.path.exists(pipeline_path):
            try:
                with open(pipeline_path, 'r') as f:
                    records = json.load(f)
                ticker_records = [r for r in records if r.get('ticker', '').upper() == ticker]
                if ticker_records:
                    latest = sorted(ticker_records, key=lambda r: r.get('date', ''), reverse=True)[0]
                    market_data = {
                        'symbol': ticker,
                        'price': float(latest.get('price', 0.0)) if latest.get('price') is not None else 0.0,
                        'day_high': float(latest.get('day_high', 0.0)) if latest.get('day_high') is not None else 0.0,
                        'volume': int(latest.get('volume', 0)) if latest.get('volume') is not None else 0,
                        'timestamp': datetime.now(),
                    }
            except Exception:
                market_data = None

        # Fallback to yfinance if pipeline data is unavailable or incomplete
        if market_data is None or market_data['price'] <= 0:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if hist.empty:
                raise RuntimeError(f"Unable to fetch market data for ticker {ticker}")

            price = float(hist['Close'].iloc[-1])
            day_high = float(hist['High'].iloc[-1])
            volume = int(hist['Volume'].iloc[-1])

            market_data = {
                'symbol': ticker,
                'price': price,
                'day_high': day_high,
                'volume': volume,
                'timestamp': datetime.now(),
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
        pipeline_path = DataService._pipeline_file_path()

        posts = []
        if os.path.exists(pipeline_path):
            try:
                with open(pipeline_path, 'r') as f:
                    records = json.load(f)
                posts = [r for r in records if r.get('ticker', '').upper() == ticker]
            except Exception:
                posts = []

        if not posts:
            # Provide fallback stub posts and guidelines for development
            posts = [
                {
                    'ticker': ticker,
                    'date': datetime.now().date().isoformat(),
                    'text': f"Sample fallback post for {ticker}",
                    'source': 'reddit',
                    'post_score': 10,
                    'price_delta_24h': 0.0,
                    'volume_delta': 0.0,
                }
            ]

        return {
            'ticker': ticker,
            'post_count': len(posts),
            'posts': posts,
            'avg_post_score': sum([p.get('post_score', 0) for p in posts]) / max(1, len(posts)),
        }

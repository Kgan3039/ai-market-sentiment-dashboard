"""Data Service - API interface to data pipeline.

Author: Mihir (with integration from Isaac data pipeline)
Responsibility: Provide data access layer for market and social media data

Integration Points:
- Loads data from Isaac data pipeline (../data/app.py)
- Reads from stock_data.json containing raw posts with market features
- Provides market data and social media data to other services

Current Status: Active integration with pipeline data when available
"""

import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from app.models.schemas import Fundamentals, HeadlineItem, MarketData


class DataService:
    """Service for managing data retrieval from external sources."""

    @staticmethod
    def _pipeline_file_path() -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_data.json")
        )

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
            try:
                import yfinance as yf

                stock = yf.Ticker(ticker)
                hist = stock.history(period='1d')
                if not hist.empty:
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
            except Exception:
                market_data = None

        if market_data is None:
            market_data = {
                'symbol': ticker,
                'price': 0.0,
                'day_high': 0.0,
                'volume': 0,
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
    def get_headlines(ticker: str, limit: int = 6) -> List[HeadlineItem]:
        """
        Get normalized headline items for a ticker.

        Uses yfinance/Yahoo Finance news because the project already depends on
        yfinance for market data and the same source can also provide company
        metadata for fundamentals. Returns an empty list when the provider is
        unavailable instead of fabricating demo headlines.
        """
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            raw_news = stock.news or []
        except Exception:
            return []

        headlines = []
        for index, item in enumerate(raw_news[:limit]):
            normalized = DataService._normalize_yfinance_news_item(ticker, item, index)
            if normalized is not None:
                headlines.append(normalized)

        return headlines

    @staticmethod
    def _normalize_yfinance_news_item(
        ticker: str, item: Dict[str, Any], index: int
    ) -> Optional[HeadlineItem]:
        content = item.get("content") if isinstance(item.get("content"), dict) else {}

        title = item.get("title") or content.get("title")
        if not title:
            return None

        provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
        canonical_url = content.get("canonicalUrl")
        click_url = content.get("clickThroughUrl")

        url = item.get("link")
        if not url and isinstance(canonical_url, dict):
            url = canonical_url.get("url")
        if not url and isinstance(click_url, dict):
            url = click_url.get("url")

        published_at = DataService._parse_publish_time(
            item.get("providerPublishTime") or content.get("pubDate") or content.get("displayTime")
        )

        source = (
            item.get("publisher")
            or provider.get("displayName")
            or provider.get("name")
            or "Yahoo Finance"
        )

        headline_id = str(item.get("id") or content.get("id") or f"{ticker}-{index}")
        summary = item.get("summary") or content.get("summary")

        return HeadlineItem(
            id=headline_id,
            ticker=ticker,
            headline=title,
            title=title,
            source=source,
            url=url,
            published_at=published_at,
            time=published_at.strftime("%b %-d, %Y") if published_at else None,
            summary=summary,
        )

    @staticmethod
    def _parse_publish_time(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def get_fundamentals(ticker: str) -> Optional[Fundamentals]:
        """
        Get company fundamentals and metadata for a ticker.

        yfinance exposes both comprehensive company info and a faster
        dictionary-like fast_info surface. The method merges the two so the UI
        can render ratios even when only lightweight metadata is available.
        """
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()

        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = DataService._safe_dict(stock.get_info())
            fast_info = DataService._safe_dict(stock.fast_info)
        except Exception:
            return None

        if not info and not fast_info:
            return None

        fundamentals = Fundamentals(
            source="Yahoo Finance via yfinance",
            company_name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=DataService._first_number(info, fast_info, "marketCap", "market_cap"),
            trailing_pe=DataService._first_number(info, fast_info, "trailingPE", "trailing_pe"),
            forward_pe=DataService._first_number(info, fast_info, "forwardPE", "forward_pe"),
            price_to_book=DataService._first_number(info, fast_info, "priceToBook", "price_to_book"),
            dividend_yield=DataService._first_number(info, fast_info, "dividendYield", "dividend_yield"),
            beta=DataService._first_number(info, fast_info, "beta"),
            eps=DataService._first_number(info, fast_info, "trailingEps", "eps"),
            revenue=DataService._first_number(info, fast_info, "totalRevenue", "revenue"),
            net_income=DataService._first_number(info, fast_info, "netIncomeToCommon", "net_income"),
            operating_cash_flow=DataService._first_number(
                info, fast_info, "operatingCashflow", "operating_cash_flow"
            ),
            debt_to_equity=DataService._first_number(info, fast_info, "debtToEquity", "debt_to_equity"),
            currency=info.get("financialCurrency") or info.get("currency") or fast_info.get("currency"),
        )

        useful_values = fundamentals.model_dump(exclude={"source"}, exclude_none=True)
        if not useful_values:
            return None

        return fundamentals

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        try:
            return dict(value)
        except Exception:
            return {}

    @staticmethod
    def _first_number(*sources_and_keys: Any) -> Optional[float]:
        sources = [value for value in sources_and_keys if isinstance(value, dict)]
        keys = [value for value in sources_and_keys if isinstance(value, str)]

        for source in sources:
            for key in keys:
                value = source.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        return None

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

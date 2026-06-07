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
from app.models.schemas import Fundamentals, HeadlineItem, MarketData, MarketHistoryPoint, SocialPostItem


class DataService:
    """Service for managing data retrieval from external sources."""

    _CACHE_TTL_SECONDS = int(os.getenv("DATA_SERVICE_CACHE_TTL_SECONDS", "900"))
    _PROVIDER_CACHE: Dict[str, Dict[str, Any]] = {}
    _PROVIDER_STATUS: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _pipeline_file_path() -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_data.json")
        )

    @staticmethod
    def _cache_key(provider: str, ticker: str, limit: Optional[int] = None) -> str:
        suffix = f":{limit}" if limit is not None else ""
        return f"{provider}:{ticker.upper()}{suffix}"

    @staticmethod
    def _clone_cached_value(value: Any) -> Any:
        if isinstance(value, list):
            return [DataService._clone_cached_value(item) for item in value]
        if hasattr(value, "model_copy"):
            return value.model_copy(deep=True)
        if hasattr(value, "copy"):
            return value.copy(deep=True)
        return value

    @staticmethod
    def _cache_lookup(key: str, allow_expired: bool = False) -> tuple[bool, Any, bool]:
        entry = DataService._PROVIDER_CACHE.get(key)
        if entry is None:
            return False, None, False

        age_seconds = (datetime.now() - entry["stored_at"]).total_seconds()
        expired = age_seconds > DataService._CACHE_TTL_SECONDS
        if expired and not allow_expired:
            return False, None, True

        return True, DataService._clone_cached_value(entry["value"]), expired

    @staticmethod
    def _cache_set(key: str, value: Any) -> None:
        DataService._PROVIDER_CACHE[key] = {
            "stored_at": datetime.now(),
            "value": DataService._clone_cached_value(value),
        }

    @staticmethod
    def _set_provider_status(
        key: str,
        *,
        available: bool,
        status: str,
        source: str,
        message: str,
        count: Optional[int] = None,
    ) -> None:
        DataService._PROVIDER_STATUS[key] = {
            "available": available,
            "status": status,
            "source": source,
            "message": message,
            "count": count,
        }

    @staticmethod
    def _provider_status(
        provider: str,
        ticker: str,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        key = DataService._cache_key(provider, ticker, limit)
        return DataService._PROVIDER_STATUS.get(
            key,
            {
                "available": False,
                "status": "unavailable",
                "source": "Yahoo Finance via yfinance",
                "message": f"{provider.title()} provider has not been queried yet.",
                "count": 0 if provider == "headlines" else None,
            },
        )

    @staticmethod
    def get_headlines_status(ticker: str, limit: int = 6) -> Dict[str, Any]:
        """Return provider status for the last headline lookup."""
        return DataService._provider_status("headlines", ticker, limit=limit)

    @staticmethod
    def get_fundamentals_status(ticker: str) -> Dict[str, Any]:
        """Return provider status for the last fundamentals lookup."""
        return DataService._provider_status("fundamentals", ticker)

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
    def _is_valid_pipeline_post(text: str, source: str) -> bool:
        text = str(text or "").strip()
        source = str(source or "").strip().lower()
        lowered_text = text.lower()

        if not text or source.startswith("mock"):
            return False

        placeholder_phrases = (
            "sample post while waiting for api approval",
            "sample fallback post",
            "discussion about",
        )
        return not any(phrase in lowered_text for phrase in placeholder_phrases)

    @staticmethod
    def _post_content_type(post: Dict[str, Any], source: str) -> str:
        explicit_type = str(post.get("content_type") or "").strip().lower()
        if explicit_type in {"social_post", "publisher_headline", "news_headline"}:
            return "publisher_headline" if explicit_type == "news_headline" else explicit_type

        source_text = str(source or "").strip().lower()
        social_sources = ("reddit", "twitter", "x.com", "stocktwits", "discord", "telegram")
        if any(source_name in source_text for source_name in social_sources):
            return "social_post"

        return "publisher_headline"

    @staticmethod
    def _pipeline_headlines(ticker: str, limit: int = 6) -> List[HeadlineItem]:
        headlines: List[HeadlineItem] = []

        for record_index, record in enumerate(DataService._get_ticker_records(ticker)):
            nested_posts = record.get("posts")
            record_posts = nested_posts if isinstance(nested_posts, list) else [record]
            for post_index, post in enumerate(record_posts):
                text = str(post.get("text", "")).strip()
                source = str(post.get("source", "Committed demo dataset")).strip()
                if not DataService._is_valid_pipeline_post(text, source):
                    continue

                published_at = None
                date_value = record.get("date")
                if date_value:
                    try:
                        published_at = datetime.fromisoformat(str(date_value))
                    except ValueError:
                        published_at = None

                headlines.append(
                    HeadlineItem(
                        id=str(post.get("id") or f"{ticker}-dataset-{record_index}-{post_index}"),
                        ticker=ticker,
                        headline=text,
                        title=text,
                        source=source,
                        published_at=published_at,
                        time=(
                            f"{published_at:%b} {published_at.day}, {published_at:%Y}"
                            if published_at
                            else None
                        ),
                    )
                )
                if len(headlines) >= limit:
                    return headlines

        return headlines

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _market_data_from_snapshot(ticker: str, market_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        price = float(market_snapshot.get("price", 0.0) or 0.0)
        status = market_snapshot.get("status") or ("cached" if price > 0 else "unavailable")
        source = market_snapshot.get("source") or (
            "Pipeline market snapshot" if price > 0 else "unavailable"
        )

        return {
            "symbol": ticker,
            "price": price,
            "day_high": float(market_snapshot.get("day_high", price) or price),
            "volume": int(market_snapshot.get("volume", 0) or 0),
            "price_delta_24h": DataService._optional_float(market_snapshot.get("price_delta_24h")),
            "percent_change_24h": DataService._optional_float(market_snapshot.get("percent_change_24h")),
            "volume_delta": DataService._optional_float(market_snapshot.get("volume_delta")),
            "source": source,
            "status": status,
            "timestamp": datetime.now(),
        }

    @staticmethod
    def _market_data_from_history(ticker: str, hist: Any, source: str) -> Optional[Dict[str, Any]]:
        if hist is None or hist.empty:
            return None

        latest = hist.iloc[-1]
        previous = hist.iloc[-2] if len(hist) > 1 else latest
        price = float(latest.get("Close", 0.0) or 0.0)
        previous_close = float(previous.get("Close", price) or price)
        price_delta = price - previous_close
        percent_change = (price_delta / previous_close * 100) if previous_close else 0.0

        previous_volume = hist["Volume"].iloc[-6:-1] if len(hist) > 1 else hist["Volume"].iloc[-1:]
        average_volume = float(previous_volume.mean() or 0.0)
        volume = int(latest.get("Volume", 0) or 0)
        volume_delta = ((volume - average_volume) / average_volume) if average_volume else 0.0

        return {
            "symbol": ticker,
            "price": price,
            "day_high": float(latest.get("High", price) or price),
            "volume": volume,
            "price_delta_24h": price_delta,
            "percent_change_24h": percent_change,
            "volume_delta": volume_delta,
            "source": source,
            "status": "live" if price > 0 else "unavailable",
            "timestamp": datetime.now(),
        }

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
            market_data = DataService._market_data_from_snapshot(ticker, market_snapshot)

        # Fallback to yfinance if pipeline data is unavailable or incomplete
        if market_data is None or market_data['price'] <= 0:
            try:
                import yfinance as yf

                stock = yf.Ticker(ticker)
                hist = stock.history(period='1mo')
                market_data = DataService._market_data_from_history(
                    ticker,
                    hist,
                    "Yahoo Finance via yfinance",
                )
            except Exception:
                market_data = None

        if market_data is None:
            market_data = {
                'symbol': ticker,
                'price': 0.0,
                'day_high': 0.0,
                'volume': 0,
                'price_delta_24h': 0.0,
                'percent_change_24h': 0.0,
                'volume_delta': 0.0,
                'source': 'unavailable',
                'status': 'unavailable',
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
    def get_market_history(ticker: str, period: str = "1mo") -> List[MarketHistoryPoint]:
        """Return recent historical close prices for compact dashboard charts."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        cache_key = DataService._cache_key(f"market_history:{period}", ticker)

        cache_hit, cached_history, _ = DataService._cache_lookup(cache_key)
        if cache_hit:
            return cached_history

        try:
            history = DataService._fetch_yfinance_market_history(ticker, period)
        except Exception:
            stale_hit, stale_history, _ = DataService._cache_lookup(cache_key, allow_expired=True)
            return stale_history if stale_hit else []

        DataService._cache_set(cache_key, history)
        return history

    @staticmethod
    def _fetch_yfinance_market_history(ticker: str, period: str) -> List[MarketHistoryPoint]:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return []

        points = []
        for index, row in hist.tail(32).iterrows():
            close = row.get("Close")
            if close is None:
                continue

            try:
                date_value = index.date().isoformat()
            except Exception:
                date_value = str(index)

            volume = row.get("Volume")
            try:
                volume_value = int(volume) if volume is not None and volume == volume else None
            except (TypeError, ValueError):
                volume_value = None

            points.append(
                MarketHistoryPoint(
                    date=date_value,
                    close=round(float(close), 2),
                    volume=volume_value,
                )
            )

        return points

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
        cache_key = DataService._cache_key("headlines", ticker, limit)

        cache_hit, cached_headlines, _ = DataService._cache_lookup(cache_key)
        if cache_hit:
            previous_status = DataService._PROVIDER_STATUS.get(cache_key, {})
            cached_source = previous_status.get("source", "Yahoo Finance via yfinance (cache)")
            DataService._set_provider_status(
                cache_key,
                available=len(cached_headlines) > 0,
                status="cached" if cached_headlines else "unavailable",
                source=cached_source,
                message=(
                    f"{len(cached_headlines)} cached headline items are available."
                    if cached_headlines
                    else "Headline provider recently returned no articles."
                ),
                count=len(cached_headlines),
            )
            return cached_headlines

        try:
            headlines = DataService._fetch_yfinance_headlines(ticker, limit)
        except Exception as exc:
            stale_hit, stale_headlines, _ = DataService._cache_lookup(cache_key, allow_expired=True)
            if stale_hit:
                DataService._set_provider_status(
                    cache_key,
                    available=len(stale_headlines) > 0,
                    status="fallback",
                    source="Yahoo Finance via yfinance (stale cache)",
                    message=(
                        f"Headline provider failed; using {len(stale_headlines)} cached items."
                        if stale_headlines
                        else "Headline provider failed and cached response was empty."
                    ),
                    count=len(stale_headlines),
                )
                return stale_headlines

            pipeline_headlines = DataService._pipeline_headlines(ticker, limit)
            if pipeline_headlines:
                DataService._cache_set(cache_key, pipeline_headlines)
                DataService._set_provider_status(
                    cache_key,
                    available=True,
                    status="cached",
                    source="Committed demo dataset",
                    message=(
                        f"Headline provider failed; using {len(pipeline_headlines)} committed demo headlines."
                    ),
                    count=len(pipeline_headlines),
                )
                return pipeline_headlines

            DataService._cache_set(cache_key, [])
            DataService._set_provider_status(
                cache_key,
                available=False,
                status="unavailable",
                source="Yahoo Finance via yfinance",
                message=f"Headline provider failed: {type(exc).__name__}.",
                count=0,
            )
            return []

        if not headlines:
            pipeline_headlines = DataService._pipeline_headlines(ticker, limit)
            if pipeline_headlines:
                DataService._cache_set(cache_key, pipeline_headlines)
                DataService._set_provider_status(
                    cache_key,
                    available=True,
                    status="cached",
                    source="Committed demo dataset",
                    message=f"Using {len(pipeline_headlines)} committed demo headlines.",
                    count=len(pipeline_headlines),
                )
                return pipeline_headlines

        DataService._cache_set(cache_key, headlines)
        DataService._set_provider_status(
            cache_key,
            available=len(headlines) > 0,
            status="live" if headlines else "unavailable",
            source="Yahoo Finance via yfinance",
            message=(
                f"{len(headlines)} headline items are available."
                if headlines
                else "Headline provider returned no articles."
            ),
            count=len(headlines),
        )
        return headlines

    @staticmethod
    def _fetch_yfinance_headlines(ticker: str, limit: int) -> List[HeadlineItem]:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        raw_news = stock.news or []
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
            time=(
                f"{published_at:%b} {published_at.day}, {published_at:%Y}"
                if published_at
                else None
            ),
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
        cache_key = DataService._cache_key("fundamentals", ticker)

        cache_hit, cached_fundamentals, _ = DataService._cache_lookup(cache_key)
        if cache_hit:
            DataService._set_provider_status(
                cache_key,
                available=cached_fundamentals is not None,
                status="cached" if cached_fundamentals is not None else "unavailable",
                source="Yahoo Finance via yfinance (cache)",
                message=(
                    "Cached company fundamentals are available."
                    if cached_fundamentals is not None
                    else "Fundamentals provider recently returned no usable fields."
                ),
            )
            return cached_fundamentals

        try:
            fundamentals = DataService._fetch_yfinance_fundamentals(ticker)
        except Exception as exc:
            stale_hit, stale_fundamentals, _ = DataService._cache_lookup(
                cache_key, allow_expired=True
            )
            if stale_hit:
                DataService._set_provider_status(
                    cache_key,
                    available=stale_fundamentals is not None,
                    status="fallback",
                    source="Yahoo Finance via yfinance (stale cache)",
                    message=(
                        "Fundamentals provider failed; using cached company fundamentals."
                        if stale_fundamentals is not None
                        else "Fundamentals provider failed and cached response was empty."
                    ),
                )
                return stale_fundamentals

            DataService._set_provider_status(
                cache_key,
                available=False,
                status="unavailable",
                source="Yahoo Finance via yfinance",
                message=f"Fundamentals provider failed: {type(exc).__name__}.",
            )
            return None

        DataService._cache_set(cache_key, fundamentals)
        DataService._set_provider_status(
            cache_key,
            available=fundamentals is not None,
            status="live" if fundamentals is not None else "unavailable",
            source="Yahoo Finance via yfinance",
            message=(
                "Company fundamentals are available."
                if fundamentals is not None
                else "Fundamentals provider returned no usable fields."
            ),
        )
        return fundamentals

    @staticmethod
    def _fetch_yfinance_fundamentals(ticker: str) -> Optional[Fundamentals]:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        info = DataService._safe_dict(stock.get_info())
        fast_info = DataService._safe_dict(stock.fast_info)

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
        Retrieve text inputs for a ticker from validated pipeline posts, falling
        back to Yahoo Finance headlines when the pipeline has no real text.

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
            nested_posts = record.get("posts")
            record_posts = nested_posts if isinstance(nested_posts, list) else [record]
            for post in record_posts:
                text = post.get("text", "")
                source = post.get("source", "unknown")
                if not DataService._is_valid_pipeline_post(text, source):
                    continue
                posts.append(
                    {
                        "ticker": ticker,
                        "date": record.get("date", datetime.now().date().isoformat()),
                        "text": text,
                        "source": source,
                        "post_score": post.get("post_score", 0),
                    }
                )

        if not posts:
            posts = []
            for headline in DataService.get_headlines(ticker, limit=8):
                if not DataService._is_valid_pipeline_post(headline.headline, headline.source):
                    continue
                posts.append(
                    {
                        "ticker": ticker,
                        "date": (
                            headline.published_at.date().isoformat()
                            if headline.published_at
                            else datetime.now().date().isoformat()
                        ),
                        "text": headline.headline,
                        "source": headline.source,
                        "post_score": 1,
                    }
                )

        return {
            "ticker": ticker,
            "post_count": len(posts),
            "posts": posts,
            "avg_post_score": sum([p.get("post_score", 0) for p in posts]) / max(1, len(posts)),
        }

    @staticmethod
    def get_social_posts(ticker: str) -> List[SocialPostItem]:
        """Return real pipeline social/news posts without fabricating fallback content."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        posts = []
        for record_index, record in enumerate(DataService._get_ticker_records(ticker)):
            nested_posts = record.get("posts")
            record_posts = nested_posts if isinstance(nested_posts, list) else [record]
            for post_index, post in enumerate(record_posts):
                text = str(post.get("text", "")).strip()
                source = str(post.get("source", "Unknown source")).strip() or "Unknown source"
                if not DataService._is_valid_pipeline_post(text, source):
                    continue

                posts.append(
                    SocialPostItem(
                        id=str(post.get("id") or f"{ticker}-{record_index}-{post_index}"),
                        ticker=ticker,
                        text=text,
                        source=source,
                        content_type=DataService._post_content_type(post, source),
                        date=record.get("date"),
                        post_score=post.get("post_score"),
                    )
                )

        return posts

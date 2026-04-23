"""Data Service - API interface to the data pipeline and market metadata sources."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.models.schemas import FundamentalsData, FundamentalRatios, FinancialSnapshot, HeadlineItem, MarketData


class DataService:
    """Service for managing data retrieval from pipeline output and market sources."""

    @staticmethod
    def _current_date() -> str:
        return datetime.now().date().isoformat()

    @staticmethod
    def _pipeline_file_path() -> str:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_data.json")
        )

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: Optional[int] = 0) -> Optional[int]:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_date(value: Any) -> str:
        if value is None:
            return DataService._current_date()

        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value).date().isoformat()
            except (OverflowError, OSError, ValueError):
                return DataService._current_date()

        text = str(value).strip()
        if not text:
            return DataService._current_date()

        if "T" in text or " " in text:
            return text

        return text

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
        ticker = ticker.upper()
        return [
            record
            for record in DataService._load_pipeline_records()
            if record.get("ticker", "").upper() == ticker
        ]

    @staticmethod
    def _extract_posts_from_record(record: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
        record_date = DataService._coerce_date(record.get("date"))
        posts = record.get("posts")

        if isinstance(posts, list):
            normalized_posts = []
            for post in posts:
                text = post.get("text") or post.get("headline") or post.get("title") or ""
                if not str(text).strip():
                    continue
                normalized_posts.append(
                    {
                        "ticker": ticker,
                        "date": DataService._coerce_date(post.get("published_at") or post.get("date") or record_date),
                        "text": str(text).strip(),
                        "source": post.get("source") or post.get("publisher") or "pipeline",
                        "post_score": DataService._safe_int(post.get("post_score"), 1) or 1,
                        "url": post.get("url") or post.get("link"),
                    }
                )
            return normalized_posts

        text = record.get("text") or record.get("headline") or record.get("title")
        if text:
            return [
                {
                    "ticker": ticker,
                    "date": record_date,
                    "text": str(text).strip(),
                    "source": record.get("source", "pipeline"),
                    "post_score": DataService._safe_int(record.get("post_score"), 1) or 1,
                    "url": record.get("url") or record.get("link"),
                }
            ]

        return []

    @staticmethod
    def _extract_headlines_from_record(record: Dict[str, Any], ticker: str) -> List[HeadlineItem]:
        record_date = DataService._coerce_date(record.get("date"))
        headline_candidates = record.get("headlines")

        if not isinstance(headline_candidates, list):
            headline_candidates = record.get("posts") if isinstance(record.get("posts"), list) else [record]

        items: List[HeadlineItem] = []
        for index, candidate in enumerate(headline_candidates):
            headline = candidate.get("headline") or candidate.get("title") or candidate.get("text") or ""
            if not str(headline).strip():
                continue

            items.append(
                HeadlineItem(
                    id=str(candidate.get("id") or f"{ticker}-{record_date}-{index}"),
                    ticker=ticker,
                    headline=str(headline).strip(),
                    source=candidate.get("source") or candidate.get("publisher") or "pipeline",
                    url=candidate.get("url") or candidate.get("link"),
                    published_at=DataService._coerce_date(
                        candidate.get("published_at")
                        or candidate.get("datetime")
                        or candidate.get("date")
                        or record_date
                    ),
                    sentiment_label=candidate.get("sentiment_label"),
                    sentiment_score=DataService._safe_float(candidate.get("sentiment_score"), None),
                )
            )

        return items

    @staticmethod
    def _extract_market_snapshot(record: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
        market_snapshot = record.get("market_data") if isinstance(record.get("market_data"), dict) else {}
        date = DataService._coerce_date(record.get("date"))

        price = DataService._safe_float(
            market_snapshot.get("price") if market_snapshot else record.get("price"),
            None,
        )
        if price is None:
            return None

        day_high = DataService._safe_float(
            market_snapshot.get("day_high") or market_snapshot.get("high") or record.get("day_high"),
            price,
        )
        volume = DataService._safe_int(
            market_snapshot.get("volume") if market_snapshot else record.get("volume"),
            0,
        )

        return {
            "ticker": ticker,
            "price": price,
            "day_high": day_high if day_high is not None else price,
            "volume": volume or 0,
            "date": date,
        }

    @staticmethod
    def _extract_feature_snapshot(record: Dict[str, Any]) -> Dict[str, float]:
        market_snapshot = record.get("market_data") if isinstance(record.get("market_data"), dict) else {}

        return {
            "price_delta_24h": DataService._safe_float(
                market_snapshot.get("price_delta_24h") if market_snapshot else record.get("price_delta_24h"),
                0.0,
            ) or 0.0,
            "volume_delta": DataService._safe_float(
                market_snapshot.get("volume_delta") if market_snapshot else record.get("volume_delta"),
                0.0,
            ) or 0.0,
        }

    @staticmethod
    def _fetch_yfinance_market_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                return None

            return {
                "ticker": ticker,
                "price": float(hist["Close"].iloc[-1]),
                "day_high": float(hist["High"].iloc[-1]),
                "volume": int(hist["Volume"].iloc[-1]),
                "date": DataService._current_date(),
            }
        except Exception:
            return None

    @staticmethod
    def _fetch_yfinance_news(ticker: str, max_items: int) -> List[HeadlineItem]:
        try:
            import yfinance as yf

            news_items = getattr(yf.Ticker(ticker), "news", []) or []
        except Exception:
            return []

        normalized: List[HeadlineItem] = []
        for index, item in enumerate(news_items[:max_items]):
            content = item.get("content") if isinstance(item.get("content"), dict) else item
            headline = content.get("title") or content.get("headline")
            if not headline:
                continue

            canonical_url = content.get("canonicalUrl")
            if isinstance(canonical_url, dict):
                url = canonical_url.get("url")
            else:
                url = content.get("link") or content.get("url")

            normalized.append(
                HeadlineItem(
                    id=str(content.get("uuid") or content.get("id") or f"{ticker}-yf-{index}"),
                    ticker=ticker,
                    headline=str(headline).strip(),
                    source=content.get("provider") or content.get("publisher") or "yfinance",
                    url=url,
                    published_at=DataService._coerce_date(
                        content.get("pubDate") or content.get("providerPublishTime") or DataService._current_date()
                    ),
                )
            )

        return normalized

    @staticmethod
    def _statement_value(statement: Any, labels: List[str]) -> Optional[float]:
        if statement is None:
            return None

        try:
            if statement.empty:
                return None
        except Exception:
            return None

        normalized_index = {str(index).strip().lower(): index for index in getattr(statement, "index", [])}
        for label in labels:
            matched_index = normalized_index.get(label.lower())
            if matched_index is None:
                continue

            series = statement.loc[matched_index]
            try:
                cleaned = series.dropna()
                if not cleaned.empty:
                    return DataService._safe_float(cleaned.iloc[0], None)
            except Exception:
                return DataService._safe_float(series, None)

        return None

    @staticmethod
    def _build_fundamentals_from_yfinance(ticker: str) -> Optional[FundamentalsData]:
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = getattr(stock, "info", {}) or {}
            annual_income = getattr(stock, "financials", None)
            quarterly_income = getattr(stock, "quarterly_financials", None)
            annual_cashflow = getattr(stock, "cashflow", None)
            quarterly_cashflow = getattr(stock, "quarterly_cashflow", None)
        except Exception:
            return None

        annual = FinancialSnapshot(
            revenue=DataService._statement_value(annual_income, ["Total Revenue", "Operating Revenue"]),
            net_income=DataService._statement_value(annual_income, ["Net Income", "Net Income Common Stockholders"]),
            operating_cash_flow=DataService._statement_value(annual_cashflow, ["Operating Cash Flow"]),
            eps=DataService._safe_float(info.get("trailingEps"), None),
        )
        quarterly = FinancialSnapshot(
            revenue=DataService._statement_value(quarterly_income, ["Total Revenue", "Operating Revenue"]),
            net_income=DataService._statement_value(quarterly_income, ["Net Income", "Net Income Common Stockholders"]),
            operating_cash_flow=DataService._statement_value(quarterly_cashflow, ["Operating Cash Flow"]),
            eps=DataService._safe_float(info.get("currentEps") or info.get("trailingEps"), None),
        )
        ratios = FundamentalRatios(
            pe=DataService._safe_float(info.get("trailingPE"), None),
            eps=DataService._safe_float(info.get("trailingEps"), None),
            roe=DataService._safe_float(info.get("returnOnEquity"), None),
            debt_to_equity=DataService._safe_float(info.get("debtToEquity"), None),
            revenue_growth_yoy=DataService._safe_float(info.get("revenueGrowth"), None),
            gross_margin=DataService._safe_float(info.get("grossMargins"), None),
        )

        if not any(
            value is not None
            for value in [
                annual.revenue,
                annual.net_income,
                annual.operating_cash_flow,
                annual.eps,
                quarterly.revenue,
                quarterly.net_income,
                quarterly.operating_cash_flow,
                quarterly.eps,
                ratios.pe,
                ratios.eps,
                ratios.roe,
                ratios.debt_to_equity,
                ratios.revenue_growth_yoy,
                ratios.gross_margin,
                info.get("longName"),
            ]
        ):
            return None

        return FundamentalsData(
            ticker=ticker,
            company_name=info.get("longName") or info.get("shortName"),
            exchange=info.get("exchange"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=DataService._safe_float(info.get("marketCap"), None),
            currency=info.get("currency"),
            annual=annual,
            quarterly=quarterly,
            ratios=ratios,
            source="yfinance",
        )

    @staticmethod
    def get_market_data(ticker: str) -> MarketData:
        """Get current market data for a stock."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        ticker_records = sorted(
            DataService._get_ticker_records(ticker),
            key=lambda record: str(record.get("date", "")),
            reverse=True,
        )

        market_data = None
        for record in ticker_records:
            market_data = DataService._extract_market_snapshot(record, ticker)
            if market_data and market_data["price"] > 0:
                break

        if market_data is None or market_data["price"] <= 0:
            market_data = DataService._fetch_yfinance_market_snapshot(ticker)

        if market_data is None:
            market_data = {
                "ticker": ticker,
                "price": 0.0,
                "day_high": 0.0,
                "volume": 0,
                "date": DataService._current_date(),
            }

        return MarketData(**market_data)

    @staticmethod
    def get_market_data_multiple(tickers: List[str]) -> Dict[str, MarketData]:
        """Get market data for multiple stocks."""
        return {ticker.upper(): DataService.get_market_data(ticker.upper()) for ticker in tickers}

    @staticmethod
    def get_social_media_data(ticker: str) -> Dict[str, Any]:
        """Retrieve social/news text items for a ticker from the pipeline."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        posts: List[Dict[str, Any]] = []
        for record in DataService._get_ticker_records(ticker):
            posts.extend(DataService._extract_posts_from_record(record, ticker))

        source = "pipeline_posts" if posts else "fallback"
        if not posts:
            posts = [
                {
                    "ticker": ticker,
                    "date": DataService._current_date(),
                    "text": f"Sample fallback post for {ticker}",
                    "source": "fallback",
                    "post_score": 1,
                    "url": None,
                }
            ]

        latest_date = max((str(post.get("date", "")) for post in posts), default=DataService._current_date())
        return {
            "ticker": ticker,
            "date": latest_date or DataService._current_date(),
            "source": source,
            "post_count": len(posts),
            "posts": posts,
            "avg_post_score": sum(post.get("post_score", 0) for post in posts) / max(1, len(posts)),
        }

    @staticmethod
    def get_feature_snapshot(ticker: str) -> Dict[str, float]:
        """Get model input feature deltas from the pipeline when present."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        ticker_records = sorted(
            DataService._get_ticker_records(ticker),
            key=lambda record: str(record.get("date", "")),
            reverse=True,
        )
        if not ticker_records:
            return {"price_delta_24h": 0.0, "volume_delta": 0.0}

        return DataService._extract_feature_snapshot(ticker_records[0])

    @staticmethod
    def get_headlines_for_ticker(ticker: str, max_items: int = 8) -> Dict[str, Any]:
        """Get headline items for the dashboard Market Pulse section."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        dedupe: set[tuple[str, str]] = set()
        headlines: List[HeadlineItem] = []

        ticker_records = sorted(
            DataService._get_ticker_records(ticker),
            key=lambda record: str(record.get("date", "")),
            reverse=True,
        )
        for record in ticker_records:
            for headline in DataService._extract_headlines_from_record(record, ticker):
                dedupe_key = (headline.headline, headline.source)
                if dedupe_key in dedupe:
                    continue
                dedupe.add(dedupe_key)
                headlines.append(headline)
                if len(headlines) >= max_items:
                    return {
                        "ticker": ticker,
                        "source": "pipeline",
                        "headlines": headlines,
                    }

        if headlines:
            return {
                "ticker": ticker,
                "source": "pipeline",
                "headlines": headlines,
            }

        fallback_news = DataService._fetch_yfinance_news(ticker, max_items=max_items)
        if fallback_news:
            return {
                "ticker": ticker,
                "source": "yfinance_news",
                "headlines": fallback_news,
            }

        return {
            "ticker": ticker,
            "source": "unavailable",
            "headlines": [],
        }

    @staticmethod
    def get_fundamentals_for_ticker(ticker: str) -> Dict[str, Any]:
        """Get company fundamentals and related metadata when available."""
        if not ticker or len(ticker.strip()) == 0:
            raise ValueError("Invalid ticker symbol")

        ticker = ticker.upper()
        for record in DataService._get_ticker_records(ticker):
            fundamentals = record.get("fundamentals")
            if isinstance(fundamentals, dict):
                try:
                    payload = FundamentalsData(
                        ticker=ticker,
                        company_name=fundamentals.get("company_name") or fundamentals.get("companyName"),
                        exchange=fundamentals.get("exchange"),
                        sector=fundamentals.get("sector"),
                        industry=fundamentals.get("industry"),
                        market_cap=DataService._safe_float(
                            fundamentals.get("market_cap") or fundamentals.get("marketCap"),
                            None,
                        ),
                        currency=fundamentals.get("currency"),
                        annual=FinancialSnapshot(
                            revenue=DataService._safe_float(
                                (fundamentals.get("annual") or {}).get("revenue"),
                                None,
                            ),
                            net_income=DataService._safe_float(
                                (fundamentals.get("annual") or {}).get("net_income"),
                                None,
                            ),
                            operating_cash_flow=DataService._safe_float(
                                (fundamentals.get("annual") or {}).get("operating_cash_flow"),
                                None,
                            ),
                            eps=DataService._safe_float(
                                (fundamentals.get("annual") or {}).get("eps"),
                                None,
                            ),
                        ),
                        quarterly=FinancialSnapshot(
                            revenue=DataService._safe_float(
                                (fundamentals.get("quarterly") or {}).get("revenue"),
                                None,
                            ),
                            net_income=DataService._safe_float(
                                (fundamentals.get("quarterly") or {}).get("net_income"),
                                None,
                            ),
                            operating_cash_flow=DataService._safe_float(
                                (fundamentals.get("quarterly") or {}).get("operating_cash_flow"),
                                None,
                            ),
                            eps=DataService._safe_float(
                                (fundamentals.get("quarterly") or {}).get("eps"),
                                None,
                            ),
                        ),
                        ratios=FundamentalRatios(
                            pe=DataService._safe_float((fundamentals.get("ratios") or {}).get("pe"), None),
                            eps=DataService._safe_float((fundamentals.get("ratios") or {}).get("eps"), None),
                            roe=DataService._safe_float((fundamentals.get("ratios") or {}).get("roe"), None),
                            debt_to_equity=DataService._safe_float(
                                (fundamentals.get("ratios") or {}).get("debt_to_equity"),
                                None,
                            ),
                            revenue_growth_yoy=DataService._safe_float(
                                (fundamentals.get("ratios") or {}).get("revenue_growth_yoy"),
                                None,
                            ),
                            gross_margin=DataService._safe_float(
                                (fundamentals.get("ratios") or {}).get("gross_margin"),
                                None,
                            ),
                        ),
                        source=fundamentals.get("source", "pipeline"),
                    )
                    return {
                        "ticker": ticker,
                        "source": "pipeline",
                        "fundamentals": payload,
                    }
                except Exception:
                    break

        payload = DataService._build_fundamentals_from_yfinance(ticker)
        if payload is not None:
            return {
                "ticker": ticker,
                "source": payload.source,
                "fundamentals": payload,
            }

        return {
            "ticker": ticker,
            "source": "unavailable",
            "fundamentals": None,
        }

"""Data pipeline output generator.

Author: Isaac
Responsibility: Collect market/news data when available and write grouped
records that downstream services can consume consistently.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import yfinance as yf
from dotenv import load_dotenv


load_dotenv()

TICKERS = ["NVDA", "TSLA"]
OUTPUT_PATH = Path(__file__).with_name("stock_data.json")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def get_yfinance_posts(ticker: str, limit: int = 15) -> list[dict[str, Any]]:
    """Fetch company headlines from Yahoo Finance via yfinance."""
    stock = yf.Ticker(ticker)
    raw_news = stock.news or []
    posts = []

    for item in raw_news[:limit]:
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        title = item.get("title") or content.get("title")
        if not title:
            continue

        provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}
        source = (
            item.get("publisher")
            or provider.get("displayName")
            or provider.get("name")
            or "Yahoo Finance"
        )
        posts.append(
            {
                "text": title,
                "source": source,
                "post_score": 1,
            }
        )

    return posts


def build_mock_market_data(ticker: str) -> dict[str, Any]:
    """Return stable placeholder market data for local development."""
    base_prices = {"NVDA": 910.5, "TSLA": 174.2}
    base_price = base_prices.get(ticker, 100.0)
    price_delta = 2.5 if ticker == "NVDA" else -1.2
    percent_change = (price_delta / base_price) * 100 if base_price else 0.0

    return {
        "price": round(base_price, 2),
        "price_delta_24h": round(price_delta, 2),
        "percent_change_24h": round(percent_change, 2),
        "volume": 1_250_000 if ticker == "NVDA" else 980_000,
    }


def get_market_snapshot(ticker: str, api_key: str | None) -> dict[str, Any]:
    """Fetch market snapshot when a Finnhub key is configured, else use mock data."""
    if not api_key:
        return build_mock_market_data(ticker)

    quote_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={api_key}"
    try:
        response = requests.get(quote_url, timeout=10)
        response.raise_for_status()
        quote = response.json()
        return {
            "price": _safe_float(quote.get("c")),
            "price_delta_24h": _safe_float(quote.get("d")),
            "percent_change_24h": _safe_float(quote.get("dp")),
            "volume": _safe_int(quote.get("v"), default=0),
        }
    except Exception:
        return build_mock_market_data(ticker)


def get_posts(ticker: str, api_key: str | None) -> list[dict[str, Any]]:
    """Fetch company news from real providers without fabricating text."""
    if not api_key:
        try:
            return get_yfinance_posts(ticker)
        except Exception:
            return []

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    news_url = (
        "https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={start_date}&to={end_date}&token={api_key}"
    )

    try:
        response = requests.get(news_url, timeout=10)
        response.raise_for_status()
        articles = response.json()
        posts = []
        for article in articles[:15]:
            posts.append(
                {
                    "text": article.get("headline", ""),
                    "source": article.get("source", "finnhub"),
                    "post_score": 1,
                }
            )
        if posts:
            return posts
    except Exception:
        pass

    try:
        return get_yfinance_posts(ticker)
    except Exception:
        return []


def build_grouped_record(ticker: str, date: str, api_key: str | None) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "date": date,
        "posts": get_posts(ticker, api_key),
        "market_data": get_market_snapshot(ticker, api_key),
    }


def main() -> None:
    current_date = datetime.now().date().isoformat()
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")

    data_out = [
        build_grouped_record(ticker=ticker, date=current_date, api_key=finnhub_api_key)
        for ticker in TICKERS
    ]

    with OUTPUT_PATH.open("w") as file:
        json.dump(data_out, file, indent=2)

    print(f"Saved grouped pipeline data to {OUTPUT_PATH}")
    print(f"Generated {len(data_out)} ticker records")


if __name__ == "__main__":
    main()

"""Data pipeline output generator.

This module writes grouped records that keep post content separate from shared
market data so downstream services do not have to deduplicate market fields per
post.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


tickers = ["NVDA", "TSLA"]
output_path = Path(__file__).with_name("stock_data.json")


def get_market_snapshot(ticker: str) -> dict[str, float | int]:
    """Fetch market snapshot data when available, otherwise return safe defaults."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if len(hist) >= 2:
            current_price = float(hist["Close"].iloc[-1])
            previous_price = float(hist["Close"].iloc[-2])
            price_delta_24h = current_price - previous_price
            percent_change_24h = (
                (price_delta_24h / previous_price) * 100 if previous_price else 0.0
            )
            volume = int(hist["Volume"].iloc[-1])

            return {
                "price": current_price,
                "price_delta_24h": price_delta_24h,
                "percent_change_24h": percent_change_24h,
                "volume": volume,
            }
    except Exception:
        pass

    return {
        "price": 0.0,
        "price_delta_24h": 0.0,
        "percent_change_24h": 0.0,
        "volume": 0,
    }


def build_posts(ticker: str) -> list[dict[str, str | int]]:
    """Return placeholder posts in the grouped contract format."""
    return [
        {
            "text": "Sample post while waiting for approved ingestion pipeline output.",
            "source": "mock_news",
            "post_score": 1,
        },
        {
            "text": f"Discussion about {ticker} while live data integration is pending.",
            "source": "mock_social",
            "post_score": 1,
        },
    ]


def main() -> None:
    current_date = datetime.now().date().isoformat()
    grouped_output = []

    for ticker in tickers:
        grouped_output.append(
            {
                "ticker": ticker,
                "date": current_date,
                "posts": build_posts(ticker),
                "market_data": get_market_snapshot(ticker),
            }
        )

    with output_path.open("w") as file:
        json.dump(grouped_output, file, indent=4)

    print(f"Saved grouped pipeline data to {output_path}")
    print(f"Generated {len(grouped_output)} ticker records")


if __name__ == "__main__":
    main()

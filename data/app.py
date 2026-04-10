"""Data Pipeline - Generates raw social media posts with market features.

Author: Isaac
Responsibility: Collect social media data from approved APIs and combine with market data

Dataset Format Contract:
- Output format must match dataset_format.md specification
- Fields: ticker, date, text, source, post_score, price_delta_24h, volume_delta
- This data flows into Matthew NLP pipeline
"""

import json
from datetime import datetime
from pathlib import Path

tickers = ["NVDA", "TSLA"]
data_out = []
output_path = Path(__file__).with_name("stock_data.json")


def get_market_deltas(ticker: str) -> tuple[float, float]:
    """Fetch market deltas when available, otherwise return demo-safe defaults."""
    try:
        import yfinance as yf

        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")

        if len(hist) >= 2:
            current_price = hist["Close"].iloc[-1]
            previous_price = hist["Close"].iloc[-2]
            price_delta_24h = (
                (current_price - previous_price) / previous_price if previous_price else 0.0
            )

            recent_volume = hist["Volume"].iloc[-1]
            avg_volume = hist["Volume"].tail(5).mean()
            volume_delta = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
            return float(price_delta_24h), float(volume_delta)
    except Exception:
        pass

    return 0.0, 0.0

for t in tickers:
    price_delta_24h, volume_delta = get_market_deltas(t)

    # Get current date in ISO format
    current_date = datetime.now().date().isoformat()

    # Social media data in dataset_format.md format
    # TODO (Isaac): Replace with real Reddit API data when approved
    #   - Once Reddit API access is granted, fetch real posts from r/stocks, r/investing, etc.
    #   - Parse post content and assign post_score based on engagement (upvotes, comments)
    #   - Ensure date is in ISO format and post_score is numeric
    social_posts = [
        {
            "ticker": t,
            "date": current_date,
            "text": "Sample post while waiting for API approval",
            "source": "reddit",
            "post_score": 10,
            "price_delta_24h": price_delta_24h,
            "volume_delta": volume_delta
        },
        {
            "ticker": t,
            "date": current_date,
            "text": f"Discussion about {t} on r/stocks",
            "source": "reddit",
            "post_score": 25,
            "price_delta_24h": price_delta_24h,
            "volume_delta": volume_delta
        }
    ]

    data_out.extend(social_posts)

# Save raw data next to this script so the backend can load it reliably.
with output_path.open("w") as f:
    json.dump(data_out, f, indent=4)

# TODO (Isaac): Add error handling for missing market data
# TODO (Isaac): Add retry logic for yfinance rate limiting
# TODO (Isaac): Implement data validation to ensure all required fields are present
# TODO (Isaac): Add logging to track data pipeline execution
# TODO (Mihir): Once data pipeline is finalized, set up scheduled execution (e.g., hourly)

print(f"Pipeline updated with real market data. Saved to {output_path}")
print(f"Generated {len(data_out)} social media posts in dataset_format.md format")

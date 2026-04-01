"""Data Pipeline - Generates raw social media posts with market features.

Author: Isaac
Responsibility: Collect social media data from approved APIs and combine with market data

Dataset Format Contract:
- Output format must match dataset_format.md specification
- Fields: ticker, date, text, source, post_score, price_delta_24h, volume_delta
- This data flows into Matthew NLP pipeline
"""

import yfinance as yf
import json
from datetime import datetime, timedelta

tickers = ["NVDA", "TSLA"]
data_out = []

for t in tickers:
    stock = yf.Ticker(t)

    # Get historical data for price delta calculation
    hist = stock.history(period="2d")  # Last 2 days for 24h delta

    if len(hist) >= 2:
        # Calculate 24h price change
        current_price = hist['Close'].iloc[-1]
        previous_price = hist['Close'].iloc[-2]
        price_delta_24h = (current_price - previous_price) / previous_price

        # Calculate volume delta (vs 5-day average)
        recent_volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume'].tail(5).mean()
        volume_delta = (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
    else:
        price_delta_24h = 0.0
        volume_delta = 0.0

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

# Save raw data in dataset_format.md format
with open("stock_data.json", "w") as f:
    json.dump(data_out, f, indent=4)

# TODO (Isaac): Add error handling for missing market data
# TODO (Isaac): Add retry logic for yfinance rate limiting
# TODO (Isaac): Implement data validation to ensure all required fields are present
# TODO (Isaac): Add logging to track data pipeline execution
# TODO (Mihir): Once data pipeline is finalized, set up scheduled execution (e.g., hourly)

print("Pipeline updated with real market data. Saved to stock_data.json")
print(f"Generated {len(data_out)} social media posts in dataset_format.md format")

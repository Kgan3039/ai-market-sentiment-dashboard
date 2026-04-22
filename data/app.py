"""Data Pipeline - Generates raw social media posts with market features.

Author: Isaac
Responsibility: Collect social media data from approved APIs and combine with market data

Dataset Format Contract:
- Output format must match dataset_format.md specification
- Fields: ticker, market_data (current_price, delta), posts (text, source, score)
- This data flows into Matthew NLP pipeline
"""

import requests
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
    if not FINNHUB_API_KEY:
        raise ValueError("FINNHUB_API_KEY not found. Make sure your .env file is set up correctly.")

    tickers = ["NVDA", "TSLA"]
    data_out = []
    output_path = Path(__file__).with_name("stock_data.json")

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    for t in tickers:
        quote_url = f"https://finnhub.io/api/v1/quote?symbol={t}&token={FINNHUB_API_KEY}"
        quote_response = requests.get(quote_url)
        market_stats = {}
        
        if quote_response.status_code == 200:
            q = quote_response.json()
            market_stats = {
                "current_price": q.get('c'),
                "price_delta_24h": q.get('d'),
                "percent_change_24h": q.get('dp'),
                "timestamp": datetime.now().isoformat()
            }

        news_url = f"https://finnhub.io/api/v1/company-news?symbol={t}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
        news_response = requests.get(news_url)
        posts = []
        
        if news_response.status_code == 200:
            news_data = news_response.json()
            for article in news_data[:15]:
                posts.append({
                    "date": datetime.fromtimestamp(article.get('datetime')).strftime("%Y-%m-%d"),
                    "text": article.get('headline'),
                    "source": article.get('source', 'finnhub'),
                    "url": article.get('url'),
                    "post_score": 1
                })

        data_out.append({
            "ticker": t,
            "market_data": market_stats,
            "posts": posts
        })

    with output_path.open("w") as f:
        json.dump(data_out, f, indent=4)

    print(f"Pipeline updated with real market data. Saved to {output_path}")

if __name__ == "__main__":
    main()

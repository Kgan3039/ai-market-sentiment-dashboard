import yfinance as yf
import json
from datetime import datetime

tickers = ["NVDA", "TSLA"]
data_out = []

for t in tickers:
    stock = yf.Ticker(t)
    info = stock.info
    
    price_data = {
        "symbol": t,
        "price": info.get("currentPrice"),
        "day_high": info.get("dayHigh"),
        "volume": info.get("volume"),
        "timestamp": datetime.now().isoformat()
    }

    reddit_data = [
        {"title": "Sample post while waiting for API approval", "score": 10},
        {"title": f"Discussion about {t} on r/stocks", "score": 25}
    ]

    data_out.append({
        "market": price_data,
        "social": reddit_data
    })


with open("stock_data.json", "w") as f:
    json.dump(data_out, f, indent=4)

print("Pipeline updated with real market data. Saved to stock_data.json")

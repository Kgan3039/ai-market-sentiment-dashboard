import requests
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") 

if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found. Make sure your .env file is set up correctly.")

tickers = ["NVDA", "TSLA"]
combined_data_out = []

end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

for t in tickers:
    quote_url = f"https://finnhub.io/api/v1/quote?symbol={t}&token={FINNHUB_API_KEY}"
    quote_response = requests.get(quote_url)
    
    current_price = None
    price_delta_24h = None
    percent_change_24h = None
    
    if quote_response.status_code == 200:
        quote_data = quote_response.json()
        current_price = quote_data.get('c')
        price_delta_24h = quote_data.get('d')
        percent_change_24h = quote_data.get('dp')

    news_url = f"https://finnhub.io/api/v1/company-news?symbol={t}&from={start_date}&to={end_date}&token={FINNHUB_API_KEY}"
    news_response = requests.get(news_url)
    
    if news_response.status_code == 200:
        news_data = news_response.json()
        
        for article in news_data[:15]: 
            row = {
                "ticker": t,
                "date": datetime.fromtimestamp(article.get('datetime')).strftime("%Y-%m-%d"), 
                "text": article.get('headline'),
                "source": article.get('source', 'finnhub'), 
                "post_score": 1,
                "current_price": current_price,
                "price_delta_24h": price_delta_24h,
                "percent_change_24h": percent_change_24h
            }
            combined_data_out.append(row)

with open("dataset_sample.json", "w") as f:
    json.dump(combined_data_out, f, indent=4)

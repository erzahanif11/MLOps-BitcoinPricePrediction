import json
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

RAW_DIR = Path("../data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_bitcoin_data():
    params = {
        "vs_currency": "usd",
        "days": "7"
    }
    response = requests.get(API_URL, params=params, timeout=30)

    response.raise_for_status() 
    return response.json()

def save_raw(data):
    now = datetime.now(timezone.utc)

    filename = f"bitcoin_data_{now.strftime('%Y%m%d_%H%M%S')}.json"
    file_path = RAW_DIR / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Data saved to {file_path}")

def main():
    print("Fetching Bitcoin data from CoinGecko...")
    data = fetch_bitcoin_data()
    save_raw(data)
    print("Data ingestion completed.")

if __name__ == "__main__":
    main()
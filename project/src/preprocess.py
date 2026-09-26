import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("../data/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "bitcoin_data.csv"

def load_raw_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def transform_data(raw_data):
    prices = raw_data.get("prices", [])
    market_caps = raw_data.get("market_caps", [])
    total_volumes = raw_data.get("total_volumes", [])

    rows = []

    for i, price_data in enumerate(prices):
        timestamp = price_data[0]
        price = price_data[1]

        market_cap = (
            market_caps[i][1] 
            if i < len(market_caps) 
            else None
        )

        total_volume = (
            total_volumes[i][1] 
            if i < len(total_volumes) 
            else None
        )

        rows.append({
            "timestamp": timestamp,
            "price": price,
            "market_cap": market_cap,
            "total_volume": total_volume
        })
    
    df = pd.DataFrame(rows)
    return df

def preproccess_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Jakarta")
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df[(df["timestamp"].dt.minute == 0) & (df["timestamp"].dt.second == 0)]
    return df

def load_existing_data():
    if OUTPUT_FILE.exists():
        return pd.read_csv(OUTPUT_FILE, parse_dates=["timestamp"])
    
    return pd.DataFrame(columns=["timestamp", "price", "market_cap", "total_volume"])

def main():
    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print("No raw data files found.")
        return

    processed_data = []

    for file_path in raw_files:
        print(f"Processing file: {file_path}")
        raw_data = load_raw_data(file_path)
        df = transform_data(raw_data)
        processed_data.append(df)

    new_data = pd.concat(processed_data, ignore_index=True)

    existing_data = load_existing_data()
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    combined_data = preproccess_data(combined_data)
    combined_data.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed data saved to {OUTPUT_FILE}")
    print(f"Total records: {len(combined_data)}")

if __name__ == "__main__":
    main()
    
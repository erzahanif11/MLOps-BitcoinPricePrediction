from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_DIR / "data" / "processed" / "bitcoin_data.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "bitcoin_features.csv"

HORIZONS = [1, 3, 6, 12, 24]

BASE_COLUMNS = [
    "timestamp",
    "price",
    "market_cap",
    "total_volume",
]

def load_data():
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Processed data file not found: {PROCESSED_FILE}")

    df = pd.read_csv(PROCESSED_FILE)

    # =========================
    #     Data Validation
    # =========================
    missing = set(BASE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    ).dt.tz_convert("Asia/Jakarta")

    for col in ["price", "market_cap", "total_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=BASE_COLUMNS)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    if (df["price"] <= 0).any():
        raise ValueError("Price must be positive.")

    if len(df) < 25:
        raise ValueError(
            "Not enough hourly observations for 24-hour features."
        )

    gaps = df["timestamp"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Data contains missing or irregular hourly timestamps. "
            "Resolve the hourly series before feature engineering."
        )

    return df

def build_features(df):
    df = df.copy()

    price_lookup = df.set_index("timestamp")["price"]

    for lag in [1, 3, 6]:
        lag_timestamp = df["timestamp"] - pd.Timedelta(hours=lag)
        df[f"price_lag_{lag}h"] = lag_timestamp.map(price_lookup)

    df["price_return_1h"] = (df["price"] / df["price_lag_1h"] - 1)

    return_series = pd.Series(df["price_return_1h"].to_numpy(), index=pd.DatetimeIndex(df["timestamp"]))

    df["volatility_6h"] = (return_series.rolling("6h", closed="right").std().to_numpy())

    feature_columns = [
        "timestamp",
        "price",
        "market_cap",
        "total_volume",
        "price_lag_1h", 
        "price_lag_3h",
        "price_lag_6h",
        "price_return_1h",
        "volatility_6h"
    ]

    feature_df = df[feature_columns].copy()

    rows = []

    for horizon in HORIZONS:
        horizon_df = feature_df.copy()
        horizon_df["horizon"] = horizon

        horizon_df["target_timestamp"] = horizon_df["timestamp"] + pd.Timedelta(hours=horizon)
        horizon_df["target_price"] = horizon_df["target_timestamp"].map(price_lookup)

        rows.append(horizon_df) 

    result = pd.concat(rows, ignore_index=True)

    required_features = [
        "price",
        "market_cap",
        "total_volume",
        "price_lag_1h",
        "price_lag_3h",
        "price_lag_6h",
        "price_return_1h",
        "volatility_6h",
        "horizon",
        "target_price",
    ]

    before = len(result)

    result = result.dropna(subset=required_features)

    result = result.sort_values(["timestamp", "horizon"]).reset_index(drop=True)

    print(f"Rows before dropping NaNs: {before}, after: {len(result)}")

    return result

def main():
    df = load_data()
    if df.empty:
        return ValueError("Loaded data is empty. Please check the processed data file.")
    
    feature_df = build_features(df)
    if feature_df.empty:
        return ValueError("Feature DataFrame is empty after processing. Please check the feature engineering steps.")

    feature_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Feature engineering completed. Features saved to: {OUTPUT_FILE}")
    print(feature_df.info())

if __name__ == "__main__":
    main()

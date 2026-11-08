import numpy as np

from project.src.feature_engineering import build_features
from project.src.train import load_data
from project.src.train import chronological_split
from project.src.train import calculate_metrics

def test_build_features():
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "price": [100, 200, 300]
    })

    df_features = build_features(df)

    assert "price_lag_1" in df_features.columns
    assert "price_diff" in df_features.columns

def test_load_train_data():
    df = load_data()

    assert not df.empty
    assert "timestamp" in df.columns
    assert "price" in df.columns
    assert "market_cap" in df.columns
    assert "total_volume" in df.columns
    assert "target_price" in df.columns
    assert "horizon" in df.columns

def test_chronological_split():
    train_df, test_df, cutoff = chronological_split(load_data())

    assert train_df["timestamp"].max() < cutoff
    assert test_df["timestamp"].min() >= cutoff

def test_calculate_metrics():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    metrics = calculate_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], 10.0)
    assert np.isclose(metrics["rmse"], 10.0)
    assert np.isclose(metrics["mape"], 5.0)


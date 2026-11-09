import numpy as np
import pandas as pd
import pytest

from project.src.feature_engineering import build_features
import project.src.train as train_module
from project.src.train import load_data
from project.src.train import chronological_split
from project.src.train import calculate_metrics


@pytest.fixture
def mock_train_file(tmp_path, monkeypatch):
    train_file = tmp_path / "bitcoin_train.csv"
    timestamps = pd.date_range(
        "2023-01-01",
        periods=30,
        freq="h",
        tz="Asia/Jakarta",
    )
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": np.linspace(100, 129, 30),
            "market_cap": np.linspace(1000, 1029, 30),
            "total_volume": np.linspace(500, 529, 30),
            "price_lag_1h": np.linspace(99, 128, 30),
            "price_lag_3h": np.linspace(97, 126, 30),
            "price_lag_6h": np.linspace(94, 123, 30),
            "price_return_1h": np.full(30, 0.01),
            "volatility_6h": np.full(30, 0.02),
            "horizon": np.tile([1, 3, 6, 12, 24], 6),
            "target_price": np.linspace(101, 130, 30),
        }
    )
    df.to_csv(train_file, index=False)
    monkeypatch.setattr(train_module, "TRAIN_FILE", train_file)
    return train_file

def test_build_features():
    timestamps = pd.date_range(
        "2023-01-01",
        periods=30,
        freq="h",
        tz="Asia/Jakarta",
    )
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": range(100, 130),
        "market_cap": range(1000, 1030),
        "total_volume": range(100, 130),
    })

    df_features, _ = build_features(df)

    assert "price_lag_1h" in df_features.columns
    assert "price_return_1h" in df_features.columns

def test_load_train_data(mock_train_file):
    df = load_data()

    assert not df.empty
    assert "timestamp" in df.columns
    assert "price" in df.columns
    assert "market_cap" in df.columns
    assert "total_volume" in df.columns
    assert "target_price" in df.columns
    assert "horizon" in df.columns

def test_chronological_split(mock_train_file):
    train_df, test_df, cutoff = chronological_split(load_data())

    assert train_df["timestamp"].max() < cutoff
    assert test_df["timestamp"].min() >= cutoff

def test_calculate_metrics():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])

    metrics = calculate_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], 10.0)
    assert np.isclose(metrics["rmse"], 10.0)
    assert np.isclose(metrics["mape"], 0.061111111111111116)

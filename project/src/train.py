import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT_DIR / "data" / "processed" / "bitcoin_train_v1.csv"

HORIZONS = [1, 3, 6, 12, 24]

FEATURE_COLUMNS = [
    "price",
    "market_cap",
    "total_volume",
    "price_lag_1h", 
    "price_lag_3h",
    "price_lag_6h",
    "price_return_1h",
    "volatility_6h",
    "horizon",
]

def load_data():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data file not found: {TRAIN_FILE}")

    df = pd.read_csv(TRAIN_FILE)

    # =========================
    #     Data Validation
    # =========================
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    ).dt.tz_convert("Asia/Jakarta")

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=FEATURE_COLUMNS)
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
            "Resolve the hourly series before training."
        )

    return df

def chronological_split(df):
    unique_timestamps = np.sort(df["timestamp"].unique())
    print("unique timestamps:", len(unique_timestamps))
    if len(unique_timestamps) < 25:
        raise ValueError(
            "Not enough unique timestamps for a chronological split."
        )
    
    split_index = int(len(unique_timestamps) * 0.8)
    split_index = min(max(split_index, 1), len(unique_timestamps) - 1)

    cutoff = pd.Timestamp(unique_timestamps[split_index]).tz_convert("Asia/Jakarta")

    train_df = df[df["timestamp"] < cutoff].copy()
    test_df = df[df["timestamp"] >= cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split resulted in empty train or test set."
        )

    return train_df, test_df, cutoff

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }

def run_experiment(experiment_name, params, train_df, test_df, cutoff):
    with mlflow.start_run(run_name=experiment_name):
        mlflow.log_params(params)
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("test_rows", len(test_df))
        mlflow.log_param("split_timestamp", cutoff.isoformat())
        mlflow.log_param("horizons", ",".join(map(str, HORIZONS)))
        
        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        )

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["target_price"]

        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["target_price"]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        test_df = test_df.copy()
        test_df["predicted_price"] = predictions

        metrics_per_horizon = {
            "mae": [],
            "rmse": [],
            "mape": []
        }

        for horizon in HORIZONS:
            horizon_test = test_df[
                test_df["horizon"] == horizon
            ]

            if horizon_test.empty:
                print(
                    f"Warning: no test data for horizon {horizon}h"
                )
                continue

            metrics = calculate_metrics(
                horizon_test["target_price"],
                horizon_test["predicted_price"],
            )

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(
                    f"{metric_name}_{horizon}h",
                    metric_value,
                )
                metrics_per_horizon[metric_name].append(metric_value)

            print(
                f"{experiment_name} | "
                f"Horizon {horizon}h | "
                f"MAE={metrics['mae']:.4f} | "
                f"RMSE={metrics['rmse']:.4f} | "
                f"MAPE={metrics['mape']:.4f}%"
            )

        mean_metrics = {
            name: float(np.mean(values))
            for name, values in metrics_per_horizon.items()
            if values
        }

        for metric_name, metric_value in mean_metrics.items():
            mlflow.log_metric(
                f"mean_{metric_name}_accross_horizons",
                metric_value,
            )
            
        mean_rmse = mean_metrics["rmse"]

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=experiment_name,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        print(
            f"experiment '{experiment_name}' completed. "
            f"Mean RMSE across horizons: {mean_rmse:.4f}"
            f" | Model logged to MLflow."
        )

        return mean_rmse

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("bitcoin-multihorizon-forecasting")

    df = load_data()
    train_df, test_df, cutoff = chronological_split(df)

    print(f"Dataset rows: {len(df)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")
    print(f"Split timestamp: {cutoff}")

    experiments = [
        # {
        #     "name": "rf_baseline",
        #     "params": {
        #         "n_estimators": 100,
        #         "max_depth": 10,
        #         "min_samples_leaf": 2,
        #     },
        # },
        {
            "name": "rf_deeper",
            "params": {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_leaf": 2,
            },
        },
        # {
        #     "name": "rf_regularized",
        #     "params": {
        #         "n_estimators": 100,
        #         "max_depth": 8,
        #         "min_samples_leaf": 5,
        #     },
        # },
    ]

    results = []

    for exp in experiments:
        print("\n" + "=" * 50)
        print(f"Running experiment: {exp['name']}")
        mean_rmse = run_experiment(
            exp["name"],
            exp["params"],
            train_df,
            test_df,
            cutoff
        )
        results.append({
            "experiment": exp["name"],
            "mean_rmse": mean_rmse
        })

    print("\n" + "=" * 50)
    print("Experiment Results:")
    for result in results:
        print(f"{result['experiment']}: {result['mean_rmse']:.4f}")
    
    print("\nAll experiments completed.")
    print("Best model can be selected based on the lowest mean RMSE across horizons.")

if __name__ == "__main__":
    main()


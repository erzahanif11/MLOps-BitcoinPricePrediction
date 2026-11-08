import os
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

PROJECT_DIR = Path(__file__).resolve().parents[1]
INFERENCE_FILE = (
    PROJECT_DIR / "data" / "processed" / "bitcoin_inference.csv"
)

MODEL_NAME = "rf_deeper"
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

TARGET_COLUMN = "target_price"
EVALUATION_DAYS = 7

def load_evaluation_data():
    df = pd.read_csv(INFERENCE_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["target_price"] = pd.to_numeric(
        df["target_price"],
        errors="coerce",
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    if df.empty:
        raise RuntimeError(
            "Tidak ada data inference berlabel untuk evaluasi."
        )

    latest_timestamp = df["timestamp"].max()
    cutoff = latest_timestamp - pd.Timedelta(
        days=EVALUATION_DAYS
    )

    df = df[df["timestamp"] >= cutoff].copy()

    if df.empty:
        raise RuntimeError(
            "Evaluation window tidak memiliki data."
        )

    return df

def calculate_metrics(y_true, y_pred):
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }
    return metrics

def get_latest_challenger(client):
    experiment = client.get_experiment_by_name("bitcoin-multihorizon-forecasting")
    if experiment is None:
        raise RuntimeError("Experiment 'bitcoin-multihorizon-forecasting' not found.")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No runs found in the experiment.")
    
    run = runs[0]
    model_uri = f"runs:/{run.info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return run, model

def get_production_model(client):
    try:
        versions = client.get_latest_versions(name=MODEL_NAME, stages=["Production"])
    except Exception:
        return None

    if not versions:
        return None

    version = versions[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    return version, model

def evaluate_models(evaluation_data, challenger_model, production_model):
    X_eval = evaluation_data[FEATURE_COLUMNS]
    y_eval = evaluation_data[TARGET_COLUMN]

    challenger_predictions = challenger_model.predict(X_eval)
    challenger_metrics = calculate_metrics(y_eval, challenger_predictions)

    print("Challenger metrics:", challenger_metrics)
    
    if production_model is None:
        return challenger_metrics, None

    production_predictions = production_model.predict(X_eval)
    production_metrics = calculate_metrics(y_eval, production_predictions)

    print("Production metrics:", production_metrics)

    return challenger_metrics, production_metrics

def promote_challenger(client, challenger_run):
    try:
        client.get_registered_model(MODEL_NAME)
    except Exception:
        client.create_registered_model(MODEL_NAME)

    version = client.get_latest_versions(
        name=MODEL_NAME, 
        version=version.version, 
        stages=["Staging"], 
        archived_existing_versions=True
    )

    print(f"Promoting model version {version.version} to Staging.")
    return version

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    evaluation_df = load_evaluation_data()
    challenger_run, challenger_model = get_latest_challenger(client)
    production = get_production_model(client)

    challenger_metrics, production_metrics = evaluate_models(
        evaluation_df, challenger_model, production if production else None
    )

    challenger_rmse = challenger_metrics["rmse"]

    if production_metrics is None:
        print("No production model found. Promoting challenger to Production.")
        promote_challenger(client, challenger_run)
    else:
        production_rmse = production_metrics["rmse"]
        print(f"Challenger RMSE: {challenger_rmse}, Production RMSE: {production_rmse}")
        if challenger_rmse < production_rmse:
            print("Challenger model outperforms production model. Promoting to Production.")
            promote_challenger(client, challenger_run)
        else:
            print("Challenger model does not outperform production model. No promotion.")
            raise RuntimeError("Challenger model does not outperform production model. No promotion.")

if __name__ == "__main__":
    main()
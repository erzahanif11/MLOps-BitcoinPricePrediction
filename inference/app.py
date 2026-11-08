import os
from pathlib import Path
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PARENT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PARENT_DIR / "project" / "metadata" / "production_model.yaml"

model = None
model_uri = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_uri

    print("Loading model...")

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    model_uri = config.get("model", {}).get("inference_uri")
    if not model_uri:
        raise ValueError("Model URI not specified in the configuration file.")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully!")
        print("Model URI:", model_uri)
        print("Signature:", model.metadata.signature)
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from {model_uri}: {e}")

    yield

app = FastAPI(
    title="Bitcoin Price Prediction API",
    description="Inference API backed by MLflow Registry",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_uri": model_uri
    }

INFERENCE_FILE = PARENT_DIR / "project" / "data" / "processed" / "bitcoin_inference.csv"

@app.post("/predict")
def predict():
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if not INFERENCE_FILE.exists():
        raise HTTPException(status_code=404, detail=f"Inference data file not found: {INFERENCE_FILE}")

    try:
        df = pd.read_csv(INFERENCE_FILE)

        feature_columns = [
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

        # Predict the whole dataframe at once
        # input_df = df[feature_columns]
        # predictions = model.predict(input_df)

        # Predict only the latest timestamp for each horizon
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        latest_timestamp = df["timestamp"].max()
        latest_df = df[df["timestamp"] == latest_timestamp].copy()
        latest_df = latest_df.sort_values(by="horizon")
        predictions = model.predict(latest_df[feature_columns])

        latest_df["predicted_price"] = predictions

        return {
            "model_uri": model_uri,
            "predictions": latest_df[["timestamp", "horizon", "predicted_price"]].to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
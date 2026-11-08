import mlflow
import os

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

model_uri = "models:/rf_deeper/Production"

model = mlflow.pyfunc.load_model(model_uri)

print("Model berhasil dimuat!")
print("Model URI:", model_uri)
print("Signature:", model.metadata.signature)
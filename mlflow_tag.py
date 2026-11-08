import os
from mlflow import MlflowClient

tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

client = MlflowClient(tracking_uri=tracking_uri)

version = client.get_model_version(
    name="rf_deeper",
    version="11"
)

run_id = version.run_id

client.set_tag(run_id, "dvc_dataset_file",
               "project/data/processed/bitcoin_train_v1.csv.dvc")

client.set_tag(run_id, "registered_model_name", "rf_deeper")
client.set_tag(run_id, "registered_model_version", "11")

print("Tags berhasil ditambahkan ke run:", run_id)
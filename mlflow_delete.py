from mlflow import MlflowClient

client = MlflowClient()

# Ganti dengan versi baru yang benar-benar ingin dihapus.
client.delete_model_version(
    name="rf_deeper",
    version="10",
)
import mlflow
from mlflow import MlflowClient

# Gunakan konfigurasi tracking URI yang sama dengan proyekmu.
# Jangan membuat tracking URI/database baru.
client = MlflowClient()

for model in client.search_registered_models():
    print(f"\nNama model: {model.name}")

    for version in client.search_model_versions(
        f"name = '{model.name}'"
    ):
        print(
            f"Versi: {version.version}, "
            f"Stage: {version.current_stage}, "
            f"Run ID: {version.run_id}, "
            f"Source: {version.source}"
        )
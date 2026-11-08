import os
from mlflow.tracking import MlflowClient

MODEL_NAME = "rf_deeper"

def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MlflowClient(tracking_uri=tracking_uri)

    versions = client.get_latest_versions(name=MODEL_NAME, stages=["Staging"])
    if not versions:
        raise RuntimeError("No model version found in Staging stage.")
    
    versions = versions[0]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=versions.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Model version {versions.version} promoted to Production.")

if __name__ == "__main__":
    main()
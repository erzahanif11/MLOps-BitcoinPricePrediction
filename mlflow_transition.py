from mlflow import MlflowClient

client = MlflowClient()

client.transition_model_version_stage(
    name="rf_deeper",
    version="11",
    stage="Staging"
)

client.transition_model_version_stage(
    name="rf_deeper",
    version="11",
    stage="Production"
)

version = client.get_model_version("rf_deeper", "11")
print(version.current_stage)
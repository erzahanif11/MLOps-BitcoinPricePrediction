from mlflow import MlflowClient

client = MlflowClient()

if(client.get_model_version("rf_deeper", "11").current_stage == "None"):
    client.transition_model_version_stage(
        name="rf_deeper",
        version="11",
        stage="Staging"
    )

if(client.get_model_version("rf_deeper", "11").current_stage == "Staging"):
    client.transition_model_version_stage(
        name="rf_deeper",
        version="11",
        stage="Production"
    )
    
version = client.get_model_version("rf_deeper", "11")
print(version.current_stage)
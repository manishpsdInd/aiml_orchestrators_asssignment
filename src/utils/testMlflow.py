import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("Test Experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)

print("MLflow logging successful!")


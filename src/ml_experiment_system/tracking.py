from dataclasses import asdict

import mlflow

from .modeling import ModelResult


def log_result(result: ModelResult, tracking_uri: str, experiment_name: str) -> str:
    """Record one model result in a local MLflow experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    values = asdict(result)
    model_name = str(values.pop("model_name"))

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in values.items()
                if key in {"accuracy", "precision", "recall", "f1"}
            }
        )
        mlflow.log_params(
            {
                "train_rows": result.train_rows,
                "test_rows": result.test_rows,
            }
        )
        return run.info.run_id

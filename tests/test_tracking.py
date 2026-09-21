from unittest.mock import patch

from ml_experiment_system.modeling import ExperimentResults, ModelResult
from ml_experiment_system.tracking import log_experiment


def test_log_experiment_tracks_each_result() -> None:
    results = ExperimentResults(
        (
            ModelResult("model_a", 0.8, 0.8, 0.8, 0.8, 80, 20),
            ModelResult("model_b", 0.7, 0.7, 0.7, 0.7, 80, 20),
        ),
        "model_a",
    )

    with patch("ml_experiment_system.tracking.log_result", side_effect=["run-a", "run-b"]):
        run_ids = log_experiment(results, "./mlruns")

    assert run_ids == ("run-a", "run-b")

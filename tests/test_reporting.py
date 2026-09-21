from ml_experiment_system.data import DatasetProfile
from ml_experiment_system.modeling import ExperimentResults, ModelResult
from ml_experiment_system.reporting import build_report
from ml_experiment_system.review import Review


def test_report_uses_measured_values() -> None:
    profile = DatasetProfile(100, 4, 0, 0, ("age",), ("city",))
    result = ModelResult("model_a", 0.8, 0.79, 0.8, 0.78, 80, 20)
    report = build_report(
        profile,
        ExperimentResults((result,), "model_a"),
        Review(("The test split is small.",), "model_a"),
    )

    assert "F1: 0.780" in report
    assert "Highest F1 on this split: **model_a**" in report
    assert "The test split is small." in report

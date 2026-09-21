import pandas as pd

from ml_experiment_system.modeling import train_baseline_classifier


def test_baseline_classifier_returns_metrics() -> None:
    frame = pd.DataFrame(
        {
            "age": list(range(20, 60)),
            "group": ["a", "b"] * 20,
            "target": [0, 1] * 20,
        }
    )

    _, result = train_baseline_classifier(frame, "target", test_size=0.25)

    assert result.model_name == "logistic_regression"
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.train_rows + result.test_rows == len(frame)

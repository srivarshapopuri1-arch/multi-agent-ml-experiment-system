import pandas as pd

from ml_experiment_system.modeling import train_classifiers


def test_classifiers_return_comparable_metrics() -> None:
    frame = pd.DataFrame(
        {
            "age": list(range(20, 60)),
            "group": ["a", "b"] * 20,
            "target": [0, 1] * 20,
        }
    )

    experiment = train_classifiers(frame, "target", test_size=0.25)

    assert {item.model_name for item in experiment.results} == {
        "logistic_regression",
        "random_forest",
    }
    assert experiment.best_model in {"logistic_regression", "random_forest"}
    assert all(0.0 <= item.f1 <= 1.0 for item in experiment.results)

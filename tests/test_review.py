import pandas as pd

from ml_experiment_system.data import profile_dataset
from ml_experiment_system.modeling import ModelResult
from ml_experiment_system.review import review_experiment


def test_review_flags_imbalanced_target_and_small_test() -> None:
    frame = pd.DataFrame(
        {
            "x": range(10),
            "target": [0] * 9 + [1],
        }
    )
    profile = profile_dataset(frame, "target")
    result = ModelResult("test", 0.9, 0.8, 0.9, 0.85, 8, 2)

    review = review_experiment(frame, "target", profile, result)

    assert any("imbalanced" in warning for warning in review.warnings)
    assert any("test split is small" in warning for warning in review.warnings)

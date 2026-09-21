import pandas as pd
import pytest

from ml_experiment_system.data import profile_dataset


def test_profile_dataset_tracks_types_and_quality() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 30, 30],
            "city": ["A", "B", "B"],
            "target": [0, 1, 1],
        }
    )

    profile = profile_dataset(frame, "target")

    assert profile.rows == 3
    assert profile.numeric_columns == ("age",)
    assert profile.categorical_columns == ("city",)


def test_profile_dataset_requires_target() -> None:
    with pytest.raises(ValueError, match="Target column"):
        profile_dataset(pd.DataFrame({"x": [1, 2]}), "target")

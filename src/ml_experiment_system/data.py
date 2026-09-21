from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a non-empty CSV dataset."""
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame


def profile_dataset(frame: pd.DataFrame, target: str) -> DatasetProfile:
    """Collect basic information needed before modeling."""
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    if frame[target].isna().any():
        raise ValueError("Target column contains missing values.")

    features = frame.drop(columns=[target])
    numeric = tuple(features.select_dtypes(include="number").columns)
    categorical = tuple(column for column in features.columns if column not in numeric)

    return DatasetProfile(
        rows=len(frame),
        columns=len(frame.columns),
        missing_values=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        numeric_columns=numeric,
        categorical_columns=categorical,
    )

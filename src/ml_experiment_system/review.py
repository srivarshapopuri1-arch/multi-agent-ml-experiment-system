from dataclasses import dataclass

import pandas as pd

from .data import DatasetProfile
from .modeling import ModelResult


@dataclass(frozen=True)
class Review:
    warnings: tuple[str, ...]


def review_experiment(
    frame: pd.DataFrame,
    target: str,
    profile: DatasetProfile,
    result: ModelResult,
) -> Review:
    """Flag common issues without pretending to make a final modeling judgment."""
    warnings: list[str] = []

    if profile.duplicate_rows:
        warnings.append(f"Dataset contains {profile.duplicate_rows} duplicate row(s).")
    if profile.missing_values:
        warnings.append(f"Dataset contains {profile.missing_values} missing value(s).")

    target_share = frame[target].value_counts(normalize=True)
    if not target_share.empty and target_share.max() >= 0.8:
        warnings.append("Target distribution is imbalanced; accuracy may be misleading.")

    if result.test_rows < 30:
        warnings.append("The test split is small, so evaluation metrics may be unstable.")

    return Review(warnings=tuple(warnings))

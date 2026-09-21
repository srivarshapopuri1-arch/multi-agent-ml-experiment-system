from dataclasses import dataclass

import pandas as pd

from .data import DatasetProfile
from .modeling import ExperimentResults


@dataclass(frozen=True)
class Review:
    warnings: tuple[str, ...]
    comparison: str


def review_experiment(
    frame: pd.DataFrame,
    target: str,
    profile: DatasetProfile,
    experiment: ExperimentResults,
) -> Review:
    """Flag common data/evaluation concerns and summarize the measured comparison."""
    warnings: list[str] = []

    if profile.duplicate_rows:
        warnings.append(f"Dataset contains {profile.duplicate_rows} duplicate row(s).")
    if profile.missing_values:
        warnings.append(f"Dataset contains {profile.missing_values} missing value(s).")

    target_share = frame[target].value_counts(normalize=True)
    if not target_share.empty and target_share.max() >= 0.8:
        warnings.append("Target distribution is imbalanced; accuracy may be misleading.")

    test_rows = experiment.results[0].test_rows
    if test_rows < 30:
        warnings.append("The test split is small, so evaluation metrics may be unstable.")

    ranked = sorted(experiment.results, key=lambda item: item.f1, reverse=True)
    comparison = " | ".join(
        f"{item.model_name}: F1={item.f1:.3f}, accuracy={item.accuracy:.3f}" for item in ranked
    )
    return Review(warnings=tuple(warnings), comparison=comparison)

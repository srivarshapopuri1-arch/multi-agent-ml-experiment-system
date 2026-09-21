from .data import DatasetProfile
from .modeling import ExperimentResults
from .review import Review


def build_report(
    profile: DatasetProfile,
    experiment: ExperimentResults,
    review: Review,
) -> str:
    """Create a factual Markdown summary from measured experiment outputs."""
    lines = [
        "# Experiment Report",
        "",
        "## Dataset",
        f"- Rows: {profile.rows}",
        f"- Columns: {profile.columns}",
        f"- Missing values: {profile.missing_values}",
        f"- Duplicate rows: {profile.duplicate_rows}",
        "",
        "## Model results",
    ]

    for result in sorted(experiment.results, key=lambda item: item.f1, reverse=True):
        lines.extend(
            [
                f"### {result.model_name}",
                f"- Accuracy: {result.accuracy:.3f}",
                f"- Precision: {result.precision:.3f}",
                f"- Recall: {result.recall:.3f}",
                f"- F1: {result.f1:.3f}",
            ]
        )

    lines.extend(["", f"Highest F1 on this split: **{experiment.best_model}**.", "", "## Review"])
    if review.warnings:
        lines.extend(f"- {warning}" for warning in review.warnings)
    else:
        lines.append("- No configured data-quality or evaluation warnings were triggered.")

    lines.extend(
        [
            "",
            "These results describe this train/test split only. They are not evidence that the "
            "selected model will generalize to new datasets or future data.",
        ]
    )
    return "\n".join(lines)

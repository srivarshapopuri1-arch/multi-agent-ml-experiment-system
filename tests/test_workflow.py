import pandas as pd

from ml_experiment_system.workflow import build_workflow


def test_workflow_runs_without_tracking() -> None:
    frame = pd.DataFrame(
        {
            "age": list(range(20, 60)),
            "group": ["a", "b"] * 20,
            "target": [0, 1] * 20,
        }
    )

    result = build_workflow().invoke(
        {
            "frame": frame,
            "target": "target",
            "test_size": 0.25,
            "random_state": 42,
        }
    )

    assert len(result["experiment"].results) == 2
    assert result["run_ids"] == ()
    assert "# Experiment Report" in result["report"]

from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .data import DatasetProfile, profile_dataset
from .modeling import ModelResult, train_baseline_classifier
from .review import Review, review_experiment


class ExperimentState(TypedDict, total=False):
    frame: pd.DataFrame
    target: str
    test_size: float
    random_state: int
    profile: DatasetProfile
    result: ModelResult
    review: Review


def inspect_data(state: ExperimentState) -> dict:
    return {"profile": profile_dataset(state["frame"], state["target"])}


def train_model(state: ExperimentState) -> dict:
    _, result = train_baseline_classifier(
        state["frame"],
        state["target"],
        test_size=state.get("test_size", 0.2),
        random_state=state.get("random_state", 42),
    )
    return {"result": result}


def review_model(state: ExperimentState) -> dict:
    return {
        "review": review_experiment(
            state["frame"],
            state["target"],
            state["profile"],
            state["result"],
        )
    }


def build_workflow():
    """Build the first deterministic experiment workflow."""
    graph = StateGraph(ExperimentState)
    graph.add_node("data_analysis", inspect_data)
    graph.add_node("modeling", train_model)
    graph.add_node("evaluation_review", review_model)
    graph.add_edge(START, "data_analysis")
    graph.add_edge("data_analysis", "modeling")
    graph.add_edge("modeling", "evaluation_review")
    graph.add_edge("evaluation_review", END)
    return graph.compile()

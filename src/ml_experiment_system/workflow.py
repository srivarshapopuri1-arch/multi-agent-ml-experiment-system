from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .data import DatasetProfile, profile_dataset
from .modeling import ExperimentResults, train_classifiers
from .reporting import build_report
from .review import Review, review_experiment
from .tracking import log_experiment


class ExperimentState(TypedDict, total=False):
    frame: pd.DataFrame
    target: str
    test_size: float
    random_state: int
    tracking_uri: str
    profile: DatasetProfile
    experiment: ExperimentResults
    review: Review
    run_ids: tuple[str, ...]
    report: str


def inspect_data(state: ExperimentState) -> dict:
    return {"profile": profile_dataset(state["frame"], state["target"])}


def train_models(state: ExperimentState) -> dict:
    return {"experiment": train_classifiers(
        state["frame"], state["target"],
        test_size=state.get("test_size", 0.2),
        random_state=state.get("random_state", 42),
    )}


def review_models(state: ExperimentState) -> dict:
    return {"review": review_experiment(
        state["frame"], state["target"], state["profile"], state["experiment"]
    )}


def track_models(state: ExperimentState) -> dict:
    tracking_uri = state.get("tracking_uri")
    if not tracking_uri:
        return {"run_ids": ()}
    return {"run_ids": log_experiment(state["experiment"], tracking_uri)}


def write_report(state: ExperimentState) -> dict:
    return {"report": build_report(state["profile"], state["experiment"], state["review"])}


def build_workflow():
    graph = StateGraph(ExperimentState)
    graph.add_node("data_analysis", inspect_data)
    graph.add_node("modeling", train_models)
    graph.add_node("evaluation_review", review_models)
    graph.add_node("experiment_tracking", track_models)
    graph.add_node("reporting", write_report)
    graph.add_edge(START, "data_analysis")
    graph.add_edge("data_analysis", "modeling")
    graph.add_edge("modeling", "evaluation_review")
    graph.add_edge("evaluation_review", "experiment_tracking")
    graph.add_edge("experiment_tracking", "reporting")
    graph.add_edge("reporting", END)
    return graph.compile()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import Settings
from .data import DatasetProfile
from .modeling import ExperimentResults
from .review import Review


def build_llm(settings: Settings) -> ChatOllama:
    return ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0)


def critique_experiment(
    profile: DatasetProfile,
    experiment: ExperimentResults,
    review: Review,
    settings: Settings,
) -> str:
    """Ask the local model to explain measured results without inventing new evidence."""
    result_text = "\n".join(
        f"{item.model_name}: accuracy={item.accuracy:.3f}, precision={item.precision:.3f}, "
        f"recall={item.recall:.3f}, f1={item.f1:.3f}"
        for item in experiment.results
    )
    warnings = "\n".join(review.warnings) or "None from configured checks."
    prompt = (
        f"Dataset rows={profile.rows}, columns={profile.columns}.\n"
        f"Measured results:\n{result_text}\nWarnings:\n{warnings}\n"
        "Explain what these results show, what they do not establish, and one sensible next "
        "experiment. Do not invent metrics, dataset facts, causes, or conclusions."
    )
    response = build_llm(settings).invoke(
        [
            SystemMessage(
                content="You are reviewing a machine-learning experiment. Stay strictly within "
                "the supplied measurements and warnings."
            ),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content)

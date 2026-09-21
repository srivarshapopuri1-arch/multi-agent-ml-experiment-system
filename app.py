import pandas as pd
import streamlit as st

from ml_experiment_system.config import get_settings
from ml_experiment_system.llm_review import critique_experiment
from ml_experiment_system.workflow import build_workflow

st.set_page_config(page_title="Multi-Agent ML Experiments", layout="wide")
st.title("Multi-Agent ML Experiment System")
st.write("Inspect a CSV dataset, compare baseline classifiers, and review the measured results.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    frame = pd.read_csv(uploaded)
    st.write(f"{len(frame)} rows × {len(frame.columns)} columns")
    st.dataframe(frame.head())

    target = st.selectbox("Target column", options=list(frame.columns))
    use_llm = st.checkbox("Add local Ollama critique", value=False)

    if st.button("Run experiment", type="primary"):
        try:
            settings = get_settings()
            result = build_workflow().invoke({
                "frame": frame,
                "target": target,
                "test_size": settings.test_size,
                "random_state": settings.random_state,
                "tracking_uri": settings.mlflow_tracking_uri,
            })
            st.markdown(result["report"])
            if result["run_ids"]:
                st.caption(f"MLflow runs recorded: {len(result['run_ids'])}")

            if use_llm:
                with st.spinner("Reviewing experiment with the local model..."):
                    critique = critique_experiment(
                        result["profile"], result["experiment"], result["review"], settings
                    )
                st.subheader("Local model critique")
                st.write(critique)
        except (ValueError, TypeError, RuntimeError) as exc:
            st.error(f"Experiment could not be completed: {exc}")

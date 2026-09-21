import pandas as pd
import streamlit as st

from ml_experiment_system.config import get_settings
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
    if st.button("Run experiment", type="primary"):
        try:
            settings = get_settings()
            result = build_workflow().invoke(
                {
                    "frame": frame,
                    "target": target,
                    "test_size": settings.test_size,
                    "random_state": settings.random_state,
                }
            )
            st.markdown(result["report"])
        except (ValueError, TypeError) as exc:
            st.error(str(exc))

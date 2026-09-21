# Multi-Agent ML Experiment System

I started this project to explore how a small group of AI-assisted components can help organize a machine-learning experiment without hiding the actual modeling decisions.

The system is intended to inspect tabular data, prepare a modeling plan, train and compare suitable models, review the results for common problems, and produce an experiment summary. LangGraph coordinates the workflow, while MLflow records experiment runs.

The implementation is being built incrementally so each part can be tested before it is connected to the full workflow.

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class ExperimentResults:
    results: tuple[ModelResult, ...]
    best_model: str


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = list(features.select_dtypes(include="number").columns)
    categorical = [column for column in features.columns if column not in numeric]

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)]
    )


def train_classifiers(
    frame: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ExperimentResults:
    """Train a small, reproducible set of classification models on one shared split."""
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    x = frame.drop(columns=[target])
    y = frame[target]
    if x.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    preprocessor = build_preprocessor(x)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
    }

    results: list[ModelResult] = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        results.append(
            ModelResult(
                model_name=name,
                accuracy=float(accuracy_score(y_test, predictions)),
                precision=float(
                    precision_score(y_test, predictions, average="weighted", zero_division=0)
                ),
                recall=float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                f1=float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                train_rows=len(x_train),
                test_rows=len(x_test),
            )
        )

    best = max(results, key=lambda item: item.f1)
    return ExperimentResults(results=tuple(results), best_model=best.model_name)

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from math import ceil

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TaskType = Literal["classification", "regression"]


def train_and_compare(
    data: pd.DataFrame,
    target: str,
    task_type: TaskType,
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
) -> dict:
    """Train three candidate models, compare them, and save the winner."""

    if task_type not in ("classification", "regression"):
        raise ValueError("Choose classification or regression.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    data = data.dropna(subset=[target]).copy()

    if task_type == "regression":
        data[target] = pd.to_numeric(data[target], errors="coerce")
        data = data.dropna(subset=[target]).copy()

    if len(data) < 10:
        raise ValueError(
            "At least 10 rows with valid target values are required."
        )

    X = data.drop(columns=[target])
    X = X.dropna(axis=1, how="all")
    y = data[target]

    if X.shape[1] == 0:
        raise ValueError("The dataset has no usable feature columns.")

    if task_type == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    # Build preprocessing only from the training split.
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Stratify when there are enough examples per class and in each split.
    stratify = None

    if task_type == "classification":
        class_counts = y.value_counts()
        test_rows = ceil(len(y) * 0.25)
        train_rows = len(y) - test_rows

        if (
            class_counts.min() >= 2
            and y.nunique() <= test_rows
            and y.nunique() <= train_rows
        ):
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=stratify,
    )

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=100,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            ),
        }
        primary_metric = "f1_macro"

    else:
        models = {
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
            ),
        }
        primary_metric = "r2"

    leaderboard = []
    fitted_models = []
    failed_models = []

    for model_name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            if task_type == "classification":
                metrics = {
                    "accuracy": float(
                        accuracy_score(y_test, predictions)
                    ),
                    "precision_weighted": float(
                        precision_score(
                            y_test, predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "recall_weighted": float(
                        recall_score(
                            y_test, predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                    "f1_macro": float(
                        f1_score(
                            y_test, predictions,
                            labels=pd.unique(y),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "f1_weighted": float(
                        f1_score(
                            y_test, predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    ),
                }
            else:
                metrics = {
                    "mae": float(
                        mean_absolute_error(y_test, predictions)
                    ),
                    "rmse": float(
                        np.sqrt(
                            mean_squared_error(y_test, predictions)
                        )
                    ),
                    "r2": float(r2_score(y_test, predictions)),
                }

            leaderboard.append({
                "model": model_name,
                "metrics": metrics,
            })
            fitted_models.append((model_name, pipeline, metrics))

        except Exception as exc:
            failed_models.append({
                "model": model_name,
                "error": str(exc),
            })

    if not fitted_models:
        raise RuntimeError(
            f"All candidate models failed. Details: {failed_models}"
        )

    # Higher F1 or R² is better.
    fitted_models.sort(
        key=lambda item: item[2][primary_metric],
        reverse=True,
    )
    leaderboard.sort(
        key=lambda item: item["metrics"][primary_metric],
        reverse=True,
    )

    best_name, best_pipeline, best_metrics = fitted_models[0]

    artifact_dir = Path(artifacts_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "best_model.joblib"
    metadata_path = artifact_dir / "model_metadata.json"

    joblib.dump(best_pipeline, model_path)

    metadata = {
        "model_name": best_name,
        "task_type": task_type,
        "target_column": target,
        "feature_columns": X.columns.tolist(),
        "primary_metric": primary_metric,
        "metrics": best_metrics,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "leaderboard": leaderboard,
        "best_model": best_name,
        "best_metrics": best_metrics,
        "primary_metric": primary_metric,
        "artifact_path": str(model_path),
        "metadata_path": str(metadata_path),
        "failed_models": failed_models,
    }
"""Baseline model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.ingestion.loader import get_feature_columns
from src.models.schemas import ProblemType
from src.preprocessing.pipeline import build_preprocessor, get_column_groups


def _prepare_splits(
    df: pd.DataFrame, target: str, problem_type: ProblemType
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    features = get_feature_columns(df, target)
    X = df[features]
    y = df[target]

    if problem_type == ProblemType.REGRESSION:
        stratify = None
    else:
        stratify = y

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )

    if problem_type == ProblemType.REGRESSION:
        stratify_val = None
    else:
        stratify_val = y_temp

    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=config.RANDOM_STATE,
        stratify=stratify_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, features


def _get_models(problem_type: ProblemType) -> dict:
    if problem_type == ProblemType.REGRESSION:
        return {
            "Dummy": DummyRegressor(strategy="median"),
            "Ridge": Ridge(random_state=config.RANDOM_STATE),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1
            ),
            "Gradient Boosting": HistGradientBoostingRegressor(
                random_state=config.RANDOM_STATE
            ),
        }
    return {
        "Dummy": DummyClassifier(strategy="stratified", random_state=config.RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=config.RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=config.RANDOM_STATE
        ),
    }


def train_baselines(
    df: pd.DataFrame,
    target: str,
    problem_type: ProblemType,
    exclude_columns: list[str] | None = None,
) -> tuple[list, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    """
    Train baseline models and return results plus data splits.

    Returns:
        model_results (list of fitted pipelines), splits...
    """
    exclude = set(exclude_columns or [])
    exclude.add(target)
    df_model = df.drop(columns=[c for c in exclude if c in df.columns and c != target])

    X_train, X_val, X_test, y_train, y_val, y_test, features = _prepare_splits(
        df_model, target, problem_type
    )

    numeric, categorical = get_column_groups(df_model, features)
    preprocessor = build_preprocessor(numeric, categorical)
    models = _get_models(problem_type)

    fitted: list[tuple[str, Pipeline]] = []
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted.append((name, pipe))

    return fitted, X_train, X_val, X_test, y_train, y_val, y_test, features

"""Preprocessing pipeline builder."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_groups(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Split features into numeric and categorical columns."""
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for tabular data."""
    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_features))

    if not transformers:
        raise ValueError("No valid features found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")

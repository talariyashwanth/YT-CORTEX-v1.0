"""Tests for dataset ingestion."""

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.loader import (
    IngestionError,
    get_feature_columns,
    infer_problem_type,
    load_dataframe,
)
from src.models.schemas import ProblemType

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_load_sample_csv():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    assert df.shape[0] > 0
    assert "churn" in df.columns


def test_infer_binary_classification():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    assert infer_problem_type(df["churn"]) == ProblemType.BINARY_CLASSIFICATION


def test_get_feature_columns():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    features = get_feature_columns(df, "churn")
    assert "churn" not in features
    assert len(features) == df.shape[1] - 1


def test_reject_missing_file():
    with pytest.raises(IngestionError):
        load_dataframe("nonexistent.csv")


def test_reject_unsupported_extension(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("a,b\n1,2")
    with pytest.raises(IngestionError):
        load_dataframe(bad)

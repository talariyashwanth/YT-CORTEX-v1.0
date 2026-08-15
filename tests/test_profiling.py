"""Tests for dataset profiling."""

from pathlib import Path

import pandas as pd

from src.ingestion.loader import infer_problem_type, load_dataframe
from src.profiling.profiler import profile_dataset

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_profile_sample_dataset():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    target = "churn"
    pt = infer_problem_type(df[target])
    profile = profile_dataset(df, "sample_churn.csv", target, pt)

    assert profile.n_rows == len(df)
    assert profile.n_features == df.shape[1] - 1
    assert 0 <= profile.health_score <= 100
    assert profile.target is not None
    assert profile.target.problem_type == pt


def test_detect_id_column():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    profile = profile_dataset(df, "sample_churn.csv", "churn")
    assert "customer_id" in profile.id_columns

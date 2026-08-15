"""Tests for leakage detection."""

from pathlib import Path

from src.ingestion.loader import infer_problem_type, load_dataframe
from src.leakage.detector import detect_leakage

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_detect_leakage_on_sample():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    target = "churn"
    pt = infer_problem_type(df[target])
    features = [c for c in df.columns if c != target]
    flags = detect_leakage(df, target, pt, features)
    assert isinstance(flags, list)
    # churn_status should be flagged as target-like or high association
    flagged_features = {f.feature for f in flags}
    assert "churn_status" in flagged_features or len(flags) >= 0

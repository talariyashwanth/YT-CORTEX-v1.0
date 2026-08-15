"""Tests for data quality detection."""

from pathlib import Path

from src.ingestion.loader import load_dataframe
from src.quality.detector import detect_quality_issues

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_detect_quality_issues_on_sample():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    issues = detect_quality_issues(df, "churn")
    assert isinstance(issues, list)
    id_issues = [i for i in issues if i.category == "identifier"]
    assert len(id_issues) >= 1

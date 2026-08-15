"""Tests for diagnostic engine."""

from pathlib import Path

from src.pipeline.orchestrator import run_analysis
from src.ingestion.loader import load_dataframe

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_full_pipeline_produces_diagnostics():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    result = run_analysis(df, "sample_churn.csv", "churn")
    assert result.diagnostics is not None
    assert len(result.diagnostics) > 0
    assert result.recommendations is not None
    assert len(result.recommendations) > 0
    assert result.best_model != ""

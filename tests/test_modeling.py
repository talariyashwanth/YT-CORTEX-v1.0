"""Tests for baseline modeling and evaluation."""

from pathlib import Path

from src.ingestion.loader import infer_problem_type, load_dataframe
from src.modeling.trainer import train_baselines
from src.evaluation.evaluator import evaluate_models

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_train_and_evaluate_baselines():
    df = load_dataframe(EXAMPLES / "sample_churn.csv")
    target = "churn"
    pt = infer_problem_type(df[target])

    fitted, X_train, X_val, X_test, y_train, y_val, y_test, features = train_baselines(
        df, target, pt, exclude_columns=["customer_id", "churn_status"]
    )
    assert len(fitted) == 4

    results = evaluate_models(
        fitted, X_train, y_train, X_val, y_val, X_test, y_test, pt, features
    )
    assert len(results) == 4
    assert all(r.primary_score is not None for r in results)
    assert results[0].name in {"Dummy", "Logistic Regression", "Random Forest", "Gradient Boosting"}

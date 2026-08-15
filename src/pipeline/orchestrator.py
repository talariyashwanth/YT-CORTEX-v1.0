"""End-to-end analysis orchestrator."""

from __future__ import annotations

import pandas as pd

from src.diagnostics.engine import build_diagnostics
from src.evaluation.evaluator import evaluate_models
from src.features.analyzer import analyze_features
from src.ingestion.loader import get_feature_columns, infer_problem_type
from src.leakage.detector import detect_leakage
from src.modeling.trainer import train_baselines
from src.models.schemas import AnalysisResult, ProblemType
from src.profiling.profiler import profile_dataset
from src.quality.detector import detect_quality_issues
from src.recommendations.engine import generate_recommendations


def run_analysis(
    df: pd.DataFrame,
    filename: str,
    target_column: str,
    problem_type: ProblemType | None = None,
) -> AnalysisResult:
    """Run the full YT CORTEX analysis pipeline."""
    pt = problem_type or infer_problem_type(df[target_column])
    features = get_feature_columns(df, target_column)

    # Phase 2: quality & leakage
    quality_issues = detect_quality_issues(df, target_column)
    leakage_flags = detect_leakage(df, target_column, pt, features)

    # Phase 1: profiling (with quality/leakage counts for health score)
    high_quality_risks = sum(
        1 for q in quality_issues if q.severity.value in ("high", "critical")
    )
    dataset_profile = profile_dataset(
        df,
        filename,
        target_column=target_column,
        problem_type=pt,
        n_quality_risks=high_quality_risks,
        n_leakage_flags=len(leakage_flags),
    )

    # Feature analysis
    feature_insights = analyze_features(df, target_column, pt, features)

    # Exclude ID/constant columns from modeling
    exclude = set(dataset_profile.id_columns + dataset_profile.constant_columns)
    leakage_features = {f.feature for f in leakage_flags if f.severity.value == "critical"}
    exclude.update(leakage_features)

    # Phase 3: modeling
    fitted, X_train, X_val, X_test, y_train, y_val, y_test, model_features = train_baselines(
        df, target_column, pt, exclude_columns=list(exclude)
    )
    model_results = evaluate_models(
        fitted, X_train, y_train, X_val, y_val, X_test, y_test, pt, model_features
    )

    best = model_results[0] if model_results else None

    result = AnalysisResult(
        dataset_profile=dataset_profile,
        quality_issues=quality_issues,
        leakage_flags=leakage_flags,
        feature_insights=feature_insights,
        model_results=model_results,
        problem_type=pt,
        target_column=target_column,
        best_model=best.name if best else "",
        primary_metric=best.primary_metric if best else "",
        best_score=best.primary_score if best else 0.0,
    )

    # Phase 4: diagnostics & recommendations
    result.diagnostics = build_diagnostics(result)
    result.recommendations = generate_recommendations(result)

    # Refresh health score with final diagnostic count
    result.dataset_profile.health_score = max(
        0.0,
        result.dataset_profile.health_score
        - len([d for d in result.diagnostics if d.severity.value == "critical"]) * 5,
    )

    return result

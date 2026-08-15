"""Diagnostic engine — aggregates all detected issues."""

from __future__ import annotations

from src import config
from src.models.schemas import (
    AnalysisResult,
    DiagnosticIssue,
    ModelResult,
    ProblemType,
    QualityIssue,
    LeakageFlag,
    Severity,
)


def _severity_rank(severity: Severity) -> int:
    return {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}[severity]


def build_diagnostics(result: AnalysisResult) -> list[DiagnosticIssue]:
    """Convert quality, leakage, and model findings into unified diagnostics."""
    diagnostics: list[DiagnosticIssue] = []

    for issue in result.quality_issues:
        diagnostics.append(
            DiagnosticIssue(
                id=issue.id,
                title=issue.title,
                category=issue.category,
                severity=issue.severity,
                evidence=issue.evidence,
                explanation=issue.explanation,
                recommendation=issue.recommendation,
                confidence="high",
            )
        )

    for flag in result.leakage_flags:
        diagnostics.append(
            DiagnosticIssue(
                id=flag.id,
                title=f"Possible leakage: {flag.feature}",
                category="leakage",
                severity=flag.severity,
                evidence=flag.evidence,
                explanation=flag.explanation,
                recommendation=flag.recommendation,
                confidence="moderate",
            )
        )

    if result.dataset_profile.target:
        for warning in result.dataset_profile.target.warnings:
            diagnostics.append(
                DiagnosticIssue(
                    id=f"target_{len(diagnostics):03d}",
                    title="Target distribution issue",
                    category="target",
                    severity=Severity.MEDIUM,
                    evidence=warning,
                    explanation="Imbalanced targets can mislead accuracy-based evaluation.",
                    recommendation="Use stratified splits and inspect F1 / PR-AUC.",
                    confidence="high",
                )
            )

    for model in result.model_results:
        if model.overfitting:
            diagnostics.append(
                DiagnosticIssue(
                    id=f"overfit_{model.name.lower().replace(' ', '_')}",
                    title=f"Possible overfitting in {model.name}",
                    category="generalization",
                    severity=Severity.HIGH,
                    evidence=(
                        f"Train {model.primary_metric}={model.train_metrics.get(model.primary_metric, 0):.3f}, "
                        f"Validation {model.primary_metric}={model.val_metrics.get(model.primary_metric, 0):.3f} "
                        f"(gap={model.overfitting_gap:.3f})."
                    ),
                    explanation="Large train-validation gap suggests the model memorizes training data.",
                    recommendation="Reduce complexity, add regularization, or collect more data.",
                    confidence="moderate",
                )
            )
        if model.underfitting:
            diagnostics.append(
                DiagnosticIssue(
                    id=f"underfit_{model.name.lower().replace(' ', '_')}",
                    title=f"Possible underfitting in {model.name}",
                    category="generalization",
                    severity=Severity.MEDIUM,
                    evidence=(
                        f"Validation {model.primary_metric}={model.val_metrics.get(model.primary_metric, 0):.3f}."
                    ),
                    explanation="Low validation performance on both train and validation suggests weak signal.",
                    recommendation="Improve features, check data quality, or increase model complexity.",
                    confidence="moderate",
                )
            )

    return sorted(diagnostics, key=lambda d: _severity_rank(d.severity))

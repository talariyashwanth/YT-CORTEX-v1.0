"""Recommendation engine — prioritized next steps."""

from __future__ import annotations

from src.models.schemas import AnalysisResult, DiagnosticIssue, Recommendation, Severity


PRIORITY_ORDER = {
    "leakage": 1,
    "identifier": 2,
    "duplicates": 3,
    "target": 4,
    "missing_values": 5,
    "generalization": 6,
    "cardinality": 7,
    "constant_column": 8,
    "near_constant": 9,
}


def _severity_rank(severity: Severity) -> int:
    return {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}[severity]


def generate_recommendations(result: AnalysisResult) -> list[Recommendation]:
    """Generate prioritized, evidence-backed recommendations."""
    recs: list[Recommendation] = []
    seen_titles: set[str] = set()

    for diag in sorted(
        result.diagnostics,
        key=lambda d: (_severity_rank(d.severity), PRIORITY_ORDER.get(d.category, 99)),
    ):
        if diag.title in seen_titles:
            continue
        seen_titles.add(diag.title)
        recs.append(
            Recommendation(
                priority=len(recs) + 1,
                title=diag.title,
                severity=diag.severity,
                action=diag.recommendation,
                rationale=f"{diag.evidence} {diag.explanation}",
                category=diag.category,
            )
        )

    # Model-based recommendation
    if result.model_results:
        best = result.model_results[0]
        dummy = next((m for m in result.model_results if m.name == "Dummy"), None)
        if dummy and best.name != "Dummy":
            improvement = best.primary_score - dummy.primary_score
            recs.append(
                Recommendation(
                    priority=len(recs) + 1,
                    title=f"Best baseline: {best.name}",
                    severity=Severity.LOW,
                    action=(
                        f"Use {best.name} as your starting point "
                        f"(validation {best.primary_metric}={best.primary_score:.3f})."
                    ),
                    rationale=(
                        f"{best.name} improves validation {best.primary_metric} by "
                        f"{improvement:.3f} over the dummy baseline."
                    ),
                    category="modeling",
                )
            )

    # Re-number priorities
    for i, rec in enumerate(recs, start=1):
        rec.priority = i

    return recs

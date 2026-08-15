"""Shared data models for YT CORTEX."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    ID = "id"
    CONSTANT = "constant"
    TARGET = "target"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    column_type: ColumnType
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
    risk_level: Severity = Severity.LOW


@dataclass
class TargetProfile:
    name: str
    problem_type: ProblemType
    distribution: dict[str, Any] = field(default_factory=dict)
    imbalance_ratio: float | None = None
    stats: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetProfile:
    filename: str
    n_rows: int
    n_columns: int
    n_features: int
    memory_mb: float
    missing_pct: float
    duplicate_count: int
    duplicate_pct: float
    numeric_features: int
    categorical_features: int
    datetime_features: int
    id_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    columns: list[ColumnProfile] = field(default_factory=list)
    target: TargetProfile | None = None
    health_score: float = 0.0
    health_summary: str = ""


@dataclass
class QualityIssue:
    id: str
    title: str
    category: str
    severity: Severity
    feature: str | None
    evidence: str
    explanation: str
    recommendation: str


@dataclass
class LeakageFlag:
    id: str
    feature: str
    severity: Severity
    signal_type: str
    signal_value: float
    evidence: str
    explanation: str
    recommendation: str


@dataclass
class FeatureInsight:
    name: str
    column_type: ColumnType
    summary: str
    stats: dict[str, Any] = field(default_factory=dict)
    target_relationship: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResult:
    name: str
    problem_type: ProblemType
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    overfitting_gap: float | None = None
    underfitting: bool = False
    overfitting: bool = False
    feature_importance: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    primary_metric: str = ""
    primary_score: float = 0.0


@dataclass
class DiagnosticIssue:
    id: str
    title: str
    category: str
    severity: Severity
    evidence: str
    explanation: str
    recommendation: str
    confidence: str = "moderate"


@dataclass
class Recommendation:
    priority: int
    title: str
    severity: Severity
    action: str
    rationale: str
    category: str


@dataclass
class AnalysisResult:
    """Complete analysis output from the orchestrator."""

    dataset_profile: DatasetProfile
    quality_issues: list[QualityIssue] = field(default_factory=list)
    leakage_flags: list[LeakageFlag] = field(default_factory=list)
    feature_insights: list[FeatureInsight] = field(default_factory=list)
    model_results: list[ModelResult] = field(default_factory=list)
    diagnostics: list[DiagnosticIssue] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    best_model: str = ""
    primary_metric: str = ""
    best_score: float = 0.0
    problem_type: ProblemType = ProblemType.BINARY_CLASSIFICATION
    target_column: str = ""

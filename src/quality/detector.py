"""Data quality issue detection."""

from __future__ import annotations

import pandas as pd

from src import config
from src.models.schemas import ColumnType, QualityIssue, Severity


def detect_quality_issues(df: pd.DataFrame, target: str) -> list[QualityIssue]:
    """Detect common data quality problems."""
    issues: list[QualityIssue] = []
    n_rows = len(df)
    issue_counter = 0

    def _next_id() -> str:
        nonlocal issue_counter
        issue_counter += 1
        return f"quality_{issue_counter:03d}"

    # Dataset-level duplicates
    dup_count = int(df.duplicated().sum())
    dup_pct = dup_count / max(n_rows, 1)
    if dup_pct >= config.DUPLICATE_MEDIUM_THRESHOLD:
        issues.append(
            QualityIssue(
                id=_next_id(),
                title="Duplicate rows detected",
                category="duplicates",
                severity=Severity.MEDIUM if dup_pct < 0.05 else Severity.HIGH,
                feature=None,
                evidence=f"{dup_count:,} duplicate rows ({dup_pct:.1%} of dataset).",
                explanation="Duplicate rows can inflate performance and distort statistics.",
                recommendation="Remove or investigate duplicate records before modeling.",
            )
        )

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        missing_pct = series.isna().mean()
        n_unique = series.nunique(dropna=True)
        unique_ratio = n_unique / max(n_rows, 1)

        # Missing values
        if missing_pct >= config.MISSING_MEDIUM_THRESHOLD:
            sev = (
                Severity.HIGH
                if missing_pct >= config.MISSING_HIGH_THRESHOLD
                else Severity.MEDIUM
            )
            issues.append(
                QualityIssue(
                    id=_next_id(),
                    title=f"High missing values in '{col}'",
                    category="missing_values",
                    severity=sev,
                    feature=col,
                    evidence=f"Missing: {missing_pct:.1%} ({int(series.isna().sum()):,} rows).",
                    explanation="Missing values can bias models and reduce effective sample size.",
                    recommendation="Review imputation strategy or investigate why values are missing.",
                )
            )

        # Constant / near-constant
        if n_unique <= config.CONSTANT_UNIQUE_THRESHOLD:
            issues.append(
                QualityIssue(
                    id=_next_id(),
                    title=f"Constant column '{col}'",
                    category="constant_column",
                    severity=Severity.MEDIUM,
                    feature=col,
                    evidence=f"Only {n_unique} unique value(s).",
                    explanation="Constant features provide no predictive information.",
                    recommendation="Drop this column from the feature set.",
                )
            )
        elif n_unique <= config.NEAR_CONSTANT_UNIQUE_THRESHOLD:
            issues.append(
                QualityIssue(
                    id=_next_id(),
                    title=f"Near-constant column '{col}'",
                    category="near_constant",
                    severity=Severity.LOW,
                    feature=col,
                    evidence=f"Only {n_unique} unique values in {n_rows:,} rows.",
                    explanation="Near-constant features rarely contribute to model performance.",
                    recommendation="Consider dropping or combining rare categories.",
                )
            )

        # Likely ID columns
        if unique_ratio >= config.ID_UNIQUENESS_RATIO and pd.api.types.is_numeric_dtype(series):
            issues.append(
                QualityIssue(
                    id=_next_id(),
                    title=f"Likely identifier column '{col}'",
                    category="identifier",
                    severity=Severity.HIGH,
                    feature=col,
                    evidence=f"{n_unique:,} unique values ({unique_ratio:.1%} of rows).",
                    explanation="ID columns can cause overfitting and do not generalize.",
                    recommendation="Exclude identifier columns from model features.",
                )
            )
        elif unique_ratio >= config.ID_UNIQUENESS_RATIO:
            issues.append(
                QualityIssue(
                    id=_next_id(),
                    title=f"Likely identifier column '{col}'",
                    category="identifier",
                    severity=Severity.HIGH,
                    feature=col,
                    evidence=f"{n_unique:,} unique values ({unique_ratio:.1%} of rows).",
                    explanation="High-cardinality text IDs can cause overfitting.",
                    recommendation="Exclude identifier columns from model features.",
                )
            )

        # High cardinality categoricals
        if (
            not pd.api.types.is_numeric_dtype(series)
            and n_unique > 50
            and unique_ratio > 0.1
            and unique_ratio < config.ID_UNIQUENESS_RATIO
        ):
            issues.append(
                QualityIssue(
                    id=_next_id(),
                    title=f"High cardinality in '{col}'",
                    category="cardinality",
                    severity=Severity.MEDIUM,
                    feature=col,
                    evidence=f"{n_unique:,} unique categories ({unique_ratio:.1%} of rows).",
                    explanation="High-cardinality categoricals can cause sparse encoding and overfitting.",
                    recommendation="Consider target encoding, grouping rare categories, or feature hashing.",
                )
            )

    return issues

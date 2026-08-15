"""Dataset profiling and health scoring."""

from __future__ import annotations

import pandas as pd

from src import config
from src.models.schemas import (
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    ProblemType,
    Severity,
    TargetProfile,
)


def _classify_column(series: pd.Series, n_rows: int, is_target: bool = False) -> ColumnType:
    if is_target:
        return ColumnType.TARGET
    n_unique = series.nunique(dropna=True)
    if n_unique <= config.CONSTANT_UNIQUE_THRESHOLD:
        return ColumnType.CONSTANT
    if n_unique / max(n_rows, 1) >= config.ID_UNIQUENESS_RATIO:
        return ColumnType.ID
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERIC
    return ColumnType.CATEGORICAL


def _numeric_stats(series: pd.Series) -> dict[str, float]:
    clean = series.dropna()
    if clean.empty:
        return {}
    return {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "skew": float(clean.skew()) if len(clean) > 2 else 0.0,
    }


def _assess_column_risk(
    column_type: ColumnType,
    missing_pct: float,
    unique_pct: float,
) -> Severity:
    if column_type in (ColumnType.ID, ColumnType.CONSTANT):
        return Severity.HIGH
    if missing_pct >= config.MISSING_HIGH_THRESHOLD:
        return Severity.HIGH
    if missing_pct >= config.MISSING_MEDIUM_THRESHOLD:
        return Severity.MEDIUM
    if column_type == ColumnType.CATEGORICAL and unique_pct >= config.HIGH_CARDINALITY_RATIO:
        return Severity.MEDIUM
    return Severity.LOW


def _profile_target(series: pd.Series, problem_type: ProblemType) -> TargetProfile:
    warnings: list[str] = []
    distribution: dict = {}
    stats: dict[str, float] = {}
    imbalance_ratio: float | None = None

    if problem_type == ProblemType.REGRESSION:
        clean = series.dropna()
        stats = _numeric_stats(clean)
        if abs(stats.get("skew", 0)) > 2:
            warnings.append("Target distribution is highly skewed.")
        distribution = {"type": "continuous", "stats": stats}
    else:
        counts = series.value_counts(dropna=False)
        total = len(series)
        distribution = {
            str(k): {"count": int(v), "pct": round(v / total * 100, 2)}
            for k, v in counts.items()
        }
        if problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ):
            proportions = counts / total
            minority = float(proportions.min())
            imbalance_ratio = minority
            if minority < config.IMBALANCE_HIGH_THRESHOLD:
                warnings.append(
                    f"Severe class imbalance detected (minority class = {minority:.1%})."
                )
            elif minority < config.IMBALANCE_MEDIUM_THRESHOLD:
                warnings.append(
                    f"Class imbalance detected (minority class = {minority:.1%})."
                )

    return TargetProfile(
        name=series.name or "target",
        problem_type=problem_type,
        distribution=distribution,
        imbalance_ratio=imbalance_ratio,
        stats=stats,
        warnings=warnings,
    )


def _compute_health_score(
    missing_pct: float,
    duplicate_pct: float,
    target: TargetProfile | None,
    n_quality_risks: int,
    n_leakage_flags: int = 0,
) -> tuple[float, str]:
  """Compute 0-100 health score."""
  missing_penalty = min(missing_pct * 100, 40)
  duplicate_penalty = min(duplicate_pct * 100, 20)
  imbalance_penalty = 0.0
  if target and target.imbalance_ratio is not None:
      if target.imbalance_ratio < config.IMBALANCE_HIGH_THRESHOLD:
          imbalance_penalty = 25
      elif target.imbalance_ratio < config.IMBALANCE_MEDIUM_THRESHOLD:
          imbalance_penalty = 12
  quality_penalty = min(n_quality_risks * 5, 20)
  leakage_penalty = min(n_leakage_flags * 10, 15)

  score = max(
      0.0,
      100.0
      - missing_penalty
      - duplicate_penalty
      - imbalance_penalty
      - quality_penalty
      - leakage_penalty,
  )

  if score >= 80:
      summary = "Good overall data health."
  elif score >= 60:
      summary = "Moderate data health — review flagged issues."
  elif score >= 40:
      summary = "Poor data health — several issues need attention."
  else:
      summary = "Critical data health — address issues before modeling."

  return round(score, 1), summary


def profile_dataset(
    df: pd.DataFrame,
    filename: str,
    target_column: str | None = None,
    problem_type: ProblemType | None = None,
    n_quality_risks: int = 0,
    n_leakage_flags: int = 0,
) -> DatasetProfile:
    """Generate a comprehensive dataset profile."""
    n_rows, n_columns = df.shape
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = duplicate_count / max(n_rows, 1)
    total_cells = n_rows * n_columns
    missing_cells = int(df.isna().sum().sum())
    missing_pct = missing_cells / max(total_cells, 1)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    columns: list[ColumnProfile] = []
    id_columns: list[str] = []
    constant_columns: list[str] = []
    numeric_count = categorical_count = datetime_count = 0

    for col in df.columns:
        series = df[col]
        is_target = col == target_column
        col_type = _classify_column(series, n_rows, is_target=is_target)
        missing_count = int(series.isna().sum())
        missing_col_pct = missing_count / max(n_rows, 1)
        unique_count = int(series.nunique(dropna=True))
        unique_pct = unique_count / max(n_rows, 1)

        if col_type == ColumnType.ID:
            id_columns.append(col)
        elif col_type == ColumnType.CONSTANT:
            constant_columns.append(col)
        elif col_type == ColumnType.NUMERIC:
            numeric_count += 1
        elif col_type == ColumnType.CATEGORICAL:
            categorical_count += 1
        elif col_type == ColumnType.DATETIME:
            datetime_count += 1

        stats = _numeric_stats(series) if col_type == ColumnType.NUMERIC else {}
        sample = series.dropna().head(5).tolist()

        columns.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                column_type=col_type,
                missing_count=missing_count,
                missing_pct=round(missing_col_pct, 4),
                unique_count=unique_count,
                unique_pct=round(unique_pct, 4),
                sample_values=sample,
                stats=stats,
                risk_level=_assess_column_risk(col_type, missing_col_pct, unique_pct),
            )
        )

    target_profile = None
    if target_column and target_column in df.columns:
        from src.ingestion.loader import infer_problem_type

        pt = problem_type or infer_problem_type(df[target_column])
        target_profile = _profile_target(df[target_column], pt)

    health_score, health_summary = _compute_health_score(
        missing_pct, duplicate_pct, target_profile, n_quality_risks, n_leakage_flags
    )

    return DatasetProfile(
        filename=filename,
        n_rows=n_rows,
        n_columns=n_columns,
        n_features=n_columns - (1 if target_column else 0),
        memory_mb=round(memory_mb, 2),
        missing_pct=round(missing_pct, 4),
        duplicate_count=duplicate_count,
        duplicate_pct=round(duplicate_pct, 4),
        numeric_features=numeric_count,
        categorical_features=categorical_count,
        datetime_features=datetime_count,
        id_columns=id_columns,
        constant_columns=constant_columns,
        columns=columns,
        target=target_profile,
        health_score=health_score,
        health_summary=health_summary,
    )

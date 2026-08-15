"""Feature-level analysis."""

from __future__ import annotations

import pandas as pd

from src.models.schemas import ColumnType, FeatureInsight, ProblemType


def analyze_features(
    df: pd.DataFrame,
    target: str,
    problem_type: ProblemType,
    feature_columns: list[str] | None = None,
) -> list[FeatureInsight]:
    """Generate per-feature insights and target relationships."""
    features = feature_columns or [c for c in df.columns if c != target]
    insights: list[FeatureInsight] = []

    for col in features:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        col_type = ColumnType.NUMERIC if is_numeric else ColumnType.CATEGORICAL
        stats: dict = {
            "missing_pct": round(series.isna().mean(), 4),
            "unique_count": int(series.nunique(dropna=True)),
        }
        summary_parts: list[str] = []

        if is_numeric:
            clean = series.dropna()
            if not clean.empty:
                stats.update(
                    {
                        "mean": round(float(clean.mean()), 4),
                        "median": round(float(clean.median()), 4),
                        "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                        "skew": round(float(clean.skew()), 4) if len(clean) > 2 else 0.0,
                    }
                )
                summary_parts.append(f"mean={stats['mean']}, std={stats['std']}")
        else:
            top_cats = series.value_counts().head(5)
            stats["top_categories"] = {str(k): int(v) for k, v in top_cats.items()}
            summary_parts.append(f"{stats['unique_count']} categories")

        target_rel: dict = {}
        if problem_type == ProblemType.REGRESSION and is_numeric:
            corr = df[[col, target]].dropna().corr().iloc[0, 1]
            target_rel["correlation"] = round(float(corr), 4)
            summary_parts.append(f"corr with target={target_rel['correlation']}")
        elif problem_type != ProblemType.REGRESSION:
            rates = df.groupby(col)[target].apply(
                lambda x: x.value_counts(normalize=True).iloc[0] if len(x) > 0 else 0
            )
            target_rel["dominant_class_rate_range"] = {
                "min": round(float(rates.min()), 4),
                "max": round(float(rates.max()), 4),
            }
            summary_parts.append("categorical target rates computed")

        insights.append(
            FeatureInsight(
                name=col,
                column_type=col_type,
                summary="; ".join(summary_parts) if summary_parts else "No summary available.",
                stats=stats,
                target_relationship=target_rel,
            )
        )

    return insights

"""Possible data leakage detection."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from src import config
from src.models.schemas import LeakageFlag, ProblemType, Severity


TARGET_LIKE_PATTERNS = [
    r"\btarget\b",
    r"\blabel\b",
    r"\boutcome\b",
    r"\bchurn\b",
    r"\bdefault\b",
    r"\bstatus\b",
    r"\bresult\b",
    r"\bclass\b",
    r"\by\b$",
    r"\bis_",
    r"\bhas_",
    r"\bfraud\b",
    r"\bconverted\b",
]


def _target_like_name(feature: str, target: str) -> bool:
    feature_lower = feature.lower()
    target_lower = target.lower()
    if feature_lower == target_lower:
        return True
    for pattern in TARGET_LIKE_PATTERNS:
        if re.search(pattern, feature_lower):
            return True
    return False


def _encode_for_mi(X: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(X):
        return X.fillna(X.median() if X.notna().any() else 0).values.reshape(-1, 1)
    codes, _ = pd.factorize(X.fillna("__MISSING__"))
    return codes.reshape(-1, 1)


def _correlation_with_target(
    df: pd.DataFrame, feature: str, target: str, problem_type: ProblemType
) -> float | None:
    try:
        if problem_type == ProblemType.REGRESSION:
            if pd.api.types.is_numeric_dtype(df[feature]):
                return float(df[[feature, target]].dropna().corr().iloc[0, 1])
            encoded = pd.factorize(df[feature].fillna("__MISSING__"))[0]
            return float(np.corrcoef(encoded, df[target].fillna(df[target].median()))[0, 1])
        else:
            if pd.api.types.is_numeric_dtype(df[feature]):
                y_codes = pd.factorize(df[target])[0]
                return float(abs(np.corrcoef(df[feature].fillna(0), y_codes)[0, 1]))
            # Categorical association via Cramér's V approximation
            contingency = pd.crosstab(df[feature].fillna("__MISSING__"), df[target])
            chi2 = float(
                ((contingency - contingency.mean().mean()) ** 2 / contingency.sum().sum()).sum()
            )
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            if min_dim <= 0 or n == 0:
                return 0.0
            return float(np.sqrt(chi2 / (n * min_dim)))
    except Exception:
        return None


def _mutual_info(
    df: pd.DataFrame, feature: str, target: str, problem_type: ProblemType
) -> float | None:
    try:
        mask = df[feature].notna() & df[target].notna()
        if mask.sum() < 10:
            return None
        X = _encode_for_mi(df.loc[mask, feature])
        y = df.loc[mask, target]
        if problem_type == ProblemType.REGRESSION:
            mi = mutual_info_regression(X, y, random_state=42)
        else:
            y_codes = pd.factorize(y)[0]
            mi = mutual_info_classif(X, y_codes, random_state=42)
        return float(mi[0])
    except Exception:
        return None


def detect_leakage(
    df: pd.DataFrame,
    target: str,
    problem_type: ProblemType,
    feature_columns: list[str] | None = None,
) -> list[LeakageFlag]:
    """Flag features that may leak target information."""
    flags: list[LeakageFlag] = []
    features = feature_columns or [c for c in df.columns if c != target]
    flag_counter = 0

    def _next_id() -> str:
        nonlocal flag_counter
        flag_counter += 1
        return f"leakage_{flag_counter:03d}"

    for feature in features:
        # Target-like naming
        if _target_like_name(feature, target):
            flags.append(
                LeakageFlag(
                    id=_next_id(),
                    feature=feature,
                    severity=Severity.HIGH,
                    signal_type="naming",
                    signal_value=1.0,
                    evidence=f"Feature name '{feature}' resembles target-related naming patterns.",
                    explanation=(
                        "Features with target-like names may contain information "
                        "recorded after the prediction event."
                    ),
                    recommendation="Investigate when this feature is created relative to the target.",
                )
            )

        corr = _correlation_with_target(df, feature, target, problem_type)
        if corr is not None and abs(corr) >= config.LEAKAGE_CORRELATION_THRESHOLD:
            flags.append(
                LeakageFlag(
                    id=_next_id(),
                    feature=feature,
                    severity=Severity.CRITICAL,
                    signal_type="correlation",
                    signal_value=abs(corr),
                    evidence=f"Association with target: {abs(corr):.3f}.",
                    explanation=(
                        "This feature has a suspiciously strong association with the target. "
                        "Possible data leakage — not confirmed."
                    ),
                    recommendation=(
                        "Investigate feature creation timing and whether it uses post-event information."
                    ),
                )
            )

        mi = _mutual_info(df, feature, target, problem_type)
        if mi is not None and mi >= config.LEAKAGE_MI_THRESHOLD:
            flags.append(
                LeakageFlag(
                    id=_next_id(),
                    feature=feature,
                    severity=Severity.CRITICAL,
                    signal_type="mutual_information",
                    signal_value=mi,
                    evidence=f"Mutual information with target: {mi:.3f}.",
                    explanation=(
                        "This feature shares unusually high information with the target. "
                        "Possible data leakage — not confirmed."
                    ),
                    recommendation="Verify this feature is available at prediction time.",
                )
            )

        # Near-perfect categorical predictor
        if not pd.api.types.is_numeric_dtype(df[feature]):
            grouped = df.groupby(feature)[target].nunique()
            if len(grouped) > 0 and (grouped == 1).mean() > 0.95:
                flags.append(
                    LeakageFlag(
                        id=_next_id(),
                        feature=feature,
                        severity=Severity.CRITICAL,
                        signal_type="near_perfect_predictor",
                        signal_value=float((grouped == 1).mean()),
                        evidence=(
                            f"{(grouped == 1).mean():.1%} of categories map to a single target value."
                        ),
                        explanation=(
                            "This categorical feature almost perfectly determines the target. "
                            "Possible leakage or data entry error."
                        ),
                        recommendation="Inspect whether this feature is derived from the target.",
                    )
                )

    # Deduplicate by feature + signal_type, keep highest severity
    seen: dict[tuple[str, str], LeakageFlag] = {}
    severity_order = {
        Severity.CRITICAL: 0,
        Severity.HIGH: 1,
        Severity.MEDIUM: 2,
        Severity.LOW: 3,
    }
    for flag in flags:
        key = (flag.feature, flag.signal_type)
        if key not in seen or severity_order[flag.severity] < severity_order[seen[key].severity]:
            seen[key] = flag

    return sorted(seen.values(), key=lambda f: severity_order[f.severity])

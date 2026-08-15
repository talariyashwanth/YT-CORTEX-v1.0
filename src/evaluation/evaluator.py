"""Model evaluation and generalization analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.models.schemas import ModelResult, ProblemType


def _classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            elif y_proba.ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return metrics


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _primary_metric_name(problem_type: ProblemType) -> str:
    return "r2" if problem_type == ProblemType.REGRESSION else "f1"


def _get_predictions(pipe, X):
    y_pred = pipe.predict(X)
    y_proba = None
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        try:
            y_proba = pipe.predict_proba(X)
        except Exception:
            pass
    return y_pred, y_proba


def _extract_feature_importance(pipe, feature_names: list[str]) -> dict[str, float]:
    model = pipe.named_steps["model"]
    preprocessor = pipe.named_steps["preprocessor"]

    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            try:
                names = preprocessor.get_feature_names_out()
            except Exception:
                names = [f"feature_{i}" for i in range(len(importances))]
            pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
            return {str(n): float(v) for n, v in pairs[:15]}
        if hasattr(model, "coef_"):
            coef = np.abs(model.coef_).flatten()
            try:
                names = preprocessor.get_feature_names_out()
            except Exception:
                names = [f"feature_{i}" for i in range(len(coef))]
            pairs = sorted(zip(names, coef), key=lambda x: x[1], reverse=True)
            return {str(n): float(v) for n, v in pairs[:15]}
    except Exception:
        pass
    return {}


def evaluate_models(
    fitted_models: list,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    problem_type: ProblemType,
    feature_names: list[str],
) -> list[ModelResult]:
    """Evaluate all fitted models on train/val/test splits."""
    primary = _primary_metric_name(problem_type)
    results: list[ModelResult] = []

    for name, pipe in fitted_models:
        y_train_pred, y_train_proba = _get_predictions(pipe, X_train)
        y_val_pred, y_val_proba = _get_predictions(pipe, X_val)
        y_test_pred, y_test_proba = _get_predictions(pipe, X_test)

        if problem_type == ProblemType.REGRESSION:
            train_m = _regression_metrics(y_train, y_train_pred)
            val_m = _regression_metrics(y_val, y_val_pred)
            test_m = _regression_metrics(y_test, y_test_pred)
            gap = train_m["r2"] - val_m["r2"]
            overfitting = gap > config.OVERFITTING_GAP_THRESHOLD
            underfitting = val_m["r2"] < config.UNDERFITTING_MAX_SCORE and not overfitting
            cm = None
        else:
            train_m = _classification_metrics(y_train, y_train_pred, y_train_proba)
            val_m = _classification_metrics(y_val, y_val_pred, y_val_proba)
            test_m = _classification_metrics(y_test, y_test_pred, y_test_proba)
            gap = train_m["f1"] - val_m["f1"]
            overfitting = gap > config.OVERFITTING_GAP_THRESHOLD
            underfitting = val_m["f1"] < config.UNDERFITTING_MAX_SCORE and not overfitting
            try:
                cm = confusion_matrix(y_test, y_test_pred).tolist()
            except Exception:
                cm = None

        importance = _extract_feature_importance(pipe, feature_names)

        results.append(
            ModelResult(
                name=name,
                problem_type=problem_type,
                train_metrics=train_m,
                val_metrics=val_m,
                test_metrics=test_m,
                overfitting_gap=round(gap, 4),
                overfitting=overfitting,
                underfitting=underfitting,
                feature_importance=importance,
                confusion_matrix=cm,
                primary_metric=primary,
                primary_score=val_m.get(primary, 0.0),
            )
        )

    return sorted(results, key=lambda r: r.primary_score, reverse=(problem_type == ProblemType.REGRESSION))

"""Dataset ingestion and validation."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src.config import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS
from src.models.schemas import ProblemType


class IngestionError(Exception):
    """Raised when dataset ingestion fails."""


def validate_file(path: Path | str) -> Path:
    """Validate file exists, extension, and size."""
    file_path = Path(path)
    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported format '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise IngestionError(
            f"File exceeds {MAX_FILE_SIZE_MB} MB limit ({size_mb:.1f} MB)."
        )
    return file_path


def load_dataframe(path: Path | str, **read_kwargs) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    file_path = validate_file(path)
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path, **read_kwargs)
        else:
            df = pd.read_excel(file_path, **read_kwargs)
    except Exception as exc:
        raise IngestionError(f"Failed to parse {file_path.name}: {exc}") from exc

    if df.empty:
        raise IngestionError("Dataset is empty.")
    if df.shape[1] < 2:
        raise IngestionError("Dataset must have at least 2 columns (features + target).")
    return df


def load_dataframe_from_bytes(
    content: bytes, filename: str, **read_kwargs
) -> pd.DataFrame:
    """Load a DataFrame from uploaded file bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise IngestionError(
            f"File exceeds {MAX_FILE_SIZE_MB} MB limit ({size_mb:.1f} MB)."
        )
    buffer = io.BytesIO(content)
    try:
        if suffix == ".csv":
            df = pd.read_csv(buffer, **read_kwargs)
        else:
            df = pd.read_excel(buffer, **read_kwargs)
    except Exception as exc:
        raise IngestionError(f"Failed to parse {filename}: {exc}") from exc

    if df.empty:
        raise IngestionError("Dataset is empty.")
    if df.shape[1] < 2:
        raise IngestionError("Dataset must have at least 2 columns (features + target).")
    return df


def infer_problem_type(series: pd.Series) -> ProblemType:
    """Infer supervised learning problem type from target column."""
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique(dropna=True)
        if n_unique <= 2:
            return ProblemType.BINARY_CLASSIFICATION
        if n_unique <= 20 and n_unique / max(len(series), 1) < 0.05:
            return ProblemType.MULTICLASS_CLASSIFICATION
        return ProblemType.REGRESSION

    n_unique = series.nunique(dropna=True)
    if n_unique <= 2:
        return ProblemType.BINARY_CLASSIFICATION
    return ProblemType.MULTICLASS_CLASSIFICATION


def get_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Return feature column names excluding the target."""
    if target not in df.columns:
        raise IngestionError(f"Target column '{target}' not found in dataset.")
    return [col for col in df.columns if col != target]

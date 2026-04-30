"""Column profiling and type classification."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from core.models import ColumnProfile


ID_KEYWORDS = (
    "id",
    "sn",
    "ssn",
    "lot",
    "panel",
    "barcode",
    "bcr",
    "serial",
    "file",
    "filename",
    "row",
)


def _sample_series(series: pd.Series, sample_size: int = 5000) -> pd.Series:
    if len(series) <= sample_size:
        return series
    return series.sample(sample_size, random_state=42)


def looks_datetime(series: pd.Series, column_name: str = "") -> bool:
    """Detect datetime dtype or date-like strings from a sample."""
    lowered = column_name.lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if any(token in lowered for token in ("date", "time", "timestamp")):
        sample = _sample_series(series).dropna().astype(str).head(200)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return bool(parsed.notna().mean() >= 0.80)
    sample = _sample_series(series).dropna().astype(str).head(200)
    if sample.empty:
        return False
    pattern_hit = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean()
    if pattern_hit < 0.70:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.80)


def is_id_like(series: pd.Series, column_name: str, unique_ratio_threshold: float = 0.70) -> bool:
    """Detect ID-like columns by name or high uniqueness."""
    lowered = column_name.lower()
    if any(re.search(rf"(^|[_\-\s]){re.escape(token)}([_\-\s]|$)", lowered) or token in lowered for token in ID_KEYWORDS):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / max(len(series), 1)
    return bool(unique_ratio >= unique_ratio_threshold and unique_count > 30)


def classify_column(series: pd.Series, column_name: str, config: dict) -> str:
    """Classify a column into v3 analysis types."""
    cat_cfg = config.get("categorical_matrix_screening", {})
    max_levels = int(cat_cfg.get("max_category_levels", 30))
    if looks_datetime(series, column_name):
        return "datetime"
    if is_id_like(series, column_name):
        return "id_like"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    unique_count = int(_sample_series(series).nunique(dropna=True))
    if unique_count <= max_levels:
        return "categorical_low_cardinality"
    return "categorical_high_cardinality"


def profile_dataframe(
    df: pd.DataFrame,
    role: str,
    config: dict,
    key_cols: Iterable[str] | None = None,
    y_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Profile columns and return availability flags."""
    key_set = set(key_cols or [])
    y_set = set(y_cols or [])
    num_cfg = config.get("numeric_matrix_screening", {})
    cat_cfg = config.get("categorical_matrix_screening", {})
    max_missing = float(num_cfg.get("max_missing_rate", 0.7))
    min_variance = float(num_cfg.get("min_variance", 1e-12))
    max_levels = int(cat_cfg.get("max_category_levels", 30))
    rows = []

    for column in df.columns:
        series = df[column]
        inferred = classify_column(series, str(column), config)
        sample = _sample_series(series)
        missing_rate = float(sample.isna().mean()) if len(sample) else 1.0
        unique_count = int(sample.nunique(dropna=True))
        available = True
        reason = ""
        if column in key_set:
            available = False
            reason = "Key 컬럼"
        elif role == "x" and column in y_set:
            available = False
            reason = "Y 컬럼"
        elif missing_rate > max_missing:
            available = False
            reason = "결측률 과다"
        elif unique_count <= 1:
            available = False
            reason = "상수 컬럼"
        elif inferred == "numeric" and float(pd.to_numeric(sample, errors="coerce").var(skipna=True) or 0.0) <= min_variance:
            available = False
            reason = "저분산 숫자 컬럼"
        elif inferred == "id_like":
            available = False
            reason = "ID성 컬럼"
        elif inferred == "datetime":
            available = False
            reason = "날짜/시간 컬럼"
        elif inferred == "categorical_high_cardinality":
            available = False
            reason = f"범주 수 {max_levels} 초과"

        rows.append(
            ColumnProfile(
                column=str(column),
                role=role,
                inferred_type=inferred,
                missing_rate=missing_rate,
                unique_count=unique_count,
                available=available,
                exclude_reason=reason,
            ).to_dict()
        )
    return pd.DataFrame(rows)


def candidate_columns(profile_df: pd.DataFrame, *types: str) -> list[str]:
    """Return available columns filtered by inferred type."""
    if profile_df is None or profile_df.empty:
        return []
    mask = profile_df["available"].fillna(False)
    if types:
        mask &= profile_df["inferred_type"].isin(types)
    return profile_df.loc[mask, "column"].astype(str).tolist()

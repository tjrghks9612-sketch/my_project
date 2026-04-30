"""Data loading, merging, and profiling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    """Normalize a string or iterable into a clean list of column names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _file_name(file_or_path) -> str:
    """Return a readable file name for a path or Streamlit upload object."""
    return getattr(file_or_path, "name", str(file_or_path))


def read_single_file(file_or_path) -> pd.DataFrame:
    """Read one CSV, XLSX, or Parquet file into a DataFrame."""
    file_name = _file_name(file_or_path)
    suffix = Path(file_name).suffix.lower()

    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    if suffix == ".csv":
        return pd.read_csv(file_or_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_or_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_or_path)

    raise ValueError(f"Unsupported file type: {suffix}. Use CSV, XLSX, or Parquet.")


def read_data_files(files) -> pd.DataFrame:
    """Read one or many files and concatenate them row-wise."""
    if files is None:
        return pd.DataFrame()

    if not isinstance(files, (list, tuple)):
        files = [files]

    frames = []
    for file in files:
        if file is None:
            continue
        frame = read_single_file(file)
        frame["_source_file"] = Path(_file_name(file)).name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _key_tuples(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Build comparable key tuples while keeping missing keys explicit."""
    if not keys:
        return pd.Series([], dtype=object)
    key_frame = df[keys].copy().astype("string").fillna("<NA>")
    return pd.Series(list(map(tuple, key_frame.to_numpy())), index=df.index)


def duplicate_key_count(df: pd.DataFrame, keys: str | Iterable[str]) -> int:
    """Count rows whose merge key appears more than once."""
    key_cols = _as_list(keys)
    if df.empty or not key_cols:
        return 0
    return int(df.duplicated(subset=key_cols, keep=False).sum())


def merge_on_keys(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    x_keys: str | Iterable[str],
    y_keys: str | Iterable[str],
    how: str = "inner",
) -> tuple[pd.DataFrame, dict]:
    """Merge X and Y data and return merge statistics."""
    x_key_cols = _as_list(x_keys)
    y_key_cols = _as_list(y_keys)

    if len(x_key_cols) != len(y_key_cols):
        raise ValueError("X key and Y key counts must be the same for composite-key merge.")
    if not x_key_cols:
        raise ValueError("Select at least one key column.")

    missing_x = [column for column in x_key_cols if column not in x_df.columns]
    missing_y = [column for column in y_key_cols if column not in y_df.columns]
    if missing_x or missing_y:
        raise KeyError(f"Missing key columns. X: {missing_x}, Y: {missing_y}")

    x_key_series = _key_tuples(x_df, x_key_cols)
    y_key_series = _key_tuples(y_df, y_key_cols)
    x_key_set = set(x_key_series)
    y_key_set = set(y_key_series)

    x_matched_rows = int(x_key_series.isin(y_key_set).sum()) if len(x_df) else 0
    y_matched_rows = int(y_key_series.isin(x_key_set).sum()) if len(y_df) else 0

    merged = pd.merge(
        x_df,
        y_df,
        left_on=x_key_cols,
        right_on=y_key_cols,
        how=how,
        suffixes=("", "_Y"),
    )

    stats = {
        "x_rows": int(len(x_df)),
        "y_rows": int(len(y_df)),
        "merged_rows": int(len(merged)),
        "x_merge_rate": float(x_matched_rows / len(x_df) * 100) if len(x_df) else 0.0,
        "y_merge_rate": float(y_matched_rows / len(y_df) * 100) if len(y_df) else 0.0,
        "x_duplicate_key_rows": duplicate_key_count(x_df, x_key_cols),
        "y_duplicate_key_rows": duplicate_key_count(y_df, y_key_cols),
        "duplicate_key_rows": duplicate_key_count(x_df, x_key_cols) + duplicate_key_count(y_df, y_key_cols),
        "x_keys": x_key_cols,
        "y_keys": y_key_cols,
        "merge_how": how,
    }
    return merged, stats


def infer_basic_type(series: pd.Series) -> str:
    """Infer a simple human-readable column type."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "binary"
    if pd.api.types.is_numeric_dtype(series):
        non_missing_unique = series.dropna().nunique()
        return "binary" if non_missing_unique == 2 else "numeric"

    sample = series.dropna().astype(str).head(80)
    looks_like_date = sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() >= 0.70 if len(sample) else False
    if looks_like_date:
        parsed = pd.to_datetime(sample, errors="coerce")
        if len(parsed) > 0 and parsed.notna().mean() >= 0.85:
            return "datetime"
    return "categorical"


def _example_values(series: pd.Series, limit: int = 3) -> str:
    """Return a short example-value string for profiling tables."""
    values = series.dropna().astype(str).unique()[:limit]
    return ", ".join(values) if len(values) else ""


def profile_columns(
    df: pd.DataFrame,
    key_cols: str | Iterable[str] | None = None,
    y_cols: str | Iterable[str] | None = None,
    max_missing_ratio: float = 0.45,
    max_id_unique_ratio: float = 0.70,
    max_categorical_unique: int = 60,
) -> pd.DataFrame:
    """Create a lightweight column profile and analysis availability flag."""
    key_cols = set(_as_list(key_cols))
    y_cols = set(_as_list(y_cols))
    rows = []

    for column in df.columns:
        series = df[column]
        missing_ratio = float(series.isna().mean()) if len(series) else 0.0
        unique_count = int(series.nunique(dropna=True))
        unique_ratio = unique_count / max(len(series), 1)
        dtype_text = infer_basic_type(series)

        available = True
        reason = ""
        if column in key_cols:
            available = False
            reason = "Key column"
        elif column in y_cols:
            available = False
            reason = "Y/result column"
        elif missing_ratio > max_missing_ratio:
            available = False
            reason = "High missing ratio"
        elif unique_count <= 1:
            available = False
            reason = "Constant or single-value column"
        elif dtype_text == "datetime":
            available = False
            reason = "Datetime is profiled but not directly analyzed"
        elif (
            dtype_text == "categorical"
            and unique_ratio >= max_id_unique_ratio
            and unique_count > 30
        ):
            available = False
            reason = "ID-like high-cardinality column"
        elif dtype_text == "categorical" and unique_count > max_categorical_unique:
            available = False
            reason = "Too many categories"

        rows.append(
            {
                "column": column,
                "type": dtype_text,
                "missing_ratio": round(missing_ratio, 4),
                "unique_count": unique_count,
                "example_values": _example_values(series),
                "available": available,
                "exclude_reason": reason,
            }
        )

    return pd.DataFrame(rows)


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return compact row/column/missing statistics for KPI cards."""
    if df is None or df.empty:
        return {"rows": 0, "columns": 0, "missing_cells": 0, "missing_ratio": 0.0}
    total_cells = int(df.shape[0] * df.shape[1])
    if total_cells > 2_000_000:
        sample = df.iloc[: min(len(df), 1000), : min(df.shape[1], 1000)]
        missing_ratio = float(sample.isna().mean().mean() * 100) if sample.size else 0.0
        missing_cells = int(total_cells * missing_ratio / 100)
    else:
        missing_cells = int(df.isna().sum().sum())
        missing_ratio = float(missing_cells / total_cells * 100) if total_cells else 0.0
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_ratio": missing_ratio,
    }

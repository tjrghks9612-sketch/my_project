"""Merge utilities for X/Y process data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from core.models import MergeManifest


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item).strip() for item in value if str(item).strip()]


def duplicate_key_rows(df: pd.DataFrame, keys: Iterable[str]) -> int:
    key_cols = _as_list(keys)
    if df.empty or not key_cols:
        return 0
    return int(df.duplicated(subset=key_cols, keep=False).sum())


def key_tuples(df: pd.DataFrame, keys: Iterable[str]) -> pd.Series:
    key_cols = _as_list(keys)
    if not key_cols:
        return pd.Series([], dtype=object)
    key_frame = df[key_cols].astype("string").fillna("<NA>")
    return pd.Series(list(map(tuple, key_frame.to_numpy())), index=df.index)


def merge_tables_on_keys(tables: list[pd.DataFrame], key_cols: Iterable[str], label: str) -> tuple[pd.DataFrame, dict]:
    """Merge files of the same role. If no keys are selected, concatenate rows."""
    key_list = _as_list(key_cols)
    if not tables:
        return pd.DataFrame(), {"file_count": 0, "rows_after": 0, "duplicate_key_rows": 0}
    if len(tables) == 1:
        df = tables[0].copy()
        return df, {"file_count": 1, "rows_after": int(len(df)), "duplicate_key_rows": duplicate_key_rows(df, key_list)}
    if not key_list:
        df = pd.concat(tables, ignore_index=True)
        return df, {"file_count": len(tables), "rows_after": int(len(df)), "duplicate_key_rows": 0, "merge_mode": "concat"}

    merged = tables[0].copy()
    for index, table in enumerate(tables[1:], start=2):
        suffix = f"_{label}{index}"
        merged = pd.merge(merged, table.copy(), on=key_list, how="outer", suffixes=("", suffix))
    return merged, {
        "file_count": len(tables),
        "rows_after": int(len(merged)),
        "duplicate_key_rows": duplicate_key_rows(merged, key_list),
        "merge_mode": "outer_key_merge",
    }


def merge_x_y(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    x_keys: Iterable[str],
    y_keys: Iterable[str],
    how: str = "inner",
) -> tuple[pd.DataFrame, MergeManifest]:
    """Merge X and Y data and return merge statistics."""
    x_key_cols = _as_list(x_keys)
    y_key_cols = _as_list(y_keys)
    if not x_key_cols or not y_key_cols:
        raise ValueError("X/Y 병합 Key를 각각 1개 이상 입력하세요.")
    if len(x_key_cols) != len(y_key_cols):
        raise ValueError("복합 Key 병합에서는 X Key 수와 Y Key 수가 같아야 합니다.")
    missing_x = [col for col in x_key_cols if col not in x_df.columns]
    missing_y = [col for col in y_key_cols if col not in y_df.columns]
    if missing_x or missing_y:
        raise KeyError(f"없는 Key 컬럼입니다. X={missing_x}, Y={missing_y}")

    x_series = key_tuples(x_df, x_key_cols)
    y_series = key_tuples(y_df, y_key_cols)
    x_key_set = set(x_series)
    y_key_set = set(y_series)
    x_matched = int(x_series.isin(y_key_set).sum()) if len(x_df) else 0
    y_matched = int(y_series.isin(x_key_set).sum()) if len(y_df) else 0
    merged = pd.merge(x_df, y_df, left_on=x_key_cols, right_on=y_key_cols, how=how, suffixes=("", "_Y"))
    manifest = MergeManifest(
        x_rows=int(len(x_df)),
        y_rows=int(len(y_df)),
        merged_rows=int(len(merged)),
        x_merge_rate=float(x_matched / len(x_df) * 100) if len(x_df) else 0.0,
        y_merge_rate=float(y_matched / len(y_df) * 100) if len(y_df) else 0.0,
        x_duplicate_key_rows=duplicate_key_rows(x_df, x_key_cols),
        y_duplicate_key_rows=duplicate_key_rows(y_df, y_key_cols),
        x_unmatched_rows=int(len(x_df) - x_matched),
        y_unmatched_rows=int(len(y_df) - y_matched),
        x_keys=x_key_cols,
        y_keys=y_key_cols,
        merge_how=how,
    )
    return merged, manifest


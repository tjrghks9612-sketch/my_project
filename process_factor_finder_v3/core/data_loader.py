"""File loading and preview helpers.

Core modules intentionally avoid UI-framework imports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import pandas as pd

from core.models import DatasetManifest


def _name(file_or_path: str | Path | BinaryIO) -> str:
    return Path(getattr(file_or_path, "name", str(file_or_path))).name


def read_single_file(file_or_path: str | Path | BinaryIO) -> pd.DataFrame:
    """Read CSV, Excel, or Parquet into a DataFrame."""
    suffix = Path(_name(file_or_path)).suffix.lower()
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)
    if suffix == ".csv":
        return pd.read_csv(file_or_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_or_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_or_path)
    raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def preview_dataframe(df: pd.DataFrame, rows: int = 100, max_columns: int = 80) -> pd.DataFrame:
    """Return a bounded preview for UI rendering."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.iloc[: min(rows, len(df)), : min(max_columns, df.shape[1])]


def dataframe_summary(df: pd.DataFrame) -> dict:
    """Return lightweight dataframe statistics."""
    if df is None or df.empty:
        return {"rows": 0, "columns": 0, "missing_ratio": 0.0}
    sample = df.iloc[: min(len(df), 1000), : min(df.shape[1], 1000)]
    missing_ratio = float(sample.isna().mean().mean()) if sample.size else 0.0
    return {"rows": int(len(df)), "columns": int(df.shape[1]), "missing_ratio": missing_ratio}


def save_dataset(
    df: pd.DataFrame,
    cache_dir: str | Path,
    dataset_id: str | None = None,
    file_paths: list[str] | None = None,
    preview_rows: int = 100,
    preview_max_columns: int = 80,
) -> DatasetManifest:
    """Persist a DataFrame to parquet plus a small preview parquet."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    dataset_id = dataset_id or uuid4().hex[:12]
    parquet_path = cache_path / f"{dataset_id}.parquet"
    preview_path = cache_path / f"{dataset_id}__preview.parquet"
    df.to_parquet(parquet_path, index=False)
    preview_dataframe(df, preview_rows, preview_max_columns).to_parquet(preview_path, index=False)
    return DatasetManifest(
        dataset_id=dataset_id,
        file_paths=file_paths or [],
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        columns=[str(col) for col in df.columns],
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        preview_path=str(preview_path),
        parquet_path=str(parquet_path),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def load_manifest_frame(manifest: DatasetManifest | dict) -> pd.DataFrame:
    """Load the full dataframe represented by a DatasetManifest."""
    path = manifest.parquet_path if isinstance(manifest, DatasetManifest) else manifest["parquet_path"]
    return pd.read_parquet(path)


def load_manifest_preview(manifest: DatasetManifest | dict) -> pd.DataFrame:
    """Load the bounded preview dataframe represented by a DatasetManifest."""
    path = manifest.preview_path if isinstance(manifest, DatasetManifest) else manifest["preview_path"]
    return pd.read_parquet(path)

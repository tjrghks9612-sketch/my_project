"""Path-based dataframe storage for Streamlit sessions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.data_loader import load_manifest_frame, read_single_file, save_dataset
from core.models import DatasetManifest


class DataStore:
    """Persist uploaded dataframes as parquet and keep manifests in session."""

    def __init__(self, cache_dir: str | Path = "output/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_files(self, files, prefix: str, preview_rows: int = 100, preview_max_columns: int = 80) -> list[DatasetManifest]:
        manifests: list[DatasetManifest] = []
        for index, file in enumerate(files or []):
            df = read_single_file(file)
            name = getattr(file, "name", f"{prefix}_{index}")
            manifest = save_dataset(
                df,
                self.cache_dir,
                dataset_id=f"{prefix}_{index}_{abs(hash(name)) % 10_000_000}",
                file_paths=[name],
                preview_rows=preview_rows,
                preview_max_columns=preview_max_columns,
            )
            manifests.append(manifest)
        return manifests

    def save_frame(self, df: pd.DataFrame, dataset_id: str, file_paths: list[str] | None = None) -> DatasetManifest:
        return save_dataset(df, self.cache_dir, dataset_id=dataset_id, file_paths=file_paths or [])

    def load(self, manifest: DatasetManifest | dict) -> pd.DataFrame:
        return load_manifest_frame(manifest)

    def load_columns(self, manifest: DatasetManifest | dict, columns: list[str]) -> pd.DataFrame:
        """Load only selected parquet columns when possible."""
        path = manifest.parquet_path if isinstance(manifest, DatasetManifest) else manifest["parquet_path"]
        available = manifest.columns if isinstance(manifest, DatasetManifest) else manifest["columns"]
        requested = []
        seen = set()
        for column in columns:
            if column in available and column not in seen:
                requested.append(column)
                seen.add(column)
        if not requested:
            return pd.DataFrame()
        return pd.read_parquet(path, columns=requested)

    def load_many(self, manifests: list[DatasetManifest | dict]) -> list[pd.DataFrame]:
        return [self.load(manifest) for manifest in manifests]

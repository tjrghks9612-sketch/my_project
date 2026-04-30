"""Shared dataclasses and constants for v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PAIR_TYPES = {
    "numeric_numeric",
    "categorical_numeric",
    "numeric_categorical",
    "categorical_categorical",
}

SCREENING_METHODS_BY_PAIR_TYPE = {
    "numeric_numeric": {"pearson", "spearman"},
    "categorical_numeric": {"onehot_eta_squared"},
    "numeric_categorical": {"onehot_eta_squared", "point_biserial"},
    "categorical_categorical": {"onehot_cramers_v"},
}


@dataclass
class DatasetManifest:
    dataset_id: str
    file_paths: list[str]
    row_count: int
    column_count: int
    columns: list[str]
    dtypes: dict[str, str]
    preview_path: str
    parquet_path: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnProfile:
    column: str
    role: str
    inferred_type: str
    missing_rate: float
    unique_count: int
    available: bool
    exclude_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MergeManifest:
    x_rows: int
    y_rows: int
    merged_rows: int
    x_merge_rate: float
    y_merge_rate: float
    x_duplicate_key_rows: int
    y_duplicate_key_rows: int
    x_unmatched_rows: int
    y_unmatched_rows: int
    x_keys: list[str]
    y_keys: list[str]
    merge_how: str = "inner"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisPlan:
    mode: str = "full_scan"
    preset: str = "균형"
    include_categorical: bool = True
    top_n_per_y: int = 100
    detailed_top_n: int = 300
    final_top_n: int = 30
    y_target: str = ""
    x_columns: list[str] = field(default_factory=list)
    y_columns: list[str] = field(default_factory=list)
    max_category_levels: int = 30
    numeric_method: str = "pearson"
    x_chunk_size: int = 5000
    y_chunk_size: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScreeningCandidate:
    x_col: str
    y_col: str
    pair_type: str
    screening_method: str
    screening_score: float
    n_samples: int
    x_missing_rate: float
    y_missing_rate: float
    screening_rank_within_y: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.update(row.pop("metrics", {}))
        return row


@dataclass
class DetailedResult:
    rank: int
    pair_type: str
    x_col: str
    y_col: str
    final_score: float
    screening_score: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    model_score: float
    r2_score: float | None
    eta_squared: float | None
    cramer_v: float | None
    sample_n: int
    interpretation: str
    direction: str
    caution: str
    screening_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    output_dir: str
    status: str
    scanned_pairs: int
    screened_candidates: int
    detailed_candidates: int
    final_results: int
    started_at: str
    finished_at: str = ""
    artifact_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

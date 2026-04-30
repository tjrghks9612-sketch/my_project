"""Statistical factor ranking engine for Process Factor Finder."""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


DEFAULT_CONFIG = {
    "scoring": {
        "weights": {
            "statistical": 0.28,
            "effect": 0.26,
            "model": 0.18,
            "stability": 0.18,
            "quality": 0.10,
        }
    },
    "thresholds": {
        "very_significant": 90,
        "significant": 75,
        "needs_check": 60,
        "weak": 40,
    },
    "analysis": {
        "max_missing_ratio": 0.45,
        "min_group_n": 15,
        "max_categories": 60,
        "max_id_unique_ratio": 0.70,
        "min_sample_n": 30,
        "sample_score_full_n": 300,
    },
    "model": {"n_estimators": 250, "random_state": 42},
    "stability": {"bootstrap_iterations": 30, "sample_fraction": 0.80, "random_state": 42},
    "pairwise_analysis": {
        "enabled": True,
        "max_pairs_warning": 1000,
        "max_pairs_hard_limit": 10000,
        "default_top_n": 20,
        "apply_fdr_correction": True,
        "fast_screening": True,
        "matrix_only": False,
        "screening_top_k_per_y": 8,
        "screening_min_score": 25,
        "screening_max_pairs_total": 120,
        "fast_model_n_estimators": 80,
        "fast_bootstrap_iterations": 10,
        "chunk_x_size": 500,
        "chunk_y_size": 16,
        "batch_flush_rows": 5000,
        "save_intermediate_results": True,
        "save_parquet": True,
        "save_csv": True,
        "downcast_float32": False,
        "large_mode_default": True,
        "recommended_max_x_columns": 3000,
        "recommended_max_y_columns": 100,
    },
    "matrix_screening": {
        "enabled": True,
        "method": "pearson",
        "dtype": "float32",
        "missing_strategy": "mean_impute",
        "top_n_per_y": 100,
        "x_chunk_size": 5000,
        "y_chunk_size": 50,
        "min_variance": 1e-12,
        "max_missing_rate": 0.7,
        "save_candidates": True,
        "candidate_output_name": "matrix_screening_candidates.parquet",
    },
    "plot": {
        "show_all_categories_default": True,
        "default_tick_angle": -45,
        "max_categories_without_warning": 30,
        "auto_height_per_category": 18,
        "min_chart_height": 420,
        "max_chart_height": 900,
    },
    "interpretation": {
        "avoid_group_word": True,
        "emphasize_candidate_not_cause": True,
        "format_integer_like_float": True,
    },
}


def load_config(path: str | Path = "config.yaml") -> dict:
    """Load YAML config and merge it with safe defaults."""
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return _deep_merge(DEFAULT_CONFIG.copy(), loaded)


def _deep_merge(base: dict, update: dict) -> dict:
    """Merge nested dictionaries without losing unspecified defaults."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    """Normalize a string or iterable into a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [item for item in value if item]


def _safe_float(value, default: float = np.nan) -> float:
    """Convert numeric-like values to float while protecting the app."""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def classify_y_type(series: pd.Series) -> str:
    """Classify Y as numeric, binary, or categorical."""
    clean = series.dropna()
    unique_count = clean.nunique()
    if unique_count == 2:
        return "binary"
    if pd.api.types.is_bool_dtype(clean):
        return "binary"
    if pd.api.types.is_numeric_dtype(clean):
        return "numeric"
    return "categorical"


def classify_feature_type(series: pd.Series, config: dict | None = None, column_name: str = "") -> str:
    """Classify X as numeric, categorical, datetime, or id_like."""
    config = config or DEFAULT_CONFIG
    analysis_cfg = config.get("analysis", {})
    clean = series.dropna()
    unique_count = int(clean.nunique())
    unique_ratio = unique_count / max(len(series), 1)
    lower_name = column_name.lower()

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if not pd.api.types.is_numeric_dtype(series) and len(clean) > 0 and ("date" in lower_name or "time" in lower_name):
        parsed = pd.to_datetime(clean.astype(str).head(100), errors="coerce")
        if parsed.notna().mean() >= 0.85:
            return "datetime"

    id_name_hint = any(token in lower_name for token in ["ssn", "id", "barcode", "serial"])
    if id_name_hint and unique_count > max(30, len(series) * 0.04):
        return "id_like"
    if (
        not pd.api.types.is_numeric_dtype(series)
        and unique_ratio >= analysis_cfg.get("max_id_unique_ratio", 0.70)
        and unique_count > 30
    ):
        return "id_like"

    if lower_name == "slot":
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        if unique_count <= 12:
            return "categorical"
        return "numeric"
    return "categorical"


def analyze_factors(
    df: pd.DataFrame,
    y_col: str,
    key_cols: str | Iterable[str] | None = None,
    y_cols: str | Iterable[str] | None = None,
    exclude_cols: str | Iterable[str] | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Rank candidate X factors for a selected Y column."""
    config = config or DEFAULT_CONFIG
    key_cols = set(_as_list(key_cols))
    y_cols = set(_as_list(y_cols))
    exclude_cols = set(_as_list(exclude_cols))
    protected_cols = key_cols | y_cols | exclude_cols | {y_col}

    if y_col not in df.columns:
        raise KeyError(f"Y column not found: {y_col}")

    y_type = classify_y_type(df[y_col])
    candidates = []
    excluded_rows = []
    for column in df.columns:
        if column in protected_cols:
            continue
        feature_type = classify_feature_type(df[column], config, column)
        reason = _exclusion_reason(df[column], feature_type, config)
        if reason:
            excluded_rows.append((column, reason))
            continue
        candidates.append({"feature": column, "feature_type": feature_type})

    if not candidates:
        return pd.DataFrame(columns=_result_columns())

    model_scores = _random_forest_importance(df, candidates, y_col, y_type, config)
    rows = []
    for candidate in candidates:
        feature = candidate["feature"]
        feature_type = candidate["feature_type"]
        pair_stats = _compute_pair_stats(df, feature, y_col, feature_type, y_type, config)
        if not pair_stats["valid"]:
            continue

        stat_score = _statistical_score(pair_stats["p_value"])
        effect_score = _effect_score(pair_stats["effect_size"], pair_stats["effect_metric"])
        stability_score = _bootstrap_stability(
            df,
            feature,
            y_col,
            feature_type,
            y_type,
            pair_stats["effect_size"],
            config,
        )
        quality_score = _quality_score(
            pair_stats["missing_ratio"],
            pair_stats["sample_n"],
            pair_stats["min_group_n"],
            feature_type,
            config,
        )
        model_score = float(model_scores.get(feature, 0.0))

        weights = config.get("scoring", {}).get("weights", DEFAULT_CONFIG["scoring"]["weights"])
        final_score = (
            weights.get("statistical", 0.28) * stat_score
            + weights.get("effect", 0.26) * effect_score
            + weights.get("model", 0.18) * model_score
            + weights.get("stability", 0.18) * stability_score
            + weights.get("quality", 0.10) * quality_score
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "judgement": _judgement(final_score, config),
                "final_score": round(float(final_score), 2),
                "stat_score": round(float(stat_score), 2),
                "effect_score": round(float(effect_score), 2),
                "model_score": round(float(model_score), 2),
                "stability_score": round(float(stability_score), 2),
                "quality_score": round(float(quality_score), 2),
                "p_value": pair_stats["p_value"],
                "effect_size": round(float(pair_stats["effect_size"]), 5),
                "sample_n": int(pair_stats["sample_n"]),
                "min_group_n": int(pair_stats["min_group_n"]),
                "missing_ratio": round(float(pair_stats["missing_ratio"]), 4),
                "direction": pair_stats["direction"],
                "metric_text": pair_stats["metric_text"],
                "caution_text": _caution_text(pair_stats, config),
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=_result_columns())
    result = result.sort_values(["final_score", "effect_score", "model_score"], ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result[_result_columns()]


def get_candidate_columns(
    df: pd.DataFrame,
    key_cols: str | Iterable[str] | None = None,
    exclude_cols: str | Iterable[str] | None = None,
    config: dict | None = None,
    role: str = "x",
) -> pd.DataFrame:
    """Return columns that are usable as X or Y candidates with exclusion reasons."""
    config = config or DEFAULT_CONFIG
    key_cols = set(_as_list(key_cols))
    exclude_cols = set(_as_list(exclude_cols))
    rows = []

    for column in df.columns:
        feature_type = classify_y_type(df[column]) if role == "y" else classify_feature_type(df[column], config, column)
        reason = ""
        if column in key_cols:
            reason = "Key 컬럼"
        elif column in exclude_cols:
            reason = "사용자 제외 컬럼"
        elif column.startswith("_") or "source_file" in column.lower():
            reason = "시스템 보조 컬럼"
        elif _is_datetime_like(df[column], column):
            reason = "날짜/시간 컬럼"
        else:
            reason = _exclusion_reason(df[column], feature_type, config)

        rows.append(
            {
                "column": column,
                "type": feature_type,
                "available": reason == "",
                "exclude_reason": reason,
                "missing_ratio": float(df[column].isna().mean()) if len(df) else 1.0,
                "unique_count": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def estimate_pairwise_risk(total_pairs: int) -> str:
    """Estimate runtime risk from total pair count."""
    if total_pairs < 5_000:
        return "low"
    if total_pairs < 50_000:
        return "medium"
    if total_pairs < 250_000:
        return "high"
    return "very high"


def downcast_numeric_frame(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Optionally reduce numeric precision to lower memory usage for wide data."""
    target_columns = list(columns) if columns is not None else df.columns.tolist()
    converted = df.copy(deep=False)
    for column in target_columns:
        if column not in converted.columns:
            continue
        series = converted[column]
        if pd.api.types.is_float_dtype(series):
            converted[column] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_integer_dtype(series):
            converted[column] = pd.to_numeric(series, downcast="integer")
    return converted


def matrix_numeric_screening(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_cols: Iterable[str],
    config: dict | None = None,
    progress_callback=None,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Screen numeric-numeric X/Y pairs with chunked NumPy matrix correlation.

    The returned candidates are only first-pass candidates. Final p-values,
    effect size, model score, stability, and final score still come from the
    existing detailed analysis path.
    """
    config = config or DEFAULT_CONFIG
    matrix_cfg = config.get("matrix_screening", DEFAULT_CONFIG["matrix_screening"])
    method = str(matrix_cfg.get("method", "pearson")).lower()
    dtype = np.float32 if str(matrix_cfg.get("dtype", "float32")).lower() == "float32" else np.float64
    missing_strategy = str(matrix_cfg.get("missing_strategy", "mean_impute"))
    top_n = max(1, int(matrix_cfg.get("top_n_per_y", 100)))
    x_chunk_size = max(1, int(matrix_cfg.get("x_chunk_size", 5000)))
    y_chunk_size = max(1, int(matrix_cfg.get("y_chunk_size", 50)))
    max_missing_rate = float(matrix_cfg.get("max_missing_rate", 0.7))
    min_variance = float(matrix_cfg.get("min_variance", 1e-12))
    output_root = Path(output_dir) if output_dir else None

    started_at = time.time()
    dropped_rows = []
    log_rows = []
    fallback = False

    try:
        x_valid, x_dropped = _matrix_filter_numeric_columns(
            df, list(x_cols), "X", max_missing_rate, min_variance
        )
        y_valid, y_dropped = _matrix_filter_numeric_columns(
            df, list(y_cols), "Y", max_missing_rate, min_variance
        )
        dropped_rows.extend(x_dropped)
        dropped_rows.extend(y_dropped)

        total_pairs = len(x_valid) * len(y_valid)
        if total_pairs == 0:
            return pd.DataFrame(columns=_matrix_candidate_columns()), pd.DataFrame(dropped_rows), pd.DataFrame(), False

        if method == "spearman":
            log_rows.append(
                _screening_log_row(
                    "matrix_screening",
                    "Spearman rank screening selected; rank transform can be slower than Pearson.",
                    0,
                    0,
                    0,
                    0,
                    0,
                    started_at,
                )
            )

        x_chunks = list(_chunk_list(x_valid, x_chunk_size))
        y_chunks = list(_chunk_list(y_valid, y_chunk_size))
        top_by_y: dict[str, list[dict]] = {y_col: [] for y_col in y_valid}
        processed_pairs = 0

        for y_chunk_index, y_chunk in enumerate(y_chunks, start=1):
            for x_chunk_index, x_chunk in enumerate(x_chunks, start=1):
                chunk_started = time.time()
                x_matrix, x_meta, y_matrix, y_meta = _prepare_matrix_pair_chunk(
                    df,
                    x_chunk,
                    y_chunk,
                    dtype=dtype,
                    missing_strategy=missing_strategy,
                    method=method,
                )
                n_samples = int(x_matrix.shape[0])
                if n_samples < 3 or x_matrix.size == 0 or y_matrix.size == 0:
                    continue

                corr = _matrix_corr(x_matrix, y_matrix)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                abs_corr = np.abs(corr)
                r2 = corr * corr
                processed_pairs += int(corr.size)

                k = min(top_n, corr.shape[0])
                if k > 0:
                    for y_idx, y_col in enumerate(y_chunk):
                        scores = abs_corr[:, y_idx]
                        if len(scores) > k:
                            candidate_idx = np.argpartition(scores, -k)[-k:]
                        else:
                            candidate_idx = np.arange(len(scores))
                        for x_idx in candidate_idx:
                            x_col = x_chunk[int(x_idx)]
                            corr_value = float(corr[int(x_idx), y_idx])
                            abs_value = float(abs_corr[int(x_idx), y_idx])
                            r2_value = float(r2[int(x_idx), y_idx])
                            score = float(np.clip(abs_value * 100.0, 0.0, 100.0))
                            top_by_y[y_col].append(
                                {
                                    "x_col": x_col,
                                    "y_col": y_col,
                                    "x_type": "numeric",
                                    "y_type": "numeric",
                                    "screening_method": method,
                                    "screening_score": score,
                                    "corr": corr_value,
                                    "abs_corr": abs_value,
                                    "r2": r2_value,
                                    "analysis_type": "numeric_numeric_matrix",
                                    "n_samples": n_samples,
                                    "x_missing_rate": float(x_meta[x_col]["missing_rate"]),
                                    "y_missing_rate": float(y_meta[y_col]["missing_rate"]),
                                }
                            )
                        top_by_y[y_col] = sorted(
                            top_by_y[y_col],
                            key=lambda row: (row["screening_score"], row["r2"]),
                            reverse=True,
                        )[:top_n]

                elapsed = max(time.time() - started_at, 1e-9)
                if progress_callback:
                    progress_callback(
                        processed_pairs,
                        max(total_pairs, 1),
                        x_chunk[0] if x_chunk else "",
                        y_chunk[0] if y_chunk else "",
                        "matrix screening",
                        {
                            "x_chunk_index": x_chunk_index,
                            "x_chunk_total": len(x_chunks),
                            "y_chunk_index": y_chunk_index,
                            "y_chunk_total": len(y_chunks),
                            "pairs_per_sec": processed_pairs / elapsed,
                            "eta_seconds": max(total_pairs - processed_pairs, 0) / max(processed_pairs / elapsed, 1e-9),
                            "candidate_count": sum(len(rows) for rows in top_by_y.values()),
                        },
                    )

                log_rows.append(
                    _screening_log_row(
                        "matrix_screening",
                        f"processed numeric matrix chunk in {time.time() - chunk_started:.2f}s",
                        (x_chunk_index - 1) * x_chunk_size,
                        (x_chunk_index - 1) * x_chunk_size + len(x_chunk),
                        (y_chunk_index - 1) * y_chunk_size,
                        (y_chunk_index - 1) * y_chunk_size + len(y_chunk),
                        processed_pairs,
                        started_at,
                    )
                )

        candidate_rows = [row for rows in top_by_y.values() for row in rows]
        candidates = pd.DataFrame(candidate_rows, columns=_matrix_candidate_columns())
        candidates = candidates.sort_values(["y_col", "screening_score", "r2"], ascending=[True, False, False]).reset_index(drop=True)

        dropped_df = pd.DataFrame(dropped_rows)
        log_df = pd.DataFrame(log_rows)
        if output_root:
            output_root.mkdir(parents=True, exist_ok=True)
            if not dropped_df.empty:
                dropped_df.to_csv(output_root / "dropped_columns.csv", index=False, encoding="utf-8-sig")
            if not log_df.empty:
                log_df.to_csv(output_root / "screening_log.csv", index=False, encoding="utf-8-sig")
            if bool(matrix_cfg.get("save_candidates", True)) and not candidates.empty:
                parquet_name = str(matrix_cfg.get("candidate_output_name", "matrix_screening_candidates.parquet"))
                parquet_path = output_root / parquet_name
                csv_path = output_root / parquet_name.replace(".parquet", ".csv")
                try:
                    candidates.to_parquet(parquet_path, index=False)
                except Exception:
                    pass
                candidates.to_csv(csv_path, index=False, encoding="utf-8-sig")

        return candidates, dropped_df, log_df, fallback
    except MemoryError:
        fallback = True
        log_rows.append(
            _screening_log_row(
                "matrix_screening_fallback",
                "MemoryError: reduce matrix chunk size. Falling back to existing pair-loop screening.",
                0,
                0,
                0,
                0,
                0,
                started_at,
            )
        )
        return pd.DataFrame(columns=_matrix_candidate_columns()), pd.DataFrame(dropped_rows), pd.DataFrame(log_rows), fallback
    except Exception as exc:
        fallback = True
        log_rows.append(
            _screening_log_row(
                "matrix_screening_fallback",
                f"{type(exc).__name__}: {exc}. Falling back to existing pair-loop screening.",
                0,
                0,
                0,
                0,
                0,
                started_at,
            )
        )
        return pd.DataFrame(columns=_matrix_candidate_columns()), pd.DataFrame(dropped_rows), pd.DataFrame(log_rows), fallback


def _matrix_candidate_columns() -> list[str]:
    """Return matrix screening candidate column order."""
    return [
        "x_col",
        "y_col",
        "x_type",
        "y_type",
        "screening_method",
        "screening_score",
        "corr",
        "abs_corr",
        "r2",
        "analysis_type",
        "n_samples",
        "x_missing_rate",
        "y_missing_rate",
    ]


def _matrix_filter_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    source: str,
    max_missing_rate: float,
    min_variance: float,
) -> tuple[list[str], list[dict]]:
    """Keep numeric columns suitable for matrix screening and log dropped columns."""
    valid = []
    dropped = []
    for column in columns:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        missing_rate = float((numeric.isna() | ~np.isfinite(numeric)).mean()) if len(numeric) else 1.0
        unique_count = int(numeric.nunique(dropna=True))
        variance = float(numeric.var(skipna=True)) if unique_count > 1 else 0.0
        reason = ""
        if missing_rate > max_missing_rate:
            reason = "missing_rate_above_matrix_threshold"
        elif unique_count <= 1:
            reason = "constant_numeric_column"
        elif not np.isfinite(variance) or variance <= min_variance:
            reason = "low_or_zero_variance"

        if reason:
            dropped.append(
                {
                    "column_name": column,
                    "source": source,
                    "reason": reason,
                    "missing_rate": missing_rate,
                    "unique_count": unique_count,
                    "variance": variance,
                }
            )
        else:
            valid.append(column)
    return valid, dropped


def _prepare_matrix_pair_chunk(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    dtype,
    missing_strategy: str,
    method: str,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    """Build aligned X/Y matrices for one chunk and apply missing strategy."""
    x_frame = df.loc[:, x_cols].apply(pd.to_numeric, errors="coerce")
    y_frame = df.loc[:, y_cols].apply(pd.to_numeric, errors="coerce")
    x_meta = _matrix_column_meta(x_frame)
    y_meta = _matrix_column_meta(y_frame)

    if missing_strategy == "drop_rows":
        valid_rows = ~(x_frame.isna().any(axis=1) | y_frame.isna().any(axis=1))
        x_frame = x_frame.loc[valid_rows]
        y_frame = y_frame.loc[valid_rows]
    elif missing_strategy == "median_impute":
        x_frame = x_frame.fillna(x_frame.median(numeric_only=True))
        y_frame = y_frame.fillna(y_frame.median(numeric_only=True))
    else:
        x_frame = x_frame.fillna(x_frame.mean(numeric_only=True))
        y_frame = y_frame.fillna(y_frame.mean(numeric_only=True))

    x_matrix = x_frame.to_numpy(dtype=dtype, copy=True)
    y_matrix = y_frame.to_numpy(dtype=dtype, copy=True)
    x_matrix = np.nan_to_num(x_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    y_matrix = np.nan_to_num(y_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if method == "spearman":
        x_matrix = _rank_matrix(x_matrix, dtype)
        y_matrix = _rank_matrix(y_matrix, dtype)
    return x_matrix, x_meta, y_matrix, y_meta


def _matrix_column_meta(frame: pd.DataFrame) -> dict:
    """Collect lightweight per-column matrix metadata."""
    meta = {}
    for column in frame.columns:
        series = frame[column]
        meta[column] = {
            "missing_rate": float(series.isna().mean()) if len(series) else 1.0,
            "variance": float(series.var(skipna=True)) if series.nunique(dropna=True) > 1 else 0.0,
        }
    return meta


def _rank_matrix(matrix: np.ndarray, dtype) -> np.ndarray:
    """Rank-transform each column for Spearman-style matrix screening."""
    ranked = pd.DataFrame(matrix).rank(axis=0, method="average", na_option="keep").to_numpy(dtype=dtype, copy=True)
    return np.nan_to_num(ranked, nan=0.0, posinf=0.0, neginf=0.0)


def _matrix_corr(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    """Calculate Pearson correlation matrix via centered matrix multiplication."""
    x_centered = x_matrix - x_matrix.mean(axis=0, keepdims=True)
    y_centered = y_matrix - y_matrix.mean(axis=0, keepdims=True)
    numerator = x_centered.T @ y_centered
    x_norm = np.sqrt(np.sum(x_centered * x_centered, axis=0))
    y_norm = np.sqrt(np.sum(y_centered * y_centered, axis=0))
    denom = np.outer(x_norm, y_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denom
    return np.clip(corr, -1.0, 1.0)


def _screening_log_row(
    stage: str,
    message: str,
    x_chunk_start: int,
    x_chunk_end: int,
    y_chunk_start: int,
    y_chunk_end: int,
    processed_pairs: int,
    started_at: float,
) -> dict:
    """Build one row for screening_log.csv."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "message": message,
        "x_chunk_start": int(x_chunk_start),
        "x_chunk_end": int(x_chunk_end),
        "y_chunk_start": int(y_chunk_start),
        "y_chunk_end": int(y_chunk_end),
        "processed_pairs": int(processed_pairs),
        "elapsed_seconds": round(float(time.time() - started_at), 3),
    }


def analyze_pairwise_chunked(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_cols: Iterable[str],
    key_cols: str | Iterable[str] | None = None,
    exclude_cols: str | Iterable[str] | None = None,
    config: dict | None = None,
    progress_callback=None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run pairwise analysis in chunks and flush intermediate results to disk."""
    config = config or DEFAULT_CONFIG
    pairwise_cfg = config.get("pairwise_analysis", DEFAULT_CONFIG["pairwise_analysis"])
    key_cols = set(_as_list(key_cols))
    exclude_cols = set(_as_list(exclude_cols))
    x_cols = [col for col in x_cols if col in df.columns]
    y_cols = [col for col in y_cols if col in df.columns]

    output_root = Path(output_dir) if output_dir else None
    writer = _ResultBatchWriter(output_root, pairwise_cfg) if output_root else None

    x_profile = get_candidate_columns(df, key_cols=key_cols, exclude_cols=exclude_cols | set(y_cols), config=config, role="x")
    y_profile = get_candidate_columns(df, key_cols=key_cols, exclude_cols=exclude_cols, config=config, role="y")

    valid_x = x_profile[(x_profile["available"]) & (x_profile["column"].isin(x_cols))][["column", "type"]].to_dict("records")
    valid_y = y_profile[(y_profile["available"]) & (y_profile["column"].isin(y_cols))]["column"].tolist()
    valid_y_types = {y_col: classify_y_type(df[y_col]) for y_col in valid_y}

    excluded_log = []
    for _, row in x_profile[~x_profile["available"] & x_profile["column"].isin(x_cols)].iterrows():
        excluded_log.append({"role": "x", "column": row["column"], "reason": row["exclude_reason"]})
    for _, row in y_profile[~y_profile["available"] & y_profile["column"].isin(y_cols)].iterrows():
        excluded_log.append({"role": "y", "column": row["column"], "reason": row["exclude_reason"]})
    if writer and excluded_log:
        writer.write_removed_columns(pd.DataFrame(excluded_log))

    if pairwise_cfg.get("downcast_float32", False):
        downcast_cols = {item["column"] for item in valid_x} | set(valid_y)
        df = downcast_numeric_frame(df, downcast_cols)

    total_pairs = sum(1 for x_item in valid_x for y_col in valid_y if x_item["column"] != y_col)
    chunk_x_size = max(1, int(pairwise_cfg.get("chunk_x_size", 500)))
    chunk_y_size = max(1, int(pairwise_cfg.get("chunk_y_size", 16)))
    batch_flush_rows = max(100, int(pairwise_cfg.get("batch_flush_rows", 5000)))
    fast_screening = bool(pairwise_cfg.get("fast_screening", True))
    matrix_only = bool(pairwise_cfg.get("matrix_only", False))
    score_config = _pairwise_fast_config(config, pairwise_cfg) if fast_screening else config

    screening_buffer = []
    detailed_rows = []
    error_rows = []
    processed_pairs = 0
    started_at = time.time()
    y_chunks = list(_chunk_list(valid_y, chunk_y_size))
    matrix_cfg = config.get("matrix_screening", DEFAULT_CONFIG["matrix_screening"])
    use_matrix = bool(matrix_cfg.get("enabled", True)) and fast_screening
    numeric_x_set: set[str] = set()
    numeric_y_set: set[str] = set()
    matrix_pair_count = 0
    matrix_active = False
    matrix_shortlist_by_y: dict[str, list[dict]] = {}
    matrix_meta = {
        "used_matrix_screening": False,
        "matrix_candidate_count": 0,
        "matrix_detailed_candidate_count": 0,
        "matrix_fallback": False,
        "matrix_candidate_csv": "",
        "matrix_candidate_parquet": "",
        "matrix_dropped_columns_csv": "",
        "matrix_screening_log_csv": "",
    }

    if use_matrix:
        numeric_x = [str(item["column"]) for item in valid_x if str(item["type"]) == "numeric"]
        numeric_y = [y_col for y_col in valid_y if valid_y_types.get(y_col) == "numeric"]
        numeric_x_set = set(numeric_x)
        numeric_y_set = set(numeric_y)
        matrix_pair_count = sum(1 for y_col in numeric_y for x_col in numeric_x if x_col != y_col)
        if matrix_pair_count:
            candidates, dropped_df, log_df, fallback = matrix_numeric_screening(
                df,
                numeric_x,
                numeric_y,
                config=config,
                progress_callback=progress_callback,
                output_dir=output_root,
            )
            matrix_meta["used_matrix_screening"] = not fallback
            matrix_meta["matrix_fallback"] = fallback
            matrix_meta["matrix_candidate_count"] = int(len(candidates))
            if output_root:
                candidate_name = str(matrix_cfg.get("candidate_output_name", "matrix_screening_candidates.parquet"))
                matrix_meta["matrix_candidate_parquet"] = str(output_root / candidate_name)
                matrix_meta["matrix_candidate_csv"] = str(output_root / candidate_name.replace(".parquet", ".csv"))
                matrix_meta["matrix_dropped_columns_csv"] = str(output_root / "dropped_columns.csv")
                matrix_meta["matrix_screening_log_csv"] = str(output_root / "screening_log.csv")
            if fallback:
                numeric_x_set = set()
                numeric_y_set = set()
                matrix_pair_count = 0
            else:
                matrix_active = True
                created_at = datetime.now().isoformat(timespec="seconds")
                detail_top_k = max(1, int(pairwise_cfg.get("screening_top_k_per_y", 8)))
                detail_max_total = max(detail_top_k, int(pairwise_cfg.get("screening_max_pairs_total", 120)))
                detail_candidates = (
                    candidates.sort_values(["y_col", "screening_score", "r2"], ascending=[True, False, False])
                    .groupby("y_col", as_index=False, group_keys=False)
                    .head(detail_top_k)
                    .sort_values(["screening_score", "r2"], ascending=False)
                    .head(detail_max_total)
                    .reset_index(drop=True)
                )
                matrix_meta["matrix_detailed_candidate_count"] = int(len(detail_candidates))
                for _, candidate in detail_candidates.iterrows():
                    x_col = str(candidate["x_col"])
                    y_col = str(candidate["y_col"])
                    row = {
                        "x_col": x_col,
                        "y_col": y_col,
                        "x_type": "numeric",
                        "y_type": "numeric",
                        "analysis_type": "numeric_numeric_matrix",
                        "final_score": round(float(candidate["screening_score"]), 2),
                        "stat_score": 0.0,
                        "effect_score": round(float(candidate["abs_corr"]) * 100.0, 2),
                        "model_score": 0.0,
                        "stability_score": 0.0,
                        "quality_score": 0.0,
                        "p_value": np.nan,
                        "adjusted_p_value": np.nan,
                        "effect_size": round(float(candidate["abs_corr"]), 5),
                        "r2_score": float(candidate["r2"]),
                        "n_samples": int(candidate["n_samples"]),
                        "sample_n": int(candidate["n_samples"]),
                        "min_group_n": int(candidate["n_samples"]),
                        "missing_rate_x": float(candidate["x_missing_rate"]),
                        "missing_rate_y": float(candidate["y_missing_rate"]),
                        "missing_ratio_x": float(candidate["x_missing_rate"]),
                        "missing_ratio_y": float(candidate["y_missing_rate"]),
                        "direction": "",
                        "metric_text": f"{candidate['screening_method']} corr={float(candidate['corr']):.3f}, R2={float(candidate['r2']):.3f}",
                        "caution_text": "Matrix screening candidate; final scoring is calculated in detailed analysis.",
                        "created_at": created_at,
                        "screening_method": str(candidate["screening_method"]),
                        "screening_score": float(candidate["screening_score"]),
                        "screening_corr": float(candidate["corr"]),
                        "screening_r2": float(candidate["r2"]),
                        "used_matrix_screening": True,
                    }
                    screening_buffer.append(row)
                    matrix_shortlist_by_y.setdefault(y_col, []).append(row)
                if writer and not dropped_df.empty:
                    dropped_for_removed = dropped_df.rename(
                        columns={"column_name": "column", "source": "role"}
                    )
                    writer.write_removed_columns(dropped_for_removed)

    loop_total_pairs = max(total_pairs - matrix_pair_count if matrix_active else total_pairs, 1)
    if not matrix_only:
        for y_chunk_index, y_chunk in enumerate(y_chunks, start=1):
            shortlisted_by_y = {y_col: [] for y_col in y_chunk}
            for y_col in y_chunk:
                y_type = valid_y_types.get(y_col) or classify_y_type(df[y_col])
                if matrix_active and y_col in numeric_y_set:
                    x_items_for_y = [
                        item
                        for item in valid_x
                        if str(item["column"]) != y_col and str(item["column"]) not in numeric_x_set
                    ]
                else:
                    x_items_for_y = [item for item in valid_x if str(item["column"]) != y_col]
                x_chunk_total = max(1, int(math.ceil(len(x_items_for_y) / chunk_x_size)))
                for x_chunk_index, x_chunk in enumerate(_chunk_list(x_items_for_y, chunk_x_size), start=1):
                    for x_item in x_chunk:
                        x_col = str(x_item["column"])
                        processed_pairs += 1
                        try:
                            pair_stats = _compute_pair_stats(df, x_col, y_col, str(x_item["type"]), y_type, config)
                            if not pair_stats["valid"]:
                                continue

                            stat_score = _statistical_score(pair_stats["p_value"])
                            effect_score = _effect_score(pair_stats["effect_size"], pair_stats["effect_metric"])
                            quality_score = _quality_score(
                                pair_stats["missing_ratio"],
                                pair_stats["sample_n"],
                                pair_stats["min_group_n"],
                                str(x_item["type"]),
                                config,
                            )
                            screening_score = _screening_score(stat_score, effect_score, quality_score, config)
                            screening_row = _screening_result_row(
                                x_col=x_col,
                                y_col=y_col,
                                x_type=str(x_item["type"]),
                                y_type=y_type,
                                pair_stats=pair_stats,
                                stat_score=stat_score,
                                effect_score=effect_score,
                                quality_score=quality_score,
                                screening_score=screening_score,
                                missing_ratio_x=float(df[x_col].isna().mean()) if len(df) else 1.0,
                                missing_ratio_y=float(df[y_col].isna().mean()) if len(df) else 1.0,
                                created_at=datetime.now().isoformat(timespec="seconds"),
                            )
                            screening_buffer.append(screening_row)
                            shortlisted_by_y[y_col] = _update_shortlist(
                                shortlisted_by_y[y_col],
                                screening_row,
                                pairwise_cfg,
                            )
                        except Exception as exc:
                            error_rows.append(
                                {
                                    "x_col": x_col,
                                    "y_col": y_col,
                                    "phase": "screening",
                                    "error": repr(exc),
                                    "created_at": datetime.now().isoformat(timespec="seconds"),
                                }
                            )

                        if progress_callback:
                            elapsed = max(time.time() - started_at, 1e-9)
                            progress_callback(
                                processed_pairs,
                                loop_total_pairs,
                                x_col,
                                y_col,
                                "빠른 선별",
                                {
                                    "x_chunk_index": x_chunk_index,
                                    "x_chunk_total": x_chunk_total,
                                    "y_chunk_index": y_chunk_index,
                                    "y_chunk_total": len(y_chunks),
                                    "pairs_per_sec": processed_pairs / elapsed,
                                    "eta_seconds": max(loop_total_pairs - processed_pairs, 0) / max(processed_pairs / elapsed, 1e-9),
                                },
                            )

                        if writer and len(screening_buffer) >= batch_flush_rows:
                            writer.append_results(pd.DataFrame(screening_buffer))
                            screening_buffer = []
                        if writer and len(error_rows) >= 100:
                            writer.append_errors(pd.DataFrame(error_rows))
                            error_rows = []

            for y_col, shortlist in shortlisted_by_y.items():
                if not shortlist:
                    continue
                matrix_shortlist_by_y.setdefault(y_col, []).extend(shortlist)

    for y_col, shortlist in matrix_shortlist_by_y.items():
        if not shortlist:
            continue
        y_type = valid_y_types.get(y_col) or classify_y_type(df[y_col])
        selected_candidates = [{"feature": row["x_col"], "feature_type": row["x_type"]} for row in shortlist]
        model_scores = _random_forest_importance(df, selected_candidates, y_col, y_type, score_config)
        for shortlisted in shortlist:
            try:
                pair_stats = shortlisted.get("pair_stats")
                if pair_stats is None:
                    pair_stats = _compute_pair_stats(
                        df,
                        shortlisted["x_col"],
                        y_col,
                        shortlisted["x_type"],
                        y_type,
                        config,
                    )
                if not pair_stats.get("valid", False):
                    continue
                scored = _score_pair_from_stats(
                    df=df,
                    feature=shortlisted["x_col"],
                    y_col=y_col,
                    feature_type=shortlisted["x_type"],
                    y_type=y_type,
                    config=score_config,
                    pair_stats=pair_stats,
                    model_score=float(model_scores.get(shortlisted["x_col"], 0.0)),
                )
                scored.update(
                    {
                        "x_feature": shortlisted["x_col"],
                        "y_target": y_col,
                        "x_type": shortlisted["x_type"],
                        "y_type": y_type,
                        "missing_ratio_x": shortlisted["missing_rate_x"],
                        "missing_ratio_y": shortlisted["missing_rate_y"],
                        "p_value_adj": np.nan,
                        "easy_interpretation": "",
                        "analysis_type": shortlisted.get("analysis_type", pair_stats["effect_metric"]),
                        "r2_score": shortlisted.get("screening_r2", np.nan),
                        "created_at": shortlisted["created_at"],
                        "screening_method": shortlisted.get("screening_method", ""),
                        "screening_score": shortlisted.get("screening_score", shortlisted.get("final_score", np.nan)),
                        "screening_corr": shortlisted.get("screening_corr", np.nan),
                        "screening_r2": shortlisted.get("screening_r2", np.nan),
                        "used_matrix_screening": bool(shortlisted.get("used_matrix_screening", False)),
                    }
                )
                scored["easy_interpretation"] = build_easy_interpretation(
                    x_feature=shortlisted["x_col"],
                    y_target=y_col,
                    row=scored,
                    pair_mode=True,
                )
                detailed_rows.append(scored)
            except Exception as exc:
                error_rows.append(
                    {
                        "x_col": shortlisted["x_col"],
                        "y_col": y_col,
                        "phase": "detailed",
                        "error": repr(exc),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )

    if writer and screening_buffer:
        writer.append_results(pd.DataFrame(screening_buffer))
    if writer and error_rows:
        writer.append_errors(pd.DataFrame(error_rows))

    if writer:
        writer.finalize()
        if writer.csv_path and writer.csv_path.exists():
            p_values = pd.read_csv(writer.csv_path, usecols=["p_value"])
            adjusted = _benjamini_hochberg(p_values["p_value"])
            writer.write_adjusted_p_values(adjusted)

    result = pd.DataFrame(detailed_rows)
    if result.empty:
        empty_result = pd.DataFrame(columns=_pair_result_columns())
        empty_result.attrs["pairwise_meta"] = {
            "x_candidate_count": len(valid_x),
            "y_candidate_count": len(valid_y),
            "total_pairs": total_pairs,
            "risk_level": estimate_pairwise_risk(total_pairs),
            "output_dir": str(output_root) if output_root else "",
            "full_result_csv": str(writer.csv_path) if writer and writer.csv_path.exists() else "",
            "full_result_parquet": str(writer.parquet_path) if writer and writer.parquet_path.exists() else "",
            "error_log_csv": str(writer.error_path) if writer and writer.error_path.exists() else "",
            "removed_columns_csv": str(writer.removed_path) if writer and writer.removed_path.exists() else "",
            "matrix_only": matrix_only,
            **matrix_meta,
        }
        return empty_result

    if writer and writer.adjusted_lookup_path and writer.adjusted_lookup_path.exists():
        adjusted_df = pd.read_csv(writer.adjusted_lookup_path)
        result = result.merge(
            adjusted_df[["x_col", "y_col", "adjusted_p_value"]],
            left_on=["x_feature", "y_target"],
            right_on=["x_col", "y_col"],
            how="left",
        )
        result["p_value_adj"] = result["adjusted_p_value"].combine_first(result["p_value_adj"])
        result = result.drop(columns=[col for col in ["x_col", "y_col", "adjusted_p_value"] if col in result.columns])

    if pairwise_cfg.get("apply_fdr_correction", True) and not result.empty:
        result["p_value_adj"] = _benjamini_hochberg(result["p_value"])

    result = result.sort_values(["final_score", "effect_score", "model_score"], ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.attrs["pairwise_meta"] = {
        "x_candidate_count": len(valid_x),
        "y_candidate_count": len(valid_y),
        "total_pairs": total_pairs,
        "risk_level": estimate_pairwise_risk(total_pairs),
        "output_dir": str(output_root) if output_root else "",
        "full_result_csv": str(writer.csv_path) if writer and writer.csv_path.exists() else "",
        "full_result_parquet": str(writer.parquet_path) if writer and writer.parquet_path.exists() else "",
        "error_log_csv": str(writer.error_path) if writer and writer.error_path.exists() else "",
        "removed_columns_csv": str(writer.removed_path) if writer and writer.removed_path.exists() else "",
        "matrix_only": matrix_only,
        **matrix_meta,
    }
    return result[_pair_result_columns()]


def analyze_pairwise(
    df: pd.DataFrame,
    x_cols: Iterable[str],
    y_cols: Iterable[str],
    key_cols: str | Iterable[str] | None = None,
    exclude_cols: str | Iterable[str] | None = None,
    config: dict | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Analyze every selected X-Y pair and return ranked candidate pairs."""
    config = config or DEFAULT_CONFIG
    key_cols = set(_as_list(key_cols))
    exclude_cols = set(_as_list(exclude_cols))
    x_cols = [col for col in x_cols if col in df.columns]
    y_cols = [col for col in y_cols if col in df.columns]

    valid_x_candidates = []
    for x_col in x_cols:
        if x_col in key_cols or x_col in exclude_cols:
            continue
        if x_col.startswith("_") or "source_file" in x_col.lower() or _is_datetime_like(df[x_col], x_col):
            continue
        feature_type = classify_feature_type(df[x_col], config, x_col)
        if _exclusion_reason(df[x_col], feature_type, config):
            continue
        valid_x_candidates.append({"feature": x_col, "feature_type": feature_type})

    valid_y_cols = []
    for y_col in y_cols:
        if y_col in key_cols or y_col in exclude_cols:
            continue
        if y_col.startswith("_") or "source_file" in y_col.lower() or _is_datetime_like(df[y_col], y_col):
            continue
        y_feature_type = classify_feature_type(df[y_col], config, y_col)
        if _exclusion_reason(df[y_col], y_feature_type, config):
            continue
        valid_y_cols.append(y_col)

    pairwise_cfg = config.get("pairwise_analysis", DEFAULT_CONFIG["pairwise_analysis"])
    total_pairs = sum(1 for candidate in valid_x_candidates for y_col in valid_y_cols if candidate["feature"] != y_col)

    screen_rows = []
    pair_index = 0
    for y_col in valid_y_cols:
        y_type = classify_y_type(df[y_col])
        x_candidates = [candidate for candidate in valid_x_candidates if candidate["feature"] != y_col]

        for candidate in x_candidates:
            x_col = candidate["feature"]
            pair_index += 1
            _emit_progress(progress_callback, pair_index, max(total_pairs, 1), x_col, y_col, "빠른 선별")

            pair_stats = _compute_pair_stats(
                df=df,
                feature=x_col,
                y_col=y_col,
                feature_type=candidate["feature_type"],
                y_type=y_type,
                config=config,
            )
            if not pair_stats["valid"]:
                continue

            stat_score = _statistical_score(pair_stats["p_value"])
            effect_score = _effect_score(pair_stats["effect_size"], pair_stats["effect_metric"])
            quality_score = _quality_score(
                pair_stats["missing_ratio"],
                pair_stats["sample_n"],
                pair_stats["min_group_n"],
                candidate["feature_type"],
                config,
            )
            screen_rows.append(
                {
                    "_screen_index": len(screen_rows),
                    "x_feature": x_col,
                    "y_target": y_col,
                    "x_type": candidate["feature_type"],
                    "y_type": y_type,
                    "pair_stats": pair_stats,
                    "screening_score": _screening_score(stat_score, effect_score, quality_score, config),
                    "p_value": pair_stats["p_value"],
                    "effect_size": pair_stats["effect_size"],
                    "missing_ratio_x": round(float(df[x_col].isna().mean()), 4),
                    "missing_ratio_y": round(float(df[y_col].isna().mean()), 4),
                }
            )

    screen_df = pd.DataFrame(screen_rows)
    if screen_df.empty:
        return pd.DataFrame(columns=_pair_result_columns())

    screen_df["p_value_adj"] = (
        _benjamini_hochberg(screen_df["p_value"]) if pairwise_cfg.get("apply_fdr_correction", True) else screen_df["p_value"]
    )

    fast_screening = bool(pairwise_cfg.get("fast_screening", True))
    if fast_screening:
        top_k = max(1, int(pairwise_cfg.get("screening_top_k_per_y", 8)))
        min_score = float(pairwise_cfg.get("screening_min_score", 25))
        max_total = max(top_k, int(pairwise_cfg.get("screening_max_pairs_total", 120)))
        selected = (
            screen_df.sort_values(["y_target", "screening_score", "effect_size"], ascending=[True, False, False])
            .groupby("y_target", as_index=False, group_keys=False)
            .head(top_k)
        )
        strong = screen_df[screen_df["screening_score"] >= min_score]
        selected = (
            pd.concat([selected, strong], ignore_index=True)
            .drop_duplicates("_screen_index")
            .sort_values(["screening_score", "effect_size"], ascending=False)
            .head(max_total)
            .reset_index(drop=True)
        )
        score_config = _pairwise_fast_config(config, pairwise_cfg)
    else:
        selected = screen_df.sort_values(["screening_score", "effect_size"], ascending=False).reset_index(drop=True)
        score_config = config

    rows = []
    score_total = max(total_pairs + len(selected), 1)
    score_done = 0
    for y_col, selected_y in selected.groupby("y_target", sort=False):
        y_type = str(selected_y["y_type"].iloc[0])
        selected_candidates = [
            {"feature": str(row["x_feature"]), "feature_type": str(row["x_type"])}
            for _, row in selected_y.iterrows()
        ]
        model_scores = (
            _random_forest_importance(df, selected_candidates, y_col, y_type, score_config)
            if selected_candidates
            else {}
        )

        for _, selected_row in selected_y.iterrows():
            x_col = str(selected_row["x_feature"])
            score_done += 1
            _emit_progress(
                progress_callback,
                total_pairs + score_done,
                score_total,
                x_col,
                y_col,
                "정밀 계산",
            )
            scored = _score_pair_from_stats(
                df=df,
                feature=x_col,
                y_col=y_col,
                feature_type=str(selected_row["x_type"]),
                y_type=y_type,
                config=score_config,
                pair_stats=selected_row["pair_stats"],
                model_score=float(model_scores.get(x_col, 0.0)),
            )
            scored.update(
                {
                    "x_feature": x_col,
                    "y_target": y_col,
                    "x_type": str(selected_row["x_type"]),
                    "y_type": y_type,
                    "analysis_type": selected_row["pair_stats"]["effect_metric"],
                    "missing_ratio_x": float(selected_row["missing_ratio_x"]),
                    "missing_ratio_y": float(selected_row["missing_ratio_y"]),
                    "p_value_adj": float(selected_row["p_value_adj"]),
                    "r2_score": np.nan,
                    "screening_method": "",
                    "screening_score": float(selected_row["screening_score"]),
                    "screening_corr": np.nan,
                    "screening_r2": np.nan,
                    "used_matrix_screening": False,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            scored["easy_interpretation"] = build_easy_interpretation(
                x_feature=x_col,
                y_target=y_col,
                row=scored,
                pair_mode=True,
            )
            rows.append(scored)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=_pair_result_columns())

    result = result.sort_values(["final_score", "effect_score", "model_score"], ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result[_pair_result_columns()]


def build_easy_interpretation(x_feature: str, y_target: str, row: dict | pd.Series, pair_mode: bool = False) -> str:
    """Build a plain-language interpretation that frames results as candidates."""
    prefix = (
        f"{x_feature} → {y_target} 조합은 전체 X-Y 비교에서 우선 확인할 유의 후보로 탐지되었습니다."
        if pair_mode
        else f"{x_feature}은 현재 선택한 Y인 {y_target}와 함께 움직이는 패턴이 관찰되어 우선 확인할 후보로 선정되었습니다."
    )
    direction = str(row.get("direction", ""))
    caution = str(row.get("caution_text", ""))
    return (
        f"{prefix} {direction} 점수는 후보 우선순위를 정하기 위한 참고 신호이며, "
        f"확정 판단이 아니라 추가 확인이 필요한 유의 후보를 제시한 것입니다. {caution}"
    )


def _result_columns() -> list[str]:
    """Return the standard output column order."""
    return [
        "rank",
        "feature",
        "feature_type",
        "judgement",
        "final_score",
        "stat_score",
        "effect_score",
        "model_score",
        "stability_score",
        "quality_score",
        "p_value",
        "effect_size",
        "sample_n",
        "min_group_n",
        "missing_ratio",
        "direction",
        "metric_text",
        "caution_text",
    ]


def _pair_result_columns() -> list[str]:
    """Return the standard pairwise output column order."""
    return [
        "rank",
        "x_feature",
        "y_target",
        "x_type",
        "y_type",
        "analysis_type",
        "judgement",
        "final_score",
        "stat_score",
        "effect_score",
        "model_score",
        "stability_score",
        "quality_score",
        "p_value",
        "p_value_adj",
        "effect_size",
        "r2_score",
        "sample_n",
        "min_group_n",
        "missing_ratio_x",
        "missing_ratio_y",
        "direction",
        "metric_text",
        "caution_text",
        "easy_interpretation",
        "screening_method",
        "screening_score",
        "screening_corr",
        "screening_r2",
        "used_matrix_screening",
        "created_at",
    ]


def _score_pair(
    df: pd.DataFrame,
    feature: str,
    y_col: str,
    feature_type: str,
    y_type: str,
    config: dict,
    model_score: float = 0.0,
) -> dict | None:
    """Score one X-Y pair using the same logic as the single-Y analysis."""
    pair_stats = _compute_pair_stats(df, feature, y_col, feature_type, y_type, config)
    if not pair_stats["valid"]:
        return None

    return _score_pair_from_stats(
        df=df,
        feature=feature,
        y_col=y_col,
        feature_type=feature_type,
        y_type=y_type,
        config=config,
        pair_stats=pair_stats,
        model_score=model_score,
    )


def _score_pair_from_stats(
    df: pd.DataFrame,
    feature: str,
    y_col: str,
    feature_type: str,
    y_type: str,
    config: dict,
    pair_stats: dict,
    model_score: float = 0.0,
) -> dict:
    """Score one X-Y pair from already computed statistical results."""
    stat_score = _statistical_score(pair_stats["p_value"])
    effect_score = _effect_score(pair_stats["effect_size"], pair_stats["effect_metric"])
    stability_score = _bootstrap_stability(
        df,
        feature,
        y_col,
        feature_type,
        y_type,
        pair_stats["effect_size"],
        config,
    )
    quality_score = _quality_score(
        pair_stats["missing_ratio"],
        pair_stats["sample_n"],
        pair_stats["min_group_n"],
        feature_type,
        config,
    )

    weights = config.get("scoring", {}).get("weights", DEFAULT_CONFIG["scoring"]["weights"])
    final_score = (
        weights.get("statistical", 0.28) * stat_score
        + weights.get("effect", 0.26) * effect_score
        + weights.get("model", 0.18) * model_score
        + weights.get("stability", 0.18) * stability_score
        + weights.get("quality", 0.10) * quality_score
    )

    return {
        "judgement": _judgement(final_score, config),
        "final_score": round(float(final_score), 2),
        "stat_score": round(float(stat_score), 2),
        "effect_score": round(float(effect_score), 2),
        "model_score": round(float(model_score), 2),
        "stability_score": round(float(stability_score), 2),
        "quality_score": round(float(quality_score), 2),
        "p_value": pair_stats["p_value"],
        "effect_size": round(float(pair_stats["effect_size"]), 5),
        "sample_n": int(pair_stats["sample_n"]),
        "min_group_n": int(pair_stats["min_group_n"]),
        "missing_ratio": round(float(pair_stats["missing_ratio"]), 4),
        "direction": pair_stats["direction"],
        "metric_text": pair_stats["metric_text"],
        "caution_text": _caution_text(pair_stats, config),
    }


def _emit_progress(progress_callback, done: int, total: int, x_col: str, y_col: str, phase: str) -> None:
    """Report pairwise progress while keeping compatibility with older callbacks."""
    if not progress_callback:
        return
    try:
        progress_callback(done, total, x_col, y_col, phase)
    except TypeError:
        progress_callback(done, total, x_col, y_col)


def _screening_score(stat_score: float, effect_score: float, quality_score: float, config: dict) -> float:
    """Calculate a lightweight score used only for pairwise pre-screening."""
    weights = config.get("scoring", {}).get("weights", DEFAULT_CONFIG["scoring"]["weights"])
    stat_w = float(weights.get("statistical", 0.28))
    effect_w = float(weights.get("effect", 0.26))
    quality_w = float(weights.get("quality", 0.10))
    total_w = max(stat_w + effect_w + quality_w, 1e-9)
    return float(np.clip((stat_w * stat_score + effect_w * effect_score + quality_w * quality_score) / total_w, 0, 100))


def _pairwise_fast_config(config: dict, pairwise_cfg: dict) -> dict:
    """Use lighter model/bootstrap settings for pairwise fine scoring."""
    fast_config = dict(config)
    fast_config["model"] = dict(config.get("model", {}))
    fast_config["stability"] = dict(config.get("stability", {}))
    fast_config["model"]["n_estimators"] = int(
        pairwise_cfg.get("fast_model_n_estimators", fast_config["model"].get("n_estimators", 80))
    )
    fast_config["stability"]["bootstrap_iterations"] = int(
        pairwise_cfg.get("fast_bootstrap_iterations", fast_config["stability"].get("bootstrap_iterations", 10))
    )
    return fast_config


def _chunk_list(items: list, chunk_size: int) -> list[list]:
    """Split a list into fixed-size chunks."""
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _screening_result_row(
    x_col: str,
    y_col: str,
    x_type: str,
    y_type: str,
    pair_stats: dict,
    stat_score: float,
    effect_score: float,
    quality_score: float,
    screening_score: float,
    missing_ratio_x: float,
    missing_ratio_y: float,
    created_at: str,
) -> dict:
    """Build one lightweight result row for full pair screening output."""
    return {
        "x_col": x_col,
        "y_col": y_col,
        "x_type": x_type,
        "y_type": y_type,
        "analysis_type": pair_stats["effect_metric"],
        "final_score": round(float(screening_score), 2),
        "stat_score": round(float(stat_score), 2),
        "effect_score": round(float(effect_score), 2),
        "model_score": 0.0,
        "stability_score": 0.0,
        "quality_score": round(float(quality_score), 2),
        "p_value": pair_stats["p_value"],
        "adjusted_p_value": np.nan,
        "effect_size": round(float(pair_stats["effect_size"]), 5),
        "r2_score": np.nan,
        "n_samples": int(pair_stats["sample_n"]),
        "sample_n": int(pair_stats["sample_n"]),
        "min_group_n": int(pair_stats["min_group_n"]),
        "missing_rate_x": round(float(missing_ratio_x), 4),
        "missing_rate_y": round(float(missing_ratio_y), 4),
        "missing_ratio_x": round(float(missing_ratio_x), 4),
        "missing_ratio_y": round(float(missing_ratio_y), 4),
        "direction": pair_stats["direction"],
        "metric_text": pair_stats["metric_text"],
        "caution_text": "",
        "created_at": created_at,
        "pair_stats": pair_stats,
    }


def _update_shortlist(shortlist: list[dict], screening_row: dict, pairwise_cfg: dict) -> list[dict]:
    """Keep only the strongest rows per Y target for detailed analysis."""
    shortlist.append(screening_row)
    top_k = max(1, int(pairwise_cfg.get("screening_top_k_per_y", 8)))
    min_score = float(pairwise_cfg.get("screening_min_score", 25))
    shortlist = sorted(shortlist, key=lambda row: (row["final_score"], row["effect_size"]), reverse=True)
    keep = shortlist[:top_k]
    strong_keep = [row for row in shortlist[top_k:] if row["final_score"] >= min_score]
    max_extra = max(0, int(pairwise_cfg.get("screening_max_pairs_total", 120)) - len(keep))
    keep.extend(strong_keep[:max_extra])
    return keep


class _ResultBatchWriter:
    """Append large screening outputs to disk without keeping them all in memory."""

    def __init__(self, output_dir: Path, pairwise_cfg: dict) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "pairwise_full_result.csv"
        self.parquet_path = self.output_dir / "pairwise_full_result.parquet"
        self.error_path = self.output_dir / "error_log.csv"
        self.removed_path = self.output_dir / "removed_columns.csv"
        self.adjusted_lookup_path = self.output_dir / "adjusted_p_lookup.csv"
        self.save_csv = bool(pairwise_cfg.get("save_csv", True))
        self.save_parquet = bool(pairwise_cfg.get("save_parquet", True))
        self._parquet_writer = None

    def append_results(self, batch_df: pd.DataFrame) -> None:
        """Append one screening batch to CSV and optional Parquet."""
        if batch_df.empty:
            return
        write_df = batch_df.drop(columns=["pair_stats"], errors="ignore")
        if self.save_csv:
            write_df.to_csv(self.csv_path, mode="a", header=not self.csv_path.exists(), index=False)
        if self.save_parquet:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pa.Table.from_pandas(write_df, preserve_index=False)
                if self._parquet_writer is None:
                    self._parquet_writer = pq.ParquetWriter(self.parquet_path, table.schema)
                self._parquet_writer.write_table(table)
            except Exception:
                self.save_parquet = False

    def append_errors(self, batch_df: pd.DataFrame) -> None:
        """Append pair-level error logs."""
        if batch_df.empty:
            return
        batch_df.to_csv(self.error_path, mode="a", header=not self.error_path.exists(), index=False)

    def write_removed_columns(self, removed_df: pd.DataFrame) -> None:
        """Persist filtered-out column names and reasons."""
        if removed_df.empty:
            return
        removed_df.to_csv(self.removed_path, index=False)

    def write_adjusted_p_values(self, adjusted: np.ndarray) -> None:
        """Write adjusted p-values back out as a side lookup file."""
        if not self.csv_path.exists():
            return
        full_df = pd.read_csv(self.csv_path)
        full_df["adjusted_p_value"] = adjusted
        full_df.to_csv(self.csv_path, index=False)
        if self.save_parquet:
            try:
                full_df.to_parquet(self.parquet_path, index=False)
            except Exception:
                pass
        full_df[["x_col", "y_col", "adjusted_p_value"]].to_csv(self.adjusted_lookup_path, index=False)

    def finalize(self) -> None:
        """Close any open writers."""
        if self._parquet_writer is not None:
            self._parquet_writer.close()


def _benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    values = pd.to_numeric(p_values, errors="coerce").fillna(1.0).to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return np.array([])

    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        running_min = min(running_min, candidate)
        adjusted[order[i]] = min(running_min, 1.0)
    return adjusted


def _is_datetime_like(series: pd.Series, column_name: str) -> bool:
    """Check whether a column should be treated as date/time-like."""
    lower_name = column_name.lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    if "date" not in lower_name and "time" not in lower_name:
        return False
    clean = series.dropna().astype(str).head(100)
    if clean.empty:
        return False
    parsed = pd.to_datetime(clean, errors="coerce")
    return bool(parsed.notna().mean() >= 0.85)


def _exclusion_reason(series: pd.Series, feature_type: str, config: dict) -> str:
    """Explain why a feature should not be analyzed directly."""
    analysis_cfg = config.get("analysis", {})
    missing_ratio = float(series.isna().mean()) if len(series) else 1.0
    unique_count = int(series.dropna().nunique())

    if missing_ratio > analysis_cfg.get("max_missing_ratio", 0.45):
        return "High missing ratio"
    if unique_count <= 1:
        return "Constant feature"
    if feature_type in {"datetime", "id_like"}:
        return f"Excluded {feature_type} feature"
    if feature_type == "categorical" and unique_count > analysis_cfg.get("max_categories", 60):
        return "Too many categories"
    return ""


def _clean_xy(df: pd.DataFrame, feature: str, y_col: str) -> pd.DataFrame:
    """Keep only the feature and Y columns and remove missing rows."""
    return df[[feature, y_col]].dropna().copy()


def _compute_pair_stats(
    df: pd.DataFrame,
    feature: str,
    y_col: str,
    feature_type: str,
    y_type: str,
    config: dict,
) -> dict:
    """Run the appropriate statistical test for one X/Y pair."""
    pair = _clean_xy(df, feature, y_col)
    sample_n = len(pair)
    missing_ratio = float(df[[feature, y_col]].isna().any(axis=1).mean()) if len(df) else 1.0
    min_sample_n = config.get("analysis", {}).get("min_sample_n", 30)
    if sample_n < min_sample_n:
        return _invalid_stats(sample_n, missing_ratio)

    if feature_type == "numeric" and y_type == "numeric":
        return _numeric_numeric(pair, feature, y_col, missing_ratio)
    if feature_type == "categorical" and y_type == "numeric":
        return _categorical_numeric(pair, feature, y_col, missing_ratio, config)
    if feature_type == "numeric" and y_type == "binary":
        return _numeric_binary(pair, feature, y_col, missing_ratio)
    if feature_type == "categorical" and y_type == "binary":
        return _categorical_categorical(pair, feature, y_col, missing_ratio, binary_y=True, config=config)
    if feature_type == "categorical" and y_type == "categorical":
        return _categorical_categorical(pair, feature, y_col, missing_ratio, binary_y=False, config=config)
    if feature_type == "numeric" and y_type == "categorical":
        return _numeric_categorical(pair, feature, y_col, missing_ratio, config)
    return _invalid_stats(sample_n, missing_ratio)


def _invalid_stats(sample_n: int, missing_ratio: float) -> dict:
    """Return a standard invalid-result object."""
    return {
        "valid": False,
        "p_value": np.nan,
        "effect_size": 0.0,
        "effect_metric": "none",
        "sample_n": int(sample_n),
        "min_group_n": 0,
        "missing_ratio": float(missing_ratio),
        "direction": "",
        "metric_text": "",
    }


def _format_category_value(value) -> str:
    """Format category values for non-statistical readers."""
    if pd.isna(value):
        return "결측"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{float(value):g}"

    text = str(value)
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
        return f"{numeric:g}"
    except Exception:
        return text


def _feature_value_label(feature: str, value) -> str:
    """Build a practical label such as Slot=1 위치 or Recipe=RCP_A 조건."""
    value_text = _format_category_value(value)
    feature_lower = feature.lower()
    if feature_lower == "slot":
        return f"Slot={value_text} 위치"
    if "position" in feature_lower or "zone" in feature_lower:
        return f"{feature}={value_text} 위치"
    if value_text.startswith(("(", "[")) and "," in value_text:
        return f"{feature} {value_text} 구간"
    return f"{feature}={value_text} 조건"


def _target_value_label(y_col: str, value) -> str:
    """Build a readable label for a Y category or binary target value."""
    return f"{y_col}={_format_category_value(value)}"


def _numeric_numeric(pair: pd.DataFrame, feature: str, y_col: str, missing_ratio: float) -> dict:
    """Analyze numeric X versus numeric Y with Spearman correlation."""
    rho, p_value = stats.spearmanr(pair[feature], pair[y_col])
    rho = _safe_float(rho, 0.0)
    p_value = _safe_float(p_value, 1.0)
    direction = f"{feature} 값이 커질수록 {y_col} 값이 {'증가' if rho >= 0 else '감소'}하는 경향이 관찰되었습니다."
    return {
        "valid": True,
        "p_value": p_value,
        "effect_size": abs(rho),
        "effect_metric": "spearman",
        "sample_n": len(pair),
        "min_group_n": len(pair),
        "missing_ratio": missing_ratio,
        "direction": direction,
        "metric_text": f"Spearman rho={rho:.3f}",
    }


def _categorical_numeric(
    pair: pd.DataFrame,
    feature: str,
    y_col: str,
    missing_ratio: float,
    config: dict,
) -> dict:
    """Analyze categorical X versus numeric Y with Kruskal-Wallis."""
    min_group = config.get("analysis", {}).get("min_group_n", 15)
    grouped = [group[y_col].to_numpy() for _, group in pair.groupby(feature) if len(group) >= min_group]
    group_sizes = pair.groupby(feature).size()
    if len(grouped) < 2:
        return _invalid_stats(len(pair), missing_ratio)

    h_stat, p_value = stats.kruskal(*grouped)
    used_n = int(sum(len(group) for group in grouped))
    group_count = len(grouped)
    eta_sq = max(0.0, (h_stat - group_count + 1) / max(used_n - group_count, 1))
    medians = pair.groupby(feature)[y_col].median().sort_values(ascending=False)
    high_group = medians.index[0]
    low_group = medians.index[-1]
    high_label = _feature_value_label(feature, high_group)
    low_label = _feature_value_label(feature, low_group)

    return {
        "valid": True,
        "p_value": _safe_float(p_value, 1.0),
        "effect_size": float(eta_sq),
        "effect_metric": "eta_squared",
        "sample_n": used_n,
        "min_group_n": int(group_sizes.min()),
        "missing_ratio": missing_ratio,
        "direction": f"{high_label}에서 {y_col} 중앙값이 상대적으로 높고, {low_label}에서 낮게 나타났습니다.",
        "metric_text": f"Kruskal H={h_stat:.2f}, eta squared={eta_sq:.3f}",
    }


def _numeric_binary(pair: pd.DataFrame, feature: str, y_col: str, missing_ratio: float) -> dict:
    """Analyze numeric X versus binary Y with Mann-Whitney U."""
    y_binary, positive_label = _binary_codes(pair[y_col])
    x_positive = pair.loc[y_binary == 1, feature]
    x_negative = pair.loc[y_binary == 0, feature]
    if len(x_positive) < 2 or len(x_negative) < 2:
        return _invalid_stats(len(pair), missing_ratio)

    u_stat, p_value = stats.mannwhitneyu(x_positive, x_negative, alternative="two-sided")
    auc = u_stat / (len(x_positive) * len(x_negative))
    effect = abs(auc - 0.5) * 2
    pos_mean = x_positive.mean()
    neg_mean = x_negative.mean()
    direction = (
        f"{_target_value_label(y_col, positive_label)} 조건에서 {feature} 평균이 "
        f"{'상대적으로 높게' if pos_mean >= neg_mean else '상대적으로 낮게'} 나타났습니다."
    )

    return {
        "valid": True,
        "p_value": _safe_float(p_value, 1.0),
        "effect_size": float(effect),
        "effect_metric": "auc_effect",
        "sample_n": len(pair),
        "min_group_n": int(min(len(x_positive), len(x_negative))),
        "missing_ratio": missing_ratio,
        "direction": direction,
        "metric_text": f"Mann-Whitney U={u_stat:.1f}, AUC-like={auc:.3f}",
    }


def _categorical_categorical(
    pair: pd.DataFrame,
    feature: str,
    y_col: str,
    missing_ratio: float,
    binary_y: bool,
    config: dict,
) -> dict:
    """Analyze categorical X versus binary/categorical Y with Chi-square."""
    min_group = config.get("analysis", {}).get("min_group_n", 15)
    counts = pair.groupby(feature).size()
    keep_groups = counts[counts >= min_group].index
    filtered = pair[pair[feature].isin(keep_groups)].copy()
    table = pd.crosstab(filtered[feature].astype(str), filtered[y_col].astype(str))
    if min(table.shape) < 2 or table.to_numpy().sum() == 0:
        return _invalid_stats(len(pair), missing_ratio)

    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / max(n * (min(table.shape) - 1), 1))
    direction = _categorical_direction(filtered, feature, y_col, binary_y)
    return {
        "valid": True,
        "p_value": _safe_float(p_value, 1.0),
        "effect_size": float(cramers_v),
        "effect_metric": "cramers_v",
        "sample_n": int(n),
        "min_group_n": int(counts.min()),
        "missing_ratio": missing_ratio,
        "direction": direction,
        "metric_text": f"Chi-square={chi2:.2f}, Cramer's V={cramers_v:.3f}",
    }


def _numeric_categorical(
    pair: pd.DataFrame,
    feature: str,
    y_col: str,
    missing_ratio: float,
    config: dict,
) -> dict:
    """Analyze numeric X versus categorical Y with Kruskal-Wallis."""
    min_group = config.get("analysis", {}).get("min_group_n", 15)
    grouped = [group[feature].to_numpy() for _, group in pair.groupby(y_col) if len(group) >= min_group]
    group_sizes = pair.groupby(y_col).size()
    if len(grouped) < 2:
        return _invalid_stats(len(pair), missing_ratio)

    h_stat, p_value = stats.kruskal(*grouped)
    used_n = int(sum(len(group) for group in grouped))
    group_count = len(grouped)
    eta_sq = max(0.0, (h_stat - group_count + 1) / max(used_n - group_count, 1))
    medians = pair.groupby(y_col)[feature].median().sort_values(ascending=False)
    direction = (
        f"{_target_value_label(y_col, medians.index[0])} 조건에서 {feature} 중앙값이 상대적으로 높고, "
        f"{_target_value_label(y_col, medians.index[-1])} 조건에서 낮게 나타났습니다."
    )
    return {
        "valid": True,
        "p_value": _safe_float(p_value, 1.0),
        "effect_size": float(eta_sq),
        "effect_metric": "eta_squared",
        "sample_n": used_n,
        "min_group_n": int(group_sizes.min()),
        "missing_ratio": missing_ratio,
        "direction": direction,
        "metric_text": f"Kruskal H={h_stat:.2f}, eta squared={eta_sq:.3f}",
    }


def _binary_codes(series: pd.Series) -> tuple[pd.Series, object]:
    """Convert a two-value series into 0/1 codes and return the positive label."""
    unique_values = sorted(series.dropna().unique())
    positive_label = unique_values[-1]
    return (series == positive_label).astype(int), positive_label


def _categorical_direction(pair: pd.DataFrame, feature: str, y_col: str, binary_y: bool) -> str:
    """Create a readable direction sentence for categorical tests."""
    if binary_y:
        y_binary, positive_label = _binary_codes(pair[y_col])
        temp = pair.assign(_y_binary=y_binary)
        rates = temp.groupby(feature)["_y_binary"].mean().sort_values(ascending=False)
        return (
            f"{_feature_value_label(feature, rates.index[0])}에서 "
            f"{_target_value_label(y_col, positive_label)} 비율이 상대적으로 높게 나타났습니다."
        )

    table = pd.crosstab(pair[feature].astype(str), pair[y_col].astype(str), normalize="index")
    max_position = np.unravel_index(np.argmax(table.to_numpy()), table.shape)
    return (
        f"{_feature_value_label(feature, table.index[max_position[0]])}에서 "
        f"{_target_value_label(y_col, table.columns[max_position[1]])} 비중이 상대적으로 높게 나타났습니다."
    )


def _statistical_score(p_value: float) -> float:
    """Convert a p-value into a 0-100 statistical score."""
    if pd.isna(p_value):
        return 0.0
    p_value = max(float(p_value), 1e-300)
    return float(np.clip((-np.log10(p_value) / 4.0) * 100, 0, 100))


def _effect_score(effect_size: float, effect_metric: str) -> float:
    """Convert effect size into a practical 0-100 score."""
    scale_map = {
        "spearman": 0.45,
        "eta_squared": 0.12,
        "auc_effect": 0.45,
        "cramers_v": 0.35,
    }
    scale = scale_map.get(effect_metric, 0.30)
    return float(np.clip((abs(effect_size) / scale) * 100, 0, 100))


def _quality_score(
    missing_ratio: float,
    sample_n: int,
    min_group_n: int,
    feature_type: str,
    config: dict,
) -> float:
    """Score data usability from missingness, sample size, and group balance."""
    analysis_cfg = config.get("analysis", {})
    full_n = analysis_cfg.get("sample_score_full_n", 300)
    min_group_target = analysis_cfg.get("min_group_n", 15) * 2

    missing_score = max(0.0, (1 - missing_ratio) * 100)
    sample_score = min(sample_n / max(full_n, 1), 1.0) * 100
    if feature_type == "categorical":
        group_score = min(min_group_n / max(min_group_target, 1), 1.0) * 100
    else:
        group_score = 100.0
    return float(np.clip(0.45 * missing_score + 0.35 * sample_score + 0.20 * group_score, 0, 100))


def _random_forest_importance(
    df: pd.DataFrame,
    candidates: list[dict],
    y_col: str,
    y_type: str,
    config: dict,
) -> dict[str, float]:
    """Train a RandomForest model and return normalized feature importances."""
    features = [candidate["feature"] for candidate in candidates]
    feature_types = {candidate["feature"]: candidate["feature_type"] for candidate in candidates}
    model_frame = df[features + [y_col]].dropna(subset=[y_col]).copy()
    if len(model_frame) < config.get("analysis", {}).get("min_sample_n", 30):
        return {feature: 0.0 for feature in features}

    x_frame = model_frame[features].copy()
    numeric_features = [feature for feature in features if feature_types[feature] == "numeric"]
    categorical_features = [feature for feature in features if feature_types[feature] == "categorical"]

    for feature in numeric_features:
        x_frame[feature] = pd.to_numeric(x_frame[feature], errors="coerce")
        x_frame[feature] = x_frame[feature].fillna(x_frame[feature].median())
    for feature in categorical_features:
        x_frame[feature] = x_frame[feature].astype("string").fillna("<Missing>")

    encoded = pd.get_dummies(x_frame, columns=categorical_features, dummy_na=False)
    if encoded.empty:
        return {feature: 0.0 for feature in features}

    y = model_frame[y_col]
    model_cfg = config.get("model", {})
    if y_type == "numeric":
        model = RandomForestRegressor(
            n_estimators=model_cfg.get("n_estimators", 250),
            random_state=model_cfg.get("random_state", 42),
            min_samples_leaf=5,
            n_jobs=-1,
        )
    else:
        if y.nunique(dropna=True) < 2:
            return {feature: 0.0 for feature in features}
        model = RandomForestClassifier(
            n_estimators=model_cfg.get("n_estimators", 250),
            random_state=model_cfg.get("random_state", 42),
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        y = y.astype(str)

    try:
        model.fit(encoded, y)
    except Exception:
        return {feature: 0.0 for feature in features}

    raw_importance = dict(zip(encoded.columns, model.feature_importances_))
    grouped = {feature: 0.0 for feature in features}
    for column, importance in raw_importance.items():
        matched_feature = _original_feature_from_encoded(column, features)
        grouped[matched_feature] = grouped.get(matched_feature, 0.0) + float(importance)

    max_importance = max(grouped.values()) if grouped else 0.0
    if max_importance <= 0:
        return {feature: 0.0 for feature in features}
    return {feature: value / max_importance * 100 for feature, value in grouped.items()}


def _original_feature_from_encoded(encoded_column: str, features: list[str]) -> str:
    """Map a one-hot encoded column back to its original feature name."""
    if encoded_column in features:
        return encoded_column
    matches = [feature for feature in features if encoded_column.startswith(f"{feature}_")]
    if not matches:
        return encoded_column
    return max(matches, key=len)


def _bootstrap_stability(
    df: pd.DataFrame,
    feature: str,
    y_col: str,
    feature_type: str,
    y_type: str,
    original_effect: float,
    config: dict,
) -> float:
    """Estimate whether a finding repeats under bootstrap resampling."""
    pair = _clean_xy(df, feature, y_col)
    if len(pair) < config.get("analysis", {}).get("min_sample_n", 30):
        return 0.0

    stability_cfg = config.get("stability", {})
    iterations = int(stability_cfg.get("bootstrap_iterations", 30))
    sample_fraction = float(stability_cfg.get("sample_fraction", 0.80))
    rng = np.random.default_rng(stability_cfg.get("random_state", 42))

    significant = []
    effect_consistent = []
    sample_size = max(20, int(len(pair) * sample_fraction))
    for _ in range(iterations):
        sampled_index = rng.choice(pair.index.to_numpy(), size=sample_size, replace=True)
        sampled = pair.loc[sampled_index].reset_index(drop=True)
        stats_result = _compute_pair_stats(sampled, feature, y_col, feature_type, y_type, config)
        if not stats_result["valid"]:
            continue
        significant.append(stats_result["p_value"] < 0.05)
        if original_effect > 0:
            effect_consistent.append(stats_result["effect_size"] >= original_effect * 0.50)
        else:
            effect_consistent.append(False)

    if not significant:
        return 0.0
    sig_score = np.mean(significant) * 100
    consistency_score = np.mean(effect_consistent) * 100 if effect_consistent else 0
    return float(np.clip(0.70 * sig_score + 0.30 * consistency_score, 0, 100))


def _judgement(final_score: float, config: dict) -> str:
    """Convert final score into a Korean judgement label."""
    thresholds = config.get("thresholds", DEFAULT_CONFIG["thresholds"])
    if final_score >= thresholds.get("very_significant", 90):
        return "매우 유의"
    if final_score >= thresholds.get("significant", 75):
        return "유의"
    if final_score >= thresholds.get("needs_check", 60):
        return "확인 필요"
    if final_score >= thresholds.get("weak", 40):
        return "약함"
    return "낮음"


def _caution_text(pair_stats: dict, config: dict) -> str:
    """Build a short caution message for each result row."""
    messages = []
    if pair_stats["p_value"] >= 0.05:
        messages.append("p-value 기준으로는 약함")
    if pair_stats["missing_ratio"] > 0.20:
        messages.append("결측률 확인 필요")
    if pair_stats["min_group_n"] and pair_stats["min_group_n"] < config.get("analysis", {}).get("min_group_n", 15):
        messages.append("일부 조건의 표본 수가 적음")
    if not messages:
        messages.append("확정 판단이 아니라 우선 확인 후보로 공정 맥락 확인 필요")
    elif not any("확정 판단" in message for message in messages):
        messages.append("확정 판단이 아니라 우선 확인 후보")
    return "; ".join(messages)

"""Fast candidate screening engines.

Screening is only for narrowing candidates. Final ranking is computed in
``detailed_analyzer`` and ``scoring``.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse

from core.models import AnalysisPlan, ScreeningCandidate

ProgressCallback = Callable[[dict], None]


def _missing_rate(series: pd.Series) -> float:
    return float(series.isna().mean()) if len(series) else 1.0


def _rank_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(method="average", na_option="keep")


def _prepare_numeric_matrix(frame: pd.DataFrame, dtype: str, missing_strategy: str) -> tuple[np.ndarray, dict[str, dict]]:
    """Convert numeric columns to a centered matrix and metadata."""
    meta: dict[str, dict] = {}
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if missing_strategy == "drop_rows":
        numeric = numeric.dropna(axis=0, how="any")
    if missing_strategy == "median_impute":
        fill_values = numeric.median(numeric_only=True)
    else:
        fill_values = numeric.mean(numeric_only=True)
    filled = numeric.fillna(fill_values).fillna(0.0)
    matrix = filled.to_numpy(dtype=np.float32 if dtype == "float32" else np.float64, copy=True)
    for column in frame.columns:
        col = pd.to_numeric(frame[column], errors="coerce")
        meta[str(column)] = {
            "missing_rate": _missing_rate(col),
            "variance": float(col.var(skipna=True)) if col.notna().sum() else 0.0,
        }
    return matrix, meta


def _corr_matrix(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation by matrix multiplication."""
    x = x_matrix - np.nanmean(x_matrix, axis=0, keepdims=True)
    y = y_matrix - np.nanmean(y_matrix, axis=0, keepdims=True)
    numerator = x.T @ y
    x_norm = np.sqrt(np.sum(x * x, axis=0))
    y_norm = np.sqrt(np.sum(y * y, axis=0))
    denominator = np.outer(x_norm, y_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denominator
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def numeric_numeric_screening(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    plan: AnalysisPlan,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Screen numeric-numeric relationships with chunked matrix correlation."""
    if not x_cols or not y_cols:
        return pd.DataFrame()
    cfg = config.get("numeric_matrix_screening", {})
    dtype = str(cfg.get("dtype", "float32"))
    missing_strategy = str(cfg.get("missing_strategy", "mean_impute"))
    method = str(plan.numeric_method or cfg.get("method", "pearson")).lower()
    x_chunk_size = max(1, int(plan.x_chunk_size or cfg.get("x_chunk_size", 5000)))
    y_chunk_size = max(1, int(plan.y_chunk_size or cfg.get("y_chunk_size", 50)))
    top_n = max(1, int(plan.top_n_per_y))
    started = time.time()
    total_pairs = len(x_cols) * len(y_cols)
    processed = 0
    top_by_y: dict[str, list[dict]] = defaultdict(list)

    x_chunks = [x_cols[i : i + x_chunk_size] for i in range(0, len(x_cols), x_chunk_size)]
    y_chunks = [y_cols[i : i + y_chunk_size] for i in range(0, len(y_cols), y_chunk_size)]
    for yi, y_chunk in enumerate(y_chunks, start=1):
        y_frame = df[y_chunk]
        if method == "spearman":
            y_frame = _rank_columns(y_frame)
        y_matrix, y_meta = _prepare_numeric_matrix(y_frame, dtype, missing_strategy)
        for xi, x_chunk in enumerate(x_chunks, start=1):
            x_frame = df[x_chunk]
            if method == "spearman":
                x_frame = _rank_columns(x_frame)
            x_matrix, x_meta = _prepare_numeric_matrix(x_frame, dtype, missing_strategy)
            corr = _corr_matrix(x_matrix, y_matrix)
            abs_corr = np.abs(corr)
            r2 = corr * corr
            for y_idx, y_col in enumerate(y_chunk):
                scores = abs_corr[:, y_idx] * 100.0
                pick_count = min(top_n, len(scores))
                if pick_count <= 0:
                    continue
                indices = np.argpartition(scores, -pick_count)[-pick_count:]
                for x_idx in indices:
                    x_col = x_chunk[int(x_idx)]
                    if x_col == y_col:
                        continue
                    row = {
                        "x_col": x_col,
                        "y_col": y_col,
                        "pair_type": "numeric_numeric",
                        "screening_method": method,
                        "screening_score": float(scores[x_idx]),
                        "corr": float(corr[x_idx, y_idx]),
                        "abs_corr": float(abs_corr[x_idx, y_idx]),
                        "r2": float(r2[x_idx, y_idx]),
                        "n_samples": int(len(df)),
                        "x_missing_rate": float(x_meta[x_col]["missing_rate"]),
                        "y_missing_rate": float(y_meta[y_col]["missing_rate"]),
                    }
                    top_by_y[y_col].append(row)
                top_by_y[y_col] = sorted(top_by_y[y_col], key=lambda item: item["screening_score"], reverse=True)[:top_n]
            processed += len(x_chunk) * len(y_chunk)
            if progress_callback:
                elapsed = max(time.time() - started, 1e-9)
                progress_callback(
                    {
                        "phase": "numeric matrix screening",
                        "processed_pairs": processed,
                        "total_pairs": total_pairs,
                        "pairs_per_sec": processed / elapsed,
                        "eta_seconds": max(total_pairs - processed, 0) / max(processed / elapsed, 1e-9),
                        "x_chunk_index": xi,
                        "x_chunk_total": len(x_chunks),
                        "y_chunk_index": yi,
                        "y_chunk_total": len(y_chunks),
                    }
                )

    rows = []
    for y_col, candidates in top_by_y.items():
        for rank, row in enumerate(sorted(candidates, key=lambda item: item["screening_score"], reverse=True), start=1):
            rows.append({**row, "screening_rank_within_y": rank})
    return pd.DataFrame(rows)


def normalize_category(
    series: pd.Series,
    min_samples_per_level: int,
    rare_level_min_ratio: float,
    rare_label: str,
    missing_label: str,
) -> pd.Series:
    """Normalize rare and missing levels for one-hot screening."""
    text = series.astype("object").where(series.notna(), missing_label).astype(str)
    counts = text.value_counts(dropna=False)
    min_count = max(min_samples_per_level, int(math.ceil(len(text) * rare_level_min_ratio)))
    rare_levels = set(counts[counts < min_count].index)
    return text.map(lambda value: rare_label if value in rare_levels and value != missing_label else value)


def one_hot(series: pd.Series) -> tuple[sparse.csr_matrix, list[str]]:
    """Build a sparse one-hot matrix."""
    categories = pd.Categorical(series)
    codes = categories.codes
    valid = codes >= 0
    rows = np.arange(len(series))[valid]
    cols = codes[valid]
    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(categories.categories)))
    return matrix, [str(value) for value in categories.categories]


def _eta_from_group_means(counts: np.ndarray, means: np.ndarray, overall: float, total_ss: float) -> float:
    if total_ss <= 0 or counts.sum() <= 1:
        return 0.0
    between = float(np.sum(counts * np.square(means - overall)))
    return max(0.0, min(1.0, between / total_ss))


def categorical_numeric_screening(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    plan: AnalysisPlan,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Screen categorical X against numeric Y using sparse one-hot group means."""
    if not x_cols or not y_cols:
        return pd.DataFrame()
    cfg = config.get("categorical_matrix_screening", {})
    min_n = int(cfg.get("min_samples_per_level", 5))
    rare_ratio = float(cfg.get("rare_level_min_ratio", 0.01))
    rare_label = str(cfg.get("rare_level_label", "__OTHER__"))
    missing_label = str(cfg.get("missing_label", "__MISSING__"))
    rows = []
    total_pairs = len(x_cols) * len(y_cols)
    processed = 0
    for x_col in x_cols:
        x_norm = normalize_category(df[x_col], min_n, rare_ratio, rare_label, missing_label)
        g, levels = one_hot(x_norm)
        counts = np.asarray(g.sum(axis=0)).ravel()
        for y_col in y_cols:
            y = pd.to_numeric(df[y_col], errors="coerce")
            valid = y.notna().to_numpy()
            if valid.sum() < 3:
                continue
            gv = g[valid]
            yv = y[valid].to_numpy(dtype=np.float64)
            counts_v = np.asarray(gv.sum(axis=0)).ravel()
            nonzero = counts_v > 0
            sums = np.asarray(gv.T @ yv).ravel()
            means = np.divide(sums, counts_v, out=np.zeros_like(sums, dtype=np.float64), where=counts_v > 0)
            overall = float(np.mean(yv))
            total_ss = float(np.sum((yv - overall) ** 2))
            eta = _eta_from_group_means(counts_v[nonzero], means[nonzero], overall, total_ss)
            rows.append(
                ScreeningCandidate(
                    x_col=x_col,
                    y_col=y_col,
                    pair_type="categorical_numeric",
                    screening_method="onehot_eta_squared",
                    screening_score=float(eta * 100.0),
                    n_samples=int(valid.sum()),
                    x_missing_rate=_missing_rate(df[x_col]),
                    y_missing_rate=_missing_rate(df[y_col]),
                    metrics={
                        "eta_squared": float(eta),
                        "n_levels": int(len(levels)),
                        "max_group_mean_diff": float(np.nanmax(means[nonzero]) - np.nanmin(means[nonzero])) if nonzero.any() else 0.0,
                    },
                ).to_dict()
            )
            processed += 1
            if progress_callback and processed % 100 == 0:
                progress_callback({"phase": "categorical-numeric screening", "processed_pairs": processed, "total_pairs": total_pairs})
    return pd.DataFrame(rows)


def numeric_categorical_screening(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    plan: AnalysisPlan,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Screen numeric X against categorical Y with sparse one-hot means."""
    if not x_cols or not y_cols:
        return pd.DataFrame()
    cfg = config.get("categorical_matrix_screening", {})
    min_n = int(cfg.get("min_samples_per_level", 5))
    rare_ratio = float(cfg.get("rare_level_min_ratio", 0.01))
    rare_label = str(cfg.get("rare_level_label", "__OTHER__"))
    missing_label = str(cfg.get("missing_label", "__MISSING__"))
    rows = []
    for y_col in y_cols:
        y_norm = normalize_category(df[y_col], min_n, rare_ratio, rare_label, missing_label)
        g, levels = one_hot(y_norm)
        counts = np.asarray(g.sum(axis=0)).ravel()
        for start in range(0, len(x_cols), max(1, plan.x_chunk_size)):
            chunk = x_cols[start : start + max(1, plan.x_chunk_size)]
            x_frame = df[chunk].apply(pd.to_numeric, errors="coerce")
            missing = x_frame.isna().mean()
            means = x_frame.mean(numeric_only=True)
            filled = x_frame.fillna(means).fillna(0.0)
            x_matrix = filled.to_numpy(dtype=np.float32, copy=True)
            sums = np.asarray(g.T @ x_matrix)
            level_means = np.divide(sums, counts[:, None], out=np.zeros_like(sums, dtype=np.float64), where=counts[:, None] > 0)
            overall = np.mean(x_matrix, axis=0)
            total_ss = np.sum((x_matrix - overall) ** 2, axis=0)
            between = np.sum(counts[:, None] * (level_means - overall) ** 2, axis=0)
            eta_values = np.divide(between, total_ss, out=np.zeros_like(between, dtype=np.float64), where=total_ss > 0)
            eta_values = np.clip(np.nan_to_num(eta_values), 0.0, 1.0)
            for idx, x_col in enumerate(chunk):
                rows.append(
                    ScreeningCandidate(
                        x_col=x_col,
                        y_col=y_col,
                        pair_type="numeric_categorical",
                        screening_method="onehot_eta_squared",
                        screening_score=float(eta_values[idx] * 100.0),
                        n_samples=int(len(df)),
                        x_missing_rate=float(missing[x_col]),
                        y_missing_rate=_missing_rate(df[y_col]),
                        metrics={"eta_squared": float(eta_values[idx]), "n_levels": int(len(levels))},
                    ).to_dict()
                )
    return pd.DataFrame(rows)


def _cramers_v(table: np.ndarray) -> tuple[float, float]:
    from scipy.stats import chi2_contingency

    if table.size == 0 or table.sum() <= 0 or min(table.shape) < 2:
        return 0.0, 1.0
    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.sum()
    denom = n * max(min(table.shape) - 1, 1)
    return float(np.sqrt(chi2 / denom)) if denom else 0.0, float(p_value)


def categorical_categorical_screening(
    df: pd.DataFrame,
    x_cols: list[str],
    y_cols: list[str],
    plan: AnalysisPlan,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Screen low-cardinality categorical pairs using one-hot contingency matrices."""
    if not x_cols or not y_cols:
        return pd.DataFrame()
    cfg = config.get("categorical_matrix_screening", {})
    min_n = int(cfg.get("min_samples_per_level", 5))
    rare_ratio = float(cfg.get("rare_level_min_ratio", 0.01))
    rare_label = str(cfg.get("rare_level_label", "__OTHER__"))
    missing_label = str(cfg.get("missing_label", "__MISSING__"))
    y_encoded = {}
    for y_col in y_cols:
        y_norm = normalize_category(df[y_col], min_n, rare_ratio, rare_label, missing_label)
        y_encoded[y_col] = one_hot(y_norm)
    rows = []
    for x_col in x_cols:
        x_norm = normalize_category(df[x_col], min_n, rare_ratio, rare_label, missing_label)
        gx, x_levels = one_hot(x_norm)
        for y_col, (gy, y_levels) in y_encoded.items():
            if x_col == y_col:
                continue
            table = (gx.T @ gy).toarray()
            cramer, p_value = _cramers_v(table)
            rows.append(
                ScreeningCandidate(
                    x_col=x_col,
                    y_col=y_col,
                    pair_type="categorical_categorical",
                    screening_method="onehot_cramers_v",
                    screening_score=float(cramer * 100.0),
                    n_samples=int(table.sum()),
                    x_missing_rate=_missing_rate(df[x_col]),
                    y_missing_rate=_missing_rate(df[y_col]),
                    metrics={"cramer_v": float(cramer), "chi2_p_value": float(p_value), "n_x_levels": len(x_levels), "n_y_levels": len(y_levels)},
                ).to_dict()
            )
    return pd.DataFrame(rows)


def combine_and_limit_candidates(candidates: list[pd.DataFrame], top_n_per_y: int) -> pd.DataFrame:
    """Combine screening candidates and keep top rows within each Y."""
    frames = [frame for frame in candidates if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna(subset=["screening_score"])
    combined = combined.sort_values(["y_col", "screening_score"], ascending=[True, False])
    combined = combined.groupby("y_col", as_index=False, group_keys=False).head(top_n_per_y)
    combined["screening_rank_within_y"] = combined.groupby("y_col")["screening_score"].rank(method="first", ascending=False).astype(int)
    return combined.sort_values("screening_score", ascending=False).reset_index(drop=True)


def screen_candidates(
    df: pd.DataFrame,
    x_numeric: list[str],
    y_numeric: list[str],
    x_categorical: list[str],
    y_categorical: list[str],
    plan: AnalysisPlan,
    config: dict,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Run all enabled screening paths and return column-level candidates."""
    frames = []
    if config.get("numeric_matrix_screening", {}).get("enabled", True):
        frames.append(numeric_numeric_screening(df, x_numeric, y_numeric, plan, config, progress_callback))
    if plan.include_categorical and config.get("categorical_matrix_screening", {}).get("enabled", True):
        frames.append(categorical_numeric_screening(df, x_categorical, y_numeric, plan, config, progress_callback))
        frames.append(numeric_categorical_screening(df, x_numeric, y_categorical, plan, config, progress_callback))
        frames.append(categorical_categorical_screening(df, x_categorical, y_categorical, plan, config, progress_callback))
    return combine_and_limit_candidates(frames, int(plan.top_n_per_y))


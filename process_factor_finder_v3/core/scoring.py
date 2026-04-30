"""Final score calculation and validation.

This is the single source of truth for final_score. UI code must only display
scores calculated here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.models import PAIR_TYPES, SCREENING_METHODS_BY_PAIR_TYPE


def statistical_score(p_value: float) -> float:
    """Convert p-value to a bounded 0-100 score."""
    if p_value is None or not np.isfinite(p_value) or p_value < 0 or p_value > 1:
        return 0.0
    if p_value <= 1e-15:
        return 100.0
    return float(np.clip(-math.log10(max(p_value, 1e-300)) / 15.0 * 100.0, 0.0, 100.0))


def effect_score(pair_type: str, effect_size: float) -> float:
    """Normalize effect metrics by pair type."""
    if effect_size is None or not np.isfinite(effect_size):
        return 0.0
    reference = {
        "numeric_numeric": 0.60,
        "categorical_numeric": 0.20,
        "numeric_categorical": 0.20,
        "categorical_categorical": 0.50,
    }.get(pair_type, 0.50)
    return float(np.clip(abs(effect_size) / reference * 100.0, 0.0, 100.0))


def quality_score(n_samples: int, x_missing_rate: float, y_missing_rate: float, min_n: int = 30) -> float:
    """Score data quality from missing rate and sample size."""
    if n_samples <= 0:
        return 0.0
    missing_penalty = max(0.0, 1.0 - (float(x_missing_rate) + float(y_missing_rate)) / 2.0)
    sample_factor = min(1.0, n_samples / max(min_n * 10.0, 1.0))
    min_factor = min(1.0, n_samples / max(min_n, 1))
    return float(np.clip(100.0 * missing_penalty * (0.65 * sample_factor + 0.35 * min_factor), 0.0, 100.0))


def stability_score(n_samples: int, effect_size: float) -> float:
    """Lightweight stability proxy from sample size and effect magnitude."""
    if n_samples <= 0 or effect_size is None or not np.isfinite(effect_size):
        return 0.0
    sample_part = min(1.0, math.sqrt(n_samples) / math.sqrt(300.0))
    effect_part = min(1.0, abs(effect_size) / 0.30)
    return float(np.clip(100.0 * (0.7 * sample_part + 0.3 * effect_part), 0.0, 100.0))


def calculate_final_score(
    pair_type: str,
    p_value: float,
    effect_size_value: float,
    model_score: float,
    n_samples: int,
    x_missing_rate: float,
    y_missing_rate: float,
    config: dict,
) -> dict[str, float]:
    """Calculate component scores and final_score."""
    weights = config.get("scoring", {}).get(
        "weights",
        {"statistical": 0.28, "effect": 0.26, "model": 0.18, "stability": 0.18, "quality": 0.10},
    )
    stat = statistical_score(p_value)
    effect = effect_score(pair_type, effect_size_value)
    model = float(np.clip(model_score if np.isfinite(model_score) else 0.0, 0.0, 100.0))
    stability = stability_score(n_samples, effect_size_value)
    quality = quality_score(n_samples, x_missing_rate, y_missing_rate)
    final = (
        float(weights.get("statistical", 0.28)) * stat
        + float(weights.get("effect", 0.26)) * effect
        + float(weights.get("model", 0.18)) * model
        + float(weights.get("stability", 0.18)) * stability
        + float(weights.get("quality", 0.10)) * quality
    )
    return {
        "stat_score": round(stat, 4),
        "effect_score": round(effect, 4),
        "model_score": round(model, 4),
        "stability_score": round(stability, 4),
        "quality_score": round(quality, 4),
        "final_score": round(float(np.clip(final, 0.0, 100.0)), 4),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Apply Benjamini-Hochberg FDR correction."""
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values >= 0) & (values <= 1)
    if not valid.any():
        return pd.Series(adjusted, index=p_values.index)
    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranks = np.arange(1, len(valid_values) + 1)
    ranked = valid_values[order] * len(valid_values) / ranks
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    valid_adjusted = np.empty_like(valid_values)
    valid_adjusted[order] = ranked
    adjusted[valid] = valid_adjusted
    return pd.Series(adjusted, index=p_values.index)


def validate_result_scores(row: dict[str, Any], config: dict) -> tuple[bool, str]:
    """Validate final result values before exposing them in UI."""
    score_min = float(config.get("scoring", {}).get("score_min", 0))
    score_max = float(config.get("scoring", {}).get("score_max", 100))
    pair_type = str(row.get("pair_type", ""))
    method = str(row.get("screening_method", ""))
    if pair_type not in PAIR_TYPES:
        return False, f"허용되지 않는 pair_type: {pair_type}"
    if method and method not in SCREENING_METHODS_BY_PAIR_TYPE.get(pair_type, set()):
        return False, f"pair_type과 screening_method 불일치: {pair_type}/{method}"
    final_score = row.get("final_score")
    if final_score is None or not np.isfinite(float(final_score)) or not score_min <= float(final_score) <= score_max:
        return False, "final_score 범위 오류"
    for p_col in ("p_value", "adjusted_p_value"):
        value = row.get(p_col)
        if value is not None and not pd.isna(value):
            if not np.isfinite(float(value)) or not 0 <= float(value) <= 1:
                return False, f"{p_col} 범위 오류"
    if int(row.get("sample_n", 0)) <= 0:
        return False, "sample_n 오류"
    for metric in ("effect_size", "model_score", "r2_score", "eta_squared", "cramer_v"):
        value = row.get(metric)
        if value is not None and not pd.isna(value) and not np.isfinite(float(value)):
            return False, f"{metric} finite 오류"
    return True, ""


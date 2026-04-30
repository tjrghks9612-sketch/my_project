from __future__ import annotations

from typing import Optional

import numpy as np


def status_from_confidence(confidence: float, failed: bool = False) -> str:
    if failed:
        return "Fail"
    if confidence >= 80:
        return "OK"
    if confidence >= 60:
        return "Check"
    return "Review Needed"


def distance_confidence(
    values: np.ndarray,
    valid_count: int,
    total_count: int,
    min_valid_count: int,
    min_valid_ratio: float,
) -> float:
    if total_count <= 0 or valid_count < min_valid_count or valid_count / total_count < min_valid_ratio:
        return 0.0
    valid_ratio = min(1.0, valid_count / total_count)
    mean = float(np.mean(values)) if len(values) else 0.0
    std = float(np.std(values)) if len(values) else mean
    cv = std / max(abs(mean), 1e-6)
    stability = max(0.0, 1.0 - cv * 4.0)
    spread_penalty = max(0.0, 1.0 - min(1.0, std / max(abs(mean), 1.0)))
    score = 100.0 * (0.55 * valid_ratio + 0.30 * stability + 0.15 * spread_penalty)
    return float(max(0.0, min(100.0, score)))


def fit_confidence(
    valid_count: int,
    total_count: int,
    fit_error: Optional[float],
    r2: Optional[float],
    min_valid_count: int,
    fit_error_threshold: float,
) -> float:
    if total_count <= 0 or valid_count < min_valid_count:
        return 0.0
    valid_ratio = min(1.0, valid_count / total_count)
    if fit_error is None:
        error_score = 0.0
    else:
        error_score = max(0.0, 1.0 - fit_error / max(fit_error_threshold * 3.0, 1.0))
    r2_score = max(0.0, min(1.0, r2 if r2 is not None else 0.0))
    r2_score = max(r2_score, error_score * 0.85)
    score = 100.0 * (0.45 * valid_ratio + 0.25 * r2_score + 0.30 * error_score)
    return float(max(0.0, min(100.0, score)))

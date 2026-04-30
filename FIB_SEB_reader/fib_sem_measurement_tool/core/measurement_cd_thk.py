from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.confidence import distance_confidence, status_from_confidence
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import DistanceResult, MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _sample_indices(length: int, count: int, margin_ratio: float = 0.12) -> np.ndarray:
    count = max(3, min(int(count), max(3, length)))
    start = int(round(length * margin_ratio))
    end = int(round(length * (1.0 - margin_ratio))) - 1
    if end <= start:
        start, end = 0, length - 1
    return np.unique(np.linspace(start, end, count).round().astype(int))


def _smooth(profile: np.ndarray) -> np.ndarray:
    if profile.size < 5:
        return profile.astype(np.float32, copy=False)
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(profile.astype(np.float32, copy=False), kernel, mode="same")


def _anchor_pair(mean_gradient: np.ndarray) -> Tuple[float, float] | None:
    if mean_gradient.size < 8:
        return None
    profile = _smooth(mean_gradient)
    length = profile.size
    center = length // 2
    margin = max(2, int(length * 0.02))
    left_region = profile[margin:max(center, margin + 1)]
    right_region = profile[min(center, length - margin - 1):length - margin]
    if left_region.size == 0 or right_region.size == 0:
        return None
    left = margin + int(np.argmax(left_region)) + 0.5
    right_offset = min(center, length - margin - 1)
    right = right_offset + int(np.argmax(right_region)) + 0.5
    if right - left < 4:
        return None
    return float(left), float(right)


def _local_peak(profile: np.ndarray, anchor: float, radius: int) -> float | None:
    if profile.size == 0:
        return None
    center = int(round(anchor - 0.5))
    start = max(0, center - radius)
    end = min(profile.size, center + radius + 1)
    if end <= start:
        return None
    local = _smooth(profile[start:end])
    return float(start + int(np.argmax(local)) + 0.5)


def _filter_distances(values: List[float], pairs: List[Tuple[float, float, float]]) -> Tuple[np.ndarray, List[Tuple[float, float, float]]]:
    array = np.asarray(values, dtype=np.float32)
    if array.size < 4:
        return array, pairs
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad <= 1e-6:
        return array, pairs
    limit = max(3.0, 3.5 * 1.4826 * mad)
    keep = np.abs(array - median) <= limit
    return array[keep], [pair for pair, ok in zip(pairs, keep) if bool(ok)]


def _finish_distance_result(
    result: DistanceResult,
    values: List[float],
    pairs: List[Tuple[float, float, float]],
    settings: MeasurementSettings,
) -> DistanceResult:
    filtered, filtered_pairs = _filter_distances(values, pairs)
    result.values_px = [float(v) for v in filtered]
    result.boundary_pairs = filtered_pairs
    result.valid_count = int(filtered.size)
    result.total_count = max(result.total_count, len(values))

    min_count = max(3, min(int(settings.advanced.minimum_valid_line_count), max(3, result.total_count)))
    if result.valid_count < min_count:
        result.confidence = 0.0
        result.status = "Fail"
        result.warning_message = f"유효 라인 부족 ({result.valid_count}/{result.total_count})"
        return result

    result.mean_px = float(np.mean(filtered))
    result.max_px = float(np.max(filtered))
    result.min_px = float(np.min(filtered))
    result.median_px = float(np.median(filtered))
    result.std_px = float(np.std(filtered))
    if settings.distance_method == "max":
        result.selected_px = result.max_px
    elif settings.distance_method == "min":
        result.selected_px = result.min_px
    else:
        result.selected_px = result.mean_px
    result.selected_method = settings.distance_method
    result.confidence = distance_confidence(filtered, result.valid_count, result.total_count, min_count, 0.25)
    result.status = status_from_confidence(result.confidence)
    if result.status != "OK":
        result.warning_message = "경계 변화량이 약하거나 라인별 편차가 있습니다"
    return result


def measure_horizontal_cd(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    image_size = (gray.shape[1], gray.shape[0])
    clean_roi = normalize_roi(roi, image_size)
    result = DistanceResult(orientation="horizontal", selected_method=settings.distance_method)
    if clean_roi is None:
        result.warning_message = "ROI가 없거나 너무 작습니다"
        return result

    x1, y1, x2, y2 = clean_roi
    crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32, copy=False)
    gradient = np.abs(np.diff(crop, axis=1))
    y_indices = _sample_indices(crop.shape[0], settings.advanced.scan_line_count)
    result.total_count = int(y_indices.size)
    anchors = _anchor_pair(np.mean(gradient[y_indices, :], axis=0))
    if anchors is None:
        result.warning_message = "좌우 경계 변화량을 찾지 못했습니다"
        return result

    radius = max(3, int(gradient.shape[1] * 0.06))
    values: List[float] = []
    pairs: List[Tuple[float, float, float]] = []
    for local_y in y_indices:
        row_gradient = gradient[local_y, :]
        left = _local_peak(row_gradient, anchors[0], radius)
        right = _local_peak(row_gradient, anchors[1], radius)
        if left is None or right is None or right - left <= 3:
            continue
        left_x = float(x1 + left)
        right_x = float(x1 + right)
        values.append(right_x - left_x)
        pairs.append((float(y1 + local_y), left_x, right_x))

    return _finish_distance_result(result, values, pairs, settings)


def measure_vertical_thk(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> DistanceResult:
    image_size = (gray.shape[1], gray.shape[0])
    clean_roi = normalize_roi(roi, image_size)
    result = DistanceResult(orientation="vertical", selected_method=settings.distance_method)
    if clean_roi is None:
        result.warning_message = "ROI가 없거나 너무 작습니다"
        return result

    x1, y1, x2, y2 = clean_roi
    crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32, copy=False)
    gradient = np.abs(np.diff(crop, axis=0))
    x_indices = _sample_indices(crop.shape[1], settings.advanced.scan_line_count)
    result.total_count = int(x_indices.size)
    anchors = _anchor_pair(np.mean(gradient[:, x_indices], axis=1))
    if anchors is None:
        result.warning_message = "상하 경계 변화량을 찾지 못했습니다"
        return result

    radius = max(3, int(gradient.shape[0] * 0.06))
    values: List[float] = []
    pairs: List[Tuple[float, float, float]] = []
    for local_x in x_indices:
        column_gradient = gradient[:, local_x]
        top = _local_peak(column_gradient, anchors[0], radius)
        bottom = _local_peak(column_gradient, anchors[1], radius)
        if top is None or bottom is None or bottom - top <= 3:
            continue
        top_y = float(y1 + top)
        bottom_y = float(y1 + bottom)
        values.append(bottom_y - top_y)
        pairs.append((float(x1 + local_x), top_y, bottom_y))

    return _finish_distance_result(result, values, pairs, settings)


def measure_distance_both(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    horizontal = measure_horizontal_cd(gray, roi, settings)
    vertical = measure_vertical_thk(gray, roi, settings)
    confidences = [item.confidence for item in (horizontal, vertical) if item.status != "Fail"]
    overall = float(np.mean(confidences)) if confidences else 0.0
    status = status_from_confidence(overall, failed=not confidences)
    warnings = "; ".join([item.warning_message for item in (horizontal, vertical) if item.warning_message])
    return MeasurementResult(
        measurement_type="distance_both",
        horizontal_cd=horizontal,
        vertical_thk=vertical,
        overall_confidence=overall,
        status=status,
        warning_message=warnings,
    )

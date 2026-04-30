from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fib_sem_measurement_tool.core.confidence import fit_confidence, status_from_confidence
from fib_sem_measurement_tool.core.measurement_cd_thk import _anchor_pair, _local_peak, _sample_indices
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import MeasurementResult, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _fit_line(points: List[Tuple[float, float]], side: str, settings: MeasurementSettings) -> TaperSideResult:
    result = TaperSideResult(side=side)
    if len(points) < 3:
        result.warning_message = "유효 taper point 부족"
        return result

    pts = np.asarray(points, dtype=np.float64)
    xs = pts[:, 0]
    ys = pts[:, 1]
    slope, intercept = np.polyfit(ys, xs, 1)
    residuals = xs - (slope * ys + intercept)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    if mad > 1e-6 and len(points) >= 5:
        keep = np.abs(residuals - median) <= max(2.0, 3.5 * 1.4826 * mad)
        if int(np.count_nonzero(keep)) >= 3:
            xs = xs[keep]
            ys = ys[keep]
            pts = pts[keep]
            slope, intercept = np.polyfit(ys, xs, 1)
            residuals = xs - (slope * ys + intercept)

    fit_error = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.0
    ss_tot = float(np.sum((xs - np.mean(xs)) ** 2))
    ss_res = float(np.sum(residuals**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0

    angle_horizontal = abs(float(np.degrees(np.arctan2(1.0, slope))))
    if angle_horizontal > 90:
        angle_horizontal = 180 - angle_horizontal
    angle_vertical = abs(90.0 - angle_horizontal)

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    x_at_min = float(slope * y_min + intercept)
    x_at_max = float(slope * y_max + intercept)

    result.angle_horizontal = angle_horizontal
    result.angle_vertical = angle_vertical
    result.fit_r2 = float(max(0.0, min(1.0, r2)))
    result.fit_error = fit_error
    result.valid_point_count = len(points)
    result.inlier_count = int(len(pts))
    result.points = [(float(x), float(y)) for x, y in pts]
    result.fit_line = (x_at_min, y_min, x_at_max, y_max)
    result.confidence = fit_confidence(
        result.inlier_count,
        len(points),
        result.fit_error,
        result.fit_r2,
        max(3, int(settings.advanced.minimum_valid_line_count * 0.6)),
        max(3.0, settings.advanced.fit_error_threshold),
    )
    result.status = status_from_confidence(result.confidence)
    if result.status != "OK":
        result.warning_message = "taper 경계 변화량 확인 필요"
    return result


def measure_taper_side(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> TaperSideResult:
    image_size = (gray.shape[1], gray.shape[0])
    clean_roi = normalize_roi(roi, image_size)
    result = TaperSideResult(side=side)
    if clean_roi is None:
        result.warning_message = "ROI가 없거나 너무 작습니다"
        return result

    x1, y1, x2, y2 = clean_roi
    crop = gray[y1:y2 + 1, x1:x2 + 1].astype(np.float32, copy=False)
    gradient = np.abs(np.diff(crop, axis=1))
    y_indices = _sample_indices(crop.shape[0], settings.advanced.scan_line_count, margin_ratio=0.10)
    anchors = _anchor_pair(np.mean(gradient[y_indices, :], axis=0))
    if anchors is None:
        result.warning_message = "taper 경계 변화량을 찾지 못했습니다"
        return result

    anchor = anchors[0] if side == "left" else anchors[1]
    radius = max(3, int(gradient.shape[1] * 0.07))
    points: List[Tuple[float, float]] = []
    for local_y in y_indices:
        edge = _local_peak(gradient[local_y, :], anchor, radius)
        if edge is None:
            continue
        points.append((float(x1 + edge), float(y1 + local_y)))

    return _fit_line(points, side, settings)


def measure_single_taper(gray: np.ndarray, roi: Sequence[int], side: str, settings: MeasurementSettings) -> MeasurementResult:
    taper = measure_taper_side(gray, roi, side, settings)
    result = MeasurementResult(measurement_type="taper_single", overall_confidence=taper.confidence)
    result.status = status_from_confidence(taper.confidence, failed=taper.status == "Fail")
    result.warning_message = taper.warning_message
    if side == "right":
        result.right_taper = taper
    else:
        result.left_taper = taper
    return result


def measure_double_taper(gray: np.ndarray, roi: Sequence[int], settings: MeasurementSettings) -> MeasurementResult:
    left = measure_taper_side(gray, roi, "left", settings)
    right = measure_taper_side(gray, roi, "right", settings)
    valid = [item for item in (left, right) if item.status != "Fail"]
    overall = float(np.mean([item.confidence for item in valid])) if valid else 0.0
    result = MeasurementResult(
        measurement_type="taper_double",
        left_taper=left,
        right_taper=right,
        overall_confidence=overall,
        status=status_from_confidence(overall, failed=not valid),
        warning_message="; ".join([item.warning_message for item in (left, right) if item.warning_message]),
    )
    if left.angle_horizontal is not None and right.angle_horizontal is not None:
        result.avg_taper_angle = float(np.mean([left.angle_horizontal, right.angle_horizontal]))
        result.taper_angle_diff = float(abs(left.angle_horizontal - right.angle_horizontal))
    return result

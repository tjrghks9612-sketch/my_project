from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from fib_sem_measurement_tool.core.image_io import to_gray
from fib_sem_measurement_tool.core.measurement_cd_thk import measure_horizontal_cd, measure_vertical_thk
from fib_sem_measurement_tool.core.measurement_taper import measure_double_taper, measure_single_taper
from fib_sem_measurement_tool.core.roi_utils import normalize_roi
from fib_sem_measurement_tool.models.result import MeasurementResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _fail_result(measurement_type: str, message: str) -> MeasurementResult:
    return MeasurementResult(
        measurement_type=measurement_type,
        overall_confidence=0.0,
        status="Fail",
        warning_message=message,
    )


def calculate_confidence(result: MeasurementResult) -> float:
    scores = []
    for item in (result.horizontal_cd, result.vertical_thk, result.left_taper, result.right_taper):
        if item is not None and item.status != "Fail":
            scores.append(item.confidence)
    if not scores:
        result.overall_confidence = 0.0
        result.status = "Fail"
        return 0.0
    result.overall_confidence = float(np.mean(scores))
    if result.overall_confidence >= 80:
        result.status = "OK"
    elif result.overall_confidence >= 60:
        result.status = "Check"
    else:
        result.status = "Review Needed"
    return result.overall_confidence


def run_measurement(image: np.ndarray, settings: MeasurementSettings) -> MeasurementResult:
    gray = to_gray(image)
    roi = settings.roi
    if roi is None:
        return _fail_result(settings.measurement_type, "ROI가 지정되지 않았습니다")
    clean_roi = normalize_roi(roi, (gray.shape[1], gray.shape[0]))
    if clean_roi is None:
        return _fail_result(settings.measurement_type, "ROI가 없거나 너무 작습니다")

    try:
        if settings.measurement_type == "distance_horizontal":
            horizontal = measure_horizontal_cd(gray, clean_roi, settings)
            result = MeasurementResult(measurement_type=settings.measurement_type, horizontal_cd=horizontal)
        elif settings.measurement_type == "distance_vertical":
            vertical = measure_vertical_thk(gray, clean_roi, settings)
            result = MeasurementResult(measurement_type=settings.measurement_type, vertical_thk=vertical)
        elif settings.measurement_type == "distance_both":
            horizontal = measure_horizontal_cd(gray, clean_roi, settings)
            vertical = measure_vertical_thk(gray, clean_roi, settings)
            result = MeasurementResult(
                measurement_type=settings.measurement_type,
                horizontal_cd=horizontal,
                vertical_thk=vertical,
                warning_message="; ".join([item.warning_message for item in (horizontal, vertical) if item.warning_message]),
            )
        elif settings.measurement_type == "taper_double":
            result = measure_double_taper(gray, clean_roi, settings)
            result.measurement_type = settings.measurement_type
        elif settings.measurement_type == "taper_single":
            result = measure_single_taper(gray, clean_roi, settings.taper_side, settings)
            result.measurement_type = settings.measurement_type
        else:
            return _fail_result(settings.measurement_type, "지원하지 않는 측정 타입입니다")
        calculate_confidence(result)
        warnings = []
        for item in (result.horizontal_cd, result.vertical_thk, result.left_taper, result.right_taper):
            if item is not None and getattr(item, "warning_message", ""):
                warnings.append(item.warning_message)
        if warnings:
            result.warning_message = "; ".join(dict.fromkeys(warnings))
        return result
    except Exception as exc:
        return _fail_result(settings.measurement_type, f"측정 중 오류: {exc}")


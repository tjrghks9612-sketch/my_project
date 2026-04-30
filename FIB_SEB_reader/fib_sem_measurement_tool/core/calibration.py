from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from fib_sem_measurement_tool.core.image_io import to_gray
from fib_sem_measurement_tool.models.settings import CalibrationSettings


def detect_scale_bar(image: np.ndarray) -> Dict[str, object]:
    gray = to_gray(image)
    height, width = gray.shape[:2]
    search_regions = [
        ("bottom", int(height * 0.68), height),
        ("top", 0, int(height * 0.32)),
    ]

    best: Optional[Dict[str, object]] = None
    for region_name, y0, y1 in search_regions:
        crop = gray[y0:y1, :]
        if crop.size == 0:
            continue
        threshold_value = max(160, int(np.percentile(crop, 94)))
        _, binary = cv2.threshold(crop, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * 0.04 or h < 1 or h > max(18, width * 0.04):
                continue
            aspect = w / max(h, 1)
            if aspect < 5:
                continue
            area = cv2.contourArea(contour)
            fill = area / max(w * h, 1)
            score = w * min(aspect / 20.0, 1.0) * max(fill, 0.2)
            candidate = {
                "status": "detected",
                "region": region_name,
                "pixel_length": float(w),
                "bbox": (int(x), int(y0 + y), int(x + w), int(y0 + y + h)),
                "score": float(score),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return {
            "status": "not_found",
            "pixel_length": None,
            "bbox": None,
            "message": "스케일바 선 후보를 찾지 못했습니다",
        }
    return best


def apply_calibration(
    pixel_length: float,
    actual_length: float,
    unit: str,
    mode: str = "auto",
) -> CalibrationSettings:
    if pixel_length <= 0 or actual_length <= 0:
        return CalibrationSettings(status="failed", mode=mode, unit=unit)
    px_to_real = float(actual_length) / float(pixel_length)
    return CalibrationSettings(
        px_to_real=px_to_real,
        unit=unit,
        mode=mode,
        detected_scale_bar_px=float(pixel_length) if mode == "auto" else None,
        actual_scale_bar_length=float(actual_length),
        status="calibrated",
        manual_pixel_length=float(pixel_length) if mode == "manual" else None,
    )


def clone_calibration(calibration: CalibrationSettings) -> CalibrationSettings:
    return replace(calibration)


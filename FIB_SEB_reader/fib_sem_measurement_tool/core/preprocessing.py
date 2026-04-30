from __future__ import annotations

import cv2
import numpy as np

from fib_sem_measurement_tool.models.settings import MeasurementSettings


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def preprocess_image(gray: np.ndarray, settings: MeasurementSettings) -> np.ndarray:
    work = gray.copy()
    if work.ndim != 2:
        raise ValueError("preprocess_image expects a grayscale image")

    blur_kernel = _odd_kernel(settings.advanced.blur_kernel)
    if blur_kernel:
        work = cv2.GaussianBlur(work, (blur_kernel, blur_kernel), 0)

    median_kernel = _odd_kernel(settings.advanced.median_filter_size)
    if median_kernel:
        work = cv2.medianBlur(work, median_kernel)

    strength = float(settings.advanced.background_correction_strength)
    if strength > 0:
        kernel_size = max(15, int(min(work.shape[:2]) * 0.08))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        background = cv2.GaussianBlur(work, (kernel_size, kernel_size), 0)
        corrected = cv2.addWeighted(work, 1.0 + strength, background, -strength, 128 * strength)
        work = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return work


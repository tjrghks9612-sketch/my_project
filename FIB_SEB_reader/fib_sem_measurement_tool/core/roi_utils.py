from __future__ import annotations

from typing import Optional, Sequence, Tuple

Roi = Tuple[int, int, int, int]
RelativeRoi = Tuple[float, float, float, float]


def normalize_roi(roi: Sequence[float], image_size: Tuple[int, int]) -> Optional[Roi]:
    if not roi or len(roi) != 4:
        return None
    width, height = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1, x2 = sorted((max(0, x1), min(width - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(height - 1, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def convert_roi_absolute_to_relative(roi: Roi, image_size: Tuple[int, int]) -> RelativeRoi:
    width, height = image_size
    if width <= 0 or height <= 0:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = roi
    return x1 / width, y1 / height, x2 / width, y2 / height


def convert_roi_relative_to_absolute(relative_roi: RelativeRoi, image_size: Tuple[int, int]) -> Optional[Roi]:
    width, height = image_size
    x1, y1, x2, y2 = relative_roi
    return normalize_roi((x1 * width, y1 * height, x2 * width, y2 * height), image_size)


def apply_roi_to_image(source_roi: Roi, source_size: Tuple[int, int], target_size: Tuple[int, int], mode: str) -> Optional[Roi]:
    if mode == "absolute_copy":
        return normalize_roi(source_roi, target_size)
    if mode == "relative_copy" or mode == "auto_adjust":
        relative = convert_roi_absolute_to_relative(source_roi, source_size)
        return convert_roi_relative_to_absolute(relative, target_size)
    return normalize_roi(source_roi, target_size)


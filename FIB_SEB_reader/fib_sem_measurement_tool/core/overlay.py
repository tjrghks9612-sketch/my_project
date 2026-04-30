from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fib_sem_measurement_tool.models.result import DistanceResult, MeasurementResult, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings


Color = Tuple[int, int, int]

ROI_COLOR: Color = (0, 210, 255)
CD_COLOR: Color = (255, 155, 40)
THK_COLOR: Color = (70, 245, 95)
TAPER_COLOR: Color = (255, 235, 40)
POINT_COLOR: Color = (255, 255, 255)
FAIL_COLOR: Color = (50, 80, 255)


@lru_cache(maxsize=12)
def _ui_font(size: int):
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure_text(text: str, font) -> Tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _draw_text(image: np.ndarray, text: str, origin: Tuple[int, int], color: Color, font_size: int = 16) -> None:
    font = _ui_font(font_size)
    x, y = origin
    tw, th = _measure_text(text, font)
    x = max(0, min(int(x), image.shape[1] - 1))
    y = max(0, min(int(y), image.shape[0] - 1))
    x2 = min(image.shape[1], x + tw + 4)
    y2 = min(image.shape[0], y + th + 6)
    if x2 <= x or y2 <= y:
        return
    patch = image[y:y2, x:x2]
    pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    draw.text((0, 0), text, fill=rgb, font=font)
    image[y:y2, x:x2] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _draw_dashed_line(image: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: Color, thickness: int = 1, dash: int = 8) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for start in range(0, length, dash * 2):
        end = min(start + dash, length)
        t1 = start / length
        t2 = end / length
        a = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        b = (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2))
        cv2.line(image, a, b, color, thickness, cv2.LINE_AA)


def _draw_dashed_rect(image: np.ndarray, roi: Sequence[int], color: Color) -> None:
    x1, y1, x2, y2 = [int(v) for v in roi]
    _draw_dashed_line(image, (x1, y1), (x2, y1), color, 1)
    _draw_dashed_line(image, (x2, y1), (x2, y2), color, 1)
    _draw_dashed_line(image, (x2, y2), (x1, y2), color, 1)
    _draw_dashed_line(image, (x1, y2), (x1, y1), color, 1)
    _label(image, "ROI", (x1 + 8, y1 + 22), ROI_COLOR)


def _label(image: np.ndarray, text: str, origin: Tuple[int, int], color: Color, scale: float = 0.55) -> None:
    font_size = max(12, int(round(scale * 28)))
    font = _ui_font(font_size)
    x, y = origin
    tw, th = _measure_text(text, font)
    x = max(4, min(x, image.shape[1] - tw - 8))
    y = max(th + 10, min(y, image.shape[0] - 8))
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 6, y - th - 8), (x + tw + 7, y + 7), (8, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    _draw_text(image, text, (x, y - th), color, font_size)


def _format_value(px_value: Optional[float], settings: MeasurementSettings) -> str:
    if px_value is None:
        return "-"
    value = px_value * settings.calibration.px_to_real
    return f"{value:.3g} {settings.calibration.unit}"


def _draw_distance(image: np.ndarray, result: DistanceResult, settings: MeasurementSettings, color: Color, label_prefix: str) -> None:
    if not result.boundary_pairs:
        return
    if result.orientation == "horizontal":
        left_points = [(int(round(left)), int(round(y))) for y, left, right in result.boundary_pairs]
        right_points = [(int(round(right)), int(round(y))) for y, left, right in result.boundary_pairs]
        for point in left_points + right_points:
            cv2.circle(image, point, 2, POINT_COLOR, -1, cv2.LINE_AA)
        cv2.polylines(image, [np.asarray(left_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.polylines(image, [np.asarray(right_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        nearest = min(result.boundary_pairs, key=lambda item: abs((item[2] - item[1]) - (result.selected_px or 0)))
        y, left, right = nearest
        y = int(round(y))
        p1 = (int(round(left)), y)
        p2 = (int(round(right)), y)
        cv2.arrowedLine(image, p1, p2, color, 1, cv2.LINE_AA, tipLength=0.02)
        cv2.arrowedLine(image, p2, p1, color, 1, cv2.LINE_AA, tipLength=0.02)
        _label(image, f"{label_prefix} {_format_value(result.selected_px, settings)}", (int((left + right) / 2) - 45, y - 8), color)
    else:
        top_points = [(int(round(x)), int(round(top))) for x, top, bottom in result.boundary_pairs]
        bottom_points = [(int(round(x)), int(round(bottom))) for x, top, bottom in result.boundary_pairs]
        for point in top_points + bottom_points:
            cv2.circle(image, point, 2, POINT_COLOR, -1, cv2.LINE_AA)
        cv2.polylines(image, [np.asarray(top_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.polylines(image, [np.asarray(bottom_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        nearest = min(result.boundary_pairs, key=lambda item: abs((item[2] - item[1]) - (result.selected_px or 0)))
        x, top, bottom = nearest
        x = int(round(x))
        p1 = (x, int(round(top)))
        p2 = (x, int(round(bottom)))
        cv2.arrowedLine(image, p1, p2, color, 1, cv2.LINE_AA, tipLength=0.02)
        cv2.arrowedLine(image, p2, p1, color, 1, cv2.LINE_AA, tipLength=0.02)
        _label(image, f"{label_prefix} {_format_value(result.selected_px, settings)}", (x + 8, int((top + bottom) / 2)), color)


def _draw_taper(image: np.ndarray, taper: TaperSideResult, color: Color) -> None:
    for x, y in taper.points:
        cv2.circle(image, (int(round(x)), int(round(y))), 2, POINT_COLOR, -1, cv2.LINE_AA)
    if taper.fit_line:
        x1, y1, x2, y2 = taper.fit_line
        cv2.line(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )
        angle = taper.angle_horizontal if taper.angle_horizontal is not None else 0.0
        mid = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
        _label(image, f"{taper.side.upper()} {angle:.1f} deg", (mid[0] + 8, mid[1]), color, scale=0.48)


def _draw_summary(image: np.ndarray, result: MeasurementResult, settings: MeasurementSettings) -> None:
    status_color = THK_COLOR if result.status == "OK" else (0, 210, 255) if result.status == "Check" else FAIL_COLOR
    lines = [
        f"{result.status} {result.overall_confidence:.0f}%",
        f"{settings.measurement_type} / {settings.edge_reference}",
    ]
    if result.horizontal_cd and result.horizontal_cd.selected_px is not None:
        lines.append(f"CD {_format_value(result.horizontal_cd.selected_px, settings)}")
    if result.vertical_thk and result.vertical_thk.selected_px is not None:
        lines.append(f"THK {_format_value(result.vertical_thk.selected_px, settings)}")
    if result.avg_taper_angle is not None:
        lines.append(f"AVG {result.avg_taper_angle:.1f} deg")

    x = 16
    y = 28
    font_size = 16
    font = _ui_font(font_size)
    width = max(220, min(360, max(_measure_text(line, font)[0] for line in lines) + 28))
    height = 20 + len(lines) * 23
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (8, 18, 28), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    for idx, line in enumerate(lines):
        _draw_text(image, line, (x + 12, y + 12 + idx * 23), status_color if idx == 0 else (225, 235, 245), font_size)


def draw_overlay(
    image: np.ndarray,
    roi: Optional[Sequence[int]],
    result: Optional[MeasurementResult],
    settings: MeasurementSettings,
    show_overlay: bool = True,
    calibration_line: Optional[Tuple[int, int, int, int]] = None,
    scale_bar_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    canvas = image.copy()
    if not show_overlay:
        return canvas
    if roi is not None:
        _draw_dashed_rect(canvas, roi, ROI_COLOR)
    if scale_bar_bbox:
        x1, y1, x2, y2 = [int(v) for v in scale_bar_bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 90), 2, cv2.LINE_AA)
        _label(canvas, "scale bar", (x1, y1 - 8), (255, 255, 90), scale=0.48)
    if calibration_line:
        x1, y1, x2, y2 = [int(v) for v in calibration_line]
        cv2.line(canvas, (x1, y1), (x2, y2), (70, 220, 255), 2, cv2.LINE_AA)
        _label(canvas, "cal line", (x1, y1 - 8), (70, 220, 255), scale=0.48)
    if result is not None:
        if result.horizontal_cd:
            _draw_distance(canvas, result.horizontal_cd, settings, CD_COLOR, "CD")
        if result.vertical_thk:
            _draw_distance(canvas, result.vertical_thk, settings, THK_COLOR, "THK")
        if result.left_taper:
            _draw_taper(canvas, result.left_taper, TAPER_COLOR)
        if result.right_taper:
            _draw_taper(canvas, result.right_taper, TAPER_COLOR)
        _draw_summary(canvas, result, settings)
    return canvas

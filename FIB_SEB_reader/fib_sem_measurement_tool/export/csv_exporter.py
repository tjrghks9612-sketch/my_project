from __future__ import annotations

import csv
from typing import Dict, Iterable, List, Optional

from fib_sem_measurement_tool.models.image_item import ImageItem
from fib_sem_measurement_tool.models.result import DistanceResult, MeasurementResult, TaperSideResult
from fib_sem_measurement_tool.models.settings import MeasurementSettings, resolve_effective_settings


CSV_COLUMNS = [
    "file_name",
    "full_path",
    "group_id",
    "group_name",
    "measurement_type",
    "roi_apply_mode",
    "roi_source_image",
    "settings_source",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "edge_reference",
    "noise_level",
    "distance_method",
    "px_to_real",
    "unit",
    "calibration_mode",
    "calibration_status",
    "detected_scale_bar_px",
    "actual_scale_bar_length",
    "taper_side",
    "left_taper_angle_horizontal",
    "right_taper_angle_horizontal",
    "left_taper_angle_vertical",
    "right_taper_angle_vertical",
    "avg_taper_angle",
    "taper_angle_diff",
    "left_fit_r2",
    "right_fit_r2",
    "left_fit_error",
    "right_fit_error",
    "left_valid_point_count",
    "right_valid_point_count",
    "left_taper_status",
    "right_taper_status",
    "horizontal_cd_mean_px",
    "horizontal_cd_max_px",
    "horizontal_cd_min_px",
    "horizontal_cd_median_px",
    "horizontal_cd_std_px",
    "horizontal_cd_selected_px",
    "horizontal_cd_selected_method",
    "horizontal_cd_mean",
    "horizontal_cd_max",
    "horizontal_cd_min",
    "horizontal_cd_median",
    "horizontal_cd_std",
    "horizontal_cd_selected",
    "horizontal_cd_valid_count",
    "horizontal_cd_confidence",
    "horizontal_cd_status",
    "vertical_thk_mean_px",
    "vertical_thk_max_px",
    "vertical_thk_min_px",
    "vertical_thk_median_px",
    "vertical_thk_std_px",
    "vertical_thk_selected_px",
    "vertical_thk_selected_method",
    "vertical_thk_mean",
    "vertical_thk_max",
    "vertical_thk_min",
    "vertical_thk_median",
    "vertical_thk_std",
    "vertical_thk_selected",
    "vertical_thk_valid_count",
    "vertical_thk_confidence",
    "vertical_thk_status",
    "overall_confidence",
    "status",
    "warning_message",
    "measurement_source",
]


def _fmt(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def _distance_values(prefix: str, row: Dict[str, object], result: Optional[DistanceResult], settings: MeasurementSettings) -> None:
    fields = [
        "mean_px",
        "max_px",
        "min_px",
        "median_px",
        "std_px",
        "selected_px",
        "selected_method",
        "valid_count",
        "confidence",
        "status",
    ]
    if result is None:
        return
    for field in fields:
        row[f"{prefix}_{field}"] = getattr(result, field)
    scaled = result.scaled(settings.calibration.px_to_real)
    row[f"{prefix}_mean"] = scaled["mean"]
    row[f"{prefix}_max"] = scaled["max"]
    row[f"{prefix}_min"] = scaled["min"]
    row[f"{prefix}_median"] = scaled["median"]
    row[f"{prefix}_std"] = scaled["std"]
    row[f"{prefix}_selected"] = scaled["selected"]


def _taper_values(side: str, row: Dict[str, object], result: Optional[TaperSideResult]) -> None:
    if result is None:
        return
    row[f"{side}_taper_angle_horizontal"] = result.angle_horizontal
    row[f"{side}_taper_angle_vertical"] = result.angle_vertical
    row[f"{side}_fit_r2"] = result.fit_r2
    row[f"{side}_fit_error"] = result.fit_error
    row[f"{side}_valid_point_count"] = result.inlier_count or result.valid_point_count
    row[f"{side}_taper_status"] = result.status


def make_result_row(item: ImageItem, settings: MeasurementSettings) -> Dict[str, object]:
    result: Optional[MeasurementResult] = item.result
    roi = settings.roi or ("", "", "", "")
    row: Dict[str, object] = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "file_name": item.file_name,
            "full_path": item.image_path,
            "group_id": item.group_id,
            "group_name": item.group_name,
            "measurement_type": settings.measurement_type,
            "roi_apply_mode": settings.roi_apply_mode,
            "roi_source_image": settings.roi_source_image,
            "settings_source": settings.settings_source,
            "roi_x1": roi[0],
            "roi_y1": roi[1],
            "roi_x2": roi[2],
            "roi_y2": roi[3],
            "edge_reference": settings.edge_reference,
            "noise_level": settings.noise_level,
            "distance_method": settings.distance_method,
            "px_to_real": settings.calibration.px_to_real,
            "unit": settings.calibration.unit,
            "calibration_mode": settings.calibration.mode,
            "calibration_status": settings.calibration.status,
            "detected_scale_bar_px": settings.calibration.detected_scale_bar_px,
            "actual_scale_bar_length": settings.calibration.actual_scale_bar_length,
            "taper_side": settings.taper_side,
        }
    )
    if result is None:
        row["status"] = "Not measured"
        return row

    row["avg_taper_angle"] = result.avg_taper_angle
    row["taper_angle_diff"] = result.taper_angle_diff
    _taper_values("left", row, result.left_taper)
    _taper_values("right", row, result.right_taper)
    _distance_values("horizontal_cd", row, result.horizontal_cd, settings)
    _distance_values("vertical_thk", row, result.vertical_thk, settings)
    row["overall_confidence"] = result.overall_confidence
    row["status"] = result.status
    row["warning_message"] = result.warning_message
    row["measurement_source"] = result.measurement_source
    return row


def export_results_to_csv(
    path: str,
    image_items: Iterable[ImageItem],
    group_settings: Dict[str, object],
    global_settings: MeasurementSettings,
) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in image_items:
            settings = resolve_effective_settings(item, group_settings, global_settings)
            row = make_result_row(item, settings)
            writer.writerow({key: _fmt(row.get(key, "")) for key in CSV_COLUMNS})


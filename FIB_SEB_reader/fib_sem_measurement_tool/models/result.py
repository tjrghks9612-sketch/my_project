from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DistanceResult:
    orientation: str
    mean_px: Optional[float] = None
    max_px: Optional[float] = None
    min_px: Optional[float] = None
    median_px: Optional[float] = None
    std_px: Optional[float] = None
    selected_px: Optional[float] = None
    selected_method: str = "mean"
    valid_count: int = 0
    total_count: int = 0
    confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    boundary_pairs: List[Tuple[float, float, float]] = field(default_factory=list)
    values_px: List[float] = field(default_factory=list)

    def scaled(self, px_to_real: float) -> Dict[str, Optional[float]]:
        if not px_to_real:
            px_to_real = 1.0
        return {
            "mean": self.mean_px * px_to_real if self.mean_px is not None else None,
            "max": self.max_px * px_to_real if self.max_px is not None else None,
            "min": self.min_px * px_to_real if self.min_px is not None else None,
            "median": self.median_px * px_to_real if self.median_px is not None else None,
            "std": self.std_px * px_to_real if self.std_px is not None else None,
            "selected": self.selected_px * px_to_real if self.selected_px is not None else None,
        }


@dataclass
class TaperSideResult:
    side: str
    angle_horizontal: Optional[float] = None
    angle_vertical: Optional[float] = None
    fit_r2: Optional[float] = None
    fit_error: Optional[float] = None
    valid_point_count: int = 0
    inlier_count: int = 0
    confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    points: List[Tuple[float, float]] = field(default_factory=list)
    fit_line: Optional[Tuple[float, float, float, float]] = None


@dataclass
class MeasurementResult:
    measurement_type: str
    horizontal_cd: Optional[DistanceResult] = None
    vertical_thk: Optional[DistanceResult] = None
    left_taper: Optional[TaperSideResult] = None
    right_taper: Optional[TaperSideResult] = None
    avg_taper_angle: Optional[float] = None
    taper_angle_diff: Optional[float] = None
    overall_confidence: float = 0.0
    status: str = "Fail"
    warning_message: str = ""
    measurement_source: str = "auto"

    def compact_summary(self, unit: str, px_to_real: float) -> str:
        chunks: List[str] = []
        if self.horizontal_cd and self.horizontal_cd.selected_px is not None:
            value = self.horizontal_cd.selected_px * px_to_real
            chunks.append(f"CD {value:.3g} {unit}")
        if self.vertical_thk and self.vertical_thk.selected_px is not None:
            value = self.vertical_thk.selected_px * px_to_real
            chunks.append(f"THK {value:.3g} {unit}")
        if self.left_taper and self.left_taper.angle_horizontal is not None:
            chunks.append(f"L {self.left_taper.angle_horizontal:.1f}°")
        if self.right_taper and self.right_taper.angle_horizontal is not None:
            chunks.append(f"R {self.right_taper.angle_horizontal:.1f}°")
        chunks.append(f"{self.status} {self.overall_confidence:.0f}%")
        return " | ".join(chunks)

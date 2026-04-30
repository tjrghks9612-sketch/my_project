from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fib_sem_measurement_tool.models.settings import MeasurementSettings


def generate_scan_lines(
    roi: Sequence[int],
    direction: str,
    count: int,
    margin_ratio: float = 0.12,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    x1, y1, x2, y2 = [int(v) for v in roi]
    count = max(3, int(count))
    margin_ratio = max(0.0, min(0.45, margin_ratio))
    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if direction == "horizontal":
        start = y1 + (y2 - y1) * margin_ratio
        end = y2 - (y2 - y1) * margin_ratio
        for y in np.linspace(start, end, count):
            iy = int(round(y))
            lines.append(((x1, iy), (x2, iy)))
    else:
        start = x1 + (x2 - x1) * margin_ratio
        end = x2 - (x2 - x1) * margin_ratio
        for x in np.linspace(start, end, count):
            ix = int(round(x))
            lines.append(((ix, y1), (ix, y2)))
    return lines


def extract_profile(gray: np.ndarray, line: Tuple[Tuple[int, int], Tuple[int, int]], thickness: int = 1) -> np.ndarray:
    (x1, y1), (x2, y2) = line
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    if y1 == y2:
        y = max(0, min(gray.shape[0] - 1, y1))
        half = max(0, thickness // 2)
        y0 = max(0, y - half)
        y3 = min(gray.shape[0], y + half + 1)
        return np.mean(gray[y0:y3, min(x1, x2) : max(x1, x2) + 1], axis=0)
    if x1 == x2:
        x = max(0, min(gray.shape[1] - 1, x1))
        half = max(0, thickness // 2)
        x0 = max(0, x - half)
        x3 = min(gray.shape[1], x + half + 1)
        return np.mean(gray[min(y1, y2) : max(y1, y2) + 1, x0:x3], axis=1)

    length = int(np.hypot(x2 - x1, y2 - y1)) + 1
    xs = np.linspace(x1, x2, length).astype(np.float32)
    ys = np.linspace(y1, y2, length).astype(np.float32)
    return cv2.remap(gray, xs.reshape(1, -1), ys.reshape(1, -1), cv2.INTER_LINEAR).reshape(-1)


def _smooth_profile(profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    if profile.size < 7:
        return profile
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    kernel /= kernel.sum()
    return np.convolve(profile, kernel, mode="same")


def detect_edge_candidates(profile: Sequence[float], settings: MeasurementSettings) -> List[Dict[str, float]]:
    data = _smooth_profile(np.asarray(profile, dtype=np.float32))
    if data.size < 8:
        return []
    grad = np.gradient(data)
    mag = np.abs(grad)
    mag[:2] = 0
    mag[-2:] = 0

    median = float(np.median(mag))
    std = float(np.std(mag))
    max_mag = float(np.max(mag))
    if max_mag <= 1e-6:
        return []

    threshold = median + settings.advanced.sensitivity * std
    threshold = max(threshold, settings.advanced.peak_prominence * max_mag)
    threshold = min(threshold, max_mag * 0.92)

    candidates: List[Dict[str, float]] = []
    for idx in range(2, len(mag) - 2):
        strength = float(mag[idx])
        if strength < threshold:
            continue
        if strength >= mag[idx - 1] and strength >= mag[idx + 1]:
            denom = float(mag[idx - 1] - 2.0 * mag[idx] + mag[idx + 1])
            if abs(denom) > 1e-6:
                offset = 0.5 * float(mag[idx - 1] - mag[idx + 1]) / denom
                offset = max(-0.75, min(0.75, offset))
            else:
                offset = 0.0
            candidates.append(
                {
                    "index": float(idx) + offset,
                    "strength": strength,
                    "signed_gradient": float(grad[idx]),
                }
            )
    return candidates


def group_edge_bands(candidates: Iterable[Dict[str, float]], settings: MeasurementSettings) -> List[Dict[str, float]]:
    sorted_candidates = sorted(candidates, key=lambda item: item["index"])
    if not sorted_candidates:
        return []

    bands: List[List[Dict[str, float]]] = [[sorted_candidates[0]]]
    for candidate in sorted_candidates[1:]:
        if candidate["index"] - bands[-1][-1]["index"] <= 3:
            bands[-1].append(candidate)
        else:
            bands.append([candidate])

    grouped: List[Dict[str, float]] = []
    for band in bands:
        peak = max(band, key=lambda item: item["strength"])
        indices = [item["index"] for item in band]
        grouped.append(
            {
                "start": float(min(indices)),
                "end": float(max(indices)),
                "center": float((min(indices) + max(indices)) / 2.0),
                "peak": float(peak["index"]),
                "strength": float(peak["strength"]),
                "signed_gradient": float(peak["signed_gradient"]),
            }
        )
    return grouped


def select_edge_from_band(edge_band: Dict[str, float], edge_reference: str, side: str) -> float:
    if edge_reference == "center":
        return float(edge_band["center"])
    return float(edge_band["peak"])


def _choose_band(bands: List[Dict[str, float]], edge_reference: str, side: str, length: int) -> Optional[Dict[str, float]]:
    if not bands:
        return None
    center = length / 2.0
    margin = max(3.0, length * 0.02)
    if side in {"left", "top"}:
        side_bands = [band for band in bands if margin <= band["peak"] <= center - 1]
        if not side_bands:
            side_bands = [band for band in bands if band["peak"] <= center]
        if not side_bands:
            return None
        if edge_reference == "outer":
            return min(side_bands, key=lambda item: item["peak"])
        if edge_reference == "inner":
            return max(side_bands, key=lambda item: item["peak"])
        return max(side_bands, key=lambda item: item["strength"])

    side_bands = [band for band in bands if center + 1 <= band["peak"] <= length - margin]
    if not side_bands:
        side_bands = [band for band in bands if band["peak"] >= center]
    if not side_bands:
        return None
    if edge_reference == "outer":
        return max(side_bands, key=lambda item: item["peak"])
    if edge_reference == "inner":
        return min(side_bands, key=lambda item: item["peak"])
    return max(side_bands, key=lambda item: item["strength"])


def find_boundary(profile: Sequence[float], side: str, settings: MeasurementSettings) -> Optional[float]:
    candidates = detect_edge_candidates(profile, settings)
    bands = group_edge_bands(candidates, settings)
    band = _choose_band(bands, settings.edge_reference, side, len(profile))
    if band is None:
        return None
    return select_edge_from_band(band, settings.edge_reference, side)


def find_boundary_pair(profile: Sequence[float], settings: MeasurementSettings) -> Optional[Tuple[float, float]]:
    candidates = detect_edge_candidates(profile, settings)
    bands = group_edge_bands(candidates, settings)
    left_band = _choose_band(bands, settings.edge_reference, "left", len(profile))
    right_band = _choose_band(bands, settings.edge_reference, "right", len(profile))
    if left_band is None or right_band is None:
        return None
    left = select_edge_from_band(left_band, settings.edge_reference, "left")
    right = select_edge_from_band(right_band, settings.edge_reference, "right")
    if right - left < 4:
        return None
    return left, right


def _side_candidates(bands: List[Dict[str, float]], edge_reference: str, side: str, length: int) -> List[Dict[str, float]]:
    center = length / 2.0
    margin = max(3.0, length * 0.02)
    if side in {"left", "top"}:
        side_bands = [band for band in bands if margin <= band["peak"] <= center - 1]
    else:
        side_bands = [band for band in bands if center + 1 <= band["peak"] <= length - margin]

    candidates: List[Dict[str, float]] = []
    for band in side_bands:
        candidates.append(
            {
                "position": select_edge_from_band(band, edge_reference, side),
                "strength": float(band["strength"]),
                "band": band,
            }
        )
    return candidates


def _dominant_position(candidates: List[Dict[str, float]], length: int) -> Optional[float]:
    if not candidates:
        return None
    positions = np.asarray([item["position"] for item in candidates], dtype=np.float32)
    strengths = np.asarray([max(item["strength"], 1e-6) for item in candidates], dtype=np.float32)
    if len(positions) == 1:
        return float(positions[0])

    bin_count = max(12, min(80, int(length / 5)))
    hist, edges = np.histogram(positions, bins=bin_count, range=(0, length - 1), weights=strengths)
    peak_idx = int(np.argmax(hist))
    bin_center = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
    window = max(5.0, length * 0.055)
    near = np.abs(positions - bin_center) <= window
    if int(np.count_nonzero(near)) >= 2:
        return float(np.average(positions[near], weights=strengths[near]))
    strongest_idx = int(np.argmax(strengths))
    return float(positions[strongest_idx])


def _choose_near_anchor(candidates: List[Dict[str, float]], anchor: Optional[float], length: int) -> Optional[float]:
    if not candidates:
        return None
    if anchor is None:
        return float(max(candidates, key=lambda item: item["strength"])["position"])

    max_strength = max(item["strength"] for item in candidates)
    tolerance = max(6.0, length * 0.08)
    hard_limit = max(tolerance * 2.5, length * 0.16)
    best_item = None
    best_score = -float("inf")
    for item in candidates:
        distance = abs(float(item["position"]) - anchor)
        if distance > hard_limit:
            continue
        strength_score = float(item["strength"]) / max(max_strength, 1e-6)
        score = strength_score - 0.55 * (distance / tolerance)
        if score > best_score:
            best_score = score
            best_item = item
    if best_item is None:
        return None
    return float(best_item["position"])


def find_coherent_boundary_pairs(profiles: Sequence[Sequence[float]], settings: MeasurementSettings) -> List[Optional[Tuple[float, float]]]:
    if not profiles:
        return []
    length = len(profiles[0])
    per_line_candidates: List[Tuple[List[Dict[str, float]], List[Dict[str, float]]]] = []
    all_left: List[Dict[str, float]] = []
    all_right: List[Dict[str, float]] = []

    for profile in profiles:
        candidates = detect_edge_candidates(profile, settings)
        bands = group_edge_bands(candidates, settings)
        left = _side_candidates(bands, settings.edge_reference, "left", len(profile))
        right = _side_candidates(bands, settings.edge_reference, "right", len(profile))
        per_line_candidates.append((left, right))
        all_left.extend(left)
        all_right.extend(right)

    left_anchor = _dominant_position(all_left, length)
    right_anchor = _dominant_position(all_right, length)
    pairs: List[Optional[Tuple[float, float]]] = []
    for left_candidates, right_candidates in per_line_candidates:
        left = _choose_near_anchor(left_candidates, left_anchor, length)
        right = _choose_near_anchor(right_candidates, right_anchor, length)
        if left is None or right is None or right - left < 4:
            pairs.append(None)
        else:
            pairs.append((left, right))
    return pairs

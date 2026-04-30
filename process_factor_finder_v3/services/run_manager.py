"""Analysis execution orchestration."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from core.data_profiler import candidate_columns, profile_dataframe
from core.detailed_analyzer import analyze_candidates
from core.models import AnalysisPlan, RunManifest
from core.screening_engine import screen_candidates

ProgressCallback = Callable[[dict], None]


class ProgressThrottle:
    """Throttle Streamlit progress updates from hot loops."""

    def __init__(self, callback: ProgressCallback | None, interval_sec: float = 0.2) -> None:
        self.callback = callback
        self.interval_sec = interval_sec
        self.last_time = 0.0
        self.last_percent = -1.0
        self.last_phase = ""

    def __call__(self, event: dict) -> None:
        if not self.callback:
            return
        now = time.time()
        total = max(float(event.get("total_pairs", 1) or 1), 1.0)
        percent = float(event.get("processed_pairs", 0)) / total
        phase = str(event.get("phase", ""))
        if now - self.last_time >= self.interval_sec or percent - self.last_percent >= 0.01 or phase != self.last_phase:
            self.last_time = now
            self.last_percent = percent
            self.last_phase = phase
            self.callback(event)


def _append_csv(path: str | Path, row: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**row, "created_at": datetime.now().isoformat(timespec="seconds")}])
    frame.to_csv(target, mode="a", header=not target.exists(), index=False, encoding="utf-8-sig")


def execute_analysis(
    df: pd.DataFrame,
    plan: AnalysisPlan,
    config: dict,
    artifact_paths: dict,
    key_cols: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, RunManifest, pd.DataFrame]:
    """Run profile, screening, detailed analysis, and scoring."""
    started_at = datetime.now().isoformat(timespec="seconds")
    throttle = ProgressThrottle(progress_callback, float(config.get("ui", {}).get("progress_update_interval_sec", 0.2)))
    throttle(
        {
            "phase": "분석 준비",
            "processed_pairs": 0,
            "total_pairs": 1,
            "pairs_per_sec": 0.0,
            "eta_seconds": 0,
            "message": "분석 대상 데이터와 설정을 확인하는 중입니다.",
        }
    )
    key_cols = key_cols or []
    requested_y = plan.y_columns or ([plan.y_target] if plan.y_target else [])
    throttle(
        {
            "phase": "컬럼 프로파일링",
            "processed_pairs": 0,
            "total_pairs": 1,
            "pairs_per_sec": 0.0,
            "eta_seconds": 0,
            "message": "Key, ID성 컬럼, 결측/상수 컬럼을 확인하는 중입니다.",
        }
    )
    x_profile = profile_dataframe(df, "x", config, key_cols=key_cols, y_cols=requested_y)
    y_profile = profile_dataframe(df, "y", config, key_cols=key_cols)
    x_numeric = [col for col in candidate_columns(x_profile, "numeric") if not plan.x_columns or col in plan.x_columns]
    y_numeric = [col for col in candidate_columns(y_profile, "numeric") if not requested_y or col in requested_y]
    x_cat = [col for col in candidate_columns(x_profile, "categorical_low_cardinality") if not plan.x_columns or col in plan.x_columns]
    y_cat = [col for col in candidate_columns(y_profile, "categorical_low_cardinality") if not requested_y or col in requested_y]
    if plan.mode == "single_y" and plan.y_target:
        y_numeric = [col for col in y_numeric if col == plan.y_target]
        y_cat = [col for col in y_cat if col == plan.y_target]

    scanned_pairs = len(x_numeric) * len(y_numeric)
    if plan.include_categorical:
        scanned_pairs += len(x_cat) * len(y_numeric) + len(x_numeric) * len(y_cat) + len(x_cat) * len(y_cat)

    throttle(
        {
            "phase": "후보 스크리닝",
            "processed_pairs": 0,
            "total_pairs": max(scanned_pairs, 1),
            "pairs_per_sec": 0.0,
            "eta_seconds": 0,
            "message": f"총 {scanned_pairs:,}개 X-Y 후보 조합을 빠르게 훑는 중입니다.",
        }
    )
    candidates = screen_candidates(
        df=df,
        x_numeric=x_numeric,
        y_numeric=y_numeric,
        x_categorical=x_cat,
        y_categorical=y_cat,
        plan=plan,
        config=config,
        progress_callback=throttle,
    )
    temp_dir = Path(artifact_paths["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    if not candidates.empty:
        candidates.to_parquet(temp_dir / "internal_screening_candidates.parquet", index=False)
    errors: list[dict] = []

    def log_error(row: dict) -> None:
        errors.append(row)
        _append_csv(artifact_paths["error_log"], row)

    throttle(
        {
            "phase": "정밀 분석",
            "processed_pairs": min(len(candidates), plan.detailed_top_n),
            "total_pairs": max(min(len(candidates), plan.detailed_top_n), 1),
            "pairs_per_sec": 0.0,
            "eta_seconds": 0,
            "message": f"스크리닝 후보 {len(candidates):,}개 중 상위 {min(len(candidates), plan.detailed_top_n):,}개를 정밀 확인하는 중입니다.",
        }
    )
    result = analyze_candidates(df, candidates, plan, config, error_callback=log_error)
    if not result.empty:
        result.to_parquet(temp_dir / "internal_detailed_results.parquet", index=False)
    throttle(
        {
            "phase": "결과 저장",
            "processed_pairs": max(scanned_pairs, 1),
            "total_pairs": max(scanned_pairs, 1),
            "pairs_per_sec": 0.0,
            "eta_seconds": 0,
            "message": "내부 결과와 컬럼 프로파일 요약을 저장하는 중입니다.",
        }
    )
    profile = pd.concat([x_profile.assign(profile_role="x"), y_profile.assign(profile_role="y")], ignore_index=True)
    profile.to_csv(Path(artifact_paths["logs_dir"]) / "column_profile_summary.csv", index=False, encoding="utf-8-sig")
    finished_at = datetime.now().isoformat(timespec="seconds")
    manifest = RunManifest(
        run_id=str(artifact_paths["run_id"]),
        output_dir=str(artifact_paths["run_dir"]),
        status="completed",
        scanned_pairs=int(scanned_pairs),
        screened_candidates=int(len(candidates)),
        detailed_candidates=int(min(len(candidates), plan.detailed_top_n)),
        final_results=int(len(result)),
        started_at=started_at,
        finished_at=finished_at,
        artifact_paths={},
    )
    return result, manifest, profile

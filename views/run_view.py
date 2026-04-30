"""Run view."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.cards import kpi_card
from components.layout import page_header, section_header
from components.messages import warning_message
from services.artifact_manager import ArtifactManager
from services.data_store import DataStore
from services.run_manager import execute_analysis
from style.copy_ko import BUTTONS, EMPTY_STATES, MODE_LABELS, PAGE_LABELS, PAGE_SUBTITLES


def _fmt_eta(seconds: float | int) -> str:
    try:
        sec = max(int(float(seconds)), 0)
    except (TypeError, ValueError):
        sec = 0
    if sec >= 3600:
        return f"{sec // 3600}시간 {(sec % 3600) // 60}분"
    if sec >= 60:
        return f"{sec // 60}분 {sec % 60}초"
    return f"{sec}초"


def _phase_help(phase: str) -> str:
    helps = {
        "분석 준비": "업로드/병합된 데이터와 분석 계획을 확인하고 있습니다.",
        "컬럼 프로파일링": "Key, ID성 컬럼, 결측률, 상수 컬럼을 확인해 분석 후보를 정리하고 있습니다.",
        "후보 스크리닝": "전체 X-Y 후보를 빠르게 훑어 정밀분석할 후보를 줄이고 있습니다.",
        "numeric matrix screening": "숫자-숫자 조합을 행렬 계산으로 빠르게 선별하고 있습니다.",
        "categorical-numeric screening": "범주-숫자 조합에서 값/조건별 차이를 빠르게 확인하고 있습니다.",
        "정밀 분석": "상위 후보에 대해 p-value, 효과 크기, 모델 점수, 안정성, 품질 점수를 계산하고 있습니다.",
        "결과 저장": "내부 결과 파일과 로그를 저장하고 있습니다.",
    }
    return helps.get(phase, "현재 단계의 후보 조합을 처리하고 있습니다.")


def _render_status(event: dict, placeholder) -> None:
    phase = str(event.get("phase", "-"))
    total = max(float(event.get("total_pairs", 1) or 1), 1.0)
    done = max(float(event.get("processed_pairs", 0) or 0), 0.0)
    percent = min(done / total * 100.0, 100.0)
    speed = float(event.get("pairs_per_sec", 0.0) or 0.0)
    eta = _fmt_eta(event.get("eta_seconds", 0))
    message = str(event.get("message", "")) or _phase_help(phase)
    updated_at = datetime.now().strftime("%H:%M:%S")
    with placeholder.container():
        st.markdown(
            f"""
            <div class="pff-run-status">
              <div class="pff-run-status-top">
                <div>
                  <div class="pff-run-phase">{phase}</div>
                  <div class="pff-run-message">{message}</div>
                </div>
                <div class="pff-run-percent">{percent:,.1f}%</div>
              </div>
              <div class="pff-run-grid">
                <div><span>처리 pair</span><strong>{int(done):,} / {int(total):,}</strong></div>
                <div><span>처리 속도</span><strong>{speed:,.1f} pair/sec</strong></div>
                <div><span>예상 남은 시간</span><strong>{eta}</strong></div>
                <div><span>마지막 업데이트</span><strong>{updated_at}</strong></div>
              </div>
              <div class="pff-run-help">{_phase_help(phase)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render(config: dict, store: DataStore) -> None:
    page_header(PAGE_LABELS["Run"], PAGE_SUBTITLES["Run"])
    plan = st.session_state.get("analysis_plan")
    merged_manifest = st.session_state.get("merged_manifest")
    merge_manifest = st.session_state.get("merge_manifest")
    if not plan or not merged_manifest or not merge_manifest:
        warning_message(EMPTY_STATES["run_need_plan"])
        return
    cols = st.columns(3)
    with cols[0]:
        kpi_card("분석 모드", MODE_LABELS.get(plan.mode, plan.mode), "저장된 계획", icon="모드", accent="cyan")
    with cols[1]:
        kpi_card("프리셋", plan.preset, "후보 수 설정", icon="필터", accent="purple")
    with cols[2]:
        kpi_card("정밀 후보", f"{plan.detailed_top_n:,}", "상세 분석 대상", icon="후보", accent="teal")

    if st.button(BUTTONS["analysis_run"], type="primary", use_container_width=True):
        artifacts = ArtifactManager(config.get("artifacts", {}).get("output_root", "output")).create_run()
        st.session_state.current_artifacts = artifacts
        progress = st.progress(0.0)
        status_area = st.empty()
        log_area = st.empty()

        def update(event: dict) -> None:
            total = max(float(event.get("total_pairs", 1) or 1), 1.0)
            done = float(event.get("processed_pairs", 0) or 0)
            progress.progress(min(done / total, 1.0))
            _render_status(event, status_area)
            log_area.caption(
                f"현재 단계: {event.get('phase', '-')} | "
                f"처리 {int(done):,}/{int(total):,} pair | "
                f"속도 {float(event.get('pairs_per_sec', 0.0) or 0.0):,.1f} pair/sec"
            )

        try:
            update(
                {
                    "phase": "데이터 로딩",
                    "processed_pairs": 0,
                    "total_pairs": 1,
                    "pairs_per_sec": 0.0,
                    "eta_seconds": 0,
                    "message": "병합된 parquet 데이터를 메모리로 읽는 중입니다.",
                }
            )
            df = store.load(merged_manifest)
            result, run_manifest, profile = execute_analysis(
                df,
                plan,
                config,
                artifacts,
                key_cols=merge_manifest.x_keys + merge_manifest.y_keys,
                progress_callback=update,
            )
            st.session_state.result_df = result
            st.session_state.run_manifest = run_manifest
            st.session_state.profile_df = profile
            progress.progress(1.0)
            _render_status(
                {
                    "phase": "분석 완료",
                    "processed_pairs": max(run_manifest.scanned_pairs, 1),
                    "total_pairs": max(run_manifest.scanned_pairs, 1),
                    "pairs_per_sec": 0.0,
                    "eta_seconds": 0,
                    "message": f"최종 결과 {len(result):,}건을 만들었습니다.",
                },
                status_area,
            )
            section_header("분석 완료")
            st.success(f"분석이 완료되었습니다. 결과 {len(result):,}건")
            with st.expander("저장 위치 보기", expanded=False):
                st.code(run_manifest.output_dir)
        except Exception as exc:
            st.error(f"분석 실패: {exc}")

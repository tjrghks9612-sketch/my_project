"""Home view."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from components.cards import action_card, hero_card, kpi_card, workflow_step
from components.layout import page_header, section_header
from style.copy_ko import APP_NAME, BUTTONS, LABELS, PAGE_SUBTITLES


def _next_action() -> tuple[str, str, str]:
    if not st.session_state.get("x_manifests") or not st.session_state.get("y_manifests"):
        return BUTTONS["go_data"], "Data", "X/Y 파일을 먼저 업로드하세요."
    if not st.session_state.get("merged_manifest"):
        return BUTTONS["go_merge"], "Merge", "Key 기준으로 X/Y 데이터를 병합하세요."
    if not st.session_state.get("analysis_plan"):
        return BUTTONS["go_plan"], "Analysis Plan", "탐색 범위와 정밀분석 후보 수를 정하세요."
    if st.session_state.get("result_df") is None:
        return BUTTONS["go_run"], "Run", "저장한 분석 계획으로 후보 탐색을 실행하세요."
    return BUTTONS["go_results"], "Results", "상위 후보와 저장 산출물을 확인하세요."


def render(config: dict, navigate: Callable[[str], None] | None = None) -> None:
    page_header(APP_NAME, PAGE_SUBTITLES["Home"])
    hero_card(
        "공정 분석 워크벤치",
        "공정 데이터(X)와 품질/결과 데이터(Y)를 Key로 병합합니다.\n"
        "전체 후보 조합을 빠르게 스크리닝하고, 상위 후보만 정밀 분석합니다.\n"
        "결과는 우선 확인 후보이며 최종 판단은 검증을 통해 진행합니다.",
        icon="홈",
        accent="blue",
    )

    x_count = len(st.session_state.get("x_manifests", []))
    y_count = len(st.session_state.get("y_manifests", []))
    merged = st.session_state.get("merged_manifest")
    result = st.session_state.get("result_df")
    cols = st.columns(4)
    with cols[0]:
        kpi_card(LABELS["x_files"], f"{x_count}개", "업로드 상태", icon="X", accent="cyan")
    with cols[1]:
        kpi_card(LABELS["y_files"], f"{y_count}개", "업로드 상태", icon="Y", accent="purple")
    with cols[2]:
        kpi_card(LABELS["merge"], "완료" if merged else "대기", "X/Y 최종 병합", icon="M", accent="teal")
    with cols[3]:
        kpi_card(LABELS["result"], 0 if result is None else len(result), "최종 후보 수", icon="R", accent="amber")

    section_header("작업 흐름", "각 단계는 같은 카드 체계로 표시해 현재 위치를 빠르게 파악할 수 있습니다.")
    flow_cols = st.columns(4)
    flow = [
        ("1", "데이터 준비", "X/Y 파일을 로드합니다."),
        ("2", "병합 확인", "Key 기준 병합 상태를 확인합니다."),
        ("3", "후보 탐색", "상위 후보만 정밀 확인합니다."),
        ("4", "결과 저장", "그래프와 캡쳐 이미지만 저장합니다."),
    ]
    for col, (number, title, caption) in zip(flow_cols, flow):
        with col:
            workflow_step(number, title, caption)

    section_header("다음 작업")
    label, page, description = _next_action()
    if action_card("현재 상태에 맞는 다음 단계", description, label, key="home_next_action"):
        if navigate:
            navigate(page)
        else:
            st.session_state.page = page
            st.session_state.nav_page = page
        st.rerun()

"""Settings view."""

from __future__ import annotations

import streamlit as st

from components.cards import section_card
from components.layout import page_header, section_header
from style.copy_ko import PAGE_LABELS, PAGE_SUBTITLES


def _json_panel(title: str, caption: str, data: dict, expanded: bool = False) -> None:
    section_header(title, caption)
    with st.expander("설정값 보기", expanded=expanded):
        st.json(data)


def render(config: dict) -> None:
    page_header(PAGE_LABELS["Settings"], PAGE_SUBTITLES["Settings"])
    section_card(
        "이 화면은 현재 설정값을 확인하는 용도입니다.\n"
        "새 설정 항목을 추가하지 않고, config.yaml에 정의된 값만 보기 좋게 나누어 보여줍니다.",
        "설정 개요",
        icon="설정",
        accent="purple",
    )
    cols = st.columns(2, gap="large")
    with cols[0]:
        _json_panel("일반 설정", "앱 이름, 언어, 대상 Python 버전", config.get("app", {}), expanded=True)
        _json_panel("대용량/화면 설정", "미리보기 제한과 UI 업데이트 간격", {**config.get("data", {}), **config.get("ui", {})})
        _json_panel("그래프/캡쳐 저장 설정", "사용자-facing 이미지 산출물 저장 기준", config.get("artifacts", {}))
    with cols[1]:
        _json_panel("숫자 matrix screening", "숫자-숫자 후보 선별 설정", config.get("numeric_matrix_screening", {}))
        _json_panel("범주 one-hot screening", "저카디널리티 범주형 후보 선별 설정", config.get("categorical_matrix_screening", {}))
        _json_panel("점수 설정", "최종 점수 검증과 가중치", config.get("scoring", {}))

"""Data upload view."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.cards import kpi_card, section_card, status_card
from components.layout import page_header, section_header
from components.messages import info_message
from components.tables import preview_table
from core.data_loader import load_manifest_preview
from services.data_store import DataStore
from style.copy_ko import BUTTONS, EMPTY_STATES, LABELS, PAGE_LABELS, PAGE_SUBTITLES


def _manifest_value(manifest, field: str):
    return getattr(manifest, field) if hasattr(manifest, field) else manifest[field]


def _show_manifests(title: str, manifests: list[dict]) -> None:
    section_header(title, "미리보기는 최대 100행, 80열까지만 표시합니다.")
    if not manifests:
        info_message(EMPTY_STATES["no_upload"])
        return
    for index, manifest in enumerate(manifests, start=1):
        files = _manifest_value(manifest, "file_paths")
        file_name = Path(files[0]).name if files else "-"
        status_card(f"{index}. {file_name}", "캐시에 저장됨", "완료", "CSV")
        cols = st.columns(3)
        with cols[0]:
            kpi_card(LABELS["rows"], f"{_manifest_value(manifest, 'row_count'):,}", icon="행", accent="cyan")
        with cols[1]:
            kpi_card(LABELS["columns"], f"{_manifest_value(manifest, 'column_count'):,}", icon="열", accent="purple")
        with cols[2]:
            kpi_card(LABELS["file"], file_name, icon="파일", accent="teal", tooltip=file_name)
        preview_table(load_manifest_preview(manifest))


def render(config: dict, store: DataStore) -> None:
    page_header(PAGE_LABELS["Data"], PAGE_SUBTITLES["Data"])
    data_cfg = config.get("data", {})
    section_card(
        "CSV, Excel, Parquet 파일을 지원합니다.\n"
        "대형 데이터는 원본 전체를 화면에 그리지 않고 최대 100행, 80열 미리보기만 표시합니다.",
        "업로드 영역",
        icon="업",
        accent="blue",
    )
    with st.form("upload_form"):
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("#### X 공정 데이터")
            x_files = st.file_uploader("X 파일 선택", type=["csv", "xlsx", "xls", "parquet"], accept_multiple_files=True)
        with right:
            st.markdown("#### Y 품질/결과 데이터")
            y_files = st.file_uploader("Y 파일 선택", type=["csv", "xlsx", "xls", "parquet"], accept_multiple_files=True)
        submitted = st.form_submit_button(BUTTONS["upload_save"], type="primary", use_container_width=True)
    if submitted:
        if x_files:
            st.session_state.x_manifests = store.save_uploaded_files(
                x_files,
                "x",
                int(data_cfg.get("preview_rows", 100)),
                int(data_cfg.get("preview_max_columns", 80)),
            )
        if y_files:
            st.session_state.y_manifests = store.save_uploaded_files(
                y_files,
                "y",
                int(data_cfg.get("preview_rows", 100)),
                int(data_cfg.get("preview_max_columns", 80)),
            )
        st.success("업로드 데이터를 캐시에 저장했습니다.")

    x_count = len(st.session_state.get("x_manifests", []))
    y_count = len(st.session_state.get("y_manifests", []))
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        kpi_card("X 파일", f"{x_count}개", "로드된 X 파일", icon="X", accent="cyan")
    with kpi_cols[1]:
        kpi_card("Y 파일", f"{y_count}개", "로드된 Y 파일", icon="Y", accent="purple")
    with kpi_cols[2]:
        x_cols = sum(_manifest_value(m, "column_count") for m in st.session_state.get("x_manifests", []))
        kpi_card("X 열", f"{x_cols:,}", "파일 합계", icon="열", accent="blue")
    with kpi_cols[3]:
        y_cols = sum(_manifest_value(m, "column_count") for m in st.session_state.get("y_manifests", []))
        kpi_card("Y 열", f"{y_cols:,}", "파일 합계", icon="열", accent="teal")

    left, right = st.columns(2, gap="large")
    with left:
        _show_manifests("X 데이터", st.session_state.get("x_manifests", []))
    with right:
        _show_manifests("Y 데이터", st.session_state.get("y_manifests", []))

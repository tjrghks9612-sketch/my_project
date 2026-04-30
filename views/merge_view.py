"""Merge view."""

from __future__ import annotations

import streamlit as st

from components.cards import key_chips, kpi_card, section_card, warning_card
from components.layout import page_header, section_header
from components.messages import warning_message
from components.tables import preview_table
from core.data_loader import load_manifest_preview
from core.merge_engine import merge_tables_on_keys, merge_x_y
from services.data_store import DataStore
from style.copy_ko import BUTTONS, EMPTY_STATES, PAGE_LABELS, PAGE_SUBTITLES


def _default_key(columns: list[str]) -> str:
    for key in ("SSN", "sn", "Serial", "LotID", "KEY", "Key", "key"):
        if key in columns:
            return key
    return columns[0] if columns else ""


def _common_columns(manifests: list) -> list[str]:
    if not manifests:
        return []
    sets = [set(manifest.columns if hasattr(manifest, "columns") else manifest["columns"]) for manifest in manifests]
    return sorted(set.intersection(*sets)) if sets else []


def _all_columns(manifests: list) -> list[str]:
    columns: set[str] = set()
    for manifest in manifests:
        columns.update(manifest.columns if hasattr(manifest, "columns") else manifest["columns"])
    return sorted(columns)


def _key_selector(label: str, columns: list[str], default_key: str = "", allow_empty: bool = False) -> list[str]:
    """Select one or more key columns in a single searchable selector."""
    if not columns:
        st.warning(f"{label} 후보 컬럼이 없습니다.")
        return []
    use_key = True
    if allow_empty:
        use_key = st.checkbox(f"{label} 사용", value=True, key=f"{label}_use")
    if not use_key:
        key_chips([], "이 단계는 건너뜁니다.")
        return []

    default_values = [default_key] if default_key in columns else [columns[0]]
    selected = st.multiselect(
        label,
        columns,
        default=default_values,
        key=f"{label}_multi",
        help="하나 또는 여러 개의 Key 컬럼을 자유롭게 선택하세요. 복합 Key는 선택한 순서대로 사용됩니다.",
    )
    if not selected:
        st.warning("최소 1개의 Key 컬럼을 선택하세요.")
    key_chips(selected)
    return selected


def render(config: dict, store: DataStore) -> None:
    page_header(PAGE_LABELS["Merge"], PAGE_SUBTITLES["Merge"])
    x_manifests = st.session_state.get("x_manifests", [])
    y_manifests = st.session_state.get("y_manifests", [])
    if not x_manifests or not y_manifests:
        warning_message(EMPTY_STATES["merge_need_upload"])
        return

    x_common = _common_columns(x_manifests)
    y_common = _common_columns(y_manifests)
    x_all = _all_columns(x_manifests)
    y_all = _all_columns(y_manifests)
    section_card(
        "여러 파일을 넣은 경우 X 내부, Y 내부를 먼저 Key 기준으로 정리합니다.\n"
        "그 다음 최종 X Key와 최종 Y Key를 기준으로 분석용 데이터를 병합합니다.",
        "Key 병합",
        icon="Key",
        accent="purple",
    )
    with st.form("merge_form"):
        st.caption("Key는 실제 업로드된 컬럼 목록에서 선택합니다. 하나 또는 여러 개의 Key를 자유롭게 선택할 수 있습니다.")
        top_cols = st.columns(2, gap="large")
        with top_cols[0]:
            section_header("X 내부 병합 Key", "X 파일이 여러 개인 경우 공통 컬럼 기준으로 정리합니다.")
            x_internal_keys = _key_selector("X 내부 병합 Key", x_common, _default_key(x_common), allow_empty=True)
        with top_cols[1]:
            section_header("Y 내부 병합 Key", "Y 파일이 여러 개인 경우 공통 컬럼 기준으로 정리합니다.")
            y_internal_keys = _key_selector("Y 내부 병합 Key", y_common, _default_key(y_common), allow_empty=True)
        bottom_cols = st.columns(2, gap="large")
        with bottom_cols[0]:
            section_header("최종 X Key", "최종 X/Y 병합에 사용할 X 컬럼입니다.")
            final_x_keys = _key_selector("최종 X Key", x_all, _default_key(x_all), allow_empty=False)
        with bottom_cols[1]:
            section_header("최종 Y Key", "최종 X/Y 병합에 사용할 Y 컬럼입니다.")
            final_y_keys = _key_selector("최종 Y Key", y_all, _default_key(y_all), allow_empty=False)
        st.markdown('<div class="pff-submit-spacer"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button(BUTTONS["merge_run"], type="primary", use_container_width=True)
    if submitted:
        try:
            x_frames = store.load_many(x_manifests)
            y_frames = store.load_many(y_manifests)
            x_merged, x_stats = merge_tables_on_keys(x_frames, x_internal_keys, "X")
            y_merged, y_stats = merge_tables_on_keys(y_frames, y_internal_keys, "Y")
            merged, merge_manifest = merge_x_y(x_merged, y_merged, final_x_keys, final_y_keys)
            st.session_state.x_merged_manifest = store.save_frame(x_merged, "x_merged")
            st.session_state.y_merged_manifest = store.save_frame(y_merged, "y_merged")
            st.session_state.merged_manifest = store.save_frame(merged, "merged")
            st.session_state.merge_manifest = merge_manifest
            st.session_state.internal_merge_stats = {"x": x_stats, "y": y_stats}
            st.success("병합이 완료되었습니다.")
        except Exception as exc:
            st.error(f"병합 실패: {exc}")

    merge_manifest = st.session_state.get("merge_manifest")
    merged_manifest = st.session_state.get("merged_manifest")
    if merge_manifest:
        section_header("병합 결과")
        merged_cols = getattr(merged_manifest, "column_count", "-") if merged_manifest else "-"
        cols = st.columns(5)
        with cols[0]:
            kpi_card("병합 행 수", f"{merge_manifest.merged_rows:,}", icon="행", accent="cyan")
        with cols[1]:
            kpi_card("병합 열 수", f"{merged_cols:,}" if isinstance(merged_cols, int) else merged_cols, icon="열", accent="teal")
        with cols[2]:
            kpi_card("X 병합률", f"{merge_manifest.x_merge_rate:.1f}%", icon="X", accent="blue")
        with cols[3]:
            kpi_card("Y 병합률", f"{merge_manifest.y_merge_rate:.1f}%", icon="Y", accent="purple")
        with cols[4]:
            kpi_card("미매칭", f"X {merge_manifest.x_unmatched_rows:,} / Y {merge_manifest.y_unmatched_rows:,}", icon="!", accent="amber")
        if merge_manifest.x_unmatched_rows or merge_manifest.y_unmatched_rows:
            warning_card("미매칭 행이 있습니다. Key 형식, 중복, 공백/대소문자 차이를 확인하세요.", "병합 확인 필요")
        section_header("병합 미리보기", "최대 100행, 80열만 표시합니다.")
        preview_table(load_manifest_preview(st.session_state.merged_manifest))

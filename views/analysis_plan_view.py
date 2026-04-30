"""Analysis planning view."""

from __future__ import annotations

import streamlit as st

from components.cards import kpi_card, section_card, status_card
from components.forms import filter_columns
from components.layout import page_header, section_header
from components.messages import warning_message
from core.models import AnalysisPlan
from services.data_store import DataStore
from style.copy_ko import BUTTONS, EMPTY_STATES, LABELS, PAGE_LABELS, PAGE_SUBTITLES, RISK_LABELS


def estimate_pair_count(x_columns: list[str], y_columns: list[str]) -> int:
    """Estimate pair count without building every X-Y pair."""
    overlap = len(set(x_columns) & set(y_columns))
    return max(len(x_columns) * len(y_columns) - overlap, 0)


def _risk(total_pairs: int) -> str:
    if total_pairs < 5_000:
        return "low"
    if total_pairs < 50_000:
        return "medium"
    if total_pairs < 250_000:
        return "high"
    return "very high"


def _mode_value(mode_label: str) -> str:
    return "single_y" if mode_label == "단일 Y 정밀 분석" else "full_scan"


def render(config: dict, store: DataStore) -> None:
    page_header(PAGE_LABELS["Analysis Plan"], PAGE_SUBTITLES["Analysis Plan"])
    merged_manifest = st.session_state.get("merged_manifest")
    y_manifest = st.session_state.get("y_merged_manifest")
    merge_manifest = st.session_state.get("merge_manifest")
    if not merged_manifest or not y_manifest or not merge_manifest:
        warning_message(EMPTY_STATES["plan_need_merge"])
        return
    merged_columns = list(merged_manifest.columns)
    y_columns_all = [col for col in y_manifest.columns if col in merged_columns and col not in merge_manifest.y_keys]
    x_columns_all = [col for col in merged_columns if col not in y_columns_all and col not in merge_manifest.x_keys]
    if not y_columns_all:
        warning_message(EMPTY_STATES["plan_no_y"])
        return

    presets = config.get("presets", {})
    default_preset = "균형" if "균형" in presets else list(presets.keys())[0]

    section_card(
        "전체 후보 빠른 탐색은 스크리닝으로 후보를 줄인 뒤 상위 후보만 정밀 분석합니다.\n"
        "단일 Y 정밀 분석은 선택한 Y 하나에 집중해 X 후보를 찾습니다.",
        "분석 모드 선택",
        icon="계획",
        accent="cyan",
    )
    mode_cols = st.columns(2, gap="large")
    with mode_cols[0]:
        status_card("전체 후보 빠른 탐색", "많은 X-Y 조합을 빠르게 훑고 상위 후보만 정밀 확인", "준비", "전체")
    with mode_cols[1]:
        status_card("단일 Y 정밀 분석", "선택한 Y 하나에 대한 영향 후보 X를 집중 탐색", "준비", "Y")

    with st.form("analysis_plan_form"):
        mode_label = st.radio("분석 모드", ["전체 후보 빠른 탐색", "단일 Y 정밀 분석"], horizontal=True)
        preset = st.selectbox(LABELS["preset"], list(presets.keys()), index=list(presets.keys()).index(default_preset))
        preset_values = presets[preset]
        include_categorical = st.checkbox(LABELS["include_categorical"], value=True)
        y_target = ""
        if mode_label == "단일 Y 정밀 분석":
            y_target = st.selectbox("분석 대상 Y", y_columns_all)
        cols = st.columns(3)
        with cols[0]:
            top_n_per_y = st.number_input(LABELS["top_n_per_y"], min_value=5, max_value=2000, value=int(preset_values.get("top_n_per_y", 100)), step=5)
        with cols[1]:
            detailed_top_n = st.number_input(LABELS["detailed_top_n"], min_value=10, max_value=5000, value=int(preset_values.get("detailed_top_n", 300)), step=10)
        with cols[2]:
            final_top_n = st.number_input(LABELS["final_top_n"], min_value=1, max_value=200, value=int(preset_values.get("final_top_n", 30)), step=1)
        with st.expander("고급 후보 선택", expanded=False):
            st.caption("전체 컬럼 목록을 직접 렌더링하지 않습니다. regex 또는 컬럼명 붙여넣기로 후보 범위만 좁힙니다.")
            x_include = st.text_input("X include regex", value="")
            x_exclude = st.text_input("X exclude regex", value="")
            x_paste = st.text_area("X 컬럼명 직접 입력", value="", height=80)
            y_include = st.text_input("Y include regex", value="")
            y_exclude = st.text_input("Y exclude regex", value="")
            y_paste = st.text_area("Y 컬럼명 직접 입력", value="", height=80)
            max_category_levels = st.number_input(
                "max_category_levels",
                min_value=2,
                max_value=200,
                value=int(config.get("categorical_matrix_screening", {}).get("max_category_levels", 30)),
                help="범주형으로 볼 최대 값 개수입니다. 30을 크게 넘기면 one-hot screening이 느려질 수 있습니다.",
            )
            numeric_method = st.selectbox("numeric matrix method", ["pearson", "spearman"])
            x_chunk_size = st.number_input(
                "X chunk size",
                min_value=100,
                max_value=50000,
                value=int(config.get("numeric_matrix_screening", {}).get("x_chunk_size", 5000)),
                step=100,
                help="숫자형 X 컬럼을 한 번에 몇 개씩 행렬 screening할지 정합니다.",
            )
            y_chunk_size = st.number_input(
                "Y chunk size",
                min_value=1,
                max_value=1000,
                value=int(config.get("numeric_matrix_screening", {}).get("y_chunk_size", 50)),
                step=1,
                help="숫자형 Y 컬럼을 한 번에 몇 개씩 행렬 screening할지 정합니다.",
            )
        submitted = st.form_submit_button(BUTTONS["plan_save"], type="primary", use_container_width=True)

    x_selected, x_warnings = filter_columns(x_columns_all, x_include, x_exclude, x_paste, "X")
    y_base = [y_target] if y_target else y_columns_all
    y_selected, y_warnings = filter_columns(y_base, y_include, y_exclude, y_paste, "Y")
    for message in x_warnings + y_warnings:
        st.warning(message)

    total_pairs = estimate_pair_count(x_selected, y_selected)
    risk = _risk(total_pairs)
    section_header("분석 예상 규모", "pair 수는 컬럼 개수 기반으로 빠르게 계산합니다.")
    cols = st.columns(4)
    with cols[0]:
        kpi_card("X 후보", f"{len(x_selected):,}", "분석 대상 X", icon="X", accent="cyan")
    with cols[1]:
        kpi_card("Y 후보", f"{len(y_selected):,}", "분석 대상 Y", icon="Y", accent="purple")
    with cols[2]:
        kpi_card("예상 pair", f"{total_pairs:,}", "전체 후보 조합", icon="쌍", accent="blue")
    with cols[3]:
        kpi_card("위험도", RISK_LABELS.get(risk, risk), "예상 소요 위험", icon="!", accent="amber")
    if max_category_levels > 100:
        st.warning("max_category_levels가 100을 초과하면 one-hot screening이 느려질 수 있습니다.")
    if not x_selected or not y_selected:
        st.warning("선택된 X 또는 Y 후보가 없습니다. regex/직접 입력 조건을 확인하세요.")

    if submitted and x_selected and y_selected:
        plan = AnalysisPlan(
            mode=_mode_value(mode_label),
            preset=preset,
            include_categorical=bool(include_categorical),
            top_n_per_y=int(top_n_per_y),
            detailed_top_n=int(detailed_top_n),
            final_top_n=int(final_top_n),
            y_target=y_target,
            x_columns=x_selected,
            y_columns=y_selected,
            max_category_levels=int(max_category_levels),
            numeric_method=str(numeric_method),
            x_chunk_size=int(x_chunk_size),
            y_chunk_size=int(y_chunk_size),
        )
        st.session_state.analysis_plan = plan
        st.success("분석 계획을 저장했습니다.")

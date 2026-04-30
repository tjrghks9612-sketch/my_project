"""Results view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components.cards import conclusion_card, kpi_card
from components.charts import pair_chart_lazy
from components.layout import page_header, section_header
from components.messages import not_causal_notice, warning_message
from components.tables import rank_list, result_table
from core.plot_engine import save_top_graphs
from services.artifact_manager import append_error_log
from services.capture_service import save_visible_results_capture
from services.data_store import DataStore
from style.copy_ko import BUTTONS, EMPTY_STATES, LABELS, MODE_LABELS, PAGE_LABELS, PAGE_SUBTITLES, PAIR_TYPE_LABELS


def _fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_p(value) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.2e}"
    except (TypeError, ValueError):
        return str(value)


def _short_mode(mode: str) -> str:
    if mode == "full_scan":
        return "전체 탐색"
    if mode == "single_y":
        return "단일 Y"
    return str(MODE_LABELS.get(mode, mode))


def apply_result_filters(result: pd.DataFrame, x_search: str, y_search: str, pair_type: str, score_threshold: int) -> pd.DataFrame:
    """Apply lightweight result filters without rendering the full table."""
    filtered = result
    if x_search:
        filtered = filtered[filtered["x_col"].astype(str).str.contains(x_search, case=False, na=False)]
    if y_search:
        filtered = filtered[filtered["y_col"].astype(str).str.contains(y_search, case=False, na=False)]
    if pair_type not in {"전체", "?꾩껜"}:
        reverse = {label: key for key, label in PAIR_TYPE_LABELS.items()}
        filtered = filtered[filtered["pair_type"] == reverse.get(pair_type, pair_type)]
    return filtered[filtered["final_score"] >= score_threshold]


def displayed_results(filtered: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return only the rows that should be rendered on screen."""
    bounded_top_n = max(1, min(int(top_n), 100))
    return filtered.sort_values("final_score", ascending=False).head(bounded_top_n).copy()


def candidate_labels(df: pd.DataFrame) -> list[str]:
    """Build selection labels only for displayed candidates."""
    if df is None or df.empty:
        return []
    return (
        df["rank"].astype(str)
        + ". "
        + df["x_col"].astype(str)
        + " → "
        + df["y_col"].astype(str)
        + " | "
        + df["final_score"].map(lambda value: f"{float(value):.1f}")
    ).tolist()


def _graph_columns(result_df: pd.DataFrame, top_n: int) -> list[str]:
    top = result_df.sort_values("final_score", ascending=False).head(top_n)
    cols = list(top["x_col"].astype(str)) + list(top["y_col"].astype(str))
    return list(dict.fromkeys(cols))


def _detail_panel(row: pd.Series | dict) -> None:
    row = pd.Series(row)
    pair_type = str(row.get("pair_type", "-"))
    conclusion_card(
        "한 줄 결론",
        str(row.get("interpretation", "")) or "선택한 후보의 상세 설명입니다.",
        [
            "우선 확인 후보",
            PAIR_TYPE_LABELS.get(pair_type, pair_type),
            f"최종 점수 {_fmt(row.get('final_score'), 1)}",
        ],
        "purple",
    )
    first = st.columns(3)
    with first[0]:
        kpi_card("최종 점수", _fmt(row.get("final_score"), 1), icon="점", accent="purple")
    with first[1]:
        kpi_card("효과 크기", _fmt(row.get("effect_size"), 3), icon="효", accent="cyan")
    with first[2]:
        metric_label = "R²" if pd.notna(row.get("r2_score")) else "모델 점수"
        metric_value = row.get("r2_score") if pd.notna(row.get("r2_score")) else row.get("model_score")
        kpi_card(metric_label, _fmt(metric_value, 3 if metric_label == "R²" else 1), icon="모", accent="blue")
    second = st.columns(3)
    with second[0]:
        kpi_card("p-value", _fmt_p(row.get("p_value")), icon="p", accent="teal")
    with second[1]:
        kpi_card("보정 p-value", _fmt_p(row.get("adjusted_p_value")), icon="q", accent="muted")
    with second[2]:
        kpi_card("표본 수", f"{int(row.get('sample_n', 0)):,}", icon="n", accent="amber")


def render(config: dict, store: DataStore) -> None:
    page_header(PAGE_LABELS["Results"], PAGE_SUBTITLES["Results"])
    result = st.session_state.get("result_df")
    run_manifest = st.session_state.get("run_manifest")
    artifacts = st.session_state.get("current_artifacts")
    merged_manifest = st.session_state.get("merged_manifest")
    if result is None or result.empty or not run_manifest or not artifacts:
        warning_message(EMPTY_STATES["result_empty"])
        return
    not_causal_notice()

    cols = st.columns(4)
    with cols[0]:
        kpi_card(LABELS["analysis_mode"], _short_mode(st.session_state.analysis_plan.mode), MODE_LABELS.get(st.session_state.analysis_plan.mode, ""), icon="모", accent="cyan")
    with cols[1]:
        kpi_card("전체 스캔 pair", f"{run_manifest.scanned_pairs:,}", "screening 대상", icon="쌍", accent="blue")
    with cols[2]:
        kpi_card("정밀 후보", f"{run_manifest.detailed_candidates:,}", "상세 분석 대상", icon="정", accent="purple")
    with cols[3]:
        kpi_card("최종 후보", f"{run_manifest.final_results:,}", "표시/저장 대상", icon="Top", accent="teal")
    with st.expander("저장 위치 보기", expanded=False):
        st.code(run_manifest.output_dir)

    pair_type_options = ["전체"] + [PAIR_TYPE_LABELS.get(value, value) for value in sorted(result["pair_type"].dropna().unique().tolist())]
    default_top_n = 10
    with st.expander("필터 열기", expanded=False):
        st.caption("필터를 조정한 뒤 적용하면 화면에는 상위 후보만 다시 표시합니다.")
        with st.form("result_filter_form"):
            fcols = st.columns([1.2, 1.2, 1.0, 1.1, 1.0])
            with fcols[0]:
                x_search = st.text_input(LABELS["x_search"], value="")
            with fcols[1]:
                y_search = st.text_input(LABELS["y_search"], value="")
            with fcols[2]:
                pair_type = st.selectbox(LABELS["pair_type"], pair_type_options)
            with fcols[3]:
                score_threshold = st.slider(LABELS["score_threshold"], min_value=0, max_value=100, value=0)
            with fcols[4]:
                top_n = st.number_input(LABELS["top_n_display"], min_value=5, max_value=100, value=default_top_n, step=5)
            st.form_submit_button(BUTTONS["filter_apply"], type="primary", use_container_width=True)
    if "x_search" not in locals():
        x_search = ""
        y_search = ""
        pair_type = "전체"
        score_threshold = 0
        top_n = default_top_n

    filtered = apply_result_filters(result, x_search, y_search, pair_type, score_threshold)
    displayed_df = displayed_results(filtered, int(top_n))
    st.caption(f"필터 결과 {len(filtered):,}건 중 상위 {len(displayed_df):,}건만 표시합니다.")

    if displayed_df.empty:
        rank_list(displayed_df, 0)
        return

    section_header("유의 후보 탐색", "후보 순위와 선택 후보 상세를 확인합니다.")
    list_col, detail_col = st.columns([1.0, 1.0], gap="large")
    with detail_col:
        labels = candidate_labels(displayed_df)
        selected = st.selectbox(LABELS["candidate_select"], labels, key="result_candidate_select")
        selected_row = displayed_df.iloc[labels.index(selected)]
        _detail_panel(selected_row)
    with list_col:
        try:
            selected_rank = int(selected_row["rank"]) if pd.notna(selected_row.get("rank")) else None
        except (TypeError, ValueError):
            selected_rank = None
        rank_list(displayed_df, len(displayed_df), selected_rank=selected_rank)
        with st.expander("상위 후보 상세 테이블", expanded=False):
            result_table(displayed_df, len(displayed_df))

    section_header("선택 후보 그래프", "선택한 후보의 X/Y 두 컬럼만 읽어 화면 가운데에 크게 표시합니다.")
    pair_chart_lazy(
        lambda: store.load_columns(merged_manifest, [str(selected_row["x_col"]), str(selected_row["y_col"])]),
        selected_row,
        "result_pair_chart",
    )

    section_header("저장", "사용자 산출물은 그래프 PNG와 현재 결과 화면 캡쳐 PNG만 제공합니다.")
    save_cols = st.columns(2, gap="large")
    with save_cols[0]:
        if st.button(BUTTONS["save_graphs"], type="primary", use_container_width=True):
            graph_n = int(st.session_state.analysis_plan.final_top_n)
            graph_result = result.sort_values("final_score", ascending=False).head(graph_n).copy()
            graph_df = store.load_columns(merged_manifest, _graph_columns(graph_result, graph_n))
            paths, errors = save_top_graphs(graph_df, graph_result, artifacts["graphs_dir"], graph_n)
            append_error_log(artifacts["error_log"], errors)
            st.success(f"그래프 {len(paths):,}개 저장 완료, {len(errors):,}개 실패")
            if errors:
                st.caption("실패 상세는 error_log.csv에 기록했습니다.")
            with st.expander("저장 위치 보기", expanded=False):
                st.code(artifacts["graphs_dir"])
    with save_cols[1]:
        if st.button(BUTTONS["save_capture"], use_container_width=True):
            capture_path = Path(artifacts["captures_dir"]) / "significance_analysis_page.png"
            try:
                saved = save_visible_results_capture(capture_path)
                st.success("현재 결과 화면 캡쳐 저장 완료")
                with st.expander("저장 위치 보기", expanded=False):
                    st.code(saved)
            except Exception as exc:
                st.error(f"현재 화면 캡쳐 실패: {exc}")
                st.caption("브라우저에서 결과 화면을 전면에 둔 뒤 다시 실행하세요.")

"""Table and ranking display helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from style.copy_ko import EMPTY_STATES, PAIR_TYPE_LABELS


def _fmt_number(value, digits: int = 3) -> str:
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


def _short_text(value: object, limit: int = 42) -> str:
    text = "-" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def preview_table(df: pd.DataFrame, rows: int = 100, columns: int = 80) -> None:
    if df is None or df.empty:
        st.info(EMPTY_STATES["no_display_data"])
        return
    st.dataframe(df.iloc[:rows, : min(columns, df.shape[1])], use_container_width=True, hide_index=True)


def rank_list(df: pd.DataFrame, top_n: int = 30, selected_rank: int | None = None) -> None:
    """Render displayed candidates as a stable, readable Streamlit table."""
    if df is None or df.empty:
        st.info(EMPTY_STATES["no_result"])
        return

    shown = df.head(top_n).copy()
    table = pd.DataFrame(
        {
            "순위": shown["rank"].map(lambda value: f"#{value}"),
            "조합": shown.apply(
                lambda row: f"{_short_text(row.get('x_col', '-'), 44)} → {_short_text(row.get('y_col', '-'), 44)}",
                axis=1,
            ),
            "조합 유형": shown["pair_type"].map(lambda value: PAIR_TYPE_LABELS.get(str(value), str(value))),
            "최종 점수": shown["final_score"].map(lambda value: max(0.0, min(float(value or 0), 100.0))),
        }
    )
    if selected_rank is not None:
        table.insert(1, "선택", shown["rank"].map(lambda value: "선택" if str(value) == str(selected_rank) else ""))

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "순위": st.column_config.TextColumn("순위", width="small"),
            "선택": st.column_config.TextColumn("선택", width="small"),
            "조합": st.column_config.TextColumn("조합", width="large"),
            "조합 유형": st.column_config.TextColumn("조합 유형", width="small"),
            "최종 점수": st.column_config.ProgressColumn(
                "최종 점수",
                min_value=0,
                max_value=100,
                format="%.1f",
                width="medium",
            ),
        },
    )


def result_table(df: pd.DataFrame, top_n: int = 30) -> None:
    """Render a limited dataframe as a secondary detail view."""
    if df is None or df.empty:
        st.info(EMPTY_STATES["no_result"])
        return
    cols = [
        "rank",
        "pair_type",
        "x_col",
        "y_col",
        "final_score",
        "screening_score",
        "effect_size",
        "model_score",
        "r2_score",
        "p_value",
        "adjusted_p_value",
        "sample_n",
        "interpretation",
    ]
    existing = [col for col in cols if col in df.columns]
    shown = df[existing].head(top_n).copy()
    rename_map = {
        "rank": "순위",
        "pair_type": "조합 유형",
        "x_col": "X",
        "y_col": "Y",
        "final_score": "최종 점수",
        "screening_score": "선별 점수",
        "effect_size": "효과 크기",
        "model_score": "모델 점수",
        "r2_score": "R²",
        "p_value": "p-value",
        "adjusted_p_value": "보정 p-value",
        "sample_n": "표본 수",
        "interpretation": "한 줄 해석",
    }
    if "pair_type" in shown.columns:
        shown["pair_type"] = shown["pair_type"].map(lambda value: PAIR_TYPE_LABELS.get(str(value), str(value)))
    for col in ("final_score", "screening_score", "effect_size", "model_score", "r2_score"):
        if col in shown.columns:
            shown[col] = shown[col].map(lambda value: _fmt_number(value, 3 if col in {"effect_size", "r2_score"} else 1))
    for col in ("p_value", "adjusted_p_value"):
        if col in shown.columns:
            shown[col] = shown[col].map(_fmt_p)
    st.dataframe(shown.rename(columns=rename_map), use_container_width=True, hide_index=True)

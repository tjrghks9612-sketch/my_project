"""Chart display wrappers."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import streamlit as st

from core.plot_engine import make_plotly_figure
from style.copy_ko import LABELS

SHOW_GRAPH_TEXT = "그래프 보기"

CHART_MODE_LABELS = {
    "auto": "자동 선택",
    "scatter": "산점도",
    "box": "박스플롯",
    "ratio": "비율/요약 그래프",
}


def _mode_options(row: pd.Series | dict) -> list[str]:
    pair_type = str(dict(row).get("pair_type", ""))
    if pair_type == "numeric_numeric":
        return ["auto", "scatter", "box"]
    if pair_type in {"categorical_numeric", "numeric_categorical"}:
        return ["auto", "box", "scatter", "ratio"]
    return ["auto", "ratio", "scatter"]


def on_demand_pair_chart(df: pd.DataFrame, row: pd.Series | dict, key: str) -> None:
    """Backward compatible chart wrapper."""
    chart_mode = st.selectbox(
        "그래프 종류",
        _mode_options(row),
        format_func=lambda value: CHART_MODE_LABELS.get(value, value),
        key=f"{key}_mode",
    )
    show = st.toggle(LABELS["show_graph"], value=False, key=key)
    if show:
        st.plotly_chart(make_plotly_figure(df, row, chart_mode=chart_mode), use_container_width=True)
    else:
        st.caption("그래프는 필요할 때만 렌더링합니다.")


def on_demand_pair_chart_lazy(load_df: Callable[[], pd.DataFrame], row: pd.Series | dict, key: str) -> None:
    """Backward compatible lazy chart wrapper with a show/hide toggle."""
    chart_mode = st.selectbox(
        "그래프 종류",
        _mode_options(row),
        format_func=lambda value: CHART_MODE_LABELS.get(value, value),
        key=f"{key}_mode",
    )
    show = st.toggle(LABELS["show_graph"], value=False, key=key)
    if not show:
        st.caption("그래프는 필요할 때만 렌더링합니다.")
        return
    with st.spinner("그래프에 필요한 컬럼만 읽는 중입니다."):
        df = load_df()
    st.plotly_chart(make_plotly_figure(df, row, chart_mode=chart_mode), use_container_width=True)


def pair_chart_lazy(load_df: Callable[[], pd.DataFrame], row: pd.Series | dict, key: str) -> None:
    """Render the selected pair chart without a separate show/hide toggle."""
    chart_mode = st.selectbox(
        "그래프 종류",
        _mode_options(row),
        format_func=lambda value: CHART_MODE_LABELS.get(value, value),
        key=f"{key}_mode",
    )
    with st.spinner("선택 후보 그래프를 그리는 중입니다. 필요한 두 컬럼만 읽습니다."):
        df = load_df()
    st.plotly_chart(make_plotly_figure(df, row, chart_mode=chart_mode), use_container_width=True)

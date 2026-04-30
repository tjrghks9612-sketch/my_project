"""Reusable premium card components."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


ACCENTS = {
    "cyan": "#22d3ee",
    "purple": "#8b5cf6",
    "blue": "#2563eb",
    "teal": "#14b8a6",
    "amber": "#f59e0b",
    "success": "#4ade80",
    "danger": "#fb7185",
    "muted": "#64748b",
}


def _text(value: object) -> str:
    return escape("-" if value is None else str(value))


def _accent(value: str | None) -> str:
    if not value:
        return ACCENTS["cyan"]
    return ACCENTS.get(value, value)


def kpi_card(
    label: str,
    value: str | int | float,
    caption: str = "",
    icon: str = "K",
    accent: str = "cyan",
    tooltip: str | None = None,
) -> None:
    """Render a KPI card with icon, accent, and overflow protection."""
    st.markdown(
        f"""
        <div class="pff-kpi-card" style="--accent:{_accent(accent)};" title="{_text(tooltip or value)}">
          <div class="pff-kpi-icon">{_text(icon)}</div>
          <div class="pff-kpi-content">
            <div class="pff-kpi-label">{_text(label)}</div>
            <div class="pff-kpi-value">{_text(value)}</div>
            <div class="pff-kpi-caption">{_text(caption)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(title: str, caption: str = "", state: str = "대기", icon: str = "S") -> None:
    """Render a compact state card for sidebar and screen summaries."""
    state_key = "ok" if state in {"완료", "준비", "성공", "있음", "저장됨"} else "wait"
    st.markdown(
        f"""
        <div class="pff-status-card">
          <div class="pff-status-left">
            <div class="pff-status-icon">{_text(icon)}</div>
            <div class="pff-status-copy">
              <div class="pff-card-title">{_text(title)}</div>
              <div class="pff-note">{_text(caption)}</div>
            </div>
          </div>
          <div class="pff-state-pill {state_key}">{_text(state)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_card(title: str, message: str, icon: str = "PFF", accent: str = "blue") -> None:
    """Render the main hero panel used on the home page."""
    st.markdown(
        f"""
        <div class="pff-hero-card" style="--accent:{_accent(accent)};">
          <div class="pff-hero-icon">{_text(icon)}</div>
          <div class="pff-hero-copy">
            <div class="pff-hero-title">{_text(title)}</div>
            <div class="pff-hero-text">{_text(message)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conclusion_card(title: str, main_text: str, meta_items: list[str] | None = None, accent: str = "purple") -> None:
    """Render a highlighted conclusion card."""
    meta_html = "".join(f"<span>{_text(item)}</span>" for item in (meta_items or []))
    st.markdown(
        f"""
        <div class="pff-conclusion" style="--accent:{_accent(accent)};">
          <div class="pff-conclusion-icon">TOP</div>
          <div class="pff-conclusion-body">
            <div class="pff-conclusion-title">{_text(title)}</div>
            <div class="pff-conclusion-main">{_text(main_text)}</div>
            <div class="pff-conclusion-meta">{meta_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def action_card(title: str, description: str, button_label: str | None = None, state: str = "ready", key: str | None = None) -> bool:
    """Render a next-step action card and optional primary button."""
    st.markdown(
        f"""
        <div class="pff-action-card" data-state="{_text(state)}">
          <div class="pff-action-topline">다음 작업</div>
          <div class="pff-card-title">{_text(title)}</div>
          <div class="pff-note">{_text(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if button_label:
        return st.button(button_label, type="primary", use_container_width=True, key=key)
    return False


def warning_card(message: str, title: str = "주의") -> None:
    _message_card(title, message, "warning")


def info_card(message: str, title: str = "안내") -> None:
    _message_card(title, message, "info")


def empty_state_card(message: str, title: str = "아직 준비된 내용이 없습니다") -> None:
    _message_card(title, message, "empty")


def section_card(message: str, title: str = "", icon: str | None = None, accent: str = "blue") -> None:
    """Render a generic panel with optional icon."""
    title_html = f'<div class="pff-card-title">{_text(title)}</div>' if title else ""
    icon_html = f'<div class="pff-panel-icon" style="--accent:{_accent(accent)};">{_text(icon)}</div>' if icon else ""
    st.markdown(
        f"""
        <div class="pff-panel pff-panel-flex">
          {icon_html}
          <div class="pff-panel-main">
            {title_html}
            <div class="pff-panel-copy">{_text(message)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def workflow_step(number: str | int, title: str, caption: str) -> None:
    """Render one workflow preview tile."""
    st.markdown(
        f"""
        <div class="pff-workflow-step">
          <div class="pff-workflow-number">{_text(number)}</div>
          <div class="pff-card-title">{_text(title)}</div>
          <div class="pff-note">{_text(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def key_chips(keys: list[str], empty_text: str = "선택된 Key 없음") -> None:
    """Render selected key columns as compact chips."""
    if not keys:
        st.caption(empty_text)
        return
    html = "".join(f'<span class="pff-key-chip">{_text(key)}</span>' for key in keys)
    st.markdown(f'<div class="pff-key-chip-wrap">{html}</div>', unsafe_allow_html=True)


def detail_grid(items: list[tuple[str, str, str]]) -> None:
    """Render equal-width detail cards with safe wrapping."""
    tiles = []
    for title, value, accent in items:
        safe_value = _text(value).replace("\n", "<br>")
        tiles.append(
            f"""
            <div class="pff-detail-tile" style="--accent:{_accent(accent)};">
              <div class="pff-detail-tile-label">{_text(title)}</div>
              <div class="pff-detail-tile-value">{safe_value}</div>
            </div>
            """
        )
    st.markdown(f'<div class="pff-detail-grid">{"".join(tiles)}</div>', unsafe_allow_html=True)


def metric_grid(row: pd.Series | dict, items: list[tuple[str, str, str, str]]) -> None:
    """Render a row of KPI cards from a result row."""
    row_dict = dict(row)
    cols = st.columns(len(items))
    for col, (label, field, icon, accent) in zip(cols, items):
        value = row_dict.get(field, "-")
        if isinstance(value, float):
            value = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
        with col:
            kpi_card(label, value, icon=icon, accent=accent)


def badge(label: str, accent: str = "cyan") -> str:
    """Return escaped badge HTML for use inside component-rendered HTML."""
    return f'<span class="pff-badge" style="--badge-color:{_accent(accent)};">{_text(label)}</span>'


def _message_card(title: str, message: str, kind: str) -> None:
    st.markdown(
        f"""
        <div class="pff-message pff-message-{_text(kind)}">
          <strong>{_text(title)}</strong>
          <span>{_text(message)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str | int | float, caption: str = "") -> None:
    """Backward compatible KPI alias."""
    kpi_card(label, value, caption)


def panel(message: str) -> None:
    """Backward compatible panel alias."""
    section_card(message)

"""Layout components."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


def load_css(path: str | Path) -> None:
    css_path = Path(path)
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "Process Factor Finder") -> None:
    st.markdown(
        f"""
        <div class="pff-topbar">
          <div>
            <div class="pff-eyebrow">{escape(str(eyebrow))}</div>
            <div class="pff-page-title">{escape(str(title))}</div>
            <div class="pff-page-subtitle">{escape(str(subtitle))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand(title: str, subtitle: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="pff-brand">
          <div class="pff-brand-mark">PFF</div>
          <div class="pff-brand-title">{escape(str(title))}</div>
          <div class="pff-brand-sub">{escape(str(subtitle))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_section(title: str) -> None:
    st.sidebar.markdown(f'<div class="pff-side-section">{escape(str(title))}</div>', unsafe_allow_html=True)


def section_header(title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="pff-section-header">
          <div class="pff-section-title">{escape(str(title))}</div>
          <div class="pff-section-caption">{escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sub_section_header(title: str, caption: str = "") -> None:
    st.markdown(f"#### {title}")
    if caption:
        safe_caption(caption)


def two_column_panel(left_ratio: int = 1, right_ratio: int = 1):
    return st.columns([left_ratio, right_ratio], gap="large")


def safe_caption(text: str) -> None:
    st.caption(text)

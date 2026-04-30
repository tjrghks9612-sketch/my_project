"""Process Factor Finder v3 Streamlit router."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from components.cards import status_card
from components.layout import load_css, sidebar_brand, sidebar_section
from services.config_service import load_config
from services.data_store import DataStore
from style.copy_ko import APP_NAME, APP_SUBTITLE, BUTTONS, LABELS, PAGE_LABELS
from views import analysis_plan_view, data_view, home_view, merge_view, results_view, run_view, settings_view


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
STYLE_PATH = BASE_DIR / "assets" / "style.css"


PAGES = ["Home", "Data", "Merge", "Analysis Plan", "Run", "Results", "Settings"]


def init_state() -> None:
    defaults = {
        "page": "Home",
        "nav_page": "Home",
        "x_manifests": [],
        "y_manifests": [],
        "x_merged_manifest": None,
        "y_merged_manifest": None,
        "merged_manifest": None,
        "merge_manifest": None,
        "analysis_plan": None,
        "result_df": None,
        "run_manifest": None,
        "current_artifacts": None,
        "profile_df": None,
        "pending_page": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.get("page") not in PAGES:
        st.session_state.page = "Home"
    if st.session_state.get("nav_page") not in PAGES:
        st.session_state.nav_page = st.session_state.page


def reset_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def move_to_page(page: str) -> None:
    """Synchronize programmatic navigation with the sidebar radio state."""
    if page in PAGES:
        st.session_state.pending_page = page


def sidebar(config: dict) -> str:
    pending_page = st.session_state.get("pending_page")
    if pending_page in PAGES:
        st.session_state.page = pending_page
        st.session_state.nav_page = pending_page
        st.session_state.pending_page = None
    sidebar_brand(APP_NAME, APP_SUBTITLE)
    sidebar_section(LABELS["navigation"])
    page = st.sidebar.radio(
        " ",
        PAGES,
        format_func=lambda key: PAGE_LABELS.get(key, key),
        label_visibility="collapsed",
        key="nav_page",
    )
    st.session_state.page = page
    x_count = len(st.session_state.get("x_manifests", []))
    y_count = len(st.session_state.get("y_manifests", []))
    with st.sidebar:
        sidebar_section(LABELS["status"])
        status_card(LABELS["x_files"], f"{x_count}개 로드됨", "완료" if x_count else "대기", "X")
        status_card(LABELS["y_files"], f"{y_count}개 로드됨", "완료" if y_count else "대기", "Y")
        status_card(LABELS["merge"], "X/Y 최종 병합", "완료" if st.session_state.get("merged_manifest") else "대기", "M")
        run_manifest = st.session_state.get("run_manifest")
        status_card(LABELS["recent_run"], run_manifest.status if run_manifest else "아직 실행 없음", "완료" if run_manifest else "대기", "R")
        st.divider()
    if st.sidebar.button(BUTTONS["reset_session"], use_container_width=True):
        reset_session()
        st.rerun()
    return page


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="PFF", layout="wide")
    load_css(STYLE_PATH)
    init_state()
    config = load_config(CONFIG_PATH)
    store = DataStore(BASE_DIR / "output" / "cache")
    page = sidebar(config)

    if page == "Home":
        home_view.render(config, move_to_page)
    elif page == "Data":
        data_view.render(config, store)
    elif page == "Merge":
        merge_view.render(config, store)
    elif page == "Analysis Plan":
        analysis_plan_view.render(config, store)
    elif page == "Run":
        run_view.render(config, store)
    elif page == "Results":
        results_view.render(config, store)
    elif page == "Settings":
        settings_view.render(config)


if __name__ == "__main__":
    main()

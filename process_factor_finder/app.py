"""Streamlit app for Process Factor Finder."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis_engine import (
    analyze_factors,
    analyze_pairwise_chunked,
    build_easy_interpretation,
    classify_y_type,
    downcast_numeric_frame,
    estimate_pairwise_risk,
    load_config,
)
from data_loader import duplicate_key_count, merge_on_keys, profile_columns, read_single_file, summarize_dataframe
from report_builder import build_excel_report, build_html_report
from ui_components import (
    conclusion_card,
    detail_chart,
    factor_rank_panel,
    kpi_card,
    load_dark_css,
    page_header,
    result_table,
    save_result_plots,
    status_card,
)


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.environ.get("PFF_RUNTIME_DIR", str(BASE_DIR))).resolve()
CONFIG_PATH = RUNTIME_DIR / "config.yaml" if (RUNTIME_DIR / "config.yaml").exists() else BASE_DIR / "config.yaml"
DATA_DIR = RUNTIME_DIR / "data" if (RUNTIME_DIR / "data").exists() else BASE_DIR / "data"
OUTPUT_DIR = RUNTIME_DIR / "output"
UI_MAX_PREVIEW_ROWS = 100
UI_MAX_PREVIEW_COLS = 80
UI_MAX_PROFILE_COLS = 800


def _load_local_file(name: str) -> pd.DataFrame:
    """Load one local sample file if it exists."""
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return read_single_file(path)


def _format_pct(value: float) -> str:
    """Format a number as a one-decimal percentage."""
    return f"{value:.1f}%"


def _file_name(file_or_path) -> str:
    """Return a readable file name for an uploaded file or path."""
    return Path(getattr(file_or_path, "name", str(file_or_path))).name


def _preview_frame(df: pd.DataFrame, max_rows: int = UI_MAX_PREVIEW_ROWS, max_cols: int = UI_MAX_PREVIEW_COLS) -> pd.DataFrame:
    """Return a UI-safe preview that never renders every wide-data column."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.iloc[:max_rows, : min(max_cols, df.shape[1])]


def _sample_missing_pct(df: pd.DataFrame, max_rows: int = 1000, max_cols: int = 500) -> float:
    """Estimate missing rate from a bounded preview slice for fast KPI rendering."""
    if df is None or df.empty:
        return 0.0
    sample = df.iloc[: min(max_rows, len(df)), : min(max_cols, df.shape[1])]
    return float(sample.isna().mean().mean() * 100) if sample.size else 0.0


def _profile_source_for_ui(merged: pd.DataFrame, key_cols: list[str], y_cols: list[str]) -> pd.DataFrame:
    """Limit column profiling input for very wide data; analysis uses its own filters."""
    if merged.shape[1] <= UI_MAX_PROFILE_COLS:
        return merged
    keep = []
    for col in list(key_cols) + list(y_cols):
        if col in merged.columns and col not in keep:
            keep.append(col)
    for col in merged.columns:
        if col not in keep:
            keep.append(col)
        if len(keep) >= UI_MAX_PROFILE_COLS:
            break
    return merged.loc[:, keep]


def _init_state() -> None:
    """Initialize Streamlit session state values used across pages."""
    defaults = {
        "x_file_tables": [],
        "y_file_tables": [],
        "x_df": pd.DataFrame(),
        "y_df": pd.DataFrame(),
        "merged_df": pd.DataFrame(),
        "merge_stats": {},
        "x_internal_stats": {},
        "y_internal_stats": {},
        "column_profile": pd.DataFrame(),
        "factor_result": pd.DataFrame(),
        "pair_result_df": pd.DataFrame(),
        "pairwise_meta": {},
        "pairwise_excel_bytes": None,
        "pairwise_html_report": "",
        "analysis_y": "",
        "auto_load_sample": True,
        "uploader_version": 0,
        "merge_key_version": 0,
        "x_upload_signature": (),
        "y_upload_signature": (),
        "pending_widget_cleanup": (),
        "session_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_analysis_outputs() -> None:
    """Clear merged data, profiles, and factor analysis results."""
    st.session_state.merged_df = pd.DataFrame()
    st.session_state.merge_stats = {}
    st.session_state.column_profile = pd.DataFrame()
    st.session_state.factor_result = pd.DataFrame()
    st.session_state.pair_result_df = pd.DataFrame()
    st.session_state.pairwise_meta = {}
    st.session_state.pairwise_excel_bytes = None
    st.session_state.pairwise_html_report = ""
    st.session_state.analysis_y = ""


def _clear_session_data() -> None:
    """Clear uploaded data, prepared data, merge results, and widget selections."""
    st.session_state.x_file_tables = []
    st.session_state.y_file_tables = []
    st.session_state.x_df = pd.DataFrame()
    st.session_state.y_df = pd.DataFrame()
    st.session_state.x_internal_stats = {}
    st.session_state.y_internal_stats = {}
    _reset_analysis_outputs()

    st.session_state.auto_load_sample = False
    st.session_state.uploader_version += 1
    st.session_state.merge_key_version += 1
    st.session_state.x_upload_signature = ()
    st.session_state.y_upload_signature = ()
    st.session_state.pending_widget_cleanup = ("x_files_", "y_files_", "x_keys_", "y_keys_", "final_x_keys_", "final_y_keys_")
    st.session_state.session_message = "세션을 초기화했습니다. 업로드 데이터, Key 선택, 병합 결과, 분석 결과를 모두 비웠습니다."


def _run_pending_widget_cleanup() -> None:
    """Remove stale dynamic widget values before widgets are recreated."""
    prefixes = tuple(st.session_state.get("pending_widget_cleanup", ()))
    if not prefixes:
        return
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]
    st.session_state.pending_widget_cleanup = ()


def _upload_signature(files) -> tuple:
    """Create a small signature so reruns do not repeatedly reload files."""
    if not files:
        return ()
    return tuple((getattr(file, "name", ""), getattr(file, "size", None)) for file in files)


def _tables_from_files(files) -> list[dict]:
    """Read uploaded files and keep each file as a visible table."""
    tables = []
    for file in files or []:
        frame = read_single_file(file)
        tables.append({"name": _file_name(file), "df": frame})
    return tables


def _concat_tables(tables: list[dict]) -> pd.DataFrame:
    """Append file tables row-wise for preview and simple one-file workflows."""
    frames = []
    for item in tables:
        frame = item["df"].copy()
        frame["_source_file"] = item["name"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_default_samples() -> None:
    """Load all existing sample X/Y files on first launch."""
    x_tables = [{"name": path.name, "df": read_single_file(path)} for path in sorted(DATA_DIR.glob("sample_x*.csv"))]
    y_tables = [{"name": path.name, "df": read_single_file(path)} for path in sorted(DATA_DIR.glob("sample_y*.csv"))]
    if x_tables:
        st.session_state.x_file_tables = x_tables
        st.session_state.x_df = _concat_tables(x_tables)
    if y_tables:
        st.session_state.y_file_tables = y_tables
        st.session_state.y_df = _concat_tables(y_tables)


def _load_all_sample_files() -> None:
    """Load existing sample_x*.csv and sample_y*.csv files without regenerating them."""
    x_tables = [{"name": path.name, "df": read_single_file(path)} for path in sorted(DATA_DIR.glob("sample_x*.csv"))]
    y_tables = [{"name": path.name, "df": read_single_file(path)} for path in sorted(DATA_DIR.glob("sample_y*.csv"))]

    st.session_state.x_file_tables = x_tables
    st.session_state.y_file_tables = y_tables
    st.session_state.x_df = _concat_tables(x_tables)
    st.session_state.y_df = _concat_tables(y_tables)
    st.session_state.x_internal_stats = {}
    st.session_state.y_internal_stats = {}
    _reset_analysis_outputs()

    st.session_state.auto_load_sample = True
    st.session_state.uploader_version += 1
    st.session_state.merge_key_version += 1
    st.session_state.x_upload_signature = ()
    st.session_state.y_upload_signature = ()
    st.session_state.pending_widget_cleanup = ("x_files_", "y_files_", "x_keys_", "y_keys_", "final_x_keys_", "final_y_keys_")
    st.session_state.session_message = "기존 sample_x*.csv, sample_y*.csv 파일을 다시 불러왔습니다."


def _load_input_data(x_files, y_files) -> None:
    """Load uploaded data while preserving each file table for preview and merging."""
    if x_files:
        signature = _upload_signature(x_files)
        if signature != st.session_state.get("x_upload_signature", ()):
            st.session_state.x_file_tables = _tables_from_files(x_files)
            st.session_state.x_df = _concat_tables(st.session_state.x_file_tables)
            st.session_state.x_internal_stats = {}
            st.session_state.x_upload_signature = signature
            st.session_state.auto_load_sample = False
            _reset_analysis_outputs()

    if y_files:
        signature = _upload_signature(y_files)
        if signature != st.session_state.get("y_upload_signature", ()):
            st.session_state.y_file_tables = _tables_from_files(y_files)
            st.session_state.y_df = _concat_tables(st.session_state.y_file_tables)
            st.session_state.y_internal_stats = {}
            st.session_state.y_upload_signature = signature
            st.session_state.auto_load_sample = False
            _reset_analysis_outputs()


def _ensure_default_data() -> None:
    """Preload sample files unless the user intentionally cleared the session."""
    if not st.session_state.get("auto_load_sample", True):
        return
    if not st.session_state.x_file_tables and not st.session_state.y_file_tables:
        _load_default_samples()


def _common_columns(tables: list[dict]) -> list[str]:
    """Return columns that exist in every file table."""
    if not tables:
        return []
    common = set(tables[0]["df"].columns)
    for item in tables[1:]:
        common &= set(item["df"].columns)
    return [column for column in tables[0]["df"].columns if column in common]


def _default_key(columns: Iterable[str]) -> list[str]:
    """Prefer SSN as the default key when present."""
    cols = list(columns)
    if "SSN" in cols:
        return ["SSN"]
    return cols[:1]


def _is_system_column(column: str) -> bool:
    """Identify helper columns that should not be analyzed as X/Y factors."""
    return column.startswith("_source_file") or "source_file" in column


def _is_datetime_name(column: str) -> bool:
    """Identify date/time columns that are not meaningful Y targets by default."""
    lower = column.lower()
    return "date" in lower or "time" in lower


def _coalesce_duplicate_columns(merged: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Coalesce duplicate columns created by key-based file merges."""
    duplicate_cols = [column for column in merged.columns if column.endswith(suffix)]
    for dup_col in duplicate_cols:
        base_col = dup_col[: -len(suffix)]
        if base_col in merged.columns:
            merged[base_col] = merged[base_col].combine_first(merged[dup_col])
            merged = merged.drop(columns=[dup_col])
        else:
            merged = merged.rename(columns={dup_col: base_col})
    return merged


def _merge_file_tables_on_keys(tables: list[dict], key_cols: list[str], label: str) -> tuple[pd.DataFrame, dict]:
    """Merge multiple files of the same side by key before X/Y final merge."""
    if not tables:
        return pd.DataFrame(), {"file_count": 0, "rows_before": 0, "rows_after": 0, "duplicate_key_rows": 0, "key_cols": key_cols}
    if not key_cols:
        raise ValueError(f"{label} 파일 내부 병합 Key를 선택하세요.")

    rows_before = sum(len(item["df"]) for item in tables)
    duplicate_rows = 0
    for item in tables:
        missing = [column for column in key_cols if column not in item["df"].columns]
        if missing:
            raise KeyError(f"{item['name']}에 Key 컬럼이 없습니다: {missing}")
        duplicate_rows += duplicate_key_count(item["df"], key_cols)

    merged = tables[0]["df"].copy()
    for index, item in enumerate(tables[1:], start=2):
        suffix = f"__file{index}"
        merged = pd.merge(merged, item["df"].copy(), on=key_cols, how="outer", suffixes=("", suffix))
        merged = _coalesce_duplicate_columns(merged, suffix)

    stats = {
        "file_count": len(tables),
        "rows_before": int(rows_before),
        "rows_after": int(len(merged)),
        "duplicate_key_rows": int(duplicate_rows),
        "key_cols": key_cols,
    }
    return merged, stats


def _sidebar(config: dict) -> tuple[list, list, dict, str]:
    """Render the sidebar and return files, analysis options, and selected page."""
    st.sidebar.markdown(
        """
        <div class="pff-brand">
            <div class="pff-logo">PFF</div>
            <div>
                <div class="pff-brand-title">Process<br>Factor Finder</div>
                <div class="pff-brand-sub">공정 유의인자 탐색</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "메뉴",
        ["개요", "병합/프로파일", "유의인자 분석"],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="pff-side-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="pff-side-section">데이터 업로드 상태</div>', unsafe_allow_html=True)
    x_df = st.session_state.get("x_df", pd.DataFrame())
    y_df = st.session_state.get("y_df", pd.DataFrame())
    x_caption = f"{len(x_df):,} rows · {len(x_df.columns)} cols" if not x_df.empty else "파일 대기"
    y_caption = f"{len(y_df):,} rows · {len(y_df.columns)} cols" if not y_df.empty else "파일 대기"
    status_card("X 데이터", "loaded", x_caption, ok=not x_df.empty)
    status_card("Y 데이터", "loaded", y_caption, ok=not y_df.empty)

    with st.sidebar.expander("파일 업로드", expanded=False):
        x_files = st.file_uploader(
            "X 공정 데이터",
            type=["csv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True,
            key=f"x_files_{st.session_state.uploader_version}",
        )
        y_files = st.file_uploader(
            "Y 품질/특성 데이터",
            type=["csv", "xlsx", "xls", "parquet"],
            accept_multiple_files=True,
            key=f"y_files_{st.session_state.uploader_version}",
        )

    analysis_cfg = config.get("analysis", {})
    pair_cfg = config.get("pairwise_analysis", {})
    matrix_cfg = config.get("matrix_screening", {})
    options = {
        "max_missing_ratio": float(analysis_cfg.get("max_missing_ratio", 0.45)),
        "min_group_n": int(analysis_cfg.get("min_group_n", 15)),
        "top_n": int(config.get("top_n", 15)),
        "large_mode": True,
        "chunk_x_size": int(pair_cfg.get("chunk_x_size", 500)),
        "chunk_y_size": int(pair_cfg.get("chunk_y_size", 16)),
        "batch_flush_rows": int(pair_cfg.get("batch_flush_rows", 5000)),
        "save_intermediate_results": bool(pair_cfg.get("save_intermediate_results", True)),
        "downcast_float32": bool(pair_cfg.get("downcast_float32", False)),
        "screening_top_k_per_y": int(pair_cfg.get("screening_top_k_per_y", 8)),
        "screening_max_pairs_total": int(pair_cfg.get("screening_max_pairs_total", 120)),
        "auto_save_plots": True,
        "matrix_method": str(matrix_cfg.get("method", "pearson")).lower(),
        "matrix_top_n_per_y": int(matrix_cfg.get("top_n_per_y", 100)),
        "matrix_x_chunk_size": int(matrix_cfg.get("x_chunk_size", 5000)),
        "matrix_y_chunk_size": int(matrix_cfg.get("y_chunk_size", 50)),
        "matrix_dtype": str(matrix_cfg.get("dtype", "float32")),
        "matrix_missing_strategy": str(matrix_cfg.get("missing_strategy", "mean_impute")),
    }

    st.sidebar.markdown('<div class="pff-side-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="pff-side-section">설정</div>', unsafe_allow_html=True)
    with st.sidebar.expander("분석 설정 열기", expanded=False):
        st.caption("대용량 데이터 기준으로 기본 동작합니다. 화면에는 미리보기와 Top 결과만 보여주고, 전체 결과는 파일로 저장합니다.")
        options["max_missing_ratio"] = st.slider(
            "최대 결측률",
            min_value=0.05,
            max_value=0.90,
            value=options["max_missing_ratio"],
            step=0.05,
            help="이 비율보다 결측이 많은 컬럼은 분석 후보에서 제외합니다. 값이 높을수록 더 많은 컬럼을 살펴보지만 노이즈가 늘 수 있습니다.",
        )
        options["min_group_n"] = st.number_input(
            "최소 조건별 표본수",
            min_value=3,
            max_value=100,
            value=options["min_group_n"],
            step=1,
            help="범주형 조건, 위치, 레시피 등을 비교할 때 각 값에 필요한 최소 데이터 수입니다. 너무 작으면 우연한 차이가 커질 수 있습니다.",
        )
        options["top_n"] = st.number_input(
            "단일 Y Top N",
            min_value=5,
            max_value=50,
            value=options["top_n"],
            step=1,
            help="단일 Y 기준 분석 화면에서 상위 몇 개 인자를 우선 표시할지 정합니다.",
        )
        st.markdown("#### 전체 X-Y matrix 분석")
        st.caption("전체 X-Y 유의쌍 탐색은 numeric X와 numeric Y만 행렬 계산으로 먼저 압축합니다. 범주형 변수는 단일 Y 분석에서 확인하세요.")
        options["screening_top_k_per_y"] = st.number_input(
            "Y별 상세 계산 후보 수",
            min_value=3,
            max_value=200,
            value=options["screening_top_k_per_y"],
            step=1,
            help="matrix screening 후 각 Y마다 상세 점수 계산으로 넘길 X 후보 수입니다. 작을수록 빠르고, 클수록 놓칠 가능성이 줄어듭니다.",
        )
        options["screening_max_pairs_total"] = st.number_input(
            "전체 상세 계산 후보 상한",
            min_value=20,
            max_value=5000,
            value=options["screening_max_pairs_total"],
            step=10,
            help="matrix로 추린 후보 중 실제 p-value, effect, model score, stability를 계산할 최대 쌍 수입니다.",
        )
        method_options = ["pearson", "spearman"]
        options["matrix_method"] = st.selectbox(
            "Matrix screening 방식",
            method_options,
            index=method_options.index(options["matrix_method"]) if options["matrix_method"] in method_options else 0,
            format_func=lambda value: "Pearson 빠른 선형 관계" if value == "pearson" else "Spearman 순위 기반 관계",
            help="Pearson은 가장 빠릅니다. Spearman은 순위 기반이라 비선형 단조 패턴에 강하지만 rank 변환 때문에 더 오래 걸립니다.",
        )
        options["matrix_top_n_per_y"] = st.number_input(
            "Matrix 후보 Top N per Y",
            min_value=5,
            max_value=2000,
            value=options["matrix_top_n_per_y"],
            step=5,
            help="행렬 계산 단계에서 각 Y별로 유지할 X 후보 수입니다. 이 값은 1차 후보 저장용이고, 상세 계산 후보 수는 위 설정으로 다시 제한됩니다.",
        )
        options["matrix_x_chunk_size"] = st.number_input(
            "Matrix X chunk size",
            min_value=100,
            max_value=50000,
            value=options["matrix_x_chunk_size"],
            step=100,
            help="한 번에 행렬 계산할 X 컬럼 수입니다. 메모리 부족이 나면 줄이고, 여유가 있으면 키워 속도를 높일 수 있습니다.",
        )
        options["matrix_y_chunk_size"] = st.number_input(
            "Matrix Y chunk size",
            min_value=1,
            max_value=1000,
            value=options["matrix_y_chunk_size"],
            step=1,
            help="한 번에 행렬 계산할 Y 컬럼 수입니다. Y가 많거나 메모리가 부족하면 줄이세요.",
        )
        dtype_options = ["float32", "float64"]
        options["matrix_dtype"] = st.selectbox(
            "Matrix dtype",
            dtype_options,
            index=dtype_options.index(options["matrix_dtype"]) if options["matrix_dtype"] in dtype_options else 0,
            help="float32는 메모리를 덜 쓰고 빠릅니다. 아주 미세한 수치 차이를 보고 싶을 때만 float64를 사용하세요.",
        )
        missing_options = ["mean_impute", "median_impute", "drop_rows"]
        options["matrix_missing_strategy"] = st.selectbox(
            "Matrix 결측 처리",
            missing_options,
            index=missing_options.index(options["matrix_missing_strategy"]) if options["matrix_missing_strategy"] in missing_options else 0,
            format_func=lambda value: {"mean_impute": "평균 대체", "median_impute": "중앙값 대체", "drop_rows": "결측 행 제외"}[value],
            help="행렬 계산 전 결측을 처리하는 방식입니다. 기본 평균 대체가 가장 빠르고 안정적입니다.",
        )
        st.markdown("#### 저장/성능")
        options["chunk_x_size"] = st.number_input(
            "상세 계산 X chunk",
            min_value=50,
            max_value=5000,
            value=options["chunk_x_size"],
            step=50,
            help="matrix 후보에 대해 상세 점수를 계산할 때 나누어 처리할 X 단위입니다.",
        )
        options["chunk_y_size"] = st.number_input(
            "상세 계산 Y chunk",
            min_value=1,
            max_value=256,
            value=options["chunk_y_size"],
            step=1,
            help="상세 점수 계산을 Y 기준으로 나누는 단위입니다.",
        )
        options["batch_flush_rows"] = st.number_input(
            "중간 저장 batch row",
            min_value=500,
            max_value=50000,
            value=options["batch_flush_rows"],
            step=500,
            help="분석 중 결과를 파일에 나누어 저장하는 단위입니다. 앱이 멈추더라도 중간 결과 확인에 도움이 됩니다.",
        )
        options["save_intermediate_results"] = st.toggle(
            "중간 결과 저장",
            value=options["save_intermediate_results"],
            help="대용량 분석에서는 켜두는 것을 권장합니다. 전체 결과, 오류 로그, 제외 컬럼 목록을 output/results에 저장합니다.",
        )
        options["downcast_float32"] = st.toggle(
            "numeric float32 downcast",
            value=options["downcast_float32"],
            help="숫자 컬럼을 float32로 낮춰 메모리 사용량을 줄입니다. 일반적인 탐색 목적에서는 권장됩니다.",
        )
        options["auto_save_plots"] = st.toggle(
            "그래프 자동 저장",
            value=options["auto_save_plots"],
            help="분석 완료 후 상위 후보 그래프를 output/plots 폴더에 HTML, 가능하면 PNG로 저장합니다.",
        )

    st.sidebar.markdown('<div class="pff-side-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="pff-side-section">세션 관리</div>', unsafe_allow_html=True)
    if st.sidebar.button("세션 클리어", use_container_width=True):
        _clear_session_data()
        st.rerun()
    if st.sidebar.button("샘플 다시 불러오기", use_container_width=True):
        _load_all_sample_files()
        st.rerun()
    if st.session_state.get("session_message"):
        st.sidebar.success(st.session_state.session_message)

    return x_files, y_files, options, page


def _file_preview_section(label: str, tables: list[dict]) -> None:
    """Show every uploaded file as its own visual table."""
    st.markdown(f'<div class="pff-section-title">{label} 파일별 미리보기</div>', unsafe_allow_html=True)
    if not tables:
        st.info(f"{label} 파일이 아직 없습니다.")
        return

    for item in tables:
        frame = item["df"]
        with st.container(border=True):
            cols = st.columns([1.4, 0.55, 0.55, 0.55])
            with cols[0]:
                st.markdown(f"**{item['name']}**")
                st.caption("각 파일은 아래에서 독립적으로 확인한 뒤 Key 기준으로 내부 병합할 수 있습니다.")
            with cols[1]:
                st.metric("Rows", f"{len(frame):,}")
            with cols[2]:
                st.metric("Columns", f"{len(frame.columns):,}")
            with cols[3]:
                missing = _sample_missing_pct(frame)
                st.metric("Missing", _format_pct(missing))
            preview = _preview_frame(frame)
            if frame.shape[1] > preview.shape[1]:
                st.caption(f"미리보기는 처음 {preview.shape[1]:,}개 컬럼만 표시합니다. 전체 컬럼은 분석/저장에는 그대로 사용됩니다.")
            st.dataframe(preview, use_container_width=True, hide_index=True)


def _overview_page() -> None:
    """Render the first page with data status and per-file previews."""
    page_header("개요", "업로드된 X/Y 파일을 먼저 눈으로 확인합니다. 여러 파일은 파일별 테이블로 따로 보여줍니다.")
    x_df = st.session_state.x_df
    y_df = st.session_state.y_df
    x_summary = summarize_dataframe(x_df)
    y_summary = summarize_dataframe(y_df)

    cols = st.columns(4)
    with cols[0]:
        kpi_card("X row 수", f"{x_summary['rows']:,}", f"{len(st.session_state.x_file_tables)} files", "#2563eb", "X")
    with cols[1]:
        kpi_card("Y row 수", f"{y_summary['rows']:,}", f"{len(st.session_state.y_file_tables)} files", "#9333ea", "Y")
    with cols[2]:
        kpi_card("X 결측률", _format_pct(_sample_missing_pct(x_df)), "preview 기준", "#22d3ee", "%")
    with cols[3]:
        kpi_card("Y 결측률", _format_pct(_sample_missing_pct(y_df)), "preview 기준", "#14b8a6", "%")

    left, right = st.columns(2)
    with left:
        _file_preview_section("X 데이터", st.session_state.x_file_tables)
    with right:
        _file_preview_section("Y 데이터", st.session_state.y_file_tables)


def _merge_profile_page(config: dict, options: dict) -> None:
    """Render X-file merge, Y-file merge, final X/Y merge, and profiling."""
    page_header("병합/프로파일", "1단계로 X 파일끼리, Y 파일끼리 Key 기준 병합한 뒤 2단계로 X/Y를 최종 병합합니다.")
    if not st.session_state.x_file_tables or not st.session_state.y_file_tables:
        st.warning("먼저 X/Y 데이터를 업로드하세요. 샘플 파일이 있으면 개요 화면에 자동으로 표시됩니다.")
        return

    st.markdown('<div class="pff-section-title">1단계 · 파일 내부 병합</div>', unsafe_allow_html=True)
    x_common = _common_columns(st.session_state.x_file_tables)
    y_common = _common_columns(st.session_state.y_file_tables)
    key_cols = st.columns([1, 1, 0.8])
    with key_cols[0]:
        x_internal_keys = st.multiselect(
            "X 파일 간 병합 Key",
            x_common,
            default=_default_key(x_common),
            key=f"x_keys_{st.session_state.merge_key_version}",
            help="여러 X 파일을 먼저 하나로 합칠 때 기준이 되는 Key 컬럼입니다.",
        )
    with key_cols[1]:
        y_internal_keys = st.multiselect(
            "Y 파일 간 병합 Key",
            y_common,
            default=_default_key(y_common),
            key=f"y_keys_{st.session_state.merge_key_version}",
            help="여러 Y 파일을 먼저 하나로 합칠 때 기준이 되는 Key 컬럼입니다.",
        )
    with key_cols[2]:
        st.write("")
        st.write("")
        internal_clicked = st.button("파일 내부 병합 준비", type="primary", use_container_width=True)

    if internal_clicked:
        try:
            st.session_state.x_df, st.session_state.x_internal_stats = _merge_file_tables_on_keys(
                st.session_state.x_file_tables,
                x_internal_keys,
                "X",
            )
            st.session_state.y_df, st.session_state.y_internal_stats = _merge_file_tables_on_keys(
                st.session_state.y_file_tables,
                y_internal_keys,
                "Y",
            )
            _reset_analysis_outputs()
            st.success("X 파일 내부 병합과 Y 파일 내부 병합이 완료되었습니다.")
        except Exception as exc:
            st.error(f"파일 내부 병합 실패: {exc}")

    if st.session_state.x_internal_stats or st.session_state.y_internal_stats:
        x_stats = st.session_state.x_internal_stats
        y_stats = st.session_state.y_internal_stats
        cols = st.columns(4)
        with cols[0]:
            kpi_card("X 내부 병합", f"{x_stats.get('rows_after', len(st.session_state.x_df)):,}", f"{x_stats.get('file_count', 1)} files", "#2563eb", "X")
        with cols[1]:
            kpi_card("Y 내부 병합", f"{y_stats.get('rows_after', len(st.session_state.y_df)):,}", f"{y_stats.get('file_count', 1)} files", "#9333ea", "Y")
        with cols[2]:
            kpi_card("X 중복 Key", f"{x_stats.get('duplicate_key_rows', 0):,}", "파일 내부", "#f59e0b", "D")
        with cols[3]:
            kpi_card("Y 중복 Key", f"{y_stats.get('duplicate_key_rows', 0):,}", "파일 내부", "#f59e0b", "D")

    st.markdown('<div class="pff-section-title">2단계 · X/Y 최종 병합</div>', unsafe_allow_html=True)
    x_df = st.session_state.x_df
    y_df = st.session_state.y_df
    final_cols = st.columns([1, 1, 0.8])
    with final_cols[0]:
        final_x_keys = st.multiselect(
            "최종 X Key",
            x_df.columns.tolist(),
            default=_default_key(x_df.columns),
            key=f"final_x_keys_{st.session_state.merge_key_version}",
            help="X 데이터와 Y 데이터를 최종 병합할 때 X 쪽에서 사용할 Key입니다.",
        )
    with final_cols[1]:
        final_y_keys = st.multiselect(
            "최종 Y Key",
            y_df.columns.tolist(),
            default=_default_key(y_df.columns),
            key=f"final_y_keys_{st.session_state.merge_key_version}",
            help="X 데이터와 Y 데이터를 최종 병합할 때 Y 쪽에서 사용할 Key입니다.",
        )
    with final_cols[2]:
        st.write("")
        st.write("")
        merge_clicked = st.button("X/Y 최종 병합", type="primary", use_container_width=True)

    if merge_clicked:
        try:
            merged, stats = merge_on_keys(x_df, y_df, final_x_keys, final_y_keys)
            profile_source = _profile_source_for_ui(merged, final_x_keys, y_df.columns.tolist())
            profile = profile_columns(
                profile_source,
                key_cols=final_x_keys,
                y_cols=y_df.columns.tolist(),
                max_missing_ratio=options["max_missing_ratio"],
            )
            st.session_state.merged_df = merged
            st.session_state.merge_stats = stats
            st.session_state.column_profile = profile
            st.session_state.factor_result = pd.DataFrame()
            st.session_state.analysis_y = ""
            st.success("X/Y 최종 병합이 완료되었습니다.")
        except Exception as exc:
            st.error(f"X/Y 최종 병합 실패: {exc}")

    stats = st.session_state.merge_stats
    if stats:
        cols = st.columns(6)
        with cols[0]:
            kpi_card("X Rows", f"{stats['x_rows']:,}", "최종 X", "#2563eb", "X")
        with cols[1]:
            kpi_card("Y Rows", f"{stats['y_rows']:,}", "최종 Y", "#9333ea", "Y")
        with cols[2]:
            kpi_card("Merged", f"{stats['merged_rows']:,}", "병합 row", "#22d3ee", "M")
        with cols[3]:
            kpi_card("Y Match", _format_pct(stats["y_merge_rate"]), "Y 기준", "#14b8a6", "%")
        with cols[4]:
            kpi_card("X Match", _format_pct(stats["x_merge_rate"]), "X 기준", "#06b6d4", "%")
        with cols[5]:
            kpi_card("Dup Key", f"{stats['duplicate_key_rows']:,}", "중복 Key row", "#f59e0b", "D")

    if not st.session_state.merged_df.empty:
        st.markdown('<div class="pff-section-title">병합 데이터 미리보기</div>', unsafe_allow_html=True)
        merged_preview = _preview_frame(st.session_state.merged_df)
        if st.session_state.merged_df.shape[1] > merged_preview.shape[1]:
            st.caption(f"대용량 UI 보호를 위해 처음 {merged_preview.shape[1]:,}개 컬럼만 표시합니다.")
        st.dataframe(merged_preview, use_container_width=True, hide_index=True)

    if not st.session_state.column_profile.empty:
        st.markdown('<div class="pff-section-title">컬럼 프로파일</div>', unsafe_allow_html=True)
        if st.session_state.merged_df.shape[1] > UI_MAX_PROFILE_COLS:
            st.caption(f"컬럼 프로파일은 UI 렉 방지를 위해 {UI_MAX_PROFILE_COLS:,}개 컬럼까지만 표시합니다. 분석 후보 필터링은 별도 로직으로 전체 컬럼을 봅니다.")
        st.dataframe(st.session_state.column_profile.head(UI_MAX_PROFILE_COLS), use_container_width=True, hide_index=True)


def _format_p_value(value: object) -> str:
    """Format p-values compactly for UI cards."""
    try:
        numeric = float(value)
    except Exception:
        return "-"
    if pd.isna(numeric):
        return "-"
    if numeric < 1e-15:
        return "< 1e-15"
    if numeric < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.4f}"


def _format_p_value_table(value: object) -> str:
    """Format p-values for result tables using scientific notation."""
    try:
        numeric = float(value)
    except Exception:
        return "-"
    if pd.isna(numeric):
        return "-"
    if numeric == 0:
        return "< 1e-300"
    if numeric < 1e-15:
        return "< 1e-15"
    return f"{numeric:.2e}"


def _timestamp_token() -> str:
    """Return a compact timestamp token for output folders."""
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(text: str) -> str:
    """Create a filesystem-safe fragment."""
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in str(text))[:120]


def _prepare_output_dir(base_name: str) -> Path:
    """Create one timestamped output folder inside output/."""
    path = OUTPUT_DIR / base_name / _timestamp_token()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _risk_caption(risk_level: str) -> str:
    """Translate pair risk into a short Korean label."""
    mapping = {
        "low": "low · 가벼움",
        "medium": "medium · 주의",
        "high": "high · 오래 걸릴 수 있음",
        "very high": "very high · 파일 저장 중심 권장",
    }
    return mapping.get(risk_level, risk_level)


def _resolve_column_selection(
    label: str,
    all_columns: list[str],
    default_columns: list[str],
    key_prefix: str,
    large_mode: bool,
) -> list[str]:
    """Avoid rendering very large multiselect widgets by using filters and paste-in selection."""
    use_all = st.checkbox(f"{label} 전체 컬럼 사용", value=True, key=f"{key_prefix}_use_all")
    if use_all:
        st.caption(f"{label} 전체 {len(default_columns):,}개 후보를 사용합니다.")
        with st.expander(f"{label} 후보 미리보기", expanded=False):
            preview = default_columns[:200]
            st.write(", ".join(preview) if preview else "표시할 컬럼이 없습니다.")
            if len(default_columns) > 200:
                st.caption(f"미리보기는 처음 200개만 표시합니다. 전체 수: {len(default_columns):,}")
        return default_columns

    include_pattern = st.text_input(f"{label} include regex", value="", key=f"{key_prefix}_include")
    exclude_pattern = st.text_input(f"{label} exclude regex", value="", key=f"{key_prefix}_exclude")
    pasted = st.text_area(
        f"{label} 직접 입력",
        value="",
        help="쉼표, 줄바꿈, 공백으로 컬럼명을 넣을 수 있습니다.",
        key=f"{key_prefix}_paste",
        height=100 if large_mode else 80,
    )

    selected = list(default_columns)
    if include_pattern:
        try:
            selected = [col for col in selected if pd.Series([col]).str.contains(include_pattern, regex=True, case=False).iloc[0]]
        except Exception:
            st.warning(f"{label} include regex가 올바르지 않습니다.")
    if exclude_pattern:
        try:
            selected = [col for col in selected if not pd.Series([col]).str.contains(exclude_pattern, regex=True, case=False).iloc[0]]
        except Exception:
            st.warning(f"{label} exclude regex가 올바르지 않습니다.")

    pasted_cols = [item.strip() for chunk in pasted.splitlines() for item in chunk.replace(",", " ").split() if item.strip()]
    if pasted_cols:
        selected = [col for col in selected if col in pasted_cols]

    with st.expander(f"{label} 선택 결과 미리보기", expanded=False):
        preview = selected[:200]
        st.write(", ".join(preview) if preview else "선택된 컬럼이 없습니다.")
        if len(selected) > 200:
            st.caption(f"미리보기는 처음 200개만 표시합니다. 선택 수: {len(selected):,}")
    return selected


def _lightweight_exclude_columns(all_columns: list[str], source_cols: list[str], key_prefix: str) -> list[str]:
    """Collect optional exclude columns without rendering a huge multiselect."""
    with st.expander("추가 제외 컬럼 설정", expanded=False):
        st.caption("대용량 데이터에서는 전체 컬럼 목록을 화면에 펼치지 않습니다. 필요한 컬럼명만 붙여넣거나 regex로 제외하세요.")
        pasted = st.text_area(
            "제외할 컬럼명 붙여넣기",
            value="",
            key=f"{key_prefix}_exclude_paste",
            help="쉼표, 줄바꿈, 공백으로 여러 컬럼명을 입력할 수 있습니다.",
            height=90,
        )
        regex = st.text_input(
            "제외 regex",
            value="",
            key=f"{key_prefix}_exclude_regex",
            help="예: ^ID_|_raw$ 처럼 패턴에 맞는 컬럼을 제외합니다.",
        )
        preview = [col for col in all_columns if col not in source_cols][:200]
        st.caption(f"컬럼 미리보기는 처음 {len(preview):,}개만 표시합니다.")
        st.write(", ".join(preview) if preview else "표시할 컬럼이 없습니다.")

    selected = set(source_cols)
    pasted_cols = {item.strip() for chunk in pasted.splitlines() for item in chunk.replace(",", " ").split() if item.strip()}
    selected.update(col for col in pasted_cols if col in all_columns)
    if regex:
        try:
            matched = pd.Index(all_columns).to_series().str.contains(regex, regex=True, case=False, na=False)
            selected.update(matched[matched].index.tolist())
        except Exception:
            st.warning("제외 regex가 올바르지 않습니다.")
    return list(selected)


def _numeric_matrix_candidates(
    merged_df: pd.DataFrame,
    y_df: pd.DataFrame,
    stats: dict,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return lightweight candidate lists for matrix-only pairwise analysis."""
    source_cols = [col for col in merged_df.columns if _is_system_column(col)]
    key_cols = set(stats.get("x_keys", [])) | set(stats.get("y_keys", []))
    y_source_cols = {col for col in y_df.columns if col in merged_df.columns}

    numeric_cols = [
        col
        for col, dtype in merged_df.dtypes.items()
        if pd.api.types.is_numeric_dtype(dtype)
        and col not in key_cols
        and col not in source_cols
        and not _is_datetime_name(col)
    ]
    x_default = [col for col in numeric_cols if col not in y_source_cols]
    y_default = [
        col
        for col in numeric_cols
        if col in y_df.columns
    ]
    x_choices = [col for col in merged_df.columns if col not in source_cols and col not in y_source_cols]
    y_choices = [col for col in y_df.columns if not _is_system_column(col)]
    return x_choices, x_default, y_choices, y_default


def _save_large_analysis_outputs(result_df: pd.DataFrame, output_dir: Path) -> dict:
    """Persist detailed result summaries for large analysis mode."""
    saved = {}
    if result_df is None or result_df.empty:
        return saved

    result_df = result_df.copy()
    result_df["adjusted_p_value"] = result_df.get("p_value_adj")
    result_df["x_col"] = result_df["x_feature"]
    result_df["y_col"] = result_df["y_target"]
    result_df["n_samples"] = result_df["sample_n"]
    result_df["missing_rate_x"] = result_df["missing_ratio_x"]
    result_df["missing_rate_y"] = result_df["missing_ratio_y"]

    csv_path = output_dir / "pairwise_detailed_result.csv"
    parquet_path = output_dir / "pairwise_detailed_result.parquet"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    result_df.to_parquet(parquet_path, index=False)
    saved["detailed_csv"] = str(csv_path)
    saved["detailed_parquet"] = str(parquet_path)

    y_top = result_df.sort_values(["y_target", "final_score"], ascending=[True, False]).groupby("y_target", as_index=False, group_keys=False).head(20)
    y_top_path = output_dir / "top_results_by_y.csv"
    y_top.to_csv(y_top_path, index=False, encoding="utf-8-sig")
    saved["top_by_y"] = str(y_top_path)

    x_repeat = result_df.groupby("x_feature").size().reset_index(name="significant_repeat_count").sort_values("significant_repeat_count", ascending=False)
    x_repeat_path = output_dir / "x_feature_repeat_summary.csv"
    x_repeat.to_csv(x_repeat_path, index=False, encoding="utf-8-sig")
    saved["x_repeat_summary"] = str(x_repeat_path)
    return saved


def _chart_input_frame(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    large_mode: bool,
    max_rows: int = 5000,
) -> pd.DataFrame:
    """Return only selected chart columns, with deterministic sampling in large mode."""
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame(columns=[x_col, y_col])
    chart_df = df.loc[:, [x_col, y_col]]
    if large_mode and len(chart_df) > max_rows:
        chart_df = chart_df.sample(n=max_rows, random_state=42)
    return chart_df


def _axis_category_count(df: pd.DataFrame, x_col: str, y_col: str, x_type: str, y_type: str, chart_mode: str) -> int:
    """Estimate how many x-axis categories a detail chart will show."""
    if x_col not in df.columns or y_col not in df.columns:
        return 0
    pair = df[[x_col, y_col]].dropna()
    if pair.empty:
        return 0
    if chart_mode == "bar" and pd.api.types.is_numeric_dtype(pair[x_col]) and pair[x_col].nunique() > 20:
        return min(8, pair[x_col].nunique())
    if chart_mode == "box" and x_type == "numeric" and y_type == "numeric":
        return min(5, pair[x_col].nunique())
    if x_type == "categorical" or chart_mode == "bar":
        return int(pair[x_col].nunique())
    if x_type == "numeric" and y_type in {"binary", "categorical"}:
        return int(pair[y_col].nunique())
    return 0


def _plot_options(config: dict, key_prefix: str, category_count: int = 0) -> tuple[bool, int]:
    """Render shared plot options and warn when many categories are shown."""
    plot_cfg = config.get("plot", {})
    show_all = st.checkbox(
        "전체 범주 표시",
        value=bool(plot_cfg.get("show_all_categories_default", True)),
        key=f"{key_prefix}_show_all_categories",
    )
    tick_angle = st.select_slider(
        "X축 라벨 각도",
        options=[0, -30, -45, -60, -75],
        value=int(plot_cfg.get("default_tick_angle", -45)),
        key=f"{key_prefix}_tick_angle",
    )
    warning_limit = int(plot_cfg.get("max_categories_without_warning", 30))
    if show_all and category_count > warning_limit:
        st.warning("범주 수가 많아 그래프가 복잡할 수 있습니다. 필요 시 필터로 확인 범위를 줄여보세요.")
    return show_all, int(tick_angle)


def _detail_metric_grid(items: list[tuple[str, str]]) -> None:
    """Render compact metric chips with native Streamlit elements."""
    for start in range(0, len(items), 5):
        cols = st.columns(5)
        for col, (label, value) in zip(cols, items[start : start + 5]):
            with col:
                with st.container(border=True):
                    st.caption(str(label))
                    st.markdown(f"**{value}**")


def _selected_result_detail(row: pd.Series, y_col: str, x_col: str, pair_mode: bool = False) -> None:
    """Show a detailed explanation for a selected factor or pair."""
    title = f"{x_col} → {y_col}" if pair_mode else f"{x_col} 상세 설명"
    easy_text = row.get("easy_interpretation")
    if not easy_text:
        easy_text = build_easy_interpretation(x_col, y_col, row, pair_mode=pair_mode)

    with st.container(border=True):
        st.markdown(f"#### 선택한 유의 후보: {title}")
        p_adj = row.get("p_value_adj", None)
        if pair_mode:
            missing_text = f"X {float(row.get('missing_ratio_x', 0)) * 100:.1f}% / Y {float(row.get('missing_ratio_y', 0)) * 100:.1f}%"
        else:
            missing_text = f"{float(row.get('missing_ratio', 0)) * 100:.1f}%"
        _detail_metric_grid(
            [
                ("판정", str(row.get("judgement", "-"))),
                ("최종 점수", f"{float(row.get('final_score', 0)):.1f}"),
                ("p-value", _format_p_value(row.get("p_value"))),
                ("보정 p-value", _format_p_value(p_adj) if p_adj is not None else "-"),
                ("표본 수", f"{int(row.get('sample_n', 0)):,}"),
                ("effect size", f"{float(row.get('effect_size', 0)):.4f}"),
                ("model score", f"{float(row.get('model_score', 0)):.1f}"),
                ("stability", f"{float(row.get('stability_score', 0)):.1f}"),
                ("quality", f"{float(row.get('quality_score', 0)):.1f}"),
                ("결측률", missing_text),
            ]
        )

        st.markdown("**관찰된 방향성**")
        st.write(row.get("direction", "-"))
        st.markdown("**쉽게 해석한 설명**")
        st.write(easy_text)
        st.markdown("**주의사항**")
        st.info(str(row.get("caution_text", "확정 판단이 아니라 우선 확인 후보로 검토하세요.")))


def _top_pair_bar_chart(pair_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Create a Top pair horizontal bar chart."""
    plot_df = pair_df.head(top_n).copy()
    plot_df["pair_name"] = plot_df["x_feature"].astype(str) + " → " + plot_df["y_target"].astype(str)
    plot_df = plot_df.sort_values("final_score", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=plot_df["final_score"],
            y=plot_df["pair_name"],
            orientation="h",
            marker_color="#8b5cf6",
            text=plot_df["final_score"].round(1),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>점수=%{x:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(460, 28 * len(plot_df) + 150),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,27,0.60)",
        margin={"l": 30, "r": 36, "t": 34, "b": 40},
        xaxis_title="Final Score",
        yaxis_title="",
    )
    fig.update_xaxes(range=[0, 105], gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def _single_y_analysis_section(config: dict, options: dict, merged_df: pd.DataFrame, y_df: pd.DataFrame, stats: dict) -> None:
    """Render the existing single-Y workflow plus selected factor details."""
    y_candidates = [
        col
        for col in y_df.columns
        if col in merged_df.columns
        and col not in stats.get("y_keys", [])
        and not _is_system_column(col)
        and not _is_datetime_name(col)
    ]
    if not y_candidates:
        st.warning("분석할 Y 컬럼이 없습니다. 최종 병합 Key와 Y 파일 구성을 확인하세요.")
        return

    control_cols = st.columns([1, 1, 0.75])
    with control_cols[0]:
        default_idx = y_candidates.index("DPU") if "DPU" in y_candidates else 0
        y_col = st.selectbox("분석 대상 Y", y_candidates, index=default_idx)
    with control_cols[1]:
        source_cols = [col for col in merged_df.columns if _is_system_column(col)]
        exclude_cols = _lightweight_exclude_columns(merged_df.columns.tolist(), source_cols, "single_y")
    with control_cols[2]:
        st.write("")
        st.write("")
        run_clicked = st.button("단일 Y 유의인자 탐색 실행", type="primary", use_container_width=True)

    if run_clicked:
        run_config = config.copy()
        run_config["analysis"] = dict(run_config.get("analysis", {}))
        run_config["analysis"]["max_missing_ratio"] = options["max_missing_ratio"]
        run_config["analysis"]["min_group_n"] = options["min_group_n"]
        try:
            result = analyze_factors(
                merged_df,
                y_col=y_col,
                key_cols=stats.get("x_keys", []),
                y_cols=y_df.columns.tolist(),
                exclude_cols=exclude_cols,
                config=run_config,
            )
            if not result.empty:
                result["easy_interpretation"] = result.apply(
                    lambda row: build_easy_interpretation(row["feature"], y_col, row, pair_mode=False),
                    axis=1,
                )
            st.session_state.factor_result = result
            st.session_state.analysis_y = y_col
            st.success("유의인자 탐색이 완료되었습니다.")
        except Exception as exc:
            st.error(f"분석 실패: {exc}")

    result = st.session_state.factor_result
    if result.empty:
        return

    left, right = st.columns([1.05, 1.05])
    with left:
        factor_rank_panel(result, int(options["top_n"]))
    with right:
        selected_feature = st.selectbox("상세 설명을 볼 인자 선택", result["feature"].tolist(), key="single_selected_feature")
        selected_row = result.loc[result["feature"] == selected_feature].iloc[0]
        _selected_result_detail(selected_row, st.session_state.analysis_y, selected_feature, pair_mode=False)

    bottom_left, bottom_right = st.columns([1.05, 1.05])
    with bottom_left:
        st.markdown('<div class="pff-section-title">결과 테이블</div>', unsafe_allow_html=True)
        result_table(result)
    with bottom_right:
        st.markdown('<div class="pff-section-title">선택 인자 상세 그래프</div>', unsafe_allow_html=True)
        show_chart = st.toggle("상세 그래프 표시", value=False, key="single_show_chart")
        st.caption("그래프는 그릴 때 데이터 샘플링과 Plotly 렌더링이 들어갑니다. 화면 조작을 빠르게 하려면 필요할 때만 켜세요.")
        if show_chart:
            chart_label = st.selectbox("시각화 방식", ["자동 추천", "산점도", "박스플롯", "비율 막대"], index=0, key="single_chart_mode")
            chart_mode = {"자동 추천": "auto", "산점도": "scatter", "박스플롯": "box", "비율 막대": "bar"}[chart_label]
            feature_type = selected_row["feature_type"]
            y_type = classify_y_type(merged_df[st.session_state.analysis_y])
            large_mode = True
            chart_df = _chart_input_frame(merged_df, selected_feature, st.session_state.analysis_y, large_mode)
            category_count = _axis_category_count(chart_df, selected_feature, st.session_state.analysis_y, feature_type, y_type, chart_mode)
            show_all, tick_angle = _plot_options(config, "single", category_count)
            fig = detail_chart(
                chart_df,
                selected_feature,
                st.session_state.analysis_y,
                feature_type,
                y_type,
                chart_mode=chart_mode,
                show_all_categories=show_all,
                tick_angle=tick_angle,
                plot_config=config.get("plot", {}),
            )
            st.plotly_chart(fig, use_container_width=True)


def _pairwise_analysis_section(config: dict, options: dict, merged_df: pd.DataFrame, y_df: pd.DataFrame, stats: dict) -> None:
    """Render the all X-Y pairwise analysis workflow."""
    st.info(
        "전체 X-Y 유의쌍 탐색은 numeric X와 numeric Y를 행렬 계산으로 빠르게 훑은 뒤, "
        "상위 후보만 상세 점수로 다시 확인합니다. 결과는 확정 판단이 아니라 우선 확인할 후보 쌍입니다."
    )

    large_mode = True
    source_cols = [col for col in merged_df.columns if _is_system_column(col)]
    x_choices, x_default, y_choices, y_default = _numeric_matrix_candidates(merged_df, y_df, stats)

    selector_cols = st.columns(2)
    with selector_cols[0]:
        x_cols = _resolve_column_selection("X 후보", x_choices, x_default, "pair_x", large_mode)
    with selector_cols[1]:
        y_cols = _resolve_column_selection("Y 후보", y_choices, y_default, "pair_y", large_mode)

    pair_count = sum(1 for x_col in x_cols for y_col in y_cols if x_col != y_col)
    matrix_x_cols = [col for col in x_cols if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col])]
    matrix_y_cols = [col for col in y_cols if col in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[col])]
    numeric_x_count = len(matrix_x_cols)
    numeric_y_count = len(matrix_y_cols)
    numeric_pair_count = sum(
        1
        for x_col in matrix_x_cols
        for y_col in matrix_y_cols
        if x_col != y_col
    )
    pair_cfg = config.get("pairwise_analysis", {})
    matrix_cfg = config.get("matrix_screening", {})
    risk_level = estimate_pairwise_risk(numeric_pair_count)
    expected_fine = min(
        numeric_pair_count,
        int(options.get("screening_top_k_per_y", pair_cfg.get("screening_top_k_per_y", 8))) * max(numeric_y_count, 1),
        int(options.get("screening_max_pairs_total", pair_cfg.get("screening_max_pairs_total", 120))),
    )
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("선택한 전체 조합", f"{pair_count:,}개")
    with metric_cols[1]:
        st.metric("Matrix 분석 조합", f"{numeric_pair_count:,}개")
    with metric_cols[2]:
        st.metric("상세 계산 후보", f"최대 {expected_fine:,}개")
    st.caption(f"예상 소요 위험도: {_risk_caption(risk_level)}")
    if pair_count > numeric_pair_count:
        st.caption(
            f"전체 X-Y 탐색은 속도를 위해 numeric-numeric 조합만 matrix로 분석합니다. "
            f"문자/범주형이 섞인 {pair_count - numeric_pair_count:,}개 조합은 이 모드에서 건너뜁니다."
        )
    if str(options.get("matrix_method", "pearson")).lower() == "spearman":
        st.warning("Spearman은 rank 변환이 필요해서 Pearson보다 느릴 수 있습니다.")

    if numeric_pair_count > int(pair_cfg.get("max_pairs_warning", 1000)):
        st.warning("Matrix 분석 조합 수가 많아 시간이 길어질 수 있습니다. 필요한 후보 컬럼만 선택하면 더 빠릅니다.")
    if risk_level == "very high":
        st.warning("very high 규모입니다. 화면 렌더링보다 파일 저장 중심으로 동작합니다.")
    st.info("대용량 기준으로 항상 실행합니다. 원본 미리보기는 제한하고, chunk 분석과 중간 저장을 중심으로 처리합니다.")

    run_pairwise = st.button("전체 X-Y 유의쌍 탐색 실행", type="primary", use_container_width=True)
    if run_pairwise:
        if not matrix_x_cols or not matrix_y_cols:
            st.warning("matrix로 분석할 numeric X 또는 numeric Y가 없습니다. 숫자형 후보 컬럼을 확인하세요.")
            return

        progress = st.progress(0)
        status = st.empty()

        def _progress_callback(
            done: int,
            total: int,
            x_col: str,
            y_col: str,
            phase: str = "분석 중",
            meta: dict | None = None,
        ) -> None:
            if phase == "matrix screening":
                ratio = 0.50 * min(done / max(total, 1), 1.0)
            elif phase == "빠른 선별":
                _progress_callback.screen_total = total
                ratio = 0.55 * min(done / max(total, 1), 1.0)
            elif phase == "정밀 계산":
                ratio = 0.50 + 0.50 * min(done / max(total, 1), 1.0)
            else:
                ratio = min(done / max(total, 1), 1.0)
            progress.progress(min(ratio, 1.0))
            meta = meta or {}
            chunk_text = ""
            if meta:
                chunk_text = (
                    f" | X chunk {meta.get('x_chunk_index', '-')}/{meta.get('x_chunk_total', '-')} "
                    f"| Y chunk {meta.get('y_chunk_index', '-')}/{meta.get('y_chunk_total', '-')}"
                    f" | {meta.get('pairs_per_sec', 0.0):,.1f} pair/s"
                    f" | ETA {pd.to_timedelta(int(meta.get('eta_seconds', 0)), unit='s')}"
                )
                if "candidate_count" in meta:
                    chunk_text += f" | 후보 {meta.get('candidate_count', 0):,}개"
            status.caption(f"{phase}: {x_col} → {y_col} ({done:,}/{total:,}){chunk_text}")

        run_config = dict(config)
        run_config["pairwise_analysis"] = dict(pair_cfg)
        run_config["pairwise_analysis"]["fast_screening"] = True
        run_config["pairwise_analysis"]["matrix_only"] = True
        run_config["pairwise_analysis"]["screening_top_k_per_y"] = int(options.get("screening_top_k_per_y", pair_cfg.get("screening_top_k_per_y", 8)))
        run_config["pairwise_analysis"]["screening_max_pairs_total"] = int(options.get("screening_max_pairs_total", pair_cfg.get("screening_max_pairs_total", 120)))
        run_config["pairwise_analysis"]["chunk_x_size"] = int(options.get("chunk_x_size", pair_cfg.get("chunk_x_size", 500)))
        run_config["pairwise_analysis"]["chunk_y_size"] = int(options.get("chunk_y_size", pair_cfg.get("chunk_y_size", 16)))
        run_config["pairwise_analysis"]["batch_flush_rows"] = int(options.get("batch_flush_rows", pair_cfg.get("batch_flush_rows", 5000)))
        run_config["pairwise_analysis"]["save_intermediate_results"] = bool(options.get("save_intermediate_results", True))
        run_config["pairwise_analysis"]["downcast_float32"] = bool(options.get("downcast_float32", False))
        run_config["matrix_screening"] = dict(matrix_cfg)
        run_config["matrix_screening"]["enabled"] = True
        run_config["matrix_screening"]["method"] = str(options.get("matrix_method", "pearson")).lower()
        run_config["matrix_screening"]["top_n_per_y"] = int(options.get("matrix_top_n_per_y", matrix_cfg.get("top_n_per_y", 100)))
        run_config["matrix_screening"]["x_chunk_size"] = int(options.get("matrix_x_chunk_size", matrix_cfg.get("x_chunk_size", 5000)))
        run_config["matrix_screening"]["y_chunk_size"] = int(options.get("matrix_y_chunk_size", matrix_cfg.get("y_chunk_size", 50)))
        run_config["matrix_screening"]["dtype"] = str(options.get("matrix_dtype", matrix_cfg.get("dtype", "float32")))
        run_config["matrix_screening"]["missing_strategy"] = str(options.get("matrix_missing_strategy", matrix_cfg.get("missing_strategy", "mean_impute")))

        output_dir = _prepare_output_dir("results")

        try:
            analysis_df = merged_df
            if options.get("downcast_float32", False):
                analysis_df = downcast_numeric_frame(merged_df, matrix_x_cols + matrix_y_cols)

            pair_result = analyze_pairwise_chunked(
                analysis_df,
                x_cols=matrix_x_cols,
                y_cols=matrix_y_cols,
                key_cols=stats.get("x_keys", []) + stats.get("y_keys", []),
                exclude_cols=source_cols,
                config=run_config,
                progress_callback=_progress_callback,
                output_dir=output_dir,
            )
            st.session_state.pair_result_df = pair_result
            st.session_state.pairwise_excel_bytes = None
            st.session_state.pairwise_html_report = ""
            pair_meta = dict(pair_result.attrs.get("pairwise_meta", {}))
            saved_outputs = _save_large_analysis_outputs(pair_result, output_dir)
            pair_meta.update(saved_outputs)
            if options.get("auto_save_plots", True) and not pair_result.empty:
                plot_dir = _prepare_output_dir("plots")
                plot_index, plot_warnings = save_result_plots(
                    analysis_df,
                    pair_result,
                    plot_dir,
                    plot_config=config.get("plot", {}),
                    top_n_final_score=int(pair_cfg.get("default_top_n", 20)),
                    top_n_effect_size=10,
                    top_n_model_score=10,
                    adjusted_p_threshold=0.05,
                )
                pair_meta["plot_dir"] = str(plot_dir)
                pair_meta["plot_count"] = int(len(plot_index))
                for message in plot_warnings[:1]:
                    st.warning(message)
            st.session_state.pairwise_meta = {
                "x_candidate_count": len(matrix_x_cols),
                "y_candidate_count": len(matrix_y_cols),
                "total_pairs": numeric_pair_count,
                "fast_screening": True,
                "matrix_only": True,
                "fine_scored_pairs": int(len(pair_result)),
                "risk_level": risk_level,
                "large_mode": large_mode,
                **pair_meta,
            }
            progress.progress(1.0)
            status.caption("전체 X-Y 유의쌍 탐색이 완료되었습니다.")
        except Exception as exc:
            st.error(f"전체 X-Y 분석 실패: {exc}")

    pair_result = st.session_state.pair_result_df
    if pair_result.empty:
        return

    filter_cols = st.columns([1, 1, 1, 0.7])
    with filter_cols[0]:
        x_filter = st.multiselect("X 기준 필터", sorted(pair_result["x_feature"].unique()))
    with filter_cols[1]:
        y_filter = st.multiselect("Y 기준 필터", sorted(pair_result["y_target"].unique()))
    with filter_cols[2]:
        judgement_filter = st.multiselect("판정 필터", sorted(pair_result["judgement"].unique()))
    with filter_cols[3]:
        top_n = st.number_input("Top N", min_value=5, max_value=100, value=int(pair_cfg.get("default_top_n", 20)), step=5)

    filtered = pair_result.copy()
    if x_filter:
        filtered = filtered[filtered["x_feature"].isin(x_filter)]
    if y_filter:
        filtered = filtered[filtered["y_target"].isin(y_filter)]
    if judgement_filter:
        filtered = filtered[filtered["judgement"].isin(judgement_filter)]

    if filtered.empty:
        st.warning("필터 조건에 맞는 유의쌍이 없습니다.")
        return

    show_pair_bar = st.toggle("Top 유의쌍 그래프 표시", value=False, key="pair_show_top_chart")
    if show_pair_bar:
        st.plotly_chart(_top_pair_bar_chart(filtered, int(top_n)), use_container_width=True)
    else:
        st.caption("Top 유의쌍 그래프는 필요할 때만 렌더링합니다. 표와 상세 설명은 바로 확인할 수 있습니다.")
    display_cols = [
        "rank",
        "x_feature",
        "y_target",
        "judgement",
        "final_score",
        "p_value",
        "p_value_adj",
        "effect_size",
        "sample_n",
        "direction",
        "caution_text",
    ]
    display_df = filtered[display_cols].head(int(top_n)).copy()
    display_df["p_value"] = display_df["p_value"].map(_format_p_value_table)
    display_df["p_value_adj"] = display_df["p_value_adj"].map(_format_p_value_table)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if st.session_state.pairwise_meta:
        meta = st.session_state.pairwise_meta
        saved_paths = [
            meta.get("full_result_csv"),
            meta.get("full_result_parquet"),
            meta.get("error_log_csv"),
            meta.get("removed_columns_csv"),
            meta.get("detailed_csv"),
            meta.get("detailed_parquet"),
            meta.get("top_by_y"),
            meta.get("x_repeat_summary"),
            meta.get("matrix_candidate_csv"),
            meta.get("matrix_candidate_parquet"),
            meta.get("matrix_dropped_columns_csv"),
            meta.get("matrix_screening_log_csv"),
            meta.get("plot_dir"),
        ]
        saved_paths = [path for path in saved_paths if path]
        if saved_paths:
            st.caption("저장 결과: " + " | ".join(saved_paths))

    pair_names = (filtered["x_feature"].astype(str) + " → " + filtered["y_target"].astype(str)).tolist()
    selected_pair_name = st.selectbox("상세 설명을 볼 X-Y 조합 선택", pair_names)
    selected_index = pair_names.index(selected_pair_name)
    selected_row = filtered.iloc[selected_index]
    _selected_result_detail(selected_row, selected_row["y_target"], selected_row["x_feature"], pair_mode=True)

    st.markdown('<div class="pff-section-title">선택 유의쌍 상세 그래프</div>', unsafe_allow_html=True)
    show_detail_chart = st.toggle("선택 유의쌍 그래프 표시", value=False, key="pair_show_detail_chart")
    st.caption("그래프 종류 변경은 Plotly를 다시 렌더링하므로, 필요한 순간에만 표시하면 화면 반응이 빨라집니다.")
    if show_detail_chart:
        chart_cols = st.columns([1, 1, 1])
        with chart_cols[0]:
            chart_label = st.selectbox("시각화 방식", ["자동 추천", "산점도", "박스플롯", "비율 막대"], index=0, key="pair_chart_mode")
            chart_mode = {"자동 추천": "auto", "산점도": "scatter", "박스플롯": "box", "비율 막대": "bar"}[chart_label]
        chart_df = _chart_input_frame(
            merged_df,
            selected_row["x_feature"],
            selected_row["y_target"],
            large_mode,
        )
        with chart_cols[1]:
            category_count = _axis_category_count(
                chart_df,
                selected_row["x_feature"],
                selected_row["y_target"],
                selected_row["x_type"],
                selected_row["y_type"],
                chart_mode,
            )
            show_all, tick_angle = _plot_options(config, "pair", category_count)

        fig = detail_chart(
            chart_df,
            selected_row["x_feature"],
            selected_row["y_target"],
            selected_row["x_type"],
            selected_row["y_type"],
            chart_mode=chart_mode,
            show_all_categories=show_all,
            tick_angle=tick_angle,
            plot_config=config.get("plot", {}),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("결과 다운로드", expanded=False):
        csv_bytes = filtered.head(int(top_n)).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        down_cols = st.columns(3)
        with down_cols[0]:
            st.download_button("화면 Top N CSV", csv_bytes, "pairwise_top_result.csv", "text/csv", use_container_width=True)
            st.caption("전체 결과는 위에 표시된 저장 경로의 CSV/Parquet 파일을 사용하세요.")
        with down_cols[1]:
            if st.button("Excel 리포트 생성", use_container_width=True):
                preview_df = merged_df.head(100) if large_mode else merged_df
                st.session_state.pairwise_excel_bytes = build_excel_report(
                    preview_df,
                    st.session_state.column_profile,
                    st.session_state.factor_result,
                    pair_result=pair_result.head(500),
                    pairwise_meta=st.session_state.pairwise_meta,
                )
            if st.session_state.pairwise_excel_bytes:
                st.download_button(
                    "Excel 다운로드",
                    st.session_state.pairwise_excel_bytes,
                    "process_factor_finder_pairwise.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        with down_cols[2]:
            if st.button("HTML 리포트 생성", use_container_width=True):
                st.session_state.pairwise_html_report = build_html_report(
                    y_col=st.session_state.analysis_y or "-",
                    merge_stats=st.session_state.merge_stats,
                    factor_result=st.session_state.factor_result,
                    conclusion="전체 X-Y 비교 결과는 확정 판단이 아니라 우선 확인할 후보 쌍입니다.",
                    pair_result=pair_result.head(500),
                    pairwise_meta=st.session_state.pairwise_meta,
                    selected_pair_interpretation=str(selected_row["easy_interpretation"]),
                )
            if st.session_state.pairwise_html_report:
                st.download_button(
                    "HTML 다운로드",
                    st.session_state.pairwise_html_report.encode("utf-8"),
                    "pairwise_report.html",
                    "text/html",
                    use_container_width=True,
                )


def _analysis_page(config: dict, options: dict) -> None:
    """Render factor analysis modes."""
    page_header("유의인자 분석", "선택한 Y 기준 분석과 전체 X-Y 유의쌍 탐색을 제공합니다.")
    merged_df = st.session_state.merged_df
    y_df = st.session_state.y_df
    stats = st.session_state.merge_stats

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        kpi_card("X row 수", f"{stats.get('x_rows', len(st.session_state.x_df)):,}", "최종 X 데이터", "#2563eb", "X")
    with kpi_cols[1]:
        kpi_card("Y row 수", f"{stats.get('y_rows', len(st.session_state.y_df)):,}", "최종 Y 데이터", "#9333ea", "Y")
    with kpi_cols[2]:
        kpi_card("병합률", _format_pct(stats.get("y_merge_rate", 0.0)), "Y 기준 병합 성공률", "#14b8a6", "%")
    with kpi_cols[3]:
        pair_count = len(st.session_state.pair_result_df) if not st.session_state.pair_result_df.empty else 0
        kpi_card("전체 유의쌍", f"{pair_count:,}", "분석 완료 쌍", "#f59e0b", "XY")

    if merged_df.empty:
        st.info("먼저 병합/프로파일 메뉴에서 Key 기준 병합을 실행하세요.")
        return

    mode = st.radio(
        "분석 모드",
        ["단일 Y 기준 분석", "전체 X-Y 유의쌍 탐색"],
        horizontal=True,
    )
    if mode == "단일 Y 기준 분석":
        _single_y_analysis_section(config, options, merged_df, y_df, stats)
    else:
        _pairwise_analysis_section(config, options, merged_df, y_df, stats)


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Process Factor Finder", layout="wide", page_icon="PFF")
    _init_state()

    config = load_config(CONFIG_PATH)
    load_dark_css(BASE_DIR / "assets" / "style.css")

    _run_pending_widget_cleanup()
    _ensure_default_data()
    x_files, y_files, options, page = _sidebar(config)
    _load_input_data(x_files, y_files)

    if page == "개요":
        _overview_page()
    elif page == "병합/프로파일":
        _merge_profile_page(config, options)
    else:
        _analysis_page(config, options)


if __name__ == "__main__":
    main()

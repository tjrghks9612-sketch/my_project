from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.data_loader import preview_dataframe, save_dataset
from core.data_profiler import profile_dataframe
from core.merge_engine import merge_x_y
from core.models import AnalysisPlan
from core.plot_engine import save_top_graphs
from core.scoring import validate_result_scores
from services.artifact_manager import ArtifactManager
from services.artifact_manager import append_error_log
from services.capture_service import save_summary_capture
from services.config_service import load_config
from services.run_manager import execute_analysis
from components.forms import filter_columns
from views.analysis_plan_view import estimate_pair_count
from views.results_view import apply_result_filters, candidate_labels, displayed_results


ROOT = Path(__file__).resolve().parents[1]


def make_dummy(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.normal(size=n)
    cat = np.where(x > 0, "A", "B")
    y_cat = np.where(x > 0, "GOOD", "BAD")
    return pd.DataFrame(
        {
            "KEY": np.arange(n),
            "x_num": x,
            "noise": rng.normal(size=n),
            "x_cat": cat,
            "cat_noise": rng.choice(["K", "L", "M"], size=n),
            "y_num": x * 2.0 + rng.normal(scale=0.2, size=n),
            "y_noise": rng.normal(size=n),
            "y_cat": y_cat,
            "SSN": [f"SN{i:04d}" for i in range(n)],
            "constant": 1,
            "mostly_missing": [np.nan] * (n - 2) + [1, 2],
            "high_card": [f"H{i:04d}" for i in range(n)],
        }
    )


def test_imports_and_core_has_no_streamlit_import() -> None:
    importlib.import_module("app")
    for module in ["home_view", "data_view", "merge_view", "analysis_plan_view", "run_view", "results_view", "settings_view"]:
        importlib.import_module(f"views.{module}")
    for path in (ROOT / "core").glob("*.py"):
        assert "streamlit" not in path.read_text(encoding="utf-8").lower()


def test_preview_and_merge(tmp_path: Path) -> None:
    df = make_dummy(150)
    manifest = save_dataset(df, tmp_path, "sample")
    assert manifest.row_count == 150
    assert preview_dataframe(df, 100, 80).shape == (100, min(80, df.shape[1]))
    merged, stats_manifest = merge_x_y(df[["KEY", "x_num"]], df[["KEY", "y_num"]], ["KEY"], ["KEY"])
    assert len(merged) == 150
    assert stats_manifest.x_merge_rate == 100
    assert stats_manifest.y_merge_rate == 100


def test_profile_exclusions_and_low_cardinality() -> None:
    cfg = load_config(ROOT / "config.yaml")
    profile = profile_dataframe(make_dummy(), "x", cfg, key_cols=["KEY"], y_cols=["y_num", "y_cat"])
    reasons = dict(zip(profile["column"], profile["exclude_reason"]))
    types = dict(zip(profile["column"], profile["inferred_type"]))
    assert reasons["SSN"] == "ID성 컬럼"
    assert reasons["high_card"] in {"ID성 컬럼", "범주 수 30 초과"}
    assert reasons["constant"] == "상수 컬럼"
    assert reasons["mostly_missing"] == "결측률 과다"
    assert types["x_cat"] == "categorical_low_cardinality"
    assert types["x_num"] == "numeric"


def test_full_analysis_detects_all_pair_types(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "config.yaml")
    df = make_dummy()
    plan = AnalysisPlan(
        mode="full_scan",
        x_columns=["x_num", "noise", "x_cat", "cat_noise"],
        y_columns=["y_num", "y_noise", "y_cat"],
        top_n_per_y=50,
        detailed_top_n=80,
        final_top_n=10,
        include_categorical=True,
    )
    artifacts = ArtifactManager(tmp_path).create_run()
    result, manifest, _ = execute_analysis(df, plan, cfg, artifacts, key_cols=["KEY"])
    assert not result.empty
    assert {"numeric_numeric", "categorical_numeric", "numeric_categorical", "categorical_categorical"}.issubset(set(result["pair_type"]))
    assert result.iloc[0]["final_score"] > 70
    assert manifest.scanned_pairs > 0


def test_statistical_metrics_match_references(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "config.yaml")
    df = make_dummy()
    plan = AnalysisPlan(
        mode="full_scan",
        x_columns=["x_num", "x_cat"],
        y_columns=["y_num", "y_cat"],
        top_n_per_y=20,
        detailed_top_n=20,
        final_top_n=10,
        include_categorical=True,
    )
    result, _, _ = execute_analysis(df, plan, cfg, ArtifactManager(tmp_path).create_run(), key_cols=["KEY"])
    nn = result[(result["x_col"] == "x_num") & (result["y_col"] == "y_num")].iloc[0]
    corr, _ = stats.pearsonr(df["x_num"], df["y_num"])
    assert abs(nn["r2_score"] - corr**2) < 1e-5
    cc = result[(result["x_col"] == "x_cat") & (result["y_col"] == "y_cat")].iloc[0]
    table = pd.crosstab(df["x_cat"], df["y_cat"])
    chi2, _, _, _ = stats.chi2_contingency(table)
    expected_v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    assert abs(cc["cramer_v"] - expected_v) < 1e-5


def test_scoring_golden_ranges_and_validation(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "config.yaml")
    df = make_dummy()
    plan = AnalysisPlan(
        mode="full_scan",
        x_columns=["x_num", "noise", "x_cat", "cat_noise", "constant", "SSN"],
        y_columns=["y_num", "y_noise", "y_cat"],
        top_n_per_y=50,
        detailed_top_n=100,
        final_top_n=20,
        include_categorical=True,
    )
    result, _, _ = execute_analysis(df, plan, cfg, ArtifactManager(tmp_path).create_run(), key_cols=["KEY"])
    strong = result[(result["x_col"] == "x_num") & (result["y_col"] == "y_num")]["final_score"].max()
    weak = result[(result["x_col"] == "noise") & (result["y_col"] == "y_noise")]["final_score"].max()
    assert strong > weak
    assert result["final_score"].between(0, 100).all()
    assert result["p_value"].between(0, 1).all()
    assert result["adjusted_p_value"].between(0, 1).all()
    assert np.isfinite(result["final_score"]).all()
    assert "constant" not in set(result["x_col"])
    assert "SSN" not in set(result["x_col"])
    for _, row in result.iterrows():
        ok, reason = validate_result_scores(row.to_dict(), cfg)
        assert ok, reason


def test_artifacts_png_and_no_user_download_reports(tmp_path: Path) -> None:
    cfg = load_config(ROOT / "config.yaml")
    df = make_dummy()
    plan = AnalysisPlan(mode="full_scan", x_columns=["x_num", "x_cat"], y_columns=["y_num", "y_cat"], detailed_top_n=20, final_top_n=3)
    artifacts = ArtifactManager(tmp_path).create_run()
    result, manifest, _ = execute_analysis(df, plan, cfg, artifacts, key_cols=["KEY"])
    graph_paths, errors = save_top_graphs(df, result, artifacts["graphs_dir"], 3)
    assert graph_paths
    assert all(Path(path).exists() and Path(path).suffix == ".png" for path in graph_paths)
    capture = save_summary_capture(result, manifest.to_dict(), Path(artifacts["captures_dir"]) / "significance_analysis_page.png", plan.mode)
    assert Path(capture).exists()
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "views").glob("*.py"))
    assert "download_button" not in source_text
    assert "to_excel" not in source_text
    assert ".pdf" not in source_text.lower()


def test_ui_performance_no_large_multiselect() -> None:
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "views").glob("*.py")
        if path.name != "merge_view.py"
    )
    assert "multiselect" not in source_text
    assert "st.multiselect" in (ROOT / "views" / "merge_view.py").read_text(encoding="utf-8")
    assert "preview_table" in source_text
    assert "그래프 보기" in (ROOT / "components" / "charts.py").read_text(encoding="utf-8")


def test_invalid_regex_is_safe() -> None:
    selected, warnings = filter_columns(["A", "B", "ABC"], include_regex="[abc", label="X")
    assert selected == ["A", "B", "ABC"]
    assert warnings


def test_pair_count_estimate_uses_lengths() -> None:
    assert estimate_pair_count(["A", "B", "C"], ["B", "Y"]) == 5
    source = (ROOT / "views" / "analysis_plan_view.py").read_text(encoding="utf-8")
    assert "for x in x_selected for y" not in source


def test_results_display_limits_selection_to_top_n() -> None:
    result = pd.DataFrame(
        {
            "rank": range(1, 11),
            "pair_type": ["numeric_numeric"] * 10,
            "x_col": [f"X{i}" for i in range(10)],
            "y_col": ["Y"] * 10,
            "final_score": list(range(10, 0, -1)),
        }
    )
    filtered = apply_result_filters(result, "", "", "전체", 0)
    shown = displayed_results(filtered, 3)
    labels = candidate_labels(shown)
    assert len(shown) == 3
    assert len(labels) == 3


def test_results_view_default_does_not_load_merged_frame() -> None:
    source = (ROOT / "views" / "results_view.py").read_text(encoding="utf-8")
    assert "store.load(merged_manifest)" not in source
    assert "load_columns" in source


def test_graph_errors_are_appended_to_error_log(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "error_log.csv"
    append_error_log(
        log_path,
        [
            {
                "phase": "save_graph",
                "x_col": "X",
                "y_col": "Y",
                "pair_type": "numeric_numeric",
                "error": "boom",
            }
        ],
    )
    logged = pd.read_csv(log_path)
    assert {"created_at", "phase", "x_col", "y_col", "pair_type", "error"}.issubset(logged.columns)
    assert logged.iloc[0]["error"] == "boom"


def test_source_distribution_ignores_output_cache_and_runs() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "output/cache/" in ignore_text
    assert "output/*_run/" in ignore_text
    assert "__pycache__/" in ignore_text


def test_copy_labels_are_centralized() -> None:
    copy_text = (ROOT / "style" / "copy_ko.py").read_text(encoding="utf-8")
    for text in ["홈", "데이터", "병합", "분석 계획", "조합 유형", "숫자-숫자"]:
        assert text in copy_text

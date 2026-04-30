"""Reusable Streamlit UI and Plotly chart helpers."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


JUDGEMENT_COLORS = {
    "매우 유의": "#a855f7",
    "유의": "#2563eb",
    "확인 필요": "#14b8a6",
    "약함": "#f59e0b",
    "낮음": "#64748b",
}


def load_dark_css(css_path: str | Path) -> None:
    """Load the dashboard CSS into Streamlit."""
    path = Path(css_path)
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Render the main dashboard header."""
    st.markdown(
        f"""
        <div class="pff-topbar">
            <div>
                <div class="pff-page-title">{escape(title)}</div>
                <div class="pff-page-subtitle">{escape(subtitle)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str | int | float,
    caption: str = "",
    accent: str = "#22d3ee",
    icon: str = "DB",
) -> None:
    """Render a premium KPI card."""
    st.markdown(
        f"""
        <div class="pff-kpi-card" style="--accent:{accent};">
            <div class="pff-kpi-icon">{escape(str(icon))}</div>
            <div class="pff-kpi-content">
                <div class="pff-kpi-label">{escape(str(label))}</div>
                <div class="pff-kpi-value">{escape(str(value))}</div>
                <div class="pff-kpi-caption">{escape(str(caption))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(title: str, value: str, caption: str, ok: bool = True) -> None:
    """Render a compact status card in the sidebar."""
    state = "완료" if ok else "대기"
    state_class = "ok" if ok else "wait"
    st.sidebar.markdown(
        f"""
        <div class="pff-side-status">
            <div class="pff-side-status-left">
                <div class="pff-side-file-icon">{escape(title[:1])}</div>
                <div>
                    <div class="pff-side-status-title">{escape(title)}</div>
                    <div class="pff-side-status-caption">{escape(caption)}</div>
                </div>
            </div>
            <div class="pff-side-state {state_class}">{state}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def judgement_badge(judgement: str) -> str:
    """Return HTML for a colored judgement badge."""
    color = JUDGEMENT_COLORS.get(judgement, "#64748b")
    return f'<span class="pff-badge" style="--badge-color:{color};">{escape(judgement)}</span>'


def format_category_value(value) -> str:
    """Format category values so integer-like values are shown as integers."""
    if pd.isna(value):
        return "결측"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{float(value):g}"
    text = str(value)
    try:
        numeric = float(text)
        return str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
    except Exception:
        return text


def natural_sort_key(value) -> list:
    """Sort values naturally: CH2 before CH10 and 2 before 10."""
    text = format_category_value(value)
    parts = re.split(r"(\d+(?:\.\d+)?)", text)
    key = []
    for part in parts:
        if part == "":
            continue
        try:
            key.append((0, float(part)))
        except ValueError:
            key.append((1, part.lower()))
    return key


def get_ordered_categories(series: pd.Series) -> list[str]:
    """Return non-empty categories in a stable natural order."""
    values = [value for value in series.dropna().unique().tolist()]
    return [format_category_value(value) for value in sorted(values, key=natural_sort_key)]


def apply_category_order(
    fig: go.Figure,
    ordered_categories: list[str],
    show_all_categories: bool = True,
    tick_angle: int = -45,
) -> None:
    """Apply category order and optionally force all x-axis labels to show."""
    if not ordered_categories:
        return
    fig.update_xaxes(categoryorder="array", categoryarray=ordered_categories)
    if show_all_categories:
        fig.update_xaxes(
            tickmode="array",
            tickvals=ordered_categories,
            ticktext=ordered_categories,
            tickangle=tick_angle,
        )


def factor_rank_panel(result_df: pd.DataFrame, top_n: int = 15) -> None:
    """Render the top factor ranking with native Streamlit widgets.

    This avoids raw HTML rows because Streamlit Markdown can sometimes show
    nested HTML as text after reruns or parser edge cases.
    """
    if result_df is None or result_df.empty:
        st.info("아직 분석 결과가 없습니다.")
        return

    display_df = result_df.head(top_n).copy()
    display_df = display_df[
        ["rank", "feature", "judgement", "final_score", "effect_score", "model_score", "stability_score"]
    ].rename(
        columns={
            "rank": "Rank",
            "feature": "인자",
            "judgement": "판정",
            "final_score": "점수",
            "effect_score": "효과",
            "model_score": "모델",
            "stability_score": "안정성",
        }
    )

    with st.container(border=True):
        header_left, header_right = st.columns([1, 0.32])
        with header_left:
            st.markdown(f"#### Top {min(top_n, len(result_df))} 유의인자")
            st.caption("통계, 효과 크기, 모델 중요도, 안정성, 데이터 품질 통합 점수")
        with header_right:
            st.markdown("**점수 기준**")
            st.caption("0-100")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=min(520, 76 + 35 * len(display_df)),
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "인자": st.column_config.TextColumn("인자", width="medium"),
                "판정": st.column_config.TextColumn("판정", width="small"),
                "점수": st.column_config.ProgressColumn("점수", min_value=0, max_value=100, format="%.1f"),
                "효과": st.column_config.ProgressColumn("효과", min_value=0, max_value=100, format="%.1f"),
                "모델": st.column_config.ProgressColumn("모델", min_value=0, max_value=100, format="%.1f"),
                "안정성": st.column_config.ProgressColumn("안정성", min_value=0, max_value=100, format="%.1f"),
            },
        )


def conclusion_card(result_df: pd.DataFrame, y_col: str) -> None:
    """Show a high-signal conclusion card from the top-ranked factor."""
    if result_df is None or result_df.empty:
        st.markdown('<div class="pff-panel-empty">아직 분석 결과가 없습니다.</div>', unsafe_allow_html=True)
        return

    top = result_df.iloc[0]
    judgement = escape(str(top["judgement"]))
    color = JUDGEMENT_COLORS.get(top["judgement"], "#64748b")
    st.markdown(
        f"""
        <div class="pff-conclusion">
            <div class="pff-trophy">1</div>
            <div class="pff-conclusion-body">
                <div class="pff-conclusion-title">한 줄 결론</div>
                <div class="pff-conclusion-main">
                    <b>{escape(str(top['feature']))}</b>이 현재 <b>{escape(y_col)}</b>에 대해 우선 확인할 유의 후보로 탐지되었습니다.
                </div>
                <div class="pff-conclusion-meta">
                    <span>점수 <b>{float(top['final_score']):.1f}</b></span>
                    <span>판정 <b style="color:{color};">{judgement}</b></span>
                    <span>품질 <b>{float(top['quality_score']):.1f}</b></span>
                    <span>확정 판단 전 확인 필요</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def evidence_panel(result_df: pd.DataFrame) -> None:
    """Render short evidence and detail cards for the top factor."""
    if result_df is None or result_df.empty:
        return

    top = result_df.iloc[0]
    p_value = _format_p_value(top["p_value"])
    stability_label = "높음" if float(top["stability_score"]) >= 75 else "보통" if float(top["stability_score"]) >= 50 else "낮음"
    st.markdown(
        f"""
        <div class="pff-evidence-grid">
            <div class="pff-evidence-card">
                <div class="pff-small-title">핵심 근거</div>
                <ul>
                    <li>{escape(str(top['direction']))}</li>
                    <li>{escape(str(top['metric_text']))}</li>
                    <li>{escape(str(top['caution_text']))}</li>
                </ul>
            </div>
            <div class="pff-evidence-card">
                <div class="pff-small-title">상세 정보 ({escape(str(top['feature']))})</div>
                <div class="pff-detail-line"><span>p-value</span><b>{p_value}</b></div>
                <div class="pff-detail-line"><span>효과 크기</span><b>{float(top['effect_size']):.4f}</b></div>
                <div class="pff-detail-line"><span>안정성</span><b>{float(top['stability_score']):.1f} ({stability_label})</b></div>
                <div class="pff-detail-line"><span>표본 수</span><b>{int(top['sample_n']):,}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_factor_bar_chart(result_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Build a horizontal bar chart for top-ranked factors."""
    plot_df = result_df.head(top_n).sort_values("final_score", ascending=True).copy()
    colors = [JUDGEMENT_COLORS.get(value, "#64748b") for value in plot_df["judgement"]]
    fig = go.Figure(
        go.Bar(
            x=plot_df["final_score"],
            y=plot_df["feature"],
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(255,255,255,0.16)", "width": 1}},
            text=plot_df["final_score"].round(1),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Final Score=%{x:.1f}<extra></extra>",
        )
    )
    _apply_dark_layout(fig, height=max(420, 28 * len(plot_df) + 130))
    fig.update_xaxes(range=[0, 105], title="Final Score")
    fig.update_yaxes(title="")
    return fig


def result_table(result_df: pd.DataFrame) -> None:
    """Render the result table with the most useful columns first."""
    if result_df is None or result_df.empty:
        st.info("분석 결과가 없습니다.")
        return

    display_cols = [
        "rank",
        "feature",
        "feature_type",
        "judgement",
        "final_score",
        "stat_score",
        "effect_score",
        "model_score",
        "stability_score",
        "quality_score",
        "p_value",
        "effect_size",
        "direction",
        "caution_text",
    ]
    st.dataframe(
        result_df[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "feature": st.column_config.TextColumn("인자", width="medium"),
            "judgement": st.column_config.TextColumn("판정", width="small"),
            "final_score": st.column_config.ProgressColumn("점수", min_value=0, max_value=100, format="%.1f"),
            "p_value": st.column_config.NumberColumn("p-value", format="%.2e"),
        },
    )


def detail_chart(
    df: pd.DataFrame,
    feature: str,
    y_col: str,
    feature_type: str,
    y_type: str,
    chart_mode: str = "auto",
    show_all_categories: bool = True,
    tick_angle: int = -45,
    plot_config: dict | None = None,
) -> go.Figure:
    """Create a detail visualization for one selected factor."""
    pair = df[[feature, y_col]].dropna().copy()
    if pair.empty:
        fig = go.Figure()
        _apply_dark_layout(fig)
        return fig

    plot_config = plot_config or {}

    if chart_mode == "scatter":
        fig = _scatter_chart(pair, feature, y_col, feature_type, y_type)
    elif chart_mode == "box":
        fig = _box_chart(pair, feature, y_col, feature_type, y_type)
    elif chart_mode == "bar":
        fig = _bar_chart(pair, feature, y_col)
    elif feature_type == "numeric" and y_type == "numeric":
        fig = _scatter_chart(pair, feature, y_col, feature_type, y_type)
    elif feature_type == "categorical" and y_type == "numeric":
        fig = _box_chart(pair, feature, y_col, feature_type, y_type)
    elif feature_type == "numeric" and y_type in {"binary", "categorical"}:
        fig = _scatter_chart(pair, feature, y_col, feature_type, y_type)
    elif feature_type == "categorical" and y_type in {"binary", "categorical"}:
        fig = _bar_chart(pair, feature, y_col)
    else:
        fig = px.histogram(pair, x=feature, color_discrete_sequence=["#8b5cf6"])

    ordered_categories = _x_axis_categories_for_chart(pair, feature, y_col, feature_type, y_type, chart_mode)
    height = _chart_height(ordered_categories, plot_config)
    _apply_dark_layout(fig, height=height)
    apply_category_order(fig, ordered_categories, show_all_categories=show_all_categories, tick_angle=tick_angle)
    fig.update_traces(marker_line_color="rgba(255,255,255,0.18)", marker_line_width=0.7)
    return fig


def sanitize_plot_filename(text: str) -> str:
    """Make a filesystem-safe filename fragment."""
    cleaned = re.sub(r"[<>:\"/\\\\|?*]+", "_", str(text)).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] or "plot"


def build_plot_artifacts(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_type: str,
    y_type: str,
    title_text: str,
    plot_config: dict | None = None,
) -> list[tuple[str, go.Figure]]:
    """Generate one or more reusable figures for a selected result row."""
    pair = df[[x_col, y_col]].dropna().copy()
    if pair.empty:
        return []

    figures = []
    main_fig = detail_chart(pair, x_col, y_col, x_type, y_type, chart_mode="auto", plot_config=plot_config or {})
    main_fig.update_layout(title={"text": title_text, "x": 0.01})
    figures.append(("main", main_fig))

    if x_type == "categorical" and y_type == "numeric":
        temp = pair.copy()
        order = get_ordered_categories(temp[x_col])
        temp[x_col] = temp[x_col].map(format_category_value)

        violin = px.violin(
            temp,
            x=x_col,
            y=y_col,
            box=True,
            points="outliers",
            category_orders={x_col: order},
            color_discrete_sequence=["#22d3ee"],
        )
        _apply_dark_layout(violin)
        apply_category_order(violin, order, show_all_categories=True, tick_angle=-45)
        violin.update_layout(title={"text": title_text + "<br><sup>Violin distribution</sup>", "x": 0.01})
        figures.append(("violin", violin))

        hist = px.histogram(
            temp,
            x=y_col,
            color=x_col,
            nbins=min(40, max(10, int(np.sqrt(len(temp))))),
        )
        _apply_dark_layout(hist)
        hist.update_layout(title={"text": title_text + "<br><sup>Distribution histogram</sup>", "x": 0.01})
        figures.append(("histogram", hist))

    if x_type == "numeric" and y_type == "numeric":
        hist_x = px.histogram(pair, x=x_col, nbins=min(50, max(10, int(np.sqrt(len(pair))))), color_discrete_sequence=["#8b5cf6"])
        _apply_dark_layout(hist_x)
        hist_x.update_layout(title={"text": title_text + f"<br><sup>{x_col} distribution</sup>", "x": 0.01})
        figures.append(("x_histogram", hist_x))

        hist_y = px.histogram(pair, x=y_col, nbins=min(50, max(10, int(np.sqrt(len(pair))))), color_discrete_sequence=["#22d3ee"])
        _apply_dark_layout(hist_y)
        hist_y.update_layout(title={"text": title_text + f"<br><sup>{y_col} distribution</sup>", "x": 0.01})
        figures.append(("y_histogram", hist_y))

    return figures


def save_result_plots(
    df: pd.DataFrame,
    result_df: pd.DataFrame,
    output_dir: str | Path,
    plot_config: dict | None = None,
    top_n_final_score: int = 20,
    top_n_effect_size: int = 10,
    top_n_model_score: int = 10,
    adjusted_p_threshold: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Save result plots as HTML and optional PNG, then return an index table."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    warnings = []

    if result_df is None or result_df.empty:
        return pd.DataFrame(), warnings

    picks = []
    picks.extend(result_df.sort_values("final_score", ascending=False).head(top_n_final_score).to_dict("records"))
    picks.extend(result_df.sort_values("effect_size", ascending=False).head(top_n_effect_size).to_dict("records"))
    if "model_score" in result_df.columns:
        picks.extend(result_df.sort_values("model_score", ascending=False).head(top_n_model_score).to_dict("records"))
    if adjusted_p_threshold is not None and "p_value_adj" in result_df.columns:
        picks.extend(result_df[result_df["p_value_adj"].fillna(1.0) <= adjusted_p_threshold].to_dict("records"))

    unique_rows = {}
    for row in picks:
        unique_rows[(str(row.get("x_feature")), str(row.get("y_target")))] = row

    index_rows = []
    png_enabled = True
    for row in unique_rows.values():
        x_col = str(row.get("x_feature", ""))
        y_col = str(row.get("y_target", ""))
        x_type = str(row.get("x_type", ""))
        y_type = str(row.get("y_type", ""))
        subtitle = (
            f"{x_col} vs {y_col}<br><sup>"
            f"final={row.get('final_score', '-')}, p={row.get('p_value', '-')}, adj_p={row.get('p_value_adj', '-')}, "
            f"effect={row.get('effect_size', '-')}, model={row.get('model_score', '-')}, n={row.get('sample_n', '-')}</sup>"
        )
        figures = build_plot_artifacts(df, x_col, y_col, x_type, y_type, subtitle, plot_config=plot_config)
        rank = int(row.get("rank", 0))
        score_text = sanitize_plot_filename(f"{float(row.get('final_score', 0)):.1f}")
        effect_text = sanitize_plot_filename(f"{float(row.get('effect_size', 0)):.4f}")
        model_text = sanitize_plot_filename(f"{float(row.get('model_score', 0)):.1f}")
        base_name = sanitize_plot_filename(f"{rank:03d}_{x_col}_{y_col}_score_{score_text}_effect_{effect_text}_model_{model_text}")

        for kind, fig in figures:
            html_path = output_path / f"{base_name}_{kind}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            png_path = output_path / f"{base_name}_{kind}.png"
            png_saved = False
            if png_enabled:
                try:
                    fig.write_image(str(png_path))
                    png_saved = True
                except Exception:
                    png_enabled = False
                    warnings.append("kaleido가 없어 PNG 저장은 건너뛰고 HTML만 저장했습니다.")
            index_rows.append(
                {
                    "rank": rank,
                    "x_col": x_col,
                    "y_col": y_col,
                    "plot_kind": kind,
                    "final_score": row.get("final_score"),
                    "effect_size": row.get("effect_size"),
                    "model_score": row.get("model_score"),
                    "p_value": row.get("p_value"),
                    "adjusted_p_value": row.get("p_value_adj"),
                    "sample_n": row.get("sample_n"),
                    "html_path": str(html_path),
                    "png_path": str(png_path) if png_saved else "",
                }
            )

    index_df = pd.DataFrame(index_rows)
    if not index_df.empty:
        index_df.to_csv(output_path / "plot_index.csv", index=False, encoding="utf-8-sig")
    return index_df, warnings


def _scatter_chart(pair: pd.DataFrame, feature: str, y_col: str, feature_type: str, y_type: str) -> go.Figure:
    """Create a scatter-like chart, using strip plots for categorical axes."""
    if feature_type == "numeric" and y_type == "numeric":
        fig = px.scatter(pair, x=feature, y=y_col, opacity=0.58, color_discrete_sequence=["#8b5cf6"])
        if len(pair) >= 3:
            x = pd.to_numeric(pair[feature], errors="coerce")
            y = pd.to_numeric(pair[y_col], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() >= 3:
                slope, intercept = np.polyfit(x[valid], y[valid], 1)
                x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=slope * x_line + intercept,
                        mode="lines",
                        name="Trend",
                        line={"color": "#22d3ee", "width": 3, "dash": "dot"},
                    )
                )
        return fig

    if feature_type == "categorical" and y_type == "numeric":
        order = get_ordered_categories(pair[feature])
        temp = pair.copy()
        temp[feature] = temp[feature].map(format_category_value)
        return px.strip(
            temp,
            x=feature,
            y=y_col,
            category_orders={feature: order},
            color_discrete_sequence=["#22d3ee"],
        )

    if feature_type == "numeric":
        return px.strip(pair, x=y_col, y=feature, color_discrete_sequence=["#14b8a6"])

    return px.strip(pair, x=feature, y=y_col, color_discrete_sequence=["#8b5cf6"])


def _box_chart(pair: pd.DataFrame, feature: str, y_col: str, feature_type: str, y_type: str) -> go.Figure:
    """Create a boxplot when the selected variables support it."""
    if feature_type == "categorical" and y_type == "numeric":
        order = get_ordered_categories(pair[feature])
        temp = pair.copy()
        temp[feature] = temp[feature].map(format_category_value)
        return px.box(
            temp,
            x=feature,
            y=y_col,
            points="outliers",
            category_orders={feature: order},
            color_discrete_sequence=["#6366f1"],
        )
    if feature_type == "numeric" and y_type in {"binary", "categorical"}:
        return px.box(pair, x=y_col, y=feature, points="outliers", color_discrete_sequence=["#14b8a6"])
    if feature_type == "numeric" and y_type == "numeric":
        temp = pair.copy()
        temp[f"{feature}_bin"] = pd.qcut(pd.to_numeric(temp[feature], errors="coerce"), q=5, duplicates="drop")
        temp[f"{feature}_bin"] = temp[f"{feature}_bin"].astype(str)
        order = get_ordered_categories(temp[f"{feature}_bin"])
        return px.box(
            temp,
            x=f"{feature}_bin",
            y=y_col,
            points="outliers",
            category_orders={f"{feature}_bin": order},
            color_discrete_sequence=["#6366f1"],
        )
    return _bar_chart(pair, feature, y_col)


def _bar_chart(pair: pd.DataFrame, feature: str, y_col: str) -> go.Figure:
    """Create a readable bar chart for categorical or binned numeric comparison."""
    temp = pair.copy()
    x_col = feature

    if pd.api.types.is_numeric_dtype(temp[feature]) and temp[feature].nunique(dropna=True) > 20:
        temp["_feature_group"] = pd.qcut(pd.to_numeric(temp[feature], errors="coerce"), q=8, duplicates="drop")
        temp["_feature_group"] = temp["_feature_group"].astype(str)
        x_col = "_feature_group"
    else:
        temp[x_col] = temp[feature].map(format_category_value)

    if pd.api.types.is_numeric_dtype(temp[y_col]):
        summary = temp.groupby(x_col, dropna=False)[y_col].agg(["mean", "count"]).reset_index()
        ordered = get_ordered_categories(summary[x_col])
        fig = go.Figure(
            go.Bar(
                x=summary[x_col].astype(str),
                y=summary["mean"],
                text=summary["mean"].round(2),
                textposition="outside",
                marker_color="#8b5cf6",
                hovertemplate="<b>%{x}</b><br>평균=%{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(yaxis_title=f"{y_col} 평균", xaxis_title=feature)
        fig.update_xaxes(categoryorder="array", categoryarray=ordered)
        return fig

    table = pd.crosstab(temp[x_col].astype(str), temp[y_col].astype(str), normalize="index") * 100
    if table.empty:
        return px.histogram(temp, x=x_col, color_discrete_sequence=["#8b5cf6"])
    fig = go.Figure()
    palette = ["#8b5cf6", "#22d3ee", "#14b8a6", "#f59e0b", "#64748b"]
    for idx, col in enumerate(table.columns):
        fig.add_trace(go.Bar(x=table.index, y=table[col], name=str(col), marker_color=palette[idx % len(palette)]))
    fig.update_layout(barmode="stack", yaxis_title="Row %", xaxis_title=feature)
    fig.update_xaxes(categoryorder="array", categoryarray=get_ordered_categories(pd.Series(table.index)))
    return fig


def _x_axis_categories_for_chart(
    pair: pd.DataFrame,
    feature: str,
    y_col: str,
    feature_type: str,
    y_type: str,
    chart_mode: str,
) -> list[str]:
    """Return categories shown on the x-axis for tick ordering."""
    if chart_mode == "bar":
        if pd.api.types.is_numeric_dtype(pair[feature]) and pair[feature].nunique(dropna=True) > 20:
            bins = pd.qcut(pd.to_numeric(pair[feature], errors="coerce"), q=8, duplicates="drop").astype(str)
            return get_ordered_categories(bins)
        return get_ordered_categories(pair[feature])
    if chart_mode == "box" and feature_type == "numeric" and y_type == "numeric":
        bins = pd.qcut(pd.to_numeric(pair[feature], errors="coerce"), q=5, duplicates="drop").astype(str)
        return get_ordered_categories(bins)
    if feature_type == "categorical":
        return get_ordered_categories(pair[feature])
    if feature_type == "numeric" and y_type in {"binary", "categorical"}:
        return get_ordered_categories(pair[y_col])
    return []


def _chart_height(ordered_categories: list[str], plot_config: dict) -> int:
    """Increase chart height when many categories are shown."""
    min_height = int(plot_config.get("min_chart_height", 420))
    max_height = int(plot_config.get("max_chart_height", 900))
    per_category = int(plot_config.get("auto_height_per_category", 18))
    if not ordered_categories:
        return min_height
    return int(np.clip(min_height + max(0, len(ordered_categories) - 12) * per_category, min_height, max_height))


def _apply_dark_layout(fig: go.Figure, height: int = 480) -> None:
    """Apply the shared dark dashboard Plotly style."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,27,0.60)",
        font={"color": "#e5e7eb", "family": "Inter, Arial, sans-serif"},
        margin={"l": 36, "r": 28, "t": 42, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.20)",
        linecolor="rgba(148,163,184,0.18)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.12)",
        zerolinecolor="rgba(148,163,184,0.20)",
        linecolor="rgba(148,163,184,0.18)",
    )


def _format_p_value(value: object) -> str:
    """Format p-values compactly for dashboard cards."""
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

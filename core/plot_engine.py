"""Graph creation and PNG artifact saving."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from style.copy_ko import GRAPH_CAPTION, PAIR_TYPE_LABELS


PLOT_COLORS = {
    "primary": "#8b5cf6",
    "secondary": "#22d3ee",
    "blue": "#2563eb",
    "teal": "#14b8a6",
    "amber": "#f59e0b",
    "grid": "rgba(148,163,184,0.14)",
}


def configure_matplotlib_font() -> None:
    """Use a Korean-capable font for PNG artifacts when the host provides one."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_font()


def sanitize_filename(text: str, limit: int = 120) -> str:
    allowed = []
    for char in str(text):
        allowed.append(char if char.isalnum() or char in {"_", "-", "."} else "_")
    return "".join(allowed)[:limit].strip("_") or "artifact"


def _short_label(text: str, limit: int = 34) -> str:
    value = str(text)
    return value if len(value) <= limit else value[: limit - 3] + "..."


def make_plotly_figure(df: pd.DataFrame, result_row: pd.Series | dict, chart_mode: str = "auto") -> go.Figure:
    """Create an on-demand Plotly figure for Results view."""
    row = dict(result_row)
    x_col = row["x_col"]
    y_col = row["y_col"]
    pair_type = row["pair_type"]
    pair = df[[x_col, y_col]].dropna()
    title = f"{_short_label(x_col)} → {_short_label(y_col)}"
    subtitle = (
        f"{PAIR_TYPE_LABELS.get(pair_type, pair_type)} | 최종 점수 {float(row.get('final_score', 0)):.1f} | "
        f"p={float(row.get('p_value', 1)):.2e} | n={int(row.get('sample_n', 0))}"
    )
    if pair.empty:
        return go.Figure().update_layout(template="plotly_dark", title=title)
    mode = _resolve_chart_mode(pair_type, chart_mode)
    if mode == "scatter":
        fig = _plot_scatter(pair, x_col, y_col, pair_type, title)
    elif mode == "box":
        fig = _plot_box(pair, x_col, y_col, pair_type, title)
    elif mode == "ratio":
        fig = _plot_ratio(pair, x_col, y_col, pair_type, title)
    else:
        fig = _plot_default(pair, x_col, y_col, pair_type, title)
    _apply_plotly_layout(fig, title, subtitle, pair_type)
    return fig


def _resolve_chart_mode(pair_type: str, chart_mode: str) -> str:
    """Map UI chart choices to concrete plot builders."""
    if chart_mode in {"scatter", "box", "ratio"}:
        return chart_mode
    if pair_type == "numeric_numeric":
        return "scatter"
    if pair_type in {"categorical_numeric", "numeric_categorical"}:
        return "box"
    return "ratio"


def _plot_default(pair: pd.DataFrame, x_col: str, y_col: str, pair_type: str, title: str) -> go.Figure:
    mode = _resolve_chart_mode(pair_type, "auto")
    if mode == "scatter":
        return _plot_scatter(pair, x_col, y_col, pair_type, title)
    if mode == "box":
        return _plot_box(pair, x_col, y_col, pair_type, title)
    return _plot_ratio(pair, x_col, y_col, pair_type, title)


def _plot_scatter(pair: pd.DataFrame, x_col: str, y_col: str, pair_type: str, title: str) -> go.Figure:
    """Scatter/strip style chart for visually checking point-level spread."""
    if pair_type == "numeric_numeric":
        return px.scatter(pair, x=x_col, y=y_col, opacity=0.58, trendline="ols", title=title)
    if pair_type == "categorical_numeric":
        return px.strip(pair, x=x_col, y=y_col, title=title)
    if pair_type == "numeric_categorical":
        return px.strip(pair, x=y_col, y=x_col, title=title)
    return px.strip(pair, x=x_col, y=y_col, color=y_col, title=title)


def _plot_box(pair: pd.DataFrame, x_col: str, y_col: str, pair_type: str, title: str) -> go.Figure:
    """Box plot chart. Numeric-numeric falls back to binned X boxes."""
    if pair_type == "numeric_numeric":
        temp = pair.copy()
        temp["_x_bin"] = pd.qcut(pd.to_numeric(temp[x_col], errors="coerce"), q=6, duplicates="drop").astype(str)
        return px.box(temp, x="_x_bin", y=y_col, points="outliers", title=title, labels={"_x_bin": f"{x_col} 구간"})
    if pair_type == "categorical_numeric":
        return px.box(pair, x=x_col, y=y_col, points="outliers", title=title)
    if pair_type == "numeric_categorical":
        return px.box(pair, x=y_col, y=x_col, points="outliers", title=title)
    return _plot_ratio(pair, x_col, y_col, pair_type, title)


def _plot_ratio(pair: pd.DataFrame, x_col: str, y_col: str, pair_type: str, title: str) -> go.Figure:
    """Ratio or summary chart for categorical comparisons."""
    temp = pair.copy()
    x_axis = x_col
    if pd.api.types.is_numeric_dtype(temp[x_col]) and temp[x_col].nunique(dropna=True) > 12:
        temp["_x_bin"] = pd.qcut(pd.to_numeric(temp[x_col], errors="coerce"), q=8, duplicates="drop").astype(str)
        x_axis = "_x_bin"
    if pd.api.types.is_numeric_dtype(temp[y_col]) and pair_type != "numeric_categorical":
        summary = temp.groupby(x_axis, dropna=False)[y_col].agg(["mean", "count"]).reset_index()
        fig = px.bar(summary, x=x_axis, y="mean", text="mean", title=title, labels={x_axis: x_col, "mean": f"{y_col} 평균"})
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        return fig
    table = pd.crosstab(temp[x_axis].astype(str), temp[y_col].astype(str), normalize="index") * 100
    if table.empty:
        return go.Figure().update_layout(template="plotly_dark", title=title)
    fig = go.Figure()
    palette = [PLOT_COLORS["primary"], PLOT_COLORS["secondary"], PLOT_COLORS["teal"], PLOT_COLORS["amber"], PLOT_COLORS["blue"]]
    for idx, col in enumerate(table.columns):
        fig.add_trace(go.Bar(x=table.index, y=table[col], name=str(col), marker_color=palette[idx % len(palette)]))
    fig.update_layout(barmode="stack", yaxis_title="비율 (%)", xaxis_title=x_col)
    return fig


def _apply_plotly_layout(fig: go.Figure, title: str, subtitle: str, pair_type: str) -> None:
    """Apply shared dashboard chart styling."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,27,0.68)",
        title={"text": f"{title}<br><sup>{subtitle}<br>{GRAPH_CAPTION}</sup>"},
        margin={"l": 54, "r": 28, "t": 92, "b": 64},
        font={"family": "Inter, Segoe UI, Malgun Gothic, Arial, sans-serif", "color": "#e5e7eb"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_traces(marker_line_color="rgba(255,255,255,0.16)", marker_line_width=0.6)
    fig.update_xaxes(automargin=True, tickangle=-35 if pair_type != "numeric_numeric" else 0, gridcolor=PLOT_COLORS["grid"], zerolinecolor="rgba(148,163,184,0.22)")
    fig.update_yaxes(automargin=True, gridcolor=PLOT_COLORS["grid"], zerolinecolor="rgba(148,163,184,0.22)")


def _caption(ax) -> None:
    ax.text(
        0.5,
        -0.18,
        GRAPH_CAPTION,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )


def save_pair_png(df: pd.DataFrame, result_row: pd.Series | dict, output_path: str | Path) -> str:
    """Save one pair graph as a matplotlib PNG."""
    row = dict(result_row)
    x_col = row["x_col"]
    y_col = row["y_col"]
    pair_type = row["pair_type"]
    pair = df[[x_col, y_col]].dropna()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    title = f"{_short_label(x_col)} → {_short_label(y_col)}"
    subtitle = (
        f"{PAIR_TYPE_LABELS.get(pair_type, pair_type)} | 최종 점수 {float(row.get('final_score', 0)):.1f} | "
        f"p={float(row.get('p_value', 1)):.2e} | effect={float(row.get('effect_size', 0)):.3f} | n={int(row.get('sample_n', 0))}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=14)
    if pair.empty:
        ax.text(0.5, 0.5, "표시할 데이터가 없습니다.", ha="center", va="center")
    elif pair_type == "numeric_numeric":
        x = pd.to_numeric(pair[x_col], errors="coerce")
        y = pd.to_numeric(pair[y_col], errors="coerce")
        ax.scatter(x, y, s=14, alpha=0.55, color=PLOT_COLORS["blue"])
        if len(pair) >= 3 and x.nunique() > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, slope * xs + intercept, color=PLOT_COLORS["primary"], linewidth=2)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    elif pair_type == "categorical_numeric":
        groups = [group[y_col].astype(float).to_numpy() for _, group in pair.groupby(x_col)]
        labels = [str(name) for name, _ in pair.groupby(x_col)]
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    elif pair_type == "numeric_categorical":
        groups = [group[x_col].astype(float).to_numpy() for _, group in pair.groupby(y_col)]
        labels = [str(name) for name, _ in pair.groupby(y_col)]
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel(y_col)
        ax.set_ylabel(x_col)
    else:
        table = pd.crosstab(pair[x_col].astype(str), pair[y_col].astype(str), normalize="index")
        image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(table.columns)), labels=[str(c) for c in table.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(table.index)), labels=[str(i) for i in table.index])
        ax.set_xlabel(y_col)
        ax.set_ylabel(x_col)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(alpha=0.18)
    _caption(ax)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def save_top_graphs(df: pd.DataFrame, result_df: pd.DataFrame, graphs_dir: str | Path, top_n: int) -> tuple[list[str], list[dict]]:
    """Save final-score Top N graph PNG files."""
    paths: list[str] = []
    errors: list[dict] = []
    if result_df is None or result_df.empty:
        return paths, errors
    graph_path = Path(graphs_dir)
    graph_path.mkdir(parents=True, exist_ok=True)
    for _, row in result_df.sort_values("final_score", ascending=False).head(top_n).iterrows():
        try:
            name = sanitize_filename(
                f"rank_{int(row['rank']):03d}__{row['pair_type']}__{row['x_col']}__{row['y_col']}__score_{float(row['final_score']):.1f}.png"
            )
            if not name.endswith(".png"):
                name += ".png"
            paths.append(save_pair_png(df, row, graph_path / name))
        except Exception as exc:
            errors.append(
                {
                    "phase": "save_graph",
                    "x_col": row.get("x_col"),
                    "y_col": row.get("y_col"),
                    "pair_type": row.get("pair_type"),
                    "error": repr(exc),
                }
            )
    return paths, errors

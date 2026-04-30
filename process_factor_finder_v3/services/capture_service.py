"""Static summary PNG generation for Results page captures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageGrab

from core.plot_engine import configure_matplotlib_font
from style.copy_ko import NOT_CAUSAL, PAIR_TYPE_LABELS


configure_matplotlib_font()


def save_visible_results_capture(capture_path: str | Path) -> str:
    """Save the currently visible desktop screen as the Results page capture."""
    output = Path(capture_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = ImageGrab.grab(all_screens=True)
    image.save(output)
    return str(output)


def save_summary_capture(
    result_df: pd.DataFrame,
    run_manifest: dict,
    capture_path: str | Path,
    mode: str,
    data_summary: dict | None = None,
) -> str:
    """Create a single static PNG that looks like an analysis page capture."""
    output = Path(capture_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    top = result_df.sort_values("final_score", ascending=False).head(10) if result_df is not None and not result_df.empty else pd.DataFrame()
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#0b1020")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.94, "Process Factor Finder v3", fontsize=24, color="#e5e7eb", weight="bold")
    ax.text(0.04, 0.90, "유의인자 분석 페이지 캡쳐", fontsize=14, color="#93c5fd")
    ax.text(0.04, 0.86, f"분석 모드: {mode}", fontsize=12, color="#cbd5e1")
    ax.text(0.04, 0.83, f"실행 ID: {run_manifest.get('run_id', '-')}", fontsize=11, color="#94a3b8")
    summary = data_summary or {}
    cards = [
        ("전체 스캔 pair", run_manifest.get("scanned_pairs", 0)),
        ("정밀 확인 후보", run_manifest.get("detailed_candidates", 0)),
        ("최종 Top 후보", len(top)),
        ("데이터 rows", summary.get("rows", "-")),
    ]
    x0 = 0.04
    for idx, (label, value) in enumerate(cards):
        x = x0 + idx * 0.23
        ax.add_patch(plt.Rectangle((x, 0.73), 0.20, 0.08, facecolor="#111827", edgecolor="#334155", linewidth=1.2))
        ax.text(x + 0.015, 0.78, str(value), fontsize=20, color="#ffffff", weight="bold")
        ax.text(x + 0.015, 0.745, label, fontsize=10, color="#94a3b8")

    ax.text(0.04, 0.67, "Top 후보", fontsize=16, color="#e5e7eb", weight="bold")
    if top.empty:
        ax.text(0.04, 0.62, "표시할 결과가 없습니다.", fontsize=12, color="#cbd5e1")
    else:
        columns = ["rank", "pair_type", "x_col", "y_col", "final_score", "p_value", "interpretation"]
        shown = top[[col for col in columns if col in top.columns]].copy()
        y = 0.62
        for _, row in shown.iterrows():
            pair_label = PAIR_TYPE_LABELS.get(str(row.get("pair_type", "")), str(row.get("pair_type", "")))
            text = (
                f"#{int(row.get('rank', 0)):02d}  {pair_label}  "
                f"{row.get('x_col', '')} → {row.get('y_col', '')}  "
                f"score={float(row.get('final_score', 0)):.1f}  p={float(row.get('p_value', 1)):.2e}"
            )
            ax.text(0.05, y, text[:150], fontsize=10, color="#e2e8f0")
            y -= 0.04
    if not top.empty:
        leader = top.iloc[0]
        ax.add_patch(plt.Rectangle((0.04, 0.16), 0.90, 0.14, facecolor="#111827", edgecolor="#7c3aed", linewidth=1.2))
        ax.text(0.06, 0.26, "대표 후보 상세", fontsize=13, color="#c4b5fd", weight="bold")
        ax.text(0.06, 0.225, str(leader.get("interpretation", ""))[:170], fontsize=10, color="#e5e7eb")
        ax.text(0.06, 0.19, str(leader.get("direction", ""))[:170], fontsize=10, color="#bae6fd")
    ax.text(0.04, 0.08, NOT_CAUSAL, fontsize=11, color="#fbbf24")
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(output)

"""Build Excel and HTML reports for analysis results."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


def build_excel_report(
    merged_df: pd.DataFrame,
    column_profile: pd.DataFrame,
    factor_result: pd.DataFrame,
    pair_result: pd.DataFrame | None = None,
    pairwise_meta: dict | None = None,
    output_path: str | Path | None = None,
) -> bytes:
    """Create an Excel workbook with preview, profile, factor, and pairwise results."""
    merged_df = merged_df if isinstance(merged_df, pd.DataFrame) else pd.DataFrame()
    column_profile = column_profile if isinstance(column_profile, pd.DataFrame) else pd.DataFrame()
    factor_result = factor_result if isinstance(factor_result, pd.DataFrame) else pd.DataFrame()
    pair_result = pair_result if isinstance(pair_result, pd.DataFrame) else pd.DataFrame()
    pairwise_meta = pairwise_meta or {}

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Always create a visible first sheet. OpenPyXL raises "At least one sheet
        # must be visible" if every optional result frame is empty or unavailable
        # during Streamlit reruns.
        pd.DataFrame(
            [
                {
                    "report": "Process Factor Finder",
                    "note": "This result is a priority list for review, not a confirmed root cause.",
                }
            ]
        ).to_excel(writer, sheet_name="report_info", index=False)
        merged_df.head(1000).to_excel(writer, sheet_name="merged_preview", index=False)
        column_profile.to_excel(writer, sheet_name="column_profile", index=False)
        factor_result.to_excel(writer, sheet_name="factor_result", index=False)
        if not pair_result.empty:
            pd.DataFrame([pairwise_meta]).to_excel(writer, sheet_name="pairwise_summary", index=False)
            pair_result.to_excel(writer, sheet_name="pairwise_result", index=False)

    data = buffer.getvalue()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def build_html_report(
    y_col: str,
    merge_stats: dict,
    factor_result: pd.DataFrame,
    conclusion: str,
    pair_result: pd.DataFrame | None = None,
    pairwise_meta: dict | None = None,
    selected_pair_interpretation: str = "",
    output_path: str | Path | None = None,
) -> str:
    """Create a standalone HTML report for sharing or archiving."""
    factor_table = (
        factor_result.head(20).to_html(index=False, escape=False, classes="pff-table")
        if factor_result is not None and not factor_result.empty
        else "<p>단일 Y 분석 결과가 없습니다.</p>"
    )
    stat_rows = "".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>"
        for key, value in merge_stats.items()
        if key not in {"x_keys", "y_keys"}
    )

    pairwise_meta = pairwise_meta or {}
    pair_section = ""
    if pair_result is not None and not pair_result.empty:
        pair_table = pair_result.head(20).to_html(index=False, escape=False, classes="pff-table")
        pair_section = f"""
    <div class="card">
      <h2>전체 X-Y 비교 요약</h2>
      <p>X 후보 수: <span class="accent">{pairwise_meta.get('x_candidate_count', '-')}</span></p>
      <p>Y 후보 수: <span class="accent">{pairwise_meta.get('y_candidate_count', '-')}</span></p>
      <p>총 X-Y 조합 수: <span class="accent">{pairwise_meta.get('total_pairs', '-')}</span></p>
      <p>많은 조합을 동시에 비교했기 때문에 보정 p-value를 함께 제공합니다.</p>
    </div>
    <div class="card">
      <h2>Top 20 유의쌍</h2>
      {pair_table}
    </div>
    <div class="card">
      <h2>주요 pair 쉬운 해석</h2>
      <p>{selected_pair_interpretation or '선택된 pair 해석이 없습니다.'}</p>
    </div>
"""

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>Process Factor Finder Report</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      background: #0b1020;
      color: #e5e7eb;
      font-family: Arial, 'Malgun Gothic', sans-serif;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    .card {{
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 16px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.28);
      padding: 24px;
      margin-bottom: 20px;
    }}
    h1, h2 {{ margin: 0 0 16px; }}
    .accent {{ color: #22d3ee; }}
    .notice {{ color: #facc15; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid rgba(148, 163, 184, 0.18); padding: 10px; text-align: left; }}
    th {{ color: #c4b5fd; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Process Factor Finder Report</h1>
      <p>분석 대상 Y: <span class="accent">{y_col}</span></p>
      <p class="notice">본 결과는 확정 판단이 아니라 우선 확인할 유의 후보를 제시합니다. 최종 판단과 QA는 사용자가 수행해야 합니다.</p>
    </div>
    <div class="card">
      <h2>한 줄 결론</h2>
      <p>{conclusion}</p>
    </div>
    <div class="card">
      <h2>병합 통계</h2>
      <table>{stat_rows}</table>
    </div>
    <div class="card">
      <h2>Top Factor</h2>
      {factor_table}
    </div>
    {pair_section}
    <div class="card">
      <h2>주의사항</h2>
      <p>전체 X-Y 비교는 많은 변수 조합을 동시에 탐색하므로, 우연히 유의하게 보이는 결과가 포함될 수 있습니다.</p>
      <p>따라서 보정 p-value, 효과 크기, 반복 안정성, 데이터 품질을 함께 확인해야 합니다. 본 결과는 확정 판단이 아니라 우선 확인할 후보 쌍입니다.</p>
    </div>
  </div>
</body>
</html>"""

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    return html


def one_line_conclusion(result_df: pd.DataFrame, y_col: str) -> str:
    """Return a concise conclusion text."""
    if result_df is None or result_df.empty:
        return "아직 유의 후보 인자 결과가 없습니다."
    top = result_df.iloc[0]
    return (
        f"{y_col} 변동 후보 1순위는 {top['feature']}이며, "
        f"최종 점수 {top['final_score']:.1f}점({top['judgement']})입니다. "
        f"{top['direction']} 확정 판단이 아니라 우선 확인 후보입니다."
    )

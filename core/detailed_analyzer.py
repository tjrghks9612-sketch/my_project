"""Detailed candidate analysis using original data."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from core.models import AnalysisPlan
from core.scoring import benjamini_hochberg, calculate_final_score, validate_result_scores

ErrorCallback = Callable[[dict], None]


def _drop_pair_na(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return df[[x_col, y_col]].dropna()


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    if frame.empty or frame["group"].nunique() < 2:
        return 0.0
    overall = float(frame["value"].mean())
    total_ss = float(((frame["value"] - overall) ** 2).sum())
    if total_ss <= 0:
        return 0.0
    grouped = frame.groupby("group")["value"]
    between = float(sum(len(group) * (float(group.mean()) - overall) ** 2 for _, group in grouped))
    return max(0.0, min(1.0, between / total_ss))


def _anova_p(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": groups}).dropna()
    arrays = [group["value"].to_numpy(dtype=float) for _, group in frame.groupby("group") if len(group) >= 2]
    if len(arrays) < 2:
        return 1.0
    try:
        return float(stats.f_oneway(*arrays).pvalue)
    except Exception:
        return 1.0


def _cramers_v(table: pd.DataFrame) -> tuple[float, float, float]:
    if table.empty or table.to_numpy().sum() <= 0 or min(table.shape) < 2:
        return 0.0, 0.0, 1.0
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    denom = n * max(min(table.shape) - 1, 1)
    cramer = float(np.sqrt(chi2 / denom)) if denom else 0.0
    return cramer, float(chi2), float(p_value)


def _direction_text(pair_type: str, x_col: str, y_col: str, frame: pd.DataFrame, effect: float) -> str:
    if frame.empty:
        return "표본이 부족해 방향성을 해석하기 어렵습니다."
    if pair_type == "numeric_numeric":
        return f"{x_col} 값이 증가할 때 {y_col}도 함께 움직이는 패턴이 관찰됩니다." if effect >= 0 else f"{x_col} 값이 증가할 때 {y_col}가 반대로 움직이는 패턴이 관찰됩니다."
    if pair_type in {"categorical_numeric", "numeric_categorical"}:
        return f"{x_col} 조건에 따라 {y_col} 분포 차이가 관찰됩니다."
    return f"{x_col}와 {y_col}의 범주 조합이 독립적이지 않은 패턴을 보입니다."


def _interpretation(pair_type: str, x_col: str, y_col: str, final_score: float) -> str:
    explanations = {
        "numeric_numeric": "두 숫자형 값이 함께 증가하거나 감소하는 패턴을 확인했습니다.",
        "categorical_numeric": "X의 그룹에 따라 Y의 평균/분포 차이가 있는지 확인했습니다.",
        "numeric_categorical": "Y의 판정/범주 그룹에 따라 X 값의 분포가 달라지는지 확인했습니다.",
        "categorical_categorical": "두 범주형 변수의 조합이 독립적이지 않은지 확인했습니다.",
    }
    return (
        f"{x_col} → {y_col} 조합은 final score {final_score:.1f}점으로 우선 확인 후보입니다. "
        f"{explanations.get(pair_type, '')} 본 결과는 공정 원인 확정이 아니라 정밀 확인 대상을 줄이기 위한 탐색 결과입니다."
    )


def analyze_numeric_numeric(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    pair = _drop_pair_na(df, x_col, y_col)
    if len(pair) < 3:
        raise ValueError("numeric-numeric 표본 부족")
    x = pd.to_numeric(pair[x_col], errors="coerce").astype(float)
    y = pd.to_numeric(pair[y_col], errors="coerce").astype(float)
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or x.nunique() <= 1 or y.nunique() <= 1:
        raise ValueError("numeric-numeric 유효 분산 부족")
    corr, p_value = stats.pearsonr(x, y)
    return {
        "p_value": float(p_value),
        "effect_size": float(abs(corr)),
        "r2_score": float(corr * corr),
        "eta_squared": np.nan,
        "cramer_v": np.nan,
        "model_score": float(min(100.0, abs(corr) * 100.0)),
        "sample_n": int(len(x)),
        "direction_effect": float(corr),
    }


def analyze_categorical_numeric(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    pair = _drop_pair_na(df, x_col, y_col)
    if len(pair) < 3 or pair[x_col].nunique() < 2:
        raise ValueError("categorical-numeric 표본 부족")
    eta = _eta_squared(pair[y_col], pair[x_col].astype(str))
    p_value = _anova_p(pair[y_col], pair[x_col].astype(str))
    return {
        "p_value": p_value,
        "effect_size": float(eta),
        "r2_score": np.nan,
        "eta_squared": float(eta),
        "cramer_v": np.nan,
        "model_score": float(min(100.0, eta / 0.20 * 100.0)),
        "sample_n": int(len(pair)),
        "direction_effect": float(eta),
    }


def analyze_numeric_categorical(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    pair = _drop_pair_na(df, x_col, y_col)
    if len(pair) < 3 or pair[y_col].nunique() < 2:
        raise ValueError("numeric-categorical 표본 부족")
    eta = _eta_squared(pair[x_col], pair[y_col].astype(str))
    p_value = _anova_p(pair[x_col], pair[y_col].astype(str))
    point = np.nan
    if pair[y_col].nunique() == 2:
        codes = pd.Categorical(pair[y_col]).codes
        try:
            point, p_value = stats.pointbiserialr(codes, pd.to_numeric(pair[x_col], errors="coerce"))
            eta = max(eta, float(point * point))
        except Exception:
            point = np.nan
    effect = float(abs(point)) if np.isfinite(point) else float(eta)
    return {
        "p_value": float(p_value),
        "effect_size": effect,
        "r2_score": np.nan,
        "eta_squared": float(eta),
        "cramer_v": np.nan,
        "point_biserial": float(point) if np.isfinite(point) else np.nan,
        "model_score": float(min(100.0, effect / 0.20 * 100.0)),
        "sample_n": int(len(pair)),
        "direction_effect": effect,
    }


def analyze_categorical_categorical(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    pair = _drop_pair_na(df, x_col, y_col)
    if len(pair) < 3 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        raise ValueError("categorical-categorical 표본 부족")
    table = pd.crosstab(pair[x_col].astype(str), pair[y_col].astype(str))
    cramer, chi2, p_value = _cramers_v(table)
    return {
        "p_value": float(p_value),
        "effect_size": float(cramer),
        "r2_score": np.nan,
        "eta_squared": np.nan,
        "cramer_v": float(cramer),
        "chi2": float(chi2),
        "model_score": float(min(100.0, cramer / 0.50 * 100.0)),
        "sample_n": int(len(pair)),
        "direction_effect": float(cramer),
    }


ANALYZERS = {
    "numeric_numeric": analyze_numeric_numeric,
    "categorical_numeric": analyze_categorical_numeric,
    "numeric_categorical": analyze_numeric_categorical,
    "categorical_categorical": analyze_categorical_categorical,
}


def analyze_candidates(
    df: pd.DataFrame,
    candidates: pd.DataFrame,
    plan: AnalysisPlan,
    config: dict,
    error_callback: ErrorCallback | None = None,
) -> pd.DataFrame:
    """Run detailed analysis for top screening candidates and score them."""
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    selected = candidates.sort_values("screening_score", ascending=False).head(int(plan.detailed_top_n)).copy()
    rows: list[dict] = []
    for _, candidate in selected.iterrows():
        x_col = str(candidate["x_col"])
        y_col = str(candidate["y_col"])
        pair_type = str(candidate["pair_type"])
        try:
            analyzer = ANALYZERS[pair_type]
            metrics = analyzer(df, x_col, y_col)
            scores = calculate_final_score(
                pair_type=pair_type,
                p_value=float(metrics["p_value"]),
                effect_size_value=float(metrics["effect_size"]),
                model_score=float(metrics["model_score"]),
                n_samples=int(metrics["sample_n"]),
                x_missing_rate=float(candidate.get("x_missing_rate", df[x_col].isna().mean())),
                y_missing_rate=float(candidate.get("y_missing_rate", df[y_col].isna().mean())),
                config=config,
            )
            row = {
                "rank": 0,
                "pair_type": pair_type,
                "x_col": x_col,
                "y_col": y_col,
                "screening_method": str(candidate.get("screening_method", "")),
                "screening_score": float(candidate.get("screening_score", 0.0)),
                "p_value": float(metrics["p_value"]),
                "adjusted_p_value": np.nan,
                "effect_size": float(metrics["effect_size"]),
                "model_score": float(scores["model_score"]),
                "r2_score": metrics.get("r2_score", np.nan),
                "eta_squared": metrics.get("eta_squared", np.nan),
                "cramer_v": metrics.get("cramer_v", np.nan),
                "sample_n": int(metrics["sample_n"]),
                "stat_score": scores["stat_score"],
                "effect_score": scores["effect_score"],
                "stability_score": scores["stability_score"],
                "quality_score": scores["quality_score"],
                "final_score": scores["final_score"],
                "direction": "",
                "interpretation": "",
                "caution": "우선 확인 후보이며 인과관계 확정은 아닙니다.",
            }
            pair_frame = _drop_pair_na(df, x_col, y_col)
            row["direction"] = _direction_text(pair_type, x_col, y_col, pair_frame, float(metrics.get("direction_effect", 0.0)))
            row["interpretation"] = _interpretation(pair_type, x_col, y_col, row["final_score"])
            rows.append(row)
        except Exception as exc:
            if error_callback:
                error_callback({"phase": "detailed", "x_col": x_col, "y_col": y_col, "pair_type": pair_type, "error": repr(exc)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["adjusted_p_value"] = benjamini_hochberg(result["p_value"])
    valid_rows = []
    for _, row in result.iterrows():
        row_dict = row.to_dict()
        ok, reason = validate_result_scores(row_dict, config)
        if ok:
            valid_rows.append(row_dict)
        elif error_callback:
            error_callback({"phase": "validation", "x_col": row_dict.get("x_col"), "y_col": row_dict.get("y_col"), "pair_type": row_dict.get("pair_type"), "error": reason})
    final = pd.DataFrame(valid_rows)
    if final.empty:
        return final
    final = final.sort_values(["final_score", "effect_score", "model_score"], ascending=False).reset_index(drop=True)
    final["rank"] = np.arange(1, len(final) + 1)
    return final


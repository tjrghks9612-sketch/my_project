"""Form helper components."""

from __future__ import annotations

import re
from collections.abc import Iterable


def parse_column_text(text: str) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,\n\r\t ]+", text) if item.strip()]


def _compile_regex(pattern: str, label: str) -> tuple[re.Pattern[str] | None, str | None]:
    if not pattern:
        return None, None
    try:
        return re.compile(pattern, flags=re.IGNORECASE), None
    except re.error:
        return None, f"{label} regex 형식이 올바르지 않아 해당 조건은 무시했습니다."


def filter_columns(
    columns: Iterable[str],
    include_regex: str = "",
    exclude_regex: str = "",
    pasted: str = "",
    label: str = "컬럼",
) -> tuple[list[str], list[str]]:
    """Filter columns safely without letting invalid regex stop the app."""
    selected = list(columns)
    warnings: list[str] = []
    include_pattern, include_warning = _compile_regex(include_regex, f"{label} include")
    exclude_pattern, exclude_warning = _compile_regex(exclude_regex, f"{label} exclude")
    if include_warning:
        warnings.append(include_warning)
    elif include_pattern is not None:
        selected = [col for col in selected if include_pattern.search(col)]
    if exclude_warning:
        warnings.append(exclude_warning)
    elif exclude_pattern is not None:
        selected = [col for col in selected if not exclude_pattern.search(col)]
    pasted_cols = parse_column_text(pasted)
    if pasted_cols:
        pasted_set = set(pasted_cols)
        selected = [col for col in selected if col in pasted_set]
    return selected, warnings

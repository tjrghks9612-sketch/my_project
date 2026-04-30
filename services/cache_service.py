"""Small cache helpers independent from Streamlit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_key(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


"""Run artifact path management."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.plot_engine import sanitize_filename


class ArtifactManager:
    """Create and own output folders for one analysis run."""

    def __init__(self, output_root: str | Path = "output") -> None:
        self.output_root = Path(output_root)

    def create_run(self) -> dict:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_run")
        run_dir = self.output_root / run_id
        paths = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "graphs_dir": str(run_dir / "graphs"),
            "captures_dir": str(run_dir / "captures"),
            "temp_dir": str(run_dir / "temp"),
            "logs_dir": str(run_dir / "logs"),
            "error_log": str(run_dir / "logs" / "error_log.csv"),
            "screening_log": str(run_dir / "logs" / "screening_log.csv"),
        }
        for key in ("graphs_dir", "captures_dir", "temp_dir", "logs_dir"):
            Path(paths[key]).mkdir(parents=True, exist_ok=True)
        return paths

    @staticmethod
    def safe_name(text: str) -> str:
        return sanitize_filename(text)


def append_error_log(log_path: str | Path, errors: list[dict]) -> None:
    """Append graph/save errors to the run error log."""
    if not errors:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().isoformat(timespec="seconds")
    rows = []
    for error in errors:
        rows.append(
            {
                "created_at": created_at,
                "phase": error.get("phase", "-"),
                "x_col": error.get("x_col", "-"),
                "y_col": error.get("y_col", "-"),
                "pair_type": error.get("pair_type", "-"),
                "error": error.get("error", "-"),
            }
        )
    frame = pd.DataFrame(rows, columns=["created_at", "phase", "x_col", "y_col", "pair_type", "error"])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")

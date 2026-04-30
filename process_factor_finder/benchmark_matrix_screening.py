"""Benchmark matrix-based numeric-numeric screening.

Default shape is intentionally modest. Increase arguments when the machine has
enough memory, for example:

python benchmark_matrix_screening.py --rows 5000 --x-cols 20000 --y-cols 10
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_engine import load_config, matrix_numeric_screening


def _memory_mb() -> float | None:
    """Return current process memory in MB when psutil is available."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--x-cols", type=int, default=5000)
    parser.add_argument("--y-cols", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--x-chunk", type=int, default=5000)
    parser.add_argument("--y-chunk", type=int, default=50)
    parser.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    rng = np.random.default_rng(20260427)
    started = time.time()
    x = rng.normal(size=(args.rows, args.x_cols)).astype(args.dtype)
    y = rng.normal(size=(args.rows, args.y_cols)).astype(args.dtype)

    if args.x_cols >= 5 and args.y_cols >= 1:
        y[:, 0] = x[:, 0] * 0.9 + rng.normal(scale=0.05, size=args.rows)

    df = pd.concat(
        [
            pd.DataFrame(x, columns=[f"x{i:05d}" for i in range(args.x_cols)]),
            pd.DataFrame(y, columns=[f"y{i:03d}" for i in range(args.y_cols)]),
        ],
        axis=1,
    )
    generation_seconds = time.time() - started

    config = load_config("config.yaml")
    config["matrix_screening"] = dict(config.get("matrix_screening", {}))
    config["matrix_screening"].update(
        {
            "enabled": True,
            "method": args.method,
            "dtype": args.dtype,
            "top_n_per_y": args.top_n,
            "x_chunk_size": args.x_chunk,
            "y_chunk_size": args.y_chunk,
            "save_candidates": True,
        }
    )

    output_dir = Path("output") / "benchmarks" / "matrix_screening" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    before_mb = _memory_mb()
    screening_started = time.time()
    candidates, dropped, log_df, fallback = matrix_numeric_screening(
        df,
        [f"x{i:05d}" for i in range(args.x_cols)],
        [f"y{i:03d}" for i in range(args.y_cols)],
        config=config,
        output_dir=output_dir,
    )
    screening_seconds = time.time() - screening_started
    after_mb = _memory_mb()

    result = {
        "rows": args.rows,
        "x_cols": args.x_cols,
        "y_cols": args.y_cols,
        "total_pairs": args.x_cols * args.y_cols,
        "method": args.method,
        "dtype": args.dtype,
        "top_n_per_y": args.top_n,
        "x_chunk": args.x_chunk,
        "y_chunk": args.y_chunk,
        "generation_seconds": round(generation_seconds, 3),
        "screening_seconds": round(screening_seconds, 3),
        "candidates": int(len(candidates)),
        "dropped_columns": int(len(dropped)),
        "fallback": bool(fallback),
        "memory_before_mb": round(before_mb, 1) if before_mb is not None else "",
        "memory_after_mb": round(after_mb, 1) if after_mb is not None else "",
        "output_dir": str(output_dir.resolve()),
    }
    result_df = pd.DataFrame([result])
    result_df.to_csv(output_dir / "benchmark_result.csv", index=False)
    (output_dir / "benchmark_result.txt").write_text(result_df.to_string(index=False), encoding="utf-8")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()

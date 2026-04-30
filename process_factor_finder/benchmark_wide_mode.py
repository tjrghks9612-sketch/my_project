"""Benchmark large-mode pairwise analysis on the wide synthetic dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from analysis_engine import analyze_pairwise_chunked, estimate_pairwise_risk, load_config
from data_loader import merge_on_keys, read_single_file


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _merge_wide_x_parts(data_dir: Path) -> pd.DataFrame:
    """Merge all wide X part parquet files on SSN."""
    part_paths = sorted(data_dir.glob("wide_sample_x_part*.parquet"))
    if not part_paths:
        raise FileNotFoundError("wide_sample_x_part*.parquet files were not found.")

    merged = read_single_file(part_paths[0])
    for path in part_paths[1:]:
        frame = read_single_file(path)
        merged = pd.merge(merged, frame, on="SSN", how="inner")
    return merged


def main() -> None:
    """Run a reproducible wide-data benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--y-cols", nargs="+", default=["DPU"], help="Y columns to analyze.")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "output" / "benchmarks" / "wide_runs"))
    parser.add_argument("--chunk-x", type=int, default=500)
    parser.add_argument("--chunk-y", type=int, default=4)
    parser.add_argument("--screen-top-k", type=int, default=8)
    parser.add_argument("--screen-max-total", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    x_df = _merge_wide_x_parts(DATA_DIR)
    y_df = read_single_file(DATA_DIR / "wide_sample_y.parquet")
    merged_df, stats = merge_on_keys(x_df, y_df, ["SSN"], ["SSN"])

    config = load_config(BASE_DIR / "config.yaml")
    config["pairwise_analysis"]["chunk_x_size"] = args.chunk_x
    config["pairwise_analysis"]["chunk_y_size"] = args.chunk_y
    config["pairwise_analysis"]["screening_top_k_per_y"] = args.screen_top_k
    config["pairwise_analysis"]["screening_max_pairs_total"] = args.screen_max_total
    config["pairwise_analysis"]["downcast_float32"] = True

    x_cols = [column for column in merged_df.columns if column != "SSN" and column not in args.y_cols]
    total_pairs = len(x_cols) * len(args.y_cols)
    print(f"Merged shape: {merged_df.shape}")
    print(f"Target Y columns: {args.y_cols}")
    print(f"Estimated total pairs: {total_pairs:,}")
    print(f"Risk: {estimate_pairwise_risk(total_pairs)}")

    output_dir = Path(args.output_dir) / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_pairwise_chunked(
        merged_df,
        x_cols=x_cols,
        y_cols=args.y_cols,
        key_cols=["SSN"],
        config=config,
        output_dir=output_dir,
    )

    elapsed = time.time() - started
    print(f"Elapsed seconds: {elapsed:.1f}")
    print(f"Detailed result rows: {len(result):,}")
    if not result.empty:
        print(result[["rank", "x_feature", "y_target", "final_score"]].head(10).to_string(index=False))
    print(f"Output dir: {output_dir}")
    meta = result.attrs.get("pairwise_meta", {})
    for key, value in meta.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

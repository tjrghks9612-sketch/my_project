"""Generate wide synthetic process X / quality Y data for large-mode evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert linear scores to probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def generate_wide_data(
    n_rows: int = 5000,
    total_x_columns: int = 20000,
    x_parts: int = 10,
    seed: int = 20260427,
    output_dir: str | Path = DATA_DIR,
) -> dict[str, str]:
    """Create 5000x20000-scale synthetic data split across multiple X parquet files."""
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if total_x_columns % x_parts != 0:
        raise ValueError("total_x_columns must be divisible by x_parts.")

    cols_per_part = total_x_columns // x_parts
    ssn = np.array([f"WIDE_SSN_{index:07d}" for index in range(n_rows)])

    slot = rng.integers(1, 13, size=n_rows)
    chamber = rng.choice(["CH1", "CH2", "CH3", "CH4"], size=n_rows, p=[0.24, 0.29, 0.24, 0.23])
    recipe = rng.choice(["RCP_A", "RCP_B", "RCP_C"], size=n_rows, p=[0.45, 0.35, 0.20])
    etch_time = rng.normal(60.0, 5.0, size=n_rows).astype(np.float32)
    anneal_temp = rng.normal(505.0, 15.0, size=n_rows).astype(np.float32)
    o2_flow = rng.normal(82.0, 7.5, size=n_rows).astype(np.float32)
    pressure = rng.normal(1.52, 0.11, size=n_rows).astype(np.float32)

    edge_slot = np.isin(slot, [1, 2, 11, 12]).astype(np.float32)
    chamber_ch2 = (chamber == "CH2").astype(np.float32)
    recipe_shift = pd.Series(recipe).map({"RCP_A": -0.10, "RCP_B": 0.18, "RCP_C": 0.52}).to_numpy(dtype=np.float32)

    dpu_lambda = np.clip(1.2 + 2.6 * edge_slot + 2.0 * chamber_ch2 + rng.normal(0, 0.35, size=n_rows), 0.15, None)
    dpu = rng.poisson(dpu_lambda).astype(np.float32)
    cd = (55.0 + 0.48 * (etch_time - 60.0) + rng.normal(0, 0.7, size=n_rows)).astype(np.float32)
    taper = (86.0 - 0.058 * (anneal_temp - 505.0) + rng.normal(0, 0.7, size=n_rows)).astype(np.float32)
    flicker_prob = _sigmoid(-1.4 + 0.13 * (78.0 - o2_flow) + 0.12 * (chamber == "CH3"))
    flicker_ng = rng.binomial(1, np.clip(flicker_prob, 0.02, 0.80)).astype(np.int8)
    brightness_drift = (0.92 * (pressure - 1.52) + recipe_shift + rng.normal(0, 0.12, size=n_rows)).astype(np.float32)

    special_features = {
        "Slot": slot,
        "Chamber": chamber,
        "Recipe": recipe,
        "Etch_Time": etch_time,
        "Anneal_Temp": anneal_temp,
        "O2_Flow": o2_flow,
        "Pressure": pressure,
        "Signal_DPU_Proxy": (0.9 * edge_slot + 0.8 * chamber_ch2 + rng.normal(0, 0.08, size=n_rows)).astype(np.float32),
        "Signal_CD_Proxy": (etch_time * 0.75 + rng.normal(0, 0.5, size=n_rows)).astype(np.float32),
        "Signal_Taper_Proxy": (-anneal_temp * 0.04 + rng.normal(0, 0.4, size=n_rows)).astype(np.float32),
        "Signal_Flicker_Proxy": ((85.0 - o2_flow) + rng.normal(0, 0.35, size=n_rows)).astype(np.float32),
        "Signal_Brightness_Proxy": (pressure * 2.0 + recipe_shift + rng.normal(0, 0.08, size=n_rows)).astype(np.float32),
        "Mostly_Missing_01": np.where(rng.random(n_rows) < 0.88, np.nan, rng.normal(size=n_rows)).astype(np.float32),
        "Constant_01": np.full(n_rows, 1.0, dtype=np.float32),
    }
    special_names = list(special_features.keys())
    special_count = len(special_names)

    created_files: dict[str, str] = {}
    running_feature_index = 0
    for part_index in range(x_parts):
        part_columns: dict[str, object] = {"SSN": ssn}
        if part_index == 0:
            for name in special_names:
                part_columns[name] = special_features[name]
            noise_count = cols_per_part - special_count
        else:
            noise_count = cols_per_part

        if noise_count < 0:
            raise ValueError("Part 1 column budget is smaller than the number of special features.")

        noise_block = rng.normal(0, 1, size=(n_rows, noise_count)).astype(np.float32)
        for local_index in range(noise_count):
            running_feature_index += 1
            column_name = f"XW_{running_feature_index:05d}"
            part_columns[column_name] = noise_block[:, local_index]

        part_df = pd.DataFrame(part_columns)
        file_name = f"wide_sample_x_part{part_index + 1:02d}.parquet"
        part_df.to_parquet(output_path / file_name, index=False)
        created_files[file_name] = str(output_path / file_name)

    y_df = pd.DataFrame(
        {
            "SSN": ssn,
            "DPU": dpu,
            "CD": cd,
            "Taper": taper,
            "Flicker_NG": flicker_ng,
            "Brightness_Drift": brightness_drift,
        }
    )
    y_name = "wide_sample_y.parquet"
    y_df.to_parquet(output_path / y_name, index=False)
    created_files[y_name] = str(output_path / y_name)

    manifest = pd.DataFrame(
        {
            "file_name": list(created_files.keys()),
            "path": list(created_files.values()),
            "rows": [n_rows] * len(created_files),
        }
    )
    manifest_name = "wide_sample_manifest.csv"
    manifest.to_csv(output_path / manifest_name, index=False, encoding="utf-8-sig")
    created_files[manifest_name] = str(output_path / manifest_name)
    return created_files


if __name__ == "__main__":
    files = generate_wide_data()
    print("Generated wide dummy data files:")
    for name, path in files.items():
        print(f"- {name}: {path}")

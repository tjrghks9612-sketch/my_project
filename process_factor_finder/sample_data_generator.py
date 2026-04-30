"""Create synthetic manufacturing X/Y data for Process Factor Finder.

The generated data is fake by design. It contains planted relationships so
users can verify that the analysis app finds the expected candidate factors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Convert any numeric array into probabilities between 0 and 1."""
    return 1 / (1 + np.exp(-value))


def _random_dates(
    rng: np.random.Generator,
    start: str,
    days: int,
    size: int,
) -> pd.Series:
    """Return random dates from a fixed date window."""
    base = pd.Timestamp(start)
    offsets = rng.integers(0, days, size=size)
    return pd.Series(base + pd.to_timedelta(offsets, unit="D"))


def _add_missing_values(
    df: pd.DataFrame,
    rng: np.random.Generator,
    columns: list[str],
    ratio: float,
) -> pd.DataFrame:
    """Inject a small amount of missing values into selected columns."""
    result = df.copy()
    for column in columns:
        mask = rng.random(len(result)) < ratio
        result.loc[mask, column] = np.nan
    return result


def _add_outliers(
    df: pd.DataFrame,
    rng: np.random.Generator,
    column: str,
    ratio: float,
    shift: float,
) -> pd.DataFrame:
    """Add simple high-side outliers to make the sample feel more realistic."""
    result = df.copy()
    mask = rng.random(len(result)) < ratio
    result.loc[mask, column] = result.loc[mask, column] + shift
    return result


def generate_sample_data(
    n_rows: int = 1500,
    seed: int = 42,
    output_dir: Path | str = DATA_DIR,
    x_filename: str = "sample_x.csv",
    y_filename: str = "sample_y.csv",
    ssn_start: int = 2026000000,
    process_start: str = "2026-01-01",
    lot_min: int = 1,
    lot_max: int = 86,
    panel_max: int = 700,
    product_labels: tuple[str, str, str] = ("OLED_A", "OLED_B", "OLED_C"),
    eqp_labels: tuple[str, ...] = ("EQP01", "EQP02", "EQP03", "EQP04", "EQP05"),
    y_only_prefix: str = "SSN_YONLY",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate fake process X data and inspection Y data.

    Planted patterns:
    - DPU increases at edge Slots and in Chamber CH2.
    - CD increases as Etch_Time becomes longer.
    - Taper decreases as Anneal_Temp becomes higher.
    - Flicker_NG increases when O2_Flow is lower.
    - Brightness_Drift is affected by Pressure and Recipe.
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ssn = np.array([f"SSN{ssn_start + i:010d}" for i in range(n_rows)])
    lot_ids = np.array([f"LOT{v:04d}" for v in rng.integers(lot_min, lot_max, size=n_rows)])
    panel_ids = np.array([f"PNL{v:05d}" for v in rng.integers(1, panel_max, size=n_rows)])
    products = rng.choice(list(product_labels), size=n_rows, p=[0.45, 0.35, 0.20])
    configs = rng.choice(["CFG_STD", "CFG_HIGH", "CFG_LOW"], size=n_rows, p=[0.58, 0.25, 0.17])
    chambers = rng.choice(["CH1", "CH2", "CH3", "CH4"], size=n_rows, p=[0.26, 0.27, 0.24, 0.23])
    recipes = rng.choice(["RCP_A", "RCP_B", "RCP_C"], size=n_rows, p=[0.48, 0.32, 0.20])
    slots = rng.integers(1, 13, size=n_rows)

    recipe_temp_shift = pd.Series(recipes).map({"RCP_A": 0, "RCP_B": 8, "RCP_C": -5}).to_numpy()
    recipe_pressure_shift = pd.Series(recipes).map({"RCP_A": 0.00, "RCP_B": 0.08, "RCP_C": -0.05}).to_numpy()

    anneal_temp = rng.normal(505, 16, size=n_rows) + recipe_temp_shift
    o2_flow = rng.normal(82, 8, size=n_rows)
    etch_time = rng.normal(60, 5.5, size=n_rows)
    pressure = rng.normal(1.52, 0.12, size=n_rows) + recipe_pressure_shift
    rf_power = rng.normal(455, 35, size=n_rows)
    gas_ratio = rng.normal(1.03, 0.08, size=n_rows)
    thickness_pre = rng.normal(710, 24, size=n_rows)
    cd_pre = rng.normal(51.5, 1.9, size=n_rows)

    x_df = pd.DataFrame(
        {
            "SSN": ssn,
            "LotID": lot_ids,
            "PanelID": panel_ids,
            "Product": products,
            "Config": configs,
            "ProcessDate": _random_dates(rng, process_start, 90, n_rows).dt.strftime("%Y-%m-%d"),
            "EqpID": rng.choice(list(eqp_labels), size=n_rows),
            "Chamber": chambers,
            "Slot": slots,
            "Recipe": recipes,
            "Anneal_Temp": np.round(anneal_temp, 2),
            "O2_Flow": np.round(o2_flow, 2),
            "Etch_Time": np.round(etch_time, 2),
            "Pressure": np.round(pressure, 4),
            "RF_Power": np.round(rf_power, 2),
            "Gas_Ratio": np.round(gas_ratio, 4),
            "Thickness_Pre": np.round(thickness_pre, 2),
            "CD_Pre": np.round(cd_pre, 3),
        }
    )

    edge_slot = np.isin(slots, [1, 2, 11, 12]).astype(float)
    chamber_ch2 = (chambers == "CH2").astype(float)
    recipe_brightness = pd.Series(recipes).map({"RCP_A": -0.10, "RCP_B": 0.18, "RCP_C": 0.55}).to_numpy()

    dpu_lambda = np.clip(1.4 + 2.7 * edge_slot + 2.1 * chamber_ch2 + rng.normal(0, 0.35, n_rows), 0.2, None)
    dpu = rng.poisson(dpu_lambda).astype(float)
    cd = 55.0 + 0.46 * (etch_time - 60) + 0.25 * (cd_pre - 51.5) + rng.normal(0, 0.85, n_rows)
    thk = thickness_pre + 0.14 * (anneal_temp - 505) - 7.0 * (pressure - 1.52) + rng.normal(0, 6.0, n_rows)
    taper = 86.0 - 0.060 * (anneal_temp - 505) + rng.normal(0, 0.75, n_rows)
    brightness_drift = 0.95 * (pressure - 1.52) + recipe_brightness + rng.normal(0, 0.13, n_rows)

    flicker_probability = _sigmoid(-1.55 + 0.145 * (78 - o2_flow) + 0.18 * (chambers == "CH3"))
    flicker_ng = rng.binomial(1, np.clip(flicker_probability, 0.02, 0.82)).astype(int)

    defect_mode = np.full(n_rows, "Normal", dtype=object)
    defect_mode[(edge_slot == 1) & (dpu >= 5)] = "Particle"
    defect_mode[flicker_ng == 1] = "Flicker"
    defect_mode[brightness_drift > 0.55] = "Brightness"
    random_defect_mask = rng.random(n_rows) < 0.035
    defect_mode[random_defect_mask] = rng.choice(["Scratch", "Particle", "Normal"], size=random_defect_mask.sum())

    y_all = pd.DataFrame(
        {
            "SSN": ssn,
            "InspectDate": (
                pd.to_datetime(x_df["ProcessDate"]) + pd.to_timedelta(rng.integers(1, 8, size=n_rows), unit="D")
            ).dt.strftime("%Y-%m-%d"),
            "DPU": dpu,
            "CD": np.round(cd, 3),
            "THK": np.round(thk, 3),
            "Taper": np.round(taper, 3),
            "Brightness_Drift": np.round(brightness_drift, 4),
            "Flicker_NG": flicker_ng,
            "Defect_Mode": defect_mode,
        }
    )

    matched_mask = rng.random(n_rows) < 0.985
    y_df = y_all.loc[matched_mask].copy()

    extra_count = max(6, int(n_rows * 0.006))
    extra_y = y_df.sample(extra_count, random_state=seed).copy()
    extra_y["SSN"] = [f"{y_only_prefix}_{i:04d}" for i in range(extra_count)]
    extra_y["DPU"] = rng.poisson(2.2, size=extra_count)
    extra_y["Flicker_NG"] = rng.binomial(1, 0.18, size=extra_count)
    y_df = pd.concat([y_df, extra_y], ignore_index=True)

    x_df = _add_missing_values(
        x_df,
        rng,
        ["Chamber", "Recipe", "Anneal_Temp", "O2_Flow", "Etch_Time", "Pressure", "CD_Pre"],
        ratio=0.012,
    )
    y_df = _add_missing_values(
        y_df,
        rng,
        ["DPU", "CD", "THK", "Taper", "Brightness_Drift", "Defect_Mode"],
        ratio=0.010,
    )

    x_df = _add_outliers(x_df, rng, "Etch_Time", ratio=0.006, shift=24.0)
    x_df = _add_outliers(x_df, rng, "Pressure", ratio=0.005, shift=0.42)
    y_df = _add_outliers(y_df, rng, "DPU", ratio=0.006, shift=16.0)
    y_df = _add_outliers(y_df, rng, "CD", ratio=0.004, shift=8.0)

    x_df.to_csv(output_path / x_filename, index=False, encoding="utf-8-sig")
    y_df.to_csv(output_path / y_filename, index=False, encoding="utf-8-sig")
    return x_df, y_df


def generate_all_sample_files(output_dir: Path | str = DATA_DIR) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate base samples plus companion tables for key-based file merge tests."""
    x_df, y_df = generate_sample_data(
        n_rows=1500,
        seed=42,
        output_dir=output_dir,
        x_filename="sample_x.csv",
        y_filename="sample_y.csv",
        ssn_start=2026000000,
        process_start="2026-01-01",
        lot_min=1,
        lot_max=86,
        panel_max=700,
        product_labels=("OLED_A", "OLED_B", "OLED_C"),
        eqp_labels=("EQP01", "EQP02", "EQP03", "EQP04", "EQP05"),
        y_only_prefix="SSN_YONLY",
    )
    x2_df, y2_df = generate_companion_sample_data(x_df, output_dir=output_dir)
    return {"sample_1": (x_df, y_df), "sample_2": (x2_df, y2_df)}


def generate_companion_sample_data(
    base_x_df: pd.DataFrame,
    seed: int = 142,
    output_dir: Path | str = DATA_DIR,
    x_filename: str = "sample_x2.csv",
    y_filename: str = "sample_y2.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate X2/Y2 with the same key but different data items from X1/Y1.

    X2 represents auxiliary process/environment/equipment state data.
    Y2 represents final/reliability/appearance quality data. Except for SSN,
    the columns are intentionally different from sample_x.csv and sample_y.csv.
    """
    rng = np.random.default_rng(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_keys = base_x_df["SSN"].dropna().astype(str).to_numpy()
    x_keep = rng.random(len(base_keys)) < 0.975
    x_ssn = base_keys[x_keep]

    n_x = len(x_ssn)
    line_ids = rng.choice(["LINE_A", "LINE_B", "LINE_C"], size=n_x, p=[0.42, 0.34, 0.24])
    route_steps = rng.choice(["COAT", "EXPOSE", "DEVELOP", "CURE"], size=n_x, p=[0.25, 0.25, 0.25, 0.25])
    shift = rng.choice(["DAY", "SWING", "NIGHT"], size=n_x, p=[0.45, 0.25, 0.30])
    operator_group = rng.choice(["OP_A", "OP_B", "OP_C", "OP_D"], size=n_x)
    mask_id = np.array([f"MASK{value:03d}" for value in rng.integers(1, 38, size=n_x)])
    carrier_id = np.array([f"CST{value:04d}" for value in rng.integers(1, 260, size=n_x)])

    humidity = rng.normal(46, 7.5, size=n_x) + (line_ids == "LINE_C") * 5.0
    queue_time = rng.gamma(shape=2.3, scale=23.0, size=n_x) + (shift == "NIGHT") * 14.0
    pm_age = rng.integers(0, 42, size=n_x)
    developer_time = rng.normal(42, 4.2, size=n_x)
    vibration = np.clip(rng.normal(0.42, 0.13, size=n_x) + (line_ids == "LINE_B") * 0.10, 0.05, None)

    x2_df = pd.DataFrame(
        {
            "SSN": x_ssn,
            "RouteStep": route_steps,
            "LineID": line_ids,
            "Shift": shift,
            "OperatorGroup": operator_group,
            "MaskID": mask_id,
            "CarrierID": carrier_id,
            "AuxProcessDate": _random_dates(rng, "2026-02-15", 75, n_x).dt.strftime("%Y-%m-%d"),
            "PreBake_Temp": np.round(rng.normal(94, 3.1, size=n_x), 2),
            "PreBake_Time": np.round(rng.normal(118, 9.0, size=n_x), 2),
            "Clean_Chem_Conc": np.round(rng.normal(2.2, 0.22, size=n_x), 4),
            "Clean_Time": np.round(rng.normal(48, 5.5, size=n_x), 2),
            "DI_Rinse_Resistivity": np.round(rng.normal(17.8, 0.75, size=n_x), 3),
            "Ambient_Humidity": np.round(humidity, 2),
            "Coater_Speed": np.round(rng.normal(1350, 70, size=n_x), 2),
            "Developer_Temp": np.round(rng.normal(23.2, 0.8, size=n_x), 3),
            "Developer_Time": np.round(developer_time, 2),
            "Queue_Time_Min": np.round(queue_time, 2),
            "PM_Age_Days": pm_age,
            "Vibration_RMS": np.round(vibration, 4),
            "Facility_Exhaust": np.round(rng.normal(820, 55, size=n_x), 2),
        }
    )

    x_extra_count = max(5, int(len(base_keys) * 0.006))
    extra_x = x2_df.sample(x_extra_count, random_state=seed).copy()
    extra_x["SSN"] = [f"SSN_X2ONLY_{i:04d}" for i in range(x_extra_count)]
    x2_df = pd.concat([x2_df, extra_x], ignore_index=True)

    y_keep = rng.random(len(x2_df)) < 0.972
    y_base = x2_df.loc[y_keep].copy()
    n_y = len(y_base)

    mura_index = 1.2 + 0.045 * (y_base["Ambient_Humidity"].to_numpy() - 46) + 0.55 * (
        y_base["LineID"].to_numpy() == "LINE_C"
    ) + rng.normal(0, 0.35, n_y)
    leakage_current = 0.85 + 0.055 * (y_base["Developer_Time"].to_numpy() - 42) + 0.018 * y_base[
        "PM_Age_Days"
    ].to_numpy() + rng.normal(0, 0.12, n_y)
    lifetime_hour = 1150 - 2.2 * y_base["Queue_Time_Min"].to_numpy() - 3.1 * y_base["PM_Age_Days"].to_numpy() + rng.normal(
        0, 42, n_y
    )
    aoi_lambda = np.clip(0.9 + 3.8 * y_base["Vibration_RMS"].to_numpy() + 0.025 * y_base["Queue_Time_Min"].to_numpy(), 0.1, None)
    aoi_ng_count = rng.poisson(aoi_lambda).astype(float)
    reliability_probability = _sigmoid(
        -2.7 + 0.055 * (y_base["Queue_Time_Min"].to_numpy() - 55) + 0.85 * (y_base["Ambient_Humidity"].to_numpy() > 55)
    )
    reliability_fail = rng.binomial(1, np.clip(reliability_probability, 0.02, 0.90)).astype(int)
    final_yield_probability = _sigmoid(2.2 - 0.42 * mura_index - 0.30 * leakage_current - 0.12 * aoi_ng_count)
    final_yield = rng.binomial(1, np.clip(final_yield_probability, 0.03, 0.98)).astype(int)
    color_shift = 0.7 + 0.028 * (y_base["PreBake_Temp"].to_numpy() - 94) + 0.020 * (
        y_base["Ambient_Humidity"].to_numpy() - 46
    ) + rng.normal(0, 0.16, n_y)

    bin_grade = np.where(
        (final_yield == 1) & (mura_index < 1.5) & (leakage_current < 1.1),
        "A",
        np.where((final_yield == 1) & (mura_index < 2.2), "B", np.where(reliability_fail == 1, "D", "C")),
    )

    y2_df = pd.DataFrame(
        {
            "SSN": y_base["SSN"].to_numpy(),
            "FinalInspectDate": (
                pd.to_datetime(y_base["AuxProcessDate"]) + pd.to_timedelta(rng.integers(3, 16, size=n_y), unit="D")
            ).dt.strftime("%Y-%m-%d"),
            "Lifetime_Hour": np.round(lifetime_hour, 2),
            "Mura_Index": np.round(mura_index, 4),
            "Leakage_Current": np.round(leakage_current, 5),
            "Color_Shift_DE": np.round(color_shift, 4),
            "Final_Yield": final_yield,
            "Reliability_Fail": reliability_fail,
            "Bin_Grade": bin_grade,
            "AOI_NG_Count": aoi_ng_count,
        }
    )

    y_extra_count = max(5, int(len(base_keys) * 0.006))
    extra_y = y2_df.sample(y_extra_count, random_state=seed + 1).copy()
    extra_y["SSN"] = [f"SSN_Y2ONLY_{i:04d}" for i in range(y_extra_count)]
    y2_df = pd.concat([y2_df, extra_y], ignore_index=True)

    x2_df = _add_missing_values(
        x2_df,
        rng,
        ["LineID", "PreBake_Temp", "Ambient_Humidity", "Developer_Time", "Queue_Time_Min", "Vibration_RMS"],
        ratio=0.012,
    )
    y2_df = _add_missing_values(
        y2_df,
        rng,
        ["Lifetime_Hour", "Mura_Index", "Leakage_Current", "Color_Shift_DE", "Bin_Grade"],
        ratio=0.010,
    )
    x2_df = _add_outliers(x2_df, rng, "Queue_Time_Min", ratio=0.006, shift=135.0)
    y2_df = _add_outliers(y2_df, rng, "AOI_NG_Count", ratio=0.005, shift=18.0)

    x2_df.to_csv(output_path / x_filename, index=False, encoding="utf-8-sig")
    y2_df.to_csv(output_path / y_filename, index=False, encoding="utf-8-sig")
    return x2_df, y2_df


def main() -> None:
    """Generate sample files and print a short creation summary."""
    generated = generate_all_sample_files()
    print("Sample data created.")
    for index, (x_df, y_df) in generated.items():
        x_keys = set(x_df["SSN"].dropna())
        y_keys = set(y_df["SSN"].dropna())
        x_merge_rate = len(x_keys & y_keys) / len(x_keys) * 100
        y_merge_rate = len(x_keys & y_keys) / len(y_keys) * 100
        suffix = "" if index == "sample_1" else "2"
        print(f"- X{suffix}: {DATA_DIR / f'sample_x{suffix}.csv'} ({len(x_df):,} rows)")
        print(f"- Y{suffix}: {DATA_DIR / f'sample_y{suffix}.csv'} ({len(y_df):,} rows)")
        print(f"- Approx. X{suffix} key match rate: {x_merge_rate:.1f}%")
        print(f"- Approx. Y{suffix} key match rate: {y_merge_rate:.1f}%")


if __name__ == "__main__":
    main()

"""Configuration loading and preset handling."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_preset(config: dict, preset_name: str) -> dict:
    config = deepcopy(config)
    preset = config.get("presets", {}).get(preset_name, {})
    config["screening"] = deep_merge(config.get("screening", {}), preset)
    return config


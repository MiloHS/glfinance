from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an experiment YAML configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"experiment", "data", "graph", "portfolio", "output"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    experiment = config["experiment"]
    for key in (
        "outer_weeks",
        "validation_weeks",
        "recalibration_weeks",
        "distance_lookback_weeks",
    ):
        if int(experiment[key]) <= 0:
            raise ValueError(f"experiment.{key} must be positive.")

    tickers = config["data"]["tickers"]
    symbols = [item["symbol"] for item in tickers]
    if len(symbols) != len(set(symbols)):
        raise ValueError("Ticker symbols must be unique.")
    for size in config["data"]["universe_sizes"]:
        if int(size) <= 1 or int(size) > len(symbols):
            raise ValueError(f"Invalid universe size: {size}")

    portfolio = config["portfolio"]
    if any(int(value) < 2 for value in portfolio["training_windows"]):
        raise ValueError("All covariance training windows must be at least 2.")
    if any(not 0 < float(value) <= 1 for value in portfolio["k_fractions"]):
        raise ValueError("All k fractions must be in (0, 1].")
    if any(not 0 <= float(value) <= 1 for value in portfolio["rho_values"]):
        raise ValueError("All rho values must be in [0, 1].")


def resolved_config(config: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """Return a copy with repository-relative paths made absolute."""
    result = deepcopy(config)
    root = Path(base_dir).resolve()
    result["data"]["cache_dir"] = str((root / result["data"]["cache_dir"]).resolve())
    result["output"]["root_dir"] = str((root / result["output"]["root_dir"]).resolve())
    return result


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


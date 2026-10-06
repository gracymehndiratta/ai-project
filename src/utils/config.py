"""Configuration loading helpers."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs(config: Dict[str, Any]) -> None:
    """Create output directories defined in the config."""
    dirs = [
        config["dataset"]["raw_dir"],
        config["dataset"]["processed_dir"],
        config["outputs"]["models_dir"],
        config["outputs"]["logs_dir"],
        config["outputs"]["plots_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

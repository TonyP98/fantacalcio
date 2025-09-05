"""Utility helpers for configuration, logging and paths."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration."""
    config_path = path or get_project_root() / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    """Resolve a path key from config to absolute Path."""
    return get_project_root() / config["paths"][key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")
    return logging.getLogger("fantacalcio")

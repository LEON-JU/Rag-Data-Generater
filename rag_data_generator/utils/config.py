"""Simple YAML config loader used by the example scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | os.PathLike | None) -> dict:
    """Load a YAML config file. Returns an empty dict when path is falsy."""

    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def apply_environment_overrides(env_values: dict | None, *, overwrite: bool = False) -> None:
    """Mirror the env block into os.environ so that legacy code keeps working."""

    if not env_values:
        return
    for key, value in env_values.items():
        if value is None:
            continue
        if overwrite or key not in os.environ:
            os.environ[key] = str(value)


def choose(*values: Any, default: Any = None) -> Any:
    """Return the first non-None value, or ``default`` when everything is None."""

    for value in values:
        if value is not None:
            return value
    return default


__all__ = ["load_yaml_config", "apply_environment_overrides", "choose"]

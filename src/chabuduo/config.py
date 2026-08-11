from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path.cwd()
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def config_path(name: str) -> Path:
    """Return a bundled configuration path."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No configuration found for {name!r}: {path}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping and reject other top-level values."""
    with path.open(encoding="utf8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def write_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

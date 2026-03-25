from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from miniswewebagent import package_dir

builtin_config_dir = package_dir / "config"


def _nest_key_value(key: str, value: Any) -> dict[str, Any]:
    parts = key.split(".")
    nested: dict[str, Any] = value
    for part in reversed(parts):
        nested = {part: nested}
    return nested


def _resolve_config_path(spec: str) -> Path | None:
    path = Path(spec).expanduser()
    if path.exists():
        return path
    builtin_path = builtin_config_dir / spec
    if builtin_path.exists():
        return builtin_path
    return None


def get_config_from_spec(spec: str) -> dict[str, Any]:
    resolved_path = _resolve_config_path(spec)
    if resolved_path is not None:
        loaded = yaml.safe_load(resolved_path.read_text())
        return loaded or {}

    if "=" not in spec:
        raise ValueError(f"Unsupported config spec: {spec!r}")

    key, raw_value = spec.split("=", 1)
    return _nest_key_value(key, yaml.safe_load(raw_value))

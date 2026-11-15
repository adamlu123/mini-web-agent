from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OSWORLD_ASSETS = (
    REPOSITORY_ROOT.parents[1] / "OSWorld-V2" / "cache" / "osworld_v2_assets"
)
DEFAULT_API_KEY_FILE = REPOSITORY_ROOT.parents[1] / ".a-creds"
DEFAULT_PROXY_CONFIG = (
    REPOSITORY_ROOT / "evaluation_examples" / "settings" / "proxy" / "dataimpulse.json"
)


def config_path() -> Path:
    return (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "cuawright"
        / "benchmarks.json"
    )


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid CUAWright configuration: {path}")
    return value


def save_config(config: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configured_value(benchmark: str, key: str) -> str | None:
    item = load_config().get(benchmark)
    if not isinstance(item, dict):
        return None
    value = item.get(key)
    return value if isinstance(value, str) and value else None


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def git_commit(path: str | Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(resolved_path(path)), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"not a Git checkout: {path}")
    return completed.stdout.strip()


def require_commit(path: str | Path, expected: str, label: str) -> Path:
    root = resolved_path(path)
    observed = git_commit(root)
    if observed != expected:
        raise RuntimeError(
            f"{label} commit mismatch: expected {expected}, got {observed}"
        )
    return root


def credential_status(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"ok": False, "error": "credential path is not configured"}
    target = resolved_path(path)
    try:
        info = target.stat()
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": stat.S_ISREG(info.st_mode) and not stat.S_IMODE(info.st_mode) & 0o077,
        "path": str(target),
        "mode": oct(stat.S_IMODE(info.st_mode)),
    }

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


SOURCE_COMMIT = "319d8502caa36b7f8077a88dca8d20d3cdb71890"
SOURCE_REPOSITORY = "https://github.com/TheoLeeCJ/WeaveBench.git"
DATASET_REVISION = "cd887bf5ee0e70faa4b250f2cd192bcc1de164ea"
VM_SHA256 = "3c1caf41cb75b8482a30d1a251545393aee8175375ab405e0d6870f8a07fa3f8"


@dataclass(frozen=True, slots=True)
class WeaveBenchSource:
    root: Path
    commit: str

    @property
    def tasks_root(self) -> Path:
        return self.root / "cache" / "tasks"

    @property
    def vm_path(self) -> Path:
        return self.root / "cache" / "vm" / "Ubuntu.qcow2"

    @property
    def judge_template_profile(self) -> Path:
        return self.root / "cache" / "judge-template" / "profile"

    @property
    def judge_template_workspace(self) -> Path:
        return self.root / "cache" / "judge-template" / "workspace"


def default_source_root() -> Path:
    configured = os.environ.get("CUAWRIGHT_WEAVEBENCH_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(
            os.environ.get(
                "XDG_DATA_HOME",
                Path.home() / ".local" / "share",
            )
        )
        / "cuawright"
        / "weavebench"
        / "source"
    )


def inspect_source(explicit: str | Path | None = None) -> WeaveBenchSource:
    root = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else default_source_root()
    )
    required = (
        root / "weavebench" / "eval" / "run_weavebench.py",
        root / "cache" / "tasks",
        root / "cache" / "vm" / "Ubuntu.qcow2",
        root / "cache" / "judge-template" / "profile" / "openclaw.json",
        root / "cache" / "judge-template" / "workspace",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "WeaveBench source is incomplete; run `cuawright setup weavebench`: "
            + ", ".join(missing)
        )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("WeaveBench source is not a Git checkout")
    commit = completed.stdout.strip()
    if commit != SOURCE_COMMIT:
        raise RuntimeError(
            f"WeaveBench source commit mismatch: expected {SOURCE_COMMIT}, got {commit}"
        )
    return WeaveBenchSource(root=root, commit=commit)

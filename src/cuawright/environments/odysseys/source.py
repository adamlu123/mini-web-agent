from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DESKTOP_RUNTIME_COMMIT = "43997ed9a88b81151bbb496f1fd1f3b9febf312c"


@dataclass(frozen=True, slots=True)
class OdysseysSource:
    root: Path
    commit: str
    dataset_path: Path
    examples_dir: Path
    judge_script: Path
    dataset_sha256: str
    judge_sha256: str

    def provenance(self) -> dict[str, Any]:
        return {
            "identity_kind": "configured-clean-git-commit",
            "root": str(self.root),
            "commit": self.commit,
            "clean": True,
            "dataset": str(self.dataset_path),
            "dataset_sha256": self.dataset_sha256,
            "examples": str(self.examples_dir),
            "judge_script": str(self.judge_script),
            "judge_sha256": self.judge_sha256,
        }


@dataclass(frozen=True, slots=True)
class DesktopRuntimeSource:
    root: Path
    commit: str

    def provenance(self) -> dict[str, Any]:
        return {
            "identity_kind": "configured-clean-git-commit",
            "root": str(self.root),
            "commit": self.commit,
            "clean": True,
        }


def inspect_source(
    *,
    root: str | Path,
    expected_commit: str,
    dataset_path: str | Path,
    examples_dir: str | Path,
    judge_script: str | Path,
    expected_dataset_sha256: str | None = None,
    expected_judge_sha256: str | None = None,
) -> OdysseysSource:
    source_root = Path(root).expanduser().resolve()
    commit = expected_commit.lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError("Odysseys source commit must be a full 40-character SHA-1")
    observed = _git(source_root, "rev-parse", "HEAD")
    if observed != commit:
        raise RuntimeError(
            f"Odysseys source commit mismatch: expected {commit}, got {observed}"
        )
    dirty = _git(source_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError("Odysseys source checkout is dirty")

    dataset = _required_file(dataset_path, "dataset")
    examples = Path(examples_dir).expanduser().resolve()
    judge = _required_file(judge_script, "judge script")
    if not examples.is_dir():
        raise FileNotFoundError(f"Odysseys examples directory not found: {examples}")
    for path, label in (
        (dataset, "dataset"),
        (examples, "examples directory"),
        (judge, "judge script"),
    ):
        if path != source_root and source_root not in path.parents:
            raise ValueError(f"Odysseys {label} must be inside the source checkout")

    dataset_sha256 = _sha256(dataset.read_bytes())
    judge_sha256 = _sha256(judge.read_bytes())
    for label, expected, observed_hash in (
        ("dataset", expected_dataset_sha256, dataset_sha256),
        ("judge script", expected_judge_sha256, judge_sha256),
    ):
        if expected is not None and expected != observed_hash:
            raise RuntimeError(
                f"Odysseys {label} hash mismatch: expected {expected}, "
                f"got {observed_hash}"
            )

    return OdysseysSource(
        root=source_root,
        commit=commit,
        dataset_path=dataset,
        examples_dir=examples,
        judge_script=judge,
        dataset_sha256=dataset_sha256,
        judge_sha256=judge_sha256,
    )


def doctor_source(config: dict[str, Any]) -> dict[str, Any]:
    try:
        source = inspect_source(
            root=config["source"],
            expected_commit=config["source_commit"],
            dataset_path=config["dataset"],
            examples_dir=config["examples"],
            judge_script=config["judge"],
            expected_dataset_sha256=config.get("dataset_sha256"),
            expected_judge_sha256=config.get("judge_sha256"),
        )
        return {"ok": True, "source": source.provenance()}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def inspect_desktop_runtime(
    root: str | Path,
    expected_commit: str = DESKTOP_RUNTIME_COMMIT,
) -> DesktopRuntimeSource:
    source_root = Path(root).expanduser().resolve()
    commit = expected_commit.lower()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(
            "Odysseys desktop runtime commit must be a full 40-character SHA-1"
        )
    required = (
        source_root / "desktop_env" / "__init__.py",
        source_root / "desktop_env" / "desktop_env.py",
        source_root / "desktop_env" / "providers" / "docker" / "provider.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Desktop runtime source is incomplete: " + ", ".join(missing)
        )
    observed = _git(source_root, "rev-parse", "HEAD")
    if observed != commit:
        raise RuntimeError(
            f"Desktop runtime commit mismatch: expected {commit}, got {observed}"
        )
    dirty = _git(source_root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError("Odysseys desktop runtime checkout is dirty")
    return DesktopRuntimeSource(root=source_root, commit=commit)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"Odysseys source is not a Git checkout: {root}"
        )
    return completed.stdout.strip()


def _required_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Odysseys {label} not found: {path}")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

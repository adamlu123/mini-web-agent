from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final


@dataclass(frozen=True, slots=True)
class OSWorldRelease:
    benchmark_release: str
    tag: str
    source_commit: str
    source_repository: str


DEFAULT_RELEASE_TAG = "v2026.06.24"
OSWORLD_RELEASES: Final[Mapping[str, OSWorldRelease]] = MappingProxyType(
    {
        "v2026.06.24": OSWorldRelease(
            benchmark_release="osworld-v2-2026.06.24",
            tag="v2026.06.24",
            source_commit="43997ed9a88b81151bbb496f1fd1f3b9febf312c",
            source_repository="https://github.com/TheoLeeCJ/deskwright-cua.git",
        ),
        "v2026.08.08": OSWorldRelease(
            benchmark_release="osworld-v2-2026.08.08",
            tag="v2026.08.08",
            source_commit="d578d2d4e0dc82b43e270fdaa7fa89d9708cd154",
            source_repository="https://github.com/xlang-ai/OSWorld-V2.git",
        ),
    }
)
_RELEASE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {release.benchmark_release: release.tag for release in OSWORLD_RELEASES.values()}
)
_DEFAULT_RELEASE = OSWORLD_RELEASES[DEFAULT_RELEASE_TAG]
RELEASE = _DEFAULT_RELEASE.benchmark_release
SOURCE_COMMIT = _DEFAULT_RELEASE.source_commit
SOURCE_REPOSITORY = _DEFAULT_RELEASE.source_repository


def resolve_release(value: str | None = None) -> OSWorldRelease:
    requested = (
        value or os.environ.get("CUAWRIGHT_OSWORLD_RELEASE") or DEFAULT_RELEASE_TAG
    ).strip()
    tag = _RELEASE_ALIASES.get(requested, requested)
    try:
        return OSWORLD_RELEASES[tag]
    except KeyError:
        supported = ", ".join(OSWORLD_RELEASES)
        raise ValueError(
            f"unsupported OSWorld release {requested!r}; expected one of: {supported}"
        ) from None


@dataclass(frozen=True, slots=True)
class OSWorldSource:
    root: Path
    commit: str
    release: OSWorldRelease

    @property
    def evaluation_examples(self) -> Path:
        return self.root / "evaluation_examples"

    def provenance(self) -> dict[str, Any]:
        return {
            "release": self.release.benchmark_release,
            "tag": self.release.tag,
            "repository": self.release.source_repository,
            "commit": self.commit,
            "root": str(self.root),
        }


def data_home() -> Path:
    return Path(
        os.environ.get(
            "XDG_DATA_HOME",
            Path.home() / ".local" / "share",
        )
    )


def default_source_root(release: str | None = None) -> Path:
    selected = resolve_release(release)
    return data_home() / "cuawright" / "osworld" / selected.benchmark_release / "source"


def resolve_source_root(
    explicit: str | Path | None = None,
    *,
    release: str | None = None,
) -> Path:
    candidate = explicit or os.environ.get("CUAWRIGHT_OSWORLD_ROOT")
    root = (
        Path(candidate).expanduser().resolve()
        if candidate
        else default_source_root(release)
    )
    _validate_layout(root)
    return root


def inspect_source(
    explicit: str | Path | None = None,
    *,
    release: str | None = None,
) -> OSWorldSource:
    selected_release = resolve_release(release)
    root = resolve_source_root(explicit, release=selected_release.tag)
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OSWorld source is not a Git checkout: "
            + (completed.stderr.strip() or str(root))
        )
    commit = completed.stdout.strip()
    if commit != selected_release.source_commit:
        raise RuntimeError(
            "OSWorld source commit mismatch for "
            f"{selected_release.tag}: expected {selected_release.source_commit}, "
            f"got {commit}"
        )
    return OSWorldSource(
        root=root,
        commit=commit,
        release=selected_release,
    )


def activate_source(
    explicit: str | Path | None = None,
    *,
    release: str | None = None,
) -> OSWorldSource:
    source = inspect_source(explicit, release=release)
    selected = source.root
    for name in ("desktop_env", "task_loader"):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        origin = Path(module_file).resolve()
        if not origin.is_relative_to(selected):
            raise RuntimeError(
                f"{name} was imported from {origin} before activating {selected}"
            )
    selected_text = str(selected)
    sys.path[:] = [entry for entry in sys.path if entry != selected_text]
    sys.path.insert(0, selected_text)
    return source


def install_source(
    *,
    destination: str | Path | None = None,
    seed_repository: str | Path | None = None,
    tasks_from: str | Path | None = None,
    release: str | None = None,
) -> OSWorldSource:
    selected_release = resolve_release(release)
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else default_source_root(selected_release.tag)
    )
    if target.exists():
        return inspect_source(target, release=selected_release.tag)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"stale OSWorld setup directory exists: {staging}")
    repository = str(seed_repository or selected_release.source_repository)
    clone = subprocess.run(
        ["git", "clone", "--no-checkout", repository, str(staging)],
        text=True,
        capture_output=True,
        check=False,
    )
    if clone.returncode != 0:
        raise RuntimeError(
            "Could not clone OSWorld source: "
            + (clone.stderr.strip() or clone.stdout.strip())
        )
    try:
        checkout = subprocess.run(
            [
                "git",
                "-C",
                str(staging),
                "checkout",
                "--detach",
                selected_release.source_commit,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if checkout.returncode != 0:
            raise RuntimeError(
                "Could not check out OSWorld source: "
                + (checkout.stderr.strip() or checkout.stdout.strip())
            )
        if tasks_from is not None:
            source_tasks = Path(tasks_from).expanduser().resolve()
            destination_tasks = staging / "evaluation_examples" / "task_class"
            destination_tasks.mkdir(parents=True, exist_ok=True)
            for path in sorted(source_tasks.glob("task_*.py")):
                (destination_tasks / path.name).write_bytes(path.read_bytes())
        staging.replace(target)
    except BaseException:
        if staging.exists():
            subprocess.run(
                ["git", "-C", str(staging), "clean", "-fdx"],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["rm", "-rf", str(staging)],
                capture_output=True,
                check=False,
            )
        raise
    return inspect_source(target, release=selected_release.tag)


def doctor_source(
    explicit: str | Path | None = None,
    *,
    release: str | None = None,
) -> dict[str, Any]:
    try:
        source = inspect_source(explicit, release=release)
        task_count = len(
            list((source.evaluation_examples / "task_class").glob("task_*.py"))
        )
        return {
            "ok": task_count == 108,
            "source": source.provenance(),
            "task_count": task_count,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def write_source_config(path: Path, source: OSWorldSource) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(source.provenance(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_layout(root: Path) -> None:
    required = (
        root / "desktop_env" / "desktop_env.py",
        root / "task_loader.py",
        root / "evaluation_examples",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "OSWorld source is incomplete; run `cuawright setup osworld`: "
            + ", ".join(missing)
        )

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
from typing import Iterable, Mapping

from packaging.version import InvalidVersion, Version


class MissingOptionalDependencyError(RuntimeError):
    """A requested CUAWright feature is not installed."""


def _older_than(installed: str, minimum: str) -> bool:
    try:
        return Version(installed) < Version(minimum)
    except InvalidVersion:
        return True


def require_optional_modules(
    *,
    benchmark: str,
    extra: str,
    modules: Iterable[str],
    minimum_versions: Mapping[str, str] | None = None,
    requires_separate_runtime: bool = False,
) -> None:
    missing = tuple(module for module in modules if find_spec(module) is None)
    outdated: list[str] = []
    if not missing:
        for distribution, minimum in (minimum_versions or {}).items():
            try:
                installed = distribution_version(distribution)
            except PackageNotFoundError:
                missing += (distribution,)
                continue
            if _older_than(installed, minimum):
                outdated.append(f"{distribution} {installed} (requires >={minimum})")
    if not missing and not outdated:
        return

    problems = []
    if missing:
        problems.append("not installed: " + ", ".join(missing))
    if outdated:
        problems.append("incompatible versions: " + ", ".join(outdated))
    guidance = (
        f"Install `cuawright[{extra}]` in a dedicated environment "
        f"(for example, `uv sync --extra {extra}`). Do not combine it with "
        "`cuawright[osworld]` or `cuawright[full]`."
        if requires_separate_runtime
        else (
            f"Install `cuawright[{extra}]` (for example, "
            f"`uv sync --extra {extra}`) or install `cuawright[full]`."
        )
    )
    raise MissingOptionalDependencyError(
        f"{benchmark} requires optional dependencies that are "
        f"{'; '.join(problems)}. {guidance}"
    )

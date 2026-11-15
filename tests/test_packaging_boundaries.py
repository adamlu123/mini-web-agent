from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement

from cuawright.benchmarks import BENCHMARKS
from cuawright.cli import run
from cuawright.dependencies import (
    MissingOptionalDependencyError,
    _older_than,
)


ROOT = Path(__file__).parents[1]
BENCHMARK_EXTRAS = (
    "osworld",
    "weavebench",
    "benchcad",
    "cadgenbench",
    "odysseys",
)
WEAVEBENCH_REQUIREMENTS = {
    "docker": ">=7.0",
    "filelock": ">=3.12",
    "flask": ">=3.0",
    "gymnasium": ">=0.29",
    "huggingface-hub": ">=0.24",
    "lxml": ">=4.9",
    "openai": ">=1.40",
    "psutil": ">=5.9",
    "requests": ">=2.31",
    "tqdm": ">=4.66",
}


def _metadata() -> dict[str, object]:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_metadata_covers_supported_optional_sets_and_conflicts() -> None:
    metadata = _metadata()
    project = metadata["project"]
    extras = project["optional-dependencies"]
    full = set(extras["full"])
    full_names = {Requirement(value).name for value in full}

    assert project["dependencies"] == ["packaging>=24"]
    assert set(BENCHMARK_EXTRAS) <= extras.keys()
    for name in (
        "providers",
        "inspection",
        "cloud-legacy",
        "osworld",
        "benchcad",
        "cadgenbench",
        "odysseys",
    ):
        assert {Requirement(value).name for value in extras[name]} <= full_names, name
        assert not any(
            requirement.startswith("cuawright[") for requirement in extras[name]
        )

    assert "python-dotenv" in full
    assert "dotenv" not in full
    assert "black" in _metadata()["dependency-groups"]["dev"]

    weavebench = {
        Requirement(value).name: str(Requirement(value).specifier)
        for value in extras["weavebench"]
    }
    assert WEAVEBENCH_REQUIREMENTS.items() <= weavebench.items()
    assert set(weavebench) <= full_names
    assert "gymnasium~=0.28.1" in full
    assert "gymnasium>=0.29" in extras["weavebench"]
    assert {"filelock>=3.12", "docker>=7.0", "lxml>=4.9", "openai>=1.40"} <= full
    assert "tqdm~=4.65.0" in full

    conflicts = metadata["tool"]["uv"]["conflicts"]
    conflict_sets = {
        frozenset(item["extra"] for item in conflict) for conflict in conflicts
    }
    assert conflict_sets == {
        frozenset(("full", "weavebench")),
        frozenset(("osworld", "weavebench")),
        frozenset(("odysseys", "weavebench")),
    }
    assert set(extras["odysseys"]) == set(extras["osworld"])


def test_weavebench_preflight_matches_pinned_host_runtime() -> None:
    descriptor = BENCHMARKS["weavebench"]
    required_imports = {
        "docker",
        "filelock",
        "flask",
        "gymnasium",
        "huggingface_hub",
        "lxml",
        "openai",
        "psutil",
        "requests",
        "tqdm",
    }

    assert required_imports <= set(descriptor.required_modules)
    assert descriptor.minimum_versions == {
        name: specifier.removeprefix(">=")
        for name, specifier in WEAVEBENCH_REQUIREMENTS.items()
    }
    assert descriptor.requires_separate_runtime is True


def test_base_import_surface_does_not_load_optional_dependencies() -> None:
    script = """
import importlib.abc
import sys

blocked = {
    "PIL", "anthropic", "boto3", "docker", "filelock", "flask", "gymnasium",
    "huggingface_hub", "lxml", "mcp", "numpy", "openai", "pandas",
    "playwright", "psutil", "pydantic", "requests", "tqdm", "yaml",
}

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise ModuleNotFoundError(f"blocked optional import: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptional())

import cuawright
from cuawright.agents import AgentLoop
from cuawright.cli import parser
from cuawright.config import prompt_registry, store
from cuawright.runtime import contracts

assert cuawright
assert AgentLoop
assert prompt_registry.PROMPT_BUNDLE_REGISTRY
assert prompt_registry.select_osworld_prompt_bundle(no_subagents=False)
assert prompt_registry.select_weavebench_prompt_bundle(
    translated_system_prompt="external"
)
assert prompt_registry.select_benchcad_prompt_bundle(condition="full")
assert prompt_registry.select_cadgenbench_prompt_bundle()
assert prompt_registry.select_odysseys_prompt_bundle()
assert store.load_config
assert contracts.TurnProvider
assert "setup" in parser().format_help()
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("benchmark_name", "missing_module"),
    (
        ("osworld", "gymnasium"),
        ("weavebench", "requests"),
        ("benchcad", "docker"),
        ("cadgenbench", "mcp"),
        ("odysseys", "gymnasium"),
    ),
)
def test_missing_benchmark_dependency_names_install_extra(
    benchmark_name: str,
    missing_module: str,
) -> None:
    descriptor = BENCHMARKS[benchmark_name]

    def available(module: str) -> object | None:
        return None if module == missing_module else object()

    with (
        patch("cuawright.cli.load_config", return_value={benchmark_name: {}}),
        patch("cuawright.dependencies.find_spec", side_effect=available),
    ):
        with pytest.raises(MissingOptionalDependencyError) as captured:
            run(Namespace(benchmark=benchmark_name))

    message = str(captured.value)
    assert descriptor.display_name in message
    assert missing_module in message
    assert f"cuawright[{benchmark_name}]" in message
    assert f"uv sync --extra {benchmark_name}" in message
    if benchmark_name == "weavebench":
        assert "dedicated environment" in message
        assert "Do not combine it" in message


@pytest.mark.parametrize(
    ("distribution", "installed", "requirement"),
    (
        ("gymnasium", "0.28.1", "gymnasium 0.28.1 (requires >=0.29)"),
        ("tqdm", "4.65.2", "tqdm 4.65.2 (requires >=4.66)"),
    ),
)
def test_weavebench_rejects_osworld_runtime_versions(
    distribution: str,
    installed: str,
    requirement: str,
) -> None:
    descriptor = BENCHMARKS["weavebench"]

    def installed_version(name: str) -> str:
        if name == distribution:
            return installed
        return descriptor.minimum_versions[name]

    with (
        patch("cuawright.cli.load_config", return_value={"weavebench": {}}),
        patch("cuawright.dependencies.find_spec", return_value=object()),
        patch(
            "cuawright.dependencies.distribution_version",
            side_effect=installed_version,
        ),
    ):
        with pytest.raises(MissingOptionalDependencyError) as captured:
            run(Namespace(benchmark="weavebench"))

    message = str(captured.value)
    assert requirement in message
    assert "dedicated environment" in message


@pytest.mark.parametrize(
    ("installed", "minimum", "older"),
    (
        ("0.29.0rc1", "0.29", True),
        ("4.66.0.dev1", "4.66", True),
        ("0.29", "0.29", False),
        ("4.66.0", "4.66", False),
        ("not-a-version", "4.66", True),
    ),
)
def test_minimum_version_checks_follow_pep440(
    installed: str,
    minimum: str,
    older: bool,
) -> None:
    assert _older_than(installed, minimum) is older

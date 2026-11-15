from __future__ import annotations

import argparse
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping

from cuawright.health.contracts import HealthAdapterMetadata
from cuawright.operations.launch import (
    add_benchcad_run_arguments,
    add_benchcad_setup_arguments,
    add_cadgenbench_run_arguments,
    add_cadgenbench_setup_arguments,
    add_osworld_run_arguments,
    add_osworld_setup_arguments,
    add_odysseys_run_arguments,
    add_odysseys_setup_arguments,
    add_weavebench_run_arguments,
    add_weavebench_setup_arguments,
    doctor_benchcad_config,
    doctor_cadgenbench_config,
    doctor_osworld_config,
    doctor_odysseys_config,
    doctor_weavebench_config,
    run_benchcad,
    run_cadgenbench,
    run_osworld,
    run_odysseys,
    run_weavebench,
    setup_benchcad,
    setup_cadgenbench,
    setup_osworld,
    setup_odysseys,
    setup_weavebench,
)


ParserBuilder = Callable[[argparse.ArgumentParser], None]
SetupHandler = Callable[[argparse.Namespace], dict[str, Any]]
DoctorHandler = Callable[[Mapping[str, Any]], dict[str, Any]]
RunHandler = Callable[[argparse.Namespace, Mapping[str, Any]], int]


@dataclass(frozen=True, slots=True)
class BenchmarkDescriptor:
    name: str
    display_name: str
    install_extra: str
    required_modules: tuple[str, ...]
    minimum_versions: Mapping[str, str]
    requires_separate_runtime: bool
    stable_conditions: tuple[str, ...]
    add_setup_arguments: ParserBuilder
    add_run_arguments: ParserBuilder
    setup_handler: SetupHandler
    doctor_handler: DoctorHandler
    run_handler: RunHandler
    health: HealthAdapterMetadata


BENCHMARKS: Final[Mapping[str, BenchmarkDescriptor]] = MappingProxyType(
    {
        "osworld": BenchmarkDescriptor(
            name="osworld",
            display_name="OSWorld V2",
            install_extra="osworld",
            required_modules=(
                "anthropic",
                "gymnasium",
                "openai",
                "PIL",
                "requests",
            ),
            minimum_versions={},
            requires_separate_runtime=False,
            stable_conditions=("default",),
            add_setup_arguments=add_osworld_setup_arguments,
            add_run_arguments=add_osworld_run_arguments,
            setup_handler=setup_osworld,
            doctor_handler=doctor_osworld_config,
            run_handler=run_osworld,
            health=HealthAdapterMetadata(
                supported=True,
                benchmark_names=("osworld", "osworld-v2"),
                progress_markers=(
                    "/opt/cuawright-tools/binder.py",
                    "/opt/cuawright-tools/critic.py",
                    "/opt/cuawright-tools/verifier.py",
                    "osw ",
                    "xdotool",
                    "pyautogui",
                    "terminal_command",
                ),
            ),
        ),
        "weavebench": BenchmarkDescriptor(
            name="weavebench",
            display_name="WeaveBench",
            install_extra="weavebench",
            required_modules=(
                "anthropic",
                "docker",
                "filelock",
                "flask",
                "gymnasium",
                "huggingface_hub",
                "lxml",
                "openai",
                "PIL",
                "psutil",
                "requests",
                "tqdm",
            ),
            minimum_versions={
                "docker": "7.0",
                "filelock": "3.12",
                "flask": "3.0",
                "gymnasium": "0.29",
                "huggingface-hub": "0.24",
                "lxml": "4.9",
                "openai": "1.40",
                "psutil": "5.9",
                "requests": "2.31",
                "tqdm": "4.66",
            },
            requires_separate_runtime=True,
            stable_conditions=("default",),
            add_setup_arguments=add_weavebench_setup_arguments,
            add_run_arguments=add_weavebench_run_arguments,
            setup_handler=setup_weavebench,
            doctor_handler=doctor_weavebench_config,
            run_handler=run_weavebench,
            health=HealthAdapterMetadata(
                supported=False,
                benchmark_names=("weavebench",),
                incomplete_reason=(
                    "WeaveBench retains CUAWright traces below its own result layout "
                    "but does not yet retain a canonical terminal manifest for the "
                    "standalone health gate"
                ),
            ),
        ),
        "benchcad": BenchmarkDescriptor(
            name="benchcad",
            display_name="BenchCAD",
            install_extra="benchcad",
            required_modules=("anthropic", "docker", "openai", "PIL", "requests"),
            minimum_versions={},
            requires_separate_runtime=False,
            stable_conditions=("full", "direct"),
            add_setup_arguments=add_benchcad_setup_arguments,
            add_run_arguments=add_benchcad_run_arguments,
            setup_handler=setup_benchcad,
            doctor_handler=doctor_benchcad_config,
            run_handler=run_benchcad,
            health=HealthAdapterMetadata(
                supported=True,
                benchmark_names=("benchcad",),
                progress_markers=(
                    "/opt/cuawright-tools/binder.py",
                    "/opt/cuawright-tools/critic.py",
                    "/opt/cuawright-tools/verifier.py",
                    "benchcad-run",
                    "benchcad-render",
                    "prediction.py",
                ),
            ),
        ),
        "cadgenbench": BenchmarkDescriptor(
            name="cadgenbench",
            display_name="CADGenBench",
            install_extra="cadgenbench",
            required_modules=(
                "anthropic",
                "docker",
                "mcp",
                "openai",
                "PIL",
                "pydantic",
                "yaml",
            ),
            minimum_versions={},
            requires_separate_runtime=False,
            stable_conditions=("default",),
            add_setup_arguments=add_cadgenbench_setup_arguments,
            add_run_arguments=add_cadgenbench_run_arguments,
            setup_handler=setup_cadgenbench,
            doctor_handler=doctor_cadgenbench_config,
            run_handler=run_cadgenbench,
            health=HealthAdapterMetadata(
                supported=True,
                benchmark_names=("cadgenbench",),
                progress_markers=(
                    "/opt/cuawright-tools/binder.py",
                    "/opt/cuawright-tools/critic.py",
                    "/opt/cuawright-tools/verifier.py",
                    "cad-mcp",
                    "workflow_hints",
                    "output.step",
                    "build123d",
                ),
                lifecycle_directory="container-lifecycle",
            ),
        ),
        "odysseys": BenchmarkDescriptor(
            name="odysseys",
            display_name="Odysseys",
            install_extra="odysseys",
            required_modules=(
                "anthropic",
                "gymnasium",
                "openai",
                "PIL",
                "requests",
            ),
            minimum_versions={},
            requires_separate_runtime=False,
            stable_conditions=("default",),
            add_setup_arguments=add_odysseys_setup_arguments,
            add_run_arguments=add_odysseys_run_arguments,
            setup_handler=setup_odysseys,
            doctor_handler=doctor_odysseys_config,
            run_handler=run_odysseys,
            health=HealthAdapterMetadata(
                supported=True,
                benchmark_names=("odysseys",),
                progress_markers=(
                    "/opt/cuawright-tools/binder.py",
                    "/opt/cuawright-tools/critic.py",
                    "/opt/cuawright-tools/verifier.py",
                    "playwright",
                    "connect_over_cdp",
                    "terminal_command",
                ),
            ),
        ),
    }
)


def benchmark_descriptor(name: str) -> BenchmarkDescriptor:
    try:
        return BENCHMARKS[name]
    except KeyError:
        raise ValueError(f"unsupported benchmark: {name!r}") from None


def benchmark_for_health_name(name: str) -> BenchmarkDescriptor:
    matches = [
        descriptor
        for descriptor in BENCHMARKS.values()
        if name in descriptor.health.benchmark_names
    ]
    if len(matches) != 1:
        raise ValueError(f"unsupported benchmark health adapter: {name!r}")
    return matches[0]

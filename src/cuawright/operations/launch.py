from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cuawright.config.store import (
    credential_status,
    require_commit,
    resolved_path,
)
from cuawright.environments.odysseys.source import (
    DESKTOP_RUNTIME_COMMIT,
    inspect_desktop_runtime,
)
from cuawright.environments.odysseys.source import (
    doctor_source as doctor_odysseys,
)
from cuawright.environments.odysseys.source import (
    inspect_source as inspect_odysseys,
)
from cuawright.environments.osworld.source import DEFAULT_RELEASE_TAG
from cuawright.environments.osworld.source import doctor_source as doctor_osworld
from cuawright.environments.osworld.source import install_source as install_osworld
from cuawright.environments.weavebench.source import (
    inspect_source as inspect_weavebench,
)

BENCHCAD_COMMIT = "52087ef4b08811c21c09b8b16d0ae36c70844865"
CADGENBENCH_COMMIT = "33304cf771fc5639144b1df9611e347251052cf8"


def add_osworld_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark-release", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--source")
    parser.add_argument("--seed-repository")
    parser.add_argument("--tasks-from")
    parser.add_argument("--vm", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--website-host-suffix", default="10.217.29.172.nip.io")


def add_weavebench_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--judge-runtime", required=True)
    parser.add_argument("--judge-image", required=True)
    parser.add_argument("--docker-bin", default="/usr/bin/docker")


def add_benchcad_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image", default="cuawright-benchcad:52087ef")
    parser.add_argument("--api-key-file", required=True)


def add_cadgenbench_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image", default="cuawright-cadgenbench:33304cf")
    parser.add_argument("--api-key-file", required=True)


def add_odysseys_setup_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--vm", required=True)
    parser.add_argument("--api-key-file", required=True)


def add_osworld_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="Run one linear Actor with no Binder, Critic, or Verifier",
    )


def add_weavebench_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--no-subagents", action="store_true")


def add_benchcad_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("condition", choices=["full", "direct"])
    parser.add_argument("--record", required=True)
    parser.add_argument("--no-subagents", action="store_true")


def add_cadgenbench_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--no-subagents", action="store_true")


def add_odysseys_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)


def setup_osworld(args: argparse.Namespace) -> dict[str, Any]:
    source = install_osworld(
        destination=args.source,
        seed_repository=args.seed_repository,
        tasks_from=args.tasks_from,
        release=args.benchmark_release,
    )
    return {
        "benchmark_release": source.release.tag,
        "source": str(source.root),
        "vm": str(resolved_path(args.vm)),
        "assets": str(resolved_path(args.assets)),
        "api_key_file": str(resolved_path(args.api_key_file)),
        "website_host_suffix": args.website_host_suffix,
    }


def setup_weavebench(args: argparse.Namespace) -> dict[str, Any]:
    source = inspect_weavebench(args.source)
    return {
        "source": str(source.root),
        "api_key_file": str(resolved_path(args.api_key_file)),
        "judge_runtime": str(resolved_path(args.judge_runtime)),
        "judge_image": args.judge_image,
        "docker_bin": str(resolved_path(args.docker_bin)),
    }


def setup_benchcad(args: argparse.Namespace) -> dict[str, Any]:
    source = require_commit(args.source, BENCHCAD_COMMIT, "BenchCAD")
    return {
        "source": str(source),
        "manifest": str(resolved_path(args.manifest)),
        "image": args.image,
        "api_key_file": str(resolved_path(args.api_key_file)),
    }


def setup_cadgenbench(args: argparse.Namespace) -> dict[str, Any]:
    source = require_commit(args.source, CADGENBENCH_COMMIT, "CADGenBench")
    return {
        "source": str(source),
        "dataset": str(resolved_path(args.dataset)),
        "image": args.image,
        "api_key_file": str(resolved_path(args.api_key_file)),
    }


def setup_odysseys(args: argparse.Namespace) -> dict[str, Any]:
    source = inspect_odysseys(
        root=args.source,
        expected_commit=args.source_commit,
        dataset_path=args.dataset,
        examples_dir=args.examples,
        judge_script=args.judge,
    )
    runtime = inspect_desktop_runtime(
        args.runtime_source,
        DESKTOP_RUNTIME_COMMIT,
    )
    return {
        "source": str(source.root),
        "source_commit": source.commit,
        "dataset": str(source.dataset_path),
        "dataset_sha256": source.dataset_sha256,
        "examples": str(source.examples_dir),
        "judge": str(source.judge_script),
        "judge_sha256": source.judge_sha256,
        "runtime_source": str(runtime.root),
        "runtime_commit": runtime.commit,
        "vm": str(resolved_path(args.vm)),
        "api_key_file": str(resolved_path(args.api_key_file)),
    }


def doctor_osworld_config(item: Mapping[str, Any]) -> dict[str, Any]:
    result = doctor_osworld(
        item["source"],
        release=str(item.get("benchmark_release", DEFAULT_RELEASE_TAG)),
    )
    result["vm_exists"] = Path(item["vm"]).is_file()
    result["assets_exist"] = Path(item["assets"]).is_dir()
    result["credential"] = credential_status(item.get("api_key_file"))
    result["ok"] = bool(
        result.get("ok")
        and result["vm_exists"]
        and result["assets_exist"]
        and result["credential"]["ok"]
    )
    return result


def doctor_weavebench_config(item: Mapping[str, Any]) -> dict[str, Any]:
    source = inspect_weavebench(item["source"])
    credential = credential_status(item.get("api_key_file"))
    return {
        "ok": (
            credential["ok"]
            and Path(item["judge_runtime"]).is_dir()
            and Path(item["docker_bin"]).is_file()
            and source.judge_template_profile.is_dir()
            and source.judge_template_workspace.is_dir()
        ),
        "source": str(source.root),
        "commit": source.commit,
        "credential": credential,
    }


def doctor_benchcad_config(item: Mapping[str, Any]) -> dict[str, Any]:
    root = require_commit(item["source"], BENCHCAD_COMMIT, "BenchCAD")
    credential = credential_status(item.get("api_key_file"))
    return {
        "ok": Path(item["manifest"]).is_file() and credential["ok"],
        "source": str(root),
        "manifest": item["manifest"],
        "credential": credential,
    }


def doctor_cadgenbench_config(item: Mapping[str, Any]) -> dict[str, Any]:
    root = require_commit(item["source"], CADGENBENCH_COMMIT, "CADGenBench")
    credential = credential_status(item.get("api_key_file"))
    return {
        "ok": Path(item["dataset"]).is_dir() and credential["ok"],
        "source": str(root),
        "dataset": item["dataset"],
        "credential": credential,
    }


def doctor_odysseys_config(item: Mapping[str, Any]) -> dict[str, Any]:
    result = doctor_odysseys(dict(item))
    credential = credential_status(item.get("api_key_file"))
    try:
        runtime = inspect_desktop_runtime(
            item["runtime_source"],
            item.get("runtime_commit", DESKTOP_RUNTIME_COMMIT),
        )
        runtime_error = None
    except Exception as exc:
        runtime = None
        runtime_error = f"{type(exc).__name__}: {exc}"
    result.update(
        {
            "credential": credential,
            "vm_exists": Path(item["vm"]).is_file(),
            "runtime_source": None if runtime is None else str(runtime.root),
            "runtime_error": runtime_error,
        }
    )
    result["ok"] = bool(
        result.get("ok")
        and result["vm_exists"]
        and credential["ok"]
        and runtime is not None
    )
    return result


def run_osworld(args: argparse.Namespace, item: Mapping[str, Any]) -> int:
    command = [
        sys.executable,
        "-m",
        "cuawright.run.osworld",
        "--task-id",
        args.task,
        "--benchmark-release",
        str(item.get("benchmark_release", DEFAULT_RELEASE_TAG)),
        "--osworld-root",
        item["source"],
        "--path-to-vm",
        item["vm"],
        "--assets-dir",
        item["assets"],
        "--api-key-file",
        item["api_key_file"],
        "--website-host-suffix",
        item["website_host_suffix"],
        "--model",
        "gpt-5.5",
        "--reasoning-effort",
        "xhigh",
        "--model-infrastructure",
        "aiservices",
        "--result-root",
        str(resolved_path(args.result_root)),
    ]
    if getattr(args, "no_subagents", False):
        command.append("--no-subagents")
    return subprocess.run(command).returncode


def run_weavebench(args: argparse.Namespace, item: Mapping[str, Any]) -> int:
    source = inspect_weavebench(item["source"])
    environment = {
        **os.environ,
        "CUAWRIGHT_SRC": str(Path(__file__).resolve().parents[2]),
        "CUAWRIGHT_WEAVEBENCH_REASONING_EFFORT": str(
            item.get("reasoning_effort", "high")
        ),
        "WEAVEBENCH_LITELLM_KEY_FILE": item["api_key_file"],
        "WEAVEBENCH_JUDGE_LITELLM_KEY_FILE": item["api_key_file"],
        "AJ_DOCKER_BIN": item["docker_bin"],
        "AJ_OPENCLAW_RUNTIME": item["judge_runtime"],
        "AJ_OPENCLAW_BIN": str(
            Path(item["judge_runtime"]) / "node_modules" / ".bin" / "openclaw"
        ),
        "AJ_OPENCLAW_DOCKER_IMAGE": item["judge_image"],
        "AJ_TEMPLATE_PROFILE": str(source.judge_template_profile),
        "AJ_TEMPLATE_WORKSPACE": str(source.judge_template_workspace),
    }
    command = [
        sys.executable,
        "-m",
        "weavebench.eval.run_weavebench",
        "--harness",
        "cuawright",
        "--tasks_root",
        str(source.tasks_root),
        "--path_to_vm",
        str(source.vm_path),
        "--domains",
        args.domain,
        "--task_filter",
        args.task,
        "--limit",
        "1",
        "--num_envs",
        "1",
        "--result_dir",
        str(resolved_path(args.result_root)),
    ]
    if getattr(args, "no_subagents", False):
        command.append("--no-subagents")
    return subprocess.run(command, cwd=source.root, env=environment).returncode


def run_benchcad(args: argparse.Namespace, item: Mapping[str, Any]) -> int:
    command = [
        sys.executable,
        "-m",
        "cuawright.run.benchcad",
        "--manifest",
        item["manifest"],
        "--record-id",
        args.record,
        "--mode",
        args.condition,
        "--image",
        item["image"],
        "--benchcad-repo",
        item["source"],
        "--api-key-file",
        item["api_key_file"],
        "--model",
        "gpt-5.5",
        "--reasoning-effort",
        "xhigh",
        "--result-root",
        str(resolved_path(args.result_root)),
    ]
    if getattr(args, "no_subagents", False):
        command.append("--no-subagents")
    return subprocess.run(command).returncode


def run_cadgenbench(args: argparse.Namespace, item: Mapping[str, Any]) -> int:
    command = [
        sys.executable,
        "-m",
        "cuawright.run.cadgenbench",
        "--dataset-dir",
        item["dataset"],
        "--fixture-id",
        args.fixture,
        "--image",
        item["image"],
        "--cadgenbench-repo",
        item["source"],
        "--api-key-file",
        item["api_key_file"],
        "--model",
        "gpt-5.5",
        "--reasoning-effort",
        "xhigh",
        "--result-root",
        str(resolved_path(args.result_root)),
    ]
    if getattr(args, "no_subagents", False):
        command.append("--no-subagents")
    return subprocess.run(command).returncode


def run_odysseys(args: argparse.Namespace, item: Mapping[str, Any]) -> int:
    command = [
        sys.executable,
        "-m",
        "cuawright.run.odysseys",
        "--task-id",
        args.task,
        "--source",
        item["source"],
        "--source-commit",
        item["source_commit"],
        "--task-source-json",
        item["dataset"],
        "--dataset-sha256",
        item["dataset_sha256"],
        "--examples-dir",
        item["examples"],
        "--judge-script",
        item["judge"],
        "--judge-sha256",
        item["judge_sha256"],
        "--runtime-source",
        item["runtime_source"],
        "--runtime-commit",
        item.get("runtime_commit", DESKTOP_RUNTIME_COMMIT),
        "--path-to-vm",
        item["vm"],
        "--api-key-file",
        item["api_key_file"],
        "--model-provider",
        "openai",
        "--model",
        "gpt-5.5",
        "--reasoning-effort",
        "xhigh",
        "--model-infrastructure",
        "aiservices",
        "--result-root",
        str(resolved_path(args.result_root)),
    ]
    return subprocess.run(command).returncode

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cuawright.config.prompt_registry import select_cadgenbench_prompt_bundle
from cuawright.environments.cadgenbench.task import (
    DEFAULT_BENCHMARK_COMMIT,
    DEFAULT_DATASET_REVISION,
    discover_tasks,
    resolve_benchmark_commit,
)
from cuawright.environments.cadgenbench.environment import CadGenBenchEnvironment
from cuawright.environments.cadgenbench.plugin import create_cadgenbench_plugin
from cuawright.environments.cadgenbench.errors import (
    CadGenBenchInfrastructureError,
)
from cuawright.environments.control import StatePersistenceError
from cuawright.environments.terminal import TaskConfig, TaskRunner
from cuawright.budget import CADGENBENCH_TOTAL_STEPS
from cuawright.models.providers import resolve_base_url, resolve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CadGenBench CUAWright harness"
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument(
        "--image",
        default="cuawright-cadgenbench:33304cf",
    )
    parser.add_argument(
        "--cadgenbench-repo",
        default=str(REPO_ROOT / "cache" / "cadgenbench"),
    )
    parser.add_argument(
        "--api-key-file",
        default=str(REPO_ROOT.parents[1] / ".a-creds"),
    )
    parser.add_argument(
        "--result-root",
        default=str(REPO_ROOT / "results-cadgenbench"),
    )
    parser.add_argument(
        "--model-provider",
        choices=["openai", "openrouter-anthropic", "anthropic"],
        default="openai",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--total-steps", type=int, default=CADGENBENCH_TOTAL_STEPS)
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="Run one linear Actor with no Binder, Critic, or Verifier",
    )
    parser.add_argument(
        "--benchmark-commit",
        default=DEFAULT_BENCHMARK_COMMIT,
    )
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _run_with_infrastructure_classification(
    run: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return run()
    except StatePersistenceError as exc:
        raise CadGenBenchInfrastructureError(
            "CADGenBench secure control-file persistence failed"
        ) from exc


def main() -> None:
    args = parse_args()
    no_subagents = bool(getattr(args, "no_subagents", False))
    model = resolve_model(args.model_provider, args.model)
    base_url = resolve_base_url(args.model_provider, args.base_url)
    if args.total_steps > CADGENBENCH_TOTAL_STEPS:
        raise ValueError(
            f"CADGenBench total steps cannot exceed {CADGENBENCH_TOTAL_STEPS}"
        )
    benchmark_commit = resolve_benchmark_commit(
        args.cadgenbench_repo,
        expected_commit=args.benchmark_commit,
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest = discover_tasks(
        dataset_dir=args.dataset_dir,
        fixture_ids=[args.fixture_id],
        benchmark_commit=benchmark_commit,
        dataset_revision=args.dataset_revision,
    )
    if len(manifest.tasks) != 1:
        raise RuntimeError("CADGenBench launcher requires exactly one fixture")
    task = manifest.tasks[0]
    prompt_bundle = select_cadgenbench_prompt_bundle(
        no_subagents=no_subagents,
    )
    assert prompt_bundle.agent_topology is not None
    config = TaskConfig(
        task_id=args.fixture_id,
        api_key_file=args.api_key_file,
        result_root=args.result_root,
        base_url=base_url,
        model=model,
        reasoning_effort=args.reasoning_effort,
        total_steps=args.total_steps,
        model_provider=args.model_provider,
    )
    runner = TaskRunner(config)
    task_payload = {"benchmark": "cadgenbench", **task.provenance()}
    (runner.run_dir / "task.json").write_text(
        json.dumps(
            task_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    plugin = create_cadgenbench_plugin(
        CadGenBenchEnvironment(
            task=task,
            run_dir=runner.run_dir,
            image=args.image,
            benchmark_commit=benchmark_commit,
        )
    )
    result = _run_with_infrastructure_classification(
        lambda: runner.run_terminal_benchmark(
            benchmark="cadgenbench",
            plugin=plugin,
            instruction=task.instruction,
            prompt_bundle=prompt_bundle,
            manifest_extra={
                "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
                "cadgenbench_repo": str(
                    Path(args.cadgenbench_repo).expanduser().resolve()
                ),
                "container_image": args.image,
                "benchmark_commit": benchmark_commit,
                "dataset_revision": args.dataset_revision,
                "manifest_sha256": manifest.manifest_sha256,
                "task_provenance": task.provenance(),
            },
            deadline_seconds=21_600,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if result.get("deadline_triggered") is True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

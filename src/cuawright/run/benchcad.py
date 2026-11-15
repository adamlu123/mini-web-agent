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

from cuawright.config.prompt_registry import select_benchcad_prompt_bundle
from cuawright.environments.benchcad.direct import run_direct
from cuawright.environments.benchcad.errors import (
    BenchCADInfrastructureError,
    BenchCADScoringInfrastructureError,
)
from cuawright.environments.benchcad.plugin import (
    BenchCADPlugin,
    BenchCADPluginConfig,
    create_benchcad_plugin,
)
from cuawright.environments.benchcad.task import load_task
from cuawright.environments.control import StatePersistenceError
from cuawright.environments.terminal import TaskConfig, TaskRunner
from cuawright.budget import DEFAULT_TOTAL_STEPS
from cuawright.models.providers import resolve_base_url, resolve_model

logger = logging.getLogger("cuawright.environments.benchcad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CUAWright or canonical direct on BenchCAD Vision2Code"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--mode", choices=["full", "direct"], default="full")
    parser.add_argument(
        "--image",
        default="cuawright-benchcad:52087ef",
    )
    parser.add_argument(
        "--benchcad-repo",
        default=str(REPO_ROOT / "cache" / "BenchCAD-main"),
    )
    parser.add_argument(
        "--api-key-file",
        default=str(REPO_ROOT.parent.parent / ".a-creds"),
    )
    parser.add_argument(
        "--result-root",
        default=str(REPO_ROOT / "results-benchcad"),
    )
    parser.add_argument(
        "--model-provider",
        choices=["openai", "openrouter-anthropic", "anthropic"],
        default="openai",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="Run one linear Actor with no Binder, Critic, or Verifier",
    )
    parser.add_argument("--defer-score", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def create_full_plugin(
    args: argparse.Namespace,
    *,
    run_dir: str | Path,
) -> BenchCADPlugin:
    """Construction hook for the benchmark-neutral unified launcher."""
    return create_benchcad_plugin(
        BenchCADPluginConfig(
            manifest_path=args.manifest,
            record_id=args.record_id,
            run_dir=run_dir,
            image=args.image,
            benchcad_repo=args.benchcad_repo,
            defer_evaluation=args.defer_score,
        )
    )


def _run_with_infrastructure_classification(
    run: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return run()
    except StatePersistenceError as exc:
        raise BenchCADInfrastructureError(
            "BenchCAD secure control-file persistence failed"
        ) from exc


def main() -> None:
    args = parse_args()
    no_subagents = bool(getattr(args, "no_subagents", False))
    model = resolve_model(args.model_provider, args.model)
    base_url = resolve_base_url(args.model_provider, args.base_url)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    manifest, task = load_task(args.manifest, args.record_id)
    if args.mode == "direct" and no_subagents:
        raise ValueError("--no-subagents applies to full mode, not direct mode")
    prompt_bundle = select_benchcad_prompt_bundle(
        condition=("full-linear" if no_subagents else args.mode),
        direct_system_prompt=(task.system_prompt if args.mode == "direct" else None),
        direct_user_prompt=task.user_prompt if args.mode == "direct" else None,
    )
    if args.mode == "direct":
        if args.model_provider != "openai":
            raise ValueError("BenchCAD direct mode currently requires provider openai")
        result = run_direct(
            task=task,
            manifest=manifest,
            result_root=args.result_root,
            api_key_file=args.api_key_file,
            base_url=base_url,
            model=model,
            reasoning_effort=args.reasoning_effort,
            image=args.image,
            benchcad_repo=args.benchcad_repo,
            benchmark_commit=manifest.benchmark_commit,
            defer_score=args.defer_score,
            prompt_bundle=prompt_bundle,
        )
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return

    assert prompt_bundle.agent_topology is not None
    config = TaskConfig(
        task_id=args.record_id,
        api_key_file=args.api_key_file,
        result_root=args.result_root,
        base_url=base_url,
        model=model,
        reasoning_effort=args.reasoning_effort,
        total_steps=args.total_steps,
        model_provider=args.model_provider,
    )
    for attempt in range(2):
        try:
            runner = TaskRunner(config)
            plugin = create_full_plugin(args, run_dir=runner.run_dir)
            result = _run_with_infrastructure_classification(
                lambda: runner.run_terminal_benchmark(
                    benchmark="benchcad",
                    plugin=plugin,
                    instruction=task.instruction,
                    prompt_bundle=prompt_bundle,
                    manifest_extra={
                        "task_manifest": str(
                            Path(args.manifest).expanduser().resolve()
                        ),
                        "container_image": args.image,
                        "benchcad_repo": str(
                            Path(args.benchcad_repo).expanduser().resolve()
                        ),
                        "defer_evaluation": args.defer_score,
                        "benchmark_commit": manifest.benchmark_commit,
                        "dataset_revision": manifest.dataset_revision,
                        "manifest_sha256": manifest.manifest_sha256,
                    },
                )
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return
        except BenchCADScoringInfrastructureError:
            raise
        except BenchCADInfrastructureError:
            if attempt == 0:
                logger.exception(
                    "BenchCAD sandbox infrastructure failed; retrying once"
                )
                continue
            raise


if __name__ == "__main__":
    main()

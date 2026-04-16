"""Iterative agent CLI entry point.

Usage:
    python -m miniswewebagent.run.iterative \\
        -c benchmark/om2w_hard_local_workspace_image_qa_flog_run_folders.yaml \\
        -c iterative.yaml \\
        --task-id <id> \\
        --tasks-file /path/to/tasks.json

The -c flags are merged left-to-right (same as mini.py).  The last config should
be (or include) iterative.yaml, which contributes the ``iterative`` section.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miniswewebagent.agents.iterative import IterativeRunner
from miniswewebagent.config import get_config_from_spec, snapshot_config_specs
from miniswewebagent.environments import get_environment
from miniswewebagent.models import get_model
from miniswewebagent.tasks.om2w import load_om2w_task
from miniswewebagent.utils.om2w_eval import export_online_mind2web_artifacts
from miniswewebagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG = "iterative.yaml"

app = typer.Typer(no_args_is_help=True)
console = Console(highlight=False)


def _timestamped_output_dir(base_dir: str | Path | None, task_id: str | None) -> Path:
    base = Path(base_dir or "outputs").expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = task_id or "adhoc"
    return base / f"{suffix}_{stamp}"


def run_one(
    *,
    task: str | None = None,
    task_id: str | None = None,
    tasks_file: Path | None = None,
    start_url: str | None = None,
    config_spec: list[str] | None = None,
    output_dir: Path | None = None,
    resolved_output_dir: Path | None = None,
    debug: bool = False,
    snapshot_config: bool = True,
) -> Any:
    config_spec = config_spec or [DEFAULT_CONFIG]
    configs = [get_config_from_spec(spec) for spec in config_spec]
    config = recursive_merge(*configs)

    run_config = config.get("run", {})
    resolved_tasks_file = tasks_file or run_config.get("tasks_file")
    resolved_task_id = task_id or run_config.get("task_id")
    resolved_task = task or run_config.get("task")
    resolved_start_url = start_url or run_config.get("start_url")

    task_record = None
    if resolved_task_id and (not resolved_task or not resolved_start_url):
        if not resolved_tasks_file:
            raise ValueError("--task-id requires --tasks-file unless --task and --start-url are both set.")
        task_record = load_om2w_task(resolved_tasks_file, resolved_task_id)
        resolved_task = resolved_task or task_record["task"]
        resolved_start_url = resolved_start_url or task_record["start_url"]

    if not resolved_task:
        raise ValueError("A task is required. Use --task or --task-id.")

    resolved_output_dir = resolved_output_dir or _timestamped_output_dir(
        output_dir or config.get("environment", {}).get("output_dir") or "outputs",
        resolved_task_id,
    )
    if snapshot_config:
        snapshot_config_specs(config_spec, resolved_output_dir, merged_config=config)

    config = recursive_merge(
        config,
        {
            "run": {
                "task": resolved_task,
                "task_id": resolved_task_id or UNSET,
                "start_url": resolved_start_url or UNSET,
                "tasks_file": str(resolved_tasks_file) if resolved_tasks_file else UNSET,
            },
            "environment": {
                "output_dir": str(resolved_output_dir),
                "start_url": resolved_start_url or UNSET,
                "headless": False if debug else UNSET,
                "devtools": True if debug else UNSET,
                "keep_open_on_exit": True if debug else UNSET,
                "prompt_before_close": True if debug else UNSET,
                "slow_mo_ms": 250 if debug else UNSET,
            },
            "model": {
                "error_log_path": str(resolved_output_dir / "runtime_errors.jsonl"),
            },
        },
    )

    model = get_model(config.get("model", {}))
    env = get_environment(config.get("environment", {}))

    # Split agent config from iterative config
    agent_config = dict(config.get("agent", {}))
    agent_config.pop("agent_class", None)  # IterativeRunner creates agents directly
    iterative_config = dict(config.get("iterative", {}))

    runner = IterativeRunner(
        model,
        env,
        agent_config=agent_config,
        iterative_config=iterative_config,
    )

    console.print(f"Running iterative task in [bold green]{resolved_output_dir}[/bold green]")

    run_exception: Exception | None = None
    close_exception: Exception | None = None
    result: dict[str, Any] = {}

    try:
        env.prepare(
            task=resolved_task,
            task_id=resolved_task_id,
            start_url=resolved_start_url,
            task_record=task_record,
        )
        result = runner.run(
            resolved_task,
            task_id=resolved_task_id or "",
            start_url=resolved_start_url or "",
            base_output_dir=resolved_output_dir,
        )
    except Exception as exc:
        run_exception = exc
        result.setdefault("exit_status", type(exc).__name__)
        result.setdefault("submission", "")
        result.setdefault("final_response", "")
        result["run_exception"] = str(exc)
    finally:
        try:
            env.close()
        except Exception as exc:
            close_exception = exc
            result.setdefault("exit_status", type(exc).__name__)
            result.setdefault("submission", "")
            result.setdefault("final_response", "")
            result.setdefault("run_exception", str(exc))
            result["close_exception"] = str(exc)
            if run_exception is None:
                run_exception = exc

    # Export judge artifacts (uses the last round's output by default)
    judge_artifacts = export_online_mind2web_artifacts(
        output_dir=resolved_output_dir,
        task=resolved_task,
        task_id=resolved_task_id,
        start_url=resolved_start_url,
        agent_result=result,
    )
    result["_output_dir"] = str(resolved_output_dir)
    result["_judge_artifacts"] = judge_artifacts

    # Persist top-level iterative summary
    summary = {
        "task_id": resolved_task_id or "",
        "task": resolved_task,
        "iterative_success": result.get("_iterative_success", False),
        "iterative_rounds": result.get("_iterative_rounds", 0),
        "final_round_dir": result.get("_final_round_dir", ""),
    }
    (resolved_output_dir / "iterative_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if close_exception is not None:
        result["_close_exception"] = str(close_exception)

    success_marker = "✓ SUCCESS" if result.get("_iterative_success") else "✗ FAILURE"
    console.print(
        f"{success_marker} after {result.get('_iterative_rounds', '?')} round(s). "
        f"Output: [bold green]{resolved_output_dir}[/bold green]"
    )

    if run_exception is not None:
        raise run_exception
    return result


@app.command()
def main(
    task: str | None = typer.Option(None, "-t", "--task"),
    task_id: str | None = typer.Option(None, "--task-id"),
    tasks_file: Path | None = typer.Option(None, "--tasks-file"),
    start_url: str | None = typer.Option(None, "--start-url"),
    config_spec: list[str] = typer.Option([DEFAULT_CONFIG], "-c", "--config"),
    output_dir: Path | None = typer.Option(None, "-o", "--output-dir"),
    debug: bool = typer.Option(False, "--debug"),
) -> Any:
    return run_one(
        task=task,
        task_id=task_id,
        tasks_file=tasks_file,
        start_url=start_url,
        config_spec=config_spec,
        output_dir=output_dir,
        debug=debug,
    )

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miniswewebagent.agents import get_agent
from miniswewebagent.config import get_config_from_spec
from miniswewebagent.environments import get_environment
from miniswewebagent.models import get_model
from miniswewebagent.tasks.om2w import load_om2w_task
from miniswewebagent.utils.om2w_eval import export_online_mind2web_artifacts
from miniswewebagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG = "mini.yaml"

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
    if resolved_task_id:
        if not resolved_tasks_file:
            raise ValueError("--task-id requires --tasks-file or run.tasks_file in config.")
        task_record = load_om2w_task(resolved_tasks_file, resolved_task_id)
        resolved_task = resolved_task or task_record["task"]
        resolved_start_url = resolved_start_url or task_record["start_url"]

    if not resolved_task:
        raise ValueError("A task is required. Use --task or --task-id.")

    resolved_output_dir = resolved_output_dir or _timestamped_output_dir(
        output_dir or config.get("environment", {}).get("output_dir") or "outputs",
        resolved_task_id,
    )

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
            "agent": {
                "output_path": str(resolved_output_dir / "trajectory.json"),
            },
        },
    )

    model = get_model(config.get("model", {}))
    env = get_environment(config.get("environment", {}))
    agent = get_agent(model, env, config.get("agent", {}), default_type="default")

    console.print(f"Running task in [bold green]{resolved_output_dir}[/bold green]")
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
        result = agent.run(
            resolved_task,
            task_id=resolved_task_id or "",
            start_url=resolved_start_url or "",
        )
    except Exception as exc:
        run_exception = exc
        if getattr(agent, "messages", None):
            result = dict(agent.messages[-1].get("extra", {}))
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
    judge_artifacts = export_online_mind2web_artifacts(
        output_dir=resolved_output_dir,
        task=resolved_task,
        task_id=resolved_task_id,
        start_url=resolved_start_url,
        agent_result=result,
    )
    result["_output_dir"] = str(resolved_output_dir)
    result["_judge_artifacts"] = judge_artifacts
    if close_exception is not None:
        result["_close_exception"] = str(close_exception)
    console.print(result.get("final_response") or result.get("submission") or "Task finished.")
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
    debug: bool = typer.Option(False, "--debug", help="Launch headed local Playwright with devtools and keep it open for inspection at the end."),
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

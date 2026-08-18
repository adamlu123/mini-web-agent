from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miniswewebagent.config import get_config_from_spec, snapshot_config_specs
from miniswewebagent.run.mini import DEFAULT_CONFIG, run_one
from miniswewebagent.utils.om2w_eval import (
    judge_result_file_path,
    run_online_mind2web_judge,
    split_jsonl_lines,
)
from miniswewebagent.utils.om2w_tasks import load_om2w_tasks
from miniswewebagent.utils.serialize import recursive_merge

app = typer.Typer(no_args_is_help=False)
console = Console(highlight=False)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BENCHMARK_CONFIG = "archive/benchmark/om2w_hard_local_workspace.yaml"
DEFAULT_OM2W_CONFIGS = [DEFAULT_CONFIG, DEFAULT_BENCHMARK_CONFIG]
DEFAULT_LOG_ROOT = REPO_ROOT / "logs"
DEFAULT_JUDGE_PYTHON = Path(sys.executable)
# Both scripts/eval entry points read task.json + screenshots/ straight out of
# each task directory and share one CLI. This one is layout-aware, so the batch
# default is its own default layout, `step-scripts`: the judge reads the executed
# steps/step_<id>.sh as the action history. Point run.judge_script at
# scripts/eval/persistent_cli.py to score the browser-steps.jsonl natural-language
# actions instead.
DEFAULT_JUDGE_SCRIPT = REPO_ROOT / "scripts" / "eval" / "persistent_cli_steps.py"


def _merged_config(config_spec: list[str]) -> dict[str, Any]:
    return recursive_merge(*(get_config_from_spec(spec) for spec in config_spec))


def _model_slug(model_name: str) -> str:
    return model_name.replace("-", "").replace(".", "").replace("/", "_")


def _write_batch_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip())
        handle.write("\n")


def _select_tasks(
    tasks_file: Path,
    task_ids: list[str],
    limit: int,
    task_level: str | None,
    offset: int = 0,
) -> list[dict[str, object]]:
    tasks = load_om2w_tasks(tasks_file)
    if task_level and task_level.lower() != "all":
        tasks = [task for task in tasks if task.get("level") == task_level]
    if task_ids:
        selected_ids = set(task_ids)
        tasks = [task for task in tasks if task["task_id"] in selected_ids]
    if offset > 0:
        tasks = tasks[offset:]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def _assign_model_endpoints(
    tasks: list[dict[str, object]],
    endpoints: list[str],
) -> dict[str, str]:
    """Pin exactly one endpoint per task, spread uniformly over the pool.

    Round-robin over the already-ordered task list, so with N tasks and E
    endpoints each endpoint gets either floor(N/E) or ceil(N/E) tasks, and a
    task keeps the same endpoint for every step of its trajectory.
    """
    if not endpoints:
        return {}
    return {
        str(task["task_id"]): endpoints[index % len(endpoints)]
        for index, task in enumerate(tasks)
    }


def _record_task_endpoint(task_output_dir: Path, task_id: str, endpoint: str) -> None:
    """Persist the endpoint assignment next to the task's trajectory."""
    task_output_dir.mkdir(parents=True, exist_ok=True)
    (task_output_dir / "model_endpoint.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "model_endpoint": endpoint,
                "assigned_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_eval_rows(result_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not result_file.exists():
        return rows
    for line in split_jsonl_lines(result_file.read_text(encoding="utf-8")):
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _resolve_judge_api_key(*, endpoint_target_uri: str) -> str:
    if endpoint_target_uri:
        return (
            os.environ.get("OPENAI_GATEWAY_API_KEY", "")
            or os.environ.get("PHYAGI_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
    return os.environ.get("OPENAI_API_KEY", "")


def _run_task_worker(
    *,
    task: dict[str, object],
    tasks_file: Path,
    config_spec: list[str],
    output_root: Path,
    log_dir: Path,
    session_id_prefix: str | None = None,
    model_endpoint: str | None = None,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    task_output_dir = output_root / task_id
    task_log_path = log_dir / f"{task_id}.log"
    task_log_path.parent.mkdir(parents=True, exist_ok=True)

    def row(*, status: str, exit_status: str, error: str = "") -> dict[str, Any]:
        return {
            "task_id": task_id,
            "task": str(task["task"]),
            "level": str(task.get("level", "")),
            "status": status,
            "error": error,
            "exit_status": exit_status,
            "output_dir": str(task_output_dir),
            "log_path": str(task_log_path),
            "result_json": str(task_output_dir / "result.json"),
            "model_endpoint": model_endpoint or "",
        }

    auto_model_overrides: dict[str, Any] = {}
    if session_id_prefix:
        auto_model_overrides["session_id"] = f"{session_id_prefix}_{task_id}"

    model_overrides: dict[str, Any] = {}
    if model_endpoint:
        model_overrides["model_name"] = model_endpoint
        # Written before the run so the assignment survives a crash.
        _record_task_endpoint(task_output_dir, task_id, model_endpoint)

    with task_log_path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                result = run_one(
                    task=str(task["task"]),
                    task_id=task_id,
                    tasks_file=tasks_file,
                    start_url=str(task.get("start_url", "")),
                    config_spec=config_spec,
                    resolved_output_dir=task_output_dir,
                    snapshot_config=False,
                    auto_model_overrides=auto_model_overrides or None,
                    model_overrides=model_overrides or None,
                )
                return row(status="ok", exit_status=str(result.get("exit_status", "")))
            except Exception as exc:
                print(traceback.format_exc())
                return row(status="error", exit_status=type(exc).__name__, error=str(exc))


@app.command()
def main(
    tasks_file: Path | None = typer.Option(None, "--tasks-file", help="Path to an Online-Mind2Web JSON file."),
    task_id: list[str] = typer.Option([], "--task-id", help="Only run the specified task id(s)."),
    limit: int = typer.Option(0, "--limit", help="Run only the first N selected tasks."),
    offset: int = typer.Option(0, "--offset", help="Skip the first N selected tasks (0-based). Combine with --limit to run a row range."),
    model_endpoint: list[str] = typer.Option(
        [],
        "--model-endpoint",
        help="Model endpoint/deployment to spread tasks over (repeatable). Each task is pinned to exactly one.",
    ),
    task_level: str | None = typer.Option(None, "--task-level", help="Filter tasks by level, e.g. hard."),
    workers: int = typer.Option(0, "--workers", help="Parallel worker processes. Defaults from config."),
    evaluate: bool | None = typer.Option(None, "--evaluate/--no-evaluate", help="Run judge after generation."),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Judge model name."),
    judge_runs: int = typer.Option(0, "--judge-runs", help="Number of parallel judge runs. Defaults from config or 3."),
    judge_num_proc: int = typer.Option(0, "--judge-num-proc", help="Judge worker processes. Defaults from config."),
    judge_python: Path | None = typer.Option(None, "--judge-python", help="Python executable for Online-Mind2Web judge."),
    judge_script: Path | None = typer.Option(None, "--judge-script", help="Path to the Online-Mind2Web judge entry point."),
    judge_endpoint: str | None = typer.Option(
        None,
        "--judge-endpoint",
        help="Judge responses API endpoint. Defaults to official OpenAI when unset.",
    ),
    log_root: Path | None = typer.Option(None, "--log-root", help="Directory for batch logs."),
    config_spec: list[str] = typer.Option(DEFAULT_OM2W_CONFIGS, "-c", "--config"),
    output_dir: Path | None = typer.Option(None, "-o", "--output-dir", help="Batch output root directory."),
) -> None:
    config = _merged_config(config_spec)
    run_config = config.get("run", {})
    agent_config = config.get("agent", {})
    env_config = config.get("environment", {})
    model_config = config.get("model", {})

    resolved_tasks_file_value = tasks_file or run_config.get("tasks_file")
    if not resolved_tasks_file_value:
        raise typer.BadParameter("--tasks-file is required unless run.tasks_file is set in config.")
    resolved_tasks_file = Path(resolved_tasks_file_value)

    resolved_task_level = task_level or run_config.get("task_level") or ""
    resolved_workers = max(1, int(workers or run_config.get("parallel_processes") or 1))
    resolved_evaluate = bool(run_config.get("judge_enabled", False)) if evaluate is None else evaluate
    resolved_judge_model = str(judge_model or run_config.get("judge_model") or "gpt-4o")
    resolved_judge_runs = max(1, int(judge_runs or run_config.get("judge_runs") or 3))
    resolved_judge_python = Path(judge_python or run_config.get("judge_python") or DEFAULT_JUDGE_PYTHON)
    resolved_judge_script = Path(judge_script or run_config.get("judge_script") or DEFAULT_JUDGE_SCRIPT)
    resolved_judge_endpoint = str(judge_endpoint or run_config.get("judge_endpoint") or "")
    resolved_log_root = Path(log_root or run_config.get("logs_root") or DEFAULT_LOG_ROOT).expanduser()
    resolved_offset = max(0, int(offset or run_config.get("task_offset") or 0))

    tasks = _select_tasks(
        resolved_tasks_file,
        task_id,
        limit,
        resolved_task_level,
        offset=resolved_offset,
    )

    resolved_model_endpoints = [
        str(item).strip()
        for item in (model_endpoint or run_config.get("model_endpoints") or [])
        if str(item).strip()
    ]
    endpoint_assignments = _assign_model_endpoints(tasks, resolved_model_endpoints)

    # Judge parallelism defaults to one worker per task, not to the generation
    # worker count: judging is IO-bound on the judge endpoint, not on browsers.
    resolved_judge_num_proc = max(
        1, int(judge_num_proc or run_config.get("judge_num_proc") or len(tasks))
    )

    model_name = str(model_config.get("model_name", "model"))
    step_limit = int(agent_config.get("step_limit", 0) or 0)
    session_slug = "bb" if env_config.get("browserbase_enabled") else "local"
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = (
        f"om2w_260220_{resolved_task_level or 'all'}_"
        f"{_model_slug(model_name)}_step{step_limit}_p{resolved_workers}_{session_slug}_{batch_stamp}"
    )

    base_output_root = Path(output_dir or env_config.get("output_dir") or "outputs").expanduser()
    batch_output_dir = base_output_root / batch_name if output_dir is None else Path(output_dir).expanduser()
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_dir = snapshot_config_specs(config_spec, batch_output_dir, merged_config=config)

    batch_log_dir = resolved_log_root / batch_name
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    batch_log_path = batch_log_dir / "batch.log"
    generation_summary_path = batch_log_dir / "generation_summary.json"
    run_summary_path = batch_log_dir / "run_summary.json"

    _write_batch_log_line(batch_log_path, f"batch_name={batch_name}")
    _write_batch_log_line(batch_log_path, f"tasks_file={resolved_tasks_file}")
    _write_batch_log_line(batch_log_path, f"task_level={resolved_task_level or '<all>'}")
    _write_batch_log_line(batch_log_path, f"workers={resolved_workers}")
    _write_batch_log_line(batch_log_path, f"judge_endpoint={resolved_judge_endpoint or '<openai>'}")
    _write_batch_log_line(batch_log_path, f"output_dir={batch_output_dir}")
    _write_batch_log_line(batch_log_path, f"config_snapshot_dir={config_snapshot_dir}")

    endpoint_counts: dict[str, int] = {}
    if endpoint_assignments:
        for endpoint in endpoint_assignments.values():
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
        _write_batch_log_line(
            batch_log_path, f"model_endpoints={json.dumps(endpoint_counts, sort_keys=True)}"
        )
        (batch_output_dir / "model_endpoint_assignments.json").write_text(
            json.dumps(
                {"counts": endpoint_counts, "assignments": endpoint_assignments},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        console.print(f"Endpoint distribution: {endpoint_counts}")

    console.print(f"Running {len(tasks)} Online-Mind2Web task(s)")
    console.print(f"Outputs: [bold green]{batch_output_dir}[/bold green]")
    console.print(f"Logs: [bold green]{batch_log_dir}[/bold green]")

    generation_rows: list[dict[str, Any]] = []
    if resolved_workers <= 1:
        for index, task in enumerate(tasks, start=1):
            row = _run_task_worker(
                task=task,
                tasks_file=resolved_tasks_file,
                config_spec=config_spec,
                output_root=batch_output_dir,
                log_dir=batch_log_dir,
                session_id_prefix=batch_name,
                model_endpoint=endpoint_assignments.get(str(task["task_id"])),
            )
            generation_rows.append(row)
            console.print(f"[{index}/{len(tasks)}] {row['task_id']} -> {row['status']}")
            _write_batch_log_line(batch_log_path, json.dumps(row, ensure_ascii=True))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            futures = {
                executor.submit(
                    _run_task_worker,
                    task=task,
                    tasks_file=resolved_tasks_file,
                    config_spec=config_spec,
                    output_root=batch_output_dir,
                    log_dir=batch_log_dir,
                    session_id_prefix=batch_name,
                    model_endpoint=endpoint_assignments.get(str(task["task_id"])),
                ): task
                for task in tasks
            }
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                generation_rows.append(row)
                completed += 1
                console.print(f"[{completed}/{len(tasks)}] {row['task_id']} -> {row['status']}")
                _write_batch_log_line(batch_log_path, json.dumps(row, ensure_ascii=True))

    generation_rows.sort(key=lambda row: row["task_id"])
    generation_summary_path.write_text(json.dumps(generation_rows, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "batch_name": batch_name,
        "tasks_file": str(resolved_tasks_file),
        "task_level": resolved_task_level,
        "workers": resolved_workers,
        "output_dir": str(batch_output_dir),
        "config_snapshot_dir": str(config_snapshot_dir),
        "log_dir": str(batch_log_dir),
        "n_tasks": len(tasks),
        "task_offset": resolved_offset,
        "model_endpoints": resolved_model_endpoints,
        "model_endpoint_counts": endpoint_counts,
        "n_failed_generation": sum(1 for row in generation_rows if row["status"] != "ok"),
        "judge_enabled": resolved_evaluate,
        "judge_model": resolved_judge_model,
        "judge_script": str(resolved_judge_script),
        "judge_runs": resolved_judge_runs,
        "judge_num_proc": resolved_judge_num_proc,
        "judge_endpoint": resolved_judge_endpoint,
    }

    if resolved_evaluate:
        api_key = _resolve_judge_api_key(endpoint_target_uri=resolved_judge_endpoint)
        if not api_key:
            required = (
                "OPENAI_GATEWAY_API_KEY, PHYAGI_API_KEY, or OPENAI_API_KEY"
                if resolved_judge_endpoint
                else "OPENAI_API_KEY"
            )
            raise RuntimeError(f"{required} is required to run the Online-Mind2Web judge.")

        def run_single_judge(run_index: int) -> dict[str, Any]:
            eval_output_dir = batch_output_dir.parent / f"{batch_output_dir.name}_eval_{run_index}"
            run_log_path = batch_log_dir / f"judge_{run_index}.log"
            completed = run_online_mind2web_judge(
                judge_python=resolved_judge_python,
                judge_script=resolved_judge_script,
                trajectories_dir=batch_output_dir,
                output_dir=eval_output_dir,
                judge_model=resolved_judge_model,
                num_proc=resolved_judge_num_proc,
                api_key=api_key,
                endpoint_target_uri=resolved_judge_endpoint,
                log_path=run_log_path,
            )
            result_file = judge_result_file_path(eval_output_dir, resolved_judge_model)
            eval_rows = _read_eval_rows(result_file)
            return {
                "run_index": run_index,
                "eval_output_dir": str(eval_output_dir),
                "judge_returncode": completed.returncode,
                "judge_result_file": str(result_file),
                "judge_log_path": str(run_log_path),
                "n_eval_rows": len(eval_rows),
                "n_judge_success": sum(1 for row in eval_rows if row.get("predicted_label") == 1),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=resolved_judge_runs) as executor:
            eval_runs = list(executor.map(run_single_judge, range(1, resolved_judge_runs + 1)))

        summary["eval_runs"] = eval_runs

    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()

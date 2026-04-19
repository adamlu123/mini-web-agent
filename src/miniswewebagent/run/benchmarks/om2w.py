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
from miniswewebagent.run.mini import DEFAULT_CONFIG, _timestamped_output_dir
from miniswewebagent.run.mini import run_one as run_one_default
from miniswewebagent.tasks.om2w import load_om2w_tasks
from miniswewebagent.utils.om2w_eval import run_online_mind2web_judge
from miniswewebagent.utils.serialize import recursive_merge

app = typer.Typer(no_args_is_help=False)
console = Console(highlight=False)

DEFAULT_BENCHMARK_CONFIG = "benchmark/om2w_hard_local_workspace.yaml"
DEFAULT_OM2W_CONFIGS = [DEFAULT_CONFIG, DEFAULT_BENCHMARK_CONFIG]
DEFAULT_LOG_ROOT = Path("/Users/lu/Documents/sandbox/mini-swe-agent/logs")
DEFAULT_JUDGE_PYTHON = Path(sys.executable)
DEFAULT_JUDGE_SCRIPT = Path(__file__).resolve().parents[4] / "om2w_judge" / "run.py"


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
) -> list[dict[str, object]]:
    tasks = load_om2w_tasks(tasks_file)
    if task_level and task_level.lower() != "all":
        tasks = [task for task in tasks if task.get("level") == task_level]
    if task_ids:
        selected_ids = set(task_ids)
        tasks = [task for task in tasks if task["task_id"] in selected_ids]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def _read_eval_rows(result_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not result_file.exists():
        return rows
    for line in result_file.read_text(encoding="utf-8").splitlines():
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


def _resolve_run_one(iterative: bool):
    if iterative:
        from miniswewebagent.run.iterative import run_one as run_one_iter
        return run_one_iter
    return run_one_default


def _run_task_worker(
    *,
    task: dict[str, object],
    tasks_file: Path,
    config_spec: list[str],
    output_root: Path,
    log_dir: Path,
    iterative: bool = False,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    task_output_dir = output_root / task_id
    task_log_path = log_dir / f"{task_id}.log"
    task_log_path.parent.mkdir(parents=True, exist_ok=True)

    run_one = _resolve_run_one(iterative)

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
                )
                row = {
                    "task_id": task_id,
                    "task": str(task["task"]),
                    "level": str(task.get("level", "")),
                    "status": "ok",
                    "error": "",
                    "exit_status": str(result.get("exit_status", "")),
                    "output_dir": str(task_output_dir),
                    "log_path": str(task_log_path),
                    "result_json": str(task_output_dir / "result.json"),
                }
                if iterative:
                    row["iterative_success"] = bool(result.get("_iterative_success", False))
                    row["iterative_rounds"] = int(result.get("_iterative_rounds", 0))
                return row
            except Exception as exc:
                print(traceback.format_exc())
                row = {
                    "task_id": task_id,
                    "task": str(task["task"]),
                    "level": str(task.get("level", "")),
                    "status": "error",
                    "error": str(exc),
                    "exit_status": type(exc).__name__,
                    "output_dir": str(task_output_dir),
                    "log_path": str(task_log_path),
                    "result_json": str(task_output_dir / "result.json"),
                }
                if iterative:
                    row["iterative_success"] = False
                    row["iterative_rounds"] = 0
                return row


@app.command()
def main(
    tasks_file: Path | None = typer.Option(None, "--tasks-file", help="Path to an Online-Mind2Web JSON file."),
    task_id: list[str] = typer.Option([], "--task-id", help="Only run the specified task id(s)."),
    limit: int = typer.Option(0, "--limit", help="Run only the first N selected tasks."),
    task_level: str | None = typer.Option(None, "--task-level", help="Filter tasks by level, e.g. hard."),
    workers: int = typer.Option(0, "--workers", help="Parallel worker processes. Defaults from config."),
    evaluate: bool | None = typer.Option(None, "--evaluate/--no-evaluate", help="Run judge after generation."),
    judge_model: str | None = typer.Option(None, "--judge-model", help="Judge model name."),
    judge_runs: int = typer.Option(0, "--judge-runs", help="Number of parallel judge runs. Defaults from config or 3."),
    judge_num_proc: int = typer.Option(0, "--judge-num-proc", help="Judge worker processes. Defaults from config."),
    judge_python: Path | None = typer.Option(None, "--judge-python", help="Python executable for Online-Mind2Web judge."),
    judge_script: Path | None = typer.Option(None, "--judge-script", help="Path to Online-Mind2Web src/run.py."),
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
    is_iterative = bool(config.get("iterative"))

    resolved_tasks_file_value = tasks_file or run_config.get("tasks_file")
    if not resolved_tasks_file_value:
        raise typer.BadParameter("--tasks-file is required unless run.tasks_file is set in config.")
    resolved_tasks_file = Path(resolved_tasks_file_value)

    resolved_task_level = task_level or run_config.get("task_level") or ""
    resolved_workers = max(1, int(workers or run_config.get("parallel_processes") or 1))
    resolved_evaluate = bool(run_config.get("judge_enabled", False)) if evaluate is None else evaluate
    resolved_judge_model = str(judge_model or run_config.get("judge_model") or "gpt-4o")
    resolved_judge_runs = max(1, int(judge_runs or run_config.get("judge_runs") or 3))
    resolved_judge_num_proc = max(1, int(judge_num_proc or run_config.get("judge_num_proc") or resolved_workers))
    resolved_judge_python = Path(judge_python or run_config.get("judge_python") or DEFAULT_JUDGE_PYTHON)
    resolved_judge_script = Path(judge_script or run_config.get("judge_script") or DEFAULT_JUDGE_SCRIPT)
    resolved_judge_endpoint = str(judge_endpoint or run_config.get("judge_endpoint") or "")
    resolved_log_root = Path(log_root or run_config.get("logs_root") or DEFAULT_LOG_ROOT).expanduser()

    tasks = _select_tasks(
        resolved_tasks_file,
        task_id,
        limit,
        resolved_task_level,
    )

    # Default judge parallelism to the number of tasks when not explicitly set
    if not (judge_num_proc or run_config.get("judge_num_proc")):
        resolved_judge_num_proc = max(1, len(tasks))

    model_name = str(model_config.get("model_name", "model"))
    step_limit = int(agent_config.get("step_limit", 0) or 0)
    session_slug = "bb" if env_config.get("browserbase_enabled") else "local"
    iter_slug = f"_iter_r{config.get('iterative', {}).get('max_rounds', 5)}" if is_iterative else ""
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = (
        f"om2w_260220_{resolved_task_level or 'all'}_"
        f"{_model_slug(model_name)}_step{step_limit}{iter_slug}_p{resolved_workers}_{session_slug}_{batch_stamp}"
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

    mode_label = "iterative" if is_iterative else "standard"
    console.print(f"Running {len(tasks)} Online-Mind2Web task(s) ({mode_label} mode)")
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
                iterative=is_iterative,
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
                    iterative=is_iterative,
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
        "iterative": is_iterative,
        "output_dir": str(batch_output_dir),
        "config_snapshot_dir": str(config_snapshot_dir),
        "log_dir": str(batch_log_dir),
        "n_tasks": len(tasks),
        "n_failed_generation": sum(1 for row in generation_rows if row["status"] != "ok"),
        "judge_enabled": resolved_evaluate,
        "judge_model": resolved_judge_model,
        "judge_runs": resolved_judge_runs,
        "judge_num_proc": resolved_judge_num_proc,
        "judge_endpoint": resolved_judge_endpoint,
    }

    if is_iterative:
        n_iter_success = sum(1 for r in generation_rows if r.get("iterative_success"))
        avg_rounds = (
            sum(r.get("iterative_rounds", 0) for r in generation_rows) / len(generation_rows)
            if generation_rows else 0
        )
        summary["iterative_success"] = n_iter_success
        summary["iterative_success_rate"] = f"{(n_iter_success / len(tasks) * 100):.1f}%" if tasks else "0%"
        summary["iterative_avg_rounds"] = round(avg_rounds, 2)
        summary["iterative_max_rounds"] = int(config.get("iterative", {}).get("max_rounds", 5))

    if resolved_evaluate and not is_iterative:
        api_key = _resolve_judge_api_key(endpoint_target_uri=resolved_judge_endpoint)
        if not api_key:
            if resolved_judge_endpoint:
                raise RuntimeError(
                    "OPENAI_GATEWAY_API_KEY, PHYAGI_API_KEY, or OPENAI_API_KEY is required to run the Online-Mind2Web judge with a gateway endpoint."
                )
            raise RuntimeError("OPENAI_API_KEY is required to run the Online-Mind2Web judge.")

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
            result_file = eval_output_dir / (
                f"WebJudge_Online_Mind2Web_Sandbox_eval_{resolved_judge_model}_score_threshold_3_auto_eval_results.json"
            )
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

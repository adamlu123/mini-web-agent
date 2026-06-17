from __future__ import annotations

import concurrent.futures
import contextlib
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import typer
from rich.console import Console

from miniswewebagent.config import get_config_from_spec, snapshot_config_specs
from miniswewebagent.run.mini import DEFAULT_CONFIG, run_one
from miniswewebagent.utils.serialize import recursive_merge

app = typer.Typer(no_args_is_help=False)
console = Console(highlight=False)

TaskFamily = Literal["all", "flight", "hotel"]

DEFAULT_TASKS_FILE = Path(__file__).with_name("webtail_flight_hotel.json")
DEFAULT_BENCHMARK_CONFIG = "benchmark/webtaibench_xml.yaml"
DEFAULT_WEBTAIL_CONFIGS = [
    DEFAULT_CONFIG,
    DEFAULT_BENCHMARK_CONFIG,
    "environment.browserbase_enabled=true",
    "environment.headless=true",
    "environment.slow_mo_ms=0",
    "environment.browser_timeout_ms=12000",
    "environment.browser_navigation_timeout_ms=45000",
    "environment.observation_timeout_ms=6000",
    "agent.step_limit=25",
    "run.parallel_processes=33",
    "run.logs_root=logs",
]
DEFAULT_LOG_ROOT = Path("logs")


def _merged_config(config_spec: list[str]) -> dict[str, Any]:
    return recursive_merge(*(get_config_from_spec(spec) for spec in config_spec))


def _model_slug(model_name: str) -> str:
    return model_name.replace("-", "").replace(".", "").replace("/", "_")


def _write_batch_log_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip())
        handle.write("\n")


def load_webtail_flight_hotel_tasks(tasks_file: str | Path = DEFAULT_TASKS_FILE) -> list[dict[str, Any]]:
    path = Path(tasks_file).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("tasks", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported WebTail task payload in {path}")

    tasks: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError(f"Unsupported task row in {path}: {item!r}")
        task_id = str(item.get("task_id", "")).strip()
        task = str(item.get("task", "")).strip()
        start_url = str(item.get("start_url") or item.get("website_url") or "").strip()
        family = str(item.get("source_family") or item.get("family") or "").strip()
        if not task_id or not task or not start_url:
            raise ValueError(f"Invalid WebTail task row in {path}: {item!r}")
        tasks.append(
            {
                "task_id": task_id,
                "task": task,
                "start_url": start_url,
                "family": family,
                "raw": item,
            }
        )
    return tasks


def _select_tasks(
    tasks_file: Path,
    task_ids: list[str],
    limit: int,
    family: TaskFamily,
) -> list[dict[str, Any]]:
    tasks = load_webtail_flight_hotel_tasks(tasks_file)
    if family != "all":
        tasks = [task for task in tasks if task.get("family") == family]
    if task_ids:
        selected_ids = set(task_ids)
        tasks = [task for task in tasks if task["task_id"] in selected_ids]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


def _run_task_worker(
    *,
    task: dict[str, object],
    config_spec: list[str],
    output_root: Path,
    log_dir: Path,
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    task_output_dir = output_root / task_id
    task_log_path = log_dir / f"{task_id}.log"
    task_log_path.parent.mkdir(parents=True, exist_ok=True)

    with task_log_path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            try:
                result = run_one(
                    task=str(task["task"]),
                    task_id=task_id,
                    start_url=str(task.get("start_url", "")),
                    config_spec=config_spec,
                    resolved_output_dir=task_output_dir,
                    snapshot_config=False,
                )
                return {
                    "task_id": task_id,
                    "task": str(task["task"]),
                    "family": str(task.get("family", "")),
                    "status": "ok",
                    "error": "",
                    "exit_status": str(result.get("exit_status", "")),
                    "output_dir": str(task_output_dir),
                    "log_path": str(task_log_path),
                    "result_json": str(task_output_dir / "result.json"),
                }
            except Exception as exc:
                print(traceback.format_exc())
                return {
                    "task_id": task_id,
                    "task": str(task["task"]),
                    "family": str(task.get("family", "")),
                    "status": "error",
                    "error": str(exc),
                    "exit_status": type(exc).__name__,
                    "output_dir": str(task_output_dir),
                    "log_path": str(task_log_path),
                    "result_json": str(task_output_dir / "result.json"),
                }


@app.command()
def main(
    tasks_file: Path = typer.Option(DEFAULT_TASKS_FILE, "--tasks-file", help="Path to a normalized WebTail hotel+flight JSON file."),
    task_id: list[str] = typer.Option([], "--task-id", help="Only run the specified task id(s)."),
    limit: int = typer.Option(0, "--limit", help="Run only the first N selected tasks."),
    family: TaskFamily = typer.Option("all", "--family", help="Filter by task family."),
    workers: int = typer.Option(0, "--workers", help="Parallel worker processes. Defaults from config."),
    log_root: Path | None = typer.Option(None, "--log-root", help="Directory for batch logs."),
    config_spec: list[str] = typer.Option(DEFAULT_WEBTAIL_CONFIGS, "-c", "--config"),
    output_dir: Path | None = typer.Option(None, "-o", "--output-dir", help="Batch output root directory."),
) -> None:
    config = _merged_config(config_spec)
    run_config = config.get("run", {})
    agent_config = config.get("agent", {})
    env_config = config.get("environment", {})
    model_config = config.get("model", {})

    resolved_tasks_file = Path(tasks_file).expanduser()
    resolved_workers = max(1, int(workers or run_config.get("parallel_processes") or 1))
    resolved_log_root = Path(log_root or run_config.get("logs_root") or DEFAULT_LOG_ROOT).expanduser()

    tasks = _select_tasks(
        resolved_tasks_file,
        task_id,
        limit,
        family,
    )

    model_name = str(model_config.get("model_name", "model"))
    step_limit = int(agent_config.get("step_limit", 0) or 0)
    session_slug = "bb" if env_config.get("browserbase_enabled") else "local"
    batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_name = (
        f"webtail_flight_hotel_{family}_"
        f"{_model_slug(model_name)}_step{step_limit}_p{resolved_workers}_{session_slug}_{batch_stamp}"
    )

    base_output_root = Path(output_dir or env_config.get("output_dir") or "outputs/webtail_flight_hotel").expanduser()
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
    _write_batch_log_line(batch_log_path, f"family={family}")
    _write_batch_log_line(batch_log_path, f"workers={resolved_workers}")
    _write_batch_log_line(batch_log_path, f"output_dir={batch_output_dir}")
    _write_batch_log_line(batch_log_path, f"config_snapshot_dir={config_snapshot_dir}")
    _write_batch_log_line(batch_log_path, f"config_spec={json.dumps(config_spec)}")

    console.print(f"Running {len(tasks)} WebTail flight/hotel task(s)")
    console.print(f"Outputs: [bold green]{batch_output_dir}[/bold green]")
    console.print(f"Logs: [bold green]{batch_log_dir}[/bold green]")

    generation_rows: list[dict[str, Any]] = []
    if resolved_workers <= 1:
        for index, task in enumerate(tasks, start=1):
            row = _run_task_worker(
                task=task,
                config_spec=config_spec,
                output_root=batch_output_dir,
                log_dir=batch_log_dir,
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
                    config_spec=config_spec,
                    output_root=batch_output_dir,
                    log_dir=batch_log_dir,
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
        "family": family,
        "workers": resolved_workers,
        "output_dir": str(batch_output_dir),
        "config_snapshot_dir": str(config_snapshot_dir),
        "log_dir": str(batch_log_dir),
        "config_spec": config_spec,
        "n_tasks": len(tasks),
        "n_failed_generation": sum(1 for row in generation_rows if row["status"] != "ok"),
    }

    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()

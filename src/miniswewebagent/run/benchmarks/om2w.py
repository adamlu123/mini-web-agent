from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from miniswewebagent.run.mini import DEFAULT_CONFIG, run_one
from miniswewebagent.tasks.om2w import load_om2w_tasks

app = typer.Typer(no_args_is_help=True)
console = Console(highlight=False)
DEFAULT_BENCHMARK_CONFIG = "benchmark/browserbase.yaml"
DEFAULT_OM2W_CONFIGS = [DEFAULT_CONFIG, DEFAULT_BENCHMARK_CONFIG]


def _select_tasks(tasks_file: Path, task_ids: list[str], limit: int) -> list[dict[str, object]]:
    tasks = load_om2w_tasks(tasks_file)
    if task_ids:
        selected_ids = set(task_ids)
        tasks = [task for task in tasks if task["task_id"] in selected_ids]
    if limit > 0:
        tasks = tasks[:limit]
    return tasks


@app.command()
def main(
    tasks_file: Path = typer.Option(..., "--tasks-file", help="Path to an Online-Mind2Web JSON file."),
    task_id: list[str] = typer.Option([], "--task-id", help="Only run the specified task id(s)."),
    limit: int = typer.Option(0, "--limit", help="Run only the first N selected tasks."),
    config_spec: list[str] = typer.Option(DEFAULT_OM2W_CONFIGS, "-c", "--config"),
    output_dir: Path | None = typer.Option(None, "-o", "--output-dir"),
) -> None:
    tasks = _select_tasks(tasks_file, task_id, limit)

    console.print(f"Running {len(tasks)} Online-Mind2Web task(s)")
    for index, task in enumerate(tasks, start=1):
        console.print(f"[{index}/{len(tasks)}] {task['task_id']}")
        try:
            run_one(
                task=task["task"],
                task_id=task["task_id"],
                tasks_file=tasks_file,
                start_url=task["start_url"],
                config_spec=config_spec,
                output_dir=output_dir,
            )
        except Exception as exc:
            console.print(f"[red]Task failed:[/red] {task['task_id']} -> {exc}")


if __name__ == "__main__":
    app()
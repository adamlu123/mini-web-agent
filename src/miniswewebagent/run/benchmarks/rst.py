"""Batch runner for RST terminal tasks.

Deliberately much smaller than ``benchmarks/om2w.py``: RST tasks carry their own
verifier, so there is no WebJudge stage, no screenshot plumbing and no separate
``*_eval_{1,2,3}`` fan-out. Each task is one process that

1. builds/starts the task container (``terminal_docker``),
2. runs the agent loop to termination,
3. runs the task's own private ``tests/test.sh`` for the external score,
4. writes ``trajectory.json`` plus ``verifier_result.json`` under
   ``<output-dir>/<task_id>/``.

Usage::

    python -m miniswewebagent.run.benchmarks.rst \
      -c generation/terminal_rst.yaml \
      -c generation/model_azure_gpt54.yaml \
      --tasks-file .../rst_short5000_grouped.jsonl \
      --group 1 --limit 2 --workers 2 \
      --output-dir outputs/terminal/smoke
"""

from __future__ import annotations

import concurrent.futures
import json
import time
import traceback
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miniswewebagent.agents import get_agent
from miniswewebagent.config import get_config_from_spec
from miniswewebagent.environments import get_environment
from miniswewebagent.models import get_model
from miniswewebagent.utils.serialize import recursive_merge

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
console = Console()


def _load_config(config_spec: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for spec in config_spec:
        config = recursive_merge(config, get_config_from_spec(spec))
    return config


def _load_tasks(tasks_file: Path) -> list[dict[str, Any]]:
    text = tasks_file.read_text(encoding="utf-8")
    if tasks_file.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    return loaded if isinstance(loaded, list) else loaded.get("tasks", [])


def _instruction_for(tasks_root: Path, task_id: str) -> str:
    path = tasks_root / task_id / "instruction.md"
    if not path.is_file():
        raise FileNotFoundError(f"missing instruction.md for {task_id}: {path}")
    return path.read_text(encoding="utf-8").strip()


def _run_one(config: dict[str, Any], record: dict[str, Any], output_root: Path) -> dict[str, Any]:
    task_id = str(record["task_id"])
    task_output = output_root / task_id
    task_output.mkdir(parents=True, exist_ok=True)

    env_config = dict(config.get("environment", {}))
    env_config["output_dir"] = str(task_output)
    tasks_root = Path(str(env_config.get("tasks_root", "."))).expanduser()

    agent_config = dict(config.get("agent", {}))
    agent_config["output_path"] = str(task_output / "trajectory.json")

    started = time.time()
    summary: dict[str, Any] = {"task_id": task_id, "band": record.get("band")}
    env = None
    try:
        instruction = _instruction_for(tasks_root, task_id)
        model = get_model(config.get("model", {}))
        env = get_environment(env_config, default_type="terminal_docker")
        agent = get_agent(model, env, agent_config, default_type="default")

        env.prepare(task=instruction, task_id=task_id, task_record=record)
        result = agent.run(instruction, task_id=task_id)
        summary["exit_status"] = result.get("exit_status", "")
        summary["n_steps"] = getattr(agent, "n_calls", None)
        agent.save(Path(agent_config["output_path"]))

        # External score: the task's own private verifier, run only now.
        verdict = env.run_private_verifier()
        summary["score"] = verdict.get("score")
        summary["partial_credit"] = verdict.get("partial_credit")
    except NotImplementedError as exc:
        # Multi-container tasks: the harness cannot bring up the compose network, so
        # the episode would fail for environment reasons and poison the pass rate.
        # Skipped rather than errored so a batch roll-up stays readable.
        summary["skipped"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        summary["error"] = f"{type(exc).__name__}: {exc}"
        (task_output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    summary["seconds"] = round(time.time() - started, 1)
    (task_output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@app.command()
def main(
    config_spec: list[str] = typer.Option([], "-c", "--config", help="Config specs, merged in order."),
    tasks_file: Path = typer.Option(None, "--tasks-file", help="JSON or JSONL task list."),
    output_dir: Path = typer.Option(None, "--output-dir", help="Root output directory."),
    group: int = typer.Option(0, "--group", help="Only tasks with this group id (0 = all)."),
    limit: int = typer.Option(0, "--limit", help="Run only the first N selected tasks."),
    workers: int = typer.Option(1, "--workers", help="Parallel worker processes."),
) -> None:
    config = _load_config(config_spec)
    run_config = config.get("run", {})

    resolved_tasks_file = tasks_file or Path(str(run_config.get("tasks_file", "")))
    tasks = _load_tasks(resolved_tasks_file.expanduser())
    if group:
        tasks = [t for t in tasks if int(t.get("group") or 0) == group]
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        raise typer.BadParameter("no tasks selected")

    output_root = (output_dir or Path(str(config.get("environment", {}).get("output_dir", "outputs/terminal")))).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]RST[/bold] {len(tasks)} tasks, {workers} workers -> {output_root}")
    started = time.time()
    summaries: list[dict[str, Any]] = []
    if workers <= 1:
        for record in tasks:
            summaries.append(_run_one(config, record, output_root))
            console.print(summaries[-1])
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, config, r, output_root): r for r in tasks}
            for future in concurrent.futures.as_completed(futures):
                summaries.append(future.result())
                console.print(summaries[-1])

    scored = [s for s in summaries if isinstance(s.get("score"), int)]
    payload = {
        "tasks": len(summaries),
        "scored": len(scored),
        "passed": sum(1 for s in scored if s["score"] == 1),
        "skipped": sum(1 for s in summaries if s.get("skipped")),
        "errors": sum(1 for s in summaries if s.get("error")),
        "seconds": round(time.time() - started, 1),
        "summaries": summaries,
    }
    (output_root / "batch_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(
        f"[bold green]done[/bold green] passed {payload['passed']}/{payload['scored']} "
        f"(skipped {payload['skipped']}, errors {payload['errors']}) in {payload['seconds']}s"
    )


if __name__ == "__main__":
    app()

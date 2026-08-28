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
import multiprocessing
import subprocess
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

# Builds and agent episodes bottleneck on different resources: a build saturates
# host CPU, disk and the registry link, while an episode is almost entirely idle
# waiting on the gateway. Sizing one pool for both caps agent concurrency at
# whatever the daemon can survive building. Instead the process pool sizes the
# agent side, and this semaphore independently caps concurrent builds, so the two
# overlap continuously rather than running in phases.
_BUILD_SEMAPHORE = None


def _init_worker(semaphore) -> None:
    global _BUILD_SEMAPHORE
    _BUILD_SEMAPHORE = semaphore
    _watch_parent()


def _watch_parent(interval: float = 5.0) -> None:
    """Exit the worker when the parent runner dies.

    ProcessPoolExecutor workers are spawned with a `multiprocessing.spawn` command
    line, so `pkill -f 'benchmarks[.]rst'` reaches only the parent. A worker that is
    mid-episode keeps running its container and gateway calls, then pulls further
    items the parent had already queued. Observed: three orphans running ~30 min
    after their parent was killed. Polling the parent pid closes that gap portably
    (PR_SET_PDEATHSIG is Linux-only).
    """
    import os
    import threading
    import time

    parent = os.getppid()

    def _poll() -> None:
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=_poll, name="parent-watchdog", daemon=True).start()


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

        if _BUILD_SEMAPHORE is not None:
            with _BUILD_SEMAPHORE:
                env.prepare(task=instruction, task_id=task_id, task_record=record)
        else:
            env.prepare(task=instruction, task_id=task_id, task_record=record)
        env_info = env.serialize().get("environment", {})
        summary["build_seconds"] = env_info.get("build_seconds")
        summary["image_was_cached"] = env_info.get("image_was_cached")
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
    workers: int = typer.Option(1, "--workers", help="Concurrent agent episodes (process pool size)."),
    build_workers: int = typer.Option(
        0, "--build-workers",
        help="Max concurrent docker builds. 0 = min(workers, 8). Keep well below "
             "--workers: builds are host-bound, episodes are gateway-bound.",
    ),
    prune_every: int = typer.Option(
        0, "--prune-every",
        help="Run `docker builder prune` after this many finished tasks (0 = never). "
             "Build cache grows ~0.8 GB per build and nothing else reclaims it.",
    ),
    prune_keep_storage: str = typer.Option(
        "20GB", "--prune-keep-storage", help="Build cache to retain when pruning."
    ),
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

    resolved_build_workers = build_workers or min(workers, 8)
    console.print(
        f"[bold]RST[/bold] {len(tasks)} tasks, {workers} agent workers, "
        f"{resolved_build_workers} concurrent builds -> {output_root}"
    )

    def maybe_prune(done: int) -> None:
        if prune_every and done and done % prune_every == 0:
            subprocess.run(
                ["docker", "builder", "prune", "-f", "--keep-storage", prune_keep_storage],
                capture_output=True, text=True,
            )
            console.print(f"[dim]pruned build cache after {done} tasks[/dim]")

    started = time.time()
    summaries: list[dict[str, Any]] = []
    if workers <= 1:
        for record in tasks:
            summaries.append(_run_one(config, record, output_root))
            console.print(summaries[-1])
            maybe_prune(len(summaries))
    else:
        semaphore = multiprocessing.Semaphore(resolved_build_workers)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(semaphore,)
        ) as executor:
            futures = {executor.submit(_run_one, config, r, output_root): r for r in tasks}
            for future in concurrent.futures.as_completed(futures):
                summaries.append(future.result())
                console.print(summaries[-1])
                maybe_prune(len(summaries))

    scored = [s for s in summaries if isinstance(s.get("score"), int)]
    payload = {
        "tasks": len(summaries),
        "scored": len(scored),
        "passed": sum(1 for s in scored if s["score"] == 1),
        "skipped": sum(1 for s in summaries if s.get("skipped")),
        "errors": sum(1 for s in summaries if s.get("error")),
        "seconds": round(time.time() - started, 1),
        "agent_workers": workers,
        "build_workers": resolved_build_workers,
        "summaries": summaries,
    }
    (output_root / "batch_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(
        f"[bold green]done[/bold green] passed {payload['passed']}/{payload['scored']} "
        f"(skipped {payload['skipped']}, errors {payload['errors']}) in {payload['seconds']}s"
    )


if __name__ == "__main__":
    app()

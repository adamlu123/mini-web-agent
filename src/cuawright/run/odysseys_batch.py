from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from cuawright.config.store import REPOSITORY_ROOT, credential_status
from cuawright.environments.odysseys.source import (
    DESKTOP_RUNTIME_COMMIT,
    OdysseysSource,
    inspect_desktop_runtime,
    inspect_source,
)
from cuawright.environments.odysseys.task import task_ids
from cuawright.operations.scheduler import (
    ProcessJob,
    ProcessMonitorState,
    atomic_write_json,
    latest_run_dir,
    objective_health_failure,
    optional_sha256,
    read_optional_json,
    run_process_jobs,
    validate_source_tree,
)


@dataclass(frozen=True, slots=True)
class OdysseysBatchConfig:
    result_root: Path
    task_ids: tuple[str, ...]
    width: int
    source: OdysseysSource
    runtime_source: Path
    vm_path: Path
    api_key_file: Path
    health_delay_seconds: int = 300


def selected_task_ids(
    dataset_path: str | Path,
    explicit: list[str],
    first: int,
) -> tuple[str, ...]:
    available = task_ids(dataset_path)
    if explicit:
        selected = tuple(value.lower() for value in explicit)
        if len(selected) != len(set(selected)):
            raise ValueError("Explicit Odysseys task ids must be unique")
        missing = [value for value in selected if value not in available]
        if missing:
            raise ValueError("Unknown Odysseys task ids: " + ", ".join(missing))
        return selected
    if isinstance(first, bool) or not isinstance(first, int) or first < 1:
        raise ValueError("--first must be a positive integer")
    return available[:first]


def prepare_batch(config: OdysseysBatchConfig, commit: str) -> None:
    root = config.result_root
    for name in ("logs", "pids", "exit", "health", "tasks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "task-ids.txt").write_text(
        "\n".join(config.task_ids) + "\n",
        encoding="utf-8",
    )
    atomic_write_json(
        root / "batch-manifest.json",
        {
            "schema_version": 1,
            "benchmark": "odysseys",
            "task_count": len(config.task_ids),
            "task_ids": list(config.task_ids),
            "selection": "explicit-or-dataset-prefix",
            "width": config.width,
            "cuawright_commit": commit,
            "source": config.source.provenance(),
            "desktop_runtime": {
                "root": str(config.runtime_source),
                "commit": DESKTOP_RUNTIME_COMMIT,
            },
            "vm": str(config.vm_path),
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
            "infrastructure": "aiservices",
            "model_interface": "terminal-only",
            "agent_topology": "orchestrated",
            "judge_execution": "external-not-started",
            "health_delay_seconds": config.health_delay_seconds,
            "health_contract": "current-registry-checker",
            "started_at_epoch": time.time(),
        },
    )


def run_batch(config: OdysseysBatchConfig) -> int:
    commit = validate_source_tree()
    prepare_batch(config, commit)
    source_path = str(REPOSITORY_ROOT / "src")
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            part for part in (source_path, os.environ.get("PYTHONPATH", "")) if part
        ),
    }
    jobs = tuple(
        ProcessJob(
            job_id=task_id,
            command=(
                sys.executable,
                "-m",
                "cuawright.run.odysseys",
                "--task-id",
                task_id,
                "--source",
                str(config.source.root),
                "--source-commit",
                config.source.commit,
                "--task-source-json",
                str(config.source.dataset_path),
                "--dataset-sha256",
                config.source.dataset_sha256,
                "--examples-dir",
                str(config.source.examples_dir),
                "--judge-script",
                str(config.source.judge_script),
                "--judge-sha256",
                config.source.judge_sha256,
                "--runtime-source",
                str(config.runtime_source),
                "--runtime-commit",
                DESKTOP_RUNTIME_COMMIT,
                "--path-to-vm",
                str(config.vm_path),
                "--api-key-file",
                str(config.api_key_file),
                "--result-root",
                str(config.result_root / "tasks" / task_id),
                "--model-provider",
                "openai",
                "--model",
                "gpt-5.5",
                "--reasoning-effort",
                "xhigh",
                "--model-infrastructure",
                "aiservices",
            ),
            cwd=REPOSITORY_ROOT,
            environment=environment,
            log_path=config.result_root / "logs" / f"process-{task_id}.log",
            pid_path=config.result_root / "pids" / f"{task_id}.pid",
            exit_path=config.result_root / "exit" / f"{task_id}.txt",
            monitor=(
                lambda process, selected_task=task_id: _monitor_task_health(
                    config,
                    selected_task,
                    process,
                )
            ),
        )
        for task_id in config.task_ids
    )
    outcomes = run_process_jobs(jobs, width=config.width)
    outcomes_by_id = {outcome.job_id: outcome for outcome in outcomes}
    rows = []
    for task_id in config.task_ids:
        task_root = config.result_root / "tasks" / task_id
        run_dir = latest_run_dir(task_root, task_id)
        result_path = (
            None if run_dir is None else run_dir / "evaluation" / "result.json"
        )
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path is not None and result_path.is_file()
            else None
        )
        evaluation = result.get("evaluation") if isinstance(result, dict) else None
        judge_status = (
            evaluation.get("status") if isinstance(evaluation, dict) else None
        )
        outcome = outcomes_by_id.get(task_id)
        return_code = None if outcome is None else outcome.return_code
        scheduler_status = None if outcome is None else outcome.status
        health = read_optional_json(config.result_root / "health" / f"{task_id}.json")
        if outcome is None:
            state = "missing"
        elif scheduler_status != "completed":
            state = (
                "interrupted"
                if scheduler_status in {"cancelled", "interrupted", "not_submitted"}
                else "failed"
            )
        elif return_code != 0:
            state = "failed"
        elif judge_status == "ready_for_external_judge":
            state = "ready_for_external_judge"
        else:
            state = "incomplete"
        rows.append(
            {
                "task_id": task_id,
                "exit_code": return_code,
                "scheduler_status": scheduler_status,
                "scheduler_error": None if outcome is None else outcome.error,
                "pid": None if outcome is None else outcome.pid,
                "pgid": None if outcome is None else outcome.pgid,
                "health": health,
                "health_failure": objective_health_failure(health),
                "state": state,
                "selected_run": None if run_dir is None else str(run_dir),
                "manifest_sha256": (
                    None
                    if run_dir is None
                    else optional_sha256(run_dir / "manifest.json")
                ),
                "result_sha256": (
                    None if result_path is None else optional_sha256(result_path)
                ),
                "judge_status": judge_status,
            }
        )
        atomic_write_json(
            config.result_root / "exit" / f"{task_id}.json",
            rows[-1],
        )
    summary = {
        "schema_version": 1,
        "benchmark": "odysseys",
        "tasks": rows,
        "ready_for_external_judge": sum(
            row["state"] == "ready_for_external_judge" for row in rows
        ),
        "incomplete": sum(row["state"] == "incomplete" for row in rows),
        "failed": sum(row["state"] == "failed" for row in rows),
        "interrupted": sum(row["state"] == "interrupted" for row in rows),
        "missing": sum(row["state"] == "missing" for row in rows),
        "health_failures": sum(row["health_failure"] for row in rows),
        "completed_at_epoch": time.time(),
    }
    atomic_write_json(config.result_root / "batch-summary.json", summary)
    return (
        1
        if any(
            summary[name]
            for name in (
                "incomplete",
                "failed",
                "interrupted",
                "missing",
                "health_failures",
            )
        )
        else 0
    )


def _monitor_task_health(
    config: OdysseysBatchConfig,
    task_id: str,
    process: ProcessMonitorState,
) -> None:
    process.completed.wait(timeout=config.health_delay_seconds)
    task_root = config.result_root / "tasks" / task_id
    run_dir = latest_run_dir(task_root, task_id)
    health_path = config.result_root / "health" / f"{task_id}.json"
    if run_dir is None:
        atomic_write_json(
            health_path,
            {
                "status": "unhealthy",
                "objective_stop_required": True,
                "fatal": ["run directory missing at automatic health check"],
                "task_id": task_id,
            },
        )
        return
    process_log = config.result_root / "logs" / f"process-{task_id}.log"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "python" / "check_terminal_run_health.py"),
        "--benchmark",
        "odysseys",
        "--run-dir",
        str(run_dir),
        "--pid",
        str(process.pid),
        "--process-log",
        str(process_log),
        "--expected-model",
        "gpt-5.5",
        "--expected-reasoning",
        "xhigh",
        "--expected-infrastructure",
        "aiservices",
        "--expected-agent-topology",
        "orchestrated",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (config.result_root / "health" / f"{task_id}.log").write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    payload = read_optional_json(run_dir / "health-check-5m.json")
    if completed.returncode not in (0, 1) or payload is None:
        payload = {
            "status": "checker_failed",
            "objective_stop_required": True,
            "fatal": ["health checker did not produce a valid factual payload"],
            "checker_returncode": completed.returncode,
            "task_id": task_id,
        }
    atomic_write_json(health_path, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an Odysseys task selection through the public scheduler"
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--task-source-json", required=True)
    parser.add_argument("--examples-dir", required=True)
    parser.add_argument("--judge-script", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--path-to-vm", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--first", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--health-delay-seconds", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = inspect_source(
        root=args.source,
        expected_commit=args.source_commit,
        dataset_path=args.task_source_json,
        examples_dir=args.examples_dir,
        judge_script=args.judge_script,
    )
    runtime = inspect_desktop_runtime(
        args.runtime_source,
        DESKTOP_RUNTIME_COMMIT,
    )
    vm_path = Path(args.path_to_vm).expanduser().resolve()
    if not vm_path.is_file():
        raise FileNotFoundError(f"Odysseys VM image not found: {vm_path}")
    api_key_file = Path(args.api_key_file).expanduser().resolve()
    if not credential_status(api_key_file)["ok"]:
        raise ValueError("Odysseys API key file must be a private regular file")
    selected = selected_task_ids(source.dataset_path, args.task_id, args.first)
    if args.health_delay_seconds < 1:
        raise ValueError("--health-delay-seconds must be positive")
    config = OdysseysBatchConfig(
        result_root=Path(args.result_root).expanduser().resolve(),
        task_ids=selected,
        width=min(args.width, len(selected)),
        source=source,
        runtime_source=runtime.root,
        vm_path=vm_path,
        api_key_file=api_key_file,
        health_delay_seconds=args.health_delay_seconds,
    )
    raise SystemExit(run_batch(config))


if __name__ == "__main__":
    main()

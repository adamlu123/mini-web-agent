#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuawright.config.store import (
    DEFAULT_API_KEY_FILE,
    DEFAULT_OSWORLD_ASSETS,
    DEFAULT_PROXY_CONFIG,
    REPOSITORY_ROOT,
    configured_value,
)
from cuawright.models.providers import PROVIDER_DEFAULT_BASE_URLS
from cuawright.operations.scheduler import (
    atomic_write_json,
    control_client_manifest,
    latest_run_dir,
    objective_health_failure,
    optional_sha256,
    read_optional_json,
    sha256_file,
    signal_process_group,
    terminate_process,
    validate_source_tree,
)


TASK_EXCLUSIONS = {
    "026": "GitLab",
    "041": "GitLab",
    "082": "unsupported nested-Docker guest setup",
}
TASK_IDS = tuple(
    f"{value:03d}" for value in range(1, 109) if f"{value:03d}" not in TASK_EXCLUSIONS
)
REPO_ROOT = REPOSITORY_ROOT
DEFAULT_ASSETS = DEFAULT_OSWORLD_ASSETS
DEFAULT_KEY = DEFAULT_API_KEY_FILE
DEFAULT_PROXY = DEFAULT_PROXY_CONFIG


@dataclass(frozen=True, slots=True)
class BatchConfig:
    result_root: Path
    task_ids: tuple[str, ...]
    width: int
    vm_path: Path
    assets_dir: Path
    api_key_file: Path
    proxy_config: Path
    website_host_suffix: str
    model: str
    reasoning_effort: str
    model_infrastructure: str
    health_delay_seconds: int
    no_subagents: bool = False
    model_provider: str = "openai"
    base_url: str = PROVIDER_DEFAULT_BASE_URLS["openai"]
    docker_ram_size_gb: int | None = None
    docker_cpu_cores: int | None = None
    user_simulator_api_key_file: Path | None = None
    user_simulator_base_url: str = PROVIDER_DEFAULT_BASE_URLS["openai"]
    anthropic_max_output_tokens: int | None = None
    prompt_profile: str = "canonical"
    disable_task_proxy: bool = False
    openai_policy_feedback_max_retries: int = 0


class BatchCoordinator:
    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._process_lock = threading.Lock()
        self._original_handlers: dict[int, Any] = {}

    def run(self) -> int:
        commit = validate_source_tree()
        self._prepare_root(commit)
        self._install_signal_handlers()
        worker_failed = False
        summary: dict[str, Any] | None = None
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.width,
                thread_name_prefix="osworld-task",
            ) as executor:
                futures = {
                    executor.submit(self._run_task, task_id): task_id
                    for task_id in self.config.task_ids
                }
                pending = set(futures)
                cleanup_started = False
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        task_id = futures[future]
                        try:
                            future.result()
                        except Exception as exc:
                            worker_failed = True
                            atomic_write_json(
                                self.config.result_root
                                / "exit"
                                / f"{task_id}.exception.json",
                                {
                                    "task_id": task_id,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                            )
                    if self.stop_event.is_set() and not cleanup_started:
                        cleanup_started = True
                        self._stop_all_processes()
            summary = self._write_summary(commit)
        finally:
            self._stop_all_processes()
            self._restore_signal_handlers()
        if self.stop_event.is_set():
            return 130
        assert summary is not None
        failed = (
            worker_failed
            or summary["nonzero_exits"] > 0
            or summary["selection_errors"] > 0
            or summary["health_failures"] > 0
        )
        return 1 if failed else 0

    def _prepare_root(self, commit: str) -> None:
        root = self.config.result_root
        for child in ("logs", "pids", "exit", "health", "tasks"):
            (root / child).mkdir(parents=True, exist_ok=True)
        (root / "task-ids.txt").write_text("\n".join(self.config.task_ids) + "\n")
        atomic_write_json(
            root / "batch-manifest.json",
            {
                "schema_version": 2,
                "benchmark": "osworld-v2",
                "task_count": len(self.config.task_ids),
                "task_ids": list(self.config.task_ids),
                "selection": (
                    "full-excluding-gitlab-and-nested-docker"
                    if self.config.task_ids == TASK_IDS
                    else "explicit"
                ),
                "excluded": (
                    dict(TASK_EXCLUSIONS) if self.config.task_ids == TASK_IDS else {}
                ),
                "width": self.config.width,
                "commit": commit,
                "provider": self.config.model_provider,
                "base_url": self.config.base_url,
                "model": self.config.model,
                "reasoning_effort": self.config.reasoning_effort,
                "infrastructure": self.config.model_infrastructure,
                "docker_ram_size_gb": self.config.docker_ram_size_gb,
                "docker_cpu_cores": self.config.docker_cpu_cores,
                "user_simulator_base_url": (
                    self.config.user_simulator_base_url
                    if self.config.user_simulator_api_key_file
                    else None
                ),
                "anthropic_max_output_tokens": (
                    self.config.anthropic_max_output_tokens
                ),
                "model_interface": "terminal-only",
                "agent_topology": (
                    "linear" if self.config.no_subagents else "orchestrated"
                ),
                "prompt_profile": self.config.prompt_profile,
                "disable_task_proxy": self.config.disable_task_proxy,
                "openai_policy_feedback_max_retries": (
                    self.config.openai_policy_feedback_max_retries
                ),
                "terminal_control": control_client_manifest(),
                "website_host_suffix": self.config.website_host_suffix,
                "proxy_config_sha256": sha256_file(self.config.proxy_config),
                "retry_policy": (
                    "one runner-level retry for InfrastructureError only; "
                    "no quality retries"
                ),
                "started_at_epoch": time.time(),
            },
        )

    def _run_task(self, task_id: str) -> None:
        if self.stop_event.is_set():
            return
        root = self.config.result_root
        task_root = root / "tasks" / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        log_path = root / "logs" / f"process-{task_id}.log"
        environment = os.environ.copy()
        environment["PROXY_CONFIG_FILE"] = str(self.config.proxy_config)
        source_path = str(REPO_ROOT / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_path, environment.get("PYTHONPATH", "")) if part
        )
        command = [
            sys.executable,
            "-m",
            "cuawright.run.osworld",
            "--task-id",
            task_id,
            "--path-to-vm",
            str(self.config.vm_path),
            "--assets-dir",
            str(self.config.assets_dir),
            "--api-key-file",
            str(self.config.api_key_file),
            "--result-root",
            str(task_root),
            "--model-provider",
            self.config.model_provider,
            "--base-url",
            self.config.base_url,
            "--model",
            self.config.model,
            "--reasoning-effort",
            self.config.reasoning_effort,
            "--model-infrastructure",
            self.config.model_infrastructure,
            "--website-host-suffix",
            self.config.website_host_suffix,
            "--prompt-profile",
            self.config.prompt_profile,
            "--openai-policy-feedback-retries",
            str(self.config.openai_policy_feedback_max_retries),
        ]
        if self.config.docker_ram_size_gb is not None:
            command.extend(
                ["--docker-ram-size-gb", str(self.config.docker_ram_size_gb)]
            )
        if self.config.docker_cpu_cores is not None:
            command.extend(["--docker-cpu-cores", str(self.config.docker_cpu_cores)])
        if self.config.user_simulator_api_key_file is not None:
            command.extend(
                [
                    "--user-simulator-api-key-file",
                    str(self.config.user_simulator_api_key_file),
                    "--user-simulator-base-url",
                    self.config.user_simulator_base_url,
                ]
            )
        if self.config.anthropic_max_output_tokens is not None:
            command.extend(
                [
                    "--anthropic-max-output-tokens",
                    str(self.config.anthropic_max_output_tokens),
                ]
            )
        if self.config.no_subagents:
            command.append("--no-subagents")
        if self.config.disable_task_proxy:
            command.append("--disable-task-proxy")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            with self._process_lock:
                self._processes[task_id] = process
                stop_after_start = self.stop_event.is_set()
            (root / "pids" / f"{task_id}.pid").write_text(f"{process.pid}\n")
            if stop_after_start:
                self._terminate_process(process)
            health = threading.Thread(
                target=self._health_check,
                args=(task_id, process, task_root, log_path),
                name=f"health-{task_id}",
                daemon=True,
            )
            health.start()
            return_code = process.wait()
            health.join()
        with self._process_lock:
            self._processes.pop(task_id, None)
        (root / "exit" / f"{task_id}.txt").write_text(f"{return_code}\n")
        self._write_task_record(task_id, return_code, task_root)

    def _health_check(
        self,
        task_id: str,
        process: subprocess.Popen[str],
        task_root: Path,
        log_path: Path,
    ) -> None:
        deadline = time.monotonic() + self.config.health_delay_seconds
        while process.poll() is None and not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self.stop_event.wait(min(1.0, remaining))
        if self.stop_event.is_set():
            return
        run_dir = latest_run_dir(task_root, task_id)
        if run_dir is None:
            atomic_write_json(
                self.config.result_root / "health" / f"{task_id}.json",
                {
                    "status": "unhealthy",
                    "objective_stop_required": True,
                    "fatal": ["run directory missing after health delay"],
                },
            )
            return
        health_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "python" / "check_terminal_run_health.py"),
            "--benchmark",
            "osworld",
            "--run-dir",
            str(run_dir),
            "--pid",
            str(process.pid),
            "--process-log",
            str(log_path),
            "--expected-model",
            self.config.model,
            "--expected-reasoning",
            self.config.reasoning_effort,
            "--expected-infrastructure",
            self.config.model_infrastructure,
            "--expected-agent-topology",
            "linear" if self.config.no_subagents else "orchestrated",
        ]
        completed = subprocess.run(
            health_command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        (self.config.result_root / "health" / f"{task_id}.log").write_text(
            completed.stdout + completed.stderr
        )
        health_payload = read_optional_json(run_dir / "health-check-5m.json")
        if completed.returncode not in (0, 1) or health_payload is None:
            atomic_write_json(
                self.config.result_root / "health" / f"{task_id}.json",
                {
                    "status": "checker_failed",
                    "objective_stop_required": True,
                    "fatal": [
                        "health checker failed to produce a valid health payload"
                    ],
                    "checker_returncode": completed.returncode,
                },
            )

    def _write_task_record(
        self,
        task_id: str,
        return_code: int,
        task_root: Path,
    ) -> None:
        attempts = []
        for run_dir in sorted(task_root.glob(f"task-{task_id}-*")):
            result_path = run_dir / "evaluation" / "result.json"
            health_path = run_dir / "health-check-5m.json"
            attempts.append(
                {
                    "run_dir": str(run_dir),
                    "manifest_sha256": optional_sha256(run_dir / "manifest.json"),
                    "result_sha256": optional_sha256(result_path),
                    "evaluated": result_path.is_file(),
                    "health": read_optional_json(health_path),
                }
            )
        atomic_write_json(
            self.config.result_root / "exit" / f"{task_id}.json",
            {
                "task_id": task_id,
                "exit_code": return_code,
                "attempts": attempts,
            },
        )

    def _write_summary(self, commit: str) -> dict[str, Any]:
        rows = []
        for task_id in self.config.task_ids:
            record = read_optional_json(
                self.config.result_root / "exit" / f"{task_id}.json"
            ) or {"task_id": task_id, "exit_code": None, "attempts": []}
            evaluated = [
                attempt for attempt in record["attempts"] if attempt.get("evaluated")
            ]
            result = None
            if len(evaluated) == 1:
                result_path = (
                    Path(evaluated[0]["run_dir"]) / "evaluation" / "result.json"
                )
                result = json.loads(result_path.read_text())
            rows.append(
                {
                    **record,
                    "coordinator_health": read_optional_json(
                        self.config.result_root / "health" / f"{task_id}.json"
                    ),
                    "selected_run": (
                        evaluated[0]["run_dir"] if len(evaluated) == 1 else None
                    ),
                    "score": result.get("score") if result else None,
                    "selection_error": (
                        None
                        if len(evaluated) == 1
                        else f"expected one evaluated attempt, found {len(evaluated)}"
                    ),
                }
            )
        for row in rows:
            selected_health = None
            if row["selected_run"] is not None:
                selected_health = read_optional_json(
                    Path(row["selected_run"]) / "health-check-5m.json"
                )
            row["health_failure"] = objective_health_failure(
                row["coordinator_health"]
            ) or objective_health_failure(selected_health)
        summary = {
            "schema_version": 2,
            "commit": commit,
            "tasks": rows,
            "evaluated": sum(row["score"] is not None for row in rows),
            "score_sum": sum(float(row["score"] or 0) for row in rows),
            "nonzero_exits": sum(row["exit_code"] not in (0, None) for row in rows),
            "selection_errors": sum(row["selection_error"] is not None for row in rows),
            "health_failures": sum(row["health_failure"] for row in rows),
            "completed_at_epoch": time.time(),
        }
        atomic_write_json(self.config.result_root / "batch-summary.json", summary)
        return summary

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, handler in self._original_handlers.items():
            signal.signal(signum, handler)

    def _handle_signal(self, signum: int, frame: object) -> None:
        del signum, frame
        self.stop_event.set()

    def _stop_all_processes(self) -> None:
        with self._process_lock:
            processes = list(self._processes.values())
        for process in processes:
            if process.poll() is None:
                signal_process_group(process, signal.SIGINT)
        deadline = time.monotonic() + 30
        for process in processes:
            if process.poll() is not None:
                continue
            timeout = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGTERM)
        self._wait_or_kill(processes, timeout=15)

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        terminate_process(process)

    def _wait_or_kill(
        self,
        processes: list[subprocess.Popen[str]],
        *,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGKILL)
        for process in processes:
            if process.poll() is None:
                process.wait(timeout=15)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the terminal-core OSWorld V2 profile at bounded width"
    )
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--width", type=int, default=54)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Run an explicit OSWorld task ID; repeat for multiple tasks",
    )
    vm_path = configured_value("osworld", "vm")
    parser.add_argument("--path-to-vm", default=vm_path, required=not vm_path)
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS))
    parser.add_argument("--api-key-file", default=str(DEFAULT_KEY))
    parser.add_argument("--proxy-config", default=str(DEFAULT_PROXY))
    parser.add_argument(
        "--model-provider",
        choices=("openai", "openrouter-anthropic", "anthropic"),
        default="openai",
    )
    parser.add_argument("--base-url", default=PROVIDER_DEFAULT_BASE_URLS["openai"])
    parser.add_argument("--docker-ram-size-gb", type=int)
    parser.add_argument("--docker-cpu-cores", type=int)
    parser.add_argument("--user-simulator-api-key-file")
    parser.add_argument(
        "--user-simulator-base-url",
        default=PROVIDER_DEFAULT_BASE_URLS["openai"],
    )
    parser.add_argument("--anthropic-max-output-tokens", type=int)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--model-infrastructure", default="aiservices")
    parser.add_argument(
        "--website-host-suffix",
        default=os.getenv("WEBSITE_HOST_SUFFIX", "10.217.29.172.nip.io"),
    )
    parser.add_argument("--health-delay-seconds", type=int, default=300)
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="Run one linear Actor per task with no specialists",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("canonical", "e-l1-direct", "e-l1-disciplined"),
        default="canonical",
        help="OSWorld Actor prompt profile forwarded unchanged to every task",
    )
    parser.add_argument(
        "--disable-task-proxy",
        action="store_true",
        help="Explicitly disable task-requested proxies for every OSWorld task",
    )
    parser.add_argument(
        "--openai-policy-feedback-retries",
        type=int,
        default=0,
        help="Append bounded benign compliance feedback after OpenAI policy blocks",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.width < 1:
        raise ValueError("width must be positive")
    if args.health_delay_seconds < 1:
        raise ValueError("health-delay-seconds must be positive")
    if not args.model.strip():
        raise ValueError("model cannot be empty")
    if not args.base_url.strip():
        raise ValueError("base-url cannot be empty")
    if args.docker_ram_size_gb is not None and args.docker_ram_size_gb < 1:
        raise ValueError("docker-ram-size-gb must be positive")
    if args.docker_cpu_cores is not None and args.docker_cpu_cores < 1:
        raise ValueError("docker-cpu-cores must be positive")
    if args.user_simulator_api_key_file and not args.user_simulator_base_url.strip():
        raise ValueError("user-simulator-base-url cannot be empty")
    if (
        args.anthropic_max_output_tokens is not None
        and args.anthropic_max_output_tokens < 1
    ):
        raise ValueError("anthropic-max-output-tokens must be positive")
    if not args.reasoning_effort.strip():
        raise ValueError("reasoning-effort cannot be empty")
    if not args.model_infrastructure.strip():
        raise ValueError("model-infrastructure cannot be empty")
    if not args.website_host_suffix.strip():
        raise ValueError("website-host-suffix cannot be empty")
    if args.prompt_profile != "canonical" and not args.no_subagents:
        raise ValueError(f"{args.prompt_profile} prompt requires --no-subagents")
    if args.openai_policy_feedback_retries < 0:
        raise ValueError("openai-policy-feedback-retries cannot be negative")
    if args.openai_policy_feedback_retries and args.model_provider != "openai":
        raise ValueError(
            "openai-policy-feedback-retries requires --model-provider openai"
        )
    task_ids = _selected_task_ids(args.task_id)
    if args.width > len(task_ids):
        args.width = len(task_ids)
    config = BatchConfig(
        result_root=Path(args.result_root).expanduser().resolve(),
        task_ids=task_ids,
        width=args.width,
        vm_path=Path(args.path_to_vm).expanduser().resolve(),
        assets_dir=Path(args.assets_dir).expanduser().resolve(),
        api_key_file=Path(args.api_key_file).expanduser().resolve(),
        proxy_config=Path(args.proxy_config).expanduser().resolve(),
        website_host_suffix=args.website_host_suffix,
        model_provider=args.model_provider,
        base_url=args.base_url,
        docker_ram_size_gb=args.docker_ram_size_gb,
        docker_cpu_cores=args.docker_cpu_cores,
        user_simulator_api_key_file=(
            Path(args.user_simulator_api_key_file).expanduser().resolve()
            if args.user_simulator_api_key_file
            else None
        ),
        user_simulator_base_url=args.user_simulator_base_url,
        anthropic_max_output_tokens=args.anthropic_max_output_tokens,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        model_infrastructure=args.model_infrastructure,
        health_delay_seconds=args.health_delay_seconds,
        no_subagents=args.no_subagents,
        prompt_profile=args.prompt_profile,
        disable_task_proxy=args.disable_task_proxy,
        openai_policy_feedback_max_retries=(args.openai_policy_feedback_retries),
    )
    raise SystemExit(BatchCoordinator(config).run())


def _selected_task_ids(values: list[str]) -> tuple[str, ...]:
    if not values:
        return TASK_IDS
    selected = tuple(value.strip().zfill(3) for value in values)
    if any(value not in TASK_IDS for value in selected):
        invalid = sorted(set(selected) - set(TASK_IDS))
        raise ValueError("Unsupported OSWorld task IDs: " + ", ".join(invalid))
    if len(selected) != len(set(selected)):
        raise ValueError("OSWorld task IDs must be unique")
    return selected


if __name__ == "__main__":
    main()

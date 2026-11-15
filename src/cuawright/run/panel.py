#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from cuawright.operations.scheduler import (
    atomic_write_json,
    control_client_manifest,
    latest_run_dir,
    objective_health_failure,
    read_optional_json,
    sha256_file,
    signal_process_group,
    terminate_process,
    validate_source_tree,
)

REPO_ROOT = REPOSITORY_ROOT

OSWORLD_TASK_IDS = (
    "002",
    "008",
    "010",
    "015",
    "017",
    "020",
    "021",
    "028",
    "029",
    "033",
    "035",
    "036",
    "037",
    "049",
    "050",
    "053",
    "055",
    "056",
    "057",
    "075",
    "080",
)
BENCHCAD_TASK_IDS = (
    "hex_nut_000006_s20260505",
    "spacer_ring_000007_s20260505",
    "washer_000005_s20260505",
)
CADGENBENCH_TASK_IDS = ("124", "211", "214")
PANEL_TASK_COUNT = (
    len(OSWORLD_TASK_IDS) + len(BENCHCAD_TASK_IDS) + len(CADGENBENCH_TASK_IDS)
)


@dataclass(frozen=True, slots=True)
class PanelConfig:
    result_root: Path
    vm_path: Path
    assets_dir: Path
    api_key_file: Path
    proxy_config: Path
    website_host_suffix: str
    benchcad_manifest: Path
    benchcad_repo: Path
    benchcad_image: str
    cadgenbench_dataset: Path
    cadgenbench_repo: Path
    cadgenbench_image: str
    health_delay_seconds: int
    checkpoint_delay_seconds: int


@dataclass(frozen=True, slots=True)
class PanelProcess:
    label: str
    benchmark: str
    task_id: str | None
    task_root: Path
    command: tuple[str, ...]
    manages_own_health: bool = False


class RegressionPanel:
    def __init__(self, config: PanelConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._logs: dict[str, Any] = {}
        self._health_threads: list[threading.Thread] = []
        self._process_lock = threading.Lock()
        self._original_handlers: dict[int, Any] = {}

    def run(self) -> int:
        commit = validate_source_tree()
        self._validate_inputs()
        self._validate_result_root()
        specs = self._process_specs()
        self._prepare_root(commit, specs)
        self._install_signal_handlers()
        checkpoint = threading.Thread(
            target=self._write_delayed_checkpoint,
            args=(commit, specs),
            name="panel-checkpoint-30m",
            daemon=True,
        )
        interrupted = False
        summary: dict[str, Any] | None = None
        try:
            for spec in specs:
                self._start_process(spec)
            checkpoint.start()
            self._wait_for_processes(specs)
            self._join_health_threads()
            summary = self._write_summary(commit, specs)
        finally:
            interrupted = self.stop_event.is_set()
            self.stop_event.set()
            self._stop_all_processes()
            if checkpoint.is_alive():
                checkpoint.join(timeout=1)
            self._restore_signal_handlers()
        if interrupted:
            return 130
        assert summary is not None
        return 1 if _panel_failed(summary) else 0

    def _validate_inputs(self) -> None:
        required_files = (
            self.config.vm_path,
            self.config.api_key_file,
            self.config.proxy_config,
            self.config.benchcad_manifest,
        )
        missing_files = [str(path) for path in required_files if not path.is_file()]
        required_dirs = (
            self.config.assets_dir,
            self.config.benchcad_repo,
            self.config.cadgenbench_dataset,
            self.config.cadgenbench_repo,
        )
        missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
        if missing_files or missing_dirs:
            raise FileNotFoundError(
                "Regression panel inputs are missing: "
                + ", ".join(missing_files + missing_dirs)
            )

    def _validate_result_root(self) -> None:
        try:
            self.config.result_root.relative_to(REPO_ROOT)
        except ValueError:
            return
        raise ValueError("Regression panel result root must be outside the repository")

    def _process_specs(self) -> tuple[PanelProcess, ...]:
        python = sys.executable
        osworld_root = self.config.result_root / "osworld"
        osworld_command = [
            python,
            "-m",
            "cuawright.run.batch",
            "--result-root",
            str(osworld_root),
            "--width",
            str(len(OSWORLD_TASK_IDS)),
            "--path-to-vm",
            str(self.config.vm_path),
            "--assets-dir",
            str(self.config.assets_dir),
            "--api-key-file",
            str(self.config.api_key_file),
            "--proxy-config",
            str(self.config.proxy_config),
            "--website-host-suffix",
            self.config.website_host_suffix,
            "--health-delay-seconds",
            str(self.config.health_delay_seconds),
        ]
        for task_id in OSWORLD_TASK_IDS:
            osworld_command.extend(("--task-id", task_id))
        specs = [
            PanelProcess(
                label="osworld-batch",
                benchmark="osworld",
                task_id=None,
                task_root=osworld_root,
                command=tuple(osworld_command),
                manages_own_health=True,
            )
        ]
        for task_id in BENCHCAD_TASK_IDS:
            task_root = self.config.result_root / "benchcad" / task_id
            specs.append(
                PanelProcess(
                    label=f"benchcad-{task_id}",
                    benchmark="benchcad",
                    task_id=task_id,
                    task_root=task_root,
                    command=(
                        python,
                        "-m",
                        "cuawright.run.benchcad",
                        "--manifest",
                        str(self.config.benchcad_manifest),
                        "--record-id",
                        task_id,
                        "--result-root",
                        str(task_root),
                        "--api-key-file",
                        str(self.config.api_key_file),
                        "--benchcad-repo",
                        str(self.config.benchcad_repo),
                        "--image",
                        self.config.benchcad_image,
                        "--model-provider",
                        "openai",
                        "--model",
                        "gpt-5.5",
                        "--reasoning-effort",
                        "xhigh",
                    ),
                )
            )
        for task_id in CADGENBENCH_TASK_IDS:
            task_root = self.config.result_root / "cadgenbench" / task_id
            specs.append(
                PanelProcess(
                    label=f"cadgenbench-{task_id}",
                    benchmark="cadgenbench",
                    task_id=task_id,
                    task_root=task_root,
                    command=(
                        python,
                        "-m",
                        "cuawright.run.cadgenbench",
                        "--dataset-dir",
                        str(self.config.cadgenbench_dataset),
                        "--fixture-id",
                        task_id,
                        "--result-root",
                        str(task_root),
                        "--api-key-file",
                        str(self.config.api_key_file),
                        "--cadgenbench-repo",
                        str(self.config.cadgenbench_repo),
                        "--image",
                        self.config.cadgenbench_image,
                        "--model-provider",
                        "openai",
                        "--model",
                        "gpt-5.5",
                        "--reasoning-effort",
                        "xhigh",
                    ),
                )
            )
        return tuple(specs)

    def _prepare_root(
        self,
        commit: str,
        specs: tuple[PanelProcess, ...],
    ) -> None:
        for child in ("logs", "pids", "health", "exit"):
            (self.config.result_root / child).mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.config.result_root / "panel-manifest.json",
            {
                "schema_version": 1,
                "commit": commit,
                "task_count": PANEL_TASK_COUNT,
                "launch_process_count": len(specs),
                "effective_width": PANEL_TASK_COUNT,
                "model": "gpt-5.5",
                "reasoning_effort": "xhigh",
                "provider": "openai",
                "infrastructure": "aiservices",
                "model_interface": "terminal-only",
                "terminal_control": control_client_manifest(),
                "health_delay_seconds": self.config.health_delay_seconds,
                "checkpoint_delay_seconds": self.config.checkpoint_delay_seconds,
                "tasks": {
                    "osworld": list(OSWORLD_TASK_IDS),
                    "benchcad": list(BENCHCAD_TASK_IDS),
                    "cadgenbench": list(CADGENBENCH_TASK_IDS),
                },
                "inputs": {
                    "benchcad_manifest": str(self.config.benchcad_manifest),
                    "benchcad_manifest_sha256": sha256_file(
                        self.config.benchcad_manifest
                    ),
                    "cadgenbench_dataset": str(self.config.cadgenbench_dataset),
                    "proxy_config_sha256": sha256_file(self.config.proxy_config),
                    "website_host_suffix": self.config.website_host_suffix,
                },
                "started_at_epoch": time.time(),
            },
        )

    def _start_process(self, spec: PanelProcess) -> None:
        if self.stop_event.is_set():
            return
        spec.task_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PROXY_CONFIG_FILE"] = str(self.config.proxy_config)
        source_path = str(REPO_ROOT / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_path, environment.get("PYTHONPATH", "")) if part
        )
        log_path = self.config.result_root / "logs" / f"{spec.label}.log"
        log = log_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                spec.command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception:
            log.close()
            raise
        with self._process_lock:
            self._processes[spec.label] = process
            self._logs[spec.label] = log
            stop_after_start = self.stop_event.is_set()
        (self.config.result_root / "pids" / f"{spec.label}.pid").write_text(
            f"{process.pid}\n",
            encoding="utf-8",
        )
        if stop_after_start:
            self._terminate_process(process)
        elif not spec.manages_own_health:
            health = threading.Thread(
                target=self._health_check,
                args=(spec, process, log_path),
                name=f"health-{spec.label}",
                daemon=True,
            )
            self._health_threads.append(health)
            health.start()

    def _health_check(
        self,
        spec: PanelProcess,
        process: subprocess.Popen[str],
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
        assert spec.task_id is not None
        run_dir = latest_run_dir(spec.task_root, spec.task_id)
        output_path = self.config.result_root / "health" / f"{spec.label}.json"
        if run_dir is None:
            atomic_write_json(
                output_path,
                {
                    "benchmark": spec.benchmark,
                    "task_id": spec.task_id,
                    "status": "unhealthy",
                    "objective_stop_required": True,
                    "fatal": ["run directory missing after health delay"],
                },
            )
            self._terminate_process(process)
            return
        completed = subprocess.run(
            (
                sys.executable,
                str(REPO_ROOT / "scripts" / "python" / "check_terminal_run_health.py"),
                "--benchmark",
                spec.benchmark,
                "--run-dir",
                str(run_dir),
                "--pid",
                str(process.pid),
                "--process-log",
                str(log_path),
            ),
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        payload = read_optional_json(run_dir / "health-check-5m.json")
        if payload is None:
            payload = {
                "benchmark": spec.benchmark,
                "task_id": spec.task_id,
                "status": "checker_failed",
                "objective_stop_required": True,
                "checker_returncode": completed.returncode,
                "fatal": ["health checker did not produce a valid payload"],
            }
        atomic_write_json(output_path, payload)
        (self.config.result_root / "health" / f"{spec.label}.log").write_text(
            completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        if (
            completed.returncode not in (0, 1)
            or payload.get("objective_stop_required") is True
        ):
            self._terminate_process(process)

    def _wait_for_processes(self, specs: tuple[PanelProcess, ...]) -> None:
        remaining = {spec.label: spec for spec in specs}
        cleanup_started = False
        while remaining:
            completed: list[str] = []
            for label, spec in remaining.items():
                with self._process_lock:
                    process = self._processes.get(label)
                if process is None:
                    completed.append(label)
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                with self._process_lock:
                    log = self._logs.pop(label, None)
                    self._processes.pop(label, None)
                if log is not None:
                    log.close()
                (self.config.result_root / "exit" / f"{spec.label}.txt").write_text(
                    f"{return_code}\n",
                    encoding="utf-8",
                )
                completed.append(label)
            for label in completed:
                remaining.pop(label, None)
            if self.stop_event.is_set() and not cleanup_started:
                cleanup_started = True
                self._stop_all_processes()
            if remaining:
                self.stop_event.wait(0.5)

    def _write_delayed_checkpoint(
        self,
        commit: str,
        specs: tuple[PanelProcess, ...],
    ) -> None:
        if self.stop_event.wait(self.config.checkpoint_delay_seconds):
            return
        atomic_write_json(
            self.config.result_root / "checkpoint-30m.json",
            self._snapshot(commit, specs, checkpoint=True),
        )

    def _join_health_threads(self) -> None:
        for health in self._health_threads:
            health.join()

    def _write_summary(
        self,
        commit: str,
        specs: tuple[PanelProcess, ...],
    ) -> dict[str, Any]:
        summary = self._snapshot(commit, specs, checkpoint=False)
        atomic_write_json(
            self.config.result_root / "panel-summary.json",
            summary,
        )
        return summary

    def _snapshot(
        self,
        commit: str,
        specs: tuple[PanelProcess, ...],
        *,
        checkpoint: bool,
    ) -> dict[str, Any]:
        rows = []
        for spec in specs:
            with self._process_lock:
                process = self._processes.get(spec.label)
            return_code = None
            exit_path = self.config.result_root / "exit" / f"{spec.label}.txt"
            if exit_path.is_file():
                return_code = int(exit_path.read_text().strip())
            row: dict[str, Any] = {
                "label": spec.label,
                "benchmark": spec.benchmark,
                "task_id": spec.task_id,
                "pid": None if process is None else process.pid,
                "alive": process is not None and process.poll() is None,
                "exit_code": return_code,
            }
            if spec.manages_own_health:
                row["batch_summary"] = read_optional_json(
                    spec.task_root / "batch-summary.json"
                )
                health_dir = spec.task_root / "health"
                row["health_files"] = (
                    len(list(health_dir.glob("*.json"))) if health_dir.is_dir() else 0
                )
            else:
                assert spec.task_id is not None
                run_dir = latest_run_dir(spec.task_root, spec.task_id)
                row["run_dir"] = None if run_dir is None else str(run_dir)
                row["health"] = read_optional_json(
                    self.config.result_root / "health" / f"{spec.label}.json"
                )
                row["result"] = (
                    None
                    if run_dir is None
                    else read_optional_json(run_dir / "evaluation" / "result.json")
                )
            rows.append(row)
        return {
            "schema_version": 1,
            "kind": "checkpoint-30m" if checkpoint else "final",
            "commit": commit,
            "task_count": PANEL_TASK_COUNT,
            "effective_width": PANEL_TASK_COUNT,
            "rows": rows,
            "captured_at_epoch": time.time(),
        }

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
            signal_process_group(process, signal.SIGINT)
        self._wait_then_signal(processes, timeout=60, signum=signal.SIGTERM)
        self._wait_then_signal(processes, timeout=20, signum=signal.SIGKILL)
        for process in processes:
            if process.poll() is None:
                process.wait(timeout=15)
        with self._process_lock:
            logs = list(self._logs.values())
            self._logs.clear()
        for log in logs:
            log.close()

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        terminate_process(process)

    def _wait_then_signal(
        self,
        processes: list[subprocess.Popen[str]],
        *,
        timeout: float,
        signum: signal.Signals,
    ) -> None:
        deadline = time.monotonic() + timeout
        for process in processes:
            if process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        for process in processes:
            if process.poll() is None:
                signal_process_group(process, signum)


def _panel_failed(summary: dict[str, Any]) -> bool:
    for row in summary["rows"]:
        if row["exit_code"] not in (0, None):
            return True
        if row["benchmark"] == "osworld":
            batch = row.get("batch_summary")
            if not isinstance(batch, dict):
                return True
            if (
                batch.get("nonzero_exits")
                or batch.get("selection_errors")
                or batch.get("health_failures")
            ):
                return True
        else:
            if objective_health_failure(row.get("health")):
                return True
            result = row.get("result")
            if not isinstance(result, dict):
                return True
            if (
                row["benchmark"] == "cadgenbench"
                and result.get("deadline_triggered") is True
            ):
                return True
    return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 27-task CUAWright regression panel"
    )
    parser.add_argument("--result-root", required=True)
    vm_path = configured_value("osworld", "vm")
    parser.add_argument("--path-to-vm", default=vm_path, required=not vm_path)
    parser.add_argument("--assets-dir", default=str(DEFAULT_OSWORLD_ASSETS))
    parser.add_argument("--api-key-file", default=str(DEFAULT_API_KEY_FILE))
    parser.add_argument("--proxy-config", default=str(DEFAULT_PROXY_CONFIG))
    parser.add_argument(
        "--website-host-suffix",
        default=os.getenv("WEBSITE_HOST_SUFFIX", "10.217.29.172.nip.io"),
    )
    parser.add_argument("--benchcad-manifest", required=True)
    parser.add_argument("--benchcad-repo", required=True)
    parser.add_argument("--benchcad-image", default="cuawright-benchcad:52087ef")
    parser.add_argument("--cadgenbench-dataset", required=True)
    parser.add_argument("--cadgenbench-repo", required=True)
    parser.add_argument(
        "--cadgenbench-image",
        default="cuawright-cadgenbench:33304cf",
    )
    parser.add_argument("--health-delay-seconds", type=int, default=300)
    parser.add_argument("--checkpoint-delay-seconds", type=int, default=1_800)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.health_delay_seconds < 1:
        raise ValueError("health-delay-seconds must be positive")
    if args.checkpoint_delay_seconds < 1:
        raise ValueError("checkpoint-delay-seconds must be positive")
    if not args.website_host_suffix.strip():
        raise ValueError("website-host-suffix cannot be empty")
    config = PanelConfig(
        result_root=Path(args.result_root).expanduser().resolve(),
        vm_path=Path(args.path_to_vm).expanduser().resolve(),
        assets_dir=Path(args.assets_dir).expanduser().resolve(),
        api_key_file=Path(args.api_key_file).expanduser().resolve(),
        proxy_config=Path(args.proxy_config).expanduser().resolve(),
        website_host_suffix=args.website_host_suffix,
        benchcad_manifest=Path(args.benchcad_manifest).expanduser().resolve(),
        benchcad_repo=Path(args.benchcad_repo).expanduser().resolve(),
        benchcad_image=args.benchcad_image,
        cadgenbench_dataset=Path(args.cadgenbench_dataset).expanduser().resolve(),
        cadgenbench_repo=Path(args.cadgenbench_repo).expanduser().resolve(),
        cadgenbench_image=args.cadgenbench_image,
        health_delay_seconds=args.health_delay_seconds,
        checkpoint_delay_seconds=args.checkpoint_delay_seconds,
    )
    raise SystemExit(RegressionPanel(config).run())


if __name__ == "__main__":
    main()

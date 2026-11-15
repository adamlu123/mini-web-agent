from __future__ import annotations

from collections import deque
import hashlib
import io
import json
import logging
import math
import os
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docker.errors import DockerException

from cuawright.environments.cadgenbench.docker_exec import run_exec
from cuawright.environments.cadgenbench.errors import CadGenBenchInfrastructureError
from cuawright.environments.cadgenbench.image import validate_image
from cuawright.environments.cadgenbench.task import CadGenBenchTask, sha256_file
from cuawright.environments.control_clients import (
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
)

COMMAND_CAPTURE_LIMIT = 1_000_000
FILE_READ_LIMIT = 128 * 1024 * 1024
CONTROL_BRIDGE_DIR = "/run/cadgenbench-control"
WORKSPACE_SIZE = 2 * 1024 * 1024 * 1024
ARTIFACT_FILE_LIMIT = 256 * 1024 * 1024
MCP_RESTART_LIMIT = 3
CONTAINER_MONITOR_INTERVAL_SECONDS = 10.0
CONTAINER_TELEMETRY_SAMPLE_LIMIT = 128
CONTAINER_LOG_TAIL_BYTES = 16_000
MCP_HTTP_PORT = 8765
MCP_HTTP_URL = f"http://127.0.0.1:{MCP_HTTP_PORT}/mcp"
MCP_STARTUP_TIMEOUT_SECONDS = 60

logger = logging.getLogger("cuawright.environments.cadgenbench.environment")

_CONTROL_CLIENT_VERIFY_PROGRAM = r"""
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = json.loads(sys.argv[2])
found = {path.name for path in root.glob("*.py")}
if found != set(expected):
    raise RuntimeError(
        "control-client file set mismatch: "
        + json.dumps({"expected": sorted(expected), "found": sorted(found)})
    )
for name, expected_hash in expected.items():
    path = root / name
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(name + " is not a regular file")
    if info.st_uid != 0 or info.st_gid != 0:
        raise RuntimeError(name + " is not root-owned")
    if stat.S_IMODE(info.st_mode) & 0o222:
        raise RuntimeError(name + " is writable")
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError(name + " has an unexpected sha256")
print(json.dumps(expected, sort_keys=True, separators=(",", ":")))
"""


class CadGenBenchEnvironment:
    def __init__(
        self,
        *,
        task: CadGenBenchTask,
        run_dir: Path,
        image: str,
        benchmark_commit: str,
    ) -> None:
        import docker

        self.task = task
        self.provenance = task.provenance()
        self.run_dir = run_dir
        self.image_ref = image
        self.benchmark_commit = benchmark_commit
        self.workspace_dir = run_dir / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.workspace_dir, 0o700)
        self.latest_valid_dir = run_dir / "latest-valid"
        self.latest_valid_dir.mkdir(parents=True, exist_ok=True)
        self.latest_valid_output = self.latest_valid_dir / "output.step"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_history_path = run_dir / "checkpoint-history.json"
        self.container_telemetry_path = run_dir / "container-telemetry.jsonl"
        self.container_failure_path = run_dir / "container-failure.json"
        self.container_lifecycle_dir = run_dir / "container-lifecycle"
        self.container_lifecycle_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_history: list[dict[str, Any]] = []
        self.mcp_restart_history: list[dict[str, Any]] = []
        self.mcp_telemetry: dict[str, Any] | None = None
        self._candidate_signatures: dict[str, tuple[int, int]] = {}
        self._container_lock = threading.RLock()
        self._telemetry_lock = threading.Lock()
        self._container_generation = 0
        self._current_tool_name: str | None = None
        self._current_stage: str | None = None
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._recent_container_samples: deque[dict[str, Any]] = deque(
            maxlen=CONTAINER_TELEMETRY_SAMPLE_LIMIT
        )
        self._peak_memory_usage_bytes = 0
        self._peak_memory_limit_bytes = 0
        self._peak_pids = 0
        self._container_exit_states: list[dict[str, Any]] = []
        self._container_shutdown_expected = False
        self._harness_kill_events: list[dict[str, Any]] = []

        self.client = docker.from_env()
        self.image_id = validate_image(self.client, image, benchmark_commit)
        self.image = self.image_id
        self.workspace_volume = None
        self.container = None
        try:
            self.workspace_volume = self.client.volumes.create(
                name=(
                    f"cuawright-cadgenbench-{task.fixture_id}-"
                    f"{uuid.uuid4().hex[:12]}"
                ),
                driver="local",
                driver_opts={
                    "type": "tmpfs",
                    "device": "tmpfs",
                    "o": (
                        f"size={WORKSPACE_SIZE},uid={os.getuid()},"
                        f"gid={os.getgid()},mode=0700"
                    ),
                },
                labels={
                    "cuawright.component": "cadgenbench-workspace",
                    "cuawright.fixture_id": task.fixture_id,
                },
            )
            initial_container = self._start_agent_container()
            self._set_current_container(initial_container, reason="initial_start")
            self._verify_control_clients(initial_container)
            self._start_container_monitor()
        except (CadGenBenchInfrastructureError, DockerException):
            self._rollback_startup()
            raise
        self.controller = self
        self.user_simulator = None
        self.action_history: list[str] = []
        self.is_environment_used = False
        self.enable_proxy = False
        self.instruction = task.instruction
        self._frozen = False
        self._removed = False
        self._volume_removed = False
        try:
            self._stage_workspace_inputs()
            self.mcp_telemetry = self._wait_for_mcp_service()
            self._write_checkpoint_history()
        except Exception:
            self._rollback_startup()
            raise

    def _verify_control_clients(self, container: Any) -> None:
        expected = dict(CLIENT_SHA256)
        try:
            result = run_exec(
                client=self.client,
                container=container,
                command=[
                    "/opt/cadgenbench-venv/bin/python",
                    "-c",
                    _CONTROL_CLIENT_VERIFY_PROGRAM,
                    CLIENT_INSTALL_ROOT,
                    json.dumps(expected, sort_keys=True, separators=(",", ":")),
                ],
                user=f"{os.getuid()}:{os.getgid()}",
                deadline=30,
                capture_limit=64 * 1024,
                workdir="/",
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
            )
        except DockerException as exc:
            raise CadGenBenchInfrastructureError(
                "CADGenBench control-client preflight failed"
            ) from exc
        if (
            result["timed_out"]
            or result["container_killed"]
            or int(result["returncode"]) != 0
        ):
            detail = result["stderr"].decode("utf-8", errors="replace").strip()
            raise CadGenBenchInfrastructureError(
                "CADGenBench control-client preflight failed"
                + (f": {detail[-2000:]}" if detail else "")
            )
        try:
            observed = json.loads(result["stdout"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CadGenBenchInfrastructureError(
                "CADGenBench control-client preflight returned invalid output"
            ) from exc
        if observed != expected:
            raise CadGenBenchInfrastructureError(
                "CADGenBench control-client preflight returned unexpected hashes"
            )

    def _rollback_startup(self) -> None:
        self._container_shutdown_expected = True
        self._stop_container_monitor()
        self._remove_current_container(
            lifecycle_reason="pre_rollback_remove",
            log_failures=True,
        )
        if self.workspace_volume is not None:
            try:
                self.workspace_volume.remove(force=True)
                self._volume_removed = True
            except DockerException:
                logger.exception("Failed to roll back CadGenBench workspace volume")
        self.client.close()

    def _agent_container_kwargs(self) -> dict[str, Any]:
        return {
            "command": [
                "/opt/cadgenbench-venv/bin/build123d-mcp",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(MCP_HTTP_PORT),
                "--no-sandbox",
                "--exec-timeout",
                "120",
            ],
            "detach": True,
            "network_mode": "none",
            "read_only": True,
            "user": f"{os.getuid()}:{os.getgid()}",
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "pids_limit": 1024,
            "mem_limit": "12g",
            "nano_cpus": 4_000_000_000,
            "tmpfs": {
                "/tmp": "rw,noexec,nosuid,size=2g,mode=1777",
                "/home/agent": (
                    "rw,nosuid,size=256m,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
                CONTROL_BRIDGE_DIR: "rw,noexec,nosuid,size=32m,mode=0711,uid=0,gid=0",
            },
            "volumes": {
                str(self.task.fixture_dir): {"bind": "/input", "mode": "ro"},
                self.workspace_volume.name: {"bind": "/workspace", "mode": "rw"},
            },
            "labels": {
                "cuawright.component": "cadgenbench-agent",
                "cuawright.fixture_id": self.task.fixture_id,
            },
        }

    def _start_agent_container(self) -> Any:
        return self.client.containers.run(
            self.image,
            **self._agent_container_kwargs(),
        )

    def _current_container(self) -> Any | None:
        lock = getattr(self, "_container_lock", None)
        if lock is None:
            return getattr(self, "container", None)
        with lock:
            return self.container

    def _current_container_generation(self) -> int:
        lock = getattr(self, "_container_lock", None)
        if lock is None:
            return int(getattr(self, "_container_generation", 0))
        with lock:
            return self._container_generation

    def _set_current_container(self, container: Any, *, reason: str) -> None:
        with self._container_lock:
            self.container = container
            self._container_generation += 1
            generation = self._container_generation
        self._container_shutdown_expected = False
        self._removed = False
        self._record_container_lifecycle_event(
            "container_generation_start",
            {
                "reason": reason,
                "container_generation": generation,
                "container_id": container.id,
                "container_name": getattr(container, "name", None),
            },
        )

    def set_runtime_context(self, *, tool_name: str, stage: str) -> None:
        self._current_tool_name = tool_name
        self._current_stage = stage

    def clear_runtime_context(self) -> None:
        self._current_tool_name = None
        self._current_stage = None

    def _append_container_telemetry_record(self, payload: dict[str, Any]) -> None:
        lock = getattr(self, "_telemetry_lock", None)
        if lock is None:
            self._append_container_telemetry_record_locked(payload)
            return
        with lock:
            self._append_container_telemetry_record_locked(payload)

    def _append_container_telemetry_record_locked(
        self,
        payload: dict[str, Any],
    ) -> None:
        telemetry_path = getattr(self, "container_telemetry_path", None)
        if telemetry_path is not None:
            line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        if payload.get("kind") != "sample":
            return
        self._recent_container_samples.append(payload)
        memory = payload.get("memory") or {}
        pids = payload.get("pids") or {}
        usage = int(memory.get("usage_bytes") or 0)
        limit = int(memory.get("limit_bytes") or 0)
        current_pids = int(pids.get("current") or 0)
        self._peak_memory_usage_bytes = max(self._peak_memory_usage_bytes, usage)
        self._peak_memory_limit_bytes = max(self._peak_memory_limit_bytes, limit)
        self._peak_pids = max(self._peak_pids, current_pids)
        state = payload.get("state") or {}
        exit_state = {
            "status": state.get("Status"),
            "exit_code": state.get("ExitCode"),
            "oom_killed": state.get("OOMKilled"),
            "dead": state.get("Dead"),
        }
        if exit_state not in self._container_exit_states:
            self._container_exit_states.append(exit_state)

    def _record_container_lifecycle_event(
        self,
        event: str,
        details: dict[str, Any],
    ) -> None:
        self._append_container_telemetry_record(
            {
                "kind": "lifecycle",
                "timestamp_utc": self._utc_now(),
                "event": event,
                **details,
            }
        )

    def _container_monitor_loop(self) -> None:
        while not self._monitor_stop.wait(CONTAINER_MONITOR_INTERVAL_SECONDS):
            try:
                container = self._current_container()
                if container is None:
                    continue
                generation = self._current_container_generation()
                sample = self._collect_container_sample(container, generation)
                self._append_container_telemetry_record(sample)
                state = sample["state"]
                if (
                    state.get("Status") != "running"
                    and not self._container_shutdown_expected
                ):
                    self._write_container_failure(
                        reason="monitor_detected_nonrunning",
                        state=state,
                        container_id=sample.get("container_id"),
                        container_generation=sample.get("container_generation"),
                    )
            except Exception as exc:  # noqa: BLE001 - telemetry must not crash task
                self._record_container_lifecycle_event(
                    "sampling_error",
                    {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

    def _start_container_monitor(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._container_monitor_loop,
            name=f"cadgenbench-container-monitor-{self.task.fixture_id}",
            daemon=True,
        )
        self._monitor_thread.start()

    def _stop_container_monitor(self) -> None:
        monitor_stop = getattr(self, "_monitor_stop", None)
        if monitor_stop is not None:
            monitor_stop.set()
        monitor_thread = getattr(self, "_monitor_thread", None)
        if monitor_thread is not None:
            monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def _sanitized_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            key: state.get(key)
            for key in (
                "Status",
                "Running",
                "Paused",
                "Restarting",
                "OOMKilled",
                "Dead",
                "Pid",
                "ExitCode",
                "Error",
                "StartedAt",
                "FinishedAt",
            )
        }

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        memory_stats = stats.get("memory_stats") or {}
        cpu_stats = stats.get("cpu_stats") or {}
        cpu_usage = cpu_stats.get("cpu_usage") or {}
        blkio_recursive = (stats.get("blkio_stats") or {}).get(
            "io_service_bytes_recursive"
        ) or []
        block_io: dict[str, int] = {}
        for item in blkio_recursive:
            operation = str(item.get("op") or "").lower()
            value = self._safe_int(item.get("value")) or 0
            if operation:
                block_io[operation] = block_io.get(operation, 0) + value
        return {
            "memory": {
                "usage_bytes": self._safe_int(memory_stats.get("usage")),
                "max_usage_bytes": self._safe_int(memory_stats.get("max_usage")),
                "limit_bytes": self._safe_int(memory_stats.get("limit")),
            },
            "pids": {
                "current": self._safe_int(
                    (stats.get("pids_stats") or {}).get("current")
                ),
            },
            "cpu": {
                "total_usage": self._safe_int(cpu_usage.get("total_usage")),
                "system_cpu_usage": self._safe_int(cpu_stats.get("system_cpu_usage")),
                "online_cpus": self._safe_int(cpu_stats.get("online_cpus")),
            },
            "block_io": block_io or None,
        }

    def _collect_container_sample(
        self, container: Any, generation: int
    ) -> dict[str, Any]:
        inspect = self.client.api.inspect_container(container.id)
        sample = {
            "kind": "sample",
            "timestamp_utc": self._utc_now(),
            "container_generation": generation,
            "container_id": container.id,
            "container_name": inspect.get("Name", "").lstrip("/")
            or getattr(container, "name", None),
            "container_status": getattr(container, "status", None),
            "state": self._sanitized_state(inspect.get("State") or {}),
        }
        try:
            stats = container.stats(stream=False)
        except DockerException as exc:
            sample["stats_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        else:
            sample.update(self._normalize_stats(stats))
        return sample

    def _capture_container_lifecycle_snapshot(
        self,
        reason: str,
        *,
        container: Any | None = None,
    ) -> dict[str, Any] | None:
        target = container or self._current_container()
        if target is None:
            return None
        try:
            inspect = self.client.api.inspect_container(target.id)
            state = self._sanitized_state(inspect.get("State") or {})
        except DockerException as exc:
            state = {"Status": "inspect_failed", "Error": str(exc)}
        try:
            logs = target.logs(stdout=True, stderr=True)
            logs_tail = logs[-CONTAINER_LOG_TAIL_BYTES:].decode(
                "utf-8", errors="replace"
            )
        except Exception as exc:  # noqa: BLE001 - best effort
            logs_tail = f"<log capture failed: {type(exc).__name__}: {exc}>"
        generation = self._current_container_generation()
        payload = {
            "timestamp_utc": self._utc_now(),
            "reason": reason,
            "container_generation": generation,
            "container_id": target.id,
            "container_name": (
                inspect.get("Name", "").lstrip("/")
                if "inspect" in locals()
                else getattr(target, "name", None)
            ),
            "state": state,
            "logs_tail": logs_tail,
        }
        path = self.container_lifecycle_dir / (
            f"{len(list(self.container_lifecycle_dir.glob('*.json'))) + 1:03d}-"
            f"{reason}-{target.id[:12]}.json"
        )
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._record_container_lifecycle_event(
            reason,
            {
                "container_generation": generation,
                "container_id": target.id,
                "container_name": payload["container_name"],
                "state": state,
                "snapshot_path": str(path.relative_to(self.run_dir)),
            },
        )
        return payload

    def _classify_container_failure(self, state: dict[str, Any]) -> list[str]:
        classifications: list[str] = []
        if state.get("OOMKilled"):
            classifications.append("oom_killed")
        if (self._safe_int(state.get("ExitCode")) or 0) != 0:
            classifications.append("nonzero_exit")
        if state.get("Error"):
            classifications.append("daemon_error")
        if not classifications:
            classifications.append("unknown")
        return classifications

    def _write_container_failure(
        self,
        *,
        reason: str,
        state: dict[str, Any],
        container_id: str | None = None,
        container_generation: int | None = None,
    ) -> None:
        if container_id is None:
            container_id = getattr(self._current_container(), "id", None)
        if container_generation is None:
            container_generation = self._current_container_generation()
        harness_killed = any(
            event.get("container_id") == container_id
            and event.get("container_generation") == container_generation
            for event in getattr(self, "_harness_kill_events", [])
        )
        payload = {
            "timestamp_utc": self._utc_now(),
            "reason": reason,
            "container_id": container_id,
            "container_generation": container_generation,
            "state": state,
            "current_tool": getattr(self, "_current_tool_name", None),
            "current_stage": getattr(self, "_current_stage", None),
            "checkpoint_count": len(getattr(self, "checkpoint_history", [])),
            "latest_valid_output_sha256": (
                sha256_file(self.latest_valid_output)
                if getattr(self, "latest_valid_output", None) is not None
                and self.latest_valid_output.is_file()
                else None
            ),
            "mcp_service": self.mcp_telemetry,
            "last_telemetry_samples": list(
                getattr(self, "_recent_container_samples", [])
            )[-10:],
            "likely_classifications": self._classify_container_failure(state),
            "exit_origin": "harness_kill" if harness_killed else "external_exit",
            "checkpoint_history_path": (
                str(self.checkpoint_history_path.relative_to(self.run_dir))
                if getattr(self, "checkpoint_history_path", None) is not None
                and getattr(self, "run_dir", None) is not None
                else None
            ),
        }
        failure_path = getattr(self, "container_failure_path", None)
        if failure_path is not None:
            failure_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _container_observability_summary(self) -> dict[str, Any]:
        return {
            "telemetry_path": str(
                self.container_telemetry_path.relative_to(self.run_dir)
            ),
            "failure_path": (
                str(self.container_failure_path.relative_to(self.run_dir))
                if self.container_failure_path.exists()
                else None
            ),
            "lifecycle_dir": str(
                self.container_lifecycle_dir.relative_to(self.run_dir)
            ),
            "peak_memory_usage_bytes": getattr(self, "_peak_memory_usage_bytes", 0),
            "peak_memory_limit_bytes": getattr(self, "_peak_memory_limit_bytes", 0),
            "peak_pids": getattr(self, "_peak_pids", 0),
            "exit_states": getattr(self, "_container_exit_states", []),
            "recreation_count": sum(
                1
                for entry in getattr(self, "mcp_restart_history", [])
                if entry.get("container_recreated")
            ),
            "harness_kill_count": len(getattr(self, "_harness_kill_events", [])),
        }

    def _on_harness_kill_container(self, details: dict[str, Any]) -> None:
        payload = {
            "timestamp_utc": self._utc_now(),
            "tool": self._current_tool_name,
            "stage": self._current_stage,
            "container_generation": self._current_container_generation(),
            **details,
        }
        self._harness_kill_events.append(payload)
        self._record_container_lifecycle_event("harness_container_kill", payload)
        container = self._current_container()
        if container is not None:
            snapshot = self._capture_container_lifecycle_snapshot(
                "pre_harness_kill_remove",
                container=container,
            )
            if snapshot is not None:
                self._write_container_failure(
                    reason="harness_kill",
                    state=snapshot["state"],
                    container_id=snapshot.get("container_id"),
                    container_generation=snapshot.get("container_generation"),
                )

    def _remove_current_container(
        self,
        *,
        lifecycle_reason: str = "pre_remove",
        log_failures: bool = False,
    ) -> None:
        current = self._current_container()
        if current is None:
            return
        self._capture_container_lifecycle_snapshot(lifecycle_reason, container=current)
        try:
            current.remove(force=True)
        except DockerException:
            if log_failures:
                logger.exception("Failed to roll back CadGenBench container")
                return
            raise
        with self._container_lock:
            if self.container is current:
                self.container = None
        self._removed = True

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _command_environment(self) -> dict[str, str]:
        return {
            "HOME": "/home/agent",
            "PATH": "/opt/cadgenbench-venv/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/opt/cuawright:/opt/cadgenbench-source/src",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "CAD_MCP_URL": MCP_HTTP_URL,
            "CAD_MCP_TIMEOUT_SECONDS": "600",
        }

    def _stage_workspace_inputs(self) -> None:
        script = (
            "from pathlib import Path\n"
            "import shutil\n"
            f"pairs = {list(self.task.staged_workspace_inputs)!r}\n"
            "source_root = Path('/input')\n"
            "target_root = Path('/workspace')\n"
            "for source_rel, target_rel in pairs:\n"
            "    source = source_root / source_rel\n"
            "    target = target_root / target_rel\n"
            "    target.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copyfile(source, target)\n"
        )
        try:
            execution = run_exec(
                client=self.client,
                container=self.container,
                command=["python", "-c", script],
                user=f"{os.getuid()}:{os.getgid()}",
                deadline=60,
                capture_limit=COMMAND_CAPTURE_LIMIT,
                workdir="/workspace",
                environment=self._command_environment(),
            )
        except DockerException as exc:
            raise CadGenBenchInfrastructureError(
                f"Failed staging CadGenBench workspace inputs: {exc}"
            ) from exc
        if int(execution["returncode"]) != 0:
            raise CadGenBenchInfrastructureError(
                "Failed staging CadGenBench workspace inputs: "
                + execution["stderr"].decode("utf-8", errors="replace")[-4000:]
            )

    def _run_service_command(
        self,
        command: list[str],
        *,
        deadline: int = 30,
    ) -> dict[str, Any]:
        try:
            return run_exec(
                client=self.client,
                container=self.container,
                command=command,
                user=f"{os.getuid()}:{os.getgid()}",
                deadline=deadline,
                capture_limit=COMMAND_CAPTURE_LIMIT,
                workdir="/workspace",
                environment=self._command_environment(),
            )
        except DockerException as exc:
            raise CadGenBenchInfrastructureError(
                f"CadGenBench MCP service command failed: {exc}"
            ) from exc

    def _wait_for_mcp_service(
        self,
        timeout_seconds: int = MCP_STARTUP_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            status = self._reload_container_status()
            if status != "running":
                raise CadGenBenchInfrastructureError(
                    f"CadGenBench MCP service container exited during startup: {status}"
                )
            result = self._run_service_command(
                ["cad-mcp", "--timeout", "10", "call", "health_check"],
                deadline=15,
            )
            if int(result["returncode"]) == 0:
                health = json.loads(result["stdout"].decode("utf-8", errors="replace"))
                version_result = self._run_service_command(
                    ["cad-mcp", "--timeout", "10", "call", "version"],
                    deadline=15,
                )
                tools_result = self._run_service_command(
                    ["cad-mcp", "--timeout", "10", "tools"],
                    deadline=15,
                )
                if int(version_result["returncode"]) != 0:
                    raise CadGenBenchInfrastructureError(
                        "CadGenBench MCP version probe failed: "
                        + version_result["stderr"].decode(
                            "utf-8",
                            errors="replace",
                        )[-4000:]
                    )
                if int(tools_result["returncode"]) != 0:
                    raise CadGenBenchInfrastructureError(
                        "CadGenBench MCP tool-list probe failed: "
                        + tools_result["stderr"].decode(
                            "utf-8",
                            errors="replace",
                        )[-4000:]
                    )
                return {
                    "transport": "streamable_http",
                    "endpoint": MCP_HTTP_URL,
                    "container_generation": self._current_container_generation(),
                    "health_check": health,
                    "version": json.loads(
                        version_result["stdout"].decode(
                            "utf-8",
                            errors="replace",
                        )
                    ),
                    "tools": json.loads(
                        tools_result["stdout"].decode(
                            "utf-8",
                            errors="replace",
                        )
                    ),
                    "ready_at": self._utc_now(),
                }
            last_error = (
                result["stderr"].decode("utf-8", errors="replace")
                or result["stdout"].decode("utf-8", errors="replace")
            )[-4000:]
            time.sleep(0.25)
        raise CadGenBenchInfrastructureError(
            "CadGenBench MCP HTTP service did not become ready within "
            f"{timeout_seconds} seconds: {last_error}"
        )

    def run_command(
        self,
        script: str,
        timeout: int,
        *,
        wall_limit_seconds: float | None = None,
    ) -> dict[str, Any]:
        command_started = time.monotonic()
        recovery = None
        try:
            if self._reload_container_status() != "running":
                recovery = self._recover_mcp_service(
                    reason="pre_command_container_not_running"
                )
        except CadGenBenchInfrastructureError as exc:
            return {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": str(exc),
                "infrastructure_error": True,
            }
        command_timeout = max(1, int(timeout))
        exec_deadline = command_timeout + 15
        kill_grace_seconds = 5.0
        if wall_limit_seconds is not None:
            remaining = max(1, math.ceil(wall_limit_seconds))
            command_timeout = min(command_timeout, remaining)
            exec_deadline = min(command_timeout + 15, remaining)
            kill_grace_seconds = min(5.0, max(0.0, remaining - command_timeout))
        command = [
            "/usr/bin/timeout",
            "--verbose",
            "--signal=TERM",
            "--kill-after=5s",
            f"{command_timeout}s",
            "/bin/bash",
            "-c",
            script,
        ]
        try:
            execution = run_exec(
                client=self.client,
                container=self.container,
                command=command,
                user=f"{os.getuid()}:{os.getgid()}",
                deadline=exec_deadline,
                capture_limit=COMMAND_CAPTURE_LIMIT,
                workdir="/workspace",
                environment=self._command_environment(),
                kill_grace_seconds=kill_grace_seconds,
                container_kill_callback=self._on_harness_kill_container,
            )
        except DockerException as exc:
            payload = {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": str(exc),
                "infrastructure_error": True,
            }
            try:
                payload["mcp_recovery"] = self._recover_mcp_service(
                    reason="command_transport_failure"
                )
                payload["infrastructure_error"] = False
            except CadGenBenchInfrastructureError as recovery_exc:
                payload["recovery_error"] = str(recovery_exc)
            return payload
        output = execution["stdout"].decode("utf-8", errors="replace")
        error = execution["stderr"].decode("utf-8", errors="replace")
        returncode = int(execution["returncode"])
        if execution["stdout_truncated"]:
            output += "\n[stdout capture truncated at 1000000 bytes]"
        if execution["stderr_truncated"]:
            error += "\n[stderr capture truncated at 1000000 bytes]"
        payload: dict[str, Any] = {
            "status": "success" if returncode == 0 else "error",
            "returncode": returncode,
            "output": output,
            "error": error,
            "infrastructure_error": False,
        }
        if execution["timed_out"] or (
            returncode == 124 and "timeout: sending signal" in error.lower()
        ):
            payload["timed_out"] = True
            payload["error"] = (
                f"Command exceeded {command_timeout} seconds; its process group was "
                f"terminated.\n{error}"
            ).strip()
        if execution["container_killed"]:
            try:
                payload["mcp_recovery"] = self._recover_mcp_service(
                    reason="command_timeout_killed_container"
                )
                payload["infrastructure_error"] = False
            except CadGenBenchInfrastructureError as exc:
                payload["infrastructure_error"] = True
                payload["recovery_error"] = str(exc)
            return payload
        if returncode == 70 and "cad-mcp" in script:
            try:
                payload["mcp_recovery"] = self._recover_mcp_service(
                    reason="cad_mcp_transport_failure"
                )
                payload["infrastructure_error"] = False
            except CadGenBenchInfrastructureError as exc:
                payload["infrastructure_error"] = True
                payload["recovery_error"] = str(exc)
            return payload
        if recovery is not None:
            payload["pre_command_recovery"] = recovery
        checkpoint_deadline = None
        if wall_limit_seconds is not None:
            checkpoint_deadline = max(
                1.0,
                wall_limit_seconds - (time.monotonic() - command_started),
            )
        checkpoint = self._checkpoint_changed_candidate(
            deadline_seconds=checkpoint_deadline,
        )
        if checkpoint is not None:
            payload["checkpoint"] = checkpoint
        self.is_environment_used = True
        return payload

    def get_file(
        self,
        path: str,
        *,
        max_bytes: int = FILE_READ_LIMIT,
    ) -> bytes | None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
            or max_bytes > FILE_READ_LIMIT
        ):
            raise ValueError(f"max_bytes must be between 1 and {FILE_READ_LIMIT}")
        try:
            stream, stat_result = self.container.get_archive(path)
        except DockerException:
            return self._get_file_via_exec(path, max_bytes=max_bytes)
        if int(stat_result.get("size", max_bytes + 1)) > max_bytes:
            return None
        archive = io.BytesIO()
        for chunk in stream:
            if archive.tell() + len(chunk) > max_bytes + 1024 * 1024:
                return None
            archive.write(chunk)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            if len(members) != 1 or members[0].size > max_bytes:
                return None
            extracted = tar.extractfile(members[0])
            if extracted is None:
                return None
            data = extracted.read(max_bytes + 1)
            return data if len(data) <= max_bytes else None

    def _get_file_via_exec(
        self,
        path: str,
        *,
        max_bytes: int = FILE_READ_LIMIT,
    ) -> bytes | None:
        try:
            execution = self.client.api.exec_create(
                self.container.id,
                ["/bin/cat", "--", path],
                user=f"{os.getuid()}:{os.getgid()}",
            )
            output_stream = self.client.api.exec_start(
                execution["Id"],
                stream=True,
                demux=True,
            )
            data = bytearray()
            for stdout, _ in output_stream:
                if stdout:
                    if len(data) + len(stdout) > max_bytes:
                        return None
                    data.extend(stdout)
            inspection = self.client.api.exec_inspect(execution["Id"])
        except DockerException:
            return None
        if int(inspection["ExitCode"]) != 0:
            return None
        return bytes(data)

    def _live_candidate_state(
        self,
        *,
        deadline_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        script = (
            "import json\n"
            "from pathlib import Path\n"
            "for name in ('output.step', 'output.stp'):\n"
            "    path = Path('/workspace') / name\n"
            "    if path.is_file() and path.stat().st_size > 0:\n"
            "        info = path.stat()\n"
            "        print(json.dumps({'path': str(path), 'size': info.st_size, "
            "'mtime_ns': info.st_mtime_ns}))\n"
            "        break\n"
        )
        result = self._run_service_command(
            ["python", "-c", script],
            deadline=max(
                1,
                min(
                    15,
                    math.ceil(deadline_seconds) if deadline_seconds is not None else 15,
                ),
            ),
        )
        if int(result["returncode"]) != 0:
            raise CadGenBenchInfrastructureError(
                "Failed inspecting the canonical CAD candidate: "
                + result["stderr"].decode("utf-8", errors="replace")[-4000:]
            )
        output = result["stdout"].decode("utf-8", errors="replace").strip()
        return json.loads(output) if output else None

    def _validate_live_candidate(
        self,
        path: str,
        *,
        deadline_seconds: float | None = None,
    ) -> dict[str, Any]:
        result = self._run_service_command(
            ["cadgenbench-validate", path],
            deadline=max(
                1,
                min(
                    180,
                    (
                        math.ceil(deadline_seconds)
                        if deadline_seconds is not None
                        else 180
                    ),
                ),
            ),
        )
        if result.get("timed_out"):
            raise CadGenBenchInfrastructureError(
                f"Timed out validating changed CAD candidate {path}"
            )
        if int(result["returncode"]) != 0:
            return {
                "is_valid": False,
                "error": result["stderr"].decode(
                    "utf-8",
                    errors="replace",
                )[-4000:],
            }
        try:
            payload = json.loads(result["stdout"].decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise CadGenBenchInfrastructureError(
                f"CadGenBench candidate validation returned invalid JSON: {exc}"
            ) from exc
        return payload

    def _checkpoint_changed_candidate(
        self,
        *,
        deadline_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        state = self._live_candidate_state(deadline_seconds=deadline_seconds)
        if state is None:
            return None
        path = str(state["path"])
        signature = (int(state["size"]), int(state["mtime_ns"]))
        signatures = getattr(self, "_candidate_signatures", {})
        if signatures.get(path) == signature:
            return None
        validation = self._validate_live_candidate(
            path,
            deadline_seconds=deadline_seconds,
        )
        signatures[path] = signature
        self._candidate_signatures = signatures
        if not validation.get("is_valid"):
            return {
                "status": "rejected_invalid",
                "path": path,
                "validation": validation,
            }
        entry = self._archive_valid_export(path, validation=validation)
        return {
            "status": "archived",
            **entry,
        }

    def _archive_valid_export(
        self,
        filename: str,
        *,
        validation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        container_path = Path(filename)
        if not container_path.is_absolute():
            container_path = Path("/workspace") / container_path
        data = self.get_file(str(container_path))
        if not data:
            raise CadGenBenchInfrastructureError(
                f"Successful export did not leave a readable candidate at {container_path}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if self.checkpoint_history and self.checkpoint_history[-1]["sha256"] == digest:
            return self.checkpoint_history[-1]
        index = len(self.checkpoint_history) + 1
        checkpoint_path = self.checkpoints_dir / f"{index:03d}-output.step"
        checkpoint_path.write_bytes(data)
        self.latest_valid_output.write_bytes(data)
        entry = {
            "index": index,
            "container_path": str(container_path),
            "sha256": digest,
            "size_bytes": len(data),
            "validation": validation,
        }
        self.checkpoint_history.append(entry)
        self._write_checkpoint_history()
        return entry

    def _write_checkpoint_history(self) -> None:
        payload = {
            "latest_valid_output": (
                str(self.latest_valid_output.relative_to(self.run_dir))
                if self.latest_valid_output.exists()
                else None
            ),
            "checkpoint_history": self.checkpoint_history,
            "mcp_restart_count": len(self.mcp_restart_history),
            "mcp_restart_history": self.mcp_restart_history,
            "mcp": self.mcp_telemetry,
        }
        self.checkpoint_history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _restore_latest_valid_candidate(self) -> None:
        target = self.workspace_dir / "output.step"
        alternate = self.workspace_dir / "output.stp"
        if not self.latest_valid_output.is_file():
            if target.is_file():
                target.rename(self.workspace_dir / "output.untrusted.step")
            if alternate.is_file():
                alternate.rename(self.workspace_dir / "output.untrusted.stp")
            return
        if target.is_file():
            current_hash = sha256_file(target)
            latest_hash = sha256_file(self.latest_valid_output)
            if current_hash != latest_hash:
                target.rename(self.workspace_dir / "output.mutable-final.step")
        if alternate.is_file():
            alternate.rename(self.workspace_dir / "output.mutable-final.stp")
        target.write_bytes(self.latest_valid_output.read_bytes())

    def _restore_live_workspace_checkpoint(self) -> bool:
        if not self.latest_valid_output.is_file():
            return False
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            data = self.latest_valid_output.read_bytes()
            info = tarfile.TarInfo(name=".restored-output.step")
            info.size = len(data)
            info.mode = 0o600
            info.uid = os.getuid()
            info.gid = os.getgid()
            tar.addfile(info, io.BytesIO(data))
        archive.seek(0)
        self.container.put_archive("/workspace", archive.read())
        execution = run_exec(
            client=self.client,
            container=self.container,
            command=[
                "/bin/bash",
                "-c",
                "mv -f /workspace/.restored-output.step /workspace/output.step && rm -f /workspace/output.stp",
            ],
            user=f"{os.getuid()}:{os.getgid()}",
            deadline=30,
            capture_limit=COMMAND_CAPTURE_LIMIT,
            workdir="/workspace",
            environment=self._command_environment(),
        )
        if int(execution["returncode"]) != 0:
            raise CadGenBenchInfrastructureError(
                "Failed restoring latest valid output into the live workspace: "
                + execution["stderr"].decode("utf-8", errors="replace")[-4000:]
            )
        return True

    def _clear_live_workspace_candidate(self) -> None:
        execution = run_exec(
            client=self.client,
            container=self.container,
            command=[
                "/bin/bash",
                "-c",
                "rm -f /workspace/output.step /workspace/output.stp",
            ],
            user=f"{os.getuid()}:{os.getgid()}",
            deadline=30,
            capture_limit=COMMAND_CAPTURE_LIMIT,
            workdir="/workspace",
            environment=self._command_environment(),
        )
        if int(execution["returncode"]) != 0:
            raise CadGenBenchInfrastructureError(
                "Failed clearing live workspace candidate after MCP transport loss: "
                + execution["stderr"].decode("utf-8", errors="replace")[-4000:]
            )

    def _reload_container_status(self) -> str:
        if self.container is None:
            return "missing"
        try:
            self.container.reload()
        except DockerException as exc:
            raise CadGenBenchInfrastructureError(
                f"Failed reloading CadGenBench container status: {exc}"
            ) from exc
        return str(getattr(self.container, "status", "unknown"))

    def _recreate_agent_container(self) -> dict[str, Any]:
        old_container = self.container
        old_id = getattr(old_container, "id", None)
        old_state = (
            self._capture_container_lifecycle_snapshot(
                "pre_recreate_remove",
                container=old_container,
            )
            if old_container is not None
            else None
        )
        old_status = (
            old_state["state"].get("Status")
            if old_state is not None
            else (
                "missing" if old_container is None else self._reload_container_status()
            )
        )
        self._container_shutdown_expected = True
        try:
            replacement = self._start_agent_container()
        except DockerException as exc:
            raise CadGenBenchInfrastructureError(
                f"Failed starting replacement CadGenBench container: {exc}"
            ) from exc
        if old_container is not None:
            try:
                old_container.remove(force=True)
            except DockerException as exc:
                try:
                    replacement.remove(force=True)
                except DockerException:
                    logger.exception(
                        "Failed rolling back replacement CadGenBench container"
                    )
                raise CadGenBenchInfrastructureError(
                    f"Failed removing stopped CadGenBench container {old_id}: {exc}"
                ) from exc
        self._set_current_container(replacement, reason="recreate_after_transport_loss")
        return {
            "old_container_id": old_id,
            "old_container_status": old_status,
            "old_container_state": (
                old_state["state"] if old_state is not None else None
            ),
            "new_container_id": replacement.id,
            "container_recreated": True,
        }

    def _recover_mcp_service(self, *, reason: str) -> dict[str, Any]:
        previous_telemetry = self.mcp_telemetry
        attempt_number = len(self.mcp_restart_history) + 1
        if attempt_number > MCP_RESTART_LIMIT:
            entry = {
                "timestamp": self._utc_now(),
                "reason": reason,
                "checkpoint_restored": False,
                "old_service": previous_telemetry,
                "new_service": None,
                "status": "restart_limit_exhausted",
            }
            self.mcp_restart_history.append(entry)
            self._write_checkpoint_history()
            raise CadGenBenchInfrastructureError(
                "Build123d MCP service restart limit "
                f"{MCP_RESTART_LIMIT} is exhausted."
            )
        checkpoint_restored = False
        container_recovery = {
            "old_container_id": getattr(self.container, "id", None),
            "old_container_status": None,
            "new_container_id": getattr(self.container, "id", None),
            "container_recreated": False,
        }
        try:
            container_status = self._reload_container_status()
            container_recovery["old_container_status"] = container_status
            if container_status != "running":
                snapshot = self._capture_container_lifecycle_snapshot(
                    "detected_nonrunning_container"
                )
                if snapshot is not None:
                    self._write_container_failure(
                        reason="mcp_recovery_detected_nonrunning_container",
                        state=snapshot["state"],
                    )
            container_recovery = self._recreate_agent_container()
            checkpoint_restored = self._restore_live_workspace_checkpoint()
            if not checkpoint_restored:
                self._clear_live_workspace_candidate()
            new_telemetry = self._wait_for_mcp_service()
            self._candidate_signatures = {}
            restored_state = self._live_candidate_state()
            if restored_state is not None:
                self._candidate_signatures[str(restored_state["path"])] = (
                    int(restored_state["size"]),
                    int(restored_state["mtime_ns"]),
                )
        except Exception as restart_exc:
            entry = {
                "timestamp": self._utc_now(),
                "reason": reason,
                **container_recovery,
                "checkpoint_restored": checkpoint_restored,
                "old_service": previous_telemetry,
                "new_service": None,
                "status": "restart_failed",
                "restart_exception_type": type(restart_exc).__name__,
                "restart_exception_message": str(restart_exc),
            }
            self.mcp_restart_history.append(entry)
            self._write_checkpoint_history()
            raise CadGenBenchInfrastructureError(
                "Build123d MCP service restart attempt "
                f"{attempt_number} failed: {restart_exc}"
            ) from restart_exc

        self.mcp_telemetry = new_telemetry
        entry = {
            "timestamp": self._utc_now(),
            "reason": reason,
            **container_recovery,
            "checkpoint_restored": checkpoint_restored,
            "old_service": previous_telemetry,
            "new_service": self.mcp_telemetry,
            "status": "restarted",
        }
        self.mcp_restart_history.append(entry)
        self._write_checkpoint_history()
        if checkpoint_restored:
            error = (
                "Build123d MCP service restarted. In-memory CAD state and MCP "
                "snapshots were lost, but the host restored the latest valid "
                "/workspace/output.step. Re-import it with `cad-mcp call "
                'import_cad_file --json \'{"path":"output.step",'
                '"name":"part"}\'`, validate it, and continue.'
            )
        else:
            error = (
                "Build123d MCP service restarted. "
                "In-memory CAD state and MCP snapshots were lost, and there was no protected "
                "latest-valid checkpoint to restore. Rebuild from the public inputs in /input "
                "and continue."
            )
        payload = {
            "status": "error",
            "error": error,
            "failure_class": "mcp_service_restarted",
            "reason": reason,
            "restart_attempt": attempt_number,
            "container_recreated": container_recovery["container_recreated"],
            "old_container_status": container_recovery["old_container_status"],
            "new_container_id": container_recovery["new_container_id"],
            "checkpoint_restored": checkpoint_restored,
            "mcp_restart_count": len(self.mcp_restart_history),
            "old_service": previous_telemetry,
            "new_service": self.mcp_telemetry,
        }
        return payload

    def _persist_workspace_artifacts(self) -> None:
        limits = {
            "prediction.py": 16 * 1024 * 1024,
            "output.step": ARTIFACT_FILE_LIMIT,
            "output.stp": ARTIFACT_FILE_LIMIT,
            "output.validation.json": 2 * 1024 * 1024,
        }
        render_names = ("iso.png", "front.png", "top.png", "right.png", "multiview.png")
        script = (
            "from pathlib import Path\n"
            "import shutil, stat\n"
            f"limits = {limits!r}\n"
            f"render_names = {render_names!r}\n"
            "archive = Path('/archive')\n"
            "workspace = Path('/workspace')\n"
            "for name, limit in limits.items():\n"
            "    source = workspace / name\n"
            "    if not source.exists():\n"
            "        continue\n"
            "    info = source.lstat()\n"
            "    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:\n"
            "        continue\n"
            "    shutil.copyfile(source, archive / name)\n"
            "render_root = workspace / 'renders'\n"
            "if render_root.is_dir():\n"
            "    (archive / 'renders').mkdir(parents=True, exist_ok=True)\n"
            "    for name in render_names:\n"
            "        source = render_root / name\n"
            "        if not source.exists():\n"
            "            continue\n"
            "        info = source.lstat()\n"
            "        if not stat.S_ISREG(info.st_mode) or info.st_size > 32 * 1024 * 1024:\n"
            "            continue\n"
            "        shutil.copyfile(source, archive / 'renders' / name)\n"
        )
        result = self.client.containers.run(
            self.image,
            command=["python", "-c", script],
            network_mode="none",
            read_only=True,
            user=f"{os.getuid()}:{os.getgid()}",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit="1g",
            nano_cpus=1_000_000_000,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=64m,mode=1777"},
            volumes={
                self.workspace_volume.name: {"bind": "/workspace", "mode": "ro"},
                str(self.workspace_dir): {"bind": "/archive", "mode": "rw"},
            },
            remove=True,
        )
        if result:
            raise RuntimeError(result.decode("utf-8", errors="replace")[-4000:])

    def _freeze_workspace(self) -> None:
        if self._frozen:
            return
        snapshot_error: Exception | None = None
        cleanup_errors: list[DockerException] = []
        try:
            self._write_checkpoint_history()
            self._capture_container_lifecycle_snapshot("pre_freeze")
            self.container.reload()
            if self.container.status == "running":
                self.container.pause()
            self._persist_workspace_artifacts()
        except (DockerException, OSError, RuntimeError, tarfile.TarError) as exc:
            snapshot_error = exc
        finally:
            try:
                if self.container is not None:
                    self._container_shutdown_expected = True
                    self._remove_current_container(lifecycle_reason="pre_freeze_remove")
            except DockerException as exc:
                logger.exception("Failed to remove CadGenBench agent container")
                cleanup_errors.append(exc)
            try:
                self.workspace_volume.remove(force=True)
                self._volume_removed = True
            except DockerException as exc:
                logger.exception("Failed to remove CadGenBench workspace volume")
                cleanup_errors.append(exc)
            self._frozen = True
        if snapshot_error is not None:
            raise snapshot_error
        if cleanup_errors:
            raise CadGenBenchInfrastructureError(
                "; ".join(str(error) for error in cleanup_errors)
            )

    def _candidate_path(self) -> Path | None:
        for name in ("output.step", "output.stp"):
            candidate = self.workspace_dir / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        return None

    def _run_public_evaluation_job(
        self,
        *,
        deadline_seconds: float | None = None,
    ) -> dict[str, Any]:
        public_dir = self.run_dir / "evaluation" / "public"
        public_dir.mkdir(parents=True, exist_ok=True)
        candidate = self._candidate_path()
        if candidate is None:
            result = {
                "status": "missing_output",
                "error": "Missing /workspace/output.step",
                "validation": None,
                "renders": None,
            }
            (public_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return result
        container = self.client.containers.run(
            self.image,
            command=[
                "python",
                "-m",
                "cuawright.environments.cadgenbench.public_eval_job",
                "--candidate",
                f"/workspace/{candidate.name}",
                "--output-dir",
                "/out",
            ],
            detach=True,
            network_mode="none",
            read_only=True,
            user=f"{os.getuid()}:{os.getgid()}",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            pids_limit=512,
            mem_limit="8g",
            nano_cpus=4_000_000_000,
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=2g,mode=1777",
                "/home/agent": (
                    "rw,nosuid,size=256m,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
            },
            volumes={
                str(self.workspace_dir): {"bind": "/workspace", "mode": "ro"},
                str(public_dir): {"bind": "/out", "mode": "rw"},
            },
            labels={
                "cuawright.component": "cadgenbench-public-eval",
                "cuawright.fixture_id": self.task.fixture_id,
            },
        )
        cleanup_error: DockerException | None = None
        try:
            wait_timeout = 720
            if deadline_seconds is not None:
                if deadline_seconds <= 0:
                    raise CadGenBenchInfrastructureError(
                        "CadGenBench public evaluation exceeded the remaining task wall time"
                    )
                wait_timeout = max(1, math.ceil(deadline_seconds))
            status = container.wait(timeout=wait_timeout)
            logs = container.logs(stdout=True, stderr=True)
            (public_dir / "job.log").write_bytes(logs)
            if int(status.get("StatusCode", -1)) != 0:
                raise CadGenBenchInfrastructureError(
                    "CadGenBench public evaluation exited unsuccessfully: "
                    + logs.decode("utf-8", errors="replace")[-4000:]
                )
        finally:
            try:
                container.remove(force=True)
            except DockerException as exc:
                cleanup_error = exc
        result_path = public_dir / "result.json"
        if not result_path.is_file():
            raise CadGenBenchInfrastructureError(
                "CadGenBench public evaluation produced no result.json"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if cleanup_error is not None:
            raise CadGenBenchInfrastructureError(
                f"Failed to clean up public evaluation container: {cleanup_error}"
            )
        return result

    def _artifact_hashes(self, base_dir: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        if not base_dir.exists():
            return hashes
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.stat().st_size > ARTIFACT_FILE_LIMIT:
                continue
            hashes[str(path.relative_to(self.run_dir))] = sha256_file(path)
        return hashes

    def evaluate(
        self,
        *,
        deadline_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._freeze_workspace()
        self._restore_latest_valid_candidate()
        public_result = self._run_public_evaluation_job(
            deadline_seconds=deadline_seconds,
        )
        prediction = self.workspace_dir / "prediction.py"
        candidate = self._candidate_path()
        evaluation_dir = self.run_dir / "evaluation"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation = {
            "score": None,
            "status": public_result.get("status"),
            "error": public_result.get("error"),
            "benchmark_commit": self.benchmark_commit,
            "container_image_id": self.image_id,
            "control_client_sha256": dict(CLIENT_SHA256),
            "prediction_sha256": (
                sha256_file(prediction) if prediction.is_file() else None
            ),
            "output_step_sha256": sha256_file(candidate) if candidate else None,
            "local_validation": public_result.get("validation"),
            "local_render_artifacts": public_result.get("renders"),
            "checkpoint_history": self.checkpoint_history,
            "mcp_restart_history": self.mcp_restart_history,
            "latest_valid_output_sha256": (
                sha256_file(self.latest_valid_output)
                if self.latest_valid_output.is_file()
                else None
            ),
            "mcp": json.loads(self.checkpoint_history_path.read_text(encoding="utf-8")),
            "container_telemetry": self._container_observability_summary(),
            "artifact_hashes": {
                **self._artifact_hashes(self.workspace_dir),
                **self._artifact_hashes(self.latest_valid_dir),
                **self._artifact_hashes(self.run_dir / "evaluation" / "public"),
            },
            **self.provenance,
        }
        (evaluation_dir / "result.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return evaluation

    def close(self) -> None:
        try:
            if not self._frozen:
                self._freeze_workspace()
            if self._frozen and not self._removed:
                try:
                    self._container_shutdown_expected = True
                    self._remove_current_container(lifecycle_reason="pre_close_remove")
                except DockerException as exc:
                    raise CadGenBenchInfrastructureError(
                        "Failed to remove CadGenBench agent container"
                    ) from exc
            if self._frozen and not self._volume_removed:
                try:
                    self.workspace_volume.remove(force=True)
                    self._volume_removed = True
                except DockerException as exc:
                    raise CadGenBenchInfrastructureError(
                        "Failed to remove CadGenBench workspace volume"
                    ) from exc
        finally:
            self._stop_container_monitor()
            self.client.close()

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
import uuid
from pathlib import Path
from typing import Any

from docker.errors import DockerException

from cuawright.environments.benchcad.docker_exec import run_exec
from cuawright.environments.benchcad.errors import BenchCADInfrastructureError
from cuawright.environments.benchcad.image import validate_image
from cuawright.environments.benchcad.scoring import score_prediction_with_retry
from cuawright.environments.benchcad.task import (
    Vision2CodeManifest,
    Vision2CodeTask,
    snapshot_task,
    task_provenance,
)
from cuawright.environments.control_clients import (
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
)

COMMAND_CAPTURE_LIMIT = 1_000_000
FILE_READ_LIMIT = 128 * 1024 * 1024
WORKSPACE_SIZE = 2 * 1024 * 1024 * 1024

logger = logging.getLogger("cuawright.environments.benchcad.environment")

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


class BenchCADEnvironment:
    def __init__(
        self,
        *,
        task: Vision2CodeTask,
        manifest: Vision2CodeManifest,
        run_dir: Path,
        image: str,
        benchcad_repo: Path,
        benchmark_commit: str,
        defer_evaluation: bool = False,
    ) -> None:
        import docker

        self.task = snapshot_task(task, run_dir)
        self.manifest = manifest
        self.provenance = task_provenance(manifest, self.task)
        self.run_dir = run_dir
        self.image_ref = image
        self.benchcad_repo = benchcad_repo
        self.benchmark_commit = benchmark_commit
        self.defer_evaluation = defer_evaluation
        self.input_dir = run_dir / "sandbox-input"
        self.workspace_dir = run_dir / "workspace"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.input_dir, 0o700)
        os.chmod(self.workspace_dir, 0o700)

        self.client = docker.from_env()
        self.image_id = validate_image(
            self.client,
            image,
            benchmark_commit,
        )
        self.image = self.image_id
        host_user = f"{os.getuid()}:{os.getgid()}"
        self.workspace_volume = None
        self.container = None
        try:
            self.workspace_volume = self.client.volumes.create(
                name=(
                    f"cuawright-benchcad-{task.record_id[:32]}-"
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
                    "cuawright.component": "benchcad-workspace",
                    "cuawright.record_id": task.record_id,
                },
            )
            self.container = self.client.containers.run(
                self.image,
                command=["sleep", "infinity"],
                detach=True,
                network_mode="none",
                read_only=True,
                user=host_user,
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
                    str(self.input_dir): {"bind": "/input", "mode": "ro"},
                    self.workspace_volume.name: {
                        "bind": "/workspace",
                        "mode": "rw",
                    },
                },
                labels={
                    "cuawright.component": "benchcad-agent",
                    "cuawright.record_id": task.record_id,
                },
            )
            self._verify_control_clients()
        except (BenchCADInfrastructureError, DockerException):
            if self.container is not None:
                try:
                    self.container.remove(force=True)
                except DockerException:
                    logger.exception("Failed to roll back BenchCAD agent container")
            if self.workspace_volume is not None:
                try:
                    self.workspace_volume.remove(force=True)
                except DockerException:
                    logger.exception("Failed to roll back BenchCAD workspace volume")
            self.client.close()
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

    def _verify_control_clients(self) -> None:
        expected = dict(CLIENT_SHA256)
        try:
            result = run_exec(
                client=self.client,
                container=self.container,
                command=[
                    "/opt/benchcad-venv/bin/python",
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
            raise BenchCADInfrastructureError(
                "BenchCAD control-client preflight failed"
            ) from exc
        if (
            result["timed_out"]
            or result["container_killed"]
            or int(result["returncode"]) != 0
        ):
            detail = result["stderr"].decode("utf-8", errors="replace").strip()
            raise BenchCADInfrastructureError(
                "BenchCAD control-client preflight failed"
                + (f": {detail[-2000:]}" if detail else "")
            )
        try:
            observed = json.loads(result["stdout"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchCADInfrastructureError(
                "BenchCAD control-client preflight returned invalid output"
            ) from exc
        if observed != expected:
            raise BenchCADInfrastructureError(
                "BenchCAD control-client preflight returned unexpected hashes"
            )

    def run_command(self, script: str, timeout: int) -> dict[str, Any]:
        command = [
            "/usr/bin/timeout",
            "--verbose",
            "--signal=TERM",
            "--kill-after=5s",
            f"{timeout}s",
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
                deadline=timeout + 15,
                capture_limit=COMMAND_CAPTURE_LIMIT,
                workdir="/workspace",
                environment={
                    "HOME": "/home/agent",
                    "PATH": ("/opt/benchcad-venv/bin:/usr/local/bin:/usr/bin:/bin"),
                    "PYTHONPATH": "/opt/benchcad-tools",
                },
            )
        except DockerException as exc:
            return {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": str(exc),
                "infrastructure_error": True,
            }
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
                f"Command exceeded {timeout} seconds; its process group was "
                f"terminated.\n{error}"
            ).strip()
        if execution["container_killed"]:
            payload["infrastructure_error"] = True
        self.is_environment_used = True
        return payload

    def get_file(self, path: str) -> bytes | None:
        try:
            stream, _ = self.container.get_archive(path)
        except DockerException:
            return self._get_file_via_exec(path)
        archive = io.BytesIO()
        for chunk in stream:
            if archive.tell() + len(chunk) > FILE_READ_LIMIT:
                return None
            archive.write(chunk)
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            if len(members) != 1:
                return None
            extracted = tar.extractfile(members[0])
            return extracted.read() if extracted is not None else None

    def _get_file_via_exec(self, path: str) -> bytes | None:
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
                    if len(data) + len(stdout) > FILE_READ_LIMIT:
                        return None
                    data.extend(stdout)
            inspection = self.client.api.exec_inspect(execution["Id"])
        except DockerException:
            return None
        if int(inspection["ExitCode"]) != 0:
            return None
        return bytes(data)

    def _persist_workspace_artifacts(self) -> None:
        limits = {
            "prediction.py": 16 * 1024 * 1024,
            "candidate.step": 256 * 1024 * 1024,
            "candidate.png": 32 * 1024 * 1024,
        }
        script = (
            "from pathlib import Path\n"
            "import shutil, stat\n"
            f"limits = {limits!r}\n"
            "for name, limit in limits.items():\n"
            "    source = Path('/workspace') / name\n"
            "    if not source.exists():\n"
            "        continue\n"
            "    info = source.lstat()\n"
            "    if not stat.S_ISREG(info.st_mode):\n"
            "        continue\n"
            "    if info.st_size > limit:\n"
            "        continue\n"
            "    shutil.copyfile(source, Path('/archive') / name)\n"
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
                self.workspace_volume.name: {
                    "bind": "/workspace",
                    "mode": "ro",
                },
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
            self.container.reload()
            if self.container.status == "running":
                self.container.pause()
            self._persist_workspace_artifacts()
        except (DockerException, OSError, RuntimeError, tarfile.TarError) as exc:
            snapshot_error = exc
        finally:
            try:
                self.container.remove(force=True)
                self._removed = True
            except DockerException as exc:
                logger.exception("Failed to remove BenchCAD agent container")
                cleanup_errors.append(exc)
            try:
                self.workspace_volume.remove(force=True)
                self._volume_removed = True
            except DockerException as exc:
                logger.exception("Failed to remove BenchCAD workspace volume")
                cleanup_errors.append(exc)
            self._frozen = True
        if snapshot_error is not None:
            raise snapshot_error
        if cleanup_errors:
            raise BenchCADInfrastructureError(
                "; ".join(str(error) for error in cleanup_errors)
            )

    def evaluate(self) -> dict[str, Any]:
        self._freeze_workspace()
        prediction = self.workspace_dir / "prediction.py"
        prediction_sha256 = (
            hashlib.sha256(prediction.read_bytes()).hexdigest()
            if prediction.is_file()
            else None
        )
        if self.defer_evaluation:
            return {
                "score": None,
                "iou": None,
                "status": "pending_offline_score",
                "error": None,
                "benchmark_commit": self.benchmark_commit,
                "container_image_id": self.image_id,
                "control_client_sha256": dict(CLIENT_SHA256),
                "prediction_sha256": prediction_sha256,
                **self.provenance,
            }
        result = score_prediction_with_retry(
            task=self.task,
            workspace_dir=self.workspace_dir,
            output_dir=self.run_dir / "evaluation" / "scorer",
            image=self.image,
            benchcad_repo=self.benchcad_repo,
            benchmark_commit=self.benchmark_commit,
        )
        if result.get("ground_truth_sha256") != self.provenance["ground_truth_sha256"]:
            raise BenchCADInfrastructureError(
                "Scorer ground-truth snapshot differs from generation provenance"
            )
        result["prediction_sha256"] = prediction_sha256
        result["control_client_sha256"] = dict(CLIENT_SHA256)
        result.update(self.provenance)
        return result

    def close(self) -> None:
        try:
            if not self._frozen:
                self._freeze_workspace()
            elif not self._removed:
                try:
                    self.container.remove(force=True)
                    self._removed = True
                except DockerException as exc:
                    raise BenchCADInfrastructureError(
                        "Failed to remove BenchCAD agent container"
                    ) from exc
            if not self._volume_removed:
                try:
                    self.workspace_volume.remove(force=True)
                    self._volume_removed = True
                except DockerException as exc:
                    raise BenchCADInfrastructureError(
                        "Failed to remove BenchCAD workspace volume"
                    ) from exc
        finally:
            self.client.close()

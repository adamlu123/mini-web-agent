from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
import threading
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from cuawright.environments.base import CommandObservation, CommandResult
from cuawright.environments.control_clients import CLIENT_INSTALL_ROOT
from cuawright.environments.desktop.plugin import (
    DesktopCommandPlugin,
    GuestWorkspace,
)
from cuawright.environments.odysseys.task import OdysseysTask
from cuawright.environments.odysseys.trajectory import (
    OdysseysTrajectoryContractError,
    validate_trajectory_contract,
)


ODYSSEYS_WORKSPACE = "/home/user/.agent/odysseys"
ODYSSEYS_SCREENSHOT_DIR = f"{ODYSSEYS_WORKSPACE}/screenshots"
ODYSSEYS_FINAL_RESPONSE_PATH = f"{ODYSSEYS_WORKSPACE}/final_response.md"
MAX_FINAL_RESPONSE_BYTES = 64 * 1024
MAX_TRAJECTORY_COMMAND_CHARACTERS = 16_000
_SUBMIT_ARGV = ["python", f"{CLIENT_INSTALL_ROOT}/submit.py"]


class EventWriter(Protocol):
    def event(self, event: str, payload: dict[str, Any]) -> None: ...


class OdysseysTrajectoryError(RuntimeError):
    """The image-based Odysseys judge trajectory could not be persisted."""


class OdysseysBenchmarkPlugin(DesktopCommandPlugin):
    """Desktop environment plus provenance-complete Odysseys evidence capture."""

    def __init__(
        self,
        desktop_env: Any,
        execute_command: Any,
        *,
        workspace: GuestWorkspace,
        task: OdysseysTask,
        judge_dir: str | Path,
        judge_script: str | Path,
        source_identity: Mapping[str, Any],
        run_store: EventWriter | None = None,
        bridge_timeout: int = 90,
    ) -> None:
        super().__init__(
            desktop_env,
            execute_command,
            workspace=workspace,
            bridge_timeout=bridge_timeout,
            bridge_error_label="Odysseys guest bridge",
        )
        self.task = task
        self.judge_dir = Path(judge_dir).resolve()
        self.external_judge_script = Path(judge_script).expanduser().resolve()
        self.source_identity = dict(source_identity)
        self.run_store = run_store
        self.screenshot_dir = self.judge_dir / "screenshots"
        self.trajectory_path = self.judge_dir / "traj.jsonl"
        self.inputs_dir = self.judge_dir / "inputs"
        self._step = 0
        self._submission_classification: str | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._evaluated = False
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_inputs()

    @property
    def action_steps(self) -> int:
        return self._step

    @property
    def submission_classification(self) -> str | None:
        return self._submission_classification

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        if _is_submit_command(command):
            error = self._final_response_error()
            if error is not None:
                return CommandResult(2, stderr=error)
        return super().run_command(command, timeout)

    def observe_command(self, observation: CommandObservation) -> None:
        classification = None
        if observation.result.completed and observation.operation in {
            "submit",
            "forced_submit",
        }:
            if observation.submission_forced is None:
                raise OdysseysTrajectoryError(
                    "completed submission observation omitted authoritative finality"
                )
            classification = "forced" if observation.submission_forced else "accepted"
        self._record_step(observation, classification=classification)

    def evaluate(self) -> Mapping[str, Any]:
        if self._evaluated:
            raise RuntimeError("Odysseys trajectory has already been finalized")
        self._evaluated = True
        final_response, response_error = self._read_final_response()
        if final_response is not None:
            _write_immutable(
                self.judge_dir / "final-response.md",
                (final_response + "\n").encode("utf-8"),
            )
        trajectory_error: str | None = None
        screenshot_paths: tuple[str, ...] = ()
        try:
            screenshot_paths = self._canonical_screenshot_paths()
            trajectory_bytes = (
                self.trajectory_path.read_bytes()
                if self.trajectory_path.is_file()
                else b""
            )
            validate_trajectory_contract(
                trajectory_bytes,
                action_count=self.action_steps,
                screenshot_count=len(screenshot_paths),
                screenshot_paths=screenshot_paths,
                submission_classification=self.submission_classification,
                expected_final_response=final_response,
                require_final=self.submission_classification in {"accepted", "forced"},
            )
        except (OSError, OdysseysTrajectoryContractError) as exc:
            trajectory_error = str(exc)
        status = (
            "ready_for_external_judge"
            if self.action_steps > 0
            and self.submission_classification in {"accepted", "forced"}
            and response_error is None
            and trajectory_error is None
            else "incomplete_judge_evidence"
        )
        evidence_errors = [
            value for value in (response_error, trajectory_error) if value is not None
        ]
        _write_immutable(self.judge_dir / "result.txt", b"0.0\n")
        artifact_hashes = self._judge_artifact_hashes(screenshot_paths)
        manifest_path = self.judge_dir / "judge-input.json"
        manifest = {
            "schema_version": 3,
            "benchmark": "odysseys",
            "task_id": self.task.task_id,
            "status": status,
            "score": None,
            "action_steps": self.action_steps,
            "screenshots": len(screenshot_paths),
            "trajectory": str(self.trajectory_path),
            "judge_run_dir": str(self.judge_dir),
            "submission_classification": self.submission_classification,
            "final_response": final_response,
            "evidence_error": "; ".join(evidence_errors) or None,
            "artifact_hashes": artifact_hashes,
        }
        _write_json_atomic(manifest_path, manifest)
        manifest_path.chmod(0o444)
        manifest_sha256 = _sha256_file(manifest_path)
        _write_immutable(
            self.judge_dir / "judge-input.sha256",
            f"{manifest_sha256}  judge-input.json\n".encode("ascii"),
        )
        judge_command = (
            [
                sys.executable,
                "-m",
                "cuawright.run.odysseys_judge",
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                manifest_sha256,
            ]
            if status == "ready_for_external_judge"
            else None
        )
        payload = {
            **manifest,
            "judge_manifest_sha256": manifest_sha256,
            "judge_command": judge_command,
        }
        if self.run_store is not None:
            self.run_store.event(
                "odysseys_trajectory_finalized",
                {
                    "task_id": self.task.task_id,
                    "status": status,
                    "action_steps": self.action_steps,
                    "judge_run_dir": str(self.judge_dir),
                    "submission_classification": self.submission_classification,
                    "final_response_present": final_response is not None,
                    "evidence_error": manifest["evidence_error"],
                },
            )
        return payload

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.desktop_env.close()

    def _record_step(
        self,
        observation: CommandObservation,
        *,
        classification: str | None,
    ) -> None:
        with self._lock:
            step = self._step + 1
            screenshot = self._get_screenshot()
            extension, media_type = _image_type(screenshot)
            filename = f"step_{step:04d}{extension}"
            relative = Path("screenshots") / filename
            guest_path = f"{ODYSSEYS_SCREENSHOT_DIR}/{filename}"
            final_response, _ = self._read_final_response()
            final = classification is not None
            row = {
                "step_num": step,
                "action_timestamp": datetime.now(UTC).strftime("%Y%m%d@%H%M%S%f"),
                "action": observation.command[:MAX_TRAJECTORY_COMMAND_CHARACTERS],
                "response": final_response or "",
                "reward": 0,
                "done": final,
                "info": {
                    "role": observation.role,
                    "stage": observation.stage,
                    "operation": observation.operation,
                    "synthetic": observation.synthetic,
                    "exit_code": observation.result.exit_code,
                    "completed": observation.result.completed,
                    "submission_forced": observation.submission_forced,
                    "submission_classification": classification,
                    "screenshot_sha256": hashlib.sha256(screenshot).hexdigest(),
                    "guest_screenshot": guest_path,
                },
                "screenshot_file": relative.as_posix(),
                "final": final,
            }
            try:
                self.write_control_file_atomic(
                    guest_path,
                    screenshot,
                    workspace=self.logical_workspace,
                )
                _write_bytes_atomic(self.judge_dir / relative, screenshot)
                _append_jsonl(self.trajectory_path, row)
            except Exception as exc:
                raise OdysseysTrajectoryError(
                    f"Could not persist trajectory screenshot {step}: {exc}"
                ) from exc
            self._step = step
            if classification is not None:
                self._submission_classification = classification
            if self.run_store is not None:
                self.run_store.event(
                    "odysseys_trajectory_step",
                    {
                        "task_id": self.task.task_id,
                        "step_num": step,
                        "role": observation.role,
                        "stage": observation.stage,
                        "operation": observation.operation,
                        "exit_code": observation.result.exit_code,
                        "completed": observation.result.completed,
                        "submission_forced": observation.submission_forced,
                        "screenshot_file": relative.as_posix(),
                        "guest_screenshot": guest_path,
                        "media_type": media_type,
                        "screenshot_sha256": row["info"]["screenshot_sha256"],
                        "final_response_present": final_response is not None,
                        "submission_classification": classification,
                        "final": final,
                    },
                )

    def _get_screenshot(self) -> bytes:
        get_screenshot = getattr(self.controller, "get_screenshot", None)
        if not callable(get_screenshot):
            raise OdysseysTrajectoryError(
                "Desktop controller does not provide get_screenshot()"
            )
        screenshot = get_screenshot()
        if not isinstance(screenshot, bytes) or not screenshot:
            raise OdysseysTrajectoryError("Desktop controller returned no screenshot")
        return screenshot

    def _final_response_error(self) -> str | None:
        _, error = self._read_final_response()
        if error is None:
            return None
        return (
            "Odysseys submission requires a final user response at "
            f"{ODYSSEYS_FINAL_RESPONSE_PATH}: {error}"
        )

    def _read_final_response(self) -> tuple[str | None, str | None]:
        try:
            data = self.read_control_file(
                ODYSSEYS_FINAL_RESPONSE_PATH,
                workspace=self.logical_workspace,
                max_bytes=MAX_FINAL_RESPONSE_BYTES + 1,
            )
        except FileNotFoundError:
            return None, "file does not exist"
        except OSError as exc:
            return None, f"file is not safely readable ({type(exc).__name__})"
        if len(data) > MAX_FINAL_RESPONSE_BYTES:
            return None, f"file exceeds {MAX_FINAL_RESPONSE_BYTES} bytes"
        try:
            response = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None, "file is not valid UTF-8"
        if not response:
            return None, "file is empty"
        return response, None

    def _snapshot_inputs(self) -> None:
        judge_bytes = self.external_judge_script.read_bytes()
        expected_dataset = self.source_identity.get("dataset_sha256")
        expected_judge = self.source_identity.get("judge_sha256")
        observed_judge = hashlib.sha256(judge_bytes).hexdigest()
        if expected_dataset != self.task.dataset_sha256:
            raise OdysseysTrajectoryError(
                "Odysseys dataset changed between source validation and snapshot"
            )
        if expected_judge != observed_judge:
            raise OdysseysTrajectoryError(
                "Odysseys judge changed between source validation and snapshot"
            )
        _write_immutable(
            self.inputs_dir / "run_full_trajectory_per_rubric.py",
            judge_bytes,
        )
        _write_immutable(self.inputs_dir / "odysseys.json", self.task.dataset_bytes)
        _write_immutable(
            self.inputs_dir / f"{self.task.task_id}.json",
            self.task.example_bytes,
        )
        _write_immutable(
            self.inputs_dir / "selected-task.json",
            (
                json.dumps(
                    self.task.task_record,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _write_json_atomic(
            self.judge_dir / "config.json",
            {
                "task_id": self.task.task_id,
                "task": self.task.instruction,
                "instruction": self.task.instruction,
                "website": self.task.website,
                "level": self.task.level,
                "reference_length": self.task.reference_length,
                "rubrics": self.task.rubrics,
                "source_identity": self.source_identity,
                "copied_inputs": {
                    "task_source": "inputs/odysseys.json",
                    "task_source_sha256": self.task.dataset_sha256,
                    "source_example": f"inputs/{self.task.task_id}.json",
                    "source_example_sha256": self.task.example_sha256,
                    "judge_script": "inputs/run_full_trajectory_per_rubric.py",
                    "judge_script_sha256": observed_judge,
                },
            },
        )

    def _canonical_screenshot_paths(self) -> tuple[str, ...]:
        paths: list[str] = []
        for path in sorted(self.screenshot_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise OdysseysTrajectoryContractError(
                    f"non-regular screenshot artifact: {path.name}"
                )
            paths.append(path.relative_to(self.judge_dir).as_posix())
        return tuple(paths)

    def _judge_artifact_hashes(
        self,
        screenshot_paths: tuple[str, ...],
    ) -> dict[str, str]:
        paths = [
            self.judge_dir / "config.json",
            self.judge_dir / "result.txt",
            self.inputs_dir / "run_full_trajectory_per_rubric.py",
            self.inputs_dir / "odysseys.json",
            self.inputs_dir / f"{self.task.task_id}.json",
            self.inputs_dir / "selected-task.json",
            *(self.judge_dir / path for path in screenshot_paths),
        ]
        if self.trajectory_path.is_file():
            paths.append(self.trajectory_path)
        final_response = self.judge_dir / "final-response.md"
        if final_response.is_file():
            paths.append(final_response)
        return {
            path.relative_to(self.judge_dir).as_posix(): _sha256_file(path)
            for path in paths
        }


def _is_submit_command(command: str) -> bool:
    try:
        return shlex.split(command, comments=False, posix=True) == _SUBMIT_ARGV
    except (TypeError, ValueError):
        return False


def _image_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif", "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    raise OdysseysTrajectoryError("Desktop screenshot has an unsupported image format")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(
        path,
        (
            json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n"
        ).encode("utf-8"),
    )


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise OdysseysTrajectoryError(f"immutable artifact already differs: {path}")
        return
    _write_bytes_atomic(path, data)
    path.chmod(0o444)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

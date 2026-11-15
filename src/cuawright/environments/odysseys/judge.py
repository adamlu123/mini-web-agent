from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from cuawright.environments.odysseys.trajectory import validate_trajectory_contract

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_JUDGE_SCRIPT = "inputs/run_full_trajectory_per_rubric.py"
_TASK_SOURCE = "inputs/odysseys.json"
_OUTPUT = "eval_results_full_traj_per_rubric.json"
_RUNNER_LOG = ".judge-process.log"
_RUNNER_PID = ".judge-process.pid"
_RUNNER_EXIT = ".judge-process.exit"
_MAX_JUDGE_OUTPUT_BYTES = 64 * 1024 * 1024
_MANIFEST_KEYS = {
    "schema_version",
    "benchmark",
    "task_id",
    "status",
    "score",
    "action_steps",
    "screenshots",
    "trajectory",
    "judge_run_dir",
    "submission_classification",
    "final_response",
    "evidence_error",
    "artifact_hashes",
}


@dataclass(slots=True)
class VerifiedJudgeInputs:
    manifest: dict[str, Any]
    original_root_fd: int
    snapshot_root_fd: int
    output_fd: int
    snapshot_name: str

    @property
    def snapshot_root(self) -> str:
        return f"/proc/self/fd/{self.snapshot_root_fd}"

    def argv(self) -> list[str]:
        root = self.snapshot_root
        return [
            sys.executable,
            f"{root}/{_JUDGE_SCRIPT}",
            "--model",
            "gemini-3.1-flash-lite-preview",
            "--runs-dir",
            root,
            "--task-source-json",
            f"{root}/{_TASK_SOURCE}",
            "--output",
            f"/proc/self/fd/{self.output_fd}",
        ]

    def publish_output(self) -> None:
        _seal_judge_output(self.output_fd)
        data = _read_limited_regular_fd(
            self.output_fd,
            _MAX_JUDGE_OUTPUT_BYTES,
        )
        _validate_judge_output(data)
        temporary = f".{_OUTPUT}.{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.original_root_fd,
        )
        try:
            _write_fd(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.rename(
                temporary,
                _OUTPUT,
                src_dir_fd=self.original_root_fd,
                dst_dir_fd=self.original_root_fd,
            )
            os.fsync(self.original_root_fd)
        finally:
            try:
                os.unlink(temporary, dir_fd=self.original_root_fd)
            except FileNotFoundError:
                pass

    def close(self) -> None:
        snapshot = Path(f"/proc/self/fd/{self.original_root_fd}") / self.snapshot_name
        try:
            _make_tree_writable(snapshot)
            shutil.rmtree(snapshot)
        except FileNotFoundError:
            pass
        for descriptor in (
            self.output_fd,
            self.snapshot_root_fd,
            self.original_root_fd,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass


def verify_judge_inputs(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    with open_verified_judge(
        manifest_path,
        expected_manifest_sha256,
    ) as verified:
        return dict(verified.manifest)


@contextmanager
def open_verified_judge(
    manifest_path: str | Path,
    expected_manifest_sha256: str,
) -> Iterator[VerifiedJudgeInputs]:
    if not _SHA256_PATTERN.fullmatch(expected_manifest_sha256):
        raise RuntimeError("Expected Odysseys manifest digest must be a SHA-256")
    path = Path(manifest_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.name != "judge-input.json":
        raise RuntimeError("Odysseys judge manifest must be named judge-input.json")
    _require_no_symlink_components(path)
    root = path.parent.resolve(strict=True)
    original_root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    snapshot_root_fd = -1
    output_fd = -1
    snapshot_name = f".verified-judge-{secrets.token_hex(16)}"
    try:
        manifest_fd = _open_relative_file(original_root_fd, path.name)
        try:
            manifest_bytes = _read_fd(manifest_fd)
        finally:
            os.close(manifest_fd)
        observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                "Odysseys judge manifest digest mismatch: expected "
                f"{expected_manifest_sha256}, got {observed_manifest_sha256}"
            )
        manifest = _validated_manifest(manifest_bytes)
        opened: dict[str, int] = {}
        try:
            for relative, expected in manifest["artifact_hashes"].items():
                descriptor = _open_relative_file(original_root_fd, relative)
                opened[relative] = descriptor
                observed = _sha256_fd(descriptor)
                if observed != expected:
                    raise RuntimeError(
                        "Odysseys judge inputs changed after finalization: " + relative
                    )
            validate_trajectory_contract(
                _read_fd(opened["traj.jsonl"]),
                action_count=manifest["action_steps"],
                screenshot_count=manifest["screenshots"],
                screenshot_paths=(
                    path
                    for path in manifest["artifact_hashes"]
                    if path.startswith("screenshots/")
                ),
                submission_classification=manifest["submission_classification"],
                expected_final_response=_validated_final_response(
                    _read_fd(opened["final-response.md"]),
                    manifest["final_response"],
                ),
                require_final=True,
            )
            os.mkdir(snapshot_name, mode=0o700, dir_fd=original_root_fd)
            snapshot_root_fd = os.open(
                snapshot_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=original_root_fd,
            )
            _copy_verified_snapshot(
                snapshot_root_fd,
                opened,
                manifest["artifact_hashes"],
            )
            output_fd = _new_judge_output_fd()
        finally:
            for descriptor in opened.values():
                os.close(descriptor)
        os.set_inheritable(snapshot_root_fd, True)
        verified = VerifiedJudgeInputs(
            manifest=manifest,
            original_root_fd=original_root_fd,
            snapshot_root_fd=snapshot_root_fd,
            output_fd=output_fd,
            snapshot_name=snapshot_name,
        )
        try:
            yield verified
        finally:
            verified.close()
        original_root_fd = -1
        snapshot_root_fd = -1
        output_fd = -1
    finally:
        if original_root_fd >= 0:
            snapshot = Path(f"/proc/self/fd/{original_root_fd}") / snapshot_name
            try:
                _make_tree_writable(snapshot)
                shutil.rmtree(snapshot)
            except FileNotFoundError:
                pass
        for descriptor in (output_fd, snapshot_root_fd, original_root_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


def _validated_manifest(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Odysseys judge manifest is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise RuntimeError(
            "Odysseys judge manifest fields do not match the fixed schema"
        )
    if value.get("schema_version") != 3 or value.get("benchmark") != "odysseys":
        raise RuntimeError("Odysseys judge manifest identity is invalid")
    if value.get("status") != "ready_for_external_judge":
        raise RuntimeError("Odysseys trajectory is not ready for external judging")
    hashes = value.get("artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise RuntimeError("Odysseys judge manifest has no artifact hashes")
    required = {
        _JUDGE_SCRIPT,
        _TASK_SOURCE,
        "config.json",
        "traj.jsonl",
        "result.txt",
        "final-response.md",
    }
    if not required <= hashes.keys():
        raise RuntimeError("Odysseys judge manifest omits fixed snapshotted inputs")
    normalized: dict[str, str] = {}
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("Odysseys judge manifest contains an invalid hash entry")
        safe = PurePosixPath(relative)
        if safe.is_absolute() or not safe.parts or ".." in safe.parts:
            raise RuntimeError("Odysseys judge manifest contains an unsafe hash path")
        if not _SHA256_PATTERN.fullmatch(expected):
            raise RuntimeError("Odysseys judge manifest contains an invalid SHA-256")
        normalized[safe.as_posix()] = expected
    value["artifact_hashes"] = normalized
    return value


def _copy_verified_snapshot(
    snapshot_root_fd: int,
    opened: Mapping[str, int],
    expected_hashes: Mapping[str, str],
) -> None:
    root = Path(f"/proc/self/fd/{snapshot_root_fd}")
    directories = {root}
    for relative, descriptor in opened.items():
        target = root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        directories.update((target.parent, *target.parent.parents))
        output = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                _write_fd(output, chunk)
            os.fsync(output)
            os.fchmod(output, 0o400)
        finally:
            os.close(output)
        if digest.hexdigest() != expected_hashes[relative]:
            raise RuntimeError(
                "Odysseys judge input changed while creating snapshot: " + relative
            )
    for name in (_RUNNER_LOG, _RUNNER_PID, _RUNNER_EXIT):
        descriptor = os.open(
            root / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        os.close(descriptor)
    for directory in sorted(
        (path for path in directories if path == root or root in path.parents),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)


def _make_tree_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    root.chmod(0o700)


def _require_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(
                f"Odysseys judge path contains a symlink component: {current}"
            )


def _open_relative_file(root_fd: int, relative: str) -> int:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError("Odysseys judge input path is unsafe")
    directory_fd = os.dup(root_fd)
    try:
        for component in path.parts[:-1]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise RuntimeError("Odysseys judge input is not a regular file")
        return descriptor
    finally:
        os.close(directory_fd)


def _read_fd(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_limited_regular_fd(descriptor: int, limit: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("Odysseys judge output is not a regular file")
    if before.st_size > limit:
        raise RuntimeError("Odysseys judge output exceeds the size limit")
    data, first_digest, first_size = _read_output_pass(
        descriptor,
        limit,
        capture=True,
    )
    after_first = os.fstat(descriptor)
    if (
        _stable_stat(before) != _stable_stat(after_first)
        or first_size != before.st_size
    ):
        raise RuntimeError("Odysseys judge output changed while being read")
    _, second_digest, second_size = _read_output_pass(
        descriptor,
        limit,
        capture=False,
    )
    after_second = os.fstat(descriptor)
    if (
        _stable_stat(after_first) != _stable_stat(after_second)
        or second_size != after_first.st_size
        or first_digest != second_digest
    ):
        raise RuntimeError("Odysseys judge output changed while being read")
    return data


def _new_judge_output_fd() -> int:
    memfd_create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    close_on_exec = getattr(os, "MFD_CLOEXEC", None)
    if not callable(memfd_create) or allow_sealing is None or close_on_exec is None:
        raise RuntimeError("Odysseys judge output requires sealable memfd support")
    descriptor = memfd_create(
        "cuawright-odysseys-judge-output",
        flags=close_on_exec | allow_sealing,
    )
    try:
        os.set_inheritable(descriptor, True)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _seal_judge_output(descriptor: int) -> None:
    required = (
        fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    )
    try:
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required)
        observed = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    except OSError as exc:
        raise RuntimeError("Odysseys judge output could not be made immutable") from exc
    if observed & required != required:
        raise RuntimeError("Odysseys judge output could not be made immutable")


def _read_output_pass(
    descriptor: int,
    limit: int,
    *,
    capture: bool,
) -> tuple[bytes, str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
        if not chunk:
            break
        digest.update(chunk)
        if capture:
            chunks.append(chunk)
        size += len(chunk)
        if size > limit:
            raise RuntimeError("Odysseys judge output exceeds the size limit")
    return b"".join(chunks), digest.hexdigest(), size


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256_fd(descriptor: int) -> str:
    return hashlib.sha256(_read_fd(descriptor)).hexdigest()


def _write_fd(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while creating Odysseys judge snapshot")
        view = view[written:]


def _validate_judge_output(data: bytes) -> None:
    if not data:
        raise RuntimeError("Odysseys judge produced an empty output")
    if len(data) > _MAX_JUDGE_OUTPUT_BYTES:
        raise RuntimeError("Odysseys judge output exceeds the size limit")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Odysseys judge output is not valid JSON") from exc
    if not isinstance(value, (dict, list)) or not value:
        raise RuntimeError(
            "Odysseys judge output must be a non-empty JSON object or list"
        )


def _validated_final_response(data: bytes, manifest_value: object) -> str:
    try:
        value = data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("immutable final response is not valid UTF-8") from exc
    if not value:
        raise RuntimeError("immutable final response is empty")
    if manifest_value != value:
        raise RuntimeError(
            "manifest final response does not match immutable final response"
        )
    return value


if __name__ == "__main__":
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "cuawright.run.odysseys_judge",
            *sys.argv[1:],
        ],
    )

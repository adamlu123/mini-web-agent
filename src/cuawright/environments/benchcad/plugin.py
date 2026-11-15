from __future__ import annotations

import base64
import os
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from docker.errors import DockerException

from cuawright.environments.benchcad.docker_exec import run_exec
from cuawright.environments.benchcad.environment import (
    FILE_READ_LIMIT,
    BenchCADEnvironment,
)
from cuawright.environments.benchcad.errors import BenchCADInfrastructureError
from cuawright.environments.benchcad.task import load_task
from cuawright.environments.base import (
    CommandResult,
    ControlFileAccessError,
)

DEFAULT_COMMAND_TIMEOUT = 120
DEFAULT_CONTROL_WORKSPACE = "/workspace/.agent"
MAX_CONTROL_WRITE_BYTES = 1024 * 1024
_BRIDGE_DEADLINE = 30
_BRIDGE_PYTHON = "/opt/benchcad-venv/bin/python"
_ARGUMENT_CHUNK_BYTES = 32 * 1024


_READ_FILE_SCRIPT = r"""
import os
import stat
import sys

path = sys.argv[1]
limit = int(sys.argv[2])
fd = None
try:
    if not path.startswith("/") or "\x00" in path:
        raise OSError("path must be absolute")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("path is not a regular file")
    if info.st_size > limit:
        raise OSError("file exceeds read limit")
    chunks = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise OSError("file exceeds read limit")
    data = b"".join(chunks)
    offset = 0
    while offset < len(data):
        offset += os.write(sys.stdout.fileno(), data[offset:])
except (OSError, ValueError) as exc:
    print(f"bridge read rejected: {exc}", file=sys.stderr)
    raise SystemExit(73)
finally:
    if fd is not None:
        os.close(fd)
"""


_READ_CONTROL_FILE_SCRIPT = r"""
import os
import stat
import sys

path = sys.argv[1]
workspace = sys.argv[2]
limit = int(sys.argv[3])

def components(value):
    if not value.startswith("/") or "\x00" in value:
        raise OSError("path must be absolute")
    parts = value.split("/")[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OSError("path is not canonical")
    return parts

directory_fd = None
file_fd = None
try:
    path_parts = components(path)
    workspace_parts = components(workspace)
    if (
        len(path_parts) <= len(workspace_parts)
        or path_parts[:len(workspace_parts)] != workspace_parts
    ):
        raise OSError("path is outside the control workspace")
    directory_fd = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    for part in path_parts[:-1]:
        next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
        os.close(directory_fd)
        directory_fd = next_fd
    file_fd = os.open(
        path_parts[-1],
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=directory_fd,
    )
    info = os.fstat(file_fd)
    if not stat.S_ISREG(info.st_mode):
        raise OSError("control path is not a regular file")
    if info.st_size > limit:
        raise OSError("control file exceeds read limit")
    chunks = []
    total = 0
    while True:
        chunk = os.read(file_fd, min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise OSError("control file exceeds read limit")
    data = b"".join(chunks)
    offset = 0
    while offset < len(data):
        offset += os.write(sys.stdout.fileno(), data[offset:])
except (OSError, ValueError) as exc:
    print(f"control read rejected: {exc}", file=sys.stderr)
    raise SystemExit(73)
finally:
    if file_fd is not None:
        os.close(file_fd)
    if directory_fd is not None:
        os.close(directory_fd)
"""


_WRITE_CONTROL_FILE_SCRIPT = r"""
import base64
import os
import secrets
import stat
import sys

path = sys.argv[1]
workspace = sys.argv[2]
data = base64.b64decode("".join(sys.argv[3:]), validate=True)

def components(value):
    if not value.startswith("/") or "\x00" in value:
        raise OSError("path must be absolute")
    parts = value.split("/")[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OSError("path is not canonical")
    return parts

directory_fd = None
temporary_name = None
temporary_fd = None
try:
    path_parts = components(path)
    workspace_parts = components(workspace)
    if (
        len(path_parts) <= len(workspace_parts)
        or path_parts[:len(workspace_parts)] != workspace_parts
    ):
        raise OSError("path is outside the control workspace")
    directory_fd = os.open(
        "/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    for index, part in enumerate(workspace_parts):
        try:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
        except FileNotFoundError:
            if index != len(workspace_parts) - 1:
                raise
            os.mkdir(part, mode=0o700, dir_fd=directory_fd)
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
        os.close(directory_fd)
        directory_fd = next_fd
    relative_parts = path_parts[len(workspace_parts):]
    for part in relative_parts[:-1]:
        next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
        os.close(directory_fd)
        directory_fd = next_fd
    destination = relative_parts[-1]
    try:
        existing = os.stat(
            destination, dir_fd=directory_fd, follow_symlinks=False
        )
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise OSError("control destination is not a regular file")
    temporary_name = f".{destination}.agent-{secrets.token_hex(12)}"
    temporary_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=directory_fd,
    )
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        offset += os.write(temporary_fd, view[offset:])
    os.fsync(temporary_fd)
    os.close(temporary_fd)
    temporary_fd = None
    os.replace(
        temporary_name,
        destination,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    temporary_name = None
    os.fsync(directory_fd)
except (OSError, ValueError) as exc:
    print(f"control write rejected: {exc}", file=sys.stderr)
    raise SystemExit(73)
finally:
    if temporary_fd is not None:
        os.close(temporary_fd)
    if temporary_name is not None and directory_fd is not None:
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
    if directory_fd is not None:
        os.close(directory_fd)
"""


@dataclass(frozen=True, slots=True)
class BenchCADPluginConfig:
    manifest_path: str | Path
    record_id: str
    run_dir: str | Path
    image: str
    benchcad_repo: str | Path
    defer_evaluation: bool = False
    default_command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    control_workspace: str = DEFAULT_CONTROL_WORKSPACE


@dataclass(slots=True)
class BenchCADPlugin:
    environment: BenchCADEnvironment
    default_command_timeout: int = DEFAULT_COMMAND_TIMEOUT
    control_workspace: str = DEFAULT_CONTROL_WORKSPACE

    def __post_init__(self) -> None:
        _validate_positive_limit(
            self.default_command_timeout,
            "default command timeout",
        )
        _canonical_absolute_path(self.control_workspace, "control workspace")

    @property
    def instruction(self) -> str:
        return self.environment.instruction

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        if not isinstance(command, str):
            raise TypeError("command must be a string")
        effective_timeout = (
            self.default_command_timeout
            if timeout is None
            else _validate_positive_limit(timeout, "command timeout")
        )
        result = self.environment.run_command(command, effective_timeout)
        if result.get("infrastructure_error"):
            raise BenchCADInfrastructureError(
                str(result.get("error") or "BenchCAD command infrastructure failed")
            )
        exit_code = int(result.get("returncode", -1))
        stdout = str(result.get("output") or "")
        stderr = str(result.get("error") or "")
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            inspected=(
                exit_code == 0
                and self.control_workspace not in command
                and _is_inspection_command(command)
            ),
            model_fields=result,
        )

    def read_file(self, path: str, *, max_bytes: int) -> bytes:
        _canonical_absolute_path(path, "file path")
        limit = _validate_read_limit(max_bytes)
        return self._run_bridge(
            [_BRIDGE_PYTHON, "-c", _READ_FILE_SCRIPT, path, str(limit)],
            capture_limit=max(limit + 1, 4096),
        )

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        self._validate_control_path(path, workspace)
        limit = _validate_read_limit(max_bytes)
        return self._run_bridge(
            [
                _BRIDGE_PYTHON,
                "-c",
                _READ_CONTROL_FILE_SCRIPT,
                path,
                workspace,
                str(limit),
            ],
            capture_limit=max(limit + 1, 4096),
        )

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        self._validate_control_path(path, workspace)
        if not isinstance(data, bytes):
            raise TypeError("control-file data must be bytes")
        if len(data) > MAX_CONTROL_WRITE_BYTES:
            raise ControlFileAccessError(
                f"control-file data exceeds {MAX_CONTROL_WRITE_BYTES} bytes"
            )
        encoded = base64.b64encode(data).decode("ascii")
        chunks = [
            encoded[index : index + _ARGUMENT_CHUNK_BYTES]
            for index in range(0, len(encoded), _ARGUMENT_CHUNK_BYTES)
        ] or [""]
        self._run_bridge(
            [
                _BRIDGE_PYTHON,
                "-c",
                _WRITE_CONTROL_FILE_SCRIPT,
                path,
                workspace,
                *chunks,
            ],
            capture_limit=4096,
        )

    def evaluate(self) -> dict[str, object]:
        return self.environment.evaluate()

    def close(self) -> None:
        self.environment.close()

    def _validate_control_path(self, path: str, workspace: str) -> None:
        canonical_workspace = _canonical_absolute_path(
            workspace,
            "control workspace",
        )
        configured_workspace = _canonical_absolute_path(
            self.control_workspace,
            "configured control workspace",
        )
        if canonical_workspace != configured_workspace:
            raise ControlFileAccessError("unexpected control workspace")
        candidate = _canonical_absolute_path(path, "control path")
        if (
            candidate == canonical_workspace
            or canonical_workspace not in candidate.parents
        ):
            raise ControlFileAccessError(
                "control path is outside the configured workspace"
            )

    def _run_bridge(
        self,
        command: Sequence[str],
        *,
        capture_limit: int,
    ) -> bytes:
        try:
            result = run_exec(
                client=self.environment.client,
                container=self.environment.container,
                command=list(command),
                user=f"{os.getuid()}:{os.getgid()}",
                deadline=_BRIDGE_DEADLINE,
                capture_limit=capture_limit,
            )
        except DockerException as exc:
            raise BenchCADInfrastructureError(
                "BenchCAD file bridge infrastructure failed"
            ) from exc
        if result["timed_out"] or result["container_killed"]:
            raise BenchCADInfrastructureError(
                "BenchCAD file bridge timed out or lost its container"
            )
        if int(result["returncode"]) != 0:
            detail = result["stderr"].decode("utf-8", errors="replace").strip()
            raise ControlFileAccessError(
                detail or "BenchCAD file bridge rejected the request"
            )
        return bytes(result["stdout"])


def create_benchcad_plugin(config: BenchCADPluginConfig) -> BenchCADPlugin:
    manifest, task = load_task(config.manifest_path, config.record_id)
    try:
        environment = BenchCADEnvironment(
            task=task,
            manifest=manifest,
            run_dir=Path(config.run_dir).expanduser().resolve(),
            image=config.image,
            benchcad_repo=Path(config.benchcad_repo).expanduser().resolve(),
            benchmark_commit=manifest.benchmark_commit,
            defer_evaluation=config.defer_evaluation,
        )
    except DockerException as exc:
        raise BenchCADInfrastructureError(
            "BenchCAD sandbox initialization failed"
        ) from exc
    return BenchCADPlugin(
        environment=environment,
        default_command_timeout=config.default_command_timeout,
        control_workspace=config.control_workspace,
    )


def _canonical_absolute_path(value: str, description: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if "\x00" in value:
        raise ControlFileAccessError(f"{description} contains a null byte")
    candidate = PurePosixPath(value)
    if (
        not candidate.is_absolute()
        or str(candidate) != value
        or any(part in {".", ".."} for part in candidate.parts)
    ):
        raise ControlFileAccessError(f"{description} must be a canonical absolute path")
    return candidate


def _validate_positive_limit(value: int, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _validate_read_limit(max_bytes: int) -> int:
    limit = _validate_positive_limit(max_bytes, "max_bytes")
    if limit > FILE_READ_LIMIT:
        raise ValueError(f"max_bytes cannot exceed {FILE_READ_LIMIT}")
    return limit


def _is_inspection_command(command: str) -> bool:
    argv = _simple_command_argv(command)
    if argv is None:
        return False
    program = argv[0]
    if program == "benchcad-render":
        if len(argv) != 3 or any(
            not argument or argument.startswith("-") for argument in argv[1:]
        ):
            return False
        step_path = PurePosixPath(argv[1])
        output_path = PurePosixPath(argv[2])
        return (
            step_path.suffix.lower() in {".step", ".stp"}
            and output_path.suffix.lower() == ".png"
            and step_path != output_path
        )
    if program in {
        "cat",
        "file",
        "head",
        "ls",
        "md5sum",
        "pwd",
        "readlink",
        "sha256sum",
        "stat",
        "tail",
        "wc",
        "zipinfo",
    }:
        return True
    if program in {"grep", "rg"}:
        return not any(
            argument == "--pre" or argument.startswith("--pre=")
            for argument in argv[1:]
        )
    if program == "find":
        return not any(_is_mutating_find_action(argument) for argument in argv[1:])
    return False


def _simple_command_argv(command: str) -> list[str] | None:
    if not command.strip() or "\n" in command or "\r" in command:
        return None
    if _has_unquoted_shell_expansion(command):
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        argv = list(lexer)
    except ValueError:
        return None
    if not argv or any(
        token and all(character in ";&|<>" for character in token) for token in argv
    ):
        return None
    return argv


def _has_unquoted_shell_expansion(command: str) -> bool:
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            escaped = False
        elif quote == "'":
            if character == "'":
                quote = None
        elif quote == '"':
            if character == '"':
                quote = None
            elif character == "\\":
                escaped = True
            elif character in "$`":
                return True
        elif character == "\\":
            escaped = True
        elif character in {"'", '"'}:
            quote = character
        elif character in "$`*?[]{}~()":
            return True
        elif character in "@!+" and command[index + 1 : index + 2] == "(":
            return True
        index += 1
    return False


def _is_mutating_find_action(argument: str) -> bool:
    return argument == "-delete" or argument.startswith(
        ("-exec", "-fls", "-fprint", "-fprintf", "-ok")
    )

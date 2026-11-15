from __future__ import annotations

import errno
import base64
import os
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping

from cuawright.environments.cadgenbench.docker_exec import run_exec
from cuawright.environments.cadgenbench.environment import (
    CONTROL_BRIDGE_DIR,
    FILE_READ_LIMIT,
    CadGenBenchEnvironment,
)
from cuawright.environments.cadgenbench.errors import CadGenBenchInfrastructureError
from cuawright.environments.base import (
    CommandResult,
    ControlFileAccessError,
)

_PYTHON = "/opt/cadgenbench-venv/bin/python"
_CONTROL_WRITE_LIMIT = 1024 * 1024
_ARGUMENT_CHUNK_BYTES = 32 * 1024
_CONTROL_READ_PROGRAM = r"""
import os
import stat
import sys

workspace, relative, raw_limit = sys.argv[1:]
limit = int(raw_limit)
flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
fd = os.open("/", flags)
try:
    for component in workspace.split("/")[1:]:
        next_fd = os.open(component, flags, dir_fd=fd)
        os.close(fd)
        fd = next_fd
    parts = relative.split("/")
    for component in parts[:-1]:
        next_fd = os.open(component, flags, dir_fd=fd)
        os.close(fd)
        fd = next_fd
    file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("control path is not a regular file")
        data = bytearray()
        while len(data) <= limit:
            chunk = os.read(file_fd, min(65536, limit + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > limit:
            raise OSError("control file exceeds the requested limit")
        os.write(1, data)
    finally:
        os.close(file_fd)
finally:
    os.close(fd)
"""

_CONTROL_WRITE_PROGRAM = r"""
import base64
import os
import secrets
import sys

workspace, relative = sys.argv[1:3]
data = base64.b64decode("".join(sys.argv[3:]), validate=True)
directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

def open_directory(path, create):
    fd = os.open("/", directory_flags)
    try:
        for component in path.split("/")[1:]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=fd)
                    child = os.open(component, directory_flags, dir_fd=fd)
                except FileExistsError:
                    child = os.open(component, directory_flags, dir_fd=fd)
            else:
                child = os.open(component, directory_flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise

workspace_fd = None
parent_fd = None
temporary_name = None
try:
    workspace_fd = open_directory(workspace, True)
    parent_fd = os.dup(workspace_fd)
    parts = relative.split("/")
    for component in parts[:-1]:
        try:
            os.mkdir(component, 0o700, dir_fd=parent_fd)
            child = os.open(component, directory_flags, dir_fd=parent_fd)
        except FileExistsError:
            child = os.open(component, directory_flags, dir_fd=parent_fd)
        os.close(parent_fd)
        parent_fd = child

    temporary_name = ".cadgenbench-control-" + secrets.token_hex(16)
    output_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        view = memoryview(data)
        while view:
            written = os.write(output_fd, view)
            view = view[written:]
        os.fchmod(output_fd, 0o600)
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    os.replace(
        temporary_name,
        parts[-1],
        src_dir_fd=parent_fd,
        dst_dir_fd=parent_fd,
    )
    temporary_name = None
    os.fsync(parent_fd)
finally:
    if temporary_name is not None and parent_fd is not None:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
    if parent_fd is not None:
        os.close(parent_fd)
    if workspace_fd is not None:
        os.close(workspace_fd)
"""


@dataclass(frozen=True, slots=True)
class CadGenBenchPluginConfig:
    command_timeout_seconds: int = 300
    bridge_timeout_seconds: int = 30
    max_file_bytes: int = FILE_READ_LIMIT
    workspace_root: str = "/workspace"
    bridge_directory: str = CONTROL_BRIDGE_DIR

    def __post_init__(self) -> None:
        for name in (
            "command_timeout_seconds",
            "bridge_timeout_seconds",
            "max_file_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _absolute_path(self.workspace_root, "workspace_root")
        _absolute_path(self.bridge_directory, "bridge_directory")

    def build(self, environment: CadGenBenchEnvironment) -> CadGenBenchPlugin:
        return CadGenBenchPlugin(environment, config=self)


class CadGenBenchPlugin:
    def __init__(
        self,
        environment: CadGenBenchEnvironment,
        *,
        config: CadGenBenchPluginConfig | None = None,
    ) -> None:
        self.environment = environment
        self.config = config or CadGenBenchPluginConfig()

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        if not isinstance(command, str):
            raise TypeError("command must be a string")
        effective_timeout = (
            self.config.command_timeout_seconds
            if timeout is None
            else _positive_integer(timeout, "timeout")
        )
        payload = self.environment.run_command(command, effective_timeout)
        if not isinstance(payload, Mapping):
            raise TypeError(
                "CadGenBenchEnvironment.run_command() must return a mapping"
            )
        if payload.get("infrastructure_error"):
            raise CadGenBenchInfrastructureError(
                str(
                    payload.get("error")
                    or "CADGenBench container or MCP infrastructure failed"
                )
            )
        exit_code = int(payload.get("returncode", -1))
        stdout = str(payload.get("output", ""))
        stderr = str(payload.get("error", ""))
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            inspected=(
                exit_code == 0
                and "/workspace/.agent" not in command
                and _is_inspection_command(command)
            ),
            model_fields=payload,
        )

    def read_file(self, path: str, *, max_bytes: int) -> bytes:
        _absolute_path(path, "path")
        limit = _positive_integer(max_bytes, "max_bytes")
        if limit > self.config.max_file_bytes:
            raise OSError(
                errno.EFBIG,
                f"requested read exceeds the {self.config.max_file_bytes}-byte bridge limit",
            )
        data = self.environment.get_file(path, max_bytes=limit)
        if data is None:
            raise OSError(errno.ENOENT, "file is unavailable or exceeds the read limit")
        if len(data) > limit:
            raise OSError(errno.EFBIG, "file exceeds the requested read limit")
        return data

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        relative = self._control_relative_path(path, workspace)
        limit = _positive_integer(max_bytes, "max_bytes")
        if limit > self.config.max_file_bytes:
            raise ControlFileAccessError("control read exceeds the bridge limit")
        result = self._trusted_exec(
            [_PYTHON, "-c", _CONTROL_READ_PROGRAM, workspace, relative, str(limit)],
            capture_limit=limit + 1,
        )
        if int(result.get("returncode", -1)) != 0 or result.get("stdout_truncated"):
            raise ControlFileAccessError(
                _bridge_error(result, "secure control read failed")
            )
        data = result.get("stdout", b"")
        if not isinstance(data, bytes) or len(data) > limit:
            raise ControlFileAccessError("secure control read returned invalid data")
        return data

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        if not isinstance(data, bytes):
            raise TypeError("control data must be bytes")
        relative = self._control_relative_path(path, workspace)
        if len(data) > _CONTROL_WRITE_LIMIT:
            raise ControlFileAccessError("control write exceeds the bridge limit")
        encoded = base64.b64encode(data).decode("ascii")
        chunks = [
            encoded[index : index + _ARGUMENT_CHUNK_BYTES]
            for index in range(0, len(encoded), _ARGUMENT_CHUNK_BYTES)
        ] or [""]
        result = self._trusted_exec(
            [
                _PYTHON,
                "-c",
                _CONTROL_WRITE_PROGRAM,
                workspace,
                relative,
                *chunks,
            ],
            capture_limit=4096,
        )
        if int(result.get("returncode", -1)) != 0:
            raise ControlFileAccessError(
                _bridge_error(result, "atomic control write failed")
            )

    def evaluate(self) -> dict[str, Any]:
        return self.environment.evaluate()

    def close(self) -> None:
        self.environment.close()

    def _control_relative_path(self, path: str, workspace: str) -> str:
        target = _absolute_path(path, "path")
        workspace_path = _absolute_path(workspace, "workspace")
        root = _absolute_path(self.config.workspace_root, "workspace_root")
        if workspace_path != root and root not in workspace_path.parents:
            raise ControlFileAccessError("workspace is outside the configured root")
        if target == workspace_path or workspace_path not in target.parents:
            raise ControlFileAccessError("control path is outside the workspace")
        return target.relative_to(workspace_path).as_posix()

    def _trusted_exec(
        self,
        command: list[str],
        *,
        capture_limit: int,
    ) -> dict[str, Any]:
        container = getattr(self.environment, "container", None)
        if container is None:
            raise ControlFileAccessError("CadGenBench container is unavailable")
        return run_exec(
            client=self.environment.client,
            container=container,
            command=command,
            user=f"{os.getuid()}:{os.getgid()}",
            deadline=self.config.bridge_timeout_seconds,
            capture_limit=capture_limit,
            workdir="/",
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
        )


def create_cadgenbench_plugin(
    environment: CadGenBenchEnvironment,
    *,
    config: CadGenBenchPluginConfig | None = None,
) -> CadGenBenchPlugin:
    return CadGenBenchPlugin(environment, config=config)


def _absolute_path(value: str, description: str) -> PurePosixPath:
    if not isinstance(value, str) or "\x00" in value:
        raise ControlFileAccessError(f"{description} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ControlFileAccessError(f"{description} must be an absolute POSIX path")
    return path


def _positive_integer(value: int, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _bridge_error(result: Mapping[str, Any], fallback: str) -> str:
    stderr = result.get("stderr", b"")
    if isinstance(stderr, bytes):
        detail = stderr.decode("utf-8", errors="replace").strip()
    else:
        detail = str(stderr).strip()
    return detail[-1000:] or fallback


def _is_inspection_command(command: str) -> bool:
    argv = _simple_command_argv(command)
    if argv is None:
        return False
    program = argv[0]
    if program == "cadgenbench-validate":
        return len(argv) == 2 and not argv[1].startswith("-")
    if program == "cadgenbench-render":
        return len(argv) == 3 and all(
            argument and not argument.startswith("-") for argument in argv[1:]
        )
    if program == "cad-mcp":
        return _is_read_only_cad_mcp(argv)
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


def _is_read_only_cad_mcp(argv: list[str]) -> bool:
    index = 1
    while index < len(argv) and argv[index].startswith("--"):
        option = argv[index]
        if option in {"--timeout", "--url"}:
            if index + 1 >= len(argv):
                return False
            if option == "--timeout":
                try:
                    if int(argv[index + 1]) <= 0:
                        return False
                except ValueError:
                    return False
            elif argv[index + 1] != "http://127.0.0.1:8765/mcp":
                return False
            index += 2
            continue
        if option.startswith("--timeout="):
            try:
                if int(option.partition("=")[2]) <= 0:
                    return False
            except ValueError:
                return False
            index += 1
            continue
        if option.startswith("--url="):
            if option.partition("=")[2] != "http://127.0.0.1:8765/mcp":
                return False
            index += 1
            continue
        return False
    remaining = argv[index:]
    if remaining in (["tools"], ["resources"]):
        return True
    if len(remaining) == 2 and remaining[0] in {"schema", "resource"}:
        return bool(remaining[1])
    if len(remaining) < 2 or remaining[0] != "call":
        return False
    read_only_tools = {
        "cross_sections",
        "find_bosses",
        "find_hole_patterns",
        "find_holes",
        "measure",
        "render_view",
        "shape_compare",
        "validate",
        "workflow_hints",
    }
    if remaining[1] not in read_only_tools:
        return False
    arguments = remaining[2:]
    if not arguments:
        return True
    if len(arguments) == 2 and arguments[0] in {"--json", "--json-file"}:
        return True
    return len(arguments) == 1 and arguments[0].startswith("--json=")

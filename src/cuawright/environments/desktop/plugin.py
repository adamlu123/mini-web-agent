from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping, Protocol

from cuawright.environments.base import CommandResult, ControlFileAccessError


_BRIDGE_PREFIX = "__CUAWRIGHT_DESKTOP_BRIDGE_V1__"


class GuestController(Protocol):
    def run_python_script(
        self,
        script: str,
        timeout: int = 90,
    ) -> Mapping[str, Any] | None: ...


CommandExecutor = Callable[[str, int | None], CommandResult]


@dataclass(frozen=True, slots=True)
class GuestWorkspace:
    """Map the harness's stable logical workspace to its guest location."""

    logical_path: str
    guest_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "logical_path",
            _absolute_clean_path(self.logical_path, "logical workspace"),
        )
        object.__setattr__(
            self,
            "guest_path",
            _absolute_clean_path(self.guest_path, "guest workspace"),
        )

    def guest_file(self, path: str, *, workspace: str) -> str:
        try:
            expected_workspace = _absolute_clean_path(workspace, "workspace")
            candidate = _absolute_clean_path(path, "control-file path")
        except (TypeError, ValueError) as exc:
            raise ControlFileAccessError(str(exc)) from exc
        if expected_workspace != self.logical_path:
            raise ControlFileAccessError("unrecognized logical workspace")

        logical = PurePosixPath(self.logical_path)
        candidate_path = PurePosixPath(candidate)
        if candidate_path == logical or logical not in candidate_path.parents:
            raise ControlFileAccessError(
                "control-file path must be beneath the logical workspace"
            )
        relative = candidate_path.relative_to(logical)
        return str(PurePosixPath(self.guest_path).joinpath(relative))


class DesktopCommandPlugin:
    """Generic command and secure file access over an existing desktop guest."""

    def __init__(
        self,
        desktop_env: Any,
        execute_command: CommandExecutor,
        *,
        workspace: GuestWorkspace,
        bridge_timeout: int = 90,
        bridge_error_label: str = "Desktop guest bridge",
    ) -> None:
        if isinstance(bridge_timeout, bool) or not isinstance(bridge_timeout, int):
            raise TypeError("bridge_timeout must be an integer")
        if bridge_timeout <= 0:
            raise ValueError("bridge_timeout must be positive")
        controller = getattr(desktop_env, "controller", None)
        if controller is None or not callable(
            getattr(controller, "run_python_script", None)
        ):
            raise TypeError("desktop_env.controller must provide run_python_script()")
        self.desktop_env = desktop_env
        self.controller: GuestController = controller
        self.execute_command = execute_command
        self.workspace = workspace
        self.bridge_timeout = bridge_timeout
        self.bridge_error_label = bridge_error_label

    @property
    def logical_workspace(self) -> str:
        return self.workspace.logical_path

    @property
    def guest_workspace(self) -> str:
        return self.workspace.guest_path

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        try:
            return self.execute_command(command, timeout)
        finally:
            self._mark_environment_used()

    def read_file(self, path: str, *, max_bytes: int) -> bytes:
        guest_path = _absolute_clean_path(path, "file path")
        limit = _read_limit(max_bytes)
        payload = self._bridge(
            _READ_FILE_SCRIPT,
            {"path": guest_path, "max_bytes": limit},
            control=False,
        )
        return _decode_data(payload, self.bridge_error_label)

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        guest_path = self.workspace.guest_file(path, workspace=workspace)
        payload = self._bridge(
            _CONTROL_READ_SCRIPT,
            {
                "root": self.workspace.guest_path,
                "path": guest_path,
                "max_bytes": _read_limit(max_bytes),
            },
            control=True,
        )
        return _decode_data(payload, self.bridge_error_label)

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        if not isinstance(data, bytes):
            raise TypeError("control-file data must be bytes")
        guest_path = self.workspace.guest_file(path, workspace=workspace)
        self._mark_environment_used()
        self._bridge(
            _CONTROL_WRITE_SCRIPT,
            {
                "root": self.workspace.guest_path,
                "path": guest_path,
                "data": base64.b64encode(data).decode("ascii"),
            },
            control=True,
        )

    def _bridge(
        self,
        template: str,
        request: Mapping[str, Any],
        *,
        control: bool,
    ) -> Mapping[str, Any]:
        script = _bridge_script(template, request)
        result = self.controller.run_python_script(
            script,
            timeout=self.bridge_timeout,
        )
        if not result or result.get("status") != "success":
            raise OSError(f"{self.bridge_error_label} execution failed")

        output = result.get("output", "")
        if not isinstance(output, str):
            raise OSError(f"{self.bridge_error_label} returned invalid output")
        response = _bridge_response(output, self.bridge_error_label)
        if response.get("ok") is True:
            return response

        kind = str(response.get("kind") or "OSError")
        message = str(response.get("message") or "guest file operation failed")
        if control:
            if kind == "FileNotFoundError":
                raise FileNotFoundError(message)
            raise ControlFileAccessError(message)
        if kind == "FileNotFoundError":
            raise FileNotFoundError(message)
        if kind == "PermissionError":
            raise PermissionError(message)
        if kind == "IsADirectoryError":
            raise IsADirectoryError(message)
        raise OSError(message)

    def _mark_environment_used(self) -> None:
        if hasattr(self.desktop_env, "is_environment_used"):
            self.desktop_env.is_environment_used = True


def _absolute_clean_path(value: str, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise ValueError(f"{description} must be absolute")
    if ".." in path.parts:
        raise ControlFileAccessError(f"{description} cannot contain traversal")
    normalized = str(path)
    if normalized == "/":
        raise ValueError(f"{description} cannot be the filesystem root")
    return normalized


def _read_limit(max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("max_bytes must be an integer")
    if max_bytes < 0:
        raise ValueError("max_bytes cannot be negative")
    return max_bytes


def _bridge_script(template: str, request: Mapping[str, Any]) -> str:
    encoded = base64.b64encode(
        json.dumps(dict(request), separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return template.replace("__REQUEST_B64__", encoded)


def _bridge_response(output: str, label: str) -> Mapping[str, Any]:
    lines = [
        line[len(_BRIDGE_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(_BRIDGE_PREFIX)
    ]
    if len(lines) != 1:
        raise OSError(f"{label} returned no unique response")
    try:
        response = json.loads(lines[0])
    except (TypeError, json.JSONDecodeError) as exc:
        raise OSError(f"{label} returned malformed JSON") from exc
    if not isinstance(response, dict):
        raise OSError(f"{label} returned a non-object response")
    return response


def _decode_data(payload: Mapping[str, Any], label: str) -> bytes:
    encoded = payload.get("data")
    if not isinstance(encoded, str):
        raise OSError(f"{label} omitted file data")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise OSError(f"{label} returned malformed file data") from exc


_READ_FILE_SCRIPT = r"""
import base64
import json
import os
import stat

PREFIX = "__CUAWRIGHT_DESKTOP_BRIDGE_V1__"
request = json.loads(base64.b64decode("__REQUEST_B64__"))

try:
    fd = os.open(request["path"], os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("path is not a regular file")
        remaining = request["max_bytes"]
        chunks = []
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    response = {"ok": True, "data": base64.b64encode(data).decode("ascii")}
except Exception as exc:
    response = {
        "ok": False,
        "kind": type(exc).__name__,
        "message": str(exc),
        "errno": getattr(exc, "errno", None),
    }
print(PREFIX + json.dumps(response, separators=(",", ":")))
"""


_CONTROL_COMMON = r"""
import base64
import json
import os
import secrets
import stat
from pathlib import PurePosixPath

PREFIX = "__CUAWRIGHT_DESKTOP_BRIDGE_V1__"
request = json.loads(base64.b64decode("__REQUEST_B64__"))

class BridgeAccessError(OSError):
    pass

def clean_absolute(value, label):
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise BridgeAccessError(label + " must be absolute and traversal-free")
    return str(path)

def open_directory_component(parent_fd, name):
    before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise BridgeAccessError("workspace path component is not a real directory")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    child_fd = os.open(name, flags, dir_fd=parent_fd)
    after = os.fstat(child_fd)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(child_fd)
        raise BridgeAccessError("workspace path changed during validation")
    return child_fd

def open_root(root):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in PurePosixPath(root).parts[1:]:
            next_fd = open_directory_component(fd, component)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise

def validated_target(root, path):
    root = clean_absolute(root, "workspace")
    path = clean_absolute(path, "path")
    if os.path.realpath(root) != root:
        raise BridgeAccessError("workspace is not canonical")
    resolved = os.path.realpath(path)
    if os.path.commonpath((root, resolved)) != root:
        raise BridgeAccessError("resolved path escapes workspace")
    relative = PurePosixPath(path).relative_to(PurePosixPath(root))
    if not relative.parts:
        raise BridgeAccessError("path must name a file beneath workspace")
    return root, relative

def open_parent(root, relative):
    fd = open_root(root)
    try:
        for component in relative.parts[:-1]:
            next_fd = open_directory_component(fd, component)
            os.close(fd)
            fd = next_fd
        return fd, relative.name
    except Exception:
        os.close(fd)
        raise
"""


_CONTROL_READ_SCRIPT = (
    _CONTROL_COMMON
    + r"""
try:
    root, relative = validated_target(request["root"], request["path"])
    parent_fd, name = open_parent(root, relative)
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise BridgeAccessError("control-file target is not a regular file")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(name, flags, dir_fd=parent_fd)
        try:
            after = os.fstat(fd)
            if not stat.S_ISREG(after.st_mode):
                raise BridgeAccessError("control-file target is not a regular file")
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise BridgeAccessError("control-file target changed during validation")
            remaining = request["max_bytes"]
            chunks = []
            while remaining:
                chunk = os.read(fd, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)
    response = {"ok": True, "data": base64.b64encode(data).decode("ascii")}
except Exception as exc:
    response = {
        "ok": False,
        "kind": type(exc).__name__,
        "message": str(exc),
        "errno": getattr(exc, "errno", None),
    }
print(PREFIX + json.dumps(response, separators=(",", ":")))
"""
)


_CONTROL_WRITE_SCRIPT = (
    _CONTROL_COMMON
    + r"""
try:
    root, relative = validated_target(request["root"], request["path"])
    parent_fd, name = open_parent(root, relative)
    temporary = None
    try:
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            before = None
        if before is not None and (
            stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode)
        ):
            raise BridgeAccessError("control-file target is not a regular file")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        for _ in range(32):
            candidate = "." + name + ".agent-" + secrets.token_hex(12)
            try:
                fd = os.open(candidate, flags, 0o600, dir_fd=parent_fd)
                temporary = candidate
                break
            except FileExistsError:
                continue
        else:
            raise OSError("could not reserve an atomic-write temporary file")

        try:
            data = base64.b64decode(request["data"], validate=True)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write to atomic temporary file")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)

        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if before is None and current is not None:
            raise BridgeAccessError("control-file target appeared during write")
        if before is not None and (
            current is None
            or stat.S_ISLNK(current.st_mode)
            or (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise BridgeAccessError("control-file target changed during write")

        os.replace(
            temporary,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary = None
        os.fsync(parent_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
    response = {"ok": True}
except Exception as exc:
    response = {
        "ok": False,
        "kind": type(exc).__name__,
        "message": str(exc),
        "errno": getattr(exc, "errno", None),
    }
print(PREFIX + json.dumps(response, separators=(",", ":")))
"""
)

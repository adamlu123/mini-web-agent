from __future__ import annotations

import json
import logging
import os
import re
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from cuawright.environments.desktop.commands import (
    is_inspection_command as is_desktop_inspection_command,
)
from cuawright.environments.desktop.guest_commands import (
    run_guest_command as run_desktop_guest_command,
)

logger = logging.getLogger("cuawright.environments.osworld")
MIN_GUEST_FREE_KB_AFTER_SETUP = 5 * 1024 * 1024
_SETUP_EXECUTE_PATCHED = False
_SETUP_LAUNCH_PATCHED = False
_LONG_LIVED_EXECUTE_SCRIPTS = frozenset(
    {
        "cd /home/user/ds2019-request/ds2019-request && npm start",
    }
)
_READ_ONLY_FILESYSTEM_MARKERS = (
    "read-only file system",
    "errno 30",
)
_GUEST_HOME_WRITE_PROBE = r"""
import os
import secrets

path = "/home/user/.cuawright-write-probe-" + secrets.token_hex(12)
fd = None
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    view = memoryview(b"ok")
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short write during guest home writability probe")
        view = view[written:]
    os.fsync(fd)
finally:
    if fd is not None:
        try:
            os.close(fd)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
"""


class InfrastructureError(RuntimeError):
    pass


class ProxyConfigurationError(ValueError):
    pass


def preflight_proxy_configuration(
    config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise ProxyConfigurationError(f"OSWorld proxy configuration not found: {path}")
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProxyConfigurationError(
            f"OSWorld proxy configuration is invalid JSON: {path}"
        ) from exc
    if not isinstance(entries, list) or not entries:
        raise ProxyConfigurationError(
            "OSWorld proxy configuration must contain at least one proxy entry"
        )

    entry = entries[0]
    if not isinstance(entry, Mapping):
        raise ProxyConfigurationError("OSWorld proxy entry must be an object")
    required = ("host", "port", "username", "password", "protocol")
    missing = [name for name in required if not entry.get(name)]
    if missing:
        raise ProxyConfigurationError(
            "OSWorld proxy entry is missing required fields: " + ", ".join(missing)
        )
    username = str(entry["username"]).strip()
    password = str(entry["password"]).strip()
    placeholders = {"username", "password", "your_username", "your_password"}
    if username.lower() in placeholders or password.lower() in placeholders:
        raise ProxyConfigurationError(
            "OSWorld proxy configuration still contains example credentials"
        )
    if path.stat().st_mode & 0o077:
        raise ProxyConfigurationError(
            "OSWorld proxy configuration containing credentials must not be "
            "group- or world-readable"
        )

    host = str(entry["host"]).strip()
    port = int(entry["port"])
    protocol = str(entry["protocol"]).strip().lower()
    proxy_url = (
        f"{protocol}://{quote(username, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}"
    )
    try:
        response = requests.get(
            "https://example.com/",
            proxies={"http": proxy_url, "https": proxy_url},
            timeout=(10, 20),
        )
        response.raise_for_status()
        response.close()
    except requests.RequestException as exc:
        raise ProxyConfigurationError(
            f"OSWorld proxy preflight failed for {host}:{port} ({type(exc).__name__})"
        ) from None
    return {
        "status": "ready",
        "host": host,
        "port": port,
        "protocol": protocol,
        "provider": entry.get("provider"),
        "country": entry.get("country"),
    }


def exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def exception_diagnostics(exc: BaseException) -> str:
    return " <- ".join(
        f"{type(item).__name__}: {item}" for item in exception_chain(exc)
    )[:4_000]


def guest_command_result_diagnostics(result: Mapping[str, Any]) -> str:
    details = [
        str(result.get(name) or "").strip()
        for name in ("error", "output")
        if str(result.get(name) or "").strip()
    ]
    if not details:
        details.append(
            f"status={result.get('status')!r}, returncode={result.get('returncode')!r}"
        )
    return " | ".join(details)[:4_000]


def raise_for_guest_command_infrastructure(
    result: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if result.get("infrastructure_error") is True:
        raise InfrastructureError(
            f"{context}: {guest_command_result_diagnostics(result)}"
        )


def raise_for_confirmed_guest_home_read_only(
    controller: Any,
    value: object,
    *,
    context: str,
) -> None:
    if not _is_read_only_filesystem_error(value):
        return
    result = run_guest_command(
        controller,
        f"python3 -c {shlex.quote(_GUEST_HOME_WRITE_PROBE)}",
        10,
    )
    if not isinstance(result, Mapping):
        raise InfrastructureError(
            f"{context}: guest home writability probe returned no result"
        )
    raise_for_guest_command_infrastructure(
        result,
        context=f"{context}: guest home writability probe failed",
    )
    if result.get("status") == "success" and result.get("returncode") in (None, 0):
        return
    diagnostics = guest_command_result_diagnostics(result)
    if _is_read_only_filesystem_error(diagnostics):
        raise InfrastructureError(
            f"{context}: /home/user is globally read-only ({diagnostics})"
        )


def _is_read_only_filesystem_error(value: object) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in _READ_ONLY_FILESYSTEM_MARKERS)


def is_guest_infrastructure_error(value: object) -> bool:
    text = str(value).lower()
    markers = (
        "500 internal server error",
        "upload failed with status 500",
        "connection reset by peer",
        "remote end closed connection",
        "connection aborted",
        "guest disk exhausted",
        "guest command service failed",
        "osworld guest bridge execution failed",
    )
    return any(marker in text for marker in markers)


def is_retryable_setup_error(exc: BaseException) -> bool:
    if any(
        isinstance(item, (requests.ConnectionError, requests.Timeout))
        for item in exception_chain(exc)
    ):
        return True
    diagnostics = exception_diagnostics(exc).lower()
    markers = (
        "connection refused",
        "connection reset",
        "connection aborted",
        "remote disconnected",
        "remote end closed",
        "server not ready",
        "environment setup returned unsuccessful status",
        "proxy setup failed before applying task runtime overrides",
        "retry limit reached",
        "failed to execute bash script after",
        "failed to execute python script after",
        "socket hang up",
        "timed out",
        "timeout",
        "500 internal server error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
    )
    return is_guest_infrastructure_error(diagnostics) or any(
        marker in diagnostics for marker in markers
    )


def install_setup_compatibility_shim() -> None:
    """Launch GUI setup commands asynchronously when upstream calls execute()."""

    global _SETUP_EXECUTE_PATCHED
    from desktop_env.controllers.setup import SetupController

    if _SETUP_EXECUTE_PATCHED:
        _install_proxy_bypass_launch_patch(SetupController)
        return

    original = SetupController.execute

    @wraps(original)
    def execute(
        self: SetupController,
        command: Any,
        stdout: str = "",
        stderr: str = "",
        shell: bool = False,
        until: Any = None,
        quiet: bool = False,
        timeout: int = 120,
    ) -> Any:
        if (
            isinstance(command, list)
            and command
            and not shell
            and not stdout
            and not stderr
            and not until
        ):
            executable = Path(command[0]).name
            args = set(command[1:])
            gui_executables = {
                "blender",
                "eog",
                "evince",
                "firefox",
                "gimp",
                "google-chrome",
                "libreoffice",
                "shotcut",
                "thunderbird",
                "vlc",
                "wps",
                "wpp",
                "et",
            }
            if executable in gui_executables and not (
                args & {"--headless", "--convert-to", "--version"}
            ):
                logger.info("Launching GUI setup command asynchronously: %s", command)
                return self.launch(command)
            if executable in {"bash", "sh"} and "-c" in command:
                script = command[command.index("-c") + 1]
                if (
                    script in _LONG_LIVED_EXECUTE_SCRIPTS
                    or _shell_requests_background_execution(script)
                ):
                    logger.info(
                        "Launching background setup command asynchronously: %s",
                        command,
                    )
                    result = self.launch(command)
                    sleep_match = re.search(r"\bsleep\s+(\d+(?:\.\d+)?)", script)
                    if sleep_match:
                        time.sleep(min(float(sleep_match.group(1)), 10.0))
                    return result
        return original(
            self,
            command,
            stdout=stdout,
            stderr=stderr,
            shell=shell,
            until=until,
            quiet=quiet,
            timeout=timeout,
        )

    SetupController.execute = execute
    _SETUP_EXECUTE_PATCHED = True
    _install_proxy_bypass_launch_patch(SetupController)


def _install_proxy_bypass_launch_patch(setup_controller_type: type[Any]) -> None:
    global _SETUP_LAUNCH_PATCHED
    if _SETUP_LAUNCH_PATCHED:
        return
    original = setup_controller_type._launch_setup

    @wraps(original)
    def launch_setup(
        self: Any,
        command: Any,
        shell: bool = False,
    ) -> Any:
        if (
            isinstance(command, list)
            and command
            and Path(command[0]).name == "google-chrome"
            and self.use_proxy
        ):
            command = list(command)
            suffix = os.environ.get("WEBSITE_HOST_SUFFIX", "").strip()
            if suffix and not any(
                str(arg).startswith("--proxy-bypass-list=") for arg in command
            ):
                command.append(f"--proxy-bypass-list=localhost;127.0.0.1;*.{suffix}")
        return original(self, command, shell=shell)

    setup_controller_type._launch_setup = launch_setup
    _SETUP_LAUNCH_PATCHED = True


def _shell_requests_background_execution(script: str) -> bool:
    """Recognize explicit shell backgrounding without mistaking ``&&`` for ``&``."""

    lexer = shlex.shlex(
        script,
        posix=True,
        punctuation_chars=";&|()<>",
    )
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    command_start = True
    for token in tokens:
        if token == "&":
            return True
        if token in {";", "&&", "||", "|", "(", ")"}:
            command_start = True
            continue
        if command_start and token in {"do", "then", "elif", "else", "!"}:
            continue
        if command_start:
            if token == "nohup":
                return True
            if "=" in token and not token.startswith(("/", "./", "../")):
                name, _, _ = token.partition("=")
                if name.replace("_", "a").isalnum():
                    continue
            command_start = False
    return False


def run_guest_command(
    controller: Any,
    script: str,
    timeout: int,
) -> dict[str, Any]:
    """Compatibility re-export of neutral process-safe desktop execution."""

    return run_desktop_guest_command(
        controller,
        script,
        timeout,
        request_module=requests,
    )


def guest_root_free_kb(controller: Any) -> int | None:
    result = run_guest_command(controller, "df -Pk / | tail -n 1", 10)
    if result.get("status") != "success":
        return None
    fields = str(result.get("output") or "").split()
    if len(fields) < 6:
        return None
    try:
        return int(fields[3])
    except ValueError:
        return None


def ensure_guest_disk_headroom(controller: Any) -> None:
    free_kb = guest_root_free_kb(controller)
    if free_kb is None:
        raise InfrastructureError(
            "Unable to verify guest disk headroom after task setup."
        )
    logger.info("Guest root free space after setup: %.2f GiB", free_kb / 1024 / 1024)
    if free_kb < MIN_GUEST_FREE_KB_AFTER_SETUP:
        raise InfrastructureError(
            "Guest disk headroom below 5 GiB after task setup: "
            f"{free_kb / 1024 / 1024:.2f} GiB free."
        )


def is_inspection_command(command: str) -> bool:
    return is_desktop_inspection_command(
        command,
        extra_inspector=_is_osw_query,
    )


def _is_osw_query(argv: list[str]) -> bool:
    if not argv or argv[0] != "osw":
        return False
    if argv in (["osw", "--help"], ["osw", "--list"]):
        return True
    if len(argv) in {3, 5} and argv[1] == "--show":
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.md", argv[2])
            or Path(argv[2]).name != argv[2]
        ):
            return False
        return len(argv) == 3 or (argv[3] == "--heading" and bool(argv[4].strip()))
    query_count = 0
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--limit":
            if index + 1 >= len(argv):
                return False
            try:
                if int(argv[index + 1]) <= 0:
                    return False
            except ValueError:
                return False
            index += 2
            continue
        if argument.startswith("--limit="):
            try:
                if int(argument.partition("=")[2]) <= 0:
                    return False
            except ValueError:
                return False
            index += 1
            continue
        if argument.startswith("-"):
            return False
        query_count += 1
        index += 1
    return query_count == 1


PhaseAdvance = Callable[[int], Any]
Evaluator = Callable[[], Any]
Closer = Callable[[], None]


@dataclass(frozen=True, slots=True)
class OSWorldTaskIdentity:
    task_id: str
    benchmark_release: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")


@dataclass(frozen=True, slots=True)
class OSWorldEnvironmentIdentity:
    provider_name: str
    vm_path: str
    snapshot_name: str | None = None

    def __post_init__(self) -> None:
        if not self.provider_name:
            raise ValueError("provider_name cannot be empty")
        if not self.vm_path:
            raise ValueError("vm_path cannot be empty")


@dataclass(frozen=True, slots=True)
class OSWorldLifecycleConfig:
    task: OSWorldTaskIdentity
    environment: OSWorldEnvironmentIdentity
    workspace: Any


@dataclass(slots=True)
class OSWorldLifecycle:
    """Compose phase changes, evaluation, and cleanup over upstream DesktopEnv."""

    desktop_env: Any
    config: OSWorldLifecycleConfig
    phase_advance: PhaseAdvance
    evaluator: Evaluator | None = None
    closer: Closer | None = None

    def advance_phase(self, phase_index: int) -> Any:
        if isinstance(phase_index, bool) or not isinstance(phase_index, int):
            raise TypeError("phase_index must be an integer")
        if phase_index < 1:
            raise ValueError("phase_index must be at least 1")
        return self.phase_advance(phase_index)

    def evaluate(self) -> Any:
        callback = self.evaluator or self.desktop_env.evaluate
        return callback()

    def close(self) -> None:
        callback = self.closer or self.desktop_env.close
        callback()

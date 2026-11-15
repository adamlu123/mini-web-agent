from __future__ import annotations

import threading
from typing import Any

from docker.errors import DockerException


def run_exec(
    *,
    client: Any,
    container: Any,
    command: list[str],
    user: str,
    deadline: int,
    capture_limit: int,
    workdir: str | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    execution = client.api.exec_create(
        container.id,
        command,
        user=user,
        workdir=workdir,
        environment=environment,
    )
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    stdout_size = 0
    stderr_size = 0
    stream_error: list[DockerException] = []
    inspection: dict[str, Any] = {}

    def consume() -> None:
        nonlocal stdout_size, stderr_size, inspection
        try:
            output_stream = client.api.exec_start(
                execution["Id"],
                stream=True,
                demux=True,
            )
            for stdout, stderr in output_stream:
                if stdout and stdout_size < capture_limit:
                    chunk = stdout[: capture_limit - stdout_size]
                    stdout_parts.append(chunk)
                    stdout_size += len(chunk)
                if stderr and stderr_size < capture_limit:
                    chunk = stderr[: capture_limit - stderr_size]
                    stderr_parts.append(chunk)
                    stderr_size += len(chunk)
            inspection = client.api.exec_inspect(execution["Id"])
        except DockerException as exc:
            stream_error.append(exc)

    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    worker.join(deadline)
    timed_out = worker.is_alive()
    killed_container = False
    if timed_out:
        current = client.api.exec_inspect(execution["Id"])
        pid = int(current.get("Pid") or 0)
        if pid > 0:
            try:
                killer = client.api.exec_create(
                    container.id,
                    ["/bin/bash", "-c", f"kill -KILL -- -{pid}"],
                    user=user,
                )
                client.api.exec_start(killer["Id"])
            except DockerException:
                pass
        worker.join(5)
        if worker.is_alive():
            container.kill()
            killed_container = True
            worker.join(5)
    if stream_error:
        raise stream_error[0]
    if not inspection:
        inspection = client.api.exec_inspect(execution["Id"])
    output = b"".join(stdout_parts)
    error = b"".join(stderr_parts)
    exit_code = inspection.get("ExitCode")
    return {
        "returncode": int(exit_code if exit_code is not None else -1),
        "stdout": output,
        "stderr": error,
        "stdout_truncated": stdout_size >= capture_limit,
        "stderr_truncated": stderr_size >= capture_limit,
        "timed_out": timed_out,
        "container_killed": killed_container,
    }

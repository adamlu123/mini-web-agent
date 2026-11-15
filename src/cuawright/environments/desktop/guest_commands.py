from __future__ import annotations

from typing import Any

import requests


GUEST_TIMEOUT_KILL_GRACE_SECONDS = 5
GUEST_SERVER_TIMEOUT_MARGIN_SECONDS = 10


def _is_command_specific_execute_server_error(
    status_code: int,
    message: str,
) -> bool:
    if status_code != 500:
        return False
    import re

    utf8_output_decode_error = re.compile(
        r"^'utf-8' codec can't decode (?:byte 0x[0-9a-f]{2} in position \d+"
        r"|bytes in position \d+-\d+): .+$",
        re.IGNORECASE | re.DOTALL,
    )
    execute_subprocess_timeout = re.compile(
        r"^Command .+ timed out after \d+(?:\.\d+)? seconds$",
        re.DOTALL,
    )
    return bool(
        utf8_output_decode_error.fullmatch(message)
        or execute_subprocess_timeout.fullmatch(message)
    )


def run_guest_command(
    controller: Any,
    script: str,
    timeout: int,
    *,
    request_module: Any = requests,
) -> dict[str, Any]:
    """Execute Bash through a desktop guest's process-safe /execute endpoint."""

    server_timeout = (
        timeout + GUEST_TIMEOUT_KILL_GRACE_SECONDS + GUEST_SERVER_TIMEOUT_MARGIN_SECONDS
    )
    payload = {
        "command": [
            "/usr/bin/timeout",
            "--verbose",
            "--signal=TERM",
            f"--kill-after={GUEST_TIMEOUT_KILL_GRACE_SECONDS}s",
            f"{timeout}s",
            "/bin/bash",
            "-c",
            script,
        ],
        "shell": False,
        "timeout": server_timeout,
    }
    try:
        response = request_module.post(
            controller.http_server + "/execute",
            json=payload,
            timeout=server_timeout + 10,
        )
    except request_module.ReadTimeout:
        return {
            "status": "error",
            "output": "",
            "error": (
                "Guest execution service did not respond after the process-safe "
                f"timeout window ({server_timeout} seconds)."
            ),
            "returncode": -1,
            "infrastructure_error": True,
        }
    except request_module.RequestException as exc:
        return {
            "status": "error",
            "output": "",
            "error": str(exc),
            "returncode": -1,
            "infrastructure_error": True,
        }

    try:
        result = response.json()
    except ValueError:
        return {
            "status": "error",
            "output": "",
            "error": response.text[:4000],
            "returncode": -1,
            "infrastructure_error": response.status_code != 200,
        }
    result.setdefault("output", "")
    result.setdefault("error", result.get("message", ""))
    result.setdefault("returncode", -1)
    returncode = result.get("returncode")
    if response.status_code != 200:
        result["status"] = "error"
        endpoint_message = str(result.get("error") or "").strip()
        result["infrastructure_error"] = not (
            _is_command_specific_execute_server_error(
                response.status_code,
                endpoint_message,
            )
        )
        result["error"] = (
            f"Guest /execute endpoint returned HTTP {response.status_code}."
            + (f" {endpoint_message}" if endpoint_message else "")
        )
        if not result["infrastructure_error"]:
            result["error"] += (
                " The command may have changed state; inspect the current state "
                "before retrying a state-changing action."
            )
        return result
    error_text = str(result.get("error") or "")
    timeout_signaled = any(
        "timeout: sending signal" in line for line in error_text.splitlines()
    )
    if returncode in (124, 137) and timeout_signaled:
        result["status"] = "error"
        result["timed_out"] = True
        result["infrastructure_error"] = False
        message = (
            f"Script execution timed out after {timeout} seconds; "
            "the command process group was terminated."
        )
        result["error"] = (
            f"{error_text.rstrip()}\n{message}" if error_text.strip() else message
        )
        return result
    result["status"] = "success" if returncode == 0 else "error"
    result["infrastructure_error"] = False
    if returncode != 0 and not str(result.get("error") or "").strip():
        result["error"] = f"Command exited with return code {returncode}."
    return result


__all__ = [
    "GUEST_SERVER_TIMEOUT_MARGIN_SECONDS",
    "GUEST_TIMEOUT_KILL_GRACE_SECONDS",
    "run_guest_command",
]

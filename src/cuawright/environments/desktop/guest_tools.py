from __future__ import annotations

import base64
import hashlib
import json
import logging
import textwrap
from typing import Any

from cuawright.environments.control_clients import (
    CLIENT_FILENAMES,
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
    CLIENT_SOURCES,
)

_PASSWORD_SUDO_WRAPPER = """
import subprocess
import sys

password, root_program, payload = sys.argv[1:4]
completed = subprocess.run(
    ["sudo", "-S", "-p", "", "python3", "-c", root_program, payload],
    input=password + "\\n",
    text=True,
    capture_output=True,
    check=False,
)
if completed.returncode != 0:
    raise RuntimeError(
        "root control-client installation failed: "
        + (completed.stderr.strip() or completed.stdout.strip())
    )
"""


def write_guest_file(
    controller: Any,
    path: str,
    content: bytes,
    mode: int,
) -> None:
    encoded = base64.b64encode(content).decode("ascii")
    script = textwrap.dedent(
        f"""
        import base64
        import os
        from pathlib import Path

        path = Path({path!r}).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode({encoded!r}))
        os.chmod(path, {mode})
        print(path)
        """
    )
    result = controller.run_python_script(script, timeout=60)
    if not result or result.get("status") != "success":
        raise RuntimeError(f"Failed to write guest file {path}: {result}")


def _control_client_material() -> tuple[str, dict[str, dict[str, int | str]]]:
    sources = dict(CLIENT_SOURCES)
    expected_hashes = dict(CLIENT_SHA256)
    if {
        name: hashlib.sha256(data).hexdigest() for name, data in sources.items()
    } != expected_hashes:
        raise RuntimeError("Packaged control-client sources do not match their hashes")

    encoded_payload = base64.b64encode(
        json.dumps(
            {
                "install_root": CLIENT_INSTALL_ROOT,
                "clients": {
                    name: base64.b64encode(data).decode("ascii")
                    for name, data in sources.items()
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    expected = {
        name: {
            "sha256": expected_hashes[name],
            "uid": 0,
            "gid": 0,
            "mode": 0o555,
        }
        for name in CLIENT_FILENAMES
    }
    return encoded_payload, expected


def _control_client_verifier(encoded_payload: str) -> str:
    return textwrap.dedent(
        f"""
        import base64
        import json
        import os
        import stat
        payload = json.loads(base64.b64decode({encoded_payload!r}))
        root = payload["install_root"]
        expected_names = set(payload["clients"])
        found_names = {{
            name
            for name in os.listdir(root)
            if name.endswith(".py")
        }}
        if found_names != expected_names:
            raise RuntimeError(
                "control-client file set mismatch: "
                + json.dumps(
                    {{
                        "expected": sorted(expected_names),
                        "found": sorted(found_names),
                    }},
                    sort_keys=True,
                )
            )
        observed = {{}}
        for name in payload["clients"]:
            path = os.path.join(root, name)
            info = os.stat(path, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError(path + " is not a regular file")
            observed[name] = {{
                "sha256": __import__("hashlib").sha256(
                    open(path, "rb").read()
                ).hexdigest(),
                "uid": info.st_uid,
                "gid": info.st_gid,
                "mode": stat.S_IMODE(info.st_mode),
            }}
        print(json.dumps(observed, sort_keys=True, separators=(",", ":")))
        """
    )


def _observed_control_clients(result: Any) -> dict[str, Any] | None:
    if not result or result.get("status") != "success":
        return None
    output = result.get("output")
    if not isinstance(output, str):
        return None
    try:
        observed = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None
    return observed if isinstance(observed, dict) else None


def control_clients_ready(controller: Any) -> bool:
    encoded_payload, expected = _control_client_material()
    result = controller.run_python_script(
        _control_client_verifier(encoded_payload),
        timeout=90,
    )
    return _observed_control_clients(result) == expected


def provision_control_clients(
    controller: Any,
    setup_controller: Any | None = None,
) -> None:
    encoded_payload, expected = _control_client_material()
    root_program = r"""
import base64
import json
import os
import shutil
import stat
import sys
from pathlib import Path

payload = json.loads(base64.b64decode(sys.argv[1]))
if shutil.which("python") is None:
    python_link = Path("/usr/local/bin/python")
    python_link.parent.mkdir(parents=True, exist_ok=True)
    if python_link.is_symlink():
        python_link.unlink()
    elif python_link.exists():
        raise RuntimeError("unusable /usr/local/bin/python already exists")
    python_link.symlink_to(Path(sys.executable).resolve())
python_path = Path(shutil.which("python")).resolve()
python_info = python_path.stat()
if python_info.st_uid != 0 or stat.S_IMODE(python_info.st_mode) & 0o022:
    raise RuntimeError("python executable is not root-owned and non-writable")
root = Path(payload["install_root"])
root.mkdir(parents=True, exist_ok=True)
os.chown(root, 0, 0)
os.chmod(root, 0o755)
expected_names = set(payload["clients"])
for existing in root.glob("*.py"):
    if existing.name not in expected_names:
        existing.unlink()
for name, encoded in payload["clients"].items():
    if "/" in name or name in {"", ".", ".."}:
        raise ValueError("invalid control-client filename")
    path = root / name
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o555)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError("control-client destination is not a regular file")
        data = base64.b64decode(encoded, validate=True)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while installing control client")
            view = view[written:]
        os.fsync(fd)
        os.fchown(fd, 0, 0)
        os.fchmod(fd, 0o555)
    finally:
        os.close(fd)
"""
    installer = textwrap.dedent(
        f"""
        import subprocess

        root_program = {root_program!r}
        completed = subprocess.run(
            ["sudo", "-n", "python3", "-c", root_program, {encoded_payload!r}],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "root control-client installation failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        """
    )

    def install_with_password() -> None:
        if setup_controller is None:
            raise RuntimeError(
                "Password-based control-client installation is unavailable"
            )
        privileged_command = [
            "python3",
            "-c",
            _PASSWORD_SUDO_WRAPPER,
            "{CLIENT_PASSWORD}",
            root_program,
            encoded_payload,
        ]
        setup_logger = logging.getLogger("desktopenv.setup")
        logging_was_disabled = setup_logger.disabled
        setup_logger.disabled = True
        try:
            setup_controller.execute(
                privileged_command,
                shell=False,
                quiet=True,
                timeout=90,
            )
        finally:
            setup_logger.disabled = logging_was_disabled

    result = controller.run_python_script(installer, timeout=90)
    if not result or result.get("status") != "success":
        failure = json.dumps(result, ensure_ascii=False, default=str).lower()
        if setup_controller is None or "sudo: a password is required" not in failure:
            raise RuntimeError(f"Failed to provision control clients: {result}")
        install_with_password()

    verifier = _control_client_verifier(encoded_payload)
    result = controller.run_python_script(verifier, timeout=90)
    if not result or result.get("status") != "success":
        failure = json.dumps(result, ensure_ascii=False, default=str).lower()
        missing_root = CLIENT_INSTALL_ROOT.lower() in failure and (
            "filenotfounderror" in failure or "no such file or directory" in failure
        )
        if setup_controller is not None and missing_root:
            install_with_password()
            result = controller.run_python_script(verifier, timeout=90)
    if not result or result.get("status") != "success":
        raise RuntimeError(f"Failed to verify provisioned control clients: {result}")
    output = result.get("output")
    if not isinstance(output, str):
        raise RuntimeError("Control-client provisioning returned invalid output")
    try:
        observed = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Control-client provisioning returned malformed output: {output!r}"
        ) from exc
    if observed != expected:
        raise RuntimeError(
            "Provisioned control clients failed hash/ownership/mode verification: "
            f"{observed}"
        )


__all__ = [
    "control_clients_ready",
    "provision_control_clients",
    "write_guest_file",
]

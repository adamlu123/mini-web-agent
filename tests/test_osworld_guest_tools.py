from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from cuawright.environments.control_clients import (
    CLIENT_FILENAMES,
    CLIENT_SHA256,
)
from cuawright.environments.desktop.guest_tools import control_clients_ready
from cuawright.environments.osworld.guest_tools import (
    _PASSWORD_SUDO_WRAPPER,
    KNOWLEDGE_FILES,
    OSW_SCRIPT,
    _provision_control_clients,
)


class CompilingController:
    def __init__(
        self,
        *,
        fail_sudo: bool = False,
        missing_root_once: bool = False,
    ) -> None:
        self.fail_sudo = fail_sudo
        self.missing_root_once = missing_root_once
        self.calls = 0

    def run_python_script(self, script: str, timeout: int = 90):
        self.calls += 1
        compile(script, "<guest-control-client-installer>", "exec")
        if self.fail_sudo and self.calls == 1:
            return {
                "status": "error",
                "output": "",
                "error": "sudo: a password is required",
            }
        if self.missing_root_once and self.calls == 2:
            return {
                "status": "error",
                "output": "",
                "error": (
                    "FileNotFoundError: [Errno 2] No such file or directory: "
                    "'/opt/cuawright-tools'"
                ),
            }
        observed = {
            name: {
                "sha256": CLIENT_SHA256[name],
                "uid": 0,
                "gid": 0,
                "mode": 0o555,
            }
            for name in CLIENT_FILENAMES
        }
        return {
            "status": "success",
            "output": json.dumps(observed, sort_keys=True),
            "error": "",
        }


class RecordingSetupController:
    def __init__(self) -> None:
        self.calls = []
        self.logger_disabled_during_execute = False

    def execute(self, command, **kwargs) -> None:
        self.logger_disabled_during_execute = logging.getLogger(
            "desktopenv.setup"
        ).disabled
        self.calls.append((command, kwargs))


class OSWorldGuestToolsTests(unittest.TestCase):
    def test_osw_script_lists_shows_and_falls_back_on_zero_and_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            kb = home / ".local" / "share" / "osw" / "kb"
            kb.mkdir(parents=True)
            (kb / "browser.md").write_text(
                textwrap.dedent(
                    """\
                    # Browser
                    ## Existing session
                    Chrome session details.
                    ```bash
                    # fenced comment is not a heading
                    ```
                    Second complete line.
                    ## Playwright
                    Playwright attachment details.
                    """
                ),
                encoding="utf-8",
            )
            script = home / "osw"
            script.write_text(OSW_SCRIPT, encoding="utf-8")
            environment = {**os.environ, "HOME": str(home)}

            listed = subprocess.run(
                [sys.executable, str(script), "--list"],
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("browser.md", listed.stdout)
            self.assertIn("Existing session", listed.stdout)
            self.assertNotIn("fenced comment", listed.stdout)

            shown = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--show",
                    "browser.md",
                    "--heading",
                    "Playwright",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Playwright attachment details", shown.stdout)
            self.assertNotIn("Chrome session details", shown.stdout)

            fallback = subprocess.run(
                [sys.executable, str(script), "Chrome Playwright"],
                env=environment,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("browser.md", fallback.stdout)
            self.assertIn("Full section: osw --show", fallback.stdout)
            self.assertIn("Second complete line.", fallback.stdout)
            self.assertNotIn("[section truncated", fallback.stdout)

    def test_browser_knowledge_prefers_existing_session_with_recovery(self) -> None:
        browser = KNOWLEDGE_FILES["browser.md"]
        self.assertIn("Inspect the existing session first", browser)
        self.assertIn("Recover only when the normal session is unavailable", browser)
        self.assertIn("/home/user/.local/share/osw/venv/bin/python", browser)
        self.assertIn("select the task page from its URL or title", browser)
        self.assertIn("page.bring_to_front()", browser)
        self.assertIn("Accessibility.getFullAXTree", browser)
        self.assertIn("real navigation and locator-driven clicks", browser)
        self.assertIn("bulk `page.content()` scraping", browser)
        self.assertNotIn("page = browser.contexts[0].pages[0]", browser)

    def test_control_client_installer_is_valid_guest_python(self) -> None:
        self.assertEqual(
            CLIENT_FILENAMES,
            (
                "binder.py",
                "critic.py",
                "verifier.py",
                "image_show.py",
                "ask_user.py",
                "submit.py",
            ),
        )
        _provision_control_clients(CompilingController())

    def test_control_client_readiness_requires_exact_verified_state(self) -> None:
        self.assertTrue(control_clients_ready(CompilingController()))

        class MissingController:
            def run_python_script(self, script: str, timeout: int = 90):
                compile(script, "<guest-control-client-verifier>", "exec")
                return {
                    "status": "error",
                    "output": "",
                    "error": "FileNotFoundError",
                }

        self.assertFalse(control_clients_ready(MissingController()))

    def test_password_fallback_uses_setup_controller_placeholder(self) -> None:
        controller = CompilingController(fail_sudo=True)
        setup_controller = RecordingSetupController()

        _provision_control_clients(controller, setup_controller)

        self.assertEqual(controller.calls, 2)
        self.assertEqual(len(setup_controller.calls), 1)
        command, options = setup_controller.calls[0]
        self.assertEqual(options, {"shell": False, "quiet": True, "timeout": 90})
        self.assertEqual(
            command[:4], ["python3", "-c", _PASSWORD_SUDO_WRAPPER, "{CLIENT_PASSWORD}"]
        )
        self.assertEqual(len(command), 6)
        self.assertTrue(setup_controller.logger_disabled_during_execute)
        self.assertFalse(logging.getLogger("desktopenv.setup").disabled)

    def test_missing_install_root_is_reinstalled_once_with_password(self) -> None:
        controller = CompilingController(missing_root_once=True)
        setup_controller = RecordingSetupController()

        _provision_control_clients(controller, setup_controller)

        self.assertEqual(controller.calls, 3)
        self.assertEqual(len(setup_controller.calls), 1)
        command, options = setup_controller.calls[0]
        self.assertEqual(options, {"shell": False, "quiet": True, "timeout": 90})
        self.assertEqual(
            command[:4],
            ["python3", "-c", _PASSWORD_SUDO_WRAPPER, "{CLIENT_PASSWORD}"],
        )

    def test_setup_replacement_keeps_hostile_passwords_inert(self) -> None:
        controller = CompilingController(fail_sudo=True)
        setup_controller = RecordingSetupController()
        _provision_control_clients(controller, setup_controller)
        template, _ = setup_controller.calls[0]
        passwords = (
            "single'quote",
            "semi;colon",
            "$(touch should-not-run)",
            "spaces and\ttabs",
            "line one\nline two",
        )

        for password in passwords:
            with self.subTest(password=password):
                command = [
                    argument.replace("{CLIENT_PASSWORD}", password)
                    for argument in template
                ]
                self.assertEqual(len(command), 6)
                self.assertEqual(command[:3], ["python3", "-c", _PASSWORD_SUDO_WRAPPER])
                self.assertEqual(command[3], password)

                completed = subprocess.CompletedProcess([], 0, "", "")
                with (
                    patch.object(sys, "argv", ["wrapper", *command[3:]]),
                    patch("subprocess.run", return_value=completed) as run,
                ):
                    exec(compile(_PASSWORD_SUDO_WRAPPER, "<sudo-wrapper>", "exec"), {})

                run.assert_called_once_with(
                    [
                        "sudo",
                        "-S",
                        "-p",
                        "",
                        "python3",
                        "-c",
                        command[4],
                        command[5],
                    ],
                    input=password + "\n",
                    text=True,
                    capture_output=True,
                    check=False,
                )


if __name__ == "__main__":
    unittest.main()

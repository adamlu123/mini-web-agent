from __future__ import annotations

import contextlib
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cuawright.environments.base import CommandResult, ControlFileAccessError
from cuawright.environments.osworld.environment import (
    GuestWorkspace,
    OSWorldBenchmarkPlugin,
)
from cuawright.environments.osworld.lifecycle import (
    OSWorldEnvironmentIdentity,
    OSWorldLifecycle,
    OSWorldLifecycleConfig,
    OSWorldTaskIdentity,
    is_inspection_command,
)
from cuawright.run.osworld import OSWorldRunner


LOGICAL_WORKSPACE = "/workspace/.agent"


class OSWorldInspectionCommandTests(unittest.TestCase):
    def test_arbitrary_python_and_shell_composition_are_not_inspection(self) -> None:
        commands = (
            "python -c \"open('/workspace/result.txt', 'w').write('changed')\"",
            "cat /workspace/result.txt && touch /workspace/changed",
            "find /workspace {-delete,-print}",
            "find /workspace *",
            "find /workspace ?delete",
            "find /workspace [-]delete",
            "find /workspace -type f -fprint0 /workspace/files.txt",
            "ffprobe -report /workspace/video.mp4",
            "unknown-inspector /workspace/result.txt",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(is_inspection_command(command))

    def test_explicit_read_only_queries_are_inspection(self) -> None:
        commands = (
            "cat /workspace/result.txt",
            "find /workspace -maxdepth 2 -type f",
            "find /workspace -name '*' -print",
            'osw "chrome playwright cdp" --limit 4',
            "osw --list",
            "osw --show browser.md",
            'osw --show browser.md --heading "Existing session"',
            "sqlite3 /workspace/state.db .schema",
            "ffprobe -show_format -show_streams /workspace/video.mp4",
            "mediainfo --Output=JSON /workspace/video.mp4",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(is_inspection_command(command))

    def test_osw_show_rejects_unsafe_or_malformed_paths(self) -> None:
        commands = (
            "osw --show ../browser.md",
            "osw --show /tmp/browser.md",
            "osw --show browser.txt",
            "osw --show browser.md --heading",
            "osw --heading Browser",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(is_inspection_command(command))


class FakeController:
    def __init__(self) -> None:
        self.scripts: list[tuple[str, int]] = []
        self.replace_observations: list[bytes] = []

    def run_python_script(self, script: str, timeout: int = 90):
        self.scripts.append((script, timeout))
        output = io.StringIO()
        original_replace = os.replace

        def observe_replace(source, destination, **kwargs):
            directory_fd = kwargs["dst_dir_fd"]
            try:
                probe_fd = os.open(destination, os.O_RDONLY, dir_fd=directory_fd)
            except FileNotFoundError:
                self.replace_observations.append(b"<missing>")
            else:
                try:
                    self.replace_observations.append(os.read(probe_fd, 1024))
                finally:
                    os.close(probe_fd)
            return original_replace(source, destination, **kwargs)

        try:
            with (
                contextlib.redirect_stdout(output),
                patch("os.replace", side_effect=observe_replace),
            ):
                exec(compile(script, "<guest-bridge>", "exec"), {})
        except Exception as exc:
            return {
                "status": "error",
                "output": output.getvalue(),
                "error": str(exc),
            }
        return {"status": "success", "output": output.getvalue(), "error": ""}


class FakeDesktopEnv:
    def __init__(self, controller: FakeController) -> None:
        self.controller = controller
        self.is_environment_used = False
        self.events: list[str] = []

    def evaluate(self):
        self.events.append("env-evaluate")
        return {"score": 0.75}

    def close(self) -> None:
        self.events.append("env-close")


class OSWorldPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".osworld-plugin-",
            dir=Path(__file__).parent,
        )
        self.guest_workspace = Path(self.temporary.name)
        self.controller = FakeController()
        self.env = FakeDesktopEnv(self.controller)
        self.commands: list[tuple[str, int | None]] = []

        def execute(command: str, timeout: int | None) -> CommandResult:
            self.commands.append((command, timeout))
            return CommandResult(0, stdout="delegated")

        self.plugin = OSWorldBenchmarkPlugin(
            self.env,
            execute,
            workspace=GuestWorkspace(
                LOGICAL_WORKSPACE,
                self.guest_workspace.as_posix(),
            ),
            bridge_timeout=17,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_ordinary_command_is_delegated_unchanged(self) -> None:
        result = self.plugin.run_command("printf '%s' ready", timeout=12)

        self.assertEqual(result, CommandResult(0, stdout="delegated"))
        self.assertEqual(self.commands, [("printf '%s' ready", 12)])
        self.assertTrue(self.env.is_environment_used)

    def test_runner_preserves_native_command_fields_and_harness_metadata(
        self,
    ) -> None:
        runner = object.__new__(OSWorldRunner)
        runner.env = SimpleNamespace(controller=object(), is_environment_used=False)
        runner.tool_step = 0
        runner._tool_step_lock = threading.Lock()
        runner.config = SimpleNamespace(
            website_host_suffix="example.invalid",
            total_steps=222,
        )
        runner.trace = Mock()
        runner.trace.command_blob.return_value = ("command.txt.gz", "digest")
        output = "o" * 20_000
        error = "e" * 20_000
        payload = {
            "status": "error",
            "returncode": 7,
            "output": output,
            "error": error,
            "infrastructure_error": False,
            "timed_out": True,
        }

        with patch(
            "cuawright.run.osworld.run_guest_command",
            return_value=payload,
        ):
            result = runner._execute_command("printf inspect", 30)

        self.assertEqual(result.stdout, output)
        self.assertEqual(result.stderr, error)
        self.assertEqual(result.model_fields["status"], "error")
        self.assertTrue(result.model_fields["timed_out"])
        self.assertEqual(
            result.model_fields["_harness"],
            {
                "tool_step": 1,
                "configured_total_steps": 222,
                "stage": "terminal-orchestrator",
            },
        )

    def test_arbitrary_read_is_bounded_in_the_guest(self) -> None:
        source = self.guest_workspace / "ordinary.bin"
        source.write_bytes(b"0123456789")

        self.assertEqual(
            self.plugin.read_file(source.as_posix(), max_bytes=4),
            b"0123",
        )
        script, timeout = self.controller.scripts[-1]
        self.assertNotIn(source.as_posix(), script)
        self.assertEqual(timeout, 17)

    def test_control_read_maps_logical_path_and_rejects_traversal(self) -> None:
        reports = self.guest_workspace / "reports"
        reports.mkdir()
        (reports / "final.md").write_bytes(b"abcdefgh")

        self.assertEqual(
            self.plugin.read_control_file(
                f"{LOGICAL_WORKSPACE}/reports/final.md",
                workspace=LOGICAL_WORKSPACE,
                max_bytes=5,
            ),
            b"abcde",
        )
        script_count = len(self.controller.scripts)
        with self.assertRaises(ControlFileAccessError):
            self.plugin.read_control_file(
                f"{LOGICAL_WORKSPACE}/../outside",
                workspace=LOGICAL_WORKSPACE,
                max_bytes=10,
            )
        self.assertEqual(len(self.controller.scripts), script_count)

    def test_control_read_rejects_target_and_component_symlinks(self) -> None:
        outside = Path(self.temporary.name + "-outside.txt")
        outside.write_bytes(b"secret")
        self.addCleanup(outside.unlink)
        (self.guest_workspace / "target-link").symlink_to(outside)

        with self.assertRaises(ControlFileAccessError):
            self.plugin.read_control_file(
                f"{LOGICAL_WORKSPACE}/target-link",
                workspace=LOGICAL_WORKSPACE,
                max_bytes=20,
            )

        outside_directory = Path(self.temporary.name + "-outside-directory")
        outside_directory.mkdir()
        self.addCleanup(outside_directory.rmdir)
        (outside_directory / "report").write_bytes(b"secret")
        self.addCleanup((outside_directory / "report").unlink)
        (self.guest_workspace / "component-link").symlink_to(
            outside_directory,
            target_is_directory=True,
        )
        with self.assertRaises(ControlFileAccessError):
            self.plugin.read_control_file(
                f"{LOGICAL_WORKSPACE}/component-link/report",
                workspace=LOGICAL_WORKSPACE,
                max_bytes=20,
            )

    def test_atomic_write_replaces_regular_file_without_partial_state(self) -> None:
        state = self.guest_workspace / "state.json"
        state.write_bytes(b"old-state")

        self.plugin.write_control_file_atomic(
            f"{LOGICAL_WORKSPACE}/state.json",
            b"new-state",
            workspace=LOGICAL_WORKSPACE,
        )

        self.assertEqual(state.read_bytes(), b"new-state")
        self.assertEqual(self.controller.replace_observations, [b"old-state"])
        self.assertEqual(
            list(self.guest_workspace.glob(".state.json.agent-*")),
            [],
        )
        self.assertTrue(self.env.is_environment_used)

    def test_atomic_write_creates_a_new_regular_file(self) -> None:
        state = self.guest_workspace / "new-state.json"

        self.plugin.write_control_file_atomic(
            f"{LOGICAL_WORKSPACE}/new-state.json",
            b"first-state",
            workspace=LOGICAL_WORKSPACE,
        )

        self.assertEqual(state.read_bytes(), b"first-state")
        self.assertEqual(self.controller.replace_observations, [b"<missing>"])

    def test_atomic_write_rejects_symlink_target(self) -> None:
        outside = Path(self.temporary.name + "-write-outside.txt")
        outside.write_bytes(b"unchanged")
        self.addCleanup(outside.unlink)
        (self.guest_workspace / "state.json").symlink_to(outside)

        with self.assertRaises(ControlFileAccessError):
            self.plugin.write_control_file_atomic(
                f"{LOGICAL_WORKSPACE}/state.json",
                b"replacement",
                workspace=LOGICAL_WORKSPACE,
            )
        self.assertEqual(outside.read_bytes(), b"unchanged")

    def test_workspace_argument_must_match_configured_mapping(self) -> None:
        with self.assertRaises(ControlFileAccessError):
            self.plugin.read_control_file(
                f"{LOGICAL_WORKSPACE}/state.json",
                workspace="/different/.agent",
                max_bytes=10,
            )


class OSWorldLifecycleTests(unittest.TestCase):
    def test_identity_workspace_and_lifecycle_delegation(self) -> None:
        controller = FakeController()
        env = FakeDesktopEnv(controller)
        phase_calls: list[int] = []
        lifecycle = OSWorldLifecycle(
            desktop_env=env,
            config=OSWorldLifecycleConfig(
                task=OSWorldTaskIdentity("task-007", "v2026.06.24"),
                environment=OSWorldEnvironmentIdentity(
                    "docker",
                    "/images/osworld.qcow2",
                    "init_state",
                ),
                workspace=GuestWorkspace(
                    LOGICAL_WORKSPACE,
                    "/home/user/.agent",
                ),
            ),
            phase_advance=lambda index: phase_calls.append(index) or f"phase-{index}",
        )

        self.assertEqual(lifecycle.advance_phase(2), "phase-2")
        self.assertEqual(phase_calls, [2])
        self.assertEqual(lifecycle.evaluate(), {"score": 0.75})
        lifecycle.close()
        self.assertEqual(env.events, ["env-evaluate", "env-close"])
        self.assertEqual(
            lifecycle.config.workspace.logical_path,
            LOGICAL_WORKSPACE,
        )


if __name__ == "__main__":
    unittest.main()

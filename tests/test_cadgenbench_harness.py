from __future__ import annotations

import base64
from collections import deque
import hashlib
import json
import shutil
import os
import subprocess
import sys
import threading
import time
import tarfile
import unittest
import uuid
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cuawright.environments.cadgenbench.container_tools import run_prediction
from cuawright.environments.cadgenbench.cad_mcp_cli import (
    _load_arguments,
    _normalize_tool_result,
)
import cuawright.environments.cadgenbench.environment as cadgen_environment
from cuawright.environments.cadgenbench.environment import CadGenBenchEnvironment
from cuawright.environments.cadgenbench.errors import CadGenBenchInfrastructureError
from cuawright.environments.cadgenbench.packaging import package_submission
from cuawright.environments.cadgenbench.plugin import (
    CadGenBenchPlugin,
    CadGenBenchPluginConfig,
    _CONTROL_READ_PROGRAM,
    _CONTROL_WRITE_PROGRAM,
    _is_inspection_command,
    create_cadgenbench_plugin,
)
from cuawright.environments.cadgenbench.task import (
    DEFAULT_BENCHMARK_COMMIT,
    DEFAULT_DATASET_REVISION,
    discover_tasks,
    load_manifest,
    validate_fixture_id,
)
from cuawright.environments.base import CommandResult, ControlFileAccessError

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDAT\x08\xd7c\xf8\xff\xff?\x00\x05\xfe"
    b"\x02\xfeA\x0b\xe6\xa5\x00\x00\x00\x00IEND\xaeB`\x82"
)
SCRATCH_ROOT = Path(__file__).resolve().parents[1] / ".test-scratch"


@contextmanager
def scratch_dir() -> Path:
    root = SCRATCH_ROOT / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _metadata_file(dataset_root: Path, relative_path: str) -> Path:
    return (
        dataset_root
        / ".cache"
        / "huggingface"
        / "download"
        / f"{relative_path}.metadata"
    )


def _write_dataset_file(
    dataset_root: Path,
    relative_path: str,
    data: bytes,
    revision: str = DEFAULT_DATASET_REVISION,
) -> None:
    path = dataset_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    metadata = _metadata_file(dataset_root, relative_path)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        f"{revision}\nsha256\n0\n",
        encoding="utf-8",
    )


def _write_generation_fixture(dataset_root: Path) -> None:
    _write_dataset_file(
        dataset_root,
        "101/description.yaml",
        (
            "description: >\n"
            "  Reproduce the geometry as accurately as possible from the drawing.\n\n"
            "input_files:\n"
            "  - input.png\n"
            "  - input2.png\n\n"
            "input_type: text+image\n"
        ).encode("utf-8"),
    )
    _write_dataset_file(dataset_root, "101/input.png", PNG_1X1)
    _write_dataset_file(dataset_root, "101/input2.png", PNG_1X1)


def _write_editing_fixture(dataset_root: Path) -> None:
    _write_dataset_file(
        dataset_root,
        "201/description.yaml",
        (
            "description: >\n"
            "  Move the slot 2mm toward +X.\n\n"
            "task_type: editing\n"
            "input_files:\n"
            "  - input.step\n\n"
            "input_type: text+step\n"
        ).encode("utf-8"),
    )
    _write_dataset_file(dataset_root, "201/input.step", b"STEP")
    _write_dataset_file(
        dataset_root, "201/edit_description.txt", b"Move the slot 2mm toward +X.\n"
    )
    _write_dataset_file(dataset_root, "201/input.mesh.npz", b"NPZ")
    for name in ("iso.png", "front.png", "top.png", "right.png"):
        _write_dataset_file(dataset_root, f"201/renders/{name}", PNG_1X1)


def _docker_client_or_skip(test_case: unittest.TestCase):
    try:
        import docker
    except ModuleNotFoundError as exc:  # pragma: no cover - local deps unavailable
        test_case.skipTest(f"docker SDK unavailable: {exc}")
    client = docker.from_env()
    try:
        client.ping()
    except Exception as exc:  # noqa: BLE001 - runtime availability
        client.close()
        test_case.skipTest(f"Docker daemon unavailable: {exc}")
    return client


def _local_test_image(client, test_case: unittest.TestCase) -> str:
    for image in (
        "cuawright-cadgenbench:33304cf",
        "cuawright-cadgenbench:33304cf-mcp359",
        "python:3.12-slim-bookworm",
    ):
        try:
            client.images.get(image)
            return image
        except Exception:
            continue
    client.close()
    test_case.skipTest("No suitable local Docker image for observability smoke test")
    raise AssertionError("unreachable")


def _observable_env_for_smoke(
    client,
    *,
    root: Path,
    image: str,
    fixture_id: str,
) -> CadGenBenchEnvironment:
    env = object.__new__(CadGenBenchEnvironment)
    env.client = client
    env.image = image
    env.image_ref = image
    env.image_id = image
    env.benchmark_commit = DEFAULT_BENCHMARK_COMMIT
    env.run_dir = root
    env.task = SimpleNamespace(fixture_dir=root / "fixture", fixture_id=fixture_id)
    env.task.fixture_dir.mkdir(parents=True, exist_ok=True)
    env.workspace_dir = root / "workspace"
    env.workspace_dir.mkdir(parents=True, exist_ok=True)
    env.latest_valid_dir = root / "latest-valid"
    env.latest_valid_dir.mkdir(parents=True, exist_ok=True)
    env.latest_valid_output = env.latest_valid_dir / "output.step"
    env.checkpoints_dir = root / "checkpoints"
    env.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    env.checkpoint_history_path = root / "checkpoint-history.json"
    env.container_telemetry_path = root / "container-telemetry.jsonl"
    env.container_failure_path = root / "container-failure.json"
    env.container_lifecycle_dir = root / "container-lifecycle"
    env.container_lifecycle_dir.mkdir(parents=True, exist_ok=True)
    env.checkpoint_history = []
    env.mcp_restart_history = []
    env.mcp_telemetry = None
    env._candidate_signatures = {}
    env._container_lock = threading.RLock()
    env._container_generation = 0
    env._current_tool_name = None
    env._current_stage = None
    env._monitor_stop = threading.Event()
    env._monitor_thread = None
    env._recent_container_samples = deque(
        maxlen=cadgen_environment.CONTAINER_TELEMETRY_SAMPLE_LIMIT
    )
    env._peak_memory_usage_bytes = 0
    env._peak_memory_limit_bytes = 0
    env._peak_pids = 0
    env._container_exit_states = []
    env._container_shutdown_expected = False
    env._harness_kill_events = []
    env.provenance = {"fixture_id": fixture_id, "task_type": "generation"}
    env.controller = env
    env.user_simulator = None
    env.action_history = []
    env.is_environment_used = False
    env.enable_proxy = False
    env.instruction = ""
    env._frozen = False
    env._removed = False
    env._volume_removed = False
    env.workspace_volume = client.volumes.create(
        name=f"cuawright-cadgenbench-{fixture_id}-{uuid.uuid4().hex[:12]}",
        driver="local",
        driver_opts={
            "type": "tmpfs",
            "device": "tmpfs",
            "o": f"size={2 * 1024 * 1024},uid={os.getuid()},gid={os.getgid()},mode=0700",
        },
        labels={
            "cuawright.component": "cadgenbench-workspace",
            "cuawright.fixture_id": fixture_id,
        },
    )

    def start_smoke_container():
        kwargs = env._agent_container_kwargs()
        kwargs["command"] = ["sleep", "infinity"]
        return client.containers.run(env.image, **kwargs)

    env._start_agent_container = start_smoke_container
    container = env._start_agent_container()
    env._set_current_container(container, reason="smoke_start")
    env._start_container_monitor()
    return env


class CadGenBenchTaskLoadingTests(unittest.TestCase):
    def test_loads_generation_and_editing_fixtures(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_generation_fixture(root)
            _write_editing_fixture(root)

            manifest = discover_tasks(
                dataset_dir=root,
                benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
            )

            self.assertEqual(
                [task.fixture_id for task in manifest.tasks], ["101", "201"]
            )
            generation, editing = manifest.tasks
            self.assertEqual(generation.task_type, "generation")
            self.assertEqual(generation.input_type, "text+image")
            self.assertEqual(
                [path.name for path in generation.image_inputs],
                ["input.png", "input2.png"],
            )
            self.assertEqual(editing.task_type, "editing")
            self.assertEqual(
                editing.input_step.name if editing.input_step else None, "input.step"
            )
            self.assertEqual(
                [path.name for path in editing.input_renders],
                ["front.png", "top.png", "right.png", "iso.png"],
            )

    def test_task_instruction_uses_public_comparator_prompt_policy(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_generation_fixture(root)
            _write_editing_fixture(root)
            manifest = discover_tasks(
                dataset_dir=root,
                benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
            )

            generation, editing = manifest.tasks
            self.assertIn("drawing `input.png`", generation.instruction)
            self.assertIn("input2.png", generation.instruction)
            self.assertEqual(
                generation.staged_workspace_inputs,
                (("input.png", "input.png"), ("input2.png", "input2.png")),
            )
            self.assertEqual(
                generation.initial_actor_image_paths,
                ("/input/input.png", "/input/input2.png"),
            )

            self.assertIn(
                "Edit the starting solid in `input.step`", editing.instruction
            )
            self.assertIn("Move the slot 2mm toward +X.", editing.instruction)
            self.assertEqual(
                editing.initial_actor_image_paths,
                (
                    "/input/renders/front.png",
                    "/input/renders/top.png",
                    "/input/renders/right.png",
                    "/input/renders/iso.png",
                ),
            )

    def test_rejects_unsafe_declared_paths(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_dataset_file(
                root,
                "101/description.yaml",
                (
                    "description: bad\n"
                    "input_files:\n"
                    "  - ../escape.png\n"
                    "input_type: text+image\n"
                ).encode("utf-8"),
            )
            with self.assertRaises(ValueError):
                discover_tasks(
                    dataset_dir=root,
                    benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
                )

    def test_load_manifest_detects_tampering(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_generation_fixture(root)
            manifest = discover_tasks(
                dataset_dir=root,
                benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
            )
            payload = {
                "benchmark": "cadgenbench",
                "dataset_dir": str(manifest.dataset_dir),
                "benchmark_commit": manifest.benchmark_commit,
                "dataset_revision": manifest.dataset_revision,
                "tasks": [task.to_manifest_entry() for task in manifest.tasks],
            }
            manifest_path = root / "panel.json"
            manifest_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            (root / "101" / "input.png").write_bytes(PNG_1X1 + b"tamper")

            with self.assertRaisesRegex(
                ValueError, "does not match the live dataset snapshot"
            ):
                load_manifest(manifest_path)

    def test_validate_fixture_id_rejects_escape(self) -> None:
        for value in ("../101", "101/extra", "", "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_fixture_id(value)


class CadGenBenchPluginTests(unittest.TestCase):
    def _environment(self):
        environment = SimpleNamespace(
            run_command=Mock(
                return_value={
                    "returncode": 0,
                    "output": "ok",
                    "error": "",
                    "checkpoint": {"status": "archived"},
                }
            ),
            get_file=Mock(return_value=PNG_1X1),
            client=Mock(),
            container=Mock(),
            _container_lock=threading.RLock(),
        )
        environment.container.put_archive.return_value = True
        return environment

    def test_mutating_mcp_commands_are_not_inspection(self) -> None:
        plugin = CadGenBenchPlugin(self._environment())
        commands = (
            "cad-mcp execute /workspace/build.py",
            "cad-mcp call export --json " '\'{"path":"/workspace/output.step"}\'',
            "cad-mcp call import_cad_file --json "
            '\'{"path":"/workspace/input.step"}\'',
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(plugin.run_command(command).inspected)

    def test_read_only_mcp_and_candidate_queries_are_inspection(self) -> None:
        commands = (
            "cad-mcp tools",
            "cad-mcp schema execute",
            "cad-mcp call measure --json '{}'",
            "cad-mcp call render_view --json " '\'{"direction":"iso"}\'',
            "cadgenbench-validate /workspace/output.step",
            "cadgenbench-render /workspace/output.step /workspace/render",
            "find /workspace -name '*' -print",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(_is_inspection_command(command))

    def test_candidate_generation_and_shell_composition_are_not_inspection(
        self,
    ) -> None:
        commands = (
            "cadgenbench-run /workspace/prediction.py " "--step /workspace/output.step",
            "cadgenbench-validate /workspace/output.step "
            "--json /workspace/validation.json",
            "cad-mcp --url http://example.invalid/mcp " "call measure --json '{}'",
            "cad-mcp tools && touch /workspace/changed",
            "find /workspace {-delete,-print}",
            "find /workspace *",
            "find /workspace ?delete",
            "find /workspace [-]delete",
            "find /workspace -type f -fprint0 /workspace/files.txt",
            "python /workspace/build.py",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(_is_inspection_command(command))

    def test_construction_api_and_command_delegation(self) -> None:
        environment = self._environment()
        config = CadGenBenchPluginConfig(command_timeout_seconds=41)

        plugin = create_cadgenbench_plugin(environment, config=config)
        result = plugin.run_command("cad-mcp tools")

        self.assertIsInstance(plugin, CadGenBenchPlugin)
        self.assertIs(config.build(environment).config, config)
        self.assertEqual(
            result,
            CommandResult(
                0,
                stdout="ok",
                inspected=True,
                model_fields={
                    "returncode": 0,
                    "output": "ok",
                    "error": "",
                    "checkpoint": {"status": "archived"},
                },
            ),
        )
        self.assertEqual(
            result.model_fields["checkpoint"],
            {"status": "archived"},
        )
        environment.run_command.assert_called_once_with("cad-mcp tools", 41)

    def test_command_infrastructure_failure_is_not_model_visible(self) -> None:
        environment = self._environment()
        environment.run_command.return_value = {
            "status": "error",
            "returncode": -1,
            "output": "",
            "error": "MCP restart limit exhausted",
            "infrastructure_error": True,
        }

        with self.assertRaisesRegex(
            CadGenBenchInfrastructureError,
            "restart limit",
        ):
            CadGenBenchPlugin(environment).run_command("cad-mcp tools")

    def test_arbitrary_image_read_is_bounded_and_delegated(self) -> None:
        environment = self._environment()
        plugin = CadGenBenchPlugin(environment)

        self.assertEqual(
            plugin.read_file("/workspace/render.png", max_bytes=len(PNG_1X1)),
            PNG_1X1,
        )
        environment.get_file.assert_called_once_with(
            "/workspace/render.png",
            max_bytes=len(PNG_1X1),
        )

        environment.get_file.return_value = PNG_1X1
        with self.assertRaises(OSError):
            plugin.read_file("/workspace/render.png", max_bytes=1)

    def test_control_read_uses_no_shell_trusted_bridge(self) -> None:
        environment = self._environment()
        plugin = CadGenBenchPlugin(environment)
        execution = {
            "returncode": 0,
            "stdout": b"request",
            "stderr": b"",
            "stdout_truncated": False,
        }

        with patch(
            "cuawright.environments.cadgenbench.plugin.run_exec",
            return_value=execution,
        ) as execute:
            data = plugin.read_control_file(
                "/workspace/.agent/requests/binder.md",
                workspace="/workspace/.agent",
                max_bytes=100,
            )

        self.assertEqual(data, b"request")
        command = execute.call_args.kwargs["command"]
        self.assertEqual(command[0], "/opt/cadgenbench-venv/bin/python")
        self.assertEqual(
            command[-3:], ["/workspace/.agent", "requests/binder.md", "100"]
        )
        self.assertNotIn("/bin/sh", command)
        self.assertNotIn("/bin/bash", command)
        self.assertEqual(
            execute.call_args.kwargs["user"],
            f"{os.getuid()}:{os.getgid()}",
        )
        environment.run_command.assert_not_called()

    def test_control_paths_reject_escape_before_container_access(self) -> None:
        environment = self._environment()
        plugin = CadGenBenchPlugin(environment)

        with patch("cuawright.environments.cadgenbench.plugin.run_exec") as execute:
            for path in (
                "/workspace/.agent/../outside",
                "/workspace/outside",
                "workspace/.agent/request.md",
            ):
                with self.subTest(path=path), self.assertRaises(ControlFileAccessError):
                    plugin.read_control_file(
                        path,
                        workspace="/workspace/.agent",
                        max_bytes=100,
                    )
            execute.assert_not_called()

    def test_control_helpers_reject_symlink_reads_and_parent_writes(self) -> None:
        with scratch_dir() as root:
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret").write_bytes(b"secret")
            (workspace / "linked").symlink_to(outside / "secret")

            read = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _CONTROL_READ_PROGRAM,
                    str(workspace),
                    "linked",
                    "100",
                ],
                check=False,
                capture_output=True,
            )
            self.assertNotEqual(read.returncode, 0)

            (workspace / "reports").symlink_to(outside, target_is_directory=True)
            write = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _CONTROL_WRITE_PROGRAM,
                    str(workspace),
                    "reports/state.json",
                    base64.b64encode(b"state").decode("ascii"),
                ],
                check=False,
                capture_output=True,
            )
            self.assertNotEqual(write.returncode, 0)
            self.assertFalse((outside / "state.json").exists())

    def test_control_write_uses_atomic_no_follow_bridge(self) -> None:
        environment = self._environment()
        plugin = CadGenBenchPlugin(environment)
        execution = {
            "returncode": 0,
            "stdout": b"",
            "stderr": b"",
            "stdout_truncated": False,
        }

        with patch(
            "cuawright.environments.cadgenbench.plugin.run_exec",
            return_value=execution,
        ) as execute:
            plugin.write_control_file_atomic(
                "/workspace/.agent/state.json",
                b'{"ok":true}',
                workspace="/workspace/.agent",
            )

        environment.container.put_archive.assert_not_called()
        command = execute.call_args.kwargs["command"]
        self.assertEqual(command[0], "/opt/cadgenbench-venv/bin/python")
        self.assertIn("os.O_NOFOLLOW", command[2])
        self.assertIn("os.replace", command[2])
        self.assertEqual(command[3], "/workspace/.agent")
        self.assertEqual(command[4], "state.json")
        self.assertEqual(
            base64.b64decode("".join(command[5:])),
            b'{"ok":true}',
        )
        environment.run_command.assert_not_called()


class CadMcpCliTests(unittest.TestCase):
    def test_load_arguments_requires_json_object(self) -> None:
        args = SimpleNamespace(json_arguments='["not", "an", "object"]', json_file=None)
        with self.assertRaisesRegex(ValueError, "JSON object"):
            _load_arguments(args)

    def test_normalize_tool_result_persists_images(self) -> None:
        image = SimpleNamespace(
            type="image",
            mimeType="image/png",
            data=__import__("base64").b64encode(PNG_1X1).decode("ascii"),
        )
        result = SimpleNamespace(
            isError=False,
            content=[
                SimpleNamespace(
                    type="text",
                    model_dump=lambda **kwargs: {"type": "text", "text": "ok"},
                ),
                image,
            ],
            structuredContent={"value": 1},
        )
        with (
            scratch_dir() as root,
            patch.dict(
                os.environ,
                {"CAD_MCP_IMAGE_DIR": str(root)},
            ),
        ):
            payload = _normalize_tool_result("render_view", result)
            self.assertEqual(payload["status"], "success")
            self.assertEqual(len(payload["images"]), 1)
            self.assertEqual(Path(payload["images"][0]).read_bytes(), PNG_1X1)


class CadGenBenchEnvironmentTests(unittest.TestCase):
    def test_command_environment_uses_migrated_cuawright_package(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)

        command_environment = environment._command_environment()

        self.assertEqual(
            command_environment["PYTHONPATH"],
            "/opt/cuawright:/opt/cadgenbench-source/src",
        )

    def test_start_agent_container_uses_hardened_config_with_existing_volume(
        self,
    ) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment.client = Mock()
        environment.client.containers.run.return_value = Mock()
        environment.image = "sha256:image"
        environment.workspace_volume = SimpleNamespace(name="workspace-volume")
        environment.task = SimpleNamespace(
            fixture_dir=Path("/fixture"), fixture_id="101"
        )

        environment._start_agent_container()

        kwargs = environment.client.containers.run.call_args.kwargs
        self.assertEqual(
            kwargs["command"][:5],
            [
                "/opt/cadgenbench-venv/bin/build123d-mcp",
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
            ],
        )
        self.assertEqual(kwargs["network_mode"], "none")
        self.assertTrue(kwargs["read_only"])
        self.assertEqual(kwargs["mem_limit"], "12g")
        self.assertEqual(kwargs["pids_limit"], 1024)
        self.assertEqual(kwargs["volumes"]["/fixture"]["mode"], "ro")
        self.assertEqual(kwargs["volumes"]["workspace-volume"]["bind"], "/workspace")
        self.assertEqual(kwargs["security_opt"], ["no-new-privileges:true"])
        self.assertEqual(kwargs["cap_drop"], ["ALL"])
        self.assertEqual(
            kwargs["tmpfs"]["/run/cadgenbench-control"],
            "rw,noexec,nosuid,size=32m,mode=0711,uid=0,gid=0",
        )

    def test_collect_container_sample_is_sanitized(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment.client = Mock()
        environment.client.api.inspect_container.return_value = {
            "Name": "/agent-1",
            "State": {
                "Status": "running",
                "Running": True,
                "Paused": False,
                "Restarting": False,
                "OOMKilled": False,
                "Dead": False,
                "Pid": 123,
                "ExitCode": 0,
                "Error": "",
                "StartedAt": "start",
                "FinishedAt": "finish",
            },
            "Config": {"Env": ["SECRET=1"]},
            "Mounts": [{"Source": "/secret"}],
        }
        container = Mock()
        container.id = "abc123"
        container.name = "agent-1"
        container.status = "running"
        container.stats.return_value = {
            "memory_stats": {"usage": 10, "max_usage": 12, "limit": 20},
            "pids_stats": {"current": 4},
            "cpu_stats": {
                "cpu_usage": {"total_usage": 30},
                "system_cpu_usage": 40,
                "online_cpus": 8,
            },
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "Read", "value": 50},
                    {"op": "Write", "value": 60},
                ]
            },
        }

        sample = environment._collect_container_sample(container, 2)

        self.assertEqual(sample["container_generation"], 2)
        self.assertEqual(sample["container_name"], "agent-1")
        self.assertEqual(sample["memory"]["usage_bytes"], 10)
        self.assertEqual(sample["pids"]["current"], 4)
        self.assertEqual(sample["cpu"]["online_cpus"], 8)
        self.assertEqual(sample["block_io"]["read"], 50)
        self.assertNotIn("Config", json.dumps(sample))
        self.assertNotIn("Mounts", json.dumps(sample))

    def test_classify_container_failure_prefers_specific_markers(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        classifications = environment._classify_container_failure(
            {"OOMKilled": True, "ExitCode": 137, "Error": ""}
        )
        self.assertIn("oom_killed", classifications)
        self.assertIn("nonzero_exit", classifications)

    def test_remove_current_container_captures_lifecycle_before_remove(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        container = Mock()
        environment.container = container
        environment._container_lock = threading.RLock()
        environment._capture_container_lifecycle_snapshot = Mock()
        environment._removed = False

        environment._remove_current_container(lifecycle_reason="pre_close_remove")

        environment._capture_container_lifecycle_snapshot.assert_called_once_with(
            "pre_close_remove",
            container=container,
        )
        container.remove.assert_called_once_with(force=True)
        self.assertTrue(environment._removed)

    def test_monitor_sampling_error_is_tolerated(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment._monitor_stop = Mock()
        environment._monitor_stop.wait.side_effect = [False, True]
        environment._current_container = Mock(return_value=Mock())
        environment._current_container_generation = Mock(return_value=1)
        environment._collect_container_sample = Mock(side_effect=RuntimeError("boom"))
        environment._record_container_lifecycle_event = Mock()
        environment._container_shutdown_expected = False

        environment._container_monitor_loop()

        environment._record_container_lifecycle_event.assert_called_once()
        self.assertEqual(
            environment._record_container_lifecycle_event.call_args.args[0],
            "sampling_error",
        )

    def test_container_observability_summary_reports_peaks_and_recreations(
        self,
    ) -> None:
        with scratch_dir() as root:
            environment = object.__new__(CadGenBenchEnvironment)
            environment.run_dir = root
            environment.container_telemetry_path = root / "container-telemetry.jsonl"
            environment.container_failure_path = root / "container-failure.json"
            environment.container_failure_path.write_text("{}", encoding="utf-8")
            environment.container_lifecycle_dir = root / "container-lifecycle"
            environment.container_lifecycle_dir.mkdir()
            environment._peak_memory_usage_bytes = 11
            environment._peak_memory_limit_bytes = 22
            environment._peak_pids = 3
            environment._container_exit_states = [
                {"status": "exited", "exit_code": 137}
            ]
            environment._harness_kill_events = [{"timestamp_utc": "now"}]
            environment.mcp_restart_history = [
                {"container_recreated": True},
                {"container_recreated": False},
            ]

            summary = environment._container_observability_summary()

            self.assertEqual(summary["peak_memory_usage_bytes"], 11)
            self.assertEqual(summary["peak_pids"], 3)
            self.assertEqual(summary["recreation_count"], 1)
            self.assertEqual(summary["harness_kill_count"], 1)

    def test_changed_valid_candidate_is_archived(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment._candidate_signatures = {}
        environment._live_candidate_state = Mock(
            return_value={
                "path": "/workspace/output.step",
                "size": 123,
                "mtime_ns": 456,
            }
        )
        environment._validate_live_candidate = Mock(return_value={"is_valid": True})
        environment._archive_valid_export = Mock(
            return_value={"index": 1, "sha256": "abc", "size_bytes": 123}
        )

        result = environment._checkpoint_changed_candidate()

        self.assertEqual(result["status"], "archived")
        environment._archive_valid_export.assert_called_once_with(
            "/workspace/output.step",
            validation={"is_valid": True},
        )
        self.assertIsNone(environment._checkpoint_changed_candidate())

    def test_changed_invalid_candidate_is_not_archived(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment._candidate_signatures = {}
        environment._live_candidate_state = Mock(
            return_value={
                "path": "/workspace/output.step",
                "size": 123,
                "mtime_ns": 456,
            }
        )
        environment._validate_live_candidate = Mock(
            return_value={"is_valid": False, "topology_errors": ["open"]}
        )
        environment._archive_valid_export = Mock()

        result = environment._checkpoint_changed_candidate()

        self.assertEqual(result["status"], "rejected_invalid")
        environment._archive_valid_export.assert_not_called()

    def test_mcp_service_recovery_restores_checkpoint(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment.container = Mock(id="container-old", status="exited")
        environment._container_lock = threading.RLock()
        environment._container_generation = 1
        environment.mcp_telemetry = {"container_generation": 1}
        environment.mcp_restart_history = []
        environment._reload_container_status = Mock(return_value="exited")
        environment._capture_container_lifecycle_snapshot = Mock(
            return_value={"state": {"Status": "exited"}}
        )
        environment._write_container_failure = Mock()
        environment._recreate_agent_container = Mock(
            return_value={
                "old_container_id": "container-old",
                "old_container_status": "exited",
                "new_container_id": "container-new",
                "container_recreated": True,
            }
        )
        environment._restore_live_workspace_checkpoint = Mock(return_value=True)
        environment._wait_for_mcp_service = Mock(
            return_value={"container_generation": 2}
        )
        environment._live_candidate_state = Mock(
            return_value={
                "path": "/workspace/output.step",
                "size": 123,
                "mtime_ns": 456,
            }
        )
        environment._write_checkpoint_history = Mock()

        result = environment._recover_mcp_service(reason="transport_failure")

        self.assertTrue(result["checkpoint_restored"])
        self.assertTrue(result["container_recreated"])
        self.assertIn("cad-mcp call import_cad_file", result["error"])
        self.assertEqual(environment.mcp_telemetry["container_generation"], 2)
        self.assertEqual(len(environment.mcp_restart_history), 1)

    def test_mcp_service_restart_limit_is_terminal(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment.mcp_telemetry = None
        environment.mcp_restart_history = [{}, {}, {}]
        environment._write_checkpoint_history = Mock()

        with self.assertRaisesRegex(
            CadGenBenchInfrastructureError,
            "restart limit 3 is exhausted",
        ):
            environment._recover_mcp_service(reason="transport_failure")

    def test_restore_live_workspace_checkpoint_puts_archive_and_renames_atomically(
        self,
    ) -> None:
        with scratch_dir() as root:
            latest = root / "latest-valid" / "output.step"
            latest.parent.mkdir(parents=True)
            latest.write_bytes(b"STEPDATA")
            environment = object.__new__(CadGenBenchEnvironment)
            environment.latest_valid_output = latest
            environment.container = Mock()
            environment.client = Mock()
            with patch(
                "cuawright.environments.cadgenbench.environment.run_exec",
                return_value={"returncode": 0, "stderr": b""},
            ) as run_exec_mock:
                restored = environment._restore_live_workspace_checkpoint()

            self.assertTrue(restored)
            put_path, archive_bytes = environment.container.put_archive.call_args.args
            self.assertEqual(put_path, "/workspace")
            with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:") as tar:
                member = tar.getmember(".restored-output.step")
                self.assertEqual(member.mode, 0o600)
                self.assertEqual(member.uid, os.getuid())
                self.assertEqual(member.gid, os.getgid())
                extracted = tar.extractfile(member)
                self.assertEqual(extracted.read(), b"STEPDATA")
            self.assertIn(
                "mv -f /workspace/.restored-output.step /workspace/output.step",
                run_exec_mock.call_args.kwargs["command"][2],
            )

    def test_checkpoint_history_records_restart_telemetry(self) -> None:
        with scratch_dir() as root:
            environment = object.__new__(CadGenBenchEnvironment)
            environment.run_dir = root
            environment.latest_valid_output = root / "latest-valid" / "output.step"
            environment.latest_valid_dir = environment.latest_valid_output.parent
            environment.latest_valid_dir.mkdir(parents=True)
            environment.checkpoint_history = []
            environment.mcp_restart_history = [
                {"status": "restarted", "reason": "transport_failure"}
            ]
            environment.mcp_telemetry = {"transport": "streamable_http"}
            environment.checkpoint_history_path = root / "checkpoint-history.json"

            environment._write_checkpoint_history()

            payload = json.loads(
                environment.checkpoint_history_path.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["mcp_restart_count"], 1)
            self.assertEqual(
                payload["mcp_restart_history"][0]["reason"],
                "transport_failure",
            )

    def test_recreate_agent_container_tracks_generations(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        old_container = Mock()
        old_container.id = "old"
        environment.container = old_container
        environment._container_lock = threading.RLock()
        environment._container_generation = 1
        environment._removed = False
        environment._capture_container_lifecycle_snapshot = Mock(
            return_value={"state": {"Status": "exited"}}
        )
        new_container = Mock()
        new_container.id = "new"
        environment._start_agent_container = Mock(return_value=new_container)
        environment._set_current_container = Mock()

        result = environment._recreate_agent_container()

        old_container.remove.assert_called_once_with(force=True)
        environment._set_current_container.assert_called_once_with(
            new_container,
            reason="recreate_after_transport_loss",
        )
        self.assertTrue(result["container_recreated"])
        self.assertEqual(result["old_container_status"], "exited")
        self.assertEqual(result["new_container_id"], "new")

    @patch.object(cadgen_environment, "CONTAINER_MONITOR_INTERVAL_SECONDS", 0.05)
    def test_container_failure_injection_smoke_records_generations_and_cleans_up(
        self,
    ) -> None:
        client = _docker_client_or_skip(self)
        fixture_id = f"9{str(uuid.uuid4().int)[:2]}"
        image = _local_test_image(client, self)
        with scratch_dir() as root:
            env = _observable_env_for_smoke(
                client,
                root=root,
                image=image,
                fixture_id=fixture_id,
            )
            try:
                time.sleep(0.15)
                env.container.stop(timeout=0)
                time.sleep(0.15)
                state = env._capture_container_lifecycle_snapshot(
                    "smoke_detected_stop"
                )["state"]
                env._write_container_failure(
                    reason="smoke_detected_stop",
                    state=state,
                )
                recreate_info = env._recreate_agent_container()
                time.sleep(0.15)
            finally:
                env._container_shutdown_expected = True
                env._stop_container_monitor()
                if env.container is not None:
                    env._remove_current_container(
                        lifecycle_reason="smoke_cleanup_remove"
                    )
                if env.workspace_volume is not None:
                    env.workspace_volume.remove(force=True)
                client.close()

            lines = [
                json.loads(line)
                for line in env.container_telemetry_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            generations = {
                line["container_generation"]
                for line in lines
                if line.get("kind") == "lifecycle"
                and line.get("event") == "container_generation_start"
            }
            exit_states = [
                line.get("state", {}).get("Status")
                for line in lines
                if line.get("kind") == "sample"
            ]
            failure_payload = json.loads(
                env.container_failure_path.read_text(encoding="utf-8")
            )

            self.assertIn(1, generations)
            self.assertIn(2, generations)
            self.assertIn(
                failure_payload["state"].get("Status"),
                {
                    "created",
                    "exited",
                    "dead",
                    "removing",
                    "stopped",
                    "unknown",
                    "inspect_failed",
                },
            )
            self.assertTrue(recreate_info["container_recreated"])
            self.assertTrue(failure_payload["likely_classifications"])
            self.assertEqual(
                client.containers.list(
                    all=True,
                    filters={"label": [f"cuawright.fixture_id={fixture_id}"]},
                ),
                [],
            )
            self.assertEqual(
                client.volumes.list(
                    filters={"label": [f"cuawright.fixture_id={fixture_id}"]},
                ),
                [],
            )
            client.close()

    def test_close_removes_current_replacement_container_once(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment._frozen = True
        environment._removed = False
        environment._volume_removed = False
        replacement = Mock()
        environment.container = replacement
        environment._container_lock = threading.RLock()
        environment._container_generation = 2
        environment.workspace_volume = Mock()
        environment.client = Mock()
        environment._monitor_stop = threading.Event()
        environment._monitor_thread = None
        environment._capture_container_lifecycle_snapshot = Mock()

        environment.close()

        replacement.remove.assert_called_once_with(force=True)
        environment.workspace_volume.remove.assert_called_once_with(force=True)

    def test_run_prediction_removes_stale_output_and_requires_new_artifact(
        self,
    ) -> None:
        with scratch_dir() as root:
            prediction = root / "prediction.py"
            output = root / "output.step"
            prediction.write_text("print('done')\n", encoding="utf-8")
            output.write_bytes(b"STALE")

            with self.assertRaisesRegex(
                RuntimeError,
                "without producing",
            ):
                run_prediction(prediction, output_step=output, timeout=5)

            self.assertFalse(output.exists())

    def test_evaluate_reports_missing_output_truthfully(self) -> None:
        with scratch_dir() as root:
            workspace = root / "workspace"
            workspace.mkdir()
            public_dir = root / "evaluation" / "public"
            public_dir.mkdir(parents=True)

            environment = object.__new__(CadGenBenchEnvironment)
            environment.run_dir = root
            environment.workspace_dir = workspace
            environment.benchmark_commit = DEFAULT_BENCHMARK_COMMIT
            environment.image_id = "sha256:image"
            environment.provenance = {"fixture_id": "101", "task_type": "generation"}
            environment._freeze_workspace = Mock()
            environment._restore_latest_valid_candidate = Mock()
            environment._run_public_evaluation_job = Mock(
                return_value={
                    "status": "missing_output",
                    "error": "Missing /workspace/output.step",
                    "validation": None,
                    "renders": None,
                }
            )
            environment.latest_valid_output = root / "latest-valid" / "output.step"
            environment.latest_valid_dir = environment.latest_valid_output.parent
            environment.latest_valid_output.parent.mkdir(parents=True)
            environment.container_telemetry_path = root / "container-telemetry.jsonl"
            environment.container_failure_path = root / "container-failure.json"
            environment.container_lifecycle_dir = root / "container-lifecycle"
            environment.container_lifecycle_dir.mkdir()
            environment.checkpoint_history = []
            environment.mcp_restart_history = []
            environment._peak_memory_usage_bytes = 0
            environment._peak_memory_limit_bytes = 0
            environment._peak_pids = 0
            environment._container_exit_states = []
            environment._harness_kill_events = []
            environment.checkpoint_history_path = root / "checkpoint-history.json"
            environment.checkpoint_history_path.write_text(
                json.dumps({"checkpoint_history": [], "mcp": None}),
                encoding="utf-8",
            )

            result = environment.evaluate()

            self.assertIsNone(result["score"])
            self.assertEqual(result["status"], "missing_output")
            self.assertIsNone(result["output_step_sha256"])
            self.assertEqual(result["prediction_sha256"], None)

    def test_evaluate_reports_invalid_output_and_hashes(self) -> None:
        with scratch_dir() as root:
            workspace = root / "workspace"
            workspace.mkdir()
            prediction = workspace / "prediction.py"
            output = workspace / "output.step"
            prediction.write_text("print('hi')\n", encoding="utf-8")
            output.write_bytes(b"STEP")

            environment = object.__new__(CadGenBenchEnvironment)
            environment.run_dir = root
            environment.workspace_dir = workspace
            environment.benchmark_commit = DEFAULT_BENCHMARK_COMMIT
            environment.image_id = "sha256:image"
            environment.provenance = {"fixture_id": "201", "task_type": "editing"}
            environment._freeze_workspace = Mock()
            environment._restore_latest_valid_candidate = Mock()
            environment._run_public_evaluation_job = Mock(
                return_value={
                    "status": "invalid_output",
                    "error": None,
                    "validation": {"is_valid": False},
                    "renders": {"composite": "/out/renders/multiview.png"},
                }
            )
            latest_valid = root / "latest-valid" / "output.step"
            latest_valid.parent.mkdir(parents=True)
            latest_valid.write_bytes(output.read_bytes())
            environment.latest_valid_output = latest_valid
            environment.latest_valid_dir = latest_valid.parent
            environment.container_telemetry_path = root / "container-telemetry.jsonl"
            environment.container_failure_path = root / "container-failure.json"
            environment.container_lifecycle_dir = root / "container-lifecycle"
            environment.container_lifecycle_dir.mkdir()
            environment.checkpoint_history = [
                {"index": 1, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
            ]
            environment.mcp_restart_history = []
            environment._peak_memory_usage_bytes = 0
            environment._peak_memory_limit_bytes = 0
            environment._peak_pids = 0
            environment._container_exit_states = []
            environment._harness_kill_events = []
            environment.checkpoint_history_path = root / "checkpoint-history.json"
            environment.checkpoint_history_path.write_text(
                json.dumps(
                    {"checkpoint_history": environment.checkpoint_history, "mcp": None}
                ),
                encoding="utf-8",
            )

            result = environment.evaluate()

            self.assertEqual(result["status"], "invalid_output")
            self.assertEqual(
                result["prediction_sha256"],
                hashlib.sha256(prediction.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["output_step_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            self.assertIn("workspace/output.step", result["artifact_hashes"])

    def test_restore_latest_valid_candidate_overwrites_mutable_final(self) -> None:
        with scratch_dir() as root:
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "output.step").write_bytes(b"UNTRUSTED")
            latest = root / "latest-valid" / "output.step"
            latest.parent.mkdir(parents=True)
            latest.write_bytes(b"TRUSTED")

            environment = object.__new__(CadGenBenchEnvironment)
            environment.workspace_dir = workspace
            environment.latest_valid_output = latest

            environment._restore_latest_valid_candidate()

            self.assertEqual((workspace / "output.step").read_bytes(), b"TRUSTED")
            self.assertEqual(
                (workspace / "output.mutable-final.step").read_bytes(), b"UNTRUSTED"
            )

    def test_freeze_workspace_cleans_up_after_snapshot_failure(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        container = Mock()
        container.status = "running"
        environment.container = container
        environment._container_lock = threading.RLock()
        environment._container_generation = 1
        environment.workspace_volume = Mock()
        environment._persist_workspace_artifacts = Mock(
            side_effect=RuntimeError("boom")
        )
        environment._write_checkpoint_history = Mock()
        environment._capture_container_lifecycle_snapshot = Mock()
        environment._frozen = False
        environment._removed = False
        environment._volume_removed = False

        with self.assertRaisesRegex(RuntimeError, "boom"):
            environment._freeze_workspace()

        container.remove.assert_called_once_with(force=True)
        environment.workspace_volume.remove.assert_called_once_with(force=True)

    def test_close_removes_remaining_resources(self) -> None:
        environment = object.__new__(CadGenBenchEnvironment)
        environment._frozen = True
        environment._removed = False
        environment._volume_removed = False
        container = Mock()
        environment.container = container
        environment._container_lock = threading.RLock()
        environment._container_generation = 1
        environment.workspace_volume = Mock()
        environment.client = Mock()
        environment._monitor_stop = threading.Event()
        environment._monitor_thread = None
        environment._capture_container_lifecycle_snapshot = Mock()

        environment.close()

        container.remove.assert_called_once_with(force=True)
        environment.workspace_volume.remove.assert_called_once_with(force=True)
        environment.client.close.assert_called_once()


class CadGenBenchPackagingTests(unittest.TestCase):
    def test_package_submission_uses_official_layout(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_generation_fixture(root)
            _write_editing_fixture(root)
            manifest = discover_tasks(
                dataset_dir=root,
                benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
            )

            run_a = root / "run-101"
            (run_a / "workspace").mkdir(parents=True)
            (run_a / "workspace" / "output.stp").write_bytes(b"STEP101")
            (run_a / "task.json").write_text(
                json.dumps(
                    {
                        "benchmark": "cadgenbench",
                        **manifest.tasks[0].provenance(),
                    }
                ),
                encoding="utf-8",
            )

            run_b = root / "run-201"
            (run_b / "workspace").mkdir(parents=True)
            (run_b / "task.json").write_text(
                json.dumps(
                    {
                        "benchmark": "cadgenbench",
                        **manifest.tasks[1].provenance(),
                    }
                ),
                encoding="utf-8",
            )

            output_zip = root / "submission.zip"
            package_submission(
                manifest=manifest,
                run_dirs=[run_a, run_b],
                output_path=output_zip,
                meta={
                    "submitter_name": "tester",
                    "submission_name": "panel",
                    "agent_url": None,
                    "notes": None,
                    "agree_to_publish": False,
                },
            )

            with zipfile.ZipFile(output_zip) as archive:
                names = set(archive.namelist())

            self.assertIn("meta.json", names)
            self.assertIn("101/output.step", names)
            self.assertIn("201/", names)
            self.assertNotIn("101/output.stp", names)

    def test_package_submission_rejects_provenance_mismatch(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_generation_fixture(root)
            manifest = discover_tasks(
                dataset_dir=root,
                benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
            )
            run_dir = root / "run-101"
            (run_dir / "workspace").mkdir(parents=True)
            (run_dir / "workspace" / "output.step").write_bytes(b"STEP101")
            bad_payload = {
                "benchmark": "cadgenbench",
                **manifest.tasks[0].provenance(),
            }
            bad_payload["dataset_revision"] = "wrong"
            (run_dir / "task.json").write_text(
                json.dumps(bad_payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "does not match the panel manifest"
            ):
                package_submission(
                    manifest=manifest,
                    run_dirs=[run_dir],
                    output_path=root / "submission.zip",
                    meta={
                        "submitter_name": "tester",
                        "submission_name": "panel",
                        "agent_url": None,
                        "notes": None,
                        "agree_to_publish": False,
                    },
                )

    def test_package_submission_rejects_public_file_hash_mismatch(self) -> None:
        with scratch_dir() as root:
            _write_dataset_file(root, "sanity_check_submission.py", b"print('ok')\n")
            _write_generation_fixture(root)
            manifest = discover_tasks(
                dataset_dir=root,
                benchmark_commit=DEFAULT_BENCHMARK_COMMIT,
            )
            run_dir = root / "run-101"
            (run_dir / "workspace").mkdir(parents=True)
            (run_dir / "workspace" / "output.step").write_bytes(b"STEP101")
            bad_payload = {
                "benchmark": "cadgenbench",
                **manifest.tasks[0].provenance(),
            }
            bad_payload["public_files"][0]["sha256"] = "0" * 64
            (run_dir / "task.json").write_text(
                json.dumps(bad_payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "does not match the panel manifest"
            ):
                package_submission(
                    manifest=manifest,
                    run_dirs=[run_dir],
                    output_path=root / "submission.zip",
                    meta={
                        "submitter_name": "tester",
                        "submission_name": "panel",
                        "agent_url": None,
                        "notes": None,
                        "agree_to_publish": False,
                    },
                )


if __name__ == "__main__":
    unittest.main()

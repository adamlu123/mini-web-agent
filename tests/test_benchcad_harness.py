from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from docker.errors import DockerException

from cuawright.environments.benchcad.environment import BenchCADEnvironment
from cuawright.environments.benchcad.image import validate_image
from cuawright.environments.benchcad.plugin import (
    _READ_CONTROL_FILE_SCRIPT,
    _READ_FILE_SCRIPT,
    _WRITE_CONTROL_FILE_SCRIPT,
    _is_inspection_command,
    BenchCADPlugin,
    BenchCADPluginConfig,
    create_benchcad_plugin,
)
from cuawright.environments.benchcad.scoring import (
    _copy_container_file,
    score_prediction,
)
from cuawright.environments.benchcad.task import load_task, validate_record_id
from cuawright.environments.base import CommandResult, ControlFileAccessError


class BenchCADManifestTests(unittest.TestCase):
    def test_loads_relative_paths_and_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (root / "target.step").write_text("step", encoding="utf-8")
            input_sha = hashlib.sha256((root / "input.png").read_bytes()).hexdigest()
            step_sha = hashlib.sha256((root / "target.step").read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "benchmark_commit": "abc123",
                        "dataset_revision": "dataset123",
                        "tasks": [
                            {
                                "record_id": "part-1",
                                "input_image": "input.png",
                                "ground_truth_step": "target.step",
                                "system_prompt": "system",
                                "user_prompt": "user",
                                "input_sha256": input_sha,
                                "ground_truth_sha256": step_sha,
                                "family": "washer",
                                "difficulty": "easy",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            loaded_manifest, task = load_task(manifest, "part-1")

            self.assertEqual(loaded_manifest.benchmark_commit, "abc123")
            self.assertEqual(task.input_image, (root / "input.png").resolve())
            self.assertIn("/workspace/prediction.py", task.instruction)
            self.assertIn("/input/input.png", task.instruction)

    def test_record_ids_cannot_escape_output_roots(self) -> None:
        for record_id in ("../escape", "/absolute", "nested/path", ""):
            with (
                self.subTest(record_id=record_id),
                self.assertRaises(ValueError),
            ):
                validate_record_id(record_id)


class BenchCADImageTests(unittest.TestCase):
    def test_image_commit_label_is_enforced(self) -> None:
        image = SimpleNamespace(
            id="sha256:image",
            attrs={"Config": {"Labels": {"org.benchcad.commit": "wrong"}}},
        )
        client = SimpleNamespace(images=SimpleNamespace(get=lambda _: image))
        with self.assertRaisesRegex(RuntimeError, "expected 'expected'"):
            validate_image(client, "image:tag", "expected")


class BenchCADPluginTests(unittest.TestCase):
    def test_candidate_generation_is_not_inspection(self) -> None:
        environment = SimpleNamespace(
            instruction="task",
            run_command=Mock(return_value={"returncode": 0, "output": "", "error": ""}),
        )
        result = BenchCADPlugin(environment).run_command(
            "benchcad-run /workspace/prediction.py " "--step /workspace/candidate.step"
        )

        self.assertFalse(result.inspected)

    def test_render_and_file_queries_are_inspection(self) -> None:
        commands = (
            "benchcad-render /workspace/candidate.step /workspace/candidate.png",
            "find /workspace -name '*' -print",
            "sha256sum /workspace/candidate.step",
            "stat /workspace/candidate.step",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(_is_inspection_command(command))

    def test_shell_composition_and_unknown_forms_are_not_inspection(self) -> None:
        commands = (
            "cat /workspace/candidate.step > /workspace/copy.step",
            "benchcad-render /workspace/candidate.step /workspace/candidate.png "
            "&& touch /workspace/changed",
            "find /workspace {-delete,-print}",
            "find /workspace *",
            "find /workspace ?delete",
            "find /workspace [-]delete",
            "benchcad-render /workspace/candidate.step /workspace/prediction.py",
            "find /workspace -type f -fprint0 /workspace/files.txt",
            "python /workspace/prediction.py",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(_is_inspection_command(command))

    def test_run_command_delegates_and_normalizes_result(self) -> None:
        environment = SimpleNamespace(
            instruction="build the part",
            run_command=Mock(
                return_value={
                    "returncode": 7,
                    "output": "stdout",
                    "error": "stderr",
                }
            ),
        )
        plugin = BenchCADPlugin(environment)

        result = plugin.run_command("benchcad-run prediction.py", timeout=45)

        self.assertEqual(
            result,
            CommandResult(
                7,
                stdout="stdout",
                stderr="stderr",
                model_fields={
                    "returncode": 7,
                    "output": "stdout",
                    "error": "stderr",
                },
            ),
        )
        environment.run_command.assert_called_once_with(
            "benchcad-run prediction.py",
            45,
        )
        self.assertEqual(plugin.instruction, "build the part")

    def test_run_command_preserves_full_streams_for_aggregate_formatting(self) -> None:
        output = "o" * 20_000
        error = "e" * 20_000
        environment = SimpleNamespace(
            instruction="task",
            run_command=Mock(
                return_value={
                    "status": "error",
                    "returncode": 7,
                    "output": output,
                    "error": error,
                    "infrastructure_error": False,
                    "timed_out": True,
                }
            ),
        )

        result = BenchCADPlugin(environment).run_command("false")

        self.assertEqual(result.stdout, output)
        self.assertEqual(result.stderr, error)
        self.assertTrue(result.model_fields["timed_out"])

    def test_run_command_applies_default_timeout(self) -> None:
        environment = SimpleNamespace(
            instruction="task",
            run_command=Mock(return_value={"returncode": 0, "output": "", "error": ""}),
        )
        plugin = BenchCADPlugin(environment, default_command_timeout=91)

        plugin.run_command("pwd")

        environment.run_command.assert_called_once_with("pwd", 91)

    def test_file_bridge_is_bounded_and_does_not_invoke_shell(self) -> None:
        environment = SimpleNamespace()
        plugin = BenchCADPlugin(environment)
        with patch.object(
            BenchCADPlugin,
            "_run_bridge",
            autospec=True,
            return_value=b"png",
        ) as bridge:
            data = plugin.read_file("/workspace/candidate.png", max_bytes=1024)

        self.assertEqual(data, b"png")
        command = bridge.call_args.args[1]
        self.assertEqual(command[0], "/opt/benchcad-venv/bin/python")
        self.assertEqual(command[1], "-c")
        self.assertNotIn("/bin/sh", command)
        self.assertEqual(bridge.call_args.kwargs["capture_limit"], 4096)

    def test_control_bridge_rejects_wrong_workspace_before_exec(self) -> None:
        plugin = BenchCADPlugin(SimpleNamespace())
        with self.assertRaises(ControlFileAccessError):
            plugin.read_control_file(
                "/workspace/other/report.json",
                workspace="/workspace/other",
                max_bytes=1024,
            )

    def test_config_factory_builds_environment_from_manifest_commit(self) -> None:
        manifest = SimpleNamespace(benchmark_commit="benchcad-commit")
        task = SimpleNamespace(record_id="part-1")
        environment = SimpleNamespace(instruction="task")
        config = BenchCADPluginConfig(
            manifest_path="manifest.json",
            record_id="part-1",
            run_dir="run",
            image="benchcad:image",
            benchcad_repo="BenchCAD",
            defer_evaluation=True,
        )
        with (
            patch(
                "cuawright.environments.benchcad.plugin.load_task",
                return_value=(manifest, task),
            ) as loader,
            patch(
                "cuawright.environments.benchcad.plugin.BenchCADEnvironment",
                return_value=environment,
            ) as environment_type,
        ):
            plugin = create_benchcad_plugin(config)

        self.assertIs(plugin.environment, environment)
        loader.assert_called_once_with("manifest.json", "part-1")
        self.assertEqual(
            environment_type.call_args.kwargs["benchmark_commit"],
            "benchcad-commit",
        )
        self.assertEqual(environment_type.call_args.kwargs["image"], "benchcad:image")
        self.assertTrue(environment_type.call_args.kwargs["defer_evaluation"])


class BenchCADSecureBridgeTests(unittest.TestCase):
    @staticmethod
    def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, "-c", script, *arguments],
            check=False,
            capture_output=True,
        )

    def test_arbitrary_file_read_honors_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "candidate.png"
            image.write_bytes(b"image-data")

            accepted = self._run(_READ_FILE_SCRIPT, str(image), "10")
            rejected = self._run(_READ_FILE_SCRIPT, str(image), "9")

        self.assertEqual(accepted.returncode, 0)
        self.assertEqual(accepted.stdout, b"image-data")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertEqual(rejected.stdout, b"")

    def test_control_read_rejects_symlink_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace" / ".agent"
            workspace.mkdir(parents=True)
            report = workspace / "report.json"
            report.write_bytes(b'{"status":"ok"}')
            outside = root / "outside.json"
            outside.write_bytes(b"secret")
            (workspace / "linked.json").symlink_to(outside)
            (workspace / "linked-parent").symlink_to(root)

            accepted = self._run(
                _READ_CONTROL_FILE_SCRIPT,
                str(report),
                str(workspace),
                "1024",
            )
            linked = self._run(
                _READ_CONTROL_FILE_SCRIPT,
                str(workspace / "linked.json"),
                str(workspace),
                "1024",
            )
            linked_parent = self._run(
                _READ_CONTROL_FILE_SCRIPT,
                str(workspace / "linked-parent" / "outside.json"),
                str(workspace),
                "1024",
            )
            traversal = self._run(
                _READ_CONTROL_FILE_SCRIPT,
                f"{workspace}/../outside.json",
                str(workspace),
                "1024",
            )

        self.assertEqual(accepted.returncode, 0)
        self.assertEqual(accepted.stdout, b'{"status":"ok"}')
        for result in (linked, linked_parent, traversal):
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b"")

    def test_control_write_is_atomic_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace" / ".agent"
            workspace.mkdir(parents=True)
            state = workspace / "state.json"
            state.write_bytes(b"old")
            payload = b'{"state":"new"}'
            encoded = base64.b64encode(payload).decode("ascii")

            written = self._run(
                _WRITE_CONTROL_FILE_SCRIPT,
                str(state),
                str(workspace),
                encoded,
            )

            outside = root / "outside.json"
            outside.write_bytes(b"outside")
            linked = workspace / "linked.json"
            linked.symlink_to(outside)
            rejected = self._run(
                _WRITE_CONTROL_FILE_SCRIPT,
                str(linked),
                str(workspace),
                base64.b64encode(b"overwrite").decode("ascii"),
            )
            leftovers = list(workspace.glob(".*.agent-*"))

            self.assertEqual(written.returncode, 0)
            self.assertEqual(state.read_bytes(), payload)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertTrue(linked.is_symlink())
            self.assertEqual(outside.read_bytes(), b"outside")
            self.assertEqual(leftovers, [])


class BenchCADEnvironmentTests(unittest.TestCase):
    def test_run_command_normalizes_exec_result(self) -> None:
        environment = object.__new__(BenchCADEnvironment)
        environment.container = SimpleNamespace(id="container-id")
        environment.client = SimpleNamespace(
            api=SimpleNamespace(
                exec_create=lambda *args, **kwargs: {"Id": "exec-id"},
                exec_start=lambda *args, **kwargs: iter([(b"ok\n", b"")]),
                exec_inspect=lambda *args, **kwargs: {"ExitCode": 0},
            )
        )
        environment.is_environment_used = False

        result = environment.run_command("echo ok", 30)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["output"], "ok\n")
        self.assertTrue(environment.is_environment_used)

    def test_get_file_falls_back_to_exec_when_archive_cannot_read_tmpfs(
        self,
    ) -> None:
        environment = object.__new__(BenchCADEnvironment)
        environment.container = SimpleNamespace(
            id="container-id",
            get_archive=Mock(side_effect=DockerException("archive unavailable")),
        )
        environment.client = SimpleNamespace(
            api=SimpleNamespace(
                exec_create=lambda *args, **kwargs: {"Id": "exec-id"},
                exec_start=lambda *args, **kwargs: iter([(b"contents", b"")]),
                exec_inspect=lambda *args, **kwargs: {"ExitCode": 0},
            )
        )

        self.assertEqual(
            environment.get_file("/workspace/prediction.py"),
            b"contents",
        )

    def test_scorer_copy_falls_back_to_exec_for_tmpfs_artifact(self) -> None:
        container = SimpleNamespace(
            id="container-id",
            get_archive=Mock(side_effect=DockerException("archive unavailable")),
        )
        client = SimpleNamespace(
            api=SimpleNamespace(
                exec_create=lambda *args, **kwargs: {"Id": "exec-id"},
                exec_start=lambda *args, **kwargs: iter([(b"step-data", b"")]),
                exec_inspect=lambda *args, **kwargs: {"ExitCode": 0},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "prediction.step"

            copied = _copy_container_file(
                client=client,
                container=container,
                source_path="/score/prediction.step",
                destination=destination,
            )

            self.assertTrue(copied)
            self.assertEqual(destination.read_bytes(), b"step-data")

    def test_missing_prediction_scores_zero_without_starting_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth = root / "target.step"
            ground_truth.write_text("step", encoding="utf-8")
            task = SimpleNamespace(
                record_id="missing",
                ground_truth_step=ground_truth,
            )
            client = Mock()
            with (
                patch("docker.from_env", return_value=client),
                patch(
                    "cuawright.environments.benchcad.scoring.validate_image",
                    return_value="sha256:image",
                ),
            ):
                result = score_prediction(
                    task=task,
                    workspace_dir=root / "workspace",
                    output_dir=root / "score",
                    image="unused",
                    benchcad_repo=root,
                    benchmark_commit="abc123",
                )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["status"], "no_code")
        self.assertEqual(result["container_image_id"], "sha256:image")
        self.assertEqual(
            result["ground_truth_sha256"],
            hashlib.sha256(b"step").hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()

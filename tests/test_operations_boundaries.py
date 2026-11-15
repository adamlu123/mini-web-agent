from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cuawright.benchmarks import BENCHMARKS, benchmark_descriptor
from cuawright.cli import main, parser
from cuawright.config.store import (
    config_path,
    credential_status,
    load_config,
    save_config,
)
from cuawright.health.contracts import (
    HEALTH_CONTRACT_NAME,
    HEALTH_CONTRACT_VERSION,
    health_contract_manifest,
    missing_event_fields,
    request_role_from_prompt_hashes,
    validate_health_contract,
)
from cuawright.environments.weavebench.adapter import CUAWrightAgent
from cuawright.operations.launch import (
    doctor_odysseys_config,
    run_odysseys,
    run_weavebench,
    setup_odysseys,
)
from cuawright.operations.scheduler import (
    atomic_write_json,
    control_client_manifest,
    latest_run_dir,
    read_optional_json,
)
from scripts.python.check_terminal_run_health import check_run_health


class BenchmarkRegistryTests(unittest.TestCase):
    def test_registry_has_exactly_five_explicit_descriptors(self) -> None:
        self.assertEqual(
            tuple(BENCHMARKS),
            ("osworld", "weavebench", "benchcad", "cadgenbench", "odysseys"),
        )
        for name, descriptor in BENCHMARKS.items():
            with self.subTest(name=name):
                self.assertEqual(descriptor.name, name)
                self.assertTrue(callable(descriptor.setup_handler))
                self.assertTrue(callable(descriptor.doctor_handler))
                self.assertTrue(callable(descriptor.run_handler))
                self.assertTrue(descriptor.health.benchmark_names)
        self.assertEqual(
            benchmark_descriptor("benchcad").stable_conditions,
            ("full", "direct"),
        )
        for name in ("osworld", "weavebench", "cadgenbench", "odysseys"):
            self.assertEqual(
                benchmark_descriptor(name).stable_conditions,
                ("default",),
            )

    def test_weavebench_health_is_explicitly_incomplete(self) -> None:
        descriptor = benchmark_descriptor("weavebench")
        self.assertFalse(descriptor.health.supported)
        self.assertIn(
            "canonical terminal manifest", descriptor.health.incomplete_reason
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = check_run_health(
                Path(temporary),
                benchmark="weavebench",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )
        self.assertEqual(result["status"], "incomplete", result)
        self.assertFalse(result["objective_stop_required"])
        self.assertEqual(result["contract"]["name"], HEALTH_CONTRACT_NAME)


class CliBoundaryTests(unittest.TestCase):
    def test_no_subagents_is_available_for_terminal_benchmarks(self) -> None:
        accepted = (
            ("osworld", "--task", "001"),
            ("weavebench", "--task", "task", "--domain", "DOC"),
            ("benchcad", "full", "--record", "record"),
            ("cadgenbench", "--fixture", "101"),
        )
        for values in accepted:
            with self.subTest(benchmark=values[0]):
                arguments = parser().parse_args(
                    ["run", *values, "--result-root", "results", "--no-subagents"]
                )
                self.assertTrue(arguments.no_subagents)
        with self.assertRaises(SystemExit):
            parser().parse_args(
                [
                    "run",
                    "odysseys",
                    "--task",
                    "a" * 40,
                    "--result-root",
                    "results",
                    "--no-subagents",
                ]
            )
        self.assertTrue(
            CUAWrightAgent(litellm_api_key="key", no_subagents=True).no_subagents
        )

    def test_benchcad_full_and_direct_conditions_remain_stable(self) -> None:
        for condition in ("full", "direct"):
            args = parser().parse_args(
                [
                    "run",
                    "benchcad",
                    condition,
                    "--record",
                    "record",
                    "--result-root",
                    "results",
                ]
            )
            self.assertEqual(args.condition, condition)

    def test_main_dispatches_through_descriptor_handler(self) -> None:
        handler = Mock(return_value=17)
        descriptor = SimpleNamespace(
            name="osworld",
            display_name="OSWorld V2",
            install_extra="osworld",
            required_modules=(),
            minimum_versions={},
            requires_separate_runtime=False,
            run_handler=handler,
        )
        with (
            patch("cuawright.cli.benchmark_descriptor", return_value=descriptor),
            patch(
                "cuawright.cli.load_config",
                return_value={"osworld": {"source": "/source"}},
            ),
        ):
            result = main(
                [
                    "run",
                    "osworld",
                    "--task",
                    "001",
                    "--result-root",
                    "results",
                ]
            )
        self.assertEqual(result, 17)
        handler.assert_called_once()


class ConfigStoreTests(unittest.TestCase):
    def test_round_trip_and_invalid_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"XDG_CONFIG_HOME": temporary}):
                self.assertEqual(
                    config_path(),
                    Path(temporary) / "cuawright" / "benchmarks.json",
                )
                self.assertEqual(load_config(), {})
                save_config({"osworld": {"vm": "/images/vm.qcow2"}})
                self.assertEqual(
                    load_config(),
                    {"osworld": {"vm": "/images/vm.qcow2"}},
                )
                config_path().write_text("[]", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "invalid CUAWright"):
                    load_config()

    def test_credential_status_requires_a_private_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            key = Path(temporary) / "key"
            key.write_text("secret", encoding="utf-8")
            key.chmod(0o600)
            self.assertTrue(credential_status(key)["ok"])
            key.chmod(0o644)
            self.assertFalse(credential_status(key)["ok"])
            self.assertFalse(credential_status(Path(temporary) / "missing")["ok"])


class PublicOperationsTests(unittest.TestCase):
    def test_odysseys_setup_records_checked_source_and_runtime_identity(
        self,
    ) -> None:
        args = SimpleNamespace(
            source="/odysseys",
            source_commit="a" * 40,
            dataset="/odysseys/data/odysseys.json",
            examples="/odysseys/data/examples",
            judge="/odysseys/judge.py",
            runtime_source="/runtime",
            vm="/images/vm.qcow2",
            api_key_file="/keys/api",
        )
        source = SimpleNamespace(
            root=Path(args.source),
            commit=args.source_commit,
            dataset_path=Path(args.dataset),
            dataset_sha256="b" * 64,
            examples_dir=Path(args.examples),
            judge_script=Path(args.judge),
            judge_sha256="c" * 64,
        )
        runtime = SimpleNamespace(root=Path("/runtime"), commit="d" * 40)
        with (
            patch(
                "cuawright.operations.launch.inspect_odysseys",
                return_value=source,
            ) as inspect_source,
            patch(
                "cuawright.operations.launch.inspect_desktop_runtime",
                return_value=runtime,
            ) as inspect_runtime,
        ):
            result = setup_odysseys(args)

        self.assertEqual(result["source_commit"], source.commit)
        self.assertEqual(result["dataset_sha256"], source.dataset_sha256)
        self.assertEqual(result["judge_sha256"], source.judge_sha256)
        self.assertEqual(result["runtime_commit"], runtime.commit)
        inspect_source.assert_called_once()
        inspect_runtime.assert_called_once()

    def test_odysseys_doctor_fails_closed_on_dirty_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vm = root / "vm.qcow2"
            key = root / "key"
            vm.write_bytes(b"vm")
            key.write_text("secret", encoding="utf-8")
            key.chmod(0o600)
            item = {
                "source": "/odysseys",
                "source_commit": "a" * 40,
                "dataset": "/odysseys/data/odysseys.json",
                "dataset_sha256": "b" * 64,
                "examples": "/odysseys/data/examples",
                "judge": "/odysseys/judge.py",
                "judge_sha256": "c" * 64,
                "runtime_source": "/runtime",
                "runtime_commit": "d" * 40,
                "vm": str(vm),
                "api_key_file": str(key),
            }
            with (
                patch(
                    "cuawright.operations.launch.doctor_odysseys",
                    return_value={"ok": True},
                ),
                patch(
                    "cuawright.operations.launch.inspect_desktop_runtime",
                    side_effect=RuntimeError(
                        "Odysseys desktop runtime checkout is dirty"
                    ),
                ),
            ):
                result = doctor_odysseys_config(item)

        self.assertFalse(result["ok"])
        self.assertIn("runtime checkout is dirty", result["runtime_error"])

    def test_odysseys_public_run_uses_fixed_profile_and_configured_paths(
        self,
    ) -> None:
        args = SimpleNamespace(
            task="a" * 40,
            result_root="results",
        )
        item = {
            "source": "/odysseys",
            "source_commit": "b" * 40,
            "dataset": "/odysseys/data/odysseys.json",
            "dataset_sha256": "c" * 64,
            "examples": "/odysseys/data/odysseys_cua_final",
            "judge": "/odysseys/scripts/python/judge.py",
            "judge_sha256": "d" * 64,
            "runtime_source": "/runtime",
            "runtime_commit": "e" * 40,
            "vm": "/images/vm.qcow2",
            "api_key_file": "/keys/api",
        }
        completed = SimpleNamespace(returncode=0)
        with patch(
            "cuawright.operations.launch.subprocess.run",
            return_value=completed,
        ) as run_process:
            self.assertEqual(run_odysseys(args, item), 0)

        command = run_process.call_args.args[0]
        self.assertIn("cuawright.run.odysseys", command)
        self.assertEqual(command[command.index("--model") + 1], "gpt-5.5")
        self.assertEqual(
            command[command.index("--reasoning-effort") + 1],
            "xhigh",
        )
        self.assertNotIn("--max-dollars", command)

    def test_weavebench_reasoning_effort_defaults_and_can_be_configured(
        self,
    ) -> None:
        args = SimpleNamespace(
            task="DOC_task_0_latex_layout_fix",
            domain="DOC",
            result_root="results",
        )
        item = {
            "source": "/weavebench",
            "api_key_file": "/keys/api",
            "docker_bin": "/usr/bin/docker",
            "judge_runtime": "/judge",
            "judge_image": "judge:latest",
        }
        source = SimpleNamespace(
            root=Path("/weavebench"),
            tasks_root=Path("/weavebench/cache/tasks"),
            vm_path=Path("/weavebench/vm.qcow2"),
            judge_template_profile=Path("/profile"),
            judge_template_workspace=Path("/workspace"),
        )
        completed = SimpleNamespace(returncode=0)
        with (
            patch(
                "cuawright.operations.launch.inspect_weavebench",
                return_value=source,
            ),
            patch(
                "cuawright.operations.launch.subprocess.run",
                return_value=completed,
            ) as run_process,
        ):
            self.assertEqual(run_weavebench(args, item), 0)
            environment = run_process.call_args.kwargs["env"]
            self.assertEqual(
                environment["CUAWRIGHT_WEAVEBENCH_REASONING_EFFORT"],
                "high",
            )

            item["reasoning_effort"] = "xhigh"
            self.assertEqual(run_weavebench(args, item), 0)
            environment = run_process.call_args.kwargs["env"]
            self.assertEqual(
                environment["CUAWRIGHT_WEAVEBENCH_REASONING_EFFORT"],
                "xhigh",
            )

    def test_scheduler_helpers_are_public_and_panel_avoids_batch_privates(
        self,
    ) -> None:
        self.assertTrue(control_client_manifest()["protocol"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "record.json"
            atomic_write_json(target, {"ok": True})
            self.assertEqual(read_optional_json(target), {"ok": True})
            run_dir = root / "task-001-one"
            run_dir.mkdir()
            self.assertEqual(latest_run_dir(root, "001"), run_dir)

        repository = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (repository / "src/cuawright/run/panel.py").read_text(encoding="utf-8")
        )
        imports = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("cuawright.run.batch", imports)
        self.assertIn("cuawright.operations.scheduler", imports)


class HealthPromptIdentityTests(unittest.TestCase):
    def test_health_contract_versions_manifest_and_event_facts(self) -> None:
        contract = health_contract_manifest()
        self.assertEqual(contract["version"], HEALTH_CONTRACT_VERSION)
        self.assertEqual(
            validate_health_contract(contract),
            (True, "versioned health contract recorded"),
        )
        self.assertEqual(
            missing_event_fields(
                {
                    "event": "control_client_executed",
                    "command": "python client.py",
                }
            ),
            (
                "python_executable",
                "script_path",
                "operation",
                "exit_code",
                "stdout_size_bytes",
                "stderr_size_bytes",
            ),
        )

    def test_health_uses_recorded_hashes_not_prompt_implementation(self) -> None:
        system_prompt = "recorded system prompt bytes"
        import hashlib

        digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        manifest = {
            "prompt_hashes": {
                "critic": {
                    "bundle": "benchcad-full",
                    "sha256": {"composed_system": digest},
                }
            }
        }
        request = {
            "input": [{"role": "system", "content": system_prompt}],
            "extra_body": {"session_id": "untyped-session"},
        }
        self.assertEqual(
            request_role_from_prompt_hashes(manifest, request),
            "critic",
        )

        repository = Path(__file__).resolve().parents[1]
        source = (repository / "scripts/python/check_terminal_run_health.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("cuawright.config.prompts", source)
        self.assertNotIn("compose_system_prompt", source)
        self.assertNotIn("TERMINAL_OVERLAY", source)


if __name__ == "__main__":
    unittest.main()

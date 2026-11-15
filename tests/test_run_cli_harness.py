from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from cuawright.artifacts import ArtifactRef
from cuawright.budget import (
    DEFAULT_TOTAL_STEPS,
    MAX_TOTAL_STEPS,
    SharedStepBudget,
)
from cuawright.config.prompt_registry import select_osworld_prompt_bundle
from cuawright.models.base import DEFAULT_MAX_OUTPUT_TOKENS
from cuawright.models.openrouter_cost import DollarBudgetExceeded
from cuawright.models.providers import resolve_base_url, resolve_model
from cuawright.run.benchcad import parse_args as parse_benchcad_args
from cuawright.run.cadgenbench import parse_args as parse_cadgenbench_args
from cuawright.run.osworld import (
    DEFAULT_OSWORLD_DISK_SIZE_GB,
    OSWorldRunConfig,
    _configure_docker_resources,
    _install_task_setup_compatibility,
    parse_args,
    remaining_retry_budget,
)
from cuawright.run.task import TaskConfig, TaskRunner


class RunCliHarnessDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        config = patch(
            "cuawright.run.osworld.configured_value",
            return_value="/images/repaired.qcow2",
        )
        self.configured_value = config.start()
        self.addCleanup(config.stop)

    def test_vm_uses_config_without_legacy_image_fallback(self) -> None:
        self.assertEqual(parse_args([]).path_to_vm, "/images/repaired.qcow2")
        self.assertEqual(
            parse_args(["--path-to-vm", "/images/explicit.qcow2"]).path_to_vm,
            "/images/explicit.qcow2",
        )
        self.configured_value.return_value = None
        with self.assertRaises(SystemExit):
            parse_args([])
        self.assertEqual(
            parse_args(["--path-to-vm", "/images/explicit.qcow2"]).path_to_vm,
            "/images/explicit.qcow2",
        )

    def test_osworld_default_guest_disk_size_remains_80_gb(self) -> None:
        self.assertEqual(DEFAULT_OSWORLD_DISK_SIZE_GB, 80)

    def test_explicit_500_step_runs_are_supported_without_changing_default(
        self,
    ) -> None:
        self.assertEqual(DEFAULT_TOTAL_STEPS, 300)
        self.assertEqual(MAX_TOTAL_STEPS, 500)
        self.assertEqual(SharedStepBudget(500).remaining_steps, 500)
        with self.assertRaisesRegex(ValueError, "hard maximum of 500"):
            SharedStepBudget(501)

    def test_docker_guest_resources_are_explicitly_overridable(self) -> None:
        desktop_env = SimpleNamespace(
            provider=SimpleNamespace(
                environment={
                    "RAM_SIZE": "4G",
                    "CPU_CORES": "4",
                }
            )
        )

        _configure_docker_resources(
            desktop_env,
            ram_size_gb=16,
            cpu_cores=8,
        )

        self.assertEqual(
            desktop_env.provider.environment,
            {"RAM_SIZE": "16G", "CPU_CORES": "8"},
        )
        arguments = parse_args(
            ["--docker-ram-size-gb", "16", "--docker-cpu-cores", "8"]
        )
        self.assertEqual(arguments.docker_ram_size_gb, 16)
        self.assertEqual(arguments.docker_cpu_cores, 8)

    def test_task_089_stages_its_local_asset_inside_the_guest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            assets_dir = Path(tmp)
            local_asset = assets_dir / "task_089" / "broken_presentation.zip"
            local_asset.parent.mkdir()
            local_asset.write_bytes(b"zip")

            module_name = "test_osworld_task_089_setup"
            task_module = ModuleType(module_name)
            task_module.PRESENTATION_ZIP_URL = str(local_asset)
            sys.modules[module_name] = task_module
            observed_urls: list[str] = []

            def original_setup(
                self: object,
                setup_controller: object,
                use_proxy: bool = False,
            ) -> None:
                setup_controller.mark_setup()
                observed_urls.append(task_module.PRESENTATION_ZIP_URL)

            task_type = type(
                "Task089",
                (),
                {"__module__": module_name, "setup": original_setup},
            )
            task = task_type()
            setup_controller = SimpleNamespace(download=Mock(), mark_setup=Mock())

            restore = _install_task_setup_compatibility(
                task,
                assets_dir=str(assets_dir),
                task_id="089",
            )
            try:
                task.setup(setup_controller)
            finally:
                restore()
                sys.modules.pop(module_name, None)

            setup_controller.download.assert_called_once_with(
                [
                    {
                        "url": str(local_asset),
                        "path": ("/tmp/cuawright-task-089/broken_presentation.zip"),
                    }
                ]
            )
            setup_controller.mark_setup.assert_called_once_with()
            self.assertEqual(
                observed_urls,
                ["file:///tmp/cuawright-task-089/broken_presentation.zip"],
            )
            self.assertEqual(task_module.PRESENTATION_ZIP_URL, str(local_asset))
            self.assertIs(task_type.setup, original_setup)

    def test_task_089_compatibility_is_scoped_to_june_release(self) -> None:
        task = SimpleNamespace(setup=Mock())

        restore = _install_task_setup_compatibility(
            task,
            assets_dir="/missing",
            task_id="089",
            benchmark_release="v2026.08.08",
        )
        restore()

        self.assertIsInstance(task.setup, Mock)

    def test_non_task_089_setup_is_unchanged(self) -> None:
        task = SimpleNamespace(setup=Mock())

        restore = _install_task_setup_compatibility(
            task,
            assets_dir="/missing",
            task_id="001",
        )
        restore()

        self.assertIsInstance(task.setup, Mock)

    def test_parser_defers_model_and_base_defaults_until_provider_is_known(
        self,
    ) -> None:
        for provider in ("openai", "openrouter-anthropic", "anthropic"):
            with self.subTest(provider=provider):
                arguments = parse_args(["--model-provider", provider])
                self.assertIsNone(arguments.model)
                self.assertIsNone(arguments.base_url)

    def test_removed_fixed_pipeline_flag_is_rejected(self) -> None:
        for flag in (
            "--fixed-pipeline",
            "--no-model-infrastructure",
        ):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parse_args([flag])

    def test_osworld_compaction_interval_can_be_overridden(self) -> None:
        self.assertEqual(parse_args(["--compact-every", "20"]).compact_every, 20)

    def test_osworld_benchmark_release_is_explicitly_selectable(self) -> None:
        self.assertEqual(parse_args([]).benchmark_release, "v2026.06.24")
        self.assertEqual(
            parse_args(
                ["--benchmark-release", "osworld-v2-2026.08.08"]
            ).benchmark_release,
            "v2026.08.08",
        )
        with self.assertRaises(SystemExit):
            parse_args(["--benchmark-release", "v2099.01.01"])

    def test_task_proxy_can_be_explicitly_disabled(self) -> None:
        self.assertTrue(parse_args(["--disable-task-proxy"]).disable_task_proxy)

    def test_benchmark_clis_fix_compaction_at_the_task_config_default(self) -> None:
        parser_argv = (
            (
                parse_benchcad_args,
                ["cuawright-benchcad", "--manifest", "m", "--record-id", "r"],
            ),
            (
                parse_cadgenbench_args,
                ["cuawright-cadgenbench", "--dataset-dir", "d", "--fixture-id", "f"],
            ),
        )
        for parser, argv in parser_argv:
            with self.subTest(parser=parser.__module__):
                with patch("sys.argv", argv):
                    arguments = parser()
                self.assertFalse(hasattr(arguments, "compact_every"))

                with patch("sys.argv", [*argv, "--compact-every", "1"]):
                    with self.assertRaises(SystemExit):
                        parser()

    def test_benchcad_direct_mode_remains_available(self) -> None:
        with patch(
            "sys.argv",
            [
                "cuawright-benchcad",
                "--manifest",
                "m",
                "--record-id",
                "r",
                "--mode",
                "direct",
            ],
        ):
            arguments = parse_benchcad_args()
        self.assertEqual(arguments.mode, "direct")

    def test_no_subagents_flag_is_available_for_terminal_benchmarks(self) -> None:
        self.assertTrue(parse_args(["--no-subagents"]).no_subagents)
        parser_argv = (
            (
                parse_benchcad_args,
                [
                    "cuawright-benchcad",
                    "--manifest",
                    "m",
                    "--record-id",
                    "r",
                    "--no-subagents",
                ],
            ),
            (
                parse_cadgenbench_args,
                [
                    "cuawright-cadgenbench",
                    "--dataset-dir",
                    "d",
                    "--fixture-id",
                    "f",
                    "--no-subagents",
                ],
            ),
        )
        for parser, argv in parser_argv:
            with self.subTest(parser=parser.__module__), patch("sys.argv", argv):
                self.assertTrue(parser().no_subagents)

    def test_default_phyagi_key_resolves_above_worktrees(self) -> None:
        self.assertEqual(
            Path(parse_args([]).api_key_file),
            Path(__file__).resolve().parents[3] / ".a-creds",
        )

    def test_model_infrastructure_is_configurable(self) -> None:
        self.assertEqual(parse_args([]).model_infrastructure, "aiservices")
        self.assertEqual(
            parse_args(["--model-infrastructure", "trapi"]).model_infrastructure,
            "trapi",
        )

    def test_provider_model_defaults(self) -> None:
        self.assertEqual(resolve_model("openai", None), "gpt-5.5")
        self.assertEqual(
            resolve_model("openrouter-anthropic", None),
            "anthropic/claude-opus-5",
        )
        self.assertEqual(resolve_model("anthropic", None), "claude-opus-5")
        self.assertEqual(resolve_model("anthropic", "custom-model"), "custom-model")

    def test_provider_base_url_defaults(self) -> None:
        self.assertEqual(
            resolve_base_url("openai", None),
            "https://maif-gateway.azure-api.net/api",
        )
        self.assertEqual(
            resolve_base_url("openrouter-anthropic", None),
            "https://openrouter.ai/api",
        )
        self.assertEqual(
            resolve_base_url("anthropic", None),
            "https://api.anthropic.com",
        )
        self.assertEqual(
            resolve_base_url("anthropic", "https://anthropic.example"),
            "https://anthropic.example",
        )

    def test_task_and_user_simulator_use_the_migrated_gateway_default(self) -> None:
        expected = "https://maif-gateway.azure-api.net/api"
        self.assertEqual(TaskConfig("001", "unused-key", "results").base_url, expected)
        self.assertEqual(
            OSWorldRunConfig("001", "unused-key", "results").user_simulator_base_url,
            expected,
        )
        self.assertEqual(parse_args([]).user_simulator_base_url, expected)
        self.assertEqual(
            parse_args(
                ["--user-simulator-base-url", "https://simulator.example/api"]
            ).user_simulator_base_url,
            "https://simulator.example/api",
        )

    def test_resolution_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
            resolve_model("unknown", None)
        with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
            resolve_base_url("unknown", None)

    def test_infrastructure_retry_uses_only_remaining_budget(self) -> None:
        self.assertIsNone(remaining_retry_budget(None, 12.0))
        self.assertAlmostEqual(remaining_retry_budget(20.0, 7.5), 12.5)
        with self.assertRaises(DollarBudgetExceeded):
            remaining_retry_budget(20.0, 20.0)

    def test_terminal_threads_preserve_32k_output_default(self) -> None:
        self.assertEqual(DEFAULT_MAX_OUTPUT_TOKENS, 32_768)
        self.assertEqual(
            TaskRunner.new_model_thread.__kwdefaults__["max_output_tokens"],
            DEFAULT_MAX_OUTPUT_TOKENS,
        )

    def test_actor_and_specialists_have_distinct_question_caches(self) -> None:
        runner = object.__new__(TaskRunner)
        runner.ask_user = Mock(side_effect=["actor answer", "specialist answer"])
        runner.trace = Mock()
        runner._actor_answered_questions = {}
        runner._specialist_answered_questions = {}

        actor_first = runner._answer_actor_question("Which date?")
        actor_repeat = runner._answer_actor_question("  which   DATE? ")
        specialist_first = runner._answer_specialist_question("Which date?")
        specialist_repeat = runner._answer_specialist_question("which date?")

        self.assertEqual(actor_first, "actor answer")
        self.assertEqual(actor_repeat, actor_first)
        self.assertEqual(specialist_first, "specialist answer")
        self.assertEqual(specialist_repeat, specialist_first)
        self.assertEqual(runner.ask_user.call_count, 2)
        roles = [call.args[1]["role"] for call in runner.trace.event.call_args_list]
        self.assertEqual(
            roles,
            [
                "terminal-actor",
                "terminal-actor",
                "terminal-specialist",
                "terminal-specialist",
            ],
        )

    def test_task_runner_constructs_the_osworld_bundle_topology(self) -> None:
        runner = object.__new__(TaskRunner)
        runner.config = SimpleNamespace(
            task_id="linear-runtime",
            total_steps=10,
            compact_every=0,
            model_provider="openai",
        )
        runner.trace = Mock()

        runtime = runner.create_runtime(
            Mock(),
            instruction="Complete the task.",
            prompt_bundle=select_osworld_prompt_bundle(no_subagents=True),
        )

        self.assertEqual(runtime.agent_topology, "linear")
        self.assertEqual(runtime.enabled_specialist_roles, ())

    def test_outcome_payload_uses_host_report_artifact(self) -> None:
        report = SimpleNamespace(
            status="complete",
            artifact=ArtifactRef(
                "specialists/phase-001/binder-01/report.json",
                "a" * 64,
                100,
                "application/json",
            ),
        )
        state = SimpleNamespace(
            accepted_reports={"binder": report},
            submission_forced=False,
            submission_missing_reports=(),
            submission=ArtifactRef(
                "submissions/phase-001/submission.json",
                "b" * 64,
                200,
                "application/json",
            ),
        )
        outcome = SimpleNamespace(
            state=state,
            loop=SimpleNamespace(
                completed=True,
                stop_reason="completed",
                steps_used=4,
                commands_executed=3,
                compactions=0,
            ),
        )
        runtime = SimpleNamespace(
            agent_topology="orchestrated",
            budget=SimpleNamespace(snapshot=lambda: {"used_steps": 4}),
        )

        submission, verification = TaskRunner.outcome_payload(outcome, runtime)

        self.assertEqual(submission["status"], "submitted")
        self.assertEqual(
            verification["reports"]["binder"],
            {
                "status": "complete",
                "report_path": "specialists/phase-001/binder-01/report.json",
                "report_sha256": "a" * 64,
            },
        )


if __name__ == "__main__":
    unittest.main()

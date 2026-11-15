from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cuawright.run.batch import (
    BatchConfig,
    BatchCoordinator,
    TASK_EXCLUSIONS,
    TASK_IDS,
    _selected_task_ids,
    main,
    parse_args,
)
from cuawright.operations.scheduler import latest_run_dir, objective_health_failure


class TerminalBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        config = patch(
            "cuawright.run.batch.configured_value",
            return_value="/images/repaired.qcow2",
        )
        self.configured_value = config.start()
        self.addCleanup(config.stop)

    def test_vm_uses_persistent_config_and_explicit_override(self) -> None:
        args = ["--result-root", "/results"]
        self.assertEqual(parse_args(args).path_to_vm, "/images/repaired.qcow2")
        self.assertEqual(
            parse_args(args + ["--path-to-vm", "/images/explicit.qcow2"]).path_to_vm,
            "/images/explicit.qcow2",
        )

    def test_vm_is_required_without_config_not_guessed_from_checkout(self) -> None:
        self.configured_value.return_value = None
        with self.assertRaises(SystemExit):
            parse_args(["--result-root", "/results"])
        self.assertEqual(
            parse_args(
                ["--result-root", "/results", "--path-to-vm", "/images/explicit.qcow2"]
            ).path_to_vm,
            "/images/explicit.qcow2",
        )

    def test_exact_supported_task_set(self) -> None:
        self.assertEqual(len(TASK_IDS), 105)
        self.assertEqual(TASK_IDS[0], "001")
        self.assertEqual(TASK_IDS[-1], "108")
        self.assertEqual(
            TASK_EXCLUSIONS,
            {
                "026": "GitLab",
                "041": "GitLab",
                "082": "unsupported nested-Docker guest setup",
            },
        )
        for task_id in TASK_EXCLUSIONS:
            self.assertNotIn(task_id, TASK_IDS)

    def test_latest_run_is_confined_to_one_task_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "task-010-old"
            newer = root / "task-010-new"
            other = root / "task-011-newest"
            for path in (older, newer, other):
                path.mkdir()
            older.touch()
            newer.touch()
            other.touch()
            older_mtime = older.stat().st_mtime
            newer_mtime = older_mtime + 10
            other_mtime = newer_mtime + 10
            older.chmod(0o700)
            import os

            os.utime(older, (older_mtime, older_mtime))
            os.utime(newer, (newer_mtime, newer_mtime))
            os.utime(other, (other_mtime, other_mtime))

            self.assertEqual(latest_run_dir(root, "010"), newer)

    def test_website_suffix_is_explicitly_configurable(self) -> None:
        arguments = parse_args(
            [
                "--result-root",
                "/tmp/results",
                "--website-host-suffix",
                "127.0.0.1.nip.io",
            ]
        )
        self.assertEqual(
            arguments.website_host_suffix,
            "127.0.0.1.nip.io",
        )

    def test_model_reasoning_and_infrastructure_are_configurable(self) -> None:
        arguments = parse_args(
            [
                "--result-root",
                "/tmp/results",
                "--model-provider",
                "anthropic",
                "--model",
                "claude-fable-5-1",
                "--reasoning-effort",
                "max",
                "--model-infrastructure",
                "anthropic-foundry",
                "--docker-ram-size-gb",
                "32",
                "--docker-cpu-cores",
                "8",
                "--user-simulator-api-key-file",
                "/tmp/openai-key",
                "--user-simulator-base-url",
                "https://simulator.example/api",
                "--anthropic-max-output-tokens",
                "64000",
            ]
        )
        self.assertEqual(arguments.model_provider, "anthropic")
        self.assertEqual(arguments.model, "claude-fable-5-1")
        self.assertEqual(arguments.reasoning_effort, "max")
        self.assertEqual(arguments.model_infrastructure, "anthropic-foundry")
        self.assertEqual(arguments.docker_ram_size_gb, 32)
        self.assertEqual(arguments.docker_cpu_cores, 8)
        self.assertEqual(
            arguments.user_simulator_api_key_file,
            "/tmp/openai-key",
        )
        self.assertEqual(
            arguments.user_simulator_base_url,
            "https://simulator.example/api",
        )
        self.assertEqual(arguments.anthropic_max_output_tokens, 64000)

    def test_e_l1_prompt_profile_is_explicit_and_requires_linear_topology(self) -> None:
        arguments = parse_args(
            [
                "--result-root",
                "/tmp/results",
                "--no-subagents",
                "--prompt-profile",
                "e-l1-direct",
            ]
        )
        self.assertTrue(arguments.no_subagents)
        self.assertEqual(arguments.prompt_profile, "e-l1-direct")

        invalid = parse_args(
            ["--result-root", "/tmp/results", "--prompt-profile", "e-l1-direct"]
        )
        with (
            patch("cuawright.run.batch.parse_args", return_value=invalid),
            patch("cuawright.run.batch.BatchCoordinator") as coordinator,
            self.assertRaisesRegex(ValueError, "requires --no-subagents"),
        ):
            main()
        coordinator.assert_not_called()

    def test_base_url_defaults_and_can_be_overridden(self) -> None:
        arguments = ["--result-root", "results"]
        self.assertEqual(
            parse_args(arguments).base_url,
            "https://maif-gateway.azure-api.net/api",
        )
        base_url = "https://trapi.research.microsoft.com/gcr/shared/openai/v1"
        self.assertEqual(
            parse_args(arguments + ["--base-url", base_url]).base_url,
            base_url,
        )

    def test_empty_base_url_is_rejected_before_launch(self) -> None:
        args = parse_args(["--result-root", "results", "--base-url", " \t"])
        with (
            patch("cuawright.run.batch.parse_args", return_value=args),
            patch("cuawright.run.batch.BatchCoordinator") as coordinator,
            self.assertRaisesRegex(ValueError, "base-url cannot be empty"),
        ):
            main()
        coordinator.assert_not_called()

    def test_explicit_task_selection_is_normalized_and_validated(self) -> None:
        self.assertEqual(_selected_task_ids(["2", "020"]), ("002", "020"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _selected_task_ids(["026"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _selected_task_ids(["082"])
        with self.assertRaisesRegex(ValueError, "unique"):
            _selected_task_ids(["002", "2"])

    def test_default_manifest_records_supported_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proxy = root / "proxy.json"
            proxy.write_text("{}")
            coordinator = BatchCoordinator(
                BatchConfig(
                    result_root=root / "results",
                    task_ids=TASK_IDS,
                    width=46,
                    vm_path=root / "repaired.qcow2",
                    assets_dir=root / "assets",
                    api_key_file=root / "key",
                    proxy_config=proxy,
                    website_host_suffix="127.0.0.1.nip.io",
                    model="gpt-5.5",
                    reasoning_effort="xhigh",
                    model_infrastructure="aiservices",
                    health_delay_seconds=300,
                )
            )

            coordinator._prepare_root("commit")
            manifest = json.loads(
                (root / "results" / "batch-manifest.json").read_text()
            )

            self.assertEqual(manifest["task_count"], 105)
            self.assertEqual(
                manifest["selection"],
                "full-excluding-gitlab-and-nested-docker",
            )
            self.assertEqual(manifest["excluded"], TASK_EXCLUSIONS)
            self.assertEqual(
                manifest["base_url"], "https://maif-gateway.azure-api.net/api"
            )

    def test_custom_base_url_is_recorded_and_forwarded_to_each_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proxy = root / "proxy.json"
            proxy.write_text("{}")
            base_url = "https://trapi.research.microsoft.com/gcr/shared/openai/v1"
            coordinator = BatchCoordinator(
                BatchConfig(
                    result_root=root / "results",
                    task_ids=("001", "002"),
                    width=1,
                    vm_path=root / "vm.qcow2",
                    assets_dir=root / "assets",
                    api_key_file=root / "unused-key",
                    proxy_config=proxy,
                    website_host_suffix="127.0.0.1.nip.io",
                    base_url=base_url,
                    model="gpt-5.6-sol",
                    reasoning_effort="max",
                    model_infrastructure="trapi",
                    health_delay_seconds=300,
                    no_subagents=True,
                    prompt_profile="e-l1-direct",
                    disable_task_proxy=True,
                    openai_policy_feedback_max_retries=10,
                )
            )
            coordinator._prepare_root("commit")
            manifest = json.loads(
                (root / "results" / "batch-manifest.json").read_text()
            )
            self.assertEqual(manifest["base_url"], base_url)
            self.assertEqual(manifest["provider"], "openai")
            self.assertEqual(manifest["infrastructure"], "trapi")
            self.assertEqual(manifest["prompt_profile"], "e-l1-direct")
            self.assertTrue(manifest["disable_task_proxy"])
            self.assertEqual(manifest["openai_policy_feedback_max_retries"], 10)
            process = Mock(pid=12345)
            process.wait.return_value = 0
            with (
                patch(
                    "cuawright.run.batch.subprocess.Popen", return_value=process
                ) as start,
                patch("cuawright.run.batch.threading.Thread"),
                patch.object(coordinator, "_write_task_record"),
            ):
                for task_id in coordinator.config.task_ids:
                    coordinator._run_task(task_id)
            self.assertEqual(start.call_count, 2)
            for call in start.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--base-url") + 1], base_url)
                self.assertEqual(
                    command[command.index("--model-provider") + 1], "openai"
                )
                self.assertEqual(
                    command[command.index("--model-infrastructure") + 1], "trapi"
                )
                self.assertIn("--no-subagents", command)
                self.assertIn("--disable-task-proxy", command)
                self.assertEqual(
                    command[command.index("--openai-policy-feedback-retries") + 1],
                    "10",
                )
                self.assertEqual(
                    command[command.index("--prompt-profile") + 1],
                    "e-l1-direct",
                )

    def test_anthropic_provider_and_docker_resources_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proxy = root / "proxy.json"
            proxy.write_text("{}")
            base_url = "https://example.services.ai.azure.com/anthropic"
            coordinator = BatchCoordinator(
                BatchConfig(
                    result_root=root / "results",
                    task_ids=("048",),
                    width=1,
                    vm_path=root / "vm.qcow2",
                    assets_dir=root / "assets",
                    api_key_file=root / "anthropic-key",
                    proxy_config=proxy,
                    website_host_suffix="127.0.0.1.nip.io",
                    model_provider="anthropic",
                    base_url=base_url,
                    docker_ram_size_gb=32,
                    docker_cpu_cores=8,
                    user_simulator_api_key_file=root / "simulator-key",
                    user_simulator_base_url="https://simulator.example/api",
                    anthropic_max_output_tokens=64000,
                    model="claude-fable-5-1",
                    reasoning_effort="max",
                    model_infrastructure="anthropic-foundry",
                    health_delay_seconds=300,
                    no_subagents=True,
                    prompt_profile="e-l1-direct",
                )
            )
            coordinator._prepare_root("commit")
            manifest = json.loads(
                (root / "results" / "batch-manifest.json").read_text()
            )
            self.assertEqual(manifest["provider"], "anthropic")
            self.assertEqual(manifest["docker_ram_size_gb"], 32)
            self.assertEqual(manifest["docker_cpu_cores"], 8)
            self.assertEqual(
                manifest["user_simulator_base_url"],
                "https://simulator.example/api",
            )
            self.assertEqual(manifest["anthropic_max_output_tokens"], 64000)
            process = Mock(pid=12345)
            process.wait.return_value = 0
            with (
                patch(
                    "cuawright.run.batch.subprocess.Popen", return_value=process
                ) as start,
                patch("cuawright.run.batch.threading.Thread"),
                patch.object(coordinator, "_write_task_record"),
            ):
                coordinator._run_task("048")
            command = start.call_args.args[0]
            self.assertEqual(
                command[command.index("--model-provider") + 1], "anthropic"
            )
            self.assertEqual(command[command.index("--docker-ram-size-gb") + 1], "32")
            self.assertEqual(command[command.index("--docker-cpu-cores") + 1], "8")
            self.assertEqual(
                command[command.index("--user-simulator-api-key-file") + 1],
                str(root / "simulator-key"),
            )
            self.assertEqual(
                command[command.index("--user-simulator-base-url") + 1],
                "https://simulator.example/api",
            )
            self.assertEqual(
                command[command.index("--anthropic-max-output-tokens") + 1],
                "64000",
            )

    def test_signal_handler_only_sets_stop_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coordinator = BatchCoordinator(
                BatchConfig(
                    result_root=root / "results",
                    task_ids=("002",),
                    width=1,
                    vm_path=root / "vm.qcow2",
                    assets_dir=root / "assets",
                    api_key_file=root / "key",
                    proxy_config=root / "proxy.json",
                    website_host_suffix="127.0.0.1.nip.io",
                    model="gpt-5.5",
                    reasoning_effort="xhigh",
                    model_infrastructure="aiservices",
                    health_delay_seconds=300,
                )
            )
            coordinator._stop_all_processes = Mock()

            coordinator._handle_signal(2, object())

            self.assertTrue(coordinator.stop_event.is_set())
            coordinator._stop_all_processes.assert_not_called()

    def test_objective_health_failure_is_counted(self) -> None:
        self.assertTrue(objective_health_failure({"objective_stop_required": True}))
        self.assertFalse(objective_health_failure({"objective_stop_required": False}))


if __name__ == "__main__":
    unittest.main()

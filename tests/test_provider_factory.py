from __future__ import annotations

import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock, patch

from cuawright.models.anthropic_client import ClaudeClientConfig
from cuawright.providers.factory import (
    ProviderRuntime,
    ProviderSpec,
    create_provider_runtime,
)


class ProviderFactoryTests(unittest.TestCase):
    def test_provider_spec_excludes_benchmark_topology(self) -> None:
        self.assertNotIn(
            "agent_topology", {field.name for field in fields(ProviderSpec)}
        )

    def test_openai_construction_preserves_client_options(self) -> None:
        spec = ProviderSpec(
            name="openai",
            api_key_file="api-key",
            base_url="https://gateway.example/api",
            model="gpt-5.5",
            reasoning_effort="xhigh",
        )
        with (
            patch.object(Path, "read_text", return_value=" secret \n"),
            patch(
                "cuawright.providers.factory.create_responses_client"
            ) as create_client,
        ):
            runtime = create_provider_runtime(spec, run_dir=Path("results/run"))

        create_client.assert_called_once_with(
            api_key="secret",
            base_url="https://gateway.example/api",
        )
        self.assertEqual(runtime.api_key, "secret")
        self.assertIs(runtime.client, create_client.return_value)
        self.assertIsNone(runtime.cost_tracker)
        self.assertIsNone(runtime.async_runner)

    def test_openrouter_construction_preserves_headers_retries_and_costs(
        self,
    ) -> None:
        spec = ProviderSpec(
            name="openrouter-anthropic",
            api_key_file="api-key",
            base_url="https://openrouter.example/api",
            model="anthropic/claude-opus-5",
            reasoning_effort="high",
            max_dollars=12.5,
        )
        with (
            patch("cuawright.providers.factory.read_api_key", return_value="secret"),
            patch("cuawright.providers.factory.OpenRouterCostTracker") as tracker_type,
            patch("cuawright.providers.factory.asyncio.Runner") as runner_type,
            patch("cuawright.providers.factory.ClaudeMessagesClient") as client_type,
        ):
            runtime = create_provider_runtime(spec, run_dir=Path("results/run"))

        tracker_type.assert_called_once_with(
            api_key="secret",
            max_dollars=12.5,
            artifact_path=Path("results/run/openrouter-cost.json"),
        )
        config = client_type.call_args.kwargs["config"]
        self.assertEqual(config.model, "anthropic/claude-opus-5")
        self.assertEqual(config.effort, "high")
        self.assertFalse(config.disable_parallel_tool_use)
        self.assertEqual(config.max_retries, 4)
        self.assertEqual(
            client_type.call_args.kwargs["default_headers"],
            {"Authorization": "******"},
        )
        self.assertEqual(
            client_type.call_args.kwargs["base_url"],
            "https://openrouter.example/api",
        )
        self.assertIs(runtime.cost_tracker, tracker_type.return_value)
        self.assertIs(runtime.async_runner, runner_type.return_value)
        self.assertIs(runtime.client, client_type.return_value)

    def test_direct_anthropic_construction_omits_openrouter_headers(self) -> None:
        spec = ProviderSpec(
            name="anthropic",
            api_key_file="api-key",
            base_url="https://anthropic.example/api",
            model="claude-opus-5",
            reasoning_effort="max",
            max_dollars=7.0,
        )
        with (
            patch("cuawright.providers.factory.read_api_key", return_value="secret"),
            patch("cuawright.providers.factory.AnthropicCostTracker") as tracker_type,
            patch("cuawright.providers.factory.asyncio.Runner") as runner_type,
            patch("cuawright.providers.factory.ClaudeMessagesClient") as client_type,
        ):
            runtime = create_provider_runtime(spec, run_dir=Path("results/run"))

        tracker_type.assert_called_once_with(
            max_dollars=7.0,
            artifact_path=Path("results/run/anthropic-cost.json"),
            pricing_model="claude-opus-5",
        )
        config = client_type.call_args.kwargs["config"]
        self.assertEqual(config.model, "claude-opus-5")
        self.assertEqual(config.effort, "max")
        self.assertFalse(config.disable_parallel_tool_use)
        self.assertEqual(config.max_retries, 4)
        self.assertEqual(config.policy_feedback_max_retries, 0)
        self.assertNotIn("default_headers", client_type.call_args.kwargs)
        self.assertIs(runtime.cost_tracker, tracker_type.return_value)
        self.assertIs(runtime.async_runner, runner_type.return_value)

    def test_foundry_anthropic_uses_compatible_request_profile(self) -> None:
        spec = ProviderSpec(
            name="anthropic",
            api_key_file="api-key",
            base_url="https://example.services.ai.azure.com/anthropic",
            model="claude-fable-5-1",
            reasoning_effort="max",
            anthropic_max_output_tokens=64_000,
        )
        with (
            patch("cuawright.providers.factory.read_api_key", return_value="secret"),
            patch("cuawright.providers.factory.AnthropicCostTracker"),
            patch("cuawright.providers.factory.asyncio.Runner"),
            patch("cuawright.providers.factory.ClaudeMessagesClient") as client_type,
        ):
            client_type.return_value.config = ClaudeClientConfig(
                model="claude-fable-5-1",
                server_compaction=False,
                thinking_prefix_mismatch_behavior="drop_block",
                disable_parallel_tool_use=True,
            )
            runtime = create_provider_runtime(spec, run_dir=Path("results/run"))

        config = client_type.call_args.kwargs["config"]
        self.assertFalse(config.server_compaction)
        self.assertEqual(config.thinking_prefix_mismatch_behavior, "drop_block")
        self.assertTrue(config.disable_parallel_tool_use)
        self.assertEqual(config.policy_feedback_max_retries, 10)
        self.assertIsNone(runtime.parallel_tool_suffix)

    def test_thread_construction_preserves_provider_request_configuration(
        self,
    ) -> None:
        spec = ProviderSpec(
            name="openrouter-anthropic",
            api_key_file="api-key",
            base_url="https://openrouter.example/api",
            model="anthropic/claude-opus-5",
            reasoning_effort="xhigh",
            infrastructure=None,
            openrouter_provider_preferences={"zdr": True},
            anthropic_min_recent_images=4,
            anthropic_image_prune_interval=5,
            anthropic_max_request_mb=20.0,
        )
        runtime = ProviderRuntime(
            spec=spec,
            api_key="secret",
            client=Mock(),
            cost_tracker=Mock(),
            claude_client=Mock(),
            async_runner=Mock(),
        )
        trace = Mock()
        with patch("cuawright.providers.factory.ClaudeMessagesThread") as thread_type:
            thread = runtime.new_thread(
                trace=trace,
                role="terminal-actor",
                system_prompt="system",
                user_prompt="user",
                tools=[{"name": "run_command"}],
                max_output_tokens=32_768,
            )

        thread_type.assert_called_once_with(
            client=runtime.claude_client,
            trace=trace,
            role="terminal-actor",
            system_prompt="system",
            user_prompt="user",
            tools=[{"name": "run_command"}],
            max_output_tokens=32_768,
            cost_tracker=runtime.cost_tracker,
            provider="openrouter-anthropic",
            async_runner=runtime.async_runner,
            provider_preferences={"zdr": True},
            compaction_reasoning_effort=None,
            anthropic_min_recent_images=4,
            anthropic_image_prune_interval=5,
            anthropic_max_request_mb=20.0,
        )
        self.assertIs(thread, thread_type.return_value)


if __name__ == "__main__":
    unittest.main()

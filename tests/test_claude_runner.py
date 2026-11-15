from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx2
from anthropic import APIStatusError, transform_schema
from PIL import Image

from cuawright.models.anthropic_cost import AnthropicCostTracker
from cuawright.models.anthropic_client import (
    ClaudeClientConfig,
    ClaudeStreamParseError,
    ClaudeTurn,
    ClaudeUsage,
    build_create_request,
    serialized_request_bytes,
)
from cuawright.models.openrouter_cost import OpenRouterGenerationCost
from cuawright.models.anthropic_messages import (
    ANTHROPIC_TRANSIENT_RETRY_DELAYS,
    CLAUDE_IMAGE_PLACEHOLDER,
    ClaudeMessagesThread,
    ClaudeRequestTooLargeError,
    _anthropic_tools,
    _claude_image_blocks,
    _claude_image_slots,
    _prepare_claude_image,
    _prune_claude_images,
)
from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA
from cuawright.run.task import TaskConfig, TaskRunner


class ClaudeMessagesThreadTests(unittest.TestCase):
    def test_transient_retries_span_an_output_rate_limit_window(self) -> None:
        self.assertEqual(
            ANTHROPIC_TRANSIENT_RETRY_DELAYS,
            (2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        )

    def test_direct_transient_status_retries_same_low_effort_request(self) -> None:
        response = httpx2.Response(
            529,
            request=httpx2.Request("POST", "https://anthropic.example/messages"),
        )
        overload = APIStatusError(
            "overloaded",
            response=response,
            body={"error": {"type": "overloaded_error"}},
        )
        message = SimpleNamespace(
            id="msg_success",
            model="claude-opus-5",
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
                output_tokens_details=SimpleNamespace(thinking_tokens=0),
            ),
            model_dump=lambda mode="json": {
                "id": "msg_success",
                "model": "claude-opus-5",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
            },
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5"),
            create_prepared_turn=AsyncMock(
                side_effect=[overload, ClaudeTurn.from_message(message)]
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=None,
                artifact_path=Path(tmp) / "anthropic-cost.json",
            )
            trace = Mock()
            thread = ClaudeMessagesThread(
                client=client,
                trace=trace,
                role="actor",
                system_prompt="system",
                user_prompt="user",
                tools=[],
                max_output_tokens=1_024,
                cost_tracker=tracker,
                provider="anthropic",
            )

            with patch("cuawright.models.anthropic_messages.time.sleep") as sleep:
                thread.call(reasoning_effort="low")

        self.assertEqual(client.create_prepared_turn.await_count, 2)
        first = client.create_prepared_turn.call_args_list[0].kwargs["request"]
        second = client.create_prepared_turn.call_args_list[1].kwargs["request"]
        self.assertIs(first, second)
        self.assertEqual(first["output_config"], {"effort": "low"})
        sleep.assert_called_once_with(2.0)
        trace.event.assert_any_call(
            "anthropic_transient_retry",
            {
                "role": "actor",
                "attempt": 1,
                "delay_seconds": 2.0,
                "status_code": 529,
            },
        )

    def test_anthropic_history_policy_config_rejects_invalid_values(self) -> None:
        required = {
            "task_id": "001",
            "api_key_file": "key",
            "result_root": "results",
        }
        invalid = (
            ("anthropic_min_recent_images", True, TypeError),
            ("anthropic_min_recent_images", 1.5, TypeError),
            ("anthropic_min_recent_images", -1, ValueError),
            ("anthropic_image_prune_interval", False, TypeError),
            ("anthropic_image_prune_interval", 0, ValueError),
            ("anthropic_max_request_mb", "24", TypeError),
            ("anthropic_max_request_mb", float("nan"), ValueError),
            ("anthropic_max_request_mb", float("inf"), ValueError),
            ("anthropic_max_request_mb", 0, ValueError),
            ("anthropic_max_request_mb", 32.1, ValueError),
        )
        for field, value, error in invalid:
            with self.subTest(field=field, value=value):
                with self.assertRaises(error):
                    TaskConfig(**required, **{field: value})

    def test_provider_preferences_are_rejected_outside_openrouter(self) -> None:
        required = {
            "task_id": "001",
            "api_key_file": "key",
            "result_root": "results",
        }
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(
                    ValueError,
                    "openrouter_provider_preferences",
                ):
                    TaskConfig(
                        **required,
                        model_provider=provider,
                        openrouter_provider_preferences={"zdr": True},
                    )

        client = SimpleNamespace(config=ClaudeClientConfig())
        with self.assertRaisesRegex(ValueError, "provider_preferences"):
            self.thread(
                client,
                provider="anthropic",
                provider_preferences={"zdr": True},
            )

    def test_direct_harness_rejects_models_without_known_pricing(self) -> None:
        with self.assertRaisesRegex(ValueError, "cost accounting"):
            TaskConfig(
                task_id="001",
                api_key_file="key",
                result_root="results",
                model_provider="anthropic",
                model="claude-sonnet-4",
            )

    def test_direct_harness_passes_base_override_and_enables_parallel_tools(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = root / "key"
            key.write_text("secret", encoding="utf-8")
            key.chmod(0o600)
            with patch(
                "cuawright.providers.factory.ClaudeMessagesClient"
            ) as client_type:
                runner = TaskRunner(
                    TaskConfig(
                        task_id="001",
                        api_key_file=str(key),
                        result_root=str(root / "results"),
                        model_provider="anthropic",
                        model="claude-opus-5",
                        base_url="https://anthropic.example/api",
                    )
                )
                try:
                    arguments = client_type.call_args.kwargs
                    self.assertEqual(
                        arguments["base_url"],
                        "https://anthropic.example/api",
                    )
                    self.assertFalse(
                        arguments["config"].disable_parallel_tool_use,
                    )
                finally:
                    assert runner.async_runner is not None
                    runner.async_runner.close()

    def test_direct_tool_schema_removes_unsupported_numeric_bounds(self) -> None:
        tools = [
            {
                "name": "run_command",
                "description": "run",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timeout": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 600,
                        },
                        "reports": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "maxLength": 1_000,
                            },
                            "minItems": 1,
                            "maxItems": 12,
                        },
                    },
                },
            }
        ]

        direct = _anthropic_tools(tools, direct=True)
        openrouter = _anthropic_tools(tools)

        timeout = direct[0]["input_schema"]["properties"]["timeout"]
        reports = direct[0]["input_schema"]["properties"]["reports"]
        self.assertEqual(
            timeout,
            {
                "type": "integer",
                "description": "{minimum: 1, maximum: 600}",
            },
        )
        self.assertEqual(
            reports,
            {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "{maxLength: 1000}",
                },
                "minItems": 1,
                "description": "{maxItems: 12}",
            },
        )
        self.assertIn(
            "minimum",
            openrouter[0]["input_schema"]["properties"]["timeout"],
        )

    def test_sole_run_command_schema_passes_strict_transform_preflight(self) -> None:
        transformed = _anthropic_tools(
            [RUN_COMMAND_TOOL_SCHEMA],
            direct=True,
        )
        self.assertEqual([tool["name"] for tool in transformed], ["run_command"])
        self.assertEqual(
            transformed[0]["input_schema"],
            transform_schema(RUN_COMMAND_TOOL_SCHEMA["parameters"]),
        )

    @staticmethod
    def thread(
        client: object,
        *,
        provider: str = "openrouter-anthropic",
        **kwargs: object,
    ) -> ClaudeMessagesThread:
        tracker = Mock()
        tracker.snapshot.return_value = {
            "max_dollars": None,
            "total_cost_usd": 0.0,
        }
        return ClaudeMessagesThread(
            client=client,
            trace=Mock(),
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=1_024,
            cost_tracker=tracker,
            provider=provider,
            **kwargs,
        )

    @staticmethod
    def prepare_direct_request(thread: ClaudeMessagesThread) -> dict[str, object]:
        request = build_create_request(
            thread.client.config,
            messages=thread.messages,
            system=thread.system_prompt,
            max_tokens=1_024,
        )
        thread._apply_direct_request_policy(request)
        return request

    def test_resizes_images_for_many_image_provider_limits(self) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="anthropic/claude-opus-5",
                effort="max",
            )
        )
        thread = self.thread(client)
        source = io.BytesIO()
        Image.new("RGB", (2_576, 1_200), "white").save(source, format="PNG")

        thread.add_image("/tmp/large.png", "image/png", source.getvalue())

        block = thread.messages[-1]["content"][1]
        decoded = base64.b64decode(block["source"]["data"])
        with Image.open(io.BytesIO(decoded)) as image:
            self.assertEqual(image.size, (1_500, 699))
        self.assertEqual(block["source"]["media_type"], "image/jpeg")

    def test_compresses_large_png_payloads_without_changing_trace_source(self) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="claude-opus-5",
                effort="max",
            )
        )
        thread = self.thread(client)
        source = io.BytesIO()
        Image.effect_noise((1_920, 1_080), 100).convert("RGB").save(
            source,
            format="PNG",
        )
        original = source.getvalue()

        thread.add_image("/tmp/frame.png", "image/png", original)

        block = thread.messages[-1]["content"][1]
        decoded = base64.b64decode(block["source"]["data"])
        self.assertEqual(block["source"]["media_type"], "image/jpeg")
        self.assertLessEqual(len(decoded), 400_000)

    def test_direct_preserves_desktop_geometry_and_reports_resize_mapping(
        self,
    ) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max")
        )
        thread = self.thread(client, provider="anthropic")
        desktop = io.BytesIO()
        Image.new("RGB", (1_920, 1_080), "white").save(desktop, format="PNG")

        thread.add_image("/home/oai/share/desktop.png", "image/png", desktop.getvalue())

        description = json.loads(thread.messages[-1]["content"][0]["text"])
        image_block = thread.messages[-1]["content"][1]
        with Image.open(
            io.BytesIO(base64.b64decode(image_block["source"]["data"]))
        ) as sent:
            self.assertEqual(sent.size, (1_920, 1_080))
        self.assertNotIn("image_geometry", description)

        large = io.BytesIO()
        Image.new("RGB", (2_576, 1_200), "white").save(large, format="PNG")
        thread.add_image("/home/oai/share/large.png", "image/png", large.getvalue())

        description = json.loads(thread.messages[-1]["content"][0]["text"])
        geometry = description["image_geometry"]
        self.assertEqual(
            geometry["source_dimensions"],
            {"width": 2_576, "height": 1_200},
        )
        self.assertEqual(
            geometry["displayed_dimensions"],
            {"width": 1_920, "height": 894},
        )
        self.assertEqual(
            geometry["desktop_dimensions"],
            {"width": 1_920, "height": 1_080},
        )
        self.assertEqual(geometry["displayed_to_desktop_scale"]["x"], 1.0)
        self.assertIn("xdotool", geometry["coordinate_guidance"])

    def test_non_desktop_sized_image_reports_coordinate_space_without_resize(
        self,
    ) -> None:
        prepared = _prepare_claude_image(
            "image/png",
            self._png_bytes((800, 600)),
            direct=True,
        )
        blocks = _claude_image_blocks(
            prepared,
            {
                "status": "success",
                "image_geometry": {"stale": "must not survive"},
            },
        )

        geometry = json.loads(blocks[0]["text"])["image_geometry"]
        self.assertEqual(
            geometry["displayed_dimensions"],
            {"width": 800, "height": 600},
        )
        self.assertEqual(
            geometry["desktop_dimensions"],
            {"width": 1_920, "height": 1_080},
        )
        self.assertNotIn("stale", geometry)

    @staticmethod
    def _png_bytes(size: tuple[int, int]) -> bytes:
        output = io.BytesIO()
        Image.new("RGB", size, "white").save(output, format="PNG")
        return output.getvalue()

    def test_batched_tool_result_uses_the_same_image_geometry_metadata(self) -> None:
        source = io.BytesIO()
        Image.new("RGB", (2_400, 1_200), "white").save(source, format="PNG")
        prepared = _prepare_claude_image(
            "image/png",
            source.getvalue(),
            direct=True,
        )
        blocks = _claude_image_blocks(
            prepared,
            {"status": "success", "image_attached": "/home/oai/share/frame.png"},
        )
        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_image",
                    "content": blocks,
                }
            ],
        }

        description = json.loads(message["content"][0]["content"][0]["text"])
        self.assertEqual(
            description["image_geometry"]["displayed_dimensions"],
            {"width": 1_920, "height": 960},
        )
        self.assertEqual(
            description["image_geometry"]["desktop_dimensions"],
            {"width": 1_920, "height": 1_080},
        )
        self.assertEqual(
            message["content"][0]["content"][1]["type"],
            "image",
        )

    def test_recursive_image_discovery_and_pruning(self) -> None:
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "AAAA",
            },
        }
        messages = [
            {
                "role": "user",
                "content": [
                    image.copy(),
                    {
                        "type": "tool_result",
                        "tool_use_id": "one",
                        "content": [
                            {"type": "text", "text": "ok"},
                            image.copy(),
                            {"wrapper": {"deeper": [image.copy()]}},
                        ],
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_future",
                        "name": "future_tool",
                        "input": {
                            "nested": image.copy(),
                        },
                    },
                    {"type": "image", "source": {"type": "url", "url": "image"}},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "not base64!",
                        },
                    },
                ],
            }
        ]

        self.assertEqual(len(_claude_image_slots(messages)), 2)
        self.assertEqual(_prune_claude_images(messages, 2), 2)
        self.assertEqual(len(_claude_image_slots(messages)), 0)
        serialized = json.dumps(messages)
        self.assertEqual(serialized.count(CLAUDE_IMAGE_PLACEHOLDER), 2)
        self.assertEqual(serialized.count('"type": "image"'), 4)

    def test_direct_image_history_is_bounded_in_cache_stable_intervals(
        self,
    ) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max")
        )
        thread = self.thread(
            client,
            provider="anthropic",
            anthropic_min_recent_images=3,
            anthropic_image_prune_interval=4,
            anthropic_max_request_mb=1.0,
        )
        omitted_counts: list[int] = []
        sizes: list[int] = []
        for index in range(1, 61):
            thread.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"image-{index}",
                            "content": [
                                {"type": "text", "text": f"image {index}"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "A" * 120_000,
                                    },
                                },
                            ],
                        }
                    ],
                }
            )
            request = self.prepare_direct_request(thread)
            serialized = json.dumps(request)
            omitted_counts.append(serialized.count(CLAUDE_IMAGE_PLACEHOLDER))
            sizes.append(serialized_request_bytes(request))

        changed_at = [
            index
            for index in range(2, 61)
            if omitted_counts[index - 1] != omitted_counts[index - 2]
        ]
        self.assertEqual(changed_at, list(range(7, 61, 4)))
        self.assertLessEqual(max(sizes), 1_000_000)
        self.assertEqual(omitted_counts[-1], 56)
        self.assertEqual(
            json.dumps(thread.messages).count(CLAUDE_IMAGE_PLACEHOLDER),
            56,
        )
        self.assertEqual(len(_claude_image_slots(thread.messages)), 4)

    def test_direct_payload_guard_fails_before_transport(self) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max"),
            create_prepared_turn=AsyncMock(),
        )
        thread = self.thread(
            client,
            provider="anthropic",
            anthropic_min_recent_images=1,
            anthropic_image_prune_interval=4,
            anthropic_max_request_mb=0.0004,
        )
        thread.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "A" * 1_000,
                        },
                    }
                ],
            }
        )

        with self.assertRaisesRegex(
            ClaudeRequestTooLargeError,
            "no API request was sent",
        ):
            thread.call(tool_choice="none")

        client.create_prepared_turn.assert_not_awaited()
        preflight = thread.trace.event.call_args.args
        self.assertEqual(preflight[0], "anthropic_request_preflight")
        self.assertEqual(preflight[1]["status"], "blocked")

    def test_direct_payload_guard_force_prunes_to_recent_minimum(self) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max")
        )
        thread = self.thread(
            client,
            provider="anthropic",
            anthropic_min_recent_images=2,
            anthropic_image_prune_interval=10,
            anthropic_max_request_mb=0.5,
        )
        for index in range(4):
            thread.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": chr(65 + index) * 200_000,
                            },
                        }
                    ],
                }
            )

        request = self.prepare_direct_request(thread)

        serialized = json.dumps(request)
        self.assertEqual(serialized.count(CLAUDE_IMAGE_PLACEHOLDER), 2)
        event = thread.trace.event.call_args.args[1]
        self.assertTrue(event["forced_to_minimum"])
        self.assertEqual(event["forced_pruned_images"], 2)
        self.assertEqual(event["retained_images"], 2)
        self.assertLess(event["after_bytes"], event["before_bytes"])
        self.assertLessEqual(event["after_bytes"], event["limit_bytes"])
        self.assertEqual(
            thread.messages[1]["content"][0]["type"],
            "text",
        )
        self.assertEqual(
            thread.messages[2]["content"][0]["type"],
            "text",
        )
        self.assertEqual(
            thread.messages[3]["content"][0]["source"]["data"],
            "C" * 200_000,
        )
        self.assertEqual(
            thread.messages[4]["content"][0]["source"]["data"],
            "D" * 200_000,
        )
        self.assertLessEqual(event["full_serializations"], 3)

    def test_direct_payload_guard_stops_after_request_fits(self) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max")
        )
        thread = self.thread(
            client,
            provider="anthropic",
            anthropic_min_recent_images=1,
            anthropic_image_prune_interval=10,
            anthropic_max_request_mb=0.45,
        )
        for index in range(3):
            thread.messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": chr(65 + index) * 200_000,
                            },
                        }
                    ],
                }
            )

        first_request = self.prepare_direct_request(thread)
        first_messages = copy.deepcopy(thread.messages)
        second_request = self.prepare_direct_request(thread)
        second_messages = copy.deepcopy(thread.messages)

        self.assertEqual(
            json.dumps(first_messages).count(CLAUDE_IMAGE_PLACEHOLDER),
            1,
        )
        self.assertEqual(len(_claude_image_slots(first_messages)), 2)
        self.assertEqual(first_messages, second_messages)
        self.assertEqual(first_request, second_request)
        self.assertLessEqual(
            serialized_request_bytes(first_request),
            thread.anthropic_max_request_bytes,
        )
        first_event = thread.trace.event.call_args_list[-2].args[1]
        second_event = thread.trace.event.call_args_list[-1].args[1]
        self.assertEqual(first_event["forced_pruned_images"], 1)
        self.assertEqual(second_event["forced_pruned_images"], 0)
        self.assertFalse(first_event["forced_to_minimum"])

    def test_pruning_preserves_compaction_blocks(self) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max")
        )
        thread = self.thread(
            client,
            provider="anthropic",
            anthropic_min_recent_images=1,
            anthropic_image_prune_interval=2,
            anthropic_max_request_mb=1.0,
        )
        image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "A" * 1_000,
            },
        }
        compaction = {
            "type": "compaction",
            "content": "stable compacted history",
            "encrypted_content": "opaque-state",
        }
        thread.messages.extend(
            [
                {"role": "user", "content": [copy.deepcopy(image)]},
                {"role": "assistant", "content": [copy.deepcopy(compaction)]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "nested",
                            "content": [
                                copy.deepcopy(image),
                                copy.deepcopy(image),
                            ],
                        }
                    ],
                },
            ]
        )

        self.prepare_direct_request(thread)
        request_messages = thread.messages

        self.assertEqual(
            request_messages[2]["content"][0],
            compaction,
        )
        self.assertEqual(len(_claude_image_slots(request_messages)), 1)
        self.assertEqual(
            json.dumps(request_messages).count(CLAUDE_IMAGE_PLACEHOLDER),
            2,
        )

    def test_openrouter_does_not_apply_direct_history_or_size_policy(self) -> None:
        message = SimpleNamespace(
            id="gen_openrouter",
            model="anthropic/claude-opus-5",
            provider="provider",
            content=[SimpleNamespace(type="text", text="done")],
            usage=SimpleNamespace(),
            stop_reason="end_turn",
            _request_id="gen_openrouter",
            model_dump=lambda mode="json": {
                "id": "gen_openrouter",
                "content": [{"type": "text", "text": "done"}],
            },
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="anthropic/claude-opus-5"),
            create_prepared_turn=AsyncMock(
                return_value=ClaudeTurn.from_message(message)
            ),
        )
        thread = self.thread(
            client,
            anthropic_min_recent_images=0,
            anthropic_image_prune_interval=1,
            anthropic_max_request_mb=0.0004,
        )
        thread.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "image",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "A" * 5_000,
                                },
                            }
                        ],
                    }
                ],
            }
        )
        thread.cost_tracker.record_generation.return_value = OpenRouterGenerationCost(
            generation_id="gen_openrouter",
            cost_usd=0.1,
            model="anthropic/claude-opus-5",
            provider="provider",
            native_prompt_tokens=100,
            native_completion_tokens=20,
            native_cached_tokens=0,
            native_reasoning_tokens=0,
        )

        thread.call(tool_choice="none")

        sent_request = client.create_prepared_turn.call_args.kwargs["request"]
        sent_messages = sent_request["messages"]
        self.assertEqual(len(_claude_image_slots(sent_messages)), 1)
        self.assertNotIn(
            "anthropic_request_preflight",
            [call.args[0] for call in thread.trace.event.call_args_list],
        )

    def test_direct_long_horizon_image_pruning_soak_is_deterministic(
        self,
    ) -> None:
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5", effort="max")
        )

        def make_thread() -> ClaudeMessagesThread:
            return self.thread(
                client,
                provider="anthropic",
                anthropic_min_recent_images=3,
                anthropic_image_prune_interval=4,
                anthropic_max_request_mb=0.12,
            )

        left = make_thread()
        right = make_thread()
        previous_messages: list[dict[str, object]] | None = None
        previous_omitted = 0
        pruning_cycles = 0
        max_full_serializations = 0

        for turn in range(1, 301):
            additions: list[dict[str, object]] = [
                {
                    "role": "user",
                    "content": (
                        f"synthetic turn {turn}: "
                        "continue inspecting the same desktop state"
                    ),
                }
            ]
            if turn % 5 == 0:
                additions.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": f"image-{turn // 5}",
                                "content": [
                                    {"type": "text", "text": f"frame {turn}"},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": "A" * 9_000,
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                )
            if turn % 40 == 0:
                additions.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "compaction",
                                "content": f"compacted through turn {turn}",
                                "encrypted_content": f"opaque-{turn}",
                            }
                        ],
                    }
                )
            left.messages.extend(copy.deepcopy(additions))
            right.messages.extend(copy.deepcopy(additions))

            left_request = self.prepare_direct_request(left)
            right_request = self.prepare_direct_request(right)
            self.assertEqual(left.messages, right.messages)
            self.assertEqual(
                json.dumps(left_request, sort_keys=True),
                json.dumps(right_request, sort_keys=True),
            )
            self.assertLessEqual(
                serialized_request_bytes(left_request),
                left.anthropic_max_request_bytes,
            )

            omitted = json.dumps(left.messages).count(CLAUDE_IMAGE_PLACEHOLDER)
            if previous_messages is not None:
                if omitted == previous_omitted:
                    self.assertEqual(
                        left.messages[: len(previous_messages)],
                        previous_messages,
                    )
                else:
                    self.assertEqual(omitted - previous_omitted, 4)
                    pruning_cycles += 1
            previous_messages = copy.deepcopy(left.messages)
            previous_omitted = omitted
            max_full_serializations = max(
                max_full_serializations,
                left.trace.event.call_args.args[1]["full_serializations"],
            )

        self.assertEqual(len(_claude_image_slots(left.messages)), 4)
        self.assertEqual(previous_omitted, 56)
        self.assertGreaterEqual(pruning_cycles, 10)
        self.assertLessEqual(max_full_serializations, 2)
        self.assertEqual(
            sum(
                1
                for message in previous_messages or []
                for block in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else []
                )
                if isinstance(block, dict) and block.get("type") == "compaction"
            ),
            7,
        )

    def test_adapts_native_tools_and_preserves_assistant_blocks(self) -> None:
        tool_use = SimpleNamespace(
            type="tool_use",
            id="toolu_1",
            name="run_command",
            input={"command": "true", "timeout": 10},
        )
        extra_tool_use = SimpleNamespace(
            type="tool_use",
            id="toolu_2",
            name="call_binder",
            input={"request": "inspect", "max_steps": 5},
        )
        message = SimpleNamespace(
            id="gen_1",
            model="anthropic/claude-opus-5",
            provider="provider",
            content=[tool_use, extra_tool_use],
            usage=SimpleNamespace(),
            model_dump=lambda mode="json": {
                "id": "gen_1",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "run_command",
                        "input": {"command": "true", "timeout": 10},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "call_binder",
                        "input": {"request": "inspect", "max_steps": 5},
                    },
                ],
            },
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="anthropic/claude-opus-5",
                effort="max",
                disable_parallel_tool_use=False,
            ),
            create_prepared_turn=AsyncMock(
                return_value=ClaudeTurn.from_message(
                    SimpleNamespace(
                        **vars(message),
                        stop_reason="tool_use",
                        _request_id="gen_1",
                    )
                )
            ),
        )
        tracker = Mock()
        tracker.snapshot.return_value = {
            "max_dollars": 16.0,
            "total_cost_usd": 0.1,
        }
        tracker.record_generation.return_value = OpenRouterGenerationCost(
            generation_id="gen_1",
            cost_usd=0.1,
            model="anthropic/claude-opus-5",
            provider="provider",
            native_prompt_tokens=100,
            native_completion_tokens=20,
            native_cached_tokens=0,
            native_reasoning_tokens=10,
        )
        trace = Mock()
        thread = ClaudeMessagesThread(
            client=client,
            trace=trace,
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[
                {
                    "type": "function",
                    "name": "run_command",
                    "description": "run",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                    "strict": True,
                }
            ],
            max_output_tokens=1_024,
            cost_tracker=tracker,
            provider="openrouter-anthropic",
            provider_preferences={
                "zdr": True,
                "data_collection": "deny",
                "require_parameters": True,
                "order": ["amazon-bedrock", "google-vertex"],
                "max_price": {
                    "prompt": "5",
                    "completion": "25",
                },
            },
        )

        response = thread.call(tool_choice={"type": "function", "name": "run_command"})

        request = client.create_prepared_turn.call_args.kwargs["request"]
        self.assertEqual(
            request["extra_body"]["provider"],
            {
                "zdr": True,
                "data_collection": "deny",
                "require_parameters": True,
                "order": ["amazon-bedrock", "google-vertex"],
                "max_price": {
                    "prompt": "5",
                    "completion": "25",
                },
            },
        )
        self.assertIn("independent and non-conflicting", request["system"])
        self.assertEqual(
            request["tools"],
            [
                {
                    "name": "run_command",
                    "description": "run",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                    },
                    "strict": True,
                }
            ],
        )
        self.assertEqual(
            request["tool_choice"],
            {
                "type": "tool",
                "name": "run_command",
                "disable_parallel_tool_use": False,
            },
        )
        self.assertEqual(
            json.loads(response["output"][0]["arguments"]),
            {"command": "true", "timeout": 10},
        )
        self.assertEqual(len(response["output"]), 2)
        self.assertIs(thread.messages[-1]["content"][0], tool_use)
        thread.add_tool_outputs([("toolu_1", "ok"), ("toolu_2", "binder result")])
        tool_results = thread.messages[-1]["content"]
        self.assertEqual(len(tool_results), 2)
        self.assertEqual(tool_results[1]["tool_use_id"], "toolu_2")
        self.assertEqual(tool_results[1]["content"], "binder result")
        tracker.ensure_request_allowed.assert_called_once()
        tracker.record_generation.assert_called_once_with("gen_1")
        trace.model_call.assert_called_once()
        traced_request = trace.model_call.call_args.args[0]
        self.assertEqual(traced_request, request)
        self.assertEqual(
            traced_request["extra_body"]["provider"],
            request["extra_body"]["provider"],
        )
        self.assertNotIn(
            "anthropic_request_preflight",
            [call.args[0] for call in trace.event.call_args_list],
        )

    def test_strips_unsupported_compaction_encrypted_content(self) -> None:
        compaction = SimpleNamespace(
            type="compaction",
            model_dump=lambda mode="json", exclude_none=True: {
                "type": "compaction",
                "content": "summary",
                "encrypted_content": None,
            },
        )
        tool_use = SimpleNamespace(
            type="tool_use",
            id="toolu_1",
            name="run_command",
            input={"command": "true"},
        )
        message = SimpleNamespace(
            id="gen_1",
            model="anthropic/claude-opus-5",
            provider="provider",
            content=[compaction, tool_use],
            usage=SimpleNamespace(),
            stop_reason="tool_use",
            _request_id="gen_1",
            model_dump=lambda mode="json": {"id": "gen_1", "content": []},
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="anthropic/claude-opus-5",
                effort="max",
            ),
            create_prepared_turn=AsyncMock(
                return_value=ClaudeTurn.from_message(message)
            ),
        )
        thread = self.thread(client)
        thread.cost_tracker.record_generation.return_value = OpenRouterGenerationCost(
            generation_id="gen_1",
            cost_usd=0.1,
            model="anthropic/claude-opus-5",
            provider="provider",
            native_prompt_tokens=100,
            native_completion_tokens=20,
            native_cached_tokens=0,
            native_reasoning_tokens=10,
        )

        thread.call()

        replay = thread.messages[-1]["content"]
        self.assertEqual(
            replay[0],
            {"type": "compaction", "content": "summary"},
        )
        self.assertIs(replay[1], tool_use)

    def test_openrouter_compaction_preserves_non_null_opaque_content(self) -> None:
        encrypted = "opaque-compaction-state"
        compaction = SimpleNamespace(
            type="compaction",
            model_dump=lambda mode="json", exclude_none=True: {
                "type": "compaction",
                "content": "summary",
                "encrypted_content": encrypted,
            },
        )
        message = SimpleNamespace(
            id="gen_1",
            model="anthropic/claude-opus-5",
            provider="provider",
            content=[compaction],
            usage=SimpleNamespace(),
            stop_reason="end_turn",
            _request_id="gen_1",
            model_dump=lambda mode="json": {"id": "gen_1", "content": []},
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="anthropic/claude-opus-5"),
            create_prepared_turn=AsyncMock(
                return_value=ClaudeTurn.from_message(message)
            ),
        )
        thread = self.thread(client)
        thread.cost_tracker.record_generation.return_value = OpenRouterGenerationCost(
            generation_id="gen_1",
            cost_usd=0.1,
            model="anthropic/claude-opus-5",
            provider="provider",
            native_prompt_tokens=100,
            native_completion_tokens=20,
            native_cached_tokens=0,
            native_reasoning_tokens=10,
        )

        thread.call()

        self.assertEqual(
            thread.messages[-1]["content"][0]["encrypted_content"],
            encrypted,
        )

    def test_direct_compaction_round_trips_official_sdk_blocks_unchanged(
        self,
    ) -> None:
        compaction = SimpleNamespace(
            type="compaction",
            content="summary",
            encrypted_content="opaque-state",
            model_dump=lambda mode="json", exclude_none=True: {
                "type": "compaction",
                "content": "summary",
                "encrypted_content": "opaque-state",
            },
        )
        message = SimpleNamespace(
            id="msg_1",
            model="claude-opus-5",
            provider=None,
            content=[compaction],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
            _request_id="req_1",
            model_dump=lambda mode="json": {"id": "msg_1", "content": []},
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5"),
            create_prepared_turn=AsyncMock(
                return_value=ClaudeTurn.from_message(message)
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=None,
                artifact_path=Path(tmp) / "cost.json",
            )
            thread = ClaudeMessagesThread(
                client=client,
                trace=Mock(),
                role="actor",
                system_prompt="system",
                user_prompt="user",
                tools=[],
                max_output_tokens=1_024,
                cost_tracker=tracker,
                provider="anthropic",
            )

            thread.call()

        self.assertIs(thread.messages[-1]["content"][0], compaction)

    def test_standalone_usage_sums_iterations_without_top_level_double_count(
        self,
    ) -> None:
        usage = ClaudeUsage()
        usage.add_message(
            {
                "usage": {
                    "input_tokens": 999,
                    "output_tokens": 999,
                    "cache_read_input_tokens": 999,
                    "cache_creation_input_tokens": 999,
                    "output_tokens_details": {"thinking_tokens": 999},
                    "iterations": [
                        {
                            "input_tokens": 10,
                            "output_tokens": 20,
                            "cache_read_input_tokens": 30,
                            "cache_creation_input_tokens": 40,
                            "output_tokens_details": {"thinking_tokens": 5},
                        },
                        {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "cache_read_input_tokens": 3,
                            "cache_creation_input_tokens": 4,
                            "output_tokens_details": {"thinking_tokens": 6},
                        },
                    ],
                }
            }
        )

        self.assertEqual(
            usage.to_dict(),
            {
                "requests": 1,
                "input_tokens": 11,
                "output_tokens": 22,
                "cache_creation_input_tokens": 44,
                "cache_read_input_tokens": 33,
                "thinking_tokens": 11,
            },
        )

    def test_direct_anthropic_uses_native_cost_and_no_openrouter_body(self) -> None:
        message = SimpleNamespace(
            id="msg_1",
            model="claude-opus-5",
            provider=None,
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="toolu_1",
                    name="run_command",
                    input={"command": "true"},
                )
            ],
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=20,
                cache_read_input_tokens=40,
                cache_creation_input_tokens=30,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=30,
                    ephemeral_1h_input_tokens=0,
                ),
                output_tokens_details=SimpleNamespace(thinking_tokens=10),
            ),
            stop_reason="tool_use",
            _request_id="req_1",
            model_dump=lambda mode="json": {"id": "msg_1", "content": []},
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="claude-opus-5",
                effort="max",
            ),
            create_prepared_turn=AsyncMock(
                return_value=ClaudeTurn.from_message(message)
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=1.0,
                artifact_path=Path(tmp) / "anthropic-cost.json",
            )
            trace = Mock()
            thread = ClaudeMessagesThread(
                client=client,
                trace=trace,
                role="actor",
                system_prompt="system",
                user_prompt="user",
                tools=[],
                max_output_tokens=1_024,
                cost_tracker=tracker,
                provider="anthropic",
            )

            built_requests: list[dict[str, object]] = []

            def build_once(*args: object, **kwargs: object) -> dict[str, object]:
                request = build_create_request(*args, **kwargs)
                built_requests.append(request)
                return request

            with patch(
                "cuawright.models.anthropic_messages.build_create_request",
                side_effect=build_once,
            ):
                response = thread.call()

        sent_request = client.create_prepared_turn.call_args.kwargs["request"]
        self.assertEqual(len(built_requests), 1)
        self.assertIs(sent_request, built_requests[0])
        self.assertEqual(
            set(client.create_prepared_turn.call_args.kwargs),
            {"request"},
        )
        traced_request = trace.model_call.call_args.args[0]
        self.assertEqual(
            traced_request,
            json.loads(json.dumps(sent_request, default=str)),
        )
        self.assertEqual(len(sent_request["messages"]), 1)
        self.assertEqual(len(thread.messages), 2)
        self.assertNotIn("session_id", traced_request)
        self.assertNotIn("provider", traced_request)
        self.assertNotIn("extra_body", traced_request)
        self.assertEqual(
            traced_request["thinking"],
            {"type": "adaptive", "display": "summarized"},
        )
        self.assertEqual(traced_request["output_config"], {"effort": "max"})
        self.assertIn("context_management", traced_request)
        self.assertEqual(traced_request["betas"], ["compact-2026-01-12"])
        self.assertIn("anthropic_cost", response)
        self.assertNotIn("openrouter_cost", response)
        trace.event.assert_any_call(
            "anthropic_cost",
            unittest.mock.ANY,
        )
        self.assertEqual(
            [call.args[0] for call in trace.event.call_args_list],
            ["anthropic_request_preflight", "anthropic_cost"],
        )

    def test_direct_pause_turn_resumes_and_accounts_each_response(self) -> None:
        def message(
            message_id: str,
            stop_reason: str,
            text: str,
        ) -> SimpleNamespace:
            content = [SimpleNamespace(type="text", text=text)]
            return SimpleNamespace(
                id=message_id,
                model="claude-opus-5",
                provider=None,
                content=content,
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                stop_reason=stop_reason,
                _request_id=f"req-{message_id}",
                model_dump=lambda mode="json": {
                    "id": message_id,
                    "content": [{"type": "text", "text": text}],
                },
            )

        paused = message("msg_pause", "pause_turn", "")
        finished = message("msg_done", "end_turn", "done")
        client = SimpleNamespace(
            config=ClaudeClientConfig(model="claude-opus-5"),
            create_prepared_turn=AsyncMock(
                side_effect=[
                    ClaudeTurn.from_message(paused),
                    ClaudeTurn.from_message(finished),
                ]
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=None,
                artifact_path=Path(tmp) / "cost.json",
            )
            trace = Mock()
            thread = ClaudeMessagesThread(
                client=client,
                trace=trace,
                role="actor",
                system_prompt="system",
                user_prompt="user",
                tools=[],
                max_output_tokens=1_024,
                cost_tracker=tracker,
                provider="anthropic",
            )

            paused_response = thread.call(tool_choice="none")
            response = thread.call(tool_choice="none")

            self.assertTrue(paused_response["continue_immediately"])
            self.assertEqual(paused_response["billed_response_count"], 1)
            self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(client.create_prepared_turn.await_count, 2)
        self.assertEqual(tracker.snapshot()["requests"], 2)
        self.assertEqual(trace.model_call.call_count, 2)
        self.assertEqual(response["billed_response_count"], 1)
        second_messages = client.create_prepared_turn.call_args_list[1].kwargs[
            "request"
        ]["messages"]
        self.assertEqual(second_messages[-1]["content"][0].type, "text")

    def test_direct_refusal_rolls_back_and_continues_with_safe_feedback(self) -> None:
        def message(
            message_id: str,
            stop_reason: str,
            content: list[SimpleNamespace],
        ) -> SimpleNamespace:
            return SimpleNamespace(
                id=message_id,
                model="claude-fable-5-1",
                provider=None,
                content=content,
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
                stop_reason=stop_reason,
                stop_details=(
                    SimpleNamespace(
                        category="cyber_policy",
                        explanation="The response triggered the cyber policy.",
                    )
                    if stop_reason == "refusal"
                    else None
                ),
                _request_id=f"req-{message_id}",
                model_dump=lambda mode="json": {
                    "id": message_id,
                    "content": [
                        {
                            "type": block.type,
                            "text": getattr(block, "text", None),
                            "id": getattr(block, "id", None),
                            "name": getattr(block, "name", None),
                            "input": getattr(block, "input", None),
                        }
                        for block in content
                    ],
                    "stop_reason": stop_reason,
                },
            )

        refused = message(
            "msg_refused",
            "refusal",
            [
                SimpleNamespace(
                    type="tool_use",
                    id="blocked-call",
                    name="run_command",
                    input={"command": "sudo blocked-command"},
                )
            ],
        )
        finished = message(
            "msg_done",
            "end_turn",
            [SimpleNamespace(type="text", text="continued safely")],
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="claude-fable-5-1",
                policy_feedback_max_retries=1,
            ),
            create_prepared_turn=AsyncMock(
                side_effect=[
                    ClaudeTurn.from_message(refused),
                    ClaudeTurn.from_message(finished),
                ]
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=None,
                artifact_path=Path(tmp) / "cost.json",
            )
            trace = Mock()
            thread = ClaudeMessagesThread(
                client=client,
                trace=trace,
                role="actor",
                system_prompt="system",
                user_prompt="user",
                tools=[],
                max_output_tokens=1_024,
                cost_tracker=tracker,
                provider="anthropic",
            )

            recovery = thread.call(tool_choice="none")
            response = thread.call(tool_choice="none")

        self.assertTrue(recovery["continue_immediately"])
        self.assertTrue(recovery["policy_feedback_retry"])
        self.assertEqual(recovery["billed_response_count"], 1)
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(client.create_prepared_turn.await_count, 2)
        self.assertEqual(tracker.snapshot()["requests"], 2)
        retry_messages = client.create_prepared_turn.call_args_list[1].kwargs[
            "request"
        ]["messages"]
        self.assertEqual(retry_messages[-1]["role"], "user")
        self.assertIn("compliance retry 1/1", retry_messages[-1]["content"])
        self.assertIn(
            "The response triggered the cyber policy",
            retry_messages[-1]["content"],
        )
        self.assertFalse(
            any(
                message.get("role") == "assistant"
                and any(
                    getattr(block, "id", None) == "blocked-call"
                    for block in message.get("content", [])
                )
                for message in retry_messages
            )
        )
        trace.event.assert_any_call(
            "anthropic_policy_feedback_retry",
            {
                "role": "actor",
                "attempt": 1,
                "max_attempts": 1,
                "source": "refusal",
                "category": "cyber_policy",
                "billed_response_count": 1,
                "feedback": unittest.mock.ANY,
            },
        )

    def test_direct_policy_error_retries_once_then_raises(self) -> None:
        response = httpx2.Response(
            200,
            request=httpx2.Request("POST", "https://anthropic.example/messages"),
        )

        def policy_error() -> APIStatusError:
            return APIStatusError(
                "Output blocked by content filtering policy",
                response=response,
                body={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Output blocked by content filtering policy",
                    }
                },
            )

        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="claude-fable-5-1",
                policy_feedback_max_retries=1,
            ),
            create_prepared_turn=AsyncMock(
                side_effect=[policy_error(), policy_error()]
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracker = AnthropicCostTracker(
                max_dollars=None,
                artifact_path=Path(tmp) / "cost.json",
            )
            trace = Mock()
            thread = ClaudeMessagesThread(
                client=client,
                trace=trace,
                role="actor",
                system_prompt="system",
                user_prompt="user",
                tools=[],
                max_output_tokens=1_024,
                cost_tracker=tracker,
                provider="anthropic",
            )

            recovery = thread.call(tool_choice="none")
            with self.assertRaises(APIStatusError):
                thread.call(tool_choice="none")

        self.assertTrue(recovery["continue_immediately"])
        self.assertEqual(recovery["billed_response_count"], 1)
        self.assertEqual(tracker.snapshot()["requests"], 0)
        self.assertEqual(client.create_prepared_turn.await_count, 2)
        retry_messages = client.create_prepared_turn.call_args_list[1].kwargs[
            "request"
        ]["messages"]
        feedback_messages = [
            message
            for message in retry_messages
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and "policy-compliant" in message["content"]
        ]
        self.assertEqual(len(feedback_messages), 1)
        self.assertIn(
            "Output blocked by content filtering policy",
            feedback_messages[0]["content"],
        )
        self.assertEqual(trace.model_call.call_count, 2)
        self.assertEqual(
            trace.model_call.call_args_list[0].args[1]["stop_reason"],
            "policy_error",
        )
        self.assertEqual(
            trace.model_call.call_args_list[0].args[1]["policy_category"],
            "content_filter",
        )
        self.assertEqual(
            [
                call.args[0]
                for call in trace.event.call_args_list
                if call.args[0] == "anthropic_policy_feedback_retry"
            ],
            ["anthropic_policy_feedback_retry"],
        )

    def test_direct_non_policy_bad_request_is_not_recovered(self) -> None:
        response = httpx2.Response(
            400,
            request=httpx2.Request("POST", "https://anthropic.example/messages"),
        )
        error = APIStatusError(
            "invalid tool schema",
            response=response,
            body={
                "error": {
                    "type": "invalid_request_error",
                    "message": "tools.0.input_schema is invalid",
                }
            },
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="claude-fable-5-1",
                policy_feedback_max_retries=1,
            ),
            create_prepared_turn=AsyncMock(side_effect=error),
        )
        thread = self.thread(client)

        with self.assertRaises(APIStatusError):
            thread.call(tool_choice="none")

        self.assertEqual(
            thread.messages,
            [{"role": "user", "content": "user"}],
        )
        self.assertFalse(
            any(
                call.args[0] == "anthropic_policy_feedback_retry"
                for call in thread.trace.event.call_args_list
            )
        )

    def test_malformed_tool_json_is_accounted_and_retried(self) -> None:
        message = SimpleNamespace(
            id="gen_good",
            model="anthropic/claude-opus-5",
            provider="provider",
            content=[SimpleNamespace(type="text", text="done")],
            usage=SimpleNamespace(),
            model_dump=lambda mode="json": {
                "id": "gen_good",
                "content": [{"type": "text", "text": "done"}],
            },
            stop_reason="end_turn",
            _request_id="gen_good",
        )
        client = SimpleNamespace(
            config=ClaudeClientConfig(
                model="anthropic/claude-opus-5",
                effort="max",
            ),
            create_prepared_turn=AsyncMock(
                side_effect=[
                    ClaudeStreamParseError(
                        "malformed tool input",
                        request_id="gen_bad",
                    ),
                    ClaudeTurn.from_message(message),
                ]
            ),
        )
        tracker = Mock()
        tracker.snapshot.return_value = {
            "max_dollars": 10.0,
            "total_cost_usd": 0.2,
        }
        tracker.record_generation.side_effect = [
            OpenRouterGenerationCost(
                generation_id="gen_bad",
                cost_usd=0.1,
                model="anthropic/claude-opus-5",
                provider="provider",
                native_prompt_tokens=100,
                native_completion_tokens=20,
                native_cached_tokens=0,
                native_reasoning_tokens=10,
            ),
            OpenRouterGenerationCost(
                generation_id="gen_good",
                cost_usd=0.1,
                model="anthropic/claude-opus-5",
                provider="provider",
                native_prompt_tokens=100,
                native_completion_tokens=20,
                native_cached_tokens=0,
                native_reasoning_tokens=10,
            ),
        ]
        trace = Mock()
        thread = ClaudeMessagesThread(
            client=client,
            trace=trace,
            role="critic",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=1_024,
            cost_tracker=tracker,
            provider="openrouter-anthropic",
        )

        recovery = thread.call(tool_choice="none")
        response = thread.call(tool_choice="none")

        self.assertTrue(recovery["continue_immediately"])
        self.assertTrue(recovery["synthetic_recovery"])
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(client.create_prepared_turn.await_count, 2)
        self.assertEqual(
            [call.args[0] for call in tracker.record_generation.call_args_list],
            ["gen_bad", "gen_good"],
        )
        first_request = client.create_prepared_turn.call_args_list[0].kwargs["request"]
        retry_request = client.create_prepared_turn.call_args_list[1].kwargs["request"]
        retry_messages = retry_request["messages"]
        self.assertTrue(
            any(
                message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and "compact valid JSON" in message["content"]
                for message in retry_messages
            )
        )
        self.assertNotEqual(first_request["messages"], retry_messages)
        self.assertEqual(trace.model_call.call_args.args[0], retry_request)
        trace.event.assert_any_call(
            "malformed_tool_response",
            unittest.mock.ANY,
        )


if __name__ == "__main__":
    unittest.main()

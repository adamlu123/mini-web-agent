from __future__ import annotations

import asyncio
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from cuawright.models.anthropic_client import (
    ClaudeClientConfig,
    ClaudeMessagesClient,
    ClaudeRunResult,
    ClaudeStopError,
    ClaudeTurn,
    ClaudeUsage,
    build_create_request,
    dump_private_trace,
    read_api_key,
    serialized_request_bytes,
)


class FakeStream:
    def __init__(self, message, request_id="req_stream") -> None:
        self.message = message
        self.request_id = request_id

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get_final_message(self):
        return self.message


def usage(
    *,
    input_tokens=10,
    output_tokens=5,
    cache_creation=0,
    cache_read=0,
    thinking_tokens=0,
):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        output_tokens_details=SimpleNamespace(thinking_tokens=thinking_tokens),
    )


class ClaudeClientTests(unittest.TestCase):
    def test_key_file_requires_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            path.write_text("secret\n")
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                read_api_key(path)
            path.chmod(0o600)
            self.assertEqual(read_api_key(path), "secret")

    def test_usage_includes_cache_and_thinking(self) -> None:
        total = ClaudeUsage()
        total.add_message(
            SimpleNamespace(
                usage=usage(
                    input_tokens=100,
                    output_tokens=40,
                    cache_creation=20,
                    cache_read=60,
                    thinking_tokens=25,
                )
            )
        )
        self.assertEqual(
            total.to_dict(),
            {
                "requests": 1,
                "input_tokens": 100,
                "output_tokens": 40,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 60,
                "thinking_tokens": 25,
            },
        )

    def test_create_turn_uses_first_class_opus_settings(self) -> None:
        thinking = SimpleNamespace(
            type="thinking",
            thinking="summary",
            signature="encrypted-signature",
        )
        text = SimpleNamespace(type="text", text="done")
        message = SimpleNamespace(
            content=[thinking, text],
            stop_reason="end_turn",
            _request_id="req_1",
            usage=usage(),
        )
        messages_api = Mock()
        messages_api.stream.return_value = FakeStream(message)
        client = SimpleNamespace(
            messages=Mock(),
            beta=SimpleNamespace(messages=messages_api),
            close=AsyncMock(),
        )
        wrapper = ClaudeMessagesClient(
            api_key="unused",
            config=ClaudeClientConfig(
                effort="xhigh",
                max_tokens=64_000,
            ),
            client=client,
        )

        turn = asyncio.run(
            wrapper.create_turn(
                messages=[{"role": "user", "content": "test"}],
                system="system",
                tools=[{"name": "tool", "input_schema": {"type": "object"}}],
            )
        )

        request = messages_api.stream.call_args.kwargs
        self.assertEqual(request["model"], "claude-opus-5")
        self.assertEqual(request["max_tokens"], 64_000)
        self.assertEqual(
            request["thinking"],
            {"type": "adaptive", "display": "summarized"},
        )
        self.assertEqual(request["output_config"], {"effort": "xhigh"})
        self.assertEqual(
            request["cache_control"],
            {"type": "ephemeral"},
        )
        self.assertEqual(request["betas"], ["compact-2026-01-12"])
        self.assertEqual(
            request["context_management"],
            {
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {
                            "type": "input_tokens",
                            "value": 150_000,
                        },
                    }
                ]
            },
        )
        self.assertEqual(turn.thinking_summaries, ["summary"])
        self.assertEqual(turn.request_id, "req_stream")

    def test_preflighted_request_is_the_exact_stream_request(self) -> None:
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            usage=usage(),
        )
        messages_api = Mock()
        messages_api.stream.return_value = FakeStream(message)
        wrapper = ClaudeMessagesClient(
            api_key="unused",
            client=SimpleNamespace(
                messages=Mock(),
                beta=SimpleNamespace(messages=messages_api),
                close=AsyncMock(),
            ),
        )
        prepared = {
            "model": "claude-opus-5",
            "max_tokens": 321,
            "messages": [{"role": "user", "content": "preflighted"}],
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "max"},
            "cache_control": {"type": "ephemeral"},
            "betas": ["compact-2026-01-12"],
            "context_management": wrapper.config.context_management,
        }

        asyncio.run(wrapper.create_prepared_turn(request=prepared))

        self.assertEqual(messages_api.stream.call_args.kwargs, prepared)

    def test_serialized_size_uses_conservative_json_escaping(self) -> None:
        request = {"messages": [{"role": "user", "content": "screen 🖥"}]}

        expected = len(
            json.dumps(request, ensure_ascii=True, default=str).encode("utf-8")
        )

        self.assertEqual(serialized_request_bytes(request), expected)

    def test_default_profile_uses_max_effort_and_provider_compaction(self) -> None:
        config = ClaudeClientConfig()
        self.assertEqual(config.effort, "max")
        self.assertTrue(config.server_compaction)
        self.assertEqual(config.cache_control, {"type": "ephemeral"})
        self.assertTrue(config.disable_parallel_tool_use)
        self.assertEqual(config.policy_feedback_max_retries, 0)
        with self.assertRaisesRegex(ValueError, "at least 50,000"):
            ClaudeClientConfig(compaction_trigger_tokens=49_999)
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            ClaudeClientConfig(policy_feedback_max_retries=True)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            ClaudeClientConfig(policy_feedback_max_retries=-1)

    def test_foundry_profile_uses_binding_control_without_compaction(self) -> None:
        config = ClaudeClientConfig(
            model="claude-fable-5-1",
            server_compaction=False,
            thinking_prefix_mismatch_behavior="drop_block",
        )

        request = build_create_request(
            config,
            messages=[{"role": "user", "content": "test"}],
        )

        self.assertEqual(
            request["thinking"]["block_binding"],
            {"prefix_mismatch_behavior": "drop_block"},
        )
        self.assertEqual(
            request["betas"],
            ["thinking-binding-controls-2026-08-01"],
        )
        self.assertNotIn("context_management", request)

    def test_tool_loop_preserves_thinking_block_and_signature(self) -> None:
        thinking = SimpleNamespace(
            type="thinking",
            thinking="Use the tool.",
            signature="encrypted-signature",
        )
        tool_use = SimpleNamespace(
            type="tool_use",
            id="toolu_1",
            name="add_numbers",
            input={"a": 2, "b": 3},
        )
        first = SimpleNamespace(
            model="claude-opus-5",
            content=[thinking, tool_use],
            stop_reason="tool_use",
            _request_id="req_1",
            usage=usage(input_tokens=20, output_tokens=10),
        )
        second = SimpleNamespace(
            model="claude-opus-5",
            content=[SimpleNamespace(type="text", text="5")],
            stop_reason="end_turn",
            _request_id="req_2",
            usage=usage(input_tokens=30, output_tokens=5),
        )
        wrapper = ClaudeMessagesClient(
            api_key="unused",
            client=SimpleNamespace(messages=Mock(), close=AsyncMock()),
        )
        wrapper.create_turn = AsyncMock(
            side_effect=[
                __import__(
                    "cuawright.models.anthropic_client",
                    fromlist=["ClaudeTurn"],
                ).ClaudeTurn.from_message(first),
                __import__(
                    "cuawright.models.anthropic_client",
                    fromlist=["ClaudeTurn"],
                ).ClaudeTurn.from_message(second),
            ]
        )

        result = asyncio.run(
            wrapper.run_tool_loop(
                user_prompt="add",
                tools=[],
                tool_handlers={
                    "add_numbers": lambda values: str(values["a"] + values["b"])
                },
            )
        )

        assistant = result.messages[1]
        self.assertIs(assistant["content"][0], thinking)
        self.assertEqual(
            assistant["content"][0].signature,
            "encrypted-signature",
        )
        tool_result_message = result.messages[2]
        self.assertEqual(
            tool_result_message["content"],
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "5",
                    "is_error": False,
                }
            ],
        )
        self.assertEqual(result.final_turn.text, "5")
        self.assertEqual(result.usage.requests, 2)

    def test_max_tokens_is_not_treated_as_success(self) -> None:
        message = SimpleNamespace(
            model="claude-opus-5",
            content=[],
            stop_reason="max_tokens",
            _request_id="req_1",
            usage=usage(),
        )
        wrapper = ClaudeMessagesClient(
            api_key="unused",
            client=SimpleNamespace(messages=Mock(), close=AsyncMock()),
        )
        wrapper.create_turn = AsyncMock(
            return_value=__import__(
                "cuawright.models.anthropic_client",
                fromlist=["ClaudeTurn"],
            ).ClaudeTurn.from_message(message)
        )
        with self.assertRaisesRegex(ClaudeStopError, "max_tokens"):
            asyncio.run(
                wrapper.run_tool_loop(
                    user_prompt="test",
                    tools=[],
                    tool_handlers={},
                )
            )

    def test_private_trace_is_created_with_mode_0600(self) -> None:
        message = SimpleNamespace(
            model="claude-opus-5",
            content=[SimpleNamespace(type="text", text="done")],
            stop_reason="end_turn",
            _request_id="req_1",
            usage=usage(),
        )
        turn = ClaudeTurn.from_message(message)
        result = ClaudeRunResult(
            final_turn=turn,
            turns=[turn],
            messages=[
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": message.content},
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            dump_private_trace(result, path)
            self.assertEqual(
                stat.S_IMODE(path.stat().st_mode),
                0o600,
            )


if __name__ == "__main__":
    unittest.main()

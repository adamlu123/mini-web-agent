from __future__ import annotations

import ast
import base64
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from anthropic import transform_schema

from cuawright.models.base import DEFAULT_MAX_OUTPUT_TOKENS, RUN_COMMAND_DESCRIPTION
from cuawright.models.anthropic_messages import (
    AnthropicMessagesAdapter,
    anthropic_messages_tools,
)
from cuawright.models.openai_responses import (
    OpenAIResponsesAdapter,
    ResponsesThread,
    create_responses_client,
    openai_responses_tools,
)
from cuawright.agents.default import AgentLoop
from cuawright.environments.base import (
    CommandResult,
    FileAttachment,
    ImageAttachment,
    MODEL_COMMAND_OUTPUT_LIMIT,
)
from cuawright.budget import SharedStepBudget
from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA, ToolCall


class FakeResponsesThread:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.tools: list[dict[str, object]] | None = None
        self.outputs: list[tuple[str, object]] = []
        self.images: list[tuple[str, str, bytes]] = []
        self.user_texts: list[str] = []

    def call(
        self,
        *,
        tools: list[dict[str, object]] | None = None,
        **_: object,
    ) -> dict[str, object]:
        self.tools = tools
        return self.response

    def add_tool_output(self, call_id: str, output: object) -> None:
        self.outputs.append((call_id, output))

    def add_image(self, path: str, mime: str, data: bytes) -> None:
        self.images.append((path, mime, data))

    def add_user_text(self, text: str) -> None:
        self.user_texts.append(text)


class FakeClaudeThread:
    provider = "anthropic"

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.tools: list[dict[str, object]] | None = None
        self.outputs: list[list[tuple[str, object]]] = []
        self.prepared: list[tuple[str, bytes]] = []
        self.user_texts: list[str] = []

    def call_provider_tools(
        self,
        *,
        tools: list[dict[str, object]],
        **_: object,
    ) -> dict[str, object]:
        self.tools = tools
        return self.response

    def add_tool_outputs(self, outputs: list[tuple[str, object]]) -> None:
        self.outputs.append(outputs)

    def prepare_image(self, mime: str, data: bytes) -> SimpleNamespace:
        self.prepared.append((mime, data))
        return SimpleNamespace(
            media_type="image/jpeg",
            data=b"compressed-image",
            dimension_metadata={
                "source_dimensions": {"width": 2_400, "height": 1_200},
                "displayed_dimensions": {"width": 1_920, "height": 960},
            },
        )

    def add_user_text(self, text: str) -> None:
        self.user_texts.append(text)


class ProviderToolSchemaTests(unittest.TestCase):
    def test_default_model_output_ceiling_remains_32768(self) -> None:
        self.assertEqual(DEFAULT_MAX_OUTPUT_TOKENS, 32_768)

    def test_translates_frozen_schema_without_mutating_it(self) -> None:
        original = copy.deepcopy(RUN_COMMAND_TOOL_SCHEMA)

        openai = openai_responses_tools((RUN_COMMAND_TOOL_SCHEMA,))
        direct = anthropic_messages_tools(
            (RUN_COMMAND_TOOL_SCHEMA,),
            direct=True,
        )
        openrouter = anthropic_messages_tools(
            (RUN_COMMAND_TOOL_SCHEMA,),
            direct=False,
        )

        self.assertEqual(RUN_COMMAND_TOOL_SCHEMA, original)
        self.assertEqual(
            openai,
            [
                {
                    "type": "function",
                    "name": "run_command",
                    "description": RUN_COMMAND_DESCRIPTION,
                    "parameters": original["parameters"],
                    "strict": True,
                }
            ],
        )
        self.assertEqual(
            direct,
            [
                {
                    "name": "run_command",
                    "description": RUN_COMMAND_DESCRIPTION,
                    "input_schema": transform_schema(original["parameters"]),
                    "strict": True,
                }
            ],
        )
        self.assertEqual(
            openrouter,
            [
                {
                    "name": "run_command",
                    "description": RUN_COMMAND_DESCRIPTION,
                    "input_schema": original["parameters"],
                    "strict": True,
                }
            ],
        )
        self.assertNotIn("type", direct[0])

    def test_report_only_turn_translates_to_no_tools(self) -> None:
        self.assertEqual(openai_responses_tools(()), [])
        self.assertEqual(anthropic_messages_tools((), direct=True), [])
        self.assertEqual(anthropic_messages_tools((), direct=False), [])


class OpenAIResponsesAdapterTests(unittest.TestCase):
    def test_phyagi_client_configuration_disables_sdk_retries(self) -> None:
        with patch("cuawright.models.openai_responses.OpenAI") as client_type:
            client = create_responses_client(api_key="not-sent")

        self.assertIs(client, client_type.return_value)
        client_type.assert_called_once_with(
            api_key="not-sent",
            base_url="https://maif-gateway.azure-api.net/api",
            max_retries=0,
        )

    def test_normalizes_only_first_call_and_associates_image_result(self) -> None:
        thread = FakeResponsesThread(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Inspecting."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "run_command",
                        "arguments": '{"command":"first","timeout":12}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "call-2",
                        "name": "run_command",
                        "arguments": '{"command":"second"}',
                    },
                ]
            }
        )
        adapter = OpenAIResponsesAdapter(thread)

        turn = adapter.next_turn(tools=(RUN_COMMAND_TOOL_SCHEMA,))

        self.assertEqual(turn.text, "Inspecting.")
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertEqual(turn.tool_calls[0].call_id, "call-1")
        self.assertEqual(
            turn.tool_calls[0].arguments,
            {"command": "first", "timeout": 12},
        )
        self.assertEqual(
            thread.tools,
            openai_responses_tools((RUN_COMMAND_TOOL_SCHEMA,)),
        )

        adapter.add_command_result(
            turn.tool_calls[0],
            CommandResult(
                exit_code=7,
                stdout="standard output",
                stderr="standard error",
                model_fields={
                    "status": "error",
                    "infrastructure_error": False,
                    "timed_out": True,
                    "_harness": {"tool_step": 9},
                },
                attachments=(
                    ImageAttachment(
                        b"png-data",
                        "image/png",
                        "/workspace/frame.png",
                    ),
                    FileAttachment(
                        b"step-data",
                        source_path="/workspace/output.step",
                    ),
                ),
            ),
        )

        call_id, output = thread.outputs[0]
        self.assertEqual(call_id, "call-1")
        metadata = json.loads(output)
        self.assertEqual(metadata["returncode"], 7)
        self.assertEqual(metadata["output"], "standard output")
        self.assertEqual(metadata["error"], "standard error")
        self.assertEqual(metadata["status"], "error")
        self.assertFalse(metadata["infrastructure_error"])
        self.assertTrue(metadata["timed_out"])
        self.assertEqual(metadata["_harness"]["tool_step"], 9)
        self.assertFalse(metadata["_harness"]["inspected"])
        self.assertEqual(
            [item["source_path"] for item in metadata["attachments"]],
            ["/workspace/frame.png", "/workspace/output.step"],
        )
        self.assertEqual(
            thread.images,
            [("/workspace/frame.png", "image/png", b"png-data")],
        )
        self.assertNotIn(
            "c3RlcC1kYXRh",
            json.dumps(output),
        )

    def test_ordinary_result_is_truncated_after_aggregate_serialization(self) -> None:
        thread = FakeResponsesThread({"output": []})
        adapter = OpenAIResponsesAdapter(thread)
        call = ToolCall("call-1", "run_command", {"command": "large", "timeout": 30})

        adapter.add_command_result(
            call,
            CommandResult(
                1,
                stdout="O" * 9_000,
                stderr="E" * 9_000,
            ),
        )

        output = thread.outputs[0][1]
        self.assertLessEqual(len(output), MODEL_COMMAND_OUTPUT_LIMIT)
        self.assertIn("OUTPUT TRUNCATED", output)
        self.assertIn("E" * 100, output)
        self.assertIn("O" * 100, output)

    def test_direct_specialist_model_output_bypasses_ordinary_truncation(self) -> None:
        thread = FakeResponsesThread({"output": []})
        adapter = OpenAIResponsesAdapter(thread)
        call = ToolCall("call-1", "run_command", {"command": "binder", "timeout": 30})
        direct_report = json.dumps(
            {
                "status": "complete",
                "report": "R" * (MODEL_COMMAND_OUTPUT_LIMIT + 1_000),
                "unresolved": [],
            }
        )

        adapter.add_command_result(
            call,
            CommandResult(
                0,
                stdout=direct_report,
                model_output=direct_report,
            ),
        )

        self.assertEqual(thread.outputs[0][1], direct_report)
        self.assertNotIn("OUTPUT TRUNCATED", thread.outputs[0][1])

    def test_report_only_turn_sends_no_tools(self) -> None:
        thread = FakeResponsesThread(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "{}"}],
                    }
                ]
            }
        )

        turn = OpenAIResponsesAdapter(thread).next_turn(tools=())

        self.assertEqual(turn.text, "{}")
        self.assertEqual(thread.tools, [])

    def test_phyagi_request_shape_and_reasoning_continuity_are_unchanged(
        self,
    ) -> None:
        response = {
            "id": "response-1",
            "output": [
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [{"type": "summary_text", "text": "summary"}],
                    "encrypted_content": "opaque-reasoning",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "run_command",
                    "arguments": '{"command":"true"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "run_command",
                    "arguments": '{"command":"false"}',
                },
            ],
        }
        create = Mock(
            return_value=SimpleNamespace(
                model_dump=lambda mode="json": copy.deepcopy(response)
            )
        )
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="stable system",
            user_prompt="dynamic user",
            tools=[],
            max_output_tokens=64_000,
            base_url="https://maif-gateway.azure-api.net/api",
        )
        thread.session_id = "stable-session"

        turn = OpenAIResponsesAdapter(thread).next_turn(
            tools=(RUN_COMMAND_TOOL_SCHEMA,)
        )

        request = create.call_args.kwargs
        self.assertEqual(thread.base_url, "https://maif-gateway.azure-api.net/api")
        self.assertEqual(request["model"], "gpt-5.5")
        self.assertEqual(
            request["reasoning"],
            {"effort": "xhigh", "summary": "auto"},
        )
        self.assertEqual(
            request["extra_body"],
            {
                "tier": "base",
                "session_id": "stable-session",
                "strict_session": True,
                "infrastructure": "aiservices",
            },
        )
        self.assertEqual(request["include"], ["reasoning.encrypted_content"])
        self.assertFalse(request["store"])
        self.assertFalse(request["parallel_tool_calls"])
        self.assertEqual(request["max_output_tokens"], 64_000)
        self.assertEqual(
            request["tools"],
            openai_responses_tools((RUN_COMMAND_TOOL_SCHEMA,)),
        )
        self.assertEqual([call.call_id for call in turn.tool_calls], ["call-1"])
        self.assertIn(
            {
                "type": "reasoning",
                "id": "reasoning-1",
                "summary": [{"type": "summary_text", "text": "summary"}],
                "encrypted_content": "opaque-reasoning",
            },
            thread.context,
        )
        trace.model_call.assert_called_once()


class AnthropicMessagesAdapterTests(unittest.TestCase):
    def test_report_only_turn_sends_no_tools(self) -> None:
        thread = FakeClaudeThread(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "{}"}],
                    }
                ]
            }
        )

        turn = AnthropicMessagesAdapter(thread).next_turn(tools=())

        self.assertEqual(turn.text, "{}")
        self.assertEqual(thread.tools, [])

    def test_preserves_multiple_calls_and_batches_multimodal_results(self) -> None:
        thread = FakeClaudeThread(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Working."}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu-1",
                        "name": "run_command",
                        "arguments": '{"command":"first","timeout":30}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu-2",
                        "name": "run_command",
                        "arguments": '{"command":"second","timeout":30}',
                    },
                ]
            }
        )
        adapter = AnthropicMessagesAdapter(thread)

        turn = adapter.next_turn(tools=(RUN_COMMAND_TOOL_SCHEMA,))

        self.assertEqual(turn.text, "Working.")
        self.assertEqual(
            [call.call_id for call in turn.tool_calls],
            ["toolu-1", "toolu-2"],
        )
        self.assertEqual(
            thread.tools,
            anthropic_messages_tools(
                (RUN_COMMAND_TOOL_SCHEMA,),
                direct=True,
            ),
        )

        adapter.add_command_result(
            turn.tool_calls[0],
            CommandResult(0, stdout="first output"),
        )
        self.assertEqual(thread.outputs, [])
        adapter.add_command_result(
            turn.tool_calls[1],
            CommandResult(
                1,
                stdout="second output",
                stderr="failed",
                attachments=(
                    ImageAttachment(
                        b"source-image",
                        "image/png",
                        "/workspace/frame.png",
                    ),
                    FileAttachment(
                        b"private-file-bytes",
                        source_path="/workspace/report.bin",
                    ),
                ),
            ),
        )

        self.assertEqual(thread.prepared, [("image/png", b"source-image")])
        self.assertEqual(len(thread.outputs), 1)
        batch = thread.outputs[0]
        self.assertEqual([call_id for call_id, _ in batch], ["toolu-1", "toolu-2"])
        first = json.loads(batch[0][1])
        self.assertEqual(first["output"], "first output")
        self.assertEqual(first["returncode"], 0)
        self.assertEqual(first["status"], "success")
        second_content = batch[1][1]
        second = json.loads(second_content[0]["text"])
        self.assertEqual(second["returncode"], 1)
        self.assertEqual(second["error"], "failed")
        self.assertEqual(
            second["attachments"][0]["image_geometry"]["displayed_dimensions"],
            {"width": 1_920, "height": 960},
        )
        self.assertEqual(second_content[1]["type"], "image")
        self.assertEqual(
            second_content[1]["source"]["data"],
            base64.b64encode(b"compressed-image").decode("ascii"),
        )
        self.assertNotIn(
            base64.b64encode(b"private-file-bytes").decode("ascii"),
            json.dumps(second_content),
        )

    def test_completion_mid_batch_drains_all_pending_claude_calls(self) -> None:
        thread = FakeClaudeThread(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "toolu-1",
                        "name": "run_command",
                        "arguments": '{"command":"first","timeout":30}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu-2",
                        "name": "run_command",
                        "arguments": '{"command":"second","timeout":30}',
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu-3",
                        "name": "run_command",
                        "arguments": '{"command":"third","timeout":30}',
                    },
                ]
            }
        )

        class Plugin:
            def __init__(self) -> None:
                self.commands = []

            def run_command(self, command, timeout=None):
                self.commands.append(command)
                return CommandResult(0, completed=command == "second")

        adapter = AnthropicMessagesAdapter(thread)
        plugin = Plugin()
        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=SharedStepBudget(2),
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(plugin.commands, ["first", "second"])
        self.assertEqual(len(thread.outputs), 1)
        self.assertEqual(
            [call_id for call_id, _ in thread.outputs[0]],
            ["toolu-1", "toolu-2", "toolu-3"],
        )
        self.assertFalse(adapter._pending_calls)

    def test_pause_continuation_preserves_provider_response_accounting(self) -> None:
        thread = FakeClaudeThread(
            {
                "output": [],
                "billed_response_count": 1,
                "continue_immediately": True,
            }
        )

        turn = AnthropicMessagesAdapter(thread).next_turn(
            tools=(RUN_COMMAND_TOOL_SCHEMA,)
        )

        self.assertTrue(turn.continue_immediately)
        self.assertEqual(turn.billed_responses, 1)

    def test_malformed_continuation_preserves_provider_response_accounting(
        self,
    ) -> None:
        thread = FakeClaudeThread(
            {
                "output": [],
                "stop_reason": "malformed_tool_arguments",
                "billed_response_count": 1,
                "continue_immediately": True,
                "synthetic_recovery": True,
            }
        )

        turn = AnthropicMessagesAdapter(thread).next_turn(
            tools=(RUN_COMMAND_TOOL_SCHEMA,)
        )

        self.assertTrue(turn.continue_immediately)
        self.assertEqual(turn.billed_responses, 1)


class AdapterRecoveryIntegrationTests(unittest.TestCase):
    def test_schema_invalid_openai_arguments_recover_through_agent_loop(self) -> None:
        thread = FakeResponsesThread(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "bad",
                        "name": "run_command",
                        "arguments": '{"timeout":5}',
                    }
                ]
            }
        )

        class Plugin:
            def run_command(self, command, timeout=None):
                raise AssertionError("schema-invalid command must not execute")

        adapter = OpenAIResponsesAdapter(thread)
        result = AgentLoop(
            adapter=adapter,
            plugin=Plugin(),
            budget=SharedStepBudget(1),
        ).run()

        self.assertFalse(result.completed)
        self.assertEqual(len(thread.outputs), 1)
        self.assertEqual(thread.outputs[0][0], "bad")
        self.assertIn("command must be a string", thread.outputs[0][1])


class AdapterImportBoundaryTests(unittest.TestCase):
    def test_adapter_modules_do_not_import_benchmark_implementations(self) -> None:
        adapters = Path(__file__).parents[1] / "src" / "cuawright" / "models"
        forbidden_roots = {
            "cuawright.environments.benchcad",
            "cuawright.environments.cadgenbench",
            "desktop_env",
            "task_loader",
        }

        for path in adapters.glob("*.py"):
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                imports: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
                self.assertFalse(
                    {name.split(".", 1)[0] for name in imports} & forbidden_roots,
                    imports,
                )


if __name__ == "__main__":
    unittest.main()

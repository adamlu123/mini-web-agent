from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cuawright.agents.orchestration import TerminalAgentRuntime
from cuawright.artifacts import NullRunStore
from cuawright.config.prompt_registry import select_benchcad_prompt_bundle
from cuawright.config.prompts import (
    OSWORLD_DEFAULT_PROMPT_BUNDLE,
    actor_user_prompt,
    budget_status_user_prompt,
    compose_system_prompt,
    json_sha256,
)
from cuawright.environments.benchcad.direct import _build_direct_request
from cuawright.environments.benchcad.terminal_prompts import (
    BENCHCAD_FULL_PROMPT_BUNDLE,
)
from cuawright.environments.cadgenbench.terminal_prompts import (
    CADGENBENCH_DEFAULT_PROMPT_BUNDLE,
)
from cuawright.models.base import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    RUN_COMMAND_TOOL_SCHEMA,
)
from cuawright.models.openai_responses import (
    OpenAIResponsesAdapter,
    ResponsesThread,
)


GOLDEN_SYSTEM_SHA256 = {
    "osworld": "cedb9426a0c3cf10abc8a4b5d286787c01456b68361af33d4ade77966b52b4e5",
    "benchcad": "8f805d1528272d6be6de9f7344ad6d004528aaf9191089f201ab968e7e07f212",
    "cadgenbench": "cdd447447b14c0fcb1dfcadc1334bb9866d745e3b75ce27628870c73276f2b44",
}


class GoldenRawRequestTests(unittest.TestCase):
    def test_benchcad_direct_request_remains_exact_passthrough(self) -> None:
        bundle = select_benchcad_prompt_bundle(
            condition="direct",
            direct_system_prompt="direct system\n",
            direct_user_prompt="direct user\n",
        )
        request = _build_direct_request(
            prompt_bundle=bundle,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            image_data="aW1hZ2U=",
            session_id="benchcad-direct-golden",
        )
        legacy_request = {
            "model": "gpt-5.5",
            "instructions": "direct system\n",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "direct user\n"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,aW1hZ2U=",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "reasoning": {"effort": "xhigh"},
            "max_output_tokens": 32_000,
            "store": False,
            "extra_body": {
                "tier": "base",
                "session_id": "benchcad-direct-golden",
                "strict_session": True,
            },
        }
        self.assertEqual(request, legacy_request)
        self.assertEqual(
            json_sha256(request),
            "937e68cc33c6ab26535d7d61aee569d0973929b29d767db0d88268fedbfbdb42",
        )
        self.assertNotIn("tools", request)
        self.assertNotIn("parallel_tool_calls", request)

    def test_benchmark_requests_preserve_message_order_and_provider_contract(
        self,
    ) -> None:
        bundles = {
            "osworld": OSWORLD_DEFAULT_PROMPT_BUNDLE,
            "benchcad": BENCHCAD_FULL_PROMPT_BUNDLE,
            "cadgenbench": CADGENBENCH_DEFAULT_PROMPT_BUNDLE,
        }
        for benchmark, bundle in bundles.items():
            with self.subTest(benchmark=benchmark):
                create = Mock(
                    return_value=SimpleNamespace(
                        model_dump=lambda mode="json": {
                            "status": "completed",
                            "output": [],
                        }
                    )
                )
                system = compose_system_prompt("actor", bundle)
                task = actor_user_prompt("Create the requested benchmark artifact.")
                budget = budget_status_user_prompt(
                    "Global model-call budget: 0/300 used; 300 remain.",
                    midpoint_due=False,
                    verifier_called=False,
                )
                thread = ResponsesThread(
                    client=SimpleNamespace(responses=SimpleNamespace(create=create)),
                    trace=Mock(),
                    model="gpt-5.5",
                    reasoning_effort="xhigh",
                    role=f"{benchmark}-actor",
                    system_prompt=system,
                    user_prompt=task,
                    tools=[],
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                    base_url="https://maif-gateway.azure-api.net/api",
                    infrastructure="aiservices",
                )
                thread.add_user_text(budget)

                OpenAIResponsesAdapter(thread).next_turn(
                    tools=(RUN_COMMAND_TOOL_SCHEMA,)
                )

                request = create.call_args.kwargs
                self.assertEqual(
                    request["input"],
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": task},
                        {"role": "user", "content": budget},
                    ],
                )
                self.assertEqual(
                    hashlib.sha256(system.encode("utf-8")).hexdigest(),
                    GOLDEN_SYSTEM_SHA256[benchmark],
                )
                self.assertIn(
                    'python /opt/cuawright-tools/binder.py --request "TEXT" '
                    "--max-steps N",
                    system,
                )
                self.assertNotIn("REQUEST_PATH", system)
                self.assertNotIn("QUESTION_PATH", system)
                self.assertIn(
                    "same run_command result returns the report directly",
                    " ".join(system.split()),
                )
                self.assertIn(
                    "Normal submission requires a latest Verifier status",
                    " ".join(system.split()),
                )
                for legacy in (
                    "agent spawn",
                    "agent report",
                    "agent image show",
                    "agent user ask",
                    "agent status",
                    "agent submit",
                ):
                    self.assertNotIn(legacy, system)
                self.assertEqual(
                    [tool["name"] for tool in request["tools"]],
                    ["run_command"],
                )
                self.assertFalse(request["parallel_tool_calls"])
                self.assertEqual(
                    request["reasoning"],
                    {"effort": "xhigh", "summary": "auto"},
                )
                self.assertEqual(request["include"], ["reasoning.encrypted_content"])
                self.assertEqual(
                    request["max_output_tokens"],
                    DEFAULT_MAX_OUTPUT_TOKENS,
                )
                self.assertFalse(request["store"])
                self.assertEqual(request["extra_body"]["tier"], "base")
                self.assertTrue(request["extra_body"]["strict_session"])
                self.assertEqual(
                    request["extra_body"]["infrastructure"],
                    "aiservices",
                )
                self.assertTrue(
                    request["extra_body"]["session_id"].startswith(
                        f"osworld-cli-{benchmark}-actor-"
                    )
                )

    def test_osworld_benchcad_and_cadgenbench_runtime_raw_requests_are_golden(
        self,
    ) -> None:
        class CompletingPlugin:
            def __init__(self) -> None:
                self.files: dict[str, bytes] = {}

            def write_control_file_atomic(
                self,
                path: str,
                data: bytes,
                *,
                workspace: str,
            ) -> None:
                del workspace
                self.files[path] = data

            def read_control_file(
                self,
                path: str,
                *,
                workspace: str,
                max_bytes: int,
            ) -> bytes:
                del workspace
                return self.files[path][:max_bytes]

            def read_file(self, path: str, *, max_bytes: int) -> bytes:
                return self.files[path][:max_bytes]

            def run_command(self, command: str, timeout: int | None = None):
                del command, timeout
                from cuawright.environments.base import CommandResult

                return CommandResult(0, completed=True)

        bundles = {
            "osworld": OSWORLD_DEFAULT_PROMPT_BUNDLE,
            "benchcad": BENCHCAD_FULL_PROMPT_BUNDLE,
            "cadgenbench": CADGENBENCH_DEFAULT_PROMPT_BUNDLE,
        }
        for benchmark, bundle in bundles.items():
            with self.subTest(benchmark=benchmark):
                create = Mock(
                    return_value=SimpleNamespace(
                        model_dump=lambda mode="json": {
                            "status": "completed",
                            "output": [
                                {
                                    "type": "function_call",
                                    "call_id": "submit",
                                    "name": "run_command",
                                    "arguments": '{"command":"true","timeout":30}',
                                }
                            ],
                        }
                    )
                )
                requests = []

                def thread_factory(*, role, system_prompt, user_prompt):
                    thread = ResponsesThread(
                        client=SimpleNamespace(
                            responses=SimpleNamespace(create=create)
                        ),
                        trace=Mock(),
                        model="gpt-5.5",
                        reasoning_effort="xhigh",
                        role=role,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        tools=[],
                        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                        base_url="https://maif-gateway.azure-api.net/api",
                        infrastructure="aiservices",
                    )
                    requests.append(thread)
                    return thread

                runtime = TerminalAgentRuntime(
                    plugin=CompletingPlugin(),
                    thread_factory=thread_factory,
                    adapter_factory=OpenAIResponsesAdapter,
                    instruction="Create the requested benchmark artifact.",
                    task_id="fixture-001",
                    total_steps=3,
                    compact_every=0,
                    run_store=NullRunStore(),
                    prompt_bundle=bundle,
                    workspace=bundle.workspace,
                )
                runtime.run()

                request = create.call_args.kwargs
                self.assertEqual(
                    hashlib.sha256(
                        request["input"][0]["content"].encode("utf-8")
                    ).hexdigest(),
                    GOLDEN_SYSTEM_SHA256[benchmark],
                )
                self.assertIn(
                    "Original task instruction:\n"
                    "Create the requested benchmark artifact.",
                    request["input"][1]["content"],
                )
                self.assertIn("[Shared budget]", request["input"][2]["content"])
                self.assertEqual(
                    [tool["name"] for tool in request["tools"]],
                    ["run_command"],
                )
                self.assertEqual(requests[0].role, "actor")


if __name__ == "__main__":
    unittest.main()

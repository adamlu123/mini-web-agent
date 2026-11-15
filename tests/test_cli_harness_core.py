from __future__ import annotations

import ast
import hashlib
import unittest
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path

from cuawright.agents.default import AgentLoop, AgentLoopProtocolError
from cuawright.artifacts import ArtifactRef, NullRunStore
from cuawright.environments.base import (
    CommandResult,
    FileAttachment,
    ImageAttachment,
)
from cuawright.environments.control_clients import (
    CLIENT_FILENAMES,
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
    CLIENT_SOURCES,
)
from cuawright.budget import SharedStepBudget
from cuawright.models.base import (
    RUN_COMMAND_TOOL_SCHEMA,
    ModelTurn,
    ToolCall,
)
from cuawright.operations.scheduler import control_client_manifest

SUBMIT_COMMAND = "python /opt/cuawright-tools/submit.py"
VERIFIER_FINAL_COMMAND = "finish verifier"


class FakeAdapter:
    def __init__(self, turns: Sequence[ModelTurn]) -> None:
        self.turns = list(turns)
        self.tools_seen: list[Sequence[Mapping[str, object]]] = []
        self.command_results: list[tuple[ToolCall, CommandResult]] = []
        self.user_texts: list[str] = []

    def next_turn(
        self,
        *,
        tools: Sequence[Mapping[str, object]],
    ) -> ModelTurn:
        self.tools_seen.append(tools)
        if not self.turns:
            raise AssertionError("AgentLoop requested an unexpected model turn")
        return self.turns.pop(0)

    def add_command_result(
        self,
        tool_call: ToolCall,
        result: CommandResult,
    ) -> None:
        self.command_results.append((tool_call, result))

    def add_user_text(self, text: str) -> None:
        self.user_texts.append(text)


class FakePlugin:
    def __init__(self, results: Sequence[CommandResult]) -> None:
        self.results = list(results)
        self.commands: list[tuple[str, int | None]] = []

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        self.commands.append((command, timeout))
        if not self.results:
            raise AssertionError("AgentLoop dispatched an unexpected command")
        return self.results.pop(0)


class RecordingRunStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object]]] = []

    def event(self, event: str, payload: Mapping[str, object]) -> None:
        self.events.append((event, payload))

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        return ArtifactRef(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media_type,
        )


def command_turn(
    call_id: str,
    command: str,
    *,
    timeout: int = 30,
) -> ModelTurn:
    arguments: dict[str, object] = {"command": command, "timeout": timeout}
    return ModelTurn(
        tool_calls=(
            ToolCall(
                call_id=call_id,
                name="run_command",
                arguments=arguments,
            ),
        )
    )


class CoreSchemaTests(unittest.TestCase):
    def test_run_command_is_the_sole_canonical_tool_schema(self) -> None:
        self.assertEqual(
            RUN_COMMAND_TOOL_SCHEMA,
            {
                "name": "run_command",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 300,
                        },
                    },
                    "required": ["command", "timeout"],
                    "additionalProperties": False,
                },
            },
        )
        self.assertIn("timeout", RUN_COMMAND_TOOL_SCHEMA["parameters"]["required"])

    def test_normalized_values_preserve_typed_attachments(self) -> None:
        source_arguments = {"command": "render"}
        call = ToolCall("call-1", "run_command", source_arguments)
        source_arguments["command"] = "changed later"
        turn = ModelTurn(text="Inspecting.", tool_calls=[call])
        image = ImageAttachment(
            data=b"png",
            media_type="image/png",
            source_path="/workspace/render.png",
        )
        file_attachment = FileAttachment(
            data=b"step",
            source_path="/workspace/output.step",
        )
        result = CommandResult(
            exit_code=0,
            stdout="ok",
            attachments=[image, file_attachment],
            model_fields={"status": "success", "_harness": {"tool_step": 3}},
            model_output='{"status":"complete"}',
        )

        self.assertEqual(call.arguments, {"command": "render"})
        self.assertIsInstance(turn.tool_calls, tuple)
        self.assertIsInstance(result.attachments, tuple)
        self.assertEqual(
            [attachment.kind for attachment in result.attachments],
            ["image", "file"],
        )
        self.assertFalse(result.completed)
        self.assertEqual(result.model_fields["_harness"]["tool_step"], 3)
        self.assertEqual(result.model_output, '{"status":"complete"}')

    def test_model_turn_requires_positive_billed_response_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "billed_responses"):
            ModelTurn(billed_responses=0)

    def test_null_run_store_returns_content_identity(self) -> None:
        artifact = NullRunStore().write_artifact(
            "reports/final.md",
            b"complete",
            media_type="text/markdown",
        )
        self.assertEqual(
            artifact.sha256,
            hashlib.sha256(b"complete").hexdigest(),
        )
        self.assertEqual(artifact.size_bytes, 8)


class ControlClientProvisioningTests(unittest.TestCase):
    def test_exact_client_sources_and_hashes_are_shared_by_all_environments(
        self,
    ) -> None:
        expected_names = (
            "binder.py",
            "critic.py",
            "verifier.py",
            "image_show.py",
            "ask_user.py",
            "submit.py",
        )
        self.assertEqual(CLIENT_FILENAMES, expected_names)
        self.assertEqual(CLIENT_INSTALL_ROOT, "/opt/cuawright-tools")

        packaged = files("cuawright.environments.control_clients")
        expected_hashes = {
            name: hashlib.sha256(packaged.joinpath(name).read_bytes()).hexdigest()
            for name in expected_names
        }
        self.assertEqual(dict(CLIENT_SHA256), expected_hashes)
        self.assertEqual(
            {
                name: hashlib.sha256(data).hexdigest()
                for name, data in CLIENT_SOURCES.items()
            },
            expected_hashes,
        )
        self.assertEqual(
            control_client_manifest()["clients"],
            expected_hashes,
        )

        repository = Path(__file__).resolve().parents[1]
        for relative in (
            "src/cuawright/environments/benchcad/container/Dockerfile",
            "src/cuawright/environments/cadgenbench/container/Dockerfile",
        ):
            dockerfile = (repository / relative).read_text(encoding="utf-8")
            self.assertIn("--chown=0:0 --chmod=0555", dockerfile)
            self.assertIn(CLIENT_INSTALL_ROOT, dockerfile)
            for name in expected_names:
                self.assertIn(f"control_clients/{name}", dockerfile)
            self.assertNotIn("control_clients/report.py", dockerfile)

        osworld_runner = (repository / "src/cuawright/run/osworld.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            osworld_runner.index("/usr/local/sbin:/usr/local/bin"),
            osworld_runner.index("$HOME/.local/bin"),
        )


class AgentLoopTests(unittest.TestCase):
    def test_dispatches_one_command_per_turn_and_stops_on_completion(self) -> None:
        adapter = FakeAdapter(
            [
                command_turn("call-1", "printf ready", timeout=5),
                command_turn("call-2", SUBMIT_COMMAND),
                command_turn("unused", "must-not-run"),
            ]
        )
        plugin = FakePlugin(
            [
                CommandResult(exit_code=0, stdout="ready"),
                CommandResult(exit_code=0, stdout="submitted", completed=True),
            ]
        )
        store = RecordingRunStore()
        budget = SharedStepBudget(5)

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=budget,
            run_store=store,
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.stop_reason, "completed")
        self.assertEqual(result.steps_used, 2)
        self.assertEqual(result.commands_executed, 2)
        self.assertEqual(
            plugin.commands,
            [
                ("printf ready", 5),
                (SUBMIT_COMMAND, 30),
            ],
        )
        self.assertEqual(len(adapter.command_results), 2)
        self.assertEqual(len(adapter.turns), 1)
        self.assertTrue(
            all(
                list(tools) == [RUN_COMMAND_TOOL_SCHEMA] for tools in adapter.tools_seen
            )
        )
        self.assertEqual(store.events[-1][0], "agent_loop_stopped")

    def test_specialist_final_turn_returns_text_without_exposing_another_tool(
        self,
    ) -> None:
        report = '{"status":"complete","report":"Bound.","unresolved":[]}'
        adapter = FakeAdapter([ModelTurn(text=report)])
        plugin = FakePlugin([])
        budget = SharedStepBudget(1)

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=budget,
            role="binder",
            text_completion_handler=lambda text: CommandResult(
                0,
                stdout=text,
                completed=True,
            ),
            final_turn_without_tools=True,
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(result.commands_executed, 0)
        self.assertEqual(adapter.tools_seen, [()])
        self.assertEqual(result.last_turn.text, report)
        self.assertEqual(result.last_command_result.stdout, report)

    def test_multi_command_turn_uses_one_model_response_step(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "run_command",
                            {"command": "first", "timeout": 3},
                        ),
                        ToolCall(
                            "call-2",
                            "run_command",
                            {"command": "second", "timeout": 30},
                        ),
                    )
                )
            ]
        )
        plugin = FakePlugin([CommandResult(0), CommandResult(0)])
        budget = SharedStepBudget(1)

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=budget,
        ).run()

        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(result.commands_executed, 2)
        self.assertEqual(plugin.commands, [("first", 3), ("second", 30)])
        self.assertEqual(
            [call.call_id for call, _ in adapter.command_results],
            ["call-1", "call-2"],
        )
        self.assertEqual(budget.by_role, {"actor": 1})

    def test_completion_mid_batch_skips_remaining_commands(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "run_command",
                            {"command": "first", "timeout": 30},
                        ),
                        ToolCall(
                            "call-2",
                            "run_command",
                            {"command": "submit", "timeout": 30},
                        ),
                        ToolCall(
                            "call-3",
                            "run_command",
                            {"command": "unused", "timeout": 30},
                        ),
                    )
                )
            ]
        )
        plugin = FakePlugin(
            [
                CommandResult(0),
                CommandResult(0, completed=True),
                CommandResult(0),
            ]
        )
        budget = SharedStepBudget(3)

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=budget,
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(result.commands_executed, 2)
        self.assertEqual(plugin.commands, [("first", 30), ("submit", 30)])
        self.assertEqual(len(plugin.results), 1)
        self.assertEqual(
            [call.call_id for call, _ in adapter.command_results],
            ["call-1", "call-2", "call-3"],
        )
        self.assertIn("Skipped because", adapter.command_results[-1][1].stderr)

    def test_unsupported_tool_returns_errors_and_recovers(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "run_command",
                            {"command": "first", "timeout": 30},
                        ),
                        ToolCall("call-2", "read_image", {"path": "image.png"}),
                    )
                ),
                command_turn("call-3", SUBMIT_COMMAND),
            ]
        )
        plugin = FakePlugin([CommandResult(0, completed=True)])
        budget = SharedStepBudget(3)

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=budget,
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(plugin.commands, [(SUBMIT_COMMAND, 30)])
        self.assertEqual(
            [call.call_id for call, _ in adapter.command_results],
            ["call-1", "call-2", "call-3"],
        )
        self.assertEqual(adapter.command_results[0][1].exit_code, 2)
        self.assertEqual(adapter.command_results[1][1].exit_code, 2)
        self.assertEqual(budget.used_steps, 2)

    def test_multiple_loops_share_global_step_accounting(self) -> None:
        budget = SharedStepBudget(3)
        actor = AgentLoop(
            adapter=FakeAdapter([command_turn("actor-1", "actor command")]),
            plugin=FakePlugin([CommandResult(0, completed=True)]),
            budget=budget,
            role="actor",
        )
        verifier = AgentLoop(
            adapter=FakeAdapter(
                [
                    command_turn("verify-1", "inspect"),
                    command_turn("verify-2", VERIFIER_FINAL_COMMAND),
                ]
            ),
            plugin=FakePlugin(
                [
                    CommandResult(0),
                    CommandResult(0, completed=True),
                ]
            ),
            budget=budget,
            role="verifier",
        )

        actor_result = actor.run()
        verifier_result = verifier.run()

        self.assertEqual(actor_result.steps_used, 1)
        self.assertEqual(verifier_result.steps_used, 2)
        self.assertEqual(budget.used_steps, 3)
        self.assertEqual(budget.remaining_steps, 0)
        self.assertEqual(budget.by_role, {"actor": 1, "verifier": 2})

    def test_budget_exhaustion_is_a_normal_incomplete_result(self) -> None:
        result = AgentLoop(
            adapter=FakeAdapter([command_turn("call-1", "keep working")]),
            plugin=FakePlugin([CommandResult(0)]),
            budget=SharedStepBudget(1),
        ).run()

        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(result.steps_used, 1)

    def test_text_only_turn_is_recovered_and_charged(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(text="Should I continue?"),
                command_turn("submit", SUBMIT_COMMAND),
            ]
        )
        result = AgentLoop(
            adapter=adapter,
            plugin=FakePlugin([CommandResult(0, completed=True)]),
            budget=SharedStepBudget(3),
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 2)
        self.assertEqual(len(adapter.user_texts), 1)
        self.assertIn("no run_command call", adapter.user_texts[0])

    def test_non_verifier_recovery_uses_inline_ask_user_form(self) -> None:
        expected = 'python /opt/cuawright-tools/ask_user.py --question "TEXT"'
        for role in ("actor", "binder", "critic"):
            with self.subTest(role=role):
                adapter = FakeAdapter(
                    [
                        ModelTurn(text="Need input."),
                        command_turn("complete", "ordinary command"),
                    ]
                )
                AgentLoop(
                    adapter=adapter,
                    plugin=FakePlugin([CommandResult(0, completed=True)]),
                    budget=SharedStepBudget(3),
                    role=role,
                ).run()

                self.assertEqual(len(adapter.user_texts), 1)
                self.assertIn(expected, adapter.user_texts[0])
                self.assertIn("standalone run_command", adapter.user_texts[0])
                self.assertNotIn("question file", adapter.user_texts[0])
                self.assertNotIn("--question PATH", adapter.user_texts[0])

    def test_malformed_tool_arguments_return_error_and_retry(self) -> None:
        malformed = ToolCall(
            "bad",
            "run_command",
            {},
            parse_error="unterminated string",
            raw_arguments='{"command":"unfinished',
        )
        adapter = FakeAdapter(
            [
                ModelTurn(tool_calls=(malformed,)),
                command_turn("good", SUBMIT_COMMAND),
            ]
        )
        plugin = FakePlugin([CommandResult(0, completed=True)])

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=SharedStepBudget(3),
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 2)
        self.assertEqual(plugin.commands, [(SUBMIT_COMMAND, 30)])
        self.assertEqual(adapter.command_results[0][0].call_id, "bad")
        self.assertEqual(adapter.command_results[0][1].exit_code, 2)
        self.assertIn("unterminated string", adapter.command_results[0][1].stderr)

    def test_verifier_text_recovery_does_not_suggest_user_question(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(text="I cannot tell."),
                command_turn("report", VERIFIER_FINAL_COMMAND),
            ]
        )
        AgentLoop(
            adapter=adapter,
            plugin=FakePlugin([CommandResult(0, completed=True)]),
            budget=SharedStepBudget(3),
            role="verifier",
        ).run()

        self.assertEqual(len(adapter.user_texts), 1)
        self.assertNotIn("ask_user.py", adapter.user_texts[0])

    def test_recovery_turns_trigger_the_normal_compaction_check(self) -> None:
        malformed = ToolCall(
            "malformed",
            "run_command",
            {},
            parse_error="unterminated string",
        )
        cases = {
            "no_tool": ModelTurn(text="No command."),
            "malformed_arguments": ModelTurn(tool_calls=(malformed,)),
            "invalid_call": ModelTurn(
                tool_calls=(ToolCall("invalid", "read_image", {"path": "x"}),)
            ),
            "rejected_completion_batch": ModelTurn(
                tool_calls=(
                    ToolCall(
                        "inspect",
                        "run_command",
                        {"command": "printf inspect", "timeout": 30},
                    ),
                    ToolCall(
                        "submit",
                        "run_command",
                        {"command": SUBMIT_COMMAND, "timeout": 30},
                    ),
                )
            ),
        }

        for name, recovery_turn in cases.items():
            with self.subTest(name=name):
                first = FakeAdapter([recovery_turn])
                second = FakeAdapter([command_turn("complete", SUBMIT_COMMAND)])

                def compact(adapter, charge_step):
                    self.assertIs(adapter, first)
                    charge_step()
                    return second

                result = AgentLoop(
                    adapter=first,
                    plugin=FakePlugin([CommandResult(0, completed=True)]),
                    budget=SharedStepBudget(4),
                    compact_every=1,
                    compaction_handler=compact,
                ).run()

                self.assertTrue(result.completed)
                self.assertEqual(result.compactions, 1)
                self.assertEqual(result.steps_used, 3)

    def test_billed_provider_failure_consumes_shared_step(self) -> None:
        class BilledFailure(RuntimeError):
            billed_response = True

        class FailingAdapter(FakeAdapter):
            def next_turn(self, *, tools):
                raise BilledFailure("recorded provider stop")

        budget = SharedStepBudget(2)
        with self.assertRaisesRegex(BilledFailure, "provider stop"):
            AgentLoop(
                adapter=FailingAdapter([]),
                plugin=FakePlugin([]),
                budget=budget,
            ).run()

        self.assertEqual(budget.used_steps, 1)
        self.assertEqual(budget.by_role, {"actor": 1})

    def test_multiple_billed_provider_failures_each_consume_shared_step(self) -> None:
        class BilledFailure(RuntimeError):
            billed_response_count = 3

        class FailingAdapter(FakeAdapter):
            def next_turn(self, *, tools):
                raise BilledFailure("recorded provider stops")

        budget = SharedStepBudget(4)
        with self.assertRaisesRegex(BilledFailure, "provider stops"):
            AgentLoop(
                adapter=FailingAdapter([]),
                plugin=FakePlugin([]),
                budget=budget,
            ).run()

        self.assertEqual(budget.used_steps, 3)
        self.assertEqual(budget.by_role, {"actor": 3})

    def test_hidden_provider_continuations_each_consume_shared_step(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            "submit",
                            "run_command",
                            {
                                "command": "python /opt/cuawright-tools/submit.py",
                                "timeout": 30,
                            },
                        ),
                    ),
                    billed_responses=3,
                )
            ]
        )
        budget = SharedStepBudget(4)

        result = AgentLoop(
            adapter=adapter,
            plugin=FakePlugin([CommandResult(0, completed=True)]),
            budget=budget,
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(budget.by_role, {"actor": 3})

    def test_provider_continuation_stops_cleanly_at_budget_boundary(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(continue_immediately=True),
                command_turn("must-not-run", SUBMIT_COMMAND),
            ]
        )

        result = AgentLoop(
            adapter=adapter,
            plugin=FakePlugin([]),
            budget=SharedStepBudget(1),
        ).run()

        self.assertFalse(result.completed)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(result.steps_used, 1)
        self.assertEqual(len(adapter.turns), 1)

    def test_compaction_consumes_one_shared_step_and_replaces_adapter(self) -> None:
        first = FakeAdapter([command_turn("call-1", "inspect")])
        second = FakeAdapter([command_turn("call-2", SUBMIT_COMMAND)])
        plugin = FakePlugin(
            [
                CommandResult(0),
                CommandResult(0, completed=True),
            ]
        )
        compacted: list[FakeAdapter] = []

        def compact(adapter, charge_step):
            self.assertIs(adapter, first)
            compacted.append(adapter)
            charge_step(1)
            return second

        budget = SharedStepBudget(4)
        result = AgentLoop(
            adapter=first,
            plugin=plugin,
            budget=budget,
            compact_every=1,
            compaction_handler=compact,
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(result.compactions, 1)
        self.assertEqual(compacted, [first])
        self.assertEqual(
            budget.by_stage,
            {"agent-loop": 2, "agent-loop-compaction": 1},
        )

    def test_compaction_configuration_must_be_complete(self) -> None:
        common = {
            "adapter": FakeAdapter([]),
            "plugin": FakePlugin([]),
            "budget": SharedStepBudget(2),
        }
        with self.assertRaisesRegex(ValueError, "configured together"):
            AgentLoop(**common, compact_every=1)
        with self.assertRaisesRegex(ValueError, "configured together"):
            AgentLoop(**common, compaction_handler=lambda adapter: adapter)

    def test_failed_compaction_parse_remains_charged(self) -> None:
        budget = SharedStepBudget(3)

        def compact(adapter, charge_step):
            charge_step()
            raise RuntimeError("empty compaction summary")

        with self.assertRaisesRegex(RuntimeError, "empty compaction"):
            AgentLoop(
                adapter=FakeAdapter([command_turn("call-1", "inspect")]),
                plugin=FakePlugin([CommandResult(0)]),
                budget=budget,
                compact_every=1,
                compaction_handler=compact,
            ).run()

        self.assertEqual(budget.used_steps, 2)
        self.assertEqual(
            budget.by_stage,
            {"agent-loop": 1, "agent-loop-compaction": 1},
        )

    def test_completion_command_must_be_the_only_call_in_turn(self) -> None:
        adapter = FakeAdapter(
            [
                ModelTurn(
                    tool_calls=(
                        ToolCall(
                            "inspect",
                            "run_command",
                            {"command": "printf inspect", "timeout": 30},
                        ),
                        ToolCall(
                            "submit",
                            "run_command",
                            {"command": SUBMIT_COMMAND, "timeout": 30},
                        ),
                    )
                ),
                command_turn("retry", "ordinary command"),
            ]
        )
        plugin = FakePlugin([CommandResult(0, completed=True)])

        result = AgentLoop(
            adapter=adapter,
            plugin=plugin,
            budget=SharedStepBudget(3),
        ).run()

        self.assertTrue(result.completed)
        self.assertEqual(plugin.commands, [("ordinary command", 30)])
        self.assertEqual(
            [call.call_id for call, _ in adapter.command_results],
            ["inspect", "submit", "retry"],
        )
        self.assertEqual(adapter.command_results[0][1].exit_code, 2)
        self.assertEqual(adapter.command_results[1][1].exit_code, 2)


class CoreImportBoundaryTests(unittest.TestCase):
    def test_core_modules_have_no_benchmark_or_provider_imports(self) -> None:
        package = Path(__file__).parents[1] / "src" / "cuawright"
        core_files = [
            package / "__init__.py",
            package / "agents" / "default.py",
            package / "artifacts.py",
            package / "environments" / "base.py",
            package / "budget.py",
            package / "models" / "base.py",
            package / "environments" / "control.py",
        ]
        forbidden_roots = {
            "anthropic",
            "cuawright.environments.benchcad",
            "cuawright.environments.cadgenbench",
            "desktop_env",
            "openai",
        }
        forbidden_modules = {
            "cuawright.models.anthropic_client",
            "cuawright.run.task",
        }

        for path in core_files:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
                self.assertFalse(
                    any(
                        name.split(".", 1)[0] in forbidden_roots
                        or name in forbidden_modules
                        for name in imports
                    ),
                    imports,
                )


if __name__ == "__main__":
    unittest.main()

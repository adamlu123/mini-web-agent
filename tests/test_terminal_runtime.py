from __future__ import annotations

import json
import shlex
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cuawright.agents.orchestration import (
    TerminalAgentRuntime,
    _PromptingAdapter,
    finalization_window,
)
from cuawright.artifacts import ArtifactRef, NullRunStore, TraceWriter
from cuawright.budget import ScopedStepBudget
from cuawright.config.prompts import OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE
from cuawright.environments.base import CommandResult
from cuawright.environments.control import AcceptedReport
from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA, ModelTurn, ToolCall
from cuawright.run.task import TaskConfig, TaskRunner


def client(operation: str, *arguments: str) -> str:
    return shlex.join(["python", f"/opt/cuawright-tools/{operation}.py", *arguments])


class MemoryPlugin:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []
        self.atomic_writes: list[tuple[str, bytes]] = []
        self.control_reads = 0

    def run_command(self, command: str, timeout: int | None = None) -> CommandResult:
        del timeout
        self.commands.append(command)
        argv = shlex.split(command)
        if len(argv) >= 2 and argv[0] == "python":
            operation = argv[1].rsplit("/", 1)[-1].removesuffix(".py")
            payload: dict[str, object] = {
                "cuawright_control": 1,
                "operation": operation,
            }
            values = iter(argv[2:])
            for name in values:
                value = next(values)
                payload[name.removeprefix("--").replace("-", "_")] = (
                    int(value) if name == "--max-steps" else value
                )
            return CommandResult(
                0,
                stdout=json.dumps(payload, separators=(",", ":")) + "\n",
            )
        return CommandResult(
            0,
            stdout="ok",
            inspected=command == "inspect-live-state",
        )

    def read_file(self, path: str, *, max_bytes: int) -> bytes:
        return self.files[path][:max_bytes]

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        self.control_reads += 1
        if not path.startswith(workspace + "/"):
            raise OSError("outside workspace")
        return self.files[path][:max_bytes]

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        if not path.startswith(workspace + "/"):
            raise OSError("outside workspace")
        self.files[path] = data
        self.atomic_writes.append((path, data))


class RecordingRunStore(NullRunStore):
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object]]] = []
        self.artifacts: dict[str, bytes] = {}

    def event(self, event: str, payload: Mapping[str, object]) -> None:
        self.events.append((event, payload))

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        self.artifacts[path] = data
        return super().write_artifact(path, data, media_type=media_type)


class FakeThread:
    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = turns
        self.user_messages: list[str] = []

    def add_user_text(self, text: str) -> None:
        self.user_messages.append(text)


class FakeAdapter:
    def __init__(self, thread: FakeThread) -> None:
        self.thread = thread
        self.results: list[tuple[ToolCall, CommandResult]] = []
        self.tools_seen: list[tuple[Mapping[str, object], ...]] = []

    def next_turn(
        self,
        *,
        tools: Sequence[Mapping[str, object]],
    ) -> ModelTurn:
        normalized = tuple(tools)
        self.tools_seen.append(normalized)
        if list(tools) not in ([], [RUN_COMMAND_TOOL_SCHEMA]):
            raise AssertionError("runtime exposed a noncanonical tool")
        return self.thread.turns.pop(0)

    def add_command_result(
        self,
        tool_call: ToolCall,
        result: CommandResult,
    ) -> None:
        self.results.append((tool_call, result))

    def add_user_text(self, text: str) -> None:
        self.thread.add_user_text(text)


def command(
    call_id: str,
    value: str,
    *,
    billed_responses: int = 1,
) -> ModelTurn:
    return ModelTurn(
        tool_calls=(
            ToolCall(
                call_id,
                "run_command",
                {"command": value, "timeout": 30},
            ),
        ),
        billed_responses=billed_responses,
    )


def report_turn(
    status: str,
    report: str,
    unresolved: list[str] | None = None,
) -> ModelTurn:
    return ModelTurn(
        text=json.dumps(
            {
                "status": status,
                "report": report,
                "unresolved": unresolved or [],
            },
            ensure_ascii=False,
        )
    )


class PromptingAdapterTests(unittest.TestCase):
    def test_provider_continuation_does_not_insert_dynamic_user_message(self) -> None:
        thread = FakeThread(
            [
                ModelTurn(continue_immediately=True),
                ModelTurn(text="done"),
            ]
        )
        adapter = _PromptingAdapter(FakeAdapter(thread), lambda: "budget status")

        first = adapter.next_turn(tools=(RUN_COMMAND_TOOL_SCHEMA,))
        second = adapter.next_turn(tools=(RUN_COMMAND_TOOL_SCHEMA,))

        self.assertTrue(first.continue_immediately)
        self.assertFalse(second.continue_immediately)
        self.assertEqual(thread.user_messages, ["budget status"])


class TerminalRuntimeTests(unittest.TestCase):
    def test_linear_topology_cannot_instantiate_specialists(self) -> None:
        plugin = MemoryPlugin()
        created_roles: list[str] = []
        prompts: list[str] = []
        actor_adapter: FakeAdapter | None = None

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            created_roles.append(role)
            prompts.extend((system_prompt, user_prompt))
            return FakeThread(
                [
                    command(
                        "forbidden-binder",
                        client(
                            "binder",
                            "--request",
                            "Try to start a specialist.",
                            "--max-steps",
                            "1",
                        ),
                    ),
                    command("submit", client("submit")),
                ]
            )

        def adapter_factory(thread: FakeThread) -> FakeAdapter:
            nonlocal actor_adapter
            actor_adapter = FakeAdapter(thread)
            return actor_adapter

        outcome = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=adapter_factory,
            instruction="Complete the task.",
            task_id="linear",
            total_steps=3,
            compact_every=0,
            agent_topology="linear",
        ).run()

        self.assertTrue(outcome.loop.completed)
        self.assertEqual(created_roles, ["actor"])
        self.assertEqual(tuple(outcome.state.accepted_reports), ())
        self.assertEqual(outcome.state.submission_missing_reports, ())
        self.assertEqual(tuple(outcome.prompt_hashes), ("actor",))
        self.assertIsNotNone(actor_adapter)
        self.assertIn(
            "disabled for this run",
            actor_adapter.results[0][1].stderr,
        )
        for forbidden in ("Binder", "Critic", "Verifier", "subagent"):
            self.assertNotIn(
                forbidden.lower(),
                "\n".join(prompts).lower(),
            )

    def test_inline_terminal_flow_uses_host_only_report_artifacts(self) -> None:
        plugin = MemoryPlugin()
        run_store = RecordingRunStore()
        threads: dict[str, list[FakeThread]] = {}
        prompts: dict[str, list[str]] = {}
        adapters: dict[str, list[FakeAdapter]] = {}

        actor_turns = [
            command(
                "ask",
                client("ask_user", "--question", "Which account should be used?"),
            ),
            command(
                "binder",
                client(
                    "binder",
                    "--request",
                    "Inspect exact requirements.",
                    "--max-steps",
                    "1",
                ),
            ),
            command(
                "verifier",
                client(
                    "verifier",
                    "--request",
                    "Inspect the saved canonical state.",
                    "--max-steps",
                    "2",
                ),
            ),
            command("submit", client("submit")),
        ]

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            del system_prompt
            prompts.setdefault(role, []).append(user_prompt)
            if role == "actor":
                turns = actor_turns
            elif role == "binder":
                turns = [report_turn("complete", "# Binder\n\nUse profile A.")]
            else:
                turns = [
                    command("inspect", "inspect-live-state"),
                    report_turn("ready", "# Verifier\n\nState is ready."),
                ]
            thread = FakeThread(list(turns))
            threads.setdefault(role, []).append(thread)
            return thread

        def adapter_factory(thread: FakeThread) -> FakeAdapter:
            adapter = FakeAdapter(thread)
            role = next(
                role for role, role_threads in threads.items() if thread in role_threads
            )
            adapters.setdefault(role, []).append(adapter)
            return adapter

        outcome = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=adapter_factory,
            instruction="Complete the task.",
            task_id="inline-flow",
            total_steps=12,
            compact_every=0,
            run_store=run_store,
            ask_user=lambda question: f"Answer: {question}",
        ).run()

        self.assertTrue(outcome.loop.completed)
        self.assertEqual(
            outcome.state.completed_roles,
            ("binder", "verifier"),
        )
        self.assertFalse(outcome.state.submission_forced)
        self.assertEqual(len(threads["actor"]), 1)
        self.assertEqual(
            [len(threads[role]) for role in ("binder", "verifier")],
            [1, 1],
        )
        self.assertEqual(
            [
                allocation.requested_steps
                for allocation in outcome.state.invocation_allocations
            ],
            [1, 2],
        )
        self.assertNotIn("# Binder\n\nUse profile A.", prompts["verifier"][0])
        self.assertIn(
            "Independently inspect the live saved and reloaded canonical state",
            prompts["verifier"][0],
        )
        self.assertFalse(
            any(
                namespace in path
                for path in plugin.files
                for namespace in ("/requests/", "/questions/", "/reports/")
            )
        )
        self.assertEqual(
            plugin.files["/workspace/.agent/task.md"], b"Complete the task."
        )
        self.assertTrue(
            all(
                report.artifact.path.endswith("/report.json")
                for report in outcome.state.accepted_reports.values()
            )
        )
        ask_result = json.loads(adapters["actor"][0].results[0][1].stdout)
        self.assertEqual(
            ask_result,
            {
                "status": "answered",
                "answer": "Answer: Which account should be used?",
            },
        )

    def test_actor_and_specialists_use_split_question_callbacks(self) -> None:
        plugin = MemoryPlugin()
        actor_questions: list[str] = []
        specialist_questions: list[str] = []
        actor_adapter: FakeAdapter | None = None

        question = "Which date should be used?"

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            if role == "actor":
                return FakeThread(
                    [
                        command(
                            "ask-actor",
                            client("ask_user", "--question", question),
                        ),
                        command(
                            "binder",
                            client(
                                "binder",
                                "--request",
                                "Resolve the date.",
                                "--max-steps",
                                "2",
                            ),
                        ),
                        command("finish", "printf done"),
                    ]
                )
            return FakeThread(
                [
                    command(
                        "ask-specialist",
                        client("ask_user", "--question", question),
                    ),
                    report_turn("complete", "Date resolved."),
                ]
            )

        def adapter_factory(thread: FakeThread) -> FakeAdapter:
            nonlocal actor_adapter
            adapter = FakeAdapter(thread)
            if actor_adapter is None:
                actor_adapter = adapter
            return adapter

        TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=adapter_factory,
            instruction="Complete the task.",
            task_id="split-ask",
            total_steps=5,
            compact_every=0,
            ask_user=lambda value: actor_questions.append(value) or "actor answer",
            specialist_ask_user=(
                lambda value: specialist_questions.append(value) or "specialist answer"
            ),
        ).run()

        self.assertEqual(actor_questions, [question])
        self.assertEqual(specialist_questions, [question])
        self.assertIsNotNone(actor_adapter)
        answers = [
            json.loads(result.stdout)["answer"]
            for _, result in actor_adapter.results
            if result.stdout.startswith("{") and "answer" in result.stdout
        ]
        self.assertEqual(answers, ["actor answer"])

    def test_malformed_bounded_final_text_is_preserved(self) -> None:
        plugin = MemoryPlugin()
        run_store = RecordingRunStore()
        malformed = '  {"status":"complete","report":"Useful partial findings\n'
        actor_adapter: FakeAdapter | None = None

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            if role == "actor":
                return FakeThread(
                    [
                        command(
                            "binder",
                            client(
                                "binder",
                                "--request",
                                "Inspect.",
                                "--max-steps",
                                "1",
                            ),
                        ),
                        command("finish", "printf done"),
                    ]
                )
            return FakeThread([ModelTurn(text=malformed)])

        def adapter_factory(thread: FakeThread) -> FakeAdapter:
            nonlocal actor_adapter
            adapter = FakeAdapter(thread)
            if actor_adapter is None:
                actor_adapter = adapter
            return adapter

        result = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=adapter_factory,
            instruction="Complete the task.",
            task_id="preserve-malformed",
            total_steps=3,
            compact_every=0,
            run_store=run_store,
        ).run()

        report = result.state.accepted_reports["binder"]
        self.assertEqual(report.status, "blocked")
        self.assertEqual(report.body, malformed)
        self.assertTrue(
            any(
                event == "specialist_invalid_text_preserved"
                for event, _ in run_store.events
            )
        )
        self.assertIsNotNone(actor_adapter)
        returned = json.loads(actor_adapter.results[0][1].stdout)
        self.assertEqual(returned["status"], "forced")
        self.assertEqual(returned["report"]["report"], malformed)

    def test_tool_turn_chatter_is_not_preserved_as_report(self) -> None:
        plugin = MemoryPlugin()

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            del system_prompt, user_prompt
            if role == "actor":
                return FakeThread(
                    [
                        command(
                            "binder",
                            client(
                                "binder",
                                "--request",
                                "Inspect.",
                                "--max-steps",
                                "2",
                            ),
                        ),
                        command("finish", "printf done"),
                    ]
                )
            return FakeThread(
                [
                    ModelTurn(
                        text="Still inspecting.",
                        tool_calls=(
                            ToolCall(
                                "inspect",
                                "run_command",
                                {"command": "printf inspect", "timeout": 30},
                            ),
                        ),
                        billed_responses=2,
                    )
                ]
            )

        result = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=FakeAdapter,
            instruction="Complete the task.",
            task_id="ignore-chatter",
            total_steps=4,
            compact_every=0,
        ).run()

        report = result.state.accepted_reports["binder"]
        self.assertEqual(report.status, "blocked")
        self.assertNotIn("Still inspecting.", report.body)
        self.assertIn("exhausted its allocated", report.body)

    def test_task_file_tracks_current_effective_phase_instruction(self) -> None:
        plugin = MemoryPlugin()
        actor_thread: FakeThread | None = None

        def phase_turns(label: str) -> list[ModelTurn]:
            return [
                command(
                    f"binder-{label}",
                    client(
                        "binder",
                        "--request",
                        f"Bind {label}.",
                        "--max-steps",
                        "1",
                    ),
                ),
                command(
                    f"verifier-{label}",
                    client(
                        "verifier",
                        "--request",
                        f"Verify {label}.",
                        "--max-steps",
                        "2",
                    ),
                ),
                command(f"submit-{label}", client("submit")),
            ]

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            nonlocal actor_thread
            del system_prompt, user_prompt
            if role == "actor":
                actor_thread = FakeThread(phase_turns("one") + phase_turns("two"))
                return actor_thread
            status = {
                "binder": "complete",
                "verifier": "ready",
            }[role]
            turns = []
            if role == "verifier":
                turns.append(command("inspect", "inspect-live-state"))
            turns.append(report_turn(status, f"{role} phase report"))
            return FakeThread(turns)

        runtime = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=FakeAdapter,
            instruction="Overall phase schedule.",
            specialist_instruction="Phase one instruction.",
            initial_actor_message="Phase 1/2.",
            task_id="multi",
            total_steps=20,
            compact_every=0,
        )

        first = runtime.run()
        first_used = runtime.budget.used_steps
        second = runtime.continue_phase(
            instruction="Phase two instruction.",
            phase_id="phase-2",
            phase_index=2,
            phase_count=2,
        )

        self.assertTrue(first.loop.completed)
        self.assertTrue(second.loop.completed)
        self.assertGreater(runtime.budget.used_steps, first_used)
        self.assertEqual(runtime.state.phase_generation, 2)
        self.assertEqual(
            runtime.state.invocation_counts,
            {
                "binder": 1,
                "critic": 0,
                "verifier": 1,
            },
        )
        self.assertEqual(len(runtime.state.invocation_allocations), 2)
        self.assertIsNotNone(actor_thread)
        self.assertTrue(
            any(
                "previous phase has been evaluated" in text
                for text in actor_thread.user_messages
            )
        )
        task_writes = [
            data
            for path, data in plugin.atomic_writes
            if path == "/workspace/.agent/task.md"
        ]
        self.assertEqual(
            task_writes,
            [b"Phase one instruction.", b"Phase two instruction."],
        )
        self.assertEqual(
            plugin.files["/workspace/.agent/task.md"],
            b"Phase two instruction.",
        )

    def test_budget_exhaustion_creates_host_forced_submission_only(self) -> None:
        plugin = MemoryPlugin()
        run_store = RecordingRunStore()

        result = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=lambda **_: FakeThread(
                [command("inspect", "printf still-working")]
            ),
            adapter_factory=FakeAdapter,
            instruction="Complete the task.",
            task_id="forced",
            total_steps=1,
            compact_every=0,
            run_store=run_store,
        ).run()

        self.assertFalse(result.loop.completed)
        self.assertTrue(result.state.submission_forced)
        self.assertEqual(
            result.state.submission_missing_reports,
            ("binder", "verifier"),
        )
        self.assertNotIn("/workspace/.agent/submission.json", plugin.files)
        self.assertIsNotNone(result.state.submission)
        submission = json.loads(
            run_store.artifacts[result.state.submission.path].decode("utf-8")
        )
        forced_text = submission["forced_actor_report"]["text"]
        self.assertEqual(json.loads(forced_text)["status"], "forced")

    def test_prior_reports_are_injected_from_host_state_without_guest_reads(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        runtime = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=lambda **_: FakeThread([]),
            adapter_factory=FakeAdapter,
            instruction="Complete the task.",
            task_id="host-prior",
            total_steps=10,
            compact_every=0,
        )
        runtime.state.accepted_reports["binder"] = AcceptedReport(
            invocation_id="binder-01",
            role="binder",
            status="complete",
            body="# Binder\n\nExact host-captured facts.",
            unresolved=("Need confirmation.",),
            artifact=ArtifactRef(
                "specialists/phase-001/binder-01/report.json",
                "a" * 64,
                128,
                "application/json",
            ),
        )

        reports = runtime._validated_prior_reports("critic")

        self.assertEqual(
            reports["binder"].report, "# Binder\n\nExact host-captured facts."
        )
        self.assertEqual(reports["binder"].unresolved, ("Need confirmation.",))
        self.assertEqual(plugin.control_reads, 0)

    def test_verifier_can_run_before_binder_or_critic(self) -> None:
        plugin = MemoryPlugin()
        verifier_prompts: list[str] = []

        def factory(*, role: str, system_prompt: str, user_prompt: str):
            del system_prompt
            if role == "actor":
                return FakeThread(
                    [
                        command(
                            "verifier",
                            client(
                                "verifier",
                                "--request",
                                "Inspect the current saved state.",
                                "--max-steps",
                                "2",
                            ),
                        ),
                        command("finish", "printf done"),
                    ]
                )
            verifier_prompts.append(user_prompt)
            return FakeThread(
                [
                    command("inspect", "inspect-live-state"),
                    report_turn("unknown", "No final candidate exists yet."),
                ]
            )

        outcome = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=factory,
            adapter_factory=FakeAdapter,
            instruction="Complete the task.",
            task_id="early-verifier",
            total_steps=4,
            compact_every=0,
        ).run()

        self.assertEqual(outcome.state.completed_roles, ("verifier",))
        self.assertEqual(len(verifier_prompts), 1)
        self.assertNotIn("Latest Binder report:", verifier_prompts[0])
        self.assertNotIn("Latest Critic report:", verifier_prompts[0])

    def test_finalization_window_and_final_specialist_turn_match_native(self) -> None:
        runtime = TerminalAgentRuntime(
            plugin=MemoryPlugin(),
            thread_factory=lambda **_: FakeThread([]),
            adapter_factory=FakeAdapter,
            instruction="Complete the task.",
            task_id="window",
            total_steps=300,
            compact_every=0,
        )
        runtime.budget.used_steps = 252
        prompt = runtime._turn_prompt("actor", runtime.budget)
        self.assertIn("Finalization window:", prompt)
        self.assertEqual(finalization_window(300), 48)
        runtime.budget.used_steps = 299
        final_prompt = runtime._turn_prompt("actor", runtime.budget)
        self.assertIn("or run", final_prompt)
        self.assertIn("Keep submit standalone", final_prompt)
        self.assertNotIn("persistent result and run", final_prompt)

        scoped = ScopedStepBudget(runtime.budget, 1)
        specialist_prompt = runtime._turn_prompt("binder", scoped)
        self.assertIn("Do not call run_command", specialist_prompt)
        self.assertIn("`status`, `report`, and `unresolved`", specialist_prompt)
        self.assertNotIn("report.py", specialist_prompt)


class TaskRunnerObserverIntegrationTests(unittest.TestCase):
    def test_terminal_benchmark_forwards_observer_to_real_broker_dispatch(
        self,
    ) -> None:
        class Plugin(MemoryPlugin):
            def __init__(self) -> None:
                super().__init__()
                self.closed = False

            def evaluate(self):
                return {"score": None}

            def close(self) -> None:
                self.closed = True

        with tempfile.TemporaryDirectory() as temporary:
            config = TaskConfig(
                task_id="observer-integration",
                api_key_file="unused",
                result_root=temporary,
                total_steps=2,
                compact_every=0,
            )
            runner = object.__new__(TaskRunner)
            runner.config = config
            runner.run_dir = Path(temporary) / "task-observer-integration"
            runner.trace = TraceWriter(runner.run_dir)
            runner.ask_user = None
            runner.provider_runtime = SimpleNamespace(close=Mock())
            runner.cost_tracker = None
            runner._actor_answered_questions = {}
            runner._specialist_answered_questions = {}
            runner._closed = False
            thread = FakeThread(
                [
                    command("ordinary", "touch visible state"),
                    command("submit", client("submit")),
                ]
            )
            runner.new_model_thread = lambda **_: thread
            plugin = Plugin()
            observations = []

            with patch(
                "cuawright.run.task.create_turn_provider",
                side_effect=lambda _provider, value: FakeAdapter(value),
            ):
                result = runner.run_terminal_benchmark(
                    benchmark="observer-test",
                    plugin=plugin,
                    instruction="Complete the task.",
                    manifest_extra={},
                    prompt_bundle=OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE,
                    command_observer=observations.append,
                )

        self.assertEqual(result["submission"]["status"], "submitted")
        self.assertEqual(
            [observation.command for observation in observations],
            ["touch visible state", client("submit")],
        )
        self.assertFalse(observations[0].result.completed)
        self.assertTrue(observations[1].result.completed)
        self.assertFalse(observations[1].submission_forced)
        self.assertTrue(plugin.closed)
        runner.provider_runtime.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

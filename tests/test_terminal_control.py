from __future__ import annotations

import hashlib
import io
import json
import shlex
import subprocess
import sys
import unittest
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from cuawright.artifacts import ArtifactRef
from cuawright.budget import SharedStepBudget
from cuawright.environments.base import (
    CommandResult,
    ImageAttachment,
    cap_command_text,
)
from cuawright.environments.control import (
    CONTROL_CLIENT_ROOT,
    MAX_CONTROL_CLIENT_STDOUT_BYTES,
    MAX_IMAGE_BYTES,
    MAX_REPORT_BYTES,
    MAX_REPORT_CHARACTERS,
    MAX_REQUEST_CHARACTERS,
    MAX_UNRESOLVED_ITEM_CHARACTERS,
    MAX_UNRESOLVED_ITEMS,
    AcceptedReport,
    SpecialistInvocation,
    SpecialistRunResult,
    TerminalCommandBroker,
    TerminalControlState,
)


WORKSPACE = "/workspace/.agent"
CLIENT_SOURCE_ROOT = (
    Path(__file__).parents[1] / "src" / "cuawright" / "environments" / "control_clients"
)


def client(name: str, *arguments: str) -> str:
    return shlex.join(["python", f"{CONTROL_CLIENT_ROOT}/{name}.py", *arguments])


class MemoryPlugin:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commands: list[tuple[str, int | None]] = []
        self.reads: list[tuple[str, int]] = []
        self.control_reads: list[tuple[str, str, int]] = []
        self.control_writes: list[tuple[str, str, bytes]] = []
        self.ordinary_results: dict[str, CommandResult] = {}
        self.client_result_transform = None

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        self.commands.append((command, timeout))
        argv = shlex.split(command)
        if (
            len(argv) >= 2
            and argv[0] == "python"
            and argv[1].startswith(f"{CONTROL_CLIENT_ROOT}/")
        ):
            source = CLIENT_SOURCE_ROOT / Path(argv[1]).name
            if not source.exists():
                return CommandResult(2, stderr=f"{source}: not found\n")
            completed = subprocess.run(
                [sys.executable, str(source), *argv[2:]],
                check=False,
                capture_output=True,
                text=True,
            )
            result = CommandResult(
                completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            if self.client_result_transform is not None:
                result = self.client_result_transform(argv[1], result)
            return result
        if command in self.ordinary_results:
            return self.ordinary_results[command]
        stdout, _ = cap_command_text(f"delegated: {command}")
        return CommandResult(0, stdout=stdout)

    def read_file(self, path: str, *, max_bytes: int) -> bytes:
        self.reads.append((path, max_bytes))
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        self.control_reads.append((path, workspace, max_bytes))
        raise AssertionError("native-inline control must not read guest control files")

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        self.control_writes.append((path, workspace, data))
        raise AssertionError("native-inline control must not write guest control files")


class RecordingRunStore:
    def __init__(self) -> None:
        self.artifacts: dict[str, tuple[bytes, str | None]] = {}
        self.events: list[tuple[str, Mapping[str, object]]] = []

    def event(self, event: str, payload: Mapping[str, object]) -> None:
        self.events.append((event, dict(payload)))

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        self.artifacts[path] = (data, media_type)
        return ArtifactRef(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media_type,
        )


class CompletingRunner:
    def __init__(
        self,
        *,
        statuses: Mapping[str, str] | None = None,
        bodies: Mapping[str, str] | None = None,
        unresolved: Mapping[str, list[str]] | None = None,
        inspect_verifier: bool = True,
        consume_step: bool = False,
    ) -> None:
        self.statuses = dict(
            statuses or {"binder": "complete", "critic": "accept", "verifier": "ready"}
        )
        self.bodies = dict(bodies or {})
        self.unresolved = dict(unresolved or {})
        self.inspect_verifier = inspect_verifier
        self.consume_step = consume_step
        self.invocations: list[SpecialistInvocation] = []
        self.report_results: list[CommandResult] = []

    def __call__(
        self,
        invocation: SpecialistInvocation,
        broker: TerminalCommandBroker,
    ) -> SpecialistRunResult:
        self.invocations.append(invocation)
        if self.consume_step:
            invocation.budget.consume(invocation.role, "test")
        if invocation.role == "verifier" and self.inspect_verifier:
            broker.run_command("inspect-live-state")
        report = {
            "status": self.statuses[invocation.role],
            "report": self.bodies.get(
                invocation.role,
                f"{invocation.role.title()} direct evidence.",
            ),
            "unresolved": self.unresolved.get(invocation.role, []),
        }
        result = broker.accept_text_report(json.dumps(report))
        self.report_results.append(result)
        return SpecialistRunResult(
            result.completed,
            1 if self.consume_step else 0,
        )


def make_broker(
    plugin: MemoryPlugin,
    *,
    budget: SharedStepBudget | None = None,
    runner=None,
    store: RecordingRunStore | None = None,
    ask_user=None,
    state: TerminalControlState | None = None,
    invocation_caps=None,
    step_limits=None,
) -> TerminalCommandBroker:
    return TerminalCommandBroker(
        plugin,
        budget=budget or SharedStepBudget(80),
        specialist_runner=runner,
        run_store=store or RecordingRunStore(),
        ask_user=ask_user,
        state=state,
        invocation_caps=invocation_caps,
        step_limits=step_limits,
    )


def spawn(
    broker: TerminalCommandBroker,
    role: str,
    *,
    request: str | None = None,
    max_steps: int = 8,
) -> CommandResult:
    return broker.run_command(
        client(
            role,
            "--request",
            request if request is not None else f"Focus the {role} review.",
            "--max-steps",
            str(max_steps),
        )
    )


def complete_roles(broker: TerminalCommandBroker) -> None:
    for role in ("binder", "critic", "verifier"):
        result = spawn(broker, role)
        if result.exit_code != 0:
            raise AssertionError(result.stderr)


class ClientRecognitionTests(unittest.TestCase):
    def test_non_control_commands_delegate_unchanged(self) -> None:
        plugin = MemoryPlugin()
        broker = make_broker(plugin)
        plugin.ordinary_results["native-command"] = CommandResult(
            0,
            stdout="native output",
            model_fields={"output": [{"type": "computer_initialize_state"}]},
            model_output="exact native output",
        )

        normal = broker.run_command(
            "printf '/opt/cuawright-tools/binder.py' | cat",
            timeout=17,
        )
        same_name = broker.run_command("python /workspace/binder.py --request /x")
        removed = broker.run_command(
            "python /opt/cuawright-tools/report.py --status ready"
        )
        native = broker.run_command("native-command")

        self.assertEqual(
            normal.stdout,
            "delegated: printf '/opt/cuawright-tools/binder.py' | cat",
        )
        self.assertEqual(
            same_name.stdout, "delegated: python /workspace/binder.py --request /x"
        )
        self.assertIn("not found", removed.stderr)
        self.assertEqual(plugin.commands[0][1], 17)
        self.assertEqual(
            native.model_fields,
            {"output": [{"type": "computer_initialize_state"}]},
        )
        self.assertEqual(native.model_output, "exact native output")

    def test_exact_client_executes_before_host_dispatch_with_inline_request(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        store = RecordingRunStore()
        broker = make_broker(plugin, runner=runner, store=store)
        request = "Inspect exact values; don't change them."

        result = spawn(broker, "binder", request=request)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(runner.invocations[0].request_content, request)
        self.assertEqual(
            runner.invocations[0].request_artifact.path,
            "specialists/phase-001/binder-01/request.txt",
        )
        self.assertTrue(
            plugin.commands[0][0].startswith(f"python {CONTROL_CLIENT_ROOT}/binder.py")
        )
        event_names = [name for name, _payload in store.events]
        self.assertLess(
            event_names.index("control_client_executed"),
            event_names.index("specialist_completed"),
        )
        self.assertEqual(plugin.control_reads, [])
        self.assertEqual(plugin.control_writes, [])

    def test_canonical_quoting_allows_punctuation_inside_inline_text(self) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        request = "Check A; compare B && C (without mutation)."

        result = spawn(make_broker(plugin, runner=runner), "binder", request=request)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(runner.invocations[0].request_content, request)

    def test_prompt_advertised_double_quoted_inline_commands_are_accepted(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        broker = make_broker(
            plugin,
            runner=runner,
            ask_user=lambda role, question: f"{role}: {question}",
        )

        binder = broker.run_command(
            "python /opt/cuawright-tools/binder.py "
            '--request "Inspect exact values; do not mutate state." '
            "--max-steps 8"
        )
        asked = broker.run_command(
            "python /opt/cuawright-tools/ask_user.py "
            '--question "Which account should I use?"'
        )

        self.assertEqual(binder.exit_code, 0)
        self.assertEqual(
            runner.invocations[0].request_content,
            "Inspect exact values; do not mutate state.",
        )
        self.assertEqual(
            json.loads(asked.stdout),
            {
                "status": "answered",
                "answer": "actor: Which account should I use?",
            },
        )

    def test_shell_wrapping_and_noncanonical_client_commands_are_rejected(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        broker = make_broker(plugin)
        base = client("submit")
        commands = [
            f"{base}; echo bad",
            f"{base} && echo bad",
            f"{base} | cat",
            f"{base} > /workspace/result",
            f"{base} $(id)",
            f"{base} `id`",
            f"{base}\nwhoami",
            f"{base} trailing",
            f'{base} "unterminated',
            f"  {base}",
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(broker.run_command(command).exit_code, 2)
        self.assertEqual(plugin.commands, [])

    def test_descriptor_validation_is_operation_typed_not_argument_equality(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        broker = make_broker(
            plugin,
            runner=runner,
            invocation_caps={"binder": 2, "critic": 2, "verifier": 3},
        )

        def substitute(_path: str, result: CommandResult) -> CommandResult:
            payload = json.loads(result.stdout)
            payload["request"] = "Trusted client descriptor request."
            payload["max_steps"] = 6
            return CommandResult(0, stdout=json.dumps(payload) + "\n")

        plugin.client_result_transform = substitute
        result = spawn(
            broker,
            "binder",
            request="Command-line request.",
            max_steps=8,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(
            runner.invocations[0].request_content,
            "Trusted client descriptor request.",
        )
        self.assertEqual(runner.invocations[0].requested_steps, 6)

    def test_composed_literal_clients_fail_before_any_execution(self) -> None:
        for name in (
            "image_show",
            "submit",
            "ask_user",
            "binder",
            "critic",
            "verifier",
        ):
            base = client(name)
            commands = [
                f"printf prepared; {base}",
                f"printf prepared && {base}",
                f"false || {base}",
                f"printf prepared | {base}",
                f"for x in a b; do {base}; done",
                f"if true; then {base}; fi",
                f"(echo prepared; {base})",
                f"echo prepared\n{base}",
                f"python - <<'PY'\nprint('prepared')\nPY\n{base}",
                f"cat <<-EOF\n\tignored\n\tEOF\n{base}",
                f"cat <<A <<'B'\nignored\nA\nignored\nB\n{base}",
                f"echo prepared; 'python' '/opt/cuawright-tools/{name}.py'",
            ]
            for command in commands:
                with self.subTest(command=command):
                    plugin = MemoryPlugin()
                    broker = make_broker(plugin)
                    result = broker.run_command(command)
                    self.assertEqual(result.exit_code, 2)
                    self.assertIn("nothing", result.stderr)
                    self.assertEqual(plugin.commands, [])
                    self.assertEqual(plugin.reads, [])
                    self.assertIsNone(broker.state.submission)

    def test_client_mentions_and_source_inspection_delegate_unchanged(self) -> None:
        base = client("image_show")
        commands = [
            f"printf '%s' '{base}'",
            f'echo ";" python /opt/cuawright-tools/image_show.py',
            "cat /opt/cuawright-tools/image_show.py",
            f"python - <<'PY'\n# unmatched source quote: '\nprint({base!r})\nPY",
            f"cat <<'EOF'\n{base}\nEOF\necho okay",
            f"cat <<-EOF\n\t{base}\n\tEOF\necho okay",
            f"printf 'multiline\n{base}\n'",
            f"printf '%s' \\\n{base}",
            f"echo okay # ; {base}",
            f"echo a\\; {base}",
            f'for x in python /opt/cuawright-tools/image_show.py; do echo "$x"; done',
            "echo okay; 'python'/opt/cuawright-tools/image_show.py",
            "echo okay; python /workspace/image_show.py",
            "printf '%s' $(printf now) python /opt/cuawright-tools/image_show.py",
            "printf '%s' $(\nprintf now\n) python /opt/cuawright-tools/image_show.py",
        ]
        for command in commands:
            with self.subTest(command=command):
                plugin = MemoryPlugin()
                broker = make_broker(plugin)
                with patch.object(
                    plugin, "run_command", return_value=CommandResult(0)
                ) as execute:
                    result = broker.run_command(command)
                execute.assert_called_once_with(command, None)
                self.assertEqual(result.exit_code, 0)
                self.assertEqual(plugin.reads, [])

    def test_ordinary_control_looking_stdout_is_not_dispatched(self) -> None:
        plugin = MemoryPlugin()
        command = "print-saved-request"
        plugin.ordinary_results[command] = CommandResult(
            0, stdout='{"cuawright_control":1,"operation":"submit"}'
        )
        broker = make_broker(plugin)
        result = broker.run_command(command)
        self.assertEqual(result.stdout, plugin.ordinary_results[command].stdout)
        self.assertFalse(result.completed)
        self.assertIsNone(broker.state.submission)

    def test_descriptor_shape_operation_and_bounds_are_still_enforced(self) -> None:
        transforms = {
            "wrong operation": {
                "cuawright_control": 1,
                "operation": "submit",
                "request": "x",
                "max_steps": 1,
            },
            "extra field": {
                "cuawright_control": 1,
                "operation": "binder",
                "request": "x",
                "max_steps": 1,
                "extra": True,
            },
            "untyped steps": {
                "cuawright_control": 1,
                "operation": "binder",
                "request": "x",
                "max_steps": "1",
            },
            "over cap": {
                "cuawright_control": 1,
                "operation": "binder",
                "request": "x",
                "max_steps": 41,
            },
        }
        for name, payload in transforms.items():
            with self.subTest(name=name):
                plugin = MemoryPlugin()
                plugin.client_result_transform = lambda _path, _result: CommandResult(
                    0,
                    stdout=json.dumps(payload) + "\n",
                )
                result = spawn(
                    make_broker(plugin, runner=CompletingRunner()),
                    "binder",
                )
                self.assertEqual(result.exit_code, 2)

        plugin = MemoryPlugin()
        plugin.client_result_transform = lambda _path, _result: CommandResult(
            0,
            stdout="x" * (MAX_CONTROL_CLIENT_STDOUT_BYTES + 1),
        )
        self.assertIn(
            "exceeds",
            spawn(make_broker(plugin, runner=CompletingRunner()), "binder").stderr,
        )


class NativeSpecialistSemanticsTests(unittest.TestCase):
    def test_critic_requires_binder_but_verifier_has_no_prerequisite(self) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        broker = make_broker(plugin, runner=runner)

        critic = spawn(broker, "critic")
        verifier = spawn(broker, "verifier")

        self.assertIn("binder must", critic.stderr)
        self.assertEqual(verifier.exit_code, 0)
        self.assertEqual(broker.state.invocation_counts["verifier"], 1)

    def test_latest_report_per_role_does_not_invalidate_downstream_reports(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        broker = make_broker(
            plugin,
            runner=runner,
            invocation_caps={"binder": 2, "critic": 2, "verifier": 3},
        )
        complete_roles(broker)

        self.assertEqual(spawn(broker, "critic").exit_code, 0)
        self.assertEqual(
            set(broker.state.accepted_reports), set(("binder", "critic", "verifier"))
        )
        verifier = broker.state.accepted_reports["verifier"]

        self.assertEqual(spawn(broker, "binder").exit_code, 0)
        self.assertEqual(
            set(broker.state.accepted_reports), set(("binder", "critic", "verifier"))
        )
        self.assertIs(broker.state.accepted_reports["verifier"], verifier)
        self.assertEqual(
            broker.state.accepted_reports["binder"].invocation_id,
            "binder-02",
        )

    def test_failed_starts_and_unreported_runs_do_not_consume_invocation_cap(
        self,
    ) -> None:
        plugin = MemoryPlugin()

        def raising_runner(
            invocation: SpecialistInvocation,
            broker: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            del invocation, broker
            raise RuntimeError("start failed")

        broker = make_broker(
            plugin,
            runner=raising_runner,
            invocation_caps={"binder": 1, "critic": 1, "verifier": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "start failed"):
            spawn(broker, "binder")
        self.assertEqual(broker.state.invocation_counts["binder"], 0)
        self.assertEqual(broker.state.invocation_allocations, [])

        def no_report_runner(
            invocation: SpecialistInvocation,
            specialist: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            del invocation, specialist
            return SpecialistRunResult(False, 0)

        broker.specialist_runner = no_report_runner
        self.assertIn("without an accepted report", spawn(broker, "binder").stderr)
        self.assertEqual(broker.state.invocation_counts["binder"], 0)

        successful = CompletingRunner()
        broker.specialist_runner = successful
        self.assertEqual(spawn(broker, "binder").exit_code, 0)
        self.assertEqual(successful.invocations[0].invocation_id, "binder-01")
        self.assertEqual(broker.state.invocation_counts["binder"], 1)
        self.assertIn("cap reached", spawn(broker, "binder").stderr)

    def test_forced_report_counts_toward_cap_and_returns_native_shape(self) -> None:
        plugin = MemoryPlugin()

        def runner(
            invocation: SpecialistInvocation,
            specialist: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            result = specialist.force_report(
                status="blocked",
                report="Binder exhausted.",
                unresolved=["No valid report was returned."],
            )
            return SpecialistRunResult(result.completed, 0, forced=True)

        broker = make_broker(
            plugin,
            runner=runner,
            invocation_caps={"binder": 1, "critic": 1, "verifier": 1},
        )
        result = spawn(broker, "binder")
        payload = json.loads(result.stdout)

        self.assertEqual(
            set(payload),
            {
                "status",
                "role",
                "allocated_steps",
                "used_steps",
                "report",
                "global_budget",
                "report_artifact",
            },
        )
        self.assertEqual(payload["status"], "forced")
        self.assertEqual(payload["report"]["status"], "blocked")
        self.assertEqual(broker.state.invocation_counts["binder"], 1)
        self.assertIn("cap reached", spawn(broker, "binder").stderr)

    def test_over_cap_max_steps_is_rejected_instead_of_clamped(self) -> None:
        for role, maximum in {"binder": 40, "critic": 32, "verifier": 24}.items():
            with self.subTest(role=role):
                plugin = MemoryPlugin()
                runner = CompletingRunner()
                broker = make_broker(plugin, runner=runner)
                if role == "critic":
                    self.assertEqual(spawn(broker, "binder").exit_code, 0)

                result = spawn(broker, role, max_steps=maximum + 1)

                self.assertEqual(result.exit_code, 2)
                self.assertFalse(result.completed)
                self.assertEqual(
                    sum(
                        command.startswith(f"python {CONTROL_CLIENT_ROOT}/{role}.py")
                        for command, _timeout in plugin.commands
                    ),
                    1,
                )
                self.assertFalse(
                    any(invocation.role == role for invocation in runner.invocations)
                )

    def test_configured_step_limits_cannot_exceed_native_role_caps(self) -> None:
        with self.assertRaisesRegex(ValueError, "native maximum 40"):
            make_broker(
                MemoryPlugin(),
                step_limits={"binder": 41, "critic": 32, "verifier": 24},
            )

    def test_requested_steps_clamp_only_to_shared_remaining_minus_one(self) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner(consume_step=True)
        budget = SharedStepBudget(8)
        broker = make_broker(plugin, runner=runner, budget=budget)

        result = spawn(broker, "binder", max_steps=40)

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(runner.invocations[0].requested_steps, 40)
        self.assertEqual(runner.invocations[0].allocated_steps, 7)
        self.assertEqual(budget.remaining_steps, 7)

    def test_one_actor_return_step_is_reserved(self) -> None:
        plugin = MemoryPlugin()
        runner = CompletingRunner()
        broker = make_broker(
            plugin,
            runner=runner,
            budget=SharedStepBudget(1),
        )

        result = spawn(broker, "binder")

        self.assertIn("one Actor return step", result.stderr)
        self.assertEqual(runner.invocations, [])

    def test_repeated_errors_remain_recoverable(self) -> None:
        plugin = MemoryPlugin()
        broker = make_broker(plugin)

        first = broker.run_command(client("submit"))
        second = broker.run_command(client("submit"))

        self.assertFalse(first.completed)
        self.assertFalse(second.completed)
        self.assertEqual(first.stderr, second.stderr)


class HostCapturedReportTests(unittest.TestCase):
    def test_report_state_and_host_artifact_are_authoritative(self) -> None:
        plugin = MemoryPlugin()
        store = RecordingRunStore()
        runner = CompletingRunner(
            bodies={"binder": "# Findings\n\nDirect evidence."},
            unresolved={"binder": ["One advisory item."]},
        )
        broker = make_broker(plugin, runner=runner, store=store)

        result = spawn(broker, "binder")
        payload = json.loads(result.stdout)
        accepted = broker.state.accepted_reports["binder"]
        artifact_data, media_type = store.artifacts[accepted.artifact.path]

        self.assertIsInstance(accepted, AcceptedReport)
        self.assertEqual(accepted.status, "complete")
        self.assertEqual(accepted.body, "# Findings\n\nDirect evidence.")
        self.assertEqual(accepted.unresolved, ("One advisory item.",))
        self.assertEqual(media_type, "application/json")
        self.assertEqual(
            json.loads(artifact_data),
            payload["report"],
        )
        self.assertEqual(payload["report_artifact"], accepted.artifact.path)
        self.assertEqual(result.model_output, result.stdout)
        self.assertEqual(plugin.control_reads, [])
        self.assertEqual(plugin.control_writes, [])

    def test_guest_report_files_cannot_change_accepted_host_state(self) -> None:
        plugin = MemoryPlugin()
        broker = make_broker(plugin, runner=CompletingRunner())
        self.assertEqual(spawn(broker, "binder").exit_code, 0)
        accepted = broker.state.accepted_reports["binder"]

        plugin.files[f"{WORKSPACE}/reports/binder-01.md"] = b"tampered"
        plugin.files[f"{WORKSPACE}/reports/binder-01.envelope.json"] = b"{}"
        critic = spawn(broker, "critic")

        self.assertEqual(critic.exit_code, 0)
        self.assertIs(broker.state.accepted_reports["binder"], accepted)
        self.assertEqual(plugin.control_reads, [])

    def test_native_report_character_and_unresolved_limits_are_preserved(
        self,
    ) -> None:
        accepted_body = "界" * MAX_REPORT_CHARACTERS
        plugin = MemoryPlugin()
        result = spawn(
            make_broker(
                plugin,
                runner=CompletingRunner(bodies={"binder": accepted_body}),
            ),
            "binder",
        )
        self.assertEqual(json.loads(result.stdout)["report"]["report"], accepted_body)
        self.assertEqual(MAX_REPORT_BYTES, MAX_REPORT_CHARACTERS * 4)

        cases = [
            {
                "status": "complete",
                "report": "x" * (MAX_REPORT_CHARACTERS + 1),
                "unresolved": [],
            },
            {
                "status": "complete",
                "report": "ok",
                "unresolved": ["x"] * (MAX_UNRESOLVED_ITEMS + 1),
            },
            {
                "status": "complete",
                "report": "ok",
                "unresolved": ["x" * (MAX_UNRESOLVED_ITEM_CHARACTERS + 1)],
            },
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                plugin = MemoryPlugin()

                def runner(
                    invocation: SpecialistInvocation,
                    specialist: TerminalCommandBroker,
                ) -> SpecialistRunResult:
                    del invocation
                    rejected = specialist.accept_text_report(json.dumps(payload))
                    self.assertEqual(rejected.exit_code, 2)
                    return SpecialistRunResult(False, 0)

                stopped = spawn(make_broker(plugin, runner=runner), "binder")
                self.assertIn("without an accepted report", stopped.stderr)

    def test_verifier_normal_report_requires_any_live_inspection(self) -> None:
        plugin = MemoryPlugin()
        attempts: list[CommandResult] = []

        def runner(
            invocation: SpecialistInvocation,
            specialist: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            del invocation
            report = json.dumps(
                {"status": "ready", "report": "Verified.", "unresolved": []}
            )
            attempts.append(specialist.accept_text_report(report))
            specialist.run_command("failed-live-check")
            accepted = specialist.accept_text_report(report)
            return SpecialistRunResult(accepted.completed, 0)

        result = spawn(make_broker(plugin, runner=runner), "verifier")

        self.assertEqual(result.exit_code, 0)
        self.assertIn("inspect the live benchmark", attempts[0].stderr)

    def test_forced_verifier_report_may_record_exhaustion_without_inspection(
        self,
    ) -> None:
        plugin = MemoryPlugin()

        def runner(
            invocation: SpecialistInvocation,
            specialist: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            del invocation
            forced = specialist.force_report(
                status="unknown",
                report="Verifier exhausted before inspection.",
                unresolved=["No inspection completed."],
            )
            return SpecialistRunResult(forced.completed, 0, forced=True)

        payload = json.loads(
            spawn(make_broker(plugin, runner=runner), "verifier").stdout
        )
        self.assertEqual(payload["status"], "forced")
        self.assertEqual(payload["report"]["status"], "unknown")


class ImageQuestionAndSubmissionTests(unittest.TestCase):
    def test_image_client_returns_normalized_typed_attachment(self) -> None:
        plugin = MemoryPlugin()
        image_path = "/workspace/reference.jpg"
        source = io.BytesIO()
        Image.new("RGB", (2, 2), (12, 34, 56)).save(source, format="JPEG")
        image = source.getvalue()
        plugin.files[image_path] = image

        result = make_broker(plugin).run_command(
            client("image_show", "--path", image_path)
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIsInstance(result.attachments[0], ImageAttachment)
        self.assertEqual(result.attachments[0].media_type, "image/png")
        self.assertTrue(result.attachments[0].data.startswith(b"\x89PNG\r\n\x1a\n"))
        with Image.open(io.BytesIO(result.attachments[0].data)) as normalized:
            self.assertEqual(normalized.size, (2, 2))
            self.assertEqual(normalized.mode, "RGB")
        self.assertIn((image_path, MAX_IMAGE_BYTES + 1), plugin.reads)

    def test_image_client_rejects_magic_only_corrupt_image(self) -> None:
        plugin = MemoryPlugin()
        image_path = "/workspace/corrupt.png"
        plugin.files[image_path] = b"\x89PNG\r\n\x1a\npixels"

        result = make_broker(plugin).run_command(
            client("image_show", "--path", image_path)
        )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("could not be decoded and normalized", result.stderr)

    def test_image_client_rejects_decompression_bomb_as_control_error(self) -> None:
        plugin = MemoryPlugin()
        image_path = "/workspace/oversized-header.png"
        source = io.BytesIO()
        Image.new("RGB", (3, 3), (12, 34, 56)).save(source, format="PNG")
        plugin.files[image_path] = source.getvalue()

        with patch(
            "cuawright.environments.control.MAX_NORMALIZED_IMAGE_PIXELS",
            1,
        ):
            result = make_broker(plugin).run_command(
                client("image_show", "--path", image_path)
            )

        self.assertEqual(result.exit_code, 2)
        self.assertIn("image dimensions exceed", result.stderr)

    def test_ask_user_passes_inline_question_without_guest_files(self) -> None:
        plugin = MemoryPlugin()
        question = "Which account should I use?"
        broker = make_broker(
            plugin,
            ask_user=lambda role, value: f"{role}: {value}",
        )

        result = broker.run_command(client("ask_user", "--question", question))

        self.assertEqual(
            json.loads(result.stdout),
            {"status": "answered", "answer": f"actor: {question}"},
        )
        self.assertEqual(plugin.control_reads, [])
        self.assertEqual(plugin.control_writes, [])

    def test_ask_user_is_unavailable_to_verifier(self) -> None:
        plugin = MemoryPlugin()
        attempts: list[CommandResult] = []

        def runner(
            invocation: SpecialistInvocation,
            specialist: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            del invocation
            attempts.append(
                specialist.run_command(
                    client("ask_user", "--question", "Can I mutate?")
                )
            )
            forced = specialist.force_report(
                status="unknown",
                report="Stopped.",
                unresolved=["Question unavailable."],
            )
            return SpecialistRunResult(forced.completed, 0, forced=True)

        spawn(
            make_broker(
                plugin,
                runner=runner,
                ask_user=lambda _role, _question: "No",
            ),
            "verifier",
        )
        self.assertIn("not available", attempts[0].stderr)

    def test_ask_user_callback_receives_specialist_role(self) -> None:
        plugin = MemoryPlugin()
        calls: list[tuple[str, str]] = []

        def ask_user(role: str, question: str) -> str:
            calls.append((role, question))
            return f"{role} answer"

        def runner(
            invocation: SpecialistInvocation,
            specialist: TerminalCommandBroker,
        ) -> SpecialistRunResult:
            asked = specialist.run_command(
                client("ask_user", "--question", "Which exact value?")
            )
            self.assertEqual(
                json.loads(asked.stdout),
                {"status": "answered", "answer": f"{invocation.role} answer"},
            )
            accepted = specialist.accept_text_report(
                json.dumps(
                    {
                        "status": "complete",
                        "report": "Binder finished.",
                        "unresolved": [],
                    }
                )
            )
            return SpecialistRunResult(accepted.completed, 0)

        result = spawn(
            make_broker(plugin, runner=runner, ask_user=ask_user),
            "binder",
        )

        self.assertEqual(result.exit_code, 0)
        self.assertEqual(calls, [("binder", "Which exact value?")])

    def test_submit_requires_current_phase_reports_and_ready_verifier(
        self,
    ) -> None:
        plugin = MemoryPlugin()
        store = RecordingRunStore()
        broker = make_broker(
            plugin,
            runner=CompletingRunner(
                statuses={
                    "binder": "blocked",
                    "critic": "concerns",
                    "verifier": "ready",
                }
            ),
            store=store,
            state=TerminalControlState(session_id="s", phase_id="p"),
        )
        complete_roles(broker)

        submitted = broker.run_command(client("submit"))
        payload = json.loads(submitted.stdout)
        provenance = json.loads(store.artifacts[payload["artifact_path"]][0])

        self.assertTrue(submitted.completed)
        self.assertFalse(payload["forced"])
        self.assertEqual(set(provenance["reports"]), {"binder", "critic", "verifier"})
        self.assertTrue(
            all(
                set(item)
                == {
                    "invocation_id",
                    "status",
                    "report_artifact",
                    "report_sha256",
                }
                for item in provenance["reports"].values()
            )
        )

        broker.state.reset_phase("phase-2")
        missing = broker.run_command(client("submit"))
        self.assertEqual(missing.exit_code, 2)
        self.assertIn("binder, critic, verifier", missing.stderr)

        complete_roles(broker)
        broker.state.accepted_reports["verifier"] = replace(
            broker.state.accepted_reports["verifier"],
            status="unknown",
        )
        unknown = broker.run_command(client("submit"))
        self.assertTrue(unknown.completed)

        broker.state.reset_phase("phase-3")
        complete_roles(broker)
        broker.state.accepted_reports["verifier"] = replace(
            broker.state.accepted_reports["verifier"],
            status="repair",
        )
        repair = broker.run_command(client("submit"))
        self.assertEqual(repair.exit_code, 2)
        self.assertIn("latest verifier status is 'repair'", repair.stderr)

    def test_forced_submission_accepts_host_captured_text(self) -> None:
        plugin = MemoryPlugin()
        store = RecordingRunStore()
        broker = make_broker(plugin, store=store)
        report = '{"status":"forced","reason":"budget exhausted"}'

        result = broker.force_submission(report)
        payload = json.loads(result.stdout)
        provenance = json.loads(store.artifacts[payload["artifact_path"]][0])

        self.assertTrue(result.completed)
        self.assertTrue(payload["forced"])
        self.assertEqual(payload["missing_reports"], ["binder", "critic", "verifier"])
        self.assertEqual(provenance["forced_actor_report"]["text"], report)
        self.assertEqual(plugin.control_reads, [])
        self.assertEqual(plugin.control_writes, [])


if __name__ == "__main__":
    unittest.main()

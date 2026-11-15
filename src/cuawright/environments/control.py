from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import warnings
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal, Mapping, Protocol, Sequence

from PIL import Image, ImageOps, UnidentifiedImageError

from cuawright.artifacts import ArtifactRef, NullRunStore, RunStore
from cuawright.budget import ScopedStepBudget, StepBudget
from cuawright.environments.base import (
    BenchmarkPlugin,
    CommandObservation,
    CommandObserver,
)
from cuawright.runtime.contracts import (
    CommandResult,
    ImageAttachment,
)


SpecialistRole = Literal["binder", "critic", "verifier"]

SPECIALIST_ROLES: tuple[SpecialistRole, ...] = (
    "binder",
    "critic",
    "verifier",
)
ROLE_STATUSES: Mapping[SpecialistRole, frozenset[str]] = {
    "binder": frozenset({"complete", "blocked"}),
    "critic": frozenset({"accept", "concerns", "blocked"}),
    "verifier": frozenset({"ready", "repair", "unknown"}),
}
DEFAULT_INVOCATION_CAPS: Mapping[SpecialistRole, int] = {
    "binder": 1,
    "critic": 2,
    "verifier": 3,
}
DEFAULT_STEP_LIMITS: Mapping[SpecialistRole, int] = {
    "binder": 40,
    "critic": 32,
    "verifier": 24,
}

DEFAULT_AGENT_WORKSPACE = "/workspace/.agent"
CONTROL_CLIENT_ROOT = "/opt/cuawright-tools"
CONTROL_CLIENT_PYTHON = "python"
MAX_CONTROL_CLIENT_STDOUT_BYTES = 16 * 1024
MAX_REQUEST_CHARACTERS = 2_000
MAX_SUBMISSION_BYTES = 256 * 1024
MAX_QUESTION_CHARACTERS = 2_000
MAX_ANSWER_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_NORMALIZED_IMAGE_PIXELS = 40_000_000
MAX_REPORT_CHARACTERS = 12_000
MAX_UNRESOLVED_ITEMS = 12
MAX_UNRESOLVED_ITEM_CHARACTERS = 1_000
MAX_JSON_ESCAPED_CHARACTER_BYTES = 12
MAX_REPORT_BYTES = MAX_REPORT_CHARACTERS * 4
MAX_TEXT_REPORT_BYTES = (
    MAX_REPORT_CHARACTERS + MAX_UNRESOLVED_ITEMS * MAX_UNRESOLVED_ITEM_CHARACTERS
) * MAX_JSON_ESCAPED_CHARACTER_BYTES + 4_096

_CLIENT_OPERATIONS: Mapping[str, str] = {
    f"{CONTROL_CLIENT_ROOT}/binder.py": "binder",
    f"{CONTROL_CLIENT_ROOT}/critic.py": "critic",
    f"{CONTROL_CLIENT_ROOT}/verifier.py": "verifier",
    f"{CONTROL_CLIENT_ROOT}/image_show.py": "image_show",
    f"{CONTROL_CLIENT_ROOT}/ask_user.py": "ask_user",
    f"{CONTROL_CLIENT_ROOT}/submit.py": "submit",
}
_SHELL_OPERATOR_CHARACTERS = frozenset(";&|<>()`")
_IMAGE_SUFFIXES: Mapping[str, frozenset[str]] = {
    "image/png": frozenset({".png"}),
    "image/jpeg": frozenset({".jpg", ".jpeg"}),
    "image/gif": frozenset({".gif"}),
    "image/webp": frozenset({".webp"}),
}


class TerminalControlError(ValueError):
    pass


class StatePersistenceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SpecialistInvocation:
    invocation_id: str
    role: SpecialistRole
    request_content: str
    request_artifact: ArtifactRef
    invocation_number: int
    requested_steps: int
    allocated_steps: int
    budget_source: Literal["model"]
    session_id: str
    phase_id: str
    workspace: str
    budget: ScopedStepBudget


@dataclass(frozen=True, slots=True)
class SpecialistRunResult:
    completed: bool
    steps_used: int
    forced: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a boolean")
        if isinstance(self.steps_used, bool) or not isinstance(self.steps_used, int):
            raise TypeError("steps_used must be an integer")
        if self.steps_used < 0:
            raise ValueError("steps_used cannot be negative")
        if not isinstance(self.forced, bool):
            raise TypeError("forced must be a boolean")


class SpecialistRunner(Protocol):
    def __call__(
        self,
        invocation: SpecialistInvocation,
        broker: TerminalCommandBroker,
    ) -> SpecialistRunResult:
        """Run one fresh specialist synchronously using broker as its plugin."""


class AskUserCallback(Protocol):
    def __call__(self, role: str, question: str) -> str:
        """Return the user's answer without exposing callback implementation state."""


@dataclass(frozen=True, slots=True)
class AcceptedReport:
    invocation_id: str
    role: SpecialistRole
    status: str
    body: str
    unresolved: tuple[str, ...]
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True)
class InvocationAllocation:
    invocation_id: str
    role: SpecialistRole
    request_content: str
    request_artifact: ArtifactRef
    requested_steps: int
    allocated_steps: int
    budget_source: Literal["model"]


@dataclass(slots=True)
class _ActiveInvocation:
    invocation_id: str
    role: SpecialistRole
    request_content: str
    request_artifact: ArtifactRef
    requested_steps: int
    allocated_steps: int
    budget_source: Literal["model"]
    previous_reports: dict[SpecialistRole, AcceptedReport]
    accepted_report: AcceptedReport | None = None
    inspected: bool = False


@dataclass(slots=True)
class TerminalControlState:
    session_id: str = "session"
    phase_id: str = "phase-1"
    invocation_counts: dict[SpecialistRole, int] = field(
        default_factory=lambda: {role: 0 for role in SPECIALIST_ROLES}
    )
    accepted_reports: dict[SpecialistRole, AcceptedReport] = field(default_factory=dict)
    invocation_allocations: list[InvocationAllocation] = field(default_factory=list)
    submission: ArtifactRef | None = None
    submission_forced: bool = False
    submission_missing_reports: tuple[SpecialistRole, ...] = ()
    submission_non_ready_reasons: tuple[str, ...] = ()
    phase_generation: int = 1
    _active: dict[str, _ActiveInvocation] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.phase_id.strip():
            raise ValueError("session_id and phase_id must be non-empty")
        if set(self.invocation_counts) != set(SPECIALIST_ROLES):
            raise ValueError("invocation_counts must contain every specialist role")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.invocation_counts.values()
        ):
            raise ValueError("invocation counts must be non-negative integers")

    @property
    def completed_roles(self) -> tuple[SpecialistRole, ...]:
        return tuple(role for role in SPECIALIST_ROLES if role in self.accepted_reports)

    def reset_phase(self, phase_id: str) -> None:
        if self._active:
            raise RuntimeError("cannot reset phase during a specialist invocation")
        if not isinstance(phase_id, str) or not phase_id.strip():
            raise ValueError("phase_id must be a non-empty string")
        self.phase_id = phase_id
        self.phase_generation += 1
        self.invocation_counts = {role: 0 for role in SPECIALIST_ROLES}
        self.accepted_reports.clear()
        self.invocation_allocations.clear()
        self.submission = None
        self.submission_forced = False
        self.submission_missing_reports = ()
        self.submission_non_ready_reasons = ()


class TerminalCommandBroker:
    """Trusted dispatcher for the provisioned terminal control clients."""

    def __init__(
        self,
        plugin: BenchmarkPlugin,
        *,
        budget: StepBudget,
        role: str = "actor",
        run_store: RunStore | None = None,
        specialist_runner: SpecialistRunner | None = None,
        ask_user: AskUserCallback | None = None,
        state: TerminalControlState | None = None,
        workspace: str = DEFAULT_AGENT_WORKSPACE,
        invocation_caps: Mapping[SpecialistRole, int] | None = None,
        step_limits: Mapping[SpecialistRole, int] | None = None,
        enabled_specialist_roles: Sequence[SpecialistRole] = SPECIALIST_ROLES,
        invocation_id: str | None = None,
        stage: str = "orchestrate",
        command_observer: CommandObserver | None = None,
    ) -> None:
        if role not in {"actor", *SPECIALIST_ROLES}:
            raise ValueError(f"unsupported agent role: {role!r}")
        self.plugin = plugin
        self.budget = budget
        self.role = role
        self.run_store = run_store or NullRunStore()
        self.specialist_runner = specialist_runner
        self.ask_user_callback = ask_user
        self.state = state or TerminalControlState()
        self.workspace = _canonical_absolute_path(workspace)
        self.invocation_caps = _validated_role_limits(
            invocation_caps or DEFAULT_INVOCATION_CAPS,
            "invocation cap",
        )
        self.step_limits = _validated_role_limits(
            step_limits or DEFAULT_STEP_LIMITS,
            "step limit",
        )
        if len(enabled_specialist_roles) != len(set(enabled_specialist_roles)) or any(
            role not in SPECIALIST_ROLES for role in enabled_specialist_roles
        ):
            raise ValueError("enabled_specialist_roles contains an invalid role")
        self.enabled_specialist_roles = tuple(enabled_specialist_roles)
        for specialist_role, maximum in DEFAULT_STEP_LIMITS.items():
            if self.step_limits[specialist_role] > maximum:
                raise ValueError(
                    f"{specialist_role} step limit cannot exceed native maximum "
                    f"{maximum}"
                )
        self.invocation_id = invocation_id
        self.stage = stage
        self.command_observer = command_observer
        if self.role == "actor" and invocation_id is not None:
            raise ValueError("actor broker cannot have a specialist invocation_id")
        if self.role != "actor" and invocation_id is None:
            raise ValueError("specialist broker requires an invocation_id")

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        argv = _shell_words(command)
        script_path = (
            argv[1]
            if len(argv) >= 2
            and argv[0] == CONTROL_CLIENT_PYTHON
            and argv[1] in _CLIENT_OPERATIONS
            else None
        )
        if script_path is None:
            if _looks_like_control_client(command) or _contains_literal_control_command(
                command
            ):
                return self._control_error_result(
                    "unsupported control client command syntax; nothing in this "
                    "run_command call was executed. Run ordinary preparation "
                    "separately, then run the control client as the only command "
                    "in its own run_command call"
                )
            if argv and argv[0] == "agent":
                self.run_store.event(
                    "legacy_control_rejected",
                    {
                        "role": self.role,
                        "command": command,
                        "reason": "legacy_agent_protocol_removed",
                    },
                )
                return self._control_error_result(
                    "the legacy agent command protocol has been removed"
                )
            result = self.plugin.run_command(command, timeout)
            if self.role == "verifier":
                self._mark_inspected()
            delegated = CommandResult(
                result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                attachments=result.attachments,
                completed=False,
                inspected=result.inspected,
                model_fields=result.model_fields,
                model_output=result.model_output,
            )
            self._observe(command, delegated)
            return delegated

        if command != command.strip() or _has_shell_operator_syntax(command):
            return self._control_error_result(
                "control clients must be exact standalone commands without "
                "shell operators; nothing in this run_command call was executed"
            )

        operation = _CLIENT_OPERATIONS[script_path]
        try:
            _parse_control_client_arguments(operation, argv[2:])
        except TerminalControlError as exc:
            return self._control_error_result(str(exc))

        executed = self.plugin.run_command(command, timeout)
        self.run_store.event(
            "control_client_executed",
            {
                "role": self.role,
                "command": command,
                "python_executable": CONTROL_CLIENT_PYTHON,
                "script_path": script_path,
                "operation": operation,
                "exit_code": executed.exit_code,
                "stdout_size_bytes": len(executed.stdout.encode("utf-8")),
                "stderr_size_bytes": len(executed.stderr.encode("utf-8")),
            },
        )
        if executed.exit_code != 0:
            result = self._control_client_failure_result(operation, executed)
            self._observe(command, result, operation=operation)
            return result
        observed_before_dispatch = operation in SPECIALIST_ROLES
        if observed_before_dispatch:
            self._observe(command, executed, operation=operation)
        try:
            request = _validate_control_client_result(
                executed,
                expected_operation=operation,
                step_limits=self.step_limits,
            )
            result = self._dispatch_client(request)
        except TerminalControlError as exc:
            result = self._control_error_result(str(exc))
        if not observed_before_dispatch:
            self._observe(command, result, operation=operation)
        accepted = None
        if result.exit_code == 0 and operation in SPECIALIST_ROLES:
            accepted = self.state.accepted_reports.get(_specialist_role(operation))
        self.run_store.event(
            "control_client_dispatched",
            {
                "role": self.role,
                "command": command,
                "script_path": script_path,
                "operation": operation,
                "exit_code": result.exit_code,
                "completed": result.completed,
                "attachment_count": len(result.attachments),
                "direct_report_returned": accepted is not None,
                "report_artifact": (
                    None if accepted is None else accepted.artifact.path
                ),
            },
        )
        return result

    def _dispatch_client(self, request: Mapping[str, object]) -> CommandResult:
        operation = request["operation"]
        if operation in SPECIALIST_ROLES:
            return self._spawn(
                _specialist_role(operation),
                str(request["request"]),
                int(request["max_steps"]),
            )
        if operation == "image_show":
            return self._image_show(str(request["path"]))
        if operation == "ask_user":
            return self._ask_user(str(request["question"]))
        if operation == "submit":
            return self._submit()
        raise TerminalControlError("unsupported control client operation")

    def _spawn(
        self,
        role: SpecialistRole,
        request_content: str,
        requested_steps: int,
    ) -> CommandResult:
        if self.role != "actor":
            raise TerminalControlError(
                "role invocation clients are valid only in actor role"
            )
        if role not in self.enabled_specialist_roles:
            raise TerminalControlError(
                f"{role} specialist invocation is disabled for this run"
            )
        if self.specialist_runner is None:
            raise TerminalControlError("no specialist runner is configured")
        if self.state.submission is not None:
            raise TerminalControlError("the current phase is already submitted")
        self._validate_dependency(role)
        count = self.state.invocation_counts[role]
        cap = self.invocation_caps[role]
        if count >= cap:
            raise TerminalControlError(f"{role} invocation cap reached ({count}/{cap})")
        if self.budget.remaining_steps < 2:
            raise TerminalControlError(
                "insufficient shared budget: one Actor return step must remain"
            )
        if requested_steps > self.step_limits[role]:
            raise TerminalControlError(
                f"{role} max-steps exceeds the {self.step_limits[role]}-step limit"
            )

        invocation_number = count + 1
        invocation_id = f"{role}-{invocation_number:02d}"
        allocated_steps = min(requested_steps, self.budget.remaining_steps - 1)
        scoped_budget = ScopedStepBudget(self.budget, allocated_steps)
        artifact_root = self._specialist_artifact_root(invocation_id)
        request_data = request_content.encode("utf-8")
        request_artifact = self._write_artifact(
            f"{artifact_root}/request.txt",
            request_data,
            media_type="text/plain",
        )
        active = _ActiveInvocation(
            invocation_id=invocation_id,
            role=role,
            request_content=request_content,
            request_artifact=request_artifact,
            requested_steps=requested_steps,
            allocated_steps=allocated_steps,
            budget_source="model",
            previous_reports=dict(self.state.accepted_reports),
        )
        allocation = InvocationAllocation(
            invocation_id=invocation_id,
            role=role,
            request_content=request_content,
            request_artifact=request_artifact,
            requested_steps=requested_steps,
            allocated_steps=allocated_steps,
            budget_source="model",
        )
        self.state._active[invocation_id] = active

        invocation = SpecialistInvocation(
            invocation_id=invocation_id,
            role=role,
            request_content=request_content,
            request_artifact=request_artifact,
            invocation_number=invocation_number,
            requested_steps=requested_steps,
            allocated_steps=allocated_steps,
            budget_source="model",
            session_id=self.state.session_id,
            phase_id=self.state.phase_id,
            workspace=self.workspace,
            budget=scoped_budget,
        )
        specialist_broker = TerminalCommandBroker(
            self.plugin,
            budget=scoped_budget,
            role=role,
            run_store=self.run_store,
            specialist_runner=self.specialist_runner,
            ask_user=self.ask_user_callback,
            state=self.state,
            workspace=self.workspace,
            invocation_caps=self.invocation_caps,
            step_limits=self.step_limits,
            enabled_specialist_roles=self.enabled_specialist_roles,
            invocation_id=invocation_id,
            stage=f"specialist-{invocation_id}",
            command_observer=self.command_observer,
        )
        starting_steps = self.budget.used_steps
        try:
            run_result = self.specialist_runner(invocation, specialist_broker)
        except StatePersistenceError:
            self._fail_invocation(active)
            raise
        except Exception:
            self._fail_invocation(active)
            raise

        actual_global_steps = self.budget.used_steps - starting_steps
        actual_steps = scoped_budget.consumed_steps
        if actual_global_steps != actual_steps:
            self._fail_invocation(active)
            raise RuntimeError("specialist runner consumed outside its scoped budget")
        if not isinstance(run_result, SpecialistRunResult):
            self._fail_invocation(active)
            raise TypeError("specialist runner must return SpecialistRunResult")
        if run_result.steps_used != actual_steps:
            self._fail_invocation(active)
            raise RuntimeError(
                "specialist runner step usage does not match the shared budget"
            )
        if not run_result.completed or active.accepted_report is None:
            self._fail_invocation(active)
            return self._control_error_result(
                f"{role} specialist stopped without an accepted report"
            )
        report = active.accepted_report
        event_recorded = False
        try:
            self.run_store.event(
                "specialist_completed",
                {
                    "role": role,
                    "invocation_id": invocation_id,
                    "status": report.status,
                    "steps_used": actual_steps,
                    "requested_steps": requested_steps,
                    "allocated_steps": allocated_steps,
                    "budget_source": "model",
                    "budget": self.budget.shared_snapshot(),
                    "request_artifact": request_artifact.path,
                    "report_artifact": report.artifact.path,
                    "report_sha256": report.artifact.sha256,
                    "report_size_bytes": report.artifact.size_bytes,
                    "forced": run_result.forced,
                },
            )
            event_recorded = True
        finally:
            if not event_recorded:
                self._fail_invocation(active)
        self._finalize_invocation(active, allocation)
        payload = {
            "status": "forced" if run_result.forced else "success",
            "role": role,
            "allocated_steps": allocated_steps,
            "used_steps": actual_steps,
            "report": {
                "status": report.status,
                "report": report.body,
                "unresolved": list(report.unresolved),
            },
            "global_budget": self.budget.shared_snapshot(),
            "report_artifact": report.artifact.path,
        }
        model_output = _json_text(payload)
        return CommandResult(
            0,
            stdout=model_output,
            model_output=model_output,
        )

    def accept_text_report(self, text: str) -> CommandResult:
        """Accept one native-shaped specialist report captured by the host."""

        try:
            encoded = text.encode("utf-8")
            if len(encoded) > MAX_TEXT_REPORT_BYTES:
                raise TerminalControlError(
                    "specialist report response exceeds the native report contract"
                )
            payload = _json_object(encoded, "specialist report response")
            if set(payload) != {"status", "report", "unresolved"}:
                raise TerminalControlError(
                    "specialist report response must contain exactly: "
                    "status, report, unresolved"
                )
            return self._accept_report(
                status=payload["status"],
                body=payload["report"],
                unresolved=payload["unresolved"],
                trusted_forced=False,
            )
        except TerminalControlError as exc:
            return _error_result(str(exc))

    def force_report(
        self,
        *,
        status: str,
        report: str,
        unresolved: list[str] | tuple[str, ...],
    ) -> CommandResult:
        """Accept one host-generated exhaustion report for the active specialist."""

        return self._accept_report(
            status=status,
            body=report,
            unresolved=unresolved,
            trusted_forced=True,
        )

    def _accept_report(
        self,
        *,
        status: object,
        body: object,
        unresolved: object,
        trusted_forced: bool,
    ) -> CommandResult:
        if self.role == "actor":
            raise TerminalControlError(
                "specialist reports are valid only in a specialist role"
            )
        role = _specialist_role(self.role)
        if self.invocation_id is None:
            raise TerminalControlError("specialist invocation identity is missing")
        active = self.state._active.get(self.invocation_id)
        if active is None or active.role != role:
            raise TerminalControlError(
                "specialist invocation is not active in the current phase"
            )
        if active.accepted_report is not None:
            raise TerminalControlError(
                "this specialist invocation already has an accepted report"
            )
        if role == "verifier" and not active.inspected and not trusted_forced:
            raise TerminalControlError(
                "inspect the live benchmark with an ordinary command or "
                "image_show.py before returning a verifier report"
            )
        if not isinstance(status, str):
            raise TerminalControlError(
                "specialist report response status must be a string"
            )
        forced_status = "unknown" if role == "verifier" else "blocked"
        if trusted_forced and status != forced_status:
            raise TerminalControlError(
                f"trusted forced {role} report status must be {forced_status!r}"
            )
        if status not in ROLE_STATUSES[role]:
            allowed = ", ".join(sorted(ROLE_STATUSES[role]))
            raise TerminalControlError(
                f"invalid {role} status {status!r}; expected one of: {allowed}"
            )
        if not isinstance(body, str) or not body.strip():
            raise TerminalControlError(
                "specialist report response report must be non-empty text"
            )
        if len(body) > MAX_REPORT_CHARACTERS:
            raise TerminalControlError(
                f"specialist report exceeds {MAX_REPORT_CHARACTERS} characters"
            )
        _validate_unresolved(unresolved, "specialist report response")
        unresolved_items = tuple(unresolved)
        artifact_data = _json_text(
            {
                "status": status,
                "report": body,
                "unresolved": list(unresolved_items),
            }
        ).encode("utf-8")
        artifact = self._write_artifact(
            f"{self._specialist_artifact_root(self.invocation_id)}/report.json",
            artifact_data,
            media_type="application/json",
        )
        accepted_report = AcceptedReport(
            invocation_id=self.invocation_id,
            role=role,
            status=status,
            body=body,
            unresolved=unresolved_items,
            artifact=artifact,
        )

        def apply_report() -> None:
            active.accepted_report = accepted_report
            accepted = dict(active.previous_reports)
            accepted[role] = accepted_report
            self.state.accepted_reports = accepted

        def rollback_report() -> None:
            active.accepted_report = None
            self.state.accepted_reports = dict(active.previous_reports)

        apply_report()
        event_recorded = False
        try:
            self.run_store.event(
                "specialist_report_accepted",
                {
                    "role": role,
                    "invocation_id": self.invocation_id,
                    "status": status,
                    "report_artifact": artifact.path,
                    "report_sha256": artifact.sha256,
                    "report_size_bytes": artifact.size_bytes,
                    "requested_steps": active.requested_steps,
                    "allocated_steps": active.allocated_steps,
                    "budget_source": active.budget_source,
                    "forced": trusted_forced,
                },
            )
            event_recorded = True
        finally:
            if not event_recorded:
                rollback_report()
        if not event_recorded:
            raise StatePersistenceError("could not record accepted specialist report")
        return _json_result(
            {
                "accepted": True,
                "status": status,
                "report_artifact": artifact.path,
                "sha256": artifact.sha256,
                "requested_steps": active.requested_steps,
                "allocated_steps": active.allocated_steps,
            },
            completed=True,
        )

    def _image_show(self, path_value: str) -> CommandResult:
        path = _canonical_absolute_path(path_value)
        source_data = self._read_image_file(path, MAX_IMAGE_BYTES)
        _image_media_type(source_data)
        data = _normalized_png(source_data)
        if len(data) > MAX_IMAGE_BYTES:
            raise TerminalControlError(
                f"normalized image exceeds the {MAX_IMAGE_BYTES}-byte limit"
            )
        media_type = "image/png"
        digest = hashlib.sha256(data).hexdigest()
        extension = next(iter(sorted(_IMAGE_SUFFIXES[media_type])))
        artifact = self._write_artifact(
            f"images/{digest}{extension}",
            data,
            media_type=media_type,
        )
        self.run_store.event(
            "image_attached",
            {
                "source_path": path,
                "media_type": media_type,
                "size_bytes": len(data),
                "sha256": artifact.sha256,
            },
        )
        self._mark_inspected()
        return CommandResult(
            0,
            stdout=_json_text(
                {
                    "source_path": path,
                    "media_type": media_type,
                    "size_bytes": len(data),
                    "sha256": artifact.sha256,
                }
            ),
            attachments=(ImageAttachment(data, media_type, path),),
        )

    def _ask_user(self, question: str) -> CommandResult:
        if self.role == "verifier":
            raise TerminalControlError(
                "ask_user.py is not available to the read-only Verifier"
            )
        if self.ask_user_callback is None:
            raise TerminalControlError("no ask-user callback is configured")
        question = _bounded_inline_text(
            question,
            "question",
            MAX_QUESTION_CHARACTERS,
            allow_empty=False,
        )
        answer = self.ask_user_callback(self.role, question)
        if not isinstance(answer, str):
            raise TerminalControlError("ask-user callback must return a string")
        encoded_answer = answer.encode("utf-8")
        if len(encoded_answer) > MAX_ANSWER_BYTES:
            raise TerminalControlError(
                f"ask-user answer exceeds {MAX_ANSWER_BYTES} bytes"
            )
        question_data = question.encode("utf-8")
        self.run_store.event(
            "user_question_answered",
            {
                "question_sha256": hashlib.sha256(question_data).hexdigest(),
                "question_size_bytes": len(question_data),
                "answer_size_bytes": len(encoded_answer),
            },
        )
        return _json_result(
            {
                "status": "answered",
                "answer": answer,
            }
        )

    def _submit(
        self,
        *,
        trusted_forced: bool = False,
        forced_actor_report: str | None = None,
    ) -> CommandResult:
        if self.role != "actor":
            raise TerminalControlError("submit.py is valid only in actor role")
        if self.state.submission is not None:
            raise TerminalControlError("the current phase is already submitted")
        missing = [
            role
            for role in self.enabled_specialist_roles
            if role not in self.state.accepted_reports
        ]
        if missing and not trusted_forced and self.budget.remaining_steps == 0:
            trusted_forced = True
        non_ready_reasons = self._submission_non_ready_reasons(missing)
        if non_ready_reasons and not trusted_forced:
            raise TerminalControlError(
                "submission is not ready: " + "; ".join(non_ready_reasons)
            )

        forced_report: dict[str, object] | None = None
        if forced_actor_report is not None:
            if (
                not isinstance(forced_actor_report, str)
                or not forced_actor_report.strip()
            ):
                raise TerminalControlError(
                    "forced submission report must be non-empty text"
                )
            forced_data = forced_actor_report.encode("utf-8")
            if len(forced_data) > MAX_SUBMISSION_BYTES:
                raise TerminalControlError(
                    "forced submission report exceeds the "
                    f"{MAX_SUBMISSION_BYTES}-byte limit"
                )
            forced_report = {
                "text": forced_actor_report,
                "sha256": hashlib.sha256(forced_data).hexdigest(),
                "size_bytes": len(forced_data),
            }

        provenance: dict[str, object] = {
            "schema_version": 1,
            "session_id": self.state.session_id,
            "phase_id": self.state.phase_id,
            "phase_generation": self.state.phase_generation,
            "forced": trusted_forced,
            "enabled_specialist_roles": list(self.enabled_specialist_roles),
            "missing_reports": missing,
            "non_ready_reasons": non_ready_reasons,
            "reports": {
                role: {
                    "invocation_id": report.invocation_id,
                    "status": report.status,
                    "report_artifact": report.artifact.path,
                    "report_sha256": report.artifact.sha256,
                }
                for role, report in self.state.accepted_reports.items()
            },
        }
        if forced_report is not None:
            provenance["forced_actor_report"] = forced_report
        artifact = self._write_artifact(
            f"submissions/phase-{self.state.phase_generation:03d}/submission.json",
            _json_text(provenance).encode("utf-8"),
            media_type="application/json",
        )

        def apply_submission() -> None:
            self.state.submission = artifact
            self.state.submission_forced = trusted_forced
            self.state.submission_missing_reports = tuple(missing)
            self.state.submission_non_ready_reasons = tuple(non_ready_reasons)

        def rollback_submission() -> None:
            self.state.submission = None
            self.state.submission_forced = False
            self.state.submission_missing_reports = ()
            self.state.submission_non_ready_reasons = ()

        apply_submission()
        event_recorded = False
        try:
            self.run_store.event(
                "submission_accepted",
                {
                    "session_id": self.state.session_id,
                    "phase_id": self.state.phase_id,
                    "artifact_path": artifact.path,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                    "forced": trusted_forced,
                    "missing_reports": missing,
                    "non_ready_reasons": non_ready_reasons,
                },
            )
            event_recorded = True
        finally:
            if not event_recorded:
                rollback_submission()
        if not event_recorded:
            raise StatePersistenceError("could not record submission")
        return _json_result(
            {
                "accepted": True,
                "artifact_path": artifact.path,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "forced": trusted_forced,
                "missing_reports": missing,
                "non_ready_reasons": non_ready_reasons,
            },
            completed=True,
        )

    def force_submission(self, report: str) -> CommandResult:
        """Record host-captured Actor output after shared-budget exhaustion."""

        result = self._submit(
            trusted_forced=True,
            forced_actor_report=report,
        )
        self._observe(
            "[host forced submission]",
            result,
            operation="forced_submit",
            synthetic=True,
        )
        return result

    def _submission_non_ready_reasons(
        self,
        missing: list[SpecialistRole],
    ) -> list[str]:
        reasons: list[str] = []
        if missing:
            reasons.append("missing accepted reports for: " + ", ".join(missing))
        if "verifier" in self.enabled_specialist_roles:
            verifier = self.state.accepted_reports.get("verifier")
            if verifier is not None and verifier.status == "repair":
                reasons.append(
                    "latest verifier status is 'repair'; fix the directly observed "
                    "failure, save and reload the canonical state, then run a "
                    "fresh Verifier"
                )
        return reasons

    def _validate_dependency(self, role: SpecialistRole) -> None:
        if role == "critic" and "binder" not in self.state.accepted_reports:
            raise TerminalControlError(
                "binder must have an accepted report before critic"
            )

    def _read_image_file(self, path: str, limit: int) -> bytes:
        try:
            data = self.plugin.read_file(path, max_bytes=limit + 1)
        except OSError as exc:
            raise TerminalControlError(
                f"could not read image ({type(exc).__name__})"
            ) from None
        if not isinstance(data, bytes):
            raise TypeError("benchmark file bridge must return bytes")
        if len(data) > limit:
            raise TerminalControlError(f"image exceeds the {limit}-byte limit")
        return data

    def _write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str,
    ) -> ArtifactRef:
        return self.run_store.write_artifact(
            path,
            data,
            media_type=media_type,
        )

    def _finalize_invocation(
        self,
        active: _ActiveInvocation,
        allocation: InvocationAllocation,
    ) -> None:
        self.state.invocation_counts[active.role] += 1
        self.state.invocation_allocations.append(allocation)
        self.state._active.pop(active.invocation_id, None)

    def _control_error_result(self, message: str) -> CommandResult:
        return _error_result(message)

    def _control_client_failure_result(
        self,
        operation: str,
        result: CommandResult,
    ) -> CommandResult:
        del operation
        return CommandResult(
            result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            attachments=result.attachments,
            completed=False,
            inspected=result.inspected,
            model_fields=result.model_fields,
            model_output=result.model_output,
        )

    def _mark_inspected(self) -> None:
        if self.role != "verifier" or self.invocation_id is None:
            return
        active = self.state._active.get(self.invocation_id)
        if active is not None:
            active.inspected = True

    def _specialist_artifact_root(self, invocation_id: str) -> str:
        return (
            f"specialists/phase-{self.state.phase_generation:03d}/" f"{invocation_id}"
        )

    def _fail_invocation(self, active: _ActiveInvocation) -> None:
        active.accepted_report = None
        self.state.accepted_reports = dict(active.previous_reports)
        self.state._active.pop(active.invocation_id, None)

    def _observe(
        self,
        command: str,
        result: CommandResult,
        *,
        operation: str | None = None,
        synthetic: bool = False,
    ) -> None:
        if self.command_observer is None:
            return
        submission_forced = (
            self.state.submission_forced
            if result.completed and operation in {"submit", "forced_submit"}
            else None
        )
        self.command_observer(
            CommandObservation(
                role=self.role,
                stage=self.stage,
                command=command,
                result=result,
                operation=operation,
                synthetic=synthetic,
                submission_forced=submission_forced,
            )
        )


def _shell_words(command: str) -> list[str]:
    if not isinstance(command, str):
        return []
    try:
        return shlex.split(command, comments=False, posix=True)
    except ValueError:
        stripped = command.lstrip()
        if re.match(r"agent(?:\s|$)", stripped):
            return ["agent", "__invalid_shell_syntax__"]
        return []


def _looks_like_control_client(command: object) -> bool:
    if not isinstance(command, str):
        return False
    names = "|".join(re.escape(PurePosixPath(path).name) for path in _CLIENT_OPERATIONS)
    return (
        re.match(
            rf"^\s*{re.escape(CONTROL_CLIENT_PYTHON)}\s+"
            rf"[\"']?{re.escape(CONTROL_CLIENT_ROOT)}/(?:{names})"
            rf"(?:[\"']|\s|[;&|<>`()])",
            command,
        )
        is not None
    )


_LITERAL_SHELL_TOKEN = re.compile(
    r"""[ \t\r]+|\#[^\n]*$|[;&|()<>]+|"""
    r"""(?:\\[^\n]|"(?:\\.|[^"\\])*"|'[^']*'|[^\s\\'";&|()<>])+"""
)


def _contains_literal_control_command(command: str) -> bool:
    """Recognize literal command positions, not arbitrary dynamic Bash execution.

    Quoted words and here-document bodies are data. This deliberately does not
    expand variables, aliases, substitutions, eval, or shell/script arguments.
    """
    if not isinstance(command, str):
        return False
    here_documents: list[tuple[str, bool]] = []
    pending = ""
    command_start = True
    python_command = False
    for line in command.replace("\\\n", "").splitlines(keepends=True):
        if here_documents:
            delimiter, strip_tabs = here_documents[0]
            candidate = line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                here_documents.pop(0)
            continue
        pending += line
        text = pending.rstrip("\r\n")
        tokens = []
        position = 0
        while position < len(text):
            match = _LITERAL_SHELL_TOKEN.match(text, position)
            if match is None:
                break
            raw = match.group()
            position = match.end()
            if raw.isspace():
                continue
            if raw.startswith("#"):
                break
            tokens.append(raw)
        else:
            position = len(text)
        if position < len(text) and not text[position:].startswith("#"):
            # A multiline quoted word is incomplete; never scan its body as code.
            continue
        pending = ""
        expect_delimiter = False
        for raw in tokens:
            if expect_delimiter:
                strip_tabs = raw.startswith("-")
                value = raw[1:] if strip_tabs else raw
                try:
                    words = shlex.split(value)
                except ValueError:
                    return False
                if len(words) != 1 or any(c in words[0] for c in "$`"):
                    return False
                here_documents.append((words[0], strip_tabs))
                expect_delimiter = False
                continue
            if raw == "<<":
                expect_delimiter = True
                continue
            if raw == ")":
                command_start = python_command = False
                continue
            if raw in {";", ";;", "&", "&&", "|", "||", "("}:
                command_start = True
                python_command = False
                continue
            try:
                words = shlex.split(raw)
            except ValueError:
                return False
            if len(words) != 1:
                command_start = python_command = False
                continue
            word = words[0]
            if python_command:
                if word in _CLIENT_OPERATIONS:
                    return True
                python_command = False
            if command_start:
                if raw in {"do", "then", "if", "elif", "while", "until", "!", "{"}:
                    continue
                python_command = word == CONTROL_CLIENT_PYTHON
                command_start = False
        command_start = True
        python_command = False
    return False


def _has_shell_operator_syntax(command: str) -> bool:
    if "\n" in command or "\r" in command:
        return True
    try:
        lexer = shlex.shlex(
            command,
            posix=True,
            punctuation_chars="".join(sorted(_SHELL_OPERATOR_CHARACTERS)),
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return True
    if any(
        token and all(character in _SHELL_OPERATOR_CHARACTERS for character in token)
        for token in tokens
    ):
        return True

    in_single_quote = False
    in_double_quote = False
    escaped = False
    for index, character in enumerate(command):
        if escaped:
            escaped = False
            continue
        if character == "\\" and not in_single_quote:
            escaped = True
            continue
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            continue
        if character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue
        if in_single_quote:
            continue
        if character == "`":
            return True
        if character == "$" and command[index + 1 : index + 2] == "(":
            return True
    return False


def _parse_control_client_arguments(
    operation: str,
    argv: list[str],
) -> None:
    if operation in SPECIALIST_ROLES:
        required = ("request", "max_steps")
    elif operation == "image_show":
        required = ("path",)
    elif operation == "ask_user":
        required = ("question",)
    elif operation == "submit":
        required = ()
    else:
        raise TerminalControlError(f"unsupported control operation: {operation}")

    parsed: dict[str, str] = {}
    index = 0
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--"):
            raise TerminalControlError(
                f"unexpected trailing control client argument: {flag!r}"
            )
        name = flag[2:].replace("-", "_")
        if name not in required:
            raise TerminalControlError(f"unknown control client argument: {flag}")
        if name in parsed:
            raise TerminalControlError(f"duplicate control client argument: {flag}")
        if index + 1 >= len(argv):
            raise TerminalControlError(f"control client argument {flag} needs a value")
        parsed[name] = argv[index + 1]
        index += 2

    missing = [name for name in required if name not in parsed]
    if missing:
        raise TerminalControlError(
            "missing required control client arguments: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    if operation in SPECIALIST_ROLES:
        _bounded_inline_text(
            parsed["request"],
            "request",
            MAX_REQUEST_CHARACTERS,
            allow_empty=True,
        )
        _positive_integer(parsed["max_steps"])
    elif operation == "image_show":
        _canonical_absolute_path(parsed["path"])
    elif operation == "ask_user":
        _bounded_inline_text(
            parsed["question"],
            "question",
            MAX_QUESTION_CHARACTERS,
            allow_empty=False,
        )


def _validate_control_client_result(
    result: CommandResult,
    *,
    expected_operation: str,
    step_limits: Mapping[SpecialistRole, int],
) -> dict[str, object]:
    if result.attachments or result.completed or result.inspected:
        raise TerminalControlError(
            "control client execution returned unexpected result metadata"
        )
    if result.stderr:
        raise TerminalControlError("control client execution wrote to stderr")
    encoded = result.stdout.encode("utf-8")
    if len(encoded) > MAX_CONTROL_CLIENT_STDOUT_BYTES:
        raise TerminalControlError(
            "control client stdout exceeds the "
            f"{MAX_CONTROL_CLIENT_STDOUT_BYTES}-byte limit"
        )
    request = _json_object(encoded, "control client stdout")
    if type(request.get("cuawright_control")) is not int:
        raise TerminalControlError("control client descriptor version must be 1")
    if request.get("cuawright_control") != 1:
        raise TerminalControlError("control client descriptor version must be 1")
    if request.get("operation") != expected_operation:
        raise TerminalControlError(
            f"control client operation must be {expected_operation!r}"
        )

    if expected_operation in SPECIALIST_ROLES:
        expected_fields = {
            "cuawright_control",
            "operation",
            "request",
            "max_steps",
        }
        role = _specialist_role(expected_operation)
        request["request"] = _bounded_inline_text(
            request.get("request"),
            "request",
            MAX_REQUEST_CHARACTERS,
            allow_empty=True,
        )
        steps = request.get("max_steps")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TerminalControlError("max-steps must be an integer")
        if not 1 <= steps <= step_limits[role]:
            raise TerminalControlError(
                f"{role} max-steps must be from 1 to {step_limits[role]}"
            )
    elif expected_operation == "image_show":
        expected_fields = {"cuawright_control", "operation", "path"}
        request["path"] = _canonical_absolute_path(request.get("path"))
    elif expected_operation == "ask_user":
        expected_fields = {"cuawright_control", "operation", "question"}
        request["question"] = _bounded_inline_text(
            request.get("question"),
            "question",
            MAX_QUESTION_CHARACTERS,
            allow_empty=False,
        )
    elif expected_operation == "submit":
        expected_fields = {"cuawright_control", "operation"}
    else:
        raise TerminalControlError(
            f"unsupported control operation: {expected_operation}"
        )
    if set(request) != expected_fields:
        raise TerminalControlError(
            "control client descriptor fields do not match its operation"
        )
    return request


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError:
        raise TerminalControlError("max-steps must be a positive integer") from None
    if parsed < 1:
        raise TerminalControlError("max-steps must be a positive integer")
    return parsed


def _bounded_inline_text(
    value: object,
    description: str,
    max_characters: int,
    *,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str):
        raise TerminalControlError(f"{description} must be text")
    if not allow_empty and not value.strip():
        raise TerminalControlError(f"{description} must be non-empty text")
    if len(value) > max_characters:
        raise TerminalControlError(f"{description} exceeds {max_characters} characters")
    return value


def _canonical_absolute_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TerminalControlError("path must be a non-empty string")
    if "\x00" in value or "\\" in value:
        raise TerminalControlError("path contains an unsafe character")
    if value.startswith("//"):
        raise TerminalControlError("path must use one absolute POSIX root")
    path = PurePosixPath(value)
    if not path.is_absolute():
        raise TerminalControlError("path must be absolute")
    if "." in path.parts or ".." in path.parts or path.as_posix() != value:
        raise TerminalControlError("path must be canonical")
    return value


def _validated_role_limits(
    values: Mapping[SpecialistRole, int],
    description: str,
) -> dict[SpecialistRole, int]:
    if set(values) != set(SPECIALIST_ROLES):
        raise ValueError(f"{description}s must contain every specialist role")
    result: dict[SpecialistRole, int] = {}
    for role in SPECIALIST_ROLES:
        value = values[role]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{role} {description} must be a positive integer")
        result[role] = value
    return result


def _specialist_role(value: str) -> SpecialistRole:
    if value not in SPECIALIST_ROLES:
        raise TerminalControlError(f"unsupported specialist role: {value!r}")
    return value


def _nonempty_utf8(data: bytes, description: str) -> str:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise TerminalControlError(f"{description} must be UTF-8") from None
    if not text.strip():
        raise TerminalControlError(f"{description} cannot be empty")
    return text


def _json_object(data: bytes, description: str) -> dict[str, object]:
    text = _nonempty_utf8(data, description)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TerminalControlError(
            f"{description} is not valid JSON: {exc.msg}"
        ) from None
    if not isinstance(value, dict):
        raise TerminalControlError(f"{description} must be a JSON object")
    return value


def _validate_unresolved(value: object, description: str) -> None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) > MAX_UNRESOLVED_ITEMS
        or any(
            not isinstance(item, str)
            or not item.strip()
            or len(item) > MAX_UNRESOLVED_ITEM_CHARACTERS
            for item in value
        )
    ):
        raise TerminalControlError(
            f"{description} unresolved must contain at most "
            f"{MAX_UNRESOLVED_ITEMS} non-empty strings of at most "
            f"{MAX_UNRESOLVED_ITEM_CHARACTERS} characters"
        )


def _image_media_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise TerminalControlError(
        "unsupported or invalid image; expected PNG, JPEG, GIF, or WebP magic"
    )


def _normalized_png(data: bytes) -> bytes:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > MAX_NORMALIZED_IMAGE_PIXELS:
                    raise TerminalControlError(
                        "image dimensions exceed the "
                        f"{MAX_NORMALIZED_IMAGE_PIXELS}-pixel limit"
                    )
                source.seek(0)
                image = ImageOps.exif_transpose(source)
                image.load()
                normalized = (
                    image.convert("RGBA")
                    if image.mode in {"RGBA", "LA"} or "transparency" in image.info
                    else image.convert("RGB")
                )
                output = io.BytesIO()
                normalized.save(output, format="PNG")
                return output.getvalue()
    except TerminalControlError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise TerminalControlError(
            f"image could not be decoded and normalized ({type(exc).__name__})"
        ) from None


def _json_text(payload: Mapping[str, object]) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _json_result(
    payload: Mapping[str, object],
    *,
    completed: bool = False,
) -> CommandResult:
    return CommandResult(0, stdout=_json_text(payload), completed=completed)


def _error_result(message: str, *, completed: bool = False) -> CommandResult:
    return CommandResult(
        2,
        stderr=f"cuawright-control: {message}\n",
        completed=completed,
    )

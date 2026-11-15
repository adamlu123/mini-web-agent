from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from cuawright.artifacts import NullRunStore, RunStore
from cuawright.budget import StepBudget
from cuawright.runtime.contracts import (
    RUN_COMMAND_TOOL_SCHEMA,
    CommandExecutor,
    CommandResult,
    Compactor,
    ModelTurn,
    RunCommandValidationError,
    ToolCall,
    TurnProvider,
    normalize_run_command,
)


class AgentLoopProtocolError(RuntimeError):
    pass


CompactionHandler = Compactor


@dataclass(frozen=True, slots=True)
class AgentLoopResult:
    completed: bool
    stop_reason: Literal["completed", "budget_exhausted"]
    steps_used: int
    commands_executed: int
    last_turn: ModelTurn | None = None
    last_command_result: CommandResult | None = None
    compactions: int = 0


@dataclass(slots=True)
class AgentLoop:
    """A linear model → run_command → environment loop."""

    adapter: TurnProvider
    plugin: CommandExecutor
    budget: StepBudget
    role: str = "actor"
    stage: str = "agent-loop"
    run_store: RunStore = field(default_factory=NullRunStore)
    compact_every: int = 0
    compaction_handler: CompactionHandler | None = None
    text_completion_handler: Callable[[str], CommandResult] | None = None
    final_turn_without_tools: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.compact_every, bool) or not isinstance(
            self.compact_every,
            int,
        ):
            raise TypeError("compact_every must be an integer")
        if self.compact_every < 0:
            raise ValueError("compact_every cannot be negative")
        if (self.compact_every > 0) != (self.compaction_handler is not None):
            raise ValueError(
                "compact_every and compaction_handler must be configured together"
            )
        if self.final_turn_without_tools and self.text_completion_handler is None:
            raise ValueError(
                "final_turn_without_tools requires a text_completion_handler"
            )

    def run(self) -> AgentLoopResult:
        starting_steps = self.budget.used_steps
        commands_executed = 0
        last_turn: ModelTurn | None = None
        last_result: CommandResult | None = None
        epoch_steps = 0
        compactions = 0

        def maybe_compact() -> None:
            nonlocal epoch_steps, compactions
            self.adapter, epoch_steps, compactions = self._maybe_compact(
                epoch_steps,
                compactions,
            )

        while self.budget.remaining_steps:
            tools = (
                ()
                if self.final_turn_without_tools and self.budget.remaining_steps == 1
                else (RUN_COMMAND_TOOL_SCHEMA,)
            )
            try:
                turn = self.adapter.next_turn(tools=tools)
            except Exception as exc:
                billed_responses = _exception_billed_responses(exc)
                if billed_responses:
                    self._charge_responses(
                        billed_responses,
                        failed_response=True,
                    )
                raise
            if not isinstance(turn, ModelTurn):
                raise AgentLoopProtocolError(
                    "ModelAdapter.next_turn() must return ModelTurn"
                )
            global_step = self._charge_responses(turn.billed_responses)
            last_turn = turn
            epoch_steps += turn.billed_responses
            self.run_store.event(
                "model_turn",
                {
                    "role": self.role,
                    "stage": self.stage,
                    "global_step": global_step,
                    "billed_responses": turn.billed_responses,
                    "continue_immediately": turn.continue_immediately,
                    "text": turn.text,
                    "tool_calls": [
                        {"call_id": call.call_id, "name": call.name}
                        for call in turn.tool_calls
                    ],
                },
            )

            if turn.continue_immediately:
                if turn.tool_calls:
                    raise AgentLoopProtocolError(
                        "A provider continuation turn cannot include tool calls"
                    )
                continue

            if not tools and turn.tool_calls:
                self._record_recovery(
                    global_step,
                    "tool_call_on_report_only_turn",
                    text=turn.text,
                )
                maybe_compact()
                continue

            if not turn.tool_calls:
                if self.text_completion_handler is not None and turn.text:
                    result = self.text_completion_handler(turn.text)
                    if not isinstance(result, CommandResult):
                        raise AgentLoopProtocolError(
                            "text_completion_handler must return CommandResult"
                        )
                    last_result = result
                    self.run_store.event(
                        "text_completion_result",
                        {
                            "role": self.role,
                            "stage": self.stage,
                            "global_step": global_step,
                            "exit_code": result.exit_code,
                            "completed": result.completed,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        },
                    )
                    if result.completed:
                        return self._finish(
                            True,
                            starting_steps,
                            commands_executed,
                            last_turn,
                            last_result,
                            compactions,
                        )
                    if self.budget.remaining_steps:
                        self.adapter.add_user_text(
                            "Your report was rejected: "
                            + (result.stderr or "invalid report")
                            + " Continue inspecting with run_command or return a "
                            "corrected final report object."
                        )
                    self._record_recovery(
                        global_step,
                        "invalid_text_completion",
                        text=turn.text,
                    )
                    maybe_compact()
                    continue
                recovery = (
                    "Your previous response contained no run_command call. "
                    "Issue the next terminal command now."
                )
                if self.role != "verifier":
                    recovery += (
                        " If required information is genuinely user-controlled, "
                        "use `python /opt/cuawright-tools/ask_user.py "
                        '--question "TEXT"` in a standalone run_command.'
                    )
                if self.budget.remaining_steps:
                    self.adapter.add_user_text(recovery)
                self._record_recovery(global_step, "no_tool_call", text=turn.text)
                maybe_compact()
                continue

            if any(call.parse_error is not None for call in turn.tool_calls):
                for index, call in enumerate(turn.tool_calls, start=1):
                    detail = call.parse_error or (
                        "Another command in this model turn had malformed JSON."
                    )
                    self._return_result(
                        call,
                        CommandResult(
                            2,
                            stderr=(
                                "run_command arguments were malformed or the batch "
                                f"could not be executed: {detail} Retry with shorter "
                                "valid JSON arguments."
                            ),
                        ),
                        global_step=global_step,
                        command_index=index,
                        commands_in_turn=len(turn.tool_calls),
                    )
                self._record_recovery(global_step, "malformed_tool_arguments")
                maybe_compact()
                continue

            try:
                commands = [
                    (call, *_run_command_arguments(call)) for call in turn.tool_calls
                ]
            except AgentLoopProtocolError as exc:
                error = CommandResult(
                    2,
                    stderr=(
                        f"Invalid run_command call: {exc}. Retry using only the "
                        "canonical run_command schema."
                    ),
                )
                for index, call in enumerate(turn.tool_calls, start=1):
                    self._return_result(
                        call,
                        error,
                        global_step=global_step,
                        command_index=index,
                        commands_in_turn=len(turn.tool_calls),
                    )
                self._record_recovery(global_step, "invalid_tool_call")
                maybe_compact()
                continue

            if len(commands) > 1 and any(
                _is_completion_command(command) for _, command, _ in commands
            ):
                error = CommandResult(
                    2,
                    stderr=(
                        "submit.py must be the only command in its run_command call"
                    ),
                )
                for index, (call, _, _) in enumerate(commands, start=1):
                    self._return_result(
                        call,
                        error,
                        global_step=global_step,
                        command_index=index,
                        commands_in_turn=len(commands),
                    )
                self._record_recovery(
                    global_step,
                    "non_standalone_completion_batch",
                )
                maybe_compact()
                continue

            for index, (call, command, timeout) in enumerate(commands, start=1):
                result = self.plugin.run_command(command, timeout)
                if not isinstance(result, CommandResult):
                    raise AgentLoopProtocolError(
                        "BenchmarkPlugin.run_command() must return CommandResult"
                    )
                commands_executed += 1
                last_result = result
                self._return_result(
                    call,
                    result,
                    global_step=global_step,
                    command_index=index,
                    commands_in_turn=len(commands),
                )
                if result.completed:
                    for skipped_index, (skipped_call, _, _) in enumerate(
                        commands[index:],
                        start=index + 1,
                    ):
                        self._return_result(
                            skipped_call,
                            CommandResult(
                                2,
                                stderr=(
                                    "Skipped because an earlier command in this "
                                    "model turn completed the agent phase."
                                ),
                            ),
                            global_step=global_step,
                            command_index=skipped_index,
                            commands_in_turn=len(commands),
                        )
                    return self._finish(
                        True,
                        starting_steps,
                        commands_executed,
                        last_turn,
                        last_result,
                        compactions,
                    )

            maybe_compact()

        return self._finish(
            False,
            starting_steps,
            commands_executed,
            last_turn,
            last_result,
            compactions,
        )

    def _charge(
        self,
        *,
        failed_response: bool = False,
        stage: str | None = None,
    ) -> int:
        charge_stage = stage or self.stage
        global_step = self.budget.consume(self.role, charge_stage)
        payload: dict[str, object] = {
            "role": self.role,
            "stage": charge_stage,
            "global_step": global_step,
            "budget": self.budget.snapshot(),
        }
        if failed_response:
            payload["failed_response"] = True
        self.run_store.event("shared_budget_step", payload)
        return global_step

    def _charge_responses(
        self,
        count: int,
        *,
        failed_response: bool = False,
        stage: str | None = None,
    ) -> int:
        global_step = self.budget.used_steps
        for response_index in range(1, count + 1):
            global_step = self._charge(
                failed_response=failed_response,
                stage=stage,
            )
            if count > 1:
                self.run_store.event(
                    "provider_response_charge",
                    {
                        "role": self.role,
                        "stage": stage or self.stage,
                        "global_step": global_step,
                        "response_index": response_index,
                        "responses_in_turn": count,
                        "failed_response": failed_response,
                    },
                )
        return global_step

    def _return_result(
        self,
        call: ToolCall,
        result: CommandResult,
        *,
        global_step: int,
        command_index: int,
        commands_in_turn: int,
    ) -> None:
        self.run_store.event(
            "command_result",
            {
                "role": self.role,
                "stage": self.stage,
                "global_step": global_step,
                "command_index": command_index,
                "commands_in_turn": commands_in_turn,
                "call_id": call.call_id,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "attachments": [
                    {
                        "kind": attachment.kind,
                        "media_type": attachment.media_type,
                        "source_path": attachment.source_path,
                        "size_bytes": len(attachment.data),
                    }
                    for attachment in result.attachments
                ],
                "completed": result.completed,
                "inspected": result.inspected,
            },
        )
        self.adapter.add_command_result(call, result)

    def _record_recovery(
        self,
        global_step: int,
        reason: str,
        *,
        text: str | None = None,
    ) -> None:
        payload = {
            "role": self.role,
            "stage": self.stage,
            "global_step": global_step,
            "reason": reason,
        }
        if text is not None:
            payload["text"] = text
        self.run_store.event("model_turn_recovery", payload)

    def _maybe_compact(
        self,
        epoch_steps: int,
        compactions: int,
    ) -> tuple[TurnProvider, int, int]:
        if (
            self.compaction_handler is None
            or epoch_steps < self.compact_every
            or self.budget.remaining_steps <= 1
        ):
            return self.adapter, epoch_steps, compactions
        charged = 0

        def charge_steps(count: int = 1) -> int:
            nonlocal charged
            if charged:
                raise AgentLoopProtocolError(
                    "Compaction responses were charged more than once"
                )
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise AgentLoopProtocolError(
                    "Compaction billed response count must be a positive integer"
                )
            charged = count
            return self._charge_responses(
                count,
                stage=f"{self.stage}-compaction",
            )

        compacted = self.compaction_handler(self.adapter, charge_steps)
        if not charged:
            raise AgentLoopProtocolError(
                "Compaction handler returned without charging its model responses"
            )
        if not hasattr(compacted, "next_turn") or not hasattr(
            compacted,
            "add_command_result",
        ):
            raise AgentLoopProtocolError(
                "Compaction handler must return a ModelAdapter"
            )
        self.run_store.event(
            "compaction",
            {
                "role": self.role,
                "stage": self.stage,
                "global_step": self.budget.used_steps,
                "compactions": compactions + 1,
                "budget": self.budget.snapshot(),
            },
        )
        return compacted, 0, compactions + 1

    def _finish(
        self,
        completed: bool,
        starting_steps: int,
        commands_executed: int,
        last_turn: ModelTurn | None,
        last_result: CommandResult | None,
        compactions: int,
    ) -> AgentLoopResult:
        result = AgentLoopResult(
            completed=completed,
            stop_reason="completed" if completed else "budget_exhausted",
            steps_used=self.budget.used_steps - starting_steps,
            commands_executed=commands_executed,
            compactions=compactions,
            last_turn=last_turn,
            last_command_result=last_result,
        )
        self.run_store.event(
            "agent_loop_stopped",
            {
                "role": self.role,
                "stage": self.stage,
                "completed": result.completed,
                "stop_reason": result.stop_reason,
                "steps_used": result.steps_used,
                "commands_executed": result.commands_executed,
                "compactions": result.compactions,
                "budget": self.budget.snapshot(),
            },
        )
        return result


def _run_command_arguments(call: ToolCall) -> tuple[str, int | None]:
    try:
        return normalize_run_command(call)
    except RunCommandValidationError as exc:
        raise AgentLoopProtocolError(str(exc)) from exc


def _is_completion_command(command: str) -> bool:
    normalized = " ".join(command.strip().split())
    return normalized == "python /opt/cuawright-tools/submit.py"


def _exception_billed_responses(exc: Exception) -> int:
    count = getattr(exc, "billed_response_count", None)
    if count is None:
        return 1 if getattr(exc, "billed_response", False) else 0
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise AgentLoopProtocolError(
            "Provider exception billed_response_count must be a non-negative integer"
        ) from exc
    return count

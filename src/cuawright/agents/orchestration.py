from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from cuawright.agents.default import AgentLoop, AgentLoopResult
from cuawright.artifacts import NullRunStore, RunStore
from cuawright.budget import SharedStepBudget, StepBudget
from cuawright.config.prompts import (
    AgentTopology,
    BenchmarkOverlay,
    OSWORLD_OVERLAY,
    PromptBundle,
    PriorReportContent,
    actor_resume_user_prompt,
    actor_user_prompt,
    budget_status_user_prompt,
    compose_system_prompt,
    compaction_user_prompt,
    phase_user_prompt,
    prompt_hash_manifest,
    specialist_user_prompt,
)
from cuawright.environments.base import BenchmarkPlugin, CommandObserver
from cuawright.environments.control import (
    AcceptedReport,
    MAX_REPORT_BYTES,
    MAX_REPORT_CHARACTERS,
    StatePersistenceError,
    SpecialistInvocation,
    SpecialistRunResult,
    TerminalCommandBroker,
    TerminalControlState,
)
from cuawright.models.base import DEFAULT_MAX_OUTPUT_TOKENS
from cuawright.runtime.contracts import (
    CommandResult,
    ModelTurn,
    ToolCall,
    TurnProvider,
    normalized_turn,
)


class ThreadFactory(Protocol):
    def __call__(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
    ) -> Any:
        """Create one fresh provider thread."""


class AdapterFactory(Protocol):
    def __call__(self, thread: Any) -> TurnProvider:
        """Wrap one provider thread in the normalized adapter."""


AskUser = Callable[[str], str]


def finalization_window(total_steps: int) -> int:
    return max(16, min(48, round(total_steps * 0.16)))


@dataclass(frozen=True, slots=True)
class TerminalRuntimeResult:
    loop: AgentLoopResult
    state: TerminalControlState
    prompt_hashes: Mapping[str, Mapping[str, Any]]


class _PromptingAdapter:
    """Append dynamic budget/final-call messages before provider turns."""

    def __init__(
        self,
        adapter: TurnProvider,
        before_turn: Callable[[], str],
    ) -> None:
        self.adapter = adapter
        self.before_turn = before_turn
        self._continuation_pending = False

    @property
    def thread(self) -> Any:
        thread = getattr(self.adapter, "thread", None)
        if thread is None:
            raise TypeError("terminal runtime requires a thread-backed adapter")
        return thread

    def next_turn(
        self,
        *,
        tools: Sequence[Mapping[str, object]],
    ) -> ModelTurn:
        if not self._continuation_pending:
            prompt = self.before_turn()
            if prompt:
                self.thread.add_user_text(prompt)
        turn = self.adapter.next_turn(tools=tools)
        self._continuation_pending = turn.continue_immediately
        return turn

    def add_command_result(
        self,
        tool_call: ToolCall,
        result: CommandResult,
    ) -> None:
        self.adapter.add_command_result(tool_call, result)

    def add_user_text(self, text: str) -> None:
        self.adapter.add_user_text(text)


class TerminalAgentRuntime:
    """Run the active actor-orchestrator through terminal commands and files."""

    def __init__(
        self,
        *,
        plugin: BenchmarkPlugin,
        thread_factory: ThreadFactory,
        adapter_factory: AdapterFactory,
        instruction: str,
        task_id: str,
        total_steps: int,
        compact_every: int,
        run_store: RunStore | None = None,
        ask_user: AskUser | None = None,
        specialist_ask_user: AskUser | None = None,
        prompt_bundle: PromptBundle | None = None,
        overlay: BenchmarkOverlay | None = None,
        workspace: str = "/workspace/.agent",
        parallel_tool_suffix: str | None = None,
        state: TerminalControlState | None = None,
        specialist_instruction: str | None = None,
        initial_actor_message: str | None = None,
        agent_topology: AgentTopology | None = None,
        command_observer: CommandObserver | None = None,
    ) -> None:
        if prompt_bundle is not None and overlay is not None:
            raise ValueError("provide prompt_bundle or overlay, not both")
        if prompt_bundle is not None:
            if prompt_bundle.composition != "terminal":
                raise ValueError(
                    f"prompt bundle {prompt_bundle.name!r} is not terminal-layered"
                )
            if (
                agent_topology is not None
                and agent_topology != prompt_bundle.agent_topology
            ):
                raise ValueError(
                    f"prompt bundle {prompt_bundle.name!r} uses topology "
                    f"{prompt_bundle.agent_topology!r}, not {agent_topology!r}"
                )
            effective_topology = prompt_bundle.agent_topology
        else:
            effective_topology = agent_topology or "orchestrated"
        if effective_topology not in {"orchestrated", "linear"}:
            raise ValueError(f"unsupported agent topology: {effective_topology!r}")
        self.plugin = plugin
        self.thread_factory = thread_factory
        self.adapter_factory = adapter_factory
        self.instruction = instruction
        self.specialist_instruction = specialist_instruction or instruction
        self.task_id = task_id
        self.budget = SharedStepBudget(total_steps)
        self.finalization_window = finalization_window(total_steps)
        self.compact_every = compact_every
        self.run_store = run_store or NullRunStore()
        self.ask_user = ask_user
        self.specialist_ask_user = specialist_ask_user or ask_user
        self.prompt_bundle = prompt_bundle
        self.overlay = overlay or OSWORLD_OVERLAY
        self.workspace = workspace
        self.parallel_tool_suffix = parallel_tool_suffix
        self.agent_topology = effective_topology
        self.command_observer = command_observer
        self.enabled_specialist_roles = (
            ("binder", "verifier") if effective_topology == "orchestrated" else ()
        )
        self.state = state or TerminalControlState(
            session_id=f"terminal-{task_id}",
            phase_id="phase-1",
        )
        self._midpoint_reminder_sent = False
        self._finalization_reminder_sent = False
        self._actor_initial_prompt = actor_user_prompt(
            instruction,
            agent_topology=self.agent_topology,
        )
        if initial_actor_message is not None:
            self._actor_initial_prompt += "\n\n" + initial_actor_message
        self._actor_adapter: _PromptingAdapter | None = None

    def run(self) -> TerminalRuntimeResult:
        directories = " ".join(
            shlex.quote(f"{self.workspace}/{name}") for name in ("cache",)
        )
        initialized = self.plugin.run_command(
            f"mkdir -p -- {shlex.quote(self.workspace)} {directories}",
            30,
        )
        if initialized.exit_code != 0:
            raise StatePersistenceError(
                "could not initialize terminal control directories"
            )
        self.plugin.write_control_file_atomic(
            f"{self.workspace}/task.md",
            self.specialist_instruction.encode("utf-8"),
            workspace=self.workspace,
        )
        self._actor_adapter = self._new_prompting_adapter(
            role="actor",
            user_prompt=self._actor_initial_prompt,
            budget=self.budget,
        )
        return self._run_current_phase()

    def continue_phase(
        self,
        *,
        instruction: str,
        phase_id: str,
        phase_index: int,
        phase_count: int,
    ) -> TerminalRuntimeResult:
        if self._actor_adapter is None:
            raise RuntimeError("initial terminal phase has not run")
        self.plugin.write_control_file_atomic(
            f"{self.workspace}/task.md",
            instruction.encode("utf-8"),
            workspace=self.workspace,
        )
        self.specialist_instruction = instruction
        self.state.reset_phase(phase_id)
        self._finalization_reminder_sent = False
        phase_message = phase_user_prompt(
            instruction,
            phase_id=phase_id,
            phase_index=phase_index,
            phase_count=phase_count,
            previous_phase_evaluated=True,
            agent_topology=self.agent_topology,
        )
        self._actor_adapter.thread.add_user_text(phase_message)
        self._actor_initial_prompt = "\n\n".join(
            (
                actor_user_prompt(
                    self.instruction,
                    agent_topology=self.agent_topology,
                ),
                phase_message,
            )
        )
        return self._run_current_phase()

    def _run_current_phase(self) -> TerminalRuntimeResult:
        if self._actor_adapter is None:
            raise RuntimeError("terminal Actor adapter is not initialized")
        broker = TerminalCommandBroker(
            self.plugin,
            budget=self.budget,
            run_store=self.run_store,
            specialist_runner=(
                self._run_specialist if self.agent_topology == "orchestrated" else None
            ),
            ask_user=(
                self._answer_user_question
                if self.ask_user is not None or self.specialist_ask_user is not None
                else None
            ),
            state=self.state,
            workspace=self.workspace,
            enabled_specialist_roles=self.enabled_specialist_roles,
            stage="orchestrate",
            command_observer=self.command_observer,
        )
        loop_runner = AgentLoop(
            adapter=self._actor_adapter,
            plugin=broker,
            budget=self.budget,
            role="actor",
            stage="orchestrate",
            run_store=self.run_store,
            compact_every=self.compact_every,
            compaction_handler=self._compact_actor if self.compact_every else None,
        )
        loop = loop_runner.run()
        if not isinstance(loop_runner.adapter, _PromptingAdapter):
            raise TypeError("terminal Actor loop returned an unexpected adapter")
        self._actor_adapter = loop_runner.adapter
        if self.state.submission is None:
            missing_reports = [
                role
                for role in self.enabled_specialist_roles
                if role not in self.state.accepted_reports
            ]
            forced_payload = {
                "status": "forced",
                "reason": "Actor loop ended without explicit submit.py",
                "completed_roles": list(self.state.completed_roles),
                "missing_reports": missing_reports,
            }
            broker.force_submission(
                json.dumps(
                    forced_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        return TerminalRuntimeResult(
            loop=loop,
            state=self.state,
            prompt_hashes={
                role: prompt_hash_manifest(
                    role,
                    self.prompt_bundle or self.overlay,
                    agent_topology=(
                        None if self.prompt_bundle is not None else self.agent_topology
                    ),
                    parallel_tool_suffix=self.parallel_tool_suffix,
                )
                for role in (
                    ("actor", *self.enabled_specialist_roles)
                    if self.agent_topology == "orchestrated"
                    else ("actor",)
                )
            },
        )

    def _run_specialist(
        self,
        invocation: SpecialistInvocation,
        broker: TerminalCommandBroker,
    ) -> SpecialistRunResult:
        prior_reports = self._validated_prior_reports(invocation.role)
        request = (
            "Independently inspect the live saved and reloaded canonical state "
            "against every explicit requirement in the original instruction. "
            "Derive expected values from task-visible evidence; do not confirm "
            "an Actor-proposed answer. Return repair only for a direct objective "
            "contradiction in canonical evaluator-facing state; missing proof or "
            "an inferred preference is unknown."
            if invocation.role == "verifier"
            else invocation.request_content
        )
        cache_path = f"{self.workspace}/cache/{invocation.invocation_id}"
        prompt = specialist_user_prompt(
            invocation.role,
            instruction=self.specialist_instruction,
            request=request,
            cache_path=cache_path,
            allocated_steps=invocation.allocated_steps,
            budget_status=_budget_text(invocation.budget),
            prior_reports=prior_reports or None,
        )
        prior_roles = sorted(prior_reports)
        self.run_store.event(
            "specialist_context_prepared",
            {
                "role": invocation.role,
                "invocation_id": invocation.invocation_id,
                "phase_id": invocation.phase_id,
                "prior_roles": prior_roles,
                "prior_reports": {
                    prior_role: {
                        "report_artifact": self.state.accepted_reports[
                            prior_role
                        ].artifact.path,
                        "report_sha256": self.state.accepted_reports[
                            prior_role
                        ].artifact.sha256,
                    }
                    for prior_role in prior_roles
                },
            },
        )
        adapter = self._new_prompting_adapter(
            role=invocation.role,
            user_prompt=prompt,
            budget=invocation.budget,
        )
        loop = AgentLoop(
            adapter=adapter,
            plugin=broker,
            budget=invocation.budget,
            role=invocation.role,
            stage=f"specialist-{invocation.invocation_id}",
            run_store=self.run_store,
            text_completion_handler=broker.accept_text_report,
            final_turn_without_tools=True,
        ).run()
        if loop.completed:
            return SpecialistRunResult(True, loop.steps_used)

        forced_status = "unknown" if invocation.role == "verifier" else "blocked"
        report_attempt = (
            loop.last_turn is not None
            and not loop.last_turn.continue_immediately
            and not loop.last_turn.tool_calls
            and loop.last_turn.text is not None
            and loop.last_command_result is not None
            and not loop.last_command_result.completed
        )
        final_text = loop.last_turn.text if report_attempt else ""
        final_text_data = final_text.encode("utf-8")
        if (
            final_text.strip()
            and len(final_text) <= MAX_REPORT_CHARACTERS
            and len(final_text_data) <= MAX_REPORT_BYTES
        ):
            body = final_text
            report_error = loop.last_command_result.stderr
            unresolved = [
                ("Final report response was rejected: " + report_error.strip())[:1000]
            ]
            self.run_store.event(
                "specialist_invalid_text_preserved",
                {
                    "role": invocation.role,
                    "invocation_id": invocation.invocation_id,
                    "report_size_bytes": len(final_text_data),
                    "reason": report_error,
                },
            )
        else:
            body = (
                f"# {invocation.role.title()} report\n\n"
                f"{invocation.role.title()} exhausted its allocated "
                f"{invocation.allocated_steps} model calls without returning a "
                "valid report."
            )
            unresolved = ["No valid specialist report was submitted."]
        forced = broker.force_report(
            status=forced_status,
            report=body,
            unresolved=unresolved,
        )
        return SpecialistRunResult(forced.completed, loop.steps_used, forced=True)

    def _answer_user_question(self, role: str, question: str) -> str:
        callback = self.ask_user if role == "actor" else self.specialist_ask_user
        if callback is None:
            raise RuntimeError("no ask-user callback is configured")
        return callback(question)

    def _new_prompting_adapter(
        self,
        *,
        role: str,
        user_prompt: str,
        budget: StepBudget,
    ) -> _PromptingAdapter:
        system_prompt = compose_system_prompt(
            role,
            self.prompt_bundle or self.overlay,
            agent_topology=(
                None if self.prompt_bundle is not None else self.agent_topology
            ),
            parallel_tool_suffix=self.parallel_tool_suffix,
        )
        thread = self.thread_factory(
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return _PromptingAdapter(
            self.adapter_factory(thread),
            lambda: self._turn_prompt(role, budget),
        )

    def _turn_prompt(self, role: str, budget: StepBudget) -> str:
        if role != "actor":
            if budget.remaining_steps != 1:
                return ""
            return (
                "This is the final allocated specialist call. Do not call "
                "run_command. Return exactly one JSON object with only `status`, "
                "`report`, and `unresolved`, using a valid status for your role."
            )
        snapshot = budget.shared_snapshot()
        midpoint_due = not self._midpoint_reminder_sent and int(
            snapshot["used_steps"]
        ) >= round(int(snapshot["total_steps"]) * 0.5)
        prompt = budget_status_user_prompt(
            _budget_text(budget),
            midpoint_due=midpoint_due,
            verifier_called="verifier" in self.state.accepted_reports,
            require_verifier=self.agent_topology == "orchestrated",
        )
        if midpoint_due:
            self._midpoint_reminder_sent = True
        finalization_due = (
            not self._finalization_reminder_sent
            and budget.remaining_steps <= self.finalization_window
        )
        if finalization_due:
            if self.agent_topology == "orchestrated":
                prompt += (
                    "\nFinalization window: stop optional exploration. Ensure the "
                    "canonical result is saved and reloaded, call Verifier now if "
                    "needed, repair only direct failures, then run "
                    "`python /opt/cuawright-tools/submit.py`."
                )
            else:
                prompt += (
                    "\nFinalization window: stop optional exploration. Ensure the "
                    "canonical result is saved, reloaded, and directly checked "
                    "against every task requirement, then run "
                    "`python /opt/cuawright-tools/submit.py`."
                )
            self._finalization_reminder_sent = True
        if budget.remaining_steps == 1:
            prompt += (
                "\nThis is the final allocated model call. Use it to save any "
                "remaining essential change, or run "
                "`python /opt/cuawright-tools/submit.py` if the result is "
                "already saved. Keep submit standalone; do not combine it "
                "with a write command. Budget exhaustion retains the current "
                "persistent result for evaluation."
            )
        return prompt

    def _compact_actor(
        self,
        adapter: TurnProvider,
        charge_steps: Callable[[int], int],
    ) -> TurnProvider:
        if not isinstance(adapter, _PromptingAdapter):
            raise TypeError("terminal compaction requires the prompting adapter")
        adapter.thread.add_user_text(
            compaction_user_prompt(
                agent_topology=self.agent_topology,
                prompt_bundle=self.prompt_bundle,
            )
        )
        billed_responses = 0
        while True:
            if self.budget.remaining_steps <= billed_responses:
                charge_steps(billed_responses)
                raise RuntimeError(
                    "Actor compaction continuation exhausted the shared budget"
                )
            try:
                call_kwargs: dict[str, Any] = {
                    "tools": [],
                    "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                    "tool_choice": "none",
                }
                compaction_effort = getattr(
                    adapter.thread,
                    "compaction_reasoning_effort",
                    None,
                )
                if compaction_effort is not None:
                    call_kwargs["reasoning_effort"] = compaction_effort
                response = adapter.thread.call(
                    **call_kwargs,
                )
            except Exception as exc:
                exception_responses = getattr(
                    exc,
                    "billed_response_count",
                    None,
                )
                if exception_responses is None:
                    exception_responses = (
                        1 if getattr(exc, "billed_response", False) else 0
                    )
                if billed_responses or exception_responses:
                    charge_steps(billed_responses + exception_responses)
                raise
            if not isinstance(response, Mapping):
                raise TypeError("compaction response must be a mapping")
            normalized = normalized_turn(
                response,
                first_tool_call_only=False,
            )
            billed_responses += normalized.billed_responses
            if not normalized.continue_immediately:
                break
        charge_steps(billed_responses)
        summary = normalized.text
        if not summary:
            raise RuntimeError("terminal Actor compaction returned no summary")
        fresh = self._new_prompting_adapter(
            role="actor",
            user_prompt=self._actor_initial_prompt,
            budget=self.budget,
        )
        fresh.thread.add_user_text(actor_resume_user_prompt(summary))
        self.run_store.event(
            "compaction_summary",
            {
                "role": "actor",
                "summary": summary,
                "budget": self.budget.snapshot(),
            },
        )
        return fresh

    def _validated_prior_reports(
        self,
        role: str,
    ) -> dict[str, PriorReportContent]:
        required = {
            "binder": (),
            "critic": ("binder",),
            "verifier": (),
        }[role]
        result: dict[str, PriorReportContent] = {}
        for prior_role in required:
            report: AcceptedReport | None = self.state.accepted_reports.get(prior_role)
            if report is not None:
                result[prior_role] = PriorReportContent(
                    status=report.status,
                    report=report.body,
                    unresolved=report.unresolved,
                )
        return result


def _budget_text(budget: StepBudget) -> str:
    snapshot = budget.snapshot()
    scope = snapshot.get("scope")
    scope_text = ""
    if isinstance(scope, Mapping):
        scope_text = (
            f" Invocation scope: {scope['used_steps']}/{scope['limit']} used; "
            f"{scope['remaining_steps']} remain."
        )
    return (
        f"Global model-call budget: {snapshot['used_steps']}/"
        f"{snapshot['total_steps']} used; {snapshot['remaining_steps']} remain."
        + scope_text
    )

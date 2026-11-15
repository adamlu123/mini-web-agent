from __future__ import annotations

import json
import signal
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from cuawright.agents.orchestration import TerminalAgentRuntime
from cuawright.artifacts import TraceWriter, TraceWriterRunStore, utc_now
from cuawright.budget import DEFAULT_TOTAL_STEPS
from cuawright.config.prompts import (
    OSWORLD_OVERLAY,
    AgentTopology,
    BenchmarkOverlay,
    PromptBundle,
    prompt_hash_manifest,
)
from cuawright.environments.base import BenchmarkPlugin, CommandObserver
from cuawright.health.contracts import health_contract_manifest
from cuawright.models.base import DEFAULT_MAX_OUTPUT_TOKENS
from cuawright.models.providers import PROVIDER_DEFAULT_BASE_URLS
from cuawright.providers.factory import (
    ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL,
    ANTHROPIC_DEFAULT_MAX_REQUEST_MB,
    ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES,
    ProviderRuntime,
    ProviderSpec,
    create_provider_runtime,
    create_turn_provider,
    parallel_tool_suffix,
)
from cuawright.runtime.contracts import TurnProvider

DEFAULT_USER_RESPONSE = (
    "I have no further information to provide. The environment and original "
    "instruction contain enough information; please continue without asking "
    "another question."
)


class TerminalBenchmarkDeadlineExceeded(RuntimeError):
    pass


@dataclass
class TaskConfig:
    task_id: str
    api_key_file: str
    result_root: str
    base_url: str = PROVIDER_DEFAULT_BASE_URLS["openai"]
    model: str = "gpt-5.5"
    reasoning_effort: str = "medium"
    total_steps: int = DEFAULT_TOTAL_STEPS
    compact_every: int = 40
    model_provider: str = "openai"
    max_dollars: float | None = None
    model_infrastructure: str | None = "aiservices"
    openrouter_provider_preferences: dict[str, Any] = field(default_factory=dict)
    anthropic_min_recent_images: int = ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES
    anthropic_image_prune_interval: int = ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL
    anthropic_max_request_mb: float = ANTHROPIC_DEFAULT_MAX_REQUEST_MB
    anthropic_max_output_tokens: int | None = None
    openai_policy_feedback_max_retries: int = 0

    def __post_init__(self) -> None:
        if self.model_provider not in {
            "openai",
            "openrouter-anthropic",
            "anthropic",
        }:
            raise ValueError(f"Unsupported model_provider: {self.model_provider!r}")
        self.provider_spec()

    def provider_spec(self) -> ProviderSpec:
        return ProviderSpec(
            name=self.model_provider,
            api_key_file=self.api_key_file,
            base_url=self.base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            max_dollars=self.max_dollars,
            infrastructure=self.model_infrastructure,
            openrouter_provider_preferences=self.openrouter_provider_preferences,
            anthropic_min_recent_images=self.anthropic_min_recent_images,
            anthropic_image_prune_interval=self.anthropic_image_prune_interval,
            anthropic_max_request_mb=self.anthropic_max_request_mb,
            anthropic_max_output_tokens=self.anthropic_max_output_tokens,
            openai_policy_feedback_max_retries=(
                self.openai_policy_feedback_max_retries
            ),
        )


class TaskRunner:
    """Thin benchmark-neutral composition of model, Agent, and Environment."""

    def __init__(
        self,
        config: TaskConfig,
        *,
        ask_user: Callable[[str], str] | None = None,
    ) -> None:
        self.config = config
        run_name = (
            f"task-{config.task_id}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        self.run_dir = Path(config.result_root).resolve() / run_name
        self.trace = TraceWriter(self.run_dir)
        self.ask_user = ask_user
        self.provider_runtime: ProviderRuntime = create_provider_runtime(
            config.provider_spec(),
            run_dir=self.run_dir,
        )
        self.api_key = self.provider_runtime.api_key
        self.client = self.provider_runtime.client
        self.cost_tracker = self.provider_runtime.cost_tracker
        self.claude_client = self.provider_runtime.claude_client
        self.async_runner = self.provider_runtime.async_runner
        self._actor_answered_questions: dict[str, str] = {}
        self._specialist_answered_questions: dict[str, str] = {}
        self._closed = False

    def new_model_thread(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> Any:
        return self.provider_runtime.new_thread(
            trace=self.trace,
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=list(tools or ()),
            max_output_tokens=max_output_tokens,
        )

    def run_terminal_benchmark(
        self,
        *,
        benchmark: str,
        plugin: BenchmarkPlugin,
        instruction: str,
        manifest_extra: Mapping[str, Any],
        prompt_bundle: PromptBundle | None = None,
        overlay: BenchmarkOverlay | None = None,
        deadline_seconds: int | None = None,
        command_observer: CommandObserver | None = None,
    ) -> dict[str, Any]:
        agent_topology = self._agent_topology(prompt_bundle)
        prompt_hashes = self._prompt_hashes(
            prompt_bundle=prompt_bundle,
            overlay=overlay,
        )
        manifest = {
            **asdict(self.config),
            **dict(manifest_extra),
            "benchmark": benchmark,
            "task_id": self.config.task_id,
            "started_at": utc_now(),
            "model_interface": "terminal-only",
            "agent_topology": agent_topology,
            "health_contract": health_contract_manifest(),
            "prompt_hashes": prompt_hashes,
        }
        self._write_manifest(manifest)
        try:
            deadline_triggered = False
            previous_handler = None
            if deadline_seconds is not None:
                if (
                    isinstance(deadline_seconds, bool)
                    or not isinstance(deadline_seconds, int)
                    or deadline_seconds < 1
                ):
                    raise ValueError("deadline_seconds must be a positive integer")
                if threading.current_thread() is not threading.main_thread():
                    raise RuntimeError(
                        "terminal benchmark deadline requires the main thread"
                    )

                def deadline_handler(signum: int, frame: Any) -> None:
                    del signum, frame
                    raise TerminalBenchmarkDeadlineExceeded(
                        "Terminal benchmark emergency cutoff reached"
                    )

                previous_handler = signal.signal(signal.SIGALRM, deadline_handler)
                signal.setitimer(signal.ITIMER_REAL, deadline_seconds)
            try:
                submission, verification, prompt_hashes = self.run_terminal_plugin(
                    plugin,
                    instruction=instruction,
                    prompt_bundle=prompt_bundle,
                    overlay=overlay,
                    command_observer=command_observer,
                )
            except TerminalBenchmarkDeadlineExceeded:
                deadline_triggered = True
                submission = {
                    "status": "deadline_exceeded",
                    "reason": "emergency_cutoff",
                    "missing_subagents": None,
                    "artifact": None,
                }
                verification = {
                    "mode": "terminal-only",
                    "agent_topology": agent_topology,
                    "completed": False,
                    "stop_reason": "emergency_cutoff",
                    "deadline_seconds": deadline_seconds,
                }
                self.trace.event(
                    "terminal_benchmark_deadline",
                    {
                        "benchmark": benchmark,
                        "task_id": self.config.task_id,
                        "deadline_seconds": deadline_seconds,
                    },
                )
            finally:
                if deadline_seconds is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    assert previous_handler is not None
                    signal.signal(signal.SIGALRM, previous_handler)

            evaluation = plugin.evaluate()
            if isinstance(evaluation, Mapping):
                raw_score = evaluation.get("score")
                score = float(raw_score) if raw_score is not None else None
            else:
                score = float(evaluation)
            manifest["prompt_hashes"] = prompt_hashes
            self._write_manifest(manifest)
            result = {
                "benchmark": benchmark,
                "task_id": self.config.task_id,
                "submission": submission,
                "verification": verification,
                "score": score,
                "evaluation": evaluation,
                "run_dir": str(self.run_dir),
                "model_cost": self.model_cost_snapshot(),
                "prompt_hashes": prompt_hashes,
                "deadline_triggered": deadline_triggered,
            }
            self._write_evaluation(result, score)
            return result
        finally:
            try:
                plugin.close()
            finally:
                self.close()

    def run_terminal_plugin(
        self,
        plugin: BenchmarkPlugin,
        *,
        instruction: str,
        prompt_bundle: PromptBundle | None = None,
        overlay: BenchmarkOverlay | None = None,
        specialist_instruction: str | None = None,
        initial_actor_message: str | None = None,
        command_observer: CommandObserver | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Mapping[str, Any]]]:
        runtime = self.create_runtime(
            plugin,
            instruction=instruction,
            prompt_bundle=prompt_bundle,
            overlay=overlay,
            specialist_instruction=specialist_instruction,
            initial_actor_message=initial_actor_message,
            command_observer=command_observer,
        )
        outcome = runtime.run()
        submission, verification = self.outcome_payload(outcome, runtime)
        (self.run_dir / "orchestration.json").write_text(
            json.dumps(
                verification,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        self.trace.event(
            "terminal_orchestration_completed",
            {"submission": submission, "verification": verification},
        )
        return submission, verification, outcome.prompt_hashes

    def create_runtime(
        self,
        plugin: BenchmarkPlugin,
        *,
        instruction: str,
        prompt_bundle: PromptBundle | None = None,
        overlay: BenchmarkOverlay | None = None,
        specialist_instruction: str | None = None,
        initial_actor_message: str | None = None,
        command_observer: CommandObserver | None = None,
    ) -> TerminalAgentRuntime:
        def thread_factory(
            *,
            role: str,
            system_prompt: str,
            user_prompt: str,
        ) -> Any:
            return self.new_model_thread(
                role=f"terminal-{role}-{uuid.uuid4().hex[:8]}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

        def adapter_factory(thread: Any) -> TurnProvider:
            return create_turn_provider(self.config.model_provider, thread)

        if prompt_bundle is not None and overlay is not None:
            raise ValueError("provide prompt_bundle or overlay, not both")
        effective_overlay = overlay or OSWORLD_OVERLAY
        agent_topology = self._agent_topology(prompt_bundle)
        workspace = (
            prompt_bundle.workspace
            if prompt_bundle is not None
            else effective_overlay.workspace
        )
        runtime = TerminalAgentRuntime(
            plugin=plugin,
            thread_factory=thread_factory,
            adapter_factory=adapter_factory,
            instruction=instruction,
            task_id=self.config.task_id,
            total_steps=self.config.total_steps,
            compact_every=self.config.compact_every,
            run_store=TraceWriterRunStore(self.trace),
            ask_user=self._answer_actor_question,
            specialist_ask_user=self._answer_specialist_question,
            workspace=workspace,
            prompt_bundle=prompt_bundle,
            overlay=None if prompt_bundle is not None else effective_overlay,
            parallel_tool_suffix=self._parallel_tool_suffix(),
            specialist_instruction=specialist_instruction,
            initial_actor_message=initial_actor_message,
            agent_topology=agent_topology,
            command_observer=command_observer,
        )
        return runtime

    @staticmethod
    def outcome_payload(
        outcome: Any,
        runtime: TerminalAgentRuntime,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reports = {
            role: {
                "status": report.status,
                "report_path": report.artifact.path,
                "report_sha256": report.artifact.sha256,
            }
            for role, report in outcome.state.accepted_reports.items()
        }
        submission = {
            "status": "forced" if outcome.state.submission_forced else "submitted",
            "missing_subagents": list(outcome.state.submission_missing_reports),
            "artifact": (
                None
                if outcome.state.submission is None
                else outcome.state.submission.path
            ),
            "sha256": (
                None
                if outcome.state.submission is None
                else outcome.state.submission.sha256
            ),
        }
        verification = {
            "mode": "terminal-only",
            "agent_topology": runtime.agent_topology,
            "completed": outcome.loop.completed,
            "stop_reason": outcome.loop.stop_reason,
            "steps_used": outcome.loop.steps_used,
            "commands_executed": outcome.loop.commands_executed,
            "compactions": outcome.loop.compactions,
            "reports": reports,
            "budget": runtime.budget.snapshot(),
        }
        return submission, verification

    def model_cost_snapshot(self) -> dict[str, Any] | None:
        return None if self.cost_tracker is None else self.cost_tracker.snapshot()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.provider_runtime.close()

    def _answer_actor_question(self, question: str) -> str:
        return self._answer_question(
            self._actor_answered_questions,
            question=question,
            role="terminal-actor",
            stage="terminal-orchestrator",
        )

    def _answer_specialist_question(self, question: str) -> str:
        return self._answer_question(
            self._specialist_answered_questions,
            question=question,
            role="terminal-specialist",
            stage="terminal-specialist",
        )

    def _answer_question(
        self,
        answered_questions: dict[str, str],
        *,
        question: str,
        role: str,
        stage: str,
    ) -> str:
        normalized = " ".join(question.lower().split())
        repeated = normalized in answered_questions
        if repeated:
            answer = answered_questions[normalized]
        else:
            answer = (
                self.ask_user(question)
                if self.ask_user is not None
                else DEFAULT_USER_RESPONSE
            )
            answered_questions[normalized] = answer
        self.trace.event(
            "user_question",
            {
                "role": role,
                "stage": stage,
                "question": question,
                "answer": answer,
                "repeated": repeated,
                "mechanism": "agent_user_ask",
            },
        )
        return answer

    def _prompt_hashes(
        self,
        *,
        prompt_bundle: PromptBundle | None,
        overlay: BenchmarkOverlay | None,
    ) -> dict[str, Mapping[str, Any]]:
        if prompt_bundle is not None and overlay is not None:
            raise ValueError("provide prompt_bundle or overlay, not both")
        prompt_config = prompt_bundle or overlay or OSWORLD_OVERLAY
        agent_topology = self._agent_topology(prompt_bundle)
        suffix = self._parallel_tool_suffix()
        return {
            role: prompt_hash_manifest(
                role,
                prompt_config,
                agent_topology=(None if prompt_bundle is not None else agent_topology),
                parallel_tool_suffix=suffix,
            )
            for role in (
                ("actor", "binder", "critic", "verifier")
                if agent_topology == "orchestrated"
                else ("actor",)
            )
        }

    def _parallel_tool_suffix(self) -> str | None:
        provider_runtime = getattr(self, "provider_runtime", None)
        if provider_runtime is not None and hasattr(
            provider_runtime,
            "parallel_tool_suffix",
        ):
            return provider_runtime.parallel_tool_suffix
        return parallel_tool_suffix(self.config.model_provider)

    @staticmethod
    def _agent_topology(prompt_bundle: PromptBundle | None) -> AgentTopology:
        if prompt_bundle is None:
            return "orchestrated"
        if prompt_bundle.agent_topology is None:
            raise ValueError("terminal prompt bundle must declare an agent topology")
        return prompt_bundle.agent_topology

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        (self.run_dir / "manifest.json").write_text(
            json.dumps(
                dict(manifest),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

    def _write_evaluation(
        self,
        result: Mapping[str, Any],
        score: float | None,
    ) -> None:
        (self.trace.evaluation_dir / "result.json").write_text(
            json.dumps(
                dict(result),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        if score is not None:
            (self.trace.evaluation_dir / "score.txt").write_text(
                f"{score}\n",
                encoding="utf-8",
            )
        self.trace.event("evaluation", dict(result))

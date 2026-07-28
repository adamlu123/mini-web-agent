"""WebAgentGenerator — terminal-style rollout against a Playwright tab.

Mirrors ``TerminalAgentGenerator`` but:

- swaps the Harbor docker provider for ``WebAgentEnvironment`` (per-rollout
  Playwright tab seeded from the task's ``start_url``);
- replaces the in-container verifier with a modular reward callable
  (``BaseRewardFn``) built from config.

The agent loop, parsing, masking, world-model bookkeeping and SkyRL output
shape are inherited unchanged: the model still emits ``<tool_call>`` blocks,
we still execute one bash command per turn, and we still feed the resulting
text back in as the observation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from echo_rl.terminal_agent.terminal_agent_generator import (
    TerminalAgentGenerator,
    TerminalTrajectoryOutput,
)

from .rewards import RewardResult, build_reward_fn
from .web_environment import WebAgentEnvironment, WebAgentEnvironmentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT-state alignment (prompt modes "sft_state"/"sft_state_debug"): both the
# observation and the parse-error feedback are rendered EXACTLY like the
# miniswewebagent eval harness (observation_template / format_error_template in
# config/benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml), which is also what
# the SFT training data contains. Observations are user turns with no trailing
# newline (verified against web_agent_seq_om2w4000_run1.json).
# ---------------------------------------------------------------------------

SFT_STATE_FORMAT_ERROR_TEMPLATE = """Format error:

{error}

Respond in exactly this unified SFT state format and nothing else:
<think>
reasoning
</think>
<bash>
exactly one shell command, or empty when done is true
</bash>
<done>false</done>
<final_response></final_response>

If the task is fully complete and verified, use:
<think>
final verification reasoning
</think>
<bash>
</bash>
<done>true</done>
<final_response>
final response
</final_response>"""


def _truncate_harness_style(text: str, limit: int) -> str:
    """miniswewebagent LocalWorkspace._truncate: head + omitted-count marker."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n\n... [{omitted} characters omitted]"


def render_sft_state_observation(
    *,
    returncode: int,
    output: str,
    exception: str,
    workspace_dir: str,
    workspace_alias: str,
    final_script_exists: bool,
    output_truncation_chars: int,
) -> str:
    """Render the harness observation_template for one executed command.

    Field-for-field mirror of LocalWorkspace._capture_observation + the yaml
    observation_template: real workspace paths are aliased to /workspace,
    success requires rc==0 AND no exception, command output is head-truncated
    with the harness marker, and final_script.py is reported once it exists.
    """
    def display(value: str) -> str:
        return value.replace(workspace_dir, workspace_alias) if workspace_dir else value

    success = returncode == 0 and not exception
    lines = [
        "Observation:",
        f"Status: {'ok' if success else 'error'}",
        f"Workspace: {workspace_alias}",
        f"Working directory: {workspace_alias}",
        f"Return code: {returncode}",
    ]
    if exception:
        lines.append(f"Exception:\n{display(exception)}")
    command_output = _truncate_harness_style(display(output), output_truncation_chars)
    if command_output:
        lines.append(f"Command output:\n{command_output}")
    if final_script_exists:
        lines.append(f"final_script.py: {workspace_alias}/final_script.py")
    # The SFT data's observation turns carry no trailing newline (the natural
    # trailing newline INSIDE command output is preserved; only the message
    # end is stripped) — verified against web_agent_seq_om2w4000_run1.json.
    return "\n".join(lines).rstrip("\n")


@dataclass
class _WebProviderResult:
    env: WebAgentEnvironment
    task_meta: dict[str, Any]


class WebAgentEnvironmentProvider:
    """Per-rollout factory for ``WebAgentEnvironment``.

    Matches the ``HarborEnvironmentProvider`` surface area used by the parent
    generator (``prepare_batch``, ``create``, ``cleanup_batch``). Because each
    rollout spins up its own Playwright tab there is no shared "image" to
    build, so ``prepare_batch`` is a no-op.
    """

    def __init__(
        self,
        *,
        env_overrides: dict[str, Any] | None = None,
        stub: bool = False,
    ) -> None:
        self._env_overrides = dict(env_overrides or {})
        self._stub = bool(stub)
        self._envs: list[WebAgentEnvironment] = []

    async def prepare_batch(self, batch_environment_data: list[dict[str, Any]], num_generations: int) -> None:
        del batch_environment_data, num_generations

    async def create(self, environment_data: dict[str, Any]) -> WebAgentEnvironment:
        meta = environment_data.get("task_meta") or {}
        if not isinstance(meta, dict):
            raise ValueError("env_extras.task_meta must be a dict.")
        cfg = WebAgentEnvironmentConfig(
            task_id=str(meta.get("task_id") or environment_data.get("path") or ""),
            task=str(meta.get("task") or ""),
            start_url=str(meta.get("start_url") or ""),
            stub=self._stub,
            **self._env_overrides,
        )
        env = WebAgentEnvironment(cfg)
        self._envs.append(env)
        return env

    async def cleanup_batch(self) -> None:
        for env in self._envs:
            try:
                await env.cleanup()
            except Exception:
                logger.exception("web-agent env cleanup failed")
        self._envs.clear()


class WebAgentGenerator(TerminalAgentGenerator):
    """Replaces docker provider + in-env verifier with web env + judge."""

    def __init__(
        self,
        generator_cfg,
        inference_engine_client,
        tokenizer,
        max_seq_len: int,
    ) -> None:
        super().__init__(generator_cfg, inference_engine_client, tokenizer, max_seq_len)
        self._reward_fn = build_reward_fn(getattr(generator_cfg, "reward", None))
        self._stub_env = bool(getattr(generator_cfg, "stub_env", False))
        env_overrides = dict(getattr(generator_cfg, "env_overrides", {}) or {})
        if "workspace_root" in env_overrides:
            env_overrides["workspace_root"] = Path(env_overrides["workspace_root"])
        # Pull the policy's HTTP endpoint off SkyRL's inference client so
        # tools that need to call the current policy (image_qa,
        # self_reflection) can route through it.
        self._policy_endpoint_url, self._policy_model_name = self._resolve_policy_endpoint(
            inference_engine_client, generator_cfg
        )
        if self._policy_endpoint_url and "policy_endpoint_url" not in env_overrides:
            env_overrides["policy_endpoint_url"] = self._policy_endpoint_url
        if self._policy_model_name and "policy_model_name" not in env_overrides:
            env_overrides["policy_model_name"] = self._policy_model_name
        self._env_overrides = env_overrides
        # SFT-state alignment: harness-style observations + format-error
        # feedback, and observations as USER turns (the SFT data has user
        # observation turns; choose_obs_role would pick "tool" on the generic
        # qwen jinja, which the ckpt never saw).
        self._sft_state_aligned = str(getattr(generator_cfg, "prompt_mode", "")) in ("sft_state", "sft_state_debug")
        if self._sft_state_aligned:
            self._obs_role = "user"

    def _format_parse_error(self, error: str) -> str:
        if not self._sft_state_aligned:
            return super()._format_parse_error(error)
        return SFT_STATE_FORMAT_ERROR_TEMPLATE.format(error=error)

    async def _execute_commands(self, environment, commands: list) -> str:
        if not self._sft_state_aligned:
            return await super()._execute_commands(environment, commands)
        # sft_state emits exactly one <bash> command per turn (the parser
        # guarantees it); render its result like the eval harness observation.
        outputs = []
        workspace_dir = ""
        try:
            workspace_dir = str(environment.workspace)
        except Exception:  # noqa: BLE001 - stub envs may have no workspace yet
            pass
        for parsed_cmd in commands:
            if parsed_cmd.error or parsed_cmd.name != "bash":
                outputs.append(self._format_parse_error(parsed_cmd.error or f"Unknown tool '{parsed_cmd.name}'."))
                continue
            cmd = parsed_cmd.arguments.get("command", "")
            timeout = float(parsed_cmd.arguments.get("timeout", 240))
            exception_info = ""
            returncode = 0
            raw_output = ""
            try:
                result = await environment.exec(cmd, timeout=timeout)
                returncode = result.return_code
                raw_output = result.stdout or ""
                if result.stderr:
                    raw_output = f"{raw_output}\n{result.stderr}" if raw_output else result.stderr
                if returncode == 124:
                    # The harness surfaces timeouts as an exception, not output.
                    exception_info = (
                        "An error occurred while executing the command: "
                        f"Command '{cmd}' timed out after {int(timeout)} seconds"
                    )
            except (RuntimeError, TimeoutError, asyncio.TimeoutError):
                returncode = -1
                exception_info = (
                    "An error occurred while executing the command: "
                    f"Command '{cmd}' timed out after {int(timeout)} seconds"
                )
            except Exception as exc:  # noqa: BLE001 - mirror the harness catch-all
                returncode = -1
                exception_info = f"An error occurred while executing the command: {exc}"
            final_script_exists = False
            if workspace_dir:
                try:
                    final_script_exists = (Path(workspace_dir) / "final_script.py").exists()
                except OSError:
                    pass
            outputs.append(
                render_sft_state_observation(
                    returncode=returncode,
                    output=raw_output,
                    exception=exception_info,
                    workspace_dir=workspace_dir,
                    workspace_alias="/workspace",
                    final_script_exists=final_script_exists,
                    output_truncation_chars=int(self.generator_cfg.max_terminal_output_chars),
                )
            )
        return "\n\n".join(outputs)

    @staticmethod
    def _resolve_policy_endpoint(inference_engine_client, generator_cfg) -> tuple[str, str]:
        infer_cfg = getattr(generator_cfg, "inference_engine", None)
        enable_http = bool(getattr(infer_cfg, "enable_http_endpoint", False)) if infer_cfg else False
        host = str(getattr(infer_cfg, "http_endpoint_host", "127.0.0.1") or "127.0.0.1")
        port = int(getattr(infer_cfg, "http_endpoint_port", 8000) or 8000)
        # Some HTTP endpoint impls expose attributes on the client directly.
        host = (
            getattr(inference_engine_client, "http_endpoint_host", host) or host
        )
        port = int(getattr(inference_engine_client, "http_endpoint_port", port) or port)
        if not enable_http:
            return "", ""
        # The SkyRL HTTP endpoint is vLLM's OpenAI app (build_app), which mounts
        # chat completions at /v1/chat/completions -- there is NO bare
        # /chat/completions route, so omitting /v1 yields 404 {"detail":"Not
        # Found"} on every image_qa / self_reflection call.
        url = f"http://{host}:{port}/v1/chat/completions"
        # Best-effort: scrape the model id from the engine init kwargs if set.
        engine_kwargs = getattr(infer_cfg, "engine_init_kwargs", {}) or {}
        model_name = str(engine_kwargs.get("model") or engine_kwargs.get("tokenizer") or "")
        return url, model_name

    async def generate(self, input_batch, disable_tqdm: bool = False):
        # Re-implement the outer scaffolding because the parent constructs a
        # HarborEnvironmentProvider directly inside generate(). We swap in
        # ``WebAgentEnvironmentProvider`` here without touching the rollout
        # loop methods we inherit.
        prompts = input_batch["prompts"]
        env_extras = input_batch.get("env_extras")
        trajectory_ids = input_batch.get("trajectory_ids")
        sampling_params = input_batch.get("sampling_params") or {}
        if env_extras is None or trajectory_ids is None:
            raise ValueError("WebAgentGenerator requires env_extras and trajectory_ids.")
        if not (len(prompts) == len(env_extras) == len(trajectory_ids)):
            raise ValueError("prompts, env_extras, and trajectory_ids must have the same length.")

        provider = WebAgentEnvironmentProvider(env_overrides=self._env_overrides, stub=self._stub_env)
        outputs: list[TerminalTrajectoryOutput | None] = [None] * len(prompts)
        semaphore = asyncio.Semaphore(self.generator_cfg.agent_max_concurrency)

        async def _worker(idx: int) -> None:
            async with semaphore:
                try:
                    outputs[idx] = await asyncio.wait_for(
                        self._run_one(
                            provider,
                            prompts[idx],
                            env_extras[idx],
                            trajectory_ids[idx],
                            sampling_params,
                        ),
                        timeout=self.generator_cfg.agent_timeout,
                    )
                except asyncio.TimeoutError:
                    outputs[idx] = self._failure_output(
                        prompts[idx], env_extras[idx], trajectory_ids[idx], "timeout"
                    )
                except Exception as exc:
                    logger.exception("Web rollout failed for %s: %s", trajectory_ids[idx], exc)
                    outputs[idx] = self._failure_output(
                        prompts[idx], env_extras[idx], trajectory_ids[idx], "error"
                    )

        # NOTE: We deliberately avoid ``asyncio.TaskGroup`` here. TaskGroup's
        # __aexit__ waits unconditionally for *every* child task, so a single
        # rollout whose cancellation never completes (e.g. a Playwright/CDP
        # ``close()`` wedged on a dead Browserbase session, or a judge call
        # stuck in a non-cancellable worker thread) would stall the whole
        # generation step — and with it the trainer's ``ray.get()`` — forever.
        # That is exactly the ~19h hang observed on run lu4duvjj. Instead we
        # impose a hard batch-level deadline and abandon stragglers as failures
        # so the trainer always makes progress.
        tasks = [asyncio.create_task(_worker(idx)) for idx in range(len(prompts))]
        conc = max(1, int(self.generator_cfg.agent_max_concurrency))
        waves = (len(tasks) + conc - 1) // conc
        # Each wave is bounded by agent_timeout; add one extra wave + 5min of
        # slack (cleanup/judge) so healthy batches never trip the deadline.
        batch_deadline = self.generator_cfg.agent_timeout * (waves + 1) + 300
        try:
            _done, pending = await asyncio.wait(tasks, timeout=batch_deadline)
            if pending:
                logger.error(
                    "Web generation batch deadline (%.0fs, %d waves) hit with "
                    "%d/%d rollouts unfinished; cancelling and recording them "
                    "as timeouts.",
                    batch_deadline, waves, len(pending), len(tasks),
                )
                for t in pending:
                    t.cancel()
                # Bounded drain so cancellation can propagate, but never block
                # the trainer waiting on a wedged task.
                await asyncio.wait(pending, timeout=60)
            for idx in range(len(prompts)):
                if outputs[idx] is None:
                    outputs[idx] = self._failure_output(
                        prompts[idx], env_extras[idx], trajectory_ids[idx], "timeout"
                    )
        finally:
            try:
                await asyncio.wait_for(provider.cleanup_batch(), timeout=120)
            except Exception:
                logger.exception("web cleanup_batch failed or timed out")
        return self._build_generator_output([o for o in outputs if o is not None])

    async def _run_one(self, provider, prompt, env_extra, trajectory_id, sampling_params):
        # Most of the rollout body is the parent's. We only have to override
        # the verifier step at the end so it uses the modular reward.
        import copy

        from echo_rl.terminal_agent.interaction import TerminalInteraction

        prompt_token_ids = list(
            env_extra.get("prompt_token_ids")
            or self.tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_dict=False
            )
        )
        interaction = TerminalInteraction(
            prompt_id=int(trajectory_id.instance_id) if str(trajectory_id.instance_id).isdigit() else 0,
            completion_id=trajectory_id.repetition_id,
            prompt_messages=copy.deepcopy(prompt),
            prompt_token_ids=prompt_token_ids,
            metadata={
                "path": env_extra.get("path", ""),
                "data_source": env_extra.get("data_source"),
            },
        )
        environment: WebAgentEnvironment | None = None
        setup_complete = False
        t_run_start = time.monotonic()
        try:
            environment = await provider.create(env_extra)
            await environment.setup()
            setup_complete = True
            await self._agent_loop(
                interaction, trajectory_id.to_string(), environment, sampling_params
            )
        except Exception as exc:
            stop_reason = "env_setup_error" if not setup_complete else "error"
            logger.warning("Environment or agent failed for %s: %s", trajectory_id, exc, exc_info=True)
            self._mark_failure(interaction, stop_reason, exc)

        # Override the in-env verifier with the modular reward function.
        reward_result = await self._score_via_reward_fn(environment, env_extra, interaction)
        interaction.reward = reward_result.reward
        interaction.correct = reward_result.correct
        if reward_result.error:
            interaction.metadata["reward_error"] = reward_result.error
        interaction.metadata.setdefault("reward_metadata", {})
        interaction.metadata["reward_metadata"].update(reward_result.metadata)

        # Mirror the parent's trace footer so SkyRL bookkeeping stays intact.
        interaction.metadata.setdefault("trace", {})["agent_run_sec"] = time.monotonic() - t_run_start

        if environment is not None:
            try:
                await asyncio.wait_for(environment.cleanup(), timeout=60)
            except Exception:
                logger.exception("web env cleanup failed or timed out")

        self._ensure_non_empty_completion(interaction)
        return TerminalTrajectoryOutput(
            trajectory_id=trajectory_id,
            prompt_token_ids=interaction.prompt_token_ids,
            response_ids=interaction.completion_token_ids,
            loss_masks=interaction.completion_masks,
            world_loss_masks=interaction.completion_observation_masks,
            world_warning_masks=interaction.completion_warning_masks,
            world_env_masks=interaction.completion_env_output_masks,
            world_full_observation_count=int(
                interaction.metadata.get("full_observation_body_count", 0)
            ),
            rollout_logprobs=interaction.completion_logprobs,
            world_model_only=bool(env_extra.get("world_model_only", False)),
            reward=interaction.reward,
            correct=interaction.correct,
            stop_reason=str(interaction.metadata.get("stop_reason", "error")),
            metrics=interaction.metrics,
            metadata=dict(interaction.metadata),
        )

    async def _score_via_reward_fn(
        self,
        environment: WebAgentEnvironment | None,
        env_extra: dict[str, Any],
        interaction,
    ) -> RewardResult:
        if environment is None:
            return RewardResult(reward=0.0, correct=False, error="no_environment")
        meta = env_extra.get("task_meta") or {}
        extra: dict[str, Any] = {"task_id": str(meta.get("task_id") or "")}
        if self._sft_state_aligned:
            # AUTHORITATIVE judge alignment: pass the rollout workspace as
            # task_dir and let osw_judge derive actions/screenshots with the
            # very functions of scripts/eval_with_original_om2w.py (latest
            # final_runs/run_<N>, plain-text bounded action log, run-dir pngs,
            # no final-response append, 0 without judging when no run exists).
            try:
                extra["task_dir"] = str(environment.workspace)
            except Exception:  # noqa: BLE001 - stub envs may lack a workspace
                pass
            actions: list[str] = []
            screenshots: list[str] = []
            final_response = ""
        else:
            actions = environment.actions_history()
            screenshots = environment.screenshot_paths()
            final_response = str(interaction.metadata.get("final_response", "") or "")
        return await self._reward_fn.score(
            task=str(meta.get("task") or ""),
            start_url=str(meta.get("start_url") or ""),
            actions=actions,
            thoughts=[],  # the qwen35 parser strips <think> blocks already
            screenshot_paths=screenshots,
            final_response=final_response,
            extra=extra,
        )

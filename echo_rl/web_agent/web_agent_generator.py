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
        url = f"http://{host}:{port}/chat/completions"
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
            extra={"task_id": str(meta.get("task_id") or "")},
        )

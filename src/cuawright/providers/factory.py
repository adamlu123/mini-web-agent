from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cuawright.artifacts import TraceWriter
from cuawright.models.anthropic_client import (
    ClaudeClientConfig,
    ClaudeMessagesClient,
    read_api_key,
)
from cuawright.models.anthropic_cost import (
    AnthropicCostTracker,
    CostAccountingError,
    pricing_for_model,
)
from cuawright.models.anthropic_messages import (
    ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL,
    ANTHROPIC_DEFAULT_MAX_REQUEST_MB,
    ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES,
    CLAUDE_PARALLEL_TOOL_PROMPT,
    AnthropicMessagesAdapter,
    ClaudeMessagesThread,
    anthropic_request_limit_bytes,
)
from cuawright.models.openai_responses import (
    OpenAIResponsesAdapter,
    ResponsesThread,
    create_responses_client,
)
from cuawright.models.openrouter_cost import OpenRouterCostTracker
from cuawright.runtime.contracts import TurnProvider

ProviderName = Literal["openai", "openrouter-anthropic", "anthropic"]
CostTracker = OpenRouterCostTracker | AnthropicCostTracker


def parallel_tool_suffix(provider: ProviderName) -> str | None:
    if provider in {"anthropic", "openrouter-anthropic"}:
        return CLAUDE_PARALLEL_TOOL_PROMPT
    if provider == "openai":
        return None
    raise ValueError(f"Unsupported model_provider: {provider!r}")


def create_turn_provider(
    provider: ProviderName,
    thread: object,
) -> TurnProvider:
    if provider in {"anthropic", "openrouter-anthropic"}:
        return AnthropicMessagesAdapter(thread)
    if provider == "openai":
        return OpenAIResponsesAdapter(thread)
    raise ValueError(f"Unsupported model_provider: {provider!r}")


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: ProviderName
    api_key_file: str
    base_url: str
    model: str
    reasoning_effort: str
    max_dollars: float | None = None
    infrastructure: str | None = "aiservices"
    openrouter_provider_preferences: Mapping[str, Any] = field(default_factory=dict)
    anthropic_min_recent_images: int = ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES
    anthropic_image_prune_interval: int = ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL
    anthropic_max_request_mb: float = ANTHROPIC_DEFAULT_MAX_REQUEST_MB
    anthropic_max_output_tokens: int | None = None
    openai_policy_feedback_max_retries: int = 0

    def __post_init__(self) -> None:
        if self.name not in {"openai", "openrouter-anthropic", "anthropic"}:
            raise ValueError(f"Unsupported model_provider: {self.name!r}")
        preferences = dict(self.openrouter_provider_preferences)
        if preferences and self.name != "openrouter-anthropic":
            raise ValueError(
                "openrouter_provider_preferences require "
                "model_provider='openrouter-anthropic'"
            )
        if self.name == "anthropic":
            try:
                pricing_for_model(self.model)
            except CostAccountingError as exc:
                raise ValueError(
                    "Direct Anthropic cost accounting has no pricing for "
                    f"{self.model!r}"
                ) from exc
        anthropic_request_limit_bytes(
            self.anthropic_min_recent_images,
            self.anthropic_image_prune_interval,
            self.anthropic_max_request_mb,
        )
        if (
            self.anthropic_max_output_tokens is not None
            and self.anthropic_max_output_tokens < 1
        ):
            raise ValueError("anthropic_max_output_tokens must be positive")
        if (
            isinstance(self.openai_policy_feedback_max_retries, bool)
            or not isinstance(self.openai_policy_feedback_max_retries, int)
            or self.openai_policy_feedback_max_retries < 0
        ):
            raise ValueError("openai_policy_feedback_max_retries must be nonnegative")
        if self.openai_policy_feedback_max_retries and self.name != "openai":
            raise ValueError(
                "openai_policy_feedback_max_retries requires model_provider='openai'"
            )
        object.__setattr__(self, "openrouter_provider_preferences", preferences)


@dataclass(slots=True)
class ProviderRuntime:
    spec: ProviderSpec
    api_key: str
    client: Any
    cost_tracker: CostTracker | None = None
    claude_client: ClaudeMessagesClient | None = None
    async_runner: asyncio.Runner | None = None
    _closed: bool = False

    @property
    def parallel_tool_suffix(self) -> str | None:
        if (
            self.claude_client is not None
            and self.claude_client.config.disable_parallel_tool_use
        ):
            return None
        return parallel_tool_suffix(self.spec.name)

    def new_thread(
        self,
        *,
        trace: TraceWriter,
        role: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> object:
        if self.spec.name in {"openrouter-anthropic", "anthropic"}:
            if self.claude_client is None or self.cost_tracker is None:
                raise RuntimeError("Claude client was not initialized")
            return ClaudeMessagesThread(
                client=self.claude_client,
                trace=trace,
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                max_output_tokens=(
                    self.spec.anthropic_max_output_tokens or max_output_tokens
                ),
                cost_tracker=self.cost_tracker,
                provider=self.spec.name,
                async_runner=self.async_runner,
                provider_preferences=(
                    self.spec.openrouter_provider_preferences
                    if self.spec.name == "openrouter-anthropic"
                    else None
                ),
                compaction_reasoning_effort=(
                    "low" if self.spec.model.startswith("claude-fable-5-1") else None
                ),
                anthropic_min_recent_images=self.spec.anthropic_min_recent_images,
                anthropic_image_prune_interval=(
                    self.spec.anthropic_image_prune_interval
                ),
                anthropic_max_request_mb=self.spec.anthropic_max_request_mb,
            )
        return ResponsesThread(
            client=self.client,
            trace=trace,
            model=self.spec.model,
            reasoning_effort=self.spec.reasoning_effort,
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            max_output_tokens=max_output_tokens,
            base_url=self.spec.base_url,
            infrastructure=self.spec.infrastructure,
            policy_feedback_max_retries=(self.spec.openai_policy_feedback_max_retries),
        )

    def turn_provider(self, thread: object) -> TurnProvider:
        return create_turn_provider(self.spec.name, thread)

    def cost_snapshot(self) -> dict[str, Any] | None:
        if self.cost_tracker is None:
            return None
        return self.cost_tracker.snapshot()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.claude_client is None:
            return
        if self.async_runner is None:
            asyncio.run(self.claude_client.close())
        else:
            self.async_runner.run(self.claude_client.close())
            self.async_runner.close()


def create_provider_runtime(
    spec: ProviderSpec,
    *,
    run_dir: Path,
) -> ProviderRuntime:
    if spec.name == "openrouter-anthropic":
        api_key = read_api_key(spec.api_key_file)
        cost_tracker = OpenRouterCostTracker(
            api_key=api_key,
            max_dollars=spec.max_dollars,
            artifact_path=run_dir / "openrouter-cost.json",
        )
        async_runner = asyncio.Runner()
        claude_client = ClaudeMessagesClient(
            api_key=api_key,
            config=ClaudeClientConfig(
                model=spec.model,
                effort=spec.reasoning_effort,
                disable_parallel_tool_use=False,
                max_retries=4,
            ),
            base_url=spec.base_url,
            default_headers={"Authorization": "******"},
        )
        return ProviderRuntime(
            spec=spec,
            api_key=api_key,
            client=claude_client,
            cost_tracker=cost_tracker,
            claude_client=claude_client,
            async_runner=async_runner,
        )
    if spec.name == "anthropic":
        api_key = read_api_key(spec.api_key_file)
        foundry = ".services.ai.azure.com/anthropic" in spec.base_url
        cost_tracker = AnthropicCostTracker(
            max_dollars=spec.max_dollars,
            artifact_path=run_dir / "anthropic-cost.json",
            pricing_model=spec.model,
        )
        async_runner = asyncio.Runner()
        claude_client = ClaudeMessagesClient(
            api_key=api_key,
            config=ClaudeClientConfig(
                model=spec.model,
                effort=spec.reasoning_effort,
                server_compaction=not foundry,
                thinking_prefix_mismatch_behavior=("drop_block" if foundry else None),
                disable_parallel_tool_use=foundry,
                max_retries=4,
                policy_feedback_max_retries=(
                    10 if foundry and "fable" in spec.model.lower() else 0
                ),
            ),
            base_url=spec.base_url,
        )
        return ProviderRuntime(
            spec=spec,
            api_key=api_key,
            client=claude_client,
            cost_tracker=cost_tracker,
            claude_client=claude_client,
            async_runner=async_runner,
        )

    api_key = Path(spec.api_key_file).read_text(encoding="utf-8").strip()
    client = create_responses_client(
        api_key=api_key,
        base_url=spec.base_url,
    )
    return ProviderRuntime(
        spec=spec,
        api_key=api_key,
        client=client,
    )

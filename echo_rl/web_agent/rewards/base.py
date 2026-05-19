from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class RewardResult:
    """Per-rollout reward signal."""

    reward: float
    correct: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRewardFn(Protocol):
    """Reward functions take a finished rollout and return a scalar.

    A rollout is summarized as:
      - ``task``: the natural-language goal
      - ``start_url``: the seed URL
      - ``actions``: shell commands the agent ran, in order (e.g. ``web goto ...``)
      - ``thoughts``: per-step thinking strings (optional, may be empty)
      - ``screenshot_paths``: per-step PNG paths the env wrote
      - ``final_response``: the agent's text output, if any
      - ``extra``: env-specific metadata (page url, console, etc.)
    """

    async def score(
        self,
        *,
        task: str,
        start_url: str,
        actions: list[str],
        thoughts: list[str],
        screenshot_paths: list[str],
        final_response: str = "",
        extra: dict[str, Any] | None = None,
    ) -> RewardResult: ...


def build_reward_fn(spec: dict[str, Any] | None) -> BaseRewardFn:
    """Build a reward callable from a dict spec.

    The spec shape is intentionally minimal so it can live in a yaml config::

        reward:
          name: osw_judge          # or "deterministic"
          judge_model: o4-mini
          judge_gateway_endpoint: http://gateway.phyagi.net/api/responses
          score_threshold: 3

    Unknown keys are forwarded to the reward's constructor so it stays easy to
    extend without touching the config plumbing.
    """
    spec = dict(spec or {})
    name = spec.pop("name", "osw_judge")
    if name == "osw_judge":
        from .osw_judge import OSWJudgeReward

        return OSWJudgeReward(**spec)
    if name == "deterministic":
        from .deterministic import DeterministicReward

        return DeterministicReward(**spec)
    raise ValueError(f"Unknown reward name {name!r}. Known: osw_judge, deterministic.")

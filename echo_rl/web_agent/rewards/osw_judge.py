from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from .base import RewardResult


class OSWJudgeReward:
    """Reward backed by the upstream Online-Mind2Web web judge.

    Loads ``om2w_judge`` from the mini-web-agent repo at runtime, so we don't
    duplicate the prompts here. The judge produces a binary success/failure
    label per rollout; we expose it as ``reward = 1.0`` for success, else
    ``0.0``. Override ``success_reward``/``failure_reward`` to shape it.

    The judge model is called via ``om2w_judge.utils.OpenaiEngine``. The
    phyagi gateway keys / endpoint that this used to route through are dead
    (April 2026), and ``OpenaiEngine`` discards ``endpoint_target_uri``
    anyway, so we just talk to ``api.openai.com`` directly with an
    ``sk-...`` key. API key resolution order: explicit ``api_key`` arg →
    ``OPENAI_API_BACKUP_KEY`` env (the working ``sk-proj-...`` that ships
    in the host's environment, intentionally checked first because the
    legacy ``OPENAI_API_KEY`` in shared cred.sh files is a stale phyagi
    token that 401s on api.openai.com) → ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        *,
        judge_model: str = "o4-mini",
        judge_gateway_endpoint: str = "",
        api_key: str | None = None,
        score_threshold: int = 3,
        mini_web_agent_root: str | None = None,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        max_new_tokens: int = 8192,
    ) -> None:
        self.judge_model = judge_model
        self.judge_gateway_endpoint = judge_gateway_endpoint or os.environ.get(
            "OPENAI_GATEWAY_ENDPOINT", ""
        )
        self.api_key = (
            api_key
            or os.environ.get("OPENAI_API_BACKUP_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.score_threshold = int(score_threshold)
        self.mini_web_agent_root = (
            mini_web_agent_root or os.environ.get("MINI_WEB_AGENT_ROOT")
            or "/home/luyadong/sandbox/mini-web-agent"
        )
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.max_new_tokens = int(max_new_tokens)

    def _import_judge(self):
        root = Path(self.mini_web_agent_root).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"mini_web_agent_root does not exist: {root}. Set MINI_WEB_AGENT_ROOT or pass "
                "mini_web_agent_root in the reward config."
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from om2w_judge.methods.webjudge_online_mind2web import WebJudge_Online_Mind2Web_eval  # type: ignore
        from om2w_judge.utils import OpenaiEngine, extract_predication  # type: ignore

        return WebJudge_Online_Mind2Web_eval, OpenaiEngine, extract_predication

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
    ) -> RewardResult:
        del start_url, thoughts  # unused; included to satisfy BaseRewardFn
        existing = [p for p in screenshot_paths if p and Path(p).exists()]
        if not existing:
            return RewardResult(
                reward=self.failure_reward,
                correct=False,
                error="no_screenshots",
                metadata={"judge_model": self.judge_model},
            )

        try:
            WebJudge, OpenaiEngine, extract_predication = self._import_judge()
        except Exception as exc:
            return RewardResult(
                reward=self.failure_reward,
                correct=False,
                error=f"judge_import_failed: {exc}",
                metadata={"judge_model": self.judge_model},
            )

        actions_for_judge = list(actions)
        if final_response:
            actions_for_judge.append(f"Final response: {final_response}")

        def _run_judge() -> tuple[int | None, str, list[Any], str]:
            engine = OpenaiEngine(
                model=self.judge_model,
                api_key=self.api_key,
                endpoint_target_uri=self.judge_gateway_endpoint,
            )
            messages, _text, _system_msg, record, key_points = asyncio.run(
                WebJudge(task, actions_for_judge, existing, engine, self.score_threshold)
            )
            response_text = engine.generate(messages, max_new_tokens=self.max_new_tokens)[0]
            predicted_label = extract_predication(response_text, "WebJudge_Online_Mind2Web_eval")
            return predicted_label, response_text, list(record or []), str(key_points or "")

        try:
            predicted_label, response_text, record, key_points = await asyncio.to_thread(_run_judge)
        except Exception as exc:
            return RewardResult(
                reward=self.failure_reward,
                correct=False,
                error=f"judge_call_failed: {exc}",
                metadata={"judge_model": self.judge_model},
            )

        correct = predicted_label == 1
        reward = self.success_reward if correct else self.failure_reward
        return RewardResult(
            reward=reward,
            correct=correct,
            metadata={
                "judge_model": self.judge_model,
                "predicted_label": predicted_label,
                "response": response_text,
                "key_points": key_points,
                "image_judge_record": record,
                "score_threshold": self.score_threshold,
            },
        )

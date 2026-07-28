from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from .base import RewardResult

_JUDGE_MODE = "WebJudge_Online_Mind2Web_eval"


class OSWJudgeReward:
    """Reward backed by the AUTHORITATIVE eval script
    ``scripts/eval_with_original_om2w.py`` (mini-web-agent repo).

    The script's per-task pipeline is reused function-for-function so RL
    rewards match eval scores exactly:

      - ``resolve_latest_run_dir``: highest ``final_runs/run_<N>``; a rollout
        with NO final run scores 0 without calling the judge (same as eval).
      - ``load_actions(log, plain_text=True)`` + ``bound_action_history``:
        every non-empty line of the run's final_script_log.txt except the
        "final response/answer:" line, bounded to 500 lines / 60k chars.
      - ``load_screenshots``: every png in the run's screenshots/ dir,
        final_execution_<N> numeric-first ordering.
      - ``robust_webjudge_online_mind2web_eval``: upstream OM2W judge prompts
        with per-image parse retries; judges even with zero screenshots.
      - ``extract_predication(response, "WebJudge_Online_Mind2Web_eval")``.

    Transport: the judge model is called via om2w_judge_sandbox's
    ``OpenaiEngine`` because it supports routing through the phyagi gateway
    (``endpoint_target_uri``); it exposes the same ``generate`` interface
    (o-series -> max_completion_tokens) as the engine the script constructs,
    so ONLY the transport differs, never the judge semantics.

    Key resolution mirrors the eval harness's _resolve_judge_api_key: with a
    gateway endpoint prefer OM2W_JUDGE_API_KEY / PHYAGI_API_KEY /
    OPENAI_GATEWAY_API_KEY, else the direct-OpenAI keys.
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
        if api_key:
            self.api_key = api_key
        elif self.judge_gateway_endpoint:
            self.api_key = (
                os.environ.get("OM2W_JUDGE_API_KEY")
                or os.environ.get("PHYAGI_API_KEY")
                or os.environ.get("OPENAI_GATEWAY_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        else:
            self.api_key = (
                os.environ.get("OM2W_JUDGE_API_KEY")
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
        self._eval_mod = None
        self._engine_cls = None

    def _import_judge(self):
        if self._eval_mod is not None:
            return self._eval_mod, self._engine_cls
        root = Path(self.mini_web_agent_root).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"mini_web_agent_root does not exist: {root}. Set MINI_WEB_AGENT_ROOT or pass "
                "mini_web_agent_root in the reward config."
            )
        # The eval script hardcodes /home/luyadong paths and relies on flat
        # `from methods import ...` / `from utils import ...` imports inside
        # om2w_judge — put THIS checkout's root and om2w_judge dir on sys.path
        # first so everything resolves against the uploaded tree.
        for p in (str(root), str(root / "om2w_judge")):
            if p not in sys.path:
                sys.path.insert(0, p)
        script = root / "scripts" / "eval_with_original_om2w.py"
        if not script.is_file():
            raise RuntimeError(f"authoritative eval script not found: {script}")
        spec = importlib.util.spec_from_file_location("eval_with_original_om2w", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules["eval_with_original_om2w"] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        from om2w_judge_sandbox.utils import OpenaiEngine  # type: ignore

        self._eval_mod = module
        self._engine_cls = OpenaiEngine
        return module, OpenaiEngine

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
        try:
            mod, engine_cls = self._import_judge()
        except Exception as exc:
            return RewardResult(
                reward=self.failure_reward,
                correct=False,
                error=f"judge_import_failed: {exc}",
                metadata={"judge_model": self.judge_model},
            )

        task_dir = str((extra or {}).get("task_dir") or "")
        judge_actions: list[str]
        judge_shots: list[str]
        run_dir_str = ""
        if task_dir:
            # SFT-aligned rollouts: derive inputs exactly like the eval script.
            run_dir = mod.resolve_latest_run_dir(Path(task_dir))
            if run_dir is None:
                # Mirror auto_eval: no final_runs artifacts -> predicted_label 0
                # without calling the judge at all.
                return RewardResult(
                    reward=self.failure_reward,
                    correct=False,
                    error=None,
                    metadata={
                        "judge_model": self.judge_model,
                        "predicted_label": 0,
                        "response": "No final_runs/run_* artifacts were available.",
                        "action_history_source": "final_script_log",
                    },
                )
            run_dir_str = str(run_dir)
            judge_actions = mod.bound_action_history(
                mod.load_actions(run_dir / "final_script_log.txt", plain_text=True)
            )
            judge_shots = mod.load_screenshots(run_dir / "screenshots")
        else:
            # Legacy (non-SFT) path: caller-provided inputs, final response
            # appended like the pre-script reward did.
            judge_actions = list(actions)
            if final_response:
                judge_actions.append(f"Final response: {final_response}")
            judge_shots = [p for p in screenshot_paths if p and Path(p).exists()]

        def _run_judge() -> tuple[int | None, str, list[Any], str]:
            engine = engine_cls(
                model=self.judge_model,
                api_key=self.api_key,
                endpoint_target_uri=self.judge_gateway_endpoint,
            )
            messages, _text, _system_msg, record, key_points = asyncio.run(
                mod.robust_webjudge_online_mind2web_eval(
                    task, judge_actions, judge_shots, engine, self.score_threshold
                )
            )
            response_text = engine.generate(messages, max_new_tokens=self.max_new_tokens)[0]
            predicted_label = mod.extract_predication(response_text, _JUDGE_MODE)
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
                "final_run_dir": run_dir_str,
                "num_judge_actions": len(judge_actions),
                "num_judge_screenshots": len(judge_shots),
            },
        )

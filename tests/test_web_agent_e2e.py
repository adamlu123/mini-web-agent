"""End-to-end smoke test for the web-agent rollout loop.

Validates the three moving parts without needing SkyRL / vLLM:

1. ``WebAgentTaskDataset`` loads om2w tasks and produces qwen35 prompts.
2. ``WebAgentEnvironment`` accepts the same shell commands the model would
   emit and returns observation text + screenshots.
3. ``build_reward_fn`` produces a ``RewardResult`` from those artifacts.

The model is replaced by a scripted policy that walks the agent through a few
deterministic commands. We default to the in-memory stub browser
(``stub=True``) and the ``deterministic`` reward (no network), so the test
runs without Playwright or the judge API. Setting ``WEB_AGENT_E2E_REAL=1``
flips both back on for an interactive end-to-end run.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Make the worktree importable when running directly via pytest from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echo_rl.web_agent.rewards import build_reward_fn  # noqa: E402
from echo_rl.web_agent.web_environment import (  # noqa: E402
    WebAgentEnvironment,
    WebAgentEnvironmentConfig,
)

_USE_REAL = os.environ.get("WEB_AGENT_E2E_REAL", "").lower() in {"1", "true", "yes"}


def _om2w_fixture_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "om2w_smoke.json"


def _ensure_om2w_fixture() -> Path:
    """Drop a tiny fixed task list into tests/data/ for offline use."""
    p = _om2w_fixture_path()
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "smoke-001",
                        "task": "Open example.com and verify the title contains 'Example'.",
                        "website_url": "https://example.com",
                        "level": "easy",
                        "reference_length": 1,
                    }
                ]
            },
            indent=2,
        )
    )
    return p


_SCRIPTED_COMMANDS = [
    "web goto https://example.com",
    "web title",
    "web snapshot 800",
    "echo 'answer: page loaded — title contains Example Domain'",
]


async def _drive(env: WebAgentEnvironment, commands: list[str]) -> list[str]:
    outputs: list[str] = []
    for cmd in commands:
        result = await env.exec(cmd, timeout=30)
        body = (result.stdout or "").strip()
        if result.stderr:
            body = f"{body}\n[stderr] {result.stderr.strip()}".strip()
        outputs.append(f"$ {cmd}\n{body}")
    return outputs


def _reward_spec_for_smoke() -> dict:
    if _USE_REAL:
        return {
            "name": "osw_judge",
            "judge_model": os.environ.get("WEB_AGENT_JUDGE_MODEL", "o4-mini"),
            "judge_gateway_endpoint": os.environ.get(
                "OPENAI_GATEWAY_ENDPOINT", "http://gateway.phyagi.net/api/responses"
            ),
            "score_threshold": 3,
        }
    return {
        "name": "deterministic",
        "expected_substrings": ["example", "page loaded"],
    }


async def _async_test_dataset_loads_and_renders_prompt() -> None:
    from transformers import AutoTokenizer

    from echo_rl.web_agent.dataset import WebAgentTaskDataset

    fixture = _ensure_om2w_fixture()
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("WEB_AGENT_TOKENIZER", "Qwen/Qwen2.5-1.5B-Instruct"),
        trust_remote_code=True,
    )
    ds = WebAgentTaskDataset(
        data_files=[str(fixture)],
        tokenizer=tokenizer,
        max_prompt_length=4096,
        parser_name="qwen35",
        num_workers=1,
    )
    assert len(ds) == 1
    item = ds[0]
    assert item["env_extras"]["task_meta"]["start_url"].startswith("https://example.com")
    assert any("web goto" in msg["content"] for msg in item["prompt"])


async def _async_test_environment_runs_scripted_rollout(tmp_path: Path) -> None:
    cfg = WebAgentEnvironmentConfig(
        task_id="smoke-001",
        task="Open example.com and verify the title contains 'Example'.",
        start_url="https://example.com",
        workspace_root=tmp_path,
        stub=not _USE_REAL,
    )
    env = WebAgentEnvironment(cfg)
    await env.setup()
    try:
        outputs = await _drive(env, _SCRIPTED_COMMANDS)
        assert len(outputs) == len(_SCRIPTED_COMMANDS)
        # The first command should leave us with a non-empty URL line.
        assert "URL:" in outputs[0]
        # We should have one screenshot per turn (plus the "start" frame).
        screenshots = env.screenshot_paths()
        assert len(screenshots) == len(_SCRIPTED_COMMANDS)
        for path in screenshots:
            assert Path(path).exists()
        actions = env.actions_history()
        assert all(action.startswith("step ") for action in actions)
    finally:
        await env.cleanup()


async def _async_test_reward_scores_finished_rollout(tmp_path: Path) -> None:
    cfg = WebAgentEnvironmentConfig(
        task_id="smoke-001",
        task="Open example.com and verify the title contains 'Example'.",
        start_url="https://example.com",
        workspace_root=tmp_path,
        stub=not _USE_REAL,
    )
    env = WebAgentEnvironment(cfg)
    await env.setup()
    try:
        await _drive(env, _SCRIPTED_COMMANDS)
        reward_fn = build_reward_fn(_reward_spec_for_smoke())
        result = await reward_fn.score(
            task=cfg.task,
            start_url=cfg.start_url,
            actions=env.actions_history(),
            thoughts=[],
            screenshot_paths=env.screenshot_paths(),
            final_response="answer: page loaded — title contains Example Domain",
            extra={"task_id": cfg.task_id},
        )
    finally:
        await env.cleanup()

    # With the deterministic reward the smoke task is solvable; with the real
    # judge the verdict may differ but we only require that .score returns
    # a structured RewardResult.
    assert result.reward in {0.0, 1.0}
    if not _USE_REAL:
        assert result.correct is True
        assert result.reward == 1.0


def test_dataset_loads_and_renders_prompt() -> None:
    asyncio.run(_async_test_dataset_loads_and_renders_prompt())


def test_environment_runs_scripted_rollout(tmp_path: Path) -> None:
    asyncio.run(_async_test_environment_runs_scripted_rollout(tmp_path))


def test_reward_scores_finished_rollout(tmp_path: Path) -> None:
    asyncio.run(_async_test_reward_scores_finished_rollout(tmp_path))


if __name__ == "__main__":
    # Allow `python tests/test_web_agent_e2e.py` for a quick local check.
    async def _amain() -> None:
        await _async_test_dataset_loads_and_renders_prompt()
        tmp = Path(os.environ.get("WEB_AGENT_E2E_TMP", "/tmp/web_agent_e2e"))
        tmp.mkdir(parents=True, exist_ok=True)
        await _async_test_environment_runs_scripted_rollout(tmp)
        await _async_test_reward_scores_finished_rollout(tmp)
        print("OK")

    asyncio.run(_amain())

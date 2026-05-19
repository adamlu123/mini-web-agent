"""Run one web-agent rollout end-to-end on a single om2w-style task.

Useful when you want to verify the env + reward outside of SkyRL but with a
real Playwright browser and a real judge call.

Usage::

    # Stubbed env + deterministic reward (no internet, no GPU, no judge):
    python -m echo_rl.web_agent.scripts.run_smoke_rollout --stub --deterministic

    # Real browser + real judge (requires OPENAI_GATEWAY_API_KEY and a Playwright install):
    python -m echo_rl.web_agent.scripts.run_smoke_rollout \\
        --task-id smoke-001 \\
        --task "Open example.com and verify the title contains 'Example'." \\
        --start-url https://example.com

The script feeds the env a hard-coded scripted "policy" (just a list of
commands) — it doesn't call any model. That's enough to confirm the rollout
loop and reward function are wired up correctly before turning on training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echo_rl.web_agent.rewards import build_reward_fn  # noqa: E402
from echo_rl.web_agent.web_environment import (  # noqa: E402
    WebAgentEnvironment,
    WebAgentEnvironmentConfig,
)


DEFAULT_COMMANDS = [
    "web goto {start_url}",
    "web title",
    "web snapshot 800",
    "echo 'answer: page loaded'",
]


async def _run(args: argparse.Namespace) -> int:
    cfg = WebAgentEnvironmentConfig(
        task_id=args.task_id,
        task=args.task,
        start_url=args.start_url,
        workspace_root=Path(args.workspace_root),
        headless=args.headless,
        stub=args.stub,
    )
    env = WebAgentEnvironment(cfg)
    await env.setup()
    try:
        commands = [c.format(start_url=args.start_url) for c in args.commands or DEFAULT_COMMANDS]
        for cmd in commands:
            result = await env.exec(cmd, timeout=args.command_timeout)
            print(f"$ {cmd}\n  rc={result.return_code}\n  stdout={result.stdout[:200]!r}")
            if result.stderr:
                print(f"  stderr={result.stderr[:200]!r}")

        reward_spec = (
            {"name": "deterministic", "expected_substrings": ["page loaded"]}
            if args.deterministic
            else {
                "name": "osw_judge",
                "judge_model": args.judge_model,
                "judge_gateway_endpoint": args.judge_endpoint,
                "score_threshold": args.score_threshold,
            }
        )
        reward_fn = build_reward_fn(reward_spec)
        result = await reward_fn.score(
            task=cfg.task,
            start_url=cfg.start_url,
            actions=env.actions_history(),
            thoughts=[],
            screenshot_paths=env.screenshot_paths(),
            final_response="answer: page loaded",
            extra={"task_id": cfg.task_id},
        )
        print(json.dumps({
            "reward": result.reward,
            "correct": result.correct,
            "error": result.error,
            "metadata": {k: v for k, v in result.metadata.items() if k != "image_judge_record"},
        }, indent=2, default=str))
        return 0 if not result.error else 1
    finally:
        await env.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="smoke-001")
    parser.add_argument("--task", default="Open example.com and verify the title contains 'Example'.")
    parser.add_argument("--start-url", default="https://example.com")
    parser.add_argument("--workspace-root", default="/tmp/web_agent_smoke")
    parser.add_argument("--stub", action="store_true", help="Use the in-memory stub browser.")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic reward.")
    parser.add_argument("--judge-model", default="o4-mini")
    parser.add_argument(
        "--judge-endpoint",
        default="http://gateway.phyagi.net/api/responses",
    )
    parser.add_argument("--score-threshold", type=int, default=3)
    parser.add_argument("--command-timeout", type=float, default=30.0)
    parser.add_argument(
        "--commands", nargs="*", default=None,
        help="Optional override list of commands (each may use {start_url} placeholder).",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cuawright.models.anthropic_client import (
    CLAUDE_OPUS_5,
    ClaudeClientConfig,
    ClaudeMessagesClient,
    dump_private_trace,
)


ADD_TOOL = {
    "name": "add_numbers",
    "description": "Add two integers and return their exact sum.",
    "input_schema": {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = (
    "You are validating an Anthropic Messages API client. "
    "Follow the user's tool-use request precisely and stay concise."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise the first-class Claude Opus 5 Messages client"
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("~/.claude-key").expanduser(),
    )
    parser.add_argument("--model", default=CLAUDE_OPUS_5)
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="max",
    )
    parser.add_argument("--max-tokens", type=int, default=4_096)
    parser.add_argument(
        "--prompt",
        default=(
            "Use add_numbers exactly once to add 37 and 58. "
            "Then answer with the resulting integer and one short sentence "
            "confirming that the tool result was used."
        ),
    )
    parser.add_argument(
        "--private-trace",
        type=Path,
        help=(
            "Optional path for the full resumable transcript, including "
            "encrypted thinking signatures. Written with mode 0600."
        ),
    )
    return parser.parse_args()


def add_numbers(arguments: dict[str, object]) -> str:
    return str(int(arguments["a"]) + int(arguments["b"]))


async def main() -> None:
    args = parse_args()
    config = ClaudeClientConfig(
        model=args.model,
        effort=args.effort,
        max_tokens=args.max_tokens,
    )
    async with ClaudeMessagesClient.from_key_file(
        args.api_key_file,
        config=config,
    ) as client:
        counted_tokens = await client.count_tokens(
            messages=[{"role": "user", "content": args.prompt}],
            system=SYSTEM_PROMPT,
            tools=[ADD_TOOL],
        )
        result = await client.run_tool_loop(
            user_prompt=args.prompt,
            system=SYSTEM_PROMPT,
            tools=[ADD_TOOL],
            tool_handlers={"add_numbers": add_numbers},
        )

    summary = result.public_summary()
    summary["counted_initial_input_tokens"] = counted_tokens
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.private_trace:
        dump_private_trace(result, args.private_trace)


if __name__ == "__main__":
    asyncio.run(main())

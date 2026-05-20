"""image_qa — ask the current policy a grounded question about screenshots.

Calls the policy model (the one being trained) through SkyRL's HTTP
endpoint, so this is a SELF-QA: the policy reads its own rollout's images
and answers the question. Useful for the agent to ground its reasoning on
visual evidence without invoking a separate judge model.

Usage from inside a `python -c '...'` snippet or a shell command:

    image_qa --image screenshots/step_0001.png \\
        --image screenshots/step_0002.png \\
        --question "Is the BMW filter chip visibly selected?"

The reply is printed to stdout as JSON with ``answer`` and ``confidence``
fields (the policy is asked to wrap its answer that way). Both fields fall
back to free-form text if the model didn't honour the JSON wrapper.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .policy_chat import chat, encode_image

_SYSTEM = (
    "You are a careful visual evaluator. Look at the supplied screenshot(s) and "
    "answer the user's question using ONLY visible evidence. If you cannot tell "
    "from the screenshots, say so. Reply with ONE JSON object on a single line:"
    ' {"answer": "<short answer>", "evidence": "<which screenshot(s) and what '
    'is visible>", "confidence": "low|medium|high"}. Do not wrap the JSON in '
    "prose or code fences."
)


def _parse_reply(raw: str) -> dict:
    raw = (raw or "").strip()
    # If the model produced a clean JSON object, use it.
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try to extract the first {...} blob.
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"answer": raw, "evidence": "", "confidence": "unknown", "raw": raw}


def run(images: list[str], question: str, *, max_tokens: int = 768) -> dict:
    if not images:
        raise ValueError("at least one --image is required")
    if not question:
        raise ValueError("--question is required")
    user_parts: list[dict] = [
        {"type": "text", "text": f"Question: {question}\n\nAttached screenshots: {len(images)}"},
    ]
    for path in images:
        user_parts.append(encode_image(path))
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user_parts},
    ]
    reply = chat(messages, max_tokens=max_tokens, temperature=0.0)
    return _parse_reply(reply)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image_qa", description=__doc__)
    parser.add_argument("--image", action="append", required=True, help="path to a screenshot (repeatable)")
    parser.add_argument("--question", required=True, help="the question to ground in the screenshots")
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args(argv)

    for p in args.image:
        if not Path(p).is_file():
            print(json.dumps({"error": f"image not found: {p}"}), file=sys.stderr)
            return 2

    try:
        result = run(args.image, args.question, max_tokens=args.max_tokens)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""self_reflection — ask the current policy whether the rollout so far has
solved the task. Mirrors webwright's self_reflection in spirit but uses the
*current* policy as the judge (self-reflection, not external grading), so
the model supervises itself during training.

Two stages:

1. **Per-image scoring.** For each screenshot the policy gets the task and
   the full critical-point list and returns ``Reasoning: ...`` + ``Score: N``
   (1-5) on its own.
2. **Aggregated verdict.** All per-image reasonings + the action log are
   stuffed into one prompt; the policy must end with ``Status: success`` or
   ``Status: failure``.

Both stages use ``echo_rl.web_agent.tools.policy_chat`` so the request hits
the same model that's being trained.

Outputs a JSON file (and prints it) with:

    {
      "predicted_label": 1 | 0 | null,
      "verdict": "success" | "failure" | "unparsed",
      "image_records": [...],
      "final_response": "<text the policy produced>"
    }

Usage from a python -c '...' snippet or a shell command:

    self_reflection \\
        --task "Find a 2022 Tesla Model 3 on CarMax." \\
        --critical-points "Open CarMax\\nApply Tesla brand filter\\n..." \\
        --action-log final_runs/run_001/final_script_log.txt \\
        --image screenshots/step_0001.png --image screenshots/step_0002.png \\
        --output final_runs/run_001/self_reflection.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .policy_chat import chat, encode_image

_IMAGE_SYSTEM = (
    "You are a harsh visual evaluator scoring evidence for a web-task rollout. "
    "Given the task, the numbered critical-points list, and ONE screenshot, "
    "return EXACTLY two labelled lines and nothing else:\n"
    "Reasoning: <1-2 sentences describing what the screenshot shows and which "
    "critical points it provides evidence for or against>\n"
    "Score: <integer 1-5 — 5 = clearly evidences a critical point, 1 = no relevant evidence>"
)

_FINAL_SYSTEM = (
    "You are an aggregated harsh judge for a web-task rollout. Combine the action "
    "log and the per-image reasonings and decide whether the task is complete. "
    "Begin your reply with a `Thoughts:` block that evaluates every critical "
    "point, then end with EXACTLY ONE line: `Status: success` or `Status: failure`."
)


def _format_critical_points(text: str) -> str:
    lines = [ln.strip() for ln in text.replace("\\n", "\n").splitlines() if ln.strip()]
    if not lines:
        return "(no critical points supplied)"
    numbered = []
    for i, ln in enumerate(lines, 1):
        if re.match(r"^\s*(\d+[.)]|-|\*)\s+", ln):
            numbered.append(ln)
        else:
            numbered.append(f"{i}. {ln}")
    return "\n".join(numbered)


def _score_image(task: str, critical_points: str, image_path: str, *, max_tokens: int = 384) -> dict:
    user = [
        {
            "type": "text",
            "text": (
                f"Task: {task}\n\nCritical points:\n{critical_points}\n\n"
                f"Score the attached screenshot."
            ),
        },
        encode_image(image_path),
    ]
    raw = chat(
        [{"role": "system", "content": _IMAGE_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    score_match = re.search(r"Score\s*:\s*(\d+)", raw)
    reasoning_match = re.search(r"Reasoning\s*:\s*(.+?)(?:\n\s*Score|$)", raw, re.DOTALL)
    score = int(score_match.group(1)) if score_match else 0
    reasoning = reasoning_match.group(1).strip() if reasoning_match else raw.strip()
    return {"image": image_path, "score": score, "reasoning": reasoning, "raw": raw}


def _final_verdict(
    task: str,
    critical_points: str,
    action_log: str,
    image_records: list[dict],
    images: list[str],
    *,
    max_tokens: int = 1024,
) -> tuple[str, str]:
    reasonings = "\n".join(
        f"- ({i + 1}) {rec.get('image', '?')} | Score: {rec.get('score', 0)} | "
        f"{rec.get('reasoning', '')}"
        for i, rec in enumerate(image_records)
    ) or "(no per-image reasonings)"
    user_text = (
        f"Task: {task}\n\nCritical points:\n{critical_points}\n\n"
        f"Action log:\n{action_log or '(empty)'}\n\n"
        f"Per-image reasonings:\n{reasonings}\n\n"
        "Decide whether the rollout completed the task. Remember to end with "
        "`Status: success` or `Status: failure`."
    )
    user: list[dict] = [{"type": "text", "text": user_text}]
    for path in images:
        try:
            user.append(encode_image(path))
        except Exception:
            continue
    raw = chat(
        [{"role": "system", "content": _FINAL_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    match = re.search(r"Status\s*:\s*(success|failure)", raw, re.IGNORECASE)
    verdict = match.group(1).lower() if match else "unparsed"
    return verdict, raw


def run(
    task: str,
    critical_points: str,
    action_log: str,
    images: list[str],
    *,
    image_max_tokens: int = 384,
    final_max_tokens: int = 1024,
) -> dict:
    cps = _format_critical_points(critical_points)
    image_records = [_score_image(task, cps, p, max_tokens=image_max_tokens) for p in images]
    verdict, response = _final_verdict(
        task, cps, action_log, image_records, images, max_tokens=final_max_tokens
    )
    predicted_label = 1 if verdict == "success" else (0 if verdict == "failure" else None)
    return {
        "predicted_label": predicted_label,
        "verdict": verdict,
        "image_records": image_records,
        "final_response": response,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="self_reflection", description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--critical-points", default="", help="newline-separated list (or '\\n' escapes)")
    parser.add_argument("--action-log", default="", help="path to log file OR literal text")
    parser.add_argument("--image", action="append", default=[], help="screenshot path (repeatable)")
    parser.add_argument("--output", default="", help="optional path to write the result JSON")
    parser.add_argument("--image-max-tokens", type=int, default=384)
    parser.add_argument("--final-max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)

    if not args.image:
        print(json.dumps({"error": "at least one --image is required"}), file=sys.stderr)
        return 2

    action_log = args.action_log
    log_path = Path(action_log)
    if action_log and log_path.is_file():
        action_log = log_path.read_text(encoding="utf-8", errors="replace")

    try:
        result = run(
            task=args.task,
            critical_points=args.critical_points,
            action_log=action_log,
            images=args.image,
            image_max_tokens=args.image_max_tokens,
            final_max_tokens=args.final_max_tokens,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        Path(args.output).expanduser().parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).expanduser().write_text(payload)
    return 0 if result.get("predicted_label") == 1 else (1 if result.get("predicted_label") == 0 else 2)


if __name__ == "__main__":
    sys.exit(main())

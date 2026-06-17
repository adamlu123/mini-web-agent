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

try:
    from policy_chat import chat, encode_image
except ImportError:
    from .policy_chat import chat, encode_image
from typing import Any, Dict, List, Optional

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


def _latest_run_dir(base: Path, auto_latest_run: str) -> Path:
    runs_base = base / auto_latest_run
    runs_base.mkdir(parents=True, exist_ok=True)
    candidates = sorted([d for d in runs_base.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not candidates:
        return runs_base / "run_001"
    return candidates[-1]


def _auto_images(run_dir: Path) -> list[str]:
    return sorted(str(p) for p in run_dir.joinpath("screenshots").glob("*.png"))


def _read_action_log(run_dir: Path) -> str:
    log_path = run_dir / "final_script_log.txt"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def _read_critical_points(workspace: Path) -> str:
    plan_path = workspace / "plan.md"
    if not plan_path.exists():
        return ""
    return plan_path.read_text(encoding="utf-8", errors="replace")


def _load_config(path: Path) -> dict:
    if path.suffix.lower() == "_file":
        path = Path(path.parent / (path.name[:-5] + ".json"))
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="replace")

def run_with_config(
    cfg: dict,
    images: list[str],
    action_log: str,
    critical_points: str,
    *,
    image_max_tokens: int = 384,
    final_max_tokens: int = 1024,
) -> dict:
    task = ""
    if "task" in cfg and cfg["task"] is not None:
        task = str(cfg["task"])
    if not task and "task" in cfg:
        task = str(cfg["task"])
    if not task and cfg.get("critical_points"):
        task = "Task inferred from critical points only; verify run artifacts separately."
    if not critical_points and "critical_points" in cfg and cfg["critical_points"] is not None:
        critical_points = str(cfg["critical_points"])

    image_records = [
        _score_image(
            task=task,
            critical_points=critical_points,
            image_path=path,
            max_tokens=image_max_tokens,
        )
        for path in images
    ]
    verdict, raw = _final_verdict(
        task=task,
        critical_points=critical_points,
        action_log=action_log,
        image_records=image_records,
        images=images,
        max_tokens=final_max_tokens,
    )
    return {
        "predicted_label": 1 if verdict == "success" else 0,
        "verdict": verdict,
        "images": images,
        "image_records": image_records,
        "final_prompt": raw,
        "final_response": raw,
    }


def _latest_run_dir(base: Path, auto_latest_run: str) -> Path:
    runs_base = base / auto_latest_run
    runs_base.mkdir(parents=True, exist_ok=True)
    candidates = sorted([d for d in runs_base.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not candidates:
        return runs_base / "run_001"
    return candidates[-1]


def _auto_images(run_dir: Path) -> list[str]:
    return sorted(str(p) for p in run_dir.joinpath("screenshots").glob("*.png"))


def _read_action_log(run_dir: Path) -> str:
    log_path = run_dir / "final_script_log.txt"
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def _read_critical_points(workspace: Path) -> str:
    plan_path = workspace / "plan.md"
    if not plan_path.exists():
        return ""
    return plan_path.read_text(encoding="utf-8", errors="replace")


def _load_config(path: Path) -> dict:
    if path.suffix.lower() == "_file":
        path = Path(path.parent / (path.name[:-5] + ".json"))
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="replace")

def _latest_run_dir(workspace: Path | None, auto_subdir: str) -> Path | None:
    if workspace is None:
        return None
    auto = workspace / auto_subdir
    if not auto.exists():
        return None
    runs = []
    for child in auto.iterdir():
        try:
            runs.append((int(child.name.split("_")[1]), child))
        except Exception:
            pass
    if not runs:
        return None
    return sorted(runs)[-1][1]


def _auto_images(run_dir: Path) -> list[str]:
    out = []
    for ext in [".png", ".jpg", ".jpeg"]:
        for path in sorted((run_dir / "screenshots").glob(f"*{ext}")):
            out.append(str(path))
    return out


def _read_action_log(run_dir: Path) -> str:
    log_path = run_dir / "final_script_log.txt"
    if log_path.exists():
        return log_path.read_text(encoding="utf-8", errors="replace")
    return ""


def run_with_config(
    cfg: dict,
    images: list[str],
    action_log: str,
    critical_points: str,
    *,
    image_max_tokens: int = 384,
    final_max_tokens: int = 1024,
) -> dict:
    task = cfg.get("task") if isinstance(cfg, dict) else ""
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
    # SFT / config contract
    parser.add_argument("--config", default="", help="path to judge_config.json (4 prompts). Enables the SFT contract.")
    parser.add_argument("--workspace-dir", default="", help="workspace root (used to auto-find the latest run + plan.md)")
    parser.add_argument("--auto-latest-run", default="final_runs",
                        help="subdir under --workspace-dir holding run_* folders; the latest run's screenshots are auto-attached")
    # RL contract
    parser.add_argument("--task", default="", help="(RL contract) task description")
    parser.add_argument("--critical-points", default="", help="(RL contract) newline-separated list (or '\\n' escapes)")
    parser.add_argument("--action-log", default="", help="path to a log file OR literal text (overrides auto-detected log)")
    parser.add_argument("--image", action="append", default=[], help="screenshot path (repeatable; overrides auto-attach)")
    parser.add_argument("--output", default="", help="optional path to write the result JSON")
    parser.add_argument("--image-max-tokens", type=int, default=384)
    parser.add_argument("--final-max-tokens", type=int, default=1024)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace_dir).expanduser() if args.workspace_dir else None

    # Resolve the action log (explicit path/text wins; else latest run's log).
    action_log = args.action_log
    if action_log and Path(action_log).is_file():
        action_log = Path(action_log).read_text(encoding="utf-8", errors="replace")

    # Resolve images: explicit --image wins; else auto-attach the latest run.
    images = list(args.image)
    run_dir = _latest_run_dir(workspace, args.auto_latest_run) if workspace else None
    if not images and run_dir is not None:
        images = _auto_images(run_dir)
    if not action_log and run_dir is not None:
        action_log = _read_action_log(run_dir)

    if not images:
        print(json.dumps({"error": "no screenshots: pass --image, or --workspace-dir with a "
                                    "final_runs/run_*/screenshots/*.png"}), file=sys.stderr)
        return 1

    try:
        if args.config:
            cfg = _load_config(Path(args.config).expanduser())
            critical_points = _read_critical_points(workspace) if workspace else ""
            result = run_with_config(
                cfg,
                images=images,
                action_log=action_log,
                critical_points=critical_points,
                image_max_tokens=args.image_max_tokens,
                final_max_tokens=args.final_max_tokens,
            )
        else:
            result = run(
                task=args.task,
                critical_points=args.critical_points,
                action_log=action_log,
                images=images,
                image_max_tokens=args.image_max_tokens,
                final_max_tokens=args.final_max_tokens,
            )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    # 0 on PASS, 1 otherwise (matches the mini-swe-webagent native tool).
    return 0 if result.get("predicted_label") == 1 else 1


if __name__ == "__main__":
    sys.exit(main())


from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


_PLACEHOLDER_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a"
    "0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360606060000000050001a5f64540"
    "0000000049454e44ae426082"
)

_MISSING_HISTORY_MARKERS = (
    "action history is missing",
    "provide the action history",
    "need the action history",
    "provided action history is incomplete",
)


def _load_debug_steps(output_dir: Path) -> list[dict[str, Any]]:
    steps_dir = output_dir / "debug" / "steps"
    rows: list[dict[str, Any]] = []
    if not steps_dir.exists():
        return rows
    for path in sorted(steps_dir.glob("step_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _fallback_actions_and_thoughts(trajectory_path: Path) -> tuple[list[str], list[str]]:
    if not trajectory_path.exists():
        return [], []

    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    actions: list[str] = []
    thoughts: list[str] = []
    for message in data.get("messages", []):
        if message.get("role") != "assistant":
            continue
        extra = message.get("extra", {})
        message_actions = extra.get("actions", [])
        if not message_actions:
            continue
        thoughts.append(str(message.get("content", "")).strip())
        actions.append("\n\n".join(str(action.get("python_code", "")).strip() for action in message_actions).strip())
    return actions, thoughts


def _has_recorded_action_history(row: dict[str, Any]) -> bool:
    action_history = row.get("action_history", [])
    if isinstance(action_history, list):
        return any(str(action).strip() for action in action_history)
    return bool(str(action_history).strip())


def _needs_missing_history_normalization(response: str) -> bool:
    normalized = response.lower()
    return any(marker in normalized for marker in _MISSING_HISTORY_MARKERS)


def _build_missing_history_failure_response(row: dict[str, Any]) -> str:
    details: list[str] = []
    exit_status = str(row.get("exit_status", "")).strip()
    run_exception = str(row.get("run_exception", "")).strip()

    if exit_status:
        details.append(f"exit status {exit_status}")
    if run_exception:
        details.append(f"run exception: {run_exception}")

    suffix = ""
    if details:
        suffix = f" The run ended before any browser actions were recorded ({'; '.join(details)})."

    return (
        "Thoughts: The action history is empty, so there is no evidence that the agent completed the task."
        f"{suffix} Under the evaluation criteria, this must be marked as failure.\n"
        'Status: "failure"'
    )


def _judge_result_file_path(output_dir: Path, judge_model: str) -> Path:
    return output_dir / f"WebJudge_Online_Mind2Web_eval_{judge_model}_score_threshold_3_auto_eval_results.json"


def _ordered_step_stems(output_dir: Path) -> list[str]:
    steps_dir = output_dir / "steps"
    if steps_dir.exists():
        stems = [path.stem for path in sorted(steps_dir.glob("step_*.py"))]
        if stems:
            return stems

    debug_steps_dir = output_dir / "debug" / "steps"
    if debug_steps_dir.exists():
        stems = [path.stem for path in sorted(debug_steps_dir.glob("step_*.json"))]
        if stems:
            return stems

    return [path.stem for path in sorted((output_dir / "screenshots").glob("step_*.png"))]


def _write_placeholder_screenshot(path: Path) -> None:
    path.write_bytes(_PLACEHOLDER_PNG_BYTES)


def normalize_online_mind2web_judge_results(*, result_file: Path) -> int:
    if not result_file.exists():
        return 0

    original_text = result_file.read_text(encoding="utf-8")
    updated_rows: list[str] = []
    replacements = 0

    for line in original_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        evaluation_details = row.get("evaluation_details")
        response = ""
        if isinstance(evaluation_details, dict):
            response = str(evaluation_details.get("response", ""))

        if not _has_recorded_action_history(row) and _needs_missing_history_normalization(response):
            normalized_response = _build_missing_history_failure_response(row)
            row["evaluation_details"] = {
                **(evaluation_details or {}),
                "response": normalized_response,
                "predicted_label": 0,
            }
            row["predicted_label"] = 0
            replacements += 1

        updated_rows.append(json.dumps(row, ensure_ascii=False))

    if replacements:
        result_file.write_text("\n".join(updated_rows) + "\n", encoding="utf-8")

    return replacements


def export_online_mind2web_artifacts(
    *,
    output_dir: Path,
    task: str,
    task_id: str | None,
    start_url: str | None,
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in trajectory_dir.glob("*_full_screenshot.*"):
        stale_path.unlink()

    copied_screenshots: list[str] = []
    for index, step_stem in enumerate(_ordered_step_stems(output_dir)):
        src = output_dir / "screenshots" / f"{step_stem}.png"
        dst = trajectory_dir / f"{index}_full_screenshot.png"
        if src.exists():
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        else:
            _write_placeholder_screenshot(dst)
        copied_screenshots.append(str(dst))

    action_history: list[str] = []
    thoughts: list[str] = []
    for row in _load_debug_steps(output_dir):
        python_code = str(row.get("python_code", "")).strip()
        if not python_code:
            continue
        action_history.append(python_code)
        thoughts.append(str(row.get("thought", "")).strip())

    if not action_history:
        action_history, thoughts = _fallback_actions_and_thoughts(output_dir / "trajectory.json")

    result_payload = {
        "task_id": task_id or output_dir.name,
        "task": task,
        "start_url": start_url or "",
        "action_history": action_history,
        "thoughts": thoughts,
        "final_result_response": str(agent_result.get("final_response") or agent_result.get("submission") or ""),
        "exit_status": str(agent_result.get("exit_status", "")),
        "submission": str(agent_result.get("submission", "")),
        "trajectory_source": "mini-swe-webagent",
        "trajectory_file": str(output_dir / "trajectory.json"),
        "debug_steps_dir": str(output_dir / "debug" / "steps"),
        "screenshot_paths": copied_screenshots,
    }
    if agent_result.get("run_exception"):
        result_payload["run_exception"] = str(agent_result["run_exception"])

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    return {
        "result_json": str(result_path),
        "trajectory_dir": str(trajectory_dir),
    }


def run_online_mind2web_judge(
    *,
    judge_python: Path,
    judge_script: Path,
    trajectories_dir: Path,
    output_dir: Path,
    judge_model: str,
    num_proc: int,
    api_key: str,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(judge_python),
        str(judge_script),
        "--mode",
        "WebJudge_Online_Mind2Web_eval",
        "--model",
        judge_model,
        "--trajectories_dir",
        str(trajectories_dir),
        "--api_key",
        api_key,
        "--output_path",
        str(output_dir),
        "--num_proc",
        str(num_proc),
        "--score_threshold",
        "3",
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            completed.stdout + ("\n" if completed.stdout and completed.stderr else "") + completed.stderr,
            encoding="utf-8",
        )
    normalize_online_mind2web_judge_results(
        result_file=_judge_result_file_path(output_dir, judge_model),
    )
    return completed

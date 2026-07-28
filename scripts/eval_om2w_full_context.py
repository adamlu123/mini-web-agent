"""Full-context OM2W WebJudge: feed the judge screenshots AND action history
AND thoughts, so base-model trajectories (which never print the SFT-style
"step N action:" lines) still get judged on real behavioral evidence.

Evidence sources per task (mini-web-agent outputs/<task_id>/):
  - actions + thoughts: raw_responses.jsonl (last attempt per turn; <bash> and
    <think> blocks). Fallbacks: command_history.sh (actions only), then
    final_runs/run_<N>/final_script_log.txt in plain-text mode.
  - screenshots: result.json["screenshot_paths"] (session-level, one per agent
    step; missing files remapped into <task>/trajectory/). Fallback:
    final_runs/run_<N>/screenshots/*.png. Evenly subsampled to
    --max_screenshots (always keeping the final frames).

Flow mirrors the vendored WebJudge: identify key points -> score each
screenshot 1-5 (with retry) -> feed selected shots + actions + thoughts to the
final verdict call. Resumable: rerun with the same --output_path to skip
already-judged task_ids.

Usage (repo root):
  python scripts/eval_om2w_full_context.py \
      --trajectories_dir /mnt/pvc/t-yifeili/evals/base4b_all_run1/outputs \
      --output_path /mnt/pvc/t-yifeili/evals/base4b_all_run1/outputs_eval_fullctx \
      --model o4-mini --num_worker 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
for extra in (_REPO, _REPO / "scripts", _REPO / "om2w_judge"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from eval_with_original_om2w import (  # noqa: E402
    FINAL_RESPONSE_RE,
    bound_action_history,
    judge_image_with_retry,
    load_actions,
    load_task_description_map,
    resolve_latest_run_dir,
)
from methods import webjudge_online_mind2web as upstream_webjudge  # noqa: E402
from om2w_judge.utils import OpenaiEngine  # noqa: E402

STATUS_RE = re.compile(r"status\s*:\s*\W*(success|failure)", re.IGNORECASE)


def extract_final_status(response: str) -> int:
    # 判词正文可能引用 "Status: success" 字样,所以取最后一个 Status 声明,
    # 而不是上游 extract_predication 的"第一个 status: 之后含 success 就算过"。
    matches = STATUS_RE.findall(response or "")
    return 1 if matches and matches[-1].lower() == "success" else 0

MODE = "WebJudge_FullContext_eval"
DEFAULT_TASKS_FILE = _REPO / "src/miniswewebagent/run/benchmarks/om2w_260220.json"

TURN_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
TURN_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL | re.IGNORECASE)

THOUGHTS_MAX_ENTRIES = 150
THOUGHTS_MAX_CHARS_EACH = 400
THOUGHTS_MAX_CHARS_TOTAL = 40_000

SYSTEM_MSG = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's step-by-step thoughts, the agent's action history (the shell commands it actually executed to drive the browser), key points for task completion, and some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots, thoughts and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.
8: The agent's thoughts are its own claims about what it did or found; treat them as intent, not as proof. Verify claims against the executed actions and the snapshots. If the thoughts claim success but neither the actions nor the snapshots support it, the task is not successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""

PROMPT_FULL = """User Task: {task}

Key Points: {key_points}

Agent Thoughts (step-by-step, agent's own reasoning; verify against actions and snapshots):
{thoughts}

Action History (shell commands the agent actually executed):
{actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{snapshot_reasons}"""


def load_turns_from_raw_responses(path: Path) -> tuple[list[str], list[str], str]:
    if not path.exists():
        return [], [], ""
    turns: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event") != "raw_text":
            continue
        text = str(row.get("raw_text", ""))
        try:
            attempt = int(row.get("attempt", 1) or 1)
        except Exception:
            attempt = 1
        if attempt <= 1 or not turns:
            turns.append(text)
        else:
            turns[-1] = text  # retry replaces the same turn
    actions: list[str] = []
    thoughts: list[str] = []
    for index, text in enumerate(turns, 1):
        think_match = TURN_THINK_RE.search(text)
        bash_match = TURN_BASH_RE.search(text)
        raw_thought = ""
        if think_match:
            raw_thought = think_match.group(1)
        elif "</think>" in text:
            # vLLM chat templates often inject the opening <think> tag, so the
            # completion contains only the closing tag; everything before it is
            # the thought.
            raw_thought = text.split("</think>", 1)[0].replace("<think>", "")
        thought = re.sub(r"\s+", " ", raw_thought).strip()
        if thought:
            thoughts.append(f"step {index}: {thought}")
        if bash_match:
            command = bash_match.group(1).strip()
            if command:
                actions.append(f"step {index}: {command}")
    return actions, thoughts, "raw_responses"


def load_actions_from_command_history(path: Path) -> list[str]:
    if not path.exists():
        return []
    actions = []
    for index, line in enumerate(
        (l.strip() for l in path.read_text(encoding="utf-8", errors="replace").splitlines()), 1
    ):
        if not line or line.startswith("#") or FINAL_RESPONSE_RE.match(line):
            continue
        actions.append(f"cmd {index}: {line}")
    return actions


def load_evidence(task_dir: Path) -> tuple[list[str], list[str], str]:
    actions, thoughts, source = load_turns_from_raw_responses(task_dir / "raw_responses.jsonl")
    if actions:
        return actions, thoughts, source
    actions = load_actions_from_command_history(task_dir / "command_history.sh")
    if actions:
        return actions, thoughts, "command_history"
    run_dir = resolve_latest_run_dir(task_dir)
    if run_dir is not None:
        actions = load_actions(run_dir / "final_script_log.txt", plain_text=True)
        if actions:
            return actions, thoughts, "final_script_log"
    return [], thoughts, "none"


def bound_thoughts(thoughts: list[str]) -> list[str]:
    clipped = [
        t if len(t) <= THOUGHTS_MAX_CHARS_EACH else t[: THOUGHTS_MAX_CHARS_EACH - 3] + "..."
        for t in thoughts
    ]
    if len(clipped) > THOUGHTS_MAX_ENTRIES:
        omitted = len(clipped) - THOUGHTS_MAX_ENTRIES
        head = THOUGHTS_MAX_ENTRIES // 2
        tail = THOUGHTS_MAX_ENTRIES - head
        clipped = clipped[:head] + [f"[{omitted} thought(s) omitted]"] + clipped[-tail:]
    total = 0
    bounded = []
    for t in clipped:
        if total + len(t) > THOUGHTS_MAX_CHARS_TOTAL:
            bounded.append("[remaining thoughts omitted: char budget reached]")
            break
        bounded.append(t)
        total += len(t)
    return bounded


def load_session_screenshots(task_dir: Path) -> list[str]:
    paths: list[str] = []
    result_path = task_dir / "result.json"
    if result_path.exists():
        try:
            recorded = json.loads(result_path.read_text(encoding="utf-8")).get("screenshot_paths") or []
        except Exception:
            recorded = []
        for raw in recorded:
            p = Path(str(raw))
            if not p.exists():
                p = task_dir / "trajectory" / p.name
            if p.exists() and p.stat().st_size > 0:
                paths.append(str(p))
    if paths:
        return paths
    run_dir = resolve_latest_run_dir(task_dir)
    if run_dir is not None and (run_dir / "screenshots").is_dir():
        shots = sorted((run_dir / "screenshots").glob("*.png"))
        paths = [str(p) for p in shots if p.stat().st_size > 0]
    return paths


def subsample_screenshots(paths: list[str], cap: int) -> list[str]:
    n = len(paths)
    if n <= cap or cap <= 0:
        return paths
    indices = sorted({round(j * (n - 1) / (cap - 1)) for j in range(cap)})
    return [paths[i] for i in indices]


async def full_context_eval(task, actions, thoughts, images_path, model, score_threshold):
    key_points = await upstream_webjudge.identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")
    try:
        key_points = key_points.split("**Key Points**:")[1]
    except Exception:
        key_points = key_points.split("Key Points:")[-1]
    key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    image_records = await asyncio.gather(
        *[judge_image_with_retry(task, p, key_points, model) for p in images_path]
    )

    whole_content_img = []
    snapshot_reasons = []
    record = []
    for image_record, image_path in zip(image_records, images_path):
        record.append({**image_record, "image_path": image_path})
        score = int(image_record.get("Score", 0) or 0)
        reasoning = str(image_record.get("Reasoning", "") or "").strip()
        if score >= score_threshold:
            b64 = upstream_webjudge.encode_image(Image.open(image_path))
            whole_content_img.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
            )
            if reasoning:
                snapshot_reasons.append(reasoning)

    whole_content_img = whole_content_img[: upstream_webjudge.MAX_IMAGE]
    snapshot_reasons = snapshot_reasons[: upstream_webjudge.MAX_IMAGE]

    text = PROMPT_FULL.format(
        task=task,
        key_points=key_points,
        thoughts="\n".join(thoughts) if thoughts else "(none recorded)",
        actions="\n".join(f"{i+1}. {a}" for i, a in enumerate(actions)) if actions else "(none recorded)",
        snapshot_reasons="\n".join(f"{i+1}. {r}" for i, r in enumerate(snapshot_reasons)) or "(no snapshot passed the relevance filter)",
    )
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": [{"type": "text", "text": text}] + whole_content_img},
    ]
    return messages, text, record, key_points


def append_row(output_json_path, row, labels, lock):
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with lock:
        labels.append(row["predicted_label"])
        with open(output_json_path, "a+", encoding="utf-8") as f_out:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def auto_eval(args, task_subset, labels, lock, model, task_map):
    output_json_path = os.path.join(
        args.output_path,
        f"{MODE}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json",
    )
    already: set[str] = set()
    if os.path.exists(output_json_path):
        for line in open(output_json_path, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    already.add(json.loads(line)["task_id"])
                except Exception:
                    pass
    print(f"[pid {os.getpid()}] already done: {len(already)}")

    for task_id in task_subset:
        if task_id in already:
            continue
        task_dir = Path(args.trajectories_dir) / task_id
        task_description = task_map.get(task_id)
        if not task_description:
            print(f"Skip {task_id}: not in tasks file")
            continue

        actions, thoughts, action_source = load_evidence(task_dir)
        actions = bound_action_history(actions)
        thoughts = bound_thoughts(thoughts)
        screenshots = subsample_screenshots(load_session_screenshots(task_dir), args.max_screenshots)

        print(
            f"[pid {os.getpid()}] {task_id}: actions={len(actions)}({action_source}) "
            f"thoughts={len(thoughts)} shots={len(screenshots)}"
        )
        if not actions and not screenshots:
            row = {
                "task_id": task_id, "mode": MODE, "action_history_source": action_source,
                "action_history": [], "thoughts": [], "screenshots_used": [],
                "image_judge_record": [], "key_points": "", "input_text": "",
                "evaluation_details": {"response": "No evidence (no actions, no screenshots).", "predicted_label": 0},
                "predicted_label": 0,
            }
            append_row(output_json_path, row, labels, lock)
            continue

        messages, text, record, key_points = asyncio.run(
            full_context_eval(task_description, actions, thoughts, screenshots, model, args.score_threshold)
        )
        response = model.generate(messages, max_new_tokens=8192)[0]
        predicted_label = extract_final_status(response)

        row = {
            "task_id": task_id,
            "mode": MODE,
            "action_history_source": action_source,
            "action_history": actions,
            "thoughts": thoughts,
            "screenshots_used": screenshots,
            "image_judge_record": record,
            "key_points": key_points,
            "input_text": text,
            "evaluation_details": {"response": response, "predicted_label": predicted_label},
            "predicted_label": predicted_label,
        }
        append_row(output_json_path, row, labels, lock)
        print(f"[pid {os.getpid()}] done {task_id}: predicted_label={predicted_label}")


def process_subset(task_subset, args, labels, lock, task_map):
    model = OpenaiEngine(model=args.model, api_key=args.api_key, endpoint_target_uri=args.endpoint_target_uri)
    auto_eval(args, task_subset, labels, lock, model, task_map)


def parallel_eval(args, num_workers: int) -> None:
    task_entries = json.loads(Path(args.tasks_file).read_text(encoding="utf-8"))
    task_map = load_task_description_map(Path(args.tasks_file))
    selected_entries = [
        e for e in task_entries
        if args.task_level in ("", "all") or str(e.get("level", "")) in args.task_level.split("+")
    ]
    if args.limit > 0:
        selected_entries = selected_entries[: args.limit]
    task_dirs = [
        str(e["task_id"]) for e in selected_entries
        if str(e["task_id"]) in task_map and os.path.isdir(os.path.join(args.trajectories_dir, str(e["task_id"])))
    ]
    print(f"Evaluating {len(task_dirs)} tasks in {args.trajectories_dir}")
    if not task_dirs:
        raise RuntimeError("No matching task directories found")

    num_workers = max(1, min(num_workers, len(task_dirs)))
    subsets = [s for s in (task_dirs[i::num_workers] for i in range(num_workers)) if s]

    if num_workers == 1:
        import threading
        lock = threading.Lock()
        labels: list[int] = []
        model = OpenaiEngine(model=args.model, api_key=args.api_key, endpoint_target_uri=args.endpoint_target_uri)
        auto_eval(args, task_dirs, labels, lock, model, task_map)
    else:
        lock = multiprocessing.Lock()
        with multiprocessing.Manager() as manager:
            labels = manager.list()
            procs = [multiprocessing.Process(target=process_subset, args=(s, args, labels, lock, task_map)) for s in subsets]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
            failed = [p.pid for p in procs if p.exitcode]
            if failed:
                raise RuntimeError(f"{len(failed)} judge worker process(es) failed: {failed}")

    output_json_path = os.path.join(
        args.output_path,
        f"{MODE}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json",
    )
    predictions: dict[str, int] = {}
    for line in open(output_json_path, encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("task_id") in task_dirs and row.get("predicted_label") in (0, 1):
            predictions[row["task_id"]] = row["predicted_label"]
    missing = sorted(set(task_dirs) - predictions.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} task(s) have no judge result")

    level_by_task = {str(e["task_id"]): str(e.get("level") or "unknown") for e in selected_entries}
    breakdown: dict[str, dict[str, int]] = {}
    for tid in task_dirs:
        b = breakdown.setdefault(level_by_task.get(tid, "unknown"), {"success": 0, "judged": 0, "total_tasks": 0})
        b["total_tasks"] += 1
        b["judged"] += 1
        b["success"] += predictions[tid]
    total, success = len(task_dirs), sum(predictions.values())
    summary = {
        "judge_result_file": output_json_path,
        "mode": MODE,
        "model": args.model,
        "score_threshold": args.score_threshold,
        "task_level": args.task_level or "all",
        "overall": {"success": success, "judged": len(predictions), "total_tasks": total,
                    "success_rate": f"{(success / total) * 100:.1f}%"},
        "level_breakdown": breakdown,
    }
    if args.summary_path:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["level_breakdown"], indent=2))
    print(f"Success rate: {success}/{total} = {(success / total) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-context OM2W WebJudge (screenshots + actions + thoughts).")
    parser.add_argument("--trajectories_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="o4-mini")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--endpoint_target_uri", "--endpoint-target-uri", dest="endpoint_target_uri",
                        type=str, default=os.getenv("OPENAI_GATEWAY_ENDPOINT", ""))
    parser.add_argument("--score_threshold", type=int, default=3)
    parser.add_argument("--task_level", "--task-level", dest="task_level", type=str, default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_screenshots", type=int, default=50,
                        help="Evenly subsample session screenshots down to this many per task.")
    parser.add_argument("--summary_path", "--summary-path", dest="summary_path")
    parser.add_argument("--tasks_file", type=str, default=str(DEFAULT_TASKS_FILE))
    parser.add_argument("--num_worker", type=int, default=32)
    args = parser.parse_args()

    if not args.api_key:
        if args.endpoint_target_uri:
            # 走 phyagi gateway 时必须配 gateway key,直连 OpenAI 的 key 会 401
            args.api_key = os.getenv("OPENAI_GATEWAY_API_KEY", "")
        if not args.api_key:
            args.api_key = os.getenv("OPENAI_API_BACKUP_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not args.api_key:
        raise SystemExit("--api_key or OPENAI_API_KEY / OPENAI_GATEWAY_API_KEY must be set")
    parallel_eval(args, args.num_worker)


if __name__ == "__main__":
    main()

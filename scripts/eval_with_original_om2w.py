"""Evaluate a mini-web-agent output directory with the original upstream
`WebJudge_Online_Mind2Web_eval` implementation (unmodified copy at
/home/luyadong/sandbox/Online-Mind2Web/src).

Per-task inputs:
  - last_actions: parsed from
      <task_folder>/final_runs/run_<highest id>/final_script_log.txt
    (every line matching "step <N> action: ...")
  - images_path: every PNG under
      <task_folder>/final_runs/run_<highest id>/screenshots/
    (sorted by the numeric suffix in "final_execution_<N>_...")
  - task: result.json["task"]

Usage:
  python eval_with_original_om2w.py \
      --trajectories_dir /home/luyadong/sandbox/mini-web-agent/outputs/iterative/0421_hard_best_v3_write_judge_once \
      --output_path /tmp/om2w_original_eval \
      --model o4-mini \
      --api_key $OPENAI_API_KEY \
      --num_worker 8
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

from PIL import Image

# Make the freshly-cloned upstream package importable as-is.
UPSTREAM_SRC = Path("/home/luyadong/sandbox/Online-Mind2Web/src")
if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))

from methods import webjudge_online_mind2web as upstream_webjudge  # noqa: E402

# Use the local engine so reasoning models like o4-mini work (it routes
# max_tokens -> max_completion_tokens and drops temperature when required).
# Upstream utils.extract_predication is the same logic but we inline its
# behavior via the local one to keep a single import source.
_REPO = Path("/home/luyadong/sandbox/mini-web-agent")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from om2w_judge.utils import OpenaiEngine, extract_predication  # noqa: E402


FINAL_RUN_DIR_RE = re.compile(r"^run_(\d+)$", re.IGNORECASE)
STEP_ACTION_RE = re.compile(r"^\s*step\s+\d+\s+action\s*:\s*.+\s*$", re.IGNORECASE)
SCREENSHOT_RE = re.compile(r"^final_execution_(\d+).*\.png$", re.IGNORECASE)
MODE = "WebJudge_Online_Mind2Web_eval"
DEFAULT_TASKS_FILE = Path(
    "/home/luyadong/sandbox/mini-web-agent/src/miniswewebagent/run/benchmarks/om2w_260220.json"
)


def load_task_description_map(tasks_file: Path) -> dict[str, str]:
    data = json.loads(Path(tasks_file).read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in data:
        tid = entry.get("task_id")
        task = entry.get("confirmed_task") or entry.get("task")
        if tid and task:
            mapping[tid] = task
    return mapping
IMAGE_PARSE_MAX_RETRIES = 10
RETRYABLE_STATUS_CODES = frozenset({400, 408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_STATUS_CODE_TEXT = frozenset(str(code) for code in RETRYABLE_STATUS_CODES)
RETRYABLE_RESPONSE_TEXT_MARKERS = (
    "rate limit",
    "ratelimit",
    "too many requests",
    "bad gateway",
    "gateway timeout",
    "temporarily unavailable",
    "service unavailable",
    "server disconnected",
    "connection reset",
    "timed out",
)


def resolve_latest_run_dir(task_dir: Path) -> Path | None:
    final_runs = task_dir / "final_runs"
    if not final_runs.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for p in final_runs.iterdir():
        if not p.is_dir():
            continue
        m = FINAL_RUN_DIR_RE.fullmatch(p.name)
        if not m:
            continue
        if not (p / "final_script_log.txt").exists() and not (p / "screenshots").is_dir():
            continue
        candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def load_actions(log_path: Path, plain_text: bool = False) -> list[str]:
    if not log_path.exists():
        return []
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    # Some final_script_log.txt files store the whole trajectory on one line
    # with literal "\n" escape sequences; normalize those to real newlines.
    normalized = raw.replace("\\n", "\n")
    out: list[str] = []
    for line in normalized.splitlines():
        s = line.strip()
        if not s:
            continue
        if plain_text or STEP_ACTION_RE.match(s):
            out.append(s)
    return out


def load_screenshots(shots_dir: Path) -> list[str]:
    if not shots_dir.is_dir():
        return []
    # Include every *.png under the screenshots dir. Sort canonical
    # final_execution_<N>*.png by their numeric suffix, and any other PNGs
    # by their leading integer (handling cp<N>_, critical_point_<N>_, etc.)
    # falling back to lexicographic order. Canonical files are emitted first
    # so the agent's official trajectory frames always lead.
    keyed: list[tuple[int, str]] = []
    extras: list[str] = []
    for name in os.listdir(shots_dir):
        if not name.lower().endswith(".png"):
            continue
        m = SCREENSHOT_RE.match(name)
        if m:
            keyed.append((int(m.group(1)), name))
        else:
            extras.append(name)

    keyed.sort(key=lambda x: (x[0], x[1]))

    def _sortkey(n: str) -> tuple[int, str]:
        m = re.match(r"^(?:cp|critical_point_)?(\d+)", n, re.IGNORECASE)
        return (int(m.group(1)) if m else 10**9, n.lower())
    extras.sort(key=_sortkey)

    ordered = [name for _, name in keyed] + extras
    return [str(shots_dir / name) for name in ordered]


def parse_image_judge_response(response: str) -> tuple[str, int]:
    score_match = re.search(r"(?is)\bscore\b[^1-5]*([1-5])\b", response)
    reasoning_match = re.search(
        r"(?is)(?:\*\*?\s*reasoning\s*\*\*?|reasoning)\s*[:\-]\s*"
        r"(.*?)(?=\n\s*(?:\d+\.\s*)?(?:\*\*?\s*score\s*\*\*?|score)\s*[:\-]|\Z)",
        response,
    )
    if score_match and reasoning_match:
        reasoning = re.sub(r"\s+", " ", reasoning_match.group(1)).strip()
        return reasoning, int(score_match.group(1))

    try:
        payload = json.loads(response)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        score = payload.get("Score", payload.get("score"))
        reasoning = payload.get("Reasoning", payload.get("reasoning"))
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        if (
            isinstance(score, int)
            and 1 <= score <= 5
            and isinstance(reasoning, str)
            and reasoning.strip()
        ):
            return re.sub(r"\s+", " ", reasoning).strip(), score

    raise ValueError("Could not parse image judge response")


def retryable_image_judge_response_error(response: str) -> str | None:
    text = response.strip()
    if not text:
        return "empty image judge response"

    lowered = text.lower()
    if text in RETRYABLE_STATUS_CODE_TEXT:
        return f"retryable gateway status text: {text}"
    for marker in RETRYABLE_RESPONSE_TEXT_MARKERS:
        if marker in lowered:
            snippet = re.sub(r"\s+", " ", text)[:160]
            return f"retryable gateway response text: {snippet}"

    try:
        payload = json.loads(text)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    status_candidates = (
        payload.get("status"),
        payload.get("status_code"),
        payload.get("code"),
    )
    for candidate in status_candidates:
        if isinstance(candidate, str) and candidate.strip().isdigit():
            candidate = int(candidate.strip())
        if isinstance(candidate, int) and candidate in RETRYABLE_STATUS_CODES:
            return f"retryable gateway response payload status: {candidate}"

    error_payload = payload.get("error")
    if isinstance(error_payload, dict):
        code = error_payload.get("code")
        if isinstance(code, str) and code.strip().isdigit():
            code = int(code.strip())
        if isinstance(code, int) and code in RETRYABLE_STATUS_CODES:
            return f"retryable gateway response error code: {code}"
        message = error_payload.get("message")
        if isinstance(message, str):
            lowered_message = message.lower()
            if any(marker in lowered_message for marker in RETRYABLE_RESPONSE_TEXT_MARKERS):
                snippet = re.sub(r"\s+", " ", message)[:160]
                return f"retryable gateway response error: {snippet}"

    message = payload.get("message")
    if isinstance(message, str):
        lowered_message = message.lower()
        if any(marker in lowered_message for marker in RETRYABLE_RESPONSE_TEXT_MARKERS):
            snippet = re.sub(r"\s+", " ", message)[:160]
            return f"retryable gateway response message: {snippet}"

    return None


async def judge_image_with_retry(task: str, image_path: str, key_points: str, model) -> dict[str, object]:
    last_response = ""
    last_error: BaseException | None = None
    for attempt in range(1, IMAGE_PARSE_MAX_RETRIES + 1):
        try:
            last_response = await upstream_webjudge.judge_image(task, image_path, key_points, model)
        except Exception as exc:
            last_error = exc
            print(
                f"Error loading/judging image on attempt {attempt}/{IMAGE_PARSE_MAX_RETRIES} for "
                f"{image_path}: {exc}"
            )
            continue

        # Try to parse first. A successful parse means the response is a real
        # judge verdict, even if the reasoning text happens to contain phrases
        # like "timed out" that would otherwise look like a transient gateway
        # error. Only fall back to the retryable detector when parsing fails.
        try:
            reasoning, score = parse_image_judge_response(last_response)
            return {
                "Response": last_response,
                "Score": score,
                "Reasoning": reasoning,
                "Attempts": attempt,
                "ParseFailed": False,
            }
        except Exception as parse_exc:
            retryable_error = retryable_image_judge_response_error(last_response)
            if retryable_error is not None:
                last_error = RuntimeError(retryable_error)
                print(
                    f"Error processing response on attempt {attempt}/{IMAGE_PARSE_MAX_RETRIES}: "
                    f"{retryable_error}"
                )
                continue
            last_error = parse_exc
            print(f"Error processing response on attempt {attempt}/{IMAGE_PARSE_MAX_RETRIES}: {parse_exc}")

    return {
        "Response": last_response,
        "Score": 0,
        "Reasoning": "",
        "Attempts": IMAGE_PARSE_MAX_RETRIES,
        "ParseFailed": True,
        "ParseError": str(last_error) if last_error is not None else "unknown",
    }


async def robust_webjudge_online_mind2web_eval(task, last_actions, images_path, model, score_threshold):
    system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the filter applied is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: \"success\" or \"failure\"
"""
    prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}

The potentially important snapshots of the webpage in the agent's trajectory and their reasons:
{thoughts}"""

    key_points = await upstream_webjudge.identify_key_points(task, model)
    key_points = key_points.replace("\n\n", "\n")
    try:
        key_points = key_points.split("**Key Points**:")[1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())
    except Exception:
        key_points = key_points.split("Key Points:")[-1]
        key_points = "\n".join(line.lstrip() for line in key_points.splitlines())

    image_records = await asyncio.gather(
        *[
            judge_image_with_retry(task, image_path, key_points, model)
            for image_path in images_path
        ]
    )

    whole_content_img = []
    whole_thoughts = []
    record = []
    for image_record, image_path in zip(image_records, images_path):
        record.append(image_record)
        score = int(image_record.get("Score", 0) or 0)
        thought = str(image_record.get("Reasoning", "") or "").strip()
        if score >= score_threshold:
            jpg_base64_str = upstream_webjudge.encode_image(Image.open(image_path))
            whole_content_img.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{jpg_base64_str}", "detail": "high"},
                }
            )
            if thought:
                whole_thoughts.append(thought)

    whole_content_img = whole_content_img[: upstream_webjudge.MAX_IMAGE]
    whole_thoughts = whole_thoughts[: upstream_webjudge.MAX_IMAGE]
    if not whole_content_img:
        prompt = """User Task: {task}

Key Points: {key_points}

Action History:
{last_actions}"""

    text = prompt.format(
        task=task,
        last_actions="\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions)),
        key_points=key_points,
        thoughts="\n".join(f"{i+1}. {thought}" for i, thought in enumerate(whole_thoughts)),
    )
    messages = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [{"type": "text", "text": text}] + whole_content_img,
        },
    ]
    return messages, text, system_msg, record, key_points


def auto_eval(args, task_subset, final_predicted_labels, lock, model, task_map):
    output_json_path = os.path.join(
        args.output_path,
        f"{MODE}_{args.model}_score_threshold_{args.score_threshold}_auto_eval_results.json",
    )
    already_ids: set[str] = set()
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_ids.add(json.loads(line)["task_id"])
                except Exception:
                    pass

    print(f"[pid {os.getpid()}] already done: {len(already_ids)}")

    for task_id in task_subset:
        if task_id in already_ids:
            continue

        task_dir = Path(args.trajectories_dir) / task_id
        output_results: dict = {}
        task_description = task_map.get(task_id)
        if not task_description:
            print(f"Skip {task_id}: task_id not found in tasks file")
            continue

        run_dir = resolve_latest_run_dir(task_dir)
        if run_dir is None:
            print(f"Skip {task_id}: no final_runs/run_* directory found")
            continue

        action_history = load_actions(
            run_dir / "final_script_log.txt", plain_text=True
        )
        screenshot_paths = load_screenshots(run_dir / "screenshots")

        if not screenshot_paths:
            print(f"Skip {task_id}: no screenshots under {run_dir}/screenshots")
            continue

        print(
            f"[pid {os.getpid()}] {task_id}: run={run_dir.name} "
            f"actions={len(action_history)} shots={len(screenshot_paths)}"
        )

        messages, text, system_msg, record, key_points = asyncio.run(
            robust_webjudge_online_mind2web_eval(
                task_description,
                action_history,
                screenshot_paths,
                model,
                args.score_threshold,
            )
        )

        response = model.generate(messages, max_new_tokens=8192)[0]
        predicted_label = extract_predication(response, MODE)

        output_results["task_id"] = task_id
        output_results["mode"] = MODE
        output_results["final_run_dir"] = str(run_dir)
        output_results["action_history"] = action_history
        output_results["action_history_source"] = "final_script_log"
        output_results["sandbox_screenshot_paths"] = screenshot_paths
        output_results["image_judge_record"] = record
        output_results["key_points"] = key_points
        output_results["input_text"] = text
        output_results["system_msg"] = system_msg
        output_results["evaluation_details"] = {
            "response": response,
            "predicted_label": predicted_label,
        }
        output_results["predicted_label"] = predicted_label

        with lock:
            final_predicted_labels.append(predicted_label)

        os.makedirs(args.output_path, exist_ok=True)
        with lock:
            with open(output_json_path, "a+", encoding="utf-8") as f_out:
                f_out.write(json.dumps(output_results) + "\n")

        print(f"[pid {os.getpid()}] done {task_id}: predicted_label={predicted_label}")


def process_subset(task_subset, args, final_predicted_labels, lock, task_map):
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key,
        endpoint_target_uri=args.endpoint_target_uri,
    )
    auto_eval(args, task_subset, final_predicted_labels, lock, model, task_map)


def parallel_eval(args, num_workers: int) -> None:
    task_map = load_task_description_map(Path(args.tasks_file))
    print(f"Loaded {len(task_map)} task descriptions from {args.tasks_file}")
    task_dirs = [
        d
        for d in sorted(os.listdir(args.trajectories_dir))
        if os.path.isdir(os.path.join(args.trajectories_dir, d))
    ]
    print(f"Evaluating {len(task_dirs)} tasks in {args.trajectories_dir}")
    if not task_dirs:
        return

    num_workers = max(1, min(num_workers, len(task_dirs)))
    task_subsets = [task_dirs[i::num_workers] for i in range(num_workers)]
    task_subsets = [s for s in task_subsets if s]

    if num_workers == 1:
        import threading
        lock = threading.Lock()
        labels: list[int] = []
        model = OpenaiEngine(
            model=args.model,
            api_key=args.api_key,
            endpoint_target_uri=args.endpoint_target_uri,
        )
        auto_eval(args, task_dirs, labels, lock, model, task_map)
        total = len(task_dirs)
        success = sum(labels)
    else:
        lock = multiprocessing.Lock()
        with multiprocessing.Manager() as manager:
            labels = manager.list()
            procs = []
            for subset in task_subsets:
                p = multiprocessing.Process(
                    target=process_subset, args=(subset, args, labels, lock, task_map)
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            total = len(task_dirs)
            success = sum(labels)

    print("Evaluation complete.")
    if total:
        print(f"Success rate: {success}/{total} = {(success / total) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run upstream WebJudge_Online_Mind2Web_eval on a mini-web-agent "
            "iterative output directory, using the latest final_runs/run_* per task."
        )
    )
    parser.add_argument("--trajectories_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="o4-mini", help="Model name for evaluation (e.g., gpt-4o, o4-mini, etc.)")
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="Defaults to $OPENAI_GATEWAY_API_KEY (if endpoint set) or $OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--endpoint_target_uri",
        "--endpoint-target-uri",
        dest="endpoint_target_uri",
        type=str,
        default=os.getenv("OPENAI_GATEWAY_ENDPOINT", ""),
        help="Optional gateway responses API endpoint. Defaults to $OPENAI_GATEWAY_ENDPOINT.",
    )
    parser.add_argument("--score_threshold", type=int, default=3)
    parser.add_argument(
        "--tasks_file",
        type=str,
        default=str(DEFAULT_TASKS_FILE),
        help="Path to the Online-Mind2Web tasks JSON with task_id and confirmed_task.",
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        default=200,
        help="Number of judge worker processes. Default 0 = one worker per task.",
    )
    args = parser.parse_args()

    if not args.api_key:
        if args.endpoint_target_uri:
            args.api_key = (
                os.getenv("OPENAI_GATEWAY_API_KEY", "")
                or os.getenv("PHYAGI_API_KEY", "")
                or os.getenv("OPENAI_API_KEY", "")
            )
        else:
            args.api_key = os.getenv("OPENAI_API_KEY", "")
    if not args.api_key:
        raise SystemExit("--api_key, OPENAI_GATEWAY_API_KEY, or OPENAI_API_KEY must be set")

    num_workers = args.num_worker
    if num_workers <= 0:
        task_count = sum(
            1
            for d in os.listdir(args.trajectories_dir)
            if os.path.isdir(os.path.join(args.trajectories_dir, d))
        )
        num_workers = max(1, task_count)
        print(f"--num_worker not set; defaulting to task count = {num_workers}")

    parallel_eval(args, num_workers)


if __name__ == "__main__":
    main()

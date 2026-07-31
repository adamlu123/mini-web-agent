"""Evaluate a mini-web-agent output directory with the original upstream
`WebJudge_Online_Mind2Web_eval` implementation, but pull `last_actions` and
`images_path` from the existing per-task `judge_result.json` (latest
`final_runs/run_<N>/judge_result.json`) instead of reading
`final_script_log.txt` and the `screenshots/` directory.

Per-task inputs (resolved per task folder):
  - latest_run = highest-numbered run_<N> under <task>/final_runs/ that
    contains a judge_result.json file
  - last_actions: judge_result.json["action_history"]
  - images_path:  judge_result.json["image_paths"]
  - task:         judge_result.json["task"] (falls back to tasks_file)

Usage mirrors eval_with_original_om2w.py:
  python eval_with_original_om2w_v2.py \
      --trajectories_dir /home/luyadong/sandbox/mini-web-agent/outputs/default/0425/0425_real_easymed_oraclejudge_combined \
      --output_path /tmp/om2w_original_eval_v2 \
      --model o4-mini \
      --api_key $OPENAI_GATEWAY_API_KEY \
      --endpoint_target_uri $OPENAI_GATEWAY_ENDPOINT \
      --num_worker 8
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

# Reuse the heavy lifting from v1 to avoid duplication.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_with_original_om2w import (  # noqa: E402
    DEFAULT_TASKS_FILE,
    FINAL_RUN_DIR_RE,
    MODE,
    OpenaiEngine,
    extract_predication,
    load_task_description_map,
    robust_webjudge_online_mind2web_eval,
)


def resolve_latest_judge_result(task_dir: Path) -> tuple[Path, Path] | None:
    """Return (run_dir, judge_result.json path) for the highest run_<N> that
    has a judge_result.json. Returns None if none found."""
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
        if not (p / "judge_result.json").is_file():
            continue
        candidates.append((int(m.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    run_dir = candidates[-1][1]
    return run_dir, run_dir / "judge_result.json"


def load_inputs_from_judge_result(jr_path: Path) -> tuple[str | None, list[str], list[str]]:
    data = json.loads(jr_path.read_text(encoding="utf-8"))
    task = data.get("task") if isinstance(data.get("task"), str) else None
    actions = data.get("action_history") or []
    if not isinstance(actions, list):
        actions = []
    actions = [str(a).strip() for a in actions if str(a).strip()]
    images = data.get("image_paths") or []
    if not isinstance(images, list):
        images = []
    images = [str(p) for p in images if isinstance(p, str) and p]
    return task, actions, images


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

        resolved = resolve_latest_judge_result(task_dir)
        if resolved is None:
            print(f"Skip {task_id}: no final_runs/run_*/judge_result.json found")
            continue
        run_dir, jr_path = resolved

        try:
            task_from_jr, action_history, screenshot_paths = load_inputs_from_judge_result(
                jr_path
            )
        except Exception as exc:
            print(f"Skip {task_id}: failed to parse {jr_path}: {exc}")
            continue

        task_description = task_from_jr or task_map.get(task_id)
        if not task_description:
            print(f"Skip {task_id}: task description not found in judge_result or tasks file")
            continue

        # Use image paths directly from judge_result.json. Error out if any
        # referenced path does not exist on disk.
        missing_images = [p for p in screenshot_paths if not os.path.isfile(p)]
        if missing_images:
            print(
                f"ERROR {task_id}: {len(missing_images)} image_paths from {jr_path} "
                f"do not exist on disk; first missing: {missing_images[0]}"
            )
            continue
        existing_images = list(screenshot_paths)

        if not existing_images:
            print(f"ERROR {task_id}: judge_result.json has no image_paths ({jr_path})")
            continue

        print(
            f"[pid {os.getpid()}] {task_id}: run={run_dir.name} jr={jr_path.name} "
            f"actions={len(action_history)} shots={len(existing_images)}"
        )

        messages, text, system_msg, record, key_points = asyncio.run(
            robust_webjudge_online_mind2web_eval(
                task_description,
                action_history,
                existing_images,
                model,
                args.score_threshold,
            )
        )

        response = model.generate(messages, max_new_tokens=8192)[0]
        predicted_label = extract_predication(response, MODE)

        output_results: dict = {
            "task_id": task_id,
            "mode": MODE,
            "final_run_dir": str(run_dir),
            "judge_result_source": str(jr_path),
            "task": task_description,
            "action_history": action_history,
            "action_history_source": "judge_result.json",
            "sandbox_screenshot_paths": existing_images,
            "image_judge_record": record,
            "key_points": key_points,
            "input_text": text,
            "system_msg": system_msg,
            "evaluation_details": {
                "response": response,
                "predicted_label": predicted_label,
            },
            "predicted_label": predicted_label,
        }

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
            "Run upstream WebJudge_Online_Mind2Web_eval using action_history and "
            "image_paths sourced from the latest final_runs/run_*/judge_result.json."
        )
    )
    parser.add_argument("--trajectories_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
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
        help="Fallback path for task descriptions when judge_result.json lacks 'task'.",
    )
    parser.add_argument(
        "--num_worker",
        type=int,
        default=0,
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

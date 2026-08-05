"""Re-run the official WebJudge_Online_Mind2Web final classification using
cached intermediate artifacts (final_user_text, final_system_msg,
image_records) saved in each task's latest final_runs/run_<N>/judge_result.json.

For each task:
  - load latest run_<N>/judge_result.json
  - build messages = [system(final_system_msg), user([text=final_user_text]
    + [image_url for image_paths[i] where image_records[i].Score >= threshold,
    capped at upstream MAX_IMAGE])]
  - call model.generate(messages); extract_predication
  - compare to the stored predicted_label

Usage:
  python verify_judge_from_cached.py \
      --trajectories_dirs \
        /home/luyadong/sandbox/mini-web-agent/outputs/default/0425_easymed_oraclejudge \
        /home/luyadong/sandbox/mini-web-agent/outputs/default/0425_real_oraclejudge \
      --output_path /tmp/verify_cached_judge \
      --model o4-mini \
      --score_threshold 3 \
      --num_worker 64
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
from pathlib import Path

from PIL import Image

UPSTREAM_SRC = Path("/home/luyadong/sandbox/Online-Mind2Web/src")
if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))
from methods import webjudge_online_mind2web as upstream_webjudge  # noqa: E402

_REPO = Path("/home/luyadong/sandbox/mini-web-agent")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from om2w_judge.utils import OpenaiEngine, extract_predication  # noqa: E402

FINAL_RUN_DIR_RE = re.compile(r"^run_(\d+)$", re.IGNORECASE)
MODE = "WebJudge_Online_Mind2Web_eval"


def resolve_latest_judge_result(task_dir: Path):
    final_runs = task_dir / "final_runs"
    if not final_runs.is_dir():
        return None
    cands: list[tuple[int, Path]] = []
    for p in final_runs.iterdir():
        if not p.is_dir():
            continue
        m = FINAL_RUN_DIR_RE.fullmatch(p.name)
        if not m:
            continue
        if not (p / "judge_result.json").is_file():
            continue
        cands.append((int(m.group(1)), p))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0])
    run_dir = cands[-1][1]
    return run_dir, run_dir / "judge_result.json"


def build_messages(jr: dict, score_threshold: int):
    final_user_text = jr.get("final_user_text") or ""
    final_system_msg = jr.get("final_system_msg") or ""
    image_records = jr.get("image_records") or []
    image_paths = jr.get("image_paths") or []

    img_blocks = []
    for rec, img_path in zip(image_records, image_paths):
        try:
            score = int(rec.get("Score", 0) or 0)
        except Exception:
            score = 0
        if score < score_threshold:
            continue
        if not img_path or not os.path.isfile(img_path):
            continue
        b64 = upstream_webjudge.encode_image(Image.open(img_path))
        img_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            }
        )
    img_blocks = img_blocks[: upstream_webjudge.MAX_IMAGE]
    messages = [
        {"role": "system", "content": final_system_msg},
        {
            "role": "user",
            "content": [{"type": "text", "text": final_user_text}] + img_blocks,
        },
    ]
    return messages, len(img_blocks)


def iter_tasks(trajectories_dirs: list[str]):
    for tdir in trajectories_dirs:
        root = Path(tdir)
        if not root.is_dir():
            continue
        for d in sorted(os.listdir(root)):
            full = root / d
            if full.is_dir():
                yield str(root), d


def process_subset(task_subset, args, results_list, lock):
    model = OpenaiEngine(
        model=args.model,
        api_key=args.api_key,
        endpoint_target_uri=args.endpoint_target_uri,
    )
    out_path = os.path.join(
        args.output_path,
        f"verify_cached_{args.model}_threshold_{args.score_threshold}.jsonl",
    )
    os.makedirs(args.output_path, exist_ok=True)
    for traj_root, task_id in task_subset:
        task_dir = Path(traj_root) / task_id
        resolved = resolve_latest_judge_result(task_dir)
        if resolved is None:
            print(f"Skip {task_id}: no judge_result.json")
            continue
        run_dir, jr_path = resolved
        try:
            jr = json.loads(jr_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skip {task_id}: read failed: {exc}")
            continue
        stored_label = jr.get("predicted_label")
        try:
            messages, n_imgs = build_messages(jr, args.score_threshold)
        except Exception as exc:
            print(f"Skip {task_id}: build_messages failed: {exc}")
            continue
        try:
            response = model.generate(messages, max_new_tokens=8192)[0]
        except Exception as exc:
            print(f"ERROR {task_id}: model.generate failed: {exc}")
            continue
        new_label = extract_predication(response, MODE)
        match = int(new_label == stored_label)
        rec = {
            "trajectories_dir": traj_root,
            "task_id": task_id,
            "final_run_dir": str(run_dir),
            "stored_predicted_label": stored_label,
            "new_predicted_label": new_label,
            "match": match,
            "n_images_attached": n_imgs,
            "response": response,
        }
        with lock:
            results_list.append((traj_root, stored_label, new_label))
            with open(out_path, "a+", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        print(
            f"[pid {os.getpid()}] {task_id}: stored={stored_label} new={new_label} "
            f"match={match} imgs={n_imgs}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories_dirs", nargs="+", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--model", default="o4-mini")
    ap.add_argument("--api_key", default="")
    ap.add_argument(
        "--endpoint_target_uri",
        default=os.getenv("OPENAI_GATEWAY_ENDPOINT", ""),
    )
    ap.add_argument("--score_threshold", type=int, default=3)
    ap.add_argument("--num_worker", type=int, default=0)
    args = ap.parse_args()

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

    tasks = list(iter_tasks(args.trajectories_dirs))
    print(f"Total tasks across {len(args.trajectories_dirs)} dirs: {len(tasks)}")
    if not tasks:
        return

    # Skip tasks already in output file.
    out_path = os.path.join(
        args.output_path,
        f"verify_cached_{args.model}_threshold_{args.score_threshold}.jsonl",
    )
    done: set[tuple[str, str]] = set()
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add((obj["trajectories_dir"], obj["task_id"]))
                except Exception:
                    pass
    if done:
        before = len(tasks)
        tasks = [(r, t) for (r, t) in tasks if (r, t) not in done]
        print(f"Already done: {len(done)}; remaining: {len(tasks)} (was {before})")

    nw = args.num_worker
    if nw <= 0:
        nw = max(1, len(tasks))
    nw = min(nw, max(1, len(tasks)))
    subsets = [tasks[i::nw] for i in range(nw)]
    subsets = [s for s in subsets if s]

    if len(subsets) == 1:
        import threading
        lock = threading.Lock()
        results: list = []
        process_subset(subsets[0], args, results, lock)
    else:
        lock = multiprocessing.Lock()
        with multiprocessing.Manager() as mgr:
            results = mgr.list()
            procs = []
            for s in subsets:
                p = multiprocessing.Process(
                    target=process_subset, args=(s, args, results, lock)
                )
                p.start()
                procs.append(p)
            for p in procs:
                p.join()
            results = list(results)

    # Aggregate from output file (covers prior runs too).
    by_dir: dict[str, dict] = {}
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                d = obj["trajectories_dir"]
                bucket = by_dir.setdefault(
                    d, {"total": 0, "match": 0, "stored_pos": 0, "new_pos": 0}
                )
                bucket["total"] += 1
                bucket["match"] += int(obj.get("match", 0))
                bucket["stored_pos"] += int(obj.get("stored_predicted_label") == 1)
                bucket["new_pos"] += int(obj.get("new_predicted_label") == 1)

    print("\n==== Summary ====")
    g_total = g_match = 0
    for d, b in by_dir.items():
        pct = (b["match"] / b["total"] * 100) if b["total"] else 0.0
        print(
            f"{d}: match={b['match']}/{b['total']} ({pct:.2f}%); "
            f"stored success={b['stored_pos']}, new success={b['new_pos']}"
        )
        g_total += b["total"]
        g_match += b["match"]
    if g_total:
        print(f"TOTAL: match={g_match}/{g_total} = {g_match/g_total*100:.2f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Diagnose mt-3ep self-reflection on externally successful OM2W tasks.

This script takes the WebJudge success set, reruns the current self_reflection
CLI against each task's latest final run, and writes a JSONL/summary comparing
external WebJudge success with mt-3ep self_reflection labels.

Typical usage after starting a local vLLM endpoint for /data/t-yifeili/ckpts/eval_mt_3ep:

  python scripts/diagnose_mt3ep_self_reflection_gap.py \
    --trajectories-dir outputs/sft_ckpt_vllm/eval_mt_3ep_om2w_all_step100_o4096_len65536 \
    --webjudge-results outputs/sft_ckpt_vllm/eval_mt_3ep_om2w_all_step100_o4096_len65536_eval_1/WebJudge_Online_Mind2Web_Sandbox_eval_o4-mini_score_threshold_3_auto_eval_results.json

Or let the script launch vLLM and clean it up afterwards:

  python scripts/diagnose_mt3ep_self_reflection_gap.py --start-vllm
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

_FINAL_RUN_DIR_RE = re.compile(r"^run_(\d+)$", re.IGNORECASE)


def _default_python() -> str:
    return os.environ.get("PY", sys.executable)


def _default_vllm_bin(python_exe: str) -> str:
    candidate = Path(python_exe).resolve().parent / "vllm"
    return str(candidate) if candidate.exists() else os.environ.get("VLLM_BIN", "vllm")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _resolve_latest_final_run_dir(task_dir: Path) -> Path | None:
    final_runs_dir = task_dir / "final_runs"
    if not final_runs_dir.is_dir():
        return None

    candidates: list[tuple[int, str, Path]] = []
    for path in final_runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = _FINAL_RUN_DIR_RE.fullmatch(path.name)
        if not match:
            continue
        log_path = path / "final_script_log.txt"
        screenshots_dir = path / "screenshots"
        if not log_path.exists() and not screenshots_dir.is_dir():
            continue
        candidates.append((int(match.group(1)), path.name, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))

    def has_real_artifacts(run_path: Path) -> bool:
        log_path = run_path / "final_script_log.txt"
        screenshots_dir = run_path / "screenshots"
        return (
            log_path.is_file()
            and log_path.stat().st_size > 0
            and screenshots_dir.is_dir()
            and any(screenshots_dir.glob("final_execution_*.png"))
        )

    for _, _, candidate in reversed(candidates):
        if has_real_artifacts(candidate):
            return candidate
    return candidates[-1][2]


def _read_latest_existing_self_label(run_dir: Path | None) -> Any:
    if run_dir is None:
        return "missing_run"
    path = run_dir / "judge_result.json"
    if not path.is_file():
        return "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace")).get("predicted_label")
    except Exception as exc:  # noqa: BLE001
        return f"unreadable:{type(exc).__name__}"


def _start_vllm(args: argparse.Namespace, python_exe: str) -> subprocess.Popen[str]:
    vllm_bin = args.vllm_bin or _default_vllm_bin(python_exe)
    cmd = [
        vllm_bin,
        "serve",
        str(args.ckpt),
        "--served-model-name",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if args.vllm_args:
        cmd.extend(args.vllm_args.split())
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "vllm.log"
    handle = log_path.open("w", encoding="utf-8")
    print("[vllm] starting", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, text=True)
    setattr(proc, "_diagnostic_log_handle", handle)
    return proc


def _wait_for_endpoint(url: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    model_url = url.rsplit("/v1/", 1)[0].rstrip("/") + "/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(model_url, timeout=5) as response:
                if response.status < 500:
                    print(f"[vllm] ready: {model_url}", flush=True)
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(5)
    raise RuntimeError(f"endpoint did not become ready: {model_url}; last_error={last_error}")


def _run_one(task_id: str, args: argparse.Namespace, python_exe: str) -> dict[str, Any]:
    task_dir = args.trajectories_dir / task_id
    latest_run = _resolve_latest_final_run_dir(task_dir)
    existing_label = _read_latest_existing_self_label(latest_run)
    task_out_dir = args.output_root / task_id
    task_out_dir.mkdir(parents=True, exist_ok=True)
    out_json = task_out_dir / "judge_result.json"
    log_path = task_out_dir / "self_reflection.log"

    row: dict[str, Any] = {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "latest_run": str(latest_run) if latest_run else "",
        "existing_self_reflection_label": existing_label,
        "output_json": str(out_json),
        "log_path": str(log_path),
    }

    config_path = task_dir / "judge_config.json"
    if not config_path.is_file():
        row.update({"status": "skipped", "error": "missing judge_config.json"})
        return row
    if latest_run is None:
        row.update({"status": "skipped", "error": "missing final run artifacts"})
        return row

    cmd = [
        python_exe,
        "-m",
        "miniswewebagent.tools.self_reflection",
        "--config",
        str(config_path),
        "--workspace-dir",
        str(task_dir),
        "--output",
        str(out_json),
        "--model",
        args.model,
        "--endpoint",
        args.endpoint,
        "--api-key",
        args.api_key,
        "--max-image-parse-retries",
        str(args.max_image_parse_retries),
        "--image-max-new-tokens",
        str(args.image_max_new_tokens),
        "--final-max-new-tokens",
        str(args.final_max_new_tokens),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    env = os.environ.copy()
    env.update(
        {
            "WEB_AGENT_POLICY_URL": args.endpoint,
            "WEB_AGENT_POLICY_MODEL": args.model,
            "OPENAI_COMPATIBLE_ENDPOINT": args.endpoint,
            "OPENAI_COMPATIBLE_MODEL": args.model,
            "OPENAI_COMPATIBLE_API_KEY": args.api_key,
            "OPENAI_GATEWAY_MODEL": args.model,
        }
    )
    completed = subprocess.run(
        cmd,
        cwd=args.repo,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.per_task_timeout_seconds,
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    row["returncode"] = completed.returncode
    row["status"] = "ok" if completed.returncode in {0, 1} else "error"
    if out_json.is_file():
        try:
            result = json.loads(out_json.read_text(encoding="utf-8", errors="replace"))
            row["new_self_reflection_label"] = result.get("predicted_label")
            row["new_self_reflection_model"] = result.get("model")
            row["new_self_reflection_endpoint"] = result.get("endpoint")
            row["final_response_head"] = " ".join(str(result.get("final_response") or "").split())[:800]
            records = result.get("image_records") if isinstance(result.get("image_records"), list) else []
            row["n_image_records"] = len(records)
            row["n_image_parse_failed"] = sum(1 for record in records if isinstance(record, dict) and record.get("ParseFailed"))
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"judge_result parse failed: {type(exc).__name__}: {exc}"
    else:
        row["new_self_reflection_label"] = "missing_output"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/data/t-yifeili/mini-web-agent"))
    parser.add_argument("--trajectories-dir", type=Path, default=Path("outputs/sft_ckpt_vllm/eval_mt_3ep_om2w_all_step100_o4096_len65536"))
    parser.add_argument("--webjudge-results", type=Path, default=Path("outputs/sft_ckpt_vllm/eval_mt_3ep_om2w_all_step100_o4096_len65536_eval_1/WebJudge_Online_Mind2Web_Sandbox_eval_o4-mini_score_threshold_3_auto_eval_results.json"))
    parser.add_argument("--output-root", type=Path, default=Path(""))
    parser.add_argument("--model", default="eval_mt_3ep")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--python", default="")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--per-task-timeout-seconds", type=int, default=900)
    parser.add_argument("--max-image-parse-retries", type=int, default=3)
    parser.add_argument("--image-max-new-tokens", type=int, default=1024)
    parser.add_argument("--final-max-new-tokens", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--start-vllm", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=Path("/data/t-yifeili/ckpts/eval_mt_3ep"))
    parser.add_argument("--vllm-bin", default="")
    parser.add_argument("--vllm-args", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-wait-seconds", type=int, default=900)
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    if not args.trajectories_dir.is_absolute():
        args.trajectories_dir = args.repo / args.trajectories_dir
    if not args.webjudge_results.is_absolute():
        args.webjudge_results = args.repo / args.webjudge_results
    if not args.output_root:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = args.repo / "diagnostics" / f"mt3ep_self_reflection_gap_{stamp}"
    elif not args.output_root.is_absolute():
        args.output_root = args.repo / args.output_root
    args.output_root.mkdir(parents=True, exist_ok=True)

    python_exe = args.python or _default_python()
    rows = _load_jsonl(args.webjudge_results)
    task_ids = [str(row.get("task_id")) for row in rows if row.get("predicted_label") == 1 and row.get("task_id")]
    if args.task_id:
        wanted = set(args.task_id)
        task_ids = [task_id for task_id in task_ids if task_id in wanted]
    if args.limit > 0:
        task_ids = task_ids[: args.limit]

    print(f"[diagnose] external WebJudge success tasks: {len(task_ids)}", flush=True)
    vllm_proc: subprocess.Popen[str] | None = None
    try:
        if args.start_vllm:
            vllm_proc = _start_vllm(args, python_exe)
            _wait_for_endpoint(args.endpoint, args.vllm_wait_seconds)

        summary_path = args.output_root / "summary.jsonl"
        all_rows: list[dict[str, Any]] = []
        with summary_path.open("w", encoding="utf-8") as handle:
            if args.workers <= 1:
                iterator = ((_run_one(task_id, args, python_exe)) for task_id in task_ids)
                for index, row in enumerate(iterator, start=1):
                    row["index"] = index
                    all_rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(index, row["task_id"], row.get("new_self_reflection_label"), row.get("status"), flush=True)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {executor.submit(_run_one, task_id, args, python_exe): task_id for task_id in task_ids}
                    for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                        try:
                            row = future.result()
                        except Exception as exc:  # noqa: BLE001
                            row = {"task_id": futures[future], "status": "error", "error": repr(exc)}
                        row["index"] = index
                        all_rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        print(index, row["task_id"], row.get("new_self_reflection_label"), row.get("status"), flush=True)

        counts = Counter(str(row.get("new_self_reflection_label")) for row in all_rows)
        statuses = Counter(str(row.get("status")) for row in all_rows)
        parse_failed_counts = sum(int(row.get("n_image_parse_failed") or 0) for row in all_rows)
        summary = {
            "n_tasks": len(task_ids),
            "labels": dict(counts),
            "statuses": dict(statuses),
            "total_image_parse_failed": parse_failed_counts,
            "summary_jsonl": str(summary_path),
            "output_root": str(args.output_root),
        }
        (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        if vllm_proc is not None:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()
            handle = getattr(vllm_proc, "_diagnostic_log_handle", None)
            if handle is not None:
                handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

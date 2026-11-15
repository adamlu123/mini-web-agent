#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cuawright.environments.benchcad.scoring import score_prediction_with_retry
from cuawright.environments.benchcad.task import load_task, task_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score an immutable BenchCAD prediction offline"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--image",
        default="cuawright-benchcad:52087ef",
    )
    parser.add_argument(
        "--benchcad-repo",
        default=str(REPO_ROOT / "cache" / "BenchCAD-main"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, task = load_task(
        args.manifest,
        args.record_id,
        validate_assets=False,
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    deferred_result_path = run_dir / "evaluation" / "result.json"
    deferred_result = json.loads(deferred_result_path.read_text(encoding="utf-8"))
    if deferred_result.get("task_id") != task.record_id:
        raise ValueError(
            "Run task ID does not match requested manifest record: "
            f"{deferred_result.get('task_id')!r} != {task.record_id!r}"
        )
    deferred_evaluation = deferred_result.get("evaluation") or {}
    if deferred_evaluation.get("status") != "pending_offline_score":
        raise ValueError("Run is not pending offline scoring")
    prediction = run_dir / "workspace" / "prediction.py"
    actual_hash = (
        hashlib.sha256(prediction.read_bytes()).hexdigest()
        if prediction.is_file()
        else None
    )
    expected_hash = deferred_evaluation.get("prediction_sha256")
    if actual_hash != expected_hash:
        raise ValueError(f"Prediction hash mismatch: {actual_hash} != {expected_hash}")
    task = replace(
        task,
        input_image=run_dir / "sandbox-input" / "input.png",
        ground_truth_step=run_dir / "ground-truth" / "target.step",
    )
    current_provenance = task_provenance(manifest, task)
    for key, value in current_provenance.items():
        if deferred_evaluation.get(key) != value:
            raise ValueError(
                f"Deferred task provenance mismatch for {key}: "
                f"{deferred_evaluation.get(key)!r} != {value!r}"
            )
    result = score_prediction_with_retry(
        task=task,
        workspace_dir=run_dir / "workspace",
        output_dir=run_dir / "evaluation" / "offline-score",
        image=args.image,
        benchcad_repo=Path(args.benchcad_repo).expanduser().resolve(),
        benchmark_commit=manifest.benchmark_commit,
    )
    if result.get("prediction_sha256") != expected_hash:
        raise RuntimeError("Scorer did not evaluate the deferred prediction")
    if result.get("ground_truth_sha256") != current_provenance["ground_truth_sha256"]:
        raise RuntimeError("Scorer did not evaluate the deferred ground truth")
    expected_image = deferred_evaluation.get("container_image_id")
    if expected_image and result.get("container_image_id") != expected_image:
        raise RuntimeError("Scorer image differs from deferred generation image")
    output = run_dir / "evaluation" / "offline-result.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

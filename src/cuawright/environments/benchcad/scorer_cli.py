from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--prediction", required=True)
    execute.add_argument("--output-dir", required=True)
    execute.add_argument("--benchcad-repo", required=True)
    execute.add_argument("--benchmark-commit", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--generated-step", required=True)
    score.add_argument("--ground-truth-step", required=True)
    score.add_argument("--output-dir", required=True)
    score.add_argument("--benchcad-repo", required=True)
    score.add_argument("--benchmark-commit", required=True)
    return parser.parse_args()


def execute(args: argparse.Namespace) -> None:
    sys.path.insert(0, args.benchcad_repo)
    from benchcad_core.scoring.exec_cq import execute_cq_to_step, extract_code

    prediction = Path(args.prediction)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = prediction.read_text(encoding="utf-8", errors="replace")
    code = extract_code(raw)
    generated_step = output_dir / "prediction.step"

    status = "ok"
    error = None
    if not code:
        status = "no_code"
        error = "No parseable CadQuery code"
    else:
        try:
            execute_cq_to_step(code, generated_step, timeout=300)
        except Exception as exc:  # noqa: BLE001 - generated code may fail arbitrarily
            status = "exec_fail"
            error = f"{type(exc).__name__}: {exc}"
    result = {
        "status": status,
        "error": error,
        "benchmark_commit": args.benchmark_commit,
        "prediction_path": str(prediction),
        "generated_step": (str(generated_step) if generated_step.is_file() else None),
    }
    (output_dir / "execution.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result))


def score(args: argparse.Namespace) -> None:
    sys.path.insert(0, args.benchcad_repo)
    from benchcad_core.scoring.iou import iou_step_vs_step

    generated_step = Path(args.generated_step)
    ground_truth = Path(args.ground_truth_step)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    iou = float(iou_step_vs_step(generated_step, ground_truth))
    result = {
        "score": iou,
        "iou": iou,
        "status": "ok",
        "error": None,
        "benchmark_commit": args.benchmark_commit,
        "generated_step": str(generated_step),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result))


def main() -> None:
    args = parse_args()
    if args.operation == "execute":
        execute(args)
    else:
        score(args)


if __name__ == "__main__":
    main()

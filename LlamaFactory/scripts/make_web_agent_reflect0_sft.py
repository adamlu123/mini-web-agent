"""Build label=0 self_reflection_final SFT examples.

For each task directory, walk final_runs from the latest run backwards and pick
the first run whose judge verdict is failure (predicted_label != 1 and the
final_response ends with ``Status: failure``). One negative per task, capped at
--max-examples, constructed with the same code path as the positives so the
format matches web_agent_seq_om2w4000_run1.json exactly.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from make_web_agent_aux_sft import _self_reflection_examples_from_result

STATUS_RE = re.compile(r"Status:\s*(success|failure)", re.IGNORECASE)


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _is_label_zero(result: dict[str, Any]) -> bool:
    label = result.get("predicted_label")
    if label in (1, "1", True):
        return False
    match = STATUS_RE.search(str(result.get("final_response") or result.get("response") or ""))
    return bool(match) and match.group(1).lower() == "failure"


def _run_number(run_dir: Path) -> int:
    match = re.search(r"(\d+)$", run_dir.name)
    return int(match.group(1)) if match else -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", required=True, help="run output roots containing task dirs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--normalize-paths", dest="normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize-paths", dest="normalize", action="store_false")
    args = parser.parse_args()

    task_dirs: list[Path] = []
    for src in args.src:
        root = Path(src).expanduser()
        task_dirs.extend(sorted(p for p in root.iterdir() if p.is_dir() and (p / "final_runs").is_dir()))

    examples: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for task_dir in task_dirs:
        if len(examples) >= args.max_examples:
            break
        stats["tasks_scanned"] += 1
        run_dirs = sorted((task_dir / "final_runs").iterdir(), key=_run_number, reverse=True)
        picked = False
        for run_dir in run_dirs:
            result_path = run_dir / "judge_result.json"
            result = _read_json_optional(result_path)
            if result is None or not _is_label_zero(result):
                continue
            new_examples, n_missing = _self_reflection_examples_from_result(
                result_path,
                result,
                normalize=args.normalize,
                max_chars=args.max_text_chars,
            )
            stats["missing_images"] += n_missing
            finals = [e for e in new_examples if e.get("aux_type") == "self_reflection_final"]
            if not finals:
                stats["label0_runs_without_final_example"] += 1
                continue
            example = finals[0]
            # source judge_result.json stores null for failure verdicts; normalize
            # the provenance field so downstream label filters see an explicit 0
            example["predicted_label"] = 0
            examples.append(example)
            picked = True
            break
        stats["tasks_with_negative" if picked else "tasks_without_negative"] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"examples\t{len(examples)}")
    for key, value in sorted(stats.items()):
        print(f"{key}\t{value}")
    verdicts = Counter(
        (STATUS_RE.search(e["conversations"][-1]["value"]) or [None, "NO_STATUS"])[1].lower() for e in examples
    )
    print(f"verdict_distribution\t{dict(verdicts)}")


if __name__ == "__main__":
    main()

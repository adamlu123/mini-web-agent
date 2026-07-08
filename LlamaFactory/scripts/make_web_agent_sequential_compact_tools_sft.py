#!/usr/bin/env python3
"""Build ordered compact-window + tool SFT data for web-agent rollouts.

This combines three auxiliary data types in a deterministic order:

1. trajectory_session examples for every trajectory. If real compact summary
    model calls exist, the trajectory is split on those boundaries and a final
    tail session is kept. For example, a 65-step trajectory with summaries after
    10, 20, ..., 60 becomes seven ordered sessions:
    [1-10 + summary_prompt -> summary], [11-20 + summary_prompt -> summary],
    ..., [61-65]. If a trajectory has no compact summaries, it is kept as one
    full session so every trajectory is used.
2. image_qa examples extracted from real image_qa tool calls.
3. self_reflection image/final examples extracted from judge_result.json files.

The compact examples are not shuffled: trajectory paths are sorted, and windows
within each trajectory are emitted in chronological order. Tool examples are
appended after all compact examples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
for _path in (_REPO, _SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from make_web_agent_aux_sft import (  # noqa: E402
    DEFAULT_SRCS,
    _compact_examples_from_traj as _session_compact_examples,
    _read_json as _read_json_optional,
    _self_reflection_examples_from_result,
)
from make_web_agent_aux_unified_sft import (  # noqa: E402
    image_qa_examples_from_task,
)
from make_web_agent_compact_window_sft import (  # noqa: E402
    DEFAULT_SUMMARY_USER_PROMPT,
    PROMPT_SYSTEMS,
    _append_turn,
    _assistant_state_from_step,
    _initial_user,
    _is_summary_payload,
    _load_raw_payloads,
    _make_cleaner,
    _original_task,
    _payload_summary,
    _read_json,
    _state_value,
    _step_rows,
    _wrapped_summary,
)
from make_web_agent_sft import build_convo, render_observation_from_debug, workspace_from_traj  # noqa: E402


DEFAULT_ALL_SRCS = [
    "/data/t-yifeili/sft_data/pae_100",
    "/data/t-yifeili/0601/0601/N500_s100_agnostic_r2_success",
    "/data/t-yifeili/webchain_sampling/outputs",
]


def _iter_files(srcs: Iterable[str | Path], name: str) -> list[Path]:
    out: list[Path] = []
    for raw_src in srcs:
        src = Path(raw_src).expanduser()
        if src.is_file() and src.name == name:
            out.append(src)
        elif src.is_dir() and (src / name).is_file():
            out.append(src / name)
        elif src.is_dir():
            for root, _dirs, files in os.walk(src, followlinks=True):
                if name in files:
                    out.append(Path(root) / name)
    return sorted(set(out))


def _iter_task_dirs(srcs: Iterable[str | Path]) -> list[Path]:
    dirs: set[Path] = set()
    for raw_src in srcs:
        src = Path(raw_src).expanduser()
        if src.is_file():
            dirs.add(src.parent)
        elif src.is_dir() and (src / "trajectory.json").is_file():
            dirs.add(src)
        elif src.is_dir():
            for root, _dirs, files in os.walk(src, followlinks=True):
                if "trajectory.json" in files:
                    dirs.add(Path(root))
    return sorted(dirs)


def _is_label_one(value: Any) -> bool:
    return value == 1 or value == "1" or value is True


def _judge_result_label_is_one(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if _is_label_one(data.get("predicted_label")):
        return True
    labels = data.get("all_predicted_labels")
    if isinstance(labels, list) and any(_is_label_one(label) for label in labels):
        return True
    return False


def _task_has_self_judge_label1(task_dir: Path) -> bool:
    for result_path in sorted(task_dir.glob("final_runs/run_*/judge_result.json")):
        if _judge_result_label_is_one(result_path):
            return True
    return False


def _raw_summaries_by_end(raw_payloads: dict[int, dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw_index in sorted(raw_payloads):
        if raw_index <= 1:
            continue
        if _is_summary_payload(raw_payloads[raw_index]):
            summary = _payload_summary(raw_payloads[raw_index])
            if summary:
                out[raw_index - 1] = summary
    return out


def _recovery_summaries_by_end(task_dir: Path) -> dict[int, str]:
    recovery_dir = task_dir / "compact_recovery"
    if not recovery_dir.is_dir():
        return {}
    out: dict[int, str] = {}
    for input_path in sorted(recovery_dir.glob("*_compact_step*_input_payload.json")):
        prefix = input_path.name.removesuffix("_input_payload.json")
        match = re.match(r"^\d+_compact_step(\d+)$", prefix)
        if not match:
            continue
        raw_output_path = recovery_dir / f"{prefix}_output_raw.txt"
        parsed_path = recovery_dir / f"{prefix}_parsed_message.json"
        summary = ""
        if raw_output_path.is_file():
            summary = raw_output_path.read_text(encoding="utf-8", errors="replace").strip()
        elif parsed_path.is_file():
            try:
                parsed = json.loads(parsed_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                summary = str(parsed.get("raw_text") or parsed.get("content") or parsed.get("thought") or "").strip()
        if summary:
            out[int(match.group(1))] = summary
    return out


def _initial_user_from_requests(task_dir: Path, clean) -> str:
    # trajectory.json messages are rewritten in place by the agent's history
    # compaction, so its first user message may already be a "## Compacted
    # History Summary" block. The step-1 debug request payload keeps the
    # pristine initial prompt the model actually saw.
    request_path = task_dir / "debug" / "requests" / "request_0001.json"
    if not request_path.is_file():
        return ""
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    for message in payload.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        text = str(content).strip()
        if text and not text.startswith("## Compacted History Summary"):
            return clean(text)
        return ""
    return ""


def build_sequential_compact_examples(
    traj_path: Path,
    *,
    normalize: bool,
    max_chars: int,
    prompt_mode: str,
    window_size: int,
) -> list[dict[str, Any]]:
    traj = _read_json(traj_path)
    task_dir = traj_path.parent
    workspace = ((traj.get("environment") or {}).get("config") or {}).get("output_dir") or task_dir
    clean = _make_cleaner(workspace, normalize=normalize, max_chars=max_chars)
    raw_payloads = _load_raw_payloads(task_dir)
    steps = _step_rows(task_dir)
    if not steps:
        return []
    summaries_by_end = _raw_summaries_by_end(raw_payloads)
    summary_source = "raw_responses" if summaries_by_end else "none"
    if not summaries_by_end:
        summaries_by_end = _recovery_summaries_by_end(task_dir)
        summary_source = "compact_recovery" if summaries_by_end else "none"

    max_step = max(steps)
    end_calls = sorted(end for end in summaries_by_end if end in steps or any(s <= end for s in steps))
    if not end_calls or end_calls[-1] < max_step:
        end_calls.append(max_step)

    agent_cfg = (((traj.get("info") or {}).get("config") or {}).get("agent") or {})
    summary_prompt = clean(agent_cfg.get("summary_user_prompt") or DEFAULT_SUMMARY_USER_PROMPT)
    original_task = _original_task(task_dir, traj)
    initial_user = _initial_user_from_requests(task_dir, clean) or _initial_user(task_dir, traj, clean)
    system = PROMPT_SYSTEMS[prompt_mode]

    examples: list[dict[str, Any]] = []
    previous_end = 0
    for compact_index, end_call in enumerate(end_calls):
        if end_call <= previous_end:
            continue
        summary = clean(summaries_by_end.get(end_call, ""))

        convo: list[dict[str, str]] = []
        if previous_end <= 0:
            _append_turn(convo, "human", initial_user)
        else:
            previous_payload = raw_payloads.get(previous_end + 1, {})
            previous_summary = _payload_summary(previous_payload)
            _append_turn(
                convo,
                "human",
                clean(_wrapped_summary(original_task=original_task, end_call=previous_end, summary=previous_summary)),
            )

        for step in sorted(s for s in steps if previous_end < s <= end_call):
            row = steps[step]
            _append_turn(convo, "gpt", _assistant_state_from_step(row, clean))
            outputs = row.get("outputs") if isinstance(row.get("outputs"), list) else []
            if outputs:
                observation_text = "\n\n".join(
                    clean(render_observation_from_debug(output)) for output in outputs if isinstance(output, dict)
                )
                if observation_text:
                    _append_turn(convo, "human", observation_text)

        if not convo:
            continue
        if summary:
            if convo[-1]["from"] == "human":
                convo[-1]["value"] = f"{convo[-1]['value'].rstrip()}\n\n{summary_prompt}"
            else:
                _append_turn(convo, "human", summary_prompt)
        while convo and convo[0]["from"] != "human":
            convo.pop(0)
        while summary and len(convo) >= 2 and convo[-1]["from"] != "human":
            convo.pop()
        if not convo or (summary and convo[-1]["from"] != "human"):
            continue
        conversations = list(convo)
        if summary:
            conversations.append({"from": "gpt", "value": _state_value(thought=summary)})

        examples.append(
            {
                "conversations": conversations,
                "system": system,
                "images": [],
                "aux_type": "trajectory_session",
                "source": str(traj_path),
                "session_source": "debug_steps",
                "summary_source": summary_source if summary else "none",
                "window_start_call": previous_end + 1,
                "window_end_call": end_call,
                "summary_raw_index": end_call + 1 if summary else None,
                "has_summary_target": bool(summary),
                "compact_index": compact_index,
                "prompt_mode": prompt_mode,
            }
        )
        previous_end = end_call

    return examples


def _compact_recovery_examples(
    traj_path: Path,
    *,
    normalize: bool,
    max_chars: int,
) -> list[dict[str, Any]]:
    recovery_dir = traj_path.parent / "compact_recovery"
    if not recovery_dir.is_dir():
        return []

    examples: list[dict[str, Any]] = []
    workspace = traj_path.parent
    clean = _make_cleaner(workspace, normalize=normalize, max_chars=max_chars)
    for input_path in sorted(recovery_dir.glob("*_compact_step*_input_payload.json")):
        prefix = input_path.name.removesuffix("_input_payload.json")
        if not re.match(r"^\d+_compact_step\d+$", prefix):
            continue
        raw_output_path = recovery_dir / f"{prefix}_output_raw.txt"
        parsed_path = recovery_dir / f"{prefix}_parsed_message.json"
        target = ""
        if raw_output_path.is_file():
            target = raw_output_path.read_text(encoding="utf-8", errors="replace").strip()
        elif parsed_path.is_file():
            try:
                parsed = json.loads(parsed_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                target = str(parsed.get("raw_text") or parsed.get("content") or parsed.get("thought") or "").strip()
        if not target:
            continue

        try:
            payload = json.loads(input_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) or not messages:
            continue

        system = ""
        convo: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = clean(message.get("content", ""))
            if role == "system" and not system:
                system = content
            elif role == "user":
                _append_turn(convo, "human", content)
            elif role == "assistant":
                _append_turn(convo, "gpt", content)

        while convo and convo[0]["from"] != "human":
            convo.pop(0)
        if not convo or convo[-1]["from"] != "human":
            continue
        match = re.search(r"compact_step(\d+)", prefix)
        summary_raw_index = int(match.group(1)) if match else None
        examples.append(
            {
                "conversations": convo + [{"from": "gpt", "value": clean(target)}],
                "system": system,
                "images": [],
                "aux_type": "trajectory_session",
                "source": str(traj_path),
                "session_source": "compact_recovery",
                "summary_source": "compact_recovery",
                "summary_raw_index": summary_raw_index,
                "has_summary_target": True,
                "compact_recovery_input": str(input_path),
            }
        )
    return examples


def _session_examples_as_state(
    traj_path: Path,
    traj: dict[str, Any],
    *,
    normalize: bool,
    max_chars: int,
    prompt_mode: str,
) -> list[dict[str, Any]]:
    examples = _session_compact_examples(
        traj_path,
        traj,
        normalize=normalize,
        max_chars=max_chars,
        compact_target="raw",
    )
    out: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        converted = dict(example)
        conv = [dict(message) for message in (example.get("conversations") or [])]
        if conv and conv[-1].get("from") == "gpt":
            conv[-1]["value"] = _state_value(thought=str(conv[-1].get("value") or ""))
        converted["conversations"] = conv
        converted["system"] = PROMPT_SYSTEMS[prompt_mode]
        converted["aux_type"] = "trajectory_session"
        converted["session_source"] = "compacted_sessions"
        converted["summary_source"] = "compacted_sessions"
        converted["has_summary_target"] = True
        converted.setdefault("compact_index", index)
        out.append(converted)
    return out


def _messages_session_example(
    traj_path: Path,
    traj: dict[str, Any],
    *,
    normalize: bool,
    max_chars: int,
    prompt_mode: str,
) -> list[dict[str, Any]]:
    messages = traj.get("messages") if isinstance(traj.get("messages"), list) else []
    if not messages:
        return []
    workspace = workspace_from_traj(traj, str(traj_path)) or str(traj_path.parent)
    norm = _make_cleaner(workspace, normalize=normalize, max_chars=0)
    convo = build_convo(messages, norm, max_chars)
    if not convo or not any(turn.get("from") == "gpt" for turn in convo):
        return []
    return [
        {
            "conversations": convo,
            "system": PROMPT_SYSTEMS[prompt_mode],
            "images": [],
            "aux_type": "trajectory_session",
            "source": str(traj_path),
            "session_source": "messages",
            "summary_source": "none",
            "has_summary_target": False,
            "compact_index": 0,
        }
    ]


def build_all_compact_examples(
    traj_path: Path,
    *,
    normalize: bool,
    max_chars: int,
    prompt_mode: str,
    window_size: int,
) -> list[dict[str, Any]]:
    raw_examples = build_sequential_compact_examples(
        traj_path,
        normalize=normalize,
        max_chars=max_chars,
        prompt_mode=prompt_mode,
        window_size=window_size,
    )
    if raw_examples:
        return raw_examples

    try:
        traj = _read_json(traj_path)
    except Exception:
        traj = {}
    session_examples = _session_examples_as_state(
        traj_path,
        traj,
        normalize=normalize,
        max_chars=max_chars,
        prompt_mode=prompt_mode,
    )
    if session_examples:
        return session_examples

    recovery_examples = _compact_recovery_examples(traj_path, normalize=normalize, max_chars=max_chars)
    if recovery_examples:
        return recovery_examples

    return _messages_session_example(
        traj_path,
        traj,
        normalize=normalize,
        max_chars=max_chars,
        prompt_mode=prompt_mode,
    )


def _escape_stray_mm_tokens(examples: list[dict[str, Any]]) -> None:
    """Escape literal <video>/<image> strings coming from webpage text.

    LlamaFactory's mm plugin treats bare "<video>"/"<image>" as media
    placeholders and hard-fails when their count mismatches the sample's
    media lists (om2w_4000 run1: 27 samples with "<video>" from video-related
    pages crashed preprocessing). We never emit videos, so every "<video>" is
    literal text; "<image>" is only escaped when the sample declares no images
    (otherwise a mismatch is a real bug and is reported instead).
    """
    n_video = n_image = 0
    for idx, ex in enumerate(examples):
        convo = ex.get("conversations") or []
        n_declared = len(ex.get("images") or [])
        n_tokens = sum((turn.get("value") or "").count("<image>") for turn in convo)
        for turn in convo:
            value = turn.get("value") or ""
            if "<video>" in value:
                n_video += value.count("<video>")
                turn["value"] = value.replace("<video>", "<video >")
            if n_declared == 0 and "<image>" in value:
                n_image += value.count("<image>")
                turn["value"] = turn["value"].replace("<image>", "<image >")
        if n_declared and n_tokens != n_declared:
            print(f"[warn] example {idx}: {n_tokens} <image> tokens vs {n_declared} declared images", file=sys.stderr)
    if n_video or n_image:
        print(f"[info] escaped stray mm tokens: <video> x{n_video}, <image> x{n_image}", file=sys.stderr)


def _write_bundle(examples: list[dict[str, Any]], *, bundle_dir: Path, dataset_name: str, stats: dict[str, Any]) -> None:
    _escape_stray_mm_tokens(examples)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_path = bundle_dir / f"{dataset_name}.json"
    data_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_info = {
        dataset_name: {
            "file_name": data_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
            },
        }
    }
    (bundle_dir / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False), encoding="utf-8")
    (bundle_dir / "manifest.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def _print_stats(stats: dict[str, Any]) -> None:
    print(f"wrote_examples\t{stats['examples']}")
    print(f"image_references\t{stats['image_references']}")
    print(f"sources\t{stats['sources']}")
    print(f"trajectory_files_scanned\t{stats['trajectory_files_scanned']}")
    print(f"self_judge_label1_required\t{stats['self_judge_label1_required']}")
    print(f"trajectory_files_label1\t{stats['trajectory_files_label1']}")
    print(f"trajectory_files_skipped_non_label1\t{stats['trajectory_files_skipped_non_label1']}")
    print(f"session_trajectories\t{stats['session_trajectories']}")
    print(f"dropped_no_supervision\t{len(stats.get('dropped_no_supervision', []))}")
    print(f"trajectory_session_examples\t{stats['counts'].get('trajectory_session', 0)}")
    print(f"trajectory_sessions_with_summary\t{stats['trajectory_sessions_with_summary']}")
    print(f"trajectory_sessions_without_summary\t{stats['trajectory_sessions_without_summary']}")
    print(f"image_qa_examples\t{stats['counts'].get('image_qa', 0)}")
    print(f"max_image_qa\t{stats['max_image_qa']}")
    print(f"self_reflection_image_examples\t{stats['counts'].get('self_reflection_image', 0)}")
    print(f"self_reflection_final_examples\t{stats['counts'].get('self_reflection_final', 0)}")
    print(f"max_self_reflection_image\t{stats['max_self_reflection_image']}")
    print(f"max_self_reflection_final\t{stats['max_self_reflection_final']}")
    print(f"missing_self_reflection_images\t{stats['missing_self_reflection_images']}")
    if stats.get("session_window_lengths"):
        lengths = stats["session_window_lengths"]
        print(f"session_window_min\t{min(lengths)}")
        print(f"session_window_max\t{max(lengths)}")
        print(f"session_window_avg\t{sum(lengths) / len(lengths):.2f}")
    for key in sorted(stats.get("session_source_counts", {})):
        print(f"session_source.{key}\t{stats['session_source_counts'][key]}")
    for key in sorted(stats.get("summary_source_counts", {})):
        print(f"summary_source.{key}\t{stats['summary_source_counts'][key]}")
    for key in sorted(stats["counts"]):
        print(f"count.{key}\t{stats['counts'][key]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", default=DEFAULT_ALL_SRCS, help="trajectory/task/source dirs for compact + tools")
    parser.add_argument("--tool-src", nargs="+", default=None, help="optional separate source dirs for image_qa/self_reflection")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--dataset-name", default="web_agent_sequential_compact_tools")
    parser.add_argument("--prompt-mode", choices=sorted(PROMPT_SYSTEMS), default="sft_state_debug")
    parser.add_argument("--window-size", type=int, default=20, help="compact window size; also keeps the last tail summary")
    parser.add_argument("--max-text-chars", type=int, default=0)
    parser.add_argument("--max-image-qa", type=int, default=1000, help="cap image_qa samples; 0 means no cap")
    parser.add_argument("--max-self-reflection", type=int, default=0, help="legacy cap on judge_result files; 0 means no cap")
    parser.add_argument("--max-self-reflection-image", type=int, default=1000, help="cap self_reflection_image samples; 0 means no cap")
    parser.add_argument("--max-self-reflection-final", type=int, default=1000, help="cap self_reflection_final samples; 0 means no cap")
    parser.add_argument("--no-normalize-paths", dest="normalize", action="store_false")
    parser.add_argument("--no-image-qa", action="store_true")
    parser.add_argument("--no-self-reflection", action="store_true")
    parser.add_argument("--require-self-judge-label1", dest="require_label1", action="store_true", default=True)
    parser.add_argument("--no-require-self-judge-label1", dest="require_label1", action="store_false")
    args = parser.parse_args()

    srcs = [str(Path(src).expanduser()) for src in args.src]
    tool_srcs = [str(Path(src).expanduser()) for src in (args.tool_src or args.src)]
    examples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    session_by_source: dict[str, int] = defaultdict(int)
    session_source_counts: Counter[str] = Counter()
    summary_source_counts: Counter[str] = Counter()
    session_window_lengths: list[int] = []
    dropped_no_supervision: list[str] = []
    skipped_non_label1: list[str] = []

    traj_paths = _iter_files(srcs, "trajectory.json")
    label1_task_dirs = {
        traj_path.parent.resolve()
        for traj_path in traj_paths
        if _task_has_self_judge_label1(traj_path.parent)
    }
    for traj_path in traj_paths:
        if args.require_label1 and traj_path.parent.resolve() not in label1_task_dirs:
            skipped_non_label1.append(str(traj_path))
            continue
        session_examples = build_all_compact_examples(
            traj_path,
            normalize=args.normalize,
            max_chars=args.max_text_chars,
            prompt_mode=args.prompt_mode,
            window_size=args.window_size,
        )
        if not session_examples:
            dropped_no_supervision.append(str(traj_path))
            continue
        examples.extend(session_examples)
        counts["trajectory_session"] += len(session_examples)
        session_by_source[str(traj_path)] += len(session_examples)
        for example in session_examples:
            session_source_counts[str(example.get("session_source") or "unknown")] += 1
            summary_source_counts[str(example.get("summary_source") or "none")] += 1
            if example.get("window_start_call") is not None and example.get("window_end_call") is not None:
                session_window_lengths.append(int(example["window_end_call"]) - int(example["window_start_call"]) + 1)

    image_qa_seen = 0
    if not args.no_image_qa:
        for task_dir in _iter_task_dirs(tool_srcs):
            if args.require_label1 and not _task_has_self_judge_label1(task_dir):
                continue
            remaining = max(0, args.max_image_qa - image_qa_seen) if args.max_image_qa else 0
            if args.max_image_qa and remaining <= 0:
                break
            new_examples = image_qa_examples_from_task(task_dir, max_examples=remaining)
            examples.extend(new_examples)
            image_qa_seen += len(new_examples)
            counts["image_qa"] += len(new_examples)

    missing_self_reflection_images = 0
    judge_seen = 0
    self_reflection_image_seen = 0
    self_reflection_final_seen = 0
    if not args.no_self_reflection:
        for result_path in _iter_files(tool_srcs, "judge_result.json"):
            if args.require_label1 and not _judge_result_label_is_one(result_path):
                continue
            if args.max_self_reflection and judge_seen >= args.max_self_reflection:
                break
            result = _read_json_optional(result_path)
            if result is None:
                continue
            new_examples, n_missing = _self_reflection_examples_from_result(
                result_path,
                result,
                normalize=args.normalize,
                max_chars=args.max_text_chars,
            )
            missing_self_reflection_images += n_missing
            if not new_examples:
                continue
            kept_examples: list[dict[str, Any]] = []
            for example in new_examples:
                aux_type = str(example.get("aux_type") or "")
                if aux_type == "self_reflection_image":
                    if args.max_self_reflection_image and self_reflection_image_seen >= args.max_self_reflection_image:
                        continue
                    self_reflection_image_seen += 1
                elif aux_type == "self_reflection_final":
                    if args.max_self_reflection_final and self_reflection_final_seen >= args.max_self_reflection_final:
                        continue
                    self_reflection_final_seen += 1
                kept_examples.append(example)
            if not kept_examples:
                if args.max_self_reflection_image and args.max_self_reflection_final:
                    if self_reflection_image_seen >= args.max_self_reflection_image and self_reflection_final_seen >= args.max_self_reflection_final:
                        break
                continue
            examples.extend(kept_examples)
            judge_seen += 1
            for example in kept_examples:
                counts[str(example.get("aux_type"))] += 1
            if args.max_self_reflection_image and args.max_self_reflection_final:
                if self_reflection_image_seen >= args.max_self_reflection_image and self_reflection_final_seen >= args.max_self_reflection_final:
                    break

    stats: dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "examples": len(examples),
        "image_references": sum(len(example.get("images") or []) for example in examples),
        "sources": srcs,
        "tool_sources": tool_srcs,
        "trajectory_files_scanned": len(traj_paths),
        "self_judge_label1_required": args.require_label1,
        "trajectory_files_label1": len(label1_task_dirs),
        "trajectory_files_skipped_non_label1": len(skipped_non_label1),
        "skipped_non_label1": skipped_non_label1,
        "session_trajectories": len(session_by_source),
        "dropped_no_supervision": dropped_no_supervision,
        "session_by_source": dict(session_by_source),
        "session_source_counts": dict(session_source_counts),
        "summary_source_counts": dict(summary_source_counts),
        "trajectory_sessions_with_summary": sum(1 for example in examples if example.get("aux_type") == "trajectory_session" and example.get("has_summary_target")),
        "trajectory_sessions_without_summary": sum(1 for example in examples if example.get("aux_type") == "trajectory_session" and not example.get("has_summary_target")),
        "session_window_lengths": session_window_lengths,
        "window_size": args.window_size,
        "counts": dict(counts),
        "missing_self_reflection_images": missing_self_reflection_images,
        "max_image_qa": args.max_image_qa,
        "max_self_reflection_image": args.max_self_reflection_image,
        "max_self_reflection_final": args.max_self_reflection_final,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.bundle_dir is not None:
        _write_bundle(examples, bundle_dir=args.bundle_dir, dataset_name=args.dataset_name, stats=stats)
    _print_stats(stats)
    print(f"output_path\t{args.out}")
    if args.bundle_dir is not None:
        print(f"bundle_dir\t{args.bundle_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
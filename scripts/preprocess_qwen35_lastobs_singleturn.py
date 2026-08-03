#!/usr/bin/env python3
"""Build a Qwen3.5 SPB last-observation, single-turn ShareGPT bundle.

The source is a mini-web-agent rollout directory containing one task directory
per trajectory.  Each output row targets exactly one assistant step:

* the row contains the real system/task/history prefix through that step;
* older command observations are stubbed at ``Command output:``;
* the current observation block is kept in full;
* every historical assistant state keeps its parsed thought and bash command;
* the final assistant in the row is the only training target (the tokenizer
  must use ``assistant_mask_mode=last``).

This mirrors ``DefaultAgent._transform_history(history_context_mode="last_obs")``
and ``_sft_state_assistant_content``.  Parsed ``extra.raw_response`` values are
authoritative; raw_responses.jsonl can include rejected retries or additional
imagined state blocks and is intentionally not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


FORMAT_VERSION = 1
DEFAULT_DATASET_NAME = "om2w_spb_lastobs_singleturn"
OBSERVATION_MARKER = "Command output:\n"
OBSERVATION_STUB = "Command output: (omitted)"
IMAGE_SENTINEL = "<image>"
ESCAPED_IMAGE_SENTINEL = "&lt;image&gt;"

# Redaction is unconditional. Path normalization is separately configurable.
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"bb_(?:live|test)_[A-Za-z0-9_\-]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-=]{16,}"),
    re.compile(
        r"(?i)\b(?:OPENAI|OPENROUTER|BROWSERBASE|ANTHROPIC|AZURE|AWS|HF|"
        r"HUGGINGFACE|GITHUB|SLACK)_[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)"
        r"\s*=\s*['\"]?[^\s'\"]{8,}"
    ),
)


class ConversionError(ValueError):
    """A trajectory cannot be converted without silently changing semantics."""


@dataclass
class SanitizationStats:
    secret_redactions: int = 0
    escaped_image_sentinels: int = 0
    workspace_path_replacements: int = 0
    portable_path_replacements: int = 0


@dataclass
class ConversionStats:
    source_trajectories: int = 0
    selected_trajectories: int = 0
    output_rows: int = 0
    historical_observations_stubbed: int = 0
    adjacent_user_blocks_merged: int = 0
    empty_thought_targets: int = 0
    exit_statuses: Counter[str] = field(default_factory=Counter)
    skipped_reasons: Counter[str] = field(default_factory=Counter)
    sanitization: SanitizationStats = field(default_factory=SanitizationStats)


@dataclass(frozen=True)
class EvalSuccessFilter:
    summary_path: Path
    result_path: Path
    success_label: int
    labels_by_task: dict[str, int]
    summary_sha256: str
    result_sha256: str
    summary_successful_tasks: int | None

    @property
    def successful_task_ids(self) -> set[str]:
        return {
            task_id
            for task_id, predicted_label in self.labels_by_task.items()
            if predicted_label == self.success_label
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_or_jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            rows = json.load(handle)
            if not isinstance(rows, list):
                raise ConversionError(f"evaluation result is not a JSON array: {path}")
            for row in rows:
                if isinstance(row, dict):
                    yield row
            return
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ConversionError(
                    f"invalid evaluation JSONL at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ConversionError(
                    f"evaluation row at {path}:{line_number} is not an object"
                )
            yield row


def load_eval_success_filter(
    summary_path: Path,
    *,
    success_label: int = 1,
) -> EvalSuccessFilter:
    summary_path = summary_path.expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"evaluation summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ConversionError(f"evaluation summary is not an object: {summary_path}")
    result_file = summary.get("result_file")
    if not isinstance(result_file, str) or not result_file:
        raise ConversionError(f"evaluation summary has no result_file: {summary_path}")
    result_path = Path(result_file).expanduser()
    if not result_path.is_absolute():
        result_path = summary_path.parent / result_path
    if not result_path.is_file():
        local_fallback = summary_path.parent / Path(result_file).name
        if local_fallback.is_file():
            result_path = local_fallback
        else:
            raise FileNotFoundError(
                f"evaluation result referenced by {summary_path} was not found: {result_path}"
            )
    result_path = result_path.resolve()

    labels_by_task: dict[str, int] = {}
    for row in _load_json_or_jsonl_rows(result_path):
        task_id = row.get("task_id")
        label = row.get("predicted_label")
        if not isinstance(task_id, str) or not task_id:
            raise ConversionError(f"evaluation result row has no task_id: {result_path}")
        if isinstance(label, bool) or not isinstance(label, (int, float)):
            raise ConversionError(
                f"evaluation result for {task_id} has invalid predicted_label={label!r}"
            )
        int_label = int(label)
        if float(label) != float(int_label):
            raise ConversionError(
                f"evaluation result for {task_id} has non-integral predicted_label={label!r}"
            )
        previous = labels_by_task.get(task_id)
        if previous is not None and previous != int_label:
            raise ConversionError(
                f"evaluation result has conflicting labels for {task_id}: "
                f"{previous} and {int_label}"
            )
        labels_by_task[task_id] = int_label

    expected_successes = summary.get("successful_tasks")
    if expected_successes is not None:
        expected_successes = int(expected_successes)
        actual_successes = sum(label == success_label for label in labels_by_task.values())
        if actual_successes != expected_successes:
            raise ConversionError(
                f"evaluation summary says successful_tasks={expected_successes}, "
                f"but {result_path} contains {actual_successes} rows with label={success_label}"
            )
    return EvalSuccessFilter(
        summary_path=summary_path,
        result_path=result_path,
        success_label=success_label,
        labels_by_task=labels_by_task,
        summary_sha256=_sha256_file(summary_path),
        result_sha256=_sha256_file(result_path),
        summary_successful_tasks=expected_successes,
    )


def _json_dump_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def text_of(content: Any) -> str:
    """Flatten the text-only message content used by this collection."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                raise ConversionError(f"unsupported message content item: {type(item).__name__}")
            item_type = str(item.get("type", ""))
            if item_type in {"text", "input_text"} and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            if item_type in {"image", "input_image", "image_url", "video", "video_url"}:
                raise ConversionError(
                    "source contains an attached image/video; this text-only SPB converter "
                    "does not silently omit multimodal input"
                )
            raise ConversionError(f"unsupported message content type: {item_type!r}")
        return "\n".join(parts)
    if content is None:
        return ""
    raise ConversionError(f"unsupported message content: {type(content).__name__}")


def trajectory_exit_status(trajectory: dict[str, Any]) -> str:
    info = trajectory.get("info")
    if isinstance(info, dict) and info.get("exit_status"):
        return str(info["exit_status"])
    for message in reversed(trajectory.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "exit":
            continue
        extra = message.get("extra")
        if isinstance(extra, dict) and extra.get("exit_status"):
            return str(extra["exit_status"])
    return "UNKNOWN"


def trajectory_task_id(trajectory: dict[str, Any], trajectory_path: Path) -> str:
    task_path = trajectory_path.parent / "task.json"
    if task_path.is_file():
        try:
            task = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            task = {}
        if isinstance(task, dict) and task.get("task_id"):
            return str(task["task_id"])
    for message in trajectory.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        for line in text_of(message.get("content", "")).splitlines():
            if line.startswith("Task ID: "):
                return line.removeprefix("Task ID: ").strip()
        break
    return trajectory_path.parent.name


def _workspace_candidates(trajectory: dict[str, Any], trajectory_path: Path) -> list[str]:
    candidates = {
        str(trajectory_path.parent.absolute()),
        str(trajectory_path.parent.resolve()),
    }
    environment = trajectory.get("environment")
    if isinstance(environment, dict):
        if environment.get("workspace_dir"):
            candidates.add(str(environment["workspace_dir"]))
        config = environment.get("config")
        if isinstance(config, dict) and config.get("output_dir"):
            candidates.add(str(config["output_dir"]))
    info = trajectory.get("info")
    config = info.get("config") if isinstance(info, dict) else None
    if isinstance(config, dict):
        environment_config = config.get("environment")
        if isinstance(environment_config, dict) and environment_config.get("output_dir"):
            candidates.add(str(environment_config["output_dir"]))
        agent_config = config.get("agent")
        if isinstance(agent_config, dict) and agent_config.get("output_path"):
            candidates.add(str(Path(str(agent_config["output_path"])).parent))
    return sorted((item for item in candidates if item and item != "."), key=len, reverse=True)


class TextSanitizer:
    """Redact secrets, protect literal image text, and optionally normalize paths."""

    def __init__(
        self,
        trajectory: dict[str, Any],
        trajectory_path: Path,
        *,
        path_mode: str,
        stats: SanitizationStats,
    ) -> None:
        self.path_mode = path_mode
        self.stats = stats
        self.workspace_paths = _workspace_candidates(trajectory, trajectory_path)
        self.portable_paths: list[tuple[str, str]] = []
        if path_mode == "repo-portable":
            repo_roots: set[str] = set()
            home_roots: set[str] = set()
            for candidate in self.workspace_paths:
                marker = "/sandbox/mini-web-agent"
                marker_index = candidate.find(marker)
                if marker_index >= 0:
                    repo_roots.add(candidate[: marker_index + len(marker)])
                home_match = re.match(r"(/home/[^/]+)", candidate)
                if home_match:
                    home_roots.add(home_match.group(1))
            self.portable_paths.extend((path, "/opt/mini-web-agent") for path in repo_roots)
            self.portable_paths.extend((path, "/home/agent") for path in home_roots)
            self.portable_paths.sort(key=lambda item: len(item[0]), reverse=True)

    def __call__(self, text: str) -> str:
        value = str(text)
        for pattern in SECRET_PATTERNS:
            value, count = pattern.subn("<REDACTED_SECRET>", value)
            self.stats.secret_redactions += count

        count = value.count(IMAGE_SENTINEL)
        if count:
            value = value.replace(IMAGE_SENTINEL, ESCAPED_IMAGE_SENTINEL)
            self.stats.escaped_image_sentinels += count

        if self.path_mode in {"workspace", "repo-portable"}:
            for source in self.workspace_paths:
                count = value.count(source)
                if count:
                    value = value.replace(source, "/workspace")
                    self.stats.workspace_path_replacements += count

        if self.path_mode == "repo-portable":
            for source, destination in self.portable_paths:
                count = value.count(source)
                if count:
                    value = value.replace(source, destination)
                    self.stats.portable_path_replacements += count
        return value


def stub_historical_observation(text: str) -> tuple[str, bool]:
    if OBSERVATION_MARKER not in text:
        return text, False
    prefix = text.split(OBSERVATION_MARKER, 1)[0]
    return prefix + OBSERVATION_STUB, True


def _raw_assistant_state(message: dict[str, Any]) -> tuple[str, str, bool, str]:
    extra = message.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    raw = extra.get("raw_response")
    if not isinstance(raw, dict):
        raise ConversionError("assistant message is missing authoritative extra.raw_response")
    required = {"thought", "bash_command", "done", "final_response"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ConversionError(f"assistant raw_response is missing fields: {missing}")

    thought = raw["thought"]
    bash_command = raw["bash_command"] or raw.get("python_code") or ""
    done = bool(raw["done"])
    final_response = raw["final_response"]
    return (
        text_of(thought).strip(),
        str(bash_command).strip(),
        done,
        str(final_response or "").strip(),
    )


def render_assistant_state(message: dict[str, Any], sanitize: TextSanitizer) -> str:
    thought, bash_command, done, final_response = _raw_assistant_state(message)
    if not done and not bash_command:
        raise ConversionError("non-terminal assistant state has an empty bash command")
    if done and bash_command:
        raise ConversionError("terminal assistant state has a non-empty bash command")
    if done and not final_response:
        raise ConversionError("terminal assistant state has an empty final_response")
    return (
        f"<think>\n{sanitize(thought)}\n</think>\n"
        f"<bash>\n{sanitize(bash_command)}\n</bash>\n"
        f"<done>{'true' if done else 'false'}</done>\n"
        f"<final_response>\n{sanitize(final_response)}\n</final_response>"
    )


def _append_turn(
    conversations: list[dict[str, str]],
    turn: dict[str, str],
    stats: ConversionStats,
) -> None:
    if conversations and conversations[-1]["from"] == turn["from"]:
        if turn["from"] != "human":
            raise ConversionError("adjacent assistant turns cannot be merged safely")
        conversations[-1]["value"] += "\n\n" + turn["value"]
        stats.adjacent_user_blocks_merged += 1
        return
    conversations.append(turn)


def build_singleturn_rows(
    trajectory: dict[str, Any],
    trajectory_path: Path,
    *,
    path_mode: str,
    stats: ConversionStats,
) -> list[dict[str, Any]]:
    """Return one prefix-to-target row for every assistant message."""

    if trajectory.get("compacted_sessions"):
        raise ConversionError(
            "compacted_sessions is non-empty; this script targets the no-compaction SPB collection"
        )
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ConversionError("trajectory has no messages")

    system_positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "system"
    ]
    if system_positions != [0]:
        raise ConversionError(f"expected exactly one leading system message, got {system_positions}")

    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistant_positions:
        raise ConversionError("trajectory has no assistant messages")

    task_id = trajectory_task_id(trajectory, trajectory_path)
    sanitizer = TextSanitizer(
        trajectory,
        trajectory_path,
        path_mode=path_mode,
        stats=stats.sanitization,
    )
    system = sanitizer(text_of(messages[0].get("content", "")).strip())
    if not system:
        raise ConversionError("system prompt is empty")

    rows: list[dict[str, Any]] = []
    for target_step, target_index in enumerate(assistant_positions, start=1):
        prior_assistants = [index for index in assistant_positions if index < target_index]
        # This is the request prefix before the target was generated. Everything
        # after the previous assistant is the full current observation/retry block.
        current_block_start = prior_assistants[-1] + 1 if prior_assistants else target_index
        conversations: list[dict[str, str]] = []

        for index, message in enumerate(messages[: target_index + 1]):
            if not isinstance(message, dict):
                raise ConversionError(f"message {index} is not an object")
            role = message.get("role")
            if role == "system":
                continue
            if role == "exit":
                continue
            if role == "user":
                value = text_of(message.get("content", "")).strip()
                if 1 < index < current_block_start:
                    value, stubbed = stub_historical_observation(value)
                    if stubbed:
                        stats.historical_observations_stubbed += 1
                value = sanitizer(value)
                if value:
                    _append_turn(
                        conversations,
                        {"from": "human", "value": value},
                        stats,
                    )
                continue
            if role == "assistant":
                _append_turn(
                    conversations,
                    {
                        "from": "gpt",
                        "value": render_assistant_state(message, sanitizer),
                    },
                    stats,
                )
                continue
            raise ConversionError(f"unsupported role {role!r} at message {index}")

        if not conversations or conversations[0]["from"] != "human":
            raise ConversionError(f"target step {target_step} does not start with a human turn")
        if conversations[-1]["from"] != "gpt":
            raise ConversionError(f"target step {target_step} does not end with the target assistant")
        if sum(turn["from"] == "gpt" for turn in conversations) != target_step:
            raise ConversionError(f"target step {target_step} lost an assistant prefix turn")

        target_thought, _, _, _ = _raw_assistant_state(messages[target_index])
        if not target_thought.strip():
            stats.empty_thought_targets += 1
        rows.append(
            {
                "id": f"{task_id}-step-{target_step:04d}",
                "system": system,
                "conversations": conversations,
                "images": [],
                "metadata": {
                    "task_id": task_id,
                    "target_step": target_step,
                    "source_message_index": target_index,
                    "history_context_mode": "last_obs",
                    "turn_mode": "single",
                    "assistant_mask_mode": "last",
                },
            }
        )
    return rows


def iter_trajectory_paths(source_dir: Path) -> Iterator[Path]:
    for task_dir in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if not task_dir.is_dir():
            continue
        trajectory_path = task_dir / "trajectory.json"
        if trajectory_path.is_file():
            yield trajectory_path


def _dataset_info(dataset_name: str, data_file_name: str) -> dict[str, Any]:
    return {
        dataset_name: {
            "file_name": data_file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
                "images": "images",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
            },
        }
    }


def preprocess(
    source_dir: Path,
    output_dir: Path,
    *,
    dataset_name: str,
    required_exit_status: str | None,
    path_mode: str,
    eval_filter: EvalSuccessFilter | None = None,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().absolute()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new path "
            "or remove the prior generated artifact explicitly"
        )

    stage_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if stage_dir.exists():
        raise FileExistsError(f"staging directory already exists: {stage_dir}")
    stage_dir.mkdir(parents=True)

    stats = ConversionStats()
    source_tasks: list[dict[str, Any]] = []
    data_file_name = f"{dataset_name}.jsonl"
    data_path = stage_dir / data_file_name

    try:
        with data_path.open("w", encoding="utf-8") as output:
            for trajectory_path in iter_trajectory_paths(source_dir):
                stats.source_trajectories += 1
                try:
                    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                    if not isinstance(trajectory, dict):
                        raise ConversionError("trajectory root is not an object")
                    status = trajectory_exit_status(trajectory)
                    stats.exit_statuses[status] += 1
                    task_id = trajectory_task_id(trajectory, trajectory_path)
                    assistant_count = sum(
                        isinstance(message, dict) and message.get("role") == "assistant"
                        for message in trajectory.get("messages") or []
                    )
                    task_record = {
                        "task_id": task_id,
                        "source_dir_name": trajectory_path.parent.name,
                        "exit_status": status,
                        "assistant_turns": assistant_count,
                    }
                    if required_exit_status is not None and status != required_exit_status:
                        reason = f"exit_status_not_{required_exit_status}"
                        stats.skipped_reasons[reason] += 1
                        source_tasks.append({**task_record, "selected": False, "reason": reason})
                        continue
                    if eval_filter is not None:
                        predicted_label = eval_filter.labels_by_task.get(task_id)
                        task_record["eval_predicted_label"] = predicted_label
                        if predicted_label is None:
                            reason = "missing_eval_label"
                            stats.skipped_reasons[reason] += 1
                            source_tasks.append(
                                {**task_record, "selected": False, "reason": reason}
                            )
                            continue
                        if predicted_label != eval_filter.success_label:
                            reason = f"eval_label_not_{eval_filter.success_label}"
                            stats.skipped_reasons[reason] += 1
                            source_tasks.append(
                                {**task_record, "selected": False, "reason": reason}
                            )
                            continue

                    rows = build_singleturn_rows(
                        trajectory,
                        trajectory_path,
                        path_mode=path_mode,
                        stats=stats,
                    )
                    for row in rows:
                        output.write(_json_dump_line(row))
                    stats.selected_trajectories += 1
                    stats.output_rows += len(rows)
                    source_tasks.append({**task_record, "selected": True, "rows": len(rows)})
                except (OSError, json.JSONDecodeError, ConversionError) as error:
                    reason = f"conversion_error:{type(error).__name__}"
                    stats.skipped_reasons[reason] += 1
                    source_tasks.append(
                        {
                            "task_id": trajectory_path.parent.name,
                            "source_dir_name": trajectory_path.parent.name,
                            "selected": False,
                            "reason": reason,
                            "error": str(error),
                        }
                    )

        if stats.output_rows == 0:
            raise ConversionError("no training rows were produced")

        dataset_info = _dataset_info(dataset_name, data_file_name)
        (stage_dir / "dataset_info.json").write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage_dir / "source_tasks.json").write_text(
            json.dumps(source_tasks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        data_sha256 = _sha256_file(data_path)
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": str(Path(__file__).resolve()),
            "source_dir": str(source_dir),
            "dataset_name": dataset_name,
            "data_file": data_file_name,
            "data_sha256": data_sha256,
            "settings": {
                "turn_mode": "single",
                "history_context_mode": "last_obs",
                "keep_history_assistant_think": True,
                "keep_history_assistant_bash": True,
                "assistant_mask_mode_required": "last",
                "required_exit_status": required_exit_status,
                "path_mode": path_mode,
                "observation_stub": OBSERVATION_STUB,
                "literal_image_sentinel_escape": ESCAPED_IMAGE_SENTINEL,
                "raw_response_source": "trajectory.messages[].extra.raw_response",
            },
            "counts": {
                "source_trajectories": stats.source_trajectories,
                "selected_trajectories": stats.selected_trajectories,
                "output_rows": stats.output_rows,
                "historical_observations_stubbed": stats.historical_observations_stubbed,
                "adjacent_user_blocks_merged": stats.adjacent_user_blocks_merged,
                "empty_thought_targets": stats.empty_thought_targets,
                "exit_statuses": dict(sorted(stats.exit_statuses.items())),
                "skipped_reasons": dict(sorted(stats.skipped_reasons.items())),
                "secret_redactions": stats.sanitization.secret_redactions,
                "escaped_image_sentinels": stats.sanitization.escaped_image_sentinels,
                "workspace_path_replacements": stats.sanitization.workspace_path_replacements,
                "portable_path_replacements": stats.sanitization.portable_path_replacements,
            },
            "known_source_quality_notes": [
                "The collection's first browser-session create command failed with "
                "`python: command not found` in every source trajectory.",
                "Many later bash commands intentionally retain collection-host Python and "
                "repository paths unless path_mode=repo-portable is selected.",
            ],
        }
        if eval_filter is not None:
            manifest["settings"]["eval_success_filter"] = {
                "summary_path": str(eval_filter.summary_path),
                "summary_sha256": eval_filter.summary_sha256,
                "result_path": str(eval_filter.result_path),
                "result_sha256": eval_filter.result_sha256,
                "success_label": eval_filter.success_label,
                "labeled_tasks": len(eval_filter.labels_by_task),
                "successful_tasks": len(eval_filter.successful_task_ids),
                "summary_successful_tasks": eval_filter.summary_successful_tasks,
            }
        (stage_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mini-web-agent trajectories to SPB last-observation-only, "
            "single-turn ShareGPT rows for Qwen3.5."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument(
        "--required-exit-status",
        default="Submitted",
        help="only select this status; pass an empty string to include every status",
    )
    parser.add_argument(
        "--path-mode",
        choices=("none", "workspace", "repo-portable"),
        default="workspace",
        help=(
            "workspace (default) only maps per-task paths to /workspace while preserving "
            "bash; repo-portable also maps the collection repo/home to generic paths"
        ),
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=None,
        help=(
            "optional WebJudge eval_summary.json; only rows whose task has "
            "--success-label in the referenced result JSON/JSONL are selected"
        ),
    )
    parser.add_argument("--success-label", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    required_exit_status = args.required_exit_status.strip() or None
    eval_filter = (
        load_eval_success_filter(args.eval_summary, success_label=args.success_label)
        if args.eval_summary is not None
        else None
    )
    manifest = preprocess(
        args.source_dir,
        args.output_dir,
        dataset_name=args.dataset_name,
        required_exit_status=required_exit_status,
        path_mode=args.path_mode,
        eval_filter=eval_filter,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

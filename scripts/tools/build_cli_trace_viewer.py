#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import inspect
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import threading
import zipfile
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from cuawright.environments.osworld.source import activate_source

activate_source()


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = REPO_ROOT / "trace_viewer"
MAX_FILES = 20_000
MAX_FILE_BYTES = 24 * 1024 * 1024
CHUNK_TARGET_BYTES = 8 * 1024 * 1024
MAX_OBSERVATION_CHARS = 80_000

BENCHMARK_PROFILES = {
    "osworld-v2": {
        "label": "OSWorld V2",
        "metricLabel": "Partial reward",
        "metricKind": "reward",
    },
    "weavebench": {
        "label": "WeaveBench",
        "metricLabel": "Task score",
        "metricKind": "binary",
    },
    "benchcad": {
        "label": "BenchCAD",
        "metricLabel": "IoU",
        "metricKind": "overlap",
    },
    "cadgenbench": {
        "label": "CADGenBench",
        "metricLabel": "Official score",
        "metricKind": "aggregate",
    },
}


SECRET_PATTERNS = (
    re.compile(
        r"""(?ix)
        ("?authorization"?\s*[:=]\s*["']?(?:bearer\s+)?)
        ([^\s,"'}]+)
        """
    ),
    re.compile(
        r"""(?ix)
        (
          "?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|cookie)"?
          \s*[:=]\s*["']?
        )
        ([^\s,"'}]+)
        """
    ),
    re.compile(
        r"""(?ix)
        (\b(?:sudo\s+)?password\s+is\s+["']?)
        ([^\s,"'}]+)
        """
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
)
HOST_PATH_PATTERN = re.compile(
    r"/(?:data/iso-osworld-v2|home/v-theolee)(?:/[^\s\"'<>]*)?"
)
INTERNAL_ID_PATTERN = re.compile(r"\bcall_[A-Za-z0-9_-]{8,}\b")
MODEL_GATEWAY_PATTERN = re.compile(r"https?://gateway\.phyagi\.net(?:/[^\s\"'<>]*)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the self-contained OSWorld V2 CLI trace viewer ZIP"
    )
    parser.add_argument(
        "--selection-map",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--webp-quality", type=int, default=84)
    parser.add_argument(
        "--rubric-map",
        type=Path,
        help="Optional reconstructed evaluator rubric JSON",
    )
    parser.add_argument(
        "--rubric-run-label",
        default="",
        help="Run key inside reconstructed rubric records",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Concurrent task builders",
    )
    parser.add_argument("--keep-build-dir", action="store_true")
    return parser.parse_args()


def redact_text(text: str) -> str:
    value = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups >= 2:
            value = pattern.sub(r"\1[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED]", value)
    value = HOST_PATH_PATTERN.sub("[HOST_PATH]", value)
    value = INTERNAL_ID_PATTERN.sub("[CALL_ID]", value)
    value = MODEL_GATEWAY_PATTERN.sub("[MODEL_GATEWAY]", value)
    return value


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in {
                "api_key",
                "api-key",
                "authorization",
                "cookie",
                "cookies",
                "encrypted_content",
            }:
                result[key] = "[REDACTED]"
            else:
                result[key] = redact(item)
        return result
    return value


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prompt_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [prompt_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if not isinstance(content, dict):
        return ""
    if isinstance(content.get("text"), str):
        return str(content["text"])
    if "content" in content:
        return prompt_text(content["content"])
    return ""


def read_request_payload(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def request_step(path: Path) -> tuple[int, str]:
    match = re.search(r"step-(\d+)", path.name)
    return (int(match.group(1)) if match else sys.maxsize, path.name)


def load_model_visible_prompts(task_dir: Path) -> dict[str, Any]:
    requests_dir = task_dir / "requests"
    request_paths = sorted(
        (path for path in requests_dir.glob("step-*.json*") if path.is_file()),
        key=request_step,
    )
    if not request_paths:
        return {
            "available": False,
            "source": "unavailable",
            "system": "",
            "user": [],
            "note": "No retained first-request sidecar was available.",
        }

    payload = read_request_payload(request_paths[0])
    if not payload:
        return {
            "available": False,
            "source": "unavailable",
            "system": "",
            "user": [],
            "note": "The retained first-request sidecar could not be read.",
        }

    system_parts = []
    user_parts = []
    top_level_system = prompt_text(
        payload.get("system") or payload.get("instructions")
    ).strip()
    if top_level_system:
        system_parts.append(top_level_system)

    messages = payload.get("input")
    if not isinstance(messages, list):
        messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").lower()
            if role == "assistant":
                break
            text = prompt_text(message.get("content")).strip()
            if not text:
                continue
            if role == "system":
                system_parts.append(text)
            elif role == "user":
                user_parts.append(text)

    system = "\n\n".join(dict.fromkeys(system_parts))
    user = list(dict.fromkeys(user_parts))
    system = redact_text(system)
    user = [redact_text(text) for text in user]
    return {
        "available": bool(system or user),
        "source": "first recorded provider request",
        "system": system,
        "user": user,
        "note": (
            "Text-only extraction. Images, request metadata, credentials, "
            "encrypted reasoning, assistant messages, and tool payloads are omitted."
        ),
    }


def response_metadata(
    task_dir: Path,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    reasoning_by_step: dict[int, list[dict[str, Any]]] = {}
    calls_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted((task_dir / "responses").glob("step-*.json.gz")):
        match = re.fullmatch(r"step-(\d+)\.json\.gz", path.name)
        if not match:
            continue
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        summaries = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    text = summary.get("text") if isinstance(summary, dict) else None
                    if isinstance(text, str) and text.strip():
                        summaries.append(
                            {
                                "text": redact_text(text),
                                "provenance": "provider_response",
                            }
                        )
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                arguments = item.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                if isinstance(call_id, str):
                    calls_by_id[call_id] = {
                        "name": item.get("name") or "tool",
                        "arguments": redact(
                            arguments if isinstance(arguments, dict) else {}
                        ),
                    }
        if summaries:
            reasoning_by_step[int(match.group(1))] = summaries
    return reasoning_by_step, calls_by_id


def cap_observation(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OBSERVATION_CHARS:
        return text, False
    head = MAX_OBSERVATION_CHARS * 3 // 4
    tail = MAX_OBSERVATION_CHARS - head
    omitted = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n[... {omitted:,} recorded characters omitted from viewer ...]\n\n"
        + text[-tail:],
        True,
    )


def public_event_metadata(kind: str | None, row: dict[str, Any]) -> dict[str, Any]:
    if kind == "terminal_orchestration_completed":
        submission = row.get("submission") or {}
        verification = row.get("verification") or {}
        reports = verification.get("reports") or {}
        return redact(
            {
                "submission": {
                    "status": submission.get("status"),
                    "missing_specialists": submission.get("missing_subagents") or [],
                },
                "verification": {
                    "completed": verification.get("completed"),
                    "stop_reason": verification.get("stop_reason"),
                    "model_steps": verification.get("steps_used"),
                    "commands": verification.get("commands_executed"),
                    "specialists": {
                        role: value.get("status")
                        for role, value in reports.items()
                        if isinstance(value, dict)
                    },
                },
            }
        )
    if kind == "evaluation":
        evaluation = row.get("evaluation")
        if isinstance(evaluation, dict):
            evaluation = {
                field: evaluation[field]
                for field in ("score", "status", "iou", "error")
                if field in evaluation
            }
        return redact(
            {
                "benchmark": row.get("benchmark"),
                "score": row.get("score"),
                "evaluation": evaluation,
            }
        )
    common = ("status", "score", "completed", "stop_reason", "steps_used", "forced")
    fields = {
        "specialist_context_prepared": ("invocation_id", "phase_id"),
        "specialist_report_accepted": (
            "invocation_id",
            "status",
            "requested_steps",
            "allocated_steps",
            "forced",
        ),
        "specialist_completed": (
            "invocation_id",
            "status",
            "steps_used",
            "requested_steps",
            "allocated_steps",
            "forced",
        ),
        "submission_accepted": ("forced", "missing_reports", "non_ready_reasons"),
        "infrastructure_failure": ("status", "reason", "error"),
        "compaction": ("reason", "summary"),
    }.get(kind, common)
    return redact({field: row[field] for field in fields if field in row})


def public_evaluation(value: Any) -> Any:
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if not isinstance(value, dict):
        return None
    output = {
        field: redact(value[field])
        for field in ("score", "status", "iou", "error")
        if field in value
    }
    rows: list[dict[str, Any]] = []

    partial_scores = value.get("partial_scores")
    if isinstance(partial_scores, dict):
        for rubric_id, item in partial_scores.items():
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            weight = item.get("weight")
            contribution = (
                float(score) * float(weight)
                if isinstance(score, (int, float)) and isinstance(weight, (int, float))
                else None
            )
            rows.append(
                {
                    "id": str(rubric_id),
                    "kind": "criterion",
                    "label": str(rubric_id).replace("_", " ").strip().title(),
                    "score": score,
                    "weight": weight,
                    "contribution": contribution,
                    "status": rubric_status(score),
                    "description": redact_text(str(item.get("description") or "")),
                    "evidence": public_value_excerpt(item.get("evidence")),
                    "details": public_value_excerpt(item.get("details")),
                }
            )

    phase_results = value.get("phase_results")
    if isinstance(phase_results, list):
        for index, item in enumerate(phase_results, start=1):
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            rows.append(
                {
                    "id": f"phase-{item.get('phase_index', index)}",
                    "kind": "phase",
                    "label": str(
                        item.get("phase_name")
                        or f"Phase {item.get('phase_index', index)}"
                    ),
                    "score": score,
                    "weight": None,
                    "contribution": score,
                    "status": (
                        "credited"
                        if isinstance(score, (int, float)) and score > 0
                        else "not_satisfied"
                    ),
                    "description": redact_text(str(item.get("instruction") or "")),
                    "evidence": "",
                    "details": "",
                }
            )

    safety = value.get("safety")
    if isinstance(safety, dict):
        for index, item in enumerate(safety.get("checks") or [], start=1):
            if not isinstance(item, dict):
                continue
            result = str(item.get("result") or "")
            rows.append(
                {
                    "id": f"safety-{index}",
                    "kind": "safety",
                    "label": str(item.get("name") or f"Safety check {index}")
                    .replace("_", " ")
                    .title(),
                    "score": None,
                    "weight": None,
                    "contribution": None,
                    "status": (
                        "satisfied" if result.upper() == "PASS" else "not_satisfied"
                    ),
                    "description": redact_text(str(item.get("description") or "")),
                    "evidence": (
                        f"Result: {result or 'unknown'}"
                        + (
                            f"; penalty: {item.get('penalty')}"
                            if item.get("penalty") is not None
                            else ""
                        )
                    ),
                    "details": "",
                }
            )

    if rows:
        counts = Counter(row["status"] for row in rows)
        output["rubric"] = {
            "rows": rows,
            "satisfied": counts.get("satisfied", 0),
            "partial": counts.get("partial", 0),
            "notSatisfied": counts.get("not_satisfied", 0),
            "credited": counts.get("credited", 0),
        }
    if value.get("evaluation_error"):
        output["evaluationError"] = redact_text(str(value["evaluation_error"]))
    warnings = value.get("warnings")
    if isinstance(warnings, list) and warnings:
        output["warnings"] = [
            redact_text(str(warning))[:1000] for warning in warnings[:20]
        ]
    return output


def rubric_status(score: Any) -> str:
    if not isinstance(score, (int, float)):
        return "unknown"
    if score >= 1.0 - 1e-9:
        return "satisfied"
    if score > 0:
        return "partial"
    return "not_satisfied"


def public_value_excerpt(value: Any, limit: int = 4000) -> str:
    if value in (None, "", {}, []):
        return ""
    if isinstance(value, str):
        text = redact_text(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        text = "\n".join(redact_text(item) for item in value)
    else:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… {len(text) - limit:,} additional characters omitted"


def merge_reconstructed_rubric(
    evaluation: Any,
    record: dict[str, Any] | None,
    run_label: str,
) -> Any:
    if not isinstance(record, dict):
        return evaluation
    definitions = record.get("criteria")
    run = (record.get("runs") or {}).get(run_label)
    if not isinstance(definitions, list) or not isinstance(run, dict):
        return evaluation
    outcomes = {
        str(item.get("id")): item
        for item in run.get("criteria") or []
        if isinstance(item, dict) and item.get("id")
    }
    rows = []
    for definition in definitions:
        if not isinstance(definition, dict) or not definition.get("id"):
            continue
        rubric_id = str(definition["id"])
        outcome = outcomes.get(rubric_id, {})
        rows.append(
            {
                "id": rubric_id,
                "kind": "criterion",
                "label": str(definition.get("label") or rubric_id),
                "score": outcome.get("score"),
                "weight": definition.get("weight"),
                "contribution": outcome.get("contribution"),
                "status": str(outcome.get("status") or "unknown"),
                "description": str(definition.get("description") or ""),
                "evidence": public_value_excerpt(outcome.get("evidence")),
                "details": "",
            }
        )
    if not rows:
        return evaluation
    if isinstance(run.get("overall"), (int, float)) and not any(
        isinstance(row.get("score"), (int, float)) for row in rows
    ):
        rows.insert(
            0,
            {
                "id": "recorded-overall-score",
                "kind": "aggregate",
                "label": "Recorded overall evaluator score",
                "score": run["overall"],
                "weight": None,
                "contribution": run["overall"],
                "status": "recorded",
                "description": (
                    "The selected run retained this aggregate score, but not "
                    "its allocation across the criteria below."
                ),
                "evidence": "",
                "details": "",
            },
        )

    existing = evaluation if isinstance(evaluation, dict) else {}
    safety_rows = [
        row
        for row in ((existing.get("rubric") or {}).get("rows") or [])
        if row.get("kind") == "safety"
    ]
    rows.extend(safety_rows)
    counts = Counter(row["status"] for row in rows)
    output = dict(existing)
    if "score" not in output:
        output["score"] = run.get("overall")
    output["rubric"] = {
        "rows": rows,
        "satisfied": counts.get("satisfied", 0),
        "partial": counts.get("partial", 0),
        "notSatisfied": counts.get("not_satisfied", 0),
        "credited": counts.get("credited", 0),
        "recorded": counts.get("recorded", 0),
        "unknown": counts.get("unknown", 0),
        "sourceSummary": str(record.get("sourceSummary") or ""),
        "aggregation": str(record.get("aggregation") or ""),
        "outcomeNote": str(run.get("outcomeNote") or ""),
        "reconstructed": True,
    }
    return output


def load_specialist_artifacts(task_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for index, path in enumerate(
        sorted((task_dir / "specialists").glob("**/report.json")), start=1
    ):
        request_path = path.with_name("request.txt")
        artifacts.append(
            {
                "artifactKey": f"specialist-{index:03d}",
                "invocationId": path.parent.name,
                "role": path.parent.name.split("-", 1)[0].title(),
                "request": (
                    redact_text(
                        request_path.read_text(encoding="utf-8", errors="replace")
                    )
                    if request_path.is_file()
                    else ""
                ),
                "report": redact(read_json(path, {})),
            }
        )
    return artifacts


def read_selection_map(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"task", "score", "source"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("Selection map requires task, score, source columns")
        rows = list(reader)
    if not rows:
        raise RuntimeError("Selection map is empty")
    task_ids = [row["task"].strip() for row in rows]
    if any(not task_id for task_id in task_ids):
        raise RuntimeError("Selection map contains an empty task ID")
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Selection map contains duplicate task IDs")
    return rows


def iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_label(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"+{hours}h {minutes:02d}m"
    if minutes:
        return f"+{minutes}m {secs:02d}s"
    return f"+{secs}s"


def role_name(raw: str | None) -> str:
    value = (raw or "").lower()
    if "binder" in value:
        return "Binder"
    if "critic" in value:
        return "Critic"
    if "verifier" in value:
        return "Verifier"
    if "actor" in value:
        return "Actor"
    return "System"


def event_role(raw_role: str | None, stage: str | None) -> str:
    stage_value = (stage or "").lower()
    for token, label in (
        ("verifier", "Verifier"),
        ("critic", "Critic"),
        ("binder", "Binder"),
        ("actor", "Actor"),
    ):
        if token in stage_value:
            return label
    return role_name(raw_role)


def image_evidence_kind(source_path: str) -> str:
    lowered = source_path.lower()
    if any(token in lowered for token in ("final", "result", "render", "artifact")):
        return "result"
    if any(token in lowered for token in ("reference", "input", "source", "target")):
        return "reference"
    if any(token in lowered for token in ("eval", "score", "verifier")):
        return "evaluator"
    return "observation"


def public_image_caption(source_path: str | None) -> str:
    if not source_path:
        return "Image supplied to the model"
    name = Path(source_path).name
    return redact_text(name or "Image supplied to the model")


def infer_benchmark(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
    if "cadgenbench" in normalized:
        return "cadgenbench"
    if "benchcad" in normalized:
        return "benchcad"
    if "weavebench" in normalized:
        return "weavebench"
    if "osworld" in normalized:
        return "osworld-v2"
    return "generic"


def collection_profile(
    rows: list[dict[str, str]],
    title: str,
) -> dict[str, str]:
    first = rows[0] if rows else {}
    benchmark = first.get("benchmark", "").strip() or infer_benchmark(title)
    profile = BENCHMARK_PROFILES.get(
        benchmark,
        {
            "label": benchmark if benchmark != "generic" else "Evaluation",
            "metricLabel": "Score",
            "metricKind": "score",
        },
    )
    return {
        "benchmark": benchmark,
        "benchmarkLabel": first.get("benchmark_label", "").strip() or profile["label"],
        "modelLabel": first.get("model_label", "").strip(),
        "metricLabel": first.get("metric_label", "").strip() or profile["metricLabel"],
        "metricKind": first.get("metric_kind", "").strip() or profile["metricKind"],
        "scoreProvenance": first.get("score_provenance", "").strip(),
    }


def public_run_metadata(task_dir: Path) -> dict[str, Any]:
    manifest = read_json(task_dir / "manifest.json", {})
    if not isinstance(manifest, dict):
        return {}
    return {
        "benchmark": manifest.get("benchmark"),
        "modelLabel": manifest.get("model"),
        "reasoningEffort": manifest.get("reasoning_effort"),
        "release": manifest.get("release"),
        "modelInterface": manifest.get("model_interface"),
    }


def parse_list_field(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def infer_related_software(task: Any) -> list[str]:
    instruction = str(task.get("instruction") or "")
    try:
        source = inspect.getsource(task.__class__)
    except (OSError, TypeError):
        source = ""
    haystack = f"{instruction}\n{source}".lower()
    labels: list[str] = []

    def add(label: str, *needles: str) -> None:
        if (
            any(needle.lower() in haystack for needle in needles)
            and label not in labels
        ):
            labels.append(label)

    add("MailHub", "mailhub")
    add("VaultBank", "vaultbank")
    add("ExpenseFlow", "expenseflow")
    add("TravelHub", "travelhub")
    add("LibreOffice Calc", "libreoffice', '--calc", 'libreoffice", "--calc', ".xlsx")
    add("LibreOffice Writer", "libreoffice', '--writer", ".docx")
    add("Document Viewer", ".pdf", "pdf forms")
    add("Windows game", "wine_game", "launch_wine_game")
    add("Chrome", "chrome", "localhost:", "build_website_url")
    add(
        "Terminal",
        "conda environment",
        "hexo",
        "~/robox",
        "~/libero",
        "keyboard teleoperation",
    )
    return labels


def choose_thumbnail(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    images = [
        image
        for event in events
        for image in event.get("images") or []
        if isinstance(image, dict) and image.get("path")
    ]
    if not images:
        return None
    for kind in ("result", "evaluator", "reference"):
        matching = [image for image in images if image.get("evidenceKind") == kind]
        if matching:
            image = matching[-1]
            break
    else:
        image = images[-1]
    return {
        field: image.get(field)
        for field in ("path", "width", "height", "caption", "evidenceKind")
        if image.get(field) is not None
    }


def event_title(row: dict[str, Any], tool_calls: list[dict[str, Any]]) -> str:
    if row.get("event") == "model_step":
        if tool_calls:
            names = ", ".join(call.get("name", "tool") for call in tool_calls)
            return f"Called {names}"
        if row.get("output_text"):
            return "Responded without a tool"
        return "Model step"
    names = {
        "subagent_report": "Subagent returned a report",
        "phase_started": "Phase started",
        "phase_evaluated": "Phase evaluated",
        "compaction": "Context compacted",
        "orchestrator_reminder": "Budget reminder",
        "actor_reminder": "Actor reminder",
        "task_submitted": "Task submitted",
        "actor_repair_submitted": "Repair submitted",
        "evaluation": "Evaluator completed",
        "user_question": "Agent asked the user",
        "response_retry": "Model request retried",
        "orchestration_completed": "Orchestration completed",
        "verification_completed": "Verification completed",
        "verification_submitted": "Verifier submitted",
        "infrastructure_failure": "Infrastructure failure",
    }
    return names.get(
        row.get("event"), row.get("event", "Event").replace("_", " ").title()
    )


def important_model_event(
    role: str,
    reasoning: list[str],
    tool_calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    images: list[dict[str, Any]],
) -> bool:
    if reasoning or images:
        return True
    names = {call.get("name") for call in tool_calls}
    if names & {
        "call_binder",
        "call_critic",
        "call_verifier",
        "submit_task",
        "read_image",
    }:
        return True
    if role == "Actor" and "run_command" in names:
        return True
    return any(item.get("status") == "error" for item in observations)


class ImageStore:
    def __init__(self, build_dir: Path, quality: int) -> None:
        self.build_dir = build_dir
        self.quality = max(40, min(95, quality))
        self.by_hash: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def add(self, source: Path) -> dict[str, Any] | None:
        if not source.exists():
            return None
        raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        with self._lock:
            cached = self.by_hash.get(raw_hash)
            if cached is not None:
                return dict(cached)
            pending = self._pending.get(raw_hash)
            if pending is None:
                pending = threading.Event()
                self._pending[raw_hash] = pending
                owner = True
            else:
                owner = False
        if not owner:
            pending.wait()
            with self._lock:
                cached = self.by_hash.get(raw_hash)
                return dict(cached) if cached is not None else None

        output_dir = self.build_dir / "media"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{raw_hash[:24]}.webp"
        temporary = output_dir / f".{raw_hash}.{threading.get_ident()}.webp"
        try:
            with Image.open(source) as image:
                image.load()
                original_width, original_height = image.size
                mode = "RGBA" if "A" in image.getbands() else "RGB"
                converted = image.convert(mode)
                if max(converted.size) > 12_000:
                    ratio = 12_000 / max(converted.size)
                    converted = converted.resize(
                        (
                            max(1, round(converted.width * ratio)),
                            max(1, round(converted.height * ratio)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                width, height = converted.size
                quality = self.quality
                while True:
                    converted.save(temporary, "WEBP", quality=quality, method=6)
                    if temporary.stat().st_size < MAX_FILE_BYTES or quality <= 45:
                        break
                    quality -= 10
                if temporary.stat().st_size >= MAX_FILE_BYTES:
                    ratio = (MAX_FILE_BYTES / temporary.stat().st_size) ** 0.5
                    resized = converted.resize(
                        (
                            max(1, int(width * ratio * 0.9)),
                            max(1, int(height * ratio * 0.9)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    resized.save(temporary, "WEBP", quality=70, method=6)
            os.replace(temporary, output_path)
            relative = output_path.relative_to(self.build_dir).as_posix()
            item = {
                "path": relative,
                "width": width,
                "height": height,
                "originalWidth": original_width,
                "originalHeight": original_height,
                "sha256": raw_hash,
                "caption": "Explicit image supplied to the model",
                "evidenceKind": "observation",
            }
            with self._lock:
                self.by_hash[raw_hash] = item
            return dict(item)
        finally:
            temporary.unlink(missing_ok=True)
            with self._lock:
                self._pending.pop(raw_hash, None)
                pending.set()


def normalize_events(
    task_dir: Path,
    image_store: ImageStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(task_dir / "trace.jsonl")
    if not rows:
        rows = read_jsonl(task_dir / "trace-slim.jsonl")
    reasoning_by_step, calls_by_id = response_metadata(task_dir)
    timestamps = [iso_timestamp(row.get("timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    started = min(timestamps) if timestamps else None
    ended = max(timestamps) if timestamps else None
    events: list[dict[str, Any]] = []
    model_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    model_by_call: dict[str, dict[str, Any]] = {}
    pending_images: defaultdict[str, deque[dict[str, Any]]] = defaultdict(deque)
    latest_budget: dict[str, Any] = {}

    for row_index, row in enumerate(rows):
        kind = row.get("event")
        timestamp = iso_timestamp(row.get("timestamp"))
        elapsed = (timestamp - started).total_seconds() if timestamp and started else 0
        if kind in {"model_step", "model_turn"}:
            raw_role = row.get("role")
            role = event_role(raw_role, row.get("stage"))
            tool_calls = []
            for raw_call in row.get("tool_calls") or []:
                if not isinstance(raw_call, dict):
                    continue
                call_id = raw_call.get("call_id")
                enriched = calls_by_id.get(call_id, {})
                call = {
                    "call_id": call_id,
                    "name": enriched.get("name") or raw_call.get("name") or "tool",
                    "arguments": redact(
                        enriched.get("arguments") or raw_call.get("arguments") or {}
                    ),
                }
                tool_calls.append(call)
            global_step = row.get("global_step") or row.get("step")
            billed = max(1, int(row.get("billed_responses") or 1))
            reasoning = []
            if isinstance(global_step, int):
                for step in range(max(1, global_step - billed + 1), global_step + 1):
                    reasoning.extend(reasoning_by_step.get(step, ()))
            if not reasoning:
                reasoning = [
                    {
                        "text": value,
                        "provenance": "trace_fallback",
                    }
                    for value in redact(row.get("reasoning_summaries") or [])
                ]
            event = {
                "id": f"e{row_index:05d}",
                "timestamp": timestamp.isoformat() if timestamp else None,
                "elapsedSeconds": elapsed,
                "elapsedLabel": elapsed_label(elapsed),
                "kind": "model",
                "role": role,
                "step": global_step,
                "stage": row.get("stage"),
                "title": event_title(row, tool_calls),
                "reasoning": reasoning,
                "toolCalls": tool_calls,
                "text": redact_text(row.get("output_text") or row.get("text") or ""),
                "observations": [],
                "images": [],
                "attachmentWarnings": [],
                "subagentInternal": role in {"Binder", "Critic", "Verifier"},
                "error": False,
            }
            events.append(event)
            model_by_key[(raw_role, row.get("stage"), row.get("step"))] = event
            for call in tool_calls:
                call_id = call.get("call_id")
                if isinstance(call_id, str):
                    model_by_call[call_id] = event
            continue

        if kind == "terminal_command":
            continue

        if kind == "image_attached":
            source_path = row.get("source_path")
            if isinstance(source_path, str):
                pending_images[source_path].append(row)
            continue

        if kind == "command_result":
            call_id = row.get("call_id")
            parent = model_by_call.get(call_id)
            if parent is not None:
                stdout, stdout_truncated = cap_observation(
                    redact_text(str(row.get("stdout") or ""))
                )
                stderr, stderr_truncated = cap_observation(
                    redact_text(str(row.get("stderr") or ""))
                )
                parent["observations"].append(
                    {
                        "tool": "run_command",
                        "status": "success" if row.get("exit_code") == 0 else "error",
                        "exitCode": row.get("exit_code"),
                        "visibility": "host_recorded",
                        "output": stdout,
                        "errorOutput": stderr,
                        "outputChars": len(str(row.get("stdout") or "")),
                        "errorChars": len(str(row.get("stderr") or "")),
                        "truncated": stdout_truncated or stderr_truncated,
                    }
                )
                if row.get("exit_code") != 0:
                    parent["error"] = True
                for attachment in row.get("attachments") or []:
                    if (
                        not isinstance(attachment, dict)
                        or attachment.get("kind") != "image"
                    ):
                        continue
                    source_path = attachment.get("source_path")
                    images_for_source = pending_images.get(source_path)
                    image_event = (
                        images_for_source.popleft() if images_for_source else {}
                    )
                    digest = image_event.get("sha256")
                    candidates = (
                        list((task_dir / "images").glob(f"{digest}.*"))
                        if digest
                        else []
                    )
                    try:
                        image = image_store.add(candidates[0]) if candidates else None
                    except (OSError, ValueError) as exc:
                        parent["attachmentWarnings"].append(
                            {
                                "sourcePath": source_path,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        image = None
                    if image:
                        image["caption"] = public_image_caption(source_path)
                        image["evidenceKind"] = image_evidence_kind(source_path or "")
                        parent["images"].append(image)
                continue

        if kind == "tool_result":
            key = (row.get("role"), row.get("stage"), row.get("step"))
            parent = model_by_key.get(key)
            if parent is None:
                parent = {
                    "id": f"e{row_index:05d}",
                    "timestamp": timestamp.isoformat() if timestamp else None,
                    "elapsedSeconds": elapsed,
                    "elapsedLabel": elapsed_label(elapsed),
                    "kind": "model",
                    "role": event_role(row.get("role"), row.get("stage")),
                    "step": row.get("step"),
                    "stage": row.get("stage"),
                    "title": f"Tool result: {row.get('tool', 'tool')}",
                    "reasoning": [],
                    "toolCalls": [],
                    "text": "",
                    "observations": [],
                    "images": [],
                    "attachmentWarnings": [],
                    "subagentInternal": event_role(row.get("role"), row.get("stage"))
                    in {"Binder", "Critic", "Verifier"},
                    "error": False,
                }
                events.append(parent)
            raw_output = redact_text(str(row.get("output") or ""))
            output, viewer_truncated = cap_observation(raw_output)
            status = (
                "error"
                if '"status": "error"' in output or row.get("status") == "error"
                else "success"
            )
            observation = {
                "tool": row.get("tool"),
                "status": status,
                "output": output,
                "visibility": "visibility_unknown",
                "outputChars": row.get("output_chars") or len(raw_output),
                "truncated": bool(row.get("truncated")) or viewer_truncated,
            }
            parent["observations"].append(observation)
            if status == "error":
                parent["error"] = True
            image_blob = row.get("image_blob")
            if image_blob:
                image = image_store.add(task_dir / image_blob)
                if image:
                    image["caption"] = public_image_caption(image_blob)
                    image["evidenceKind"] = image_evidence_kind(str(image_blob))
                    parent["images"].append(image)
            continue

        if kind in {
            "shared_budget_step",
            "bindings_drafted",
            "bindings_submitted",
            "planning_completed",
            "control_client_executed",
            "control_client_dispatched",
            "image_attached",
            "provider_response_charge",
            "response_retry",
            "orchestrator_reminder",
            "actor_reminder",
        }:
            if kind == "shared_budget_step" and isinstance(row.get("budget"), dict):
                latest_budget = redact(row["budget"])
            continue

        role = event_role(row.get("role"), row.get("stage"))
        report = None
        if kind == "subagent_report":
            report = redact(row.get("report"))
        text_fields = []
        for field in (
            "summary",
            "reason",
            "question",
            "answer",
            "instruction",
            "phase_name",
        ):
            if row.get(field):
                text_fields.append(f"{field.replace('_', ' ').title()}: {row[field]}")
        error = kind == "infrastructure_failure" or bool(row.get("error"))
        events.append(
            {
                "id": f"e{row_index:05d}",
                "timestamp": timestamp.isoformat() if timestamp else None,
                "elapsedSeconds": elapsed,
                "elapsedLabel": elapsed_label(elapsed),
                "kind": kind or "event",
                "role": role,
                "step": row.get("global_step") or row.get("step"),
                "stage": row.get("stage"),
                "title": event_title(row, []),
                "reasoning": [],
                "toolCalls": [],
                "text": redact_text("\n".join(text_fields)),
                "observations": [],
                "images": [],
                "report": report,
                "subagentInternal": False,
                "error": error,
                "important": True,
                "raw": public_event_metadata(kind, row),
            }
        )

    for event in events:
        if "important" not in event:
            event["important"] = important_model_event(
                event["role"],
                event.get("reasoning") or [],
                event.get("toolCalls") or [],
                event.get("observations") or [],
                event.get("images") or [],
            )
        for call in event.get("toolCalls") or []:
            call.pop("call_id", None)
        event["searchText"] = " ".join(
            [
                str(event.get("title") or ""),
                str(event.get("text") or ""),
                json.dumps(event.get("reasoning") or [], ensure_ascii=False),
                json.dumps(event.get("toolCalls") or [], ensure_ascii=False),
                json.dumps(event.get("observations") or [], ensure_ascii=False),
                json.dumps(event.get("report") or {}, ensure_ascii=False),
            ]
        ).lower()

    events.sort(key=lambda item: (item.get("timestamp") or "", item["id"]))
    return events, {
        "startedAt": started.isoformat() if started else None,
        "endedAt": ended.isoformat() if ended else None,
        "durationSeconds": (
            (ended - started).total_seconds() if started and ended else None
        ),
        "budget": latest_budget,
    }


def load_task_details(task_id: str) -> tuple[str, str, list[str]]:
    os.environ.setdefault("WEBSITE_HOST_SUFFIX", "10.217.29.172.nip.io")
    examples_dir = locate_osworld_examples()
    source_root = examples_dir.parent
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from task_loader import load_task_config

    task = load_task_config(
        None,
        task_id=task_id,
        base_dir=str(examples_dir),
        domain="tasks",
        eval_version="v2",
    )
    phase_getter = getattr(task, "get_phases", None)
    phases = phase_getter() if callable(phase_getter) else []
    if phases:
        instruction = "\n\n".join(
            f"{phase.get('name', f'Phase {index}')}:\n{phase['instruction']}"
            for index, phase in enumerate(phases, start=1)
        )
        return (
            f"Sequential multi-phase task ({len(phases)} phases)",
            instruction,
            parse_list_field(task.get("related_apps")) or infer_related_software(task),
        )
    instruction = str(task["instruction"])
    title = instruction.splitlines()[0][:100] if instruction else f"Task {task_id}"
    return (
        title,
        instruction,
        parse_list_field(task.get("related_apps")) or infer_related_software(task),
    )


def locate_osworld_examples() -> Path:
    candidates = []
    configured = os.environ.get("CUAWRIGHT_OSWORLD_SOURCE", "").strip()
    if configured:
        source = Path(configured).expanduser()
        candidates.append(
            source
            if source.name == "evaluation_examples"
            else source / "evaluation_examples"
        )
    candidates.extend(
        [
            REPO_ROOT / "evaluation_examples",
            Path("/data/iso-osworld-v2/OSWorld-V2/evaluation_examples"),
        ]
    )
    install_root = Path.home() / ".local" / "share" / "cuawright" / "osworld"
    candidates.extend(
        path / "source" / "evaluation_examples"
        for path in sorted(install_root.glob("*"), reverse=True)
    )
    for candidate in candidates:
        if (candidate / "task_class").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "OSWorld evaluation_examples not found; set CUAWRIGHT_OSWORLD_SOURCE"
    )


def load_instruction(task_id: str) -> tuple[str, str]:
    title, instruction, _ = load_task_details(task_id)
    return title, instruction


def resolve_task_dir(
    results_root: Path,
    source: str,
    task_id: str,
) -> Path | None:
    if source == "UNEVALUATED_ZERO":
        return None
    source_path = Path(source)
    if (source_path / "trace.jsonl").is_file() or (
        source_path / "trace-slim.jsonl"
    ).is_file():
        return source_path
    run_root = source_path if source_path.is_absolute() else results_root / source
    candidates = sorted(run_root.glob(f"task-{task_id}-*"))
    return candidates[-1] if candidates else None


def write_js(
    path: Path,
    prefix: str,
    value: Any,
    suffix: str = ";",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    path.write_text(f"{prefix}{payload}{suffix}\n", encoding="utf-8")


def chunk_events(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 2
    for event in events:
        encoded_size = len(
            json.dumps(event, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        if current and current_size + encoded_size > CHUNK_TARGET_BYTES:
            chunks.append(current)
            current = []
            current_size = 2
        current.append(event)
        current_size += encoded_size + 1
    if current or not chunks:
        chunks.append(current)
    return chunks


def instruction_snippet(instruction: str, limit: int = 180) -> str:
    value = " ".join(instruction.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def is_strict_pass(score: float) -> bool:
    return abs(score - 1.0) < 1e-9


def result_summary(
    events: list[dict[str, Any]],
    orchestration: dict[str, Any],
    result_record: dict[str, Any],
    evaluation: Any,
    evaluation_status: str,
) -> dict[str, Any]:
    submission_status = ""
    verification_status = ""
    stop_reason = orchestration.get("stop_reason") or ""
    specialists = {}
    reports = orchestration.get("reports") or {}
    if isinstance(reports, dict):
        specialists = {
            str(role).title(): value.get("status")
            for role, value in reports.items()
            if isinstance(value, dict) and value.get("status")
        }
    if orchestration.get("completed"):
        verification_status = "complete"

    submission = result_record.get("submission")
    if isinstance(submission, dict):
        submission_status = str(submission.get("status") or "")
    verification = result_record.get("verification")
    if isinstance(verification, dict):
        if (
            verification.get("completed")
            or verification.get("final")
            or verification.get("audit")
        ):
            verification_status = "complete"
        stop_reason = stop_reason or str(verification.get("stop_reason") or "")
        result_reports = verification.get("reports")
        if isinstance(result_reports, dict) and not specialists:
            specialists = {
                str(role).title(): value.get("status")
                for role, value in result_reports.items()
                if isinstance(value, dict) and value.get("status")
            }

    for event in reversed(events):
        raw = event.get("raw") or {}
        if not isinstance(raw, dict):
            continue
        if not submission_status:
            submission = raw.get("submission")
            if isinstance(submission, dict):
                submission_status = str(submission.get("status") or "")
            elif event.get("kind") == "submission_accepted":
                submission_status = "accepted"
            elif event.get("kind") in {"task_submitted", "actor_repair_submitted"}:
                submission_status = "submitted"
        if event.get("kind") == "terminal_orchestration_completed":
            verification = raw.get("verification")
            if isinstance(verification, dict):
                if verification.get("completed"):
                    verification_status = "complete"
                verification_status = verification_status or str(
                    verification.get("status") or ""
                )
                stop_reason = stop_reason or str(verification.get("stop_reason") or "")
                raw_specialists = verification.get("specialists")
                if isinstance(raw_specialists, dict) and not specialists:
                    specialists = {
                        str(role).title(): status
                        for role, status in raw_specialists.items()
                        if status
                    }

    evaluator_status = evaluation_status
    if isinstance(evaluation, dict) and evaluation.get("status"):
        evaluator_status = str(evaluation["status"])
    return {
        "submissionStatus": submission_status or "not recorded",
        "verificationStatus": verification_status or "not recorded",
        "stopReason": stop_reason,
        "evaluatorStatus": evaluator_status,
        "specialists": specialists,
    }


def build_task(
    row: dict[str, str],
    build_dir: Path,
    results_root: Path,
    image_store: ImageStore,
    rubric_record: dict[str, Any] | None = None,
    rubric_run_label: str = "",
) -> tuple[dict[str, Any], list[str]]:
    task_id = row["task"]
    raw_score = row["score"].strip()
    score = float(raw_score) if raw_score else None
    source = row["source"]
    instruction = row.get("instruction", "").strip()
    title = row.get("title", "").strip()
    loaded_related_software: list[str] = []
    if not instruction:
        loaded_title, instruction, loaded_related_software = load_task_details(task_id)
        title = title or loaded_title
    elif not title:
        title = instruction.splitlines()[0][:100] if instruction else task_id
    task_dir = resolve_task_dir(results_root, source, task_id)
    events: list[dict[str, Any]] = []
    timing = {
        "startedAt": None,
        "endedAt": None,
        "durationSeconds": None,
    }
    orchestration = {}
    result_record: dict[str, Any] = {}
    evaluation = None
    subagents = []
    image_count = 0
    status = "unevaluated" if task_dir is None else "evaluated"
    source_paths = []
    run_metadata: dict[str, Any] = {}
    prompts = {
        "available": False,
        "source": "unavailable",
        "system": "",
        "user": [],
        "note": "No retained first-request sidecar was available.",
    }

    if task_dir is not None:
        events, timing = normalize_events(task_dir, image_store)
        image_count = sum(len(event.get("images") or []) for event in events)
        orchestration = redact(read_json(task_dir / "orchestration.json", {}))
        result_record = read_json(task_dir / "evaluation" / "result.json", {})
        evaluation = public_evaluation(result_record.get("evaluation"))
        for path in sorted((task_dir / "subagents").glob("*.json")):
            subagents.append(redact(read_json(path, {})))
        subagents.extend(load_specialist_artifacts(task_dir))
        source_paths.append(str(task_dir))
        run_metadata = public_run_metadata(task_dir)
        prompts = load_model_visible_prompts(task_dir)
    evaluation = merge_reconstructed_rubric(
        evaluation,
        rubric_record,
        rubric_run_label,
    )

    benchmark = (
        row.get("benchmark", "").strip()
        or str(run_metadata.get("benchmark") or "")
        or "generic"
    )
    metric_label = row.get("metric_label", "").strip() or "Score"
    metric_kind = row.get("metric_kind", "").strip() or "score"
    evaluation_status = row.get("evaluation_status", "").strip()
    if not evaluation_status:
        if score is not None:
            evaluation_status = "evaluated"
        elif isinstance(evaluation, dict) and evaluation.get("status"):
            evaluation_status = str(evaluation["status"])
        elif task_dir is not None:
            evaluation_status = "trace available"
        else:
            evaluation_status = "unavailable"
    trace_status = "available" if task_dir is not None else "missing"
    related_software = (
        parse_list_field(row.get("related_software")) or loaded_related_software
    )
    thumbnail = choose_thumbnail(events)

    budget = orchestration.get("budget") or timing.get("budget") or {}
    calls_used = budget.get("used_steps")
    role_usage = budget.get("by_role") or {}
    model_events = [event for event in events if event.get("kind") == "model"]
    if calls_used is None:
        calls_used = len(model_events)
    if not role_usage:
        role_usage = dict(
            Counter(
                str(event.get("role") or "System").lower() for event in model_events
            )
        )
    task_data_dir = build_dir / "data" / "tasks" / task_id
    chunks = chunk_events(events)
    chunk_paths = []
    for index, chunk in enumerate(chunks):
        relative = f"data/tasks/{task_id}/chunk-{index:03d}.js"
        write_js(
            build_dir / relative,
            "window.TRACE_TASK_CHUNKS.push(",
            chunk,
            ");",
        )
        chunk_paths.append(relative)

    meta = {
        "id": task_id,
        "title": title,
        "instruction": instruction,
        "status": status,
        "traceStatus": trace_status,
        "evaluationStatus": evaluation_status,
        "score": score,
        "strictPass": score is not None and is_strict_pass(score),
        "benchmark": benchmark,
        "benchmarkLabel": row.get("benchmark_label", "").strip() or benchmark,
        "modelLabel": row.get("model_label", "").strip()
        or str(run_metadata.get("modelLabel") or ""),
        "reasoningEffort": row.get("reasoning_effort", "").strip()
        or str(run_metadata.get("reasoningEffort") or ""),
        "release": str(run_metadata.get("release") or ""),
        "modelInterface": str(run_metadata.get("modelInterface") or ""),
        "metricLabel": metric_label,
        "metricKind": metric_kind,
        "scoreProvenance": row.get("score_provenance", "").strip(),
        "category": row.get("category", "").strip(),
        "taskType": row.get("task_type", "").strip(),
        "relatedSoftware": related_software,
        "thumbnail": thumbnail,
        "callsUsed": calls_used,
        "roleUsage": role_usage,
        "durationSeconds": timing["durationSeconds"],
        "imageCount": image_count,
        "eventCount": len(events),
        "chunks": chunk_paths,
        "subagents": subagents,
        "evaluation": evaluation,
        "prompts": prompts,
        "resultSummary": result_summary(
            events,
            orchestration if isinstance(orchestration, dict) else {},
            result_record if isinstance(result_record, dict) else {},
            evaluation,
            evaluation_status,
        ),
        "omitted": [
            "raw API requests and responses beyond sanitized launch prompt text",
            "encrypted reasoning",
            "credentials and cookies",
            "host paths and environment configuration",
            "provider retries and internal identifiers",
        ],
    }
    meta_path = f"data/tasks/{task_id}/meta.js"
    write_js(build_dir / meta_path, "window.TRACE_TASK_META=", meta)
    catalog_entry = {
        "id": task_id,
        "title": title,
        "instruction": instruction,
        "instructionSnippet": instruction_snippet(instruction),
        "status": status,
        "traceStatus": trace_status,
        "evaluationStatus": evaluation_status,
        "score": score,
        "strictPass": score is not None and is_strict_pass(score),
        "benchmark": benchmark,
        "benchmarkLabel": row.get("benchmark_label", "").strip() or benchmark,
        "modelLabel": row.get("model_label", "").strip()
        or str(run_metadata.get("modelLabel") or ""),
        "reasoningEffort": row.get("reasoning_effort", "").strip()
        or str(run_metadata.get("reasoningEffort") or ""),
        "metricLabel": metric_label,
        "metricKind": metric_kind,
        "scoreProvenance": row.get("score_provenance", "").strip(),
        "category": row.get("category", "").strip(),
        "taskType": row.get("task_type", "").strip(),
        "relatedSoftware": related_software,
        "thumbnail": thumbnail,
        "callsUsed": calls_used,
        "durationSeconds": timing["durationSeconds"],
        "imageCount": image_count,
        "eventCount": len(events),
        "resultSummary": meta["resultSummary"],
        "dataPath": meta_path,
    }
    return catalog_entry, source_paths


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_site(build_dir: Path) -> dict[str, Any]:
    files = [path for path in build_dir.rglob("*") if path.is_file()]
    if len(files) >= MAX_FILES:
        raise RuntimeError(
            f"Extracted site has {len(files)} files; limit is below {MAX_FILES}"
        )
    oversized = [
        (path, path.stat().st_size)
        for path in files
        if path.stat().st_size >= MAX_FILE_BYTES
    ]
    if oversized:
        details = ", ".join(
            f"{path.relative_to(build_dir)}={size}" for path, size in oversized
        )
        raise RuntimeError(f"Files exceed the 24 MiB cap: {details}")
    largest = max(files, key=lambda path: path.stat().st_size)
    return {
        "fileCount": len(files),
        "uncompressedBytes": sum(path.stat().st_size for path in files),
        "largestFile": largest.relative_to(build_dir).as_posix(),
        "largestFileBytes": largest.stat().st_size,
    }


def write_package_manifest(
    build_dir: Path,
    catalog: dict[str, Any],
    source_paths: Iterable[str],
) -> None:
    files = []
    for path in sorted(build_dir.rglob("*")):
        if not path.is_file() or path.name == "package-manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(build_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    payload = {
        "format": "osworld-v2-cli-trace-viewer-v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "taskCount": len(catalog["tasks"]),
        "sourceTaskCount": len(set(source_paths)),
        "limits": {
            "maximumFilesExclusive": MAX_FILES,
            "maximumFileBytesExclusive": MAX_FILE_BYTES,
        },
        "omittedDataPolicy": [
            "raw API requests and responses",
            "encrypted reasoning",
            "credentials, cookies, and authorization data",
            "evaluator caches",
            "host paths and environment configuration",
            "provider retries and internal request identifiers",
            "command output beyond the capped observation shown to the model",
        ],
        "files": files,
    }
    (build_dir / "package-manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_zip(build_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(build_dir).as_posix())
    temporary.replace(output)


def build_site(
    rows: list[dict[str, str]],
    build_dir: Path,
    results_root: Path,
    webp_quality: int,
    title: str = "OSWorld V2 CLI traces",
    task_workers: int = 1,
    rubric_map: dict[str, Any] | None = None,
    rubric_run_label: str = "",
) -> dict[str, Any]:
    if not STATIC_ROOT.exists():
        raise RuntimeError(f"Static viewer source is missing: {STATIC_ROOT}")
    shutil.copytree(STATIC_ROOT, build_dir)
    image_store = ImageStore(build_dir, webp_quality)
    profile = collection_profile(rows, title)
    prepared_rows = []
    for row in rows:
        prepared = dict(row)
        prepared.setdefault("benchmark", profile["benchmark"])
        prepared.setdefault("benchmark_label", profile["benchmarkLabel"])
        prepared.setdefault("model_label", profile["modelLabel"])
        prepared.setdefault("metric_label", profile["metricLabel"])
        prepared.setdefault("metric_kind", profile["metricKind"])
        prepared.setdefault("score_provenance", profile["scoreProvenance"])
        if not prepared.get("instruction", "").strip():
            loaded_title, instruction, related_software = load_task_details(
                prepared["task"]
            )
            prepared["instruction"] = instruction
            prepared["title"] = prepared.get("title", "").strip() or loaded_title
            if related_software and not prepared.get("related_software", "").strip():
                prepared["related_software"] = json.dumps(related_software)
        prepared_rows.append(prepared)
    task_count = len(prepared_rows)

    def build_one(index: int, row: dict[str, str]) -> tuple[int, dict, list[str]]:
        rubric_record = (
            (rubric_map.get("tasks") or {}).get(row["task"])
            if isinstance(rubric_map, dict)
            else None
        )
        task, task_sources = build_task(
            row,
            build_dir,
            results_root,
            image_store,
            rubric_record=rubric_record,
            rubric_run_label=rubric_run_label,
        )
        return index, task, task_sources

    completed: list[tuple[dict, list[str]] | None] = [None] * task_count
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(task_count, task_workers))
    ) as executor:
        futures = {
            executor.submit(build_one, index, row): (index, row)
            for index, row in enumerate(prepared_rows)
        }
        for future in concurrent.futures.as_completed(futures):
            index, row = futures[future]
            _, task, task_sources = future.result()
            completed[index] = (task, task_sources)
            finished = sum(item is not None for item in completed)
            print(
                f"[{finished:03d}/{task_count}] task {row['task']}: "
                f"{task['status']}, {task.get('eventCount', task['score'])}"
            )

    if any(item is None for item in completed):
        raise RuntimeError("Task build completed without every result")
    tasks = [item[0] for item in completed if item is not None]
    source_paths = [
        source for item in completed if item is not None for source in item[1]
    ]

    scored = [task["score"] for task in tasks if task["score"] is not None]
    catalog = {
        "format": "osworld-v2-cli-trace-viewer-v1",
        "title": title,
        "collection": profile,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "taskCount": len(tasks),
            "evaluated": sum(task["status"] == "evaluated" for task in tasks),
            "strictPasses": sum(task["strictPass"] for task in tasks),
            "scoreCount": len(scored),
            "scoreSum": sum(scored),
            "mean": statistics.fmean(scored) if scored else None,
        },
        "tasks": tasks,
    }
    write_js(build_dir / "data" / "catalog.js", "window.TRACE_CATALOG=", catalog)
    write_package_manifest(build_dir, catalog, source_paths)
    site_stats = validate_site(build_dir)
    return {
        "catalog": catalog,
        "site": site_stats,
    }


def main() -> None:
    args = parse_args()
    rows = read_selection_map(args.selection_map)
    temporary = tempfile.TemporaryDirectory(prefix="osworld-trace-viewer-")
    build_dir = Path(temporary.name) / "site"
    title = rows[0].get("viewer_title", "").strip() or "OSWorld V2 CLI traces"
    built = build_site(
        rows,
        build_dir,
        args.results_root,
        args.webp_quality,
        title=title,
        task_workers=args.workers,
        rubric_map=read_json(args.rubric_map, {}) if args.rubric_map else None,
        rubric_run_label=args.rubric_run_label,
    )
    build_zip(build_dir, args.output)
    zip_size = args.output.stat().st_size
    summary = {
        **built["site"],
        "zipPath": str(args.output),
        "zipBytes": zip_size,
        "zipSha256": hash_file(args.output),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if args.keep_build_dir:
        destination = args.output.with_suffix("")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(build_dir, destination)
        print(f"Extracted site copied to {destination}")
    temporary.cleanup()


if __name__ == "__main__":
    main()

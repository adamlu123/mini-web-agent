from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import typer
from rich.console import Console

from miniswewebagent import package_dir
from miniswewebagent.models.phyagi_model import parse_bash_answer_output

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object on line {line_number} of '{path.name}'.")
        rows.append(payload)
    return rows


def _isoformat_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _preview_text(value: str, limit: int = 180) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _safe_join(base: Path, *parts: str) -> Path:
    candidate = (base.joinpath(*parts)).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise FileNotFoundError("Path escapes the configured viewer root.") from exc
    return candidate


def _status_from_result(result: dict[str, Any]) -> str:
    exit_status = str(result.get("exit_status", "")).strip().lower()
    if exit_status == "submitted":
        return "submitted"
    if result.get("run_exception") or result.get("close_exception"):
        return "error"
    if exit_status and exit_status != "submitted":
        return "error"
    if result.get("final_result_response") or result.get("submission"):
        return "finished"
    return "incomplete"


def _judge_status_from_label(predicted_label: Any) -> str:
    if predicted_label == 1:
        return "success"
    if predicted_label == 0:
        return "failure"
    return "unknown"


def _judge_model_from_filename(file_name: str) -> str:
    marker = "_eval_"
    suffix = "_score_threshold_"
    if marker in file_name and suffix in file_name:
        return file_name.split(marker, 1)[1].split(suffix, 1)[0]
    return file_name


def _extract_observation(step_payload: dict[str, Any]) -> dict[str, Any]:
    outputs = step_payload.get("outputs", [])
    if not isinstance(outputs, list):
        return {}
    for item in reversed(outputs):
        if not isinstance(item, dict):
            continue
        observation = item.get("observation")
        if isinstance(observation, dict):
            return observation
    return {}


def _message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "input_text"}:
                parts.append(str(part.get("text") or ""))
            elif part.get("type") in {"image_url", "input_image"}:
                parts.append("[image]")
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _normalize_for_match(value: str) -> str:
    return " ".join(str(value or "").split())


def _parsed_sft_raw_responses(task_dir: Path) -> list[dict[str, Any]]:
    raw_path = task_dir / "raw_responses.jsonl"
    if not raw_path.exists():
        return []
    parsed_rows: list[dict[str, Any]] = []
    try:
        rows = _load_jsonl(raw_path)
    except Exception:
        return []
    for row in rows:
        raw_text = str(row.get("raw_text") or "")
        if not raw_text:
            continue
        try:
            parsed = parse_bash_answer_output(raw_text)
            parsed["_raw_text"] = raw_text
            parsed_rows.append(parsed)
        except Exception:
            continue
    return parsed_rows


def _raw_response_matches_step(parsed: dict[str, Any], *, command: str, final_response: str) -> bool:
    parsed_command = str(parsed.get("bash_command") or parsed.get("command") or "")
    if command and _normalize_for_match(parsed_command) == _normalize_for_match(command):
        return True
    parsed_final = str(parsed.get("final_response") or "")
    if final_response and _normalize_for_match(parsed_final) == _normalize_for_match(final_response):
        return True
    return False


def _raw_response_fallbacks_by_step(task_dir: Path, step_paths: list[Path]) -> dict[str, dict[str, Any]]:
    raw_rows = _parsed_sft_raw_responses(task_dir)
    if not raw_rows:
        return {}
    matched: dict[str, dict[str, Any]] = {}
    raw_index = 0
    for step_path in step_paths:
        try:
            step_payload = _load_json(step_path)
        except Exception:
            continue
        raw_response = step_payload.get("raw_response") if isinstance(step_payload.get("raw_response"), dict) else {}
        command = str(step_payload.get("bash_command") or step_payload.get("command_text") or raw_response.get("bash_command") or "")
        final_response = str(step_payload.get("final_response") or raw_response.get("final_response") or "")
        for index in range(raw_index, len(raw_rows)):
            parsed = raw_rows[index]
            if _raw_response_matches_step(parsed, command=command, final_response=final_response):
                matched[step_path.name] = parsed
                raw_index = index + 1
                break
    return matched


def _merge_raw_response(raw_response: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    if not fallback:
        return raw_response
    merged = dict(raw_response)
    for key in ("thought", "bash_command", "python_code", "done", "final_response"):
        if merged.get(key) in (None, "") and fallback.get(key) not in (None, ""):
            merged[key] = fallback.get(key)
    return merged


def _raw_text_from_payload(step_payload: dict[str, Any], raw_fallback: dict[str, Any] | None = None) -> str:
    raw_text = step_payload.get("raw_text")
    if raw_text:
        return str(raw_text)
    if raw_fallback and raw_fallback.get("_raw_text"):
        return str(raw_fallback.get("_raw_text"))
    return ""


def _normalize_trace_messages(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(trajectory.get("messages") or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip() or "unknown"
        content_text = _message_content_text(message.get("content"))
        extra = message.get("extra") if isinstance(message.get("extra"), dict) else {}
        observation = extra.get("observation") if isinstance(extra.get("observation"), dict) else {}
        actions = extra.get("actions") if isinstance(extra.get("actions"), list) else []
        raw_response = extra.get("raw_response") if isinstance(extra.get("raw_response"), dict) else {}
        raw_text = str(extra.get("raw_text") or "")

        bash_command = ""
        python_code = ""
        if isinstance(raw_response, dict):
            bash_command = str(raw_response.get("bash_command") or "")
            python_code = str(raw_response.get("python_code") or "")
        if not bash_command and actions:
            action = actions[0] if isinstance(actions[0], dict) else {}
            bash_command = str(action.get("bash_command") or action.get("command") or "")
            python_code = str(action.get("python_code") or "")

        normalized.append(
            {
                "index": index,
                "role": role,
                "content": content_text,
                "thought": str(raw_response.get("thought") or content_text if role == "assistant" else ""),
                "bashCommand": bash_command,
                "pythonCode": python_code,
                "done": bool(extra.get("done", False)),
                "finalResponse": str(extra.get("final_response") or raw_response.get("final_response") or ""),
                "interruptType": str(extra.get("interrupt_type") or ""),
                "observation": observation,
                "command": str(observation.get("command") or ""),
                "returnCode": observation.get("returncode"),
                "commandOutput": str(observation.get("command_output") or ""),
                "exception": str(observation.get("exception") or ""),
                "rawResponse": raw_response,
                "rawText": raw_text,
            }
        )
    return normalized


def _system_message_from_trajectory(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    for message in trajectory.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "system":
            normalized = _normalize_trace_messages({"messages": [message]})
            return normalized[0] if normalized else None
    system_template = (
        trajectory.get("info", {})
        .get("config", {})
        .get("agent", {})
        .get("system_template", "")
    )
    if system_template:
        return {
            "index": 0,
            "role": "system",
            "content": str(system_template),
            "thought": "",
            "bashCommand": "",
            "pythonCode": "",
            "done": False,
            "finalResponse": "",
            "interruptType": "",
            "observation": {},
            "command": "",
            "returnCode": None,
            "commandOutput": "",
            "exception": "",
            "rawResponse": {},
            "rawText": "",
        }
    return None


def _messages_from_step_files(task_dir: Path, trajectory: dict[str, Any], step_paths: list[Path]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    raw_fallbacks = _raw_response_fallbacks_by_step(task_dir, step_paths)
    system_message = _system_message_from_trajectory(trajectory)
    if system_message is not None:
        messages.append({**system_message, "index": 0})

    for step_position, step_path in enumerate(step_paths, start=1):
        step_payload = _load_json(step_path)
        raw_response = step_payload.get("raw_response") if isinstance(step_payload.get("raw_response"), dict) else {}
        raw_fallback = raw_fallbacks.get(step_path.name)
        display_raw_response = _merge_raw_response(raw_response, raw_fallback)
        bash_command = str(step_payload.get("bash_command") or step_payload.get("command_text") or "")
        python_code = str(step_payload.get("python_code") or "")
        thought = str(step_payload.get("thought") or raw_response.get("thought") or (raw_fallback or {}).get("thought") or "")
        assistant_index = len(messages)
        messages.append(
            {
                "index": assistant_index,
                "role": "assistant",
            "content": thought,
            "thought": thought,
                "bashCommand": bash_command,
                "pythonCode": python_code,
                "done": bool(step_payload.get("done", False)),
                "finalResponse": str(step_payload.get("final_response") or raw_response.get("final_response") or ""),
                "interruptType": f"step_{step_payload.get('step') or step_position}",
                "observation": {},
                "command": "",
                "returnCode": None,
                "commandOutput": "",
                "exception": "",
                "rawResponse": display_raw_response,
                "rawText": _raw_text_from_payload(step_payload, raw_fallback),
            }
        )
        outputs = step_payload.get("outputs") if isinstance(step_payload.get("outputs"), list) else []
        for output_index, output in enumerate(outputs, start=1):
            if not isinstance(output, dict):
                continue
            observation = output.get("observation") if isinstance(output.get("observation"), dict) else {}
            user_index = len(messages)
            messages.append(
                {
                    "index": user_index,
                    "role": "user",
                    "content": _message_content_text([{"type": "input_text", "text": str(observation.get("command_output") or output.get("output") or "")}]),
                    "thought": "",
                    "bashCommand": "",
                    "pythonCode": "",
                    "done": False,
                    "finalResponse": "",
                    "interruptType": f"observation_{step_payload.get('step') or step_position}.{output_index}",
                    "observation": observation,
                    "command": str(observation.get("command") or ""),
                    "returnCode": observation.get("returncode"),
                    "commandOutput": str(observation.get("command_output") or output.get("output") or ""),
                    "exception": str(observation.get("exception") or output.get("exception_info") or ""),
                    "rawResponse": {},
                    "rawText": "",
                }
            )
    return messages


def _step_files(task_dir: Path) -> list[Path]:
    steps_dir = task_dir / "debug" / "steps"
    if not steps_dir.exists():
        return []
    return sorted(steps_dir.glob("step_*.json"))


def _trajectory_screenshot(task_dir: Path, index: int) -> Path | None:
    candidate = task_dir / "trajectory" / f"{index}_full_screenshot.png"
    return candidate if candidate.exists() else None


def _raw_screenshot(task_dir: Path, index: int) -> Path | None:
    candidate = task_dir / "screenshots" / f"step_{index + 1:04d}.png"
    return candidate if candidate.exists() else None


def _build_step_detail(
    task_dir: Path,
    step_index: int,
    step_payload: dict[str, Any],
    raw_fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observation = _extract_observation(step_payload)
    screenshot = _raw_screenshot(task_dir, step_index) or _trajectory_screenshot(task_dir, step_index)
    screenshot_rel = str(screenshot.relative_to(task_dir)) if screenshot else ""
    raw_response = step_payload.get("raw_response") if isinstance(step_payload.get("raw_response"), dict) else {}
    thought = str(step_payload.get("thought") or raw_response.get("thought") or (raw_fallback or {}).get("thought") or "").strip()
    return {
        "index": step_index,
        "step": int(step_payload.get("step") or step_index + 1),
        "thought": thought,
        "action": str(
            step_payload.get("command_text")
            or step_payload.get("bash_command")
            or step_payload.get("python_code")
            or ""
        ).strip(),
        "done": bool(step_payload.get("done")),
        "finalResponse": str(step_payload.get("final_response") or "").strip(),
        "url": str(observation.get("url") or ""),
        "title": str(observation.get("title") or ""),
        "success": observation.get("success"),
        "exception": str(observation.get("exception") or "").strip(),
        "consoleOutput": str(observation.get("console_output") or "").strip(),
        "recentConsole": str(observation.get("recent_console") or "").strip(),
        "ariaSnapshot": str(observation.get("aria_snapshot") or "").strip(),
        "screenshotRelPath": screenshot_rel,
        "rawResponse": _merge_raw_response(raw_response, raw_fallback),
        "rawText": _raw_text_from_payload(step_payload, raw_fallback),
    }


def _build_fallback_steps(task_dir: Path, result: dict[str, Any]) -> list[dict[str, Any]]:
    actions = result.get("action_history", [])
    thoughts = result.get("thoughts", [])
    screenshot_paths = result.get("screenshot_paths", [])
    steps: list[dict[str, Any]] = []
    total = max(len(actions), len(thoughts), len(screenshot_paths))
    for index in range(total):
        screenshot_rel = ""
        shot = _trajectory_screenshot(task_dir, index)
        if shot:
            screenshot_rel = str(shot.relative_to(task_dir))
        steps.append(
            {
                "index": index,
                "step": index + 1,
                "thought": str(thoughts[index]) if index < len(thoughts) else "",
                "action": str(actions[index]) if index < len(actions) else "",
                "done": False,
                "finalResponse": "",
                "url": "",
                "title": "",
                "success": None,
                "exception": "",
                "consoleOutput": "",
                "recentConsole": "",
                "ariaSnapshot": "",
                "screenshotRelPath": screenshot_rel,
                "rawResponse": None,
                "rawText": "",
            }
        )
    return steps


@dataclass(frozen=True)
class RunRef:
    run_id: str
    path: Path


class TraceCatalog:
    ROOT_RUN_ID = "__root__"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir.resolve()

    def _is_run_dir(self, path: Path) -> bool:
        return path.is_dir() and any((child / "result.json").exists() for child in path.iterdir() if child.is_dir())

    def _run_dir(self, run_id: str) -> Path:
        if run_id == self.ROOT_RUN_ID:
            return self.root_dir
        return _safe_join(self.root_dir, run_id)

    def _task_dir(self, run_id: str, task_id: str) -> Path:
        return _safe_join(self._run_dir(run_id), task_id)

    def _judge_files(self, run_dir: Path) -> list[Path]:
        files = list(run_dir.glob("WebJudge_*_auto_eval_results.json"))
        sibling_pattern = f"{run_dir.name}_eval_*"
        for sibling in sorted(run_dir.parent.glob(sibling_pattern)):
            if sibling.is_dir():
                files.extend(sibling.glob("WebJudge_*_auto_eval_results.json"))
        return sorted(files)

    def _judge_entries_for_task(self, run_dir: Path, *, task_key: str, task_id: str) -> list[dict[str, Any]]:
        judge_entries: list[dict[str, Any]] = []
        task_identifiers = {task_key.strip(), task_id.strip()}
        task_identifiers.discard("")

        for judge_path in self._judge_files(run_dir):
            matching_row = next(
                (
                    row
                    for row in _load_jsonl(judge_path)
                    if str(row.get("task_id") or "").strip() in task_identifiers
                ),
                None,
            )
            if matching_row is None:
                continue

            evaluation_details = matching_row.get("evaluation_details")
            if not isinstance(evaluation_details, dict):
                evaluation_details = {}

            predicted_label = matching_row.get("predicted_label")
            judge_entries.append(
                {
                    "id": judge_path.name,
                    "fileName": judge_path.name,
                    "filePath": str(judge_path),
                    "updatedAt": _isoformat_mtime(judge_path),
                    "model": _judge_model_from_filename(judge_path.name),
                    "predictedLabel": predicted_label,
                    "status": _judge_status_from_label(predicted_label),
                    "response": str(evaluation_details.get("response") or ""),
                    "evaluationDetails": evaluation_details,
                    "imageJudgeRecord": matching_row.get("image_judge_record"),
                    "raw": matching_row,
                }
            )

        return judge_entries

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.root_dir.exists():
            return []

        if self._is_run_dir(self.root_dir):
            task_count = sum(1 for item in self.root_dir.iterdir() if (item / "result.json").exists())
            return [
                {
                    "id": self.ROOT_RUN_ID,
                    "name": self.root_dir.name,
                    "taskCount": task_count,
                    "updatedAt": _isoformat_mtime(self.root_dir),
                }
            ]

        run_dirs = [path for path in self.root_dir.rglob("*") if self._is_run_dir(path)]
        runs: list[dict[str, Any]] = []
        for child in sorted(run_dirs, key=lambda path: path.stat().st_mtime, reverse=True):
            task_count = sum(1 for item in child.iterdir() if (item / "result.json").exists())
            run_id = str(child.relative_to(self.root_dir))
            if not task_count:
                continue
            runs.append(
                {
                    "id": run_id,
                    "name": run_id,
                    "taskCount": task_count,
                    "updatedAt": _isoformat_mtime(child),
                }
            )
        return runs

    def list_tasks(self, run_id: str) -> list[dict[str, Any]]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run '{run_id}' was not found.")

        tasks: list[dict[str, Any]] = []
        for task_dir in sorted(run_dir.iterdir()):
            result_path = task_dir / "result.json"
            if not result_path.exists():
                continue
            result = _load_json(result_path)
            resolved_task_id = str(result.get("task_id") or task_dir.name)
            judges = self._judge_entries_for_task(run_dir, task_key=task_dir.name, task_id=resolved_task_id)
            best_judge_status = "unknown"
            if any(judge.get("status") == "success" for judge in judges):
                best_judge_status = "success"
            elif any(judge.get("status") == "failure" for judge in judges):
                best_judge_status = "failure"
            step_count = len(_step_files(task_dir)) or len(result.get("action_history", []))
            final_text = str(result.get("final_result_response") or result.get("submission") or "")
            tasks.append(
                {
                    "id": task_dir.name,
                    "taskId": resolved_task_id,
                    "title": str(result.get("task") or ""),
                    "status": _status_from_result(result),
                    "judgeStatus": best_judge_status,
                    "stepCount": step_count,
                    "finalPreview": _preview_text(final_text),
                    "updatedAt": _isoformat_mtime(result_path),
                }
            )
        return tasks

    def task_detail(self, run_id: str, task_id: str) -> dict[str, Any]:
        task_dir = self._task_dir(run_id, task_id)
        run_dir = self._run_dir(run_id)
        result_path = task_dir / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Task '{task_id}' was not found in run '{run_id}'.")

        result = _load_json(result_path)
        trajectory: dict[str, Any] = {}
        trajectory_path = task_dir / "trajectory.json"
        if trajectory_path.exists():
            trajectory = _load_json(trajectory_path)
        step_paths = _step_files(task_dir)
        if step_paths:
            raw_fallbacks = _raw_response_fallbacks_by_step(task_dir, step_paths)
            steps = [
                _build_step_detail(task_dir, idx, _load_json(path), raw_fallbacks.get(path.name))
                for idx, path in enumerate(step_paths)
            ]
        else:
            steps = _build_fallback_steps(task_dir, result)

        resolved_task_id = str(result.get("task_id") or task_id)
        judges = self._judge_entries_for_task(run_dir, task_key=task_id, task_id=resolved_task_id)

        return {
            "runId": run_id,
            "taskKey": task_id,
            "status": _status_from_result(result),
            "taskId": resolved_task_id,
            "task": str(result.get("task") or ""),
            "startUrl": str(result.get("start_url") or ""),
            "finalResult": str(result.get("final_result_response") or result.get("submission") or ""),
            "exitStatus": str(result.get("exit_status") or ""),
            "runException": str(result.get("run_exception") or ""),
            "closeException": str(result.get("close_exception") or ""),
            "stepCount": len(steps),
            "judges": judges,
            "result": result,
            "steps": steps,
            "messages": _messages_from_step_files(task_dir, trajectory, step_paths) if step_paths else _normalize_trace_messages(trajectory),
        }

    def artifact(self, run_id: str, task_id: str, file_rel_path: str) -> Path:
        task_dir = self._task_dir(run_id, task_id)
        return _safe_join(task_dir, file_rel_path)


class TraceViewerHandler(BaseHTTPRequestHandler):
    catalog: TraceCatalog
    static_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                self._write_json({"rootDir": str(self.catalog.root_dir), "runs": self.catalog.list_runs()})
                return
            if parsed.path == "/api/tasks":
                params = parse_qs(parsed.query)
                run_id = self._require_query(params, "run")
                self._write_json({"runId": run_id, "tasks": self.catalog.list_tasks(run_id)})
                return
            if parsed.path == "/api/task":
                params = parse_qs(parsed.query)
                run_id = self._require_query(params, "run")
                task_id = self._require_query(params, "task")
                self._write_json(self.catalog.task_detail(run_id, task_id))
                return
            if parsed.path == "/artifact":
                params = parse_qs(parsed.query)
                run_id = self._require_query(params, "run")
                task_id = self._require_query(params, "task")
                file_rel = self._require_query(params, "file")
                self._write_file(self.catalog.artifact(run_id, task_id, file_rel))
                return
            self._serve_static(parsed.path)
        except FileNotFoundError as exc:
            self._write_error(HTTPStatus.NOT_FOUND, str(exc))
        except ValueError as exc:
            self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive path for manual use
            self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        del format, args

    def _require_query(self, params: dict[str, list[str]], key: str) -> str:
        value = params.get(key, [""])[0].strip()
        if not value:
            raise ValueError(f"Missing required query parameter: {key}")
        return value

    def _write_json(self, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_error(self, status: HTTPStatus, message: str) -> None:
        payload = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {path.name}")
        content = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_static(self, request_path: str) -> None:
        relative = request_path.lstrip("/") or "index.html"
        target = _safe_join(self.static_dir, relative)
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            target = self.static_dir / "index.html"
        self._write_file(target)


def make_handler(catalog: TraceCatalog, static_dir: Path) -> type[TraceViewerHandler]:
    class _Handler(TraceViewerHandler):
        pass

    _Handler.catalog = catalog
    _Handler.static_dir = static_dir.resolve()
    return _Handler


def build_server(root_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    catalog = TraceCatalog(root_dir)
    static_dir = package_dir / "viewer"
    return ThreadingHTTPServer((host, port), make_handler(catalog, static_dir))


@app.command()
def main(
    root: Path = typer.Option(Path("outputs/default"), "--root", help="Directory containing run folders."),
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind."),
    port: int = typer.Option(8009, "--port", min=0, max=65535, help="Port to bind."),
) -> None:
    root_dir = root.expanduser().resolve()
    if not root_dir.exists():
        raise typer.BadParameter(f"Root directory '{root_dir}' does not exist.")

    server = build_server(root_dir, host, port)
    bound_host, bound_port = server.server_address[:2]
    console.print(f"Trace viewer root: [bold]{root_dir}[/bold]")
    console.print(f"Open [bold green]http://{bound_host}:{bound_port}[/bold green]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Stopping trace viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    app()

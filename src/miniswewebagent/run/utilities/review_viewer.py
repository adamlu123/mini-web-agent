from __future__ import annotations

import json
import mimetypes
import re
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import typer
from rich.console import Console

from miniswewebagent import package_dir

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


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


def _safe_join(base: Path, *parts: str) -> Path:
    candidate = base.joinpath(*parts).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise FileNotFoundError("Path escapes the configured viewer root.") from exc
    return candidate


def _isoformat_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _preview_text(value: str, limit: int = 180) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."


def _natural_sort_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _image_id_from_name(name: str) -> int:
    match = re.search(r"final_execution_(\d+)", name.lower())
    if match:
        return int(match.group(1))
    return 10**9


def _image_sort_key(path: Path, screenshots_dir: Path) -> tuple[int, list[Any]]:
    relative = str(path.relative_to(screenshots_dir))
    name = path.name.lower()
    priority = 0 if name.startswith("final_execution") else 1
    return priority, [_image_id_from_name(name)], _natural_sort_key(relative)


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


def _extract_reason_from_response(response: str) -> tuple[str, str]:
    text = str(response or "").strip()
    status_match = re.search(r"Status:\s*\"?(success|failure)\"?", text, flags=re.IGNORECASE)
    thoughts_match = re.search(r"Thoughts:\s*(.*?)(?:\n\s*Status:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    status = status_match.group(1).lower() if status_match else "unknown"
    reason = thoughts_match.group(1).strip() if thoughts_match else text
    return status, _clean_reason(reason)


def _clean_reason(reason: str) -> str:
    text = " ".join(str(reason or "").split())
    text = re.sub(r"\s*Status:\s*(success|failure)\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _parse_summary_row(line: str) -> tuple[str, str, str] | None:
    match = re.match(
        r"^\|\s*`(?P<task_id>[^`]+)`\s*\|\s*`?(?P<status>[^`|]+)`?\s*\|\s*(?P<reason>.*)\|\s*$",
        line.strip(),
    )
    if not match:
        return None
    task_id = match.group("task_id").strip()
    status = match.group("status").strip().lower()
    reason = _clean_reason(match.group("reason").replace(r"\|", "|"))
    return task_id, status, reason


class ReviewCatalog:
    def __init__(self, runs_root: Path, judge_root: Path):
        self.runs_root = runs_root.resolve()
        self.judge_root = judge_root.resolve()
        self._judge_cache: dict[tuple[str, float], dict[str, dict[str, str]]] = {}

    def _run_dir(self, run_id: str) -> Path:
        return _safe_join(self.runs_root, run_id)

    def _task_dir(self, run_id: str, task_key: str) -> Path:
        return _safe_join(self._run_dir(run_id), task_key)

    def _judge_path(self, judge_id: str) -> Path:
        return _safe_join(self.judge_root, judge_id)

    def _judge_lookup(self, judge_id: str | None) -> dict[str, dict[str, str]]:
        if not judge_id:
            return {}
        judge_path = self._judge_path(judge_id)
        cache_key = (str(judge_path), judge_path.stat().st_mtime)
        if cache_key in self._judge_cache:
            return self._judge_cache[cache_key]

        suffix = judge_path.suffix.lower()
        lookup: dict[str, dict[str, str]] = {}
        if suffix == ".md":
            for line in judge_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_summary_row(line)
                if parsed is None:
                    continue
                task_id, status, reason = parsed
                lookup[task_id] = {
                    "status": status or "unknown",
                    "reason": reason,
                    "source": "summary",
                    "model": "",
                    "response": "",
                }
            sibling_jsons = sorted(judge_path.parent.glob("WebJudge_*_auto_eval_results.json"))
            sibling_json = sibling_jsons[0] if sibling_jsons else None
            if sibling_json is not None:
                for row in _load_jsonl(sibling_json):
                    task_id = str(row.get("task_id") or "").strip()
                    if task_id not in lookup:
                        continue
                    evaluation_details = row.get("evaluation_details")
                    if not isinstance(evaluation_details, dict):
                        evaluation_details = {}
                    lookup[task_id]["model"] = _judge_model_from_filename(sibling_json.name)
                    lookup[task_id]["response"] = str(evaluation_details.get("response") or "")
        elif suffix == ".json":
            summary_lookup: dict[str, tuple[str, str]] = {}
            sibling_summary = judge_path.parent / "SUMMARY.md"
            if sibling_summary.exists():
                for line in sibling_summary.read_text(encoding="utf-8").splitlines():
                    parsed = _parse_summary_row(line)
                    if parsed is None:
                        continue
                    task_id, status, reason = parsed
                    summary_lookup[task_id] = (status, reason)
            for row in _load_jsonl(judge_path):
                task_id = str(row.get("task_id") or "").strip()
                if not task_id:
                    continue
                evaluation_details = row.get("evaluation_details")
                if not isinstance(evaluation_details, dict):
                    evaluation_details = {}
                response = str(evaluation_details.get("response") or "")
                status, reason = _extract_reason_from_response(response)
                if status == "unknown":
                    status = _judge_status_from_label(row.get("predicted_label"))
                summary_status, summary_reason = summary_lookup.get(task_id, ("", ""))
                lookup[task_id] = {
                    "status": summary_status or status,
                    "reason": summary_reason or reason,
                    "source": "json",
                    "model": _judge_model_from_filename(judge_path.name),
                    "response": response,
                }
        else:
            raise ValueError(f"Unsupported judge file type: {judge_path.name}")

        self._judge_cache[cache_key] = lookup
        return lookup

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.runs_root.exists():
            return []

        runs: list[dict[str, Any]] = []
        for child in sorted(self.runs_root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if not child.is_dir():
                continue
            task_count = sum(1 for item in child.iterdir() if (item / "result.json").exists())
            if not task_count:
                continue
            runs.append(
                {
                    "id": child.name,
                    "name": child.name,
                    "taskCount": task_count,
                    "updatedAt": _isoformat_mtime(child),
                }
            )
        return runs

    def list_judge_files(self) -> list[dict[str, Any]]:
        if not self.judge_root.exists():
            return []

        paths = sorted(
            self.judge_root.rglob("*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        files: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                continue
            if path.name == "SUMMARY.md":
                kind = "summary"
            elif re.match(r"WebJudge_.*_auto_eval_results\.json$", path.name):
                kind = "json"
            else:
                continue
            relative = str(path.relative_to(self.judge_root))
            files.append(
                {
                    "id": relative,
                    "name": relative,
                    "kind": kind,
                    "updatedAt": _isoformat_mtime(path),
                }
            )
        return files

    def list_tasks(self, run_id: str, judge_id: str | None = None) -> list[dict[str, Any]]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run '{run_id}' was not found.")

        judge_lookup = self._judge_lookup(judge_id)
        tasks: list[dict[str, Any]] = []
        for task_dir in sorted(run_dir.iterdir()):
            result_path = task_dir / "result.json"
            if not result_path.exists():
                continue
            result = _load_json(result_path)
            resolved_task_id = str(result.get("task_id") or task_dir.name)
            judge_entry = judge_lookup.get(resolved_task_id) or judge_lookup.get(task_dir.name) or {}
            image_count = len(self._task_images(task_dir))
            tasks.append(
                {
                    "id": task_dir.name,
                    "taskId": resolved_task_id,
                    "title": str(result.get("task") or ""),
                    "imageCount": image_count,
                    "judgeStatus": str(judge_entry.get("status") or "unknown"),
                    "judgeReasonPreview": _preview_text(str(judge_entry.get("reason") or "")),
                    "updatedAt": _isoformat_mtime(result_path),
                }
            )
        return tasks

    def _task_images(self, task_dir: Path) -> list[Path]:
        shots_dir = task_dir / "screenshots"
        if not shots_dir.exists():
            return []
        files = [
            path
            for path in shots_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.name.lower().startswith("final_execution")
        ]
        return sorted(files, key=lambda path: _image_sort_key(path, shots_dir))

    def task_detail(self, run_id: str, task_key: str, judge_id: str | None = None) -> dict[str, Any]:
        task_dir = self._task_dir(run_id, task_key)
        result_path = task_dir / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Task '{task_key}' was not found in run '{run_id}'.")

        result = _load_json(result_path)
        resolved_task_id = str(result.get("task_id") or task_key)

        judge_entry = {}
        if judge_id:
            judge_lookup = self._judge_lookup(judge_id)
            judge_entry = judge_lookup.get(resolved_task_id) or judge_lookup.get(task_key) or {}

        images = [
            {
                "name": str(path.relative_to(task_dir / "screenshots")),
                "relPath": str(path.relative_to(task_dir)),
                "updatedAt": _isoformat_mtime(path),
            }
            for path in self._task_images(task_dir)
        ]

        return {
            "runId": run_id,
            "taskKey": task_key,
            "taskId": resolved_task_id,
            "task": str(result.get("task") or ""),
            "startUrl": str(result.get("start_url") or ""),
            "finalResponse": str(result.get("final_result_response") or result.get("submission") or ""),
            "lastAction": str((result.get("action_history") or [""])[-1] or ""),
            "lastThought": str((result.get("thoughts") or [""])[-1] or ""),
            "exitStatus": str(result.get("exit_status") or ""),
            "images": images,
            "judge": {
                "fileId": judge_id or "",
                "status": str(judge_entry.get("status") or "unknown"),
                "reason": str(judge_entry.get("reason") or ""),
                "source": str(judge_entry.get("source") or ""),
                "model": str(judge_entry.get("model") or ""),
                "response": str(judge_entry.get("response") or ""),
            },
            "result": result,
        }

    def artifact(self, run_id: str, task_key: str, file_rel_path: str) -> Path:
        task_dir = self._task_dir(run_id, task_key)
        return _safe_join(task_dir, file_rel_path)


class ReviewViewerHandler(BaseHTTPRequestHandler):
    catalog: ReviewCatalog
    static_dir: Path

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/runs":
                self._write_json(
                    {
                        "runsRoot": str(self.catalog.runs_root),
                        "judgeRoot": str(self.catalog.judge_root),
                        "runs": self.catalog.list_runs(),
                    }
                )
                return
            if parsed.path == "/api/judges":
                self._write_json({"judgeFiles": self.catalog.list_judge_files()})
                return
            if parsed.path == "/api/tasks":
                params = parse_qs(parsed.query)
                run_id = self._require_query(params, "run")
                judge_id = params.get("judge", [""])[0].strip() or None
                self._write_json({"runId": run_id, "tasks": self.catalog.list_tasks(run_id, judge_id)})
                return
            if parsed.path == "/api/task":
                params = parse_qs(parsed.query)
                run_id = self._require_query(params, "run")
                task_id = self._require_query(params, "task")
                judge_id = params.get("judge", [""])[0].strip() or None
                self._write_json(self.catalog.task_detail(run_id, task_id, judge_id))
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
        except Exception as exc:  # pragma: no cover
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


def make_handler(catalog: ReviewCatalog, static_dir: Path) -> type[ReviewViewerHandler]:
    class _Handler(ReviewViewerHandler):
        pass

    _Handler.catalog = catalog
    _Handler.static_dir = static_dir.resolve()
    return _Handler


def build_server(runs_root: Path, judge_root: Path, host: str, port: int) -> ThreadingHTTPServer:
    catalog = ReviewCatalog(runs_root, judge_root)
    static_dir = package_dir / "review_viewer"
    return ThreadingHTTPServer((host, port), make_handler(catalog, static_dir))


@app.command()
def main(
    runs_root: Path = typer.Option(
        Path("outputs/sandbox"),
        "--runs-root",
        help="Directory containing run folders.",
    ),
    judge_root: Path = typer.Option(
        Path("om2w_judge"),
        "--judge-root",
        help="Directory containing judge outputs.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface to bind."),
    port: int = typer.Option(8010, "--port", min=0, max=65535, help="Port to bind."),
) -> None:
    resolved_runs_root = runs_root.expanduser().resolve()
    resolved_judge_root = judge_root.expanduser().resolve()
    if not resolved_runs_root.exists():
        raise typer.BadParameter(f"Runs root '{resolved_runs_root}' does not exist.")
    if not resolved_judge_root.exists():
        raise typer.BadParameter(f"Judge root '{resolved_judge_root}' does not exist.")

    server = build_server(resolved_runs_root, resolved_judge_root, host, port)
    bound_host, bound_port = server.server_address[:2]
    console.print(f"Review viewer runs root: [bold]{resolved_runs_root}[/bold]")
    console.print(f"Review viewer judge root: [bold]{resolved_judge_root}[/bold]")
    console.print(f"Open [bold green]http://{bound_host}:{bound_port}[/bold green]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("Stopping review viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    app()

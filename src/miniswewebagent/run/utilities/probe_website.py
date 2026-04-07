from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miniswewebagent.run.mini import DEFAULT_CONFIG, run_one

DEFAULT_PROBE_CONFIG = "benchmark/om2w_probe_subagent.yaml"
DEFAULT_PROBE_CONFIGS = [DEFAULT_CONFIG, DEFAULT_PROBE_CONFIG]
_MAX_SLUG_LENGTH = 48

app = typer.Typer(no_args_is_help=True)
console = Console(highlight=False)


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not normalized:
        return "probe"
    return normalized[:_MAX_SLUG_LENGTH].strip("-") or "probe"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _copy_if_exists(source: Path, target: Path) -> str:
    if not source.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _copy_parent_context(parent_workspace: Path, probe_workspace: Path) -> dict[str, str]:
    copied = {
        "parent_task_json": _copy_if_exists(parent_workspace / "task.json", probe_workspace / "parent_task.json"),
        "parent_plan_md": _copy_if_exists(parent_workspace / "plan.md", probe_workspace / "parent_plan.md"),
        "parent_final_script": _copy_if_exists(
            parent_workspace / "final_script.py", probe_workspace / "parent_final_script.py"
        ),
    }
    return {key: value for key, value in copied.items() if value}


def _render_probe_task(*, parent_task: str, objective: str, probe_name: str, request_path: Path) -> str:
    parts = [
        "You are running a focused website-probing sub-agent.",
        f"Probe name: {probe_name}.",
        f"Probe objective: {objective.strip()}",
        f"Probe request file: {request_path.name}.",
    ]
    if parent_task.strip():
        parts.append(f"Parent task: {parent_task.strip()}")
    parts.append(
        "Inspect the site to discover controls, selector recipes, hidden drawers/dropdowns, "
        "result-surface entry points, and blockers. Do not try to fully solve the parent task."
    )
    return "\n".join(parts)


def _write_probe_request(
    *,
    probe_workspace: Path,
    probe_id: str,
    probe_name: str,
    objective: str,
    start_url: str,
    parent_workspace: Path,
    parent_task_record: dict[str, Any],
    copied_context_files: dict[str, str],
) -> Path:
    request_path = probe_workspace / "probe_request.json"
    request_payload = {
        "probe_id": probe_id,
        "probe_name": probe_name,
        "objective": objective,
        "start_url": start_url,
        "parent_workspace": str(parent_workspace),
        "parent_task": parent_task_record.get("task") or parent_task_record.get("confirmed_task") or "",
        "parent_task_id": parent_task_record.get("task_id", ""),
        "copied_context_files": copied_context_files,
        "expected_outputs": {
            "probe_summary_json": "probe_summary.json",
            "probe_summary_md": "probe_summary.md",
            "probe_script": "probe_script.py",
        },
    }
    request_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")
    return request_path


def _render_controls_markdown(summary: dict[str, Any]) -> list[str]:
    controls = summary.get("controls", [])
    if not isinstance(controls, list) or not controls:
        return ["No control inventory was recorded."]

    lines = ["| Constraint | State | Surface | Locator | Evidence |", "| --- | --- | --- | --- | --- |"]
    for item in controls:
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("constraint", "")).replace("\n", " "),
                    str(item.get("state", "")).replace("\n", " "),
                    str(item.get("surface", "")).replace("\n", " "),
                    str(item.get("recommended_locator", "")).replace("\n", " "),
                    str(item.get("evidence", "")).replace("\n", " "),
                ]
            )
            + " |"
        )
    return lines


def _render_probe_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Probe Website Summary",
        "",
        f"- Probe ID: `{summary.get('probe_id', '')}`",
        f"- Probe name: `{summary.get('probe_name', '')}`",
        f"- Objective: {summary.get('objective', '')}",
        f"- Exit status: `{summary.get('exit_status', '')}`",
        f"- Site status: `{summary.get('site_status', '')}`",
        f"- Start URL: {summary.get('start_url', '')}",
        f"- Final URL: {summary.get('final_url', '')}",
        f"- Probe run dir: `{summary.get('probe_run_dir', '')}`",
        "",
        "## Summary",
        "",
        str(summary.get("summary", "")).strip() or "No summary was recorded.",
        "",
        "## Controls",
        "",
        *_render_controls_markdown(summary),
        "",
        "## Blockers",
        "",
    ]

    blockers = summary.get("blockers", [])
    if isinstance(blockers, list) and blockers:
        lines.extend([f"- {str(item)}" for item in blockers])
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Suggested Next Actions", ""])
    next_actions = summary.get("suggested_next_actions", [])
    if isinstance(next_actions, list) and next_actions:
        lines.extend([f"- {str(item)}" for item in next_actions])
    else:
        lines.append("- None recorded.")

    lines.extend(["", "## Artifacts", ""])
    artifact_lines = []
    artifacts = summary.get("artifacts", {})
    if isinstance(artifacts, dict):
        for key in ("probe_script", "screenshots", "logs"):
            value = artifacts.get(key)
            if isinstance(value, list):
                artifact_lines.append(f"- {key}: {', '.join(str(item) for item in value)}")
            elif value:
                artifact_lines.append(f"- {key}: {value}")
    if artifact_lines:
        lines.extend(artifact_lines)
    else:
        lines.append("- No artifacts were recorded.")

    source_probe_markdown = str(summary.get("source_probe_markdown", "")).strip()
    if source_probe_markdown:
        lines.extend(["", "## Probe Agent Notes", "", source_probe_markdown])

    return "\n".join(lines).strip() + "\n"


def _load_probe_summary(probe_workspace: Path) -> dict[str, Any]:
    summary_path = probe_workspace / "probe_summary.json"
    payload = _load_json(summary_path)
    return payload if payload else {}


def _normalize_probe_summary(
    *,
    probe_id: str,
    probe_name: str,
    objective: str,
    start_url: str,
    parent_workspace: Path,
    probe_workspace: Path,
    copied_context_files: dict[str, str],
    result: dict[str, Any],
) -> dict[str, Any]:
    loaded = _load_probe_summary(probe_workspace)
    summary = loaded if loaded else {}
    summary.setdefault("summary", str(result.get("final_response") or result.get("submission") or "").strip())
    summary.setdefault("site_status", "unknown")
    summary.setdefault("controls", [])
    summary.setdefault("blockers", [])
    summary.setdefault("suggested_next_actions", [])
    summary.setdefault("artifacts", {})
    if not isinstance(summary.get("artifacts"), dict):
        summary["artifacts"] = {}

    artifacts = summary["artifacts"]
    artifacts.setdefault("probe_script", "probe_script.py" if (probe_workspace / "probe_script.py").exists() else "")
    artifacts.setdefault(
        "screenshots",
        [str(path.relative_to(probe_workspace)) for path in sorted((probe_workspace / "screenshots").glob("*"))[:10]],
    )
    artifacts.setdefault(
        "logs",
        [str(path.relative_to(probe_workspace)) for path in sorted((probe_workspace / "logs").glob("*.log"))[:10]],
    )

    summary.update(
        {
            "probe_id": probe_id,
            "probe_name": probe_name,
            "objective": objective,
            "start_url": start_url,
            "exit_status": str(result.get("exit_status", "")),
            "run_exception": str(result.get("run_exception", "")),
            "final_response": str(result.get("final_response") or result.get("submission") or ""),
            "probe_run_dir": str(probe_workspace),
            "parent_workspace": str(parent_workspace),
            "copied_context_files": copied_context_files,
            "probe_summary_path": str(probe_workspace / "probe_summary.json")
            if (probe_workspace / "probe_summary.json").exists()
            else "",
            "probe_markdown_path": str(probe_workspace / "probe_summary.md")
            if (probe_workspace / "probe_summary.md").exists()
            else "",
            "source_probe_markdown": (probe_workspace / "probe_summary.md").read_text(encoding="utf-8")
            if (probe_workspace / "probe_summary.md").exists()
            else "",
            "trajectory_path": str(probe_workspace / "trajectory.json")
            if (probe_workspace / "trajectory.json").exists()
            else "",
            "result_json": str(probe_workspace / "result.json") if (probe_workspace / "result.json").exists() else "",
        }
    )
    return summary


def _write_parent_reports(parent_workspace: Path, summary: dict[str, Any]) -> dict[str, str]:
    report_dir = parent_workspace / "probe_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    probe_id = str(summary.get("probe_id", "probe"))
    json_path = report_dir / f"{probe_id}.json"
    markdown_path = report_dir / f"{probe_id}.md"
    latest_json_path = report_dir / "latest.json"
    latest_markdown_path = report_dir / "latest.md"

    json_payload = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    markdown_payload = _render_probe_markdown(summary)

    json_path.write_text(json_payload, encoding="utf-8")
    markdown_path.write_text(markdown_payload, encoding="utf-8")
    latest_json_path.write_text(json_payload, encoding="utf-8")
    latest_markdown_path.write_text(markdown_payload, encoding="utf-8")

    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "latest_json_path": str(latest_json_path),
        "latest_markdown_path": str(latest_markdown_path),
    }


def run_probe(
    *,
    workspace_dir: Path,
    objective: str,
    probe_name: str = "",
    start_url: str = "",
    max_steps: int = 8,
    config_spec: list[str] | None = None,
) -> dict[str, Any]:
    parent_workspace = workspace_dir.expanduser().resolve()
    parent_workspace.mkdir(parents=True, exist_ok=True)

    resolved_probe_name = probe_name.strip() or _slugify(objective)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_id = f"{stamp}_{_slugify(resolved_probe_name)}"
    probe_workspace = parent_workspace / "probe_runs" / probe_id
    probe_workspace.mkdir(parents=True, exist_ok=True)

    copied_context_files = _copy_parent_context(parent_workspace, probe_workspace)
    parent_task_record = _load_json(parent_workspace / "task.json")
    resolved_start_url = (
        start_url.strip()
        or str(parent_task_record.get("start_url") or parent_task_record.get("website") or "").strip()
    )
    request_path = _write_probe_request(
        probe_workspace=probe_workspace,
        probe_id=probe_id,
        probe_name=resolved_probe_name,
        objective=objective,
        start_url=resolved_start_url,
        parent_workspace=parent_workspace,
        parent_task_record=parent_task_record,
        copied_context_files=copied_context_files,
    )
    task_text = _render_probe_task(
        parent_task=str(parent_task_record.get("task") or parent_task_record.get("confirmed_task") or ""),
        objective=objective,
        probe_name=resolved_probe_name,
        request_path=request_path,
    )

    merged_config_spec = list(config_spec or DEFAULT_PROBE_CONFIGS)
    merged_config_spec.extend(
        [
            "run.judge_enabled=false",
            f"agent.step_limit={max(1, int(max_steps))}",
        ]
    )

    try:
        result = run_one(
            task=task_text,
            task_id=probe_id,
            start_url=resolved_start_url or None,
            config_spec=merged_config_spec,
            resolved_output_dir=probe_workspace,
        )
    except Exception as exc:
        result = {
            "exit_status": type(exc).__name__,
            "submission": "",
            "final_response": "",
            "run_exception": str(exc),
            "_output_dir": str(probe_workspace),
        }

    summary = _normalize_probe_summary(
        probe_id=probe_id,
        probe_name=resolved_probe_name,
        objective=objective,
        start_url=resolved_start_url,
        parent_workspace=parent_workspace,
        probe_workspace=probe_workspace,
        copied_context_files=copied_context_files,
        result=result,
    )
    report_paths = _write_parent_reports(parent_workspace, summary)
    summary["report_paths"] = report_paths
    _write_parent_reports(parent_workspace, summary)
    return summary


@app.command()
def main(
    workspace_dir: Path = typer.Option(..., "--workspace-dir", help="Parent task workspace directory."),
    objective: str = typer.Option(..., "--objective", help="Focused selector/control discovery objective."),
    probe_name: str = typer.Option("", "--probe-name", help="Optional short name for this probe run."),
    start_url: str = typer.Option("", "--start-url", help="Optional override for the site entry URL."),
    max_steps: int = typer.Option(8, "--max-steps", help="Step budget for the nested probe agent."),
    config_spec: list[str] = typer.Option([], "-c", "--config", help="Additional config specs for the probe run."),
) -> None:
    summary = run_probe(
        workspace_dir=workspace_dir,
        objective=objective,
        probe_name=probe_name,
        start_url=start_url,
        max_steps=max_steps,
        config_spec=[*DEFAULT_PROBE_CONFIGS, *config_spec],
    )
    console.print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    app()

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from miniswewebagent.agents import get_agent
from miniswewebagent.config import get_config_from_spec, snapshot_config_specs
from miniswewebagent.environments import get_environment
from miniswewebagent.models import get_model
from miniswewebagent.tasks.om2w import load_om2w_task
from miniswewebagent.utils.om2w_eval import export_online_mind2web_artifacts
from miniswewebagent.utils.serialize import UNSET, recursive_merge

_SESSION_ID_MAX_LEN = 128
_SESSION_ID_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_session_id(value: str) -> str:
    cleaned = _SESSION_ID_ALLOWED.sub("_", value).strip("_")
    if len(cleaned) > _SESSION_ID_MAX_LEN:
        cleaned = cleaned[:_SESSION_ID_MAX_LEN].rstrip("_")
    return cleaned


def _apply_auto_model_overrides(
    config: dict[str, Any],
    auto_model_overrides: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Set phyagi-only model keys when YAML did not pin them. Returns applied dict for logging."""
    if not auto_model_overrides:
        return None
    model_cfg = config.setdefault("model", {})
    # Treat missing model_class as phyagi (matches get_model default at models/__init__.py:24).
    model_class = str(model_cfg.get("model_class") or "phyagi")
    if model_class != "phyagi":
        return None
    applied: dict[str, Any] = {}
    for key, value in auto_model_overrides.items():
        if value in (None, ""):
            continue
        if model_cfg.get(key):
            continue
        if key == "session_id":
            value = _sanitize_session_id(str(value))
            if not value:
                continue
        model_cfg[key] = value
        applied[key] = value
    return applied or None


def _load_final_script(scripts_dir: Path, task_id: str) -> str:
    """Load final_script.py from a previous run for a task."""
    script_path = scripts_dir / task_id / "final_script.py"
    if not script_path.exists():
        return ""
    return script_path.read_text(encoding="utf-8")


def _copy_task_cli_tool(
    cli_tool_dir: Path, task_id: str, workspace_dir: Path, dest_name: str = "task_cli_tool.py"
) -> str:
    """Copy `cli_tool_dir/<task_id>/final_script.py` into the workspace.

    Returns the destination path as a string (suitable for a template var) or
    "" when no source script is available.
    """
    if not task_id:
        return ""
    src = cli_tool_dir / task_id / "final_script.py"
    if not src.is_file():
        return ""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    dest = workspace_dir / dest_name
    shutil.copy2(src, dest)
    return str(dest)


def _copy_task_cli_tools_multi(
    cli_tool_dir: Path,
    by_website_file: Path,
    task_id: str,
    workspace_dir: Path,
) -> list[str]:
    """Copy every oracle CLI belonging to the same website as ``task_id``.

    Looks up ``task_id`` in the by-website index file (a list of records
    ``{"website": ..., "tasks": [{"task_id": ...}, ...]}``), gathers all
    sibling task_ids, and copies each one's
    ``cli_tool_dir/<sibling>/final_script.py`` into ``workspace_dir`` under an
    anonymized name (``cli_tool_1.py``, ``cli_tool_2.py``, ...). The order is
    deterministic (alphabetical by sibling task_id) but the filenames do NOT
    encode which CLI was authored for which task — the agent must inspect each
    one to pick the right tool.

    Returns the list of destination paths (absolute, as strings) in the
    assigned order; empty list when no siblings or sources are found.
    """
    if not task_id:
        return []
    if not by_website_file.is_file():
        return []
    records = json.loads(by_website_file.read_text(encoding="utf-8"))
    sibling_ids: list[str] | None = None
    for record in records:
        ids = [t.get("task_id", "") for t in record.get("tasks", [])]
        if task_id in ids:
            sibling_ids = sorted(i for i in ids if i)
            break
    if not sibling_ids:
        return []
    workspace_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx, sid in enumerate(sibling_ids, start=1):
        src = cli_tool_dir / sid / "final_script.py"
        if not src.is_file():
            continue
        dest = workspace_dir / f"cli_tool_{idx}.py"
        shutil.copy2(src, dest)
        paths.append(str(dest))
    return paths


def _load_command_usage_instruction(instruction_dir: Path, task_id: str) -> str:
    """Load the textual CLI usage instruction for a task.

    The instruction file lives at ``instruction_dir/<task_id>`` (no extension)
    and is expected to be a single-line ``python final_script.py --flag value
    ...`` invocation. The original ``final_script.py`` filename is rewritten to
    ``task_cli_tool.py`` so the instruction matches the file copied into the
    workspace.
    """
    if not task_id:
        return ""
    path = instruction_dir / task_id
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text.replace("final_script.py", "task_cli_tool.py")


def _load_explore_history(explore_dir: Path, task_id: str) -> str:
    """Load previous explore trajectory messages (text only) for a task."""
    traj_path = explore_dir / task_id / "trajectory.json"
    if not traj_path.exists():
        return ""
    traj = json.loads(traj_path.read_text(encoding="utf-8"))
    parts = []
    for msg in traj.get("messages", []):
        role = msg.get("role", "unknown")
        if role == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content if p.get("type") in ("text", "input_text")
            )
        if not content:
            continue
        parts.append(f"[{role}]\n{content}")
    return "\n\n---\n\n".join(parts)

DEFAULT_CONFIG = "mini.yaml"

app = typer.Typer(no_args_is_help=True)
console = Console(highlight=False)


def _timestamped_output_dir(base_dir: str | Path | None, task_id: str | None) -> Path:
    base = Path(base_dir or "outputs").expanduser()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = task_id or "adhoc"
    return base / f"{suffix}_{stamp}"


def run_one(
    *,
    task: str | None = None,
    task_id: str | None = None,
    tasks_file: Path | None = None,
    start_url: str | None = None,
    config_spec: list[str] | None = None,
    output_dir: Path | None = None,
    resolved_output_dir: Path | None = None,
    debug: bool = False,
    snapshot_config: bool = True,
    auto_model_overrides: dict[str, Any] | None = None,
) -> Any:
    config_spec = config_spec or [DEFAULT_CONFIG]
    configs = [get_config_from_spec(spec) for spec in config_spec]
    config = recursive_merge(*configs)

    run_config = config.get("run", {})
    resolved_tasks_file = tasks_file or run_config.get("tasks_file")
    resolved_task_id = task_id or run_config.get("task_id")
    resolved_task = task or run_config.get("task")
    resolved_start_url = start_url or run_config.get("start_url")

    task_record = None
    should_load_task_record = bool(resolved_task_id) and (not resolved_task or not resolved_start_url)
    if should_load_task_record:
        if not resolved_tasks_file:
            raise ValueError("--task-id requires --tasks-file unless --task and --start-url are both set.")
        task_record = load_om2w_task(resolved_tasks_file, resolved_task_id)
        resolved_task = resolved_task or task_record["task"]
        resolved_start_url = resolved_start_url or task_record["start_url"]

    if not resolved_task:
        raise ValueError("A task is required. Use --task or --task-id.")

    resolved_output_dir = resolved_output_dir or _timestamped_output_dir(
        output_dir or config.get("environment", {}).get("output_dir") or "outputs",
        resolved_task_id,
    )
    if snapshot_config:
        snapshot_config_specs(config_spec, resolved_output_dir, merged_config=config)

    config = recursive_merge(
        config,
        {
            "run": {
                "task": resolved_task,
                "task_id": resolved_task_id or UNSET,
                "start_url": resolved_start_url or UNSET,
                "tasks_file": str(resolved_tasks_file) if resolved_tasks_file else UNSET,
            },
            "environment": {
                "output_dir": str(resolved_output_dir),
                "start_url": resolved_start_url or UNSET,
                "headless": False if debug else UNSET,
                "devtools": True if debug else UNSET,
                "keep_open_on_exit": True if debug else UNSET,
                "prompt_before_close": True if debug else UNSET,
                "slow_mo_ms": 250 if debug else UNSET,
            },
            "model": {
                "error_log_path": str(resolved_output_dir / "runtime_errors.jsonl"),
            },
            "agent": {
                "output_path": str(resolved_output_dir / "trajectory.json"),
            },
        },
    )

    applied_auto = _apply_auto_model_overrides(config, auto_model_overrides)

    model = get_model(config.get("model", {}))
    env = get_environment(config.get("environment", {}))
    agent = get_agent(model, env, config.get("agent", {}), default_type="default")

    console.print(f"Running task in [bold green]{resolved_output_dir}[/bold green]")
    if applied_auto:
        console.print(f"Auto model overrides applied: {applied_auto}")
    run_exception: Exception | None = None
    close_exception: Exception | None = None
    result: dict[str, Any] = {}
    try:
        env.prepare(
            task=resolved_task,
            task_id=resolved_task_id,
            start_url=resolved_start_url,
            task_record=task_record,
        )
        cli_tool_path = (
            _copy_task_cli_tool(
                Path(run_config["cli_tool_dir"]),
                resolved_task_id or "",
                resolved_output_dir,
            )
            if run_config.get("cli_tool_dir") and resolved_task_id
            else ""
        )
        command_usage_instruction = (
            _load_command_usage_instruction(
                Path(run_config["cli_tool_instruction_dir"]), resolved_task_id or ""
            )
            if run_config.get("cli_tool_instruction_dir") and resolved_task_id
            else ""
        )
        available_cli_tools_list = (
            _copy_task_cli_tools_multi(
                Path(run_config["cli_tool_dir"]),
                Path(run_config["cli_tool_by_website_file"]),
                resolved_task_id or "",
                resolved_output_dir,
            )
            if run_config.get("cli_tool_by_website_file")
            and run_config.get("cli_tool_dir")
            and resolved_task_id
            else []
        )
        available_cli_tools = "\n".join(available_cli_tools_list)
        result = agent.run(
            resolved_task,
            task_id=resolved_task_id or "",
            start_url=resolved_start_url or "",
            explore_history=_load_explore_history(
                Path(run_config["explore_history_dir"]), resolved_task_id
            ) if run_config.get("explore_history_dir") and resolved_task_id else "",
            final_script=_load_final_script(
                Path(run_config["final_script_dir"]), resolved_task_id
            ) if run_config.get("final_script_dir") and resolved_task_id else "",
            task_cli_tool_path=cli_tool_path,
            command_usage_instruction=command_usage_instruction,
            available_cli_tools=available_cli_tools,
            available_cli_tools_list=available_cli_tools_list,
        )
    except Exception as exc:
        run_exception = exc
        if getattr(agent, "messages", None):
            result = dict(agent.messages[-1].get("extra", {}))
        result.setdefault("exit_status", type(exc).__name__)
        result.setdefault("submission", "")
        result.setdefault("final_response", "")
        result["run_exception"] = str(exc)
    finally:
        try:
            env.close()
        except Exception as exc:
            close_exception = exc
            result.setdefault("exit_status", type(exc).__name__)
            result.setdefault("submission", "")
            result.setdefault("final_response", "")
            result.setdefault("run_exception", str(exc))
            result["close_exception"] = str(exc)
            if run_exception is None:
                run_exception = exc
        # Release any cloud browser sessions recorded by open_browser_session,
        # in case an agent step aborted before reaching `await browser.close()`.
        try:
            from miniswewebagent.tools.browser_session import release_recorded_sessions

            released = release_recorded_sessions(resolved_output_dir / "browser_sessions.jsonl")
            if released:
                failures = [r for r in released if not r[1]]
                console.print(
                    f"Released {len(released) - len(failures)}/{len(released)} cloud browser session(s)."
                )
                for sid, _ok, err in failures:
                    console.print(f"[yellow]  session {sid} release failed: {err}[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Browser session cleanup skipped: {exc}[/yellow]")
    judge_artifacts = export_online_mind2web_artifacts(
        output_dir=resolved_output_dir,
        task=resolved_task,
        task_id=resolved_task_id,
        start_url=resolved_start_url,
        agent_result=result,
    )
    result["_output_dir"] = str(resolved_output_dir)
    result["_judge_artifacts"] = judge_artifacts
    if close_exception is not None:
        result["_close_exception"] = str(close_exception)
    console.print(result.get("final_response") or result.get("submission") or "Task finished.")
    if run_exception is not None:
        raise run_exception
    return result


@app.command()
def main(
    task: str | None = typer.Option(None, "-t", "--task"),
    task_id: str | None = typer.Option(None, "--task-id"),
    tasks_file: Path | None = typer.Option(None, "--tasks-file"),
    start_url: str | None = typer.Option(None, "--start-url"),
    config_spec: list[str] = typer.Option([DEFAULT_CONFIG], "-c", "--config"),
    output_dir: Path | None = typer.Option(None, "-o", "--output-dir"),
    debug: bool = typer.Option(False, "--debug", help="Launch headed local Playwright with devtools and keep it open for inspection at the end."),
) -> Any:
    return run_one(
        task=task,
        task_id=task_id,
        tasks_file=tasks_file,
        start_url=start_url,
        config_spec=config_spec,
        output_dir=output_dir,
        debug=debug,
    )

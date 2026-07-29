from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from miniswewebagent import Environment, Model, __version__
from miniswewebagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded
from miniswewebagent.utils.browser_evidence import (
    DEFAULT_BROWSER_STEPS_FILE,
    load_browser_steps,
    optional_file_digest,
    trajectory_evidence_digest,
)
from miniswewebagent.utils.serialize import recursive_merge


DEFAULT_SUMMARY_USER_PROMPT = """You are about to have your working context compacted to save tokens.

Write a concise but COMPLETE summary of everything relevant from the conversation above so that a fresh
agent with only this summary (plus the original system prompt and task instructions) can continue the
task without losing progress. Include:

- The original task goal and all critical points / constraints.
- The workspace directory and key file paths (plan.md, judge_config.json, final_script.py, final_runs/).
- Which critical points have been satisfied, which are still open, and any known blockers.
- Key findings from prior exploration (working selectors, URLs, ARIA labels, pitfalls to avoid).
- The latest final_runs/run_<id>/ state, most recent self_reflection verdict, and the next action to take.

Write the summary as plain prose and bullet lists. Do NOT issue a new bash_command. Do NOT set done=true.
Put the entire summary in the `thought` field (or equivalent text field) and leave action fields empty."""

_BAD_COMPACT_SUMMARIES = {
    "",
    "task complete.",
    "task complete",
    "done.",
    "done",
    "(empty summary)",
}


class AgentConfig(BaseModel):
    system_template: str
    instance_template: str
    render_system_template: bool = True
    step_limit: int = 15
    debug_log: bool = True
    attach_instance_template_after_observation: bool = False
    attach_plan_md_after_observation: bool = False
    require_self_reflection_success: bool = False
    judge_mode: str = "tool"
    judge_upstream_src: str = "/home/luyadong/sandbox/Online-Mind2Web/src"
    judge_model: str = "o4-mini"
    judge_gateway_endpoint: str = ""
    judge_score_threshold: int = 3
    trajectory_manifest: str = DEFAULT_BROWSER_STEPS_FILE
    trajectory_judge_result: str = "reflection/judge_result.json"
    trajectory_plan_file: str = "plan.md"
    trajectory_judge_config: str = "judge_config.json"
    summary_every_n_steps: int = 0
    summary_user_prompt: str = DEFAULT_SUMMARY_USER_PROMPT
    summary_max_output_tokens: int = 0
    summary_response_mode: str = ""
    # >0: query with a sliding window instead of the full history. The window
    # keeps system + the initial task message + the last N assistant turns
    # (each with the user block that precedes it). self.messages still records
    # the full history; only the model request is windowed.
    context_window_steps: int = 0
    # History-content transform for the model request (composable with the
    # window; self.messages keeps full history):
    #   "full"           - no transform
    #   "last_obs"       - older observations keep only the template head with
    #                      'Command output: (omitted)'; latest obs stays full
    #   "last_obs_think" - additionally, assistant turns before the last
    #                      completed step keep only their <think> block
    history_context_mode: str = "full"
    output_path: Path | None = None


def _sanitize_message_for_disk(message: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(message)
    content = cloned.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "input_image":
                part["image_url"] = "<omitted:data-url>"
    return cloned


def _observation_for_markdown(observation: dict[str, Any], *, model_usage: dict[str, Any] | None = None) -> dict[str, Any]:
    cloned = copy.deepcopy(observation)
    cloned.pop("aria_snapshot", None)
    if model_usage:
        cloned["model_usage"] = copy.deepcopy(model_usage)
    return cloned


def _message_content_for_markdown(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
                continue
            part_type = part.get("type")
            if part_type in {"text", "input_text"}:
                parts.append(str(part.get("text", "")))
            elif part_type in {"image_url", "input_image"}:
                parts.append("[image omitted]")
            else:
                parts.append(json.dumps(part, indent=2, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _action_text(action: dict[str, Any]) -> str:
    return str(action.get("bash_command") or action.get("command") or action.get("python_code") or "").strip()


def _python_action_text(action: dict[str, Any]) -> str:
    return str(action.get("python_code") or "").strip()


def _markdown_code_fence_language(*, bash_command_text: str, python_code_text: str) -> str:
    if bash_command_text:
        return "bash"
    if python_code_text:
        return "python"
    return ""


# ---------------------------------------------------------------------------
# om2w judge helpers (upstream WebJudge_Online_Mind2Web_eval, used when
# AgentConfig.judge_mode == "om2w"). Mirrors scripts/eval_with_original_om2w.py.
# ---------------------------------------------------------------------------

_OM2W_STEP_ACTION_RE = re.compile(r"^\s*step\s+\d+\s+action\s*:\s*.+\s*$", re.IGNORECASE)
_OM2W_SHOT_RE = re.compile(r"^final_execution_(\d+).*\.png$", re.IGNORECASE)


def _om2w_load_actions(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    raw = log_path.read_text(encoding="utf-8", errors="replace").replace("\\n", "\n")
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if s:
            out.append(s)
    return out


def _om2w_load_screenshots(shots_dir: Path) -> list[Path]:
    if not shots_dir.is_dir():
        return []
    keyed: list[tuple[int, Path]] = []
    fallback: list[Path] = []
    for p in shots_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".png":
            continue
        m = _OM2W_SHOT_RE.match(p.name)
        if m:
            keyed.append((int(m.group(1)), p))
        else:
            fallback.append(p)
    if keyed:
        keyed.sort(key=lambda t: (t[0], t[1].name))
        return [p for _, p in keyed]
    fallback.sort(key=lambda p: p.name)
    return fallback


def _om2w_read_cached_result(result_path: Path) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _om2w_run_eval(
    *,
    task: str,
    actions: list[str],
    screenshot_paths: list[Path],
    oracle_judge_model: str,
    judge_gateway_endpoint: str,
    score_threshold: int,
    upstream_src: str,
) -> tuple[Any, str, list[dict[str, Any]], str]:
    """Run upstream WebJudge_Online_Mind2Web_eval on one run's artifacts.

    Returns ``(predicted_label, response_text, image_judge_record, key_points)``.
    ``predicted_label`` is whatever ``extract_predication`` returns (typically 1,
    0, or None for unparseable).
    """
    upstream_path = Path(upstream_src)
    if upstream_path.is_dir() and str(upstream_path) not in sys.path:
        sys.path.insert(0, str(upstream_path))

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from methods import webjudge_online_mind2web as upstream_webjudge  # type: ignore[import]
    from om2w_judge.utils import OpenaiEngine, extract_predication  # type: ignore[import]

    model = OpenaiEngine(
        model=oracle_judge_model,
        endpoint_target_uri=judge_gateway_endpoint or "",
    )
    messages, _text, _system_msg, record, key_points = asyncio.run(
        upstream_webjudge.WebJudge_Online_Mind2Web_eval(
            task,
            list(actions),
            [str(p) for p in screenshot_paths],
            model,
            score_threshold,
        )
    )
    response_text = model.generate(messages, max_new_tokens=8192)[0]
    predicted_label = extract_predication(response_text, "WebJudge_Online_Mind2Web_eval")
    return predicted_label, response_text, list(record or []), str(key_points or "")


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict[str, Any]] = []
        self.compacted_sessions: list[list[dict[str, Any]]] = []
        self.model = model
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}
        self.n_calls = 0
        self.n_format_errors = 0

    def _debug_dir(self) -> Path | None:
        if self.config.output_path is None:
            return None
        return self.config.output_path.parent / "debug"

    def _write_debug_step_artifact(
        self,
        *,
        step_index: int,
        assistant_message: dict[str, Any],
        outputs: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.config.debug_log:
            return
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return
        steps_dir = debug_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

        extra = assistant_message.get("extra", {})
        actions = extra.get("actions", [])
        action_text = "\n\n".join(_action_text(action) for action in actions if _action_text(action))
        python_code_text = "\n\n".join(
            _python_action_text(action) for action in actions if _python_action_text(action)
        )
        bash_command_text = "\n\n".join(
            str(action.get("bash_command", "")).strip()
            for action in actions
            if str(action.get("bash_command", "")).strip()
        )
        code_fence_language = _markdown_code_fence_language(
            bash_command_text=bash_command_text,
            python_code_text=python_code_text,
        )
        payload = {
            "step": step_index,
            "thought": assistant_message.get("content", ""),
            "python_code": python_code_text,
            "bash_command": bash_command_text,
            "command_text": action_text,
            "raw_response": extra.get("raw_response", {}),
            "raw_text": extra.get("raw_text", ""),
            "done": extra.get("done", False),
            "final_response": extra.get("final_response", ""),
            "outputs": outputs or [],
        }
        (steps_dir / f"step_{step_index:04d}.json").write_text(json.dumps(payload, indent=2))

        summary_path = debug_dir / "steps.md"
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(f"## Step {step_index}\n\n")
            # Attach the model input only for the first step
            if step_index == 1:
                user_input_text = ""
                for msg in reversed(self.messages):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            # Multi-part message: join text parts
                            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
                            user_input_text = "\n".join(p for p in parts if p)
                        else:
                            user_input_text = str(content)
                        break
                if user_input_text:
                    handle.write("### Model Input\n\n")
                    handle.write(f"{user_input_text}\n\n")
            handle.write("### Thought\n\n")
            handle.write(f"{payload['thought']}\n\n")
            handle.write("### Generated Code\n\n")
            handle.write(f"```{code_fence_language}\n")
            handle.write(f"{payload['command_text']}\n")
            handle.write("```\n\n")
            if outputs:
                observation = outputs[0].get("observation", {})
                markdown_observation = _observation_for_markdown(
                    observation,
                    model_usage=extra.get("usage"),
                )
                handle.write("### Observation\n\n")
                handle.write("```json\n")
                handle.write(f"{json.dumps(markdown_observation, indent=2)}\n")
                handle.write("```\n\n")

    def _write_debug_request_artifact(self, *, step_index: int, messages: list[dict[str, Any]]) -> None:
        if not self.config.debug_log:
            return
        debug_dir = self._debug_dir()
        if debug_dir is None:
            return

        requests_dir = debug_dir / "requests"
        user_messages_dir = debug_dir / "user_messages"
        serialized_requests_dir = debug_dir / "serialized_requests"
        requests_dir.mkdir(parents=True, exist_ok=True)
        user_messages_dir.mkdir(parents=True, exist_ok=True)
        serialized_requests_dir.mkdir(parents=True, exist_ok=True)

        sanitized_messages = [_sanitize_message_for_disk(message) for message in messages]
        user_messages = [
            {
                "message_index": message_index,
                "content": message.get("content", ""),
            }
            for message_index, message in enumerate(sanitized_messages)
            if message.get("role") == "user"
        ]
        latest_user_message = user_messages[-1] if user_messages else None
        payload = {
            "step": step_index,
            "messages": sanitized_messages,
            "user_messages": user_messages,
            "latest_user_message": latest_user_message,
        }
        (requests_dir / f"request_{step_index:04d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        request_markdown_path = requests_dir / f"request_{step_index:04d}.md"
        with request_markdown_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Step {step_index} Model Request\n\n")
            for message_index, message in enumerate(sanitized_messages, start=1):
                role = str(message.get("role", ""))
                handle.write(f"## Message {message_index}: {role}\n\n")
                handle.write("```text\n")
                handle.write(_message_content_for_markdown(message.get("content", "")))
                handle.write("\n```\n\n")

        user_markdown_path = user_messages_dir / f"step_{step_index:04d}.md"
        with user_markdown_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Step {step_index} User Messages\n\n")
            if not user_messages:
                handle.write("No user messages in this request.\n")
            else:
                for user_message_index, message in enumerate(user_messages, start=1):
                    handle.write(f"## User Message {user_message_index} (request index {message['message_index']})\n\n")
                    handle.write("```text\n")
                    handle.write(_message_content_for_markdown(message.get("content", "")))
                    handle.write("\n```\n\n")

        serialize_request_for_debug = getattr(self.model, "serialize_request_for_debug", None)
        if not callable(serialize_request_for_debug):
            return
        try:
            serialized_request = serialize_request_for_debug(messages)
        except Exception as exc:  # noqa: BLE001 - debug logging must not break eval
            serialized_request = {"error": repr(exc)}

        (serialized_requests_dir / f"request_{step_index:04d}.json").write_text(
            json.dumps(serialized_request, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        serialized_markdown_path = serialized_requests_dir / f"request_{step_index:04d}.md"
        with serialized_markdown_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# Step {step_index} Serialized Model Request\n\n")
            if "error" in serialized_request:
                handle.write(f"Serialization error: {serialized_request['error']}\n")
                return
            for message_index, message in enumerate(serialized_request.get("messages", []), start=1):
                role = str(message.get("role", ""))
                handle.write(f"## Message {message_index}: {role}\n\n")
                handle.write("```text\n")
                handle.write(_message_content_for_markdown(message.get("content", "")))
                handle.write("\n```\n\n")

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def _system_prompt_content(self) -> str:
        """Return the system prompt at the configured template boundary."""
        if not self.config.render_system_template:
            return self.config.system_template
        return self._render_template(self.config.system_template)

    def _host_workspace_dir(self) -> Path | None:
        """Return the real filesystem workspace path, not the model-facing alias."""
        try:
            serialized = self.env.serialize()
        except Exception:  # noqa: BLE001 - fall back to template vars for nonstandard envs
            serialized = {}
        env_info = serialized.get("environment", {}) if isinstance(serialized, dict) else {}
        if isinstance(env_info, dict):
            workspace_dir = env_info.get("workspace_dir")
            if workspace_dir:
                return Path(str(workspace_dir))

        workspace_dir = self.get_template_vars().get("workspace_dir")
        if not workspace_dir:
            return None
        return Path(str(workspace_dir))

    def _agent_workspace_dir(self) -> str:
        workspace_dir = self.get_template_vars().get("workspace_dir")
        if workspace_dir:
            return str(workspace_dir)
        host_workspace_dir = self._host_workspace_dir()
        return str(host_workspace_dir) if host_workspace_dir is not None else ""

    def _agent_path(self, path: Path) -> str:
        host_workspace_dir = self._host_workspace_dir()
        agent_workspace_dir = self._agent_workspace_dir().rstrip("/")
        if host_workspace_dir is not None and agent_workspace_dir:
            try:
                relative_path = path.resolve().relative_to(host_workspace_dir.resolve())
            except (OSError, ValueError):
                pass
            else:
                return str(Path(agent_workspace_dir) / relative_path)
        return str(path)

    def _plan_md_message(self) -> dict[str, Any] | None:
        workspace_dir = self._host_workspace_dir()
        if workspace_dir is None:
            return None
        plan_path = workspace_dir / "plan.md"
        if not plan_path.exists() or not plan_path.is_file():
            return None
        plan_text = plan_path.read_text(encoding="utf-8").strip()
        if not plan_text:
            return None
        return self.model.format_message(role="user", content=f"Current plan.md:\n\n{plan_text}")

    def _self_reflection_gate_error(self) -> str | None:
        """Return an error string if done=true should be blocked pending judge success."""
        if not self.config.require_self_reflection_success:
            return None
        mode = (self.config.judge_mode or "tool").strip().lower()
        if mode == "om2w":
            return self._om2w_gate_error()
        if mode == "trajectory":
            return self._trajectory_gate_error()
        return self._tool_gate_error()

    def _trajectory_gate_error(self) -> str | None:
        """Require a fresh successful reflection over every incremental browser step."""
        # Resolve the REAL workspace path: with workspace_alias set, template
        # vars carry the alias (e.g. /workspace) which does not exist on host.
        workspace = self._host_workspace_dir()
        if workspace is None:
            workspace_value = self.get_template_vars().get("workspace_dir")
            if not workspace_value:
                return (
                    "Completion blocked: judge_mode=trajectory requires a workspace_dir. "
                    "Do not set done=true."
                )
            workspace = Path(workspace_value).resolve()
        rows = load_browser_steps(workspace, self.config.trajectory_manifest)
        if not rows:
            return (
                "Completion blocked: no incremental browser steps are recorded in "
                f"{workspace / self.config.trajectory_manifest}. Use the persistent browser-session "
                "CLI before setting done=true."
            )
        judge_path = Path(self.config.trajectory_judge_result)
        if not judge_path.is_absolute():
            judge_path = workspace / judge_path
        if not judge_path.is_file():
            return (
                f"Completion blocked: {judge_path} does not exist. Run self_reflection with "
                f"--scope trajectory --trajectory-manifest {self.config.trajectory_manifest} "
                f"--output {self.config.trajectory_judge_result}, then retry done=true only after "
                "predicted_label == 1."
            )
        try:
            judge_data = json.loads(judge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"Completion blocked: could not parse {judge_path}: {exc}. Re-run reflection."
        if judge_data.get("scope") != "trajectory":
            return (
                f"Completion blocked: {judge_path} is not a trajectory-scoped reflection. "
                "Re-run self_reflection with --scope trajectory."
            )
        predicted_label = judge_data.get("predicted_label")
        if predicted_label != 1:
            return (
                f"Completion blocked: trajectory reflection predicted_label={predicted_label!r}; "
                "inspect its feedback, continue with incremental browser steps, and reflect again."
            )
        current_digest = trajectory_evidence_digest(workspace, rows)
        if judge_data.get("evidence_digest") != current_digest:
            return (
                "Completion blocked: trajectory reflection is stale because browser steps or "
                "screenshots changed after it ran. Reflect over the current trajectory again."
            )
        covered = max((int(row.get("browser_step") or 0) for row in rows), default=0)
        if int(judge_data.get("covered_through_browser_step") or 0) != covered:
            return (
                f"Completion blocked: reflection covers browser step "
                f"{judge_data.get('covered_through_browser_step')}, but the trajectory now reaches "
                f"step {covered}. Reflect again."
            )
        plan_path = Path(self.config.trajectory_plan_file)
        if not plan_path.is_absolute():
            plan_path = workspace / plan_path
        if not plan_path.is_file():
            return f"Completion blocked: required trajectory plan does not exist at {plan_path}."
        if judge_data.get("plan_digest", "") != optional_file_digest(plan_path):
            return "Completion blocked: plan.md changed after reflection. Reflect again."
        judge_config_path = Path(self.config.trajectory_judge_config)
        if not judge_config_path.is_absolute():
            judge_config_path = workspace / judge_config_path
        if not judge_config_path.is_file():
            return (
                "Completion blocked: required trajectory judge config does not exist at "
                f"{judge_config_path}."
            )
        if judge_data.get("judge_config_digest", "") != optional_file_digest(judge_config_path):
            return "Completion blocked: judge_config.json changed after reflection. Reflect again."
        return None

    def _tool_gate_error(self) -> str | None:
        """Require final_runs/run_<latest>/judge_result.json with predicted_label == 1."""
        workspace_dir = self._host_workspace_dir()
        agent_workspace_dir = self._agent_workspace_dir()
        if workspace_dir is None:
            return (
                "Completion blocked: require_self_reflection_success is enabled but no workspace_dir is "
                "available. Cannot locate final_runs/run_<id>/judge_result.json. Do not set done=true."
            )
        final_runs_dir = workspace_dir / "final_runs"
        if not final_runs_dir.is_dir():
            return (
                "Completion blocked: no final_runs/ directory exists yet. You must run final_script.py "
                "in a final_runs/run_<id>/ folder and then run "
                "`python -m miniswewebagent.tools.self_reflection --config judge_config.json "
                "--workspace-dir \"{0}\" --output final_runs/run_<id>/judge_result.json` with "
                "predicted_label == 1 before setting done=true."
            ).format(agent_workspace_dir or str(workspace_dir))
        run_dirs: list[tuple[int, Path]] = []
        for entry in final_runs_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("run_"):
                continue
            suffix = entry.name[len("run_"):]
            try:
                run_id = int(suffix)
            except ValueError:
                continue
            run_dirs.append((run_id, entry))
        if not run_dirs:
            return (
                "Completion blocked: final_runs/ contains no run_<id>/ folders. Create "
                "final_runs/run_<id>/, execute final_script.py there, then run self_reflection and "
                "only set done=true after judge_result.json reports predicted_label == 1."
            )
        run_dirs.sort(key=lambda item: item[0])
        latest_run_id, latest_run_dir = run_dirs[-1]
        judge_path = latest_run_dir / "judge_result.json"
        judge_path_for_agent = self._agent_path(judge_path)
        if not judge_path.is_file():
            return (
                f"Completion blocked: {judge_path_for_agent} does not exist. Run "
                f"`python -m miniswewebagent.tools.self_reflection --config judge_config.json "
                f"--workspace-dir \"{agent_workspace_dir or str(workspace_dir)}\" "
                f"--output {judge_path_for_agent}` against the latest run "
                f"(run_{latest_run_id}) and only set done=true after it exits 0 with "
                f"predicted_label == 1."
            )
        try:
            judge_data = json.loads(judge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return (
                f"Completion blocked: could not parse {judge_path_for_agent}: {exc}. Re-run self_reflection "
                f"against run_{latest_run_id} and only set done=true after predicted_label == 1."
            )
        predicted_label = judge_data.get("predicted_label")
        if predicted_label != 1:
            return (
                f"Completion blocked: {judge_path_for_agent} has predicted_label={predicted_label!r} "
                f"(expected 1). Diagnose the failure from judge_result.json, fix final_script.py, "
                f"re-run it in a new final_runs/run_{latest_run_id + 1}/ folder, and re-run "
                f"self_reflection. Only set done=true after self_reflection exits 0 with "
                f"predicted_label == 1."
            )
        return None

    def _om2w_gate_error(self) -> str | None:
        """Run the upstream WebJudge_Online_Mind2Web_eval on the latest run and gate on it.

        Mirrors scripts/eval_with_original_om2w.py: reads last_actions from
        final_script_log.txt (matching ``step <N> action: ...``), screenshots from
        ``screenshots/final_execution_<N>*.png``, then calls the upstream judge.
        The verdict is cached in ``final_runs/run_<id>/om2w_judge_result.json``.
        """
        workspace_dir = self._host_workspace_dir()
        if workspace_dir is None:
            return (
                "Completion blocked: judge_mode=om2w but no workspace_dir available. "
                "Cannot locate final_runs/run_<id>/ artifacts."
            )
        final_runs_dir = workspace_dir / "final_runs"
        if not final_runs_dir.is_dir():
            return (
                "Completion blocked: no final_runs/ directory exists yet. Run final_script.py "
                "inside final_runs/run_<id>/ before setting done=true (judge_mode=om2w)."
            )
        run_dirs: list[tuple[int, Path]] = []
        for entry in final_runs_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("run_"):
                continue
            try:
                run_id = int(entry.name[len("run_"):])
            except ValueError:
                continue
            run_dirs.append((run_id, entry))
        if not run_dirs:
            return (
                "Completion blocked: final_runs/ contains no run_<id>/ folders. Create "
                "final_runs/run_<id>/, execute final_script.py there, then retry done=true."
            )
        run_dirs.sort(key=lambda item: item[0])
        latest_run_id, latest_run_dir = run_dirs[-1]

        result_path = latest_run_dir / "om2w_judge_result.json"
        cached = _om2w_read_cached_result(result_path)
        if cached is not None:
            predicted = cached.get("predicted_label")
            if predicted == 1:
                return None
            return (
                f"Completion blocked: om2w judge reported predicted_label={predicted!r} for "
                f"run_{latest_run_id}. See {self._agent_path(result_path)}. Diagnose the failure, fix "
                f"final_script.py, re-run it in final_runs/run_{latest_run_id + 1}/, and retry "
                f"done=true."
            )

        log_path = latest_run_dir / "final_script_log.txt"
        shots_dir = latest_run_dir / "screenshots"
        actions = _om2w_load_actions(log_path)
        screenshots = _om2w_load_screenshots(shots_dir)
        if not screenshots:
            return (
                f"Completion blocked: no screenshots found under {self._agent_path(shots_dir)}. "
                "final_script.py must save final_execution_<N>*.png screenshots before "
                "done=true with judge_mode=om2w."
            )

        task = str(
            self.extra_template_vars.get("task")
            or self.get_template_vars().get("task")
            or ""
        )
        if not task:
            return (
                "Completion blocked: om2w judge requires the task description but none was "
                "found in template vars."
            )

        try:
            predicted_label, response_text, record, key_points = _om2w_run_eval(
                task=task,
                actions=actions,
                screenshot_paths=screenshots,
                oracle_judge_model=self.config.judge_model,
                judge_gateway_endpoint=self.config.judge_gateway_endpoint,
                score_threshold=self.config.judge_score_threshold,
                upstream_src=self.config.judge_upstream_src,
            )
        except Exception as exc:  # noqa: BLE001 — surface error to the agent transcript
            return (
                f"Completion blocked: om2w judge failed to run ({exc!r}). Fix the underlying "
                "issue (missing credentials, import error, etc.) and retry done=true."
            )

        payload = {
            "predicted_label": predicted_label,
            "response": response_text,
            "key_points": key_points,
            "image_judge_record": record,
            "action_history": actions,
            "screenshots": [str(p) for p in screenshots],
            "oracle_judge_model": self.config.judge_model,
            "mode": "WebJudge_Online_Mind2Web_eval",
        }
        try:
            result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

        if predicted_label == 1:
            return None
        return (
            f"Completion blocked: om2w judge reported predicted_label={predicted_label!r} for "
            f"run_{latest_run_id}. See {self._agent_path(result_path)} for the full response and per-image "
            f"reasonings. Diagnose, fix final_script.py, re-run it in "
            f"final_runs/run_{latest_run_id + 1}/, and retry done=true."
        )

    def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
        self.messages.extend(messages)
        return list(messages)

    def _compact_history(self) -> None:
        """Summarize the running transcript via an LLM call and reset messages to [system, summary].

        Preserves the original system message. Replaces every non-system message with a single user
        message containing the summary. The summarization call is made with the current messages
        plus a user prompt instructing the model to produce a complete compact summary.
        """
        if not self.messages:
            return
        system_message = next((m for m in self.messages if m.get("role") == "system"), None)
        if system_message is None:
            return
        # Match the SFT data layout (make_web_agent_sequential_compact_tools_sft.py):
        # the summary prompt is joined onto the last observation user message with
        # "\n\n" rather than sent as a separate user message. Merge on a copy so a
        # failed compaction leaves self.messages untouched. Falls back to a
        # standalone user message when the last turn is not a user message (the
        # SFT builder does the same for steps without an observation).
        summary_messages = list(self.messages)
        merged = False
        last_message = summary_messages[-1]
        if last_message.get("role") == "user":
            merged_last = copy.deepcopy(last_message)
            content = merged_last.get("content")
            if isinstance(content, str) and content.strip():
                merged_last["content"] = f"{content.rstrip()}\n\n{self.config.summary_user_prompt}"
                merged = True
            elif isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = f"{str(part.get('text', '')).rstrip()}\n\n{self.config.summary_user_prompt}"
                        merged = True
                        break
            if merged:
                summary_messages[-1] = merged_last
        if not merged:
            summary_messages.append(
                self.model.format_message(
                    role="user",
                    content=self.config.summary_user_prompt,
                    extra={"interrupt_type": "HistoryCompactionRequest"},
                )
            )
        model_config = getattr(self.model, "config", None)
        old_max_output_tokens: Any = None
        old_response_mode: Any = None
        should_restore_max_output_tokens = False
        should_restore_response_mode = False
        if self.config.summary_max_output_tokens > 0 and model_config is not None and hasattr(model_config, "max_output_tokens"):
            old_max_output_tokens = getattr(model_config, "max_output_tokens")
            setattr(model_config, "max_output_tokens", self.config.summary_max_output_tokens)
            should_restore_max_output_tokens = True
        if self.config.summary_response_mode and model_config is not None and hasattr(model_config, "response_mode"):
            old_response_mode = getattr(model_config, "response_mode")
            setattr(model_config, "response_mode", self.config.summary_response_mode)
            should_restore_response_mode = True
        try:
            response = self.model.query(summary_messages)
        except FormatError as exc:
            # Strict sft_state parsing can reject a summary that was merely
            # truncated mid-<think> (no closing tags). Salvage the raw text and
            # let _extract_compaction_summary clean it — an imperfect summary
            # beats skipping compaction and letting the context grow unbounded.
            raw_text = ""
            for message in getattr(exc, "messages", None) or []:
                extra = message.get("extra") if isinstance(message, dict) else None
                if isinstance(extra, dict) and extra.get("model_response"):
                    raw_text = str(extra["model_response"])
                    break
            if not raw_text:
                return
            response = {"content": "", "extra": {"raw_text": raw_text}}
        except Exception:  # noqa: BLE001 - never fail the run due to compaction
            return
        finally:
            if should_restore_max_output_tokens:
                setattr(model_config, "max_output_tokens", old_max_output_tokens)
            if should_restore_response_mode:
                setattr(model_config, "response_mode", old_response_mode)
        # The SFT data wrapper says "compacted after step {end_call}" where
        # end_call is the last action call (10/20/...), i.e. NOT counting the
        # compaction call itself — capture it before incrementing n_calls.
        compacted_after = self.n_calls
        # Count the compaction LLM call as one step toward api_calls / step_limit.
        self.n_calls += 1
        summary_text = self._extract_compaction_summary(response)
        if not summary_text:
            # Compaction is an optimization. If the summarization call fails to
            # produce a meaningful summary, preserve the existing history rather
            # than replacing it with an empty or misleading placeholder.
            return
        original_task = str(self.extra_template_vars.get("task") or "").strip()
        summary_message = self.model.format_message(
            role="user",
            content=(
                "## Compacted History Summary\n"
                f"Original task: {original_task}\n"
                f"(context was compacted after step {compacted_after}; earlier turns have been replaced "
                "by the summary below)\n\n"
                f"{summary_text}\n\n## End of Compacted Summary"
            ),
            extra={"interrupt_type": "HistoryCompactionSummary"},
        )
        # Archive the full pre-compaction session (persistent-browser repo layout:
        # chronological sessions = compacted_sessions + [messages]).
        self.compacted_sessions.append(copy.deepcopy(self.messages))
        self.messages = [system_message, summary_message]

    def _extract_compaction_summary(self, response: dict[str, Any]) -> str:
        extra = response.get("extra", {}) if isinstance(response.get("extra"), dict) else {}
        raw_response = extra.get("raw_response", {}) if isinstance(extra.get("raw_response"), dict) else {}
        candidates = [
            str(extra.get("final_response") or ""),
            str(response.get("content") or ""),
            str(raw_response.get("final_response") or ""),
            str(raw_response.get("thought") or ""),
            str(extra.get("raw_text") or ""),
        ]
        for candidate in candidates:
            summary = self._clean_compaction_summary(candidate)
            if summary:
                return summary
        return ""

    def _clean_compaction_summary(self, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""

        json_summary = self._summary_from_json_payload(text)
        if json_summary is not None:
            return json_summary

        # Prefer an explicit <answer> block only when it is substantive. SFT
        # compact calls sometimes emit <answer>Task complete.</answer> and put
        # the real continuation summary in <think>.
        answer_values = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        for answer in reversed(answer_values):
            cleaned = self._strip_markup_summary(answer)
            if self._is_valid_compaction_summary(cleaned):
                return cleaned

        final_response_values = re.findall(r"<final_response>(.*?)</final_response>", text, flags=re.DOTALL | re.IGNORECASE)
        for final_response in reversed(final_response_values):
            cleaned = self._strip_markup_summary(final_response)
            if self._is_valid_compaction_summary(cleaned):
                return cleaned

        think_values = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
        for thought in reversed(think_values):
            cleaned = self._strip_markup_summary(thought)
            if self._is_valid_compaction_summary(cleaned):
                return cleaned

        cleaned = self._strip_markup_summary(text)
        return cleaned if self._is_valid_compaction_summary(cleaned) else ""

    def _summary_from_json_payload(self, value: str) -> str | None:
        text = str(value or "").strip()
        candidates = [text]
        if not text.startswith("{"):
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                candidates.append(match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            summary = str(
                payload.get("thought")
                or payload.get("summary")
                or payload.get("final_response")
                or ""
            ).strip()
            cleaned = self._strip_markup_summary(summary)
            if self._is_valid_compaction_summary(cleaned):
                return cleaned
            return ""
        return None

    def _strip_markup_summary(self, value: str) -> str:
        text = str(value or "").strip()
        text = re.sub(r"<bash\b[^>]*>.*?</bash>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"<python_code\b[^>]*>.*?</python_code>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"</?(?:think|answer|bash|done|final_response|python_code)>", "", text, flags=re.IGNORECASE).strip()
        return text

    def _is_valid_compaction_summary(self, value: str) -> bool:
        normalized = " ".join(str(value or "").strip().lower().split())
        if normalized in _BAD_COMPACT_SUMMARIES:
            return False
        return len(normalized) >= 40

    def run(self, task: str = "", **kwargs) -> dict[str, Any]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.n_calls = 0
        self.n_format_errors = 0
        self.add_messages(
            self.model.format_message(role="system", content=self._system_prompt_content()),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        if self.extra_template_vars.get("explore_history"):
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content="## Previous Explore History\n"
                    "Below is the message log from a prior live-browser exploration of this exact task.\n"
                    "Use it to understand the site layout, available controls, aria snapshots, and pitfalls.\n"
                    "Do NOT repeat failed approaches. Build on what was learned.\n\n"
                    + self.extra_template_vars["explore_history"]
                    + "\n\n## End of Explore History",
                ),
            )

        while True:
            try:
                self.step()
            except InterruptAgentFlow as exc:
                if isinstance(exc, FormatError):
                    self.n_format_errors += 1
                self.add_messages(*exc.messages)
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
            if (
                self.config.summary_every_n_steps > 0
                and self.n_calls > 0
                and self.n_calls % self.config.summary_every_n_steps == 0
            ):
                self._compact_history()
                self.save(self.config.output_path)
        return self.messages[-1].get("extra", {})

    def _windowed_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sliding-window view of the history for the model request.

        With context_window_steps=N > 0: keep messages[0] (system) and
        messages[1] (initial task), drop everything between them and the user
        block preceding the N-th assistant message from the end. The window
        therefore holds the last N assistant turns plus their surrounding user
        messages (including the current pending observation). Short histories
        are returned unchanged.
        """
        n_steps = self.config.context_window_steps
        if n_steps <= 0:
            return messages
        assistant_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        if len(assistant_idx) <= n_steps:
            return messages
        cut = assistant_idx[-n_steps]
        while cut > 2 and messages[cut - 1].get("role") == "user":
            cut -= 1
        if cut <= 2:
            return messages
        # Merge the task message with the user block at the window start into a
        # single user message ("\n\n"-joined) so the request has no consecutive
        # user turns — matching how the SFT data merges consecutive human turns.
        block_end = cut
        while block_end < len(messages) and messages[block_end].get("role") == "user":
            block_end += 1
        parts = [messages[1].get("content")] + [m.get("content") for m in messages[cut:block_end]]
        if all(isinstance(p, str) for p in parts):
            merged = dict(messages[1])
            merged["content"] = "\n\n".join(p.strip() for p in parts)
            return [messages[0], merged] + messages[block_end:]
        return messages[:2] + messages[cut:]

    _THINK_BLOCK_RE = re.compile(r"<think>\n.*?\n</think>", re.S)

    def _transform_history(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply history_context_mode to the model request (mirrors the
        lastobs / lastobs_think SFT bundle construction byte-for-byte)."""
        mode = self.config.history_context_mode
        if mode not in ("last_obs", "last_obs_think"):
            return messages
        assistant_idx = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        # Everything after the last assistant is the current observation block
        # (matches the SFT bundles, where format-error retries are merged into
        # the current human turn and kept in full).
        current_block_start = (assistant_idx[-1] + 1) if assistant_idx else len(messages)
        def _map_text(m: dict[str, Any], fn) -> dict[str, Any]:
            """Apply fn to the message text, supporting str and parts-list content."""
            content = m.get("content")
            if isinstance(content, str):
                new = fn(content)
                if new == content:
                    return m
                m = dict(m)
                m["content"] = new
                return m
            if isinstance(content, list):
                parts = []
                changed = False
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text") \
                            and isinstance(part.get("text"), str):
                        new = fn(part["text"])
                        if new != part["text"]:
                            part = {**part, "text": new}
                            changed = True
                    parts.append(part)
                if not changed:
                    return m
                m = dict(m)
                m["content"] = parts
                return m
            return m

        def _stub(text: str) -> str:
            if "Command output:\n" in text:
                return text.split("Command output:\n", 1)[0] + "Command output: (omitted)"
            return text

        def _think(text: str) -> str:
            match = self._THINK_BLOCK_RE.search(text)
            return match.group(0) if match else text

        out = []
        for i, m in enumerate(messages):
            role = m.get("role")
            if role == "user" and i > 1 and i < current_block_start:
                m = _map_text(m, _stub)
            elif mode == "last_obs_think" and role == "assistant" \
                    and assistant_idx and i != assistant_idx[-1]:
                m = _map_text(m, _think)
            out.append(m)
        return out

    def step(self) -> list[dict[str, Any]]:
        return self.execute_actions(self.query())

    def query(self) -> dict[str, Any]:
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                self.model.format_message(
                    role="exit",
                    content="Step limit exceeded.",
                    extra={"exit_status": "LimitsExceeded", "submission": ""},
                )
            )
        step_index = self.n_calls + 1
        query_messages = self._transform_history(self._windowed_messages(self.messages))
        self._write_debug_request_artifact(step_index=step_index, messages=query_messages)
        message = self.model.query(query_messages)
        self.n_calls += 1
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        extra = message.get("extra", {})
        if extra.get("done"):
            gate_error = self._self_reflection_gate_error()
            if gate_error is not None:
                extra["done"] = False
                return self.add_messages(
                    self.model.format_message(
                        role="user",
                        content=gate_error,
                        extra={"interrupt_type": "SelfReflectionGate"},
                    )
                )
            self._write_debug_step_artifact(step_index=self.n_calls, assistant_message=message, outputs=[])
            return self.add_messages(
                self.model.format_message(
                    role="exit",
                    content=extra.get("final_response", "Task completed."),
                    extra={
                        "exit_status": "Submitted",
                        "submission": extra.get("final_response", ""),
                        "final_response": extra.get("final_response", ""),
                    },
                )
            )
        outputs = [self.env.execute(action) for action in extra.get("actions", [])]
        self._write_debug_step_artifact(step_index=self.n_calls, assistant_message=message, outputs=outputs)
        observation_messages = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        if self.config.attach_instance_template_after_observation:
            observation_messages.append(
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template))
            )
        if self.config.attach_plan_md_after_observation:
            plan_message = self._plan_md_message()
            if plan_message is not None:
                observation_messages.append(plan_message)
        return self.add_messages(*observation_messages)

    def serialize(self, *extra_dicts) -> dict[str, Any]:
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        return recursive_merge(
            {
                "info": {
                    "config": {
                        "agent": self.config.model_dump(mode="json"),
                        "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    },
                    "mini_version": __version__,
                    "exit_status": last_extra.get("exit_status", ""),
                    "submission": last_extra.get("submission", ""),
                    "api_calls": self.n_calls,
                    "format_errors": self.n_format_errors,
                },
                "messages": [_sanitize_message_for_disk(message) for message in self.messages],
                "compacted_sessions": [
                    [_sanitize_message_for_disk(message) for message in session]
                    for session in self.compacted_sessions
                ],
                "trajectory_format": "mini-swe-webagent-0.1",
            },
            self.model.serialize(),
            self.env.serialize(),
            *extra_dicts,
        )

    def save(self, path: Path | None, *extra_dicts) -> dict[str, Any]:
        data = self.serialize(*extra_dicts)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data

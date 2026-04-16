"""Iterative agent runner with context reset between rounds.

Each round:
  1. Runs a fresh DefaultAgent (clean message history).
  2. After done=true, enforces a completion gate (final_runs/run_N/ must exist).
  3. Invokes WebJudge_Online_Mind2Web_Sandbox_eval on the round's artifacts.
  4. If the judge returns success → stop.
  5. If failure → start a new round with a fresh instance prompt that hard-attaches
     5 artifacts from the previous round (final_script, command_output, screenshots,
     judge response, round_dir path).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined
from jinja2 import Template as JinjaTemplate

from miniswewebagent.agents.default import DefaultAgent
from miniswewebagent.exceptions import FormatError, InterruptAgentFlow
from miniswewebagent.models.phyagi_model import image_part_from_path, text_part

_FINAL_RUN_DIR_RE = re.compile(r"^run_(\d+)$", re.IGNORECASE)
_FINAL_EXECUTION_RE = re.compile(r"final_execution_.*\.png$", re.IGNORECASE)
_FINAL_SCRIPT_ACTION_RE = re.compile(r"^\s*step\s+\d+(?:\s+action)?\s*:\s*.+\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    success: bool
    response_text: str
    record: list = field(default_factory=list)
    key_points: str = ""


@dataclass
class RoundArtifacts:
    round_id: int
    round_dir: Path
    final_script: str
    command_output: str
    screenshot_paths: list[Path]
    judge_result: JudgeResult | None = None


# ---------------------------------------------------------------------------
# Inner agent subclass
# ---------------------------------------------------------------------------

class _IterativeInnerAgent(DefaultAgent):
    """DefaultAgent that accepts a pre-built instance message, bypassing template rendering."""

    def run(
        self,
        task: str = "",
        *,
        _override_instance_message: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.n_calls = 0
        self.n_format_errors = 0

        system_msg = self.model.format_message(
            role="system",
            content=self._render_template(self.config.system_template),
        )
        if _override_instance_message is not None:
            inst_msg = _override_instance_message
        else:
            inst_msg = self.model.format_message(
                role="user",
                content=self._render_template(self.config.instance_template),
            )
        self.add_messages(system_msg, inst_msg)

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

        return self.messages[-1].get("extra", {})


# ---------------------------------------------------------------------------
# Iterative runner
# ---------------------------------------------------------------------------

class IterativeRunner:
    """Outer loop that runs DefaultAgent rounds with context reset between them.

    Config keys (all under the ``iterative`` section in YAML):
      max_rounds            int   Maximum number of rounds (default 5).
      judge_mode            str   "om2w" (external judge) or "workspace" (agent-created judge.py).
      judge_model           str   Model name for the om2w judge (e.g. "gpt-4o").
      judge_gateway_endpoint str  OpenAI-compatible gateway; uses env var if empty.
      judge_score_threshold int   Image relevance threshold for the om2w judge (default 3).
      round1_instance_template      str  Jinja2 template for round 1.
      next_round_instance_template  str  Jinja2 template for rounds > 1 (text parts only;
                                         screenshot images are appended automatically).
    """

    def __init__(
        self,
        model,
        env,
        *,
        agent_config: dict[str, Any],
        iterative_config: dict[str, Any],
    ) -> None:
        self.model = model
        self.env = env
        self.agent_config = agent_config

        ic = iterative_config
        self.max_rounds: int = int(ic.get("max_rounds", 5))
        self.judge_mode: str = str(ic.get("judge_mode", "om2w"))
        self.judge_model: str = str(ic.get("judge_model", "o4-mini"))
        self.judge_gateway_endpoint: str = str(ic.get("judge_gateway_endpoint", ""))
        self.judge_score_threshold: int = int(ic.get("judge_score_threshold", 3))
        self.round1_instance_template: str = str(ic.get("round1_instance_template", ""))
        self.next_round_instance_template: str = str(ic.get("next_round_instance_template", ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        *,
        task_id: str = "",
        start_url: str = "",
        base_output_dir: Path | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the iterative loop and return the final agent result dict."""
        prev_artifacts: RoundArtifacts | None = None
        last_result: dict[str, Any] = {}

        for round_id in range(1, self.max_rounds + 1):
            print(f"\n[IterativeRunner] ===== Round {round_id}/{self.max_rounds} =====")

            round_output_dir = (base_output_dir / f"round_{round_id}") if base_output_dir else None

            # Build per-round template variables — start with env + model vars so
            # that workspace_dir, task_metadata_path, final_script_path etc. are
            # available to the Jinja2 templates.
            try:
                env_vars = self.env.get_template_vars()
            except Exception:
                env_vars = {}
            try:
                model_vars = self.model.get_template_vars()
            except Exception:
                model_vars = {}

            template_vars: dict[str, Any] = {
                **env_vars,
                **model_vars,
                "task": task,
                "task_id": task_id,
                "start_url": start_url,
                "round_id": round_id,
                **kwargs,
            }
            if prev_artifacts is not None:
                template_vars.update({
                    "prev_round_id": prev_artifacts.round_id,
                    "prev_round_dir": str(prev_artifacts.round_dir),
                    "prev_final_script": prev_artifacts.final_script,
                    "prev_command_output": prev_artifacts.command_output,
                    "prev_judge_response": (
                        prev_artifacts.judge_result.response_text
                        if prev_artifacts.judge_result else ""
                    ),
                })

            # Build the instance message.
            # Use next_round_instance_template only when we actually have artifacts from a
            # previous round to attach; otherwise fall back to round1_instance_template.
            template_str = (
                self.next_round_instance_template
                if prev_artifacts is not None
                else self.round1_instance_template
            )
            rendered_text = JinjaTemplate(template_str, undefined=StrictUndefined).render(**template_vars)

            if prev_artifacts is not None and prev_artifacts.screenshot_paths:
                content: list[dict[str, Any]] = [text_part(rendered_text)]
                for sc_path in prev_artifacts.screenshot_paths:
                    if sc_path.exists():
                        content.append(image_part_from_path(sc_path))
                instance_msg = self.model.format_message(role="user", content=content)
            else:
                instance_msg = self.model.format_message(role="user", content=rendered_text)

            # Create a fresh agent and run one round
            agent_cfg = dict(self.agent_config)
            if round_output_dir:
                agent_cfg["output_path"] = str(round_output_dir / "trajectory.json")

            agent = _IterativeInnerAgent(self.model, self.env, **agent_cfg)
            agent.extra_template_vars = dict(template_vars)

            last_result = agent.run(
                task,
                task_id=task_id,
                start_url=start_url,
                round_id=round_id,
                **kwargs,
                _override_instance_message=instance_msg,
            )

            # Collect artifacts from the round (completion gate)
            workspace_dir = self._get_workspace_dir()
            round_artifacts = _collect_round_artifacts(
                round_id=round_id,
                workspace_dir=workspace_dir,
                agent_output_dir=round_output_dir or (
                    Path(agent_cfg["output_path"]).parent if agent_cfg.get("output_path") else None
                ),
                agent=agent,
            )

            if round_artifacts is None:
                print(
                    f"[IterativeRunner] Round {round_id}: completion gate failed — "
                    "no valid final_runs/run_N/ directory found. Skipping judge."
                )
                prev_artifacts = None
                continue

            # Run judge
            print(f"[IterativeRunner] Running judge ({self.judge_mode}) on: {round_artifacts.round_dir}")
            if self.judge_mode == "workspace":
                judge_result = _run_workspace_judge(
                    task=task,
                    artifacts=round_artifacts,
                    workspace_dir=workspace_dir,
                )
            else:
                judge_result = _run_judge(
                    task=task,
                    artifacts=round_artifacts,
                    judge_model=self.judge_model,
                    judge_gateway_endpoint=self.judge_gateway_endpoint,
                    judge_score_threshold=self.judge_score_threshold,
                )
            round_artifacts.judge_result = judge_result

            # Persist judge result next to the round folder
            _save_judge_result(round_artifacts)

            verdict = "SUCCESS" if judge_result.success else "FAILURE"
            print(f"[IterativeRunner] Round {round_id} judge verdict: {verdict}")

            if judge_result.success:
                last_result["_iterative_success"] = True
                last_result["_iterative_rounds"] = round_id
                last_result["_final_round_dir"] = str(round_artifacts.round_dir)
                return last_result

            prev_artifacts = round_artifacts

        # Exhausted all rounds
        last_result["_iterative_success"] = False
        last_result["_iterative_rounds"] = self.max_rounds
        if prev_artifacts is not None:
            last_result["_final_round_dir"] = str(prev_artifacts.round_dir)
        return last_result

    def _get_workspace_dir(self) -> Path | None:
        try:
            ws = self.env.get_template_vars().get("workspace_dir", "")
            return Path(ws) if ws else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Artifact collection (completion gate)
# ---------------------------------------------------------------------------

def _collect_round_artifacts(
    *,
    round_id: int,
    workspace_dir: Path | None,
    agent_output_dir: Path | None,
    agent: _IterativeInnerAgent,
) -> RoundArtifacts | None:
    """Locate the latest final_runs/run_N/ dir and populate a RoundArtifacts object.

    Also copies final_script.py into the round dir and saves command_output.txt if
    those files are not already present.
    """
    if workspace_dir is None:
        return None

    round_dir = _resolve_latest_final_run_dir(workspace_dir)
    if round_dir is None:
        return None

    # --- final_script.py ---
    final_script_src = workspace_dir / "final_script.py"
    final_script_dst = round_dir / "final_script.py"
    if final_script_src.exists() and not final_script_dst.exists():
        shutil.copy2(final_script_src, final_script_dst)
    final_script = ""
    if final_script_dst.exists():
        final_script = final_script_dst.read_text(encoding="utf-8")
    elif final_script_src.exists():
        final_script = final_script_src.read_text(encoding="utf-8")

    # --- command_output.txt ---
    cmd_output_path = round_dir / "command_output.txt"
    if not cmd_output_path.exists():
        command_output = _extract_command_output_from_debug_steps(agent_output_dir, agent)
        if command_output:
            cmd_output_path.write_text(command_output, encoding="utf-8")
    else:
        command_output = cmd_output_path.read_text(encoding="utf-8")

    # --- screenshots ---
    # Primary: final_execution_*.png inside the run dir's screenshots/ folder.
    # Fallback (when the run dir has no screenshots): use the most-recent
    # workspace-level step screenshots captured during this round.
    screenshots_dir = round_dir / "screenshots"
    screenshot_paths: list[Path] = []
    if screenshots_dir.is_dir():
        screenshot_paths = sorted(
            [p for p in screenshots_dir.iterdir() if _FINAL_EXECUTION_RE.match(p.name)],
            key=lambda p: _final_execution_sort_key(p.name),
        )

    if not screenshot_paths and workspace_dir is not None:
        screenshot_paths = _fallback_workspace_screenshots(workspace_dir)

    return RoundArtifacts(
        round_id=round_id,
        round_dir=round_dir,
        final_script=final_script,
        command_output=command_output,
        screenshot_paths=screenshot_paths,
    )


def _extract_command_output_from_debug_steps(
    agent_output_dir: Path | None,
    agent: _IterativeInnerAgent,
) -> str:
    """Search debug step JSON files for the stdout/stderr of the final_script.py run."""
    steps_dir: Path | None = None

    if agent_output_dir:
        candidate = agent_output_dir / "debug" / "steps"
        if candidate.exists():
            steps_dir = candidate

    if steps_dir is None and agent.config.output_path:
        candidate = Path(agent.config.output_path).parent / "debug" / "steps"
        if candidate.exists():
            steps_dir = candidate

    if steps_dir is None:
        return ""

    steps: list[dict[str, Any]] = []
    for path in sorted(steps_dir.glob("step_*.json")):
        try:
            steps.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

    # Return command_output from the last step that ran something python/final_script related
    for step in reversed(steps):
        bash_cmd = str(step.get("bash_command", "")).lower()
        if "final_script" in bash_cmd or ("python" in bash_cmd and "final" in bash_cmd):
            for out in step.get("outputs", []):
                obs = out.get("observation", {})
                cmd_output = obs.get("command_output", "")
                if cmd_output:
                    return str(cmd_output)

    # Fallback: last step with any command_output
    for step in reversed(steps):
        for out in step.get("outputs", []):
            obs = out.get("observation", {})
            cmd_output = obs.get("command_output", "")
            if cmd_output:
                return str(cmd_output)

    return ""


def _fallback_workspace_screenshots(workspace_dir: Path, max_screenshots: int = 8) -> list[Path]:
    """Return the most-recent step screenshots from the workspace screenshots/ dir.

    Used when a final_runs/run_N/screenshots/ dir has no final_execution_*.png files
    (e.g., the agent hit step-limit before creating a proper final run).  The
    workspace-level screenshots are per-step captures of the agent's last actions
    and give the next round visual context about what was tried.
    """
    ws_screenshots_dir = workspace_dir / "screenshots"
    if not ws_screenshots_dir.is_dir():
        return []

    # Collect step_NNNN.png files and sort numerically
    _STEP_RE = re.compile(r"step_(\d+)\.png$", re.IGNORECASE)
    candidates: list[tuple[int, Path]] = []
    for p in ws_screenshots_dir.iterdir():
        m = _STEP_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        # Fallback to any PNG sorted by name
        candidates = [(0, p) for p in ws_screenshots_dir.iterdir() if p.suffix.lower() == ".png"]

    candidates.sort(key=lambda t: t[0])
    # Return the last max_screenshots to keep context window manageable
    return [p for _, p in candidates[-max_screenshots:]]


def _save_judge_result(artifacts: RoundArtifacts) -> None:
    """Write judge_result.json next to the final_runs/run_N/ folder."""
    if artifacts.judge_result is None:
        return
    payload = {
        "success": artifacts.judge_result.success,
        "response_text": artifacts.judge_result.response_text,
        "key_points": artifacts.judge_result.key_points,
        "record": artifacts.judge_result.record,
    }
    out_path = artifacts.round_dir / "judge_result.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Judge invocation
# ---------------------------------------------------------------------------

_JUDGE_VERDICT_RE = re.compile(r"JUDGE VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)


def _run_workspace_judge(
    *,
    task: str,
    artifacts: RoundArtifacts,
    workspace_dir: Path | None,
) -> JudgeResult:
    """Call the agent-created judge.py in the workspace directory."""
    if workspace_dir is None:
        return JudgeResult(success=False, response_text="[Workspace judge: workspace_dir is None]")

    judge_path = workspace_dir / "judge.py"
    plan_path = workspace_dir / "plan.md"

    if not judge_path.exists():
        return JudgeResult(success=False, response_text="[Workspace judge: judge.py not found]")
    if not plan_path.exists():
        return JudgeResult(success=False, response_text="[Workspace judge: plan.md not found]")

    cmd = [
        sys.executable, str(judge_path),
        "--task", task,
        "--plan", str(plan_path),
        "--run-dir", str(artifacts.round_dir),
    ]

    try:
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=180,
            cwd=str(workspace_dir),
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired as exc:
        output = (getattr(exc, "stdout", "") or "") + (getattr(exc, "stderr", "") or "")
        return JudgeResult(success=False, response_text=f"[Workspace judge timed out]\n{output}")
    except Exception as exc:
        return JudgeResult(success=False, response_text=f"[Workspace judge failed: {exc}]")

    match = _JUDGE_VERDICT_RE.search(output)
    success = match is not None and match.group(1).upper() == "PASS"

    return JudgeResult(
        success=success,
        response_text=output.strip(),
    )


def _run_judge(
    *,
    task: str,
    artifacts: RoundArtifacts,
    judge_model: str,
    judge_gateway_endpoint: str,
    judge_score_threshold: int,
) -> JudgeResult:
    """Call WebJudge_Online_Mind2Web_Sandbox_eval on the round's artifacts."""
    # Ensure om2w_judge is importable (it lives next to mini-web-agent source)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from om2w_judge.methods.webjudge_online_mind2web_sandbox import (  # type: ignore[import]
            WebJudge_Online_Mind2Web_Sandbox_eval,
        )
        from om2w_judge.utils import OpenaiEngine, extract_predication  # type: ignore[import]
    except ImportError as exc:
        return JudgeResult(success=False, response_text=f"[Judge import failed: {exc}]")

    judge_llm = OpenaiEngine(
        model=judge_model,
        endpoint_target_uri=judge_gateway_endpoint or "",
    )

    # Action history from final_script_log.txt
    log_path = artifacts.round_dir / "final_script_log.txt"
    last_actions: list[str] = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if _FINAL_SCRIPT_ACTION_RE.match(line.strip()):
                last_actions.append(line.strip())

    screenshot_paths_str = [str(p) for p in artifacts.screenshot_paths]

    try:
        messages, _text, _sys, record, key_points = asyncio.run(
            WebJudge_Online_Mind2Web_Sandbox_eval(
                task,
                None,  # thoughts — not used by this judge variant
                last_actions,
                screenshot_paths_str,
                judge_llm,
                judge_score_threshold,
            )
        )
    except Exception as exc:
        return JudgeResult(success=False, response_text=f"[Judge eval failed: {exc}]")

    try:
        response_text = judge_llm.generate(messages, max_new_tokens=8192)[0]
    except Exception as exc:
        return JudgeResult(success=False, response_text=f"[Judge generate failed: {exc}]")

    predicted = extract_predication(response_text, "WebJudge_Online_Mind2Web_Sandbox_eval")
    return JudgeResult(
        success=bool(predicted),
        response_text=response_text,
        record=record,
        key_points=key_points,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_latest_final_run_dir(workspace_dir: Path) -> Path | None:
    final_runs_dir = workspace_dir / "final_runs"
    if not final_runs_dir.is_dir():
        return None

    candidates: list[tuple[int, str, Path]] = []
    for path in final_runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = _FINAL_RUN_DIR_RE.fullmatch(path.name)
        if not match:
            continue
        log_path = path / "final_script_log.txt"
        screenshots_dir = path / "screenshots"
        if not log_path.exists() and not screenshots_dir.is_dir():
            continue
        candidates.append((int(match.group(1)), path.name, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _final_execution_sort_key(filename: str) -> tuple[int, int, str]:
    match = re.search(r"final_execution_(\d+)", filename)
    if match:
        return (0, int(match.group(1)), filename)
    return (1, 0, filename)

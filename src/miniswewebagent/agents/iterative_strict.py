"""Strict iterative agent runner.

Difference from ``iterative.py``:

- The **inner round** ends as soon as ``self_reflection`` writes a
  ``judge_result.json`` file under ``<workspace>/final_runs/run_*/`` during
  this round, regardless of whether the agent emitted ``<done>true</done>``.
  An exit message with ``exit_status="SelfReflectionCompleted"`` is injected
  to break the step loop.
- The ``step_limit`` on the inner agent still applies (default 50).
- The outer loop is unchanged: the round verdict is read from that same
  ``judge_result.json`` via ``_read_tool_judge_result``, and the run stops
  on judge success or ``max_rounds`` exhaustion.

Use by setting ``iterative.runner_class: strict`` in the config, or by
passing ``-c iterative_strict.yaml`` to ``miniswewebagent.run.iterative``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined
from jinja2 import Template as JinjaTemplate

from miniswewebagent.agents.iterative import (
    IterativeRunner,
    RoundArtifacts,
    _IterativeInnerAgent,
    _collect_round_artifacts,
    _read_tool_judge_result,
    _run_judge,
    _run_workspace_judge,
    _save_judge_result,
)
from miniswewebagent.exceptions import FormatError, InterruptAgentFlow
from miniswewebagent.models.phyagi_model import image_part_from_path, text_part


# ---------------------------------------------------------------------------
# Inner agent: exits on self_reflection completion
# ---------------------------------------------------------------------------

class _StrictInnerAgent(_IterativeInnerAgent):
    """Inner agent that exits as soon as self_reflection writes judge_result.json."""

    def run(
        self,
        task: str = "",
        *,
        _override_instance_message: dict[str, Any] | None = None,
        _workspace_dir: Path | None = None,
        _round_start_ts: float = 0.0,
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

            if _workspace_dir is not None and _self_reflection_completed(
                _workspace_dir, since_ts=_round_start_ts
            ):
                print(
                    "[StrictIterative] self_reflection produced judge_result.json — "
                    "forcing round exit (ignoring agent done flag)."
                )
                self.add_messages(
                    self.model.format_message(
                        role="exit",
                        content="self_reflection completed; round terminated by StrictIterativeRunner.",
                        extra={
                            "exit_status": "SelfReflectionCompleted",
                            "submission": "",
                            "final_response": "",
                        },
                    )
                )
                self.save(self.config.output_path)
                break

        return self.messages[-1].get("extra", {})


def _self_reflection_completed(workspace_dir: Path, *, since_ts: float) -> bool:
    """Return True iff any ``final_runs/run_*/judge_result.json`` has mtime >= since_ts."""
    final_runs_dir = workspace_dir / "final_runs"
    if not final_runs_dir.is_dir():
        return False
    for run_dir in final_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        jr = run_dir / "judge_result.json"
        try:
            if jr.is_file() and jr.stat().st_mtime >= since_ts:
                return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Outer runner: identical to IterativeRunner but uses _StrictInnerAgent
# ---------------------------------------------------------------------------

class StrictIterativeRunner(IterativeRunner):
    """Outer loop that swaps in ``_StrictInnerAgent`` for the per-round agent."""

    def run(
        self,
        task: str,
        *,
        task_id: str = "",
        start_url: str = "",
        base_output_dir: Path | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        prev_artifacts: RoundArtifacts | None = None
        last_result: dict[str, Any] = {}

        for round_id in range(1, self.max_rounds + 1):
            print(
                f"\n[StrictIterativeRunner] ===== Round {round_id}/{self.max_rounds} ====="
            )

            round_output_dir = (
                base_output_dir / f"round_{round_id}" if base_output_dir else None
            )

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
                template_vars.update(
                    {
                        "prev_round_id": prev_artifacts.round_id,
                        "prev_round_dir": str(prev_artifacts.round_dir),
                        "prev_final_script": prev_artifacts.final_script,
                        "prev_command_output": prev_artifacts.command_output,
                        "prev_judge_response": (
                            prev_artifacts.judge_result.response_text
                            if prev_artifacts.judge_result
                            else ""
                        ),
                    }
                )

            template_str = (
                self.next_round_instance_template
                if prev_artifacts is not None
                else self.round1_instance_template
            )
            rendered_text = JinjaTemplate(
                template_str, undefined=StrictUndefined
            ).render(**template_vars)

            if prev_artifacts is not None and prev_artifacts.screenshot_paths:
                content: list[dict[str, Any]] = [text_part(rendered_text)]
                for sc_path in prev_artifacts.screenshot_paths:
                    if sc_path.exists():
                        content.append(image_part_from_path(sc_path))
                instance_msg = self.model.format_message(role="user", content=content)
            else:
                instance_msg = self.model.format_message(
                    role="user", content=rendered_text
                )

            agent_cfg = dict(self.agent_config)
            if round_output_dir:
                agent_cfg["output_path"] = str(round_output_dir / "trajectory.json")

            agent = _StrictInnerAgent(self.model, self.env, **agent_cfg)
            agent.extra_template_vars = dict(template_vars)

            workspace_dir = self._get_workspace_dir()
            round_start_ts = time.time()

            last_result = agent.run(
                task,
                task_id=task_id,
                start_url=start_url,
                round_id=round_id,
                **kwargs,
                _override_instance_message=instance_msg,
                _workspace_dir=workspace_dir,
                _round_start_ts=round_start_ts,
            )

            round_artifacts = _collect_round_artifacts(
                round_id=round_id,
                workspace_dir=workspace_dir,
                agent_output_dir=round_output_dir
                or (
                    Path(agent_cfg["output_path"]).parent
                    if agent_cfg.get("output_path")
                    else None
                ),
                agent=agent,
            )

            if round_artifacts is None:
                print(
                    f"[StrictIterativeRunner] Round {round_id}: completion gate failed — "
                    "no valid final_runs/run_N/ directory found. Skipping judge."
                )
                prev_artifacts = None
                continue

            print(
                f"[StrictIterativeRunner] Running judge ({self.judge_mode}) on: "
                f"{round_artifacts.round_dir}"
            )
            if self.judge_mode == "workspace":
                judge_result = _run_workspace_judge(
                    task=task,
                    artifacts=round_artifacts,
                    workspace_dir=workspace_dir,
                )
            elif self.judge_mode == "tool":
                judge_result = _read_tool_judge_result(artifacts=round_artifacts)
            else:
                judge_result = _run_judge(
                    task=task,
                    artifacts=round_artifacts,
                    judge_model=self.judge_model,
                    judge_gateway_endpoint=self.judge_gateway_endpoint,
                    judge_score_threshold=self.judge_score_threshold,
                )
            round_artifacts.judge_result = judge_result

            _save_judge_result(round_artifacts)

            verdict = "SUCCESS" if judge_result.success else "FAILURE"
            print(
                f"[StrictIterativeRunner] Round {round_id} judge verdict: {verdict}"
            )

            if judge_result.success:
                last_result["_iterative_success"] = True
                last_result["_iterative_rounds"] = round_id
                last_result["_final_round_dir"] = str(round_artifacts.round_dir)
                return last_result

            prev_artifacts = round_artifacts

        last_result["_iterative_success"] = False
        last_result["_iterative_rounds"] = self.max_rounds
        if prev_artifacts is not None:
            last_result["_final_round_dir"] = str(prev_artifacts.round_dir)
        return last_result

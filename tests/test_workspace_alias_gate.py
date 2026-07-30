from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from miniswewebagent.agents.default import DefaultAgent  # noqa: E402


class _FakeModel:
    def get_template_vars(self):
        return {}

    def format_message(self, **kwargs):
        return {"role": kwargs["role"], "content": kwargs.get("content", ""), "extra": kwargs.get("extra", {})}


class _FakeEnv:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    def get_template_vars(self):
        return {
            "workspace_dir": "/workspace",
            "output_dir": "/workspace",
            "task_metadata_path": "/workspace/task.json",
            "final_script_path": "/workspace/final_script.py",
        }

    def serialize(self):
        return {"environment": {"workspace_dir": str(self.workspace_dir)}}


def _agent(workspace_dir: Path) -> DefaultAgent:
    return DefaultAgent(
        _FakeModel(),
        _FakeEnv(workspace_dir),
        system_template="",
        instance_template="",
        require_self_reflection_success=True,
    )


def test_tool_gate_uses_real_workspace_when_prompt_uses_alias(tmp_path):
    workspace_dir = tmp_path / "real_ws"
    run_dir = workspace_dir / "final_runs" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "judge_result.json").write_text(json.dumps({"predicted_label": 1}))

    assert _agent(workspace_dir)._tool_gate_error() is None


def test_tool_gate_error_keeps_alias_in_agent_message(tmp_path):
    workspace_dir = tmp_path / "real_ws"
    (workspace_dir / "final_runs" / "run_1").mkdir(parents=True)

    error = _agent(workspace_dir)._tool_gate_error()

    assert error is not None
    assert "/workspace/final_runs/run_1/judge_result.json" in error
    assert str(workspace_dir) not in error


def test_plan_md_message_uses_real_workspace_when_prompt_uses_alias(tmp_path):
    workspace_dir = tmp_path / "real_ws"
    workspace_dir.mkdir()
    (workspace_dir / "plan.md").write_text("# Critical Points\n- [ ] CP1")

    message = _agent(workspace_dir)._plan_md_message()

    assert message is not None
    assert "CP1" in message["content"]
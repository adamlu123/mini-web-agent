from __future__ import annotations

import json
from pathlib import Path

from miniswewebagent.agents.default import DefaultAgent
from miniswewebagent.utils.browser_evidence import (
    append_jsonl,
    load_browser_steps,
    optional_file_digest,
    trajectory_evidence_digest,
)


class _StubModel:
    def get_template_vars(self, **kwargs):
        return kwargs


class _StubEnvironment:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def get_template_vars(self, **kwargs):
        return {"workspace_dir": str(self.workspace), **kwargs}


def _make_agent(workspace: Path) -> DefaultAgent:
    return DefaultAgent(
        _StubModel(),
        _StubEnvironment(workspace),
        system_template="system",
        instance_template="instance",
        require_self_reflection_success=True,
        judge_mode="trajectory",
    )


def test_trajectory_gate_accepts_fresh_pass_and_rejects_new_step(tmp_path: Path) -> None:
    plan = tmp_path / "plan.md"
    config = tmp_path / "judge_config.json"
    plan.write_text("- [x] exact filter", encoding="utf-8")
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "browser-steps.jsonl"
    append_jsonl(
        manifest,
        {
            "browser_step": 1,
            "agent_step": 2,
            "session_epoch": 1,
            "action": "Apply the exact filter",
            "success": True,
            "url_after": "https://example.com/results",
        },
    )
    rows = load_browser_steps(tmp_path)
    result_path = tmp_path / "reflection" / "judge_result.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "scope": "trajectory",
                "predicted_label": 1,
                "covered_through_browser_step": 1,
                "evidence_digest": trajectory_evidence_digest(tmp_path, rows),
                "plan_digest": optional_file_digest(plan),
                "judge_config_digest": optional_file_digest(config),
            }
        ),
        encoding="utf-8",
    )

    agent = _make_agent(tmp_path)
    assert agent._trajectory_gate_error() is None

    append_jsonl(
        manifest,
        {
            "browser_step": 2,
            "agent_step": 4,
            "session_epoch": 1,
            "action": "Open a new tab",
            "success": True,
        },
    )
    assert "stale" in str(agent._trajectory_gate_error()).lower()

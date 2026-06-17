"""Contract tests for the in-rollout self-judge CLIs.

Pins the eval-time tools to the SFT / mini-swe-webagent invocation the 9B model
was trained to emit, WITHOUT needing a policy endpoint (we stub the policy
chat). Two things are asserted:

1. self_reflection accepts the trained command
   (`--config ... --workspace-dir ... --auto-latest-run ... --output ...`),
   auto-attaches the latest run's screenshots, fills the prompt placeholders
   (including a tolerant `{critical_points}` the model also authors), writes
   judge_result.json with predicted_label, and exits 0 on success / 1 on fail.
2. The legacy RL contract (`--task ... --image ...`) still works.
3. image_qa accepts `--workspace-dir` and resolves relative --image paths.

Run: WANDB_DISABLED=true pytest -q tests/test_self_reflection_contract.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "echo_rl" / "web_agent" / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import image_qa  # noqa: E402
import self_reflection  # noqa: E402


def _fake_chat_factory(final_status: str):
    """Return a chat() stub: image stage -> 'Score: 5', final stage -> status."""

    def fake_chat(messages, **kw):
        system = ""
        if messages and isinstance(messages[0].get("content"), str):
            system = messages[0]["content"].lower()
        if "score" in system and "screenshot" in system:  # image-judge stage
            return "Reasoning: the chip is clearly selected.\nScore: 5"
        return f"Thoughts: every critical point is satisfied.\nStatus: {final_status}"

    return fake_chat


def _build_ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    run = ws / "final_runs" / "run_3" / "screenshots"
    run.mkdir(parents=True)
    # two screenshots (bytes don't need to be a valid PNG for encode_image)
    (run / "final_execution_1_apply.png").write_bytes(b"\x89PNG\r\n\x1a\nfake1")
    (run / "final_execution_2_final.png").write_bytes(b"\x89PNG\r\n\x1a\nfake2")
    (ws / "final_runs" / "run_3" / "final_script_log.txt").write_text(
        "step 1 action: applied filter\nFinal Response: done\n"
    )
    (ws / "plan.md").write_text("# Critical Points\n- [ ] CP1: apply filter\n- [ ] CP2: show results\n")
    # judge_config.json deliberately also references {critical_points} (which the
    # native tool does NOT fill) to prove tolerant substitution doesn't crash.
    (ws / "judge_config.json").write_text(json.dumps({
        "image_judge_system_prompt": "You are a harsh evaluator. Score this screenshot. Reply 'Reasoning:' then 'Score: 1-5'.",
        "image_judge_user_prompt": "Task: T. Score the attached screenshot against the critical points.",
        "final_verdict_system_prompt": "Aggregated judge. End with 'Status: success' or 'Status: failure'.",
        "final_verdict_user_prompt": "Log:\n{action_history_log}\n\nReasonings:\n{image_reasonings}\n\nCPs:\n{critical_points}\n\nVerdict?",
    }))
    return ws


def test_sft_contract_pass(tmp_path, monkeypatch):
    ws = _build_ws(tmp_path)
    monkeypatch.setattr(self_reflection, "chat", _fake_chat_factory("success"))
    out = ws / "final_runs" / "run_3" / "judge_result.json"
    rc = self_reflection.main([
        "--config", str(ws / "judge_config.json"),
        "--workspace-dir", str(ws),
        "--auto-latest-run", "final_runs",
        "--output", str(out),
    ])
    assert rc == 0
    assert out.is_file()
    res = json.loads(out.read_text())
    assert res["predicted_label"] == 1
    assert res["verdict"] == "success"
    # auto-attached BOTH screenshots from the latest run
    assert len(res["images"]) == 2
    assert len(res["image_records"]) == 2
    # placeholders were filled; {critical_points} survived tolerantly (filled too)
    assert "applied filter" in res["final_prompt"]
    assert "Score: 5" in res["final_prompt"]
    assert "CP1" in res["final_prompt"]


def test_sft_contract_fail_exit_code(tmp_path, monkeypatch):
    ws = _build_ws(tmp_path)
    monkeypatch.setattr(self_reflection, "chat", _fake_chat_factory("failure"))
    out = ws / "final_runs" / "run_3" / "judge_result.json"
    rc = self_reflection.main([
        "--config", str(ws / "judge_config.json"),
        "--workspace-dir", str(ws),
        "--output", str(out),
    ])
    assert rc == 1
    assert json.loads(out.read_text())["predicted_label"] == 0


def test_rl_contract_still_works(tmp_path, monkeypatch):
    ws = _build_ws(tmp_path)
    img = ws / "final_runs" / "run_3" / "screenshots" / "final_execution_1_apply.png"
    monkeypatch.setattr(self_reflection, "chat", _fake_chat_factory("success"))
    rc = self_reflection.main([
        "--task", "Apply the filter",
        "--critical-points", "1. apply filter\n2. show results",
        "--action-log", "step 1 action: applied filter",
        "--image", str(img),
    ])
    assert rc == 0


def test_image_qa_workspace_dir(tmp_path, monkeypatch):
    ws = _build_ws(tmp_path)
    monkeypatch.setattr(image_qa, "chat", lambda *a, **k: '{"answer": "yes", "evidence": "chip", "confidence": "high"}')
    rc = image_qa.main([
        "--workspace-dir", str(ws),
        "--image", "final_runs/run_3/screenshots/final_execution_1_apply.png",
        "--question", "Is the chip selected?",
    ])
    assert rc == 0

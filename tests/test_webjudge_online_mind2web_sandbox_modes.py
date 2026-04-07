import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from om2w_judge.methods import webjudge_online_mind2web_sandbox as sandbox
from om2w_judge.utils import extract_predication


async def _fake_identify_key_points(task, model):
    del task, model
    return "Key Points:\n- apply all required filters"


def test_sandbox_eval_default_uses_action_history_without_thoughts(monkeypatch):
    monkeypatch.setattr(sandbox, "identify_key_points", _fake_identify_key_points)

    messages, text, system_msg, record, key_points = asyncio.run(
        sandbox.WebJudge_Online_Mind2Web_Sandbox_eval(
            "Find the cheapest valid listing",
            ["thought one"],
            ["click filter", "click sort"],
            [],
            model=None,
            score_threshold=3,
        )
    )

    assert "Action History:" in text
    assert "Trajectory Thoughts:" not in text
    assert "agent's action history" in system_msg
    assert "agent's thoughts, action history" not in system_msg
    assert messages[1]["content"][0]["text"] == text
    assert record == []
    assert key_points.strip()


def test_sandbox_eval_with_thoughts_and_thoughts_only_modes(monkeypatch):
    monkeypatch.setattr(sandbox, "identify_key_points", _fake_identify_key_points)

    _, with_text, with_system_msg, _, _ = asyncio.run(
        sandbox.WebJudge_Online_Mind2Web_Sandbox_WithThoughts_eval(
            "Find the cheapest valid listing",
            ["thought one"],
            ["click filter"],
            [],
            model=None,
            score_threshold=3,
        )
    )
    _, thoughts_only_text, thoughts_only_system_msg, _, _ = asyncio.run(
        sandbox.WebJudge_Online_Mind2Web_Sandbox_ThoughtsOnly_eval(
            "Find the cheapest valid listing",
            ["thought one"],
            [],
            model=None,
            score_threshold=3,
        )
    )

    assert "Trajectory Thoughts:" in with_text
    assert "Action History:" in with_text
    assert "agent's thoughts, action history" in with_system_msg

    assert "Trajectory Thoughts:" in thoughts_only_text
    assert "Action History:" not in thoughts_only_text
    assert "agent's thoughts, key points" in thoughts_only_system_msg


def test_extract_predication_supports_with_thoughts_mode():
    assert extract_predication('Thoughts: ok\nStatus: "success"', "WebJudge_Online_Mind2Web_Sandbox_WithThoughts_eval") == 1


def test_extract_predication_uses_final_status_line():
    response = (
        "Thoughts: The agent applied filters including Status: Production/Stable and verified the top result.\n"
        "Status: success"
    )
    assert extract_predication(response, "WebJudge_Online_Mind2Web_Sandbox_eval") == 1


def test_extract_predication_supports_inline_and_curly_quoted_status():
    inline_response = (
        "Thoughts: All required filters are active and results are displayed. Status: “success”"
    )
    assert extract_predication(inline_response, "WebJudge_Online_Mind2Web_Sandbox_eval") == 1

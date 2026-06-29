from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from miniswewebagent.models.phyagi_model import parse_bash_answer_output, parse_sft_state_output  # noqa: E402
from miniswewebagent.models.openrouter_model import _serialize_chat_messages  # noqa: E402
from miniswewebagent.models.openai_compatible_model import OpenAICompatibleModel  # noqa: E402


def test_sft_parser_accepts_missing_opening_think_tag():
    parsed = parse_bash_answer_output(
        "I should inspect the task before acting.\n"
        "</think>\n"
        "<bash>\n"
        "cd /workspace && sed -n '1,120p' task.json\n"
        "</bash>"
    )

    assert parsed["thought"] == "I should inspect the task before acting."
    assert parsed["bash_command"] == "cd /workspace && sed -n '1,120p' task.json"
    assert parsed["done"] is False


def test_sft_parser_keeps_normal_think_tag_behavior():
    parsed = parse_bash_answer_output(
        "<think>done after self reflection passed</think>\n"
        "<answer>final answer</answer>"
    )

    assert parsed["thought"] == "done after self reflection passed"
    assert parsed["bash_command"] == ""
    assert parsed["done"] is True
    assert parsed["final_response"] == "final answer"


def test_sft_state_parser_accepts_action_turn():
    parsed = parse_sft_state_output(
        "<think>inspect task metadata</think>\n"
        "<bash>cd /workspace && cat task.json</bash>\n"
        "<done>false</done>\n"
        "<final_response></final_response>"
    )

    assert parsed["thought"] == "inspect task metadata"
    assert parsed["bash_command"] == "cd /workspace && cat task.json"
    assert parsed["done"] is False
    assert parsed["final_response"] == ""


def test_sft_state_parser_accepts_done_turn():
    parsed = parse_sft_state_output(
        "finished and verified\n"
        "</think>\n"
        "<bash>\n</bash>\n"
        "<done>true</done>\n"
        "<final_response>Done.</final_response>"
    )

    assert parsed["thought"] == "finished and verified"
    assert parsed["bash_command"] == ""
    assert parsed["done"] is True
    assert parsed["final_response"] == "Done."


def test_sft_state_chat_serializer_restores_assistant_xml_history():
    serialized = _serialize_chat_messages(
        [
            {
                "role": "assistant",
                "content": "inspect task metadata",
                "extra": {
                    "raw_response": {
                        "thought": "inspect task metadata",
                        "bash_command": "cd /workspace && cat task.json",
                        "python_code": "",
                        "done": False,
                        "final_response": "",
                    },
                    "raw_text": "inspect task metadata</think><bash>cd /workspace && cat task.json</bash>",
                },
            }
        ],
        response_mode="sft_state",
    )

    assert serialized == [
        {
            "role": "assistant",
            "content": (
                "<think>\n"
                "inspect task metadata\n"
                "</think>\n"
                "<bash>\n"
                "cd /workspace && cat task.json\n"
                "</bash>\n"
                "<done>false</done>\n"
                "<final_response>\n"
                "\n"
                "</final_response>"
            ),
        }
    ]


def test_chat_serializer_keeps_plain_assistant_content_outside_sft_state():
    serialized = _serialize_chat_messages(
        [
            {
                "role": "assistant",
                "content": "plain thought",
                "extra": {"raw_response": {"thought": "full", "bash_command": "echo hi"}},
            }
        ],
        response_mode="xml",
    )

    assert serialized == [{"role": "assistant", "content": "plain thought"}]


def test_openai_compatible_debug_request_matches_sft_state_chat_payload():
    model = OpenAICompatibleModel(
        model_name="local-policy",
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        api_key="dummy",
        response_mode="sft_state",
    )
    payload = model.serialize_request_for_debug(
        [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": "plain thought",
                "extra": {
                    "raw_response": {
                        "thought": "plain thought",
                        "bash_command": "echo hi",
                        "done": False,
                        "final_response": "",
                    }
                },
            },
        ]
    )

    assert payload["model"] == "local-policy"
    assert payload["response_mode"] == "sft_state"
    assert payload["messages"][1]["content"] == (
        "<think>\n"
        "plain thought\n"
        "</think>\n"
        "<bash>\n"
        "echo hi\n"
        "</bash>\n"
        "<done>false</done>\n"
        "<final_response>\n"
        "\n"
        "</final_response>"
    )

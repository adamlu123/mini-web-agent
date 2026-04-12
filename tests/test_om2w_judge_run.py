import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from om2w_judge.run import load_final_script_action_history, resolve_latest_final_run_dir, resolve_sandbox_artifact_dir
from om2w_judge.utils import OpenaiEngine


def test_load_final_script_action_history_reads_only_step_lines(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "final_script_log.txt").write_text(
        "step 1 action: search destination\n"
        "FINAL_URL: https://example.com\n"
        "step 2 action: apply exact filters\n",
        encoding="utf-8",
    )

    assert load_final_script_action_history(task_dir) == [
        "step 1 action: search destination",
        "step 2 action: apply exact filters",
    ]


def test_resolve_latest_final_run_dir_prefers_highest_numeric_id(tmp_path):
    task_dir = tmp_path / "task"
    (task_dir / "final_runs" / "run_002" / "screenshots").mkdir(parents=True)
    (task_dir / "final_runs" / "run_010" / "screenshots").mkdir(parents=True)
    (task_dir / "final_runs" / "run_010" / "final_script_log.txt").write_text(
        "step 1 action: final run\n",
        encoding="utf-8",
    )

    latest = resolve_latest_final_run_dir(task_dir)

    assert latest == task_dir / "final_runs" / "run_010"
    assert resolve_sandbox_artifact_dir(task_dir) == latest


def test_load_final_script_action_history_prefers_latest_final_run_folder(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "final_script_log.txt").write_text(
        "step 1 action: legacy root action\n",
        encoding="utf-8",
    )
    (task_dir / "final_runs" / "run_003").mkdir(parents=True)
    (task_dir / "final_runs" / "run_003" / "final_script_log.txt").write_text(
        "step 1 action: newer folder action\n"
        "step 2 action: final confirmation\n",
        encoding="utf-8",
    )

    assert load_final_script_action_history(task_dir) == [
        "step 1 action: newer folder action",
        "step 2 action: final confirmation",
    ]


def test_resolve_sandbox_artifact_dir_falls_back_to_task_root(tmp_path):
    task_dir = tmp_path / "task"
    task_dir.mkdir()

    assert resolve_latest_final_run_dir(task_dir) is None
    assert resolve_sandbox_artifact_dir(task_dir) == task_dir


def test_openai_engine_uses_gateway_endpoint_when_requested(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": 'Status: "success"',
                            }
                        ],
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout: int):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("om2w_judge.utils.httpx.Client", FakeClient)

    engine = OpenaiEngine(
        api_key="dummy-gateway-key",
        model="gpt-5.4",
        endpoint_target_uri="http://gateway.example/api/responses",
    )
    responses = engine.generate(
        [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Task"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc", "detail": "high"},
                    },
                ],
            },
        ],
        max_new_tokens=321,
    )

    payload = captured["json"]
    assert responses == ['Status: "success"']
    assert captured["url"] == "http://gateway.example/api/responses"
    assert captured["headers"]["Authorization"] == "Bearer dummy-gateway-key"
    assert payload["model"] == "gpt-5.4"
    assert payload["max_output_tokens"] == 321
    assert "messages" not in payload
    assert payload["input"][0]["role"] == "developer"
    assert payload["input"][0]["content"][0] == {"type": "input_text", "text": "system prompt"}
    assert payload["input"][1]["content"][0] == {"type": "input_text", "text": "Task"}
    assert payload["input"][1]["content"][1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,abc",
        "detail": "high",
    }

import subprocess
from pathlib import Path

from miniswewebagent.utils.om2w_eval import run_online_mind2web_judge
from om2w_judge.utils import _extract_response_text, _serialize_response_input


def test_serialize_response_input_maps_chat_content() -> None:
    messages = [
        {"role": "system", "content": "judge"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "task"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
            ],
        },
    ]

    serialized = _serialize_response_input(messages)

    assert serialized[0]["role"] == "developer"
    assert serialized[1]["content"][1]["type"] == "input_image"
    assert _extract_response_text({"output_text": "Status: success"}) == "Status: success"


def test_runner_uses_native_judge_cli(monkeypatch, tmp_path: Path) -> None:
    captured: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured.extend(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_online_mind2web_judge(
        judge_python=Path("python"),
        judge_script=Path("run.py"),
        trajectories_dir=tmp_path / "outputs",
        output_dir=tmp_path / "eval",
        judge_model="o4-mini",
        num_proc=32,
        api_key="test",
        endpoint_target_uri="http://gateway.test/api/responses",
    )

    assert captured[captured.index("--mode") + 1] == "WebJudge_Online_Mind2Web_eval"
    assert captured[captured.index("--num_worker") + 1] == "32"
    assert captured[captured.index("--endpoint_target_uri") + 1] == (
        "http://gateway.test/api/responses"
    )

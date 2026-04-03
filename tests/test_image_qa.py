from __future__ import annotations

import json
from pathlib import Path

from miniswewebagent.tools.image_qa import run_image_qa


def test_run_image_qa_returns_structured_json(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

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
                                "text": json.dumps(
                                    {
                                        "answer": "The selected chip is BMW.",
                                        "evidence": ["A BMW chip is visible near the top."],
                                        "unknown": False,
                                        "confidence": 0.81,
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict, json: dict) -> FakeResponse:
            assert json["model"] == "gpt-5.4"
            assert json["input"][0]["content"][1]["type"] == "input_image"
            assert json["input"][0]["content"][1]["detail"] == "high"
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.tools.image_qa.httpx.Client", FakeClient)

    result = run_image_qa(
        image_path=image_path,
        question="What chip is selected?",
        api_key="dummy",
        endpoint="http://gateway.example/api/responses",
        model="gpt-5.4",
        timeout_seconds=30,
    )

    assert result["image_path"] == str(image_path)
    assert result["answer"] == "The selected chip is BMW."
    assert result["unknown"] is False

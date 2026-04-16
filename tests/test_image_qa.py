from __future__ import annotations

import json
from pathlib import Path

from miniswewebagent.tools.image_qa import run_image_qa


def test_run_image_qa_returns_structured_json(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    class FakeResponse:
        status_code = 200

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
            assert len(json["input"][0]["content"]) == 2
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
    assert result["image_paths"] == [str(image_path)]
    assert result["answer"] == "The selected chip is BMW."
    assert result["unknown"] is False


def test_run_image_qa_supports_multiple_images(monkeypatch, tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"\x89PNG\r\n\x1a\n")
    image_b.write_bytes(b"\x89PNG\r\n\x1a\n")

    class FakeResponse:
        status_code = 200

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
                                        "answer": "The first image shows BMW and the second shows the sorted list.",
                                        "evidence": [
                                            "BMW chip is visible in image 1.",
                                            "Lowest price sort is visible in image 2.",
                                        ],
                                        "unknown": False,
                                        "confidence": 0.9,
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
            content = json["input"][0]["content"]
            assert json["model"] == "gpt-5.4"
            assert content[0]["type"] == "input_text"
            assert [part["type"] for part in content[1:]] == ["input_image", "input_image"]
            assert all(part["detail"] == "high" for part in content[1:])
            return FakeResponse()

    monkeypatch.setattr("miniswewebagent.tools.image_qa.httpx.Client", FakeClient)

    result = run_image_qa(
        image_paths=[image_a, image_b],
        question="What do these two screenshots show together?",
        api_key="dummy",
        endpoint="http://gateway.example/api/responses",
        model="gpt-5.4",
        timeout_seconds=30,
    )

    assert result["image_paths"] == [str(image_a), str(image_b)]
    assert "image_path" not in result
    assert result["unknown"] is False


def test_run_image_qa_retries_on_transient_4xx(monkeypatch, tmp_path: Path) -> None:
    import httpx

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    calls = {"count": 0}

    class Retry400Response:
        status_code = 400
        text = "transient"

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("bad request", request=None, response=None)  # type: ignore[arg-type]

    class OkResponse:
        status_code = 200

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
                                        "answer": "OK",
                                        "evidence": [],
                                        "unknown": False,
                                        "confidence": 1.0,
                                    }
                                ),
                            }
                        ],
                    }
                ]
            }

    class FakeClient:
        def __init__(self, timeout: int):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict, json: dict):
            calls["count"] += 1
            if calls["count"] < 3:
                return Retry400Response()
            return OkResponse()

    monkeypatch.setattr("miniswewebagent.tools.image_qa.httpx.Client", FakeClient)
    monkeypatch.setattr("miniswewebagent.tools.image_qa.time.sleep", lambda _s: None)

    result = run_image_qa(
        image_path=image_path,
        question="ok?",
        api_key="dummy",
        endpoint="http://gateway.example/api/responses",
        model="gpt-5.4",
        timeout_seconds=5,
        max_attempts=4,
        retry_base_delay=0.0,
    )

    assert calls["count"] == 3
    assert result["answer"] == "OK"


def test_run_image_qa_raises_after_max_attempts(monkeypatch, tmp_path: Path) -> None:
    import httpx

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    calls = {"count": 0}

    class Always400Response:
        status_code = 400
        text = "persistent error"

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError("bad request", request=None, response=None)  # type: ignore[arg-type]

    class FakeClient:
        def __init__(self, timeout: int):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url: str, headers: dict, json: dict):
            calls["count"] += 1
            return Always400Response()

    monkeypatch.setattr("miniswewebagent.tools.image_qa.httpx.Client", FakeClient)
    monkeypatch.setattr("miniswewebagent.tools.image_qa.time.sleep", lambda _s: None)

    import pytest

    with pytest.raises(httpx.HTTPStatusError):
        run_image_qa(
            image_path=image_path,
            question="ok?",
            api_key="dummy",
            endpoint="http://gateway.example/api/responses",
            model="gpt-5.4",
            timeout_seconds=5,
            max_attempts=3,
            retry_base_delay=0.0,
        )

    assert calls["count"] == 3

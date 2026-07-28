import base64
import copy
import struct
import zlib

from miniswewebagent.models.openai_compatible_model import OpenAICompatibleModel
from miniswewebagent.models.openrouter_model import BLACK_56_PNG_DATA_URL, OpenRouterModel


def _assistant_message(index: int, size: int = 600) -> dict:
    return {
        "role": "assistant",
        "content": "short",
        "extra": {
            "raw_response": {
                "thought": f"turn {index} " + ("x" * size),
                "bash_command": "echo done",
                "done": False,
                "final_response": "",
            }
        },
    }


def test_sliding_window_estimates_serialized_sft_state() -> None:
    model = OpenRouterModel(
        model_name="test",
        openrouter_api_key="dummy",
        response_mode="sft_state",
        max_context_tokens=450,
        sliding_window_keep_turns=1,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        _assistant_message(1),
        {"role": "user", "content": "observation 1"},
        _assistant_message(2),
        {"role": "user", "content": "observation 2"},
    ]

    assert model._estimate_tokens(messages) < model.config.max_context_tokens
    assert model._estimate_request_tokens(messages) > model.config.max_context_tokens

    payload = model._build_payload(messages)

    assert len(payload["messages"]) == 5
    assert payload["messages"][2]["content"].startswith("[Context truncated:")
    assert "turn 1" not in str(payload["messages"])
    assert "turn 2" in str(payload["messages"])


def test_request_estimate_reserves_image_tokens() -> None:
    model = OpenRouterModel(model_name="test", openrouter_api_key="dummy")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "abc"},
                {"type": "input_image", "image_url": "data:image/png;base64,test"},
            ],
        }
    ]

    assert model._estimate_request_tokens(messages) == 2049


def _decode_png_rgb(data_url: str) -> tuple[int, int, bytes]:
    raw = base64.b64decode(data_url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    width = height = 0
    compressed = b""
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        data = raw[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
            assert data[8:10] == b"\x08\x02"  # 8-bit truecolor RGB
        elif chunk_type == b"IDAT":
            compressed += data
        elif chunk_type == b"IEND":
            break
    scanlines = zlib.decompress(compressed)
    pixels = b"".join(
        scanlines[row * (width * 3 + 1) + 1 : (row + 1) * (width * 3 + 1)]
        for row in range(height)
    )
    return width, height, pixels


def test_black_56_policy_injects_exact_training_image_before_estimation() -> None:
    model = OpenRouterModel(
        model_name="test",
        openrouter_api_key="dummy",
        text_only_image_policy="black_56",
    )
    messages = [{"role": "user", "content": "abc"}]
    original = copy.deepcopy(messages)

    assert model._estimate_request_tokens(messages) == 2049
    payload = model._build_payload(messages)

    assert messages == original
    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": BLACK_56_PNG_DATA_URL},
    }
    assert content[1] == {"type": "text", "text": "abc"}
    width, height, pixels = _decode_png_rgb(content[0]["image_url"]["url"])
    assert (width, height) == (56, 56)
    assert pixels == bytes(56 * 56 * 3)


def test_black_56_policy_never_injects_when_request_has_a_real_image() -> None:
    model = OpenRouterModel(
        model_name="test",
        openrouter_api_key="dummy",
        text_only_image_policy="black_56",
    )
    messages = [
        {"role": "user", "content": "task"},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,real"},
                {"type": "text", "text": "observation"},
            ],
        },
    ]

    payload = model._build_payload(messages)

    assert payload["messages"][0]["content"] == "task"
    serialized = str(payload["messages"])
    assert BLACK_56_PNG_DATA_URL not in serialized
    assert "data:image/png;base64,real" in serialized


def test_live_and_debug_requests_share_alignment_sliding_and_stops() -> None:
    model = OpenAICompatibleModel(
        model_name="local-policy",
        endpoint="http://127.0.0.1:8000/v1/chat/completions",
        api_key="dummy",
        response_mode="sft_state",
        text_only_image_policy="black_56",
        stop_sequences=["<|im_end|>"],
        max_context_tokens=450,
        sliding_window_keep_turns=1,
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        _assistant_message(1),
        {"role": "user", "content": "observation 1"},
        _assistant_message(2),
        {"role": "user", "content": "observation 2"},
    ]

    live = model._build_payload(messages)
    debug = model.serialize_request_for_debug(messages)

    assert {key: debug[key] for key in live} == live
    assert live["stop"] == ["<|im_end|>"]
    assert live["messages"][1]["content"][0]["image_url"]["url"] == BLACK_56_PNG_DATA_URL
    assert live["messages"][2]["content"].startswith("[Context truncated:")
    assert "turn 1" not in str(live["messages"])
    assert "turn 2" in str(live["messages"])

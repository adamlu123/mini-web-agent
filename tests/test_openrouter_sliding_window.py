from miniswewebagent.models.openrouter_model import OpenRouterModel


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

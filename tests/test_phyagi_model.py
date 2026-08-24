from __future__ import annotations

import json
from pathlib import Path

import pytest
import httpx

from miniswewebagent.exceptions import FormatError
from miniswewebagent.models.phyagi_model import (
    MAX_JSON_PARSE_RETRIES,
    PhyagiModel,
    _extract_reasoning_summary,
    _is_output_truncated,
    _serialize_response_input,
    parse_sft_state_output,
    parse_xml_output,
)


class _FakeOpenAIResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self) -> dict:
        return self._payload


def _make_fake_async_openai(payload_factory, *, side_effect=None):
    """Build a FakeAsyncOpenAI class that mimics openai.AsyncOpenAI.

    payload_factory: dict OR callable(call_index, kwargs) -> dict returned via .model_dump().
    side_effect: optional callable(call_index, kwargs); may raise to simulate transient errors.
    """

    state = {"calls": 0, "last_kwargs": None}

    class FakeResponses:
        async def create(self, **kwargs):
            state["calls"] += 1
            state["last_kwargs"] = kwargs
            if side_effect is not None:
                side_effect(state["calls"], kwargs)
            payload = (
                payload_factory(state["calls"], kwargs)
                if callable(payload_factory)
                else payload_factory
            )
            return _FakeOpenAIResponse(payload)

    class FakeAsyncOpenAI:
        def __init__(self, *, api_key: str, base_url: str, timeout) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.responses = FakeResponses()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return FakeAsyncOpenAI, state


def _xml_message_payload(text: str, *, usage: dict | None = None) -> dict:
    payload: dict = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def test_parse_xml_output_with_cdata() -> None:
    parsed = parse_xml_output(
        """
<response>
  <thought>Tried: open form. Failed because: no execution error. Next step: fill route.</thought>
  <python_code><![CDATA[
await page.get_by_role("textbox", name="From").fill("Singapore")
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )

    assert parsed["thought"].startswith("Tried:")
    assert parsed["bash_command"] == ""
    assert 'fill("Singapore")' in parsed["python_code"]
    assert parsed["done"] is False
    assert parsed["final_response"] == ""


def test_parse_xml_output_accepts_bash_command() -> None:
    parsed = parse_xml_output(
        """
<response>
  <thought>Inspect the workspace.</thought>
  <bash_command><![CDATA[
echo hello
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )

    assert parsed["thought"] == "Inspect the workspace."
    assert parsed["bash_command"] == "echo hello"
    assert parsed["python_code"] == ""
    assert parsed["done"] is False
    assert parsed["final_response"] == ""


def test_parse_xml_output_repairs_done_true_when_command_present() -> None:
    parsed = parse_xml_output(
        """
<response>
    <thought>Inspect the workspace.</thought>
    <bash_command><![CDATA[
echo hello
]]></bash_command>
    <done>true</done>
    <final_response>ready</final_response>
</response>
        """.strip()
    )

    assert parsed["bash_command"] == "echo hello"
    # parse_xml_output does not repair done; _repair_done_with_command runs in _parse_model_output for json.
    assert parsed["done"] is True
    assert parsed["final_response"] == "ready"


def test_phyagi_model_formats_observations_from_template() -> None:
    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        observation_template="Status={{ 'ok' if observation.success else 'error' }} URL={{ observation.url }}",
    )

    messages = model.format_observation_messages(
        {},
        [
            {
                "observation": {
                    "success": True,
                    "url": "https://example.com",
                    "title": "Example",
                    "screenshot_path": "",
                }
            }
        ],
    )

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Status=ok URL=https://example.com"


def test_phyagi_model_attaches_observation_screenshot_by_default(tmp_path: Path) -> None:
    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        observation_template="Status={{ 'ok' if observation.success else 'error' }}",
    )

    messages = model.format_observation_messages(
        {},
        [
            {
                "observation": {
                    "success": True,
                    "url": "https://example.com",
                    "title": "Example",
                    "screenshot_path": str(screenshot),
                }
            }
        ],
    )

    assert [part["type"] for part in messages[0]["content"]] == ["input_text", "input_image"]


def test_phyagi_model_can_disable_observation_screenshot_attachment(tmp_path: Path) -> None:
    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"\x89PNG\r\n\x1a\n")

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        observation_template="Status={{ 'ok' if observation.success else 'error' }}",
        attach_observation_screenshot=False,
    )

    messages = model.format_observation_messages(
        {},
        [
            {
                "observation": {
                    "success": True,
                    "url": "https://example.com",
                    "title": "Example",
                    "screenshot_path": str(screenshot),
                }
            }
        ],
    )

    assert [part["type"] for part in messages[0]["content"]] == ["input_text"]


def test_phyagi_model_query_parses_xml_mode(monkeypatch) -> None:
    fake_cls, state = _make_fake_async_openai(
        _xml_message_payload(
            """
<response>
  <thought>Next step</thought>
  <python_code><![CDATA[
await page.title()
]]></python_code>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        )
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["content"] == "Next step"
    assert message["extra"]["actions"] == [{"python_code": "await page.title()"}]
    assert message["extra"]["done"] is False

    kwargs = state["last_kwargs"]
    assert "messages" not in kwargs
    assert "input" in kwargs
    assert "text" not in kwargs  # xml mode does not request json_schema
    assert kwargs["max_output_tokens"] == 4000
    assert kwargs["extra_body"] == {
        "tier": "base",
        "session_id": model.config.session_id,
        "strict_session": True,
    }


def test_phyagi_model_query_parses_xml_mode_with_bash_command(monkeypatch) -> None:
    fake_cls, _state = _make_fake_async_openai(
        _xml_message_payload(
            """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        )
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["content"] == "Inspect files"
    assert message["extra"]["actions"] == [{"bash_command": "pwd", "command": "pwd"}]
    assert message["extra"]["done"] is False
    assert message["extra"]["usage"]["last_response"]["input_tokens"] == 0


def test_phyagi_model_query_repairs_json_done_true_when_command_present(monkeypatch) -> None:
    fake_cls, _state = _make_fake_async_openai(
        _xml_message_payload(
            json.dumps(
                {
                    "thought": "Run the script first.",
                    "python_code": "print('hello')",
                    "done": True,
                    "final_response": "done",
                }
            )
        )
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="json",
        format_error_template="error: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["extra"]["actions"] == [{"python_code": "print('hello')"}]
    assert message["extra"]["done"] is False
    assert message["extra"]["final_response"] == "done"
    assert message["extra"]["raw_response"]["done"] is False


def test_phyagi_model_query_raises_format_error_for_bad_xml(monkeypatch) -> None:
    fake_cls, _state = _make_fake_async_openai(_xml_message_payload("not valid xml"))
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )

    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "Task"}])


def test_phyagi_model_query_raises_format_error_for_invalid_bash_command(monkeypatch) -> None:
    fake_cls, _state = _make_fake_async_openai(
        _xml_message_payload(
            """
<response>
  <thought>Write a script.</thought>
  <bash_command><![CDATA[
from pathlib import Path
root = Path(".")
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        )
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="error: {{ error }}",
    )

    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "Task"}])


def test_phyagi_model_query_repairs_plain_text_with_format_retry(monkeypatch) -> None:
    def factory(call_index: int, _kwargs: dict) -> dict:
        if call_index == 1:
            text = "Need more evidence before clicking Filters."
        else:
            text = """
<response>
  <thought>Retry worked</thought>
  <bash_command><![CDATA[
ls
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
            """.strip()
        return _xml_message_payload(text)

    fake_cls, state = _make_fake_async_openai(factory)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        format_error_template="Repair XML: {{ error }}",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert state["calls"] == 2
    assert message["content"] == "Retry worked"
    assert message["extra"]["actions"] == [{"bash_command": "ls", "command": "ls"}]


def test_phyagi_model_query_retries_transient_gateway_errors(monkeypatch) -> None:
    payload = _xml_message_payload(
        """
<response>
  <thought>Retry worked</thought>
  <python_code><![CDATA[
]]></python_code>
  <done>true</done>
  <final_response>ok</final_response>
</response>
        """.strip()
    )

    def side_effect(call_index: int, _kwargs: dict) -> None:
        if call_index < 3:
            request = httpx.Request("POST", "https://gateway.phyagi.net/api/responses")
            response = httpx.Response(502, request=request)
            raise httpx.HTTPStatusError("502 Bad Gateway", request=request, response=response)

    fake_cls, state = _make_fake_async_openai(payload, side_effect=side_effect)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    async def fake_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("miniswewebagent.models.phyagi_model.asyncio.sleep", fake_sleep)

    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml")
    message = model.query([{"role": "user", "content": "Task"}])

    assert state["calls"] == 3
    assert message["extra"]["done"] is True
    assert message["extra"]["final_response"] == "ok"


def test_phyagi_model_query_exposes_gateway_usage_in_template_vars(monkeypatch) -> None:
    payload = _xml_message_payload(
        """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip(),
        usage={
            "input_tokens": 123,
            "output_tokens": 45,
            "total_tokens": 168,
            "input_tokens_details": {"cached_tokens": 7},
            "output_tokens_details": {"reasoning_tokens": 11},
        },
    )
    fake_cls, _state = _make_fake_async_openai(payload)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml")
    model.query([{"role": "user", "content": "Task"}])

    template_vars = model.get_template_vars()

    assert template_vars["last_request_message_count"] == 1
    assert template_vars["last_request_text_chars"] == 4
    assert template_vars["last_request_input_tokens"] == 123
    assert template_vars["last_request_output_tokens"] == 45
    assert template_vars["last_request_total_tokens"] == 168
    assert template_vars["last_request_cached_input_tokens"] == 7
    assert template_vars["last_request_reasoning_output_tokens"] == 11
    assert template_vars["cumulative_input_tokens"] == 123
    assert template_vars["cumulative_output_tokens"] == 45
    usage_snapshot = model._usage_snapshot()
    assert usage_snapshot["last_request"]["input_tokens"] == 123
    assert usage_snapshot["last_request"]["cached_input_tokens"] == 7
    assert "text_chars" not in usage_snapshot["last_request"]
    assert usage_snapshot["cumulative_request"]["input_tokens"] == 123
    assert usage_snapshot["last_response"]["input_tokens"] == 123


def test_phyagi_model_query_logs_raw_text(tmp_path: Path, monkeypatch) -> None:
    payload = _xml_message_payload(
        """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )
    fake_cls, _state = _make_fake_async_openai(payload)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    runtime_log = tmp_path / "runtime_errors.jsonl"
    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml", error_log_path=runtime_log)
    model.query([{"role": "user", "content": "Task"}])

    raw_log = tmp_path / "raw_responses.jsonl"
    assert raw_log.exists()
    assert '"event": "raw_text"' in raw_log.read_text(encoding="utf-8")
    assert "<bash_command><![CDATA[" in raw_log.read_text(encoding="utf-8")


def test_phyagi_model_query_uses_zero_usage_when_gateway_omits_usage(monkeypatch) -> None:
    payload = _xml_message_payload(
        """
<response>
  <thought>Inspect files</thought>
  <bash_command><![CDATA[
pwd
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )
    fake_cls, _state = _make_fake_async_openai(payload)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(openai_gateway_api_key="dummy", response_mode="xml")
    model.query([{"role": "user", "content": "Task"}])

    template_vars = model.get_template_vars()

    assert template_vars["last_request_input_tokens"] == 0
    assert template_vars["cumulative_input_tokens"] == 0
    assert template_vars["last_request_text_chars"] == 4
    assert template_vars["cumulative_request_text_chars"] == 4
    usage_snapshot = model._usage_snapshot()
    assert usage_snapshot["last_request"]["input_tokens"] == 0
    assert usage_snapshot["cumulative_request"]["input_tokens"] == 0
    assert "text_chars" not in usage_snapshot["last_request"]


def test_phyagi_model_extra_body_includes_session_fields(monkeypatch) -> None:
    payload = _xml_message_payload(
        """
<response>
  <thought>x</thought>
  <bash_command><![CDATA[
echo ok
]]></bash_command>
  <done>false</done>
  <final_response></final_response>
</response>
        """.strip()
    )
    fake_cls, state = _make_fake_async_openai(payload)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="xml",
        cache_ttl=1209600,
        session_id="sess-1",
        strict_session=True,
    )
    model.query([{"role": "user", "content": "Task"}])

    extra_body = state["last_kwargs"]["extra_body"]
    assert extra_body == {
        "tier": "base",
        "cache_ttl": 1209600,
        "session_id": "sess-1",
        "strict_session": True,
    }


def test_phyagi_model_includes_reasoning_effort_in_payload() -> None:
    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        reasoning_effort="xhigh",
    )

    payload = model._build_payload([{"role": "user", "content": "Task"}])

    assert payload["reasoning"] == {"effort": "xhigh", "summary": "auto"}


def test_phyagi_model_reasoning_summary_can_be_disabled() -> None:
    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        reasoning_effort="xhigh",
        reasoning_summary=None,
    )

    payload = model._build_payload([{"role": "user", "content": "Task"}])

    assert payload["reasoning"] == {"effort": "xhigh"}


def test_phyagi_model_omits_reasoning_block_without_effort() -> None:
    model = PhyagiModel(openai_gateway_api_key="dummy")

    payload = model._build_payload([{"role": "user", "content": "Task"}])

    assert "reasoning" not in payload


def test_extract_reasoning_summary_joins_summary_items() -> None:
    payload = {
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "first"}]},
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "second"}]},
            {"type": "message", "content": [{"text": "{}"}]},
        ]
    }

    assert _extract_reasoning_summary(payload) == "first\n\nsecond"


def test_extract_reasoning_summary_handles_missing_summaries() -> None:
    # The gateway returns reasoning items with an empty summary list when it
    # chooses not to emit one, which is common even at high effort.
    assert _extract_reasoning_summary({"output": [{"type": "reasoning", "summary": []}]}) == ""
    assert _extract_reasoning_summary({"output": [{"type": "message"}]}) == ""
    assert _extract_reasoning_summary({}) == ""


def test_phyagi_model_legacy_endpoint_is_normalized_to_sdk_base_url() -> None:
    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        openai_gateway_endpoint="http://gateway.phyagi.net/api/responses",
    )
    assert model._sdk_base_url() == "http://gateway.phyagi.net/api"


def test_serialize_response_input_uses_output_text_for_assistant_messages() -> None:
    serialized = _serialize_response_input(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ]
    )

    assert serialized[0]["role"] == "developer"
    assert serialized[0]["content"][0]["type"] == "input_text"
    assert serialized[1]["role"] == "user"
    assert serialized[1]["content"][0]["type"] == "input_text"
    assert serialized[2]["role"] == "assistant"
    assert serialized[2]["content"][0]["type"] == "output_text"


def test_serialize_response_input_preserves_complete_raw_assistant_response() -> None:
    raw_response = {
        "thought": "Create the plan before browsing.",
        "bash_command": "printf 'plan' > plan.md",
        "done": False,
        "final_response": "",
    }

    serialized = _serialize_response_input(
        [
            {
                "role": "assistant",
                "content": raw_response["thought"],
                "extra": {
                    "actions": [{"bash_command": raw_response["bash_command"]}],
                    "raw_response": raw_response,
                },
            }
        ]
    )

    assistant_text = serialized[0]["content"][0]["text"]
    assert json.loads(assistant_text) == raw_response
    assert raw_response["bash_command"] in assistant_text


_MULTI_BLOCK_SFT_STATE = """<think>
Create the persistent browser session.
</think>
<bash>
python -m browser_session create --workspace-dir "/workspace"
</bash>
<done>false</done>
<final_response></final_response>

<think>
Now inspect the page.
</think>
<bash>
python -m browser_session step --workspace-dir "/workspace" --action "inspect" --code-file - <<'PY'
print(await page.title())
PY
</bash>
<done>false</done>
<final_response></final_response>

<think>
The task is complete.
</think>
<bash>
</bash>
<done>true</done>
<final_response>All done.</final_response>"""


def test_parse_sft_state_output_first_block_stops_at_first_state() -> None:
    parsed = parse_sft_state_output(_MULTI_BLOCK_SFT_STATE, first_block=True)

    assert parsed["done"] is False
    assert parsed["bash_command"] == 'python -m browser_session create --workspace-dir "/workspace"'
    assert parsed["final_response"] == ""


def test_parse_sft_state_output_default_keeps_last_block() -> None:
    parsed = parse_sft_state_output(_MULTI_BLOCK_SFT_STATE)

    assert parsed["done"] is True
    assert parsed["bash_command"] == ""
    assert parsed["final_response"] == "All done."


@pytest.mark.parametrize(
    "raw",
    [
        "<think>\nt\n</think>\n<bash>\nls\n</bash>\n<done>false</done>\n<final_response></final_response>",
        "<think>\nt\n</think>\n<bash>\n</bash>\n<done>true</done>\n<final_response>42</final_response>",
        "prefill thought\n</think>\n<bash>\nls\n</bash>\n<done>false</done>\n<final_response></final_response>",
    ],
)
def test_parse_sft_state_output_first_block_is_a_noop_for_single_blocks(raw: str) -> None:
    assert parse_sft_state_output(raw, first_block=True) == parse_sft_state_output(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "<think>x</think>",
        "<think>x</think><bash>ls</bash><done>true</done><final_response>y</final_response>",
        "<think>x</think><bash></bash><done>false</done><final_response></final_response>",
    ],
)
def test_parse_sft_state_output_first_block_still_rejects_invalid_states(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_sft_state_output(raw, first_block=True)


def _json_message_payload_with_reasoning(text: str, summary: list | None) -> dict:
    reasoning_item: dict = {"type": "reasoning", "summary": summary or []}
    return {
        "output": [
            reasoning_item,
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 4},
        },
    }


_JSON_STEP = '{"thought":"t","bash_command":"ls","done":false,"final_response":""}'


def test_query_writes_reasoning_summary_to_task_folder(monkeypatch, tmp_path) -> None:
    fake_cls, _ = _make_fake_async_openai(
        _json_message_payload_with_reasoning(
            _JSON_STEP, [{"type": "summary_text", "text": "**Plan**\nDo the thing."}]
        )
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="json_schema",
        reasoning_effort="xhigh",
        error_log_path=tmp_path / "runtime_errors.jsonl",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["extra"]["reasoning_summary"] == "**Plan**\nDo the thing."

    log_path = tmp_path / "reasoning.jsonl"
    assert log_path.exists()
    record = json.loads(log_path.read_text().strip())
    assert record["event"] == "reasoning_summary"
    assert record["reasoning_summary"] == "**Plan**\nDo the thing."
    assert record["reasoning_effort"] == "xhigh"
    assert record["reasoning_tokens"] == 4


def test_query_skips_reasoning_log_when_no_summary_returned(monkeypatch, tmp_path) -> None:
    fake_cls, _ = _make_fake_async_openai(
        _json_message_payload_with_reasoning(_JSON_STEP, [])
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="json_schema",
        reasoning_effort="xhigh",
        error_log_path=tmp_path / "runtime_errors.jsonl",
    )
    message = model.query([{"role": "user", "content": "Task"}])

    assert message["extra"]["reasoning_summary"] == ""
    assert not (tmp_path / "reasoning.jsonl").exists()


def test_reasoning_summary_is_not_replayed_into_model_context(monkeypatch, tmp_path) -> None:
    fake_cls, state = _make_fake_async_openai(
        _json_message_payload_with_reasoning(
            _JSON_STEP, [{"type": "summary_text", "text": "SECRET_REASONING"}]
        )
    )
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="json_schema",
        reasoning_effort="xhigh",
        error_log_path=tmp_path / "runtime_errors.jsonl",
    )
    assistant = model.query([{"role": "user", "content": "Task"}])
    model.query([{"role": "user", "content": "Task"}, assistant])

    assert "SECRET_REASONING" not in json.dumps(state["last_kwargs"]["input"])


def _truncated_payload(max_output_tokens: int = 16000) -> dict:
    return {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "reasoning", "summary": []}],
        "usage": {
            "input_tokens": 5000,
            "output_tokens": max_output_tokens,
            "total_tokens": 5000 + max_output_tokens,
            "output_tokens_details": {"reasoning_tokens": max_output_tokens},
        },
    }


def test_is_output_truncated_detects_budget_exhaustion() -> None:
    assert _is_output_truncated(_truncated_payload(), 16000) is True
    # same shape but without the status field: inferred from token accounting
    payload = _truncated_payload()
    del payload["status"], payload["incomplete_details"]
    assert _is_output_truncated(payload, 16000) is True


def test_is_output_truncated_ignores_normal_responses() -> None:
    payload = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 500,
            "output_tokens_details": {"reasoning_tokens": 400},
        },
    }
    assert _is_output_truncated(payload, 16000) is False
    assert _is_output_truncated({}, 16000) is False


def test_truncated_response_fails_fast_without_retrying(monkeypatch, tmp_path) -> None:
    # A truncated response is unchanged by retrying, so the model must give up
    # after a single call instead of burning MAX_JSON_PARSE_RETRIES more budgets.
    fake_cls, state = _make_fake_async_openai(_truncated_payload())
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="json_schema",
        reasoning_effort="xhigh",
        max_output_tokens=16000,
        format_error_template="error: {{ error }}",
        error_log_path=tmp_path / "runtime_errors.jsonl",
    )
    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "Task"}])

    assert state["calls"] == 1


def test_unparseable_but_untruncated_response_still_retries(monkeypatch, tmp_path) -> None:
    payload = {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "not json"}]}],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 10},
        },
    }
    fake_cls, state = _make_fake_async_openai(payload)
    monkeypatch.setattr("miniswewebagent.models.phyagi_model.AsyncOpenAI", fake_cls)

    model = PhyagiModel(
        openai_gateway_api_key="dummy",
        response_mode="json_schema",
        max_output_tokens=16000,
        format_error_template="error: {{ error }}",
        error_log_path=tmp_path / "runtime_errors.jsonl",
    )
    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "Task"}])

    assert state["calls"] == MAX_JSON_PARSE_RETRIES + 1

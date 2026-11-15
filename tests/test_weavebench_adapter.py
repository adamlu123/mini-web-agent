from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cuawright.config.prompts import actor_user_prompt, compose_system_prompt
from cuawright.environments.weavebench import adapter as weavebench
from cuawright.environments.weavebench import prompts as weavebench_prompts
from cuawright.environments.control_clients import CLIENT_FILENAMES
from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA
from cuawright.models.openai_responses import (
    OpenAIResponsesAdapter,
    ResponsesThread,
)


def _source_prompt() -> str:
    return (
        "Preserve this exact task contract.\n"
        "Built-in helper tools available in BOTH modes: `exec` (shell), "
        "`read`/`write`/`edit` (files), `process` (long-running jobs), "
        "`pdf` (structured PDF extraction including tables), `web_fetch` "
        "(HTTP), `memory_search` (recall prior context). Pick whichever is "
        "most direct for the sub-step.\n"
        "Working directory: `/tmp_workspace/`; Do NOT touch `/tmp_workspace/gt/`. "
        "The OpenClaw framework's own `## Workspace` section in the underlying "
        "system prompt has been configured to point at the SAME `/tmp_workspace/` "
        "path (via `agents.defaults.workspace`), so the two sources agree. "
        "Anything you write under `/home/user/.openclaw/...` is OpenClaw's "
        "internal agent state — invisible to the grader and effectively lost.\n"
        "Output discipline: verify required output "
        "(use `exec` like `ls -la <path>` or `read` a few of the created files).\n"
        'The sudo password is "password" when sudo is needed.\n'
        "The screen resolution is 1920x1080 pixels.\n"
        "An initial desktop screenshot was saved to /tmp/init_screenshot.png; "
        "you can read it with the `image` tool, or call `__computer__` with "
        "`actions:[]` (empty batch) to grab a fresh capture.\n"
        "- GUI is exposed via ONE function tool: `__computer__`. It runs one or "
        "more GUI actions and ALWAYS returns a fresh screenshot.\n"
        "- You MUST drive the GUI by calling `__computer__`.\n"
        "- DO NOT stop after a single observation from `__computer__`.\n"
        "- Write supporting artifacts via `exec`/`write`.\n"
        "- Coordinates are absolute screen pixels in the 1920x1080 system.\n"
        "=== GUI MODE ANTI-FABRICATION POLICY ===\n"
        "This run has a real Linux desktop available via the `__computer__` tool.\n"
        "Do NOT generate placeholder PNG files using `PIL.Image.new()`.\n"
        "  • `__computer__` — preferred. Every call already auto-screenshots\n"
        "    the resulting desktop; you can save that buffer to disk.\n"
        "If `pyautogui.screenshot()` or `__computer__` keeps failing, stop.\n"
    )


def test_prompt_translation_preserves_constraints_and_fails_closed(monkeypatch):
    translated = weavebench.translate_system_prompt(
        _source_prompt(), client_password="public-evaluation"
    )
    for required in (
        "Preserve this exact task contract.",
        "/tmp_workspace/",
        "/tmp_workspace/gt/",
        "Do NOT touch",
        "Output discipline",
        "1920x1080",
        "GUI MODE ANTI-FABRICATION POLICY",
        "Do NOT generate placeholder PNG",
        "sole model-visible tool is `run_command`",
        "/home/user/.agent/",
        "DISPLAY=:0",
        "pyautogui",
        "xdotool",
        "accessibility",
        'sudo password is "public-evaluation"',
    ):
        assert required in translated
    for unavailable in (
        "`exec`",
        "`read`",
        "`write`",
        "`edit`",
        "`process`",
        "`pdf`",
        "`web_fetch`",
        "`memory_search`",
        "`image`",
        "__computer__",
    ):
        assert unavailable not in translated
    assert set(weavebench.localized_role_instructions()) == {
        "actor",
        "binder",
        "critic",
        "verifier",
    }
    assert all(
        "OSWorld V2" not in text
        for text in weavebench.localized_role_instructions().values()
    )
    with pytest.raises(RuntimeError, match="expected 1 exact occurrence"):
        weavebench.translate_system_prompt(
            _source_prompt().replace("Built-in helper tools available", "Helper tools")
        )
    monkeypatch.setattr(
        weavebench_prompts,
        "ROLE_INSTRUCTIONS",
        {"actor": "drift"},
    )
    with pytest.raises(RuntimeError, match="re-audit"):
        weavebench_prompts.localized_role_instructions(
            weavebench_prompts.ROLE_INSTRUCTIONS
        )


def test_memfd_secret_is_host_memory_only_and_always_closed(monkeypatch):
    observed = None
    with weavebench._api_key_memfd("host-secret") as path:
        observed = path
        assert path.startswith("/proc/self/fd/")
        assert Path(path).read_text() == "host-secret"
    assert observed is not None and not Path(observed).exists()

    monkeypatch.delattr(weavebench.os, "memfd_create")
    with pytest.raises(RuntimeError, match="memfd_create"):
        with weavebench._api_key_memfd("host-secret"):
            pass


def test_campaign_model_and_base_are_validated():
    bare = weavebench.CUAWrightAgent(litellm_api_key="key")
    assert bare.model == "gpt-5.5"
    assert bare.litellm_base_url == "https://maif-gateway.azure-api.net/api"
    alias = weavebench.CUAWrightAgent(
        model="openai/gpt-5.5",
        litellm_base_url="https://maif-gateway.azure-api.net/api/",
        litellm_api_key="key",
    )
    assert alias.model == "gpt-5.5"
    pinned_runner = weavebench.CUAWrightAgent(
        model="gpt-5.5",
        litellm_base_url="https://gateway.phyagi.net/api",
        litellm_api_key="key",
    )
    assert pinned_runner.litellm_base_url == "https://maif-gateway.azure-api.net/api"
    with pytest.raises(ValueError, match="AIServices.*OpenRouter"):
        weavebench.CUAWrightAgent(
            litellm_base_url="https://trapi.research.microsoft.com/gcr/shared/openai/v1",
            litellm_api_key="key",
        )
    with pytest.raises(ValueError, match="AIServices.*OpenRouter"):
        weavebench.CUAWrightAgent(
            litellm_base_url="https://openrouter.ai/api/v1",
            litellm_api_key="key",
        )
    with pytest.raises(ValueError, match="campaign model"):
        weavebench.CUAWrightAgent(model="gpt-5.6-sol", litellm_api_key="key")
    with pytest.raises(ValueError, match="AIServices.*OpenRouter"):
        weavebench.CUAWrightAgent(
            model="openai/gpt-5.5",
            litellm_base_url="https://openrouter.ai/api/v1",
            litellm_api_key="key",
        )
    with pytest.raises(ValueError, match="does not support native CUA"):
        weavebench.CUAWrightAgent(
            litellm_api_key="key",
            cua_native=True,
        )


def test_no_subagents_selects_linear_prompt_bundle():
    agent = weavebench.CUAWrightAgent(
        litellm_api_key="key",
        no_subagents=True,
    )

    assert agent.no_subagents
    assert agent._prompt_bundle(_source_prompt()).name == "weavebench-linear"


def test_wall_timeout_can_be_overridden_for_recovery(monkeypatch):
    monkeypatch.setenv("CUAWRIGHT_WEAVEBENCH_WALL_TIMEOUT_SECONDS", "7200")
    agent = weavebench.CUAWrightAgent(
        litellm_api_key="key",
        timeout=3600,
    )

    assert agent.timeout == 7200

    monkeypatch.setenv("CUAWRIGHT_WEAVEBENCH_WALL_TIMEOUT_SECONDS", "invalid")
    with pytest.raises(ValueError, match="must be an integer"):
        weavebench.CUAWrightAgent(litellm_api_key="key")


def test_wall_timeout_escapes_exception_normalization_and_runs_finally(monkeypatch):
    installed, timers, cleaned = {}, [], []

    def fake_signal(kind, handler):
        previous = installed.get(kind, "previous")
        installed[kind] = handler
        return previous

    monkeypatch.setattr(weavebench.signal, "signal", fake_signal)
    monkeypatch.setattr(
        weavebench.signal,
        "setitimer",
        lambda kind, seconds: timers.append((kind, seconds)),
    )

    def request_exception_normalizer():
        try:
            installed[weavebench.signal.SIGALRM](weavebench.signal.SIGALRM, None)
        except Exception as exc:
            raise RuntimeError("requests normalized the timeout") from exc
        finally:
            cleaned.append(True)

    with pytest.raises(weavebench.CUAWrightWallTimeout):
        with weavebench._wall_timeout(9):
            request_exception_normalizer()
    assert cleaned == [True]
    assert timers == [
        (weavebench.signal.ITIMER_REAL, 9),
        (weavebench.signal.ITIMER_REAL, 0),
    ]
    assert installed[weavebench.signal.SIGALRM] == "previous"


def _write_gzip(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_nested_specialist_trace_export_preserves_ids_order_usage_and_reasoning(
    tmp_path,
):
    native = tmp_path / "native"
    for step, created, call_id, summary in (
        (1, 100, "outer-binder", "Delegate carefully."),
        (2, 101, "nested-command", "Inspect evidence."),
    ):
        _write_gzip(
            native / "responses" / f"step-{step:04d}.json.gz",
            {
                "model": "gpt-5.5",
                "created_at": created,
                "usage": {
                    "input_tokens": 10 + step,
                    "output_tokens": 5 + step,
                    "total_tokens": 15 + 2 * step,
                    "input_tokens_details": {"cached_tokens": step},
                    "output_tokens_details": {"reasoning_tokens": step + 1},
                },
                "output": [
                    {"type": "reasoning", "summary": [{"text": summary}]},
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "run_command",
                        "arguments": "{}",
                    },
                ],
            },
        )
    events = [
        {
            "timestamp": "1970-01-01T00:01:40.1+00:00",
            "event": "model_turn",
            "tool_calls": [{"call_id": "outer-binder"}],
        },
        {
            "timestamp": "1970-01-01T00:01:40.2+00:00",
            "event": "terminal_command",
            "command": "python /opt/cuawright-tools/binder.py --request x --max-steps 3",
            "timeout": 30,
        },
        {
            "timestamp": "1970-01-01T00:01:41.1+00:00",
            "event": "model_turn",
            "tool_calls": [{"call_id": "nested-command"}],
        },
        {
            "timestamp": "1970-01-01T00:01:41.2+00:00",
            "event": "terminal_command",
            "command": "cat evidence.txt",
            "timeout": 30,
        },
        {
            "timestamp": "1970-01-01T00:01:41.3+00:00",
            "event": "command_result",
            "call_id": "nested-command",
            "role": "binder",
            "stage": "specialist-binder-01",
            "exit_code": 0,
            "stdout": "nested evidence",
            "stderr": "",
            "completed": False,
            "inspected": True,
        },
        {
            "timestamp": "1970-01-01T00:01:41.4+00:00",
            "event": "command_result",
            "call_id": "outer-binder",
            "role": "actor",
            "stage": "orchestrate",
            "exit_code": 0,
            "stdout": "binder report",
            "stderr": "",
            "completed": False,
            "inspected": False,
        },
    ]
    (native / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    chat = tmp_path / "chat.jsonl"
    weavebench.export_cuawright_chat(native, chat)
    records = [json.loads(line) for line in chat.read_text().splitlines()]
    sequence, results = [], {}
    for record in records:
        message = record.get("message", {})
        for block in message.get("content", []):
            if block.get("type") == "tool_use":
                sequence.append(("use", block["id"]))
            elif block.get("type") == "tool_result":
                sequence.append(("result", block["tool_use_id"]))
                text = next(
                    item["text"]
                    for item in block["content"]
                    if item.get("type") == "text"
                )
                results[block["tool_use_id"]] = json.loads(text)
    assert sequence == [
        ("use", "outer-binder"),
        ("use", "nested-command"),
        ("result", "nested-command"),
        ("result", "outer-binder"),
    ]
    assert results["nested-command"]["command"] == "cat evidence.txt"
    assert results["nested-command"]["stdout"] == "nested evidence"
    assistants = [r["message"] for r in records if r["message"]["role"] == "assistant"]
    assert assistants[0]["usage"]["input"] == 10
    assert assistants[0]["usage"]["cacheRead"] == 1
    assert assistants[0]["usage"]["totalTokens"] == 17
    assert assistants[0]["usage"]["reasoning"] == 2
    assert assistants[0]["content"][0] == {
        "type": "thinking",
        "thinking": "Delegate carefully.",
    }
    assert (native / "responses" / "step-0001.json.gz").is_file()
    assert not (tmp_path / "results.tar.gz").exists()


def test_usage_arithmetic_clamps_uncached_input():
    assert weavebench._usage(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 7,
                "total_tokens": 107,
                "input_tokens_details": {"cached_tokens": 30},
            }
        }
    ) == {"input": 70, "cacheRead": 30, "output": 7, "totalTokens": 107}
    assert (
        weavebench._usage(
            {
                "usage": {
                    "input_tokens": 20,
                    "total_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 25},
                }
            }
        )["input"]
        == 0
    )


class _Trace:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.calls = []

    def command_blob(self, step, text):
        path = self.run_dir / "command-output" / f"{step}.txt.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
        return str(path.relative_to(self.run_dir)), "digest"

    def event(self, name, payload):
        self.calls.append((name, payload))

    def model_call(self, request, response):
        self.calls.append(("model_call", request, response))


class _Env:
    is_environment_used = False

    def __init__(self):
        self.controller = SimpleNamespace()

    def __getattr__(self, name):
        if name in {"close", "reset", "evaluate", "archive", "judge"}:
            raise AssertionError(f"adapter attempted VM-owned operation {name}")
        raise AttributeError(name)


def test_golden_run_maps_budget_command_bridge_and_closes_only_runner(
    tmp_path, monkeypatch
):
    captured, requests = {}, []

    class Response:
        def model_dump(self, mode="json"):
            return {"model": "gpt-5.5", "created_at": 1, "output": []}

    class Runner:
        def __init__(self, config):
            captured["config"] = config
            captured["key_path"] = config.api_key_file
            assert Path(config.api_key_file).read_text() == "host-secret"
            self.run_dir = Path(config.result_root) / "native-run"
            (self.run_dir / "responses").mkdir(parents=True)
            self.trace = _Trace(self.run_dir)
            self.closed = False
            captured["runner"] = self

        def run_terminal_plugin(self, plugin, *, instruction, prompt_bundle):
            captured["plugin"] = plugin
            captured["prompt_bundle"] = prompt_bundle
            result = plugin.run_command("pwd", 300)
            assert result.exit_code == 0 and result.inspected
            client = SimpleNamespace(
                responses=SimpleNamespace(
                    create=lambda **kwargs: requests.append(kwargs) or Response()
                )
            )
            thread = ResponsesThread(
                client=client,
                trace=self.trace,
                model=self.config.model,
                reasoning_effort=self.config.reasoning_effort,
                role="weavebench-actor",
                system_prompt=compose_system_prompt("actor", prompt_bundle),
                user_prompt=actor_user_prompt(instruction),
                tools=[],
                max_output_tokens=32768,
                base_url=self.config.base_url,
                infrastructure=self.config.model_infrastructure,
            )
            OpenAIResponsesAdapter(thread).next_turn(tools=(RUN_COMMAND_TOOL_SCHEMA,))

        @property
        def config(self):
            return captured["config"]

        def close(self):
            self.closed = True

    guest_calls = []

    def guest_command(controller, script, timeout):
        guest_calls.append((controller, script, timeout))
        return {"status": "success", "returncode": 0, "output": "/tmp_workspace\n"}

    monkeypatch.setattr(weavebench, "TaskRunner", Runner)
    monkeypatch.setattr(weavebench, "run_guest_command", guest_command)
    env = _Env()
    agent = weavebench.CUAWrightAgent(
        litellm_api_key="host-secret",
        max_steps=200,
        task_id="WEB_task_1",
        timeout=30,
    )
    status = agent.run(
        env,
        "Do the staged task.",
        tmp_path,
        system_prompt_override=_source_prompt(),
        already_configured=True,
    )
    config = captured["config"]
    assert (
        config.model,
        config.reasoning_effort,
        config.total_steps,
        config.model_provider,
        config.model_infrastructure,
        config.compact_every,
    ) == ("gpt-5.5", "high", 200, "openai", "aiservices", 40)
    request = requests[0]
    assert [tool["name"] for tool in request["tools"]] == ["run_command"]
    assert request["reasoning"] == {"effort": "high", "summary": "auto"}
    assert request["extra_body"]["infrastructure"] == "aiservices"
    assert request["parallel_tool_calls"] is False
    assert request["store"] is False
    assert captured["prompt_bundle"].name == "weavebench-default"
    assert "Do the staged task." in request["input"][1]["content"]
    assert "WeaveBench task" in request["input"][0]["content"]
    assert "/tmp_workspace" in guest_calls[0][1]
    assert "$HOME/.local/bin" in guest_calls[0][1]
    assert "export DISPLAY=:0" in guest_calls[0][1]
    assert ". /tmp/openclaw_task_env.sh" in guest_calls[0][1]
    assert guest_calls[0][2] == 300
    assert env.is_environment_used is True
    assert captured["runner"].closed
    assert not Path(captured["key_path"]).exists()
    assert status["exit_code"] == 0 and status["agent_done"] is True
    assert (tmp_path / "agent.log").read_text().startswith("AGENT_EXIT=0\n")
    assert (tmp_path / "execution_status.json").is_file()
    assert not (tmp_path / "results.tar.gz").exists()
    artifact_text = []
    for path in tmp_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                artifact_text.append(handle.read())
        else:
            artifact_text.append(path.read_text(encoding="utf-8", errors="ignore"))
    assert "host-secret" not in "".join(artifact_text)


def test_golden_run_accepts_configured_reasoning_effort(tmp_path, monkeypatch):
    captured = {}

    class Runner:
        def __init__(self, config):
            captured["config"] = config
            self.run_dir = Path(config.result_root) / "native-run"
            self.run_dir.mkdir(parents=True)
            self.trace = _Trace(self.run_dir)

        def run_terminal_plugin(self, *args, **kwargs):
            return None

        def close(self):
            pass

    monkeypatch.setenv("CUAWRIGHT_WEAVEBENCH_REASONING_EFFORT", "xhigh")
    monkeypatch.setattr(weavebench, "TaskRunner", Runner)
    monkeypatch.setattr(
        weavebench,
        "run_guest_command",
        lambda *args, **kwargs: {
            "status": "success",
            "returncode": 0,
            "output": "",
            "error": "",
        },
    )

    status = weavebench.CUAWrightAgent(
        litellm_api_key="key",
        task_id="WEB_task_1",
    ).run(
        _Env(),
        "Task.",
        tmp_path,
        system_prompt_override=_source_prompt(),
        already_configured=True,
    )

    assert status["exit_code"] == 0
    assert captured["config"].reasoning_effort == "xhigh"


def test_run_exports_linked_image_evidence_without_base64(tmp_path, monkeypatch):
    image = b"\x89PNG\r\n\x1a\nmock-observation"
    digest = __import__("hashlib").sha256(image).hexdigest()
    observed = {}

    class Runner:
        def __init__(self, config):
            self.run_dir = Path(config.result_root) / "image-run"
            (self.run_dir / "responses").mkdir(parents=True)
            (self.run_dir / "images").mkdir()
            self.trace = _Trace(self.run_dir)
            observed["runner"] = self

        def run_terminal_plugin(self, *args, **kwargs):
            _write_gzip(
                self.run_dir / "responses" / "step-0001.json.gz",
                {
                    "model": "gpt-5.5",
                    "created_at": 100,
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "total_tokens": 15,
                    },
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "image-call",
                            "name": "run_command",
                            "arguments": json.dumps(
                                {
                                    "command": (
                                        "python /opt/cuawright-tools/image_show.py "
                                        "--path /tmp/screen.png"
                                    ),
                                    "timeout": 30,
                                }
                            ),
                        }
                    ],
                },
            )
            (self.run_dir / "images" / f"{digest}.png").write_bytes(image)
            events = [
                {
                    "timestamp": "1970-01-01T00:01:40.1+00:00",
                    "event": "model_turn",
                    "tool_calls": [{"call_id": "image-call"}],
                },
                {
                    "timestamp": "1970-01-01T00:01:40.2+00:00",
                    "event": "image_attached",
                    "source_path": "/tmp/screen.png",
                    "media_type": "image/png",
                    "size_bytes": len(image),
                    "sha256": digest,
                },
                {
                    "timestamp": "1970-01-01T00:01:40.3+00:00",
                    "event": "terminal_command",
                    "command": (
                        "python /opt/cuawright-tools/image_show.py "
                        "--path /tmp/screen.png"
                    ),
                    "timeout": 30,
                },
                {
                    "timestamp": "1970-01-01T00:01:40.4+00:00",
                    "event": "command_result",
                    "call_id": "image-call",
                    "role": "actor",
                    "stage": "orchestrate",
                    "exit_code": 0,
                    "stdout": "{}",
                    "stderr": "",
                    "attachments": [
                        {
                            "kind": "image",
                            "media_type": "image/png",
                            "source_path": "/tmp/screen.png",
                            "size_bytes": len(image),
                        }
                    ],
                    "completed": False,
                    "inspected": True,
                },
            ]
            (self.run_dir / "trace.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(weavebench, "TaskRunner", Runner)
    status = weavebench.CUAWrightAgent(litellm_api_key="key").run(
        _Env(),
        "inspect the real desktop",
        tmp_path,
        system_prompt_override=_source_prompt(),
        already_configured=True,
    )
    assert status["exit_code"] == 0
    assert observed["closed"]
    copied = tmp_path / "trajectory_images" / f"{digest}.png"
    assert copied.read_bytes() == image
    manifest = json.loads(
        (tmp_path / "trajectory_images" / "manifest.json").read_text()
    )
    assert manifest == [
        {
            "tool_use_id": "image-call",
            "path": f"trajectory_images/{digest}.png",
            "media_type": "image/png",
            "sha256": digest,
            "source_path": "/tmp/screen.png",
        }
    ]
    chat = (tmp_path / "chat.jsonl").read_text()
    assert "image-call" in chat
    assert f"trajectory_images/{digest}.png" in chat
    assert '"type": "file"' in chat
    assert "base64" not in chat
    assert not (tmp_path / "results.tar.gz").exists()


@pytest.mark.parametrize(
    ("failure", "exit_code", "timed_out"),
    [
        (RuntimeError("host-secret failure"), 1, False),
        (weavebench.CUAWrightWallTimeout("deadline"), 124, True),
    ],
)
def test_failure_and_timeout_close_runner_memfd_and_redact(
    tmp_path, monkeypatch, failure, exit_code, timed_out
):
    observed = {}

    class Runner:
        def __init__(self, config):
            observed["path"] = config.api_key_file
            self.run_dir = Path(config.result_root) / "partial-run"
            self.run_dir.mkdir(parents=True)
            self.trace = _Trace(self.run_dir)
            observed["runner"] = self

        def run_terminal_plugin(self, *args, **kwargs):
            raise failure

        def close(self):
            observed["closed"] = True

    monkeypatch.setattr(weavebench, "TaskRunner", Runner)
    agent = weavebench.CUAWrightAgent(litellm_api_key="host-secret", timeout=30)
    status = agent.run(
        _Env(),
        "task",
        tmp_path,
        system_prompt_override=_source_prompt(),
        already_configured=True,
    )
    assert observed["closed"]
    assert not Path(observed["path"]).exists()
    assert status["exit_code"] == exit_code
    assert status["timed_out"] is timed_out
    persisted = (
        (tmp_path / "agent.log").read_text()
        + (tmp_path / "execution_status.json").read_text()
        + (tmp_path / "chat.jsonl").read_text()
    )
    assert "host-secret" not in persisted
    if not timed_out:
        assert "******" in persisted


def test_bootstrap_provisions_only_clients_osw_and_knowledge(monkeypatch):
    assert CLIENT_FILENAMES == (
        "binder.py",
        "critic.py",
        "verifier.py",
        "image_show.py",
        "ask_user.py",
        "submit.py",
    )
    provisioned, writes = [], []
    monkeypatch.setattr(
        weavebench,
        "_provision_control_clients",
        lambda controller, setup: provisioned.append((controller, setup)),
    )
    monkeypatch.setattr(
        weavebench,
        "_write_guest_file",
        lambda controller, path, content, mode: writes.append(
            (controller, path, content, mode)
        ),
    )

    class Controller:
        def run_python_script(self, script):
            return {"output": script}

        def run_bash_script(self, *args, **kwargs):
            raise AssertionError("WeaveBench bootstrap must not run Bash/packages")

    controller = Controller()
    setup = SimpleNamespace()
    env = SimpleNamespace(controller=controller, setup_controller=setup)
    weavebench.CUAWrightAgent(litellm_api_key="key").bootstrap(env)
    compatible, compatible_setup = provisioned[0]
    assert compatible.run_python_script("code", timeout=1) == {"output": "code"}
    assert compatible_setup._controller is setup
    expected = [
        (
            "/home/user/.local/bin/osw",
            weavebench.OSW_SCRIPT.encode("utf-8"),
            0o755,
        ),
        *[
            (
                f"/home/user/.local/share/osw/kb/{name}",
                content.encode("utf-8"),
                0o644,
            )
            for name, content in weavebench.WEAVEBENCH_KNOWLEDGE_FILES.items()
        ],
    ]
    assert [(path, content, mode) for _, path, content, mode in writes] == expected
    assert all(item[0] is compatible for item in writes)
    assert "OSWorld provisions Playwright" not in "".join(
        weavebench.WEAVEBENCH_KNOWLEDGE_FILES.values()
    )
    assert "availability varies by task image" in (
        weavebench.WEAVEBENCH_KNOWLEDGE_FILES["browser.md"]
    )


def test_setup_controller_adapts_execute_endpoint(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"returncode": 0, "output": "ok"}

    monkeypatch.setattr(
        weavebench.requests,
        "post",
        lambda url, *, json, timeout: calls.append((url, json, timeout)) or Response(),
    )
    setup = SimpleNamespace(
        http_server="http://guest:5000",
        client_password="public",
        screen_width=1920,
        screen_height=1080,
    )
    result = weavebench.CompatibleSetupController(setup).execute(
        ["sh", "-c", "echo {CLIENT_PASSWORD}:{SCREEN_WIDTH_HALF}"], timeout=20
    )
    assert result["output"] == "ok"
    assert calls == [
        (
            "http://guest:5000/setup/execute",
            {"command": ["sh", "-c", "echo public:960"], "shell": False},
            30,
        )
    ]

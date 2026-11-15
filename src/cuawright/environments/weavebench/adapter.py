"""Host-side CUAWright adapter for an existing WeaveBench DesktopEnv."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import signal
import shutil
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from cuawright.config.prompts import BenchmarkOverlay, PromptBundle
from cuawright.environments.base import CommandResult
from cuawright.environments.desktop.commands import (
    is_inspection_command as is_desktop_inspection_command,
)
from cuawright.environments.desktop.guest_commands import run_guest_command
from cuawright.environments.desktop.guest_tools import (
    provision_control_clients,
    write_guest_file,
)
from cuawright.environments.desktop.knowledge import DESKTOP_KNOWLEDGE_SCRIPT
from cuawright.environments.desktop.plugin import (
    CommandExecutor,
    DesktopCommandPlugin,
    GuestWorkspace,
)
from cuawright.environments.terminal import TaskConfig, TaskRunner
from cuawright.environments.weavebench.prompts import (
    localized_linear_actor_instructions,
    localized_role_instructions,
    translate_system_prompt,
)
from cuawright.config.prompt_registry import select_weavebench_prompt_bundle

# Black is disabled to avoid turning canonical chat records into hundreds of
# incidental lines. Prompt sources and exact replacements live in prompts.py.
# fmt: off

logger = logging.getLogger("cuawright.environments.weavebench")
AGENT_WORKSPACE = "/home/user/.agent"
GUEST_WORKSPACE = "/tmp_workspace"
MODEL = "gpt-5.5"
DEFAULT_BASE_URL = "https://maif-gateway.azure-api.net/api"
_PINNED_RUNNER_BASE_URL = "https://gateway.phyagi.net/api"
COMMAND_TIMEOUT_MAX = 300
OSW_SCRIPT = DESKTOP_KNOWLEDGE_SCRIPT
_provision_control_clients = provision_control_clients
_write_guest_file = write_guest_file

WEAVEBENCH_KNOWLEDGE_FILES = {
    "browser.md": """# Browser tasks

## Inspect existing state first

Preserve the website, account, profile, and tab opened by task warmup. Inspect
the current Chrome processes, windows, and pages before launching or restarting
a browser. Browser and Python package availability varies by task image.

Probe the common guest-local CDP endpoints when developer tools are allowed:

```bash
curl -fsS http://127.0.0.1:1337/json/version ||
  curl -fsS http://127.0.0.1:9222/json/version
```

Before using Playwright, locate a working interpreter rather than assuming the
OSWorld virtual environment exists:

```bash
for python in python3 /home/user/.local/share/osw/venv/bin/python; do
  [ -x "$python" ] && "$python" -c 'import playwright' && echo "$python"
done
```

If no CDP session is usable, inspect the real `DISPLAY=:0` desktop and use
visible UI control. Launch or repair Chrome only after confirming that no valid
task session would be replaced. If the task forbids CDP or developer tools, use
visible UI only.
""",
    "native-apps.md": """# Native desktop applications

Use the real `DISPLAY=:0` desktop through accessibility inspection, pyautogui,
xdotool, and ordinary application commands. Capture genuine evidence with:

```bash
DISPLAY=:0 python3 - <<'PY'
import pyautogui
pyautogui.screenshot('/tmp_workspace/screen.png')
PY
```

Inspect screenshots with `python /opt/cuawright-tools/image_show.py --path
ABSOLUTE_PATH`. When an artifact is edited externally while an application has
it open, save it, reopen the canonical path in the named application, and
verify persistent state before submission.
""",
}


class CUAWrightWallTimeout(BaseException):
    pass


def normalize_model_and_base(model: str, base_url: str) -> tuple[str, str]:
    normalized_base = (base_url or "").rstrip("/")
    # The pinned external launcher still supplies its original gateway URL.
    if normalized_base == _PINNED_RUNNER_BASE_URL:
        normalized_base = DEFAULT_BASE_URL
    if normalized_base != DEFAULT_BASE_URL:
        raise ValueError(
            "CUAWright WeaveBench uses AIServices infrastructure and requires "
            f"{DEFAULT_BASE_URL!r}; OpenRouter or another provider base is incompatible"
        )
    if model == "openai/gpt-5.5":
        logger.warning("normalizing exact WeaveBench model alias to bare gpt-5.5")
        model = MODEL
    if model != MODEL:
        raise ValueError(
            "CUAWright WeaveBench requires campaign model 'gpt-5.5' "
            "(or exact 'openai/gpt-5.5' with the Phyagi base)"
        )
    return model, normalized_base


class CompatibleController:
    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def run_python_script(self, script: str, timeout: int = 90) -> Any:
        del timeout
        return self._controller.run_python_script(script)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._controller, name)


class CompatibleSetupController:
    def __init__(self, controller: Any) -> None:
        self._controller = controller

    def execute(self, command: list[str], stdout: str = "", stderr: str = "",
                shell: bool = False, until: Mapping[str, Any] | None = None,
                quiet: bool = False, timeout: int = 120) -> Mapping[str, Any]:
        del stdout, stderr, until, quiet
        if not command:
            raise ValueError("setup command cannot be empty")
        values = {
            "{CLIENT_PASSWORD}": self._controller.client_password,
            "{SCREEN_WIDTH}": self._controller.screen_width,
            "{SCREEN_HEIGHT}": self._controller.screen_height,
            "{SCREEN_WIDTH_HALF}": self._controller.screen_width // 2,
            "{SCREEN_HEIGHT_HALF}": self._controller.screen_height // 2,
        }
        rendered = []
        for item in command:
            for old, new in values.items():
                item = item.replace(old, str(new))
            rendered.append(item)
        response = requests.post(self._controller.http_server + "/setup/execute",
                                 json={"command": rendered, "shell": shell},
                                 timeout=max(1, int(timeout)) + 10)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, Mapping):
            raise RuntimeError("WeaveBench setup endpoint returned invalid JSON")
        code = result.get("returncode")
        if isinstance(code, int) and not isinstance(code, bool) and code:
            detail = str(result.get("error") or result.get("output") or "")
            raise RuntimeError(f"WeaveBench privileged setup failed: {detail[:1000]}")
        return result


class _DesktopEnvView:
    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "controller", CompatibleController(env.controller))

    @property
    def is_environment_used(self) -> bool:
        return bool(getattr(self._env, "is_environment_used", False))

    @is_environment_used.setter
    def is_environment_used(self, value: bool) -> None:
        self._env.is_environment_used = value


class WeaveBenchDesktopPlugin(DesktopCommandPlugin):
    """WeaveBench command/file adapter without OSWorld lifecycle semantics."""

    def __init__(
        self,
        desktop_env: Any,
        execute_command: CommandExecutor,
        *,
        workspace: GuestWorkspace,
        bridge_timeout: int = 90,
    ) -> None:
        super().__init__(
            desktop_env,
            execute_command,
            workspace=workspace,
            bridge_timeout=bridge_timeout,
            bridge_error_label="OSWorld guest bridge",
        )


def _is_knowledge_query(argv: Sequence[str]) -> bool:
    if not argv or argv[0] != "osw":
        return False
    if argv in (["osw", "--help"], ["osw", "--list"]):
        return True
    if len(argv) in {3, 5} and argv[1] == "--show":
        if (
            not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*\.md",
                argv[2],
            )
            or Path(argv[2]).name != argv[2]
        ):
            return False
        return len(argv) == 3 or (argv[3] == "--heading" and bool(argv[4].strip()))
    query_count = 0
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--limit":
            if index + 1 >= len(argv):
                return False
            try:
                if int(argv[index + 1]) <= 0:
                    return False
            except ValueError:
                return False
            index += 2
            continue
        if argument.startswith("--limit="):
            try:
                if int(argument.partition("=")[2]) <= 0:
                    return False
            except ValueError:
                return False
            index += 1
            continue
        if argument.startswith("-"):
            return False
        query_count += 1
        index += 1
    return query_count == 1


def is_inspection_command(command: str) -> bool:
    return is_desktop_inspection_command(
        command,
        extra_inspector=_is_knowledge_query,
    )


@contextmanager
def _wall_timeout(seconds: int) -> Iterator[None]:
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1:
        raise ValueError("CUAWright wall timeout must be a positive integer")
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("CUAWright WeaveBench adapter requires SIGALRM")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("CUAWright wall timeout requires the host main thread")

    def handler(signum: int, frame: Any) -> None:
        del signum, frame
        raise CUAWrightWallTimeout(f"host wall timeout after {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def _api_key_memfd(secret: str) -> Iterator[str]:
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("Linux os.memfd_create is required; persistent key files are forbidden")
    fd = os.memfd_create("cuawright-api-key", getattr(os, "MFD_CLOEXEC", 0))
    try:
        data = memoryview(secret.encode())
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise OSError("short write while loading API key")
            data = data[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        yield f"/proc/self/fd/{fd}"
    finally:
        os.close(fd)


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "******") if secret else text


def _json_gzip(path: Path) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _usage(response: Mapping[str, Any]) -> dict[str, int]:
    raw = response.get("usage")
    if not isinstance(raw, Mapping):
        return {}
    result = {}
    details = raw.get("input_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
    cached = cached if isinstance(cached, int) and not isinstance(cached, bool) else 0
    input_tokens = raw.get("input_tokens")
    if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
        result["input"] = max(0, input_tokens - cached)
    result["cacheRead"] = cached
    for old, new in (("output_tokens", "output"), ("total_tokens", "totalTokens")):
        if isinstance(raw.get(old), int) and not isinstance(raw[old], bool):
            result[new] = raw[old]
    details = raw.get("output_tokens_details")
    if isinstance(details, Mapping) and isinstance(details.get("reasoning_tokens"), int):
        result["reasoning"] = details["reasoning_tokens"]
    return result


def _content(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = []
    for item in response.get("output", ()):
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "reasoning":
            content.extend({"type": "thinking", "thinking": summary["text"]}
                           for summary in item.get("summary", ())
                           if isinstance(summary, Mapping)
                           and isinstance(summary.get("text"), str))
        elif item.get("type") == "message":
            content.extend({"type": "text", "text": part["text"]}
                           for part in item.get("content", ())
                           if isinstance(part, Mapping)
                           and part.get("type") == "output_text"
                           and isinstance(part.get("text"), str))
        elif item.get("type") == "function_call":
            arguments = item.get("arguments", "{}")
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
            content.append({"type": "tool_use", "id": item.get("call_id"),
                            "name": item.get("name"), "input": arguments})
    return content


def _trace(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            events.append(event)
    return events


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except ValueError:
        return float("inf")


def export_cuawright_chat(native_run_dir: Path | None, chat_path: Path,
                           *, model: str = MODEL) -> None:
    """Convert native raw responses and trace events to WeaveBench chat JSONL."""
    records = []
    if native_run_dir is not None:
        events, pending, inputs = _trace(native_run_dir / "trace.jsonl"), [], {}
        pending_images, images_by_call = [], {}
        for event in events:
            if event.get("event") == "model_turn":
                pending.extend(call["call_id"] for call in event.get("tool_calls", ())
                               if isinstance(call, Mapping)
                               and isinstance(call.get("call_id"), str))
            elif event.get("event") == "terminal_command" and pending:
                inputs[pending.pop(0)] = {
                    "command": event.get("command"), "timeout": event.get("timeout")}
            elif event.get("event") == "image_attached":
                pending_images.append(event)
            elif event.get("event") == "command_result":
                call_id = event.get("call_id")
                if call_id in pending:
                    pending.remove(call_id)
                attachments = event.get("attachments")
                count = sum(1 for item in attachments
                            if isinstance(item, Mapping) and item.get("kind") == "image") \
                    if isinstance(attachments, list) else 0
                if isinstance(call_id, str) and count:
                    images_by_call[call_id] = pending_images[:count]
                    del pending_images[:count]
        image_refs = {}
        compatibility_dir = chat_path.parent / "trajectory_images"
        if images_by_call:
            if compatibility_dir.exists():
                shutil.rmtree(compatibility_dir)
            compatibility_dir.mkdir(parents=True)
            manifest = []
            for call_id, observations in images_by_call.items():
                for observation in observations:
                    digest = observation.get("sha256")
                    if not isinstance(digest, str):
                        raise RuntimeError("native image event omitted sha256")
                    matches = list((native_run_dir / "images").glob(f"{digest}.*"))
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"native image artifact {digest} was not uniquely available"
                        )
                    data = matches[0].read_bytes()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise RuntimeError(f"native image artifact {digest} failed integrity")
                    destination = compatibility_dir / matches[0].name
                    destination.write_bytes(data)
                    reference = {
                        "path": f"trajectory_images/{destination.name}",
                        "media_type": observation.get("media_type"),
                        "sha256": digest,
                        "source_path": observation.get("source_path"),
                    }
                    image_refs.setdefault(call_id, []).append(reference)
                    manifest.append({"tool_use_id": call_id, **reference})
            (compatibility_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        for path in sorted((native_run_dir / "responses").glob("step-*.json.gz")):
            response = _json_gzip(path)
            if response is None:
                continue
            message = {"role": "assistant", "model": response.get("model") or model,
                       "content": _content(response)}
            if usage := _usage(response):
                message["usage"] = usage
            records.append({"type": "message", "timestamp": response.get("created_at"),
                            "message": message,
                            "source": path.relative_to(native_run_dir).as_posix()})
        for event in events:
            if event.get("event") != "command_result":
                continue
            call_id = event.get("call_id")
            result = {**inputs.get(call_id, {})}
            result.update({key: event.get(key) for key in (
                "role", "stage", "exit_code", "stdout", "stderr", "completed", "inspected")})
            result["image_artifacts"] = image_refs.get(call_id, [])
            content = [{"type": "text",
                        "text": json.dumps(result, ensure_ascii=False, sort_keys=True)}]
            content.extend({
                "type": "image",
                "source": {
                    "type": "file",
                    "path": reference["path"],
                    "media_type": reference["media_type"],
                },
            } for reference in image_refs.get(call_id, []))
            records.append({
                "type": "message", "timestamp": event.get("timestamp"),
                "message": {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": call_id,
                    "content": content}]},
                "source": "trace.jsonl",
            })
    records = [item for _, item in sorted(
        enumerate(records), key=lambda pair: (_timestamp(pair[1].get("timestamp")), pair[0]))]
    if not records:
        records = [{"type": "run_status", "message": {
            "role": "system", "content": "CUAWright produced no model artifacts."}}]
    chat_path.parent.mkdir(parents=True, exist_ok=True)
    chat_path.write_text("".join(json.dumps(record, ensure_ascii=False, default=str) + "\n"
                                 for record in records), encoding="utf-8")


class CUAWrightAgent:
    """Duck-typed WeaveBench harness backed by CUAWright's TaskRunner."""
    def __init__(self, model: str = MODEL, litellm_base_url: str = DEFAULT_BASE_URL,
                 litellm_api_key: str = "", client_password: str = "password",
                 timeout: int = 3600, gui: bool = True, max_steps: int = 200,
                 cua_native: bool = False, task_id: str | None = None,
                 no_subagents: bool = False) -> None:
        model, litellm_base_url = normalize_model_and_base(model, litellm_base_url)
        if not litellm_api_key:
            raise ValueError("CUAWright requires the WeaveBench-supplied API key")
        if not gui:
            raise ValueError("CUAWright supports WeaveBench GUI mode only")
        self.model, self.litellm_api_key = model, litellm_api_key
        self.litellm_base_url = litellm_base_url.rstrip("/")
        self.client_password, self.task_id = client_password, task_id
        self.timeout, self.max_steps = int(timeout), int(max_steps)
        timeout_override = os.environ.get(
            "CUAWRIGHT_WEAVEBENCH_WALL_TIMEOUT_SECONDS",
            "",
        ).strip()
        if timeout_override:
            try:
                self.timeout = int(timeout_override)
            except ValueError as exc:
                raise ValueError(
                    "CUAWRIGHT_WEAVEBENCH_WALL_TIMEOUT_SECONDS must be an integer"
                ) from exc
            if self.timeout < 1:
                raise ValueError(
                    "CUAWRIGHT_WEAVEBENCH_WALL_TIMEOUT_SECONDS must be positive"
                )
        self.no_subagents = bool(no_subagents)
        if self.timeout < 1 or self.max_steps < 1:
            raise ValueError("timeout and max_steps must be positive")
        if cua_native:
            raise ValueError("CUAWright does not support native CUA mode")

    def bootstrap(self, env: Any) -> None:
        setup = getattr(env, "setup_controller", None)
        controller = CompatibleController(env.controller)
        _provision_control_clients(
            controller,
            None if setup is None else CompatibleSetupController(setup),
        )
        _write_guest_file(
            controller,
            "/home/user/.local/bin/osw",
            OSW_SCRIPT.encode("utf-8"),
            0o755,
        )
        for name, content in WEAVEBENCH_KNOWLEDGE_FILES.items():
            _write_guest_file(
                controller,
                f"/home/user/.local/share/osw/kb/{name}",
                content.encode("utf-8"),
                0o644,
            )

    def configure(self, env: Any) -> None:
        del env

    def _overlay(self, prompt: str) -> BenchmarkOverlay:
        """Compatibility adapter for integrations still inspecting overlays."""
        return BenchmarkOverlay(
            name="weavebench",
            system_prompt=translate_system_prompt(prompt, client_password=self.client_password),
            role_instructions=localized_role_instructions(),
            linear_actor_instructions=localized_linear_actor_instructions(),
            workspace=AGENT_WORKSPACE,
        )

    def _prompt_bundle(self, prompt: str) -> PromptBundle:
        translated = translate_system_prompt(
            prompt,
            client_password=self.client_password,
        )
        return select_weavebench_prompt_bundle(
            translated_system_prompt=translated,
            no_subagents=self.no_subagents,
        )

    def run(self, env: Any, instruction: str, output_dir: Path,
            system_prompt_override: str | None = None,
            already_configured: bool = False, **_ignored: Any) -> dict[str, Any]:
        output_dir = Path(output_dir)
        native_root = output_dir / "cuawright"
        native_root.mkdir(parents=True, exist_ok=True)
        started, runner, native_run_dir = time.monotonic(), None, None
        timed_out, exit_code, error = False, 1, None
        try:
            if not already_configured:
                self.bootstrap(env)
                self.configure(env)
            if system_prompt_override is None:
                raise ValueError("CUAWright requires WeaveBench's system prompt")
            with _api_key_memfd(self.litellm_api_key) as key_path:
                config = TaskConfig(
                    task_id=self.task_id or output_dir.name,
                    api_key_file=key_path,
                    result_root=str(native_root),
                    base_url=self.litellm_base_url,
                    model=MODEL,
                    reasoning_effort=os.environ.get(
                        "CUAWRIGHT_WEAVEBENCH_REASONING_EFFORT",
                        "high",
                    ),
                    total_steps=self.max_steps,
                    model_provider="openai",
                    model_infrastructure="aiservices",
                )
                runner = TaskRunner(config)
                native_run_dir, tool_step = Path(runner.run_dir), 0
                try:
                    def execute(command: str, timeout: int | None) -> CommandResult:
                        nonlocal tool_step
                        tool_step += 1
                        seconds = max(
                            1, min(int(timeout or COMMAND_TIMEOUT_MAX), COMMAND_TIMEOUT_MAX))
                        script = ('export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:'
                                  '/usr/bin:/sbin:/bin:$HOME/.local/bin"\n'
                                  "export DISPLAY=:0\n"
                                  "[ ! -f /tmp/openclaw_task_env.sh ] || "
                                  ". /tmp/openclaw_task_env.sh\n"
                                  f"cd {GUEST_WORKSPACE}\n{command}")
                        result = run_guest_command(env.controller, script, seconds) or {
                            "status": "error", "returncode": -1, "output": "",
                            "error": "VM command returned no result"}
                        code = result.get("returncode")
                        code = int(code) if isinstance(code, int) and not isinstance(code, bool) else 1
                        if result.get("status") == "error" and code == 0:
                            code = 1
                        raw = json.dumps(result, ensure_ascii=False, default=str)
                        blob, digest = runner.trace.command_blob(tool_step, raw)
                        runner.trace.event("terminal_command", {
                            "role": "terminal-agent", "stage": "terminal-orchestrator",
                            "tool_step": tool_step, "command": command, "timeout": seconds,
                            "exit_code": code, "output_chars": len(raw),
                            "output_sha256": digest, "output_blob": blob})
                        return CommandResult(
                            code, str(result.get("output") or ""), str(result.get("error") or ""),
                            inspected=code == 0 and is_inspection_command(command),
                            model_fields={**result, "_harness": {
                                "tool_step": tool_step, "configured_total_steps": self.max_steps,
                                "stage": "terminal-orchestrator"}})

                    plugin = WeaveBenchDesktopPlugin(
                        _DesktopEnvView(env), execute,
                        workspace=GuestWorkspace(AGENT_WORKSPACE, AGENT_WORKSPACE))
                    with _wall_timeout(self.timeout):
                        prompt_bundle = self._prompt_bundle(system_prompt_override)
                        runner.run_terminal_plugin(
                            plugin, instruction=instruction,
                            prompt_bundle=prompt_bundle)
                    exit_code = 0
                finally:
                    runner.close()
        except CUAWrightWallTimeout as exc:
            timed_out, exit_code, error = True, 124, str(exc)
        except Exception as exc:  # retain partial native artifacts
            error = f"{type(exc).__name__}: {exc}"
            logger.error("CUAWright run failed: %s", _redact(error, self.litellm_api_key))
        finally:
            elapsed = round(time.monotonic() - started, 2)
            try:
                export_cuawright_chat(native_run_dir, output_dir / "chat.jsonl", model=self.model)
            except Exception as exc:
                detail = _redact(f"{type(exc).__name__}: {exc}", self.litellm_api_key)
                error = f"{error}; chat export failed: {detail}" if error else detail
                exit_code = max(exit_code, 1)
                (output_dir / "chat.jsonl").write_text(
                    '{"type":"run_status","message":{"role":"system",'
                    '"content":"CUAWright trace export failed."}}\n', encoding="utf-8")
            safe_error = _redact(error or "", self.litellm_api_key) or None
            status = {
                "agent_done": not timed_out, "elapsed_seconds": elapsed,
                "timed_out": timed_out, "exit_code": exit_code, "model": self.model,
                "gui": True, "error": safe_error,
                "trajectory_image_dir": (
                    "trajectory_images"
                    if (output_dir / "trajectory_images").is_dir() else None),
                "cuawright_run_dir": (
                    str(native_run_dir.relative_to(output_dir))
                    if native_run_dir is not None else None),
            }
            (output_dir / "execution_status.json").write_text(
                json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
            log = [f"AGENT_EXIT={exit_code}", "RETRIES_USED=0"]
            if safe_error:
                log.append(f"ERROR={safe_error}")
            (output_dir / "agent.log").write_text("\n".join(log) + "\n", encoding="utf-8")
        return status

# fmt: on

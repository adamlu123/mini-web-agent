from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping

from cuawright.artifacts import utc_now
from cuawright.config.prompt_registry import select_odysseys_prompt_bundle
from cuawright.environments.base import CommandResult
from cuawright.environments.control import StatePersistenceError
from cuawright.environments.control_clients import (
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
)
from cuawright.environments.desktop.commands import is_inspection_command
from cuawright.environments.desktop.guest_commands import run_guest_command
from cuawright.environments.desktop.guest_tools import (
    provision_control_clients,
    write_guest_file,
)
from cuawright.environments.desktop.knowledge import DESKTOP_KNOWLEDGE_SCRIPT
from cuawright.environments.desktop.plugin import GuestWorkspace
from cuawright.environments.odysseys.environment import (
    ODYSSEYS_SCREENSHOT_DIR,
    ODYSSEYS_WORKSPACE,
    OdysseysBenchmarkPlugin,
    OdysseysTrajectoryError,
)
from cuawright.environments.odysseys.source import (
    DESKTOP_RUNTIME_COMMIT,
    DesktopRuntimeSource,
    OdysseysSource,
    inspect_desktop_runtime,
    inspect_source,
)
from cuawright.environments.odysseys.task import (
    ODYSSEYS_DEFAULT_DOMAIN,
    OdysseysTask,
    load_odysseys_task,
)
from cuawright.environments.terminal import TaskConfig, TaskRunner
from cuawright.health.contracts import health_contract_manifest
from cuawright.models.anthropic_messages import (
    ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL,
    ANTHROPIC_DEFAULT_MAX_REQUEST_MB,
    ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES,
)
from cuawright.models.openrouter_cost import DollarBudgetExceeded
from cuawright.models.providers import resolve_base_url, resolve_model
from cuawright.operations.scheduler import atomic_write_json


logger = logging.getLogger("cuawright.run.odysseys")
ODYSSEYS_TOTAL_STEPS = 100
_DESKTOP_RUNTIME_ACTIVATION_LOCK = threading.RLock()
_BROWSER_KNOWLEDGE = """# Odysseys Chrome procedure

Task setup opens the required visible Chrome profile and page. Connect to the
existing browser with `/home/user/.local/share/osw/venv/bin/python` and
`playwright.sync_api`, normally at `http://127.0.0.1:1337`. Reuse an existing
context and page, select the task page from its URL or title, and call
`page.bring_to_front()` before important visual actions. Do not start another
Chrome, use a fresh profile, or substitute hidden requests for visible work.

Use real navigation, clicks, typing, uploads, and accessibility locators. Wait
for meaningful content to render and leave the evidence visible before each
command returns. If port 1337 is unavailable, inspect the visible browser and
try the existing 9222 mirror; do not probe arbitrary ports.
"""


def _require_no_odysseys_proxy(example: Mapping[str, Any]) -> None:
    if bool(example.get("proxy", False)):
        raise RuntimeError(
            "Odysseys task requests proxy setup, but the pinned Odysseys "
            "runtime disables proxies rather than using an unverified pool"
        )


class InfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DesktopRuntimeBindings:
    source: DesktopRuntimeSource
    desktop_env_type: Any
    default_disk_size_gb: int


@dataclass
class OdysseysRunConfig(TaskConfig):
    reasoning_effort: str = "xhigh"
    total_steps: int = ODYSSEYS_TOTAL_STEPS
    source: str = ""
    source_commit: str = ""
    dataset_path: str = ""
    examples_dir: str = ""
    judge_script: str = ""
    dataset_sha256: str = ""
    judge_sha256: str = ""
    runtime_source: str = ""
    runtime_commit: str = DESKTOP_RUNTIME_COMMIT
    domain: str = ODYSSEYS_DEFAULT_DOMAIN
    vm_path: str = ""
    headless: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model_provider == "openai" and self.max_dollars is not None:
            raise ValueError(
                "--max-dollars is unsupported for Odysseys OpenAI runs because "
                "the current provider factory has no enforceable local cost tracker"
            )


class OdysseysRunner(TaskRunner):
    """Run Odysseys through current CUAWright provider/runtime boundaries."""

    config: OdysseysRunConfig

    def __init__(self, config: OdysseysRunConfig) -> None:
        super().__init__(config)
        self.prompt_bundle = select_odysseys_prompt_bundle()
        self.env: Any | None = None
        self.tool_step = 0
        self._tool_step_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        source = inspect_source(
            root=self.config.source,
            expected_commit=self.config.source_commit,
            dataset_path=self.config.dataset_path,
            examples_dir=self.config.examples_dir,
            judge_script=self.config.judge_script,
            expected_dataset_sha256=self.config.dataset_sha256,
            expected_judge_sha256=self.config.judge_sha256,
        )
        task = self._load_task()
        _require_no_odysseys_proxy(task.example)
        vm_path = Path(self.config.vm_path).expanduser().resolve()
        if not vm_path.is_file():
            raise FileNotFoundError(f"Odysseys VM image not found: {vm_path}")
        with _activate_desktop_runtime(
            self.config.runtime_source,
            self.config.runtime_commit,
        ) as runtime:
            return self._run_with_desktop_runtime(source, task, vm_path, runtime)

    def _run_with_desktop_runtime(
        self,
        source: OdysseysSource,
        task: OdysseysTask,
        vm_path: Path,
        runtime: DesktopRuntimeBindings,
    ) -> dict[str, Any]:
        terminal_started = False
        judge_dir = self.run_dir / "odysseys-judge" / task.task_id
        manifest_extra = {
            "release": "configured-external-identity",
            "environment": "osworld-compatible-docker",
            "odysseys_source": source.provenance(),
            "desktop_runtime": runtime.source.provenance(),
            "source_example": str(task.example_path),
            "source_example_sha256": task.example_sha256,
            "task_level": task.level,
            "reference_length": task.reference_length,
            "judge_run_dir": str(judge_dir),
            "terminal_control": {
                "protocol": "python-clients-v1",
                "install_root": CLIENT_INSTALL_ROOT,
                "clients": dict(CLIENT_SHA256),
            },
            "trajectory_capture": {
                "active_roles": ["actor", "binder", "verifier"],
                "captured_role_domain": [
                    "actor",
                    "binder",
                    "critic",
                    "verifier",
                ],
                "includes_actor_specialist_invocations": True,
                "guest_screenshot_dir": ODYSSEYS_SCREENSHOT_DIR,
                "host_judge_dir": str(judge_dir),
                "format": "osworld-traj-jsonl-v2-role-stage",
            },
        }
        self._write_manifest(
            {
                **asdict(self.config),
                **manifest_extra,
                "benchmark": "odysseys",
                "task_id": self.config.task_id,
                "started_at": utc_now(),
                "model_interface": "terminal-only",
                "agent_topology": self.prompt_bundle.agent_topology,
                "health_contract": health_contract_manifest(),
                "prompt_hashes": self._prompt_hashes(
                    prompt_bundle=self.prompt_bundle,
                    overlay=None,
                ),
            }
        )
        try:
            try:
                volume_size = task.example.get("volume_size")
                self.env = runtime.desktop_env_type(
                    provider_name="docker",
                    path_to_vm=str(vm_path),
                    os_type="Ubuntu",
                    action_space="pyautogui",
                    cache_dir=str(self.run_dir / "evaluator-cache"),
                    headless=self.config.headless,
                    require_a11y_tree=True,
                    enable_proxy=False,
                    volume_size=(
                        volume_size
                        if volume_size is not None
                        else runtime.default_disk_size_gb
                    ),
                    force_disable_recording=True,
                )
                self.env.reset(task_config=dict(task.example))
            except Exception as exc:
                if _is_infrastructure_error(exc):
                    raise InfrastructureError(
                        f"Odysseys guest setup infrastructure failed: {exc}"
                    ) from exc
                raise

            self.ask_user = (
                self.env.user_simulator.respond
                if self.env.user_simulator is not None
                else None
            )
            _provision_guest_tools(self.env.controller, self.env.setup_controller)
            _ensure_guest_disk_headroom(self.env.controller)
            plugin = self._create_plugin(
                task=task,
                judge_dir=judge_dir,
                source_identity=source.provenance(),
            )
            atomic_write_json(
                self.run_dir / "args.json",
                {"mind2web_tasks_path": str(judge_dir / "inputs" / "odysseys.json")},
            )
            terminal_started = True
            try:
                return self.run_terminal_benchmark(
                    benchmark="odysseys",
                    plugin=plugin,
                    instruction=task.instruction,
                    prompt_bundle=self.prompt_bundle,
                    manifest_extra=manifest_extra,
                    command_observer=plugin.observe_command,
                )
            except OdysseysTrajectoryError as exc:
                raise InfrastructureError(
                    f"Odysseys screenshot trajectory capture failed: {exc}"
                ) from exc
            except Exception as exc:
                if _is_infrastructure_error(exc):
                    raise InfrastructureError(
                        f"Odysseys guest runtime infrastructure failed: {exc}"
                    ) from exc
                raise
        finally:
            if not terminal_started:
                try:
                    if self.env is not None:
                        self.env.close()
                finally:
                    self.close()

    def _load_task(self) -> OdysseysTask:
        return load_odysseys_task(
            task_id=self.config.task_id,
            dataset_path=self.config.dataset_path,
            examples_dir=self.config.examples_dir,
            domain=self.config.domain,
        )

    def _create_plugin(
        self,
        *,
        task: OdysseysTask,
        judge_dir: Path,
        source_identity: Mapping[str, Any],
    ) -> OdysseysBenchmarkPlugin:
        if self.env is None:
            raise RuntimeError("Environment is not initialized")
        workspace = self.prompt_bundle.workspace
        initialization = run_guest_command(
            self.env.controller,
            (
                f"install -d -m 0700 {workspace} {workspace}/cache "
                f"{ODYSSEYS_WORKSPACE} {ODYSSEYS_SCREENSHOT_DIR}"
            ),
            30,
        )
        if (
            not initialization
            or initialization.get("status") == "error"
            or initialization.get("returncode") not in (None, 0)
        ):
            raise InfrastructureError(
                "Could not initialize the Odysseys agent workspace: "
                f"{initialization}"
            )
        return OdysseysBenchmarkPlugin(
            self.env,
            self._execute_command,
            workspace=GuestWorkspace(workspace, workspace),
            task=task,
            judge_dir=judge_dir,
            judge_script=self.config.judge_script,
            source_identity=source_identity,
            run_store=self.trace,
        )

    def _execute_command(
        self,
        command: str,
        timeout: int | None,
    ) -> CommandResult:
        if self.env is None:
            raise RuntimeError("Environment is not initialized")
        with self._tool_step_lock:
            self.tool_step += 1
            tool_step = self.tool_step
        effective_timeout = max(1, min(int(timeout or 300), 300))
        wrapped = (
            'export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:'
            '/sbin:/bin:$HOME/.local/bin"\n'
            "export DISPLAY=:0\n"
            f"{command}"
        )
        result = run_guest_command(self.env.controller, wrapped, effective_timeout)
        self.env.is_environment_used = True
        if not result:
            result = {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": "VM command returned no result",
                "infrastructure_error": True,
            }
        if result.get("infrastructure_error") is True:
            raise InfrastructureError(
                "Odysseys guest command service failed: "
                + str(result.get("error") or result)
            )
        result["_harness"] = {
            "tool_step": tool_step,
            "configured_total_steps": self.config.total_steps,
            "stage": "terminal-orchestrator",
            "benchmark": "odysseys",
        }
        full_text = json.dumps(result, ensure_ascii=False, default=str)
        blob, digest = self.trace.command_blob(tool_step, full_text)
        stdout = str(result.get("output") or "")
        stderr = str(result.get("error") or "")
        returncode = result.get("returncode")
        exit_code = (
            int(returncode)
            if isinstance(returncode, int) and not isinstance(returncode, bool)
            else 0
        )
        if result.get("status") == "error" and exit_code == 0:
            exit_code = 1
        if exit_code != 0 and "no space left on device" in (
            f"{stderr}\n{stdout}".lower()
        ):
            raise InfrastructureError("Odysseys guest disk exhausted")
        self.trace.event(
            "terminal_command",
            {
                "role": "terminal-agent",
                "stage": "terminal-orchestrator",
                "benchmark": "odysseys",
                "tool_step": tool_step,
                "command": command,
                "timeout": effective_timeout,
                "exit_code": exit_code,
                "output_chars": len(full_text),
                "output_sha256": digest,
                "output_blob": blob,
            },
        )
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            inspected=(exit_code == 0 and is_inspection_command(command)),
            model_fields=result,
        )


def remaining_retry_budget(
    max_dollars: float | None,
    spent_dollars: float,
) -> float | None:
    if max_dollars is None:
        return None
    remaining = max_dollars - spent_dollars
    if remaining <= 0:
        raise DollarBudgetExceeded(
            "Dollar budget was exhausted before the infrastructure retry: "
            f"${spent_dollars:.6f} spent of ${max_dollars:.2f}"
        )
    return remaining


def _run_with_infrastructure_classification(
    run: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return run()
    except StatePersistenceError as exc:
        raise InfrastructureError("Secure control-file persistence failed") from exc


@contextmanager
def _activate_desktop_runtime(
    root: str | Path,
    expected_commit: str,
) -> Iterator[DesktopRuntimeBindings]:
    with _DESKTOP_RUNTIME_ACTIVATION_LOCK:
        with _activate_desktop_runtime_locked(root, expected_commit) as runtime:
            yield runtime


@contextmanager
def _activate_desktop_runtime_locked(
    root: str | Path,
    expected_commit: str,
) -> Iterator[DesktopRuntimeBindings]:
    source = inspect_desktop_runtime(root, expected_commit)
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "desktop_env" or name.startswith("desktop_env.")
    }
    for name in previous_modules:
        sys.modules.pop(name, None)
    selected = str(source.root)
    sys.path[:] = [selected, *(item for item in previous_path if item != selected)]
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    try:
        package_module = importlib.import_module("desktop_env")
        desktop_module = importlib.import_module("desktop_env.desktop_env")
        provider_module = importlib.import_module(
            "desktop_env.providers.docker.provider"
        )
        _require_module_origin(package_module, source.root)
        _require_module_origin(desktop_module, source.root)
        _require_module_origin(provider_module, source.root)
        for name, module in tuple(sys.modules.items()):
            if (
                (name == "desktop_env" or name.startswith("desktop_env."))
                and isinstance(module, ModuleType)
                and getattr(module, "__file__", None) is not None
            ):
                _require_module_origin(module, source.root)
        desktop_env_type = getattr(desktop_module, "DesktopEnv", None)
        default_disk_size = getattr(provider_module, "DEFAULT_DISK_SIZE_GB", None)
        if not callable(desktop_env_type):
            raise RuntimeError("Configured desktop runtime has no DesktopEnv class")
        if (
            isinstance(default_disk_size, bool)
            or not isinstance(default_disk_size, int)
            or default_disk_size < 1
        ):
            raise RuntimeError(
                "Configured desktop runtime has invalid DEFAULT_DISK_SIZE_GB"
            )
        yield DesktopRuntimeBindings(
            source=source,
            desktop_env_type=desktop_env_type,
            default_disk_size_gb=default_disk_size,
        )
    finally:
        for name in tuple(sys.modules):
            if name == "desktop_env" or name.startswith("desktop_env."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode
        importlib.invalidate_caches()


def _require_module_origin(module: ModuleType, root: Path) -> None:
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str):
        raise RuntimeError(f"{module.__name__} has no concrete module origin")
    origin = Path(raw_origin).resolve()
    if origin != root and root not in origin.parents:
        raise RuntimeError(
            f"{module.__name__} loaded from {origin}, "
            f"outside configured runtime {root}"
        )


def _provision_guest_tools(controller: Any, setup_controller: Any) -> None:
    provision_control_clients(controller, setup_controller)
    write_guest_file(
        controller,
        "/home/user/.local/bin/osw",
        DESKTOP_KNOWLEDGE_SCRIPT.encode("utf-8"),
        0o755,
    )
    write_guest_file(
        controller,
        "/home/user/.local/share/osw/kb/odysseys-browser.md",
        _BROWSER_KNOWLEDGE.encode("utf-8"),
        0o644,
    )


def _ensure_guest_disk_headroom(controller: Any) -> None:
    result = run_guest_command(controller, "df -Pk / | tail -n 1", 10)
    fields = str(result.get("output") or "").split() if result else []
    if result.get("status") != "success" or len(fields) < 6:
        raise InfrastructureError("Unable to verify Odysseys guest disk headroom")
    try:
        free_kb = int(fields[3])
    except ValueError as exc:
        raise InfrastructureError("Invalid guest disk headroom response") from exc
    if free_kb < 5 * 1024 * 1024:
        raise InfrastructureError(
            "Odysseys guest disk headroom is below 5 GiB after setup"
        )


def _is_infrastructure_error(value: object) -> bool:
    text = str(value).lower()
    return any(
        marker in text
        for marker in (
            "connection refused",
            "connection reset",
            "connection aborted",
            "remote end closed",
            "server not ready",
            "timed out",
            "timeout",
            "500 internal server error",
            "502 bad gateway",
            "503 service unavailable",
            "504 gateway timeout",
            "guest command service failed",
            "guest disk",
        )
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUAWright on Odysseys")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--task-source-json", required=True)
    parser.add_argument("--examples-dir", required=True)
    parser.add_argument("--judge-script", required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--judge-sha256", required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--runtime-commit", default=DESKTOP_RUNTIME_COMMIT)
    parser.add_argument("--domain", default=ODYSSEYS_DEFAULT_DOMAIN)
    parser.add_argument("--path-to-vm", required=True)
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument(
        "--model-provider",
        choices=["openai", "openrouter-anthropic", "anthropic"],
        default="openai",
    )
    parser.add_argument("--max-dollars", type=float)
    parser.add_argument(
        "--anthropic-min-recent-images",
        type=int,
        default=ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES,
    )
    parser.add_argument(
        "--anthropic-image-prune-interval",
        type=int,
        default=ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL,
    )
    parser.add_argument(
        "--anthropic-max-request-mb",
        type=float,
        default=ANTHROPIC_DEFAULT_MAX_REQUEST_MB,
    )
    parser.add_argument("--model-infrastructure", default="aiservices")
    parser.add_argument("--total-steps", type=int, default=ODYSSEYS_TOTAL_STEPS)
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.model_provider == "openai" and args.max_dollars is not None:
        raise ValueError(
            "--max-dollars is unsupported for Odysseys OpenAI runs because "
            "OpenAI cost enforcement is unavailable"
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = OdysseysRunConfig(
        task_id=args.task_id,
        source=args.source,
        source_commit=args.source_commit,
        dataset_path=args.task_source_json,
        examples_dir=args.examples_dir,
        judge_script=args.judge_script,
        dataset_sha256=args.dataset_sha256,
        judge_sha256=args.judge_sha256,
        runtime_source=args.runtime_source,
        runtime_commit=args.runtime_commit,
        domain=args.domain,
        vm_path=args.path_to_vm,
        api_key_file=args.api_key_file,
        result_root=args.result_root,
        base_url=resolve_base_url(args.model_provider, args.base_url),
        model=resolve_model(args.model_provider, args.model),
        reasoning_effort=args.reasoning_effort,
        total_steps=args.total_steps,
        headless=not args.no_headless,
        model_provider=args.model_provider,
        max_dollars=args.max_dollars,
        model_infrastructure=args.model_infrastructure,
        anthropic_min_recent_images=args.anthropic_min_recent_images,
        anthropic_image_prune_interval=args.anthropic_image_prune_interval,
        anthropic_max_request_mb=args.anthropic_max_request_mb,
    )
    retry_budget = config.max_dollars
    for attempt in range(2):
        runner = OdysseysRunner(replace(config, max_dollars=retry_budget))
        try:
            result = _run_with_infrastructure_classification(runner.run)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
            return
        except InfrastructureError as exc:
            if attempt == 0:
                spent = (
                    runner.cost_tracker.total_cost_usd
                    if runner.cost_tracker is not None
                    else 0.0
                )
                retry_budget = remaining_retry_budget(retry_budget, spent)
                logger.warning(
                    "Odysseys infrastructure failed; retrying once in a fresh VM: %s",
                    exc,
                )
                continue
            raise


if __name__ == "__main__":
    main()

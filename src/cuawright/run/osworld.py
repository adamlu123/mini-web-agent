from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cuawright.artifacts import utc_now
from cuawright.budget import DEFAULT_TOTAL_STEPS
from cuawright.config.prompt_registry import select_osworld_prompt_bundle
from cuawright.config.prompts import phase_user_prompt
from cuawright.config.store import configured_value
from cuawright.environments.base import CommandResult
from cuawright.environments.control import StatePersistenceError
from cuawright.environments.control_clients import (
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
)
from cuawright.environments.desktop.guest_tools import provision_control_clients
from cuawright.environments.osworld.compatibility import (
    install_release_compatibility,
)
from cuawright.environments.osworld.environment import (
    GuestWorkspace,
    OSWorldBenchmarkPlugin,
)
from cuawright.environments.osworld.guest_tools import provision_guest_tools
from cuawright.environments.osworld.lifecycle import (
    InfrastructureError,
    ensure_guest_disk_headroom,
    exception_diagnostics,
    is_guest_infrastructure_error,
    is_inspection_command,
    is_retryable_setup_error,
    preflight_proxy_configuration,
    raise_for_confirmed_guest_home_read_only,
    raise_for_guest_command_infrastructure,
    run_guest_command,
)
from cuawright.environments.osworld.source import (
    DEFAULT_RELEASE_TAG,
    activate_source,
    resolve_release,
)
from cuawright.environments.terminal import TaskConfig, TaskRunner
from cuawright.health.contracts import health_contract_manifest
from cuawright.models.anthropic_messages import (
    ANTHROPIC_DEFAULT_IMAGE_PRUNE_INTERVAL,
    ANTHROPIC_DEFAULT_MAX_REQUEST_MB,
    ANTHROPIC_DEFAULT_MIN_RECENT_IMAGES,
)
from cuawright.models.openrouter_cost import DollarBudgetExceeded
from cuawright.models.providers import (
    PROVIDER_DEFAULT_BASE_URLS,
    resolve_base_url,
    resolve_model,
)

logger = logging.getLogger("cuawright.run.osworld")
DEFAULT_OSWORLD_DISK_SIZE_GB = 80


def _configure_docker_resources(
    desktop_env: Any,
    *,
    ram_size_gb: int | None,
    cpu_cores: int | None,
) -> None:
    if ram_size_gb is None and cpu_cores is None:
        return
    environment = getattr(getattr(desktop_env, "provider", None), "environment", None)
    if not isinstance(environment, dict):
        raise TypeError("Docker provider does not expose a resource environment")
    if ram_size_gb is not None:
        environment["RAM_SIZE"] = f"{ram_size_gb}G"
    if cpu_cores is not None:
        environment["CPU_CORES"] = str(cpu_cores)
    logger.info(
        "Configured Docker guest resources: RAM_SIZE=%s CPU_CORES=%s",
        environment.get("RAM_SIZE"),
        environment.get("CPU_CORES"),
    )


def _install_task_setup_compatibility(
    task: Any,
    *,
    assets_dir: str,
    task_id: str,
    benchmark_release: str = DEFAULT_RELEASE_TAG,
) -> Callable[[], None]:
    if task_id != "089" or resolve_release(benchmark_release).tag != "v2026.06.24":
        return lambda: None

    local_asset = Path(assets_dir).resolve() / "task_089" / "broken_presentation.zip"
    if not local_asset.is_file():
        raise FileNotFoundError(f"OSWorld task 089 asset not found: {local_asset}")

    task_type = task.__class__
    original_setup = task_type.setup
    guest_asset = "/tmp/cuawright-task-089/broken_presentation.zip"
    task_module = sys.modules[task_type.__module__]

    def setup_with_staged_asset(
        self: Any,
        setup_controller: Any,
        use_proxy: bool = False,
    ) -> Any:
        setup_controller.download([{"url": str(local_asset), "path": guest_asset}])
        original_url = task_module.PRESENTATION_ZIP_URL
        task_module.PRESENTATION_ZIP_URL = Path(guest_asset).as_uri()
        try:
            return original_setup(self, setup_controller, use_proxy)
        finally:
            task_module.PRESENTATION_ZIP_URL = original_url

    task_type.setup = setup_with_staged_asset

    def restore() -> None:
        task_type.setup = original_setup

    return restore


@dataclass
class OSWorldRunConfig(TaskConfig):
    benchmark_release: str = DEFAULT_RELEASE_TAG
    osworld_root: str = ""
    vm_path: str = ""
    docker_ram_size_gb: int | None = None
    docker_cpu_cores: int | None = None
    assets_dir: str = ""
    evaluation_examples_dir: str = "evaluation_examples"
    website_host_suffix: str = "10.217.29.172.nip.io"
    headless: bool = True
    user_simulator_api_key_file: str | None = None
    user_simulator_base_url: str = PROVIDER_DEFAULT_BASE_URLS["openai"]
    no_subagents: bool = False
    prompt_profile: str = "canonical"
    disable_task_proxy: bool = False


class OSWorldRunner(TaskRunner):
    """Compose CUAWright with the explicitly vendored upstream OSWorld engine."""

    config: OSWorldRunConfig

    def __init__(self, config: OSWorldRunConfig) -> None:
        super().__init__(config)
        self.prompt_bundle = select_osworld_prompt_bundle(
            no_subagents=config.no_subagents,
            prompt_profile=config.prompt_profile,
        )
        self.env: Any | None = None
        self.tool_step = 0
        self._tool_step_lock = threading.Lock()

    def run(self) -> dict[str, Any]:
        source = activate_source(
            self.config.osworld_root or None,
            release=self.config.benchmark_release,
        )
        from desktop_env.desktop_env import DesktopEnv
        from task_loader import load_task_config

        install_release_compatibility(source.release.tag)
        os.environ["OSWORLD_FILE_BASE_URL"] = str(
            Path(self.config.assets_dir).resolve()
        )
        os.environ["WEBSITE_HOST_SUFFIX"] = self.config.website_host_suffix
        if self.config.model_provider == "openai":
            self._configure_openai_evaluation(self.api_key, self.config.base_url)
        elif self.config.user_simulator_api_key_file:
            simulator_key = (
                Path(self.config.user_simulator_api_key_file)
                .read_text(encoding="utf-8")
                .strip()
            )
            if not simulator_key:
                raise ValueError("User simulator API key file is empty")
            self._configure_openai_evaluation(
                simulator_key,
                self.config.user_simulator_base_url,
            )
        os.environ["OSWORLD_EVAL_SAVE_RAW_DIR"] = str(
            self.run_dir / "evaluation" / "model-calls"
        )

        prompt_hashes = self._prompt_hashes(
            prompt_bundle=self.prompt_bundle,
            overlay=None,
        )
        config_manifest = asdict(self.config)
        config_manifest.pop("no_subagents")
        manifest = {
            **config_manifest,
            "osworld_source": source.provenance(),
            "started_at": utc_now(),
            "release": source.release.tag,
            "benchmark": "osworld-v2",
            "model_interface": "terminal-only",
            "agent_topology": self.prompt_bundle.agent_topology,
            "terminal_control": {
                "protocol": "python-clients-v1",
                "install_root": CLIENT_INSTALL_ROOT,
                "clients": dict(CLIENT_SHA256),
            },
            "health_contract": health_contract_manifest(),
            "prompt_hashes": prompt_hashes,
        }
        self._write_manifest(manifest)
        task = load_task_config(
            None,
            task_id=self.config.task_id,
            base_dir=self.config.evaluation_examples_dir,
            domain="tasks",
            eval_version="v2",
        )
        restore_task_setup = _install_task_setup_compatibility(
            task,
            assets_dir=self.config.assets_dir,
            task_id=self.config.task_id,
            benchmark_release=source.release.tag,
        )
        instruction = task["instruction"]
        task_proxy_requested = bool(task.get("proxy", False))
        task_volume_size = task.get("volume_size")
        phase_getter = getattr(task, "get_phases", None)
        phases = phase_getter() if callable(phase_getter) else []

        try:
            if task_proxy_requested and not self.config.disable_task_proxy:
                proxy_path = os.environ.get("PROXY_CONFIG_FILE", "")
                manifest["task_proxy"] = {
                    "requested": True,
                    "enabled": True,
                    "preflight": preflight_proxy_configuration(proxy_path),
                }
            else:
                manifest["task_proxy"] = {
                    "requested": task_proxy_requested,
                    "enabled": False,
                    "preflight": {
                        "status": (
                            "operator-disabled"
                            if task_proxy_requested
                            else "not-requested"
                        )
                    },
                }
            self._write_manifest(manifest)
            try:
                self.env = DesktopEnv(
                    provider_name="docker",
                    path_to_vm=self.config.vm_path,
                    os_type="Ubuntu",
                    action_space="pyautogui",
                    cache_dir=str(self.run_dir / "evaluator-cache"),
                    headless=self.config.headless,
                    require_a11y_tree=True,
                    enable_proxy=not self.config.disable_task_proxy,
                    volume_size=(
                        task_volume_size
                        if task_volume_size is not None
                        else DEFAULT_OSWORLD_DISK_SIZE_GB
                    ),
                    force_disable_recording=True,
                )
                _configure_docker_resources(
                    self.env,
                    ram_size_gb=self.config.docker_ram_size_gb,
                    cpu_cores=self.config.docker_cpu_cores,
                )
                provision_control_clients(
                    self.env.controller,
                    self.env.setup_controller,
                )
                self.env.reset(task_config=task)
            except Exception as exc:
                if self.env is not None:
                    raise_for_confirmed_guest_home_read_only(
                        self.env.controller,
                        exception_diagnostics(exc),
                        context="Guest setup infrastructure failed",
                    )
                if is_retryable_setup_error(exc):
                    raise InfrastructureError(
                        "Guest setup infrastructure failed: "
                        f"{exception_diagnostics(exc)}"
                    ) from exc
                raise

            self.ask_user = (
                self.env.user_simulator.respond
                if self.env.user_simulator is not None
                else None
            )
            try:
                provision_guest_tools(
                    self.env.controller,
                    self.env.setup_controller,
                )
            except Exception as exc:
                raise_for_confirmed_guest_home_read_only(
                    self.env.controller,
                    exception_diagnostics(exc),
                    context="Guest provisioning infrastructure failed",
                )
                if (
                    is_retryable_setup_error(exc)
                    or "no space left on device" in str(exc).lower()
                ):
                    raise InfrastructureError(
                        f"Guest provisioning infrastructure failed: {exc}"
                    ) from exc
                raise
            ensure_guest_disk_headroom(self.env.controller)

            try:
                if phases:
                    submission, verification, evaluation, prompt_hashes = (
                        self._run_multi_phase(task, phases)
                    )
                else:
                    submission, verification, prompt_hashes = self.run_terminal_plugin(
                        self._create_plugin(),
                        instruction=instruction,
                        prompt_bundle=self.prompt_bundle,
                    )
                    self.env.action_history.append("DONE")
                    evaluation = self.env.evaluate()
            except Exception as exc:
                diagnostics = exception_diagnostics(exc)
                raise_for_confirmed_guest_home_read_only(
                    self.env.controller,
                    diagnostics,
                    context="Guest runtime infrastructure failed",
                )
                if is_guest_infrastructure_error(diagnostics):
                    raise InfrastructureError(
                        f"Guest runtime infrastructure failed: {diagnostics}"
                    ) from exc
                raise

            score = (
                float(evaluation.get("score", 0.0))
                if isinstance(evaluation, dict)
                else float(evaluation)
            )
            manifest["prompt_hashes"] = prompt_hashes
            self._write_manifest(manifest)
            result = {
                "benchmark": "osworld-v2",
                "task_id": self.config.task_id,
                "submission": submission,
                "verification": verification,
                "score": score,
                "evaluation": evaluation,
                "run_dir": str(self.run_dir),
                "model_cost": self.model_cost_snapshot(),
                "prompt_hashes": prompt_hashes,
            }
            self._write_evaluation(result, score)
            return result
        finally:
            restore_task_setup()
            try:
                if self.env is not None:
                    self.env.close()
            finally:
                self.close()

    @staticmethod
    def _configure_openai_evaluation(api_key: str, base_url: str) -> None:
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OSWORLD_EVAL_MODEL_PROVIDER"] = "openai"
        os.environ["OSWORLD_EVAL_MODEL_API_KEY"] = api_key
        os.environ["OSWORLD_EVAL_MODEL_BASE_URL"] = base_url

    def _create_plugin(self) -> OSWorldBenchmarkPlugin:
        if self.env is None:
            raise RuntimeError("Environment is not initialized")
        workspace = "/home/user/.agent"
        initialization = run_guest_command(
            self.env.controller,
            (
                f"install -d -m 0700 {workspace} "
                f"{workspace}/requests {workspace}/reports {workspace}/cache"
            ),
            30,
        )
        if (
            not initialization
            or initialization.get("status") == "error"
            or initialization.get("returncode") not in (None, 0)
        ):
            raise InfrastructureError(
                f"Could not initialize the terminal agent workspace: {initialization}"
            )
        return OSWorldBenchmarkPlugin(
            self.env,
            self._execute_command,
            workspace=GuestWorkspace(workspace, workspace),
        )

    def _run_multi_phase(
        self,
        task: Any,
        phases: list[dict[str, Any]],
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        Mapping[str, Mapping[str, Any]],
    ]:
        if self.env is None:
            raise RuntimeError("Environment is not initialized")
        use_proxy = bool(task.get("proxy", False) and self.env.enable_proxy)
        phase_schedule = "\n".join(
            (
                f"{index}. {phase.get('name', f'Phase {index}')} "
                f"(weight {float(phase.get('weight', 1.0)):.2f})"
            )
            for index, phase in enumerate(phases, start=1)
        )
        first_instruction = phases[0]["instruction"]
        opening_instruction = (
            "This is a sequential multi-phase task. One shared model-call budget "
            "covers every phase and is never reset. Reserve enough calls for later "
            "phases. Submitting the current phase triggers its evaluation and the "
            "next phase setup.\n\n"
            f"Phase schedule:\n{phase_schedule}"
        )
        runtime = self.create_runtime(
            self._create_plugin(),
            instruction=opening_instruction,
            prompt_bundle=self.prompt_bundle,
            specialist_instruction=first_instruction,
            initial_actor_message=phase_user_prompt(
                first_instruction,
                phase_id="phase-1",
                phase_index=1,
                phase_count=len(phases),
                previous_phase_evaluated=False,
                agent_topology=self.prompt_bundle.agent_topology,
            ),
        )
        phase_results: list[dict[str, Any]] = []
        total_score = 0.0
        last_submission: dict[str, Any] = {"status": "forced"}
        last_verification: dict[str, Any] = {}
        prompt_hashes: Mapping[str, Mapping[str, Any]] = {}

        for phase_index, phase in enumerate(phases, start=1):
            if phase_index > 1:
                self.env._step_no = 0
                self.env.action_history.clear()
                self.env._traj_no += 1
                try:
                    phase["setup"](self.env.setup_controller, use_proxy=use_proxy)
                except Exception as exc:
                    raise_for_confirmed_guest_home_read_only(
                        self.env.controller,
                        exception_diagnostics(exc),
                        context="Guest phase setup infrastructure failed",
                    )
                    if is_retryable_setup_error(exc):
                        raise InfrastructureError(
                            "Guest phase setup infrastructure failed: "
                            f"{exception_diagnostics(exc)}"
                        ) from exc
                    raise
                self.env.is_environment_used = True
                pause = float(phase.get("pause_after_setup_seconds", 5) or 0)
                if pause:
                    time.sleep(pause)

            phase_name = phase.get("name", f"Phase {phase_index}")
            phase_instruction = phase["instruction"]
            self.env.instruction = phase_instruction
            self.trace.event(
                "phase_started",
                {
                    "phase_index": phase_index,
                    "phase_name": phase_name,
                    "instruction": phase_instruction,
                    "budget": runtime.budget.snapshot(),
                },
            )
            outcome = (
                runtime.run()
                if phase_index == 1
                else runtime.continue_phase(
                    instruction=phase_instruction,
                    phase_id=f"phase-{phase_index}",
                    phase_index=phase_index,
                    phase_count=len(phases),
                )
            )
            last_submission, last_verification = self.outcome_payload(
                outcome,
                runtime,
            )
            prompt_hashes = outcome.prompt_hashes
            self.env.action_history.append("DONE")
            phase_score = float(phase["evaluate"](self.env))
            total_score += phase_score
            phase_result = {
                "phase_index": phase_index,
                "phase_name": phase_name,
                "instruction": phase_instruction,
                "score": phase_score,
                "submission": last_submission,
                "verification": last_verification,
                "budget_after": runtime.budget.snapshot(),
            }
            phase_results.append(phase_result)
            self.trace.event("phase_evaluated", phase_result)
            gate_min = phase.get("gate_min_score")
            if gate_min is not None and phase_score < float(gate_min):
                break
            if phase.get("gate") and phase_score <= 0.0:
                break
            if runtime.budget.remaining_steps == 0:
                break

        score = round(max(0.0, min(1.0, total_score)), 4)
        verification = {
            **last_verification,
            "multi_phase": {
                "phase_count": len(phases),
                "phase_results": phase_results,
            },
            "budget": runtime.budget.snapshot(),
        }
        for name in ("orchestration.json", "verification.json"):
            (self.run_dir / name).write_text(
                json.dumps(
                    verification,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
        return (
            last_submission,
            verification,
            {"score": score, "phase_results": phase_results},
            prompt_hashes,
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
            f"export WEBSITE_HOST_SUFFIX="
            f"{json.dumps(self.config.website_host_suffix)}\n{command}"
        )
        result = run_guest_command(self.env.controller, wrapped, effective_timeout)
        self.env.is_environment_used = True
        if not result:
            result = {
                "status": "error",
                "returncode": -1,
                "output": "",
                "error": "VM command returned no result",
            }
        raise_for_guest_command_infrastructure(
            result,
            context="Guest command service failed",
        )
        result["_harness"] = {
            "tool_step": tool_step,
            "configured_total_steps": self.config.total_steps,
            "stage": "terminal-orchestrator",
        }
        full_text = json.dumps(result, ensure_ascii=False, default=str)
        blob, digest = self.trace.command_blob(tool_step, full_text)
        stdout = str(result.get("output") or "")
        stderr = str(result.get("error") or "")
        returncode = result.get("returncode")
        exit_code = int(returncode) if isinstance(returncode, int) else 0
        if result.get("status") == "error" and exit_code == 0:
            exit_code = 1
        command_failed = exit_code != 0
        if command_failed:
            raise_for_confirmed_guest_home_read_only(
                self.env.controller,
                f"{stderr}\n{stdout}",
                context="Guest command failed",
            )
        disk_exhausted = command_failed and (
            "no space left on device" in f"{stderr}\n{stdout}".lower()
        )
        if disk_exhausted:
            raise InfrastructureError("Guest disk exhausted")
        self.trace.event(
            "terminal_command",
            {
                "role": "terminal-agent",
                "stage": "terminal-orchestrator",
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


def _default_vm_path() -> str | None:
    return configured_value("osworld", "vm")


def _default_assets_dir() -> str:
    candidates = [
        REPO_ROOT / "cache" / "osworld_v2_assets",
        REPO_ROOT.parents[1] / "OSWorld-V2" / "cache" / "osworld_v2_assets",
    ]
    return str(next((path for path in candidates if path.is_dir()), candidates[0]))


def _positive_decimal_string(value: str) -> str:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("must be a decimal number") from exc
    if not amount.is_finite() or amount <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return format(amount, "f")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUAWright on OSWorld V2")
    parser.add_argument("--task-id", default="099")
    parser.add_argument(
        "--benchmark-release",
        type=lambda value: resolve_release(value).tag,
        default=resolve_release().tag,
    )
    parser.add_argument("--osworld-root")
    vm_path = _default_vm_path()
    parser.add_argument("--path-to-vm", default=vm_path, required=not vm_path)
    parser.add_argument("--docker-ram-size-gb", type=_positive_int)
    parser.add_argument("--docker-cpu-cores", type=_positive_int)
    parser.add_argument("--assets-dir", default=_default_assets_dir())
    parser.add_argument(
        "--api-key-file",
        default=str(REPO_ROOT.parents[1] / ".a-creds"),
    )
    parser.add_argument(
        "--result-root",
        default=str(Path.cwd() / "results-cuawright-osworld"),
    )
    parser.add_argument(
        "--website-host-suffix",
        default="10.217.29.172.nip.io",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", default="medium")
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
    parser.add_argument("--anthropic-max-output-tokens", type=int)
    parser.add_argument("--user-simulator-api-key-file")
    parser.add_argument(
        "--user-simulator-base-url",
        default=PROVIDER_DEFAULT_BASE_URLS["openai"],
    )
    parser.add_argument("--model-infrastructure", default="aiservices")
    parser.add_argument(
        "--disable-task-proxy",
        action="store_true",
        help="Explicitly bypass a task-requested proxy and use direct guest egress",
    )
    parser.add_argument("--openrouter-zdr", action="store_true")
    parser.add_argument(
        "--openrouter-data-collection",
        choices=["allow", "deny"],
    )
    parser.add_argument("--openrouter-require-parameters", action="store_true")
    parser.add_argument(
        "--openrouter-provider-order",
        action="append",
        default=[],
        metavar="PROVIDER",
    )
    parser.add_argument(
        "--openrouter-max-prompt-price",
        type=_positive_decimal_string,
        metavar="USD_PER_MILLION",
    )
    parser.add_argument(
        "--openrouter-max-completion-price",
        type=_positive_decimal_string,
        metavar="USD_PER_MILLION",
    )
    parser.add_argument("--no-openrouter-fallbacks", action="store_true")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument(
        "--compact-every",
        type=int,
        default=TaskConfig.__dataclass_fields__["compact_every"].default,
        help="Compact the Actor history after this many model calls; 0 disables it",
    )
    parser.add_argument(
        "--no-subagents",
        action="store_true",
        help="Run one linear Actor with no Binder, Critic, or Verifier",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["canonical", "e-l1-direct", "e-l1-disciplined"],
        default="canonical",
    )
    parser.add_argument(
        "--openai-policy-feedback-retries",
        type=int,
        default=0,
        help="Append bounded benign compliance feedback after OpenAI policy blocks",
    )
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    no_subagents = bool(getattr(args, "no_subagents", False))
    source = activate_source(
        args.osworld_root,
        release=args.benchmark_release,
    )
    if not args.model_infrastructure.strip():
        raise ValueError("model-infrastructure cannot be empty")
    if args.openai_policy_feedback_retries < 0:
        raise ValueError("openai-policy-feedback-retries cannot be negative")
    if args.openai_policy_feedback_retries and args.model_provider != "openai":
        raise ValueError(
            "openai-policy-feedback-retries requires --model-provider openai"
        )
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    preferences: dict[str, object] = {}
    if args.openrouter_zdr:
        preferences["zdr"] = True
    if args.openrouter_data_collection is not None:
        preferences["data_collection"] = args.openrouter_data_collection
    if args.openrouter_require_parameters:
        preferences["require_parameters"] = True
    if args.openrouter_provider_order:
        preferences["order"] = args.openrouter_provider_order
    if args.no_openrouter_fallbacks:
        preferences["allow_fallbacks"] = False
    max_price: dict[str, str] = {}
    if args.openrouter_max_prompt_price is not None:
        max_price["prompt"] = args.openrouter_max_prompt_price
    if args.openrouter_max_completion_price is not None:
        max_price["completion"] = args.openrouter_max_completion_price
    if max_price:
        preferences["max_price"] = max_price
    if preferences and args.model_provider != "openrouter-anthropic":
        raise ValueError(
            "OpenRouter routing flags require --model-provider openrouter-anthropic"
        )
    config = OSWorldRunConfig(
        task_id=args.task_id,
        benchmark_release=source.release.tag,
        osworld_root=str(source.root),
        vm_path=args.path_to_vm,
        docker_ram_size_gb=args.docker_ram_size_gb,
        docker_cpu_cores=args.docker_cpu_cores,
        assets_dir=args.assets_dir,
        evaluation_examples_dir=str(source.evaluation_examples),
        api_key_file=args.api_key_file,
        result_root=args.result_root,
        website_host_suffix=args.website_host_suffix,
        base_url=resolve_base_url(args.model_provider, args.base_url),
        model=resolve_model(args.model_provider, args.model),
        reasoning_effort=args.reasoning_effort,
        total_steps=args.total_steps,
        compact_every=args.compact_every,
        headless=not args.no_headless,
        model_provider=args.model_provider,
        max_dollars=args.max_dollars,
        user_simulator_api_key_file=args.user_simulator_api_key_file,
        user_simulator_base_url=args.user_simulator_base_url,
        model_infrastructure=args.model_infrastructure,
        openrouter_provider_preferences=preferences,
        anthropic_min_recent_images=args.anthropic_min_recent_images,
        anthropic_image_prune_interval=args.anthropic_image_prune_interval,
        anthropic_max_request_mb=args.anthropic_max_request_mb,
        anthropic_max_output_tokens=args.anthropic_max_output_tokens,
        no_subagents=no_subagents,
        prompt_profile=args.prompt_profile,
        disable_task_proxy=args.disable_task_proxy,
        openai_policy_feedback_max_retries=(args.openai_policy_feedback_retries),
    )
    retry_budget = config.max_dollars
    for attempt in range(2):
        attempt_config = replace(config, max_dollars=retry_budget)
        runner = OSWorldRunner(attempt_config)
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
                    "Task failed due to guest infrastructure; retrying once "
                    "in a fresh VM: %s",
                    exc,
                )
                continue
            raise


if __name__ == "__main__":
    main()

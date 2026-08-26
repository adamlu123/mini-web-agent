"""Docker-backed terminal environment for RST (Recursive Synthetic Terminal Tasks).

Each task package ships its own runtime:

    <tasks_root>/<task_id>/
        instruction.md
        task.toml
        environment/Dockerfile      <- built into the image the agent lives in
        solution/solve.sh           <- PRIVATE, never mounted
        tests/test.sh               <- PRIVATE, never mounted
        tests/test_state.py

This environment builds that image, starts one long-lived container per task, and
runs every agent command inside it with ``docker exec``. Two directories matter:

* **task workspace** (the image's ``WORKDIR``, usually ``/app``) — the state the
  private verifier inspects afterwards. Harness artifacts must stay out of it.
* **harness mount** (``/harness`` in the container) — a bind mount of the host
  output directory, holding ``plan.md``, ``judge_config.json`` and
  ``final_runs/run_<id>/``. Bind-mounting it is what lets ``DefaultAgent``'s
  completion gate read ``judge_result.json`` from the host while the agent writes
  it from inside the container.

``serialize()`` reports the **host** path (the gate resolves it from there, see
``DefaultAgent._host_workspace_dir``) while ``get_template_vars()`` reports the
container-side alias (what the model is told to use), matching the existing
``workspace_alias`` split in ``local_workspace``.

The ``self_reflect`` shim
-------------------------
``miniswewebagent.tools.self_reflection`` needs the repo, its venv and the judge
credentials, none of which exist inside a task container. Any command whose first
token is ``self_reflect`` is therefore executed **on the host** instead, with
``/harness`` rewritten to the real output directory. Everything else runs in the
container.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

SELF_REFLECT_SHIM = "self_reflect"


class TerminalDockerEnvironmentConfig(BaseModel):
    tasks_root: Path = Path(".")
    output_dir: Path = Path("outputs/terminal/default")
    harness_mount: str = "/harness"
    private_dirs: list[str] = Field(default_factory=lambda: ["tests", "solution"])
    platform: str = "linux/amd64"
    reuse_images: bool = True
    build_timeout_seconds: int = 900
    command_timeout_seconds: int = 240
    output_truncation_chars: int = 24000
    recent_files_limit: int = 40
    shell: str = "/bin/bash"
    env: dict[str, str] = Field(default_factory=dict)
    docker_binary: str = "docker"
    # Fallback when the built image declares no WORKDIR.
    default_task_workspace: str = "/app"
    # Interpreter used for the host-side self_reflect shim. Defaults to the
    # interpreter running the harness, so it inherits the same venv.
    host_python: str = ""
    # Judge routing for the self_reflect shim. self_reflection defaults to TRAPI
    # Kimi (needs `az login --scope api://trapi`), which a laptop will not have.
    # Setting judge_endpoint + judge_model routes it through the OpenAI
    # chat-completions path instead, via the `policy` sentinel.
    judge_endpoint: str = ""
    judge_model: str = ""
    judge_api_key_env: str = ""
    # "" = infer from judge_endpoint: a /responses URL (phyagi gateway) takes the
    # real deployment name, anything else is an OpenAI chat-completions server and
    # needs self_reflection's `policy` sentinel, since passing a chat URL with a
    # real model name routes it to the responses backend instead. Force with
    # "responses" or "policy_chat".
    judge_backend: str = ""
    container_name_prefix: str = "rst"
    remove_container_on_close: bool = True


class TerminalDockerEnvironment:
    # This environment executes shell commands only.
    accepts_legacy_python_code = True

    def __init__(self, *, config_class: type = TerminalDockerEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.config.output_dir = self.config.output_dir.expanduser()
        self.config.tasks_root = self.config.tasks_root.expanduser()
        self._step_index = 0
        self._prepared_task: dict[str, Any] = {}
        self._task_id: str = ""
        self._task_dir: Path | None = None
        self._image_tag: str = ""
        self._container: str = ""
        self._task_workspace: str = self.config.default_task_workspace

    # ---------------------------------------------------------------- helpers

    def _workspace_dir(self) -> Path:
        return self.config.output_dir

    def _steps_dir(self) -> Path:
        return self._workspace_dir() / "steps"

    def _logs_dir(self) -> Path:
        return self._workspace_dir() / "logs"

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}\n\n... [{len(text) - limit} characters omitted]"

    def _docker(self, *args: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.config.docker_binary, *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

    def _environment_digest(self, environment_dir: Path) -> str:
        """Content hash over the build context, so identical envs share an image."""
        digest = hashlib.sha256()
        digest.update(self.config.platform.encode())
        for path in sorted(p for p in environment_dir.rglob("*") if p.is_file()):
            digest.update(str(path.relative_to(environment_dir)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()[:16]

    def _image_exists(self, tag: str) -> bool:
        return self._docker("image", "inspect", tag).returncode == 0

    # ---------------------------------------------------------------- prepare

    def prepare(self, **kwargs) -> None:
        self._prepared_task = dict(kwargs)
        self._step_index = 0
        self._task_id = str(kwargs.get("task_id") or "").strip()
        if not self._task_id:
            raise ValueError("terminal_docker requires task_id to locate the task package.")

        task_dir = (self.config.tasks_root / self._task_id).resolve()
        if not task_dir.is_dir():
            raise FileNotFoundError(f"RST task package not found: {task_dir}")
        self._task_dir = task_dir

        environment_dir = task_dir / "environment"
        dockerfile = environment_dir / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError(f"Task package has no environment/Dockerfile: {task_dir}")
        for compose_name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml"):
            if (environment_dir / compose_name).is_file() or (task_dir / compose_name).is_file():
                raise NotImplementedError(
                    f"Task {self._task_id} ships {compose_name}; multi-container tasks are not "
                    "supported by terminal_docker yet."
                )

        workspace_dir = self._workspace_dir()
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self._steps_dir().mkdir(parents=True, exist_ok=True)
        self._logs_dir().mkdir(parents=True, exist_ok=True)
        (workspace_dir / "task.json").write_text(json.dumps(kwargs, indent=2, default=str), encoding="utf-8")

        # --- build ---------------------------------------------------------
        self._image_tag = f"{self.config.container_name_prefix}:{self._environment_digest(environment_dir)}"
        if not (self.config.reuse_images and self._image_exists(self._image_tag)):
            build = self._docker(
                "build",
                "--platform", self.config.platform,
                "-t", self._image_tag,
                "-f", str(dockerfile),
                str(environment_dir),
                timeout=self.config.build_timeout_seconds,
            )
            (workspace_dir / "docker_build.log").write_text(
                (build.stdout or "") + (build.stderr or ""), encoding="utf-8"
            )
            if build.returncode != 0:
                raise RuntimeError(
                    f"docker build failed for {self._task_id} (see docker_build.log):\n"
                    f"{(build.stderr or build.stdout or '')[-2000:]}"
                )

        # --- task workspace = image WORKDIR --------------------------------
        inspect = self._docker("image", "inspect", "-f", "{{.Config.WorkingDir}}", self._image_tag)
        workdir = (inspect.stdout or "").strip()
        self._task_workspace = workdir or self.config.default_task_workspace

        # --- run ------------------------------------------------------------
        self._container = (
            f"{self.config.container_name_prefix}-{self._task_id[-12:]}-{os.getpid()}"
        )
        self._docker("rm", "-f", self._container)
        run_args = [
            "run", "-d",
            "--name", self._container,
            "--platform", self.config.platform,
            "-v", f"{workspace_dir.resolve()}:{self.config.harness_mount}",
            "-w", self._task_workspace,
            "--entrypoint", "/bin/sh",
        ]
        for key, value in self.config.env.items():
            run_args += ["-e", f"{key}={value}"]
        run_args += [self._image_tag, "-c", "sleep infinity"]
        started = self._docker(*run_args, timeout=self.config.build_timeout_seconds)
        if started.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {self._task_id}:\n{(started.stderr or started.stdout or '')[-2000:]}"
            )

        # The private verifier and reference solution are deliberately absent from
        # the container: nothing mounts task_dir itself, only the harness dir.

    # ---------------------------------------------------------------- execute

    def _run_self_reflect_on_host(self, command: str) -> tuple[str, int]:
        """Execute the self_reflect shim on the host, not in the container."""
        host_workspace = str(self._workspace_dir().resolve())
        try:
            argv = shlex.split(command)[1:]
        except ValueError as exc:
            return f"self_reflect: could not parse arguments: {exc}", 2
        argv = [arg.replace(self.config.harness_mount, host_workspace) for arg in argv]

        python = self.config.host_python or sys.executable
        full = [python, "-m", "miniswewebagent.tools.self_reflection"]
        if not any(a == "--config" for a in argv):
            full += ["--config", f"{host_workspace}/judge_config.json"]
        if not any(a == "--workspace-dir" for a in argv):
            full += ["--workspace-dir", host_workspace]
        # Route the judge explicitly when configured. `--model policy` is the
        # sentinel that selects the chat-completions backend; the real deployment
        # name comes from OPENAI_COMPATIBLE_MODEL below.
        judge_env = dict(os.environ)
        if self.config.judge_endpoint and not any(a == "--endpoint" for a in argv):
            backend = self.config.judge_backend or (
                "responses" if "/responses" in self.config.judge_endpoint.lower() else "policy_chat"
            )
            full += ["--endpoint", self.config.judge_endpoint]
            key = os.environ.get(self.config.judge_api_key_env, "") if self.config.judge_api_key_env else ""
            if backend == "responses":
                # phyagi gateway: real deployment name, key from OPENAI_GATEWAY_API_KEY.
                if self.config.judge_model:
                    full += ["--model", self.config.judge_model]
                if key:
                    full += ["--api-key", key]
            else:
                # OpenAI chat-completions (Azure, vLLM): the sentinel selects that
                # backend; the served model name comes from the environment.
                full += ["--model", "policy"]
                judge_env["OPENAI_COMPATIBLE_ENDPOINT"] = self.config.judge_endpoint
                if self.config.judge_model:
                    judge_env["OPENAI_COMPATIBLE_MODEL"] = self.config.judge_model
                if key:
                    full += ["--api-key", key]
                    judge_env["OPENAI_COMPATIBLE_API_KEY"] = key
        full += argv
        try:
            result = subprocess.run(
                full, text=True, capture_output=True,
                timeout=self.config.command_timeout_seconds * 4,
                encoding="utf-8", errors="replace", env=judge_env,
            )
        except Exception as exc:  # noqa: BLE001
            return f"self_reflect failed on host: {exc}", -1
        output = (result.stdout or "") + (result.stderr or "")
        return output.replace(host_workspace, self.config.harness_mount), result.returncode

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        self._step_index += 1
        command = str(
            action.get("command") or action.get("bash_command") or action.get("python_code") or ""
        ).strip()

        self._steps_dir().mkdir(parents=True, exist_ok=True)
        (self._steps_dir() / f"step_{self._step_index:04d}.sh").write_text(
            command.rstrip() + "\n", encoding="utf-8"
        )

        exception_info = ""
        if command.split(maxsplit=1)[:1] == [SELF_REFLECT_SHIM]:
            output, returncode = self._run_self_reflect_on_host(command)
        else:
            exec_cwd = cwd or self._task_workspace
            try:
                result = self._docker(
                    "exec", "-w", exec_cwd, self._container,
                    self.config.shell, "-lc", command,
                    timeout=self.config.command_timeout_seconds,
                )
                output = (result.stdout or "") + (result.stderr or "")
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                output = ""
                returncode = -1
                exception_info = (
                    f"Command timed out after {self.config.command_timeout_seconds}s "
                    "and was killed."
                )
            except Exception as exc:  # noqa: BLE001
                output = ""
                returncode = -1
                exception_info = f"An error occurred while executing the command: {exc}"

        if output:
            self._logs_dir().mkdir(parents=True, exist_ok=True)
            (self._logs_dir() / f"step_{self._step_index:04d}.log").write_text(output, encoding="utf-8")

        observation = {
            "success": returncode == 0 and not exception_info,
            "exception": exception_info,
            "command": command,
            "returncode": returncode,
            "cwd": cwd or self._task_workspace,
            "workspace_dir": self.config.harness_mount,
            "task_workspace": self._task_workspace,
            "command_output": self._truncate(output, self.config.output_truncation_chars),
            "workspace_files": self._recent_task_files(),
            # Fields the shared observation/judge plumbing expects to exist.
            "url": "",
            "title": "",
            "aria_snapshot": "",
            "console_output": "",
            "recent_console": "",
            "screenshot_path": "",
            "new_screenshots": [],
            "recent_screenshots": [],
        }
        return {
            "output": output,
            "returncode": returncode,
            "exception_info": exception_info,
            "observation": observation,
        }

    def _recent_task_files(self) -> list[str]:
        """Newest files in the task workspace, so the model sees what it changed."""
        result = self._docker(
            "exec", self._container, "/bin/sh", "-c",
            f"find {shlex.quote(self._task_workspace)} -type f -printf '%T@ %p\\n' 2>/dev/null "
            f"| sort -rn | head -n {self.config.recent_files_limit} | cut -d' ' -f2-",
        )
        if result.returncode != 0:
            return []
        return [line for line in (result.stdout or "").splitlines() if line.strip()]

    # ---------------------------------------------------------------- verify

    def run_private_verifier(self) -> dict[str, Any]:
        """Post-hoc scoring: copy tests/ in, run test.sh, read the Harbor reward.

        Called by the runner AFTER the episode ends. Never reachable by the agent:
        the files are copied in only at this point, and the container is discarded
        immediately afterwards.
        """
        if self._task_dir is None or not self._container:
            return {"score": None, "error": "environment not prepared"}
        tests_dir = self._task_dir / "tests"
        if not tests_dir.is_dir():
            return {"score": None, "error": "task package has no tests/"}

        copied = self._docker("cp", f"{tests_dir}/.", f"{self._container}:/tests")
        if copied.returncode != 0:
            return {"score": None, "error": f"docker cp failed: {copied.stderr[-400:]}"}

        # Harbor provisions /logs/verifier before invoking test.sh. Most RST tasks
        # also mkdir it themselves, but only 51/100 TB-Lite tasks do, and the rest
        # fail their final `echo N > /logs/verifier/reward.txt` redirect even when
        # every assertion passed. Creating it here is the harness's job.
        self._docker("exec", self._container, "/bin/sh", "-c", "mkdir -p /logs/verifier")

        try:
            result = self._docker(
                "exec", "-w", self._task_workspace, self._container,
                "/bin/bash", "-lc", "bash /tests/test.sh",
                timeout=max(self.config.command_timeout_seconds * 4, 900),
            )
            log = (result.stdout or "") + (result.stderr or "")
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            log, returncode = "verifier timed out", -1

        (self._workspace_dir() / "verifier_log.txt").write_text(log, encoding="utf-8")

        reward = self._docker(
            "exec", self._container, "/bin/sh", "-c", "cat /logs/verifier/reward.txt 2>/dev/null"
        )
        score: int | None = None
        raw = (reward.stdout or "").strip()
        if raw:
            try:
                score = int(float(raw))
            except ValueError:
                score = None

        ctrf = self._docker(
            "exec", self._container, "/bin/sh", "-c", "cat /logs/verifier/ctrf.json 2>/dev/null"
        )
        partial: dict[str, Any] = {}
        if (ctrf.stdout or "").strip():
            try:
                summary = json.loads(ctrf.stdout)["results"]["summary"]
                total = int(summary.get("tests") or 0)
                passed = int(summary.get("passed") or 0)
                partial = {
                    "tests_total": total,
                    "tests_passed": passed,
                    "partial_credit": (passed / total) if total else None,
                }
            except Exception:  # noqa: BLE001
                partial = {}

        payload = {"score": score, "returncode": returncode, **partial}
        (self._workspace_dir() / "verifier_result.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return payload

    # ---------------------------------------------------------------- wiring

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "workspace_dir": self.config.harness_mount,
            "task_workspace": self._task_workspace,
            "task_id": self._task_id,
            "start_url": "",
            "output_dir": str(self._workspace_dir()),
            **kwargs,
        }

    def serialize(self) -> dict:
        return {
            "environment": {
                "config": self.config.model_dump(mode="json"),
                "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                # Host path: DefaultAgent._host_workspace_dir reads this to find
                # final_runs/run_<id>/judge_result.json for the completion gate.
                "workspace_dir": str(self._workspace_dir().resolve()),
                "task_workspace": self._task_workspace,
                "image": self._image_tag,
                "container": self._container,
            }
        }

    def close(self) -> None:
        if self._container and self.config.remove_container_on_close:
            self._docker("rm", "-f", self._container)
            self._container = ""

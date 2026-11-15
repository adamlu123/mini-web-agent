from __future__ import annotations

import ast
import base64
import hashlib
import json
import mmap
import os
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping
from unittest.mock import Mock, patch

import pytest

from cuawright.artifacts import ArtifactRef
from cuawright.budget import SharedStepBudget
from cuawright.config.prompt_registry import select_odysseys_prompt_bundle
from cuawright.config.prompts import compose_system_prompt
from cuawright.environments.base import CommandObservation, CommandResult
from cuawright.environments.control import (
    SpecialistInvocation,
    SpecialistRunResult,
    TerminalCommandBroker,
)
from cuawright.environments.odysseys.environment import (
    ODYSSEYS_FINAL_RESPONSE_PATH,
    ODYSSEYS_SCREENSHOT_DIR,
    OdysseysBenchmarkPlugin,
)
from cuawright.environments.odysseys.judge import (
    _MAX_JUDGE_OUTPUT_BYTES,
    _read_limited_regular_fd,
    _read_output_pass,
    verify_judge_inputs,
)
from cuawright.environments.odysseys.prompts import localized_role_instructions
from cuawright.environments.odysseys.source import (
    inspect_desktop_runtime,
    inspect_source,
)
from cuawright.environments.odysseys.task import OdysseysTask, load_odysseys_task
from cuawright.environments.desktop.plugin import GuestWorkspace
from cuawright.operations.scheduler import ProcessOutcome
from cuawright.run.odysseys import (
    DesktopRuntimeBindings,
    OdysseysRunConfig,
    OdysseysRunner,
    _activate_desktop_runtime,
    _require_no_odysseys_proxy,
    parse_args,
)
from cuawright.run.odysseys_judge import main as judge_main


TASK_ID = "a" * 40
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/wcAAusB9Wl2nWQAAAAASUVORK5CYII="
)
SUBMIT = "python /opt/cuawright-tools/submit.py"


class RecordingStore:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, object]]] = []
        self.artifacts: dict[str, bytes] = {}

    def event(self, event: str, payload: Mapping[str, object]) -> None:
        self.events.append((event, dict(payload)))

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        self.artifacts[path] = data
        return ArtifactRef(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media_type,
        )


class Controller:
    def run_python_script(self, script: str, timeout: int = 90):
        del script, timeout
        return {"status": "success", "output": ""}

    def get_screenshot(self) -> bytes:
        return PNG


class DesktopEnv:
    def __init__(self) -> None:
        self.controller = Controller()
        self.is_environment_used = False
        self.closed = False

    def close(self) -> None:
        self.closed = True


class MemoryOdysseysPlugin(OdysseysBenchmarkPlugin):
    def __init__(self, *args, **kwargs) -> None:
        self.files: dict[str, bytes] = {}
        super().__init__(*args, **kwargs)

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        del workspace
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path][:max_bytes]

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        del workspace
        self.files[path] = data


def create_source(root: Path) -> tuple[Path, Path, Path, str]:
    dataset = root / "data" / "odysseys.json"
    examples = root / "data" / "odysseys_cua_final"
    example = examples / "examples" / "mind2web_chrome" / f"{TASK_ID}.json"
    judge = root / "scripts" / "python" / "run_full_trajectory_per_rubric.py"
    example.parent.mkdir(parents=True)
    judge.parent.mkdir(parents=True)
    instruction = "Research the visible public web and summarize the result."
    dataset.write_text(
        json.dumps(
            [
                {
                    "task_id": TASK_ID,
                    "confirmed_task": instruction,
                    "website": "https://www.google.com",
                    "level": "easy",
                    "reference_length": 3,
                    "rubrics": {
                        "R1": {
                            "requirement": "Keep the evidence visible.",
                            "verification": "Inspect the saved screenshots.",
                        }
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    example.write_text(
        json.dumps(
            {
                "id": TASK_ID,
                "snapshot": "chrome",
                "instruction": instruction,
                "source": "https://www.google.com",
                "config": [],
                "evaluator": {"func": "infeasible"},
                "proxy": False,
            }
        ),
        encoding="utf-8",
    )
    judge.write_text("# exact external judge fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "fixture"],
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return dataset, examples, judge, commit


def create_runtime_source(root: Path) -> str:
    files = {
        "desktop_env/__init__.py": "",
        "desktop_env/desktop_env.py": "class DesktopEnv:\n    pass\n",
        "desktop_env/providers/__init__.py": "",
        "desktop_env/providers/docker/__init__.py": "",
        ("desktop_env/providers/docker/provider.py"): "DEFAULT_DISK_SIZE_GB = 64\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "runtime fixture"],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def task(root: Path) -> tuple[OdysseysTask, Path, str]:
    dataset, examples, judge, commit = create_source(root)
    loaded = load_odysseys_task(
        task_id=TASK_ID,
        dataset_path=dataset,
        examples_dir=examples,
    )
    return loaded, judge, commit


def plugin(tmp_path: Path):
    loaded, judge, commit = task(tmp_path / "source")
    env = DesktopEnv()
    store = RecordingStore()

    def execute(command: str, timeout: int | None) -> CommandResult:
        del timeout
        argv = shlex.split(command)
        if len(argv) >= 2 and argv[1].startswith("/opt/cuawright-tools/"):
            operation = Path(argv[1]).stem
            payload: dict[str, Any] = {
                "cuawright_control": 1,
                "operation": operation,
            }
            values = iter(argv[2:])
            for name in values:
                value = next(values)
                payload[name.removeprefix("--").replace("-", "_")] = (
                    int(value) if name == "--max-steps" else value
                )
            return CommandResult(0, stdout=json.dumps(payload) + "\n")
        return CommandResult(0, stdout="ok", inspected=True)

    judge_dir = tmp_path / "run" / "odysseys-judge" / TASK_ID
    adapter = MemoryOdysseysPlugin(
        env,
        execute,
        workspace=GuestWorkspace("/home/user/.agent", "/home/user/.agent"),
        task=loaded,
        judge_dir=judge_dir,
        judge_script=judge,
        source_identity={
            "commit": commit,
            "clean": True,
            "dataset_sha256": loaded.dataset_sha256,
            "judge_sha256": hashlib.sha256(judge.read_bytes()).hexdigest(),
        },
        run_store=store,
    )
    return adapter, env, store, judge_dir


def rows(judge_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (judge_dir / "traj.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_task_loading_and_clean_source_identity(tmp_path: Path) -> None:
    dataset, examples, judge, commit = create_source(tmp_path / "source")
    source = inspect_source(
        root=tmp_path / "source",
        expected_commit=commit,
        dataset_path=dataset,
        examples_dir=examples,
        judge_script=judge,
    )
    loaded = load_odysseys_task(
        task_id=TASK_ID,
        dataset_path=dataset,
        examples_dir=examples,
    )

    assert source.commit == commit
    assert source.dataset_sha256 == loaded.dataset_sha256
    assert loaded.task_record["rubrics"]["R1"]["requirement"]
    assert loaded.instruction == loaded.example["instruction"]
    with pytest.raises(RuntimeError, match="commit mismatch"):
        inspect_source(
            root=tmp_path / "source",
            expected_commit="b" * 40,
            dataset_path=dataset,
            examples_dir=examples,
            judge_script=judge,
        )
    dataset.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="dirty"):
        inspect_source(
            root=tmp_path / "source",
            expected_commit=commit,
            dataset_path=dataset,
            examples_dir=examples,
            judge_script=judge,
        )


def test_desktop_runtime_is_clean_preferred_and_restored(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    commit = create_runtime_source(runtime_root)
    inspected = inspect_desktop_runtime(runtime_root, commit)
    assert inspected.root == runtime_root.resolve()

    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    wrong_package = ModuleType("desktop_env")
    wrong_package.__file__ = str(tmp_path / "wrong" / "desktop_env" / "__init__.py")
    wrong_desktop = ModuleType("desktop_env.desktop_env")
    wrong_desktop.__file__ = str(tmp_path / "wrong" / "desktop_env" / "desktop_env.py")
    with patch.dict(
        sys.modules,
        {
            "desktop_env": wrong_package,
            "desktop_env.desktop_env": wrong_desktop,
        },
    ):
        with _activate_desktop_runtime(runtime_root, commit) as runtime:
            assert sys.path[0] == str(runtime_root.resolve())
            loaded = sys.modules["desktop_env.desktop_env"]
            assert (
                Path(loaded.__file__).resolve().is_relative_to(runtime_root.resolve())
            )
            assert runtime.desktop_env_type.__module__ == "desktop_env.desktop_env"
            assert runtime.default_disk_size_gb == 64
        assert sys.path == previous_path
        assert sys.dont_write_bytecode is previous_dont_write_bytecode
        assert sys.modules["desktop_env"] is wrong_package
        assert sys.modules["desktop_env.desktop_env"] is wrong_desktop

        def activate_once(_: int) -> str:
            with _activate_desktop_runtime(runtime_root, commit) as runtime:
                return str(runtime.source.root)

        assert [activate_once(0), activate_once(1)] == [
            str(runtime_root.resolve()),
            str(runtime_root.resolve()),
        ]
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert (
                list(executor.map(activate_once, range(4)))
                == [str(runtime_root.resolve())] * 4
            )

    assert not list(runtime_root.rglob("__pycache__"))
    assert (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(runtime_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
        ).strip()
        == ""
    )
    (runtime_root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="runtime checkout is dirty"):
        inspect_desktop_runtime(runtime_root, commit)


def test_prompt_bundle_is_independent_and_hash_pinned() -> None:
    bundle = select_odysseys_prompt_bundle()
    roles = localized_role_instructions()
    assert bundle.name == "odysseys-default"
    assert bundle.benchmark == "odysseys"
    assert set(roles) == {"actor", "binder", "critic", "verifier"}
    assert {
        role: hashlib.sha256(
            compose_system_prompt(role, bundle).encode("utf-8")
        ).hexdigest()
        for role in roles
    } == {
        "actor": "21f6b7a28cbc58ca83c22bfce5193fe4601258d9e843cfdce2759d4388d8000c",
        "binder": "db9c46018965b8c8f63374302ae0f24be3bc03201301bd78632e2432e2828c05",
        "critic": "8d646351754eec746dc8bdd9f5c1da92451d5caa640f356aaa1e4798b3e4320b",
        "verifier": "c89b3d9e09ca371e2661461e92c759e088b005e5afc6c1145a50e1f9db54b9c7",
    }
    actor = compose_system_prompt("actor", bundle)
    assert ODYSSEYS_FINAL_RESPONSE_PATH in actor
    assert "after every dispatched Actor and specialist command" in " ".join(
        actor.split()
    )

    package = Path(__file__).parents[1] / "src" / "cuawright"
    imports = set()
    for path in (package / "environments" / "odysseys").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
    assert not any(
        module.startswith(
            (
                "cuawright.environments.osworld",
                "cuawright.environments.weavebench",
                "cuawright.environments.benchcad",
                "cuawright.environments.cadgenbench",
            )
        )
        for module in imports
    )


def test_broker_capture_orders_actor_invocations_and_all_specialist_commands(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path)

    def specialist(
        invocation: SpecialistInvocation,
        broker: TerminalCommandBroker,
    ) -> SpecialistRunResult:
        result = broker.run_command(f"{invocation.role} ordinary command")
        assert result.exit_code == 0
        statuses = {
            "binder": "complete",
            "critic": "accept",
            "verifier": "ready",
        }
        report = broker.accept_text_report(
            json.dumps(
                {
                    "status": statuses[invocation.role],
                    "report": "Visible evidence inspected.",
                    "unresolved": [],
                }
            )
        )
        return SpecialistRunResult(report.completed, 0)

    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        specialist_runner=specialist,
        enabled_specialist_roles=("binder", "critic", "verifier"),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    invocations = [
        (
            f"python /opt/cuawright-tools/{role}.py "
            "--request 'Inspect visible evidence' --max-steps 1"
        )
        for role in ("binder", "critic", "verifier")
    ]
    assert all(broker.run_command(command).exit_code == 0 for command in invocations)
    captured = rows(judge_dir)
    assert [row["action"] for row in captured] == [
        invocations[0],
        "binder ordinary command",
        invocations[1],
        "critic ordinary command",
        invocations[2],
        "verifier ordinary command",
    ]
    assert [row["info"]["role"] for row in captured] == [
        "actor",
        "binder",
        "actor",
        "critic",
        "actor",
        "verifier",
    ]
    assert [row["info"]["stage"] for row in captured] == [
        "orchestrate",
        "specialist-binder-01",
        "orchestrate",
        "specialist-critic-01",
        "orchestrate",
        "specialist-verifier-01",
    ]


def test_rejected_accepted_and_budget_forced_submission_finality(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path / "accepted")
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )

    rejected = broker.run_command(SUBMIT)
    assert rejected.exit_code == 2
    assert not rows(judge_dir)[-1]["final"]
    assert adapter.submission_classification is None

    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    accepted = broker.run_command(SUBMIT)
    assert accepted.completed
    assert rows(judge_dir)[-1]["final"]
    assert rows(judge_dir)[-1]["info"]["submission_classification"] == "accepted"
    assert adapter.submission_classification == "accepted"

    forced, _, forced_store, forced_dir = plugin(tmp_path / "budget-forced")
    forced.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Best available answer.\n"
    budget = SharedStepBudget(1)
    budget.consume("actor", "last-model-turn")
    forced_broker = TerminalCommandBroker(
        forced,
        budget=budget,
        enabled_specialist_roles=("binder",),
        run_store=forced_store,
        command_observer=forced.observe_command,
    )
    forced_result = forced_broker.run_command(SUBMIT)
    assert forced_result.completed
    assert rows(forced_dir)[-1]["info"]["submission_classification"] == "forced"
    assert forced.submission_classification == "forced"
    forced_evaluation = forced.evaluate()
    assert forced_evaluation["submission_classification"] == "forced"
    assert forced_evaluation["status"] == "ready_for_external_judge"

    missing, _, missing_store, _ = plugin(tmp_path / "forced-without-response")
    missing_broker = TerminalCommandBroker(
        missing,
        budget=SharedStepBudget(1),
        enabled_specialist_roles=(),
        run_store=missing_store,
        command_observer=missing.observe_command,
    )
    missing_broker.force_submission('{"status":"forced"}')
    missing_evaluation = missing.evaluate()
    assert missing_evaluation["submission_classification"] == "forced"
    assert missing_evaluation["status"] == "incomplete_judge_evidence"
    assert missing_evaluation["judge_command"] is None
    assert "file does not exist" in missing_evaluation["evidence_error"]


def test_screenshots_final_response_cleanup_and_immutable_judge_inputs(
    tmp_path: Path,
) -> None:
    adapter, env, store, judge_dir = plugin(tmp_path)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)

    evaluation = adapter.evaluate()
    manifest = judge_dir / "judge-input.json"
    manifest_digest = evaluation["judge_manifest_sha256"]
    assert evaluation["status"] == "ready_for_external_judge"
    assert evaluation["judge_command"][1:3] == [
        "-m",
        "cuawright.run.odysseys_judge",
    ]
    assert evaluation["judge_command"][-2:] == [
        "--manifest-sha256",
        manifest_digest,
    ]
    assert (judge_dir / "final-response.md").read_text().strip() == (
        "Final visible answer."
    )
    assert (judge_dir / "screenshots" / "step_0001.png").read_bytes() == PNG
    assert adapter.files[f"{ODYSSEYS_SCREENSHOT_DIR}/step_0001.png"] == PNG
    config = json.loads((judge_dir / "config.json").read_text())
    assert config["copied_inputs"]["judge_script_sha256"]
    assert config["copied_inputs"]["task_source_sha256"]
    assert (judge_dir / "inputs" / "odysseys.json").read_bytes() == (
        adapter.task.dataset_bytes
    )
    assert (
        judge_dir / "inputs" / "run_full_trajectory_per_rubric.py"
    ).read_bytes() == adapter.external_judge_script.read_bytes()
    verify_judge_inputs(manifest, manifest_digest)

    selected = judge_dir / "inputs" / "selected-task.json"
    selected_bytes = selected.read_bytes()
    selected.chmod(0o644)
    selected.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after finalization"):
        verify_judge_inputs(manifest, manifest_digest)
    selected.write_bytes(selected_bytes)

    manifest_bytes = manifest.read_bytes()
    manifest.chmod(0o644)
    injected = json.loads(manifest_bytes)
    injected["judge_argv"] = ["-c", "raise SystemExit('injected')"]
    manifest.write_text(json.dumps(injected), encoding="utf-8")
    injected_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="fixed schema"):
        verify_judge_inputs(manifest, injected_digest)

    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        verify_judge_inputs(manifest, manifest_digest)
    manifest.write_bytes(manifest_bytes)

    adapter.external_judge_script.write_text("# mutated judge\n", encoding="utf-8")
    verify_judge_inputs(manifest, manifest_digest)

    adapter.close()
    adapter.close()
    assert env.closed


def test_deleted_screenshot_before_evaluate_is_not_judge_ready(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    (judge_dir / "screenshots" / "step_0001.png").unlink()

    evaluation = adapter.evaluate()

    assert evaluation["status"] == "incomplete_judge_evidence"
    assert evaluation["judge_command"] is None
    assert "without canonical hashes" in evaluation["evidence_error"]


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("duplicate", "references must be unique"),
        ("unsafe", "unsafe trajectory screenshot reference"),
        ("missing", "without canonical hashes"),
        ("orphan", "orphan files"),
        ("response", "final trajectory response"),
        ("response-artifact", "manifest final response"),
    ),
)
def test_judge_manifest_reconciles_screenshot_references(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path / case)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    evaluation = adapter.evaluate()
    manifest_path = judge_dir / "judge-input.json"
    manifest_path.chmod(0o600)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if case in {"duplicate", "unsafe", "response"}:
        trajectory_path = judge_dir / "traj.jsonl"
        trajectory = [
            json.loads(line)
            for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        ]
        if case == "response":
            trajectory[-1]["response"] = ""
        else:
            trajectory[-1]["screenshot_file"] = (
                trajectory[0]["screenshot_file"]
                if case == "duplicate"
                else "../unsafe.png"
            )
        trajectory_path.write_text(
            "".join(json.dumps(row) + "\n" for row in trajectory),
            encoding="utf-8",
        )
        manifest["artifact_hashes"]["traj.jsonl"] = hashlib.sha256(
            trajectory_path.read_bytes()
        ).hexdigest()
    elif case == "missing":
        manifest["artifact_hashes"].pop("screenshots/step_0001.png")
    elif case == "orphan":
        orphan = judge_dir / "screenshots" / "step_9999.png"
        orphan.write_bytes(PNG)
        manifest["artifact_hashes"]["screenshots/step_9999.png"] = hashlib.sha256(
            PNG
        ).hexdigest()
    else:
        final_response = judge_dir / "final-response.md"
        final_response.chmod(0o600)
        final_response.write_text("Different response.\n", encoding="utf-8")
        manifest["artifact_hashes"]["final-response.md"] = hashlib.sha256(
            final_response.read_bytes()
        ).hexdigest()

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match=message):
        verify_judge_inputs(manifest_path, manifest_digest)
    assert evaluation["status"] == "ready_for_external_judge"


def test_judge_invocation_pins_verified_parent_and_fixed_inputs(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    evaluation = adapter.evaluate()
    manifest = judge_dir / "judge-input.json"
    moved = judge_dir.with_name(judge_dir.name + "-original")
    (judge_dir / "eval_results_full_traj_per_rubric.json").write_text(
        '{"old": true}\n',
        encoding="utf-8",
    )
    expected_config = (judge_dir / "config.json").read_bytes()
    expected_trajectory = (judge_dir / "traj.jsonl").read_bytes()
    expected_screenshot = (judge_dir / "screenshots" / "step_0001.png").read_bytes()

    def inspect_run(jobs, *, width):
        assert width == 1
        job = jobs[0]
        argv = list(job.command)
        assert len(job.pass_fds) == 2
        judge_dir.rename(moved)
        judge_dir.mkdir(parents=True)
        (judge_dir / "screenshots").mkdir()
        (judge_dir / "config.json").write_text("malicious", encoding="utf-8")
        (judge_dir / "traj.jsonl").write_text("malicious", encoding="utf-8")
        (judge_dir / "screenshots" / "step_0001.png").write_bytes(b"malicious")
        assert Path(argv[1]).read_text(encoding="utf-8") == (
            "# exact external judge fixture\n"
        )
        assert (
            json.loads(Path(argv[7]).read_text(encoding="utf-8"))[0]["task_id"]
            == TASK_ID
        )
        assert argv[2:4] == ["--model", "gemini-3.1-flash-lite-preview"]
        assert "-c" not in argv
        assert argv[5].startswith("/proc/self/fd/")
        assert argv[9].startswith("/proc/self/fd/")
        snapshot = Path(argv[5])
        assert not (snapshot / "eval_results_full_traj_per_rubric.json").exists()
        assert (snapshot / "config.json").read_bytes() == expected_config
        assert (snapshot / "traj.jsonl").read_bytes() == expected_trajectory
        assert (
            snapshot / "screenshots" / "step_0001.png"
        ).read_bytes() == expected_screenshot
        Path(argv[9]).write_text('{"score": 1}\n', encoding="utf-8")
        return (ProcessOutcome(job.job_id, 0, 123, pgid=123),)

    with patch(
        "cuawright.run.odysseys_judge.run_process_jobs",
        side_effect=inspect_run,
    ):
        assert (
            judge_main(
                [
                    "--manifest",
                    str(manifest),
                    "--manifest-sha256",
                    evaluation["judge_manifest_sha256"],
                ]
            )
            == 0
        )
    assert (moved / "eval_results_full_traj_per_rubric.json").read_text() == (
        '{"score": 1}\n'
    )
    adapter.close()


@pytest.mark.parametrize(
    ("return_code", "status", "output", "error", "expected_return"),
    (
        (1, "completed", '{"partial": true}\n', None, 1),
        (0, "completed", "{", "not valid JSON", None),
        (
            0,
            "descendants_terminated",
            '{"late": true}\n',
            None,
            1,
        ),
    ),
)
def test_judge_failure_does_not_replace_canonical_output(
    tmp_path: Path,
    return_code: int,
    status: str,
    output: str,
    error: str | None,
    expected_return: int | None,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path / f"case-{return_code}-{error}")
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    evaluation = adapter.evaluate()
    canonical = judge_dir / "eval_results_full_traj_per_rubric.json"
    canonical.write_text('{"old": true}\n', encoding="utf-8")

    def failed_run(jobs, *, width):
        assert width == 1
        job = jobs[0]
        argv = list(job.command)
        assert job.pass_fds
        Path(argv[9]).write_text(output, encoding="utf-8")
        return (
            ProcessOutcome(
                job.job_id,
                return_code,
                123,
                status=status,
                pgid=123,
            ),
        )

    call = lambda: judge_main(
        [
            "--manifest",
            str(judge_dir / "judge-input.json"),
            "--manifest-sha256",
            evaluation["judge_manifest_sha256"],
        ]
    )
    with patch(
        "cuawright.run.odysseys_judge.run_process_jobs",
        side_effect=failed_run,
    ):
        if error is None:
            assert call() == expected_return
        else:
            with pytest.raises(RuntimeError, match=error):
                call()
    assert canonical.read_text(encoding="utf-8") == '{"old": true}\n'


def test_sparse_oversized_judge_output_is_rejected_before_publish(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    evaluation = adapter.evaluate()
    canonical = judge_dir / "eval_results_full_traj_per_rubric.json"
    canonical.write_text('{"old": true}\n', encoding="utf-8")

    def oversized_run(jobs, *, width):
        assert width == 1
        job = jobs[0]
        argv = list(job.command)
        os.truncate(argv[9], _MAX_JUDGE_OUTPUT_BYTES + 1)
        return (ProcessOutcome(job.job_id, 0, 123, pgid=123),)

    with (
        patch(
            "cuawright.run.odysseys_judge.run_process_jobs",
            side_effect=oversized_run,
        ),
        pytest.raises(RuntimeError, match="exceeds the size limit"),
    ):
        judge_main(
            [
                "--manifest",
                str(judge_dir / "judge-input.json"),
                "--manifest-sha256",
                evaluation["judge_manifest_sha256"],
            ]
        )
    assert canonical.read_text(encoding="utf-8") == '{"old": true}\n'


def test_growing_judge_output_is_rejected_during_streaming_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "growing.json"
    output.write_text('{"score": 1}\n', encoding="utf-8")
    descriptor = os.open(output, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        values = list(before)
        values[6] = before.st_size + 1
        after = os.stat_result(values)
        with (
            patch(
                "cuawright.environments.odysseys.judge.os.fstat",
                side_effect=(before, after),
            ),
            pytest.raises(RuntimeError, match="changed while being read"),
        ):
            _read_limited_regular_fd(descriptor, _MAX_JUDGE_OUTPUT_BYTES)
    finally:
        os.close(descriptor)


def test_same_size_concurrent_judge_output_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "same-size.json"
    output.write_text('{"score": 1}\n', encoding="utf-8")
    descriptor = os.open(output, os.O_RDONLY)
    mutate = threading.Event()
    mutated = threading.Event()

    def mutate_output() -> None:
        assert mutate.wait(5)
        output.write_text('{"score": 2}\n', encoding="utf-8")
        mutated.set()

    thread = threading.Thread(target=mutate_output)
    thread.start()

    def read_pass(fd: int, limit: int, *, capture: bool):
        result = _read_output_pass(fd, limit, capture=capture)
        if capture:
            mutate.set()
            assert mutated.wait(5)
        return result

    try:
        with (
            patch(
                "cuawright.environments.odysseys.judge._read_output_pass",
                side_effect=read_pass,
            ),
            pytest.raises(RuntimeError, match="changed while being read"),
        ):
            _read_limited_regular_fd(descriptor, _MAX_JUDGE_OUTPUT_BYTES)
    finally:
        mutate.set()
        thread.join(timeout=5)
        os.close(descriptor)
    assert not thread.is_alive()


def test_writable_judge_output_mapping_fails_closed_before_publication(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    evaluation = adapter.evaluate()
    canonical = judge_dir / "eval_results_full_traj_per_rubric.json"
    canonical.write_text('{"old": true}\n', encoding="utf-8")
    active_mapping: mmap.mmap | None = None

    def mutable_run(jobs, *, width):
        nonlocal active_mapping
        assert width == 1
        job = jobs[0]
        output = b'{"score": 1}\n'
        os.write(job.pass_fds[-1], output)
        active_mapping = mmap.mmap(
            job.pass_fds[-1],
            len(output),
            access=mmap.ACCESS_WRITE,
        )
        active_mapping[:] = b'{"score": 2}\n'
        return (ProcessOutcome(job.job_id, 0, 123, pgid=123),)

    try:
        with (
            patch(
                "cuawright.run.odysseys_judge.run_process_jobs",
                side_effect=mutable_run,
            ),
            pytest.raises(RuntimeError, match="could not be made immutable"),
        ):
            judge_main(
                [
                    "--manifest",
                    str(judge_dir / "judge-input.json"),
                    "--manifest-sha256",
                    evaluation["judge_manifest_sha256"],
                ]
            )
    finally:
        if active_mapping is not None:
            active_mapping.close()
    assert canonical.read_text(encoding="utf-8") == '{"old": true}\n'


def test_same_size_mutation_after_trusted_read_cannot_publish(
    tmp_path: Path,
) -> None:
    adapter, _, store, judge_dir = plugin(tmp_path)
    adapter.files[ODYSSEYS_FINAL_RESPONSE_PATH] = b"Final visible answer.\n"
    broker = TerminalCommandBroker(
        adapter,
        budget=SharedStepBudget(10),
        enabled_specialist_roles=(),
        run_store=store,
        command_observer=adapter.observe_command,
    )
    broker.run_command("open visible result")
    broker.run_command(SUBMIT)
    evaluation = adapter.evaluate()
    canonical = judge_dir / "eval_results_full_traj_per_rubric.json"
    canonical.write_text('{"old": true}\n', encoding="utf-8")
    output_fd = -1

    def successful_run(jobs, *, width):
        nonlocal output_fd
        assert width == 1
        job = jobs[0]
        output_fd = job.pass_fds[-1]
        os.write(output_fd, b'{"score": 1}\n')
        return (ProcessOutcome(job.job_id, 0, 123, pgid=123),)

    def mutate_after_read(_data: bytes) -> None:
        os.pwrite(output_fd, b'{"score": 2}\n', 0)

    with (
        patch(
            "cuawright.run.odysseys_judge.run_process_jobs",
            side_effect=successful_run,
        ),
        patch(
            "cuawright.environments.odysseys.judge._validate_judge_output",
            side_effect=mutate_after_read,
        ),
        pytest.raises(PermissionError),
    ):
        judge_main(
            [
                "--manifest",
                str(judge_dir / "judge-input.json"),
                "--manifest-sha256",
                evaluation["judge_manifest_sha256"],
            ]
        )
    assert canonical.read_text(encoding="utf-8") == '{"old": true}\n'


def test_openai_max_dollars_is_rejected_not_ignored(tmp_path: Path) -> None:
    arguments = parse_args(
        [
            "--task-id",
            TASK_ID,
            "--source",
            "source",
            "--source-commit",
            "a" * 40,
            "--task-source-json",
            "dataset",
            "--examples-dir",
            "examples",
            "--judge-script",
            "judge",
            "--dataset-sha256",
            "b" * 64,
            "--judge-sha256",
            "c" * 64,
            "--runtime-source",
            "runtime",
            "--path-to-vm",
            "vm",
            "--api-key-file",
            "key",
            "--result-root",
            "results",
        ]
    )
    assert arguments.reasoning_effort == "xhigh"
    assert arguments.total_steps == 100
    with pytest.raises(ValueError, match="unsupported.*OpenAI"):
        OdysseysRunConfig(
            task_id=TASK_ID,
            api_key_file=str(tmp_path / "key"),
            result_root=str(tmp_path / "results"),
            max_dollars=1.0,
        )
    supported = OdysseysRunConfig(
        task_id=TASK_ID,
        api_key_file=str(tmp_path / "key"),
        result_root=str(tmp_path / "results"),
        model_provider="openrouter-anthropic",
        model="anthropic/claude-opus-5",
        max_dollars=1.0,
    )
    assert supported.max_dollars == 1.0


def test_odysseys_proxy_requests_fail_closed() -> None:
    _require_no_odysseys_proxy({"proxy": False})
    with pytest.raises(RuntimeError, match="disables proxies"):
        _require_no_odysseys_proxy({"proxy": True})


def test_runner_composes_desktop_runtime_taskrunner_and_observer(
    tmp_path: Path,
) -> None:
    dataset, examples, judge, commit = create_source(tmp_path / "source")
    vm = tmp_path / "vm.qcow2"
    vm.write_bytes(b"vm")
    key = tmp_path / "key"
    key.write_text("secret\n", encoding="utf-8")
    provider = SimpleNamespace(
        api_key="secret",
        client=Mock(),
        cost_tracker=None,
        claude_client=None,
        async_runner=None,
        close=Mock(),
    )
    config = OdysseysRunConfig(
        task_id=TASK_ID,
        api_key_file=str(key),
        result_root=str(tmp_path / "results"),
        source=str(tmp_path / "source"),
        source_commit=commit,
        dataset_path=str(dataset),
        examples_dir=str(examples),
        judge_script=str(judge),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        judge_sha256=hashlib.sha256(judge.read_bytes()).hexdigest(),
        runtime_source=str(tmp_path / "runtime"),
        vm_path=str(vm),
    )
    fake_env = SimpleNamespace(
        controller=object(),
        setup_controller=object(),
        user_simulator=None,
        reset=Mock(),
        close=Mock(),
    )
    desktop_type = Mock(return_value=fake_env)
    plugin = Mock()
    plugin.observe_command = Mock()
    expected = {"benchmark": "odysseys"}

    with patch(
        "cuawright.run.task.create_provider_runtime",
        return_value=provider,
    ):
        runner = OdysseysRunner(config)
    runtime_source = SimpleNamespace(
        provenance=Mock(
            return_value={
                "root": str(tmp_path / "runtime"),
                "commit": config.runtime_commit,
                "clean": True,
            }
        )
    )
    runtime = DesktopRuntimeBindings(
        source=runtime_source,
        desktop_env_type=desktop_type,
        default_disk_size_gb=64,
    )
    with (
        patch(
            "cuawright.run.odysseys._activate_desktop_runtime",
            return_value=nullcontext(runtime),
        ),
        patch("cuawright.run.odysseys._provision_guest_tools") as provision,
        patch("cuawright.run.odysseys._ensure_guest_disk_headroom") as headroom,
        patch.object(runner, "_create_plugin", return_value=plugin),
        patch.object(
            runner,
            "run_terminal_benchmark",
            return_value=expected,
        ) as terminal,
    ):
        assert runner.run() == expected

    desktop_type.assert_called_once()
    assert desktop_type.call_args.kwargs["enable_proxy"] is False
    fake_env.reset.assert_called_once()
    provision.assert_called_once_with(fake_env.controller, fake_env.setup_controller)
    headroom.assert_called_once_with(fake_env.controller)
    assert terminal.call_args.kwargs["prompt_bundle"].name == "odysseys-default"
    assert terminal.call_args.kwargs["command_observer"] is plugin.observe_command
    assert terminal.call_args.kwargs["benchmark"] == "odysseys"
    manifest = json.loads((runner.run_dir / "manifest.json").read_text())
    assert manifest["trajectory_capture"]["captured_role_domain"] == [
        "actor",
        "binder",
        "critic",
        "verifier",
    ]
    assert manifest["trajectory_capture"]["active_roles"] == [
        "actor",
        "binder",
        "verifier",
    ]
    runner.close()
    provider.close.assert_called_once()

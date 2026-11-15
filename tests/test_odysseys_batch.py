from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cuawright.config.store import REPOSITORY_ROOT
from cuawright.environments.odysseys.source import inspect_source
from cuawright.operations.scheduler import (
    ProcessJob,
    ProcessMonitorState,
    ProcessOutcome,
    _signal_process_member,
    _terminate_process_groups,
    run_process_jobs,
)
from cuawright.run.odysseys_batch import (
    OdysseysBatchConfig,
    _monitor_task_health,
    prepare_batch,
    run_batch,
    selected_task_ids,
)
from test_odysseys_harness import create_source


def test_batch_selects_dataset_order_and_explicit_tasks(tmp_path: Path) -> None:
    dataset, _, _, _ = create_source(tmp_path / "source")
    assert selected_task_ids(dataset, [], 10) == ("a" * 40,)
    assert selected_task_ids(dataset, ["a" * 40], 10) == ("a" * 40,)


def test_batch_manifest_pins_source_and_fixed_profile(tmp_path: Path) -> None:
    dataset, examples, judge, commit = create_source(tmp_path / "source")
    source = inspect_source(
        root=tmp_path / "source",
        expected_commit=commit,
        dataset_path=dataset,
        examples_dir=examples,
        judge_script=judge,
    )
    root = tmp_path / "results"
    config = OdysseysBatchConfig(
        result_root=root,
        task_ids=("a" * 40,),
        width=1,
        source=source,
        runtime_source=tmp_path / "runtime",
        vm_path=tmp_path / "vm.qcow2",
        api_key_file=tmp_path / "key",
    )

    prepare_batch(config, "cuawright-commit")

    manifest = json.loads((root / "batch-manifest.json").read_text())
    assert manifest["benchmark"] == "odysseys"
    assert manifest["task_ids"] == ["a" * 40]
    assert manifest["source"]["commit"] == commit
    assert len(manifest["source"]["dataset_sha256"]) == 64
    assert len(manifest["source"]["judge_sha256"]) == 64
    assert manifest["model"] == "gpt-5.5"
    assert manifest["reasoning_effort"] == "xhigh"
    assert manifest["judge_execution"] == "external-not-started"
    assert manifest["health_contract"] == "current-registry-checker"


def test_batch_uses_current_automatic_health_checker(tmp_path: Path) -> None:
    dataset, examples, judge, commit = create_source(tmp_path / "source")
    source = inspect_source(
        root=tmp_path / "source",
        expected_commit=commit,
        dataset_path=dataset,
        examples_dir=examples,
        judge_script=judge,
    )
    task_id = "a" * 40
    root = tmp_path / "results"
    run_dir = root / "tasks" / task_id / f"task-{task_id}-run"
    run_dir.mkdir(parents=True)
    (root / "logs").mkdir(parents=True)
    (root / "health").mkdir(parents=True)
    config = OdysseysBatchConfig(
        result_root=root,
        task_ids=(task_id,),
        width=1,
        source=source,
        runtime_source=tmp_path / "runtime",
        vm_path=tmp_path / "vm.qcow2",
        api_key_file=tmp_path / "key",
        health_delay_seconds=1,
    )
    completed = threading.Event()
    completed.set()
    process = ProcessMonitorState(pid=123, pgid=123, completed=completed)
    payload = {
        "status": "healthy",
        "objective_stop_required": False,
        "checks": {"health_adapter": {"ok": True}},
    }

    def health_check(command, **kwargs):
        assert kwargs["cwd"] == REPOSITORY_ROOT
        assert command[command.index("--benchmark") + 1] == "odysseys"
        assert command[command.index("--expected-agent-topology") + 1] == (
            "orchestrated"
        )
        (run_dir / "health-check-5m.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="healthy\n", stderr="")

    with patch(
        "cuawright.run.odysseys_batch.subprocess.run",
        side_effect=health_check,
    ):
        _monitor_task_health(config, task_id, process)

    assert json.loads((root / "health" / f"{task_id}.json").read_text()) == payload
    assert (root / "health" / f"{task_id}.log").read_text() == "healthy\n"


def test_health_monitor_cannot_reap_sigint_spawned_descendant(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "leader-ready"
    child_pid_path = tmp_path / "handler-child.pid"
    monitor_active = threading.Event()
    monitor_done = threading.Event()
    script = """
import signal
import subprocess
import sys
import time

ready, child_pid = sys.argv[1:3]

def handle_sigint(_signum, _frame):
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, signal, sys, time;"
                "signal.signal(signal.SIGINT, signal.SIG_IGN);"
                "open(sys.argv[1], 'w').write(str(os.getpid()));"
                "time.sleep(60)"
            ),
            child_pid,
        ]
    )
    raise SystemExit(0)

signal.signal(signal.SIGINT, handle_sigint)
open(ready, "w").write("ready")
while True:
    time.sleep(1)
"""

    def monitor(state: ProcessMonitorState) -> None:
        assert not hasattr(state, "poll")
        assert not hasattr(state, "wait")
        monitor_active.set()
        assert state.completed.wait(5)
        monitor_done.set()

    job = ProcessJob(
        job_id="health-active",
        command=(
            sys.executable,
            "-c",
            script,
            str(ready),
            str(child_pid_path),
        ),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "health-active.log",
        pid_path=tmp_path / "pids" / "health-active.pid",
        exit_path=tmp_path / "exit" / "health-active.txt",
        monitor=monitor,
    )

    def interrupt_when_monitored(_futures):
        deadline = time.monotonic() + 5
        while (
            not ready.is_file() or not monitor_active.is_set()
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file()
        assert monitor_active.is_set()
        raise KeyboardInterrupt
        yield

    with (
        patch(
            "cuawright.operations.scheduler.as_completed",
            side_effect=interrupt_when_monitored,
        ),
        patch(
            "cuawright.operations.scheduler._terminate_process_groups",
            side_effect=lambda groups: _terminate_process_groups(
                groups,
                interrupt_timeout=0.2,
                terminate_timeout=2,
                kill_timeout=2,
            ),
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)

    assert monitor_done.is_set()
    assert outcomes[0].status == "interrupted"
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_health_polling_precedes_late_descendant_cleanup(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "late-child.pid"
    monitor_observed_child = threading.Event()
    script = """
import subprocess
import sys

subprocess.Popen(
    [
        sys.executable,
        "-c",
        (
            "import os, signal, sys, time;"
            "signal.signal(signal.SIGINT, signal.SIG_IGN);"
            "open(sys.argv[1], 'w').write(str(os.getpid()));"
            "time.sleep(60)"
        ),
        sys.argv[1],
    ]
)
"""

    def monitor(state: ProcessMonitorState) -> None:
        assert state.completed.wait(5)
        deadline = time.monotonic() + 5
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        os.kill(child_pid, 0)
        monitor_observed_child.set()

    job = ProcessJob(
        job_id="health-after-exit",
        command=(
            sys.executable,
            "-c",
            script,
            str(child_pid_path),
        ),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "health-after-exit.log",
        pid_path=tmp_path / "pids" / "health-after-exit.pid",
        exit_path=tmp_path / "exit" / "health-after-exit.txt",
        monitor=monitor,
    )

    with (
        patch(
            "cuawright.operations.scheduler._terminate_process_groups",
            side_effect=lambda groups: _terminate_process_groups(
                groups,
                interrupt_timeout=0.2,
                terminate_timeout=2,
                kill_timeout=2,
            ),
        ),
        patch(
            "cuawright.operations.scheduler.os.killpg",
            side_effect=AssertionError("bare PGID signaling is forbidden"),
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)

    assert monitor_observed_child.is_set()
    assert outcomes[0].status == "descendants_terminated"
    assert outcomes[0].return_code != 0
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_public_process_scheduler_records_pid_log_and_exit(tmp_path: Path) -> None:
    job = ProcessJob(
        job_id="one",
        command=(sys.executable, "-c", "print('scheduled')"),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "one.log",
        pid_path=tmp_path / "pids" / "one.pid",
        exit_path=tmp_path / "exit" / "one.txt",
    )

    outcomes = run_process_jobs((job,), width=1)

    assert outcomes[0].return_code == 0
    assert (tmp_path / "logs" / "one.log").read_text().strip() == "scheduled"
    assert int((tmp_path / "pids" / "one.pid").read_text()) > 0
    assert (tmp_path / "exit" / "one.txt").read_text().strip() == "0"


@pytest.mark.parametrize(
    "observed_identity",
    (
        (42, 1001),
        (43, 1000),
    ),
)
def test_pidfd_signal_rejects_reused_or_regrouped_pid(
    observed_identity: tuple[int, int],
) -> None:
    read_fd, write_fd = os.pipe()
    try:
        with (
            patch("cuawright.operations.scheduler.os.pidfd_open", return_value=read_fd),
            patch(
                "cuawright.operations.scheduler._process_identity",
                return_value=observed_identity,
            ),
            patch(
                "cuawright.operations.scheduler.signal.pidfd_send_signal",
            ) as send_signal,
        ):
            assert not _signal_process_member(
                123,
                expected_start_time=1000,
                expected_pgid=42,
                signum=signal.SIGTERM,
            )
        send_signal.assert_not_called()
        read_fd = -1
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        os.close(write_fd)


def test_scheduler_fails_before_launch_without_pidfd_signaling(
    tmp_path: Path,
) -> None:
    job = ProcessJob(
        job_id="unsupported",
        command=(sys.executable, "-c", "pass"),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "unsupported.log",
        pid_path=tmp_path / "pids" / "unsupported.pid",
        exit_path=tmp_path / "exit" / "unsupported.txt",
    )
    with (
        patch("cuawright.operations.scheduler.os.pidfd_open", None),
        patch("cuawright.operations.scheduler.subprocess.Popen") as popen,
        pytest.raises(RuntimeError, match="pidfd signaling"),
    ):
        run_process_jobs((job,), width=1)
    popen.assert_not_called()


def assert_scheduler_reaped(pid: int) -> None:
    with pytest.raises(ChildProcessError):
        os.waitid(
            os.P_PID,
            pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )


def test_pidfd_open_failure_keeps_workload_gated_and_records_launcher(
    tmp_path: Path,
) -> None:
    launched: subprocess.Popen[str] | None = None
    real_popen = subprocess.Popen
    workload_started = tmp_path / "workload-started"
    descendant_pid_path = tmp_path / "descendant.pid"
    script = """
import pathlib
import subprocess
import sys
import time

subprocess.Popen(
    [
        sys.executable,
        "-c",
        (
            "import os, pathlib, sys, time;"
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
            "time.sleep(60)"
        ),
        sys.argv[2],
    ]
)
pathlib.Path(sys.argv[1]).write_text("started")
time.sleep(60)
"""

    def tracked_popen(command, *args, **kwargs):
        nonlocal launched
        launched = real_popen(command, *args, **kwargs)
        return launched

    def fail_pidfd_open(_pid: int) -> int:
        time.sleep(0.3)
        raise OSError(errno.EMFILE, "too many open files")

    job = ProcessJob(
        job_id="pidfd-open-failure",
        command=(
            sys.executable,
            "-c",
            script,
            str(workload_started),
            str(descendant_pid_path),
        ),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "pidfd-open-failure.log",
        pid_path=tmp_path / "pids" / "pidfd-open-failure.pid",
        exit_path=tmp_path / "exit" / "pidfd-open-failure.txt",
    )

    with (
        patch(
            "cuawright.operations.scheduler.subprocess.Popen",
            side_effect=tracked_popen,
        ),
        patch(
            "cuawright.operations.scheduler.os.pidfd_open",
            side_effect=fail_pidfd_open,
        ),
        patch(
            "cuawright.operations.scheduler.os.killpg",
            side_effect=AssertionError("bare PGID signaling is forbidden"),
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)

    assert launched is not None
    assert_scheduler_reaped(launched.pid)
    assert launched.returncode == -signal.SIGKILL
    assert not workload_started.exists()
    assert not descendant_pid_path.exists()
    assert outcomes == (
        ProcessOutcome(
            job_id="pidfd-open-failure",
            return_code=-signal.SIGKILL,
            pid=launched.pid,
            status="worker_error",
            error="OSError: [Errno 24] too many open files",
        ),
    )
    assert job.pid_path.read_text(encoding="utf-8") == f"{launched.pid}\n"
    assert job.exit_path.read_text(encoding="utf-8") == (f"{-signal.SIGKILL}\n")


def test_gate_release_failure_reaps_registered_launcher_without_delay(
    tmp_path: Path,
) -> None:
    launched: subprocess.Popen[str] | None = None
    real_popen = subprocess.Popen
    workload_started = tmp_path / "workload-started"

    def tracked_popen(command, *args, **kwargs):
        nonlocal launched
        launched = real_popen(command, *args, **kwargs)
        return launched

    job = ProcessJob(
        job_id="gate-release-failure",
        command=(
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(workload_started),
        ),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "gate-release-failure.log",
        pid_path=tmp_path / "pids" / "gate-release-failure.pid",
        exit_path=tmp_path / "exit" / "gate-release-failure.txt",
    )

    started = time.monotonic()
    with (
        patch(
            "cuawright.operations.scheduler.subprocess.Popen",
            side_effect=tracked_popen,
        ),
        patch(
            "cuawright.operations.scheduler._release_process_gate",
            side_effect=BrokenPipeError(errno.EPIPE, "broken pipe"),
        ),
        patch(
            "cuawright.operations.scheduler.os.killpg",
            side_effect=AssertionError("bare PGID signaling is forbidden"),
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert launched is not None
    assert_scheduler_reaped(launched.pid)
    assert launched.returncode == -signal.SIGKILL
    assert not workload_started.exists()
    assert outcomes == (
        ProcessOutcome(
            job_id="gate-release-failure",
            return_code=-signal.SIGKILL,
            pid=launched.pid,
            status="worker_error",
            error="BrokenPipeError: [Errno 32] broken pipe",
        ),
    )
    assert job.pid_path.read_text(encoding="utf-8") == f"{launched.pid}\n"
    assert job.exit_path.read_text(encoding="utf-8") == (f"{-signal.SIGKILL}\n")


def test_registration_failure_replaces_stale_pid_artifact(
    tmp_path: Path,
) -> None:
    launched: subprocess.Popen[str] | None = None
    real_popen = subprocess.Popen
    pid_path = tmp_path / "pids" / "stale.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("1\n", encoding="utf-8")

    def tracked_popen(command, *args, **kwargs):
        nonlocal launched
        launched = real_popen(command, *args, **kwargs)
        return launched

    job = ProcessJob(
        job_id="stale-pid",
        command=(sys.executable, "-c", "raise AssertionError('gate failed')"),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "stale-pid.log",
        pid_path=pid_path,
        exit_path=tmp_path / "exit" / "stale-pid.txt",
    )

    with (
        patch(
            "cuawright.operations.scheduler.subprocess.Popen",
            side_effect=tracked_popen,
        ),
        patch(
            "cuawright.operations.scheduler.os.pidfd_open",
            side_effect=OSError(errno.EMFILE, "too many open files"),
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)

    assert launched is not None
    assert_scheduler_reaped(launched.pid)
    assert launched.returncode == -signal.SIGKILL
    assert outcomes[0].pid == launched.pid
    assert outcomes[0].return_code == -signal.SIGKILL
    assert outcomes[0].status == "worker_error"
    assert pid_path.read_text(encoding="utf-8") == f"{launched.pid}\n"
    assert job.exit_path.read_text(encoding="utf-8") == (f"{-signal.SIGKILL}\n")


def test_process_scheduler_interrupts_active_group_and_records_every_job(
    tmp_path: Path,
) -> None:
    long_started = threading.Event()
    real_popen = subprocess.Popen

    def controlled_popen(command, *args, **kwargs):
        if command[-1] == "worker-failure":
            assert long_started.wait(5)
            raise RuntimeError("deterministic worker failure")
        process = real_popen(command, *args, **kwargs)
        if command[-1] == "long-child":
            long_started.set()
        return process

    jobs = tuple(
        ProcessJob(
            job_id=name,
            command=(
                sys.executable,
                "-c",
                "import time; time.sleep(60)",
                marker,
            ),
            cwd=tmp_path,
            environment=dict(os.environ),
            log_path=tmp_path / "logs" / f"{name}.log",
            pid_path=tmp_path / "pids" / f"{name}.pid",
            exit_path=tmp_path / "exit" / f"{name}.txt",
        )
        for name, marker in (
            ("active", "long-child"),
            ("failure", "worker-failure"),
            ("queued", "queued-child"),
        )
    )

    with patch(
        "cuawright.operations.scheduler.subprocess.Popen",
        side_effect=controlled_popen,
    ):
        outcomes = run_process_jobs(jobs, width=2)

    by_id = {outcome.job_id: outcome for outcome in outcomes}
    assert set(by_id) == {"active", "failure", "queued"}
    assert by_id["active"].status == "interrupted"
    assert by_id["failure"].status == "worker_error"
    assert by_id["queued"].status in {"cancelled", "interrupted"}
    for job in jobs:
        assert job.exit_path.is_file()
        assert int(job.exit_path.read_text(encoding="utf-8")) != 0
    with pytest.raises(ProcessLookupError):
        os.kill(by_id["active"].pid, 0)


def test_process_scheduler_handles_keyboard_interrupt_before_waiting(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    real_popen = subprocess.Popen

    def tracked_popen(command, *args, **kwargs):
        process = real_popen(command, *args, **kwargs)
        started.set()
        time.sleep(0.1)
        return process

    def interrupted(_futures):
        assert started.wait(5)
        raise KeyboardInterrupt
        yield

    job = ProcessJob(
        job_id="interrupt",
        command=(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "interrupt.log",
        pid_path=tmp_path / "pids" / "interrupt.pid",
        exit_path=tmp_path / "exit" / "interrupt.txt",
    )

    with (
        patch(
            "cuawright.operations.scheduler.subprocess.Popen",
            side_effect=tracked_popen,
        ),
        patch(
            "cuawright.operations.scheduler.as_completed",
            side_effect=interrupted,
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)

    assert outcomes[0].status == "interrupted"
    assert outcomes[0].return_code != 0
    assert job.exit_path.is_file()
    with pytest.raises(ProcessLookupError):
        os.kill(outcomes[0].pid, 0)


def test_process_scheduler_kills_group_after_leader_exits(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "descendant.pid"
    leader_script = """
import subprocess
import sys

subprocess.Popen(
    [
        sys.executable,
        "-c",
        (
            "import os, signal, sys, time;"
            "signal.signal(signal.SIGINT, signal.SIG_IGN);"
            "open(sys.argv[1], 'w').write(str(os.getpid()));"
            "time.sleep(60)"
        ),
        sys.argv[1],
    ]
)
"""

    leader = ProcessJob(
        job_id="leader",
        command=(
            sys.executable,
            "-c",
            leader_script,
            str(child_pid_path),
            "leader-with-child",
        ),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "leader.log",
        pid_path=tmp_path / "pids" / "leader.pid",
        exit_path=tmp_path / "exit" / "leader.txt",
    )

    def interrupt_after_leader_exit(_futures):
        deadline = time.monotonic() + 5
        while (
            not child_pid_path.is_file() or not leader.exit_path.is_file()
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()
        assert leader.exit_path.is_file()
        raise KeyboardInterrupt
        yield

    with (
        patch(
            "cuawright.operations.scheduler.as_completed",
            side_effect=interrupt_after_leader_exit,
        ),
        patch(
            "cuawright.operations.scheduler._terminate_process_groups",
            side_effect=lambda groups: _terminate_process_groups(
                groups,
                interrupt_timeout=0.1,
                terminate_timeout=2,
                kill_timeout=2,
            ),
        ),
        patch(
            "cuawright.operations.scheduler.os.killpg",
            side_effect=AssertionError("bare PGID signaling is forbidden"),
        ),
    ):
        outcomes = run_process_jobs((leader,), width=1)

    by_id = {outcome.job_id: outcome for outcome in outcomes}
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert by_id["leader"].status == "interrupted"
    assert by_id["leader"].return_code == 130
    assert by_id["leader"].pgid == by_id["leader"].pid
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_completed_process_group_is_retired_before_later_cancellation(
    tmp_path: Path,
) -> None:
    job = ProcessJob(
        job_id="completed",
        command=(sys.executable, "-c", "pass"),
        cwd=tmp_path,
        environment=dict(os.environ),
        log_path=tmp_path / "logs" / "completed.log",
        pid_path=tmp_path / "pids" / "completed.pid",
        exit_path=tmp_path / "exit" / "completed.txt",
    )

    def interrupt_after_completion(_futures):
        deadline = time.monotonic() + 5
        while not job.exit_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.exit_path.is_file()
        raise KeyboardInterrupt
        yield

    cancellation_groups: list[list[object]] = []

    def capture_groups(groups):
        cancellation_groups.append(list(groups))
        return set(), set()

    with (
        patch(
            "cuawright.operations.scheduler.as_completed",
            side_effect=interrupt_after_completion,
        ),
        patch(
            "cuawright.operations.scheduler._terminate_process_groups",
            side_effect=capture_groups,
        ),
    ):
        outcomes = run_process_jobs((job,), width=1)

    assert outcomes[0].status == "completed"
    assert outcomes[0].return_code == 0
    assert cancellation_groups == [[], []]


def test_batch_summary_counts_ready_incomplete_and_failed(tmp_path: Path) -> None:
    dataset, examples, judge, commit = create_source(tmp_path / "source")
    source = inspect_source(
        root=tmp_path / "source",
        expected_commit=commit,
        dataset_path=dataset,
        examples_dir=examples,
        judge_script=judge,
    )
    task_ids = ("a" * 40, "b" * 40, "c" * 40)
    root = tmp_path / "results"
    config = OdysseysBatchConfig(
        result_root=root,
        task_ids=task_ids,
        width=3,
        source=source,
        runtime_source=tmp_path / "runtime",
        vm_path=tmp_path / "vm.qcow2",
        api_key_file=tmp_path / "key",
    )
    ready_run = root / "tasks" / task_ids[0] / f"task-{task_ids[0]}-ready"
    ready_result = ready_run / "evaluation" / "result.json"
    ready_result.parent.mkdir(parents=True)
    (ready_run / "manifest.json").write_text("{}\n", encoding="utf-8")
    ready_result.write_text(
        json.dumps({"evaluation": {"status": "ready_for_external_judge"}}),
        encoding="utf-8",
    )
    incomplete_run = root / "tasks" / task_ids[1] / f"task-{task_ids[1]}-incomplete"
    incomplete_result = incomplete_run / "evaluation" / "result.json"
    incomplete_result.parent.mkdir(parents=True)
    (incomplete_run / "manifest.json").write_text("{}\n", encoding="utf-8")
    incomplete_result.write_text(
        json.dumps({"evaluation": {"status": "incomplete_judge_evidence"}}),
        encoding="utf-8",
    )
    outcomes = (
        ProcessOutcome(task_ids[0], 0, 1),
        ProcessOutcome(task_ids[1], 0, 2),
        ProcessOutcome(task_ids[2], 1, 3),
    )

    with (
        patch(
            "cuawright.run.odysseys_batch.validate_source_tree",
            return_value="commit",
        ),
        patch(
            "cuawright.run.odysseys_batch.run_process_jobs",
            return_value=outcomes,
        ),
    ):
        assert run_batch(config) == 1

    summary = json.loads((root / "batch-summary.json").read_text())
    assert summary["ready_for_external_judge"] == 1
    assert summary["incomplete"] == 1
    assert summary["failed"] == 1


def test_batch_rejects_interrupted_zero_exit_as_ready(tmp_path: Path) -> None:
    dataset, examples, judge, commit = create_source(tmp_path / "source")
    source = inspect_source(
        root=tmp_path / "source",
        expected_commit=commit,
        dataset_path=dataset,
        examples_dir=examples,
        judge_script=judge,
    )
    task_id = "a" * 40
    root = tmp_path / "results"
    config = OdysseysBatchConfig(
        result_root=root,
        task_ids=(task_id,),
        width=1,
        source=source,
        runtime_source=tmp_path / "runtime",
        vm_path=tmp_path / "vm.qcow2",
        api_key_file=tmp_path / "key",
    )
    run_dir = root / "tasks" / task_id / f"task-{task_id}-ready"
    result_path = run_dir / "evaluation" / "result.json"
    result_path.parent.mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    result_path.write_text(
        json.dumps({"evaluation": {"status": "ready_for_external_judge"}}),
        encoding="utf-8",
    )
    interrupted = (
        ProcessOutcome(
            task_id,
            0,
            123,
            status="interrupted",
            pgid=123,
        ),
    )

    with (
        patch(
            "cuawright.run.odysseys_batch.validate_source_tree",
            return_value="commit",
        ),
        patch(
            "cuawright.run.odysseys_batch.run_process_jobs",
            return_value=interrupted,
        ),
    ):
        assert run_batch(config) == 1

    summary = json.loads((root / "batch-summary.json").read_text())
    assert summary["ready_for_external_judge"] == 0
    assert summary["interrupted"] == 1
    assert summary["tasks"][0]["state"] == "interrupted"
    assert summary["tasks"][0]["scheduler_status"] == "interrupted"

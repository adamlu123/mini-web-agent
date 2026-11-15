from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from cuawright.config.store import REPOSITORY_ROOT
from cuawright.environments.control_clients import (
    CLIENT_FILENAMES,
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
)

_GATED_EXEC = """\
import os
import sys

gate_fd = int(sys.argv[1])
try:
    token = os.read(gate_fd, 1)
finally:
    os.close(gate_fd)
if token != b"1":
    raise SystemExit(126)
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


def validate_source_tree() -> str:
    control_client_manifest()
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def control_client_manifest() -> dict[str, Any]:
    hashes = dict(CLIENT_SHA256)
    if tuple(hashes) != CLIENT_FILENAMES:
        raise RuntimeError("control-client source set does not match the required set")
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes.values()
    ):
        raise RuntimeError("control-client source manifest contains an invalid sha256")
    return {
        "protocol": "python-clients-v1",
        "install_root": CLIENT_INSTALL_ROOT,
        "clients": hashes,
        "direct_report_return": True,
        "prior_report_injection": True,
        "typed_image_attachments": True,
        "legacy_agent_protocol": False,
    }


def latest_run_dir(task_root: Path, task_id: str) -> Path | None:
    values = sorted(
        task_root.glob(f"task-{task_id}-*"),
        key=lambda path: path.stat().st_mtime,
    )
    return values[-1] if values else None


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def objective_health_failure(payload: dict[str, Any] | None) -> bool:
    return payload is not None and payload.get("objective_stop_required") is True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def signal_process_group(
    process: subprocess.Popen[str],
    signum: signal.Signals,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def terminate_process(
    process: subprocess.Popen[str],
    *,
    interrupt_timeout: float = 30,
    terminate_timeout: float = 15,
    kill_timeout: float = 15,
) -> None:
    signal_process_group(process, signal.SIGINT)
    try:
        process.wait(timeout=interrupt_timeout)
    except subprocess.TimeoutExpired:
        signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=terminate_timeout)
        except subprocess.TimeoutExpired:
            signal_process_group(process, signal.SIGKILL)
            process.wait(timeout=kill_timeout)


@dataclass(frozen=True, slots=True)
class ProcessMonitorState:
    pid: int
    pgid: int
    completed: threading.Event


@dataclass(frozen=True, slots=True)
class ProcessJob:
    job_id: str
    command: tuple[str, ...]
    cwd: Path
    environment: dict[str, str]
    log_path: Path
    pid_path: Path
    exit_path: Path
    monitor: Callable[[ProcessMonitorState], None] | None = None
    pass_fds: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    job_id: str
    return_code: int
    pid: int
    status: str = "completed"
    error: str | None = None
    pgid: int | None = None


@dataclass(slots=True)
class _ProcessGroup:
    job_id: str
    process: subprocess.Popen[str]
    pgid: int
    leader_start_time: int
    known_members: dict[int, int]
    pidfd: int
    completed: threading.Event
    discovery_lock: threading.Lock


class _ProcessRegistrationError(RuntimeError):
    def __init__(
        self,
        cause: BaseException,
        *,
        pid: int,
        return_code: int,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.pid = pid
        self.return_code = return_code


def run_process_jobs(
    jobs: tuple[ProcessJob, ...],
    *,
    width: int,
) -> tuple[ProcessOutcome, ...]:
    """Run independent process jobs at bounded width with group cleanup."""

    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("process scheduler width must be a positive integer")
    if not jobs:
        return ()
    _require_pidfd_signaling()
    process_groups: dict[str, _ProcessGroup] = {}
    lock = threading.Lock()
    cancel_event = threading.Event()
    jobs_by_id = {job.job_id: job for job in jobs}
    if len(jobs_by_id) != len(jobs):
        raise ValueError("process scheduler job ids must be unique")

    def record(
        job: ProcessJob,
        return_code: int,
        pid: int,
        status: str,
        error: str | None = None,
        pgid: int | None = None,
    ) -> ProcessOutcome:
        job.exit_path.parent.mkdir(parents=True, exist_ok=True)
        job.exit_path.write_text(f"{return_code}\n", encoding="utf-8")
        return ProcessOutcome(job.job_id, return_code, pid, status, error, pgid)

    def run_job(job: ProcessJob) -> ProcessOutcome:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        job.pid_path.parent.mkdir(parents=True, exist_ok=True)
        job.exit_path.parent.mkdir(parents=True, exist_ok=True)
        with job.log_path.open("w", encoding="utf-8") as log:
            with lock:
                if cancel_event.is_set():
                    job.pid_path.write_text("0\n", encoding="utf-8")
                    return record(job, 130, 0, "cancelled")
                gate_read_fd, gate_write_fd = os.pipe2(os.O_CLOEXEC)
                try:
                    try:
                        process = subprocess.Popen(
                            _gated_command(job.command, gate_read_fd),
                            cwd=job.cwd,
                            env=job.environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                            pass_fds=tuple(
                                dict.fromkeys((*job.pass_fds, gate_read_fd))
                            ),
                        )
                    except OSError as exc:
                        log.write(f"{type(exc).__name__}: {exc}\n")
                        job.pid_path.write_text("0\n", encoding="utf-8")
                        return record(
                            job,
                            -1,
                            0,
                            "launch_error",
                            f"{type(exc).__name__}: {exc}",
                        )
                    os.close(gate_read_fd)
                    gate_read_fd = -1
                    group = _new_process_group(job.job_id, process)
                    process_groups[job.job_id] = group
                    try:
                        _release_process_gate(gate_write_fd)
                    except BaseException as exc:
                        process_groups.pop(job.job_id, None)
                        return_code = _abort_registered_process(group)
                        raise _ProcessRegistrationError(
                            exc,
                            pid=process.pid,
                            return_code=return_code,
                        ) from exc
                finally:
                    for descriptor in (gate_read_fd, gate_write_fd):
                        if descriptor >= 0:
                            os.close(descriptor)
            job.pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
            monitor_errors: list[BaseException] = []
            monitor_thread: threading.Thread | None = None
            if job.monitor is not None:
                monitor_state = ProcessMonitorState(
                    pid=process.pid,
                    pgid=group.pgid,
                    completed=group.completed,
                )

                def monitor() -> None:
                    try:
                        job.monitor(monitor_state)
                    except BaseException as exc:
                        monitor_errors.append(exc)

                monitor_thread = threading.Thread(
                    target=monitor,
                    name=f"monitor-{job.job_id}",
                    daemon=True,
                )
                monitor_thread.start()
            try:
                return_code = _wait_for_group_leader(group)
            finally:
                with lock:
                    if not _owned_process_group_alive(group):
                        process_groups.pop(job.job_id, None)
                    _close_pidfd(group)
                    group.completed.set()
                if monitor_thread is not None:
                    monitor_thread.join()
            if monitor_errors:
                raise RuntimeError(
                    f"process monitor failed for {job.job_id}: {monitor_errors[0]}"
                ) from monitor_errors[0]
        return record(
            job,
            return_code,
            process.pid,
            "interrupted" if cancel_event.is_set() else "completed",
            pgid=group.pgid,
        )

    outcomes: dict[str, ProcessOutcome] = {}
    failure: BaseException | None = None
    executor = ThreadPoolExecutor(
        max_workers=min(width, len(jobs)),
        thread_name_prefix="cuawright-process",
    )
    futures: dict[Future[ProcessOutcome], str] = {}
    try:
        futures = {executor.submit(run_job, job): job.job_id for job in jobs}
        for future in as_completed(futures):
            try:
                outcome = future.result()
            except BaseException as exc:
                failure = exc
                break
            outcomes[outcome.job_id] = outcome
    except BaseException as exc:
        failure = exc

    if failure is None:
        executor.shutdown(wait=True)
        with lock:
            _retire_drained_process_groups(process_groups)
            lingering_groups = list(process_groups.values())
        terminated_ids, survivor_ids = _terminate_process_groups(lingering_groups)
        for job_id in terminated_ids:
            outcome = outcomes[job_id]
            job = jobs_by_id[job_id]
            outcomes[job_id] = record(
                job,
                outcome.return_code if outcome.return_code else 137,
                outcome.pid,
                (
                    "termination_failed"
                    if job_id in survivor_ids
                    else "descendants_terminated"
                ),
                (
                    "process group remained alive after SIGKILL"
                    if job_id in survivor_ids
                    else "process group retained descendants after leader exit"
                ),
                outcome.pgid,
            )
        return tuple(sorted(outcomes.values(), key=lambda item: item.job_id))

    with lock:
        cancel_event.set()
        _retire_drained_process_groups(process_groups)
        groups = list(process_groups.values())
    for future in futures:
        future.cancel()
    interrupted_ids, survivor_ids = _terminate_process_groups(groups)
    executor.shutdown(wait=True, cancel_futures=True)
    with lock:
        _retire_drained_process_groups(process_groups)
        settled_groups = list(process_groups.values())
    settled_interrupted, survivor_ids = _terminate_process_groups(settled_groups)
    interrupted_ids.update(settled_interrupted)

    for future, job_id in futures.items():
        if job_id in outcomes:
            continue
        job = jobs_by_id[job_id]
        try:
            outcome = future.result()
        except CancelledError:
            job.pid_path.parent.mkdir(parents=True, exist_ok=True)
            job.pid_path.write_text("0\n", encoding="utf-8")
            outcome = record(job, 130, 0, "cancelled")
        except BaseException as exc:
            group = process_groups.get(job_id)
            registration_error = (
                exc if isinstance(exc, _ProcessRegistrationError) else None
            )
            if registration_error is not None:
                pid = registration_error.pid
                job.pid_path.parent.mkdir(parents=True, exist_ok=True)
                job.pid_path.write_text(f"{pid}\n", encoding="utf-8")
            else:
                pid = 0 if group is None else group.process.pid
                try:
                    pid = int(job.pid_path.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    job.pid_path.parent.mkdir(parents=True, exist_ok=True)
                    job.pid_path.write_text(f"{pid}\n", encoding="utf-8")
            cause = registration_error.cause if registration_error is not None else exc
            outcome = record(
                job,
                (
                    registration_error.return_code
                    if registration_error is not None
                    else -1
                ),
                pid,
                "worker_error",
                f"{type(cause).__name__}: {cause}",
                None if group is None else group.pgid,
            )
        if job_id in interrupted_ids and outcome.status == "completed":
            outcome = record(
                job,
                outcome.return_code if outcome.return_code else 130,
                outcome.pid,
                "interrupted",
                outcome.error,
                outcome.pgid,
            )
        if job_id in survivor_ids:
            outcome = record(
                job,
                outcome.return_code if outcome.return_code else 137,
                outcome.pid,
                "termination_failed",
                "process group remained alive after SIGKILL",
                outcome.pgid,
            )
        outcomes[job_id] = outcome
    for job_id in interrupted_ids:
        outcome = outcomes.get(job_id)
        if outcome is None or outcome.status != "completed":
            continue
        job = jobs_by_id[job_id]
        outcomes[job_id] = record(
            job,
            outcome.return_code if outcome.return_code else 130,
            outcome.pid,
            "interrupted",
            outcome.error,
            outcome.pgid,
        )
    for job_id in survivor_ids:
        outcome = outcomes.get(job_id)
        if outcome is None or outcome.status == "termination_failed":
            continue
        job = jobs_by_id[job_id]
        outcomes[job_id] = record(
            job,
            outcome.return_code if outcome.return_code else 137,
            outcome.pid,
            "termination_failed",
            "process group remained alive after SIGKILL",
            outcome.pgid,
        )
    for job in jobs:
        if job.job_id in outcomes:
            continue
        job.pid_path.parent.mkdir(parents=True, exist_ok=True)
        job.pid_path.write_text("0\n", encoding="utf-8")
        outcomes[job.job_id] = record(
            job,
            130,
            0,
            "not_submitted",
            f"{type(failure).__name__}: {failure}",
        )
    return tuple(sorted(outcomes.values(), key=lambda item: item.job_id))


def _terminate_process_groups(
    groups: list[_ProcessGroup],
    *,
    interrupt_timeout: float = 30,
    terminate_timeout: float = 15,
    kill_timeout: float = 15,
) -> tuple[set[str], set[str]]:
    alive = [group for group in groups if _owned_process_group_alive(group)]
    interrupted = {group.job_id for group in alive}
    alive = _signal_and_wait_for_groups(alive, signal.SIGINT, interrupt_timeout)
    alive = _signal_and_wait_for_groups(alive, signal.SIGTERM, terminate_timeout)
    alive = _signal_and_wait_for_groups(alive, signal.SIGKILL, kill_timeout)
    return interrupted, {group.job_id for group in alive}


def _signal_and_wait_for_groups(
    groups: list[_ProcessGroup],
    signum: signal.Signals,
    timeout: float,
) -> list[_ProcessGroup]:
    deadline = time.monotonic() + timeout
    remaining = list(groups)
    signaled: set[tuple[int, int]] = set()
    while remaining and time.monotonic() < deadline:
        observed: list[_ProcessGroup] = []
        for group in remaining:
            members = _owned_process_group_members(group)
            if not members:
                continue
            observed.append(group)
            for pid, start_time in members.items():
                identity = (pid, start_time)
                if identity in signaled:
                    continue
                if _signal_process_member(
                    pid,
                    start_time,
                    group.pgid,
                    signum,
                ):
                    signaled.add(identity)
        remaining = observed
        if remaining:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return [group for group in remaining if _owned_process_group_alive(group)]


def _signal_process_member(
    pid: int,
    expected_start_time: int,
    expected_pgid: int,
    signum: signal.Signals,
) -> bool:
    pidfd = -1
    try:
        pidfd = os.pidfd_open(pid)
        if _process_identity(pid) != (expected_pgid, expected_start_time):
            return False
        signal.pidfd_send_signal(pidfd, signum)
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False
    finally:
        if pidfd >= 0:
            os.close(pidfd)


def _gated_command(command: tuple[str, ...], gate_fd: int) -> list[str]:
    return [
        sys.executable,
        "-I",
        "-S",
        "-c",
        _GATED_EXEC,
        str(gate_fd),
        *command,
    ]


def _release_process_gate(gate_fd: int) -> None:
    os.write(gate_fd, b"1")


def _abort_registered_process(group: _ProcessGroup) -> int:
    try:
        try:
            signal.pidfd_send_signal(group.pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                group.process.kill()
            except ProcessLookupError:
                pass
        with group.discovery_lock:
            return group.process.wait()
    finally:
        _close_pidfd(group)
        group.completed.set()


def _new_process_group(
    job_id: str,
    process: subprocess.Popen[str],
) -> _ProcessGroup:
    pidfd_open = getattr(os, "pidfd_open", None)
    if not callable(pidfd_open):
        raise RuntimeError("process scheduler requires os.pidfd_open")
    pidfd = -1
    try:
        pidfd = pidfd_open(process.pid)
        pgid, start_time = _process_identity(process.pid)
        if pgid != process.pid:
            raise RuntimeError("scheduled process did not create its own process group")
        return _ProcessGroup(
            job_id=job_id,
            process=process,
            pgid=pgid,
            leader_start_time=start_time,
            known_members={process.pid: start_time},
            pidfd=pidfd,
            completed=threading.Event(),
            discovery_lock=threading.Lock(),
        )
    except BaseException as exc:
        return_code: int
        try:
            if pidfd >= 0:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            return_code = process.wait()
        finally:
            if pidfd >= 0:
                os.close(pidfd)
        raise _ProcessRegistrationError(
            exc,
            pid=process.pid,
            return_code=return_code,
        ) from exc


def _wait_for_group_leader(
    group: _ProcessGroup,
) -> int:
    poller = select.poll()
    poller.register(group.pidfd, select.POLLIN)
    poller.poll()
    with group.discovery_lock:
        _owned_process_group_members_unlocked(group)
        return group.process.wait()


def _owned_process_group_alive(group: _ProcessGroup) -> bool:
    return bool(_owned_process_group_members(group))


def _owned_process_group_members(group: _ProcessGroup) -> dict[int, int]:
    with group.discovery_lock:
        return _owned_process_group_members_unlocked(group)


def _owned_process_group_members_unlocked(
    group: _ProcessGroup,
) -> dict[int, int]:
    members = _process_group_members(group.pgid)
    if not any(
        members.get(pid) == start_time
        for pid, start_time in group.known_members.items()
    ):
        return {}
    group.known_members.update(members)
    return members


def _process_group_members(pgid: int) -> dict[int, int]:
    members: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            observed_pgid, start_time = _process_identity(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, ValueError):
            continue
        if observed_pgid == pgid:
            members[int(entry.name)] = start_time
    return members


def _process_identity(pid: int) -> tuple[int, int]:
    value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    end = value.rfind(")")
    if end < 0:
        raise ValueError("invalid /proc stat record")
    fields = value[end + 2 :].split()
    return int(fields[2]), int(fields[19])


def _retire_drained_process_groups(
    groups: dict[str, _ProcessGroup],
) -> None:
    for job_id, group in tuple(groups.items()):
        if _owned_process_group_alive(group):
            continue
        groups.pop(job_id, None)
        _close_pidfd(group)


def _close_pidfd(group: _ProcessGroup) -> None:
    if group.pidfd < 0:
        return
    try:
        os.close(group.pidfd)
    except OSError:
        pass
    group.pidfd = -1


def _require_pidfd_signaling() -> None:
    if not callable(getattr(os, "pidfd_open", None)) or not callable(
        getattr(signal, "pidfd_send_signal", None)
    ):
        raise RuntimeError(
            "process scheduler requires pidfd signaling and will not launch "
            "without identity-stable process control"
        )

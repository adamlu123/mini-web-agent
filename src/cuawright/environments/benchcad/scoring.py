from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from docker.errors import DockerException

from cuawright.environments.benchcad.docker_exec import run_exec
from cuawright.environments.benchcad.errors import BenchCADScoringInfrastructureError
from cuawright.environments.benchcad.image import validate_image
from cuawright.environments.benchcad.task import Vision2CodeTask

SCORER_FILE_LIMIT = 256 * 1024 * 1024

logger = logging.getLogger("cuawright.environments.benchcad.scoring")


def _snapshot_file(source: Path, destination: Path, limit: int) -> str:
    digest = hashlib.sha256()
    total = 0
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        while chunk := input_handle.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"{source} exceeds {limit} bytes")
            digest.update(chunk)
            output_handle.write(chunk)
    os.replace(temporary, destination)
    os.chmod(destination, 0o400)
    return digest.hexdigest()


def _copy_container_file(
    *,
    client: Any,
    container: Any,
    source_path: str,
    destination: Path,
) -> bool:
    def copy_via_exec() -> bool:
        try:
            execution = client.api.exec_create(
                container.id,
                ["/bin/cat", "--", source_path],
            )
            output_stream = client.api.exec_start(
                execution["Id"],
                stream=True,
                demux=True,
            )
            data = bytearray()
            for stdout, _ in output_stream:
                if stdout:
                    if len(data) + len(stdout) > SCORER_FILE_LIMIT:
                        raise RuntimeError(
                            f"Scorer artifact {source_path} exceeds "
                            f"{SCORER_FILE_LIMIT} bytes"
                        )
                    data.extend(stdout)
            inspection = client.api.exec_inspect(execution["Id"])
        except DockerException:
            return False
        if int(inspection["ExitCode"]) != 0:
            return False
        destination.write_bytes(data)
        return True

    try:
        stream, _ = container.get_archive(source_path)
        data = bytearray()
        for chunk in stream:
            if len(data) + len(chunk) > SCORER_FILE_LIMIT:
                raise RuntimeError(
                    f"Scorer artifact {source_path} exceeds "
                    f"{SCORER_FILE_LIMIT} bytes"
                )
            data.extend(chunk)
        with tarfile.open(fileobj=BytesIO(data), mode="r:") as tar:
            members = [member for member in tar.getmembers() if member.isfile()]
            if len(members) != 1:
                return copy_via_exec()
            source = tar.extractfile(members[0])
            if source is None:
                return copy_via_exec()
            destination.write_bytes(source.read())
        return True
    except (DockerException, OSError, RuntimeError, tarfile.TarError):
        return copy_via_exec()


def _snapshot_benchcad_source(
    repo: Path,
    commit: str,
    destination: Path,
) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=True,
        capture_output=True,
    )
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", commit],
        check=True,
        capture_output=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")


def score_prediction(
    *,
    task: Vision2CodeTask,
    workspace_dir: Path,
    output_dir: Path,
    image: str,
    benchcad_repo: Path,
    benchmark_commit: str,
) -> dict[str, Any]:
    import docker

    prediction = workspace_dir / "prediction.py"
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    if not prediction.is_file():
        ground_truth_snapshot = output_dir / "ground-truth.step"
        ground_truth_sha256 = _snapshot_file(
            task.ground_truth_step,
            ground_truth_snapshot,
            1024 * 1024 * 1024,
        )
        client = docker.from_env()
        try:
            image_id = validate_image(client, image, benchmark_commit)
        finally:
            client.close()
        result = {
            "score": 0.0,
            "iou": 0.0,
            "status": "no_code",
            "error": "Missing /workspace/prediction.py",
            "benchmark_commit": benchmark_commit,
            "container_image_id": image_id,
            "prediction_sha256": None,
            "ground_truth_sha256": ground_truth_sha256,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        return result
    submission_snapshot = output_dir / "submission-snapshot" / "prediction.py"
    prediction_sha256 = _snapshot_file(
        prediction,
        submission_snapshot,
        16 * 1024 * 1024,
    )
    ground_truth_snapshot = output_dir / "ground-truth.step"
    ground_truth_sha256 = _snapshot_file(
        task.ground_truth_step,
        ground_truth_snapshot,
        1024 * 1024 * 1024,
    )

    adapter_dir = Path(__file__).resolve().parent
    source_snapshot = output_dir.parent / "benchcad-source"
    execution_output = output_dir / "execution-artifacts"
    execution_output.mkdir(parents=True, exist_ok=True)
    os.chmod(execution_output, 0o700)
    _snapshot_benchcad_source(
        benchcad_repo,
        benchmark_commit,
        source_snapshot,
    )
    client = docker.from_env()
    image_id = validate_image(client, image, benchmark_commit)
    host_user = f"{os.getuid()}:{os.getgid()}"
    cleanup_errors: list[DockerException] = []
    try:
        execution_container = client.containers.run(
            image_id,
            command=["sleep", "infinity"],
            detach=True,
            network_mode="none",
            read_only=True,
            user=host_user,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            pids_limit=512,
            mem_limit="8g",
            nano_cpus=4_000_000_000,
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=2g,mode=1777",
                "/score": (
                    "rw,nosuid,size=2g,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
                "/home/agent": (
                    "rw,nosuid,size=256m,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
            },
            volumes={
                str(submission_snapshot.parent): {
                    "bind": "/submission",
                    "mode": "ro",
                },
                str(adapter_dir): {"bind": "/adapter", "mode": "ro"},
            },
            labels={
                "cuawright.component": "benchcad-scorer",
                "cuawright.record_id": task.record_id,
            },
        )
        execution_result = run_exec(
            client=client,
            container=execution_container,
            command=[
                "python",
                "/adapter/scorer_cli.py",
                "execute",
                "--prediction",
                "/submission/prediction.py",
                "--output-dir",
                "/score",
                "--benchcad-repo",
                "/opt/benchcad-tools",
                "--benchmark-commit",
                benchmark_commit,
            ],
            user=host_user,
            deadline=330,
            capture_limit=1_000_000,
        )
        execution_logs = execution_result["stdout"] + execution_result["stderr"]
        (output_dir / "execution-container.log").write_bytes(execution_logs)
        if execution_result["timed_out"]:
            return {
                "score": 0.0,
                "iou": 0.0,
                "status": "exec_fail",
                "error": "Submission execution exceeded the 330 second deadline",
                "benchmark_commit": benchmark_commit,
                "container_image_id": image_id,
                "prediction_sha256": prediction_sha256,
                "ground_truth_sha256": ground_truth_sha256,
            }
        if int(execution_result["returncode"]) != 0:
            raise BenchCADScoringInfrastructureError(
                "Trusted submission executor exited unsuccessfully: "
                + execution_logs.decode(
                    "utf-8",
                    errors="replace",
                )[-4000:]
            )
        copied = _copy_container_file(
            client=client,
            container=execution_container,
            source_path="/score/execution.json",
            destination=execution_output / "execution.json",
        )
        if not copied or not (execution_output / "execution.json").is_file():
            raise BenchCADScoringInfrastructureError(
                "Failed to transfer scorer artifact execution.json"
            )
        execution_result = json.loads(
            (execution_output / "execution.json").read_text(encoding="utf-8")
        )
        if execution_result["status"] != "ok":
            return {
                **execution_result,
                "score": 0.0,
                "iou": 0.0,
                "container_image_id": image_id,
                "prediction_sha256": prediction_sha256,
                "ground_truth_sha256": ground_truth_sha256,
            }
        copied = _copy_container_file(
            client=client,
            container=execution_container,
            source_path="/score/prediction.step",
            destination=execution_output / "prediction.step",
        )
        if not copied or not (execution_output / "prediction.step").is_file():
            raise BenchCADScoringInfrastructureError(
                "Failed to transfer scorer artifact prediction.step"
            )

        score_container = client.containers.run(
            image_id,
            command=[
                "python",
                "/adapter/scorer_cli.py",
                "score",
                "--generated-step",
                "/generated/prediction.step",
                "--ground-truth-step",
                "/ground-truth/target.step",
                "--output-dir",
                "/score",
                "--benchcad-repo",
                "/opt/benchcad",
                "--benchmark-commit",
                benchmark_commit,
            ],
            detach=True,
            network_mode="none",
            read_only=True,
            user=host_user,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            pids_limit=256,
            mem_limit="8g",
            nano_cpus=4_000_000_000,
            tmpfs={
                "/tmp": "rw,noexec,nosuid,size=2g,mode=1777",
                "/home/agent": (
                    "rw,nosuid,size=256m,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
            },
            volumes={
                str(execution_output): {"bind": "/generated", "mode": "ro"},
                str(ground_truth_snapshot): {
                    "bind": "/ground-truth/target.step",
                    "mode": "ro",
                },
                str(output_dir): {"bind": "/score", "mode": "rw"},
                str(source_snapshot): {"bind": "/opt/benchcad", "mode": "ro"},
                str(adapter_dir): {"bind": "/adapter", "mode": "ro"},
            },
            labels={
                "cuawright.component": "benchcad-iou-scorer",
                "cuawright.record_id": task.record_id,
            },
        )
        try:
            score_status = score_container.wait(timeout=720)
        except requests.exceptions.ReadTimeout as exc:
            raise BenchCADScoringInfrastructureError(
                "IoU scorer exceeded the 720 second host deadline"
            ) from exc
        score_logs = score_container.logs(stdout=True, stderr=True)
        (output_dir / "score-container.log").write_bytes(score_logs)
        if int(score_status.get("StatusCode", -1)) != 0:
            raise BenchCADScoringInfrastructureError(
                "IoU scorer exited unsuccessfully: "
                + score_logs.decode("utf-8", errors="replace")[-4000:]
            )
    finally:
        if "execution_container" in locals():
            try:
                execution_container.remove(force=True)
            except DockerException as exc:
                cleanup_errors.append(exc)
        if "score_container" in locals():
            try:
                score_container.remove(force=True)
            except DockerException as exc:
                cleanup_errors.append(exc)
        try:
            client.close()
        except DockerException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            if sys.exc_info()[0] is None:
                raise BenchCADScoringInfrastructureError(
                    "Failed to clean up scorer resources: "
                    + "; ".join(str(error) for error in cleanup_errors)
                )
            for error in cleanup_errors:
                logger.error("Scorer cleanup failed: %s", error)

    result_path = output_dir / "result.json"
    if not result_path.is_file():
        raise BenchCADScoringInfrastructureError("IoU scorer produced no result.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["container_image_id"] = image_id
    result["prediction_sha256"] = prediction_sha256
    result["ground_truth_sha256"] = ground_truth_sha256
    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def score_prediction_with_retry(**kwargs) -> dict[str, Any]:
    for attempt in range(2):
        try:
            return score_prediction(**kwargs)
        except BenchCADScoringInfrastructureError:
            if attempt == 1:
                raise
        except DockerException as exc:
            if attempt == 1:
                raise BenchCADScoringInfrastructureError(
                    f"Docker scoring infrastructure failed: {exc}"
                ) from exc
    raise RuntimeError("BenchCAD scoring retry loop exited unexpectedly")

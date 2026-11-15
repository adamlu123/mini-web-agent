from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def validate_record_id(record_id: str) -> str:
    if not RECORD_ID.fullmatch(record_id):
        raise ValueError(
            "BenchCAD record IDs must be a single safe path component: "
            f"{record_id!r}"
        )
    return record_id


@dataclass(frozen=True)
class Vision2CodeTask:
    record_id: str
    input_image: Path
    ground_truth_step: Path
    system_prompt: str
    user_prompt: str
    expected_input_sha256: str
    expected_ground_truth_sha256: str
    family: str | None = None
    difficulty: str | None = None

    @property
    def instruction(self) -> str:
        return (
            f"{self.system_prompt}\n\n"
            f"{self.user_prompt}\n\n"
            "The target composite is available at /input/input.png. "
            "Use `python /opt/cuawright-tools/image_show.py --path "
            "/input/input.png` through run_command to inspect it. Save the final "
            "executable CadQuery program to "
            "/workspace/prediction.py. The program must bind the final geometry "
            "to `result`."
        )


@dataclass(frozen=True)
class Vision2CodeManifest:
    benchmark_commit: str
    dataset_revision: str | None
    manifest_sha256: str
    tasks: tuple[Vision2CodeTask, ...]


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_manifest(
    path: str | Path,
    *,
    validate_assets: bool = True,
) -> Vision2CodeManifest:
    manifest_path = Path(path).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    payload: dict[str, Any] = json.loads(manifest_bytes)
    task_rows: list[Vision2CodeTask] = []
    for row in payload["tasks"]:
        input_image = _resolve_path(row["input_image"], manifest_path.parent)
        ground_truth_step = _resolve_path(
            row["ground_truth_step"],
            manifest_path.parent,
        )
        task_rows.append(
            Vision2CodeTask(
                record_id=validate_record_id(str(row["record_id"])),
                input_image=input_image,
                ground_truth_step=ground_truth_step,
                system_prompt=str(row["system_prompt"]),
                user_prompt=str(row["user_prompt"]),
                expected_input_sha256=_manifest_asset_hash(
                    row,
                    "input_sha256",
                    input_image,
                ),
                expected_ground_truth_sha256=_manifest_asset_hash(
                    row,
                    "ground_truth_sha256",
                    ground_truth_step,
                ),
                family=(str(row["family"]) if row.get("family") is not None else None),
                difficulty=(
                    str(row["difficulty"])
                    if row.get("difficulty") is not None
                    else None
                ),
            )
        )
    tasks = tuple(task_rows)
    if not tasks:
        raise ValueError("BenchCAD manifest contains no tasks")
    ids = [task.record_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("BenchCAD manifest contains duplicate record IDs")
    if validate_assets:
        for task in tasks:
            if not task.input_image.is_file():
                raise FileNotFoundError(task.input_image)
            if not task.ground_truth_step.is_file():
                raise FileNotFoundError(task.ground_truth_step)
    return Vision2CodeManifest(
        benchmark_commit=str(payload["benchmark_commit"]),
        dataset_revision=(
            str(payload["dataset_revision"])
            if payload.get("dataset_revision") is not None
            else None
        ),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        tasks=tasks,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_asset_hash(
    row: dict[str, Any],
    field: str,
    path: Path,
) -> str:
    value = row.get(field)
    if value is not None:
        return str(value)
    if not path.is_file():
        raise ValueError(
            f"Legacy BenchCAD manifest omits {field!r} and asset is unavailable: "
            f"{path}"
        )
    return sha256_file(path)


def task_provenance(
    manifest: Vision2CodeManifest,
    task: Vision2CodeTask,
) -> dict[str, str | None]:
    input_sha256 = sha256_file(task.input_image)
    ground_truth_sha256 = sha256_file(task.ground_truth_step)
    if input_sha256 != task.expected_input_sha256:
        raise ValueError("Task input does not match the manifest SHA-256")
    if ground_truth_sha256 != task.expected_ground_truth_sha256:
        raise ValueError("Task ground truth does not match the manifest SHA-256")
    return {
        "benchmark_commit": manifest.benchmark_commit,
        "dataset_revision": manifest.dataset_revision,
        "manifest_sha256": manifest.manifest_sha256,
        "input_sha256": input_sha256,
        "ground_truth_sha256": ground_truth_sha256,
    }


def snapshot_task(task: Vision2CodeTask, run_dir: Path) -> Vision2CodeTask:
    input_snapshot = run_dir / "sandbox-input" / "input.png"
    ground_truth_snapshot = run_dir / "ground-truth" / "target.step"
    input_snapshot.parent.mkdir(parents=True, exist_ok=True)
    ground_truth_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task.input_image, input_snapshot)
    shutil.copyfile(task.ground_truth_step, ground_truth_snapshot)
    os_mode = 0o400
    input_snapshot.chmod(os_mode)
    ground_truth_snapshot.chmod(os_mode)
    return replace(
        task,
        input_image=input_snapshot,
        ground_truth_step=ground_truth_snapshot,
    )


def load_task(
    path: str | Path,
    record_id: str,
    *,
    validate_assets: bool = True,
) -> tuple[
    Vision2CodeManifest,
    Vision2CodeTask,
]:
    manifest = load_manifest(path, validate_assets=validate_assets)
    for task in manifest.tasks:
        if task.record_id == record_id:
            return manifest, task
    raise KeyError(f"Record {record_id!r} is not present in {path}")

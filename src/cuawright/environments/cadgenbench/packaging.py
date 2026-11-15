from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from cuawright.environments.cadgenbench.task import (
    CadGenBenchManifest,
    CadGenBenchTask,
)

CANDIDATE_NAMES = ("output.step", "output.stp")


def _task_payload_from_run_dir(run_dir: str | Path) -> dict[str, Any]:
    task_path = Path(run_dir).expanduser().resolve() / "task.json"
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "cadgenbench":
        raise ValueError(f"Run dir is not a CadGenBench harness run: {run_dir}")
    return payload


def fixture_id_from_run_dir(run_dir: str | Path) -> str:
    return str(_task_payload_from_run_dir(run_dir)["fixture_id"])


def discover_candidate(run_dir: str | Path) -> Path | None:
    workspace = Path(run_dir).expanduser().resolve() / "workspace"
    for name in CANDIDATE_NAMES:
        candidate = workspace / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _manifest_provenance(task: CadGenBenchTask) -> dict[str, Any]:
    provenance = task.provenance()
    return {
        "benchmark_commit": provenance["benchmark_commit"],
        "dataset_revision": provenance["dataset_revision"],
        "fixture_id": provenance["fixture_id"],
        "task_type": provenance["task_type"],
        "declared_input_files": provenance["declared_input_files"],
        "public_files": provenance["public_files"],
    }


def _run_provenance(task_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_commit": task_payload.get("benchmark_commit"),
        "dataset_revision": task_payload.get("dataset_revision"),
        "fixture_id": task_payload.get("fixture_id"),
        "task_type": task_payload.get("task_type"),
        "declared_input_files": task_payload.get("declared_input_files"),
        "public_files": task_payload.get("public_files"),
    }


def validate_run_provenance(task: CadGenBenchTask, run_dir: str | Path) -> None:
    payload = _task_payload_from_run_dir(run_dir)
    expected = _manifest_provenance(task)
    actual = _run_provenance(payload)
    if actual != expected:
        raise ValueError(
            f"Run provenance for fixture {task.fixture_id} does not match the panel manifest"
        )


def package_submission(
    *,
    manifest: CadGenBenchManifest,
    run_dirs: list[str | Path],
    output_path: str | Path,
    meta: dict[str, Any],
) -> Path:
    mapping: dict[str, Path] = {}
    for run_dir in run_dirs:
        payload = _task_payload_from_run_dir(run_dir)
        fixture_id = str(payload["fixture_id"])
        if fixture_id in mapping:
            raise ValueError(f"Duplicate run dir supplied for fixture {fixture_id}")
        mapping[fixture_id] = Path(run_dir).expanduser().resolve()

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("meta.json", json.dumps(meta, indent=2) + "\n")
        for task in manifest.tasks:
            archive.writestr(f"{task.fixture_id}/", "")
            run_dir = mapping.get(task.fixture_id)
            if run_dir is None:
                continue
            validate_run_provenance(task, run_dir)
            candidate = discover_candidate(run_dir)
            if candidate is None:
                continue
            archive.write(candidate, arcname=f"{task.fixture_id}/output.step")
    return destination

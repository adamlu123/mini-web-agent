from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from cuawright.environments.cadgenbench.public_prompts import (
    editing_prompt,
    generation_prompt,
)

DEFAULT_BENCHMARK_COMMIT = "33304cf771fc5639144b1df9611e347251052cf8"
DEFAULT_DATASET_REVISION = "f76f965585817c621d6ea0d150d745adf670e66e"
FIXTURE_ID = re.compile(r"[0-9]{3}\Z")
TASK_TYPES = {"generation", "editing"}
INPUT_TYPES = {"text+image", "text+step"}
GENERATION_INPUT = "text+image"
EDITING_INPUT = "text+step"
GENERATION_VIEWS = ("input.png", "input2.png", "input3.png", "input4.png")
EDITING_RENDER_NAMES = ("front.png", "top.png", "right.png", "iso.png")


def validate_fixture_id(fixture_id: str) -> str:
    if not FIXTURE_ID.fullmatch(fixture_id):
        raise ValueError(
            "CadGenBench fixture IDs must be a safe numeric path component: "
            f"{fixture_id!r}"
        )
    return fixture_id


def resolve_benchmark_commit(
    repo: str | Path,
    expected_commit: str = DEFAULT_BENCHMARK_COMMIT,
) -> str:
    repo_path = Path(repo).expanduser().resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != expected_commit:
        raise ValueError(
            f"CadGenBench repo {repo_path} is at {commit}, expected "
            f"{expected_commit}"
        )
    if subprocess.check_output(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        text=True,
    ).strip():
        raise ValueError(f"CadGenBench repo is dirty: {repo_path}")
    return commit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PublicFile:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CadGenBenchTask:
    fixture_id: str
    fixture_dir: Path
    task_type: Literal["generation", "editing"]
    input_type: Literal["text+image", "text+step"]
    description: str
    declared_input_files: tuple[str, ...]
    benchmark_commit: str
    dataset_revision: str
    public_files: tuple[PublicFile, ...]
    image_inputs: tuple[Path, ...] = ()
    input_step: Path | None = None
    edit_description_path: Path | None = None
    input_renders: tuple[Path, ...] = ()
    mesh_sidecar: Path | None = None

    @property
    def instruction(self) -> str:
        if self.task_type == "generation":
            prompt = generation_prompt("/workspace/output.step")
            if len(self.image_inputs) > 1:
                names = ", ".join(path.name for path in self.image_inputs)
                prompt += (
                    "\n\n## Public drawing set\n"
                    f"This fixture provides multiple authoritative drawings: {names}. "
                    "Inspect all of them and combine their views and callouts when "
                    "constructing the part."
                )
            return prompt
        if self.edit_description_path is None:
            raise ValueError(
                f"Editing fixture {self.fixture_id} is missing edit_description.txt"
            )
        return editing_prompt(
            self.edit_description_path.read_text(encoding="utf-8"),
            output_path="/workspace/output.step",
        )

    @property
    def staged_workspace_inputs(self) -> tuple[tuple[str, str], ...]:
        if self.task_type == "generation":
            return tuple((path.name, path.name) for path in self.image_inputs)
        return (
            ("input.step", "input.step"),
            ("edit_description.txt", "edit_description.txt"),
            *tuple(
                (f"renders/{name}", f"renders/{name}") for name in EDITING_RENDER_NAMES
            ),
        )

    @property
    def initial_actor_image_paths(self) -> tuple[str, ...]:
        if self.task_type == "generation":
            return tuple(f"/input/{path.name}" for path in self.image_inputs)
        return tuple(f"/input/renders/{name}" for name in EDITING_RENDER_NAMES)

    def provenance(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "task_type": self.task_type,
            "input_type": self.input_type,
            "benchmark_commit": self.benchmark_commit,
            "dataset_revision": self.dataset_revision,
            "declared_input_files": list(self.declared_input_files),
            "public_files": [
                {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.public_files
            ],
        }

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "task_type": self.task_type,
            "input_type": self.input_type,
            "description": self.description,
            "declared_input_files": list(self.declared_input_files),
            "public_files": [
                {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "size_bytes": item.size_bytes,
                }
                for item in self.public_files
            ],
        }


@dataclass(frozen=True)
class CadGenBenchManifest:
    dataset_dir: Path
    benchmark_commit: str
    dataset_revision: str
    manifest_sha256: str
    tasks: tuple[CadGenBenchTask, ...]


def discover_tasks(
    *,
    dataset_dir: str | Path,
    benchmark_commit: str,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    fixture_ids: list[str] | None = None,
    limit: int | None = None,
) -> CadGenBenchManifest:
    dataset_root = Path(dataset_dir).expanduser().resolve()
    all_ids = _discover_fixture_ids(dataset_root)
    requested = (
        [validate_fixture_id(item) for item in fixture_ids] if fixture_ids else all_ids
    )
    missing = [item for item in requested if item not in set(all_ids)]
    if missing:
        raise KeyError(f"Unknown CadGenBench fixture IDs: {', '.join(sorted(missing))}")
    ordered = sorted(requested, key=int)
    if limit is not None:
        ordered = ordered[:limit]
    tasks = tuple(
        _build_task(
            dataset_root=dataset_root,
            fixture_id=fixture_id,
            benchmark_commit=benchmark_commit,
            dataset_revision=dataset_revision,
        )
        for fixture_id in ordered
    )
    manifest_payload = {
        "benchmark": "cadgenbench",
        "dataset_dir": str(dataset_root),
        "benchmark_commit": benchmark_commit,
        "dataset_revision": dataset_revision,
        "tasks": [task.to_manifest_entry() for task in tasks],
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return CadGenBenchManifest(
        dataset_dir=dataset_root,
        benchmark_commit=benchmark_commit,
        dataset_revision=dataset_revision,
        manifest_sha256=manifest_sha256,
        tasks=tasks,
    )


def load_manifest(
    path: str | Path,
    *,
    validate_assets: bool = True,
) -> CadGenBenchManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("benchmark") != "cadgenbench":
        raise ValueError("Manifest benchmark must be 'cadgenbench'")
    dataset_dir = Path(str(payload["dataset_dir"])).expanduser().resolve()
    benchmark_commit = str(payload["benchmark_commit"])
    dataset_revision = str(payload["dataset_revision"])
    task_ids: list[str] = []
    tasks: list[CadGenBenchTask] = []
    for entry in payload["tasks"]:
        fixture_id = validate_fixture_id(str(entry["fixture_id"]))
        task_ids.append(fixture_id)
        expected = {
            "fixture_id": fixture_id,
            "task_type": str(entry["task_type"]),
            "input_type": str(entry["input_type"]),
            "description": str(entry["description"]),
            "declared_input_files": [
                str(item) for item in entry["declared_input_files"]
            ],
            "public_files": [
                {
                    "relative_path": str(item["relative_path"]),
                    "sha256": str(item["sha256"]),
                    "size_bytes": int(item["size_bytes"]),
                }
                for item in entry["public_files"]
            ],
        }
        if validate_assets:
            task = _build_task(
                dataset_root=dataset_dir,
                fixture_id=fixture_id,
                benchmark_commit=benchmark_commit,
                dataset_revision=dataset_revision,
            )
            if task.to_manifest_entry() != expected:
                raise ValueError(
                    f"Manifest entry for fixture {fixture_id} does not match the live dataset snapshot"
                )
        else:
            task = _task_from_manifest_entry(
                dataset_dir=dataset_dir,
                benchmark_commit=benchmark_commit,
                dataset_revision=dataset_revision,
                entry=expected,
            )
        tasks.append(task)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Manifest contains duplicate fixture IDs")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = CadGenBenchManifest(
        dataset_dir=dataset_dir,
        benchmark_commit=benchmark_commit,
        dataset_revision=dataset_revision,
        manifest_sha256=manifest_sha256,
        tasks=tuple(tasks),
    )
    if validate_assets:
        for task in manifest.tasks:
            _validate_live_task(task)
    return manifest


def _discover_fixture_ids(dataset_root: Path) -> list[str]:
    return sorted(
        [
            validate_fixture_id(path.name)
            for path in dataset_root.iterdir()
            if path.is_dir() and FIXTURE_ID.fullmatch(path.name)
        ],
        key=int,
    )


def _task_from_manifest_entry(
    *,
    dataset_dir: Path,
    benchmark_commit: str,
    dataset_revision: str,
    entry: dict[str, Any],
) -> CadGenBenchTask:
    fixture_id = str(entry["fixture_id"])
    fixture_dir = dataset_dir / fixture_id
    public_files = tuple(
        PublicFile(
            relative_path=str(item["relative_path"]),
            sha256=str(item["sha256"]),
            size_bytes=int(item["size_bytes"]),
        )
        for item in entry["public_files"]
    )
    task_type = str(entry["task_type"])
    if task_type == "generation":
        image_inputs = tuple(
            fixture_dir / name for name in entry["declared_input_files"]
        )
        return CadGenBenchTask(
            fixture_id=fixture_id,
            fixture_dir=fixture_dir,
            task_type="generation",
            input_type=GENERATION_INPUT,
            description=str(entry["description"]),
            declared_input_files=tuple(
                str(item) for item in entry["declared_input_files"]
            ),
            benchmark_commit=benchmark_commit,
            dataset_revision=dataset_revision,
            public_files=public_files,
            image_inputs=image_inputs,
        )
    render_paths = tuple(
        fixture_dir / "renders" / name for name in EDITING_RENDER_NAMES
    )
    return CadGenBenchTask(
        fixture_id=fixture_id,
        fixture_dir=fixture_dir,
        task_type="editing",
        input_type=EDITING_INPUT,
        description=str(entry["description"]),
        declared_input_files=tuple(str(item) for item in entry["declared_input_files"]),
        benchmark_commit=benchmark_commit,
        dataset_revision=dataset_revision,
        public_files=public_files,
        input_step=fixture_dir / "input.step",
        edit_description_path=fixture_dir / "edit_description.txt",
        input_renders=render_paths,
        mesh_sidecar=fixture_dir / "input.mesh.npz",
    )


def _load_description(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid description.yaml at {path}")
    return data


def _effective_task_type(
    description: dict[str, Any],
) -> Literal["generation", "editing"]:
    raw = description.get("task_type")
    if raw is None:
        return "generation"
    task_type = str(raw)
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task_type {task_type!r}")
    return task_type  # type: ignore[return-value]


def _metadata_revision(dataset_root: Path, relative_path: str) -> str:
    metadata = (
        dataset_root
        / ".cache"
        / "huggingface"
        / "download"
        / f"{relative_path}.metadata"
    )
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing Hugging Face metadata for {relative_path}")
    first_line = metadata.read_text(encoding="utf-8").splitlines()
    if not first_line or not first_line[0].strip():
        raise ValueError(f"Malformed metadata for {relative_path}")
    return first_line[0].strip()


def _public_file(
    dataset_root: Path, fixture_dir: Path, relative_path: str
) -> PublicFile:
    path = fixture_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return PublicFile(
        relative_path=relative_path,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _build_task(
    *,
    dataset_root: Path,
    fixture_id: str,
    benchmark_commit: str,
    dataset_revision: str,
) -> CadGenBenchTask:
    fixture_dir = dataset_root / fixture_id
    description_path = fixture_dir / "description.yaml"
    description_data = _load_description(description_path)
    description = str(description_data.get("description") or "").strip()
    if not description:
        raise ValueError(f"Fixture {fixture_id} has an empty description")
    task_type = _effective_task_type(description_data)
    input_type = str(description_data.get("input_type") or "")
    if input_type not in INPUT_TYPES:
        raise ValueError(
            f"Fixture {fixture_id} has unsupported input_type {input_type!r}"
        )
    raw_input_files = description_data.get("input_files")
    if not isinstance(raw_input_files, list) or not raw_input_files:
        raise ValueError(f"Fixture {fixture_id} declares no input_files")
    declared_input_files = tuple(str(item) for item in raw_input_files)
    if any(
        Path(item).is_absolute() or ".." in Path(item).parts
        for item in declared_input_files
    ):
        raise ValueError(f"Fixture {fixture_id} declares an unsafe input file path")

    public_files = [_public_file(dataset_root, fixture_dir, "description.yaml")]
    for relative_path in ["description.yaml", *declared_input_files]:
        if (
            _metadata_revision(dataset_root, f"{fixture_id}/{relative_path}")
            != dataset_revision
        ):
            raise ValueError(
                f"Fixture {fixture_id} file {relative_path} does not match dataset revision {dataset_revision}"
            )
    if (
        _metadata_revision(dataset_root, "sanity_check_submission.py")
        != dataset_revision
    ):
        raise ValueError(
            f"sanity_check_submission.py does not match dataset revision {dataset_revision}"
        )

    if task_type == "generation":
        if input_type != GENERATION_INPUT:
            raise ValueError(
                f"Fixture {fixture_id} generation task must use {GENERATION_INPUT}"
            )
        image_inputs = tuple(fixture_dir / name for name in declared_input_files)
        if not image_inputs:
            raise ValueError(f"Fixture {fixture_id} has no generation input image")
        for path in image_inputs:
            if path.suffix.lower() != ".png" or not path.name.startswith("input"):
                raise ValueError(
                    f"Fixture {fixture_id} declares an invalid generation input {path.name}"
                )
            public_files.append(_public_file(dataset_root, fixture_dir, path.name))
        discovered_images = sorted(
            path.name for path in fixture_dir.glob("input*.png") if path.is_file()
        )
        if sorted(declared_input_files) != discovered_images:
            raise ValueError(
                f"Fixture {fixture_id} declared input images do not match the snapshot contents"
            )
        task = CadGenBenchTask(
            fixture_id=fixture_id,
            fixture_dir=fixture_dir,
            task_type="generation",
            input_type=GENERATION_INPUT,
            description=description,
            declared_input_files=declared_input_files,
            benchmark_commit=benchmark_commit,
            dataset_revision=dataset_revision,
            public_files=tuple(
                sorted(public_files, key=lambda item: item.relative_path)
            ),
            image_inputs=image_inputs,
        )
    else:
        if input_type != EDITING_INPUT:
            raise ValueError(
                f"Fixture {fixture_id} editing task must use {EDITING_INPUT}"
            )
        if declared_input_files != ("input.step",):
            raise ValueError(
                f"Fixture {fixture_id} editing task must declare only input.step"
            )
        input_step = fixture_dir / "input.step"
        edit_description = fixture_dir / "edit_description.txt"
        mesh_sidecar = fixture_dir / "input.mesh.npz"
        render_paths = tuple(
            fixture_dir / "renders" / name for name in EDITING_RENDER_NAMES
        )
        for relative_path in (
            "input.step",
            "edit_description.txt",
            "input.mesh.npz",
            *[f"renders/{name}" for name in EDITING_RENDER_NAMES],
        ):
            if (
                _metadata_revision(dataset_root, f"{fixture_id}/{relative_path}")
                != dataset_revision
            ):
                raise ValueError(
                    f"Fixture {fixture_id} file {relative_path} does not match dataset revision {dataset_revision}"
                )
        public_files.extend(
            [
                _public_file(dataset_root, fixture_dir, "input.step"),
                _public_file(dataset_root, fixture_dir, "edit_description.txt"),
                _public_file(dataset_root, fixture_dir, "input.mesh.npz"),
            ]
        )
        for path in render_paths:
            public_files.append(
                _public_file(
                    dataset_root, fixture_dir, str(path.relative_to(fixture_dir))
                )
            )
        task = CadGenBenchTask(
            fixture_id=fixture_id,
            fixture_dir=fixture_dir,
            task_type="editing",
            input_type=EDITING_INPUT,
            description=description,
            declared_input_files=declared_input_files,
            benchmark_commit=benchmark_commit,
            dataset_revision=dataset_revision,
            public_files=tuple(
                sorted(public_files, key=lambda item: item.relative_path)
            ),
            input_step=input_step,
            edit_description_path=edit_description,
            input_renders=render_paths,
            mesh_sidecar=mesh_sidecar,
        )
    _validate_live_task(task)
    return task


def _validate_live_task(task: CadGenBenchTask) -> None:
    fixture_dir = task.fixture_dir
    for item in task.public_files:
        path = fixture_dir / item.relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != item.sha256:
            raise ValueError(
                f"Fixture {task.fixture_id} file {item.relative_path} SHA-256 mismatch"
            )
        if path.stat().st_size != item.size_bytes:
            raise ValueError(
                f"Fixture {task.fixture_id} file {item.relative_path} size mismatch"
            )

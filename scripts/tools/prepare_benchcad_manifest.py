#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cuawright.environments.benchcad.image import validate_image
from cuawright.environments.benchcad.task import sha256_file, validate_record_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render BenchCAD Vision2Code inputs and write a run manifest"
    )
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchcad-repo", required=True)
    parser.add_argument("--image", default="cuawright-benchcad:52087ef")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-revision", default=None)
    return parser.parse_args()


def _load_prompts(benchcad_repo: Path) -> tuple[str, str]:
    path = benchcad_repo / "Vision2Code" / "pipeline" / "prompt.py"
    spec = importlib.util.spec_from_file_location("benchcad_prompt", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.SYSTEM_PROMPT), str(module.USER_PROMPT)


def _render_input(
    *,
    image: str,
    benchmark_commit: str,
    step: Path,
    output: Path,
) -> None:
    import docker

    output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.tmp.png")
    client = docker.from_env()
    try:
        image_id = validate_image(
            client,
            image,
            benchmark_commit,
        )
        container = client.containers.run(
            image_id,
            command=[
                "python",
                "-c",
                (
                    "from pathlib import Path;"
                    "from benchcad_core.scoring.views import composite_for_step;"
                    "composite_for_step(Path('/source/target.step'),"
                    f"Path('/output/{temporary.name}'))"
                ),
            ],
            detach=True,
            network_mode="none",
            read_only=True,
            user=f"{os.getuid()}:{os.getgid()}",
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit="8g",
            nano_cpus=4_000_000_000,
            tmpfs={"/tmp": "rw,noexec,nosuid,size=2g,mode=1777"},
            environment={
                "HOME": "/tmp",
                "PYTHONPATH": "/opt/benchcad-tools",
            },
            volumes={
                str(step): {"bind": "/source/target.step", "mode": "ro"},
                str(output.parent): {"bind": "/output", "mode": "rw"},
            },
        )
        status = container.wait(timeout=720)
        if int(status.get("StatusCode", -1)) != 0:
            logs = container.logs(stdout=True, stderr=True).decode(
                "utf-8",
                errors="replace",
            )
            raise RuntimeError(logs[-4000:])
    finally:
        if "container" in locals():
            container.remove(force=True)
        client.close()
    if not temporary.is_file():
        raise RuntimeError(f"Renderer produced no image at {temporary}")
    data = temporary.read_bytes()
    if len(data) < 8 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Renderer produced an invalid PNG at {temporary}")
    os.replace(temporary, output)


def main() -> None:
    args = parse_args()
    records_path = Path(args.records_jsonl).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    benchcad_repo = Path(args.benchcad_repo).expanduser().resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(benchcad_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(benchcad_repo), "status", "--porcelain"],
        text=True,
    ).strip():
        raise RuntimeError(f"BenchCAD checkout is dirty: {benchcad_repo}")
    records: list[dict[str, Any]] = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        records = records[: args.limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(benchcad_repo), "archive", "--format=tar", commit],
        check=True,
        capture_output=True,
    ).stdout
    with tempfile.TemporaryDirectory(
        dir=output_path.parent,
        prefix=".benchcad-source-",
    ) as source_directory:
        source_snapshot = Path(source_directory)
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as tar:
            tar.extractall(source_snapshot, filter="data")
        system_prompt, user_prompt = _load_prompts(source_snapshot)
    input_root = output_path.parent / (
        f"{output_path.stem}-inputs-{commit[:12]}-{uuid.uuid4().hex[:12]}"
    )
    tasks = []
    preparation_error: Exception | None = None
    try:
        for record in records:
            record_id = validate_record_id(str(record["record_id"]))
            step = (data_dir / record["step_path"]).resolve()
            image = input_root / record_id / "input.png"
            _render_input(
                image=args.image,
                benchmark_commit=commit,
                step=step,
                output=image,
            )
            tasks.append(
                {
                    "record_id": record_id,
                    "input_image": str(image),
                    "ground_truth_step": str(step),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "input_sha256": sha256_file(image),
                    "ground_truth_sha256": sha256_file(step),
                    "family": record.get("family"),
                    "difficulty": record.get("difficulty"),
                }
            )
    except Exception as exc:  # noqa: BLE001 - preserve render error with cleanup
        preparation_error = exc
    if preparation_error is not None:
        try:
            shutil.rmtree(input_root)
        except OSError as cleanup_error:
            raise RuntimeError(
                f"Manifest preparation failed ({preparation_error}); "
                f"cleanup also failed for {input_root}: {cleanup_error}"
            ) from cleanup_error
        raise preparation_error
    payload = {
        "benchmark_commit": commit,
        "dataset_revision": args.dataset_revision,
        "tasks": tasks,
    }
    manifest_temporary = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )
    manifest_temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(manifest_temporary, output_path)
    print(output_path)


if __name__ == "__main__":
    main()

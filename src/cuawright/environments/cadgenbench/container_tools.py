from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

DEFAULT_VIEWS = ("iso", "front", "top", "right")


def run_prediction(
    prediction: str | Path,
    *,
    output_step: str | Path = "/workspace/output.step",
    timeout: int = 120,
) -> Path:
    prediction_path = Path(prediction).expanduser().resolve()
    output_path = Path(output_step).expanduser().resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        [
            "/usr/bin/timeout",
            "--verbose",
            "--signal=TERM",
            "--kill-after=5s",
            f"{timeout}s",
            sys.executable,
            str(prediction_path),
        ],
        cwd=prediction_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Prediction script failed with return code {result.returncode}: {error}"
        )

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Prediction script completed without producing {output_path}"
        )
    return output_path


def validation_payload(step_path: str | Path) -> dict[str, Any]:
    from cadgenbench.common.validity import analyze_step

    candidate = Path(step_path).expanduser().resolve()
    analysis = analyze_step(candidate)
    validation = analysis.validation
    measurements = analysis.measurements
    return {
        "is_valid": validation.is_valid,
        "is_watertight": validation.is_watertight,
        "solid_count": measurements.solid_count,
        "shell_count": measurements.shell_count,
        "face_count": measurements.face_count,
        "volume": measurements.volume,
        "bbox": {
            "x": measurements.bounding_box.size_x,
            "y": measurements.bounding_box.size_y,
            "z": measurements.bounding_box.size_z,
        },
        "topology_errors": list(validation.topology_errors),
    }


def write_validation(
    step_path: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    payload = validation_payload(step_path)
    destination = Path(output_json).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def render_candidate(
    step_path: str | Path,
    output_dir: str | Path,
    *,
    views: tuple[str, ...] = DEFAULT_VIEWS,
) -> dict[str, Any]:
    from cadgenbench.common.viewer import render_step

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    renders = render_step(step_path, views=views)
    image_paths: list[str] = []
    tiles: list[Image.Image] = []
    for rendered in renders:
        path = destination / f"{rendered.name}.png"
        path.write_bytes(rendered.data)
        image_paths.append(str(path))
        with Image.open(path) as image:
            tiles.append(image.convert("RGB"))
    if not tiles:
        raise RuntimeError("Renderer produced no images")
    width = max(image.width for image in tiles)
    height = max(image.height for image in tiles)
    composite = Image.new("RGB", (width * 2, height * 2), color=(255, 255, 255))
    for index, tile in enumerate(tiles[:4]):
        composite.paste(tile, box=((index % 2) * width, (index // 2) * height))
    composite_path = destination / "multiview.png"
    composite.save(composite_path, format="PNG")
    return {
        "views": list(views),
        "images": image_paths,
        "composite": str(composite_path),
    }


def public_evaluate_candidate(
    candidate_step: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    step_path = Path(candidate_step).expanduser().resolve()
    if not step_path.is_file():
        result = {
            "status": "missing_output",
            "error": f"Missing candidate STEP at {step_path}",
            "validation": None,
            "renders": None,
        }
    else:
        try:
            validation = write_validation(step_path, output_root / "validation.json")
        except Exception as exc:  # noqa: BLE001 - truthful load failure
            result = {
                "status": "load_failed",
                "error": str(exc),
                "validation": None,
                "renders": None,
            }
        else:
            render_error = None
            render_payload = None
            try:
                render_payload = render_candidate(
                    step_path,
                    output_root / "renders",
                )
            except Exception as exc:  # noqa: BLE001 - keep validity result
                render_error = str(exc)
            result = {
                "status": (
                    "pending_official_score"
                    if validation["is_valid"]
                    else "invalid_output"
                ),
                "error": render_error,
                "validation": validation,
                "renders": render_payload,
            }
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result

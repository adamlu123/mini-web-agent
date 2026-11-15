from __future__ import annotations

import base64
import json
import random
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from cuawright.config.prompts import PromptBundle, render_direct_prompt_bundle
from cuawright.environments.benchcad.image import validate_image
from cuawright.environments.benchcad.scoring import score_prediction_with_retry
from cuawright.environments.benchcad.task import (
    Vision2CodeManifest,
    Vision2CodeTask,
    snapshot_task,
    task_provenance,
)

RETRY_DELAYS = (2, 4, 8, 16, 30, 45, 60, 90, 120)


def _create_response(client: OpenAI, request: dict[str, Any]):
    for attempt in range(10):
        try:
            return client.responses.create(**request)
        except (APIConnectionError, APITimeoutError, RateLimitError):
            if attempt == 9:
                raise
        except APIStatusError as exc:
            if exc.status_code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == 9:
                raise
        delay = RETRY_DELAYS[attempt] + random.uniform(0.0, 3.0)
        time.sleep(delay)
    raise RuntimeError("Direct response retry loop exited unexpectedly")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _build_direct_request(
    *,
    prompt_bundle: PromptBundle,
    model: str,
    reasoning_effort: str,
    image_data: str,
    session_id: str,
) -> dict[str, Any]:
    system_prompt, user_prompt = render_direct_prompt_bundle(prompt_bundle)
    return {
        "model": model,
        "instructions": system_prompt,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                        "detail": "high",
                    },
                ],
            }
        ],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": 32_000,
        "store": False,
        "extra_body": {
            "tier": "base",
            "session_id": session_id,
            "strict_session": True,
        },
    }


def run_direct(
    *,
    task: Vision2CodeTask,
    manifest: Vision2CodeManifest,
    result_root: str,
    api_key_file: str,
    base_url: str,
    model: str,
    reasoning_effort: str,
    image: str,
    benchcad_repo: str,
    benchmark_commit: str,
    prompt_bundle: PromptBundle,
    defer_score: bool = False,
) -> dict[str, Any]:
    import docker

    docker_client = docker.from_env()
    try:
        image_id = validate_image(docker_client, image, benchmark_commit)
    finally:
        docker_client.close()
    run_name = (
        f"direct-{task.record_id}-"
        f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = Path(result_root).expanduser().resolve() / run_name
    workspace = run_dir / "workspace"
    requests_dir = run_dir / "requests"
    responses_dir = run_dir / "responses"
    evaluation_dir = run_dir / "evaluation"
    for path in (workspace, requests_dir, responses_dir, evaluation_dir):
        path.mkdir(parents=True, exist_ok=True)

    task = snapshot_task(task, run_dir)
    provenance = task_provenance(manifest, task)
    image_data = base64.b64encode(task.input_image.read_bytes()).decode("ascii")
    request = _build_direct_request(
        prompt_bundle=prompt_bundle,
        model=model,
        reasoning_effort=reasoning_effort,
        image_data=image_data,
        session_id=f"benchcad-direct-{uuid.uuid4().hex}",
    )
    (requests_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    api_key = Path(api_key_file).read_text(encoding="utf-8").strip()
    client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0)
    response_obj = _create_response(client, request)
    response = _jsonable(response_obj)
    (responses_dir / "response.json").write_text(
        json.dumps(response, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    raw = response_obj.output_text or ""
    (workspace / "prediction.py").write_text(raw, encoding="utf-8")
    if defer_score:
        import hashlib

        evaluation = {
            "score": None,
            "iou": None,
            "status": "pending_offline_score",
            "error": None,
            "benchmark_commit": benchmark_commit,
            "container_image_id": image_id,
            "prediction_sha256": hashlib.sha256(
                (workspace / "prediction.py").read_bytes()
            ).hexdigest(),
            **provenance,
        }
    else:
        evaluation = score_prediction_with_retry(
            task=task,
            workspace_dir=workspace,
            output_dir=evaluation_dir / "scorer",
            image=image_id,
            benchcad_repo=Path(benchcad_repo).expanduser().resolve(),
            benchmark_commit=benchmark_commit,
        )
        if evaluation.get("ground_truth_sha256") != provenance["ground_truth_sha256"]:
            raise RuntimeError(
                "Scorer ground-truth snapshot differs from direct provenance"
            )
    result = {
        "task_id": task.record_id,
        "condition": "canonical_direct",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "score": (
            float(evaluation["score"]) if evaluation.get("score") is not None else None
        ),
        "evaluation": evaluation,
        "usage": _jsonable(getattr(response_obj, "usage", None)),
        "run_dir": str(run_dir),
        "provenance": provenance,
    }
    (evaluation_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return result

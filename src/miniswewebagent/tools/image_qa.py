from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from miniswewebagent.models.phyagi_model import _extract_response_text, text_part


def _build_prompt(question: str) -> str:
    return (
        "Answer the user's question using only visible evidence from the screenshot. "
        "If the answer is not visible, say so instead of guessing.\n\n"
        f"Question: {question.strip()}"
    )


def _high_detail_image_part_from_path(image_path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type or 'image/png'};base64,{encoded}",
        "detail": "high",
    }


def _resolve_image_path(image_path: str, workspace_dir: str = "") -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        base_dir = Path(workspace_dir) if workspace_dir else Path.cwd()
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return path


def _gateway_config(args: argparse.Namespace) -> tuple[str, str, str]:
    api_key = (
        args.api_key
        or os.environ.get("OPENAI_GATEWAY_API_KEY", "")
        or os.environ.get("PHYAGI_API_KEY", "")
    )
    if not api_key:
        raise RuntimeError("Missing OPENAI_GATEWAY_API_KEY or PHYAGI_API_KEY.")

    endpoint = args.endpoint or os.environ.get("OPENAI_GATEWAY_ENDPOINT", "http://gateway.phyagi.net/api/responses")
    model = args.model or os.environ.get("OPENAI_GATEWAY_MODEL", "gpt-5.4")
    return api_key, endpoint, model


def run_image_qa(
    *,
    image_path: Path,
    question: str,
    api_key: str,
    endpoint: str,
    model: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    image_part = _high_detail_image_part_from_path(image_path)
    payload = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    text_part(_build_prompt(question)),
                    image_part,
                ],
            }
        ],
        "max_output_tokens": 500,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "image_qa_answer",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "answer": {"type": "string"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unknown": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer", "evidence", "unknown", "confidence"],
                },
                "strict": True,
            }
        },
    }

    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
        )
        response.raise_for_status()
        response_payload = response.json()

    raw_text = _extract_response_text(response_payload).strip()
    parsed = json.loads(raw_text)
    return {
        "image_path": str(image_path),
        "question": question,
        **parsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a visual question about a local image and print JSON.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--question", required=True, help="Question to answer from the image.")
    parser.add_argument("--workspace-dir", default="", help="Optional base directory for relative image paths.")
    parser.add_argument("--model", default="", help="Override the gateway model name.")
    parser.add_argument("--endpoint", default="", help="Override the gateway endpoint.")
    parser.add_argument("--api-key", default="", help="Override the gateway API key.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Gateway request timeout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    image_path = _resolve_image_path(args.image, workspace_dir=args.workspace_dir)
    api_key, endpoint, model = _gateway_config(args)
    result = run_image_qa(
        image_path=image_path,
        question=args.question,
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

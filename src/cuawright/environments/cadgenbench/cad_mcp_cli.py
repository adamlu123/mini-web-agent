from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl, TypeAdapter

DEFAULT_URL = "http://127.0.0.1:8765/mcp"
DEFAULT_TIMEOUT_SECONDS = 600
RESOURCE_URI = TypeAdapter(AnyUrl)


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if isinstance(value, dict):
        return {key: _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    return value


def _load_arguments(args: argparse.Namespace) -> dict[str, Any]:
    sources = sum(value is not None for value in (args.json_arguments, args.json_file))
    if sources > 1:
        raise ValueError("Use only one of --json or --json-file.")
    if args.json_file is not None:
        payload = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    elif args.json_arguments is not None:
        payload = json.loads(args.json_arguments)
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("Tool arguments must decode to a JSON object.")
    return payload


def _save_image(tool_name: str, index: int, item: Any) -> dict[str, Any]:
    mime_type = str(getattr(item, "mimeType", "image/png"))
    extension = {
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(mime_type, ".png")
    safe_tool = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in tool_name
    )
    image_dir = Path(os.environ.get("CAD_MCP_IMAGE_DIR", "/workspace/.cad-mcp/images"))
    image_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}-{safe_tool}-{index}{extension}"
    path = image_dir / name
    data = base64.b64decode(item.data)
    path.write_bytes(data)
    return {
        "type": "image",
        "mime_type": mime_type,
        "path": str(path),
        "size_bytes": len(data),
    }


def _normalize_tool_result(tool_name: str, result: Any) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    images: list[str] = []
    for index, item in enumerate(getattr(result, "content", ()) or (), start=1):
        if getattr(item, "type", None) == "image":
            saved = _save_image(tool_name, index, item)
            content.append(saved)
            images.append(saved["path"])
            continue
        content.append(_model_dump(item))
    return {
        "status": "error" if getattr(result, "isError", False) else "success",
        "tool": tool_name,
        "structured_content": _model_dump(getattr(result, "structuredContent", None)),
        "content": content,
        "images": images,
    }


async def _run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    timeout = timedelta(seconds=args.timeout)
    async with streamablehttp_client(
        args.url,
        timeout=min(30, args.timeout),
        sse_read_timeout=args.timeout + 30,
    ) as (read_stream, write_stream, _):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=timeout,
        ) as session:
            await session.initialize()
            if args.command == "tools":
                result = await session.list_tools()
                tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                    }
                    for tool in result.tools
                ]
                return {"status": "success", "tools": tools}, 0
            if args.command == "schema":
                result = await session.list_tools()
                for tool in result.tools:
                    if tool.name == args.tool:
                        return {
                            "status": "success",
                            "tool": tool.name,
                            "description": tool.description,
                            "parameters": _model_dump(tool.inputSchema),
                        }, 0
                return {
                    "status": "error",
                    "error": f"Unknown MCP tool: {args.tool}",
                }, 2
            if args.command == "resources":
                result = await session.list_resources()
                return {
                    "status": "success",
                    "resources": [_model_dump(item) for item in result.resources],
                }, 0
            if args.command == "resource":
                result = await session.read_resource(
                    RESOURCE_URI.validate_python(args.uri)
                )
                return {
                    "status": "success",
                    "uri": args.uri,
                    "contents": [_model_dump(item) for item in result.contents],
                }, 0
            if args.command == "execute":
                arguments = {"code": Path(args.file).read_text(encoding="utf-8")}
                tool_name = "execute"
            else:
                arguments = _load_arguments(args)
                tool_name = args.tool
            result = await session.call_tool(
                tool_name,
                arguments or None,
                read_timeout_seconds=timeout,
            )
            payload = _normalize_tool_result(tool_name, result)
            return payload, 2 if payload["status"] == "error" else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cad-mcp",
        description=(
            "Call the fixture-local persistent Build123d MCP session. "
            "Tool results are JSON; returned images are saved under "
            "/workspace/.cad-mcp/images."
        ),
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("CAD_MCP_URL", DEFAULT_URL),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(
            os.environ.get(
                "CAD_MCP_TIMEOUT_SECONDS",
                str(DEFAULT_TIMEOUT_SECONDS),
            )
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tools", help="List available MCP tools.")
    schema = commands.add_parser("schema", help="Show one tool's input schema.")
    schema.add_argument("tool")
    commands.add_parser("resources", help="List MCP resources.")
    resource = commands.add_parser("resource", help="Read one MCP resource.")
    resource.add_argument("uri")
    call = commands.add_parser("call", help="Call an MCP tool.")
    call.add_argument("tool")
    call.add_argument("--json", dest="json_arguments")
    call.add_argument("--json-file")
    execute = commands.add_parser(
        "execute",
        help="Call execute with Python source read from a file.",
    )
    execute.add_argument("file")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        payload, returncode = asyncio.run(_run(args))
    except (OSError, TimeoutError, RuntimeError, ValueError) as exc:
        payload = {
            "status": "infrastructure_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        returncode = 70
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()

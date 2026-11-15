from __future__ import annotations

import gzip
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: str
    sha256: str
    size_bytes: int
    media_type: str | None = None


class RunStore(Protocol):
    def event(self, event: str, payload: Mapping[str, object]) -> None:
        """Record one structured run event."""

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        """Persist an artifact and return its stable identity."""


class NullRunStore:
    def event(self, event: str, payload: Mapping[str, object]) -> None:
        pass

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        return ArtifactRef(
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media_type,
        )


class TraceWriterRunStore:
    """RunStore adapter for the existing TraceWriter surface."""

    def __init__(self, trace_writer: object) -> None:
        if not hasattr(trace_writer, "event") or not hasattr(trace_writer, "run_dir"):
            raise TypeError("trace_writer must provide event() and run_dir")
        self._trace_writer = trace_writer

    def event(self, event: str, payload: Mapping[str, object]) -> None:
        writer_event = getattr(self._trace_writer, "event")
        writer_event(event, dict(payload))

    def write_artifact(
        self,
        path: str,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> ArtifactRef:
        relative = PurePosixPath(path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("artifact path must be a safe relative path")
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")

        run_dir = Path(getattr(self._trace_writer, "run_dir"))
        target = run_dir.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return ArtifactRef(
            path=relative.as_posix(),
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            media_type=media_type,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, default=str))
        handle.write("\n")


def write_json_gzip(path: Path, payload: Any) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, default=str)


class TraceWriter:
    """Persist exact provider traffic plus a compact event stream."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.requests_dir = run_dir / "requests"
        self.responses_dir = run_dir / "responses"
        self.command_dir = run_dir / "command-output"
        self.images_dir = run_dir / "images"
        self.evaluation_dir = run_dir / "evaluation"
        for path in (
            run_dir,
            self.requests_dir,
            self.responses_dir,
            self.command_dir,
            self.images_dir,
            self.evaluation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.trace_path = run_dir / "trace.jsonl"
        self.slim_path = run_dir / "trace-slim.jsonl"
        self.call_index = 0
        self._lock = threading.Lock()

    def model_call(self, request: dict[str, Any], response: dict[str, Any]) -> int:
        with self._lock:
            self.call_index += 1
            write_json_gzip(
                self.requests_dir / f"step-{self.call_index:04d}.json.gz",
                request,
            )
            write_json_gzip(
                self.responses_dir / f"step-{self.call_index:04d}.json.gz",
                response,
            )
            return self.call_index

    def event(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        slim_payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            base = {"timestamp": utc_now(), "event": event}
            append_jsonl(self.trace_path, {**base, **payload})
            append_jsonl(
                self.slim_path,
                {**base, **(slim_payload if slim_payload is not None else payload)},
            )

    def command_blob(self, step: int, text: str) -> tuple[str, str]:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        relative = f"command-output/step-{step:04d}-{digest[:12]}.txt.gz"
        with gzip.open(self.run_dir / relative, "wt", encoding="utf-8") as handle:
            handle.write(text)
        return relative, digest

    def image_blob(self, step: int, source_path: str, data: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(data).hexdigest()
        suffix = Path(source_path).suffix.lower() or ".png"
        relative = f"images/step-{step:04d}-{digest[:12]}{suffix}"
        (self.run_dir / relative).write_bytes(data)
        return relative, digest

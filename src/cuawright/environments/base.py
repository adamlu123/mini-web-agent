from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cuawright.runtime.contracts import (
    MODEL_COMMAND_OUTPUT_LIMIT,
    Attachment,
    CommandExecutor,
    CommandResult,
    FileAttachment,
    ImageAttachment,
    cap_command_text,
)


class ControlFileAccessError(OSError):
    """Expected rejection from the trusted, workspace-confined file bridge."""


@dataclass(frozen=True, slots=True)
class CommandObservation:
    role: str
    stage: str
    command: str
    result: CommandResult
    operation: str | None = None
    synthetic: bool = False
    submission_forced: bool | None = None


class CommandObserver(Protocol):
    def __call__(self, observation: CommandObservation) -> None:
        """Observe one completed live-environment dispatch in issue order."""


class BenchmarkPlugin(CommandExecutor, Protocol):
    def read_file(
        self,
        path: str,
        *,
        max_bytes: int,
    ) -> bytes:
        """Read at most max_bytes from one benchmark-visible absolute path."""

    def read_control_file(
        self,
        path: str,
        *,
        workspace: str,
        max_bytes: int,
    ) -> bytes:
        """Securely read a regular file confined beneath workspace.

        Implementations must reject symlinks, traversal, non-regular files, and
        resolved paths outside workspace without invoking a shell.
        """

    def write_control_file_atomic(
        self,
        path: str,
        data: bytes,
        *,
        workspace: str,
    ) -> None:
        """Atomically write beneath workspace without following symlinks."""

    def evaluate(self) -> object:
        """Freeze and evaluate the benchmark's canonical final state."""

    def close(self) -> None:
        """Release benchmark environment resources."""


__all__ = [
    "MODEL_COMMAND_OUTPUT_LIMIT",
    "Attachment",
    "BenchmarkPlugin",
    "CommandObservation",
    "CommandObserver",
    "CommandResult",
    "ControlFileAccessError",
    "FileAttachment",
    "ImageAttachment",
    "cap_command_text",
]

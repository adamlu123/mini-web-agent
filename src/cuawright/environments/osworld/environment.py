from __future__ import annotations

from typing import Any

from cuawright.environments.desktop.plugin import (
    CommandExecutor,
    DesktopCommandPlugin,
    GuestWorkspace,
)
from cuawright.environments.osworld.lifecycle import OSWorldLifecycle


class OSWorldBenchmarkPlugin(DesktopCommandPlugin):
    """OSWorld identity and lifecycle over neutral desktop guest plumbing."""

    def __init__(
        self,
        desktop_env: Any,
        execute_command: CommandExecutor,
        *,
        workspace: GuestWorkspace,
        bridge_timeout: int = 90,
        lifecycle: OSWorldLifecycle | None = None,
    ) -> None:
        super().__init__(
            desktop_env,
            execute_command,
            workspace=workspace,
            bridge_timeout=bridge_timeout,
            bridge_error_label="OSWorld guest bridge",
        )
        self.lifecycle = lifecycle

    def evaluate(self) -> object:
        if self.lifecycle is not None:
            return self.lifecycle.evaluate()
        return self.desktop_env.evaluate()

    def close(self) -> None:
        if self.lifecycle is not None:
            self.lifecycle.close()
            return
        self.desktop_env.close()


__all__ = [
    "GuestWorkspace",
    "OSWorldBenchmarkPlugin",
]

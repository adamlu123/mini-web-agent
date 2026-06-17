from __future__ import annotations

from typing import Any


class InterruptAgentFlow(Exception):
    """Control-flow exception that appends one or more messages to the agent."""

    def __init__(self, *messages: dict[str, Any] | list[dict[str, Any]]):
        if len(messages) == 1 and isinstance(messages[0], list):
            normalized = list(messages[0])
        else:
            normalized = [message for message in messages if isinstance(message, dict)]
        self.messages = normalized
        summary = "; ".join(str(message.get("content", ""))[:120] for message in self.messages)
        super().__init__(summary)


class FormatError(InterruptAgentFlow):
    pass


class LimitsExceeded(InterruptAgentFlow):
    pass


class Submitted(InterruptAgentFlow):
    pass

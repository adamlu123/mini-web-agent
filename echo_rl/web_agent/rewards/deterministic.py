from __future__ import annotations

import re
from typing import Any

from .base import RewardResult


class DeterministicReward:
    """Placeholder for a deterministic per-task reward.

    Drop-in stand-in for the OSW judge: every task is expected to ship its own
    verifier ``check_fn(rollout)`` keyed by ``task_id``. Until per-task
    verifiers exist, this falls back to a tiny heuristic so the training loop
    can run end-to-end and to make wiring easy to test:

    - 1.0 if the agent's ``final_response`` contains any of the configured
      ``expected_substrings`` (per-task or a global list).
    - 0.0 otherwise.

    Replace this with the real verifier once the task suite defines one.
    """

    def __init__(
        self,
        *,
        per_task_expected: dict[str, list[str]] | None = None,
        expected_substrings: list[str] | None = None,
        case_insensitive: bool = True,
    ) -> None:
        self.per_task_expected = per_task_expected or {}
        self.expected_substrings = list(expected_substrings or [])
        self.case_insensitive = bool(case_insensitive)

    def _match(self, text: str, needles: list[str]) -> bool:
        if not needles:
            return False
        haystack = text.lower() if self.case_insensitive else text
        for needle in needles:
            if not needle:
                continue
            probe = needle.lower() if self.case_insensitive else needle
            if probe in haystack:
                return True
        return False

    async def score(
        self,
        *,
        task: str,
        start_url: str,
        actions: list[str],
        thoughts: list[str],
        screenshot_paths: list[str],
        final_response: str = "",
        extra: dict[str, Any] | None = None,
    ) -> RewardResult:
        del start_url, actions, thoughts, screenshot_paths  # unused
        task_id = (extra or {}).get("task_id", "")
        needles = list(self.per_task_expected.get(task_id, []))
        needles.extend(self.expected_substrings)
        haystack = final_response or ""
        # Also grep any explicit "answer: ..." line if the agent surfaced one.
        match = re.search(r"answer\s*[:=]\s*(.+)", haystack, re.IGNORECASE)
        if match:
            haystack = haystack + "\n" + match.group(1)
        matched = self._match(haystack, needles)
        return RewardResult(
            reward=1.0 if matched else 0.0,
            correct=matched,
            metadata={
                "needles": needles,
                "task_id": task_id,
                "matched": matched,
            },
        )

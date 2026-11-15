from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_TOTAL_STEPS = 300
MAX_TOTAL_STEPS = 500
CADGENBENCH_TOTAL_STEPS = 150


class StepBudget(Protocol):
    @property
    def used_steps(self) -> int: ...

    @property
    def remaining_steps(self) -> int: ...

    def consume(self, role: str, stage: str) -> int: ...

    def snapshot(self) -> dict[str, object]: ...

    def shared_snapshot(self) -> dict[str, object]: ...


@dataclass
class SharedStepBudget:
    total_steps: int
    used_steps: int = 0
    by_role: dict[str, int] = field(default_factory=dict)
    by_stage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.total_steps > MAX_TOTAL_STEPS:
            raise ValueError(
                f"total_steps cannot exceed the hard maximum of {MAX_TOTAL_STEPS}"
            )

    @property
    def remaining_steps(self) -> int:
        return self.total_steps - self.used_steps

    @property
    def progress_percent(self) -> float:
        return 100.0 * self.used_steps / self.total_steps

    def consume(self, role: str, stage: str) -> int:
        if self.remaining_steps < 1:
            raise RuntimeError("Shared model-call budget is exhausted")
        self.used_steps += 1
        self.by_role[role] = self.by_role.get(role, 0) + 1
        self.by_stage[stage] = self.by_stage.get(stage, 0) + 1
        return self.used_steps

    def status_text(self) -> str:
        role_usage = ", ".join(
            f"{role}={steps}" for role, steps in sorted(self.by_role.items())
        )
        if not role_usage:
            role_usage = "none yet"
        return (
            f"Global model-call budget: {self.used_steps}/{self.total_steps} used "
            f"({self.progress_percent:.1f}%); {self.remaining_steps} remain. "
            f"Usage by role: {role_usage}."
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "total_steps": self.total_steps,
            "used_steps": self.used_steps,
            "remaining_steps": self.remaining_steps,
            "progress_percent": round(self.progress_percent, 3),
            "by_role": dict(sorted(self.by_role.items())),
            "by_stage": dict(sorted(self.by_stage.items())),
        }

    def shared_snapshot(self) -> dict[str, object]:
        return self.snapshot()


@dataclass
class ScopedStepBudget:
    """A hard local ceiling that consumes from one underlying shared budget."""

    parent: StepBudget
    limit: int
    _consumed_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("scope limit must be an integer")
        if self.limit < 1:
            raise ValueError("scope limit must be positive")
        if self.limit > self.parent.remaining_steps:
            raise ValueError("scope limit cannot exceed the remaining parent budget")

    @property
    def consumed_steps(self) -> int:
        return self._consumed_steps

    @property
    def used_steps(self) -> int:
        return self.parent.used_steps

    @property
    def remaining_steps(self) -> int:
        return min(
            self.limit - self._consumed_steps,
            self.parent.remaining_steps,
        )

    def consume(self, role: str, stage: str) -> int:
        if self.remaining_steps < 1:
            raise RuntimeError("Scoped model-call budget is exhausted")
        global_step = self.parent.consume(role, stage)
        self._consumed_steps += 1
        return global_step

    def snapshot(self) -> dict[str, object]:
        snapshot = self.parent.shared_snapshot()
        snapshot["scope"] = {
            "limit": self.limit,
            "used_steps": self._consumed_steps,
            "remaining_steps": self.remaining_steps,
        }
        return snapshot

    def shared_snapshot(self) -> dict[str, object]:
        return self.parent.shared_snapshot()

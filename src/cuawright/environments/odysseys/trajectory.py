from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


_SCREENSHOT_PATTERN = re.compile(
    r"^screenshots/step_(?P<step>[0-9]{4})\.(?:png|jpg|jpeg|gif|webp)$"
)


class OdysseysTrajectoryContractError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedTrajectory:
    rows: tuple[Mapping[str, Any], ...]
    screenshot_paths: tuple[str, ...]


def validate_trajectory_contract(
    trajectory_bytes: bytes,
    *,
    action_count: int,
    screenshot_count: int,
    screenshot_paths: Iterable[str],
    submission_classification: str | None,
    expected_final_response: str | None,
    require_final: bool,
) -> ValidatedTrajectory:
    if (
        isinstance(action_count, bool)
        or not isinstance(action_count, int)
        or action_count < 0
    ):
        raise OdysseysTrajectoryContractError("action count must be non-negative")
    if (
        isinstance(screenshot_count, bool)
        or not isinstance(screenshot_count, int)
        or screenshot_count < 0
    ):
        raise OdysseysTrajectoryContractError("screenshot count must be non-negative")
    try:
        text = trajectory_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OdysseysTrajectoryContractError("trajectory is not valid UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise OdysseysTrajectoryContractError(
                f"trajectory line {line_number} is empty"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OdysseysTrajectoryContractError(
                f"trajectory line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise OdysseysTrajectoryContractError(
                f"trajectory line {line_number} is not an object"
            )
        rows.append(row)
    if len(rows) != action_count:
        raise OdysseysTrajectoryContractError(
            f"trajectory rows {len(rows)} do not match action count {action_count}"
        )

    available = tuple(screenshot_paths)
    if len(available) != len(set(available)):
        raise OdysseysTrajectoryContractError(
            "canonical screenshot hash entries are duplicated"
        )
    for path in available:
        _canonical_screenshot(path)
    references: list[str] = []
    reference_steps: list[int] = []
    final_indices: list[int] = []
    for expected_step, row in enumerate(rows, start=1):
        step = row.get("step_num")
        if step != expected_step:
            raise OdysseysTrajectoryContractError(
                "trajectory step numbers must be consecutive from one"
            )
        screenshot = row.get("screenshot_file")
        if not isinstance(screenshot, str):
            raise OdysseysTrajectoryContractError(
                f"trajectory step {expected_step} has no screenshot_file"
            )
        screenshot_step = _canonical_screenshot(screenshot)
        references.append(screenshot)
        reference_steps.append(screenshot_step)
        if row.get("final") is True:
            final_indices.append(expected_step)
    if len(references) != len(set(references)):
        raise OdysseysTrajectoryContractError(
            "trajectory screenshot references must be unique"
        )
    for expected_step, screenshot_step in enumerate(reference_steps, start=1):
        if screenshot_step != expected_step:
            raise OdysseysTrajectoryContractError(
                f"trajectory step {expected_step} references noncanonical screenshot"
            )
    reference_set = set(references)
    available_set = set(available)
    missing = sorted(reference_set - available_set)
    orphaned = sorted(available_set - reference_set)
    if missing:
        raise OdysseysTrajectoryContractError(
            "trajectory references screenshots without canonical hashes: "
            + ", ".join(missing)
        )
    if orphaned:
        raise OdysseysTrajectoryContractError(
            "canonical screenshot hashes include orphan files: " + ", ".join(orphaned)
        )
    if screenshot_count != len(references):
        raise OdysseysTrajectoryContractError(
            "screenshot count does not match trajectory references"
        )

    if require_final:
        if submission_classification not in {"accepted", "forced"}:
            raise OdysseysTrajectoryContractError(
                "ready trajectory lacks an accepted or forced submission"
            )
        if not isinstance(expected_final_response, str) or not expected_final_response:
            raise OdysseysTrajectoryContractError(
                "ready trajectory lacks a nonempty final response"
            )
        if final_indices != [action_count] or not rows:
            raise OdysseysTrajectoryContractError(
                "ready trajectory must have exactly one final last row"
            )
        final_row = rows[-1]
        if final_row.get("done") is not True:
            raise OdysseysTrajectoryContractError(
                "final trajectory row must also be done"
            )
        if final_row.get("response") != expected_final_response:
            raise OdysseysTrajectoryContractError(
                "final trajectory response does not match immutable final response"
            )
        info = final_row.get("info")
        if (
            not isinstance(info, Mapping)
            or info.get("submission_classification") != submission_classification
        ):
            raise OdysseysTrajectoryContractError(
                "final trajectory row submission classification does not match"
            )
    elif final_indices:
        raise OdysseysTrajectoryContractError(
            "incomplete trajectory cannot contain a final row"
        )
    return ValidatedTrajectory(tuple(rows), tuple(references))


def _canonical_screenshot(path: str) -> int:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise OdysseysTrajectoryContractError(
            f"unsafe trajectory screenshot reference: {path!r}"
        )
    match = _SCREENSHOT_PATTERN.fullmatch(candidate.as_posix())
    if match is None:
        raise OdysseysTrajectoryContractError(
            f"noncanonical trajectory screenshot reference: {path!r}"
        )
    return int(match.group("step"))

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Mapping


HEALTH_CONTRACT_NAME: Final = "cuawright-terminal-health"
HEALTH_CONTRACT_VERSION: Final = 1

TRACE_EVENT_REQUIRED_FIELDS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "control_client_executed": (
            "command",
            "python_executable",
            "script_path",
            "operation",
            "exit_code",
            "stdout_size_bytes",
            "stderr_size_bytes",
        ),
        "control_client_dispatched": (
            "command",
            "script_path",
            "operation",
            "exit_code",
            "completed",
            "attachment_count",
            "direct_report_returned",
        ),
        "specialist_context_prepared": (
            "role",
            "invocation_id",
            "phase_id",
            "prior_roles",
            "prior_reports",
        ),
        "specialist_completed": (
            "role",
            "invocation_id",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class HealthAdapterMetadata:
    supported: bool
    benchmark_names: tuple[str, ...]
    progress_markers: tuple[str, ...] = ()
    lifecycle_directory: str | None = None
    incomplete_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.benchmark_names:
            raise ValueError("health adapter must name at least one benchmark")
        if self.supported and self.incomplete_reason is not None:
            raise ValueError(
                "supported health adapter cannot have an incomplete reason"
            )
        if not self.supported and not self.incomplete_reason:
            raise ValueError("unsupported health adapter requires an incomplete reason")


def health_contract_manifest() -> dict[str, Any]:
    return {
        "name": HEALTH_CONTRACT_NAME,
        "version": HEALTH_CONTRACT_VERSION,
        "event_required_fields": {
            event: list(fields) for event, fields in TRACE_EVENT_REQUIRED_FIELDS.items()
        },
        "prompt_identity": "manifest-prompt-hashes-v1",
    }


def validate_health_contract(value: Any) -> tuple[bool, str]:
    if value is None:
        return True, "legacy manifest without an explicit health contract"
    if not isinstance(value, Mapping):
        return False, "health_contract must be an object"
    if value.get("name") != HEALTH_CONTRACT_NAME:
        return False, f"unexpected health contract name: {value.get('name')!r}"
    if value.get("version") != HEALTH_CONTRACT_VERSION:
        return False, f"unsupported health contract version: {value.get('version')!r}"
    expected_events = health_contract_manifest()["event_required_fields"]
    if value.get("event_required_fields") != expected_events:
        return False, "health contract event fields do not match version 1"
    if value.get("prompt_identity") != "manifest-prompt-hashes-v1":
        return False, "health contract prompt identity does not match version 1"
    return True, "versioned health contract recorded"


def missing_event_fields(row: Mapping[str, Any]) -> tuple[str, ...]:
    event = row.get("event")
    if not isinstance(event, str):
        return ()
    required = TRACE_EVENT_REQUIRED_FIELDS.get(event, ())
    return tuple(field for field in required if field not in row)


def request_role_from_prompt_hashes(
    manifest: Mapping[str, Any],
    request: Mapping[str, Any],
) -> str | None:
    system_texts = [value for value in _system_texts(request) if isinstance(value, str)]
    if not system_texts:
        return None
    recorded = manifest.get("prompt_hashes")
    if not isinstance(recorded, Mapping):
        return None
    observed = {
        hashlib.sha256(text.encode("utf-8")).hexdigest() for text in system_texts
    }
    matches: list[str] = []
    for role, entry in recorded.items():
        if not isinstance(role, str) or not isinstance(entry, Mapping):
            continue
        hashes = entry.get("sha256")
        if not isinstance(hashes, Mapping):
            continue
        composed = hashes.get("composed_system")
        if isinstance(composed, str) and composed in observed:
            matches.append(role)
    return matches[0] if len(matches) == 1 else None


def _system_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, Mapping):
        instructions = value.get("instructions")
        if isinstance(instructions, str):
            texts.append(instructions)
        role = value.get("role")
        content = value.get("content")
        if role == "system":
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping):
                        text = item.get("text")
                        if isinstance(text, str):
                            texts.append(text)
        for item in value.values():
            texts.extend(_system_texts(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            texts.extend(_system_texts(item))
    return texts

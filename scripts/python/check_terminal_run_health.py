#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shlex
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cuawright.environments.control_clients import (
    CLIENT_FILENAMES,
    CLIENT_INSTALL_ROOT,
    CLIENT_SHA256,
)
from cuawright.environments.control import (
    MAX_REPORT_BYTES,
    MAX_REPORT_CHARACTERS,
    MAX_UNRESOLVED_ITEM_CHARACTERS,
    MAX_UNRESOLVED_ITEMS,
)
from cuawright.benchmarks import BENCHMARKS, benchmark_for_health_name
from cuawright.health.contracts import (
    HEALTH_CONTRACT_NAME,
    HEALTH_CONTRACT_VERSION,
    missing_event_fields,
    request_role_from_prompt_hashes,
    validate_health_contract,
)


FATAL_MARKERS = (
    "authenticationerror",
    "permissiondenied",
    "model.no_endpoint",
    "invalid_request_error",
    "unknown tool",
    "no such file or directory: 'agent'",
    "no such file or directory: '/opt/cuawright-tools/",
    "legacy agent command protocol has been removed",
    "request_too_large",
)
SPECIALIST_ROLES = ("binder", "critic", "verifier")
NATIVE_REPORT_STATUSES = frozenset(
    {"complete", "blocked", "accept", "concerns", "ready", "repair", "unknown"}
)
ROLE_REPORT_STATUSES = {
    "binder": frozenset({"complete", "blocked"}),
    "critic": frozenset({"accept", "concerns", "blocked"}),
    "verifier": frozenset({"ready", "repair", "unknown"}),
}
PRIOR_AWARE_ROLES = ("critic", "verifier")
REQUIRED_PRIOR_ROLES = {
    "critic": ("binder",),
    "verifier": (),
}
ALLOWED_PRIOR_ROLES = {
    "critic": ("binder",),
    "verifier": ("binder", "critic"),
}
EXPECTED_CLIENT_PATHS = {f"{CLIENT_INSTALL_ROOT}/{name}" for name in CLIENT_FILENAMES}
EXPECTED_CLIENT_OPERATIONS = {
    f"{CLIENT_INSTALL_ROOT}/{name}": name.removesuffix(".py")
    for name in CLIENT_FILENAMES
}
LEGACY_AGENT_COMMAND = re.compile(
    r"(?:^|[;&|\n])\s*agent\s+"
    r"(?:spawn|report|status|submit|image\s+show|user\s+ask)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the predefined five-minute terminal-run health gate"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS),
        required=True,
    )
    parser.add_argument("--pid", type=int)
    parser.add_argument("--process-log")
    parser.add_argument("--expected-model", default="gpt-5.5")
    parser.add_argument("--expected-reasoning", default="xhigh")
    parser.add_argument("--expected-infrastructure", default="aiservices")
    parser.add_argument(
        "--expected-agent-topology",
        choices=["orchestrated", "linear"],
        default="orchestrated",
    )
    parser.add_argument("--max-stale-seconds", type=int, default=180)
    return parser.parse_args()


def check_run_health(
    run_dir: Path,
    *,
    benchmark: str,
    expected_model: str,
    expected_reasoning: str,
    expected_infrastructure: str = "aiservices",
    expected_agent_topology: str = "orchestrated",
    pid: int | None = None,
    process_log: Path | None = None,
    max_stale_seconds: int = 180,
) -> dict[str, Any]:
    now = time.time()
    checks: dict[str, dict[str, Any]] = {}
    fatal: list[str] = []
    warnings: list[str] = []
    descriptor = benchmark_for_health_name(benchmark)
    adapter = descriptor.health
    checks["health_adapter"] = {
        "ok": adapter.supported,
        "benchmark": descriptor.name,
        "supported": adapter.supported,
        "reason": adapter.incomplete_reason,
    }
    if not adapter.supported:
        payload = {
            "schema_version": 3,
            "contract": {
                "name": HEALTH_CONTRACT_NAME,
                "version": HEALTH_CONTRACT_VERSION,
            },
            "checked_at_epoch": now,
            "benchmark": benchmark,
            "run_dir": str(run_dir),
            "status": "incomplete",
            "objective_stop_required": False,
            "fatal": [],
            "warnings": [adapter.incomplete_reason],
            "checks": checks,
        }
        output = run_dir / "health-check-5m.json"
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return payload

    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    model_provider = "openai"
    checks["manifest"] = {
        "ok": manifest is not None,
        "path": str(manifest_path),
    }
    if manifest is None:
        fatal.append("manifest.json is missing or invalid")
        manifest = {}
    else:
        model_provider = str(manifest.get("model_provider") or "openai")
        contract_ok, contract_detail = validate_health_contract(
            manifest.get("health_contract")
        )
        checks["manifest_health_contract"] = {
            "ok": contract_ok,
            "detail": contract_detail,
            "recorded": manifest.get("health_contract"),
        }
        if not contract_ok:
            fatal.append(contract_detail)
        _expect(
            checks,
            fatal,
            "model",
            manifest.get("model") == expected_model,
            manifest.get("model"),
            expected_model,
        )
        _expect(
            checks,
            fatal,
            "reasoning_effort",
            manifest.get("reasoning_effort") == expected_reasoning,
            manifest.get("reasoning_effort"),
            expected_reasoning,
        )
        _expect(
            checks,
            fatal,
            "agent_topology",
            manifest.get("agent_topology", "orchestrated") == expected_agent_topology,
            manifest.get("agent_topology", "orchestrated"),
            expected_agent_topology,
        )
        _expect(
            checks,
            fatal,
            "terminal_interface",
            manifest.get("model_interface") in {None, "terminal-only"},
            manifest.get("model_interface"),
            "terminal-only once prompt hashes are written",
        )

    completed = (run_dir / "evaluation" / "result.json").is_file()
    process_alive = _process_alive(pid) if pid is not None else None
    checks["process"] = {
        "ok": completed or process_alive is not False,
        "pid": pid,
        "alive": process_alive,
        "completed": completed,
    }
    if process_alive is False and not completed:
        fatal.append("task process exited without an evaluation result")

    trace_path = run_dir / "trace.jsonl"
    trace_rows = _read_jsonl(trace_path)
    trace_mtime = trace_path.stat().st_mtime if trace_path.is_file() else None
    trace_fresh = trace_mtime is not None and now - trace_mtime <= max_stale_seconds
    checks["trace"] = {
        "ok": bool(trace_rows) or completed,
        "events": len(trace_rows),
        "fresh": trace_fresh,
        "age_seconds": None if trace_mtime is None else round(now - trace_mtime, 3),
    }

    log_text = ""
    log_fresh = False
    if process_log is not None and process_log.is_file():
        log_text = process_log.read_text(encoding="utf-8", errors="replace")
        log_fresh = now - process_log.stat().st_mtime <= max_stale_seconds
    fatal_hits = [marker for marker in FATAL_MARKERS if marker in log_text.lower()]
    checks["process_log"] = {
        "ok": not fatal_hits,
        "path": None if process_log is None else str(process_log),
        "fresh": log_fresh,
        "fatal_markers": fatal_hits,
    }
    fatal.extend(f"fatal process log marker: {marker}" for marker in fatal_hits)

    expected_hashes = dict(CLIENT_SHA256)
    client_sources_ok = tuple(expected_hashes) == CLIENT_FILENAMES and all(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
        for digest in expected_hashes.values()
    )
    checks["control_client_sources"] = {
        "ok": client_sources_ok,
        "install_root": CLIENT_INSTALL_ROOT,
        "clients": expected_hashes,
    }
    if not client_sources_ok:
        fatal.append("host control-client source manifest is invalid")
    manifest_control = manifest.get("terminal_control")
    if isinstance(manifest_control, Mapping):
        manifest_control_ok = (
            manifest_control.get("protocol") == "python-clients-v1"
            and manifest_control.get("install_root") == CLIENT_INSTALL_ROOT
            and manifest_control.get("clients") == expected_hashes
        )
        checks["manifest_control_clients"] = {
            "ok": manifest_control_ok,
            "recorded": dict(manifest_control),
        }
        if not manifest_control_ok:
            fatal.append("manifest control-client hashes or protocol are incorrect")
    else:
        checks["manifest_control_clients"] = {
            "ok": True,
            "recorded": None,
        }

    requests = _gzip_json_values(run_dir / "requests")
    request = requests[-1] if requests else None
    if request is None:
        setup_progress = process_alive is True or (
            process_alive is not False and (trace_fresh or log_fresh)
        )
        checks["provider_request"] = {
            "ok": setup_progress,
            "state": "environment_setup" if setup_progress else "missing",
        }
        if not setup_progress and not completed:
            fatal.append("no provider request and no fresh setup heartbeat")
    else:
        tools = request.get("tools") or []
        tool_names = [tool.get("name") for tool in tools if isinstance(tool, Mapping)]
        extra_body = request.get("extra_body") or {}
        compaction_request = not tool_names and _is_compaction_request(request)
        specialist_report_request = (
            not tool_names and _is_final_specialist_report_request(request)
        )
        _expect(
            checks,
            fatal,
            "sole_tool",
            tool_names == ["run_command"]
            or compaction_request
            or specialist_report_request,
            tool_names,
            "run_command or recognized no-tool compaction/specialist report",
        )
        checks["sole_tool"]["compaction"] = compaction_request
        checks["sole_tool"]["specialist_report"] = specialist_report_request
        if model_provider == "openai":
            _expect(
                checks,
                fatal,
                "strict_session",
                extra_body.get("strict_session") is True,
                extra_body.get("strict_session"),
                True,
            )
            _expect(
                checks,
                fatal,
                "infrastructure",
                extra_body.get("infrastructure") == expected_infrastructure,
                extra_body.get("infrastructure"),
                expected_infrastructure,
            )
            _expect(
                checks,
                fatal,
                "session_id",
                bool(extra_body.get("session_id")),
                extra_body.get("session_id"),
                "non-empty",
            )
        else:
            checks["strict_session"] = {
                "ok": True,
                "actual": None,
                "expected": f"not applicable for provider {model_provider}",
            }
            _expect(
                checks,
                fatal,
                "infrastructure",
                manifest.get("model_infrastructure") == expected_infrastructure,
                manifest.get("model_infrastructure"),
                expected_infrastructure,
            )
            if model_provider == "openrouter-anthropic":
                _expect(
                    checks,
                    fatal,
                    "session_id",
                    bool(extra_body.get("session_id")),
                    extra_body.get("session_id"),
                    "non-empty",
                )
            else:
                checks["session_id"] = {
                    "ok": True,
                    "actual": None,
                    "expected": f"not applicable for provider {model_provider}",
                }

    commands = _extract_commands([*trace_rows, *requests])
    executed_client_events = [
        row for row in trace_rows if row.get("event") == "control_client_executed"
    ]
    dispatched_client_events = [
        row for row in trace_rows if row.get("event") == "control_client_dispatched"
    ]
    legacy_rejection_events = [
        row for row in trace_rows if row.get("event") == "legacy_control_rejected"
    ]
    legacy_commands = [
        command for command in commands if _is_active_legacy_agent_command(command)
    ]
    (
        recovered_invalid_client_commands,
        invalid_client_commands,
        observed_direct_client_commands,
    ) = _classify_direct_client_attempts(trace_rows)
    exact_client_paths: list[str] = []
    for command in commands:
        argv = _standalone_command_words(command)
        if argv is None:
            continue
        python_script = (
            _python_script_argument(argv)
            if argv and Path(argv[0]).name.startswith("python")
            else None
        )
        active_client_attempt = (
            bool(argv) and argv[0].startswith(f"{CLIENT_INSTALL_ROOT}/")
        ) or (
            python_script is not None
            and python_script.startswith(f"{CLIENT_INSTALL_ROOT}/")
        )
        if not active_client_attempt:
            continue
        if len(argv) >= 2 and argv[0] == "python" and argv[1] in EXPECTED_CLIENT_PATHS:
            exact_client_paths.append(argv[1])
        elif argv and argv[0] in EXPECTED_CLIENT_PATHS:
            if command not in observed_direct_client_commands:
                invalid_client_commands.append(command)
        else:
            invalid_client_commands.append(command)
    client_event_failures: list[str] = []
    for row in executed_client_events:
        script_path = row.get("script_path")
        missing_fields = missing_event_fields(row)
        if (
            missing_fields
            or row.get("python_executable") != "python"
            or script_path not in EXPECTED_CLIENT_PATHS
            or row.get("operation") != EXPECTED_CLIENT_OPERATIONS.get(script_path)
            or not isinstance(row.get("command"), str)
            or row.get("command") not in commands
            or not isinstance(row.get("exit_code"), int)
            or not isinstance(row.get("stdout_size_bytes"), int)
            or not isinstance(row.get("stderr_size_bytes"), int)
        ):
            client_event_failures.append(
                f"invalid control_client_executed event for {script_path!r}"
            )
    for row in dispatched_client_events:
        script_path = row.get("script_path")
        missing_fields = missing_event_fields(row)
        if (
            missing_fields
            or script_path not in EXPECTED_CLIENT_PATHS
            or row.get("operation") != EXPECTED_CLIENT_OPERATIONS.get(script_path)
            or not isinstance(row.get("command"), str)
            or row.get("command") not in commands
            or not isinstance(row.get("exit_code"), int)
            or not isinstance(row.get("completed"), bool)
            or not isinstance(row.get("attachment_count"), int)
            or not isinstance(row.get("direct_report_returned"), bool)
        ):
            client_event_failures.append(
                f"invalid control_client_dispatched event for {script_path!r}"
            )
    successful_client_events = [
        row for row in executed_client_events if row.get("exit_code") == 0
    ]
    successful_client_commands = Counter(
        str(row.get("command")) for row in successful_client_events
    )
    dispatched_client_commands = Counter(
        str(row.get("command")) for row in dispatched_client_events
    )
    remaining_dispatches = dispatched_client_commands.copy()
    unmatched_client_events: list[Mapping[str, Any]] = []
    for row in successful_client_events:
        command = str(row.get("command"))
        if remaining_dispatches[command] > 0:
            remaining_dispatches[command] -= 1
        else:
            unmatched_client_events.append(row)
    missing_dispatches = Counter(
        str(row.get("command")) for row in unmatched_client_events
    )
    active_specialist_counts: Counter[str] = Counter()
    for row in trace_rows:
        if row.get("event") == "specialist_context_prepared":
            role = str(row.get("role"))
            if role in SPECIALIST_ROLES:
                active_specialist_counts[role] += 1
        elif row.get("event") == "control_client_dispatched":
            operation = str(row.get("operation"))
            if (
                row.get("role") == "actor"
                and operation in SPECIALIST_ROLES
                and active_specialist_counts[operation] > 0
            ):
                active_specialist_counts[operation] -= 1
    pending_specialist_dispatches: Counter[str] = Counter()
    recent_dispatches: Counter[str] = Counter()
    orphaned_dispatches: Counter[str] = Counter()
    pending_role_counts: Counter[str] = Counter()
    for row in unmatched_client_events:
        command = str(row.get("command"))
        operation = str(row.get("operation"))
        if (
            process_alive is True
            and operation in SPECIALIST_ROLES
            and pending_role_counts[operation] < active_specialist_counts[operation]
        ):
            pending_role_counts[operation] += 1
            pending_specialist_dispatches[command] += 1
        elif (
            operation not in SPECIALIST_ROLES
            and process_alive is True
            and _event_age_seconds(row, now) <= 5
        ):
            recent_dispatches[command] += 1
        else:
            orphaned_dispatches[command] += 1
    dispatch_pending = bool(pending_specialist_dispatches or recent_dispatches)
    checks["control_protocol"] = {
        "ok": (
            not legacy_commands
            and not legacy_rejection_events
            and not invalid_client_commands
            and not client_event_failures
            and not orphaned_dispatches
        ),
        "legacy_commands": legacy_commands,
        "legacy_rejection_events": len(legacy_rejection_events),
        "invalid_client_commands": invalid_client_commands,
        "recovered_invalid_client_commands": recovered_invalid_client_commands,
        "client_event_failures": client_event_failures,
        "missing_dispatches": dict(missing_dispatches),
        "pending_specialist_dispatches": dict(pending_specialist_dispatches),
        "recent_dispatches": dict(recent_dispatches),
        "orphaned_dispatches": dict(orphaned_dispatches),
        "dispatch_pending": dispatch_pending,
        "exact_client_paths": sorted(set(exact_client_paths)),
        "executed_events": len(executed_client_events),
        "dispatched_events": len(dispatched_client_events),
    }
    if legacy_commands or legacy_rejection_events:
        fatal.append("active legacy `agent ...` control command detected")
    if invalid_client_commands:
        fatal.append("non-exact /opt/cuawright-tools control command detected")
    if recovered_invalid_client_commands:
        warnings.append(
            "non-exact control-client command failed without host dispatch "
            "and was later recovered exactly"
        )
    if client_event_failures:
        fatal.append("control-client execution/dispatch trace is invalid")
    if dispatch_pending:
        warnings.append("control-client host dispatch is still completing")
    if orphaned_dispatches:
        fatal.append("successful control-client execution was not dispatched")
    if request is not None and not exact_client_paths and not completed:
        warnings.append("no exact Python control-client command has completed yet")

    command_rows = [
        row
        for row in trace_rows
        if row.get("event") in {"command_result", "terminal_command"}
    ]
    checks["terminal_progress"] = {
        "ok": bool(command_rows) or completed or request is None,
        "command_events": len(command_rows),
    }
    if request is not None and not command_rows and not completed:
        warnings.append(
            "model request exists but no terminal command has completed yet"
        )

    specialist_completions = [
        row for row in trace_rows if row.get("event") == "specialist_completed"
    ]
    specialist_context_events = [
        row for row in trace_rows if row.get("event") == "specialist_context_prepared"
    ]
    specialist_client_commands = [
        command
        for command in commands
        if any(
            f"{CLIENT_INSTALL_ROOT}/{role}.py" in command for role in SPECIALIST_ROLES
        )
    ]
    linear_specialists_ok = not (
        specialist_completions
        or specialist_context_events
        or specialist_client_commands
    )
    checks["linear_specialist_isolation"] = {
        "ok": expected_agent_topology != "linear" or linear_specialists_ok,
        "expected_topology": expected_agent_topology,
        "completion_events": len(specialist_completions),
        "context_events": len(specialist_context_events),
        "client_commands": specialist_client_commands,
    }
    if expected_agent_topology == "linear" and not linear_specialists_ok:
        fatal.append("linear run attempted or created a specialist")
    direct_report_roles: list[str] = []
    for row in command_rows:
        result_payload = _json_object(row.get("stdout"))
        if _is_direct_report(result_payload):
            direct_report_roles.append(_direct_report_role(result_payload))
    completed_roles = Counter(
        str(row.get("role"))
        for row in specialist_completions
        if row.get("role") in SPECIALIST_ROLES
    )
    returned_roles = Counter(direct_report_roles)
    traced_direct_roles = Counter(
        str(row.get("role"))
        for row in specialist_completions
        if row.get("role") in SPECIALIST_ROLES
        and row.get("direct_report_returned") is True
        and isinstance(row.get("report_size_bytes"), int)
        and 0 < int(row["report_size_bytes"]) <= MAX_REPORT_BYTES
    )
    dispatched_direct_roles = Counter(
        str(row.get("operation"))
        for row in dispatched_client_events
        if row.get("operation") in SPECIALIST_ROLES
        and row.get("exit_code") == 0
        and row.get("direct_report_returned") is True
    )
    required_direct_reports = max(
        sum(completed_roles.values()),
        sum(dispatched_direct_roles.values()),
    )
    direct_reports_ok = len(direct_report_roles) >= required_direct_reports
    direct_reports_pending = (
        not direct_reports_ok
        and process_alive is True
        and trace_mtime is not None
        and now - trace_mtime <= 5
    )
    checks["direct_report_return"] = {
        "ok": direct_reports_ok or direct_reports_pending,
        "pending": direct_reports_pending,
        "specialist_completions": dict(completed_roles),
        "direct_report_results": dict(returned_roles),
        "direct_report_result_count": len(direct_report_roles),
        "required_direct_report_count": required_direct_reports,
        "traced_direct_reports": dict(traced_direct_roles),
        "dispatched_direct_reports": dict(dispatched_direct_roles),
    }
    if direct_reports_pending:
        warnings.append("direct specialist report dispatch is still completing")
    elif not direct_reports_ok:
        fatal.append(
            "specialist completion is missing its direct bounded report result"
        )

    context_events = specialist_context_events
    context_failures: list[dict[str, Any]] = []
    completed_keys: set[tuple[str, str]] = set()
    completed_invocations: list[str] = []
    for completion in specialist_completions:
        role = completion.get("role")
        if role not in PRIOR_AWARE_ROLES:
            continue
        invocation_id = completion.get("invocation_id")
        if not isinstance(invocation_id, str) or not invocation_id:
            continue
        completed_keys.add((role, invocation_id))
        completed_invocations.append(invocation_id)

    contexts_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in context_events:
        role = row.get("role")
        invocation_id = row.get("invocation_id")
        if role in PRIOR_AWARE_ROLES and isinstance(invocation_id, str):
            contexts_by_key.setdefault((role, invocation_id), []).append(row)

    captured_requests = _captured_prior_injection_requests(requests)
    inline_requests = _captured_inline_prior_injection_requests(requests, manifest)
    captured_keys = set(captured_requests)
    inline_context_keys = {
        key for key in contexts_by_key if inline_requests.get(key[0])
    }
    required_keys = completed_keys | captured_keys | inline_context_keys
    context_evidence: dict[str, list[str]] = {role: [] for role in PRIOR_AWARE_ROLES}
    request_evidence: dict[str, list[str]] = {role: [] for role in PRIOR_AWARE_ROLES}
    missing_context_events: list[str] = []
    missing_request_evidence: list[str] = []
    inline_evidence_remaining = {
        role: list(results) for role, results in inline_requests.items()
    }
    for role, invocation_id in sorted(required_keys):
        matching_contexts = contexts_by_key.get((role, invocation_id), [])
        valid_context = next(
            (row for row in matching_contexts if _valid_prior_context_event(row, role)),
            None,
        )
        if valid_context is None and matching_contexts:
            missing_context_events.append(invocation_id)
            context_failures.append(
                _prior_context_failure(
                    matching_contexts[-1] if matching_contexts else None,
                    role,
                    invocation_id,
                )
            )
        elif valid_context is not None:
            context_evidence[role].append(invocation_id)

        request_valid = captured_requests.get((role, invocation_id))
        if request_valid is True:
            request_evidence[role].append(invocation_id)
        else:
            expected_inline_roles = (
                tuple(valid_context.get("prior_roles", ()))
                if valid_context is not None
                else None
            )
            matching_inline = next(
                (
                    evidence
                    for evidence in inline_evidence_remaining.get(role, [])
                    if evidence["valid"]
                    and (
                        expected_inline_roles is None
                        or evidence["prior_roles"] == expected_inline_roles
                    )
                ),
                None,
            )
            if matching_inline is not None:
                inline_evidence_remaining[role].remove(matching_inline)
                request_evidence[role].append(invocation_id)
            elif (role, invocation_id) in completed_keys or request_valid is False:
                missing_request_evidence.append(invocation_id)

    invalid_inline_roles = sorted(
        role
        for role, evidence_items in inline_requests.items()
        if any(not evidence["valid"] for evidence in evidence_items)
    )
    prior_injection_ok = (
        not context_failures
        and not missing_context_events
        and not missing_request_evidence
        and not invalid_inline_roles
    )
    checks["prior_report_injection"] = {
        "ok": prior_injection_ok,
        "context_events": len(context_events),
        "completed_invocations": sorted(completed_invocations),
        "active_request_invocations": sorted(
            invocation_id for role, invocation_id in captured_keys - completed_keys
        ),
        "request_race_invocations": sorted(
            invocation_id
            for role, invocation_id in set(contexts_by_key) - required_keys
        ),
        "context_evidence_invocations": {
            role: sorted(invocations)
            for role, invocations in context_evidence.items()
            if invocations
        },
        "request_evidence_roles": sorted(
            role for role, invocations in request_evidence.items() if invocations
        ),
        "request_evidence_invocations": {
            role: sorted(invocations)
            for role, invocations in request_evidence.items()
            if invocations
        },
        "captured_request_invocations": sorted(
            invocation_id for role, invocation_id in captured_keys
        ),
        "inline_request_roles": sorted(inline_requests),
        "inline_request_prior_roles": {
            role: [list(evidence["prior_roles"]) for evidence in evidence_items]
            for role, evidence_items in inline_requests.items()
        },
        "invalid_inline_request_roles": invalid_inline_roles,
        "failures": context_failures,
        "missing_context_events": sorted(missing_context_events),
        "missing_request_evidence": sorted(missing_request_evidence),
    }
    if not prior_injection_ok:
        fatal.append(
            "active/completed Critic/Verifier specialist is missing valid "
            "context metadata or captured prior-report provider evidence"
        )

    image_events = [row for row in trace_rows if row.get("event") == "image_attached"]
    image_results = [
        attachment
        for row in command_rows
        for attachment in row.get("attachments", [])
        if isinstance(attachment, Mapping) and attachment.get("kind") == "image"
    ]
    image_client_count = sum(
        path == f"{CLIENT_INSTALL_ROOT}/image_show.py" for path in exact_client_paths
    )
    image_dispatches = [
        row
        for row in dispatched_client_events
        if row.get("operation") == "image_show"
        and row.get("exit_code") == 0
        and row.get("attachment_count") == 1
    ]
    typed_images_ok = (
        len(image_results) >= len(image_events)
        and image_client_count >= len(image_events)
        and len(image_dispatches) >= len(image_events)
    )
    typed_images_pending = (
        not typed_images_ok
        and process_alive is True
        and trace_mtime is not None
        and now - trace_mtime <= 5
    )
    checks["typed_image_attachments"] = {
        "ok": typed_images_ok or typed_images_pending,
        "pending": typed_images_pending,
        "image_events": len(image_events),
        "image_command_results": len(image_results),
        "image_client_commands": image_client_count,
        "image_dispatches": len(image_dispatches),
    }
    if typed_images_pending:
        warnings.append("typed image command result is still being recorded")
    elif not typed_images_ok:
        fatal.append("image client did not produce a typed image attachment")

    combined_progress = "\n".join(
        json.dumps(row, ensure_ascii=False, default=str) for row in trace_rows[-200:]
    ).lower()
    benchmark_ok, benchmark_detail = _benchmark_progress(
        benchmark,
        combined_progress,
        run_dir,
    )
    checks["benchmark_progress"] = {
        "ok": benchmark_ok or completed or request is None,
        "detail": benchmark_detail,
    }
    if request is not None and not benchmark_ok and not completed:
        warnings.append(benchmark_detail)

    status = "unhealthy" if fatal else "healthy"
    if not fatal and request is None and not completed:
        status = "healthy_setup"
    payload = {
        "schema_version": 3,
        "contract": {
            "name": HEALTH_CONTRACT_NAME,
            "version": HEALTH_CONTRACT_VERSION,
        },
        "checked_at_epoch": now,
        "benchmark": benchmark,
        "run_dir": str(run_dir),
        "status": status,
        "objective_stop_required": bool(fatal),
        "fatal": fatal,
        "warnings": warnings,
        "checks": checks,
    }
    output = run_dir / "health-check-5m.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _benchmark_progress(
    benchmark: str,
    trace_text: str,
    run_dir: Path,
) -> tuple[bool, str]:
    descriptor = benchmark_for_health_name(benchmark)
    adapter = descriptor.health
    lifecycle = (
        run_dir / adapter.lifecycle_directory
        if adapter.lifecycle_directory is not None
        else None
    )
    progress = any(marker in trace_text for marker in adapter.progress_markers)
    if lifecycle is not None:
        progress = progress or lifecycle.is_dir()
    return (
        progress,
        f"{descriptor.display_name} has not yet recorded benchmark progress",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _gzip_json_values(directory: Path) -> list[dict[str, Any]]:
    candidates = sorted(directory.glob("*.json.gz")) if directory.is_dir() else []
    values: list[dict[str, Any]] = []
    for path in candidates[-256:]:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _extract_commands(values: Iterable[Any]) -> list[str]:
    commands: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            command = value.get("command")
            if isinstance(command, str):
                commands.append(command)
            arguments = value.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    visit(parsed)
            for key, item in value.items():
                if key not in {"command", "arguments"}:
                    visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    return commands


def _is_active_legacy_agent_command(command: str) -> bool:
    argv = _standalone_command_words(command)
    return bool(
        argv
        and argv[0] == "agent"
        and len(argv) > 1
        and argv[1] in {"spawn", "report", "status", "submit", "image", "user"}
    )


def _classify_direct_client_attempts(
    trace_rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], set[str]]:
    recovery_pairs = _successful_client_pairs(trace_rows)
    used_pairs: set[int] = set()
    recovered: list[str] = []
    invalid: list[str] = []
    observed_commands: set[str] = set()
    for index, row in enumerate(trace_rows):
        if row.get("event") != "terminal_command":
            continue
        command = row.get("command")
        if not isinstance(command, str):
            continue
        argv = _standalone_command_words(command)
        if not argv or argv[0] not in EXPECTED_CLIENT_PATHS:
            continue
        observed_commands.add(command)
        script_path = argv[0]
        arguments = tuple(argv[1:])
        if not _failed_client_attempt_has_no_dispatch(
            command,
            row,
            trace_rows,
        ):
            invalid.append(command)
            continue
        matching_pair = next(
            (
                pair_index
                for pair_index, pair in enumerate(recovery_pairs)
                if pair_index not in used_pairs
                and pair[0] > index
                and pair[2] == script_path
                and pair[3] == arguments
            ),
            None,
        )
        if matching_pair is None:
            invalid.append(command)
            continue
        used_pairs.add(matching_pair)
        recovered.append(command)
    return recovered, invalid, observed_commands


def _successful_client_pairs(
    trace_rows: list[dict[str, Any]],
) -> list[tuple[int, int, str, tuple[str, ...]]]:
    pairs: list[tuple[int, int, str, tuple[str, ...]]] = []
    used_dispatches: set[int] = set()
    for executed_index, row in enumerate(trace_rows):
        if row.get("event") != "control_client_executed" or row.get("exit_code") != 0:
            continue
        command = row.get("command")
        script_path = row.get("script_path")
        if not isinstance(command, str) or script_path not in EXPECTED_CLIENT_PATHS:
            continue
        argv = _standalone_command_words(command)
        if not argv or len(argv) < 2 or argv[0] != "python" or argv[1] != script_path:
            continue
        dispatched_index = next(
            (
                index
                for index, dispatched in enumerate(trace_rows)
                if index > executed_index
                and index not in used_dispatches
                and dispatched.get("event") == "control_client_dispatched"
                and dispatched.get("command") == command
                and dispatched.get("script_path") == script_path
                and dispatched.get("exit_code") == 0
            ),
            None,
        )
        if dispatched_index is None:
            continue
        used_dispatches.add(dispatched_index)
        pairs.append(
            (
                executed_index,
                dispatched_index,
                script_path,
                tuple(argv[2:]),
            )
        )
    return pairs


def _failed_client_attempt_has_no_dispatch(
    command: str,
    terminal_row: Mapping[str, Any],
    trace_rows: list[dict[str, Any]],
) -> bool:
    if (
        not isinstance(terminal_row.get("exit_code"), int)
        or terminal_row.get("exit_code") == 0
    ):
        return False
    matching_results = [
        row
        for row in trace_rows
        if row.get("event") == "command_result" and row.get("command") == command
    ]
    if any(
        not isinstance(row.get("exit_code"), int)
        or row.get("exit_code") == 0
        or row.get("attachments")
        or row.get("completed") is True
        or row.get("inspected") is True
        for row in matching_results
    ):
        return False
    return not any(
        row.get("command") == command
        and row.get("event") in {"control_client_executed", "control_client_dispatched"}
        for row in trace_rows
    )


def _standalone_command_words(command: str) -> list[str] | None:
    if "\n" in command or "\r" in command:
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError:
        return None
    if any(
        token and all(character in ";&|()<>" for character in token) for token in tokens
    ):
        return None
    return tokens


def _python_script_argument(argv: list[str]) -> str | None:
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return argv[index + 1] if index + 1 < len(argv) else None
        if (
            argument == "-"
            or argument == "-c"
            or argument.startswith("-c")
            or argument == "-m"
            or argument.startswith("-m")
        ):
            return None
        if argument in {"-W", "-X"}:
            index += 2
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument
    return None


def _json_object(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _is_compaction_request(request: Mapping[str, Any]) -> bool:
    text = "\n".join(_string_values(request)).lower()
    return "concise" in text and "handoff" in text and "compaction" in text


def _is_final_specialist_report_request(request: Mapping[str, Any]) -> bool:
    text = "\n".join(_string_values(request)).lower()
    normalized = text.translate(str.maketrans("", "", "`'\""))
    return bool(
        "json object" in normalized
        and "status" in normalized
        and "report" in normalized
        and "unresolved" in normalized
    )


def _is_direct_report(value: Mapping[str, Any] | None) -> bool:
    if value is None:
        return False
    direct_native = bool(
        set(value) == {"status", "report", "unresolved"}
        and _is_native_report_payload(value)
    )
    if direct_native:
        return True
    if value.get("role") not in SPECIALIST_ROLES:
        return False
    report = value.get("report")
    native_shaped = bool(
        value.get("status") in {"success", "forced"}
        and _is_nonnegative_integer(value.get("used_steps"))
        and _is_positive_integer(value.get("allocated_steps"))
        and int(value["used_steps"]) <= int(value["allocated_steps"])
        and isinstance(report, Mapping)
        and _is_native_report_payload(report, role=str(value["role"]))
        and isinstance(value.get("global_budget"), Mapping)
    )
    legacy_shaped = bool(
        value is not None
        and isinstance(value.get("status"), str)
        and isinstance(report, str)
        and isinstance(value.get("summary"), str)
        and isinstance(value.get("unresolved"), list)
        and isinstance(value.get("report_path"), str)
        and isinstance(value.get("usage"), Mapping)
    )
    return native_shaped or legacy_shaped


def _is_native_report_payload(
    value: Mapping[str, Any],
    *,
    role: str | None = None,
) -> bool:
    status = value.get("status")
    report = value.get("report")
    unresolved = value.get("unresolved")
    valid_statuses = (
        NATIVE_REPORT_STATUSES if role is None else ROLE_REPORT_STATUSES.get(role, ())
    )
    return bool(
        status in valid_statuses
        and isinstance(report, str)
        and bool(report.strip())
        and len(report) <= MAX_REPORT_CHARACTERS
        and len(report.encode("utf-8")) <= MAX_REPORT_BYTES
        and isinstance(unresolved, list)
        and len(unresolved) <= MAX_UNRESOLVED_ITEMS
        and all(
            isinstance(item, str)
            and bool(item.strip())
            and len(item) <= MAX_UNRESOLVED_ITEM_CHARACTERS
            for item in unresolved
        )
    )


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _direct_report_role(value: Mapping[str, Any]) -> str:
    role = value.get("role")
    if role in SPECIALIST_ROLES:
        return str(role)
    return {
        "complete": "binder",
        "accept": "critic",
        "concerns": "critic",
        "ready": "verifier",
        "repair": "verifier",
        "unknown": "verifier",
    }.get(str(value.get("status")), "native")


def _valid_prior_context_event(row: Mapping[str, Any], role: str) -> bool:
    prior_roles = row.get("prior_roles")
    reports = row.get("prior_reports")
    return bool(
        _valid_prior_roles(role, prior_roles)
        and isinstance(row.get("phase_id"), str)
        and row.get("phase_id")
        and isinstance(reports, Mapping)
        and list(reports) == list(prior_roles)
        and all(isinstance(reports[prior_role], Mapping) for prior_role in prior_roles)
    )


def _valid_prior_roles(role: str, prior_roles: Any) -> bool:
    if not isinstance(prior_roles, list) or not all(
        isinstance(prior_role, str) for prior_role in prior_roles
    ):
        return False
    actual = tuple(prior_roles)
    allowed = ALLOWED_PRIOR_ROLES[role]
    required = REQUIRED_PRIOR_ROLES[role]
    return bool(
        actual == tuple(prior_role for prior_role in allowed if prior_role in actual)
        and all(prior_role in actual for prior_role in required)
    )


def _prior_context_failure(
    row: Mapping[str, Any] | None,
    role: str,
    invocation_id: str,
) -> dict[str, Any]:
    required_roles = list(REQUIRED_PRIOR_ROLES[role])
    allowed_roles = list(ALLOWED_PRIOR_ROLES[role])
    reports = None if row is None else row.get("prior_reports")
    actual_roles = None if row is None else row.get("prior_roles")
    return {
        "role": role,
        "invocation_id": invocation_id,
        "required_prior_roles": required_roles,
        "allowed_prior_roles": allowed_roles,
        "actual_prior_roles": actual_roles,
        "prior_reports_valid": bool(
            row is not None
            and _valid_prior_roles(role, actual_roles)
            and isinstance(reports, Mapping)
            and list(reports) == actual_roles
            and all(
                isinstance(reports[prior_role], Mapping) for prior_role in actual_roles
            )
        ),
    }


def _captured_prior_injection_requests(
    requests: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    captured: dict[tuple[str, str], bool] = {}
    for request in requests:
        for text in _string_values(request):
            for role in PRIOR_AWARE_ROLES:
                if re.search(rf"^Role:\s*{role}\s*$", text, re.MULTILINE) is None:
                    continue
                output_pattern = re.compile(
                    rf"^Output Markdown:[^\S\r\n]+"
                    rf"/(?:workspace|home/user)/\.agent/reports/"
                    rf"(?P<invocation_id>{role}-[A-Za-z0-9_-]+)\.md"
                    rf"[^\S\r\n]*$",
                    re.MULTILINE,
                )
                for match in output_pattern.finditer(text):
                    key = (role, match.group("invocation_id"))
                    prior_roles = _prior_roles_in_prompt(text)
                    valid = _valid_prior_roles(role, list(prior_roles)) and all(
                        _has_injected_report_section(text, prior_role)
                        for prior_role in prior_roles
                    )
                    captured[key] = captured.get(key, False) or valid
    return captured


def _captured_inline_prior_injection_requests(
    requests: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    captured: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        text = "\n".join(_string_values(request))
        if "Request file:" in text or "Output Markdown:" in text:
            continue
        role = _specialist_role_from_request(request, manifest=manifest, text=text)
        if role not in PRIOR_AWARE_ROLES or "Main Actor request:" not in text:
            continue
        prior_roles = _prior_roles_in_prompt(text)
        captured.setdefault(role, []).append(
            {
                "prior_roles": prior_roles,
                "valid": _valid_prior_roles(role, list(prior_roles))
                and all(
                    _has_inline_report_section(text, prior_role)
                    for prior_role in prior_roles
                ),
            }
        )
    return captured


def _prior_roles_in_prompt(text: str) -> tuple[str, ...]:
    return tuple(
        prior_role
        for prior_role in ("binder", "critic")
        if (
            f"Latest {prior_role.title()} report:" in text
            or f"Latest validated {prior_role.title()} report:" in text
        )
    )


def _specialist_role_from_request(
    request: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    text: str | None = None,
) -> str | None:
    extra_body = request.get("extra_body")
    session_id = (
        extra_body.get("session_id") if isinstance(extra_body, Mapping) else None
    )
    if isinstance(session_id, str):
        session_match = re.match(
            r"^osworld-cli-(binder|critic|verifier)-",
            session_id,
        )
        if session_match is not None:
            return session_match.group(1)

    if manifest is not None:
        role = request_role_from_prompt_hashes(manifest, request)
        if role in SPECIALIST_ROLES:
            return role

    if text is None:
        text = "\n".join(_string_values(request))
    for role in SPECIALIST_ROLES:
        if re.search(rf"^Role:\s*{role}\s*$", text, re.MULTILINE | re.IGNORECASE):
            return role
    return None


def _has_inline_report_section(text: str, prior_role: str) -> bool:
    marker = f"Latest {prior_role.title()} report:"
    marker_index = text.find(marker)
    if marker_index < 0:
        marker = f"Latest validated {prior_role.title()} report:"
        marker_index = text.find(marker)
    if marker_index < 0:
        return False
    section = text[marker_index + len(marker) :]
    boundaries = [
        section.find(next_marker)
        for next_marker in (
            "\n\nLatest Binder report:",
            "\n\nLatest Critic report:",
            "\n\nLatest Verifier report:",
            "\n\nLatest validated ",
            "\n\nInspect the live canonical task state now.",
            "\n\nWhen finished,",
        )
        if section.find(next_marker) >= 0
    ]
    if boundaries:
        section = section[: min(boundaries)]
    required_markers = ("Status:", "Unresolved:", "Report body:")
    marker_positions = [section.find(required) for required in required_markers]
    if any(position < 0 for position in marker_positions):
        return False
    if marker_positions != sorted(marker_positions):
        return False
    report_body = section[marker_positions[-1] + len(required_markers[-1]) :]
    return bool(report_body.strip())


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_values(item)


def _has_injected_report_section(text: str, prior_role: str) -> bool:
    marker = f"Latest validated {prior_role.title()} report:"
    marker_index = text.find(marker)
    if marker_index < 0:
        return False
    section = text[marker_index + len(marker) :]
    boundaries = [
        section.find(next_marker)
        for next_marker in (
            "\n\nLatest validated ",
            "\n\nInspect the live canonical task state now.",
            "\n\nWrite both output files",
        )
        if section.find(next_marker) >= 0
    ]
    if boundaries:
        section = section[: min(boundaries)]
    required_markers = (
        "Body provenance:",
        "Envelope provenance:",
        "Envelope meaning:",
        "Report body:",
    )
    marker_positions = [section.find(required) for required in required_markers]
    if any(position < 0 for position in marker_positions) or marker_positions != sorted(
        marker_positions
    ):
        return False
    report_marker = required_markers[-1]
    report_body = section[marker_positions[-1] + len(report_marker) :]
    return bool(report_body.strip())


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _event_age_seconds(row: Mapping[str, Any], now: float) -> float:
    timestamp = row.get("timestamp")
    if not isinstance(timestamp, str):
        return float("inf")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    if parsed.tzinfo is None:
        return float("inf")
    return max(0.0, now - parsed.timestamp())


def _expect(
    checks: dict[str, dict[str, Any]],
    fatal: list[str],
    name: str,
    ok: bool,
    actual: Any,
    expected: Any,
) -> None:
    checks[name] = {
        "ok": ok,
        "actual": actual,
        "expected": expected,
    }
    if not ok:
        fatal.append(f"{name}: expected {expected!r}, got {actual!r}")


def main() -> None:
    args = parse_args()
    payload = check_run_health(
        Path(args.run_dir).expanduser().resolve(),
        benchmark=args.benchmark,
        expected_model=args.expected_model,
        expected_reasoning=args.expected_reasoning,
        expected_infrastructure=args.expected_infrastructure,
        expected_agent_topology=args.expected_agent_topology,
        pid=args.pid,
        process_log=(
            None
            if args.process_log is None
            else Path(args.process_log).expanduser().resolve()
        ),
        max_stale_seconds=args.max_stale_seconds,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(1 if payload["objective_stop_required"] else 0)


if __name__ == "__main__":
    main()

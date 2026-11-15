from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from cuawright.config.prompts import (
    BenchmarkOverlay,
    OSWORLD_OVERLAY,
    PriorReportContent,
    ROLE_INSTRUCTIONS,
    compose_system_prompt,
    specialist_user_prompt,
)
from cuawright.environments.benchcad.terminal_prompts import (
    BENCHCAD_TERMINAL_OVERLAY,
)
from cuawright.environments.cadgenbench.terminal_prompts import (
    CADGENBENCH_TERMINAL_OVERLAY,
)
from cuawright.environments.control_clients import CLIENT_SHA256
from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA
from scripts.python.check_terminal_run_health import (
    _gzip_json_values,
    _read_jsonl,
    _specialist_role_from_request,
    check_run_health,
)


def _generated_provider_request(
    role: str,
    overlay: BenchmarkOverlay,
    user_prompt: str,
    *,
    session_id: str | None,
) -> dict:
    extra_body = {
        "tier": "base",
        "strict_session": True,
        "infrastructure": "aiservices",
    }
    if session_id is not None:
        extra_body["session_id"] = session_id
    return {
        "model": "gpt-5.5",
        "input": [
            {"role": "system", "content": compose_system_prompt(role, overlay)},
            {"role": "user", "content": user_prompt},
        ],
        "tools": [RUN_COMMAND_TOOL_SCHEMA],
        "parallel_tool_calls": False,
        "reasoning": {"effort": "xhigh", "summary": "auto"},
        "include": ["reasoning.encrypted_content"],
        "store": False,
        "max_output_tokens": 32768,
        "extra_body": extra_body,
    }


def _check_odysseys_control_trace(rows: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        (run_dir / "requests").mkdir()
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "model": "gpt-5.5",
                    "reasoning_effort": "xhigh",
                    "model_interface": "terminal-only",
                    "terminal_control": {
                        "protocol": "python-clients-v1",
                        "install_root": "/opt/cuawright-tools",
                        "clients": dict(CLIENT_SHA256),
                    },
                }
            ),
            encoding="utf-8",
        )
        with gzip.open(
            run_dir / "requests" / "step-0001.json.gz",
            "wt",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "tools": [{"name": "run_command"}],
                    "extra_body": {
                        "session_id": "session",
                        "strict_session": True,
                        "infrastructure": "aiservices",
                    },
                },
                handle,
            )
        (run_dir / "trace.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return check_run_health(
            run_dir,
            benchmark="odysseys",
            expected_model="gpt-5.5",
            expected_reasoning="xhigh",
        )


def _successful_image_client_rows(command: str) -> list[dict]:
    return [
        {
            "event": "control_client_executed",
            "role": "actor",
            "command": command,
            "python_executable": "python",
            "script_path": "/opt/cuawright-tools/image_show.py",
            "operation": "image_show",
            "exit_code": 0,
            "stdout_size_bytes": 100,
            "stderr_size_bytes": 0,
        },
        {
            "event": "image_attached",
            "source_path": "/workspace/frame.png",
        },
        {
            "event": "control_client_dispatched",
            "role": "actor",
            "command": command,
            "script_path": "/opt/cuawright-tools/image_show.py",
            "operation": "image_show",
            "exit_code": 0,
            "completed": False,
            "attachment_count": 1,
            "direct_report_returned": False,
        },
        {
            "event": "command_result",
            "command": command,
            "exit_code": 0,
            "attachments": [
                {
                    "kind": "image",
                    "media_type": "image/png",
                    "source_path": "/workspace/frame.png",
                    "size_bytes": 100,
                }
            ],
            "completed": False,
            "inspected": False,
        },
    ]


class TerminalHealthTests(unittest.TestCase):
    def test_jsonl_uses_lf_framing_for_unicode_line_separators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            value = "next\u0085line\u2028paragraph\u2029separator"
            trace.write_text(
                json.dumps({"event": "model_turn", "text": value}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _read_jsonl(trace),
                [{"event": "model_turn", "text": value}],
            )

    def test_jsonl_preserves_complete_records_before_partial_utf8_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            complete = {"event": "terminal_command", "command": "echo ok"}
            trace.write_bytes(
                json.dumps(complete).encode("utf-8")
                + b"\n"
                + b'{"event":"partial","text":"\xe2\x80'
            )

            self.assertEqual(_read_jsonl(trace), [complete])

    def test_inflight_gzip_eof_and_utf8_tails_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            complete = {"model": "gpt-5.5"}
            (directory / "step-0001.json.gz").write_bytes(
                gzip.compress(json.dumps(complete).encode())
            )
            (directory / "step-0002.json.gz").write_bytes(
                gzip.compress(b'{"inflight": true}')[:12]
            )
            (directory / "step-0003.json.gz").write_bytes(
                gzip.compress(b'{"partial_utf8":"\xe2\x80"}')
            )

            self.assertEqual(_gzip_json_values(directory), [complete])

    def test_healthy_active_phyagi_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                        "terminal_control": {
                            "protocol": "python-clients-v1",
                            "install_root": "/opt/cuawright-tools",
                            "clients": dict(CLIENT_SHA256),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "trapi",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": (
                                "python /opt/cuawright-tools/binder.py "
                                "--request /workspace/.agent/requests/binder-01.md"
                            ),
                        },
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": (
                                "python /opt/cuawright-tools/binder.py "
                                "--request /workspace/.agent/requests/binder-01.md"
                            ),
                            "python_executable": "python",
                            "script_path": "/opt/cuawright-tools/binder.py",
                            "operation": "binder",
                            "exit_code": 0,
                            "stdout_size_bytes": 120,
                            "stderr_size_bytes": 0,
                        },
                        {
                            "event": "specialist_context_prepared",
                            "role": "binder",
                            "invocation_id": "binder-01",
                            "phase_id": "phase-1",
                            "prior_roles": [],
                            "prior_reports": {},
                        },
                        {
                            "event": "specialist_completed",
                            "role": "binder",
                            "invocation_id": "binder-01",
                            "request_artifact": "specialists/binder-01/request.txt",
                            "report_artifact": "specialists/binder-01/report.json",
                            "report_sha256": "a" * 64,
                            "report_size_bytes": 18,
                        },
                        {
                            "event": "control_client_dispatched",
                            "role": "actor",
                            "command": (
                                "python /opt/cuawright-tools/binder.py "
                                "--request /workspace/.agent/requests/binder-01.md"
                            ),
                            "script_path": "/opt/cuawright-tools/binder.py",
                            "operation": "binder",
                            "exit_code": 0,
                            "completed": False,
                            "attachment_count": 0,
                            "direct_report_returned": True,
                            "report_artifact": "specialists/binder-01/report.json",
                        },
                        {
                            "event": "command_result",
                            "stdout": json.dumps(
                                {
                                    "role": "binder",
                                    "status": "success",
                                    "allocated_steps": 8,
                                    "used_steps": 4,
                                    "report": {
                                        "status": "complete",
                                        "report": "The task is bound.",
                                        "unresolved": [],
                                    },
                                    "global_budget": {"remaining_steps": 196},
                                    "report_artifact": (
                                        "specialists/binder-01/report.json"
                                    ),
                                }
                            ),
                            "attachments": [],
                        },
                        {
                            "event": "terminal_command",
                            "command": (
                                "python /opt/cuawright-tools/image_show.py "
                                "--path /workspace/frame.png"
                            ),
                        },
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": (
                                "python /opt/cuawright-tools/image_show.py "
                                "--path /workspace/frame.png"
                            ),
                            "python_executable": "python",
                            "script_path": "/opt/cuawright-tools/image_show.py",
                            "operation": "image_show",
                            "exit_code": 0,
                            "stdout_size_bytes": 100,
                            "stderr_size_bytes": 0,
                        },
                        {
                            "event": "image_attached",
                            "source_path": "/workspace/frame.png",
                        },
                        {
                            "event": "control_client_dispatched",
                            "role": "actor",
                            "command": (
                                "python /opt/cuawright-tools/image_show.py "
                                "--path /workspace/frame.png"
                            ),
                            "script_path": ("/opt/cuawright-tools/image_show.py"),
                            "operation": "image_show",
                            "exit_code": 0,
                            "completed": False,
                            "attachment_count": 1,
                            "direct_report_returned": False,
                            "report_path": None,
                        },
                        {
                            "event": "command_result",
                            "stdout": "{}",
                            "attachments": [
                                {
                                    "kind": "image",
                                    "media_type": "image/png",
                                    "source_path": "/workspace/frame.png",
                                    "size_bytes": 100,
                                }
                            ],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
                expected_infrastructure="trapi",
            )

            self.assertEqual(result["status"], "healthy")
            self.assertFalse(result["objective_stop_required"])
            self.assertTrue((run_dir / "health-check-5m.json").is_file())
            self.assertTrue(result["checks"]["manifest_control_clients"]["ok"])
            self.assertEqual(
                result["checks"]["control_protocol"]["exact_client_paths"],
                [
                    "/opt/cuawright-tools/binder.py",
                    "/opt/cuawright-tools/image_show.py",
                ],
            )
            self.assertTrue(result["checks"]["direct_report_return"]["ok"])
            self.assertTrue(result["checks"]["typed_image_attachments"]["ok"])

    def test_anthropic_run_uses_manifest_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "evaluation").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "claude-fable-5-1",
                        "reasoning_effort": "max",
                        "model_provider": "anthropic",
                        "model_infrastructure": "anthropic-foundry",
                        "model_interface": "terminal-only",
                        "agent_topology": "linear",
                        "terminal_control": {
                            "protocol": "python-clients-v1",
                            "install_root": "/opt/cuawright-tools",
                            "clients": dict(CLIENT_SHA256),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump({"tools": [{"name": "run_command"}]}, handle)
            (run_dir / "evaluation" / "result.json").write_text(
                json.dumps({"score": 1.0}),
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="claude-fable-5-1",
                expected_reasoning="max",
                expected_infrastructure="anthropic-foundry",
                expected_agent_topology="linear",
            )

            self.assertTrue(result["checks"]["strict_session"]["ok"])
            self.assertTrue(result["checks"]["infrastructure"]["ok"])
            self.assertTrue(result["checks"]["session_id"]["ok"])
            self.assertNotIn(
                "strict_session: expected True, got None",
                result["fatal"],
            )

    def test_active_legacy_agent_command_is_objectively_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "input": [
                            {
                                "type": "function_call",
                                "arguments": json.dumps(
                                    {"command": "agent spawn binder"}
                                ),
                            }
                        ],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": "agent spawn binder",
                        },
                        {
                            "event": "legacy_control_rejected",
                            "role": "actor",
                            "command": "agent spawn binder",
                            "reason": "legacy_agent_protocol_removed",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertEqual(result["status"], "unhealthy")
            self.assertTrue(result["objective_stop_required"])
            self.assertTrue(any("legacy" in item for item in result["fatal"]))

    def test_dependent_specialist_requires_prior_report_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": (
                                "python /opt/cuawright-tools/critic.py "
                                "--request /workspace/.agent/requests/critic-01.md"
                            ),
                        },
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": (
                                "python /opt/cuawright-tools/critic.py "
                                "--request /workspace/.agent/requests/critic-01.md"
                            ),
                            "python_executable": "python",
                            "script_path": "/opt/cuawright-tools/critic.py",
                            "operation": "critic",
                            "exit_code": 0,
                            "stdout_size_bytes": 120,
                            "stderr_size_bytes": 0,
                        },
                        {
                            "event": "specialist_context_prepared",
                            "role": "critic",
                            "invocation_id": "critic-01",
                            "phase_id": "phase-1",
                            "prior_roles": ["binder"],
                            "prior_reports": {
                                "binder": {
                                    "body_path": (
                                        "/workspace/.agent/reports/binder-01.md"
                                    ),
                                    "envelope_path": (
                                        "/workspace/.agent/reports/"
                                        "binder-01.envelope.json"
                                    ),
                                    "body_sha256": "a" * 64,
                                    "envelope_sha256": "b" * 64,
                                }
                            },
                        },
                        {
                            "event": "specialist_completed",
                            "role": "critic",
                            "invocation_id": "critic-01",
                            "direct_report_returned": True,
                            "report_size_bytes": 9,
                        },
                        {
                            "event": "control_client_dispatched",
                            "role": "actor",
                            "command": (
                                "python /opt/cuawright-tools/critic.py "
                                "--request /workspace/.agent/requests/critic-01.md"
                            ),
                            "script_path": "/opt/cuawright-tools/critic.py",
                            "operation": "critic",
                            "exit_code": 0,
                            "completed": False,
                            "attachment_count": 0,
                            "direct_report_returned": True,
                            "report_path": ("/workspace/.agent/reports/critic-01.md"),
                        },
                        {
                            "event": "command_result",
                            "stdout": json.dumps(
                                {
                                    "role": "critic",
                                    "status": "accept",
                                    "report": "Accepted.",
                                    "summary": "Accepted",
                                    "unresolved": [],
                                    "report_path": (
                                        "/workspace/.agent/reports/critic-01.md"
                                    ),
                                    "usage": {
                                        "invocation_steps": 3,
                                        "shared_remaining_steps": 190,
                                    },
                                }
                            ),
                            "attachments": [],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertTrue(result["objective_stop_required"])
            self.assertFalse(result["checks"]["prior_report_injection"]["ok"])
            self.assertEqual(
                result["checks"]["prior_report_injection"][
                    "context_evidence_invocations"
                ],
                {"critic": ["critic-01"]},
            )
            self.assertEqual(
                result["checks"]["prior_report_injection"]["missing_request_evidence"],
                ["critic-01"],
            )

            with gzip.open(
                run_dir / "requests" / "step-0002.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "input": [
                            {
                                "role": "user",
                                "content": (
                                    "Role: critic\n"
                                    "Request file: "
                                    "/workspace/.agent/requests/critic.md\n"
                                    "Output Markdown: "
                                    "/home/user/.agent/reports/critic-01.md\n"
                                    "Latest validated Binder report:\n"
                                    "Body provenance: "
                                    "/workspace/.agent/reports/binder-01.md\n"
                                    "Envelope provenance: "
                                    "/workspace/.agent/reports/"
                                    "binder-01.envelope.json\n"
                                    "Envelope meaning:\n"
                                    '{"status":"complete"}\n'
                                    "Report body:\nBound work."
                                ),
                            }
                        ],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )

            healthy = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(healthy["objective_stop_required"])
            self.assertTrue(healthy["checks"]["prior_report_injection"]["ok"])
            self.assertEqual(
                healthy["checks"]["prior_report_injection"]["request_evidence_roles"],
                ["critic"],
            )
            self.assertEqual(
                healthy["checks"]["prior_report_injection"][
                    "request_evidence_invocations"
                ],
                {"critic": ["critic-01"]},
            )

            with gzip.open(
                run_dir / "requests" / "step-0003.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "input": [
                            {
                                "role": "user",
                                "content": (
                                    "Role: verifier\n"
                                    "Request file: "
                                    "/workspace/.agent/requests/verifier-01.md\n"
                                    "Output Markdown: "
                                    "/workspace/.agent/reports/verifier-01.md\n"
                                    "Latest validated Binder report:\n"
                                    "Body provenance: "
                                    "/workspace/.agent/reports/binder-01.md\n"
                                    "Envelope provenance: "
                                    "/workspace/.agent/reports/"
                                    "binder-01.envelope.json\n"
                                    "Envelope meaning:\n"
                                    '{"status":"complete"}\n'
                                    "Report body:\nBound work.\n\n"
                                    "Latest validated Critic report:\n"
                                    "Body provenance: "
                                    "/workspace/.agent/reports/critic-01.md\n"
                                    "Envelope provenance: "
                                    "/workspace/.agent/reports/"
                                    "critic-01.envelope.json\n"
                                    "Envelope meaning:\n"
                                    '{"status":"accept"}\n'
                                    "Report body:\nAccepted."
                                ),
                            }
                        ],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            verifier_command = (
                "python /opt/cuawright-tools/verifier.py "
                "--request /workspace/.agent/requests/verifier-01.md"
            )
            with (run_dir / "trace.jsonl").open("a", encoding="utf-8") as handle:
                for row in (
                    {
                        "event": "terminal_command",
                        "command": verifier_command,
                    },
                    {
                        "event": "control_client_executed",
                        "role": "actor",
                        "command": verifier_command,
                        "python_executable": "python",
                        "script_path": "/opt/cuawright-tools/verifier.py",
                        "operation": "verifier",
                        "exit_code": 0,
                        "stdout_size_bytes": 120,
                        "stderr_size_bytes": 0,
                    },
                    {
                        "event": "specialist_context_prepared",
                        "role": "verifier",
                        "invocation_id": "verifier-01",
                        "phase_id": "phase-1",
                        "prior_roles": ["binder", "critic"],
                        "prior_reports": {
                            "binder": {
                                "body_path": ("/workspace/.agent/reports/binder-01.md"),
                                "envelope_path": (
                                    "/workspace/.agent/reports/"
                                    "binder-01.envelope.json"
                                ),
                                "body_sha256": "a" * 64,
                                "envelope_sha256": "b" * 64,
                            },
                            "critic": {
                                "body_path": ("/workspace/.agent/reports/critic-01.md"),
                                "envelope_path": (
                                    "/workspace/.agent/reports/"
                                    "critic-01.envelope.json"
                                ),
                                "body_sha256": "c" * 64,
                                "envelope_sha256": "d" * 64,
                            },
                        },
                    },
                    {
                        "event": "specialist_completed",
                        "role": "verifier",
                        "invocation_id": "verifier-01",
                        "direct_report_returned": True,
                        "report_size_bytes": 6,
                    },
                    {
                        "event": "control_client_dispatched",
                        "role": "actor",
                        "command": verifier_command,
                        "script_path": "/opt/cuawright-tools/verifier.py",
                        "operation": "verifier",
                        "exit_code": 0,
                        "completed": False,
                        "attachment_count": 0,
                        "direct_report_returned": True,
                        "report_path": ("/workspace/.agent/reports/verifier-01.md"),
                    },
                    {
                        "event": "command_result",
                        "stdout": json.dumps(
                            {
                                "role": "verifier",
                                "status": "ready",
                                "report": "Ready.",
                                "summary": "Ready",
                                "unresolved": [],
                                "report_path": (
                                    "/workspace/.agent/reports/verifier-01.md"
                                ),
                                "usage": {
                                    "invocation_steps": 3,
                                    "shared_remaining_steps": 187,
                                },
                            }
                        ),
                        "attachments": [],
                    },
                ):
                    handle.write(json.dumps(row) + "\n")

            verified = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(verified["objective_stop_required"])
            self.assertEqual(
                verified["checks"]["prior_report_injection"]["request_evidence_roles"],
                ["critic", "verifier"],
            )
            self.assertEqual(
                verified["checks"]["prior_report_injection"][
                    "request_evidence_invocations"
                ],
                {
                    "critic": ["critic-01"],
                    "verifier": ["verifier-01"],
                },
            )

    def test_dependent_specialist_is_healthy_before_provider_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": "echo preparing critic",
                        },
                        {
                            "event": "specialist_context_prepared",
                            "role": "critic",
                            "invocation_id": "critic-01",
                            "phase_id": "phase-1",
                            "prior_roles": ["binder"],
                            "prior_reports": {
                                "binder": {
                                    "body_path": (
                                        "/workspace/.agent/reports/binder-01.md"
                                    ),
                                    "envelope_path": (
                                        "/workspace/.agent/reports/"
                                        "binder-01.envelope.json"
                                    ),
                                    "body_sha256": "a" * 64,
                                    "envelope_sha256": "b" * 64,
                                }
                            },
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertEqual(result["status"], "healthy")
            self.assertFalse(result["objective_stop_required"])
            self.assertTrue(result["checks"]["prior_report_injection"]["ok"])
            self.assertEqual(
                result["checks"]["prior_report_injection"]["completed_invocations"],
                [],
            )
            self.assertEqual(
                result["checks"]["prior_report_injection"]["request_race_invocations"],
                ["critic-01"],
            )

    def test_active_specialist_dispatch_can_remain_pending_beyond_trace_grace(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            command = (
                "python /opt/cuawright-tools/binder.py "
                "--request /workspace/.agent/requests/binder.md"
            )
            nested_command = (
                "python /opt/cuawright-tools/binder.py "
                "--request /workspace/.agent/requests/nested.md"
            )
            trace_path = run_dir / "trace.jsonl"
            rows = [
                {"event": "terminal_command", "command": command},
                {
                    "event": "control_client_executed",
                    "role": "actor",
                    "command": command,
                    "python_executable": "python",
                    "script_path": "/opt/cuawright-tools/binder.py",
                    "operation": "binder",
                    "exit_code": 0,
                    "stdout_size_bytes": 120,
                    "stderr_size_bytes": 0,
                },
                {
                    "event": "specialist_context_prepared",
                    "role": "binder",
                    "invocation_id": "binder-01",
                    "phase_id": "phase-1",
                    "prior_roles": [],
                    "prior_reports": {},
                },
                {
                    "event": "terminal_command",
                    "command": nested_command,
                },
                {
                    "event": "control_client_executed",
                    "role": "binder",
                    "command": nested_command,
                    "python_executable": "python",
                    "script_path": "/opt/cuawright-tools/binder.py",
                    "operation": "binder",
                    "exit_code": 0,
                    "stdout_size_bytes": 120,
                    "stderr_size_bytes": 0,
                },
                {
                    "event": "control_client_dispatched",
                    "role": "binder",
                    "command": nested_command,
                    "script_path": "/opt/cuawright-tools/binder.py",
                    "operation": "binder",
                    "exit_code": 2,
                    "completed": False,
                    "attachment_count": 0,
                    "direct_report_returned": False,
                    "report_path": None,
                },
            ]
            trace_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            stale = trace_path.stat().st_mtime - 30
            os.utime(trace_path, (stale, stale))

            active = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
                pid=os.getpid(),
                max_stale_seconds=1,
            )

            self.assertFalse(active["objective_stop_required"])
            self.assertTrue(active["checks"]["control_protocol"]["dispatch_pending"])
            self.assertEqual(
                active["checks"]["control_protocol"]["pending_specialist_dispatches"],
                {command: 1},
            )

            trace_path.write_text(
                "\n".join(json.dumps(row) for row in rows[:2]) + "\n",
                encoding="utf-8",
            )
            stale = trace_path.stat().st_mtime - 30
            os.utime(trace_path, (stale, stale))

            orphaned = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
                pid=os.getpid(),
                max_stale_seconds=1,
            )

            self.assertTrue(orphaned["objective_stop_required"])
            self.assertEqual(
                orphaned["checks"]["control_protocol"]["orphaned_dispatches"],
                {command: 1},
            )

    def test_report_heredoc_can_mention_client_without_becoming_an_attempt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            report_command = (
                "mkdir -p /workspace/.agent/reports\n"
                "cat > /workspace/.agent/reports/binder-01.md <<'EOF'\n"
                "Inspect with `python /opt/cuawright-tools/image_show.py "
                "--path /workspace/render.png`.\n"
                "EOF"
            )
            (run_dir / "trace.jsonl").write_text(
                json.dumps(
                    {
                        "event": "terminal_command",
                        "command": report_command,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertEqual(
                result["checks"]["control_protocol"]["invalid_client_commands"],
                [],
            )

            for ordinary_command in (
                "python worker.py /opt/cuawright-tools/status.py",
                "python -c 'print(1)' /opt/cuawright-tools/status.py",
                "python -c'print(1)' /opt/cuawright-tools/status.py",
                "python -m worker /opt/cuawright-tools/status.py",
                "python -mjson.tool /opt/cuawright-tools/status.py",
                "python - /opt/cuawright-tools/status.py",
            ):
                with self.subTest(ordinary_command=ordinary_command):
                    (run_dir / "trace.jsonl").write_text(
                        json.dumps(
                            {
                                "event": "terminal_command",
                                "command": ordinary_command,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    ordinary = check_run_health(
                        run_dir,
                        benchmark="benchcad",
                        expected_model="gpt-5.5",
                        expected_reasoning="xhigh",
                    )
                    self.assertFalse(ordinary["objective_stop_required"])
                    self.assertEqual(
                        ordinary["checks"]["control_protocol"][
                            "invalid_client_commands"
                        ],
                        [],
                    )

            for invalid_command in (
                "python3 /opt/cuawright-tools/status.py",
                "python -u /opt/cuawright-tools/status.py",
                "/opt/cuawright-tools/status.py",
                "python /opt/cuawright-tools/report.py --status blocked",
            ):
                with self.subTest(invalid_command=invalid_command):
                    (run_dir / "trace.jsonl").write_text(
                        json.dumps(
                            {
                                "event": "terminal_command",
                                "command": invalid_command,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    invalid = check_run_health(
                        run_dir,
                        benchmark="benchcad",
                        expected_model="gpt-5.5",
                        expected_reasoning="xhigh",
                    )

                    self.assertTrue(invalid["objective_stop_required"])
                    self.assertEqual(
                        invalid["checks"]["control_protocol"][
                            "invalid_client_commands"
                        ],
                        [invalid_command],
                    )

    def test_failed_control_client_syntax_recovered_exactly_is_warning(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                        "terminal_control": {
                            "protocol": "python-clients-v1",
                            "install_root": "/opt/cuawright-tools",
                            "clients": dict(CLIENT_SHA256),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            invalid_command = (
                "/opt/cuawright-tools/image_show.py " "--path /workspace/frame.png"
            )
            exact_command = f"python {invalid_command}"
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": invalid_command,
                            "exit_code": 2,
                        },
                        {
                            "event": "terminal_command",
                            "command": exact_command,
                            "exit_code": 0,
                        },
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": exact_command,
                            "python_executable": "python",
                            "script_path": ("/opt/cuawright-tools/image_show.py"),
                            "operation": "image_show",
                            "exit_code": 0,
                            "stdout_size_bytes": 100,
                            "stderr_size_bytes": 0,
                        },
                        {
                            "event": "image_attached",
                            "source_path": "/workspace/frame.png",
                        },
                        {
                            "event": "control_client_dispatched",
                            "role": "actor",
                            "command": exact_command,
                            "script_path": ("/opt/cuawright-tools/image_show.py"),
                            "operation": "image_show",
                            "exit_code": 0,
                            "completed": False,
                            "attachment_count": 1,
                            "direct_report_returned": False,
                        },
                        {
                            "event": "command_result",
                            "command": exact_command,
                            "exit_code": 0,
                            "attachments": [
                                {
                                    "kind": "image",
                                    "media_type": "image/png",
                                    "source_path": "/workspace/frame.png",
                                    "size_bytes": 100,
                                }
                            ],
                            "completed": False,
                            "inspected": False,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="odysseys",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertEqual(result["status"], "healthy")
            self.assertEqual(
                result["checks"]["control_protocol"]["invalid_client_commands"],
                [],
            )
            self.assertEqual(
                result["checks"]["control_protocol"][
                    "recovered_invalid_client_commands"
                ],
                [invalid_command],
            )
            self.assertIn(
                "non-exact control-client command failed without host dispatch "
                "and was later recovered exactly",
                result["warnings"],
            )

    def test_recovery_requires_equivalent_client_arguments(self) -> None:
        invalid_command = (
            "/opt/cuawright-tools/image_show.py --path /workspace/first.png"
        )
        different_command = (
            "python /opt/cuawright-tools/image_show.py " "--path /workspace/second.png"
        )
        result = _check_odysseys_control_trace(
            [
                {
                    "event": "terminal_command",
                    "command": invalid_command,
                    "exit_code": 2,
                },
                *_successful_image_client_rows(different_command),
            ]
        )

        self.assertTrue(result["objective_stop_required"])
        self.assertEqual(
            result["checks"]["control_protocol"]["invalid_client_commands"],
            [invalid_command],
        )
        self.assertEqual(
            result["checks"]["control_protocol"]["recovered_invalid_client_commands"],
            [],
        )

    def test_recovery_requires_successful_execution_before_dispatch(self) -> None:
        invalid_command = (
            "/opt/cuawright-tools/image_show.py --path /workspace/frame.png"
        )
        exact_command = f"python {invalid_command}"
        result = _check_odysseys_control_trace(
            [
                {
                    "event": "terminal_command",
                    "command": invalid_command,
                    "exit_code": 2,
                },
                {
                    "event": "control_client_dispatched",
                    "role": "actor",
                    "command": exact_command,
                    "script_path": "/opt/cuawright-tools/image_show.py",
                    "operation": "image_show",
                    "exit_code": 0,
                    "completed": False,
                    "attachment_count": 1,
                    "direct_report_returned": False,
                },
            ]
        )

        self.assertTrue(result["objective_stop_required"])
        self.assertEqual(
            result["checks"]["control_protocol"]["invalid_client_commands"],
            [invalid_command],
        )
        self.assertEqual(
            result["checks"]["control_protocol"]["recovered_invalid_client_commands"],
            [],
        )

    def test_one_canonical_pair_recovers_only_one_identical_attempt(
        self,
    ) -> None:
        invalid_command = (
            "/opt/cuawright-tools/image_show.py --path /workspace/frame.png"
        )
        exact_command = f"python {invalid_command}"
        result = _check_odysseys_control_trace(
            [
                {
                    "event": "terminal_command",
                    "command": invalid_command,
                    "exit_code": 2,
                },
                {
                    "event": "terminal_command",
                    "command": invalid_command,
                    "exit_code": 2,
                },
                *_successful_image_client_rows(exact_command),
            ]
        )

        self.assertTrue(result["objective_stop_required"])
        self.assertEqual(
            result["checks"]["control_protocol"]["invalid_client_commands"],
            [invalid_command],
        )
        self.assertEqual(
            result["checks"]["control_protocol"]["recovered_invalid_client_commands"],
            [invalid_command],
        )

    def test_legacy_command_detection_excludes_heredoc_bodies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            heredoc = "cat <<'EOF'\nagent spawn binder\nEOF"
            (run_dir / "trace.jsonl").write_text(
                json.dumps({"event": "terminal_command", "command": heredoc}) + "\n",
                encoding="utf-8",
            )

            report_text = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(report_text["objective_stop_required"])
            chained = "agent spawn binder"
            (run_dir / "trace.jsonl").write_text(
                json.dumps({"event": "terminal_command", "command": chained}) + "\n",
                encoding="utf-8",
            )
            active = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )
            self.assertTrue(active["objective_stop_required"])
            self.assertEqual(
                active["checks"]["control_protocol"]["legacy_commands"],
                [chained],
            )

    def test_completed_failed_specialist_does_not_leave_pending_allowance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            first_command = (
                "python /opt/cuawright-tools/binder.py "
                "--request /workspace/.agent/requests/first.md"
            )
            active_command = (
                "python /opt/cuawright-tools/binder.py "
                "--request /workspace/.agent/requests/second.md"
            )
            trace_path = run_dir / "trace.jsonl"
            trace_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": first_command,
                        },
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": first_command,
                            "python_executable": "python",
                            "script_path": "/opt/cuawright-tools/binder.py",
                            "operation": "binder",
                            "exit_code": 0,
                            "stdout_size_bytes": 120,
                            "stderr_size_bytes": 0,
                        },
                        {
                            "event": "specialist_context_prepared",
                            "role": "binder",
                            "invocation_id": "binder-01",
                            "phase_id": "phase-1",
                            "prior_roles": [],
                            "prior_reports": {},
                        },
                        {
                            "event": "control_client_dispatched",
                            "role": "actor",
                            "command": first_command,
                            "script_path": "/opt/cuawright-tools/binder.py",
                            "operation": "binder",
                            "exit_code": 2,
                            "completed": False,
                            "attachment_count": 0,
                            "direct_report_returned": False,
                            "report_path": None,
                        },
                        {
                            "event": "terminal_command",
                            "command": active_command,
                        },
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": active_command,
                            "python_executable": "python",
                            "script_path": "/opt/cuawright-tools/binder.py",
                            "operation": "binder",
                            "exit_code": 0,
                            "stdout_size_bytes": 120,
                            "stderr_size_bytes": 0,
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            stale = trace_path.stat().st_mtime - 30
            os.utime(trace_path, (stale, stale))

            result = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
                pid=os.getpid(),
                max_stale_seconds=1,
            )

            self.assertTrue(result["objective_stop_required"])
            self.assertEqual(
                result["checks"]["control_protocol"]["orphaned_dispatches"],
                {active_command: 1},
            )

    def test_active_dependent_request_without_injection_is_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "input": [
                            {
                                "role": "user",
                                "content": (
                                    "Role: critic\n"
                                    "Request file: "
                                    "/workspace/.agent/requests/critic.md\n"
                                    "Output Markdown: "
                                    "/workspace/.agent/reports/critic-01.md\n"
                                    "Main Actor request:\nReview the work."
                                ),
                            }
                        ],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "event": "terminal_command",
                            "command": "echo running critic",
                        },
                        {
                            "event": "specialist_context_prepared",
                            "role": "critic",
                            "invocation_id": "critic-01",
                            "phase_id": "phase-1",
                            "prior_roles": ["binder"],
                            "prior_reports": {
                                "binder": {
                                    "body_path": (
                                        "/workspace/.agent/reports/binder-01.md"
                                    ),
                                    "envelope_path": (
                                        "/workspace/.agent/reports/"
                                        "binder-01.envelope.json"
                                    ),
                                    "body_sha256": "a" * 64,
                                    "envelope_sha256": "b" * 64,
                                }
                            },
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertEqual(result["status"], "unhealthy")
            self.assertTrue(result["objective_stop_required"])
            self.assertEqual(
                result["checks"]["prior_report_injection"][
                    "active_request_invocations"
                ],
                ["critic-01"],
            )
            self.assertEqual(
                result["checks"]["prior_report_injection"]["missing_request_evidence"],
                ["critic-01"],
            )

    def test_wrong_tool_or_routing_is_objectively_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "read_image"}],
                        "extra_body": {
                            "strict_session": False,
                            "infrastructure": "other",
                        },
                    },
                    handle,
                )

            result = check_run_health(
                run_dir,
                benchmark="benchcad",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertEqual(result["status"], "unhealthy")
            self.assertTrue(result["objective_stop_required"])
            self.assertTrue(any("sole_tool" in item for item in result["fatal"]))

    def test_fresh_setup_without_request_is_not_killed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            run_dir.mkdir(exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                    }
                ),
                encoding="utf-8",
            )
            log = run_dir / "process.log"
            log.write_text("Starting isolated environment\n", encoding="utf-8")

            result = check_run_health(
                run_dir,
                benchmark="cadgenbench",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
                process_log=log,
            )

            self.assertEqual(result["status"], "healthy_setup")
            self.assertFalse(result["objective_stop_required"])

    def test_live_pid_is_healthy_during_slow_first_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                    }
                ),
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
                pid=os.getpid(),
                max_stale_seconds=0,
            )

            self.assertEqual(result["status"], "healthy_setup")
            self.assertFalse(result["objective_stop_required"])

    def test_tagged_no_tool_compaction_request_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0041.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "input": [
                            {
                                "role": "user",
                                "content": (
                                    "Do not call run_command. Produce a concise "
                                    "handoff for compaction."
                                ),
                            }
                        ],
                        "tools": [],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                json.dumps({"event": "compaction"}) + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertTrue(result["checks"]["sole_tool"]["compaction"])

    def test_tagged_no_tool_specialist_report_request_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0041.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "input": [
                            {
                                "role": "user",
                                "content": (
                                    "This is the final allocated specialist call. "
                                    "Do not call run_command. Return exactly one "
                                    "JSON object with only `status`, `report`, and "
                                    "`unresolved`."
                                ),
                            }
                        ],
                        "tools": [],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                json.dumps({"event": "model_turn", "role": "binder"}) + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertTrue(result["checks"]["sole_tool"]["specialist_report"])

    def test_native_direct_report_needs_no_guest_report_path_or_trace_event(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            command = (
                "python /opt/cuawright-tools/binder.py "
                '--request "Bind the task." --max-steps 8'
            )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"event": "terminal_command", "command": command},
                        {
                            "event": "command_result",
                            "role": "actor",
                            "stdout": json.dumps(
                                {
                                    "status": "complete",
                                    "report": "The task is bound.",
                                    "unresolved": [],
                                }
                            ),
                            "attachments": [],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertEqual(
                result["checks"]["direct_report_return"]["direct_report_result_count"],
                1,
            )
            self.assertEqual(
                result["checks"]["direct_report_return"]["direct_report_results"],
                {"binder": 1},
            )

    def test_inline_prior_reports_need_no_guest_control_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                        "prompt_hashes": {
                            "critic": {
                                "sha256": {
                                    "composed_system": hashlib.sha256(
                                        ROLE_INSTRUCTIONS["critic"].encode("utf-8")
                                    ).hexdigest()
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            generated_prompt = specialist_user_prompt(
                "critic",
                instruction="Complete the task.",
                request="Review the work.",
                cache_path="/workspace/.agent/cache/critic-01",
                allocated_steps=6,
                budget_status="190 calls remain.",
                prior_reports={
                    "binder": PriorReportContent(
                        status="complete",
                        report="The requirements are bound.",
                        unresolved=(),
                    )
                },
            )
            self.assertNotIn("Summary:", generated_prompt)
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "instructions": ROLE_INSTRUCTIONS["critic"],
                        "input": [
                            {
                                "role": "user",
                                "content": generated_prompt,
                            }
                        ],
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            command = (
                "python /opt/cuawright-tools/critic.py "
                '--request "Review the work." --max-steps 6'
            )
            (run_dir / "trace.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"event": "terminal_command", "command": command},
                        {
                            "event": "control_client_executed",
                            "role": "actor",
                            "command": command,
                            "python_executable": "python",
                            "script_path": "/opt/cuawright-tools/critic.py",
                            "operation": "critic",
                            "exit_code": 0,
                            "stdout_size_bytes": 100,
                            "stderr_size_bytes": 0,
                        },
                        {
                            "event": "specialist_context_prepared",
                            "role": "critic",
                            "invocation_id": "critic-01",
                            "phase_id": "phase-1",
                            "prior_roles": ["binder"],
                            "prior_reports": {
                                "binder": {
                                    "report_artifact": (
                                        "specialists/binder-01/report.json"
                                    ),
                                    "report_sha256": "a" * 64,
                                }
                            },
                        },
                        {
                            "event": "specialist_completed",
                            "role": "critic",
                            "invocation_id": "critic-01",
                            "request_artifact": "specialists/critic-01/request.txt",
                            "report_artifact": "specialists/critic-01/report.json",
                            "report_sha256": "b" * 64,
                            "report_size_bytes": 100,
                        },
                        {
                            "event": "control_client_dispatched",
                            "role": "actor",
                            "command": command,
                            "script_path": "/opt/cuawright-tools/critic.py",
                            "operation": "critic",
                            "exit_code": 0,
                            "completed": False,
                            "attachment_count": 0,
                            "direct_report_returned": True,
                            "report_artifact": "specialists/critic-01/report.json",
                        },
                        {
                            "event": "command_result",
                            "role": "actor",
                            "stdout": json.dumps(
                                {
                                    "status": "success",
                                    "role": "critic",
                                    "allocated_steps": 6,
                                    "used_steps": 3,
                                    "report": {
                                        "status": "accept",
                                        "report": "The work satisfies the brief.",
                                        "unresolved": [],
                                    },
                                    "global_budget": {"remaining_steps": 190},
                                }
                            ),
                            "attachments": [],
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertTrue(result["checks"]["prior_report_injection"]["ok"])
            self.assertEqual(
                result["checks"]["prior_report_injection"]["inline_request_roles"],
                ["critic"],
            )
            self.assertEqual(
                result["checks"]["prior_report_injection"][
                    "context_evidence_invocations"
                ],
                {"critic": ["critic-01"]},
            )
            self.assertEqual(
                result["checks"]["prior_report_injection"][
                    "request_evidence_invocations"
                ],
                {"critic": ["critic-01"]},
            )

    def test_generated_critic_prompt_without_binder_is_unhealthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                        "prompt_hashes": {
                            "critic": {
                                "sha256": {
                                    "composed_system": hashlib.sha256(
                                        ROLE_INSTRUCTIONS["critic"].encode("utf-8")
                                    ).hexdigest()
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            generated_prompt = specialist_user_prompt(
                "critic",
                instruction="Complete the task.",
                request="Review the work.",
                cache_path="/workspace/.agent/cache/critic-01",
                allocated_steps=6,
                budget_status="190 calls remain.",
                prior_reports={
                    "binder": PriorReportContent(
                        status="complete",
                        report="The requirements are bound.",
                        unresolved=(),
                    )
                },
            )
            before_report, report_and_suffix = generated_prompt.split(
                "\n\nLatest Binder report:",
                1,
            )
            _, after_report = report_and_suffix.split("\n\nWhen finished,", 1)
            generated_without_binder = (
                before_report + "\n\nWhen finished," + after_report
            )
            with gzip.open(
                run_dir / "requests" / "step-0001.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "instructions": ROLE_INSTRUCTIONS["critic"],
                        "input": [
                            {
                                "role": "user",
                                "content": generated_without_binder,
                            }
                        ],
                        "tools": [{"name": "run_command"}],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                json.dumps(
                    {
                        "event": "terminal_command",
                        "command": "echo reviewing current state",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertTrue(result["objective_stop_required"])
            self.assertFalse(result["checks"]["prior_report_injection"]["ok"])
            self.assertEqual(
                result["checks"]["prior_report_injection"][
                    "invalid_inline_request_roles"
                ],
                ["critic"],
            )

    def test_generated_verifier_prompt_matches_available_prior_context(self) -> None:
        cases = (
            ("empty", {}, [], True),
            (
                "binder_only",
                {
                    "binder": PriorReportContent(
                        status="complete",
                        report="The requirements are bound.",
                        unresolved=(),
                    )
                },
                ["binder"],
                True,
            ),
            (
                "context_mismatch",
                {
                    "binder": PriorReportContent(
                        status="complete",
                        report="The requirements are bound.",
                        unresolved=(),
                    )
                },
                [],
                False,
            ),
        )
        for name, prior_reports, context_prior_roles, expected_healthy in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary)
                (run_dir / "requests").mkdir()
                (run_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "model": "gpt-5.5",
                            "reasoning_effort": "xhigh",
                            "model_interface": "terminal-only",
                            "prompt_hashes": {
                                "verifier": {
                                    "sha256": {
                                        "composed_system": hashlib.sha256(
                                            ROLE_INSTRUCTIONS["verifier"].encode(
                                                "utf-8"
                                            )
                                        ).hexdigest()
                                    }
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                generated_prompt = specialist_user_prompt(
                    "verifier",
                    instruction="Complete the task.",
                    request="Verify the current state.",
                    cache_path="/workspace/.agent/cache/verifier-01",
                    allocated_steps=6,
                    budget_status="190 calls remain.",
                    prior_reports=prior_reports or None,
                )
                self.assertNotIn("Summary:", generated_prompt)
                with gzip.open(
                    run_dir / "requests" / "step-0001.json.gz",
                    "wt",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        {
                            "instructions": ROLE_INSTRUCTIONS["verifier"],
                            "input": [
                                {
                                    "role": "user",
                                    "content": generated_prompt,
                                }
                            ],
                            "tools": [{"name": "run_command"}],
                            "extra_body": {
                                "session_id": "session",
                                "strict_session": True,
                                "infrastructure": "aiservices",
                            },
                        },
                        handle,
                    )
                command = (
                    "python /opt/cuawright-tools/verifier.py "
                    '--request "Verify the current state." --max-steps 6'
                )
                context_reports = {
                    role: {
                        "report_artifact": f"specialists/{role}-01/report.json",
                        "report_sha256": "a" * 64,
                    }
                    for role in context_prior_roles
                }
                (run_dir / "trace.jsonl").write_text(
                    "\n".join(
                        json.dumps(row)
                        for row in (
                            {"event": "terminal_command", "command": command},
                            {
                                "event": "specialist_context_prepared",
                                "role": "verifier",
                                "invocation_id": "verifier-01",
                                "phase_id": "phase-1",
                                "prior_roles": context_prior_roles,
                                "prior_reports": context_reports,
                            },
                            {
                                "event": "specialist_completed",
                                "role": "verifier",
                                "invocation_id": "verifier-01",
                                "status": "ready",
                                "steps_used": 3,
                                "requested_steps": 6,
                                "allocated_steps": 6,
                                "budget_source": "model",
                                "budget": {"remaining_steps": 190},
                                "request_artifact": (
                                    "specialists/verifier-01/request.txt"
                                ),
                                "report_artifact": (
                                    "specialists/verifier-01/report.json"
                                ),
                                "report_sha256": "b" * 64,
                                "report_size_bytes": 100,
                                "forced": False,
                            },
                            {
                                "event": "command_result",
                                "role": "actor",
                                "stdout": json.dumps(
                                    {
                                        "status": "success",
                                        "role": "verifier",
                                        "allocated_steps": 6,
                                        "used_steps": 3,
                                        "report": {
                                            "status": "ready",
                                            "report": "The current state is ready.",
                                            "unresolved": [],
                                        },
                                        "global_budget": {"remaining_steps": 190},
                                    }
                                ),
                                "attachments": [],
                            },
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )

                result = check_run_health(
                    run_dir,
                    benchmark="osworld",
                    expected_model="gpt-5.5",
                    expected_reasoning="xhigh",
                )

                self.assertEqual(
                    result["objective_stop_required"],
                    not expected_healthy,
                )
                self.assertEqual(
                    result["checks"]["prior_report_injection"]["ok"],
                    expected_healthy,
                )
                self.assertEqual(
                    result["checks"]["prior_report_injection"][
                        "context_evidence_invocations"
                    ],
                    {"verifier": ["verifier-01"]},
                )
                self.assertEqual(
                    result["checks"]["prior_report_injection"][
                        "request_evidence_invocations"
                    ],
                    ({"verifier": ["verifier-01"]} if expected_healthy else {}),
                )
                self.assertEqual(
                    result["checks"]["prior_report_injection"][
                        "inline_request_prior_roles"
                    ],
                    {"verifier": [list(prior_reports)]},
                )

    def test_benchmark_specialist_role_identity_uses_session_then_recorded_hash(
        self,
    ) -> None:
        binder_report = PriorReportContent(
            status="complete",
            report="The requirements are bound.",
            unresolved=(),
        )
        for overlay in (
            OSWORLD_OVERLAY,
            BENCHCAD_TERMINAL_OVERLAY,
            CADGENBENCH_TERMINAL_OVERLAY,
        ):
            for role in ("binder", "critic", "verifier"):
                with self.subTest(benchmark=overlay.name, role=role):
                    prompt = specialist_user_prompt(
                        role,
                        instruction="Complete the benchmark task.",
                        request=f"Perform {role} inspection.",
                        cache_path=f"/workspace/.agent/cache/{role}-01",
                        allocated_steps=6,
                        budget_status="190 calls remain.",
                        prior_reports=(
                            {"binder": binder_report} if role == "critic" else None
                        ),
                    )
                    session_request = _generated_provider_request(
                        role,
                        overlay,
                        prompt,
                        session_id=f"osworld-cli-{role}-0123456789abcdef",
                    )
                    self.assertEqual(
                        _specialist_role_from_request(session_request),
                        role,
                    )
                    fallback_request = _generated_provider_request(
                        role,
                        overlay,
                        prompt,
                        session_id=None,
                    )
                    system_prompt = fallback_request["input"][0]["content"]
                    self.assertEqual(
                        _specialist_role_from_request(
                            fallback_request,
                            manifest={
                                "prompt_hashes": {
                                    role: {
                                        "sha256": {
                                            "composed_system": hashlib.sha256(
                                                system_prompt.encode("utf-8")
                                            ).hexdigest()
                                        }
                                    }
                                }
                            },
                        ),
                        role,
                    )

    def test_benchmark_generated_requests_validate_early_verifier_priors(
        self,
    ) -> None:
        binder_report = PriorReportContent(
            status="complete",
            report="The requirements are bound.",
            unresolved=(),
        )
        cases = tuple(
            (benchmark, overlay, verifier_prior_roles)
            for benchmark, overlay in (
                ("benchcad", BENCHCAD_TERMINAL_OVERLAY),
                ("cadgenbench", CADGENBENCH_TERMINAL_OVERLAY),
            )
            for verifier_prior_roles in ([], ["binder"])
        )
        for benchmark, overlay, verifier_prior_roles in cases:
            with (
                self.subTest(
                    benchmark=benchmark,
                    verifier_prior_roles=verifier_prior_roles,
                ),
                tempfile.TemporaryDirectory() as temporary,
            ):
                run_dir = Path(temporary)
                (run_dir / "requests").mkdir()
                (run_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "model": "gpt-5.5",
                            "reasoning_effort": "xhigh",
                            "model_interface": "terminal-only",
                        }
                    ),
                    encoding="utf-8",
                )
                prompts = {
                    "binder": specialist_user_prompt(
                        "binder",
                        instruction="Complete the benchmark task.",
                        request="Bind the requirements.",
                        cache_path="/workspace/.agent/cache/binder-01",
                        allocated_steps=6,
                        budget_status="190 calls remain.",
                    ),
                    "critic": specialist_user_prompt(
                        "critic",
                        instruction="Complete the benchmark task.",
                        request="Critique the requirements.",
                        cache_path="/workspace/.agent/cache/critic-01",
                        allocated_steps=6,
                        budget_status="184 calls remain.",
                        prior_reports={"binder": binder_report},
                    ),
                    "verifier": specialist_user_prompt(
                        "verifier",
                        instruction="Complete the benchmark task.",
                        request="Verify the current state early.",
                        cache_path="/workspace/.agent/cache/verifier-01",
                        allocated_steps=6,
                        budget_status="178 calls remain.",
                        prior_reports=(
                            {"binder": binder_report} if verifier_prior_roles else None
                        ),
                    ),
                }
                for index, role in enumerate(
                    ("binder", "critic", "verifier"),
                    start=1,
                ):
                    with gzip.open(
                        run_dir / "requests" / f"step-{index:04d}.json.gz",
                        "wt",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(
                            _generated_provider_request(
                                role,
                                overlay,
                                prompts[role],
                                session_id=(
                                    f"osworld-cli-{role}-{overlay.name}-012345"
                                ),
                            ),
                            handle,
                        )

                rows = []
                for role, prior_roles, status in (
                    ("critic", ["binder"], "accept"),
                    ("verifier", verifier_prior_roles, "ready"),
                ):
                    command = (
                        f"python /opt/cuawright-tools/{role}.py "
                        f'--request "Perform {role} inspection." --max-steps 6'
                    )
                    rows.extend(
                        (
                            {"event": "terminal_command", "command": command},
                            {
                                "event": "specialist_context_prepared",
                                "role": role,
                                "invocation_id": f"{role}-01",
                                "phase_id": "phase-1",
                                "prior_roles": prior_roles,
                                "prior_reports": {
                                    prior_role: {
                                        "report_artifact": (
                                            f"specialists/{prior_role}-01/report.json"
                                        ),
                                        "report_sha256": "a" * 64,
                                    }
                                    for prior_role in prior_roles
                                },
                            },
                            {
                                "event": "specialist_completed",
                                "role": role,
                                "invocation_id": f"{role}-01",
                                "status": status,
                                "steps_used": 3,
                                "requested_steps": 6,
                                "allocated_steps": 6,
                                "budget_source": "model",
                                "budget": {"remaining_steps": 175},
                                "request_artifact": (
                                    f"specialists/{role}-01/request.txt"
                                ),
                                "report_artifact": (
                                    f"specialists/{role}-01/report.json"
                                ),
                                "report_sha256": "b" * 64,
                                "report_size_bytes": 100,
                                "forced": False,
                            },
                            {
                                "event": "command_result",
                                "role": "actor",
                                "stdout": json.dumps(
                                    {
                                        "status": "success",
                                        "role": role,
                                        "allocated_steps": 6,
                                        "used_steps": 3,
                                        "report": {
                                            "status": status,
                                            "report": f"{role.title()} report.",
                                            "unresolved": [],
                                        },
                                        "global_budget": {"remaining_steps": 175},
                                    }
                                ),
                                "attachments": [],
                            },
                        )
                    )
                (run_dir / "trace.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                    encoding="utf-8",
                )

                result = check_run_health(
                    run_dir,
                    benchmark=benchmark,
                    expected_model="gpt-5.5",
                    expected_reasoning="xhigh",
                )

                self.assertFalse(result["objective_stop_required"])
                self.assertTrue(result["checks"]["prior_report_injection"]["ok"])
                self.assertEqual(
                    result["checks"]["prior_report_injection"][
                        "inline_request_prior_roles"
                    ],
                    {
                        "critic": [["binder"]],
                        "verifier": [verifier_prior_roles],
                    },
                )
                self.assertEqual(
                    result["checks"]["prior_report_injection"][
                        "request_evidence_roles"
                    ],
                    ["critic", "verifier"],
                )

    def test_untagged_no_tool_specialist_report_request_is_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "requests").mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-5.5",
                        "reasoning_effort": "xhigh",
                        "model_interface": "terminal-only",
                    }
                ),
                encoding="utf-8",
            )
            with gzip.open(
                run_dir / "requests" / "step-0042.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "input": [
                            {
                                "role": "user",
                                "content": (
                                    "No tools are available. Return one JSON object "
                                    "with status, report, and unresolved."
                                ),
                            }
                        ],
                        "extra_body": {
                            "session_id": "session",
                            "strict_session": True,
                            "infrastructure": "aiservices",
                        },
                    },
                    handle,
                )
            (run_dir / "trace.jsonl").write_text(
                json.dumps({"event": "model_turn", "role": "critic"}) + "\n",
                encoding="utf-8",
            )

            result = check_run_health(
                run_dir,
                benchmark="osworld",
                expected_model="gpt-5.5",
                expected_reasoning="xhigh",
            )

            self.assertFalse(result["objective_stop_required"])
            self.assertTrue(result["checks"]["sole_tool"]["specialist_report"])


if __name__ == "__main__":
    unittest.main()

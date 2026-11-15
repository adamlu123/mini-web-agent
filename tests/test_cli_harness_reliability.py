from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from cuawright.environments.base import CommandResult
from cuawright.environments.osworld.lifecycle import (
    _GUEST_HOME_WRITE_PROBE,
    InfrastructureError,
    guest_root_free_kb as _guest_root_free_kb,
    is_guest_infrastructure_error as _is_guest_infrastructure_error,
    is_retryable_setup_error as _is_retryable_setup_error,
    run_guest_command as _run_guest_execute_once,
)
from cuawright.models.openai_responses import (
    POLICY_RETRY_DELAYS,
    RESPONSE_RETRY_DELAYS,
    ResponsesThread,
    _exception_diagnostics,
    _is_policy_rejection,
    _is_transient_response_error,
)
from cuawright.run.osworld import OSWorldRunner
from desktop_env.desktop_env import EnvironmentSetupError
from desktop_env.providers.docker.provider import (
    DEFAULT_DISK_SIZE_GB,
    DockerProvider,
)


class RetryClassificationTests(unittest.TestCase):
    def test_generic_connection_error_is_transient(self) -> None:
        self.assertTrue(_is_transient_response_error(Exception("Connection error.")))

    def test_gateway_5xx_family_is_transient(self) -> None:
        for status in (500, 502, 504, 520, 524, 529, 599):
            with self.subTest(status=status):
                self.assertTrue(
                    _is_transient_response_error(
                        Exception(f"gateway returned {status}")
                    )
                )

    def test_policy_rejections_are_identified(self) -> None:
        for message in (
            "Error code: 400 content_policy_violation",
            "Error code: 400 cyber_policy: content was flagged for possible "
            "cybersecurity risk",
            "Error code: 400 {'code': 'invalid_prompt'} prompt was flagged as "
            "potentially violating our usage policy",
        ):
            with self.subTest(message=message):
                self.assertTrue(_is_policy_rejection(Exception(message)))
        self.assertFalse(_is_policy_rejection(Exception("400 malformed request")))

    def test_policy_then_success_retries_exact_unchanged_request(self) -> None:
        success = SimpleNamespace(
            model_dump=lambda mode="json": {"status": "completed", "output": []}
        )
        create = Mock(
            side_effect=[
                Exception("400 cyber_policy: cybersecurity risk"),
                success,
            ]
        )
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[{"type": "function", "name": "run_command"}],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )
        thread.add_image("/workspace/example.png", "image/png", b"image")
        original_context = copy.deepcopy(thread.context)
        tools = [{"type": "function", "name": "run_command", "strict": True}]

        with (
            patch("cuawright.models.openai_responses.time.sleep") as sleep,
            patch("cuawright.models.openai_responses.random.uniform", return_value=0.0),
            patch.object(thread, "_drop_blocked_images") as drop_images,
        ):
            response = thread.call(tools=tools, tool_choice="auto")

        self.assertEqual(response["status"], "completed")
        self.assertEqual(create.call_count, 2)
        first = create.call_args_list[0].kwargs
        second = create.call_args_list[1].kwargs
        self.assertEqual(first, second)
        self.assertEqual(first["input"], original_context)
        self.assertEqual(first["tools"], tools)
        self.assertEqual(thread.context[: len(original_context)], original_context)
        sleep.assert_called_once_with(POLICY_RETRY_DELAYS[0])
        drop_images.assert_not_called()
        trace.event.assert_called_once_with(
            "policy_retry",
            {
                "role": "actor",
                "attempt": 1,
                "max_attempts": 10,
                "delay_seconds": 1.0,
                "exception_type": "Exception",
                "status_code": None,
                "billed_response_count": 0,
                "error": "Exception: 400 cyber_policy: cybersecurity risk",
            },
        )

    def test_policy_feedback_retry_appends_benign_context_when_enabled(self) -> None:
        success = SimpleNamespace(
            model_dump=lambda mode="json": {"status": "completed", "output": []}
        )
        create = Mock(
            side_effect=[
                Exception("400 cyber_policy: cybersecurity risk"),
                success,
            ]
        )
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
            policy_feedback_max_retries=2,
        )

        response = thread.call()

        self.assertEqual(response["status"], "completed")
        self.assertEqual(create.call_count, 2)
        first = create.call_args_list[0].kwargs["input"]
        second = create.call_args_list[1].kwargs["input"]
        self.assertEqual(
            first,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
        )
        self.assertEqual(second[:-1], first)
        self.assertEqual(second[-1]["role"], "user")
        self.assertIn("compliance retry 1/2", second[-1]["content"])
        self.assertIn("clearly benign", second[-1]["content"])
        trace.event.assert_called_once()
        self.assertEqual(trace.event.call_args.args[0], "policy_feedback_retry")

    def test_policy_feedback_exhaustion_is_bounded(self) -> None:
        create = Mock(side_effect=[Exception("400 cyber_policy") for _ in range(3)])
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
            policy_feedback_max_retries=2,
        )

        with self.assertRaisesRegex(RuntimeError, "cyber_policy"):
            thread.call()

        self.assertEqual(create.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in trace.event.call_args_list],
            [
                "policy_feedback_retry",
                "policy_feedback_retry",
                "policy_feedback_retries_exhausted",
            ],
        )

    def test_model_infrastructure_can_be_omitted(self) -> None:
        success = SimpleNamespace(
            model_dump=lambda mode="json": {"status": "completed", "output": []}
        )
        create = Mock(return_value=success)
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=Mock(),
            model="gpt-5.6-sol",
            reasoning_effort="max",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
            infrastructure=None,
        )

        thread.call()

        self.assertNotIn(
            "infrastructure",
            create.call_args.kwargs["extra_body"],
        )
        self.assertEqual(create.call_args.kwargs["extra_body"]["tier"], "base")
        self.assertTrue(create.call_args.kwargs["extra_body"]["strict_session"])

    def test_policy_exhaustion_uses_all_attempts_and_emits_final_event(self) -> None:
        create = Mock(
            side_effect=[Exception("400 content_policy_violation") for _ in range(10)]
        )
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[{"type": "function", "name": "run_command"}],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )
        thread.add_image("/workspace/example.png", "image/png", b"image")
        original_context = copy.deepcopy(thread.context)

        with (
            patch("cuawright.models.openai_responses.time.sleep") as sleep,
            patch("cuawright.models.openai_responses.random.uniform", return_value=0.0),
            patch.object(thread, "_drop_blocked_images") as drop_images,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Responses API call failed: Exception: 400 content_policy_violation",
            ):
                thread.call()

        self.assertEqual(create.call_count, 10)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            list(POLICY_RETRY_DELAYS),
        )
        events = [call.args[0] for call in trace.event.call_args_list]
        self.assertEqual(events, ["policy_retry"] * 9 + ["policy_retries_exhausted"])
        exhausted = trace.event.call_args_list[-1].args[1]
        self.assertEqual(exhausted["attempt"], 10)
        self.assertEqual(exhausted["max_attempts"], 10)
        self.assertEqual(thread.context, original_context)
        self.assertTrue(
            all(
                call.kwargs == create.call_args_list[0].kwargs
                for call in create.call_args_list[1:]
            )
        )
        drop_images.assert_not_called()

    def test_policy_retry_preserves_billed_count_and_status(self) -> None:
        policy_error = Exception("400 content_policy_violation")
        policy_error.billed_response_count = 2
        policy_error.status_code = 400
        success = SimpleNamespace(
            model_dump=lambda mode="json": {"status": "completed", "output": []}
        )
        create = Mock(side_effect=[policy_error, success])
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )

        with (
            patch("cuawright.models.openai_responses.time.sleep"),
            patch("cuawright.models.openai_responses.random.uniform", return_value=0.0),
        ):
            response = thread.call()

        self.assertEqual(response["billed_response_count"], 3)
        event = trace.event.call_args.args[1]
        self.assertEqual(event["billed_response_count"], 2)
        self.assertEqual(event["status_code"], 400)

    def test_malformed_ordinary_400_is_not_policy_or_retried(self) -> None:
        create = Mock(side_effect=Exception("400 malformed request"))
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )

        with patch("cuawright.models.openai_responses.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                thread.call()

        self.assertEqual(create.call_count, 1)
        sleep.assert_not_called()
        trace.event.assert_not_called()

    def test_transient_429_keeps_response_retry_schedule(self) -> None:
        success = SimpleNamespace(
            model_dump=lambda mode="json": {"status": "completed", "output": []}
        )
        create = Mock(side_effect=[Exception("429 rate limit"), success])
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )

        with (
            patch("cuawright.models.openai_responses.time.sleep") as sleep,
            patch("cuawright.models.openai_responses.random.uniform", return_value=0.0),
        ):
            thread.call()

        self.assertEqual(create.call_count, 2)
        sleep.assert_called_once_with(RESPONSE_RETRY_DELAYS[0])
        self.assertEqual(trace.event.call_args.args[0], "response_retry")

    def test_exception_cause_is_retained(self) -> None:
        cause = OSError("No route to host")
        error = RuntimeError("Connection error")
        error.__cause__ = cause
        diagnostics = _exception_diagnostics(error)
        self.assertIn("RuntimeError: Connection error", diagnostics)
        self.assertIn("OSError: No route to host", diagnostics)
        self.assertTrue(_is_transient_response_error(error))

    def test_connection_reset_setup_error_is_retryable(self) -> None:
        cause = requests.ConnectionError("connection reset by peer")
        error = EnvironmentSetupError("Custom task setup failed")
        error.__cause__ = cause
        self.assertTrue(_is_retryable_setup_error(error))

    def test_responses_request_uses_aiservices_infrastructure(self) -> None:
        create = Mock(
            return_value=SimpleNamespace(model_dump=lambda mode="json": {"output": []})
        )
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=Mock(),
            model="gpt-5.5",
            reasoning_effort="xhigh",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )

        thread.call()

        extra_body = create.call_args.kwargs["extra_body"]
        self.assertTrue(extra_body["strict_session"])
        self.assertEqual(extra_body["infrastructure"], "aiservices")

    def test_deterministic_setup_error_is_not_retryable(self) -> None:
        self.assertFalse(
            _is_retryable_setup_error(
                EnvironmentSetupError("Task setup rejected invalid configuration")
            )
        )

    def test_control_channel_setup_failures_are_retryable(self) -> None:
        for message in (
            "Environment setup returned unsuccessful status",
            "Proxy setup failed before applying task runtime overrides",
            "Environment control server not ready before applying task runtime overrides",
        ):
            with self.subTest(message=message):
                self.assertTrue(
                    _is_retryable_setup_error(EnvironmentSetupError(message))
                )

    def test_guest_bridge_failure_is_runtime_infrastructure(self) -> None:
        self.assertTrue(
            _is_guest_infrastructure_error(
                "StatePersistenceError <- OSError: "
                "OSWorld guest bridge execution failed"
            )
        )

    def test_generic_connection_error_exhausts_all_attempts(self) -> None:
        create = Mock(side_effect=Exception("Connection error."))
        trace = Mock()
        thread = ResponsesThread(
            client=SimpleNamespace(responses=SimpleNamespace(create=create)),
            trace=trace,
            model="gpt-5.5",
            reasoning_effort="medium",
            role="actor",
            system_prompt="system",
            user_prompt="user",
            tools=[],
            max_output_tokens=128,
            base_url="https://example.invalid",
        )

        with (
            patch("cuawright.models.openai_responses.time.sleep"),
            patch("cuawright.models.openai_responses.random.uniform", return_value=0.0),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Responses API call failed: Exception: Connection error",
            ):
                thread.call()

        self.assertEqual(create.call_count, 10)
        self.assertEqual(trace.event.call_count, 9)


class GuestExecutionTests(unittest.TestCase):
    @staticmethod
    def response(payload: dict, status_code: int = 200) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.json.return_value = payload
        response.text = json.dumps(payload)
        return response

    def test_execute_normalizes_nonzero_returncode(self) -> None:
        controller = SimpleNamespace(http_server="http://guest")
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.post",
            return_value=self.response(
                {
                    "status": "success",
                    "output": "out",
                    "error": "err",
                    "returncode": 7,
                }
            ),
        ) as post:
            result = _run_guest_execute_once(controller, "exit 7", 30)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["returncode"], 7)
        self.assertEqual(result["error"], "err")
        command = post.call_args.kwargs["json"]["command"]
        self.assertEqual(command[-3:], ["/bin/bash", "-c", "exit 7"])
        self.assertIn("--kill-after=5s", command)

    def test_execute_timeout_is_deterministic_not_infrastructure(self) -> None:
        controller = SimpleNamespace(http_server="http://guest")
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.post",
            return_value=self.response(
                {
                    "status": "success",
                    "output": "",
                    "error": (
                        "timeout: sending signal TERM to command " "'/bin/bash'\n"
                    ),
                    "returncode": 124,
                }
            ),
        ):
            result = _run_guest_execute_once(controller, "sleep 60", 5)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["infrastructure_error"])
        self.assertIn("process group was terminated", result["error"])

    def test_child_exit_124_without_timeout_marker_is_not_timeout(self) -> None:
        controller = SimpleNamespace(http_server="http://guest")
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.post",
            return_value=self.response(
                {
                    "status": "success",
                    "output": "",
                    "error": "",
                    "returncode": 124,
                }
            ),
        ):
            result = _run_guest_execute_once(controller, "exit 124", 5)

        self.assertEqual(result["status"], "error")
        self.assertNotIn("timed_out", result)
        self.assertFalse(result["infrastructure_error"])

    def test_execute_non_200_is_protocol_infrastructure_failure(self) -> None:
        controller = SimpleNamespace(http_server="http://guest")
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.post",
            return_value=self.response(
                {"message": "Not Found"},
                status_code=404,
            ),
        ):
            result = _run_guest_execute_once(controller, "echo ok", 5)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["infrastructure_error"])
        self.assertIn("HTTP 404", result["error"])

    def test_command_specific_http_500_is_an_ordinary_error(self) -> None:
        messages = (
            (
                "'utf-8' codec can't decode byte 0xe8 in position 91: "
                "invalid continuation byte"
            ),
            (
                "'utf-8' codec can't decode bytes in position 0-1: "
                "unexpected end of data"
            ),
            (
                "Command '['/usr/bin/timeout', '5s', '/bin/bash', '-c', "
                "'nohup npm start &']' timed out after 20 seconds"
            ),
        )
        controller = SimpleNamespace(http_server="http://guest")
        for message in messages:
            with self.subTest(message=message):
                with patch(
                    "cuawright.environments.osworld.lifecycle.requests.post",
                    return_value=self.response(
                        {"status": "error", "message": message},
                        status_code=500,
                    ),
                ) as post:
                    result = _run_guest_execute_once(controller, "ambiguous", 5)

                self.assertEqual(result["status"], "error")
                self.assertEqual(result["returncode"], -1)
                self.assertFalse(result["infrastructure_error"])
                self.assertIn("HTTP 500", result["error"])
                self.assertIn("may have changed state", result["error"])
                post.assert_called_once()

    def test_unknown_http_500_remains_infrastructure(self) -> None:
        controller = SimpleNamespace(http_server="http://guest")
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.post",
            return_value=self.response(
                {"status": "error", "message": "internal state is unavailable"},
                status_code=500,
            ),
        ):
            result = _run_guest_execute_once(controller, "echo ok", 5)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["infrastructure_error"])

    def test_server_text_and_pipe_failures_do_not_prove_service_loss(self) -> None:
        with self.assertRaises(UnicodeDecodeError):
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import os; os.write(1, bytes([232]))",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        self.assertEqual(
            subprocess.run(
                ["printf", "after-binary-ok"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout,
            "after-binary-ok",
        )

        with self.assertRaises(subprocess.TimeoutExpired):
            subprocess.run(
                ["/bin/bash", "-c", "sleep 0.2 & exit 0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=0.05,
                check=False,
            )
        self.assertEqual(
            subprocess.run(
                ["printf", "after-timeout-ok"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout,
            "after-timeout-ok",
        )

    def test_execute_non_json_404_is_protocol_infrastructure_failure(self) -> None:
        controller = SimpleNamespace(http_server="http://guest")
        response = Mock()
        response.status_code = 404
        response.json.side_effect = ValueError("not json")
        response.text = "<html>Not Found</html>"
        with patch(
            "cuawright.environments.osworld.lifecycle.requests.post",
            return_value=response,
        ):
            result = _run_guest_execute_once(controller, "echo ok", 5)

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["infrastructure_error"])
        self.assertIn("Not Found", result["error"])

    def test_guest_root_free_space_parsing(self) -> None:
        controller = SimpleNamespace()
        with patch(
            "cuawright.environments.osworld.lifecycle.run_guest_command",
            return_value={
                "status": "success",
                "output": ("/dev/sda3 50800000 29000000 21800000 58% /\n"),
            },
        ):
            self.assertEqual(_guest_root_free_kb(controller), 21_800_000)

    def test_guest_home_writability_probe_is_valid_python(self) -> None:
        compile(_GUEST_HOME_WRITE_PROBE, "<guest-home-write-probe>", "exec")
        self.assertIn("os.fsync(fd)", _GUEST_HOME_WRITE_PROBE)
        self.assertIn("os.unlink(path)", _GUEST_HOME_WRITE_PROBE)

    def test_guest_home_probe_never_unlinks_a_file_it_did_not_create(self) -> None:
        with (
            patch("secrets.token_hex", return_value="existing"),
            patch("os.open", side_effect=FileExistsError("already exists")),
            patch("os.unlink") as unlink,
        ):
            with self.assertRaises(FileExistsError):
                exec(compile(_GUEST_HOME_WRITE_PROBE, "<guest-home-probe>", "exec"), {})
        unlink.assert_not_called()


class OSWorldRunnerInfrastructureTests(unittest.TestCase):
    @staticmethod
    def runner() -> OSWorldRunner:
        runner = object.__new__(OSWorldRunner)
        runner.env = SimpleNamespace(controller=object(), is_environment_used=False)
        runner.config = SimpleNamespace(
            website_host_suffix="example.invalid",
            total_steps=300,
        )
        runner.tool_step = 0
        runner._tool_step_lock = threading.Lock()
        runner.trace = Mock()
        runner.trace.command_blob.return_value = ("command-output.txt.gz", "digest")
        return runner

    def test_explicit_guest_infrastructure_result_is_escalated(self) -> None:
        runner = self.runner()
        result = {
            "status": "error",
            "output": "",
            "error": "connection reset by peer",
            "returncode": -1,
            "infrastructure_error": True,
        }

        with patch(
            "cuawright.run.osworld.run_guest_command",
            return_value=result,
        ):
            with self.assertRaisesRegex(
                InfrastructureError,
                "Guest command service failed.*connection reset by peer",
            ):
                runner._execute_command("echo ok", 30)

    def test_deterministic_failure_and_process_timeout_remain_ordinary(self) -> None:
        cases = (
            {
                "status": "error",
                "output": "",
                "error": "permission denied",
                "returncode": 7,
                "infrastructure_error": False,
            },
            {
                "status": "error",
                "output": "",
                "error": "process group was terminated",
                "returncode": 124,
                "timed_out": True,
                "infrastructure_error": False,
            },
            {
                "status": "error",
                "output": "",
                "error": (
                    "Guest /execute endpoint returned HTTP 500. "
                    "'utf-8' codec can't decode byte 0xe8 in position 91: "
                    "invalid continuation byte"
                ),
                "returncode": -1,
                "infrastructure_error": False,
            },
        )
        for result in cases:
            with self.subTest(result=result):
                runner = self.runner()
                with patch(
                    "cuawright.run.osworld.run_guest_command",
                    return_value=result,
                ) as execute:
                    command_result = runner._execute_command("false", 30)

                self.assertIsInstance(command_result, CommandResult)
                self.assertEqual(command_result.exit_code, result["returncode"])
                execute.assert_called_once()

    def test_confirmed_guest_home_read_only_is_escalated(self) -> None:
        runner = self.runner()
        command_failure = {
            "status": "error",
            "output": "",
            "error": "Read-only file system",
            "returncode": 1,
            "infrastructure_error": False,
        }
        probe_failure = {
            "status": "error",
            "output": "",
            "error": (
                "cannot create /home/user/.cuawright-write-probe: "
                "Read-only file system"
            ),
            "returncode": 1,
            "infrastructure_error": False,
        }

        with (
            patch(
                "cuawright.run.osworld.run_guest_command",
                return_value=command_failure,
            ),
            patch(
                "cuawright.environments.osworld.lifecycle.run_guest_command",
                return_value=probe_failure,
            ),
        ):
            with self.assertRaisesRegex(
                InfrastructureError,
                "/home/user is globally read-only",
            ):
                runner._execute_command(
                    "mkdir -p /home/user/Desktop/OSWorld",
                    30,
                )

    def test_protected_read_only_target_with_writable_home_is_ordinary(self) -> None:
        runner = self.runner()
        command_failure = {
            "status": "error",
            "output": "",
            "error": "cannot create /proc/example: Read-only file system",
            "returncode": 1,
            "infrastructure_error": False,
        }
        successful_probe = {
            "status": "success",
            "output": "",
            "error": "",
            "returncode": 0,
            "infrastructure_error": False,
        }

        with (
            patch(
                "cuawright.run.osworld.run_guest_command",
                return_value=command_failure,
            ) as execute,
            patch(
                "cuawright.environments.osworld.lifecycle.run_guest_command",
                return_value=successful_probe,
            ) as probe,
        ):
            result = runner._execute_command("touch /proc/example", 30)

        self.assertEqual(result.exit_code, 1)
        execute.assert_called_once()
        probe.assert_called_once()

    def test_probe_transport_failure_is_escalated(self) -> None:
        runner = self.runner()
        command_failure = {
            "status": "error",
            "output": "",
            "error": "Read-only file system",
            "returncode": 1,
            "infrastructure_error": False,
        }
        probe_transport_failure = {
            "status": "error",
            "output": "",
            "error": "guest execution service did not respond",
            "returncode": -1,
            "infrastructure_error": True,
        }

        with (
            patch(
                "cuawright.run.osworld.run_guest_command",
                return_value=command_failure,
            ),
            patch(
                "cuawright.environments.osworld.lifecycle.run_guest_command",
                return_value=probe_transport_failure,
            ),
        ):
            with self.assertRaisesRegex(
                InfrastructureError,
                "guest home writability probe failed",
            ):
                runner._execute_command("touch /home/user/Desktop/output", 30)


class DockerVolumeTests(unittest.TestCase):
    def test_default_and_explicit_volume_sizes(self) -> None:
        with patch("desktop_env.providers.docker.provider.docker.from_env"):
            provider = DockerProvider(region="")

        self.assertEqual(
            provider.environment["DISK_SIZE"],
            f"{DEFAULT_DISK_SIZE_GB}G",
        )
        self.assertEqual(provider.default_volume_size, DEFAULT_DISK_SIZE_GB)
        provider.prepare_volume("vm.qcow2", None, "Ubuntu")
        self.assertEqual(provider.environment["DISK_SIZE"], "80G")
        provider.prepare_volume("vm.qcow2", 50, "Ubuntu")
        self.assertEqual(provider.environment["DISK_SIZE"], "50G")

    def test_default_volume_is_expanded(self) -> None:
        with patch("desktop_env.providers.docker.provider.docker.from_env"):
            provider = DockerProvider(region="")
        with patch(
            "desktop_env.providers.docker.provider.expand_guest_volume"
        ) as expand:
            provider.finalize_volume(
                "vm.qcow2",
                None,
                "Ubuntu",
                controller=Mock(),
                setup_controller=Mock(),
                client_password="",
            )
        expand.assert_called_once()

    def test_stop_removes_anonymous_container_volumes(self) -> None:
        with patch("desktop_env.providers.docker.provider.docker.from_env"):
            provider = DockerProvider(region="")
        container = Mock()
        provider.container = container

        with patch("desktop_env.providers.docker.provider.time.sleep"):
            provider.stop_emulator("vm.qcow2")

        container.stop.assert_called_once_with()
        container.remove.assert_called_once_with(force=True, v=True)
        self.assertIsNone(provider.container)

    def test_failed_start_forces_removal_even_when_stop_fails(self) -> None:
        with patch("desktop_env.providers.docker.provider.docker.from_env"):
            provider = DockerProvider(region="")
        container = Mock()
        container.stop.side_effect = RuntimeError("stop failed")
        provider.client.containers.run.return_value = container

        with patch.object(
            provider,
            "_get_published_port",
            side_effect=RuntimeError("port failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "port failure"):
                provider.start_emulator("vm.qcow2", True, "Ubuntu")

        container.stop.assert_called_once_with()
        container.remove.assert_called_once_with(force=True, v=True)
        self.assertIsNone(provider.container)


if __name__ == "__main__":
    unittest.main()

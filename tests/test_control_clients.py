from __future__ import annotations

import ast
import json
import subprocess
import sys
import unittest
from pathlib import Path

from cuawright.environments.control_clients import (
    CLIENT_FILENAMES,
    CLIENT_SHA256,
    CLIENT_SOURCES,
    client_source_hashes,
    iter_control_clients,
)


CLIENT_ROOT = (
    Path(__file__).parents[1] / "src" / "cuawright" / "environments" / "control_clients"
)
CASES = {
    "binder.py": (
        ["--request", "Inspect the task facts.", "--max-steps", "7"],
        "binder",
    ),
    "critic.py": (
        ["--request", "Challenge the Binder's assumptions.", "--max-steps", "7"],
        "critic",
    ),
    "verifier.py": (
        ["--request", "Inspect the saved result; report evidence.", "--max-steps", "7"],
        "verifier",
    ),
    "image_show.py": (["--path", "/workspace/render.png"], "image_show"),
    "ask_user.py": (
        ["--question", "Which account should I use?"],
        "ask_user",
    ),
    "submit.py": ([], "submit"),
}


def run_client(name: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLIENT_ROOT / name), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


class ControlClientTests(unittest.TestCase):
    def test_active_clients_emit_one_small_operation_bound_json_object(self) -> None:
        self.assertEqual(set(CLIENT_FILENAMES), set(CASES))
        self.assertNotIn("report.py", CLIENT_FILENAMES)
        self.assertFalse((CLIENT_ROOT / "report.py").exists())
        for name, (arguments, operation) in CASES.items():
            with self.subTest(name=name):
                result = run_client(name, *arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                self.assertLessEqual(len(result.stdout.encode()), 16 * 1024)
                self.assertEqual(result.stdout.count("\n"), 1)
                payload = json.loads(result.stdout)
                self.assertEqual(payload["cuawright_control"], 1)
                self.assertEqual(payload["operation"], operation)

    def test_inline_request_and_question_text_round_trip_exactly(self) -> None:
        request = "Inspect exact values; don't mutate state. 😀"
        question = "Which profile's exact ID should be used?"

        binder = json.loads(
            run_client(
                "binder.py",
                "--request",
                request,
                "--max-steps",
                "12",
            ).stdout
        )
        asked = json.loads(run_client("ask_user.py", "--question", question).stdout)

        self.assertEqual(binder["request"], request)
        self.assertEqual(binder["max_steps"], 12)
        self.assertEqual(asked["question"], question)

    def test_required_unknown_and_trailing_arguments_fail(self) -> None:
        required_clients = {
            "binder.py",
            "critic.py",
            "verifier.py",
            "image_show.py",
            "ask_user.py",
        }
        for name in required_clients:
            with self.subTest(name=name):
                self.assertEqual(run_client(name).returncode, 2)
        for name, (arguments, _operation) in CASES.items():
            with self.subTest(name=name):
                self.assertEqual(
                    run_client(name, *arguments, "--unknown", "value").returncode,
                    2,
                )
                self.assertEqual(
                    run_client(name, *arguments, "trailing").returncode,
                    2,
                )

    def test_role_max_steps_is_required_typed_and_role_bounded(self) -> None:
        limits = {"binder.py": 40, "critic.py": 32, "verifier.py": 24}
        for name, maximum in limits.items():
            with self.subTest(name=name):
                request = ["--request", "Focus"]
                self.assertEqual(run_client(name, *request).returncode, 2)
                accepted = run_client(
                    name,
                    *request,
                    "--max-steps",
                    str(maximum),
                )
                self.assertEqual(accepted.returncode, 0, accepted.stderr)
                self.assertEqual(json.loads(accepted.stdout)["max_steps"], maximum)
                for invalid in ("0", "-1", "many", str(maximum + 1)):
                    self.assertEqual(
                        run_client(
                            name,
                            *request,
                            "--max-steps",
                            invalid,
                        ).returncode,
                        2,
                    )

    def test_inline_text_limits_are_enforced(self) -> None:
        accepted_request = run_client(
            "binder.py",
            "--request",
            "x" * 2_000,
            "--max-steps",
            "1",
        )
        rejected_request = run_client(
            "binder.py",
            "--request",
            "x" * 2_001,
            "--max-steps",
            "1",
        )
        accepted_question = run_client(
            "ask_user.py",
            "--question",
            "q" * 2_000,
        )
        rejected_question = run_client(
            "ask_user.py",
            "--question",
            "q" * 2_001,
        )
        empty_question = run_client("ask_user.py", "--question", "   ")

        self.assertEqual(accepted_request.returncode, 0)
        self.assertEqual(rejected_request.returncode, 2)
        self.assertEqual(accepted_question.returncode, 0)
        self.assertEqual(rejected_question.returncode, 2)
        self.assertEqual(empty_question.returncode, 2)

    def test_submit_requires_no_actor_generated_report(self) -> None:
        result = run_client("submit.py")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            json.loads(result.stdout),
            {"cuawright_control": 1, "operation": "submit"},
        )

    def test_clients_use_only_stdlib_and_contain_no_privileged_logic(self) -> None:
        allowed_imports = {"__future__", "argparse", "json", "pathlib"}
        forbidden_text = (
            "openai",
            "anthropic",
            "api_key",
            "evaluator",
            "socket",
            "http",
            "subprocess",
            "specialist_runner",
            "run_store",
        )
        for name in CLIENT_FILENAMES:
            with self.subTest(name=name):
                source = (CLIENT_ROOT / name).read_text(encoding="utf-8")
                tree = ast.parse(source)
                imports = {
                    node.module.split(".", 1)[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                }
                imports.update(
                    alias.name.split(".", 1)[0]
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                )
                self.assertLessEqual(imports, allowed_imports)
                lowered = source.lower()
                self.assertTrue(all(text not in lowered for text in forbidden_text))

    def test_packaged_source_hash_manifest_is_complete_and_stable(self) -> None:
        first = client_source_hashes()
        second = client_source_hashes()
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(CLIENT_FILENAMES))
        self.assertTrue(all(len(digest) == 64 for digest in first.values()))
        self.assertEqual(first, dict(CLIENT_SHA256))
        self.assertEqual(set(CLIENT_SOURCES), set(CLIENT_FILENAMES))
        self.assertEqual(
            list(iter_control_clients()),
            [
                (name, CLIENT_SOURCES[name], CLIENT_SHA256[name])
                for name in CLIENT_FILENAMES
            ],
        )


if __name__ == "__main__":
    unittest.main()

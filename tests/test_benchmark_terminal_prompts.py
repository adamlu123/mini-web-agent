from __future__ import annotations

import unittest

from cuawright.environments.benchcad.prompts import (
    ACTOR_SYSTEM_PROMPT as BENCH_ACTOR_SOURCE,
)
from cuawright.environments.benchcad.terminal_prompts import (
    ACTOR_TERMINAL_PROMPT as BENCH_ACTOR_TERMINAL,
    BENCHCAD_TERMINAL_OVERLAY,
)
from cuawright.environments.cadgenbench.prompts import (
    ACTOR_SYSTEM_PROMPT as CAD_ACTOR_SOURCE,
)
from cuawright.environments.cadgenbench.terminal_prompts import (
    ACTOR_TERMINAL_PROMPT as CAD_ACTOR_TERMINAL,
    CADGENBENCH_TERMINAL_OVERLAY,
)
from cuawright.config.prompts import compose_system_prompt


class BenchmarkTerminalPromptTests(unittest.TestCase):
    def test_benchmark_actor_sources_remain_mostly_verbatim(self) -> None:
        self.assertGreaterEqual(
            len(BENCH_ACTOR_TERMINAL.split()),
            len(BENCH_ACTOR_SOURCE.split()) * 0.90,
        )
        self.assertGreaterEqual(
            len(CAD_ACTOR_TERMINAL.split()),
            len(CAD_ACTOR_SOURCE.split()) * 0.90,
        )

    def test_benchcad_uses_terminal_only_surface(self) -> None:
        system = compose_system_prompt("actor", BENCHCAD_TERMINAL_OVERLAY)
        self.assertIn("benchcad-run", system)
        self.assertIn("benchcad-render", system)
        self.assertIn(
            "python /opt/cuawright-tools/image_show.py --path",
            system,
        )
        self.assertIn("python /opt/cuawright-tools/submit.py", system)
        self.assertNotIn("agent image show", system)
        self.assertNotIn("agent submit", system)
        self.assertNotIn("read_image", system)
        self.assertNotIn("submit_task", system)

    def test_cadgenbench_keeps_build123d_mcp_cli(self) -> None:
        system = compose_system_prompt("actor", CADGENBENCH_TERMINAL_OVERLAY)
        self.assertIn("cad-mcp", system)
        self.assertIn("127.0.0.1:8765/mcp", system)
        self.assertIn(
            "python /opt/cuawright-tools/image_show.py --path",
            system,
        )
        self.assertIn("python /opt/cuawright-tools/submit.py", system)
        self.assertNotIn("agent image show", system)
        self.assertNotIn("agent submit", system)
        self.assertNotIn("read_image", system)
        self.assertNotIn("submit_task", system)

    def test_all_benchmarks_receive_only_terminal_control_translation(
        self,
    ) -> None:
        for overlay in (BENCHCAD_TERMINAL_OVERLAY, CADGENBENCH_TERMINAL_OVERLAY):
            with self.subTest(overlay=overlay.name):
                actor = " ".join(compose_system_prompt("actor", overlay).split())
                binder = " ".join(compose_system_prompt("binder", overlay).split())
                critic = " ".join(compose_system_prompt("critic", overlay).split())
                verifier = " ".join(compose_system_prompt("verifier", overlay).split())
                self.assertIn("only command in its run_command call", actor)
                self.assertIn(
                    "Normal submission requires a latest Verifier status",
                    actor,
                )
                self.assertIn(
                    'binder.py --request "TEXT" --max-steps N',
                    actor,
                )
                self.assertNotIn('critic.py --request "TEXT" --max-steps N', actor)
                self.assertIn(
                    'verifier.py --request "TEXT" --max-steps N',
                    actor,
                )
                ask_user = 'ask_user.py --question "TEXT"'
                self.assertIn(ask_user, actor)
                self.assertIn(ask_user, binder)
                self.assertIn(ask_user, critic)
                self.assertNotIn(ask_user, verifier)
                self.assertIn("No Binder or Critic report is included", verifier)
                self.assertNotIn("terminal `cat`", verifier)
                self.assertNotIn("REQUEST_PATH", actor)
                self.assertNotIn("QUESTION_PATH", actor)

    def test_benchmark_verifiers_keep_native_independence_guidance(self) -> None:
        bench = " ".join(
            compose_system_prompt("verifier", BENCHCAD_TERMINAL_OVERLAY).split()
        )
        cad = " ".join(
            compose_system_prompt("verifier", CADGENBENCH_TERMINAL_OVERLAY).split()
        )
        for prompt in (bench, cad):
            self.assertIn("Independently inspect", prompt)
            self.assertIn("Do not trust Actor completion claims", prompt)
        self.assertIn(
            "Report every independently checkable requirement exactly once as "
            "pass, fail, or unknown",
            bench,
        )
        self.assertIn("without relying on Binder or Actor conclusions", cad)

    def test_benchcad_critic_keeps_native_binder_authority(self) -> None:
        critic = " ".join(
            compose_system_prompt("critic", BENCHCAD_TERMINAL_OVERLAY).split()
        )
        self.assertIn(
            "The original Binder remains authoritative and will adjudicate your "
            "candidate diff",
            critic,
        )

    def test_linear_benchmark_actors_have_no_specialist_guidance(self) -> None:
        for overlay in (BENCHCAD_TERMINAL_OVERLAY, CADGENBENCH_TERMINAL_OVERLAY):
            with self.subTest(overlay=overlay.name):
                prompt = compose_system_prompt(
                    "actor",
                    overlay,
                    agent_topology="linear",
                )
                for forbidden in (
                    "Binder",
                    "Critic",
                    "Verifier",
                    "subagent",
                    "binder.py",
                    "critic.py",
                    "verifier.py",
                ):
                    self.assertNotIn(forbidden.lower(), prompt.lower())
                self.assertIn("submit.py", prompt)


if __name__ == "__main__":
    unittest.main()

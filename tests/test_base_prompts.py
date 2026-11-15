from __future__ import annotations

import unittest

from cuawright.config.prompts import (
    ACTIVE_PROMPT_TRANSFORMS,
    ACTOR_ROLE_INSTRUCTIONS,
    BINDER_CALL_MAX,
    BINDER_ROLE_INSTRUCTIONS,
    COMPACTION_HANDOFF_REQUEST,
    CRITIC_CALL_MAX,
    CRITIC_ROLE_INSTRUCTIONS,
    OSWORLD_OVERLAY,
    OSWORLD_ROLE_INSTRUCTIONS,
    ROLE_INSTRUCTIONS,
    SHARED_SYSTEM_PREFIX,
    SOLE_TERMINAL_TOOL_SCHEMA,
    TERMINAL_CONTROL_INSTRUCTIONS,
    VERIFIER_CALL_MAX,
    VERIFIER_ROLE_INSTRUCTIONS,
    BenchmarkOverlay,
    actor_resume_user_prompt,
    actor_user_prompt,
    budget_status_user_prompt,
    compose_system_prompt,
    compaction_user_prompt,
    phase_user_prompt,
    prompt_hash_manifest,
    prompt_source_manifest,
    PriorReportContent,
    specialist_user_prompt,
    text_sha256,
)
from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA
from cuawright.config.prompts import (
    BINDER_SUBAGENT_SYSTEM_PROMPT,
    CRITIC_SUBAGENT_SYSTEM_PROMPT,
    ORCHESTRATOR_COMPACTION_REQUEST,
    ORCHESTRATOR_SYSTEM_PROMPT,
    VERIFIER_SUBAGENT_SYSTEM_PROMPT,
)
from cuawright.config.prompts import ACTOR_SYSTEM_PROMPT


REMOVED_NATIVE_TOOLS = (
    "read_image",
    "submit_task",
    "submit_subagent_report",
    "call_binder",
    "call_critic",
    "call_verifier",
)

REMOVED_AGENT_PROTOCOL = (
    "agent spawn",
    "agent report",
    "agent image show",
    "agent user ask",
    "agent status",
    "agent submit",
)


def normalized(text: str) -> str:
    return " ".join(text.split())


class FullProductionPromptPreservationTests(unittest.TestCase):
    def test_audited_source_hashes_match_pre_reorganization_prompts(self) -> None:
        expected = {
            "actor": "718c006a6b0cd776820833144e7b68f76465cc40d8c23a38fab51a01444dba2c",
            "orchestrator": "6cf1ddfa8d53446c415230773381d6b53a05399f6b5b31518c518c372985ca31",
            "binder": "a28ba05716ca65945bf7f0fd9b209b5a095269e3b7874f47923786b665283ed5",
            "critic": "c4776fa96cc5dc6ac52e4bf8173c8a06442098c2836a6a5346484c179e0d84a6",
            "verifier": "a5866fb716cb168a005c23372c3788d80f22905995bdb7a4f19c651264236c80",
            "compaction": "450330601bc5e15f39035df228d8bf3b8ecf4f267fa5a58705312cb7509e7ea8",
        }
        manifest = prompt_source_manifest()
        self.assertEqual(
            {
                name: source["source_sha256"]
                for name, source in manifest["sources"].items()
            },
            expected,
        )

    def test_only_active_actor_orchestrator_sources_are_transformed(self) -> None:
        self.assertEqual(
            set(ACTIVE_PROMPT_TRANSFORMS),
            {"actor", "orchestrator", "binder", "critic", "verifier", "compaction"},
        )
        self.assertEqual(
            ACTIVE_PROMPT_TRANSFORMS["actor"].source_text,
            ACTOR_SYSTEM_PROMPT,
        )
        orchestrator_extension = ORCHESTRATOR_SYSTEM_PROMPT.removeprefix(
            ACTOR_SYSTEM_PROMPT
        ).strip()
        self.assertEqual(
            ACTIVE_PROMPT_TRANSFORMS["orchestrator"].source_text,
            orchestrator_extension,
        )
        self.assertEqual(
            ACTIVE_PROMPT_TRANSFORMS["binder"].source_text,
            BINDER_SUBAGENT_SYSTEM_PROMPT,
        )
        self.assertEqual(
            ACTIVE_PROMPT_TRANSFORMS["critic"].source_text,
            CRITIC_SUBAGENT_SYSTEM_PROMPT,
        )
        self.assertEqual(
            ACTIVE_PROMPT_TRANSFORMS["verifier"].source_text,
            VERIFIER_SUBAGENT_SYSTEM_PROMPT,
        )
        self.assertEqual(
            ACTIVE_PROMPT_TRANSFORMS["compaction"].source_text,
            ORCHESTRATOR_COMPACTION_REQUEST,
        )

    def test_every_replacement_is_local_and_source_anchored(self) -> None:
        allowed_kinds = {
            "terminal-interface",
            "generic-benchmark",
            "path-coordination",
        }
        for name, transform in ACTIVE_PROMPT_TRANSFORMS.items():
            with self.subTest(name=name):
                rendered = transform.render()
                self.assertTrue(rendered.strip())
                for replacement in transform.replacements:
                    self.assertIn(replacement.kind, allowed_kinds)
                    self.assertEqual(
                        transform.source_text.count(replacement.old),
                        replacement.expected_count,
                    )
                    self.assertIn(replacement.new, rendered)

    def test_full_sources_keep_at_least_ninety_percent_of_words(self) -> None:
        manifest = prompt_source_manifest()
        for name, source in manifest["sources"].items():
            with self.subTest(name=name):
                self.assertGreaterEqual(
                    source["rendered_words"],
                    source["source_words"] * 0.90,
                )

    def test_fixed_pipeline_prompts_are_explicitly_excluded(self) -> None:
        excluded = prompt_source_manifest()["excluded_prompt_architectures"]
        self.assertEqual(
            excluded,
            ["fixed-pipeline Binder/Critic/Verifier and fixed-pipeline compaction"],
        )

    def test_actor_retains_production_operational_guidance(self) -> None:
        required = (
            "task-local Playwright",
            "Coordinate-based xdotool/pyautogui clicking is a last resort",
            "stale state or file locks may matter",
            "formulas, links, layers, relationships, charts, or native objects",
            "earliest ordered binding that changes an evaluator-facing file",
            "visibly conflicts with the reference is disproven",
            "remove duplicate, stale, or conflicting artifacts",
            "Ask at most once for the same blocker",
            "safest task-faithful fallback",
            "By roughly 50% of the shared model-call budget",
            "live saved/reloaded task state remains authoritative",
            "Insufficient evidence remains unresolved",
            "no Binder report",
            "blocks normal submission",
        )
        for text in required:
            self.assertIn(normalized(text), normalized(ACTOR_ROLE_INSTRUCTIONS))

    def test_specialists_are_the_current_short_orchestrator_roles(self) -> None:
        self.assertIn(
            "concise checklist of how a typical evaluator would vet",
            BINDER_ROLE_INSTRUCTIONS,
        )
        self.assertIn(
            "unsupported assumptions, omissions, incorrect exact values",
            normalized(CRITIC_ROLE_INSTRUCTIONS),
        )
        self.assertIn(
            "actual saved and reloaded canonical state",
            normalized(VERIFIER_ROLE_INSTRUCTIONS),
        )
        self.assertIn(
            "canonical surface to inspect",
            normalized(BINDER_ROLE_INSTRUCTIONS),
        )
        self.assertIn(
            "direct evidence in this task uniquely supports it",
            normalized(BINDER_ROLE_INSTRUCTIONS),
        )
        self.assertIn(
            "Do not transfer task-specific values",
            normalized(BINDER_ROLE_INSTRUCTIONS),
        )
        self.assertIn("receive no Binder report", VERIFIER_ROLE_INSTRUCTIONS)
        self.assertIn("Return `repair` only", VERIFIER_ROLE_INSTRUCTIONS)
        self.assertIn("is `unknown`, not `repair`", VERIFIER_ROLE_INSTRUCTIONS)
        self.assertNotIn("field-level patches", CRITIC_ROLE_INSTRUCTIONS)
        self.assertNotIn("Report every Binder requirement", VERIFIER_ROLE_INSTRUCTIONS)


class TerminalInterfacePromptTests(unittest.TestCase):
    def test_only_run_command_tool_schema_is_exposed(self) -> None:
        self.assertIs(SOLE_TERMINAL_TOOL_SCHEMA, RUN_COMMAND_TOOL_SCHEMA)
        self.assertEqual(SOLE_TERMINAL_TOOL_SCHEMA["name"], "run_command")

    def test_terminal_prompts_remove_native_tool_names(self) -> None:
        prompts = [
            *(compose_system_prompt(role) for role in ROLE_INSTRUCTIONS),
            compaction_user_prompt(),
            actor_user_prompt("Complete the visible task."),
            actor_resume_user_prompt("Continue from the saved state."),
        ]
        for prompt in prompts:
            for native_tool in REMOVED_NATIVE_TOOLS:
                with self.subTest(native_tool=native_tool):
                    self.assertNotIn(native_tool, prompt)

    def test_actor_maps_every_control_surface_to_terminal(self) -> None:
        prompt = compose_system_prompt("actor")
        required = (
            'python /opt/cuawright-tools/binder.py --request "TEXT" --max-steps N',
            'python /opt/cuawright-tools/verifier.py --request "TEXT" --max-steps N',
            "same run_command result returns the report directly",
            "python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH",
            'python /opt/cuawright-tools/ask_user.py --question "TEXT"',
            "python /opt/cuawright-tools/submit.py",
        )
        for text in required:
            self.assertIn(normalized(text), normalized(prompt))
        self.assertNotIn(
            'python /opt/cuawright-tools/critic.py --request "TEXT" --max-steps N',
            prompt,
        )
        for legacy in REMOVED_AGENT_PROTOCOL:
            self.assertNotIn(legacy, prompt)

    def test_specialist_status_and_budget_contracts_are_preserved(self) -> None:
        self.assertEqual(
            (BINDER_CALL_MAX, CRITIC_CALL_MAX, VERIFIER_CALL_MAX), (40, 32, 24)
        )
        actor = compose_system_prompt("actor")
        self.assertIn("Choose N explicitly", normalized(actor))
        self.assertIn("task-specific work needed", normalized(actor))
        self.assertIn(
            "Valid role ranges are Binder 1-40 and Verifier 1-24",
            normalized(actor),
        )
        self.assertIn("range maxima are limits, not defaults", normalized(actor))
        expectations = {
            "binder": ("complete", "blocked"),
            "critic": ("accept", "concerns", "blocked"),
            "verifier": ("ready", "repair", "unknown"),
        }
        for role, statuses in expectations.items():
            prompt = specialist_user_prompt(
                role,
                instruction="Complete task.",
                request="Inspect.",
                cache_path=f"/workspace/.agent/cache/{role}",
                allocated_steps=4,
                budget_status="1/300 used",
                prior_reports=(
                    None
                    if role == "binder"
                    else {
                        "binder": PriorReportContent(
                            status="complete",
                            report="Binder report.",
                            unresolved=(),
                        ),
                        **(
                            {
                                "critic": PriorReportContent(
                                    status="accept",
                                    report="Critic report.",
                                    unresolved=(),
                                )
                            }
                            if role == "verifier"
                            else {}
                        ),
                    }
                ),
            )
            with self.subTest(role=role):
                self.assertIn("exactly one JSON object", prompt)
                self.assertIn("`status`", prompt)
                self.assertIn("`report`", prompt)
                self.assertIn("`unresolved`", prompt)
                self.assertNotIn("report.py", prompt)
                self.assertNotIn("Output Markdown:", prompt)
                self.assertNotIn("Output envelope:", prompt)
                for status in statuses:
                    self.assertIn(f"`{status}`", prompt)

    def test_specialist_request_may_be_empty_like_native_schema(self) -> None:
        prompt = specialist_user_prompt(
            "binder",
            instruction="Complete task.",
            request="",
            cache_path="/workspace/.agent/cache/binder",
            allocated_steps=4,
            budget_status="1/300 used",
        )

        self.assertIn("Main Actor request:\n", prompt)

    def test_ask_user_is_available_to_actor_binder_and_critic_only(self) -> None:
        syntax = 'python /opt/cuawright-tools/ask_user.py --question "TEXT"'
        for role in ("actor", "binder", "critic"):
            with self.subTest(role=role):
                prompt = compose_system_prompt(role)
                self.assertIn(syntax, prompt)
                self.assertIn(
                    "only command in its run_command call",
                    normalized(prompt),
                )
        self.assertNotIn(syntax, compose_system_prompt("verifier"))

    def test_prior_report_contents_are_injected_without_provenance_ceremony(
        self,
    ) -> None:
        prompt = specialist_user_prompt(
            "critic",
            instruction="Complete task.",
            request="Challenge the acceptance brief.",
            cache_path="/workspace/.agent/cache/critic-01",
            allocated_steps=12,
            budget_status="10/300 used",
            prior_reports={
                "binder": PriorReportContent(
                    status="complete",
                    report="# Binder\n\nUse exact value 42.",
                    unresolved=(),
                )
            },
        )
        self.assertIn("Latest Binder report:", prompt)
        self.assertIn("# Binder\n\nUse exact value 42.", prompt)
        self.assertIn("Status: complete", prompt)
        self.assertNotIn("Summary:", prompt)
        self.assertNotIn("Body provenance:", prompt)
        self.assertNotIn("Envelope provenance:", prompt)
        self.assertNotIn("cat --", prompt)
        self.assertNotIn("bodies are not injected", prompt)

    def test_only_critic_requires_a_prior_binder_report(self) -> None:
        common = {
            "instruction": "Complete task.",
            "request": "Inspect.",
            "cache_path": "/workspace/.agent/cache/x",
            "allocated_steps": 4,
            "budget_status": "1/300 used",
        }
        with self.assertRaisesRegex(ValueError, "binder"):
            specialist_user_prompt("critic", **common)

        without_prior = specialist_user_prompt("verifier", **common)
        self.assertNotIn("Latest Binder report:", without_prior)
        self.assertNotIn("Latest Critic report:", without_prior)

        with_partial_prior = specialist_user_prompt(
            "verifier",
            **common,
            prior_reports={
                "critic": PriorReportContent(
                    status="concerns",
                    report="Inspect the saved artifact directly.",
                    unresolved=("Exact visual match remains uncertain.",),
                )
            },
        )
        self.assertIn("Latest Critic report:", with_partial_prior)
        self.assertNotIn("Latest Binder report:", with_partial_prior)

    def test_control_commands_have_minimal_standalone_guidance(self) -> None:
        prompt = normalized(compose_system_prompt("actor"))
        self.assertIn(
            "every `/opt/cuawright-tools/*.py` control client as the only command "
            "in its run_command call",
            prompt,
        )
        self.assertIn("Do not combine a control client with another command", prompt)
        self.assertNotIn("write its input files first", prompt)

    def test_actor_always_starts_with_binder_before_task_state_inspection(self) -> None:
        prompt = actor_user_prompt("Complete the visible task.")
        self.assertIn(
            "Always start by running Binder before inspecting or modifying task "
            "state.",
            prompt,
        )
        self.assertIn("Critic is not part of the active default architecture", prompt)
        self.assertIn("Binder may be called only once", ACTOR_ROLE_INSTRUCTIONS)
        self.assertIn(
            "preserve a reversible backup or checkpoint",
            ACTOR_ROLE_INSTRUCTIONS,
        )
        self.assertNotIn("normally run Binder first", prompt)

    def test_ready_verifier_submission_gate_is_explicit(self) -> None:
        prompt = normalized(compose_system_prompt("actor"))
        self.assertIn("Normal submission requires a latest Verifier status", prompt)
        self.assertNotIn("critic.py", prompt)

    def test_linear_actor_prompt_contains_no_specialist_surface(self) -> None:
        prompt = compose_system_prompt("actor", agent_topology="linear")
        dynamic = "\n\n".join(
            (
                actor_user_prompt(
                    "Complete the visible task.",
                    agent_topology="linear",
                ),
                phase_user_prompt(
                    "Continue the task.",
                    phase_id="phase-2",
                    phase_index=2,
                    phase_count=2,
                    previous_phase_evaluated=True,
                    agent_topology="linear",
                ),
                compaction_user_prompt(agent_topology="linear"),
            )
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
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden.lower(), (prompt + dynamic).lower())
        self.assertIn("image_show.py", prompt)
        self.assertIn("ask_user.py", prompt)
        self.assertIn("submit.py", prompt)
        self.assertNotIn("/opt/cuawright-tools/*.py", prompt)
        with self.assertRaisesRegex(ValueError, "only the actor"):
            compose_system_prompt("binder", agent_topology="linear")


class CacheAndDynamicPromptTests(unittest.TestCase):
    def test_system_composition_order_is_cache_stable(self) -> None:
        system = compose_system_prompt("actor")
        effective_role = OSWORLD_ROLE_INSTRUCTIONS["actor"].replace(
            "/workspace/.agent",
            OSWORLD_OVERLAY.workspace,
        )
        effective_terminal = TERMINAL_CONTROL_INSTRUCTIONS.replace(
            "/workspace/.agent",
            OSWORLD_OVERLAY.workspace,
        )
        self.assertTrue(system.startswith(effective_role))
        self.assertLess(
            system.index(effective_role),
            system.index(effective_terminal),
        )

    def test_osworld_chrome_guidance_is_actor_only_and_conditional(self) -> None:
        actor = normalized(compose_system_prompt("actor"))
        self.assertIn("normally reuse the setup-opened session", actor)
        self.assertIn("/home/user/.local/share/osw/venv/bin/python", actor)
        self.assertIn("http://127.0.0.1:1337", actor)
        self.assertIn("recover or launch Chrome as needed", actor)
        for role in ("binder", "critic", "verifier"):
            with self.subTest(role=role):
                prompt = compose_system_prompt(role)
                self.assertNotIn("http://127.0.0.1:1337", prompt)
                self.assertIn(
                    "use `osw --show browser.md` for guidance on browser usage",
                    prompt,
                )

    def test_overlay_can_supply_benchmark_role_prompts(self) -> None:
        role_prompts = {
            role: f"{role} benchmark role prompt" for role in ROLE_INSTRUCTIONS
        }
        overlay = BenchmarkOverlay(
            name="custom",
            system_prompt="Custom benchmark overlay.",
            role_instructions=role_prompts,
        )

        system = compose_system_prompt("critic", overlay)

        self.assertIn(role_prompts["critic"], system)
        self.assertNotIn(CRITIC_ROLE_INSTRUCTIONS, system)
        self.assertEqual(
            prompt_hash_manifest("critic", overlay)["sha256"]["role_prompt"],
            text_sha256(role_prompts["critic"]),
        )

    def test_dynamic_values_are_absent_from_system(self) -> None:
        system = compose_system_prompt("actor")
        dynamic_values = (
            "task-003-dynamic",
            "phase-2-dynamic",
            "17/300 dynamic",
            "/workspace/.agent/reports/dynamic.md",
        )
        for value in dynamic_values:
            self.assertNotIn(value, system)

    def test_osworld_system_has_no_request_file_ceremony(self) -> None:
        system = compose_system_prompt("actor")
        self.assertNotIn("/home/user/.agent/requests", system)
        self.assertNotIn("/workspace/.agent/requests", system)
        self.assertNotIn("REQUEST_PATH", system)
        self.assertNotIn("QUESTION_PATH", system)

    def test_original_actor_message_is_stable_across_compaction(self) -> None:
        initial = actor_user_prompt("Do the task.")
        resumed = actor_resume_user_prompt("Saved output at /tmp/final.")
        self.assertEqual(
            initial,
            actor_user_prompt("Do the task."),
        )
        self.assertNotIn("Saved output", initial)
        self.assertIn("Saved output", resumed)

    def test_budget_and_phase_are_separate_dynamic_messages(self) -> None:
        budget = budget_status_user_prompt(
            "21/300 used",
            midpoint_due=True,
            verifier_called=False,
            later_phases_remain=True,
        )
        phase = phase_user_prompt(
            "Create the second artifact.",
            phase_id="phase-2",
            phase_index=2,
            phase_count=3,
            previous_phase_evaluated=True,
        )
        self.assertIn("shared budget never resets", budget)
        self.assertIn("previous phase has been evaluated", phase)
        self.assertIn("state persists", phase)
        self.assertIn("fresh Binder and an independent fresh Verifier", phase)

    def test_compaction_preserves_native_handoff_without_path_ceremony(self) -> None:
        self.assertIn(
            "Subagents called and findings to retain",
            COMPACTION_HANDOFF_REQUEST,
        )
        self.assertIn("Next terminal command", COMPACTION_HANDOFF_REQUEST)
        self.assertNotIn("provenance paths", COMPACTION_HANDOFF_REQUEST)
        self.assertNotIn("request/report paths", COMPACTION_HANDOFF_REQUEST)

    def test_hash_manifest_covers_exact_runtime_suffix(self) -> None:
        without_suffix = prompt_hash_manifest("actor")
        with_suffix = prompt_hash_manifest(
            "actor",
            parallel_tool_suffix="Provider-specific stable suffix.",
        )
        self.assertIsNone(without_suffix["sha256"]["parallel_tool_suffix"])
        self.assertNotEqual(
            without_suffix["sha256"]["composed_system"],
            with_suffix["sha256"]["composed_system"],
        )
        self.assertNotEqual(
            without_suffix["sha256"]["source_audit"],
            without_suffix["sha256"]["composed_system"],
        )

    def test_custom_overlay_requires_no_registry(self) -> None:
        overlay = BenchmarkOverlay(
            name="benchcad",
            system_prompt="Benchmark overlay: BenchCAD.",
        )
        system = compose_system_prompt("actor", overlay)
        self.assertIn("Benchmark overlay: BenchCAD.", system)
        self.assertEqual(prompt_hash_manifest("actor", overlay)["overlay"], "benchcad")


if __name__ == "__main__":
    unittest.main()

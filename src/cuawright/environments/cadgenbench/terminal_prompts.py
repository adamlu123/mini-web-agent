from __future__ import annotations

from cuawright.config.prompts import (
    ACTOR_INTERFACE_ADDITION,
    ASK_USER_TERMINAL_INSTRUCTION,
    COMPACTION_HANDOFF_REQUEST,
    LINEAR_COMPACTION_HANDOFF_REQUEST,
    LINEAR_TERMINAL_CONTROL_INSTRUCTIONS,
    SHARED_SYSTEM_PREFIX,
    TERMINAL_CONTROL_INSTRUCTIONS,
    BenchmarkOverlay,
    PromptAsset,
    PromptBundle,
)

from cuawright.environments.cadgenbench.prompts import (
    ACTOR_SYSTEM_PROMPT,
    BINDER_SUBAGENT_SYSTEM_PROMPT,
    CRITIC_SUBAGENT_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    VERIFIER_SUBAGENT_SYSTEM_PROMPT,
)


def _replace(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    rendered = source
    for old, new in replacements:
        if rendered.count(old) != 1:
            raise RuntimeError(f"CADGenBench prompt source changed around {old!r}")
        rendered = rendered.replace(old, new, 1)
    return rendered


ACTOR_TERMINAL_PROMPT = "\n\n".join(
    (
        _replace(
            ACTOR_SYSTEM_PROMPT,
            (
                (
                    "Use run_command for ordinary Bash and the `cad-mcp` "
                    "command-line client, and\nread_image for public inputs or "
                    "candidate renders.",
                    "Use run_command for ordinary Bash and the `cad-mcp` "
                    "command-line client, and `python "
                    "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH` for "
                    "public inputs or candidate renders.",
                ),
                (
                    "inspect them with read_image.",
                    "inspect them with `python "
                    "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`.",
                ),
                (
                    "Call submit_task when the persisted\nresult is ready.",
                    "Run `python /opt/cuawright-tools/submit.py` when the "
                    "persisted\nresult is ready and no Verifier `repair` remains.",
                ),
            ),
        ),
        ORCHESTRATOR_SYSTEM_PROMPT,
        ACTOR_INTERFACE_ADDITION,
        ASK_USER_TERMINAL_INSTRUCTION,
    )
)

LINEAR_ACTOR_TERMINAL_PROMPT = "\n\n".join(
    (
        _replace(
            ACTOR_SYSTEM_PROMPT,
            (
                (
                    "Complete the task and prove the supplied\nbindings.",
                    "Complete the task and directly verify every\nrequirement.",
                ),
                (
                    "Use run_command for ordinary Bash and the `cad-mcp` "
                    "command-line client, and\nread_image for public inputs or "
                    "candidate renders.",
                    "Use run_command for ordinary Bash and the `cad-mcp` "
                    "command-line client, and `python "
                    "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH` for "
                    "public inputs or candidate renders.",
                ),
                (
                    "inspect them with read_image.",
                    "inspect them with `python "
                    "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`.",
                ),
                (
                    "Work linearly through the ordered bindings:",
                    "Work linearly through the task requirements:",
                ),
                (
                    "Call submit_task when the persisted\nresult is ready.",
                    "Run `python /opt/cuawright-tools/submit.py` when the "
                    "persisted\nresult is ready.",
                ),
            ),
        ),
        ASK_USER_TERMINAL_INSTRUCTION,
    )
)

BINDER_TERMINAL_PROMPT = _replace(
    BINDER_SUBAGENT_SYSTEM_PROMPT,
    (
        (
            "You may use run_command, read_image, and `cad-mcp`\ninspection "
            "commands.",
            "You may use run_command, `python "
            "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`, and "
            "`cad-mcp`\ninspection commands.",
        ),
        (
            "Finish\nwith submit_subagent_report.",
            "Return exactly one JSON object with `status`, `report`, and "
            "`unresolved`. Do not wrap it in Markdown fences.",
        ),
    ),
) + (
    "\n\nUse status `complete` or `blocked`, a non-empty report string, and an "
    "unresolved string list.\n\n" + ASK_USER_TERMINAL_INSTRUCTION
)

CRITIC_TERMINAL_PROMPT = _replace(
    CRITIC_SUBAGENT_SYSTEM_PROMPT,
    (
        (
            "You may use run_command, read_image, and `cad-mcp`\ninspection "
            "commands.",
            "You may use run_command, `python "
            "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`, and "
            "`cad-mcp`\ninspection commands.",
        ),
        (
            "Finish\nwith submit_subagent_report.",
            "The validated Binder report is already in your initial message. Return "
            "exactly one JSON object with `status`, `report`, and `unresolved`. Do "
            "not wrap it in Markdown fences.",
        ),
    ),
) + (
    "\n\nUse status `accept`, `concerns`, or `blocked`, a non-empty report "
    "string, and an unresolved string list.\n\n" + ASK_USER_TERMINAL_INSTRUCTION
)

VERIFIER_TERMINAL_PROMPT = _replace(
    VERIFIER_SUBAGENT_SYSTEM_PROMPT,
    (
        (
            "Use run_command, read_image, and\n`cad-mcp` inspection commands",
            "Use run_command, `python /opt/cuawright-tools/image_show.py --path "
            "ABSOLUTE_PATH`, and\n`cad-mcp` inspection commands",
        ),
        (
            "submissions, or trajectory data. Finish with submit_subagent_report.",
            "submissions, or trajectory data. No Binder or Critic report is "
            "included in your initial message. Return exactly one JSON object with "
            "`status`, `report`, and `unresolved`. Do not wrap it in Markdown "
            "fences.",
        ),
    ),
) + (
    "\n\nIndependently inspect the container against the original instruction "
    "without relying on Binder or Actor conclusions. Do not trust Actor completion "
    "claims or candidate expected values. Return `repair` only for a direct "
    "objective contradiction in canonical candidate state; missing proof or an "
    "inferred preference is `unknown`.\n\nUse status `ready`, `repair`, or "
    "`unknown`, a non-empty report string, and an unresolved string list."
)

CADGENBENCH_TERMINAL_OVERLAY = BenchmarkOverlay(
    name="cadgenbench",
    system_prompt=(
        "Benchmark overlay: CADGenBench. Build123d-MCP remains container-local at "
        "127.0.0.1:8765/mcp and is controlled only through the cad-mcp terminal "
        "CLI. Persistent work is under /workspace."
    ),
    role_instructions={
        "actor": ACTOR_TERMINAL_PROMPT,
        "binder": BINDER_TERMINAL_PROMPT,
        "critic": CRITIC_TERMINAL_PROMPT,
        "verifier": VERIFIER_TERMINAL_PROMPT,
    },
    linear_actor_instructions=LINEAR_ACTOR_TERMINAL_PROMPT,
    workspace="/workspace/.agent",
)

CADGENBENCH_DEFAULT_PROMPT_BUNDLE = PromptBundle(
    name="cadgenbench-default",
    benchmark="cadgenbench",
    condition="default",
    composition="terminal",
    agent_topology="orchestrated",
    workspace=CADGENBENCH_TERMINAL_OVERLAY.workspace,
    role_assets=tuple(
        (
            role,
            PromptAsset(
                f"cadgenbench-default-{role}",
                CADGENBENCH_TERMINAL_OVERLAY.role_instructions[role],
            ),
        )
        for role in ("actor", "binder", "critic", "verifier")
    ),
    shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
    terminal_control=PromptAsset(
        "orchestrated-terminal-control",
        TERMINAL_CONTROL_INSTRUCTIONS,
    ),
    system_overlay=PromptAsset(
        "cadgenbench-system-overlay",
        CADGENBENCH_TERMINAL_OVERLAY.system_prompt,
    ),
    compaction=PromptAsset(
        "orchestrated-compaction",
        COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=CADGENBENCH_TERMINAL_OVERLAY.name,
)

CADGENBENCH_LINEAR_PROMPT_BUNDLE = PromptBundle(
    name="cadgenbench-linear",
    benchmark="cadgenbench",
    condition="linear",
    composition="terminal",
    agent_topology="linear",
    workspace=CADGENBENCH_TERMINAL_OVERLAY.workspace,
    role_assets=(
        (
            "actor",
            PromptAsset(
                "cadgenbench-linear-actor",
                LINEAR_ACTOR_TERMINAL_PROMPT,
            ),
        ),
    ),
    shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
    terminal_control=PromptAsset(
        "linear-terminal-control",
        LINEAR_TERMINAL_CONTROL_INSTRUCTIONS,
    ),
    system_overlay=PromptAsset(
        "cadgenbench-system-overlay",
        CADGENBENCH_TERMINAL_OVERLAY.system_prompt,
    ),
    compaction=PromptAsset(
        "linear-compaction",
        LINEAR_COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=CADGENBENCH_TERMINAL_OVERLAY.name,
)

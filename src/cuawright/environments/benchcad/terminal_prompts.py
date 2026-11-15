from __future__ import annotations

from cuawright.config.prompts import (
    ACTOR_INTERFACE_ADDITION,
    ACTIVE_PROMPT_TRANSFORMS,
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

from cuawright.environments.benchcad.prompts import (
    ACTOR_SYSTEM_PROMPT,
    BINDER_SYSTEM_PROMPT,
    critic_system_prompt,
    verifier_system_prompt,
)


_CACHE_TOKEN = "__CACHE_PATH__"


def _replace(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    rendered = source
    for old, new in replacements:
        if rendered.count(old) != 1:
            raise RuntimeError(f"BenchCAD prompt source changed around {old!r}")
        rendered = rendered.replace(old, new, 1)
    return rendered


_ACTOR_TERMINAL_CORE = _replace(
    ACTOR_SYSTEM_PROMPT,
    (
        (
            "Use run_command for ordinary Bash and Python and read_image for "
            "the target\ncomposite or candidate renders.",
            "Use run_command for ordinary Bash and Python. Inspect the target "
            "composite or candidate renders with `python "
            "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`.",
        ),
        (
            "Call submit_task when the persisted result is ready.",
            "Run `python /opt/cuawright-tools/submit.py` when the "
            "persisted result is ready and no Verifier `repair` remains.",
        ),
    ),
)

ACTOR_TERMINAL_PROMPT = "\n\n".join(
    (
        _ACTOR_TERMINAL_CORE,
        ACTIVE_PROMPT_TRANSFORMS["orchestrator"].render(),
        ACTOR_INTERFACE_ADDITION,
        ASK_USER_TERMINAL_INSTRUCTION,
    )
)

LINEAR_ACTOR_TERMINAL_PROMPT = (
    _replace(
        ACTOR_SYSTEM_PROMPT,
        (
            (
                "Complete the task and prove the supplied bindings.",
                "Complete the task and directly verify every requirement.",
            ),
            (
                "Use run_command for ordinary Bash and Python and read_image for "
                "the target\ncomposite or candidate renders.",
                "Use run_command for ordinary Bash and Python. Inspect the target "
                "composite or candidate renders with `python "
                "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`.",
            ),
            (
                "Work linearly through the ordered bindings:",
                "Work linearly through the task requirements:",
            ),
            (
                "Verify the canonical final program,",
                "Directly verify the canonical final program,",
            ),
            (
                "Call submit_task when the persisted result is ready.",
                "Run `python /opt/cuawright-tools/submit.py` when the persisted "
                "result is ready.",
            ),
        ),
    )
    + "\n\n"
    + ASK_USER_TERMINAL_INSTRUCTION
)

BINDER_TERMINAL_PROMPT = _replace(
    BINDER_SYSTEM_PROMPT,
    (
        (
            "Use run_command for ordinary Bash and Python inside the container and "
            "read_image\nfor /input/input.png or candidate renders.",
            "Use run_command for ordinary Bash and Python inside the container and "
            "`python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH` for "
            "/input/input.png or candidate renders.",
        ),
        (
            "Finish by calling submit_bindings.",
            "Return exactly one JSON object with `status`, `report`, and "
            "`unresolved`. Do not wrap it in Markdown fences.",
        ),
    ),
) + (
    "\n\nUse status `complete` or `blocked`, a non-empty report string, and an "
    "unresolved string list.\n\n" + ASK_USER_TERMINAL_INSTRUCTION
)

CRITIC_TERMINAL_PROMPT = _replace(
    critic_system_prompt(_CACHE_TOKEN),
    (
        (
            f"Put temporary artifacts under {_CACHE_TOKEN}.",
            "Put temporary artifacts under the cache path supplied in the user "
            "message.",
        ),
    ),
) + (
    "\n\nThe validated Binder report is already in your initial message. Return "
    "exactly one JSON object with `status`, `report`, and `unresolved`. Do not "
    "wrap it in Markdown fences. Use status `accept`, `concerns`, or `blocked`, "
    "a non-empty report string, and an unresolved string list.\n\n"
    + ASK_USER_TERMINAL_INSTRUCTION
)

VERIFIER_TERMINAL_PROMPT = _replace(
    verifier_system_prompt(_CACHE_TOKEN),
    (
        (
            "final Binder requirements.",
            "requirements derived directly from the original instruction.",
        ),
        (
            f"Put temporary artifacts only under {_CACHE_TOKEN}.",
            "Put temporary artifacts only under the cache path supplied in the user "
            "message.",
        ),
        (
            "Report every binding exactly once as pass, fail, or unknown.",
            "Report every independently checkable requirement exactly once as pass, "
            "fail, or unknown.",
        ),
    ),
) + (
    "\n\nDerive checks independently from the original instruction and live "
    "candidate. No Binder or Critic report is included. Return exactly one JSON "
    "object with `status`, `report`, and `unresolved`; do not wrap it in Markdown "
    "fences. Use status `ready`, `repair`, or `unknown`, a non-empty report string, "
    "and an unresolved string list. Return `repair` only for a direct objective "
    "contradiction in canonical candidate state; missing proof or inferred "
    "preference is `unknown`."
)

BENCHCAD_TERMINAL_OVERLAY = BenchmarkOverlay(
    name="benchcad",
    system_prompt=(
        "Benchmark overlay: BenchCAD Vision2Code. Public input is under /input; "
        "persistent work is under /workspace. Use benchcad-run and "
        "benchcad-render only through run_command."
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

BENCHCAD_FULL_PROMPT_BUNDLE = PromptBundle(
    name="benchcad-full",
    benchmark="benchcad",
    condition="full",
    composition="terminal",
    agent_topology="orchestrated",
    workspace=BENCHCAD_TERMINAL_OVERLAY.workspace,
    role_assets=tuple(
        (
            role,
            PromptAsset(
                f"benchcad-full-{role}",
                BENCHCAD_TERMINAL_OVERLAY.role_instructions[role],
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
        "benchcad-full-system-overlay",
        BENCHCAD_TERMINAL_OVERLAY.system_prompt,
    ),
    compaction=PromptAsset(
        "orchestrated-compaction",
        COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=BENCHCAD_TERMINAL_OVERLAY.name,
)

BENCHCAD_FULL_LINEAR_PROMPT_BUNDLE = PromptBundle(
    name="benchcad-full-linear",
    benchmark="benchcad",
    condition="full-linear",
    composition="terminal",
    agent_topology="linear",
    workspace=BENCHCAD_TERMINAL_OVERLAY.workspace,
    role_assets=(
        (
            "actor",
            PromptAsset(
                "benchcad-full-linear-actor",
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
        "benchcad-full-system-overlay",
        BENCHCAD_TERMINAL_OVERLAY.system_prompt,
    ),
    compaction=PromptAsset(
        "linear-compaction",
        LINEAR_COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=BENCHCAD_TERMINAL_OVERLAY.name,
)

BENCHCAD_DIRECT_PROMPT_BUNDLE = PromptBundle(
    name="benchcad-direct",
    benchmark="benchcad",
    condition="direct",
    composition="direct",
    workspace="/workspace/.agent",
    direct_system=PromptAsset("benchcad-direct-system", None),
    direct_user=PromptAsset("benchcad-direct-user", None),
)

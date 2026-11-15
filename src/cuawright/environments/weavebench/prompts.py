from __future__ import annotations

from typing import Mapping

from cuawright.config.prompts import (
    COMPACTION_HANDOFF_REQUEST,
    LINEAR_ACTOR_ROLE_INSTRUCTIONS,
    LINEAR_COMPACTION_HANDOFF_REQUEST,
    LINEAR_TERMINAL_CONTROL_INSTRUCTIONS,
    ROLE_INSTRUCTIONS,
    SHARED_SYSTEM_PREFIX,
    TERMINAL_CONTROL_INSTRUCTIONS,
    AgentRole,
    PromptAsset,
    PromptBundle,
)

# WeaveBench's external prompt is translated byte-for-byte at runtime. Its
# compatibility-localized role source is composed directly here and never
# inherits an OSWorld overlay or named OSWorld prompt bundle.

WEAVEBENCH_EXTERNAL_SYSTEM_ASSET = "weavebench-external-translated-system"

_ROLE_REPLACEMENTS = {
    "actor": (
        "You are Actor for an OSWorld V2 task running in an\nUbuntu desktop VM.",
        "You are Actor for a WeaveBench task running in an\nUbuntu desktop VM.",
    ),
    "binder": (
        "the main Actor on an OSWorld V2 task.",
        "the main Actor on a WeaveBench task.",
    ),
    "critic": (
        "the main Actor on an OSWorld V2 task.",
        "the main Actor on a WeaveBench task.",
    ),
    "verifier": (
        "the main Actor on an OSWorld V2 task.",
        "the main Actor on a WeaveBench task.",
    ),
}


def _replace(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected 1 exact occurrence, found {count}")
    return text.replace(old, new, 1)


def localized_role_instructions(
    source: Mapping[AgentRole, str] = ROLE_INSTRUCTIONS,
) -> dict[AgentRole, str]:
    if set(source) != set(_ROLE_REPLACEMENTS):
        raise RuntimeError("CUAWright role map changed; re-audit WeaveBench overlay")
    return {
        role: _replace(
            prompt,
            *_ROLE_REPLACEMENTS[role],
            f"CUAWright {role} role localization",
        )
        for role, prompt in source.items()
    }


def localized_linear_actor_instructions(
    source: str = LINEAR_ACTOR_ROLE_INSTRUCTIONS,
) -> str:
    return _replace(
        source,
        *_ROLE_REPLACEMENTS["actor"],
        "CUAWright linear actor role localization",
    )


def translate_system_prompt(
    prompt: str,
    *,
    client_password: str = "password",
) -> str:
    """Translate only OpenClaw vocabulary in WeaveBench's audited GUI prompt."""
    replacements = [
        (
            "Built-in helper tools available in BOTH modes: `exec` (shell), "
            "`read`/`write`/`edit` (files), `process` (long-running jobs), "
            "`pdf` (structured PDF extraction including tables), `web_fetch` "
            "(HTTP), `memory_search` (recall prior context). Pick whichever is "
            "most direct for the sub-step.\n",
            "The sole model-visible tool is `run_command`. Use it for shell and "
            "file operations, and use ordinary terminal programs or Python "
            "libraries for documents and network access when the task permits.\n",
            "helper tools",
        ),
        (
            "The OpenClaw framework's own `## Workspace` section in the "
            "underlying system prompt has been configured to point at the SAME "
            "`/tmp_workspace/` path (via `agents.defaults.workspace`), so the "
            "two sources agree. Anything you write under "
            "`/home/user/.openclaw/...` is OpenClaw's internal agent state — "
            "invisible to the grader and effectively lost.\n",
            "CUAWright control state lives under `/home/user/.agent/`, outside "
            "the benchmark workspace and invisible to grading. Keep all "
            "evaluator-facing inputs and outputs under `/tmp_workspace/` unless "
            "the task explicitly requests another absolute path.\n",
            "workspace",
        ),
        (
            "(use `exec` like `ls -la <path>` or `read` a few of the created files).",
            "(use `run_command` with commands such as `ls -la <path>` and narrow content inspection).",
            "output discipline",
        ),
        (
            "with the `image` tool, or call `__computer__` with `actions:[]` "
            "(empty batch) to grab a fresh capture.",
            "with `python /opt/cuawright-tools/image_show.py --path "
            "/tmp/init_screenshot.png` as the only command in its `run_command` "
            "call; capture fresh screenshots from `DISPLAY=:0` with pyautogui "
            "or `gnome-screenshot`.",
            "initial screenshot",
        ),
    ]
    for old, new, label in replacements:
        prompt = _replace(prompt, old, new, f"WeaveBench {label} translation")

    gui_start = (
        "- GUI is exposed via ONE function tool: `__computer__`. It runs one or more "
        "GUI actions"
    )
    gui_end = "- Coordinates are absolute screen pixels in the "
    if prompt.count(gui_start) != 1 or prompt.count(gui_end) != 1:
        raise RuntimeError("WeaveBench GUI instructions changed; re-audit translation")
    start = prompt.index(gui_start)
    end = prompt.index(gui_end, start)
    terminal_gui = (
        "- The sole model-visible tool remains `run_command`. Drive the real desktop on "
        "`DISPLAY=:0` through terminal-launched pyautogui, xdotool, AT-SPI/accessibility "
        "inspection, and ordinary application commands. Do not merely describe GUI actions.\n"
        "- Observe and act iteratively. Capture the real desktop to a file with pyautogui "
        "or `gnome-screenshot`, then inspect it by running `python "
        "/opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH` as the only command in "
        "its `run_command` call. An observation alone does not complete the task; continue "
        "with concrete actions unless the requested persistent result is already verified.\n"
        "- Many tasks require both GUI state and files. Use terminal commands to drive the "
        "GUI, capture genuine evidence, create the requested artifacts, and verify every "
        "required filename under `/tmp_workspace/` before finishing.\n"
    )
    prompt = prompt[:start] + terminal_gui + prompt[end:]
    replacements = [
        (
            "This run has a real Linux desktop available via the `__computer__` tool.",
            "This run has a real Linux desktop available through terminal-driven "
            "pyautogui, xdotool, and accessibility commands.",
            "anti-fabrication desktop",
        ),
        (
            "  • `__computer__` — preferred. Every call already auto-screenshots\n"
            "    the resulting desktop; you can save that buffer to disk.\n",
            "  • `pyautogui.screenshot()` or `gnome-screenshot` against the real\n"
            "    `DISPLAY=:0`, followed by image_show.py for visual inspection.\n",
            "anti-fabrication capture",
        ),
        (
            "or `__computer__` keeps failing",
            "or real-display screenshot capture keeps failing",
            "anti-fabrication failure",
        ),
        (
            'The sudo password is "password" when sudo is needed.',
            f'The sudo password is "{client_password}" when sudo is needed.',
            "VM password",
        ),
    ]
    for old, new, label in replacements:
        prompt = _replace(prompt, old, new, f"WeaveBench {label} translation")
    forbidden = (
        "`exec`",
        "`read`",
        "`write`",
        "`edit`",
        "`process`",
        "`pdf`",
        "`web_fetch`",
        "`memory_search`",
        "`image`",
        "__computer__",
    )
    remaining = [name for name in forbidden if name in prompt]
    if remaining:
        raise RuntimeError(
            "Unavailable tools remain advertised: " + ", ".join(remaining)
        )
    return prompt


def weavebench_prompt_bundle_templates(
    *,
    role_instructions: Mapping[AgentRole, str] | None = None,
    linear_actor_instructions: str | None = None,
) -> tuple[PromptBundle, PromptBundle]:
    roles = role_instructions or localized_role_instructions()
    linear_actor = linear_actor_instructions or localized_linear_actor_instructions()
    system_overlay = PromptAsset(WEAVEBENCH_EXTERNAL_SYSTEM_ASSET, None)
    default = PromptBundle(
        name="weavebench-default",
        benchmark="weavebench",
        condition="default",
        composition="terminal",
        agent_topology="orchestrated",
        workspace="/home/user/.agent",
        role_assets=tuple(
            (
                role,
                PromptAsset(f"weavebench-default-{role}", roles[role]),
            )
            for role in ("actor", "binder", "critic", "verifier")
        ),
        shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
        terminal_control=PromptAsset(
            "orchestrated-terminal-control",
            TERMINAL_CONTROL_INSTRUCTIONS,
        ),
        system_overlay=system_overlay,
        compaction=PromptAsset(
            "orchestrated-compaction",
            COMPACTION_HANDOFF_REQUEST,
        ),
        overlay_name="weavebench",
    )
    linear = PromptBundle(
        name="weavebench-linear",
        benchmark="weavebench",
        condition="linear",
        composition="terminal",
        agent_topology="linear",
        workspace="/home/user/.agent",
        role_assets=(
            (
                "actor",
                PromptAsset(
                    "weavebench-linear-actor",
                    linear_actor,
                ),
            ),
        ),
        shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
        terminal_control=PromptAsset(
            "linear-terminal-control",
            LINEAR_TERMINAL_CONTROL_INSTRUCTIONS,
        ),
        system_overlay=system_overlay,
        compaction=PromptAsset(
            "linear-compaction",
            LINEAR_COMPACTION_HANDOFF_REQUEST,
        ),
        overlay_name="weavebench",
    )
    return default, linear


(
    WEAVEBENCH_DEFAULT_PROMPT_BUNDLE,
    WEAVEBENCH_LINEAR_PROMPT_BUNDLE,
) = weavebench_prompt_bundle_templates()

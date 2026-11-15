from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

from cuawright.models.base import RUN_COMMAND_TOOL_SCHEMA

BINDER_CALL_MAX = 40
CRITIC_CALL_MAX = 32
VERIFIER_CALL_MAX = 24


ACTOR_SYSTEM_PROMPT = """You are Actor for an OSWorld V2 task running in an
Ubuntu desktop VM. Complete the task and prove the supplied bindings.

Use run_command for ordinary VM Bash and read_image for VM image paths.
`osw "search terms"` searches the local procedure knowledge base. Use ordinary
Bash, Python, and installed applications for all task actions.
Before broad recursive scans, AppImage extraction, mass page/frame rendering,
large archive expansion, or large package installation, confirm the operation
is necessary and check guest free space. Prefer narrow paths, metadata,
streaming/incremental processing, and representative samples. Do not extract an
AppImage merely to run it, and clean task-created temporary artifacts after use.
Prefer native application surfaces when application state matters. For Chrome,
consult the knowledge base for CDP and task-local Playwright setup; the guest
system Python may not contain Playwright. Prefer accessibility roles/names over
brittle raw DOM selectors.
Prefer reliable native commands or direct artifact operations when appropriate,
and prefer accessibility-element discovery and actions over fragile coordinate
clicking. Keyboard shortcuts through xdotool are acceptable when needed.
Coordinate-based xdotool/pyautogui clicking is a last resort for surfaces such
as games or inaccessible canvases; calibrate the active window/screenshot and
confirm each state transition when using it.
Directly manipulate files when the evaluated deliverable is fundamentally an
artifact. Reopen externally edited files in GIMP, LibreOffice, WPS, or similar
applications when stale state or file locks may matter. Save exact requested
paths. Do not bypass browser work by editing hidden backend state or application
databases. Do not read hidden backend state, cookies, or private API endpoints
to infer or verify completion.

Stay on the application, website, account, and task state opened by setup. Do
not replace it with a similar local app, `chrome://` page, fresh browser
profile, or different user session. If the task forbids CDP/developer tools,
use only visible UI input, screenshots, AT-SPI, and ordinary application
controls.

Prefer canonical persistent state: the initialized app/account, exact requested
path/name/format, native editable structure, minimal baseline-preserving changes,
strict syntax, and one clean result without duplicates or stale alternatives.
Among equally valid results, choose the simplest one that survives reload and
can be directly proven.
Ensure underlying formulas, links, layers, relationships, charts, or native
objects truly exist rather than merely looking correct. Validate each upstream
artifact before downstream export/import. Use a named application when required;
if editing externally, reopen the result there. Verify the exact scored surface
and remove abandoned outputs before submission.

Work linearly through the ordered bindings. For each binding, prefer:
inspect → change → save → reload/reopen → read back the canonical result.
Source-analysis bindings may be handled during inspection. When asked for a
minimum deliverable, move to the earliest ordered binding that changes an
evaluator-facing file or application state rather than continuing setup.

Use concise commands and narrow output. A candidate that visibly conflicts with
the reference is disproven, not "close enough"; continue investigating and test
alternatives systematically. Create a provisional evaluator-facing deliverable
by the midpoint of the execution budget, then improve it; do not spend the whole
run inspecting without writing the requested output. Verify every binding before
submission. During verification, inspect the canonical final application/file
state and remove duplicate, stale, or conflicting artifacts left by abandoned
attempts.
Phrases such as "looks about right", "should be fine", "probably correct", or
similar are never completion evidence. For objective bindings, obtain direct
readback. For subjective visual bindings, inspect the actual final render,
identify concrete matched and missing features, and preserve uncertainty rather
than declaring success without support.
Call submit_task when complete. If a required exact fact is absent,
contradictory, or only the user can provide it, call ask_user with one concise
question rather than guessing. Also call ask_user when a required action is
otherwise impossible without user-controlled information, authorization,
resources, a missing source/attachment, or a decision that cannot be inferred
safely from the task state. Do not ask for facts recoverable from the VM, for
optional polish, or for hidden evaluator preferences. Ask at most once for the
same blocker, then use the safest task-faithful fallback if no further
information is available."""


ORCHESTRATOR_SYSTEM_PROMPT = (
    ACTOR_SYSTEM_PROMPT
    + """

You are also the main orchestrator. You own the task from first inspection
through final submission, are the only role allowed to modify judge-relevant
state, and remain authoritative over all subagent advice.

Call the Binder near the start. Binder inspects task-visible evidence and
returns a concise acceptance brief with exact facts, canonical targets, and
unresolved questions; it does not perform the task.

Execute the task yourself using the ordinary VM tools. By roughly 50% of the
shared model-call budget, create a real persistent evaluator-facing artifact or
application state rather than continuing pure investigation.

When a candidate result is saved and reloaded, call the Verifier. Verifier
independently and read-only inspects the actual canonical state against the
original instruction. It receives no Binder report and must derive its own
checks. A `repair` report blocks normal submission: repair the direct objective
failure, save/reload again, and call a fresh Verifier. An `unknown` report does
not justify changing known-good state; preserve the best directly proven result
and submit only after your own final canonical readback. Verifier never repairs
or submits.

Binder, Verifier, compaction, and your own turns all consume one shared hard
budget. Choose Binder budget deliberately. For execution-heavy tasks such as
games or long visual manipulation, keep Binder brief so task execution retains
the budget it needs. Binder may be called only once.
Repair directly evidenced failures when useful. Insufficient evidence remains
unresolved and is not proof of readiness. If a concrete issue is unrecoverable,
preserve the best task-faithful state for the final forced submission.
Before applying Verifier feedback, preserve a reversible backup or checkpoint
of each canonical file or local state that will change when doing so is safe
and task-faithful. After repair, call a fresh Verifier. Roll back when the repair
introduces a new directly evidenced failure or leaves its target failure
unresolved; `unknown` alone is not evidence that known-good state should be
replaced.
You may call Binder once and Verifier at most three times per task, or
per phase when the task explicitly advances through multiple sequential phases.

No structured bindings are supplied at startup in this mode. Where the
baseline Actor instructions mention supplied or ordered bindings, use the
latest Binder report as your planning input.

Do not create a self-authored completion ledger or treat subagent prose as
proof. The live saved/reloaded task state remains authoritative."""
)


BINDER_SUBAGENT_SYSTEM_PROMPT = """You are a fresh Binder subagent supporting
the main Actor on an OSWorld V2 task. Inspect task-visible evidence and return a
concise checklist of how a typical evaluator would vet the final result.

For each independently checkable requirement, state what must be true, the
canonical surface to inspect, and the final persistent transition such as save,
export, import, send, sign, or submit. Apply only recurring public evaluator
preferences: the initialized app/account/profile, exact requested paths and
serialization, native editable structure, complete sets without extras,
saved/reloaded persistence, baseline preservation, and ordered handoffs. Do not
transfer task-specific values, paths, examples, or inferred rules from other
tasks. State an exact fact only when direct evidence in this task uniquely
supports it; otherwise keep it unresolved.

Do not perform the task or modify likely final output locations. Use ordinary
VM inspection tools, keep temporary artifacts only in the cache path supplied
in the user message, never access evaluator code or hidden answers, and finish
by calling submit_subagent_report."""


CRITIC_SUBAGENT_SYSTEM_PROMPT = """You are a fresh Critic subagent supporting
the main Actor on an OSWorld V2 task. Independently challenge the Binder report
and current task-visible evidence for unsupported assumptions, omissions,
incorrect exact values, wrong native representations, stale paths or profiles,
extra artifacts, and incomplete persistent handoffs.

Identify only the highest-impact unresolved assumption that could materially
change evaluator-facing state, and propose the cheapest direct check available.
Do not enumerate speculative alternatives or delay execution when task-visible
evidence cannot distinguish them.

Do not perform the task or perturb judge-relevant state. Use ordinary VM
inspection tools, keep temporary artifacts only in the cache path supplied in
the user message, never infer hidden evaluator preferences, and finish by
calling submit_subagent_report."""


VERIFIER_SUBAGENT_SYSTEM_PROMPT = """You are a fresh read-only Verifier
subagent supporting the main Actor on an OSWorld V2 task. Inspect the actual
saved and reloaded canonical state against the original instruction, report
direct evidence, and prioritize only concrete repairable failures.

Derive checks independently from the original instruction and live state. You
receive no Binder report, and the Actor request is not evidence that its
candidate or expected values are correct. Independently check the initialized
app/account/profile, exact path and serialization, native structure, complete
sets, persistence, baseline preservation, duplicates, and ordered handoffs
where relevant. Prefer the canonical persistent representation an evaluator can
read over transient UI wording or appearance.

Return `repair` only for a direct objective contradiction in canonical
evaluator-facing state. Before recommending repair, state the directly proven
passes that must remain unchanged. Missing proof, subjective concern, a UI-only
discrepancy, an alternative task-faithful representation, or an inferred hidden
preference is `unknown`, not `repair`. Return `ready` when every objective
requirement has direct evidence.

Do not modify judge-relevant files, application state, accounts, browser state,
or outputs. Keep temporary artifacts only in the cache path supplied in the
user message, never access evaluator code or hidden answers, and finish by
calling submit_subagent_report."""


ORCHESTRATOR_COMPACTION_REQUEST = """Do not call a tool. Produce a concise
handoff for a fresh main-Actor context using exactly these headings:

Current stage:
Subagents called and findings to retain:
Completed persistent work:
Canonical artifacts/state and readback:
Current blocker or failed assumptions:
Remaining work:
Next tool call:

Preserve exact task facts, paths, identifiers, unresolved risks, the latest
Verifier feedback, and which subagents have already been called. Do not include
large raw command output or claim completion without direct saved-state proof."""


AgentRole = Literal["actor", "binder", "critic", "verifier"]
SpecialistRole = Literal["binder", "critic", "verifier"]
AgentTopology = Literal["orchestrated", "linear"]
PromptComposition = Literal["terminal", "direct"]
ChangeKind = Literal[
    "terminal-interface",
    "generic-benchmark",
    "path-coordination",
]


@dataclass(frozen=True, slots=True)
class PriorReportContent:
    status: str
    report: str
    unresolved: tuple[str, ...]

    def __post_init__(self) -> None:
        _required_text("status", self.status)
        _required_text("report", self.report)
        if not isinstance(self.unresolved, tuple) or any(
            not isinstance(item, str) for item in self.unresolved
        ):
            raise ValueError("unresolved must be a tuple of strings")


AGENT_WORKSPACE: Final = "/workspace/.agent"
SPECIALIST_CALL_MAX: Final[Mapping[SpecialistRole, int]] = {
    "binder": BINDER_CALL_MAX,
    "critic": CRITIC_CALL_MAX,
    "verifier": VERIFIER_CALL_MAX,
}


@dataclass(frozen=True, slots=True)
class PromptAsset:
    """One immutable prompt source, optionally bound from benchmark input."""

    name: str
    text: str | None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("prompt asset name cannot be empty")
        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("prompt asset text must be a string or None")

    def bind(self, text: str) -> PromptAsset:
        if self.text is not None:
            raise ValueError(f"prompt asset {self.name!r} is already bound")
        if not isinstance(text, str):
            raise ValueError("bound prompt asset text must be a string")
        return replace(self, text=text)

    def render(self, *, workspace: str | None = None) -> str:
        if self.text is None:
            raise ValueError(f"prompt asset {self.name!r} requires external text")
        if workspace is None:
            return self.text
        return self.text.replace(AGENT_WORKSPACE, workspace)


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """A complete named benchmark-condition prompt composition."""

    name: str
    benchmark: str
    condition: str
    composition: PromptComposition
    workspace: str
    agent_topology: AgentTopology | None = None
    role_assets: tuple[tuple[AgentRole, PromptAsset], ...] = ()
    shared_prefix: PromptAsset | None = None
    terminal_control: PromptAsset | None = None
    system_overlay: PromptAsset | None = None
    compaction: PromptAsset | None = None
    direct_system: PromptAsset | None = None
    direct_user: PromptAsset | None = None
    overlay_name: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("bundle name", self.name),
            ("benchmark name", self.benchmark),
            ("condition name", self.condition),
        ):
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        if not self.workspace.startswith("/") or not self.workspace.endswith(".agent"):
            raise ValueError("prompt bundle workspace must be an absolute .agent path")
        role_names = tuple(role for role, _ in self.role_assets)
        if len(role_names) != len(set(role_names)):
            raise ValueError("prompt bundle role assets must be unique")
        if self.composition == "terminal":
            if self.agent_topology not in {"orchestrated", "linear"}:
                raise ValueError("terminal prompt bundle requires an agent topology")
            expected_roles = (
                ("actor", "binder", "critic", "verifier")
                if self.agent_topology == "orchestrated"
                else ("actor",)
            )
            if role_names != expected_roles:
                raise ValueError(
                    "terminal prompt bundle role order must match its topology"
                )
            if any(
                asset is None
                for asset in (
                    self.shared_prefix,
                    self.terminal_control,
                    self.system_overlay,
                    self.compaction,
                )
            ):
                raise ValueError("terminal prompt bundle is missing a prompt layer")
            if self.direct_system is not None or self.direct_user is not None:
                raise ValueError("terminal prompt bundle cannot contain direct prompts")
        elif self.composition == "direct":
            if self.agent_topology is not None:
                raise ValueError("direct prompt bundle cannot have an agent topology")
            if self.role_assets or any(
                asset is not None
                for asset in (
                    self.shared_prefix,
                    self.terminal_control,
                    self.system_overlay,
                    self.compaction,
                )
            ):
                raise ValueError("direct prompt bundle cannot contain terminal layers")
            if self.direct_system is None or self.direct_user is None:
                raise ValueError(
                    "direct prompt bundle requires system and user prompts"
                )
        else:
            raise ValueError(f"unsupported prompt composition: {self.composition!r}")

    @property
    def unbound_asset_names(self) -> tuple[str, ...]:
        return tuple(asset.name for asset in self._assets() if asset.text is None)

    def bind_external_assets(self, values: Mapping[str, str]) -> PromptBundle:
        expected = set(self.unbound_asset_names)
        provided = set(values)
        if provided != expected:
            raise ValueError(
                "external prompt assets must match exactly; expected "
                f"{sorted(expected)!r}, received {sorted(provided)!r}"
            )

        def bound(asset: PromptAsset | None) -> PromptAsset | None:
            if asset is None or asset.text is not None:
                return asset
            return asset.bind(values[asset.name])

        return replace(
            self,
            role_assets=tuple((role, bound(asset)) for role, asset in self.role_assets),
            shared_prefix=bound(self.shared_prefix),
            terminal_control=bound(self.terminal_control),
            system_overlay=bound(self.system_overlay),
            compaction=bound(self.compaction),
            direct_system=bound(self.direct_system),
            direct_user=bound(self.direct_user),
        )

    def role_asset(self, role: AgentRole) -> PromptAsset:
        for asset_role, asset in self.role_assets:
            if asset_role == role:
                return asset
        raise ValueError(f"prompt bundle {self.name!r} has no role {role!r}")

    def _assets(self) -> tuple[PromptAsset, ...]:
        return (
            *(asset for _, asset in self.role_assets),
            *(
                asset
                for asset in (
                    self.shared_prefix,
                    self.terminal_control,
                    self.system_overlay,
                    self.compaction,
                    self.direct_system,
                    self.direct_user,
                )
                if asset is not None
            ),
        )


@dataclass(frozen=True, slots=True)
class AuditedReplacement:
    old: str
    new: str
    kind: ChangeKind
    reason: str
    expected_count: int = 1

    def __post_init__(self) -> None:
        if not self.old:
            raise ValueError("replacement source text cannot be empty")
        if not self.reason.strip():
            raise ValueError("replacement reason cannot be empty")
        if self.expected_count < 1:
            raise ValueError("replacement expected_count must be positive")


@dataclass(frozen=True, slots=True)
class SourcePromptTransform:
    """The complete active production prompt plus localized replacements."""

    name: str
    source_text: str
    replacements: tuple[AuditedReplacement, ...]

    def render(self) -> str:
        rendered = self.source_text
        for replacement in self.replacements:
            count = rendered.count(replacement.old)
            if count != replacement.expected_count:
                raise RuntimeError(
                    f"{self.name}: expected {replacement.expected_count} "
                    f"occurrence(s) of audited source text, found {count}"
                )
            rendered = rendered.replace(
                replacement.old,
                replacement.new,
                replacement.expected_count,
            )
        return rendered


def _replacement(
    old: str,
    new: str,
    kind: ChangeKind,
    reason: str,
    *,
    expected_count: int = 1,
) -> AuditedReplacement:
    return AuditedReplacement(
        old=old,
        new=new,
        kind=kind,
        reason=reason,
        expected_count=expected_count,
    )


_ORCHESTRATOR_EXTENSION = ORCHESTRATOR_SYSTEM_PROMPT.removeprefix(
    ACTOR_SYSTEM_PROMPT
).strip()
if ORCHESTRATOR_SYSTEM_PROMPT != (
    ACTOR_SYSTEM_PROMPT + "\n\n" + _ORCHESTRATOR_EXTENSION
):
    raise RuntimeError("production orchestrator prompt is not Actor plus extension")


ACTIVE_PROMPT_TRANSFORMS: Final[Mapping[str, SourcePromptTransform]] = {
    "actor": SourcePromptTransform(
        name="actor",
        source_text=ACTOR_SYSTEM_PROMPT,
        replacements=(
            _replacement(
                "Use run_command for ordinary VM Bash and read_image for VM image "
                "paths.",
                "Use run_command for ordinary terminal commands. To inspect an "
                "image path, run `python /opt/cuawright-tools/image_show.py "
                "--path ABSOLUTE_PATH`.",
                "terminal-interface",
                "run_command is the sole model-visible tool.",
            ),
            _replacement(
                "Call submit_task when complete.",
                "When complete, after Verifier inspection and with no outstanding "
                "`repair` report, run `python /opt/cuawright-tools/submit.py`.",
                "terminal-interface",
                "Submission is controlled through the terminal.",
            ),
            _replacement(
                "call ask_user with one concise\nquestion rather than guessing.",
                "run `python /opt/cuawright-tools/ask_user.py --question "
                '"TEXT"` as the only command in its run_command call, with one '
                "concise question, rather than guessing.",
                "terminal-interface",
                "User questions are controlled through the terminal.",
            ),
            _replacement(
                "Also call ask_user when a required action is",
                "Also use that command when a required action is",
                "terminal-interface",
                "Retain the production blocker policy with the terminal command.",
            ),
        ),
    ),
    "orchestrator": SourcePromptTransform(
        name="orchestrator",
        source_text=_ORCHESTRATOR_EXTENSION,
        replacements=(
            _replacement(
                "Execute the task yourself using the ordinary VM tools.",
                "Execute the task yourself using ordinary terminal commands.",
                "terminal-interface",
                "The sole tool now carries terminal commands.",
            ),
        ),
    ),
    "binder": SourcePromptTransform(
        name="binder",
        source_text=BINDER_SUBAGENT_SYSTEM_PROMPT,
        replacements=(
            _replacement(
                "Use ordinary\nVM inspection tools,",
                "Use run_command,",
                "terminal-interface",
                "The same sole terminal tool remains available.",
            ),
            _replacement(
                "finish\nby calling submit_subagent_report.",
                "finish by returning exactly one JSON object with `status`, "
                "`report`, and `unresolved`. Do not wrap it in Markdown fences.",
                "terminal-interface",
                "The host captures the specialist response as terminal output.",
            ),
        ),
    ),
    "critic": SourcePromptTransform(
        name="critic",
        source_text=CRITIC_SUBAGENT_SYSTEM_PROMPT,
        replacements=(
            _replacement(
                "Use ordinary VM\ninspection tools,",
                "Use run_command,",
                "terminal-interface",
                "The same sole terminal tool remains available.",
            ),
            _replacement(
                "finish by\ncalling submit_subagent_report.",
                "finish by returning exactly one JSON object with `status`, "
                "`report`, and `unresolved`. Do not wrap it in Markdown fences.",
                "terminal-interface",
                "The host captures the specialist response as terminal output.",
            ),
        ),
    ),
    "verifier": SourcePromptTransform(
        name="verifier",
        source_text=VERIFIER_SUBAGENT_SYSTEM_PROMPT,
        replacements=(
            _replacement(
                "finish by\ncalling submit_subagent_report.",
                "finish by returning exactly one JSON object with `status`, "
                "`report`, and `unresolved`. Do not wrap it in Markdown fences.",
                "terminal-interface",
                "The host captures the specialist response as terminal output.",
            ),
        ),
    ),
    "compaction": SourcePromptTransform(
        name="compaction",
        source_text=ORCHESTRATOR_COMPACTION_REQUEST,
        replacements=(
            _replacement(
                "Do not call a tool.",
                "Do not call run_command.",
                "terminal-interface",
                "run_command is the sole model-visible tool.",
            ),
            _replacement(
                "Next tool call:",
                "Next terminal command:",
                "terminal-interface",
                "The next action is expressed as a terminal command.",
            ),
        ),
    ),
}

LINEAR_ACTOR_PROMPT_TRANSFORM = SourcePromptTransform(
    name="linear-actor",
    source_text=ACTOR_SYSTEM_PROMPT,
    replacements=(
        _replacement(
            "Complete the task and prove the supplied bindings.",
            "Complete the task and directly verify every requirement.",
            "generic-benchmark",
            "Linear mode receives the task directly without Binder output.",
        ),
        _replacement(
            "Use run_command for ordinary VM Bash and read_image for VM image "
            "paths.",
            "Use run_command for ordinary terminal commands. To inspect an "
            "image path, run `python /opt/cuawright-tools/image_show.py "
            "--path ABSOLUTE_PATH`.",
            "terminal-interface",
            "run_command is the sole model-visible tool.",
        ),
        _replacement(
            "Work linearly through the ordered bindings. For each binding, prefer:",
            "Work linearly through the task requirements. For each requirement, "
            "prefer:",
            "generic-benchmark",
            "Linear mode has task requirements rather than specialist bindings.",
        ),
        _replacement(
            "Source-analysis bindings may be handled during inspection.",
            "Source-analysis requirements may be handled during inspection.",
            "generic-benchmark",
            "Linear mode has no Binder-produced bindings.",
        ),
        _replacement(
            "move to the earliest ordered binding that changes an\n"
            "evaluator-facing file or application state",
            "move to the earliest task requirement that changes an\n"
            "evaluator-facing file or application state",
            "generic-benchmark",
            "Linear mode plans directly from the task instruction.",
        ),
        _replacement(
            "Verify every binding before\nsubmission.",
            "Verify every requirement before\nsubmission.",
            "generic-benchmark",
            "Linear mode verifies the task directly.",
        ),
        _replacement(
            "Call submit_task when complete.",
            "When complete, run `python /opt/cuawright-tools/submit.py`.",
            "terminal-interface",
            "Linear submission is controlled through the terminal without "
            "Verifier guidance.",
        ),
        _replacement(
            "call ask_user with one concise\nquestion rather than guessing.",
            "run `python /opt/cuawright-tools/ask_user.py --question "
            '"TEXT"` as the only command in its run_command call, with one '
            "concise question, rather than guessing.",
            "terminal-interface",
            "User questions are controlled through the terminal.",
        ),
        _replacement(
            "Also call ask_user when a required action is",
            "Also use that command when a required action is",
            "terminal-interface",
            "Retain the production blocker policy with the terminal command.",
        ),
    ),
)


SHARED_SYSTEM_PREFIX = ""

TERMINAL_CONTROL_INSTRUCTIONS = """Run every
`/opt/cuawright-tools/*.py` control client as the only command in its
run_command call. Do not combine a control client with another command,
redirection, a pipe, or shell substitution. Use `image_show.py --path
ABSOLUTE_PATH` to inspect an image."""

LINEAR_TERMINAL_CONTROL_INSTRUCTIONS = """The available control clients are
`python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`,
`python /opt/cuawright-tools/ask_user.py --question "TEXT"`, and
`python /opt/cuawright-tools/submit.py`. Run a control client as the only
command in its run_command call. Do not combine it with another command,
redirection, a pipe, or shell substitution."""

OSWORLD_SYSTEM_OVERLAY = ""

ACTOR_INTERFACE_ADDITION = """Invoke specialists with inline request text:
`python /opt/cuawright-tools/binder.py --request "TEXT" --max-steps N`,
or `python /opt/cuawright-tools/verifier.py --request "TEXT" --max-steps N`.
Each client must be the only command in its run_command call. Choose N
explicitly for the task-specific work needed. Valid role ranges are Binder
1-40 and Verifier 1-24; the range maxima are limits, not defaults, and the
shared remaining budget may be lower. The same run_command result returns the
report directly. Normal submission requires a latest Verifier status of
`ready` or `unknown`; `repair` requires additional Actor work and a fresh
Verifier. An `unknown` report must not trigger speculative mutation."""

ASK_USER_TERMINAL_INSTRUCTION = """If user input is genuinely required, run
`python /opt/cuawright-tools/ask_user.py --question "TEXT"` as the only command
in its run_command call. Put the concise question directly in the command."""

SPECIALIST_INTERFACE_ADDITIONS: Final[Mapping[SpecialistRole, str]] = {
    "binder": ASK_USER_TERMINAL_INSTRUCTION,
    "critic": ASK_USER_TERMINAL_INSTRUCTION,
    "verifier": "",
}

ROLE_INSTRUCTIONS: Final[Mapping[AgentRole, str]] = {
    "actor": "\n\n".join(
        (
            ACTIVE_PROMPT_TRANSFORMS["actor"].render(),
            ACTIVE_PROMPT_TRANSFORMS["orchestrator"].render(),
            ACTOR_INTERFACE_ADDITION,
        )
    ),
    "binder": "\n\n".join(
        (
            ACTIVE_PROMPT_TRANSFORMS["binder"].render(),
            SPECIALIST_INTERFACE_ADDITIONS["binder"],
        )
    ),
    "critic": "\n\n".join(
        (
            ACTIVE_PROMPT_TRANSFORMS["critic"].render(),
            SPECIALIST_INTERFACE_ADDITIONS["critic"],
        )
    ),
    "verifier": "\n\n".join(
        (
            ACTIVE_PROMPT_TRANSFORMS["verifier"].render(),
            SPECIALIST_INTERFACE_ADDITIONS["verifier"],
        )
    ),
}

ACTOR_ROLE_INSTRUCTIONS = ROLE_INSTRUCTIONS["actor"]
LINEAR_ACTOR_ROLE_INSTRUCTIONS = LINEAR_ACTOR_PROMPT_TRANSFORM.render()
BINDER_ROLE_INSTRUCTIONS = ROLE_INSTRUCTIONS["binder"]
CRITIC_ROLE_INSTRUCTIONS = ROLE_INSTRUCTIONS["critic"]
VERIFIER_ROLE_INSTRUCTIONS = ROLE_INSTRUCTIONS["verifier"]
COMPACTION_HANDOFF_REQUEST = ACTIVE_PROMPT_TRANSFORMS["compaction"].render()
COMPACTION_REQUEST = COMPACTION_HANDOFF_REQUEST
SOLE_TERMINAL_TOOL_SCHEMA: Final = RUN_COMMAND_TOOL_SCHEMA

OSWORLD_ACTOR_PROMPT_TRANSFORM = SourcePromptTransform(
    name="osworld-actor",
    source_text=ACTOR_ROLE_INSTRUCTIONS,
    replacements=(
        _replacement(
            "For Chrome,\nconsult the knowledge base for CDP and task-local "
            "Playwright setup; the guest\nsystem Python may not contain Playwright.",
            "For Chrome, first inspect and normally reuse the setup-opened session "
            "and task\ntab. When CDP is allowed and available, use "
            "`/home/user/.local/share/osw/venv/bin/python` with\n"
            "`http://127.0.0.1:1337` and select the page by URL or title; otherwise "
            "inspect\nthe live state and recover or launch Chrome as needed.",
            "generic-benchmark",
            "OSWorld provisions this interpreter and normally initializes Chrome "
            "before Actor starts, while some tasks may still require recovery.",
        ),
    ),
)

OSWORLD_LINEAR_ACTOR_PROMPT_TRANSFORM = SourcePromptTransform(
    name="osworld-linear-actor",
    source_text=LINEAR_ACTOR_ROLE_INSTRUCTIONS,
    replacements=OSWORLD_ACTOR_PROMPT_TRANSFORM.replacements,
)

OSWORLD_SPECIALIST_BROWSER_GUIDANCE = (
    "For browser inspection, preserve the setup-opened session; when CDP is "
    "allowed, use `osw --show browser.md` for guidance on browser usage."
)

OSWORLD_ROLE_INSTRUCTIONS: Final[Mapping[AgentRole, str]] = {
    "actor": OSWORLD_ACTOR_PROMPT_TRANSFORM.render(),
    "binder": (
        ROLE_INSTRUCTIONS["binder"] + "\n\n" + OSWORLD_SPECIALIST_BROWSER_GUIDANCE
    ),
    "critic": (
        ROLE_INSTRUCTIONS["critic"] + "\n\n" + OSWORLD_SPECIALIST_BROWSER_GUIDANCE
    ),
    "verifier": (
        ROLE_INSTRUCTIONS["verifier"] + "\n\n" + OSWORLD_SPECIALIST_BROWSER_GUIDANCE
    ),
}

OSWORLD_LINEAR_ACTOR_INSTRUCTIONS = OSWORLD_LINEAR_ACTOR_PROMPT_TRANSFORM.render()

LINEAR_COMPACTION_HANDOFF_REQUEST = """Do not call run_command. Produce a concise
handoff for a fresh context of the same Actor using exactly these headings:

Current stage:
Completed persistent work:
Canonical artifacts/state and readback:
Current blocker or failed assumptions:
Remaining work:
Next terminal command:

Preserve exact task facts, paths, identifiers, unresolved risks, and direct
saved-state evidence. Do not include large raw command output or claim
completion without direct proof."""


@dataclass(frozen=True, slots=True)
class BenchmarkOverlay:
    name: str
    system_prompt: str
    role_instructions: Mapping[AgentRole, str] | None = None
    linear_actor_instructions: str | None = None
    workspace: str = AGENT_WORKSPACE

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("overlay name cannot be empty")
        if not self.workspace.startswith("/") or not self.workspace.endswith(".agent"):
            raise ValueError("overlay workspace must be an absolute .agent path")
        if self.role_instructions is not None:
            if set(self.role_instructions) != set(ROLE_INSTRUCTIONS):
                raise ValueError(
                    "overlay role_instructions must contain every agent role"
                )
            if any(
                not isinstance(prompt, str) or not prompt.strip()
                for prompt in self.role_instructions.values()
            ):
                raise ValueError("overlay role instructions cannot be empty")
        if self.linear_actor_instructions is not None:
            if (
                not isinstance(self.linear_actor_instructions, str)
                or not self.linear_actor_instructions.strip()
            ):
                raise ValueError("linear_actor_instructions cannot be empty")


OSWORLD_OVERLAY: Final = BenchmarkOverlay(
    name="osworld-v2",
    system_prompt=OSWORLD_SYSTEM_OVERLAY,
    role_instructions=OSWORLD_ROLE_INSTRUCTIONS,
    linear_actor_instructions=OSWORLD_LINEAR_ACTOR_INSTRUCTIONS,
    workspace="/home/user/.agent",
)

OSWORLD_DEFAULT_PROMPT_BUNDLE: Final = PromptBundle(
    name="osworld-default",
    benchmark="osworld",
    condition="default",
    composition="terminal",
    agent_topology="orchestrated",
    workspace=OSWORLD_OVERLAY.workspace,
    role_assets=tuple(
        (role, PromptAsset(f"osworld-default-{role}", OSWORLD_ROLE_INSTRUCTIONS[role]))
        for role in ("actor", "binder", "critic", "verifier")
    ),
    shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
    terminal_control=PromptAsset(
        "orchestrated-terminal-control",
        TERMINAL_CONTROL_INSTRUCTIONS,
    ),
    system_overlay=PromptAsset("osworld-system-overlay", OSWORLD_SYSTEM_OVERLAY),
    compaction=PromptAsset(
        "orchestrated-compaction",
        COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=OSWORLD_OVERLAY.name,
)

OSWORLD_INVENTORY_ACTOR_INSTRUCTIONS: Final = """You are Actor in an Ubuntu desktop VM. Complete the task, verify the result, and submit.

Treat the task instruction as the central, top-level requirement. Follow its exact wording, named application, requested result, and restrictions.

First, re-read the instruction. As a general guide, inventory whichever parts of the task-relevant environment are useful, such as open applications, documents, browser tabs, accounts, and files in locations like Desktop, Downloads, and Documents. Let the task context determine the method and scope. Inspect any source or reference that must be matched. If a required file or fact is missing, ask the user one concise question.

Use run_command for commands, automation, application control, and inspection.

- Reuse the application, browser instance, profile, document, and state opened by setup.
- Use the application and output format named in the task.
- Complete every explicit requirement and avoid unrelated changes.
- Prefer minimal edits and exact typed values.
- Save, export, send, upload, or download to the exact requested destination.
- For GIMP, do not save or export unless explicitly requested.

Verify the final result directly:
- visual result: inspect a full-size screenshot;
- exact value or property: read it back;
- file or document: reopen the saved artifact;
- web action: reload and confirm the resulting record;
- behavior: exercise it and confirm the state change.

If verification finds a defect, repair it and verify again. Stop when the task is complete.

Tools:
- Image: `python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`
- Guidance: `osw "terms"`
- Ask user: `python /opt/cuawright-tools/ask_user.py --question "TEXT"`
- Submit: `python /opt/cuawright-tools/submit.py`

Run a `/opt/cuawright-tools/*.py` tool as the only command in its run_command call.

Browser:
- Use the existing Chrome instance and profile.
- Connect at `http://127.0.0.1:1337` with `/home/user/.local/share/osw/venv/bin/python`.
- Prefer `Accessibility.getFullAXTree` to inspect controls and state.
- Use CDP, xdotool, screenshots, keyboard, and mouse to operate user-facing controls.

When complete and verified, run submit.py."""

OSWORLD_NO_SUBAGENTS_PROMPT_BUNDLE: Final = PromptBundle(
    name="osworld-no-subagents",
    benchmark="osworld",
    condition="no-subagents",
    composition="terminal",
    agent_topology="linear",
    workspace=OSWORLD_OVERLAY.workspace,
    role_assets=(
        (
            "actor",
            PromptAsset(
                "osworld-no-subagents-actor",
                OSWORLD_INVENTORY_ACTOR_INSTRUCTIONS,
            ),
        ),
    ),
    shared_prefix=PromptAsset("empty-shared-prefix", ""),
    terminal_control=PromptAsset("empty-terminal-control", ""),
    system_overlay=PromptAsset("empty-system-overlay", ""),
    compaction=PromptAsset(
        "linear-compaction",
        LINEAR_COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=OSWORLD_OVERLAY.name,
)

E_L1_DIRECT_ACTOR_INSTRUCTIONS: Final = """You are Actor for an OSWorld V2 task in an Ubuntu desktop VM. Complete the task in the state opened by setup and verify the requested result.

Use run_command for terminal commands, code, and application control. Inspect images with `python /opt/cuawright-tools/image_show.py --path ABSOLUTE_PATH`. Search local procedure guidance with `osw "terms"`.

Keep these alignment defaults:
- Bind exact values and constraints to task-visible sources. Preserve uncertainty instead of guessing.
- Reuse the application, account, browser profile, task tab, and state opened by setup.
- Produce the result at the exact requested path, format, application, or account surface.
- Preserve unrelated state. Avoid duplicates and abandoned alternatives.
- Keep required formulas, links, layers, effects, relationships, and other native objects functional, not merely similar-looking.
- Complete every requested save, export, upload, import, send, purchase, or submission handoff exactly once as appropriate.
- Reopen, reload, or directly read back the final persistent result before submitting.

Do not confuse performing an action with achieving its intended effect. Before finishing, directly inspect the persistent result from the application, account, tab, profile, or file the user will later use. If an expected existing object is absent, first suspect the wrong state or profile rather than recreating it. Resolve visible contradictions, duplicates, missing outputs, unsupported guesses, and actions that produced no real state transition.

When CDP is allowed, connect to the existing Chrome endpoint at `http://127.0.0.1:1337` using `/home/user/.local/share/osw/venv/bin/python`. If developer tools are forbidden, use visible UI, screenshots, accessibility, and ordinary controls.

Do not read or modify hidden backend state, cookies, private APIs, evaluator code, or application databases to infer completion. If an exact required fact can only come from the user, ask one concise question with `python /opt/cuawright-tools/ask_user.py --question "TEXT"`.

When the task is complete, run `python /opt/cuawright-tools/submit.py`.

Run every `/opt/cuawright-tools/*.py` client as the only command in its run_command call."""

E_L1_DISCIPLINED_ACTOR_INSTRUCTIONS: Final = (
    E_L1_DIRECT_ACTOR_INSTRUCTIONS
    + """

Execution discipline:
- Convert the instruction into a checklist of required persistent outputs and exact constraints. Create and save a usable candidate within the first 40 model calls.
- Batch related inspections and edits. Investigate only evidence that can change an explicit requirement; do not repeatedly optimize inferred hidden fidelity.
- After a candidate exists, allow at most two verification-and-repair cycles for the same issue. If no direct contradiction remains, submit immediately.
- Independently recompute quantitative results once and sanity-check units, ranges, and scale.
- Before submission, enumerate every required filename, field, formula, relationship, upload, and saved application state, then read each one back directly."""
)

OSWORLD_E_L1_DIRECT_PROMPT_BUNDLE: Final = PromptBundle(
    name="osworld-e-l1-direct",
    benchmark="osworld",
    condition="e-l1-direct",
    composition="terminal",
    agent_topology="linear",
    workspace=OSWORLD_OVERLAY.workspace,
    role_assets=(
        (
            "actor",
            PromptAsset(
                "osworld-e-l1-direct-actor",
                E_L1_DIRECT_ACTOR_INSTRUCTIONS,
            ),
        ),
    ),
    shared_prefix=PromptAsset("empty-shared-prefix", ""),
    terminal_control=PromptAsset("empty-terminal-control", ""),
    system_overlay=PromptAsset("empty-system-overlay", ""),
    compaction=PromptAsset(
        "linear-compaction",
        LINEAR_COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=OSWORLD_OVERLAY.name,
)

OSWORLD_E_L1_DISCIPLINED_PROMPT_BUNDLE: Final = PromptBundle(
    name="osworld-e-l1-disciplined",
    benchmark="osworld",
    condition="e-l1-disciplined",
    composition="terminal",
    agent_topology="linear",
    workspace=OSWORLD_OVERLAY.workspace,
    role_assets=(
        (
            "actor",
            PromptAsset(
                "osworld-e-l1-disciplined-actor",
                E_L1_DISCIPLINED_ACTOR_INSTRUCTIONS,
            ),
        ),
    ),
    shared_prefix=PromptAsset("empty-shared-prefix", ""),
    terminal_control=PromptAsset("empty-terminal-control", ""),
    system_overlay=PromptAsset("empty-system-overlay", ""),
    compaction=PromptAsset(
        "linear-compaction",
        LINEAR_COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name=OSWORLD_OVERLAY.name,
)


def prompt_bundle_from_overlay(
    overlay: BenchmarkOverlay,
    *,
    agent_topology: AgentTopology,
) -> PromptBundle:
    """Compatibility adapter for callers that still provide BenchmarkOverlay."""

    if agent_topology == "linear":
        role_assets = (
            (
                "actor",
                PromptAsset(
                    f"{overlay.name}-linear-actor",
                    overlay.linear_actor_instructions or LINEAR_ACTOR_ROLE_INSTRUCTIONS,
                ),
            ),
        )
        terminal_control = LINEAR_TERMINAL_CONTROL_INSTRUCTIONS
        compaction = LINEAR_COMPACTION_HANDOFF_REQUEST
    elif agent_topology == "orchestrated":
        role_prompts = overlay.role_instructions or ROLE_INSTRUCTIONS
        role_assets = tuple(
            (
                role,
                PromptAsset(f"{overlay.name}-{role}", role_prompts[role]),
            )
            for role in ("actor", "binder", "critic", "verifier")
        )
        terminal_control = TERMINAL_CONTROL_INSTRUCTIONS
        compaction = COMPACTION_HANDOFF_REQUEST
    else:
        raise ValueError(f"unsupported agent topology: {agent_topology!r}")
    return PromptBundle(
        name=f"compat-{overlay.name}-{agent_topology}",
        benchmark=overlay.name,
        condition=agent_topology,
        composition="terminal",
        agent_topology=agent_topology,
        workspace=overlay.workspace,
        role_assets=role_assets,
        shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
        terminal_control=PromptAsset(
            f"{agent_topology}-terminal-control",
            terminal_control,
        ),
        system_overlay=PromptAsset(
            f"{overlay.name}-system-overlay",
            overlay.system_prompt,
        ),
        compaction=PromptAsset(f"{agent_topology}-compaction", compaction),
        overlay_name=overlay.name,
    )


def render_prompt_bundle_system(
    bundle: PromptBundle,
    role: AgentRole,
) -> str:
    if bundle.composition != "terminal":
        raise ValueError(f"prompt bundle {bundle.name!r} is not terminal-layered")
    if bundle.unbound_asset_names:
        raise ValueError(
            f"prompt bundle {bundle.name!r} has unbound external assets: "
            + ", ".join(bundle.unbound_asset_names)
        )
    assert bundle.shared_prefix is not None
    assert bundle.terminal_control is not None
    assert bundle.system_overlay is not None
    layers = (
        bundle.shared_prefix.render(),
        bundle.role_asset(role).render(workspace=bundle.workspace),
        bundle.terminal_control.render(workspace=bundle.workspace),
        bundle.system_overlay.render(),
    )
    return "\n\n".join(layer for layer in layers if layer)


def render_direct_prompt_bundle(bundle: PromptBundle) -> tuple[str, str]:
    if bundle.composition != "direct":
        raise ValueError(f"prompt bundle {bundle.name!r} is not direct")
    if bundle.unbound_asset_names:
        raise ValueError(
            f"prompt bundle {bundle.name!r} has unbound external assets: "
            + ", ".join(bundle.unbound_asset_names)
        )
    assert bundle.direct_system is not None
    assert bundle.direct_user is not None
    return (
        bundle.direct_system.render(),
        bundle.direct_user.render(),
    )


def compose_system_prompt(
    role: AgentRole,
    overlay: BenchmarkOverlay | PromptBundle = OSWORLD_OVERLAY,
    *,
    agent_topology: AgentTopology | None = None,
    parallel_tool_suffix: str | None = None,
) -> str:
    if isinstance(overlay, PromptBundle):
        bundle = overlay
        if bundle.composition != "terminal":
            raise ValueError(f"prompt bundle {bundle.name!r} is not terminal-layered")
        if agent_topology is not None and agent_topology != bundle.agent_topology:
            raise ValueError(
                f"prompt bundle {bundle.name!r} uses topology "
                f"{bundle.agent_topology!r}, not {agent_topology!r}"
            )
        effective_topology = bundle.agent_topology
    else:
        effective_topology = agent_topology or "orchestrated"
        bundle = prompt_bundle_from_overlay(
            overlay,
            agent_topology=effective_topology,
        )
    if effective_topology == "linear" and role != "actor":
        raise ValueError("linear topology supports only the actor role")
    system_prompt = render_prompt_bundle_system(bundle, role)
    if parallel_tool_suffix is not None:
        system_prompt = "\n\n".join(
            (
                system_prompt,
                _required_text("parallel_tool_suffix", parallel_tool_suffix),
            )
        )
    return system_prompt


def actor_user_prompt(
    instruction: str,
    *,
    agent_topology: AgentTopology = "orchestrated",
) -> str:
    if agent_topology not in {"orchestrated", "linear"}:
        raise ValueError(f"unsupported agent topology: {agent_topology!r}")
    sections = [
        f"Original task instruction:\n{_required_text('instruction', instruction)}",
    ]
    if agent_topology == "orchestrated":
        sections.append(
            "You are the persistent main Actor. Always start by running Binder "
            "before inspecting or modifying task state. Critic is not part of "
            "the active default architecture."
        )
    else:
        sections.append(
            "You are the persistent Actor. Complete the task directly from first "
            "inspection through final submission."
        )
    return "\n\n".join(section for section in sections if section)


def actor_resume_user_prompt(handoff: str) -> str:
    return (
        "Resume the same task after context compaction. The original instruction "
        "above is unchanged.\n\nCompacted handoff:\n"
        + _required_text("handoff", handoff)
        + "\n\nChoose and run the next terminal command."
    )


def budget_status_user_prompt(
    status_text: str,
    *,
    midpoint_due: bool,
    verifier_called: bool,
    require_verifier: bool = True,
    later_phases_remain: bool = False,
) -> str:
    guidance: list[str] = []
    if midpoint_due:
        guidance.append(
            "The midpoint has been reached: ensure a real persistent "
            "evaluator-facing result exists now."
        )
    if require_verifier and not verifier_called:
        guidance.append(
            "No Verifier report has been received yet; reserve enough budget to "
            "save/reload a candidate and run Verifier before submission."
        )
    if later_phases_remain:
        guidance.append(
            "The shared budget never resets; reserve enough calls for later phases."
        )
    suffix = " ".join(guidance)
    return f"[Shared budget] {_required_text('status_text', status_text)}" + (
        f"\n{suffix}" if suffix else ""
    )


def phase_user_prompt(
    instruction: str,
    *,
    phase_id: str,
    phase_index: int,
    phase_count: int,
    previous_phase_evaluated: bool,
    agent_topology: AgentTopology = "orchestrated",
) -> str:
    if agent_topology not in {"orchestrated", "linear"}:
        raise ValueError(f"unsupported agent topology: {agent_topology!r}")
    if isinstance(phase_index, bool) or not isinstance(phase_index, int):
        raise ValueError("phase_index must be an integer")
    if isinstance(phase_count, bool) or not isinstance(phase_count, int):
        raise ValueError("phase_count must be an integer")
    if phase_count < 1 or phase_index < 1 or phase_index > phase_count:
        raise ValueError("phase_index must be within phase_count")
    transition = (
        "The previous phase has been evaluated; its state persists."
        if previous_phase_evaluated
        else "This is the first phase."
    )
    sections = [
        f"Phase ID: {_required_text('phase_id', phase_id)}",
        f"Phase {phase_index}/{phase_count}. {transition}",
        "One shared model-call budget covers every phase and never resets.",
        f"Current phase instruction:\n{_required_text('instruction', instruction)}",
    ]
    if agent_topology == "orchestrated":
        sections.append(
            "Run a fresh Binder and an independent fresh Verifier for this phase. "
            "A latest Verifier status of `repair` blocks normal submission; "
            "submitting evaluates this phase and advances the next setup."
        )
    else:
        sections.append(
            "Continue directly as the same Actor. Submitting evaluates this phase "
            "and advances the next setup."
        )
    return "\n\n".join(sections)


def specialist_user_prompt(
    role: SpecialistRole,
    *,
    instruction: str,
    request: str,
    cache_path: str,
    allocated_steps: int,
    budget_status: str,
    prior_reports: Mapping[str, PriorReportContent] | None = None,
) -> str:
    if role not in {"binder", "critic", "verifier"}:
        raise ValueError(f"unsupported specialist role: {role!r}")
    maximum = SPECIALIST_CALL_MAX[role]
    if (
        isinstance(allocated_steps, bool)
        or not isinstance(allocated_steps, int)
        or allocated_steps < 1
        or allocated_steps > maximum
    ):
        raise ValueError(f"{role} allocated_steps must be between 1 and {maximum}")

    required_prior = {
        "binder": set(),
        "critic": {"binder"},
        "verifier": set(),
    }[role]
    provided_prior = set(prior_reports or {})
    missing = required_prior - provided_prior
    if missing:
        raise ValueError(
            f"{role} requires prior reports for: " + ", ".join(sorted(missing))
        )

    sections = [
        f"Temporary inspection cache: {_required_text('cache_path', cache_path)}",
        (
            f"Invocation budget: at most {allocated_steps} model calls. "
            f"{_required_text('budget_status', budget_status)}"
        ),
        f"Original task instruction:\n{_required_text('instruction', instruction)}",
        f"Main Actor request:\n{request}",
    ]

    if prior_reports:
        for name, report in sorted(prior_reports.items()):
            if not isinstance(report, PriorReportContent):
                raise ValueError(f"prior_reports[{name!r}] must be PriorReportContent")
            sections.append(
                "\n".join(
                    (
                        f"Latest {name.title()} report:",
                        f"Status: {report.status}",
                        "Unresolved: " + canonical_json(report.unresolved),
                        "Report body:",
                        report.report,
                    )
                )
            )

    if role == "verifier":
        sections.append(
            "Inspect the live canonical task state now. Treat the Actor request "
            "as a focus hint, not evidence that work is complete."
        )
    statuses = {
        "binder": "`complete` or `blocked`",
        "critic": "`accept`, `concerns`, or `blocked`",
        "verifier": "`ready`, `repair`, or `unknown`",
    }[role]
    sections.append(
        "When finished, stop calling run_command and return exactly one JSON "
        f"object with `status` {statuses}, a non-empty `report` string, and an "
        "`unresolved` string list. Do not wrap the object in Markdown fences."
    )
    return "\n\n".join(section for section in sections if section)


def compaction_user_prompt(
    *,
    agent_topology: AgentTopology = "orchestrated",
    prompt_bundle: PromptBundle | None = None,
) -> str:
    if prompt_bundle is not None:
        if prompt_bundle.composition != "terminal":
            raise ValueError(
                f"prompt bundle {prompt_bundle.name!r} is not terminal-layered"
            )
        if prompt_bundle.agent_topology != agent_topology:
            raise ValueError(
                f"prompt bundle {prompt_bundle.name!r} uses topology "
                f"{prompt_bundle.agent_topology!r}, not {agent_topology!r}"
            )
        assert prompt_bundle.compaction is not None
        return prompt_bundle.compaction.render()
    if agent_topology == "orchestrated":
        return COMPACTION_HANDOFF_REQUEST
    if agent_topology == "linear":
        return LINEAR_COMPACTION_HANDOFF_REQUEST
    raise ValueError(f"unsupported agent topology: {agent_topology!r}")


def prompt_source_manifest(
    *,
    agent_topology: AgentTopology = "orchestrated",
) -> dict[str, Any]:
    """Audit the complete active prompt sources and localized substitutions."""

    if agent_topology == "linear":
        transforms: Mapping[str, SourcePromptTransform] = {
            "linear_actor": LINEAR_ACTOR_PROMPT_TRANSFORM,
        }
    elif agent_topology == "orchestrated":
        transforms = ACTIVE_PROMPT_TRANSFORMS
    else:
        raise ValueError(f"unsupported agent topology: {agent_topology!r}")
    return {
        "schema_version": 1,
        "sources": {
            name: {
                "source_sha256": text_sha256(transform.source_text),
                "source_words": len(transform.source_text.split()),
                "rendered_words": len(transform.render().split()),
                "replacement_count": len(transform.replacements),
                "replacements": [
                    {
                        "kind": replacement.kind,
                        "reason": replacement.reason,
                        "old_sha256": text_sha256(replacement.old),
                        "new_sha256": text_sha256(replacement.new),
                    }
                    for replacement in transform.replacements
                ],
            }
            for name, transform in transforms.items()
        },
        "excluded_prompt_architectures": [
            "fixed-pipeline Binder/Critic/Verifier and fixed-pipeline compaction"
        ],
    }


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    return text_sha256(canonical_json(value))


def prompt_hash_manifest(
    role: AgentRole,
    overlay: BenchmarkOverlay | PromptBundle = OSWORLD_OVERLAY,
    *,
    agent_topology: AgentTopology | None = None,
    parallel_tool_suffix: str | None = None,
) -> dict[str, Any]:
    if isinstance(overlay, PromptBundle):
        bundle = overlay
        if agent_topology is not None and agent_topology != bundle.agent_topology:
            raise ValueError(
                f"prompt bundle {bundle.name!r} uses topology "
                f"{bundle.agent_topology!r}, not {agent_topology!r}"
            )
        effective_topology = bundle.agent_topology
    else:
        effective_topology = agent_topology or "orchestrated"
        bundle = prompt_bundle_from_overlay(
            overlay,
            agent_topology=effective_topology,
        )
    if effective_topology is None:
        raise ValueError("prompt hash manifest requires a terminal prompt bundle")
    system_prompt = compose_system_prompt(
        role,
        bundle,
        parallel_tool_suffix=parallel_tool_suffix,
    )
    role_prompt = bundle.role_asset(role).render(workspace=bundle.workspace)
    assert bundle.terminal_control is not None
    terminal_control = bundle.terminal_control.render(workspace=bundle.workspace)
    assert bundle.system_overlay is not None
    system_overlay = bundle.system_overlay.render()
    assert bundle.compaction is not None
    compaction = bundle.compaction.render()
    return {
        "schema_version": 1,
        "role": role,
        "overlay": bundle.overlay_name or bundle.benchmark,
        "bundle": bundle.name,
        "agent_topology": effective_topology,
        "sha256": {
            "source_audit": json_sha256(
                prompt_source_manifest(agent_topology=effective_topology)
            ),
            "shared_prefix": text_sha256(SHARED_SYSTEM_PREFIX),
            "role_prompt": text_sha256(role_prompt),
            "terminal_control": text_sha256(terminal_control),
            "overlay": text_sha256(system_overlay),
            "parallel_tool_suffix": (
                text_sha256(parallel_tool_suffix)
                if parallel_tool_suffix is not None
                else None
            ),
            "composed_system": text_sha256(system_prompt),
            "compaction": text_sha256(compaction),
            "sole_tool_schema": json_sha256(SOLE_TERMINAL_TOOL_SCHEMA),
        },
    }


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")
    return value


def _optional_line(label: str, value: str | None) -> str:
    if value is None:
        return ""
    return f"{label}: {_required_text(label, value)}"

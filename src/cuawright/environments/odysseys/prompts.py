from __future__ import annotations

from typing import Final

from cuawright.config.prompts import (
    COMPACTION_HANDOFF_REQUEST,
    ROLE_INSTRUCTIONS,
    SHARED_SYSTEM_PREFIX,
    TERMINAL_CONTROL_INSTRUCTIONS,
    PromptAsset,
    PromptBundle,
)


_ROLE_REPLACEMENTS: Final = {
    "actor": (
        "You are Actor for an OSWorld V2 task running in an\nUbuntu desktop VM.",
        "You are Actor for an Odysseys task running in an\nUbuntu desktop VM.",
    ),
    "binder": (
        "the main Actor on an OSWorld V2 task.",
        "the main Actor on an Odysseys task.",
    ),
    "critic": (
        "the main Actor on an OSWorld V2 task.",
        "the main Actor on an Odysseys task.",
    ),
    "verifier": (
        "the main Actor on an OSWorld V2 task.",
        "the main Actor on an Odysseys task.",
    ),
}

ODYSSEYS_SYSTEM_OVERLAY = """Odysseys is evaluated from the complete
chronological history of live-environment commands, saved desktop screenshots,
visible browser state, and final response. Work only through the live public web
and the visible Chrome profile opened by task setup. Do not use evaluator code,
rubric files, reference trajectories, cached answers, hidden site state, or
private APIs.

Task setup has already started the browser; do not rediscover this. Chrome is
running under `DISPLAY=:0` with a tab open, its DevTools endpoint is live at
`http://127.0.0.1:1337` and mirrored on `http://127.0.0.1:9222`, and Playwright
is installed at `~/.local/share/osw/venv/bin/python`. Drive Chrome with that
interpreter over `connect_over_cdp`, reusing the existing context and page:

```bash
cd "$HOME"
OSW_CDP_URL=http://127.0.0.1:1337 "$HOME/.local/share/osw/venv/bin/python" SCRIPT
```

```python
browser = p.chromium.connect_over_cdp(os.environ["OSW_CDP_URL"])
page = browser.contexts[0].pages[0]
```

The guest system Python has neither `playwright` nor `websocket`, and no CDP
client is installed for Node. Do not hand-roll a WebSocket or raw CDP client, do
not probe for ports or packages, and do not launch a second Chrome or a fresh
profile. Run `osw "chrome playwright cdp"` for the full procedure, including
repair if the venv import check fails.

The harness saves a fresh desktop screenshot after every dispatched Actor and
specialist command in `/home/user/.agent/odysseys/screenshots/` and mirrors it
into the host judging directory. Actor specialist invocations and specialists'
ordinary commands are retained with their role and stage; no environment action
is omitted based on role or prompt promises. Before each task or evidence
command finishes, leave the browser in a meaningful visible state that provides
useful chronological evidence. Keep every page or tab requested as final visual
proof open through submission.

Driving Chrome over CDP does not by itself produce judge evidence. A rubric can
only credit what the saved screenshot shows, so bring the page the Actor acted
on to the foreground and let it finish rendering before the command returns.
Prefer `page.bring_to_front()` and real navigation and clicks on the visible tab
over background fetches, `page.content()` scraping, hidden tabs, or headless
requests that leave the desktop unchanged. When a step produces a result worth
crediting, make that result visible on screen rather than only in command output.

The text given back to the user is also judge evidence. The main Actor must
write the exact final response as non-empty UTF-8 text to
`/home/user/.agent/odysseys/final_response.md`. Update that one canonical file
if the answer changes, before submitting. A normal submit is rejected until it
exists. Binder, Critic, and Verifier must not create or edit it. Do not put
secrets, hidden evaluator information, or claims unsupported by the visible
trajectory in that response."""


def localized_role_instructions() -> dict[str, str]:
    if set(ROLE_INSTRUCTIONS) != set(_ROLE_REPLACEMENTS):
        raise RuntimeError("CUAWright role map changed; re-audit Odysseys prompts")
    localized: dict[str, str] = {}
    for role, prompt in ROLE_INSTRUCTIONS.items():
        old, new = _ROLE_REPLACEMENTS[role]
        count = prompt.count(old)
        if count != 1:
            raise RuntimeError(
                f"Odysseys {role} localization expected one source anchor, "
                f"found {count}"
            )
        localized[role] = prompt.replace(old, new, 1)
    return localized


_LOCALIZED_ROLES = localized_role_instructions()

ODYSSEYS_DEFAULT_PROMPT_BUNDLE = PromptBundle(
    name="odysseys-default",
    benchmark="odysseys",
    condition="default",
    composition="terminal",
    agent_topology="orchestrated",
    workspace="/home/user/.agent",
    role_assets=tuple(
        (
            role,
            PromptAsset(f"odysseys-default-{role}", _LOCALIZED_ROLES[role]),
        )
        for role in ("actor", "binder", "critic", "verifier")
    ),
    shared_prefix=PromptAsset("shared-system-prefix", SHARED_SYSTEM_PREFIX),
    terminal_control=PromptAsset(
        "orchestrated-terminal-control",
        TERMINAL_CONTROL_INSTRUCTIONS,
    ),
    system_overlay=PromptAsset(
        "odysseys-system-overlay",
        ODYSSEYS_SYSTEM_OVERLAY,
    ),
    compaction=PromptAsset(
        "orchestrated-compaction",
        COMPACTION_HANDOFF_REQUEST,
    ),
    overlay_name="odysseys",
)

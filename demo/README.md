# Converting `final_script.py` into a Codex Skill

This guide shows how to turn a verified `mini-web-agent` output script into a local Codex Skill.

## 1. Start from `final_script.py`

After a successful `mini-web-agent` run, use:

```text
outputs/<task-id>/final_script.py
```

This script should already contain the verified browser workflow. Before turning it into a skill, make sure it can run from a clean shell and accepts reusable inputs through CLI flags, for example with `argparse`.

## 2. Create the skill files

Create a local skill folder:

```bash
SKILL_NAME="my-browser-task"
SKILL_DIR="$HOME/.agents/skills/$SKILL_NAME"

mkdir -p "$SKILL_DIR"
cp outputs/<task-id>/final_script.py "$SKILL_DIR/final_script.py"
```

The folder should contain:

```text
~/.agents/skills/<skill-name>/
├── SKILL.md
├── final_script.py
└── <skill_name>_cli.py
```

For concrete examples, see [`google-flights-comparison`](./google-flights-comparison/) and [`ticketmaster-classical-tickets`](./ticketmaster-classical-tickets/). They use the same three-file pattern: `SKILL.md`, `final_script.py`, and a small CLI wrapper.

## 3. Add a CLI wrapper

Create `~/.agents/skills/<skill-name>/<skill_name>_cli.py`.

```python
#!/usr/bin/env python3
"""Run the verified browser automation script for this skill."""

from pathlib import Path
import os
import runpy
import sys


SKILL_DIR = Path(__file__).resolve().parent
SOURCE_SCRIPT = Path(
    os.environ.get("SOURCE_SCRIPT", str(SKILL_DIR / "final_script.py"))
).expanduser()


def main() -> None:
    if not SOURCE_SCRIPT.exists():
        raise SystemExit(f"Source script not found: {SOURCE_SCRIPT}")

    os.environ.setdefault("LOCAL_BROWSER_CDP_URL", "http://127.0.0.1:9222")
    os.environ.setdefault("WORKSPACE_DIR", str(SKILL_DIR))

    sys.argv[0] = str(SOURCE_SCRIPT)
    runpy.run_path(str(SOURCE_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
```

> [!TIP]
> You can replace `SOURCE_SCRIPT` with a skill-specific override variable, such as `GOOGLE_FLIGHTS_SOURCE_SCRIPT` or `TICKETMASTER_SOURCE_SCRIPT`.

## 4. Add `SKILL.md`

Create `~/.agents/skills/<skill-name>/SKILL.md`.

````markdown
---
name: my-browser-task
description: Use a verified browser automation CLI to complete <specific task>. Use when the user asks to <trigger condition>.
---

# My Browser Task

Use this skill when the user explicitly mentions `$my-browser-task` or asks for this workflow by name.

## Procedure

1. Translate the user's request into CLI flags.
2. Run the wrapper:

   ```bash
   LOCAL_BROWSER_CDP_URL="${LOCAL_BROWSER_CDP_URL:-http://127.0.0.1:9222}" python {baseDir}/my_browser_task_cli.py
   ```

3. Read the script output and report the result.
4. If the browser or website is unavailable, report the failure. Do not fabricate results.
````

`name` and `description` are the fields Codex uses to decide when the skill is relevant. Keep the description specific and action-oriented.

## 5. Register the skill with Codex

Codex discovers local skills from:

```text
~/.agents/skills/<skill-name>/
```

Once the folder contains `SKILL.md`, `final_script.py`, and the CLI wrapper, start a new Codex/agent session so the skill metadata is loaded.

> [!IMPORTANT]
> Do not install your own skills under `~/.codex/skills/.system/`. That directory is reserved for Codex-managed/system skills. Skills created from `mini-web-agent` outputs should go under `~/.agents/skills`.

> [!NOTE]
> To force a specific skill, mention its `$<skill-name>` handle in the prompt, followed by the task. Example: `$google-flights-comparison find the best round-trip flight from Hong Kong to Rio de Janeiro`.

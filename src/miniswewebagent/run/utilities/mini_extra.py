#!/usr/bin/env python3
from __future__ import annotations

import sys
from importlib import import_module

from rich.console import Console

subcommands = [
    (
        "miniswewebagent.run.utilities.inspector",
        ["inspect", "i", "inspector"],
        "Run inspector (browse trajectories)",
    ),
    (
        "miniswewebagent.run.utilities.trace_viewer",
        ["trace-viewer", "traces", "viewer"],
        "Serve a browser trace viewer for outputs/default",
    ),
]


def get_docstring() -> str:
    lines = [
        "This is the [yellow]central entry point for extra commands[/yellow] from mini-swe-webagent.",
        "",
        "Available sub-commands:",
        "",
    ]
    for _, aliases, description in subcommands:
        alias_text = " or ".join(f"[bold green]{alias}[/bold green]" for alias in aliases)
        lines.append(f"  {alias_text}: {description}")
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    if not args or len(args) == 1 and args[0] in ["-h", "--help"]:
        Console().print(get_docstring())
        return
    for module_path, aliases, _ in subcommands:
        if args[0] in aliases:
            import_module(module_path).app(args[1:], prog_name=f"mini-web-extra {aliases[0]}")
            return
    Console().print(get_docstring())
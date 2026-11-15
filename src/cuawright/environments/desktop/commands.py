from __future__ import annotations

import shlex
from collections.abc import Callable, Sequence


InspectionExtension = Callable[[Sequence[str]], bool]


def is_inspection_command(
    command: str,
    *,
    extra_inspector: InspectionExtension | None = None,
) -> bool:
    """Classify a simple desktop command as read-only inspection."""

    if ".agent/" in command or command.rstrip().endswith(".agent"):
        return False
    argv = simple_command_argv(command)
    if argv is None:
        return False
    program = argv[0]
    if program in {
        "cat",
        "file",
        "head",
        "ls",
        "md5sum",
        "pwd",
        "readlink",
        "sha256sum",
        "stat",
        "tail",
        "wc",
        "zipinfo",
    }:
        return True
    if program in {"grep", "rg"}:
        return not any(
            argument == "--pre" or argument.startswith("--pre=")
            for argument in argv[1:]
        )
    if program == "find":
        return not any(_is_mutating_find_action(argument) for argument in argv[1:])
    if program == "unzip":
        return len(argv) >= 3 and argv[1] in {"-l", "-p", "-t", "-v", "-Z"}
    if program == "sqlite3":
        return _is_read_only_sqlite(argv)
    if program == "ffprobe":
        return _is_read_only_ffprobe(argv)
    if program == "mediainfo":
        return _is_read_only_mediainfo(argv)
    return bool(extra_inspector is not None and extra_inspector(argv))


def simple_command_argv(command: str) -> list[str] | None:
    if not command.strip() or "\n" in command or "\r" in command:
        return None
    if _has_unquoted_shell_expansion(command):
        return None
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        argv = list(lexer)
    except ValueError:
        return None
    if not argv or any(
        token and all(character in ";&|<>" for character in token) for token in argv
    ):
        return None
    return argv


def _has_unquoted_shell_expansion(command: str) -> bool:
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            escaped = False
        elif quote == "'":
            if character == "'":
                quote = None
        elif quote == '"':
            if character == '"':
                quote = None
            elif character == "\\":
                escaped = True
            elif character in "$`":
                return True
        elif character == "\\":
            escaped = True
        elif character in {"'", '"'}:
            quote = character
        elif character in "$`*?[]{}~()":
            return True
        elif character in "@!+" and command[index + 1 : index + 2] == "(":
            return True
        index += 1
    return False


def _is_read_only_sqlite(argv: Sequence[str]) -> bool:
    if len(argv) != 3:
        return False
    query = argv[2].strip().lower()
    return query in {
        ".databases",
        ".dump",
        ".indexes",
        ".schema",
        ".tables",
    }


def _is_mutating_find_action(argument: str) -> bool:
    return argument == "-delete" or argument.startswith(
        ("-exec", "-fls", "-fprint", "-fprintf", "-ok")
    )


def _is_read_only_ffprobe(argv: Sequence[str]) -> bool:
    value_options = {
        "-loglevel",
        "-of",
        "-print_format",
        "-read_intervals",
        "-select_streams",
        "-show_data_hash",
        "-show_entries",
        "-show_optional_fields",
        "-v",
    }
    flag_options = {
        "-bitexact",
        "-byte_binary_prefix",
        "-count_frames",
        "-count_packets",
        "-hide_banner",
        "-prefix",
        "-pretty",
        "-private",
        "-sexagesimal",
        "-show_chapters",
        "-show_error",
        "-show_format",
        "-show_frames",
        "-show_packets",
        "-show_programs",
        "-show_streams",
        "-unit",
    }
    input_count = 0
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument in flag_options:
            index += 1
            continue
        if argument in value_options:
            if index + 1 >= len(argv):
                return False
            index += 2
            continue
        if argument.startswith("-"):
            return False
        input_count += 1
        index += 1
    return input_count == 1


def _is_read_only_mediainfo(argv: Sequence[str]) -> bool:
    input_count = 0
    for argument in argv[1:]:
        lowered = argument.lower()
        if lowered in {"--details=1", "--full", "--output=json", "--output=xml"}:
            continue
        if lowered.startswith("--language="):
            continue
        if lowered.startswith("--inform=") and "file://" not in lowered:
            continue
        if argument.startswith("-"):
            return False
        input_count += 1
    return input_count == 1


__all__ = [
    "InspectionExtension",
    "is_inspection_command",
    "simple_command_argv",
]

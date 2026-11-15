"""Guest-local searchable procedure knowledge client."""

DESKTOP_KNOWLEDGE_SCRIPT = r'''#!/usr/bin/env python3
import argparse
import pathlib
import re
import shlex
import sqlite3

ROOT = pathlib.Path.home() / ".local" / "share" / "osw"
MAX_SEARCH_CHARS = 4000
MAX_SHOW_CHARS = 8000


def sections(path):
    heading = path.stem
    chunks = []
    current = []
    fence = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
        match = None if fence else re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            if current:
                chunks.append((heading, "\n".join(current)))
                current = []
            heading = match.group(1).strip()
        current.append(line)
    if current:
        chunks.append((heading, "\n".join(current)))
    return chunks


def build_index(connection):
    connection.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(path, heading, body)"
    )
    connection.execute("DELETE FROM docs")
    for path in sorted((ROOT / "kb").glob("*.md")):
        connection.executemany(
            "INSERT INTO docs(path, heading, body) VALUES (?, ?, ?)",
            [
                (path.name, chunk_heading, body)
                for chunk_heading, body in sections(path)
            ],
        )
    connection.commit()


def query_rows(connection, query, limit):
    return connection.execute(
        """
        SELECT path, heading, body, bm25(docs)
        FROM docs
        WHERE docs MATCH ?
        ORDER BY bm25(docs)
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()


def fallback_query(query):
    tokens = re.findall(r"[A-Za-z0-9_]+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


def search(query, limit):
    connection = sqlite3.connect(":memory:")
    build_index(connection)
    try:
        rows = query_rows(connection, query, limit)
    except sqlite3.OperationalError:
        rows = []
    if not rows:
        fallback = fallback_query(query)
        if not fallback:
            print("No searchable terms. Use `osw --list` to inspect available topics.")
            return
        try:
            rows = query_rows(connection, fallback, limit)
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        print(
            "No knowledge-base matches. Use `osw --list`, then "
            "`osw --show FILE --heading HEADING`."
        )
        return
    for path, heading, body, score in rows:
        command = (
            f"osw --show {shlex.quote(path)} "
            f"--heading {shlex.quote(heading)}"
        )
        print(
            f"## {path} — {heading} (score={score:.4f})\n"
            f"{bounded(body, MAX_SEARCH_CHARS)}\nFull section: {command}\n"
        )


def knowledge_path(name):
    if pathlib.Path(name).name != name or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*\.md", name
    ):
        raise SystemExit("Knowledge file must be a listed Markdown basename.")
    path = ROOT / "kb" / name
    if not path.is_file():
        raise SystemExit(f"Unknown knowledge file: {name}. Use `osw --list`.")
    return path


def bounded(text, limit=MAX_SHOW_CHARS):
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[section truncated; use `osw --show`]"


def list_knowledge():
    paths = sorted((ROOT / "kb").glob("*.md"))
    if not paths:
        print("No knowledge files are installed.")
        return
    for path in paths:
        print(path.name)
        for heading, _ in sections(path):
            print(f"  - {heading}")


def show(name, heading):
    path = knowledge_path(name)
    if heading is None:
        print(bounded(path.read_text(encoding="utf-8")))
        return
    matches = [
        body
        for section_heading, body in sections(path)
        if section_heading.casefold() == heading.casefold()
    ]
    if not matches:
        available = ", ".join(section_heading for section_heading, _ in sections(path))
        raise SystemExit(f"Unknown heading {heading!r}. Available: {available}")
    print(bounded("\n\n".join(matches)))


def main():
    parser = argparse.ArgumentParser(prog="osw")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show")
    parser.add_argument("--heading")
    args = parser.parse_args()
    modes = sum((args.query is not None, args.list, args.show is not None))
    if modes != 1:
        parser.error("provide exactly one query, --list, or --show FILE")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.heading is not None and args.show is None:
        parser.error("--heading requires --show FILE")
    if args.list:
        list_knowledge()
        return
    if args.show is not None:
        show(args.show, args.heading)
        return
    search(args.query, args.limit)


if __name__ == "__main__":
    main()
'''

__all__ = ["DESKTOP_KNOWLEDGE_SCRIPT"]

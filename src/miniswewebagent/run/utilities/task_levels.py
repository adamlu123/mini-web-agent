"""Task difficulty-level lookup used by the run-compare and leaderboard views.

Run outputs (``result.json``) do not carry a task's difficulty level, only its
``task_id``. This module resolves ``task_id -> level`` ("easy" / "medium" /
"hard") by reading the bundled Online-Mind2Web-style benchmark task files
under ``run/benchmarks/``, reusing the same normalization as the batch runner
(:func:`miniswewebagent.utils.om2w_tasks.load_om2w_tasks`).
"""

from __future__ import annotations

from pathlib import Path

from miniswewebagent import package_dir
from miniswewebagent.utils.om2w_tasks import load_om2w_tasks

# Bundled benchmark task files that carry a "level" field, relative to
# run/benchmarks/. Add new benchmark files here as they gain a "level" column.
# Entries may also be absolute paths (e.g. from tests), which take precedence
# over the benchmarks directory per pathlib join semantics.
DEFAULT_TASK_LEVEL_SOURCES: tuple[str, ...] = (
    "om2w_260220.json",
    "odysseys/odysseys.json",
)


def _benchmarks_dir() -> Path:
    return package_dir / "run" / "benchmarks"


def load_task_levels(sources: tuple[str, ...] | None = None) -> dict[str, str]:
    """Builds a ``task_id -> level`` lookup from bundled benchmark task files.

    Missing source files are skipped silently so this stays usable even if a
    benchmark file is renamed or removed. When multiple sources define the
    same ``task_id``, the later source (in ``sources`` order) wins.
    """
    resolved_sources = DEFAULT_TASK_LEVEL_SOURCES if sources is None else sources
    benchmarks_dir = _benchmarks_dir()

    levels: dict[str, str] = {}
    for relative_path in resolved_sources:
        source_path = benchmarks_dir / relative_path
        if not source_path.exists():
            continue
        for task in load_om2w_tasks(source_path):
            task_id = str(task.get("task_id") or "").strip()
            if not task_id:
                continue
            levels[task_id] = str(task.get("level") or "").strip().lower()
    return levels

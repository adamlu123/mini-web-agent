"""Terminal report for comparing benchmark runs, without the HTTP viewer.

Wraps ``run_compare.compare_runs()`` so a CI log or a quick terminal check
can get the same leaderboard + baseline diff that the ``/api/compare`` route
on ``mini-web-traces`` serves, without starting a server.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from miniswewebagent.run.utilities.run_compare import compare_runs
from miniswewebagent.run.utilities.task_levels import load_task_levels

app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console(highlight=False)


def _print_leaderboard(report: dict) -> None:
    table = Table(title=f"Leaderboard ({report['rootDir']})")
    table.add_column("Run")
    table.add_column("Total", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Failure", justify="right")
    table.add_column("Unknown", justify="right")
    table.add_column("Success rate", justify="right")
    for row in report["leaderboard"]:
        overall = row["overall"]
        marker = " (baseline)" if row["runId"] == report["baselineId"] else ""
        table.add_row(
            f"{row['runId']}{marker}",
            str(row["totalTasks"]),
            str(overall["success"]),
            str(overall["failure"]),
            str(overall["unknown"]),
            f"{overall['successRate']:.1%}",
        )
    console.print(table)


def _print_diff_summary(report: dict) -> None:
    diff_summary = report.get("diffSummary") or {}
    if not diff_summary:
        return

    console.print(f"\n[bold]Task-level diff vs baseline '{report['baselineId']}':[/bold]")
    table = Table()
    table.add_column("Run")
    table.add_column("Improved", justify="right")
    table.add_column("Regressed", justify="right")
    table.add_column("Same success", justify="right")
    table.add_column("Same fail", justify="right")
    table.add_column("Unknown", justify="right")
    for run_id, diff in diff_summary.items():
        table.add_row(
            run_id,
            str(diff.get("improved", 0)),
            str(diff.get("regressed", 0)),
            str(diff.get("sameSuccess", 0)),
            str(diff.get("sameFail", 0)),
            str(diff.get("unknown", 0)),
        )
    console.print(table)


@app.command()
def main(
    runs: str = typer.Option(
        ..., "--runs", help="Comma-separated run folder names to compare, e.g. 'baseline,candidate'."
    ),
    runs_root: Path = typer.Option(
        Path("outputs/default"), "--runs-root", help="Directory containing run folders."
    ),
    baseline: str = typer.Option(
        None, "--baseline", help="Run id to treat as baseline; defaults to the first id in --runs."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the raw comparison JSON instead of tables."),
) -> None:
    resolved_root = runs_root.expanduser().resolve()
    if not resolved_root.exists():
        raise typer.BadParameter(f"Runs root '{resolved_root}' does not exist.")

    run_ids = [run_id.strip() for run_id in runs.split(",") if run_id.strip()]
    if not run_ids:
        raise typer.BadParameter("--runs must include at least one comma-separated run id.")
    for run_id in run_ids:
        if not (resolved_root / run_id).is_dir():
            raise typer.BadParameter(f"Run '{run_id}' was not found under '{resolved_root}'.")

    report = compare_runs(resolved_root, run_ids, baseline_id=baseline, task_levels=load_task_levels())

    if as_json:
        console.print_json(json.dumps(report))
        return

    _print_leaderboard(report)
    _print_diff_summary(report)


if __name__ == "__main__":
    app()

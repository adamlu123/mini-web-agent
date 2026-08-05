# Historical scripts

The dated subdirectories contain snapshots of one-off experiments and analysis
commands. [`2026-04/`](2026-04/) holds the April 2026 workflows and is retained
for provenance only:

- `run_*.sh`, `run_command.sh`, and `relaunch_monitor_M3.sh` capture historical
  launch commands and machine-specific paths.
- `eval_*`, `materialize_*`, `token_usage_*`, and `verify_*` capture analysis
  workflows used for archived reports.
- `build_*` and `convert_*` capture one-time artifact transformations.
- `test_*` and `start_review_viewer.sh` are superseded diagnostics and tooling.

These files are not supported entry points. Active scripts, package code,
configuration, and tests must not import or execute anything in this directory.
Use the canonical commands listed in [`scripts/README.md`](../README.md).

Historical scripts may contain stale absolute paths or assumptions about an old
workspace layout. Reproduce them from the commit that generated the associated
report rather than adapting them in place.

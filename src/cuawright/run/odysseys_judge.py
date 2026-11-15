from __future__ import annotations

import argparse
import os
from pathlib import Path

from cuawright.environments.odysseys.judge import (
    _RUNNER_EXIT,
    _RUNNER_LOG,
    _RUNNER_PID,
    open_verified_judge,
)
from cuawright.operations.scheduler import ProcessJob, run_process_jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify immutable Odysseys inputs, then invoke its saved judge"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    with open_verified_judge(args.manifest, args.manifest_sha256) as verified:
        root = Path(verified.snapshot_root)
        outcome = run_process_jobs(
            (
                ProcessJob(
                    job_id="odysseys-judge",
                    command=tuple(verified.argv()),
                    cwd=root,
                    environment=dict(os.environ),
                    log_path=root / _RUNNER_LOG,
                    pid_path=root / _RUNNER_PID,
                    exit_path=root / _RUNNER_EXIT,
                    pass_fds=(
                        verified.snapshot_root_fd,
                        verified.output_fd,
                    ),
                ),
            ),
            width=1,
        )[0]
        if outcome.status == "completed" and outcome.return_code == 0:
            verified.publish_output()
            return 0
        return outcome.return_code if outcome.return_code != 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

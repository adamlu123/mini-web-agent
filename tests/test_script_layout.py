"""Contract tests for the concern-based scripts layout."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

CANONICAL_SHELL_SCRIPTS = (
    "cluster/om2w/judge_only/run.sh",
    "cluster/om2w/judge_only/submit.sh",
    "cluster/om2w/qwen35_9b/run.sh",
    "cluster/om2w/qwen35_9b/submit.sh",
    "cluster/om2w/qwen36_27b/run.sh",
    "cluster/om2w/qwen36_27b/submit.sh",
    "local/om2w/run.sh",
    "local/om2w/serve_qwen35.sh",
    "local/om2w/shards.sh",
    "review/tunnel.sh",
    "review/viewer.sh",
)

LEGACY_SHELL_WRAPPERS = {
    "run_om2w_judge_only_cluster.sh": "cluster/om2w/judge_only/run.sh",
    "submit_om2w_judge_only_cluster.sh": "cluster/om2w/judge_only/submit.sh",
    "run_qwen35_9b_p0_checkpoint_om2w_eval.sh": "cluster/om2w/qwen35_9b/run.sh",
    "submit_qwen35_9b_p0_checkpoint_om2w_eval.sh": "cluster/om2w/qwen35_9b/submit.sh",
    "run_qwen36_27b_om2w_cluster_eval.sh": "cluster/om2w/qwen36_27b/run.sh",
    "submit_qwen36_27b_om2w_cluster_eval.sh": "cluster/om2w/qwen36_27b/submit.sh",
    "om2w_vllm_sharded.sh": "local/om2w/shards.sh",
    "run_om2w_vllm_local.sh": "local/om2w/run.sh",
    "serve_vllm_qwen35_local.sh": "local/om2w/serve_qwen35.sh",
    "start_public_tunnel.sh": "review/tunnel.sh",
    "start_remote_review_viewer.sh": "review/viewer.sh",
}


def test_active_shell_scripts_parse() -> None:
    paths = [str(SCRIPTS_DIR / path) for path in CANONICAL_SHELL_SCRIPTS]
    paths.extend(str(SCRIPTS_DIR / path) for path in LEGACY_SHELL_WRAPPERS)
    paths.extend(
        [
            str(SCRIPTS_DIR / "lib/cluster_runtime.sh"),
            str(SCRIPTS_DIR / "lib/cluster_submit.sh"),
        ]
    )
    subprocess.run(["bash", "-n", *paths], check=True)


@pytest.mark.parametrize(
    ("legacy_path", "canonical_path"),
    LEGACY_SHELL_WRAPPERS.items(),
)
def test_legacy_shell_entry_point_forwards_to_canonical(
    legacy_path: str,
    canonical_path: str,
) -> None:
    wrapper = SCRIPTS_DIR / legacy_path
    assert os.access(wrapper, os.X_OK)
    assert canonical_path in wrapper.read_text(encoding="utf-8")


def test_cluster_staging_uses_runtime_allowlist(tmp_path: Path) -> None:
    staged = tmp_path / "mini-web-agent"
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; mwa_stage_cluster_repo "$2" "$3"',
            "bash",
            str(SCRIPTS_DIR / "lib/cluster_submit.sh"),
            str(REPO_ROOT),
            str(staged),
        ],
        check=True,
    )

    assert (staged / "scripts/cluster/om2w/qwen35_9b/run.sh").is_file()
    assert (staged / "scripts/eval/persistent_cli_steps.py").is_file()
    assert (
        staged / "src/miniswewebagent/evaluation/om2w/runner.py"
    ).is_file()
    assert not (staged / "scripts/archive").exists()
    assert not (staged / "scripts/.tools").exists()
    assert not list(staged.rglob("__pycache__"))
    assert not list(staged.rglob("*.py[co]"))


def test_cluster_runtime_rejects_unsafe_staging_destination(
    tmp_path: Path,
) -> None:
    upload = tmp_path / "upload"
    upload.mkdir()
    unsafe_destination = tmp_path / "workspace"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; mwa_copy_staged_repo "$2" "$3"',
            "bash",
            str(SCRIPTS_DIR / "lib/cluster_runtime.sh"),
            str(upload),
            str(unsafe_destination),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "refusing unsafe local staging path" in completed.stderr
    assert not unsafe_destination.exists()


def test_repo_does_not_ship_downloaded_cloudflared_binary() -> None:
    assert not (SCRIPTS_DIR / ".tools/cloudflared").exists()


def test_tunnel_helper_accepts_existing_cloudflared_binary() -> None:
    env = os.environ.copy()
    env["CLOUDFLARED_BIN"] = "/bin/true"
    completed = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "review/tunnel.sh"), "8123"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "http://127.0.0.1:8123" in completed.stdout

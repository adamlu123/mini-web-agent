from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "docker" / "run_dist_eval_q35_image.sh"


def _manifestless_runner_env(tmp_path: Path) -> dict[str, str]:
    source_repo = tmp_path / "source-repo"
    (source_repo / "configs").mkdir(parents=True)
    (source_repo / "configs" / "qwen3_5_train_aligned.jinja").write_text(
        "{{ messages }}\n", encoding="utf-8"
    )
    tasks_file = (
        source_repo
        / "src"
        / "miniswewebagent"
        / "run"
        / "benchmarks"
        / "om2w_260220.json"
    )
    tasks_file.parent.mkdir(parents=True)
    tasks_file.write_text("[]\n", encoding="utf-8")
    evaluator = source_repo / "scripts" / "eval_with_original_om2w.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("# preflight fixture\n", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    credentials = tmp_path / "cred.sh"
    credentials.touch()

    return {
        **os.environ,
        "DATA_ROOT": str(tmp_path / "data"),
        "JOB_NAME": "runtime-mode-test",
        "EVAL_CKPT": str(checkpoint),
        "EVAL_RUN_ID": "runtime-mode-test",
        "REPO": str(source_repo),
        "LOCAL_REPO": str(tmp_path / "local-repo"),
        "CANONICAL_REPO_LINK": str(tmp_path / "canonical" / "mini-web-agent"),
        "CREDS_FILE": str(credentials),
        "EVAL_SKIP_BOOTSTRAP": "1",
        "EVAL_CONTRACT_PREFLIGHT_ONLY": "1",
    }


def _run_runner(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_manifestless_checkpoint_uses_exact_legacy_compatibility_defaults(
    tmp_path: Path,
) -> None:
    env = _manifestless_runner_env(tmp_path)
    # Inherited aligned-profile settings must not leak into the manifestless
    # compatibility path.
    env["TEXT_ONLY_IMAGE_POLICY"] = "black_56"
    env["STOP_SEQUENCES_JSON"] = '["<|im_end|>"]'

    result = _run_runner(env)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert (
        "contract=manifestless_compat max_len=65536 "
        "input_budget=48000 output=4096"
    ) in combined
    assert "image_policy=none stop_sequences=[]" in combined
    assert "mm_processor=<checkpoint defaults>" in combined
    assert (
        "contract preflight complete; exiting before credentials/GPU startup"
        in combined
    )


def test_manifestless_checkpoint_preserves_explicit_empty_chat_template(
    tmp_path: Path,
) -> None:
    env = _manifestless_runner_env(tmp_path)
    env["CHAT_TEMPLATE"] = ""

    result = _run_runner(env)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "template= image_policy=none" in combined


def test_required_manifest_fails_before_bootstrap_or_gpu_startup(
    tmp_path: Path,
) -> None:
    env = _manifestless_runner_env(tmp_path)
    env["REQUIRE_RUNTIME_MANIFEST"] = "1"

    result = _run_runner(env)

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "checkpoint is missing required" in combined
    assert "install missing python deps" not in combined
    assert "GPU preflight" not in combined

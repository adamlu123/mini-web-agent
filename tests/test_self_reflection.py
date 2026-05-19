from __future__ import annotations

from pathlib import Path

import pytest

from miniswewebagent.tools.self_reflection import (
    _infer_run_dir_from_images,
    _load_action_history_log,
    _render_final_verdict_user_prompt,
    _resolve_artifact_dir,
    _gateway_config,
    build_parser,
)


def test_build_parser_rejects_removed_screenshots_dir_arg() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--config", "judge_config.json", "--screenshots-dir", "shots"])


def test_resolve_artifact_dir_prefers_run_folder_inferred_from_images(tmp_path: Path) -> None:
    run_dir = tmp_path / "final_runs" / "run_003"
    screenshots_dir = run_dir / "screenshots"
    screenshots_dir.mkdir(parents=True)
    image_path = screenshots_dir / "final_execution_2_filtered.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "final_script_log.txt").write_text(
        "step 1 action: opened results\nFinal response: done\n",
        encoding="utf-8",
    )

    artifact_dir = _resolve_artifact_dir(
        images=[image_path],
        discovered_run_dir=None,
        output_path="",
        workspace_dir=str(tmp_path),
    )

    assert _infer_run_dir_from_images([image_path]) == run_dir
    assert artifact_dir == run_dir
    assert _load_action_history_log(artifact_dir) == "step 1 action: opened results\nFinal response: done"


def test_resolve_artifact_dir_falls_back_to_output_run_folder(tmp_path: Path) -> None:
    run_dir = tmp_path / "final_runs" / "run_004"
    run_dir.mkdir(parents=True)
    (run_dir / "final_script_log.txt").write_text("step 1 action: fallback\n", encoding="utf-8")

    artifact_dir = _resolve_artifact_dir(
        images=[],
        discovered_run_dir=None,
        output_path=str(run_dir / "judge_result.json"),
        workspace_dir=str(tmp_path),
    )

    assert artifact_dir == run_dir
    assert _load_action_history_log(artifact_dir) == "step 1 action: fallback"


def test_render_final_verdict_user_prompt_injects_placeholders() -> None:
    rendered = _render_final_verdict_user_prompt(
        "Action history:\n{action_history_log}\n\nReasonings:\n{image_reasonings}",
        action_history_log="step 1 action: search",
        image_reasonings="1. The Dallas filter is visible.",
    )

    assert rendered == (
        "Action history:\nstep 1 action: search\n\n"
        "Reasonings:\n1. The Dallas filter is visible."
    )


def test_render_final_verdict_user_prompt_appends_backward_compatible_sections() -> None:
    rendered = _render_final_verdict_user_prompt(
        "Task block.",
        action_history_log="step 1 action: search",
        image_reasonings="1. The Dallas filter is visible.",
    )

    assert rendered == (
        "Task block.\n\n"
        "Action history log:\nstep 1 action: search\n\n"
        "Image reasonings:\n1. The Dallas filter is visible."
    )


def test_gateway_config_defaults_to_trapi_kimi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_GATEWAY_MODEL", raising=False)

    cfg = _gateway_config(api_key="", endpoint="", model="")

    assert cfg.backend == "trapi_kimi"
    assert cfg.model == "Kimi-K2.5_1"
    assert cfg.api_key == ""
    assert cfg.endpoint == (
        "https://trapi.research.microsoft.com/"
        "gcr/shared/openai/deployments/Kimi-K2.5_1/chat/completions"
        "?api-version=2024-10-21"
    )


def test_gateway_config_preserves_legacy_responses_override() -> None:
    cfg = _gateway_config(api_key="sk-test", endpoint="", model="gpt-5.4")

    assert cfg.backend == "responses"
    assert cfg.model == "gpt-5.4"
    assert cfg.api_key == "sk-test"
    assert cfg.endpoint == "http://gateway.phyagi.net/api/responses"
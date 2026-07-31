from __future__ import annotations

import json
from pathlib import Path

import pytest

from miniswewebagent.tools.self_reflection import (
    _infer_run_dir_from_images,
    _load_action_history_log,
    _load_trajectory_scope,
    _render_final_verdict_user_prompt,
    _resolve_artifact_dir,
    _gateway_config,
    build_parser,
    main,
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
    monkeypatch.delenv("OPENAI_GATEWAY_ENDPOINT", raising=False)

    cfg = _gateway_config(api_key="", endpoint="", model="")

    assert cfg.backend == "trapi_kimi"
    assert cfg.model == "Kimi-K2.5_1"
    assert cfg.api_key == ""
    assert cfg.endpoint == (
        "https://trapi.research.microsoft.com/"
        "gcr/shared/openai/deployments/Kimi-K2.5_1/chat/completions"
        "?api-version=2024-10-21"
    )


def test_gateway_config_preserves_legacy_responses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_GATEWAY_ENDPOINT", raising=False)

    cfg = _gateway_config(api_key="sk-test", endpoint="", model="gpt-5.4")

    assert cfg.backend == "responses"
    assert cfg.model == "gpt-5.4"
    assert cfg.api_key == "sk-test"
    assert cfg.endpoint == "http://gateway.phyagi.net/api/responses"


def test_gateway_config_honours_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_GATEWAY_ENDPOINT is read alongside OPENAI_GATEWAY_MODEL."""
    monkeypatch.setenv("OPENAI_GATEWAY_ENDPOINT", "http://judge.internal/api/responses")
    monkeypatch.setenv("OPENAI_GATEWAY_MODEL", "gpt-5.4")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "sk-env")

    cfg = _gateway_config(api_key="", endpoint="", model="")

    assert cfg.backend == "responses"
    assert cfg.endpoint == "http://judge.internal/api/responses"
    assert cfg.model == "gpt-5.4"


def test_gateway_config_rejects_policy_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A policy chat-completions URL must fail loudly, not 404 on every reflection.

    Pointing OPENAI_GATEWAY_* at the local vLLM server used to send judge requests
    for a locally-served deployment to the phyagi /responses gateway.
    """
    monkeypatch.setenv("OPENAI_GATEWAY_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
    monkeypatch.setenv("OPENAI_GATEWAY_MODEL", "sft_ckpt")
    monkeypatch.setenv("OPENAI_GATEWAY_API_KEY", "sk-env")

    with pytest.raises(RuntimeError, match="chat-completions URL"):
        _gateway_config(api_key="", endpoint="", model="")


def test_load_trajectory_scope_uses_all_saved_images_across_epochs(tmp_path: Path) -> None:
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    (screenshots / "one.png").write_bytes(b"one")
    (screenshots / "two.png").write_bytes(b"two")
    (tmp_path / "browser-steps.jsonl").write_text(
        "\n".join(
            [
                '{"browser_step": 1, "agent_step": 2, "session_epoch": 1, '
                '"action": "Open results", "success": true, '
                '"screenshot_path": "screenshots/one.png"}',
                '{"browser_step": 2, "agent_step": 3, "session_epoch": 2, '
                '"action": "Recover in a new tab", "success": true, '
                '"screenshot_path": "screenshots/two.png"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scope = _load_trajectory_scope(tmp_path)

    assert [path.name for path in scope["images"]] == ["one.png", "two.png"]
    assert scope["session_epochs"] == [1, 2]
    assert scope["covered_through_browser_step"] == 2
    assert "Recover in a new tab" in scope["action_history_log"]
    assert "Session epoch: 2" in scope["image_contexts"][str(scope["images"][1])]


def test_trajectory_main_caches_image_judgment_and_writes_freshness_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    screenshot = tmp_path / "screenshots" / "browser_step_0001.png"
    screenshot.parent.mkdir()
    screenshot.write_bytes(b"png")
    (tmp_path / "browser-steps.jsonl").write_text(
        json.dumps(
            {
                "browser_step": 1,
                "agent_step": 2,
                "session_epoch": 1,
                "action": "Apply the exact filter",
                "success": True,
                "screenshot_path": "screenshots/browser_step_0001.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "plan.md").write_text("- [x] exact filter", encoding="utf-8")
    config = tmp_path / "judge_config.json"
    config.write_text(
        json.dumps(
            {
                "image_judge_system_prompt": "judge image",
                "image_judge_user_prompt": "check all constraints",
                "final_verdict_system_prompt": "judge trajectory",
                "final_verdict_user_prompt": "{action_history_log}\n{image_reasonings}",
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_call_gateway(**kwargs):
        calls.append(kwargs["tag"])
        if kwargs["tag"] == "self_reflection.image":
            return "Reasoning: The exact filter is visibly selected.\nScore: 5"
        return "Thoughts: All critical points are evidenced.\nStatus: success"

    monkeypatch.setattr(
        "miniswewebagent.tools.self_reflection._call_gateway", fake_call_gateway
    )
    args = [
        "--scope",
        "trajectory",
        "--workspace-dir",
        str(tmp_path),
        "--config",
        str(config),
        "--output",
        "reflection/judge_result.json",
        "--model",
        "gpt-5.4",
        "--api-key",
        "test",
    ]

    assert main(args) == 0
    payload = json.loads(
        (tmp_path / "reflection" / "judge_result.json").read_text(encoding="utf-8")
    )
    assert payload["scope"] == "trajectory"
    assert payload["covered_through_browser_step"] == 1
    assert payload["image_count"] == 1
    assert payload["image_records"][0]["CacheHit"] is False
    assert calls == ["self_reflection.image", "self_reflection.final"]

    calls.clear()
    assert main(args) == 0
    assert calls == ["self_reflection.final"]

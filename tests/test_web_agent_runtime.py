from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import miniswewebagent.utils.web_agent_runtime as runtime_contract
from miniswewebagent.utils.web_agent_runtime import (
    SFT_STATE_DEBUG_PROMPT_ASSET_CONTRACT,
    classify_checkpoint,
    load_web_agent_runtime,
    main,
    preflight_web_agent_runtime,
    validate_sft_state_debug_prompt_assets,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _model_config(
    *,
    max_position_embeddings: int = 32768,
) -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "vision_config": {
            "hidden_size": 1280,
            "depth": 32,
            "patch_size": 16,
        },
        "text_config": {
            "max_position_embeddings": max_position_embeddings,
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 248320,
            "head_dim": 256,
        },
    }


def _manifest(template_hash: str) -> dict:
    return {
        "schema_version": 1,
        "contract_id": "legacy_blank_v1",
        "checkpoint_kind": "full_multimodal",
        "base_model_id": "Qwen/Qwen3.5-4B",
        "training_max_sequence_length": 32768,
        "prompt_mode": "sft_state_debug",
        "chat_template": {
            "file": "chat_template.jinja",
            "sha256": template_hash,
        },
        "processor": {
            "backend": "pil",
            "min_pixels": 3136,
            "max_pixels": 262144,
        },
        "text_only_image_policy": {
            "mode": "black_56",
            "width": 56,
            "height": 56,
            "rgb": [0, 0, 0],
            "placement": "first_user_prefix",
        },
        "terminal_token_policy": "legacy_endoftext",
        "stop_sequences": ["<|im_end|>"],
        "inference_defaults": {
            "max_model_len": 32768,
            "max_context_tokens": 16000,
            "max_output_tokens": 4096,
            "sliding_window_keep_turns": 1,
            "min_vllm_version": "0.24.0",
        },
    }


def _write_checkpoint(
    root: Path,
    *,
    include_vision: bool = True,
) -> Path:
    root.mkdir()
    template = b"{{ messages }}\n"
    (root / "chat_template.jinja").write_bytes(template)
    _write_json(root / "config.json", _model_config())
    _write_json(
        root / "preprocessor_config.json",
        {"size": {"shortest_edge": 3136, "longest_edge": 262144}},
    )
    weight_map = {
        "model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00001-of-00002.safetensors",
    }
    if include_vision:
        weight_map["model.visual.patch_embed.proj.weight"] = "model-00002-of-00002.safetensors"
    for filename in set(weight_map.values()):
        (root / filename).touch()
    _write_json(
        root / "model.safetensors.index.json",
        {"metadata": {"total_size": 0}, "weight_map": weight_map},
    )
    _write_json(
        root / "web_agent_runtime.json",
        _manifest(hashlib.sha256(template).hexdigest()),
    )
    return root


def test_preflight_validates_full_checkpoint_and_resolves_safe_overrides(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")

    resolved = preflight_web_agent_runtime(
        checkpoint,
        max_model_len=24576,
        max_context_tokens=12000,
        max_output_tokens=2048,
        vllm_version="0.24.1",
    )

    assert resolved["tensor_layout"]["kind"] == "full_multimodal"
    assert resolved["resolved_inference"] == {
        "max_model_len": 24576,
        "max_context_tokens": 12000,
        "max_output_tokens": 2048,
        "sliding_window_keep_turns": 1,
    }
    assert resolved["resolved_model_config"]["text_only_image_policy"] == "black_56"
    assert resolved["resolved_model_config"]["stop_sequences"] == ["<|im_end|>"]
    assert {
        name: value["sha256"]
        for name, value in resolved["resolved_prompt_assets"].items()
    } == {
        name: value["sha256"]
        for name, value in SFT_STATE_DEBUG_PROMPT_ASSET_CONTRACT.items()
    }


def test_sft_state_debug_prompt_assets_match_real_training_bundle() -> None:
    resolved = validate_sft_state_debug_prompt_assets()

    assert set(resolved) == {"system", "instructions"}
    for name, expected in SFT_STATE_DEBUG_PROMPT_ASSET_CONTRACT.items():
        assert Path(resolved[name]["file"]).name == expected["file"]
        assert resolved[name]["sha256"] == expected["sha256"]


def test_sft_state_debug_prompt_asset_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_dir = tmp_path / "sft_assets"
    asset_dir.mkdir()
    source_dir = runtime_contract._SFT_STATE_DEBUG_PROMPT_ASSET_DIR
    for expected in SFT_STATE_DEBUG_PROMPT_ASSET_CONTRACT.values():
        filename = expected["file"]
        (asset_dir / filename).write_bytes((source_dir / filename).read_bytes())
    (asset_dir / "state_system_debug.txt").write_bytes(b"drifted prompt\n")
    monkeypatch.setattr(
        runtime_contract, "_SFT_STATE_DEBUG_PROMPT_ASSET_DIR", asset_dir
    )

    with pytest.raises(
        ValueError, match="system prompt asset SHA-256 mismatch"
    ):
        validate_sft_state_debug_prompt_assets()


def test_preflight_prefers_modern_processor_config(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    _write_json(
        checkpoint / "processor_config.json",
        {
            "image_processor": {
                "size": {"shortest_edge": 3136, "longest_edge": 262144}
            }
        },
    )
    # The modern file is authoritative when present, even if the legacy file
    # contains stale base-model defaults.
    _write_json(
        checkpoint / "preprocessor_config.json",
        {"size": {"shortest_edge": 65536, "longest_edge": 16777216}},
    )

    assert preflight_web_agent_runtime(checkpoint)["contract_id"] == "legacy_blank_v1"

    _write_json(
        checkpoint / "processor_config.json",
        {
            "image_processor": {
                "size": {"shortest_edge": 65536, "longest_edge": 16777216}
            }
        },
    )
    with pytest.raises(ValueError, match="processor_config.json pixel bounds"):
        preflight_web_agent_runtime(checkpoint)


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (
            lambda root: _write_json(
                root / "preprocessor_config.json",
                {"size": {"shortest_edge": 65536, "longest_edge": 16777216}},
            ),
            "pixel bounds",
        ),
        (
            lambda root: _write_json(
                root / "config.json", _model_config(max_position_embeddings=65536)
            ),
            "max_position_embeddings",
        ),
        (
            lambda root: (root / "chat_template.jinja").write_text(
                "changed", encoding="utf-8"
            ),
            "SHA-256 mismatch",
        ),
    ],
)
def test_preflight_rejects_checkpoint_contract_drift(
    tmp_path: Path, mutate, error_match: str
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    mutate(checkpoint)

    with pytest.raises(ValueError, match=error_match):
        preflight_web_agent_runtime(checkpoint)


def test_preflight_rejects_language_only_and_unsafe_limits(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint", include_vision=False)

    assert classify_checkpoint(checkpoint).kind == "language_only"
    with pytest.raises(ValueError, match="full_multimodal"):
        preflight_web_agent_runtime(checkpoint)

    full_checkpoint = _write_checkpoint(tmp_path / "full")
    with pytest.raises(ValueError, match="exceeds contract default"):
        preflight_web_agent_runtime(full_checkpoint, max_context_tokens=48000)
    with pytest.raises(ValueError, match="older than required"):
        preflight_web_agent_runtime(full_checkpoint, vllm_version="0.18.0")


def test_manifest_schema_rejects_policy_mismatch_and_extra_fields(
    tmp_path: Path,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    manifest_path = checkpoint / "web_agent_runtime.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["text_only_image_policy"]["mode"] = "none"
    manifest["unexpected"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="Invalid runtime contract"):
        load_web_agent_runtime(checkpoint)


def test_manifest_schema_rejects_unknown_contract_id(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    manifest_path = checkpoint / "web_agent_runtime.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract_id"] = "unsupported_contract"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="Invalid runtime contract"):
        load_web_agent_runtime(checkpoint)


def test_preflight_cli_writes_resolved_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoint")
    output = tmp_path / "logs" / "resolved_runtime.json"

    rc = main(
        [
            "preflight",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--max-context-tokens",
            "12000",
            "--vllm-version",
            "0.24.0",
        ]
    )

    assert rc == 0
    stdout_value = json.loads(capsys.readouterr().out)
    output_value = json.loads(output.read_text())
    assert stdout_value == output_value
    assert output_value["resolved_inference"]["max_context_tokens"] == 12000
    assert output_value["resolved_prompt_assets"]["system"]["sha256"] == (
        "688cd91919e5d55643b9124c007f012cc8e39cd4d45bb9cdd633aa02f65c7359"
    )
    assert output_value["resolved_prompt_assets"]["instructions"]["sha256"] == (
        "4398df6d982712c2f08d0f662a77f6a8a63a25d714996d25fbc01fefa62c5225"
    )

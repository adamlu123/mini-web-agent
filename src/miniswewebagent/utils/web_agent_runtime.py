"""Training/runtime contract validation for WebWright policy checkpoints.

The contract lives beside an exported Hugging Face checkpoint as
``web_agent_runtime.json``.  This module intentionally performs only local,
read-only validation so launch scripts can fail before allocating a GPU or
starting vLLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RUNTIME_MANIFEST_FILENAME = "web_agent_runtime.json"
VISION_KEY_PREFIXES = ("model.visual.", "visual.")
LANGUAGE_KEY_PREFIXES = ("model.language_model.", "language_model.")
SFT_STATE_DEBUG_PROMPT_ASSET_CONTRACT = {
    "system": {
        "file": "state_system_debug.txt",
        "sha256": "688cd91919e5d55643b9124c007f012cc8e39cd4d45bb9cdd633aa02f65c7359",
    },
    "instructions": {
        "file": "state_instructions_debug.txt",
        "sha256": "4398df6d982712c2f08d0f662a77f6a8a63a25d714996d25fbc01fefa62c5225",
    },
}
_SFT_STATE_DEBUG_PROMPT_ASSET_DIR = (
    Path(__file__).resolve().parents[3] / "echo_rl" / "web_agent" / "sft_assets"
)
QWEN35_4B_TEXT_SIGNATURE = {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "num_hidden_layers": 32,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "vocab_size": 248320,
    "head_dim": 256,
}


class ChatTemplateContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_safe_relative_path(self) -> "ChatTemplateContract":
        path = Path(self.file)
        if path.is_absolute() or ".." in path.parts or self.file != path.name:
            raise ValueError("chat_template.file must be a filename in the checkpoint root")
        return self


class ProcessorContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["pil"]
    min_pixels: Literal[3136]
    max_pixels: Literal[262144]


class TextOnlyImagePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["black_56"]
    width: Literal[56]
    height: Literal[56]
    rgb: tuple[Literal[0], Literal[0], Literal[0]]
    placement: Literal["first_user_prefix"]


class InferenceDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_model_len: Literal[32768]
    max_context_tokens: Literal[16000]
    max_output_tokens: Literal[4096]
    sliding_window_keep_turns: Literal[1]
    min_vllm_version: Literal["0.24.0"]


class WebAgentRuntimeContract(BaseModel):
    """Strict schema emitted by the aligned PhiTrain HF exporter."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    contract_id: Literal["legacy_blank_v1"]
    checkpoint_kind: Literal["full_multimodal"]
    base_model_id: Literal["Qwen/Qwen3.5-4B"]
    training_max_sequence_length: Literal[32768]
    prompt_mode: Literal["sft_state_debug"]
    chat_template: ChatTemplateContract
    processor: ProcessorContract
    text_only_image_policy: TextOnlyImagePolicy
    terminal_token_policy: Literal["legacy_endoftext"]
    stop_sequences: list[Literal["<|im_end|>"]]
    inference_defaults: InferenceDefaults

    @model_validator(mode="after")
    def validate_contract_coherence(self) -> "WebAgentRuntimeContract":
        if self.stop_sequences != ["<|im_end|>"]:
            raise ValueError("stop_sequences must contain exactly one '<|im_end|>' entry")
        return self


@dataclass(frozen=True)
class TensorLayout:
    kind: Literal["full_multimodal", "language_only", "invalid"]
    tensor_count: int
    language_tensor_count: int
    vision_tensor_count: int


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sft_state_debug_prompt_assets() -> dict[str, dict[str, str]]:
    """Validate the exact local prompt bundle used by ``prompt_mode``.

    Schema-1 WebWright checkpoints were trained with these two byte-exact
    assets.  The runtime imports the same files through
    ``echo_rl.web_agent.prompts``.  Keeping this check local and fail-closed
    prevents a checkout with prompt drift from passing checkpoint preflight.
    """

    resolved: dict[str, dict[str, str]] = {}
    for name, expected in SFT_STATE_DEBUG_PROMPT_ASSET_CONTRACT.items():
        asset_path = _SFT_STATE_DEBUG_PROMPT_ASSET_DIR / expected["file"]
        if not asset_path.is_file():
            raise ValueError(
                f"sft_state_debug {name} prompt asset is missing: {asset_path}"
            )
        actual_hash = _sha256(asset_path)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"sft_state_debug {name} prompt asset SHA-256 mismatch: "
                f"expected {expected['sha256']}, got {actual_hash} ({asset_path})"
            )
        resolved[name] = {
            "file": str(asset_path.resolve()),
            "sha256": actual_hash,
        }
    return resolved


def load_web_agent_runtime(checkpoint: str | Path) -> WebAgentRuntimeContract:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    manifest_path = checkpoint_path / RUNTIME_MANIFEST_FILENAME
    data = _read_json_object(manifest_path)
    try:
        return WebAgentRuntimeContract.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Invalid runtime contract {manifest_path}: {exc}") from exc


def checkpoint_tensor_keys(checkpoint: str | Path) -> set[str]:
    """Return checkpoint tensor names without loading tensor payloads.

    An HF shard index is preferred.  Unindexed safetensors checkpoints are
    inspected through ``safe_open`` only when necessary.
    """

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} has no non-empty weight_map")
        bad_entries = [
            key
            for key, filename in weight_map.items()
            if (
                not isinstance(key, str)
                or not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or not filename.endswith(".safetensors")
            )
        ]
        if bad_entries:
            raise ValueError(f"{index_path} contains invalid weight_map entries")
        missing_shards = sorted(
            {
                filename
                for filename in weight_map.values()
                if not (checkpoint_path / filename).is_file()
            }
        )
        if missing_shards:
            raise ValueError(
                f"{index_path} references missing shard(s): {', '.join(missing_shards)}"
            )
        return set(weight_map)

    shard_paths = sorted(checkpoint_path.glob("*.safetensors"))
    if not shard_paths:
        raise ValueError(f"No HF safetensors weights found in {checkpoint_path}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError(
            "safetensors is required to inspect an unindexed checkpoint"
        ) from exc

    keys: set[str] = set()
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt") as handle:
            keys.update(handle.keys())
    if not keys:
        raise ValueError(f"No tensors found in {checkpoint_path}")
    return keys


def classify_tensor_keys(keys: set[str]) -> TensorLayout:
    language_count = sum(key.startswith(LANGUAGE_KEY_PREFIXES) for key in keys)
    vision_count = sum(key.startswith(VISION_KEY_PREFIXES) for key in keys)
    if language_count and vision_count:
        kind: Literal["full_multimodal", "language_only", "invalid"] = "full_multimodal"
    elif language_count:
        kind = "language_only"
    else:
        kind = "invalid"
    return TensorLayout(
        kind=kind,
        tensor_count=len(keys),
        language_tensor_count=language_count,
        vision_tensor_count=vision_count,
    )


def classify_checkpoint(checkpoint: str | Path) -> TensorLayout:
    return classify_tensor_keys(checkpoint_tensor_keys(checkpoint))


def _processor_pixel_bounds(config: dict[str, Any]) -> tuple[Any, Any]:
    min_pixels = config.get("min_pixels")
    max_pixels = config.get("max_pixels")
    size = config.get("size")
    if isinstance(size, dict):
        if min_pixels is None:
            min_pixels = size.get("shortest_edge")
        if max_pixels is None:
            max_pixels = size.get("longest_edge")
    return min_pixels, max_pixels


def _load_processor_pixel_bounds(checkpoint: Path) -> tuple[Path, tuple[Any, Any]]:
    """Load modern processor_config.json first, then the HF legacy fallback."""

    modern_path = checkpoint / "processor_config.json"
    if modern_path.is_file():
        modern_config = _read_json_object(modern_path)
        image_processor = modern_config.get("image_processor")
        if not isinstance(image_processor, dict):
            raise ValueError(
                f"{modern_path} must contain an image_processor object"
            )
        return modern_path, _processor_pixel_bounds(image_processor)

    legacy_path = checkpoint / "preprocessor_config.json"
    return legacy_path, _processor_pixel_bounds(_read_json_object(legacy_path))


def _parse_version(value: str) -> tuple[int, int, int]:
    match = re.match(r"^\s*(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"Cannot parse vLLM version {value!r}")
    return tuple(int(part) for part in match.groups())


def resolve_inference_defaults(
    contract: WebAgentRuntimeContract,
    *,
    max_model_len: int | None = None,
    max_context_tokens: int | None = None,
    max_output_tokens: int | None = None,
    sliding_window_keep_turns: int | None = None,
    allow_training_contract_override: bool = False,
) -> dict[str, int]:
    defaults = contract.inference_defaults
    requested = {
        "max_model_len": defaults.max_model_len if max_model_len is None else max_model_len,
        "max_context_tokens": (
            defaults.max_context_tokens if max_context_tokens is None else max_context_tokens
        ),
        "max_output_tokens": (
            defaults.max_output_tokens if max_output_tokens is None else max_output_tokens
        ),
        "sliding_window_keep_turns": (
            defaults.sliding_window_keep_turns
            if sliding_window_keep_turns is None
            else sliding_window_keep_turns
        ),
    }
    if any(value <= 0 for value in requested.values()):
        raise ValueError("Resolved inference limits must all be positive")

    if not allow_training_contract_override:
        for key in ("max_model_len", "max_context_tokens", "max_output_tokens"):
            if requested[key] > getattr(defaults, key):
                raise ValueError(
                    f"{key}={requested[key]} exceeds contract default {getattr(defaults, key)}"
                )
        if requested["sliding_window_keep_turns"] != defaults.sliding_window_keep_turns:
            raise ValueError(
                "sliding_window_keep_turns conflicts with the training contract; "
                "set ALLOW_TRAINING_CONTRACT_OVERRIDE=1 to override"
            )
    if requested["max_model_len"] > contract.training_max_sequence_length:
        raise ValueError(
            "max_model_len exceeds the checkpoint training_max_sequence_length"
        )
    if requested["max_context_tokens"] + requested["max_output_tokens"] > requested["max_model_len"]:
        raise ValueError(
            "max_context_tokens + max_output_tokens exceeds max_model_len"
        )
    return requested


def preflight_web_agent_runtime(
    checkpoint: str | Path,
    *,
    max_model_len: int | None = None,
    max_context_tokens: int | None = None,
    max_output_tokens: int | None = None,
    sliding_window_keep_turns: int | None = None,
    vllm_version: str | None = None,
    allow_training_contract_override: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    contract = load_web_agent_runtime(checkpoint_path)

    resolved_prompt_assets = validate_sft_state_debug_prompt_assets()

    template_path = checkpoint_path / contract.chat_template.file
    actual_template_hash = _sha256(template_path)
    if actual_template_hash != contract.chat_template.sha256:
        raise ValueError(
            f"Chat template SHA-256 mismatch: expected {contract.chat_template.sha256}, "
            f"got {actual_template_hash}"
        )

    model_config = _read_json_object(checkpoint_path / "config.json")
    text_config = model_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("config.json must contain a text_config object")
    if model_config.get("model_type") != "qwen3_5" or (
        "Qwen3_5ForConditionalGeneration"
        not in (model_config.get("architectures") or [])
    ):
        raise ValueError(
            "config.json is not a Qwen3_5ForConditionalGeneration checkpoint"
        )
    actual_signature = {
        key: text_config.get(key) for key in QWEN35_4B_TEXT_SIGNATURE
    }
    if actual_signature != QWEN35_4B_TEXT_SIGNATURE:
        raise ValueError(
            "config.json text_config does not match Qwen/Qwen3.5-4B: "
            f"{actual_signature!r}"
        )
    configured_max = text_config.get("max_position_embeddings")
    if configured_max != contract.training_max_sequence_length:
        raise ValueError(
            "config.json text_config.max_position_embeddings does not match "
            f"training_max_sequence_length ({configured_max!r} != "
            f"{contract.training_max_sequence_length})"
        )

    processor_path, processor_bounds = _load_processor_pixel_bounds(checkpoint_path)
    actual_min_pixels, actual_max_pixels = processor_bounds
    expected_bounds = (contract.processor.min_pixels, contract.processor.max_pixels)
    if (actual_min_pixels, actual_max_pixels) != expected_bounds:
        raise ValueError(
            f"{processor_path.name} pixel bounds do not match the runtime contract "
            f"({(actual_min_pixels, actual_max_pixels)!r} != {expected_bounds!r})"
        )

    tensor_layout = classify_checkpoint(checkpoint_path)
    if tensor_layout.kind != contract.checkpoint_kind:
        raise ValueError(
            f"Checkpoint tensor layout is {tensor_layout.kind}, expected "
            f"{contract.checkpoint_kind}"
        )

    if vllm_version is not None and _parse_version(vllm_version) < _parse_version(
        contract.inference_defaults.min_vllm_version
    ):
        raise ValueError(
            f"vLLM {vllm_version} is older than required "
            f"{contract.inference_defaults.min_vllm_version}"
        )

    resolved_inference = resolve_inference_defaults(
        contract,
        max_model_len=max_model_len,
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        sliding_window_keep_turns=sliding_window_keep_turns,
        allow_training_contract_override=allow_training_contract_override,
    )
    return {
        **contract.model_dump(mode="json"),
        "checkpoint_path": str(checkpoint_path),
        "tensor_layout": asdict(tensor_layout),
        "resolved_prompt_assets": resolved_prompt_assets,
        "resolved_inference": resolved_inference,
        "resolved_model_config": {
            "text_only_image_policy": contract.text_only_image_policy.mode,
            "stop_sequences": list(contract.stop_sequences),
            "max_context_tokens": resolved_inference["max_context_tokens"],
            "max_output_tokens": resolved_inference["max_output_tokens"],
            "sliding_window_keep_turns": resolved_inference["sliding_window_keep_turns"],
        },
        "vllm_version": vllm_version,
    }


def write_resolved_runtime(
    output_path: str | Path,
    resolved: dict[str, Any],
) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="validate a full multimodal checkpoint runtime contract"
    )
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--output")
    preflight.add_argument("--max-model-len", type=int)
    preflight.add_argument("--max-context-tokens", type=int)
    preflight.add_argument("--max-output-tokens", type=int)
    preflight.add_argument("--sliding-window-keep-turns", type=int)
    preflight.add_argument("--vllm-version")
    preflight.add_argument("--allow-training-contract-override", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        resolved = preflight_web_agent_runtime(
            args.checkpoint,
            max_model_len=args.max_model_len,
            max_context_tokens=args.max_context_tokens,
            max_output_tokens=args.max_output_tokens,
            sliding_window_keep_turns=args.sliding_window_keep_turns,
            vllm_version=args.vllm_version,
            allow_training_contract_override=args.allow_training_contract_override,
        )
        if args.output:
            write_resolved_runtime(args.output, resolved)
        print(json.dumps(resolved, sort_keys=True))
        return 0
    except ValueError as exc:
        parser = _build_parser()
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Tokenize an SPB last-observation bundle for PhiTrain Qwen3.5 VLM SFT.

This is a deliberately small wrapper around PhiTrain's WebWright tokenizer. The
checked-out tokenizer labels every assistant turn, while the requested
single-turn setting requires loss only on the final assistant in each prefix
row. This wrapper reuses the production rendering/processor/blank-image code,
then replaces labels with an explicit ``mask_all_but_last_n=1`` mask.

Output is PhiTrain's parquet-ref VLM SFT layout:

    <output>/samples/part-*.parquet
    <output>/images/part-*.parquet
    <output>/manifest.json
    <output>/tokenization_run.json

Rows longer than ``--max-seq-len`` are dropped whole; they are never truncated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PHITRAIN_ROOT = Path("/home/luyadong/sandbox/aifsdk/phitrain")
DEFAULT_PROCESSOR_PATH = Path("/data/yadonglu/hf/Qwen3.5-9B")
DEFAULT_CHAT_TEMPLATE = (
    DEFAULT_PHITRAIN_ROOT
    / "scripts/tools/data/tokenization/tokenizers/"
    "Qwen3.5-no-auto-think/chat_template.jinja"
)
IGNORE_INDEX = -100
PROMPT_VERSION = "spb_lastobs_singleturn_qwen35_train_aligned_v1"

_ROWS: list[dict[str, Any]] | None = None
_PROCESSOR: Any = None
_TURN_TOKENS: Any = None
_WEB_TOKENIZER: Any = None
_MASKING: Any = None
_TORCH: Any = None
_BUNDLE_DIR: Path | None = None
_PROCESSOR_PATH: str | None = None
_DATASET_NAME: str | None = None
_INCLUDE_DEBUG_TEXT = False


@dataclass(frozen=True)
class TokenizedResult:
    source_index: int
    source_row_id: str
    token_count: int
    row: dict[str, Any] | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * percentile))
    return int(sorted_values[max(0, min(index, len(sorted_values) - 1))])


def _count_unmasked_label_spans(labels: list[int]) -> int:
    count = 0
    in_span = False
    for label in labels:
        if label != IGNORE_INDEX and not in_span:
            count += 1
            in_span = True
        elif label == IGNORE_INDEX:
            in_span = False
    return count


def _import_phitrain(phitrain_root: Path) -> tuple[Any, Any, Any]:
    """Import the requested checkout, not the venv's unrelated editable clone."""

    root = phitrain_root.expanduser().resolve()
    expected = root / "scripts/tools/data/tokenization/tokenize_webwright_vlm_sft.py"
    if not expected.is_file():
        raise FileNotFoundError(f"PhiTrain WebWright tokenizer not found: {expected}")
    sys.path.insert(0, str(root))

    # Imported only after the explicit checkout is first on sys.path.
    import torch
    from scripts.tools.data.tokenization import fara_qwen35_masking
    from scripts.tools.data.tokenization import tokenize_webwright_vlm_sft

    resolved = Path(tokenize_webwright_vlm_sft.__file__).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(
            f"imported WebWright tokenizer from {resolved}, expected it under {root}"
        )
    return tokenize_webwright_vlm_sft, fara_qwen35_masking, torch


def _set_worker_state(
    rows: list[dict[str, Any]],
    *,
    bundle_dir: Path,
    processor: Any,
    turn_tokens: Any,
    web_tokenizer: Any,
    masking: Any,
    torch_module: Any,
    processor_path: str,
    dataset_name: str,
    include_debug_text: bool,
) -> None:
    global _BUNDLE_DIR
    global _DATASET_NAME
    global _INCLUDE_DEBUG_TEXT
    global _MASKING
    global _PROCESSOR
    global _PROCESSOR_PATH
    global _ROWS
    global _TORCH
    global _TURN_TOKENS
    global _WEB_TOKENIZER

    _ROWS = rows
    _BUNDLE_DIR = bundle_dir
    _PROCESSOR = processor
    _TURN_TOKENS = turn_tokens
    _WEB_TOKENIZER = web_tokenizer
    _MASKING = masking
    _TORCH = torch_module
    _PROCESSOR_PATH = processor_path
    _DATASET_NAME = dataset_name
    _INCLUDE_DEBUG_TEXT = include_debug_text


def _clear_worker_state() -> None:
    global _BUNDLE_DIR
    global _DATASET_NAME
    global _INCLUDE_DEBUG_TEXT
    global _MASKING
    global _PROCESSOR
    global _PROCESSOR_PATH
    global _ROWS
    global _TORCH
    global _TURN_TOKENS
    global _WEB_TOKENIZER

    _ROWS = None
    _BUNDLE_DIR = None
    _PROCESSOR = None
    _TURN_TOKENS = None
    _WEB_TOKENIZER = None
    _MASKING = None
    _TORCH = None
    _PROCESSOR_PATH = None
    _DATASET_NAME = None
    _INCLUDE_DEBUG_TEXT = False


def _tokenize_source_index(source_index: int) -> TokenizedResult:
    if (
        _ROWS is None
        or _BUNDLE_DIR is None
        or _PROCESSOR is None
        or _TURN_TOKENS is None
        or _WEB_TOKENIZER is None
        or _MASKING is None
        or _TORCH is None
        or _PROCESSOR_PATH is None
        or _DATASET_NAME is None
    ):
        raise RuntimeError("tokenizer worker state is not initialized")

    source_row = _ROWS[source_index]
    source_row_id = str(source_row.get("id", source_index))
    prepared = _WEB_TOKENIZER._tokenize_sharegpt_row(
        source_row,
        source_index,
        bundle_dir=_BUNDLE_DIR,
        processor=_PROCESSOR,
        turn_tokens=_TURN_TOKENS,
        processor_path=_PROCESSOR_PATH,
        source_dataset=_DATASET_NAME,
        max_seq_len=None,
    )
    if prepared is None:
        raise RuntimeError(f"PhiTrain tokenizer rejected row {source_row_id}")

    input_ids = _TORCH.tensor(prepared["input_ids"], dtype=_TORCH.long)
    train_mask = _MASKING.construct_conversation_mask(
        input_ids,
        _TURN_TOKENS.assistant_start,
        _TURN_TOKENS.assistant_end,
        mask_all_but_last_n=1,
    )
    if not bool(train_mask.any()):
        raise RuntimeError(f"row {source_row_id} has no target assistant token span")

    labels = _TORCH.full_like(input_ids, IGNORE_INDEX)
    labels[train_mask] = input_ids[train_mask]
    # PhiTrain's Qwen pretokenizers append and supervise a terminal
    # <|endoftext|> after the final assistant turn.
    labels[-1] = input_ids[-1]
    prepared["labels"] = labels.tolist()
    prepared["assistant_label_span_count"] = _count_unmasked_label_spans(prepared["labels"])
    if prepared["assistant_label_span_count"] != 1:
        raise RuntimeError(
            f"row {source_row_id} has "
            f"{prepared['assistant_label_span_count']} unmasked spans; expected exactly 1"
        )
    prepared["prompt_version"] = PROMPT_VERSION
    if not _INCLUDE_DEBUG_TEXT:
        prepared.pop("rendered_text_debug", None)
        prepared.pop("unmasked_label_text_debug", None)

    return TokenizedResult(
        source_index=source_index,
        source_row_id=source_row_id,
        token_count=len(prepared["input_ids"]),
        row=prepared,
    )


def _iter_results(
    source_indices: list[int],
    *,
    num_workers: int,
    tqdm_module: Any,
) -> Iterable[TokenizedResult]:
    if num_workers == 1:
        for source_index in tqdm_module.tqdm(
            source_indices,
            total=len(source_indices),
            desc="Tokenizing SPB mask-last",
            unit="row",
        ):
            yield _tokenize_source_index(source_index)
        return

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("num_workers > 1 requires multiprocessing start method 'fork'")
    context = mp.get_context("fork")
    with context.Pool(processes=num_workers) as pool:
        results = pool.imap(_tokenize_source_index, source_indices, chunksize=1)
        yield from tqdm_module.tqdm(
            results,
            total=len(source_indices),
            desc="Tokenizing SPB mask-last",
            unit="row",
        )


def tokenize(args: argparse.Namespace) -> dict[str, Any]:
    phitrain_root = args.phitrain_root.expanduser().resolve()
    processor_path = args.processor_path.expanduser().resolve()
    chat_template = args.chat_template.expanduser().resolve()
    bundle_dir = args.bundle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().absolute()

    if not processor_path.is_dir():
        raise FileNotFoundError(f"Qwen processor/model directory not found: {processor_path}")
    if not chat_template.is_file():
        raise FileNotFoundError(f"chat template not found: {chat_template}")
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle directory not found: {bundle_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new path "
            "or remove the prior generated artifact explicitly"
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    web_tokenizer, masking, torch_module = _import_phitrain(phitrain_root)
    import tqdm

    rows, dataset_name = web_tokenizer._load_bundle(bundle_dir, args.dataset_name)
    source_indices = web_tokenizer._select_smoke_indices(len(rows), args.sample_rows)
    config = web_tokenizer.TokenizationConfig(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    processor = web_tokenizer._load_hf_processor(str(processor_path), config)
    processor.tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    turn_tokens = web_tokenizer._precompute_turn_tokens(processor)
    _set_worker_state(
        rows,
        bundle_dir=bundle_dir,
        processor=processor,
        turn_tokens=turn_tokens,
        web_tokenizer=web_tokenizer,
        masking=masking,
        torch_module=torch_module,
        processor_path=str(processor_path),
        dataset_name=dataset_name,
        include_debug_text=args.include_debug_text,
    )

    stage_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if stage_dir.exists():
        raise FileExistsError(f"staging directory already exists: {stage_dir}")
    stage_dir.mkdir(parents=True)

    all_lengths: list[int] = []
    retained_lengths: list[int] = []
    dropped: list[dict[str, Any]] = []

    def retained_rows() -> Iterable[dict[str, Any]]:
        for result in _iter_results(
            source_indices,
            num_workers=args.num_workers,
            tqdm_module=tqdm,
        ):
            all_lengths.append(result.token_count)
            if result.token_count > args.max_seq_len:
                dropped.append(
                    {
                        "source_index": result.source_index,
                        "source_row_id": result.source_row_id,
                        "token_count": result.token_count,
                    }
                )
                continue
            if result.row is None:
                raise RuntimeError(f"retained row {result.source_row_id} has no payload")
            retained_lengths.append(result.token_count)
            yield result.row

    try:
        manifest = web_tokenizer.write_fara_parquet_ref_dataset(
            retained_rows(),
            stage_dir,
            sample_rows_per_shard=args.sample_rows_per_shard,
            writer_workers=args.writer_workers,
        )
        if manifest["row_count"] == 0:
            raise ValueError(
                f"max_seq_len={args.max_seq_len} retained no tokenized training rows"
            )

        all_lengths.sort()
        retained_lengths.sort()
        manifest.update(
            {
                "source_row_count": len(source_indices),
                "retained_row_count": manifest["row_count"],
                "dropped_overlength_count": len(dropped),
                "max_seq_len": args.max_seq_len,
                "assistant_mask_mode": "last",
                "history_context_mode": "last_obs",
                "turn_mode": "single",
                "prompt_version": PROMPT_VERSION,
                "token_lengths_all": {
                    "min": all_lengths[0] if all_lengths else 0,
                    "p50": _percentile(all_lengths, 0.50),
                    "p90": _percentile(all_lengths, 0.90),
                    "p99": _percentile(all_lengths, 0.99),
                    "max": all_lengths[-1] if all_lengths else 0,
                },
                "token_lengths_retained": {
                    "min": retained_lengths[0] if retained_lengths else 0,
                    "p50": _percentile(retained_lengths, 0.50),
                    "p90": _percentile(retained_lengths, 0.90),
                    "p99": _percentile(retained_lengths, 0.99),
                    "max": retained_lengths[-1] if retained_lengths else 0,
                },
            }
        )
        (stage_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage_dir / "dropped_overlength_rows.json").write_text(
            json.dumps(dropped, indent=2) + "\n",
            encoding="utf-8",
        )

        provenance = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "command": shlex.join([str(sys.executable), *sys.argv]),
            "params": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "dataset_name": dataset_name,
            "bundle_manifest_sha256": _sha256_file(bundle_dir / "manifest.json"),
            "chat_template_sha256": _sha256_file(chat_template),
            "processor_path": str(processor_path),
            "phitrain_root": str(phitrain_root),
            "resolved_web_tokenizer_module": str(Path(web_tokenizer.__file__).resolve()),
            "resolved_masking_module": str(Path(masking.__file__).resolve()),
            "tokenization_config": asdict(config),
            "label_contract": {
                "ignore_index": IGNORE_INDEX,
                "assistant_mask_mode": "last",
                "target_assistant_spans_per_row": 1,
                "terminal_eos_supervised": True,
                "mask_empty_think": False,
            },
        }
        (stage_dir / "tokenization_run.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    finally:
        _clear_worker_state()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize a Qwen3.5 SPB last-observation single-turn bundle with "
            "loss only on the final assistant."
        )
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--phitrain-root", type=Path, default=DEFAULT_PHITRAIN_ROOT)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR_PATH)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--sample-rows-per-shard", type=int, default=16)
    parser.add_argument("--writer-workers", type=int, default=2)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--include-debug-text", action="store_true")
    args = parser.parse_args(argv)
    if args.max_seq_len <= 0:
        parser.error("--max-seq-len must be positive")
    if args.num_workers <= 0:
        parser.error("--num-workers must be positive")
    if args.sample_rows is not None and args.sample_rows <= 0:
        parser.error("--sample-rows must be positive")
    if args.sample_rows_per_shard <= 0:
        parser.error("--sample-rows-per-shard must be positive")
    if args.writer_workers <= 0:
        parser.error("--writer-workers must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = tokenize(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

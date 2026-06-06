"""WebAgentTaskDataset — loads Online-Mind2Web style tasks.

The terminal-agent dataset ships a tarred Docker image per row. For the web
agent the "environment" is just a Playwright tab, so the per-row payload is
only the task metadata (id, prompt text, starting URL, level). We keep the
same column names that the rollout glue expects, but the ``task_binary``
column is now an empty bytes placeholder.

Input formats handled (auto-detected by file extension):

- ``.csv`` — the canonical ``Online_Mind2Web.csv`` shape with columns
  ``task_id, confirmed_task, website, reference_length, level``.
- ``.json`` — either ``[{"id": ..., "task": ..., "website_url": ...}, ...]``
  or ``{"tasks": [...]}`` or the older ``om2w_*.json`` shape with
  ``task_id`` / ``confirmed_task`` / ``website``.

Outputs a HuggingFace ``datasets.Dataset`` with columns:
- ``prompt``     — chat-template messages list
- ``path``       — task id, used for trajectory_id and metric grouping
- ``task_binary``— empty bytes (kept for compatibility with the generator)
- ``task_meta``  — JSON-serialized {"task_id", "task", "start_url", "level"}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import datasets
from transformers import PreTrainedTokenizerBase

from .prompts import format_web_task_prompt


def _normalize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    task_id = str(
        record.get("task_id")
        or record.get("id")
        or ""
    ).strip()
    task_text = str(
        record.get("task")
        or record.get("confirmed_task")
        or record.get("query")
        or ""
    ).strip()
    start_url = str(
        record.get("start_url")
        or record.get("website_url")
        or record.get("website")
        or record.get("url")
        or ""
    ).strip()
    if not (task_id and task_text and start_url):
        return None
    return {
        "task_id": task_id,
        "task": task_text,
        "start_url": start_url,
        "level": str(record.get("level", "") or "").strip(),
        "reference_length": int(record.get("reference_length") or 0),
    }


def load_om2w_records(source: str | Path) -> list[dict[str, Any]]:
    path = Path(source).expanduser()
    items: list[Any]
    if path.suffix.lower() == ".csv":
        import csv

        with path.open(encoding="utf-8", newline="") as f:
            items = list(csv.DictReader(f))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("tasks") or payload.get("data") or []
        else:
            items = payload
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_record(item)
        if normalized is not None:
            out.append(normalized)
    return out


class WebAgentTaskDataset:
    """Dataset wrapper compatible with the terminal-agent generator interface."""

    def __init__(
        self,
        data_files: list[str | dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        parser_name: str = "qwen35",
        prompt_key: str = "prompt",
        path_key: str = "path",
        task_binary_key: str = "task_binary",
        num_workers: int = 4,
        max_rows: int | None = None,
        max_rows_per_source: int | None = None,
        add_instruction_prefix: bool = True,
        prompt_mode: str = "default",
    ) -> None:
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.parser_name = parser_name
        self.prompt_key = prompt_key
        self.path_key = path_key
        self.task_binary_key = task_binary_key
        self.num_workers = num_workers
        self.max_rows = max_rows
        self.max_rows_per_source = max_rows_per_source
        self.add_instruction_prefix = add_instruction_prefix
        self.prompt_mode = prompt_mode
        self._read_files_and_tokenize()

    def _normalize_source(self, source: str | dict[str, Any]) -> tuple[str, str]:
        if isinstance(source, str):
            return source, os.path.splitext(os.path.basename(source))[0]
        if not isinstance(source, dict):
            raise TypeError(f"web-agent data source must be a path or dict, got {type(source)}")
        path = source.get("path") or source.get("file") or source.get("data")
        if not isinstance(path, str):
            raise ValueError("web-agent data source dict must include a string 'path', 'file', or 'data'.")
        data_source = source.get("name") or source.get("data_source") or os.path.splitext(os.path.basename(path))[0]
        return path, str(data_source)

    def _load_file(self, source: str) -> datasets.Dataset:
        ext = os.path.splitext(source)[-1].lower()
        if ext == ".parquet":
            return datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
        # JSON / JSONL handled by HF for line-delimited, or by us for normal arrays.
        records = load_om2w_records(source)
        return datasets.Dataset.from_list(records)

    def _make_prompt(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        return format_web_task_prompt(
            task=record["task"],
            start_url=record["start_url"],
            parser_name=self.parser_name,
            tokenizer=self.tokenizer,
            add_instruction_prefix=self.add_instruction_prefix,
            mode=self.prompt_mode,
            task_id=record["task_id"],
        )

    def _read_files_and_tokenize(self) -> None:
        loaded = []
        for source in self.data_files:
            path, data_source = self._normalize_source(source)
            dataset = self._load_file(path)
            if self.max_rows_per_source is not None:
                dataset = dataset.select(range(min(self.max_rows_per_source, len(dataset))))
            dataset = dataset.map(lambda row, _ds=data_source: {**row, "_data_source": _ds})
            loaded.append(dataset)
        if not loaded:
            raise ValueError("No data sources provided to WebAgentTaskDataset.")
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded)
        if self.max_rows is not None:
            self.dataframe = self.dataframe.select(range(min(self.max_rows, len(self.dataframe))))

        def _build(row: dict[str, Any]) -> dict[str, Any]:
            prompt = self._make_prompt(row)
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            return {
                **row,
                self.prompt_key: prompt,
                self.path_key: row["task_id"],
                self.task_binary_key: b"",  # placeholder — env is Playwright not Docker
                "task_meta": json.dumps(
                    {
                        "task_id": row["task_id"],
                        "task": row["task"],
                        "start_url": row["start_url"],
                        "level": row.get("level", ""),
                        "reference_length": row.get("reference_length", 0),
                    }
                ),
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_count": len(prompt_token_ids),
            }

        self.dataframe = self.dataframe.map(_build)
        self.dataframe = self.dataframe.filter(
            lambda row: row["prompt_token_count"] <= self.max_prompt_length
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe[index]
        meta = json.loads(row["task_meta"])
        env_extras = {
            "path": row[self.path_key],
            "task_binary": row[self.task_binary_key],
            "task_meta": meta,
            "prompt_token_ids": row["prompt_token_ids"],
            "data_source": row.get("_data_source"),
            "world_model_only": False,
        }
        return {
            "prompt": row[self.prompt_key],
            "env_class": "web_agent",
            "env_extras": env_extras,
            "uid": str(index),
        }

    def __len__(self) -> int:
        return len(self.dataframe)

    def collate_fn(self, item_list):
        return item_list

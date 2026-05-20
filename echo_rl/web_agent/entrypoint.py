"""Web-agent SkyRL entrypoint.

Mirrors ``echo_rl.terminal_agent.entrypoint`` but swaps the dataset and
generator. Reuses ``EchoTerminalAgentSkyRLConfig`` so all SkyRL training
plumbing (FSDP workers, world-model loss hook, optimizer, schedulers) is
unchanged.

Note: we intentionally do NOT use ``from __future__ import annotations`` here
because SkyRL's ``build_nested_dataclass`` reads dataclass field types
directly and cannot resolve stringified annotations.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ray
from omegaconf import OmegaConf

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from echo_rl.terminal_agent.chat_template import load_chat_template, resolve_chat_template_path
from echo_rl.terminal_agent.entrypoint import (
    EchoTerminalAgentExp,
    EchoTerminalAgentSkyRLConfig,
    TerminalAgentGeneratorConfig,
)

from .dataset import WebAgentTaskDataset
from .web_agent_generator import WebAgentGenerator


@dataclass
class WebAgentGeneratorConfig(TerminalAgentGeneratorConfig):
    """Adds web-specific knobs on top of the terminal-agent config."""

    reward: dict[str, Any] = field(default_factory=lambda: {"name": "osw_judge"})
    env_overrides: dict[str, Any] = field(default_factory=dict)
    stub_env: bool = False
    # Selects which system-prompt mode to use from
    # ``echo_rl.web_agent.prompts._INSTRUCTION_PREFIX_REGISTRY``.
    # Currently: "default" (single persistent tab pre-injected) or
    # "self_launch_persistent_browser" (agent orchestrates a persistent
    # Browserbase session plus on-demand exploration sessions itself).
    prompt_mode: str = "default"


@dataclass
class WebAgentSkyRLConfig(EchoTerminalAgentSkyRLConfig):
    generator: WebAgentGeneratorConfig = field(default_factory=WebAgentGeneratorConfig)


class WebAgentExp(EchoTerminalAgentExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        self._ensure_chat_template_loaded()
        return WebAgentGenerator(
            generator_cfg=cfg.generator,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        self._ensure_chat_template_loaded()
        ds = WebAgentTaskDataset(
            data_files=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            parser_name=self.cfg.generator.parser_name,
            prompt_key=self.cfg.generator.prompt_key,
            path_key=self.cfg.generator.path_key,
            task_binary_key=self.cfg.generator.task_binary_key,
            num_workers=self.cfg.generator.dataset_num_workers,
            max_rows=self.cfg.generator.dataset_max_rows,
            prompt_mode=self.cfg.generator.prompt_mode,
        )
        assert len(ds) >= self.cfg.trainer.train_batch_size, (
            f"dataset should be at least as large as train_batch_size "
            f"{self.cfg.trainer.train_batch_size}, got size {len(ds)}"
        )
        return ds

    def get_eval_dataset(self):
        self._ensure_chat_template_loaded()
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return WebAgentTaskDataset(
                data_files=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                parser_name=self.cfg.generator.parser_name,
                prompt_key=self.cfg.generator.prompt_key,
                path_key=self.cfg.generator.path_key,
                task_binary_key=self.cfg.generator.task_binary_key,
                num_workers=self.cfg.generator.dataset_num_workers,
                max_rows=self.cfg.generator.dataset_max_rows,
                max_rows_per_source=self.cfg.generator.eval_dataset_max_rows_per_source,
                prompt_mode=self.cfg.generator.prompt_mode,
            )
        return None


def _load_config(argv: list[str]) -> WebAgentSkyRLConfig:
    config_path: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a config path")
            config_path = argv[i + 1]
            i += 2
            continue
        if arg.startswith("config="):
            config_path = arg.split("=", 1)[1]
            i += 1
            continue
        remaining.append(arg)
        i += 1

    if config_path is None:
        return WebAgentSkyRLConfig.from_cli_overrides(remaining)

    base = OmegaConf.load(Path(config_path))
    overrides = OmegaConf.from_cli(remaining)
    merged = OmegaConf.merge(base, overrides)
    return WebAgentSkyRLConfig.from_dict_config(merged)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = WebAgentExp(cfg)
    exp.run()


def main() -> None:
    cfg = _load_config(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.generator.step_wise_trajectories:
        raise ValueError("WebAgentGenerator requires generator.step_wise_trajectories=false.")
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for web-agent training.")
    if cfg.trainer.strategy not in ("fsdp", "fsdp2"):
        raise ValueError("Web-agent training currently supports only fsdp/fsdp2.")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

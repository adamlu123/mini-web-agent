"""Eval-only entrypoint for the web-agent (measure base-model performance).

Mirrors ``skyrl.train.entrypoints.main_generate`` (SkyRL's generic eval-only
entrypoint) but subclasses ``WebAgentExp`` so it reuses the web-agent dataset,
generator, prompts (echo_rl/web_agent/prompts.py), and OSW judge reward instead
of the generic gym ones.

It runs a single pass of ``evaluate()`` over ``data.val_data`` -- no FSDP
policy/ref workers, no optimizer, no training loop, no checkpoints. The vLLM
engine loads the *base* HF weights from ``trainer.policy.model.path``, so the
logged ``eval/<data_source>/...`` metrics are the untuned base performance.

Each ``data.val_data`` entry's ``name`` becomes the ``data_source`` bucket in
the logged metrics, so listing several parquets scores each one separately in
the same run.

Config handling and validation are identical to the training entrypoint
(``echo_rl.web_agent.entrypoint``): same ``_load_config`` and ``validate_cfg``,
so every rollout knob (max_turns, max context/seq lengths, eval sampling params,
eval_n_samples_per_prompt, env overrides, judge, ...) behaves exactly as it does
for the eval step inside a training run.
"""

import asyncio
import sys
from typing import Any

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl.train.evaluate import evaluate
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray

from .entrypoint import WebAgentExp, _load_config


class WebAgentEvalOnlyExp(WebAgentExp):
    def get_train_dataset(self):
        # Eval-only: no training dataset, so skip WebAgentExp's train_batch_size
        # assertion (which would otherwise require data.train_data).
        return None

    async def run(self, inference_engine_client: InferenceEngineInterface) -> dict[str, Any]:
        assert self.eval_dataset is not None, (
            "eval-only entrypoint requires data.val_data and trainer.eval_interval > 0"
        )

        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        results: dict[str, Any] = await evaluate(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg):
    exp = WebAgentEvalOnlyExp(cfg)
    # Build the inference client from a sync context so colocated-mode setup can
    # run its own asyncio.run() for the initial sleep step (mirrors main_generate).
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = _load_config(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.generator.step_wise_trajectories:
        raise ValueError("WebAgentGenerator requires generator.step_wise_trajectories=false.")
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for web-agent eval.")
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from web-agent eval-only run: {metrics}")


if __name__ == "__main__":
    main()

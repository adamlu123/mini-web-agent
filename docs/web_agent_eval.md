# Web-agent evaluation / inference

How to run the web-agent **eval-only** (inference) pipeline — measure a model's
Online-Mind2Web pass@1 without any training, FSDP policy/ref workers, optimizer,
or checkpoints. The vLLM engine just serves HF weights and rolls out the same
agent loop / OSW judge as a training-time eval step.

For training, see [`web_agent_training.md`](web_agent_training.md). Latest
numbers live in [`../eval_outputs/RESULTS.md`](../eval_outputs/RESULTS.md).

## What this is

`echo_rl.web_agent.eval_entrypoint` subclasses the training `WebAgentExp` and
runs a single `evaluate()` pass over `data.val_data`, so it reuses the exact
web-agent dataset, generator, prompts, env, and reward used in training — only
the FSDP policy/ref workers, optimizer, and training loop are dropped. Each
`val_data` entry's `name` becomes its own `data_source` bucket, so one run scores
the easy `train` (68) and `val` (12) parquets separately, plus an `all` (80)
aggregate.

`avg_score == pass_at_1` because the local driver sets
`eval_n_samples_per_prompt=1`.

## The inference-code bundle

| File | Role |
| --- | --- |
| `echo_rl/web_agent/eval_entrypoint.py` | eval-only SkyRL entrypoint (no training) |
| `run_local_eval.sh` | local 4×GPU driver (`MODEL=4b\|9b`, optional `CKPT=`) |
| `configs/qwen35_4b_web_agent_easy_eval.yaml` | 4B eval config (derived from the 4B easy training config) |
| `configs/qwen35_9b_web_agent_easy_eval.yaml` | 9B eval config |
| `echo_rl/web_agent/scripts/convert_fsdp_ckpt_to_hf.py` | merge an FSDP2 sharded checkpoint → HF safetensors so it can be evaluated |
| `echo_rl/web_agent/scripts/collect_eval_results.py` | consolidate all `eval_outputs/*` runs into one markdown table |
| `docker/submit_eval_easy.sh` + `docker/run_eval_in_pod.sh` | submit the same eval to the bonete61 cluster |

## 1. Local eval of the base model (4×GPU)

Needs Browserbase + an OpenAI key for the o4-mini judge (creds come from
`/home/luyadong/cred.sh`; see the script header). Then:

```bash
MODEL=4b bash run_local_eval.sh     # Qwen3.5-4B base   (or MODEL=9b)
```

Output lands in `eval_outputs/<model>_easy_<timestamp>/`:

- `exports/dumped_evals/eval_only/aggregated_results.jsonl` — the metrics
- `exports/dumped_evals/eval_only/web_agent_om2w_easy_{train,val}.jsonl` — per-task rollouts
- `rollouts/<id>/.persistent_session.json` — one per task
- the console log prints `Metrics from web-agent eval-only run: {...}`

> **Why `colocate_all=false`.** The driver forces it. With the training default
> `colocate_all=true`, SkyRL sleeps the vLLM engine at level=2 right after
> startup expecting an NCCL weight-sync from the FSDP policy worker. Eval-only
> has no policy worker, so the engine wakes with corrupted weights and emits pure
> gibberish → every turn parse-errors → all scores 0. `colocate_all=false` skips
> the sleep. (See the 0-score `4b_easy_20260528_*` runs in `RESULTS.md` — those
> predate this fix.)

## 2. Local eval of a trained checkpoint

SkyRL writes the policy as an **FSDP2 sharded** checkpoint
(`outputs/<run>/ckpts/global_step_N/policy/model_world_size_4_rank_*.pt`), which
vLLM cannot load directly. Merge it to HF safetensors first:

```bash
python -m echo_rl.web_agent.scripts.convert_fsdp_ckpt_to_hf \
    --ckpt    outputs/qwen35_4b_easy_4gpu/ckpts/global_step_9/policy \
    --output  outputs/qwen35_4b_easy_4gpu/hf/global_step_9 \
    --aux-from Qwen/Qwen3.5-4B          # see note below
```

The converter concatenates each parameter's per-rank `Shard(dim=0)` DTensor
locals and trims FSDP's trailing-rank padding, then writes sharded safetensors +
index + config + tokenizer. Then point the eval at it:

```bash
MODEL=4b RUN_TAG=step9 \
  CKPT=outputs/qwen35_4b_easy_4gpu/hf/global_step_9 \
  bash run_local_eval.sh
```

`CKPT` overrides `trainer.policy.model.path`; `RUN_TAG` tags the output dir
(`eval_outputs/4b_easy_step9_<timestamp>/`).

> **Why `--aux-from`.** Qwen3.5 is a multimodal arch (`Qwen3_5ForConditional`
> `Generation`, `model_type: qwen3_5`), so vLLM loads it as a VL model and
> requires `preprocessor_config.json` / `video_preprocessor_config.json`, which
> the FSDP checkpoint does not store. `--aux-from <base model id or dir>` copies
> those processor files from the base model snapshot. Without them vLLM raises
> `OSError: Can't load image processor for ...`.

## 3. Cluster eval (bonete61)

Submits the same eval-only run on B200s (4 GPUs, no policy/ref):

```bash
MODEL=4b bash docker/submit_eval_easy.sh      # or MODEL=9b
```

The heavy in-pod setup lives in `docker/run_eval_in_pod.sh` (uploaded with the
repo); the Volcano `--cmd` is a tiny one-liner that execs it (keeps the
create-request body WAF-clean behind Cloudflare). The judge key is delivered via
the `echo-rl-openai` k8s secret. See the script headers for details. Override
the config / gpus with `CONFIG=... GPUS=... bash docker/submit_eval_easy.sh`.

## 4. Reading / consolidating results

```bash
python -m echo_rl.web_agent.scripts.collect_eval_results \
    --root eval_outputs --out eval_outputs/RESULTS.md
```

Scans every `eval_outputs/*/exports/dumped_evals/eval_only/
aggregated_results.jsonl` and emits the per-run table (train/val/all pass@1,
avg turns/tokens, parse/env errors). See
[`../eval_outputs/RESULTS.md`](../eval_outputs/RESULTS.md) for the current
headline (base 4B 7.5%, base 9B 5.0% on easy/80).

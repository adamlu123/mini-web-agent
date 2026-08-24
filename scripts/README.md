# Script command guide

Use the concern-based paths below for commands and automation. Executable
scripts live in concern-specific subdirectories; the `scripts/` root contains
only this guide.

## Folders

- `archive/`: Old scripts kept for reference; do not use for active jobs.
- `cluster/`: Submit and run cluster evaluation jobs.
- `data/`: Prepare and tokenize training data.
- `eval/`: Evaluate completed agent runs.
- `lib/`: Shared helpers used by other scripts.
- `local/`: Serve models and run evaluations locally.
- `review/`: Serve results and open review tunnels.
- `usage/`: Report gateway spend from Application Insights.

## Command matrix

| Concern | Canonical command | Purpose |
| --- | --- | --- |
| Local OM2W | `bash scripts/local/om2w/serve_qwen35.sh` | Serve a local Qwen3.5 checkpoint with vLLM. |
| Local OM2W | `bash scripts/local/om2w/run.sh` | Run OM2W against one OpenAI-compatible local endpoint. |
| Local OM2W | `bash scripts/local/om2w/shards.sh <start\|stop\|status\|logs\|sessions>` | Split and manage a run across local vLLM ports. |
| Qwen3.5 cluster eval | `bash scripts/cluster/om2w/qwen35_9b/submit.sh [--dry-run]` | Submit the Qwen3.5-9B checkpoint evaluation. |
| Qwen3.5 cluster runtime | `bash scripts/cluster/om2w/qwen35_9b/run.sh` | Internal entry point used inside the Qwen3.5 job. |
| Qwen3.5-4B cluster eval | `bash scripts/cluster/om2w/qwen35_4b/submit.sh [--dry-run]` | Submit the Qwen3.5-4B RL actor checkpoint evaluation. |
| Qwen3.5-4B cluster runtime | `bash scripts/cluster/om2w/qwen35_4b/run.sh` | Internal entry point used inside the Qwen3.5-4B job. |
| Qwen3.5-4B base cluster eval | `bash scripts/cluster/om2w/qwen35_4b_base/submit.sh [--dry-run]` | Submit the untrained Qwen3.5-4B student baseline evaluation. |
| Qwen3.5-4B base cluster runtime | `bash scripts/cluster/om2w/qwen35_4b_base/run.sh` | Internal entry point used inside the Qwen3.5-4B base job. |
| Phi-4-14B cluster eval | `bash scripts/cluster/om2w/phi4_14b/submit.sh [--dry-run]` | Submit the Phi-4-14B webwright SFT checkpoint evaluation. |
| Phi-4-14B cluster runtime | `bash scripts/cluster/om2w/phi4_14b/run.sh` | Internal entry point used inside the Phi-4-14B job. |
| Qwen3.6 cluster eval | `bash scripts/cluster/om2w/qwen36_27b/submit.sh [--dry-run]` | Submit the Qwen3.6-27B evaluation. |
| Qwen3.6 cluster runtime | `bash scripts/cluster/om2w/qwen36_27b/run.sh` | Internal entry point used inside the Qwen3.6 job. |
| Qwen3.8 cluster eval | `bash scripts/cluster/om2w/qwen38_27b/submit.sh [--dry-run]` | Submit the stock Qwen3.8-27B evaluation. |
| Qwen3.8 cluster runtime | `bash scripts/cluster/om2w/qwen38_27b/run.sh` | Internal entry point used inside the Qwen3.8 job. |
| Judge-only cluster eval | `bash scripts/cluster/om2w/judge_only/submit.sh [--dry-run]` | Submit judging for existing run artifacts. |
| Judge-only cluster runtime | `bash scripts/cluster/om2w/judge_only/run.sh` | Internal entry point used inside the judge-only job. |
| Persistent evaluation | `python scripts/eval/persistent_cli.py --help` | Evaluate `browser-steps.jsonl` trajectories. |
| Persistent step evaluation | `python scripts/eval/persistent_cli_steps.py --help` | Evaluate `steps/*.sh` or `result.json` action histories. |
| Qwen3.5 data | `python scripts/data/qwen35/preprocess_lastobs_singleturn.py --help` | Build a last-observation, single-turn ShareGPT bundle. |
| Qwen3.5 data | `python scripts/data/qwen35/tokenize_lastobs_singleturn.py --help` | Tokenize the bundle for PhiTrain VLM SFT. |
| Review | `bash scripts/review/viewer.sh [port]` | Serve run and judge artifacts locally. |
| Review | `bash scripts/review/tunnel.sh [port]` | Create a public Cloudflare tunnel to the viewer. |
| Gateway spend | `python scripts/usage/gateway_spend.py [--hours N]` | Report your gateway spend by model, resolving `$OPENAI_GATEWAY_API_KEY` to its owner. |
| Gateway spend | `python scripts/usage/gateway_spend.py --team Fara` | Report a team's spend per user. |

## Common examples

Serve a checkpoint and run one local evaluation:

```bash
CKPT=/path/to/checkpoint GPUS=0 TP=1 PORT=8000 \
  bash scripts/local/om2w/serve_qwen35.sh

CFG=eval/om2w_spb_vllm_sw10.yaml \
ENDPOINT=http://127.0.0.1:8000/v1 \
OUT=outputs/qwen35_om2w \
  bash scripts/local/om2w/run.sh
```

Manage a run sharded over two already-running servers:

```bash
PORTS="8000 8001" NTASKS=40 TASK_LEVEL=medium WORKERS=8 \
  bash scripts/local/om2w/shards.sh start
bash scripts/local/om2w/shards.sh status
bash scripts/local/om2w/shards.sh stop
```

Preview a cluster submission before creating a job:

```bash
bash scripts/cluster/om2w/qwen35_9b/submit.sh --dry-run
```

Serve and share the review viewer:

```bash
bash scripts/review/viewer.sh 8010
# In a second terminal:
bash scripts/review/tunnel.sh 8010
```

## Environment conventions

- `REPO` overrides automatic repository-root discovery for local scripts.
- `CREDENTIALS_FILE` selects the credential file used by the local runner;
  `CRED_FILE` remains a legacy alias.
- `VENV_BIN` selects the runner virtual environment's `bin` directory.
- Cluster launchers list their supported environment overrides in `--help`.
- Script-specific model, endpoint, output, and worker defaults are documented
  in each entry point's header or help output.
- The tunnel helper prefers `cloudflared` on `PATH`; otherwise it downloads the
  pinned 2026.3.0 binary into the user cache and verifies the GitHub release
  checksum. `CLOUDFLARED_BIN` selects an existing custom binary.

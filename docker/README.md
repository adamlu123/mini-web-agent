# Cluster scripts — echo_rl web-agent RL on bonete61 (B200)

Launch scripts for running `echo_rl.web_agent` RL on the Microsoft Lambda
**bonete61** cluster (NVIDIA B200, 8 GPU/node) using the **generic qwen3.5 NGC
image** rather than a hand-built echo-rl image:

```
aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415
```

This image bakes the compiled stack (torch 2.10 / TE 2.13 / deepspeed /
flash-attn / **vllm 0.18 + cu130**) on which **Qwen3.5 actually runs** (the older
echo-rl image's vllm 0.19 corrupts Qwen3.5 weights → multilingual gibberish).

## Scripts

| Script | What it does |
|--------|--------------|
| `submit_debug_q35_image.sh` | Submit a long-lived 8×B200 **debug** pod (`sleep infinity`); `kubectl exec` into it for an interactive, fully-bootstrapped shell. |
| `run_debug_q35_image.sh` | In-pod driver for the debug pod (uploaded, run via the submit's tiny WAF-safe `--cmd`). |
| `submit_train_q35_image.sh` | Submit a non-interactive **training** job (runs `entrypoint` to completion, `--follow-logs`). Default config = Qwen3.5-9B / 8 GPU. |
| `run_train_q35_image.sh` | In-pod driver for the training job (same bootstrap as debug, but launches training instead of sleeping). |
| `requirements.txt` | **Load-bearing** — the drivers parse it to decide which deps to install. Do not delete. |

## Usage

```bash
# Interactive debug pod (8 GPU), then exec in and run any config by hand:
bash docker/submit_debug_q35_image.sh
kubectl -n bonete61 get pods | grep cua          # wait for Running + .debug_ready
kubectl -n bonete61 exec -it <pod> -- bash       # lands ready: env+creds set, cwd=code/SkyRL
python -m echo_rl.web_agent.entrypoint --config echo_configs/qwen35_9b_web_agent_easy_8gpu.yaml

# Non-interactive training job (defaults to 9B/8GPU, full 16/16/10 hyperparams):
bash docker/submit_train_q35_image.sh
CONFIG=echo_configs/qwen35_4b_web_agent_easy_4gpu.yaml GPUS=4 bash docker/submit_train_q35_image.sh

# Tear down:
kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false
```

Training hyperparameters live entirely in the config (`echo_configs/*.yaml` inside
the uploaded SkyRL); the submit scripts pass no overrides — only resources, image,
creds (via k8s secret volumes), and `TRAIN_CONFIG`.

## How the drivers bootstrap the env (q35-image specifics)

The generic image is a near-superset of `requirements.txt` but its **compiled
CUDA/torch stack must NOT be touched** (installing the cu12 wheels from
`requirements.txt` shadows the image's cu130 NCCL → `undefined symbol:
ncclAlltoAll` → torch dead). So each driver:

1. **Deny-lists** the compiled stack (`nvidia-*`, `cuda-*`, `nixl*`, torch/vllm/
   triton/flash*/flashinfer/xgrammar…) — never reinstalled; the image's versions stand.
2. **Force-upgrades** the few app deps the image ships too old/mismatched:
   `omegaconf` 2.0.0 → 2.3.0 (its `${oc.env:…}` resolver needs ≥2.1) + its
   version-locked `antlr4-python3-runtime` 4.9.3, and pins `ray` → 2.51.1
   (SkyRL's pin; the image's ray has drifted across rebuilds).
3. Installs the genuinely-missing pure-add deps (`--no-deps`), incl. **vllm-router**
   (a pure-Python proxy the image lacks; SkyRL's inference setup needs it).
4. `pip install --no-deps -e` the 4 editables (SkyRL, skyrl-gym, skyrl-agent,
   mini-web-agent).
5. Adds a defensive `ray.util.placement_group` re-export shim (ray 2.55 moved
   `PlacementGroupSchedulingStrategy` to `ray.util.scheduling_strategies`).

Creds come from two k8s secret volumes (`echo-rl-creds`, `echo-rl-openai`) so no
secret ever lands in the `kubectl create` request body (Cloudflare WAF).

> **Note:** the configs set `distributed_executor_backend: mp` (not the default
> `ray`). With TP=1 on this image's vllm 0.18, the ray-executor path mis-places
> all engines onto GPU 0 → CUDA OOM; `mp` makes SkyRL set per-engine
> `CUDA_VISIBLE_DEVICES` so the engines spread across the GPUs.

# Web-agent RL — cluster job submission

Two entry points on the Lambda cluster:

- `docker/submit_interactive_8gpu.sh` — interactive 8×B200 pod for debugging.
- `docker/submit_real_train_batch.sh` — batch RL training (single 8-GPU node, time-capped).

## Prerequisites (one-time)

The `echo-rl-creds` k8s secret must already exist in the target namespace. It mounts a sanitized `cred.sh` at `/run/secrets/echo-rl-creds/cred.sh` inside the pod.

```
kubectl create secret generic echo-rl-creds \
    --from-file=cred.sh=<path to sanitized cred> -n bonete61
```

## Variables (shared by both scripts)

All have defaults; override via env before invocation.

| Variable               | Default                                                         | Purpose                                       |
| ---------------------- | --------------------------------------------------------------- | --------------------------------------------- |
| `SUBMIT`               | `/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh` | Lambda submission entrypoint.                |
| `MINI_WEB_AGENT_DIR`   | `/data/t-yifeili/mini-web-agent`                                | Local mini-web-agent repo uploaded to pod.    |
| `SKYRL_DIR`            | `/data/t-yifeili/SkyRL`                                         | Local SkyRL repo uploaded to pod.             |
| `IMAGE`                | `aifrontiers.azurecr.io/t-yifeili/echo-rl:latest`               | Container image.                              |
| `PRIORITY`             | `medium`                                                        | Label in `JOB_NAME` only.                     |
| `PRIORITY_CLASS_NAME`  | `medium`                                                        | Actual k8s PriorityClass (Volcano scheduling). Valid: `high` / `medium` / `low`. |

Batch-only:

| Variable | Default                                          | Purpose                                       |
| -------- | ------------------------------------------------ | --------------------------------------------- |
| `CONFIG` | `echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml` | SkyRL training config (relative to `SkyRL/`). |

## Interactive debug (8×B200 pod)

Submit:

```
bash docker/submit_interactive_8gpu.sh
```

Once the pod is running, attach and run inside:

```
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY

cd $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/SkyRL
pip install --no-deps -e . -e ./skyrl-gym -e ./skyrl-agent
cd ../mini-web-agent && pip install --no-deps -e .

export ECHO_RL_DATA=$(pwd)/data/web_agent
export OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
mkdir -p $OUTPUT_DIR $ECHO_RL_DATA

cd ../SkyRL
python -m echo_rl.web_agent.entrypoint \
    --config echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml
```

`Ctrl-D` exits the pod, which auto-deletes the job and releases the GPUs.

Common overrides:

```
PRIORITY=high PRIORITY_CLASS_NAME=high \
    bash docker/submit_interactive_8gpu.sh

IMAGE=aifrontiers.azurecr.io/t-yifeili/echo-rl:<tag> \
    bash docker/submit_interactive_8gpu.sh
```

## Batch training run

Submit and follow logs:

```
bash docker/submit_real_train_batch.sh
```

The script handles everything inside the pod automatically: sources creds, installs editables, sets `MINI_WEB_AGENT_ROOT` / `ECHO_RL_DATA` / `OUTPUT_DIR`, runs `nvidia-smi -L`, then launches:

```
python -m echo_rl.web_agent.entrypoint --config ${CONFIG}
```

No in-script time cap — the run finishes when training completes (or you kill the Volcano job manually: `kubectl -n bonete61 delete job.batch.volcano.sh/<job-name>`).

Common overrides:

```
# Different config
CONFIG=echo_configs/qwen35_4b_web_agent.yaml \
    bash docker/submit_real_train_batch.sh

# Higher priority
PRIORITY=high PRIORITY_CLASS_NAME=high \
    bash docker/submit_real_train_batch.sh
```

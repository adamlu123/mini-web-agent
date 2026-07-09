---
name: bonete-submit
description: Submit GPU/CPU training jobs to the bonete61 Volcano cluster (NVIDIA B200) via aifsdk's submit_job.sh. Use when the user wants to launch, name, monitor, or retrieve outputs from training/eval jobs on the lambda/bonete61 cluster. Covers code upload, pod naming for the GPU monitor dashboard, WandB hookup, checkpoint persistence on Vast PVC, and how to copy ckpts back to the gcrsandbox dev box. Triggers on phrases like "submit to bonete", "run training on B200", "kubectl bonete61", "/mnt/pvc", "lambda cluster job", "GPU monitor others bucket", "copy ckpt from PVC".
---

# Submitting jobs to bonete61

bonete61 is Microsoft's internal Volcano-on-k8s cluster (NVIDIA B200 × 8/node). The canonical submitter is `/data/<alias>/aifsdk/clusters/lambda/submission/submit_job.sh`. It wraps Volcano YAML, code upload (tar + `kubectl cp` to an uploader pod, then onto Vast PVC), ACR pull secrets, and `--follow-logs`.

## Cluster facts (don't re-derive)

| What | Value |
|---|---|
| Namespace | `bonete61` |
| Volcano queue | `bonete61` |
| Default image | `nvcr.io/nvidia/pytorch:25.08-py3` (already cached on most nodes) |
| Service account | `aif-bonete-uami` (workload identity for Azure) |
| PVC | `pvc-vast-bonete61`, **550 TiB Vast NFS**, mounted at `/mnt/pvc` **inside pods only** |
| GPU per node | 8 × NVIDIA-B200 |
| PriorityClasses available | **`high` / `medium` / `low` only** (no `p0`/`p1` — that's a different concept) |
| ACR | `aifrontiers.azurecr.io` |
| WandB host | `https://microsoft-research.wandb.io`, default project `phitrain` |

## Step 1 — Pre-flight (.gitignore)

`submit_job.sh` uses `tar --exclude-vcs-ignores` (GNU tar). **Trap: GNU tar does NOT honor `dir/` style patterns** — only bare names. If you write `outputs/` or `docker/wheels/` in `.gitignore`, tar silently includes them and your upload balloons.

Correct `.gitignore` entries:
```gitignore
# bare names — GNU tar's --exclude-vcs-ignores ignores `dir/` patterns
outputs
docker/wheels
.claude
*.whl
```

Heavy upload artifacts to exclude in any mini-web-agent / SkyRL style repo:
- `docker/wheels/` (causal_conv1d, flash_attn → 300 MB+ each)
- `outputs/` (previous training runs, screenshots, etc.)
- Banner / doc assets (`*.gif`, `*.pdf`) unless they're actual training data

Validate by dry-running tar locally:
```bash
cd <repo>
tmp=$(mktemp /tmp/dryrun_XXX.tar.gz)
tar --exclude-vcs-ignores --exclude-vcs -czf "$tmp" .
du -sh "$tmp"
tar -tzvf "$tmp" | awk '{print $3, $6}' | sort -rn | head -10
rm "$tmp"
```
Aim for <50 MB. If a single file >5 MB shows up that isn't training-relevant, add to `.gitignore`.

## Step 2 — Name the job correctly (the GPU monitor "others" bucket gotcha)

The internal GPU monitor dashboard (at `https://gcrazgdl1510-80-g5a7bugxb4fchsgc.b01.azurefd.net/`) **parses pod names** to bucket jobs. Format used:

```
<user>-<priority>-<workstream>-<project>-<rand>-<role>-<idx>
agoswami-p0-rlscaling-tv4-v5-c1500-study-02d4c-master-0
```

`submit_job.sh` builds pod names from `JOB_NAME=${USER_ALIAS}-${PRIORITY}-${PROJECT_NAME}-job`. So:

| env var | sets | dashboard bucket |
|---|---|---|
| `PRIORITY=p0` / `p1` / `p2` | name segment | priority bucket (else → **Other**) |
| `PROJECT_NAME=<workstream>` | name segment | workstream bucket (else → **Other**) |
| `PRIORITY_CLASS_NAME=high`/`medium`/`low` | **real** k8s priorityClass | scheduling/preemption only |

⚠️ **`PRIORITY` and `PRIORITY_CLASS_NAME` are different things.** `PRIORITY=p0` only renames the pod for the dashboard; setting `PRIORITY_CLASS_NAME=p0` will fail admission (no such PriorityClass).

**Known workstream buckets (verified via the dashboard):** `phinext`, `rlscaling`, `cua`, `omniagent`, `memento`, `worldmodels`, `socialreasoning`, `activevision`, `magui`, `infinitecontext`, `racing`. If the project doesn't fit, **ask the team / dashboard admin first** before guessing — unknown names still land in **Other**.

Recommended invocation:
```bash
PRIORITY=p1 \
PROJECT_NAME=<workstream-from-list-above> \
PRIORITY_CLASS_NAME=medium \
bash /data/<alias>/<repo>/docker/submit_real_train_batch.sh
```

(Use `p0 + high` only if you've been told you have that quota.)

## Step 3 — Submit (GPU training)

The typical training launcher (e.g. `submit_real_train_batch.sh`) wraps `submit_job.sh` like this. Key flags:

```bash
bash submit_job.sh \
  --upload <local-dir-1> [<local-dir-2> ...]   # tar each, cp to PVC at /mnt/pvc/<alias>/runs/<JOB>/<dir>
  --image aifrontiers.azurecr.io/<...>:tag \
  --acr                                         # only if pulling from aifrontiers.azurecr.io
  --node 1 --gpu-per-node 8 \
  --cpu 64 --memory 512Gi --shm 64Gi \
  --secret-volume <secret>:/run/secrets/<path>  # mount k8s Secret (e.g., creds)
  --follow-logs                                  # tail master pod stdout after start
  --cmd "<bash one-liner — runs inside container>"
```

Inside the `--cmd`, these env vars are auto-injected by `submit_job.sh`:

```
JOB_NAME           = <USER_ALIAS>-<PRIORITY>-<PROJECT_NAME>-job-<rand>
USER_ALIAS         = your alias
PVC_MOUNT          = /mnt/pvc
DATA_ROOT          = /mnt/pvc/<alias>
OUTPUT_DIR         = /mnt/pvc/<alias>/outputs/<JOB_NAME>     ← persistent ckpt dir
NPROC_PER_NODE     = <gpu-per-node>
MASTER_ADDR/PORT, WORLD_SIZE, RANK     ← multi-node only
WANDB_API_KEY, WANDB_BASE_URL, WANDB_HOST, WANDB_PROJECT, WANDB_NAME
```

Path to uploaded code inside container: `/mnt/pvc/<alias>/runs/<JOB_NAME>/<basename-of-uploaded-dir>`.

## Step 3b — Submit (CPU only)

Same script, no `--gpu-per-node`. For interactive PVC browsing / quick scripts:

```bash
# Interactive shell with PVC mounted, no GPU, auto-cleans on exit
bash submit_job.sh --interactive --cpu 4 --memory 16Gi
```

For batch CPU work:
```bash
bash submit_job.sh \
  --upload <dir> \
  --cpu 16 --memory 64Gi \
  --image nvcr.io/nvidia/pytorch:25.08-py3 \
  --cmd "python preprocess.py"
```

⚠️ Don't set `PRIORITY=p0/p1` for CPU work — those are GPU quota buckets. Use `PRIORITY=p3` (or anything that doesn't match a GPU bucket) and `PRIORITY_CLASS_NAME=low`.

## Step 4 — WandB

Already wired. `submit_job.sh` injects `WANDB_*` env vars. **Do not** re-export `WANDB_MODE=offline` inside `--cmd` — that defeats it. Default project is `phitrain`; override per-run with `WANDB_PROJECT=<name>` prefix to `bash submit_real_train_batch.sh`.

Run name = `WANDB_NAME` = `JOB_FULLNAME` (so it matches the pod name in the GPU dashboard). View at `https://microsoft-research.wandb.io/<entity>/<project>`.

## Step 5 — Save checkpoints

Your training code's save_path **must point to `$OUTPUT_DIR`** (= `/mnt/pvc/<alias>/outputs/<JOB_NAME>/`). For SkyRL, that's in the hydra config (typically `trainer.save_path` or `output_dir`).

Anything written under `$OUTPUT_DIR` persists on Vast PVC forever (until manually deleted). Pod death, job deletion, force-delete — none affect it.

**Size budget**: PVC is 550 TiB but **shared across the whole namespace and currently ~91% full**. A 9B FSDP training ckpt (bf16 model + fp32 Adam) is ~120 GB; a 4B one is ~50 GB. Set `save_total_limit` (or equivalent) so you don't accumulate 20 of them.

Anything written to other paths (container `/tmp`, container cwd, `./wandb` when offline) **is lost** when the pod terminates.

## Step 6 — Copy ckpt back to the gcrsandbox dev box

**The Vast PVC is NOT mounted on the dev box.** `/data/<alias>/...` on `gcrsandbox521` is a local ext4 RAID, completely separate from `pvc-vast-bonete61`. Three ways to get a checkpoint off:

### A — One-shot CPU pod + `kubectl cp` (good for 1–10 GB files)

```bash
JOB="<alias>-p3-cpcopy-$RANDOM"
NS=bonete61
cat <<YAML | kubectl create -n $NS -f -
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: $JOB
spec:
  queue: $NS
  minAvailable: 1
  policies: [{event: TaskCompleted, action: CompleteJob}]
  tasks:
    - name: master
      replicas: 1
      template:
        spec:
          schedulerName: volcano
          priorityClassName: low
          serviceAccountName: aif-bonete-uami
          restartPolicy: Never
          volumes:
            - name: data
              persistentVolumeClaim: {claimName: pvc-vast-bonete61}
          containers:
            - name: master
              image: nvcr.io/nvidia/pytorch:25.08-py3
              command: ["sh","-c","sleep 3600"]    # keep alive 1h to kubectl cp out
              volumeMounts: [{name: data, mountPath: /mnt/pvc}]
              resources: {requests: {cpu: "1", memory: 2Gi}, limits: {cpu: "1", memory: 2Gi}}
YAML

# Wait for ready
POD=$(kubectl -n $NS get pods -l volcano.sh/job-name=$JOB -o jsonpath='{.items[0].metadata.name}')
kubectl -n $NS wait --for=condition=Ready pod/$POD --timeout=180s

# Copy
kubectl -n $NS cp $POD:/mnt/pvc/<alias>/outputs/<JOB_FULLNAME>/checkpoint-100.pt ./checkpoint-100.pt

# Cleanup
kubectl -n $NS delete job.batch.volcano.sh/$JOB --wait=false
```

`kubectl cp` is single-connection SPDY exec — ~10–30 MB/s typical. For >50 GB files this is painfully slow.

### B — `--interactive` browse + transfer (most ergonomic for browsing)

```bash
bash submit_job.sh --interactive --cpu 4 --memory 16Gi
# Inside the pod: ls /mnt/pvc/<alias>/outputs/<JOB>/
# In another terminal: kubectl -n bonete61 cp <interactive-pod>:/mnt/pvc/... ./
# Exit shell when done — job auto-deletes
```

### C — Push to Azure Blob from inside the training pod (fastest for big ckpt)

The runner pod has `aif-bonete-uami` workload identity which can `az login --service-principal` and write to `aifrontiers` storage account. Append to the `--cmd`:

```bash
echo '[run] === uploading ckpt to blob ==='
az login --service-principal \
    --tenant "$AZURE_TENANT_ID" \
    --username "$AZURE_CLIENT_ID" \
    --federated-token "$(cat "$AZURE_FEDERATED_TOKEN_FILE")" \
    --allow-no-subscriptions --output none
az storage copy --recursive \
    -s "$OUTPUT_DIR" \
    -d "https://aifrontiers.blob.core.windows.net/data/bonete/ckpts/$JOB_NAME/" \
    --auth-mode login
```

Then from the dev box, `azcopy` (multi-connection) pulls it down at line-rate.

## Step 7 — Monitor and debug

```bash
# Where is my job?
kubectl -n bonete61 get jobs.batch.volcano.sh -l submitter=<alias>
kubectl -n bonete61 get pods -l submitter=<alias>

# Live logs (after script's --follow-logs detached, or to reattach)
kubectl -n bonete61 logs -f <pod-name>

# Why won't my pod start? (image pull / scheduling failures)
kubectl -n bonete61 describe pod <pod-name> | sed -n '/Events:/,$p'

# Force-kill a stuck job
kubectl -n bonete61 delete job.batch.volcano.sh/<job-name> --wait=false
kubectl -n bonete61 delete pods -l volcano.sh/job-name=<job-name> --grace-period=0 --force
```

Image pull (~19 GB custom image like `aifrontiers.azurecr.io/<alias>/echo-rl:latest`) is the biggest startup delay: **3–4 min cold pull**, **~10 s if cached on the node**. Repeated submissions tend to land on the same node and hit the cache.

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| Pod stuck in `Pending` for >10 min, event says `no PriorityClass with name p0` | Set `PRIORITY_CLASS_NAME=p0` | Use `high`/`medium`/`low`; `p0` is a name label only |
| Job in dashboard "Other" bucket despite using 8 GPUs | Pod name lacks `p<N>` and/or workstream segment | Set both `PRIORITY=p1` AND `PROJECT_NAME=<workstream>` |
| `kubectl cp` step takes minutes for a small repo | `.gitignore` patterns use trailing `/`, tar didn't skip them | Bare names: `outputs` not `outputs/` |
| Training "succeeded" but `/mnt/pvc/<alias>/outputs/<JOB>/` only has `skyrl_logs/` | Trainer's `save_path` not pointing at `$OUTPUT_DIR`, or `timeout` killed it before first `save_step` | Verify config; bump timeout |
| `/data/<alias>/outputs/...` empty on dev box | PVC isn't mounted on dev box | Use Step 6 to copy out |
| WandB run is offline / no live curves | `WANDB_MODE=offline` set somewhere in `--cmd` | Remove it; envs from `submit_job.sh` will sync to wandb |
| `az login` inside pod errors `AZURE_FEDERATED_TOKEN_FILE not set` | Pod missing `azure.workload.identity/use: "true"` label or wrong SA | `submit_job.sh` sets both by default; check if you've overridden `--service-account` |

## When NOT to use this skill

- Submitting to a different cluster (b200 / cordillera / singularity / AML compute) — those have their own wrappers (`/data/yadonglu/agento/volcano/vsubmit.sh` for b200, `az ml job create` for AML).
- AML experiment_name / display_name registration — bonete61 is **not** attached to any AML workspace; AML portal will never see these jobs. Use WandB instead.

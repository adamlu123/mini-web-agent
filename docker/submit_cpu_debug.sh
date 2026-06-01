#!/usr/bin/env bash
# Launch a long-lived CPU-only DEBUG pod on bonete61 -- no GPU, just a place to
# browse the PVC (training logs, outputs, ckpts) and run quick scripts. The pod
# runs `sleep infinity` and is submitted DETACHED (no blocking TTY).
#
# CPU work must NOT use the p0/p1 GPU-quota buckets, so this defaults to
# PRIORITY=p3 + PRIORITY_CLASS_NAME=low (see the bonete-submit skill).
#
# submit_job.sh requires --upload even for an idle pod (unless --interactive,
# which blocks on a TTY), so we upload a tiny throwaway placeholder dir.
#
#   bash docker/submit_cpu_debug.sh
#   CPU=16 MEM=64Gi bash docker/submit_cpu_debug.sh   # override resources
#
# After it returns, exec into the pod. Per the WAF note, use an EMPTY
# interactive shell -- don't pass big `-c` payloads (they get a 403):
#   kubectl -n bonete61 exec -it <pod> -- bash
#
# Tear down when done:
#   kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false

set -euo pipefail

SUBMIT="${SUBMIT:-/data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.08-py3}"
CPU="${CPU:-8}"
MEM="${MEM:-32Gi}"
NS="${NS:-bonete61}"

[[ -x "$SUBMIT" || -f "$SUBMIT" ]] || { echo "[error] submit_job.sh not found at $SUBMIT"; exit 1; }

export PATH="$HOME/.krew/bin:$PATH"

# CPU work: stay out of the GPU-quota dashboard buckets.
export PRIORITY="${PRIORITY:-p3}"
export PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-low}"
export PROJECT_NAME="${PROJECT_NAME:-debug}"

# Tiny throwaway dir just to satisfy submit_job.sh's --upload requirement.
PLACEHOLDER="$(mktemp -d /tmp/cpu_debug_placeholder_XXXX)"
echo "cpu debug pod placeholder" > "$PLACEHOLDER/README.txt"
trap 'rm -rf "$PLACEHOLDER"' EXIT

echo "[submit_cpu_debug] CPU=$CPU MEM=$MEM IMAGE=$IMAGE PRIORITY=$PRIORITY (no GPU)"

bash "$SUBMIT" \
    --upload "$PLACEHOLDER" \
    --image "$IMAGE" \
    --cpu "$CPU" --memory "$MEM" \
    --cmd "sleep infinity"

echo
echo "[submit_cpu_debug] submitted. Find the pod with:"
echo "    kubectl -n $NS get pods -l volcano.sh/job-name=<JOB_FQN-printed-above>"
echo "Then exec in (empty shell -- WAF-safe):"
echo "    kubectl -n $NS exec -it <pod> -- bash"

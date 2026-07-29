#!/bin/bash
# 原版 OM2W WebJudge(persistent-cli steps 版)集群 runner。
# 必需 env: TRAJ_DIR OUT_DIR;可选 JUDGE_MODEL NUM_WORKER EXPECTED_TASKS。
# 约定: 上传树在 $DATA_ROOT/runs/$JOB_NAME/mini-web-agent;webchain secret 已挂载;
#       vendored WebJudge 从 /home/luyadong/sandbox/mini-web-agent/om2w_judge 加载(软链)。
set -euo pipefail
: "${TRAJ_DIR:?}" "${OUT_DIR:?}"

REPO="$DATA_ROOT/runs/$JOB_NAME/mini-web-agent"
mkdir -p /home/luyadong/sandbox
ln -sfn "$REPO" /home/luyadong/sandbox/mini-web-agent
. /run/secrets/webchain-sampling/cred.sh

cd "$REPO"
pip install --quiet backoff 2>/dev/null || pip install backoff
python scripts/eval_persistent_cli_steps_with_original_om2w.py \
  --trajectories_dir "$TRAJ_DIR" \
  --output_path "$OUT_DIR" \
  --model "${JUDGE_MODEL:-o4-mini}" \
  --score_threshold 3 \
  --num_worker "${NUM_WORKER:-32}" \
  --expected_tasks "${EXPECTED_TASKS:-300}"
echo PCLI_JUDGE_DONE

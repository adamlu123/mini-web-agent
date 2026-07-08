# 被 submit_sft_q35_image.sh / submit_sft_eval_q35_image.sh source。
# 堵死 wandb 401 打挂 job 的坑(af8b0、luyadong adf40 两次教训):
# submit_job.sh 在 WANDB_API_KEY 为空时会 fallback 到微软内部实例的 local- key,
# 与我们强制的 WANDB_HOST=api.wandb.ai 不配对,job 会在数据预处理+权重加载全部
# 走完之后才在 trainer 初始化处 401 炸掉,白烧几分钟 GPU。
#
# 行为:未设置 key 时自动从共享 key 文件加载(run 记入 flyhero99 账号,
# run 名带提交者 alias 可区分);加载不到则拒绝提交并给出解法。

WANDB_SHARED_KEY_FILE="${WANDB_SHARED_KEY_FILE:-/data/t-yifeili/.secrets/wandb_api_key}"

if [[ -z "${WANDB_API_KEY:-}" && -r "$WANDB_SHARED_KEY_FILE" ]]; then
    export WANDB_API_KEY="$(cat "$WANDB_SHARED_KEY_FILE")"
    echo "[wandb-preflight] WANDB_API_KEY 未设置,已自动加载共享 key: $WANDB_SHARED_KEY_FILE"
    echo "[wandb-preflight] run 将记入 flyhero99 的 W&B 账号(run 名带你的 alias,可区分)"
fi

if [[ -z "${WANDB_API_KEY:-}" || "${WANDB_API_KEY}" == local-* ]]; then
    echo "[error] WANDB_API_KEY 未设置或是内部实例的 local- fallback key,与 WANDB_HOST=${WANDB_HOST:-https://api.wandb.ai} 不配对。" >&2
    echo "        直接提交会在 trainer 初始化后 wandb 401 打挂整个 job。提交前任选其一:" >&2
    echo "        A. export WANDB_API_KEY=<你自己的key>    # https://wandb.ai/authorize 获取" >&2
    echo "        B. export WANDB_API_KEY=\$(cat /data/t-yifeili/.secrets/wandb_api_key)   # 用共享 key" >&2
    exit 1
fi

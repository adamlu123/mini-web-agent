#!/usr/bin/env bash
# 生成不含训练数据的轻量代码树,用于集群提交时的 --upload。
# 数据走 PVC 固定路径(见 docker/upload_data_to_pvc.sh),代码树里不带
# LlamaFactory/data/web_agent_*(raw json + bundles,总计几十 GB)。
#
# 用法:
#   bash docker/make_light_code_tree.sh          # 输出树路径到 stdout 最后一行
#   MINI_WEB_AGENT_DIR=$(bash docker/make_light_code_tree.sh | tail -1) \
#     bash docker/submit_sft_q35_image.sh        # 与现有 submit 脚本配合
#
# basename 保持 mini-web-agent 不变(pod 内 --cmd 路径依赖它)。
set -euo pipefail

SRC="${SRC:-/data/t-yifeili/mini-web-agent}"
LIGHT_ROOT="${LIGHT_ROOT:-/tmp/mwa-light}"
LIGHT="$LIGHT_ROOT/$(basename "$SRC")"

mkdir -p "$LIGHT"
# --filter=':- .gitignore' 逐目录遵循 .gitignore(等价旧流程 tar 的
# --exclude-vcs-ignores,排掉 eval_outputs/logs/ckpt 等本地产物);
# 再显式排掉数据(bundles 未被 gitignore)和 .git 本身。
rsync -a --delete --delete-excluded \
  --filter=':- .gitignore' \
  --exclude '.git' \
  --exclude 'LlamaFactory/data/web_agent_*' \
  --exclude 'LlamaFactory/saves' \
  "$SRC/" "$LIGHT/"

SIZE=$(du -sh "$LIGHT" | cut -f1)
echo "[info] 轻量代码树: $LIGHT ($SIZE)" >&2
if [[ $(du -sm "$LIGHT" | cut -f1) -gt 2048 ]]; then
  echo "[warn] 轻量树 >2GB,可能有数据没被排除,检查 du -sh $LIGHT/* 排查" >&2
fi
echo "$LIGHT"

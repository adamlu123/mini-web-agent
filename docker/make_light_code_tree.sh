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

# 默认取脚本所在仓库,任何 user 的 checkout 都能直接用
SRC="${SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 按用户隔离,同一台 dev box 多人并发提交互不覆盖
LIGHT_ROOT="${LIGHT_ROOT:-/tmp/mwa-light-$(whoami | cut -d@ -f1)}"
LIGHT="$LIGHT_ROOT/$(basename "$SRC")"

mkdir -p "$LIGHT"
# 显式排除清单。不能用 --filter=':- .gitignore':根 .gitignore 有
# `LlamaFactory/**` + `!...` 反选组合,rsync 不支持 gitignore 的 `!` 语法,
# 会把整个 LlamaFactory 源码排掉(pod 里 pip install -e 找不到 setup.py)。
rsync -a --delete --delete-excluded \
  --exclude '.git' \
  --exclude 'eval_outputs' \
  --exclude 'logs' \
  --exclude 'outputs' \
  --exclude 'wandb' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'LlamaFactory/data/web_agent_*' \
  --exclude 'LlamaFactory/saves' \
  --exclude 'LlamaFactory/output' \
  "$SRC/" "$LIGHT/"

# 删掉树里的 .gitignore:submit_job.sh 打 tar 用 --exclude-vcs-ignores,
# 会重新应用根 .gitignore 的 `LlamaFactory/**`,把源码从 tar 里剔掉
# (rsync 阶段留下的文件到 tar 阶段又丢了)。排除逻辑全部由上面的
# rsync 显式清单承担,.gitignore 在上传树里没有别的用途。
find "$LIGHT" -name .gitignore -delete

# 护栏:pod 内 bootstrap 要 pip install -e LlamaFactory,核心文件必须在,
# 且必须真的能进 tar(复现 submit_job.sh 的打包参数)
for f in LlamaFactory/pyproject.toml LlamaFactory/src docker/run_sft_q35_image.sh; do
  [[ -e "$LIGHT/$f" ]] || { echo "[error] 轻量树缺 $f,排除规则有误" >&2; exit 1; }
done
# 注意不能用 grep -q:提前退出会让 tar 收 SIGPIPE,在 pipefail 下误报失败
n=$(tar --exclude-vcs-ignores --exclude-vcs -cf - -C "$LIGHT" . 2>/dev/null \
    | tar -tf - | grep -c 'LlamaFactory/pyproject.toml' || true)
if [[ "$n" -eq 0 ]]; then
  echo "[error] tar --exclude-vcs-ignores 会丢 LlamaFactory/pyproject.toml,上传树仍有残留 ignore 规则" >&2
  exit 1
fi

SIZE=$(du -sh "$LIGHT" | cut -f1)
echo "[info] 轻量代码树: $LIGHT ($SIZE)" >&2
if [[ $(du -sm "$LIGHT" | cut -f1) -gt 2048 ]]; then
  echo "[warn] 轻量树 >2GB,可能有数据没被排除,检查 du -sh $LIGHT/* 排查" >&2
fi
echo "$LIGHT"

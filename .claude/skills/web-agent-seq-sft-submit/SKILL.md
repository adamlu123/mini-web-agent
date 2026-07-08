---
name: web-agent-seq-sft-submit
description: >-
  Submit a multi-node Qwen3.5-9B SFT run on a sequential-compact-tools web-agent
  dataset (e.g. web_agent_seq_om2w4000_run1) to the bonete61 B200 cluster. Use when
  the user wants to: train on a seq/om2w4000-style ShareGPT dataset, package a
  portable image bundle, pick the upload-vs-PVC data strategy, launch the 4-node
  submit_job.sh command, monitor the run, or serve/eval the resulting checkpoint
  with correct <think> alignment. Also covers GUEST SUBMISSION: another user
  submitting from yifeili's sandbox (/data/t-yifeili) with their own
  USER_ALIAS / quota / dashboard / W&B attribution. Complements
  web-agent-sft-cluster (single-node train→eval pipeline + eval gotchas) and
  docs/qwen3_5_think_alignment.md.
---

# Seq-SFT 训练提交 recipe（以 om2w4000_run1 为例）

把一份 sequential-compact-tools ShareGPT 数据集训成 Qwen3.5-9B full-SFT ckpt，
4 节点 B200 提交。上一次成功照此跑通的 run 记录在
`docs/web_agent_seq_label1_4node_sft_20260630.md`（job `7fc3a`）。

所有命令默认在 dev box 的 `/data/t-yifeili/mini-web-agent` 下执行。

## TL;DR

数据与代码分离：bundle 一次性上传到 PVC **固定数据路径**
`/mnt/pvc/experiments/t-yifeili/data/<bundle名>`，训练配置的
`dataset_dir`/`media_dir` 写这个绝对路径；提交时只上传**不含数据的轻量代码树**。

```bash
cd /data/t-yifeili/mini-web-agent

# 1. 打包 portable bundle（图片拷进 bundle、路径改相对；必须在图片所在的机器上跑）
python LlamaFactory/scripts/package_web_agent_images.py \
  --input LlamaFactory/data/web_agent_seq_om2w4000_run1.json \
  --out-dir LlamaFactory/data/web_agent_seq_om2w4000_run1_portable_bundle \
  --json-name web_agent_seq_om2w4000_run1_portable.json \
  --dataset-name web_agent_seq_om2w4000_run1_portable
# 确认输出 manifest 里 missing_images: 0

# 2. bundle 上传到 PVC 固定数据路径（每份数据只需一次；分块重试+校验+原子落位）
bash docker/upload_data_to_pvc.sh \
  LlamaFactory/data/web_agent_seq_om2w4000_run1_portable_bundle
# 成功后数据在 /mnt/pvc/experiments/t-yifeili/data/web_agent_seq_om2w4000_run1_portable_bundle

# 3. 配置文件的 dataset_dir/media_dir 指向上面的固定路径（绝对路径）
#    LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml

# 4. 提交。NODES 任意指定（1=单节点 smoke，4=正式 4 节点）；
#    LIGHT_UPLOAD=1 默认开启：自动 rsync 出 ~150MB 的轻量代码树再上传
#    （docker/make_light_code_tree.sh，遵循 .gitignore + 排除 LlamaFactory/data/web_agent_*）
NODES=4 \
CONFIG=examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml \
WANDB_PROJECT=web-agent-sft AZBLOB_AUTO_PUSH=0 \
bash docker/submit_sft_q35_image.sh
```

`submit_sft_q35_image.sh` 的其他可覆盖项：`GPUS`(默认8)、`PRIORITY`(p0)、
`PROJECT_NAME`(cua)、`PRIORITY_CLASS_NAME`(high)、`LIGHT_UPLOAD=0` 回退整树上传。
全局 batch = NODES×GPUS×1，改 NODES 时注意 steps/epoch 随之变化
（7774 样本 / (NODES×8) ≈ steps/epoch）。

`--follow-logs` 在 pod 还在跑时就会 detach（打印 `Pod final phase: Running` 后退出 0），
**不是失败**，用下面 Monitoring 的 kubectl 命令跟。

## 环境准备（一次性）

- dev box 上要有：`/data/t-yifeili/mini-web-agent`（本仓库）、
  `/data/t-yifeili/aifsdk`（提交驱动 `clusters/lambda/submission/submit_job.sh`）。
- `kubectl` + krew：`export PATH="$HOME/.krew/bin:$PATH"`；namespace `bonete61`。
- Secret volume `echo-rl-creds` 已在集群配好（HF token 等），挂到
  `/run/secrets/echo-rl-creds`。
- W&B：**必须 `WANDB_HOST=https://api.wandb.ai`（个人 endpoint）**。默认的
  Microsoft endpoint 会 401 并在 trainer 初始化后把整个 job 打挂（label1 run 的
  `af8b0` 就是这么死的）。项目 `web-agent-sft`，历史 run 在 `flyhero99/web-agent-sft`。
- **数据不在 git 里**（2026-07-07 起 `LlamaFactory/data/*.json` 已 gitignore、
  LFS 追踪已移除）。数据的权威副本：dev box 本地 `LlamaFactory/data/` +
  PVC 固定路径 `/mnt/pvc/experiments/t-yifeili/data/`。历史 LFS 版本仍可从
  commit `9da2b6c` 恢复（`git show 9da2b6c:LlamaFactory/data/... | git-lfs smudge`）。

## Step 1 — 数据

**用哪份数据**：raw ShareGPT json 只存 dev box 本地（不进 git）：

```text
LlamaFactory/data/web_agent_seq_om2w4000_run1.json   (459 MB, 仅本地)
  7774 examples = 5222 trajectory_session + 552 image_qa
                + 1000 self_reflection_image + 1000 self_reflection_final
  46060 gpt turns（其中 2552 个 judge 类 turn 无 <think>，见下面"think 对齐"）
```

生成脚本（从 rollout 输出重新生成时用）：
`LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py`，
源轨迹在 `/data/t-yifeili/om2w_4000_sampling/outputs/om2w_4000_run1/`。

**必须打包 portable bundle**：raw json 里 `images` 是 dev box 的绝对路径
（`/data/t-yifeili/om2w_4000_sampling/...`），pod 里不存在。
`package_web_agent_images.py`（命令见 TL;DR）把 6322 张图拷进
`<bundle>/images/` 并把路径改成相对；bundle 里自动生成 `dataset_info.json`
（**不需要**手动注册到 `LlamaFactory/data/dataset_info.json`——config 的
`dataset_dir` 直接指 bundle 目录）。打包必须在图片所在机器上跑；
manifest 里 `missing_images` 必须为 0。

**think 对齐（重要，详见 `docs/qwen3_5_think_alignment.md`）**：

- 2552 个无 `<think>` 的 judge turn 会被 LlamaFactory `ReasoningTemplate`
  注入空 `<think>\n\n</think>\n\n` 并计 loss——预期行为，不用改数据。
  以后新生成数据时建议在生成脚本里显式拼上这个前缀（字节精确）。
- 训练格式 `template: qwen3_5` + `mask_history: false` 会把历史轮 `<think>`
  留在 prompt 里 ⇒ **训出的 ckpt serving 时必须挂
  `configs/qwen3_5_train_aligned.jinja`**（见"训练之后"）。

## Step 2 — 训练配置

现成配置：`LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml`。
新数据集复制它改三处：`dataset_dir`/`media_dir`/`dataset`（指向新 bundle）和
`output_dir`/`run_name`。不可动的关键项（踩过坑）：

- `deepspeed: examples/deepspeed/ds_z2_config.json` — **9B 必须 ZeRO-2**；
  z3 会从随机初始化开始训（step-1 loss ≈ 12.4）。健康的 step-1 loss < 1。
- `enable_liger_kernel: false` — liger 在该 image 上有问题（有 noliger smoke 佐证）。
- `template: qwen3_5`、`mask_history: false`、`enable_thinking: true`、
  `cutoff_len: 40000`、`image_max_pixels: 262144`（serving 端对齐要用同值）。
- 4 节点 x 8 GPU x batch 1 = 全局 batch 32；7774 样本 ≈ 243 steps/epoch。

## Step 3 — 数据上传到 PVC 固定路径（默认模式，2026-07-07 起）

**约定**：所有训练数据 bundle 都放 PVC 固定路径
`/mnt/pvc/experiments/t-yifeili/data/<bundle名>`，训练 yaml 的
`dataset_dir`/`media_dir` 写这个**绝对路径**；提交 job 时只上传轻量代码树
（TL;DR Step 4 的 rsync 排除法）。数据与 job 生命周期解耦——不再依赖
`/mnt/pvc/t-yifeili/runs/<旧JOB_NAME>/...` 这类会随旧 job 清理而失效的路径。

```bash
# 每份新数据一次；重跑会原子覆盖同名目录
bash docker/upload_data_to_pvc.sh LlamaFactory/data/<bundle目录> [目标名]
```

脚本机制（`docker/upload_data_to_pvc.sh`）：起 1c/2Gi 的 uploader pod 挂 PVC →
tar 按 512M 分块 `kubectl cp`（每块独立重试 5 次，断连只重传当前块，解决了
label1 `b47dc` 整包 connection reset 的问题）→ pod 内解压到 `<dest>.tmp`
后原子 `mv` → 文件数+字节数校验 → 自动删 uploader job。
**不要**用 `cp -a` 直接往 PVC 拷数据（权限报错，`013704` 的教训）。

查看已有数据：
```bash
export PATH="$HOME/.krew/bin:$PATH"
kubectl -n bonete61 exec <任一挂PVC的pod> -- ls -lh /mnt/pvc/experiments/t-yifeili/data/
```

**旧路径迁移**：早期 yaml 里指向 `/mnt/pvc/t-yifeili/runs/<旧JOB_NAME>/...` 的
数据，如还要复用，在挂 PVC 的 pod 里 `cp -r` 到固定路径下再改 yaml（pod 内
拷贝没有 dev box 的权限问题）。

## Guest 提交 — 别人从 yifeili 的 sandbox 提交，归属记在自己名下

适用场景：另一个用户直接登录 yifeili 的 sandbox、`cd /data/t-yifeili/mini-web-agent`
提交 job（共享工作树、数据、aifsdk、kubectl 凭证都用现成的——这些已被接受），
但 **job 命名 / quota bucket / GPU dashboard / submitter label / W&B run 必须
记在 guest 自己名下**。做法：提交时覆盖归属相关的环境变量。

### 第一步（每个新 guest 一次性）：验证 USER_ALIAS 可覆盖

`submit_job.sh` 用 `USER_ALIAS` 拼 job 名（`<alias>-<PRIORITY>-<PROJECT_NAME>-job-<rand>`）、
`submitter=` label、PVC 上传路径 `/mnt/pvc/<alias>/runs/<JOB_NAME>/`。先确认它接受
环境变量覆盖而不是硬编码 `whoami`：

```bash
grep -n "USER_ALIAS" /data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh | head
```

- 形如 `USER_ALIAS="${USER_ALIAS:-$(whoami)}"` → 直接 export 覆盖即可（走下面的命令）。
- 若是硬编码 `$(whoami)` → 不要改 aifsdk 源码，写个 wrapper 或让 guest 先
  `USER_ALIAS=<guest> bash -c '...'` 验证 job 名后再提正式任务。

### Guest 提交命令

与 TL;DR 的命令相同，只把归属相关的 env 换成 guest 的：

```bash
cd /data/t-yifeili/mini-web-agent

USER_ALIAS=<guest-alias> \
NODES=4 \
CONFIG=examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml \
WANDB_PROJECT=web-agent-sft AZBLOB_AUTO_PUSH=0 \
PRIORITY=p1 PROJECT_NAME=<guest 的 workstream> PRIORITY_CLASS_NAME=medium \
bash docker/submit_sft_q35_image.sh
```

数据同样读固定路径 `/mnt/pvc/experiments/t-yifeili/data/...`（跨用户可读，
guest 不用重新上传数据）；`LIGHT_UPLOAD=1` 默认生效。

覆盖 `USER_ALIAS` 后自动跟着走 guest 名下的东西（`--cmd` 里全是 `$USER_ALIAS`/`$JOB_NAME`
变量，不用改）：

| 项 | 变成 |
|---|---|
| job/pod 名、dashboard bucket、quota | `<guest-alias>-p1-<workstream>-job-...` |
| `submitter=` label（`kubectl get jobs -l submitter=<guest>`） | guest |
| 代码上传路径 | `/mnt/pvc/<guest-alias>/runs/<JOB_NAME>/mini-web-agent` |
| 训后 ckpt PVC 同步路径 | `/mnt/pvc/<guest-alias>/models/<output_dir>` |
| W&B run 名 | 带 guest alias 的 `JOB_NAME`（账号仍是 yifeili 的，见下） |

### Guest 提交的注意事项

1. **W&B 沿用 yifeili 的账号/项目**（`flyhero99/web-agent-sft`，key 来自 sandbox
   环境，不用另配）。run 名 = `JOB_NAME` 已带 guest alias，在同一 project 里
   天然可区分。guest **不要** export 自己的 `WANDB_API_KEY`——key 与 HOST 不匹配
   会在 trainer 初始化后 401 打挂 job（`af8b0` 教训）。第一次先 `--node 1`
   smoke 一把再上 4 节点。
2. **训练 pod 的运行环境与 sandbox 无关，谁提交都一样**。python/torch 栈分三层：
   ①容器镜像（`--image` 那个 `nvidia25.11-pytorch2.10.0-...` 镜像，烧好
   python + torch 2.10 / transformers / deepspeed / accelerate）；
   ②in-pod bootstrap（`docker/run_sft_q35_image.sh` 开头：
   `pip install --no-deps -e LlamaFactory` + trl==0.24.0 + peft 等，装的是
   **上传快照里的 LlamaFactory 源码**）；③上传的代码快照本身。
   sandbox 的 `/data/t-yifeili/miniconda3` 只用于本地脚本/eval，训练 pod 不碰它。
   所以 guest "用 yifeili 的环境"仅指提交时的本地工具（aifsdk、kubectl 凭证）
   和代码快照来源。
3. **`PROJECT_NAME` 用 guest 自己的 workstream**（见 bonete-submit skill 的已知
   bucket 列表），否则 dashboard 进 "Other"。`PRIORITY`/`PRIORITY_CLASS_NAME`
   按 guest 自己的配额来，别默认 `p0/high`。
4. **PVC 引用路径不用改**：pvcdata 类配置里指向 `/mnt/pvc/t-yifeili/...` 的
   dataset 路径跨用户可读，guest 的 pod 读它没问题；只有**写入**（上传、ckpt sync）
   走 guest 目录。
5. **共享工作树纪律**：提交前 `git status` 确认树的状态是双方预期的（`--upload`
   打包的是当前实时状态）；guest 改配置用**新文件名**（复制 yaml 改名），不要在
   原 yaml 上就地改；`output_dir`/`run_name` 用 guest 专属名字。
6. **k8s 层的审计身份仍是 yifeili**（kubectl 凭证没换）——label/命名归 guest，
   集群审计日志归 yifeili，这点双方知情即可。
7. 收尾：guest 的 ckpt 在 `/mnt/pvc/<guest-alias>/models/...`，拷回 dev box 的
   方法见 bonete-submit skill Step 6。

## Step 4 — 监控

```bash
export PATH="$HOME/.krew/bin:$PATH"
NS=bonete61
POD=<JOB_NAME>-master-0     # JOB_NAME 来自 submit 输出 "Created job:"
kubectl -n $NS logs -f --tail=50 $POD | grep -E --line-buffered \
  "Num examples|Num Epochs|Total train batch size|Total optimization steps|loss|train_loss|wandb|SFT exited rc=|OutOfMemory|CUDA out of memory|ChildFailedError|Saving model|\[sync\]|\[merge\]"
```

健康信号：`Num examples ≈ 7770`、`Total train batch size = 32`、首个 loss < 1、
W&B run 链接出现。**不要**裸 grep `error/Traceback`——训练数据本身充满这些字符串。
杀 job：`kubectl -n bonete61 delete job.batch.volcano.sh/<JOB_FQN> --wait=false`。

## 训练之后

`docker/run_sft_q35_image.sh` 训完自动：①把 HF ckpt 同步到稳定 PVC 路径
`/mnt/pvc/$USER/models/<output_dir去掉saves/>`；②**merge vision tower**
（文本 SFT 会丢 vision 键，不 merge 的 ckpt 无法作为多模态模型加载；`MERGE_VISION=0` 关闭）。

**Serving / eval 必读**：

- vLLM 起服务必须挂 train-aligned 模板，且起完先跑 MARKER 探针
  （命令见 `docs/qwen3_5_think_alignment.md` §2.2）：
  ```bash
  vllm serve "$CKPT" ... \
    --chat-template <repo>/configs/qwen3_5_train_aligned.jinja \
    --mm-processor-kwargs '{"max_pixels":262144}'
  ```
  或一劳永逸：`cp configs/qwen3_5_train_aligned.jinja $CKPT/chat_template.jinja`。
- teacher-forced 复现检查用 `scripts/sft_replay_all_cases_thinkfix.py`（自带模板探针）。
- om2w 全量/集群 eval 流程与 gotchas 见 `web-agent-sft-cluster` skill
  （SFT-ALIGNED eval config、单引擎、colocate_all=false 等）。

## 关键文件

| 路径 | 作用 |
|------|------|
| `LlamaFactory/data/web_agent_seq_om2w4000_run1.json` | raw 数据（仅 dev box 本地） |
| `docker/upload_data_to_pvc.sh` | bundle → PVC 固定数据路径（分块+校验） |
| `/mnt/pvc/experiments/t-yifeili/data/` | PVC 固定数据根目录（pod 内） |
| `LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py` | rollout → seq ShareGPT |
| `LlamaFactory/scripts/package_web_agent_images.py` | 打 portable bundle |
| `LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml` | 本次训练配置 |
| `docker/run_sft_q35_image.sh` | in-pod：train + PVC sync + vision merge |
| `docs/web_agent_seq_label1_4node_sft_20260630.md` | 上次 4-node run 全记录 |
| `docs/qwen3_5_think_alignment.md` | `<think>` 对齐手册（serving 必读） |
| `configs/qwen3_5_train_aligned.jinja` | train-aligned chat template |

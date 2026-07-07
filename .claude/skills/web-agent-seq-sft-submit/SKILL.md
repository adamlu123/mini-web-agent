---
name: web-agent-seq-sft-submit
description: >-
  Submit a multi-node Qwen3.5-9B SFT run on a sequential-compact-tools web-agent
  dataset (e.g. web_agent_seq_om2w4000_run1) to the bonete61 B200 cluster. Use when
  the user wants to: train on a seq/om2w4000-style ShareGPT dataset, package a
  portable image bundle, pick the upload-vs-PVC data strategy, launch the 4-node
  submit_job.sh command, monitor the run, or serve/eval the resulting checkpoint
  with correct <think> alignment. Complements web-agent-sft-cluster (single-node
  train→eval pipeline + eval gotchas) and docs/qwen3_5_think_alignment.md.
---

# Seq-SFT 训练提交 recipe（以 om2w4000_run1 为例）

把一份 sequential-compact-tools ShareGPT 数据集训成 Qwen3.5-9B full-SFT ckpt，
4 节点 B200 提交。上一次成功照此跑通的 run 记录在
`docs/web_agent_seq_label1_4node_sft_20260630.md`（job `7fc3a`）。

所有命令默认在 dev box 的 `/data/t-yifeili/mini-web-agent` 下执行。

## TL;DR

```bash
cd /data/t-yifeili/mini-web-agent

# 1. 打包 portable bundle（图片拷进 bundle、路径改相对；必须在图片所在的机器上跑）
python LlamaFactory/scripts/package_web_agent_images.py \
  --input LlamaFactory/data/web_agent_seq_om2w4000_run1.json \
  --out-dir LlamaFactory/data/web_agent_seq_om2w4000_run1_portable_bundle \
  --json-name web_agent_seq_om2w4000_run1_portable.json \
  --dataset-name web_agent_seq_om2w4000_run1_portable
# 确认输出 manifest 里 missing_images: 0

# 2. 配置文件（已就绪，检查 dataset_dir/output_dir 即可）
#    LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml

# 3. 提交（4 节点 x 8 B200）
WANDB_HOST=https://api.wandb.ai WANDB_PROJECT=web-agent-sft \
PRIORITY=p0 PROJECT_NAME=cua PRIORITY_CLASS_NAME=high \
bash /data/t-yifeili/aifsdk/clusters/lambda/submission/submit_job.sh \
  --upload /data/t-yifeili/mini-web-agent \
  --image aifrontiers.azurecr.io/nvidia25.11-pytorch2.10.0-te2.13-deepspeed0.18.9-fa2main-vllm0.18.0:20260415 \
  --node 4 --gpu-per-node 8 --cpu 64 --memory 512Gi --shm 64Gi \
  --secret-volume echo-rl-creds:/run/secrets/echo-rl-creds \
  --extra-env-vars 'SFT_CONFIG=examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml,NPROC=8,AZBLOB_AUTO_PUSH=0' \
  --follow-logs \
  --cmd 'exec bash $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent/docker/run_sft_q35_image.sh'
```

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
- 本地(WSL)没有 git-lfs 时，用 standalone 二进制拉 LFS 数据：
  ```bash
  # 下载 git-lfs release 解包后：
  git show origin/rl_yifei:LlamaFactory/data/web_agent_seq_om2w4000_run1.json \
    | git-lfs smudge > LlamaFactory/data/web_agent_seq_om2w4000_run1.json
  ```

## Step 1 — 数据

**用哪份数据**：raw ShareGPT json 走 git LFS 存在仓库里：

```text
LlamaFactory/data/web_agent_seq_om2w4000_run1.json   (459 MB, LFS)
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

## Step 3 — 上传策略（二选一）

1. **随代码上传 bundle**（默认，TL;DR 的 `--upload /data/t-yifeili/mini-web-agent`）：
   bundle 几个 GB，上传可能因 connection reset 失败（label1 的 `b47dc`）。失败就重试或改用方式 2。
2. **PVC 复用**（bundle 已在某次 job 上传过 PVC 时）：复制一份 `*_pvcdata.yaml`，
   把 `dataset_dir`/`media_dir` 改成绝对 PVC 路径
   `/mnt/pvc/t-yifeili/runs/<旧JOB_NAME>/mini-web-agent/LlamaFactory/data/<bundle>`，
   然后 `--upload` 一个**不含 bundle 的轻量代码树**（rsync 排除 `LlamaFactory/data`
   到 /tmp 再上传）。不要用 `cp -a` 往 PVC 拷数据（权限报错，`013704` 的教训）。

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
| `LlamaFactory/data/web_agent_seq_om2w4000_run1.json` | raw 数据（LFS） |
| `LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py` | rollout → seq ShareGPT |
| `LlamaFactory/scripts/package_web_agent_images.py` | 打 portable bundle |
| `LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml` | 本次训练配置 |
| `docker/run_sft_q35_image.sh` | in-pod：train + PVC sync + vision merge |
| `docs/web_agent_seq_label1_4node_sft_20260630.md` | 上次 4-node run 全记录 |
| `docs/qwen3_5_think_alignment.md` | `<think>` 对齐手册（serving 必读） |
| `configs/qwen3_5_train_aligned.jinja` | train-aligned chat template |

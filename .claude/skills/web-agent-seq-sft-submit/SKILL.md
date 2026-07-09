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

所有命令在 dev box 的 mini-web-agent checkout 根目录下执行(任何用户的
checkout 都可以——docker/ 下脚本的路径和身份默认值均自动推导,见"多用户"一节)。

## 运行协议(执行本 skill 前必读)

提交训练前**必须先用 AskUserQuestion 问用户**,不要自行假设:

1. **用几个 node?** 选项建议:`1`(单节点 smoke,验证数据/管线,~8 GPU)、
   `4`(正式,32 GPU,推荐)、自定义。全局 batch = NODES×8,
   steps/epoch = 样本数 / (NODES×8),提问时把这层影响说清楚。
2. **数据要不要重新打包上传?** 选项:
   - **直接用 PVC 上已有 bundle**(数据没变时;用下面"查看已有数据"的命令列出
     `/mnt/pvc/experiments/<alias>/data/` 现状供用户确认);
   - **重新打包+上传**(raw json 更新过时;走 TL;DR Step 1-2,并同步更新
     yaml 的 dataset_dir/media_dir/dataset 和 output_dir/run_name)。
3. **用 p0 还是 p1?**(`PRIORITY`)。注意坑:p0/p1 本身只进 job 名和 GPU
   dashboard 分桶,**真正的调度优先级是 `PRIORITY_CLASS_NAME`**(默认 high)——
   用户选 p1 时一并问要不要把 class 降成 medium,按用户配额习惯来。
4. **训完要不要自动跑全量 eval?** 要的话改用组合脚本
   `docker/submit_sft_eval_q35_image.sh`(见"训后自动全量 eval"一节;
   同一个 job 里 train→ckpt sync→vision merge→harness 全量评测,**所有训练
   节点并行分片**,总并发 80,断点可续)。
   不要的话用 `docker/submit_sft_q35_image.sh`(纯训练),之后可随时用
   `docker/submit_dist_eval_q35_image.sh` 单独评(见"分布式 eval 模块")。

问完再执行。杀正在跑的 job、删 PVC 上的旧数据这类动作也要先确认。

## TL;DR

数据与代码分离：bundle 一次性上传到 PVC **固定数据路径**
`/mnt/pvc/experiments/<提交者alias>/data/<bundle名>`（alias 自动取自 whoami,
如 t-yifeili），训练配置的 `dataset_dir`/`media_dir` 写这个绝对路径；
提交时只上传**不含数据的轻量代码树**。PVC 跨用户可读——别人可以直接引用
你目录下的 bundle,不用重复上传。

```bash
cd <你的 mini-web-agent checkout 根目录>

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
# 落到 /mnt/pvc/experiments/<提交者alias>/data/<bundle名>

# 3. 配置文件的 dataset_dir/media_dir 指向上面的固定路径（绝对路径）
#    LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml

# 4. 提交。NODES 按"运行协议"问到的值填；
#    LIGHT_UPLOAD=1 默认开启：自动 rsync 出 ~170MB 的轻量代码树再上传
#    （docker/make_light_code_tree.sh，显式排除数据/产物并删树内 .gitignore——
#     不能依赖 gitignore 语义,rsync 和 tar 都吃不了 `!` 反选,踩过两层坑）
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

- dev box 上要有:本仓库的 checkout(任意路径,脚本按自身位置推导仓库根)和
  aifsdk(提交驱动;默认用 `/data/t-yifeili/aifsdk`,可读即可,或
  `SUBMIT=<自己的>/clusters/lambda/submission/submit_job.sh` 覆盖)。
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
`/mnt/pvc/experiments/<提交者alias>/data/<bundle名>`（alias 自动取自 whoami,
可用 `USER_ALIAS`/`DEST_ROOT` 覆盖），训练 yaml 的 `dataset_dir`/`media_dir`
写这个**绝对路径**；提交 job 时只上传轻量代码树（TL;DR Step 4 的 rsync 排除法）。
数据与 job 生命周期解耦——不再依赖 `/mnt/pvc/<alias>/runs/<旧JOB_NAME>/...`
这类会随旧 job 清理而失效的路径。现有 om2w4000 系列 bundle 都在
`/mnt/pvc/experiments/t-yifeili/data/` 下,跨用户可读,别人直接引用即可。

```bash
# 每份新数据一次；重跑会原子覆盖同名目录
bash docker/upload_data_to_pvc.sh LlamaFactory/data/<bundle目录> [目标名]
```

脚本机制（`docker/upload_data_to_pvc.sh`）：起 1c/2Gi 的 uploader pod 挂 PVC →
tar 按 512M 分块 `kubectl cp`（每块独立重试 5 次，断连只重传当前块，解决了
label1 `b47dc` 整包 connection reset 的问题）→ pod 内解压到 `<dest>.tmp`
后原子 `mv` → 文件数+字节数校验 → 自动删 uploader job。
**不要**用 `cp -a` 直接往 PVC 拷数据（权限报错，`013704` 的教训）。

查看已有数据（"运行协议"第 2 问让用户确认时用这个）：
```bash
export PATH="$HOME/.krew/bin:$PATH"
kubectl -n bonete61 exec <任一挂PVC的pod> -- ls -lh /mnt/pvc/experiments/<alias>/data/
```

**旧路径迁移**：早期 yaml 里指向 `/mnt/pvc/t-yifeili/runs/<旧JOB_NAME>/...` 的
数据，如还要复用，在挂 PVC 的 pod 里 `cp -r` 到固定路径下再改 yaml（pod 内
拷贝没有 dev box 的权限问题）。

## 训后自动全量 eval（一条 job:train → 全量评测,2026-07-09 起为 harness 多节点版）

用组合脚本代替纯训练脚本,训完在**同一个 job** 里自动评测刚出炉的 ckpt:

```bash
NODES=4 \
SFT_CONFIG=examples/train_full/<训练yaml> \
WANDB_PROJECT=web-agent-sft AZBLOB_AUTO_PUSH=0 \
bash docker/submit_sft_eval_q35_image.sh
```

**eval 走 mini-web-agent 自带的 OM2W harness(vllm serve + agent 循环,非
SkyRL),训练用几个节点、eval 就用几个节点并行**(机制见下面"分布式 eval
模块"一节,这里只是把它链在训练后面):

- master 训完做 ckpt 同步 + vision merge,把 ckpt 路径写进
  `$OUTPUT_DIR/.ckpt_ready_for_eval` sentinel;
- worker 节点训完**不退出**,等到 sentinel 后各自起本节点 vLLM(tp=8)
  跑 1/NODES 的任务分片;
- master 跑完自己的分片后等所有分片 done 标记,统一 judge + 汇总。

默认行为(都可 env 覆盖):

| 项 | 默认 | 说明 |
|---|---|---|
| `EVAL_BACKEND` | `harness` | 设 `skyrl` 回旧的 SkyRL eval_entrypoint(仅 master 评,需 SkyRL 上传 + echo-rl-openai secret) |
| `TASK_LEVEL` | `all` | 300 任务(easy80+medium143+hard77);可选 easy/medium/hard |
| `TOTAL_WORKERS` | `80` | 全 job browserbase 总并发,自动均分到各节点(80 是验证过的安全水位,加大前先确认配额) |
| `EVAL_RUN_ID` | `$JOB_NAME` | eval 稳定 id;job 被杀后可用独立 eval 提交 + 同一 id 断点续评 |
| `RETRY_FAILED` | `0` | 置 1 重跑 result.json 带 run_exception 的任务 |

与纯训练脚本的差异:多挂 webchain-sampling secret(browserbase + judge key,
由 `CREDENTIALS_FILE`,默认 `/data/t-yifeili/webchain_sampling/cred.sh` 自动
创建);harness 后端**不再上传 SkyRL**。eval 失败/被杀不影响已同步的 ckpt,
拿 pod 日志里的 `EVAL_RUN_ID` 用 `submit_dist_eval_q35_image.sh` 断点续评。
最终结果:`/mnt/pvc/<alias>/evals/<EVAL_RUN_ID>/logs/<EVAL_RUN_ID>/run_summary_judge.json`
(总分 + per-level breakdown)。

## 分布式 eval 模块（独立提交 / 多节点数据并行 / 断点续评,2026-07-09 起）

`docker/submit_dist_eval_q35_image.sh` 把 harness eval 做成独立可提交的模块,
评任意 PVC 上的 HF ckpt(须已 vision merge):

```bash
EVAL_CKPT=/mnt/pvc/<alias>/models/qwen35_9b/full/<name> \
NODES=4 TASK_LEVEL=all \
bash docker/submit_dist_eval_q35_image.sh
```

三个核心能力:

1. **指定节点数**:`NODES=N`,每个节点独立起一个 tp=8 的 vLLM serve +
   自己的 agent worker 池,不存在跨节点推理依赖。
2. **多节点数据并行**:任务按文件序 `idx % N == rank` 均分
   (`om2w.py --num-shards/--shard-index`),所有分片写同一个 PVC 输出目录
   `/mnt/pvc/<alias>/evals/<EVAL_RUN_ID>/outputs/<task_id>/`;master 等
   `shards/shard_*.done`(内容校验 JOB_NAME,旧 job 残留不误触发)后对全目录
   `--judge-only` 统一判分。
3. **断点续评**:`<task_id>/result.json` 是完成判据。job 被杀后**原样重提同一
   `EVAL_RUN_ID`** 即接着跑(`--resume` 默认开);`RETRY_FAILED=1` 额外重跑
   run_exception 的任务;全部任务已完成时重提 = 只重跑 judge。judge 本身也
   增量(om2w_judge 跳过已判分 task_id)。

默认参数:`TOTAL_WORKERS=80` 均分到各节点(**总 browserbase 并发 = 80**,
配额没确认前别加);`MAX_MODEL_LEN=65536` + 客户端 sliding window
`MAX_CONTEXT_TOKENS=48000`(防长会话 vLLM 400);serve 自动挂
`configs/qwen3_5_train_aligned.jinja`(think 对齐,置 `CHAT_TEMPLATE=` 用
ckpt 自带模板);benchmark 配置默认 `benchmark/om2w_sft_state_debug_vllm_sft_ckpt.yaml`。

产物布局(`/mnt/pvc/<alias>/evals/<EVAL_RUN_ID>/`):
`outputs/<task_id>/result.json` 逐任务;`outputs_eval_1/` judge 明细;
`logs/<EVAL_RUN_ID>/generation_summary_shard*.json` 各分片生成汇总、
`run_summary_judge.json` 最终总分 + level_breakdown、`vllm_shard*.log`。

注意:eval 值不值得多节点看瓶颈——300 任务 80 并发约 1-2 小时,瓶颈是浏览器
I/O 和长尾任务;多节点主要在你把 TOTAL_WORKERS 提上去(配额允许)时才显著提速。

## 多用户提交（2026-07-08 起脚本已 user-agnostic）

docker/ 下所有脚本的身份与路径默认值都是自动推导的,**任何用户从自己的
checkout 直接跑 TL;DR 命令即可**,不需要特殊配置:

| 项 | 默认推导 |
|---|---|
| 仓库根(`MINI_WEB_AGENT_DIR`/`SRC`) | 脚本自身所在仓库 |
| `USER_ALIAS`(job 名/quota/PVC 归属) | `whoami` 去掉 @domain |
| 数据上传目录(`DEST_ROOT`) | `/mnt/pvc/experiments/<自己alias>/data` |
| 轻量树(`LIGHT_ROOT`) | `/tmp/mwa-light-<自己alias>`(同机多人不冲突) |
| aifsdk(`SUBMIT`) | `/data/t-yifeili/aifsdk/...`(可读即可,可覆盖) |

每个用户自己要准备的只有:bonete61 的 kubectl 凭证、和 `WANDB_HOST` 匹配的
`WANDB_API_KEY`、自己的 `PROJECT_NAME` workstream。
读别人的数据不用重传(PVC 跨用户可读),写入(上传/ckpt)自动落自己名下。

**W&B key(2026-07-08 起提交脚本自动预检,不再需要手动处理)**:
`submit_job.sh` 从提交者 shell 环境取 `WANDB_API_KEY`,取不到会 fallback 到
微软内部 wandb 实例的 key——与强制的 `WANDB_HOST=https://api.wandb.ai` 不配对,
job 会在数据预处理+权重加载全走完后才在 trainer 初始化处 401 死掉
(af8b0、luyadong `adf40` 两次教训,每次白烧几分钟 GPU)。

现在两个 submit 脚本提交前都会 source `docker/wandb_key_preflight.sh`:
- shell 里没有 `WANDB_API_KEY` → 自动加载共享 key
  `/data/t-yifeili/.secrets/wandb_api_key`(run 进 flyhero99/web-agent-sft
  项目,run 名带提交者 alias,可区分);
- 加载不到(文件不可读)或 key 是 `local-` 开头的内部 fallback → **拒绝提交**
  并打印解法,绝不带病上集群。

想用自己账号的,提交前 `export WANDB_API_KEY=<自己的key>`
(https://wandb.ai/authorize)即可覆盖自动行为;可放进 `~/.bashrc` 一劳永逸。
因 wandb 打挂的历史 job 记得删(`kubectl -n bonete61 get jobs -l submitter=<alias>`),
否则一直占卡。

## Guest 提交 — 别人从 yifeili 的 sandbox 提交，归属记在自己名下

适用场景：另一个用户直接登录 yifeili 的 sandbox、`cd /data/t-yifeili/mini-web-agent`
提交 job（共享工作树、数据、aifsdk、kubectl 凭证都用现成的——这些已被接受），
但 **job 命名 / quota bucket / GPU dashboard / submitter label / W&B run 必须
记在 guest 自己名下**。做法：提交时覆盖归属相关的环境变量
（在自己 sandbox 有 checkout 的话,优先走上面"多用户提交",什么都不用覆盖）。

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
（`MERGE_VISION=0` 关闭）。

### 为什么 ckpt 缺权重、为什么不改 trainer 端保存

qwen3_5 full-SFT 存下来的 ckpt 只有 `model.language_model.*` + `lm_head`
（760 tensors）。缺的是**两类**：`model.visual.*`（333 个，冻结的 vision tower）
和 `mtp.*`（15 个，投机解码 MTP 头——HF 训练类根本不实例化它，只存在于 base
safetensors 里）。vLLM 加载 `Qwen3_5ForConditionalGeneration` 两类都要。

**结论：不改 trainer、维持 merge 方案**。理由：
1. 就算改 trainer 存下 vision，`mtp.*` 依然缺（训练时模型对象里就没有），
   照样要从 base 补——trainer 端改动永远做不出 standalone ckpt，merge 省不掉；
2. vision 冻结不变，每个 ckpt 重复存 ~2GB 纯浪费（save_steps=200 +
   save_total_limit=2 + rsync + blob 全链路放大）；
3. 要动 LlamaFactory fork 的 deepspeed/save_only_model 保存内部，风险大于收益；
   merge 方案已在 9B eval 全链路验证过。

集群 job 的**最终** ckpt 已自动 merge（日志确认走了 `[merge] completing VL ckpt`
而非 `[merge][warn]`）。需要**手动 merge** 的场景：中间步 `checkpoint-<N>/`、
本地训练产物、merge 步骤失败的 job。

### 手动 merge 命令

```bash
# 判断是否已 merge:两个文件都在即可直接用
ls "$CKPT"/vision.safetensors "$CKPT"/model.safetensors.index.json

# dev box(base 快照在本地 hf_cache)
/data/t-yifeili/miniconda3/envs/echo-rl/bin/python \
  /data/t-yifeili/mini-web-agent/scripts/merge_vision_from_base.py \
  --ckpt "$CKPT" \
  --base "$(ls -d /data/t-yifeili/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)"

# pod 内(base 在 HF_HOME)
python scripts/merge_vision_from_base.py --ckpt "$CKPT" \
  --base "$(ls -d $HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)"
```

机制：只新增 `vision.safetensors`（缺的 348 个 tensor）+ 重写
`model.safetensors.index.json`，18GB 的 `model.safetensors` 不动——从 pod 往
dev box 补拉时只需这两个小文件。成功标志：打印
`tensors to copy from base (missing in ckpt): 348`。

### Inference（merge 后的 ckpt 直接起服务）

```bash
CKPT=/path/to/merged_ckpt
# 一劳永逸:把 train-aligned 模板写进 ckpt,以后 serve 不用带 --chat-template
cp /data/t-yifeili/mini-web-agent/configs/qwen3_5_train_aligned.jinja "$CKPT/chat_template.jinja"

vllm serve "$CKPT" \
  --served-model-name policy \
  --mm-processor-kwargs '{"max_pixels":262144}'    # 与训练 image_max_pixels 对齐

# 快速冒烟(起完必跑 MARKER 探针验证模板对齐,见 docs/qwen3_5_think_alignment.md §2.2)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"policy","messages":[{"role":"user","content":"hi"}],"max_tokens":32}' | head -c 400
```

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

## 断点续 train（warm restart,job 被杀后从中间 ckpt 接着训）

训练 yaml 都是 `save_only_model: true`(省 PVC 空间),checkpoint-* 里**没有
optimizer/DeepSpeed 状态,`resume_from_checkpoint` 真断点续跑不可能**。正确
做法是 warm restart:把中间 ckpt 当 `model_name_or_path` 接着训。丢的只是
optimizer 动量和最后一个 save_steps 之后的进度,影响很小。四步:

1. **备份 ckpt 出易失目录**(在挂 PVC 的 pod 里)。中间 ckpt 落在
   `/mnt/pvc/<alias>/code/mini-web-agent/LlamaFactory/saves/<output_dir>/checkpoint-<N>`,
   同 yaml 重跑会被 `overwrite_output_dir` + save_total_limit 轮转顶掉:
   ```bash
   mkdir -p /mnt/pvc/<alias>/models
   cp -a <saves里的checkpoint-N> /mnt/pvc/<alias>/models/<run名>_ckpt<N>_bak
   ```
2. **vision merge**(必做!文本 SFT 的中间 ckpt 缺 vision tower,直接当
   `model_name_or_path` 加载会把视觉塔随机初始化):
   ```bash
   BASE=$(ls -d /mnt/pvc/<alias>/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1)
   python /mnt/pvc/<alias>/code/mini-web-agent/scripts/merge_vision_from_base.py \
     --ckpt <备份目录> --base "$BASE"
   ```
3. **写续训 yaml**(现成例子:`examples/train_full/*_cont1000.yaml`(2 node)、
   `*_cont600_4node.yaml`(4 node)),关键折算:
   - `model_name_or_path` = 备份目录绝对路径;
   - **epoch 按剩余量折算**:steps/epoch = 样本数/(NODES×8);ckpt 的 epoch =
     N/每epoch步数;`num_train_epochs` = 目标总 epoch − 已训 epoch;
   - **学习率从原 cosine 中断处接**:查
     `<ckpt>/trainer_state.json` 里最后的 learning_rate,把它设为新
     `learning_rate`,`warmup_ratio: 0.01`,cosine 衰减到 0(不要从原 LR 重新
     warmup,二次大 LR 会冲击已训权重);
   - `output_dir`/`run_name` 加 `_cont<N>` 后缀,别覆盖原 run。
4. **提交时 NODES 必须和折算假设一致**(global batch 变了 epoch/LR 折算就错了)。

## 关键文件

| 路径 | 作用 |
|------|------|
| `LlamaFactory/data/web_agent_seq_om2w4000_run1.json` | raw 数据（仅 dev box 本地） |
| `docker/upload_data_to_pvc.sh` | bundle → PVC 固定数据路径（分块+校验） |
| `/mnt/pvc/experiments/t-yifeili/data/` | PVC 固定数据根目录（pod 内） |
| `LlamaFactory/scripts/make_web_agent_sequential_compact_tools_sft.py` | rollout → seq ShareGPT |
| `LlamaFactory/scripts/package_web_agent_images.py` | 打 portable bundle |
| `LlamaFactory/examples/train_full/qwen35_9b_web_agent_seq_om2w4000_run1_40k_4node.yaml` | 本次训练配置 |
| `docker/run_sft_q35_image.sh` | in-pod：train + PVC sync + vision merge（+链分布式 eval） |
| `docker/run_dist_eval_q35_image.sh` | in-pod：每节点 vLLM+分片生成,master judge 汇总 |
| `docker/submit_dist_eval_q35_image.sh` | 独立提交多节点/断点续评的 harness eval |
| `src/miniswewebagent/run/benchmarks/om2w.py` | harness 入口(--num-shards/--resume/--judge-only) |
| `scripts/merge_vision_from_base.py` | 手动补全 ckpt（vision+mtp，命令见"训练之后"） |
| `docs/web_agent_seq_label1_4node_sft_20260630.md` | 上次 4-node run 全记录 |
| `docs/qwen3_5_think_alignment.md` | `<think>` 对齐手册（serving 必读） |
| `configs/qwen3_5_train_aligned.jinja` | train-aligned chat template |

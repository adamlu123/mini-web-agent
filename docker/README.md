# echo-rl on Lambda cluster — quick start

跑 `echo_rl.web_agent` 训练在 bonete61 (B200) cluster。

## Setup（一次性）

下面这几步都已经做过了——只是写在这里方便换机器/新成员复现。

```bash
# 1. Cluster access (OIDC -> Lambda)
bash /data/t-yifeili/aifsdk/clusters/lambda/submission/volcano/setup_cluster_access.sh
echo 'export PATH="$HOME/.krew/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# 2. Azure CLI（push 镜像、读 ACR）
az login --use-device-code --tenant 72f988bf-86f1-41af-91ab-2d7cd011db47

# 3. Build + push 镜像（30-60 min，只首次需要）
cd /data/t-yifeili/mini-web-agent
bash docker/build.sh
# 结果：aifrontiers.azurecr.io/t-yifeili/echo-rl:latest

# 4. K8s pull secret + service account 挂载（绕 AcrPull RBAC）
ADMIN_PW=$(az acr credential show --name aifrontiers --query 'passwords[0].value' -o tsv)
kubectl -n bonete61 create secret docker-registry acr-pull \
    --docker-server=aifrontiers.azurecr.io \
    --docker-username=aifrontiers --docker-password="$ADMIN_PW"
kubectl -n bonete61 patch serviceaccount aif-bonete-uami \
    -p '{"imagePullSecrets":[{"name":"acr-pull"}]}'

# 5. 把 cred.sh 放进 k8s secret（OPENAI/BROWSERBASE/HF）
sed -E 's|/data/t-yifeili/tmp/huggingface|/mnt/pvc/t-yifeili/hf_cache|g' \
    /home/t-yifeili/cred.sh > /tmp/cred-cluster.sh
kubectl create secret generic echo-rl-creds \
    --from-file=cred.sh=/tmp/cred-cluster.sh -n bonete61
rm /tmp/cred-cluster.sh
```

每次新 shell 还需要：

```bash
export PATH="$HOME/.krew/bin:$PATH"
kubectl auth whoami     # token 过期就 `kubectl oidc-login clean && kubectl oidc-login setup ...`
```

## 提交训练 job

**节点不用手动申请** —— submit 脚本通过 Volcano 描述资源需求（`--node 1 --gpu-per-node 8` 等），Volcano scheduler 自动从 cluster 里挑一台满足条件的 node 调度 pod 上去。Pod 在那个 node 上自动拉镜像（用 setup 步 4 挂的 admin secret）、起容器、跑你的 `--cmd`。整个过程对你透明。

```bash
cd /data/t-yifeili/mini-web-agent
TRAIN_TIMEOUT_SEC=86400 bash docker/submit_real_train_batch.sh
```

`TRAIN_TIMEOUT_SEC` 是 python 进程的 hard timeout（防卡死），1500 = 25 min（适合 smoke），86400 = 24 h（真训练）。脚本会：

1. 上传 `mini-web-agent` + `SkyRL` 到 PVC（~3 min）
2. 创 Volcano vcjob（8 B200 + 64 CPU + 512 GiB RAM）
3. Pod ready 后自动跑：source creds → `pip install -e` 4 个 editable → 起 `python -m echo_rl.web_agent.entrypoint`
4. `--follow-logs` 流式打日志到你的终端
5. 训练退出或 timeout → pod 退出 → GPU 释放

改 config：`CONFIG=echo_configs/<your-yaml> bash docker/submit_real_train_batch.sh`

改优先级：`PRIORITY=high PRIORITY_CLASS_NAME=high bash docker/submit_real_train_batch.sh`。脚本默认 `medium`。`PRIORITY` 只是 job 名前缀；`PRIORITY_CLASS_NAME` 是真调度优先级。bonete61 注册的合法值：**`high` / `medium` / `low`**（用 `kubectl create --dry-run=server -f pod.yaml` 验证过）。

后续看日志：

```bash
JOB=$(kubectl get vcjob -n bonete61 -l submitter=t-yifeili --sort-by=.metadata.creationTimestamp -o name | tail -1 | cut -d/ -f2)
kubectl logs -n bonete61 -f ${JOB}-master-0
```

杀任务：

```bash
kubectl delete vcjob -n bonete61 ${JOB}
```

## Interactive debug

申一个 8-GPU pod 拿 shell，自己 paste 命令跑实验：

```bash
cd /data/t-yifeili/mini-web-agent
bash docker/submit_interactive_8gpu.sh
```

等几分钟 pod ready，你会直接进 bash。然后：

```bash
source /run/secrets/echo-rl-creds/cred.sh
unset OPENAI_GATEWAY_API_KEY

cd $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/SkyRL
pip install --no-deps -e . -e ./skyrl-gym -e ./skyrl-agent
cd ../mini-web-agent && pip install --no-deps -e .

export MINI_WEB_AGENT_ROOT=$PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/mini-web-agent
export ECHO_RL_DATA=$MINI_WEB_AGENT_ROOT/data/web_agent
export OUTPUT_DIR=$PVC_MOUNT/$USER_ALIAS/outputs/$JOB_NAME
mkdir -p $OUTPUT_DIR
export WANDB_MODE=offline

cd $PVC_MOUNT/$USER_ALIAS/runs/$JOB_NAME/SkyRL
python -m echo_rl.web_agent.entrypoint --config echo_configs/qwen35_4b_web_agent_hard_4gpu.yaml
```

代码在 PVC 上持久，改 yaml 重跑不用重新上传/`pip install -e`。**Ctrl-D 退出 → job 自动删 → GPU 释放**。

进已经在跑的 batch job 看一眼：

```bash
kubectl exec -it -n bonete61 ${JOB}-master-0 -- bash
```

---

## Smoke test（验证镜像）

```bash
cd /data/t-yifeili/mini-web-agent
bash docker/submit_smoke.sh
```

1 GPU，~10 min。验证：镜像能拉、editables 能装、Playwright 能起 Chromium。改了镜像或者升级依赖之后跑一下。

### 9B / Qwen3.5 vLLM smoke

```bash
cd /data/t-yifeili/mini-web-agent
bash docker/submit_smoke_9b_triton.sh                 # 默认 echo-rl:latest
# 换镜像对照：
IMAGE=aifrontiers.azurecr.io/<base>:<tag> \
  EXTRA_PIP="--upgrade transformers" \
  JOB_TAG=smoke9bnv bash docker/submit_smoke_9b_triton.sh
```

1 GPU，~6 min。在 vLLM 里加载 `Qwen/Qwen3.5-9B` 并生成 1 个 token —— 这会强制
JIT 编译 Qwen3.5 Gated-DeltaNet（GDN/FLA）那条 **Triton kernel** 路径，是排查
"9B 报 triton 错" 最便宜的复现手段，不用烧整个训练。默认带 `--follow-logs`，
所以报错会直接流到你终端（这些 pod 失败后会很快 abort + GC，不 stream 就抓不到
traceback）。脚本里两个 python probe 都写成**真文件**再跑（不是 `python - <<EOF`）
并带 `if __name__ == "__main__":` 守卫 —— vLLM V1 用 spawn 起 EngineCore worker，
会 re-import probe 文件，缺守卫会递归 spawn 报错，缺真文件会 `<stdin>` not found。

## 镜像选型：跑 9B 用哪个？（2026-05 实测）

> **结论：跑 Qwen3.5-9B 就用本仓库的 `t-yifeili/echo-rl:latest`，别换同学给的
> `nvidia25.08-...-vllm0.10.3.dev` 镜像。** 后者更旧，根本不支持 Qwen3.5。

用上面的 9B smoke 在 cluster 上实测两个镜像：

| | `t-yifeili/echo-rl:latest`（本仓库） | `nvidia25.08-...-vllm0.10.3.dev20250918`（同学给的） |
|---|---|---|
| torch | 2.10.0+cu128 | 2.10.0.dev+cu130 |
| triton | **3.6.0** | 3.4.0 |
| vllm | **0.19.0** | 0.10.2rc3 |
| transformers | **5.3.0** | 4.56.1 |
| 加载 Qwen3.5-9B | ✅ GENERATION OK，GDN Triton kernel 正常 JIT 编译 | ❌ `model type 'qwen3_5' ... Transformers does not recognize` |

- 同学镜像的 transformers 4.56 不认识 `qwen3_5`；容器里 `pip install --upgrade
  transformers` 也只到 released 4.56.1，仍不够（Qwen3.5 要 transformers 5.x），
  而且它的 vllm 0.10.2 也太旧。要硬掰过来等于重建当前镜像，得不偿失。
- 当前镜像的栈（transformers 5.3 / vllm 0.19 / triton 3.6）已经能正常跑 9B
  **推理 + GDN Triton 编译**，所以 "9B triton 报错" 不是镜像/triton 版本问题。
  真正的报错若出现，优先怀疑 **训练路径（FSDP2）** 或被误判成 triton 的
  **KV-cache OOM / config**（见下）。

### 9B 上过的坑（config，不是镜像）

- **KV-cache OOM**：vLLM 的 `max_model_len` 默认取模型自带的 262144，
  `max_num_batched_tokens: 262144` 在显存小的卡上会把 activation buffer 撑爆、
  KV cache 只剩几 GiB 然后 OOM（本地 4×H100 80GB 上复现过）。
  `qwen35_9b_web_agent_easy_4gpu.yaml` 已把它降到 `32768`（够单条
  `max_seq_len=34096` 的 prefill）；B200 180GB 的 8gpu 配置显存够，暂时保留 262144。

## 修改并 rebuild 环境

只有 **改了 pip dep / playwright / apt 系统包** 时才要重 build。**改 echo_rl 或 SkyRL 代码不用** —— editables 每次 submit 都重装。

```bash
# 1. 本地装好新 dep（在 echo-rl conda env 里）
conda activate echo-rl
pip install <new-pkg>

# 2. 重新生成 requirements.txt（去掉 editables）
conda run -n echo-rl pip freeze \
  | grep -vE '^-e |^(skyrl|skyrl-gym|skyrl-agent|echo-rl)==' \
  > docker/requirements.txt

# 3. （如果 dep 需要 CUDA 编译且 PyPI 没 prebuilt wheel）
#    本地装好后，把 ~/.cache/pip/wheels/.../*.whl 拷到 docker/wheels/
#    否则 ACR 2-core agent 编译会卡几十分钟
#
#    Fresh clone 上 build.sh 之前一定要先 populate docker/wheels/，
#    因为这俩 .whl 太大没进 git（gitignore'd）。当前需要：
#      causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl
#      flash_attn-2.8.1-cp312-cp312-linux_x86_64.whl
#    路径在 ~/.cache/pip/wheels/{ea,88}/.../  下面，find 一下即可：
#      mkdir -p docker/wheels
#      find ~/.cache/pip/wheels -name 'causal_conv1d-*.whl' -exec cp {} docker/wheels/ \;
#      find ~/.cache/pip/wheels -name 'flash_attn-*.whl' -exec cp {} docker/wheels/ \;

# 4. 重 build
bash docker/build.sh                 # 默认 tag=YYYYMMDD + latest
TAG=v2 bash docker/build.sh          # 自定义 tag
```

`build.sh` 同时打 `:<date>` 和 `:latest`，submit 脚本默认用 `:latest` 自动拿到新版本。

---

## 文件参考

| 文件 | 用途 |
|---|---|
| `Dockerfile` | 镜像（NVIDIA pytorch + 269 pip + Playwright） |
| `requirements.txt` | 固定 pip 版本 |
| `wheels/` | 预编译 `causal-conv1d` / `flash_attn` wheel |
| `build.sh` | ACR cloud build + push |
| `submit_real_train_batch.sh` | 8-GPU batch 训练（自动 timeout 保护） |
| `submit_uv_9b_easy.sh` | 8-GPU batch 训练，复用 PVC 上 persistent uv venv |
| `submit_interactive_8gpu.sh` | 8-GPU interactive shell |
| `submit_smoke.sh` | 1-GPU 镜像 smoke 测试 |
| `submit_smoke_9b_triton.sh` | 1-GPU Qwen3.5-9B vLLM smoke（复现/排除 GDN Triton 编译问题） |
| `submit_via_condapack.sh` | 备选：默认 NVIDIA image + conda-pack tar（不需要重 build 镜像） |

---

## Persistent uv venv on PVC

`docker/submit_uv_9b_easy.sh` 在 PVC 上建一份 uv venv，跨 job 复用 —— 省掉每 job
重装 editables 的 30s + 不用每次拉 wandb 之类的轻量 dep。

PVC layout：
```
/mnt/pvc/t-yifeili/code/SkyRL/            ← 稳定路径，每个 job rsync 一次
/mnt/pvc/t-yifeili/code/mini-web-agent/   ← 同上
/mnt/pvc/t-yifeili/envs/echo-rl-uv/.venv  ← uv venv（--system-site-packages）
/mnt/pvc/t-yifeili/envs/echo-rl-uv/bin/uv ← uv binary
```

`.venv` 用 `--system-site-packages` 创建，继承 image 里的 torch / vllm /
flash-attn / transformers 等重 dep；editable .pth 指向 `/mnt/pvc/.../code/`
稳定路径。Image 改了就 rebuild + 删 venv 重建；改 echo-rl 代码不用任何重建。

跑：
```bash
cd /data/t-yifeili/mini-web-agent
bash docker/submit_uv_9b_easy.sh
# 改 config: CONFIG=echo_configs/qwen35_9b_web_agent_hard_8gpu.yaml bash docker/submit_uv_9b_easy.sh
```

第一次运行 ~1 min 装 rsync + uv + venv + editables；后续运行 ~10s
（rsync + uv pip install -e 都是 noop / 增量）。

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
| `submit_interactive_8gpu.sh` | 8-GPU interactive shell |
| `submit_smoke.sh` | 1-GPU 镜像 smoke 测试 |
| `submit_via_condapack.sh` | 备选：默认 NVIDIA image + conda-pack tar（不需要重 build 镜像） |

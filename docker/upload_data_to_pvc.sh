#!/usr/bin/env bash
# 把本地数据目录(通常是 portable bundle)上传到 PVC 的固定数据路径:
#   /mnt/pvc/experiments/t-yifeili/data/<name>
# 训练配置的 dataset_dir/media_dir 直接指这个绝对路径,代码上传就不用再带数据。
#
# 用法:
#   bash docker/upload_data_to_pvc.sh <本地目录> [目标名]
#   目标名缺省 = 本地目录 basename。
#
# 机制(与 aifsdk submit_job.sh 的 uploader 一致):起一个挂 PVC 的 1c2Gi CPU pod,
# tar 分块 kubectl cp 进去(每块独立重试,断了只重传当前块),pod 内解压到
# <dest>.tmp 后原子替换,最后按文件数+字节数校验。结束自动删 uploader job。
#
# 可覆盖的环境变量:
#   DEST_ROOT   (默认 /mnt/pvc/experiments/t-yifeili/data)
#   NAMESPACE   (默认 bonete61)   PVC_CLAIM_NAME (默认 pvc-vast-bonete61)
#   CHUNK_SIZE  (默认 512M)       PROJECT_NAME   (默认 cua)
set -euo pipefail

SRC_DIR="${1:?用法: $0 <本地目录> [目标名]}"
SRC_DIR="$(readlink -f "$SRC_DIR")"
[[ -d "$SRC_DIR" ]] || { echo "[error] 不是目录: $SRC_DIR"; exit 1; }
DEST_NAME="${2:-$(basename "$SRC_DIR")}"

NAMESPACE="${NAMESPACE:-bonete61}"
PVC_CLAIM_NAME="${PVC_CLAIM_NAME:-pvc-vast-bonete61}"
PVC_MOUNT="${PVC_MOUNT:-/mnt/pvc}"
DEST_ROOT="${DEST_ROOT:-$PVC_MOUNT/experiments/t-yifeili/data}"
DEST_DIR="$DEST_ROOT/$DEST_NAME"
CHUNK_SIZE="${CHUNK_SIZE:-512M}"
USER_ALIAS="${USER_ALIAS:-$(whoami | tr '@.' '--')}"
PROJECT_NAME="${PROJECT_NAME:-cua}"
PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-high}"

export PATH="$HOME/.krew/bin:$PATH"

echo "[info] 上传 $SRC_DIR ($(du -sh "$SRC_DIR" | cut -f1)) -> $DEST_DIR"

# ---- 1. 起 uploader pod ----------------------------------------------------
UPLOADER_JOB_FQN="$(kubectl create -n "$NAMESPACE" -f - -o name <<YAML
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  generateName: cpu-dataupload-
  namespace: ${NAMESPACE}
  labels:
    submitter: ${USER_ALIAS}
    workstream: ${PROJECT_NAME}
    job_type: uploader
spec:
  queue: ${NAMESPACE}
  minAvailable: 1
  tasks:
    - name: master
      replicas: 1
      template:
        spec:
          schedulerName: volcano
          priorityClassName: ${PRIORITY_CLASS_NAME}
          restartPolicy: Never
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: ${PVC_CLAIM_NAME}
          containers:
            - name: master
              image: nvcr.io/nvidia/pytorch:25.08-py3
              command: ["/bin/sh", "-c", "sleep 1d"]
              volumeMounts:
                - name: data
                  mountPath: ${PVC_MOUNT}
              resources:
                requests: &requests
                  cpu: 1
                  memory: 2Gi
                limits: *requests
YAML
)"
UPLOADER_JOB_NAME="${UPLOADER_JOB_FQN#*/}"
echo "[info] uploader job: $UPLOADER_JOB_NAME"
cleanup() {
  kubectl -n "$NAMESPACE" delete "job.batch.volcano.sh/$UPLOADER_JOB_NAME" --wait=false >/dev/null 2>&1 || true
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

POD=""
DEADLINE=$((SECONDS + 1800))
while [[ -z "$POD" ]]; do
  POD="$(kubectl -n "$NAMESPACE" get pods -l "volcano.sh/job-name=$UPLOADER_JOB_NAME" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
  [[ $SECONDS -lt $DEADLINE ]] || { echo "[error] 等 uploader pod 超时"; exit 1; }
  sleep 2
done
kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$POD" --timeout=30m
echo "[info] uploader pod ready: $POD"

# ---- 2. tar + 分块 ---------------------------------------------------------
TMP_DIR="$(mktemp -d "/tmp/pvc_upload_XXXXXX")"
TAR_FILE="$TMP_DIR/data.tar.gz"
echo "[info] 打 tar..."
COPYFILE_DISABLE=1 tar -C "$SRC_DIR" -czf "$TAR_FILE" .
echo "[info] tar 大小: $(du -sh "$TAR_FILE" | cut -f1),按 $CHUNK_SIZE 分块"
split -b "$CHUNK_SIZE" -d -a 4 "$TAR_FILE" "$TMP_DIR/chunk."
rm -f "$TAR_FILE"
CHUNKS=("$TMP_DIR"/chunk.*)
echo "[info] 共 ${#CHUNKS[@]} 块"

REMOTE_TMP="/tmp/pvc_upload_$$"
kubectl -n "$NAMESPACE" exec "$POD" -- mkdir -p "$REMOTE_TMP"

# ---- 3. 逐块 cp,每块最多重试 5 次 -----------------------------------------
i=0
for chunk in "${CHUNKS[@]}"; do
  i=$((i + 1))
  base="$(basename "$chunk")"
  attempts=0
  until kubectl -n "$NAMESPACE" cp "$chunk" "$POD:$REMOTE_TMP/$base" --retries=3; do
    attempts=$((attempts + 1))
    [[ $attempts -lt 5 ]] || { echo "[error] 块 $base 重试 5 次仍失败"; exit 1; }
    echo "[warn] 块 $base cp 失败($attempts/5),5s 后重试..."
    sleep 5
  done
  echo "[info] 已传 $i/${#CHUNKS[@]}: $base"
done

# ---- 4. pod 内解压到 .tmp,原子替换 -----------------------------------------
echo "[info] pod 内解压 -> $DEST_DIR"
kubectl -n "$NAMESPACE" exec "$POD" -- sh -c "
  set -e
  mkdir -p '$DEST_ROOT'
  rm -rf '$DEST_DIR.tmp'
  mkdir -p '$DEST_DIR.tmp'
  cat $REMOTE_TMP/chunk.* | tar -C '$DEST_DIR.tmp' -xzpf - --warning=no-unknown-keyword
  rm -rf '$REMOTE_TMP' '$DEST_DIR'
  mv '$DEST_DIR.tmp' '$DEST_DIR'
"

# ---- 5. 校验:文件数 + 总字节数 --------------------------------------------
LOCAL_COUNT=$(find "$SRC_DIR" -type f | wc -l)
LOCAL_BYTES=$(du -sb "$SRC_DIR" | cut -f1)
read -r REMOTE_COUNT REMOTE_BYTES < <(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "echo \$(find '$DEST_DIR' -type f | wc -l) \$(du -sb '$DEST_DIR' | cut -f1)")
echo "[info] 校验 本地 files=$LOCAL_COUNT bytes=$LOCAL_BYTES / 远端 files=$REMOTE_COUNT bytes=$REMOTE_BYTES"
if [[ "$LOCAL_COUNT" != "$REMOTE_COUNT" ]]; then
  echo "[error] 文件数不一致,上传可能不完整(远端目录仍保留,可重跑覆盖)"; exit 1
fi
# NFS/文件系统 block 差异会让 du -sb 略有出入,只在差 >1% 时报错
DIFF=$(( LOCAL_BYTES > REMOTE_BYTES ? LOCAL_BYTES - REMOTE_BYTES : REMOTE_BYTES - LOCAL_BYTES ))
if (( DIFF * 100 > LOCAL_BYTES )); then
  echo "[error] 字节数偏差 >1%,请检查"; exit 1
fi

echo "[ok] 上传完成: $DEST_DIR"
echo "[ok] 训练配置里写: dataset_dir/media_dir = $DEST_DIR"

#!/usr/bin/env bash
# Pull a checkpoint directory off the lambda-cluster PVC to local disk.
#
#   POD=luyadong-pvcpull-223e6-master-0 \
#   SRC=/mnt/pvc/datasets/rlscaling/agoswami/ckpts_intermediate_01/<run>/6000/llama \
#   DST=/data/yadonglu/ckpts/<name> \
#   bash scripts/local/om2w/pull_ckpt_from_pvc.sh
#
# Why not plain `kubectl cp` / `kubectl exec -- cat`:
#
#   1. The msr02 API server sits behind Cloudflare, which intermittently 403s the
#      SPDY connection upgrade that exec/cp use. Websockets
#      (KUBECTL_REMOTE_COMMAND_WEBSOCKETS=1) get through, but a long multi-line
#      quoted `bash -c` still trips the WAF -- short single-line `sh -c` is fine.
#
#   2. The websocket stream SILENTLY TRUNCATES ITS TAIL on close, exit code 0:
#        head -c 1000000000  -> 997277696 bytes locally
#        1 KiB data + 4 KiB zero pad -> 4836 bytes (data intact, pad eaten)
#      The loss is whatever was in flight when the remote process exited (~2.5 MB
#      at 82 MB/s, and it varies run to run), so it is NOT fixed by retrying --
#      a one-shot 58 GB `cat` yields a corrupt checkpoint that still loads.
#
# Each file is therefore fetched in CHUNK_MB slices via `dd skip=`, and every
# slice is followed by `sleep` (lets the stream drain) AND PAD_MB of zeros (so
# any residual loss eats padding, never data). The slice is then cut back to its
# exact expected length. Finally every file is md5-verified against the pod, so a
# bad transfer fails loudly instead of silently. Resumable: completed chunks and
# completed files are skipped on rerun.
set -euo pipefail

POD="${POD:?set POD}"
SRC="${SRC:?set SRC}"
DST="${DST:?set DST}"
CHUNK_MB="${CHUNK_MB:-2048}"
PAD_MB="${PAD_MB:-16}"
DRAIN_SECS="${DRAIN_SECS:-3}"
MAX_TRIES="${MAX_TRIES:-8}"

export KUBECTL_REMOTE_COMMAND_WEBSOCKETS=true

# kubectl exec with retry on Cloudflare WAF blocks (plain argv, no shell).
kx() {
  local i out
  for i in $(seq 1 "$MAX_TRIES"); do
    # generous: the verify step md5sums ~47 GB per shard off VAST NFS pod-side
    out=$(timeout 2400 kubectl exec "$POD" -- "$@" 2>&1) && {
      [[ "$out" == *Cloudflare* ]] || { printf '%s' "$out"; return 0; }
    }
    [[ "$out" == *Cloudflare* ]] || { printf '%s' "$out" >&2; return 1; }
    sleep $((i * 5))
  done
  echo "[error] kubectl exec blocked after $MAX_TRIES tries: $*" >&2
  return 1
}

mkdir -p "$DST"

echo "[info] listing $SRC"
mapfile -t ENTRIES < <(kx find "$SRC" -maxdepth 1 -type f -printf '%s %f\n' | sort -k2)
[ "${#ENTRIES[@]}" -gt 0 ] || { echo "[error] no files found at $SRC" >&2; exit 1; }

total=0
for e in "${ENTRIES[@]}"; do total=$((total + ${e%% *})); done
echo "[info] ${#ENTRIES[@]} files, $((total / 1024 / 1024)) MiB total"

for e in "${ENTRIES[@]}"; do
  size="${e%% *}"; name="${e#* }"
  out="$DST/$name"

  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
    echo "[skip] $name (already $size bytes)"
    continue
  fi

  chunk_bytes=$((CHUNK_MB * 1024 * 1024))
  nchunks=$(( (size + chunk_bytes - 1) / chunk_bytes ))
  part_dir="$DST/.parts/$name"
  mkdir -p "$part_dir"
  echo "[pull] $name ($((size / 1024 / 1024)) MiB, $nchunks chunk(s) of ${CHUNK_MB}MiB)"

  for ((c = 0; c < nchunks; c++)); do
    part="$part_dir/$(printf '%05d' "$c")"
    want=$chunk_bytes
    [ $(( (c + 1) * chunk_bytes )) -gt "$size" ] && want=$(( size - c * chunk_bytes ))

    if [ -f "$part" ] && [ "$(stat -c %s "$part")" = "$want" ]; then
      echo "  [have] chunk $((c + 1))/$nchunks"
      continue
    fi

    # dd `skip` counts BLOCKS OF bs (1 MiB), not chunks -- so it is the chunk's
    # byte offset in MiB, NOT the chunk index. Getting this wrong reads chunk c
    # from offset c MiB: the assembled file has the exact right SIZE and a valid
    # safetensors header (chunk 0 is correct), so it loads in vLLM and passes a
    # tensor-shape audit while every weight past the first chunk is garbage.
    # Only the md5 check below catches it.
    skip_mib=$((c * CHUNK_MB))
    ok=0
    for ((t = 1; t <= MAX_TRIES; t++)); do
      timeout 1800 kubectl exec "$POD" -- sh -c \
        "dd if=$SRC/$name bs=1048576 skip=$skip_mib count=$CHUNK_MB status=none; sleep $DRAIN_SECS; dd if=/dev/zero bs=1048576 count=$PAD_MB status=none" \
        > "$part.raw" 2>/dev/null || true

      raw=$(stat -c %s "$part.raw" 2>/dev/null || echo 0)
      # need the full chunk; anything beyond it is drain padding we discard
      if [ "$raw" -ge "$want" ]; then
        head -c "$want" "$part.raw" > "$part"
        rm -f "$part.raw"
        ok=1; break
      fi
      echo "  [retry] chunk $((c + 1))/$nchunks: raw $raw < want $want (try $t)"
      rm -f "$part.raw"
      sleep $((t * 3))
    done
    [ "$ok" = 1 ] || { echo "[error] $name chunk $c failed after $MAX_TRIES tries" >&2; exit 1; }
    echo "  [ok] chunk $((c + 1))/$nchunks"
  done

  cat "$part_dir"/* > "$out"
  got=$(stat -c %s "$out")
  [ "$got" = "$size" ] || { echo "[error] $name assembled to $got, want $size" >&2; exit 1; }
  rm -rf "$part_dir"
  echo "[ok] $name assembled ($got bytes)"
done
rmdir "$DST/.parts" 2>/dev/null || true

echo "[info] verifying md5 against pod (reads the full checkpoint on both ends)"
fail=0
for e in "${ENTRIES[@]}"; do
  name="${e#* }"
  remote=$(kx md5sum "$SRC/$name"); remote="${remote%% *}"
  local_md5=$(md5sum "$DST/$name"); local_md5="${local_md5%% *}"
  if [ "$remote" = "$local_md5" ]; then
    echo "[md5 ok] $name $local_md5"
  else
    echo "[md5 MISMATCH] $name local=$local_md5 remote=$remote" >&2; fail=1
  fi
done
[ "$fail" = 0 ] || { echo "[error] checksum verification failed" >&2; exit 1; }
echo "[done] $DST"

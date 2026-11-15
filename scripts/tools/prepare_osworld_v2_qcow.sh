#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "usage: $0 /absolute/output/directory" >&2
  exit 2
fi

for command in curl unzip sha256sum qemu-img qemu-nbd partprobe e2fsck flock; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

output_dir=$1
mkdir -p "$output_dir"
archive="$output_dir/osworld-v2-ubuntu-x86.qcow2.zip"
archive_part="$archive.part"
source_qcow="$output_dir/osworld-v2-ubuntu-x86-source.qcow2"
source_part="$source_qcow.part"
repaired_qcow="$output_dir/osworld-v2-ubuntu-x86-repaired.qcow2"

archive_url='https://huggingface.co/datasets/xlangai/v2-image/resolve/main/osworld-v2-ubuntu-x86.qcow2.zip'
archive_sha='14b08aa7ba6c023ecb91d46de8df5de32af4d1d6bd75ea925519caf9677fc8b3'
source_sha='28b617987f3edf14edd835069cdde6a6e4708be8e4ae4eb957200d5da335186a'
canonical_repaired_sha='b4674bda74e70d485578bd794f816a8d1f846be44d755fdb9c4aa04acb4814de'

if [[ -e "$repaired_qcow" ]]; then
  qemu-img check "$repaired_qcow"
  actual_repaired_sha=$(sha256sum "$repaired_qcow" | cut -d' ' -f1)
  if [[ -n "${CUAWRIGHT_REQUIRE_REPAIRED_SHA256:-}" ]]; then
    printf '%s  %s\n' "$CUAWRIGHT_REQUIRE_REPAIRED_SHA256" "$repaired_qcow" \
      | sha256sum --check
  fi
  printf 'repaired_sha256=%s\n' "$actual_repaired_sha" >&2
  echo "$repaired_qcow"
  exit 0
fi
if [[ -e "$source_qcow" || -e "$source_part" ]]; then
  echo "source/staging QCOW already exists; refusing an ambiguous repair" >&2
  exit 1
fi

curl -L --fail --retry 50 --retry-delay 10 --retry-all-errors --continue-at - \
  --output "$archive_part" "$archive_url"
printf '%s  %s\n' "$archive_sha" "$archive_part" | sha256sum --check
mv "$archive_part" "$archive"
unzip -t "$archive"
unzip -p "$archive" osworld-v2-ubuntu-x86.qcow2 >"$source_part"
mv "$source_part" "$source_qcow"
printf '%s  %s\n' "$source_sha" "$source_qcow" | sha256sum --check
qemu-img check "$source_qcow"

cp --reflink=auto --sparse=always "$source_qcow" "$repaired_qcow"

exec 9>"$output_dir/nbd-repair.lock"
flock 9
sudo -n modprobe nbd max_part=16
device=
for index in $(seq 15 -1 0); do
  candidate="/dev/nbd$index"
  if [[ -b "$candidate" ]] \
    && [[ -z "$(cat "/sys/block/nbd$index/pid" 2>/dev/null || true)" ]] \
    && ! lsblk -n -o MOUNTPOINTS "$candidate" | grep -q .; then
    device=$candidate
    break
  fi
done
if [[ -z "$device" ]]; then
  echo "no unused NBD device is available" >&2
  exit 1
fi

cleanup() {
  sudo -n qemu-nbd --disconnect "$device" >/dev/null 2>&1 || true
}
trap cleanup EXIT
sudo -n qemu-nbd --format=qcow2 --connect="$device" "$repaired_qcow"
sudo -n udevadm settle
sudo -n partprobe "$device"
sudo -n udevadm settle
for _ in $(seq 1 30); do
  [[ -b "${device}p3" ]] && break
  sleep 1
done
[[ -b "${device}p3" ]]
! findmnt -rn -S "$device" | grep -q .
! findmnt -rn -S "${device}p3" | grep -q .
sudo -n e2fsck -f -y "${device}p3" || [[ $? -eq 1 ]]
sudo -n e2fsck -fn "${device}p3"
sudo -n qemu-nbd --disconnect "$device"
trap - EXIT
sudo -n udevadm settle

qemu-img check "$repaired_qcow"
printf '%s  %s\n' "$source_sha" "$source_qcow" | sha256sum --check
actual_repaired_sha=$(sha256sum "$repaired_qcow" | cut -d' ' -f1)
if [[ -n "${CUAWRIGHT_REQUIRE_REPAIRED_SHA256:-}" ]]; then
  printf '%s  %s\n' "$CUAWRIGHT_REQUIRE_REPAIRED_SHA256" "$repaired_qcow" \
    | sha256sum --check
fi
printf 'repaired_sha256=%s\n' "$actual_repaired_sha" >&2
if [[ "$actual_repaired_sha" == "$canonical_repaired_sha" ]]; then
  echo "The fresh repair happens to match the historical canonical image." >&2
else
  echo "Fresh-repair image: do not label it the historical canonical image." >&2
fi
echo "$repaired_qcow"

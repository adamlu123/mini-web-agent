#!/usr/bin/env bash
# Shared helpers for OM2W cluster submission scripts.
#
# This file is sourced by launchers. Keep job-specific defaults and submission
# arguments in the launcher so each cluster job remains easy to audit.

mwa_cluster_repo_root() {
    local launcher_dir="$1"
    (
        cd "$launcher_dir/../../../.."
        pwd
    )
}

mwa_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

mwa_json_array_length() {
    python - "$1" <<'PY'
import json
import sys

print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
PY
}

mwa_stage_cluster_repo() {
    local repository_root="$1"
    local upload_dir="$2"
    local -a root_paths=(
        src
        agent_runtime
        om2w_judge
        pyproject.toml
        README.md
    )
    local -a script_paths=(
        scripts/cluster
        scripts/eval
        scripts/lib
    )

    mkdir -p "$upload_dir"
    tar \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='*.pyo' \
        -C "$repository_root" \
        -cf - \
        "${root_paths[@]}" \
        "${script_paths[@]}" |
        tar -C "$upload_dir" -xf -
}

mwa_apply_credentials_secret() {
    local namespace="$1"
    local secret_name="$2"
    local credentials_file="$3"

    kubectl -n "$namespace" create secret generic "$secret_name" \
        --from-file=cred.sh="$credentials_file" \
        --dry-run=client -o yaml |
        kubectl -n "$namespace" apply -f -
}

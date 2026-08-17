#!/bin/bash
# Usage: ./job.sh [run.py options]
# Detached: ./job.sh --detach [run.py options]
set -euo pipefail

worker_flag="__lab3_hpc_worker__"

if [[ "${1:-}" == "$worker_flag" ]]; then
    shift
    if [[ ! -f run.py ]]; then
        echo "error: run.py not found in compute working directory: $PWD" >&2
        exit 2
    fi
    if [[ ! -x /opt/lab3-venv/bin/python ]]; then
        echo "error: lab3 Python environment is unavailable" >&2
        exit 127
    fi
    export PYTHONUNBUFFERED=1
    exec /opt/lab3-venv/bin/python run.py "$@"
fi

# The installed hpc CLI does not currently parse #HPC directives from a script
# path. Submit explicitly, then re-enter this file in worker mode on the GPU.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$repo_dir/output"
cd "$repo_dir"

hpc_args=(
    submit
    --partition lab3
    --cpu 8
    --gpu 1
    --mem 32Gi
    --time 5m
    --name lab3-gdn-prefill
    --output "output/lab3_%j.log"
    --export NONE
    --chdir .
)

if [[ "${1:-}" == "--detach" ]]; then
    hpc_args+=(--detach)
    shift
fi

# --export NONE keeps the worker reproducible, so forward only the kernel
# tuning variables explicitly set by the caller.
worker_cmd=(env)
gdn_env_names=(
    GDN_IMPL
    GDN_DV_SPLIT
    GDN_RS
    GDN_PREFETCH
    GDN_MEMORY_IO
    GDN_GATE_CP
    GDN_GATE_CP_THRESHOLD
    GDN_GATE_CP_MIN_CHUNKS
)
for env_name in "${gdn_env_names[@]}"; do
    if [[ -v "$env_name" ]]; then
        worker_cmd+=("$env_name=${!env_name}")
    fi
done
worker_cmd+=(./job.sh "$worker_flag" "$@")

exec hpc "${hpc_args[@]}" "${worker_cmd[@]}"

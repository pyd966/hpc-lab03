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

exec hpc "${hpc_args[@]}" ./job.sh "$worker_flag" "$@"

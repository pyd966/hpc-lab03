#!/bin/bash
#HPC --partition=lab3
#HPC --cpu=8
#HPC --gpu=1
#HPC --mem=32Gi
#HPC --time=15m
#HPC --name=lab3-gdn-ncu-full

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 CASE KERNEL_NAME REPORT_BASENAME" >&2
    exit 2
fi

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

case_name=$1
kernel_name=$2
report_basename=$3

ncu \
    --set full \
    --section PmSampling_WarpStates \
    --import-source yes \
    --clock-control none \
    --replay-mode kernel \
    --kernel-name "$kernel_name" \
    --launch-count 1 \
    --force-overwrite \
    -o "$report_basename" \
    python run.py --case "$case_name" --warmup 0 --repetitions 1 \
        --output-format csv

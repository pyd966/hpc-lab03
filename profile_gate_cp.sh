#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

profile_one() {
    local case_name=$1
    local kernel_name=$2
    local basename=$3

    ncu \
        --set full \
        --section PmSampling_WarpStates \
        --import-source yes \
        --clock-control none \
        --replay-mode kernel \
        --kernel-name "$kernel_name" \
        --launch-count 1 \
        --force-overwrite \
        -o "$basename" \
        /opt/lab3-venv/bin/python run.py \
            --case "$case_name" --warmup 0 --repetitions 1 \
            --output-format csv

    ncu --import "$basename.ncu-rep" --page details \
        > "$basename.details.txt"
    ncu --import "$basename.ncu-rep" --page raw --csv \
        > "$basename.raw.csv"
    ncu --import "$basename.ncu-rep" --page source --csv \
        > "$basename.source.csv"
}

if [[ $# -ne 3 ]]; then
    echo "usage: $0 CASE KERNEL_NAME REPORT_BASENAME" >&2
    exit 2
fi

profile_one "$1" "$2" "$3"

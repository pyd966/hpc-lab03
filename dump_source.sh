#!/bin/bash
#HPC --partition=lab3
#HPC --cpu=8
#HPC --gpu=1
#HPC --mem=32Gi
#HPC --time=5m
#HPC --name=lab3-gdn-source

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

python dump_kernel_source.py "$@"

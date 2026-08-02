#!/bin/bash
#HPC --partition=lab3
#HPC --cpu=8
#HPC --gpu=1
#HPC --mem=32Gi
#HPC --time=5m
#HPC --name=lab3-gdn-prefill
#HPC --output=output/lab3_%j.log

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

python run.py "$@"

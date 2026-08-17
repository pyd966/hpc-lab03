#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

profile_chain() {
    ./profile_gate_cp.sh \
        chain_equal \
        get_gate_cp_warmup_sp4_kernel \
        output/ncu_gate_policy_chain_warmup_full
    ./profile_gate_cp.sh \
        chain_equal \
        prepare_gate_cp_dv64x2_sp4_kernel \
        output/ncu_gate_policy_chain_prepare_full
    ./profile_gate_cp.sh \
        chain_equal \
        residual_mha_dv64x2_rs_io_qkva_reuse_so_cp4_kernel \
        output/ncu_gate_policy_chain_main_full
}

profile_d128_case() {
    local case_name=$1
    local report_tag=$2
    ./profile_gate_cp.sh \
        "$case_name" \
        get_gate_cp_warmup_sp2_kernel \
        "output/ncu_gate_policy_${report_tag}_warmup_full"
    ./profile_gate_cp.sh \
        "$case_name" \
        prepare_gate_cp_dv128x1_sp2_kernel \
        "output/ncu_gate_policy_${report_tag}_prepare_full"
    ./profile_gate_cp.sh \
        "$case_name" \
        residual_gva_dv128x1_rs_io_qkva_cp2_kernel \
        "output/ncu_gate_policy_${report_tag}_main_full"
}

case "${1:-}" in
    chain)
        profile_chain
        ;;
    batch)
        profile_d128_case batch_split_gva batch
        ;;
    deep)
        profile_d128_case deep_gva_state deep
        ;;
    *)
        echo "usage: $0 chain|batch|deep" >&2
        exit 2
        ;;
esac

#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

profile_chain() {
    ./profile_gate_cp.sh \
        chain_equal \
        get_gate_cp_warmup_sp4_kernel \
        output/ncu_gate_budget_chain_warmup_full
    ./profile_gate_cp.sh \
        chain_equal \
        prepare_gate_cp_dv64x2_sp4_kernel \
        output/ncu_gate_budget_chain_prepare_full
    ./profile_gate_cp.sh \
        chain_equal \
        residual_mha_dv64x2_rs_io_qkva_reuse_so_cp4_kernel \
        output/ncu_gate_budget_chain_main_full
}

profile_long() {
    ./profile_gate_cp.sh \
        long_low_gva \
        get_gate_cp_warmup_sp8_kernel \
        output/ncu_gate_budget_long_warmup_full
    ./profile_gate_cp.sh \
        long_low_gva \
        prepare_gate_cp_dv128x1_sp8_kernel \
        output/ncu_gate_budget_long_prepare_full
    ./profile_gate_cp.sh \
        long_low_gva \
        residual_gva_dv128x1_rs_io_qkva_cp8_kernel \
        output/ncu_gate_budget_long_main_full
}

profile_batch() {
    ./profile_gate_cp.sh \
        batch_split_gva \
        residual_gva_dv128x1_rs_io_qkva_kernel \
        output/ncu_gate_budget_batch_main_full
}

case "${1:-}" in
    chain)
        profile_chain
        ;;
    long)
        profile_long
        ;;
    batch)
        profile_batch
        ;;
    *)
        echo "usage: $0 chain|long|batch" >&2
        exit 2
        ;;
esac

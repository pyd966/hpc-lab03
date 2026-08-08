# Residual-first baseline: Z = A @ (beta * (V - exp(g) * K @ S)).
import os

import torch
import tilelang
import tilelang.language as T

from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)
from student.grouped_wgmma import GROUPED_WGMMA_PRELUDE
from student.tilelang_fwd_document import gdn_prefill_forward_document


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
MIG_SM_COUNT = 14
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K**-0.5
USE_DOCUMENT_FORM = os.environ.get("GDN_IMPL", "residual") == "document"
DV_SPLIT_MODE = os.environ.get("GDN_DV_SPLIT", "32")
DV_SPLIT_CONFIGS = {
    "off": (HEAD_DIM_V, 1),
    "64": (64, 2),
    "32": (32, 4),
}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_residual_first(
    H,
    Hg,
    qk_dtype,
    v_dtype,
    gate_dtype,
    accum_dtype,
    use_initial_state,
    low_parallel,
    gva_variant,
    dv_tile,
    dv_parts,
    prefetch_a,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    qk_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    initial_shape = state_shape if use_initial_state else (1,)
    parallel_name = "low_parallel" if low_parallel else "normal"
    attention_name = "gva" if gva_variant else "mha"
    kernel_name = (
        "residual_"
        + parallel_name
        + "_"
        + attention_name
        + "_dv"
        + str(dv_tile)
        + "x"
        + str(dv_parts)
    )
    a_pipeline_stage = 0 if prefetch_a else 1
    dv32_accum_layout = T.Fragment(
        [CHUNK_SIZE, 32],
        forward_thread_fn=lambda i, j: (
            j // 16 * 128 + i // 16 * 32 + i % 8 * 4 + j % 8 // 2
        ),
        forward_index_fn=lambda i, j: (
            j % 16 // 8 * 4 + i % 16 // 8 * 2 + j % 2
        ),
    )
    if dv_tile == 32:
        pipeline_order = [
            3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
        ]
        pipeline_stage = [
            1, 0, 0, 0, 1, 1, a_pipeline_stage, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]
    else:
        pipeline_order = [
            3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30,
        ]
        pipeline_stage = [
            1, 0, 0, 0, 1, 1, a_pipeline_stage, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ]

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        a: T.Tensor(a_shape, dtype=qk_dtype),
        initial_state: T.Tensor(initial_shape, dtype=accum_dtype),
        output: T.Tensor(v_shape, dtype=v_dtype),
        final_state: T.Tensor(state_shape, dtype=accum_dtype),
        chunks_per_batch: T.int32,
    ):
        T.func_attr({"global_symbol": kernel_name})
        # Value columns are independent recurrent chains. One block owns one
        # [128, dv_tile] state slice and traverses all chunks serially.
        with T.Kernel(
            batch_size * H * dv_parts,
            threads=256,
            prelude=GROUPED_WGMMA_PRELUDE,
        ) as (block,):
            owner = block // dv_parts
            dv_part = block % dv_parts
            bb = owner // H
            bh = owner % H
            bhg = bh // (H // Hg)
            dv_left = dv_part * dv_tile

            # Shared tiles used by the recurrent chunk schedule.
            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            v_shared = T.alloc_shared((CHUNK_SIZE, dv_tile), dtype=v_dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            z_shared = T.alloc_shared((CHUNK_SIZE, dv_tile), dtype=v_dtype)
            z_state_shared = T.alloc_shared(
                (CHUNK_SIZE, dv_tile), dtype=v_dtype
            )
            state_shared = T.alloc_shared(
                (HEAD_DIM_K, dv_tile), dtype=v_dtype
            )
            score_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype
            )
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            g_exp_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            g_inv_exp_shared = T.alloc_shared((CHUNK_SIZE,), dtype=accum_dtype)
            g_last_exp = T.alloc_shared((1,), dtype=accum_dtype)

            state = T.alloc_fragment(
                (HEAD_DIM_K, dv_tile), dtype=accum_dtype
            )
            z = T.alloc_fragment((CHUNK_SIZE, dv_tile), dtype=accum_dtype)
            out = T.alloc_fragment((CHUNK_SIZE, dv_tile), dtype=accum_dtype)
            score = T.alloc_fragment(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype
            )
            if dv_tile == 32:
                T.annotate_layout(
                    {
                        q_shared: make_full_bank_swizzled_layout(q_shared),
                        k_shared: make_full_bank_swizzled_layout(k_shared),
                        state_shared: make_quarter_bank_swizzled_layout(
                            state_shared
                        ),
                        z: dv32_accum_layout,
                        out: dv32_accum_layout,
                    }
                )
            if use_initial_state:
                T.copy(
                    initial_state[
                        bb,
                        bh,
                        0:HEAD_DIM_K,
                        dv_left : dv_left + dv_tile,
                    ],
                    state,
                )
            else:
                T.clear(state)

            # Q and V have only one shared-memory buffer. Their next chunks are
            # loaded after the current chunk reaches its final consumer.
            T.copy(
                q[bb, 0:CHUNK_SIZE, bhg, 0:HEAD_DIM_K],
                q_shared,
            )
            T.copy(
                v[
                    bb,
                    0:CHUNK_SIZE,
                    bh,
                    dv_left : dv_left + dv_tile,
                ],
                v_shared,
            )

            # K remains loop-pipelined. Q and V use explicit single-buffer
            # prefetches after their final current-chunk consumers.
            for chunk in T.Pipelined(
                chunks_per_batch,
                order=pipeline_order,
                stage=pipeline_stage,
            ):
                left = chunk * CHUNK_SIZE
                right = left + CHUNK_SIZE

                T.copy(state, state_shared)
                T.copy(
                    k[bb, left : left + CHUNK_SIZE, bhg, 0:HEAD_DIM_K],
                    k_shared,
                )
                T.copy(g[bb, left : left + CHUNK_SIZE, bh], g_shared)
                T.copy(beta[bb, left : left + CHUNK_SIZE, bh], beta_shared)

                if right <= num_tokens:
                    g_last_exp[0] = T.exp2(
                        g_shared[CHUNK_SIZE - 1] * LOG2E
                    )
                else:
                    g_last_exp[0] = T.exp2(
                        g[bb, num_tokens - 1, bh] * LOG2E
                    )

                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, dv_tile):
                    state[dim_k, dim_v] *= g_last_exp[0]

                # A is independent of G0/G1/G2 and is available when the
                # residual consumes z immediately after the combined group.
                T.copy(
                    a[bb, left : left + CHUNK_SIZE, bh, 0:CHUNK_SIZE],
                    a_shared,
                )

                # G0 commits first, performs all independent gate work, and
                # waits only at the boundary where z becomes the next operand.
                if dv_tile == 32:
                    T.call_extern(
                        "handle",
                        "student_wgmma_g0_gamma",
                        T.access_ptr(k_shared, "r"),
                        T.access_ptr(state_shared, "r"),
                        T.access_ptr(z, "rw"),
                        T.access_ptr(g_shared, "r"),
                        T.access_ptr(g_exp_shared, "w"),
                        T.access_ptr(g_inv_exp_shared, "w"),
                    )
                else:
                    T.wgmma_gemm(
                        k_shared,
                        state_shared,
                        z,
                        clear_accum=True,
                    )
                    for token in T.Parallel(CHUNK_SIZE):
                        g_exp_shared[token] = T.exp2(
                            g_shared[token] * LOG2E
                        )
                    for token in T.Parallel(CHUNK_SIZE):
                        g_inv_exp_shared[token] = (
                            1.0 / g_exp_shared[token]
                        )
                    T.wait_wgmma(0)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    if left + token < num_tokens:
                        z[token, dim] = beta_shared[token] * (
                            v_shared[token, dim]
                            - g_exp_shared[token] * z[token, dim]
                        )
                    else:
                        z[token, dim] = 0

                # G1/G2 start only after z has consumed G0. They remain in
                # flight together with G3 until the wait_group(1) below.
                T.wgmma_gemm(
                    q_shared,
                    state_shared,
                    out,
                    clear_accum=True,
                )
                T.wgmma_gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                )
                if chunk + 1 < chunks_per_batch:
                    T.copy(
                        v[
                            bb,
                            right : right + CHUNK_SIZE,
                            bh,
                            dv_left : dv_left + dv_tile,
                        ],
                        v_shared,
                    )
                T.copy(z, z_shared)
                T.wgmma_gemm(
                    a_shared,
                    z_shared,
                    z,
                    clear_accum=True,
                )

                # G1 and G2 must be complete before their accumulators are
                # transformed. G3 may continue while this work executes.
                T.wait_wgmma(1)
                if chunk + 1 < chunks_per_batch:
                    T.copy(
                        q[
                            bb,
                            right : right + CHUNK_SIZE,
                            bhg,
                            0:HEAD_DIM_K,
                        ],
                        q_shared,
                    )
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    out[token, dim] *= SCALE * g_exp_shared[token]

                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if row >= col and left + row < num_tokens:
                        score[row, col] *= (
                            SCALE
                            * g_exp_shared[row]
                            * g_inv_exp_shared[col]
                        )
                    else:
                        score[row, col] = 0
                T.copy(score, score_shared)

                # G3 is first read here. Waiting also makes it safe to
                # overwrite the residual tile that G3 consumed.
                T.wait_wgmma(0)
                T.copy(z, z_shared)

                # Preserve raw Z for output and separately materialize its
                # end-of-chunk-decayed form for the recurrent state update.
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    z[token, dim] *= (
                        g_last_exp[0] * g_inv_exp_shared[token]
                    )
                T.copy(z, z_state_shared)

                T.wgmma_gemm(
                    score_shared,
                    z_shared,
                    out,
                    clear_accum=False,
                )
                T.wgmma_gemm(
                    k_shared,
                    z_state_shared,
                    state,
                    transpose_A=True,
                    clear_accum=False,
                )

                # Retire the output update but leave the state update in flight
                # while the completed output is stored to global memory.
                T.wait_wgmma(1)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    if left + token < num_tokens:
                        output[
                            bb, left + token, bh, dv_left + dim
                        ] = out[token, dim]
                T.wait_wgmma(0)

            T.copy(
                state,
                final_state[
                    bb,
                    bh,
                    0:HEAD_DIM_K,
                    dv_left : dv_left + dv_tile,
                ],
            )

    return kernel


def tilelang_residual_first_low_parallel(*args, **kwargs):
    return tilelang_residual_first(*args, low_parallel=True, **kwargs)


def tilelang_residual_first_normal(*args, **kwargs):
    return tilelang_residual_first(*args, low_parallel=False, **kwargs)


# q/k: [B, T, Hq, 128] BF16
# v: [B, T, Hv, 128] BF16
# g_cumsum/beta: [B, T, Hv] FP32
# A: [B, T, Hv, 64] BF16
# initial_state/final_state: [B, Hv, 128, 128] FP32
def gdn_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if USE_DOCUMENT_FORM:
        return gdn_prefill_forward_document(
            q,
            k,
            v,
            g_cumsum,
            beta,
            A,
            initial_state,
        )

    batch_size, num_tokens, num_heads_qk, _ = q.shape
    num_heads_v = v.shape[2]
    chunks_per_batch = tilelang.cdiv(num_tokens, CHUNK_SIZE)

    output = torch.empty_like(v)
    final_state = torch.empty(
        (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
        dtype=torch.float32,
        device=v.device,
    )

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((1,), dtype=torch.float32, device=v.device)
    low_parallel = batch_size * num_heads_v < MIG_SM_COUNT
    gva_variant = num_heads_v != num_heads_qk
    split_requested = DV_SPLIT_MODE in DV_SPLIT_CONFIGS
    candidate_dv_tile, candidate_dv_parts = DV_SPLIT_CONFIGS.get(
        DV_SPLIT_MODE, (HEAD_DIM_V, 1)
    )
    split_profitable = (
        low_parallel
        and chunks_per_batch >= 64
        and batch_size * num_heads_v * candidate_dv_parts
        <= 2 * MIG_SM_COUNT
    )
    if split_requested and split_profitable:
        dv_tile, dv_parts = candidate_dv_tile, candidate_dv_parts
    else:
        dv_tile, dv_parts = HEAD_DIM_V, 1
    kernel_factory = (
        tilelang_residual_first_low_parallel
        if low_parallel
        else tilelang_residual_first_normal
    )
    recurrent = kernel_factory(
        num_heads_v,
        num_heads_qk,
        qk_dtype=q.dtype,
        v_dtype=v.dtype,
        gate_dtype=g_cumsum.dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        gva_variant=gva_variant,
        dv_tile=dv_tile,
        dv_parts=dv_parts,
        prefetch_a=dv_parts == 1,
    )
    recurrent(
        q,
        k,
        v,
        g_cumsum,
        beta,
        A,
        initial_state,
        output,
        final_state,
        chunks_per_batch,
    )
    return output, final_state

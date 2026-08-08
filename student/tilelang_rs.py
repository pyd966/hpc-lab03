import tilelang
import tilelang.language as T

from tilelang.intrinsics.wgmma_macro_generator import (
    TensorCoreIntrinEmitter as WGMMAEmitter,
)
from tilelang.layout import make_full_bank_swizzled_layout


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K**-0.5


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_residual_first_full_chunks_rs(
    H,
    Hg,
    qk_dtype,
    v_dtype,
    gate_dtype,
    accum_dtype,
    use_initial_state,
    dv_tile,
    dv_parts,
    prefetch_q,
    prefetch_k,
    prefetch_v,
    prefetch_a,
):
    """Full-chunk path with transposed recurrent fragments and RS WGMMA."""
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    qk_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    initial_shape = state_shape if use_initial_state else (1,)
    q_stage = 0 if prefetch_q else 1
    k_stage = 0 if prefetch_k else 1
    v_stage = 0 if prefetch_v else 1
    a_stage = 0 if prefetch_a else 1
    prefetch_inputs = prefetch_q or prefetch_k or prefetch_v or prefetch_a
    gate_stage = 0 if prefetch_inputs else 1
    # Explicit WGMMA expands before PipelinePlanning; only copy blocks remain.
    pipeline_order = [5, 4, 0, 1, 2, 3, 6, 7]
    pipeline_stage = [
        q_stage,
        k_stage,
        v_stage,
        a_stage,
        gate_stage,
        gate_stage,
        1,
        1,
    ]
    attention_name = "gva" if H != Hg else "mha"
    prefetch_tag = (
        ("q" if prefetch_q else "")
        + ("k" if prefetch_k else "")
        + ("v" if prefetch_v else "")
        + ("a" if prefetch_a else "")
    )
    if not prefetch_tag:
        prefetch_tag = "none"
    kernel_name = (
        "residual_"
        + attention_name
        + "_dv"
        + str(dv_tile)
        + "x"
        + str(dv_parts)
        + "_rs_io_"
        + prefetch_tag
    )

    # Hopper WGMMA always covers 64 rows. Dispatch keeps D=32 on the SS path.
    value_threads = 128 if dv_tile == 64 else 256
    value_row_warps = value_threads // 32
    score_col_warps = value_threads // 128
    project_rs = WGMMAEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float32",
        b_transposed=True,
        block_row_warps=value_row_warps,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=64,
        chunk=HEAD_DIM_K,
    )
    correction_rs = WGMMAEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float32",
        b_transposed=True,
        block_row_warps=value_row_warps,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=64,
        chunk=CHUNK_SIZE,
    )
    update_rs = WGMMAEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float32",
        block_row_warps=value_row_warps,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=128,
        chunk=CHUNK_SIZE,
    )
    score_ss = WGMMAEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float32",
        b_transposed=True,
        block_row_warps=4,
        block_col_warps=score_col_warps,
        warp_row_tiles=16,
        warp_col_tiles=64 // score_col_warps,
        chunk=HEAD_DIM_K,
    )

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
        with T.Kernel(batch_size * H * dv_parts, threads=value_threads) as (block,):
            owner = block // dv_parts
            dv_part = block % dv_parts
            bb = owner // H
            bh = owner % H
            bhg = bh // (H // Hg)
            dv_left = dv_part * dv_tile

            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            v_shared = T.alloc_shared((CHUNK_SIZE, dv_tile), dtype=v_dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            output_shared = T.alloc_shared(
                (CHUNK_SIZE, dv_tile), dtype=v_dtype
            )
            score_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype
            )
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            gamma_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            inv_gamma_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            gamma_last = T.alloc_shared((1,), dtype=gate_dtype)

            state_t = T.alloc_fragment(
                (dv_tile, HEAD_DIM_K), dtype=accum_dtype
            )
            state_operand = T.alloc_fragment(
                (dv_tile, HEAD_DIM_K), dtype=v_dtype
            )
            z_operand = T.alloc_fragment(
                (dv_tile, CHUNK_SIZE), dtype=v_dtype
            )
            z_t = T.alloc_fragment(
                (dv_tile, CHUNK_SIZE), dtype=accum_dtype
            )
            out_t = T.alloc_fragment(
                (dv_tile, CHUNK_SIZE), dtype=accum_dtype
            )
            score = T.alloc_fragment(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype
            )

            q_layout = make_full_bank_swizzled_layout(q_shared)
            k_layout = make_full_bank_swizzled_layout(k_shared)
            a_layout = make_full_bank_swizzled_layout(a_shared)
            score_layout = make_full_bank_swizzled_layout(score_shared)
            T.annotate_layout(
                {
                    q_shared: q_layout,
                    k_shared: k_layout,
                    a_shared: a_layout,
                    score_shared: score_layout,
                    state_operand: project_rs.make_mma_load_layout(state_operand),
                    z_operand: correction_rs.make_mma_load_layout(z_operand),
                    z_t: project_rs.make_mma_store_layout(z_t),
                    out_t: project_rs.make_mma_store_layout(out_t),
                    state_t: update_rs.make_mma_store_layout(state_t),
                    score: score_ss.make_mma_store_layout(score),
                }
            )
            project_rs._assign_b_shared_layout(k_layout)
            correction_rs._assign_b_shared_layout(a_layout)
            update_rs._assign_b_shared_layout(k_layout)
            score_ss._assign_a_shared_layout(q_layout)
            score_ss._assign_b_shared_layout(k_layout)

            T.clear(state_t)
            if use_initial_state:
                for dim_v, dim_k in T.Parallel(dv_tile, HEAD_DIM_K):
                    state_t[dim_v, dim_k] = initial_state[
                        bb, bh, dim_k, dv_left + dim_v
                    ]

            for chunk in T.Pipelined(
                chunks_per_batch,
                order=pipeline_order,
                stage=pipeline_stage,
            ):
                left = chunk * CHUNK_SIZE
                right = left + CHUNK_SIZE

                T.copy(q[bb, left:right, bhg, 0:HEAD_DIM_K], q_shared)
                T.copy(k[bb, left:right, bhg, 0:HEAD_DIM_K], k_shared)
                T.copy(
                    v[
                        bb,
                        left:right,
                        bh,
                        dv_left : dv_left + dv_tile,
                    ],
                    v_shared,
                )
                T.copy(a[bb, left:right, bh, 0:CHUNK_SIZE], a_shared)
                T.copy(g[bb, left:right, bh], g_shared)
                T.copy(beta[bb, left:right, bh], beta_shared)

                T.copy(state_t, state_operand)
                project_rs.wgmma(
                    state_operand[0:dv_tile, 0:HEAD_DIM_K],
                    k_shared[0:CHUNK_SIZE, 0:HEAD_DIM_K],
                    z_t[0:dv_tile, 0:CHUNK_SIZE],
                    clear_accum=True,
                    wg_wait=-1,
                )
                project_rs.wgmma(
                    state_operand[0:dv_tile, 0:HEAD_DIM_K],
                    q_shared[0:CHUNK_SIZE, 0:HEAD_DIM_K],
                    out_t[0:dv_tile, 0:CHUNK_SIZE],
                    clear_accum=True,
                    wg_wait=-1,
                )
                for token in T.Parallel(CHUNK_SIZE):
                    gamma_shared[token] = T.exp2(
                        g_shared[token] * LOG2E
                    )
                    inv_gamma_shared[token] = 1.0 / gamma_shared[token]
                gamma_last[0] = gamma_shared[CHUNK_SIZE - 1]
                for dim_v, dim_k in T.Parallel(dv_tile, HEAD_DIM_K):
                    state_t[dim_v, dim_k] *= gamma_last[0]

                # Retire K@S but keep Q@S in flight while residual work starts.
                T.warpgroup_wait(1)
                for dim_v, token in T.Parallel(dv_tile, CHUNK_SIZE):
                    z_t[dim_v, token] = beta_shared[token] * (
                        v_shared[token, dim_v]
                        - gamma_shared[token] * z_t[dim_v, token]
                    )
                T.copy(z_t, z_operand)
                correction_rs.wgmma(
                    z_operand[0:dv_tile, 0:CHUNK_SIZE],
                    a_shared[0:CHUNK_SIZE, 0:CHUNK_SIZE],
                    z_t[0:dv_tile, 0:CHUNK_SIZE],
                    clear_accum=True,
                    wg_wait=-1,
                )
                score_ss.wgmma(
                    q_shared[0:CHUNK_SIZE, 0:HEAD_DIM_K],
                    k_shared[0:CHUNK_SIZE, 0:HEAD_DIM_K],
                    score[0:CHUNK_SIZE, 0:CHUNK_SIZE],
                    clear_accum=True,
                    wg_wait=-1,
                )

                # Q@S and A@residual retire; QK remains outstanding.
                T.warpgroup_wait(1)
                for dim_v, token in T.Parallel(dv_tile, CHUNK_SIZE):
                    z_t[dim_v, token] *= (
                        gamma_last[0] * inv_gamma_shared[token]
                    )
                T.copy(z_t, z_operand)

                T.warpgroup_wait(0)
                for dim_v, token in T.Parallel(dv_tile, CHUNK_SIZE):
                    out_t[dim_v, token] *= (
                        SCALE * gamma_shared[token]
                    )
                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if row >= col:
                        score[row, col] *= (
                            SCALE
                            * gamma_shared[row]
                            * inv_gamma_shared[CHUNK_SIZE - 1]
                        )
                    else:
                        score[row, col] = 0
                T.copy(score, score_shared)

                correction_rs.wgmma(
                    z_operand[0:dv_tile, 0:CHUNK_SIZE],
                    score_shared[0:CHUNK_SIZE, 0:CHUNK_SIZE],
                    out_t[0:dv_tile, 0:CHUNK_SIZE],
                    clear_accum=False,
                    wg_wait=-1,
                )
                update_rs.wgmma(
                    z_operand[0:dv_tile, 0:CHUNK_SIZE],
                    k_shared[0:CHUNK_SIZE, 0:HEAD_DIM_K],
                    state_t[0:dv_tile, 0:HEAD_DIM_K],
                    clear_accum=False,
                    wg_wait=-1,
                )
                T.warpgroup_wait(1)
                for dim_v, token in T.Parallel(dv_tile, CHUNK_SIZE):
                    output_shared[token, dim_v] = out_t[dim_v, token]
                T.copy(
                    output_shared,
                    output[
                        bb,
                        left:right,
                        bh,
                        dv_left : dv_left + dv_tile,
                    ],
                )
                T.warpgroup_wait(0)

            for dim_v, dim_k in T.Parallel(dv_tile, HEAD_DIM_K):
                final_state[bb, bh, dim_k, dv_left + dim_v] = state_t[
                    dim_v, dim_k
                ]

    return kernel


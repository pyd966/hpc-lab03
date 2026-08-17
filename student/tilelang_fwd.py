# Residual-first baseline: Z = A @ (beta * (V - exp(g) * K @ S)).
import math
import os

import torch
import tilelang
import tilelang.language as T

from student.tilelang_fwd_document import gdn_prefill_forward_document
from student.tilelang_rs import (
    tilelang_get_gate_cp_warmup,
    tilelang_prepare_gate_cp_states,
    tilelang_residual_first_full_chunks_rs,
)


CHUNK_SIZE = 64
HEAD_DIM_K = 128
HEAD_DIM_V = 128
MIG_SM_COUNT = 14
LOG2E = 1.4426950408889634
SCALE = HEAD_DIM_K**-0.5
USE_DOCUMENT_FORM = os.environ.get("GDN_IMPL", "residual") == "document"
DV_SPLIT_MODE = os.environ.get("GDN_DV_SPLIT", "auto")
RS_MODE = os.environ.get("GDN_RS", "auto")
PREFETCH_MODE = os.environ.get("GDN_PREFETCH", "auto")
MEMORY_IO_MODE = os.environ.get("GDN_MEMORY_IO", "auto")
GATE_CP_MODE = os.environ.get("GDN_GATE_CP", "auto")
GATE_CP_THRESHOLD = float(os.environ.get("GDN_GATE_CP_THRESHOLD", "-10.0"))
GATE_CP_MIN_CHUNKS = int(os.environ.get("GDN_GATE_CP_MIN_CHUNKS", "128"))
DV_SPLIT_CONFIGS = {
    "off": (HEAD_DIM_V, 1),
    "64": (64, 2),
    "32": (32, 4),
}
PREFETCH_INPUTS = {
    "off": (False, False, False, False),
    "qkv": (True, True, True, False),
    "qkva": (True, True, True, True),
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
    dv_tile,
    dv_parts,
    prefetch_k,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    qk_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    initial_shape = state_shape if use_initial_state else (1,)
    k_pipeline_stage = 0 if prefetch_k else 1
    attention_name = "gva" if H != Hg else "mha"
    kernel_name = (
        "residual_"
        + attention_name
        + "_dv"
        + str(dv_tile)
        + "x"
        + str(dv_parts)
        + ("_pfk" if prefetch_k else "_base")
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
        # Value columns are independent; each block owns one state slice.
        with T.Kernel(batch_size * H * dv_parts, threads=256) as (block,):
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
            z_shared = T.alloc_shared((CHUNK_SIZE, dv_tile), dtype=v_dtype)
            state_shared = T.alloc_shared(
                (HEAD_DIM_K, dv_tile), dtype=v_dtype
            )
            score_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype
            )
            gamma_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            inv_gamma_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            gamma_last = T.alloc_shared((1,), dtype=gate_dtype)

            state = T.alloc_fragment(
                (HEAD_DIM_K, dv_tile), dtype=accum_dtype
            )
            z = T.alloc_fragment((CHUNK_SIZE, dv_tile), dtype=accum_dtype)
            out = T.alloc_fragment((CHUNK_SIZE, dv_tile), dtype=accum_dtype)
            score = T.alloc_fragment(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype
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

            # Statement 1 is the K copy: only it crosses iterations. The
            # recurrent state work remains in stage 1.
            for chunk in T.Pipelined(
                chunks_per_batch,
                order=[
                    1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                ],
                stage=[
                    1, k_pipeline_stage, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                ],
            ):
                left = chunk * CHUNK_SIZE
                right = left + CHUNK_SIZE

                T.copy(state, state_shared)
                T.copy(
                    k[bb, left:right, bhg, 0:HEAD_DIM_K], k_shared
                )
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    if left + token < num_tokens:
                        q_shared[token, dim] = q[bb, left + token, bhg, dim]
                        if dim < dv_tile:
                            v_shared[token, dim] = v[
                                bb, left + token, bh, dv_left + dim
                            ]
                    else:
                        q_shared[token, dim] = 0
                        if dim < dv_tile:
                            v_shared[token, dim] = 0
                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if left + row < num_tokens:
                        a_shared[row, col] = a[bb, left + row, bh, col]
                    else:
                        a_shared[row, col] = 0
                for token in T.Parallel(CHUNK_SIZE):
                    if left + token < num_tokens:
                        gamma_shared[token] = T.exp2(
                            g[bb, left + token, bh] * LOG2E
                        )
                        inv_gamma_shared[token] = 1.0 / gamma_shared[token]
                        beta_shared[token] = beta[bb, left + token, bh]
                    else:
                        gamma_shared[token] = 1.0
                        inv_gamma_shared[token] = 1.0
                        beta_shared[token] = 0
                if right <= num_tokens:
                    gamma_last[0] = gamma_shared[CHUNK_SIZE - 1]
                else:
                    gamma_last[0] = T.exp2(
                        g[bb, num_tokens - 1, bh] * LOG2E
                    )

                # Residual-first form:
                #   R = beta * (V - exp(g) * K @ S)
                #   Z = A @ R
                T.gemm(k_shared, state_shared, z, clear_accum=True)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    if left + token < num_tokens:
                        z[token, dim] = beta_shared[token] * (
                            v_shared[token, dim]
                            - gamma_shared[token] * z[token, dim]
                        )
                    else:
                        z[token, dim] = 0
                T.copy(z, z_shared)
                T.gemm(a_shared, z_shared, z, clear_accum=True)
                T.copy(z, z_shared)

                # Contribution from the state at the start of the chunk.
                T.gemm(q_shared, state_shared, out, clear_accum=True)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    out[token, dim] *= SCALE * gamma_shared[token]

                # Causal in-chunk contribution: tril(Q K^T * decay) @ Z.
                T.gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                )
                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if row >= col and left + row < num_tokens:
                        score[row, col] *= (
                            SCALE
                            * gamma_shared[row]
                            * inv_gamma_shared[col]
                        )
                    else:
                        score[row, col] = 0
                T.copy(score, score_shared)
                T.gemm(score_shared, z_shared, out, clear_accum=False)

                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    if left + token < num_tokens:
                        output[
                            bb, left + token, bh, dv_left + dim
                        ] = out[token, dim]

                # S' = gamma_last * S + K^T @ ((gamma_last / gamma_i) * Z_i).
                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, dv_tile):
                    state[dim_k, dim_v] *= gamma_last[0]
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    z[token, dim] *= (
                        gamma_last[0] * inv_gamma_shared[token]
                    )
                T.copy(z, z_shared)
                T.gemm(
                    k_shared,
                    z_shared,
                    state,
                    transpose_A=True,
                    clear_accum=False,
                )

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

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_residual_first_full_chunks(
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
    """Fast path for sequence lengths divisible by CHUNK_SIZE."""
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
    # Issue the strided gate loads first so their latency overlaps the larger,
    # contiguous Q/K/V/A copies before the common async-consumer wait.
    pipeline_order = [6, 2, 3, 4, 5, 0, 1] + list(range(7, 30))
    pipeline_stage = [
        1,
        q_stage,
        k_stage,
        v_stage,
        a_stage,
        gate_stage,
        gate_stage,
    ] + [1] * 23
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
        + "_io_"
        + prefetch_tag
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
        with T.Kernel(batch_size * H * dv_parts, threads=256) as (block,):
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
            z_shared = T.alloc_shared((CHUNK_SIZE, dv_tile), dtype=v_dtype)
            state_shared = T.alloc_shared(
                (HEAD_DIM_K, dv_tile), dtype=v_dtype
            )
            score_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype
            )
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            gamma_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            inv_gamma_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            gamma_last = T.alloc_shared((1,), dtype=gate_dtype)

            state = T.alloc_fragment(
                (HEAD_DIM_K, dv_tile), dtype=accum_dtype
            )
            z = T.alloc_fragment((CHUNK_SIZE, dv_tile), dtype=accum_dtype)
            out = T.alloc_fragment((CHUNK_SIZE, dv_tile), dtype=accum_dtype)
            score = T.alloc_fragment(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype
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

            for chunk in T.Pipelined(
                chunks_per_batch,
                order=pipeline_order,
                stage=pipeline_stage,
            ):
                left = chunk * CHUNK_SIZE
                right = left + CHUNK_SIZE

                T.copy(state, state_shared)
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

                # Keep independent WGMMA groups in flight and wait only at the
                # first operation that consumes each accumulator fragment.
                T.wgmma_gemm(k_shared, state_shared, z, clear_accum=True)
                for token in T.Parallel(CHUNK_SIZE):
                    gamma_shared[token] = T.exp2(
                        g_shared[token] * LOG2E
                    )
                    inv_gamma_shared[token] = 1.0 / gamma_shared[token]
                gamma_last[0] = gamma_shared[CHUNK_SIZE - 1]
                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, dv_tile):
                    state[dim_k, dim_v] *= gamma_last[0]
                # G0 is the only outstanding group; z is first consumed here.
                T.warpgroup_wait(0)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    z[token, dim] = beta_shared[token] * (
                        v_shared[token, dim]
                        - gamma_shared[token] * z[token, dim]
                    )
                T.copy(z, z_shared)
                T.wgmma_gemm(a_shared, z_shared, z, clear_accum=True)
                T.wgmma_gemm(q_shared, state_shared, out, clear_accum=True)
                T.wgmma_gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                )

                # Retire G1 (A @ residual), leaving G2/G3 in flight while
                # zhat is formed. Neither out nor score is read yet.
                T.warpgroup_wait(2)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    z[token, dim] *= (
                        gamma_last[0] * inv_gamma_shared[token]
                    )
                T.copy(z, z_shared)

                # out and score are first consumed below, so G2/G3 must now
                # both be complete.
                T.warpgroup_wait(0)
                for token, dim in T.Parallel(CHUNK_SIZE, dv_tile):
                    out[token, dim] *= SCALE * gamma_shared[token]
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

                # Both updates consume zhat. The score's column decay was
                # algebraically folded into its row scale above.
                T.wgmma_gemm(
                    score_shared,
                    z_shared,
                    out,
                    clear_accum=False,
                )
                T.wgmma_gemm(
                    k_shared,
                    z_shared,
                    state,
                    transpose_A=True,
                    clear_accum=False,
                )
                # Retire only G4 so the state update overlaps the output
                # shared/global store while G5 remains in flight.
                T.warpgroup_wait(1)

                T.copy(out, z_shared)
                T.copy(
                    z_shared,
                    output[
                        bb,
                        left:right,
                        bh,
                        dv_left : dv_left + dv_tile,
                    ],
                )
                # The next chunk copies state to shared, so G5 retires here.
                T.warpgroup_wait(0)

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


def _select_gate_cp_parts(
    batch_size,
    full_chunks_only,
    rs_available,
    base_blocks,
    blocks_per_sm,
    chunks_per_batch,
):
    """Choose power-of-two sequence parallelism from occupancy and chain length."""
    if (
        GATE_CP_MODE == "off"
        or batch_size != 1
        or not full_chunks_only
        or not rs_available
    ):
        return 1

    if GATE_CP_MODE in ("auto", "on"):
        resident_slots = MIG_SM_COUNT * blocks_per_sm
        if (
            base_blocks >= resident_slots
            or chunks_per_batch < GATE_CP_MIN_CHUNKS
        ):
            return 1
        # FlashQLA's latency model minimizes
        #   local_chunks + total_block_chunks / (slots * local_chunks).
        # The empirical factor 3 and power-of-two rounding follow upstream;
        # resident_slots is adapted to this 14-SM MIG and our CTA resources.
        estimated_local_chunks = 3.0 * math.sqrt(
            base_blocks * chunks_per_batch / resident_slots
        )
        local_chunks = 2 ** round(math.log2(estimated_local_chunks))
        local_chunks = max(local_chunks, 4)
        seq_parts = 1
        while (
            seq_parts < 8
            and chunks_per_batch // seq_parts > local_chunks
        ):
            seq_parts *= 2
        # Keep equal static slices; unsupported divisibility falls back by
        # powers of two instead of adding a second tail specialization.
        while seq_parts > 1 and chunks_per_batch % seq_parts != 0:
            seq_parts //= 2
        return seq_parts

    try:
        seq_parts = int(GATE_CP_MODE)
    except ValueError:
        return 1
    if (
        seq_parts not in (2, 4, 8)
        or chunks_per_batch % seq_parts != 0
    ):
        return 1
    return seq_parts


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
    gate_cp_states = initial_state
    gate_cp_warmup_counts = None
    gate_cp_warmup = None
    gate_cp_prepare = None
    seq_parts = 1
    use_rs = False
    full_chunks_only = num_tokens % CHUNK_SIZE == 0
    state_owners = batch_size * num_heads_v
    if DV_SPLIT_MODE == "auto":
        # D=64 has one warp group per CTA. It wins when both value parts
        # spread over a single 14-SM wave; 1..512 chunk sweeps show no
        # additional chain-length crossover on the full-chunk RS path.
        split_profitable = state_owners * 2 <= MIG_SM_COUNT
        dv_tile, dv_parts = (
            DV_SPLIT_CONFIGS["64"]
            if split_profitable
            else DV_SPLIT_CONFIGS["off"]
        )
    else:
        dv_tile, dv_parts = DV_SPLIT_CONFIGS.get(
            DV_SPLIT_MODE, DV_SPLIT_CONFIGS["off"]
        )
    rs_available = RS_MODE in ("auto", "on") and dv_tile >= 64
    if MEMORY_IO_MODE == "auto":
        use_memory_io = full_chunks_only or rs_available
    else:
        use_memory_io = MEMORY_IO_MODE == "on" and (
            full_chunks_only or rs_available
        )
    # TileLang requires all async producers of a consumer in one stage.
    if PREFETCH_MODE in ("on", "k", "qk"):
        use_memory_io = False
    if use_memory_io:
        if PREFETCH_MODE == "auto":
            prefetch_profile = "qkva"
        elif PREFETCH_MODE == "on":
            prefetch_profile = "k"
        else:
            prefetch_profile = PREFETCH_MODE
        prefetch_q, prefetch_k, prefetch_v, prefetch_a = (
            PREFETCH_INPUTS.get(prefetch_profile, PREFETCH_INPUTS["off"])
        )
        use_rs = rs_available
        seq_parts = _select_gate_cp_parts(
            batch_size=batch_size,
            full_chunks_only=full_chunks_only,
            rs_available=use_rs,
            base_blocks=state_owners * dv_parts,
            blocks_per_sm=2 if dv_tile == 64 else 1,
            chunks_per_batch=chunks_per_batch,
        )
        kernel_factory = (
            tilelang_residual_first_full_chunks_rs
            if use_rs
            else tilelang_residual_first_full_chunks
        )
        kernel_kwargs = dict(
            qk_dtype=q.dtype,
            v_dtype=v.dtype,
            gate_dtype=g_cumsum.dtype,
            accum_dtype="float32",
            use_initial_state=use_initial_state,
            dv_tile=dv_tile,
            dv_parts=dv_parts,
            prefetch_q=prefetch_q,
            prefetch_k=prefetch_k,
            prefetch_v=prefetch_v,
            prefetch_a=prefetch_a,
        )
        if use_rs:
            kernel_kwargs["reuse_output_shared"] = (
                dv_tile == 64
                and (
                    state_owners * dv_parts > MIG_SM_COUNT
                    or seq_parts > 1
                )
            )
            kernel_kwargs["has_tail"] = not full_chunks_only
            kernel_kwargs["seq_parts"] = seq_parts
            if seq_parts > 1:
                gate_cp_states = torch.empty(
                    (
                        batch_size,
                        seq_parts - 1,
                        num_heads_v,
                        HEAD_DIM_K,
                        HEAD_DIM_V,
                    ),
                    dtype=torch.float32,
                    device=v.device,
                )
                gate_cp_warmup_counts = torch.empty(
                    (batch_size, seq_parts - 1, num_heads_v),
                    dtype=torch.int32,
                    device=v.device,
                )
                gate_cp_warmup = tilelang_get_gate_cp_warmup(
                    num_heads_v,
                    gate_dtype=g_cumsum.dtype,
                    seq_parts=seq_parts,
                    gate_threshold=GATE_CP_THRESHOLD,
                )
                gate_cp_prepare = tilelang_prepare_gate_cp_states(
                    num_heads_v,
                    num_heads_qk,
                    qk_dtype=q.dtype,
                    v_dtype=v.dtype,
                    gate_dtype=g_cumsum.dtype,
                    accum_dtype="float32",
                    use_initial_state=use_initial_state,
                    dv_tile=dv_tile,
                    dv_parts=dv_parts,
                    seq_parts=seq_parts,
                )
        recurrent = kernel_factory(
            num_heads_v,
            num_heads_qk,
            **kernel_kwargs,
        )
    else:
        if PREFETCH_MODE == "auto":
            prefetch_k = dv_parts > 1 and chunks_per_batch >= 64
        else:
            prefetch_k = PREFETCH_MODE in ("on", "k")
        recurrent = tilelang_residual_first(
            num_heads_v,
            num_heads_qk,
            qk_dtype=q.dtype,
            v_dtype=v.dtype,
            gate_dtype=g_cumsum.dtype,
            accum_dtype="float32",
            use_initial_state=use_initial_state,
            dv_tile=dv_tile,
            dv_parts=dv_parts,
            prefetch_k=prefetch_k,
        )
    if gate_cp_warmup is not None:
        gate_cp_warmup(
            g_cumsum,
            gate_cp_warmup_counts,
            chunks_per_batch,
        )
    if gate_cp_prepare is not None:
        gate_cp_prepare(
            k,
            v,
            g_cumsum,
            beta,
            A,
            initial_state,
            gate_cp_warmup_counts,
            gate_cp_states,
            chunks_per_batch,
        )
    if use_rs:
        recurrent(
            q,
            k,
            v,
            g_cumsum,
            beta,
            A,
            initial_state,
            gate_cp_states,
            output,
            final_state,
            chunks_per_batch,
        )
    else:
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

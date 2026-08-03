"""Document-form baseline retained for comparison and profiling.

This follows the lab equations literally:
    W = A @ (beta * exp(g) * K)
    U = A @ (beta * V)
    Z = U - W @ S
Set GDN_IMPL=document before importing student.tilelang_fwd to select it.
"""

import torch
import tilelang
import tilelang.language as T


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
def tilelang_prepare_w_u(H, Hg, qk_dtype, v_dtype, gate_dtype, accum_dtype):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    k_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    wu_shape = (batch_size, num_tokens, H, HEAD_DIM_V)

    @T.prim_func
    def kernel(
        k: T.Tensor(k_shape, dtype=qk_dtype),
        v: T.Tensor(v_shape, dtype=v_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        beta: T.Tensor(gate_shape, dtype=gate_dtype),
        a: T.Tensor(a_shape, dtype=qk_dtype),
        w: T.Tensor(wu_shape, dtype=qk_dtype),
        u: T.Tensor(wu_shape, dtype=v_dtype),
        total_chunks: T.int32,
    ):
        with T.Kernel(total_chunks * H, threads=128) as (block,):
            chunk = block // H
            bh = block % H
            bb = chunk % batch_size
            local_chunk = chunk // batch_size
            bhg = bh // (H // Hg)
            left = local_chunk * CHUNK_SIZE

            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            operand_shared = T.alloc_shared(
                (CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype
            )
            result = T.alloc_fragment(
                (CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype
            )

            for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                if left + row < num_tokens:
                    a_shared[row, col] = a[bb, left + row, bh, col]
                else:
                    a_shared[row, col] = 0

            for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                if left + token < num_tokens:
                    operand_shared[token, dim] = (
                        k[bb, left + token, bhg, dim]
                        * beta[bb, left + token, bh]
                        * T.exp2(g[bb, left + token, bh] * LOG2E)
                    )
                else:
                    operand_shared[token, dim] = 0
            T.gemm(a_shared, operand_shared, result, clear_accum=True)
            for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                if left + token < num_tokens:
                    w[bb, left + token, bh, dim] = result[token, dim]

            for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                if left + token < num_tokens:
                    operand_shared[token, dim] = (
                        v[bb, left + token, bh, dim]
                        * beta[bb, left + token, bh]
                    )
                else:
                    operand_shared[token, dim] = 0
            T.gemm(a_shared, operand_shared, result, clear_accum=True)
            for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                if left + token < num_tokens:
                    u[bb, left + token, bh, dim] = result[token, dim]

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tilelang_document_output_and_state(
    H,
    Hg,
    qk_dtype,
    v_dtype,
    gate_dtype,
    accum_dtype,
    use_initial_state,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")
    qk_shape = (batch_size, num_tokens, Hg, HEAD_DIM_K)
    v_shape = (batch_size, num_tokens, H, HEAD_DIM_V)
    gate_shape = (batch_size, num_tokens, H)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    initial_shape = state_shape if use_initial_state else (1,)

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, dtype=qk_dtype),
        k: T.Tensor(qk_shape, dtype=qk_dtype),
        g: T.Tensor(gate_shape, dtype=gate_dtype),
        w: T.Tensor(v_shape, dtype=qk_dtype),
        u: T.Tensor(v_shape, dtype=v_dtype),
        initial_state: T.Tensor(initial_shape, dtype=accum_dtype),
        output: T.Tensor(v_shape, dtype=v_dtype),
        final_state: T.Tensor(state_shape, dtype=accum_dtype),
        chunks_per_batch: T.int32,
    ):
        with T.Kernel(batch_size * H, threads=256) as (block,):
            bb = block // H
            bh = block % H
            bhg = bh // (H // Hg)

            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            w_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            z_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=v_dtype)
            state_shared = T.alloc_shared(
                (HEAD_DIM_K, HEAD_DIM_V), dtype=v_dtype
            )
            score_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype
            )
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            g_last = T.alloc_shared((1,), dtype=gate_dtype)

            state = T.alloc_fragment(
                (HEAD_DIM_K, HEAD_DIM_V), dtype=accum_dtype
            )
            z = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            out = T.alloc_fragment((CHUNK_SIZE, HEAD_DIM_V), dtype=accum_dtype)
            score = T.alloc_fragment(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=accum_dtype
            )

            if use_initial_state:
                T.copy(initial_state[bb, bh, 0:HEAD_DIM_K, 0:HEAD_DIM_V], state)
            else:
                T.clear(state)

            for chunk in T.serial(chunks_per_batch):
                left = chunk * CHUNK_SIZE
                right = left + CHUNK_SIZE

                T.copy(state, state_shared)
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_K):
                    if left + token < num_tokens:
                        q_shared[token, dim] = q[bb, left + token, bhg, dim]
                        k_shared[token, dim] = k[bb, left + token, bhg, dim]
                        w_shared[token, dim] = w[bb, left + token, bh, dim]
                    else:
                        q_shared[token, dim] = 0
                        k_shared[token, dim] = 0
                        w_shared[token, dim] = 0
                for token in T.Parallel(CHUNK_SIZE):
                    if left + token < num_tokens:
                        g_shared[token] = g[bb, left + token, bh]
                    else:
                        g_shared[token] = 0
                if right <= num_tokens:
                    g_last[0] = g_shared[CHUNK_SIZE - 1]
                else:
                    g_last[0] = g[bb, num_tokens - 1, bh]

                T.gemm(w_shared, state_shared, z, clear_accum=True)
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    if left + token < num_tokens:
                        z[token, dim] = u[bb, left + token, bh, dim] - z[token, dim]
                    else:
                        z[token, dim] = 0
                T.copy(z, z_shared)

                T.gemm(q_shared, state_shared, out, clear_accum=True)
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    out[token, dim] *= SCALE * T.exp2(
                        g_shared[token] * LOG2E
                    )

                T.gemm(
                    q_shared,
                    k_shared,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                )
                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if row >= col and left + row < num_tokens:
                        score[row, col] *= SCALE * T.exp2(
                            (g_shared[row] - g_shared[col]) * LOG2E
                        )
                    else:
                        score[row, col] = 0
                T.copy(score, score_shared)
                T.gemm(score_shared, z_shared, out, clear_accum=False)

                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    if left + token < num_tokens:
                        output[bb, left + token, bh, dim] = out[token, dim]

                for dim_k, dim_v in T.Parallel(HEAD_DIM_K, HEAD_DIM_V):
                    state[dim_k, dim_v] *= T.exp2(g_last[0] * LOG2E)
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    z[token, dim] *= T.exp2(
                        (g_last[0] - g_shared[token]) * LOG2E
                    )
                T.copy(z, z_shared)
                T.gemm(
                    k_shared,
                    z_shared,
                    state,
                    transpose_A=True,
                    clear_accum=False,
                )

            T.copy(state, final_state[bb, bh, 0:HEAD_DIM_K, 0:HEAD_DIM_V])

    return kernel


def gdn_prefill_forward_document(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_tokens, num_heads_qk, _ = q.shape
    num_heads_v = v.shape[2]
    chunks_per_batch = tilelang.cdiv(num_tokens, CHUNK_SIZE)
    total_chunks = batch_size * chunks_per_batch

    w = torch.empty_like(v)
    u = torch.empty_like(v)
    output = torch.empty_like(v)
    final_state = torch.empty(
        (batch_size, num_heads_v, HEAD_DIM_K, HEAD_DIM_V),
        dtype=torch.float32,
        device=v.device,
    )

    prepare = tilelang_prepare_w_u(
        num_heads_v,
        num_heads_qk,
        qk_dtype=q.dtype,
        v_dtype=v.dtype,
        gate_dtype=g_cumsum.dtype,
        accum_dtype="float32",
    )
    prepare(k, v, g_cumsum, beta, A, w, u, total_chunks)

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((1,), dtype=torch.float32, device=v.device)
    recurrent = tilelang_document_output_and_state(
        num_heads_v,
        num_heads_qk,
        qk_dtype=q.dtype,
        v_dtype=v.dtype,
        gate_dtype=g_cumsum.dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
    )
    recurrent(
        q,
        k,
        g_cumsum,
        w,
        u,
        initial_state,
        output,
        final_state,
        chunks_per_batch,
    )
    return output, final_state

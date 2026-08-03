# Residual-first baseline: Z = A @ (beta * (V - exp(g) * K @ S)).
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
def tilelang_residual_first(
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
    a_shape = (batch_size, num_tokens, H, CHUNK_SIZE)
    state_shape = (batch_size, H, HEAD_DIM_K, HEAD_DIM_V)
    initial_shape = state_shape if use_initial_state else (1,)

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
        # One block owns one recurrent state; chunks must be traversed serially.
        with T.Kernel(batch_size * H, threads=256) as (block,):
            bb = block // H
            bh = block % H
            bhg = bh // (H // Hg)

            q_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            k_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_K), dtype=qk_dtype)
            v_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=v_dtype)
            a_shared = T.alloc_shared((CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype)
            z_shared = T.alloc_shared((CHUNK_SIZE, HEAD_DIM_V), dtype=v_dtype)
            state_shared = T.alloc_shared(
                (HEAD_DIM_K, HEAD_DIM_V), dtype=v_dtype
            )
            score_shared = T.alloc_shared(
                (CHUNK_SIZE, CHUNK_SIZE), dtype=qk_dtype
            )
            g_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
            beta_shared = T.alloc_shared((CHUNK_SIZE,), dtype=gate_dtype)
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
                        v_shared[token, dim] = v[bb, left + token, bh, dim]
                    else:
                        q_shared[token, dim] = 0
                        k_shared[token, dim] = 0
                        v_shared[token, dim] = 0
                for row, col in T.Parallel(CHUNK_SIZE, CHUNK_SIZE):
                    if left + row < num_tokens:
                        a_shared[row, col] = a[bb, left + row, bh, col]
                    else:
                        a_shared[row, col] = 0
                for token in T.Parallel(CHUNK_SIZE):
                    if left + token < num_tokens:
                        g_shared[token] = g[bb, left + token, bh]
                        beta_shared[token] = beta[bb, left + token, bh]
                    else:
                        g_shared[token] = 0
                        beta_shared[token] = 0
                if right <= num_tokens:
                    g_last[0] = g_shared[CHUNK_SIZE - 1]
                else:
                    g_last[0] = g[bb, num_tokens - 1, bh]

                # Residual-first form:
                #   R = beta * (V - exp(g) * K @ S)
                #   Z = A @ R
                T.gemm(k_shared, state_shared, z, clear_accum=True)
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    if left + token < num_tokens:
                        z[token, dim] = beta_shared[token] * (
                            v_shared[token, dim]
                            - T.exp2(g_shared[token] * LOG2E) * z[token, dim]
                        )
                    else:
                        z[token, dim] = 0
                T.copy(z, z_shared)
                T.gemm(a_shared, z_shared, z, clear_accum=True)
                T.copy(z, z_shared)

                # Contribution from the state at the start of the chunk.
                T.gemm(q_shared, state_shared, out, clear_accum=True)
                for token, dim in T.Parallel(CHUNK_SIZE, HEAD_DIM_V):
                    out[token, dim] *= SCALE * T.exp2(
                        g_shared[token] * LOG2E
                    )

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

                # S' = exp(g_last) S + K^T @ (exp(g_last-g_i) Z_i).
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
    recurrent = tilelang_residual_first(
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

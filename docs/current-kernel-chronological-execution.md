# 当前 residual-first kernel：从 launch 到结束的时间线

本文对应当前默认实现
[student/tilelang_fwd.py](../student/tilelang_fwd.py)。重点不是重新推导算法，而是按照真实执行顺序回答：

1. Python wrapper 在 launch 前做什么；
2. 每个 public case 到底 launch 多少 blocks、是否切分 `dv`；
3. 一个 CTA 从获得 SM 资源开始，依次把哪些数据搬到哪里；
4. `T.Pipelined` 预取了什么，没有预取什么；
5. 每个 chunk 内六次 GEMM 和 element-wise 操作以什么顺序发生；
6. state 如何跨越很多 chunks，最终如何写回 global memory。

这里讨论的是默认 residual 实现。设置 `GDN_IMPL=document` 会切换到保留的原始公式实现，不属于本文范围。

## 1. 固定维度和符号

当前题目的固定维度为：

```text
chunk size C = 64
key/query dimension dk = 128
value dimension dv = 128
```

对一个 batch、一个 value head 和一个 chunk，使用下列符号：

| 名称 | shape | global dtype | 含义 |
| --- | --- | --- | --- |
| `Q` | `[64, 128]` | BF16 | 当前 chunk 的 query |
| `K` | `[64, 128]` | BF16 | 当前 chunk 的 normalized key |
| `V` | `[64, 128]` | BF16 | 当前 chunk 的 value |
| `A` | `[64, 64]` | BF16 | 预处理得到的 causal correction matrix |
| `g` | `[64]` | FP32 | chunk-local log gate prefix sum |
| `beta` | `[64]` | FP32 | delta update strength |
| `S` | `[128, 128]` | FP32 | chunk 入口 recurrent state |
| `O` | `[64, 128]` | BF16 | 最终 output |

完整 forward 在进入本文 kernel 前，已经由其他预处理 kernels 得到：

```text
raw g -> chunk-local g_cumsum
k, g_cumsum, beta -> A
```

本文 kernel 的输入已经是 `g_cumsum` 和 `A`。因此下面的时间线从
`gdn_prefill_forward(..., g_cumsum, beta, A, ...)` 开始。

## 2. Launch 前：wrapper 决定是否切分 dv

Python wrapper 首先读取：

```text
B   = batch_size
T   = num_tokens
Hq  = num_heads_qk
Hv  = num_heads_v
Nc  = ceil(T / 64)
owners = B * Hv
```

一个 owner 指一个独立的 `(batch, value_head)` recurrent state chain。

随后 wrapper 在 global memory 中分配：

```text
output      [B, T, Hv, 128]       BF16
final_state [B, Hv, 128, 128]     FP32
```

这些 `torch.empty` 分配以及后面的 kernel launch 都发生在每次被测函数调用中。

默认 `GDN_DV_SPLIT=32`，候选配置是：

```text
dv_tile  = 32
dv_parts = 4
```

只有同时满足以下条件才真正切分：

```text
owners < 14
Nc >= 64
owners * 4 <= 2 * 14
```

三个条件分别表示：

1. 原始 owner 数少于 MIG 的 14 个 SM，确实存在 grid parallelism 不足；
2. 至少有 64 个 chunks，让额外 blocks 的收益足以摊薄启动和重复计算；
3. 切分后的 blocks 不超过 28，因为 NCU 已确认 `dv=32` 时最多可驻留 2 blocks/SM。

如果任一条件不满足，则使用：

```text
dv_tile  = 128
dv_parts = 1
```

环境变量还允许实验：

```text
GDN_DV_SPLIT=off   -> 1 x 128
GDN_DV_SPLIT=64    -> 2 x 64
GDN_DV_SPLIT=32    -> 4 x 32
```

`3 x 48` 已经删除。它在当前 TileLang/Hopper lowering 上可以编译，但数值不正确。

## 3. 每个 public case 实际如何切分

当前 8 个 public cases 的 launch 决策如下：

| case | `B` | `T` | `Hq` | `Hv` | owners | chunks/owner | 配置 | grid blocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `short_tail_state` | 1 | 1025 | 2 | 8 | 8 | 17 | `1 x 128` | 8 |
| `chain_equal` | 1 | 8192 | 4 | 4 | 4 | 128 | `4 x 32` | 16 |
| `parallel_equal` | 1 | 2048 | 16 | 16 | 16 | 32 | `1 x 128` | 16 |
| `parallel_gva` | 1 | 2048 | 4 | 16 | 16 | 32 | `1 x 128` | 16 |
| `long_low_gva` | 1 | 32768 | 2 | 8 | 8 | 512 | `1 x 128` | 8 |
| `batch_split_gva` | 4 | 8192 | 2 | 8 | 32 | 128 | `1 x 128` | 32 |
| `wide_gva_state` | 1 | 8192 | 16 | 64 | 64 | 128 | `1 x 128` | 64 |
| `deep_gva_state` | 1 | 16384 | 8 | 32 | 32 | 256 | `1 x 128` | 32 |

逐项解释：

- `short_tail_state`：owners 只有 8，但只有 17 chunks，小于 64，所以不切。
- `chain_equal`：owners=4、chunks=128，切分后 grid=16，并且 `16 <= 28`，因此切成四片。
- `parallel_equal`：owners=16，原始 grid 已超过 14 个 SM，不属于低并行 case。
- `parallel_gva`：同样有 16 owners；GVA 只改变 Q/K head 映射，不改变 owner 数。
- `long_low_gva`：owners=8。如果切成四片会产生 32 blocks，超过两 CTA/SM 的总容量 28。
  少数 SM 必须再执行第三条完整的 512-chunk chain，实测反而显著变慢，因此不切。
- `batch_split_gva`：四个 batch 使 owners 达到 32，不切。
- `wide_gva_state`：64 个 value heads 已经提供 64 blocks，不切。
- `deep_gva_state`：32 个 value heads 已经提供 32 blocks，不切。

所以在当前 public cases 中，只有 `chain_equal` 默认使用 `4 x 32`。

## 4. Kernel grid 与 CTA ownership

实际 launch 为：

```text
grid.x  = B * Hv * dv_parts
block.x = 256 threads = 8 warps
```

block index 被解释为：

```text
owner   = block / dv_parts
part    = block % dv_parts

batch   = owner / Hv
v_head  = owner % Hv
qk_head = v_head / (Hv / Hq)

dv_left = part * dv_tile
```

未切分时，`part=0`，一个 CTA 拥有：

```text
S[:, 0:128]
V[:, 0:128]
O[:, 0:128]
```

`4 x 32` 时，同一个 `(batch, v_head)` 产生四个独立 CTA：

```text
part 0 owns dv [  0,  32)
part 1 owns dv [ 32,  64)
part 2 owns dv [ 64,  96)
part 3 owns dv [ 96, 128)
```

这四个 CTA 不需要相互通信，因为所有 value 列都是独立的：

```text
S = [S0 | S1 | S2 | S3]
V = [V0 | V1 | V2 | V3]
Z = [Z0 | Z1 | Z2 | Z3]
O = [O0 | O1 | O2 | O3]
```

对每一片 `p`：

```text
K @ S       -> K @ Sp
A @ R       -> A @ Rp
Q @ S       -> Q @ Sp
score @ Z   -> score @ Zp
K^T @ Z     -> K^T @ Zp
```

没有任何 GEMM 会把不同 `dv` 列相加，因此不需要 atomic、reduction 或跨 CTA barrier。

GVA 情况下，多个 value heads 共享一个 Q/K head：

```text
qk_head = v_head / (Hv / Hq)
```

它只影响 Q/K 从 global memory 的哪个 head 读取。每个 value head 仍然有自己的 V、g、beta、A、state
和 output。

## 5. CTA 获得 SM 后分配哪些片上资源

### 5.1 Register fragments

每个 CTA 分配：

| fragment | shape | dtype | 用途 |
| --- | --- | --- | --- |
| `state` | `[128, D]` | FP32 | 跨所有 chunks 存活的 recurrent state |
| `z` | `[64, D]` | FP32 | residual correction / corrected value |
| `out` | `[64, D]` | FP32 | output accumulator |
| `score` | `[64, 64]` | FP32 | causal QK score accumulator |

其中 `D=dv_tile`。

这些 fragment 不是一块共享的 register array。TileLang 将元素分布到 256 个 threads 的物理寄存器中。

仅计算主要 payload：

```text
D=128:
state 64 + z 32 + out 32 + score 16 = 144 registers/thread

D=32:
state 16 + z 8 + out 8 + score 16 = 48 registers/thread
```

NCU 包含地址、谓词和控制变量后的实测是：

| 配置 | registers/thread | local spill |
| --- | ---: | ---: |
| `1 x 128` | 197 | 0 |
| `4 x 32` | 106 | 0 |

### 5.2 Shared-memory buffers

逻辑 shared buffers 为：

| buffer | shape | dtype | `D=128` | `D=32` |
| --- | --- | --- | ---: | ---: |
| `q_shared` | `[64,128]` | BF16 | 16 KiB | 16 KiB |
| `k_shared` | `[64,128]` | BF16 | 16 KiB | 16 KiB |
| `v_shared` | `[64,D]` | BF16 | 16 KiB | 4 KiB |
| `a_shared` | `[64,64]` | BF16 | 8 KiB | 8 KiB |
| `z_shared` | `[64,D]` | BF16 | 16 KiB | 4 KiB |
| `state_shared` | `[128,D]` | BF16 | 32 KiB | 8 KiB |
| `score_shared` | `[64,64]` | BF16 | 8 KiB | 8 KiB |
| `g_shared` | `[64]` | FP32 | 256 B | 256 B |
| `beta_shared` | `[64]` | FP32 | 256 B | 256 B |
| `g_exp_shared` | `[64]` | FP32 | 256 B | 256 B |
| `g_inv_exp_shared` | `[64]` | FP32 | 256 B | 256 B |
| `g_last_exp` | `[1]` | FP32 | 4 B | 4 B |

软件 pipeline 会为 stage-0 输入生成 ping-pong storage。

当前 `D=128` 会双缓冲：

```text
Q, K, V, A, g, beta
```

当前 `D=32` 会双缓冲：

```text
Q, K, V-slice, g, beta
```

`D=32` 暂时不双缓冲 A。A 在 stage 1 才装载，以减小 shared footprint。

不会双缓冲：

```text
state_shared, z_shared, score_shared,
g_exp_shared, g_inv_exp_shared, g_last_exp
```

NCU 实测：

| 配置 | dynamic shared/block | shared block limit |
| --- | ---: | ---: |
| `1 x 128` | 173.58 KB | 1 block/SM |
| `4 x 32` | 103.95 KB | 2 blocks/SM |

这里的 `state_shared` 只是当前 chunk 的 BF16 GEMM operand。真正跨 chunks 保存精度的是 FP32 register
fragment `state`。

## 6. Kernel 开始：state 初始化

在进入 chunk loop 前，每个 CTA 只执行一次 state 初始化。

有 `initial_state` 时：

```text
global initial_state[
    batch, v_head, 0:128, dv_left:dv_left+D
]
        |
        | global -> register fragment, FP32
        v
state[128, D]
```

没有 `initial_state` 时，CTA 在寄存器中将自己的整个 `state[128,D]` 清零。

切分后的四个 CTA 分别读取 initial state 的四个不重叠列区间。

此时还没有 `state_shared` 的有效内容。它要等每个 chunk 的 stage 1 开始时，才从当前 FP32 state
生成。

## 7. Pipeline prologue：为 chunk 0 准备只读输入

`T.Pipelined` 有两个逻辑 stage：

```text
stage 0: 准备不依赖 recurrent state 的输入
stage 1: 使用当前 state 完成整个 chunk
```

在第一个 chunk 计算开始前，pipeline prologue 先从 global address space 读取 chunk 0 的：

```text
Q[64,128]       -> q_shared
K[64,128]       -> k_shared
V[64,D]         -> v_shared
g[64]           -> g_shared
beta[64]        -> beta_shared
```

这些 global load 实际可能命中 L2/L1，也可能访问 HBM；源码只能指定 global address space，不能保证
cache hit。

未切分 `D=128` 时，prologue 还预取：

```text
A[64,64] -> a_shared
```

切分 `D=32` 时，A 不在 stage 0，稍后在 chunk 的 stage 1 才读取。

四个 `dv` parts 会重复读取相同的 Q、K、g、beta、A。它们只读取不同的 V slice。相邻 CTA 的重复
global reads 有机会从 L2 获得复用，但源码没有显式跨 CTA 共享。

## 8. 稳态 pipeline：chunk c 计算时预取 chunk c+1

pipeline 进入稳态后，可以把执行想象为：

```text
shared set 0: load chunk 0 | compute chunk 0 | load chunk 2 | compute chunk 2
shared set 1:              | load chunk 1    | compute chunk 1 | load chunk 3
```

更准确地说，在 stage 1 消费 chunk `c` 的一套 Q/K/V/g/beta buffer 时，stage 0 可以用另一套物理
buffer 准备 chunk `c+1`。

被重叠的是：

```text
global -> shared input load
             与
当前 chunk 的 GEMM / element-wise / state update
```

没有被重叠的是 state chain：

```text
S_c -> compute chunk c -> S_(c+1) -> compute chunk c+1
```

`state_shared(c+1)` 不能提前生成，因为它必须来自 chunk `c` 最后才得到的 `S_(c+1)`。

`z_shared` 和 `score_shared` 也没有 ping-pong；它们在同一个 chunk 内反复作为不同 GEMM 的
producer/consumer buffer 使用，编译器在必要位置插入同步。

## 9. 一个 chunk 的完整 stage-1 时间线

下面严格按照当前源码顺序描述 chunk `c`。

### 9.1 确定 token 区间

```text
left  = c * 64
right = left + 64
```

完整 chunk 满足 `right <= T`。最后一个不完整 chunk 使用相同的矩阵 tile，但 element-wise 和输出
写回会屏蔽无效 token。

### 9.2 把 FP32 state 转成 BF16 shared operand

```text
FP32 state register fragment [128,D]
        |
        | T.copy, FP32 -> BF16
        v
BF16 state_shared [128,D]
```

这是每个 chunk 都会发生的搬运。

`state` 本身没有离开 register file，也没有被 BF16 值覆盖。`state_shared` 只是供 Tensor Core
读取的低精度快照。

对于 `D=32`，此时还会执行当前 chunk 的：

```text
global A[64,64] -> a_shared
```

它没有与上一个 chunk 的计算重叠。对于 `D=128`，A 已由 stage 0 预取。

### 9.3 计算 gate cache

对 64 个 token 并行计算：

```text
gamma[i]     = exp(g[i])
inv_gamma[i] = 1 / gamma[i]
```

实现使用：

```text
exp(g) = exp2(g * log2(e))
```

结果分别保存在：

```text
g_exp_shared
g_inv_exp_shared
```

完整 chunk：

```text
gamma_last = gamma[63]
```

tail chunk：

```text
gamma_last = exp(g[T-1])
```

因此后面所有 decay 都使用缓存值，不再对每个矩阵元素重复调用 exp。

### 9.4 GEMM 1：入口 state 对 K 的预测

```text
z = K @ state_shared
```

shape：

```text
[64,128] BF16 @ [128,D] BF16 -> [64,D] FP32 accumulator
```

它计算每个 token 从 chunk 入口 state 读出的预测 value。

### 9.5 Element-wise：形成 prediction residual

对有效 token 和本 CTA 的 D 个 value 列：

```text
z[i,d] = beta[i] * (
    V[i,d] - gamma[i] * z[i,d]
)
```

这里的 `z` 变成 residual `R`。

tail chunk 的无效 token 被显式写为 0，确保后续固定大小 GEMM 不产生无效更新。

### 9.6 Register -> shared：准备 GEMM 2

```text
FP32 z fragment [64,D]
        |
        | T.copy, FP32 -> BF16
        v
BF16 z_shared [64,D]
```

这次量化发生在 residual 形成之后。

### 9.7 GEMM 2：A 修正 chunk 内 causal feedback

```text
z = A @ z_shared
```

shape：

```text
[64,64] BF16 @ [64,D] BF16 -> [64,D] FP32
```

此后 `z` 表示 residual-first 公式中的 corrected value `Z`。

随后再次：

```text
FP32 z -> BF16 z_shared
```

因为后面的 `score @ Z` 要把 Z 当作 Tensor Core 输入。

### 9.8 GEMM 3：旧 state 对 output 的贡献

```text
out = Q @ state_shared
```

shape：

```text
[64,128] BF16 @ [128,D] BF16 -> [64,D] FP32
```

随后：

```text
out[i,d] *= scale * gamma[i]
scale = 1 / sqrt(128)
```

此时 `out` 只包含 chunk 入口旧 state 的贡献。

### 9.9 GEMM 4：计算 chunk 内 QK score

```text
score = Q @ K^T
```

shape：

```text
[64,128] BF16 @ [128,64] BF16 -> [64,64] FP32
```

然后应用 causal mask、scale 和相对 decay：

```text
if row >= col and row is valid:
    score[row,col] *= (
        scale * gamma[row] * inv_gamma[col]
    )
else:
    score[row,col] = 0
```

因为 `col <= row`，只要 row 是有效 token，col 也一定有效。

`score` 不依赖 value 列，所以 `4 x 32` 的四个 CTA 会把这一步完整重复四次。

随后：

```text
FP32 score fragment -> BF16 score_shared
```

### 9.10 GEMM 5：当前 chunk 写入对 output 的贡献

```text
out += score_shared @ z_shared
```

shape：

```text
[64,64] BF16 @ [64,D] BF16 -> [64,D] FP32 accumulate
```

因此最终：

```text
out = scale * gamma * (Q @ S_c)
    + causal_scaled_QK @ Z
```

### 9.11 把 output 写回 global memory

对有效 token：

```text
FP32 out fragment
    -> BF16 output[
        batch,
        left + token,
        v_head,
        dv_left : dv_left + D
    ]
```

四个 parts 写互不重叠的 output 列，所以不存在 write race。

### 9.12 在寄存器中衰减旧 state

```text
state[k,d] *= gamma_last
```

`state` 仍然是 FP32 register fragment。这里真正让 FP32 精度参与跨 chunk 累积。

### 9.13 把 Z 衰减到 chunk 末尾

对每个有效 token：

```text
z[i,d] *= gamma_last * inv_gamma[i]
```

然后第三次执行：

```text
FP32 z -> BF16 z_shared
```

这份 z_shared 专门用于 state update。

### 9.14 GEMM 6：把当前 chunk 的写入累加到 state

```text
state += K^T @ z_shared
```

shape：

```text
[128,64] BF16 @ [64,D] BF16 -> [128,D] FP32 accumulate
```

`clear_accum=False`，所以它累加到已经乘过 `gamma_last` 的 FP32 state 上。

至此：

```text
state == S_(c+1)
```

下一 chunk 才能把这份新 state 转成自己的 `state_shared`。这就是不能并行执行同一 state chain 中
多个 chunks 的根本依赖。

## 10. 多个 chunks 的完整时间关系

对一个 CTA，可以把整个 kernel 压缩成：

```text
load/clear S0

pipeline prologue:
    prefetch readonly inputs for chunk 0

for c = 0 .. Nc-1:
    concurrently prefetch readonly inputs for chunk c+1

    state_shared = bf16(S_c)
    load A_c here if D=32
    cache exp(g_c), reciprocal, gamma_last

    R_c = beta * (V_c - gamma * K_c @ S_c)
    Z_c = A_c @ R_c

    O_c = scale * gamma * Q_c @ S_c
        + causal(scale * Q_c @ K_c^T * decay) @ Z_c
    store O_c

    S_(c+1) = gamma_last * S_c
            + K_c^T @ (gamma_last / gamma * Z_c)

pipeline epilogue:
    drain the last chunk; there is no chunk Nc to prefetch

store S_Nc to final_state
```

不同 owner 的 CTA 完全独立，可以在不同 SM 并行。相同 owner 的四个 dv parts 也完全独立，只是读取
相同的 Q/K/A/g/beta。

## 11. Kernel 结束：final state 写回

所有 chunks 完成后，每个 CTA 执行一次：

```text
FP32 state register fragment [128,D]
        |
        | T.copy
        v
global final_state[
    batch,
    v_head,
    0:128,
    dv_left:dv_left+D
]
```

切分的四个 CTA 写不重叠的 32 列。kernel 返回前，所有 output 和 final state 的 global stores 都已经
完成。

## 12. dv 切分增加了什么计算

未切分时，每 chunk/head 的六次 GEMM 总计：

```text
4,718,592 MAC
```

其中唯一完全不随 D 缩小的是：

```text
Q @ K^T = 524,288 MAC
```

其余五个 GEMM 的 N 维随 `dv_tile` 成比例缩小。

`4 x 32` 时，每个 CTA 的 GEMM 工作量为：

```text
value-dependent work / 4 + one full QK
= 1,572,864 MAC
= 1/3 of the original CTA
```

四个 CTA 总工作量：

```text
4 * 1,572,864
= 6,291,456 MAC
= 1.3333 * unsplit work
```

所以 dv 切分不是减少总计算，而是用约 33.3% 的额外 GEMM MAC 换取：

1. 更大的 grid；
2. 更短的单 CTA state chain step；
3. 更低的 register/shared 占用；
4. 两个独立 CTA 同驻一个 SM 时的 latency hiding。

`chain_equal` 从 4 blocks 增加到 16 blocks，实测从约 0.884 ms 降至约 0.661 ms。

`long_low_gva` 如果强制 `4 x 32`，会从 8 blocks 变成 32 blocks，超过 28-block residency capacity；
实测约 4.88 ms，慢于未切分的约 3.44 ms。因此默认 selector 不切这个 case。

## 13. 当前 profile 对时间线的验证

`chain_equal, 4 x 32` 的 NCU 结果：

| 指标 | 数值 |
| --- | ---: |
| grid | 16 blocks |
| block | 256 threads |
| registers/thread | 106 |
| dynamic shared/block | 103.95 KB |
| theoretical block limit | 2 blocks/SM |
| theoretical occupancy | 25% |
| achieved occupancy | 14.47% |
| local/shared spill | 0 |
| no eligible warp | 72.91% |
| active warps/scheduler | 2.34 |
| eligible warps/scheduler | 0.38 |

理论上每个 SM 可以驻留两个 CTA，但 grid 只有 16 blocks、设备有 14 SM，所以只有两个 SM 会获得第二个
CTA；这解释了 achieved occupancy 仍远低于理论 25%。

即使完成 dv 切分，scheduler 仍有 72.91% 的周期没有 eligible warp。当前主要问题依旧是六段较短
HGMMA 之间的依赖、barrier、shared load 和 state chain，而不是 register spill。

## 14. 必须保持的正确性不变量

后续修改 kernel 时需要保持：

1. 一个 CTA 只写自己的 `(batch, v_head, dv slice)`。
2. 同一个 CTA 内 chunks 严格按时间顺序更新同一份 FP32 state。
3. 只有不依赖 state 的只读输入可以跨 chunks 预取。
4. `state_shared` 必须由当前 chunk 入口的 FP32 state 生成。
5. residual 形成后、A 修正后和 state-update decay 后的三次 z->shared 转换不能混淆。
6. output 必须在 z 被改成 chunk-end decay 形式之前完成。
7. tail 的无效 token 必须令 residual/Z 为 0，并禁止 output 写回。
8. tail 的 `gamma_last` 必须来自全序列最后一个有效 token。
9. final state 必须保持 FP32 global dtype。
10. split parts 只能切 value 列；不能把同一个 state element 交给多个 CTA 写。

# 当前 residual-first GDN 算法与 TileLang kernel 全流程

本文讲解当前 [student/tilelang_fwd.py](../student/tilelang_fwd.py) 中默认启用的 residual-first
实现。目标是同时回答三件事：它在数学上算什么、为什么与实验文档公式等价、每一步最终如何映射到
当前 GPU kernel。

## 1. 问题、形状与符号

实验固定：

```text
chunk size C = 64
key/query dimension dk = 128
value dimension dv = 128
```

对一个 batch `b`、一个 value head `h`、一个 chunk，省略 `b/h/chunk` 下标后：

| 符号 | 形状 | dtype in interface | 含义 |
| --- | ---: | --- | --- |
| `Q` | `[C, dk]` | BF16 | 当前 chunk query |
| `K` | `[C, dk]` | BF16 | 当前 chunk key，已 L2 normalize |
| `V` | `[C, dv]` | BF16 | 当前 chunk value |
| `g` | `[C]` | FP32 | `log(gamma)`，chunk-local gate prefix sum |
| `gamma=exp(g)` | `[C]` | 不显式全局存储 | 从 chunk 开头到各 token 的累计 decay |
| `beta` | `[C]` | FP32 | delta write strength |
| `A` | `[C, C]` | BF16 | KKT 单位下三角矩阵的逆 |
| `S` | `[dk, dv]` | FP32 | 进入当前 chunk 时的 recurrent state |
| `Z` | `[C, dv]` | FP32 accumulator/BF16 shared | 当前 chunk 经校正后的写入 value |
| `O` | `[C, dv]` | BF16 output | 当前 chunk 输出 |

尾块有效长度可能小于 64。kernel 仍然使用固定 64-row tile，无效输入补零，写输出时用 predicate
屏蔽。

GVA 下 `H_v` 可以大于 `H_q`。令 `G=H_v/H_q`，value head `h` 使用：

```text
qk_head = floor(h / G)
```

Q/K 可以由多个 value heads 共享，但每个 value head 有自己的 V、g、beta、A、S 和 O。

## 2. 从逐 token Gated Delta Rule 开始

对第 `t` 个 token，先衰减旧状态：

```text
S_bar_t = alpha_t * S_(t-1)
```

key 从旧状态读出当前预测：

```text
v_hat_t = k_t @ S_bar_t
```

然后只沿 key `k_t` 指向的方向修正状态：

```text
S_t = S_bar_t + beta_t * k_t^T @ (v_t - v_hat_t)
```

这里：

- `alpha_t` 是全局遗忘比例。
- `beta_t` 是当前关联的覆盖强度。
- `v_t - v_hat_t` 是 residual，即真实 value 与旧状态预测之间的误差。
- `k_t^T @ residual` 是一个 `[dk, dv]` rank-1 update。

如果 `beta_t=0`，当前 token 不写状态；如果 key 已归一化且 `beta_t=1`，当前 key 方向上的旧预测被
完整纠正为 `v_t`。

逐 token 算法因 `S_t` 依赖 `S_(t-1)`，整条序列不能直接当作独立 token 并行。chunk-wise 算法的
目的，是让 chunk 之间继续保持串行 state chain，但把 64 个 token 内的大部分运算改写成 GEMM。

## 3. Gate 的 chunk-local 表示

输入 forward 前，预处理对每个 64-token chunk 独立计算：

```text
raw_g_i = log(alpha_i)
g_i = sum_(j<=i) raw_g_j
gamma_i = exp(g_i) = product_(j<=i) alpha_j
```

因此：

```text
gamma_i / gamma_j = exp(g_i - g_j)
```

表示同一个 chunk 内从 token `j` 到 token `i` 的累计 decay。使用 log-space 是为了把很多小于 1 的
乘法变成稳定的加法。kernel 收到的是 `g`，需要时用：

```text
exp(g) = exp2(g * log2(e))
```

代码中的 `LOG2E` 和 `T.exp2` 就是在做这件事。

## 4. A 在修正什么

如果直接并行计算所有 token 的 residual，它们都会使用 chunk 开头的同一个 `S`，但真实逐 token
递推中，较早 token 的 update 会改变较晚 token 的预测。因此需要一个严格下三角耦合矩阵补回这种
chunk 内因果反馈。

定义：

```text
M = I + StrictLower(B * Gamma * K K^T * Gamma^(-1))
A = M^(-1)
```

其中 `B=Diag(beta)`、`Gamma=Diag(gamma)`。元素形式为：

```text
M[i,j] = beta_i * <k_i,k_j> * gamma_i/gamma_j,  i > j
M[i,i] = 1
M[i,j] = 0,                                      i < j
```

`A` 仍然是单位下三角矩阵。它相当于解一个 causal triangular system，把“所有 token 暂时基于 chunk
入口 state 得到的 residual”修正成与逐 token 顺序更新等价的写入量。`g_cumsum` 和 `A` 都在学生
forward 核心计时区间外由 preprocessing kernel 算好；当前 residual kernel 只读取它们。

## 5. 文档公式与 residual-first 变换

实验文档先定义：

```text
W = A @ (B * Gamma * K)       # [C,C] @ [C,dk] -> [C,dk]
U = A @ (B * V)               # [C,C] @ [C,dv] -> [C,dv]
Z = U - W @ S                 # [C,dv]
```

把 W/U 展开：

```text
Z = A @ (B*V) - A @ (B*Gamma*K) @ S
  = A @ (B*V - B*Gamma*K@S)
  = A @ (B * (V - Gamma*K@S))
```

由于 `B` 和 `Gamma` 都是按 token 逐行缩放的对角阵，可以逐行写成：

```text
P_i = k_i @ S
R_i = beta_i * (v_i - gamma_i * P_i)
Z   = A @ R
```

这就是 residual-first：先让 K 从 chunk 入口状态读出预测 `P`，立刻计算 prediction residual `R`，
再用 `A` 修正 chunk 内的相互影响。

这不是近似，也没有改变 dtype 之前的代数结果。它消除了全局 W/U，并减少一次
`[C,C] x [C,dk]` GEMM。固定 `C=64, dk=dv=128` 时：

```text
document MAC/chunk/head
  = 3*C*dk*dv + 2*C^2*dk + 2*C^2*dv
  = 5,242,880

residual-first MAC/chunk/head
  = 3*C*dk*dv + 1*C^2*dk + 2*C^2*dv
  = 4,718,592

saving = C^2*dk = 524,288 MAC = 10%
```

profile 中只比较 recurrent kernel，document 为 4.234 ms，residual 为 3.845 ms，约 9.2% 时间减少，
与 10% MAC reduction 接近。总速度提升更大，是因为 residual 还融合掉了整个 W/U prepare kernel 和
W/U 的 global-memory round trip。

## 6. 输出公式

先定义 causal score：

```text
P_ij = <q_i,k_j> * gamma_i/gamma_j,  j <= i
P_ij = 0,                              j > i
```

则当前 chunk 输出为：

```text
O = scale * (Gamma * Q @ S + P @ Z)
scale = 1/sqrt(dk)
```

逐 token 展开：

```text
O_i = scale * [
    gamma_i * q_i @ S
    + sum_(j<=i) (<q_i,k_j> * gamma_i/gamma_j) * Z_j
]
```

第一项是 chunk 开始前已有 state 对当前 query 的贡献；第二项是当前 chunk 到 token `i` 为止的新写入
贡献。`j<=i` 的 lower-triangular mask 保证 token 不能看到未来 token。

注意 scale 只用于 output，不用于 state update。

## 7. State 更新公式

处理完当前 chunk 后，旧状态先衰减到 chunk 最后一个有效 token：

```text
S_decay = gamma_last * S
```

token `i` 的写入也要从它所在位置衰减到 chunk 末尾：

```text
Z_end_i = gamma_last/gamma_i * Z_i
```

因此：

```text
S_next = gamma_last * S + K^T @ Z_end
       = gamma_last * S
         + sum_i k_i^T @ (gamma_last/gamma_i * Z_i)
```

`S_next` 成为下一 chunk 的 `S`。这条依赖就是当前 kernel 必须 `T.serial(chunks_per_batch)` 的原因。

## 8. Python wrapper 和 launch mapping

`gdn_prefill_forward` 先读取：

```text
B, T, Hq, Hv
chunks_per_batch = ceil(T/64)
```

它只分配两个 global output：

```text
output      [B,T,Hv,128] BF16
final_state [B,Hv,128,128] FP32
```

然后专门化 JIT kernel 的 `H=Hv`、`Hg=Hq`、dtype 和是否存在 initial state。动态维度只有 B/T。

实际 launch 是：

```python
with T.Kernel(batch_size * H, threads=256) as (block,):
```

对应：

```text
grid.x  = B * Hv
block.x = 256 threads = 8 warps
```

block 映射：

```text
bb  = block // Hv
bh  = block % Hv
bhg = bh // (Hv/Hq)
```

也就是一个 block 独占一个 `(batch, value_head)` 的完整 `[128,128]` recurrent state，并从第一个
chunk 串行走到最后一个 chunk。TileLang 把 block 内的 `T.Parallel` 循环迭代分给 256 个物理线程；
它绝不表示启动 `64*128` 个线程。

## 9. Shared memory 与 fragment

每个 block 分配：

| buffer | shape/dtype | 逻辑大小 | 用途 |
| --- | --- | ---: | --- |
| `q_shared` | `[64,128]` BF16 | 16 KiB | Q Tensor Core operand |
| `k_shared` | `[64,128]` BF16 | 16 KiB | K Tensor Core operand |
| `v_shared` | `[64,128]` BF16 | 16 KiB | V |
| `a_shared` | `[64,64]` BF16 | 8 KiB | triangular inverse A |
| `z_shared` | `[64,128]` BF16 | 16 KiB | R/Z/decayed-Z Tensor Core operand |
| `state_shared` | `[128,128]` BF16 | 32 KiB | state 的 Tensor Core operand copy |
| `score_shared` | `[64,64]` BF16 | 8 KiB | causal QK score |
| `g_shared` | `[64]` FP32 | 256 B | log gamma |
| `beta_shared` | `[64]` FP32 | 256 B | beta |
| `g_last` | `[1]` FP32 | 4 B | chunk 最后有效 gate |

逻辑合计约 112.5 KiB；对齐和 driver reservation 后 NCU 报告 116,240 B/block。

fragment 是分布式 register tile：

```text
state [128,128] FP32
z     [64,128]  FP32
out   [64,128]  FP32
score [64,64]   FP32
```

这些矩阵不会在每个线程里各复制一份。TileLang 根据 GEMM layout 把 fragment element 分布到 warps 和
threads。最终 NCU 实测为 200 registers/thread，即 51,200 registers/block。register 和 shared
memory 都把 residency 限制为 1 block/SM。

state fragment 一直跨 chunk 保持 FP32；每个 chunk 开头复制为 BF16 `state_shared` 供 Tensor Core
读取。GEMM accumulation 使用 FP32。Z 在 element-wise/GEMM accumulator 中是 FP32，但复制到
`z_shared` 时转换为 BF16。这些 dtype 边界是当前实现的数值语义和误差来源之一。

## 10. 一个 chunk 的实际执行顺序

### 10.1 准备 state 与输入 tile

第一次进入 kernel：

```text
有 initial_state: global FP32 -> state fragment
无 initial_state: clear state fragment to zero
```

每个 chunk 开头：

```text
state fragment FP32 -> state_shared BF16
global Q/K/V/A BF16  -> corresponding shared buffers
global g/beta FP32   -> shared vectors
```

当前实现使用 `T.copy` 和 `T.Parallel`，没有显式 async copy 或 double buffer。尾块将无效 Q/K/V/A
row 填零，g/beta 也填零。`g_last` 取最后一个有效 token，而不是固定取 row 63。

### 10.2 GEMM 1：用 K 从旧 state 读取预测

```text
z = K @ S
[64,128] x [128,128] -> [64,128]
```

代码是：

```python
T.gemm(k_shared, state_shared, z, clear_accum=True)
```

此时 `z_i = k_i @ S`，是每个 token 基于 chunk 入口 state 的预测。

### 10.3 Element-wise：形成 raw residual R

```text
z_i = beta_i * (v_i - exp(g_i) * z_i)
```

现在 z fragment 被原地改写为 R。随后复制到 BF16 `z_shared`，作为下一次 Tensor Core GEMM operand。

### 10.4 GEMM 2：用 A 修正 chunk 内因果反馈

```text
z = A @ R
[64,64] x [64,128] -> [64,128]
```

这一步之后 z 才是公式中的 Z。再次写到 `z_shared`，后面的 output 和 state update 都复用它。

### 10.5 GEMM 3：旧 state 对 output 的贡献

```text
out = Q @ S
[64,128] x [128,128] -> [64,128]
out_i *= scale * exp(g_i)
```

`out` 是 FP32 accumulator，此时只包含 chunk 入口 state 的贡献。

### 10.6 GEMM 4：构造 QK score

```text
score = Q @ K^T
[64,128] x [128,64] -> [64,64]
```

随后 element-wise 做：

```text
if row >= col:
    score[row,col] *= scale * exp(g[row]-g[col])
else:
    score[row,col] = 0
```

这一步同时加入 causal mask、gate decay 和 `1/sqrt(dk)`。结果转换到 BF16 `score_shared`。

### 10.7 GEMM 5：当前 chunk 对 output 的贡献

```text
out += score @ Z
[64,64] x [64,128] -> [64,128]
```

代码使用 `clear_accum=False`，因此保留 GEMM 3 已经放在 out accumulator 中的旧 state contribution。
有效 row 最后写入 global BF16 output。

### 10.8 Element-wise：把旧 state 衰减到 chunk 末尾

```text
state *= exp(g_last)
```

state fragment 仍为 FP32。代码当前在 `[128,128]` 的 `T.Parallel` 循环表达这个操作；应检查生成代码
是否将相同的 `exp(g_last)` 提升复用，否则会产生大量重复 SFU 工作。

### 10.9 Element-wise：把 Z 衰减到 chunk 末尾

```text
z_i *= exp(g_last - g_i)
```

然后将 decayed Z 转为 BF16 `z_shared`。

### 10.10 GEMM 6：更新 state

```text
state += K^T @ z
[128,64] x [64,128] -> [128,128]
```

这里同样使用 `clear_accum=False`，把 rank-64 chunk update 累加到已经衰减的 FP32 state fragment。
随后进入下一个 `T.serial` chunk。所有 chunk 完成后才把 state 写到 global FP32 `final_state`。

## 11. 为什么 chunks 串行、chunk 内并行

必须串行的是：

```text
S_(c+1) depends on S_c
```

因此当前一个 block 内的 chunk loop 不能简单换成 `T.Parallel` 或 `T.Pipelined` 后同时计算多个完整
chunk。尤其 GEMM 1 (`K@S`) 和 GEMM 3 (`Q@S`) 必须等当前 `S` 可用。

可以提前并行/流水的是：

```text
load Q_(c+1), K_(c+1), V_(c+1), A_(c+1), g_(c+1), beta_(c+1)
```

这些输入不依赖 `S_c`。所以合理的 pipeline 是“当前 chunk 计算 + 下一 chunk 数据搬运”，而不是让
多个 chunk 同时更新同一个 state。

## 12. 精确的计算量

每 chunk/head 六次 GEMM：

| GEMM | MACs |
| --- | ---: |
| `K @ S` | `C*dk*dv` |
| `A @ R` | `C^2*dv` |
| `Q @ S` | `C*dk*dv` |
| `Q @ K^T` | `C^2*dk` |
| `score @ Z` | `C^2*dv` |
| `K^T @ Z` | `C*dk*dv` |

合计：

```text
3*C*dk*dv + C^2*dk + 2*C^2*dv
= 4,718,592 MAC/chunk/head
= 9,437,184 FLOP/chunk/head（每 MAC 按 2 FLOP）
```

还没有计入 exp2、乘法、mask、copy、dtype conversion、synchronization 和尾块 predicate。

## 13. 当前 profile 揭示的真正执行状态

以 `long_low_gva: B=1, T=32768, Hq=2, Hv=8` 为例：

```text
chunks/head = 32768/64 = 512
grid = B*Hv = 8 blocks
block = 256 threads = 8 warps
MIG SMs = 14
```

结果：

- 只有 8 个 SM 能拿到 block，另外 6 个 SM 从头到尾空闲。
- 每个 block 串行做 512 个 chunks，每 chunk 六次 GEMM。
- 200 registers/thread 和约 113.5 KiB shared memory 都限制为 1 block/SM。
- achieved occupancy 为 12.51%，waves/SM 为 0.57。
- kernel 约 3.845 ms；DRAM 约 52.5 GB/s，只达到实例 peak 的约 20.6%。
- memory-pipeline utilization 高于 compute，但两者都远未饱和；根因首先是 grid 太小和长 state chain。

因此“再减少一点 global load”或“把 occupancy 数字提高”不一定直接解决 low-head case。要让 14 个 SM
都工作，需要改变 state owner 的 block 分解，或得到能跨 chunk/子块并行组合的数学形式。

## 14. 从当前代码出发的优化检查表

1. `g_last` 的 `exp2` 是否被提升到循环外并在整个 state tile 复用？
2. `exp(g_i)` 和各种 gate ratio 是否可以每 token/score 预计算，减少 SFU 指令？
3. 下一 chunk 的 global-to-shared load 能否用 `T.Pipelined`/async copy/TMA 与当前 GEMM 重叠？
4. ping-pong buffer 增加的 shared memory 是否仍允许目标 occupancy？
5. `q/k/v/a/z/score/state` 的 shared layout 是否存在 bank conflict 或 Tensor Core operand replay？
6. TileLang 对六次 `T.gemm` 各自生成了 `mma` 还是 `wgmma`，shape 是否充分利用 N>=64 的 Hopper
   Tensor Core throughput？
7. `state FP32 fragment -> BF16 shared` 每 chunk 的 conversion/copy 能否减少或与其他操作重叠？
8. GVA 中同一个 Q/K head 被多个 value heads 重复加载，能否利用 L2、cluster DSM 或调整 block ownership？
9. 对 `B*Hv < 14` 的 case，能否把一个 state 的 dv/dk tile 拆给多个 cooperating blocks？同步和 reduction
   成本是否小于多用 SM 的收益？
10. 对 `B*Hv >= 14` 的 case，降低 registers/shared 以允许更多 blocks/SM 是否真正改善 stall reason？

这十项分别对应 SFU、pipeline、shared-memory capacity/layout、Tensor Core、GVA reuse 和 grid-level
parallelism。后续每次改动都应先明确它针对哪一项，再用相同 case 的 NCU/NSYS 数据验证。


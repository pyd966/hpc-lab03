# 当前 residual-first GDN 算法与 TileLang kernel 全流程

本文讲解当前 [student/tilelang_fwd.py](../student/tilelang_fwd.py) 中默认启用的 residual-first
实现。目标是同时回答三件事：它在数学上算什么、为什么与实验文档公式等价、每一步最终如何映射到
当前 GPU kernel。当前版本包含两项优化：第一，在每个 chunk 的输入准备阶段预计算 `gamma`、
`1/gamma` 和 `gamma_last`；第二，把独立的 value 列切成多个 `dv` slice，由多个 CTA 并行维护。

## 1. 问题、形状与符号

实验固定：

```text
chunk size C = 64
key/query dimension dk = 128
value dimension dv = 128
runtime value tile D in {128, 64, 32}
dv parts P = dv / D
```

对一个 batch `b`、一个 value head `h`、一个 chunk，省略 `b/h/chunk` 下标后：

| 符号 | 形状 | dtype in interface | 含义 |
| --- | ---: | --- | --- |
| `Q` | `[C, dk]` | BF16 | 当前 chunk query |
| `K` | `[C, dk]` | BF16 | 当前 chunk key，已 L2 normalize |
| `V` | `[C, dv]` | BF16 | 当前 chunk value |
| `g` | `[C]` | FP32 | `log(gamma)`，chunk-local gate prefix sum |
| `gamma=exp(g)` | `[C]` | FP32 shared | 从 chunk 开头到各 token 的累计 decay |
| `inv_gamma=1/gamma` | `[C]` | FP32 shared | gate ratio 的复用因子 |
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
乘法变成稳定的加法。kernel 收到的是 `g`，但当前版本在 chunk 开头统一执行：

```text
gamma_i     = exp2(g_i * log2(e))
inv_gamma_i = 1 / gamma_i
gamma_last  = gamma_(last valid token)
```

完整 chunk 的 `gamma_last` 直接读取 `gamma_shared[63]`；尾块为避免动态 shared-memory 下标的
静态越界警告，从最后一个全局 `g` 额外计算一次 `exp2`。这次额外计算每个 `(batch, head)` 最多发生
一次。代码中的 `LOG2E` 和 `T.exp2` 就是在做 `exp(g)`；后续所需的
`exp(g_i-g_j)` 和 `exp(g_last-g_i)` 分别改写为 `gamma_i*inv_gamma_j` 和
`gamma_last*inv_gamma_i`。

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
state_owners = B * Hv
```

它只分配两个 global output，接口 shape 没有因分片而改变：

```text
output      [B,T,Hv,128] BF16
final_state [B,Hv,128,128] FP32
```

wrapper 根据实测规则选择 value tile `D` 和 part 数 `P=128/D`：

```text
if mode == auto
   and chunks_per_batch >= 64
   and state_owners < 14
   and state_owners * 4 <= 28:
    D = 32, P = 4
else:
    D = 128, P = 1
```

`GDN_DV_SPLIT=off|64|32` 可以强制选择 `128x1/64x2/32x4`，只用于消融测试。JIT kernel
专门化 `H=Hv`、`Hg=Hq`、dtype、initial-state 分支、`D` 和 `P`；动态维度仍只有 B/T。

实际 launch 是：

```python
with T.Kernel(batch_size * H * dv_parts, threads=256) as (block,):
```

对应：

```text
grid.x  = B * Hv * P
block.x = 256 threads = 8 warps
```

block 映射：

```text
owner   = block // P
dv_part = block % P
bb      = owner // Hv
bh      = owner % Hv
bhg     = bh // (Hv/Hq)
dv_left = dv_part * D
```

一个 block 独占 `(batch, value_head, dv_part)` 的 `[128,D]` recurrent-state slice，并从第一个
chunk 串行走到最后一个 chunk。不同 part 写入互不重叠的
`output[...,dv_left:dv_left+D]` 和 `final_state[...,dv_left:dv_left+D]`，不需要跨 CTA reduction
或同步。TileLang 把 `T.Parallel` 迭代分给 256 个物理线程，它不表示为每个矩阵 element 启动一个线程。

## 9. Shared memory 与 fragment

令当前 CTA 的 value tile 宽度为 `D`。每个 block 分配：

| buffer | shape/dtype | 逻辑大小 | 是否随 D 变化 | 用途 |
| --- | --- | ---: | --- | --- |
| `q_shared` | `[64,128]` BF16 | 16 KiB | 否 | Q Tensor Core operand |
| `k_shared` | `[64,128]` BF16 | 16 KiB | 否 | K Tensor Core operand |
| `v_shared` | `[64,D]` BF16 | `128D` B | 是 | 当前 value-column slice |
| `a_shared` | `[64,64]` BF16 | 8 KiB | 否 | triangular inverse A |
| `z_shared` | `[64,D]` BF16 | `128D` B | 是 | R/Z/decayed-Z operand |
| `state_shared` | `[128,D]` BF16 | `256D` B | 是 | state slice 的 Tensor Core copy |
| `score_shared` | `[64,64]` BF16 | 8 KiB | 否 | causal QK score |
| `gamma_shared` | `[64]` FP32 | 256 B | 否 | 预计算的 `exp(g)` |
| `inv_gamma_shared` | `[64]` FP32 | 256 B | 否 | 预计算的 `1/exp(g)` |
| `beta_shared` | `[64]` FP32 | 256 B | 否 | beta |
| `gamma_last` | `[1]` FP32 | 4 B | 否 | 最后有效 token 的 `exp(g)` |

固定部分为 49,924 B，可变部分为 `512D` B：

| 配置 | source-level shared/block |
| --- | ---: |
| `D=128, P=1` | 115,460 B = 112.754 KiB |
| `D=64, P=2` | 82,692 B = 80.754 KiB |
| `D=32, P=4` | 66,308 B = 64.754 KiB |

fragment 是分布式 register tile：

```text
state [128,D]  FP32
z     [64,D]   FP32
out   [64,D]   FP32
score [64,64]  FP32
```

source-level fragment 数据总量为 `16 KiB + D KiB`。它们分布在 256 个线程的 registers 中，
并不是每个线程保存完整 tile。紧邻本次改动的同一 kernel 路径对照如下：

| 配置 | source fragment/block | NCU registers/thread | source shared/block | NCU dynamic shared/block | blocks/SM 上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 修改前 `D=128,P=1` | 144 KiB | 202 | 112.754 KiB | 115.47 KiB | 1 |
| 消融 `D=64,P=2` | 80 KiB | 未 profile | 80.754 KiB | 未 profile | 未 profile |
| 当前 split `D=32,P=4` | 48 KiB | 104 | 64.754 KiB | 66.32 KiB | 2 |

source 数字只统计显式 buffer；NCU 数字包含 backend layout/alignment，并反映编译后的寄存器
liveness，因此两列不应直接换算。以后每版同时报告这两种口径。

state fragment 跨 chunk 保持 FP32；每个 chunk 开头复制为 BF16 `state_shared`。Z/out GEMM
accumulation 使用 FP32，`z_shared` 和 `score_shared` 是 BF16 operand 边界。这些 dtype 转换没有
因 dv 分片而改变。

## 10. 一个 chunk 的实际执行顺序

### 10.1 准备 state 与输入 tile

第一次进入 kernel：

```text
有 initial_state: global FP32 state[...,dv_left:dv_left+D] -> state fragment
无 initial_state: clear [128,D] state fragment to zero
```

每个 chunk 开头：

```text
state fragment [128,D] FP32 -> state_shared [128,D] BF16
global Q/K/A BF16            -> full shared tiles (duplicated by every part)
global V[...,dv_left:+D]      -> v_shared [64,D] (partitioned)
global g/beta FP32            -> shared gate vectors (duplicated by every part)
gamma_shared[i]      = exp2(global_g[i] * LOG2E)
inv_gamma_shared[i]  = 1 / gamma_shared[i]
gamma_last           = gamma_shared[63] or exp2(global tail g * LOG2E)
```

`state` 使用 `T.copy`，Q/K/V/A/g/beta 使用 `T.Parallel`。这里没有 `T.async_copy`、TMA、
`T.Pipelined`、double buffer 或跨 chunk prefetch；所有当前 chunk 的装载和 gate 预计算结束后才开始
GEMM 1。这里仍然没有跨 chunk prefetch。拆分后 Q/K/A/g/beta 和 gate 预计算会被每个 part 重复，
V/state/output/final_state 则按 value 列分片。尾块将无效 Q/K/V/A row 填零，将无效
`gamma/inv_gamma` 设为 1、`beta` 设为 0；`gamma_last` 始终对应最后一个有效 token。

### 10.2 GEMM 1：用 K 从旧 state 读取预测

```text
z = K @ S
[64,128] x [128,D] -> [64,D]
```

代码是：

```python
T.gemm(k_shared, state_shared, z, clear_accum=True)
```

此时 `z_i = k_i @ S`，是每个 token 基于 chunk 入口 state 的预测。

### 10.3 Element-wise：形成 raw residual R

```text
z_i = beta_i * (v_i - gamma_i * z_i)
```

这里的 `gamma_i` 读取 `gamma_shared`，不再执行 `exp2`。z fragment 被原地改写为 R，随后转换并复制
到 BF16 `z_shared`，作为下一次 Tensor Core GEMM operand。

### 10.4 GEMM 2：用 A 修正 chunk 内因果反馈

```text
z = A @ R
[64,64] x [64,D] -> [64,D]
```

这一步之后 z 才是公式中的 Z。再次写到 `z_shared`，后面的 output 和 state update 都复用它。

### 10.5 GEMM 3：旧 state 对 output 的贡献

```text
out = Q @ S
[64,128] x [128,D] -> [64,D]
out_i *= scale * gamma_i
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
    score[row,col] *= scale * gamma[row] * inv_gamma[col]
else:
    score[row,col] = 0
```

这一步同时加入 causal mask、gate decay 和 `1/sqrt(dk)`。结果转换到 BF16 `score_shared`。

### 10.7 GEMM 5：当前 chunk 对 output 的贡献

```text
out += score @ Z
[64,64] x [64,D] -> [64,D]
```

代码使用 `clear_accum=False`，因此保留 GEMM 3 已经放在 out accumulator 中的旧 state contribution。
有效 row 最后写入 global BF16 `output[...,dv_left:dv_left+D]`。

### 10.8 Element-wise：把旧 state 衰减到 chunk 末尾

```text
state *= gamma_last
```

state fragment 仍为 FP32。`gamma_last` 已在 chunk 准备阶段形成一个 FP32 shared 标量，
`[128,D]` 的 `T.Parallel` 循环只读取并乘上该值，不再重复执行 `exp2`。

### 10.9 Element-wise：把 Z 衰减到 chunk 末尾

```text
z_i *= gamma_last * inv_gamma_i
```

然后将 decayed Z 转为 BF16 `z_shared`。

### 10.10 GEMM 6：更新 state

```text
state += K^T @ z
[128,64] x [64,D] -> [128,D]
```

这里同样使用 `clear_accum=False`，把 rank-64 chunk update 累加到已经衰减的 FP32 state fragment。
随后进入下一个 `T.serial` chunk。所有 chunk 完成后，CTA 才把自己的 `[128,D]` state slice
写到 global FP32 `final_state[...,dv_left:dv_left+D]`。

## 11. 为什么 chunks 串行、chunk 内并行

必须串行的是：

```text
S_(c+1) depends on S_c
```

因此每个 `(batch, head, dv_part)` block 内的 chunk loop 不能简单换成 `T.Parallel` 或
`T.Pipelined` 后同时计算多个完整 chunk。dv 分片只并行独立的 value 列；GEMM 1 (`K@S`) 和
GEMM 3 (`Q@S`) 仍必须等待本 part 的当前 `S`。

可以提前并行/流水的是：

```text
load Q_(c+1), K_(c+1), V_(c+1), A_(c+1), g_(c+1), beta_(c+1)
```

这些输入不依赖 `S_c`。所以合理的 pipeline 是“当前 chunk 计算 + 下一 chunk 数据搬运”，而不是让
多个 chunk 同时更新同一个 state。

## 12. 精确的计算量

令一个 CTA 的 value tile 为 `D`，每个 head 有 `P=128/D` 个 CTA。单个 CTA 每 chunk 的六次 GEMM：

| GEMM | MACs/CTA |
| --- | ---: |
| `K @ S` | `C*dk*D` |
| `A @ R` | `C^2*D` |
| `Q @ S` | `C*dk*D` |
| `Q @ K^T` | `C^2*dk` |
| `score @ Z` | `C^2*D` |
| `K^T @ Z` | `C*dk*D` |

因此：

```text
MAC/CTA/chunk  = 3*C*dk*D + C^2*dk + 2*C^2*D
MAC/head/chunk = P * MAC/CTA/chunk
               = 3*C*dk*dv + 2*C^2*dv + P*C^2*dk
```

value-dependent work总量不变，只有 `Q@K^T` 被每个 part 重复：

| 配置 | MAC/CTA/chunk | MAC/head/chunk | 相对额外 MAC |
| --- | ---: | ---: | ---: |
| `D=128,P=1` | 4,718,592 | 4,718,592 | 0% |
| `D=64,P=2` | 2,621,440 | 5,242,880 | 11.1% |
| `D=32,P=4` | 1,572,864 | 6,291,456 | 33.3% |

gamma 操作也会按 P 重复。下表是未拆分 `D=128,P=1` 时，一个完整 chunk/head 的源码逻辑
element 数；拆分时把每个 value-dependent 行按 `D/128` 缩小后再乘 P，总量不变，而 score 和 chunk
准备行会乘 P：

| gate 操作位置 | gamma 优化前 | gamma 优化后 |
| --- | ---: | ---: |
| residual `gamma_i` | 8,192 次 `exp2` | 8,192 次 shared read/multiply |
| output state contribution `gamma_i` | 8,192 次 `exp2` | 8,192 次 shared read/multiply |
| lower-triangular score ratio | 2,080 次 `exp2` | 2,080 次 `gamma*inv_gamma` |
| state tile `gamma_last` | 16,384 次 `exp2` | 16,384 次 shared scalar multiply |
| Z decay ratio | 8,192 次 `exp2` | 8,192 次 `gamma_last*inv_gamma` |
| chunk 准备 | 0 | 64 次 `exp2` + 64 次 reciprocal |

因此 dv split 是用重复的 QK/gate/A/Q/K global work 换取更多 CTA 和更小的 register/shared footprint，
只应在原 grid 严重不足且 chunk 链足够长时启用。

## 13. 优化前 profile 揭示的执行状态

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
- 该较早 profile 中 registers 和 shared memory 都把 residency 限制为 1 block/SM；紧邻 dv split
  版本的精确资源对照见第 9 节。
- achieved occupancy 为 12.51%，waves/SM 为 0.57。
- kernel 约 3.845 ms；DRAM 约 52.5 GB/s，只达到实例 peak 的约 20.6%。
- memory-pipeline utilization 高于 compute，但两者都远未饱和；根因首先是 grid 太小和长 state chain。

dv 分片正是对 state owner 的 block 分解，但 profile 也说明不能只追求更多 block：owner=8 时强制
拆分会因重复 QK/A/gate 工作而显著退化，所以最终必须按 owner 数和 chunk 链长度选择性启用。

## 14. 优化 1：预计算 gamma、1/gamma 与 gamma_last

### 14.1 实现与同步边界

每个 chunk 的 gate 装载循环现在完成三件事：从 global FP32 `g/beta` 读取有效 token，计算并写入
`gamma_shared`，再计算 `inv_gamma_shared`。这一步结束后，后续五处 gate 使用都变成 shared load
加普通乘法。完整 chunk 的 `gamma_last` 在该并行循环结束后由所有线程统一执行的控制流读取 row 63；
尾块则对全局最后一个 `g` 提前做一次 `exp2`。

曾尝试让最后一个有效 token 在 `T.Parallel` 的条件分支内直接写 `gamma_last`。该版本能编译但首个
kernel launch 不返回，说明当前 TileLang lowering 在这类分歧 shared-memory 路径上存在同步风险。
最终版本把标量写入放回并行循环之后的统一控制流，避免让部分线程绕过潜在的 block barrier。

这项优化没有改变 launch grid、block threads、六次 GEMM、fragment shape 或 global output 分配；只把
`g_shared[64]` 替换为 `gamma_shared[64] + inv_gamma_shared[64]`，shared memory 逻辑用量增加 256 B。

### 14.2 测试方法

- 设备：`NVIDIA H800 PCIe MIG 1g.10gb`，lab3 分区，1 GPU、8 CPU、32 GiB host memory。
- 命令：`./job.sh --output-format csv`，即每个 case 10 次 warmup、100 次重复并取 CUDA event 中位数。
- 计时范围：仅学生函数内部核心 forward，包含输出分配和一次 recurrent kernel launch，不含 g/A 预处理。
- 基线日志：`output/gamma_precompute_baseline_56516.log`。
- 最终日志：`output/gamma_precompute_final_56568.log`。
- 正确性：8 个 case 的 BF16 output 和 FP32 final state 均为 `PASS`。

### 14.3 时间、speedup 与预计分数

| case | baseline (ms) | optimized (ms) | speedup | `p=t100/t` | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.147712 | 0.144288 | 1.0237x | 2.3983 | 120.00 |
| `chain_equal` | 0.947344 | 0.922480 | 1.0270x | 0.5396 | 78.97 |
| `parallel_equal` | 0.563232 | 0.545008 | 1.0334x | 0.9381 | 96.68 |
| `parallel_gva` | 0.504080 | 0.495856 | 1.0166x | 0.9913 | 99.53 |
| `long_low_gva` | 3.766192 | 3.676608 | 1.0244x | 0.5056 | 77.59 |
| `batch_split_gva` | 2.913376 | 2.829680 | 1.0296x | 0.5414 | 78.79 |
| `wide_gva_state` | 5.264464 | 5.085760 | 1.0351x | 0.4771 | 76.29 |
| `deep_gva_state` | 5.970896 | 5.777808 | 1.0334x | 0.4900 | 76.70 |

公开 8 case 的预计简单平均为 **88.07 分**。实验文档只公布 60/100 分 turning point，没有给出完整
插值函数，因此这里明确采用以下估算曲线：

```text
p <= p60:       score = 60 * p / p60
p60 < p <= 1:   score = 60 + 40 * (p-p60) / (1-p60)
p > 1:          score = min(120, 100 + 20 * (p-1))
```

这只是公开 case 估计，不包含占最终分数 40% 的隐藏 case。8 个 case 均有 1.0166x--1.0351x 提升；
收益不如源码层面的 `exp2` 数量降幅大，说明六次 Tensor Core GEMM、shared/register movement 以及
编译器原有的部分公共子表达式处理仍占主要时间。

## 15. 优化 2：沿 dv 拆分 recurrent state

### 15.1 独立性与实现

固定一个 `(batch, value_head)` 后，state 的每个 value 列只参与同列的 `K@S`、residual、
`A@R`、`Q@S`、`score@Z` 和 `K^T@Z`。不同列之间没有 reduction 或数据依赖，因此可以把
`dv=128` 切成 P 个连续 slice：

```text
CTA p owns columns [p*D, (p+1)*D)
state fragment/shared = [128,D]
V/Z/out                = [64,D]
```

Q/K/A、score 和 gate 与 value 列无关，但为了避免 CTA 间同步，它们由每个 part 独立加载和计算。
initial state、V、output 和 final state 按 `dv_left` 切片，所有 global writes 都是 disjoint 的。

自动分派选择 `D=32,P=4` 的条件是：chunk 数至少 64、原 state owner 少于 14，并且拆后 grid 不超过
28 blocks；其余情况使用 `D=128,P=1`。这不是对公开 case 名称的硬编码，而是针对 H800 MIG 14 SM
和重复计算成本的 shape 规则。

### 15.2 并行度扫描

第一轮使用 5 warmups、20 repetitions，覆盖 initial-state+尾块、4-owner MHA 和 8-owner GVA：

| 配置 | `short_tail_state` ms | `chain_equal` ms | `long_low_gva` ms | 结论 |
| --- | ---: | ---: | ---: | --- |
| `D=128,P=1` | 0.145632 | 0.938128 | 3.803824 | 基准 |
| `D=64,P=2` | 0.235648 | 0.887072 | 6.712880 | chain 小幅受益，其余显著退化 |
| `D=32,P=4` | 0.215920 | 0.807248 | 6.587808 | chain 最优，其余显著退化 |
| `D=16,P=8` | FAIL | FAIL | FAIL | 默认 fragment layout 在 N=16 时结果错误，移除 |

对 `chain_equal` 再用正式 10/100 协议复测：

| 配置 | grid blocks | median ms | 相对 `128x1` |
| --- | ---: | ---: | ---: |
| `128x1` | 4 | 0.925408 | 1.0000x |
| `64x2` | 8 | 0.876640 | 1.0556x |
| `32x4` | 16 | 0.793184 | **1.1667x** |

`32x4` 最好，因为它把 4-block grid 扩到 16 blocks，基本覆盖 14 个 SM；继续缩到 N=16 不仅增加
重复工作，当前 TileLang layout 也不满足正确性。

原始扫描日志为 `output/dv_split_scan_{off,64,32,16}_*.log`，正式 chain 日志为
`output/dv_split_chain_{off,64,32}_*.log`。

### 15.3 NCU 资源验证

对 `chain_equal` 的修改前 `residual_mha_dv128x1` 和当前 `residual_mha_dv32x4` 进行 NCU basic
profile，二者使用相同的 256-thread block：

| 指标 | 修改前 `D=128,P=1` | 当前 `D=32,P=4` |
| --- | ---: | ---: |
| grid blocks | 4 | 16 |
| registers/thread | 202 | 104 |
| dynamic shared/block | 115.47 KiB | 66.32 KiB |
| theoretical blocks/SM | 1 | 2 |
| theoretical occupancy | 12.50% | 25.00% |
| achieved occupancy | 12.50% | 14.34% |
| achieved active warps/SM | 8.00 | 9.18 |
| profiled duration | 0.912 ms | 0.770 ms |
| compute / memory throughput | 6.20% / 17.74% | 20.25% / 42.48% |

报告分别保存在 `output/ncu_dv_split_off_chain.ncu-rep` 和 `output/ncu_dv_split_chain.ncu-rep`。
grid 从 4 增到 16 是主要收益来源；寄存器和 shared 缩小也把理论 residency 从 1 block/SM 提到 2，
但本 case 只有 16 blocks，尚不足以在所有 14 个 SM 同时放满两块。

### 15.4 最终 8-case 结果

下面以优化 1 的 gamma-precompute 版本为 baseline。最终自动分派使用默认 10 warmups、100
repetitions，所有 output/final-state 正确性检查均通过：

| case | gamma baseline ms | dv auto ms | speedup | `p=t100/t` | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.144288 | 0.144560 | 0.9981x | 2.3938 | 120.00 |
| `chain_equal` | 0.922480 | 0.792768 | **1.1636x** | 0.6279 | 83.00 |
| `parallel_equal` | 0.545008 | 0.550848 | 0.9894x | 0.9281 | 96.15 |
| `parallel_gva` | 0.495856 | 0.497872 | 0.9960x | 0.9873 | 99.31 |
| `long_low_gva` | 3.676608 | 3.672976 | 1.0010x | 0.5061 | 77.61 |
| `batch_split_gva` | 2.829680 | 2.834256 | 0.9984x | 0.5406 | 78.75 |
| `wide_gva_state` | 5.085760 | 5.091840 | 0.9988x | 0.4766 | 76.27 |
| `deep_gva_state` | 5.777808 | 5.791568 | 0.9976x | 0.4888 | 76.64 |

预计分数沿用优化 1 中声明的分段线性估算。公开 8 case 的预计简单平均从 88.07 提升到
**88.47 分**；未选择 split 的 case 变化在约 -1.1% 到 +0.1%，属于跨任务时钟/测量波动范围。
最终完整日志为 `output/dv_split_auto_final_56860.log`。

## 16. 从当前代码出发的优化检查表

1. 下一 chunk 的 global-to-shared load 能否用 `T.Pipelined`/async copy/TMA 与当前 GEMM 重叠？
2. ping-pong buffer 增加的 shared memory 是否仍允许目标 occupancy？
3. `q/k/v/a/z/score/state` 的 shared layout 是否存在 bank conflict 或 Tensor Core operand replay？
4. TileLang 对六次 `T.gemm` 各自生成了 `mma` 还是 `wgmma`，shape 是否充分利用 N>=32 的 Hopper
   Tensor Core throughput？
5. `state FP32 fragment -> BF16 shared` 每 chunk 的 conversion/copy 能否减少或与其他操作重叠？
6. GVA 中同一个 Q/K head 被多个 value heads/parts 重复加载，能否利用 L2、cluster DSM 或调整 ownership？
7. 对 `B*Hv >= 14` 的 case，降低 registers/shared 以允许更多 blocks/SM 是否真正改善 stall reason？

这些项目分别对应 pipeline、shared-memory capacity/layout、Tensor Core、GVA reuse 和 grid-level
parallelism。后续每次改动都应先明确它针对哪一项，再用相同 case 的 benchmark 与 NCU/NSYS 数据验证。


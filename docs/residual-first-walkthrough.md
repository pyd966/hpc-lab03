# 当前 residual-first GDN 算法与 TileLang kernel 全流程

本文讲解当前 [student/tilelang_fwd.py](../student/tilelang_fwd.py) 中默认启用的 residual-first
实现。目标是同时回答三件事：它在数学上算什么、为什么与实验文档公式等价、每一步最终如何映射到
当前 GPU kernel。

本文对应提交 `57a2ace`。特别地，当前版本已经实现 gate 指数缓存和两阶段输入预取；文中会明确区分
源码里的逻辑 buffer、`T.Pipelined` lowering 后的物理 ping-pong buffer，以及 FP32 register state。

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

在加入本轮 gate 缓存和 pipeline 之前，只比较 recurrent kernel 时，document 为 4.234 ms、residual
为 3.845 ms，约减少 9.2%，与 10% MAC reduction 接近。总速度提升更大，因为 residual 还融合了 W/U prepare kernel 和
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

`S_next` 成为下一 chunk 的 `S`。这条依赖要求 recurrent compute 按 chunk 顺序提交。当前源码使用
`T.Pipelined`，但只有不依赖 state 的下一 chunk 输入 load 进入前一 stage；六次 GEMM 和 state update
仍全部位于同一个依赖 stage，语义上依然是串行 state chain。

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

然后专门化 JIT kernel 的 `H=Hv`、`Hg=Hq`、dtype、是否存在 initial state，以及两个分类参数：

```text
low_parallel = (B * Hv < 14)
gva_variant  = (Hv != Hq)
```

`low_parallel` 在 wrapper 中选择 `tilelang_residual_first_low_parallel` 或
`tilelang_residual_first_normal`；`gva_variant` 区分 GVA/MHA kernel 名。当前 checkpoint 中四个名字最终仍
进入同一个 kernel body，尚未改变 grid、tile 或算法。这是为后续真正分流保留的 specialization 边界，
不能把“生成了不同 kernel 名”误解为“已经执行不同优化”。

动态维度只有 B/T。`H/Hg`、dtype 和 `use_initial_state` 都会触发独立 JIT specialization。

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

每个 block 在源码中声明的逻辑 shared allocations 如下：

| buffer | shape/dtype | 单 stage 大小 | 是否被 pipeline 双缓冲 | 用途 |
| --- | --- | ---: | --- | --- |
| `q_shared` | `[64,128]` BF16 | 16 KiB | 是 | Q Tensor Core operand |
| `k_shared` | `[64,128]` BF16 | 16 KiB | 是 | K Tensor Core operand |
| `v_shared` | `[64,128]` BF16 | 16 KiB | 是 | V |
| `a_shared` | `[64,64]` BF16 | 8 KiB | 是 | triangular inverse A |
| `g_shared` | `[64]` FP32 | 256 B | 是 | log gamma |
| `beta_shared` | `[64]` FP32 | 256 B | 是 | beta |
| `z_shared` | `[64,128]` BF16 | 16 KiB | 否 | R/Z/decayed-Z operand |
| `state_shared` | `[128,128]` BF16 | 32 KiB | 否 | 当前 state 的 GEMM operand |
| `score_shared` | `[64,64]` BF16 | 8 KiB | 否 | causal QK score |
| `g_exp_shared` | `[64]` FP32 | 256 B | 否 | `gamma_i=exp(g_i)` |
| `g_inv_exp_shared` | `[64]` FP32 | 256 B | 否 | `1/gamma_i` |
| `g_last_exp` | `[1]` FP32 | 4 B | 否 | 最后一个有效 token 的 gamma |

如果只把源码中的每个数组算一次，逻辑合计为：

```text
Q/K/V             48 KiB
A/Z/state/score   64 KiB
四个 FP32 vector   1 KiB
g_last_exp         4 B
------------------------
base              113.004 KiB
```

`T.Pipelined` 把 stage-0 producer 的 Q/K/V/A/g/beta 变成两套物理 buffer，额外增加：

```text
16 + 16 + 16 + 8 + 0.25 + 0.25 = 56.5 KiB
```

因此 lowering 后的理论 payload 约为 `169.504 KiB = 173,572 B`。NCU 的 `Kbyte` 使用十进制单位，
实测 dynamic shared memory 是 `173.584 kB = 173,584 B = 169.516 KiB/block`，另有
`1.024 kB = 1 KiB/block` driver shared memory。12 B 差额来自 alignment/compiler bookkeeping。
169.516 KiB 已经使 shared-memory block limit 等于 1。

fragment 是分布式 register tile：

```text
state [128,128] FP32 = 16,384 registers = 64 registers/thread
z     [64,128]  FP32 =  8,192 registers = 32 registers/thread
out   [64,128]  FP32 =  8,192 registers = 32 registers/thread
score [64,64]   FP32 =  4,096 registers = 16 registers/thread
---------------------------------------------------------------
纯 payload                         = 144 registers/thread
```

这些矩阵不会在每个线程里各复制一份。TileLang 根据 GEMM layout 把 fragment elements 分布给 256 个
threads；地址、索引、predicate、pipeline state 和临时 operand 还需要额外 registers。当前 NCU 实测
为 `197 registers/thread`，即至少 `50,432` 个 32-bit registers/block，没有 local/shared spill。
register limit 和 shared-memory limit 都只允许 1 block/SM。

state fragment 在整个 chunk chain 中保持 FP32；每个 chunk 开头才转换为 BF16 `state_shared`，供
Tensor Core 作为 input operand。所有 `T.gemm` accumulator 是 FP32。R/Z 和 score 在 fragment
中是 FP32，但写入 `z_shared`/`score_shared` 时变为 BF16，然后被下一个 GEMM 读取。这些转换点既
降低 shared-memory 容量，也是当前算法相对纯 FP32 reference 的主要舍入边界。

## 10. 一个 chunk 的实际执行顺序

### 10.1 准备 state 与输入 tile

第一次进入 kernel：

```text
有 initial_state: global FP32 -> state fragment
无 initial_state: clear state fragment to zero
```

每个 pipeline iteration 分成两类操作：

```text
stage 0（与 state 无关，可预取）
    global Q/K/V/A/g/beta -> ping-pong shared buffer

stage 1（依赖当前 S，严格按 chunk 顺序）
    state FP32 fragment -> state_shared BF16
    gate exponent preprocessing
    六次 GEMM、element-wise、output store、state update
```

源码只声明一份 `q_shared` 等名字，但 `T.Pipelined` 根据 stage annotation 为 stage-0 producers
生成两套物理 storage。稳态下，在 stage 1 消费 chunk `c` 的一套 buffer 时，另一套可以准备
chunk `c+1`。pipeline prologue 先装入第一个 chunk，epilogue 则排空最后一个 chunk。

state 不能进入 stage 0，因为 `state_shared(c)` 必须来自 `S_c`，而 `S_c` 直到 chunk `c-1`
的最后一次 `K^T@Z` 完成后才存在。于是当前 ping-pong 隐藏的是 global-to-shared input latency，
不是 chunk recurrence。

输入可用后，每个 token 只做一次指数并缓存：

```text
gamma_i     = exp2(g_i * log2(e))
inv_gamma_i = 1 / gamma_i
gamma_last  = gamma_63                    # 完整 chunk
gamma_last  = exp(g_[last valid token])   # tail chunk
```

后面的三类比例都改为乘缓存值：

```text
exp(g_i)              -> gamma_i
exp(g_row-g_col)      -> gamma_row * inv_gamma_col
exp(g_last-g_i)       -> gamma_last * inv_gamma_i
```

因此每个完整 chunk/head 只有 64 次 `exp2` 和 64 次 reciprocal，不再为每个 output element、
score element 或 state element 重复调用指数函数。尾块仍使用固定 64-row tile；越界 load 由 copy
lowering 做 predicate/zero-fill，residual 和 output store 还显式检查 `left+token<num_tokens`。
`gamma_last` 始终来自最后一个有效 token，不会错误地读取 row 63。

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
z_i = beta_i * (v_i - gamma_i * z_i)
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
    score[row,col] *= scale * gamma_row * inv_gamma_col
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
state *= gamma_last
```

state fragment 仍为 FP32。`gamma_last` 已经在 shared 中缓存为一个 FP32 scalar；`[128,128]` 的
`T.Parallel` 循环只执行乘法，不再对每个 state element 调用 `exp2`。这一步更新的是长期保存的 FP32
state，而不是前面用于 GEMM input 的 BF16 `state_shared` 副本。

### 10.9 Element-wise：把 Z 衰减到 chunk 末尾

```text
z_i *= gamma_last * inv_gamma_i
```

然后将 decayed Z 转为 BF16 `z_shared`。

### 10.10 GEMM 6：更新 state

```text
state += K^T @ z
[128,64] x [64,128] -> [128,128]
```

这里同样使用 `clear_accum=False`，把 rank-64 chunk update 累加到已经衰减的 FP32 state fragment。
随后 pipeline 才能让下一个 chunk 的 recurrent stage 消费这个新 state。所有 chunks 完成后，kernel
只执行一次 `state -> final_state` 的 global FP32 store。

## 11. 为什么 chunks 串行、chunk 内并行

必须串行的是：

```text
S_(c+1) depends on S_c
```

因此不能把依赖 state 的 recurrent statements 放到同一个 pipeline 的不同前置 stage，让多个完整
chunks 同时执行。尤其 GEMM 1 (`K@S`) 和 GEMM 3 (`Q@S`) 必须读取同一个刚刚完成的 `S_c`，
GEMM 6 又必须成为下一 iteration 的 producer。

当前 `T.Pipelined` 合法，是因为它只移动以下与 `S_c` 无关的 producer：

```text
load Q_(c+1), K_(c+1), V_(c+1), A_(c+1), g_(c+1), beta_(c+1)
```

稳态时间线是：

```text
time ---->

shared set 0:  load chunk 0 | compute chunk 0 | load chunk 2 | compute chunk 2
shared set 1:               | load chunk 1    | compute chunk 1 | load chunk 3
state chain:      S0 -------> S1 -------------> S2 -------------> S3
```

这里的图只表达 storage ping-pong；`compute chunk 0/1/2` 仍然按 state arrows 串行。实现没有创建第二
份 state，也没有做跨 chunk prefix scan。换句话说，当前 pipeline 重叠“输入搬运”和“已有 state 上的
计算”，没有重叠两个 state transitions。

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

GEMM 之外，每个完整 chunk/head 还包含：

| 操作 | logical elements |
| --- | ---: |
| `gamma=exp(g)` | 64 exp2 |
| `inv_gamma=1/gamma` | 64 reciprocal |
| residual formation | `C*dv = 8,192` |
| old-state output scaling | `C*dv = 8,192` |
| score gate/mask | `C^2 = 4,096` |
| FP32 state decay | `dk*dv = 16,384` |
| Z-to-chunk-end decay | `C*dv = 8,192` |
| output store | `C*dv = 8,192` |

还要加上 global/shared copies、FP32-to-BF16 conversion、six-GEMM synchronization、pipeline control 和
尾块 predicates。这里的 logical element 数不等于机器指令数：`T.Parallel` 会把 iteration domain
分给 256 threads，`T.gemm` 则会进一步 lower 成 warp-group Tensor Core instructions。当前每个
chunk/head 的主算法工作量仍是 4,718,592 MAC；gate 缓存改变的是额外 SFU 工作，不改变这些 MAC。

## 13. 当前 profile 揭示的真正执行状态

以 `long_low_gva: B=1, T=32768, Hq=2, Hv=8` 为例：

```text
chunks/head = 32768/64 = 512
grid = B*Hv = 8 blocks
block = 256 threads = 8 warps
MIG SMs = 14
```

对应提交 `57a2ace` 的 NCU 数据：

| 指标 | 当前值 | 直接含义 |
| --- | ---: | --- |
| grid | 8 blocks | 最多只使用 8/14 个 SM |
| block | 256 threads = 8 warps | 每个 active SM 只有 8 resident warps |
| registers/thread | 197 | `50,432 registers/block`，register block limit=1 |
| dynamic shared/block | 173,584 B = 169.52 KiB | shared-memory block limit=1 |
| achieved occupancy | 12.50% | 8/64 warps |
| waves/SM | 0.57 | 整个 grid 小于 SM 数 |
| No Eligible | 76.71% | scheduler 经常找不到可发射 warp |
| SM throughput | 13.35% | 计算管线没有接近饱和 |
| DRAM throughput | 22.83% | 也不是 HBM bandwidth bound |
| L1/TEX throughput | 58.19% | active blocks 内 shared/L1 activity 较显著 |
| local/shared spill | 0 B | 高 register 数尚未造成 spill |
| profiled duration | 3.52 ms | NCU instrumentation 下的单次 duration |
| stable benchmark | 3.42--3.45 ms | 非 profile 的 median 范围 |

因此当前低并行 case 的第一问题不是 HBM peak，也不是单纯把 occupancy 数字从 12.5% 改到 25%。
`512 threads/block` 实验虽然把 active warps 翻倍并降低 long-scoreboard stall，却让 TileLang 生成
53.5% 更多 instructions；grid 仍是 8，六个 SM 仍空闲，最后反而慢约 8%。要获得新的 Tensor Core
资源，必须增加独立 blocks，或者使用能并行组合 chunk transitions 的数学形式。

## 14. End-to-end 伪代码

下面把 wrapper、block ownership、pipeline 和一个 chunk 内的所有数据变换压缩到同一份伪代码中。
所有矩阵表达式都针对一个 `(batch, value_head)`：

```text
function gdn_prefill_forward(Q, K, V, g, beta, A, initial_state):
    B, T, Hq, dk = shape(Q)
    Hv, dv = V.heads, V.value_dim
    n_chunks = ceil(T / 64)

    allocate output[B,T,Hv,dv] BF16
    allocate final_state[B,Hv,dk,dv] FP32

    select kernel name from:
        low_parallel = (B*Hv < 14)
        gva_variant  = (Hv != Hq)
    # 当前四种名字的 body 相同

    launch grid=B*Hv blocks, 256 threads/block

kernel(block):
    bb  = block // Hv
    bh  = block % Hv
    bhg = bh // (Hv/Hq)

    if initial_state exists:
        state[dk,dv] FP32 fragment = initial_state[bb,bh]
    else:
        state = 0

    pipeline over chunk c:
        stage 0:
            prefetch Q[c,bb,bhg], K[c,bb,bhg]
            prefetch V[c,bb,bh], A[c,bb,bh]
            prefetch g[c,bb,bh], beta[c,bb,bh]
            use the alternate shared-memory buffer set

        stage 1:
            state_shared BF16 = cast(state FP32)

            gamma[i]     = exp2(g[i] * log2(e))
            inv_gamma[i] = 1 / gamma[i]
            gamma_last   = gamma[last_valid]

            # GEMM 1: prediction from entry state
            P = K @ state_shared

            # FP32 residual, invalid tail rows become zero
            R[i,:] = beta[i] * (V[i,:] - gamma[i] * P[i,:])

            # BF16 boundary, then GEMM 2
            Z = A @ cast_bf16(R)

            # GEMM 3: output contribution from entry state
            O = Q @ state_shared
            O[i,:] *= scale * gamma[i]

            # GEMM 4 and causal/gate epilogue
            score = Q @ K^T
            score[i,j] =
                scale * score[i,j] * gamma[i] * inv_gamma[j], if j <= i
                0,                                              otherwise

            # BF16 boundaries, then GEMM 5
            O += cast_bf16(score) @ cast_bf16(Z)

            if token i is valid:
                output[bb, token_i, bh, :] = cast_bf16(O[i,:])

            # State transition to chunk end
            state[:,:] *= gamma_last
            Z_end[i,:] = Z[i,:] * gamma_last * inv_gamma[i]

            # BF16 boundary, then GEMM 6 with FP32 accumulation
            state += K^T @ cast_bf16(Z_end)

    final_state[bb,bh,:,:] = state FP32
    return
```

这份伪代码中，`P/R/Z/O/score/Z_end` 是逻辑名字；实际代码复用 `z` fragment 依次承载
`P -> R -> Z -> Z_end`，复用 `out` 作为两个 output GEMM 的 FP32 accumulator。这样避免了额外
global intermediates，但也形成了必须遵守的 overwrite 顺序。

## 15. 数据所有权、同步和正确性不变量

当前 kernel 的关键不变量如下：

1. 一个 block 在整个 kernel lifetime 内唯一拥有一个完整 `S[128,128]`；其他 block 不读写它。
2. chunks 按时间顺序更新同一份 FP32 state；只有输入 shared buffers 可以跨 iteration ping-pong。
3. `state_shared` 是当前 FP32 state 的 BF16 snapshot。它同时服务 `K@S` 和 `Q@S`，但之后的
   `K^T@Z` 更新必须落回 FP32 fragment。
4. `z_shared` 被三次覆盖：先写 R、再写 Z、最后写 `Z_end`。每次覆盖前，前一个消费者必须结束。
5. `score_shared` 只保存已经完成 causal mask、scale 和 gate ratio 的 BF16 score。
6. output 的旧 state contribution 与 in-chunk contribution 在同一个 FP32 `out` fragment 中累加，
   只在最终 global store 时转成 BF16。
7. 完整 chunk 的 `gamma_last=gamma[63]`；tail chunk 必须使用最后一个有效 token。
8. invalid tail rows 的 residual 为零，invalid output 不写回，因此固定 64-row Tensor Core tile 不改变
   有效 token 的数学结果。
9. GVA 中 `bhg=floor(bh/(Hv/Hq))`，所以多个 value heads 共享 global Q/K head，但它们仍有独立的
   V/g/beta/A/state/output。
10. 当前没有 local/shared spilling。FP32 state 的精度跨 chunks 保留；BF16 误差只在 GEMM operand
    snapshot 和最终 output 上引入。

## 16. 当前已经做了什么、还没有做什么

| 项目 | 当前状态 |
| --- | --- |
| residual-first 数学变换 | 已实现 |
| W/U global intermediate | 已消除 |
| `gamma`、`1/gamma`、`gamma_last` 缓存 | 已实现 |
| Q/K/V/A/g/beta 两阶段预取 | 已实现 |
| ping-pong physical shared storage | 由 `T.Pipelined` lowering 生成 |
| low-parallel / normal specialization | 只有入口和 kernel 名，body 尚未分化 |
| MHA / GVA specialization | 只有 kernel 名，尚未复用 GVA 的 Q/K work |
| `dv` 分片增加 grid | 尚未实现 |
| triangular GEMM 跳过上三角 tile | 尚未实现 |
| warp specialization / TMA multicast | 尚未实现 |
| 跨 chunk affine scan / WY form | 尚未实现 |

因此，当前算法可以概括为：一个 block 持久拥有一个完整 FP32 state，使用六次 Tensor Core GEMM 完成
一个 chunk 的 residual correction、causal output 和 state transition；gate 的指数只算一次，下一
chunk 的只读输入通过 shared-memory ping-pong 预取，但 state chain 本身仍然严格串行。


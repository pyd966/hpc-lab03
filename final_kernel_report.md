# GDN Prefill 最终版 Kernel 完整报告

本文只描述最终提交 `dcb410d` 对应的默认执行路径。它是一份独立报告，不替代
`walkthrough.md`、`walkthrough_v_shared_swizzle.md` 或
`walkthrough_tail_specialization.md`。

本文中的资源数字有三种口径，阅读时必须区分：

1. **logical fragment size**：按 TileLang 中 fragment 的 shape 和 dtype 计算的逻辑容量；
2. **source-level shared size**：按源码中的 shared buffer 及 pipeline stage 计算的容量；
3. **NCU launch resource**：lowering、storage planning、alignment 完成后，硬件实际看到的
   registers/thread 与 dynamic shared/block。

fragment 的逻辑容量不能直接等同于寄存器文件占用。最终是否 spill、每个 CTA 能否驻留，
应以生成代码和 NCU launch statistics 为准。

## 1. 最终版本概览

最终 kernel 的核心结构可以概括为：

```text
host/JIT specialization
  -> 按 B*Hv 选择 D=64x2 或 D=128x1
  -> 按 T%64 静态选择 full-only 或 full-prefix+tail
  -> 一次 kernel launch

每个 CTA
  -> 读取一次 initial state（或清零）
  -> state 始终以 FP32 transposed fragment 驻留
  -> 对 floor(T/64) 个完整 chunk：
       Q/K/V/A/g/beta 双缓冲 cp.async prefetch
       六组异步 WGMMA + 穿插 gamma/elementwise
       full output 经 shared + TMA 写回
  -> 若 T%64 != 0：
       单级 predicated load + zero padding
       复用同一组 fragments 执行六组 WGMMA
       只写 valid output rows
  -> 等最后一次 state update 完成
  -> 写一次 final state
```

最终版包含以下关键优化：

- 将 `exp(g)`、`1/gamma`、`gamma_last` 提前到最早可执行位置，与前两组 WGMMA 重叠；
- 调整六组矩阵乘和 elementwise 的顺序，只在结果第一次真正被消费前等待；
- 使用 RS WGMMA 方向，让 recurrent state 长期留在寄存器，不再每个 chunk 经 shared
  往返；
- 对 Q/K/V/A/g/beta 做完整块双缓冲 prefetch；
- 对 Q/K/V/A/score 使用 full-bank-swizzled shared layout；
- 根据 `B*Hv` 自动沿 value dimension 分成一个或两个 CTA；
- 在适用的 D=64 路径中允许 score/output shared storage 复用；
- 对非 64 整除序列，在同一 launch 内运行优化完整前缀和一个独立 predicated tail；
- output 每 chunk 写回，但 FP32 recurrent state 只在 kernel 开始和结束接触 global memory。

## 2. 输入、输出与数学对象

输入/输出 tensor 为：

| tensor | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `q` | `[B,T,Hq,128]` | BF16 | query |
| `k` | `[B,T,Hq,128]` | BF16 | key |
| `v` | `[B,T,Hv,128]` | BF16 | value |
| `g` | `[B,T,Hv]` | FP32 | 已预处理的 cumulative gate |
| `beta` | `[B,T,Hv]` | FP32 | residual gate |
| `A` | `[B,T,Hv,64]` | BF16 | chunk 内变换矩阵 |
| `initial_state` | `[B,Hv,128,128]` | FP32 | 可选初始 recurrent state |
| `output` | `[B,T,Hv,128]` | BF16 | 输出 |
| `final_state` | `[B,Hv,128,128]` | FP32 | 最终 recurrent state |

固定常量：

```text
C = chunk size = 64
K = key/query dimension = 128
V = value dimension = 128
scale = 1/sqrt(128)
```

对一个 chunk，令 `S` 是进入 chunk 时的 `[128,128]` state，
`gamma_i=exp(g_i)`，`gamma_last` 是最后一个有效 token 的 gamma。方便描述时，最终
计算等价于：

```text
Z0       = K @ S
O0       = Q @ S
R        = beta * (V - diag(gamma) @ Z0)
Z1       = A @ R
O        = scale * diag(gamma) @ O0
Score    = causal(scale * diag(gamma) @ (Q @ K^T)
                  @ diag(1/gamma))
O        = O + Score @ Z1
S_next   = gamma_last * S
           + K^T @ (diag(gamma_last/gamma) @ Z1)
```

源码使用转置的 value-major fragment，因此实际执行的是这些式子的转置形式。例如
`S^T @ K^T=(K@S)^T`，不改变数学结果。

实现还移动了部分 gate factor：

```text
score_tmp[row,col]
    = causal(scale * gamma[row] / gamma_last * QK[row,col])

zhat[col]
    = gamma_last / gamma[col] * Z1[col]

score_tmp @ zhat
    = causal(scale * gamma[row] / gamma[col] * QK[row,col]) @ Z1
```

这样 `gamma_last/gamma[col]` 只需乘到 Z 上一次，随后同一份 `zhat` 同时供 output
correction 和 state update 使用。

## 3. Host 侧 dispatch 与 launch

### 3.1 CPU memory 到 HBM 在什么时候发生

进入 `gdn_prefill_forward` 时，输入已经是 CUDA tensor。CPU memory 到 GPU HBM 的数据
创建或拷贝发生在调用者创建 CUDA tensor、预处理和 `.to(device)` 的阶段，不发生在本
kernel 内。

wrapper 在 launch 前：

1. 从 shape 读取 `B,T,Hq,Hv`；
2. 计算 `chunks_per_batch=ceil(T/64)`；
3. 在 GPU 上分配 BF16 `output` 和 FP32 `final_state`；
4. 若没有 initial state，创建一个不会被 kernel 读取的 dummy CUDA tensor；
5. 选择静态专门化的 TileLang kernel，并从 JIT cache 获取编译结果；
6. 将 device pointers 和 `chunks_per_batch` 作为参数发起一次 launch。

JIT 首次编译耗时不属于 CUDA-event benchmark 的 kernel median。

### 3.2 value dimension 自动拆分

令：

```text
owners = B * Hv
```

目标 GPU slice 有 14 个 SM。默认 `GDN_DV_SPLIT=auto` 使用：

```text
2 * owners <= 14  -> dv_tile=64,  dv_parts=2
2 * owners >  14  -> dv_tile=128, dv_parts=1
```

其依据不是针对某个 public case 写死，而是 CTA wave 数：

- owner 很少时，D=128 每个 owner 只有一个 CTA，无法铺满 14 SM；拆成两个完全独立的
  value-column CTA 可以增加并行度；
- owner 已经足够多时，继续拆分只会复制 Q/K/A/g/beta load 和 `Q@K^T` 计算，收益通常
  不能抵消额外工作；
- chunk chain length 不改变同一 owner 内 state 的时间依赖，因此 sweep 中没有出现一个
  比 wave count 更稳定的 chain-length crossover。

D=64 的两个 CTA 分别拥有 value columns `[0,64)` 与 `[64,128)`。它们不共享 state，
不做 reduction，也不需要跨 CTA 同步。

### 3.3 tail 静态专门化

```text
full_chunks_only = (T % 64 == 0)
has_tail         = not full_chunks_only
```

`has_tail` 是 JIT specialization 参数：

- `has_tail=False`：只生成完整块 pipeline，不分配 tail shared；
- `has_tail=True`：完整块 pipeline 只迭代 `floor(T/64)` 次，然后执行一次 tail；
- 没有在每个完整块里增加动态边界分支，也没有让完整块 copy 携带 predicate；
- full-only 和 tail kernel 使用不同 symbol/cache entry。

这保证一个 `T=64F+r` 的任务执行 `F` 个 fast full chunks 加至多一个 tail，而不是因为
`r!=0` 就让全部 `F+1` 个 chunk 回退。

### 3.4 grid、threads 和 CTA 映射

```text
grid.x  = B * Hv * dv_parts
threads = 128, if dv_tile=64
          256, if dv_tile=128
```

每个 CTA 内：

```text
owner   = blockIdx.x // dv_parts
part    = blockIdx.x %  dv_parts
bb      = owner // Hv
bh      = owner %  Hv
bhg     = bh // (Hv/Hq)
dv_left = part * dv_tile
```

`bhg` 实现 GVA/MQA head mapping。多个 value heads 可能读取同一个 q/k head，但各 CTA
仍独立发出 load；这些重复 load 通常可由 L2 reuse 降低实际 HBM traffic。

## 4. CTA 内长期状态与 fragment 分配

### 4.1 为什么使用 transposed RS 数据流

传统 SS 数据流会为了让 WGMMA 读取 state，把大 state operand 写入 shared。最终 RS
数据流让 A operand 直接来自 registers：

```text
state_t[dv,k]       = S^T
state_operand[dv,k] = BF16(S^T)
z_t[dv,token]       = Z^T
out_t[dv,token]     = O^T
```

FP32 `state_t` 从 kernel prologue 到 epilogue 始终驻留 fragment。每个 chunk 只做一次
`state_t -> state_operand` 的 FP32 到 BF16 转换，为 RS WGMMA 准备 operand；没有把完整
state 放进 shared。

### 4.2 fragment 明细

| fragment | 作用 | D=64 shape/dtype | D=64 logical | D=128 shape/dtype | D=128 logical |
| --- | --- | --- | ---: | --- | ---: |
| `state_t` | FP32 recurrent state | `[64,128]` FP32 | 32 KiB | `[128,128]` FP32 | 64 KiB |
| `state_operand` | state 的 BF16 RS operand | `[64,128]` BF16 | 16 KiB | `[128,128]` BF16 | 32 KiB |
| `z_operand` | residual/Z 的 BF16 RS operand | `[64,64]` BF16 | 8 KiB | `[128,64]` BF16 | 16 KiB |
| `z_t` | FP32 Z/residual accumulator | `[64,64]` FP32 | 16 KiB | `[128,64]` FP32 | 32 KiB |
| `out_t` | FP32 output accumulator | `[64,64]` FP32 | 16 KiB | `[128,64]` FP32 | 32 KiB |
| `score` | FP32 64x64 score | `[64,64]` FP32 | 16 KiB | `[64,64]` FP32 | 16 KiB |
| **合计** | | | **104 KiB/CTA** | | **192 KiB/CTA** |

平均分摊到线程：

```text
D=64 : 104 KiB / 128 threads = 832 B/thread
       = 208 个 32-bit register-equivalent/thread

D=128: 192 KiB / 256 threads = 768 B/thread
       = 192 个 32-bit register-equivalent/thread
```

这只是 fragment 元素的平均逻辑分布。地址、loop index、predicate、descriptor 和临时值也
会占寄存器，编译器还会做 live-range reuse，因此最终 NCU 数字是：

| specialization | registers/thread | spill |
| --- | ---: | --- |
| D=64 full representative | 255 | 有 compiler spill（长链中累计） |
| D=64 tail synthetic | 255 | 704 B local + 128 B shared requests |
| D=128 full representative | 246 | 0 |
| D=128 public tail | 250 | 0 |

### 4.3 fragment 生命周期与复用

| fragment | 产生 | 首次/后续消费 | 生命周期结束或复用点 |
| --- | --- | --- | --- |
| `state_t` | initial-state load/zero | G0、G1、state decay、G5 | kernel 结束后写 final state |
| `state_operand` | 每 chunk `state_t` 转 BF16 | G0、G1 | G1 operand 已被 WGMMA 接收后可覆盖 |
| `z_t` | G0 | residual elementwise；随后被 G2 覆盖 | G2 后变为 Z1，再原地乘 decay |
| `z_operand` | residual 转 BF16 | G2；随后由 decayed Z1 重写 | G4、G5 都接收 operand 后可覆盖 |
| `out_t` | G1 | output scale；G4 累加 | 写 output staging 后可覆盖 |
| `score` | G3 | gate/mask；转 BF16 到 score shared | copy 完成后可覆盖 |

tail 不新增任何 fragment。完整前缀结束后，它复用同一组
`state_t/state_operand/z_t/z_operand/out_t/score`。

## 5. Shared memory 与 ping-pong buffer

### 5.1 完整块 producer 的物理双缓冲

源码只声明一份 `q_shared` 等对象，但 `T.Pipelined` 将 stage-0 producers lowering 成
两个物理 stage。完整块的 ping-pong 占用为：

| producer | 单 stage D=64 | 双 stage D=64 | 单 stage D=128 | 双 stage D=128 |
| --- | ---: | ---: | ---: | ---: |
| Q BF16 `[64,128]` | 16 KiB | 32 KiB | 16 KiB | 32 KiB |
| K BF16 `[64,128]` | 16 KiB | 32 KiB | 16 KiB | 32 KiB |
| V BF16 `[64,dv]` | 8 KiB | 16 KiB | 16 KiB | 32 KiB |
| A BF16 `[64,64]` | 8 KiB | 16 KiB | 8 KiB | 16 KiB |
| g FP32 `[64]` | 0.25 KiB | 0.5 KiB | 0.25 KiB | 0.5 KiB |
| beta FP32 `[64]` | 0.25 KiB | 0.5 KiB | 0.25 KiB | 0.5 KiB |
| **producer 合计** | **48.5 KiB** | **97 KiB** | **56.5 KiB** | **113 KiB** |

这里的双缓冲只服务 global-to-shared input pipeline。`gamma`、`1/gamma`、`gamma_last`、
score 和 output staging 生命周期短，不跨 chunk 预取，所以没有双缓冲。

### 5.2 非 ping-pong shared

| object | D=64 | D=128 | 用途 |
| --- | ---: | ---: | --- |
| `score_shared` BF16 `[64,64]` | 8 KiB | 8 KiB | G3 FP32 score 转 BF16，供 G4 |
| `output_shared` BF16 `[64,dv]` | 8 KiB | 16 KiB | full chunk 的 TMA output staging |
| `gamma_shared` FP32 `[64]` | 0.25 KiB | 0.25 KiB | `exp(g)` |
| `inv_gamma_shared` FP32 `[64]` | 0.25 KiB | 0.25 KiB | `1/gamma` |
| `gamma_last` FP32 `[1]` | 0.003906 KiB | 0.003906 KiB | 最后一个有效 gamma |

默认 fast shared 合计：

```text
D=64 : 97 + 8 + 8 + 0.25 + 0.25 + 0.003906
       = 113.503906 KiB source logical
       = 113.516 KiB NCU aligned dynamic shared

D=128: 113 + 8 + 16 + 0.25 + 0.25 + 0.003906
       = 137.503906 KiB source logical
       = 137.516 KiB NCU aligned dynamic shared
```

`g_shared` 和 `beta_shared` 已包含在 producer ping-pong 表中，不应再重复相加。

### 5.3 D=64 score/output storage 复用

当手工强制配置产生大量 D=64 CTAs，代码可令：

```text
output_shared = score_shared
```

这是合法的，因为 G4 是 score shared 的最后消费者。`warpgroup_wait(1)` 先退休较老的 G4，
确认 WGMMA 不再读取 score，然后才把 `out_t` 覆盖写入同一地址。compact D=64 的
source logical shared 因而为：

```text
97 + 8 + 0.25 + 0.25 + 0.003906 = 105.503906 KiB
```

默认 auto 策略只在 `2*owners<=14` 时选择 D=64，此时 CTA 本来不超过一个 14-SM wave，
所以 `reuse_output_shared` 默认不会为了无用的同-SM residency 增加别名约束；手工强制
D=64 且 CTA 数超过一 wave 时才启用 compact path 和 `min_blocks_per_sm(2)` annotation。

### 5.4 tail single-stage shared

有 tail 的 specialization 另外声明：

| tail object | D=64 | D=128 |
| --- | ---: | ---: |
| Q tail | 16 KiB | 16 KiB |
| K tail | 16 KiB | 16 KiB |
| V tail | 8 KiB | 16 KiB |
| A tail | 8 KiB | 8 KiB |
| g tail | 0.25 KiB | 0.25 KiB |
| beta tail | 0.25 KiB | 0.25 KiB |
| **nominal declarations** | **48.5 KiB** | **56.5 KiB** |

tail 发生在完整 pipeline 完全结束之后。这些 buffer 与部分已经死亡的 full-prefix storage
生命周期不重叠，TileLang storage planner 会合并物理地址。因此实际增加不是简单相加，
而是两条路径都只增加 40 KiB：

| path | full-only dynamic shared | tail dynamic shared | actual delta |
| --- | ---: | ---: | ---: |
| D=64 | 113.516 KiB | 153.516 KiB | +40 KiB |
| D=128 | 137.516 KiB | 177.516 KiB | +40 KiB |

NCU 的 UI 使用十进制 Kbyte，分别显示 157.200 Kbyte 和 181.776 Kbyte；换算成二进制
即上表数值。每个 launch 另有 1.024 decimal Kbyte driver shared bookkeeping。

### 5.5 shared layout 与 bank conflict

`make_full_bank_swizzled_layout` 应用于 Q、K、V、A 和 score shared。swizzle 改变 shared
地址到 bank 的映射，不改变逻辑矩阵：

- Q/K 同时服务 RS project 和 SS score WGMMA；
- V 被 residual elementwise 以 `[token,dv]` 读取；
- A 服务 `Z^T @ A^T`；
- score 从 FP32 fragment 转 BF16 后服务 `Zhat^T @ Score^T`；
- tail Q/K/V/A 使用与 full buffer 相同的逻辑 swizzle 规则。

swizzle 的目标是让一个 warp/warpgroup 的并行 lane 不集中访问同一 shared bank。最终
D=128 wide profile 中先前 V 路径的 bank conflicts 已消除，MIO throttle 降到 0.54 CPI；
这也是为什么最终版保留 full-bank layout，而不是使用普通 row-major shared。

## 6. 完整 chunk 的 input pipeline

### 6.1 每 chunk global traffic

每个 CTA、每个完整 chunk 的输入字节数：

| input | D=64 | D=128 | lowered copy |
| --- | ---: | ---: | --- |
| Q | 16 KiB | 16 KiB | 16-byte `cp.async` |
| K | 16 KiB | 16 KiB | 16-byte `cp.async` |
| V slice | 8 KiB | 16 KiB | 16-byte `cp.async` |
| A | 8 KiB | 8 KiB | 16-byte `cp.async` |
| g + beta | 0.5 KiB | 0.5 KiB | 4-byte `cp.async` |
| **input total** | **48.5 KiB** | **56.5 KiB** | |
| output | 8 KiB | 16 KiB | shared-to-global TMA store |

这些是指令请求字节，不等于 DRAM 实际字节：所有 global request 先查询 L2，L2 miss 才
访问 HBM。代码没有显式的“HBM 搬到 L2”指令。`cp.async` 的语义是 global 地址到
shared；L2 是硬件自动管理的中间层。

D=64 的两个 CTA 会复制 Q/K/A/g/beta 和 score GEMM，V/output/state 则各处理不同 value
slice。因此 D=64 每个 logical owner 每 chunk 的请求总量为 97 KiB input + 16 KiB
output；重复的 Q/K 等有机会从 L2 命中，但仍消耗 load instructions 和 L2/shared 带宽。

### 6.2 prologue、steady state 与 epilogue

令完整块数为 `F=floor(T/64)`：

```text
pipeline stage 0 = current chunk, parity i&1
pipeline stage 1 = next chunk,    parity (i+1)&1

prologue:
    cp.async full chunk 0 -> stage 0
    cp.async_commit

steady state for chunk i:
    cp.async full chunk i+1 -> other stage, if it exists
    cp.async_commit
    cp_async_wait<1>             # current stage ready; next may remain pending
    __syncthreads                # CTA-wide publication
    compute full chunk i

epilogue:
    不发起越界 prefetch
    排空最后一个已提交 copy group
    compute full chunk F-1
```

源文件中的 pipeline metadata 为：

```text
order = [5,4,0,1,2,3,6,7]
stage = [q_stage,k_stage,v_stage,a_stage,gate_stage,gate_stage,1,1]
```

默认 `qkva` profile 令 Q/K/V/A 和 gate producers 全部位于 prefetch stage。顺序由
TileLang PipelinePlanning 在 WGMMA macro 展开后调度；生成代码证明确实出现了双 stage
offset、`cp_async_commit`、`cp_async_wait<1>` 和 CTA barrier。

prefetch 能隐藏的是下一 chunk 的 global/L2 latency。它不能隐藏：

- 当前 chunk 尚未到达 shared 前的必要 wait；
- shared bank/MIO latency；
- WGMMA result dependency；
- recurrent `state_t(i+1)` 对 G5(i) 的真实依赖；
- CTA 数不足造成的 SM underfill。

这解释了为什么仅增加 ping-pong buffer 的收益有限：每个 chunk 已有很高计算密度，且
剩余 stall 中相当一部分来自 barrier、WGMMA dependency 和 state chain，而不是纯 HBM
带宽。

## 7. 一个完整 chunk 的逐步执行

下面以单个 CTA 的 value slice 为单位说明。`G0..G5` 表示提交顺序中的六组 WGMMA。

### 7.1 chunk 入口

1. pipeline 已把 Q/K/V/A/g/beta 的当前 parity stage 搬入 shared；
2. `cp_async_wait<1>` 确认当前 copy group 完成，同时允许至多一个更新的 group 留在途；
3. `__syncthreads` 确认所有线程都可读取当前 shared stage；
4. FP32 `state_t` 已包含上一 chunk 完成后的状态；
5. `T.copy(state_t,state_operand)` 将 state 转成 BF16 RS operand。

### 7.2 G0 与 G1：先发出两个 state projection

```text
G0: z_t   = state_operand @ K_shared^T
           = (K @ S)^T

G1: out_t = state_operand @ Q_shared^T
           = (Q @ S)^T
```

两组 WGMMA 都异步提交，提交后没有立刻 `wait_group 0`。这正是最终计算顺序优化的第一处：
在 Tensor Core 工作时，标量/FP32 pipeline 继续处理 gate。

### 7.3 提前计算 gamma，并衰减 state

```text
gamma[token]     = exp2(g[token] * log2(e))
inv_gamma[token] = 1 / gamma[token]
gamma_last       = gamma[63]
state_t         *= gamma_last
```

每个 gamma 只做一次 exp，每个 inverse 只做一次除法。它们被提前到 G0/G1 后，既不依赖
WGMMA result，又会被后续 residual、score 和 state update 多次使用。

`gamma_last` 是 shared 中的一个 FP32 标量。完整块固定由 token 63 写入；lowering 生成
必要的 CTA synchronization，保证其它线程在衰减 state 前看到该值。

### 7.4 第一个精确 wait：只退休 G0

```text
warpgroup_wait(1)
```

此时 G0 是较老 group，G1 仍可在途。马上要读取的是 G0 的 `z_t`，而不是 G1 的
`out_t`，所以只等到还剩一个 outstanding group：

```text
z_t = beta * (V^T - gamma * z_t)
```

然后把 FP32 residual 转成 BF16 `z_operand`。这段 elementwise 与仍在执行的 G1 重叠。

### 7.5 G2 与 G3：correction 和 score 并行发出

```text
G2: z_t   = z_operand @ A_shared^T
           = (A @ residual)^T

G3: score = Q_shared @ K_shared^T
```

G2 使用 RS operand；G3 的两个 operand 都来自 shared，是 SS score WGMMA。两者互不依赖，
因此连续提交。

### 7.6 第二个精确 wait：留下 G3

```text
warpgroup_wait(1)
```

在当前提交序列中，这个 wait 退休 G1 和 G2，只保留最新的 G3：

- G1 的 `out_t` 暂时还不读取，但已自然完成；
- G2 的 `z_t` 现在必须用于 `zhat`；
- G3 的 score 尚未使用，可继续执行。

随后：

```text
z_t[dv,token] *= gamma_last * inv_gamma[token]
T.copy(z_t,z_operand)  # FP32 -> BF16
```

此时 `z_operand` 已成为同时供 output 和 state 使用的 `zhat^T`。

### 7.7 第三个 wait：在 score 第一次使用前排空

```text
warpgroup_wait(0)
```

现在才第一次读取 G3 的 score，同时也开始读取 G1 的 out：

```text
out_t[dv,row] *= scale * gamma[row]

if row >= col:
    score[row,col] *= scale * gamma[row] / gamma_last
else:
    score[row,col]  = 0
```

causal mask、scale 和 gate 在 FP32 accumulator 上完成。之后 score 转成 BF16 写入
`score_shared`。fragment-to-shared copy 后必须有 CTA publication barrier，G4 的
warpgroup 才能安全把它作为 shared operand 读取。

### 7.8 G4 与 G5：output correction 和 state update

```text
G4: out_t   += z_operand @ score_shared^T
              = (Score_tmp @ Zhat)^T

G5: state_t += z_operand @ K_shared
              = S_decay^T + Zhat^T @ K
```

二者读取同一份 `z_operand`，但写不同 FP32 accumulators，因此连续异步提交。G5 更新后的
`state_t` 是下一 chunk 的唯一 recurrent input。

### 7.9 第四个 wait：先退休 G4，G5 与 output store 重叠

```text
warpgroup_wait(1)
```

G4 较老，wait 后 output 已就绪，但 G5 可继续在途：

1. 将 `out_t[dv,token]` 转置/转 BF16 写 `output_shared[token,dv]`；
2. 若 score/output alias，G4 此时已不再读取 score shared，所以覆盖合法；
3. 对 full chunk 发起 shared-to-global TMA store；
4. generated code 使用 `tma_store_arrive()` 和 `tma_store_wait<0>()`，保证 staging storage
   在下一次复用前写出完成。

### 7.10 第五个 wait：下一 chunk 前退休 G5

```text
warpgroup_wait(0)
```

下一 chunk 一开始就会把新的 `state_t` 转成 `state_operand` 并提交 G0/G1，因此 G5 必须
在这里完成。这是不可消除的 recurrent dependency；把 wait 再后移会读到未完成的 state。

## 8. 六组 WGMMA 汇总与计算量

| group | 数学形式（转置实现） | M/N/K | FLOPs per CTA |
| --- | --- | --- | ---: |
| G0 | `S^T @ K^T` | `dv x 64 x 128` | `16384*dv` |
| G1 | `S^T @ Q^T` | `dv x 64 x 128` | `16384*dv` |
| G2 | `R^T @ A^T` | `dv x 64 x 64` | `8192*dv` |
| G3 | `Q @ K^T` | `64 x 64 x 128` | `1,048,576` |
| G4 | `Zhat^T @ Score^T` | `dv x 64 x 64` | `8192*dv` |
| G5 | `Zhat^T @ K` | `dv x 128 x 64` | `16384*dv` |

按一次 fused multiply-add 为 2 FLOPs：

```text
FLOPs/CTA/chunk = 65536*dv + 1,048,576

D=64 : 5,242,880 FLOPs/CTA/chunk
D=128: 9,437,184 FLOPs/CTA/chunk
```

D=64 一个 logical owner 有两个 CTA，因此总计 10,485,760 FLOPs/chunk，比 D=128 的
9,437,184 多 1,048,576，恰好是被两个 value parts 重复计算的一次 G3 score GEMM。

只用 input+output global request 估算、不计 initial/final state 和 elementwise：

```text
D=64 : 5,242,880 / ((48.5+8)*1024) = 90.62 FLOP/B
D=128: 9,437,184 / ((56.5+16)*1024) = 127.12 FLOP/B
```

这是算法/请求层面的近似 arithmetic intensity，不是 Nsight roofline 的实际 DRAM
intensity。L2 hit、D=64 duplicated load、shared traffic、spill 和首尾 state traffic 都会让
硬件计数不同。

initial state 和 final state 每 owner 各为 64 KiB。D=128 由一个 CTA 各读写 64 KiB；
D=64 两个 CTA 各读写自己的 32 KiB slice，总字节仍为 64 KiB。它们只在 kernel 边界
发生一次，链越长，摊到每个 chunk 的成本越低。

## 9. tail chunk 的完整执行

令：

```text
left         = floor(T/64)*64
valid_tokens = T-left, 取值 1..63
```

### 9.1 为什么 tail 不进入 ping-pong loop

完整块 pipeline 要求固定 shape、无 predicate 的 producer，才能保持原有 prologue、steady
state 和 epilogue。tail 只出现一次，没有“下一 tail chunk”可预取，单独做双缓冲没有
意义。把动态 predicate 放进完整 pipeline 还会改变每个完整块的生成代码。

最终方案先排空 full pipeline，再用独立 single-stage tail buffer。这样：

- full prefix 没有边界判断；
- tail 只付一次 predicate/zero-fill 成本；
- state fragment 不落 HBM，也不需要第二次 launch；
- full 与 tail 之间只有同一个 CTA 内的 state dependency。

### 9.2 predicated load 与 zero padding

tail cooperative load 执行：

```text
if token < valid_tokens:
    load Q, K, V, g, beta
else:
    write zero to tail shared

if A row < valid_tokens:
    load complete A row
else:
    write zero row
```

这些 load 不是下一迭代 prefetch，也没有 ping-pong stage。generated source 在进入 tail 前
先执行 `cp_async_wait<0>` 和 CTA barrier，保证 full pipeline 的所有 async copies 已排空；
tail 数据写完 shared 后再通过 CTA barrier 发布给 WGMMA。

### 9.3 tail gamma、mask 与最后一行

```text
gamma_last = gamma[valid_tokens-1]
```

而不是固定取 row 63。无效 token 的 g/beta 被写为 0，gamma 因而为 1，但这些行不会产生
可见输出。score predicate 为：

```text
row >= col and row < valid_tokens
```

只要 row 有效且满足 causal condition，就必然有 `col<=row<valid_tokens`，所以 score 不会
读取无效 Z column。state update 中无效 K rows 为 0，也不会污染 state。

### 9.4 tail 的六组 WGMMA 和 output

tail 复用与 full chunk 相同的 G0..G5 顺序及 `wait(1)/wait(1)/wait(0)/wait(1)/wait(0)`
依赖边界。区别只有：

- WGMMA 仍按硬件固定 64 rows 计算，zero padding 保证无效行不贡献；
- output 不使用整块 TMA store，而是每个 `(dv,token)` 做 `token<valid_tokens` 的
  predicated global store；
- G5 完成后 state 留在同一个 `state_t`，随后只写一次 final state。

如果 `T<64`，完整块循环执行 0 次，CTA 直接从 initial/zero state 进入上述 tail，仍使用
同一个 RS kernel。

## 10. 同步机制：每一种 wait 在保护什么

最终版同时存在四类不同同步。它们不能互相替代。

### 10.1 `cp_async_commit/wait`

对象：global-to-shared async copy group。

- `cp_async_commit()` 提交一批 Q/K/V/A/g/beta copy；
- `cp_async_wait<1>()` 保证当前 stage 已完成，同时允许一个 future group 仍 pending；
- pipeline 结束/进入 tail 前使用 `cp_async_wait<0>()` 全部排空。

它只保证 copy engine 的完成度，不自动让 CTA 所有线程在控制流上会合。

### 10.2 `__syncthreads`

对象：CTA 内线程与 shared-memory publication。

主要出现在：

- async copy wait 后，所有线程开始消费当前 stage 前；
- gamma/gamma_last 写入后，其他线程读取前；
- FP32 fragment 转 BF16 score shared 后，G4 读取前；
- output staging/TMA 使用和 stage reuse 的边界；
- tail zero-fill 完成后，WGMMA 读取 tail shared 前。

barrier stall 并不自动意味着 barrier 多余。这里很多 barrier 对 shared producer/consumer 的
正确性是必要的；优化方向只能是减少不必要的 shared round-trip 或改善各 warp 到达时间，
不能直接删除同步。

### 10.3 `warpgroup_arrive/warpgroup_wait<N>`

对象：异步 WGMMA committed groups 和 accumulator dependency。

语义可以理解为“等待直到 outstanding WGMMA groups 不超过 N 个”。最终顺序：

| point | outstanding before wait | wait | wait 后可安全读取 | 仍尝试 overlap |
| --- | --- | --- | --- | --- |
| residual 前 | G0,G1 | `wait(1)` | G0 `z_t` | G1 |
| Z decay 前 | G1,G2,G3 | `wait(1)` | G1 out、G2 z | G3 |
| score gate 前 | G3 | `wait(0)` | G3 score | 无 |
| output staging 前 | G4,G5 | `wait(1)` | G4 out | G5 state update |
| 下一 chunk 前 | G5 | `wait(0)` | 新 state | 无 |

这实现了用户期望的原则：仅在某个结果第一次要被使用之前等待，而不是每个 GEMM 后立刻
全排空。

### 10.4 TMA store arrive/wait

对象：full output 的 shared-to-global store。

`T.copy(output_shared,output_slice)` lowering 为 TMA store。generated code可见：

```text
tma_store(...)
tma_store_arrive()
tma_store_wait<0>()
```

它保证 TMA 已经完成读取 output shared，之后 shared storage才可被下一个 chunk 安全复用。
tail 因为只有部分有效 rows，使用 predicated direct global store，不走整 tile TMA。

### 10.5 为什么没有跨 CTA 同步

CTA 的 owner/value slice 互不重叠：

- 不同 `(B,Hv)` 拥有不同 state/output；
- 同一 owner 的两个 D=64 parts 拥有不同 value columns；
- q/k 即使共享只被读取；
- 没有跨 part reduction。

因此 kernel 不需要 cooperative-groups grid barrier，也不需要拆成多个 launch。kernel 返回
本身就是 host 观察 output/final-state 的全局完成边界。

## 11. 从内存层次看完整生命周期

| 时刻 | 数据移动 | 机制 | 是否与其它工作重叠 |
| --- | --- | --- | --- |
| 调用 kernel 前 | CPU/预处理 -> GPU HBM | PyTorch/CUDA tensor creation/copy | 不属于 kernel |
| wrapper | 分配 output/final-state HBM | CUDA allocator | launch 前 |
| CTA prologue | initial state global -> FP32 fragment | cooperative global load | 只发生一次 |
| full pipeline prologue | chunk 0 global(L2/HBM) -> shared stage 0 | `cp.async` | 尚无前一 chunk 可重叠 |
| full steady state | chunk i+1 global -> other shared stage | `cp.async` | 与 chunk i 计算重叠 |
| chunk 内 | state FP32 fragment -> BF16 fragment | register conversion | 不访问 HBM/shared state |
| chunk 内 | FP32 score -> BF16 score shared | shared store | G3/G4 间同步边界 |
| full output | FP32 out -> BF16 output shared -> global | shared store + TMA | 与 G5 部分重叠 |
| tail load | valid global -> tail shared，invalid zero | predicated load/store | 单级，不 prefetch 下一块 |
| tail output | FP32 out -> BF16 global | predicated store | 只写有效 rows |
| CTA epilogue | FP32 state fragment -> final-state global | cooperative store | 只发生一次 |

long scoreboard 表示 warp 在等待 L1TEX 管理的 global/local/shared dependency，常见来源包括
当前 stage 的数据到达、global output/state store dependency，以及 spill；它不等价于“每个
chunk 正在手工从 HBM 搬到 L2”。L2/HBM 选择由 cache hit/miss 自动决定。

## 12. Occupancy 与资源瓶颈

代表性 NCU launch resources：

| path | threads | regs/thread | dynamic shared | register block limit | shared block limit | theoretical occ. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D=64 full fast | 128 | 255 | 113.516 KiB | 2 | 1 | 6.25% |
| D=64 tail | 128 | 255 | 153.516 KiB | 2 | 1 | 6.25% |
| D=128 full | 256 | 246 | 137.516 KiB | 1 | 1 | 12.50% |
| D=128 tail | 256 | 250 | 177.516 KiB | 1 | 1 | 12.50% |

解释：

- D=128 同时被 registers 和 shared 限制为 1 CTA/SM；
- D=64 fast 的 register limit 理论上允许 2 CTA，但默认 shared allocation 使 NCU 报告
  shared block limit=1；
- D=64 compact score/output alias 可把 source shared 降到约 105.516 KiB，并专门用于 CTA
  多于一 wave、提高同-SM residency 有意义的强制 D=64 场景；
- auto D=64 只在 CTA 不超过 14 时启用，主要收益是 grid 从 owners 扩到 2*owners，而不是
  同一个 SM 同驻两个 block；
- tail 多出的 40 KiB 没有让已测路径的 resident blocks/SM 再下降，因此换取 tail fast
  prefix 是合理的。

低 occupancy 并不意味着 kernel 必然慢。一个 D=128 CTA 有 8 warps，WGMMA 本身提供大量
Tensor Core work；真正问题是当 grid 小于 14 时，部分 SM 完全没有 CTA，或者 active warps
又同时停在同一个 recurrent/barrier dependency 上。

## 13. 最终 NCU profile

所有最终 profile 使用 full metric set，并额外采集：

```text
--set full
--section PmSampling_WarpStates
--import-source yes
--clock-control none
--replay-mode kernel
```

报告包含 `Warp State Statistics`、raw metrics、source/SASS correlation；不是只采一个精简
stall counter set。

### 13.1 D=128 full：wide_gva_state

Artifact：

- `output/ncu_d128_vswizzle_wide_full.ncu-rep`
- `output/ncu_d128_vswizzle_wide_full_details.txt`
- `output/ncu_d128_vswizzle_wide_full_raw.csv`
- `output/ncu_d128_vswizzle_wide_full_source.csv`

| metric | value |
| --- | ---: |
| NCU duration | 2.24 ms |
| registers/thread | 246 |
| dynamic shared/block | 137.516 KiB |
| theoretical/achieved occupancy | 12.50% / 12.50% |
| spill requests | 0 |
| warp cycles per issued instruction | 5.16 |
| barrier | 1.06 CPI |
| wait | 0.70 CPI |
| MIO throttle | 0.54 CPI |
| long scoreboard | 0.45 CPI |
| short scoreboard | 0.38 CPI |
| GMMA | 0.20 CPI |

这里 barrier 是最大单项，但它保护 pipeline stage、shared operand 和 TMA staging；不能仅凭
1.06 CPI 判定为冗余。MIO 已因 V/shared swizzle 明显下降，long scoreboard 也不再是主导。

### 13.2 D=64 full：chain_equal

Artifact：

- `output/ncu_vswizzle_chain_full.ncu-rep`
- `output/ncu_vswizzle_chain_full_details.txt`
- `output/ncu_vswizzle_chain_full_raw.csv`
- `output/ncu_vswizzle_chain_full_source.csv`

| metric | value |
| --- | ---: |
| NCU duration | 322.40 us |
| registers/thread | 255 |
| dynamic shared/block | 113.516 KiB |
| theoretical/achieved occupancy | 6.25% / 6.25% |
| warp cycles per issued instruction | 3.17 |
| wait | 0.99 CPI |
| barrier | 0.56 CPI |
| long scoreboard | 0.24 CPI |
| short scoreboard | 0.16 CPI |
| MIO throttle | 0.04 CPI |
| GMMA | 0.03 CPI |

这个长链 D=64 path 的 global latency 已被 pipeline 较好摊薄，主要剩余项是显式 dependency
wait 和 CTA barrier。profile 中存在少量 compiler spilling request，但没有成为主要 CPI。

### 13.3 D=128 public tail：short_tail_state

Artifact：

- `output/ncu_tail_rs_short_full.ncu-rep`
- `output/ncu_tail_rs_short_full_details.txt`
- `output/ncu_tail_rs_short_full_raw.csv`
- `output/ncu_tail_rs_short_full_source.csv`
- `output/ncu_tail_rs_short_full.log`

| metric | value |
| --- | ---: |
| NCU duration | 66.62 us |
| grid/block | 8 CTAs / 256 threads |
| registers/thread | 250 |
| dynamic shared/block | 177.516 KiB |
| theoretical/achieved occupancy | 12.50% / 12.49% |
| local/shared spill | 0 / 0 |
| warp cycles per issued instruction | 5.44 |
| barrier | 1.121 CPI |
| wait | 0.694 CPI |
| long scoreboard | 0.594 CPI |
| MIO throttle | 0.496 CPI |
| not selected | 0.400 CPI |
| short scoreboard | 0.304 CPI |
| GMMA | 0.198 CPI |

grid 只有 8 CTAs，小于 14 SM，首先受到 launch underfill 影响。最终 1.63x 收益来自 16 个
完整块改走 RS/ping-pong，而不是让这个 shape 的 grid 本身铺满 GPU。

### 13.4 D=64 synthetic tail

Artifact：

- `output/ncu_tail_rs_dv64_full.ncu-rep`
- `output/ncu_tail_rs_dv64_full_details.txt`
- `output/ncu_tail_rs_dv64_full_raw.csv`
- `output/ncu_tail_rs_dv64_full_source.csv`
- `output/ncu_tail_rs_dv64_full.log`

| metric | value |
| --- | ---: |
| NCU duration | 13.41 us |
| grid/block | 8 CTAs / 128 threads |
| registers/thread | 255 |
| dynamic shared/block | 153.516 KiB |
| theoretical/achieved occupancy | 6.25% / 6.25% |
| warp cycles per issued instruction | 6.51 |
| local/shared spill requests | 704 B / 128 B |
| long scoreboard | 2.391 CPI |
| wait | 0.868 CPI |
| no instruction | 0.735 CPI |
| barrier | 0.706 CPI |
| short scoreboard | 0.369 CPI |
| LG throttle | 0.193 CPI |
| GMMA | 0.090 CPI |

这是只有一个 full chunk 加短 tail、grid 也只有 8 CTA 的 synthetic workload。下一 chunk
prefetch 几乎没有足够 steady-state 距离来摊薄，且少量 spill/global dependency 对总指令的
比例较大，所以 long scoreboard 很高。它不能代表长链 full path 的 stall 构成。

## 14. 最终 8-case correctness、时间与预计分数

测量为 10 warmups + 100 repetitions 的 CUDA-event median。`before` 是上一版 kernel
提交 `268342d` 的正式结果；speedup 是 `before/after`。公开评分使用
`p=t100/after`，当 `p>1` 时 `score=min(120,100+20*(p-1))`。

| case | `(B,T,Hq,Hv)` | path | before ms | final ms | speedup | p | estimated score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| short_tail_state | `(1,1025,2,8)` | D128 full-prefix+tail | 0.142096 | **0.087232** | **1.6289x** | 3.9670 | **120.00** |
| chain_equal | `(1,8192,4,4)` | D64x2 full | 0.345648 | 0.345120 | 1.0015x | 1.4424 | 108.85 |
| parallel_equal | `(1,2048,16,16)` | D128 full | 0.278224 | 0.278832 | 0.9978x | 1.8336 | 116.67 |
| parallel_gva | `(1,2048,4,16)` | D128 full | 0.255040 | 0.253744 | 1.0051x | 1.9372 | 118.74 |
| long_low_gva | `(1,32768,2,8)` | D128 full | 1.760048 | 1.747600 | 1.0071x | 1.0636 | 101.27 |
| batch_split_gva | `(4,8192,2,8)` | D128 full | 1.328464 | 1.338800 | 0.9923x | 1.1444 | 102.89 |
| wide_gva_state | `(1,8192,16,64)` | D128 full | 2.220288 | 2.260384 | 0.9823x | 1.0736 | 101.47 |
| deep_gva_state | `(1,16384,8,32)` | D128 full | 2.595504 | 2.632928 | 0.9858x | 1.0753 | 101.51 |
| **mean** | | | | | | | **108.93** |

8 个 case 的 output 和 final state 全部 PASS，预计分数全部超过 100。除
`short_tail_state` 外，其余 7 个都生成 `has_tail=False` 的原 full-only body；这些 case
的约 -1.8% 到 +0.7% 差异属于跨集群 job 的测量/频率波动。

最终 benchmark 原始日志：`output/lab3_66288.log`。

## 15. tail correctness 边界覆盖

额外验证不只覆盖 public 的 tail=1，还覆盖：

| case property | selected path | result |
| --- | --- | --- |
| T=1，无完整块 | D64x2 RS tail | PASS |
| T=31，无完整块，带 initial state | D64x2 RS tail | PASS |
| T=63，无完整块，带 initial state | D128 RS tail | PASS |
| T=65，tail=1 | D64x2 RS full+tail | PASS |
| T=95，tail=31，mixed gate | D64x2 RS full+tail | PASS |
| T=127，tail=63，带 initial state | D64x2 RS full+tail | PASS |
| T=8191，127 full + tail=63 | D64x2 RS full+tail | PASS |
| T=95，tail=31，带 initial state | D128 RS full+tail | PASS |

原始日志：

- `output/lab3_66150.log`：initial D128 tail；
- `output/lab3_66240.log`：D64/D128 tail=1/31/63 和长链；
- `output/lab3_66892.log`：零完整块 T=1/31/63。

## 16. 生成代码验证

源码层面的优化只有在 lowering 后仍保持预期顺序才成立。最终生成代码检查确认：

1. full loop 有 parity-based shared offset，确实是两级 ping-pong；
2. Q/K/V/A 使用 16-byte `cp_async`，g/beta 使用 4-byte `cp_async`；
3. 每轮出现 `cp_async_commit` 和 `cp_async_wait<1>`；
4. G0/G1 后先执行 gate math，再在 residual 首次使用 Z 前 `warpgroup_wait<1>`；
5. G2/G3 后在 Z decay 前 `wait<1>`，score 首次读取前才 `wait<0>`；
6. G4/G5 后只先退休 G4，G5 与 output staging/TMA overlap；
7. 下一 chunk 读取 state 前出现最终 `wait<0>`；
8. full output 使用 `tma_store_arrive/wait<0>`；
9. tail 前有 `cp_async_wait<0>`，tail 使用独立 storage 和 predicates；
10. tail output 只对有效 token 做 global store；
11. final state 在 full+tail 全部结束后只写一次。

相关 generated source：

- `output/tail_split_full_dv128_generated.log`：最终 full-only D128；
- `output/tail_rs_dv128_generated.log`：最终 D128 full-prefix+tail；
- `output/vswizzle_dv64_generated.log`：D64 full path。

将 full-only source 与 tail source 的 full-prefix body 统一参数命名后，计算 body 保持一致；
tail specialization 没有把完整块变成 predicated fallback。

## 17. 如何理解最终瓶颈

最终 kernel 不是单一的 HBM bandwidth-bound：

- 每 chunk 有约 5.24M/9.44M WGMMA FLOPs，输入请求只有 48.5/56.5 KiB；
- full pipeline 已把下一 chunk global load 与当前计算重叠；
- state 的时间递归使 G5 到下一 chunk G0 存在不可并行的真实依赖；
- shared operand publication 和 ping-pong stage reuse 需要 CTA barrier；
- WGMMA accumulator 第一次使用前必须 wait；
- D=128 受 registers+shared 限制为 1 CTA/SM；
- owner 少时即使单 CTA 很快，也可能因为 grid 小于 14 而 underfill；
- 极短 tail workload 没有足够 iterations 建立 steady-state pipeline。

因此 profile 中 `stall_barrier`、`stall_wait`、`stall_long_scoreboard` 会同时存在：

- barrier：warp 到达 CTA shared-memory 边界的时间不一致；
- wait：异步 WGMMA 结果的真实消费者已到达；
- long scoreboard：L1TEX/global/local/shared dependency，短 case 中还会放大少量 spill；
- MIO throttle：shared/load-store issue 压力，已由 swizzle 显著缓解；
- no instruction/not selected：低 active/eligible warp 和调度 underfill。

这也说明为什么继续无条件增加第三个 buffer 不一定有收益：它会增加 shared，占用不会改善
recurrent dependency，还可能降低 residency。最终选择的是“完整块双缓冲，tail 单级”，
正好对应两种路径可利用的并行距离。

## 18. 文件与复现实验

核心代码：

- `student/tilelang_fwd.py`：host dispatch、dv split、full/tail specialization；
- `student/tilelang_rs.py`：最终 RS full-prefix + predicated-tail kernel；
- `dump_kernel_source.py`：生成 full/tail lowered source；
- `profile_full.sh`：完整 NCU profile；
- `job.sh`：集群提交 wrapper。

公开 8-case benchmark：

```bash
./job.sh --warmup 10 --repetitions 100 --output-format csv
```

profile 可通过 `profile_full.sh` 指定 case CSV 运行。检查 profile 时至少同时查看：

```text
*.ncu-rep          可交互 NCU 报告
*_details.txt      full textual sections，含 Warp State Statistics
*_raw.csv          原始 metrics
*_source.csv       source/SASS correlation
*.log              job 和采集日志
```

## 19. 最终结论

最终版把一个 owner 的完整生命周期压缩在单 CTA state fragment 中：input 以双缓冲从
global/L2 进入 shared，六组 WGMMA 以精确 wait 边界执行，gamma 和 elementwise 穿插在
Tensor Core latency 中，output 每 chunk 写回，而 state 只在 kernel 首尾访问 global。

对于非整除序列，同一次 launch 先运行完全相同的优化 full pipeline，再执行一次
single-stage predicated tail。tail 增加 40 KiB 实际 shared、零额外 fragment，且没有降低
已测 resident-block 数。这个设计既保住完整块性能，也覆盖 tail=1..63、零完整块、长链、
initial state、D64/D128 和 GVA head mapping。

最终正式结果为 8/8 correctness PASS、8/8 estimated score >100、mean estimated score
108.93。资源上，D64/D128 的 logical fragments 分别为 104/192 KiB per CTA；full shared
为 113.516/137.516 KiB，tail shared 为 153.516/177.516 KiB；代表性实际寄存器为
255/246 至 250 registers per thread。

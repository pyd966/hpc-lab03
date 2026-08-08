# 优化 4：全输入流水与合并 output 写回

本文只描述提交 `a6c155a` 中的完整实现状态，是一份独立、可从头阅读的执行流程记录。
它不依赖 `report.md`，也不要求读者先阅读其他优化轮次。

## 1. 本轮改动与适用路径

本轮在完整 chunk 路径中完成两项修改：

1. 将 Q、K、V、A、g 和 beta 的 global-to-shared copy 全部放到
   `T.Pipelined` 的 producer stage 0，由编译器生成跨 chunk 双缓冲。
2. 将 WGMMA 的 `out` fragment 先重排到已经空闲的 `z_shared`，再由 shared memory
   合并写回 global output，消除原 fragment-to-global 写回中的大量额外 sectors。

wrapper 的自动选择规则为：

```text
T % 64 == 0: tilelang_residual_first_full_chunks, 默认 Q/K/V/A/g/beta 流水
T % 64 != 0: 原有带 predicate 的 tail-safe kernel
```

因此 8 个公开 case 中只有 `short_tail_state` 走旧 fallback，其余 7 个 case 走本轮 fast path。

## 2. 数学对象与 chunk 递推

固定 chunk 长度 `C=64`，key/query 维度 `Dk=128`，一个 CTA 处理 value 维的一段 `D`。
记进入当前 chunk 的 recurrent state 为 `S`，gate 为 `gamma=exp(g)`，则核心步骤为：

```text
R       = beta * (V - diag(gamma) * K*S)
Z       = A*R
O       = scale * diag(gamma) * Q*S
score   = causal(scale * diag(gamma) * Q*K^T * diag(1/gamma))
O      += score*Z
S_next  = gamma_last*S + K^T*(diag(gamma_last/gamma)*Z)
```

实现使用 `exp2(g*log2(e))`。每个 token 的 `gamma` 和 `1/gamma` 每个 chunk 只计算一次，
`gamma_last` 在并行 gate 循环结束后的统一控制流中从第 63 行读取。

## 3. Python wrapper 到 kernel launch

进入 `gdn_prefill_forward` 时，Q/K/V/g/beta/A/initial-state 已经是 CUDA tensor；
CPU 到 GPU HBM 的传输不在这个学生 forward 内发生。host wrapper 依次：

1. 从 tensor shape 取得 `B,T,Hq,Hv`，计算 `chunks_per_batch=ceil(T/64)`。
2. 按自动策略选择 value tile。当前只有 `chain_equal` 使用 `D/P=32/4`，其余为 `128/1`。
3. 在 GPU HBM 分配 BF16 `output[B,T,Hv,128]` 和 FP32
   `final_state[B,Hv,128,128]`。
4. 选择按 head 数、dtype、initial-state、D/P 和 prefetch profile 专门化并缓存的 JIT kernel。
5. 发起一次 kernel launch：

```text
grid.x  = B * Hv * P
block.x = 256 threads = 8 warps
```

每个 CTA 解码如下：

```text
owner   = block // P
dv_part = block % P
bb      = owner // Hv
bh      = owner % Hv
bhg     = bh // (Hv/Hq)
dv_left = dv_part * D
```

CTA 独占 `state[128,D]`、对应 output value slice 和 final-state slice，不需要跨 CTA
reduction 或同步。不同 chunk 依赖前一 chunk 的 `S_next`，所以 chunk loop 必须在 CTA 内串行。

## 4. Fragment 分配

fragment 是跨 256 threads 分布的逻辑 register tile，不是每个线程各持有完整矩阵：

| fragment | shape/dtype | 逻辑容量 |
| --- | ---: | ---: |
| `state` | `[128,D]` FP32 | `512D` B |
| `z` | `[64,D]` FP32 | `256D` B |
| `out` | `[64,D]` FP32 | `256D` B |
| `score` | `[64,64]` FP32 | 16 KiB |

总 fragment 容量为 `16 KiB + D KiB`：D32 为 48 KiB，D64 为 80 KiB，D128 为 144 KiB。
本轮前后 fragment shape 和容量不变。

## 5. Shared-memory 分配

| buffer | shape/dtype | D32 | D128 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `q_shared` | `[64,128]` BF16 | 16 KiB | 16 KiB | Q operand |
| `k_shared` | `[64,128]` BF16 | 16 KiB | 16 KiB | K operand |
| `v_shared` | `[64,D]` BF16 | 4 KiB | 16 KiB | V slice |
| `a_shared` | `[64,64]` BF16 | 8 KiB | 8 KiB | A operand |
| `z_shared` | `[64,D]` BF16 | 4 KiB | 16 KiB | Z、output staging、decayed Z 复用 |
| `state_shared` | `[128,D]` BF16 | 8 KiB | 32 KiB | state WGMMA operand |
| `score_shared` | `[64,64]` BF16 | 8 KiB | 8 KiB | gated causal score |
| `g_shared` | `[64]` FP32 | 256 B | 256 B | global g staging |
| `gamma_shared` | `[64]` FP32 | 256 B | 256 B | `exp(g)` |
| `inv_gamma_shared` | `[64]` FP32 | 256 B | 256 B | `1/exp(g)` |
| `beta_shared` | `[64]` FP32 | 256 B | 256 B | global beta staging |
| `gamma_last` | `[1]` FP32 | 4 B | 4 B | chunk-end decay |

Q/K/V/A/g/beta 是 stage-0 producers，因此 lowering 为其增加一个 pipeline stage；
state/z/score/gamma 不跨迭代双缓冲：

| 配置 | 单-stage 显式 shared | pipeline 额外 stage | lowering 前逻辑合计 |
| --- | ---: | ---: | ---: |
| D32/P4 | 65.004 KiB | 44.500 KiB | 109.504 KiB |
| D64/P2 | 81.004 KiB | 48.500 KiB | 129.504 KiB |
| D128/P1 | 113.004 KiB | 56.500 KiB | 169.504 KiB |

本轮前后的目标资源对比：

| 路径 | fragment | source pipelined shared | registers/thread | NCU dynamic shared/block | blocks/SM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 修改前 D32 K-only | 48 KiB | 80.754 KiB | 114 | 82.70 KiB | 2 |
| 修改后 D32 all-input | 48 KiB | 109.504 KiB | 112 | 112.144 KiB | 2 |
| 修改前 D128 no-prefetch | 144 KiB | 112.754 KiB | 202 | 115.47 KiB | 1 |
| 修改后 D128 all-input | 144 KiB | 169.504 KiB | 208 | 173.584 KiB | 1 |

D32 虽增加 shared，仍可达到 2 blocks/SM；D128 修改前已经被 registers/shared 限制为
1 block/SM，所以本轮额外 stage 没有再次降低 residency。

## 6. Pipeline prologue 与 steady state

`T.Pipelined` 把 Q/K/V/A/g/beta copy 放在 stage 0，其余操作放在 stage 1。prologue 对
chunk 0 提交：

```text
Q [64,128] BF16 -> q_shared[next]
K [64,128] BF16 -> k_shared[next]
V [64,D]   BF16 -> v_shared[next]
A [64,64]  BF16 -> a_shared[next]
g/beta [64] FP32 -> g_shared[next]/beta_shared[next]
commit async-copy group
```

当前生成代码使用 `LDGSTS`/`cp.async`，没有使用 TMA。global load 先查询 L2，L2 miss 才从
HBM 取数；kernel 不会显式执行“HMB 到 L2”的 copy。cp.async 让 global load 的返回值直接进入
shared，并允许 warp 在结果返回前继续执行无依赖工作。

steady state 中，CTA 计算 chunk `c` 时，另一组 shared stage 已经接收 chunk `c+1` 的输入。
第一次消费当前输入前，lowering 插入 async wait/`BAR.SYNC.DEFER_BLOCKING`。这样 V latency
可由 state copy 和 `K@S` 覆盖，A latency 可由 `K@S`、residual 与 Z materialization 覆盖，
Q/K 在各自后续 GEMM 前也获得更长 prefetch distance。

TileLang 当前要求同一 consumer 的异步 producers 处于同一个 producer stage。将 Q/K 提前而把
V/g/beta 留在另一个 async stage 会生成错误 wait count，所以本轮使用统一的全输入 producer stage。

## 7. 单个 chunk 的完整时序

输入 wait 完成后，每个完整 chunk 严格按以下流程执行：

1. 将 FP32 `state` fragment 转成 BF16，写入 `state_shared[128,D]`。它依赖上一 chunk，
   不能随输入一起跨 chunk prefetch。
2. 从 `g_shared` 计算 64 个 `gamma=exp2(g*LOG2E)` 和 `inv_gamma=1/gamma`，再取
   `gamma_last=gamma[63]`。
3. GEMM 1：`z = K @ state`，即 `[64,128] x [128,D]`，FP32 accumulate。
4. Elementwise：`z = beta * (V - gamma*z)`，然后把 FP32 fragment 转成 BF16 写 `z_shared`。
5. GEMM 2：`z = A @ z_shared`，得到修正 Z，再写回 `z_shared`。
6. GEMM 3：`out = Q @ state`，逐 row 乘 `scale*gamma`。
7. GEMM 4：`score = Q @ K^T`，施加 causal mask、scale 和
   `gamma[row]*inv_gamma[col]`，转 BF16 写 `score_shared`。
8. GEMM 5：`out += score_shared @ z_shared`。
9. 将 `out` fragment 写到已经不再被 GEMM 5 消费的 `z_shared`，再执行合并的
   shared-to-global copy 到本 CTA 的 output slice。该复用不增加 shared allocation。
10. Elementwise：`state *= gamma_last`；`z *= gamma_last*inv_gamma[row]`，把 decayed Z
    写回 `z_shared`。
11. GEMM 6：`state += K^T @ z_shared`。更新后的 FP32 state fragment 留在 registers，
    直接成为下一 chunk 的输入。
12. 当前 stage 完成最后一次 K 消费后才允许被后续迭代覆盖；与此同时下一 chunk 的 async
    输入通常已经提交。

epilogue 排空最后一组 async copy，不发起越界输入。所有 chunks 完成后，CTA 将 FP32
`state[128,D]` 写入独占的 global final-state slice，kernel 结束并返回 output/final-state。

## 8. 内存访问总结

| 时点 | 路径 | 是否可与计算重叠 |
| --- | --- | --- |
| wrapper | HBM 中分配 output/final-state | kernel launch 前 |
| pipeline stage 0 | Q/K/V/A/g/beta：global(L2/HBM) -> shared | 是，和前一 chunk 计算重叠 |
| chunk 开始 | state fragment -> `state_shared` | 否，依赖上一 chunk state |
| GEMM 间 | fragment -> shared：Z、score | 作为下一 GEMM 的同步边界 |
| output | out fragment -> `z_shared` -> global | 合并 store；复用 shared |
| kernel 结束 | state fragment -> global final-state | 只发生一次 |

g/beta 的布局是 `[B,T,H]`。一个固定-head CTA 跨 token 访问时带 H stride，所以剩余 global
excess sectors 主要来自这两项。若在当前 CTA 内强行读取连续 head，会引入无用数据；本轮选择通过
async stage 隐藏 latency。

## 9. Profile 结果与原始文件

最终 D32 `chain_equal` full profile：

- Nsight Compute 报告：`output/ncu_full_chain_warp.ncu-rep`
- 同次采集日志：`output/ncu_full_chain_warp_58723.log`
- 采集方式：NCU `--set full`、显式增加 `PmSampling_WarpStates`、嵌入可用 source，47 passes
- profiled duration：624.67 us
- registers：112/thread
- dynamic shared：112.144 KiB/block
- theoretical / achieved occupancy：25.0% / 14.53%
- scheduler cycles：`one-or-more eligible=29.42%`，`no eligible=70.58%`
- Warp State Statistics not-issued samples：`long_scoreboard=2922`，`barrier=2542`，
  `short_scoreboard=818`，`wait=807`，`mio_throttle=194`，`warpgroup_arrive=57`
- Warp cycles per issued instruction：7.71
- DRAM throughput：63.56 GB/s
- global excessive sectors：98304

修改前 K-only D32 为 766.85 us、114 registers/thread、82.70 KiB dynamic shared、
7602 long-scoreboard samples 和 360448 excessive sectors。最终版本把 duration 缩短 1.246x，
long-scoreboard samples 降低 60.1%，excessive sectors 降低 72.7%。

最终 D128 `wide_gva_state` full profile：

- Nsight Compute 报告：`output/ncu_full_wide_warp.ncu-rep`
- 同次采集日志：`output/ncu_full_wide_warp_58733.log`
- 采集方式：NCU `--set full`、显式增加 `PmSampling_WarpStates`、嵌入可用 source，47 passes
- profiled duration：3.80 ms
- registers：208/thread
- dynamic shared：173.584 KiB/block
- theoretical / achieved occupancy：12.5% / 12.5%
- scheduler cycles：`one-or-more eligible=24.09%`，`no eligible=75.91%`
- Warp State Statistics not-issued samples：`barrier=22544`，`long_scoreboard=17398`，
  `short_scoreboard=6797`，`wait=4256`，`mio_throttle=3611`，`warpgroup_arrive=247`
- Warp cycles per issued instruction：8.21
- grid：64 blocks，4.57 waves

两个报告均实际包含 Speed of Light/roofline、PM Sampling、PM Sampling Warp States、Compute、
Memory Workload、Scheduler、Warp State Statistics、Instruction、Launch、Occupancy、Workload
Distribution 和 Source Counters。H800 MIG 无权采集属于共享 GPU 单元的 14 个 PCIe/CTC metrics；
NCU 会报告这些指标 unavailable，其余当前实例可访问的 full-set metrics 均已采集。

D32 当前 `long_scoreboard` 与 barrier 已同量级；DRAM throughput 只有峰值的约 24.9%，
Tensor pipeline active 约 11.4%。所以继续增加同类 ping-pong buffer 不再是首选方向。

## 10. 完整 8-case 结果

baseline 是本轮修改前提交 `e9bb60a` 的 K-only 自动分派，日志为
`output/memopt_baseline_58343.log`。最终日志为 `output/memopt_final_staged_58438.log`。
两者均为 10 warmups、100 repetitions、CUDA event 中位数；所有 output 和 final state 检查 PASS。

| case | baseline ms | final ms | speedup | `p=t100/t` | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.140864 | 0.142048 | 0.9917x | 2.4361 | 120.00 |
| `chain_equal` | 0.783488 | 0.650736 | 1.2040x | 0.7650 | 89.26 |
| `parallel_equal` | 0.547600 | 0.413312 | 1.3249x | 1.2370 | 104.74 |
| `parallel_gva` | 0.497008 | 0.397344 | 1.2508x | 1.2371 | 104.74 |
| `long_low_gva` | 3.602192 | 3.195232 | 1.1274x | 0.5817 | 81.04 |
| `batch_split_gva` | 2.792480 | 2.267488 | 1.2315x | 0.6757 | 85.00 |
| `wide_gva_state` | 5.114848 | 3.802176 | 1.3452x | 0.6382 | 83.59 |
| `deep_gva_state` | 5.791072 | 4.420480 | 1.3101x | 0.6404 | 83.57 |

预计分数使用公开 60/100 turning points 做分段线性插值：

```text
p <= p60:       score = 60 * p / p60
p60 < p <= 1:   score = 60 + 40 * (p-p60) / (1-p60)
p > 1:          score = min(120, 100 + 20 * (p-1))
```

该估计不代表隐藏 case 官方结果。公开 8 case 简单平均约 93.99；当前达到 100+ 的是
`short_tail_state`、`parallel_equal` 和 `parallel_gva`。

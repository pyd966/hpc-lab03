# Compute-order and explicit-WGMMA walkthrough

## 1. 本轮范围与结论

本轮只修改 full-chunk fast path，也就是 `num_tokens % 64 == 0` 时走的
`tilelang_residual_first_full_chunks`。尾块路径仍使用 `tilelang_residual_first`，因此
`short_tail_state` 不经过新计算顺序。

本轮最终保留四项变化：

1. stage-0 的 `g/beta` strided load 先于 Q/K/V/A 发射，用后续较大的连续 copy 隐藏 gate load
   latency。
2. 六个 GEMM 改为显式 `T.wgmma_gemm`，并用 `T.warpgroup_wait` 控制 group retirement。
3. 递归关键路径优先：`K@S -> residual -> A@residual` 排在 `Q@S/QK` 前，避免 Q/score
   阻塞下一 chunk 必须使用的 state update。
4. 使用 `zhat` 等价变换，让 output update 和 state update 读取同一份 shared tile，从而在不增加
   shared memory 的情况下并发发射最后两个 WGMMA group。

正式 8 case 全部正确。相对本轮修改前的 all-input pipeline，正式中位数在 6 个 full-chunk case
上提升，`long_low_gva` 提升 1.0801x；`deep_gva_state` 回退 1.6%，tail fallback 的差异是 1.8%。
公开 case 的预计简单平均从 93.99 增加到 94.76。当前仍不是“全部 100+”版本。

## 2. 数学等价变换

每个 chunk 长度 `C=64`，key dim `K=128`，当前 block 负责的 value 宽度记为 `D=dv_tile`。
进入 chunk 时的递归 state 记为 `S`。

原始 residual-first 形式为：

```text
gamma_i = exp(g_i)
R        = beta * (V - diag(gamma) * K*S)
Z        = A*R
O_base   = scale * diag(gamma) * Q*S
P_ij     = causal(i,j) * scale * gamma_i/gamma_j * (Q*K^T)_ij
O        = O_base + P*Z
S_next   = gamma_last*S + K^T*(diag(gamma_last/gamma)*Z)
```

定义：

```text
zhat_j = gamma_last/gamma_j * Z_j
```

则 output update 的每一项满足：

```text
(gamma_i/gamma_j) * Z_j
= (gamma_i/gamma_last) * zhat_j
```

所以可以改写为：

```text
P_hat_ij = causal(i,j) * scale * gamma_i/gamma_last * (Q*K^T)_ij
O        = O_base + P_hat*zhat
S_next   = gamma_last*S + K^T*zhat
```

这样最后两个 GEMM 都读取 `zhat`，不再需要同时保存 raw Z 和 decayed Z。该变换经过 8 个公开
case 的 BF16 output 与 FP32 final-state 检查，全部 PASS。

## 3. 从 Python 调用到 kernel launch

`gdn_prefill_forward` 在 CPU 上完成以下 dispatch：

1. 从 Q/V shape 得到 `B, T, Hg, H`，计算 `chunks_per_batch = ceil(T/64)`。
2. 在 GPU 上分配 BF16 `output[B,T,H,128]` 和 FP32
   `final_state[B,H,128,128]`。
3. `GDN_DV_SPLIT=auto` 时，仅当 `chunks>=64`、state owner 少于 14 个 MIG SM，并且四分后 block
   数不超过 28 时选择 `(dv_tile,dv_parts)=(32,4)`；否则使用 `(128,1)`。
4. 只有 `T % 64 == 0` 才进入本轮 full-chunk fast path。默认
   `GDN_MEMORY_IO=auto,GDN_PREFETCH=auto` 会选择 Q/K/V/A 全输入 pipeline。
5. JIT specialization 包含 H、Hg、dtype、是否有 initial state、dv tile/parts 和 prefetch flags。
6. launch grid 为 `B*H*dv_parts` 个 CTA，每个 CTA 256 threads，也就是两个 warpgroup。

PyTorch 输入在 kernel launch 之前已经是 CUDA tensor。CPU 在这里传递 device pointer 和动态 shape，
kernel 内没有 CPU memory 到 HBM 的传输。若输入最初来自 CPU，host-to-device copy 发生在调用本函数
之前的 tensor 创建或 `.to("cuda")` 阶段。

## 4. CTA ownership 与 head 映射

每个 block 计算：

```text
owner   = block // dv_parts
dv_part = block % dv_parts
bb      = owner // H
bh      = owner % H
bhg     = bh // (H/Hg)
dv_left = dv_part * dv_tile
```

因此一个 CTA 独占 `[128,D]` 的 state slice，并串行遍历该 owner 的全部 chunks。不同 value 列没有
递归依赖，dv split 才能成立。GVA 中多个 value heads 映射到同一个 Q/K head，但每个 value head 的
V/A/g/beta/state/output 仍独立。

## 5. Fragment 分配：修改前与修改后

源码中的四个 FP32 fragment 没有增加或删除：

| fragment | logical shape | D=32 | D=64 | D=128 | lifetime |
| --- | --- | ---: | ---: | ---: | --- |
| `state` | `[128,D]` | 16 KiB | 32 KiB | 64 KiB | 跨全部 chunks |
| `z` | `[64,D]` | 8 KiB | 16 KiB | 32 KiB | residual、Z、zhat |
| `out` | `[64,D]` | 8 KiB | 16 KiB | 32 KiB | base output 到最终写回 |
| `score` | `[64,64]` | 16 KiB | 16 KiB | 16 KiB | QK 到 score shared copy |
| **源码 fragment 总量** |  | **48 KiB** | **80 KiB** | **144 KiB** | 每 block 分布量 |

这些字节是整个 CTA 的逻辑 fragment 总量，不表示每个 thread 持有完整矩阵。generated CUDA 中：

```text
D=32:  state[16], z[8],  out[8],  score[16] FP32/thread
D=128: state[64], z[32], out[32], score[16] FP32/thread
```

本轮前后的源码 fragment 总量完全相同，但较短的 live range 让 NCU 的物理寄存器分配下降：

| specialization | 修改前 regs/thread | 修改后 regs/thread | 变化 |
| --- | ---: | ---: | ---: |
| D=32 chain | 112 | 105 | -7 |
| D=128 wide | 208 | 206 | -2 |

这里没有通过减少数学 accumulator 精度换寄存器，所有 WGMMA accumulation 仍是 FP32。

## 6. Shared memory 分配：修改前与修改后

full-chunk path 的 stage-0 Q/K/V/A/g/beta 会由 pipeline lowering 扩成两个 stage。下表是 lowering
后的物理 shared 占用；本轮前后完全相同：

| buffer | dtype/shape | D=32 | D=64 | D=128 | stage |
| --- | --- | ---: | ---: | ---: | --- |
| `q_shared` | BF16 `[64,128]` | 32 KiB | 32 KiB | 32 KiB | ping-pong |
| `k_shared` | BF16 `[64,128]` | 32 KiB | 32 KiB | 32 KiB | ping-pong |
| `v_shared` | BF16 `[64,D]` | 8 KiB | 16 KiB | 32 KiB | ping-pong |
| `a_shared` | BF16 `[64,64]` | 16 KiB | 16 KiB | 16 KiB | ping-pong |
| `z_shared` | BF16 `[64,D]` | 4 KiB | 8 KiB | 16 KiB | single |
| `state_shared` | BF16 `[128,D]` | 8 KiB | 16 KiB | 32 KiB | single |
| `score_shared` | BF16 `[64,64]` | 8 KiB | 8 KiB | 8 KiB | single |
| `g_shared` | FP32 `[64]` | 0.5 KiB | 0.5 KiB | 0.5 KiB | ping-pong |
| `beta_shared` | FP32 `[64]` | 0.5 KiB | 0.5 KiB | 0.5 KiB | ping-pong |
| `gamma_shared` | FP32 `[64]` | 0.25 KiB | 0.25 KiB | 0.25 KiB | single |
| `inv_gamma_shared` | FP32 `[64]` | 0.25 KiB | 0.25 KiB | 0.25 KiB | single |
| `gamma_last` | FP32 `[1]` | 0.004 KiB | 0.004 KiB | 0.004 KiB | single |
| **total** |  | **109.504 KiB** | **129.504 KiB** | **169.504 KiB** |  |

NCU 用十进制 Kbyte 显示 D=32 为 112.14 Kbyte/block，D=128 为 173.58 Kbyte/block，与上面的
109.504/169.504 KiB 一致。本轮没有添加 `z_state_shared`。这是一个重要约束：D=32 当前能达到
25% theoretical occupancy；多加 4 KiB/block 会让两个 CTA 的 shared 总量超过单 SM 上限，理论
occupancy 会降到 12.5%。zhat 变换避免了该回退。

## 7. HBM、L2、shared 与 prefetch

### 7.1 每个 chunk 的 global traffic

每个 CTA 每 chunk 从 global memory 读取：

```text
Q:     64*128*2 = 16 KiB
K:     64*128*2 = 16 KiB
A:      64*64*2 =  8 KiB
V:       64*D*2
g+beta: 2*64*4  = 0.5 KiB
```

并写出 `64*D*2` bytes output。state 只在 kernel prologue/epilogue 各访问一次 global memory，
每次 `128*D*4` bytes，不是每 chunk 访问。

D=128 每 CTA 每 chunk 的输入加 output 约 72.5 KiB；D=32 每 CTA 约 48.5 KiB。dv=32 一个 head
使用四个 CTA，Q/K/A/g/beta 和 QK GEMM 会被四份重复，这是用额外工作量换低 owner case 并行度。

### 7.2 数据经过哪些层级

TileLang 的 stage-0 copy lower 为 `cp.async`/等价 async global-to-shared copy。它不是“每个 chunk
显式把 HBM 搬到 L2”的 API：global load 总是先查询 L2，cache miss 才由硬件从 HBM 填入 L2，再
送到 shared。L2 命中时不会访问 HBM。

本轮把 physically strided 的 g/beta copy 放在 pipeline order 的最前面，然后发射 Q/K/V/A 的较大
copy。这样 gate cache miss 可以和后续 copy 的发射重叠。pipeline 结构为：

```text
prologue:    async load chunk 0 into stage 0
steady n:   async load chunk n+1 into alternate stage
            compute chunk n from current stage
epilogue:    compute final prefetched chunk
```

Q/K/V/A/g/beta 才有 next-chunk prefetch。state、z、score、gamma 是当前 chunk 的生产结果，不能
跨递归边界预取。每次 Tensor Core 消费 shared operand 前仍需要 async-copy consumer barrier；该
barrier 与 WGMMA accumulator wait 是两种不同同步。

## 8. 单个 chunk 的完整时间线

下面按最终 generated CUDA 的实际依赖顺序描述 steady-state chunk。`G0` 到 `G5` 是六个 committed
WGMMA groups。

| phase | operation | fragment/shared/global activity | wait condition |
| --- | --- | --- | --- |
| 0 | prefetch next chunk | g/beta 先发，随后 Q/K/V/A 写 alternate shared stage | 当前 stage 消费前 async-copy barrier |
| 1 | state operand | `state` FP32 fragment 转 BF16 写 `state_shared` | shared operand ready 后进入 WGMMA |
| 2 | G0 `K @ S` | K/state shared -> FP32 `z` | 不立即等待 |
| 3 | gate work | `exp2(g*log2e)`、`1/gamma`、`gamma_last` 写 shared | 与 G0 重叠 |
| 4 | old-state decay | FP32 `state *= gamma_last` | G0 读的是已物化的旧 `state_shared`，无冲突 |
| 5 | consume G0 | residual 首次读取 `z` | `warpgroup_wait<0>` |
| 6 | residual | `z = beta*(V-gamma*z)` | V/beta/gamma shared -> z fragment |
| 7 | residual operand | z fragment 转 BF16 写 `z_shared` | A GEMM operand boundary |
| 8 | G1 `A @ residual` | A/z shared -> FP32 `z` | 最老 group，递归关键路径优先 |
| 9 | G2 `Q @ S` | Q/state shared -> FP32 `out` | 排在 G1 后 |
| 10 | G3 `Q @ K^T` | Q/K shared -> FP32 `score` | 排在 G2 后 |
| 11 | consume G1 | 只要求 A 完成，Q/QK 继续 in flight | `warpgroup_wait<2>` |
| 12 | build zhat | `z *= gamma_last*inv_gamma`; BF16 写 `z_shared` | 覆盖 G2/G3 的尾部 latency |
| 13 | consume G2/G3 | out/score 即将首次读取 | `warpgroup_wait<0>` |
| 14 | output base | `out *= scale*gamma[row]` | FP32 fragment elementwise |
| 15 | causal score | lower mask；`score *= scale*gamma[row]/gamma_last` | FP32 fragment elementwise |
| 16 | score operand | score fragment 转 BF16 写 `score_shared` | output update operand boundary |
| 17 | G4 `score @ zhat` | score/zhat shared，累加 FP32 `out` | 与 G5 连续发射 |
| 18 | G5 `K^T @ zhat` | K/zhat shared，累加 FP32 `state` | 与 G4 outstanding |
| 19 | consume G4 only | output 即将写回，state 尚不消费 | `warpgroup_wait<1>` |
| 20 | output store | out fragment -> `z_shared` -> global output | G5 在写回期间继续执行 |
| 21 | recurrent boundary | 下一 chunk 即将把 state 写到 `state_shared` | `warpgroup_wait<0>` |

最后一个 chunk 走 pipeline epilogue 的同构代码。循环结束后，最终 FP32 `state` fragment 直接 copy
到 `final_state[B,H,128,dv_slice]`。

## 9. 为什么没有完全照提议的源码顺序

最初候选按 `K@S, Q@S, gamma, state decay, QK, residual, A@z` 发射。它让 Q/QK 排在 A 前面，
但硬件 Tensor Core queue 仍按 group 顺序执行；下一 chunk 的 state 依赖 A，因此这种顺序把递归
关键路径推迟了。20-repetition smoke 结果为：

| candidate | chain ms | long ms | wide ms | conclusion |
| --- | ---: | ---: | ---: | --- |
| gate-first，普通同步 GEMM | 0.6368 | 2.9163 | 3.6712 | gate load 顺序有效，但无 compute overlap |
| K,Q,QK,A explicit groups | 0.6411 | 3.1148 | 3.9227 | A 被 Q/QK 阻塞 |
| 同顺序加 zhat | 0.6316 | 3.1302 | 4.0070 | 双尾 GEMM可并发，但前半关键路径仍差 |
| K,Q,A,QK | 0.6261 | 3.0106 | 3.7816 | A 前移后改善 |
| **K,A,Q,QK recurrent-first** | **0.6077** | **2.9759** | **3.7900** | 最终顺序 |
| Q/QK 再拆一次 partial wait | 0.6355 | 3.0305 | 3.7882 | 多一次 wait 开销大于覆盖收益 |

这些是 5 warmups、20 repetitions 的候选筛选，不替代第 12 节正式 10/100 数据。最终顺序仍遵守
“只有结果首次被消费前才等待”，但优先缩短跨 chunk 的递归链，而不是只按公式书写顺序排列。

## 10. generated CUDA 与 wait 证据

不能只看 TileLang 源码。本轮分别导出了 D=32 和 D=128 的完整 generated CUDA：

```text
output/order_generated_dv32_59630.log
output/order_generated_dv128_59690.log
```

D=32 steady-state 主循环中实际生成的 group wait 为：

```text
tl::warpgroup_wait<0>();  // G0 -> residual
tl::warpgroup_wait<2>();  // retire G1, keep G2/G3
tl::warpgroup_wait<0>();  // G2/G3 -> out/score elementwise
tl::warpgroup_wait<1>();  // retire G4, keep G5 during output store
tl::warpgroup_wait<0>();  // G5 -> next chunk state use
```

早期实现使用 `T.wait_wgmma`，generated SASS 中无法确认非零 wait count。最终改用直接 lower 为
`tl.warpgroup_wait` 的 `T.warpgroup_wait`，generated CUDA 明确保留 `<2>` 与 `<1>`。NCU 的 SASS
source page 将这些 PTX-level waits 显示为 `WARPGROUP.DEPBAR.LE gsb0,0x0`；该 SASS immediate
不能直接当作源码 wait count 解读，因此 group count 的核对以 generated CUDA template argument
为准，SASS 用于确认 wait/arrive/commit 的实际存在与 PC stall attribution。

## 11. 每 chunk 计算量

按一个 CTA、一个 chunk 计，六个 GEMM 的 MAC 数为：

| GEMM | MACs |
| --- | ---: |
| `K@S` | `64*128*D` |
| `A@residual` | `64*64*D` |
| `Q@S` | `64*128*D` |
| `Q@K^T` | `64*64*128` |
| `score@zhat` | `64*64*D` |
| `K^T@zhat` | `128*64*D` |
| **total** | **`32768*D + 524288` MACs** |

D=128 是 4,718,592 MACs，约 9.437 MFLOP；D=32 是 1,572,864 MACs，约 3.146 MFLOP/block。
dv=32 的四个 part 合计约 6.291M MACs/head/chunk，因为 value-independent 的 QK 被重复四次，
比未拆分多 33.3% GEMM MACs。

## 12. 正式 8-case 时间、speedup 与预计分数

修改前基线是 `output/memopt_final_staged_58438.log`。最终版本分成两个作业，避免一个 Python
进程连续编译多个 D128 specialization 时被集群结束：

```text
output/order_final_main_59637.log
output/order_final_state_59655.log
```

两边都使用默认 10 warmups、100 repetitions 和 CUDA event median。8 个 case 的 output/final state
全部 PASS。

| case | previous ms | final ms | speedup | `p=t100/t` | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.142048 | 0.144704 | 0.9816x | 2.3914 | 120.00 |
| `chain_equal` | 0.650736 | 0.614640 | 1.0587x | 0.8099 | 91.31 |
| `parallel_equal` | 0.413312 | 0.408720 | 1.0112x | 1.2509 | 105.02 |
| `parallel_gva` | 0.397344 | 0.387504 | 1.0254x | 1.2685 | 105.37 |
| `long_low_gva` | 3.195232 | 2.958256 | 1.0801x | 0.6283 | 83.16 |
| `batch_split_gva` | 2.267488 | 2.214960 | 1.0237x | 0.6917 | 85.74 |
| `wide_gva_state` | 3.802176 | 3.707360 | 1.0256x | 0.6546 | 84.34 |
| `deep_gva_state` | 4.420480 | 4.491200 | 0.9843x | 0.6304 | 83.11 |

预计分数沿用公开 60/100 turning points 的分段线性估算：

```text
p <= p60:       score = 60 * p / p60
p60 < p <= 1:   score = 60 + 40 * (p-p60) / (1-p60)
p > 1:          score = min(120, 100 + 20 * (p-1))
```

公开 8 case 的预计简单平均为 **94.76**，上一版为 93.99。达到 100+ 的仍是 tail、
`parallel_equal` 和 `parallel_gva`。该估算不代表隐藏 case 的官方结果。

## 13. 完整 NCU profile 与资源结果

可复现命令封装在 `profile_full.sh`，配置为：

```text
--set full
--section PmSampling_WarpStates
--import-source yes
--clock-control none
--replay-mode kernel
--launch-count 1
```

两份最终报告均完成 47 passes，并包含 Warp State Statistics：

```text
output/ncu_order_chain_full.ncu-rep
output/ncu_order_chain_full_59614.log
output/ncu_order_chain_full_details.txt
output/ncu_order_chain_full_source.csv

output/ncu_order_wide_full.ncu-rep
output/ncu_order_wide_full_59674.log
output/ncu_order_wide_full_details.txt
output/ncu_order_wide_full_source.csv
```

H800 MIG 无法收集 14 个由多个 MIG instance 共享的 PCIe/CTC 指标；NCU 明确列出这些 unavailable
metrics，其他 full sections 与 sampling sections 均已收集。

### 13.1 D=32 chain

| metric | previous | final | observation |
| --- | ---: | ---: | --- |
| NCU duration | 624.67 us | 579.87 us | 1.0773x |
| regs/thread | 112 | 105 | live range 缩短 |
| dynamic shared | 112.14 Kbyte | 112.14 Kbyte | 不变 |
| theoretical occupancy | 25.0% | 25.0% | 不变 |
| achieved occupancy | 14.53% | 14.33% | 基本不变 |
| long scoreboard | 2.21 CPI | 1.90 CPI | -14.0% |
| barrier | 2.07 CPI | 3.13 CPI | 显式 retirement 增加 barrier |
| not-issued long-scoreboard samples | 2922 | 2770 | -5.2% |
| not-issued barrier samples | 2542 | 3756 | +47.8% |

final Warp State Statistics 还包括 short-scoreboard 428、wait 900、MIO throttle 25、
warpgroup-arrive 203 个 not-issued samples；`No Eligible` 为 71.96%，warp cycles per issued
instruction 为 8.11。

### 13.2 D=128 wide

| metric | previous | final | observation |
| --- | ---: | ---: | --- |
| NCU duration | 3.80 ms | 3.78 ms | 小幅改善 |
| regs/thread | 208 | 206 | -2 |
| dynamic shared | 173.58 Kbyte | 173.58 Kbyte | 不变 |
| theoretical/achieved occupancy | 12.5% | 12.5% | 单 CTA/SM |
| long scoreboard | 1.95 CPI | 1.62 CPI | -16.9% |
| barrier | 2.61 CPI | 3.05 CPI | +16.9% |
| not-issued long-scoreboard samples | 17398 | 14493 | -16.7% |
| not-issued barrier samples | 22544 | 24733 | +9.7% |

final Warp State Statistics 还包括 short-scoreboard 7305、wait 4387、MIO throttle 1970、
warpgroup-arrive 1010 个 not-issued samples；`No Eligible` 为 75.60%，warp cycles per issued
instruction 为 8.24。

### 13.3 Profile 结论

本轮确实降低了 long-scoreboard CPI，尤其 D=128 降低约 17%；代价是显式 group retirement 把一部分
等待转移为 barrier。D=32 的计算重排与较低寄存器数使总时长仍明显下降；D=128 中 barrier 增长抵消
了大部分 scoreboard 收益，这解释了 wide 只有约 2.6% 正式提升、deep 甚至小幅回退。

下一轮若继续沿该方向，重点不应是再插更多 wait，而应减少 shared/Tensor consumer barrier，或者
让 barrier 前有更多真正独立的 SIMT/global-store 工作。另一个方向是降低 D=128 的 206 regs/thread，
但在 shared 已限定为单 CTA/SM 时，只有把资源降到足以改变 occupancy 或明显降低 spill/issue cost
才有价值。

# GDN prefill kernel walkthrough: D=64 双 CTA residency

本文描述当前工作树中的完整实现。本轮只优化 D=64 RS 路径，D=128 没有做
fragment/shared-memory 改造。结论、执行顺序和资源数字均来自最终 TileLang
源码、生成 CUDA、14 个 synthetic 边界 case、正式 8 case 和包含 Warp State
Statistics 的完整 NCU profile。

## 1. 本轮结论

本轮找到的最低成本 shared-memory 优化是复用两个生命周期不重叠、形状完全相同的
BF16 buffer：

```text
score_shared  [64,64] BF16 -- G4 读完后生命周期结束
output_shared [64,64] BF16 -- G4 结束后才开始写
```

D=64 下两者都是 8 KiB。compact specialization 让它们指向同一块 shared，动态
shared 从 113.504 KiB 降到 105.504 KiB；fragment 不变，仍为 104 KiB
logical fragment/block。NCU 实测仍为 255 registers/thread，但 shared 和
register 两项都允许 2 blocks/SM：

| resource | 修改前 D=64 fast | 修改后 D=64 compact |
| --- | ---: | ---: |
| logical fragment/block | 104 KiB | 104 KiB |
| registers/thread | 255 | 255 |
| dynamic shared/block | 113.504 KiB | 105.504 KiB |
| driver shared/block | 1,024 B | 1,024 B |
| register block limit | 2 | 2 |
| shared-memory block limit | 1 | 2 |
| resident blocks/SM | 1 | 2 |
| theoretical occupancy | 6.25% | 12.5% |

修改前两个 block 需要
`2*(116,228 dynamic + 1,024 driver)=234,504 B`，超过每个 SM 的
233,472 B，差 1,032 B，即每 block 至少要省 516 B。本轮实际每 block 省
8,192 B；修改后两个 block 共需 218,120 B。寄存器侧两个 block 使用
`2*128*255=65,280` 个 32-bit register，也刚好低于 65,536 上限。

2 blocks/SM 对强制 D=64 的 owner>=8 case 有实际收益。例如 owners=8、
128 chunks 从 0.697312 ms 降到 0.612016 ms，提升 1.139x。但它仍慢于
D=128 的 0.511632 ms，因此最终 auto dispatch 仍保持 owners=7/8 的原边界。
本轮采用双特化：

- owners<=7 的自动 D=64 继续使用原来的 1-block fast specialization，不支付
  compact layout 的任何代价。
- 强制 D=64 且 `2*owners>14` 时使用 2-block compact specialization。
- auto 的高 owner case 继续使用更快的 D=128。

这不是按 public case 过拟合：决策来自固定 Hq=Hv=1、独立扫描 owners=6--14 和
chunks=32/128 的 synthetic 边界测试。

## 2. 尝试过的方案

本轮实际尝试了多种 fragment/shared-memory 缩减方案：

| 方案 | 资源效果 | 性能/正确性结果 | 决策 |
| --- | --- | --- | --- |
| output 直接写 global，删除 output buffer | -8 KiB shared | chain 约 0.462 ms，较原 0.376 ms 慢约 23% | 拒绝 |
| 关闭全部 ping-pong prefetch | 大幅减 shared | 全部正确但明显变慢 | 拒绝 |
| 只关闭 A ping-pong | -8 KiB 理论值 | TileLang pipeline lowering 失败 | 拒绝 |
| g/beta/gamma/inverse 合并 gate workspace | 可跨过 2 CTA 门槛 | 生命周期变长、同步增加，边界性能差 | 拒绝 |
| g 或 beta 直接 global read | 减少小 gate buffer | chain 约 0.518 ms | 拒绝 |
| gate 改 FP16 | 少量 shared | chain 约 0.467 ms | 拒绝 |
| V 仅预取 56/60 rows | 减少部分 V ping-pong | chain 约 0.530/0.704 ms | 拒绝 |
| aggressive storage merge pass | 期望编译器自动复用 | 实际 shared 反而增加且无收益 | 拒绝 |
| score/output 生命周期别名 | -8 KiB shared，fragment 不变 | 14 个边界 case 全部 PASS，owner>=8 D64 加速 | 采用 |

这些结果说明 gate buffer 不是好的第一目标。g/beta 每个 stage 仅 256 B，但当前
chunk 消费与下一 chunk prefetch 的生命周期重叠；删除 ping-pong 会破坏软件流水
或把 global latency 放回关键路径。相反，score 和 output 各有 8 KiB，且边界明确：
G4 完成后 score 不再使用，output 才开始 materialize。

实验文件包括：

- `output/dv64_direct_out_gamma_sync_boundary.log`
- `output/dv64_prefetch_off_public_63586.log`
- `output/dv64_v56_smoke_63651.log`
- `output/dv64_v60_smoke_63659.log`
- `output/dv64_gate_workspace_chain_full.ncu-rep`
- `output/ncu_dv64_aggressive_merge_chain_full.ncu-rep`
- `output/dv64_wait1_single_writer_boundary.log`

## 3. Host dispatch 和 kernel launch

完整 forward 在 `run.py` 中先调用 `chunk_local_cumsum` 生成累计 gate，再调用
`kkt_solve` 生成 A，最后进入 `gdn_prefill_forward`。到 recurrent kernel launch
时，Q/K/V/g/beta/A/initial_state 都已经是 CUDA tensor，位于 GPU HBM。CPU memory
到 HBM 的传输发生在 PyTorch tensor 创建或 `.to("cuda")` 阶段，不在本 kernel
内部发生。

host 计算：

```text
owners = B * Hv
full_chunks_only = (T % 64 == 0)
```

最终 auto 规则仍是：

```text
full_chunks_only and 2*owners <= 14 SM -> D=64 RS, dv_parts=2
otherwise                             -> D=128 RS, dv_parts=1
```

对于 RS kernel，compact 标志为：

```text
reuse_output_shared = (dv_tile == 64 and owners*dv_parts > 14)
```

所以 auto 的 owners<=7 D=64 编译为 fast specialization；`GDN_DV_SPLIT=64`
强制高 owner 扫描时才编译 `_reuse_so` compact specialization。

D=64 launch：

```text
grid.x = B * Hv * 2
blockDim.x = 128 threads = 4 warps = 1 warp group
owner   = blockIdx.x / 2
dv_part = blockIdx.x % 2
bb      = owner / Hv
bh      = owner % Hv
bhg     = bh / (Hv/Hq)
dv_left = dv_part * 64
```

每个 CTA 独占一个 `[64 value dims, 128 key dims]` state slice，并顺序处理该
owner 的所有 chunks。不同 chunk 之间存在 recurrent state 依赖，因此 chain
不能沿时间拆给多个 CTA；D=64 并行来自把 128 个 value columns 拆成两个独立 CTA。

compact kernel 标注 `T.annotate_min_blocks_per_sm(2)`。最终 CUDA 已确认：

```text
__launch_bounds__(128, 2)
residual_*_dv64x2_rs_io_qkva_reuse_so_kernel
```

## 4. Fragment 分配

这里的 “logical fragment” 是 TileLang 声明矩阵的逻辑容量，不等于最终 register
file 字节数。编译器会按线程 layout 分布、复用标量并在达到 255 registers/thread
后产生 spill。

### 4.1 修改前

| fragment | dtype/shape | logical size | 生命周期 |
| --- | --- | ---: | --- |
| `state_t` | FP32 `[64,128]` | 32 KiB | 整个 kernel |
| `state_operand` | BF16 `[64,128]` | 16 KiB | 每 chunk 的 G0/G1 |
| `z_operand` | BF16 `[64,64]` | 8 KiB | residual 后到 G4/G5 |
| `z_t` | FP32 `[64,64]` | 16 KiB | 当前 chunk |
| `out_t` | FP32 `[64,64]` | 16 KiB | 当前 chunk |
| `score` | FP32 `[64,64]` | 16 KiB | G3 到 score staging |
| **合计** | | **104 KiB** | |

### 4.2 修改后

fragment 声明完全不变，仍是 **104 KiB/block**。最终生成 CUDA 每线程声明：

```text
float       state_t[64]
bfloat16_t  state_operand[64]
float       z_t[32]
float       out_t[32]
bfloat16_t  z_operand[32]
float       score[32]
```

NCU 实测修改前后都是 255 registers/thread。最终 compact profile 的整个
`parallel_equal` launch 报告 110,080 条 local-memory spilling request 和 128 条
shared-memory spilling request。它们不是新增 fragment，但说明 D=64 已经顶到
寄存器上限；这也解释了 occupancy 翻倍后 long-scoreboard 仍然很高。后续若要继续
优化 D=64，缩短 fragment lifetime、让 `state_operand` 或 `score` 更早释放，比
增加新 fragment 更合理。

## 5. Shared memory 分配

Q/K/V/A/g/beta 是 pipeline stage-0 producer，lowering 为它们生成两份
ping-pong storage。gamma/inverse、score 和 output 是单 buffer。

### 5.1 修改前 fast D=64

| shared object | 单份 | stage 数 | block total |
| --- | ---: | ---: | ---: |
| Q BF16 `[64,128]` | 16 KiB | 2 | 32 KiB |
| K BF16 `[64,128]` | 16 KiB | 2 | 32 KiB |
| V BF16 `[64,64]` | 8 KiB | 2 | 16 KiB |
| A BF16 `[64,64]` | 8 KiB | 2 | 16 KiB |
| g FP32 `[64]` | 0.25 KiB | 2 | 0.5 KiB |
| beta FP32 `[64]` | 0.25 KiB | 2 | 0.5 KiB |
| score BF16 `[64,64]` | 8 KiB | 1 | 8 KiB |
| output BF16 `[64,64]` | 8 KiB | 1 | 8 KiB |
| gamma + inverse FP32 | 0.5 KiB | 1 | 0.5 KiB |
| gamma_last FP32 | 4 B | 1 | 4 B |
| **合计** | | | **113.504 KiB = 116,228 B** |

### 5.2 修改后 compact D=64

`score_shared` 和 `output_shared` 指向同一块 8 KiB physical storage，其余
对象不变：

| item | before | after | change |
| --- | ---: | ---: | ---: |
| Q/K/V/A ping-pong | 96 KiB | 96 KiB | 0 |
| g/beta ping-pong | 1 KiB | 1 KiB | 0 |
| score + output | 16 KiB | 8 KiB | -8 KiB |
| gamma/inverse/last | 0.504 KiB | 0.504 KiB | 0 |
| **dynamic shared** | **113.504 KiB** | **105.504 KiB** | **-8 KiB** |

NCU 使用十进制单位显示为 108.05 Kbyte dynamic + 1.02 Kbyte driver。生成 CUDA
中 score staging 和 output staging/TMA 都使用
`((bfloat16_t*)buf_dyn_shmem)[49664]`，确认它们是同一 physical address，而不只是
源码中名称相同。

这个别名只适用于 D=64：D=64 output 是 `[64,64]`，恰好与 score 一样大；
D=128 output 是 `[64,128]`，本轮明确不改。

## 6. 从 launch 到结束的完整执行

### 6.1 Prologue

1. CTA 根据 block id 计算 batch、value head、QK head 和 value slice。
2. 分配上述 shared storage 和 fragment。compact specialization 在这里把
   output handle 绑定到 score storage。
3. `state_t` 清零；若有 initial state，则从 HBM 读取 FP32
   `initial_state[128,64]` slice，并转置分布到 `state_t[64,128]` fragment。
   state 只在 kernel 边界读一次，不会每 chunk 回 HBM。
4. pipeline 为 chunk 0 发出 Q/K/V/A/g/beta 的 global-to-shared async copy 并
   `cp_async_commit()`。

### 6.2 Ping-pong prefetch 和内存层级

steady-state 处理 chunk `c` 时，生成代码先把 chunk `c+1` 发到另一 stage，再
处理当前 chunk：

```text
issue cp.async for next V/A/g/beta/K/Q
cp_async_commit
convert current state_t -> state_operand
cp_async_wait<1>
__syncthreads
consume current shared stage
```

因此 next chunk 的 global-memory latency 与当前 chunk 计算重叠。最后一个 peeled
iteration 没有 next chunk，使用 `cp_async_wait<0>`。

每个 D=64 CTA、每 chunk 的输入：

| tensor | bytes/CTA/chunk | copy |
| --- | ---: | --- |
| Q | 16 KiB | 16-byte cp.async transactions |
| K | 16 KiB | 16-byte cp.async transactions |
| V slice | 8 KiB | 16-byte cp.async transactions |
| A | 8 KiB | 16-byte cp.async transactions |
| g + beta | 0.5 KiB | 4-byte cp.async transactions |
| **合计** | **48.5 KiB** | |

一个 logical owner 有两个 D=64 CTA，所以合计输入为 97 KiB/chunk；Q/K/A/g/beta
被复制两次，V 的两个 slice 合计仍为 16 KiB。每 CTA 每 chunk 输出 8 KiB，
logical owner 合计 16 KiB。initial/final state 每 CTA 为 32 KiB，两个 part 合计
64 KiB。

`cp.async` 发出的是 global address 请求，并不是软件显式执行 “HBM -> L2” copy。
若数据不在 cache，请求由 HBM 经 L2/L1/shared 路径服务；若命中则由相应 cache
服务。CPU-to-HBM 更早发生，不属于这次 kernel launch。

### 6.3 当前 chunk 的数学和等待点

fragment 采用转置方向以使用 Hopper RS WGMMA：

```text
G0: Z0^T   = S^T @ K^T
G1: O0^T   = S^T @ Q^T
R^T        = beta * (V^T - gamma * Z0^T)
G2: Z1^T   = R^T @ A^T
G3: score  = Q @ K^T
Zhat^T     = Z1^T * gamma_last / gamma
G4: O^T   += Zhat^T @ score^T
G5: S^T   += Zhat^T @ K
```

逐步顺序：

| step | operation | result/location | wait/prefetch relation |
| ---: | --- | --- | --- |
| 1 | FP32 state 转 BF16 operand | register `state_operand` | next chunk copy 已发出 |
| 2 | G0 `S^T@K^T` | async WGMMA -> `z_t` | 不立即等待 |
| 3 | G1 `S^T@Q^T` | async WGMMA -> `out_t` | 不立即等待 |
| 4 | `gamma=exp(g)`、`1/gamma` | shared gamma arrays | 与 G0/G1 重叠 |
| 5 | thread 63 写 `gamma_last` | shared scalar | 避免跨线程发布竞态 |
| 6 | `state_t *= gamma_last` | FP32 fragment | G0/G1 仍可在途 |
| 7 | `warpgroup_wait<1>` | retire G0 | G1 保持在途 |
| 8 | residual `beta*(V-gamma*Z0)` | 覆盖 `z_t` | 第一次读取 G0 |
| 9 | residual 转 BF16 | `z_operand` fragment | 无 shared staging |
| 10 | G2 `R^T@A^T` | async -> `z_t` | |
| 11 | G3 `Q@K^T` | async -> FP32 `score` | |
| 12 | `warpgroup_wait<1>` | retire G1/G2 | G3 保持在途 |
| 13 | `Zhat=Z1*gamma_last/gamma` | `z_t`，再转 `z_operand` | 第一次读取 G2 |
| 14 | `warpgroup_wait<0>` | retire G3 | score 首次读取前唯一等待 |
| 15 | scale `out_t` | `scale*gamma*O0` | |
| 16 | causal/gated score | FP32 fragment | lower triangle |
| 17 | score FP32 -> BF16 shared | alias buffer | G4 operand |
| 18 | G4 `Zhat^T@score^T` | async 累加 `out_t` | |
| 19 | G5 `Zhat^T@K` | async 累加 `state_t` | |
| 20 | `warpgroup_wait<1>` | retire older G4 | G5 保持在途 |
| 21 | 用 output 覆盖 alias buffer | `out_t -> shared` | score 生命周期已结束 |
| 22 | output TMA store | shared -> HBM | `tma_store_wait<0>` |
| 23 | `warpgroup_wait<0>` | retire G5 | 下一 chunk 使用 state 前 |

步骤 20 可以使用 `wait<1>`，因为 G4/G5 分别 commit，G4 是较老 group，也是
score shared 的唯一读者；等待到只剩一个 group 后，剩下的 G5 只读取 K 和
`z_operand`，覆盖 score/output alias 不会影响它。最终生成 CUDA 在该位置确认为
`warpgroup_wait<1>`，而不是更保守的 `wait<0>`。

score 的缩放拆成：

```text
score *= scale * gamma[row] / gamma_last
Zhat  *= gamma_last / gamma[col]
```

两项在 G4 中相乘后恢复
`scale*gamma[row]/gamma[col]`，同时让 `gamma_last/gamma` 直接复用于 state
update。

### 6.4 gamma_last 发布修复

两 CTA residency 最初暴露了旧代码中被低 occupancy 掩盖的竞态：64 个线程并行写
gamma，而另一个线程可能在 token 63 写入完成前读取 `gamma_shared[63]`。最终改为
负责 token 63 的线程在同一个 parallel loop 中直接写 `gamma_last`：

```text
if token == 63:
    gamma_last = gamma[token]
```

生成 CUDA 确认为 `threadIdx.x == 63` 的单 writer。编译器仍会在 shared producer
和所有 state consumer 之间保留必要的 `__syncthreads()`；修复消除的是错误的
跨线程读取和额外手工同步需求，不是宣称 gamma 路径完全没有 CTA barrier。

### 6.5 Epilogue

每个 chunk 的 output 已由 TMA 写回 HBM。最后一次 G5 完成后，CTA 将 FP32
`state_t[64,128]` 转回 API 的 `final_state[128,64]` layout，coalesced 写回
HBM。kernel 返回后，host event/synchronize 才能观察完整 output 和 final state。

## 7. Occupancy 边界扫描

最终 forced-D64 文件为
`output/dv64_wait1_single_writer_boundary.log`，3 warmups、10 repetitions，
14 case 全部 PASS。表中的 old D64 是修改前 1-block kernel；final D64 在 owners
6/7 使用 fast specialization，在 owners>=8 使用 compact specialization。

| owners | chunks | D=128 ms | old D64 ms | final D64 ms | old/final | best |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 6 | 32 | 0.150160 | 0.110848 | 0.114544 | 0.968x | D64 fast |
| 7 | 32 | 0.151504 | 0.111440 | 0.114368 | 0.974x | D64 fast |
| 8 | 32 | 0.151408 | 0.195008 | 0.175104 | 1.114x | D128 |
| 9 | 32 | 0.151456 | 0.196288 | 0.177120 | 1.108x | D128 |
| 10 | 32 | 0.151248 | 0.197056 | 0.182320 | 1.081x | D128 |
| 12 | 32 | 0.153088 | 0.197760 | 0.186944 | 1.058x | D128 |
| 14 | 32 | 0.176352 | 0.199504 | 0.193952 | 1.029x | D128 |
| 6 | 128 | 0.510576 | 0.361024 | 0.364224 | 0.991x | D64 fast |
| 7 | 128 | 0.510624 | 0.361648 | 0.364976 | 0.991x | D64 fast |
| 8 | 128 | 0.511632 | 0.697312 | 0.612016 | 1.139x | D128 |
| 9 | 128 | 0.512032 | 0.698832 | 0.639136 | 1.093x | D128 |
| 10 | 128 | 0.512368 | 0.699344 | 0.677968 | 1.032x | D128 |
| 12 | 128 | 0.525264 | 0.701600 | 0.680032 | 1.032x | D128 |
| 14 | 128 | 0.616160 | 0.711536 | 0.678992 | 1.048x | D128 |

owners>=8 时，2 CTA residency 确实消除了旧 D64 的第二-wave 阶跃并带来
1.03--1.14x 收益，但没有补回相对 D=128 重复加载 Q/K/A/g/beta 的成本。因此 auto
边界不应从 7 放宽。

另外用 `output/dv64_wait1_compact_gva_state.log` 强制 compact D=64 跑了
`parallel_gva`、带 initial state 的 `wide_gva_state` 和 `deep_gva_state`，三项也
全部 PASS，覆盖了 boundary CSV 没有包含的 GVA head mapping 和 state-load prologue。

## 8. 正式 8 case

前 7 个 case：
`output/dv64_single_writer_final_8case.log`。集群 5 分钟上限在编译第 8 个 case
时终止任务，因此 deep 用相同的 10 warmups、100 repetitions 单独补跑：
`output/dv64_single_writer_final_deep_100.log`。8 个结果全部 PASS。

“上一轮”是 commit `476411c` 的正式结果。预计分数按课程给出的 60/100 turning
points 分段公式计算。

| case | 上一轮 ms | 本轮 ms | speedup | p=t100/t | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.143008 | 0.142176 | 1.0059x | 2.4339 | 120.00 |
| `chain_equal` | 0.381504 | 0.379104 | 1.0063x | 1.3131 | 106.26 |
| `parallel_equal` | 0.296416 | 0.294480 | 1.0066x | 1.7362 | 114.72 |
| `parallel_gva` | 0.293072 | 0.293328 | 0.9991x | 1.6758 | 113.52 |
| `long_low_gva` | 1.974816 | 1.977888 | 0.9984x | 0.9398 | 97.27 |
| `batch_split_gva` | 1.496288 | 1.527040 | 0.9799x | 1.0033 | 100.07 |
| `wide_gva_state` | 2.618208 | 2.534304 | 1.0331x | 0.9575 | 98.07 |
| `deep_gva_state` | 3.052224 | 3.015712 | 1.0121x | 0.9388 | 97.20 |

预计平均分为 **105.89**，上一轮为 105.67。正式 auto case 中 compact D64 不会被
选择，所以除 gamma_last 正确性修复外，这一轮主要建立的是高并行 D64 specialization；
表中 D=128 case 的小幅正负变化主要是跨作业测量波动，不能归因于 shared alias。

目前仍低于 100 分的是 long、wide、deep。long 使用 D=128（owners=8），wide/deep
也使用 D=128，因此按本轮约束“不优化 D=128”，它们不会因 D=64 residency 自动
达到 100。

## 9. 完整 NCU profile

最终 profile case 是强制 D=64 的 `parallel_equal`，grid=32，chunks=32，确保
实际测到 compact 2-block specialization。命令包含
`--set full --section PmSampling_WarpStates --import-source yes`，共 47 passes。

可直接用 Nsight Compute GUI 打开：

- `output/ncu_dv64_wait1_single_writer_parallel_full.ncu-rep`

可读导出：

- `output/ncu_dv64_wait1_single_writer_parallel_full_details.txt`
- `output/ncu_dv64_wait1_single_writer_parallel_full_raw.csv`
- `output/ncu_dv64_wait1_single_writer_parallel_full_source.csv`

`details.txt` 包含完整 Warp State Statistics 和所有 section，`raw.csv` 是全部
metric，`source.csv` 是 CUDA/SASS 与 stall sample 的关联。

关键数据：

| metric | final compact D64 |
| --- | ---: |
| NCU duration | 292.26 us |
| grid / waves per SM | 32 / 1.14 |
| registers/thread | 255 |
| dynamic + driver shared/block | 108.05 + 1.02 decimal kB |
| block limit: registers/shared | 2 / 2 |
| theoretical occupancy | 12.50% |
| achieved occupancy | 12.34% |
| active / eligible warps per scheduler | 1.88 / 0.31 |
| DRAM / SM throughput | 50.72% / 19.81% |
| warp cycles per issued instruction | 7.13 |
| stall long scoreboard | 2.31 CPI |
| stall wait | 1.04 CPI |
| stall barrier | 0.79 CPI |
| stall GMMA | 0.52 CPI |
| stall short scoreboard | 0.50 CPI |
| stall MIO throttle | 0.28 CPI |

Warp State not-issued samples：

| reason | samples |
| --- | ---: |
| long scoreboard | 1,303 |
| wait | 560 |
| barrier | 451 |
| warpgroup arrive | 337 |
| short scoreboard | 283 |
| MIO throttle | 158 |

occupancy 目标已经实现，但 eligible warps 仍只有 0.31/scheduler，long scoreboard
仍是第一 stall。NCU 同时指出 local memory 占 L1TEX 请求的显著部分，绝大多数
local load/store 来自 register spill。这说明单纯增加 resident CTA 只能部分隐藏
延迟；当前下一层瓶颈已经转向 255-register 上限、fragment spill 及其 L1/L2
依赖。shared alias 没有增加 fragment，因此没有恶化这一点，但也没有解决它。

## 10. 最终生成 CUDA

最终生成文件：

- `output/dv64_wait1_single_writer_final_generated.log`

已人工核对：

- kernel 是 `__launch_bounds__(128,2)`；
- Q/K/V/A/g/beta 有双 stage `cp.async`，steady-state 为
  `cp_async_wait<1>`，尾 chunk 为 `wait<0>`；
- 五个 recurrent/value GEMM 是 RS WGMMA，QK 是 SS WGMMA；
- gamma_last 由 `threadIdx.x==63` 写；
- G0/G1/G2/G3 的 wait 顺序为 `1,1,0`；
- G4 后使用 `warpgroup_wait<1>`，G5 与 output staging/store 重叠；
- score/output 使用同一 shared offset；
- output 使用 TMA store + `tma_store_wait<0>`；
- 下一 chunk 前有最终 `warpgroup_wait<0>` 保证 state update 完成。

## 11. 下一步

在继续保持 “暂不改 D=128” 的约束下，D=64 下一步不应再优先压 shared：
105.504 KiB 已让 shared block limit 达到 2，继续减少不会把 128-thread CTA 提升到
3 blocks/SM，因为寄存器已经把 block limit 卡在 2。

更有价值的 D=64 方向是降低 register/spill：

1. 缩短 `state_operand` 生命周期，研究 G0/G1 发出后是否能让其物理寄存器更早
   复用给 score 或 Z。
2. 研究 score 的分块 materialization，减少 16 KiB FP32 score fragment，但必须
   保持 G3/G4 Tensor Core 利用率，避免重新引入高 barrier。
3. 用 source-correlated NCU 定位 110,080 条 spill request 的具体数组/指令，再做
   定向 lifetime split；不能靠盲目限制 max registers，因为这通常只会增加 spill。

不过要让 long/wide/deep 全部超过 100，最终仍必须回到 D=128：这三个 case 的 auto
路径不经过 compact D=64。本轮验证了“提高 D64 occupancy 能改善强制高 owner
性能”，也同时证明它不足以替代 D=128 高 owner kernel。

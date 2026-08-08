# GDN prefill kernel walkthrough: RS WGMMA round

本文件描述 commit 前工作树中的完整实现，即“优化 1：把 recurrent operand 从
shared/shared WGMMA 改为 register/shared WGMMA”完成后的版本。内容以实际 TileLang
lowering 和 H800 MIG profile 为准，不是伪代码设计稿。

## 1. 本轮结论

原 D=128 路径把 FP32 recurrent state 和中间 Z 转成 BF16 后写到 shared memory，再由
SS WGMMA 读回。新路径把 state 转置保存在 fragment 中，BF16 operand 也保存在 fragment
中，用 Hopper RS WGMMA 直接读取寄存器端 A operand：

```text
old: FP32 fragment -> BF16 shared -> SS WGMMA
new: FP32 fragment -> BF16 fragment -> RS WGMMA
```

这样删除了每个 chunk 约 176 KiB/block 的 state/Z shared operand 流量及其同步边界。
D=128 的动态 shared 从 169.504 KiB 降到 137.504 KiB；代价是源码声明的 fragment
总量从 144 KiB/block 增到 192 KiB/block，NCU 实测寄存器由 206 增到 249
register/thread。占用率仍为 12.5%，所以加速来自更短的数据路径，而不是 occupancy
上升。

公开 8 case 全部 PASS，正式计时的简单平均预计分数从 94.76 提升到 **104.06**。

## 2. Host dispatch 和 kernel launch

入口仍是 `student/tilelang_fwd.py`。host 先完成输入预处理，构造 Q/K/V、gate、
beta、A、initial state 和输出 tensor；这些 tensor 在 kernel launch 前已经位于 GPU
HBM。CPU 到 HBM 的传输由 PyTorch tensor 创建/搬运阶段完成，不发生在本 kernel 内。

dispatch 顺序如下：

1. 若序列含不足 64 token 的 tail，tail 继续走原 SS 路径。
2. full-chunk 路径根据 `GDN_DV_SPLIT` 选择 D=128、64 或 32。
3. `GDN_RS=auto` 时，D>=64 选择
   `tilelang_residual_first_full_chunks_rs`；D=32 因 Hopper WGMMA 的 M 维固定为
   64，继续使用验证过的 SS kernel。
4. `GDN_RS=off` 可强制回到旧 SS 路径，用于 A/B。

RS launch grid 为：

```text
grid.x = B * Hv * dv_parts
owner = blockIdx.x / dv_parts
dv_part = blockIdx.x % dv_parts
batch = owner / Hv
value_head = owner % Hv
qk_head = value_head / (Hv / Hq)
```

线程和 value 分块：

| path | dv_tile | dv_parts | threads/CTA | warp groups | grid blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| RS D=128 | 128 | 1 | 256 | 2 | `B*Hv` |
| RS D=64 | 64 | 2 | 128 | 1 | `2*B*Hv` |
| SS D=32 | 32 | 4 | 256 | 2 | `4*B*Hv` |

D=128 的两个 warp group 沿转置 state 的 value-row 方向分工，每组负责 64 个 value
rows。每个 CTA 独占一个 `(batch,value_head,dv_part)` 的整条 chunk chain，state
在 chunk 间一直驻留寄存器，因此不同 chunk 不能拆给不同 CTA。

## 3. 数学方向为什么转置

逻辑公式仍是：

```text
Z0    = K @ S
O0    = Q @ S
R     = beta * (V - gamma * Z0)
Z1    = A @ R
Zhat  = gamma_last / gamma * Z1
score = causal(scale * gamma[row] / gamma[col] * Q @ K^T)
O     = scale * gamma * O0 + score @ Zhat
S     = gamma_last * S + K^T @ Zhat
```

RS WGMMA 要求 A operand 来自寄存器、B operand 来自 shared。为了让大的 recurrent
operand 位于 A 侧，新 kernel 保存 `S^T[D,128]`、`Z^T[D,64]` 和
`O^T[D,64]`：

```text
Z0^T   = S^T @ K^T
O0^T   = S^T @ Q^T
Z1^T   = R^T @ A^T
O^T   += Zhat^T @ score^T
S^T   += Zhat^T @ K
```

只有 `score = Q @ K^T` 保持 SS WGMMA，因为 Q、K 已在 shared，且 score 本身只有
64x64。

## 4. Fragment 分配

“fragment KiB”是源码逻辑容量，不等同于物理 register file 占用；后者还受 layout、
标量和编译器 lifetime reuse 影响。

### 4.1 修改前

| fragment | D=32 | D=64 | D=128 |
| --- | ---: | ---: | ---: |
| state FP32 `[128,D]` | 16 KiB | 32 KiB | 64 KiB |
| z FP32 `[64,D]` | 8 KiB | 16 KiB | 32 KiB |
| out FP32 `[64,D]` | 8 KiB | 16 KiB | 32 KiB |
| score FP32 `[64,64]` | 16 KiB | 16 KiB | 16 KiB |
| **合计** | **48 KiB** | **80 KiB** | **144 KiB** |

旧 D=128 NCU：206 registers/thread，无 local/shared spill。

### 4.2 修改后

| fragment | dtype/shape | D=64 | D=128 | 生命周期 |
| --- | --- | ---: | ---: | --- |
| `state_t` | FP32 `[D,128]` | 32 KiB | 64 KiB | 整个 kernel |
| `state_operand` | BF16 `[D,128]` | 16 KiB | 32 KiB | 每 chunk 前两次 RS |
| `z_operand` | BF16 `[D,64]` | 8 KiB | 16 KiB | residual 后到两个 update |
| `z_t` | FP32 `[D,64]` | 16 KiB | 32 KiB | 当前 chunk |
| `out_t` | FP32 `[D,64]` | 16 KiB | 32 KiB | 当前 chunk |
| `score` | FP32 `[64,64]` | 16 KiB | 16 KiB | QK 到 score shared |
| **源码合计** | | **104 KiB** | **192 KiB** | |

生成的 D=128 CUDA 对每个线程声明：

```text
state_t[64] float
state_operand[64] bfloat16
z_t[32] float
out_t[32] float
z_operand[32] bfloat16
score[16] float
```

NCU 实测为 249 registers/thread，零 local-memory spilling、零 shared-memory
spilling。D=32 没有改变，仍为 48 KiB/block；此前 NCU 实测 105
registers/thread。

## 5. Shared memory 分配

Q/K/V/A/g/beta 是 pipeline stage-0 producer，lowering 为它们生成 ping-pong 两份；
output、score、gamma、inv_gamma 是 single buffer。

### 5.1 修改后明细

| shared object | 单份 | stage 数 | D=64 | D=128 |
| --- | ---: | ---: | ---: | ---: |
| Q BF16 `[64,128]` | 16 KiB | 2 | 32 KiB | 32 KiB |
| K BF16 `[64,128]` | 16 KiB | 2 | 32 KiB | 32 KiB |
| V BF16 `[64,D]` | D/8 KiB | 2 | 16 KiB | 32 KiB |
| A BF16 `[64,64]` | 8 KiB | 2 | 16 KiB | 16 KiB |
| g FP32 `[64]` | 0.25 KiB | 2 | 0.5 KiB | 0.5 KiB |
| beta FP32 `[64]` | 0.25 KiB | 2 | 0.5 KiB | 0.5 KiB |
| output BF16 `[64,D]` | D/8 KiB | 1 | 8 KiB | 16 KiB |
| score BF16 `[64,64]` | 8 KiB | 1 | 8 KiB | 8 KiB |
| gamma + inverse | 0.5 KiB | 1 | 0.5 KiB | 0.5 KiB |
| gamma_last | 4 B | 1 | 4 B | 4 B |
| **合计** | | | **113.504 KiB** | **137.504 KiB** |

### 5.2 前后对照

| path | 修改前 shared | 修改后 shared | 变化 |
| --- | ---: | ---: | ---: |
| D=32 SS | 109.504 KiB | 109.504 KiB | 不变 |
| D=64 | 129.504 KiB | 113.504 KiB | -16 KiB |
| D=128 | 169.504 KiB | 137.504 KiB | -32 KiB |

减少量正好是删除的 `state_shared[128,D]`；原 `z_shared[64,D]` 改名并仅承担
output staging，容量不变。D=128 NCU 报告 140.82 decimal Kbyte/block，即
137.504 KiB。

## 6. 从 launch 到结束的逐步执行

### 6.1 Prologue

1. 每个 CTA 计算 owner、head 映射和 value slice。
2. `state_t` 清零；有 initial state 时，从 HBM 直接 coalesced load FP32 state 到
   转置 fragment。这个 load 只发生一次，不是每个 chunk 都访问 HBM state。
3. pipeline 为 chunk 0 发出 Q/K/V/A/g/beta 的 `cp.async.global.shared` 并
   `cp_async_commit`。生成 CUDA 已确认每次 BF16 主矩阵 copy 使用 16-byte
   transaction；g/beta 使用 4-byte transaction。

### 6.2 Steady-state prefetch

处理 chunk `c` 时，loop 顶部先把 chunk `c+1` 的六类输入发到另一个 shared
stage，然后才转换当前 state operand。计算真正读取当前 shared stage 之前执行：

```text
cp_async_wait<1>()
__syncthreads()
```

因此 current stage 已完成，而 next stage 可以继续在途。最后一个 peeled iteration
没有下一 chunk，使用 `cp_async_wait<0>()`。这是实际生成代码中的 prefetch，不是仅
在 TileLang IR 中声明但 lowering 后消失的重排。

每个 D=128 CTA、每个 chunk 的 HBM 输入为：

| input | bytes |
| --- | ---: |
| Q | 16 KiB |
| K | 16 KiB |
| V slice | 16 KiB |
| A | 8 KiB |
| g + beta | 0.5 KiB |
| **合计** | **56.5 KiB** |

输出每 chunk 写 16 KiB。state 仅在 kernel 边界读/写各 64 KiB。cache miss 时请求从
HBM 经 L2/L1 到 shared/register；cache hit 时可由 L2 服务。代码本身不会逐 chunk
执行“HBM 搬到 L2”的显式指令，`cp.async` 发的是 global-memory request，缓存层级由
硬件决定。

### 6.3 当前 chunk 的计算和等待点

| 顺序 | 操作 | operand/结果位置 | 同步语义 |
| ---: | --- | --- | --- |
| 1 | FP32 `state_t` 转 BF16 | register -> `state_operand` | 无 shared |
| 2 | G0 `S^T @ K^T` | RS，结果 `z_t` | async |
| 3 | G1 `S^T @ Q^T` | RS，结果 `out_t` | async |
| 4 | gamma、inverse、gamma_last | shared gate -> shared scalar arrays | 与 G0/G1 重叠 |
| 5 | `state_t *= gamma_last` | FP32 fragment | 与 G0/G1 重叠 |
| 6 | retire G0 | `warpgroup_wait<1>` | G1 保持在途 |
| 7 | residual | `beta*(V-gamma*z_t)` | 首次读取 G0 |
| 8 | residual 转 BF16 | `z_t -> z_operand`，均为 fragment | 无 shared |
| 9 | G2 `R^T @ A^T` | RS，覆盖 `z_t` | async |
| 10 | G3 `Q @ K^T` | SS，结果 `score` | async |
| 11 | retire G1/G2 | `warpgroup_wait<1>` | G3 保持在途 |
| 12 | `Zhat^T=Z1^T*gamma_last/gamma` | FP32 `z_t` | 首次读取 G2 |
| 13 | Zhat 转 BF16 | `z_t -> z_operand` | 无 shared |
| 14 | retire G3 | `warpgroup_wait<0>` | score 首次读取前 |
| 15 | scale `out_t` | `scale*gamma*out_t` | G1 已在步骤 11 完成 |
| 16 | causal/gated score | lower mask，`scale*gamma[row]/gamma_last` | FP32 fragment |
| 17 | score operand staging | score fragment -> BF16 `score_shared` | 保留的 SS 边界 |
| 18 | G4 `Zhat^T @ score^T` | RS，累加 `out_t` | async |
| 19 | G5 `Zhat^T @ K` | RS，累加 `state_t` | async |
| 20 | retire G4 | `warpgroup_wait<1>` | G5 与 output store 重叠 |
| 21 | output staging/store | `out_t -> output_shared -> HBM` | TMA store + wait |
| 22 | retire G5 | `warpgroup_wait<0>` | 下一 chunk 使用 state 前 |

步骤 16 使用 `gamma[row]/gamma_last`，步骤 12 已把
`gamma_last/gamma[col]` 乘进 Zhat；两者在 GEMM 中相消为原公式
`gamma[row]/gamma[col]`，避免再次逐 score element 读取 inverse[col]。

这里的五个 wait 在最终 CUDA 中按预期为：

```text
wait<1>  // G0 complete, G1 outstanding
wait<1>  // G1/G2 complete, G3 outstanding
wait<0>  // G3 complete
wait<1>  // G4 complete, G5 outstanding
wait<0>  // G5 complete before next chunk
```

生成源码同时确认五个 recurrent/value GEMM 是 `wgmma_rs`，QK 是
`wgmma_ss`。没有在结果真正被读取前增加额外的 WGMMA wait。

### 6.4 Epilogue

每个 chunk 的 output 已在步骤 21 写回 HBM。chain 结束后，CTA 把 FP32
`state_t[D,128]` 按最终 API 的 `[128,D]` layout coalesced 写入
`final_state`。所有线程结束后 kernel 返回；host 随后的 event/synchronize 才能观察
完整输出。

## 7. 本轮消除的 shared traffic

D=128 旧路径每 chunk 的主要 recurrent operand traffic：

| traffic | 估算 |
| --- | ---: |
| state FP32 fragment -> BF16 shared write | 32 KiB |
| 两次 SS WGMMA 读 state shared | 64 KiB |
| residual/Zhat 两次 BF16 z shared write | 32 KiB |
| A、score、state-update 三次读 z shared | 48 KiB |
| **合计** | **176 KiB/block/chunk** |

新 RS 路径把这些全部改为 register operand。它仍保留 Q/K/A/score 的 shared Tensor
Core operand，以及 output staging。新代价是 register-register layout conversion，
profile 中 MIO throttle 从旧 wide 的 0.23 CPI 增到 1.16 CPI；但 barrier 和
long-scoreboard 的下降远大于这个代价。

## 8. 正确性和正式 8 case

正式任务：`output/rs_final_8case_60486.log`，10 warmups、100 repetitions，8 个
case 均为 PASS。

| case | 上一版 ms | 本轮 ms | speedup | `p=t100/t` | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.144704 | 0.146080 | 0.9906x | 2.3689 | 120.00 |
| `chain_equal` | 0.614640 | 0.617040 | 0.9961x | 0.8067 | 91.17 |
| `parallel_equal` | 0.408720 | 0.297456 | 1.3741x | 1.7188 | 114.38 |
| `parallel_gva` | 0.387504 | 0.289280 | 1.3395x | 1.6992 | 113.98 |
| `long_low_gva` | 2.958256 | 1.978032 | 1.4956x | 0.9397 | 97.27 |
| `batch_split_gva` | 2.214960 | 1.494240 | 1.4823x | 1.0254 | 100.51 |
| `wide_gva_state` | 3.707360 | 2.535616 | 1.4621x | 0.9570 | 98.05 |
| `deep_gva_state` | 4.491200 | 3.021616 | 1.4864x | 0.9370 | 97.12 |

预计分数使用课程文档公开的 60/100 turning points 分段线性计算，100 分以上按
`min(120,100+20*(p-1))`。公开 8 case 简单平均为 **104.06**。

short 和 chain 仍走未修改的 tail/D32 SS path，其 1% 内波动属于测量噪声；其余
D=128 case 得到 1.34x--1.50x 加速。

## 9. 完整 NCU profile

可直接在 Nsight Compute GUI 打开的报告：

- `output/ncu_rs_long_full.ncu-rep`
- `output/ncu_rs_wide_full.ncu-rep`

可读导出：

- `output/ncu_rs_long_full_details.txt`
- `output/ncu_rs_long_full_source.csv`
- `output/ncu_rs_wide_full_details.txt`
- `output/ncu_rs_wide_full_source.csv`

两份报告均由 `--set full --section PmSampling_WarpStates` 生成，共 47 passes；
`details.txt` 使用 `--print-details all`，包含完整 Warp State Statistics，
`source.csv` 包含 SASS/source 关联的每类 stall sample。

### 9.1 wide 前后对比

| metric | 修改前 SS | 修改后 RS | 变化 |
| --- | ---: | ---: | ---: |
| NCU duration | 3.78 ms | 2.54 ms | 1.488x |
| registers/thread | 206 | 249 | +43 |
| dynamic shared/block | 173.58 kB | 140.82 kB | -32 KiB |
| theoretical/achieved occupancy | 12.5% / 12.5% | 12.5% / 12.5% | 不变 |
| DRAM throughput | 44.27% | 65.68% | +21.41 pp |
| compute throughput | 22.53% | 33.05% | +10.52 pp |
| stall barrier | 3.05 CPI | 1.02 CPI | -66.6% |
| stall long scoreboard | 1.62 CPI | 0.55 CPI | -66.0% |
| stall short scoreboard | 0.94 CPI | 0.50 CPI | -46.8% |
| stall MIO throttle | 0.23 CPI | 1.16 CPI | +0.93 CPI |

RS 版本的 not-issued samples 包括
`mio_throttle=9213, barrier=7798, wait=4453, long_scoreboard=3956,
short_scoreboard=3149, warpgroup_arrive=1306`。当前最大的新瓶颈是 fragment
layout/convert 带来的 MIO pressure，而不是此前 dominant 的 barrier/long scoreboard。

### 9.2 long 低并行 case

| metric | RS profile |
| --- | ---: |
| NCU duration | 1.98 ms |
| grid / waves per SM | 8 CTAs / 0.57 |
| registers/thread | 249 |
| dynamic shared/block | 140.82 kB |
| theoretical/achieved occupancy | 12.5% / 12.5% |
| DRAM / compute throughput | 40.08% / 21.46% |
| barrier / long scoreboard | 0.98 / 0.52 CPI |

NCU 明确指出 grid 只有 8 blocks，小于 14 SM；这也是下一轮重新评估 dv 拆分的直接
依据。RS 已缩短单 CTA 的关键路径，所以旧版基于 SS kernel 测出的拆分阈值不能直接
沿用。

## 10. 生成代码与剩余限制

最终 D=128 CUDA：

- `output/rs_generated_dv128_60595.log`

它包含 kernel launch bounds、每线程数组、`cp_async` 双缓冲、RS/SS WGMMA、
五个 wait 和 TMA output store，可用于逐条核对上述流程。

当前限制：

1. D=32 不能直接使用 M=32 RS WGMMA，仍保留 SS path。
2. D=64 虽能使用一个 warp group 的 RS kernel，但是否值得把 D=128 拆成两个 CTA
   取决于 14-SM MIG 上的可驻留 blocks、owner 数和 chain 长度，不能只按公开 case
   名称 dispatch。
3. 249 registers/thread 已接近 255 上限；继续增加 fragment 很可能 spill。
4. shared 降低没有提升 D=128 occupancy，因为 register 和 shared 均把每 SM 限制为
   一个 CTA。

下一轮将独立扫描 D=128 RS、D=64 RS、D=32 SS，在额外 synthetic
`B*Hv x chain_length` 网格上总结规则，再提交另一 commit。

# GDN prefill kernel walkthrough: RS WGMMA + measured dv dispatch

本文件描述当前 commit 前工作树中的完整实现：recurrent operand 使用 RS WGMMA，
并按实测的 H800 MIG CTA wave 边界选择 D=64 或 D=128。内容以实际 TileLang
lowering、生成 CUDA、公开及 synthetic sweep 和完整 NCU profile 为准。

## 1. 本轮结论

RS 基础路径把 state 转置保存在 fragment 中，BF16 operand 也保存在 fragment 中，用
Hopper RS WGMMA 直接读取寄存器端 A operand：

```text
old: FP32 fragment -> BF16 shared -> SS WGMMA
new: FP32 fragment -> BF16 fragment -> RS WGMMA
```

这样删除了每个 chunk 约 176 KiB/block 的 state/Z shared operand 流量及其同步边界。
D=128 的动态 shared 从 169.504 KiB 降到 137.504 KiB；代价是源码声明的 fragment
总量从 144 KiB/block 增到 192 KiB/block，NCU 实测寄存器由 206 增到 249
register/thread。占用率仍为 12.5%，所以加速来自更短的数据路径，而不是 occupancy
上升。

本轮重新扫描 D=128、D=64、D=32 后，旧 auto 的 D=32 条件被替换为：

```text
owners = B * Hv
full_chunks_only and 2 * owners <= 14 SM  -> D=64 RS, two value parts
otherwise                                 -> D=128 RS, one value part
```

测试覆盖 1--512 chunks；对 full-chunk RS 路径，chain length 没有改变交叉点，真正
的断点是 owners=7/8。D=32 在所有扫描点都没有胜过 D=64。公开 8 case 全部 PASS，
`chain_equal` 从 0.617040 ms 降到 0.381504 ms；预计平均分从 104.06 提升到
**105.67**。

## 2. Host dispatch 和 kernel launch

入口仍是 `student/tilelang_fwd.py`。host 先完成输入预处理，构造 Q/K/V、gate、
beta、A、initial state 和输出 tensor；这些 tensor 在 kernel launch 前已经位于 GPU
HBM。CPU 到 HBM 的传输由 PyTorch tensor 创建/搬运阶段完成，不发生在本 kernel 内。

最终 dispatch 顺序如下：

1. 计算 `owners=B*Hv`、`chunks=ceil(T/64)` 和 `full_chunks_only=(T%64==0)`。
2. `GDN_DV_SPLIT=auto` 且 `full_chunks_only && 2*owners<=14` 时选择 D=64；
   其余选择 D=128。非 64 对齐输入不会拆分，因为它们不能走 full-chunk RS kernel。
3. `GDN_DV_SPLIT=off/64/32` 可强制三档，用于 A/B；D=32 不进入 auto。
4. `GDN_RS=auto` 时，D>=64 选择
   `tilelang_residual_first_full_chunks_rs`；D=32 因 Hopper WGMMA 的 M 维固定为
   64，继续使用验证过的 SS kernel。
5. `GDN_RS=off` 可强制回到旧 SS 路径，用于 A/B。

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
rows。D=64 的两个 CTA 各持有一个 64-column state slice，每个 CTA 只有一个 warp
group。每个 CTA 独占 `(batch,value_head,dv_part)` 的整条 chunk chain，state 在
chunk 间一直驻留寄存器，因此不同 chunk 不能拆给不同 CTA。

### 2.1 当前允许的并行度

实际设备查询见 `output/dv_hardware_props_60791.log`：

| resource | H800 PCIe MIG 1g.10gb |
| --- | ---: |
| SM | 14 |
| max resident warps/SM | 64 |
| registers/SM | 65,536 x 32-bit |
| max registers/thread | 255 |
| shared memory/SM | 233,472 B |
| opt-in shared/block | 232,448 B |
| max threads/SM | 2,048 |

架构上限与 [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
一致。实际 kernel 的资源限制比架构 warp 上限更早生效：

| path | regs/thread | dynamic shared | resident CTA/SM | active warps/SM |
| --- | ---: | ---: | ---: | ---: |
| D=128 RS | 249 | 137.504 KiB | 1 | 8 |
| D=64 RS | 255 | 113.504 KiB | 1 | 4 |
| D=32 SS | 105 | 109.504 KiB | 2 | 16 |

D=64 虽然 register 理论上允许两个 128-thread CTA，但每块还有约 1 KiB driver
shared；两块总 shared 超过 228 KiB，因此 NCU 给出的 shared block limit 是 1。

当 owners<=7 时，D=64 的 `2*owners` 个 CTA 能在 14 SM 上一次铺开；总 value
warp-group 数与 D=128 相同，但使用两倍 SM。owners=8 时出现第 15、16 个 CTA，
必须执行第二 wave，同时 Q/K/A/score 已被复制两次，所以性能发生阶跃式反转。

### 2.2 三档扫描结果

公开 case 强制三档，5 warmups、30 repetitions：

| case | owners | chunks | D=128 RS ms | D=64 RS ms | D=32 SS ms | best |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| short_tail_state | 8 | tail | 0.142000 | 0.225408 | 0.231632 | D=128 |
| chain_equal | 4 | 128 | 0.522624 | 0.376064 | 0.617520 | D=64 |
| parallel_equal | 16 | 32 | 0.297888 | 0.323904 | 0.460800 | D=128 |
| parallel_gva | 16 | 32 | 0.288368 | 0.313776 | 0.452608 | D=128 |
| long_low_gva | 8 | 512 | 1.991760 | 2.974432 | 4.412448 | D=128 |
| batch_split_gva | 32 | 128 | 1.502576 | 1.891648 | 3.016416 | D=128 |
| wide_gva_state | 64 | 128 | 2.552000 | 4.218192 | 5.813280 | D=128 |
| deep_gva_state | 32 | 256 | 3.058672 | 4.110816 | 5.976896 | D=128 |

为了避免拟合公开 shape，另用固定 Hq=Hv=1、只改变 B 的 synthetic case 分离
owners 和 chain。7/8 边界：

| owners | D=128, 32 chunks | D=64, 32 chunks | D=128, 128 chunks | D=64, 128 chunks |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.150160 | 0.110848 | 0.510576 | 0.361024 |
| 7 | 0.151504 | 0.111440 | 0.510624 | 0.361648 |
| 8 | 0.151408 | 0.195008 | 0.511632 | 0.697312 |
| 9 | 0.151456 | 0.196288 | 0.512032 | 0.698832 |
| 14 | 0.176352 | 0.199504 | 0.616160 | 0.711536 |

chain-length scan（owners=1；owners=4/7 结论相同）：

| chunks | D=128 ms | D=64 ms | D64 speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.033328 | 0.032896 | 1.013x |
| 2 | 0.033504 | 0.032576 | 1.028x |
| 4 | 0.042080 | 0.035792 | 1.176x |
| 8 | 0.059072 | 0.046416 | 1.273x |
| 16 | 0.092016 | 0.067328 | 1.367x |
| 32 | 0.161568 | 0.111264 | 1.452x |
| 128 | 0.559424 | 0.363072 | 1.541x |
| 512 | 2.110688 | 1.376288 | 1.534x |

因此 chain 越长收益越稳定，但不存在需要编码的最小 chunks 阈值。auto synthetic
验证见 `output/dv_auto_synth_61023.log`：owners=1/2/4/7 与强制 D=64 一致，
owners=14/28 与强制 D=128 一致，16 case 全部 PASS。

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

### 4.3 本轮 auto 活跃路径的前后变化

本轮没有增加 fragment 声明，也没有改变任一 kernel 的静态资源；改变的是低 owner
shape 实际 launch 哪个 kernel。`chain_equal` 是公开 case 中唯一发生切换的路径：

| resource | 上一轮 auto: D=32 SS | 本轮 auto: D=64 RS |
| --- | ---: | ---: |
| logical fragment/block | 48 KiB | 104 KiB |
| NCU registers/thread | 105 | 255 |
| threads/block | 256 | 128 |
| logical registers/block | 26,880 | 32,640 |
| dynamic shared/block | 109.504 KiB | 113.504 KiB |
| grid for owners=4 | 16 CTA | 8 CTA |
| resident CTA/SM | 2 | 1 |
| theoretical occupancy | 25% | 6.25% |

D=64 生成 CUDA 每线程正好是 `state_t[64]`、`state_operand[64]`、`z_t[32]`、
`out_t[32]`、`z_operand[32]` 和 `score[32]`。虽然局部 occupancy 更低，D=64
只复制两次 QK/score 而不是 D=32 的四次，并且 8 个 CTA 一次铺到 8 个 SM；因此
端到端更快。D=128 的 192 KiB logical fragment、249 registers/thread 和
137.504 KiB shared 均未改变。

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

D=64 每个 CTA 只读 8 KiB V slice、写 8 KiB output，单 CTA 输入为 48.5 KiB；
但一个 logical owner 有两个 CTA，所以聚合输入是 97 KiB/chunk。相对 D=128 多出的
40.5 KiB 是第二份 Q/K/A/g/beta，这就是拆分的复制成本。两个 part 的 state 边界
流量各 32 KiB，聚合后仍与 D=128 相同。

D=128 输出每 chunk 写 16 KiB。state 仅在 kernel 边界读/写各 64 KiB。cache miss 时请求从
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

正式任务：`output/dv_auto_final_8case_60996.log`，10 warmups、100 repetitions，
8 case 均为 PASS。

| case | 上一轮 ms | 本轮 ms | speedup | `p=t100/t` | 预计分数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `short_tail_state` | 0.146080 | 0.143008 | 1.0215x | 2.4198 | 120.00 |
| `chain_equal` | 0.617040 | 0.381504 | 1.6174x | 1.3048 | 106.10 |
| `parallel_equal` | 0.297456 | 0.296416 | 1.0035x | 1.7248 | 114.50 |
| `parallel_gva` | 0.289280 | 0.293072 | 0.9871x | 1.6772 | 113.54 |
| `long_low_gva` | 1.978032 | 1.974816 | 1.0016x | 0.9413 | 97.34 |
| `batch_split_gva` | 1.494240 | 1.496288 | 0.9986x | 1.0240 | 100.48 |
| `wide_gva_state` | 2.535616 | 2.618208 | 0.9685x | 0.9268 | 96.68 |
| `deep_gva_state` | 3.021616 | 3.052224 | 0.9900x | 0.9276 | 96.69 |

预计分数使用课程文档公开的 60/100 turning points 分段线性计算，100 分以上按
`min(120,100+20*(p-1))`。公开 8 case 简单平均为 **105.67**，比上一轮
104.06 增加 1.61 分。

只有 `chain_equal` 按规则从 D=32 切到 D=64；其余 case 的 dispatch 与上一轮完全
相同，因此其小幅正负变化是跨任务时钟/测量波动，不应被解释为规则导致的回退。

## 9. 完整 NCU profile

本轮新生成、可直接由 Nsight Compute GUI 打开的报告：

- `output/ncu_dv_auto_chain_full.ncu-rep`
- `output/ncu_dv_auto_wide_full.ncu-rep`

完整可读导出：

- `output/ncu_dv_auto_chain_full_details.txt`
- `output/ncu_dv_auto_chain_full_source.csv`
- `output/ncu_dv_auto_wide_full_details.txt`
- `output/ncu_dv_auto_wide_full_source.csv`

两份报告均由 `--set full --section PmSampling_WarpStates` 生成，共 47 passes；
`details.txt` 使用 `--print-details all`，包含完整 Warp State Statistics，
`source.csv` 包含 SASS/source 关联的全部 stall sample。

### 9.1 chain：上一轮 D=32 对本轮 D=64

| metric | D=32 SS | D=64 RS | 变化 |
| --- | ---: | ---: | ---: |
| NCU duration | 579.87 us | 346.98 us | 1.671x |
| grid / waves per SM | 16 / 0.57 | 8 / 0.57 | CTA 数减半 |
| registers/thread | 105 | 255 | +150 |
| dynamic shared/block | 112.14 kB | 116.24 kB | +4 KiB |
| theoretical occupancy | 25.0% | 6.25% | -18.75 pp |
| achieved occupancy | 14.33% | 6.25% | -8.08 pp |
| DRAM throughput | 25.17% | 42.02% | +16.85 pp |
| compute throughput | 26.00% | 16.43% | -9.57 pp |
| stall barrier | 3.13 CPI | 0.56 CPI | -82.1% |
| stall long scoreboard | 1.90 CPI | 0.19 CPI | -90.0% |
| stall short scoreboard | 0.48 CPI | 0.52 CPI | +0.04 CPI |
| stall wait | 0.77 CPI | 0.99 CPI | +0.22 CPI |
| stall MIO throttle | 0.02 CPI | 0.05 CPI | +0.03 CPI |

D=64 的 occupancy 数字更低并不矛盾：occupancy 是“每个已激活 SM 内的 active
warps”，不表示 grid 使用了多少个 SM。D=64 使用 8 SM 而 D=128 只会使用 4 SM；
相比旧 D=32，它又把重复的 QK/score 从四份减到两份。因此执行的 tensor work 更少，
compute-throughput 百分比下降但 wall time 缩短。

D=64 not-issued samples 为
`wait=1178, barrier=705, short_scoreboard=673, long_scoreboard=246,
mio_throttle=63, warpgroup_arrive=61`。long-scoreboard 已不是主要瓶颈，当前主要
空泡来自显式 WGMMA wait、barrier 和仅一个 warp group/SM 带来的低 eligible warps。

### 9.2 wide：高 owner D=128 分支

| metric | current D=128 |
| --- | ---: |
| NCU duration | 2.58 ms |
| grid / waves per SM | 64 / 4.57 |
| registers/thread | 249 |
| dynamic shared/block | 140.82 kB |
| theoretical/achieved occupancy | 12.5% / 12.5% |
| DRAM / compute throughput | 64.68% / 33.04% |
| MIO / barrier / wait | 1.16 / 1.02 / 0.74 CPI |
| long / short scoreboard | 0.54 / 0.50 CPI |

该结果与上一轮 D=128 profile 的 2.54 ms、0.55 long-scoreboard CPI 一致，确认新
dispatch 没有改变高 owner kernel。

## 10. 生成代码、扫描文件和限制

最终生成 CUDA：

- D=64：`output/dv_auto_generated_dv64_61045.log`
- D=128：`output/rs_generated_dv128_60595.log`

两者都包含 launch bounds、每线程 fragment 数组、`cp_async` 双缓冲、RS/SS
WGMMA、五个精确 wait 和 TMA output store。D=64 为
`__launch_bounds__(128,1)`，D=128 为 `__launch_bounds__(256,1)`。

原始 sweep：

- public 三档：`output/dv_rs_public_off_60808.log`、
  `output/dv_rs_public_64_60838.log`、`output/dv_rs_public_32_60861.log`
- owners x long chain：`output/dv_rs_synth_off_60890.log`、
  `output/dv_rs_synth_64_60905.log`、`output/dv_rs_synth_32_60918.log`
- 1--16 chunks：`output/dv_rs_short_off_60941.log`、
  `output/dv_rs_short_64_60948.log`
- 7/8 boundary：`output/dv_rs_boundary_off_60969.log`、
  `output/dv_rs_boundary_64_60982.log`

当前限制和下一步：

1. D=64 shared 加 1 KiB driver allocation 后只比“两 CTA/SM”上限多约 516 B
   dynamic shared。若复用 gamma/inverse/gate buffer 省下至少 516 B，就可能让
   D=64 达到两个 CTA/SM；届时必须重新扫描规则，owners=8 以上可能受益。
2. D=64 已用满 255 registers/thread，继续增加 fragment 会 spill。
3. D=32 因缺少自然的 M=32 RS WGMMA，在当前实现中始终输给 D=64；保留它仅用于
   强制实验，不进入 auto。
4. 当前公开 long/wide/deep 仍约 96.7--97.3 分。要达到每 case 100+，下一步比继续
   调 dv 更有前景的是压缩 D=64 shared 以改变 residency，以及降低 D=128 的
   fragment conversion/MIO pressure。

# Gate-CP compile-budget final walkthrough

## 1. 本轮目标与结论

上一版 gate-decay context parallelism（下文简称 CP）只按 GPU 执行时间选择
`p=1/2/4/8`。这在单个 case 的 microbenchmark 中有效，但 TileLang 会针对每组静态
shape 和 `p` 即时编译 kernel。一次 OJ 进程连续运行 public 和 hidden case 时，CP case
需要 warmup、state preparation、main 三个 specialization；普通 case 只需要一个 main
specialization。上一版公开 8 case 共触发 16 次编译，给 285 秒总墙钟限制留下的空间
太小。

本轮不修改任何 GPU kernel body，只收紧 host dispatch：

```text
GDN_GATE_CP_MIN_CHUNKS: 64 -> 128

if base_blocks >= resident_slots:
    p = 1
```

含义是：只有原始 main grid 连一个驻留波次都填不满，并且每条 recurrent chain 至少有
128 个完整 chunk 时，自动策略才允许为它额外编译并启动 CP 三段路径。数字模式
`GDN_GATE_CP=2/4/8` 仍可强制实验，不受这个自动 eligibility gate 限制。

正式 8-case 结果全部 PASS。实际自动选择变为：

```text
short=off, chain=p4, parallel_equal=off, parallel_gva=off,
long=p8, batch=off, wide=off, deep=off
```

公开集合的 JIT specialization 数从 16 降到 12；根据 TileLang 日志时间戳，累计 compile
时间约从 126 秒降到 107 秒，首个 compile 开始到最后一个 compile 完成的跨度从约
147 秒降到 124 秒。这个改变针对 OJ 的进程级 285 秒限制。它不能单独证明某个未知
hidden shape 一定完成，但确实移除了 public batch/deep 的四个低收益 specialization。

## 2. 符号、shape 和基础映射

```text
B           batch size
T           每个 batch 的 token 数
Hq          Q/K head 数
Hv          V/state head 数
C           floor(T / 64)，完整 chunk 数
CHUNK       64
Dk          128
Dv          128
dv_tile     一个 CTA 覆盖的 value width，64 或 128
dv_parts    Dv / dv_tile，2 或 1
owners      B * Hv
base_blocks owners * dv_parts
p           context-parallel parts，1/2/4/8
```

每个普通 main CTA 负责一个 `(batch, value_head, dv_part)`，沿时间顺序更新自己的 state
tile。CP 再沿时间把同一个 owner 复制为 `p` 个 CTA：

```text
main_grid    = base_blocks * p
local_chunks = C / p
```

当前设备是 14-SM H800 PCIe MIG 1g.10gb。D64 compact main 可驻留 2 CTA/SM，所以
`resident_slots=28`；D128 main 因 shared memory 和寄存器只能驻留 1 CTA/SM，所以
`resident_slots=14`。

## 3. Python dispatch 到 kernel launch

进入 `gdn_prefill_forward` 时，Q/K/V、chunk-local cumulative gate、beta、A 和可选 initial
state 已经是 CUDA tensor。benchmark 中 gate cumsum 和 KKT solve 在学生 core 计时区间
之前完成。CPU host memory 到 HBM 的 tensor 构造或预处理搬运不发生在下面的 main
chunk loop 内。

host dispatch 按如下顺序执行：

1. 读取动态 `B,T,Hq,Hv`，检查 dtype、shape 和 `T % 64`。
2. 在 GPU 上分配 output 和 final_state。
3. 按 `owners=B*Hv` 选择 D64/D128。D64 只有在 `owners*2 <= 14` 时自动启用。
4. 检查 RS full-chunk fast path 和 memory-IO/prefetch profile。
5. 调用 `_select_gate_cp_parts` 决定 `p`。
6. 由静态参数取得或 JIT 编译 TileLang kernel factory。
7. `p=1` 时启动一个 main kernel；`p>1` 时分配 CP workspace 并依次启动三个 kernel。

TileLang specialization key 实际包含 Hq/Hv、是否有 initial state、dv tile/parts、RS、
prefetch profile、tail 和 `p` 等静态信息。因此两个 case 即使都选择 `p=1`，只要 head
shape 或 initial-state flag 不同，仍可能各自编译一个 main kernel。CP 又引入两个额外
kernel factory，所以减少不必要的 CP case 比只减少 main grid 更能降低累计 JIT 时间。

## 4. 当前自动 CP eligibility

共同拒绝条件是：

```text
GDN_GATE_CP=off
存在 tail
没有进入 RS fast path
C < 128
base_blocks >= resident_slots
```

最后一条是本轮关键。若 base grid 已经至少有一个完整驻留波次，切时间虽然可能让长
任务多几个执行波次，但不再解决“GPU 没有足够 CTA”这个一阶问题；它却一定增加
warmup/prepare launch、workspace 和 JIT specialization。自动 OJ policy 因而关闭它。

如果仍满足 eligibility，D64 保留局部链长平方根模型：

```text
estimated_local_chunks = 3 * sqrt(base_blocks * C / resident_slots)
target = max(4, next_power_of_two(estimated_local_chunks))
```

从 `p=1` 开始翻倍，直到 `C/p <= target`，最多到 8。D128 保留上一版的离散波次成本
模型，在 `p=1/2/4/8` 间比较 main waves、prepare waves、gate scan 和 GVA Q/K reuse。
选择以后如果 `p` 不能整除 `C`，按 2 的幂回退。代码没有按 public case 名称或精确
shape 写特判。

公开 case 的决策展开如下：

| case | C | base blocks | resident slots | result |
|---|---:|---:|---:|---:|
| short_tail_state | 16 full + tail | 8 | 14 | tail，off |
| chain_equal | 128 | 8 | 28 | underfilled，p4 |
| parallel_equal | 32 | 16 | 14 | full grid，off |
| parallel_gva | 32 | 16 | 14 | full grid，off |
| long_low_gva | 512 | 8 | 14 | underfilled，p8 |
| batch_split_gva | 128 | 32 | 14 | full grid，off |
| wide_gva_state | 128 | 64 | 14 | full grid，off |
| deep_gva_state | 256 | 32 | 14 | full grid，off |

## 5. 为什么 gate decay 可以准备分段 state

对一个 chunk，令 `S` 是起点 state，`gamma_i=exp(g_i)`，当前 residual-first kernel 的
等价数学关系是：

```text
P          = K @ S
O_state    = Q @ S
R          = beta * (V - gamma * P)
Z          = A @ R
Zhat_i     = gamma_last / gamma_i * Z_i
Score      = Lower(Q @ K^T)
O          = scale * gamma_i * O_state
             + scale * gamma_i / gamma_last * Score @ Zhat
S_next     = gamma_last * S + K^T @ Zhat
```

实际 kernel 把 state/Z 转置成以 Dv 为行的 fragment，以匹配 RS WGMMA。`g` 是 chunk 内
log-gate prefix sum；chunk 最后一项就是整块 decay 的 log 值。跨 chunk decay 在 log
space 中求和。

对时间片边界 `b`，warmup kernel 从边界前一个 chunk 向前累计 chunk-end gate。当累计
值低于默认阈值 -10 时，更早 state 对当前边界的乘法权重低于 `exp(-10)`。prepare
kernel 只从该 suffix 起点重放 state recurrence。如果一直扫到 chunk 0 都没越过阈值，
它从真实 initial state（或零 state）重放 `[0,b)`；这是精确但较慢的 fallback。

## 6. p=1 的单 launch 路径

batch/deep 在本版不再启动 warmup 和 prepare：

```text
host alloc output/final_state
    |
    `-- launch main, grid=base_blocks
            load real initial_state or zero state
            process all C chunks in order
            store output slices and final_state
```

每个 owner 只有一个 CTA，所以 state recurrence 完全保留在该 CTA 的 FP32 state
fragment 中。没有跨 CTA state workspace，也没有 launch 间的 HBM round trip。

## 7. p>1 的三 launch 时间线

```text
host alloc output/final_state/warmup_counts/cp_states in HBM
    |
    |-- launch 1: get_gate_cp_warmup_sp{p}
    |       read chunk-end g through L2/HBM
    |       write warmup_counts
    |
    |-- launch 2: prepare_gate_cp_dv{tile}x{parts}_sp{p}
    |       read warmup_counts and suffix K/V/A/g/beta
    |       write cp_states
    |
    `-- launch 3: residual_*_cp{p}
            part 0 uses real initial state
            parts 1..p-1 load cp_states
            all parts write disjoint output slices
            last part writes final_state
```

三个 kernel 在同一 CUDA stream 顺序提交。launch 2 在 launch 1 完成后才观察
warmup_counts，launch 3 在 launch 2 完成后才观察 cp_states；CUDA stream ordering
提供全局内存可见性，不需要在单个 kernel 中做跨 CTA grid barrier。main 内各 part
写不同 token 区间，也不需要跨 CTA 同步。

## 8. Launch 1: gate warmup scan

```text
grid    = B * (p - 1)
threads = ceil(Hv / 32) * 32
```

一个 CTA 对应 `(batch, nonzero_seq_part)`，线程覆盖 Hv heads。每个 head 倒序读取边界前
各 chunk 的 endpoint gate，维护累计 log gate，第一次越过阈值时记录 suffix 长度。

逻辑 fragment：

| item | shape | dtype | bytes |
|---|---|---|---:|
| gate_value | `[Hv]` | FP32 | `4*Hv` |
| gate_sum | `[Hv]` | FP32 | `4*Hv` |
| warmup_count | `[Hv]` | INT32 | `4*Hv` |
| total | | | `12*Hv` |

源码没有用户 shared memory，也没有 ping-pong。每一步直接 global load 一个 endpoint；
连续倒序访问先查 L2，miss 才由 HBM 填充。最后 global store warmup_counts。NCU 所示
1 KiB/block 是 driver allocation，不是 TileLang 声明的 shared array。

## 9. Launch 2: CP state preparation

```text
grid    = base_blocks * (p - 1)
threads = 128 for D64, 256 for D128
```

CTA 映射到 `(seq_part,batch,Hv,dv_part)`，读 warmup_count，决定 suffix 起点，然后只重放
state recurrence。它不读 Q，也不计算 Q@S、Q@K 或 output，因此每 chunk 只有三组
WGMMA：

```text
HBM/L2 -> shared: K,V,A,g,beta
state_t -> BF16 state_operand
G0: state_operand @ K^T -> z_t
compute gamma, inv_gamma, gamma_last
state_t *= gamma_last
wait G0 at first z_t use
z_t = beta * (V - gamma * z_t)
z_t -> BF16 z_operand
G1: z_operand @ A^T -> z_t
wait G1 at first z_t use
z_t *= gamma_last / gamma
z_t -> BF16 z_operand
G2: z_operand @ K -> state_t, accumulate
wait G2 before next recurrent chunk
```

prepare 使用 serial loop，不做 software-pipeline/ping-pong。它以较小 shared footprint
换取约 4.4--4.9 CPI 的 long-scoreboard；prepare 是有限 suffix，给它加双缓冲会增加
资源和编译复杂度。

逻辑资源：

| resource | D64 | D128 |
|---|---:|---:|
| state_t FP32 fragment | 32 KiB | 64 KiB |
| state_operand BF16 fragment | 16 KiB | 32 KiB |
| z_t FP32 fragment | 16 KiB | 32 KiB |
| z_operand BF16 fragment | 8 KiB | 16 KiB |
| total fragment | 72 KiB | 144 KiB |
| logical shared | 33.004 KiB | 41.004 KiB |
| ping-pong | 0 | 0 |

shared 的 D64/D128 分项分别是 K 16/16 KiB、V 8/16 KiB、A 8/8 KiB，以及
g/gamma/inv_gamma/beta/gamma_last 1.004 KiB。

## 10. Launch 3 / p=1 main kernel

main 的 chunk 计算体在 CP 与非 CP 路径中相同；差别只是起点 state、grid 和本 CTA
处理的 chunk 范围。

### 10.1 每个 full chunk 的计算与 wait

```text
prefetch current/next Q,K,V,A,g,beta into staged shared buffers

state_t -> state_operand
G0: state_operand @ K^T -> z_t
G1: state_operand @ Q^T -> out_t

compute gamma, inv_gamma, gamma_last
state_t *= gamma_last

wait_group(1)                 # retire G0; G1 remains in flight
z_t = beta * (V - gamma*z_t)
z_t -> z_operand
G2: z_operand @ A^T -> z_t
G3: Q @ K^T -> score

wait_group(1)                 # out_t and A result available; G3 may continue
z_t *= gamma_last / gamma
z_t -> z_operand

wait_group(0)                 # score is consumed for the first time
out_t *= scale * gamma
score = scale * gamma / gamma_last * Lower(score)
score -> score_shared

G4: z_operand @ score^T -> out_t, accumulate
G5: z_operand @ K -> state_t, accumulate

wait_group(1)                 # retire G4 only
out_t -> output_shared -> HBM output
wait_group(0)                 # retire G5 before next state use
```

gamma、`1/gamma` 和 gamma_last 都在 GEMM 发出后提前计算。wait 不紧跟 GEMM；它位于
对应结果第一次被消费之前。G0/G1 后插入 gate 与 state elementwise，G2/G3 之间插入
residual conversion，G5 则与 output conversion/store 重叠。

### 10.2 ping-pong 与内存访问

main loop 使用 `T.Pipelined`，默认 qkva profile 为 Q/K/V/A/g/beta 分配两页 alternating
shared buffer：计算 chunk n 时，global-to-shared async copy 提前请求 chunk n+1。逻辑
ping-pong 容量是 D64 97 KiB、D128 113 KiB。

这些 copy 的 global 地址首先经过 L2。L2 是硬件 cache，不是软件每轮显式执行的
“HBM->L2 copy”；命中时从 L2 返回，miss 才访问 HBM。async copy 的目标是 shared。
gamma、inv_gamma、gamma_last、score/output storage 和 state fragment 没有双缓冲。

pipeline stage handoff、shared operand 可见性和 WGMMA producer/consumer 顺序由 TileLang
生成的 barrier 与显式 warpgroup wait 共同保证。D64 在 G4 退休后把 score shared 当作
output shared 使用；D128 保留独立 output shared。

### 10.3 main fragment/shared

| fragment | D64 | D128 |
|---|---:|---:|
| state_t FP32 | 32 KiB | 64 KiB |
| state_operand BF16 | 16 KiB | 32 KiB |
| z_t FP32 | 16 KiB | 32 KiB |
| z_operand BF16 | 8 KiB | 16 KiB |
| out_t FP32 | 16 KiB | 32 KiB |
| score FP32 | 16 KiB | 16 KiB |
| total logical fragment | 104 KiB | 192 KiB |

| shared group | D64 compact | D128 |
|---|---:|---:|
| staged Q, two pages | 32 KiB | 32 KiB |
| staged K, two pages | 32 KiB | 32 KiB |
| staged V, two pages | 16 KiB | 32 KiB |
| staged A, two pages | 16 KiB | 16 KiB |
| staged g+beta, two pages | 1 KiB | 1 KiB |
| score | 8 KiB | 8 KiB |
| output | alias score | 16 KiB |
| gamma/inv_gamma/gamma_last | 0.504 KiB | 0.504 KiB |
| total logical shared | 105.504 KiB | 137.504 KiB |
| ping-pong subset | 97 KiB | 113 KiB |

逻辑 fragment byte 数是整个 CTA tile，不是每线程寄存器。NCU 编译结果中 D64 main 是
255 regs/thread；D128 CP main 是 249 regs/thread，本轮新 profile 的非 CP batch main
是 246 regs/thread。

## 11. 本轮前后资源占用

本轮只修改 host policy，所以相同 specialization 的资源没有变化：

| kernel | fragment before/after | shared before/after | ping-pong before/after |
|---|---:|---:|---:|
| warmup | `12*Hv` B / same | 0 / same | 0 / same |
| prepare D64 | 72 KiB / same | 33.004 KiB / same | 0 / same |
| prepare D128 | 144 KiB / same | 41.004 KiB / same | 0 / same |
| main D64 compact | 104 KiB / same | 105.504 KiB / same | 97 KiB / same |
| main D128 | 192 KiB / same | 137.504 KiB / same | 113 KiB / same |

变化的是一个 forward 是否分配 CP workspace、launch 数和 grid，不是单 CTA footprint。
batch/deep 从三次 launch 变成一次，也因此完全省掉 warmup/prepare 的临时 fragment 和
shared 使用；这里的“省掉”是这些 kernel 不再 launch，不是 main CTA 变小。

## 12. NCU full profile

命令包含 `--set full --section PmSampling_WarpStates --import-source yes`，每个目标 kernel
做 47-pass replay，并导出 report/details/raw/source。动态/allocated shared 使用 KiB；
stall 数值是每条 issued instruction 的平均 CPI。

| case/launch | grid | threads | duration | regs/thread | dynamic/allocated shared | achieved occupancy | barrier CPI | long scoreboard CPI | wait CPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain warmup | 3 | 32 | 9.57 us | 25 | 0 / 1.000 | 1.56% | 0.000 | 9.893 | 2.793 |
| chain prepare D64 | 24 | 128 | 19.87 us | 173 | 33.016 / 34.125 | 10.35% | 1.315 | 4.863 | 0.855 |
| chain main D64 | 32 | 128 | 227.49 us | 255 | 105.516 / 106.625 | 11.54% | 0.684 | 0.693 | 0.807 |
| long warmup | 7 | 32 | 29.28 us | 25 | 0 / 1.000 | 1.56% | 0.000 | 8.220 | 2.924 |
| long prepare D128 | 56 | 256 | 72.35 us | 168 | 41.016 / 42.125 | 12.35% | 2.313 | 4.419 | 0.827 |
| long main D128 | 64 | 256 | 1.16 ms | 249 | 137.516 / 138.625 | 12.48% | 1.149 | 0.327 | 0.752 |
| batch main D128, CP off | 32 | 256 | 1.31 ms | 246 | 137.516 / 138.625 | 12.50% | 1.033 | 0.467 | 0.694 |

warmup 的 long-scoreboard 高是单 warp 倒序 global load 的延迟，但 kernel 很短。prepare
没有 ping-pong，所以仍有约 4.4--4.9 CPI。main 的 long-scoreboard 只有 0.327--0.693
CPI，说明 qkva ping-pong 正在主路径上隐藏 global load latency。本轮目标是减少 JIT，
不是继续改变这些 stall。

每个下面的 base name 都有 `.ncu-rep`、`.details.txt`、`.raw.csv`、`.source.csv`：

```text
output/ncu_gate_budget_chain_warmup_full
output/ncu_gate_budget_chain_prepare_full
output/ncu_gate_budget_chain_main_full
output/ncu_gate_budget_long_warmup_full
output/ncu_gate_budget_long_prepare_full
output/ncu_gate_budget_long_main_full
output/ncu_gate_budget_batch_main_full
```

profile job 日志：

```text
output/gate_budget_chain_profile_111879.log
output/gate_budget_long_profile_111882.log
output/gate_budget_batch_profile_111877.log
```

## 13. 8-case 性能、speedup 和预计分数

正式日志是 `output/lab3_111870.log`，设备为 14-SM H800 MIG，10 次 warmup、100 次
repetition，CUDA event median，8 个 output/final-state correctness 均 PASS。上一版是
`output/lab3_109719.log`。speedup 定义为 `previous/current`。

公开 `t100` 和估分公式：

```text
ratio = t100 / current
score = min(120, 80 + 20 * ratio)
```

| case | p | previous ms | current ms | speedup | t100/current | estimated score |
|---|---:|---:|---:|---:|---:|---:|
| short_tail_state | off | 0.089216 | 0.088480 | 1.0083x | 3.9110 | 120.00 |
| chain_equal | 4 | 0.300064 | 0.300224 | 0.9995x | 1.6581 | 113.16 |
| parallel_equal | off | 0.278608 | 0.279968 | 0.9951x | 1.8261 | 116.52 |
| parallel_gva | off | 0.255248 | 0.255584 | 0.9987x | 1.9232 | 118.46 |
| long_low_gva | 8 | 1.295616 | 1.294352 | 1.0010x | 1.4361 | 108.72 |
| batch_split_gva | off | 1.262064 | 1.343280 | 0.9395x | 1.1406 | 102.81 |
| wide_gva_state | off | 2.234944 | 2.267184 | 0.9858x | 1.0703 | 101.41 |
| deep_gva_state | off | 2.385488 | 2.591536 | 0.9205x | 1.0924 | 101.85 |

预计公开平均分约 110.37，仍然每个 case 都超过 100。相比上一版约 110.85，执行分数
下降约 0.48；换来的是公开序列少 4 个 JIT specialization 和约 23 秒的 compile-span
余量。这个 tradeoff 是为 OJ 的 process timeout 服务，而不是声称 batch/deep kernel
本身变快。

## 14. JIT 时间验证

| revision/policy | public specializations | summed compile time | first-to-last compile span |
|---|---:|---:|---:|
| previous CP policy | 16 | about 126 s | about 147 s |
| compile-budget policy | 12 | about 107 s | about 124 s |

新的 12 个 specialization 来自：tail main 1 个、chain CP 3 个、parallel MHA main 1 个、
parallel GVA main 1 个、long CP 3 个、batch main 1 个、wide main 1 个、deep main 1 个。
不同 head/initial-state specialization 不能假定互相复用。

## 15. 限制和后续判断

- `base_blocks >= resident_slots` 是保守的 OJ policy，会放弃某些 base grid 已满但拆分后
  仍有小幅 GPU runtime 收益的 shape。
- 当前阈值和 resident capacity 针对 14-SM MIG。换 GPU 或改变 main resource 后必须重算
  `resident_slots`。
- gate threshold 只影响 prepare suffix；没达到阈值时会做精确 full-prefix fallback。
- tail 仍不进入 CP，因为当前 CP 要求等长静态 slices。
- 是否彻底解决 hidden-4 只能由下一次 OJ 完整提交确认；本地证据证明的是 public JIT
  budget 已减少，而不是掌握 hidden shape。

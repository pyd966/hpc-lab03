# Gate-decay context parallelism walkthrough

## 1. 本轮结果和实现边界

本轮在上一最终版 residual-first RS kernel 上加入了 gate-driven intra-card
context parallelism。它利用 log gate 的指数衰减，把一条很长、但 `B*Hv` 很小的
recurrent chain 切成多个可并行的时间分片。

最终默认 `auto` 只改变两个公开 case：

| case | 原 base grid | chunks | sequence parts | main grid | 每个 main CTA 的 chunks |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chain_equal` | 8 | 128 | 8 | 64 | 16 |
| `long_low_gva` | 8 | 512 | 8 | 64 | 64 |

其他 6 个公开 case 继续使用原 specialization。特别是：

- `B>1` 不启用；
- 有 tail 的任务不启用；
- 不是 RS fast path 的任务不启用；
- 原 grid 已填满可驻留 CTA slots 的任务不启用；
- 少于 128 chunks 的任务默认不启用。

因此这轮没有把新的近似或 preparation launch 扩散到 tail、batch split 或高 head
并行 case。

最终 8 个公开 case 全部 PASS。相对上一最终版：

- `chain_equal`: `0.345120 -> 0.306416 ms`，`1.1263x`；
- `long_low_gva`: `1.747600 -> 1.301936 ms`，`1.3423x`；
- 预计公开 8-case 简单平均分：`108.93 -> 110.33`。

最终 benchmark 原始日志是 `output/lab3_108960.log`。

## 2. 和 FlashQLA 的关系

参考来源是：

- Qwen FlashQLA blog；
- 官方仓库 `https://github.com/QwenLM/FlashQLA`；
- 检查的源码提交：`c18a4860ea9cb937f1075d606b4823d6ae34e880`；
- 关键源码：
  `flash_qla/ops/gated_delta_rule/chunk/cp_context.py`、
  `flash_qla/ops/gated_delta_rule/chunk/hopper/cp_fwd.py` 和
  `flash_qla/ops/gated_delta_rule/chunk/hopper/prepare_h.py`。

这不是把 FlashQLA kernel 原样复制进来，而是对当前固定
`chunk=64, DK=DV=128`、residual-first、RS WGMMA 数据流的适配。

FlashQLA 的关键思想是：

1. 把一条长序列切成多个 context-parallel 子序列；
2. 从子序列边界向前累加 chunk-end log gate；
3. 累计 log decay 小于阈值（默认 `-10`）后，认为更早 state 的贡献已经足够小；
4. 用少量 suffix chunks 生成该子序列所需的 initial state；
5. 所有子序列在主 forward 中并行运行。

官方实现还计算 state transfer/correction matrix，在不能达到衰减阈值时顺序修正各
子序列的 state。本实现选择了更适合当前代码的版本：

- 阈值命中：从 0 state 重放边界前的短 suffix；
- 阈值未命中：从 chunk 0 和真实 initial state 开始重放，得到精确 fallback；
- 不生成 transfer matrix；
- preparation 直接生成每个时间分片的 FP32 initial state。

这样多做少量 state-only chunk，但没有引入矩阵 correction kernel，也不改变主
forward 的六组 WGMMA 顺序。

## 3. 数学依据

### 3.1 当前每个 chunk 的 state recurrence

令 chunk 起点 state 为 `S`，当前代码的等价计算为：

```text
P       = K @ S
R_i     = beta_i * (V_i - gamma_i * P_i)
Z       = A @ R
Zhat_i  = (gamma_last / gamma_i) * Z_i
S_next  = gamma_last * S + K^T @ Zhat
```

其中 `gamma_i = exp(g_cumsum_i)`，`g_cumsum` 是每个 64-token chunk
内部的 log gate prefix sum。chunk 最后一个 token 的
`g_cumsum[(chunk+1)*64-1]` 因而正好是该完整 chunk 的总 log decay。

跨多个 chunk 的线性 gate 乘积可以在 log space 中相加。对边界 `b` 向前扫描：

```text
decay_log(W) =
    sum(g_cumsum[(c+1)*64-1], c=b-W,...,b-1)
```

当 `decay_log(W) < -10` 时：

```text
exp(decay_log(W)) < exp(-10) ~= 4.54e-5
```

在 normalized K、sigmoid beta 和当前 delta-rule update 下，旧 state 经过这些
chunks 后的影响被 gate 快速压低。因此 preparation 可以从 0 开始只重放这 W 个
chunks。这里是阈值近似，不声称数学上严格等于完整前缀；最终正确性仍由 output 和
final state 的 `rtol=atol=5e-3` 检查约束。

### 3.2 fallback 为什么是精确的

如果从边界向前扫描到 chunk 0 仍没有达到阈值，warmup-count kernel 返回
`W=b`。preparation 此时：

- `warmup_start=0`；
- 有 initial state 时加载真实 initial state；
- 从 chunk 0 顺序执行相同的 state recurrence；
- 不计算/写回 output，但 state 的数学路径和主 kernel 一致。

所以 fallback 只是更慢，不会因为强行截断产生额外误差。

### 3.3 为什么 warmup 不需要 Q 和 score

边界 initial state 只依赖：

```text
K, V, A, g, beta, previous state
```

它不依赖 Q，也不依赖：

```text
Q @ state
Q @ K^T
score @ Zhat
output store
```

因此 preparation 每个 warmup chunk 只执行 3 组 WGMMA：

```text
G0: P^T     = S^T @ K^T
G1: Z^T     = R^T @ A^T
G2: S^T    += Zhat^T @ K
```

主 kernel 原来每个 chunk 的 6 组 WGMMA 不变。

## 4. Python dispatch 和自动并行度

### 4.1 环境变量

新增：

| variable | default | meaning |
| --- | --- | --- |
| `GDN_GATE_CP` | `auto` | `off/on/auto/2/4/8` |
| `GDN_GATE_CP_THRESHOLD` | `-10.0` | log decay threshold |
| `GDN_GATE_CP_MIN_CHUNKS` | `128` | auto 最短 chain |

`job.sh` 仍使用 `--export NONE`，但现在显式转发白名单中的 `GDN_*`
kernel tuning variables。因此例如：

```bash
GDN_GATE_CP=4 ./job.sh --detach --case chain_equal
```

会把 `GDN_GATE_CP=4` 传入 GPU worker，而不会把任意 host 环境全部导出。

### 4.2 auto 的适用条件

`_select_gate_cp_parts` 要求：

```text
B == 1
full_chunks_only
RS fast path
chunks >= 128
base_blocks < resident_slots
```

定义：

```text
base_blocks    = B * Hv * dv_parts
resident_slots = 14 SM * blocks_per_sm
```

本轮实测资源下：

- D64 compact main: `blocks_per_sm=2`；
- D128 main: `blocks_per_sm=1`。

### 4.3 平方根模型

采用 FlashQLA 的经验模型并适配 14-SM MIG：

```text
estimated_local_chunks =
    3 * sqrt(base_blocks * total_chunks / resident_slots)

local_chunks = nearest_power_of_two(estimated_local_chunks)
local_chunks = max(local_chunks, 4)
```

然后从 1 开始把 `seq_parts` 乘 2，直到每片不长于
`local_chunks`，当前上限为 8。若 chunks 不能被该并行度整除，则按 2 的幂回退。

公开 case 推导如下：

| case | base blocks | resident slots | total chunks | estimated local | rounded local | parts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain D64 | 8 | 28 | 128 | 18.14 | 16 | 8 |
| long D128 | 8 | 14 | 512 | 51.31 | 64 | 8 |

这条规则使用 `B*Hv*dv_parts`、chain length、SM 数和实际 resident blocks，
不是按公开 case 名称选择。

## 5. Host 侧分配和 launch timeline

以启用 CP 的调用为例，`gdn_prefill_forward` 的完整时间线是：

```text
CPU dispatch
  |
  |-- torch.empty(output)                  HBM allocation
  |-- torch.empty(final_state)             HBM allocation
  |-- torch.empty(warmup_counts)           HBM allocation
  |-- torch.empty(cp_states)               HBM allocation
  |
  |-- launch 1: get_gate_cp_warmup
  |
  |-- launch 2: prepare_gate_cp_states
  |
  |-- launch 3: residual RS main, grid *= seq_parts
  |
return output, final_state
```

这些 kernel 在同一个当前 CUDA stream 中顺序提交。launch 2 对 launch 1 的 global
结果、launch 3 对 launch 2 的 `cp_states` 读取由 stream ordering 保证；host
不插入 `cudaDeviceSynchronize`，kernel 内也没有跨 CTA global barrier。

调用前 q/k/v/g/beta/A/initial_state 已是 CUDA tensor。本函数不执行 CPU memory 到
HBM 的 tensor copy。新增 `torch.empty` 只分配 HBM workspace，内容随后由 GPU
kernel 写入。

### 5.1 global workspace

| case | warmup_counts | cp_states |
| --- | ---: | ---: |
| chain: `[1,7,4]` INT32 | 112 B | `[1,7,4,128,128]` FP32 = 1.75 MiB |
| long: `[1,7,8]` INT32 | 224 B | `[1,7,8,128,128]` FP32 = 3.50 MiB |

`seq_part=0` 直接使用 raw initial state/zero，因此 workspace 只保存另外 7 片。

## 6. Launch 1: gate warmup-count kernel

### 6.1 launch mapping

对 `seq_parts=8`：

```text
grid = B * (seq_parts - 1)
block = ceildiv(Hv, 32) * 32 threads
```

chain profile：

```text
grid=7, block=32
```

每个 CTA 负责一个非首时间分片，CTA 内沿 Hv 并行。它为每个 head 从分片边界向前
扫描 chunk-end log gate。

### 6.2 每一步

对边界 `target_begin`：

1. `gate_sum[head]=0`；
2. `warmup_count[head]=target_begin`，先假设要 fallback 到 chunk 0；
3. 从 `target_begin-1` 向 0 扫描；
4. 每次只从 HBM 读取一个 chunk-end `g_cumsum` 标量/head；
5. 第一次满足 `gate_sum < threshold` 时记录 `offset+1`；
6. 后续扫描不再覆盖第一次命中值；
7. 写 `warmup_counts[part-1,head]` 到 HBM。

没有 shared memory、WGMMA 或 ping-pong。逻辑 fragment 为：

| fragment | chain H=4 |
| --- | ---: |
| gate_value FP32[H] | 16 B |
| gate_sum FP32[H] | 16 B |
| warmup_count INT32[H] | 16 B |
| total | 48 B |

NCU 报告 25 registers/thread、0 dynamic shared。该 kernel grid 只有 7，因此
`9.02 us` 中 launch/underfill 占比很高；它不是主吞吐 kernel。

## 7. Launch 2: state-only preparation kernel

### 7.1 launch mapping

```text
base_blocks = B * Hv * dv_parts
grid = base_blocks * (seq_parts - 1)
```

两个公开 CP case 都是：

```text
grid = 8 * 7 = 56
```

映射为：

```text
seq_part  = block // base_blocks + 1
value_blk = block % base_blocks
owner     = value_blk // dv_parts
dv_part   = value_blk % dv_parts
bb        = owner // Hv
bh        = owner % Hv
```

D64 使用 128 threads，D128 使用 256 threads。

### 7.2 state 初始化

每个 CTA 从 `warmup_counts[bb,seq_part-1,bh]` 读取 W：

```text
warmup_start = target_begin - W
```

然后：

- 默认清零 FP32 `state_t`；
- 若 `warmup_start==0` 且调用有 initial state，则加载对应 dv slice；
- 否则保持 0，表示阈值已经允许丢弃更早 prefix。

### 7.3 每个 warmup chunk 的内存访问和计算

preparation 使用动态长度的 `T.serial(W)`。这里故意不使用 ping-pong：

- W 通常只有少量 chunks；
- TileLang 当前不允许把另一个 buffer 中读出的动态 W 用进
  `T.Pipelined` 的跨 scope 地址重写；
- 这部分不在长 steady-state 主循环中，额外 double buffer 收益小。

每个 chunk：

1. 从 HBM 搬 K、V 当前 dv slice、A、g、beta 到单 stage shared；
2. FP32 `state_t` 转成 BF16 `state_operand`；
3. 提交 G0：`S^T @ K^T`；
4. shared g 上计算 gamma、inverse gamma、gamma_last；
5. FP32 state 乘 gamma_last；
6. `warpgroup_wait<0>`，第一次读取 G0；
7. 计算 residual `beta*(V-gamma*P)`；
8. residual FP32 转 BF16 `z_operand`；
9. 提交 G1：`A @ residual`；
10. `warpgroup_wait<0>`，形成 `Zhat`；
11. Zhat FP32 转 BF16 `z_operand`；
12. 提交 G2：`state += K^T@Zhat`；
13. `warpgroup_wait<0>`，下一 chunk 前 state 完成。

结束后把 FP32 state slice 写入 `cp_states` HBM。

### 7.4 preparation fragment

| fragment | D64 | D128 |
| --- | ---: | ---: |
| state_t FP32[dv,128] | 32 KiB | 64 KiB |
| state_operand BF16[dv,128] | 16 KiB | 32 KiB |
| z_t FP32[dv,64] | 16 KiB | 32 KiB |
| z_operand BF16[dv,64] | 8 KiB | 16 KiB |
| logical total | 72 KiB | 144 KiB |
| compiled registers/thread | 173 | 168 |

fragment logical bytes表示 DSL tensor 总量，不等于每线程寄存器字节；RS layout 把
fragment 分布到整个 warp group。

### 7.5 preparation shared memory

| allocation | D64 | D128 | stages |
| --- | ---: | ---: | ---: |
| K BF16[64,128] | 16 KiB | 16 KiB | 1 |
| V BF16[64,dv] | 8 KiB | 16 KiB | 1 |
| A BF16[64,64] | 8 KiB | 8 KiB | 1 |
| g/gamma/inv/beta FP32[64] | 1 KiB | 1 KiB | 1 |
| gamma_last | 4 B | 4 B | 1 |
| logical total | 33.004 KiB | 41.004 KiB | |
| NCU dynamic shared | 33.81 Kbyte | 42.00 Kbyte | |

NCU 的 Kbyte 是报告显示单位；折算及 compiler alignment 后约为
33.02 KiB/41.02 KiB。preparation 没有 ping-pong buffer。

## 8. Launch 3: context-parallel main kernel

### 8.1 grid 和分片

```text
grid = B * Hv * dv_parts * seq_parts
```

chain 和 long 的最终 NCU 都报告 `grid=64`，不是 8：

- chain: `1*4*2*8=64`；
- long: `1*8*1*8=64`。

主 kernel 映射：

```text
seq_part   = block // base_blocks
value_blk  = block % base_blocks
chunk_begin = seq_part * (chunks / seq_parts)
```

各分片写互不重叠的 output token range，所以不需要 CTA 间同步。只有
`seq_part==seq_parts-1` 写 final state；其他 7 片不竞争 final-state 地址。

`seq_part=0` 从 raw initial state/zero 开始；其余分片从 `cp_states` HBM 加载已
准备好的 FP32 state。

### 8.2 主循环的 ping-pong prefetch

主循环仍是原来的 `T.Pipelined` qkva profile。Q/K/V/A/g/beta 的 HBM 到 shared
copy 被放在 pipeline producer stage，下一 chunk 的输入与当前 chunk 的 recurrent
计算重叠。

输入 ping-pong 物理容量：

| input buffers | D64 | D128 |
| --- | ---: | ---: |
| Q, 2 stages | 32 KiB | 32 KiB |
| K, 2 stages | 32 KiB | 32 KiB |
| V, 2 stages | 16 KiB | 32 KiB |
| A, 2 stages | 16 KiB | 16 KiB |
| g + beta, 2 stages each | 1 KiB | 1 KiB |
| total input ping-pong | 97 KiB | 113 KiB |

gamma/inverse gamma、score 和 output staging 仍是单 buffer。NCU source/SASS 中
可以看到 TMA descriptor prefetch `UTMACCTL.PF`、shared-to-global
`UTMASTG.4D`，以及生成的 `WARPGROUP.DEPBAR.LE` 等待。

### 8.3 每个正式 chunk 的完整顺序

主循环沿用上一最终版的 latency-hiding 顺序：

```text
G0: P^T      = S^T @ K^T
G1: out^T    = S^T @ Q^T
gamma / inverse / gamma_last
state       *= gamma_last
wait<1>      retire G0, keep G1
residual     = beta * (V - gamma * P)
G2: Z^T      = residual^T @ A^T
G3: score    = Q @ K^T
wait<1>      retire G1/G2, keep G3
Zhat         = gamma_last/gamma * Z
wait<0>      retire G3 before score use
out         *= scale*gamma
score        = lower(score)*scale*gamma[row]/gamma_last
G4: out^T   += Zhat^T @ score^T
G5: state^T += Zhat^T @ K
wait<1>      G4 done, G5 remains
out_t -> shared -> output HBM
wait<0>      G5 done before next state copy
```

context parallelism 不在这个循环中增加跨 chunk wait。每片只把循环次数从
128/512 降为 16/64。

## 9. Main fragment、shared 和 compact D64

### 9.1 main fragment

| fragment | D64 | D128 |
| --- | ---: | ---: |
| state_t FP32[dv,128] | 32 KiB | 64 KiB |
| state_operand BF16[dv,128] | 16 KiB | 32 KiB |
| z_t FP32[dv,64] | 16 KiB | 32 KiB |
| z_operand BF16[dv,64] | 8 KiB | 16 KiB |
| out_t FP32[dv,64] | 16 KiB | 32 KiB |
| score FP32[64,64] | 16 KiB | 16 KiB |
| logical total | 104 KiB | 192 KiB |
| compiled registers/thread | 255 | 249 |

和上一最终版相比，main fragment shape 完全不变。新增 fragments 只存在于独立
warmup/preparation kernels。

### 9.2 main shared memory

| item | D64 compact CP | D128 CP |
| --- | ---: | ---: |
| input ping-pong | 97 KiB | 113 KiB |
| score BF16 | 8 KiB | 8 KiB |
| output BF16 | 与 score alias | 16 KiB |
| gamma + inverse | 0.5 KiB | 0.5 KiB |
| gamma_last/alignment | 少量 | 少量 |
| logical total | about 105.504 KiB | about 137.504 KiB |
| NCU dynamic shared | 108.05 Kbyte | 140.82 Kbyte |
| binary-unit physical | about 105.52 KiB | about 137.52 KiB |

D64 CP 会启用已有的 score/output lifetime alias：

- G4 读完 score 后，score shared 不再有读者；
- `warpgroup_wait<1>` 保护 G4 完成；
- 同一块 8 KiB shared 随后作为 output staging；
- fragment 不变，shared 减少 8 KiB。

这项 compact 修正把 D64 CP main 从：

| D64 CP main | non-compact | final compact |
| --- | ---: | ---: |
| shared/block | about 113.516 KiB | about 105.516 KiB |
| shared block limit | 1 | 2 |
| theoretical occupancy | 6.25% | 12.50% |
| end-to-end chain | 0.319136 ms | 0.306368 ms smoke |

上一最终版的非 CP D64 fragment/shared 是 104 KiB/about 113.516 KiB；D128 是
192 KiB/about 137.516 KiB。本轮最终：

- 非 CP specialization 不改变这些资源；
- D64 CP main 是 104 KiB/about 105.516 KiB；
- D128 CP main 是 192 KiB/about 137.516 KiB；
- preparation 单独使用 72/144 KiB logical fragment 和约 33/41 KiB shared。

## 10. 同步方式

### 10.1 kernel 间

```text
warmup-count launch
    -> stream dependency
state-preparation launch
    -> stream dependency
main launch
```

没有 host synchronization，也没有 global atomic/barrier。

### 10.2 warmup-count 内

每个 head 的 scan 在 fragment 上顺序累加，没有 shared-memory producer/consumer
关系。

### 10.3 preparation 内

preparation 是 serial chunk loop：

- global/shared copy 完成后 WGMMA 消费 shared；
- 每组 WGMMA 的 accumulator 在首次读取前使用 `warpgroup_wait<0>`；
- state update 在下一 chunk 开始前完成；
- 不存在 ping-pong stage reuse。

### 10.4 main 内

主循环保留：

- compiler-managed async pipeline stage/barrier；
- G0/G1/G2/G3 的延迟 wait；
- G4/G5 的 `wait<1>`，让 state update 和 output staging 重叠；
- 下一 chunk 复制 state 前的 `wait<0>`；
- score/output alias 前的生命周期 wait。

NCU source CSV 中实际存在 `WARPGROUP.DEPBAR.LE gsb0,0/1`，没有把所有 WGMMA
提交后立即逐组等待。

## 11. 并行度扫描

在相同代码上显式测试 `GDN_GATE_CP=off/2/4/8`，5 warmups + 30 repetitions：

| parts | chain ms | long ms | 结论 |
| ---: | ---: | ---: | --- |
| off | 0.351888 | 1.880896 | 对照 |
| 2 | 0.396864 | 1.951760 | 16 CTA 形成不均匀 wave，额外 launch 无法摊薄 |
| 4 | 0.329472 | 1.473344 | 开始有效缩短关键路径 |
| 8 | 0.321184 | 1.300576 | 扫描中最优 |

上表的 D64 part=8 还没有启用最终 compact alias。启用 alias 后 chain 50-repeat
结果为 `0.306368 ms`，因此最终 auto 保留 8 parts。

为什么 part=2 反而慢：

- base grid 是 8；
- D128 非 compact main 每 SM 只能驻留一个 CTA；
- 16 个等长 CTA 在 14 SM 上是 14+2 两个 wave；
- 最后 2 个 CTA 仍执行半条 chain，关键路径接近原完整 chain；
- 同时多付 warmup-count 和 preparation 成本。

平方根模型选择更多、更短的分片，目的不只是让 grid 大于 SM 数，而是减少最后一个
wave 的长尾。

## 12. 正确性和边界验证

### 12.1 公开 8 case

`output/lab3_108960.log` 中 8 个 output/final state 全部 PASS。

### 12.2 额外 CP validation

case 文件：`evaluation/gate_cp_validation_cases.csv`。

默认 `threshold=-10`：

| case | property | result |
| --- | --- | --- |
| cp_initial_random | long chain + initial state | PASS |
| cp_initial_mixed | 一半 token gate=1 + initial state | PASS |
| cp_low_head_initial | Hv=1, dv split, 8 time parts | PASS |

日志：`output/lab3_108971.log`。

强制 `GDN_GATE_CP_THRESHOLD=-100` 后，正常 random gate 在边界前无法达到阈值，
所有非首分片 fallback 到 chunk 0。对应 initial-state case 仍 PASS，日志：
`output/lab3_108973.log`。

### 12.3 tail 和 batch

tail、`B>1` 在 dispatch 前就令 `seq_parts=1`。因此：

- tail 继续使用原 full-prefix + one-tail specialization；
- batch split 继续使用原 batch/head/dv grid；
- 不存在 CP 分片和 tail 分片重叠写 output 的情况。

## 13. 最终 8-case 时间、speedup 和预计分数

测量为 10 warmups + 50 repetitions 的 CUDA-event median。上一最终版来自
`output/lab3_66288.log`，本轮来自 `output/lab3_108960.log`。

公开快档估分沿用：

```text
p = t100 / t
p > 1: score = min(120, 100 + 20*(p-1))
```

| case | previous final ms | gate-CP final ms | speedup | p=t100/t | estimated score |
| --- | ---: | ---: | ---: | ---: | ---: |
| short_tail_state | 0.087232 | 0.086944 | 1.0033x | 3.9801 | 120.00 |
| chain_equal | 0.345120 | **0.306416** | **1.1263x** | 1.6246 | **112.49** |
| parallel_equal | 0.278832 | 0.278000 | 1.0030x | 1.8391 | 116.78 |
| parallel_gva | 0.253744 | 0.254448 | 0.9972x | 1.9318 | 118.64 |
| long_low_gva | 1.747600 | **1.301936** | **1.3423x** | 1.4277 | **108.55** |
| batch_split_gva | 1.338800 | 1.342976 | 0.9969x | 1.1408 | 102.82 |
| wide_gva_state | 2.260384 | 2.224224 | 1.0163x | 1.0910 | 101.82 |
| deep_gva_state | 2.632928 | 2.633600 | 0.9997x | 1.0750 | 101.50 |
| **mean** | | | | | **110.33** |

未启用 CP 的约 -0.3% 到 +1.6% 是跨 job/频率波动；它们使用静态
`seq_parts=1` block mapping，恢复了原 `owner=block//dv_parts`，没有执行新的
dynamic context mapping。

## 14. 完整 NCU profile

所有 profile 使用：

```text
--set full
--section PmSampling_WarpStates
--import-source yes
--replay-mode kernel
--clock-control none
```

每份都已导出 `.ncu-rep`、`.details.txt`、`.raw.csv` 和 `.source.csv`。
details 中均存在 `Section: Warp State Statistics`；raw CSV 还包含每种
`smsp__average_warps_issue_stalled_*` CPI。

### 14.1 launch/resource summary

| kernel | duration | grid/block | regs/thread | dynamic shared | theoretical/achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| chain warmup-count | 9.02 us | 7/32 | 25 | 0 | 50.0% / 1.56% |
| chain D64 prepare | 38.82 us | 56/128 | 173 | 33.81 Kbyte | 12.5% / 11.91% |
| chain D64 compact main | 212 us | 64/128 | 255 | 108.05 Kbyte | 12.5% / 11.60% |
| long D128 prepare | 72.06 us | 56/256 | 168 | 42.00 Kbyte | 12.5% / 12.35% |
| long D128 main | 1.18 ms | 64/256 | 249 | 140.82 Kbyte | 12.5% / 12.48% |

NCU duration 是每个 kernel 在独立 full replay collection 中的值，不能直接相加替代
正常 benchmark；它主要用于同一 report 内的硬件分析。

### 14.2 Warp State CPI

| kernel | total cycles/issue | barrier | GMMA | long SB | short SB | MIO | not selected | wait |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain warmup | 12.37 | 0.000 | 0.000 | 7.478 | 0.035 | 0.000 | 0.000 | 2.824 |
| chain prepare | 9.79 | 1.385 | 0.066 | 5.084 | 0.438 | 0.184 | 0.206 | 0.866 |
| chain compact main | 5.05 | 0.688 | 0.567 | 0.769 | 0.354 | 0.120 | 0.258 | 0.820 |
| long prepare | 10.44 | 2.327 | 0.054 | 4.542 | 0.424 | 0.102 | 0.385 | 0.827 |
| long main | 5.10 | 1.151 | 0.195 | 0.329 | 0.435 | 0.468 | 0.355 | 0.752 |

解释：

- warmup-count/prepare 很短，HBM scalar scan 或 serial global/shared load 使 long
  scoreboard 占比高；
- 它们只占边界 preparation，不随每个正式 output chunk 重复全部 Q/score 工作；
- context-parallel main 的 long scoreboard 已较低；
- D64 compact 把 active warps/scheduler 从约 1 提到 1.89，barrier CPI 只有 0.688；
- long D128 仍是 1 block/SM，但 64 个 CTA、每片 64 chunks 使 wave 分布均匀。

### 14.3 profile 文件

chain warmup-count：

- `output/ncu_gate_cp_chain_warmup_full.ncu-rep`
- `output/ncu_gate_cp_chain_warmup_full.details.txt`
- `output/ncu_gate_cp_chain_warmup_full.raw.csv`
- `output/ncu_gate_cp_chain_warmup_full.source.csv`

chain preparation：

- `output/ncu_gate_cp_chain_prepare_full.ncu-rep`
- `output/ncu_gate_cp_chain_prepare_full.details.txt`
- `output/ncu_gate_cp_chain_prepare_full.raw.csv`
- `output/ncu_gate_cp_chain_prepare_full.source.csv`

chain final compact main：

- `output/ncu_gate_cp_chain_main_full.ncu-rep`
- `output/ncu_gate_cp_chain_main_full.details.txt`
- `output/ncu_gate_cp_chain_main_full.raw.csv`
- `output/ncu_gate_cp_chain_main_full.source.csv`

long preparation：

- `output/ncu_gate_cp_long_prepare_full.ncu-rep`
- `output/ncu_gate_cp_long_prepare_full.details.txt`
- `output/ncu_gate_cp_long_prepare_full.raw.csv`
- `output/ncu_gate_cp_long_prepare_full.source.csv`

long main：

- `output/ncu_gate_cp_long_main_full.ncu-rep`
- `output/ncu_gate_cp_long_main_full.details.txt`
- `output/ncu_gate_cp_long_main_full.raw.csv`
- `output/ncu_gate_cp_long_main_full.source.csv`

本轮可复现 profile 入口是：

```bash
./profile_gate_cp.sh CASE KERNEL_NAME REPORT_BASENAME
```

它需要在带 NCU 的 lab3 GPU worker 中运行。脚本会自动生成 report 和三种文本导出。

## 15. 当前限制和后续空间

1. 当前 CP 只支持等长、整除的 full-chunk 分片；不支持 CP 内 tail。
2. 当前最大 `seq_parts=8`，避免 workspace 和 preparation launch 无界增长。
3. `threshold=-10` 沿用 FlashQLA；显式调高阈值会减少 warmup 但扩大近似误差。
4. preparation 使用 serial single-stage shared load。其 long scoreboard 为
   4.5--5.1 CPI，是下一步最明确的局部优化点；可以研究先独立生成
   `warmup_start` 后对固定上限做 masked two-stage pipeline。
5. D128 main 仍由 registers/shared 限制为 1 block/SM；本轮收益来自时间分片和更均匀
   waves，没有解决 D128 单 CTA 资源问题。
6. 新增 CP state workspace 是以 HBM 容量换并行度。公开 case 最大为 3.5 MiB，
   但更大的 Hv/parts 需要在 dispatch 中继续受上限约束。

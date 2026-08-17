# Gate-decay context parallelism policy tuning walkthrough

## 1. 本轮目标、结论和边界

本轮不改变 TileLang 主 kernel 的数学路径和单 CTA 数据流，而是重新决定 gate-driven
context parallelism (下文简写为 gate CP) 是否启用，以及序列应该切成
`seq_parts = 2/4/8` 中的哪一种。

旧规则有两个明显限制：

1. 只允许 `B == 1`，因此 `batch_split_gva` 和其他 batch workload 即使 CTA 数不足也
   不能沿序列展开；
2. D128 路径只使用一个平方根启发式，没有显式计算主 kernel 波次数和额外
   warmup/prepare launch 的成本。

最终实现具有以下性质：

- 不检查 case 名称，不包含公开 case 的 shape 特判；
- 第一阶输入是 `B * Hv * dv_parts`、`ceil(T / 64)` 和每个 SM 可驻留的 CTA 数；
- 第二阶输入是 `Hq / Hv`，用于描述 GVA 的 Q/K cache reuse；
- D64 compact 路径继续使用局部链长的平方根模型；
- D128 路径在 `off/2/4/8` 间显式比较 main、prepare 和 scalar scan 的波次成本；
- 仍只处理无 tail 的 full-chunk RS 路径；
- `GDN_GATE_CP_MIN_CHUNKS` 从 128 降为 64；
- 显式 `GDN_GATE_CP=2/4/8` 可用于强制实验，`auto/on` 使用同一自动选择器。

在 38 个合成 shape 上，所有执行都通过 output 和 final state 正确性检查。29 个 shape
完整测了 `off/2/4/8`，另外 9 个边界 shape 测了 `off/4/8`；自动 policy 在 38/38
个点上都处于实测候选最优值的 3% 以内：

```text
mean(policy_time / measured_oracle_time) = 1.000553
max (policy_time / measured_oracle_time) = 1.017185
```

最大误差点是 `B=1,Hq=2,Hv=8,chunks=128`：policy 选择 p4，实测 p8 快
1.72%。这是一个很平的边界点，不值得添加 shape 特判。

正式 8-case 测试使用 10 次 warmup 和 100 次计时，全部 PASS。预计平均分由上一版
110.33 提高到 110.85。正式日志为 `output/lab3_109719.log`。

## 2. 记号和并行层次

本文使用：

```text
B          batch size
T          每个 batch 的 token 数
Hq         Q/K head 数
Hv         V/state head 数
C          chunks_per_batch = T / 64（CP 只处理整除 64 的路径）
dv_tile    一个 CTA 计算的 value 维度，64 或 128
dv_parts   128 / dv_tile，分别是 2 或 1
owners     B * Hv
base_blocks = owners * dv_parts
p          seq_parts，取 1/2/4/8；p=1 就是关闭 CP
SM         当前 H800 MIG 实例有 14 个 SM
```

这里的 “D64/D128 路径” 指 `dv_tile=64/128`，不是说输入 state 的完整 Dv 改变了；
完整 Dv 始终为 128。D64 用两个 CTA 分别计算 `[0:64]` 和 `[64:128]`。

自动 dv split 先于 gate CP 选择：

```text
if B * Hv * 2 <= 14:
    dv_tile  = 64
    dv_parts = 2
else:
    dv_tile  = 128
    dv_parts = 1
```

随后 gate CP 把每个 base block 再复制 p 份，每份处理一个连续时间片：

```text
main_grid = B * Hv * dv_parts * p
local_chunks = C / p
```

例如 `chain_equal` 的 `B=1,Hv=4`，先得到 D64 的 8 个 base blocks。最终 policy
选择 p4，所以 main grid 是 32，而不是 8；每个 CTA 处理 128/4 = 32 chunks。

## 3. 为什么 gate 衰减允许切开 recurrent chain

### 3.1 每个 chunk 的等价数学路径

令 chunk 起点 state 为 `S`，`gamma_i = exp(g_i)`，当前 residual-first kernel 的
等价关系为：

```text
P          = K @ S
O_state    = Q @ S
R_i        = beta_i * (V_i - gamma_i * P_i)
Z          = A @ R
Zhat_i     = gamma_last / gamma_i * Z_i
Score      = Lower(Q @ K^T)
O          = scale * gamma_i * O_state
             + scale * gamma_i / gamma_last * Score @ Zhat
S_next     = gamma_last * S + K^T @ Zhat
```

kernel 内实际把 state 和 Z 转置成以 Dv 为行的 fragment，从而直接匹配 RS WGMMA；
上述公式只是更容易阅读的非转置写法。

`g` 是每个 64-token chunk 内的 log-gate prefix sum。因此每个 chunk 最后一个 token
的 `g[(chunk+1)*64-1]` 就是该 chunk 的总 log decay。跨 chunk 的乘法衰减可以在
log space 中相加。

### 3.2 warmup suffix

对于时间片边界 b，launch 1 从 b 前一个 chunk 向前扫描：

```text
decay_log(W) = sum(chunk_end_log_gate[b-W : b])
```

默认阈值是 -10。当累计值小于 -10 时：

```text
exp(decay_log(W)) < exp(-10) ~= 4.54e-5
```

preparation kernel 可以从 0 state 开始，只重放边界前 W 个 chunks，得到该时间片的
近似 initial state。如果扫到 chunk 0 仍未命中阈值，则 `warmup_start=0`，加载真实
initial state（如果存在），完整重放 `[0,b)`。后一条是精确 fallback，只会变慢，
不会引入截断误差。

preparation 只需构造 state，因此不读 Q，也不计算 Q@S、Q@K 或 output。每个 warmup
chunk 只有 3 组 WGMMA，而主 kernel 有 6 组。

## 4. 自动 policy

### 4.1 共同 eligibility

以下任一条件成立时直接返回 p1：

```text
GDN_GATE_CP == off
存在 tail，即 T % 64 != 0
未进入 RS WGMMA fast path
C < GDN_GATE_CP_MIN_CHUNKS（默认 64）
```

与旧版不同，这里不再要求 `B == 1`。如果最终 p 不能整除 C，则按 2 的幂回退，直到
能形成等长静态 slice。当前不为 CP 单独增加 tail slice specialization。

### 4.2 D64 compact 路径

D64 CP main 复用 score/output shared storage，实测可以 2 CTA/SM，因此：

```text
resident_slots = 14 * 2 = 28

estimated_local_chunks =
    3 * sqrt(base_blocks * C / resident_slots)

target_local_chunks =
    max(4, next_power_of_two(estimated_local_chunks))
```

然后从 p1 开始翻倍，直到 `C / p <= target_local_chunks`，最多 p8。

这里使用向上取整的 power of two，而不是旧版的最近 power of two。边界点会优先选择
更小的 p，原因是更大的 p 不仅增加 CTA，还线性增加 `p-1` 份 cp-state preparation。

两个例子：

```text
B=1,Hv=1,C=64:
base_blocks=2, estimated=6.41, target=8,  policy=p8

B=1,Hv=4,C=128:
base_blocks=8, estimated=18.14, target=32, policy=p4
```

### 4.3 D128 波次成本模型

D128 main 受寄存器和 shared memory 限制，只能 1 CTA/SM，所以 resident slots 是 14。
对 p1/2/4/8 计算：

```text
base_cost(p1) = C * ceil(base_blocks / 14)

main_waves(p)    = ceil(base_blocks * p       / 14)
prepare_waves(p) = ceil(base_blocks * (p - 1) / 14)

main_cost(p) = C * main_waves(p) / p
prepare_cost(p) = 15 * prepare_waves(p)
scan_cost(p) = 0.05 * C * (p - 1) / p

qk_reuse_fraction = max(0, 1 - Hq/Hv)
qk_reuse_bonus = 0.05 * C * qk_reuse_fraction

total_cost(p) =
    main_cost + prepare_cost + scan_cost - qk_reuse_bonus
```

最后选择 cost 最小者。各项含义：

- `main_cost` 计算 p 倍 grid 在 14 SM 上要跑几波，并除以每个 CTA 缩短后的链长；
- `prepare_cost` 计算 p-1 组边界 state CTA 的波次数；常数 15 是跨所有扫描 shape 使用的
  preparation wave 等价成本，不针对单个公开 case；
- `scan_cost` 描述 gate endpoint 的串行读取和额外 launch；
- GVA 中多个 V head 共享较少的 Q/K head，L2 对重复 Q/K tile 的复用更好，所以给所有
  CP 候选一个小的 cache reuse allowance；
- ceil 波次使 B 和 Hv 不再被孤立看待，真正的一阶变量是 `B*Hv`。

这个模型能表达几个仅凭“原 grid 是否填满 SM”无法表达的现象：

- base grid 8、C=128 的 D128 MHA 关闭 CP 更好；
- 同样 base grid 8、C=256/512 时 p8 才开始摊平 preparation；
- base grid 16、C=128/256 时 p4 最好，因为 p4 正好形成合适波次；
- base grid 32 时 p2 比 p4/p8 更好；
- base grid 64、C=128 已有充足并行度，关闭 CP 更好；
- base grid 12 即使 C=1024 仍可能关闭，因为每个 CP 候选都会引入不利的离散波次。

### 4.4 batch/head/deep 如何影响选择

结论不是“batch 大就开”或“head 少就开”，而是：

1. `B*Hv*dv_parts` 决定 base CTA 数和每个候选的离散波次数；
2. C 决定 main 串行链有多深，以及 prepare 成本能否被摊薄；
3. `Hq/Hv` 只在 GVA 时对 L2 reuse 做小修正；
4. B 和 Hv 的分解通常不重要，只要乘积相同。

实测 `B1H8`、`B2H4`、`B4H2` 在 D128、C=256 的 p8 时间分别为
0.853056、0.846608、0.844256 ms，支持把 `B*Hv` 作为一阶变量。D128、owner=32
的 C=256/512 均选择 p2；owner=64、C=128 则关闭。

## 5. 38-shape 强制模式扫描

测试定义在 `evaluation/gate_cp_policy_sweep_cases.csv`。计时使用 5 次 warmup、20 次
repeat，单位为 ms。`-` 表示该补充边界点没有跑 p2；policy regret 的 oracle 只在
实际测过的候选中取最小值。

| case | B | Hq/Hv | C | off | p2 | p4 | p8 | policy | best | regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d64_b1h1_c64 | 1 | 1/1 | 64 | 0.182912 | 0.142176 | 0.104064 | 0.094912 | 8 | 8 | 1.0000 |
| d64_b1h1_c128 | 1 | 1/1 | 128 | 0.336752 | 0.227360 | 0.147952 | 0.132416 | 8 | 8 | 1.0000 |
| d64_b1h1_c256 | 1 | 1/1 | 256 | 0.648976 | 0.394176 | 0.237648 | 0.202544 | 8 | 8 | 1.0000 |
| d64_b1h1_c512 | 1 | 1/1 | 512 | 1.258656 | 0.729952 | 0.411856 | 0.345120 | 8 | 8 | 1.0000 |
| d64_b1h4_c64 | 1 | 4/4 | 64 | 0.190976 | 0.193664 | 0.185024 | 0.209200 | 4 | 4 | 1.0000 |
| d64_b1h4_c128 | 1 | 4/4 | 128 | 0.349968 | 0.321152 | 0.299392 | 0.308688 | 4 | 4 | 1.0000 |
| d64_b1h4_c256 | 1 | 4/4 | 256 | 0.671920 | 0.586928 | 0.524224 | 0.485664 | 8 | 8 | 1.0000 |
| d64_b1h4_c512 | 1 | 4/4 | 512 | 1.318880 | 1.094240 | 0.996624 | 0.875760 | 8 | 8 | 1.0000 |
| d64_b2h2_c256 | 2 | 2/2 | 256 | 0.667088 | 0.574576 | 0.520080 | 0.475056 | 8 | 8 | 1.0000 |
| d64_b4h1_c256 | 4 | 1/1 | 256 | 0.644400 | 0.574304 | 0.526832 | 0.482528 | 8 | 8 | 1.0000 |
| d128_b1h8_c64 | 1 | 8/8 | 64 | 0.246336 | 0.328016 | 0.301312 | 0.334288 | off | off | 1.0000 |
| d128_b1h8_c128 | 1 | 8/8 | 128 | 0.462752 | 0.580448 | 0.495968 | 0.504912 | off | off | 1.0000 |
| d128_b1h8_c256 | 1 | 8/8 | 256 | 0.893952 | 1.093232 | 0.902272 | 0.853056 | 8 | 8 | 1.0000 |
| d128_b1h8_c512 | 1 | 8/8 | 512 | 1.751168 | 2.117488 | 1.705808 | 1.545264 | 8 | 8 | 1.0000 |
| d128_b1h8g2_c128 | 1 | 2/8 | 128 | 0.477536 | 0.541456 | 0.446064 | 0.438528 | 4 | 8 | 1.0172 |
| d128_b1h8g2_c512 | 1 | 2/8 | 512 | 1.812720 | 1.956176 | 1.505184 | 1.314720 | 8 | 8 | 1.0000 |
| d128_b2h4_c256 | 2 | 4/4 | 256 | 0.888336 | 1.086608 | 0.897072 | 0.846608 | 8 | 8 | 1.0000 |
| d128_b4h2_c256 | 4 | 2/2 | 256 | 0.891856 | 1.079552 | 0.887936 | 0.844256 | 8 | 8 | 1.0000 |
| d128_b1h12_c256 | 1 | 12/12 | 256 | 1.018000 | 1.130240 | 1.219344 | 1.240528 | off | off | 1.0000 |
| d128_b1h14_c256 | 1 | 14/14 | 256 | 1.203248 | 1.260864 | 1.324608 | 1.423776 | off | off | 1.0000 |
| d128_b1h16_c128 | 1 | 16/16 | 128 | 1.024368 | 0.888992 | 0.848512 | 0.963632 | 4 | 4 | 1.0000 |
| d128_b1h16_c256 | 1 | 16/16 | 256 | 1.996992 | 1.683360 | 1.543088 | 1.662096 | 4 | 4 | 1.0000 |
| d128_b1h16_c512 | 1 | 16/16 | 512 | 3.988320 | 3.271424 | 2.939600 | 3.059984 | 4 | 4 | 1.0000 |
| d128_b2h8_c256 | 2 | 8/8 | 256 | 1.994416 | 1.668816 | 1.532624 | 1.635888 | 4 | 4 | 1.0000 |
| d128_b4h4_c256 | 4 | 4/4 | 256 | 1.987200 | 1.653792 | 1.511504 | 1.615744 | 4 | 4 | 1.0000 |
| d128_b1h8g4_c64 | 1 | 4/8 | 64 | 0.248816 | - | 0.277168 | 0.304912 | off | off | 1.0000 |
| d128_b1h8g4_c128 | 1 | 4/8 | 128 | 0.466768 | - | 0.448480 | 0.449104 | 4 | 4 | 1.0000 |
| d128_b1h8g4_c256 | 1 | 4/8 | 256 | 0.904656 | - | 0.796608 | 0.735296 | 8 | 8 | 1.0000 |
| d128_b1h8g1_c64 | 1 | 1/8 | 64 | 0.253664 | - | 0.273712 | 0.297392 | off | off | 1.0000 |
| d128_b1h8g1_c128 | 1 | 1/8 | 128 | 0.470624 | - | 0.442816 | 0.441120 | 4 | 8 | 1.0038 |
| d128_b1h8g1_c256 | 1 | 1/8 | 256 | 0.908848 | - | 0.794016 | 0.735776 | 8 | 8 | 1.0000 |
| d128_b1h16_c64 | 1 | 16/16 | 64 | 0.523232 | 0.492240 | 0.499840 | 0.607424 | 2 | 2 | 1.0000 |
| d128_b1h12_c512 | 1 | 12/12 | 512 | 2.016384 | - | 2.318336 | 2.273328 | off | off | 1.0000 |
| d128_b1h12_c1024 | 1 | 12/12 | 1024 | 4.009072 | - | 4.484416 | 4.335392 | off | off | 1.0000 |
| d128_b1h32g8_c256 | 1 | 8/32 | 256 | 2.664816 | 2.382368 | 2.543968 | 2.586976 | 2 | 2 | 1.0000 |
| d128_b1h32g8_c512 | 1 | 8/32 | 512 | 5.351056 | 4.632896 | 4.776096 | 4.833264 | 2 | 2 | 1.0000 |
| d128_b4h8g2_c128 | 4 | 2/8 | 128 | 1.352704 | 1.285424 | 1.374592 | 1.506608 | 2 | 2 | 1.0000 |
| d128_b1h64g16_c128 | 1 | 16/64 | 128 | 2.275232 | - | 2.585248 | 2.908192 | off | off | 1.0000 |

原始强制模式日志：

```text
D64:  output/lab3_109341.log  output/lab3_109380.log
      output/lab3_109399.log  output/lab3_109426.log
D128: output/lab3_109356.log  output/lab3_109385.log
      output/lab3_109408.log  output/lab3_109437.log
补充: output/lab3_109480.log  output/lab3_109507.log
      output/lab3_109516.log  output/lab3_109534.log
      output/lab3_109547.log  output/lab3_109568.log
      output/lab3_109590.log
auto: output/lab3_109641.log  output/lab3_109655.log
```

## 6. Host 侧完整 launch timeline

调用 `gdn_prefill_forward` 时，Q/K/V/g/beta/A/initial_state 已经是 CUDA tensor。本函数
不会把 CPU memory 搬到 HBM；CPU 到 HBM 的传输发生在调用方创建或 `.cuda()` 这些
tensor 时。当前函数只用 `torch.empty` 在 HBM 分配 output、final_state 和可选 CP
workspace。

启用 p>1 时，单次 forward 的时间线是：

```text
CPU dispatch and specialization selection
  |
  |-- allocate output and final_state in HBM
  |-- allocate warmup_counts and cp_states in HBM
  |
  |-- launch 1: get_gate_cp_warmup_sp{p}
  |      read chunk-end g from HBM/L2
  |      write warmup_counts to HBM
  |
  |-- launch 2: prepare_gate_cp_dv{tile}x{parts}_sp{p}
  |      read warmup_counts
  |      replay K/V/A/g/beta suffix
  |      write cp_states to HBM
  |
  |-- launch 3: residual_*_cp{p}
         read initial_state or cp_states
         process p time slices in parallel
         write output; last time slice writes final_state
```

三个 kernel 提交到同一个当前 CUDA stream。launch 2 读取 launch 1 的结果、launch 3
读取 launch 2 的结果由 stream ordering 保证。host 没有插入
`cudaDeviceSynchronize`，kernel 间也没有跨 CTA global barrier。

p=1 时不分配 CP workspace，也不提交前两个 launch，直接执行原 main kernel。

### 6.1 global workspace

```text
warmup_counts bytes = B * (p-1) * Hv * 4
cp_states bytes = B * (p-1) * Hv * 128 * 128 * 4
```

代表 case：

| case | p | warmup_counts | cp_states |
|---|---:|---:|---:|
| chain, B1 H4 | 4 | 48 B | 768 KiB |
| long, B1 H8 | 8 | 224 B | 3.50 MiB |
| batch, B4 H8 | 2 | 128 B | 2.00 MiB |
| deep, B1 H32 | 2 | 128 B | 2.00 MiB |

## 7. Launch 1: warmup-count kernel

### 7.1 grid 和工作

```text
grid    = B * (p - 1)
threads = ceil(Hv / 32) * 32
```

一个 CTA 对应一个 `(batch, nonzero_seq_part)`，线程并行覆盖所有 Hv heads。每个 head
维护累计 log gate 和第一次越过阈值时的 suffix 长度。扫描是串行 offset loop，因为
每一步依赖前一步累计值。

### 7.2 fragment、shared 和访问

| storage | shape | dtype | bytes |
|---|---|---|---:|
| gate_value | `[Hv]` | FP32 | `4*Hv` |
| gate_sum | `[Hv]` | FP32 | `4*Hv` |
| warmup_fragment | `[Hv]` | INT32 | `4*Hv` |
| total fragment | | | `12*Hv` |

没有用户声明的 shared memory，也没有 ping-pong buffer。每个迭代直接读取一个 chunk
endpoint gate；同一 CTA 后续读取连续倒序地址，其他 boundary CTA 可能从 L2 命中相同
gate 数据。最后一次 global store 写 `warmup_counts`。

NCU 报告 launch allocation 为 1 KiB/block，这是 CUDA/TileLang 的 per-block allocation，
不是源码中声明的 shared array。

## 8. Launch 2: cp-state preparation kernel

### 8.1 grid 映射

```text
grid    = base_blocks * (p - 1)
threads = 128 for dv_tile=64, 256 for dv_tile=128
```

block 首先映射到 seq_part，再映射到 `(B,Hv,dv_part)`。每个 CTA 读取对应的
warmup_count，决定 `warmup_start`，顺序重放 suffix，最后把 `[128,dv_tile]` state tile
写入 cp_states。

### 8.2 每个 warmup chunk

```text
1. HBM/L2 -> shared: K, V, A, g, beta
2. fragment state_t -> BF16 state_operand
3. G0: state_operand @ K^T -> z_t             (K @ S)
4. 计算 gamma、1/gamma、gamma_last
5. state_t *= gamma_last
6. wait G0，计算 beta * (V - gamma * z_t)
7. z_t -> BF16 z_operand
8. G1: z_operand @ A^T -> z_t                 (A @ residual)
9. wait G1，z_t *= gamma_last / gamma
10. z_t -> BF16 z_operand
11. G2: z_operand @ K -> state_t, accumulate  (state update)
12. wait G2 后进入下一 chunk
```

prepare loop 是 `T.serial`，没有 software pipeline，也没有 ping-pong prefetch。每组
global-to-shared copy 在同一 chunk 内使用；TileLang 生成必要的 copy/use 同步。这样保持
prepare shared 较小，因为它通常只占总耗时的一小部分，额外 double buffer 会直接增加
波次资源压力。

### 8.3 logical resource

fragment：

| fragment | D64 | D128 |
|---|---:|---:|
| state_t FP32 `[dv,128]` | 32 KiB | 64 KiB |
| state_operand BF16 `[dv,128]` | 16 KiB | 32 KiB |
| z_t FP32 `[dv,64]` | 16 KiB | 32 KiB |
| z_operand BF16 `[dv,64]` | 8 KiB | 16 KiB |
| total | 72 KiB | 144 KiB |

shared：

| shared group | D64 | D128 |
|---|---:|---:|
| K | 16 KiB | 16 KiB |
| V | 8 KiB | 16 KiB |
| A | 8 KiB | 8 KiB |
| g/gamma/inv_gamma/beta/gamma_last | 1.004 KiB | 1.004 KiB |
| logical total | 33.004 KiB | 41.004 KiB |
| ping-pong | 0 | 0 |

## 9. Launch 3: context-parallel main kernel

### 9.1 grid、initial state 和 final state

```text
grid = base_blocks * p
local chunks = C / p
```

`seq_part=0` 从真实 initial state 或零 state 开始；其他 part 从 cp_states 加载准备好的
state。各 CTA 写互不重叠的 output token slice，所以不需要跨 CTA 同步。只有
`seq_part == p-1` 的 CTA 写 final_state，避免多个 part 竞争同一地址。

### 9.2 full chunk 的计算和 wait 顺序

主 loop 中每个 chunk 的顺序是：

```text
prefetch current/next Q,K,V,A,g,beta into staged shared buffers

state_t -> state_operand
G0: state_operand @ K^T -> z_t       # K @ S
G1: state_operand @ Q^T -> out_t     # Q @ S

compute gamma, inv_gamma, gamma_last
state_t *= gamma_last

wait_group(1)                        # retire G0; G1 stays in flight
z_t = beta * (V - gamma * z_t)
z_t -> z_operand
G2: z_operand @ A^T -> z_t           # A @ residual
G3: Q @ K^T -> score

wait_group(1)                        # Q@S and A path available;
z_t *= gamma_last / gamma            # QK remains outstanding
z_t -> z_operand

wait_group(0)                        # score is now consumed
out_t *= scale * gamma
score = scale * gamma[row] / gamma_last * Lower(score)
score -> score_shared

G4: z_operand @ score^T -> out_t, accumulate
G5: z_operand @ K -> state_t, accumulate

wait_group(1)                        # retire G4 only
out_t -> output_shared -> HBM output # score storage can now be reused
wait_group(0)                        # retire G5 before next state copy
```

这个顺序保持先前专门做过的 latency hiding：GEMM 结果不立即全部 wait。gamma/state
elementwise 工作插在 G0/G1 后；residual 和 operand conversion 插在后续 WGMMA 之间；
G5 state update 在 output conversion/store 时保持 in flight。每次 wait 都放在第一次真正
消费对应结果之前。

### 9.3 ping-pong prefetch

main loop 使用 `T.Pipelined`：

```text
pipeline_order = [5,4,0,1,2,3,6,7]
pipeline_stage = [0,0,0,0,0,0,1,1]  # q,k,v,a,g,beta are staged
```

默认 `qkva` profile 把 Q/K/V/A 以及同一 gate stage 的 g/beta 放入 stage 0。编译器为
这些 global-to-shared copy 分配两页 alternating buffers，在计算 chunk n 时预取
chunk n+1。gamma、inv_gamma、score/output storage 和 state fragment 不做 ping-pong。

HBM 请求首先经过 L2；是否真的访问 HBM 取决于 cache hit。所谓 prefetch 是提前发出
global-to-shared async copy，不是每个 chunk 都显式把 HBM 数据“先搬进 L2”。L2 是硬件
cache，copy miss 时才由 HBM 填充。

### 9.4 main logical fragment

| fragment | D64 | D128 |
|---|---:|---:|
| state_t FP32 | 32 KiB | 64 KiB |
| state_operand BF16 | 16 KiB | 32 KiB |
| z_t FP32 | 16 KiB | 32 KiB |
| z_operand BF16 | 8 KiB | 16 KiB |
| out_t FP32 | 16 KiB | 32 KiB |
| score FP32 | 16 KiB | 16 KiB |
| total logical fragment | 104 KiB | 192 KiB |

这些是整个 CTA 的逻辑 tile byte 数，不等于“每线程寄存器 byte 数”。layout 把 tile 分散
到 128/256 个线程；实际 compiled register 数见 NCU 表。

### 9.5 main shared 和 ping-pong

| shared group | D64 compact CP | D128 CP |
|---|---:|---:|
| staged Q, two pages | 32 KiB | 32 KiB |
| staged K, two pages | 32 KiB | 32 KiB |
| staged V, two pages | 16 KiB | 32 KiB |
| staged A, two pages | 16 KiB | 16 KiB |
| staged g+beta, two pages | 1 KiB | 1 KiB |
| score | 8 KiB | 8 KiB |
| output | aliases score, 0 extra | 16 KiB |
| gamma+inv_gamma+gamma_last | 0.504 KiB | 0.504 KiB |
| logical total | 105.504 KiB | 137.504 KiB |
| ping-pong staged inputs | 97 KiB | 113 KiB |

D64 在 G4 完成后复用 score_shared 作为 output_shared。显式 `wait_group(1)` 保证 score
消费者已经退休，然后才覆盖该 shared storage。D128 为保持既有性能，仍保留独立的
16 KiB output shared。

## 10. 本轮前后 resource 对比

本轮只改变 host policy，没有修改 warmup、prepare 或 main kernel body。因此相同
specialization 的 fragment、shared 和 ping-pong 占用在修改前后完全一致：

| kernel | logical fragment before/after | logical shared before/after | ping-pong before/after |
|---|---:|---:|---:|
| warmup | `12*Hv` B / same | 0 / same | 0 / same |
| prepare D64 | 72 KiB / same | 33.004 KiB / same | 0 / same |
| prepare D128 | 144 KiB / same | 41.004 KiB / same | 0 / same |
| main D64 compact | 104 KiB / same | 105.504 KiB / same | 97 KiB / same |
| main D128 | 192 KiB / same | 137.504 KiB / same | 113 KiB / same |

变化的是 launch 数、main grid、每个 main CTA 的 chain length 和 HBM workspace 总量，
不是单 CTA resource。

## 11. 完整 NCU profile

### 11.1 方法

`profile_gate_cp_policy.sh` 对三类启用 CP 的代表 case 分别 profile warmup、prepare 和
main，共 9 个 launch：

```text
chain: D64, p4
batch: D128, p2
deep:  D128, p2
```

底层命令使用：

```text
--set full
--section PmSampling_WarpStates
--import-source yes
--clock-control none
--replay-mode kernel
```

9 个 `.details.txt` 都已确认包含 `Section: Warp State Statistics`。每个 base name 同时
生成 `.ncu-rep`、`.details.txt`、`.raw.csv` 和 `.source.csv`。

### 11.2 launch resource 和 stall CPI

`dynamic smem` 和 `allocated smem` 是 NCU 实测，单位 KiB；下表从 NCU 的 byte 值换算。

| case/launch | grid | threads | time | regs/thread | dynamic/allocated smem | active warps | barrier CPI | long scoreboard CPI | wait CPI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| chain warmup | 3 | 32 | 9.632 us | 25 | 0 / 1.000 | 1.56% | 0.000 | 10.386 | 2.793 |
| chain prepare D64 | 24 | 128 | 19.264 us | 173 | 33.016 / 34.125 | 10.34% | 1.332 | 4.814 | 0.861 |
| chain main D64 | 32 | 128 | 226.560 us | 255 | 105.516 / 106.625 | 11.50% | 0.679 | 0.667 | 0.812 |
| batch warmup | 4 | 32 | 9.600 us | 22 | 0 / 1.000 | 1.56% | 0.000 | 13.707 | 2.871 |
| batch prepare D128 | 32 | 256 | 50.816 us | 168 | 41.016 / 42.125 | 12.35% | 2.242 | 4.242 | 0.827 |
| batch main D128 | 64 | 256 | 1.175072 ms | 249 | 137.516 / 138.625 | 12.49% | 1.109 | 0.340 | 0.753 |
| deep warmup | 1 | 32 | 15.680 us | 22 | 0 / 1.000 | 1.56% | 0.000 | 15.591 | 2.910 |
| deep prepare D128 | 32 | 256 | 50.112 us | 168 | 41.016 / 42.125 | 12.33% | 2.313 | 4.230 | 0.844 |
| deep main D128 | 64 | 256 | 2.319936 ms | 249 | 137.516 / 138.625 | 12.49% | 0.964 | 0.336 | 0.767 |

解释：

- warmup 只有 1 到 4 个单 warp CTA，long scoreboard CPI 很高但总时间只有约
  10-16 us；它反映串行 endpoint load latency，不是 main bottleneck；
- prepare 的 long scoreboard 约 4.2-4.8 CPI，原因是它没有 ping-pong，短 suffix 中
  global load latency 难以摊薄；但总成本约 19-51 us；
- main 的 long scoreboard 已降到 0.34-0.67 CPI，说明 qkva ping-pong 对主路径有效；
- main 的 barrier 约 0.68-1.11 CPI，来自 pipeline stage 交接、shared operand 可见性和
  WGMMA producer/consumer 顺序，是当前显式 overlap 的代价；
- D64 255 regs/thread、D128 249 regs/thread 与上一版完全相同，证实本轮 policy 没改变
  compiled per-CTA resource。

### 11.3 artifacts

以下 9 个 base name 均有四种文件：

```text
output/ncu_gate_policy_chain_warmup_full
output/ncu_gate_policy_chain_prepare_full
output/ncu_gate_policy_chain_main_full
output/ncu_gate_policy_batch_warmup_full
output/ncu_gate_policy_batch_prepare_full
output/ncu_gate_policy_batch_main_full
output/ncu_gate_policy_deep_warmup_full
output/ncu_gate_policy_deep_prepare_full
output/ncu_gate_policy_deep_main_full
```

profile job 日志：

```text
output/gate_policy_chain_profile_109807.log
output/gate_policy_batch_profile_109823.log
output/gate_policy_deep_profile_109840.log
```

## 12. 正式 8-case 结果

预计分数按实验定义：

```text
ratio = t_100 / time
score = min(120, 80 + 20 * ratio)
```

上一版是 commit `1656299` 的 gate-CP policy，日志 `output/lab3_108960.log`。本版日志
`output/lab3_109719.log`。speedup 是 `previous/current`。

| case | auto p | previous ms | current ms | speedup | t100/current | estimated score |
|---|---:|---:|---:|---:|---:|---:|
| short_tail_state | off | 0.086944 | 0.089216 | 0.9745x | 3.8788 | 120.00 |
| chain_equal | 4 | 0.306416 | 0.300064 | 1.0212x | 1.6590 | 113.18 |
| parallel_equal | off | 0.278000 | 0.278608 | 0.9978x | 1.8351 | 116.70 |
| parallel_gva | off | 0.254448 | 0.255248 | 0.9969x | 1.9258 | 118.52 |
| long_low_gva | 8 | 1.301936 | 1.295616 | 1.0049x | 1.4347 | 108.69 |
| batch_split_gva | 2 | 1.342976 | 1.262064 | 1.0641x | 1.2140 | 104.28 |
| wide_gva_state | off | 2.224224 | 2.234944 | 0.9952x | 1.0858 | 101.72 |
| deep_gva_state | 2 | 2.633600 | 2.385488 | 1.1040x | 1.1868 | 103.74 |

简单平均预计分数：110.85。小于约 0.5% 的负向变化属于不同 job 的测量噪声；本轮真正
新增的收益是 batch p2 的 6.41% 和 deep p2 的 10.40%，同时 chain 从旧 p8 改为 p4 后
小幅改善。

## 13. private-shape 外推和限制

本策略有意保留以下安全边界：

- tail 仍走单链路径，因为 CP 目前要求等长静态 slice；
- p 最大为 8，避免 cp_states workspace 和 preparation grid 无界增长；
- 模型常数针对当前 14-SM H800 MIG 和当前 compiled resource；换完整 H800 或修改 kernel
  resource 后，必须把 SM 数和 blocks/SM 一起重新校准；
- gate threshold 的收益依赖实际 gate decay。最坏情况下 preparation 从序列开头完整
  重放，正确性仍成立，但计时模型会低估 prepare 成本；
- `Hq/Hv` 只是 cache reuse 二阶项，不应代替波次计算；
- 不对某个公开 B/T/H 直接返回固定 p，因此 private case 会按 owners、chain depth、
  discrete waves 和 GVA ratio 自然落入相邻 regime。

如果未来修改 fragment/shared 使 D128 达到 2 CTA/SM，当前 D128 cost model 的
`resident_slots` 和波次结果会整体改变，必须重新扫 policy；这也是把 resident capacity
作为模型输入而不是写 shape 表的原因。

# GDN prefill kernel walkthrough: full-prefix RS + predicated tail

本文描述本轮“完整 chunk 与尾部 chunk 分离执行”的最终实现。它是一份独立文档，
不覆盖 `walkthrough.md` 或 `walkthrough_v_shared_swizzle.md`，也不写入实验者自己的
`report.md`。

本轮解决的问题是：旧 dispatch 只要发现 `T % 64 != 0`，就让整个序列的所有 chunk
回退到 predicated base kernel。现在改为一次 RS kernel launch 内先用原来的优化 pipeline
处理 `floor(T/64)` 个完整 chunk，再只对最后一个不完整 chunk 执行 predicated tail。
公开 `short_tail_state(T=1025)` 因而从 17 个 fallback chunk 变为 16 个优化完整 chunk加
1 个 tail，100 次重复的 median 从 0.142096 ms 降到 0.087232 ms，即 1.6289x。

## 1. 为什么不是在每次循环开头写一个动态 if

从数学正确性看，可以在 `for chunk` 中判断 `right <= T`。uniform branch 本身的指令成本
确实很低，但这里真正需要保护的是完整块的编译形态：

1. RS 完整块循环由 `T.Pipelined` 做 PipelinePlanning。Q/K/V/A/g/beta 的 stage-0
   producer 被 lowering 扩成 ping-pong shared storage，并生成跨 iteration 的
   `cp.async commit/wait`。
2. 如果整个循环都按 `ceil(T/64)` 迭代并给每个 producer 加动态 predicate，完整 chunk
   也要携带边界地址判断；pipeline prologue、steady state 和 epilogue 都不再是原来的
   exact fast path。
3. 更直接地在 pipeline 外复用同名 shared buffer 处理 tail 也不可行：PipelinePlanning
   已将该 buffer 的物理 shape 从 `[64,N]` 扩成 `[2,64,N]`，循环外仍按二维 layout
   访问会在 LayoutInference 阶段失败。第一次实现正是因此报
   `InputShape()=[2,64,128], shape=[64,128]`。

最终方案把判断提升到 host/JIT specialization：

```text
has_tail = (T % 64 != 0)

has_tail = false:
    T.Pipelined(chunks_per_batch)       # 原完整块 kernel
    不生成任何 tail 代码或 tail shared

has_tail = true:
    T.Pipelined(T // 64)                # 只含必定完整的 prefix
    执行一次 predicated tail            # 不在每个 chunk 重复判断
```

因此用户提出的核心目标已经实现：完整 chunk 与 tail 在同一 launch 内走不同策略；
完整 chunk 没有尾块 predicate，只有一次尾部处理。静态 `has_tail=False` 还保证其余 7 个
公开整块 case 不承担额外 shared allocation。

## 2. Host dispatch 与 kernel launch

输入保持为：

```text
q, k          [B,T,Hq,128] BF16
v             [B,T,Hv,128] BF16
g, beta       [B,T,Hv] FP32
A             [B,T,Hv,64] BF16
initial_state [B,Hv,128,128] FP32, optional
output        [B,T,Hv,128] BF16
final_state   [B,Hv,128,128] FP32
```

`gdn_prefill_forward` 在 launch 前分配 output/final-state，并按 `owners=B*Hv` 选择 dv：

```text
2*owners <= 14 SM -> dv_tile=64, dv_parts=2
otherwise         -> dv_tile=128, dv_parts=1
```

这个规则现在同样适用于有 tail 的 shape。此前 `full_chunks_only` 会禁止 tail shape 使用
D=64；现在低 owner 的 private tail case 也能使用两个独立 value-column part。

RS kernel 的 grid 为：

```text
grid.x  = B * Hv * dv_parts
threads = 128 when dv_tile=64
          256 when dv_tile=128
```

CTA 映射：

```text
owner   = block // dv_parts
part    = block % dv_parts
bb      = owner // Hv
bh      = owner % Hv
bhg     = bh // (Hv/Hq)
dv_left = part * dv_tile
```

D=64 的两个 CTA 分别拥有 value columns `[0,64)` 和 `[64,128)`，没有 reduction 或
跨 CTA 同步。每个 CTA 内的 chunk 仍按时间顺序串行更新自己的 recurrent state。

有 tail 时 kernel symbol 增加 `_tail`，例如：

```text
residual_gva_dv128x1_rs_io_qkva_tail_kernel
residual_gva_dv64x2_rs_io_qkva_tail_kernel
```

这样 TileLang cache 会分别保存 full-only 与 tail specialization。

## 3. Fragment 分配：修改前与修改后

### 3.1 旧 tail fallback

修改前，只要有 tail，默认 dispatch 使用 `tilelang_residual_first`。对公开
`short_tail_state`，它是 D=128、256-thread CTA：

| fragment | shape/dtype | logical size |
| --- | --- | ---: |
| `state` | `[128,128]` FP32 | 64 KiB |
| `z` | `[64,128]` FP32 | 32 KiB |
| `out` | `[64,128]` FP32 | 32 KiB |
| `score` | `[64,64]` FP32 | 16 KiB |
| **total** | | **144 KiB/CTA** |

### 3.2 新 RS full-prefix + tail

新 tail specialization 与 RS full specialization 共用同一组 transposed fragments：

| fragment | D=64 shape/dtype | D=64 | D=128 shape/dtype | D=128 |
| --- | --- | ---: | --- | ---: |
| `state_t` | `[64,128]` FP32 | 32 KiB | `[128,128]` FP32 | 64 KiB |
| `state_operand` | `[64,128]` BF16 | 16 KiB | `[128,128]` BF16 | 32 KiB |
| `z_operand` | `[64,64]` BF16 | 8 KiB | `[128,64]` BF16 | 16 KiB |
| `z_t` | `[64,64]` FP32 | 16 KiB | `[128,64]` FP32 | 32 KiB |
| `out_t` | `[64,64]` FP32 | 16 KiB | `[128,64]` FP32 | 32 KiB |
| `score` | `[64,64]` FP32 | 16 KiB | `[64,64]` FP32 | 16 KiB |
| **total** | | **104 KiB/CTA** | | **192 KiB/CTA** |

尾块没有新增 fragment；它在完整 prefix 结束后复用同一个 `state_t/state_operand/z_t/
z_operand/out_t/score`。因此有两种有用的 before/after 口径：

| comparison | before | after | change |
| --- | ---: | ---: | ---: |
| 旧 D=128 tail fallback -> 新 D=128 RS tail | 144 KiB | 192 KiB | +48 KiB |
| D=128 RS full -> D=128 RS tail | 192 KiB | 192 KiB | 0 |
| D=64 RS full -> D=64 RS tail | 104 KiB | 104 KiB | 0 |

NCU 实测新 D=128 tail 为 250 registers/thread，新 D=64 tail 为 255 registers/thread。
旧 fallback 使用更少的 logical fragments，但计算次序、state operand staging 和 WGMMA
方向不同；本轮的收益来自让完整 prefix 使用更快的 RS kernel，而不是降低寄存器占用。

## 4. Shared memory：修改前与修改后

### 4.1 完整块 specialization 不变

full-only specialization 没有 tail shared。原有 source-level shared 合计保持：

| path | source logical dynamic shared | prior NCU dynamic shared |
| --- | ---: | ---: |
| D=64 fast | 113.504 KiB | 113.516 KiB after alignment |
| D=64 compact | 105.504 KiB | about 105.516 KiB after alignment |
| D=128 | 137.504 KiB | 137.516 KiB after alignment |

它们包括双缓冲 Q/K/V/A/g/beta，以及单缓冲 score、output、gamma、inverse gamma 和
`gamma_last`。本轮没有改变这些 full-only allocations。

### 4.2 tail specialization 的声明

为避免循环外访问被 PipelinePlanning 扩成三维的 buffer，tail specialization 声明单独
的 single-stage storage：

| tail object | D=64 | D=128 |
| --- | ---: | ---: |
| Q BF16 `[64,128]` | 16 KiB | 16 KiB |
| K BF16 `[64,128]` | 16 KiB | 16 KiB |
| V BF16 `[64,dv_tile]` | 8 KiB | 16 KiB |
| A BF16 `[64,64]` | 8 KiB | 8 KiB |
| g FP32 `[64]` | 0.25 KiB | 0.25 KiB |
| beta FP32 `[64]` | 0.25 KiB | 0.25 KiB |
| **nominal extra declarations** | **48.5 KiB** | **56.5 KiB** |

这些对象只在完整 prefix 完成后存活。TileLang storage planner 因而把一部分 tail storage
与已经死亡的 full-prefix storage 复用。最终 NCU 不是简单相加，而是：

| NCU resource | D=64 full fast | D=64 tail | D=128 full | D=128 tail |
| --- | ---: | ---: | ---: | ---: |
| dynamic shared/block | 113.516 KiB | **153.516 KiB** | 137.516 KiB | **177.516 KiB** |
| actual increase | | **+40 KiB** | | **+40 KiB** |
| registers/thread | 255 | 255 | 246 | 250 |
| shared block limit | 1 | 1 | 1 | 1 |
| register block limit | 2 | 2 | 1 | 1 |
| theoretical occupancy | 6.25% | 6.25% | 12.50% | 12.50% |

D=128 tail NCU 的十进制显示是 181.776 Kbyte dynamic + 1.024 Kbyte driver；D=64 是
157.200 Kbyte dynamic + 1.024 Kbyte driver。两条路径的 resident blocks/SM 都没有下降。

相对旧 D=128 no-prefetch fallback，旧 source-level shared 为 112.754 KiB；新 tail 的
shared 明显更大。这个资源成本没有降低 occupancy，因为旧、新都已经是 1 block/SM，
却换来了 16 个完整 chunk 的 RS/ping-pong fast path。

## 5. 从 launch 到完整 prefix pipeline

### 5.1 launch 前

测试输入在 `make_inputs(..., device="cuda")` 以及 preprocessing 阶段已经成为 CUDA
Tensor。CPU memory -> HBM 的传输发生在这些 tensor 创建/拷贝时，不在本 kernel launch
内部。launch 时只传 device pointer、shape 标量和 output TMA descriptor。

PyTorch 在 host wrapper 中先取得 output/final-state 的 device storage。第一次遇到某个
`(H,Hg,dv,use_initial_state,has_tail)` specialization 时 TileLang JIT 编译；后续调用复用
cache。JIT 时间不进入 benchmark median。

### 5.2 CTA prologue

每个 CTA：

1. 计算 `(bb,bh,bhg,dv_left)`。
2. 分配前述 fragments/shared storage 并应用 Q/K/V/A/score/V 的 swizzled layouts。
3. 清零 FP32 `state_t`。
4. 若有 initial state，从 HBM 读取一次 `[128,dv_tile]` FP32 slice，并按 RS WGMMA
   需要转置分布成 `state_t[dv_tile,128]`。state 在 chunk 间只驻留 registers，不会每个
   chunk 回写 HBM。

### 5.3 input prefetch

令 `F=T//64`。tail specialization 的 pipeline 只迭代 F 个必定完整的 chunk。
Q/K/V/A/g/beta 都是 stage-0 producer，lowering 生成两套 ping-pong stage：

```text
prologue: cp.async chunk 0 -> shared stage 0; commit
steady:   cp.async chunk i+1 -> shared stage (i+1)&1
          compute chunk i    <- shared stage i&1
epilogue: wait and compute final full chunk
```

每 CTA、每完整 chunk 的 input traffic：

| input | D=64 | D=128 | transfer form |
| --- | ---: | ---: | --- |
| Q | 16 KiB | 16 KiB | 16-byte cp.async |
| K | 16 KiB | 16 KiB | 16-byte cp.async |
| V slice | 8 KiB | 16 KiB | 16-byte cp.async |
| A | 8 KiB | 8 KiB | 16-byte cp.async |
| g + beta | 0.5 KiB | 0.5 KiB | 4-byte cp.async |
| **total** | **48.5 KiB** | **56.5 KiB** | |

`cp.async` 发出 global request，地址先查询 L2，miss 才访问 HBM；代码没有一条“强制从
HBM 搬到 L2”的显式指令。copy 的返回值直接进入 shared。下一 chunk 的 request 可以与
当前 chunk 的独立计算重叠。consumer 前的 `cp_async_wait<1>` 和 CTA barrier 才保证当前
stage 可读。

D=64 一个 logical owner 有两个 CTA，所以 Q/K/A/g/beta 会按 part 重复读取，两个 V
slice 合计仍覆盖完整 128 columns。

## 6. 每个完整 chunk 的计算、fragment 与等待点

RS fragment 使用 value-major 转置方向。六个 WGMMA group 是：

```text
G0: Z0^T      = S^T @ K^T
G1: O0^T      = S^T @ Q^T
G2: Z1^T      = R^T @ A^T
G3: score     = Q @ K^T
G4: out^T    += Zhat^T @ score^T
G5: state^T  += Zhat^T @ K
```

单个完整 chunk 的严格顺序：

| step | operation | storage / first dependency |
| ---: | --- | --- |
| 1 | FP32 `state_t` 转 BF16 | `state_operand` fragment |
| 2 | 提交 G0 `S^T@K^T` | async WGMMA -> `z_t` |
| 3 | 提交 G1 `S^T@Q^T` | async WGMMA -> `out_t` |
| 4 | `gamma=exp2(g*log2(e))` | `gamma_shared`，与 G0/G1 重叠 |
| 5 | `inv_gamma=1/gamma`，发布 `gamma_last` | shared arrays/scalar |
| 6 | `state_t *= gamma_last` | 不依赖 G0/G1 result |
| 7 | `warpgroup_wait<1>` | 只在首次使用 G0 前 retire G0，G1 仍可在途 |
| 8 | `R^T=beta*(V-gamma*Z0)` | 覆盖 `z_t`，读取 swizzled V shared |
| 9 | FP32 residual -> BF16 | `z_operand` fragment |
| 10 | 提交 G2 `R^T@A^T` | async WGMMA -> `z_t` |
| 11 | 提交 G3 `Q@K^T` | async WGMMA -> `score` |
| 12 | `warpgroup_wait<1>` | 首次读 G2 前等待；G3 保持 outstanding |
| 13 | `Zhat^T=Z1^T*gamma_last/gamma` | `z_t`，再写 `z_operand` |
| 14 | `warpgroup_wait<0>` | score 首次使用前才 retire G3 |
| 15 | `out_t *= scale*gamma` | G1 result 的首次 elementwise use |
| 16 | causal mask/gate score | FP32 `score` fragment |
| 17 | score FP32 -> BF16 shared | `score_shared`，G4 operand |
| 18 | 提交 G4 `Zhat^T@score^T` | 累加 `out_t` |
| 19 | 提交 G5 `Zhat^T@K` | 累加 `state_t` |
| 20 | `warpgroup_wait<1>` | older G4 完成；只读 K 的 G5 可继续在途 |
| 21 | `out_t` FP32 -> BF16 output shared | score storage 已不再被消费 |
| 22 | TMA shared -> global output | 8 KiB/CTA D64，16 KiB/CTA D128 |
| 23 | `warpgroup_wait<0>` | 下一 chunk 使用 updated state 前完成 G5 |

这满足之前约定的 codegen 检查目标：GEMM 后没有立刻无条件等待，而是插入独立 gamma、
state decay、另一个 GEMM 或 rescale 工作；只在 result 的首次 consumer 之前等待。

## 7. 单个 tail 的执行

完整 prefix 的最后一个 `warpgroup_wait<0>` 后，`state_t` 已是进入 tail 的正确 state。
此时执行一次 tail，令：

```text
left  = F*64
R     = T-left, 1 <= R <= 63
```

### 7.1 tail global -> shared

尾部只执行一次，当前实现不用 cp.async/ping-pong prefetch，而是 cooperative predicated
load：

- Q/K/V 对 `token<R` 从 global 读，否则 shared 写 0；
- A 对 `row<R` 读取完整 64 columns，否则整行写 0；
- g/beta 对 `token<R` 读取，否则写 0；
- `g=0` 使无效 token 的 gamma/inverse 都为 1，`beta=0` 使 residual 为 0。

这些 load 发生在 full pipeline 结束之后，不与最后一个 full chunk 重叠。它避免再增加
一套 tail ping-pong stage，也避免提前覆盖 full-prefix 尚在使用的 shared。未来可以研究
把 tail load 提前到最后一个 full chunk 计算期间，但必须证明独立 storage 的额外 lifetime
和 synchronization 成本低于被隐藏的 latency。

### 7.2 tail math

Tail 复用与完整 chunk 完全相同的 G0--G5、fragments 和 wait 顺序。差异只有：

1. `gamma_last=gamma[R-1]`，不是固定 `gamma[63]`。
2. score 保留 `row>=col && row<R`，invalid rows 清零；invalid K rows已经为 0。
3. `Zhat` 的 invalid token 由 `beta=0` 变为 0，因此 A 的多余 columns和 state update
   都不会产生贡献。
4. output 只在 `token<R` 时直接从 `out_t` 写 global。因为不是完整 64-row tile，tail
   不使用 output shared + TMA store。

Tail 的 G4/G5 后仍是 `wait<1>`：首次读取 out_t 时只等待 G4；direct output store 完成后
`wait<0>`，再把最终 `state_t` 写回 global final_state。整个 kernel 只有一次 final-state
store，也没有 full/tail 之间的中间 state HBM round trip 或第二次 kernel launch。

## 8. 生成 CUDA 检查

D=128 tail CUDA 保存在：

- `output/tail_rs_dv128_generated.log`

已确认：

- prologue 只在 `63 < num_tokens` 时发出第一个 full-stage cp.async；
- steady loop 只覆盖 `num_tokens >> 6` 个完整 chunks；
- full prefix 是 `cp_async_commit/wait` ping-pong，不带 tail row predicate；
- pipeline epilogue写最后一个完整 chunk 后才进入 tail；
- tail predicate 使用 `num_tokens & 63`；
- tail 的 q/k/v/A/g/beta shared physical ranges 与 full 结束后的死 storage 合并；
- G0--G5 的 commit 和 `wait<1>, wait<1>, wait<0>, wait<1>, wait<0>` 边界符合第 6、7 节；
- final tail output 是逐元素 predicated global store，final state 只写一次。

当前 full-only CUDA 保存在：

- `output/tail_split_full_dv128_generated.log`

它与上一版 `output/d128_vswizzle_generated.log` 逐行 diff。将内部 loop-bound 别名
`pipelined_chunks` 归一化为 `chunks_per_batch` 后，kernel body 完全相同；唯一剩余文本
差异是两个动态标量参数在函数声明中的顺序。所有 copy predicate、shared offset、WGMMA、
wait 和 TMA store 指令均未改变。

## 9. 正确性与边界覆盖

正式 8 case 全部 PASS。额外 synthetic 矩阵覆盖：

| case | shape property | selected path | result |
| --- | --- | --- | --- |
| `tail_only_dv64_1` | T=1, no full chunk | D64x2 RS tail | PASS |
| `tail_only_dv64_31` | T=31, no full chunk, initial state | D64x2 RS tail | PASS |
| `tail_only_dv128_63` | T=63, no full chunk, initial state | D128 RS tail | PASS |
| `tail_dv64_1` | T=65, tail=1 | D64x2 RS tail | PASS |
| `tail_dv64_31` | T=95, tail=31, mixed gate | D64x2 RS tail | PASS |
| `tail_dv64_63` | T=127, tail=63, initial state | D64x2 RS tail | PASS |
| `long_tail_dv64` | T=8191, 127 full + tail=63 | D64x2 RS tail | PASS |
| `tail_dv128_31` | T=95, tail=31, initial state | D128 RS tail | PASS |

相关日志：

- initial D=128 tail compile/correctness: `output/lab3_66150.log`
- synthetic D64/D128 boundary matrix: `output/lab3_66240.log`
- zero-full-chunk T=1/31/63 matrix: `output/lab3_66892.log`
- official 8-case benchmark: `output/lab3_66288.log`

## 10. 8-case 时间、speedup 与预计分数

统计为 10 warmups + 100 repetitions 的 median。before 是上一版
`walkthrough_v_shared_swizzle.md` 的正式结果；预计分数使用公开规则
`p=t100/t`，当 `p>1` 时 `score=min(120,100+20*(p-1))`。

| case | before ms | after ms | speedup | p=t100/t | estimated score |
| --- | ---: | ---: | ---: | ---: | ---: |
| short_tail_state | 0.142096 | **0.087232** | **1.6289x** | 3.9670 | 120.00 |
| chain_equal | 0.345648 | 0.345120 | 1.0015x | 1.4424 | 108.85 |
| parallel_equal | 0.278224 | 0.278832 | 0.9978x | 1.8336 | 116.67 |
| parallel_gva | 0.255040 | 0.253744 | 1.0051x | 1.9372 | 118.74 |
| long_low_gva | 1.760048 | 1.747600 | 1.0071x | 1.0636 | 101.27 |
| batch_split_gva | 1.328464 | 1.338800 | 0.9923x | 1.1444 | 102.89 |
| wide_gva_state | 2.220288 | 2.260384 | 0.9823x | 1.0736 | 101.47 |
| deep_gva_state | 2.595504 | 2.632928 | 0.9858x | 1.0753 | 101.51 |
| **mean** | | | | | **108.93** |

只有 `short_tail_state` 改变动态执行路径。其余 7 个使用 `has_tail=False`，生成 CUDA
body 已证明保持不变，因此 -1.8% 到 +0.7% 属于不同集群作业的频率/测量波动。最终
8 个公开 case 仍全部预计超过 100 分。

## 11. 完整 NCU profile

两份 profile 都使用：

```text
--set full
--section PmSampling_WarpStates
--import-source yes
--clock-control none
--replay-mode kernel
47 passes
```

`details.txt` 明确包含 `Section: Warp State Statistics`；raw CSV 包含完整
`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`；source CSV 将 source/SASS
与 stall samples 关联，不是只采一个精简 metric set。

### 11.1 D=128 public tail: short_tail_state

Artifacts：

- `output/ncu_tail_rs_short_full.ncu-rep`
- `output/ncu_tail_rs_short_full_details.txt`
- `output/ncu_tail_rs_short_full_raw.csv`
- `output/ncu_tail_rs_short_full_source.csv`
- `output/ncu_tail_rs_short_full.log`

Launch/resources：

| metric | value |
| --- | ---: |
| NCU kernel duration | 66.62 us |
| grid / block | 8 CTAs / 256 threads |
| registers/thread | 250 |
| dynamic shared/block | 181.776 decimal Kbyte = 177.516 KiB |
| block limit registers/shared | 1 / 1 |
| theoretical / achieved occupancy | 12.50% / 12.49% |
| local/shared spilling requests | 0 / 0 |
| warp cycles per issued instruction | 5.44 |

完整 warp-stall CPI 中主要项目：

| stall reason | CPI |
| --- | ---: |
| barrier | 1.121 |
| wait | 0.694 |
| long scoreboard | 0.594 |
| MIO throttle | 0.496 |
| not selected | 0.400 |
| short scoreboard | 0.304 |
| GMMA | 0.198 |
| math pipe throttle | 0.165 |
| dispatch | 0.143 |
| LG throttle | 0.143 |

该 profile 的 grid 只有 8 CTAs，小于 14 SM，所以 NCU 明确报告 launch underfill。tail
优化的 1.63x 收益主要来自前 16 个 chunks 不再运行旧 fallback；它没有解决 8-owner
shape 的全 GPU 并行度上限。

### 11.2 D=64 synthetic tail: tail_dv64_31

Artifacts：

- `output/ncu_tail_rs_dv64_full.ncu-rep`
- `output/ncu_tail_rs_dv64_full_details.txt`
- `output/ncu_tail_rs_dv64_full_raw.csv`
- `output/ncu_tail_rs_dv64_full_source.csv`
- `output/ncu_tail_rs_dv64_full.log`

Launch/resources：

| metric | value |
| --- | ---: |
| NCU kernel duration | 13.41 us |
| grid / block | 8 CTAs / 128 threads |
| registers/thread | 255 |
| dynamic shared/block | 157.200 decimal Kbyte = 153.516 KiB |
| block limit registers/shared | 2 / 1 |
| theoretical / achieved occupancy | 6.25% / 6.25% |
| warp cycles per issued instruction | 6.51 |
| local/shared spilling requests | 704 B / 128 B |

主要 stall CPI：

| stall reason | CPI |
| --- | ---: |
| long scoreboard | 2.391 |
| wait | 0.868 |
| no instruction | 0.735 |
| barrier | 0.706 |
| short scoreboard | 0.369 |
| LG throttle | 0.193 |
| IMC miss | 0.106 |
| drain | 0.100 |
| GMMA | 0.090 |
| MIO throttle | 0.034 |

这个 synthetic case 只有一个完整 chunk 加31-row tail，并且 grid 也只有8 CTA；高
long-scoreboard 说明这种极短、低-owner private shape 仍主要受 launch underfill 和无法被
多个 chunks 摊薄的 global dependency 影响。它不影响本轮 correctness 结论，但提示下一步
若专门优化短 tail，可以研究 tail load 与最后一个 full chunk 的 overlap。

## 12. 本轮结论

本轮不是给整个 `ceil(T/64)` 循环换一套未优化 tail 策略，而是：

```text
single launch
  -> optimized RS ping-pong full prefix
  -> exactly one predicated tail
  -> one final-state store
```

它避免额外 launch 和中间 state HBM traffic，也让 public tail case 获得 1.6289x。代价是
有 tail 的 specialization 增加 40 KiB actual dynamic shared；fragment 相对 RS full 不变，
相对旧 D=128 fallback 增加 48 KiB。资源增长没有进一步降低 occupancy。

隐藏 case 的适用范围也比旧版完整：任意 tail=1..63 都走相同策略，低 owner shape 还能按
统一 `B*Hv` 规则选择 D=64，已经通过不同 tail 长度、initial state、mixed gate、长链和
GVA head mapping 的边界测试。

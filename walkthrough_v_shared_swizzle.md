# GDN prefill kernel walkthrough: V shared-memory swizzle

本文描述 commit `80bdd85` 之后这一轮的最终实现。它是独立文档，不覆盖
`walkthrough.md`。本轮实际提交的代码改动只有一项：给 RS kernel 的
`v_shared` 加上 TileLang full-bank-swizzled layout。gamma barrier 换序和
head-major gate 都做了实验，但没有进入最终代码；后文会给出失败原因和 A/B 数据。

最终结果是 8 个公开 case 全部通过正确性，且按公开 checkpoint 估算都超过 100 分，
公开 case 均分约为 109.00。

## 1. 本轮改动及原理

### 1.1 原来的问题

RS kernel 将当前 chunk 的 V slice 从 global memory 异步复制到
`v_shared[token, dim_v]`，然后在 residual 中读取：

```text
R^T[dim_v, token]
  = beta[token] * (V_shared[token, dim_v]
                   - gamma[token] * Z0^T[dim_v, token])
```

旧版 `v_shared` 使用默认二维 shared layout。Hopper shared memory 有 32 个 bank，
bank index 可以近似理解为 `(byte_address / 4) % 32`。BF16 每个元素 2 B；当同一
warp 按 residual fragment 的线程映射同时读取 V 时，多个 lane 的地址会周期性地落到
同一个 bank。硬件必须把同一条 shared-load 指令拆成多个 wavefront 串行服务。

修改前 D=128 `wide_gva_state` full NCU 的证据：

| metric | 修改前 |
| --- | ---: |
| shared LSU load requests | 5,603,328 |
| shared LSU load bank conflicts | 6,291,456 |
| average conflict degree | 2.9-way |
| excessive load wavefront ratio | 38.16% |
| stall MIO throttle | 1.16 CPI |

这个冲突不是 HBM miss，也不是 L2 miss。V 已经在 shared memory 中，stall 发生在
SM 内部 shared-bank/L1TEX MIO 路径。

### 1.2 采用的修改

最终代码在已有 Q/K/A/score layout 旁加入：

```python
v_layout = make_full_bank_swizzled_layout(v_shared)
T.annotate_layout(
    {
        ...
        v_shared: v_layout,
        ...
    }
)
```

swizzle 保持逻辑索引 `v_shared[token, dim_v]` 不变，只改变它到 physical shared
address 的映射。低位 column bits 会与 row/group bits 组合，使一条 warp load 中原本
聚集到相同 bank 的地址分散到不同 bank。producer 和 consumer 都使用同一个 layout：

- global-to-shared `cp.async` 按 swizzled address 写 V；
- residual 的 shared load 按相同 swizzled address 读 V；
- 数学、dtype、元素数量和 global tensor layout 均不变。

它没有增加 padding，也没有增加一次 copy。与“把 V 转置到另一个 buffer”不同，
swizzle 是编译期地址变换，因此没有额外 kernel launch 或 HBM traffic。

### 1.3 profile 证明

相同 D=128 wide full profile 的结果：

| metric | 修改前 | 修改后 | 变化 |
| --- | ---: | ---: | ---: |
| NCU duration | 2.58 ms | 2.24 ms | 1.152x |
| shared LSU load conflicts | 6,291,456 | 0 | -100% |
| LDGSTS bank conflicts | 819,015 | 807,341 | 基本不变 |
| stall MIO throttle | 1.16 CPI | 0.54 CPI | -53.4% |
| stall long scoreboard | 0.54 CPI | 0.45 CPI | -16.7% |
| stall short scoreboard | 0.50 CPI | 0.38 CPI | -24.0% |
| warp cycles / issued instruction | 5.94 | 5.16 | -13.1% |
| eligible warps / scheduler | 0.45 | 0.54 | +20.0% |

目标是 ordinary shared LSU load，所以该计数降到 0。剩余约 0.81 M
`LDGSTS` conflicts 属于 global-to-shared async copy 路径，不是本次 residual V load
冲突；继续优化它需要改变 cp.async transaction/layout，而不是再次修改 consumer。

D=64 的最终 `chain_equal` full profile也报告：

| metric | 当前 D=64 |
| --- | ---: |
| shared LSU load conflicts | 0 |
| LDGSTS bank conflicts | 18,806 |
| stall MIO throttle | 0.04 CPI |

此前 D=64 compact profile（case/specialization 不同，不能直接比较 duration）有
397,074 个 shared LSU load conflicts。两条 RS specialization 上 ordinary V load
都被消除，和本轮 D=64 benchmark 的提升方向一致。

## 2. Host dispatch 和 kernel launch

`gdn_prefill_forward` 的公开输入保持不变：

```text
q, k        [B,T,Hq,128] BF16
v           [B,T,Hv,128] BF16
g, beta     [B,T,Hv] FP32
A           [B,T,Hv,64] BF16
initial S   [B,Hv,128,128] FP32, optional
```

函数先分配 BF16 output 和 FP32 final state。PyTorch allocation 只取得 device
storage，不把本 chunk 的输入从 CPU 重传到 GPU。测试输入以及计时区外得到的
`g_cumsum/A` 在 launch 前已经是 CUDA tensor。

### 2.1 specialization 选择

令 `owners = B * Hv`，H800 MIG 有 14 个 SM。full-chunk RS 路径的 auto 规则是：

```text
T % 64 != 0
    -> tail-capable base kernel

T % 64 == 0 and 2*owners <= 14
    -> D=64, dv_parts=2, 128 threads/CTA

otherwise
    -> D=128, dv_parts=1, 256 threads/CTA
```

full-chunk 路径继续使用 Q/K/V/A/g/beta 全输入 ping-pong prefetch。D=64 在显式强制
且 CTA 数较大时还能使用 score/output shared alias compact specialization；auto
低-owner D=64 保持 fast specialization。

### 2.2 grid 到数据的映射

RS launch 的 grid 是：

```text
grid.x = B * Hv * dv_parts
```

每个 CTA 计算一个 `(batch, value_head, value_slice)`：

```text
owner   = block // dv_parts
part    = block % dv_parts
bb      = owner // Hv
bh      = owner % Hv
bhg     = bh // (Hv/Hq)
dv_left = part * dv_tile
```

D=128 时一个 CTA 独占完整 value head。D=64 时两个 CTA 分别拥有 value columns
`[0,64)` 和 `[64,128)`；二者没有 reduction 或跨 CTA 通信。chunk 的时间递推仍由
同一个 CTA 串行执行，保证 state dependency 正确。

生成 CUDA 已确认：

```text
D=128: __launch_bounds__(256,1)
D=64 compact: __launch_bounds__(128,2)
```

## 3. Fragment 分配：修改前与修改后

“logical fragment size”是 TileLang 声明矩阵的逻辑容量，最终由 layout 分布到线程
寄存器；它不等于硬件 register-file allocation。

| fragment | dtype/shape D=64 | D=64 size | dtype/shape D=128 | D=128 size |
| --- | --- | ---: | --- | ---: |
| `state_t` | FP32 `[64,128]` | 32 KiB | FP32 `[128,128]` | 64 KiB |
| `state_operand` | BF16 `[64,128]` | 16 KiB | BF16 `[128,128]` | 32 KiB |
| `z_operand` | BF16 `[64,64]` | 8 KiB | BF16 `[128,64]` | 16 KiB |
| `z_t` | FP32 `[64,64]` | 16 KiB | FP32 `[128,64]` | 32 KiB |
| `out_t` | FP32 `[64,64]` | 16 KiB | FP32 `[128,64]` | 32 KiB |
| `score` | FP32 `[64,64]` | 16 KiB | FP32 `[64,64]` | 16 KiB |
| **total** | | **104 KiB/CTA** | | **192 KiB/CTA** |

本轮修改前后声明完全相同：

| path | before | after | change |
| --- | ---: | ---: | ---: |
| D=64 logical fragments | 104 KiB | 104 KiB | 0 |
| D=128 logical fragments | 192 KiB | 192 KiB | 0 |

D=128 生成 CUDA 的每线程数组是：

```text
float state_t[64]
bfloat16_t state_operand[64]
float z_t[32]
float out_t[32]
bfloat16_t z_operand[32]
float score[16]
```

D=128 NCU registers/thread 从 249 变为 246。源码没有删除 fragment，这个 3-register
差异来自 swizzled address lowering/编译器分配，不应解释为 logical fragment 缩小。
D=64 当前 full profile 仍是 255 registers/thread。

## 4. Shared memory 分配：修改前与修改后

Q/K/V/A/g/beta 都是 pipeline stage-0 producer，lowering 为它们分配两份
ping-pong buffer。gamma、inverse gamma、score 和 output 是单 buffer。

### 4.1 D=64 fast

| shared object | one stage | stages | total |
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
| **dynamic shared** | | | **113.504 KiB** |

D=64 compact 将 score/output 的 8 KiB storage 别名，因此是 105.504 KiB。

### 4.2 D=128

| shared object | one stage | stages | total |
| --- | ---: | ---: | ---: |
| Q BF16 `[64,128]` | 16 KiB | 2 | 32 KiB |
| K BF16 `[64,128]` | 16 KiB | 2 | 32 KiB |
| V BF16 `[64,128]` | 16 KiB | 2 | 32 KiB |
| A BF16 `[64,64]` | 8 KiB | 2 | 16 KiB |
| g FP32 `[64]` | 0.25 KiB | 2 | 0.5 KiB |
| beta FP32 `[64]` | 0.25 KiB | 2 | 0.5 KiB |
| score BF16 `[64,64]` | 8 KiB | 1 | 8 KiB |
| output BF16 `[64,128]` | 16 KiB | 1 | 16 KiB |
| gamma + inverse FP32 | 0.5 KiB | 1 | 0.5 KiB |
| gamma_last FP32 | 4 B | 1 | 4 B |
| **dynamic shared** | | | **137.504 KiB** |

swizzle 只改变 physical address permutation，不增加 allocation：

| path | before | after | change |
| --- | ---: | ---: | ---: |
| D=64 fast shared | 113.504 KiB | 113.504 KiB | 0 |
| D=64 compact shared | 105.504 KiB | 105.504 KiB | 0 |
| D=128 shared | 137.504 KiB | 137.504 KiB | 0 |

NCU 的十进制显示与此一致：D=64 fast 为 116.24 Kbyte/block，D=128 为
140.82 Kbyte/block。

### 4.3 Occupancy

| metric | D=128 before | D=128 after | current D=64 fast |
| --- | ---: | ---: | ---: |
| registers/thread | 249 | 246 | 255 |
| dynamic shared | 140.82 decimal kB | 140.82 decimal kB | 116.24 decimal kB |
| block limit registers | 1 | 1 | 2 |
| block limit shared | 1 | 1 | 1 |
| theoretical occupancy | 12.50% | 12.50% | 6.25% |
| achieved occupancy | 12.50% | 12.50% | 6.25% |

本轮收益不是 occupancy 增加，而是相同 resident warp 数下，每条 V shared load
需要的 wavefront 更少，从而让 warp 更早成为 eligible。

## 5. 从 launch 到 kernel 结束

### 5.1 Prologue

1. CTA 解码 `bb/bh/bhg/dv_part`，确定唯一输出和 state slice。
2. 分配第 3、4 节列出的 fragments/shared buffers，并构造 Q/K/V/A/score 的
   swizzled layouts 和 WGMMA descriptors。
3. `state_t` 清零。若有 initial state，每 CTA 从 global memory 读取自己的
   FP32 `[128,dv_tile]` slice，并以转置 fragment `[dv_tile,128]` 保存。
4. pipeline prologue 为 chunk 0 发出 Q/K/V/A/g/beta 的 `cp.async`，随后
   `cp_async_commit()`。此时只是请求已发出，consumer 还不能读取 shared stage。

initial state 只在 kernel prologue 读一次，所有 chunk 之间都驻留在 `state_t`
fragment；不会每 chunk 回写 HBM。

### 5.2 每 chunk 的 global-memory traffic

| tensor | D=64 CTA/chunk | D=128 CTA/chunk | transfer |
| --- | ---: | ---: | --- |
| Q | 16 KiB | 16 KiB | 16-byte cp.async |
| K | 16 KiB | 16 KiB | 16-byte cp.async |
| V slice | 8 KiB | 16 KiB | 16-byte cp.async |
| A | 8 KiB | 8 KiB | 16-byte cp.async |
| g + beta | 0.5 KiB | 0.5 KiB | 4-byte cp.async |
| **input total** | **48.5 KiB** | **56.5 KiB** | |
| output | 8 KiB | 16 KiB | shared-to-global TMA |

D=64 一个 logical owner 有两个 CTA，因此 Q/K/A/g/beta 会重复加载，V/output 两个
slice 合起来等于 D=128 的完整 128 columns。

`cp.async` 使用 global virtual address。软件并没有在每个 chunk 显式执行一次
“HBM 搬到 L2”：cache miss 才由 HBM 服务，命中则从 L2/L1 路径服务，最后写入
shared。CPU-to-HBM copy 在 input tensor 创建/传入 CUDA device 时已经发生，早于
这里的 kernel launch。

### 5.3 Ping-pong prefetch

steady-state chunk `c` 的生成代码顺序是：

```text
issue cp.async for chunk c+1 into stage (c+1)&1
cp_async_commit
convert current FP32 state_t to BF16 state_operand
cp_async_wait<1>
__syncthreads
consume chunk c from stage c&1
```

`wait<1>` 允许下一组 copy 保持 outstanding，只要求当前 stage 已完成。于是
chunk `c+1` 的 memory latency 与 chunk `c` 的 WGMMA/elementwise 计算重叠。
最后一个 peeled iteration 没有下一 chunk，使用 `cp_async_wait<0>`。

V swizzle 不改变这个 pipeline。它只让两个 stage 中每个 V tile 的 shared physical
layout 都使用相同 bank-friendly permutation。

### 5.4 数学顺序

为了匹配 RS WGMMA，state/value 方向在 fragment 中转置。六个 GEMM group 是：

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

详细执行和第一次依赖点：

| step | operation | location | wait / overlap |
| ---: | --- | --- | --- |
| 1 | FP32 state 转 BF16 | `state_operand` fragment | next prefetch 已发出 |
| 2 | G0 `S^T@K^T` | async WGMMA -> `z_t` | 不立即等 |
| 3 | G1 `S^T@Q^T` | async WGMMA -> `out_t` | 不立即等 |
| 4 | `gamma=exp2(g*log2(e))` | `gamma_shared` | G0/G1 仍可在途 |
| 5 | `1/gamma` | `inv_gamma_shared` | 与上一步同一 producer region |
| 6 | thread 63 写 `gamma_last` | shared scalar | 单 writer |
| 7 | `state_t *= gamma_last` | FP32 fragment | G0/G1 仍未强制全等完 |
| 8 | `warpgroup_wait<1>` | retire older G0 | G1 可保持在途 |
| 9 | residual `beta*(V-gamma*Z0)` | 覆盖 `z_t` | 第一次读取 G0 |
| 10 | residual FP32 -> BF16 | `z_operand` fragment | 无 z shared staging |
| 11 | G2 `R^T@A^T` | async -> `z_t` | |
| 12 | G3 `Q@K^T` | async -> FP32 `score` | |
| 13 | `warpgroup_wait<1>` | retire G1/G2 | G3 保持在途 |
| 14 | `Zhat=Z1*gamma_last/gamma` | `z_t -> z_operand` | 第一次读取 G2 |
| 15 | `warpgroup_wait<0>` | retire G3 | score 首次读取前 |
| 16 | `out=scale*gamma*out` | `out_t` | G1 已在 step 13 完成 |
| 17 | lower/scale/gate score | `score` | causal mask |
| 18 | score FP32 -> BF16 | `score_shared` | G4 operand |
| 19 | G4 `out += score@Zhat` | async -> `out_t` | |
| 20 | G5 `state += K^T@Zhat` | async -> `state_t` | |
| 21 | `warpgroup_wait<1>` | retire older G4 | G5 保持在途 |
| 22 | output FP32 -> BF16 | `output_shared` | score 已不再被读 |
| 23 | output TMA store | shared -> global | `tma_store_wait<0>` |
| 24 | `warpgroup_wait<0>` | retire G5 | next chunk 使用 state 前 |

score 的 gate 被等价拆分：

```text
score[row,col] *= scale * gamma[row] / gamma_last
Zhat[col]      *= gamma_last / gamma[col]
```

二者在 G4 相乘恢复 `scale*gamma[row]/gamma[col]`。同一个 Zhat 又直接用于 G5
state update。

### 5.5 生成代码中的真实等待

D=128 生成 CUDA 显示：

- input publication 是 `cp_async_wait<1>(); __syncthreads();`；
- G0/G1 commit 后，在 gamma 之前没有 `warpgroup_wait`，但有一个 compiler-inserted
  CTA rendezvous；
- gamma producer 后的 `__syncthreads` 发布 gamma/inverse/gamma_last；
- 第一次读取 G0 的 residual 前才出现 `warpgroup_wait<1>`；
- G3 的结果只在 score elementwise 前用 `warpgroup_wait<0>`；
- G4/G5 之后先 `wait<1>` 写 output，最后 `wait<0>` 才进入下一 state iteration。

steady-state 每 chunk 的 CTA barrier inventory：

| barrier | generated position | purpose |
| ---: | --- | --- |
| B0 | `cp_async_wait<1>` 之后 | 发布当前 Q/K/V/A/g/beta ping-pong stage |
| B1 | G0/G1 commit 后、gamma 前 | compiler-inserted shared-stage rendezvous；不 retire WGMMA |
| B2 | gamma/inverse/gamma_last producer 后 | 发布跨线程 gate shared values |
| B3 | score fragment 写 shared 前 | 进入 score shared staging 边界 |
| B4 | score shared 写完后、G4 前 | 发布 WGMMA 的 score operand |
| B5 | `warpgroup_wait<1>` 后、output shared 写前 | 保护 score/output 生命周期和 output staging |
| B6 | output shared 写完后、TMA 前 | 发布 TMA source |

也就是说，包含 input publication 在内，生成 CUDA 的 steady-state loop 实际有 7 个
动态 `__syncthreads()` 位置；NCU 的 `launch__barrier_count=4` 是分配到的 hardware
barrier resource 数，不是动态执行的 barrier statement 数。最后 peeled iteration
具有同构的 barrier 序列，只把 B0 前的 wait 改为 `cp_async_wait<0>`。

`__syncthreads` 是 CTA/shared publication 边界，不等价于 WGMMA completion wait。
因此生成代码满足“结果第一次被 consumer 使用之前才等待”的目标；G0/G1 与 gamma
计算在 WGMMA async timeline 上仍有重叠。

### 5.6 Epilogue

每个 chunk 的 output 已通过 TMA 写入 global output。最后一个 G5 完成后，CTA 将
`state_t[dim_v,dim_k]` 转回公开 layout
`final_state[bb,bh,dim_k,dv_left+dim_v]` 并写 global memory。kernel 返回后，
PyTorch/CUDA stream dependency 保证调用者读取 output/final state 时看到完成结果。

## 6. 两个未采用的实验

### 6.1 gamma barrier 合并

目标是把 gamma exp/inverse/gamma_last 更靠前，并尝试复用 input-publication
barrier。简单重排第一次使 TileLang PipelinePlanning 报：

```text
pipeline_stmts.size() == order_array.size() failed: 9 vs. 8
```

把 pipeline metadata 从 8 项补到 9 项后可以编译，但正确性失败：

| case | failure |
| --- | --- |
| long_low_gva | final_state 128,173 / 131,072 elements mismatch (97.8%) |
| wide_gva_state | output 396,925 / 67,108,864 elements mismatch |

原因是这里有两个不同的 publication：

1. `cp.async` 写完 g_shared 后，需要 async-proxy wait 加 CTA barrier，普通 shared
   load 才能安全读取；
2. 各线程计算 gamma/inverse、thread 63 写 gamma_last 后，又需要 CTA barrier，
   其他线程才能交叉读取这些 shared values。

把 gamma source block 放到 G0/G1 之前并不能自动把两个 publication 合成一个；
PipelinePlanning 还会按 stage/order 重排 producer。实验生成的 schedule 没有保住
所需 happens-before，结果不是浮点误差而是大面积错误。因此最终恢复原顺序。

若以后继续做，必须使用明确的 proxy fence/mbarrier 或改变 gamma 的线程分布，使
producer/consumer 不再跨线程；不能只删除一个 `__syncthreads`。

### 6.2 head-major gate

原接口的 g 是 `[B,T,Hv]`。固定 head 读取连续 64 token 时，global address stride
是 `Hv*4 B`，理论上改成 `[B,Hv,T]` 可让 gate copy 连续、减少 global sectors。

但课程接口明确规定 `g_cumsum` 必须是 `[B,T,Hv]`，且评测只收取 `student/`；
不能依赖修改计时区外的 preprocessing。实验因此在
`gdn_prefill_forward` 的计时区内做 `transpose(1,2).contiguous()`，再让专用 RS
kernel 按 head-major 读取。转置的 launch、读写和临时显存都计时。

100-repetition A/B：

| case | token-major | head-major | result |
| --- | ---: | ---: | ---: |
| wide_gva_state | 2.221216 ms | 2.228560 ms | 0.33% slower |
| deep_gva_state | 2.628288 ms | 2.587328 ms | 1.58% faster |

deep 再用 30 warmup + 300 repetitions 复测：

| layout | deep time |
| --- | ---: |
| token-major | 2.617360 ms |
| head-major | 2.601424 ms |
| improvement | 0.61% |

收益缩小且 wide 退化。为了 0.61% 增加每次调用的额外 kernel launch、临时 tensor 和
layout specialization 不值得，也不具备可推广的 shape rule，因此最终撤回。

## 7. 8-case 最终结果

before 是上一 commit `80bdd85` 的 100-repetition median；after 是本轮最终代码。
score 使用公开规则 `p=t100/t`，当 `p>1` 时
`score=min(120, 100+20*(p-1))`。

| case | before ms | after ms | speedup | p=t100/t | estimated score |
| --- | ---: | ---: | ---: | ---: | ---: |
| short_tail_state | 0.142176 | 0.142096 | 1.0006x | 2.4353 | 120.00 |
| chain_equal | 0.379104 | 0.345648 | 1.0968x | 1.4402 | 108.80 |
| parallel_equal | 0.294480 | 0.278224 | 1.0584x | 1.8376 | 116.75 |
| parallel_gva | 0.293328 | 0.255040 | 1.1501x | 1.9274 | 118.55 |
| long_low_gva | 1.977888 | 1.760048 | 1.1238x | 1.0561 | 101.12 |
| batch_split_gva | 1.527040 | 1.328464 | 1.1495x | 1.1533 | 103.07 |
| wide_gva_state | 2.534304 | 2.220288 | 1.1414x | 1.0929 | 101.86 |
| deep_gva_state | 3.015712 | 2.595504 | 1.1619x | 1.0908 | 101.82 |
| **mean** | | | | | **109.00** |

全部 8 case 正确性 PASS。原 8-case job 在第 8 个 case 编译时被作业时限终止，
deep 随后单独以相同 10 warmup / 100 repetitions 补跑并 PASS；不是用不完整作业
中的旧数值。

## 8. Profile、codegen 和 benchmark 文件

### 8.1 D=128 full profile: wide_gva_state

采集包含 `--set full --section PmSampling_WarpStates --import-source yes`，47 passes。
`details.txt` 包含完整 Warp State Statistics。

- `output/ncu_d128_vswizzle_wide_full.ncu-rep`
- `output/ncu_d128_vswizzle_wide_full_details.txt`
- `output/ncu_d128_vswizzle_wide_full_raw.csv`
- `output/ncu_d128_vswizzle_wide_full_source.csv`
- `output/ncu_d128_vswizzle_wide_full.log`

关键 final warp-state CPI：

| reason | CPI |
| --- | ---: |
| barrier | 1.06 |
| wait | 0.70 |
| MIO throttle | 0.54 |
| long scoreboard | 0.45 |
| short scoreboard | 0.38 |
| GMMA | 0.20 |

### 8.2 D=64 full profile: chain_equal

同样是 full + Warp State + source，47 passes，MIG 上使用
`--clock-control none`。

- `output/ncu_vswizzle_chain_full.ncu-rep`
- `output/ncu_vswizzle_chain_full_details.txt`
- `output/ncu_vswizzle_chain_full_raw.csv`
- `output/ncu_vswizzle_chain_full_source.csv`
- `output/ncu_vswizzle_chain_full.log`

关键 final warp-state CPI：

| reason | CPI |
| --- | ---: |
| wait | 0.99 |
| barrier | 0.56 |
| long scoreboard | 0.24 |
| short scoreboard | 0.16 |
| MIO throttle | 0.04 |
| GMMA | 0.03 |

### 8.3 Generated CUDA

- D=128: `output/d128_vswizzle_generated.log`
- D=64 compact: `output/vswizzle_dv64_generated.log`

已检查 `cp.async` ping-pong address、swizzled V address、WGMMA commit/wait 顺序、
gamma 的两个 CTA synchronization boundary、score/output staging 和 TMA store。

### 8.4 Benchmark 和 rejected experiments

- final first 7 cases: `output/vswizzle_final_8case.log`
- final deep补跑: `output/vswizzle_final_deep.log`
- initial D=128 swizzle validation: `output/d128_vswizzle_wide_long.log`
- gamma compile failure: `output/d128_vswizzle_gamma_early.log`
- gamma correctness failure: `output/d128_vswizzle_gamma_early2.log`
- head-major first A/B:
  `output/d128_vswizzle_headg_wide_deep.log` and
  `output/d128_vswizzle_tokeng_wide_deep.log`
- head-major 300-repeat deep A/B:
  `output/d128_headg_deep_repeat300.log` and
  `output/d128_tokeng_deep_repeat300.log`

## 9. 本轮结论

本轮没有降低 shared allocation 或 logical fragments，也没有提高 occupancy。它直接
修复了一个 NCU 已定位、每 chunk 重复发生的 shared-bank mapping 问题。由于 V 是
residual 的逐元素输入，每个 chunk 都必须读取，消除冲突对短链、高并行、长链和
state case 都有效，最终 speedup 为 1.0006x--1.1619x。

gamma 和 head-major 两项都遵守“先生成代码/正确性，再看性能”的筛选：前者无法在
简单 PipelinePlanning 换序下保住 publication correctness，后者在合法的计时区内
成本抵消收益。它们没有留在最终代码中，也没有被算入最终表。

当前公开 case 已全部超过 100，但 D=128 仍只有 1 block/SM，barrier/wait 是 swizzle
之后最大的 stall。下一轮若继续，应针对 explicit shared publication 或缩短 D=128
fragment lifetime 做低层实验，同时保留本轮 V layout，避免 bank conflict 回归。

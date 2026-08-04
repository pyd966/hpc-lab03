# H800 PCIe MIG 1g.10gb 架构与优化参数

本文只讨论本实验实际分配到的设备：`NVIDIA H800 PCIe MIG 1g.10gb`，而不是完整的
80 GB H800。本文的数字分成三类：

- **实例实测属性**：来自本仓库 `output/ncu_residual_long.ncu-rep` 中的 Nsight Compute
  device attributes。这是制定本实验优化策略时优先级最高的数据。
- **官方架构规格**：来自 NVIDIA Hopper Architecture、CUDA Programming Guide、Hopper
  Tuning Guide 和 MIG User Guide。
- **微基准数据**：来自直接在 H800 PCIe 上做 pointer-chase、带宽和 Tensor Core
  microbenchmark 的研究论文。NVIDIA 没有承诺固定的 cache/DRAM latency，因此这些延迟只能当作
  量级和相对关系，不能当作硬件 ABI。

## 1. 实例速查表

| 属性 | 当前实例 | 来源/含义 |
| --- | ---: | --- |
| 架构 | Hopper, compute capability 9.0 | NCU 实测 |
| MIG profile | `1g.10gb` | 1 个 SM slice + 1 个 memory slice |
| SM 数 | **14** | NCU 实测 |
| 最大 core clock | 1.755 GHz | NCU 实测；运行中会动态变化 |
| memory clock | 1.593 GHz | NCU 实测 |
| 可见 HBM | 10,468,982,784 B = **9.75 GiB** | 市场名称为 10 GB |
| HBM bus width | **640 bit** | NCU 实测 |
| 理论 HBM bandwidth | **254.88 GB/s = 237.38 GiB/s** | `2 * 1.593 GHz * 640 / 8` |
| L2 | 6,553,600 B = **6.25 MiB** | 完整 50 MiB L2 的 1/8 |
| ECC | 开启 | NCU 实测 |
| copy engine | 1 | H100/H800 `1g.10gb` profile |

MIG 的 compute 和 memory 不是按同一个分母切分。官方定义的 `1g.10gb` 是：

- SM：完整 MIG compute capacity 的约 `1/7`；本实例实际暴露 14 个 SM。
- HBM 容量、memory controller、DRAM bus 和 L2：完整设备的 `1/8`。
- 一条独立的 memory path 和一个 copy engine。

因此不能看到“10 GB”就把完整 H800 的所有指标除以 8：计算资源按 SM 实际数计算，内存资源才按
memory slice 计算。MIG 还为实例隔离 crossbar port、L2 bank、memory controller 和 DRAM bus，
其他 MIG 实例正常情况下不会抢走这里列出的 cache 容量和 DRAM bandwidth QoS。

参考：NVIDIA [MIG concepts](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/concepts.html)、
[H100 supported MIG profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
和 [MIG isolation](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/introduction.html)。

## 2. 从 grid 到计算单元

对本实验最有用的层次关系是：

```text
CUDA/TileLang grid
  -> thread block / CTA（只能常驻在一个 SM）
    -> 14 个可见 SM 中的某一个
      -> 4 个 SM processing block / sub-partition
        -> 1 个 warp scheduler + 1 个 dispatch unit
        -> 16K x 32-bit register partition
        -> CUDA arithmetic pipelines / Tensor Core / LSU / SFU
      -> 整个 SM 共享的 L1 data cache / shared memory / instruction cache / TMA
  -> 所有 14 个 SM 共享该 MIG 实例的 6.25 MiB L2 和 9.75 GiB HBM
```

一个 Hopper SM 分成 **4 个处理分区**。每个分区有自己的 warp scheduler、dispatch、L0
instruction cache 和 16K 个 32-bit registers。官方 GH100 SM 框图给出的每分区主要执行资源为：

| 每个 SM processing block | 数量 | 整个 SM | 主要用途 |
| --- | ---: | ---: | --- |
| Warp scheduler / dispatch | 1 | 4 | 每周期从 ready warp 中选择并发射指令 |
| FP32 lane | 32 | 128 | FP32 add/mul/FMA 和常规浮点 element-wise |
| INT32 lane | 16 | 64 | 地址、索引、整数运算 |
| FP64 lane | 16 | 64 | FP64 add/mul/FMA；本 lab 核心 kernel 基本不用 |
| 4th-gen Tensor Core | 1 | 4 | BF16/FP16/TF32/FP8/FP64/INT8 matrix MMA |
| LD/ST unit | 16 | 64 | shared/global/local load-store 指令 |
| SFU | 4 | 16 | `exp2`, reciprocal, rsqrt, log2, sin/cos 等近似特殊函数 |

这张表描述的是物理 pipeline，不表示一个 warp 会被永久绑定到某一组 ALU。warp 被分配给某个
scheduler 后，该 scheduler 向对应 pipeline 发射它的指令。一个 32-thread warp 的一条指令也不一定
在一个周期内完成；要同时看该指令的 issue throughput、dependency latency 和可用 warp 数。

官方 SM 框图和资源数见 NVIDIA
[Hopper Architecture In-Depth, Figure 4](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)。

### 2.1 调度和常驻上限

当前实例每个 SM 的硬件上限为：

| 资源 | 每 SM 上限 |
| --- | ---: |
| warp schedulers | 4 |
| scheduler 最大 resident warps | 16 |
| resident warps | 64 |
| resident threads | 2048 |
| resident blocks | 32 |
| threads/block | 1024 |
| registers | 65,536 个 32-bit register = 256 KiB |
| registers/block | 65,536 |
| registers/thread | 255 |
| shared memory/SM | 228 KiB |
| opt-in shared memory/block | 227 KiB（另有约 1 KiB driver reservation） |

warp scheduler 每个 issue 时刻选择一条依赖已经满足的 warp 指令。一个 SM 最多每周期从四个
scheduler 各发射一条指令。某个 warp 因 register dependency、Tensor Core completion、barrier 或
memory dependency 暂停时，需要同一 SM 上的其他 ready warps 或同一 warp 内独立指令来隐藏延迟。

occupancy 只是“active warps / 64”，不是性能本身。高 occupancy 可能隐藏延迟，但不能修复差的
coalescing、bank conflict、低 arithmetic intensity，也不能让只有 8 个 block 的 grid 使用 14 个 SM。

这些上限同时由本实例 NCU attributes 和 CUDA
[compute capability 9.0 tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
确认。

## 3. 计算吞吐、指令延迟与本 lab 的对应

### 3.1 CUDA arithmetic pipelines

CUDA Best Practices Guide 给出的 CC 9.0 理论吞吐单位是“每 SM、每周期产生的 result 数”：

| 运算 | result/clock/SM | 在 1.755 GHz、14 SM 下的峰值 |
| --- | ---: | ---: |
| FP32 add/mul/FMA | 128 | FMA 按 2 FLOP 计约 **6.29 TFLOP/s** |
| FP64 add/mul/FMA | 64 | FMA 按 2 FLOP 计约 **3.14 TFLOP/s** |
| FP16 `x2` add/mul/FMA | 256 | FMA 按 2 FLOP 计约 **12.58 TFLOP/s** |
| approximate `exp2/log2/rcp/rsqrt/sin/cos` | 16 | 约 280 G result/s |
| INT32 add/sub | 128 | 约 3.14 T result/s |
| INT32 multiply/MAD | 64 | 约 1.57 T result/s |

FMA 的一个“result”包含一次乘和一次加，所以 FLOP/s 计算乘 2。频率会因动态时钟、功耗和
Tensor Core workload 改变，上表只是上限。

本 lab 中 `beta`、gate 和 state element-wise 运算主要走 FP32/ALU；`exp2` 的理论吞吐只有 FP32
add/FMA 的 1/8。重复计算同一个 gate 的 `exp2` 很可能比普通乘加昂贵，应检查编译器是否做了 CSE，
必要时显式预计算并复用 `gamma` 或 decay。

NVIDIA 公布的是吞吐而不是 Hopper 上每种 scalar instruction 的固定 dependency latency。不能把
“128 results/clock”理解成单条 FMA 延迟为一个周期。可靠做法是在 NCU 中看 eligible warps、issue
active 和 stall reason，或针对最终 SASS 做 dependency microbenchmark。

吞吐表见 NVIDIA
[CUDA Best Practices Guide, Native Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#throughput-of-native-arithmetic-instructions)。

### 3.2 Tensor Core

每个 SM 有 4 个第四代 Tensor Cores，因此当前实例共 **56 个 Tensor Cores**。支持与本 lab 最相关的
BF16/FP16 输入和 FP32 accumulator，也支持 TF32、FP8、FP64、INT8 等。

H800 PCIe 全卡的 dense Tensor Core 峰值约为 BF16/FP16 756.5 TFLOP/s、TF32 378 TFLOP/s、
FP8 1513 TFLOP/s。按相同频率下的 14/114 SM 比例，当前 MIG slice 的粗略上限分别约为：

| Tensor Core data type | 当前实例粗略 dense 峰值 |
| --- | ---: |
| BF16/FP16 | **92.9 TFLOP/s** |
| TF32 | **46.4 TFLOP/s** |
| FP8 | **185.8 TFLOP/s** |

结构化 2:4 sparsity 的理论峰值可以再翻倍，但当前 GDN 矩阵不是这种稀疏格式。

Hopper 同时支持：

- `mma`: 一个 warp（32 threads）同步执行。
- `wgmma`: 一个 warp group（4 warps，128 threads）异步执行，并可从 shared memory 取 operand。

H800 microbenchmark 测得 `mma.m16n8k16` 的 FP16/BF16-to-FP32 completion latency 约 24 cycles；
`wgmma.m64nNk16` 的 latency 随 N 增长，N=64/128/256 时约为 32/64/128 cycles。它们是**指令**延迟，
不是整个 `T.gemm` 的延迟。TileLang 会把一个 GEMM 拆成多条指令、load、barrier 和 accumulator 操作；
最终用了哪种 shape 必须看生成的 PTX/SASS 或 NCU source page。

H800 Tensor Core 的实测分析见
[Hopper microbenchmark: Tensor Core](https://arxiv.org/html/2501.12084#S6)。

### 3.3 TMA、async copy 和 DPX

Hopper 的 Tensor Memory Accelerator 可以由少量线程发起 1D--5D global/shared transfer，硬件负责
地址生成，传输期间其他 warp 可以计算。它还能在同一个 thread-block cluster 内访问 distributed
shared memory。对本 lab 的意义是：chunk `c+1` 的 Q/K/V/A/g/beta 与 state 无关，可以在 chunk `c`
计算时预取；真正不能提前的是依赖 `S_c` 的 GEMM。

当前 residual kernel 使用 `T.copy` 和 `T.Parallel` load，没有显式 `T.async_copy`、TMA、ping-pong
buffer 或 `T.Pipelined`，所以尚未利用这部分硬件。

DPX 是 max/min + add 等 dynamic-programming 指令，本 GDN kernel 没有对应运算，不是优化重点。

参考 NVIDIA [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)。

## 4. 内存层次

```text
thread-private logical values
  -> registers / distributed fragments
  -> shared memory 或 L1 data cache（同一 SM 的 unified 256 KiB pool）
  -> 该 MIG 实例共享的 6.25 MiB L2
  -> 9.75 GiB HBM2e
```

### 4.1 Registers 与 local memory

register 是 SM 上最低延迟的 operand storage。每个 SM 只有 65,536 个 32-bit registers，按 warp/block
分配。`T.alloc_fragment` 表示逻辑 tile，TileLang 将它分布到参与计算的线程 registers 中，并不是每个
线程各保存完整矩阵。

register 使用量向上取整后决定每个 SM 能同时放多少 block。超过编译器可分配范围的值会 spill 到
`local memory`。CUDA 的 local memory 只是线程私有的地址空间，物理上仍在 device memory，并经过
L1/L2；它绝不是另一块片上 SRAM。因此要同时观察 NCU 的 registers/thread 和 local load/store。

### 4.2 L1 / shared memory

每个 Hopper SM 有 **256 KiB unified L1 data cache/shared-memory pool**。可选择的 shared carveout 为
0、8、16、32、64、100、132、164、196、228 KiB，最多 228 KiB/SM 和 227 KiB/block。shared 分得
越多，留给普通 L1 cache 的空间通常越少。

shared memory 有 32 个 bank，连续 32-bit word 映射到连续 bank；每个 bank 每周期 32 bit。因此理想
吞吐是：

```text
32 banks * 4 B = 128 B/clock/SM
```

H800 microbenchmark 实测约 127.9 B/clock/SM，L1 约 124.1 B/clock/SM。若一个 warp 的不同地址落到
同一个 bank，就会分批串行；多个线程读取同一个地址则可 broadcast。BF16 tile 的布局尤其需要让
Tensor Core operand load 满足 swizzle/layout 约束，而不是只看二维数组形状。

### 4.3 L2

本实例有 **6.25 MiB L2**，由 14 个 SM 共享，而且是 MIG 独占的 1/8 L2 slice。它缓存 global/local
访问，也是不同 block 复用只读 Q/K/A 的主要层次。完整 H800 的 microbenchmark 测得 L2 带宽约为
3942--4472 B/clock；MIG 的有效 L2 throughput 与分到的 L2 bank/crossbar 有关，不能只按 SM 数缩放，
应使用 NCU 的 `lts__throughput` 和 sector 数实测。

### 4.4 HBM2e

本实例看到 640-bit memory bus，memory clock 为 1.593 GHz。按 DDR 传输计算：

```text
bandwidth = 2 * 1.593e9 * 640 / 8 = 254.88 GB/s
```

这是连续、合并、足够并行访问下的理论值。ECC、读写混合、sector 浪费、地址不连续、block 太少、
cache 行为和指令吞吐都会降低实际值。完整 H800 streaming microbenchmark 达到理论 bandwidth 的约
91%；当前 residual `long_low_gva` 只有 8 blocks，NCU 只看到约 52.5 GB/s 和 20.6% DRAM peak，
不是因为 HBM 上限只有 52.5 GB/s，而是没有足够并行请求且 kernel 还混合了大量计算。

global memory 优化优先检查：warp 访问是否连续且自然对齐、是否重复加载、是否生成过多 32-byte
sectors、写出的中间量是否马上又读回、以及 load 能否与 Tensor Core 计算重叠。

## 5. Latency：可用的量级，不是固定常数

H800 PCIe pointer-chase microbenchmark 给出：

| memory level | H800 latency | 按 1.755 GHz 粗略换算 |
| --- | ---: | ---: |
| shared memory | 29 cycles | 16.5 ns |
| L1 hit | 32 cycles | 18.2 ns |
| L2 hit | 264.5--502 cycles | 150.7--286.0 ns |
| global/HBM | 656 cycles | 373.8 ns |

L2 是 partitioned cache。相对 SM 较近的 L2 hit 实测约 258 cycles，较远 partition 约 414 cycles；
L2 miss 还会根据 memory partition 距离出现约 556--744 cycles 的组别。

这些数字不能机械代入 kernel runtime：

- 它们是 dependent pointer chase，刻意禁止 memory-level parallelism。
- coalescing、cache state、TLB、ECC、clock 和地址到 partition 的映射都会改变结果。
- 实际 kernel 可用其他 warps、independent instructions、async copy 和 double buffering 隐藏 latency。
- MIG 使用相同 Hopper memory hierarchy，但只暴露隔离后的 partition；仍应针对本实例复测关键路径。

来源：[H800 memory latency and throughput microbenchmark](https://arxiv.org/html/2501.12084#S4)。

## 6. 当前 residual kernel 在这台实例上的映射

`long_low_gva` 的 `B=1, Hv=8`，kernel grid 是 `B * Hv = 8` blocks，每 block 256 threads = 8 warps。
NCU 实测每个 block：

| 资源 | residual kernel | 硬件后果 |
| --- | ---: | --- |
| grid | 8 blocks | 14 个 SM 中最多只有 8 个工作，6 个完全空闲 |
| threads | 256 = 8 warps | 单 block 理论 occupancy 为 `8/64=12.5%` |
| registers/thread | 200 | 每 block 51,200 registers，只能 1 block/SM |
| shared/block | 116,240 B（含 driver 部分） | 两个 block 超过 228 KiB，只能 1 block/SM |
| waves/SM | 0.57 | `8/14`，连一整波都没有 |
| achieved occupancy | 12.51% | 与 8 resident warps 完全一致 |

这意味着当前 long/low-head case 的第一瓶颈不是 HBM peak，也不是“把 occupancy 从 12.5% 调到
25%”这么简单，而是 grid 只有 8 个不可再分的 state owners。即使把每 block 资源砍半，如果 grid
仍是 8，仍然只能占 8 个 SM。

## 7. 对优化策略的直接结论

按当前 profile，优先级如下：

1. **先解决低 grid 并行度。** 尝试把一个 `(batch, value_head)` state 的工作拆给多个 block/cluster，
   或寻找可并行 scan/分块合并的状态表达式。代价是跨 block 同步、DSM/L2 通信和更复杂的 reduction。
2. **把下一 chunk 的独立数据加载与当前计算重叠。** state 有因果依赖，但 Q/K/V/A/g/beta 没有。
   `T.Pipelined`、async copy、TMA 和 ping-pong shared buffer 都值得测试。
3. **减少/复用 `exp2`。** SFU 吞吐明显低于 FP32 FMA；当前代码在二维 element loop 中反复表达相同
   token gate，不能默认编译器一定跨线程/fragment 消除它。
4. **继续保持 W/U fusion。** document 形式在 `long_low_gva` 多出约 256 MiB 理论 W/U 往返，实测
   总 DRAM traffic 比 residual 多约 271 MiB。
5. **资源压缩要结合 grid。** 对 `Hv=8`，单纯把 block residency 从 1 提到 2 不会增加 active SM；对
   `Hv=32/64` 或 batch 较大 case，它才可能通过更多 resident warps 隐藏延迟。
6. **检查 shared layout/bank conflict。** 容量只回答“能否常驻”，不回答每次 Tensor Core operand
   load 是否达到 128 B/clock/SM。
7. **利用 GVA 的 Q/K 复用。** 多个 value heads 共享同一 Q/K head；当前不同 block 会重复加载 Q/K，
   通常依赖 L2 命中。跨 head 合并或更好的 cache reuse 可能减少 traffic，但会增加 block 资源。
8. **最后才根据 roofline 微调。** 当前 residual 的 NCU 指标约为 memory throughput 33%、SM
   throughput 13%、Tensor pipe 12%，根因是低 grid 和串行 chunk chain，不能简单贴上“纯 memory
   bound”标签。

## 8. 数据来源

- 本实例实测：[Nsight Compute report](../output/ncu_residual_long.ncu-rep)
- NVIDIA [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- NVIDIA [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- NVIDIA [CUDA Compute Capability 9.0 tables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- NVIDIA [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- NVIDIA [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/)
- Luo et al., [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking](https://arxiv.org/html/2501.12084)


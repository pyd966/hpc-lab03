# Lab3 Report

## step 1

通过阅读实验文档，发现等价数学变换是一种可行的方法。但是数学变换会影响计算模式、访存模式，所以我们应该先定下来数学形式。

通过查阅 FlashQLA, FLA, FLahInfer 的实现，我发现 FlashQLA 的实现跟我们的推导很类似，只不过做了一个小修改。

我们推导中需要计算 $U,V$ 这两个中间矩阵，而 FlashQLA 直接把 $U-WS$ 作为一个整体，计算 $U-WS=AB(V-\Gamma KS)$。

两种方法谁更好？我们先看一下它们各自的计算量。

对于文档中的方法，计算 $U=ABV$ 需要 $C^2+C^2d$，计算 $W=AB\Gamma K$ 按照最优方案需要 $C+C^2+C^2d$，计算 $Z=U-WS$ 需要 $Cd^2+Cd$。这里 $d=d_k=d_v$。一共需要 $2C^2d+Cd^2+Cd+2C^2+C=2113600$ 次。

对于 FlashQLA 的方法，需要 $Cd^2+2Cd+C^2=1069056$ 次。

对比下来，显然 FlashQLA 的计算量要低不少。并且这种方法自动完成了 kernel fusion（原始方法我们会把 $U,V$ 的计算分离，因为它们没有依赖链），也减少了 memory 压力。

所以我们先采用了这个变换，实测取得了 1.53x-2.11x 的优化效果，基本符合预期。

## step 2

先不急着 profile，有一些非常显然的优化我们可以采用。

比方说我们可以把 $\gamma$ 的那个 exp 计算提在循环外算了。这个取得了 1.016-1.035x 的优化。

## step 3

是时候进行一下 profile 看一眼瓶颈在哪了。并且也是时候根据 case 的数据模式进行分析。

我们先看 case。简单来说，除了 correctness test 以外，这些 case 大概可以按照并行度大小分成两类。并行度小的有 4 个 chain，并行度大的有 64 个 chain。并行度大小会很大程度上影响我们的策略，所以这里最好写两个策略。

再看 profile 结果。

现在 shared memory 115KB/block，202 threads/block，51712 regs/block，这说明我们没办法在一个 SM 上塞两个 block。这很大程度上影响了我们的并行度。

此外，很多 case 自己的并行度（`B*Hv`）就不高。

对 long_low_gva 进行 profile，发现 mem 和 compute 的利用率都很低。这证实了我们的猜想，当前的问题是 latency bound，主要思路是用更高的并行度掩盖掉 latency。

这时我们发现一个非常重要的特性：state 与 out 的更新在 $dv$ 维度是彼此独立的。也就是说我们可以从 $dv$ 维度凭空切分出更高的并行度出来。

尝试之后，发现在当前版本下，只有 `chain_equal` case 取得了 1.16x 的收益，别的 case 反而略有下降。因为切分之后会有重复的 `Q @ K^T` 计算，所以当前收益不大是正常的。也许随着我们的进一步优化，切分 dv 会更有前景（毕竟现在理论并行度上限也不高，我们想进一步做切分 dv 还得优化 fragment 和 shared mem 占用）

## step4

我们发现 shared mem 占用并不算高，所以考虑使用 pingpong buffer 做 prefetch。
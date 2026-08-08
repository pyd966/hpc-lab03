我正在参加学校的 HPC 课程，这是其中的一个 lab。

lab 的详细要求在 docs/Lab3-GDN-Prefill/index.md 中，集群使用方法在 docs/guide/* 中，文件结构在 README.md 中。

请注意，本机为 devpod，没有 GPU 使用权限。提交代码请通过 `hpc` 命令提交到计算集群使用。集群配置为 NVIDIA H800 MIG 10G，CPU 8 cores，内存 32G。你应该通过搜索，从权威渠道了解 H800 的具体参数。

本机以及计算集群都配置有 Nsight Compute 与 Nsight Systems，可以用来 profiling。

请注意，只有 student/ 下的文件会被收取，其中 `tilelang_fwd.py` 必定会被收取。
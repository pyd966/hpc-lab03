import argparse

import torch

from student.tilelang_fwd import tilelang_residual_first_full_chunks
from student.tilelang_rs import tilelang_residual_first_full_chunks_rs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", type=int, choices=(32, 64, 128), required=True)
    parser.add_argument("--rs", action="store_true")
    args = parser.parse_args()

    if args.dv == 32:
        heads, qk_heads, dv_parts = 4, 4, 4
    elif args.dv == 64:
        heads, qk_heads, dv_parts = 16, 4, 2
    else:
        heads, qk_heads, dv_parts = 64, 16, 1

    kernel_factory = (
        tilelang_residual_first_full_chunks_rs
        if args.rs
        else tilelang_residual_first_full_chunks
    )
    kernel = kernel_factory(
        heads,
        qk_heads,
        qk_dtype=torch.bfloat16,
        v_dtype=torch.bfloat16,
        gate_dtype=torch.float32,
        accum_dtype="float32",
        use_initial_state=args.dv == 128,
        dv_tile=args.dv,
        dv_parts=dv_parts,
        prefetch_q=True,
        prefetch_k=True,
        prefetch_v=True,
        prefetch_a=True,
    )
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()

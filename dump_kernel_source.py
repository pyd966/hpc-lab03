import argparse

import torch

from student.tilelang_fwd import tilelang_residual_first_full_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dv", type=int, choices=(32, 128), required=True)
    args = parser.parse_args()

    if args.dv == 32:
        heads, qk_heads, dv_parts = 4, 4, 4
    else:
        heads, qk_heads, dv_parts = 64, 16, 1

    kernel = tilelang_residual_first_full_chunks(
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

def _build_grouped_wgmma_prelude():
    z_regs = ",".join(f"%{i}" for i in range(8))
    lines = [
        "{",
        ".reg .pred p0, p1, pact;",
        ".reg .b64 dk, ds, gs, es, is, off;",
        ".reg .u32 tid;",
        ".reg .f32 gx, ex, inv;",
        "mov.u32 tid, %%tid.x;",
        "setp.ne.u32 p0, tid, tid;",
        "setp.eq.u32 p1, tid, tid;",
        "mov.b64 dk, %8;",
        "mov.b64 ds, %9;",
        "wgmma.fence.sync.aligned;",
        "fence.proxy.async.shared::cta;",
    ]
    for ki in range(7):
        scale_d = "p0" if ki == 0 else "p1"
        lines.append(
            "wgmma.mma_async.sync.aligned."
            f"m64n16k16.f32.bf16.bf16 {{{z_regs}}}, "
            f"dk, ds, {scale_d}, 1, 1, 0, 1;"
        )
        k_step = 506 if ki == 3 else 2
        lines.append(
            f"add.u64 dk, dk, {k_step}; "
            "add.u64 ds, ds, 32;"
        )
    lines.extend(
        [
            "setp.lt.u32 pact, tid, 64;",
            "mul.wide.u32 off, tid, 4;",
            "cvta.to.shared.u64 gs, %10;",
            "cvta.to.shared.u64 es, %11;",
            "cvta.to.shared.u64 is, %12;",
            "add.u64 gs, gs, off;",
            "add.u64 es, es, off;",
            "add.u64 is, is, off;",
            "@pact ld.shared.f32 gx, [gs];",
            "@pact mul.rn.f32 gx, gx, 0f3fb8aa3b;",
            "@pact ex2.approx.f32 ex, gx;",
            "@pact rcp.approx.f32 inv, ex;",
            "@pact st.shared.f32 [es], ex;",
            "@pact st.shared.f32 [is], inv;",
            "bar.sync 0;",
            (
                "wgmma.mma_async.sync.aligned."
                f"m64n16k16.f32.bf16.bf16 {{{z_regs}}}, "
                "dk, ds, p1, 1, 1, 0, 1;"
            ),
            "wgmma.commit_group.sync.aligned;",
            "wgmma.wait_group.sync.aligned 0;",
            "}",
        ]
    )
    asm_source = "".join(f'      "{line}\\n"\n' for line in lines)
    return (
        r"""
#include <tl_templates/cuda/gemm.h>

TL_DEVICE void student_wgmma_g0_gamma(
    bfloat16_t* k, bfloat16_t* state, float* z,
    float* g, float* g_exp, float* g_inv) {
  tl::GmmaDescriptor desc_k;
  tl::GmmaDescriptor desc_state;
  tl::initialize_wgmma_descriptor<1, 1, 64>(desc_k, k);
  tl::initialize_wgmma_descriptor<3, 1024, 16>(desc_state, state);

  uint64_t dk = uint64_t(desc_k);
  uint64_t ds = uint64_t(
      desc_state + ((threadIdx.x >> 7) * 256));
  asm volatile(
"""
        + asm_source
        + r"""      : "=f"(z[0]), "=f"(z[1]), "=f"(z[2]), "=f"(z[3]),
        "=f"(z[4]), "=f"(z[5]), "=f"(z[6]), "=f"(z[7])
      : "l"(dk), "l"(ds), "l"(g), "l"(g_exp), "l"(g_inv)
      : "memory");
}
"""
    )


GROUPED_WGMMA_PRELUDE = _build_grouped_wgmma_prelude()

//===- eltwise_universal.cc ---------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- ONE eltwise kernel serving GELU, LayerNorm and softmax,
// selected per call by an opcode (tasks/0032).
// SPDX-License-Identifier: Apache-2.0
//
// WHY
// ---
// tasks/0031 measured the design-switch bill at ~115 ms of a 420 ms encode --
// 2.2-2.6 ms per switch, 49 switches. tasks/0029/0030 proved that operations
// sharing ONE static design switch for free (they are just instruction
// streams over one hw_context). The GEMM shapes already share a design via
// RTP loop bounds; the three eltwise ops must share one WORKER to join them,
// because a core runs exactly one program.
//
// The opcode arrives as a runtime parameter and the branch costs one scalar
// compare per 6,144-element tile -- noise. The three impls are the exact
// production kernels, included from their own files so the thing dispatched
// is byte-for-byte the thing validated in isolation; NPUE_ELTWISE_IMPL_ONLY
// strips their extern "C" wrappers and unused variants so the combined ELF
// stays inside the 16 KB core program memory.
//
// THE SHARED TILE: 6,144 bf16 elements, one size for all three ops:
//   GELU       6,144 = 6x1024 flat elements   (elementwise, any split works)
//   LayerNorm    16 rows x 384                (exactly the validated block)
//   softmax      96 rows x 64                 (96 % 4 = 0 for the il4 kernel)
// L1: in 2x12 KB + out 2x12 KB + params 3 KB = 51 KB, inside the 63 KB budget.

#define NPUE_ELTWISE_IMPL_ONLY
#include "gelu_poly.cc"
#include "layernorm.cc"
#include "softmax.cc"

#define ELT_OP_GELU 0
#define ELT_OP_LAYERNORM 1
#define ELT_OP_SOFTMAX 2

extern "C" {

void eltwise_universal_6144(bfloat16 *restrict input, float *restrict params,
                            bfloat16 *restrict output, int32_t op) {
  if (op == ELT_OP_GELU) {
    gelu_poly_impl(input, output, 6144);
  } else if (op == ELT_OP_LAYERNORM) {
    layernorm_il4_impl(input, params, output, 16);
  } else {
    softmax_il4_impl(input, output, 96);
  }
}

} // extern "C"

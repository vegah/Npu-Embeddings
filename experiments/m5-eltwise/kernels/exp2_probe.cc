//===- exp2_probe.cc ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- run exp2_poly on whatever is handed to it, nothing else.
// SPDX-License-Identifier: Apache-2.0
//
// softmax gave rows summing to 0 where exp2_poly(0) must be 1. Rather than
// reason about which of to_fixed / upshift / vector_cast is behaving
// differently than assumed, feed it a known ramp and read the answer back.
// Includes the same header softmax.cc does, so this measures the shipped code.

#include "aie_kernel_utils.h"
#include "exp2_poly.h"
#include <stdint.h>

using namespace aie;

extern "C" {

void exp2_probe_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  event0();
  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)input);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output);
  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone_f = aie::broadcast<float, 16>(1.0f);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < 1024; i += 16) {
    aie::vector<bfloat16, 16> xb = *it_in++;
    aie::vector<float, 16> x = aie::mul(xb, vone_bf).to_vector<float>();
    aie::vector<float, 16> e = exp2_poly(x);
    *it_out++ = aie::mul(e, vone_f).to_vector<bfloat16>();
  }
  event1();
}

} // extern "C"

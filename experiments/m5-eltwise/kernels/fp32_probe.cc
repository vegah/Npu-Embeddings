//===- fp32_probe.cc ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- how many mantissa bits does aie::vector<float> arithmetic
// actually carry on AIE2P?
// SPDX-License-Identifier: Apache-2.0
//
// tasks/0015 inferred, from a GELU polynomial diverging 3.9e-03 from its own
// numpy fp32 model, that float vector ops on this part are effectively bf16.
// That was an inference from an application kernel. This measures it directly.
//
// THE TEST
//
//     out = (1.0f + eps) - 1.0f
//
// with eps an exact power of two supplied per lane. IEEE fp32 has a 24-bit
// mantissa, so this returns eps for every eps down to 2^-23. bf16 has 8, so it
// returns 0 as soon as eps < 2^-8.
//
// Powers of two are exactly representable in bf16, so eps survives the input
// conversion and the output conversion untouched. Whatever comes back is what
// the arithmetic did, not what the format did.
//
// The subtraction is what makes it observable: without it the difference sits
// below the bf16 output grid and cannot be seen.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

void fp32_probe_impl(bfloat16 *restrict input_vector,
                     bfloat16 *restrict output_vector,
                     const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vmone = aie::broadcast<float, 16>(-1.0f);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < vector_size; i += 16) {
    aie::vector<bfloat16, 16> eb = *it_in++;
    aie::vector<float, 16> eps = aie::mul(eb, vone_bf).to_vector<float>();

    aie::vector<float, 16> t = aie::add(vone, eps);   // 1 + eps
    aie::vector<float, 16> d = aie::add(t, vmone);    // - 1

    *it_out++ = aie::mul(d, vone).to_vector<bfloat16>();
  }

  event1();
  return;
}

// Same test, but routed through the MULTIPLY path. The add path above is only
// half the question: every Horner step in a polynomial kernel is
// aie::mul(...).to_vector<float>(), which goes through an accumulator. If the
// accumulator carries fewer mantissa bits than fp32, that is where a polynomial
// loses precision -- and the add probe would never see it.
void fp32_probe_mul_impl(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  event0();

  auto it_in = aie::begin_restrict_vector<16>((bfloat16 *)input_vector);
  auto it_out = aie::begin_restrict_vector<16>((bfloat16 *)output_vector);

  const aie::vector<bfloat16, 16> vone_bf =
      aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  const aie::vector<float, 16> vone = aie::broadcast<float, 16>(1.0f);
  const aie::vector<float, 16> vmone = aie::broadcast<float, 16>(-1.0f);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < vector_size; i += 16) {
    aie::vector<bfloat16, 16> eb = *it_in++;
    aie::vector<float, 16> eps = aie::mul(eb, vone_bf).to_vector<float>();

    aie::vector<float, 16> t = aie::add(vone, eps);          // 1 + eps, exact
    // The operation under test: one multiply-by-1.0 round trip through the
    // accumulator. Mathematically the identity; anything lost here is format.
    aie::vector<float, 16> m = aie::mul(t, vone).to_vector<float>();
    aie::vector<float, 16> d = aie::add(m, vmone);           // - 1

    *it_out++ = aie::mul(d, vone).to_vector<bfloat16>();
  }

  event1();
  return;
}

extern "C" {

void fp32_probe_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  fp32_probe_impl(input, output, 1024);
}

void fp32_probe_mul_bf16(bfloat16 *restrict input, bfloat16 *restrict output) {
  fp32_probe_mul_impl(input, output, 1024);
}

} // extern "C"

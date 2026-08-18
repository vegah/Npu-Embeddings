//===- saxpy.cc ---------------------------------------------------*- C++ -*-===//
//
// Derived from mlir-aie programming_examples/getting_started/01_SAXPY/saxpy.cc
//   Copyright (C) 2025 Advanced Micro Devices, Inc.
//   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Modifications for NpuEmbeddings (Apache-2.0):
//   - vec_size is supplied by the build via -DSAXPY_VEC_SIZE instead of being
//     hardcoded, so the Python design and the kernel cannot silently disagree.
//     The upstream version hardcodes 4096 and produces wrong results if the
//     tensor size differs; making it a compile-time -D removes that footgun.
//
// PURPOSE (M1): the simplest possible kernel that still exercises everything we
// need — a vector op on the AIE core, bracketed by event0()/event1() so the
// hardware trace yields a cycle count. If this works, the toolchain works.
//
//===---------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#ifndef SAXPY_VEC_SIZE
#define SAXPY_VEC_SIZE 4096
#endif

constexpr int vec_size = SAXPY_VEC_SIZE;
constexpr int lanes = 64; // 64 bf16 lanes = one 1024-bit vector register

extern "C" {

// z = 3*x + y, bf16 in / bf16 out, fp32 accumulate.
//
// event0()/event1() bracket the work region. get_trace_summary.py pairs these
// two events to report cycles per invocation -- see docs/05-measurement/.
void saxpy(bfloat16 *restrict x, bfloat16 *restrict y, bfloat16 *restrict z) {
  event0();
  ::aie::vector<bfloat16, lanes> a_v = ::aie::broadcast<bfloat16, lanes>(3.f);
#pragma clang loop min_iteration_count(4)
  for (int i = 0; i < vec_size; i += lanes) {
    ::aie::vector<bfloat16, lanes> x_v = ::aie::load_v<lanes>(x);
    x += lanes;
    ::aie::vector<bfloat16, lanes> y_v = ::aie::load_v<lanes>(y);
    y += lanes;
    ::aie::accum<accfloat, lanes> ax_v = ::aie::mul(x_v, a_v);
    ::aie::accum<accfloat, lanes> z_v = ::aie::add(ax_v, y_v);
    ::aie::vector<bfloat16, lanes> z_v_converted = z_v.to_vector<bfloat16>();
    ::aie::store_v(z, z_v_converted);
    z += lanes;
  }
  event1();
}

// Scalar reference. Same result, no vectorisation -- selectable from Python by
// changing the ExternalFunction name. Useful as an ablation: the vector/scalar
// cycle ratio tells you how much the vector unit is actually buying.
void saxpy_scalar(bfloat16 *x, bfloat16 *y, bfloat16 *z) {
  event0();
  float a = 3.f;
  for (int i = 0; i < vec_size; ++i) {
    z[i] = a * x[i] + y[i];
  }
  event1();
}

} // extern "C"

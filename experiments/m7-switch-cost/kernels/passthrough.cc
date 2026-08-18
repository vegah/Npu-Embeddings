//===- passthrough.cc ---------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- the simplest kernel that still moves data.
// SPDX-License-Identifier: Apache-2.0
//
// This exists to be a CONTROL, not to be useful. tasks/0024 measured a hardware
// context switch at ~55 us + ~286 us per column and showed the cost does not
// depend on how different the two configurations are (the same xclbin in two
// contexts costs the same as two different designs). That does not settle
// whether it depends on how MUCH configuration there is.
//
// So this is a 1-column design with one worker and two ObjectFifos, to be
// compared against a 1-column GEMM with four workers, layout-transforming
// access patterns and many more buffer descriptors. Same width, opposite ends
// of dataflow complexity.
//
// No scalar float math anywhere -- research/notes/0001, measured at 1617x.

#include <aie_api/aie.hpp>

extern "C" {

void passthrough_bf16(bfloat16 *__restrict in, bfloat16 *__restrict out) {
  constexpr int kTile = 1024;
  constexpr int kVec = 32;
  for (int i = 0; i < kTile; i += kVec)
    aie::store_v(out + i, aie::load_v<kVec>(in + i));
}

}  // extern "C"

//===- npue_pack.hpp ----------------------------------------*- C++ -*-===//
//
// NpuEmbeddings -- build a .npue from an upstream checkpoint, without Python.
// SPDX-License-Identifier: Apache-2.0
//
// The release does not ship the model. These weights belong to
// sentence-transformers/all-MiniLM-L6-v2, and a user is better served by
// fetching them from the canonical source with a checksum to verify than by
// trusting a 66 MB blob in someone's zip file. This is what makes that
// practical: two downloads and one command, no Python, no toolchain.
//
// The result must be BYTE-IDENTICAL to tools/pack_npue.py's output. Two
// implementations of one binary layout is a real risk -- a disagreement would
// mean correctly-sized weights in the wrong order, which no size check
// catches -- so it is verified rather than assumed
// (tools/verify_pack_parity.py).

#pragma once

#include <cstdint>
#include <string>

namespace npue {

// `layout_json` and `layout_hash` describe the pre-tiled B layout and must be
// exactly what the compiled designs expect; the runtime compares the hash
// before it will dispatch, so a mismatch fails loudly instead of producing
// plausible garbage.
void prepare_model(const std::string &safetensors, const std::string &vocab,
                   const std::string &config_json_path,
                   const std::string &out, const std::string &source_sha,
                   const std::string &layout_json,
                   const std::string &layout_hash,
                   int64_t tile_k, int64_t tile_n, int64_t max_seq,
                   void (*log)(const std::string &) = nullptr);

}  // namespace npue

# 0022 — M7 GATE: a C++ runtime, no Python in the process

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done (**gate passed**)

## Goal

The roadmap's M7 is *"Pure C++ + XRT, no Python at runtime."* Prove the chain
end to end from C++:

```
.npue (mmap)  ->  bf16 weights  ->  XRT  ->  NPU  ->  HuggingFace oracle
```

## What was built

**`tools/export_xclbin.py`** — build-time. IRON has no C++ frontend
(`docs/00-overview`, ground rule 3), so designs are compiled once here and the
two artifacts XRT needs are copied out: `final.xclbin` and `insts.bin`.
Everything else in the JIT cache is intermediate and the runtime never sees it.

**`tools/export_validation.py`** — build-time. Dumps the check vectors as raw
fp32. Writing a second safetensors parser in C++ to verify one number would be
the wrong trade; these are check data, and nothing in the inference path reads
them.

**`runtime/`** — the runtime proper:

| | |
|---|---|
| `npue.{hpp,cpp}` | `.npue` reader. `CreateFileMappingW`/`MapViewOfFile`, and a hand-written scanner for the JSON directory — not a general parser, and it says so. The directory is written by our own packer with known shape, and a JSON dependency in a runtime whose selling point is a lean native binary is the same trade `docs/00-overview` rejects for the tokenizer. |
| `npu_device.{hpp,cpp}` | XRT dispatch. One `Design` owns one xclbin, its instruction stream and its buffers, loaded once and kept — F1's resident-xclbin prescription, and [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s 150 µs makes anything that reloads per call a non-starter. |
| `main.cpp` | The gate: layer 0 QKV, weights from the `.npue`, compared against the oracle. |

**The weights are never transformed.** The designs are exported with
`pretiled=True` so the bytes mapped out of the `.npue` go to a DMA descriptor
exactly as they sit on disk. That is what M4 built the format for, and
[0007](../0007-m5-pretiled-gemm-on-npu/TASK.md) measured pre-tiling as a
throughput wash — so it costs nothing and buys a runtime that does no work at
load.

## Commands

```powershell
# build time (Python allowed)
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python tools\export_xclbin.py
& "C:\Users\vegar\.conda\envs\iron\python.exe" tools\export_validation.py

# the runtime
cd runtime
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
$env:PATH = "C:\Xilinx\XRT\bin;" + $env:PATH
.\build\npuembed.exe ..
```

## Result

```
NpuEmbeddings C++ runtime
  model    ../models/all-MiniLM-L6-v2.npue
           77 tensors, 68.77 MB of weights, hidden 384, 6 layers
           checkpoint 53aa51172d142c89
  design   qkv  [256, 384] x [384, 1152]

  rel. Frobenius vs HF-derived golden        1.507e-03
  max abs difference                         1.722e-02

PASS -- tolerance 5e-03, no Python in this process
```

**1.507e-03 is the number [0011](../0011-m5-first-op-validated/TASK.md) got from
Python — to every digit printed.** The C++ runtime is not merely correct, it is
byte-identical to the path already validated against HuggingFace. That is the
strongest available evidence that nothing was quietly reinterpreted on the way
across: same weights, same layout, same kernel, same answer.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `fatal error C1083: Cannot open include file: 'boost/any.hpp'` | `xrt/detail/any.h` picks `std::any` when `__cplusplus >= 201703L` and Boost otherwise. **MSVC reports `199711L` unless `/Zc:__cplusplus` is passed — even under `-std:c++17`.** It reads as a missing dependency and is a missing flag | `/Zc:__cplusplus`. Boost is wired in as a fallback so the build works either way |
| All four designs got the **same** xclbin | The exporter snapshotted cache mtimes and took what changed. Every design was already cached, so *nothing* changed, and it fell back to "newest" four times | Identify the directory by what the design provably contains — the three memref element counts |
| First C++ run: `rel_fro 1.186` | The `.npue` stores B **pre-tiled**; the exported design consumed **row-major**. Correctly-sized bytes in the wrong order | Export with `pretiled=True` |
| …and the shape match alone could not tell those apart | Both declare the same three memrefs | Also match B's innermost DMA stride: pre-tiled strides the size-`k` dimension by `n`, row-major by `N` |

The last two are the same lesson twice: **a buffer-size check catches a wrong
size, never a wrong layout.** Both times the failure was confident and quiet.

## Artifacts

- `runtime/{CMakeLists.txt,include/,src/}`, `runtime/artifacts/` (gitignored)
- `tools/export_xclbin.py`, `tools/export_validation.py`

## Next

The gate is one GEMM. A complete C++ encoder additionally needs:

1. **The remaining kernels exported** — GELU, LayerNorm, softmax are all
   compiled designs already; the exporter handles GEMMs only.
2. **The orchestration** — 6 layers of the sequence
   [0017](../0017-m6-full-encode/TASK.md) runs in Python.
3. **The WordPiece tokenizer**, ~500–700 LOC, still unwritten. `docs/04-model`
   specifies it exactly and `docs/00-overview` explains why vendoring costs more
   than writing it.
4. **Then the performance work this milestone exists for.**
   [0018](../0018-npu-vs-cpu/TASK.md) measured 8.4 ms of Python glue per
   dispatch against 150 µs of hardware; this runtime is where that goes away,
   and it should now be measured rather than assumed.

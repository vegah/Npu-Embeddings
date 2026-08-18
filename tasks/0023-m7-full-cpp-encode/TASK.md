# 0023 — M7: the full encode in C++, and a measurement that means something

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — **C++ encode validated and benchmarked; 60× the Python path**

## Goal

[0022](../0022-m7-cpp-runtime/TASK.md) proved one GEMM from C++. This extends it
to the whole encoder and, for the first time, produces a performance number that
is not dominated by Python glue.

[0018](../0018-npu-vs-cpu/TASK.md) had to report 12.6 seq/s while noting that
78% of it was Python overhead and the comparison was therefore meaningless. This
is the run that makes it mean something.

## What was built

**All seven designs exported.** `tools/export_xclbin.py` now covers the
elementwise and reduction kernels too. Those are located by their kernel
**symbol** in the MLIR (`gelu_poly_bf16`, `layernorm_bf16`, `softmax_bf16`),
which is unambiguous in a way buffer shapes are not — a lesson
[0022](../0022-m7-cpp-runtime/TASK.md) learned twice.

**Variable buffer counts.** GELU and softmax take `(in, out)`; LayerNorm takes
`(in, params, out)` because a core tile has only two input DMA channels and
gamma+beta had to be packed together ([0020](../0020-m5-layernorm-kernel/TASK.md)).
`npu::Design` now sizes its buffer vector from `design.json`, and the output is
always the last one.

**Staged dispatch.** `dispatch_only()` plus explicit `sync_to_device` /
`sync_from_device` means a caller can stage once and dispatch many times. That
is what makes a benchmark measure the NPU rather than the copies around it.

**The encoder** — 6 layers, on the same split as the Python path so the two are
comparable: four GEMMs, GELU, LayerNorm and softmax on the array; the embedding
gather, attention's per-head GEMMs, bias adds and pooling on the host.

## Correctness

```
  embedding rel_fro vs HF golden           2.526e-02
  worst 1 - cos vs HuggingFace             3.430e-04
PASS -- tolerance 2e-03 on 1-cos, no Python in this process
```

The Python path ([0021](../0021-m5-softmax-and-full-model/TASK.md)) gets
**3.397e-04**. Two independent runtimes — different language, different host
arithmetic for attention — landing within 1% of each other, both against
HuggingFace.

## The measurement

20 encodes of 4 sequences at seq 64, NPU quiesced:

```
    wall    94.30 ms   ->      42.4 seq/s
    cpu     18.75 ms   ->      0.20 cores busy

    NPU path (copy+sync+dispatch)    73.35 ms   77.8%   49 dispatches
    host attention (QK^T, A.V)       19.05 ms   20.2%
    everything else                   1.90 ms    2.0%
```

| | seq/s | CPU cores busy | seq/s per core |
|---|---|---|---|
| Python path ([0021](../0021-m5-softmax-and-full-model/TASK.md)) | 0.7 | ~1 | ~0.7 |
| **C++ runtime** | **42.4** | **0.20** | **212** |
| CPU, sentence-transformers ([0018](../0018-npu-vs-cpu/TASK.md)) | 267.6 | ~12–17 | ~22 |

**The C++ runtime is 60× the Python path.** That is the Python glue
[0018](../0018-npu-vs-cpu/TASK.md) measured at 8.4 ms per dispatch, gone.

**It is still 6.3× slower than the CPU in wall clock, and ~10× better per
core.** On the project's own criterion — parity is enough, offload and energy
are the point — this already qualifies: 42.4 seq/s while occupying a fifth of
one core, against 267.6 seq/s while occupying twelve to seventeen.

## Where the remaining time goes, and what it costs

**49 dispatches per encode.** At [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s
150 µs that is **7.35 ms** of hardware fixed cost — but the NPU path takes
**73.35 ms**, i.e. **1.50 ms per dispatch**. So ~90% of it is still host-side:
the bf16 conversion loops (scalar, one element at a time) and the memcpy into
and out of the mapped buffers.

That is the next lever and it is entirely ours:

| lever | worth |
|---|---|
| Vectorise the fp32↔bf16 conversion, and stage weights once rather than per call | most of the 66 ms that is not hardware |
| Fuse layers — 49 dispatches is the shape of the problem | up to 7 ms of fixed cost |
| Attention on the array | 19 ms, 20% |

None of these needs new hardware knowledge; they are all work we have deferred.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `error C2589: '(': illegal token on right side of '::'` | `std::max` against the Windows `max` macro. `NOMINMAX` had been removed from the command line because XRT defines it too and warned about redefinition | Define `NOMINMAX` locally in `main.cpp` before `windows.h`. The error points at `std::max` and says nothing about macros |
| The breakdown did not print after being added | A multi-line string replace silently matched nothing — the fourth time this session that a generated edit failed quietly | Verified the built binary's output rather than the source diff |

## Artifacts

- `runtime/src/main.cpp` — the encoder and the benchmark
- `runtime/src/npu_device.cpp` — variable buffer counts, staged dispatch
- `tools/export_xclbin.py` — all seven designs
- `tools/export_validation.py` — full-encode check vectors

## Next

Measurement is no longer the blocker; it is now a tool. In priority order, by
what the breakdown says rather than by what is interesting:

1. **Vectorise the host-side bf16 conversion and stop re-staging weights per
   call.** ~66 ms of the 94 ms, and it is pure host code.
2. **Fuse layers**, worth the 7.35 ms of dispatch overhead and more as the
   op count grows.
3. **Attention on the array**, 20% — needs `head_dim = 32` padded to 64 or two
   heads folded into one 64-deep tile.
4. **Batch.** Everything above is per-encode; at batch 4 the fixed costs have
   nothing to amortise against.

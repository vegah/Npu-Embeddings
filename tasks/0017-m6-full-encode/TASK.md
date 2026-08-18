# 0017 — M6 GATE: a full MiniLM encode, on the NPU, validated end to end

- **Date** 2026-08-17
- **Milestone** M6
- **Status** done (**gate passed — bankable**)

## Goal

The roadmap calls M6 *"Full encode path in Python — bankable, real value even if
C++ never happens."* Take tokens to a normalized sentence embedding, with the
NPU doing the work, and check the result against HuggingFace.

## What was built

**Not a second encoder.** The M3 reference already routes every matmul through a
swappable `self.gemm`; M6 added the same for the activation (`self.gelu_fn`). So
this supplies NPU-backed implementations of those two hooks and runs the *same*
forward pass that `check_reference.py` proves correct against HuggingFace. There
is one encoder in this project and it is the oracle.

Every weight comes out of the packed `.npue` from M4. Both sources carry the
checkpoint sha256 and the run refuses to start if they disagree.

### What runs where, and why

**NPU:**

- the four projection/FFN GEMMs per layer — **24 dispatches**, 94.7% of encoder
  FLOPs at seq 128, validated individually in
  [0012](../0012-m5-all-layer-gemms/TASK.md)
- **GELU**, via our own polynomial kernel from
  [0015](../0015-m5-gelu-polynomial/TASK.md)

**Host:**

- **embedding lookup** — a gather, never a multiply. `.npue` stores it un-tiled
  and fp32 for exactly this reason; tiling it would only hurt.
- **LayerNorm and softmax** — `docs/04-model` requires both in fp32.
  [0016](../0016-m5-fp32-probe/TASK.md) established fp32 *is* available on the
  array, so these are "not written yet", not "not possible".
- **the attention GEMMs** — per-head `[64,32]x[32,64]` and `[64,64]x[64,32]`,
  which fail the whole-array design's `M % (m*4) == 0` constraint. 5.3% of FLOPs,
  and F3 says attention is not where encoder time goes.
- bias adds and pooling — elementwise, fp32 in `.npue` by design.

The fallback is **automatic and counted**. Every GEMM the NPU cannot express is
tallied by shape and printed, so "what still runs on the host" is a measured
number in the output rather than a claim in a comment.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings

python reference\encode_npu.py                    # GEMMs + GELU on the NPU
python reference\encode_npu.py --cpu-gelu         # isolates what the GELU kernel costs
python reference\encode_npu.py --emulate-bfp16
```

## Result

```
  dispatched to the NPU:
      6 x GEMM 256x384x1152      (qkv)
      6 x GEMM 256x384x384       (attn_out)
      6 x GEMM 256x384x1536      (ffn_up)
      6 x GEMM 256x1536x384      (ffn_down)
      6 x GELU (polynomial kernel)
  fell back to the host:
    288 x GEMM 64x32x64          (attention, per-head shapes do not tile)
    288 x GEMM 64x64x32
    LayerNorm, softmax, bias adds, embedding gather, pooling
```

| configuration | worst `1 - cos` vs HF | similarity shift | embedding rel_fro |
|---|---|---|---|
| **bf16, GEMMs on NPU, GELU on host** | **1.087e-05** | 4.231e-04 | 4.555e-03 |
| **bf16, GEMMs + GELU on NPU** | **2.050e-05** | 6.718e-04 | 5.902e-03 |
| bfp16, GEMMs + GELU on NPU | 2.346e-03 | 6.993e-03 | 6.697e-02 |

**All three PASS.** A `1 - cos` of 2.05e-05 against HuggingFace is far below
anything a downstream user of an embedding model can resolve.

### Three things this confirms

**M3's simulation predicted this hardware, twice.** M3 predicted the bf16
datapath end to end at `1-cos = 1.271e-05`; hardware with GEMMs on the NPU gives
**1.087e-05**. And [0008](../0008-m5-bfp16-real-data/TASK.md)'s *recalibrated*
bfp16 model predicted `1-cos 1.773e-03` and a similarity shift of `6.492e-03`;
hardware gives **2.346e-03** and **6.993e-03** — within 1.3x and **1.08%**
respectively.

**Our GELU kernel costs 1.9x on the final cosine** — 1.087e-05 to 2.050e-05 —
and that is still 5x under tolerance. Worth the `--cpu-gelu` switch existing: it
turns "how good is the kernel" into a measured number rather than an argument.

**bfp16 is now priced end to end on real hardware.** 115x worse than bf16 on
cosine, for 5.5x throughput. The M8 question is whether a 7.0e-03 similarity
shift costs more than 0.3 MTEB points — a specific, testable question.

### Not measured: speed

This is a correctness milestone. Each dispatch costs ~150 us
([0010](../0010-m5-b-reuse-and-cost-model/TASK.md)) and this issues 30 of them
per encode with no fusion, so wall clock here would say nothing about the design
and is deliberately not reported. M7 is where that changes.

## Problems hit

Nothing new. Notably, **five distinct compiled designs dispatched 30 times in
one process, interleaved, all correct** — which is the behaviour
[0009](../0009-m5-sync-misdiagnosis/TASK.md) predicted once the host-to-device
sync bug was fixed, and a useful confirmation that the retracted "two designs
corrupt a process" rule really was our own bug.

## Artifacts

- `reference/encode_npu.py`
- `reference/goldens/encode_npu_{bf16,bf16_cpugelu,bfp16}.json`
- `reference/encoder.py` gained the `gelu_fn` hook

## Next

M6 is bankable: there is a working NPU-accelerated MiniLM encoder validated
against HuggingFace. What remains is performance and completeness, in that order:

1. **The 150 us per dispatch is the whole performance story** and this encode
   pays it 30 times. Fusing whole layers is the lever
   ([0010](../0010-m5-b-reuse-and-cost-model/TASK.md)), and it matters more now
   that the op count is real rather than hypothetical.
2. **LayerNorm and softmax kernels** — no precision obstacle remains after
   [0016](../0016-m5-fp32-probe/TASK.md); they simply have not been written.
3. **Attention on the array** needs the `head_dim = 32` handling
   (`docs/04-model`): pad to 64, or fold two heads into one 64-deep tile.
4. **M8 can start now for the accuracy half** — the bfp16-vs-bf16 decision needs
   MTEB, and both configurations produce embeddings today.

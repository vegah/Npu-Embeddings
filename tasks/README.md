# tasks/

A dated, append-only log of every unit of work done on this project.

## Why this exists

Two reasons, and the second is the demanding one:

1. **Traceability.** Knowing *why* a decision was made six weeks later, without
   re-deriving it.
2. **Replayability.** The stated goal is that **someone could throw away the code and
   rebuild the whole solution from `tasks/` alone.** That is a high bar, and it has a
   concrete consequence: every task must record the **exact commands actually run**,
   not a prose description of them. If a command isn't written down, that step is not
   reproducible and the task is incomplete.

## Layout

```
tasks/
  README.md
  0001-<slug>/
    TASK.md          <- required
    <artifacts>      <- optional: logs, traces, generated files, screenshots
```

Numbering is sequential and never reused. Slugs are short and kebab-case
(`0003-first-gemm-trace`). A task folder is never edited to hide a failure — see below.

## Failures are the valuable part

This is a learning project. A task that says *"tried X, it failed with error Y,
because Z, so we did W instead"* is worth more than one that says *"implemented X"*.

**Never delete or rewrite a failed attempt.** Append the correction. The dead ends are
the part that cannot be reconstructed from the source tree, and they are exactly what
saves time when the same wall is hit again.

## Template

```markdown
# NNNN — <Title>

- **Date** YYYY-MM-DD
- **Milestone** M<N> (from the project plan)
- **Status** in-progress | done | abandoned | superseded by NNNN

## Goal
What we are trying to achieve, in one or two sentences. If it is a gate, say what
passing looks like.

## Context
What was true before this task. Link prior tasks: `0002-slug/TASK.md`.

## What was done
Narrative, in order. Include the reasoning, not just the actions.

## Commands
```powershell
# Every command actually run, copy-pasteable, with the env assumed.
```

## Result
What happened. Include real output/numbers, not summaries of them.

## Problems hit
Each with: symptom -> cause -> fix. Include the ones we worked around rather
than solved, and say which is which.

## Artifacts
Files produced, where they live, whether they are checked in.

## Next
What this unblocks, and what the next task should be.
```

## Conventions

- **Numbers come from measurement, not memory.** Any performance figure in a TASK.md
  must be traceable to a trace file or a command output stored in the task folder.
  See `docs/05-measurement/` for what counts as a valid measurement — in particular,
  **wall-clock timing is never a valid NPU performance claim** on this project.
- **Link outward, don't duplicate.** Point at `docs/` for durable explanation and at
  `research/` for source material. `TASK.md` records *what happened on a particular
  day*; `docs/` records *what is true*.
- When a task invalidates an earlier conclusion, mark the old one
  `superseded by NNNN` and say why. Do not silently correct it.

## Index

| # | Task | Milestone | Status |
|---|---|---|---|
| [0001](0001-scaffold-and-research-index/TASK.md) | Scaffold repo, index research, write foundational docs | M0 | done |
| [0002](0002-m1-hello-npu/TASK.md) | Hello NPU — first design built, run, and traced (**gate passed**) | M1 | done |
| [0003](0003-m2-bf16-gemm/TASK.md) | bf16 GEMM on one core, traced — bfp16 flag worth **5.5×**; fp32 accumulate essential; DMA BD limit blocks `ffn_down` | M2 | done (single-core) |
| [0004](0004-m2-multicore-gemm/TASK.md) | Whole-array GEMM, traced — 99.8% core scaling but **compute is only ~7% of NPU time**; 8-col array is untraceable | M2 | done |
| [0005](0005-m3-python-reference/TASK.md) | Python reference encoder + goldens (**gate passed**, rel_fro ≤1e-6 vs HF) — and bfp16 is **6× worse on real activations** than on M2's uniform inputs | M3 | done |
| [0006](0006-m4-npue-pretiling/TASK.md) | `.npue` pre-tiled weight container (**gate passed**, round-trip bit-exact); M2's `tile_n=32` is **illegal at 8 columns**. *Its BD and performance claims are corrected by [0007](0007-m5-pretiled-gemm-on-npu/TASK.md)* | M4 | done |
| [0007](0007-m5-pretiled-gemm-on-npu/TASK.md) | Pre-tiled GEMM on hardware — `tile_n=48` **confirmed at 8 columns**, but pre-tiling is a **wash** and `ffn_down` never needed it on the whole array | M5 | done |
| [0008](0008-m5-bfp16-real-data/TASK.md) | bfp16 on **real activations, on hardware**: distribution barely matters (0.85×), refuting [0005](0005-m3-python-reference/TASK.md)'s 6.0×. *Its "two designs" trap is corrected by [0009](0009-m5-sync-misdiagnosis/TASK.md); the measurements stand* | M5 | done |
| [0009](0009-m5-sync-misdiagnosis/TASK.md) | The "NPU corruption" was **our own missing host→device sync** — `A.numpy()[:] =` never reaches the device, and only the first dispatch per process hides it. A falsifying sequence killed a theory that fitted every prior observation | M5 | done |
| [0010](0010-m5-b-reuse-and-cost-model/TASK.md) | B reuse **blocked** by two hardware limits, but a validated cost model — **t = 150 µs + traffic/33 GB/s**, ±1.4% — prices it at 1.26–1.68× and shows we are **dispatch-bound, not bandwidth-bound** | M5 | done |
| [0011](0011-m5-first-op-validated/TASK.md) | **GATE: first real encoder op on the NPU** — layer 0 QKV from `.npue` weights and golden activations, rel_fro **1.5e-03** vs the HuggingFace oracle. Confirms the `1/√32` fold landed in Q | M5 | done |
| [0012](0012-m5-all-layer-gemms/TASK.md) | **All four per-layer GEMMs validated** at both precisions. bfp16 damage is concentrated in `ffn_down` (1-cos 8.9e-03, 34-120x the others) because its input is post-GELU — heavy within-block dynamic-range tail | M5 | done |
| [0013](0013-m5-first-eltwise-kernel/TASK.md) | **First elementwise op on the array.** It runs — but IRON's built-in GELU lands at **1.33e-02**, and the three-way split blames the **LUT** (1.32e-02), not the tanh formula (2.0e-03). We write our own | M5 | done |
| [0014](0014-m5-own-gelu-kernel/TASK.md) | **Own kernel path works** (ExternalFunction + Peano), but fp32 intermediates bought only 1%. Error isolated to **`aie::tanh` itself (~1%)**. Trap: `aie::tanh<float>` silently emits an EMPTY function and the core hangs | M5 | done |
| [0015](0015-m5-gelu-polynomial/TASK.md) | **GELU without a transcendental**: `max(x,0)+poly(min(|x|,4))` exploits that `GELU(x)-max(x,0)` is even. **4.31e-03 on hardware, 3.1x better than `aie::tanh`**. *Its inference about AIE float precision is refuted by [0016](0016-m5-fp32-probe/TASK.md)* | M5 | done |
| [0016](0016-m5-fp32-probe/TASK.md) | **Refutes [0015](0015-m5-gelu-polynomial/TASK.md)**: AIE `vector<float>` carries a full **24-bit mantissa** on both add and multiply. fp32 LayerNorm and softmax stay achievable | M5 | done |
| [0017](0017-m6-full-encode/TASK.md) | **M6 GATE: full MiniLM encode on the NPU** — 24 GEMMs + GELU dispatched, **1-cos 2.05e-05** vs HuggingFace. M3's simulation predicted both the bf16 and bfp16 hardware results | M6 | done |
| [0018](0018-npu-vs-cpu/TASK.md) | **NPU vs CPU, first head-to-head.** GEMM engine **4.3-4.8x faster than CPU**, but a full encode is **21x slower** — 8.4 ms per dispatch of Python glue against 150 us of hardware | M8 | done |
| [0019](0019-offload-and-energy/TASK.md) | **Offload measured: the NPU path costs 159x less CPU** for 3.3x less wall time. CPU path occupies 11-17 threads; NPU path a third of one core. Package power not readable on this machine | M8 | done |
| [0020](0020-m5-layernorm-kernel/TASK.md) | **LayerNorm on the array** — first kernel with a reduction; passes at both sites (3.3e-03, 3.6e-03). Found: a core tile has only **2 in / 2 out DMA channels**, and the L1 budget caps this kernel at 20 rows/call | M5 | done |
| [0021](0021-m5-softmax-and-full-model/TASK.md) | **Softmax on the array, and a FULLY WORKING MODEL** — 24 GEMM + 6 GELU + 13 LayerNorm + 6 softmax all dispatched, **1-cos 3.4e-04** vs HuggingFace. Only per-head attention GEMMs remain on the host | M5/M6 | done |
| [0022](0022-m7-cpp-runtime/TASK.md) | **M7 GATE: a C++ runtime, no Python in the process** — mmap `.npue`, XRT dispatch, **rel_fro 1.507e-03, identical to the Python path**. Weights go to DMA untouched | M7 | done |
| [0023](0023-m7-full-cpp-encode/TASK.md) | **Full encode in C++, and a measurement that means something** — **42.4 seq/s at 0.20 cores**, 60x the Python path. Breakdown: 78% NPU path, 20% host attention. 1.50 ms per dispatch against 150 us of hardware | M7 | done |
| [0024](0024-m7-dispatch-cost-anatomy/TASK.md) | **What the 1.5 ms per dispatch actually is** — not host memcpy but **changing design**, at **~25 µs + 7.2 µs per lock** (89 µs to 2.4 ms). Not reconfiguration-by-difference, not residency, not eviction. Scales with descriptors, not columns — 4.1× spread at one column. Narrower therefore wins: 8 cols is 33% slower than 2. 42.4 → 52.4 seq/s | M7 | done |
| [0025](0025-m7-batching-and-crossover/TASK.md) | **Batching, and the width default as a prediction** — batch 16 gives **72.2 seq/s** (1.35x at constant width, 1.70x over 0023). Predicted 4-col crossover at M≈947, measured tie at M=1024. Two processes scale 1.46x *regardless of width* — host overlap, not array concurrency. Third fail-open cache bug, this one produced 1-cos 8.651e-02 | M7 | done |
| [0026](0026-m7-closing-on-cpu/TASK.md) | **Chasing CPU parity: 42.4 → 251.3 seq/s, and where it stops** — 6 routes measured. **Parity NOT reached** and shown unreachable: eltwise is 209 of 347 ms and a degree probe puts it at the machine's fp32 throughput limit (941 µs per Horner step, throughput not latency). Free eltwise would still only give 427 vs 710. **3.2× better per core.** Fourth fail-open, this one in the validation: NaN scored 0.000e+00 and PASSED | M7 | done |
| [0027](0027-m7-width-hypothesis/TASK.md) | **Is MiniLM the wrong shape for this machine?** — yes, measured. Elementwise work is **1 : 3h** of MACs, so its share falls as 1/h: **29.7% at h=384, 18.1% at h=768**, both within 7% of the structural prediction. Also corrects 0026 AND itself: the blocker was never arithmetic but per-core shim streams — **GELU now runs at 8 columns, 3.87x alone / 1.13x end-to-end**. Parity-h projection **withdrawn** (the CPU has the same 1:3h structure) | M7 | done |
| [0029](0029-m7-one-xclbin-probe/TASK.md) | **One xclbin, many instruction streams — step 0 confirms it** — two runtime sequences over one static design: xclbins differ only in UUIDs, a foreign insts stream reproduces all 6.3 M elements exactly, and alternation in ONE context costs alone-price (500 µs vs 497) while the two-context control pays 479 µs — 0024's model said 486. **The 49 switches per encode are removable without fusion** | M7 | done |
| [0030](0030-m7-expert-review-tests/TASK.md) | **Testing the expert review, end to end** — 7/10 claims confirmed on hardware, 1 refuted by an existing measurement, 2 deferred with a measured pricing. **251.3 → 298.5 seq/s** (eltwise at 8 cols) and **1-cos 3.397e-04 → 2.469e-04** (poly softmax, unblocked by the stack fix). Fifth fail-open closed with purge-before-build | M7 | done |
| [0031](0031-m7-eltwise-ilp/TASK.md) | **Row-interleaved LN and softmax, and the switch bill measured whole** — LN 1.54x, softmax 1.44x in isolation (the exp2 Horner had to be STEP-interleaved; sequential inlined calls gained zero), encode 296 → 305 seq/s at 1-cos 2.469e-04 unchanged. Fourth stack-trap bite — this one HANGS (timeout) instead of corrupting. And the headline: **~115 ms of the 420 ms encode is design switching (28%)**, measured design by design — a LayerNorm call is now 78% switch | M7 | done |
| [0032](0032-m7-one-xclbin-production/TASK.md) | **The one-xclbin encode: 305 → 611 seq/s, 1-cos 1.086e-05** — the spatial-split universal worker died on 16 KB program memory (softmax-il4 ELF alone is 12.2 KB); the pivot: no eltwise op earns a 2.3 ms switch, so LN, softmax and GELU moved to the host in fp32 — each move faster AND more accurate (M6's 1.09e-05 prediction landed exactly). The NPU is pure GEMM: four shapes as instruction streams over ONE xclbin (67-70 UUID bytes apart), one hw_context, **zero switches** — 24 dispatches at alone-price (3,142 µs). Wall vs CPU: 0.42× → **0.86×** | M7 | done |
| [0033](0033-m7-pipelined-lanes/TASK.md) | **Pipelined lanes: 611 → 833 seq/s — CPU WALL PARITY PASSED (1.17×)** — two encodes in two threads over the one unified design, every NPU interaction under one mutex (the array serializes anyway), each lane owning its A/C slots; lanes self-organize into anti-phase and overlap 1.49× — the two-process 1.46× reproduced in-process for free. Lanes verified BIT-IDENTICAL (memcmp over 3.1 M floats) at 1-cos 1.086e-05. Three lanes plateau: the C-readback is DRAM-bound (~340 MB/encode) — shrink the lane, don't add lanes | M7 | done |
| [0034](0034-m8-energy/TASK.md) | **Energy, measured on Windows after all — 1.94× better per sequence** — the blocker was the WRONG COUNTER SET: `\Power Meter` has no instances, `\Energy Meter` has 14 including RAPL_Package0_PKG. Units calibrated from the data (energy is picowatt-hours; two counter paths agree to 0.1%). Control experiment proves the meter sees the NPU (+9.1 W on a zero-host-work dispatch soak vs +3.9 W for its driver thread ⇒ NPU block ≈5.2 W). Differential method (two encode counts, subtract) cancels startup AND is immune to the idle drift. **44.0 J/1000 seq vs the CPU's 85.3**, three runs within 6% | M8 | done |
| [0035](0035-m8-mteb-gate/TASK.md) | **M8 GATE PASSED — MTEB cannot tell the NPU from the CPU** — five tasks (3 STS + classification + clustering), mean delta **+0.04 points**, worst single task −0.01, four of five within ±0.01. Built the bridge that made it runnable: `--encode-file` in the runtime plus an mteb 2.x encoder that tokenizes in Python and crosses to C++ as FILES (bridge self-test: 1.078e-05 vs sentence-transformers). Both sides pinned to seq 64 so the comparison is paired. **bf16+fp32-accumulate confirmed as production; bfp16 finally closed out** | M8 | done |
| [0036](0036-m8-tokenizer/TASK.md) | **The WordPiece tokenizer — TEXT IN, VECTOR OUT, no Python** — Unicode tables GENERATED from `unicodedata` (UCD 15.1.0) so they match by construction; **6,826/6,826 texts byte-identical to HuggingFace** at two lengths. The differential test caught two real bugs on its first run: I had implemented Python's final-sigma rule when the shipped model uses the FAST tokenizer (no context), and `[CLS]` in user text was being split. Vocabulary now rides inside the `.npue` (new U8 dtype) so deploying is ONE file. End to end 1-cos 6.10e-05, top-10 neighbour overlap 99.01%. Found a pre-existing bug: `pack_npue.py` has been broken since `gemm_b_layout` was factored out — the packer had not run in months | M8 | done |
| [0037](0037-m9-tiers-endpoint/TASK.md) | **Batch tiers + an OpenAI-shaped /v1/embeddings endpoint** — one xclbin now carries 16 streams (4 shapes x 4 batch tiers, all 15 identity checks 64-69 bytes), so requests are RIGHT-SIZED, not padded: a 1-text request went ~210 ms -> 15 ms, and 512 texts run at 918 seq/s. `--serve` verified against the official `openai` client (base64 included — it is the client's default). The endpoint test caught a real bug: extra lanes never inherited the tier table and silently kept the OLD flat slot contract, running the wrong GEMMs — sixth fail-open, now fails closed | M9 | done |

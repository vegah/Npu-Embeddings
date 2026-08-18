# 0004 — M2 (multi-core): whole-array bf16 GEMM, traced

- **Date** 2026-08-17
- **Milestone** M2 (completion)
- **Status** done

## Goal

Extend M2 from one core to the array: get a **traced** multi-core bf16 GEMM, measure
per-core efficiency and scaling, and find out whether single-core numbers extrapolate.

## Context

[0003](../0003-m2-bf16-gemm/TASK.md) established single-core bf16→f32 at 137.3
MACs/cycle (53.6% of peak) with bfp16 emulation, and flagged that `whole_array.py` —
the canonical multi-core design — **has no trace support at all**.

## What was done

Copied `whole_array.py` into `experiments/m2-bf16-gemm/gemm_whole_array.py` (Apache-2.0,
attributed) and added tracing. Our changes, all marked `# NPUE:` in the file:

1. `trace_config` / `trace_row` / `trace_col` / `trace_egress_col` parameters.
2. Exactly **one** worker marked `trace=1`. Every core runs the same program on a
   different tile, so one core's `event0..event1` window *is* the per-core cost;
   tracing 32 cores would only overflow the buffer.
3. `rt.enable_trace(..., egress_shim_col=...)` in the runtime sequence.
4. Explicit `set_current_device` (per [note 0002](../../research/notes/0002-iron-silent-arch-fallback.md)).
5. Dropped the AOT/CLI/taps machinery; added a measurement driver and a wall-clock
   benchmark mode.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings\experiments\m2-bf16-gemm

python gemm_whole_array.py --scaling --emulate-bfp16              # traced sweep
python gemm_whole_array.py --bench --scaling --emulate-bfp16      # wall-clock sweep
foreach ($M in @(512,1024,2048,4096)) {                           # intensity sweep
  python gemm_whole_array.py --bench --cols 8 --emulate-bfp16 -M $M -K 512 -N 512 --iters 30
}
```

## Result 1 — tracing breaks routing on most array widths

**The design compiles at every width. Adding one trace flow breaks three of four.**
Compile-only probe, bf16→f32, 512³, tile 64×64×32, bfp16 on:

| cols | cores | no trace | with trace | error with trace |
|---|---|---|---|---|
| 1 | 4 | OK | **FAIL** | `'aie.packet_rules' op packet switched source DMA0 cannot match another connect or masterset operation` |
| 2 | 8 | OK | **FAIL** at default; **OK** at `(trace_col=1, egress_shim_col=1)` | `'aie.masterset' op targets same destination South: 3 ...` |
| **4** | **16** | OK | **OK** | — |
| 8 | 32 | OK | **FAIL** | `Unable to find a legal routing` |

For cols=8 an exhaustive search over `trace_col` 0–3 × `egress_shim_col` 0–7 (32
combinations) found **nothing**. Alternative routing strategies also fail, in two
distinct ways:

| aiecc flags | cols=8 + trace |
|---|---|
| baseline | `Unable to find a legal routing` |
| `--packet-sw-objFifos` | `'aie.device' op max number of packet IDs reached` |
| `--placer=sa_placer` | `Unable to find a legal routing` |
| `--packet-sw-objFifos --placer=sa_placer` | `max number of packet IDs reached` |

Circuit-switched routing and the packet-ID space are **both** exhausted. This is a
genuine toolchain/hardware limit, not a misconfiguration.

> **Consequence for our measurement doctrine: the fully-packed 8-column array cannot
> be core-traced.** Traceable widths are 2 and 4 columns only, and the working
> `(trace_col, egress_shim_col)` pairs are recorded in `TRACE_ROUTING` in the script.

## Result 2 — per-core compute efficiency is flat and high

Traced, bf16→f32 + bfp16, 512³, tile 64×64×32:

| config | cores | avg cycles/tile | per-core MACs/cyc | % of 256 peak | array MACs/cyc |
|---|---|---|---|---|---|
| single core ([0003](../0003-m2-bf16-gemm/TASK.md)) | 1 | 954.7 | 137.3 | 53.6% | — |
| 2 cols | 8 | 923.1 | 142.0 | 55.5% | 1135.9 |
| 4 cols | 16 | 925.2 | 141.7 | 55.3% | 2266.7 |

**8 → 16 cores scales 1.998× (99.8%).** Per-core cost does not degrade — the compute
side of the array is healthy.

## Result 3 — but end-to-end scaling collapses, and compute is ~7% of the time

Wall clock, no trace (permitted for end-to-end throughput per
[docs/05-measurement](../../docs/05-measurement/README.md); **not** a kernel claim):

| cols | cores | npu time | e2e time | TFLOP/s | scaling vs 1 col |
|---|---|---|---|---|---|
| 1 | 4 | 785.9 µs | 1210.1 µs | 0.34 | 1.00× |
| 2 | 8 | 379.8 µs | 827.1 µs | 0.71 | 2.07× (103%) |
| 4 | 16 | 304.6 µs | 802.9 µs | 0.88 | 2.58× (**64%**) |
| 8 | 32 | 243.0 µs | 663.6 µs | 1.10 | 3.23× (**40%**) |

**The trace says 99.8% scaling; wall clock says 40%. Both are correct, and the gap is
the finding.**

From the traced rate, compute time at 32 cores is
`512³ / (141.7 × 32) = 29,600 cycles = 16.4 µs` at 1.808 GHz.
Measured NPU time: **243 µs**.

> **Compute is ~7% of NPU-side time at 512³. The other ~93% is data movement and
> dispatch.** Adding cores shrinks the 7% and leaves the 93% alone, which is exactly
> why wall-clock scaling collapses while per-core efficiency stays flat.

This is findings **F1** (dispatch overhead dominates) and **F2** (bandwidth-bound)
confirmed on our own silicon, not inherited from the papers.

## Result 4 — arithmetic intensity is the lever, and there is a ceiling

cols=8, K=N=512, sweeping M:

| M | npu time | TFLOP/s |
|---|---|---|
| 512 | 229.7 µs | 1.17 |
| 1024 | 307.7 µs | 1.74 |
| 2048 | 477.2 µs | 2.25 |
| 4096 | 821.7 µs | **2.61** (best 2.74) |

A linear fit gives **≈0.168 µs per M row plus ≈144 µs fixed cost per dispatch**.

Two consequences:

1. **The fixed ~144 µs is the dispatch overhead F1 warns about**, measured directly.
   TileFuse described it as "millisecond-range"; we see ~0.14 ms. A MiniLM encode at
   M=256 would be almost entirely this overhead.
2. **Even as M → ∞ the asymptote is only ~3.1 TFLOP/s**, versus a compute-bound
   ~16 TFLOP/s from the traced per-core rate. So we are **~5× off compute peak even
   with infinite work** — the limit is data movement, not dispatch alone, and not the
   vector units.

Rough bandwidth check at M=4096: A 4 MB + C 8 MB + B re-streamed 16× (8 MB) ≈ 20 MB in
821.7 µs ≈ **24 GB/s**, against the ~40–60 GB/s the NPU is thought to get. So we are not
even saturating the available bandwidth — there is headroom that better data movement
(B reuse, larger megatiles, offline pre-tiling) should recover.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| cols 1/2/8 "compile failed", message truncated | My handler printed only the first line | Widened the probe; got the real routing errors |
| cols=8 + trace unroutable | Circuit-switch and packet-ID exhaustion | **Not solved.** Documented; 8-col measured by wall clock only |
| cols=2 + trace unroutable at defaults | Trace egress collided with data flows | `(trace_col=1, egress_shim_col=1)`, found by search |
| `run_iters` returned `n/a` for all timings | Assumed `.npu.mean` in ns; actual API is `Stats.avg_us/min_us/max_us` in **microseconds** | Read `benchmark.py`, fixed extraction |

## Artifacts

`experiments/m2-bf16-gemm/artifacts/` — `trace_wa{2,4}c_*.{txt,json}`,
`scaling_bfp16_512x512x512.json`, `bench_bfp16_*.json` (one per M).

## Next

M2 is complete. The results reshape what matters next:

1. **Data movement is now the whole game.** Per-core compute is at 55% of peak and
   scales linearly; the array is starved. Priorities shift to B-reuse across row
   blocks, larger megatiles staged in L2 ([2602.06063](https://arxiv.org/abs/2602.06063)
   measured 5.9 → 13.7 TOPS purely from megatile size), and offline pre-tiling.
2. **M4 pre-tiling was already required** (the K=1536 BD limit from
   [0003](../0003-m2-bf16-gemm/TASK.md)); it is now also the main performance lever.
3. **Batch aggressively.** With ~144 µs fixed cost per dispatch and MiniLM's M=256
   sequences, a single encode is pure overhead. Batch many sentences into one launch
   and fuse whole layers — F1's prescription, now with our own numbers behind it.
4. ~~**Re-examine whether we need 8 columns at all.**~~ **Tested and answered: yes we
   need them.** At M=4096, K=N=512:

   | cols | cores | TFLOP/s |
   |---|---|---|
   | 4 (traceable) | 16 | 1.78 |
   | 8 (untraceable) | 32 | **2.56** |

   8 columns is worth **1.44×** even while data-movement bound — well short of the 2×
   ideal, but far too much to give up for measurement convenience.

   **So the production configuration is permanently untraceable**, and our measurement
   strategy must be explicitly two-track:
   - **per-core cycles and vector utilisation** → measured at **4 columns**, traced,
     which is legitimate because per-core cost is shape- and width-independent
     (137.3 → 142.0 → 141.7 MACs/cycle from 1 to 8 to 16 cores);
   - **end-to-end throughput** → measured at **8 columns** by wall clock, always
     labelled as such, with the NPU quiesced.

   This must be written into `docs/05-measurement/` as standing policy, because it is
   a permanent property of the toolchain rather than a temporary workaround.

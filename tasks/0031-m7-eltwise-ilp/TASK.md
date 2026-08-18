# 0031 — M7: row-interleaved LayerNorm and softmax, and the switch bill measured whole

- **Date** 2026-08-18
- **Milestone** M7
- **Status** done — LN 1.54×, softmax 1.44× in isolation; encode 296 → **305 seq/s**,
  `1-cos` unchanged at **2.469e-04**. And the real finding: **~115 ms of the
  420 ms encode is design switching** — 28%, measured design by design.

## Goal

The expert re-review (after [`0030`](../0030-m7-expert-review-tests/TASK.md))
diagnosed LN and softmax as **latency-bound, not throughput-bound**: softmax
measured ~3,100 cycles/row against ~500 of issued work, LN ~7,800 against
~1,000. Both process one row at a time, and everything within a row is one
dependency chain (LN: 24 dependent accumulator adds per pass; softmax:
max → sub → exp2 → sum → inv). The cure that fixed GELU in 0026 — four
independent chains interleaved — had never been applied to them.

## What was done

1. **`layernorm_il4_bf16`** (`kernels/layernorm.cc`): four rows per iteration,
   four independent accumulator chains per pass, gamma/beta loaded once per
   vector index instead of once per row. Bit-identical numerics per row.
2. **`softmax_poly_il4_bf16`** (`kernels/softmax.cc`): four rows interleaved,
   exponentials staged through a **local stack spill buffer** (16 live bf16
   vectors would exhaust the 12×512-bit register file).
3. Variant/stack plumbing in `layernorm_kernel.py`, `softmax_kernel.py`, and
   `tools/export_xclbin.py` (`--ln-variant`, `--sm-variant`, `--eltwise-only`
   to rebuild eltwise into an existing export without recompiling GEMMs).

## The two failures that were the actual lessons

**(a) Four sequential `exp2_poly()` calls interleave NOTHING.** The first il4
softmax measured 5,548 µs against the one-row kernel's 5,356 — zero gain —
because the inliner emits each `exp2_poly(g0..g3)` call as a complete 7-step
serial Horner chain and the scheduler does not interleave across them. GELU
never hit this because its macro interleaves each Horner **step** across the
four chains. Rewriting exp2 step-interleaved (`SM_EXP2_STEP`) took softmax to
**3,725 µs (1.44×)**. Lesson: *interleaving must be written at the step level
in source order; independent chains in sequence are not enough.*

**(b) The stack trap, fourth bite — and this one HANGS.** The step-interleaved
pass 2 keeps ~28 live vectors; at stack 0x2000 the design **timed out on the
array** (`ERT_CMD_STATE_TIMEOUT`) instead of corrupting. 0x4000 fixed it.
Previous bites corrupted silently (GELU 4-chain, exp2-in-softmax, gelu
epilogue); this is the first that deadlocked — presumably the overrun clobbered
fifo lock state. The exporter now sets 0x4000 for `poly_il4`.

## Measured (all wall-clock, NPU quiesced, `--probe-design` = 100 reps)

Isolated per call, batch-128 shapes, 8 columns:

| design | before µs | after µs | gain |
|---|---|---|---|
| layernorm | 1,115 | **725** | 1.54× |
| softmax (poly) | 5,356 | **3,725** | 1.44× |

Full encode (`artifacts_b128il`, `--threads 16 --bench 5`, two runs):
**307.6 / 302.3 seq/s** against baseline `artifacts_b128e8` 296.2 the same
session. `1-cos` **2.469e-04, identical** — the kernels are bit-identical per
row by construction, and validated against goldens in isolation first
(LN 3.326e-03, softmax 4.278e-03, both PASS, both exactly the pre-il4 values).

### Why the gain is small at encode level: the switch bill, now complete

`--probe-design` on every design in the production set:

| design | alone µs | switch µs | calls | switch ms/encode |
|---|---|---|---|---|
| qkv | 3,104 | 2,378 | 6 | 14.3 |
| attn_out | 1,326 | 2,499 | 6 | 15.0 |
| ffn_up | 3,953 | 2,379 | 6 | 14.3 |
| ffn_down | ~4,000 | ~2,400 | 6 | ~14.4 |
| gelu | 4,693 | 2,205 | 6 | 13.2 |
| layernorm | **725** | **2,569** | 13 | 33.4 |
| softmax | 3,725 | 2,298 | 6 | 13.8 |
| **total** | 134 ms | | 49 | **~118 ms** |

Cross-check: sum of alone-costs (134 ms) + switch bill (~115) = 249 ms ≈ the
encode's measured `dispatch + wait` of 249–250 ms. The model closes.

Two consequences:

1. **A LayerNorm call is now 78% switch.** The kernel is 725 µs; getting into
   and out of the design costs 2,569. Further eltwise kernel work is worth at
   most ~35 ms total even if all three kernels became free.
2. **The 0024-era "~60 ms of switches" is stale**: the 8-column eltwise
   designs bought compute speed with more descriptors (locks), exactly as
   note 0004's model predicts, and the per-switch price rose to 2.2–2.6 ms.
   **Switching is now 28% of the encode — the single largest line item.**

## Exact commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1
cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
# isolated golden validation (bit-identical => same numbers as 0030)
python experiments\m5-eltwise\layernorm_kernel.py --variant il4 --stack 0x2000 --cols 1
python experiments\m5-eltwise\softmax_kernel.py  --variant poly_il4 --stack 0x4000 --cols 1
# production export (eltwise only, GEMMs untouched)
Copy-Item -Recurse runtime\artifacts_b128e8 runtime\artifacts_b128il
python tools\export_xclbin.py --eltwise-only --batch 128 --elt-cols 8 --out runtime\artifacts_b128il
# validate, bench, probe
cd runtime
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 16
.\build\npuembed.exe .. --artifacts artifacts_b128il --threads 16 --bench 5
.\build\npuembed.exe .. --probe-design artifacts_b128il/layernorm   # etc.
```

## Next

The switch bill (~115 ms) is now the largest single item, and the eltwise
column-split pricing from 0030 is obsolete on two counts (il4 shrank the
penalty, the measured bill doubled the prize). The **unified one-xclbin
design** — GEMM columns + an opcode-switched eltwise worker, all ops as
instruction streams over one context — is priced at net **−55 to −115 ms**
depending on split, with every mechanism individually proven. That is task
0032.

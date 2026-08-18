# 0025 — M7: batching, and turning the width default into a prediction

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — **52.4 → 72.2 seq/s**; two cost-model predictions tested and confirmed;
  a third fail-open cache bug found, having produced a wrong answer

## Goal

[0024](../0024-m7-dispatch-cost-anatomy/TASK.md) established that changing
design costs ~25 µs + 7.2 µs per lock and that the encoder pays it on all 49
dispatches. Two levers survive that finding: **batching** (the switch bill is
fixed per encode, so more sequences per encode amortises it) and **fusion**.
Batching needs no new kernels, so it goes first.

It also left `--cols 2` justified as "chosen for headroom", which is a reason
that can never turn out to be wrong. This task replaces it with a threshold that
can.

## Batching

`M = batch * seq`: every GEMM in the encoder is over all tokens of all sequences
at once, so batching is purely a larger M. `tools/export_xclbin.py --batch N`,
and the C++ side reads batch back **from the loaded design's M** rather than
holding a constant, so runtime and xclbin cannot disagree.

The goldens are batch 4. For larger batches those four sequences are **tiled**
to fill the design: the array does the full work, so throughput is honest, and
no accuracy claim is made beyond batch 4 — where validation still runs.

**Batch 16 validates at exactly the same `1-cos` 3.430e-04 as batch 4.** A 4×
larger M changes nothing numerically, which is the expected result and worth
having rather than assuming.

## The width default as a prediction

From `--probe-design`, fitting `t_alone = f + c·M/cols` to the 1- and 2-column
points and then *checking* the other two:

```
  762 = f + 256c/1        451 = f + 256c/2      ->  c = 2.43 us/row, f = 140 us
    4 cols:  140 + 256*2.43/4 = 295 us   (measured 288)
    8 cols:  140 + 256*2.43/8 = 218 us   (measured 221)
```

**f = 140 µs independently reproduces M2's and
[0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s ~144–150 µs fixed dispatch
cost**, from a completely different measurement. Combining it with 0024's
switch model (locks ≈ 48 + 40·(cols−1)) gives crossovers:

| | predicted crossover | batch |
|---|---|---|
| 2 columns overtakes 1 | M ≈ 237 | ~4 |
| **4 columns overtakes 2** | **M ≈ 947** | **~15** |
| 8 columns overtakes 4 | M ≈ 3793 | ~59 |

### Tested

Three repeats per cell, all designs re-exported after the bug below:

| GEMM columns | M = 256 (batch 4) | M = 1024 (batch 16) |
|---|---|---|
| 1 | 51.9 (range 2.3) | 62.9 (1.3) |
| 2 | **53.6** (1.5) | **72.2** (0.4) |
| 4 | 46.1 (0.2) | **72.0** (1.5) |

- **2 vs 1**: predicted to separate above M ≈ 237. At M = 256 they are 1.7 apart
  with overlapping ranges; at M = 1024, 9.3 apart (+15%). Direction and growth
  both as predicted — the gap opens as M leaves the fixed-cost regime.
- **4 vs 2**: predicted crossover M ≈ 947. At M = 256, 4 columns is 14% *worse*;
  at M = 1024 they **tie** (72.0 vs 72.2). Measured crossover is within ~8% of
  the prediction.

So the default is now a claim: `--cols 2` is right for M ≤ ~950, and **4 becomes
correct at batch 16 and above**. Next time the batch changes, the model tests
itself.

**Batching itself is worth 1.35×** at constant width (53.6 → 72.2), and 1.70×
against where [0023](../0023-m7-full-cpp-encode/TASK.md) started at 42.4.

## Do two processes share the array?

[0024](../0024-m7-dispatch-cost-anatomy/TASK.md) asserted that one partition over
eight columns means one live configuration at a time. That was an inference from
a report, not a measurement. Two processes, each with its own context, both
dispatching, is a cheaper test than instrumenting FastFlowLM and cleaner than
threads (two threads share a context unless explicitly given two):

| designs | 1 process | 2 processes | aggregate |
|---|---|---|---|
| 1 column | 51.9 | 38.2 + 38.4 | **1.48×** |
| 2 columns | 52.8 | 39.0 + 39.2 | **1.48×** |
| 4 columns | 46.3 | 33.1 + 33.5 | **1.44×** |

Neither 1.0× nor 2.0×, and — decisively — **independent of design width**.
Spatial partitioning requires narrow designs to scale better; they do not. So
the 1.46× is host-side overlap, not array concurrency: with dispatch+wait at 72%
of wall, full serialisation on the array predicts `1/0.72 = 1.39×`, and 1.44–1.48
is that plus the submit and sync that also overlap.

**Serialisation stands, now with evidence.** It also answers the FastFlowLM
question by proxy: their six active contexts are almost certainly serialised
through the same array too.

Still untested: whether the partition width is settable at context creation. The
XRT C++ header was the wrong place to look — `cfg_param_type` is a `map<string,
uint32_t>` with no relevant key, but column count is decided in the driver
create-hwctx call, not at XRT level. On Linux that is
`amdxdna_drm_create_hwctx` in the `amd/xdna-driver` UAPI header, which carries a
tile count the driver converts to columns. **No such header exists locally** —
the Windows driver is closed and its XRT shim takes another path — but that
public header is the best available documentation of the same firmware
interface, and reading it is still cheaper than either alternative.

## The third fail-open, and it produced a wrong answer

Re-exporting batch 4 **after** batch 16 existed silently produced **`1-cos`
8.651e-02** instead of 3.430e-04.

`find_cache_by_markers` matched the kernel symbol and the column count. Neither
distinguishes a batch-4 GELU from a batch-16 one — both are `gelu_poly_bf16` on
one column — and the mtime tie-break handed back the batch-16 xclbin. The
runtime's buffer-size assertion did not catch it **because this script writes
the sizes into `design.json` from the request rather than reading them from the
artifact**, so the design file confidently described a design it did not
contain.

Third instance of this class in one file ([0022](../0022-m7-cpp-runtime/TASK.md),
[0024](../0024-m7-dispatch-cost-anatomy/TASK.md), here), and the third time it
failed open. Fixed by pinning the buffer memref size as a marker, the same way
the GEMM path already pins M·K, K·N and M·N. Match counts went 3 → 1.

### And the one it was hunting: the layout hash was never checked

The `.npue` stamps a `layout_hash` into every tiled tensor precisely so a wrong
layout cannot be mistaken for a right one. `tools/npue.py` compares it and
refuses. **The C++ runtime parsed it and never looked at it** — and the header
even documented empty as "not tiled", making absence indistinguishable from
"fine".

That is exactly the failure [0022](../0022-m7-cpp-runtime/TASK.md) shipped:
pre-tiled weights into a row-major design, right sizes, wrong order, `rel_fro`
1.186. The lesson recorded then was *"a buffer-size check catches a wrong size,
never a wrong layout"* — and the mechanism that catches a wrong layout had been
in the format since M4, unread.

Now `export_xclbin.py` writes `b_layout_hash` into `design.json` and
`Encoder::stage_all()` refuses to stage a weight unless **both sides stated a
hash and the hashes agree**. An empty hash on either side is a failure, not a
pass.

Enabling it immediately reported a mismatch on a *correct* file — because
`export_xclbin.py` built the layout dict by hand and omitted `"dtype": "BF16"`,
which the packer includes. The packer had the same literal duplicated twice
internally. All three now call `npue.gemm_b_layout()`; the duplication was the
bug, not the hash.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `1-cos 8.651e-02` after a re-export that changed nothing | batch-16 eltwise xclbin selected for a batch-4 export; symbol + width matched, size did not | pin the buffer memref as a marker |
| Layout check failed on a known-good file | `dtype` missing from a hand-built layout dict | one canonical `gemm_b_layout()` |
| Older artifact sets stopped loading | they predate `b_layout_hash`; the new check refuses to guess | re-export (this is the check working) |

## Artifacts

- `tools/export_xclbin.py` — `--batch`, size-pinned matching, `b_layout_hash`
- `tools/npue.py` — `gemm_b_layout()`, the canonical descriptor
- `runtime/src/main.cpp` — runtime batch from the design's M, tiled inputs,
  layout-hash enforcement
- `runtime/artifacts{1,2,,_b16,_b16c1,_b16c4}/` — the sweep

## Next

1. **Rebuild at `--cols 4` for batch ≥ 16** — the model says it wins there and
   the measurement agrees; batch 32+ should separate them properly.
2. **Fusion** is the only remaining way to remove switches rather than amortise
   them.
3. **Host attention is still ~20%** and untouched.
4. **Read the `amdxdna_drm_create_hwctx` UAPI header** to settle whether
   partition width is a knob.

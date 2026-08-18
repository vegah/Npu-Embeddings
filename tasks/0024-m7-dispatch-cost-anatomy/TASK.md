# 0024 — M7: what the 1.5 ms per dispatch actually is

- **Date** 2026-08-17
- **Milestone** M7
- **Status** done — **context switching identified and priced; 42.4 → 52.4 seq/s**

## Goal

[0023](../0023-m7-full-cpp-encode/TASK.md) measured 49 dispatches at 1.50 ms
each against 150 µs of hardware and concluded that "~90% of the NPU path is
still host-side bf16 conversion and memcpy". This task set out to remove that
host cost.

**That conclusion was wrong**, and the first fix is what proved it.

## What was fixed, and what it bought

Three host-side costs, priced before writing any code:

| | per encode |
|---|---|
| Weights memcpy'd from the mapped `.npue` into a BO on **every** dispatch | 21.2 MB that never changes |
| bf16 conversion, scalar | 13.76 M elements |
| GEMM output: memcpy, then a second pass for bias | 21.2 MB × 2 passes |

All three were removed — `Design::stage()`/`bind()` keeps all 24 weight sets and
13 LayerNorm parameter sets device-resident, the conversions became AVX2, and
the bias add now reads the result buffer in place.

**Result: 42.4 → 46.3 seq/s. Nine percent.**

The end-to-end result did not move because the host cost was never the problem.
It is worth keeping anyway — it is now 2.5 ms of a 76 ms encode — but the
interesting part is what it revealed.

Correctness is unchanged at every step: **`1-cos` 3.430e-04**, the same digits as
[0023](../0023-m7-full-cpp-encode/TASK.md). The AVX2 conversion is bit-identical
to the scalar one by construction (same integer ops, and after `>> 16` the
values cannot saturate `packus`), and the end-to-end number confirms it.

## Where the time really goes

Splitting `t_npu` five ways, then splitting the dispatch itself:

```
      bf16 convert (both ways)        1.17 ms    1.5%
      sync to device                  0.43 ms    0.6%
      dispatch + wait                55.08 ms   72.2%     1124 us each
        submit (build + start)        1.07 ms    1.4%       22 us each
        wait (hardware)              53.90 ms   70.6%     1100 us each
      sync from device                0.30 ms    0.4%
      read out + bias                 0.66 ms    0.9%
```

Building and submitting the command costs 22 µs. Everything else is the wait.

But the wait does not scale with work: `attn_out` does a quarter of `ffn_up`'s
MACs and takes 0.9× as long, and **GELU — pure elementwise, no multiply at all —
was the slowest op in the model at 2184 µs**. Something other than arithmetic
was being measured.

## The probe

`--probe` dispatches with no host work in the loop, first repeating one design,
then alternating two:

```
    same design repeatedly:            alternating:
      qkv               280 us           qkv <-> ffn_up          1519 us
      attn_out          165 us           qkv <-> gelu            1980 us
      ffn_up            340 us           layernorm <-> softmax    721 us
      ffn_down          343 us
```

Repeating one design, the numbers scale with work exactly as expected and agree
with [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s cost model. **The
~1200 µs is the cost of changing design.** The encoder changes design on every
one of its 49 dispatches, so it pays it 49 times.

`--probe-pair` loads only two designs instead of seven: **1509 µs**, against
1519 with seven resident. Resource pressure from seven contexts is not the
cause, so trimming the resident set cannot help.

## Reconfiguration, or context switch?

Those are different problems with different fixes, and the aggregate cannot
tell them apart. `--probe-ctx` loads the **same xclbin into two contexts** and
alternates — identical configuration data, two contexts:

| | 2 cols | 4 cols | 8 cols |
|---|---|---|---|
| `qkv` alone, one context | 451 µs | 288 µs | 221 µs |
| **`qkv` ↔ `qkv′`, same xclbin, two contexts** | **1061 µs** | **1478 µs** | **2569 µs** |
| `qkv` ↔ `ffn_up`, two contexts | 1122 µs | 1527 µs | 2612 µs |

**Identical configuration costs the same as different configuration.** It is not
that new data must be loaded; it is the hardware context switch itself.

Subtracting the same-context cost isolates the switch, and it is close to
linear in the columns the context occupies (1, 2, 4, 8 → 360, 610, 1190,
2348 µs):

> **switch ≈ 55 µs + 286 µs per column** — within 5% at all four widths.

That is **10–17× [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)'s ~150 µs
per dispatch**, and it makes the context switch the most expensive single
operation anywhere in this stack — more expensive than the arithmetic it
surrounds.

### …and that per-column model is wrong

Two results were left in tension: A→A′ says the cost does not depend on how
*different* the configurations are, while 286 µs/column says it clearly depends
on how *much* configuration there is. Both can hold at once, and then the
diagnosis is sharper than "it is not configuration data": **the driver tears
down and rebuilds per-column state unconditionally, without exploiting that it
is identical.** That is a missed fast path, not inevitable work.

286 µs per column is also far too much to be bulk DMA of state — a column's ELFs
are tens of KB — and entirely plausible as a few thousand individual MMIO writes
to BD registers, stream switches and locks. That predicts the cost tracks the
**number of configured objects**, not the column count as such.

`experiments/m7-switch-cost/` tests it with a control: a minimal 1-column design
(one worker, two ObjectFifos, a vector copy) against a 1-column GEMM at
identical width. A per-column model requires them to cost the same.

| design | cols | `aie.dma_bd` | `aie.lock` | switch |
|---|---|---|---|---|
| passthrough | 1 | 6 | 8 | **89 µs** |
| gelu | 1 | 24 | 32 | **298 µs** |
| qkv | 1 | 63 | 48 | **365 µs** |
| qkv | 2 | 110 | 88 | **628 µs** |
| qkv | 4 | 204 | 168 | **1190 µs** |
| qkv | 8 | 388 | 320 | **2348 µs** |

**4.1× spread at one column — the per-column model is refuted.** What survives,
across a 65× range of descriptor counts and an 8× range of widths:

> **switch ≈ 25 µs + 7.2 µs per lock**
> (equivalently ≈ 49 µs + 5.8 µs per DMA buffer descriptor) — within ~15%
> at every point.

Columns only ever correlated because more columns means more descriptors. And
this is a **design lever**: simpler dataflow buys cheaper switches, which is a
connection none of the papers in `research/papers/` describe.

### Not eviction either

`xrt-smi examine --report aie-partitions`, sampled twice during an alternating
run, shows all seven `npuembed.exe` contexts simultaneously **`Active`** with
**`Suspensions = 0` and `Migrations = 0`** throughout, while their submission
counters climb (180 → 388 each; 389 → 840 for LayerNorm, which is dispatched 13
times per encode against 6 for the rest). Contexts are never evicted, and
switching between co-resident contexts still costs the full amount. That also
explains the `--probe-pair` result directly: 2 resident and 7 resident are the
same because residency was never the constraint.

The report shows **one partition spanning all eight columns**:

```
  Partition Index   : 0
    Columns: [0, 1, 2, 3, 4, 5, 6, 7]
```

If that is the granularity, only one design's configuration can be live on the
array at a time regardless of how narrow the designs are — which is why two
1-column designs occupying 2 of 8 columns still pay to swap.

It also shows `WorkloadsSessionHost.exe` holding three resident contexts with
non-zero suspension counts (308, 3, 32) during our runs. **CLAUDE.md rule 1 is
not hypothetical**: the NPU is shared, right now, on this machine.

Written up as [note 0004](../../research/notes/0004-context-switch-cost.md).

## The consequence: narrower is faster

Wider designs compute faster and switch slower. While switching dominates,
switching wins:

| GEMM columns | seq/s | `qkv` alone | switch |
|---|---|---|---|
| 1 | 51.4 | 762 µs | 360 µs |
| **2** | **53.0** | 451 µs | 610 µs |
| 4 | 46.0 | 288 µs | 1190 µs |
| 8 | 35.4 | 221 µs | 2348 µs |

**Eight columns is 33% slower end to end than two**, even though every kernel in
it is 1.4–1.9× faster in isolation. The curve has a real minimum at 2: below it
the compute cost takes over, above it the switch cost does.

The export default is now `--cols 2`, and the comment there says it is only
correct while switches dominate.

`--elt-cols` also became a parameter. GELU, LayerNorm and softmax had `n_cols=1`
hard-coded since M5 — written to prove the kernels correct, and correctness does
not need width — which is why GELU was the slowest op in the model. At 8 columns
GELU does not compile (`no ShimNOCTile has sufficient DMA capacity`); 2 works.

## Result

```
    wall    76.32 ms   ->      52.4 seq/s
    cpu     25.00 ms   ->      0.33 cores busy
```

**42.4 → 52.4 seq/s**, correctness unchanged at `1-cos` 3.430e-04.

## A bug found on the way: mtime is not identity

Asking for a 1-column GELU produced the **2-column xclbin**, silently.

`find_cache_by_markers` matched the kernel symbol and broke ties by mtime. A
JIT **cache hit does not restamp the directory**, so asking for a width that was
already built left it looking older than the width built moments ago, and the
newer one won. This is [0022](../0022-m7-cpp-runtime/TASK.md)'s "all four
designs got the same xclbin" in a new guise, and CLAUDE.md's stale-cache trap.

The fix is to match on width the same way the GEMM path already matched on
shape — from the artifact's contents. One trap inside the fix: **`aie.mlir` has
no tile coordinates at all**. It is pre-placement and contains only
`aie.logical_tile<CoreTile>(?, ?)`, so counting columns there returns 0 for
every design — a check that fails open. The placed form is
`input_with_addresses.mlir`, the same file the trace flow has to use, for the
same reason.

The GEMM path had the identical latent hole — a 4-column and an 8-column `qkv`
declare the same memrefs and the same strides — and stayed correct only because
each width happened to be built fresh. Both now check the placed MLIR; matches
went from "3 cache dirs" to "1 cache dir".

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `no ShimNOCTile has sufficient DMA capacity for 0 input/1 output channels` | GELU at 8 and at 4 columns needs more shim DMA channels than exist | 2 columns for the eltwise designs |
| `ValueError: 'runtime\artifacts8\manifest.json' is not in the subpath of ...` | `relative_to` on a relative `--out`, in the final `print` — after every artifact was already written | `resolve()` then `relative_to` in a `try` |
| Requested 1-column GELU, got the 2-column one | mtime tie-break against a JIT cache hit | match the placed MLIR's column count (above) |

## Artifacts

- `runtime/src/main.cpp` — `--probe`, `--probe-pair`, `--probe-ctx`,
  `--artifacts`, the five-way `t_npu` split, AVX2 conversion
- `runtime/src/npu_device.cpp` — `stage()`/`bind()`, submit/wait timers
- `tools/export_xclbin.py` — `--elt-cols`, width-aware cache matching
- `runtime/artifacts{1,2,,8}/` — the four builds compared

## Next

The width knob is now at its optimum and is worth ~1.14×. Everything past that
needs the **number of switches** to come down, because that is the variable the
cost is proportional to:

1. **Batch.** 49 switches per encode regardless of how many sequences are in it.
   The switch cost is fixed per encode, so it amortises directly. Needs designs
   rebuilt at larger M; no new kernels.
2. **Fuse.** Several kernels in one xclbin is the only way to make a switch not
   happen at all. This is what "one dispatch per encoder layer" (F1) means
   concretely, and the price of not doing it is now measured rather than assumed.
3. **Then re-measure width.** `--cols 2` is a consequence of switching
   dominating, not a property of the hardware. Once switches are rare, the
   8-column designs — which really are 1.4–1.9× faster — should win.
4. **Host attention is 21%** and unchanged by any of this.

Note also that operator-major reordering (all layers' `qkv`, then all `ffn_up`)
does **not** apply here: layer *L*+1 depends on layer *L*, so the 49 dispatches
are a chain, not a set. Batching and fusion are the two levers that survive that.

# 0020 — M5: LayerNorm on the array, the first kernel with a reduction

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **kernel passes at both sites; two hardware limits found**

## Goal

Put LayerNorm on the array. It is the first op that needs a **reduction** —
everything before it was elementwise (GELU) or a GEMM, and both produce output
from local data. LayerNorm must sum a whole row before it can emit a single
value, which is a different shape of problem for a dataflow array.

It is also the last thing, with softmax, keeping a whole encoder layer from
being fusable.

## Why not the shipped kernel

`aie_kernels/aie2p/layer_norm.cc` exists. Unlike the GELU case, no measurement
was needed — three things in its source rule it out:

1. **`const float gamma = 1.0f; const float beta = 0.0f;`** — hardcoded. It
   takes no learned parameters at all, and BERT has 384 of each per LN site.
2. **`constexpr float epsilon = 1e-5f`.** MiniLM uses **1e-12**
   (`docs/04-model`). Four orders of magnitude, inside a square root.
3. **`variance = (sum_sq / cols) - mean*mean`** — the one-pass formula, which
   subtracts two nearly equal large numbers. `docs/04-model` records that
   post-LN BERT carries hidden dims at ±50–100 among values near ±1, which is
   exactly the input that makes it cancel.

## What ours does

`experiments/m5-eltwise/kernels/layernorm.cc`:

- **Two passes.** Sum → mean, then sum of `(x − mean)²` → variance. Costs one
  extra read of a row already resident in L1, and is stable regardless of
  outlier magnitude.
- **Biased variance** (÷N), PyTorch's convention.
- **eps inside the sqrt**, and eps = 1e-12 from the `.npue` config.
- **fp32 throughout** — [0016](../0016-m5-fp32-probe/TASK.md) established that
  is real on this part, not nominal.
- **Learned per-channel gamma/beta**, fp32, read from the `.npue`.

Per-row scalar work is three operations — two multiplies by a precomputed
reciprocal (never a divide) and one `invsqrt` — against 384 elements.
`research/notes/0001` forbids scalar float *in the inner loop*, and this is not
that.

## Two hardware limits, both hit on the way

### A core tile has 2 input and 2 output DMA channels

Passing `input`, `gamma`, `beta` and `output` separately is four streams:

```
error: tile (0, 3) requires 3 input/1 output DMA channels,
       but only 2 input/2 output available
note:  reduce the LTO's DMA fanin (e.g. via memtile staging)
```

Fixed by packing gamma and beta into **one** 768-float buffer, gamma first. Not
a packaging preference — a hardware constraint on how many distinct streams a
kernel can take. **This matters directly for fusion**: a fused encoder layer
wants many inputs, and it gets two.

### The L1 budget bites at 20 rows

The first version processed 64 rows of 384 per call, and produced CLAUDE.md
trap 3:

```
error: 'aie.tile' op Basic sequential allocation failed
```

With double buffering on input and output,
`2·(rows·384·2)·2 + 2·384·4 < 65536`:

| rows/call | L1 | |
|---|---|---|
| 16 | 52,224 B | ok |
| 20 | 64,512 B | ok |
| 24 | 76,800 B | over |
| 64 | 199,680 B | over |

64 rows wanted ~200 KB against a 64 KB L1. Settled on **16**, which leaves
headroom.

## Result

256 rows × 384, 4 cores, 16 rows per call, gamma/beta from the `.npue`:

| site | NPU vs golden | NPU vs CPU model | bf16 floor |
|---|---|---|---|
| `L0.ln1` | **3.326e-03** | 3.659e-03 | 2.058e-03 |
| `L0.ln2` | **3.615e-03** | 3.875e-03 | 2.249e-03 |

**Both PASS** at 5e-03, at roughly 1.6× the bf16 output floor.

**`aie::invsqrt` is fine.** [0015](../0015-m5-gelu-polynomial/TASK.md) warned it
should be assumed as coarse as `aie::tanh` (~1%) until measured. If it were,
LayerNorm — which multiplies by it — would land near 1e-02. It lands at
3.3e-03 against a 2.1e-03 floor, so invsqrt contributes little. **That warning
is now discharged**, and it matters because invsqrt is the only transcendental
LayerNorm needs.

## An inconclusive diagnostic, reported as inconclusive

Two independent kernels now diverge from their own numpy models by almost the
same amount — GELU **3.886e-03** ([0015](../0015-m5-gelu-polynomial/TASK.md)),
LayerNorm **3.659e-03** here. A shared cause is likelier than a coincidence, and
the shared step is the fp32 → bf16 store. Round-to-nearest-even and truncation
are distinguishable, so the runner asks:

```
  on the 49.6% of values where the two differ:
    matches round-to-nearest-even :  46.8%
    matches truncation            :  53.2%
    mean |NPU| - |exact|          : -6.234e-04
```

**53/47 is not a signal.** The toward-zero bias is real and consistent across
both sites, but if the store simply truncated, the match rate would be ~100%.
So the store rounding is not the explanation, or not the whole one.

For LayerNorm a better candidate is **summation order**: the mean and variance
sum 384 values whose dynamic range spans two orders of magnitude, the NPU
accumulates in 16 lanes and then reduces, and numpy sums pairwise. Different
orders give different answers on that data. But that cannot explain GELU, which
has no reduction at all.

Left open rather than resolved into a story that fits one kernel and not the
other.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| `requires 3 input/1 output DMA channels, but only 2 input/2 output available` | A core tile has 2 in / 2 out DMA channels; four separate streams do not fit | Pack gamma+beta into one buffer |
| `'aie.tile' op Basic sequential allocation failed` | 64 rows/call needs ~200 KB of a 64 KB L1 | 16 rows/call; the budget formula is in the kernel header |
| `no member named 'to_vector' in aie::vector<float,16>` — **again** | Narrowing fp32→bf16 is an accumulator method, not a vector one. Same trap as [0014](../0014-m5-own-gelu-kernel/TASK.md) | `aie::mul(v, 1.0f).to_vector<bfloat16>()`. Second time; now noted in the kernel itself |

## Artifacts

- `experiments/m5-eltwise/kernels/layernorm.cc`
- `experiments/m5-eltwise/layernorm_kernel.py`
- `artifacts/layernorm_L0_ln1.json`, `artifacts/layernorm_L0_ln2.json`

## Next

**Softmax is the last op before a layer can be fused.** It needs a row max and a
row sum — two reductions — plus `exp`. `aie::exp2` remains unmeasured and
should be assumed coarse until it is not, exactly as `invsqrt` was until today.

After that, the fusion work from [0010](../0010-m5-b-reuse-and-cost-model/TASK.md)
becomes possible, and the 2-input/2-output DMA limit found here is the first
thing it will run into.

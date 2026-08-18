# 0013 — M5: first elementwise op on the array, and IRON's GELU is not accurate enough

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **op runs; built-in kernel rejected on accuracy**

## Goal

Put the first **non-GEMM** op on the array. Everything through
[0012](../0012-m5-all-layer-gemms/TASK.md) has been matmul; a fused encoder
layer also needs GELU, LayerNorm, softmax and bias adds, and none of those had
touched a core.

GELU is the smallest one that matters, and it carried a question that could not
be settled on paper.

## What was in doubt

IRON ships `kernels.gelu()`, and two things about it disagree with
`docs/04-model`:

1. **It is the tanh approximation, not exact erf.** The doc calls tanh a
   landmine: *"~1e-3 systematically-biased error that shows up as an
   unexplainable relative-Frobenius floor"*.
2. **It is LUT-backed in bf16**, and IRON's own docstring suggests verifying it
   with `rtol=0.128` — a **12.8%** relative tolerance.

Also worth recording for later: `kernels.softmax()` computes softmax
independently **per 1024-element tile with no cross-tile reduction**. Attention
needs softmax per row over the sequence, so it is structurally unusable for us
unless `seq == 1024`. And there is **no LayerNorm kernel at all**. All the LUT
kernels are fixed at `tile_size = 1024`.

### The formula question, answered on CPU first

Before spending an NPU run, the tanh-vs-erf cost was measured on the real
`L0.ffn_up` activations:

| variant | rel_fro vs exact-erf golden |
|---|---|
| exact erf, fp32 | 4.9e-18 (it *is* the golden) |
| **tanh approximation, fp32** | **5.621e-04** |
| exact erf, **bf16 output** | 1.689e-03 |
| tanh approximation, **bf16 output** | 1.779e-03 |

The tanh error is real and genuinely biased — mean signed error −2.860e-05,
where a symmetric error would be ~0. **But it sits below the bf16 quantisation
floor.** On a bf16 datapath tanh costs about 5% extra, not a doubling.

So the doc's rule is right for an fp32 reference and **too strict for our
datapath**. That is a useful narrowing: the formula was never the problem.

## What was done

`experiments/m5-eltwise/gelu_kernel.py` — a plain elementwise design: one
in/out ObjectFifo pair per core, `kernels.gelu(1024)` in the core loop,
contiguous equal slices per core. No mem-tile staging, because an elementwise op
has no reuse to exploit and no reduction to share — unlike the GEMM, where B is
broadcast down a column.

Fed the genuine `L0.ffn_up` activations from the goldens (393,216 elements =
exactly 384 tiles of 1024) and compared three ways, so a failure would say
*which* layer of approximation caused it.

## Commands

```powershell
cd C:\dev\mlir-aie; . .\iron_env.ps1; cd C:\Users\vegar\Documents\GitHub\NpuEmbeddings
python experiments\m5-eltwise\gelu_kernel.py
```

## Result

```
GELU on the array: 393,216 elements = 384 tiles of 1024, 1 cols x 4 rows
  input  = golden L0.ffn_up (real pre-activations)
  target = golden L0.gelu   (exact erf, fp32)

  comparison                                       rel_fro
  NPU vs exact-erf golden (what we need)         1.332e-02
  NPU vs fp32 tanh-GELU (is the LUT faithful?)   1.318e-02
  fp32 tanh-GELU vs golden (formula cost alone)   1.975e-03

  max abs error            6.162e-02
  max pointwise rel error  1.000e+00

FAIL -- tolerance 5e-03
```

**The op runs correctly as a program.** The design compiles, dispatches, and
returns plausible GELU values. That is the first elementwise kernel on the array
and it works.

**The accuracy is not usable.** And the three-way split says exactly why:

- The **formula** costs 1.975e-03.
- The **kernel** is at 1.318e-02 *from the formula it claims to implement* —
  6.7× worse than the approximation itself, and ~8× worse than bf16 rounding
  alone (1.689e-03).
- Max pointwise relative error is **1.000e+00**: at least one element is 100%
  wrong.

So the LUT, not the tanh substitution, is what disqualifies it. Against our
GEMMs at 1.4–2.4e-03, dropping this in would make a single elementwise op the
largest single error source in the layer.

**Decision: write our own GELU kernel.** Which is what this project set out to
do — *"we build everything here"*, `docs/00-overview.md` ground rule 2.

The path is confirmed to exist: `ExternalFunction(name, source_file=...,
arg_types=[...], include_dirs=[...])` compiles our own `.cc` through Peano, and
is exactly how IRON builds its own kernels.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| Built-in GELU fails at 1.33e-02 where the format floor is 1.69e-03 | LUT approximation quality, not the tanh formula | Write our own. The three-way comparison is what made the attribution possible — comparing only against the golden would have wrongly blamed tanh |
| `kernels.softmax()` cannot express attention softmax | It reduces per 1024-element tile, with no cross-tile reduction | Not solved. Recorded now so it does not surprise us during attention |

## Artifacts

- `experiments/m5-eltwise/gelu_kernel.py`
- `experiments/m5-eltwise/artifacts/gelu_kernel.json`

## Next

1. **Write `gelu.cc`** — vectorised, exact erf or a high-accuracy polynomial,
   bf16 in/out. `research/notes/0001` is the binding constraint here: **never
   scalar float in a kernel body**, measured at 1,617× slower. Validate against
   `L0.gelu` with the same three-way split, target ≤ 2e-03.
2. **LayerNorm has no built-in at all** and needs a reduction plus an inverse
   square root, in fp32. It is the harder kernel and it is required by
   `docs/04-model` to stay fp32.
3. **Softmax needs a row-wise formulation** over the sequence, not per-1024-tile.
4. Only then is fusing a whole layer into one dispatch a realistic target — and
   the 150 µs from [0010](../0010-m5-b-reuse-and-cost-model/TASK.md) is why it
   matters.

Worth noting the shape of the remaining work changed today: the GEMMs were
assembly of existing parts, and everything left is kernels we have to write.

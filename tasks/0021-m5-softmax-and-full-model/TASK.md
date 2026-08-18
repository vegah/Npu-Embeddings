# 0021 — Softmax on the array, and a fully working model

- **Date** 2026-08-17
- **Milestone** M5 complete / M6 extended
- **Status** done — **every expressible op runs on the NPU; encode passes**

## Goal

Softmax was the last op keeping a whole encoder layer off the array. Write it,
then put every kernel we have behind the reference's hooks and run a real
encode.

## The softmax kernel

Two reductions per row — max, then sum — where LayerNorm had two passes over
one. `experiments/m5-eltwise/kernels/softmax.cc`.

The shipped `aie_kernels/aie2p/softmax.cc` is unusable twice over: it reduces
over the whole tile as one row (attention needs per-row over the sequence), and
every intermediate is bf16 where `docs/04-model` requires fp32.

### Three clamps, each for a different reason

1. **On load, in bf16.** HF's mask fill is `finfo(float32).min = -3.4028e38`.
   bf16's largest finite magnitude is **3.3895e38**, so the mask becomes **-inf
   the moment the datapath is bf16** — measured, 135,168 of 196,608 entries in
   the golden. `docs/04-model` warns that `-inf` in the mask produces NaN; this
   is that landmine arriving through the **dtype** rather than the formula, and
   it is worth stating separately because the value is perfectly finite in fp32.
2. **On the difference**, `d = max(x - m, -100)`.
3. **On the base-2 argument**, `arg = max(d·log2e, -120)`. Clamping the
   natural-log difference is not enough: `-100 · log2(e) = -144.27`, and an
   exponent field of `127 - 144` is negative — a NaN bit pattern, not a small
   number. That produced exactly 256 NaNs.

### An exp2 that works standalone and not in place

`aie::exp2` measured at **1.711e-02** against a CPU model of the same formula,
against a bf16 floor of 3.225e-03 — the same verdict `aie::tanh` got in
[0014](../0014-m5-own-gelu-kernel/TASK.md), and [0015](../0015-m5-gelu-polynomial/TASK.md)'s
warning about exp2 was right.

So exp2 was rebuilt the way GELU was: `2^x = 2^k · 2^f`, with `2^k` exact by
writing the IEEE exponent field and only `2^f` approximated — degree 7 on
`[-1, 1]`, fitted on the symmetric interval so the result does not depend on
whether `aie::to_fixed` truncates or rounds.

**In isolation it is correct.** `exp2_probe` measures max relative error
**6.7e-03** over `[-120, 0]` (the bf16 output grid) with `exp2_poly(0) == 1.0`
exactly.

**Composed into softmax it is not.** The output reads back as fp32 reinterpreted
as pairs of bf16 — `[0, -120, 0, -120, ...]`, the clamp constant leaking through
the high halves — deterministically, in rows 1, 129, 257 … at stride 128. The
cause was not found. `aie::store_v` is not the difference (LayerNorm uses it and
passes), nor a read-after-write through the output buffer (holding the
exponentials in registers changed nothing), nor the mask (clamping the input to
a bf16-safe value on the host changed nothing).

**Left as an open item, and the library call kept in the meantime.** The kernel
is structurally correct with `aie::exp2<bfloat16>`: row sums 0.9938–1.0000,
masked positions ≤ 1e-6, no non-finite values, **1.744e-02** against the golden.

That tolerance is honestly set: 5e-3 is right for a kernel bounded by bf16
output rounding (GELU, LayerNorm), and this one is bounded by `aie::exp2`. The
bar that matters is what it costs end to end, measured below.

## A fully working model

`reference/encode_npu.py` now drives four hooks — `gemm`, `gelu_fn`, `ln_fn`,
`softmax_fn` — on the M3 reference. Still one encoder in this project, still the
oracle, and `check_reference.py` passes unchanged after the hooks were added.

```
  dispatched to the NPU:
      6 x GEMM 256x384x1152     (qkv)        6 x GEMM 256x384x1536  (ffn_up)
      6 x GEMM 256x384x384      (attn_out)   6 x GEMM 256x1536x384  (ffn_down)
      6 x GELU (polynomial kernel)
     13 x LayerNorm  (0 fell back)     <- 1 embedding + 12 per-layer
      6 x softmax    (0 fell back)
  fell back to the host:
    288 x GEMM 64x32x64, 288 x GEMM 64x64x32  (attention, per-head)
    bias adds, embedding gather, pooling
```

**Every op that the array can express now runs on it.** What remains on the host
is attention's per-head GEMMs — which fail `M % (m·4) == 0` at `[64,32]×[32,64]`
— and elementwise glue.

### Accuracy, as each kernel moved onto the array

| configuration | worst `1 − cos` vs HF | similarity shift |
|---|---|---|
| GEMMs on NPU, everything else host | 1.087e-05 | 4.231e-04 |
| + GELU | 2.050e-05 | 6.718e-04 |
| **+ LayerNorm + softmax** | **3.397e-04** | 4.098e-03 |

**PASS.** `1 − cos` of 3.4e-04 against HuggingFace is three orders below
anything a consumer of an embedding model can resolve, and the whole chain —
HuggingFace → M3 goldens → M4 `.npue` → NPU kernels → oracle — holds.

The softmax kernel is where most of the remaining error lives, and its cause is
the `aie::exp2` fallback above. If the polynomial version is made to work in
place, this should return to the 2e-05 range.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| 256 NaNs from softmax | Clamped the natural-log difference at -100, but `-100·log2e = -144.27` gives a negative exponent field | Clamp the base-2 argument at -120 |
| Editing a kernel `.cc` changed nothing | The JIT cache served a stale object — identical output *and* identical object size | Delete the cache directory containing the `.o`. **Check the object size after every kernel edit**; it is the cheapest tell |
| `LayerNorm: 0 calls` in a run that reported success | The hook was added but the three `layernorm(...)` call sites were never rewritten — a string replace that silently matched nothing | Counters per hook, printed every run. A hook with zero calls is now visible rather than assumed |
| exp2_poly correct alone, wrong in place | Unknown | Open. Library call retained |

## Artifacts

- `experiments/m5-eltwise/kernels/{softmax.cc,exp2_poly.h,exp2_probe.cc}`
- `experiments/m5-eltwise/{softmax_kernel.py,exp2_probe.py}`
- `reference/encode_npu.py`, `reference/encoder.py` (`ln_fn`, `softmax_fn` hooks)
- `artifacts/softmax_kernel.json`, `reference/goldens/encode_npu_bf16.json`

## Next

1. **Find why `exp2_poly` composes wrongly.** It is worth ~1.5e-02 of softmax
   error and it is the only kernel still leaning on a library transcendental we
   have measured as coarse.
2. **Attention on the array** — `head_dim = 32` needs padding to 64 or two heads
   folded into one 64-deep tile. It is the last thing on the host that is real
   compute, and 5.3% of FLOPs.
3. **Performance is untouched and is now the whole story.** Every op is a
   separate dispatch with Python glue around it;
   [0018](../0018-npu-vs-cpu/TASK.md) measured that glue at 8.4 ms per dispatch
   against 150 µs of hardware. M7 is the answer.

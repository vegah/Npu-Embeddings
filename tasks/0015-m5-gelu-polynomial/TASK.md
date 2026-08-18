# 0015 — M5: GELU without a transcendental, and AIE float is not fp32

- **Date** 2026-08-17
- **Milestone** M5
- **Status** done — **usable GELU kernel; a bigger finding underneath it**

## Goal

[0014](../0014-m5-own-gelu-kernel/TASK.md) pinned the GELU error on `aie::tanh`
itself (~1% accurate), which no caller-side precision can fix. Evaluate GELU
with **no transcendental call at all**.

## The rewrite that makes it easy

Fitting GELU or tanh directly is awkward: both saturate, so a polynomial needs
high degree and misbehaves outside its range. But

```
GELU(-x) = -x*Phi(-x) = -x*(1 - Phi(x)) = GELU(x) - x
```

so `c(x) = GELU(x) - max(x,0)` satisfies `c(-x) = c(x)` — **c is even**. It is
also a bump that decays like a Gaussian: `c(0)=0`, `|c|` peaks near 0.17 around
`|x|=0.75`, `c(4) = -1.3e-04`, `c(5) = -2.9e-06`. So

```
GELU(x) = max(x, 0) + c(min(|x|, R))
```

and **clamping costs nothing** because c is already ~0 at the edge — no branch,
no select, no masking. Every operation is a native vector op: `abs`, `min`,
`max`, `mul`, `add`.

Designed on CPU first (`design_gelu_poly.py`), fitting c at Chebyshev nodes and
picking the **cheapest** fit within 5% of the best, since each extra degree is
another mul+add in the inner loop:

| fit | rel_fro on real activations |
|---|---|
| deg 6, clamp 4 | 4.571e-03 |
| **deg 8, clamp 4** | **2.494e-03** |
| deg 12, clamp 5 | 2.465e-03 |
| — bf16 output floor | 2.465e-03 |

Degree 8 is at 1.01× the floor. Degree 12 buys 1%.

## Result

`experiments/m5-eltwise/kernels/gelu_poly.cc`, on hardware, against the real
`L0.ffn_up` activations and the exact-erf golden:

| comparison | rel_fro |
|---|---|
| **NPU vs exact-erf golden** | **4.312e-03** |
| NPU vs CPU model of the same polynomial | 3.886e-03 |
| CPU model vs golden (the design limit) | 1.923e-03 |
| `aie::tanh` kernels, for reference | 1.316e-02 |

**PASS**, and **3.1× better than `aie::tanh`.** Object is 3408 bytes — checked,
because [0014](../0014-m5-own-gelu-kernel/TASK.md) established that a silently
empty function is a live failure mode on this toolchain.

## The finding underneath

Read the middle two rows. The **design** is at 1.923e-03, essentially the bf16
floor. But the hardware diverges from a numpy fp32 model of *the same
polynomial* by **3.886e-03**.

IEEE fp32 over ~17 operations would accumulate ~1e-06. We measure 3.9e-03,
which is 2⁻⁸ — **bf16 territory**.

**Inference: `aie::vector<float>` arithmetic on AIE2P is not IEEE fp32.** The
native multiplier is bf16, and asking for float appears to buy range rather than
mantissa. Stated as an inference, not a proven mechanism — but it is consistent
with, and retroactively explains, [0014](../0014-m5-own-gelu-kernel/TASK.md):
rewriting the entire polynomial from bf16 to fp32 there improved the result by
**1%**, which was baffling at the time. The fp32 was never real.

If it holds, it has consequences well beyond GELU:

- **`docs/04-model` requires LayerNorm in fp32** ("bf16 mean/variance over 384
  elements with a ±100 outlier loses everything"). If float vector ops are
  effectively bf16, that requirement cannot be met with straightforward vector
  arithmetic on this part.
- The same question applies to softmax, which the same doc requires in fp32.

This wants a dedicated, minimal experiment — a single long multiply chain with a
known answer — rather than more inference from application kernels.

## Problems hit

| Symptom | Cause | Fix |
|---|---|---|
| Hardware 1.7× worse than the CPU design predicted | Hardware float arithmetic is not fp32 | Not fixed. Isolated by comparing hardware against a CPU model of the *same* polynomial, which separated design error from arithmetic error |
| `max pointwise rel error 1.1e+11` | The clamp leaves `c(4) = -1.3e-04` where true GELU is ~0 (very negative x). Tiny absolute, unbounded relative | Not a defect. Pointwise relative error is the wrong metric where the target is zero |

## Artifacts

- `experiments/m5-eltwise/design_gelu_poly.py`, `artifacts/gelu_poly.json`
- `experiments/m5-eltwise/kernels/gelu_poly.cc`
- `artifacts/gelu_kernel_poly.json`

## Next

GELU is good enough to use: 4.3e-03 against GEMMs at 1.4–2.4e-03.

The open question is now bigger than any one kernel: **measure whether AIE float
vector arithmetic is fp32 or bf16**, because LayerNorm and softmax both depend
on the answer and `docs/04-model` assumes fp32 is available.

---

## Correction — "AIE float is not fp32" is refuted

Added 2026-08-17, same day, by direct measurement
([`tasks/0016`](../0016-m5-fp32-probe/TASK.md)).

**The inference above is wrong.** `aie::vector<float>` arithmetic on AIE2P
carries a **full 24-bit mantissa** on both the add and the multiply path.
Measured with `(1.0f + eps) - 1.0f` and `((1.0f + eps) * 1.0f) - 1.0f` for
`eps = 2^-1 … 2^-24`: eps survives to **2^-23** in both cases, exactly as IEEE
fp32 requires, and dies at 2^-24.

So the consequences drawn above do **not** follow. In particular **fp32
LayerNorm and fp32 softmax remain achievable** on this part, and
`docs/04-model`'s requirement stands unchallenged.

**What is still unexplained:** the GELU kernel diverges from a numpy fp32 model
of the same polynomial by 3.886e-03, and it is now known *not* to be the
arithmetic precision. Remaining suspects, none tested: `aie::abs` / `aie::min` /
`aie::max` on float, or the fp32 → bf16 conversion's rounding mode. Recorded as
open rather than guessed at again.

The kernel result itself is unaffected: 4.312e-03, PASS, 3.1× better than
`aie::tanh`.

**The lesson is the same one as [0009](../0009-m5-sync-misdiagnosis/TASK.md).**
A plausible mechanism that explained two independent observations — the 1%
improvement in [0014](../0014-m5-own-gelu-kernel/TASK.md) and the 2× gap here —
was still wrong. Both times the fix was a *direct* measurement of the suspected
component rather than more inference from application-level results, and both
times it took one short experiment.
